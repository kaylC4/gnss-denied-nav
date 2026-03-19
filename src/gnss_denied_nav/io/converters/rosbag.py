"""
RosbagConverter — converte un ROS bag nel formato flat (Parquet + immagini).

Dipendenze richieste (NON installate di default)
------------------------------------------------
    pip install rosbags          # parser puro-Python, senza ROS installato
    pip install opencv-python
    pip install pandas pyarrow

rosbags (https://gitlab.com/ternaris/rosbags) permette di leggere bag ROS 1
e ROS 2 senza avere ROS installato — ideale per uso offline su laptop.

Formato flat prodotto
---------------------
<output_dir>/
    imu.parquet      [timestamp_ns, ax, ay, az, gx, gy, gz]
    gnss.parquet     [timestamp_ns, lat, lon, alt_wgs84_m, alt_agl_m, is_gt]
    frames.parquet   [timestamp_ns, filename]
    images/
        <timestamp_ns>.png
        ...

Sorgenti PPK (ground truth GPS post-processato)
------------------------------------------------
Il PPK può provenire da tre sorgenti, configurabili tramite il YAML:

  1. Topic nel bag  — impostare topics["gnss_gt"] al nome del topic
                      (es. "/fix_ppk", sensor_msgs/NavSatFix)

  2. File .pos esterno — lasciare topics["gnss_gt"] = None e passare
                         ppk_pos_path al costruttore.
                         Il file viene parsato con PosParser e le righe
                         vengono aggiunte a gnss.parquet con is_gt=True.

  3. Nessun PPK disponibile — topics["gnss_gt"] = None, ppk_pos_path = None.
                              gnss.parquet non avrà righe con is_gt=True.

Uso — caso 1 (PPK nel bag)
--------------------------
    conv = RosbagConverter(topics={
        "camera":   "/camera/image_raw",
        "imu":      "/imu/data",
        "gnss_gt":  "/fix_ppk",
        "gnss_in":  "/fix",
    })
    conv.convert("data/quarry1_synced.bag", "data/quarry1_flat/")

Uso — caso 2 (PPK da file .pos)
--------------------------------
    conv = RosbagConverter(
        topics={
            "camera":   "/camera/image_color/compressed",
            "imu":      "/imu/data",
            "gnss_gt":  None,
            "gnss_in":  "/fix",
        },
        ppk_pos_path="data/flight_dataset5.pos",
        ppk_quality_max=1,        # solo fix (default)
        ppk_gps_leapseconds=18,   # GPS-UTC offset (18 s dopo 2017-01-01)
    )
    conv.convert("data/lighthouse_benchmarking.bag", "data/lighthouse_flat/")

Uso — caso 3 (nessun PPK)
--------------------------
    conv = RosbagConverter(
        topics={
            "camera":   "/camera/image_color/compressed",
            "imu":      "/imu/data",
            "gnss_gt":  None,
            "gnss_in":  "/fix",
        },
    )
    conv.convert("data/lighthouse_benchmarking.bag", "data/lighthouse_flat/")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gnss_denied_nav.interfaces.base import Converter


class RosbagConverter(Converter):
    """
    Converte un ROS 1 bag (.bag) nel formato flat.

    Parametri
    ---------
    topics : dict
        Mappa ruolo → nome topic ROS.
        Chiavi attese: "camera", "imu", "gnss_gt", "gnss_in".
        "gnss_gt" può essere None se il PPK non è nel bag.
    force : bool
        Se True, riconverte anche se output_dir esiste già.
        Default False: se la cartella esiste, è un no-op.
    ppk_pos_path : str | None
        Path a un file .pos RTKLIB da usare come ground truth PPK esterno.
        Utilizzato solo se topics["gnss_gt"] è None.
        Se None e gnss_gt è None, gnss.parquet non avrà righe is_gt=True.
    ppk_quality_max : int
        Qualità massima accettata dal file .pos (Q <= ppk_quality_max).
        1 = solo fix (default), 2 = fix + float.
    ppk_gps_leapseconds : int
        Offset GPS Time − UTC [s]. 18 s dopo il 2017-01-01 (default).
    """

    name = "rosbag"

    def __init__(
        self,
        topics: dict[str, str | None],
        force: bool = False,
        ppk_pos_path: str | None = None,
        ppk_quality_max: int = 1,
        ppk_gps_leapseconds: int = 18,
    ) -> None:
        self._topics = topics
        self._force = force
        self._ppk_pos_path = ppk_pos_path
        self._ppk_quality_max = ppk_quality_max
        self._ppk_gps_leapseconds = ppk_gps_leapseconds

    def convert(self, source_path: str, output_dir: str) -> None:
        """
        Legge il bag (e opzionalmente il file .pos) e scrive il formato flat.

        Solleva
        -------
        FileNotFoundError : se source_path o ppk_pos_path non esistono
        RuntimeError      : se rosbags/dipendenze non sono installate
        """
        source = Path(source_path)
        out = Path(output_dir)

        if not source.exists():
            raise FileNotFoundError(f"Bag non trovato: {source}")

        if out.exists() and not self._force:
            # Niente da fare — la cache è già pronta
            return

        # ── import lazy: rosbags è opzionale ─────────────────────────────────
        try:
            from rosbags.rosbag1 import Reader
        except ImportError as exc:
            raise RuntimeError(
                "Il pacchetto 'rosbags' non è installato.\nEsegui: pip install rosbags"
            ) from exc

        try:
            import cv2
            import numpy as np
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "Dipendenze mancanti. Esegui:\n  pip install opencv-python pandas pyarrow"
            ) from exc

        out.mkdir(parents=True, exist_ok=True)
        (out / "images").mkdir(exist_ok=True)

        imu_rows: list[dict[str, Any]] = []
        gnss_rows: list[dict[str, Any]] = []
        frame_rows: list[dict[str, Any]] = []

        with Reader(source) as bag:
            for connection, timestamp_ns, rawdata in bag.messages():
                topic = connection.topic
                msgtype = connection.msgtype

                # ── IMU ───────────────────────────────────────────────────────
                if topic == self._topics.get("imu"):
                    msg = bag.deserialize(rawdata, msgtype)
                    imu_rows.append(
                        {
                            "timestamp_ns": timestamp_ns,
                            "ax": msg.linear_acceleration.x,
                            "ay": msg.linear_acceleration.y,
                            "az": msg.linear_acceleration.z,
                            "gx": msg.angular_velocity.x,
                            "gy": msg.angular_velocity.y,
                            "gz": msg.angular_velocity.z,
                        }
                    )

                # ── GNSS (ground truth nel bag e/o input) ─────────────────────
                elif topic in (
                    t
                    for t in (
                        self._topics.get("gnss_gt"),
                        self._topics.get("gnss_in"),
                    )
                    if t is not None
                ):
                    msg = bag.deserialize(rawdata, msgtype)
                    is_gt = topic == self._topics.get("gnss_gt")
                    gnss_rows.append(
                        {
                            "timestamp_ns": timestamp_ns,
                            "lat": msg.latitude,
                            "lon": msg.longitude,
                            "alt_wgs84_m": msg.altitude,
                            # alt_agl_m viene calcolato separatamente (GNSS - DEM);
                            # inizializzato a 0.0, il pipeline lo sovrascrive
                            "alt_agl_m": 0.0,
                            "is_gt": is_gt,
                        }
                    )

                # ── Camera ────────────────────────────────────────────────────
                elif topic == self._topics.get("camera"):
                    msg = bag.deserialize(rawdata, msgtype)
                    if "CompressedImage" in msgtype:
                        # sensor_msgs/CompressedImage → numpy via JPEG/PNG decode
                        buf = np.frombuffer(msg.data, dtype=np.uint8)
                        img_np = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    else:
                        # sensor_msgs/Image → numpy (raw)
                        img_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                            msg.height, msg.width, -1
                        )
                    if img_np is None:
                        continue
                    # ROS pubblica BGR — salviamo PNG
                    filename = f"{timestamp_ns}.png"
                    cv2.imwrite(str(out / "images" / filename), img_np)

                    frame_rows.append(
                        {
                            "timestamp_ns": timestamp_ns,
                            "filename": filename,
                        }
                    )

        # ── PPK da file .pos esterno (caso 2) ─────────────────────────────────
        # Si usa solo se gnss_gt non era un topic nel bag.
        if self._ppk_pos_path is not None and self._topics.get("gnss_gt") is None:
            from gnss_denied_nav.io.converters.pos_parser import PosParser  # lazy

            ppk_df = PosParser(
                quality_max=self._ppk_quality_max,
                gps_leapseconds=self._ppk_gps_leapseconds,
            ).parse(self._ppk_pos_path)

            # Manteniamo solo le colonne del schema gnss.parquet
            # (q_flag è colonna diagnostica del PosParser, non del flat format)
            ppk_rows = ppk_df[
                ["timestamp_ns", "lat", "lon", "alt_wgs84_m", "alt_agl_m", "is_gt"]
            ].to_dict("records")
            gnss_rows.extend(ppk_rows)

        # ── Scrivi Parquet ────────────────────────────────────────────────────
        pd.DataFrame(imu_rows).to_parquet(out / "imu.parquet", index=False)
        pd.DataFrame(gnss_rows).to_parquet(out / "gnss.parquet", index=False)
        pd.DataFrame(frame_rows).to_parquet(out / "frames.parquet", index=False)
