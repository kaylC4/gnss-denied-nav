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
    gnss.parquet      [timestamp_ns, lat, lon, alt_wgs84_m, alt_agl_m, is_gt]
    odometry.parquet  [timestamp_ns, roll_deg, pitch_deg, yaw_deg]
    frames.parquet    [timestamp_ns, filename]
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

import math
import sys
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
            from rosbags.typesys import Stores, get_typestore
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
        odom_rows: list[dict[str, Any]] = []
        frame_rows: list[dict[str, Any]] = []

        typestore = get_typestore(Stores.ROS1_NOETIC)

        with Reader(source) as _bag:
            bag: Any = _bag  # rosbags stubs variano tra versioni

            total_msgs = sum(c.msgcount for c in bag.connections)
            print(f"  Messaggi totali nel bag: {total_msgs:,}")
            print(f"  Topic attivi: camera={self._topics.get('camera')}  "
                  f"imu={self._topics.get('imu')}  gnss={self._topics.get('gnss_in')}")
            print()

            processed = 0
            _BAR_WIDTH = 30
            _PRINT_EVERY = max(1, total_msgs // 200)  # aggiorna ogni ~0.5%

            for connection, _bag_ts, rawdata in bag.messages():
                topic = connection.topic
                msgtype = connection.msgtype

                # ── IMU ───────────────────────────────────────────────────────
                if topic == self._topics.get("imu"):
                    msg = typestore.deserialize_ros1(rawdata, msgtype)
                    ts = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
                    imu_rows.append(
                        {
                            "timestamp_ns": ts,
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
                    msg = typestore.deserialize_ros1(rawdata, msgtype)
                    ts = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
                    is_gt = topic == self._topics.get("gnss_gt")
                    gnss_rows.append(
                        {
                            "timestamp_ns": ts,
                            "lat": msg.latitude,
                            "lon": msg.longitude,
                            "alt_wgs84_m": msg.altitude,
                            # alt_agl_m viene calcolato separatamente (GNSS - DEM);
                            # NaN finché non viene eseguito il post-processing con DEM
                            "alt_agl_m": float("nan"),
                            "is_gt": is_gt,
                        }
                    )

                # ── Odometry ──────────────────────────────────────────────────
                elif topic == self._topics.get("odometry"):
                    msg = typestore.deserialize_ros1(rawdata, msgtype)
                    ts = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
                    q = msg.pose.pose.orientation
                    # Quaternione → roll / pitch / yaw [deg]
                    sinr = 2.0 * (q.w * q.x + q.y * q.z)
                    cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
                    roll_deg = math.degrees(math.atan2(sinr, cosr))
                    sinp = 2.0 * (q.w * q.y - q.z * q.x)
                    pitch_deg = math.degrees(
                        math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
                    )
                    siny = 2.0 * (q.w * q.z + q.x * q.y)
                    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                    yaw_deg = math.degrees(math.atan2(siny, cosy))
                    odom_rows.append(
                        {
                            "timestamp_ns": ts,
                            "roll_deg": roll_deg,
                            "pitch_deg": pitch_deg,
                            "yaw_deg": yaw_deg,
                        }
                    )

                # ── Camera ────────────────────────────────────────────────────
                elif topic == self._topics.get("camera"):
                    msg = typestore.deserialize_ros1(rawdata, msgtype)
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
                    ts = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
                    filename = f"{ts}.png"
                    cv2.imwrite(str(out / "images" / filename), img_np)

                    frame_rows.append(
                        {
                            "timestamp_ns": ts,
                            "filename": filename,
                        }
                    )

                processed += 1
                if processed % _PRINT_EVERY == 0 or processed == total_msgs:
                    pct = processed / total_msgs if total_msgs else 1.0
                    filled = int(_BAR_WIDTH * pct)
                    bar = "█" * filled + "░" * (_BAR_WIDTH - filled)
                    line = (
                        f"  [{bar}] {pct:5.1%}  "
                        f"IMU: {len(imu_rows):5,}  "
                        f"GNSS: {len(gnss_rows):4,}  "
                        f"Odom: {len(odom_rows):4,}  "
                        f"Frame: {len(frame_rows):4,}"
                    )
                    sys.stdout.write(f"\r\033[K{line}")
                    sys.stdout.flush()

            sys.stdout.write("\n")
            sys.stdout.flush()

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
        pd.DataFrame(odom_rows).to_parquet(out / "odometry.parquet", index=False)
        pd.DataFrame(frame_rows).to_parquet(out / "frames.parquet", index=False)
