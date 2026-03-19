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

Uso
---
    from gnss_denied_nav.io.converters.rosbag import RosbagConverter

    conv = RosbagConverter(topics={
        "camera":   "/camera/image_raw",
        "imu":      "/imu/data",
        "gnss_gt":  "/fix_ppk",
        "gnss_in":  "/fix",
    })
    conv.convert(
        source_path="data/quarry1_synced.bag",
        output_dir="data/quarry1_flat/",
    )
"""
from __future__ import annotations

from pathlib import Path

from gnss_denied_nav.interfaces.base import Converter


class RosbagConverter(Converter):
    """
    Converte un ROS 1 bag (.bag) nel formato flat.

    Parametri
    ---------
    topics : dict
        Mappa ruolo → nome topic ROS.
        Chiavi attese: "camera", "imu", "gnss_gt", "gnss_in".
    force : bool
        Se True, riconverte anche se output_dir esiste già.
        Default False: se la cartella esiste, è un no-op.
    """

    name = "rosbag"

    def __init__(
        self,
        topics: dict[str, str],
        force: bool = False,
    ) -> None:
        self._topics = topics
        self._force  = force

    def convert(self, source_path: str, output_dir: str) -> None:
        """
        Legge il bag e scrive il formato flat in output_dir.

        Solleva
        -------
        FileNotFoundError : se source_path non esiste
        RuntimeError      : se rosbags non è installato
        """
        source = Path(source_path)
        out    = Path(output_dir)

        if not source.exists():
            raise FileNotFoundError(f"Bag non trovato: {source}")

        if out.exists() and not self._force:
            # Niente da fare — la cache è già pronta
            return

        # ── import lazy: rosbags è opzionale ─────────────────────────────────
        try:
            from rosbags.rosbag1 import Reader                    # type: ignore[import]
            from rosbags.typesys import get_types_from_msg, register_types  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "Il pacchetto 'rosbags' non è installato.\n"
                "Esegui: pip install rosbags"
            ) from exc

        try:
            import cv2                                             # type: ignore[import]
            import numpy as np
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "Dipendenze mancanti. Esegui:\n"
                "  pip install opencv-python pandas pyarrow"
            ) from exc

        out.mkdir(parents=True, exist_ok=True)
        (out / "images").mkdir(exist_ok=True)

        imu_rows    : list[dict] = []
        gnss_rows   : list[dict] = []
        frame_rows  : list[dict] = []

        with Reader(source) as bag:
            for connection, timestamp_ns, rawdata in bag.messages():
                topic = connection.topic
                msgtype = connection.msgtype

                # ── IMU ───────────────────────────────────────────────────────
                if topic == self._topics.get("imu"):
                    msg = bag.deserialize(rawdata, msgtype)
                    imu_rows.append({
                        "timestamp_ns": timestamp_ns,
                        "ax": msg.linear_acceleration.x,
                        "ay": msg.linear_acceleration.y,
                        "az": msg.linear_acceleration.z,
                        "gx": msg.angular_velocity.x,
                        "gy": msg.angular_velocity.y,
                        "gz": msg.angular_velocity.z,
                    })

                # ── GNSS (ground truth e input) ───────────────────────────────
                elif topic in (
                    t for t in (
                        self._topics.get("gnss_gt"),
                        self._topics.get("gnss_in"),
                    ) if t is not None
                ):
                    msg = bag.deserialize(rawdata, msgtype)
                    is_gt = (topic == self._topics.get("gnss_gt"))
                    gnss_rows.append({
                        "timestamp_ns": timestamp_ns,
                        "lat":          msg.latitude,
                        "lon":          msg.longitude,
                        "alt_wgs84_m":  msg.altitude,
                        # alt_agl_m viene calcolato separatamente (GNSS - DEM)
                        # qui lo inizializziamo a 0.0; il pipeline lo sovrascrive
                        "alt_agl_m":    0.0,
                        "is_gt":        is_gt,
                    })

                # ── Camera ────────────────────────────────────────────────────
                elif topic == self._topics.get("camera"):
                    msg = bag.deserialize(rawdata, msgtype)

                    if "CompressedImage" in msgtype:
                        # sensor_msgs/CompressedImage → numpy array via JPEG/PNG decode
                        buf = np.frombuffer(msg.data, dtype=np.uint8)
                        img_np = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    else:
                        # sensor_msgs/Image → numpy array (raw)
                        img_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                            msg.height, msg.width, -1
                        )
                    # ROS pubblica BGR — salviamo PNG
                    filename = f"{timestamp_ns}.png"
                    cv2.imwrite(str(out / "images" / filename), img_np)

                    frame_rows.append({
                        "timestamp_ns": timestamp_ns,
                        "filename":     filename,
                    })

        # ── Scrivi Parquet ────────────────────────────────────────────────────
        pd.DataFrame(imu_rows).to_parquet(out / "imu.parquet",    index=False)
        pd.DataFrame(gnss_rows).to_parquet(out / "gnss.parquet",  index=False)
        pd.DataFrame(frame_rows).to_parquet(out / "frames.parquet", index=False)
