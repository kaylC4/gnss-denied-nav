"""
extract_bag.py — CLI per convertire un ROS bag nel formato flat (Parquet + PNG).

Uso:
    extract-bag --bag data/lighthouse.bag --out data/lighthouse_flat/
    extract-bag --bag data/lighthouse.bag --pos data/lighthouse.pos --out data/flat/

Wrapper sottile attorno a RosbagConverter.  I topic di default corrispondono
alla configurazione lighthouse_benchmarking.yaml.
"""

from __future__ import annotations

import argparse


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="extract-bag",
        description="Converte un ROS 1 bag nel formato flat (Parquet + immagini PNG).",
    )
    p.add_argument(
        "--bag",
        required=True,
        help="Path al file .bag ROS 1.",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Cartella di output per il dataset flat.",
    )

    # ── topic override ──────────────────────────────────────────────────────
    p.add_argument(
        "--camera-topic",
        default="/camera/image_color/compressed",
        help="Topic camera (default: /camera/image_color/compressed).",
    )
    p.add_argument(
        "--imu-topic",
        default="/imu/data",
        help="Topic IMU (default: /imu/data).",
    )
    p.add_argument(
        "--gnss-topic",
        default="/fix",
        help="Topic GNSS di ingresso (default: /fix).",
    )
    p.add_argument(
        "--gnss-gt-topic",
        default=None,
        help="Topic GNSS ground truth nel bag (default: nessuno).",
    )
    p.add_argument(
        "--odometry-topic",
        default="/Odometry",
        help="Topic nav_msgs/Odometry per roll/pitch/yaw (default: /Odometry).",
    )

    # ── PPK da file .pos ────────────────────────────────────────────────────
    p.add_argument(
        "--pos",
        default=None,
        help="Path a un file .pos RTKLIB per il ground truth PPK.",
    )
    p.add_argument(
        "--ppk-quality-max",
        type=int,
        default=1,
        help="Qualità massima accettata dal .pos: 1=fix (default), 2=fix+float.",
    )
    p.add_argument(
        "--ppk-leapseconds",
        type=int,
        default=18,
        help="Offset GPST−UTC in secondi (default: 18, valido ≥ 2017-01-01).",
    )

    # ── opzioni generali ────────────────────────────────────────────────────
    p.add_argument(
        "--force",
        action="store_true",
        help="Riconverti anche se la cartella di output esiste già.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    from gnss_denied_nav.io.converters.rosbag import RosbagConverter

    converter = RosbagConverter(
        topics={
            "camera": args.camera_topic,
            "imu": args.imu_topic,
            "gnss_gt": args.gnss_gt_topic,
            "gnss_in": args.gnss_topic,
            "odometry": args.odometry_topic,
        },
        force=args.force,
        ppk_pos_path=args.pos,
        ppk_quality_max=args.ppk_quality_max,
        ppk_gps_leapseconds=args.ppk_leapseconds,
    )

    import time
    from pathlib import Path

    print(f"Conversione: {args.bag} → {args.out}")
    t0 = time.monotonic()
    converter.convert(source_path=args.bag, output_dir=args.out)
    elapsed = time.monotonic() - t0

    out = Path(args.out)
    n_images = len(list((out / "images").glob("*.png"))) if (out / "images").exists() else 0
    size_mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1_048_576

    print(f"\n  Completato in {elapsed:.1f}s")
    print(f"  Immagini estratte : {n_images:,}")
    print(f"  Dimensione output : {size_mb:.1f} MB  ({args.out})")


if __name__ == "__main__":
    main()
