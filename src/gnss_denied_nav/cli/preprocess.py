"""
preprocess.py — CLI per eseguire la pipeline di preprocessing Stage 1-6.

Uso:
    preprocess-pipeline --config config/lighthouse_benchmarking.yaml \\
                        --data   data/lighthouse_10_flat/ \\
                        --out    data/lighthouse_preprocessed/

    python -m gnss_denied_nav.cli.preprocess --config ... --data ...

Legge frames.parquet e gnss.parquet dal dataset flat prodotto da extract-bag,
esegue i 6 stage in sequenza su ogni frame e scrive le immagini preprocessate
insieme a frames_preprocessed.parquet con schema [timestamp_ns, filename].
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

_BAR_WIDTH = 30
_IS_TTY = sys.stdout.isatty()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="preprocess-pipeline",
        description="Esegue la pipeline di preprocessing Stage 1-6 su un dataset flat.",
    )
    p.add_argument("--config", default="config/config.yaml", help="Path al file YAML della pipeline.")
    p.add_argument(
        "--data",
        required=True,
        help="Cartella flat dataset (output di extract-bag).",
    )
    p.add_argument(
        "--out",
        default="data/data_default",
        help="Directory di destinazione immagini preprocessate (default: <data>/preprocessed/).",
    )
    p.add_argument(
        "--frames",
        default=None,
        metavar="START:END",
        help="Range di frame da processare, es. '0:500' o '100:200'.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Sovrascrive output esistente se presente.",
    )
    return p


def _parse_frame_range(s: str, total: int) -> tuple[int, int]:
    """Converte '0:500' in (0, 500), clampato a [0, total]."""
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError(f"--frames deve essere nel formato START:END, ricevuto: {s!r}")
    start_raw = parts[0].strip()
    end_raw   = parts[1].strip()

    # ── start ────────────────────────────────────────────────────────────────
    if not start_raw:
        start = 0
        print("Warning: START non specificato, uso 0.")
    else:
        start = int(start_raw)
        if start < 0:
            print(f"Warning: START={start} è negativo, uso 0.")
            start = 0

    # ── end ──────────────────────────────────────────────────────────────────
    if not end_raw:
        end = total
        print(f"Warning: END non specificato, uso {total} (totale frame).")
    else:
        end = int(end_raw)
        if end > total:
            print(f"Warning: END={end} supera il totale ({total}), uso {total}.")
            end = total

    # ── consistenza ──────────────────────────────────────────────────────────
    if start >= end:
        raise ValueError(
            f"Range non valido: START={start} >= END={end}. "
            "Specifica un intervallo con START < END."
        )

    start = max(0, min(start, total))
    end = max(start, min(end, total))
    return start, end


def _last_alt_agl(
    gnss_ts: "np.ndarray",   # type: ignore[name-defined]  # noqa: F821
    gnss_alt: "np.ndarray",  # type: ignore[name-defined]  # noqa: F821
    ts: int,
    max_dt_ns: int = 5_000_000_000,  # 5 secondi — soglia massima di scarto accettabile, unità di misura nanosecondi [ns]
) -> float:
    """
    Restituisce la quota AGL del fix GNSS con timestamp più vicino a ts,
    sia esso precedente o successivo.
    Se lo scarto supera max_dt_ns, o nessun fix valido è trovato, ritorna 0.0.
    """
    import numpy as np

    # ── Candidati: indice sinistro (≤ ts) e destro (> ts) ───────────────────
    right_idx = int(np.searchsorted(gnss_ts, ts, side="right"))  # primo > ts
    left_idx  = right_idx - 1                                     # ultimo ≤ ts

    candidates = []
    for idx in (left_idx, right_idx):
        if 0 <= idx < len(gnss_ts):
            dt  = abs(int(gnss_ts[idx]) - ts)
            val = float(gnss_alt[idx])
            if not np.isnan(val) and val > 0:
                candidates.append((dt, val))

    if not candidates:
        # Nessun fix valido nell'intero array
        print("Warning: nessun fix GNSS valido trovato, quota impostata a 0.0.")
        return 0.0

    # ── Prendi il candidato con scarto minore ────────────────────────────────
    best_dt, best_val = min(candidates, key=lambda x: x[0])

    if best_dt > max_dt_ns:
        print(
            f"Warning: fix più vicino ha scarto {best_dt / 1e9:.2f}s "
            f"(soglia {max_dt_ns / 1e9:.0f}s), quota impostata a 0.0."
        )
        return 0.0

    return best_val


def _print_progress(
    current: int,
    total: int,
    t0: float,
    processed: int,
    skipped: int,
) -> None:
    """Stampa una riga di progresso aggiornata in-place, stile rosbag converter."""
    elapsed = time.monotonic() - t0
    pct = current / total if total > 0 else 0.0
    fps = processed / elapsed if elapsed > 0 else 0.0
    eta = (total - current) / fps if fps > 0 else 0.0
    filled = int(_BAR_WIDTH * pct)
    bar = "█" * filled + "░" * (_BAR_WIDTH - filled)
    line = (
        f"  [{bar}] {pct:5.1%}"
        f"  {current}/{total}"
        f"  ok={processed} skip={skipped}"
        f"  {fps:.1f} fr/s  ETA {eta:.0f}s"
    )
    if _IS_TTY:
        term_w = shutil.get_terminal_size(fallback=(120, 24)).columns
        sys.stdout.write(f"\r{line[:term_w].ljust(term_w)}")
    else:
        sys.stdout.write(f"{line}\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    import cv2
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from gnss_denied_nav.config import PipelineConfig
    from gnss_denied_nav.preprocessing.pipeline import PreprocessingPipeline

    # ── Carica config ────────────────────────────────────────────────────────
    cfg = PipelineConfig.from_yaml(args.config)

    # ── Percorsi ─────────────────────────────────────────────────────────────
    data_dir = Path(args.data)
    out_dir = Path(args.out) if args.out else data_dir / "preprocessed"

    if out_dir.exists() and not args.force:
        existing = list(out_dir.glob("*.png"))
        if existing:
            print(
                f"Errore: {out_dir} contiene già {len(existing)} immagini. "
                "Usa --force per sovrascrivere.",
                file=sys.stderr,
            )
            sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Carica frames e GNSS ─────────────────────────────────────────────────
    frames_df = (
        pd.read_parquet(data_dir / "frames.parquet")
        .sort_values("timestamp_ns")
        .reset_index(drop=True)
    )
    gnss_df = (
        pd.read_parquet(data_dir / "gnss.parquet")
        .sort_values("timestamp_ns")
        .reset_index(drop=True)
    )
    if "is_gt" in gnss_df.columns:
        gnss_df = gnss_df[~gnss_df["is_gt"].astype(bool)].reset_index(drop=True)

    gnss_ts = gnss_df["timestamp_ns"].to_numpy(dtype=np.int64)
    # Usa alt_agl_m se disponibile, altrimenti alt_wgs84_m come fallback
    if gnss_df["alt_agl_m"].notna().any():
        gnss_alt = gnss_df["alt_agl_m"].to_numpy(dtype=np.float64)
        alt_source = "alt_agl_m"
    elif "alt_wgs84_m" in gnss_df.columns and gnss_df["alt_wgs84_m"].notna().any():
        gnss_alt = gnss_df["alt_wgs84_m"].to_numpy(dtype=np.float64)
        alt_source = "alt_wgs84_m (fallback)"
    else:
        print("Errore: nessun dato di altitudine disponibile in gnss.parquet.", file=sys.stderr)
        sys.exit(1)

    # ── Selezione range frame ────────────────────────────────────────────────
    total_frames = len(frames_df)
    if args.frames:
        start_idx, end_idx = _parse_frame_range(args.frames, total_frames)
        frames_df = frames_df.iloc[start_idx:end_idx].reset_index(drop=True)
    n_frames = len(frames_df)

    print(f"Dataset  : {data_dir}")
    print(f"Output   : {out_dir}")
    print(f"Config   : {args.config}")
    print(f"Frame    : {n_frames} (di {total_frames} totali)")
    print(f"Altitudine: {alt_source}")
    print()

    # ── Setup pipeline ───────────────────────────────────────────────────────
    pipeline = PreprocessingPipeline(cfg)
    if cfg.inspection.enabled:
        pipeline.prepare_inspection(n_frames)

    # ── Iterazione frame ─────────────────────────────────────────────────────
    images_dir = data_dir / "images"
    out_records: list[dict[str, object]] = []
    processed = 0
    skipped = 0
    t0 = time.monotonic()

    for i, row in enumerate(frames_df.itertuples(index=False)):
        ts: int = int(row.timestamp_ns)  # type: ignore[attr-defined]
        filename: str = str(row.filename)  # type: ignore[attr-defined]
        alt_agl_m = _last_alt_agl(gnss_ts, gnss_alt, ts)

        img = cv2.imread(str(images_dir / filename))
        if img is None:
            skipped += 1
            _print_progress(i + 1, n_frames, t0, processed, skipped)
            continue

        result = pipeline.run(img, ts, filename, alt_agl_m, frame_index=i)

        out_name = f"{ts}_preprocessed.png"
        cv2.imwrite(str(out_dir / out_name), result.image)
        out_records.append({"timestamp_ns": ts, "filename": out_name})
        processed += 1

        _print_progress(i + 1, n_frames, t0, processed, skipped)

    print()  # nuova riga dopo la progress bar

    # ── Finalize inspection ──────────────────────────────────────────────────
    pipeline.finalize_inspection()

    # ── Scrivi frames_preprocessed.parquet ──────────────────────────────────
    if out_records:
        table = pa.table(
            {
                "timestamp_ns": pa.array(
                    [r["timestamp_ns"] for r in out_records], type=pa.int64()
                ),
                "filename": pa.array(
                    [r["filename"] for r in out_records], type=pa.string()
                ),
            }
        )
        pq.write_table(table, out_dir / "frames_preprocessed.parquet")

    # ── Riepilogo finale ─────────────────────────────────────────────────────
    elapsed = time.monotonic() - t0
    size_mb = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / 1_048_576

    print(f"Frame processati  : {processed:,}")
    print(f"Frame skippati    : {skipped:,}")
    print(f"Tempo totale      : {elapsed:.1f}s")
    print(f"Dimensione output : {size_mb:.1f} MB  ({out_dir})")


if __name__ == "__main__":
    main()
