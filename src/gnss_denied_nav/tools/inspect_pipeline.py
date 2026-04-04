"""
inspect_pipeline.py — GUI di ispezione degli output intermedi della pipeline.

Avvia con:
    python -m gnss_denied_nav.tools.inspect_pipeline
    inspect-pipeline          (se installato con pip install -e .[viz])

Layout
------
┌─ SFOGLIA │ path ─────────────────────────────── frame N / M ───────────────┐
│  Stage 1          │  Stage 2          │  Stage 3                           │
│  Undistort        │  Warp Nadir       │  North Align                       │
│                   │                   │                                    │
│  Stage 4          │  Stage 5          │  Stage 6                           │
│  GSD Match        │  Crop/Pad         │  Domain Norm                       │
│                   │                   │                                    │
├─── PARAMETRI (stage selezionato) ──────────────────────────────────────────┤
│  { "K_new": [...], "balance": 0.0, ... }                                  │
└──────────── slider per scorrere frame campionati ──────────────────────────┘

Dipendenze (gruppo viz):
    pip install -e ".[viz]"   →  matplotlib
    Pillow e pandas già nelle deps core.
"""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import pandas as pd
from PIL import Image, ImageTk

# ── palette (coerente con verify_dataset) ────────────────────────────────────
BG = "#1a1b2e"
PANEL_BG = "#252640"
BORDER = "#3b3d5c"
ACCENT = "#7c3aed"
TEXT_FG = "#e2e8f0"
MUTED = "#64748b"
WHITE = "#ffffff"
SELECTED_BG = "#3b3d6c"

_STAGE_LABELS: dict[int, str] = {
    1: "Stage 1 — Undistort",
    2: "Stage 2 — Warp Nadir",
    3: "Stage 3 — North Align",
    4: "Stage 4 — GSD Match",
    5: "Stage 5 — Crop / Pad",
    6: "Stage 6 — Domain Norm",
}

_STAGE_FILES: dict[int, str] = {
    1: "s1_undistort.png",
    2: "s2_warp_nadir.png",
    3: "s3_north_align.png",
    4: "s4_gsd_match.png",
    5: "s5_crop_pad.png",
    6: "s6_domain_norm.png",
}

_PARAMS_FILES: dict[int, str] = {
    1: "s1_params.json",
    2: "s2_params.json",
    3: "s3_params.json",
    4: "s4_params.json",
    5: "s5_params.json",
    6: "s6_params.json",
}


class PipelineInspector(tk.Tk):
    """Finestra principale dell'inspector di pipeline."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Pipeline Inspector — GNSS-denied Nav")
        self.configure(bg=BG)
        self.minsize(1100, 700)

        # stato
        self._root_path: Path | None = None
        self._index_df: pd.DataFrame | None = None
        self._n_frames: int = 0
        self._current_idx: int = 0
        self._selected_stage: int = 1
        self._photo_refs: dict[int, ImageTk.PhotoImage | None] = {}

        self._build_ui()

    # ── costruzione UI ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self._build_toolbar()
        self._build_stage_grid()
        self._build_params_panel()
        self._build_slider()

    def _build_toolbar(self) -> None:
        bar = tk.Frame(self, bg=BG, pady=8)
        bar.pack(fill="x", padx=14)

        tk.Button(
            bar,
            text="SFOGLIA",
            command=self._browse,
            bg=ACCENT,
            fg=WHITE,
            font=("Helvetica", 10, "bold"),
            relief="flat",
            padx=12,
            pady=4,
            cursor="hand2",
            activebackground="#6d28d9",
            activeforeground=WHITE,
        ).pack(side="left")

        self._path_var = tk.StringVar(value="Seleziona la cartella debug/stages…")
        tk.Entry(
            bar,
            textvariable=self._path_var,
            bg=PANEL_BG,
            fg=TEXT_FG,
            insertbackground=TEXT_FG,
            font=("Helvetica", 10),
            relief="flat",
            width=60,
            disabledbackground=PANEL_BG,
            disabledforeground=MUTED,
        ).pack(side="left", padx=(8, 20), ipady=4)

        self._frame_info_var = tk.StringVar(value="")
        tk.Label(
            bar,
            textvariable=self._frame_info_var,
            bg=BG,
            fg=TEXT_FG,
            font=("Helvetica", 10),
        ).pack(side="right")

    def _build_stage_grid(self) -> None:
        """Griglia 2×3 per i 6 stage."""
        self._grid_frame = tk.Frame(self, bg=BG)
        self._grid_frame.pack(fill="both", expand=True, padx=14, pady=4)

        self._stage_panels: dict[int, tk.Frame] = {}
        self._stage_img_labels: dict[int, tk.Label] = {}

        for i, stage in enumerate(range(1, 7)):
            row, col = divmod(i, 3)

            panel = tk.Frame(
                self._grid_frame,
                bg=PANEL_BG,
                highlightbackground=BORDER,
                highlightthickness=1,
                cursor="hand2",
            )
            panel.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")
            panel.bind("<Button-1>", lambda e, s=stage: self._select_stage(s))

            label_text = _STAGE_LABELS.get(stage, f"Stage {stage}")
            header = tk.Label(
                panel,
                text=label_text,
                bg=PANEL_BG,
                fg=MUTED,
                font=("Helvetica", 8, "bold"),
            )
            header.pack(pady=(4, 0))
            header.bind("<Button-1>", lambda e, s=stage: self._select_stage(s))

            img_label = tk.Label(
                panel, bg=PANEL_BG, text="—", fg=MUTED, font=("Helvetica", 16),
            )
            img_label.pack(fill="both", expand=True, padx=4, pady=4)
            img_label.bind("<Button-1>", lambda e, s=stage: self._select_stage(s))

            self._stage_panels[stage] = panel
            self._stage_img_labels[stage] = img_label

        # Rendi la griglia espandibile
        for col in range(3):
            self._grid_frame.columnconfigure(col, weight=1)
        for row in range(2):
            self._grid_frame.rowconfigure(row, weight=1)

    def _build_params_panel(self) -> None:
        """Pannello inferiore per i parametri JSON dello stage selezionato."""
        params_frame = tk.Frame(
            self, bg=PANEL_BG, highlightbackground=BORDER, highlightthickness=1,
        )
        params_frame.pack(fill="x", padx=14, pady=(4, 2))

        self._params_header_var = tk.StringVar(value="PARAMETRI — Stage 1")
        tk.Label(
            params_frame,
            textvariable=self._params_header_var,
            bg=PANEL_BG,
            fg=MUTED,
            font=("Helvetica", 8, "bold"),
        ).pack(anchor="w", padx=8, pady=(4, 0))

        self._params_text = tk.Text(
            params_frame,
            bg=PANEL_BG,
            fg=TEXT_FG,
            font=("Courier", 9),
            height=6,
            wrap="word",
            relief="flat",
            insertbackground=TEXT_FG,
            selectbackground=ACCENT,
            state="disabled",
        )
        self._params_text.pack(fill="x", padx=8, pady=(2, 6))

    def _build_slider(self) -> None:
        bottom = tk.Frame(self, bg=BG)
        bottom.pack(fill="x", padx=14, pady=(2, 10))

        self._frame_label = tk.Label(
            bottom,
            text="frame — / —",
            bg=BG,
            fg=MUTED,
            font=("Helvetica", 9),
        )
        self._frame_label.pack(side="right", padx=6)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Flat.Horizontal.TScale",
            background=BG,
            troughcolor=PANEL_BG,
            sliderthickness=18,
            sliderrelief="flat",
        )

        self._slider_var = tk.DoubleVar(value=0)
        self._slider = ttk.Scale(
            bottom,
            from_=0,
            to=0,
            orient="horizontal",
            variable=self._slider_var,
            command=self._on_slider,
            style="Flat.Horizontal.TScale",
        )
        self._slider.pack(fill="x")

    # ── dataset loading ──────────────────────────────────────────────────────

    def _browse(self) -> None:
        folder = filedialog.askdirectory(title="Seleziona cartella debug/stages")
        if folder:
            self._load_inspection(Path(folder))

    def _load_inspection(self, root: Path) -> None:
        index_path = root / "index.parquet"
        if not index_path.exists():
            self._path_var.set(f"✗ index.parquet non trovato in {root}")
            return

        self._root_path = root
        self._path_var.set(str(root))
        self._index_df = pd.read_parquet(index_path)
        self._n_frames = len(self._index_df)

        self._slider.configure(to=max(0, self._n_frames - 1))
        self._slider_var.set(0)
        self._current_idx = 0
        self._update_all(0)

    # ── interazione ──────────────────────────────────────────────────────────

    def _on_slider(self, value: str) -> None:
        idx = int(float(value))
        if idx != self._current_idx:
            self._current_idx = idx
            self._update_all(idx)

    def _select_stage(self, stage: int) -> None:
        # Deseleziona precedente
        old = self._stage_panels.get(self._selected_stage)
        if old:
            old.configure(highlightbackground=BORDER, highlightthickness=1)

        self._selected_stage = stage

        # Evidenzia nuovo
        new = self._stage_panels.get(stage)
        if new:
            new.configure(highlightbackground=ACCENT, highlightthickness=2)

        self._update_params(self._current_idx)

    # ── aggiornamento ────────────────────────────────────────────────────────

    def _update_all(self, idx: int) -> None:
        if self._index_df is None or self._root_path is None:
            return

        row = self._index_df.iloc[idx]
        frame_dir = Path(str(row["frame_dir"]))
        ts_ns = int(row["timestamp_ns"])

        self._frame_info_var.set(f"ts: {ts_ns}")
        self._frame_label.configure(text=f"frame {idx + 1} / {self._n_frames}")

        # Aggiorna tutte le 6 miniature
        for stage in range(1, 7):
            self._update_stage_image(frame_dir, stage)

        self._update_params(idx)

    def _update_stage_image(self, frame_dir: Path, stage: int) -> None:
        label = self._stage_img_labels[stage]
        img_file = _STAGE_FILES.get(stage, "")
        img_path = frame_dir / img_file

        if not img_path.exists():
            label.configure(image="", text="N/A")
            self._photo_refs[stage] = None
            return

        try:
            img = Image.open(img_path).convert("RGB")
            max_w = max(label.winfo_width(), 120)
            max_h = max(label.winfo_height(), 80)
            img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._photo_refs[stage] = photo
            label.configure(image=photo, text="")
        except Exception as exc:
            label.configure(image="", text=f"⚠ {exc}")
            self._photo_refs[stage] = None

    def _update_params(self, idx: int) -> None:
        stage = self._selected_stage
        self._params_header_var.set(
            f"PARAMETRI — {_STAGE_LABELS.get(stage, f'Stage {stage}')}"
        )

        # Evidenzia pannello selezionato
        for s, panel in self._stage_panels.items():
            if s == stage:
                panel.configure(highlightbackground=ACCENT, highlightthickness=2)
            else:
                panel.configure(highlightbackground=BORDER, highlightthickness=1)

        if self._index_df is None:
            return

        row = self._index_df.iloc[idx]
        frame_dir = Path(str(row["frame_dir"]))
        params_file = _PARAMS_FILES.get(stage, "")
        params_path = frame_dir / params_file

        self._params_text.configure(state="normal")
        self._params_text.delete("1.0", "end")

        if params_path.exists():
            try:
                data = json.loads(params_path.read_text())
                text = json.dumps(data, indent=2, ensure_ascii=False)
                self._params_text.insert("1.0", text)
            except Exception as exc:
                self._params_text.insert("1.0", f"Errore: {exc}")
        else:
            self._params_text.insert("1.0", "(parametri non disponibili per questo stage)")

        self._params_text.configure(state="disabled")


# ── entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    app = PipelineInspector()
    app.geometry("1200x750")
    app.mainloop()


if __name__ == "__main__":
    main()
