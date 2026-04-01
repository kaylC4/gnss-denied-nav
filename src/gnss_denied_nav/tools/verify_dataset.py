"""
verify_dataset.py — GUI di verifica del dataset flat estratto.

Avvia con:
    python -m gnss_denied_nav.tools.verify_dataset
    verify-dataset          (se installato con pip install -e .[viz])

Layout
------
┌─ SFOGLIA │ path ─────────────────── ● GPS │ ALTITUDE: Xm │ TIME: HH:MM:SS GMT ─┐
│  IMMAGINE DRONE  │     POSA DRONE (IMU)    │        GPS  vs  PPK               │
│                  │                         │                                   │
└──────────── slider per scorrere dataset ──────────────────────────────────────┘

Dipendenze (gruppo viz):
    pip install -e ".[viz]"   →  matplotlib, folium
    Pillow e pandas già nelle deps core.
"""

from __future__ import annotations

import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path
from tkinter import filedialog, ttk

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PIL import Image, ImageTk

# ── palette ──────────────────────────────────────────────────────────────────
BG = "#1a1b2e"
PANEL_BG = "#252640"
BORDER = "#3b3d5c"
ACCENT = "#7c3aed"
GPS_COLOR = "#60a5fa"
PPK_COLOR = "#34d399"
TEXT_FG = "#e2e8f0"
MUTED = "#64748b"
IMU_PITCH = "#f472b6"
IMU_ROLL  = "#fbbf24"
IMU_YAW   = "#60a5fa"
WHITE = "#ffffff"


# ── widget helpers ────────────────────────────────────────────────────────────


def _style_ax(ax: Axes) -> None:
    """Applica la palette scura a un Axes matplotlib."""
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=MUTED, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(0.5)


def _dark_figure(w: float, h: float) -> Figure:
    fig = Figure(figsize=(w, h), facecolor=PANEL_BG, tight_layout=True)
    return fig


# ── applicazione principale ───────────────────────────────────────────────────


class DatasetViewer(tk.Tk):
    """Finestra principale del visualizzatore di dataset flat."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Dataset Verifier — GNSS-denied Nav")
        self.configure(bg=BG)
        self.minsize(900, 520)

        # stato dataset
        self._root_path: Path | None = None
        self._frames_df: pd.DataFrame | None = None
        self._imu_df: pd.DataFrame | None = None
        self._odom_df: pd.DataFrame | None = None
        self._gnss_df: pd.DataFrame | None = None
        self._n_frames: int = 0
        self._current_idx: int = 0

        # handle cursore traiettoria e riferimento immagine
        self._photo_ref: ImageTk.PhotoImage | None = None
        self._cursor_dot: Line2D | None = None
        self._traj_xlim: tuple[float, float] = (0.0, 1.0)
        self._traj_ylim: tuple[float, float] = (0.0, 1.0)

        self._build_ui()

    # ── costruzione UI ─────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self._build_toolbar()
        self._build_panels()
        self._build_slider()

    def _build_toolbar(self) -> None:
        bar = tk.Frame(self, bg=BG, pady=8)
        bar.pack(fill="x", padx=14)

        # Pulsante Sfoglia
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

        # Campo path
        self._path_var = tk.StringVar(value="Seleziona una cartella dataset flat…")
        tk.Entry(
            bar,
            textvariable=self._path_var,
            bg=PANEL_BG,
            fg=TEXT_FG,
            insertbackground=TEXT_FG,
            font=("Helvetica", 10),
            relief="flat",
            width=54,
            disabledbackground=PANEL_BG,
            disabledforeground=MUTED,
        ).pack(side="left", padx=(8, 20), ipady=4)

        # Badge GPS  ● GPS
        self._gps_canvas = tk.Canvas(bar, width=12, height=12, bg=BG, highlightthickness=0)
        self._gps_circle = self._gps_canvas.create_oval(1, 1, 11, 11, fill=MUTED, outline="")
        self._gps_canvas.pack(side="left")
        tk.Label(bar, text=" GPS", bg=BG, fg=TEXT_FG, font=("Helvetica", 10)).pack(side="left")

        # Altitude
        self._alt_var = tk.StringVar(value="ALTITUDE: — m")
        tk.Label(bar, textvariable=self._alt_var, bg=BG, fg=TEXT_FG, font=("Helvetica", 10)).pack(
            side="left", padx=(18, 0)
        )

        # Time
        self._time_var = tk.StringVar(value="TIME: --:--:-- GMT")
        tk.Label(bar, textvariable=self._time_var, bg=BG, fg=TEXT_FG, font=("Helvetica", 10)).pack(
            side="left", padx=(18, 0)
        )

    def _build_panels(self) -> None:
        container = tk.Frame(self, bg=BG)
        container.pack(fill="both", expand=True, padx=14, pady=4)

        # ── Immagine drone ──────────────────────────────────────────────────
        left = tk.Frame(container, bg=PANEL_BG, highlightbackground=BORDER, highlightthickness=1)
        left.pack(side="left", fill="both", expand=True, padx=(0, 5))
        tk.Label(
            left, text="IMMAGINE DRONE", bg=PANEL_BG, fg=MUTED, font=("Helvetica", 8, "bold")
        ).pack(pady=(5, 0))
        self._img_label = tk.Label(left, bg=PANEL_BG, text="—", fg=MUTED, font=("Helvetica", 24))
        self._img_label.pack(fill="both", expand=True, padx=5, pady=5)

        # ── Posa drone (IMU) ────────────────────────────────────────────────
        mid = tk.Frame(container, bg=PANEL_BG, highlightbackground=BORDER, highlightthickness=1)
        mid.pack(side="left", fill="both", expand=True, padx=5)
        tk.Label(
            mid, text="POSA DRONE", bg=PANEL_BG, fg=MUTED, font=("Helvetica", 8, "bold")
        ).pack(pady=(5, 0))
        self._imu_fig = _dark_figure(3.8, 2.6)
        self._imu_ax = self._imu_fig.add_subplot(111)
        _style_ax(self._imu_ax)
        self._imu_canvas = FigureCanvasTkAgg(self._imu_fig, master=mid)
        self._imu_canvas.get_tk_widget().configure(bg=PANEL_BG)
        self._imu_canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

        # ── Traiettoria GPS / PPK ───────────────────────────────────────────
        right = tk.Frame(container, bg=PANEL_BG, highlightbackground=BORDER, highlightthickness=1)
        right.pack(side="left", fill="both", expand=True, padx=(5, 0))
        tk.Label(
            right, text="GPS  vs  PPK", bg=PANEL_BG, fg=MUTED, font=("Helvetica", 8, "bold")
        ).pack(pady=(5, 0))
        self._traj_fig = _dark_figure(3.8, 2.6)
        self._traj_ax = self._traj_fig.add_subplot(111)
        _style_ax(self._traj_ax)
        self._traj_canvas = FigureCanvasTkAgg(self._traj_fig, master=right)
        self._traj_canvas.get_tk_widget().configure(bg=PANEL_BG)
        self._traj_canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

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

        # Stile ttk
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

    # ── dataset loading ───────────────────────────────────────────────────────

    def _browse(self) -> None:
        folder = filedialog.askdirectory(title="Seleziona cartella dataset flat")
        if folder:
            self._load_dataset(Path(folder))

    def _load_dataset(self, root: Path) -> None:
        required = ["frames.parquet", "imu.parquet", "gnss.parquet"]
        missing = [f for f in required if not (root / f).exists()]
        if missing:
            self._path_var.set(f"✗ file mancanti: {', '.join(missing)}")
            return

        self._root_path = root
        self._path_var.set(str(root))

        self._frames_df = (
            pd.read_parquet(root / "frames.parquet")
            .sort_values("timestamp_ns")
            .reset_index(drop=True)
        )
        self._imu_df = (
            pd.read_parquet(root / "imu.parquet")
            .sort_values("timestamp_ns")
            .reset_index(drop=True)
        )
        self._gnss_df = (
            pd.read_parquet(root / "gnss.parquet")
            .sort_values("timestamp_ns")
            .reset_index(drop=True)
        )
        odom_path = root / "odometry.parquet"
        self._odom_df = (
            pd.read_parquet(odom_path).sort_values("timestamp_ns").reset_index(drop=True)
            if odom_path.exists()
            else None
        )
        self._n_frames = len(self._frames_df)
        self._slider.configure(to=max(0, self._n_frames - 1))
        self._slider_var.set(0)
        self._current_idx = 0

        self._draw_trajectory_base()
        self._update_frame(0)

    # ── slider callback ───────────────────────────────────────────────────────

    def _on_slider(self, value: str) -> None:
        idx = int(float(value))
        if idx != self._current_idx:
            self._current_idx = idx
            self._update_frame(idx)

    # ── aggiornamento per-frame ────────────────────────────────────────────────

    def _update_frame(self, idx: int) -> None:
        if self._frames_df is None:
            return
        row = self._frames_df.iloc[idx]
        ts_ns = int(row["timestamp_ns"])
        filename = str(row["filename"])

        self._update_image(filename)
        self._update_attitude_plot(ts_ns)
        self._update_trajectory_cursor(ts_ns)
        self._update_statusbar(ts_ns)
        self._frame_label.configure(text=f"frame {idx + 1} / {self._n_frames}")

    def _update_image(self, filename: str) -> None:
        if self._root_path is None:
            return
        path = self._root_path / "images" / filename
        if not path.exists():
            self._img_label.configure(image="", text=f"⚠ {filename}\nnon trovato")
            return
        try:
            img = Image.open(path).convert("RGB")
            # Fit nell'area disponibile preservando aspect ratio
            max_w = max(self._img_label.winfo_width(), 100)
            max_h = max(self._img_label.winfo_height(), 80)
            img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._photo_ref = photo  # evita garbage collection
            self._img_label.configure(image=photo, text="")
        except Exception as exc:
            self._img_label.configure(image="", text=f"⚠ {exc}")

    def _update_attitude_plot(self, ts_ns: int) -> None:
        ax = self._imu_ax
        ax.cla()
        _style_ax(ax)

        # ── Sorgente preferita: odometry (angoli fusi, no deriva) ─────────────
        if self._odom_df is not None:
            odom_ts = self._odom_df["timestamp_ns"].to_numpy(dtype=np.int64)
            center_idx = int(np.searchsorted(odom_ts, ts_ns))
            half = 50  # ±50 sample a ~30 Hz ≈ ±1.7 s
            lo = max(0, center_idx - half)
            hi = min(len(self._odom_df), center_idx + half)
            window = self._odom_df.iloc[lo:hi]

            if len(window) > 0:
                t_rel = (window["timestamp_ns"].to_numpy(dtype=np.float64) - ts_ns) / 1e9
                ax.plot(
                    t_rel, window["pitch_deg"], color=IMU_PITCH, linewidth=0.9, label="pitch °"
                )
                ax.plot(t_rel, window["roll_deg"], color=IMU_ROLL, linewidth=0.9, label="roll °")
                ax.plot(t_rel, window["yaw_deg"], color=IMU_YAW, linewidth=0.9, label="yaw °")
                ax.axvline(0.0, color=WHITE, linewidth=0.8, linestyle="--", alpha=0.6)
                ax.axhline(0.0, color=MUTED, linewidth=0.4, linestyle=":", alpha=0.4)
                ax.legend(fontsize=6, facecolor=PANEL_BG, edgecolor=BORDER,
                          labelcolor=TEXT_FG, loc="upper right")
                ax.set_xlabel("t [s]", color=MUTED, fontsize=7)
                ax.set_ylabel("gradi", color=MUTED, fontsize=7)
                self._imu_fig.tight_layout(pad=0.4)
                self._imu_canvas.draw()
                return

        # ── Fallback: pitch e roll dall'accelerometro IMU ─────────────────────
        if self._imu_df is None:
            self._imu_canvas.draw()
            return

        imu_ts = self._imu_df["timestamp_ns"].to_numpy(dtype=np.int64)
        center_idx = int(np.searchsorted(imu_ts, ts_ns))
        half = 300  # ±300 sample a 400 Hz ≈ ±0.75 s
        lo = max(0, center_idx - half)
        hi = min(len(self._imu_df), center_idx + half)
        window = self._imu_df.iloc[lo:hi]

        if len(window) == 0:
            self._imu_canvas.draw()
            return

        t_rel = (window["timestamp_ns"].to_numpy(dtype=np.float64) - ts_ns) / 1e9
        ax_v = window["ax"].to_numpy(dtype=np.float64)
        ay_v = window["ay"].to_numpy(dtype=np.float64)
        az_v = window["az"].to_numpy(dtype=np.float64)
        pitch = np.degrees(np.arctan2(-ax_v, np.sqrt(ay_v ** 2 + az_v ** 2)))
        roll  = np.degrees(np.arctan2(ay_v, az_v))

        ax.plot(t_rel, pitch, color=IMU_PITCH, linewidth=0.9, label="pitch °")
        ax.plot(t_rel, roll,  color=IMU_ROLL,  linewidth=0.9, label="roll °")
        ax.axvline(0.0, color=WHITE, linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axhline(0.0, color=MUTED, linewidth=0.4, linestyle=":", alpha=0.4)
        ax.legend(fontsize=6, facecolor=PANEL_BG, edgecolor=BORDER,
                  labelcolor=TEXT_FG, loc="upper right")
        ax.set_xlabel("t [s]", color=MUTED, fontsize=7)
        ax.set_ylabel("gradi (no yaw — riesegui extract-bag)", color=MUTED, fontsize=6)
        self._imu_fig.tight_layout(pad=0.4)
        self._imu_canvas.draw()

    def _draw_trajectory_base(self) -> None:
        """Disegna traiettoria completa (una sola volta al caricamento)."""
        if self._gnss_df is None:
            return
        ax = self._traj_ax
        ax.cla()
        _style_ax(ax)

        gps = self._gnss_df[~self._gnss_df["is_gt"].astype(bool)]
        ppk = self._gnss_df[self._gnss_df["is_gt"].astype(bool)]

        if len(gps):
            ax.plot(
                gps["lon"],
                gps["lat"],
                color=GPS_COLOR,
                linewidth=1.0,
                linestyle=":",
                label="GPS",
                alpha=0.85,
            )
        if len(ppk):
            ax.plot(
                ppk["lon"], ppk["lat"], color=PPK_COLOR, linewidth=1.3, linestyle="-", label="PPK"
            )

        ax.legend(
            fontsize=6,
            facecolor=PANEL_BG,
            edgecolor=BORDER,
            labelcolor=TEXT_FG,
            loc="upper right",
        )
        ax.set_xlabel("lon", color=MUTED, fontsize=7)
        ax.set_ylabel("lat", color=MUTED, fontsize=7)
        self._traj_fig.tight_layout(pad=0.4)
        self._traj_canvas.draw()

        # Salva limiti per non fare auto-zoom al cursore
        self._traj_xlim = ax.get_xlim()
        self._traj_ylim = ax.get_ylim()
        self._cursor_dot = None

    def _update_trajectory_cursor(self, ts_ns: int) -> None:
        if self._gnss_df is None:
            return
        ax = self._traj_ax

        # Rimuovi punto precedente
        if self._cursor_dot is not None:
            try:
                self._cursor_dot.remove()
            except Exception:
                pass
            self._cursor_dot = None

        # Ultimo fix GPS (non GT) ≤ ts_ns
        gps = self._gnss_df[~self._gnss_df["is_gt"].astype(bool)]
        gps_ts = gps["timestamp_ns"].to_numpy(dtype=np.int64)
        idx = int(np.searchsorted(gps_ts, ts_ns, side="right")) - 1

        if idx >= 0:
            r = gps.iloc[idx]
            (dot,) = ax.plot(
                float(r["lon"]),
                float(r["lat"]),
                "o",
                color=WHITE,
                markersize=5,
                zorder=6,
                markeredgewidth=0,
            )
            self._cursor_dot = dot

        # Mantieni zoom fisso
        ax.set_xlim(self._traj_xlim)
        ax.set_ylim(self._traj_ylim)
        self._traj_canvas.draw()

    def _update_statusbar(self, ts_ns: int) -> None:
        if self._gnss_df is None:
            return

        gps = self._gnss_df[~self._gnss_df["is_gt"].astype(bool)]
        gps_ts = gps["timestamp_ns"].to_numpy(dtype=np.int64)
        idx = int(np.searchsorted(gps_ts, ts_ns, side="right")) - 1

        if idx >= 0:
            alt = float(gps.iloc[idx]["alt_agl_m"])
            self._alt_var.set(f"ALTITUDE: {alt:.1f} m")
            self._gps_canvas.itemconfig(self._gps_circle, fill=GPS_COLOR)
        else:
            self._alt_var.set("ALTITUDE: — m")
            self._gps_canvas.itemconfig(self._gps_circle, fill=MUTED)

        dt = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc)
        self._time_var.set(f"TIME: {dt.strftime('%H:%M:%S')} GMT")


# ── entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    app = DatasetViewer()
    app.geometry("1180x620")
    app.mainloop()


if __name__ == "__main__":
    main()
