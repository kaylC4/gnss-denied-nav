"""
PosParser — legge file .pos di RTKLIB e produce un DataFrame compatibile
con il formato gnss.parquet usato nel resto della pipeline.

Formato .pos (RTKLIB)
---------------------
Righe di intestazione che iniziano con '%', poi dati tabulati:

    YYYY/MM/DD HH:MM:SS.SSS  lat  lon  height  Q  ns  sdn  sde  sdu  ...

Il timestamp è in GPS Time (GPST), non UTC.
GPS Time = UTC + gps_leapseconds  (18 secondi dopo il 2017-01-01).
Il converter sottrae automaticamente l'offset per produrre Unix timestamp [ns].

Flag di qualità Q:
    1 = fix      (più preciso — default: si mantengono solo questi)
    2 = float
    3 = sbas
    4 = dgps
    5 = single
    6 = ppp

Output DataFrame
----------------
    timestamp_ns  int64    Unix timestamp [ns] (UTC)
    lat           float64  latitudine WGS84 [deg]
    lon           float64  longitudine WGS84 [deg]
    alt_wgs84_m   float64  quota ellissoidica WGS84 [m]
    alt_agl_m     float64  quota AGL (inizializzata a 0.0, sovrascritta dal pipeline)
    is_gt         bool     sempre True (questa è la traccia ground-truth)
    q_flag        int64    flag qualità originale RTKLIB (per diagnostica)

Uso
---
    from gnss_denied_nav.io.converters.pos_parser import PosParser

    df = PosParser(quality_max=1).parse("flight_dataset5.pos")
    # oppure accetta anche Q=2 (float):
    df = PosParser(quality_max=2).parse("flight_dataset5.pos")
"""

from __future__ import annotations

import calendar
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class PosParser:
    """
    Converte un file .pos RTKLIB in un DataFrame con schema gnss.parquet.

    Parametri
    ---------
    quality_max : int
        Valore massimo del flag Q accettato (incluso).
        quality_max=1 → solo fix (default, più preciso).
        quality_max=2 → fix + float.
        quality_max=5 → tutto tranne PPP.
    gps_leapseconds : int
        Offset GPS Time − UTC in secondi interi.
        18 s per date dopo il 2017-01-01 (valore di default).
        Verificare se il dataset è più vecchio del 2017 (usare 17 s).
    """

    def __init__(
        self,
        quality_max: int = 1,
        gps_leapseconds: int = 18,
    ) -> None:
        self._quality_max = quality_max
        self._gps_leapseconds = gps_leapseconds

    def parse(self, path: str) -> pd.DataFrame:
        """
        Legge il file .pos e restituisce un DataFrame.

        Parametri
        ---------
        path : str
            Path al file .pos (RTKLIB output).

        Solleva
        -------
        FileNotFoundError : se il file non esiste
        RuntimeError      : se pandas non è installato
        ValueError        : se il file non contiene righe valide dopo il filtro qualità
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "Il pacchetto 'pandas' non è installato.\nEsegui: pip install pandas pyarrow"
            ) from exc

        pos_path = Path(path)
        if not pos_path.exists():
            raise FileNotFoundError(f"File .pos non trovato: {pos_path}")

        # ── Lettura grezza ────────────────────────────────────────────────────
        # Ogni riga dati ha 15 colonne separate da spazi (2 per timestamp + 13).
        # Le righe con '%' sono commenti e vengono saltate.
        col_names = [
            "date",
            "time",
            "lat",
            "lon",
            "height",
            "Q",
            "ns",
            "sdn",
            "sde",
            "sdu",
            "sdne",
            "sdeu",
            "sdun",
            "age",
            "ratio",
        ]
        df_raw = pd.read_csv(
            pos_path,
            sep=r"\s+",
            comment="%",
            header=None,
            names=col_names,
        )

        # ── Filtro qualità ────────────────────────────────────────────────────
        df_raw = df_raw[df_raw["Q"] <= self._quality_max].copy()

        if df_raw.empty:
            raise ValueError(
                f"Nessuna riga con Q <= {self._quality_max} trovata in {pos_path}.\n"
                f"Prova ad aumentare quality_max (es. quality_max=2 per accettare anche float)."
            )

        # ── Conversione timestamp GPST → Unix [ns] ────────────────────────────
        ts_ns = [self._gpst_to_unix_ns(d, t) for d, t in zip(df_raw["date"], df_raw["time"])]

        # ── Costruzione output DataFrame ──────────────────────────────────────
        out = pd.DataFrame(
            {
                "timestamp_ns": ts_ns,
                "lat": df_raw["lat"].values,
                "lon": df_raw["lon"].values,
                "alt_wgs84_m": df_raw["height"].values,
                "alt_agl_m": float("nan"),  # calcolata con DEM in post-processing
                "is_gt": True,
                "q_flag": df_raw["Q"].values,
            }
        )

        return out.sort_values("timestamp_ns").reset_index(drop=True)

    # ── helper privato ────────────────────────────────────────────────────────

    def _gpst_to_unix_ns(self, date_str: str, time_str: str) -> int:
        """
        Converte una coppia (date, time) in GPST a Unix timestamp [ns].

        GPST = UTC + gps_leapseconds  →  UTC = GPST - gps_leapseconds
        """
        # "2022/02/28" + "20:19:14.600"
        dt_gpst = datetime.strptime(f"{date_str} {time_str}", "%Y/%m/%d %H:%M:%S.%f")
        dt_utc = dt_gpst - timedelta(seconds=self._gps_leapseconds)

        # calendar.timegm interpreta la struttura come UTC (nessuna dipendenza
        # dal timezone locale della macchina, a differenza di mktime)
        unix_s = calendar.timegm(dt_utc.timetuple())
        unix_ns = unix_s * 1_000_000_000 + dt_utc.microsecond * 1_000
        return int(unix_ns)
