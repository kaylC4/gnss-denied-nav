"""Stub — TileProvider offline da archivio MBTiles locale."""

from __future__ import annotations

from gnss_denied_nav.interfaces.base import TileProvider
from gnss_denied_nav.interfaces.contracts import LatLon, TileMosaic


class MBTilesTileProvider(TileProvider):
    """
    TODO:
        - Apertura del file .mbtiles (SQLite) tramite percorso configurabile
        - Query tile x/y/z tramite bounding box e GSD target
        - Assemblaggio in TileMosaic con georeferenziazione bbox e gsd_m
    """

    def __init__(self, mbtiles_path: str, zoom: int = 17) -> None:
        self._path = mbtiles_path
        self._zoom = zoom

    def get_mosaic(self, center: LatLon, radius_m: float, gsd_m: float) -> TileMosaic:
        raise NotImplementedError("MBTilesTileProvider.get_mosaic() — da implementare")

    @property
    def name(self) -> str:
        return "offline_mbtiles"
