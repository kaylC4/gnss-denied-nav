"""Stub — TileProvider online via Google Maps Static API."""

from __future__ import annotations

from gnss_denied_nav.interfaces.base import TileProvider
from gnss_denied_nav.interfaces.contracts import LatLon, TileMosaic


class GoogleMapsTileProvider(TileProvider):
    """
    TODO:
        - Download tile satellitari via Google Maps Static API (chiave configurabile)
        - Gestione cache locale per evitare richieste duplicate
        - Assemblaggio in TileMosaic con georeferenziazione bbox e gsd_m
    """

    def __init__(self, api_key: str, zoom: int = 17) -> None:
        self._api_key = api_key
        self._zoom = zoom

    def get_mosaic(self, center: LatLon, radius_m: float, gsd_m: float) -> TileMosaic:
        raise NotImplementedError("GoogleMapsTileProvider.get_mosaic() — da implementare")

    @property
    def name(self) -> str:
        return "google_maps"
