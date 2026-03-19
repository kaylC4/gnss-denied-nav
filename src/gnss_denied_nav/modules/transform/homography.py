"""Stub — ViewTransformer via omografia inversa (drone → ortometrico)."""
from __future__ import annotations

import numpy as np

from gnss_denied_nav.interfaces.base import ViewTransformer
from gnss_denied_nav.interfaces.contracts import CameraPose, TransformedQuery


class HomographyViewTransformer(ViewTransformer):
    """
    Approccio inverso: trasforma il frame drone nel reference frame ortometrico.
    Applicato una sola volta sulla query — non sulle N patch candidate.
    TODO:
        - H = K @ [r1 | r2 | t/alt_agl] — omografia piano-piano
        - cv2.warpPerspective con INTER_LINEAR + BORDER_CONSTANT
        - valid_mask: True dove il frustum copre il piano del suolo
    """
    def transform(
        self,
        drone_frame: np.ndarray,
        pose: CameraPose,
        camera_matrix: np.ndarray,
        target_gsd_m: float,
        target_size_px: int,
    ) -> TransformedQuery:
        raise NotImplementedError("HomographyViewTransformer.transform() — da implementare")

    @property
    def name(self) -> str:
        return "homography_inverse"
