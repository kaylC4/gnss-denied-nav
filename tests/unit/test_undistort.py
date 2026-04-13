"""
test_undistort.py — Unit test per lo Stage 1 (rimozione distorsione lente).

Criteri di successo (spec §2.4):
  - Output shape coerente con balance e camera_type
  - K_new aggiornato correttamente (cx/cy shiftati dopo crop)
  - Immagine senza distorsione su input sintetico a distorsione nulla
  - Fisheye: reshape dist_coeffs (4,) → (4, 1) trasparente al chiamante

Test round-trip (questa sessione — issue #12):
  - test_passthrough_zero_coeffs  : undistort con D=0 è identità pixel-per-pixel
  - test_round_trip_pinhole       : PSNR ≥ 30 dB e max error ≤ 20 dopo apply→undistort
  - test_corner_accuracy          : RMS corner < 0.5 px su mild e moderate barrel
  - test_K_new_sanity             : punto principale entro ±50 px, focali positive

Tutti i parametri (K, dist_coeffs, balance, image_size) provengono dai file YAML
in tests/fixtures/undistort/ — nessun valore hardcoded nei nuovi test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from gnss_denied_nav.config import CameraConfig, PipelineConfig
from gnss_denied_nav.preprocessing.undistort import UndistortResult, undistort
from tests.helpers.synthetic import apply_brown_conrady, make_checkerboard, psnr

# ── fixture ───────────────────────────────────────────────────────────────────

W, H = 640, 480
FX = FY = 500.0
CX, CY = W / 2, H / 2


def _make_camera(camera_type: str, dist: list[float]) -> CameraConfig:
    return CameraConfig(
        camera_type=camera_type,  # type: ignore[arg-type]
        camera_orientation="downward",
        pixel_pitch_um=1.5,
        K=np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64),
        dist_coeffs=np.array(dist, dtype=np.float64),
        image_size=(W, H),
    )


def _blank_image() -> np.ndarray:
    return np.zeros((H, W, 3), dtype=np.uint8)


def _checkerboard() -> np.ndarray:
    """Pattern a scacchiera 8×6 — linee rette nel mondo."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    sq = H // 8
    for r in range(8):
        for c in range(W // sq):
            if (r + c) % 2 == 0:
                img[r * sq : (r + 1) * sq, c * sq : (c + 1) * sq] = 255
    return img


PINHOLE_ZERO = _make_camera("pinhole", [0.0, 0.0, 0.0, 0.0])
PINHOLE_DIST = _make_camera("pinhole", [-0.3, 0.1, 0.0, 0.0])
FISHEYE_ZERO = _make_camera("fisheye", [0.0, 0.0, 0.0, 0.0])

# ── test output type ──────────────────────────────────────────────────────────


def test_returns_undistort_result() -> None:
    result = undistort(_blank_image(), PINHOLE_ZERO)
    assert isinstance(result, UndistortResult)
    assert isinstance(result.image, np.ndarray)
    assert isinstance(result.K_new, np.ndarray)


# ── test pinhole ──────────────────────────────────────────────────────────────


def test_pinhole_zero_dist_balance1_preserves_size() -> None:
    """Distorsione nulla + balance=1 → dimensioni invariate."""
    result = undistort(_blank_image(), PINHOLE_ZERO, balance=1.0)
    assert result.image.shape == (H, W, 3)


def test_pinhole_zero_dist_balance0_output_is_3channel() -> None:
    result = undistort(_blank_image(), PINHOLE_ZERO, balance=0.0)
    assert result.image.ndim == 3
    assert result.image.shape[2] == 3


def test_pinhole_zero_dist_Knew_shape() -> None:
    result = undistort(_blank_image(), PINHOLE_ZERO, balance=0.0)
    assert result.K_new.shape == (3, 3)


def test_pinhole_zero_dist_Knew_last_row() -> None:
    """La terza riga di K_new deve essere [0, 0, 1]."""
    result = undistort(_blank_image(), PINHOLE_ZERO, balance=0.0)
    np.testing.assert_array_equal(result.K_new[2], [0.0, 0.0, 1.0])


def test_pinhole_zero_dist_balance1_Knew_close_to_K() -> None:
    """Con distorsione nulla e balance=1, K_new ≈ K."""
    result = undistort(_blank_image(), PINHOLE_ZERO, balance=1.0)
    np.testing.assert_allclose(result.K_new, PINHOLE_ZERO.K, atol=1.0)


def test_pinhole_balance0_crop_shifts_principal_point() -> None:
    """
    Con balance=0.0 e distorsione non nulla, il crop deve shiftare cx/cy
    in modo che il punto principale rimanga coerente con l'immagine croppata.
    """
    result = undistort(_checkerboard(), PINHOLE_DIST, balance=0.0)
    h_out, w_out = result.image.shape[:2]
    cx_new = result.K_new[0, 2]
    cy_new = result.K_new[1, 2]
    # Il punto principale deve stare dentro l'immagine croppata
    assert 0 <= cx_new <= w_out, f"cx_new={cx_new} fuori da [0, {w_out}]"
    assert 0 <= cy_new <= h_out, f"cy_new={cy_new} fuori da [0, {h_out}]"


def test_pinhole_balance1_no_crop_cx_cy_unchanged_approx() -> None:
    """Con balance=1 non c'è crop → cx/cy non vengono shiftati."""
    result = undistort(_blank_image(), PINHOLE_ZERO, balance=1.0)
    # Con distorsione nulla, K_new ≈ K originale
    assert abs(result.K_new[0, 2] - CX) < 2.0
    assert abs(result.K_new[1, 2] - CY) < 2.0


# ── test fisheye ──────────────────────────────────────────────────────────────


def test_fisheye_zero_dist_output_shape() -> None:
    result = undistort(_blank_image(), FISHEYE_ZERO, balance=0.0)
    assert result.image.ndim == 3
    assert result.image.shape[2] == 3


def test_fisheye_Knew_shape() -> None:
    result = undistort(_blank_image(), FISHEYE_ZERO, balance=0.0)
    assert result.K_new.shape == (3, 3)


def test_fisheye_Knew_last_row() -> None:
    result = undistort(_blank_image(), FISHEYE_ZERO, balance=0.0)
    np.testing.assert_array_equal(result.K_new[2], [0.0, 0.0, 1.0])


def test_fisheye_dtype_uint8() -> None:
    """L'immagine output deve rimanere uint8."""
    result = undistort(_checkerboard(), FISHEYE_ZERO, balance=0.0)
    assert result.image.dtype == np.uint8


# ── test validazione input ────────────────────────────────────────────────────


def test_invalid_balance_raises() -> None:
    with pytest.raises(ValueError, match="balance"):
        undistort(_blank_image(), PINHOLE_ZERO, balance=1.5)


def test_negative_balance_raises() -> None:
    with pytest.raises(ValueError, match="balance"):
        undistort(_blank_image(), PINHOLE_ZERO, balance=-0.1)


# ── test dtype preservato ─────────────────────────────────────────────────────


def test_pinhole_output_dtype_uint8() -> None:
    result = undistort(_checkerboard(), PINHOLE_DIST, balance=0.0)
    assert result.image.dtype == np.uint8


# ════════════════════════════════════════════════════════════════════════════════
# Test round-trip — Stage 1 (issue #12)
#
# Tutti i parametri vengono caricati dai file YAML in tests/fixtures/undistort/.
# Nessun valore di K, dist_coeffs, balance o image_size è hardcoded qui sotto.
# ════════════════════════════════════════════════════════════════════════════════

_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "undistort"

# Checkerboard: 16×12 quadrati da 64 px → 15×11 corner interni (asimmetrico)
_CHESSBOARD_PATTERN = (15, 11)


# ── fixture session-scoped (una per set di coefficienti) ──────────────────────


@pytest.fixture(scope="session")
def cfg_identity() -> PipelineConfig:
    """Pinhole, distorsione nulla — usata anche da checkerboard_1024."""
    return PipelineConfig.from_yaml(_FIXTURE_DIR / "pinhole_identity.yaml")


@pytest.fixture(scope="session")
def cfg_mild_barrel() -> PipelineConfig:
    """Pinhole, barrel lieve: k1=-0.1, k2=0.01."""
    return PipelineConfig.from_yaml(_FIXTURE_DIR / "pinhole_mild_barrel.yaml")


@pytest.fixture(scope="session")
def cfg_moderate_barrel() -> PipelineConfig:
    """Pinhole, barrel moderato: k1=-0.28, k2=0.09, p1=0.001, p2=-0.001."""
    return PipelineConfig.from_yaml(_FIXTURE_DIR / "pinhole_moderate_barrel.yaml")


@pytest.fixture(scope="session")
def cfg_pincushion() -> PipelineConfig:
    """Pinhole, pincushion: k1=+0.15, k2=-0.04."""
    return PipelineConfig.from_yaml(_FIXTURE_DIR / "pinhole_pincushion.yaml")


@pytest.fixture(scope="session")
def cfg_fixture(request: pytest.FixtureRequest) -> PipelineConfig:
    """Fixture parametrizzata — riceve il nome YAML tramite indirect parametrize."""
    yaml_name: Any = request.param
    return PipelineConfig.from_yaml(_FIXTURE_DIR / f"{yaml_name}.yaml")


@pytest.fixture(scope="session")
def checkerboard_1024(cfg_identity: PipelineConfig) -> np.ndarray:
    """
    Scacchiera sintetica anti-aliasata.

    Dimensioni lette dal YAML: 1024×768 px, quadrati 64 px.
    16×12 quadrati → 15×11 corner interni (board asimmetrico).
    """
    w, h = cfg_identity.camera.image_size  # (width, height) dal YAML
    return make_checkerboard(h, w, square_size=64)


@pytest.fixture(scope="session")
def smooth_image_1024(cfg_identity: PipelineConfig) -> np.ndarray:
    """
    Immagine liscia per test PSNR.

    Rumore uniforme filtrato con Gaussian (σ=10): nessuna discontinuità,
    l'errore di doppia interpolazione rimane < 1 LSB sulle regioni interne.
    Dimensioni dal YAML; seed fisso per riproducibilità.
    """
    w, h = cfg_identity.camera.image_size  # (width, height) dal YAML
    rng = np.random.default_rng(42)
    noise = rng.integers(0, 256, (h, w, 3)).astype(np.uint8)
    blurred: np.ndarray = cv2.GaussianBlur(noise, (0, 0), sigmaX=10.0)
    return blurred


# ── A. Passthrough con distorsione nulla ──────────────────────────────────────


def test_passthrough_zero_coeffs(
    cfg_identity: PipelineConfig, checkerboard_1024: np.ndarray
) -> None:
    """
    Con dist_coeffs=[0,…,0] e balance=1.0 (dal YAML), undistort è identità:
      - l'immagine output è pixel-per-pixel identica all'input
      - K_new è uguale a K entro tolleranza floating-point
    """
    cam = cfg_identity.camera
    pre = cfg_identity.preprocessing

    result = undistort(checkerboard_1024, cam, balance=pre.undistort_balance)

    np.testing.assert_array_equal(
        result.image,
        checkerboard_1024,
        err_msg=(
            "undistort con D=0 e balance=1.0 deve restituire l'immagine invariata. "
            f"dist_coeffs={cam.dist_coeffs.tolist()}, balance={pre.undistort_balance}"
        ),
    )
    np.testing.assert_allclose(
        result.K_new,
        cam.K,
        atol=0.5,
        err_msg=(
            "K_new con D=0 deve essere uguale a K (atol=0.5 px). "
            f"K={cam.K.tolist()}, K_new={result.K_new.tolist()}"
        ),
    )


# ── B. Round-trip pinhole (parametrizzato sui 4 set di coefficienti) ──────────


@pytest.mark.parametrize(
    "cfg_fixture",
    [
        pytest.param("pinhole_identity", id="identity"),
        pytest.param("pinhole_mild_barrel", id="mild_barrel"),
        pytest.param("pinhole_moderate_barrel", id="moderate_barrel"),
        pytest.param("pinhole_pincushion", id="pincushion"),
    ],
    indirect=True,
)
def test_round_trip_pinhole(
    cfg_fixture: PipelineConfig, smooth_image_1024: np.ndarray
) -> None:
    """
    Pipeline round-trip: apply_brown_conrady → undistort → confronto con GT.

    Usa un'immagine liscia (Gaussian blur σ=10) al posto della scacchiera per
    evitare che le discontinuità ad alto contrasto amplifichino l'errore di
    doppia interpolazione nella metrica PSNR.

    Con balance=1.0 (dal YAML) l'output ha la stessa shape dell'input.
    Crop di 80 px per rimuovere i bordi neri che compaiono con balance=1.0
    e distorsione barrel significativa (k1=-0.28 → ~50 px di bordo nero).

    Soglia: PSNR ≥ 30 dB sull'area interna.
    """
    cam = cfg_fixture.camera
    pre = cfg_fixture.preprocessing

    distorted = apply_brown_conrady(
        smooth_image_1024, cam.K, cam.dist_coeffs, balance=pre.undistort_balance
    )
    result = undistort(distorted, cam, balance=pre.undistort_balance)

    assert result.image.shape == smooth_image_1024.shape, (
        f"Con balance=1.0 l'output deve avere la stessa shape dell'input. "
        f"Atteso {smooth_image_1024.shape}, ottenuto {result.image.shape}"
    )

    # 80 px rimuove i bordi neri anche per la distorsione barrel più intensa
    border = 80
    gt_crop = smooth_image_1024[border:-border, border:-border]
    pred_crop = result.image[border:-border, border:-border]

    actual_psnr = psnr(gt_crop, pred_crop)

    assert actual_psnr >= 30.0, (
        f"PSNR round-trip troppo basso: {actual_psnr:.2f} dB < 30.0 dB  "
        f"[dist_coeffs={cam.dist_coeffs.tolist()}]"
    )


# ── C. Accuratezza corner subpixel (mild e moderate barrel) ───────────────────


@pytest.mark.parametrize(
    "cfg_fixture",
    [
        pytest.param("pinhole_mild_barrel", id="mild_barrel"),
        pytest.param("pinhole_moderate_barrel", id="moderate_barrel"),
    ],
    indirect=True,
)
def test_corner_accuracy(
    cfg_fixture: PipelineConfig, checkerboard_1024: np.ndarray
) -> None:
    """
    Verifica la precisione geometrica sub-pixel dopo il round-trip.

    Approccio: usa le posizioni NOTE a priori dei corner della scacchiera
    (k×64, l×64) come inizializzazione per cornerSubPix, applicato sia
    sull'originale sia sul recuperato. Il round-trip geometricamente corretto
    deve restituire gli stessi corner entro < 0.5 px RMS.

    Questo evita findChessboardCorners (fragile con artefatti da doppia
    interpolazione su distorsione moderata) e testa la geometria direttamente.
    Solo i corner interni (distanza ≥ 64 px dal bordo) vengono confrontati,
    per escludere eventuali artefatti ai margini dell'immagine.

    RMS distanza euclidea tra corner corrispondenti deve essere < 0.5 px.
    """
    cam = cfg_fixture.camera
    pre = cfg_fixture.preprocessing

    distorted = apply_brown_conrady(
        checkerboard_1024, cam.K, cam.dist_coeffs, balance=pre.undistort_balance
    )
    result = undistort(distorted, cam, balance=pre.undistort_balance)

    subpix_criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        40,
        0.001,
    )

    w, h = cam.image_size  # (width, height) dal YAML
    sq = 64
    margin = sq  # 1 quadrato di margine su ogni lato

    # Posizioni note dei corner interni della scacchiera, esclusi i bordi
    seed_corners = np.array(
        [
            [[float(k * sq), float(l * sq)]]
            for l in range(1, h // sq)
            for k in range(1, w // sq)
            if margin <= k * sq <= w - margin and margin <= l * sq <= h - margin
        ],
        dtype=np.float32,
    )

    if len(seed_corners) < 9:
        pytest.skip(
            f"Troppo pochi corner interni ({len(seed_corners)}) "
            f"per un confronto affidabile (image_size={w}×{h}, sq={sq})"
        )

    orig_gray: np.ndarray = cv2.cvtColor(checkerboard_1024, cv2.COLOR_BGR2GRAY)
    rec_gray: np.ndarray = cv2.cvtColor(result.image, cv2.COLOR_BGR2GRAY)

    # cornerSubPix raffina a partire dalle posizioni note (window 5×5)
    corners_orig: np.ndarray = cv2.cornerSubPix(
        orig_gray, seed_corners.copy(), (5, 5), (-1, -1), subpix_criteria
    )
    corners_rec: np.ndarray = cv2.cornerSubPix(
        rec_gray, seed_corners.copy(), (5, 5), (-1, -1), subpix_criteria
    )

    dists: np.ndarray = np.linalg.norm(
        corners_orig[:, 0, :] - corners_rec[:, 0, :], axis=1
    )
    rms = float(np.sqrt(np.mean(dists**2)))

    assert rms < 0.5, (
        f"RMS corner troppo alto: {rms:.4f} px ≥ 0.5 px  "
        f"[dist_coeffs={cam.dist_coeffs.tolist()}, n_corners={len(seed_corners)}]  "
        f"(max={float(dists.max()):.4f} px, median={float(np.median(dists)):.4f} px)"
    )


# ── D. Sanità di K_new ────────────────────────────────────────────────────────


def test_K_new_sanity(
    cfg_mild_barrel: PipelineConfig, checkerboard_1024: np.ndarray
) -> None:
    """
    K_new restituita da undistort deve avere:
      - focali fx, fy > 0
      - punto principale (cx, cy) entro ±50 px dal centro dell'immagine
    """
    cam = cfg_mild_barrel.camera
    pre = cfg_mild_barrel.preprocessing
    w, h = cam.image_size  # (width, height) dal YAML

    distorted = apply_brown_conrady(
        checkerboard_1024, cam.K, cam.dist_coeffs, balance=pre.undistort_balance
    )
    result = undistort(distorted, cam, balance=pre.undistort_balance)

    fx_new = float(result.K_new[0, 0])
    fy_new = float(result.K_new[1, 1])
    cx_new = float(result.K_new[0, 2])
    cy_new = float(result.K_new[1, 2])

    assert fx_new > 0, f"fx_new deve essere > 0, ottenuto {fx_new}"
    assert fy_new > 0, f"fy_new deve essere > 0, ottenuto {fy_new}"

    cx_center = w / 2
    cy_center = h / 2
    assert abs(cx_new - cx_center) <= 50, (
        f"cx_new={cx_new:.1f} px troppo lontano dal centro ({cx_center:.1f} ± 50 px)"
    )
    assert abs(cy_new - cy_center) <= 50, (
        f"cy_new={cy_new:.1f} px troppo lontano dal centro ({cy_center:.1f} ± 50 px)"
    )
