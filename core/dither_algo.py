"""
Dithering algorithms - map pixels to a palette with error diffusion or ordered dither.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from core.pixel_snapper_algo import bgr_to_features

DITHER_NONE = "None"
DITHER_FLOYD_STEINBERG = "Floyd-Steinberg"
DITHER_ATKINSON = "Atkinson"
DITHER_ORDERED = "Ordered (Bayer)"

VALID_DITHER_METHODS = (
    DITHER_NONE,
    DITHER_FLOYD_STEINBERG,
    DITHER_ATKINSON,
    DITHER_ORDERED,
)

_BAYER_4X4 = np.array(
    [
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5],
    ],
    dtype=np.float32,
) / 16.0 - 0.5


def _nearest_index(feat: np.ndarray, palette_features: np.ndarray) -> int:
    diff = palette_features - feat
    return int(np.argmin(np.sum(diff * diff, axis=1)))


def _add_error(
    working: np.ndarray,
    opaque_mask: np.ndarray,
    y: int,
    x: int,
    error: np.ndarray,
    factor: float,
) -> None:
    h, w = opaque_mask.shape
    if y < 0 or y >= h or x < 0 or x >= w:
        return
    if not opaque_mask[y, x]:
        return
    working[y, x] += error * factor


def _error_diffuse(
    feature_grid: np.ndarray,
    palette_features: np.ndarray,
    palette_bgr: np.ndarray,
    opaque_mask: np.ndarray,
    method: str,
) -> np.ndarray:
    h, w, _ = feature_grid.shape
    working = feature_grid.astype(np.float64).copy()
    result_bgr = np.zeros((h, w, 3), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            if not opaque_mask[y, x]:
                continue

            old = working[y, x].copy()
            idx = _nearest_index(old, palette_features)
            chosen = palette_features[idx]
            error = old - chosen
            working[y, x] = chosen
            result_bgr[y, x] = palette_bgr[idx]

            if method == DITHER_FLOYD_STEINBERG:
                _add_error(working, opaque_mask, y, x + 1, error, 7.0 / 16.0)
                _add_error(working, opaque_mask, y + 1, x - 1, error, 3.0 / 16.0)
                _add_error(working, opaque_mask, y + 1, x, error, 5.0 / 16.0)
                _add_error(working, opaque_mask, y + 1, x + 1, error, 1.0 / 16.0)
            elif method == DITHER_ATKINSON:
                for dy, dx, factor in (
                    (0, 1, 1.0 / 8.0),
                    (0, 2, 1.0 / 8.0),
                    (1, -1, 1.0 / 8.0),
                    (1, 0, 1.0 / 8.0),
                    (1, 1, 1.0 / 8.0),
                    (2, 0, 1.0 / 8.0),
                ):
                    _add_error(working, opaque_mask, y + dy, x + dx, error, factor)

    return result_bgr


def _vectorized_map(
    feature_grid: np.ndarray,
    palette_features: np.ndarray,
    palette_bgr: np.ndarray,
    opaque_mask: np.ndarray,
    bias_grid: np.ndarray | None = None,
) -> np.ndarray:
    h, w, _ = feature_grid.shape
    k = len(palette_bgr)

    biased = feature_grid.astype(np.float32)
    if bias_grid is not None:
        biased = biased + bias_grid[:, :, np.newaxis]

    diff = biased[:, :, np.newaxis, :] - palette_features[np.newaxis, np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=3)
    indices = np.argmin(dist_sq, axis=2)
    mapped = palette_bgr[indices]

    result_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    result_bgr[opaque_mask] = mapped[opaque_mask]
    return result_bgr


def _ordered_bias_grid(h: int, w: int, palette_features: np.ndarray) -> np.ndarray:
    """Build per-pixel bias for ordered dither from palette feature spread."""
    if len(palette_features) < 2:
        return np.zeros((h, w), dtype=np.float32)

    spread = float(np.mean(np.std(palette_features, axis=0)))
    spread = max(spread, 1.0)
    ys = np.arange(h)[:, np.newaxis] % 4
    xs = np.arange(w)[np.newaxis, :] % 4
    return _BAYER_4X4[ys, xs] * spread


def dither_to_palette(
    image: np.ndarray,
    palette_bgr: np.ndarray,
    method: str,
    space: str,
    weights: Tuple[float, float, float],
) -> np.ndarray:
    """
    Map image pixels to palette colors using the selected dithering method.

    Distance is computed in the same weighted color space used for quantization.
    Only opaque pixels are dithered; alpha is preserved.
    """
    if image is None or len(palette_bgr) == 0:
        return image.copy() if image is not None else None

    has_alpha = image.shape[2] == 4
    bgr = image[:, :, :3]
    alpha = image[:, :, 3] if has_alpha else None
    h, w = bgr.shape[:2]

    opaque_mask = np.ones((h, w), dtype=bool) if alpha is None else alpha > 0
    if not np.any(opaque_mask):
        return image.copy()

    flat_bgr = bgr.reshape(-1, 3)
    flat_features = bgr_to_features(flat_bgr, space, weights)
    feature_grid = flat_features.reshape(h, w, -1)
    palette_features = bgr_to_features(palette_bgr, space, weights)

    method = method if method in VALID_DITHER_METHODS else DITHER_NONE

    if method in (DITHER_FLOYD_STEINBERG, DITHER_ATKINSON):
        result_bgr = _error_diffuse(
            feature_grid, palette_features, palette_bgr, opaque_mask, method
        )
    elif method == DITHER_ORDERED:
        bias = _ordered_bias_grid(h, w, palette_features)
        result_bgr = _vectorized_map(
            feature_grid, palette_features, palette_bgr, opaque_mask, bias
        )
    else:
        result_bgr = _vectorized_map(
            feature_grid, palette_features, palette_bgr, opaque_mask, None
        )

    result_bgr[~opaque_mask] = bgr[~opaque_mask]

    if has_alpha:
        result = image.copy()
        result[:, :, :3] = result_bgr
        return result
    return np.dstack((result_bgr, np.full((h, w), 255, dtype=np.uint8)))
