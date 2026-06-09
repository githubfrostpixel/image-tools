"""
Pixel Snapper algorithm - grid detection and resampling ported from spritefusion-pixel-snapper.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import cv2
import numpy as np

COLOR_SPACE_RGB = "RGB"
COLOR_SPACE_LAB = "LAB"
COLOR_SPACE_HSV = "HSV"
VALID_COLOR_SPACES = (COLOR_SPACE_RGB, COLOR_SPACE_LAB, COLOR_SPACE_HSV)

# Scales hue onto the unit circle with comparable magnitude to S/V channels.
HUE_CIRCLE_RADIUS = 90.0


@dataclass
class SnapperConfig:
    k_colors: int = 16
    k_seed: int = 42
    max_kmeans_iterations: int = 15
    peak_threshold_multiplier: float = 0.2
    peak_distance_filter: int = 4
    walker_search_window_ratio: float = 0.35
    walker_min_search_window: float = 2.0
    walker_strength_threshold: float = 0.5
    min_cuts_per_axis: int = 4
    fallback_target_segments: int = 64
    max_step_ratio: float = 1.8
    pixel_size_override: Optional[float] = None
    quantize_before_detect: bool = True
    enable_snapping: bool = True
    color_space: str = COLOR_SPACE_RGB
    channel_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)


def _normalize_weights(weights: Tuple[float, float, float]) -> Tuple[float, float, float]:
    total = weights[0] + weights[1] + weights[2]
    if total <= 0:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return (weights[0] / total, weights[1] / total, weights[2] / total)


def _resolve_color_space(space: str) -> str:
    upper = (space or COLOR_SPACE_RGB).upper()
    if upper in VALID_COLOR_SPACES:
        return upper
    return COLOR_SPACE_RGB


def bgr_pixels_to_space(bgr_pixels: np.ndarray, space: str) -> np.ndarray:
    """Convert Nx3 BGR pixels to the target color space."""
    space = _resolve_color_space(space)
    if space == COLOR_SPACE_RGB:
        return bgr_pixels.astype(np.float32)

    reshaped = bgr_pixels.astype(np.uint8).reshape(-1, 1, 3)
    if space == COLOR_SPACE_LAB:
        converted = cv2.cvtColor(reshaped, cv2.COLOR_BGR2LAB)
    else:
        converted = cv2.cvtColor(reshaped, cv2.COLOR_BGR2HSV)
    return converted.reshape(-1, 3).astype(np.float32)


def space_pixel_to_bgr(pixel: np.ndarray, space: str) -> np.ndarray:
    """Convert a single pixel from native space back to BGR."""
    space = _resolve_color_space(space)
    if space == COLOR_SPACE_RGB:
        return np.clip(pixel, 0, 255).astype(np.uint8)

    if space == COLOR_SPACE_LAB:
        arr = np.clip(pixel, 0, 255).astype(np.uint8).reshape(1, 1, 3)
        return cv2.cvtColor(arr, cv2.COLOR_LAB2BGR)[0, 0]

    hsv = pixel.astype(np.float32).copy()
    hsv[0] = np.clip(hsv[0], 0, 179)
    hsv[1:] = np.clip(hsv[1:], 0, 255)
    arr = hsv.astype(np.uint8).reshape(1, 1, 3)
    return cv2.cvtColor(arr, cv2.COLOR_HSV2BGR)[0, 0]


def palette_native_to_bgr(palette: np.ndarray, space: str) -> np.ndarray:
    return np.array([space_pixel_to_bgr(p, space) for p in palette], dtype=np.uint8)


def bgr_to_features(
    bgr_pixels: np.ndarray, space: str, weights: Tuple[float, float, float]
) -> np.ndarray:
    """Convert BGR pixels to weighted feature vectors for palette distance."""
    native_pixels = bgr_pixels_to_space(bgr_pixels, space)
    return _build_cluster_features(native_pixels, space, weights)


def _build_cluster_features(
    native_pixels: np.ndarray, space: str, weights: Tuple[float, float, float]
) -> np.ndarray:
    w0, w1, w2 = _normalize_weights(weights)
    space = _resolve_color_space(space)

    if space == COLOR_SPACE_HSV:
        hue = native_pixels[:, 0]
        theta = hue * (2.0 * np.pi / 180.0)
        cos_h = np.cos(theta) * HUE_CIRCLE_RADIUS * w0
        sin_h = np.sin(theta) * HUE_CIRCLE_RADIUS * w0
        sat = native_pixels[:, 1] * w1
        val = native_pixels[:, 2] * w2
        return np.column_stack([cos_h, sin_h, sat, val]).astype(np.float32)

    weighted = native_pixels.astype(np.float32).copy()
    weighted[:, 0] *= w0
    weighted[:, 1] *= w1
    weighted[:, 2] *= w2
    return weighted


def _mean_native_pixel(members: np.ndarray, space: str) -> np.ndarray:
    space = _resolve_color_space(space)
    if space == COLOR_SPACE_HSV:
        theta = members[:, 0] * (2.0 * np.pi / 180.0)
        mean_sin = float(np.mean(np.sin(theta)))
        mean_cos = float(np.mean(np.cos(theta)))
        if abs(mean_sin) < 1e-6 and abs(mean_cos) < 1e-6:
            mean_hue = float(np.mean(members[:, 0]))
        else:
            mean_hue = (np.arctan2(mean_sin, mean_cos) / (2.0 * np.pi)) * 180.0
            if mean_hue < 0:
                mean_hue += 180.0
        return np.array(
            [mean_hue, float(np.mean(members[:, 1])), float(np.mean(members[:, 2]))],
            dtype=np.float32,
        )
    return np.mean(members, axis=0).astype(np.float32)


def _cluster_representatives(
    native_pixels: np.ndarray, labels: np.ndarray, k: int, space: str
) -> np.ndarray:
    flat_labels = labels.reshape(-1)
    representatives = np.zeros((k, 3), dtype=np.float32)
    has_members = np.zeros(k, dtype=bool)
    fallback = _mean_native_pixel(native_pixels, space)

    for cluster_id in range(k):
        members = native_pixels[flat_labels == cluster_id]
        if len(members) == 0:
            representatives[cluster_id] = fallback
            continue
        has_members[cluster_id] = True
        representatives[cluster_id] = _mean_native_pixel(members, space)

    if not np.any(has_members):
        representatives[:] = fallback
    return representatives


def _cluster_palette_from_opaque(
    opaque_bgr: np.ndarray, config: SnapperConfig, space: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Run K-means on opaque pixels; return palette_bgr and per-pixel cluster labels."""
    native_pixels = bgr_pixels_to_space(opaque_bgr, space)
    features = _build_cluster_features(native_pixels, space, config.channel_weights)

    k = min(config.k_colors, len(native_pixels))
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        config.max_kmeans_iterations,
        0.01,
    )
    _, labels, _ = cv2.kmeans(
        features,
        k,
        None,
        criteria,
        attempts=3,
        flags=cv2.KMEANS_PP_CENTERS,
    )

    palette_native = _cluster_representatives(native_pixels, labels, k, space)
    palette_bgr = palette_native_to_bgr(palette_native, space)
    return palette_bgr, labels.reshape(-1).astype(int)


def compute_palette(image: np.ndarray, config: SnapperConfig) -> np.ndarray:
    """Compute K-means palette as BGR colors for opaque pixels."""
    if config.k_colors <= 0 or image is None:
        return np.zeros((0, 3), dtype=np.uint8)

    bgr = image[:, :, :3]
    alpha = image[:, :, 3] if image.shape[2] == 4 else None
    space = _resolve_color_space(config.color_space)
    opaque_mask = np.ones(bgr.shape[:2], dtype=bool) if alpha is None else alpha > 0
    opaque_bgr = bgr[opaque_mask]

    if len(opaque_bgr) == 0:
        return np.zeros((0, 3), dtype=np.uint8)

    palette_bgr, _ = _cluster_palette_from_opaque(opaque_bgr, config, space)
    return palette_bgr


def quantize(image: np.ndarray, config: SnapperConfig) -> np.ndarray:
    """Reduce opaque pixels to k_colors using K-means in the selected color space."""
    if config.k_colors <= 0:
        return image.copy()

    has_alpha = image.shape[2] == 4
    bgr = image[:, :, :3]
    alpha = image[:, :, 3] if has_alpha else None
    space = _resolve_color_space(config.color_space)

    opaque_mask = np.ones(bgr.shape[:2], dtype=bool) if alpha is None else alpha > 0
    opaque_bgr = bgr[opaque_mask]

    if len(opaque_bgr) == 0:
        return image.copy()

    palette_bgr, flat_labels = _cluster_palette_from_opaque(opaque_bgr, config, space)
    quantized_bgr = palette_bgr[flat_labels]

    result_bgr = bgr.copy()
    result_bgr[opaque_mask] = quantized_bgr

    if has_alpha:
        result = image.copy()
        result[:, :, :3] = result_bgr
        return result
    return np.dstack((result_bgr, np.full(bgr.shape[:2], 255, dtype=np.uint8)))


def quantize_rgb(image: np.ndarray, config: SnapperConfig) -> np.ndarray:
    """Backward-compatible alias for RGB quantization."""
    rgb_config = replace(config, color_space=COLOR_SPACE_RGB)
    return quantize(image, rgb_config)


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    bgr = image[:, :, :3].astype(np.float64)
    alpha = image[:, :, 3]
    gray = 0.114 * bgr[:, :, 0] + 0.587 * bgr[:, :, 1] + 0.299 * bgr[:, :, 2]
    gray[alpha == 0] = 0.0
    return gray


def compute_profiles(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build edge-strength profiles along X and Y axes."""
    h, w = image.shape[:2]
    if w < 3 or h < 3:
        raise ValueError("Image too small (minimum 3x3)")

    gray = _to_grayscale(image)
    col_proj = np.zeros(w, dtype=np.float64)
    row_proj = np.zeros(h, dtype=np.float64)

    for x in range(1, w - 1):
        col_proj[x] = np.sum(np.abs(gray[:, x + 1] - gray[:, x - 1]))

    for y in range(1, h - 1):
        row_proj[y] = np.sum(np.abs(gray[y + 1, :] - gray[y - 1, :]))

    return col_proj, row_proj


def estimate_step_size(profile: np.ndarray, config: SnapperConfig) -> Optional[float]:
    if len(profile) == 0:
        return None

    max_val = float(np.max(profile))
    if max_val == 0.0:
        return None

    threshold = max_val * config.peak_threshold_multiplier
    peaks: List[int] = []
    for i in range(1, len(profile) - 1):
        if (
            profile[i] > threshold
            and profile[i] > profile[i - 1]
            and profile[i] > profile[i + 1]
        ):
            peaks.append(i)

    if len(peaks) < 2:
        return None

    clean_peaks = [peaks[0]]
    for peak in peaks[1:]:
        if peak - clean_peaks[-1] > config.peak_distance_filter - 1:
            clean_peaks.append(peak)

    if len(clean_peaks) < 2:
        return None

    diffs = [float(clean_peaks[i + 1] - clean_peaks[i]) for i in range(len(clean_peaks) - 1)]
    diffs.sort()
    return diffs[len(diffs) // 2]


def resolve_step_sizes(
    step_x_opt: Optional[float],
    step_y_opt: Optional[float],
    width: int,
    height: int,
    config: SnapperConfig,
) -> Tuple[float, float]:
    if config.pixel_size_override is not None:
        px = config.pixel_size_override
        return px, px

    if step_x_opt is not None and step_y_opt is not None:
        sx, sy = step_x_opt, step_y_opt
        ratio = sx / sy if sx > sy else sy / sx
        if ratio > config.max_step_ratio:
            smaller = min(sx, sy)
            return smaller, smaller
        avg = (sx + sy) / 2.0
        return avg, avg

    if step_x_opt is not None:
        return step_x_opt, step_x_opt

    if step_y_opt is not None:
        return step_y_opt, step_y_opt

    fallback_step = max((min(width, height) / config.fallback_target_segments), 1.0)
    return fallback_step, fallback_step


def walk(profile: np.ndarray, step_size: float, limit: int, config: SnapperConfig) -> List[int]:
    if len(profile) == 0:
        raise ValueError("Cannot walk on empty profile")

    cuts = [0]
    current_pos = 0.0
    search_window = max(
        step_size * config.walker_search_window_ratio,
        config.walker_min_search_window,
    )
    mean_val = float(np.sum(profile)) / len(profile)

    while current_pos < limit:
        target = current_pos + step_size
        if target >= limit:
            cuts.append(limit)
            break

        start_search = max(int(target - search_window), int(current_pos + 1))
        end_search = min(int(target + search_window), limit)

        if end_search <= start_search:
            current_pos = target
            continue

        segment = profile[start_search:end_search]
        max_idx = start_search + int(np.argmax(segment))
        max_val = float(profile[max_idx])

        if max_val > mean_val * config.walker_strength_threshold:
            cuts.append(max_idx)
            current_pos = float(max_idx)
        else:
            cuts.append(int(target))
            current_pos = target

    return cuts


def sanitize_cuts(cuts: List[int], limit: int) -> List[int]:
    if limit == 0:
        return [0]

    result = list(cuts)
    has_zero = False
    has_limit = False

    for i, value in enumerate(result):
        if value == 0:
            has_zero = True
        if value >= limit:
            result[i] = limit
        if result[i] == limit:
            has_limit = True

    if not has_zero:
        result.append(0)
    if not has_limit:
        result.append(limit)

    result = sorted(set(result))
    return result


def snap_uniform_cuts(
    profile: np.ndarray,
    limit: int,
    target_step: float,
    config: SnapperConfig,
    min_required: int,
) -> List[int]:
    if limit == 0:
        return [0]
    if limit == 1:
        return [0, 1]

    if target_step > 0 and np.isfinite(target_step):
        desired_cells = int(round(limit / target_step))
    else:
        desired_cells = 0

    desired_cells = max(desired_cells, max(min_required - 1, 1))
    desired_cells = min(desired_cells, limit)

    cell_width = limit / desired_cells
    search_window = max(
        cell_width * config.walker_search_window_ratio,
        config.walker_min_search_window,
    )
    mean_val = float(np.sum(profile)) / len(profile) if len(profile) > 0 else 0.0

    cuts = [0]
    for idx in range(1, desired_cells):
        target = cell_width * idx
        prev = cuts[-1]
        if prev + 1 >= limit:
            break

        start = max(int(np.floor(target - search_window)), prev + 1, 0)
        end = min(int(np.ceil(target + search_window)), limit - 1)
        if end < start:
            start = prev + 1
            end = start

        profile_limit = max(len(profile) - 1, 0)
        best_idx = min(start, profile_limit)
        best_val = -1.0
        for i in range(start, min(end, profile_limit) + 1):
            value = float(profile[i]) if i < len(profile) else 0.0
            if value > best_val:
                best_val = value
                best_idx = i

        strength_threshold = mean_val * config.walker_strength_threshold
        if best_val < strength_threshold:
            fallback_idx = int(round(target))
            if fallback_idx <= prev:
                fallback_idx = prev + 1
            if fallback_idx >= limit:
                fallback_idx = max(limit - 1, prev + 1)
            best_idx = fallback_idx

        cuts.append(best_idx)

    if cuts[-1] != limit:
        cuts.append(limit)

    return sanitize_cuts(cuts, limit)


def stabilize_cuts(
    profile: np.ndarray,
    cuts: List[int],
    limit: int,
    sibling_cuts: List[int],
    sibling_limit: int,
    config: SnapperConfig,
) -> List[int]:
    if limit == 0:
        return [0]

    cuts = sanitize_cuts(cuts, limit)
    min_required = min(max(config.min_cuts_per_axis, 2), limit + 1)
    axis_cells = max(len(cuts) - 1, 0)
    sibling_cells = max(len(sibling_cuts) - 1, 0)
    sibling_has_grid = (
        sibling_limit > 0
        and sibling_cells >= max(min_required - 1, 0)
        and sibling_cells > 0
    )

    steps_skewed = False
    if sibling_has_grid and axis_cells > 0:
        axis_step = limit / axis_cells
        sibling_step = sibling_limit / sibling_cells
        step_ratio = axis_step / sibling_step
        steps_skewed = (
            step_ratio > config.max_step_ratio
            or step_ratio < 1.0 / config.max_step_ratio
        )

    has_enough = len(cuts) >= min_required
    if has_enough and not steps_skewed:
        return cuts

    if sibling_has_grid:
        target_step = sibling_limit / sibling_cells
    elif config.fallback_target_segments > 1:
        target_step = limit / config.fallback_target_segments
    elif axis_cells > 0:
        target_step = limit / axis_cells
    else:
        target_step = float(limit)

    if not np.isfinite(target_step) or target_step <= 0:
        target_step = 1.0

    return snap_uniform_cuts(profile, limit, target_step, config, min_required)


def stabilize_both_axes(
    profile_x: np.ndarray,
    profile_y: np.ndarray,
    raw_col_cuts: List[int],
    raw_row_cuts: List[int],
    width: int,
    height: int,
    config: SnapperConfig,
) -> Tuple[List[int], List[int]]:
    col_cuts_pass1 = stabilize_cuts(
        profile_x, raw_col_cuts, width, raw_row_cuts, height, config
    )
    row_cuts_pass1 = stabilize_cuts(
        profile_y, raw_row_cuts, height, raw_col_cuts, width, config
    )

    col_cells = max(len(col_cuts_pass1) - 1, 1)
    row_cells = max(len(row_cuts_pass1) - 1, 1)
    col_step = width / col_cells
    row_step = height / row_cells
    step_ratio = col_step / row_step if col_step > row_step else row_step / col_step

    if step_ratio > config.max_step_ratio:
        target_step = min(col_step, row_step)

        if col_step > target_step * 1.2:
            final_col_cuts = snap_uniform_cuts(
                profile_x,
                width,
                target_step,
                config,
                config.min_cuts_per_axis,
            )
        else:
            final_col_cuts = col_cuts_pass1

        if row_step > target_step * 1.2:
            final_row_cuts = snap_uniform_cuts(
                profile_y,
                height,
                target_step,
                config,
                config.min_cuts_per_axis,
            )
        else:
            final_row_cuts = row_cuts_pass1

        return final_col_cuts, final_row_cuts

    return col_cuts_pass1, row_cuts_pass1


def _mode_rgba(cell: np.ndarray) -> np.ndarray:
    if cell.size == 0:
        return np.array([0, 0, 0, 0], dtype=np.uint8)

    flat = cell.reshape(-1, 4)
    tuples = [tuple(pixel) for pixel in flat]
    counter = Counter(tuples)
    best = counter.most_common(1)[0][0]
    return np.array(best, dtype=np.uint8)


def resample(image: np.ndarray, col_cuts: List[int], row_cuts: List[int]) -> np.ndarray:
    if len(col_cuts) < 2 or len(row_cuts) < 2:
        raise ValueError("Insufficient grid cuts for resampling")

    out_w = len(col_cuts) - 1
    out_h = len(row_cuts) - 1
    result = np.zeros((out_h, out_w, 4), dtype=np.uint8)

    for y_i in range(out_h):
        ys, ye = row_cuts[y_i], row_cuts[y_i + 1]
        for x_i in range(out_w):
            xs, xe = col_cuts[x_i], col_cuts[x_i + 1]
            if xe <= xs or ye <= ys:
                continue
            cell = image[ys:ye, xs:xe]
            result[y_i, x_i] = _mode_rgba(cell)

    return result


def snap_in_place(
    original: np.ndarray,
    working: np.ndarray,
    col_cuts: List[int],
    row_cuts: List[int],
) -> np.ndarray:
    result = original.copy()
    for y_i in range(len(row_cuts) - 1):
        ys, ye = row_cuts[y_i], row_cuts[y_i + 1]
        for x_i in range(len(col_cuts) - 1):
            xs, xe = col_cuts[x_i], col_cuts[x_i + 1]
            if xe <= xs or ye <= ys:
                continue
            cell = working[ys:ye, xs:xe]
            mode_color = _mode_rgba(cell)
            result[ys:ye, xs:xe] = mode_color
    return result


def process_pixel_snap(
    image: np.ndarray,
    config: SnapperConfig,
    downsample: bool = True,
    output_scale: int = 1,
) -> np.ndarray:
    """Run the full pixel snapper pipeline on a BGRA image."""
    if image is None:
        return None

    h, w = image.shape[:2]
    if w < 3 or h < 3:
        return image.copy()

    if config.pixel_size_override is not None:
        px = config.pixel_size_override
        max_px = min(w, h) / 2.0
        if not np.isfinite(px) or px < 1.0 or px > max_px:
            config = replace(config, pixel_size_override=None)

    working = (
        quantize(image, config)
        if config.quantize_before_detect
        else image.copy()
    )

    if not config.enable_snapping:
        return working

    profile_x, profile_y = compute_profiles(working)
    step_x_opt = estimate_step_size(profile_x, config)
    step_y_opt = estimate_step_size(profile_y, config)
    step_x, step_y = resolve_step_sizes(step_x_opt, step_y_opt, w, h, config)

    raw_col_cuts = walk(profile_x, step_x, w, config)
    raw_row_cuts = walk(profile_y, step_y, h, config)
    col_cuts, row_cuts = stabilize_both_axes(
        profile_x, profile_y, raw_col_cuts, raw_row_cuts, w, h, config
    )

    if downsample:
        output = resample(working, col_cuts, row_cuts)
        if output_scale > 1:
            new_w = output.shape[1] * output_scale
            new_h = output.shape[0] * output_scale
            output = cv2.resize(output, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        return output

    return snap_in_place(image, working, col_cuts, row_cuts)
