"""Read-only, GT-after-forward diagnostics for the existing soft Q.

These functions never construct a retrieval query or alter a prediction.  The
area-grid and Q roundtrips are *label-transfer probes*: they use GT deliberately
and therefore are not model outputs or deployable upper bounds.
"""

from __future__ import annotations

import numpy as np


CLASSES = ("Background", "Intact", "Damaged", "Destroyed")
SIZE_BINS = ((1, 16), (17, 64), (65, 256), (257, None))
BUILDING_PROBABILITY_BINS = ((0.0, 0.1), (0.1, 0.3), (0.3, 0.5),
                             (0.5, 0.7), (0.7, 1.000001))


def area_class_fractions(target: np.ndarray, grid_shape: tuple[int, int]):
    """Replicate torch `interpolate(one_hot, mode='area')` for integer ratios.

    Returns [h,w,4] fractions and [h,w] valid mass.  The integer-ratio guard
    prevents a quietly different resizing rule from entering this audit.
    """
    gt = np.asarray(target)
    if gt.ndim != 2 or len(grid_shape) != 2:
        raise ValueError("Expected 2-D GT and grid shape")
    height, width = gt.shape
    gh, gw = (int(v) for v in grid_shape)
    if gh <= 0 or gw <= 0 or height % gh or width % gw:
        raise ValueError("Exact area-grid probe requires integer downsample ratios")
    bh, bw = height // gh, width // gw
    valid = (gt >= 0) & (gt < 4)
    reshape = lambda x: x.reshape(gh, bh, gw, bw).mean(axis=(1, 3))
    fractions = np.stack([reshape(valid & (gt == c)) for c in range(4)], axis=-1)
    return fractions, reshape(valid)


def q_label_roundtrip(fractions: np.ndarray, valid_mass: np.ndarray,
                      q: np.ndarray, ids: np.ndarray):
    """GT feature-grid fractions -> soft Q regions -> feature-grid fractions.

    The same raw Q footprint is used on both sides, matching `region_targets`.
    Unlabelled grid cells are excluded when constructing region distributions.
    """
    f = np.asarray(fractions, dtype=np.float64)
    v = np.asarray(valid_mass, dtype=np.float64)
    a = np.asarray(q, dtype=np.float64)
    r = np.asarray(ids)
    if f.ndim != 3 or f.shape[2] != 4 or v.shape != f.shape[:2]:
        raise ValueError("Expected [h,w,4] fractions and [h,w] valid mass")
    if a.ndim != 2 or a.shape != r.shape or a.shape[1] != f.shape[0] * f.shape[1]:
        raise ValueError("Q and region IDs must be [routes, h*w]")
    if (not np.issubdtype(r.dtype, np.integer) or (r < 0).any() or
            not np.isfinite(a).all() or (a < -1e-6).any() or
            not np.allclose(a.sum(axis=0), 1, atol=2e-5, rtol=0)):
        raise ValueError("Invalid Q partition or region IDs")
    if (not np.isfinite(f).all() or not np.isfinite(v).all() or
            (f < -1e-6).any() or (v < -1e-6).any() or
            (f.sum(axis=-1) - v > 2e-5).any()):
        raise ValueError("Invalid label fractions")
    region_count = int(r.max()) + 1
    rid = r.ravel()
    weighted = a * v.ravel()[None]
    region_mass = np.bincount(rid, weights=weighted.ravel(),
                              minlength=region_count)
    class_mass = np.stack([
        np.bincount(rid, weights=(a * f[..., c].ravel()[None]).ravel(),
                    minlength=region_count)
        for c in range(4)
    ], axis=-1)
    region_label = class_mass / np.maximum(region_mass[:, None], 1e-12)
    rebuilt = (region_label[r] * a[..., None]).sum(axis=0)
    return rebuilt.reshape(f.shape), region_label, region_mass


def nearest_grid_pixels(values: np.ndarray, output_shape: tuple[int, int]):
    """Deterministic pixel association for the oracle probe, not real decoding."""
    array = np.asarray(values)
    if array.ndim < 2:
        raise ValueError("Grid values need spatial dimensions")
    h, w = array.shape[:2]
    oh, ow = (int(v) for v in output_shape)
    if min(h, w, oh, ow) <= 0:
        raise ValueError("Invalid output shape")
    yy = np.minimum(np.arange(oh) * h // oh, h - 1)
    xx = np.minimum(np.arange(ow) * w // ow, w - 1)
    return array[np.ix_(yy, xx)]


def probe_prediction(target: np.ndarray, q: np.ndarray, ids: np.ndarray,
                     grid_shape: tuple[int, int]):
    """Return GT-derived grid and Q roundtrip predictions for diagnostics only."""
    gt = np.asarray(target)
    fractions, valid = area_class_fractions(gt, grid_shape)
    rebuilt, labels, mass = q_label_roundtrip(fractions, valid, q, ids)
    grid_pred = nearest_grid_pixels(fractions.argmax(-1), gt.shape)
    q_pred = nearest_grid_pixels(rebuilt.argmax(-1), gt.shape)
    return grid_pred, q_pred, labels, mass


def confusion(target: np.ndarray, prediction: np.ndarray):
    gt = np.asarray(target)
    pred = np.asarray(prediction)
    if gt.shape != pred.shape:
        raise ValueError("GT and prediction shape differ")
    valid = (gt >= 0) & (gt < 4)
    if ((pred[valid] < 0) | (pred[valid] >= 4)).any():
        raise ValueError("Prediction outside four classes")
    return np.bincount((4 * gt[valid].astype(np.int64) +
                        pred[valid].astype(np.int64)), minlength=16).reshape(4, 4)


def prediction_confusion_by_building_probability(
    target: np.ndarray, prediction: np.ndarray, building_grid: np.ndarray,
) -> np.ndarray:
    """Five fixed-B-bin 4x4 confusions from one already-frozen prediction.

    B is nearest-expanded only for this error attribution. Summing all bins
    preserves the full valid-GT denominator, even for low-B missed buildings.
    """
    gt, pred = np.asarray(target), np.asarray(prediction)
    b = np.asarray(building_grid, dtype=np.float64)
    if gt.ndim != 2 or pred.shape != gt.shape or b.ndim != 2:
        raise ValueError("Expected matching GT/prediction and 2-D B grid")
    if not np.isfinite(b).all() or (b < -1e-6).any() or (b > 1 + 1e-6).any():
        raise ValueError("Invalid predicted building probability")
    expanded = nearest_grid_pixels(b, gt.shape)
    valid = (gt >= 0) & (gt < 4)
    if ((pred[valid] < 0) | (pred[valid] >= 4)).any():
        raise ValueError("Prediction outside four classes")
    output = []
    for lower, upper in BUILDING_PROBABILITY_BINS:
        selected = valid & (expanded >= lower) & (expanded < upper)
        output.append(np.bincount(
            (4 * gt[selected].astype(np.int64) + pred[selected].astype(np.int64)),
            minlength=16,
        ).reshape(4, 4))
    result = np.stack(output)
    if not np.array_equal(result.sum(axis=0), confusion(gt, pred)):
        raise ValueError("Building-probability bins lost valid GT pixels")
    return result


def region_mixing_mass(region_label: np.ndarray, labelled_mass: np.ndarray) -> dict:
    """Separate foreground/background dilution from I/D/X mixing.

    Fixed thresholds are descriptive strata, not building instance labels or
    learned acceptance rules. Report continuous mean purity as well.
    """
    labels = np.asarray(region_label, dtype=np.float64)
    mass = np.asarray(labelled_mass, dtype=np.float64)
    if labels.ndim != 2 or labels.shape[1] != 4 or mass.shape != labels.shape[:1]:
        raise ValueError("Expected [regions,4] labels and region masses")
    if (not np.isfinite(labels).all() or not np.isfinite(mass).all()
            or (labels < -1e-6).any() or (mass < -1e-6).any()):
        raise ValueError("Invalid region labels or masses")
    valid = mass > 1e-8
    fg = labels[:, 1:].sum(axis=1)
    purity = labels[:, 1:].max(axis=1) / np.maximum(fg, 1e-12)
    bg_mixed = valid & (fg >= 0.2) & (fg < 0.8)
    class_mixed = valid & (fg >= 0.8) & (purity < 0.8)
    denominator = float(mass[valid].sum())
    fg_mass = float(mass[valid & (fg >= 0.8)].sum())
    return dict(
        labelled_mass=denominator,
        labelled_regions=int(valid.sum()),
        background_foreground_mixed_mass=float(mass[bg_mixed].sum()),
        foreground_class_mixed_mass=float(mass[class_mixed].sum()),
        high_foreground_mass=fg_mass,
        foreground_purity_weighted_sum=float((mass[valid & (fg >= 0.8)]
                                                * purity[valid & (fg >= 0.8)]).sum()),
        threshold_rule="foreground 0.2..0.8 = background/foreground mixed; foreground >=0.8 and class purity <0.8 = foreground-class mixed",
    )


def building_probability_stats(building_grid: np.ndarray,
                               fractions: np.ndarray,
                               thresholds=(0.1, 0.3, 0.5, 0.7)):
    """Assess exact A1 building probability against post-forward GT fractions.

    `building_grid` must be captured from the input to `front.partition`, not
    reconstructed from the upsampled location logits.  This is a grid-cell
    diagnostic; mixed cells contribute fractionally to each GT class.
    """
    b = np.asarray(building_grid, dtype=np.float64)
    f = np.asarray(fractions, dtype=np.float64)
    if b.ndim != 2 or f.shape != (*b.shape, 4):
        raise ValueError("B must align with [h,w,4] area fractions")
    if (not np.isfinite(b).all() or (b < -1e-6).any() or (b > 1 + 1e-6).any()
            or not np.isfinite(f).all() or (f < -1e-6).any()):
        raise ValueError("Invalid building probabilities or GT fractions")
    out = {}
    for c, name in enumerate(CLASSES):
        weights = f[..., c]
        mass = float(weights.sum())
        out[name] = {
            "gt_grid_mass": mass,
            "mean_predicted_building_probability": (
                float((weights * b).sum() / mass) if mass else None),
            "low_probability_mass_fraction": {
                str(t): (float((weights * (b < t)).sum() / mass) if mass else None)
                for t in thresholds
            },
        }
    return out


def component_proxy_stats(target: np.ndarray, grid_pred: np.ndarray,
                          q_pred: np.ndarray):
    """Per-class 4-connected raster components; NOT building instances."""
    from scipy import ndimage

    gt = np.asarray(target)
    a, b = np.asarray(grid_pred), np.asarray(q_pred)
    if gt.ndim != 2 or a.shape != gt.shape or b.shape != gt.shape:
        raise ValueError("Expected matching 2-D label maps")
    cross = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    out = {}
    for c, name in enumerate(CLASSES[1:], start=1):
        labelled, count = ndimage.label(gt == c, structure=cross)
        area = np.bincount(labelled.ravel(), minlength=count + 1)
        area[0] = 0
        rows = []
        for lower, upper in SIZE_BINS:
            selected = (area >= lower) & ((area <= upper) if upper else True)
            selected[0] = False
            pixel_mask = selected[labelled] & (gt == c)
            pixels = int(pixel_mask.sum())
            rows.append(dict(
                pixel_area_range=f"{lower}-{upper or 'inf'}",
                connected_raster_components=int(selected.sum()),
                gt_pixels=pixels,
                grid_oracle_correct=int(((a == c) & pixel_mask).sum()),
                q_oracle_correct=int(((b == c) & pixel_mask).sum()),
            ))
        out[name] = rows
    return out


def component_building_stats(target: np.ndarray, building_grid: np.ndarray,
                             threshold: float = 0.5):
    """A1 B coverage on per-class connected raster components by pixel area.

    Nearest expansion reflects the source feature cell at each pixel.  A
    connected class raster is only a size proxy, never a BRIGHT building ID.
    """
    from scipy import ndimage

    gt, b = np.asarray(target), np.asarray(building_grid, dtype=np.float64)
    if gt.ndim != 2 or b.ndim != 2 or not np.isfinite(b).all():
        raise ValueError("Expected 2-D GT and finite 2-D building grid")
    if (b < -1e-6).any() or (b > 1 + 1e-6).any() or not 0 <= threshold <= 1:
        raise ValueError("Building probabilities and threshold must be 0..1")
    expanded = nearest_grid_pixels(b, gt.shape)
    cross = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    out = {}
    for c, name in enumerate(CLASSES[1:], start=1):
        labels, count = ndimage.label(gt == c, structure=cross)
        area = np.bincount(labels.ravel(), minlength=count + 1)
        summed = np.bincount(labels.ravel(), weights=expanded.ravel(),
                             minlength=count + 1)
        covered = np.bincount(labels.ravel(), weights=(expanded >= threshold).ravel(),
                              minlength=count + 1)
        area[0], summed[0], covered[0] = 0, 0.0, 0.0
        rows = []
        for lower, upper in SIZE_BINS:
            selected = (area >= lower) & ((area <= upper) if upper else True)
            selected[0] = False
            selected_area = int(area[selected].sum())
            rows.append(dict(
                pixel_area_range=f"{lower}-{upper or 'inf'}",
                connected_raster_components=int(selected.sum()),
                gt_pixels=selected_area,
                predicted_building_probability_sum=float(summed[selected].sum()),
                gt_pixels_at_or_above_threshold=int(covered[selected].sum()),
                components_majority_below_threshold=int(
                    (covered[selected] < 0.5 * area[selected]).sum()),
            ))
        out[name] = rows
    return out
