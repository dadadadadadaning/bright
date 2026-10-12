"""CPU-only, prediction-after-the-fact statistics. No model or GT routing here."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


FOREGROUND_BINS = ("fg_lt020", "fg_020_080", "fg_ge080", "unlabelled")
CLASS_PURITY_BINS = (
    "no_foreground",
    "class_lt080",
    "class_080_095",
    "class_ge095",
    "unlabelled",
)
ARMS = ("j2", "j3")


def classify_regions(distribution, labelled_mass):
    """Partition GT composition for reporting; NEVER use these bins to retrieve."""
    d = np.asarray(distribution, dtype=np.float64)
    mass = np.asarray(labelled_mass, dtype=np.float64)
    if d.ndim != 2 or d.shape[1] != 4 or mass.shape != (len(d),):
        raise ValueError(
            "Expected [regions,4] distributions and matching labelled mass"
        )
    if (
        not np.isfinite(d).all()
        or not np.isfinite(mass).all()
        or (d < -1e-6).any()
        or (mass < 0).any()
    ):
        raise ValueError("Invalid region distributions or mass")
    valid = mass > 1e-6
    if not np.allclose(d[valid].sum(1), 1, atol=2e-5, rtol=0):
        raise ValueError("Nonempty region distributions must sum to one")
    fg = d[:, 1:].sum(1)
    cond = np.zeros(len(d), dtype=np.float64)
    has_fg = valid & (fg > 1e-6)
    cond[has_fg] = d[has_fg, 1:].max(1) / fg[has_fg]
    fbin = np.full(len(d), 3, dtype=np.int8)
    fbin[valid & (fg < 0.2)] = 0
    fbin[valid & (fg >= 0.2) & (fg < 0.8)] = 1
    fbin[valid & (fg >= 0.8)] = 2
    pbin = np.full(len(d), 4, dtype=np.int8)
    pbin[valid & ~has_fg] = 0
    pbin[has_fg & (cond < 0.8)] = 1
    pbin[has_fg & (cond >= 0.8) & (cond < 0.95)] = 2
    pbin[has_fg & (cond >= 0.95)] = 3
    return dict(
        foreground_bin=fbin,
        class_purity_bin=pbin,
        foreground_fraction=fg,
        class_purity=cond,
        valid=valid,
    )


def route_maps(q, ids, region_bins, grid_shape, output_shape):
    """Map each output pixel to its highest-Q source region (nearest grid resize).

    These maps define diagnostic cohorts only. Real Memory uses all soft routes,
    and its probability upsampling is bilinear rather than this hard assignment.
    """
    q = np.asarray(q)
    ids = np.asarray(ids)
    if (
        q.ndim != 2
        or ids.shape != q.shape
        or len(grid_shape) != 2
        or len(output_shape) != 2
    ):
        raise ValueError(
            "Expected matching [routes,cells] Q/IDs and two spatial shapes"
        )
    h, w = (int(v) for v in grid_shape)
    out_h, out_w = (int(v) for v in output_shape)
    if min(h, w, out_h, out_w) <= 0 or q.shape[1] != h * w or not np.isfinite(q).all():
        raise ValueError("Invalid Q or spatial shapes")
    if (q < -1e-6).any() or not np.allclose(q.sum(0), 1, atol=2e-5, rtol=0):
        raise ValueError("Q must be a nonnegative partition of unity")
    if not np.issubdtype(ids.dtype, np.integer) or (ids < 0).any():
        raise ValueError("Region IDs must be nonnegative integers")
    route = q.argmax(0)
    region_id = ids[route, np.arange(h * w)].reshape(h, w)
    yy = np.minimum((np.arange(out_h) * h // out_h), h - 1)
    xx = np.minimum((np.arange(out_w) * w // out_w), w - 1)
    output = {}
    for name, values in region_bins.items():
        values = np.asarray(values)
        if values.ndim != 1 or ids.max() >= len(values):
            raise ValueError("Region bins do not cover all Q IDs")
        output[name] = values[region_id][np.ix_(yy, xx)]
    return output


def soft_route_exposure(q, ids, region_flags, grid_shape):
    """Return feature-grid Q mass assigned to a marked set of regions."""
    q, ids = np.asarray(q, dtype=np.float64), np.asarray(ids)
    flags = np.asarray(region_flags, dtype=bool)
    h, w = (int(v) for v in grid_shape)
    if q.ndim != 2 or ids.shape != q.shape or q.shape[1] != h * w:
        raise ValueError("Q/IDs do not match feature-grid shape")
    if flags.ndim != 1 or (ids < 0).any() or ids.max() >= len(flags):
        raise ValueError("Flags do not cover Q region IDs")
    if (
        not np.isfinite(q).all()
        or (q < -1e-6).any()
        or not np.allclose(q.sum(0), 1, atol=2e-5)
    ):
        raise ValueError("Invalid Q weights")
    return (q * flags[ids]).sum(0).reshape(h, w)


def joint_counts(target, old, new, bin_map=None, number_of_bins=None):
    """Integer GT x old x new counts, optionally grouped by a diagnostic bin."""
    y, a, b = (np.asarray(value) for value in (target, old, new))
    if y.shape != a.shape or y.shape != b.shape or y.ndim != 2:
        raise ValueError("Expected matching two-dimensional GT and prediction maps")
    valid = (y >= 0) & (y < 4)
    if ((a[valid] < 0) | (a[valid] >= 4) | (b[valid] < 0) | (b[valid] >= 4)).any():
        raise ValueError("Prediction outside four classes")
    code = (
        y[valid].astype(np.int64) * 16
        + a[valid].astype(np.int64) * 4
        + b[valid].astype(np.int64)
    )
    if bin_map is None:
        return np.bincount(code, minlength=64).reshape(4, 4, 4)
    bins = np.asarray(bin_map)
    if bins.shape != y.shape or number_of_bins is None or number_of_bins < 1:
        raise ValueError("Bin map must match GT and declare its number of bins")
    selected = bins[valid].astype(np.int64)
    if ((selected < 0) | (selected >= number_of_bins)).any():
        raise ValueError("Bin value outside declared range")
    return np.bincount(selected * 64 + code, minlength=number_of_bins * 64).reshape(
        number_of_bins, 4, 4, 4
    )


def focus(joint, include_joint=False):
    """Use fixed J1 denominators for rescue and for originally-correct harm."""
    t = np.asarray(joint, dtype=np.int64)
    if t.shape != (4, 4, 4) or (t < 0).any():
        raise ValueError("Expected nonnegative [GT,J1,candidate] joint counts")
    d_wrong = int(t[2].sum() - t[2, 2].sum())
    d_correct = int(t[2, 2].sum())
    i_correct = int(t[1, 1].sum())
    rescue = int(t[2, :, 2].sum() - t[2, 2, 2])
    harm = int(t[2, 2].sum() - t[2, 2, 2])
    i_to_d = int(t[1, 1, 2])
    fp_old = int(t[[0, 1, 3], 2, :].sum())
    fp_new = int(t[[0, 1, 3], :, 2].sum())
    out = dict(
        valid_pixels=int(t.sum()),
        damaged_j1_wrong=d_wrong,
        damaged_rescue=rescue,
        damaged_rescue_rate=rescue / d_wrong if d_wrong else None,
        damaged_j1_correct=d_correct,
        damaged_harm=harm,
        damaged_harm_rate=harm / d_correct if d_correct else None,
        intact_j1_correct=i_correct,
        intact_correct_to_damaged=i_to_d,
        intact_correct_to_damaged_rate=i_to_d / i_correct if i_correct else None,
        intact_gt_old_not_d_to_new_d=int(t[1, [0, 1, 3], 2].sum()),
        damaged_false_positive_old=fp_old,
        damaged_false_positive_new=fp_new,
        damaged_false_positive_delta=fp_new - fp_old,
    )
    if include_joint:
        out["gt_j1_candidate"] = t.tolist()
    return out


class StratifiedAccumulator:
    def __init__(self, arms=ARMS):
        self.arms = tuple(arms)
        self.groups = {
            "all": ("all",),
            "foreground_fraction": FOREGROUND_BINS,
            "foreground_class_purity": CLASS_PURITY_BINS,
            "joint_foreground_and_purity": tuple(
                f"{f}/{p}" for f in FOREGROUND_BINS for p in CLASS_PURITY_BINS
            ),
        }
        self.total = self._empty()
        self.events = defaultdict(self._empty)
        self.images = 0

    def _empty(self):
        return {
            group: {
                name: {arm: np.zeros((4, 4, 4), dtype=np.int64) for arm in self.arms}
                for name in names
            }
            for group, names in self.groups.items()
        }

    def add(self, event, target, j1, candidates, maps):
        if set(candidates) != set(self.arms):
            raise ValueError("Candidate arms do not match accumulator")
        fg = np.asarray(maps["foreground_bin"])
        purity = np.asarray(maps["class_purity_bin"])
        if fg.shape != np.asarray(target).shape or purity.shape != fg.shape:
            raise ValueError("Cohort maps do not match GT")
        if (
            (fg < 0)
            | (fg >= len(FOREGROUND_BINS))
            | (purity < 0)
            | (purity >= len(CLASS_PURITY_BINS))
        ).any():
            raise ValueError("Invalid diagnostic bin")
        joint_bin = fg.astype(np.int64) * len(CLASS_PURITY_BINS) + purity
        per_image = {}
        for arm, candidate in candidates.items():
            arrays = {
                "all": joint_counts(target, j1, candidate)[None],
                "foreground_fraction": joint_counts(
                    target, j1, candidate, fg, len(FOREGROUND_BINS)
                ),
                "foreground_class_purity": joint_counts(
                    target, j1, candidate, purity, len(CLASS_PURITY_BINS)
                ),
                "joint_foreground_and_purity": joint_counts(
                    target,
                    j1,
                    candidate,
                    joint_bin,
                    len(FOREGROUND_BINS) * len(CLASS_PURITY_BINS),
                ),
            }
            for group, array in arrays.items():
                for name, one in zip(self.groups[group], array, strict=True):
                    self.total[group][name][arm] += one
                    self.events[event][group][name][arm] += one
            per_image[arm] = focus(arrays["all"][0])
        self.images += 1
        return per_image

    def _report(self, storage, include_joint):
        return {
            group: {
                name: {
                    arm: focus(storage[group][name][arm], include_joint)
                    for arm in self.arms
                }
                for name in names
            }
            for group, names in self.groups.items()
        }

    def report(self):
        return dict(
            images=self.images,
            overall=self._report(self.total, True),
            by_event={
                event: self._report(value, False)
                for event, value in sorted(self.events.items())
            },
        )


class ExposureAccumulator:
    """Q-soft mixed-region exposure in fixed J1 and candidate change cohorts."""

    COHORTS = (
        "intact_j1_correct",
        "damaged_j1_wrong",
        "intact_correct_to_damaged",
        "damaged_rescued",
        "damaged_harmed",
    )
    EXPOSURES = ("background_foreground_mixed", "foreground_class_mixed")

    def __init__(self, arms=ARMS):
        self.arms = tuple(arms)
        self.overall = self._empty()
        self.events = defaultdict(self._empty)

    def _empty(self):
        return {
            arm: {
                cohort: {name: dict(sum=0.0, count=0) for name in self.EXPOSURES}
                for cohort in self.COHORTS
            }
            for arm in self.arms
        }

    def add(self, event, target, j1, candidates, exposures):
        y, old = np.asarray(target), np.asarray(j1)
        if (
            y.shape != old.shape
            or set(candidates) != set(self.arms)
            or set(exposures) != set(self.EXPOSURES)
        ):
            raise ValueError("Exposure cohort input mismatch")
        checked = {}
        for name, field in exposures.items():
            field = np.asarray(field, dtype=np.float64)
            if (
                field.shape != y.shape
                or not np.isfinite(field).all()
                or (field < -2e-5).any()
                or (field > 1 + 2e-5).any()
            ):
                raise ValueError("Invalid soft Q exposure field")
            checked[name] = field
        for arm, candidate in candidates.items():
            new = np.asarray(candidate)
            if new.shape != y.shape:
                raise ValueError("Exposure candidate shape mismatch")
            cohorts = {
                "intact_j1_correct": (y == 1) & (old == 1),
                "damaged_j1_wrong": (y == 2) & (old != 2),
                "intact_correct_to_damaged": (y == 1) & (old == 1) & (new == 2),
                "damaged_rescued": (y == 2) & (old != 2) & (new == 2),
                "damaged_harmed": (y == 2) & (old == 2) & (new != 2),
            }
            for cohort, mask in cohorts.items():
                count = int(mask.sum())
                for name, field in checked.items():
                    amount = float(field[mask].sum())
                    for dest in (self.overall, self.events[event]):
                        dest[arm][cohort][name]["sum"] += amount
                        dest[arm][cohort][name]["count"] += count

    @staticmethod
    def _report(values):
        return {
            arm: {
                cohort: {
                    name: dict(
                        **row, mean=row["sum"] / row["count"] if row["count"] else None
                    )
                    for name, row in fields.items()
                }
                for cohort, fields in cohorts.items()
            }
            for arm, cohorts in values.items()
        }

    def report(self):
        return dict(
            overall=self._report(self.overall),
            by_event={
                event: self._report(value)
                for event, value in sorted(self.events.items())
            },
        )
