"""Reconstruct recorded sample order and report replay drift without hiding it."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from numbers import Real
from pathlib import Path

from inspect_failure import sha256
from probe import candidate_from_journal


METRICS = (
    "original_loss", "nll", "lovasz", "grad_norm_preclip",
    "reader_grad_abs_sum", "parameter_update_l2", "parameter_l2",
)


def reconstruct_order(torch, manifest, plan, journal_path):
    """Return a verified first-epoch candidate, full order, and immutable-source rows."""
    journal_path = Path(journal_path)
    before = sha256(journal_path)
    candidate = candidate_from_journal(torch, manifest, plan, journal_path)
    if candidate["journal_sha256"] != before:
        raise ValueError("Journal changed during candidate reconstruction")
    seed = manifest["seed"]
    loader = torch.utils.data.DataLoader(
        list(manifest["train_names"]), batch_size=1, shuffle=True,
        num_workers=0, drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )
    order = [str(batch[0]) for batch in loader]
    payload = journal_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != before:
        raise ValueError("Journal changed before replay rows were captured")
    rows_by_step = {}
    for index, line in enumerate(payload.decode("utf-8").splitlines(keepends=True)):
        if not line.endswith("\n"):
            raise ValueError("Journal has an unterminated row")
        row = json.loads(line)
        step = index + 1
        if (not isinstance(row, dict) or type(row.get("step")) is not int
                or row["step"] != step or index >= len(order)
                or row.get("image") != order[index]
                or type(row.get("epoch")) is not int or row["epoch"] != 0
                or type(row.get("epoch_seed")) is not int
                or row["epoch_seed"] != seed):
            raise ValueError("Recorded sample identity differs at step %s" % step)
        rows_by_step[step] = row
    last = candidate["last_complete_step"]
    if (len(rows_by_step) != last or candidate["verified_prefix_count"] != last
            or candidate["candidate_step"] != last + 1
            or candidate["image"] != order[last] or sha256(journal_path) != before):
        raise ValueError("Journal or reconstructed sample order changed")
    return candidate, order, rows_by_step


def _finite_number(value, description):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(description + " must be a finite numeric scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(description + " must be finite")
    return result


def _counts():
    return dict(numeric_comparisons=0, exact_matches=0, within_tolerance=0,
                outside_tolerance=0, paired_none=0,
                first_difference=None, first_outside_tolerance=None,
                max_absolute_difference=None)


class TrajectoryComparison:
    """Fail on identity/state errors; accumulate finite metric drift as evidence."""

    def __init__(self, arms, atol=1e-5, rtol=1e-4):
        self.arms = tuple(arms)
        if (not self.arms or any(not isinstance(a, str) or not a for a in self.arms)
                or len(set(self.arms)) != len(self.arms)):
            raise ValueError("Arm identities must be nonempty unique strings")
        self.atol = _finite_number(atol, "atol")
        self.rtol = _finite_number(rtol, "rtol")
        if self.atol < 0 or self.rtol < 0:
            raise ValueError("Comparison tolerances must be nonnegative")
        self._summary = dict(
            arms=list(self.arms), atol=self.atol, rtol=self.rtol,
            steps_compared=0, first_step=None, last_step=None,
            per_field={field: _counts() for field in METRICS}, **_counts(),
        )

    def observe(self, step, image, rows, original_row):
        """Compare original update rows with one complete replay update."""
        if type(step) is not int or step <= 0:
            raise ValueError("Replay step must be a positive integer")
        if not isinstance(image, str) or not image:
            raise ValueError("Replay image must be a nonempty string")
        last = self._summary["last_step"]
        if last is not None and step != last + 1:
            raise ValueError("Replay observations must be consecutive and unique")
        if (not isinstance(original_row, dict)
                or type(original_row.get("step")) is not int
                or original_row["step"] != step
                or original_row.get("image") != image):
            raise ValueError("Original/replay step or image identity differs")
        original_arms = original_row.get("arms")
        expected_arms = set(self.arms)
        if (not isinstance(rows, dict) or set(rows) != expected_arms
                or not isinstance(original_arms, dict)
                or set(original_arms) != expected_arms):
            raise ValueError("Original/replay arm identities differ")

        # Fully validate this step before changing accumulated evidence.
        comparisons = []
        for arm in self.arms:
            actual, expected = rows[arm], original_arms[arm]
            if not isinstance(actual, dict) or not isinstance(expected, dict):
                raise ValueError("Arm update rows must be objects: " + arm)
            for label, row in (("replay", actual), ("original", expected)):
                if type(row.get("optimizer_step")) is not int or row["optimizer_step"] != step:
                    raise ValueError("%s optimizer counter differs: %s step %s" % (label, arm, step))
            actual_lr = _finite_number(actual.get("lr"), "Replay learning rate")
            expected_lr = _finite_number(expected.get("lr"), "Original learning rate")
            if actual_lr != expected_lr:
                raise ValueError("Replay learning rate differs: %s step %s" % (arm, step))
            for field in METRICS:
                if field not in actual or field not in expected:
                    raise ValueError("Missing trajectory metric: %s.%s" % (arm, field))
                left, right = actual[field], expected[field]
                if left is None and right is None:
                    comparisons.append((field, None))
                    continue
                if left is None or right is None:
                    raise ValueError("Unpaired null trajectory metric: %s.%s" % (arm, field))
                left = _finite_number(left, "Replay " + arm + "." + field)
                right = _finite_number(right, "Original " + arm + "." + field)
                difference = abs(left - right)
                tolerance = self.atol + self.rtol * abs(right)
                if not math.isfinite(difference) or not math.isfinite(tolerance):
                    raise ValueError("Trajectory comparison overflow: %s.%s" % (arm, field))
                comparisons.append((field, dict(
                    step=step, image=image, arm=arm, field=field,
                    actual=left, expected=right, abs_diff=difference,
                    tolerance=tolerance,
                )))
        for field, event in comparisons:
            self._record(self._summary, event)
            self._record(self._summary["per_field"][field], event)
        if self._summary["first_step"] is None:
            self._summary["first_step"] = step
        self._summary["last_step"] = step
        self._summary["steps_compared"] += 1

    @staticmethod
    def _record(summary, event):
        if event is None:
            summary["paired_none"] += 1
            return
        summary["numeric_comparisons"] += 1
        if event["abs_diff"] == 0:
            summary["exact_matches"] += 1
        elif summary["first_difference"] is None:
            summary["first_difference"] = event
        if event["abs_diff"] <= event["tolerance"]:
            summary["within_tolerance"] += 1
        else:
            summary["outside_tolerance"] += 1
            if summary["first_outside_tolerance"] is None:
                summary["first_outside_tolerance"] = event
        largest = summary["max_absolute_difference"]
        if event["abs_diff"] > 0 and (largest is None or event["abs_diff"] > largest["abs_diff"]):
            summary["max_absolute_difference"] = event

    def summary(self):
        result = copy.deepcopy(self._summary)
        result["all_exact"] = result["exact_matches"] == result["numeric_comparisons"]
        result["all_within_tolerance"] = result["outside_tolerance"] == 0
        return result
