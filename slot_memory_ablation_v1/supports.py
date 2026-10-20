"""Support-only ablations for the unchanged JointRegionMemoryClassifier.

The single-slot arms retain the two-slot ABI and parameter count. Their second
slot is structural padding: it has zero route mass and is never attended to.
Only the frozen runtime's predicted query labels are used here; query targets
are deliberately not inspected.
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import Iterable

import torch


ARM_SPECS = {
    "e": dict(slot_mode="single", support_mode="local", label_mode="predicted", source_arm="b"),
    "f": dict(slot_mode="single", support_mode="memory", label_mode="train", source_arm="d"),
    "a": dict(slot_mode="spatial", support_mode="local", label_mode="predicted", source_arm="a"),
    "c": dict(slot_mode="spatial", support_mode="memory", label_mode="train", source_arm="c"),
    "b": dict(slot_mode="soft", support_mode="local", label_mode="predicted", source_arm="b"),
    "d": dict(slot_mode="soft", support_mode="memory", label_mode="train", source_arm="d"),
    "r": dict(slot_mode="soft", support_mode="random_legal", label_mode="train", source_arm="d"),
    "l": dict(slot_mode="soft", support_mode="memory", label_mode="permuted_train", source_arm="d"),
    "n": dict(slot_mode="soft", support_mode="memory", label_mode="none", source_arm="d"),
}


def _hash_tensor(value):
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(tuple(value.shape)).encode())
    h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _seed(seed, *parts):
    payload = json.dumps([int(seed), *parts], ensure_ascii=False, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def _weighted_mean(values, weights):
    """Merge the two roles with their own mass, without a positive epsilon."""
    numerator = (torch.where(weights[..., None] > 0, values, 0) * weights[..., None]).sum(-2)
    mass = weights.sum(-1)
    return numerator / torch.where(mass > 0, mass, 1)[..., None], mass


def _first_slot(value):
    """Append an exactly zero second slot to a parent value."""
    return torch.stack((value, torch.zeros_like(value)), -2)


def _first_scalar(value):
    return torch.stack((value, torch.zeros_like(value)), -1)


class SupportFactory:
    """Build reusable ablation inputs without modifying frozen runtime inputs.

    ``prepare(example, arms)`` returns ordinary model keyword dictionaries.
    ``forward(model, example, arm, support)`` is required for arm N: normalized
    uniform labels satisfy the original model ABI, while a scoped forward hook
    removes the label-embedding output exactly. The hook is removed even when
    the model raises. Call forwards sequentially on any given model.

    ``random_choices`` retains only the latest image's CPU ID tensor to bound
    memory. The full current tensor also remains in the returned R support;
    ``random_records`` retains a compact reproducibility ledger for all images.
    ``permutation`` maps recipient parent row to donor parent row.
    """

    def __init__(self, runtime, seed):
        self.runtime = runtime
        self.seed = int(seed)
        self.bank = runtime.bank
        self.retrieval_bank = runtime.retrieval_bank
        self.soft = runtime.values["rules"]["soft"]
        self.eligible = self.retrieval_bank["eligible"]
        if self.eligible.dtype != torch.bool or self.eligible.ndim != 1:
            raise ValueError("Retrieval eligibility must be a boolean parent-row vector")
        self.single_table = self._single_table()
        self.permutation, self._permutation_strata = self._label_permutation()
        self.permuted_labels = self.soft["slot_labels"][self.permutation]
        self.no_labels = torch.where(
            self.soft["slot_valid"][..., None],
            torch.full_like(self.soft["slot_labels"], 0.25), 0,
        )
        self.random_choices = {}
        self.random_records = {}
        self.last_diagnostics = {}

    def _validate_memory_ids(self, example, ids):
        if ids.dtype != torch.long or ids.ndim != 2:
            raise ValueError("Memory parent IDs must be int64 [regions, K]")
        if bool(((ids < -1) | (ids >= len(self.eligible))).any()):
            raise ValueError("Memory parent IDs are out of range")
        chosen = ids[ids >= 0]
        if not bool(self.eligible[chosen].all()):
            raise ValueError("Memory selected an ineligible parent")
        for field, key in (("events", "event_id"), ("images", "image_id")):
            identity = int(example.diagnostics[key])
            if identity >= 0 and bool((self.retrieval_bank[field][chosen] == identity).any()):
                raise ValueError("Memory selected a same-query " + field + " parent")

    def _single_table(self):
        features, mass = _weighted_mean(self.soft["slot_f"], self.soft["slot_mass"])
        labels, label_mass = _weighted_mean(self.soft["slot_labels"], self.soft["label_mass"])
        valid = (mass > 0) & (label_mass > 0)
        return dict(
            support_f=_first_slot(features), support_t=self.bank["t128"],
            support_labels=_first_slot(labels),
            support_fraction=_first_scalar((mass > 0).to(features.dtype)),
            support_valid=_first_scalar(valid),
        )

    def _label_permutation(self):
        valid = self.soft["slot_valid"].detach().cpu()
        eligible = self.eligible.detach().cpu()
        events = self.retrieval_bank["events"].detach().cpu()
        if valid.shape != (len(eligible), 2):
            raise ValueError("Soft support validity does not match parent rows")
        mapping = torch.arange(len(eligible), dtype=torch.long)
        strata = []
        grouped = {}
        patterns = valid[:, 0].long() + 2 * valid[:, 1].long()
        for row in torch.where(eligible)[0].tolist():
            grouped.setdefault((int(events[row]), int(patterns[row])), []).append(row)
        for (event_id, pattern), rows in sorted(grouped.items()):
            order = list(rows)
            rng = random.Random(_seed(self.seed, "label-parent-permutation-v2", event_id, pattern))
            rng.shuffle(order)
            if len(order) > 1:
                offset = rng.randrange(1, len(order))
                for index, row in enumerate(order):
                    mapping[row] = order[(index + offset) % len(order)]
            strata.append(dict(
                event_id=event_id, pattern=[bool(pattern & 1), bool(pattern & 2)], count=len(rows),
                fixed_points=int(sum(mapping[row].item() == row for row in rows)),
                weak_small_stratum=len(rows) < 3,
            ))
        return mapping.to(self.soft["slot_labels"].device), strata

    def _single_query(self, example):
        soft = example.supports["b"]
        features, mass = _weighted_mean(soft["slot_f"], soft["slot_fraction"])
        ids = example.common["ids"]
        if ids.ndim == 2:
            ids = ids.unsqueeze(0).expand(example.common["q"].shape[0], -1, -1)
        q = example.common["q"]
        return dict(
            slot_f=_first_slot(features),
            slot_fraction=_first_scalar((mass > 0).to(features.dtype)),
            slot_valid=_first_scalar(mass > 0),
            route_q=torch.cat((q, torch.zeros_like(q)), 1),
            route_ids=torch.cat((2 * ids, 2 * ids + 1), 1),
        )

    def _single_local(self, example):
        soft = example.supports["b"]
        features, mass = _weighted_mean(soft["support_f"], soft["support_fraction"])
        labels, _ = _weighted_mean(soft["support_labels"], soft["support_fraction"])
        valid = (mass > 0) & soft["support_valid"].any(-1)
        return dict(
            support_ids=soft["support_ids"],
            support_f=_first_slot(features), support_t=soft["support_t"],
            support_labels=_first_slot(labels),
            support_fraction=_first_scalar((mass > 0).to(features.dtype)),
            support_valid=_first_scalar(valid),
        )

    def _random_ids(self, example):
        source = example.supports["d"]
        slot_active = source["slot_valid"][0].any(-1).detach().cpu()
        reference_counts = (source["support_ids"] >= 0).sum(-1).detach().cpu()
        active = slot_active & (reference_counts > 0)
        regions, topk = source["support_ids"].shape
        if len(active) != regions:
            raise ValueError("Random support query rows and validity differ")
        event_id = int(example.diagnostics["event_id"])
        image_id = int(example.diagnostics["image_id"])
        legal = self.eligible.clone()
        if event_id >= 0:
            legal &= self.retrieval_bank["events"] != event_id
        if image_id >= 0:
            legal &= self.retrieval_bank["images"] != image_id
        legal_rows = torch.where(legal)[0].detach().cpu().tolist()
        chosen = torch.full((regions, topk), -1, dtype=torch.long)
        for region in torch.where(active)[0].tolist():
            count = min(int(reference_counts[region]), len(legal_rows))
            rng = random.Random(_seed(self.seed, "random-legal-v1", str(example.image), region))
            if count:
                chosen[region, :count] = torch.tensor(rng.sample(legal_rows, count), dtype=torch.long)
        self.random_choices = {str(example.image): chosen}
        self.random_records[str(example.image)] = dict(
            support_ids_sha256=_hash_tensor(chosen), shape=list(chosen.shape),
            reference_parent_counts_sha256=_hash_tensor(reference_counts),
            legal_parent_count=len(legal_rows), active_queries=int(active.sum()),
            event_id=event_id, image_id=image_id,
        )
        return chosen.to(source["support_ids"].device)

    @torch.no_grad()
    def prepare(self, example, arms: Iterable[str]):
        arms = tuple(arms)
        if not arms or len(arms) != len(set(arms)) or any(arm not in ARM_SPECS for arm in arms):
            raise ValueError("Arms must be a nonempty unique subset of " + ",".join(ARM_SPECS))
        result = {}
        checked = set()
        single_query = self._single_query(example) if any(arm in ("e", "f") for arm in arms) else None
        for arm in arms:
            source = ARM_SPECS[arm]["source_arm"]
            support = dict(example.supports[source])
            if arm in ("e", "f"):
                support.update(single_query)
                support.update(self._single_local(example) if arm == "e" else self.single_table)
            elif arm == "r":
                support["support_ids"] = self._random_ids(example)
            elif arm == "l":
                support["support_labels"] = self.permuted_labels
                # A selected feature row and its label donor must both pass
                # image/event exclusion. Same-event strata preserve this in
                # the normal bank; explicit validation also detects malformed
                # image/event metadata rather than silently leaking labels.
                donor_ids = torch.where(
                    support["support_ids"] >= 0,
                    self.permutation[support["support_ids"].clamp_min(0)], -1,
                )
                self._validate_memory_ids(example, donor_ids)
            elif arm == "n":
                support["support_labels"] = self.no_labels
            if ARM_SPECS[arm]["support_mode"] != "local" and id(support["support_ids"]) not in checked:
                self._validate_memory_ids(example, support["support_ids"])
                checked.add(id(support["support_ids"]))
            result[arm] = support
        self.last_diagnostics = {str(example.image): {
            arm: dict(self._diagnostics(support),
                      selection_legal=ARM_SPECS[arm]["support_mode"] != "local")
            for arm, support in result.items()
        }}
        return result

    @staticmethod
    def _diagnostics(support):
        ids = support["support_ids"]
        valid = support["support_valid"]
        selected = valid[ids.clamp_min(0)] & (ids >= 0)[..., None]
        return dict(
            supported_queries=int(selected.any(-1).any(-1).sum()),
            parent_supported_queries=int((ids >= 0).any(-1).sum()),
            valid_parent_slots=int((ids >= 0).sum()),
            valid_slots=int(selected.sum()), total_parent_slots=ids.numel(),
            query_nonempty_slots=int(support["slot_valid"].sum()),
        )

    def forward(self, model, example, arm, support):
        if arm not in ARM_SPECS:
            raise ValueError("Unknown arm: " + str(arm))
        if arm != "n":
            return model(**example.common, **support)
        hook = model.label_embedding.register_forward_hook(
            lambda module, inputs, output: torch.zeros_like(output)
        )
        try:
            return model(**example.common, **support)
        finally:
            hook.remove()

    def provenance(self):
        eligible = self.eligible
        identity = torch.arange(len(eligible), device=self.permutation.device)
        changed_rows = (self.permutation != identity) & eligible
        changed_labels = (
            (self.permuted_labels != self.soft["slot_labels"]).flatten(1).any(-1) & eligible
        )
        return dict(
            format="BRIGHT_SLOT_MEMORY_SUPPORT_ABLATIONS_V1", seed=self.seed,
            arms={key: dict(value) for key, value in ARM_SPECS.items()},
            query_GT_used_for_supports=False, frozen_input_mutation=False,
            single_slot=dict(
                abi="two slots; slot 0 active, slot 1 structural zero",
                query_and_local_weights="soft slot_fraction",
                train_feature_weights="soft slot_mass",
                train_label_weights="soft label_mass, ignoring invalid GT mass",
                route="first nine=Q to even IDs; next nine=0 to odd IDs",
                parameter_count="unchanged JointRegionMemoryClassifier",
            ),
            label_permutation=dict(
                recipient_to_donor=self.permutation.detach().cpu().tolist(),
                mapping_sha256=_hash_tensor(self.permutation),
                permuted_labels_sha256=_hash_tensor(self.permuted_labels),
                rule="seeded parent derangement within eligible parent-event and two-slot validity strata",
                strata=[dict(record) for record in self._permutation_strata],
                event_strata_count=len({record["event_id"] for record in self._permutation_strata}),
                event_validity_strata_count=len(self._permutation_strata),
                eligible_rows=int(eligible.sum()), changed_parent_rows=int(changed_rows.sum()),
                actual_label_changed_rows=int(changed_labels.sum()),
                changed_parent_fraction=float(changed_rows.sum()) / max(int(eligible.sum()), 1),
                actual_label_changed_fraction=float(changed_labels.sum()) / max(int(eligible.sum()), 1),
                donor_exclusion="Donor parent event equals recipient parent event; selected donors also explicitly validated.",
                limitation="Singleton event/validity strata stay fixed; small strata or repeated labels can make this control weak.",
            ),
            random_supports=dict(
                rule="uniform without replacement matching D parent count per query, eligible and excluding same event/image",
                seed_rule="SHA256 JSON [seed, random-legal-v1, image, parent region index]; local Python Random",
                padding=-1, records={key: dict(value) for key, value in self.random_records.items()},
            ),
            no_label=dict(
                model_input="uniform normalized distributions on valid slots",
                implementation="scoped label_embedding forward hook returns exact zero, removed in finally",
                feature_keys_and_support_ids="unchanged",
            ),
        )
