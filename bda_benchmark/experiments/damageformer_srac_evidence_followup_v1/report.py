"""CPU/std-library-only analysis of existing evidence; never rerun or train a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


LABELS = ("Background", "Intact", "Damaged", "Destroyed")
LIMITS = [
    "Partial/failed audits are descriptive debugging evidence, not full-Val method comparisons.",
    "Region Hit@K is at least one same-dominant-class neighbor, not recall of all relevant items or pixel accuracy.",
    "Cosine-weighted label evidence is not Reader attention or a calibrated probability of correctness.",
    "Off/random/shuffle compare the same trained Reader; off is not the independently trained J1.",
    "Event-composition decomposition explains label purity, not mIoU or causal performance changes.",
    "A/B/C probes show sensitivity; they do not by themselves prove irreversible information loss.",
]


def finite_tree(value, path="root"):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite value at " + path)
    if isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, path + "/" + str(key))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            finite_tree(item, path + "/" + str(i))


def ratio(num, den):
    return num / den if den else None


def count(value, path="count"):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Expected nonnegative integer: " + path)
    return value


def tensor(value, shape, path="tensor"):
    if not shape:
        return count(value, path)
    if not isinstance(value, list) or len(value) != shape[0]:
        raise ValueError("Wrong count tensor shape: " + path)
    return [tensor(item, shape[1:], path) for item in value]


def add_tree(left, right, path="counts"):
    """Strict schema-preserving sum. Never sum per-image mIoU or percentages."""
    if isinstance(right, dict):
        if not isinstance(left, dict) or set(left) != set(right):
            raise ValueError("Count keys changed at " + path)
        return {key: add_tree(left[key], right[key], path + "/" + key) for key in right}
    if isinstance(right, list):
        if not isinstance(left, list) or len(left) != len(right):
            raise ValueError("Count shape changed at " + path)
        return [add_tree(a, b, path) for a, b in zip(left, right)]
    for value in (left, right):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid nonnegative counter at " + path)
    return left + right


def confusion_metrics(cm):
    cm = tensor(cm, (4, 4), "confusion")
    rows = [sum(row) for row in cm]
    cols = [sum(cm[i][j] for i in range(4)) for j in range(4)]
    classes, ious = {}, []
    for c, name in enumerate(LABELS):
        tp, fp, fn = cm[c][c], cols[c] - cm[c][c], rows[c] - cm[c][c]
        iou = ratio(tp, tp + fp + fn)
        ious.append(iou)
        classes[name] = dict(tp=tp, fp=fp, fn=fn, support=rows[c], iou=iou,
                             precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn),
                             f1=ratio(2 * tp, 2 * tp + fp + fn))
    return dict(count=sum(rows), confusion=cm, classes=classes,
                valid_miou=sum(value or 0. for value in ious[1:]) / 3 if sum(rows) else None)


def transition_metrics(joint):
    joint = tensor(joint, (4, 4, 4), "joint[GT,old,new]")
    old = [[sum(joint[g][o]) for o in range(4)] for g in range(4)]
    new = [[sum(joint[g][o][n] for o in range(4)) for n in range(4)] for g in range(4)]
    before, after = confusion_metrics(old), confusion_metrics(new)
    classes = {}
    for c, name in enumerate(LABELS):
        a, b = before["classes"][name], after["classes"][name]
        kept = joint[c][c][c]
        harm, rescue = a["tp"] - kept, b["tp"] - kept
        classes[name] = dict(harm=harm, rescue=rescue, net_tp=rescue-harm,
                             harm_rate=ratio(harm, a["tp"]), rescue_rate=ratio(rescue, a["fn"]),
                             old=a, new=b, delta_fp=b["fp"]-a["fp"])
    return dict(classes=classes, damaged_to_intact_harm=joint[2][2][1],
                intact_prediction_to_damaged_rescue=joint[2][1][2],
                intact_gt_correct_to_damaged_harm=joint[1][1][2],
                destroyed_gt_correct_to_damaged_harm=joint[3][3][2],
                gt_old_new=joint)


def cohort_rates(rows):
    output = {}
    for arm, cohorts in rows.items():
        output[arm] = {}
        for name, row in cohorts.items():
            n = count(row["count"])
            supported = count(row["any_supported_pixels"])
            positive = count(row["evidence_damaged"])
            final = count(row["final_damaged"])
            overlap = count(row["evidence_damaged_and_final_damaged"])
            if not 0 <= overlap <= min(positive, final) or not positive <= supported <= n or final > n:
                raise ValueError("Inconsistent cohort counts: " + arm + "/" + name)
            mass = row["support_mass_sum"]
            missing = row["unsupported_mass_sum"]
            epsilon = max(1e-5, n * 1e-6)
            if min(mass, missing) < 0 or max(mass, missing) > n + epsilon or abs(mass + missing - n) > epsilon:
                raise ValueError("Inconsistent cohort support mass")
            output[arm][name] = dict(**row,
                any_support_fraction=ratio(supported, n), mean_supported_Q_mass=ratio(mass, n),
                evidence_damaged_fraction_of_supported=ratio(positive, supported),
                evidence_damaged_fraction_of_cohort=ratio(positive, n),
                final_damaged_fraction_of_cohort=ratio(final, n),
                final_damaged_given_evidence_damaged=ratio(overlap, positive))
    return output


def retrieval_rates(rows):
    result = {}
    for arm, values in rows.items():
        result[arm] = {}
        for c, name in enumerate(LABELS):
            row = values[str(c)]
            q, s, hit, slots, same = [count(row[key]) for key in
                ("queries", "supported_queries", "hit_at_k_count", "valid_slots", "same_class_slots")]
            if not hit <= s <= q or same > slots or slots < s:
                raise ValueError("Inconsistent region retrieval counts")
            result[arm][name] = dict(**row, hit_at_k=ratio(hit, q), supported_fraction=ratio(s, q),
                                     same_class_fraction_of_valid_neighbors=ratio(same, slots))
        result[arm]["all_damaged_label_mass"] = values.get("all_damaged_label_mass")
    return result


def difficult_retrieval_rates(rows, cohorts):
    """New fixed-pixel-cohort support descriptors, distinct from region-level Hit@K."""
    if set(rows) != set(cohorts):
        raise ValueError("Difficult retrieval arms disagree with evidence cohorts")
    result = {}
    for arm, groups in rows.items():
        if set(groups) != set(cohorts[arm]):
            raise ValueError("Difficult retrieval cohort names disagree")
        result[arm] = {}
        for name, row in groups.items():
            n = count(row["count"])
            if n != cohorts[arm][name]["count"]:
                raise ValueError("Difficult retrieval cohort counts disagree")
            supported = row["support_mass_sum"]
            tolerance = max(1e-5, n * 1e-6)
            if supported < 0 or supported > n + tolerance:
                raise ValueError("Invalid difficult retrieval support mass")
            for field in ("damaged_hit_mass_sum", "damaged_neighbor_fraction_mass_sum", "intact_neighbor_fraction_mass_sum"):
                if row[field] < 0 or row[field] > supported + tolerance:
                    raise ValueError("Invalid difficult retrieval projected mass: " + field)
            if abs(supported - cohorts[arm][name]["support_mass_sum"]) > tolerance:
                raise ValueError("Difficult retrieval/evidence support mass disagrees")
            result[arm][name] = dict(**row,
                weighted_support_coverage=ratio(supported, n),
                damaged_hit_mass_per_supported_mass=ratio(row["damaged_hit_mass_sum"], supported),
                damaged_neighbor_fraction_per_supported_mass=ratio(row["damaged_neighbor_fraction_mass_sum"], supported),
                intact_neighbor_fraction_per_supported_mass=ratio(row["intact_neighbor_fraction_mass_sum"], supported))
    return result


def aggregate_memory(rows):
    """Aggregate image matrices and raw counters, rejecting duplicated/mismatched rows."""
    ids, events, total, per_event = [], {}, None, {}
    for row in rows:
        finite_tree(row)
        name, event = row.get("image"), row.get("event")
        if not isinstance(name, str) or not name or name in events or not isinstance(event, str) or not event:
            raise ValueError("Missing/duplicate image identity or event")
        ids.append(name)
        events[name] = event
        cm = {key: tensor(value["confusion"], (4, 4)) for key, value in row["metrics"].items()}
        if not {"j1", "j2", "j3"}.issubset(cm):
            raise ValueError("Missing J1/J2/J3 metrics")
        gt_rows = [sum(line) for line in cm["j1"]]
        if any([sum(line) for line in value] != gt_rows for value in cm.values()):
            raise ValueError("Prediction modes use different GT cohorts")
        trans = {}
        for key, value in row["transitions"].items():
            old, new = key.split("_to_", 1)
            joint = tensor(value["gt_old_new"], (4, 4, 4))
            if old not in cm or new not in cm:
                raise ValueError("Transition refers to missing predictions")
            old_cm = [[sum(joint[g][o]) for o in range(4)] for g in range(4)]
            new_cm = [[sum(joint[g][o][n] for o in range(4)) for n in range(4)] for g in range(4)]
            if old_cm != cm[old] or new_cm != cm[new]:
                raise ValueError("Transition marginal disagrees with confusion")
            trans[key] = joint
        cohorts = row["difficult_cohorts"]
        if set(cohorts) != {"j2", "j3"} or set(row["retrieval"]) != {"j2", "j3"}:
            raise ValueError("Missing paired Memory arms")
        cohort_rates(cohorts)
        retrieval_rates(row["retrieval"])
        for arm in ("j2", "j3"):
            wrong = cohorts[arm]["damaged_j1_wrong"]
            if wrong["count"] != gt_rows[2] - cm["j1"][2][2]:
                raise ValueError("Difficult cohort is not the fixed J1-wrong GT-Damaged cohort")
            if "j1_to_" + arm not in trans:
                raise ValueError("Missing paired J1 transition")
            joint = trans["j1_to_" + arm]
            if wrong["final_damaged"] != sum(joint[2][o][2] for o in range(4) if o != 2):
                raise ValueError("Cohort rescue disagrees with paired transition")
        item = dict(confusion=cm, transitions=trans, cohorts=cohorts, retrieval=row["retrieval"])
        if "difficult_retrieval" in row:
            difficult_retrieval_rates(row["difficult_retrieval"], cohorts)
            item["difficult_retrieval"] = row["difficult_retrieval"]
        total = item if total is None else add_tree(total, item)
        per_event[event] = item if event not in per_event else add_tree(per_event[event], item)
    if total is None:
        return dict(image_count=0, images=[], metrics={}, transitions={}, difficult_cohorts={}, retrieval={}, per_event={})
    def finish(item):
        value = dict(metrics={key: confusion_metrics(value) for key, value in item["confusion"].items()},
                    transitions={key: transition_metrics(value) for key, value in item["transitions"].items()},
                    difficult_cohorts=cohort_rates(item["cohorts"]), retrieval=retrieval_rates(item["retrieval"]))
        value["difficult_retrieval"] = (difficult_retrieval_rates(item["difficult_retrieval"], item["cohorts"])
                                         if "difficult_retrieval" in item else None)
        return value
    return dict(image_count=len(ids), images=ids, **finish(total),
                per_event={key: finish(value) for key, value in per_event.items()})


def gate_status(summary, audit, image_count):
    """Never infer successful reproduction from scores, subset success, or 450 alone."""
    reasons = []
    for field in ("complete", "gates_passed", "full_validation_evidence"):
        if summary.get(field) is not True:
            reasons.append("summary/" + field + " is not true")
        if audit is not None and field in audit and audit[field] is not True:
            reasons.append("audit/" + field + " is not true")
    if summary.get("smoke") is not False or summary.get("diagnostic_subset", False):
        reasons.append("smoke/subset or unknown smoke flag")
    if image_count != 450 or summary.get("image_count") != 450:
        reasons.append("not a confirmed 450-image evaluation")
    if summary.get("test_accessed") is not False:
        reasons.append("No-Test status is not confirmed")
    for source in (summary, audit or {}):
        if source.get("failure"):
            reasons.append("failure record exists")
        if source.get("training") is True or source.get("weights_updated") is True:
            reasons.append("not a frozen read-only audit")
        repro = source.get("reproduction")
        if isinstance(repro, dict) and repro.get("passed") is not True:
            reasons.append("reproduction did not pass")
        for field in ("frozen_evidence_unchanged", "input_files_unchanged", "code_unchanged",
                      "numerical_unchanged", "weights_buffers_unchanged", "numerical_execution_unchanged",
                      "followup_source_unchanged"):
            if source.get(field) is False:
                reasons.append(field + " failed")
        if "pool_guard" in source:
            guard = source["pool_guard"]
            if not isinstance(guard, dict) or guard.get("restored") is not True:
                reasons.append("pool guard restoration not confirmed")
            elif guard.get("failures") or guard.get("restore_conflicts"):
                reasons.append("pool guard contains failures/restore conflicts")
    repro = summary.get("reproduction") or (audit or {}).get("reproduction")
    if not isinstance(repro, dict) or repro.get("passed") is not True:
        reasons.append("full reference reproduction is not confirmed")
    if audit is None:
        reasons.append("final audit.json missing")
    return dict(formal_full_val=not reasons, reference_gate_pending=bool(reasons),
                classification="formal_full_val" if not reasons else "debug_only_partial_or_failed",
                reasons=list(dict.fromkeys(reasons)), original_failure=summary.get("failure"))


def damaged_stats(row):
    n = count(row["dominant_counts"][2])
    mass = row["soft_label_sum"][2] - row["damaged_soft_sum_non_damaged_dominant"]
    if mass < -1e-6 or mass > n + 1e-5 or (n == 0 and abs(mass) > 1e-6):
        raise ValueError("Inconsistent Damaged-majority label mass")
    return n, ratio(max(mass, 0), n)


def inventory_report(data):
    finite_tree(data)
    bank = data["bank"]
    bn, bmean = damaged_stats(bank)
    result = dict(bank=bank, train_shards_scanned=data.get("train_shards_scanned") is True,
                  matched_source_shards=data.get("matched_source_shards") is True,
                  selection_quality_is_not_retrieval_accuracy=True,
                  eligible_dominant_D_after_same_event_exclusion={}, composition=None)
    for event, row in data.get("by_event", {}).items():
        available = bn - count(row["dominant_counts"][2])
        if available < 0:
            raise ValueError("Event bank count exceeds total")
        result["eligible_dominant_D_after_same_event_exclusion"][event] = available
    result["eligibility_note"] = "Val disjoint from Train; strict same-event exclusion. Counts are available rows, NOT relevant-neighbor coverage. Unseen query event retains all bank rows."
    if not (result["train_shards_scanned"] and result["matched_source_shards"]):
        result["composition_missing_reason"] = "All-Train/source-row verification not complete"
        return result
    sn, smean = damaged_stats(data["all_train"]["eligible_regions"])
    source_events, bank_events = data["all_train_by_event"], data["by_event"]
    if set(bank_events) - set(source_events):
        raise ValueError("Bank event has no source event")
    weighted_source, total_s, total_b, events = 0., 0, 0, {}
    for event, rows in source_events.items():
        ns, ms = damaged_stats(rows["eligible_regions"])
        nb, mb = damaged_stats(bank_events[event]) if event in bank_events else (0, None)
        if nb > ns:
            raise ValueError("Bank exceeds eligible event candidates")
        total_s += ns
        total_b += nb
        if nb:
            weighted_source += nb * ms
        events[event] = dict(source_D_count=ns, bank_D_count=nb, source_D_mean=ms, bank_D_mean=mb,
                             within_event_delta_pp=100 * (mb-ms) if nb else None)
    if total_s != sn or total_b != bn:
        raise ValueError("Per-event D counts do not match totals")
    standardized = ratio(weighted_source, bn)
    result["composition"] = dict(source_D_mean=smean, source_reweighted_to_bank_events=standardized,
        bank_D_mean=bmean, total_drop_pp=100 * (smean-bmean) if sn and bn else None,
        event_composition_drop_pp=100 * (smean-standardized) if sn and bn else None,
        within_event_selection_drop_pp=100 * (standardized-bmean) if bn else None,
        per_event=events, units="unweighted per-token GT label fraction among D-majority tokens; not pixels/instances")
    result["bank_D_purity_bins"] = {
        key: dict(count=row["dominant_counts"][2], fraction=ratio(row["dominant_counts"][2], bn))
        for key, row in data.get("bank_quality", {}).get("purity", {}).items()}
    return result


def read_json(path, ledger):
    raw = Path(path).read_bytes()
    ledger[str(Path(path).resolve())] = hashlib.sha256(raw).hexdigest()
    value = json.loads(raw.decode("utf-8-sig"))
    finite_tree(value)
    return value


def verify_saved_subset(saved, calculated, path="audit"):
    """Final raw fields must match journal-derived values; derived rates may be extra."""
    if isinstance(saved, dict):
        if not isinstance(calculated, dict):
            raise ValueError("Saved aggregate schema differs: " + path)
        for key, value in saved.items():
            if key not in calculated:
                raise ValueError("Saved aggregate key missing: " + path + "/" + key)
            verify_saved_subset(value, calculated[key], path + "/" + key)
    elif isinstance(saved, list):
        if not isinstance(calculated, list) or len(saved) != len(calculated):
            raise ValueError("Saved aggregate shape differs: " + path)
        for a, b in zip(saved, calculated):
            verify_saved_subset(a, b, path)
    elif saved != calculated:
        raise ValueError("Saved aggregate disagrees with journal: " + path)


def memory_report(root, ledger):
    root = Path(root)
    summary = read_json(root / "summary.json", ledger)
    audit = read_json(root / "audit.json", ledger) if (root / "audit.json").is_file() else None
    journal = root / "images.jsonl"
    if journal.is_file():
        raw = journal.read_bytes()
        ledger[str(journal.resolve())] = hashlib.sha256(raw).hexdigest()
        # Never discard a malformed/truncated last line or silently deduplicate.
        rows = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
        evidence = aggregate_memory(rows)
        if audit is not None:
            if evidence["image_count"] != audit.get("image_count"):
                raise ValueError("Memory journal/final audit image counts disagree")
            if set(evidence["metrics"]) != set(audit["metrics"]):
                raise ValueError("Memory journal/final audit prediction modes disagree")
            for name, metric in evidence["metrics"].items():
                if metric["confusion"] != audit["metrics"][name]["confusion"]:
                    raise ValueError("Memory journal/final confusion disagrees: " + name)
            for field in ("difficult_cohorts", "difficult_retrieval"):
                if field in audit:
                    verify_saved_subset(audit[field], evidence.get(field), field)
            for name, row in audit["transitions"].items():
                if name not in evidence["transitions"] or row["gt_old_new"] != evidence["transitions"][name]["gt_old_new"]:
                    raise ValueError("Memory journal/final transition disagrees: " + name)
    elif audit is not None:
        evidence = dict(image_count=count(audit["image_count"]), images=None,
            identity_check="No images.jsonl; per-image identity cannot be independently checked",
            metrics={key: confusion_metrics(value["confusion"]) for key, value in audit["metrics"].items()},
            transitions={key: transition_metrics(value["gt_old_new"]) for key, value in audit["transitions"].items()},
            difficult_cohorts=cohort_rates(audit["difficult_cohorts"]), retrieval=retrieval_rates(audit["retrieval"]))
        evidence["difficult_retrieval"] = (difficult_retrieval_rates(audit["difficult_retrieval"], audit["difficult_cohorts"])
                                            if "difficult_retrieval" in audit else None)
    else:
        evidence = aggregate_memory([])
    status = gate_status(summary, audit, evidence["image_count"])
    if (root / "inventory.json").is_file():
        evidence["inventory"] = inventory_report(read_json(root / "inventory.json", ledger))
    return dict(root=str(root.resolve()), status=status, **evidence)


def feature_report(root, ledger):
    root = Path(root)
    summary = read_json(root / "summary.json", ledger)
    final_path = root / "audit.json"
    path = final_path if final_path.is_file() else root / "audit_partial.json"
    body = read_json(path, ledger) if path.is_file() else {}
    final = body if final_path.is_file() else None
    result = dict(root=str(root.resolve()), status=gate_status(summary, final, body.get("image_count", 0)),
                  image_count=body.get("image_count", 0),
                  metrics={key: confusion_metrics(value["confusion"]) for key, value in body.get("metrics", {}).items()},
                  damaged_margin_sums=body.get("damaged_margin_sums"), transitions={},
                  ABC=dict(A="pixel_grid: same F, original affine head, grid softmax then resize",
                           B="feature_mean: mean F through Q, unpool F, affine head then softmax",
                           C="region_mean: mean F through Q, affine head/softmax, then probability unpool",
                           native_control="a1 uses original resize-logits-before-softmax; do not confuse A with native A1"))
    for key, row in body.get("class_transitions", {}).items():
        result["transitions"][key] = transition_metrics(row["joint_gt_reference_candidate"])
    if not result["transitions"] and body.get("per_image"):
        ids, totals = set(), None
        for row in body["per_image"]:
            if row["image"] in ids:
                raise ValueError("Duplicate feature image")
            ids.add(row["image"])
            joint = {key: tensor(value, (4, 4, 4)) for key, value in row["joint_gt_reference_candidate"].items()}
            totals = joint if totals is None else add_tree(totals, joint)
        if len(ids) != result["image_count"]:
            raise ValueError("Feature per-image count mismatch")
        result["transitions"] = {key: transition_metrics(value) for key, value in (totals or {}).items()}
    return result


def build_report(feature=None, memory=None):
    if not feature and not memory:
        raise ValueError("Specify --feature and/or --memory")
    ledger, result = {}, dict(format="SRAC_EVIDENCE_FOLLOWUP_REPORT_V1", no_training=True,
                              no_inference=True, interpretation_limits=LIMITS)
    if feature:
        result["feature"] = feature_report(feature, ledger)
    if memory:
        result["memory"] = memory_report(memory, ledger)
    result["check_status"] = {
        name: dict(code_implemented=True, input_supplied=stage in result,
                   formal_full_val=stage in result and result[stage]["status"]["formal_full_val"])
        for name, stage in (("A_pixel", "feature"), ("B_mean_feature", "feature"), ("C_region_probability", "feature"),
                            ("Memory_fixed_difficult_cohort", "memory"), ("Memory_same_reader_interventions", "memory"))}
    feature_modes = result.get("feature", {}).get("metrics", {})
    for check, mode in (("A_pixel", "pixel_grid"), ("B_mean_feature", "feature_mean"), ("C_region_probability", "region_mean")):
        result["check_status"][check]["formal_full_val"] &= mode in feature_modes
    required_pairs = {arm + "_to_" + arm + "_" + mode for arm in ("j2", "j3") for mode in ("off", "random", "shuffle_labels")}
    result["check_status"]["Memory_same_reader_interventions"]["formal_full_val"] &= required_pairs.issubset(result.get("memory", {}).get("transitions", {}))
    result["input_sha256"] = ledger
    # The reader must not combine a file that changed while it was being read.
    for path, before in ledger.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != before:
            raise RuntimeError("Input changed while reporting: " + path)
    return result


def display(result):
    def pct(value):
        return "无定义" if value is None else f"{100 * value:.4f}%"
    for stage in ("feature", "memory"):
        if stage not in result:
            continue
        row = result[stage]
        print(f"\n{stage}: {row['image_count']}张；完整有效Val={row['status']['formal_full_val']}")
        if not row["status"]["formal_full_val"]:
            print("以下仅为排错线索，不是完整Val结论：" + "; ".join(row["status"]["reasons"]))
        for name, metric in row["metrics"].items():
            d = metric["classes"]["Damaged"]
            print(f"{name}: FG mIoU={pct(metric['valid_miou'])}; D P/R/F1={pct(d['precision'])}/{pct(d['recall'])}/{pct(d['f1'])}")
            print("  四类TP/FP/FN：" + "；".join(f"{label}={c['tp']}/{c['fp']}/{c['fn']}" for label, c in metric["classes"].items()))
        for pair, trans in row["transitions"].items():
            detail = "；".join(f"{name}伤害{c['harm']}/救回{c['rescue']}/ΔFP{c['delta_fp']:+d}" for name, c in trans["classes"].items())
            print(pair + ": " + detail)
            print(f"  D正确→I伤害={trans['damaged_to_intact_harm']}；I正确→D伤害={trans['intact_gt_correct_to_damaged_harm']}；X正确→D伤害={trans['destroyed_gt_correct_to_damaged_harm']}")
        for arm, cohorts in row.get("difficult_cohorts", {}).items():
            c = cohorts["damaged_j1_wrong"]
            print(f"{arm}固定J1错判D像素={c['count']}：有任意支持={pct(c['any_support_fraction'])}；支持Q质量={pct(c['mean_supported_Q_mass'])}")
            print(f"  标签证据支持D={c['evidence_damaged']}；其中最终救回={c['evidence_damaged_and_final_damaged']} ({pct(c['final_damaged_given_evidence_damaged'])})；全部救回率={pct(c['final_damaged_fraction_of_cohort'])}")
        for arm, retrieval in row.get("retrieval", {}).items():
            d = retrieval["Damaged"]
            print(f"{arm} D主类区域查询={d['queries']}；Hit@K={pct(d['hit_at_k'])}；有效邻居中D比例={pct(d['same_class_fraction_of_valid_neighbors'])}（不是上面的困难像素指标）")
        for arm, groups in (row.get("difficult_retrieval") or {}).items():
            d = groups["damaged_j1_wrong"]
            print(f"{arm}同一J1错判D像素：支持质量覆盖={pct(d['weighted_support_coverage'])}；支持质量中D命中={pct(d['damaged_hit_mass_per_supported_mass'])}；邻居D/I比例={pct(d['damaged_neighbor_fraction_per_supported_mass'])}/{pct(d['intact_neighbor_fraction_per_supported_mass'])}")
        if stage == "memory" and not row.get("difficult_retrieval"):
            print("旧结果缺少同一困难像素上的检索质量统计；不使用全部D主类区域Hit@K替代。")
        inventory = row.get("inventory")
        if inventory:
            print(f"独立库存检查：全Train扫描={inventory['train_shards_scanned']}；来源匹配={inventory['matched_source_shards']}")
            print("排除同事件后的D条目数（不是相关性保证）：", inventory["eligible_dominant_D_after_same_event_exclusion"])
            comp = inventory.get("composition")
            if comp:
                print("D标签占比：源=" + pct(comp["source_D_mean"]) + "；事件重加权=" + pct(comp["source_reweighted_to_bank_events"]) + "；实际库=" + pct(comp["bank_D_mean"]))
                print("纯度下降分解(pp)：", {key: comp[key] for key in ("total_drop_pp", "event_composition_drop_pp", "within_event_selection_drop_pp")})
    completed = sum(value["formal_full_val"] for value in result["check_status"].values())
    print(f"\n已实现5项检查；本报告输入中通过完整Val门槛的检查={completed}/5。没有自动认定根因。")
    print("A→B变差≠信息永久丢失；B/C含Softmax顺序差异；标签投票≠Reader注意力。")
    print("失败审计先排错；通过后逐段看库覆盖→检索→救回，并同时核查Intact/Destroyed伤害。")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature")
    p.add_argument("--memory")
    p.add_argument("--output", help="Optional NEW JSON file; exclusive creation, never overwrites")
    args = p.parse_args()
    if args.output and Path(args.output).exists():
        raise FileExistsError("Preserve existing report: " + args.output)
    result = build_report(args.feature, args.memory)
    display(result)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        print("报告已保存：", path)


if __name__ == "__main__":
    main()
