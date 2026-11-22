"""Analysis utilities for the cross-policy CTU experiment.

This module consolidates the baseline summary, answer linking, cross-policy CTU
ranking agreement, and robustness analyses from the research notebook.

It intentionally does not perform answer extraction, Answer F1, next-query nDCG,
or CTU calculation; those belong to the evaluation pipeline.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable



POLICY_A = "direct_iterative"
POLICY_B = "question_decomposition"
POLICIES = (POLICY_A, POLICY_B)



def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return records



def save_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path



def save_json(data: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path



def summarize_baseline(
    experiment_records: list[dict[str, Any]],
    summary_path: str | Path | None = None,
) -> dict[str, Any]:
    """Summarize baseline run statuses and successful-run metrics by policy."""
    status_counts = {
        policy: Counter(
            record.get("status", "unknown")
            for record in experiment_records
            if record.get("policy") == policy
        )
        for policy in POLICIES
    }


    successful_records = [
        record for record in experiment_records
        if record.get("status") == "success"
    ]


    policy_summary = {}
    for policy in POLICIES:
        records = [r for r in successful_records if r.get("policy") == policy]
        policy_summary[policy] = {
            "successful_runs": len(records),
            "mean_supporting_recall": _mean(records, "supporting_recall"),
            "mean_rounds": _mean(records, "num_rounds"),
            "mean_unique_documents": _mean(records, "num_unique_documents"),
        }


    summary_df_records = [dict(record) for record in experiment_records]
    summary = {
        "total_runs": len(experiment_records),
        "status_by_policy": {
            policy: dict(counts) for policy, counts in status_counts.items()
        },
        "successful_run_summary": policy_summary,
        "records": summary_df_records,
    }

    if summary_path is not None:
        path = Path(summary_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # CSV is retained for compatibility with the original notebook output.
        import pandas as pd
        pd.DataFrame(experiment_records).to_csv(
            path, index=False, encoding="utf-8-sig"
        )
        print(f"Baseline summary saved: {path}")

    print("Status by policy:")
    for policy, counts in status_counts.items():
        print(f"  {policy}: {dict(counts)}")
    print("\nSuccessful execution summary:")
    for policy, metrics in policy_summary.items():
        print(f"\nPolicy: {policy}")
        print(f"  Successful runs: {metrics['successful_runs']}")
        print(f"  Mean supporting recall: {metrics['mean_supporting_recall']}")
        print(f"  Mean rounds: {metrics['mean_rounds']}")
        print(f"  Mean documents: {metrics['mean_unique_documents']}")

    return summary



def _mean(records: list[dict[str, Any]], key: str) -> float | None:
    values = [r[key] for r in records if r.get(key) is not None]
    return sum(values) / len(values) if values else None



def build_metric_input(
    interventions: list[dict[str, Any]],
    questions_by_id: dict[str, dict[str, Any]],
    replay_by_intervention: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build metric records from interventions and replay outputs."""
    metric_input = []
    seen_ids = set()


    for intervention in interventions:
        intervention_id = intervention["intervention_id"]
        if intervention_id in seen_ids:
            raise ValueError(f"Duplicate intervention_id: {intervention_id}")
        seen_ids.add(intervention_id)


        question_id = intervention["question_id"]
        if question_id not in questions_by_id:
            raise KeyError(f"Missing question: {question_id}")
        if intervention_id not in replay_by_intervention:
            raise KeyError(f"Missing replay: {intervention_id}")


        replay = replay_by_intervention[intervention_id]
        replay_result = replay.get("result", replay)
        if replay.get("status") == "failed":
            raise RuntimeError(f"Replay failed: {intervention_id}")
        if replay_result.get("answer") is None:
            raise ValueError(f"Replay answer missing: {intervention_id}")


        gold = questions_by_id[question_id]
        metric_input.append({
            "intervention_id": intervention_id,
            "question_id": question_id,
            "policy": intervention["policy"],
            "intervention_round": intervention["intervention_round"],
            "gold_answer": gold["answer"],
            "supporting_facts": gold["supporting_facts"],
            "baseline_answer": None,
            "counterfactual_answer": replay_result["answer"],
            "baseline_next_query": intervention.get("baseline_next_query"),
            "counterfactual_next_query": replay_result.get("first_replay_query"),
            "target_doc_id": intervention["target_doc_id"],
            "target_document_title": intervention.get("target_doc_title"),
            "status": "baseline_answer_pending",
        })


    print(f"Metric input records: {len(metric_input)}")
    print(f"Unique intervention IDs: {len(seen_ids)}")
    print("Metric input validation: PASS")
    return metric_input



def link_baseline_and_counterfactual_answers(
    metric_input: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]],
    counterfactual_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Link baseline answers by question-policy and replay answers by intervention."""
    baseline_by_key = {}
    for record in baseline_records:
        if record.get("status") != "success":
            raise ValueError(f"Baseline record is not successful: {record}")
        key = (record["question_id"], record["policy"])
        if key in baseline_by_key:
            raise ValueError(f"Duplicate baseline question-policy key: {key}")
        baseline_by_key[key] = record["answer"]


    counterfactual_by_id = {}
    for replay in counterfactual_records:
        intervention_id = replay["intervention_id"]
        if replay.get("status") != "success":
            raise RuntimeError(f"Counterfactual replay failed: {intervention_id}")
        result = replay.get("result", replay)
        if intervention_id in counterfactual_by_id:
            raise ValueError(f"Duplicate counterfactual ID: {intervention_id}")
        counterfactual_by_id[intervention_id] = result["answer"]


    for item in metric_input:
        baseline_key = (item["question_id"], item["policy"])
        intervention_id = item["intervention_id"]
        if baseline_key not in baseline_by_key:
            raise KeyError(f"Missing baseline answer: {baseline_key}")
        if intervention_id not in counterfactual_by_id:
            raise KeyError(f"Missing counterfactual answer: {intervention_id}")
        item["baseline_answer"] = baseline_by_key[baseline_key]
        item["counterfactual_answer"] = counterfactual_by_id[intervention_id]


    if not all(
        item.get("baseline_answer") is not None
        and item.get("counterfactual_answer") is not None
        for item in metric_input
    ):
        raise ValueError("Answer linking validation failed: missing answers")


    print("ANSWER LINKING VALIDATION: PASS")
    print(f"Metric input records: {len(metric_input)}")
    print(f"Baseline records: {len(baseline_by_key)}")
    print(f"Counterfactual records: {len(counterfactual_by_id)}")
    return metric_input



def _spearman(records: list[dict[str, Any]]) -> float | None:
    if len(records) < 2:
        return None
    import numpy as np
    from scipy.stats import spearmanr

    x = np.asarray([r["ctu_direct_iterative"] for r in records], dtype=float)
    y = np.asarray([r["ctu_question_decomposition"] for r in records], dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    rho = float(spearmanr(x, y).statistic)
    return rho if np.isfinite(rho) else None



def analyze_cross_policy_ctu(
    ctu_records: list[dict[str, Any]],
    run_dir: str | Path,
    n_bootstrap: int = 2000,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Pair CTU by question-document and compute pooled rho + question bootstrap."""
    import numpy as np
    from scipy.stats import spearmanr


    if not ctu_records:
        raise ValueError("No CTU records provided")
    if {r.get("policy") for r in ctu_records} != set(POLICIES):
        raise ValueError("CTU records must contain both expected policies")



    by_policy = {policy: {} for policy in POLICIES}
    for record in ctu_records:
        key = (record["question_id"], record["target_doc_id"])
        policy = record["policy"]
        if key in by_policy[policy]:
            raise ValueError(f"Duplicate CTU pair for {policy}: {key}")
        by_policy[policy][key] = record



    a_records = list(by_policy[POLICY_A].values())
    b_records = list(by_policy[POLICY_B].values())
    a_keys, b_keys = set(by_policy[POLICY_A]), set(by_policy[POLICY_B])
    common_keys = sorted(a_keys & b_keys)
    a_only_keys = a_keys - b_keys
    b_only_keys = b_keys - a_keys



    paired_records = []
    for key in common_keys:
        a, b = by_policy[POLICY_A][key], by_policy[POLICY_B][key]
        paired_records.append({
            "question_id": key[0],
            "target_doc_id": key[1],
            "ctu_direct_iterative": a["adapted_ctu"],
            "ctu_question_decomposition": b["adapted_ctu"],
            "delta_ctu": a["adapted_ctu"] - b["adapted_ctu"],
        })
    if not paired_records:
        raise ValueError("No question-document pairs matched across policies")



    x = np.asarray([r["ctu_direct_iterative"] for r in paired_records], dtype=float)
    y = np.asarray([r["ctu_question_decomposition"] for r in paired_records], dtype=float)
    pooled = spearmanr(x, y)
    observed_rho = float(pooled.statistic)
    observed_p = float(pooled.pvalue)



    groups = defaultdict(list)
    for record in paired_records:
        groups[record["question_id"]].append(record)
    question_ids = sorted(groups)
    rng = random.Random(random_seed)
    bootstrap_rhos = []
    for _ in range(n_bootstrap):
        sampled_qids = [rng.choice(question_ids) for _ in question_ids]
        sampled = [r for qid in sampled_qids for r in groups[qid]]
        rho = _spearman(sampled)
        if rho is not None:
            bootstrap_rhos.append(rho)
    if not bootstrap_rhos:
        raise RuntimeError("No valid bootstrap correlations")
    ci_lower, ci_upper = np.percentile(bootstrap_rhos, [2.5, 97.5])



    matched_per_question = Counter(r["question_id"] for r in paired_records)
    unique_questions = sorted({key[0] for key in a_keys | b_keys})
    summary = {
        "policy_a": POLICY_A,
        "policy_b": POLICY_B,
        "n_policy_a": len(a_records),
        "n_policy_b": len(b_records),
        "n_common_pairs": len(common_keys),
        "n_a_only_pairs": len(a_only_keys),
        "n_b_only_pairs": len(b_only_keys),
        "n_unique_questions": len(unique_questions),
        "spearman_rho": observed_rho,
        "asymptotic_p_value": observed_p,
        "bootstrap_requested": n_bootstrap,
        "bootstrap_valid": len(bootstrap_rhos),
        "bootstrap_mean_rho": float(np.mean(bootstrap_rhos)),
        "bootstrap_median_rho": float(np.median(bootstrap_rhos)),
        "bootstrap_ci_95": [float(ci_lower), float(ci_upper)],
        "bootstrap_seed": random_seed,
        "matched_pairs_per_question": dict(matched_per_question),
    }


    run_dir = Path(run_dir)
    save_jsonl(paired_records, run_dir / "ctu_paired_comparison.jsonl")
    save_json({"summary": summary, "bootstrap_rhos": bootstrap_rhos}, run_dir / "ctu_bootstrap_rhos.json")

    print("PHASE 4 — CROSS-POLICY CTU RANKING AGREEMENT")
    print(f"Policy A records: {len(a_records)} | Policy B records: {len(b_records)}")
    print(f"Matched pairs: {len(common_keys)} | A-only: {len(a_only_keys)} | B-only: {len(b_only_keys)}")
    print(f"Unique questions: {len(unique_questions)}")
    print(f"Pooled Spearman rho: {observed_rho:.4f} | p-value: {observed_p:.6f}")
    print(f"Question-level bootstrap 95% CI: ({ci_lower:.4f}, {ci_upper:.4f})")
    print("PHASE 4 VALIDATION: PASS")
    return {"paired_records": paired_records, "summary": summary, "bootstrap_rhos": bootstrap_rhos}



def analyze_robustness(
    paired_records: list[dict[str, Any]],
    overall_rho: float,
    run_dir: str | Path,
) -> dict[str, Any]:
    """Perform leave-one-question-out and within-question Spearman analysis."""
    import numpy as np


    question_groups = defaultdict(list)
    for record in paired_records:
        question_groups[record["question_id"]].append(record)
    question_ids = sorted(question_groups)



    loqo_results = []
    for excluded_qid in question_ids:
        remaining = [r for r in paired_records if r["question_id"] != excluded_qid]
        loqo_results.append({
            "excluded_question_id": excluded_qid,
            "n_remaining_questions": len(question_ids) - 1,
            "n_remaining_pairs": len(remaining),
            "spearman_rho": _spearman(remaining),
        })
    valid_loqo = [r["spearman_rho"] for r in loqo_results if r["spearman_rho"] is not None]



    within_results = []
    for qid in question_ids:
        records = question_groups[qid]
        within_results.append({
            "question_id": qid,
            "n_matched_pairs": len(records),
            "spearman_rho": _spearman(records),
        })
    valid_within = [r["spearman_rho"] for r in within_results if r["spearman_rho"] is not None]



    def stats(values):
        if not values:
            return {"n_valid": 0, "min": None, "max": None, "mean": None, "median": None}
        return {
            "n_valid": len(values),
            "min": float(min(values)),
            "max": float(max(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
        }


    robustness_summary = {
        "overall_spearman_rho": overall_rho,
        "loqo": {
            "n_questions": len(question_ids),
            **stats(valid_loqo),
            "results": loqo_results,
        },
        "within_question": {
            "n_questions": len(question_ids),
            **stats(valid_within),
            "results": within_results,
        },
    }
    path = save_json(robustness_summary, Path(run_dir) / "ctu_robustness_results.json")
    print("PHASE 5 — ROBUSTNESS ANALYSIS")
    print(f"LOQO valid runs: {len(valid_loqo)}/{len(question_ids)}")
    print(f"Within-question valid correlations: {len(valid_within)}/{len(question_ids)}")
    print(f"Robustness results saved: {path}")
    print("PHASE 5 VALIDATION: PASS")
    return robustness_summary
