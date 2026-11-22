from src.ctu_evaluation import (
    evaluate_next_query_comparison,
    save_next_query_evaluation
)

pilot_ctu_result = {}

PILOT_CTU_PATH = RUN_DIR / "pilot_ctu_result.json"

# Baseline vs. Counterfactual next-query evaluation
next_query_result = evaluate_next_query_comparison(
    test_intervention=test_intervention,
    test_replay_result=test_replay_result,
    documents_by_question=documents_by_question,
    k=10
)

# Print results
print("=" * 65)
print("CORRECTED NEXT-QUERY EVALUATION")
print("=" * 65)

print(
    f"Baseline nDCG@10: "
    f"{next_query_result['baseline_next_query_ndcg']:.4f}"
)

print(
    f"Counterfactual nDCG@10: "
    f"{next_query_result['counterfactual_next_query_ndcg']:.4f}"
)

print(
    f"Next-query delta: "
    f"{next_query_result['next_query_delta']:+.4f}"
)

print("\nBaseline ranking:")
for i, (title, rel) in enumerate(
    zip(
        next_query_result["baseline_ranked_titles"],
        next_query_result["baseline_ranked_relevance"]
    ),
    start=1
):
    print(f"{i}. {title} | relevance={rel}")

print("\nCounterfactual ranking:")
for i, (title, rel) in enumerate(
    zip(
        next_query_result["counterfactual_ranked_titles"],
        next_query_result["counterfactual_ranked_relevance"]
    ),
    start=1
):
    print(f"{i}. {title} | relevance={rel}")


# Save evaluation result
save_next_query_evaluation(
    result=next_query_result,
    output_path=PILOT_CTU_PATH
)


# CELL_82
import re
import json
import math
import string
from collections import Counter
from pathlib import Path

# 1. Load extraction results

ANSWER_EXTRACTION_PATH = (
    RUN_DIR / "answer_extraction" / "answer_extractions.jsonl"
)

assert ANSWER_EXTRACTION_PATH.exists(), (
    f"Answer extraction file not found: {ANSWER_EXTRACTION_PATH}"
)

extraction_records = load_extraction_records(
    ANSWER_EXTRACTION_PATH
)

successful_extractions = [
    r for r in extraction_records
    if r.get("status") == "success"
]

extraction_by_key = {
    (r["intervention_id"], r["answer_type"]): r
    for r in successful_extractions
}

expected_extraction_keys = {
    (item["intervention_id"], answer_type)
    for item in metric_input
    for answer_type in ["baseline", "counterfactual"]
}

actual_extraction_keys = {
    (r["intervention_id"], r["answer_type"])
    for r in successful_extractions
}

assert len(successful_extractions) == 240, (
    f"Expected 240 successful extractions, "
    f"got {len(successful_extractions)}"
)

assert len(extraction_by_key) == 240, (
    "Duplicate extraction records detected."
)

assert actual_extraction_keys == expected_extraction_keys, (
    "Extraction records do not match expected intervention-answer pairs."
)

# 2. HotpotQA standard answer normalization and F1

def normalize_answer(text):

    def remove_articles(s):
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def remove_punctuation(s):
        return "".join(
            ch for ch in s
            if ch not in string.punctuation
        )

    def white_space_fix(s):
        return " ".join(s.split())

    text = str(text).lower()
    text = remove_punctuation(text)
    text = remove_articles(text)

    return white_space_fix(text)


def compute_token_f1(prediction, gold):

    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


# 3. nDCG@10

def compute_ndcg_at_10(query, question_id):

    ranked_docs = retrieve(
        query=query,
        question_id=question_id,
        top_k=10,
        excluded=None
    )

    docs = documents_by_question[question_id]

    supporting_doc_ids = {
        d["doc_id"]
        for d in docs
        if d.get("is_supporting", False)
    }

    relevance = [
        1 if d["doc_id"] in supporting_doc_ids else 0
        for d in ranked_docs
    ]

    dcg = sum(
        rel / math.log2(rank + 2)
        for rank, rel in enumerate(relevance)
    )

    ideal_relevance = sorted(
        [
            1 if d["doc_id"] in supporting_doc_ids else 0
            for d in docs
        ],
        reverse=True
    )[:10]

    idcg = sum(
        rel / math.log2(rank + 2)
        for rank, rel in enumerate(ideal_relevance)
    )

    if idcg == 0:
        return None

    return dcg / idcg


# 4. Calculate metrics for all interventions

metric_components = []

for item in metric_input:

    iid = item["intervention_id"]
    qid = item["question_id"]

    baseline_extraction = extraction_by_key[
        (iid, "baseline")
    ]

    cf_extraction = extraction_by_key[
        (iid, "counterfactual")
    ]

    baseline_answer = baseline_extraction["extracted_answer"]
    cf_answer = cf_extraction["extracted_answer"]

    gold_answer = item["gold_answer"]

    assert isinstance(baseline_answer, str), (
        f"Invalid baseline answer type for {iid}"
    )

    assert isinstance(cf_answer, str), (
        f"Invalid counterfactual answer type for {iid}"
    )

    baseline_f1 = compute_token_f1(
        baseline_answer,
        gold_answer
    )

    cf_f1 = compute_token_f1(
        cf_answer,
        gold_answer
    )

    baseline_ndcg = compute_ndcg_at_10(
        item["baseline_next_query"],
        qid
    )

    cf_ndcg = compute_ndcg_at_10(
        item["counterfactual_next_query"],
        qid
    )

    delta_answer = baseline_f1 - cf_f1

    delta_next_query = (
        baseline_ndcg - cf_ndcg
        if baseline_ndcg is not None and cf_ndcg is not None
        else None
    )

    record = {
        "intervention_id": iid,
        "question_id": qid,
        "execution_id": item.get("execution_id"),
        "policy": item["policy"],
        "intervention_round": item["intervention_round"],
        "target_doc_id": item["target_doc_id"],

        "gold_answer": gold_answer,
        "baseline_extracted_answer": baseline_answer,
        "counterfactual_extracted_answer": cf_answer,

        "baseline_answer_f1": baseline_f1,
        "counterfactual_answer_f1": cf_f1,
        "delta_answer": delta_answer,

        "baseline_next_query": item["baseline_next_query"],
        "counterfactual_next_query": item["counterfactual_next_query"],

        "baseline_next_query_ndcg10": baseline_ndcg,
        "counterfactual_next_query_ndcg10": cf_ndcg,
        "delta_next_query": delta_next_query,

        "status": (
            "valid"
            if baseline_ndcg is not None and cf_ndcg is not None
            else "missing_ndcg"
        )
    }

    metric_components.append(record)

# 5. Save results

METRIC_COMPONENTS_PATH = RUN_DIR / "metric_components.jsonl"

with open(METRIC_COMPONENTS_PATH, "w", encoding="utf-8") as f:
    for record in metric_components:
        f.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )

# 6. Integrity validation

assert len(metric_components) == 120, (
    f"Expected 120 metric records, got {len(metric_components)}"
)

assert len({
    r["intervention_id"] for r in metric_components
}) == 120, "Duplicate intervention IDs detected"

assert all(
    0.0 <= r["baseline_answer_f1"] <= 1.0
    and 0.0 <= r["counterfactual_answer_f1"] <= 1.0
    for r in metric_components
), "Answer F1 out of range"

assert all(
    r["baseline_next_query_ndcg10"] is None
    or 0.0 <= r["baseline_next_query_ndcg10"] <= 1.0
    for r in metric_components
), "Baseline nDCG out of range"

assert all(
    r["counterfactual_next_query_ndcg10"] is None
    or 0.0 <= r["counterfactual_next_query_ndcg10"] <= 1.0
    for r in metric_components
), "Counterfactual nDCG out of range"

valid_records = [
    r for r in metric_components
    if r["status"] == "valid"
]

missing_records = [
    r for r in metric_components
    if r["status"] != "valid"
]

# 7. Summary

print("=" * 70)
print("METRIC COMPONENTS SUMMARY")
print("=" * 70)

print("Total interventions:", len(metric_components))
print("Valid:", len(valid_records))
print("Missing nDCG:", len(missing_records))

print("\nMean Answer F1")

print(
    "Baseline:",
    round(
        sum(r["baseline_answer_f1"] for r in metric_components)
        / len(metric_components),
        4
    )
)

print(
    "Counterfactual:",
    round(
        sum(r["counterfactual_answer_f1"] for r in metric_components)
        / len(metric_components),
        4
    )
)

print("\nMean nDCG@10")

if valid_records:

    print(
        "Baseline:",
        round(
            sum(r["baseline_next_query_ndcg10"] for r in valid_records)
            / len(valid_records),
            4
        )
    )

    print(
        "Counterfactual:",
        round(
            sum(r["counterfactual_next_query_ndcg10"] for r in valid_records)
            / len(valid_records),
            4
        )
    )

print("\nDelta summary")

print(
    "Mean delta_answer:",
    round(
        sum(r["delta_answer"] for r in metric_components)
        / len(metric_components),
        4
    )
)

if valid_records:

    print(
        "Mean delta_next_query:",
        round(
            sum(r["delta_next_query"] for r in valid_records)
            / len(valid_records),
            4
        )
    )

print("\nPolicy breakdown")

for policy in sorted({
    r["policy"] for r in metric_components
}):

    subset = [
        r for r in metric_components
        if r["policy"] == policy
    ]

    valid_subset = [
        r for r in subset
        if r["delta_next_query"] is not None
    ]

    mean_query_delta = (
        sum(r["delta_next_query"] for r in valid_subset)
        / len(valid_subset)
        if valid_subset
        else None
    )

    print(
        policy,
        "| N =", len(subset),
        "| Mean answer delta =",
        round(
            sum(r["delta_answer"] for r in subset)
            / len(subset),
            4
        ),
        "| Mean query delta =",
        (
            round(mean_query_delta, 4)
            if mean_query_delta is not None
            else "N/A"
        )
    )

# 8. Save and validate output

print("\nSaved:", METRIC_COMPONENTS_PATH)

assert METRIC_COMPONENTS_PATH.exists()

with open(METRIC_COMPONENTS_PATH, "r", encoding="utf-8") as f:
    saved_records = [
        json.loads(line)
        for line in f
        if line.strip()
    ]

assert len(saved_records) == 120

assert len({
    r["intervention_id"] for r in saved_records
}) == 120

print("Output records:", len(saved_records))
print("METRIC COMPONENTS VALIDATION: PASS")

# CELL_83
import json
import math
from pathlib import Path

# 1. Validate Phase 2 input

assert len(metric_components) == 120, (
    f"Expected 120 metric records, got {len(metric_components)}"
)

assert all(
    r["delta_answer"] is not None
    and r["delta_next_query"] is not None
    for r in metric_components
), "Missing delta values detected. Cannot compute CTU."

# 2. Min-max normalization

def minmax_normalize(values):

    min_value = min(values)
    max_value = max(values)

    if max_value == min_value:
        return [0.0] * len(values)

    return [
        (value - min_value) / (max_value - min_value)
        for value in values
    ]


delta_answer_values = [
    r["delta_answer"]
    for r in metric_components
]

delta_query_values = [
    r["delta_next_query"]
    for r in metric_components
]

normalized_answer = minmax_normalize(
    delta_answer_values
)

normalized_query = minmax_normalize(
    delta_query_values
)

# 3. Calculate Adapted CTU

ctu_components = []

for i, item in enumerate(metric_components):

    normalized_delta_answer = normalized_answer[i]
    normalized_delta_query = normalized_query[i]

    adapted_ctu = (
        normalized_delta_answer
        + normalized_delta_query
    ) / 2

    record = {
        "intervention_id": item["intervention_id"],
        "question_id": item["question_id"],
        "execution_id": item.get("execution_id"),
        "policy": item["policy"],
        "intervention_round": item["intervention_round"],
        "target_doc_id": item["target_doc_id"],

        "delta_answer": item["delta_answer"],
        "delta_next_query": item["delta_next_query"],

        "normalized_delta_answer": normalized_delta_answer,
        "normalized_delta_next_query": normalized_delta_query,

        "adapted_ctu": adapted_ctu
    }

    ctu_components.append(record)

# 4. Save results

ADAPTED_CTU_PATH = RUN_DIR / "adapted_ctu.jsonl"

with open(ADAPTED_CTU_PATH, "w", encoding="utf-8") as f:

    for record in ctu_components:

        f.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )

# 5. Integrity validation

assert len(ctu_components) == 120

assert len({
    r["intervention_id"] for r in ctu_components
}) == 120, "Duplicate intervention IDs detected"

assert all(
    0.0 <= r["normalized_delta_answer"] <= 1.0
    and 0.0 <= r["normalized_delta_next_query"] <= 1.0
    and 0.0 <= r["adapted_ctu"] <= 1.0
    for r in ctu_components
), "CTU values out of range"

assert all(
    math.isfinite(r["adapted_ctu"])
    for r in ctu_components
), "Non-finite CTU values detected"

# 6. Summary

print("=" * 70)
print("PHASE 3 — ADAPTED CTU SUMMARY")
print("=" * 70)

print("Total interventions:", len(ctu_components))

print(
    "Mean adapted CTU:",
    round(
        sum(r["adapted_ctu"] for r in ctu_components)
        / len(ctu_components),
        4
    )
)

print(
    "Mean normalized answer delta:",
    round(
        sum(r["normalized_delta_answer"] for r in ctu_components)
        / len(ctu_components),
        4
    )
)

print(
    "Mean normalized query delta:",
    round(
        sum(r["normalized_delta_next_query"] for r in ctu_components)
        / len(ctu_components),
        4
    )
)

print("\nPolicy breakdown")

for policy in sorted({
    r["policy"] for r in ctu_components
}):

    subset = [
        r for r in ctu_components
        if r["policy"] == policy
    ]

    mean_ctu = (
        sum(r["adapted_ctu"] for r in subset)
        / len(subset)
    )

    mean_answer = (
        sum(r["normalized_delta_answer"] for r in subset)
        / len(subset)
    )

    mean_query = (
        sum(r["normalized_delta_next_query"] for r in subset)
        / len(subset)
    )

    print(
        policy,
        "| N =", len(subset),
        "| Mean CTU =", round(mean_ctu, 4),
        "| Answer =", round(mean_answer, 4),
        "| Query =", round(mean_query, 4)
    )

# 7. Validate saved output

assert ADAPTED_CTU_PATH.exists()

with open(ADAPTED_CTU_PATH, "r", encoding="utf-8") as f:

    saved_ctu = [
        json.loads(line)
        for line in f
        if line.strip()
    ]

assert len(saved_ctu) == 120

assert len({
    r["intervention_id"] for r in saved_ctu
}) == 120

print("\nSaved:", ADAPTED_CTU_PATH)
print("Output records:", len(saved_ctu))
print("PHASE 3 VALIDATION: PASS")
