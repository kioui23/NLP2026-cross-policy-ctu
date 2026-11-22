import json
import csv
import math
from pathlib import Path
from collections import defaultdict

RUN_ID = "96c5b40d-d5a9-45da-8af0-5d998be18d34"
LOG_PATH = Path("hw2_results/trajectory_log.jsonl")
RUN_DIR = Path("hw2_results/full_10_question_baseline") / RUN_ID
OUT_DIR = RUN_DIR / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POLICY_A = "direct_iterative"
POLICY_B = "question_decomposition"


def average_rank(values):
    """Tie-aware average ranks, 1 = best."""
    sorted_values = sorted(values)
    result = {}
    i = 0
    while i < len(sorted_values):
        j = i
        while j < len(sorted_values) and sorted_values[j] == sorted_values[i]:
            j += 1
        rank = (i + 1 + j) / 2
        result[sorted_values[i]] = rank
        i = j
    return result


def spearman(x, y):
    if len(x) < 2:
        return None

    rx = average_rank(x)
    ry = average_rank(y)

    # Original values are mapped to their ranks
    xx = [rx[v] for v in x]
    yy = [ry[v] for v in y]

    mx = sum(xx) / len(xx)
    my = sum(yy) / len(yy)

    numerator = sum((a-mx)*(b-my) for a, b in zip(xx, yy))
    denominator = math.sqrt(
        sum((a-mx)**2 for a in xx) *
        sum((b-my)**2 for b in yy)
    )

    if denominator == 0:
        return None
    return numerator / denominator


# Load only the target run's retrieval events
records = []
with LOG_PATH.open(encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("run_id") == RUN_ID and r.get("event_type") == "retrieval":
            records.append(r)

# question -> policy -> doc_id -> best rank / score
data = defaultdict(lambda: defaultdict(dict))
question_text = {}

for r in records:
    qid = r.get("question_id")
    policy = r.get("policy")
    question_text[qid] = question_text.get(qid, "")

    # Policy A: ranked retrieval documents
    docs = r.get("retrieved_docs") or []

    # Policy B: documents are logged under action.new_documents
    if policy == POLICY_B:
        docs = r.get("action", {}).get("new_documents", []) or []
        current_round = r.get("observation", {}).get("round")
    else:
        current_round = r.get("round")

    for rank, doc in enumerate(docs, start=1):
        if isinstance(doc, dict):
            doc_id = doc.get("doc_id")
            title = doc.get("title", "")
            score = doc.get("score")
        else:
            doc_id = doc
            title = str(doc).split("::")[-1] if doc else ""
            score = None

        if not doc_id:
            continue

        existing = data[qid][policy].get(doc_id)

        if existing is None:
            data[qid][policy][doc_id] = {
                "best_rank": rank,
                "title": title,
                "score": score,
                "first_round": current_round,
            }
        elif rank < existing["best_rank"]:
            existing["best_rank"] = rank

# Load questions from experiment results for readable report
results_path = RUN_DIR / "experiment_results.jsonl"
if results_path.exists():
    with results_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            qid = r.get("question_id")
            if qid:
                question_text[qid] = r.get("question", "")

overlap_rows = []
ranking_rows = []

all_qids = sorted(question_text.keys())

for qid in all_qids:
    a = data[qid][POLICY_A]
    b = data[qid][POLICY_B]

    set_a = set(a.keys())
    set_b = set(b.keys())

    shared = set_a & set_b
    only_a = set_a - set_b
    only_b = set_b - set_a
    union = set_a | set_b

    jaccard = len(shared) / len(union) if union else 0.0

    shared_sorted = sorted(shared)

    ranks_a = [a[d]["best_rank"] for d in shared_sorted]
    ranks_b = [b[d]["best_rank"] for d in shared_sorted]

    rank_corr = spearman(ranks_a, ranks_b) if len(shared) >= 2 else None

    overlap_rows.append({
        "question_id": qid,
        "question": question_text.get(qid, ""),
        "policy_a_docs": len(set_a),
        "policy_b_docs": len(set_b),
        "shared_docs": len(shared),
        "only_a_docs": len(only_a),
        "only_b_docs": len(only_b),
        "union_docs": len(union),
        "jaccard": round(jaccard, 4),
        "shared_doc_rank_spearman": (
            round(rank_corr, 4) if rank_corr is not None else ""
        ),
    })

    # Document-level ranking comparison
    for doc_id in sorted(union):
        ranking_rows.append({
            "question_id": qid,
            "question": question_text.get(qid, ""),
            "doc_id": doc_id,
            "title": (a.get(doc_id) or b.get(doc_id) or {}).get("title", ""),
            "retrieved_by_a": doc_id in set_a,
            "retrieved_by_b": doc_id in set_b,
            "rank_a": a.get(doc_id, {}).get("best_rank", ""),
            "rank_b": b.get(doc_id, {}).get("best_rank", ""),
            "score_a": a.get(doc_id, {}).get("score", ""),
            "score_b": b.get(doc_id, {}).get("score", ""),
            "first_round_a": a.get(doc_id, {}).get("first_round", ""),
            "first_round_b": b.get(doc_id, {}).get("first_round", ""),
        })


# Write question-level overlap analysis
overlap_path = OUT_DIR / "question_overlap.csv"
with overlap_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=overlap_rows[0].keys() if overlap_rows else [])
    writer.writeheader()
    writer.writerows(overlap_rows)

# Write document-level ranking comparison
ranking_path = OUT_DIR / "document_ranking.csv"
with ranking_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=ranking_rows[0].keys() if ranking_rows else [])
    writer.writeheader()
    writer.writerows(ranking_rows)

# Aggregate statistics
valid_jaccard = [r["jaccard"] for r in overlap_rows]
valid_corr = [
    r["shared_doc_rank_spearman"]
    for r in overlap_rows
    if r["shared_doc_rank_spearman"] != ""
]

avg_jaccard = sum(valid_jaccard) / len(valid_jaccard) if valid_jaccard else 0
avg_rank_corr = sum(valid_corr) / len(valid_corr) if valid_corr else None

total_a = sum(r["policy_a_docs"] for r in overlap_rows)
total_b = sum(r["policy_b_docs"] for r in overlap_rows)
total_shared = sum(r["shared_docs"] for r in overlap_rows)

summary_path = OUT_DIR / "analysis_summary.txt"

with summary_path.open("w", encoding="utf-8") as f:
    f.write("CROSS-POLICY RETRIEVAL ANALYSIS\n")
    f.write("=" * 45 + "\n")
    f.write(f"Run ID: {RUN_ID}\n")
    f.write(f"Questions analyzed: {len(overlap_rows)}\n")
    f.write(f"Retrieval events: {len(records)}\n\n")

    f.write("OVERALL STATISTICS\n")
    f.write("-" * 45 + "\n")
    f.write(f"Total unique-doc counts (summed per question): A={total_a}, B={total_b}\n")
    f.write(f"Shared document counts (summed per question): {total_shared}\n")
    f.write(f"Mean Jaccard similarity: {avg_jaccard:.4f}\n")
    f.write(
        f"Mean shared-document rank Spearman: "
        f"{avg_rank_corr:.4f}\n" if avg_rank_corr is not None
        else "Mean shared-document rank Spearman: N/A\n"
    )
    f.write(f"Questions with >=2 shared docs: {len(valid_corr)}\n\n")

    f.write("QUESTION-LEVEL RESULTS\n")
    f.write("-" * 45 + "\n")
    for r in overlap_rows:
        f.write(
            f"{r['question_id']} | "
            f"Shared={r['shared_docs']} | "
            f"Jaccard={r['jaccard']:.4f} | "
            f"Rank rho={r['shared_doc_rank_spearman']}\n"
        )
        f.write(f"  {r['question']}\n")

print("ANALYSIS COMPLETE")
print(f"Questions analyzed: {len(overlap_rows)}")
print(f"Retrieval events: {len(records)}")
print(f"Mean Jaccard: {avg_jaccard:.4f}")
print(
    f"Mean Rank Spearman: {avg_rank_corr:.4f}"
    if avg_rank_corr is not None else "Mean Rank Spearman: N/A"
)
print("\nGenerated files:")
print(overlap_path)
print(ranking_path)
print(summary_path)
print("\n=== QUESTION OVERLAP PREVIEW ===")
for r in overlap_rows:
    print(
        f"{r['question_id'][:8]} | "
        f"A={r['policy_a_docs']:2} B={r['policy_b_docs']:2} | "
        f"Shared={r['shared_docs']:2} | "
        f"Jaccard={r['jaccard']:.3f} | "
        f"Rank rho={r['shared_doc_rank_spearman']}"
    )
