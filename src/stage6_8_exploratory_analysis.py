import json
import csv
from pathlib import Path
from difflib import SequenceMatcher
from collections import defaultdict

BASE = Path("hw2_results/full_10_question_baseline/96c5b40d-d5a9-45da-8af0-5d998be18d34")
ANALYSIS = Path("analysis")
ANALYSIS.mkdir(exist_ok=True)

def load_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

interventions = load_jsonl(BASE / "ctu_preparation/interventions.jsonl")
replays = load_jsonl(BASE / "ctu_preparation/counterfactual_replay/replay_results.jsonl")
baseline_results = load_jsonl(BASE / "experiment_results.jsonl")

replay_by_id = {r["intervention_id"]: r for r in replays}
baseline_by_key = {(r["question_id"], r["policy"]): r for r in baseline_results}

records = []
policy_doc_impact = defaultdict(list)
policy_doc_change = defaultdict(list)
policy_doc_effort = defaultdict(list)

for intervention in interventions:
    iid = intervention["intervention_id"]
    replay = replay_by_id.get(iid)
    baseline = baseline_by_key.get(
        (intervention["question_id"], intervention["policy"])
    )

    if replay is None or baseline is None:
        continue

    base_answer = str(baseline.get("answer", ""))
    cf_answer = str(replay.get("answer", ""))

    answer_similarity = SequenceMatcher(
        None, base_answer.lower().strip(), cf_answer.lower().strip()
    ).ratio()

    answer_changed = base_answer.strip().lower() != cf_answer.strip().lower()

    baseline_rounds = baseline.get("num_rounds")
    replay_rounds = len(replay.get("round_records", []))

    remaining_baseline_rounds = (
        max(0, baseline_rounds - intervention["intervention_round"])
        if baseline_rounds is not None else None
    )

    effort_delta_proxy = (
        replay_rounds - remaining_baseline_rounds
        if remaining_baseline_rounds is not None else None
    )

    # Answer perturbation proxy: larger means removal changed the answer more.
    answer_perturbation = 1 - answer_similarity

    row = {
        "intervention_id": iid,
        "question_id": intervention["question_id"],
        "policy": intervention["policy"],
        "target_doc_id": intervention["target_doc_id"],
        "target_doc_title": intervention.get("target_doc_title", ""),
        "intervention_round": intervention["intervention_round"],
        "baseline_answer": base_answer,
        "counterfactual_answer": cf_answer,
        "answer_changed": answer_changed,
        "answer_similarity": answer_similarity,
        "answer_perturbation_proxy": answer_perturbation,
        "baseline_total_rounds": baseline_rounds,
        "baseline_remaining_rounds_proxy": remaining_baseline_rounds,
        "counterfactual_replay_rounds": replay_rounds,
        "effort_delta_proxy": effort_delta_proxy,
        "counterfactual_first_query": replay.get("first_replay_query", "")
    }
    records.append(row)

    policy = intervention["policy"]
    doc = intervention["target_doc_id"]
    policy_doc_impact[(policy, doc)].append(answer_perturbation)
    policy_doc_change[(policy, doc)].append(int(answer_changed))

    if effort_delta_proxy is not None:
        policy_doc_effort[(policy, doc)].append(effort_delta_proxy)

# Aggregate document-level ranking
doc_summary = []
for (policy, doc), values in policy_doc_impact.items():
    changes = policy_doc_change[(policy, doc)]
    efforts = policy_doc_effort.get((policy, doc), [])
    doc_summary.append({
        "policy": policy,
        "target_doc_id": doc,
        "intervention_count": len(values),
        "mean_answer_perturbation_proxy": sum(values) / len(values),
        "answer_change_rate": sum(changes) / len(changes),
        "mean_effort_delta_proxy": sum(efforts) / len(efforts) if efforts else None
    })

# Spearman correlation implemented without external dependencies
def rank_values(values):
    sorted_vals = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j+1][1] == sorted_vals[i][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[sorted_vals[k][0]] = avg_rank
        i = j + 1
    return ranks

def pearson(x, y):
    if len(x) < 2:
        return None
    mx, my = sum(x)/len(x), sum(y)/len(y)
    numerator = sum((a-mx)*(b-my) for a,b in zip(x,y))
    dx = sum((a-mx)**2 for a in x)
    dy = sum((b-my)**2 for b in y)
    if dx == 0 or dy == 0:
        return None
    return numerator / (dx*dy)**0.5

policies = sorted({r["policy"] for r in records})
transfer_results = []

if len(policies) >= 2:
    p1, p2 = policies[:2]
    d1 = {d: s["mean_answer_perturbation_proxy"]
           for d, s in []}
    agg = {(r["policy"], r["target_doc_id"]): r
           for r in doc_summary}

    docs1 = {doc for (p, doc) in policy_doc_impact if p == p1}
    docs2 = {doc for (p, doc) in policy_doc_impact if p == p2}
    shared_docs = sorted(docs1 & docs2)

    x = [agg[(p1, d)]["mean_answer_perturbation_proxy"] for d in shared_docs]
    y = [agg[(p2, d)]["mean_answer_perturbation_proxy"] for d in shared_docs]

    rho = pearson(rank_values(x), rank_values(y)) if len(shared_docs) >= 2 else None

    transfer_results.append({
        "policy_1": p1,
        "policy_2": p2,
        "shared_document_count": len(shared_docs),
        "spearman_rank_correlation": rho,
        "shared_documents": shared_docs
    })

# Write detailed intervention-level CSV
intervention_csv = ANALYSIS / "stage6_intervention_analysis.csv"
if records:
    with intervention_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

# Write document rankings CSV
doc_csv = ANALYSIS / "stage7_document_utility_proxy_rankings.csv"
doc_summary.sort(
    key=lambda r: (r["policy"], -r["mean_answer_perturbation_proxy"])
)
if doc_summary:
    with doc_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(doc_summary[0].keys()))
        writer.writeheader()
        writer.writerows(doc_summary)

# Write report
report = ANALYSIS / "stage6_8_exploratory_results.txt"
with report.open("w", encoding="utf-8") as f:
    f.write("STAGE 6-8 EXPLORATORY COUNTERFACTUAL ANALYSIS\n")
    f.write("="*60 + "\n")
    f.write("This is NOT official answer F1 or reproduced CTU.\n")
    f.write("Answer perturbation is a string-change proxy only.\n\n")

    f.write(f"Matched interventions with successful replay: {len(records)}\n")
    f.write(f"Policies: {policies}\n\n")

    for policy in policies:
        subset = [r for r in records if r["policy"] == policy]
        f.write(f"[{policy}]\n")
        f.write(f"N={len(subset)}\n")
        f.write(f"Answer changed rate={sum(r['answer_changed'] for r in subset)/len(subset):.4f}\n")
        f.write(f"Mean answer perturbation proxy={sum(r['answer_perturbation_proxy'] for r in subset)/len(subset):.4f}\n")
        f.write(f"Mean answer similarity={sum(r['answer_similarity'] for r in subset)/len(subset):.4f}\n")

        effort = [r["effort_delta_proxy"] for r in subset if r["effort_delta_proxy"] is not None]
        if effort:
            f.write(f"Mean effort delta proxy={sum(effort)/len(effort):.4f}\n")
        f.write("\n")

    f.write("[Cross-policy document ranking transfer]\n")
    for result in transfer_results:
        f.write(f"Policies: {result['policy_1']} vs {result['policy_2']}\n")
        f.write(f"Shared target documents: {result['shared_document_count']}\n")
        f.write(f"Spearman rank correlation: {result['spearman_rank_correlation']}\n")
        f.write("Shared docs:\n")
        for doc in result["shared_documents"]:
            f.write(f"  - {doc}\n")
        f.write("\n")

    f.write("Interpretation cautions:\n")
    f.write("- Answer perturbation does not measure correctness.\n")
    f.write("- Effort delta is a proxy based on replay rounds vs remaining baseline rounds.\n")
    f.write("- Document ranking is exploratory and may be sparse.\n")
    f.write("- Pilot size is small; no generalization claims.\n")
    f.write("- Official CTU is not computed.\n")

print("ANALYSIS COMPLETE")
print(f"Matched replays: {len(records)}")
print(f"Intervention analysis: {intervention_csv}")
print(f"Document rankings: {doc_csv}")
print(f"Summary report: {report}")
