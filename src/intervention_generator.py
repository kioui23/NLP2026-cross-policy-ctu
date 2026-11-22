# CELL_41
import json
from pathlib import Path
from collections import defaultdict

# 1. Load clean trajectories


CLEAN_TRAJECTORY_PATH = RUN_DIR / "clean_trajectories.jsonl"

with open(CLEAN_TRAJECTORY_PATH, "r", encoding="utf-8") as f:
    clean_events = [
        json.loads(line)
        for line in f
        if line.strip()
    ]

# 2. Build document lookup


document_lookup = {}

for question_id, docs in documents_by_question.items():
    for doc in docs:
        document_lookup[doc["doc_id"]] = doc


# 3. Group events by question and policy

grouped_events = defaultdict(list)

for event in clean_events:
    key = (event["question_id"], event["policy"])
    grouped_events[key].append(event)

for key in grouped_events:
    grouped_events[key].sort(
        key=lambda x: x.get("round") or 0
    )

# 4. Extract document-level interventions

interventions = []

for (question_id, policy), events in grouped_events.items():

    retrieval_events = [
        e for e in events
        if e["event_type"] == "retrieval"
    ]

    query_events = {
        e["round"]: e
        for e in events
        if e["event_type"] == "query_generation"
    }

    decomposition_event = next(
        (
            e for e in events
            if e["event_type"] == "question_decomposition"
        ),
        None
    )

    # Documents retrieved in each round
    docs_by_round = defaultdict(list)

    for event in retrieval_events:
        round_num = event["round"]

        for doc_info in event["retrieved_docs"]:
            doc_id = doc_info["doc_id"]

            if doc_id not in document_lookup:
                raise KeyError(
                    f"Document not found: {doc_id}"
                )

            docs_by_round[round_num].append(
                document_lookup[doc_id]
            )

    # Only rounds 1-3 have a subsequent query
    for round_num in [1, 2, 3]:

        if round_num not in query_events:
            continue

        next_round = round_num + 1

        if next_round not in query_events:
            continue

        baseline_next_query = query_events[next_round]["query"]

        prefix_documents = [
            doc
            for previous_round in range(1, round_num)
            for doc in docs_by_round[previous_round]
        ]

        current_round_docs = docs_by_round[round_num]

        for target_doc in current_round_docs:

            target_doc_id = target_doc["doc_id"]

            sibling_documents = [
                doc
                for doc in current_round_docs
                if doc["doc_id"] != target_doc_id
            ]

            excluded_doc_ids = {
                doc["doc_id"]
                for doc in (
                    prefix_documents
                    + sibling_documents
                    + [target_doc]
                )
            }

            intervention = {
                "intervention_id": str(uuid.uuid4()),
                "question_id": question_id,
                "policy": policy,
                "target_doc_id": target_doc_id,
                "target_doc_title": target_doc["title"],
                "intervention_round": round_num,

                "prefix_documents": prefix_documents,
                "sibling_documents": sibling_documents,

                "excluded_doc_ids": sorted(
                    excluded_doc_ids
                ),

                "baseline_next_query": baseline_next_query,

                "baseline_next_query_call_id": (
                    query_events[next_round].get("call_id")
                ),

                "baseline_next_round": next_round,

                "decomposition": (
                    decomposition_event.get("observation")
                    if decomposition_event else None
                )
            }

            interventions.append(intervention)

# 5. Validation

intervention_keys = [
    (
        item["question_id"],
        item["policy"],
        item["target_doc_id"]
    )
    for item in interventions
]

assert len(intervention_keys) == len(set(intervention_keys)), \
    "Duplicate intervention detected."

assert all(
    item["target_doc_id"] in item["excluded_doc_ids"]
    for item in interventions
)

assert all(
    item["baseline_next_query"]
    for item in interventions
)

# Save intervention manifest
INTERVENTION_PATH = RUN_DIR / "interventions.jsonl"

with open(INTERVENTION_PATH, "w", encoding="utf-8") as f:
    for item in interventions:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


# 6. Summary

summary = defaultdict(int)

for item in interventions:
    summary[item["policy"]] += 1

print("=" * 60)
print("INTERVENTION EXTRACTION")
print("=" * 60)

print("Total interventions:", len(interventions))

for policy, count in summary.items():
    print(f"{policy}: {count}")

print("Unique intervention targets:", len(set(intervention_keys)))
print("Saved:", INTERVENTION_PATH)
