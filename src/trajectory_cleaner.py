# CELL_33
# Extract clean trajectories

import json
from pathlib import Path
from collections import Counter, defaultdict

CLEAN_TRAJECTORY_PATH = RUN_DIR / "clean_trajectories.jsonl"
CLEAN_AUDIT_PATH = RUN_DIR / "clean_trajectory_validation.json"

expected_keys = {
    (item["question_id"], policy)
    for item in selected_questions
    for policy in [
        "direct_iterative",
        "question_decomposition"
    ]
}

grouped = defaultdict(list)

for event in trajectory_records:
    key = (
        event.get("question_id"),
        event.get("policy")
    )
    grouped[key].append(event)


clean_records = []
clean_audit = []

for key in sorted(expected_keys):

    question_id, policy = key
    events = grouped.get(key, [])

    events.sort(
        key=lambda x: x.get("timestamp", "")
    )

    # 정상 실행의 시작 이벤트
    if policy == "direct_iterative":
        start_events = [
            e for e in events
            if e.get("event_type") == "query_generation"
            and e.get("round") == 1
        ]

    else:
        start_events = [
            e for e in events
            if e.get("event_type") == "question_decomposition"
        ]

    # 최종 답변 이벤트
    final_events = [
        e for e in events
        if e.get("event_type") == "final_answer"
    ]

    if not start_events or not final_events:
        clean_audit.append({
            "question_id": question_id,
            "policy": policy,
            "status": "invalid",
            "reason": "Missing start or final event"
        })
        continue

    # 마지막 final_answer 기준
    final_event = final_events[-1]
    final_time = final_event["timestamp"]

    # final_answer 이전의 가장 최근 시작 이벤트
    valid_starts = [
        e for e in start_events
        if e["timestamp"] <= final_time
    ]

    if not valid_starts:
        clean_audit.append({
            "question_id": question_id,
            "policy": policy,
            "status": "invalid",
            "reason": "No start event before final answer"
        })
        continue

    start_event = valid_starts[-1]
    start_time = start_event["timestamp"]

    # 선택한 실행의 이벤트만 추출
    selected_events = [
        e for e in events
        if start_time <= e["timestamp"] <= final_time
    ]

    selected_events.sort(
        key=lambda x: x.get("timestamp", "")
    )

    clean_records.extend(selected_events)

    # 이벤트 종류별 검사
    counts = Counter(
        e.get("event_type")
        for e in selected_events
    )

    retrieval_events = [
        e for e in selected_events
        if e.get("event_type") == "retrieval"
    ]

    retrieval_rounds = sorted([
        e["round"]
        for e in retrieval_events
        if e.get("round") is not None
    ])

    retrieved_doc_ids = [
        doc["doc_id"]
        for e in retrieval_events
        for doc in e.get("retrieved_docs", [])
        if doc.get("doc_id")
    ]

    unique_doc_ids = set(retrieved_doc_ids)

    expected_query_count = 4
    expected_retrieval_count = 4

    if policy == "direct_iterative":
        expected_counts = {
            "query_generation": 4,
            "retrieval": 4,
            "final_answer": 1
        }
    else:
        expected_counts = {
            "question_decomposition": 1,
            "query_generation": 4,
            "retrieval": 4,
            "final_answer": 1
        }

    count_check = all(
        counts.get(event_type, 0) == expected_count
        for event_type, expected_count in expected_counts.items()
    )

    round_check = retrieval_rounds == [1, 2, 3, 4]

    document_check = (
        len(retrieved_doc_ids) == 8
        and len(unique_doc_ids) == 8
    )

    status = (
        "pass"
        if count_check and round_check and document_check
        else "check_required"
    )

    clean_audit.append({
        "question_id": question_id,
        "policy": policy,
        "status": status,
        "selected_start_time": start_time,
        "selected_final_time": final_time,
        "event_counts": dict(counts),
        "retrieval_rounds": retrieval_rounds,
        "retrieved_document_entries": len(retrieved_doc_ids),
        "unique_documents": len(unique_doc_ids),
        "duplicate_documents": (
            len(retrieved_doc_ids) - len(unique_doc_ids)
        ),
        "count_check": count_check,
        "round_check": round_check,
        "document_check": document_check
    })


# 정제된 trajectory 저장
clean_records.sort(
    key=lambda x: (
        x.get("question_id", ""),
        x.get("policy", ""),
        x.get("timestamp", "")
    )
)

with open(
    CLEAN_TRAJECTORY_PATH, "w", encoding="utf-8"
) as f:
    for event in clean_records:
        f.write(
            json.dumps(event, ensure_ascii=False) + "\n"
        )


# 검증 결과 저장
passed = sum(
    item["status"] == "pass"
    for item in clean_audit
)

validation = {
    "expected_pairs": len(expected_keys),
    "audited_pairs": len(clean_audit),
    "passed_pairs": passed,
    "check_required_pairs": len(clean_audit) - passed,
    "total_clean_events": len(clean_records),
    "pair_details": clean_audit
}

with open(
    CLEAN_AUDIT_PATH, "w", encoding="utf-8"
) as f:
    json.dump(
        validation,
        f,
        ensure_ascii=False,
        indent=2
    )


print("=" * 65)
print("CLEAN TRAJECTORY VALIDATION")
print("=" * 65)

print(f"Expected pairs: {validation['expected_pairs']}")
print(f"Audited pairs: {validation['audited_pairs']}")
print(f"Passed pairs: {passed}")
print(f"Check required: {validation['check_required_pairs']}")
print(f"Clean events: {len(clean_records)}")

print("\nPair-level results:")

for item in clean_audit:
    print(
        f"{item['policy']} | "
        f"{item['question_id']} | "
        f"{item['status']}"
    )

print("\nSaved:")
print(CLEAN_TRAJECTORY_PATH)
print(CLEAN_AUDIT_PATH)
