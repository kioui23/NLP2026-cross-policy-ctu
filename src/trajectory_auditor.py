

# CELL_31
# Trajectory Log Audit

import json
from pathlib import Path
from collections import defaultdict, Counter

# 1. Trajectory 로그 파일
TRAJECTORY_AUDIT_PATH = LOG_DIR / "trajectories.jsonl"

# 2. JSONL 파일 읽기
trajectory_records = []
malformed_lines = []

with open(
    TRAJECTORY_AUDIT_PATH, "r", encoding="utf-8"
) as f:

    for line_number, line in enumerate(f, start=1):
        if not line.strip():
            continue

        try:
            trajectory_records.append(json.loads(line))
        except json.JSONDecodeError as e:
            malformed_lines.append({
                "line": line_number,
                "error": str(e)
            })


print("=" * 65)
print("TRAJECTORY LOG OVERVIEW")
print("=" * 65)

print(f"Total events: {len(trajectory_records)}")
print(f"Malformed lines: {len(malformed_lines)}")


# 3. 전체 event_type 분포
event_type_counts = Counter(
    record.get("event_type", "UNKNOWN")
    for record in trajectory_records
)

print("\n[Event Type Distribution]")

for event_type, count in event_type_counts.most_common():
    print(f"{event_type}: {count}")


# 4. question_id + policy 기준으로 그룹화
grouped_events = defaultdict(list)

for record in trajectory_records:
    key = (
        record.get("question_id"),
        record.get("policy")
    )
    grouped_events[key].append(record)


# 5. 각 실험 조합별 검색 기록 분석
audit_results = []

for (question_id, policy), events in sorted(
    grouped_events.items()
):

    events.sort(
        key=lambda x: x.get("timestamp", "")
    )

    # 검색 문서가 기록된 이벤트만 추출
    retrieval_events = [
        event for event in events
        if event.get("retrieved_docs")
    ]

    # 검색 round 확인
    retrieval_rounds = sorted({
        event["round"]
        for event in retrieval_events
        if event.get("round") is not None
    })

    # 검색된 전체 문서 ID 수집
    retrieved_doc_ids = []

    for event in retrieval_events:
        for doc in event.get("retrieved_docs", []):
            doc_id = doc.get("doc_id")

            if doc_id:
                retrieved_doc_ids.append(doc_id)

    # 중복 문서 검사
    doc_counts = Counter(retrieved_doc_ids)

    duplicate_docs = {
        doc_id: count
        for doc_id, count in doc_counts.items()
        if count > 1
    }

    # 이벤트 종류 확인
    types = [
        event.get("event_type", "UNKNOWN")
        for event in events
    ]

    # 시작/종료 관련 이벤트 후보 확인
    start_candidates = [
        event_type for event_type in types
        if "start" in event_type.lower()
        or "begin" in event_type.lower()
    ]

    end_candidates = [
        event_type for event_type in types
        if any(
            keyword in event_type.lower()
            for keyword in ["complete", "finish", "final", "answer", "end"]
        )
    ]

    audit_results.append({
        "question_id": question_id,
        "policy": policy,
        "total_events": len(events),
        "retrieval_events": len(retrieval_events),
        "retrieval_rounds": retrieval_rounds,
        "num_retrieved_doc_entries": len(retrieved_doc_ids),
        "num_unique_docs": len(set(retrieved_doc_ids)),
        "duplicate_docs": duplicate_docs,
        "event_types": sorted(set(types)),
        "start_candidates": start_candidates,
        "end_candidates": end_candidates
    })


# 6. 질문·정책별 요약
print("\n" + "=" * 65)
print("QUESTION-POLICY AUDIT")
print("=" * 65)

for result in audit_results:

    print(
        f"\n[{result['policy']}] "
        f"{result['question_id']}"
    )

    print(f"  Total events: {result['total_events']}")
    print(f"  Retrieval events: {result['retrieval_events']}")
    print(f"  Retrieval rounds: {result['retrieval_rounds']}")
    print(f"  Document entries: {result['num_retrieved_doc_entries']}")
    print(f"  Unique documents: {result['num_unique_docs']}")

    if result["duplicate_docs"]:
        print(f"  Duplicate docs: {result['duplicate_docs']}")
    else:
        print("  Duplicate docs: None")

    print(f"  Start event candidates: {result['start_candidates']}")
    print(f"  End event candidates: {result['end_candidates']}")


# 7. 예상 실험 조합과 실제 로그 조합 비교
expected_keys = {
    (item["question_id"], policy)
    for item in selected_questions
    for policy in [
        "direct_iterative",
        "question_decomposition"
    ]
}

actual_keys = set(grouped_events.keys())

missing_log_keys = expected_keys - actual_keys
unexpected_log_keys = actual_keys - expected_keys


# 8. 로그 품질 요약
print("\n" + "=" * 65)
print("AUDIT SUMMARY")
print("=" * 65)

print(f"Expected question-policy pairs: {len(expected_keys)}")
print(f"Pairs with trajectory logs: {len(actual_keys)}")
print(f"Missing pairs: {len(missing_log_keys)}")
print(f"Unexpected pairs: {len(unexpected_log_keys)}")
print(f"Malformed JSON lines: {len(malformed_lines)}")

if missing_log_keys:
    print("\nMissing pairs:")
    for key in sorted(missing_log_keys):
        print(key)

if unexpected_log_keys:
    print("\nUnexpected pairs:")
    for key in sorted(unexpected_log_keys):
        print(key)


# 9. 감사 결과 저장
AUDIT_OUTPUT_PATH = RUN_DIR / "trajectory_audit.json"

audit_report = {
    "total_events": len(trajectory_records),
    "malformed_lines": malformed_lines,
    "event_type_counts": dict(event_type_counts),
    "expected_pairs": len(expected_keys),
    "actual_pairs": len(actual_keys),
    "missing_pairs": [
        list(key) for key in sorted(missing_log_keys)
    ],
    "unexpected_pairs": [
        list(key) for key in sorted(unexpected_log_keys)
    ],
    "pair_details": audit_results
}

with open(
    AUDIT_OUTPUT_PATH, "w", encoding="utf-8"
) as f:
    json.dump(
        audit_report,
        f,
        ensure_ascii=False,
        indent=2
    )

print(f"\nAudit report saved: {AUDIT_OUTPUT_PATH}")
