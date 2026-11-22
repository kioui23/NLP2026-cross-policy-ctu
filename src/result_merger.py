
# CELL_30
# Merge and validate experiment results


import json
import csv
from pathlib import Path
from collections import defaultdict

# 1. 입력 파일
ORIGINAL_RESULTS_PATH = RUN_DIR / "experiment_results.jsonl"
RESUME_RESULTS_PATH = RUN_DIR / "resume_results.jsonl"

# 2. 출력 파일
MERGED_JSONL_PATH = RUN_DIR / "merged_experiment_results.jsonl"
MERGED_CSV_PATH = RUN_DIR / "merged_experiment_results.csv"
VALIDATION_PATH = RUN_DIR / "merge_validation.json"


def load_jsonl(path):
    records = []

    if not path.exists():
        print(f"[WARNING] 파일이 없습니다: {path}")
        return records

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


# 3. 두 파일 읽기
original_records = load_jsonl(ORIGINAL_RESULTS_PATH)
resume_records = load_jsonl(RESUME_RESULTS_PATH)

all_records = original_records + resume_records

print("입력 결과")
print(f"- Original: {len(original_records)}")
print(f"- Resume: {len(resume_records)}")
print(f"- Total: {len(all_records)}")


# 4. question_id + policy 기준으로 결과 그룹화
grouped_records = defaultdict(list)

for record in all_records:
    key = (
        record["question_id"],
        record["policy"]
    )
    grouped_records[key].append(record)


# 5. 성공한 결과만 선택
merged_records = []
duplicate_success_keys = []
failed_only_keys = []

for key, records in grouped_records.items():

    successful = [
        r for r in records
        if r.get("status") == "success"
    ]

    if not successful:
        failed_only_keys.append(key)
        continue

    # 성공 결과가 여러 개라면 가장 최근 결과 선택
    successful.sort(
        key=lambda r: r.get("timestamp", ""),
        reverse=True
    )

    selected = successful[0].copy()

    if len(successful) > 1:
        duplicate_success_keys.append(key)

    selected["source"] = (
        "original"
        if selected in original_records
        else "resume"
    )

    merged_records.append(selected)


# 6. 결과 정렬
merged_records.sort(
    key=lambda r: (
        r["question_id"],
        r["policy"]
    )
)


# 7. JSONL 저장
with open(
    MERGED_JSONL_PATH, "w", encoding="utf-8"
) as f:
    for record in merged_records:
        f.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )


# 8. CSV 저장
# 리스트/딕셔너리 필드는 JSON 문자열로 변환
csv_columns = [
    "question_id",
    "policy",
    "status",
    "answer",
    "gold_answer",
    "exact_match",
    "supporting_retrieved",
    "supporting_total",
    "supporting_recall",
    "num_rounds",
    "num_unique_documents",
    "elapsed_sec",
    "source",
    "timestamp"
]

with open(
    MERGED_CSV_PATH,
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=csv_columns,
        extrasaction="ignore"
    )

    writer.writeheader()

    for record in merged_records:
        writer.writerow(record)


# 9. 질문별 정책 완성 여부 검증
expected_question_ids = [
    item["question_id"]
    for item in selected_questions
]

expected_policies = [
    "direct_iterative",
    "question_decomposition"
]

actual_keys = {
    (r["question_id"], r["policy"])
    for r in merged_records
}

missing_keys = []

for question_id in expected_question_ids:
    for policy in expected_policies:
        key = (question_id, policy)

        if key not in actual_keys:
            missing_keys.append(key)


# 10. Supporting Recall 정책별 평균
recall_by_policy = defaultdict(list)

for record in merged_records:
    recall = record.get("supporting_recall")

    if recall is not None:
        recall_by_policy[record["policy"]].append(
            float(recall)
        )

recall_summary = {}

for policy, values in recall_by_policy.items():
    recall_summary[policy] = {
        "count": len(values),
        "mean": sum(values) / len(values)
    }


# 11. 검증 결과 저장
validation = {
    "expected_combinations": (
        len(expected_question_ids) * len(expected_policies)
    ),
    "merged_successful_combinations": len(merged_records),
    "missing_combinations": [
        {
            "question_id": qid,
            "policy": policy
        }
        for qid, policy in missing_keys
    ],
    "duplicate_success_combinations": [
        {
            "question_id": qid,
            "policy": policy
        }
        for qid, policy in duplicate_success_keys
    ],
    "failed_only_combinations": [
        {
            "question_id": qid,
            "policy": policy
        }
        for qid, policy in failed_only_keys
    ],
    "supporting_recall_summary": recall_summary,
    "is_complete": (
        len(merged_records) == 20
        and len(missing_keys) == 0
    )
}

with open(
    VALIDATION_PATH, "w", encoding="utf-8"
) as f:
    json.dump(
        validation,
        f,
        ensure_ascii=False,
        indent=2
    )


# 12. 최종 출력
print("\n" + "=" * 60)
print("MERGE VALIDATION")
print("=" * 60)

print(f"Expected combinations: {validation['expected_combinations']}")
print(f"Merged successful combinations: {len(merged_records)}")
print(f"Missing combinations: {len(missing_keys)}")
print(f"Duplicate successful combinations: {len(duplicate_success_keys)}")
print(f"Failed-only combinations: {len(failed_only_keys)}")

print("\nSupporting Recall Summary")

for policy, summary in recall_summary.items():
    print(
        f"{policy}: "
        f"{summary['mean']:.3f} "
        f"(n={summary['count']})"
    )

print("\nSaved files:")
print(MERGED_JSONL_PATH)
print(MERGED_CSV_PATH)
print(VALIDATION_PATH)

print("\nFinal status:")
print("PASS" if validation["is_complete"] else "CHECK REQUIRED")
