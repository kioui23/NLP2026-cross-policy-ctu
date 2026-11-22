#!/usr/bin/env python3
"""Checkpointed counterfactual replay runner.

Dry-run by default. Pass --execute to make Gemini/API calls.
Completed interventions are checkpointed to JSONL and skipped on resume.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ID = "96c5b40d-d5a9-45da-8af0-5d998be18d34"
POLICY_A = "direct_iterative"
POLICY_B = "question_decomposition"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl_line(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        f.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default=DEFAULT_RUN_ID)
    ap.add_argument("--execute", action="store_true", help="Actually call the LLM API. Omitted means dry-run.")
    ap.add_argument("--limit", type=int, default=0, help="Limit interventions for a small test; 0 means all.")
    ap.add_argument("--policy", choices=[POLICY_A, POLICY_B, "both"], default="both")
    args = ap.parse_args()


    run_dir = ROOT / "hw2_results" / "full_10_question_baseline" / args.run_id
    prep_dir = run_dir / "ctu_preparation"
    intervention_path = prep_dir / "interventions.jsonl"
    clean_path = prep_dir / "clean_trajectories.jsonl"
    question_path = ROOT / "hw2_results" / "hotpotqa_selected_questions.json"
    docs_path = ROOT / "hw2_results" / "hotpotqa_documents.json"

    for p in (intervention_path, clean_path, question_path, docs_path):
        if not p.exists():
            raise SystemExit(f"Missing required input: {p}")

    interventions = read_jsonl(intervention_path)
    clean_events = read_jsonl(clean_path)
    selected_questions = json.loads(question_path.read_text(encoding="utf-8"))
    # Load corpus to initialize the retriever used by replay functions.
    all_documents = json.loads(docs_path.read_text(encoding="utf-8"))



    policy_filter = {POLICY_A, POLICY_B} if args.policy == "both" else {args.policy}
    selected = [x for x in interventions if x.get("policy") in policy_filter]
    if args.limit:
        selected = selected[:args.limit]

    out_dir = prep_dir / "counterfactual_replay"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "replay_results.jsonl"
    failures_path = out_dir / "failures.jsonl"
    manifest_path = out_dir / "replay_manifest.json"

    # Resume based on completed result IDs. Failed IDs are eligible for retry.
    completed_ids = set()
    if results_path.exists():
        for row in read_jsonl(results_path):
            if row.get("intervention_id"):
                completed_ids.add(row["intervention_id"])
    failed_ids = set()
    if failures_path.exists():
        for row in read_jsonl(failures_path):
            if row.get("intervention_id"):
                failed_ids.add(row["intervention_id"])

    all_policy_counts = {p: sum(i.get("policy") == p for i in interventions) for p in (POLICY_A, POLICY_B)}
    if all_policy_counts[POLICY_A] != all_policy_counts[POLICY_B]:
        print(f"WARNING: Intervention counts are imbalanced: {all_policy_counts}. Review intervention generation before full execution.")

    if not args.execute:
        remaining = [i for i in selected if i["intervention_id"] not in completed_ids]
        print("DRY RUN — no API calls will be made")
        print(f"Selected interventions: {len(selected)}")
        print(f"Already completed: {sum(i['intervention_id'] in completed_ids for i in selected)}")
        print(f"Pending: {len(remaining)}")
        print(f"Policies: { {p: sum(i.get('policy') == p for i in selected) for p in sorted(policy_filter)} }")
        print(f"Output directory: {out_dir}")
        print("To execute, rerun with --execute. Completed intervention IDs will be skipped.")
        return

    manifest = {
        "run_id": args.run_id,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "selected_count": len(selected),
        "policies": sorted(policy_filter),
        "limit": args.limit,
        "checkpoint_file": str(results_path),
        "resume_behavior": "skip IDs in replay_results.jsonl; failures are retried unless --retry-failed is false (default retries)",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # Configure BM25 only in execution mode; dry-run requires no retrieval dependencies.
    from src.document_processor import group_documents_by_question
    from src.retriever import configure_retriever
    configure_retriever(group_documents_by_question(all_documents))
    from src.counterfactual_replay import replay_policy_a, replay_policy_b
    pending = [i for i in selected if i["intervention_id"] not in completed_ids]
    print(f"EXECUTION MODE — API calls enabled. Pending: {len(pending)}")
    for idx, intervention in enumerate(pending, start=1):
        iid = intervention["intervention_id"]
        print(f"[{idx}/{len(pending)}] {intervention.get('policy')} | round {intervention.get('intervention_round')} | {iid}", flush=True)
        try:
            if intervention["policy"] == POLICY_A:
                result = replay_policy_a(intervention, selected_questions)
            elif intervention["policy"] == POLICY_B:
                result = replay_policy_b(intervention, clean_events, selected_questions)
            else:
                raise ValueError(f"Unknown policy: {intervention['policy']}")
            result["intervention_id"] = iid
            result["run_id"] = args.run_id
            result["intervention_round"] = intervention.get("intervention_round")
            result["status"] = "completed"
            write_jsonl_line(results_path, result)
            completed_ids.add(iid)
            print("  checkpoint saved", flush=True)
        except Exception as exc:
            failure = {
                "intervention_id": iid,
                "run_id": args.run_id,
                "question_id": intervention.get("question_id"),
                "policy": intervention.get("policy"),
                "intervention_round": intervention.get("intervention_round"),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            write_jsonl_line(failures_path, failure)
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue

    print(f"Finished. Completed this selection: {len(completed_ids)} IDs recorded.")
    print(f"Results: {results_path}")
    print(f"Failures: {failures_path}")


if __name__ == "__main__":
    main()
