
# Full Experiment Runner
# CELL_24 + CELL_25

import json
import re
import time

import pandas as pd

from .experiment_logger import (
    RESULTS_DIR,
    RUN_ID,
    EXPERIMENT_CONFIG,
    utc_now,
)
from .llm_utils import (
    GEMINI_MODEL,
    TEMPERATURE,
)
from .policy_a import run_policy_a
from .policy_b import run_policy_b
from .retriever import Retriever
from .document_processor import group_documents_by_question



# Experiment Configuration


EXPERIMENT_NAME = "full_10_question_baseline"

RETRIEVAL_TOP_K = EXPERIMENT_CONFIG["retrieval_top_k"]
MAX_RETRIEVAL_ROUNDS = EXPERIMENT_CONFIG["max_retrieval_rounds"]
MAX_UNIQUE_DOCUMENTS = 8



# Experiment Output Paths


RUN_DIR = RESULTS_DIR / EXPERIMENT_NAME / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

LLM_LOG_PATH = RESULTS_DIR / "llm_calls.jsonl"
TRAJECTORY_LOG_PATH = RESULTS_DIR / "trajectory_log.jsonl"

RESULTS_PATH = RUN_DIR / "experiment_results.jsonl"
SUMMARY_PATH = RUN_DIR / "experiment_summary.csv"
MANIFEST_PATH = RUN_DIR / "experiment_manifest.json"



# Evaluation


def normalize_answer(text):
    """HotpotQA-style Exact Match normalization."""

    text = str(text).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)

    return " ".join(text.split())


def evaluate_result(
    result,
    gold_answers,
    supporting_doc_ids,
):
    """Evaluate Exact Match and supporting document recall."""

    question_id = result["question_id"]

    retrieved_ids = {
        doc["doc_id"]
        for doc in result.get("retrieved_documents", [])
    }

    question_supporting_ids = {
        doc_id
        for doc_id in supporting_doc_ids
        if doc_id.startswith(question_id + "::")
    }

    retrieved_supporting_ids = (
        retrieved_ids & question_supporting_ids
    )

    gold = gold_answers[question_id]
    predicted = result.get("answer", "")

    return {
        "exact_match": int(
            normalize_answer(predicted)
            == normalize_answer(gold)
        ),
        "supporting_retrieved": len(
            retrieved_supporting_ids
        ),
        "supporting_total": len(
            question_supporting_ids
        ),
        "supporting_recall": (
            len(retrieved_supporting_ids)
            / len(question_supporting_ids)
            if question_supporting_ids
            else None
        ),
    }



# Result Saving


def save_experiment_record(record):
    """Append one experiment record to JSONL."""

    with open(
        RESULTS_PATH,
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )


def save_summary(records):
    """Save experiment progress to CSV."""

    pd.DataFrame(records).to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )



# Full Experiment Runner


def run_full_experiment(
    selected_questions,
    all_documents,
):
    """
    Run Policy A and Policy B on 10 selected questions.

    Total expected runs:
        10 questions x 2 policies = 20 runs

    Failed runs are recorded without stopping the experiment.
    """


    # Validate inputs


    if len(selected_questions) != 10:
        raise ValueError(
            f"Expected 10 questions, "
            f"got {len(selected_questions)}"
        )

    question_ids = {
        q["question_id"]
        for q in selected_questions
    }

    if len(question_ids) != 10:
        raise ValueError(
            "Question IDs must be unique."
        )

    if RETRIEVAL_TOP_K != 2:
        raise ValueError(
            f"Expected top_k=2, "
            f"got {RETRIEVAL_TOP_K}"
        )

    if MAX_RETRIEVAL_ROUNDS != 4:
        raise ValueError(
            f"Expected max_rounds=4, "
            f"got {MAX_RETRIEVAL_ROUNDS}"
        )


    # Supporting documents and gold answers


    supporting_doc_ids = {
        doc["doc_id"]
        for doc in all_documents
        if doc.get("is_supporting", False)
    }

    gold_answers = {
        item["question_id"]: item["answer"]
        for item in selected_questions
    }


    # Shared Retriever
    documents_by_question = group_documents_by_question(all_documents)
    retriever = Retriever(documents_by_question)

    # Policies


    policies = [
        ("direct_iterative", run_policy_a),
        ("question_decomposition", run_policy_b),
    ]

    experiment_records = []


    # Experiment manifest


    experiment_manifest = {
        "run_id": RUN_ID,
        "experiment": EXPERIMENT_NAME,
        "dataset": "HotpotQA",
        "split": "validation",
        "num_questions": len(selected_questions),
        "policies": [
            policy_name
            for policy_name, _ in policies
        ],
        "model": GEMINI_MODEL,
        "temperature": TEMPERATURE,
        "retriever": "BM25",
        "top_k": RETRIEVAL_TOP_K,
        "max_rounds": MAX_RETRIEVAL_ROUNDS,
        "max_unique_documents": MAX_UNIQUE_DOCUMENTS,
        "seed": 42,
        "created_at": utc_now(),
        "results_path": str(RESULTS_PATH),
        "summary_path": str(SUMMARY_PATH),
        "llm_log_path": str(LLM_LOG_PATH),
        "trajectory_log_path": str(TRAJECTORY_LOG_PATH),
    }

    with open(
        MANIFEST_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            experiment_manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )


    # Experiment setup


    print("Full experiment setup: PASS")
    print("RUN_ID:", RUN_ID)
    print("Questions:", len(selected_questions))
    print("Policies:", experiment_manifest["policies"])
    print("Model:", GEMINI_MODEL)
    print("Top-k:", RETRIEVAL_TOP_K)
    print("Max rounds:", MAX_RETRIEVAL_ROUNDS)
    print("Output directory:", RUN_DIR)


    # Main experiment loop


    for question_index, question_item in enumerate(
        selected_questions,
        start=1,
    ):

        question_id = question_item["question_id"]
        question = question_item["question"]

        print("\n" + "=" * 70)
        print(
            f"Question {question_index}/10: "
            f"{question_id}"
        )
        print(question)

        for policy_name, policy_function in policies:

            print(
                f"\nRunning policy: {policy_name}"
            )

            start_time = time.perf_counter()

            try:

                # Execute policy
                result = policy_function(
                    question=question,
                    question_id=question_id,
                    retriever=retriever,
                )

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                # Evaluate
                evaluation = evaluate_result(
                    result=result,
                    gold_answers=gold_answers,
                    supporting_doc_ids=supporting_doc_ids,
                )

                # Create experiment record
                record = {
                    "run_id": RUN_ID,
                    "question_id": question_id,
                    "policy": policy_name,
                    "status": "success",
                    "question": question,
                    "answer": result.get("answer"),
                    "num_rounds": result.get("num_rounds"),
                    "num_unique_documents": result.get(
                        "num_unique_documents"
                    ),
                    "retrieved_doc_ids": [
                        doc["doc_id"]
                        for doc in result.get(
                            "retrieved_documents",
                            [],
                        )
                    ],
                    "round_records": result.get(
                        "round_records",
                    ),
                    "subquestions": result.get(
                        "subquestions",
                    ),
                    "elapsed_seconds": elapsed,
                    **evaluation,
                }

                # Save result
                save_experiment_record(record)
                experiment_records.append(record)

                print("Status: SUCCESS")
                print("Rounds:", record["num_rounds"])
                print(
                    "Unique documents:",
                    record["num_unique_documents"],
                )
                print(
                    "Supporting recall:",
                    record["supporting_recall"],
                )
                print(
                    "Exact match:",
                    record["exact_match"],
                )
                print(
                    f"Elapsed: {elapsed:.2f} sec"
                )

            except Exception as e:

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                record = {
                    "run_id": RUN_ID,
                    "question_id": question_id,
                    "policy": policy_name,
                    "status": "failed",
                    "question": question,
                    "error": repr(e),
                    "elapsed_seconds": elapsed,
                }

                save_experiment_record(record)
                experiment_records.append(record)

                print("Status: FAILED")
                print("Error:", repr(e))
                print(
                    "Continuing to the next "
                    "policy/question..."
                )


        # Save intermediate progress


        save_summary(experiment_records)

        success_count = sum(
            r["status"] == "success"
            for r in experiment_records
        )

        failure_count = sum(
            r["status"] == "failed"
            for r in experiment_records
        )

        print("\nCurrent progress:")
        print(
            "Completed runs:",
            len(experiment_records),
            "/ 20",
        )
        print("Success:", success_count)
        print("Failure:", failure_count)


    # Final summary


    save_summary(experiment_records)

    success_count = sum(
        r["status"] == "success"
        for r in experiment_records
    )

    failure_count = sum(
        r["status"] == "failed"
        for r in experiment_records
    )

    print("\n" + "=" * 70)
    print("FULL EXPERIMENT FINISHED")
    print("Total runs:", len(experiment_records))
    print("Success:", success_count)
    print("Failure:", failure_count)
    print("Results:", RESULTS_PATH)
    print("Summary:", SUMMARY_PATH)
    print("LLM logs:", LLM_LOG_PATH)
    print("Trajectory logs:", TRAJECTORY_LOG_PATH)

    return experiment_records


if __name__ == "__main__":
    from src.data_loader import load_hotpotqa
    from src.document_processor import extract_documents

    experiment_data, selected_questions, selected_indices = load_hotpotqa(
        seed=42,
        num_questions=10,
    )
    all_documents = extract_documents(experiment_data)

    run_full_experiment(selected_questions, all_documents)
