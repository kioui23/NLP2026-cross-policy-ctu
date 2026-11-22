# Trajectory Event Logger
# CELL_16 + CELL_36

import uuid

from .experiment_logger import (
    RUN_ID,
    append_jsonl,
    utc_now,
)
from .execution_tracker import BASELINE_EXECUTION_ID


trajectory_logs = []


def log_trajectory(
    question_id,
    policy,
    event_type,
    round_num=None,
    query=None,
    subquestion_id=None,
    retrieved_docs=None,
    observation=None,
    action=None,
    call_id=None,
    extra=None,
    execution_id=BASELINE_EXECUTION_ID,
):
    """
    Save one search-agent event.

    execution_id separates baseline and counterfactual executions.
    """

    event = {
        "run_id": RUN_ID,
        "execution_id": execution_id,
        "event_id": str(uuid.uuid4()),
        "question_id": question_id,
        "policy": policy,
        "event_type": event_type,
        "round": round_num,
        "query": query,
        "subquestion_id": subquestion_id,

        # Store document IDs and retrieval scores.
        # Gold labels are deliberately excluded.
        "retrieved_docs": [
            {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "score": doc.get("score"),
            }
            for doc in (retrieved_docs or [])
        ],

        "observation": observation,
        "action": action,

        # Link this event to its corresponding LLM call.
        "call_id": call_id,

        "extra": extra or {},
        "timestamp": utc_now(),
    }

    append_jsonl(event, "trajectory_log.jsonl")
    trajectory_logs.append(event)

    return event["event_id"]


print("Execution-aware trajectory logger created.")