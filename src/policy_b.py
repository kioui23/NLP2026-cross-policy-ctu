# Policy B: Question Decomposition-based Search
# CELL_21 + CELL_22 + CELL_23

import json
import re

from .llm_utils import generate_llm
from .document_processor import format_evidence
from .trajectory_logger import log_trajectory
from .experiment_logger import EXPERIMENT_CONFIG



# Configuration


POLICY_B = "question_decomposition"

MAX_RETRIEVAL_ROUNDS = EXPERIMENT_CONFIG["max_retrieval_rounds"]
RETRIEVAL_TOP_K = EXPERIMENT_CONFIG["retrieval_top_k"]



# JSON Response Parser


def parse_json_response(response_text):
    """
    Parse JSON response from LLM.
    Handles responses wrapped in Markdown code fences.
    """

    if not response_text:
        raise ValueError("LLM returned an empty response.")

    cleaned = response_text.strip()

    # Remove Markdown code fences
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned
    )

    return json.loads(cleaned)



# CELL_21: Question Decomposition


def decompose_question(question, question_id=None):
    """
    Decompose a complex question into independently searchable
    subquestions.
    """

    prompt = f"""
You are a question decomposition assistant.

Given a question, decompose it into up to 4 subquestions
that can be independently searched and answered.

Each subquestion should:
- Focus on one piece of information.
- Be independently searchable.
- Help answer the original question.

Return your response in the following JSON format:

{{
    "subquestions": [
        {{
            "id": 1,
            "question": "Subquestion 1"
        }},
        {{
            "id": 2,
            "question": "Subquestion 2"
        }}
    ]
}}

Original question:
{question}
"""

    response_text, call_id = generate_llm(
        prompt=prompt,
        task="question_decomposition",
        question_id=question_id,
        policy=POLICY_B,
        max_output_tokens=256
    )

    parsed = parse_json_response(response_text)

    subquestions = parsed.get("subquestions", [])

    # Validate and normalize subquestions
    normalized = []

    for item in subquestions:

        if not isinstance(item, dict):
            continue

        subquestion_id = item.get("id")
        subquestion_text = item.get("question", "").strip()

        if subquestion_id is None or not subquestion_text:
            continue

        normalized.append({
            "id": subquestion_id,
            "question": subquestion_text
        })

    if not normalized:
        raise ValueError(
            f"Question decomposition failed for question_id={question_id}"
        )

    # Log decomposition
    log_trajectory(
        question_id=question_id,
        policy=POLICY_B,
        event_type="question_decomposition",
        observation=question,
        action=normalized,
        call_id=call_id,
        extra={
            "num_subquestions": len(normalized)
        }
    )

    return normalized



# CELL_22: Select Subquestion and Generate Search Query


def select_subquestion_and_query(
    question,
    subquestions,
    retrieved_documents,
    attempted_subquestion_ids,
    question_id=None,
    round_num=None
):
    """
    Select the most useful unanswered subquestion and generate
    a search query.
    """

    # Format previously retrieved evidence
    evidence_text = format_evidence(retrieved_documents)

    subquestion_text = "\n".join(
        [
            f"{item['id']}. {item['question']}"
            for item in subquestions
        ]
    )

    attempted_text = (
        str(attempted_subquestion_ids)
        if attempted_subquestion_ids
        else "None"
    )

    prompt = f"""
You are a search agent answering a complex question.

Your task is to:
1. Select the most useful subquestion to investigate next.
2. Generate a search query to retrieve relevant information.

Original question:
{question}

Subquestions:
{subquestion_text}

Previously attempted subquestion IDs:
{attempted_text}

Previously retrieved evidence:
{evidence_text}

Choose a subquestion that is likely to provide the most useful
new information for answering the original question.

Avoid selecting a subquestion that has already been attempted
unless it is necessary.

Return your response in the following JSON format:

{{
    "selected_subquestion_id": 1,
    "query": "Search query"
}}
"""

    response_text, call_id = generate_llm(
        prompt=prompt,
        task="subquestion_selection_and_query_generation",
        question_id=question_id,
        policy=POLICY_B,
        max_output_tokens=256
    )

    parsed = parse_json_response(response_text)

    selected_id = parsed.get("selected_subquestion_id")
    query = parsed.get("query", "").strip()

    # Validate selected subquestion
    selected_item = next(
        (
            item
            for item in subquestions
            if item["id"] == selected_id
        ),
        None
    )

    if selected_item is None:
        raise ValueError(
            f"Invalid selected subquestion ID: {selected_id}"
        )

    if not query:
        raise ValueError("LLM generated an empty search query.")

    return selected_item, query, call_id



# CELL_23: Policy B Execution


def run_policy_b(
    question,
    question_id=None,
    retriever=None,
    start_round=1,
    initial_subquestions=None,
    initial_retrieved_documents=None,
    initial_attempted_subquestion_ids=None
):
    """
    Execute Policy B.

    Policy B:
    1. Decompose the original question.
    2. Select a subquestion.
    3. Generate a search query.
    4. Retrieve documents.
    5. Repeat until the retrieval limit is reached.
    6. Generate the final answer using accumulated evidence.
    """


    # Step 1. Question decomposition


    # Restore decomposition state when replaying
    if initial_subquestions is None:
        subquestions = decompose_question(
            question=question,
            question_id=question_id
        )
    else:
        subquestions = [item.copy() for item in initial_subquestions]

    # Restore evidence state
    retrieved_documents = [
        doc.copy()
        for doc in (initial_retrieved_documents or [])
    ]

    retrieved_doc_ids = {
        doc["doc_id"]
        for doc in retrieved_documents
    }

    # Restore attempted subquestion state
    attempted_subquestion_ids = list(
        initial_attempted_subquestion_ids or []
    )

    round_records = []


    # Step 2. Iterative search


    for round_num in range(
        start_round,
        MAX_RETRIEVAL_ROUNDS + 1
    ):

        # Select subquestion and generate query
        selected_item, query, call_id = (
            select_subquestion_and_query(
                question=question,
                subquestions=subquestions,
                retrieved_documents=retrieved_documents,
                attempted_subquestion_ids=attempted_subquestion_ids,
                question_id=question_id,
                round_num=round_num
            )
        )

        subquestion_id = selected_item["id"]

        attempted_subquestion_ids.append(
            subquestion_id
        )

        # Log query generation
        log_trajectory(
            question_id=question_id,
            policy=POLICY_B,
            event_type="query_generation",
            observation={
                "original_question": question,
                "selected_subquestion": selected_item,
                "retrieved_document_count": len(
                    retrieved_documents
                )
            },
            action={
                "query": query,
                "subquestion_id": subquestion_id
            },
            call_id=call_id,
            extra={
                "round": round_num
            }
        )


        # Retrieval


        new_docs = retriever.retrieve(
            query=query,
            question_id=question_id,
            top_k=RETRIEVAL_TOP_K,
            excluded=retrieved_doc_ids
        )

        # Stop if no new documents are retrieved
        if not new_docs:

            log_trajectory(
                question_id=question_id,
                policy=POLICY_B,
                event_type="retrieval",
                observation={
                    "query": query,
                    "round": round_num
                },
                action={
                    "new_documents": [],
                    "stop_reason": "no_new_documents"
                },
                extra={
                    "num_new_documents": 0
                }
            )

            break


        # Update retrieved documents


        for doc in new_docs:

            doc_id = doc["doc_id"]

            if doc_id not in retrieved_doc_ids:

                retrieved_doc_ids.add(doc_id)

                retrieved_documents.append(doc)

        # Record successful retrieval round
        round_records.append({
            "round": round_num,
            "subquestion_id": subquestion_id,
            "query": query,
            "num_new_documents": len(new_docs)
        })


        # Log retrieval


        log_trajectory(
            question_id=question_id,
            policy=POLICY_B,
            event_type="retrieval",
            observation={
                "query": query,
                "round": round_num
            },
            action={
                "new_documents": [
                    doc["doc_id"]
                    for doc in new_docs
                ]
            },
            extra={
                "num_new_documents": len(new_docs),
                "total_unique_documents": len(
                    retrieved_documents
                )
            }
        )


    # Step 3. Format accumulated evidence


    final_evidence = format_evidence(
        retrieved_documents
    )


    # Step 4. Generate final answer


    final_prompt = f"""
Answer the original question using the retrieved evidence.

Original question:
{question}

Retrieved evidence:
{final_evidence}

Instructions:
- Use only the provided evidence.
- Do not invent unsupported facts.
- If the evidence is insufficient, state that clearly.
- Provide a concise answer.
"""

    final_answer, final_call_id = generate_llm(
        prompt=final_prompt,
        task="final_answer",
        question_id=question_id,
        policy=POLICY_B,
        max_output_tokens=256
    )


    # Step 5. Log final answer


    log_trajectory(
        question_id=question_id,
        policy=POLICY_B,
        event_type="final_answer",
        observation={
            "question": question,
            "retrieved_documents": len(
                retrieved_documents
            )
        },
        action={
            "answer": final_answer
        },
        call_id=final_call_id,
        extra={
            "num_rounds": len(round_records),
            "num_unique_documents": len(
                retrieved_documents
            )
        }
    )


    # Step 6. Return experiment result


    return {
        "question_id": question_id,
        "policy": POLICY_B,
        "question": question,
        "answer": final_answer,
        "subquestions": subquestions,
        "retrieved_documents": retrieved_documents,
        "num_rounds": len(round_records),
        "num_unique_documents": len(
            retrieved_documents
        ),
        "round_records": round_records
    }