
from src.execution_tracker import create_execution_id
from src.experiment_logger import EXPERIMENT_CONFIG
from src.llm_utils import generate_llm
from src.policy_a import POLICY_A, format_evidence
from src.policy_b import POLICY_B, parse_json_response
from src.retriever import retrieve
from src.trajectory_logger import log_trajectory


# ============================================================
# 1. Replay Configuration
# ============================================================

RETRIEVAL_TOP_K = EXPERIMENT_CONFIG["retrieval_top_k"]
MAX_RETRIEVAL_ROUNDS = EXPERIMENT_CONFIG["max_retrieval_rounds"]


# ============================================================
# 2. Shared Utilities
# ============================================================

def get_question_text(question_id, selected_questions):
    """Retrieve the original question text by question ID."""

    return next(
        q["question"]
        for q in selected_questions
        if q["question_id"] == question_id
    )


def initialize_replay_state(intervention):
    """
    Initialize the replay evidence state.

    Preserves:
    - Prefix documents
    - Same-round sibling documents

    Excludes:
    - Documents already retrieved
    - The target document removed by intervention
    """

    retrieved_documents = (
        list(intervention["prefix_documents"])
        + list(intervention["sibling_documents"])
    )

    retrieved_doc_ids = {
        doc["doc_id"]
        for doc in retrieved_documents
    }

    excluded_doc_ids = set(
        intervention["excluded_doc_ids"]
    )

    # Ensure the target document can never be retrieved again.
    excluded_doc_ids.add(
        intervention["target_doc_id"]
    )

    # Avoid retrieving documents that are already preserved.
    excluded_doc_ids.update(retrieved_doc_ids)

    return (
        retrieved_documents,
        retrieved_doc_ids,
        excluded_doc_ids,
    )


def retrieve_counterfactual_documents(
    query,
    question_id,
    excluded_doc_ids,
):
    """Retrieve new documents while excluding previously seen documents."""

    return retrieve(
        query=query,
        question_id=question_id,
        top_k=RETRIEVAL_TOP_K,
        excluded=excluded_doc_ids,
    )


def log_replay_event(
    intervention,
    execution_id,
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
):
    """Log a replay event with intervention metadata."""

    event_extra = {
        "intervention_id": intervention["intervention_id"],
        "target_doc_id": intervention["target_doc_id"],
    }

    if extra:
        event_extra.update(extra)

    return log_trajectory(
        question_id=intervention["question_id"],
        policy=policy,
        event_type=event_type,
        round_num=round_num,
        query=query,
        subquestion_id=subquestion_id,
        retrieved_docs=retrieved_docs,
        observation=observation,
        action=action,
        call_id=call_id,
        execution_id=execution_id,
        extra=event_extra,
    )


# ============================================================
# 3. Policy A: Direct Iterative Search Replay
# ============================================================

def replay_policy_a(
    intervention,
    selected_questions,
):
    """
    Replay Policy A from the round after intervention.

    Preserves the original direct iterative search strategy.
    """

    question_id = intervention["question_id"]
    execution_id = create_execution_id()

    question = get_question_text(
        question_id,
        selected_questions,
    )

    (
        retrieved_documents,
        retrieved_doc_ids,
        excluded_doc_ids,
    ) = initialize_replay_state(intervention)

    first_replay_round = (
        intervention["intervention_round"] + 1
    )

    first_replay_query = None
    round_records = []

    # --------------------------------------------------------
    # Replay search rounds
    # --------------------------------------------------------

    for round_num in range(
        first_replay_round,
        MAX_RETRIEVAL_ROUNDS + 1,
    ):

        evidence_text = format_evidence(
            retrieved_documents
        )

        query_prompt = f"""
You are a search agent answering a multi-hop question.

Original question:
{question}

Previously retrieved evidence:
{evidence_text}

Your task:
Identify what information is still missing.
Generate one new search query that can retrieve
additional evidence needed to answer the original question.

Avoid repeating information already retrieved.

Return only the search query.
"""

        query, call_id = generate_llm(
            prompt=query_prompt,
            policy=POLICY_A,
            max_output_tokens=128,
            execution_id=execution_id,
        )

        query = query.strip()

        if not query:
            raise ValueError(
                "Empty counterfactual query."
            )

        if first_replay_query is None:
            first_replay_query = query

        log_replay_event(
            intervention=intervention,
            execution_id=execution_id,
            policy=POLICY_A,
            event_type="query_generation",
            round_num=round_num,
            query=query,
            action="Counterfactual query generation",
            call_id=call_id,
        )

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        new_docs = retrieve_counterfactual_documents(
            query=query,
            question_id=question_id,
            excluded_doc_ids=excluded_doc_ids,
        )

        if not new_docs:

            log_replay_event(
                intervention=intervention,
                execution_id=execution_id,
                policy=POLICY_A,
                event_type="retrieval",
                round_num=round_num,
                query=query,
                action="No unseen candidate documents remain",
            )

            break

        retrieved_documents.extend(new_docs)

        for doc in new_docs:
            doc_id = doc["doc_id"]

            retrieved_doc_ids.add(doc_id)
            excluded_doc_ids.add(doc_id)

        log_replay_event(
            intervention=intervention,
            execution_id=execution_id,
            policy=POLICY_A,
            event_type="retrieval",
            round_num=round_num,
            query=query,
            retrieved_docs=new_docs,
            action="Counterfactual retrieval",
        )

        round_records.append({
            "round": round_num,
            "query": query,
            "retrieved_doc_ids": [
                doc["doc_id"]
                for doc in new_docs
            ],
        })

    # --------------------------------------------------------
    # Final answer generation
    # --------------------------------------------------------

    final_prompt = f"""
Answer the original question using the retrieved evidence.

Original question:
{question}

Retrieved evidence:
{format_evidence(retrieved_documents)}

Instructions:
- Use only the provided evidence.
- Do not invent unsupported facts.
- If the evidence is insufficient, state that clearly.
- Provide a concise answer.
"""

    answer, call_id = generate_llm(
        prompt=final_prompt,
        policy=POLICY_A,
        max_output_tokens=256,
        execution_id=execution_id,
    )

    log_replay_event(
        intervention=intervention,
        execution_id=execution_id,
        policy=POLICY_A,
        event_type="final_answer",
        retrieved_docs=retrieved_documents,
        observation=answer,
        action="Counterfactual final answer",
        call_id=call_id,
    )

    return {
        "intervention_id": intervention["intervention_id"],
        "execution_id": execution_id,
        "question_id": question_id,
        "policy": POLICY_A,
        "target_doc_id": intervention["target_doc_id"],
        "answer": answer,
        "first_replay_query": first_replay_query,
        "retrieved_documents": retrieved_documents,
        "round_records": round_records,
    }


# ============================================================
# 4. Policy B: Question Decomposition Replay
# ============================================================

def replay_policy_b(
    intervention,
    clean_events,
    selected_questions,
):
    """
    Replay Policy B while preserving the original decomposition.

    The original subquestions are reused.
    New question decomposition is not performed.
    """

    question_id = intervention["question_id"]
    execution_id = create_execution_id()

    question = get_question_text(
        question_id,
        selected_questions,
    )

    # Preserve the original decomposition.
    subquestions = intervention["decomposition"]

    if not subquestions:
        raise ValueError(
            "Missing baseline decomposition."
        )

    (
        retrieved_documents,
        retrieved_doc_ids,
        excluded_doc_ids,
    ) = initialize_replay_state(intervention)

    intervention_round = intervention["intervention_round"]

    # --------------------------------------------------------
    # Recover attempted subquestions before intervention
    # --------------------------------------------------------

    prior_events = [
        event
        for event in clean_events
        if event["question_id"] == question_id
        and event["policy"] == POLICY_B
        and event["event_type"] == "query_generation"
        and event.get("round") is not None
        and event["round"] <= intervention_round
    ]

    attempted_ids = list(dict.fromkeys(
        event["subquestion_id"]
        for event in sorted(
            prior_events,
            key=lambda x: x["round"],
        )
        if event.get("subquestion_id") is not None
    ))

    first_replay_query = None
    round_records = []

    # --------------------------------------------------------
    # Replay search rounds
    # --------------------------------------------------------

    for round_num in range(
        intervention_round + 1,
        MAX_RETRIEVAL_ROUNDS + 1,
    ):

        evidence_text = format_evidence(
            retrieved_documents
        )

        subquestion_text = "\n".join(
            f"ID {item['id']}: {item['question']}"
            for item in subquestions
        )

        prompt = f"""
You are a question-decomposition-based search agent.

Original question:
{question}

Decomposed subquestions:
{subquestion_text}

Previously attempted subquestion IDs:
{attempted_ids}

Previously retrieved evidence:
{evidence_text if evidence_text else "No evidence retrieved yet."}

Task:
1. Identify which information is still missing.
2. Select the most useful subquestion to investigate next.
3. Generate one concise search query for that subquestion.

Selection rules:
- Prefer a subquestion that has not been attempted yet.
- If all subquestions have been attempted, select the one
  that still needs stronger or more specific evidence.
- Use retrieved evidence to avoid redundant searches.
- Do not answer the original question.
- Return valid JSON only:

{{
  "selected_subquestion_id": 1,
  "query": "Concise search query"
}}
"""

        response_text, call_id = generate_llm(
            prompt=prompt,
            policy=POLICY_B,
            max_output_tokens=128,
            execution_id=execution_id,
        )

        parsed = parse_json_response(
            response_text
        )

        selected_id = parsed.get(
            "selected_subquestion_id"
        )

        raw_query = parsed.get("query")

        query = (
            raw_query.strip()
            if isinstance(raw_query, str)
            else ""
        )

        valid_ids = {
            item["id"]
            for item in subquestions
        }

        if selected_id not in valid_ids:
            raise ValueError(
                f"Invalid subquestion ID: {selected_id}"
            )

        selected_item = next(
            item
            for item in subquestions
            if item["id"] == selected_id
        )

        # Fallback to the selected subquestion itself.
        if not query:
            query = selected_item["question"].strip()

        if not query:
            raise ValueError(
                "Counterfactual query is empty."
            )

        attempted_ids.append(selected_id)

        if first_replay_query is None:
            first_replay_query = query

        log_replay_event(
            intervention=intervention,
            execution_id=execution_id,
            policy=POLICY_B,
            event_type="query_generation",
            round_num=round_num,
            query=query,
            subquestion_id=selected_id,
            action="Counterfactual subquestion selection",
            call_id=call_id,
            extra={
                "subquestion": selected_item["question"],
            },
        )

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        new_docs = retrieve_counterfactual_documents(
            query=query,
            question_id=question_id,
            excluded_doc_ids=excluded_doc_ids,
        )

        if not new_docs:

            log_replay_event(
                intervention=intervention,
                execution_id=execution_id,
                policy=POLICY_B,
                event_type="retrieval",
                round_num=round_num,
                query=query,
                subquestion_id=selected_id,
                action="No unseen candidate documents remain",
            )

            break

        retrieved_documents.extend(new_docs)

        for doc in new_docs:
            doc_id = doc["doc_id"]

            retrieved_doc_ids.add(doc_id)
            excluded_doc_ids.add(doc_id)

        log_replay_event(
            intervention=intervention,
            execution_id=execution_id,
            policy=POLICY_B,
            event_type="retrieval",
            round_num=round_num,
            query=query,
            subquestion_id=selected_id,
            retrieved_docs=new_docs,
            action="Counterfactual retrieval",
        )

        round_records.append({
            "round": round_num,
            "subquestion_id": selected_id,
            "query": query,
            "retrieved_doc_ids": [
                doc["doc_id"]
                for doc in new_docs
            ],
        })

    # --------------------------------------------------------
    # Final answer generation
    # --------------------------------------------------------

    final_prompt = f"""
Answer the original question using the retrieved evidence.

Original question:
{question}

Decomposed subquestions:
{subquestions}

Retrieved evidence:
{format_evidence(retrieved_documents)}

Instructions:
- Use only the provided evidence.
- Do not invent unsupported facts.
- If the evidence is insufficient, state that clearly.
- Provide a concise answer.
"""

    answer, call_id = generate_llm(
        prompt=final_prompt,
        policy=POLICY_B,
        max_output_tokens=256,
        execution_id=execution_id,
    )

    log_replay_event(
        intervention=intervention,
        execution_id=execution_id,
        policy=POLICY_B,
        event_type="final_answer",
        retrieved_docs=retrieved_documents,
        observation=answer,
        action="Counterfactual final answer",
        call_id=call_id,
    )

    return {
        "intervention_id": intervention["intervention_id"],
        "execution_id": execution_id,
        "question_id": question_id,
        "policy": POLICY_B,
        "target_doc_id": intervention["target_doc_id"],
        "answer": answer,
        "first_replay_query": first_replay_query,
        "retrieved_documents": retrieved_documents,
        "round_records": round_records,
    }


print("Counterfactual replay functions initialized.")
