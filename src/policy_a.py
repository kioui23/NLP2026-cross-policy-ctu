# POLICY A: DIRECT ITERATIVE SEARCH

from .llm_utils import generate_llm
from .trajectory_logger import log_trajectory

POLICY_A = "direct_iterative"

MAX_RETRIEVAL_ROUNDS = 4
RETRIEVAL_TOP_K = 2


def format_evidence(docs):
    """Format retrieved documents without gold labels."""

    formatted = []

    for doc in docs:
        formatted.append(
            f"""
Document ID: {doc['doc_id']}
Title: {doc['title']}
Content:
{doc['text']}
"""
        )

    return "\n".join(formatted)


def run_policy_a(
    question,
    question_id,
    retriever,
    start_round=1,
    initial_retrieved_documents=None
):
    """
    Direct iterative search.

    1. Generate a search query.
    2. Retrieve Top-2 documents.
    3. Use accumulated evidence to generate the next query.
    4. Repeat for up to four rounds.
    5. Generate a final answer.
    """

    retrieved_documents = [
        doc.copy()
        for doc in (initial_retrieved_documents or [])
    ]
    retrieved_doc_ids = {
        doc["doc_id"] for doc in retrieved_documents
    }
    round_records = []

    for round_num in range(start_round, MAX_RETRIEVAL_ROUNDS + 1):

        # 1. Generate search query
        if round_num == 1:

            query_prompt = f"""
You are a search agent answering a multi-hop question.

Original question:
{question}

Generate one concise search query that will help answer
the original question.

Return only the search query.
"""

        else:

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
            task="query_generation",
            question_id=question_id,
            policy=POLICY_A,
            max_output_tokens=128
        )

        query = query.strip()

        if not query:
            raise ValueError(
                f"Empty search query at round {round_num}"
            )

        log_trajectory(
            question_id=question_id,
            policy=POLICY_A,
            event_type="query_generation",
            round_num=round_num,
            query=query,
            action="Generate search query",
            call_id=call_id
        )

        # 2. BM25 retrieval
        new_docs = retriever.retrieve(
            query=query,
            question_id=question_id,
            top_k=RETRIEVAL_TOP_K,
            excluded=retrieved_doc_ids
        )

        if not new_docs:
            log_trajectory(
                question_id=question_id,
                policy=POLICY_A,
                event_type="retrieval",
                round_num=round_num,
                query=query,
                action="No unseen candidate documents remain"
            )
            break

        # 3. Update retrieved documents
        for doc in new_docs:
            retrieved_documents.append(doc)
            retrieved_doc_ids.add(doc["doc_id"])

        log_trajectory(
            question_id=question_id,
            policy=POLICY_A,
            event_type="retrieval",
            round_num=round_num,
            query=query,
            retrieved_docs=new_docs,
            action="Retrieve top-k documents"
        )

        # 4. Save round information
        round_records.append({
            "round": round_num,
            "query": query,
            "retrieved_doc_ids": [
                doc["doc_id"] for doc in new_docs
            ]
        })

    final_evidence = format_evidence(
        retrieved_documents
    )

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
        policy=POLICY_A,
        max_output_tokens=256
    )

    log_trajectory(
        question_id=question_id,
        policy=POLICY_A,
        event_type="final_answer",
        retrieved_docs=retrieved_documents,
        observation=final_answer,
        action="Generate final answer",
        call_id=final_call_id
    )

    return {
        "question_id": question_id,
        "policy": POLICY_A,
        "question": question,
        "answer": final_answer,
        "retrieved_documents": retrieved_documents,
        "round_records": round_records,
        "num_rounds": len(round_records),
        "num_unique_documents": len(retrieved_doc_ids)
    }
