# BM25 RETRIEVAL

import re
from rank_bm25 import BM25Okapi


def tokenize(text):
    """Tokenize text using lowercase alphanumeric tokens."""
    return re.findall(r"\w+", str(text).lower())


class Retriever:
    """BM25 retriever with explicit document data and per-question index caching."""

    def __init__(self, documents_by_question):
        self.documents_by_question = documents_by_question
        self.bm25_indexes = {}

    def get_bm25_index(self, question_id):
        """Create or retrieve a BM25 index for a specific question."""
        if question_id not in self.bm25_indexes:
            question_docs = self.documents_by_question[question_id]

            tokenized_docs = [
                tokenize(doc["text"])
                for doc in question_docs
            ]

            self.bm25_indexes[question_id] = BM25Okapi(
                tokenized_docs
            )

        return self.bm25_indexes[question_id]

    def omit_document(self, retrieved_docs, omitted_doc_id):
        """
        Remove one document from an existing retrieval result.

        This is a counterfactual document omission intervention:
        remaining documents are preserved in their original order,
        and no replacement documents are retrieved.
        """
        if not any(
            doc.get("doc_id") == omitted_doc_id
            for doc in retrieved_docs
        ):
            raise ValueError(
                f"Document not found in retrieval result: {omitted_doc_id}"
            )

        return [
            doc.copy()
            for doc in retrieved_docs
            if doc.get("doc_id") != omitted_doc_id
        ]

    def retrieve(
        self,
        query,
        question_id,
        top_k=2,
        excluded=None,
    ):
        """Retrieve top-k documents using BM25."""

        excluded = excluded or set()

        question_docs = self.documents_by_question[question_id]
        bm25 = self.get_bm25_index(question_id)

        tokenized_query = tokenize(query)
        scores = bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(question_docs)),
            key=lambda idx: scores[idx],
            reverse=True,
        )

        retrieved_docs = []

        for idx in ranked_indices:
            doc = question_docs[idx]

            if doc["doc_id"] in excluded:
                continue

            retrieved_doc = doc.copy()
            retrieved_doc["score"] = float(scores[idx])
            retrieved_docs.append(retrieved_doc)

            if len(retrieved_docs) >= top_k:
                break

        return retrieved_docs


# ------------------------------------------------------------
# Legacy compatibility layer
# This will be removed after all callers use Retriever instances.
# ------------------------------------------------------------

bm25_indexes = {}
documents_by_question = None
_legacy_retriever = None


def configure_retriever(documents_by_question):
    """Configure the legacy module-level retrieval interface."""
    global _legacy_retriever, bm25_indexes

    _legacy_retriever = Retriever(documents_by_question)
    bm25_indexes = _legacy_retriever.bm25_indexes


def get_bm25_index(question_id):
    if _legacy_retriever is None:
        raise RuntimeError(
            "Legacy retriever is not configured. "
            "Call configure_retriever(documents_by_question) first."
        )

    return _legacy_retriever.get_bm25_index(question_id)


def retrieve(
    query,
    question_id,
    top_k=2,
    excluded=None,
):
    if _legacy_retriever is None:
        raise RuntimeError(
            "Legacy retriever is not configured. "
            "Call configure_retriever(documents_by_question) first."
        )

    return _legacy_retriever.retrieve(
        query=query,
        question_id=question_id,
        top_k=top_k,
        excluded=excluded,
    )
