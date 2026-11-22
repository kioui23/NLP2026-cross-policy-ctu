# DOCUMENT EXTRACTION

from collections import defaultdict
import json
from pathlib import Path


def build_documents(example):
    """Extract candidate documents for one HotpotQA question."""

    question_id = example["id"]

    titles = example["context"]["title"]
    sentences = example["context"]["sentences"]

    supporting_titles = set(
        example["supporting_facts"]["title"]
    )

    supporting_sentence_ids = defaultdict(set)

    for title, sent_id in zip(
        example["supporting_facts"]["title"],
        example["supporting_facts"]["sent_id"],
    ):
        supporting_sentence_ids[title].add(sent_id)

    documents = []

    for title, sentence_list in zip(titles, sentences):
        supporting_ids = sorted(
            supporting_sentence_ids.get(title, set())
        )

        supporting_sentences = [
            sentence_list[idx]
            for idx in supporting_ids
            if 0 <= idx < len(sentence_list)
        ]

        document = {
            "question_id": question_id,
            "doc_id": f"{question_id}::{title}",
            "title": title,
            "text": " ".join(sentence_list),
            "sentences": sentence_list,
            "is_supporting": title in supporting_titles,
            "supporting_sentences": supporting_sentences,
        }

        documents.append(document)

    return documents


def extract_documents(experiment_data):
    """Extract candidate documents for all selected questions."""

    all_documents = []

    for example in experiment_data:
        question_documents = build_documents(example)
        all_documents.extend(question_documents)

    return all_documents


def validate_documents(experiment_data, all_documents):
    """Validate extracted documents."""

    question_ids = {
        item["id"]
        for item in experiment_data
    }

    assert len(question_ids) == len(experiment_data), (
        "Duplicate question IDs found."
    )

    assert all(
        doc["question_id"] in question_ids
        for doc in all_documents
    ), "Unknown question ID found in documents."

    doc_ids = [
        doc["doc_id"]
        for doc in all_documents
    ]

    assert len(doc_ids) == len(set(doc_ids)), (
        "Duplicate document IDs found."
    )

    return True


def group_documents_by_question(all_documents):
    """Group documents by question ID."""

    documents_by_question = defaultdict(list)

    for doc in all_documents:
        documents_by_question[doc["question_id"]].append(doc)

    return documents_by_question


def print_document_statistics(experiment_data, all_documents):
    """Print document extraction statistics."""

    documents_by_question = group_documents_by_question(
        all_documents
    )

    print("Document extraction completed.")
    print("Total questions:", len(experiment_data))
    print("Total documents:", len(all_documents))

    print("\nDocuments per question:")

    for question_id, docs in documents_by_question.items():
        print(question_id, ":", len(docs))

    supporting_count = sum(
        doc["is_supporting"]
        for doc in all_documents
    )

    print("\nSupporting documents:", supporting_count)
    print(
        "Distractor documents:",
        len(all_documents) - supporting_count,
    )

    return documents_by_question


def save_documents(all_documents, output_dir):
    """Save extracted documents to JSON."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    documents_path = output_dir / "hotpotqa_documents.json"

    with open(documents_path, "w", encoding="utf-8") as f:
        json.dump(
            all_documents,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nSaved:", documents_path)

    return documents_path


def preview_documents(documents_by_question, question_id):
    """Preview documents belonging to one question."""

    print("\nSample documents:")

    for doc in documents_by_question[question_id]:
        print("\nTitle:", doc["title"])
        print("Document ID:", doc["doc_id"])
        print("Supporting:", doc["is_supporting"])
        print("Text preview:", doc["text"][:250])


if __name__ == "__main__":
    from .data_loader import load_hotpotqa

    experiment_data, selected_questions, selected_indices = (
        load_hotpotqa()
    )

    all_documents = extract_documents(experiment_data)

    validate_documents(
        experiment_data,
        all_documents,
    )

    documents_by_question = print_document_statistics(
        experiment_data,
        all_documents,
    )

    save_documents(
        all_documents,
        Path("hw2_results"),
    )

    sample_qid = experiment_data[0]["id"]

    preview_documents(
        documents_by_question,
        sample_qid,
    )


def format_evidence(docs):
    """Format retrieved documents for LLM prompts."""

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
