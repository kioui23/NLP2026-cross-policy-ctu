# HOTPOTQA DATA LOADING

from datasets import load_dataset
from pathlib import Path
import random
import json


# Experiment configuration
SEED = 42
NUM_QUESTIONS = 10

DATASET_NAME = "hotpotqa/hotpot_qa"
DATASET_CONFIG = "distractor"
DATASET_SPLIT = "validation"


def load_hotpotqa(
    seed=SEED,
    num_questions=NUM_QUESTIONS,
    dataset_name=DATASET_NAME,
    dataset_config=DATASET_CONFIG,
    dataset_split=DATASET_SPLIT,
    output_dir=None,
):
    """
    Load HotpotQA and reproducibly select questions.

    Returns:
        experiment_data: HuggingFace Dataset containing selected examples
        selected_questions: question-level metadata
        selected_indices: indices selected from the original dataset
    """

    print("Loading HotpotQA...")

    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=dataset_split,
    )

    print("Dataset loaded successfully.")
    print("Dataset size:", len(dataset))
    print("Available columns:", dataset.column_names)

    # Reproducible question selection
    rng = random.Random(seed)

    selected_indices = sorted(
        rng.sample(
            range(len(dataset)),
            num_questions,
        )
    )

    experiment_data = dataset.select(selected_indices)

    print("\nSelected questions:", len(experiment_data))
    print("Selected indices:", selected_indices)

    # Extract question-level information
    selected_questions = []

    for item in experiment_data:
        question_record = {
            "question_id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "type": item["type"],
            "level": item["level"],
            "supporting_facts": [
                {
                    "title": title,
                    "sentence_id": sentence_id,
                }
                for title, sentence_id in zip(
                    item["supporting_facts"]["title"],
                    item["supporting_facts"]["sent_id"],
                )
            ],
        }

        selected_questions.append(question_record)

    # Save selected question metadata only when requested
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        questions_path = output_dir / "hotpotqa_selected_questions.json"

        with open(questions_path, "w", encoding="utf-8") as f:
            json.dump(
                selected_questions,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print("\nSaved:", questions_path)

    return experiment_data, selected_questions, selected_indices


def preview_questions(selected_questions):
    """Print a preview of selected questions."""

    for i, item in enumerate(selected_questions, 1):
        print(f"\nQuestion {i}")
        print("ID:", item["question_id"])
        print("Question:", item["question"])
        print("Answer:", item["answer"])
        print("Type:", item["type"])
        print("Level:", item["level"])

        print(
            "Supporting documents:",
            sorted({
                fact["title"]
                for fact in item["supporting_facts"]
            }),
        )


if __name__ == "__main__":
    experiment_data, selected_questions, selected_indices = load_hotpotqa(
        output_dir=Path("hw2_results")
    )

    preview_questions(selected_questions)
