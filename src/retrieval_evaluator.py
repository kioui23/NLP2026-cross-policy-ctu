
# CELL_9
# ADDITIONAL RETRIEVAL DIAGNOSTICS


k_values = [1, 2, 3, 5, 10]

recall_rows = []

for question in selected_questions:
    question_id = question["question_id"]

    for k in k_values:
        result = evaluate_retrieval(
            question_id=question_id,
            top_k=k
        )

        recall_rows.append(result)

recall_by_k = pd.DataFrame(recall_rows)

# Mean Recall@k across 10 questions
recall_summary = (
    recall_by_k
    .groupby("top_k")["recall"]
    .mean()
    .reset_index(name="mean_recall")
)

print("Mean Recall@k")
display(recall_summary)

# Per-question Recall@k
recall_pivot = recall_by_k.pivot(
    index="question_id",
    columns="top_k",
    values="recall"
)

print("Per-question Recall@k")
display(recall_pivot)

# Save diagnostic results
recall_by_k.to_csv(
    "hw2_results/bm25_recall_by_question.csv",
    index=False
)

recall_summary.to_csv(
    "hw2_results/bm25_recall_summary.csv",
    index=False
)

print("Saved retrieval diagnostic results.")
