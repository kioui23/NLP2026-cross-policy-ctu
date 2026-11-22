# Cross-Policy Counterfactual Document-Impact Analysis

This repository contains the experiment code and the data needed to reproduce the **exploratory Stage 6–8 analysis reported in the HW2 pilot** comparing direct iterative search and question-decomposition search.

## Scope and reported result

The exploratory analysis uses document-removal replay outputs from a 10-question HotpotQA validation pilot. The analysis uses **answer-string perturbation as a proxy**, not official Counterfactual Trajectory Utility (CTU) or correctness-based utility.

Reported pilot results:
- Successful replays: 107/108 (Direct: 60/60; Question decomposition: 47/48)
- Mean answer-perturbation proxy: Direct 0.3237; QD 0.4149
- Shared target documents: 31
- Spearman rank correlation over shared documents: 0.7453

These are descriptive exploratory results. They do not establish causal-utility transfer or generalization.

## Repository contents

- `src/`: experiment, replay, and analysis scripts (canonical Python files only; backup variants omitted).
- `analysis/`: reported exploratory summary and CSV outputs.
- `hw2_results/.../experiment_results.jsonl`: baseline answer records needed for analysis.
- `hw2_results/.../interventions.jsonl`: document-removal intervention records.
- `hw2_results/.../replay_results.jsonl`: successful counterfactual replay records.
- `requirements.txt`: dependencies used by the project.
- `.env.example`: placeholder for API-key configuration; never commit a real `.env`.

The `llm_calls.jsonl`, full trajectory logs, cached data, virtual environment, API credentials, and backup scripts are intentionally not included in this public reproduction package.

## Environment

The project was developed using Python 3.11.8.

Check your Python version:

\`\`\`bash
python3.11 --version
\`\`\`

Create and activate a virtual environment:

\`\`\`bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
\`\`\`

## Reproduce the reported exploratory analysis (no API calls)

From the repository root:

```bash
python src/stage6_8_exploratory_analysis.py
```

This script reads the baseline results, interventions, and replay results stored under `hw2_results/full_10_question_baseline/<run_id>/` and writes:

- `analysis/stage6_intervention_analysis.csv`
- `analysis/stage7_document_utility_proxy_rankings.csv`
- `analysis/stage6_8_exploratory_results.txt`

The analysis uses Python standard-library modules and does not make model/API calls. The input data for this specific pilot are included so the reported exploratory analysis can be rerun.

## Run a new baseline experiment (API calls required)

A new baseline run uses the HotpotQA dataset loader and Gemini API. It may incur API usage/costs and is not required to reproduce the exploratory analysis above.

1. Copy `.env.example` to `.env` and insert your own Google Gemini API key:

```bash
cp .env.example .env
```

2. Run the experiment from the repository root:

```bash
python -m src.experiment_runner
```

The experiment selects 10 questions from the HotpotQA validation split using seed 42 and saves run-specific outputs under `hw2_results/`.

## Important notes

- The exploratory analysis script is tied to the included pilot run ID and input directory structure.
- The full experiment runner creates a new run ID; it does not regenerate the exact included pilot run.
- The pilot uses Gemini 3.5 Flash-Lite, temperature 0, BM25 retrieval, up to two new documents per round, and a maximum of four retrieval rounds.
- The exploratory answer-perturbation proxy is based on normalized answer-string similarity. It is not answer F1, nDCG, or official CTU.
- A replay failure is recorded in the pilot: one Question decomposition intervention failed due to an invalid subquestion identifier.
- The effort-delta proxy was zero for both policies in this analysis and is not informative.

## Academic use

This is an exploratory course-project reproduction package. It should not be interpreted as a validated CTU implementation.
