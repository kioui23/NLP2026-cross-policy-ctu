
# Cross-Policy Counterfactual Document-Impact Analysis

## 1. Environment Setup

### Requirements

- Python 3.11.8
- Google Gemini API key

### Clone the repository

```bash
git clone https://github.com/kioui23/NLP2026-cross-policy-ctu.git
cd NLP2026-cross-policy-ctu
```

### Create and activate a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Configure the API

Copy the environment template:

```bash
cp .env.example .env
```

Open `.env` and add your Gemini API key:

```text
GOOGLE_API_KEY=your_google_api_key_here
```

Do not commit `.env` or share your API key.

## 3. Prepare the Dataset

### Select HotpotQA questions

Run:

```bash
python -m src.data_loader
```

This generates:

```text
hw2_results/hotpotqa_selected_questions.json
```

### Extract documents

Run:

```bash
python -m src.document_processor
```

This generates:

```text
hw2_results/hotpotqa_documents.json
```

These files are used by the baseline experiment and counterfactual pipeline.

## 4. Run the Baseline Experiment

Run both search policies on the selected questions:

```bash
python -m src.experiment_runner
```

The experiment schedules 10 questions for each policy (20 runs total). A new run ID is generated for each execution and printed in the console. The run ID is also recorded in `experiment_manifest.json`.

Baseline outputs are saved under:

```text
hw2_results/
└── full_10_question_baseline/
    └── [RUN_ID]/
        ├── experiment_results.jsonl
        ├── experiment_summary.csv
        └── experiment_manifest.json
```

Shared logs are also appended to:

```text
hw2_results/llm_calls.jsonl
hw2_results/trajectory_log.jsonl
```

Some runs may fail. Check successful and failed runs in the output before proceeding. Save the generated run ID for the following steps.

## 5. Prepare Counterfactual Interventions

Replace `[RUN_ID]` with the run ID from the baseline experiment:

```bash
python src/ctu_prepare.py --run-id [RUN_ID]
```

This step does not call the LLM API. It loads the baseline trajectories for the selected run, validates trajectory pairs, excludes incomplete pairs, and generates document-removal interventions.

Outputs:

```text
hw2_results/full_10_question_baseline/[RUN_ID]/
└── ctu_preparation/
    ├── clean_trajectories.jsonl
    ├── clean_trajectory_validation.json
    └── interventions.jsonl
```

Intervention counts may differ between policies because only validated baseline trajectory pairs are used.

## 6. Inspect Counterfactual Replay (Dry Run)

Run the replay runner without `--execute` to inspect the planned interventions:

```bash
python -m src.counterfactual_runner --run-id [RUN_ID]
```

The default mode is dry-run. It reports the selected intervention count, completed and pending interventions, counts by policy, and output directory. No LLM API calls are made.

Confirm that the run ID and output directory are correct before executing replay.

## 7. Execute Counterfactual Replay

Counterfactual replay uses LLM API calls and may incur API costs.

```bash
python -m src.counterfactual_runner --run-id [RUN_ID] --execute
```

Results are checkpointed as interventions finish:

```text
hw2_results/full_10_question_baseline/[RUN_ID]/
└── ctu_preparation/
    └── counterfactual_replay/
        ├── replay_results.jsonl
        ├── failures.jsonl
        └── replay_manifest.json
```

Completed intervention IDs are skipped when the script is resumed. Failed interventions are recorded separately and may be retried.

### Run a limited replay

Use `--limit` to execute a small number of interventions:

```bash
python -m src.counterfactual_runner --run-id [RUN_ID] --limit 1 --execute
```

### Run a limited replay for one policy

```bash
python -m src.counterfactual_runner --run-id [RUN_ID] --policy direct_iterative --limit 1 --execute
```

Available policy values:

```text
direct_iterative
question_decomposition
both
```

## 8. Run the Exploratory Analysis

Run the offline analysis script:

```bash
python src/stage6_8_exploratory_analysis.py
```

No LLM API calls are required for this step.

The script generates:

```text
analysis/
└── stage6_8_exploratory_results.txt
```

The script is pinned to a specific run. Its output reflects that run and may not match results from a separately reproduced experiment. Running it may overwrite the existing summary file.

For a newly reproduced experiment, use the run ID and raw outputs from that experiment when performing run-specific analysis.
