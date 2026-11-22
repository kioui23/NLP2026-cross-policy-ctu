\# Cross-Policy Counterfactual Document-Impact Analysis



\## 1. Overview



This repository contains the experimental code and data for an exploratory study comparing the document-impact rankings of two search-agent policies:



\- \*\*Direct Iterative Search\*\*

\- \*\*Question Decomposition Search\*\*



The study investigates whether documents that influence an agent's answer under one search policy also exhibit similar impact under another policy.



The experiment uses HotpotQA and evaluates document-removal interventions through counterfactual replay.



The current pilot focuses on answer-text perturbation as an exploratory proxy for document impact. It does not implement or report official Counterfactual Trajectory Utility (CTU).



\---



\## 2. Experimental Setup



\| Component | Configuration |

\|---|---|

\| Dataset | HotpotQA |

\| Dataset split | Validation |

\| Number of questions | 10 |

\| Sampling seed | 42 |

\| Model | Gemini 3.5 Flash-Lite |

\| Temperature | 0 |

\| Retrieval method | BM25 |

\| Top-k retrieval | 2 |

\| Maximum retrieval rounds | 4 |

\| Maximum unique documents | 8 |

\| Policies | Direct iterative search, Question decomposition |



Both policies use the same model and retrieval setup.



\### Policies



\*\*Direct Iterative Search\*\*



The agent iteratively searches for evidence by generating queries, retrieving documents, and refining its search based on the retrieved evidence.



\*\*Question Decomposition Search\*\*



The agent decomposes the original question into subquestions and performs retrieval based on the decomposed questions.



\---



\## 3. Repository Structure



\`\`\`text

cross_policy_ctu_public_package/

│

├── README.md

├── requirements.txt

├── .env.example

│

├── src/

│   ├── experiment_runner.py

│   ├── data_loader.py

│   ├── document_processor.py

│   ├── policy_a.py

│   ├── policy_b.py

│   ├── retriever.py

│   ├── ctu_prepare.py

│   ├── counterfactual_runner.py

│   ├── counterfactual_replay.py

│   └── stage6_8_exploratory_analysis.py

│

├── analysis/

│   ├── stage6_intervention_analysis.csv

│   ├── stage7_document_utility_proxy_rankings.csv

│   └── stage6_8_exploratory_results.txt

│

└── hw2_results/

&#x20;   └── full_10_question_baseline/

&#x20;       └── [run_id]/

&#x20;           ├── experiment_results.jsonl

&#x20;           ├── experiment_summary.csv

&#x20;           ├── experiment_manifest.json

&#x20;           └── ctu_preparation/

&#x20;               ├── clean_trajectories.jsonl

&#x20;               ├── clean_trajectory_validation.json

&#x20;               ├── interventions.jsonl

&#x20;               └── counterfactual_replay/

&#x20;                   ├── replay_results.jsonl

&#x20;                   ├── failures.jsonl

&#x20;                   └── replay_manifest.json

\`\`\`



The repository contains the pilot data required to reproduce the reported exploratory analysis.



\---



\## 4. Environment Setup



The project was developed using Python 3.11.8.



\### 4.1 Clone the repository



\`\`\`bash

git clone https\://github.com/kioui23/NLP2026-cross-policy-ctu.git



**cd** NLP2026-cross-policy-ctu

\`\`\`



\### 4.2 Create a virtual environment



\`\`\`bash

python3.11 -m venv .venv

\`\`\`



Activate the environment:



\*\*macOS / Linux\*\*



\`\`\`bash

**source** .venv/bin/activate

\`\`\`



\### 4.3 Install dependencies



\`\`\`bash

python -m pip install --upgrade pip



pip install -r requirements.txt

\`\`\`



\---



\# 5. Reproduce the Bundled Pilot Analysis

This section describes the offline exploratory analysis script. No LLM API calls are required.

From the repository root, run:

\`\`\`bash
python src/stage6_8_exploratory_analysis.py
\`\`\`

The script generates:

\`\`\`text
analysis/
├── stage6_intervention_analysis.csv
├── stage7_document_utility_proxy_rankings.csv
└── stage6_8_exploratory_results.txt
\`\`\`

The analysis script is currently pinned to the original pilot run. Its computed output may not match the final manuscript results reported in Section 8. The TXT summary included in the repository has been aligned with the final manuscript, but rerunning the script may overwrite it with the script's run-specific results. See Section 10 for details.

\---



\# 6. Experimental Reproduction



This section describes how to reproduce the experimental pipeline using API calls rather than only analyzing the bundled pilot data.



The experimental reproduction consists of:



1\. Dataset preparation

2\. Baseline experiment

3\. Trajectory preparation and intervention generation

4\. Counterfactual replay

5\. Analysis



Some steps require Gemini API calls and may incur API costs.



\---



\## 6.1 Configure the API



Copy the environment template:



\`\`\`bash

cp .env.example .env

\`\`\`



Open \`.env\` and insert your own Google Gemini API key.



\`\`\`text

GOOGLE_API_KEY=your_google_api_key_here

\`\`\`



Do not commit \`.env\` or share your API key.



\---



\## 6.2 Prepare the Dataset



Download HotpotQA and reproduce the question selection using seed 42.



Run:



\`\`\`bash

python -m src.data_loader

\`\`\`



This generates:



\`\`\`text

hw2_results/hotpotqa_selected_questions.json

\`\`\`



Next, extract and save the documents associated with the selected questions:



\`\`\`bash

python -m src.document_processor

\`\`\`



This generates:



\`\`\`text

hw2_results/hotpotqa_documents.json

\`\`\`



These files are required by the intervention preparation and counterfactual replay pipeline.



\---



\## 6.3 Run the Baseline Experiment



Execute:



\`\`\`bash

python -m src.experiment_runner

\`\`\`



The experiment runs both search policies on the selected 10 HotpotQA questions.



Expected total:



\`\`\`text

10 questions × 2 policies = 20 runs

\`\`\`



The experiment includes:



\- Direct Iterative Search

\- Question Decomposition Search



The experiment runner generates a new run ID for each execution. The run ID is printed in the console and recorded in \`experiment_manifest.json\`.



Baseline outputs are saved under:



\`\`\`text

hw2_results/

└── full_10_question_baseline/

&#x20;   └── [RUN_ID]/

&#x20;       ├── experiment_results.jsonl

&#x20;       ├── experiment_summary.csv

&#x20;       └── experiment_manifest.json

\`\`\`



The experiment also appends records to the shared logging files:



\`\`\`text

hw2_results/llm_calls.jsonl

hw2_results/trajectory_log.jsonl

\`\`\`



\*\*Note:\*\* Although 20 runs are scheduled, some runs may fail due to API errors or policy execution errors. Check the successful and failed runs separately before proceeding.



Record the generated run ID. It is required for subsequent intervention preparation and counterfactual replay.



\---



\## 6.4 Prepare Counterfactual Interventions



After the baseline experiment finishes, prepare clean trajectories and document-removal interventions.



Replace \`[RUN_ID]\` with the run ID generated by the baseline experiment.



\`\`\`bash

python src/ctu_prepare.py \\

&#x20;   \--run-id [RUN_ID]

\`\`\`



This stage does not make LLM/API calls.



The script:



1\. Loads the baseline trajectory logs.

2\. Selects events belonging to the specified run.

3\. Validates baseline trajectory pairs.

4\. Excludes incomplete trajectory pairs.

5\. Extracts document-level interventions from validated trajectories.

6\. Saves the clean trajectories and intervention records.



Generated files:



\`\`\`text

hw2_results/full_10_question_baseline/[RUN_ID]/

└── ctu_preparation/

&#x20;   ├── clean_trajectories.jsonl

&#x20;   ├── clean_trajectory_validation.json

&#x20;   └── interventions.jsonl

\`\`\`



The generated interventions contain the target document and the information required to replay the trajectory after document removal.



\*\*Note:\*\* The number of interventions may differ between policies because interventions are generated from successfully completed baseline trajectories. Incomplete trajectory pairs are excluded. Therefore, intervention counts are not necessarily balanced across policies.



\---



\## 6.5 Inspect the Counterfactual Replay



Before making API calls, perform a dry run.



\`\`\`bash

python -m src.counterfactual_runner \\

&#x20;   \--run-id [RUN_ID]

\`\`\`



The script runs in dry-run mode by default.



It reports:



\- Number of selected interventions

\- Number of completed interventions

\- Number of pending interventions

\- Intervention counts by policy

\- Output directory



No LLM/API calls are made during the dry run.



Review the intervention counts and confirm that the selected run ID and output directory correspond to the intended baseline experiment before executing the replay.



\---



\## 6.6 Execute Counterfactual Replay



Counterfactual replay requires LLM/API calls and may incur API costs.



To execute the selected interventions:



\`\`\`bash

python -m src.counterfactual_runner \\

&#x20;   \--run-id [RUN_ID] \\

&#x20;   \--execute

\`\`\`



The replay runner executes counterfactual trajectories using the configured LLM.



Results are checkpointed to JSONL files as each intervention finishes.



\`\`\`text

hw2_results/full_10_question_baseline/[RUN_ID]/

└── ctu_preparation/

&#x20;   └── counterfactual_replay/

&#x20;       ├── replay_results.jsonl

&#x20;       ├── failures.jsonl

&#x20;       └── replay_manifest.json

\`\`\`



Completed intervention IDs are skipped when the script is resumed.



Failed interventions are recorded separately and may be retried in a subsequent execution.



\### Optional: Run a Limited Replay



For a small execution test, use \`--limit\`.



Example:



\`\`\`bash

python -m src.counterfactual_runner \\

&#x20;   \--run-id [RUN_ID] \\

&#x20;   \--limit 1 \\

&#x20;   \--execute

\`\`\`



This executes only one selected intervention.



To restrict execution to one policy:



\`\`\`bash

python -m src.counterfactual_runner \\

&#x20;   \--run-id [RUN_ID] \\

&#x20;   \--policy direct_iterative \\

&#x20;   \--limit 1 \\

&#x20;   \--execute

\`\`\`



Available policies:



\`\`\`text

direct_iterative

question_decomposition

both

\`\`\`





\# 7. Exploratory Analysis



The exploratory analysis compares baseline answers with counterfactual replay answers after document removal.



The current analysis uses answer-string similarity as an exploratory proxy.



For each matched baseline and counterfactual result, the analysis calculates:



\### Answer Similarity



Normalized string similarity between the baseline answer and counterfactual answer.



\### Answer Changed Rate



The proportion of counterfactual answers that differ from the baseline answer.



\### Answer Perturbation Proxy



\`\`\`text

Answer Perturbation Proxy = 1 - Answer Similarity

\`\`\`



Higher values indicate greater answer-text change following document removal.



\### Effort Delta Proxy



The analysis compares the number of replay rounds with an estimated number of remaining baseline rounds.



This metric is exploratory and should not be interpreted as a validated measure of search efficiency.



\### Cross-Policy Ranking Correlation



Documents are ranked independently under the two policies based on their mean answer-perturbation proxy.



The analysis then calculates Spearman rank correlation for documents shared between both policy rankings.



\---



\# 8. Results and Interpretation

The final manuscript reports 96 successful matched counterfactual interventions.

| Policy | N | Mean Perturbation | Median Perturbation | Answer Changed | Identical Answer |
|---|---:|---:|---:|---:|---:|
| Direct Iterative | 60 | 0.3310 | 0.2090 | 51 (85.0%) | 9 (15.0%) |
| Question Decomposition | 36 | 0.4301 | 0.4732 | 35 (97.2%) | 1 (2.8%) |

The document-ranking comparison includes 27 shared target documents and yields a Spearman rank correlation of 0.7861.

These results indicate a positive association between the two policies' document-level answer-perturbation rankings within the shared-document subset. Question Decomposition exhibited higher mean answer perturbation than Direct Iterative Search in this pilot.

Answer perturbation measures textual change rather than factual correctness, retrieval quality, or validated causal utility. The findings are therefore exploratory and do not establish transfer of causal document utility across search policies.

\---



\# 9. Limitations



The current pilot has several limitations.



\- The experiment uses only 10 HotpotQA validation questions.

\- The number of successful counterfactual replays differs between policies.

\- Answer perturbation is based on string similarity and does not directly measure answer correctness.

\- A changed answer may be equally correct, more correct, or less correct than the original answer.

\- The analysis uses a proxy rather than official CTU.

\- The document rankings are based on a limited shared-document subset.

\- The effort-delta proxy is not a validated search-efficiency metric.

\- No generalization claims can be made from the current pilot.



The results should therefore be interpreted as exploratory evidence of cross-policy association in answer-text sensitivity rankings.



\---



\# 10. Reproducibility Notes

The repository includes bundled analysis outputs from an earlier pilot run and results from a later experimental run reported in the final manuscript. These should not be treated as a single experiment.

### Bundled Offline Analysis

The bundled offline analysis can be run without API calls:

\`\`\`bash
python src/stage6_8_exploratory_analysis.py
\`\`\`

The analysis script is currently pinned to the original pilot run. Its outputs are not identical to the final manuscript results in Section 8.

### Final Manuscript Results

The final manuscript reports the later run with 96 successful matched counterfactual interventions:

- Direct Iterative: 60 interventions
- Question Decomposition: 36 interventions
- Shared target documents: 27
- Spearman rank correlation: 0.7861

Reported answer-perturbation statistics are listed in Section 8.

### Experimental Reproduction

Experimental reproduction requires:

- Downloading HotpotQA
- Selecting the questions using the specified seed
- Running both baseline search policies
- Generating document-removal interventions
- Executing counterfactual replay
- Analyzing the resulting outputs

API-based reproduction may produce different results due to model/API behavior, even when the experimental configuration is unchanged.

\---



\# 11. Citation and Academic Use



This repository is an exploratory reproduction package developed for an academic course project.



The included pilot data and analysis scripts are provided to support transparency and reproducibility.



The current implementation should not be interpreted as a validated reproduction of official CTU or as a definitive evaluation of document causal utility.