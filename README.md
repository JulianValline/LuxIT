# LuxIT:  A Luxembourgish Instruction Tuning Dataset from Monolingual Seed Data


An end-to-end pipeline for generating, evaluating, filtering, fine-tuning, and benchmarking Luxembourgish instruction-tuning datasets across multiple LLM families.

## Overview

This repository contains the code used to create and evaluate LuxIT, covering five stages:

1. **Synthetic Data Generation** — Generate Luxembourgish instruction-response pairs from RTL news articles and Luxembourgish Wikipedia using DeepSeek R1.
2. **LLM-as-Judge Scoring** — Score generated pairs on four quality dimensions using OpenAI models, with parallel processing support.
3. **Fine-Tuning** — Fine-tune models from multiple families (Qwen, Gemma, Llama, Mistral, Phi, GLM, EuroLLM, Olmo) using Unsloth and LoRA.
4. **Language Exam Evaluation** — Benchmark base and fine-tuned models on standardized Luxembourgish language exams (A1–C2).
5. **Multitask NLP Benchmarking** — Evaluate base and fine-tuned models on downstream Luxembourgish NLP tasks (Intent Classification, RTE, Sentiment Analysis, SST, WNLI) in zero-shot and few-shot settings.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── .env.example
│
├── generation/
│   └── lux_synthetic_data_generation.py      # Synthetic data generation from RTL/Wikipedia
│
├── scoring/
│   ├── lux_dataset_evaluator.py              # Parallel LLM-as-Judge evaluator
│   ├── run_parallel_eval.sh                  # Tmux-based parallel evaluation launcher
│   └── filter_lux_dataset.py                 # Quality filtering by score thresholds
│
├── training/
│   ├── multi_model_finetuning.py             # Universal fine-tuning pipeline
│   ├── configs.py                            # Dataclass configs for models and training
│   ├── training_runner.sh                    # Shell wrapper for training runs
│   └── example_config.yaml                   # Example YAML configuration
│
├── exam_evaluation/
│   ├── evaluate_language_exams.py             # Luxembourgish exam evaluation pipeline
│   └── config.py                             # Model and exam configuration
│
└── multitask_benchmarking/
    ├── multitask_evaluation.py               # Multitask NLP evaluation pipeline
    └── example_config.json                   # Example JSON configuration
```

## Prerequisites

- Python 3.10+
- CUDA-compatible GPU (required for fine-tuning and exam evaluation)
- API keys:
  - **Nebius API key** (for synthetic data generation via DeepSeek R1)
  - **OpenAI API key** (for LLM-as-Judge scoring)
- `tmux` (optional, for parallel scoring)
- `micromamba` or `conda` (recommended for environment management)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/JulianValline/LuxIT.git
cd LuxIT
```

### 2. Create a virtual environment

```bash
# Using conda/micromamba (recommended)
micromamba create -n luxit python=3.10
micromamba activate luxit

# Or using venv
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Unsloth installation can be environment-specific. If you encounter issues, consult the [Unsloth installation guide](https://github.com/unslothai/unsloth#installation) for your CUDA version. The `training_runner.sh` script also includes an `--install-deps` flag that can handle installation interactively.

### 4. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```
NEBIUS_API_KEY=your-nebius-api-key-here
OPENAI_API_KEY=your-openai-api-key-here
```

---

## Pipeline Stages

### Stage 1: Synthetic Data Generation

Generates Luxembourgish instruction-response pairs from seed articles using DeepSeek R1 via the Nebius API.

**Supported sources:**
- RTL news articles (Luxembourgish)
- Luxembourgish Wikipedia articles

**Input format:** JSON files containing articles with `title`, `text`, and optionally `date`/`header` fields.

**Usage:**

```bash
cd generation/

# Process all sources (RTL + Wikipedia)
python lux_synthetic_data_generation.py

# Process only RTL articles with a limit
python lux_synthetic_data_generation.py --rtl-only --max-rtl 100

# Process only Wikipedia articles
python lux_synthetic_data_generation.py --wiki-only --max-wiki 50

# Set limits for both sources
python lux_synthetic_data_generation.py --max-rtl 6 --max-wiki 4

# Combine existing datasets from separate runs
python lux_synthetic_data_generation.py --combine-only
```

**Key features:**
- Checkpoint-based recovery — safely resume interrupted generation runs
- Per-source sample limits for controlled generation
- Outputs in both JSON and CSV formats with ShareGPT conversation structure
- Configurable number of instruction-response pairs per article (default: 3)

**Configuration:** Edit the `Config` dataclass at the top of the script, or use command-line arguments. Source article files should be placed in a `Data/` directory relative to the script (configurable via `Config.base_path`).

**Output:** JSON and CSV files containing instruction-response pairs in ShareGPT format, ready for scoring or direct fine-tuning.

---

### Stage 2: LLM-as-Judge Scoring

Evaluates each instruction-response pair on four quality dimensions (1–3 scale) using an OpenAI model as judge.

**Scoring criteria:**

| Dimension | Score 1 | Score 2 | Score 3 |
|---|---|---|---|
| **Linguistic Quality** | Significant errors, wrong language | Minor errors, slightly unnatural | Fluent, idiomatic Luxembourgish |
| **Factual Accuracy** | Contains factual errors | Mostly accurate, minor issues | Completely accurate |
| **Instruction Adherence** | Fails to follow instruction | Partially follows instruction | Perfectly follows all constraints |
| **Helpfulness & Relevance** | Nonsensical or unhelpful | Basic but adequate | Genuinely useful and comprehensive |

#### Single-worker evaluation

```bash
cd scoring/

python lux_dataset_evaluator.py \
  --dataset ../data/instruction_answers_rtl.json \
  --output ../results/evaluated.jsonl \
  --model gpt-5-mini
```

#### Parallel evaluation (recommended for large datasets)

```bash
# Launch 4 parallel workers via tmux
./run_parallel_eval.sh ../data/dataset.jsonl ../results/evaluated 4 --model gpt-5-mini

# Monitor progress
tmux attach-session -t luxeval

# After all workers complete, merge results
python lux_dataset_evaluator.py --merge --output ../results/evaluated --num-workers 4
```

**Supported input formats:** CSV, JSON, JSONL

**Key features:**
- Parallel processing via tmux with configurable worker count
- Checkpoint-based recovery per worker — safe to interrupt and resume
- Exponential backoff with jitter for rate limit handling
- Atomic file writes to prevent data corruption

#### Quality filtering

After scoring, filter the dataset to remove low-quality entries:

```bash
python filter_lux_dataset.py
```

Edit the `__main__` block at the bottom of the script to configure paths and thresholds:

```python
results = filter_evaluated_dataset(
    input_path="results/evaluated_merged.jsonl",
    output_base_path="data/LuxIT_large",
    output_format="jsonl",
    min_acceptable_score=2,        # Reject any entry with a score of 1
    save_rejected=True,            # Save rejected entries for review
    generate_report_file=True,     # Generate detailed statistics report
    prepare_training_set=True,     # Create a training-ready version (no score columns)
)
```

**Outputs:**
- `*_filtered.jsonl` — Accepted entries with scores
- `*_training.jsonl` — Training-ready file (instruction + response only)
- `*_rejected.jsonl` — Rejected entries with rejection reasons
- `*_report.txt` — Detailed filtering statistics

---

### Stage 3: Fine-Tuning

Fine-tune language models using Unsloth with LoRA adapters. Supports multiple model families with automatic detection and family-specific configurations.

**Supported model families:**

| Family  | Loader | Example Models                                         |
|---------|---|--------------------------------------------------------|
| Qwen    | FastLanguageModel | Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen2.5-7B                 |
| Gemma   | FastModel | Gemma-3-1B-IT                                          |
| Llama   | FastLanguageModel | Llama-3.1-8B, Llama-3.2-1B                             |
| Mistral | FastLanguageModel | Mistral-7B-Instruct-v0.3, Ministral-3-3B-2512-Instruct |
| Phi     | FastLanguageModel | Phi-4                                                  |
| GLM     | FastLanguageModel | GLM-4-9B-0414                                          |
| EuroLLM | FastLanguageModel | EuroLLM-1.7B-Instruct                                  |
| Olmo    | FastLanguageModel | Olmo-3-7B-Instruct                                     |
| Apertus | HuggingFace | Apertus-8B-Instruct-2509                               |

> **Note on Apertus:** Apertus-8B is fine-tuned following the official [swiss-ai/apertus-finetuning-recipes](https://github.com/swiss-ai/apertus-finetuning-recipes) repository, which is built on TRL, Accelerate, and Transformers. To use these recipes directly:
>
> ```bash
> # Install and launch LoRA training
> uv pip install -r requirements.txt
> python sft_train.py --config configs/sft_lora.yaml
>
> # Full-parameter training (4 GPUs)
> accelerate launch --config_file configs/zero3.yaml sft_train.py --config configs/sft_full.yaml
> ```
>
> After training, load the resulting adapter via the `huggingface` model type in `multitask_evaluation.py` by setting `lora_path` to your output directory (`Apertus-FT/output/apertus_lora/`).

#### Quick start

```bash
cd training/

# Run with a YAML config
./training_runner.sh --config example_config.yaml

# Override model or output directory
./training_runner.sh --config example_config.yaml \
  --model unsloth/Qwen2.5-0.5B-Instruct \
  --output-dir ./output_qwen2.5

# Resume from checkpoint
./training_runner.sh --config example_config.yaml \
  --resume ./output/checkpoints/checkpoint-500

# Install dependencies interactively
./training_runner.sh --install-deps
```

#### Running directly with Python

```bash
python multi_model_finetuning.py --config example_config.yaml

# With overrides
python multi_model_finetuning.py \
  --config example_config.yaml \
  --model unsloth/Meta-Llama-3.1-8B-Instruct \
  --output-dir ./output_llama \
  --wandb

# Evaluation only (on existing model)
python multi_model_finetuning.py --config example_config.yaml --eval-only
```

#### Configuration

Create a YAML config file (see `example_config.yaml` for a complete reference):

```yaml
model:
  model_name: "unsloth/Qwen2.5-7B-Instruct"
  max_seq_length: 2048
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05

training:
  dataset_name: "./data/LuxIT_large_training.jsonl"
  output_dir: "./output_qwen2.5-7b"
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8
  num_train_epochs: 2
  learning_rate: 0.0001
  eval_strategy: "steps"
  eval_steps: 500
  save_steps: 500
  use_wandb: true
  wandb_project: "Luxembourgish-finetuning"
```

**Key features:**
- Automatic model family detection with family-specific LoRA configurations
- Response masking (trains only on assistant responses, not user prompts)
- JSONL dataset support with memory-mapped loading for large datasets
- ChatML and ShareGPT format handling with automatic normalization
- Checkpoint-based training resumption
- Weights & Biases integration
- ROUGE evaluation on held-out test set

**Dataset format:** The training script expects JSONL files where each line contains a `messages` (or `conversations`) field with a list of role/content pairs:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

ShareGPT format (`from`/`value` keys, `human`/`gpt` roles) is also supported and automatically normalized.

---

### Stage 4: Language Exam Evaluation

Benchmark base and fine-tuned models on standardized Luxembourgish language exams (A1–C2 levels) from the INLL (Institut National des Langues Luxembourg).

#### Setup

1. Place exam TSV files in `exam_evaluation/exam_files/`
2. Place LoRA adapters in `exam_evaluation/lora_adapters/`
3. Edit `exam_evaluation/config.py` to configure models and exams

#### Running evaluations

```bash
cd exam_evaluation/

# Run all configured models and exams
python evaluate_language_exams.py

# Force re-evaluation (ignore cached results)
python evaluate_language_exams.py --force
```

#### Configuring models

Edit `config.py` to add or remove models:

```python
MODELS = [
    # Base model
    ModelConfig(
        name="qwen2.5-7b-base",
        base_model="unsloth/Qwen2.5-7B-Instruct",
    ),
    # Fine-tuned model with LoRA
    ModelConfig(
        name="qwen2.5-7b-luxit-large",
        base_model="unsloth/Qwen2.5-7B-Instruct",
        lora_path=str(LORA_DIR / "qwen2.5-7b-luxit-large"),
    ),
]
```

**Key features:**
- Supports both base models and LoRA fine-tuned models
- Automatic model family detection (FastModel for Gemma, FastLanguageModel for others)
- Skip-existing logic — only evaluates models that haven't been scored yet
- Post-processing with Levenshtein distance for robust answer matching
- Per-category and per-level score breakdowns
- Results saved as JSON per model and a combined TSV summary

**Output:** Results are saved to `evaluation_results/` with per-model directories containing raw outputs, formatted answers, and score JSON files, plus a `final_summary.tsv` comparing all models.

---

### Stage 5: Multitask NLP Benchmarking

Evaluate base and fine-tuned models across a suite of Luxembourgish NLP classification tasks in zero-shot and few-shot settings, reporting F1 and accuracy metrics for each task.

**Supported tasks:**

| Task | Description | Labels                                           |
|---|---|--------------------------------------------------|
| **Intent Classification** | Classify user intent from Luxembourgish utterances | various intent labels                            |
| **Recognizing Textual Entailment (RTE)** | Determine if a hypothesis is entailed by a premise | `0` (not entailed), `1` (entailed)               |
| **Sentiment Analysis** | Three-way sentiment classification | `0` (positive), `1` (negative), `2` (neutral)    |
| **Stanford Sentiment Treebank (SST)** | Binary sentiment classification | `0` (negative), `1` (positive)                   |
| **Winograd NLI (WNLI)** | Winograd-schema coreference inference | `0` (wrong antecedent), `1` (correct antecedent) |

#### Setup

1. Place benchmark dataset directories under `multitask_benchmarking/benchmarks/`, one subdirectory per task (e.g. `benchmarks/IC/`, `benchmarks/RTE/`, etc.)
2. Each task directory must contain split files named `train.{json,jsonl,tsv,csv}` and `test.{json,jsonl,tsv,csv}`
3. Create a JSON config file specifying which models and tasks to evaluate (see `example_config.json`)

#### Running evaluations

```bash
cd multitask_benchmarking/

# Run with a config file (default: 0, 3, and 5 shots)
python multitask_evaluation.py --config example_config.json

# Specify custom shot counts
python multitask_evaluation.py --config example_config.json --shots 0,3,5,10

# Override output and checkpoint paths
python multitask_evaluation.py \
  --config example_config.json \
  --output results/my_run.json \
  --checkpoint checkpoints/my_run.json \
  --log-file logs/my_run.log
```

#### Configuration

All models and tasks are defined in a single JSON config file. See `example_config.json` for a full reference:

```json
{
  "models": [
    {
      "name": "qwen2.5-0.5b-baseline",
      "model_path": "unsloth/Qwen2.5-0.5B-Instruct",
      "model_type": "unsloth_fast_language_model",
      "lora_path": null,
      "max_seq_length": 2048,
      "load_in_4bit": false,
      "use_chat_template": true,
      "unsloth_chat_template": "chatml"
    },
    {
      "name": "qwen2.5-0.5b-LuxIT-large",
      "model_path": "unsloth/Qwen2.5-0.5B-Instruct",
      "model_type": "unsloth_fast_language_model",
      "lora_path": "../training/output_qwen/final_model",
      "max_seq_length": 2048,
      "load_in_4bit": false,
      "use_chat_template": true,
      "unsloth_chat_template": null
    }
  ],
  "tasks": [
    {
      "task_type": "intent_classification",
      "dataset_path": "./benchmarks/IC",
      "label_column": "label",
      "text_column": "text",
      "labels": ["affirm", "check_balance", "transfer_money", "..."]
    },
    {
      "task_type": "rte",
      "dataset_path": "./benchmarks/RTE",
      "label_column": "label",
      "text_column": "sentence1",
      "text_column_2": "sentence2",
      "labels": [0, 1]
    }
  ]
}
```

**`model_type` options:** `unsloth_fast_language_model` (Llama, Qwen, Mistral, etc.), `unsloth_fast_model` (Gemma), `huggingface` (standard HuggingFace models).

**Key features:**
- Zero-shot and few-shot evaluation with configurable shot counts
- Stratified few-shot sampling — ensures all label classes are represented in the prompt
- Luxembourgish prompt templates built in for all supported tasks
- Chat template support via tokenizer or Unsloth's `get_chat_template`
- Label extraction with word-boundary matching and longest-match priority (handles multi-digit numeric labels correctly)
- Checkpoint-based recovery — saves progress every 50 samples and on interrupt
- F1 micro, macro, and weighted + accuracy reported per model/task/shot combination
- Unparseable model outputs are counted as incorrect rather than excluded, giving an honest metric

**Output:** Results are saved to the specified JSON file with per-combination metrics and an aggregate summary:

```json
{
  "evaluation_timestamp": "...",
  "results": [
    {
      "model_name": "qwen2.5-0.5b-LuxIT-large",
      "task_name": "intent_classification",
      "shot_type": "few_shot_3",
      "f1_micro": 0.812,
      "f1_macro": 0.794,
      "f1_weighted": 0.809,
      "accuracy": 0.812,
      "num_samples": 350,
      "num_failed": 4
    }
  ],
  "summary": {
    "by_model": { "..." : "..." },
    "by_task":  { "..." : "..." },
    "overall_f1_macro": 0.731
  }
}
```

---

## Environment Variables

| Variable | Used By | Description |
|---|---|---|
| `NEBIUS_API_KEY` | Synthetic data generation | API key for Nebius (DeepSeek R1 access) |
| `OPENAI_API_KEY` | LLM-as-Judge scoring | API key for OpenAI models |

---

## Common Workflows

### Full pipeline (end-to-end)

```bash
# 1. Generate synthetic data from source articles
cd generation/
python lux_synthetic_data_generation.py --max-rtl 1000 --max-wiki 500

# 2. Score the generated dataset
cd ../scoring/
./run_parallel_eval.sh ../generation/instruction_answers_rtl.json ../results/evaluated 4
# Wait for completion, then merge:
python lux_dataset_evaluator.py --merge --output ../results/evaluated --num-workers 4

# 3. Filter by quality
python filter_lux_dataset.py

# 4. Fine-tune a model
cd ../training/
./training_runner.sh --config example_config.yaml

# 5. Evaluate on language exams
cd ../exam_evaluation/
python evaluate_language_exams.py

# 6. Benchmark on downstream NLP tasks
cd ../multitask_benchmarking/
python multitask_evaluation.py --config example_config.json
```

### Resume interrupted work

All pipeline stages support checkpointing. Simply re-run the same command and processing resumes from the last checkpoint:

```bash
# Re-running any of these picks up where it left off:
python lux_synthetic_data_generation.py --rtl-only
python lux_dataset_evaluator.py --dataset data.jsonl --output results.jsonl --worker-id 0 --num-workers 4
./training_runner.sh --config config.yaml --resume ./output/checkpoints/checkpoint-500
python multitask_evaluation.py --config example_config.json --checkpoint eval_checkpoint.json
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Citation

If you use this pipeline or the resulting datasets/models in your research, please cite:

```bibtex
@misc{valline2025luxitluxembourgishinstructiontuning,
      title={LuxIT: A Luxembourgish Instruction Tuning Dataset from Monolingual Seed Data}, 
      author={Julian Valline and Cedric Lothritz and Jordi Cabot},
      year={2025},
      eprint={2510.24434},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.24434}, 
}
```