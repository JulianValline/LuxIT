"""
Configuration file for Luxembourgish Language Exam Evaluation

Edit this file to configure which models and exams to run.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class ModelConfig:
    """Configuration for a model to evaluate."""
    name: str                          # Display name for the model
    base_model: str                    # HuggingFace model ID or path
    lora_path: Optional[str] = None    # Path to LoRA adapter (None for base model only)
    max_seq_length: int = 2048
    load_in_4bit: bool = False
    dtype: Optional[torch.dtype] = None
    # Chat template options:
    # - None: Auto-detect (use LoRA tokenizer if available, else base model)
    # - "path/to/tokenizer": Load chat template from tokenizer at this path
    # - A Jinja2 template string: Use exact template
    chat_template: Optional[str] = None
    enable_thinking: Optional[bool] = False              # Default to False for standard evaluation


@dataclass
class ExamConfig:
    """Configuration for a language exam."""
    level: str                         # e.g., "a1", "a2", "b1", "b2", "c1", "c2"
    test_file: str                     # Path to the test TSV file
    ground_truth_file: str             # Path to the ground truth TSV file


# =============================================================================
# DIRECTORY CONFIGURATION
# =============================================================================

TEST_DIR = Path("exam_files")
LORA_DIR = Path("lora_adapters")
OUTPUT_DIR = "evaluation_results"


# =============================================================================
# EXAM CONFIGURATION
# =============================================================================

EXAMS = [
    ExamConfig(
        level="a1",
        test_file=str(TEST_DIR / "inll_a1_luxembourgish_test.tsv"),
        ground_truth_file=str(TEST_DIR / "INLL_A1.tsv")
    ),
    ExamConfig(
        level="a2",
        test_file=str(TEST_DIR / "inll_a2_luxembourgish_test.tsv"),
        ground_truth_file=str(TEST_DIR / "INLL_A2.tsv")
    ),
    ExamConfig(
        level="b1",
        test_file=str(TEST_DIR / "inll_b1_luxembourgish_test.tsv"),
        ground_truth_file=str(TEST_DIR / "INLL_B1.tsv")
    ),
    ExamConfig(
        level="b2",
        test_file=str(TEST_DIR / "inll_b2_luxembourgish_test.tsv"),
        ground_truth_file=str(TEST_DIR / "INLL_B2.tsv")
    ),
    ExamConfig(
        level="c1",
        test_file=str(TEST_DIR / "inll_c1_luxembourgish_test.tsv"),
        ground_truth_file=str(TEST_DIR / "INLL_C1.tsv")
    ),
    ExamConfig(
        level="c2",
        test_file=str(TEST_DIR / "inll_c2_luxembourgish_test.tsv"),
        ground_truth_file=str(TEST_DIR / "INLL_C2.tsv")
    ),
]


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Chat template behavior:
# 1. For LoRA models: If LoRA directory contains a tokenizer (tokenizer.json,
#    tokenizer_config.json, or tokenizer.model), the chat template from that 
#    tokenizer will be used automatically.
# 2. If no tokenizer in LoRA dir: Falls back to base model's chat template (warning logged)
# 3. Can be overwritten with chat_template parameter:
#    - Path to a tokenizer directory: chat_template="./my_tokenizer/"
#    - Jinja2 template string: chat_template="{% for message in messages %}..."

MODELS = [
    # -------------------------------------------------------------------------
    # BASE MODELS (for comparison)
    # Uses the default chat template from each model
    # -------------------------------------------------------------------------

    # Qwen models (use FastLanguageModel)
    ModelConfig(
        name="qwen3-0.6b-base",
        base_model="unsloth/Qwen3-0.6B",
        lora_path=None,
        enable_thinking=False,
    ),

    ModelConfig(
        name="qwen3-8b-base",
        base_model="unsloth/Qwen3-8B",
        lora_path=None,
        enable_thinking=False,
    ),

    ModelConfig(
        name="qwen2.5-7b-base",
        base_model="unsloth/Qwen2.5-7B-Instruct",
        lora_path=None,
    ),

    ModelConfig(
        name="qwen2.5-0.5b-base",
        base_model="unsloth/Qwen2.5-0.5B-Instruct",
        lora_path=None,
    ),

    ModelConfig(
        name="qwen2.5-1.5b-base",
        base_model="unsloth/Qwen2.5-1.5B-Instruct",
        lora_path=None,
    ),
    
    # GLM models (use FastLanguageModel)
    ModelConfig(
        name="glm4-4-9b-0414-base",
        base_model="THUDM/GLM-4-9B-0414",
        lora_path=None,
    ),

    # Mistral models (use FastLanguageModel)
    ModelConfig(
        name="mistral-7b-base",
        base_model="unsloth/mistral-7b-instruct-v0.3",
        lora_path=None,
    ),

    # Llama3 models (use FastLanguageModel)
    ModelConfig(
        name="llama-3.1-8b-base",
        base_model="unsloth/Meta-Llama-3.1-8B-Instruct",
        lora_path=None,
        # chat_template=None means use base model's default template
    ),

    ModelConfig(
        name="llama-3.2-1b-base",
        base_model="unsloth/Llama-3.2-1B-Instruct",
        lora_path=None,
        # chat_template=None means use base model's default template
    ),

    # Phi-4 models (use FastLanguageModel)
    ModelConfig(
        name="phi-4-base",
        base_model="unsloth/phi-4",
        lora_path=None,
        # chat_template=None means use base model's default template
    ),

    # -------------------------------------------------------------------------
    # LORA FINE-TUNED MODELS
    # Will use tokenizer from LoRA directory if saved during training
    # -------------------------------------------------------------------------

    # Qwen with LoRA (uses FastLanguageModel)
    ModelConfig(
        name="qwen3-0.6b-luxit-large",
        base_model="unsloth/Qwen3-0.6B",
        lora_path=str(LORA_DIR / "qwen3-0.6b-luxit-large"),
        chat_template="chatml", # was used during training since the standard tokenizer was generating empty  tokens
        enable_thinking=False,
    ),

    ModelConfig(
        name="qwen3-8b-luxit-large",
        base_model="unsloth/Qwen3-8B",
        lora_path=str(LORA_DIR / "qwen3-8b-luxit-large"),
        chat_template="chatml", # was used during training since the standard tokenizer was generating empty  tokens
        enable_thinking=False,
    ),

    ModelConfig(
        name="qwen2.5-7b-luxit-large",
        base_model="unsloth/Qwen2.5-7B-Instruct",
        lora_path=str(LORA_DIR / "qwen2.5-7b-luxit-large"),
    ),

    ModelConfig(
        name="qwen2.5-0.5b-luxit-large",
        base_model="unsloth/Qwen2.5-0.5B-Instruct",
        lora_path=str(LORA_DIR / "qwen2.5-0.5b-luxit-large"),
    ),

    ModelConfig(
        name="qwen2.5-1.5b-luxit-large",
        base_model="unsloth/Qwen2.5-1.5B-Instruct",
        lora_path=str(LORA_DIR / "qwen2.5-1.5b-luxit-large"),
    ),
    
    # GLM with LoRA (uses FastLanguageModel)
    ModelConfig(
        name="glm-4-9b-0414-luxit-large",
        base_model="THUDM/GLM-4-9B-0414",
        lora_path=str(LORA_DIR / "glm-4-9b-0414-luxit-large"),
    ),

    # Mistral with LoRA (uses FastLanguageModel)
    ModelConfig(
        name="mistral-7b-luxit-large",
        base_model="unsloth/mistral-7b-instruct-v0.3",
        lora_path=str(LORA_DIR / "mistral-7b-luxit-large"),
    ),

    # Llama3 with LoRA (uses FastLanguageModel)
    ModelConfig(
        name="llama-3.1-8b-luxit-large",
        base_model="unsloth/Meta-Llama-3.1-8B-Instruct",
        lora_path=str(LORA_DIR / "llama-3.1-8b-luxit-large"),
        # chat_template=None means use base model's default template
    ),

    # No max_grad_norm during LoRA training
    ModelConfig(
        name="llama-3.1-8b-luxit-large-no_grad",
        base_model="unsloth/Meta-Llama-3.1-8B-Instruct",
        lora_path=str(LORA_DIR / "llama-3.1-8b-luxit-large-no_grad"),
        # chat_template=None means use base model's default template
    ),

    ModelConfig(
        name="llama-3.2-1b-luxit-large",
        base_model="unsloth/Llama-3.2-1B-Instruct",
        lora_path=str(LORA_DIR / "llama-3.2-1b-luxit-large"),
        # chat_template=None means use base model's default template
    ),

    # Phi-4 with LoRA (uses FastLanguageModel)
    ModelConfig(
        name="phi-4-luxit-large",
        base_model="unsloth/phi-4",
        lora_path=str(LORA_DIR / "phi-4-(14b)-luxit-large"),
        # chat_template=None means use base model's default template
    ),
    
]