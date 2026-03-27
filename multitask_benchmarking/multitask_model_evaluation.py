#!/usr/bin/env python3

"""
Luxembourgish NLP Model Evaluation Script

This script evaluates fine-tuned models on various NLP tasks including:
- Intent Classification
- Recognizing Textual Entailment (RTE)
- Sentiment Analysis
- Sentence Negation
- Stanford Sentiment Treebank (SST)
- Winograd Natural Language Inference (WNLI)

Supports:
- Unsloth FastModel (Gemma)
- Unsloth FastLanguageModel (Llama)
- Standard HuggingFace models (Apertus, etc.)
- Custom LoRA adapters
"""

import json
import logging
import os
import sys
import traceback
import signal
import atexit
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import random
import warnings
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_file: str = "evaluation.log", level: int = logging.INFO):
    """Setup logging to both file and console."""
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

logger = logging.getLogger(__name__)


# ENUMS AND DATA CLASSES


class ModelType(Enum):
    """Supported model types."""
    UNSLOTH_FAST_MODEL = "unsloth_fast_model"  # For Gemma
    UNSLOTH_FAST_LANGUAGE_MODEL = "unsloth_fast_language_model"  # For Llama
    HUGGINGFACE = "huggingface"  # For standard HF models like Apertus


class TaskType(Enum):
    """Supported evaluation tasks."""
    INTENT_CLASSIFICATION = "intent_classification"
    RECOGNIZING_TEXTUAL_ENTAILMENT = "rte"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    SENTENCE_NEGATION = "sentence_negation"
    STANFORD_SENTIMENT_TREEBANK = "sst"
    WINOGRAD_NLI = "wnli"


@dataclass
class ModelConfig:
    """Configuration for a model to evaluate."""
    name: str
    model_path: str
    model_type: ModelType
    lora_path: Optional[str] = None
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    dtype: Optional[str] = None  # None for auto, "float16", "bfloat16"
    use_chat_template: bool = True  # Whether to use tokenizer's chat template
    # Unsloth-specific chat template name (e.g., "llama-3", "llama-3.1", "gemma", "mistral", "chatml", etc.)
    # If set, will use unsloth.chat_templates.get_chat_template() to apply the template
    # This ensures the same template used during fine-tuning is used during inference
    unsloth_chat_template: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.model_type, str):
            self.model_type = ModelType(self.model_type)


@dataclass
class TaskConfig:
    """Configuration for an evaluation task."""
    task_type: TaskType
    dataset_path: str
    label_column: str = "label"
    text_column: str = "text"
    text_column_2: Optional[str] = None  # For pair tasks like RTE, WNLI
    labels: List[str] = field(default_factory=list)
    # Prompt configuration
    use_system_message: bool = False  # Set to False if not fine-tuned with system messages
    system_message: Optional[str] = None  # Custom system message (only if use_system_message=True)
    use_luxembourgish_templates: bool = True  # Use built-in Luxembourgish instruction templates
    # Custom templates (override defaults if set)
    instruction_template: Optional[str] = None  # Custom instruction with {text}, {text_2} placeholders
    few_shot_format: Optional[str] = None  # Custom few-shot format with {text}, {text_2}, {label} placeholders

    def __post_init__(self):
        if isinstance(self.task_type, str):
            self.task_type = TaskType(self.task_type)


@dataclass
class EvaluationResult:
    """Results from a single evaluation run."""
    model_name: str
    task_name: str
    shot_type: str  # e.g. "zero_shot_0", "few_shot_3"
    num_shots: int
    f1_micro: float
    f1_macro: float
    f1_weighted: float
    accuracy: float
    num_samples: int                # Samples entering metric calculation (valid GT labels). Unparseable predictions within this count are treated as wrong
    num_total_samples: int = 0      # Total samples in the test set (0 = unknown, e.g. loaded from old checkpoint)
    num_failed: int = 0             # Samples that could not be evaluated (predicted "unknown")
    predictions: List[Any] = field(default_factory=list)
    ground_truth: List[Any] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        result = asdict(self)
        # Remove large lists for summary
        result.pop('predictions', None)
        result.pop('ground_truth', None)
        return result


@dataclass
class CheckpointState:
    """State for checkpoint recovery."""
    completed_evaluations: List[Dict] = field(default_factory=list)
    current_model: Optional[str] = None
    current_task: Optional[str] = None
    current_shot_type: Optional[str] = None
    partial_predictions: List[Any] = field(default_factory=list)
    partial_ground_truth: List[Any] = field(default_factory=list)
    last_sample_idx: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# MODEL LOADING

class ModelLoader:
    """Handles loading models from various sources."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._loaded_model_name = None

    def load_model(self, config: ModelConfig) -> Tuple[Any, Any]:
        """
        Load a model based on its configuration.
        Args:
            config: ModelConfig specifying how to load the model
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model: {config.name}")
        logger.info(f"  Model path: {config.model_path}")
        logger.info(f"  Model type: {config.model_type.value}")
        logger.info(f"  LoRA path: {config.lora_path}")
        # Clear previous model from memory
        self._clear_model()
        try:
            if config.model_type == ModelType.UNSLOTH_FAST_MODEL:
                model, tokenizer = self._load_unsloth_fast_model(config)
            elif config.model_type == ModelType.UNSLOTH_FAST_LANGUAGE_MODEL:
                model, tokenizer = self._load_unsloth_fast_language_model(config)
            elif config.model_type == ModelType.HUGGINGFACE:
                model, tokenizer = self._load_huggingface_model(config)
            else:
                raise ValueError(f"Unknown model type: {config.model_type}")
            self.model = model
            self.tokenizer = tokenizer
            self._loaded_model_name = config.name
            logger.info(f"Successfully loaded model: {config.name}")
            return model, tokenizer
        except Exception as e:
            logger.error(f"Failed to load model {config.name}: {e}")
            logger.error(traceback.format_exc())
            raise

    def _load_unsloth_fast_model(self, config: ModelConfig) -> Tuple[Any, Any]:
        """Load model using Unsloth's FastModel (for Gemma)."""
        try:
            from unsloth import FastModel
        except ImportError:
            logger.error("Unsloth not installed. Install with: pip install unsloth")
            raise
        dtype = None
        if config.dtype == "float16":
            dtype = torch.float16
        elif config.dtype == "bfloat16":
            dtype = torch.bfloat16
        model, tokenizer = FastModel.from_pretrained(
            model_name=config.model_path,
            max_seq_length=config.max_seq_length,
            load_in_4bit=config.load_in_4bit,
            dtype=dtype,
        )
        # Load LoRA adapter if specified
        if config.lora_path:
            logger.info(f"Loading LoRA adapter from: {config.lora_path}")
            # Load tokenizer from LoRA path if it exists
            # This ensures we use the same chat template used during fine-tuning
            lora_tokenizer_path = Path(config.lora_path)
            if (lora_tokenizer_path / "tokenizer_config.json").exists():
                logger.info(f"Loading tokenizer from LoRA path to preserve chat template")
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(config.lora_path)
                logger.info(f"Tokenizer chat template: {getattr(tokenizer, 'chat_template', 'default')[:100] if getattr(tokenizer, 'chat_template', None) else 'None'}...")
            else:
                logger.warning(f"No tokenizer found in LoRA path. Using base model tokenizer. "
                             f"This may cause issues if you used a custom chat template during fine-tuning!")
            # Load the LoRA adapter weights
            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, config.lora_path)
                logger.info("LoRA adapter loaded successfully with PeftModel")
            except Exception as e:
                logger.warning(f"PeftModel loading failed, trying alternative: {e}")
                try:
                    model.load_adapter(config.lora_path)
                    logger.info("LoRA adapter loaded with load_adapter()")
                except Exception as e2:
                    logger.error(f"Failed to load LoRA adapter: {e2}")
                    raise
        # Apply Unsloth chat template if specified
        # This overrides any template loaded from the tokenizer
        if config.unsloth_chat_template:
            logger.info(f"Applying Unsloth chat template: {config.unsloth_chat_template}")
            try:
                from unsloth.chat_templates import get_chat_template
                tokenizer = get_chat_template(
                    tokenizer,
                    chat_template=config.unsloth_chat_template,
                )
                logger.info(f"Unsloth chat template '{config.unsloth_chat_template}' applied successfully")
            except ImportError:
                logger.error("Could not import unsloth.chat_templates. Make sure Unsloth is installed correctly.")
                raise
            except Exception as e:
                logger.error(f"Failed to apply Unsloth chat template: {e}")
                raise
        # Enable inference mode
        FastModel.for_inference(model)
        return model, tokenizer

    def _load_unsloth_fast_language_model(self, config: ModelConfig) -> Tuple[Any, Any]:
        """Load model using Unsloth's FastLanguageModel (for Llama)."""
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            logger.error("Unsloth not installed. Install with: pip install unsloth")
            raise
        dtype = None
        if config.dtype == "float16":
            dtype = torch.float16
        elif config.dtype == "bfloat16":
            dtype = torch.bfloat16
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_path,
            max_seq_length=config.max_seq_length,
            load_in_4bit=config.load_in_4bit,
            dtype=dtype,
        )
        # Load LoRA adapter if specified
        if config.lora_path:
            logger.info(f"Loading LoRA adapter from: {config.lora_path}")
            # Load tokenizer from LoRA path if it exists there
            # This ensures we use the same chat template used during fine-tuning
            lora_tokenizer_path = Path(config.lora_path)
            if (lora_tokenizer_path / "tokenizer_config.json").exists():
                logger.info(f"Loading tokenizer from LoRA path to preserve chat template")
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(config.lora_path)
                logger.info(f"Tokenizer chat template: {getattr(tokenizer, 'chat_template', 'default')[:100] if getattr(tokenizer, 'chat_template', None) else 'None'}...")
            else:
                logger.warning(f"No tokenizer found in LoRA path. Using base model tokenizer. "
                             f"This may cause issues if you used a custom chat template during fine-tuning!")
            # Load the LoRA adapter weights
            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, config.lora_path)
                logger.info("LoRA adapter loaded successfully with PeftModel")
            except Exception as e:
                logger.warning(f"PeftModel loading failed, trying alternative: {e}")
                try:
                    model.load_adapter(config.lora_path)
                    logger.info("LoRA adapter loaded with load_adapter()")
                except Exception as e2:
                    logger.error(f"Failed to load LoRA adapter: {e2}")
                    raise
        # Apply Unsloth chat template if specified
        # This overrides any template loaded from the tokenizer
        if config.unsloth_chat_template:
            logger.info(f"Applying Unsloth chat template: {config.unsloth_chat_template}")
            try:
                from unsloth.chat_templates import get_chat_template
                tokenizer = get_chat_template(
                    tokenizer,
                    chat_template=config.unsloth_chat_template,
                )
                logger.info(f"Unsloth chat template '{config.unsloth_chat_template}' applied successfully")
            except ImportError:
                logger.error("Could not import unsloth.chat_templates. Make sure Unsloth is installed correctly.")
                raise
            except Exception as e:
                logger.error(f"Failed to apply Unsloth chat template: {e}")
                raise
        # Enable inference mode
        FastLanguageModel.for_inference(model)
        return model, tokenizer

    def _load_huggingface_model(self, config: ModelConfig) -> Tuple[Any, Any]:
        """Load model using standard HuggingFace transformers."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        # Configure quantization if needed
        quantization_config = None
        if config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        # Determine dtype
        dtype = torch.float16
        if config.dtype == "bfloat16":
            dtype = torch.bfloat16
        elif config.dtype == "float32":
            dtype = torch.float32
        # Determine tokenizer path - prefer LoRA path if it has a tokenizer
        tokenizer_path = config.model_path
        if config.lora_path:
            lora_tokenizer_path = Path(config.lora_path)
            if (lora_tokenizer_path / "tokenizer_config.json").exists():
                tokenizer_path = config.lora_path
                logger.info(f"Loading tokenizer from LoRA path to preserve chat template")
            else:
                logger.warning(f"No tokenizer found in LoRA path. Using base model tokenizer.")
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True
        )
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            quantization_config=quantization_config,
            torch_dtype=dtype if not config.load_in_4bit else None,
            device_map="auto",
            trust_remote_code=True
        )
        # Load LoRA adapter if specified
        if config.lora_path:
            logger.info(f"Loading LoRA adapter from: {config.lora_path}")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, config.lora_path)
            logger.info("LoRA adapter loaded successfully")
        model.eval()
        return model, tokenizer

    def _clear_model(self):
        """Clear current model from memory."""
        if self.model is not None:
            logger.info(f"Clearing model from memory: {self._loaded_model_name}")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self._loaded_model_name = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# DATASET LOADING

class DatasetLoader:
    """Handles loading evaluation datasets."""

    SUPPORTED_EXTENSIONS = ['.json', '.tsv', '.csv', '.jsonl']

    @staticmethod
    def _find_dataset_file(dataset_path: str, split: str) -> Path:
        """
        Find the dataset file for a given split, checking multiple extensions.

        Args:
            dataset_path: Path to dataset directory
            split: Which split to load ("train", "dev", "test")
        Returns:
            Path to the found file
        Raises:
            FileNotFoundError: If no matching file is found
        """
        base_path = Path(dataset_path)
        for ext in DatasetLoader.SUPPORTED_EXTENSIONS:
            file_path = base_path / f"{split}{ext}"
            if file_path.exists():
                return file_path
        # If not found, raise error with helpful message
        tried_files = [str(base_path / f"{split}{ext}") for ext in DatasetLoader.SUPPORTED_EXTENSIONS]
        raise FileNotFoundError(
            f"Dataset file not found for split '{split}'. "
            f"Looked for: {', '.join(tried_files)}"
        )

    @staticmethod
    def _load_json(file_path: Path) -> List[Dict]:
        """Load data from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Handle both list format and dict format with 'data' key
        if isinstance(data, dict):
            if 'data' in data:
                data = data['data']
            else:
                # Convert single dict to list
                data = [data]
        return data

    @staticmethod
    def _load_jsonl(file_path: Path) -> List[Dict]:
        """Load data from JSON Lines file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    data.append(json.loads(line))
        return data

    @staticmethod
    def _load_tsv(file_path: Path) -> List[Dict]:
        """Load data from TSV file."""
        import csv
        data = []
        with open(file_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                # Convert empty strings to None and strip whitespace
                cleaned_row = {
                    k: (v.strip() if v and v.strip() else None)
                    for k, v in row.items()
                }
                data.append(cleaned_row)
        return data

    @staticmethod
    def _load_csv(file_path: Path) -> List[Dict]:
        """Load data from CSV file."""
        import csv
        data = []
        with open(file_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                # Convert empty strings to None and strip whitespace
                cleaned_row = {
                    k: (v.strip() if v and v.strip() else None)
                    for k, v in row.items()
                }
                data.append(cleaned_row)
        return data

    @staticmethod
    def load_dataset(dataset_path: str, split: str = "test") -> List[Dict]:
        """
        Load a dataset split from disk. Supports JSON, JSONL, TSV, and CSV formats.

        Args:
            dataset_path: Path to dataset directory containing train/dev/test files
            split: Which split to load ("train", "dev", "test")
        Returns:
            List of dictionaries containing the data
        """
        file_path = DatasetLoader._find_dataset_file(dataset_path, split)
        logger.info(f"Loading dataset from: {file_path}")
        # Load based on file extension
        ext = file_path.suffix.lower()
        if ext == '.json':
            data = DatasetLoader._load_json(file_path)
        elif ext == '.jsonl':
            data = DatasetLoader._load_jsonl(file_path)
        elif ext == '.tsv':
            data = DatasetLoader._load_tsv(file_path)
        elif ext == '.csv':
            data = DatasetLoader._load_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        logger.info(f"Loaded {len(data)} samples from {split} split ({ext} format)")
        return data

    @staticmethod
    def load_few_shot_examples(dataset_path: str, num_shots: int,
                                seed: int = 42,
                                label_column: str = "label") -> List[Dict]:
        """
        Load few-shot examples from the training set.
        If num_shots >= number of unique labels, performs stratified sampling
        to ensure all classes are represented. This is important for tasks with
        many labels where random sampling would
        routinely leave most labels undemonstrated.
        If num_shots < number of unique labels, falls back to random sampling
        """
        train_data = DatasetLoader.load_dataset(dataset_path, split="train")
        random.seed(seed)
        if len(train_data) < num_shots:
            logger.warning(f"Training set has only {len(train_data)} samples, "
                        f"requested {num_shots} shots")
            num_shots = len(train_data)
        # Group by label
        from collections import defaultdict
        buckets: Dict[Any, List[Dict]] = defaultdict(list)
        for ex in train_data:
            buckets[ex[label_column]].append(ex)
        unique_labels = list(buckets.keys())
        num_unique = len(unique_labels)
        if num_shots < num_unique:
            # Can't cover all labels, fall back to random sampling and warn
            logger.warning(
                f"num_shots ({num_shots}) < unique labels ({num_unique}). "
                f"Falling back to random sampling; not all labels will be demonstrated."
            )
            return random.sample(train_data, num_shots)
        # Stratified: shuffle within each bucket, then do round-robin selection
        # so examples are spread as evenly as possible across labels.
        for label in unique_labels:
            random.shuffle(buckets[label])
        examples = []
        label_cycle = list(unique_labels)
        random.shuffle(label_cycle)
        pointers = {label: 0 for label in unique_labels}
        for i in range(num_shots):
            label = label_cycle[i % num_unique]
            # If we've exhausted examples for this label, wrap around
            idx = pointers[label] % len(buckets[label])
            examples.append(buckets[label][idx])
            pointers[label] += 1
        logger.info(
            f"Stratified few-shot sampling: {num_shots} examples "
            f"across {num_unique} labels (seed={seed})"
        )
        return examples


# PROMPT TEMPLATES

class PromptTemplates:
    """Task-specific prompt templates with support for Luxembourgish."""

    @staticmethod
    def get_template(task_type: TaskType, luxembourgish: bool = False) -> Dict[str, str]:
        """
        Get prompt templates for a specific task.

        The Luxembourgish templates are the ground truth. English templates mirror
        the same label mappings so scores are consistent regardless of language setting.

        Args:
            task_type: The type of task
            luxembourgish: If True, return Luxembourgish templates
        Returns:
            Dictionary with 'system', 'instruction', and 'few_shot_format' keys
        """
        # English templates — label mappings kept in sync with the Luxembourgish templates below
        english_templates = {
            TaskType.INTENT_CLASSIFICATION: {
                "system": "You are an expert at classifying the intent of Luxembourgish text. Respond with only the intent label.",
                "instruction": "Classify the intent of the following Luxembourgish text.\n\nText: {text}\n\nIntent:",
                "few_shot_format": "Text: {text}\nIntent: {label}\n\n",
            },
            TaskType.RECOGNIZING_TEXTUAL_ENTAILMENT: {
                "system": "You are an expert at determining textual entailment in Luxembourgish. Respond with '1' for entailment or '0' for not_entailment.",
                "instruction": "Determine if the hypothesis is entailed by the premise.\n\nPremise: {text}\nHypothesis: {text_2}\n\nAnswer:",
                "few_shot_format": "Premise: {text}\nHypothesis: {text_2}\nAnswer: {label}\n\n",
            },
            TaskType.SENTIMENT_ANALYSIS: {
                # label mapping now matches the Luxembourgish template ('0' positive, '1' negative, '2' neutral)
                "system": "You are an expert at analyzing sentiment in Luxembourgish text. Respond with '0' for positive, '1' for negative, or '2' for neutral.",
                "instruction": "Analyze the sentiment of the following Luxembourgish text.\n\nText: {text}\n\nSentiment:",
                "few_shot_format": "Text: {text}\nSentiment: {label}\n\n",
            },
            TaskType.SENTENCE_NEGATION: {
                # range now starts at '0' to match the Luxembourgish template ('0' to '13')
                "system": "You are an expert at negating sentences in Luxembourgish. You can correctly place the word 'net' in order to turn the sentence negative. Respond with the correct position of the word 'net' in the sentence, ranging from '0' to '13'",
                "instruction": "Determine where to put the word 'net' in the following Luxembourgish sentence.\n\nSentence: {text}\n\nAnswer:",
                "few_shot_format": "Sentence: {text}\nAnswer: {label}\n\n",
            },
            TaskType.STANFORD_SENTIMENT_TREEBANK: {
                "system": "You are an expert at sentiment analysis in Luxembourgish. Respond with '1' for positive, '0' for negative",
                "instruction": "Classify the sentiment of the following Luxembourgish text.\n\nText: {text}\n\nSentiment:",
                "few_shot_format": "Text: {text}\nSentiment: {label}\n\n",
            },
            TaskType.WINOGRAD_NLI: {
                "system": "You are an expert at Winograd-style natural language inference in Luxembourgish. Respond with '1' for correct_antecedent or '0' for wrong_antecedent.",
                "instruction": "You will be given a 'Winograd Schema' — a pair of sentences that differ by only one or two words. These words create a semantic shift that changes which noun a specific pronoun refers to. Your goal is to identify the correct antecedent (the noun the pronoun refers to) based on common-sense reasoning and the logical constraints provided by the context.\n\nSentence 1: {text}\nSentence 2: {text_2}\n\nAnswer:",
                "few_shot_format": "Sentence 1: {text}\nSentence 2: {text_2}\nAnswer: {label}\n\n",
            },
        }
        # Luxembourgish templates — ground truth for label mappings
        luxembourgish_templates = {
            TaskType.INTENT_CLASSIFICATION: {
                "system": None,  # No system message
                "instruction": "Klassifizéier d'Intentioun vum follgende lëtzebuergeschen Text. Äntwer nëmme mam Intentiouns-Label.\n\nText: {text}\n\nIntentioun:",
                "few_shot_format": "Text: {text}\nIntentioun: {label}\n\n",
            },
            TaskType.RECOGNIZING_TEXTUAL_ENTAILMENT: {
                "system": None,
                "instruction": "Bestëmm ob d'Hypothees vun der Prämiss ofgeleet ka ginn. Äntwer mat '1' fir kann ofgeleet ginn oder '0' fir kann net ofgeleet ginn.\n\nPrämiss: {text}\nHypothees: {text_2}\n\nÄntwert:",
                "few_shot_format": "Prämiss: {text}\nHypothees: {text_2}\nÄntwert: {label}\n\n",
            },
            TaskType.SENTIMENT_ANALYSIS: {
                "system": None,
                "instruction": "Analyséier de Sentiment vum follgende lëtzebuergeschen Text. Äntwert mat '0' fir positiv, '1' fir negativ oder '2' fir neutral.\n\nText: {text}\n\nSentiment:",
                "few_shot_format": "Text: {text}\nSentiment: {label}\n\n",
            },
            TaskType.SENTENCE_NEGATION: {
                "system": None,
                "instruction": "Bestëmm, wou d'Wuert 'net' am follgende lëtzebuergesche Saz stoe muss, fir de Saz negativ ze maachen. Äntwert mat der korrekter Positioun vum Wuert 'net' am Saz, am Beräich vun '0' bis '13'. \n\nSaz: {text}\n\nÄntwert:",
                "few_shot_format": "Saz: {text}\nÄntwert: {label}\n\n",
            },
            TaskType.STANFORD_SENTIMENT_TREEBANK: {
                "system": None,
                "instruction": "Klassifizéier de Sentiment vum follgende lëtzebuergeschen Text. Äntwer mat '1' fir positiv, '0' fir negativ.\n\nText: {text}\n\nSentiment:",
                "few_shot_format": "Text: {text}\nSentiment: {label}\n\n",
            },
            TaskType.WINOGRAD_NLI: {
                "system": None,
                "instruction": "Du kriss e 'Winograd-Schema', e Sazpuer, dat sech nëmmen an engem oder zwee Wierder ënnerscheet. Dës Wierder verursaachen eng semantesch Verschibung, déi ännert, op wéi en Nomen sech e bestëmmte Pronomen bezitt. Dain Zil ass et, de richtegen Antezedent (den Nomen, op dee sech de Pronomen bezitt) ze identifizéieren, baséiert op gesondem Mënscheverstand an de logesche Restriktiounen, déi duerch de Kontext virgi ginn. Äntwer mat '1', wann s du de korrekten Antezedent identifizéiert hues, a mat '0' fir de falschen Antezedent.\n\nSaz 1: {text}\nSaz 2: {text_2}\n\nÄntwert:",
                "few_shot_format": "Saz 1: {text}\nSaz 2: {text_2}\nÄntwert: {label}\n\n",
            },
        }
        templates = luxembourgish_templates if luxembourgish else english_templates
        return templates.get(task_type, english_templates[TaskType.INTENT_CLASSIFICATION])

    @staticmethod
    def build_prompt(task_config: TaskConfig, sample: Dict,
                     few_shot_examples: Optional[List[Dict]] = None) -> str:
        """
        Build a complete prompt for evaluation (non-chat format).

        Args:
            task_config: Configuration for the task
            sample: The sample to evaluate
            few_shot_examples: Optional few-shot examples
        Returns:
            Complete prompt string
        """
        template = PromptTemplates.get_template(
            task_config.task_type,
            luxembourgish=task_config.use_luxembourgish_templates
        )
        # Use custom templates if provided in task_config, otherwise use defaults
        instruction_template = task_config.instruction_template or template["instruction"]
        few_shot_template = task_config.few_shot_format or template["few_shot_format"]
        prompt_parts = []
        # Add system message only if use_system_message is True
        if task_config.use_system_message:
            system_msg = task_config.system_message if task_config.system_message is not None else template.get("system")
            if system_msg:
                prompt_parts.append(system_msg)
                prompt_parts.append("")
        # Add labels info if available (convert to strings for display)
        if task_config.labels:
            str_labels = [str(label) for label in task_config.labels]
            label_prefix = "Méiglech Labelen" if task_config.use_luxembourgish_templates else "Possible labels"
            prompt_parts.append(f"{label_prefix}: {', '.join(str_labels)}")
            prompt_parts.append("")
        # Add few-shot examples if provided
        if few_shot_examples:
            examples_header = "Beispiller:" if task_config.use_luxembourgish_templates else "Examples:"
            classify_prompt = "Elo klassifizéieren:" if task_config.use_luxembourgish_templates else "Now classify:"
            prompt_parts.append(examples_header)
            prompt_parts.append("")
            for example in few_shot_examples:
                # Convert label to string in case it's numeric
                label_val = example.get(task_config.label_column, "")
                example_text = few_shot_template.format(
                    text=example.get(task_config.text_column, ""),
                    text_2=example.get(task_config.text_column_2, "") if task_config.text_column_2 else "",
                    label=str(label_val) if label_val is not None else ""
                )
                prompt_parts.append(example_text)
            prompt_parts.append(classify_prompt)
            prompt_parts.append("")
        # Add the actual sample
        instruction = instruction_template.format(
            text=sample.get(task_config.text_column, ""),
            text_2=sample.get(task_config.text_column_2, "") if task_config.text_column_2 else ""
        )
        prompt_parts.append(instruction)
        return "\n".join(prompt_parts)

    @staticmethod
    def build_chat_messages(task_config: TaskConfig, sample: Dict,
                            few_shot_examples: Optional[List[Dict]] = None) -> List[Dict[str, str]]:
        """
        Build messages list for chat template formatting.

        Label information is always injected into the conversation so the model
        knows the valid output space:
        - If use_system_message is True, labels are appended to the system message.
        - If use_system_message is False, labels are prepended to the FIRST user
          message regardless of whether few-shot examples are present.

        Args:
            task_config: Configuration for the task
            sample: The sample to evaluate
            few_shot_examples: Optional few-shot examples
        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        template = PromptTemplates.get_template(
            task_config.task_type,
            luxembourgish=task_config.use_luxembourgish_templates
        )
        messages = []
        # Use custom templates if provided in task_config, otherwise use defaults
        instruction_template = task_config.instruction_template or template["instruction"]
        few_shot_template = task_config.few_shot_format or template["few_shot_format"]
        # Label prefix based on language
        label_prefix = "Méiglech Labelen" if task_config.use_luxembourgish_templates else "Possible labels"
        str_labels = [str(label) for label in task_config.labels] if task_config.labels else []
        label_line = f"{label_prefix}: {', '.join(str_labels)}" if str_labels else ""
        # Add system message only if use_system_message is True
        if task_config.use_system_message:
            system_msg = task_config.system_message if task_config.system_message is not None else template.get("system")
            if system_msg:
                # Append label info to system message if available
                if label_line:
                    system_msg += f"\n\n{label_line}"
                messages.append({"role": "system", "content": system_msg})
        # Add few-shot examples as user/assistant pairs
        if few_shot_examples:
            for example in few_shot_examples:
                # User turn with the example input
                example_instruction = instruction_template.format(
                    text=example.get(task_config.text_column, ""),
                    text_2=example.get(task_config.text_column_2, "") if task_config.text_column_2 else ""
                )
                messages.append({"role": "user", "content": example_instruction})
                # Assistant turn with the correct label (convert to string)
                label_val = example.get(task_config.label_column, "")
                messages.append({"role": "assistant", "content": str(label_val) if label_val is not None else ""})
        # Final user message with the actual sample
        final_instruction = instruction_template.format(
            text=sample.get(task_config.text_column, ""),
            text_2=sample.get(task_config.text_column_2, "") if task_config.text_column_2 else ""
        )
        messages.append({"role": "user", "content": final_instruction})
        # When there is no system message, prepend the label info to the FIRST user
        # message so the model always knows the valid output space regardless of whether
        # few-shot examples are present.
        if not task_config.use_system_message and label_line:
            for i, msg in enumerate(messages):
                if msg["role"] == "user":
                    messages[i] = {"role": "user", "content": f"{label_line}\n\n{msg['content']}"}
                    break
        return messages

    @staticmethod
    def apply_chat_template(tokenizer: Any, messages: List[Dict[str, str]],
                           add_generation_prompt: bool = True) -> str:
        """
        Apply the tokenizer's chat template to format messages.

        Args:
            tokenizer: The tokenizer with chat_template
            messages: List of message dicts
            add_generation_prompt: Whether to add the generation prompt
        Returns:
            Formatted prompt string
        """
        try:
            # Check if tokenizer has a chat template
            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
                formatted = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt
                )
                return formatted
            else:
                # Fallback to simple concatenation if no chat template
                logger.warning("Tokenizer has no chat_template, using fallback formatting")
                parts = []
                for msg in messages:
                    role = msg['role'].upper()
                    content = msg['content']
                    parts.append(f"{role}: {content}")
                return "\n\n".join(parts) + "\n\nASSISTANT:"
        except Exception as e:
            logger.warning(f"Failed to apply chat template: {e}. Using fallback.")
            parts = []
            for msg in messages:
                role = msg['role'].upper()
                content = msg['content']
                parts.append(f"{role}: {content}")
            return "\n\n".join(parts) + "\n\nASSISTANT:"


# EVALUATION ENGINE

class EvaluationEngine:
    """Main evaluation engine."""

    def __init__(self, checkpoint_file: str = "eval_checkpoint.json"):
        self.model_loader = ModelLoader()
        self.checkpoint_file = checkpoint_file
        self.checkpoint_state = self._load_checkpoint()
        self.results: List[EvaluationResult] = []
        # Setup signal handlers for graceful interruption
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self._save_checkpoint)

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        logger.warning(f"Received signal {signum}. Saving checkpoint and exiting...")
        self._save_checkpoint()
        sys.exit(1)

    def _load_checkpoint(self) -> CheckpointState:
        """Load checkpoint from disk."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded checkpoint from {self.checkpoint_file}")
                logger.info(f"  Completed evaluations: {len(data.get('completed_evaluations', []))}")
                return CheckpointState(**data)
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
        return CheckpointState()

    def _save_checkpoint(self):
        """Save checkpoint to disk."""
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(asdict(self.checkpoint_state), f, indent=2)
            logger.info(f"Checkpoint saved to {self.checkpoint_file}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def _is_completed(self, model_name: str, task_name: str, shot_type: str) -> bool:
        """Check if an evaluation has already been completed."""
        for result in self.checkpoint_state.completed_evaluations:
            if (result.get('model_name') == model_name and
                result.get('task_name') == task_name and
                result.get('shot_type') == shot_type):
                return True
        return False

    def _generate_response(self, model: Any, tokenizer: Any,
                          prompt: str, max_new_tokens: int = 50) -> str:
        """
        Generate a response from the model.

        Args:
            model: The loaded model
            tokenizer: The tokenizer
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
        Returns:
            Generated text
        """
        # Tokenize input
        inputs = tokenizer(
            text=prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        # Move to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding for consistency
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        # Decode only the new tokens
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.strip()

    def _extract_label(self, response: str, valid_labels: List[Any]) -> str:
        """
        Extract a valid label from the model response.

        Matching priority (highest to lowest):
          1. Exact match of the full response.
          2. Response starts with a valid label (longest match first).
          3. Response contains a valid label as a standalone token (longest match first).

        Labels are sorted by descending string length so that longer labels
        (e.g. "13") are matched before shorter prefixes (e.g. "1"). This
        prevents multi-digit numeric labels from being misclassified.

        Args:
            response: Raw model response
            valid_labels: List of valid labels (can be strings or numbers)
        Returns:
            Extracted label as a string, or 'unknown' if no match found
        """
        import re
        response_clean = response.strip()
        response_lower = response_clean.lower()
        # Convert all labels to strings and strip whitespace for comparison
        str_labels = [str(label).strip() for label in valid_labels]
        # Sort labels by descending length so longer labels are checked
        # first. This prevents "1" from matching before "10", "11", "12", "13".
        sorted_labels = sorted(str_labels, key=len, reverse=True)
        # Exact match (case-insensitive)
        for str_label in sorted_labels:
            if str_label.lower() == response_lower:
                return str_label
        # Response starts with a valid label (longest match first)
        for str_label in sorted_labels:
            if response_lower.startswith(str_label.lower()):
                # Verify the match is at a word boundary to avoid
                # partial matches like "1" matching the start of "10 ..."
                label_len = len(str_label)
                if label_len >= len(response_clean):
                    # Label covers the entire response already matched in exact check,
                    # but handle edge case of whitespace differences
                    return str_label
                # Check the character right after the match is a word boundary
                next_char = response_lower[label_len]
                if not next_char.isalnum() and next_char != '_':
                    return str_label
        # Response contains a valid label as a standalone token
        # Use word-boundary regex to avoid substring false positives
        # (e.g. "order" matching inside "order_card").
        for str_label in sorted_labels:
            # Escape the label for regex safety and match with word boundaries
            pattern = r'(?<![a-zA-Z0-9_])' + re.escape(str_label.lower()) + r'(?![a-zA-Z0-9_])'
            if re.search(pattern, response_lower):
                return str_label
        logger.debug(f"Could not extract label from response: '{response}'")
        return "unknown"

    def evaluate_task(self, model: Any, tokenizer: Any,
                      model_config: ModelConfig, task_config: TaskConfig,
                      shot_type: str = "zero_shot", num_shots: int = 0,
                      shot_label: Optional[str] = None,
                      resume_from_idx: int = 0) -> EvaluationResult:
        """
        Evaluate a model on a single task.

        Args:
            model: Loaded model
            tokenizer: Tokenizer
            model_config: Model configuration
            task_config: Task configuration
            shot_type: "zero_shot" or "few_shot" — controls whether few-shot
                       examples are loaded; not stored in the result directly
            num_shots: Number of few-shot examples (only for few_shot)
            shot_label: Human-readable label stored in the result (e.g. "zero_shot_0",
                        "few_shot_3"). Defaults to f"{shot_type}_{num_shots}" if omitted.
            resume_from_idx: Index to resume from (for checkpoint recovery)
        Returns:
            EvaluationResult with metrics
        """
        resolved_shot_label = shot_label if shot_label is not None else f"{shot_type}_{num_shots}"
        task_name = task_config.task_type.value
        logger.info(f"Evaluating {model_config.name} on {task_name} ({resolved_shot_label})")
        # Load test data
        test_data = DatasetLoader.load_dataset(task_config.dataset_path, split="test")
        # Load few-shot examples if needed
        few_shot_examples = None
        if shot_type == "few_shot" and num_shots > 0:
            few_shot_examples = DatasetLoader.load_few_shot_examples(
                task_config.dataset_path, num_shots,
                label_column=task_config.label_column  # uses correct column per task
            )
        # Initialize or resume predictions
        if resume_from_idx > 0:
            predictions = self.checkpoint_state.partial_predictions.copy()
            ground_truth = self.checkpoint_state.partial_ground_truth.copy()
            logger.info(f"Resuming from sample {resume_from_idx}")
        else:
            predictions = []
            ground_truth = []
        # Update checkpoint state
        self.checkpoint_state.current_model = model_config.name
        self.checkpoint_state.current_task = task_name
        self.checkpoint_state.current_shot_type = resolved_shot_label
        # Process samples
        for idx, sample in enumerate(tqdm(test_data[resume_from_idx:],
                                          desc=f"{task_name} ({resolved_shot_label})")):
            actual_idx = idx + resume_from_idx
            try:
                # Build prompt - use chat template if configured
                if model_config.use_chat_template:
                    messages = PromptTemplates.build_chat_messages(
                        task_config, sample, few_shot_examples
                    )
                    prompt = PromptTemplates.apply_chat_template(
                        tokenizer, messages, add_generation_prompt=True
                    )
                else:
                    prompt = PromptTemplates.build_prompt(
                        task_config, sample, few_shot_examples
                    )
                # Generate response
                response = self._generate_response(model, tokenizer, prompt)
                # Extract label
                predicted_label = self._extract_label(response, task_config.labels)
                true_label = sample.get(task_config.label_column, "unknown")
                # Convert ground truth to string for consistent comparison
                true_label = str(true_label) if true_label is not None else "unknown"
                predictions.append(predicted_label)
                ground_truth.append(true_label)
                # Update checkpoint periodically
                if (actual_idx + 1) % 50 == 0:
                    self.checkpoint_state.partial_predictions = predictions.copy()
                    self.checkpoint_state.partial_ground_truth = ground_truth.copy()
                    self.checkpoint_state.last_sample_idx = actual_idx + 1
                    self._save_checkpoint()
            except Exception as e:
                logger.error(f"Error processing sample {actual_idx}: {e}")
                predictions.append("unknown")
                true_label = sample.get(task_config.label_column, "unknown")
                ground_truth.append(str(true_label) if true_label is not None else "unknown")

        # METRIC CALCULATION

        num_total = len(ground_truth)
        # we identify indices where the ground truth is valid.
        # (Only samples where the ground truth is 'unknown' should be truly excluded).
        valid_gt_indices = [i for i, g in enumerate(ground_truth) if g != "unknown"]
        num_valid_gt = len(valid_gt_indices)
        # Only count failures within samples that actually have valid ground truth,
        # since those are the only ones that enter the metric calculation.
        num_failed = sum(1 for i in valid_gt_indices if predictions[i] == "unknown")
        if num_failed > 0:
            logger.warning(
                f"{num_failed}/{num_valid_gt} samples returned 'unknown'. "
                f"These are treated as incorrect predictions."
            )
        if num_valid_gt > 0:
            # Prepare data for sklearn
            current_preds = [predictions[i] for i in valid_gt_indices]
            current_truth = [ground_truth[i] for i in valid_gt_indices]
            # Build the full set of valid label strings from task_config
            all_configured_labels = sorted(
                set([str(label).strip() for label in task_config.labels]),
                key=lambda x: (len(x), x)
            )
            # Map labels to integers for sklearn
            label_to_idx = {l: i for i, l in enumerate(all_configured_labels)}
            # Handle 'unknown' in predictions:
            # We map 'unknown' to a special index that is NOT in the labels list
            # passed to f1_score. This counts as a false negative for the
            # correct class and a false positive for nothing, effectively penalizing F1.
            unknown_mapping_idx = -1
            preds_numeric = [label_to_idx.get(p, unknown_mapping_idx) for p in current_preds]
            truth_numeric = [label_to_idx.get(g, unknown_mapping_idx) for g in current_truth]
            # The indices sklearn should focus on (excludes the unknown_mapping_idx)
            all_label_indices = list(range(len(all_configured_labels)))
            # Metrics
            f1_micro = f1_score(truth_numeric, preds_numeric, average='micro',
                                labels=all_label_indices, zero_division=0)
            f1_macro = f1_score(truth_numeric, preds_numeric, average='macro',
                                labels=all_label_indices, zero_division=0)
            f1_weighted = f1_score(truth_numeric, preds_numeric, average='weighted',
                                    labels=all_label_indices, zero_division=0)
            accuracy = accuracy_score(truth_numeric, preds_numeric)
        else:
            f1_micro = f1_macro = f1_weighted = accuracy = 0.0
        # Create result
        result = EvaluationResult(
            model_name=model_config.name,
            task_name=task_name,
            shot_type=resolved_shot_label,
            num_shots=num_shots,
            f1_micro=f1_micro,
            f1_macro=f1_macro,
            f1_weighted=f1_weighted,
            accuracy=accuracy,
            num_samples=num_valid_gt,  # actual denominator used in F1/accuracy calculation
            num_total_samples=num_total,
            num_failed=num_failed,
            predictions=predictions,
            ground_truth=ground_truth
        )
        logger.info(f"Results for {model_config.name} on {task_name} ({resolved_shot_label}):")
        logger.info(f"  F1 (micro): {f1_micro:.4f}")
        logger.info(f"  F1 (macro): {f1_macro:.4f}")
        logger.info(f"  F1 (weighted): {f1_weighted:.4f}")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Samples: {num_valid_gt - num_failed}/{num_valid_gt} parseable outputs ({num_failed} unparseable, treated as wrong)")
        # Clear partial checkpoint
        self.checkpoint_state.partial_predictions = []
        self.checkpoint_state.partial_ground_truth = []
        self.checkpoint_state.last_sample_idx = 0
        return result

    def run_evaluation(self, model_configs: List[ModelConfig],
                       task_configs: List[TaskConfig],
                       few_shot_counts: List[int] = [0, 3, 5],
                       output_file: str = "evaluation_results.json") -> List[EvaluationResult]:
        """
        Run full evaluation across all models and tasks.

        Args:
            model_configs: List of model configurations
            task_configs: List of task configurations
            few_shot_counts: List of shot counts (0 = zero-shot)
            output_file: Path to save results
        Returns:
            List of all evaluation results
        """
        logger.info("="*80)
        logger.info("STARTING EVALUATION")
        logger.info(f"Models: {[m.name for m in model_configs]}")
        logger.info(f"Tasks: {[t.task_type.value for t in task_configs]}")
        logger.info(f"Shot counts: {few_shot_counts}")
        logger.info("="*80)
        all_results = []
        # Load any previously completed results
        for completed in self.checkpoint_state.completed_evaluations:
            result = EvaluationResult(**completed)
            all_results.append(result)
        for model_config in model_configs:
            try:
                # Load model
                model, tokenizer = self.model_loader.load_model(model_config)
                for task_config in task_configs:
                    for num_shots in few_shot_counts:
                        shot_type = "zero_shot" if num_shots == 0 else "few_shot"
                        shot_label = f"{shot_type}_{num_shots}"
                        # Check if already completed
                        if self._is_completed(model_config.name,
                                             task_config.task_type.value,
                                             shot_label):
                            logger.info(f"Skipping completed: {model_config.name} / "
                                       f"{task_config.task_type.value} / {shot_label}")
                            continue
                        # Check if we need to resume from partial
                        resume_idx = 0
                        if (self.checkpoint_state.current_model == model_config.name and
                            self.checkpoint_state.current_task == task_config.task_type.value and
                            self.checkpoint_state.current_shot_type == shot_label):
                            resume_idx = self.checkpoint_state.last_sample_idx
                        try:
                            # Run evaluation
                            result = self.evaluate_task(
                                model, tokenizer, model_config, task_config,
                                shot_type=shot_type, num_shots=num_shots,
                                shot_label=shot_label,
                                resume_from_idx=resume_idx
                            )
                            all_results.append(result)
                            # Mark as completed in checkpoint
                            self.checkpoint_state.completed_evaluations.append(result.to_dict())
                            self._save_checkpoint()
                        except Exception as e:
                            logger.error(f"Error in evaluation: {e}")
                            logger.error(traceback.format_exc())
                            continue
            except Exception as e:
                logger.error(f"Error loading model {model_config.name}: {e}")
                logger.error(traceback.format_exc())
                continue
        # Save final results
        self._save_results(all_results, output_file)
        return all_results

    def _save_results(self, results: List[EvaluationResult], output_file: str):
        """Save results to JSON file."""
        results_dict = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "results": [r.to_dict() for r in results],
            "summary": self._generate_summary(results)
        }
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        logger.info(f"Results saved to {output_file}")

    def _generate_summary(self, results: List[EvaluationResult]) -> Dict:
        """Generate a summary of results."""
        summary = {
            "by_model": {},
            "by_task": {},
            "overall_f1_macro": 0.0
        }
        for result in results:
            # By model
            if result.model_name not in summary["by_model"]:
                summary["by_model"][result.model_name] = []
            summary["by_model"][result.model_name].append({
                "task": result.task_name,
                "shot_type": result.shot_type,
                "f1_macro": result.f1_macro,
                "num_samples": result.num_samples,
                "num_total_samples": result.num_total_samples,
                "num_failed": result.num_failed,
            })
            # By task
            if result.task_name not in summary["by_task"]:
                summary["by_task"][result.task_name] = []
            summary["by_task"][result.task_name].append({
                "model": result.model_name,
                "shot_type": result.shot_type,
                "f1_macro": result.f1_macro,
                "num_samples": result.num_samples,
                "num_total_samples": result.num_total_samples,
                "num_failed": result.num_failed,
            })
        # Overall average
        if results:
            summary["overall_f1_macro"] = np.mean([r.f1_macro for r in results])
        return summary


# CONFIGURATION BUILDER


def build_default_task_configs(base_path: str) -> List[TaskConfig]:
    """
    Build default task configurations.

    Args:
        base_path: Base path to dataset directories
    Returns:
        List of TaskConfig objects
    """
    tasks = [
        TaskConfig(
            task_type=TaskType.INTENT_CLASSIFICATION,
            dataset_path=os.path.join(base_path, "intent_classification"),
            label_column="intent",
            text_column="text",
            labels=["inform", "question", "request", "greeting", "goodbye", "affirm", "deny", "other"]
        ),
        TaskConfig(
            task_type=TaskType.RECOGNIZING_TEXTUAL_ENTAILMENT,
            dataset_path=os.path.join(base_path, "rte"),
            label_column="label",
            text_column="premise",
            text_column_2="hypothesis",
            labels=["entailment", "not_entailment"]
        ),
        TaskConfig(
            task_type=TaskType.SENTIMENT_ANALYSIS,
            dataset_path=os.path.join(base_path, "sentiment"),
            label_column="sentiment",
            text_column="text",
            labels=["positive", "negative", "neutral"]
        ),
        TaskConfig(
            task_type=TaskType.SENTENCE_NEGATION,
            dataset_path=os.path.join(base_path, "negation"),
            label_column="negated",
            text_column="sentence",
            labels=["negated", "not_negated"]
        ),
        TaskConfig(
            task_type=TaskType.STANFORD_SENTIMENT_TREEBANK,
            dataset_path=os.path.join(base_path, "sst"),
            label_column="label",
            text_column="text",
            labels=["very_negative", "negative", "neutral", "positive", "very_positive"]
        ),
        TaskConfig(
            task_type=TaskType.WINOGRAD_NLI,
            dataset_path=os.path.join(base_path, "wnli"),
            label_column="label",
            text_column="sentence",
            text_column_2="question",
            labels=["entailment", "not_entailment"]
        ),
    ]
    return tasks



# MAIN ENTRY POINT


def main():
    """Main entry point for the evaluation script."""
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned models on NLP tasks")
    parser.add_argument("--config", type=str, help="Path to configuration JSON file")
    parser.add_argument("--datasets", type=str, default="./datasets",
                       help="Base path to dataset directories")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                       help="Output file for results")
    parser.add_argument("--checkpoint", type=str, default="eval_checkpoint.json",
                       help="Checkpoint file for recovery")
    parser.add_argument("--log-file", type=str, default="evaluation.log",
                       help="Log file path")
    parser.add_argument("--shots", type=str, default="0,3,5",
                       help="Comma-separated list of shot counts")
    args = parser.parse_args()
    # Setup logging
    setup_logging(args.log_file)
    logger.info("Starting evaluation script")
    # Parse shot counts
    shot_counts = [int(x.strip()) for x in args.shots.split(",")]
    # Load or build configurations
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        model_configs = [ModelConfig(**m) for m in config.get('models', [])]
        task_configs = [TaskConfig(**t) for t in config.get('tasks', [])]
    else:
        # Use example configurations
        logger.warning("No config file provided, using example configurations")
        model_configs = [
            # Example Gemma model with Unsloth FastModel
            ModelConfig(
                name="gemma-2b-lb-finetuned",
                model_path="google/gemma-2b",
                model_type=ModelType.UNSLOTH_FAST_MODEL,
                lora_path="./models/gemma-2b-lb-lora",
                max_seq_length=2048,
                load_in_4bit=True
            ),
            # Example Llama model with Unsloth FastLanguageModel
            ModelConfig(
                name="llama-3-8b-lb-finetuned",
                model_path="meta-llama/Meta-Llama-3-8B",
                model_type=ModelType.UNSLOTH_FAST_LANGUAGE_MODEL,
                lora_path="./models/llama-3-8b-lb-lora",
                max_seq_length=2048,
                load_in_4bit=True
            ),
            # Example HuggingFace model (no Unsloth)
            ModelConfig(
                name="apertus-8b",
                model_path="apertus/apertus-8b",
                model_type=ModelType.HUGGINGFACE,
                lora_path=None,
                max_seq_length=2048,
                load_in_4bit=True
            ),
        ]
        task_configs = build_default_task_configs(args.datasets)
    # Run evaluation
    engine = EvaluationEngine(checkpoint_file=args.checkpoint)
    results = engine.run_evaluation(
        model_configs=model_configs,
        task_configs=task_configs,
        few_shot_counts=shot_counts,
        output_file=args.output
    )
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    for result in results:
        print(f"\n{result.model_name} | {result.task_name} | {result.shot_type}")
        print(f"  F1 (macro): {result.f1_macro:.4f}")
        print(f"  F1 (micro): {result.f1_micro:.4f}")
        print(f"  F1 (weighted): {result.f1_weighted:.4f}")
        print(f"  Accuracy: {result.accuracy:.4f}")
        print(f"  Samples: {result.num_samples}/{result.num_total_samples} valid ({result.num_failed} failed)")
    print(f"\nResults saved to: {args.output}")
    print(f"Checkpoint file: {args.checkpoint}")


if __name__ == "__main__":
    main()