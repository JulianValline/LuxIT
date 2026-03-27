# -*- coding: utf-8 -*-
"""
Universal Fine-tuning Pipeline for Multiple Model Families with Unsloth
Supports: Qwen, Gemma, GLM, Llama, Phi, EuroLLM, Olmo and Mistral models
Includes: Automatic model detection, family-specific configurations, comprehensive logging
"""

import os
import sys
import json
import yaml
import torch
import logging
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List
from dataclasses import asdict
from configs import ModelFamily, ModelSpecificConfig, ModelConfig, TrainingConfig

# Import Unsloth components - handle both FastModel and FastLanguageModel
try:
    from unsloth import FastLanguageModel
except ImportError:
    FastLanguageModel = None

try:
    from unsloth import FastModel
except ImportError:
    FastModel = None

from unsloth.chat_templates import get_chat_template, train_on_responses_only

from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
from transformers import TrainerCallback
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm
import evaluate


class ModelFamilyDetector:
    """Detects model family and returns appropriate configuration"""
    
    @staticmethod
    def detect_family(model_name: str) -> ModelFamily:
        """Detect model family from model name"""
        model_lower = model_name.lower()
        
        if "qwen" in model_lower:
            return ModelFamily.QWEN
        elif "gemma" in model_lower:
            return ModelFamily.GEMMA
        elif "glm" in model_lower:
            return ModelFamily.GLM
        elif "llama" in model_lower or "meta-llama" in model_lower:
            return ModelFamily.LLAMA
        elif "mistral" in model_lower or "ministral" in model_lower:
            return ModelFamily.MISTRAL
        elif "phi" in model_lower:
            return ModelFamily.PHI
        elif "eurollm" in model_lower:
            return ModelFamily.EUROLLM
        elif "olmo" in model_lower:
            return ModelFamily.OLMO
        else:
            return ModelFamily.UNKNOWN
    
    @staticmethod
    def get_model_config(model_name: str) -> ModelSpecificConfig:
        """Get model-specific configuration based on model family"""
        family = ModelFamilyDetector.detect_family(model_name)
        
        configs = {
            ModelFamily.QWEN: ModelSpecificConfig(
                family=ModelFamily.QWEN,
                use_fast_model=False,
                chat_template= "qwen-2.5", #"chatml", #None,  # Qwen-3 needs chatml because of think tokens
                lora_r=32,
                lora_alpha=32,
                lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj"],
                load_in_4bit=False,
                load_in_8bit=False,
                use_gradient_checkpointing="unsloth",
                remove_bos_prefix=False
            ),
            ModelFamily.GEMMA: ModelSpecificConfig(
                family=ModelFamily.GEMMA,
                use_fast_model=True,  # Gemma uses FastModel
                chat_template="gemma-3",
                lora_r=8,
                lora_alpha=8,
                lora_target_modules=None,  # Gemma uses different parameters
                load_in_4bit=False,
                load_in_8bit=False,
                use_gradient_checkpointing=None,
                remove_bos_prefix=True,  # Since the tokenizer and the chat template both add a BOS token
                # Gemma-specific
                finetune_vision_layers=False,
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True
            ),
            ModelFamily.GLM: ModelSpecificConfig(
                family=ModelFamily.GLM,
                use_fast_model=False,
                chat_template=None,  # GLM doesn't use get_chat_template
                lora_r=16,
                lora_alpha=16,
                lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj"],
                load_in_4bit=True,
                load_in_8bit=False,
                use_gradient_checkpointing="unsloth",
                remove_bos_prefix=False
            ),
            ModelFamily.LLAMA: ModelSpecificConfig(
                family=ModelFamily.LLAMA,
                use_fast_model=False,
                chat_template="llama-3.1",
                lora_r=16,
                lora_alpha=16,
                lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj"],
                load_in_4bit=True,
                load_in_8bit=False,
                use_gradient_checkpointing="unsloth",
                remove_bos_prefix=False
            ),
            ModelFamily.MISTRAL: ModelSpecificConfig(
                family=ModelFamily.MISTRAL,
                use_fast_model=False,
                chat_template="mistral",
                lora_r=16,
                lora_alpha=16,
                lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj"],
                load_in_4bit=True,
                load_in_8bit=False,
                use_gradient_checkpointing="unsloth",
                remove_bos_prefix=True # Since the tokenizer and the chat template both add a BOS token
            ),
            ModelFamily.PHI: ModelSpecificConfig(
                family=ModelFamily.PHI,
                use_fast_model=False,
                chat_template="phi-4",
                lora_r=16,
                lora_alpha=32,
                lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj"],
                load_in_4bit=True,
                load_in_8bit=False,
                use_gradient_checkpointing="unsloth",
                remove_bos_prefix=False
            ),
            ModelFamily.EUROLLM: ModelSpecificConfig(
                family=ModelFamily.EUROLLM,
                use_fast_model=False,
                chat_template="chatml", # Uses the chatml template and there is no specific "eurollm" template in unsloth
                lora_r=16,
                lora_alpha=32,
                lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj"],
                load_in_4bit=False,
                load_in_8bit=False,
                use_gradient_checkpointing="unsloth",
                remove_bos_prefix=False
            ),
            ModelFamily.OLMO: ModelSpecificConfig(
                family=ModelFamily.OLMO,
                use_fast_model=False,
                chat_template=None, # Olmo uses a cutsom version of chatml chat-template, hence we use native template
                lora_r=16,
                lora_alpha=32,
                lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj"],
                load_in_4bit=False,
                load_in_8bit=False,
                use_gradient_checkpointing="unsloth",
                remove_bos_prefix=False
            ),
        }
        
        if family == ModelFamily.UNKNOWN:
            # Default configuration for unknown models
            return ModelSpecificConfig(
                family=ModelFamily.UNKNOWN,
                use_fast_model=False,
                chat_template=None,
                lora_r=16,
                lora_alpha=16,
                load_in_4bit=False,
                load_in_8bit=False
            )
        
        return configs[family]


class EnhancedLogger:
    """Enhanced logging system with file and console outputs"""
    
    def __init__(self, log_dir: str, name: str = "training"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_logging(name)
        
    def setup_logging(self, name: str):
        """Configure logging with both file and console handlers"""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        self.logger.propagate = False  # Prevents double logging

        self.logger.handlers = []
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # File handler for all logs
        all_log_file = self.log_dir / f"training_{self.timestamp}.log"
        file_handler = logging.FileHandler(all_log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logging initialized. Logs saved to {self.log_dir}")
        
    def get_logger(self):
        return self.logger


class MetricsTracker:
    """Track and save training metrics"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": [],
            "epoch": [],
            "step": []
        }
        
    def add_metrics(self, metrics_dict: Dict[str, Any], step: int):
        """Add metrics for a specific step"""
        for key, value in metrics_dict.items():
            if key in self.metrics:
                self.metrics[key].append(value)
        self.metrics["step"].append(step)
        
    def save_metrics(self, filename: str = "training_metrics.json"):
        """Save metrics to JSON file"""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)


class CustomCallback(TrainerCallback):
    """Custom callback for enhanced monitoring and logging"""
    
    def __init__(self, logger, metrics_tracker):
        self.logger = logger
        self.metrics_tracker = metrics_tracker
        self.start_time = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = datetime.now()
        self.logger.info("Training started")
        self.logger.info(f"Total training steps: {state.max_steps}")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.metrics_tracker.add_metrics(logs, state.global_step)
            if "loss" in logs:
                self.logger.info(f"Step {state.global_step}: loss={logs['loss']:.4f}")
            if "eval_loss" in logs:
                self.logger.info(f"Step {state.global_step}: eval_loss={logs['eval_loss']:.4f}")
                
    def on_save(self, args, state, control, **kwargs):
        self.logger.info(f"Checkpoint saved at step {state.global_step}")
        self.metrics_tracker.save_metrics()


class UniversalFineTuner:
    """Universal fine-tuning class for multiple model families"""
    
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig):
        self.model_config = model_config
        self.training_config = training_config
        
        # Detect model family and get specific configuration
        self.model_specific_config = ModelFamilyDetector.get_model_config(
            self.model_config.model_name
        )
        
        # YAML configuration always takes priority
        self.model_specific_config.lora_r = self.model_config.lora_r
        self.model_specific_config.lora_alpha = self.model_config.lora_alpha
        self.model_specific_config.load_in_4bit = self.model_config.load_in_4bit
        self.model_specific_config.load_in_8bit = self.model_config.load_in_8bit
        self.model_specific_config.use_gradient_checkpointing = self.model_config.use_gradient_checkpointing

        # Override target modules if specified in config
        if self.model_config.lora_target_modules:
            self.model_specific_config.lora_target_modules = self.model_config.lora_target_modules

        # Set up directories
        self._setup_directories()
        
        # Initialize logger
        self.logger_handler = EnhancedLogger(self.training_config.logging_dir)
        self.logger = self.logger_handler.get_logger()
        
        # Log detected model family
        self.logger.info(f"Detected model family: {self.model_specific_config.family.value}")
        
        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker(self.training_config.output_dir)
        
        # Save configurations
        self._save_configs()
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        
    def _setup_directories(self):
        """Create necessary directories"""
        for dir_path in [self.training_config.output_dir, 
                         self.training_config.logging_dir,
                         self.training_config.checkpoint_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
    def _save_configs(self):
        """Save configuration files for reproducibility"""
        config_dir = Path(self.training_config.output_dir) / "configs"
        config_dir.mkdir(exist_ok=True)
        
        # Save all configurations
        configs_to_save = {
            "model_config": asdict(self.model_config),
            "training_config": asdict(self.training_config),
            "model_specific_config": {
                "family": self.model_specific_config.family.value,
                "use_fast_model": self.model_specific_config.use_fast_model,
                "chat_template": self.model_specific_config.chat_template,
                "lora_r": self.model_specific_config.lora_r,
                "lora_alpha": self.model_specific_config.lora_alpha,
            }
        }
        
        with open(config_dir / "all_configs.json", 'w') as f:
            json.dump(configs_to_save, f, indent=2, default=str)
            
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with family-specific configuration"""
        self.logger.info(f"Setting up {self.model_specific_config.family.value} model and tokenizer...")
        
        try:
            # Choose the appropriate loader based on model family
            if self.model_specific_config.use_fast_model:
                # Use FastModel for Gemma
                if FastModel is None:
                    raise ImportError("FastModel not available in Unsloth")
                    
                self.model, self.tokenizer = FastModel.from_pretrained(
                    model_name=self.model_config.model_name,
                    max_seq_length=self.model_config.max_seq_length,
                    load_in_4bit=self.model_specific_config.load_in_4bit,
                    load_in_8bit=self.model_specific_config.load_in_8bit,
                    full_finetuning=self.model_config.full_finetuning,
                )
                
                # Apply Gemma-specific LoRA configuration
                self.model = FastModel.get_peft_model(
                    self.model,
                    finetune_vision_layers=self.model_specific_config.finetune_vision_layers,
                    finetune_language_layers=self.model_specific_config.finetune_language_layers,
                    finetune_attention_modules=self.model_specific_config.finetune_attention_modules,
                    finetune_mlp_modules=self.model_specific_config.finetune_mlp_modules,
                    r=self.model_specific_config.lora_r,
                    lora_alpha=self.model_specific_config.lora_alpha,
                    lora_dropout=self.model_config.lora_dropout,
                    bias=self.model_config.bias,
                    random_state=self.model_config.random_state,
                )
            else:
                # Use FastLanguageModel for other models
                if FastLanguageModel is None:
                    raise ImportError("FastLanguageModel not available in Unsloth")
                    
                # Load model with family-specific settings
                load_kwargs = {
                    "model_name": self.model_config.model_name,
                    "max_seq_length": self.model_config.max_seq_length,
                    "dtype": self.model_config.dtype,
                    "load_in_4bit": self.model_specific_config.load_in_4bit or self.model_config.load_in_4bit,
                    "load_in_8bit": self.model_specific_config.load_in_8bit or self.model_config.load_in_8bit,
                }
                
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
                
                # Apply standard LoRA configuration
                self.model = FastLanguageModel.get_peft_model(
                    self.model,
                    r=self.model_specific_config.lora_r,
                    target_modules=self.model_specific_config.lora_target_modules,
                    lora_alpha=self.model_specific_config.lora_alpha,
                    lora_dropout=self.model_config.lora_dropout,
                    bias=self.model_config.bias,
                    use_gradient_checkpointing=self.model_specific_config.use_gradient_checkpointing,
                    random_state=self.model_config.random_state,
                    use_rslora=self.model_config.use_rslora,
                )
            
            # Apply chat template if specified
            if self.model_specific_config.chat_template:
                self.tokenizer = get_chat_template(
                    self.tokenizer,
                    chat_template=self.model_specific_config.chat_template
                )
                
            self.logger.info("Model and tokenizer setup complete")
            self._log_model_info()
            
        except Exception as e:
            self.logger.error(f"Failed to setup model: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
            
    def _log_model_info(self):
        """Log model information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model family: {self.model_specific_config.family.value}")
        self.logger.info(f"LoRA configuration from YAML:")
        self.logger.info(f"  - rank (r): {self.model_specific_config.lora_r}")
        self.logger.info(f"  - alpha: {self.model_specific_config.lora_alpha}")
        self.logger.info(f"  - dropout: {self.model_config.lora_dropout}")
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
        
        if torch.cuda.is_available():
            gpu_stats = torch.cuda.get_device_properties(0)
            self.logger.info(f"GPU: {gpu_stats.name}")
            self.logger.info(f"GPU Memory: {gpu_stats.total_memory / 1024**3:.2f} GB")

    def _load_jsonl_dataset(self, file_path: str) -> Dataset:
        """
        Load a JSONL dataset efficiently using HuggingFace's native loader.
        This uses Arrow memory-mapping to avoid OOM errors on large datasets.
        """
        self.logger.info(f"Loading JSONL dataset efficiently from: {file_path}")
        
        try:
            # maps the file from disk instead of loading it
            dataset = load_dataset("json", data_files=file_path, split="train")
            self.logger.info(f"Loaded {len(dataset)} examples via memory-mapping")
            return dataset
        except Exception as e:
            self.logger.error(f"Failed to load dataset efficiently: {e}")
            raise

    def _detect_messages_column(self, dataset: Dataset) -> str:
        """
        Detect which column contains the conversation messages.
        """
        column_names = dataset.column_names
        
        # Priority order for column detection
        possible_columns = [
            self.training_config.messages_column,  # User-specified column first
            "messages",
            "conversations", 
            "conversation",
            "chat",
            "dialogue"
        ]
        
        for col in possible_columns:
            if col in column_names:
                self.logger.info(f"Detected messages column: '{col}'")
                return col
        
        raise ValueError(
            f"Could not find messages column. Available columns: {column_names}. "
            f"Expected one of: {possible_columns}"
        )

    @staticmethod
    def _normalize_messages(messages: List[Dict]) -> List[Dict]:
        """
        Normalize message format to standard ChatML.
        Handles variations like "from"/"value" vs "role"/"content".
        """
        normalized = []
        
        role_mapping = {
            "human": "user",
            "gpt": "assistant",
            "bot": "assistant"
        }
        
        for msg in messages:
            # Handle different key formats
            if "role" in msg and "content" in msg:
                role = msg["role"].lower()
                content = msg["content"]
            elif "from" in msg and "value" in msg:
                role = msg["from"].lower()
                content = msg["value"]
            else:
                # Skip malformed messages
                continue
            
            # Normalize role names
            role = role_mapping.get(role, role)
            
            normalized.append({
                "role": role,
                "content": content
            })
        
        return normalized
            
    def prepare_datasets(self) -> Tuple[Dataset, Dataset, List]:
        """Load and prepare datasets with JSONL and ChatML support"""
        self.logger.info("Preparing datasets...")
        
        try:
            dataset_name = self.training_config.dataset_name
            
            # Determine loading method based on dataset path/name
            if dataset_name.endswith('.jsonl'):
                # Direct JSONL file loading
                self.logger.info(f"Loading JSONL dataset from: {dataset_name}")
                dataset = self._load_jsonl_dataset(dataset_name)
                final_dataset = self._create_splits_from_dataset(dataset)
                
            elif dataset_name.startswith("disk:"):
                dataset_path = dataset_name.replace("disk:", "")
                self.logger.info(f"Loading dataset from disk: {dataset_path}")
                
                # Check if it's a JSONL file
                if dataset_path.endswith('.jsonl'):
                    dataset = self._load_jsonl_dataset(dataset_path)
                    final_dataset = self._create_splits_from_dataset(dataset)
                else:
                    # Load from disk (HuggingFace format)
                    dataset = load_from_disk(dataset_path)
                    
                    if hasattr(dataset, 'keys'):
                        if 'train' in dataset:
                            final_dataset = dataset
                        else:
                            available_split = list(dataset.keys())[0]
                            dataset = dataset[available_split]
                            final_dataset = self._create_splits_from_dataset(dataset)
                    else:
                        final_dataset = self._create_splits_from_dataset(dataset)
                    
            elif dataset_name.startswith("/") or dataset_name.startswith("./"):
                # Absolute or relative path
                self.logger.info(f"Loading dataset from path: {dataset_name}")
                
                if dataset_name.endswith('.jsonl'):
                    dataset = self._load_jsonl_dataset(dataset_name)
                    final_dataset = self._create_splits_from_dataset(dataset)
                else:
                    dataset = load_from_disk(dataset_name)
                    if hasattr(dataset, 'keys') and 'train' in dataset:
                        final_dataset = dataset
                    else:
                        final_dataset = self._create_splits_from_dataset(dataset)
                    
            else:
                # Standard HuggingFace Hub loading
                self.logger.info(f"Loading dataset from HuggingFace Hub: {dataset_name}")
                dataset = load_dataset(dataset_name, split="train")
                final_dataset = self._create_splits_from_dataset(dataset)
            
            # Log dataset info
            self.logger.info(f"Dataset splits - Train: {len(final_dataset['train'])}, "
                        f"Val: {len(final_dataset.get('validation', []))}, "
                        f"Test: {len(final_dataset.get('test', []))}")
            
            # Process datasets based on model family
            train_dataset, val_dataset, test_conversations = self._process_datasets_for_family(final_dataset)
            
            return train_dataset, val_dataset, test_conversations
            
        except Exception as e:
            self.logger.error(f"Failed to prepare datasets: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def _create_splits_from_dataset(self, dataset: Dataset) -> DatasetDict:
        """Helper method to create train/val/test splits from a single dataset"""
        total_size = len(dataset)
        test_size = int(total_size * self.training_config.test_split)
        val_size = int(total_size * self.training_config.validation_split)
        
        # First split: separate test set
        train_val_split = dataset.train_test_split(
            test_size=test_size,
            shuffle=True,
            seed=self.training_config.seed
        )
        
        # Second split: separate validation from training
        val_ratio = val_size / (total_size - test_size)
        train_split = train_val_split['train'].train_test_split(
            test_size=val_ratio,
            shuffle=True,
            seed=self.training_config.seed
        )
        
        final_dataset = DatasetDict({
            'train': train_split['train'],
            'validation': train_split['test'],
            'test': train_val_split['test']
        })
        
        return final_dataset
                
    def _process_datasets_for_family(self, dataset_dict: DatasetDict) -> Tuple[Dataset, Dataset, List]:
        """
        Process datasets according to model family requirements.
        Handles ChatML formatted messages directly without unnecessary conversion.
        """
        # Detect the messages column
        messages_col = self._detect_messages_column(dataset_dict["train"])
        
        # Extract test conversations for evaluation
        test_conversations = []
        for example in dataset_dict["test"]:
            messages = example[messages_col]
            test_conversations.append(self._normalize_messages(messages))

        # local variables to avoid high RAM usage
        tokenizer = self.tokenizer 
        remove_bos_prefix = self.model_specific_config.remove_bos_prefix
        bos_token = tokenizer.bos_token if tokenizer.bos_token else None
        
        # Captures the static normalization function locally
        normalize_func = UniversalFineTuner._normalize_messages
        
        # Define formatting function based on format detection
        def formatting_prompts_func(examples):
            """Format messages to text using the tokenizer's chat template"""
            messages_list = examples[messages_col]
            texts = []
            
            for messages in messages_list:
                # Normalize. 
                # This handles ShareGPT, standard ChatML, and mixed formats safely.
                normalized = normalize_func(messages)
                
                # Apply chat template
                text = tokenizer.apply_chat_template(
                    normalized, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                
                # Remove BOS prefix for Gemma or Mistral if needed
                # Get the actual BOS token from the tokenizer (e.g., "<s>", "<bos>")
                if remove_bos_prefix and bos_token:
                    text = text.removeprefix(bos_token)
                
                texts.append(text)
            
            return {"text": texts}
        
        # Apply formatting to train and validation datasets
        train_dataset = dataset_dict["train"].map(
            formatting_prompts_func, 
            batched=True,
            #batch_size=100,         # default 1000
            #writer_batch_size=100,  # Write to disk every 100 items
            remove_columns=dataset_dict["train"].column_names
        )
        
        val_dataset = dataset_dict["validation"].map(
            formatting_prompts_func, 
            batched=True,
            # batch_size=100,         # default 1000
            # writer_batch_size=100,  # Write to disk every 100 items
            remove_columns=dataset_dict["validation"].column_names
        )
        
        # Log sample formatted text
        if len(train_dataset) > 0:
            sample_text = train_dataset[0]["text"]
            self.logger.info(f"Sample formatted text (first 500 chars):\n{sample_text[:500]}...")
        
        return train_dataset, val_dataset, test_conversations

    def _apply_response_masking(self, trainer):
        """
        Applies masking to user inputs based on the model family so the model
        only trains on assistant responses.
        """
        family = self.model_specific_config.family
        self.logger.info(f"Applying response masking for {family.value}...")
        
        try:
            instruction_part = None
            response_part = None
            
            if family == ModelFamily.GEMMA:
                # Gemma 2/3 specific tokens
                instruction_part = "<start_of_turn>user\n"
                response_part = "<start_of_turn>model\n"
            
            elif family == ModelFamily.LLAMA:
                # Llama 3.0/3.1 specific tokens
                instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n"
                response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n"
            
            elif family == ModelFamily.MISTRAL:
                # Standard Mistral/Zephyr format (Mapped via Unsloth's "mistral" template)
                instruction_part = "[INST]"
                response_part = "[/INST]"

            elif family == ModelFamily.PHI:
                # Phi-4 uses standard ChatML with specific separator tokens
                instruction_part = "<|im_start|>user<|im_sep|>"
                response_part = "<|im_start|>assistant<|im_sep|>"
            
            elif family == ModelFamily.QWEN:
                # Qwen uses standard ChatML
                instruction_part = "<|im_start|>user\n"
                response_part = "<|im_start|>assistant\n"
            
            elif family == ModelFamily.GLM:
                # GLM-4 style
                instruction_part = "<|user|>\n" 
                response_part = "<|assistant|>\n"

            elif family == ModelFamily.EUROLLM:
                # EuroLLM chat format
                instruction_part = "<|im_start|>user\n"
                response_part = "<|im_start|>assistant\n"

            elif family == ModelFamily.OLMO:
                # OLMo-3 uses chatml template without new line
                instruction_part = "<|im_start|>user"
                response_part = "<|im_start|>assistant"
            
            if instruction_part and response_part:
                trainer = train_on_responses_only(
                    trainer,
                    instruction_part=instruction_part,
                    response_part=response_part,
                )
                self.logger.info(f"Successfully applied masking: '{instruction_part.strip()}' -> '{response_part.strip()}'")
            else:
                self.logger.warning(f"Response masking skipped: No default delimiters defined for family {family.value}")

        except Exception as e:
            self.logger.error(f"Failed to apply response masking: {e}")
            self.logger.warning("Continuing training without masking (full sequence training)...")
            
        return trainer
            
    def train(self, train_dataset: Dataset, val_dataset: Dataset):
        """Execute training with model-specific configuration"""
        self.logger.info(f"Starting training for {self.model_specific_config.family.value} model...")
        
        try:
            # Setup wandb if requested
            if self.training_config.use_wandb:
                import wandb
                wandb.init(
                    project=self.training_config.wandb_project,
                    name=self.training_config.wandb_run_name or f"{self.model_specific_config.family.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    config={
                        "model_config": asdict(self.model_config),
                        "training_config": asdict(self.training_config),
                        "model_family": self.model_specific_config.family.value
                    }
                )
                report_to = "wandb"
            else:
                report_to = "none"
                
            # Prepare training arguments
            training_args = SFTConfig(
                output_dir=self.training_config.checkpoint_dir,
                dataset_text_field="text",
                
                # Training parameters
                per_device_train_batch_size=self.training_config.per_device_train_batch_size,
                per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
                gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
                warmup_steps=self.training_config.warmup_steps,
                warmup_ratio=self.training_config.warmup_ratio,
                num_train_epochs=self.training_config.num_train_epochs,
                max_steps=self.training_config.max_steps,
                max_grad_norm=self.training_config.max_grad_norm,
                
                # Optimizer
                learning_rate=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay,
                optim=self.training_config.optim,
                lr_scheduler_type=self.training_config.lr_scheduler_type,
                
                # Logging and evaluation
                logging_dir=self.training_config.logging_dir,
                logging_steps=self.training_config.logging_steps,
                eval_strategy=self.training_config.eval_strategy,
                eval_steps=self.training_config.eval_steps,
                
                # Saving
                save_strategy="steps",
                save_steps=self.training_config.save_steps,
                save_total_limit=self.training_config.save_total_limit,
                
                # Best model
                load_best_model_at_end=self.training_config.load_best_model_at_end,
                metric_for_best_model=self.training_config.metric_for_best_model,
                greater_is_better=self.training_config.greater_is_better,
                
                # Precision
                fp16=self.training_config.fp16,
                bf16=self.training_config.bf16,
                
                # Other
                seed=self.training_config.seed,
                dataloader_num_workers=self.training_config.dataloader_num_workers,
                remove_unused_columns=self.training_config.remove_unused_columns,
                report_to=report_to,
                
                # Resume from checkpoint
                resume_from_checkpoint=self.training_config.resume_from_checkpoint,
            )
            
            # Initialize trainer
            trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                args=training_args,
                callbacks=[CustomCallback(self.logger, self.metrics_tracker)]
            )

            # Apply Response Masking
            trainer = self._apply_response_masking(trainer)

            self.logger.info("Checking if response masking is applied correctly on a sample:")
            self.logger.info("Without masking (input_ids):")
            self.logger.info(self.tokenizer.decode(trainer.train_dataset[100]["input_ids"]))
            self.logger.info("With masking (labels):")
            self.logger.info(self.tokenizer.decode([self.tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]).replace(self.tokenizer.pad_token, " "))
            
            # check masking start
            def verify_masking(trainer, tokenizer, num_samples=3):
                print(f"\n{'='*20} MASKING VERIFICATION {'='*20}")
                dataset = trainer.train_dataset
                
                for i in range(num_samples):
                    sample = dataset[i]
                    input_ids = sample["input_ids"]
                    labels = sample["labels"]
                    
                    # Decode the learned parts only
                    # We replace ignored tokens (-100) with a placeholder to visualize them
                    visualized_tokens = []
                    for token_id, label in zip(input_ids, labels):
                        if label == -100:
                            visualized_tokens.append("[M]")
                        else:
                            visualized_tokens.append(tokenizer.decode([token_id]))
                            
                    full_text = "".join(visualized_tokens)
                    
                    print(f"\n--- SAMPLE {i+1} ---")
                    print("Expected: User instruction should be [M]. Assistant response should be readable.")
                    print(f"RESULT:\n{full_text[:500]}...") # Show first 500 chars
                    print("-" * 50)

            
            verify_masking(trainer, self.tokenizer)
            # check masking end
            
            # Log initial GPU memory
            if torch.cuda.is_available():
                self.logger.info(f"Initial GPU memory: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
                
            # Train
            train_result = trainer.train(
                resume_from_checkpoint=self.training_config.resume_from_checkpoint
            )
            
            # Log final metrics
            self.logger.info(f"Training completed in {train_result.metrics['train_runtime']:.2f} seconds")
            self.logger.info(f"Final loss: {train_result.metrics['train_loss']:.4f}")
            
            if torch.cuda.is_available():
                self.logger.info(f"Peak GPU memory: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
                
            # Save final model
            model_family_name = self.model_specific_config.family.value
            final_model_path = Path(self.training_config.output_dir) / f"final_model_{model_family_name}"
            self.logger.info(f"Saving final model to {final_model_path}")
            trainer.save_model(str(final_model_path))
            self.tokenizer.save_pretrained(str(final_model_path))
            
            return trainer
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
                
    def evaluate_model(self, test_conversations: List, num_samples: int = 10):
        """Comprehensive evaluation on test set"""
        self.logger.info(f"Evaluating {self.model_specific_config.family.value} model on {num_samples} test samples...")
        
        try:
            # Initialize metrics
            rouge = evaluate.load('rouge')
            predictions = []
            references = []
            
            # Prepare model for inference if needed
            if self.model_specific_config.family in [ModelFamily.GLM, ModelFamily.LLAMA]:
                if hasattr(FastLanguageModel, 'for_inference'):
                    FastLanguageModel.for_inference(self.model)
            
            # Generate predictions
            for i, example in enumerate(tqdm(test_conversations[:num_samples], desc="Evaluating")):
                if len(example) < 2:
                    continue
                    
                prompt = example[0]['content']
                reference = example[1]['content']
                
                # Apply chat template for the prompt
                if self.model_specific_config.chat_template:
                    messages = [{"role": "user", "content": prompt}]
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    formatted_prompt = prompt

                # Generate response
                # Multimodal models use a Processor (Text+Image), so we must explicitly use text=...
                # otherwise the prompt is interpreted as an image.
                inputs = self.tokenizer(text=formatted_prompt, return_tensors="pt", truncation=True).to(self.model.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    
                # Decode and clean prediction
                full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                prediction = full_output[len(formatted_prompt):].strip()
                
                predictions.append(prediction)
                references.append(reference)
                
                # Log sample predictions
                if i < 3:
                    self.logger.info(f"\nSample {i+1}:")
                    self.logger.info(f"Prompt: {prompt[:100]}...")
                    self.logger.info(f"Prediction: {prediction[:100]}...")
                    self.logger.info(f"Reference: {reference[:100]}...")
                    
            # Calculate metrics
            rouge_scores = rouge.compute(predictions=predictions, references=references)
            
            # Log results
            eval_results = {
                "model_family": self.model_specific_config.family.value,
                "rouge1": rouge_scores['rouge1'],
                "rouge2": rouge_scores['rouge2'],
                "rougeL": rouge_scores['rougeL'],
                "num_samples": num_samples
            }
            
            self.logger.info("\n=== Evaluation Results ===")
            for metric, value in eval_results.items():
                if metric not in ["num_samples", "model_family"]:
                    self.logger.info(f"{metric}: {value:.4f}")
                    
            # Save evaluation results
            eval_path = Path(self.training_config.output_dir) / "evaluation_results.json"
            with open(eval_path, 'w') as f:
                json.dump(eval_results, f, indent=2)
                
            return eval_results
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
                
    def run_inference_examples(self):
        """Run example inferences with model-specific generation parameters"""
        self.logger.info(f"\n=== Running Inference Examples for {self.model_specific_config.family.value} ===")
        
        test_prompts = [
            "Continue the sequence: 1, 1, 2, 3, 5, 8,",
            "Explain quantum computing in simple terms.",
            "Write a Python function to calculate fibonacci numbers.",
        ]
        
        # Set model-specific generation parameters
        gen_params = {
            ModelFamily.LLAMA: {"temperature": 1.5, "min_p": 0.1},
            ModelFamily.GLM: {"temperature": 0.8, "top_p": 0.8, "do_sample": True},
            ModelFamily.GEMMA: {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
        }.get(self.model_specific_config.family, {"temperature": 0.7, "top_p": 0.9})
        
        for prompt in test_prompts:
            self.logger.info(f"\nPrompt: {prompt}")
            
            # Format prompt with chat template if available
            if self.model_specific_config.chat_template:
                messages = [{"role": "user", "content": prompt}]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                text = prompt

            # Multimodal models use a Processor (Text+Image), so we must explicitly use text=...
            # otherwise the prompt is interpreted as an image.
            inputs = self.tokenizer(text=text, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **gen_params
                )
                
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(text):].strip()
            
            self.logger.info(f"Response: {response[:200]}...")
                
    def run_full_pipeline(self):
        """Execute the complete training pipeline"""
        self.logger.info("="*50)
        self.logger.info(f"Starting Fine-tuning Pipeline for {self.model_specific_config.family.value}")
        self.logger.info("="*50)
        
        try:
            # Setup
            self.setup_model_and_tokenizer()
            
            # Prepare data
            train_dataset, val_dataset, test_conversations = self.prepare_datasets()
            
            # Train
            trainer = self.train(train_dataset, val_dataset)
            
            # Evaluate
            eval_results = self.evaluate_model(test_conversations)
            
            # Run inference examples
            self.run_inference_examples()
            
            self.logger.info("\n" + "="*50)
            self.logger.info("Pipeline completed successfully!")
            self.logger.info("="*50)
            
            return {
                "trainer": trainer,
                "eval_results": eval_results,
                "model_family": self.model_specific_config.family.value
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file"""
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(path, 'r') as f:
        if path.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif path.suffix == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description="Universal Fine-tuning Pipeline with Unsloth")
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (YAML or JSON)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Override model name from config"
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for model and logs"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override maximum training steps"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation on existing model"
    )
    
    args = parser.parse_args()
    
    # Load configurations
    if args.config:
        config_dict = load_config_from_file(args.config)
        model_config = ModelConfig(**config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
    else:
        model_config = ModelConfig()
        training_config = TrainingConfig()
        
    # Override with command line arguments
    if args.model:
        model_config.model_name = args.model
    if args.resume:
        training_config.resume_from_checkpoint = args.resume
    if args.output_dir:
        training_config.output_dir = args.output_dir
    if args.wandb:
        training_config.use_wandb = True
    if args.max_steps is not None:
        training_config.max_steps = args.max_steps
        
    # Initialize trainer
    fine_tuner = UniversalFineTuner(model_config, training_config)
    
    # Run pipeline
    if args.eval_only:
        fine_tuner.setup_model_and_tokenizer()
        _, _, test_conversations = fine_tuner.prepare_datasets()
        fine_tuner.evaluate_model(test_conversations)
        fine_tuner.run_inference_examples()
    else:
        fine_tuner.run_full_pipeline()


if __name__ == "__main__":
    main()
