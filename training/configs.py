"""Configuration classes for model fine-tuning experiments"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

class ModelFamily(Enum):
    """Enumeration of supported model families"""
    QWEN = "qwen"
    GEMMA = "gemma"
    GLM = "glm"
    LLAMA = "llama"
    MISTRAL = "mistral"
    PHI = "phi"
    EUROLLM = "eurollm"
    OLMO = "olmo"
    UNKNOWN = "unknown"

@dataclass
class ModelSpecificConfig:
    """Model family specific configurations"""
    family: ModelFamily
    use_fast_model: bool = False
    chat_template: Optional[str] = None
    lora_r: int = 16
    lora_alpha: int = 16
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    use_gradient_checkpointing: str = "unsloth"
    remove_bos_prefix: bool = False
    # Gemma-specific parameters
    finetune_vision_layers: Optional[bool] = None
    finetune_language_layers: Optional[bool] = None
    finetune_attention_modules: Optional[bool] = None
    finetune_mlp_modules: Optional[bool] = None

@dataclass
class ModelConfig:
    """Configuration for model setup"""
    model_name: str = "unsloth/Qwen3-0.6B"
    max_seq_length: int = 2048
    dtype: Optional[str] = None
    full_finetuning: bool = False
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 3407
    use_rslora: bool = False
    load_in_4bit: bool = False
    load_in_8bit: bool = False

@dataclass
class TrainingConfig:
    """Configuration for training"""
    dataset_name: str = "mlabonne/FineTome-100k"
    output_dir: str = "./output"
    logging_dir: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    messages_column: str = "messages"
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    warmup_ratio: Optional[float] = None
    num_train_epochs: int = 1
    max_steps: int = -1
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    save_total_limit: int = 3
    eval_strategy: str = "steps"
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    load_best_model_at_end: bool = True
    optim: str = "adamw_8bit"
    lr_scheduler_type: str = "linear"
    fp16: bool = False
    bf16: bool = False
    validation_split: float = 0.05
    test_split: float = 0.05
    seed: int = 3407
    dataloader_num_workers: int = 4  
    remove_unused_columns: bool = True
    resume_from_checkpoint: Optional[str] = None
    use_wandb: bool = False
    wandb_project: str = "unsloth-finetuning"
    wandb_run_name: Optional[str] = None

    def __post_init__(self):
        if self.logging_dir is None:
            self.logging_dir = os.path.join(self.output_dir, "logs")
        if self.checkpoint_dir is None:
            self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
