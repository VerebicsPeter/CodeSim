import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import (
    AutoModel,
    PreTrainedModel,
)
from peft import LoraConfig, get_peft_model

from model.code_sim_models import PoolingStrategy, cls_pooling_strat

FINETUNING_STRATEGIES = [
    "binary_cls_simpl",
    "binary_cls_sbert",
]

CONTRASTIVE_FINETUNING_STRATEGIES = [
    "triplet_loss",
    "info_nce_loss",
    "combined_loss",
]


@dataclass
class BaseConfig:
    # model related
    pretrained_model_name: str = "huggingface/CodeBERTa-small-v1"
    pretrained_model: PreTrainedModel | None = None
    freeze_model: bool = False  # NOTE: if true the encoder model is not finetuned
    pooling_strat: PoolingStrategy = cls_pooling_strat
    # LoRA config
    lora_config: LoraConfig | None = None
    # workers
    num_workers: int = 0
    # num rows to cap dataloaders
    num_rows: int | None = None
    # epochs and batch size
    epochs: int = 4
    bs: int = 20  # batch size
    iters_to_accumulate: int = 2
    
    def patched_init_model(self):
        print(f"Pretrained checkpoint name: {self.pretrained_model_name}")
        
        if self.pretrained_model is None:
            print("Initializing encoder model.")
            self.pretrained_model = AutoModel.from_pretrained(self.pretrained_model_name)
        
        device = self.pretrained_model.device
        
        if self.lora_config is not None:
            print("Wrapping encoder model with LoRA config for parameter efficient finetuning.")
            self.pretrained_model = get_peft_model(self.pretrained_model, self.lora_config)
        
        if torch.cuda.device_count() > 1:
            print("Wrapping encoder model with DataParallel for parameter multiple GPU usage.")
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
            self.pretrained_model = nn.DataParallel(self.pretrained_model)
        
        self.pretrained_model.device = device

@dataclass
class CodeSimClassifierConfig(BaseConfig):
    finetuning_strategy: str = FINETUNING_STRATEGIES[0]
    # Learning rates and weight decays
    lr: float = 1e-5
    wd: float = 1e-5
    # Model specific parameters
    dropout_rate: float = 0.2
    shuffle_dataloader: bool = True


@dataclass
class CodeSimContrastiveClassifierConfig(BaseConfig):
    finetuning_strategy: str = CONTRASTIVE_FINETUNING_STRATEGIES[0]
    num_batches: int = 200  # Number of random batches to sample
    # Learning rates and weight decays
    lr_enc: float = 1e-5  # Encoder learning rate
    wd_enc: float = 1e-5  # Encoder weight decay
    # Model specific parameters
    dropout_rate: float = 0.2
    # Loss function hyperparameters
    margin: float = 1.0
    temp: float = 0.05      # InfoNCE-inspired loss temperature
    num_negatives: int = 1  # InfoNCE-inspired loss hard negatives
    w_1: float = 1.0  # Weight for InfoNCE Loss
    w_2: float = 1.0  # Weight for Triplet Loss
