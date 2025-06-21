from dataclasses import dataclass
from transformers import (
    AutoModel,
    PreTrainedModel,
)
from peft import LoraConfig, get_peft_model

from model.code_sim_models import PoolingStrategy, cls_pooling_strat

FINETUNING_STRATEGIES = {"binary_cls_simpl", "binary_cls_sbert"}


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
    
    def init_model(self):
        print(f"Pretrained checkpoint name: {self.pretrained_model_name}")
        
        if self.pretrained_model is None:
            print("Initializing encoder model.")
            self.pretrained_model = AutoModel.from_pretrained(self.pretrained_model_name)
        
        if self.lora_config is not None:
            print("Wrapping model with lora config for parameter efficient finetuning.")
            self.pretrained_model = get_peft_model(self.pretrained_model, self.lora_config)


@dataclass
class CodeSimClassifierConfig(BaseConfig):
    finetuning_strategy: str = "binary_cls_simpl"
    # Learning rates and weight decays
    lr: float = 1e-5
    wd: float = 1e-5
    # Model specific parameters
    dropout_rate: float = 0.2
    shuffle_dataloader: bool = True


@dataclass
class CodeSimContrastiveClassifierConfig(BaseConfig):
    # Learning rates and weight decays
    lr_enc: float = 1e-5  # Encoder learning rate
    wd_enc: float = 1e-5  # Encoder weight decay
    # Model specific parameters
    dropout_rate: float = 0.2
    # Loss function hyperparameters
    margin: float = 1.0
    use_info_nce_inspired_loss: bool = False
    temp: float = 0.05      # InfoNCE-inspired loss temperature
    num_negatives: int = 1  # InfoNCE-inspired loss hard negatives
    num_batches: int = 200  # Number of random batches to sample

