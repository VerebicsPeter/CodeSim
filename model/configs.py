from transformers import (
    AutoModel,
    PreTrainedModel,
)
from dataclasses import dataclass, asdict

# TODO: add pooling strat to config
FINETUNING_STRATEGIES = {"binary_cls_simpl", "binary_cls_sbert"}


@dataclass
class BaseConfig:
    pretrained_bert_name: str = "huggingface/CodeBERTa-small-v1"
    pretrained_bert: PreTrainedModel | None = None
    freeze_bert: bool = False  # NOTE: if true the BERT model is not finetuned
    epochs: int = 4
    num_workers: int = 0
    num_rows: int | None = None
    bs: int = 20  # batch size
    iters_to_accumulate: int = 2
    
    def init_model(self):
        print(f"Pretrained checkpoint name: {self.pretrained_bert_name}")
        
        if self.pretrained_bert is None:
            print("Initializing bert model.")
            self.pretrained_bert = AutoModel.from_pretrained(self.pretrained_bert_name)


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
    num_batches: int = 200
