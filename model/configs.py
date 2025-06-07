from transformers import (
    AutoModel,
    PreTrainedModel,
)
from dataclasses import dataclass, asdict


FINETUNING_STRATEGIES = {"binary_cls_simpl", "binary_cls_sbert"}


@dataclass
class BaseConfig:
    pretrained_bert_name: str = "huggingface/CodeBERTa-small-v1"
    pretrained_bert: PreTrainedModel | None = None
    epochs: int = 4
    num_workers: int = 0
    num_rows: int | None = None
    bs: int = 20  # batch size
    iters_to_accumulate: int = 2
    freeze_bert: bool = False  # NOTE: if true the BERT model is not finetuned
    
    def post_init(self):
        """Initialize tokenizer and model after creating config."""
        
        print(f"Pretrained checkpoint name: {self.pretrained_bert_name}")
        
        if self.pretrained_bert is None:
            print("Initializing bert model.")
            self.pretrained_bert = AutoModel.from_pretrained(self.pretrained_bert_name)
            print("Initialized  bert model.")


@dataclass
class BasicCodeSimClassifierConfig(BaseConfig):
    finetuning_strategy: str = "binary_cls_simpl"
    # Learning rates and weight decays
    lr: float = 1e-5
    wd: float = 1e-5
    # Model specific parameters
    dropout_rate: float = 0.2
    shuffle_dataloader: bool = True


@dataclass
class TripletCodeSimClassifierConfig(BaseConfig):
    # Learning rates and weight decays
    lr_enc: float = 1e-5  # Encoder learning rate
    wd_enc: float = 1e-5  # Encoder weight decay
    # Model specific parameters
    dropout_rate: float = 0.2
    shuffle_dataloader: bool = True
    # Loss function hyperparameters
    margin: float = 1.0
    use_info_nce_inspired_loss: bool = False


@dataclass
class CombinedCodeSimClassifierConfig(BaseConfig):
    # Learning rates and weight decays
    lr_bert: float = 1e-5
    wd_bert: float = 1e-5
    lr_proj: float = 1e-4
    wd_proj: float = 1e-4
    # Model specific parameters
    dropout_rate: float = 0.2
    shuffle_dataloader: bool = True
    # Loss function hyperparameters
    margin: float = 1.0  # triplet loss function hyperparameter
    w_emb: float = 1.0  # loss component weight
    w_cls: float = 1.0  # loss component weight


TRAIN_ARGS = {
    "basic":    asdict(BasicCodeSimClassifierConfig()),
    "triplet":  asdict(TripletCodeSimClassifierConfig()),
    "combined": asdict(CombinedCodeSimClassifierConfig()),
}
