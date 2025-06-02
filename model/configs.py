from dataclasses import dataclass

@dataclass
class BasicCodeSimClassifierConfig:
    finetuning_strategy="linear_binary_cls"
    pretrained_bert_name="huggingface/CodeBERTa-small-v1"
    epochs=4
    lr=1e-5
    wd=1e-5
    bs=20
    iters_to_accumulate=2
    freeze_bert=False
    dropout_rate=0.2
    shuffle_dataloader=True
    num_rows=5000    

@dataclass
class TripletCodeSimClassifierConfig:
    pretrained_bert_name="huggingface/CodeBERTa-small-v1"
    epochs= 4
    # Learning rates and weight decays
    lr_enc=1e-5  # Encoder learning rate
    wd_enc=1e-5  # Encoder weight decay
    lr_cls=1e-3  # Classifier learning rate  
    wd_cls=1e-3  # Classifier weight decay
    # Batch size
    bs=20
    iters_to_accumulate=2
    # Model specific parameters
    freeze_bert=False  # NOTE: if true the BERT model is not finetuned
    dropout_rate=0.2
    shuffle_dataloader=True
    num_rows=5000
    # Loss function hyperparameters
    margin=1.0
    use_info_nce_inspired_loss=False

@dataclass
class CombinedCodeSimClassifierConfig:
    pretrained_bert_name= "huggingface/CodeBERTa-smal-v1"
    epochs=4
    # Learning rates and weight decays
    lr_bert=1e-5
    wd_bert=1e-5
    lr_proj=1e-4
    wd_proj=1e-4
    # Batch size
    bs=20
    iters_to_accumulate=2
    # Model specific parameters
    freeze_bert=False  #NOTE: if true the BERT model is not finetuned
    dropout_rate=0.2
    shuffle_dataloader=True
    num_rows=5000
    # Loss function hyperparameters
    margin=1.0  # triplet loss function hyperparameter
    w_emb=1.0  # loss component weight
    w_cls=1.0  # loss component weight


def __filter_dict(d: dict[str,]):  # filter privates
    return {k:v for k,v in d.items() if not k.startswith("__")}

TRAIN_ARGS = {
    "basic"   :__filter_dict(BasicCodeSimClassifierConfig   .__dict__),
    "triplet" :__filter_dict(TripletCodeSimClassifierConfig .__dict__),
    "combined":__filter_dict(CombinedCodeSimClassifierConfig.__dict__),
}
