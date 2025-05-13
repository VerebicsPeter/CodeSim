import model.code_sim_models

TRAIN_ARGS = {
    "basic": {
        "finetuning_strategy": "linear_binary_cls",
        "pretrained_bert_name": "huggingface/CodeBERTa-small-v1",
        "epochs": 4,
        "lr": 1e-5,  # Learning rate
        "wd": 1e-5,  # Weight decay
        # Batch size
        "bs": 20,
        # The gradient accumulation adds gradients over an effective batch of size : bs * iters_to_accumulate.
        # If set to "1", you get the usual batch size
        "iters_to_accumulate": 2,
        # Model specific parameters
        "freeze_bert": False,  # NOTE: if true the BERT model is not finetuned
        "dropout_rate": 0.2,
        "shuffle_dataloader": True,
        "num_rows": 5000,
    },
    "triplet": {
        "pretrained_bert_name": "huggingface/CodeBERTa-small-v1",
        "epochs": 4,
        # Learning rates and weight decays
        "lr_enc": 1e-5,  # Encoder learning rate
        "wd_enc": 1e-5,  # Encoder weight decay
        "lr_cls": 1e-3,  # Classifier learning rate  
        "wd_cls": 1e-5,  # Classifier weight decay
        # Batch size
        "bs": 20,
        "iters_to_accumulate": 2,
        # Model specific parameters
        "freeze_bert": False,  # NOTE: if true the BERT model is not finetuned
        "dropout_rate": 0.2,
        "shuffle_dataloader": True,
        "num_rows": 5000,
        "margin": 1.0,  # Triplet loss function hyperparameter
        "use_info_nce_inspired_loss": False,
    },
    "combined": {
        "pretrained_bert_name": "huggingface/CodeBERTa-small-v1",
        "epochs": 4,
        # Learning rates and weight decays
        "lr_bert": 1e-5,
        "wd_bert": 1e-5,
        "lr_proj": 1e-4,
        "wd_proj": 1e-5,
        # Batch size
        "bs": 20,
        "iters_to_accumulate": 2,
        # Model specific parameters
        "freeze_bert": False,  # NOTE: if true the BERT model is not finetuned
        "dropout_rate": 0.2,
        "shuffle_dataloader": True,
        "num_rows": 5000,
        "margin": 1.0,  # Triplet loss function hyperparameter
        "w_emb": 1.0,  # loss component weight
        "w_cls": 1.0,  # loss component weight
    },
}
