import argparse
import random
import numpy as np
import pandas as pd
import pprint as pp
import python_minifier

import torch
import torch.nn as nn
from torch.utils.data import(
    Subset,
    DataLoader,
    random_split,
)

from pytorch_metric_learning import losses

from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)

import model.code_sim_models as code_sim_models
import model.code_sim_datasets as code_sim_datasets

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: Maybe load URLS from a .env or something

paired_dataset_url  = "https://drive.google.com/uc?export=download&id=1pUErbyZw1fBC5gIe6KT7BWga7h6Bfr4l"  # 100k dataset
triplet_dataset_url = "https://drive.google.com/uc?export=download&id=11aBIxIMEMKoGyJ9IdUHY2XQv1ZzfyXd2"  # 100k dataset

contrastive_dataset_url_labeled   = "https://drive.google.com/uc?export=download&id=1UteITBYXcBLt2hXviy71jQr-oXceVcs5"
contrastive_dataset_url_unlabeled = "https://drive.google.com/uc?export=download&id=1iHHgOcJQ_qp3sk3d7w1zpWBvsgDqrPJV"


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def test_forward_passes(pretrained_bert_name: str = "huggingface/CodeBERTa-small-v1"):
    code = """print("Hello, World!")"""
    
    bert_tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_name)
    bert = AutoModel.from_pretrained(pretrained_bert_name).to(DEVICE)
    
    inputs = bert_tokenizer(code, return_tensors='pt', truncation=True, padding=True)
    model1 = code_sim_models.FinetunedCodeSimilarityModel(bert).to(DEVICE)
    model2 = code_sim_models.FinetunedCodeSimilaritySBERT(bert).to(DEVICE)
    model3 = code_sim_models.ContrastiveCodeSimilarityModel(bert).to(DEVICE)

    emb1 = model1(inputs)
    print("Model 1 output shape:", emb1.shape)
    
    emb2 = model2(inputs)
    print("Model 2 output shape:", emb2.shape)
    
    emb3 = model3(inputs)
    print("Model 3 output shape:", emb3.shape)


def test_predict_passes(pretrained_bert_name: str = "huggingface/CodeBERTa-small-v1"):
    code_a = """print("Hello, World!")"""
    code_b = """def add(x,y): return x+y"""
    
    bert = AutoModel.from_pretrained(pretrained_bert_name).to(DEVICE)
    model1 = code_sim_models.FinetunedCodeSimilarityModel(bert).to(DEVICE)
    model2 = code_sim_models.FinetunedCodeSimilaritySBERT(bert).to(DEVICE)
    model3 = code_sim_models.ContrastiveCodeSimilarityModel(bert).to(DEVICE)

    pred1 = model1.predict(code_a, code_b)
    print("Model 1 prediction output:", pred1, "shaped", pred1.shape)
    
    pred2 = model2.predict(code_a, code_b)
    print("Model 2 prediction output:", pred2, "shaped", pred2.shape)
    
    pred3 = model3.predict(code_a, code_b)
    print("Model 3 prediction output:", pred3, "shaped", pred3.shape)


def create_codenet_dataset(data_path: str, data_type: str, tokenizer, num_rows=50_000):
    if data_type not in {"paired", "triplet"}:
        raise ValueError("Invalid dataset type.")
    if data_type == "paired":
        cls = code_sim_datasets.CodeNetPairDataset
    if data_type == "triplet":
        cls = code_sim_datasets.CodeNetTripletDataset
    
    print(data_path)
    df = pd.read_csv(
        data_path,
        header=0,
        names=cls.COLUMNS
    )
    
    if cls is code_sim_datasets.CodeNetPairDataset:
        print("CodeNet data loaded. Data type: paired")
        print(df['label'].value_counts())
    else:
        print("CodeNet data loaded. Data type: triplet")
    
    pp.pp(df.head())
    
    dataset = cls.from_pandas_df(
        df,
        tokenizer=tokenizer,
        num_rows=num_rows
    )
    return dataset


def train_finetuned(
    data_type="paired",
    pretrained_bert_name: str = "huggingface/CodeBERTa-small-v1",
    epochs = 4,
    lr = 1e-5,  # Learning rate
    wd = 1e-5,  # Weight decay
    bs = 20,    # Batch size
    # The gradient accumulation adds gradients over an effective batch of size : bs * iters_to_accumulate.
    # If set to "1", you get the usual batch size
    iters_to_accumulate = 2,
    # Model specific parameters
    freeze_bert = False,  # NOTE: if true the BERT model is not finetuned
    dropout_rate = 0.2,
    shuffle_dataloader = True,
    num_rows=5000,
):
    if data_type not in {"paired", "triplet"}:
        raise ValueError("Invalid finetuning method.")
    if data_type == "paired":
        dataset_url = paired_dataset_url
    if data_type == "triplet":
        dataset_url = triplet_dataset_url
    
    bert_tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_name)
    
    dataset = create_codenet_dataset(dataset_url, data_type=data_type, num_rows=num_rows, tokenizer=bert_tokenizer)
    
    # TODO: Don't hardcode the train split ratio!
    train_len = int(0.8 * len(dataset))
    valid_len = len(dataset) - train_len
    train_data, valid_data = random_split(dataset, [train_len, valid_len])
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=shuffle_dataloader)
    valid_loader = DataLoader(valid_data, batch_size=bs, shuffle=shuffle_dataloader)
    
    bert_model = AutoModel.from_pretrained(pretrained_bert_name).to(DEVICE)
    
    if data_type == "paired":
        model_cls = code_sim_models.FinetunedCodeSimilarityModel
    if data_type == "triplet":
        model_cls = code_sim_models.FinetunedCodeSimilaritySBERT
    
    model = model_cls(
        bert_model,
        freeze_bert=freeze_bert,
        dropout_rate=dropout_rate,
    )
    model.to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    
   # Training and warmup steps
    # NOTE: Necessary to take into account Gradient accumulation
    num_training_steps = (epochs * len(train_loader)) // iters_to_accumulate
    num_warmup_steps = int(num_training_steps * 0.05)  # 5% warmup
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        # Necessary to take into account Gradient accumulation
        num_training_steps=num_training_steps,
        # The number of steps for the warmup phase.
        num_warmup_steps=num_warmup_steps,
    )
    
    if data_type == "paired":
        loss_func = nn.BCEWithLogitsLoss()
        loss_hook = code_sim_models.compute_loss_finetuned_logit
    if data_type == "triplet":
        # TODO: parametrize hyperparameters
        loss_func = nn.TripletMarginLoss(margin=1.0)
        loss_hook = code_sim_models.compute_loss_finetuned_SBERT
    
    trainer = code_sim_models.CodeSimilarityTrainer(
        model,
        (train_loader, valid_loader),
        loss_func=loss_func,
        # loss strategy
        loss_hook=loss_hook,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
    )
    trainer.train(epochs=epochs, iters_to_accumulate=iters_to_accumulate)


def train_contrastive(
    pretrained_bert_name: str = "huggingface/CodeBERTa-small-v1",
    epochs = 4,
    # Learning rates and weight decays
    lr_bert = 1e-5,
    wd_bert = 1e-4,  # bert often needs smaller weight decay
    lr_proj = 1e-4,
    wd_proj = 1e-3,
    bs = 20,  # NOTE: Bigger batch size generally leads to better results in contrastive learning
    iters_to_accumulate = 2,
    # Model specific parameters
    freeze_bert = False,  # NOTE: if true the BERT model is not finetuned
    dropout_rate = 0.2,
    shuffle_dataloader = True,
    # Flag for self supervised training
    is_self_supervised = False,
    # Temperature hyperparameter for NTXent loss
    temperature=0.5,
):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_name)
    
    if is_self_supervised:
        url = contrastive_dataset_url_unlabeled
        cls = code_sim_datasets.UnlabeledCodeDataset
    else:
        url = contrastive_dataset_url_labeled
        cls = code_sim_datasets.LabeledCodeDataset
    
    dataset = cls.from_csv_data(
        path=url,
        tokenizer=tokenizer,
        aug_funcs=[],  # NOTE: precalculated data augmentation functions can be added here
        device=DEVICE,
    )
    print("Initial dataset size:", len(dataset))
    
    # TODO: don't hardcode this number V
    sample_size = min(len(dataset), 50_000)
    dataset = Subset(dataset, list(range(sample_size)))
    
    print("Sampled dataset size:", len(dataset))
    
    train_len = int(0.8 * len(dataset))
    valid_len = len(dataset) - train_len
    train_data, valid_data = random_split(dataset, [train_len, valid_len])
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=shuffle_dataloader)
    valid_loader = DataLoader(valid_data, batch_size=bs, shuffle=shuffle_dataloader)
    
    if shuffle_dataloader: print("Dataloaders will be shuffled...")
    
    bert_model = AutoModel.from_pretrained(pretrained_bert_name).to(DEVICE)
    
    model = code_sim_models.ContrastiveCodeSimilarityModel(
        bert_model,
        freeze_bert=freeze_bert,
        dropout_rate=dropout_rate,
    )
    model.to(DEVICE)
    
    # TODO: Maybe don't pass frozen params, IDK...
    # NOTE: Allow different lr and wd for BERT and projection head params
    param_groups = [
        {"params": model.bert.parameters(), "lr": lr_bert, "weight_decay": wd_bert},
        {"params": model.proj.parameters(), "lr": lr_proj, "weight_decay": wd_proj},
    ]
    optimizer = torch.optim.AdamW(param_groups)
    
    # Training and warmup steps
    # NOTE: Necessary to take into account Gradient accumulation
    num_training_steps = (epochs * len(train_loader)) // iters_to_accumulate
    num_warmup_steps = int(num_training_steps * 0.05)  # 5% warmup
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        # The number of steps for the warmup phase.
        num_warmup_steps=num_warmup_steps,
    )
    
    ntxent_loss = losses.NTXentLoss(temperature=temperature)
    # Wrap the NTXent loss function if needed
    ntxent_loss = losses.SelfSupervisedLoss(ntxent_loss) if is_self_supervised else ntxent_loss
    
    trainer = code_sim_models.CodeSimilarityTrainer(
        model,
        (train_loader, valid_loader),
        loss_func=ntxent_loss,
        loss_hook=code_sim_models.compute_loss_contrastive,  # loss strategy
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
    )
    trainer.train(epochs=epochs, iters_to_accumulate=iters_to_accumulate)


TRAIN_FUNCS = {
    "finetuned": (
        train_finetuned,
        {
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
            "data_type":"paired",
        }
    ),
    "contrastive": (
        train_contrastive,
        {
            "pretrained_bert_name": "huggingface/CodeBERTa-small-v1",
            "epochs": 4,
            # Learning rates and weight decays
            "lr_bert": 1e-5,
            "wd_bert": 1e-4,
            "lr_proj": 1e-4,
            "wd_proj": 1e-3,
            # Batch size
            "bs": 20,  # NOTE: Bigger batch size generally leads to better results in contrastive learning
            "iters_to_accumulate": 2,
            # Model specific parameters
            "freeze_bert": False,  # NOTE: if true the BERT model is not finetuned
            "dropout_rate": 0.2,
            "shuffle_dataloader": True,
            "is_self_supervised": False,
            "temperature": 0.5,
        }
    ),
}


if __name__ == "__main__":
    # TODO: Maybe load this from a .env or something
    set_seed(42)
    # Parse the model type first
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        choices=TRAIN_FUNCS.keys(),
        required=True, help="Model type to train",
    )
    known_args, unknown_args = parser.parse_known_args()
    # Select the train function and default parameters
    train_func, default_params = TRAIN_FUNCS[known_args.model_type]
    # Parse the rest of the parameters based on default ones
    parser = argparse.ArgumentParser()
    for param, default in default_params.items():
        parser.add_argument(f"--{param}", type=type(default), default=default)
    args = parser.parse_args(unknown_args)
    # Train the model
    train_func(**vars(args))
