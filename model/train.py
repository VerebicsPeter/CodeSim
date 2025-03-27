import gdown
import argparse
import random
import numpy as np
import pandas as pd
import pprint as pp

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (
    Dataset,
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

from sklearn.metrics import classification_report

import model.code_sim_models as code_sim_models
import model.code_sim_datasets as code_sim_datasets

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def download_dataset(url, output_file):
    gdown.download(url, output_file, quiet=False)


FINETUNING_STRATEGIES = {
    "linear_binary_cls",
    "sbert_binary_cls",
    "sbert_triplet_cls",
}

DATASET_TYPE = {
    "paired",
    "triplet",
}

# TODO: Maybe load URLS from a .env or something
DATASET_URLS = {
    "paired": "https://drive.google.com/uc?export=download&id=1pUErbyZw1fBC5gIe6KT7BWga7h6Bfr4l",
    "triplet": "https://drive.google.com/uc?export=download&id=11aBIxIMEMKoGyJ9IdUHY2XQv1ZzfyXd2",
    "contrastive_labeled": "https://drive.google.com/uc?export=download&id=1UteITBYXcBLt2hXviy71jQr-oXceVcs5",
    "contrastive_unlabeled": "https://drive.google.com/uc?export=download&id=1iHHgOcJQ_qp3sk3d7w1zpWBvsgDqrPJV",
}


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

    inputs = bert_tokenizer(code, return_tensors="pt", truncation=True, padding=True)
    model1 = code_sim_models.CodeSimLinearCLS(bert).to(DEVICE)
    model2 = code_sim_models.CodeSimSBertTripletCLS(bert).to(DEVICE)
    model3 = code_sim_models.CodeSimContrastiveCLS(bert).to(DEVICE)
    model4 = code_sim_models.CodeSimSBertLinearCLS(bert).to(DEVICE)

    emb1 = model1(inputs)
    print("Model 1 output shape:", emb1.shape)

    emb2 = model2(inputs)
    print("Model 2 output shape:", emb2.shape)

    emb3 = model3(inputs)
    print("Model 3 output shape:", emb3.shape)
    
    emb4 = model4(inputs, inputs)
    print("Model 4 output shape:", emb4.shape)


def test_predict_passes(pretrained_bert_name: str = "huggingface/CodeBERTa-small-v1"):
    code_a = """print("Hello, World!")"""
    code_b = """def add(x,y): return x+y"""

    bert = AutoModel.from_pretrained(pretrained_bert_name).to(DEVICE)
    model1 = code_sim_models.CodeSimLinearCLS(bert).to(DEVICE)
    model2 = code_sim_models.CodeSimSBertTripletCLS(bert).to(DEVICE)
    model3 = code_sim_models.CodeSimContrastiveCLS(bert).to(DEVICE)
    model4 = code_sim_models.CodeSimSBertLinearCLS(bert).to(DEVICE)

    pred1 = model1.predict(code_a, code_b)
    print("Model 1 prediction output:", pred1, "shaped", pred1.shape)

    pred2 = model2.predict(code_a, code_b)
    print("Model 2 prediction output:", pred2, "shaped", pred2.shape)

    pred3 = model3.predict(code_a, code_b)
    print("Model 3 prediction output:", pred3, "shaped", pred3.shape)
    
    pred4 = model4.predict(code_a, code_b)
    print("Model 4 prediction output:", pred4, "shaped", pred4.shape)


def Create_CodeNet_paired_dataset(data_path: str, tokenizer, num_rows=5000,
                                  return_single_encoding=True):
    download_dataset(data_path, "dataset.csv")
    df = pd.read_csv(
        "dataset.csv", header=0,
        names=code_sim_datasets.CodeNetPairDataset.COLUMNS
    )
    print("CodeNet data loaded. Data type: paired")
    pp.pp(df)

    dataset = code_sim_datasets.CodeNetPairDataset.from_pandas_df(
        df,
        tokenizer=tokenizer,
        num_rows=num_rows,
        return_single_encoding=return_single_encoding,
    )
    return dataset


def Create_CodeNet_triplet_dataset(data_path: str, tokenizer, num_rows=5000):
    download_dataset(data_path, "dataset.csv")
    df = pd.read_csv(
        "dataset.csv", header=0,
        names=code_sim_datasets.CodeNetTripletDataset.COLUMNS
    )
    print("CodeNet data loaded. Data type: paired")
    pp.pp(df)

    dataset = code_sim_datasets.CodeNetTripletDataset.from_pandas_df(
        df,
        tokenizer=tokenizer,
        num_rows=num_rows,
    )
    return dataset


def train_finetuned(
    finetuning_strategy="linear_binary_cls",
    pretrained_bert_name: str = "huggingface/CodeBERTa-small-v1",
    epochs=4,
    lr=1e-5,  # Learning rate
    wd=1e-5,  # Weight decay
    bs=20,  # Batch size
    # The gradient accumulation adds gradients over an effective batch of size : bs * iters_to_accumulate.
    # If set to "1", you get the usual batch size
    iters_to_accumulate=2,
    # Model specific parameters
    freeze_bert=False,  # NOTE: if true the BERT model is not finetuned
    dropout_rate=0.2,
    shuffle_dataloader=True,
    num_rows=5000,
):
    if finetuning_strategy not in FINETUNING_STRATEGIES:
        raise ValueError("Invalid finetuning strategy.")
    
    if finetuning_strategy == "linear_binary_cls":
        model_cls = code_sim_models.CodeSimLinearCLS
        loss_func = nn.BCEWithLogitsLoss()
        loss_hook = code_sim_models.compute_loss_logit
    
    if finetuning_strategy == "sbert_binary_cls":
        model_cls = code_sim_models.CodeSimSBertLinearCLS
        loss_func = nn.BCEWithLogitsLoss()
        loss_hook = code_sim_models.compute_loss_SBERT_logit
    
    if finetuning_strategy == "sbert_triplet_cls":
        model_cls = code_sim_models.CodeSimSBertTripletCLS
        loss_func = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1 - F.cosine_similarity(x, y),
            margin=1.0
        )
        loss_hook = code_sim_models.compute_loss_SBERT_triplet

    tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_name)
    
    if finetuning_strategy in {"linear_binary_cls", "sbert_binary_cls"}:
        return_single_encoding = finetuning_strategy == "linear_binary_cls"
        dataset = Create_CodeNet_paired_dataset(
            DATASET_URLS["paired"],
            tokenizer=tokenizer,
            num_rows=num_rows,
            return_single_encoding=return_single_encoding,
        )
    else:
        dataset = Create_CodeNet_triplet_dataset(
            DATASET_URLS["triplet"],
            tokenizer=tokenizer,
            num_rows=num_rows,
        )
    
    # TODO: Don't hardcode the train split ratio!
    train_len = int(0.8 * len(dataset))
    valid_len = len(dataset) - train_len
    train_data, valid_data = random_split(dataset, [train_len, valid_len])
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=shuffle_dataloader)
    valid_loader = DataLoader(valid_data, batch_size=bs, shuffle=shuffle_dataloader)

    bert_model = AutoModel.from_pretrained(pretrained_bert_name).to(DEVICE)

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
    num_warmup_steps = int(num_training_steps * 0.1)  # 10% warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        # Necessary to take into account Gradient accumulation
        num_training_steps=num_training_steps,
        # The number of steps for the warmup phase.
        num_warmup_steps=num_warmup_steps,
    )

    trainer = code_sim_models.CodeSimilarityTrainer(
        model,
        (train_loader, valid_loader),
        loss_func=loss_func,
        loss_hook=loss_hook,  # loss strategy
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
    )
    trainer.train(epochs=epochs, iters_to_accumulate=iters_to_accumulate)
    
    if finetuning_strategy in {"linear_binary_cls", "sbert_binary_cls"}:
        eval_model(eval_data=valid_data, model=model, threshold=0.5)
        eval_model(eval_data=valid_data, model=model, threshold=0.6)
        eval_model(eval_data=valid_data, model=model, threshold=0.7)
        eval_model(eval_data=valid_data, model=model, threshold=0.8)
        eval_model(eval_data=valid_data, model=model, threshold=0.9)
    else:
        eval_model_triplet(eval_data=valid_data, model=model, threshold=0.5)
        eval_model_triplet(eval_data=valid_data, model=model, threshold=0.7)
        eval_model_triplet(eval_data=valid_data, model=model, threshold=0.9)


def train_contrastive(
    pretrained_bert_name: str = "huggingface/CodeBERTa-small-v1",
    epochs=4,
    # Learning rates and weight decays
    lr_bert=1e-5,
    wd_bert=1e-4,  # bert often needs smaller weight decay
    lr_proj=1e-4,
    wd_proj=1e-3,
    bs=20,  # NOTE: Bigger batch size generally leads to better results in contrastive learning
    iters_to_accumulate=2,
    # Model specific parameters
    freeze_bert=False,  # NOTE: if true the BERT model is not finetuned
    dropout_rate=0.2,
    shuffle_dataloader=True,
    # Flag for self supervised training
    is_self_supervised=False,
    # Temperature hyperparameter for NTXent loss
    temperature=0.5,
    num_rows=5000,
):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_name)

    if is_self_supervised:
        url = DATASET_URLS["contrastive_selfsup"]
        cls = code_sim_datasets.SelfSupCodeDataset
        ntxent_loss = losses.NTXentLoss(temperature=temperature)
    else:
        url = DATASET_URLS["contrastive_labeled"]
        cls = code_sim_datasets.LabeledCodeDataset
        # Wrap the NTXent loss function if training with a self supervised method
        ntxent_loss = losses.SelfSupervisedLoss(
            losses.NTXentLoss(temperature=temperature)
        )

    dataset = cls.from_csv_data(
        path=url,
        tokenizer=tokenizer,
        aug_funcs=[],  # NOTE: precalculated data augmentation functions can be added here
        device=DEVICE,
    )
    print("Initial dataset size:", len(dataset))

    # TODO: don't hardcode this number V
    sample_size = min(len(dataset), num_rows)
    dataset = Subset(dataset, list(range(sample_size)))

    print("Sampled dataset size:", len(dataset))

    train_len = int(0.8 * len(dataset))
    valid_len = len(dataset) - train_len
    train_data, valid_data = random_split(dataset, [train_len, valid_len])
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=shuffle_dataloader)
    valid_loader = DataLoader(valid_data, batch_size=bs, shuffle=shuffle_dataloader)

    if shuffle_dataloader:
        print("Dataloaders will be shuffled...")

    bert_model = AutoModel.from_pretrained(pretrained_bert_name).to(DEVICE)

    model = code_sim_models.CodeSimContrastiveCLS(
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
    num_warmup_steps = int(num_training_steps * 0.1)  # 10% warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        # The number of steps for the warmup phase.
        num_warmup_steps=num_warmup_steps,
    )

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


def eval_model(eval_data: Subset, model: code_sim_models.SimilarityClassifier, threshold: float):
    class RawCodeWrapper(Dataset):
        def __init__(self, subset: Subset):
            # Subset of the original dataset
            self.subset = subset

        def __getitem__(self, idx):
            original_idx = self.subset.indices[idx]
            return (
                self.subset.dataset.codes_a[original_idx],
                self.subset.dataset.codes_b[original_idx],
                self.subset.dataset.labels[original_idx],
            )

        def __len__(self):
            return len(self.subset)
    
    dataset = RawCodeWrapper(eval_data)
    
    model.eval()
    
    true_labels, predictions = [], []
    with torch.no_grad():
        for (codes1, codes2, labels) in tqdm(DataLoader(dataset, batch_size=20)):
            preds = model.predict(codes1, codes2)
            # Store predictions and labels
            true_labels.extend(labels.cpu().tolist())
            predictions.extend(preds.cpu().tolist())
    
    report = classification_report(true_labels,
                                   [int(prediction > threshold) for prediction in predictions])
    print(report)


def eval_model_triplet(eval_data: Subset, model: code_sim_models.SimilarityClassifier, threshold: float):
    class TripletRawCodeWrapper(Dataset):
        def __init__(self, subset: Subset):
            # Subset of the original dataset
            self.subset = subset

        def __getitem__(self, idx):
            original_idx = self.subset.indices[idx]
            return (
                self.subset.dataset.codes_a[original_idx],
                self.subset.dataset.codes_p[original_idx],
                self.subset.dataset.codes_n[original_idx],
            )

        def __len__(self):
            return len(self.subset)
    
    dataset = TripletRawCodeWrapper(eval_data)
    
    model.eval()
    
    true_labels, predictions = [], []
    with torch.no_grad():
        for (codes_a, codes_p, codes_n) in tqdm(DataLoader(dataset, batch_size=20)):
            preds_p = model.predict(codes_a, codes_p)
            preds_n = model.predict(codes_a, codes_n)
            # Convert to lists
            preds_p = preds_p.cpu().tolist()
            preds_n = preds_n.cpu().tolist()
            # Store predictions and labels
            true_labels.extend([1]*len(preds_p))
            predictions.extend(preds_p)
            true_labels.extend([0]*len(preds_n))
            predictions.extend(preds_n)
    
    report = classification_report(true_labels,
                                   [int(prediction > threshold) for prediction in predictions])
    print(report)

TRAIN_FUNCS = {
    "finetuned": (
        train_finetuned,
        {
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
            "num_rows": 5000,
        },
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
        required=True,
        help="Model type to train",
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
