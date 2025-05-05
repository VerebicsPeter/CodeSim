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

from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)

from sklearn.metrics import classification_report, roc_curve

from model.code_sim_models import (
    SimilarityClassifier,
    CodeSimLinearCLS,
    CodeSimCombinedModel,
    CodeSimSBertLinearCLS,
    CodeSimSBertTripletCLS,
)
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

# TODO: Create proper dataset with train, validation, evalualion splits for clean evaluation,
# idea: pick a set of 'evaluation' problems distinct from training and validation problems
# TODO: Maybe load URLS from a .env or something
DATASET_URLS = {
    "paired" : "https://drive.google.com/uc?export=download&id=1pUErbyZw1fBC5gIe6KT7BWga7h6Bfr4l",
    "triplet": "https://drive.google.com/uc?export=download&id=11aBIxIMEMKoGyJ9IdUHY2XQv1ZzfyXd2",
    # NOTE: Old datasets
    #"contrastive_labeled"  : "https://drive.google.com/uc?export=download&id=1UteITBYXcBLt2hXviy71jQr-oXceVcs5",
    #"contrastive_unlabeled": "https://drive.google.com/uc?export=download&id=1iHHgOcJQ_qp3sk3d7w1zpWBvsgDqrPJV",
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
    code_p = """hello_str = "Hello, World!"; print(hello_str)"""
    code_n = """def add(x,y): return x+y"""

    bert_tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_name)
    bert = AutoModel.from_pretrained(pretrained_bert_name).to(DEVICE)

    params = {
        "padding":'max_length',  # Pad to max_length
        "max_length": bert_tokenizer.model_max_length,
        "truncation":True,       # Truncate to max_length
        "return_tensors":'pt'    # Return torch.Tensor objects
    }
    inputs = bert_tokenizer(code, **params)
    inputs_p = bert_tokenizer(code_p, **params)
    inputs_n = bert_tokenizer(code_n, **params)
    
    model1 = CodeSimLinearCLS(bert).to(DEVICE)
    model2 = CodeSimSBertTripletCLS(bert).to(DEVICE)
    model3 = CodeSimSBertLinearCLS(bert).to(DEVICE)
    model4 = CodeSimCombinedModel(bert).to(DEVICE)

    emb1 = model1(inputs)
    print("Model 1 output shape:", emb1.shape)

    emb2 = model2(inputs)
    print("Model 2 output shape:", emb2.shape)

    emb3 = model3(inputs, inputs)
    print("Model 3 output shape:", emb3.shape)

    emb4 = model4(inputs)
    print("Model 4 output shape:", emb4.shape)
    
    apn_inputs = { 
        key: torch.cat([
            inputs[key], inputs_p[key], inputs_n[key],
            # NOTE: other triplets may be added here...
        ])
        for key in inputs.keys()
    }
    
    t_es, t_ls = model4.forward_train(apn_inputs)
    print("Model 4 triplet output shape:\n", t_es.shape, t_ls.shape)



def test_predict_passes(pretrained_bert_name: str = "huggingface/CodeBERTa-small-v1"):
    code_a = """print("Hello, World!")"""
    code_b = """def add(x,y): return x+y"""

    bert = AutoModel.from_pretrained(pretrained_bert_name).to(DEVICE)
    model1 = CodeSimLinearCLS(bert).to(DEVICE)
    model2 = CodeSimSBertTripletCLS(bert).to(DEVICE)
    model3 = CodeSimSBertLinearCLS(bert).to(DEVICE)
    model4 = CodeSimCombinedModel(bert).to(DEVICE)

    pred1 = model1.predict(code_a, code_b)
    print("Model 1 prediction output:", pred1, "shaped", pred1.shape)

    pred2 = model2.predict(code_a, code_b)
    print("Model 2 prediction output:", pred2, "shaped", pred2.shape)

    pred3 = model3.predict(code_a, code_b)
    print("Model 3 prediction output:", pred3, "shaped", pred3.shape)
    return
    pred4 = model4.predict(code_a, code_b)
    print("Model 4 prediction output:", pred4, "shaped", pred4.shape)


def Create_CodeNet_paired_dataset(data_path: str, tokenizer, num_rows=5000,
                                  return_single_encoding=True
):
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
    print("CodeNet data loaded. Data type: triplet")
    pp.pp(df)

    dataset = code_sim_datasets.CodeNetTripletDataset.from_pandas_df(
        df,
        tokenizer=tokenizer,
        num_rows=num_rows, 
    )
    return dataset


def finetune_model(
    finetuning_strategy="linear_binary_cls",
    pretrained_bert_name: str = "huggingface/CodeBERTa-small-v1",
    epochs=4,
    lr=1e-5,  # Learning rate
    wd=1e-5,  # Weight decay
    bs=1,  # Batch size
    # The gradient accumulation adds gradients over an effective batch of size : bs * iters_to_accumulate.
    # If set to "1", you get the usual batch size
    iters_to_accumulate=2,
    # Model specific parameters
    freeze_bert=False,  # NOTE: if true the BERT model is not finetuned
    dropout_rate=0.2,
    shuffle_dataloader=True,
    num_rows=50,
    margin=1.0,  # Loss function hyperparameter
):
    if finetuning_strategy not in FINETUNING_STRATEGIES:
        raise ValueError("Invalid finetuning strategy.")

    if finetuning_strategy == "linear_binary_cls":
        model_cls = code_sim_models.CodeSimLinearCLS
        loss_func = nn.BCEWithLogitsLoss()
        loss_hook = code_sim_models.compute_loss_logit
        evaluator = eval_model

    if finetuning_strategy == "sbert_binary_cls":
        model_cls = code_sim_models.CodeSimSBertLinearCLS
        loss_func = nn.BCEWithLogitsLoss()
        loss_hook = code_sim_models.compute_loss_SBERT_logit
        evaluator = eval_model

    if finetuning_strategy == "sbert_triplet_cls":
        distance_function = lambda x, y: 1 - F.cosine_similarity(x, y)
        model_cls = code_sim_models.CodeSimSBertTripletCLS
        loss_func = nn.TripletMarginWithDistanceLoss(distance_function=distance_function, margin=margin)
        loss_hook = code_sim_models.compute_loss_SBERT_triplet
        evaluator = eval_model_triplet

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

    def print_reports(y_true, y_pred):
        # TODO: save reports
        report_1 = classification_report(y_true, [int(pred > 0.5) for pred in y_pred])
        report_2 = classification_report(y_true, [int(pred > 0.7) for pred in y_pred])
        report_3 = classification_report(y_true, [int(pred > 0.9) for pred in y_pred])
        print(report_1)
        print(report_2)
        print(report_3)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        print("FPR", fpr)
        print("TPR", tpr)
        print("THRESHS:", thresholds)

    y_true, y_pred = evaluator(eval_data=valid_data, model=model)
    print_reports(y_true, y_pred)


def finetune_model_combined(
    pretrained_bert_name: str = "huggingface/CodeBERTa-small-v1",
    epochs=4,
    # Learning rates and weight decays
    lr_bert=1e-5,
    wd_bert=1e-4,  # bert often needs smaller weight decay
    lr_proj=1e-4,
    wd_proj=1e-3,
    bs=1,
    iters_to_accumulate=2,
    # Model specific parameters
    freeze_bert=False,  # NOTE: if true the BERT model is not finetuned
    dropout_rate=0.2,
    shuffle_dataloader=True,
    num_rows=50,
):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_name)

    dataset = Create_CodeNet_triplet_dataset(
        data_path=DATASET_URLS["triplet"],
        tokenizer=tokenizer,
        num_rows=num_rows,
    )

    train_len = int(0.8 * len(dataset))
    valid_len = len(dataset) - train_len
    train_data, valid_data = random_split(dataset, [train_len, valid_len])
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=shuffle_dataloader)
    valid_loader = DataLoader(valid_data, batch_size=bs, shuffle=shuffle_dataloader)

    if shuffle_dataloader:
        print("Dataloaders will be shuffled...")

    bert_model = AutoModel.from_pretrained(pretrained_bert_name).to(DEVICE)

    model = CodeSimCombinedModel(
        bert_model,
        freeze_bert=freeze_bert,
        dropout_rate=dropout_rate,
    )
    model.to(DEVICE)

    # NOTE: Allow different lr and wd for BERT and projection head params
    param_groups = [
        {"params": model.bert.parameters(), "lr": lr_bert, "weight_decay": wd_bert},
        {"params": model.emb_head.parameters(), "lr": lr_proj, "weight_decay": wd_proj},
        {"params": model.cls_head.parameters(), "lr": lr_proj, "weight_decay": wd_proj},
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
    
    loss_func = code_sim_models.Create_CombinedLoss()

    trainer = code_sim_models.CodeSimilarityTrainer(
        model,
        (train_loader, valid_loader),
        loss_func=loss_func,
        loss_hook=code_sim_models.compute_loss_combined,  # loss strategy
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
    )
    trainer.train(epochs=epochs, iters_to_accumulate=iters_to_accumulate)
    
    # TODO: evaluation here


def eval_model(eval_data: Subset, model: SimilarityClassifier):
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

    y_true, y_pred = [], []
    with torch.no_grad():
        for codes1, codes2, labels in tqdm(DataLoader(dataset, batch_size=20)):
            preds = model.predict(codes1, codes2)
            # Store predictions and labels
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    return y_true, y_pred


def eval_model_triplet(eval_data: Subset, model: SimilarityClassifier):
    class RawCodeWrapper(Dataset):
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

    dataset = RawCodeWrapper(eval_data)

    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for codes_a, codes_p, codes_n in tqdm(DataLoader(dataset, batch_size=20)):
            preds_p = model.predict(codes_a, codes_p)
            preds_n = model.predict(codes_a, codes_n)
            # Convert to lists
            preds_p = preds_p.cpu().tolist()
            preds_n = preds_n.cpu().tolist()
            # Store predictions and labels
            y_true.extend([1] * len(preds_p))
            y_true.extend([0] * len(preds_n))
            y_pred.extend(preds_p)
            y_pred.extend(preds_n)

    return y_true, y_pred


TRAIN_FUNCS = {
    "basic": (
        finetune_model,
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
    "combined": (
        finetune_model_combined,
        {
            "pretrained_bert_name": "huggingface/CodeBERTa-small-v1",
            "epochs": 4,
            # Learning rates and weight decays
            "lr_bert": 1e-5,
            "wd_bert": 1e-5,
            "lr_proj": 1e-4,
            "wd_proj": 1e-4,
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
