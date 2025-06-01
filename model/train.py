import argparse
import numpy as np
import matplotlib.pyplot as plt
import random

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (
    DataLoader,
    random_split,
)

import datasets
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)

from sklearn.metrics import classification_report, roc_curve, auc

from model.code_sim_models import (
    CodeSimLinearCLS,
    CodeSimSBertLinearCLS,
    CodeSimSBertTripletENC,
    CodeSimSBertTripletCLS,
    CodeSimCombinedModel,
)
import model.code_sim_models as code_sim_models
import model.code_sim_datasets as code_sim_datasets
import model.configs as configs
from model.configs import TRAIN_ARGS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    bert = AutoModel.from_pretrained(pretrained_bert_name).to(DEVICE)
    bert_tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_name)

    params = {
        "padding":'max_length',  # Pad to max_length
        "max_length": bert_tokenizer.model_max_length,
        "truncation":True,       # Truncate to max_length
        "return_tensors":'pt'    # Return torch.Tensor objects
    }
    inputs = bert_tokenizer(code, **params)
    inputs_p = bert_tokenizer(code_p, **params)
    inputs_n = bert_tokenizer(code_n, **params)
    
    #pooling_strat=code_sim_models.AttentionPooler(768,768)
    model1 = CodeSimLinearCLS(bert).to(DEVICE)
    model2 = CodeSimSBertLinearCLS(bert).to(DEVICE)
    model3_1 = CodeSimSBertTripletENC(bert).to(DEVICE)
    model3_2 = CodeSimSBertTripletCLS(bert.config.hidden_size).to(DEVICE)
    model4 = CodeSimCombinedModel(bert).to(DEVICE)

    emb1 = model1(inputs)
    print(f"{model1.__class__.__name__} output shape:", emb1.shape)

    emb2 = model2(inputs, inputs)
    print(f"{model2.__class__.__name__} output shape:", emb2.shape)

    emb3 = model3_1(inputs)
    print(f"{model3_1.__class__.__name__} ENC output shape:", emb3.shape)
    out3 = model3_2(emb3, emb3)
    print(f"{model3_2.__class__.__name__} CLS output shape:", out3.shape)

    emb4 = model4(inputs)
    print(f"{model4.__class__.__name__} output shape:", emb4.shape)
    
    apn_inputs = { 
        key: torch.cat([inputs[key], inputs_p[key], inputs_n[key]]) for key in inputs.keys()
    }
    t_es, t_ls = model4.forward_train(apn_inputs)
    print(f"{model4.__class__.__name__} output shape for train pass:", t_es.shape, t_ls.shape)


def print_reports(y_true, y_pred, thresholds=(.5,.7,.9)):
    
    for threshold in thresholds:
        report = classification_report(y_true, [int(pred > threshold) for pred in y_pred])
        print(f"REPORT @ threshold={threshold}")
        print(report)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    _auc = auc(fpr, tpr)
    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'(AUC = {_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig("roc_curve.png")
    plt.show()


def get_loaders(dataset, bs, shuffle, train_ratio):
    if train_ratio < 0 or train_ratio > 1:
        raise ValueError("Train ratio must be between 0 and 1!")
    
    train_len = int(train_ratio * len(dataset))
    valid_len = len(dataset) - train_len
    train_data, valid_data = random_split(dataset, [train_len, valid_len])
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=shuffle)
    valid_loader = DataLoader(valid_data, batch_size=bs, shuffle=shuffle)
    return train_loader, valid_loader


def get_scheduler(loader, optimizer, epochs, iters_to_accumulate, warmup=0.1):
    if warmup < 0 or warmup > 1:
        raise ValueError("Warmup must be between 0 and 1!")
    
    # NOTE: Necessary to take into account Gradient accumulation
    num_training_steps = (epochs * len(loader)) // iters_to_accumulate
    num_warmup_steps = int(num_training_steps * warmup)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
    )
    return scheduler


def finetune_model(config: configs.BasicCodeSimClassifierConfig):
    if config.finetuning_strategy not in {"binary_cls_simpl", "binary_cls_sbert",}:
        raise ValueError("Invalid finetuning strategy.")
    if config.finetuning_strategy == "binary_cls_simpl":
        model_cls = code_sim_models.CodeSimLinearCLS
        loss_func = nn.BCEWithLogitsLoss()
        loss_hook = code_sim_models.compute_loss_logit
    if config.finetuning_strategy == "binary_cls_sbert":
        model_cls = code_sim_models.CodeSimSBertLinearCLS
        loss_func = nn.BCEWithLogitsLoss()
        loss_hook = code_sim_models.compute_loss_logit_SBert

    # Dataset creation
    return_single_encoding = config.finetuning_strategy == "binary_cls_simpl"  # Specifies encoding scheme in dataset
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_bert_name)
    dataset = code_sim_datasets.Create_CodeNet_paired_dataset(tokenizer=tokenizer, num_rows=config.num_rows, return_single_encoding=return_single_encoding)
    train_loader, valid_loader = get_loaders(dataset, config.bs, config.shuffle_dataloader, train_ratio=.8)

    bert_model = AutoModel.from_pretrained(config.pretrained_bert_name).to(DEVICE)

    model = model_cls(
        bert_model,
        freeze_bert=config.freeze_bert,
        dropout_rate=config.dropout_rate,
    )
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.wd)
    scheduler = get_scheduler(train_loader, optimizer, config.epochs, config.iters_to_accumulate)
    trainer = code_sim_models.CodeSimilarityTrainer(
        model,
        (train_loader, valid_loader),
        loss_func=loss_func,
        loss_hook=loss_hook,  # loss strategy
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
    )
    trainer.train(epochs=config.epochs, iters_to_accumulate=config.iters_to_accumulate)
    
    y_true, y_pred = eval_model_classifier(eval_data=valid_loader, model=model)
    print_reports(y_true, y_pred)


def finetune_model_triplet(config: configs.TripletCodeSimClassifierConfig, use_poj=False):
    distance_function = lambda x, y: 1 - F.cosine_similarity(x, y)
    loss_func = nn.TripletMarginWithDistanceLoss(distance_function=distance_function, margin=config.margin)
    
    if not config.use_info_nce_inspired_loss:
        loss_hook = code_sim_models.compute_loss_triplet
    else:
        loss_hook = code_sim_models.compute_loss_triplet_2
    
    # Dataset creation
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_bert_name)
    if use_poj:
        poj_dataset = datasets.load_dataset("semeru/Code-Code-CloneDetection-POJ104")
        train_dataset = code_sim_datasets.POJ104TripletDataset(poj_dataset["train"], tokenizer)
        valid_dataset = code_sim_datasets.POJ104TripletDataset(poj_dataset["validation"], tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=config.bs, shuffle=config.shuffle_dataloader)
        valid_loader = DataLoader(valid_dataset, batch_size=config.bs, shuffle=config.shuffle_dataloader)
    else:
        dataset = code_sim_datasets.Create_CodeNet_triplet_dataset(tokenizer=tokenizer, num_rows=config.num_rows)
        train_loader, valid_loader = get_loaders(dataset, config.bs, config.shuffle_dataloader, train_ratio=.8)

    bert_model = AutoModel.from_pretrained(config.pretrained_bert_name).to(DEVICE)

    enc_model = CodeSimSBertTripletENC(
        bert_model,
        freeze_bert=config.freeze_bert,
        dropout_rate=config.dropout_rate,
    )
    enc_model.to(DEVICE)

    optimizer = torch.optim.AdamW(enc_model.parameters(), lr=config.lr_enc, weight_decay=config.wd_enc)
    scheduler = get_scheduler(train_loader, optimizer, config.epochs, config.iters_to_accumulate)
    trainer = code_sim_models.CodeSimilarityTrainer(
        enc_model,
        (train_loader, valid_loader),
        loss_func=loss_func,
        loss_hook=loss_hook,  # loss strategy
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
    )
    trainer.train(epochs=config.epochs, iters_to_accumulate=config.iters_to_accumulate)
    
    y_true, y_pred = eval_model_triplet_simpl(eval_data=valid_loader, model=trainer.model)
    print_reports(y_true, y_pred)
    
    return
    # TODO: Train classifier model AND cache the embeddings before!!!
    cls_model = None
    
    y_true, y_pred = eval_model_triplet_chead(eval_data=valid_loader, cls_model=cls_model, enc_model=enc_model)
    print_reports(y_true, y_pred)


def finetune_model_combined(config: configs.CombinedCodeSimClassifierConfig):
    # Dataset creation
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_bert_name)
    
    dataset = code_sim_datasets.Create_CodeNet_triplet_dataset(tokenizer=tokenizer,
        num_rows=config.num_rows,
    )
    train_loader, valid_loader = get_loaders(dataset, config.bs, config.shuffle_dataloader, train_ratio=.8)

    bert_model = AutoModel.from_pretrained(config.pretrained_bert_name).to(DEVICE)

    model = CodeSimCombinedModel(
        bert_model,
        freeze_bert=config.freeze_bert,
        dropout_rate=config.dropout_rate,
    )
    model.to(DEVICE)
    
    loss_func = code_sim_models.Create_CombinedLoss(config.w_emb, config.w_cls, margin=config.margin)

    # NOTE: Allow different lr and wd for BERT and projection head params
    param_groups = [
        {"params": model.bert.parameters(), "lr": config.lr_bert, "weight_decay": config.wd_bert},
        {"params": model.emb_head.parameters(), "lr": config.lr_proj, "weight_decay": config.wd_proj},
        {"params": model.cls_head.parameters(), "lr": config.lr_proj, "weight_decay": config.wd_proj},
    ]
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = get_scheduler(train_loader, optimizer, config.epochs, config.iters_to_accumulate)
    trainer = code_sim_models.CodeSimilarityTrainer(
        model,
        (train_loader, valid_loader),
        loss_func=loss_func,
        loss_hook=code_sim_models.compute_loss_combined,  # loss strategy
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
    )
    trainer.train(epochs=config.epochs, iters_to_accumulate=config.iters_to_accumulate)
    # TODO: Evaluation logic here


def count_overlapping_pids(
    seed_value, factory,
    pretrained_bert_name="huggingface/CodeBERTa-small-v1",
    num_rows=50_000,
):
    set_seed(seed_value=seed_value)
    # Dataset creation
    tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_name)
    dataset = factory(tokenizer=tokenizer, num_rows=num_rows)
    
    train_len = int(0.8 * len(dataset))
    valid_len = len(dataset) - train_len
    train_data, valid_data = random_split(dataset, [train_len, valid_len])
    print(len(train_data), len(valid_data))
    
    # Access problem IDs from the splits
    train_pids = {dataset.pids[i] for i in train_data.indices}
    valid_pids = {dataset.pids[i] for i in valid_data.indices}
    valid_pids_l = [dataset.pids[i] for i in valid_data.indices]    
    
    # Count overlaps
    overlap_pids = train_pids.intersection(valid_pids)
    print(f"Count of problem IDs:\n{len(train_pids.union(valid_pids))}")
    print(f"Count of overlapping problem IDs between splits:\n{len(overlap_pids)}")
    num_overlapping_valid_rows = sum(1 for pid in valid_pids_l if pid in train_pids)
    print(f"Count of overlapping rows in validation split:\n{num_overlapping_valid_rows}")
    print(f"Ratio of overlapping rows in validation split:\n{num_overlapping_valid_rows/len(valid_data)}")


def eval(model, num_rows=5000):
    model.to(DEVICE)
    
    set_seed(42)
    tokenizer = code_sim_models.get_tokenizer(model.bert)

    if isinstance(model, CodeSimLinearCLS):
        dataset = code_sim_datasets.Create_CodeNet_paired_dataset(
            tokenizer=tokenizer,
            num_rows=num_rows,
            return_single_encoding=True,
        )
        evaluator = eval_model_classifier
    elif isinstance(model, CodeSimSBertLinearCLS):
        dataset = code_sim_datasets.Create_CodeNet_paired_dataset(
            tokenizer=tokenizer,
            num_rows=num_rows,
            return_single_encoding=False,
        )
        evaluator = eval_model_classifier
    elif isinstance(model, CodeSimSBertTripletCLS):
        dataset = code_sim_datasets.Create_CodeNet_triplet_dataset(
            tokenizer=tokenizer,
            num_rows=num_rows,
        )
        evaluator = eval_model_triplet_chead
    elif isinstance(model, CodeSimSBertTripletENC):
        dataset = code_sim_datasets.Create_CodeNet_triplet_dataset(
            tokenizer=tokenizer,
            num_rows=num_rows,
        )
        evaluator = eval_model_triplet_simpl
    else:
        raise ValueError(f"Invalid model type. {model.__class__.__name__}")

    # NOTE/TODO:
    # This replicates the split in the training function to create validation set of unseen data, 
    # this could be avoided by pre-splitting the datasets...
    _, valid_loader = get_loaders(dataset, bs=20, shuffle=False, train_ratio=0.8)
    
    y_true, y_pred = evaluator(eval_data=valid_loader, model=model)
    print_reports(y_true, y_pred)
    return y_true, y_pred


@torch.no_grad
def eval_model_classifier(eval_data: DataLoader,
                          model: CodeSimLinearCLS | CodeSimSBertLinearCLS):
    model.eval()
    y_true, y_pred = [], []
    for data in tqdm(eval_data):
        if isinstance(model, CodeSimLinearCLS):
            encs, labels = data
            code_sim_models.put_batch_encoding_to_device(encs, model.bert.device)
            logits = model.forward(encs)
        else:
            encs_u, encs_v, labels = data
            code_sim_models.put_batch_encoding_to_device(encs_u, model.bert.device)
            code_sim_models.put_batch_encoding_to_device(encs_v, model.bert.device)
            logits = model.forward(encs_u, encs_v)
        preds = torch.sigmoid(logits.squeeze(-1))
        # Store predictions and labels
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    return y_true, y_pred


@torch.no_grad
def eval_model_triplet_simpl(eval_data: DataLoader,
                             model: CodeSimSBertTripletENC):
    model.eval()
    y_true, y_pred = [], []
    for data in tqdm(eval_data):
        encs_a, encs_p, encs_n = data
        batch_size = encs_a["input_ids"].shape[0]
        inputs = {key: torch.cat([encs_a[key], encs_p[key], encs_n[key]]) for key in encs_a}
        code_sim_models.put_batch_encoding_to_device(inputs, model.bert.device)
        outputs = model.forward(inputs)
        embs_a, embs_p, embs_n = outputs.split(batch_size)
        # Calculate the pairwise cosine similarities
        dst_func = lambda x, y: F.cosine_similarity(x, y, dim=1)
        dst_p = dst_func(embs_a, embs_p)
        dst_n = dst_func(embs_a, embs_n)
        preds_p = dst_p.cpu().tolist()
        preds_n = dst_n.cpu().tolist()
        # Store predictions and labels
        y_true.extend([1] * len(preds_p))
        y_true.extend([0] * len(preds_n))
        y_pred.extend(preds_p)
        y_pred.extend(preds_n)
    return y_true, y_pred


@torch.no_grad
def eval_model_triplet_chead(eval_data: DataLoader,
                            enc_model: CodeSimSBertTripletENC,
                            cls_model: CodeSimSBertTripletCLS):
    enc_model.eval()
    cls_model.eval()
    y_true, y_pred = [], []
    for data in tqdm(eval_data):
        encs_a, encs_p, encs_n = data
        batch_size = encs_a["input_ids"].shape[0]
        inputs = {key: torch.cat([encs_a[key], encs_p[key], encs_n[key]]) for key in encs_a}
        code_sim_models.put_batch_encoding_to_device(inputs, enc_model.bert.device)
        outputs = enc_model.forward(inputs)
        embs_a, embs_p, embs_n = outputs.split(batch_size)
        preds_p = cls_model.forward(embs_a, embs_p).argmax(dim=1).long()
        preds_n = cls_model.forward(embs_a, embs_n).argmax(dim=1).long()
        # Convert to lists
        preds_p = preds_p.cpu().tolist()
        preds_n = preds_n.cpu().tolist()
        # Store predictions and labels
        y_true.extend([1] * len(preds_p))
        y_true.extend([0] * len(preds_n))
        y_pred.extend(preds_p)
        y_pred.extend(preds_n)
    return y_true, y_pred


if __name__ == "__main__":
    set_seed(42)  # TODO: Maybe load this from a .env or something
    
    TRAIN_FUNC = {
        "basic": finetune_model,
        "triplet": finetune_model_triplet,
        "combined": finetune_model_combined,
    }
    
    # Parse the model type first
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=TRAIN_FUNC.keys(), required=True, help="Model type to train")
    
    known_args, unknown_args = parser.parse_known_args()
    # Select the train function and its default parameters
    train_func = TRAIN_FUNC[known_args.model_type]
    train_args = TRAIN_ARGS[known_args.model_type]
    
    # Parse the rest of the parameters based on default ones
    parser = argparse.ArgumentParser()
    for param, default in train_args.items():
        parser.add_argument(f"--{param}", type=type(default), default=default)
    args = parser.parse_args(unknown_args)
    
    # Train the model
    train_func(**vars(args))
