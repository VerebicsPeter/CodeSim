import argparse
import numpy as np
import random

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
    CodeSimSBertLinearCLS,
    CodeSimSBertTripletENC,
    CodeSimSBertTripletCLS,
    CodeSimCombinedModel,
)
import model.code_sim_models as code_sim_models
import model.code_sim_datasets as code_sim_datasets

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
    model2 = CodeSimSBertLinearCLS(bert).to(DEVICE)
    model3 = CodeSimSBertTripletENC(bert).to(DEVICE)
    model4 = CodeSimCombinedModel(bert).to(DEVICE)

    emb1 = model1(inputs)
    print("Model 1 output shape:", emb1.shape)

    emb2 = model2(inputs, inputs)
    print("Model 2 output shape:", emb2.shape)

    emb3 = model3(inputs)
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
    model2 = CodeSimSBertLinearCLS(bert).to(DEVICE)
    model3 = CodeSimSBertTripletCLS(bert.config.hidden_size).to(DEVICE)
    model4 = CodeSimCombinedModel(bert).to(DEVICE)

    pred1 = model1.predict(code_a, code_b)
    print("Model 1 prediction output:", pred1, "shaped", pred1.shape)

    pred2 = model2.predict(code_a, code_b)
    print("Model 2 prediction output:", pred2, "shaped", pred2.shape)

    try:
        pred3 = model3.predict(code_a, code_b)
        print("Model 3 prediction output:", pred3, "shaped", pred3.shape)
    except Exception as e: print(e)
    
    try:
        pred4 = model4.predict(code_a, code_b)
        print("Model 4 prediction output:", pred4, "shaped", pred4.shape)
    except Exception as e: print(e)


# TODO: save reports instead of just printing
def print_reports(y_true, y_pred):
    report_1 = classification_report(y_true, [int(pred > 0.5) for pred in y_pred])
    report_2 = classification_report(y_true, [int(pred > 0.7) for pred in y_pred])
    report_3 = classification_report(y_true, [int(pred > 0.9) for pred in y_pred])
    print(report_1)
    print(report_2)
    print(report_3)
    # TODO: plot and save ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    print("FPR", fpr)
    print("TPR", tpr)
    print("THRESHS:", thresholds)


def finetune_model(
    finetuning_strategy="binary_cls_simpl",
    pretrained_bert_name="huggingface/CodeBERTa-small-v1",
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
):
    if finetuning_strategy not in {"binary_cls_simpl", "binary_cls_sbert",}:
        raise ValueError("Invalid finetuning strategy.")

    if finetuning_strategy == "binary_cls_simpl":
        model_cls = code_sim_models.CodeSimLinearCLS
        loss_func = nn.BCEWithLogitsLoss()
        loss_hook = code_sim_models.compute_loss_logit

    if finetuning_strategy == "binary_cls_sbert":
        model_cls = code_sim_models.CodeSimSBertLinearCLS
        loss_func = nn.BCEWithLogitsLoss()
        loss_hook = code_sim_models.compute_loss_logit_SBert

    # Dataset creation
    return_single_encoding = finetuning_strategy == "binary_cls_simpl"  # Specifies encoding scheme in dataset...
    tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_name)
    dataset = code_sim_datasets.Create_CodeNet_paired_dataset(tokenizer=tokenizer, num_rows=num_rows, return_single_encoding=return_single_encoding)
    
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
        pooling_strat=code_sim_models.AttentionPooler(encoder_dim=bert_model.conifg.hidden_size,
                                                      attention_dim=bert_model.conifg.hidden_size)
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
    
    y_true, y_pred = eval_model(eval_data=valid_data, model=model)
    print_reports(y_true, y_pred)


def finetune_model_triplet(
    pretrained_bert_name="huggingface/CodeBERTa-small-v1",
    epochs=4,
    lr_enc=1e-5,  # Encoder learning rate
    wd_enc=1e-5,  # Encoder weight decay
    lr_cls=1e-3,  # Classifier learning rate  
    wd_cls=1e-5,  # Classifier weight decay
    bs=1,  # Batch size
    iters_to_accumulate=2,
    # Model specific parameters
    freeze_bert=False,  # NOTE: if true the BERT model is not finetuned
    dropout_rate=0.2,
    shuffle_dataloader=True,
    num_rows=50,
    margin=1.0,  # Triplet loss function hyperparameter
):
    distance_function = lambda x, y: 1 - F.cosine_similarity(x, y)
    loss_func = nn.TripletMarginWithDistanceLoss(distance_function=distance_function, margin=margin)
    loss_hook = code_sim_models.compute_loss_triplet
    
    # Dataset creation
    tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_name)
    dataset = code_sim_datasets.Create_CodeNet_triplet_dataset(tokenizer=tokenizer, num_rows=num_rows)
    # TODO: Don't hardcode the train split ratio!
    train_len = int(0.8 * len(dataset))
    valid_len = len(dataset) - train_len
    train_data, valid_data = random_split(dataset, [train_len, valid_len])
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=shuffle_dataloader)
    valid_loader = DataLoader(valid_data, batch_size=bs, shuffle=shuffle_dataloader)

    bert_model = AutoModel.from_pretrained(pretrained_bert_name).to(DEVICE)

    enc_model = CodeSimSBertTripletENC(
        bert_model,
        freeze_bert=freeze_bert,
        dropout_rate=dropout_rate,
    )
    enc_model.to(DEVICE)

    optimizer = torch.optim.AdamW(enc_model.parameters(), lr=lr_enc, weight_decay=wd_enc)

    # Training and warmup steps
    # TODO: Don't hardcode the warmup ratio!
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
        enc_model,
        (train_loader, valid_loader),
        loss_func=loss_func,
        loss_hook=loss_hook,  # loss strategy
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
    )
    trainer.train(epochs=epochs, iters_to_accumulate=iters_to_accumulate)
    
    CLS_MODEL_PATH = "best_cls_model.pth"
    cls_model = trainer.train_cls_head(epochs=epochs, lr=lr_cls, weight_decay=wd_cls)
    y_true, y_pred = eval_model_triplet(eval_data=valid_data, cls_model=cls_model, enc_model=enc_model)
    torch.save(cls_model.state_dict(), CLS_MODEL_PATH)
    print_reports(y_true, y_pred)


def finetune_model_combined(
    pretrained_bert_name="huggingface/CodeBERTa-small-v1",
    epochs=4,
    # Learning rates and weight decays
    lr_bert=1e-5,
    wd_bert=1e-5,  # bert often needs smaller weight decay
    lr_proj=1e-4,
    wd_proj=1e-5,
    bs=1,
    iters_to_accumulate=2,
    # Model specific parameters
    freeze_bert=False,  # NOTE: if true the BERT model is not finetuned
    dropout_rate=0.2,
    shuffle_dataloader=True,
    num_rows=50,
    margin=1.0,
    w_emb=1.0,
    w_cls=1.0,
):
    # Dataset creation
    tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_name)
    
    dataset = code_sim_datasets.Create_CodeNet_triplet_dataset(
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
    
    loss_func = code_sim_models.Create_CombinedLoss(w_emb, w_cls, margin=margin)

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
    
    # TODO: Evaluation logic here


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
        evaluator = eval_model
    elif isinstance(model, CodeSimSBertLinearCLS):
        dataset = code_sim_datasets.Create_CodeNet_paired_dataset(
            tokenizer=tokenizer,
            num_rows=num_rows,
            return_single_encoding=False,
        )
        evaluator = eval_model
    elif isinstance(model, CodeSimSBertTripletCLS):
        dataset = code_sim_datasets.Create_CodeNet_triplet_dataset(
            tokenizer=tokenizer,
            num_rows=num_rows,
        )
        evaluator = eval_model_triplet
    elif isinstance(model, CodeSimSBertTripletENC):
        dataset = code_sim_datasets.Create_CodeNet_triplet_dataset(
            tokenizer=tokenizer,
            num_rows=num_rows,
        )
        evaluator = eval_model_triplet_simpl
    else:
        raise ValueError(f"Invalid model type. {model.__class__.__name__}")

    # NOTE, TODO This replicates the split in the training function to create validation set of
    # unseen data, this could be avoided by pre-splitting the datasets...
    train_len = int(0.8 * len(dataset))
    valid_len = len(dataset) - train_len
    _, valid_data = random_split(dataset, [train_len, valid_len])
    
    y_true, y_pred = evaluator(eval_data=valid_data, model=model)
    print_reports(y_true, y_pred)
    return y_true, y_pred


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


def eval_model_triplet(eval_data, enc_model: CodeSimSBertTripletENC, cls_model: CodeSimSBertTripletCLS):
    enc_model.eval()
    cls_model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for encs_a, encs_p, encs_n in tqdm(DataLoader(eval_data, batch_size=20)):
            code_sim_models.put_batch_encoding_to_device(encs_a, DEVICE)
            code_sim_models.put_batch_encoding_to_device(encs_p, DEVICE)
            code_sim_models.put_batch_encoding_to_device(encs_n, DEVICE)
            embs_a = enc_model(encs_a)  # anchor
            embs_p = enc_model(encs_p)  # positive
            embs_n = enc_model(encs_n)  # negative
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


def eval_model_triplet_simpl(eval_data: Subset, model):
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
    "triplet": (
        finetune_model_triplet,
        {
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
        }
    ),
    # TODO: set combined loss parameters with grid search, as defaults could lead to unbalanced loss
    "combined": (
        finetune_model_combined,
        {
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
    ),
}


if __name__ == "__main__":
    set_seed(42)  # TODO: Maybe load this from a .env or something
    
    # Parse the model type first
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=TRAIN_FUNCS.keys(), required=True, help="Model type to train")
    
    known_args, unknown_args = parser.parse_known_args()
    
    # Select the train function and its default parameters
    train_func, default_params = TRAIN_FUNCS[known_args.model_type]
    
    # Parse the rest of the parameters based on default ones
    parser = argparse.ArgumentParser()
    for param, default in default_params.items():
        parser.add_argument(f"--{param}", type=type(default), default=default)
    args = parser.parse_args(unknown_args)
    
    # Train the model
    train_func(**vars(args))
