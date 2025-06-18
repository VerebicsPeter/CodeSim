import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import get_linear_schedule_with_warmup

from sklearn.metrics import classification_report, roc_curve, auc

from model.code_sim_models import (
    CodeSimLinearClassifierCross,
    CodeSimLinearClassifierSBert,
    CodeSimContrastiveEncoder,
)
import model.code_sim_models as code_sim_models
import model.code_sim_datasets as code_sim_datasets
import model.metrics as metrics
import model.configs as configs
from model.utils import set_seed


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def finetune_model(config: configs.CodeSimClassifierConfig):
    config.init_model()
    
    if config.finetuning_strategy not in configs.FINETUNING_STRATEGIES:
        raise ValueError(f"Invalid finetuning strategy: {config.finetuning_strategy}.")
    
    if config.finetuning_strategy == "binary_cls_simpl":
        model_cls = code_sim_models.CodeSimLinearClassifierCross
        loss_func = nn.BCEWithLogitsLoss()
        loss_hook = code_sim_models.compute_loss_logit_Cross
    
    if config.finetuning_strategy == "binary_cls_sbert":
        model_cls = code_sim_models.CodeSimLinearClassifierSBert
        loss_func = nn.BCEWithLogitsLoss()
        loss_hook = code_sim_models.compute_loss_logit_SBert

    # Dataset Creation
    # NOTE: Finetuning strategy specifies encoding scheme in dataset:
    return_single_encoding = config.finetuning_strategy == "binary_cls_simpl"
    tokenizer_max_length = 512 if return_single_encoding else 256
    
    train_data, valid_data, test_data = code_sim_datasets.Create_CodeNet_paired_dataset(
        tokenizer_name=config.pretrained_bert_name,
        tokenizer_max_length=tokenizer_max_length,
        return_single_encoding=return_single_encoding,
    )
    train_loader, valid_loader, test_loader = code_sim_datasets.get_loaders(
        train_data, valid_data, test_data,
        config.bs,
        config.shuffle_dataloader,
        config.num_workers,
        config.num_rows,
    )

    # Model Creation
    model = model_cls(
        config.pretrained_bert,
        freeze_bert=config.freeze_bert,
        dropout_rate=config.dropout_rate,
    )
    model.to(DEVICE)
    
    # Trainer
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
    
    y_true, y_pred = eval_model_classifier(eval_data=test_loader, model=model)
    print_reports(y_true, y_pred)


def finetune_model_contrastive(config: configs.CodeSimContrastiveClassifierConfig, use_poj=False):
    config.init_model()
    
    if config.use_info_nce_inspired_loss:
        loss_func = None  # TODO: maybe make a custom loss for the hardcoded loss in method below
        loss_hook = lambda trainer, batched_data: code_sim_models.compute_loss_tuplet(trainer, batched_data, config.temp)
    else:
        distance_function = lambda x, y: 1 - F.cosine_similarity(x, y)  # cosine distance
        loss_func = nn.TripletMarginWithDistanceLoss(distance_function=distance_function, margin=config.margin)
        loss_hook = code_sim_models.compute_loss_triplet
    
    # Dataset Creation
    if use_poj:
        poj_dataset = code_sim_datasets.Create_POJ104_triplet_dataset(
            tokenizer_name=config.pretrained_bert_name,
        )
        train_dataset, valid_dataset, test_dataset_map, test_dataset_cls = poj_dataset
        train_loader = DataLoader(train_dataset, batch_size=config.bs, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=config.bs, shuffle=False)
        test_loader_map = DataLoader(test_dataset_map, batch_size=config.bs, shuffle=False)
        test_loader_cls = DataLoader(test_dataset_cls, batch_size=config.bs, shuffle=False)
    else:
        train_data, valid_data, test_data = code_sim_datasets.Create_CodeNet_triplet_dataset(
            tokenizer_name=config.pretrained_bert_name,
            tokenizer_max_length=256,
            num_negatives=config.num_negatives
        )
        
        train_loader = DataLoader(
            train_data, 
            batch_sampler=code_sim_datasets.CodeNetRandomTripletBatchSampler(
                pids=train_data.pids,
                num_batches=config.num_batches,
                num_pids_per_batch=config.bs
            ),
            collate_fn=code_sim_datasets.custom_collate_triplet
        )
        
        valid_loader = DataLoader(
            valid_data, batch_size=config.bs,
            sampler=code_sim_datasets.CodeNetDefaultTripletSampler(valid_data.pids),
            collate_fn=code_sim_datasets.custom_collate_triplet
        )
        
        test_loader = DataLoader(
            test_data, batch_size=config.bs,
            sampler=code_sim_datasets.CodeNetDefaultTripletSampler(test_data.pids),
            collate_fn=code_sim_datasets.custom_collate_triplet
        )

    # Model Creation
    model = CodeSimContrastiveEncoder(
        config.pretrained_bert,
        freeze_bert=config.freeze_bert,
        dropout_rate=config.dropout_rate,
    )
    model.to(DEVICE)

    # Trainer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr_enc, weight_decay=config.wd_enc)
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
    
    if use_poj:
        print("Evaluating on POJ-104 retriaval...")
        mapr           = eval_model_contrastive_map(eval_data=test_loader_map, model=trainer.model)
        print(f"MAP @ R=499 : {mapr}")
        print("Evaluating on POJ-104 classification...")
        y_true, y_pred = eval_model_contrastive_cls(eval_data=test_loader_cls, model=trainer.model)
        print_reports(y_true, y_pred)
    else:
        y_true, y_pred = eval_model_contrastive_cls(eval_data=test_loader, model=trainer.model)
        print_reports(y_true, y_pred)


def eval(model, num_rows=5000):
    set_seed(42)
    model.to(DEVICE)

    if isinstance(model, CodeSimLinearClassifierCross):
        dataset = code_sim_datasets.Create_CodeNet_paired_dataset(
            tokenizer_name=model.bert.name_or_path,
            tokenizer_max_length=512,
            return_single_encoding=True,
        )
    elif isinstance(model, CodeSimLinearClassifierSBert):
        dataset = code_sim_datasets.Create_CodeNet_paired_dataset(
            tokenizer_name=model.bert.name_or_path,
            tokenizer_max_length=256,
            return_single_encoding=False,
        )
    else:
        raise ValueError(f"Invalid model type. {model.__class__.__name__}")

    # NOTE/TODO:
    _, _, test_loader = code_sim_datasets.get_loaders(
        *dataset,
        bs=20,
        shuffle=False,
        num_rows=num_rows,
    )
    
    y_true, y_pred = eval_model_classifier(eval_data=test_loader, model=model)
    print_reports(y_true, y_pred)
    return y_true, y_pred


@torch.no_grad
def eval_model_classifier(eval_data: DataLoader, model: CodeSimLinearClassifierCross | CodeSimLinearClassifierSBert):
    model.eval()
    y_true, y_pred = [], []
    for data in tqdm(eval_data):
        if isinstance(model, CodeSimLinearClassifierCross):
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
def eval_model_contrastive_cls(eval_data: DataLoader, model: CodeSimContrastiveEncoder):
    model.eval()
    y_true, y_pred = [], []
    dst_func = lambda x, y: F.cosine_similarity(x, y, dim=1)
    for data in tqdm(eval_data):
        encs_a, encs_p, encs_n = data
        batch_size = encs_a["input_ids"].shape[0]
        inputs = {key: torch.cat([encs_a[key], encs_p[key], encs_n[key]]) for key in encs_a}
        code_sim_models.put_batch_encoding_to_device(inputs, model.bert.device)
        outputs = model.forward(inputs)
        embs_a, embs_p, embs_n = outputs.split(batch_size)
        # Calculate the pairwise cosine similarities
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
def eval_model_contrastive_map(eval_data: DataLoader, model: CodeSimContrastiveEncoder):
    model.eval()
    all_embs = []
    all_lbls = []
    for data in tqdm(eval_data):
        encs, lbls = data
        code_sim_models.put_batch_encoding_to_device(encs, model.bert.device)
        embs = model.forward(encs)
        lbls = lbls.to(model.bert.device)
        all_embs.append(embs)
        all_lbls.append(lbls)
    all_embs = torch.cat(all_embs, dim=0)
    all_lbls = torch.cat(all_lbls, dim=0)
    map_at_R = metrics.calculate_map_at_R(all_embs, all_lbls, R=499)
    return map_at_R
