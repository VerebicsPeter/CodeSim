# TODO: Parametrize data path
# TODO: Implement data aggregator and compute metrics hooks

from tqdm import tqdm
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, get_linear_schedule_with_warmup


from model.code_sim_models import (
    CodeSimLinearClassifierCross,
    CodeSimLinearClassifierSBert,
    CodeSimContrastiveEncoder,
    CodeSimilarityTrainer,
    defaultdict,
    aggr_data_classifier,
    aggr_data_contrastive_cls,
    aggr_data_contrastive_map,
)
import model.code_sim_models as code_sim_models
import model.code_sim_datasets as code_sim_datasets
import model.configs as configs
import model.metrics as metrics
from model.metrics import print_reports
from model.utils import set_seed


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NO_DECAY = ['bias', 'LayerNorm.weight']


def get_param_groups(model, wd):
    param_groups = [
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in NO_DECAY)],
            'weight_decay': 0.0
        },
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in NO_DECAY)],
            'weight_decay': wd
        },
    ]
    print("No decay:", NO_DECAY)
    return param_groups


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
    elif config.finetuning_strategy == "binary_cls_simpl":
        model_cls = code_sim_models.CodeSimLinearClassifierCross
        loss_func = nn.BCEWithLogitsLoss()
        loss_hook = code_sim_models.compute_loss_logit_Cross
    elif config.finetuning_strategy == "binary_cls_sbert":
        model_cls = code_sim_models.CodeSimLinearClassifierSBert
        loss_func = nn.BCEWithLogitsLoss()
        loss_hook = code_sim_models.compute_loss_logit_SBert

    # NOTE: strategy specifies encoding scheme
    return_single_encoding = config.finetuning_strategy == "binary_cls_simpl"
    # Dataset Creation
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
    tokenizer_args={
        "return_tensors": "pt",
        "padding": "max_length", "max_length": 512 if return_single_encoding else 256,
        "truncation": True,
    }
    
    train_data, valid_data, test_data = code_sim_datasets.Create_CodeNet_paired_dataset(
        tokenizer=tokenizer,
        tokenizer_args=tokenizer_args,
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
        config.pretrained_model,
        freeze_bert=config.freeze_model,
        pooling_strat=config.pooling_strat,
        dropout_rate=config.dropout_rate,
    )
    model.to(DEVICE)
    
    if torch.cuda.device_count() > 1:
        print("Wrapping model with DataParallel for multiple GPU usage.",
             f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = nn.DataParallel(model)
    
    # Trainer
    param_groups = get_param_groups(model, wd=config.wd_enc)
    optimizer = torch.optim.AdamW(param_groups, lr=config.lr_enc)
    scheduler = get_scheduler(train_loader, optimizer, config.epochs, config.iters_to_accumulate)
    
    trainer = CodeSimilarityTrainer(
        model,
        (train_loader, valid_loader),
        loss_func=loss_func,
        loss_hook=loss_hook,
        aggr_hook=aggr_data_classifier,
        compute_metrics=metrics.calculate_cls_metrics,
        target_metrics=["F1", "roc_auc"],
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        
    )
    trainer.train(epochs=config.epochs, iters_to_accumulate=config.iters_to_accumulate)
    
    print("Evaluating.")
    y_true, y_pred = eval_model_classifier(eval_data=test_loader, model=model)
    print_reports(y_true, y_pred)


def finetune_model_contrastive(config: configs.CodeSimContrastiveClassifierConfig):
    config.init_model()
    
    distance_function = lambda x, y: 1 - F.cosine_similarity(x, y)  # cosine distance
    
    if config.finetuning_strategy not in configs.CONTRASTIVE_FINETUNING_STRATEGIES:
        raise ValueError(f"Invalid finetuning strategy: {config.finetuning_strategy}.")
    elif config.finetuning_strategy == "triplet_loss":
        loss_func = nn.TripletMarginWithDistanceLoss(distance_function=distance_function, margin=config.margin)
        loss_hook = code_sim_models.compute_loss_triplet
    elif config.finetuning_strategy == "info_nce_loss":
        # TODO: implement InfoNCE loss as a custom loss function
        loss_func = None
        loss_hook = lambda trainer, batched_data: code_sim_models.compute_loss_tuplet(
            trainer, batched_data, config.temp
        )
    elif config.finetuning_strategy == "combined_loss":
        # TODO: implement combined loss as a custom loss function
        loss_func = nn.TripletMarginWithDistanceLoss(distance_function=distance_function, margin=config.margin)
        loss_hook = lambda trainer, batched_data: code_sim_models.compute_loss_combined(
            trainer, batched_data, config.temp, config.w_1, config.w_2
        )
    
    # Dataset Creation
    tokenizer=AutoTokenizer.from_pretrained(config.pretrained_model_name)
    tokenizer_args = {
        "return_tensors": "pt",
        "padding": "max_length", "max_length": 256,
        "truncation": True,
    }
    
    train_data, valid_data, test_data = code_sim_datasets.Create_CodeNet_triplet_dataset(
        tokenizer=tokenizer,
        tokenizer_args=tokenizer_args,
        num_negatives=config.num_negatives
    )
    
    train_loader = DataLoader(
        train_data, 
        batch_sampler=code_sim_datasets.RandomTripletBatchSampler(
            pids=train_data.problem_ids,
            num_batches=config.num_batches,
            num_pids_per_batch=config.bs
        ),
        collate_fn=code_sim_datasets.custom_collate_triplet
    )
    
    valid_loader = DataLoader(
        valid_data, batch_size=config.bs,
        sampler=code_sim_datasets.DefaultTripletSampler(valid_data.problem_ids),
        collate_fn=code_sim_datasets.custom_collate_triplet
    )
    
    test_loader = DataLoader(test_data, batch_size=config.bs)

    # Model Creation
    model = CodeSimContrastiveEncoder(
        config.pretrained_model,
        freeze_enc_model=config.freeze_model,
        pooling_strat=config.pooling_strat,
        dropout_rate=config.dropout_rate,
    )
    model.to(DEVICE)
    
    if torch.cuda.device_count() > 1:
        print("Wrapping model with DataParallel for multiple GPU usage.",
             f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = nn.DataParallel(model)
    
    # Trainer
    param_groups = get_param_groups(model, wd=config.wd_enc)
    optimizer = torch.optim.AdamW(param_groups, lr=config.lr_enc)
    scheduler = get_scheduler(train_loader, optimizer, config.epochs, config.iters_to_accumulate)
    
    trainer = CodeSimilarityTrainer(
        model,
        (train_loader, valid_loader),
        loss_func=loss_func,
        loss_hook=loss_hook,
        aggr_hook=aggr_data_contrastive_cls,
        compute_metrics=metrics.calculate_cls_metrics,
        target_metrics=["F1", "roc_auc"],
        loss_checkpointing=True,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
    )
    trainer.train(epochs=config.epochs, iters_to_accumulate=config.iters_to_accumulate)
    
    print("Evaluating.")
    y_true, y_pred = eval_model_contrastive_cls(eval_data=test_loader, model=trainer.model)
    print_reports(y_true, y_pred)


def finetune_model_on_POJ_104(config: configs.CodeSimContrastiveClassifierConfig):
    config.init_model()
    
    distance_function = lambda x, y: 1 - F.cosine_similarity(x, y)  # cosine distance
    
    if config.finetuning_strategy not in configs.CONTRASTIVE_FINETUNING_STRATEGIES:
        raise ValueError(f"Invalid finetuning strategy for POJ: {config.finetuning_strategy}")
    elif config.finetuning_strategy == "combined_loss":
        raise ValueError(f"Invalid finetuning strategy for POJ: {config.finetuning_strategy}")
    elif config.finetuning_strategy == "triplet_loss":
        loss_func = nn.TripletMarginWithDistanceLoss(distance_function=distance_function, margin=config.margin)
        loss_hook = code_sim_models.compute_loss_triplet
    elif config.finetuning_strategy == "info_nce_loss":
        # TODO: implement InfoNCE loss as a custom loss function
        loss_func = None
        loss_hook = partial(code_sim_models.compute_loss_tuplet, temp=config.temp)
    
    # Dataset Creation
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
    tokenizer_args = {
        "return_tensors": "pt",
        "padding": "max_length",
        "truncation": True,
    }
    
    train_data, valid_data, test_data = code_sim_datasets.Create_POJ104_triplet_dataset(
        tokenizer=tokenizer,
        tokenizer_args=tokenizer_args,
        sample_negative=config.finetuning_strategy == "triplet_loss",
    )
    
    train_loader = DataLoader(
        train_data,
        batch_sampler=code_sim_datasets.RandomTripletBatchSampler(
            pids=train_data.problem_ids,
            num_batches=config.num_batches,
            num_pids_per_batch=config.bs
        ),
        collate_fn=code_sim_datasets.custom_collate_POJpair
    )
    valid_loader = DataLoader(valid_data, batch_size=config.bs, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=config.bs, shuffle=False)
    
    # Model Creation
    model = CodeSimContrastiveEncoder(
        config.pretrained_model,
        freeze_enc_model=config.freeze_model,
        pooling_strat=config.pooling_strat,
        dropout_rate=config.dropout_rate,
    )
    model.to(DEVICE)
    
    if torch.cuda.device_count() > 1:
        print("Wrapping model with DataParallel for multiple GPU usage.",
             f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = nn.DataParallel(model)
    
    # Trainer
    param_groups = get_param_groups(model, wd=config.wd_enc)
    optimizer = torch.optim.AdamW(param_groups, lr=config.lr_enc)
    scheduler = get_scheduler(train_loader, optimizer, config.epochs, config.iters_to_accumulate)
    
    trainer = CodeSimilarityTrainer(
        model,
        (train_loader, valid_loader),
        loss_func=loss_func,
        loss_hook=loss_hook,
        aggr_hook=aggr_data_contrastive_map,
        compute_metrics=metrics.calculate_map_metrics,
        target_metrics=["map_r"],
        loss_checkpointing=False,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
    )
    trainer.train(epochs=config.epochs, iters_to_accumulate=config.iters_to_accumulate)
    
    print("Evaluating.")
    result = eval_model_contrastive_map(eval_data=test_loader, model=trainer.model)
    print(result)


def eval(model: CodeSimContrastiveEncoder 
              | CodeSimLinearClassifierCross 
              | CodeSimLinearClassifierSBert,
         num_rows=5000):
    set_seed(42)
    model.to(DEVICE)
    
    tokenizer=AutoTokenizer.from_pretrained(model.enc_model.name_or_path)
    
    if isinstance(model, CodeSimLinearClassifierCross):
        dataset = code_sim_datasets.Create_CodeNet_paired_dataset(
            tokenizer=tokenizer,
            tokenizer_args={
                "return_tensors": "pt",
                "padding": "max_length", "max_length": 512,
                "truncation": True,
            },
            return_single_encoding=True,
        )
        eval_func = eval_model_classifier
    elif isinstance(model, CodeSimLinearClassifierSBert):
        dataset = code_sim_datasets.Create_CodeNet_paired_dataset(
            tokenizer=tokenizer,
            tokenizer_args={
                "return_tensors": "pt",
                "padding": "max_length", "max_length": 256,
                "truncation": True,
            },
            return_single_encoding=False,
        )
        eval_func = eval_model_classifier
    elif isinstance(model, CodeSimContrastiveEncoder):
        print("Evaluating ", model.enc_model.name_or_path)
        dataset = code_sim_datasets.Create_CodeNet_triplet_dataset(
            tokenizer=tokenizer,
            tokenizer_args={
                "return_tensors": "pt",
                "padding": "max_length", "max_length": 256,
                "truncation": True,
            },
            num_negatives=1,
        )
        eval_func = eval_model_contrastive_cls
    else:
        raise ValueError(f"Invalid model type. {model.__class__.__name__}")

    _,_, test_loader = code_sim_datasets.get_loaders(*dataset, bs=20, shuffle=False, num_rows=num_rows)
    
    y_true, y_pred = eval_func(eval_data=test_loader, model=model)
    print_reports(y_true, y_pred)
    return y_true, y_pred


@torch.no_grad
def eval_model_classifier(eval_data: DataLoader, model: CodeSimLinearClassifierCross | CodeSimLinearClassifierSBert):
    model.eval()
    aggr_data = defaultdict(list)
    for data in tqdm(eval_data):
        aggr_data_classifier(model, data, aggr_data)
    y_true = aggr_data['y_true']
    y_pred = aggr_data['y_pred']
    return y_true, y_pred


@torch.no_grad
def eval_model_contrastive_cls(eval_data: DataLoader, model: CodeSimContrastiveEncoder):
    model.eval()
    aggr_data = defaultdict(list)
    for data in tqdm(eval_data):
        aggr_data_contrastive_cls(model, data, aggr_data)
    y_true = aggr_data['y_true']
    y_pred = aggr_data['y_pred']
    return y_true, y_pred


@torch.no_grad
def eval_model_contrastive_map(eval_data: DataLoader, model: CodeSimContrastiveEncoder):
    model.eval()
    aggr_data = defaultdict(list)
    for data in tqdm(eval_data):
        aggr_data_contrastive_map(model, data, aggr_data)
    all_embs = torch.cat(aggr_data["all_embs"], dim=0)
    all_lbls = torch.cat(aggr_data["all_lbls"], dim=0)
    result = metrics.calculate_map_at_R(all_embs, all_lbls, R=499)
    return result
