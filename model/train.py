from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, get_linear_schedule_with_warmup


from model.code_sim_models import (
    CodeSimLinearClassifierCross,
    CodeSimLinearClassifierSBert,
    CodeSimContrastiveEncoder,
    CodeSimTrainer,
    ContrastiveLoss,
    defaultdict,
    EvalDataWrapper,
    pass_data_classifier,
    aggr_data_classifier,
    embd_data_contrastive_cls,
    aggr_data_contrastive_cls,
    embd_data_contrastive_map,
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

# Cosine Distance
distance_function = lambda x, y: 1 - F.cosine_similarity(x, y)


def get_tokenizer_args(max_length = None) -> dict:
    args = {
        "return_tensors": "pt",
        "truncation": True,
        "padding": "max_length",
    }
    if max_length: args["max_length"] = max_length
    return args


def get_param_groups(model: nn.Module, wd):
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


def get_optimizer(model: nn.Module, lr, wd):
    param_groups = get_param_groups(model, wd=wd)
    optimizer = torch.optim.AdamW(param_groups, lr=lr)
    return optimizer


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


def finetune_classifier_model(
    config: configs.CodeSimClassifierConfig,
    data_path="peterverebics/CodeNet_Python_118"
):
    config.init_model()
    
    if config.finetuning_strategy not in configs.FINETUNING_STRATEGIES:
        raise ValueError(f"Invalid finetuning strategy: {config.finetuning_strategy}.")
    elif config.finetuning_strategy == "binary_cls_cross":
        model_cls = code_sim_models.CodeSimLinearClassifierCross
        loss_func = nn.BCEWithLogitsLoss()
        loss_hook = code_sim_models.compute_loss_logit_Cross
    elif config.finetuning_strategy == "binary_cls_sbert":
        model_cls = code_sim_models.CodeSimLinearClassifierSBert
        loss_func = nn.BCEWithLogitsLoss()
        loss_hook = code_sim_models.compute_loss_logit_SBert

    # NOTE: strategy specifies encoding scheme
    return_single_encoding = config.finetuning_strategy == "binary_cls_cross"
    
    # Dataset Creation
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
    tokenizer_args=get_tokenizer_args(512 if return_single_encoding else 256)
    
    train_data, valid_data, test_data = code_sim_datasets.Create_CodeNet_paired_dataset(
        tokenizer, tokenizer_args, data_path=data_path,
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
    
    # Trainer
    optimizer = get_optimizer(model, config.lr_enc, config.wd_enc)
    scheduler = get_scheduler(train_loader, optimizer, config.epochs, config.iters_to_accumulate)
    
    def aggr_data_loss(data, aggr_data: defaultdict[str, list]):
        aggr_data["losses"].append(loss_func(data[0].squeeze(-1), data[1].float()).item())
    
    trainer = CodeSimTrainer(
        model,
        train_loader=train_loader,
        valid_loader_wrappers={
            "CodeNet":
            EvalDataWrapper(
                valid_loader,
                data_embedder=pass_data_classifier,
                aggr_hooks={
                    "loss":
                        aggr_data_loss,
                    "F1":
                        aggr_data_classifier,
                    "roc_auc":
                        aggr_data_classifier,
                },
                metr_hooks={
                    "loss":
                        lambda **kwargs: sum(kwargs["losses"])/len(kwargs["losses"]),
                    "F1":
                        lambda **kwargs: metrics.calculate_cls_metrics(**kwargs)["F1"],
                    "roc_auc":
                        lambda **kwargs: metrics.calculate_cls_metrics(**kwargs)["roc_auc"],
                },
            ),
        },
        loss_func=loss_func,
        loss_hook=loss_hook,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
    )
    trainer.train(epochs=config.epochs, iters_to_accumulate=config.iters_to_accumulate)
    
    print("Evaluating.")
    y_true, y_pred = eval_classifier_model(eval_data=test_loader, model=model)
    print_reports(y_true, y_pred)


def finetune_contrastive_model_on_CodeNet(
    config: configs.CodeSimContrastiveClassifierConfig,
    data_path="peterverebics/CodeNet_Python_118"
):
    config.init_model()
    
    if config.finetuning_strategy not in configs.CONTRASTIVE_FINETUNING_STRATEGIES:
        raise ValueError(f"Invalid finetuning strategy: {config.finetuning_strategy}.")
    elif config.finetuning_strategy == "triplet_loss":
        loss_func = nn.TripletMarginWithDistanceLoss(distance_function=distance_function, margin=config.margin)
        loss_hook = code_sim_models.compute_loss_triplet
    elif config.finetuning_strategy == "info_nce_loss":
        loss_func = ContrastiveLoss(temp=config.temp)
        loss_hook = code_sim_models.compute_loss_tuplet
    elif config.finetuning_strategy == "combined_loss":
        local_loss_func = nn.TripletMarginWithDistanceLoss(distance_function=distance_function, margin=config.margin)
        loss_func = ContrastiveLoss(local_loss_func=local_loss_func,
                                    temp=config.temp, w1=config.w_1, w2=config.w_2)
        
        loss_hook = code_sim_models.compute_loss_combined
    
    # Dataset Creation
    tokenizer=AutoTokenizer.from_pretrained(config.pretrained_model_name)
    tokenizer_args = get_tokenizer_args(256)
    
    train_data, valid_data, test_data = code_sim_datasets.Create_CodeNet_triplet_dataset(
        tokenizer, tokenizer_args, data_path=data_path,
        num_negatives=config.num_negatives
    )
    
    train_loader, valid_loader, test_loader = code_sim_datasets.get_CodeNet_loaders(
        train_data, valid_data, test_data, config
    )

    # Model Creation
    model = CodeSimContrastiveEncoder(
        config.pretrained_model,
        freeze_enc_model=config.freeze_model,
        pooling_strat=config.pooling_strat,
        dropout_rate=config.dropout_rate,
    )
    model.to(DEVICE)
    
    # Trainer
    optimizer = get_optimizer(model, config.lr_enc, config.wd_enc)
    scheduler = get_scheduler(train_loader, optimizer, config.epochs, config.iters_to_accumulate)
    
    def aggr_data_loss(data, aggr_data: defaultdict[str, list]):
        aggr_data["losses"].append(loss_func(*data).item())
        
    trainer = CodeSimTrainer(
        model,
        train_loader=train_loader,
        valid_loader_wrappers={
            "CodeNet":
            EvalDataWrapper(
                valid_loader,
                data_embedder=embd_data_contrastive_cls,
                aggr_hooks={
                    "loss":
                        aggr_data_loss,
                    "F1":
                        aggr_data_contrastive_cls,
                    "roc_auc":
                        aggr_data_contrastive_cls,
                },
                metr_hooks={
                    "loss":
                        lambda **kwargs: sum(kwargs["losses"])/len(kwargs["losses"]),
                    "F1":
                        lambda **kwargs: metrics.calculate_cls_metrics(**kwargs)["F1"],
                    "roc_auc":
                        lambda **kwargs: metrics.calculate_cls_metrics(**kwargs)["roc_auc"],
                },
            ),
        },
        loss_func=loss_func,
        loss_hook=loss_hook,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
    )
    trainer.train(epochs=config.epochs, iters_to_accumulate=config.iters_to_accumulate)
    
    print("Evaluating.")
    y_true, y_pred = eval_contrastive_model_cls(eval_data=test_loader, model=trainer.model)
    print_reports(y_true, y_pred)


def finetune_contrastive_model_on_POJ_104(
    config: configs.CodeSimContrastiveClassifierConfig,
    data_path="semeru/Code-Code-CloneDetection-POJ104"
):
    config.init_model()
    
    if config.finetuning_strategy not in configs.CONTRASTIVE_FINETUNING_STRATEGIES:
        raise ValueError(f"Invalid finetuning strategy for POJ: {config.finetuning_strategy}")
    elif config.finetuning_strategy == "combined_loss":
        raise ValueError(f"Invalid finetuning strategy for POJ: {config.finetuning_strategy}")
    elif config.finetuning_strategy == "triplet_loss":
        loss_func = nn.TripletMarginWithDistanceLoss(distance_function=distance_function, margin=config.margin)
        loss_hook = code_sim_models.compute_loss_triplet
    elif config.finetuning_strategy == "info_nce_loss":
        loss_func = ContrastiveLoss(temp=config.temp)
        loss_hook = code_sim_models.compute_loss_tuplet
    
    # Dataset Creation
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
    tokenizer_args = get_tokenizer_args()
    
    train_data, valid_data, test_data = code_sim_datasets.Create_POJ104_triplet_dataset(
        tokenizer, tokenizer_args, data_path=data_path,
        sample_negative=config.finetuning_strategy == "triplet_loss",
    )
    
    train_loader, valid_loader, test_loader = code_sim_datasets.get_POJ104_loaders(
        train_data, valid_data, test_data, config
    )
    
    # Model Creation
    model = CodeSimContrastiveEncoder(
        config.pretrained_model,
        freeze_enc_model=config.freeze_model,
        pooling_strat=config.pooling_strat,
        dropout_rate=config.dropout_rate,
    )
    model.to(DEVICE)
    
    # Trainer
    optimizer = get_optimizer(model, config.lr_enc, config.wd_enc)
    scheduler = get_scheduler(train_loader, optimizer, config.epochs, config.iters_to_accumulate)
    
    def aggr_data_loss(data, aggr_data: defaultdict[str, list]):
        aggr_data["losses"].append(loss_func(*data).item())
    
    trainer = CodeSimTrainer(
        model,
        train_loader=train_loader,
        valid_loader_wrappers={
            "POJ-104":
            EvalDataWrapper(
                valid_loader,
                data_embedder=embd_data_contrastive_map,
                aggr_hooks={
                    "loss":
                        aggr_data_loss,
                    "map_r":
                        aggr_data_contrastive_map,
                },
                metr_hooks={
                    "loss":
                        lambda **kwargs: sum(kwargs["losses"])/len(kwargs["losses"]),
                    "map_r":
                        lambda **kwargs: metrics.calculate_map_metrics(**kwargs)["map_r"],
                },
            ),
        },
        loss_func=loss_func,
        loss_hook=loss_hook,
        loss_checkpointing=False,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
    )
    trainer.train(epochs=config.epochs, iters_to_accumulate=config.iters_to_accumulate)
    
    print("Evaluating.")
    result = eval_contrastive_model_map(eval_data=test_loader, model=trainer.model)
    print(result)


def eval(model: CodeSimContrastiveEncoder | CodeSimLinearClassifierCross | CodeSimLinearClassifierSBert,
         data_path="peterverebics/CodeNet_Python_118", num_rows=None):
    set_seed(42)
    model.to(DEVICE)
    
    tokenizer=AutoTokenizer.from_pretrained(model.enc_model.name_or_path)
    
    if isinstance(model, CodeSimLinearClassifierCross):
        dataset = code_sim_datasets.Create_CodeNet_paired_dataset(
            tokenizer=tokenizer,
            tokenizer_args=get_tokenizer_args(512),
            data_path=data_path,
            return_single_encoding=True,
        )
        eval_func = eval_classifier_model
    elif isinstance(model, CodeSimLinearClassifierSBert):
        dataset = code_sim_datasets.Create_CodeNet_paired_dataset(
            tokenizer=tokenizer,
            tokenizer_args=get_tokenizer_args(256),
            data_path=data_path,
            return_single_encoding=False,
        )
        eval_func = eval_classifier_model
    elif isinstance(model, CodeSimContrastiveEncoder):
        print("Evaluating ", model.enc_model.name_or_path)
        dataset = code_sim_datasets.Create_CodeNet_triplet_dataset(
            tokenizer=tokenizer,
            tokenizer_args=get_tokenizer_args(256),
            data_path=data_path,
            num_negatives=1,
        )
        eval_func = eval_contrastive_model_cls
    else:
        raise ValueError(f"Invalid model type. {model.__class__.__name__}")

    _,_, test_loader = code_sim_datasets.get_loaders(*dataset, bs=20, shuffle=False, num_rows=num_rows)
    
    y_true, y_pred = eval_func(eval_data=test_loader, model=model)
    print_reports(y_true, y_pred)
    return y_true, y_pred


@torch.no_grad
def eval_classifier_model(eval_data: DataLoader, model: CodeSimLinearClassifierCross | CodeSimLinearClassifierSBert):
    model.eval()
    aggr_data = defaultdict(list)
    for data in tqdm(eval_data):
        data = pass_data_classifier(data, model, DEVICE)
        aggr_data_classifier(data, aggr_data)
    y_true = aggr_data['y_true']
    y_pred = aggr_data['y_pred']
    return y_true, y_pred


@torch.no_grad
def eval_contrastive_model_cls(eval_data: DataLoader, model: CodeSimContrastiveEncoder):
    model.eval()
    aggr_data = defaultdict(list)
    for data in tqdm(eval_data):
        emb_data = embd_data_contrastive_cls(data, model, DEVICE)
        aggr_data_contrastive_cls(emb_data, aggr_data)
    y_true = aggr_data['y_true']
    y_pred = aggr_data['y_pred']
    return y_true, y_pred


@torch.no_grad
def eval_contrastive_model_map(eval_data: DataLoader, model: CodeSimContrastiveEncoder):
    model.eval()
    aggr_data = defaultdict(list)
    for data in tqdm(eval_data):
        emb_data = embd_data_contrastive_map(data, model, DEVICE)
        aggr_data_contrastive_map(emb_data, aggr_data)
    all_embs = torch.cat(aggr_data["all_embs"], dim=0)
    all_lbls = torch.cat(aggr_data["all_lbls"], dim=0)
    result = metrics.calculate_map_at_R(all_embs, all_lbls, R=499)
    return result
