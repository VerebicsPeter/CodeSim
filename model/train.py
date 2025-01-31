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

# 40k dataset
#paired_dataset_url = "https://drive.google.com/uc?export=download&id=1fT4aEyIRzlqDcMFyyKkxgYN3-TGB5qqu"
# 100k dataset (TODO: train on this)
paired_dataset_url = "https://drive.google.com/uc?export=download&id=1pUErbyZw1fBC5gIe6KT7BWga7h6Bfr4l"

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


def train_finetuned(
    pretrained_bert_name: str = "huggingface/CodeBERTa-small-v1",
    epochs = 4,
    lr = 1e-5,  # Learning rate
    wd = 1e-5,  # Weight decay
    bs = 20,    # Batch size
    # The gradient accumulation adds gradients over an effective batch of size : bs * iters_to_accumulate.
    # If set to "1", you get the usual batch size
    iters_to_accumulate = 2,
    # Model specific parameters
    freeze_bert  = False,  # NOTE: if true the BERT model is not finetuned
    dropout_rate = 0.2,
    
):
    bert_tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_name)
    
    df = pd.read_csv(
        paired_dataset_url,
        header=0,
        names=code_sim_datasets.CodeNetPairDataset.COLUMNS
    )
    df = df.drop(columns=['pid', 'sid_1', 'sid_2'])
    print(df['label'].value_counts())
    
    pp.pp(df.head())
    
    dataset = code_sim_datasets.CodeNetPairDataset.from_pandas_df(
        df,
        tokenizer=bert_tokenizer,
        num_pairs=50_000,
    )
    
    shuffle = True
    train_len = int(0.8 * len(dataset))
    valid_len = len(dataset) - train_len
    train_data, valid_data = random_split(dataset, [train_len, valid_len])
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=shuffle)
    valid_loader = DataLoader(valid_data, batch_size=bs, shuffle=shuffle)
    
    bert_model = AutoModel.from_pretrained(pretrained_bert_name).to(DEVICE)
    
    model = code_sim_models.BERTCodeSimilarityModel(
        bert_model,
        freeze_bert =freeze_bert,
        dropout_rate=dropout_rate,
    )
    model.to(DEVICE)
    
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        # The number of steps for the warmup phase.
        num_warmup_steps=0,
        # Necessary to take into account Gradient accumulation
        num_training_steps=(epochs * len(train_loader)) // iters_to_accumulate
    )
    
    trainer = code_sim_models.BERTCodeSimilarityTrainer(
        model,
        (train_loader, valid_loader),
        loss_func=loss_func,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
    )
    trainer.train(epochs=epochs)


def train_contrastive(
    pretrained_bert_name: str = "huggingface/CodeBERTa-small-v1",
    epochs = 4,
    lr = 1e-5,  # Learning rate
    wd = 1e-5,  # Weight decay
    bs = 20,  # NOTE: Bigger batch size generally leads to better results in contrastive learning
    is_self_supervised = False,
):
    # Augment function
    def minify(code: str) -> str:
        try: return python_minifier.minify(code)
        except Exception as error:
            print(f'Error while minifying: {error}')
        return code

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
        aug_funcs=[minify],
        device=DEVICE,
    )
    
    sample_size = 25_000
    dataset = Subset(dataset, list(range(sample_size)))
    
    shuffle = False
    train_len = int(0.8 * len(dataset))
    valid_len = len(dataset) - train_len
    train_data, valid_data = random_split(dataset, [train_len, valid_len])
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=shuffle)
    valid_loader = DataLoader(valid_data, batch_size=bs, shuffle=shuffle)
    
    transformer = AutoModel.from_pretrained(pretrained_bert_name).to(DEVICE)
    transformer.eval()
    
    embedding_pipeline = code_sim_models.create_embedding_pipeline(transformer)
    
    model = code_sim_models.ContrastiveCodeSimilarityModel(embedding_pipeline).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    
    ntxent_loss = losses.NTXentLoss(temperature=0.5)
    # Wrap the NTXent loss function if needed
    ntxent_loss = losses.SelfSupervisedLoss(ntxent_loss) if is_self_supervised else ntxent_loss
    
    trainer = code_sim_models.ContrastiveCodeSimilarityTrainer(
        model,
        (train_loader, valid_loader),
        loss_func=ntxent_loss,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    trainer.train(epochs=epochs)


TRAIN_FUNCTIONS = {
    "finetuned": (train_finetuned,
    {
        "pretrained_bert_name": "huggingface/CodeBERTa-small-v1",
        "epochs": 4,
        "lr": 1e-5,  # Learning rate
        "wd": 1e-5,  # Weight decay
        "bs": 20,    # Batch size
        # The gradient accumulation adds gradients over an effective batch of size : bs * iters_to_accumulate.
        # If set to "1", you get the usual batch size
        "iters_to_accumulate": 2,
        # Model specific parameters
        "freeze_bert": False,  # NOTE: if true the BERT model is not finetuned
        "dropout_rate": 0.2,
    }),
    "contrastive": (train_contrastive,
    {
        "pretrained_bert_name": "huggingface/CodeBERTa-small-v1",
        "epochs": 4,
        "lr": 1e-5,  # Learning rate
        "wd": 1e-5,  # Weight decay
        "bs": 20,  # NOTE: Bigger batch size generally leads to better results in contrastive learning
        "is_self_supervised": False,
    }),
}


if __name__ == "__main__":
    set_seed(42)
    
    # Parse the model type first
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=TRAIN_FUNCTIONS.keys(), required=True, help="Model type to train")
    known_args, unknown_args = parser.parse_known_args()
    # Select the train function and default parameters
    train_func, default_params = TRAIN_FUNCTIONS[known_args.model]
    
    # Parse the rest of the parameters based on default ones
    parser = argparse.ArgumentParser()
    for param, default in default_params.items():
        parser.add_argument(f"--{param}", type=type(default), default=default)
    args = parser.parse_args(unknown_args)
    
    # Train the model
    train_func(**vars(args))
