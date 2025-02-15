import argparse
import random
import numpy as np
import pandas as pd
import pprint as pp

import torch
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm

import transformers

from sklearn.metrics import classification_report

import model.code_sim_models as code_sim_models
import model.code_sim_datasets as code_sim_datasets


MODEL_TYPES = {
    "contrative": code_sim_models.ContrastiveCodeSimilarityModel,
    "finetuned":  code_sim_models.FinetunedCodeSimilarityModel,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


paired_dataset_url = "https://drive.google.com/uc?export=download&id=1pUErbyZw1fBC5gIe6KT7BWga7h6Bfr4l"


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def evaluate_preds(
    eval_df: pd.DataFrame,
    model_type: str,
    model_path: str,
    pretrained_bert_name: str,
    threshold = 0.5,
):
    class DataframeWrapper(Dataset):
        def __init__(self, df):
            self.df = df
            
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            return row['src_1'], row['src_2'], row['label']
    
    
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Invalid model type: {model_type}.")
    
    cls = MODEL_TYPES[model_type]

    bert = transformers.AutoModel.from_pretrained(pretrained_bert_name).to(DEVICE)
    
    # Create and load the model
    model: code_sim_models.SimilarityClassifier
    model = cls(bert).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    dataset = DataframeWrapper(eval_df)

    true_labels, predictions = [], []
    with torch.no_grad():
        for (srcs1, srcs2, labels) in tqdm(DataLoader(dataset, batch_size=20)):
            preds = model.predict(srcs1, srcs2, threshold=threshold)
            # Store predictions and labels
            true_labels.extend(labels.cpu().tolist())
            predictions.extend(preds.cpu().tolist())
    torch.cuda.empty_cache()
    return true_labels, predictions


if __name__ == "__main__":
    # TODO: Maybe load this from a .env or something
    set_seed(42)
    
    # NOTE: This is a dummy dataframe for now, a evaluation dataframe is still needed...
    df = pd.read_csv(
        paired_dataset_url,
        header=0,
        names=code_sim_datasets.CodeNetPairDataset.COLUMNS
    )
    df_downsampled = code_sim_datasets.downsample_df(df, samples_per_class=25_000, seed=42)
    df_not_sampled = df[~df.index.isin(df_downsampled.index)]
    df_evaluation  = code_sim_datasets.downsample_df(df_not_sampled, samples_per_class=500, seed=420)
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        choices=MODEL_TYPES.keys(),
        required=True, help="Model type to evaluate"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True, help="Model state dict path"
    )
    parser.add_argument(
        "--pretrained_bert_name",
        type=str,
        required=False,
        default="huggingface/CodeBERTa-small-v1"
    )
    args = parser.parse_args()
    
    ls, ps = evaluate_preds(
        df_evaluation,
        model_type=args.model_type,
        model_path=args.model_path,
        pretrained_bert_name=args.pretrained_bert_name,
    )
    report = classification_report(ls, ps)
    print(report)
