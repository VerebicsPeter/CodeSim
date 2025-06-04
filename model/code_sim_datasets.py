# Dataset wrappers for CodeNet data
import gdown
import pandas as pd
import pprint as pp
import random
#import datasets
import transformers
import torch
from torch.utils.data import Dataset
from typing import Iterable
from collections import defaultdict


DATASET_TYPE = {
    "paired",
    "triplet",
}

# TODO: Maybe load URLS from a .env or something
# TODO: Create proper dataset with train, validation, evalualion splits for clean evaluation,
# idea: pick a set of 'evaluation' problems distinct from training and validation problems
COLUMNS = ["problem_id", "submission_id", "status", "code"]
NEW_DATASET_URL_SMALL = "https://drive.google.com/uc?export=download&id=1hZ4_QjTcmesQYsPo75G1lyIe__HFfwbG"
NEW_DATASET_URL_LARGE = "https://drive.google.com/uc?export=download&id=10ok2e2BmWRVhn_V6tRBZzLfeYQ0eaJA2"
DATASET_URL = NEW_DATASET_URL_LARGE

# NOTE: Therse are old datasets
DATASET_URLS = {
    "paired" : "https://drive.google.com/uc?export=download&id=1pUErbyZw1fBC5gIe6KT7BWga7h6Bfr4l",
    "triplet": "https://drive.google.com/uc?export=download&id=11aBIxIMEMKoGyJ9IdUHY2XQv1ZzfyXd2",
}


def download_dataset(url, output_file):
    gdown.download(url, output_file, quiet=False)


def get_batch_encodings(
    codes: Iterable[str],
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
    device: str = "cpu",
) -> transformers.BatchEncoding:
    MODEL_MAX_LEN = tokenizer.model_max_length

    inputs = tokenizer(
        codes,
        truncation=True,
        # Pad to "MAX_LEN + 1" to detect sequences that are too long
        padding="max_length",
        max_length=MODEL_MAX_LEN + 1,
        return_tensors="pt",
    )

    # Mask out sequences that are longer than `MODEL_MAX_LEN`
    l_mask = inputs["attention_mask"].sum(dim=1) <= MODEL_MAX_LEN
    inputs = {k: v[l_mask, :MODEL_MAX_LEN] for k, v in inputs.items()}
    # Move tensors to the specified device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs

# TODO: augment pairs by flipping the order of the pair
class CodeNetPairDataset(Dataset):
    """Dataset for BERT encodings from CodeNet code pairs."""

    def __init__(
        self,
        pids,
        pairs,
        labels,
        tokenizer: (
            transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast
        ),
        tokenizer_max_length: int = 256,
        return_single_encoding: bool = True,
    ):
        super().__init__()
        
        assert len(pids) == len(pairs) == len(labels), "Length MUST match!"
        self.pids = pids
        
        self.pid_to_idx = defaultdict(list)
        for idx, pid in enumerate(pids): self.pid_to_idx[pid].append(idx)
        
        self.pairs = pairs
        self.labels = labels
        
        self.tokenizer = tokenizer
        self.tokenizer_params = {
            "padding":'max_length',  # Pad to max_length
            "max_length": tokenizer_max_length,
            "truncation":True,       # Truncate to max_length
            "return_tensors":'pt'    # Return torch.Tensor objects
        }
        self.return_single_encoding = return_single_encoding
        
        self.encoded_pairs = [self._encode_pair(pair) for pair in pairs]
        

    def _encode_pair(self, pair):
        code_a, code_b = pair
        if self.return_single_encoding:
            # Encode the sequences for sequence pair classification
            # ([CLS], code_a tokens , [SEP], code_b tokens, [SEP])
            encoding = self.tokenizer(code_a, code_b, **self.tokenizer_params)
            return {k: v.squeeze(0) for k, v in encoding.items()}
        else:
            encodings = self.tokenizer([code_a, code_b], **self.tokenizer_params)
            return (
                {k: v[0] for k,v in encodings.items()},
                {k: v[1] for k,v in encodings.items()}
            )
    
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        
        encoding = self.encoded_pairs[idx]
        if self.return_single_encoding:
            return  encoding, label
        else:
            return *encoding, label

    def __len__(self):
        return len(self.labels)

    @classmethod
    def from_pandas_df(
        cls,
        df: pd.DataFrame,
        tokenizer,
        tokenizer_max_length: int = 256,
        return_single_encoding: bool = True,
    ):
        pids, pairs, labels = [], [], []
        
        for pid, group_df in df.groupby("problem_id"):
            assert (group_df[:200]["status"] == "Accepted").all()
            assert (group_df[200:]["status"] != "Accepted").all()
            
            anchors = group_df[:100]["code"].to_list()
            positives = group_df[100:200]["code"].to_list()
            negatives = group_df[200:300]["code"].to_list()
            
            assert len(anchors) == len(positives) == len(negatives), "Lengths MUST match!"
            
            pids.extend([pid] * len(anchors)*2)
            
            pairs.extend([*zip(anchors, positives), *zip(anchors, negatives)])
            
            labels.extend([1] * len(anchors) + [0] * len(anchors))
        
        return cls(pids, pairs, labels, tokenizer, tokenizer_max_length, return_single_encoding)


class CodeNetTripletDataset(Dataset):
    """Dataset for BERT encodings from CodeNet code triplets."""

    def __init__(
        self,
        pids,
        triplets,
        tokenizer: (
            transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast
        ),
        tokenizer_max_length: int = 256,
    ):
        super().__init__()
        
        assert len(pids) == len(triplets), "Length MUST match!"
        self.pids = pids
        
        self.pid_to_idx = defaultdict(list)
        for idx, pid in enumerate(pids): self.pid_to_idx[pid].append(idx)
        
        self.triplets = triplets
        
        self.tokenizer = tokenizer
        self.tokenizer_params = {
            "padding": "max_length",  # Pad to max_length
            "max_length": tokenizer_max_length,
            "truncation": True,  # Truncate to max_length
            "return_tensors": "pt",  # Return torch.Tensor objects
        }
        
        self.encoded_triplets = [self._encode_triplet(triplet) for triplet in triplets]
    
    
    def _encode_triplet(self, triplet):
        code_a, code_p, code_n = triplet
        encodings = self.tokenizer([code_a, code_p, code_n], **self.tokenizer_params)
        return (
            {k: v[0] for k,v in encodings.items()},
            {k: v[1] for k,v in encodings.items()},
            {k: v[2] for k,v in encodings.items()}
        )

    def __getitem__(self, idx):
        # Return the tokenized encodings
        encodings = self.encoded_triplets[idx]
        return encodings

    def __len__(self):
        return len(self.triplets)

    @classmethod
    def from_pandas_df(
        cls,
        df: pd.DataFrame,
        tokenizer,
        tokenizer_max_length: int = 256,
    ):
        pids, triplets = [], []
        
        for pid, group_df in df.groupby("problem_id"):
            assert (group_df[:200]["status"] == "Accepted").all()
            assert (group_df[200:]["status"] != "Accepted").all()
            
            anchors = group_df[:100]["code"].to_list()
            positives = group_df[100:200]["code"].to_list()
            negatives = group_df[200:300]["code"].to_list()
            
            assert len(anchors) == len(positives) == len(negatives), "Lengths MUST match!"
            
            pids.extend([pid] * len(anchors))
            triplets.extend(zip(anchors, positives, negatives))

        return cls(pids, triplets, tokenizer, tokenizer_max_length)


class CodeNetRandomTripletDataset(Dataset):
    """Dataset for BERT encodings from CodeNet code triplets."""

    def __init__(
        self,
        pids,
        positives,
        negatives,
        pid_to_pos,
        pid_to_neg,
        tokenizer: (
            transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast
        ),
        tokenizer_max_length: int = 256,
    ):
        super().__init__()
        
        assert len(pids) == len(positives), "Length MUST match!"
        self.pids = pids
        self.positives = positives
        self.negatives = negatives
        self.pid_to_pos = pid_to_pos
        self.pid_to_neg = pid_to_neg
        
        self.tokenizer = tokenizer
        self.tokenizer_params = {
            "padding": "max_length",  # Pad to max_length
            "max_length": tokenizer_max_length,
            "truncation": True,  # Truncate to max_length
            "return_tensors": "pt",  # Return torch.Tensor objects
        }

    def __getitem__(self, idx):
        pid = self.pids[idx]
        anchor = self.positives[idx]
        positive = random.choice(self.pid_to_pos[pid])
        negative = random.choice(self.pid_to_neg[pid])
        encodings = self.tokenizer([anchor, positive, negative], **self.tokenizer_params)
        # Remove batch dimensions
        enc_a = {k: v[0] for k, v in encodings.items()}
        enc_p = {k: v[1] for k, v in encodings.items()}
        enc_n = {k: v[2] for k, v in encodings.items()}
        return enc_a, enc_p, enc_n

    def __len__(self):
        return len(self.positives)

    @classmethod
    def from_pandas_df(
        cls,
        df: pd.DataFrame,
        tokenizer,
        tokenizer_max_length: int = 256,
    ):
        pids, pos, neg = [], [], []
        pid_to_pos = {}
        pid_to_neg = {}
        
        for pid, group_df in df.groupby("problem_id"):
            assert (group_df[:200]["status"] == "Accepted").all()
            assert (group_df[200:]["status"] != "Accepted").all()
            positives = group_df[:200]["code"].to_list()
            negatives = group_df[200:]["code"].to_list()
            pos.extend(positives)
            neg.extend(negatives)
            pids.extend([pid] * len(positives))
            pid_to_pos[pid] = positives
            pid_to_neg[pid] = negatives

        return cls(pids, pos, neg, pid_to_pos, pid_to_neg, tokenizer, tokenizer_max_length)


def Create_CodeNet_paired_dataset(
    tokenizer,
    tokenizer_max_length=256,
    data_path=DATASET_URL,
    return_single_encoding=True,
):
    download_dataset(data_path, "dataset.csv")
    df = pd.read_csv("dataset.csv", header=0, names=COLUMNS)
    #df = pd.read_csv(DATASET_URL, header=0, names=COLUMNS)  # FOR TESTING
    print("CodeNet data loaded. Data type: paired")
    pp.pp(df)
    
    dataset = CodeNetPairDataset.from_pandas_df(
        df,
        tokenizer=tokenizer,
        tokenizer_max_length=tokenizer_max_length,
        return_single_encoding=return_single_encoding,
    )
    return dataset


def Create_CodeNet_triplet_dataset(
    tokenizer,
    tokenizer_max_length=256,
    data_path=DATASET_URL,
):
    download_dataset(data_path, "dataset.csv")
    df = pd.read_csv("dataset.csv", header=0, names=COLUMNS)
    #df = pd.read_csv(DATASET_URL, header=0, names=COLUMNS)  # FOR TESTING 
    print("CodeNet data loaded. Data type: triplet")
    pp.pp(df)

    dataset = CodeNetRandomTripletDataset.from_pandas_df(
        df,
        tokenizer=tokenizer,
        tokenizer_max_length=tokenizer_max_length,
    )
    return dataset


class POJDataset(Dataset):
    """Simple wrapper for the dataset 'semeru/Code-Code-CloneDetection-POJ104'"""
    
    def __init__(self, poj_dataset, tokenizer: (transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast)):
        self.tokenizer = tokenizer
        self.tokenizer_params = {
            "padding": "max_length",  # Pad to max_length
            "max_length": self.tokenizer.model_max_length,
            "truncation": True,  # Truncate to max_length
            "return_tensors": "pt",  # Return torch.Tensor objects
        }
        self.dataset = poj_dataset

    def __getitem__(self, idx):
        item = self.dataset[idx]
        enc = self.tokenizer(item["code"], **self.tokenizer_params)
        enc = {k: v.squeeze(0) for k, v in enc.items()}  # remove batch dim
        lbl = torch.tensor(int(item["label"])).long()
        return enc, lbl

    def __len__(self):
        return len(self.dataset)


class POJ104TripletDataset(Dataset):
    """Simple wrapper for sampling triplets from the dataset 'semeru/Code-Code-CloneDetection-POJ104'"""
    
    def __init__(self, poj_dataset, tokenizer: (transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast)):
        self.dataset = poj_dataset
        self.lbl_to_idx = defaultdict(list)
        for idx, item in enumerate(poj_dataset):
            self.lbl_to_idx[item["label"]].append(idx)
        self.labels = list(self.lbl_to_idx.keys())
        self.tokenizer = tokenizer
        self.tokenizer_params = {
            "padding": "max_length",  # Pad to max_length
            "max_length": self.tokenizer.model_max_length,
            "truncation": True,  # Truncate to max_length
            "return_tensors": "pt",  # Return torch.Tensor objects
        }

    # TODO: maybe seed this explicitly...
    def __getitem__(self, idx):
        anchor = self.dataset[idx]

        # Positive sample
        pos_label = anchor["label"]
        pos_index = idx
        pos_indices = self.lbl_to_idx[pos_label]
        while pos_index == idx: pos_index = random.choice(pos_indices)
        positive = self.dataset[pos_index]

        # Negative sample
        neg_label = random.choice([lbl for lbl in self.labels if lbl != pos_label])
        neg_index = random.choice(self.lbl_to_idx[neg_label])
        negative = self.dataset[neg_index]

        code_a, code_p, code_n = anchor["code"], positive["code"], negative["code"]
        # Encode the sequences for sequence pair similarity
        encodings = self.tokenizer([code_a, code_p, code_n], **self.tokenizer_params)
        # Remove batch dimensions
        enc_a = {k: v[0] for k, v in encodings.items()}
        enc_p = {k: v[1] for k, v in encodings.items()}
        enc_n = {k: v[2] for k, v in encodings.items()}
        # Return the tokenized encodings
        return enc_a, enc_p, enc_n

    def __len__(self):
        return len(self.dataset)


def Create_POJ104_triplet_dataset(tokenizer):
    poj_dataset = datasets.load_dataset("semeru/Code-Code-CloneDetection-POJ104")
    train_dataset = POJ104TripletDataset(poj_dataset["train"], tokenizer)
    valid_dataset = POJ104TripletDataset(poj_dataset["validation"], tokenizer)
    test_dataset_map = POJDataset(poj_dataset["test"], tokenizer)
    test_dataset_cls = POJ104TripletDataset(poj_dataset["test"], tokenizer)
    return train_dataset, valid_dataset, test_dataset_map, test_dataset_cls
