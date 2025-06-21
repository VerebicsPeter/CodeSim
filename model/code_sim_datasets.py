# Dataset wrappers for CodeNet data
import os
import gdown
import pandas as pd
import pprint as pp
import random
import datasets
import transformers
from transformers import default_data_collator
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, Subset
from collections import defaultdict

# TODO: Maybe load URLS from a .env or something
# TODO: Create proper dataset with train, validation, evalualion splits for clean evaluation,
# idea: pick a set of 'evaluation' problems distinct from training and validation problems
COLUMNS = ["problem_id", "submission_id", "status", "code"]
DATASET_URL_SMALL = "https://drive.google.com/uc?export=download&id=1hZ4_QjTcmesQYsPo75G1lyIe__HFfwbG"
DATASET_URL_LARGE = "https://drive.google.com/uc?export=download&id=10ok2e2BmWRVhn_V6tRBZzLfeYQ0eaJA2"
DATASET_URL = DATASET_URL_LARGE

# NOTE: Therse are old datasets
OLD_DATASET_URLS = {
    "paired" : "https://drive.google.com/uc?export=download&id=1pUErbyZw1fBC5gIe6KT7BWga7h6Bfr4l",
    "triplet": "https://drive.google.com/uc?export=download&id=11aBIxIMEMKoGyJ9IdUHY2XQv1ZzfyXd2",
}


def custom_collate_triplet(batch):
    A, P, NS = [], [], []
    for a,p,ns in batch:
        A.append(a)
        P.append(p)
        NS.extend(ns)
    encsA = default_data_collator(A)
    encsP = default_data_collator(P)
    encsNS = default_data_collator(NS) 
    return encsA, encsP, encsNS


def get_tokenizer_instance(tokenizer_name: str):
    if tokenizer_name == "Qwen/Qwen3-Embedding-0.6B":
        return transformers.AutoTokenizer.from_pretrained(tokenizer_name, padding_side='left')
    else:
        return transformers.AutoTokenizer.from_pretrained(tokenizer_name)


def get_tokenizer_params(max_length: int):
    return {
        "padding": "max_length",  # Pad to max_length
        "max_length": max_length,
        "truncation": True,  # Truncate to max_length
        "return_tensors": "pt",  # Return torch.Tensor objects
    }


def split_df(df: pd.DataFrame, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, random_state=42):
    assert train_ratio + valid_ratio + test_ratio == 1.0, "Ratios must sum to 1"
    # Initializes empty lists to collect split data
    dfs = defaultdict(list)
    
    # Shuffle and split each class separately
    for problem_id, group in df.groupby("problem_id"):
        group = group.sample(frac=1, random_state=random_state)  # shuffle
        
        n_total = len(group)
        n_train = int(n_total * train_ratio)
        n_valid = int(n_total * valid_ratio)

        train = group.iloc[:n_train]
        valid = group.iloc[ n_train:
                           n_train + n_valid]
        test  = group.iloc[n_train + n_valid:]

        dfs["train"].append(train)
        dfs["valid"].append(valid)
        dfs["test"].append(test)

    train_df = pd.concat(dfs["train"]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    valid_df = pd.concat(dfs["valid"]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = pd.concat(dfs["test"]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return train_df, valid_df, test_df


def load_dataset(url=DATASET_URL, columns=COLUMNS):
    path = "dataset.csv"
    
    if not os.path.isfile(path=path):
        output = path
        gdown.download(url=url, output=output, quiet=False)
    
    df = pd.read_csv("dataset.csv", header=0, names=columns)
    print('\n', df.describe(), '\n')
    print("Splitting dataset...")
    
    train_df, valid_df, test_df = split_df(df)
    
    print(f"Train size: {len(train_df)}, Valid size: {len(valid_df)}, Test size: {len(test_df)}")
    return train_df, valid_df, test_df


def get_loaders(train_data, valid_data, test_data, bs, shuffle, num_workers=4, num_rows=None):
    if num_rows is not None:
        print(f"Limiting dataset to {num_rows} rows.")
        train_data = Subset(train_data, range(num_rows))
        valid_data = Subset(valid_data, range(num_rows))
        test_data  = Subset(test_data , range(num_rows))
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=shuffle, num_workers=num_workers)
    valid_loader = DataLoader(valid_data, batch_size=bs, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_data , batch_size=bs, shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader, test_loader


def encode_tuple(t: tuple[str], tokenizer, tokenizer_params):
    encodings = tokenizer(t, **tokenizer_params)
    # return tuple of encodings (used in default collate_fn)
    return tuple({k: v[i] for k,v in encodings.items()} for i in range(len(t)))


# TODO: augment pairs by flipping the order of the pair
class CodeNetPairDataset(Dataset):
    """Dataset for BERT encodings from (fixed) CodeNet code pairs."""

    def __init__(
        self,
        pids, pairs, labels,
        tokenizer_name: str,
        tokenizer_max_length: int = 256,
        return_single_encoding: bool = True,
    ):
        super().__init__()
        
        assert len(pids) == len(pairs) == len(labels), "Length MUST match!"
        self.pids = pids
        self.pairs = pairs
        self.labels = labels
        self.return_single_encoding = return_single_encoding
        
        self.tokenizer = get_tokenizer_instance(tokenizer_name)
        self.tokenizer_params = get_tokenizer_params(tokenizer_max_length)
        self.encoded_pairs = [self._encode_pair(pair) for pair in pairs]

    def _encode_pair(self, pair):
        code_a, code_b = pair
        if self.return_single_encoding:
            # Encode the sequences for sequence pair classification
            # ([CLS], code_a tokens , [SEP], code_b tokens, [SEP])
            encoding = self.tokenizer(code_a, code_b, **self.tokenizer_params)
            return {k: v.squeeze(0) for k, v in encoding.items()}
        else:
            encodings = encode_tuple(pair, self.tokenizer, self.tokenizer_params)
            return encodings
    
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
        tokenizer_name: str,
        tokenizer_max_length: int = 256,
        return_single_encoding: bool = True,
    ):
        pids, pairs, labels = [], [], []
        
        for pid, group_df in df.groupby("problem_id"):
            df_pos = group_df[group_df["status"] == "Accepted"]
            df_neg = group_df[group_df["status"] != "Accepted"]
            positives = df_pos["code"].to_list()
            negatives = df_neg["code"].to_list()
            n = min(len(positives), len(negatives))
            if n == 0: continue
            #print("sampled:", n, "pairs for problem ID:", pid)
            positives = positives[:n]
            negatives = negatives[:n]
            pids.extend([pid] * 2 * n)
            pairs.extend([*zip(positives, reversed(positives)), *zip(positives, negatives)])
            labels.extend([1] * n + [0] * n)
        
        return cls(pids, pairs, labels, tokenizer_name, tokenizer_max_length, return_single_encoding)


class CodeNetTripletDataset(Dataset):
    """Dataset for BERT encodings from (fixed) CodeNet code triplets."""

    def __init__(
        self,
        pids, triplets,
        tokenizer_name: str,
        tokenizer_max_length: int = 256,
    ):
        super().__init__()
        
        assert len(pids) == len(triplets), "Length MUST match!"
        self.pids = pids
        self.triplets = triplets
        
        self.tokenizer = get_tokenizer_instance(tokenizer_name)
        self.tokenizer_params = get_tokenizer_params(tokenizer_max_length)
        self.encoded_triplets = [self._encode_triplet(triplet) for triplet in triplets]

    def _encode_triplet(self, triplet):
        anchor, positive, negative = triplet
        encodings = encode_tuple((anchor, positive, negative), self.tokenizer, self.tokenizer_params)
        return encodings

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
        tokenizer_name: str,
        tokenizer_max_length: int = 256,
    ):
        pids, triplets = [], []
        
        for pid, group_df in df.groupby("problem_id"):
            df_pos = group_df[group_df["status"] == "Accepted"]
            df_neg = group_df[group_df["status"] != "Accepted"]
            positives = df_pos["code"].to_list()
            pos1 = positives[:len(positives)//2 ]
            pos2 = positives[ len(positives)//2:]
            negatives = df_neg["code"].to_list()
            n = min(len(pos1), len(pos2), len(negatives))
            if n == 0: continue
            #print("sampled:", n, "triplets for problem ID:", pid)
            pids.extend([pid] * n)
            triplets.extend(zip(pos1[:n], pos2[:n], negatives[:n]))

        return cls(pids, triplets, tokenizer_name, tokenizer_max_length)


class CodeNetRandomTripletDataset(Dataset):
    """Dataset for BERT encodings from CodeNet code triplets."""

    def __init__(
        self,
        # NOTE: PID contains the list of problem IDs to sample in an epoch
        pid_to_pos,
        pid_to_neg,
        tokenizer_name: str,
        tokenizer_max_length: int = 256,
        num_negatives: int = 1, # number of same problem (hard) negatives
        deterministic: bool = False,
    ):
        super().__init__()
        all_pids = set(pid_to_pos.keys()) & set(pid_to_neg.keys())
        self.pids = list(all_pids)
        self.pid_to_pos = pid_to_pos
        self.pid_to_neg = pid_to_neg
        self.num_negatives = num_negatives
        self.deterministic = deterministic
        
        self.tokenizer = get_tokenizer_instance(tokenizer_name)
        self.tokenizer_params = get_tokenizer_params(tokenizer_max_length)
        
        enc_func_1 = lambda code: self.tokenizer(code, **self.tokenizer_params)
        enc_func_2 = lambda enc: {k: v.squeeze(0) for k, v in enc.items()}
        
        self.pid_to_pos_enc = {
            pid: list(map(enc_func_2, map(enc_func_1, pos_codes)))
            for pid, pos_codes in pid_to_pos.items()
        }
        self.pid_to_neg_enc = {
            pid: list(map(enc_func_2, map(enc_func_1, pos_codes)))
            for pid, pos_codes in pid_to_neg.items()
        }

    def __getitem__(self, pid: str):
        rng = random.Random(x=42) if self.deterministic else random
        # NOTE: anchor and postive may be the same code, see SimCSE paper
        a = rng.choice(self.pid_to_pos_enc[pid])
        p = rng.choice(self.pid_to_pos_enc[pid])
        ns = rng.sample(self.pid_to_neg_enc[pid], k=self.num_negatives)
        return a, p, ns

    def __len__(self):
        return len(self.pids)

    @classmethod
    def from_pandas_df(
        cls,
        df: pd.DataFrame,
        tokenizer_name: str,
        tokenizer_max_length: int = 256,
        num_negatives: int = 1,
        deterministic: bool = False,
    ):
        pid_to_pos = {}
        pid_to_neg = {}
        
        for pid, group_df in df.groupby("problem_id"):
            df_pos = group_df[group_df["status"] == "Accepted"]
            df_neg = group_df[group_df["status"] != "Accepted"]
            positives = df_pos["code"].to_list()
            negatives = df_neg["code"].to_list()
            pid_to_pos[pid] = positives
            pid_to_neg[pid] = negatives

        return cls(pid_to_pos, pid_to_neg, tokenizer_name, tokenizer_max_length, num_negatives, deterministic)


class CodeNetDefaultTripletSampler(Sampler):
    def __init__(self, pids):
        self.pids = pids
    
    def __iter__(self):
        return iter(self.pids)
    
    def __len__(self):
        return len(self.pids)


class CodeNetRandomTripletBatchSampler(Sampler):
    def __init__(self, pids, num_batches, num_pids_per_batch):
        self.pids = pids
        self.num_batches = num_batches
        self.num_pids_per_batch = num_pids_per_batch

    def __iter__(self):
        for _ in range(self.num_batches):
            yield random.sample(self.pids, self.num_pids_per_batch)

    def __len__(self):
        return self.num_batches


def Create_CodeNet_paired_dataset(
    tokenizer_name: str,
    tokenizer_max_length=256,
    data_path=DATASET_URL,
    return_single_encoding=True,
):
    print("Creating CodeNet dataset. Data type: paired")
    train_df, valid_df, test_df = load_dataset(url=data_path)
    
    _kwargs = {
        "tokenizer_name": tokenizer_name,
        "tokenizer_max_length": tokenizer_max_length,
        "return_single_encoding": return_single_encoding,
    }
    
    train_ds = CodeNetPairDataset.from_pandas_df(train_df, **_kwargs)
    valid_ds = CodeNetPairDataset.from_pandas_df(valid_df, **_kwargs)
    test_ds = CodeNetPairDataset.from_pandas_df(test_df, **_kwargs)
    return train_ds, valid_ds, test_ds


def Create_CodeNet_triplet_dataset(
    tokenizer_name,
    tokenizer_max_length=256,
    data_path=DATASET_URL,
    num_negatives=1,
):
    print("Creating CodeNet dataset. Data type: triplet")
    train_df, valid_df, test_df = load_dataset(url=data_path)
    
    _kwargs = {
        "tokenizer_name": tokenizer_name,
        "tokenizer_max_length": tokenizer_max_length,
    }
    
    train_ds = CodeNetRandomTripletDataset.from_pandas_df(train_df, num_negatives=num_negatives, **_kwargs)
    valid_ds = CodeNetRandomTripletDataset.from_pandas_df(valid_df, num_negatives=num_negatives, deterministic=True, **_kwargs)
    test_ds = CodeNetTripletDataset.from_pandas_df(test_df , **_kwargs)
    return train_ds, valid_ds, test_ds


class POJ104Dataset(Dataset):
    """Simple wrapper for the dataset 'semeru/Code-Code-CloneDetection-POJ104'"""
    
    def __init__(self, poj_dataset, tokenizer_name: str):
        self.dataset = poj_dataset
        
        self.tokenizer = get_tokenizer_instance(tokenizer_name)
        self.tokenizer_params = get_tokenizer_params(self.tokenizer.model_max_length)

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
    
    def __init__(self, poj_dataset, tokenizer_name: str):
        self.dataset = poj_dataset
        self.lbl_to_idx = defaultdict(list)
        for idx, item in enumerate(poj_dataset):
            self.lbl_to_idx[item["label"]].append(idx)
        self.labels = list(self.lbl_to_idx.keys())
        
        self.tokenizer = get_tokenizer_instance(tokenizer_name)
        self.tokenizer_params = get_tokenizer_params(self.tokenizer.model_max_length)

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
        enc_a, enc_p, enc_n = encode_tuple((code_a, code_p, code_n), self.tokenizer, self.tokenizer_params)
        # Return the tokenized encodings
        return enc_a, enc_p, enc_n

    def __len__(self):
        return len(self.dataset)


def Create_POJ104_triplet_dataset(tokenizer_name: str):
    poj_dataset = datasets.load_dataset("semeru/Code-Code-CloneDetection-POJ104")
    train_dataset = POJ104TripletDataset(poj_dataset["train"], tokenizer_name)
    valid_dataset = POJ104TripletDataset(poj_dataset["validation"], tokenizer_name)
    test_dataset_map = POJ104Dataset(poj_dataset["test"], tokenizer_name)
    test_dataset_cls = POJ104TripletDataset(poj_dataset["test"], tokenizer_name)
    return train_dataset, valid_dataset, test_dataset_map, test_dataset_cls
