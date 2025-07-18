# Dataset wrappers for CodeNet data
from tqdm import tqdm
import random
import datasets
from transformers import default_data_collator
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, Subset
from collections import defaultdict
from functools import partial

from model import configs


def tokenize_fn(tokenizer, tokenizer_args, examples):
    return tokenizer(examples["code"], **tokenizer_args)


def init_problem_indices(hf_dataset):
    problem_id_to_passing_idxs = defaultdict(list)
    problem_id_to_failing_idxs = defaultdict(list)
    
    for idx, item in tqdm(enumerate(hf_dataset)):
        if item["status"] == "Accepted":
            problem_id_to_passing_idxs[item["problem_id"]].append(idx)
        else:
            problem_id_to_failing_idxs[item["problem_id"]].append(idx)
    
    return {
        "id_to_passing": problem_id_to_passing_idxs,
        "id_to_failing": problem_id_to_failing_idxs
    }


class CodeNetPairDataset(Dataset):
    """Dataset for fixed CodeNet code pairs."""

    def __init__(self, hf_dataset, num_pairs_per_problem=100, random_state=42,
                 return_single_encoding=False, tokenizer=None, tokenizer_args=None):
        self.hf_dataset = hf_dataset
        self.problem_ids = sorted(set(hf_dataset["problem_id"]))
        result = init_problem_indices(hf_dataset)
        self.problem_id_to_passing_idxs = result["id_to_passing"]
        self.problem_id_to_failing_idxs = result["id_to_failing"]
        
        self.return_single_encoding = return_single_encoding
        # Tokenizer instance for singe-encoding tokenization
        if not self.return_single_encoding:
            assert tokenizer is not None, "Please pass tokenizer for single-encoding tokenization."
            self.tokenizer = tokenizer
            self.tokenizer_args = tokenizer_args or {}
        
        self.pair_idxs = []
        rng = random.Random(random_state)
        N = num_pairs_per_problem
        for pid in self.problem_ids:
            pos_idxs = rng.sample(self.problem_id_to_passing_idxs[pid], 2*N)
            neg_idxs = rng.sample(self.problem_id_to_failing_idxs[pid],   N)
            # Positive pairs (passing, passing)
            self.pair_idxs.extend(zip(pos_idxs[:N], pos_idxs[N:], N*[1]))
            # Negative pairs (passing, failing)
            self.pair_idxs.extend(zip(pos_idxs[:N], neg_idxs[:N], N*[0]))
    
    def __len__(self):
        return len(self.pair_idxs)
    
    def __getitem__(self, idx):
        keys = ["input_ids", "attention_mask"]
        idx_1, idx_2, label = self.pair_idxs[idx]
        if not self.return_single_encoding:
            code_1 = self.hf_dataset[idx_1]["code"]
            code_2 = self.hf_dataset[idx_2]["code"]
            enc = self.tokenizer(code_1, code_2, **self.tokenizer_args)
            return enc, label
        else:
            enc_1 = {k:v for k,v in self.hf_dataset[idx_1].items() if k in keys}
            enc_2 = {k:v for k,v in self.hf_dataset[idx_2].items() if k in keys}
            return enc_1, enc_2, label


class CodeNetTripletDataset(Dataset):
    """Dataset for fixed CodeNet code triplets."""

    def __init__(self, hf_dataset, num_triplets_per_problem=100, random_state=42):
        self.hf_dataset = hf_dataset
        self.problem_ids = sorted(set(hf_dataset["problem_id"]))
        result = init_problem_indices(hf_dataset)
        self.problem_id_to_passing_idxs = result["id_to_passing"]
        self.problem_id_to_failing_idxs = result["id_to_failing"]
        
        self.triplet_idxs = []
        rng = random.Random(random_state)
        N = num_triplets_per_problem
        for pid in self.problem_ids:
            pos_idxs = rng.sample(self.problem_id_to_passing_idxs[pid], 2*N)
            neg_idxs = rng.sample(self.problem_id_to_failing_idxs[pid],   N)
            # anchor, positive, negative triplet
            self.triplet_idxs.extend(zip(pos_idxs[:N], pos_idxs[N:], neg_idxs))
    
    def __len__(self):
        return len(self.triplet_idxs)
    
    def __getitem__(self, idx):
        keys = ["input_ids", "attention_mask"]
        a_idx, p_idx, n_idx = self.triplet_idxs[idx]
        a_enc = {k:v for k,v in self.hf_dataset[a_idx].items() if k in keys}
        p_enc = {k:v for k,v in self.hf_dataset[p_idx].items() if k in keys}
        n_enc = {k:v for k,v in self.hf_dataset[n_idx].items() if k in keys}
        return a_enc, p_enc, n_enc


class CodeNetRandomTripletDataset(Dataset):
    def __init__(self, hf_dataset, deterministic = False, num_negatives = 1):
        self.hf_dataset = hf_dataset
        self.problem_ids = sorted(set(hf_dataset["problem_id"]))
        result = init_problem_indices(hf_dataset)
        self.problem_id_to_passing_idxs = result["id_to_passing"]
        self.problem_id_to_failing_idxs = result["id_to_failing"]
        
        self.deterministic = deterministic
        self.num_negatives = num_negatives

    def __len__(self):
        return len(self.problem_ids)

    def __getitem__(self, pid: str):
        keys = ["input_ids", "attention_mask"]
        rng = random.Random(x=42) if self.deterministic else random
        # NOTE: anchor and postive may be the same code, see SimCSE paper
        a_idx = rng.choice(self.problem_id_to_passing_idxs[pid])
        p_idx = rng.choice(self.problem_id_to_passing_idxs[pid])
        n_idxs = rng.sample(self.problem_id_to_failing_idxs[pid], k=self.num_negatives)
        a_enc = {k:v for k,v in self.hf_dataset[a_idx].items() if k in keys}
        p_enc = {k:v for k,v in self.hf_dataset[p_idx].items() if k in keys}
        n_encs = [{k:v for k,v in self.hf_dataset[n_idx].items() if k in keys} for n_idx in n_idxs]
        return a_enc, p_enc, n_encs


class POJ104Dataset(Dataset):
    """Simple wrapper for the POJ-104 dataset."""
    
    def __init__(self, poj_dataset, tokenizer, tokenizer_args):
        self.dataset = poj_dataset
        self.tokenizer, self.tokenizer_args = tokenizer, tokenizer_args

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        enc = self.tokenizer(item["code"], **self.tokenizer_args)
        enc = {k: v.squeeze(0) for k, v in enc.items()}  # remove batch dim
        label = torch.tensor(int(item["label"])).long()
        return enc, label


class POJ104RandomTripletDataset(Dataset):
    """Simple wrapper for the POJ-104 dataset."""

    def __init__(self, poj_dataset, deterministic = False, sample_negative = False):
        self.poj_dataset = poj_dataset
        self.problem_ids = sorted(set(poj_dataset["label"]))
        self.problem_id_to_encs = defaultdict(list)
        self.deterministic = deterministic
        self.sample_negative = sample_negative

        for item in tqdm(poj_dataset):
            self.problem_id_to_encs[item["label"]].append({
                "input_ids"     : item["input_ids"],
                "attention_mask": item["attention_mask"]
            })

    def __len__(self):
        return len(self.problem_ids)

    def __getitem__(self, pid):
        rng = random.Random(x=42) if self.deterministic else random
        # Positive sample
        enc_1, enc_2 = rng.sample(self.problem_id_to_encs[pid], k=2)

        if not self.sample_negative:
            return enc_1, enc_2
        else:
            # NOTE: this is for triplet loss training
            pid_neg = rng.choice(list(filter(lambda x : x!=pid, self.problem_ids)))
            enc_neg = rng.choice(self.problem_id_to_encs[pid_neg])
            return enc_1, enc_2, enc_neg


def Create_CodeNet_paired_dataset(
    tokenizer,
    tokenizer_args,
    data_path="peterverebics/CodeNet_Python_118",
    return_single_encoding=True,
):
    print("Creating CodeNet dataset. Data type: paired")
    tokenize = partial(tokenize_fn, tokenizer, tokenizer_args)
    
    dataset = datasets.load_dataset(data_path).map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"], output_all_columns=True)
    
    train_ds = CodeNetPairDataset(dataset["train"],
                                  return_single_encoding=return_single_encoding,
                                  tokenizer=tokenizer, tokenizer_args=tokenizer_args)
    valid_ds = CodeNetPairDataset(dataset["validation"],
                                  return_single_encoding=return_single_encoding,
                                  tokenizer=tokenizer, tokenizer_args=tokenizer_args)
    test_ds  = CodeNetPairDataset(dataset["test"],
                                  return_single_encoding=return_single_encoding,
                                  tokenizer=tokenizer, tokenizer_args=tokenizer_args)
    return train_ds, valid_ds, test_ds


def Create_CodeNet_triplet_dataset(
    tokenizer,
    tokenizer_args,
    data_path="peterverebics/CodeNet_Python_118",
    num_negatives=1,
):
    print("Creating CodeNet dataset. Data type: triplet")
    tokenize = partial(tokenize_fn, tokenizer, tokenizer_args)
    
    dataset = datasets.load_dataset(data_path).map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"], output_all_columns=True)
    
    train_ds = CodeNetRandomTripletDataset(dataset["train"],
                                           num_negatives=num_negatives, deterministic=False)
    valid_ds = CodeNetRandomTripletDataset(dataset["validation"],
                                           num_negatives=num_negatives, deterministic=True)
    test_ds = CodeNetTripletDataset(dataset["test"])
    return train_ds, valid_ds, test_ds


def Create_POJ104_triplet_dataset(
    tokenizer,
    tokenizer_args,
    data_path="semeru/Code-Code-CloneDetection-POJ104",
    sample_negative=False,
):
    tokenize = partial(tokenize_fn, tokenizer, tokenizer_args)
    dataset = datasets.load_dataset(data_path).map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"], output_all_columns=True)
    
    train_ds = POJ104RandomTripletDataset(dataset["train"],
                                          sample_negative=sample_negative)
    valid_ds = POJ104Dataset(dataset["validation"],
                             tokenizer, tokenizer_args)
    test_ds  = POJ104Dataset(dataset["test"],
                             tokenizer, tokenizer_args)
    return train_ds, valid_ds, test_ds


class DefaultTripletSampler(Sampler):
    def __init__(self, pids):
        self.pids = pids
    
    def __iter__(self):
        return iter(self.pids)
    
    def __len__(self):
        return len(self.pids)


class RandomTripletBatchSampler(Sampler):
    def __init__(self, pids, num_batches, num_pids_per_batch):
        self.pids = pids
        self.num_batches = num_batches
        self.num_pids_per_batch = num_pids_per_batch

    def __iter__(self):
        for _ in range(self.num_batches):
            yield random.sample(self.pids, self.num_pids_per_batch)

    def __len__(self):
        return self.num_batches


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


def custom_collate_POJpair(batch):
    A, P = [], []
    for a,p in batch:
        A.append(a)
        P.append(p)
    encsA = default_data_collator(A)
    encsP = default_data_collator(P)
    # Hacky dummy encoding for negative placeholder
    encs_dummy = {k: torch.zeros(0, v.shape[-1], dtype=v.dtype) for k, v in encsA.items()}
    return encsA, encsP, encs_dummy


def get_loaders(train_data, valid_data, test_data, bs, shuffle, num_workers=4, num_rows=None):
    if num_rows is not None:
        print(f"Limiting dataset to {num_rows} rows.")
        train_data = Subset(train_data, range(num_rows))
        valid_data = Subset(valid_data, range(num_rows))
        test_data = Subset(test_data, range(num_rows))
    
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=shuffle, num_workers=num_workers)
    valid_loader = DataLoader(valid_data, batch_size=bs, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=bs, shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader, test_loader


def get_CodeNet_loaders(
    train_data: CodeNetRandomTripletDataset,
    valid_data: CodeNetRandomTripletDataset,  # TODO: Change to CodeNetTripletDataset
    test_data : CodeNetTripletDataset,
    config: configs.CodeSimContrastiveClassifierConfig
):
    train_loader = DataLoader(
        train_data, 
        batch_sampler=RandomTripletBatchSampler(
            pids=train_data.problem_ids,
            num_batches=config.num_batches,
            num_pids_per_batch=config.bs
        ),
        collate_fn=custom_collate_triplet
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=config.bs,
        sampler=DefaultTripletSampler(valid_data.problem_ids),
        collate_fn=custom_collate_triplet
    )
    test_loader = DataLoader(
        test_data,
        batch_size=config.bs
    )
    return train_loader, valid_loader, test_loader


def get_POJ104_loaders(
    train_data: POJ104RandomTripletDataset, 
    valid_data: POJ104Dataset, 
    test_data : POJ104Dataset, 
    config: configs.CodeSimContrastiveClassifierConfig
):
    train_loader = DataLoader(
        train_data,
        batch_sampler=RandomTripletBatchSampler(
            pids=train_data.problem_ids,
            num_batches=config.num_batches,
            num_pids_per_batch=config.bs
        ),
        collate_fn=custom_collate_POJpair
    )
    valid_loader = DataLoader(valid_data, batch_size=config.bs, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=config.bs, shuffle=False)
    return train_loader, valid_loader, test_loader
