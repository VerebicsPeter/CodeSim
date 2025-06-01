# Dataset wrappers for CodeNet data
import gdown
import pandas as pd
import pprint as pp
import random
import transformers
from torch.utils.data import Dataset
from typing import Iterable
from collections import defaultdict


DATASET_TYPE = {
    "paired",
    "triplet",
}

# TODO: Create proper dataset with train, validation, evalualion splits for clean evaluation,
# idea: pick a set of 'evaluation' problems distinct from training and validation problems
# TODO: Maybe load URLS from a .env or something
DATASET_URLS = {
    "paired" : "https://drive.google.com/uc?export=download&id=1pUErbyZw1fBC5gIe6KT7BWga7h6Bfr4l",
    "triplet": "https://drive.google.com/uc?export=download&id=11aBIxIMEMKoGyJ9IdUHY2XQv1ZzfyXd2",
    # NOTE: Old datasets
    #"contrastive_labeled"  : "https://drive.google.com/uc?export=download&id=1UteITBYXcBLt2hXviy71jQr-oXceVcs5",
    #"contrastive_unlabeled": "https://drive.google.com/uc?export=download&id=1iHHgOcJQ_qp3sk3d7w1zpWBvsgDqrPJV",
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


class CodeNetPairDataset(Dataset):
    """Dataset for BERT encodings from CodeNet code pairs."""

    """Dataframe columns' schema"""
    COLUMNS = [
        "pid",  # CodeNet problem ID
        "sid_1",  # CodeNet solution ID of 'src_1'
        "sid_2",  # CodeNet solution ID of 'src_2'
        "src_1",  # CodeNet solution code of 'sid_1'
        "src_2",  # CodeNet solution code of 'sid_2'
        "label",  # Label indicating if 'src_1' and 'src_2' both solve 'pid'
    ]

    def __init__(
        self,
        pids,
        codes_a,
        codes_b,
        labels,
        tokenizer: (
            transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast
        ),
        return_single_encoding: bool,
    ):
        super().__init__()
        assert len(codes_a) == len(codes_b) == len(labels), "Length MUST match!"
        self.pids = pids
        self.codes_a = codes_a
        self.codes_b = codes_b
        self.labels = labels
        self.tokenizer = tokenizer
        self.tokenizer_params = {
            "padding":'max_length',  # Pad to max_length
            "max_length": self.tokenizer.model_max_length,
            "truncation":True,       # Truncate to max_length
            "return_tensors":'pt'    # Return torch.Tensor objects
        }
        self.return_single_encoding = return_single_encoding

    def __getitem__(self, idx):
        code_a = self.codes_a[idx]
        code_b = self.codes_b[idx]
        label = self.labels[idx]
        
        if self.return_single_encoding:
            # Encode the sequences for sequence pair classification
            # ([CLS], code_a tokens , [SEP], code_b tokens, [SEP])
            encoding = self.tokenizer(code_a, code_b, **self.tokenizer_params)
            # Remove batch dimension
            encoding = {k: v.squeeze(0) for k, v in encoding.items()}
            return encoding, label
        else:
            enc_u = self.tokenizer(code_a, **self.tokenizer_params)
            enc_v = self.tokenizer(code_b, **self.tokenizer_params)
            # Remove batch dimension
            enc_u = {k: v.squeeze(0) for k, v in enc_u.items()}
            enc_v = {k: v.squeeze(0) for k, v in enc_v.items()}
            return enc_u, enc_v, label

    def __len__(self):
        return len(self.labels)

    @classmethod
    def from_pandas_df(cls, df: pd.DataFrame, tokenizer, num_rows=5000, return_single_encoding: bool = True):
        # Filter sequences that don't fit the model's max length
        def filter_too_long_sequences(row):
            if return_single_encoding:
                source = row["src_1"] + row["src_2"]
                tokens = tokenizer.encode(source, truncation=False)
                return len(tokens) <= tokenizer.model_max_length
            else:
                u_fits = len(tokenizer.encode(row["src_1"], truncation=False)) <= tokenizer.model_max_length
                v_fits = len(tokenizer.encode(row["src_2"], truncation=False)) <= tokenizer.model_max_length
                return u_fits and v_fits
        
        def sample_df(df: pd.DataFrame, samples_per_class, seed=42, drop_old_index=True):
            pos_df = df[df["label"] == 1]
            neg_df = df[df["label"] == 0]
            pos_sampled = neg_df.sample(
                n=min(samples_per_class, len(pos_df)), random_state=seed
            )
            neg_sampled = pos_df.sample(
                n=min(samples_per_class, len(neg_df)), random_state=seed
            )
            # Combine the sampled dataframes
            sampled_df = pd.concat([pos_sampled, neg_sampled]).reset_index(
                drop=drop_old_index
            )
            return sampled_df

        print("Filtering dataset, this might take a while...")
        df = df[df.apply(filter_too_long_sequences, axis=1)]
        print("Filtered dataset:", df.shape)
        df = sample_df(df, samples_per_class=(num_rows // 2))
        print("Sampled dataset:", df.shape)
        
        pids = df["pid"].to_list()
        codes_a = df["src_1"].to_list()
        codes_b = df["src_2"].to_list()
        labels = df["label"].to_list()
        return cls(pids, codes_a, codes_b, labels, tokenizer, return_single_encoding)


class CodeNetTripletDataset(Dataset):
    """Dataset for BERT encodings from CodeNet code triplets."""

    """Dataframe columns' schema"""
    COLUMNS = [
        "pid",  # CodeNet problem ID
        "sid_a",  # CodeNet solution ID of 'src_a'
        "sid_p",  # CodeNet solution ID of 'src_p'
        "sid_n",  # CodeNet solution ID of 'src_n'
        "src_a",  # CodeNet solution code of anchor
        "src_p",  # CodeNet solution code of positive pair of anchor
        "src_n",  # CodeNet solution code of negative pair of anchor
    ]

    def __init__(
        self,
        pids,
        codes_a,
        codes_p,
        codes_n,
        tokenizer: (
            transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast
        ),
    ):
        super().__init__()
        assert len(codes_a) == len(codes_p) == len(codes_n), "Length MUST match!"
        self.pids = pids
        self.codes_a = codes_a
        self.codes_p = codes_p
        self.codes_n = codes_n
        self.tokenizer = tokenizer
        self.tokenizer_params = {
            "padding": "max_length",  # Pad to max_length
            "max_length": self.tokenizer.model_max_length,
            "truncation": True,  # Truncate to max_length
            "return_tensors": "pt",  # Return torch.Tensor objects
        }

    def __getitem__(self, idx):
        code_a = self.codes_a[idx]
        code_p = self.codes_p[idx]
        code_n = self.codes_n[idx]
        # Encode the sequences for sequence pair similarity
        encodings = self.tokenizer([code_a, code_p, code_n], **self.tokenizer_params)
        # Remove batch dimensions
        enc_a = {k: v[0] for k, v in encodings.items()}
        enc_p = {k: v[1] for k, v in encodings.items()}
        enc_n = {k: v[2] for k, v in encodings.items()}
        # Return the tokenized encodings
        return enc_a, enc_p, enc_n

    def __len__(self):
        return len(self.codes_a)

    @classmethod
    def from_pandas_df(cls, df: pd.DataFrame, tokenizer, num_rows=5000):
        # Filter sequences that don't fit the model's max length
        def filter_too_long_sequences(row):
            encode = lambda code: tokenizer.encode(code, truncation=False)
            fits_model = all(
                map(
                    lambda x: len(encode(x)) <= tokenizer.model_max_length,
                    [row["src_a"], row["src_p"], row["src_n"]],
                )
            )
            return fits_model

        print("Processing dataset:", df.shape)
        print("Filtering dataset, this might take a while...")
        df = df[df.apply(filter_too_long_sequences, axis=1)]
        print("Filtered dataset:", df.shape)
        df = df.sample(num_rows)
        print("Sampled dataset:", df.shape)

        pids = df["pid"].to_list()
        codes_a = df["src_a"].to_list()
        codes_p = df["src_p"].to_list()
        codes_n = df["src_n"].to_list()
        return cls(pids, codes_a, codes_p, codes_n, tokenizer)


def Create_CodeNet_paired_dataset(
    tokenizer,
    data_path=DATASET_URLS["paired"],
    num_rows=5000,
    return_single_encoding=True
):
    download_dataset(data_path, "dataset.csv")
    df = pd.read_csv(
        "dataset.csv", header=0,
        names=CodeNetPairDataset.COLUMNS
    )
    print("CodeNet data loaded. Data type: paired")
    pp.pp(df)

    dataset = CodeNetPairDataset.from_pandas_df(
        df,
        tokenizer=tokenizer,
        num_rows=num_rows,
        return_single_encoding=return_single_encoding,
    )
    return dataset


def Create_CodeNet_triplet_dataset(
    tokenizer,
    data_path=DATASET_URLS["triplet"],
    num_rows=5000,
):
    download_dataset(data_path, "dataset.csv")
    df = pd.read_csv(
        "dataset.csv", header=0,
        names=CodeNetTripletDataset.COLUMNS
    )
    print("CodeNet data loaded. Data type: triplet")
    pp.pp(df)

    dataset = CodeNetTripletDataset.from_pandas_df(
        df,
        tokenizer=tokenizer,
        num_rows=num_rows, 
    )
    return dataset


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

    def __getitem__(self, idx):  # TODO: maybe seed this explicitly
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
