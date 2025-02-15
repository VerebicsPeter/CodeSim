import pandas as pd
import transformers
import torch
from torch.utils.data import Dataset
from typing import Iterable, Callable


def get_batch_encodings(
    codes: Iterable[str],
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
    device: str = "cpu"
) -> transformers.BatchEncoding:
    MAX_LEN = tokenizer.model_max_length

    inputs = tokenizer(
        codes,
        truncation=True,
        # Pad to "MAX_LEN + 1" to detect sequences that are too long
        padding="max_length", max_length=MAX_LEN + 1,
        return_tensors="pt",
    )

    # Mask sequences that are longer than "MAX_LEN"
    l_mask = inputs["attention_mask"].sum(dim=1) <= MAX_LEN

    inputs = { k: v[l_mask, :MAX_LEN] for k, v in inputs.items() }
    # Move tensors to the specified device
    inputs = { k: v.to(device) for k, v in inputs.items() }

    return inputs


def get_numeric_labels(labels: Iterable[str]) -> torch.Tensor:
    """Transform string labels to int labels for the NTXent loss function."""
    pos_labels = [ label for label in labels if label.endswith('1') ]
    labels_map = { label: i for i, label in enumerate(sorted(set(pos_labels))) }
    int_labels = torch.Tensor([labels_map.get(label, -1) for label in labels])
    neg_indices = (int_labels == -1).nonzero(as_tuple=True)[0]
    M = max(int_labels)
    int_labels[neg_indices] = torch.arange(M + 1, M + 1 + len(neg_indices))
    return int_labels


def augment(df: pd.DataFrame, *functions):
    """Calculates data augmentations on a CodeNet sampled dataframe with labeled source code."""
    # Dataframe to augment (dataframe containing passing CodeNet examples)
    to_aug = df[df['label'].apply(lambda label: label.endswith('1'))]
    augs = []
    for function in functions:
        aug = to_aug.copy()
        aug.loc[:, 'source'] = aug['source'].apply(function)
        augs.append(aug)
    df = pd.concat([df, *augs], ignore_index=True)
    # Sort the dataframe so matching labels are next to eachother.
    df.sort_values(by='label', inplace=True)
    return df


class LabeledCodeDataset(Dataset):
    """Labeled code dataset for code snippets from CodeNet.
    
        NOTE:
        
        Solutions from the same problem with "ACCEPTED" status are labeled by the same number.
        Other solutions are labeled by distinct numbers.
    """
    
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
        codes: Iterable[str],
        labels: Iterable[str],
        device: str,
    ):
        assert len(codes) == len(labels)
        self.inputs = get_batch_encodings(codes, tokenizer, device)
        self.labels = get_numeric_labels(labels)

    def __getitem__(self, idx):
        input = {k: v[idx] for k, v in self.inputs.items()}
        label = self.labels[idx]
        return input, label

    def __len__(self):
        return self.inputs['input_ids'].shape[0]

    @classmethod
    def from_csv_data(cls, path: str, tokenizer, aug_funcs: Iterable[Callable], device: str):
        df = pd.read_csv(path)
        print(df.shape)

        if aug_funcs:
            print('Augmenting data (this might take a while)...')
            codes = augment(df, *aug_funcs)
            print(codes.shape)

        codes = df['source']
        codes = codes.to_list()

        labels = df['label']
        labels = labels.to_list()

        return cls(tokenizer, codes, labels, device)


class UnlabeledCodeDataset(Dataset):
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
        ref_codes: Iterable[str],
        aug_codes: Iterable[str],
        device: str,
    ):
        assert len(ref_codes) == len(aug_codes)
        self.ref_inputs = get_batch_encodings(ref_codes, tokenizer, device)
        self.aug_inputs = get_batch_encodings(aug_codes, tokenizer, device)

    def __getitem__(self, idx):
        # Return both reference and augmented code inputs for a given index
        ref_input = {k: v[idx] for k, v in self.ref_inputs.items()}
        aug_input = {k: v[idx] for k, v in self.aug_inputs.items()}
        return ref_input, aug_input

    def __len__(self):
        return self.ref_inputs["input_ids"].shape[0]

    @classmethod
    def from_csv_data(cls, path: str, tokenizer, aug_funcs: Iterable[Callable], device: str):
        df = pd.read_csv(path)
        print(df.shape)

        ref_codes = df['file_content']
        ref_codes = ref_codes.to_list()

        aug_codes = df['file_content'].apply(aug_funcs[0])
        aug_codes = aug_codes.to_list()
        # TODO: multiple augmentations

        return cls(tokenizer, ref_codes, aug_codes, device)


def downsample_df(df: pd.DataFrame, samples_per_class, seed=42):
    pos_df = df[df['label'] == 1]
    neg_df = df[df['label'] == 0]
    pos_sampled = neg_df.sample(n=min(samples_per_class, len(pos_df)), random_state=seed)
    neg_sampled = pos_df.sample(n=min(samples_per_class, len(neg_df)), random_state=seed)
    # Combine the downsampled dataframes
    downsampled_df = pd.concat([pos_sampled, neg_sampled]).reset_index(drop=True)
    return downsampled_df


class CodeNetPairDataset(Dataset):
    """Dataset for BERT encodings from CodeNet code pairs."""
    
    """Dataframe columns' schema"""
    COLUMNS = [
        'pid',    # CodeNet problem ID
        'sid_1',  # CodeNet solution ID of 'src_1'
        'sid_2',  # CodeNet solution ID of 'src_2'
        'src_1',  # CodeNet solution code of 'sid_1'
        'src_2',  # CodeNet solution code of 'sid_2'
        'label',  # Label indicating if 'src_1' and 'src_2' both solve 'pid'
    ]
    
    def __init__(
        self, 
        codes_a, codes_b, labels, 
        tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
    ):
        super().__init__()
        assert len(codes_a) == len(codes_b) == len(labels), "Length MUST match!"
        self.codes_a = codes_a
        self.codes_b = codes_b
        self.labels = labels
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        code_a = self.codes_a[idx]
        code_b = self.codes_b[idx]
        label = self.labels[idx]
        # Encode the sequences for sequence pair classification
        # ([CLS], code_a tokens , [SEP], code_b tokens, [SEP])
        encoding = self.tokenizer(
            code_a, code_b,
            padding='max_length',  # Pad to max_length
            max_length=self.tokenizer.model_max_length,
            truncation=True,       # Truncate to max_length
            return_tensors='pt'    # Return torch.Tensor objects
        )
        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}

        return encoding, label

    def __len__(self):
        return len(self.labels)

    @classmethod
    def from_pandas_df(cls, df: pd.DataFrame, tokenizer, num_pairs=5000):
        # Filter sequences that don't fit the model's max length
        def filter_too_long_sequences(row):
            source = row['src_1'] + row['src_2']
            tokens = tokenizer.encode(source, truncation=False)
            return len(tokens) <= tokenizer.model_max_length

        print('Filtering dataset, this might take a while...')
        df = df[df.apply(filter_too_long_sequences, axis=1)]
        print('filtered dataset:', df.shape)
        df = downsample_df(df, samples_per_class=(num_pairs // 2))
        print('Downsampled dataset:', df.shape)

        codes_a = df['src_1'].to_list()
        codes_b = df['src_2'].to_list()
        labels  = df['label'].to_list()
        return cls(codes_a, codes_b, labels, tokenizer)
