import torch
from torch.utils.data import Dataset
import transformers
import pandas as pd
import pprint as pp
from typing import Iterable


device = "cuda" if torch.cuda.is_available() else "cpu"


def get_batch_encodings(
    codes: Iterable[str],
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
    device: str = "cpu",
) -> transformers.BatchEncoding:
    MAX_LEN = tokenizer.model_max_length
    inputs = tokenizer(
        codes,
        truncation=True,
        # Pad to "MAX_LEN + 1" to detect sequences that are too long
        padding="max_length",
        max_length=MAX_LEN + 1,
        return_tensors="pt",
    )

    # Mask sequences that are longer than "MAX_LEN"
    l_mask = inputs["attention_mask"].sum(dim=1) <= MAX_LEN

    inputs = {k: v[l_mask, :MAX_LEN] for k, v in inputs.items()}
    # Move tensors to the specified device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    return inputs


class UnlabeledCodeDataset(Dataset):
    def __init__(
        self,
        tokenizer: (
            transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast
        ),
        ref_codes: Iterable[str],
        aug_codes: Iterable[str],
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
    def from_csv_data(cls, path: str, tokenizer, aug_func, sample_size=0):
        df = pd.read_csv(path)
        print(df.shape)

        if sample_size:
            print("sampling dataframe...")
            df = df.sample(sample_size, ignore_index=True)
            print(df.shape)

        ref_codes = df["source"]
        aug_codes = df["source"].apply(aug_func)

        return cls(tokenizer, ref_codes.to_list(), aug_codes.to_list())


if __name__ == "__main__":
    chckpt = "huggingface/CodeBERTa-small-v1"
    tokenizer = transformers.AutoTokenizer.from_pretrained(chckpt)

    data = pd.read_csv("dataset/output_single.csv", header=0, names=["source", "label"])
    pp.pp(data)

    code = data["source"].to_list()

    dataset = UnlabeledCodeDataset(tokenizer, code, code)

    print("\nDataset loaded:")
    print(dataset.ref_inputs["input_ids"].shape)
    
    ref, aug = dataset[0]
    pp.pp(ref)
    print('-'*100)
    pp.pp(aug)
