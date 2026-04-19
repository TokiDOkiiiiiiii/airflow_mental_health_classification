# PyTorch Dataset extracted directly from the BERT notebook.

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class MentalHealthDataset(Dataset):
    """Tokenising dataset — exact replica of the notebook's MentalHealthDataset."""

    def __init__(
        self,
        text: list[str],
        label: list[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
    ):
        self.text = text
        self.label = label
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            str(self.text[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(int(self.label[idx]), dtype=torch.long),
        }
