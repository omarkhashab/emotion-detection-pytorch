"""
Data pipeline: raw text  ->  token IDs  ->  padded GPU-ready batches.

    load_emotion_splits()      Hugging Face DatasetDict (train / validation / test)
            |
    EmotionDataset             torch Dataset: tokenizes once, returns one example
            |
    DataLoader + collator      groups examples into batches, pads per batch
            |
    {"input_ids", "attention_mask", "labels"}   -> fed to the model

Design choice - dynamic padding:
    We truncate to MAX_LENGTH but do NOT pad in the Dataset. The collator pads
    each batch only to the length of its longest sentence. Emotion texts are
    short (~20 tokens on average), so this is much faster than padding every
    example to 128 tokens, and the attention mask tells the model which
    positions are padding.
"""

from __future__ import annotations

import torch
from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, PreTrainedTokenizerBase

from src.config import BATCH_SIZE, DATASET_CONFIG, DATASET_NAME, MAX_LENGTH, MODEL_NAME, SEED

SPLITS = ("train", "validation", "test")


def load_emotion_splits() -> DatasetDict:
    """Download (or load from cache) the predefined train/validation/test splits.

    The three splits are kept separate on purpose:
      * train      -> updates the weights
      * validation -> chooses the best epoch / checkpoint
      * test       -> touched only once at the end, for the final report
    """
    return load_dataset(DATASET_NAME, DATASET_CONFIG)


def load_tokenizer(name_or_path: str = MODEL_NAME) -> PreTrainedTokenizerBase:
    """Load the WordPiece tokenizer that DistilBERT was pretrained with.

    Using the *paired* tokenizer matters: the model's embedding table only
    understands the token IDs produced by this exact vocabulary.
    """
    return AutoTokenizer.from_pretrained(name_or_path)


class EmotionDataset(Dataset):
    """A PyTorch Dataset over one split.

    Texts are tokenized once in __init__ (fast, batched Rust tokenizer) instead
    of on every __getitem__ call, so each epoch only does cheap list lookups.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = MAX_LENGTH,
    ) -> None:
        self.texts = texts    # kept for error analysis (we want to read the mistakes)
        self.labels = labels
        self.encodings = tokenizer(texts, truncation=True, max_length=max_length)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, list[int] | int]:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def build_dataloaders(
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
    splits: tuple[str, ...] = SPLITS,
) -> dict[str, DataLoader]:
    """Create one DataLoader per requested split.

    * Only the training loader shuffles (so the model doesn't learn the order).
    * A seeded generator makes the shuffle order reproducible.
    * DataCollatorWithPadding pads input_ids/attention_mask per batch and turns
      everything into tensors.
    """
    data = load_emotion_splits()
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    generator = torch.Generator().manual_seed(seed)

    loaders: dict[str, DataLoader] = {}
    for split in splits:
        # list(...) because newer `datasets` versions return a lazy Column object.
        dataset = EmotionDataset(list(data[split]["text"]), list(data[split]["label"]), tokenizer)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=collator,
            generator=generator if split == "train" else None,
        )
    return loaders
