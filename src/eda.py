"""
Exploratory data analysis (Section 1.2) - look at the data before modelling it.

Usage:
    python -m src.eda

What it checks and why:
    columns & label names   -> confirm our LABELS mapping matches the dataset card
    sample rows             -> get a feel for the language (informal, first-person)
    null / empty text       -> broken rows would crash or confuse training
    duplicates              -> repeated texts inflate scores; conflicting labels = noise
    train/test overlap      -> leakage would make test results look better than reality
    token lengths           -> justifies MAX_LENGTH=128 (how much gets truncated?)
    label balance           -> motivates Macro F1 instead of plain accuracy

Outputs:
    results/eda/class_distribution.png
    results/eda/token_length_distribution.png
    results/eda/eda_summary.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import EDA_DIR, LABELS, MAX_LENGTH
from src.dataset import SPLITS, load_emotion_splits, load_tokenizer
from src.visualize import plot_class_distribution, plot_token_lengths


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    data = load_emotion_splits()
    frames = {split: data[split].to_pandas() for split in SPLITS}
    summary: dict = {}

    # 1) Structure -------------------------------------------------------------
    section("1. Splits, columns and label names")
    print(data)
    dataset_labels = data["train"].features["label"].names
    print("Label names from the dataset card:", dataset_labels)
    assert dataset_labels == LABELS, "Label order differs from src/config.py LABELS!"
    print("OK - matches our LABELS mapping:", dict(enumerate(LABELS)))

    # 2) Samples ---------------------------------------------------------------
    section("2. Sample text-label pairs (2 per emotion, train split)")
    train = frames["train"].assign(emotion=lambda df: df["label"].map(dict(enumerate(LABELS))))
    samples = train.groupby("emotion").sample(n=2, random_state=42)
    with pd.option_context("display.max_colwidth", 90):
        print(samples[["emotion", "text"]].to_string(index=False))

    # 3) Data quality ----------------------------------------------------------
    section("3. Null / empty texts and duplicates")
    quality = {}
    for split, df in frames.items():
        texts = df["text"]
        dup_mask = texts.duplicated(keep=False)
        conflicting = df[dup_mask].groupby("text")["label"].nunique().gt(1).sum()
        quality[split] = {
            "rows": len(df),
            "null_text": int(texts.isna().sum()),
            "empty_text": int(texts.fillna("").str.strip().eq("").sum()),
            "duplicate_rows": int(texts.duplicated().sum()),
            "duplicate_texts_with_conflicting_labels": int(conflicting),
        }
    print(pd.DataFrame(quality).T.to_string())

    train_texts = set(frames["train"]["text"])
    overlap = {s: int(frames[s]["text"].isin(train_texts).sum()) for s in ("validation", "test")}
    print(f"\nTexts that also appear in train (possible leakage): {overlap}")
    summary["quality"] = quality
    summary["overlap_with_train"] = overlap

    # 4) Lengths ---------------------------------------------------------------
    section("4. Text length (words and tokens)")
    word_counts = frames["train"]["text"].str.split().str.len()
    print("Words per text (train):\n", word_counts.describe().round(1).to_string())

    tokenizer = load_tokenizer()
    token_lengths = [len(ids) for ids in tokenizer(list(frames["train"]["text"]))["input_ids"]]
    truncated = int(np.sum(np.array(token_lengths) > MAX_LENGTH))
    print(
        f"\nTokens per text (train): mean {np.mean(token_lengths):.1f}, "
        f"95th pct {np.percentile(token_lengths, 95):.0f}, max {max(token_lengths)}"
    )
    print(f"Texts longer than MAX_LENGTH={MAX_LENGTH}: {truncated} ({100 * truncated / len(token_lengths):.2f}%)")
    summary["tokens"] = {
        "mean": float(np.mean(token_lengths)),
        "p95": float(np.percentile(token_lengths, 95)),
        "max": int(max(token_lengths)),
        "truncated_at_max_length": truncated,
    }
    plot_token_lengths(token_lengths, MAX_LENGTH, EDA_DIR / "token_length_distribution.png")

    # 5) Label balance ---------------------------------------------------------
    section("5. Label balance")
    counts = {
        split: {LABELS[i]: int(n) for i, n in df["label"].value_counts().sort_index().items()}
        for split, df in frames.items()
    }
    balance = pd.DataFrame(counts)
    balance["train_%"] = (100 * balance["train"] / balance["train"].sum()).round(1)
    print(balance.to_string())
    ratio = max(counts["train"].values()) / min(counts["train"].values())
    print(f"\nImbalance ratio (largest / smallest class, train): {ratio:.1f}x")
    summary["label_counts"] = counts
    summary["imbalance_ratio_train"] = round(ratio, 2)
    plot_class_distribution(counts, LABELS, EDA_DIR / "class_distribution.png")

    (EDA_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary -> {EDA_DIR / 'eda_summary.json'}")


if __name__ == "__main__":
    main()
