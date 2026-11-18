"""
Manual PyTorch training loop for experiments A and B (no Hugging Face Trainer).

Usage:
    python -m src.train                  # run A then B
    python -m src.train --experiment B   # run one experiment
    python -m src.train --epochs 1       # quick smoke test

One epoch, step by step:

    for each batch in train_loader:                       model.train()
        logits = model(input_ids, attention_mask)         forward pass
        loss   = CrossEntropy(logits, labels)             how wrong are we?
        optimizer.zero_grad()                             clear old gradients
        loss.backward()                                   compute new gradients
        clip_grad_norm_(...)                              cap exploding gradients
        optimizer.step()                                  update trainable weights
        scheduler.step()                                  decay the learning rate
    validate on validation set                            model.eval() + no_grad
    if val Macro F1 improved: save checkpoint             model selection

Outputs per experiment:
    checkpoints/<name>/            best checkpoint (weights + config + tokenizer)
    results/training/<name>_history.json    per-epoch metrics, timing, parameter counts
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from tqdm.auto import tqdm

from src.config import (
    CHECKPOINT_DIR,
    EXPERIMENTS,
    MAX_GRAD_NORM,
    TRAINING_DIR,
    SEED,
    WEIGHT_DECAY,
    ExperimentConfig,
    get_device,
    set_seed,
)
from src.dataset import build_dataloaders, load_tokenizer
from src.evaluate import compute_metrics, run_inference, split_batch
from src.model import build_model, count_parameters


def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, device) -> float:
    """Run one pass over the training set and return the mean training loss."""
    model.train()  # enables dropout
    total_loss, n_examples = 0.0, 0

    for batch in tqdm(loader, desc="train", leave=False):
        inputs, labels = split_batch(batch, device)

        logits = model(**inputs).logits   # forward pass -> shape (batch, 6)
        loss = loss_fn(logits, labels)    # explicit CrossEntropy (mean over batch)

        optimizer.zero_grad()             # gradients accumulate by default; reset them
        loss.backward()                   # backpropagation: d(loss)/d(weight) for trainable weights
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()                  # weight update (frozen params have no grad -> untouched)
        scheduler.step()                  # learning-rate schedule advances once per batch

        total_loss += loss.item() * labels.size(0)
        n_examples += labels.size(0)

    return total_loss / n_examples


def run_experiment(exp: ExperimentConfig, epochs: int | None = None) -> dict:
    """Train one experiment from fresh pretrained weights and keep its best epoch."""
    epochs = epochs or exp.epochs
    device = get_device()
    print(f"\n=== Experiment {exp.key}: {exp.description} | device={device} ===")

    # Same seed before data + model creation -> same shuffling and same head init
    # for A and B, so the only difference between them is what we choose to train.
    set_seed(SEED)
    tokenizer = load_tokenizer()
    loaders = build_dataloaders(tokenizer, batch_size=exp.batch_size, splits=("train", "validation"))
    model = build_model(exp.num_unfrozen_layers).to(device)

    trainable, total = count_parameters(model)
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # Only hand trainable parameters to the optimizer - frozen ones need no state.
    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=exp.learning_rate,
        weight_decay=WEIGHT_DECAY,
    )
    # Linear decay from the initial LR down to 0 over all training steps.
    total_steps = epochs * len(loaders["train"])
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps)
    loss_fn = nn.CrossEntropyLoss()

    ckpt_dir = CHECKPOINT_DIR / exp.name
    history = {
        "experiment": exp.name,
        "description": exp.description,
        "learning_rate": exp.learning_rate,
        "batch_size": exp.batch_size,
        "device": str(device),
        "trainable_params": trainable,
        "total_params": total,
        "epochs": [],
        "best_epoch": None,
        "best_val_macro_f1": -1.0,
    }

    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(model, loaders["train"], optimizer, scheduler, loss_fn, device)
        val = run_inference(model, loaders["validation"], device)
        val_metrics = compute_metrics(val.labels, val.preds)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val.loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        history["epochs"].append(record)
        print(
            f"epoch {epoch}/{epochs} | train_loss {train_loss:.4f} | val_loss {val.loss:.4f} "
            f"| val_acc {val_metrics['accuracy']:.4f} | val_macro_f1 {val_metrics['macro_f1']:.4f}"
        )

        # Model selection on VALIDATION Macro F1 (never on the test set).
        if val_metrics["macro_f1"] > history["best_val_macro_f1"]:
            history["best_val_macro_f1"] = val_metrics["macro_f1"]
            history["best_epoch"] = epoch
            model.save_pretrained(ckpt_dir)      # weights + config.json (with id2label)
            tokenizer.save_pretrained(ckpt_dir)  # tokenizer travels with the model
            print(f"  -> new best checkpoint saved to {ckpt_dir}")

    history["train_seconds"] = time.perf_counter() - start
    print(f"Finished in {history['train_seconds'] / 60:.1f} min | best epoch {history['best_epoch']}")

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    (TRAINING_DIR / f"{exp.name}_history.json").write_text(json.dumps(history, indent=2))

    # Free GPU memory before the next experiment starts.
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EmotionSense experiments.")
    parser.add_argument("--experiment", choices=["A", "B", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=None, help="Override the 3-epoch budget (e.g. 1 for a smoke test).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keys = ["A", "B"] if args.experiment == "all" else [args.experiment]
    for key in keys:
        run_experiment(EXPERIMENTS[key], epochs=args.epochs)
    print("\nDone. Next step: python -m src.evaluate")


if __name__ == "__main__":
    main()
