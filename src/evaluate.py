"""
Evaluation: reusable helpers + the final test-set report.

Two roles in one module:
  1. Library  - `run_inference` and `compute_metrics` are imported by train.py
                to score the validation set after every epoch.
  2. Script   - `python -m src.evaluate` loads each experiment's best
                checkpoint (selected on VALIDATION Macro F1 during training),
                scores it on the TEST set once, and writes every report
                artifact into results/. It also exports the winner to saved_model/.

    checkpoints/A_*  ─┐                       ┌─► results/comparison.csv
                      ├─► test-set inference ─┼─► results/confusion_matrix.png
    checkpoints/B_*  ─┘                       ├─► results/training_curves.png
                                              ├─► results/evaluation/misclassified_examples.csv
                                              ├─► results/evaluation/classification_report_*.txt
                                              └─► saved_model/  (best experiment)
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, PreTrainedModel

from src.config import (
    CHECKPOINT_DIR,
    EXPERIMENTS,
    ID2LABEL,
    LABELS,
    EVALUATION_DIR,
    RESULTS_DIR,
    TRAINING_DIR,
    SAVED_MODEL_DIR,
    get_device,
)
from src.dataset import build_dataloaders, load_tokenizer
from src.visualize import plot_confusion_matrix, plot_training_curves


# --------------------------------------------------------------------------- #
# Reusable helpers
# --------------------------------------------------------------------------- #
@dataclass
class InferenceOutput:
    loss: float          # mean cross-entropy over the split
    labels: np.ndarray   # true class ids,       shape (N,)
    preds: np.ndarray    # predicted class ids,  shape (N,)
    probs: np.ndarray    # softmax probabilities, shape (N, 6)


def split_batch(batch: dict, device: torch.device) -> tuple[dict, torch.Tensor]:
    """Move a collated batch to `device` and separate model inputs from labels."""
    labels = batch["labels"].to(device)
    inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
    return inputs, labels


@torch.no_grad()  # no computation graph -> less memory, faster, and no weight updates
def run_inference(model: PreTrainedModel, loader: DataLoader, device: torch.device) -> InferenceOutput:
    """Score a whole split without updating weights."""
    model.eval()  # disables dropout so predictions are deterministic
    loss_fn = nn.CrossEntropyLoss(reduction="sum")  # sum, then divide by N for an exact mean

    total_loss = 0.0
    all_probs, all_labels = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        inputs, labels = split_batch(batch, device)
        logits = model(**inputs).logits
        total_loss += loss_fn(logits, labels).item()
        all_probs.append(torch.softmax(logits, dim=-1).cpu())
        all_labels.append(labels.cpu())

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    return InferenceOutput(
        loss=total_loss / len(labels),
        labels=labels,
        preds=probs.argmax(axis=1),
        probs=probs,
    )


def compute_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    """Accuracy + macro-averaged precision / recall / F1.

    Macro averaging gives each of the 6 emotions equal weight. The dataset is
    imbalanced (lots of joy/sadness, little surprise), so plain accuracy could
    look good while rare emotions are ignored - Macro F1 exposes that.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


# --------------------------------------------------------------------------- #
# Final test-set report
# --------------------------------------------------------------------------- #
def main() -> None:
    device = get_device()
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer()
    test_loader = build_dataloaders(tokenizer, splits=("test",))["test"]
    test_texts = test_loader.dataset.texts

    rows, histories, test_outputs = [], {}, {}
    for exp in EXPERIMENTS.values():
        ckpt_dir = CHECKPOINT_DIR / exp.name
        history_path = TRAINING_DIR / f"{exp.name}_history.json"
        if not ckpt_dir.exists() or not history_path.exists():
            print(f"[skip] {exp.name}: no checkpoint found - run `python -m src.train --experiment {exp.key}` first.")
            continue

        print(f"\nEvaluating {exp.name} on the test set ...")
        model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(device)
        output = run_inference(model, test_loader, device)
        history = json.loads(history_path.read_text())

        histories[exp.name] = history
        test_outputs[exp.name] = output
        rows.append({
            "experiment": exp.name,
            **{f"test_{k}": round(v, 4) for k, v in compute_metrics(output.labels, output.preds).items()},
            "best_val_macro_f1": round(history["best_val_macro_f1"], 4),
            "best_epoch": history["best_epoch"],
            "trainable_params": history["trainable_params"],
            "total_params": history["total_params"],
            "train_time_sec": round(history["train_seconds"], 1),
        })

        # Per-class precision/recall/F1 - shows WHICH emotions are hard.
        report = classification_report(output.labels, output.preds, target_names=LABELS, digits=4)
        (EVALUATION_DIR / f"classification_report_{exp.name}.txt").write_text(report)

    if not rows:
        raise SystemExit("No trained experiments found. Train first with `python -m src.train`.")

    # 1) Comparison table (Section 3.2 of the assignment).
    comparison = pd.DataFrame(rows)
    comparison.to_csv(RESULTS_DIR / "comparison.csv", index=False)
    print("\n" + comparison.to_string(index=False))

    # 2) Pick the winner by VALIDATION Macro F1 - the test set never drives decisions.
    best_name = max(histories, key=lambda name: histories[name]["best_val_macro_f1"])
    best_output = test_outputs[best_name]
    print(f"\nBest experiment (by validation Macro F1): {best_name}")

    # 3) Plots.
    cm = confusion_matrix(best_output.labels, best_output.preds, labels=list(range(len(LABELS))))
    plot_confusion_matrix(cm, LABELS, RESULTS_DIR / "confusion_matrix.png", title=f"Confusion matrix - {best_name} (test set)")
    plot_training_curves(histories, RESULTS_DIR / "training_curves.png")

    # 4) Misclassified test examples, most confident mistakes first (raw material
    #    for the qualitative error analysis in the README).
    wrong = np.flatnonzero(best_output.preds != best_output.labels)
    errors = pd.DataFrame({
        "text": [test_texts[i] for i in wrong],
        "true_label": [ID2LABEL[int(best_output.labels[i])] for i in wrong],
        "predicted_label": [ID2LABEL[int(best_output.preds[i])] for i in wrong],
        "confidence": best_output.probs[wrong].max(axis=1).round(4),
    }).sort_values("confidence", ascending=False)
    errors.to_csv(EVALUATION_DIR / "misclassified_examples.csv", index=False)
    print(f"Saved {len(errors)} misclassified test examples.")

    # 5) Export the winner (weights + tokenizer + id2label config) for the app.
    if SAVED_MODEL_DIR.exists():
        shutil.rmtree(SAVED_MODEL_DIR)
    shutil.copytree(CHECKPOINT_DIR / best_name, SAVED_MODEL_DIR)
    print(f"Exported {best_name} -> {SAVED_MODEL_DIR}")


if __name__ == "__main__":
    main()
