"""
All plotting in one place, so every figure shares the same look.

Colour rules used here:
  * Each entity keeps one colour everywhere: experiment A / train split = blue,
    experiment B / validation split = orange, test split = aqua.
  * Text and axes use neutral ink, never the series colour.
  * The confusion matrix uses a single-hue (blue) light->dark scale, because
    it encodes magnitude, not categories.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files only - works on servers / Colab without a display

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Validated categorical order (colour-blind safe for adjacent pairs).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK_SECONDARY, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#86b6ef", "#2a78d6", "#0d366b"])

SPLIT_COLORS = {"train": BLUE, "validation": ORANGE, "test": AQUA}
EXPERIMENT_COLORS = [BLUE, ORANGE]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "legend.frameon": False,
    "font.size": 10,
})


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved figure -> {path}")


def plot_class_distribution(counts: dict[str, dict[str, int]], labels: list[str], path: Path) -> None:
    """Grouped bars: share of each emotion per split.

    Showing percentages (not raw counts) lets us compare a 16k split with a
    2k split and confirm all three splits share the same imbalance.
    """
    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.5))

    for i, (split, split_counts) in enumerate(counts.items()):
        total = sum(split_counts.values())
        shares = [100 * split_counts.get(label, 0) / total for label in labels]
        bars = ax.bar(x + (i - 1) * width, shares, width * 0.92, label=split, color=SPLIT_COLORS[split])
        if split == "train":  # label one series only, to avoid clutter
            ax.bar_label(bars, fmt="%.1f%%", fontsize=8, color=INK_SECONDARY, padding=2)

    ax.set_xticks(x, [label.capitalize() for label in labels])
    ax.set_ylabel("Share of split (%)")
    ax.set_title("Class distribution per split")
    ax.grid(axis="x", visible=False)
    ax.legend(title="Split")
    _save(fig, path)


def plot_token_lengths(lengths: list[int], max_length: int, path: Path) -> None:
    """Histogram of tokens per training example, with the truncation limit marked."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lengths, bins=40, color=BLUE, edgecolor=SURFACE, linewidth=0.8)
    ax.axvline(max_length, color=INK_SECONDARY, linestyle="--", linewidth=1.5)
    ax.text(max_length, ax.get_ylim()[1] * 0.9, f"  max_length = {max_length}", color=INK_SECONDARY)
    ax.axvline(np.percentile(lengths, 95), color=MUTED, linestyle=":", linewidth=1.5)
    ax.set_xlabel("Tokens per text (incl. [CLS] and [SEP])")
    ax.set_ylabel("Number of texts")
    ax.set_title("Token length distribution - train split (dotted line = 95th percentile)")
    _save(fig, path)


def plot_training_curves(histories: dict[str, dict], path: Path) -> None:
    """Left: train (solid) vs validation (dashed) loss. Right: validation Macro F1."""
    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(11, 4.2))

    for color, (name, history) in zip(EXPERIMENT_COLORS, histories.items()):
        epochs = [e["epoch"] for e in history["epochs"]]
        ax_loss.plot(epochs, [e["train_loss"] for e in history["epochs"]], color=color, lw=2, marker="o", ms=6, label=f"{name} - train")
        ax_loss.plot(epochs, [e["val_loss"] for e in history["epochs"]], color=color, lw=2, ls="--", marker="s", ms=6, label=f"{name} - validation")
        ax_f1.plot(epochs, [e["val_macro_f1"] for e in history["epochs"]], color=color, lw=2, marker="o", ms=6, label=name)
        # Mark the epoch that was saved as the checkpoint.
        best = history["best_epoch"]
        ax_f1.scatter([best], [history["best_val_macro_f1"]], s=140, facecolors="none", edgecolors=INK, linewidths=1.5, zorder=3)

    for ax in (ax_loss, ax_f1):
        ax.set_xlabel("Epoch")
        ax.xaxis.get_major_locator().set_params(integer=True)
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.set_title("Training vs validation loss")
    ax_loss.legend(fontsize=8)
    ax_f1.set_ylabel("Validation Macro F1")
    ax_f1.set_title("Validation Macro F1 (circle = saved checkpoint)")
    ax_f1.legend(fontsize=8)
    _save(fig, path)


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], path: Path, title: str) -> None:
    """Confusion matrix coloured by row-normalised recall, annotated with raw counts.

    Rows = true emotion, columns = predicted emotion. Row normalisation makes
    rare classes (e.g. surprise) as readable as frequent ones.
    """
    row_share = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(row_share, cmap=SEQUENTIAL_BLUE, vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Share of true class")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]}\n{row_share[i, j]:.0%}",
                ha="center", va="center", fontsize=8,
                color="#ffffff" if row_share[i, j] > 0.5 else INK,
            )

    ticks = np.arange(len(labels))
    names = [label.capitalize() for label in labels]
    ax.set_xticks(ticks, names, rotation=30, ha="right")
    ax.set_yticks(ticks, names)
    ax.set_xlabel("Predicted emotion")
    ax.set_ylabel("True emotion")
    ax.set_title(title)
    ax.grid(False)
    _save(fig, path)
