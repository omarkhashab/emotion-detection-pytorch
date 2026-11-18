"""
Central configuration: every constant the project depends on lives here.

Why a single config module?
    Training, evaluation, inference and the Streamlit app must all agree on the
    same label mapping, model name, sequence length and file paths. Defining them
    once prevents subtle bugs such as the app using a different label order than
    the model was trained with.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Model & data identifiers
# --------------------------------------------------------------------------- #
MODEL_NAME = "distilbert-base-uncased"
DATASET_NAME = "dair-ai/emotion"
DATASET_CONFIG = "split"  # predefined 16k / 2k / 2k split (the other config is "unsplit")

# Label mapping required by the assignment (matches the dataset card).
LABELS: list[str] = ["sadness", "joy", "love", "anger", "fear", "surprise"]
ID2LABEL: dict[int, str] = dict(enumerate(LABELS))
LABEL2ID: dict[str, int] = {label: idx for idx, label in ID2LABEL.items()}
NUM_LABELS = len(LABELS)

# --------------------------------------------------------------------------- #
# Shared hyperparameters
# --------------------------------------------------------------------------- #
MAX_LENGTH = 128      # tokens; longer texts are truncated (the EDA shows almost none are)
BATCH_SIZE = 16       # reduce to 8 if your GPU runs out of memory
SEED = 42
MAX_GRAD_NORM = 1.0   # gradient clipping keeps updates stable
WEIGHT_DECAY = 0.01   # standard AdamW regularisation for Transformers

# --------------------------------------------------------------------------- #
# Paths (resolved relative to the repository root, so commands work from anywhere)
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"          # metrics, plots, histories (committed)
# Headline deliverables (comparison.csv, confusion_matrix.png, training_curves.png)
# live directly in RESULTS_DIR; supporting files are grouped by pipeline stage:
EDA_DIR = RESULTS_DIR / "eda"                # EDA plots + summary          (src/eda.py)
TRAINING_DIR = RESULTS_DIR / "training"      # per-epoch histories           (src/train.py)
EVALUATION_DIR = RESULTS_DIR / "evaluation"  # per-class reports, errors     (src/evaluate.py)
LOGS_DIR = RESULTS_DIR / "logs"              # console output of each run
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"   # best checkpoint per experiment (git-ignored)
SAVED_MODEL_DIR = ROOT_DIR / "saved_model"  # the single exported model used by the app


# --------------------------------------------------------------------------- #
# Experiment definitions (Section 2.1 of the assignment)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExperimentConfig:
    """Everything that differs between experiment A and experiment B."""

    key: str                  # short id used on the command line ("A" / "B")
    name: str                 # folder / file name
    description: str
    num_unfrozen_layers: int  # how many of the 6 Transformer blocks are trainable
    learning_rate: float
    epochs: int = 3
    batch_size: int = BATCH_SIZE


EXPERIMENTS: dict[str, ExperimentConfig] = {
    "A": ExperimentConfig(
        key="A",
        name="A_feature_extraction",
        description="Frozen DistilBERT backbone, trainable classification head",
        num_unfrozen_layers=0,
        learning_rate=1e-4,
    ),
    "B": ExperimentConfig(
        key="B",
        name="B_partial_finetuning",
        description="Last 2 Transformer layers + classification head trainable",
        num_unfrozen_layers=2,
        learning_rate=2e-5,
    ),
}


# --------------------------------------------------------------------------- #
# Reproducibility & hardware helpers
# --------------------------------------------------------------------------- #
def set_seed(seed: int = SEED) -> None:
    """Seed every random number generator we use so runs are repeatable.

    Seeding right before building a model also means experiments A and B start
    with an *identical* randomly-initialised classification head, which makes
    their comparison fairer.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Use the GPU when available, otherwise fall back to the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
