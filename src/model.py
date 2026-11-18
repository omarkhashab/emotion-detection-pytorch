"""
Model construction and layer freezing.

DistilBertForSequenceClassification, as loaded by AutoModelForSequenceClassification:

    input_ids ──► embeddings ──► transformer.layer[0] ... layer[5] ──► [CLS] vector
                 └──────────────── model.distilbert (pretrained) ───┘        │
                                                                            ▼
                                       pre_classifier (768→768) ─► ReLU ─► dropout
                                                                            ▼
                                              classifier (768→6)  ─►  6 logits
                                       └──── classification head (NEW, random init) ───┘

Experiment A freezes everything inside `model.distilbert`.
Experiment B additionally unfreezes the last two blocks: layer[4] and layer[5].
"""

from __future__ import annotations

from torch import nn
from transformers import AutoModelForSequenceClassification, PreTrainedModel

from src.config import ID2LABEL, LABEL2ID, MODEL_NAME, NUM_LABELS


def build_model(num_unfrozen_layers: int, model_name: str = MODEL_NAME) -> PreTrainedModel:
    """Load pretrained DistilBERT with a fresh 6-class head and freeze the backbone.

    Args:
        num_unfrozen_layers: how many of the final Transformer blocks to train.
            0 -> feature extraction (experiment A); 2 -> partial fine-tuning (B).

    Every call starts from the original pretrained weights, so experiment B never
    inherits anything from experiment A.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,  # stored in config.json -> the saved model knows its labels
        label2id=LABEL2ID,
    )

    # 1) Freeze the whole pretrained backbone (embeddings + all 6 blocks).
    for param in model.distilbert.parameters():
        param.requires_grad = False

    # 2) Unfreeze the last N Transformer blocks. They hold the most task-specific,
    #    "high-level" features, so they benefit most from adaptation.
    if num_unfrozen_layers > 0:
        for block in model.distilbert.transformer.layer[-num_unfrozen_layers:]:
            for param in block.parameters():
                param.requires_grad = True

    # The head (pre_classifier + classifier) lives outside `model.distilbert`,
    # so it was never frozen and is always trained.
    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts - our proxy for training cost."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
