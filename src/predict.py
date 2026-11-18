"""
Reusable inference: one string in -> predicted emotion + all six probabilities out.

    "I am so excited!"
          │ tokenizer (same one used in training, loaded from saved_model/)
          ▼
    input_ids + attention_mask ──► model.eval() + no_grad ──► 6 logits
                                                                │ softmax
                                                                ▼
    Prediction(label="joy", confidence=0.9x, probabilities={sadness: .., joy: .., ...})

Both the Streamlit app and the command line use this module, so preprocessing
and inference logic exist in exactly one place.

Usage:
    python -m src.predict "I am so excited about tomorrow!"
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import MAX_LENGTH, SAVED_MODEL_DIR, get_device


@dataclass(frozen=True)
class Prediction:
    label: str                       # top emotion, e.g. "joy"
    confidence: float                # its softmax probability
    probabilities: dict[str, float]  # all six emotions, in fixed label-id order


class EmotionClassifier:
    """Loads the exported model once; call `predict(text)` as often as you like."""

    def __init__(self, model_dir: str | Path = SAVED_MODEL_DIR, device: torch.device | None = None) -> None:
        model_dir = Path(model_dir)
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(
                f"No trained model found in '{model_dir}'. "
                "Run `python -m src.train` and then `python -m src.evaluate` to create it."
            )
        self.device = device or get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        # Labels come from the saved config.json, so they always match the weights.
        self.id2label: dict[int, str] = {int(i): name for i, name in self.model.config.id2label.items()}

    @torch.no_grad()
    def predict(self, text: str) -> Prediction:
        """Classify one text. Raises ValueError for blank input."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Please enter some text - the input is empty.")

        inputs = self.tokenizer(
            text.strip(), truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        ).to(self.device)
        logits = self.model(**inputs).logits[0]

        # .cpu() + float() -> identical plain-Python output on CPU and GPU.
        probs = torch.softmax(logits, dim=-1).cpu().tolist()
        probabilities = {self.id2label[i]: float(p) for i, p in enumerate(probs)}
        best_id = int(torch.argmax(logits).item())
        return Prediction(
            label=self.id2label[best_id],
            confidence=probabilities[self.id2label[best_id]],
            probabilities=probabilities,
        )


@lru_cache(maxsize=1)
def get_classifier() -> EmotionClassifier:
    """Process-wide singleton so repeated calls don't reload the model."""
    return EmotionClassifier()


def predict_emotion(text: str) -> Prediction:
    """Convenience function: classify a string with the exported model."""
    return get_classifier().predict(text)


if __name__ == "__main__":
    sentence = " ".join(sys.argv[1:]) or "I am so excited about tomorrow!"
    try:
        result = predict_emotion(sentence)
    except (ValueError, FileNotFoundError) as err:
        sys.exit(f"Error: {err}")

    print(f"Text:       {sentence}")
    print(f"Prediction: {result.label.upper()} ({result.confidence:.2%})")
    for label, prob in sorted(result.probabilities.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {label:<9} {prob:6.2%}  {'#' * round(prob * 40)}")
