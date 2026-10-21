"""Language-detection model — artifact loading and inference.

The trained artifact is a single self-describing bundle that contains the
scikit-learn pipeline, the class labels and the training metadata.  Bundling
them together removes the drift risk of keeping a hard-coded label list
next to a pickle file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib

from .preprocessing import clean_text

__version__ = "0.2.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

#: Path to the trained bundle.  Override with ``MODEL_ARTIFACT`` to load a
#: model from outside the package (useful for mounted volumes / A/B tests).
ARTIFACT_PATH = Path(
    os.environ.get("MODEL_ARTIFACT", BASE_DIR / f"language_detector-{__version__}.joblib")
).expanduser()

#: The source dataset misspells two labels.  We keep the raw label for
#: traceability and expose a corrected spelling to API consumers.
LABEL_ALIASES: dict[str, str] = {
    "Portugeese": "Portuguese",
    "Sweedish": "Swedish",
}

with open(ARTIFACT_PATH, "rb") as _fh:
    _bundle: dict[str, Any] = joblib.load(_fh)

#: The fitted ``Pipeline(CountVectorizer -> MultinomialNB)``.
pipeline = _bundle["pipeline"]

#: Class labels, ordered exactly like ``pipeline.classes_`` indices.
classes: list[str] = list(_bundle["classes"])

#: Training metadata (metrics, sklearn version, dataset fingerprint, ...).
metadata: dict[str, Any] = dict(_bundle.get("metadata", {}))

#: scikit-learn version the artifact was trained with.  Loading a pickle
#: trained on a different version emits ``InconsistentVersionWarning``.
trained_with = metadata.get("sklearn_version", "unknown")


def canonical_label(label: str) -> str:
    """Return the correctly-spelled name for a dataset label."""
    return LABEL_ALIASES.get(label, label)


def predict(text: str) -> str:
    """Detect the language of ``text``.

    Args:
        text: Raw input text in any language.

    Returns:
        The detected language name, e.g. ``"English"``.

    Example:
        >>> predict("Ciao, come stai?")
        'Italian'
    """
    return predict_detailed(text)["language"]


def predict_detailed(text: str) -> dict[str, Any]:
    """Detect the language of ``text`` and return the full prediction record.

    Args:
        text: Raw input text in any language.

    Returns:
        A dict with the detected ``language``, the raw dataset ``label``,
        the numeric ``index``, and the model's ``confidence`` (max class
        probability, rounded to 4 decimals).

    Example:
        >>> predict_detailed("Hello there")["language"]
        'English'
    """
    cleaned = clean_text(text)
    index = int(pipeline.predict([cleaned])[0])
    raw_label = classes[index]

    confidence = float("nan")
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba([cleaned])[0]
        confidence = round(float(max(proba)), 4)

    return {
        "language": canonical_label(raw_label),
        "label": raw_label,
        "index": index,
        "confidence": confidence,
    }


def predict_batch(texts: list[str]) -> list[dict[str, Any]]:
    """Detect the language of every string in ``texts``.

    Args:
        texts: Raw input texts.

    Returns:
        One prediction record per input, in the same order.
    """
    if not texts:
        return []
    cleaned = [clean_text(t) for t in texts]
    indices = pipeline.predict(cleaned)

    confidences: list[float] = []
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(cleaned)
        confidences = [round(float(row.max()), 4) for row in proba]
    else:
        confidences = [float("nan")] * len(indices)

    results = []
    for index, confidence in zip(indices, confidences, strict=True):
        index = int(index)
        raw_label = classes[index]
        results.append(
            {
                "language": canonical_label(raw_label),
                "label": raw_label,
                "index": index,
                "confidence": confidence,
            }
        )
    return results


def supported_languages() -> list[str]:
    """Return the list of languages the model can detect."""
    return [canonical_label(c) for c in classes]
