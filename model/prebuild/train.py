"""Reproducible training script for the language-detection model.

Trains a ``CountVectorizer(char_wb 2-4) -> MultinomialNB`` pipeline on the
Kaggle *Language Detection* dataset and writes a single self-describing
artifact that bundles the pipeline, the class labels and the training
metadata.

Usage (from the repository root):

    python model/prebuild/train.py

Output:

    model/language_detector-0.2.0.joblib
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path

# Allow `python model/prebuild/train.py` from the repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib  # noqa: E402
import sklearn  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.naive_bayes import MultinomialNB  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402

from model.preprocessing import clean_text  # noqa: E402

MODEL_VERSION = "0.2.0"
DEFAULT_DATASET = Path(__file__).resolve().parent / "Language Detection.csv"
DEFAULT_OUTPUT = REPO_ROOT / "model" / f"language_detector-{MODEL_VERSION}.joblib"

#: Character n-grams beat word unigrams on this task, especially for scripts
#: that do not separate words with spaces (Hindi, Kannada, Tamil, ...).
VECTORIZER_PARAMS = {
    "analyzer": "char_wb",
    "ngram_range": (2, 4),
    "min_df": 1,
    "lowercase": True,
}

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_dataset(path: Path) -> tuple[list[str], list[str]]:
    """Read the CSV dataset and return ``(texts, labels)``."""
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or "Text" not in rows[0] or "Language" not in rows[0]:
        raise ValueError(f"Unexpected CSV schema in {path}. Expected columns 'Text' and 'Language'.")
    texts = [r["Text"] for r in rows]
    labels = [r["Language"] for r in rows]
    return texts, labels


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of a file (used to fingerprint the dataset)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def train(dataset_path: Path, output_path: Path) -> dict:
    """Train the model, evaluate it and write the artifact."""
    raw_texts, raw_labels = load_dataset(dataset_path)
    texts = [clean_text(t) for t in raw_texts]

    encoder = LabelEncoder()
    y = encoder.fit_transform(raw_labels)
    classes = [str(c) for c in encoder.classes_]

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = Pipeline(
        [
            ("vectorizer", CountVectorizer(**VECTORIZER_PARAMS)),
            ("multinominalNB", MultinomialNB()),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted")), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }

    per_class = {}
    report = classification_report(y_test, y_pred, target_names=classes, zero_division=0, output_dict=True)
    for cls in classes:
        per_class[cls] = {
            "precision": round(float(report[cls]["precision"]), 4),
            "recall": round(float(report[cls]["recall"]), 4),
            "f1": round(float(report[cls]["f1-score"]), 4),
            "support": int(report[cls]["support"]),
        }

    metadata = {
        "model_version": MODEL_VERSION,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "vectorizer": {k: list(v) if isinstance(v, tuple) else v for k, v in VECTORIZER_PARAMS.items()},
        "algorithm": "MultinomialNB",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "metrics": metrics,
        "per_class": per_class,
        "classes": classes,
        "dataset": {
            "name": dataset_path.name,
            "rows": len(raw_texts),
            "class_distribution": dict(Counter(raw_labels)),
            "sha256": file_sha256(dataset_path),
        },
    }

    bundle = {"pipeline": pipeline, "classes": classes, "metadata": metadata}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path, compress=3)

    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the language-detection model.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Path to the CSV dataset.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Where to write the artifact.")
    args = parser.parse_args()

    if not args.dataset.exists():
        print(
            f"Dataset not found: {args.dataset}\n"
            "Download it from https://www.kaggle.com/datasets/basilb2s/language-detection "
            "and place 'Language Detection.csv' in model/prebuild/.",
            file=sys.stderr,
        )
        return 1

    metadata = train(args.dataset, args.output)
    metrics = metadata["metrics"]

    print("\n" + "=" * 62)
    print(f"  MODEL CARD — language-detector {metadata['model_version']}")
    print("=" * 62)
    print(f"  Dataset        : {metadata['dataset']['name']} ({metadata['dataset']['rows']} rows)")
    print(f"  Classes        : {len(metadata['classes'])}")
    print(f"  Vectorizer     : char_wb {metadata['vectorizer']['ngram_range']}")
    print(f"  Algorithm      : {metadata['algorithm']}")
    print(f"  scikit-learn   : {metadata['sklearn_version']}")
    print("-" * 62)
    print(f"  Accuracy       : {metrics['accuracy']:.4f}")
    print(f"  F1 (weighted)  : {metrics['f1_weighted']:.4f}")
    print(f"  F1 (macro)     : {metrics['f1_macro']:.4f}")
    print(f"  Train / test   : {metrics['train_samples']} / {metrics['test_samples']}")
    print("-" * 62)
    print(f"  Artifact       : {args.output}  ({args.output.stat().st_size / 1_000_000:.2f} MB)")
    print("=" * 62 + "\n")

    print(json.dumps(metadata["per_class"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
