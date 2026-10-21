"""FastAPI service that exposes the language-detection model over HTTP.

Run locally (from the repository root):

    uvicorn main:app --reload --port 8000

Then open http://127.0.0.1:8000/docs for the interactive Swagger UI.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model.model import __version__ as model_version
from model.model import metadata, predict_batch, predict_detailed, supported_languages

app = FastAPI(
    title="Language Detection API",
    description=(
        "Detect the language of a piece of text using a scikit-learn "
        "`CountVectorizer` + `MultinomialNB` pipeline trained on "
        f"{metadata.get('dataset', {}).get('rows', '10k+')} labelled samples "
        "across 17 languages."
    ),
    version=model_version,
    license_info={"name": "MIT"},
    contact={
        "name": "GitHub",
        "url": "https://github.com/Manohargov/Language-Detection-Docker_FastAPI_Render",
    },
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# ── Schemas ──────────────────────────────────────────────────────────────────
class TextIn(BaseModel):
    """A single piece of text to classify."""

    text: Annotated[str, Field(min_length=1, max_length=10_000, examples=["Ciao, come stai?"])]


class BatchTextIn(BaseModel):
    """A batch of texts to classify in one round-trip."""

    texts: Annotated[list[str], Field(min_length=1, max_length=100, examples=[["Hello", "Bonjour"]])]


class PredictionOut(BaseModel):
    """The detection result for one text."""

    language: str = Field(description="Detected language, e.g. 'English'", examples=["Italian"])
    label: str = Field(description="Raw dataset label for the detected language", examples=["Italian"])
    index: int = Field(description="Index of the detected class", examples=[8])
    confidence: float = Field(description="Max class probability, 0-1", examples=[0.98])


class BatchPredictionOut(BaseModel):
    """Detection results for a batch of texts."""

    predictions: list[PredictionOut]


class HealthOut(BaseModel):
    """Service and model health information."""

    health_check: str = Field(examples=["OK"])
    model_version: str = Field(examples=["0.2.0"])
    status: str = Field(examples=["ok"])
    languages: list[str]
    metrics: dict[str, Any]
    sklearn_version: str


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", response_model=HealthOut, tags=["system"], summary="Service & model status")
@app.get("/health", response_model=HealthOut, tags=["system"], summary="Service & model status")
def health() -> HealthOut:
    """Return service health, model version, supported languages and training metrics."""
    return HealthOut(
        health_check="OK",
        model_version=model_version,
        status="ok",
        languages=supported_languages(),
        metrics=metadata.get("metrics", {}),
        sklearn_version=metadata.get("sklearn_version", "unknown"),
    )


@app.post(
    "/predict",
    response_model=PredictionOut,
    tags=["prediction"],
    summary="Detect the language of a text",
)
def predict(payload: TextIn) -> PredictionOut:
    """Classify ``payload.text`` and return the detected language.

    Raises:
        HTTPException: 422 if the text is empty or too long.
    """
    if not payload.text.strip():
        raise HTTPException(status_code=422, detail="Input text must not be empty or whitespace only.")
    return PredictionOut(**predict_detailed(payload.text))


@app.post(
    "/predict/batch",
    response_model=BatchPredictionOut,
    tags=["prediction"],
    summary="Detect the language of several texts at once",
)
def predict_batch_endpoint(payload: BatchTextIn) -> BatchPredictionOut:
    """Classify up to 100 texts in a single request."""
    if not any(t.strip() for t in payload.texts):
        raise HTTPException(status_code=422, detail="At least one text must not be empty.")
    return BatchPredictionOut(predictions=[PredictionOut(**p) for p in predict_batch(payload.texts)])
