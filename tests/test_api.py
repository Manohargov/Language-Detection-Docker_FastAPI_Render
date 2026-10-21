"""API and model tests.

Run with:  pytest -q
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from main import app
from model.model import __version__, predict, predict_detailed, supported_languages

client = TestClient(app)


# ── System endpoints ─────────────────────────────────────────────────────────
def test_root_returns_health():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["health_check"] == "OK"
    assert body["status"] == "ok"
    assert body["model_version"] == __version__


def test_health_alias_is_available():
    assert client.get("/health").status_code == 200


def test_health_lists_supported_languages():
    body = client.get("/health").json()
    assert len(body["languages"]) == 17
    assert "English" in body["languages"]
    assert "Portuguese" in body["languages"]  # corrected spelling


def test_openapi_schema_is_valid():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "Language Detection API"
    assert "/predict" in schema["paths"]


# ── Prediction endpoint ─────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Ciao, come stai?", "Italian"),
        ("Hello, how are you today?", "English"),
        ("Bonjour tout le monde", "French"),
        ("Guten Tag, wie geht es dir?", "German"),
        ("Hola, ¿cómo estás?", "Spanish"),
        ("Привет, как дела?", "Russian"),
        ("Γειά σου κόσμε", "Greek"),
        ("مرحبا بالعالم", "Arabic"),
    ],
)
def test_predict_detects_expected_language(text, expected):
    response = client.post("/predict", json={"text": text})
    assert response.status_code == 200
    assert response.json()["language"] == expected


def test_predict_response_shape():
    body = client.post("/predict", json={"text": "Hello world"}).json()
    assert set(body) == {"language", "label", "index", "confidence"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["index"], int)


def test_predict_strips_punctuation_and_digits():
    """The cleaning step must not change the detected language."""
    plain = client.post("/predict", json={"text": "Hello there friend"}).json()
    noisy = client.post("/predict", json={"text": "Hello [there] 123 !@# friend"}).json()
    assert plain["language"] == noisy["language"] == "English"


def test_predict_rejects_empty_text():
    assert client.post("/predict", json={"text": ""}).status_code == 422


def test_predict_rejects_whitespace_only_text():
    assert client.post("/predict", json={"text": "   "}).status_code == 422


def test_predict_rejects_missing_field():
    assert client.post("/predict", json={}).status_code == 422


def test_predict_rejects_non_string():
    assert client.post("/predict", json={"text": 12345}).status_code == 422


# ── Batch endpoint ───────────────────────────────────────────────────────────
def test_batch_predict():
    response = client.post(
        "/predict/batch",
        json={"texts": ["Hello there", "Ciao come stai", "Bonjour tout le monde"]},
    )
    assert response.status_code == 200
    languages = [p["language"] for p in response.json()["predictions"]]
    assert languages == ["English", "Italian", "French"]


def test_batch_predict_empty_list_is_rejected():
    assert client.post("/predict/batch", json={"texts": []}).status_code == 422


# ── Model layer ──────────────────────────────────────────────────────────────
def test_model_predict_helper():
    assert predict("Ciao, come stai?") == "Italian"


def test_predict_detailed_confidence_is_populated():
    record = predict_detailed("Hello, how are you?")
    assert record["language"] == "English"
    assert 0.0 <= record["confidence"] <= 1.0


def test_supported_languages_are_canonical():
    langs = supported_languages()
    assert "Portuguese" in langs
    assert "Swedish" in langs
    assert "Portugeese" not in langs  # dataset typo is corrected
