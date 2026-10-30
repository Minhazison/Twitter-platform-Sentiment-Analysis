import pickle
import re
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ---------------------------------------------------------------- paths
BASE_DIR = Path(__file__).resolve().parent.parent
VECTORIZER_FILE = BASE_DIR / "Models" / "tfidf_vectorizer.pkl"
MODEL_FILE = BASE_DIR / "Models" / "sentiment_svm_model.pkl"

# ------------------------------------------- preprocessing (same as notebook!)
KEEP = {"no", "not", "nor", "never", "cannot", "nothing"}
STOP_WORDS = set(ENGLISH_STOP_WORDS) - KEEP
URL_RE = re.compile(r"https?://\S+|www\.\S+|pic\.twitter\.com/\S+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")


def clean_text(text):
    text = str(text).lower().replace("<unk>", " ")
    text = URL_RE.sub(" ", text)
    text = NON_ALNUM_RE.sub(" ", text)
    return " ".join(t for t in text.split() if t not in STOP_WORDS)


# ---------------------------------------------------------------- load
for f in (VECTORIZER_FILE, MODEL_FILE):
    if not f.exists():
        raise SystemExit(f"Missing file: {f}\nRun the notebook first so the Models folder is created.")

# Only load pickle files you created yourself - pickle can run arbitrary code.
with open(VECTORIZER_FILE, "rb") as f:
    vectorizer = pickle.load(f)
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)


def predict_texts(texts):
    X = vectorizer.transform([clean_text(t) for t in texts])
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)
    else:
        d = model.decision_function(X)
        e = np.exp(d - d.max(axis=1, keepdims=True))
        scores = e / e.sum(axis=1, keepdims=True)
    classes = list(model.classes_)
    out = []
    for text, row in zip(texts, scores):
        i = int(row.argmax())
        out.append({
            "text": text,
            "sentiment": classes[i],
            "confidence": round(float(row[i]), 4),
            "probabilities": {c: round(float(p), 4) for c, p in zip(classes, row)},
        })
    return out


# -------------------------------------------------------------- routes
app = Flask(__name__)


@app.get("/")
def home():
    return jsonify(message="Sentiment API is running",
                   endpoints=["/api/health", "/api/predict", "/api/predict/batch"])


@app.get("/api/health")
def health():
    return jsonify(status="ok", model=type(model).__name__)


@app.post("/api/predict")
def predict():
    body = request.get_json(silent=True) or {}
    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify(error='Send JSON like {"text": "your tweet"}'), 400
    if len(text) > 1000:
        return jsonify(error="Text too long (max 1000 characters)"), 400
    return jsonify(predict_texts([text])[0])


@app.post("/api/predict/batch")
def predict_batch():
    body = request.get_json(silent=True) or {}
    texts = body.get("texts")
    if (not isinstance(texts, list) or not texts or len(texts) > 100
            or not all(isinstance(t, str) and t.strip() for t in texts)):
        return jsonify(error='Send {"texts": [..]} with 1-100 non-empty strings'), 400
    return jsonify(results=predict_texts(texts))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)