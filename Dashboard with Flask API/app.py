import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from dashboard import DASHBOARD_HTML

# ---------------------------------------------------------------- paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "Models"
VECTORIZER_FILE = MODELS_DIR / "tfidf_vectorizer.pkl"
MODEL_FILE = MODELS_DIR / "sentiment_svm_model.pkl"
METRICS_FILE = MODELS_DIR / "metrics.json"
DATA_PATH = next(BASE_DIR.glob("twitter_training - twitter_training.csv"), BASE_DIR / "twitter_training - twitter_training.csv")

LABELS = ["Positive", "Negative", "Neutral", "Irrelevant"]
COLUMNS = ["ID", "Brand", "Sentiment", "Text"]

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
for f in (VECTORIZER_FILE, MODEL_FILE, METRICS_FILE, DATA_PATH):
    if not f.exists():
        raise SystemExit(f"Missing file: {f}\nRun the notebook first so the Models folder is created.")

# Only load pickle files you created yourself - pickle can run arbitrary code.
with open(VECTORIZER_FILE, "rb") as f:
    vectorizer = pickle.load(f)
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)
with open(METRICS_FILE) as f:
    metrics_data = json.load(f)

df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS)
df = df.dropna(subset=["Text"])
df = df[df["Sentiment"].isin(LABELS)].copy()
df["Text"] = df["Text"].astype(str)
df = df.reset_index(drop=True)

by_brand = pd.crosstab(df["Brand"], df["Sentiment"]).reindex(columns=LABELS, fill_value=0)
dist = df["Sentiment"].value_counts().reindex(LABELS, fill_value=0)
stats_data = {
    "total": int(len(df)),
    "brand_count": int(df["Brand"].nunique()),
    "distribution": {k: int(v) for k, v in dist.items()},
    "by_brand": json.loads(by_brand.to_json(orient="index")),
    "brands": sorted(df["Brand"].unique().tolist()),
}


# ------------------------------------------------------------- helpers
def predict_texts(texts):
    X = vectorizer.transform([clean_text(t) for t in texts])
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)
    else:  # LinearSVC has no probabilities: softmax over the margins
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


def net_sentiment(counts):
    total = sum(counts.values())
    return round((counts["Positive"] - counts["Negative"]) / total, 4) if total else 0.0


# -------------------------------------------------------------- routes
app = Flask(__name__)


@app.get("/")
def dashboard():
    return Response(DASHBOARD_HTML, mimetype="text/html")


@app.get("/api/health")
def health():
    return jsonify(status="ok", model=metrics_data["best_model"], rows=stats_data["total"])


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


@app.get("/api/stats")
def stats():
    return jsonify(stats_data)


@app.get("/api/metrics")
def metrics():
    return jsonify(metrics_data)


@app.get("/api/brands")
def brands():
    return jsonify(stats_data["brands"])


@app.get("/api/brand")
def brand():
    name = request.args.get("name", "")
    counts = stats_data["by_brand"].get(name)
    if counts is None:
        return jsonify(error=f"Unknown brand: {name}"), 404
    return jsonify(brand=name, total=int(sum(counts.values())),
                   distribution=counts, net_sentiment=net_sentiment(counts))


@app.get("/api/samples")
def samples():
    d = df
    b, s = request.args.get("brand"), request.args.get("sentiment")
    try:
        limit = max(1, min(int(request.args.get("limit", 10)), 100))
    except ValueError:
        return jsonify(error="limit must be an integer"), 400
    if b:
        d = d[d["Brand"] == b]
    if s:
        d = d[d["Sentiment"] == s]
    if d.empty:
        return jsonify(samples=[])
    d = d.sample(min(limit, len(d)))
    return jsonify(samples=d[["Brand", "Sentiment", "Text"]].to_dict(orient="records"))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)