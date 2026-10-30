import json
import pickle
import re
from pathlib import Path
from flask import Flask, render_template_string, request
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

app = Flask(__name__)

KEEP = {"no", "not", "nor", "never", "cannot", "nothing"}
STOP_WORDS = set(ENGLISH_STOP_WORDS) - KEEP
URL_RE = re.compile(r"https?://\S+|www\.\S+|pic\.twitter\.com/\S+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")

def clean_text(text: str) -> str:
    text = str(text).lower().replace("<unk>", " ")
    text = URL_RE.sub(" ", text)
    text = NON_ALNUM_RE.sub(" ", text)
    return " ".join(t for t in text.split() if t not in STOP_WORDS)

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

def find_file(filename: str) -> Path:
    for folder in [BASE_DIR / "Models", ROOT_DIR / "Models", BASE_DIR, ROOT_DIR]:
        p = folder / filename
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find {filename}")

with open(find_file("tfidf_vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)
with open(find_file("sentiment_svm_model.pkl"), "rb") as f:
    model = pickle.load(f)

metrics_path = find_file("metrics.json")
with open(metrics_path, "r", encoding="utf-8") as f:
    metrics = json.load(f)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Twitter Sentiment Analysis Dashboard</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f0f2f5; margin: 0; padding: 30px; }
    .card { max-width: 760px; margin: auto; background: white; padding: 28px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); }
    h1 { color: #1da1f2; margin-top: 0; }
    .metrics { display: flex; gap: 12px; margin-bottom: 20px; }
    .metric { background: #e8f5fe; color: #1da1f2; padding: 8px 16px; border-radius: 8px; font-weight: bold; }
    textarea { width: 100%; height: 90px; padding: 12px; border: 1px solid #ccd6dd; border-radius: 8px; font-size: 15px; box-sizing: border-box; }
    button { background: #1da1f2; color: white; border: none; padding: 10px 20px; border-radius: 20px; font-size: 16px; font-weight: bold; cursor: pointer; margin-top: 10px; }
    button:hover { background: #0c85d0; }
    .result { margin-top: 24px; padding: 16px; border-radius: 8px; }
    .Positive { background: #d4edda; color: #155724; border-left: 6px solid #28a745; }
    .Negative { background: #f8d7da; color: #721c24; border-left: 6px solid #dc3545; }
    .Neutral { background: #cce5ff; color: #004085; border-left: 6px solid #007bff; }
    .Irrelevant { background: #e2e3e5; color: #383d41; border-left: 6px solid #6c757d; }
  </style>
</head>
<body>
  <div class="card">
    <h1>🐦 Twitter Sentiment Analysis Dashboard</h1>
    <div class="metrics">
      <div class="metric">Model: {{ metrics['best_model'] }}</div>
      <div class="metric">Accuracy: 71.95%</div>
      <div class="metric">Macro F1: 70.79%</div>
    </div>
    <form method="POST">
      <label><b>Enter Tweet text:</b></label><br><br>
      <textarea name="tweet" placeholder="Type a tweet to analyze...">{{ tweet }}</textarea><br>
      <button type="submit">Analyze Sentiment</button>
    </form>
    {% if sentiment %}
    <div class="result {{ sentiment }}">
      <h2>Predicted Sentiment: {{ sentiment }}</h2>
      <p><b>Cleaned text:</b> <code>{{ cleaned }}</code></p>
    </div>
    {% endif %}
  </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def dashboard():
    sentiment = None
    cleaned = ""
    tweet = ""
    if request.method == "POST":
        tweet = request.form.get("tweet", "")
        if tweet.strip():
            cleaned = clean_text(tweet)
            X = vectorizer.transform([cleaned])
            sentiment = str(model.predict(X)[0])
    return render_template_string(HTML_TEMPLATE, tweet=tweet, sentiment=sentiment, cleaned=cleaned, metrics=metrics)

if __name__ == "__main__":
   app.run(host="127.0.0.1", port=5001, debug=False)