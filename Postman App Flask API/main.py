import json
import pickle
import re
from pathlib import Path
from flask import Flask, jsonify, request
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

app = Flask(__name__)

# Preprocessing from your notebook
KEEP = {"no", "not", "nor", "never", "cannot", "nothing"}
STOP_WORDS = set(ENGLISH_STOP_WORDS) - KEEP
URL_RE = re.compile(r"https?://\S+|www\.\S+|pic\.twitter\.com/\S+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")

def clean_text(text: str) -> str:
    text = str(text).lower().replace("<unk>", " ")
    text = URL_RE.sub(" ", text)
    text = NON_ALNUM_RE.sub(" ", text)
    return " ".join(t for t in text.split() if t not in STOP_WORDS)

# Locate Models folder (in current folder, parent folder, or root)
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

def find_file(filename: str) -> Path:
    for folder in [BASE_DIR / "Models", ROOT_DIR / "Models", BASE_DIR, ROOT_DIR]:
        p = folder / filename
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find {filename}")

vec_path = find_file("tfidf_vectorizer.pkl")
model_path = find_file("sentiment_svm_model.pkl")

with open(vec_path, "rb") as f:
    vectorizer = pickle.load(f)
with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running", "model": "Linear SVC", "classes": list(model.classes_)})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    tweet = data.get("text", "")
    if not tweet.strip():
        return jsonify({"error": "Tweet text is empty"}), 400

    cleaned = clean_text(tweet)
    X = vectorizer.transform([cleaned])
    prediction = str(model.predict(X)[0])

    return jsonify({
        "input_text": tweet,
        "cleaned_text": cleaned,
        "sentiment": prediction
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)