# 🐦 Twitter Platform Sentiment Analysis Dashboard
### A Feature-Rich, API-Driven Architecture for Social Media Sentiment Classification

An end-to-end Natural Language Processing (NLP) and Machine Learning system that classifies Twitter tweets into four sentiment categories: **Positive**, **Negative**, **Neutral**, and **Irrelevant**. 

The project features a negation-aware text cleaning pipeline, TF-IDF vectorization, benchmarked scikit-learn classifiers (Linear SVC, Logistic Regression, Multinomial Naive Bayes), a dedicated **Flask REST API** for Postman testing, and an interactive **Web Dashboard**.

---

## 📌 Project Overview

- **Dataset**: Over 74,000 labeled tweets (`twitter_training - twitter_training.csv`) spanning global brands, tech entities, and gaming communities.
- **Preprocessing**: Cleans raw social media text by removing URLs, special tokens (`<unk>`), and non-alphanumeric noise while **preserving essential negation words** (`not`, `never`, `no`, `cannot`, etc.) to prevent polarity reversal.
- **Feature Extraction**: Sub-linear TF-IDF vectorization with unigrams and bigrams (`ngram_range=(1, 2)`), capped at 5,000 top features.
- **Model Benchmark**: Linear SVC outperformed baseline models with **71.95% accuracy** and **70.79% Macro F1**.
- **Deployment Architecture**:
  1. **Postman App Flask API**: Lightweight RESTful microservice for automated endpoint testing and integration.
  2. **Dashboard with Flask API**: Web-based user interface for real-time sentiment predictions.

---

## 📊 Model Performance & Benchmark

All models were evaluated on an independent test set of 12,097 stratified tweets:

| Model | Accuracy | Macro F1 | Status |
| :--- | :---: | :---: | :---: |
| **Linear Support Vector Classifier (Linear SVC)** | **71.95%** | **70.79%** | **Selected Best Model** |
| Logistic Regression | 69.42% | 67.95% | Baseline |
| Multinomial Naive Bayes | 63.81% | 61.18% | Baseline |

### Linear SVC Detailed Classification Report
| Sentiment Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Positive** | 0.70 | 0.75 | 0.72 | 3,219 |
| **Negative** | 0.75 | 0.80 | 0.78 | 3,707 |
| **Neutral** | 0.72 | 0.68 | 0.70 | 3,036 |
| **Irrelevant** | 0.68 | 0.59 | 0.63 | 2,135 |
| **Weighted Average** | **0.72** | **0.72** | **0.72** | **12,097** |

---

## 📁 Repository Structure

```text
Twitter-platform-Sentiment-Analysis/
├── Dashboard with Flask API/
│   └── main.py                     # Interactive Web Dashboard (Port 5001)
├── Models/
│   ├── metrics.json                # Stored model evaluation benchmarks
│   ├── sentiment_svm_model.pkl     # Trained Linear SVC model
│   └── tfidf_vectorizer.pkl        # Fitted TF-IDF Vectorizer (5,000 features)
├── Postman App Flask API/
│   └── main.py                     # Flask REST API for Postman integration (Port 5000)
├── requirements.txt                # Python package dependencies
├── Twitter_platform_Sentiment_Analysis_Dashboard_A_Feature_Rich_API_Driven_Architecture.ipynb
├── twitter_training - twitter_training.csv
└── README.md
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9, 3.10, or 3.11
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/mahmudulhasantechnology-cmyk/Twitter-platform-Sentiment-Analysis-Dashboard-A-Feature-Rich-API-Driven-Architecture-.git
cd Twitter-platform-Sentiment-Analysis-Dashboard-A-Feature-Rich-API-Driven-Architecture-
```

### 3. Create a Virtual Environment & Install Dependencies
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

---

## 🛠️ Running the Services

### Option A: Launch the Web Dashboard
Navigate to the dashboard directory and run `main.py`:
```bash
cd "Dashboard with Flask API"
python main.py
```
Open your browser and navigate to:
```text
http://127.0.0.1:5001
```

---

### Option B: Launch the Postman Flask REST API
Open a separate terminal window, navigate to the API directory, and run `main.py`:
```bash
cd "Postman App Flask API"
python main.py
```
The server will start on:
```text
http://127.0.0.1:5000
```

---

## 📬 Testing the API via Postman

### 1. Health & Model Status
- **Method**: `GET`
- **Endpoint**: `http://127.0.0.1:5000/health`
- **Sample Response**:
  ```json
  {
    "classes": [
      "Irrelevant",
      "Negative",
      "Neutral",
      "Positive"
    ],
    "model": "Linear SVC",
    "status": "running"
  }
  ```

### 2. Predict Tweet Sentiment
- **Method**: `POST`
- **Endpoint**: `http://127.0.0.1:5000/predict`
- **Headers**:
  - `Content-Type: application/json`
- **Request Body (raw JSON)**:
  ```json
  {
    "text": "I absolutely love this game! The new graphics are amazing."
  }
  ```
- **Response**:
  ```json
  {
    "input_text": "I absolutely love this game! The new graphics are amazing.",
    "cleaned_text": "absolutely love game new graphics amazing",
    "sentiment": "Positive"
  }
  ```

---

## 💡 Tech Stack
- **Language**: Python
- **Machine Learning & NLP**: Scikit-Learn, NumPy, Pandas, Matplotlib
- **Web Frameworks**: Flask, Werkzeug, Jinja2
- **API Testing**: Postman

---

## 👤 Author
Developed by **Mahmudul Hasan**  
GitHub: [@mahmudulhasantechnology-cmyk](https://www.google.com/search?q=https://github.com/mahmudulhasantechnology-cmyk)
