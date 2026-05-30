# 💊 DrugSense — Drug Review Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.10+-3572A5?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.3-000000?style=for-the-badge&logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-2E7D32?style=for-the-badge)

> **NLP-powered Drug Recommendation System** — Classifies patient drug reviews across positive, neutral, and negative sentiment using a Bidirectional LSTM model, with aspect-level analysis, drug recommendations, safety warnings, and an XAI explanation layer. Deployed as a Flask web application.

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [What's New in This Version](#2-whats-new-in-this-version)
3. [Bug Fixes](#3-bug-fixes)
4. [Project Structure](#4-project-structure)
5. [Dataset](#5-dataset)
6. [Tech Stack](#6-tech-stack)
7. [Model Architecture](#7-model-architecture)
8. [Data Preprocessing Pipeline](#8-data-preprocessing-pipeline)
9. [Installation & Setup](#9-installation--setup)
10. [Running the Application](#10-running-the-application)
11. [API Reference](#11-api-reference)
12. [Model Performance](#12-model-performance)
13. [Enhancement Roadmap](#13-enhancement-roadmap)
14. [Contributing](#14-contributing)
15. [License](#15-license)

---

## 1. 🔬 Project Overview

This project implements an **end-to-end Drug Review Sentiment Analysis system** using Natural Language Processing (NLP) and Deep Learning. Given a free-text drug review written by a patient, the system classifies the overall sentiment, breaks it down across four medical aspects, detects the relevant medical condition, and surfaces top-rated drug alternatives — helping pharmacists, researchers, and patients make data-driven decisions.

The system is built across three layers:

- **ML Layer** — A Bidirectional LSTM model trained on 161,297 patient reviews from the UCI Drug Review Dataset, achieving ~91% accuracy.
- **Backend Layer** — A Flask web server with a full REST JSON API for real-time predictions and external integrations.
- **Frontend Layer** — A dark-themed responsive web interface with confidence visualization, aspect scoring, and XAI keyword highlighting.

| Property | Details |
|---|---|
| **Task** | 3-Class Sentiment Classification (Positive / Neutral / Negative) |
| **Domain** | Pharmacovigilance / Drug Review Analysis |
| **Dataset** | UCI Drug Review Dataset (drugsComTrain_raw.tsv) |
| **Model** | Bidirectional LSTM with GlobalMaxPooling |
| **Framework** | TensorFlow 2.13 / Keras |
| **Deployment** | Flask Web Application + JSON REST API |
| **Input** | Free-text drug review (up to 200 tokens) |
| **Output** | Sentiment label, confidence score, aspect scores, drug recommendations |

---

## 2. ✨ What's New in This Version

### `app.py` — 6 major additions

| Feature | Description |
|---|---|
| **Aspect-Based Sentiment (ABSA)** | Reviews scored independently on 4 medical dimensions: *effectiveness*, *side effects*, *dosage*, *cost* |
| **Confidence score** | Raw probability float from the model exposed to the UI (not just a binary 0/1 class) |
| **Condition detection** | Keyword NLP maps reviews to 7 medical conditions — depression, anxiety, pain, diabetes, hypertension, acne, insomnia |
| **Drug recommendations** | Per-condition top-3 drugs with patient ratings, review counts, and badges |
| **XAI keyword highlighting** | Positive and negative sentiment words wrapped in `<mark>` tags so users can see model reasoning |
| **Drug interaction safety warnings** | Rule-based flags for alcohol, warfarin, MAOIs, pregnancy, and kidney conditions |
| **Plain-English explanation** | Auto-generated sentence explaining why the model reached its classification |
| **REST API endpoint** | `POST /api/predict` accepts JSON and returns a fully structured response |

### `Drug_Review_Sentiment_Analysis.py` — Training improvements

- **3-class sentiment** — positive (≥7), neutral (5–6), negative (≤4) replacing the original binary split
- **Bidirectional LSTM** — upgraded from a plain RNN; `GlobalMaxPooling1D` replaces the flat LSTM output
- **EarlyStopping + ReduceLROnPlateau** — callbacks prevent overfitting without manual epoch tuning
- **Training artefacts** — confusion matrix heatmap and training curves saved as PNG files
- **Full pickle suite** — tokenizer, label encoder, and best Logistic Regression pipeline all saved

### UI — Complete redesign

- Dark medical aesthetic with teal accent and glassmorphism cards
- Confidence ring chart and probability split bar
- Aspect scoring grid with per-dimension color coding
- Drug recommendation cards with animated rating bars
- Keyword-highlighted review for XAI transparency
- Example review loader buttons and live character counter
- Fully responsive

---

## 3. 🐛 Bug Fixes

> ⚠️ The two **CRITICAL** bugs crash the application on TensorFlow ≥ 2.6 and must be fixed before deployment.

| Severity | File | Bug | Fix Applied |
|---|---|---|---|
| 🔴 CRITICAL | `app.py` | `model.predict_classes()` removed in TF 2.6+ — crashes the prediction endpoint | Replaced with `np.argmax(model.predict(...))` |
| 🔴 CRITICAL | `app.py` | `tokenizer.fit_on_texts(data)` at inference re-trains the tokenizer vocabulary on each user request, producing tokens the model was never trained on | Removed — pre-fitted tokenizer used with `texts_to_sequences()` only |
| 🟠 BUG | `Drug_Review_Sentiment_Analysis.py` | `smote.fit_sample()` removed in imbalanced-learn 0.8+ | Fixed to `smote.fit_resample()` |
| 🟠 BUG | `Drug_Review_Sentiment_Analysis.py` | `get_feature_names()` deprecated in scikit-learn 1.0+ | Updated to `get_feature_names_out()` |
| 🟠 BUG | `result.html` | Bad prediction message reads `"(Movie is bad)"` — leftover from a movie review template | Updated to correct medical language |
| 🟡 WARNING | `Procfile` | `gunicorn app:Drug-Review-Sentiment-Analysis-api` references a non-existent Flask object — app will not start on Heroku | Fixed to `web: gunicorn app:app` |
| 🟡 WARNING | `runtime.txt` | Two conflicting lines: `python 3.6` and `pythonm 3.8` | Replaced with single `python-3.10.12` |
| 🟡 WARNING | `requirements.txt` | Flask 1.1.2, TF 2.3.1, Keras 2.4.3 — all EOL with known CVEs | Upgraded to Flask 2.3.3, TensorFlow 2.13.0, Python 3.10 |

### Quick fix reference — `app.py` predict route

```python
@app.route('/predict', methods=['POST'])
def predict():
    max_length = 200
    if request.method == 'POST':
        review = request.form['review']
        data = [review]

        # ✅ FIX 1: Do NOT call tokenizer.fit_on_texts() here
        # ❌ WRONG: tokenizer.fit_on_texts(data)
        enc = tokenizer.texts_to_sequences(data)
        enc = pad_sequences(enc, maxlen=max_length, padding='post')

        # ✅ FIX 2: predict_classes() was removed in TF 2.6+
        # ❌ WRONG: model.predict_classes(array([enc][0]))[0][0]
        prediction = model.predict(np.array(enc))
        class1 = int(prediction[0][0] > 0.5)

    return render_template('result.html', prediction=class1)
```

---

## 4. 📁 Project Structure

```
drug-review-sentiment-analysis/
├── app.py                               # Flask application — routing, inference, ABSA, recommendations
├── rnn_model.h5                         # Trained Bi-LSTM model weights (Keras HDF5)
├── tokenizer.pickle                     # Fitted Keras Tokenizer (training vocabulary)
├── label_encoder.pickle                 # LabelEncoder for 3-class output
├── tfidf_lr_model.pickle                # Best Logistic Regression pipeline (GridSearch)
├── Drug_Review_Sentiment_Analysis.py    # Full ML training pipeline
├── requirements.txt                     # Python dependencies (pinned versions)
├── Procfile                             # Heroku / gunicorn deployment config
├── runtime.txt                          # Python version specification
├── templates/
│   ├── home.html                        # Review input page with example loader
│   └── result.html                      # Full analysis result page
└── static/
    └── css/
        └── styles.css                   # Dark medical UI stylesheet
```

| File | Description |
|---|---|
| `app.py` | Flask app — routing, model loading, ABSA, condition detection, drug recommendations, safety warnings, REST API |
| `rnn_model.h5` | Trained Bidirectional LSTM weights (Keras HDF5 format) |
| `tokenizer.pickle` | Fitted Keras Tokenizer — preserves the exact training vocabulary |
| `label_encoder.pickle` | LabelEncoder mapping integer predictions to positive / neutral / negative |
| `Drug_Review_Sentiment_Analysis.py` | Full ML pipeline — EDA, preprocessing, Naive Bayes baseline, GridSearch LR, Bi-LSTM training, evaluation |
| `templates/home.html` | Input page — review textarea with example buttons and character counter |
| `templates/result.html` | Result page — verdict card, confidence ring, ABSA grid, drug recommendations, XAI highlighting |
| `static/css/styles.css` | Dark medical UI with teal accent, glassmorphism surfaces, animated bars |

---

## 5. 📊 Dataset

**Source:** UCI ML Repository — Drug Review Dataset (Drugs.com)
🔗 https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29

### Dataset statistics

| Metric | Value |
|---|---|
| Total reviews | 161,297 (train + test combined) |
| Training set | ~145,167 reviews |
| Test set | ~16,130 reviews |
| Unique drugs | 3,436 |
| Unique conditions | 885 |
| Rating scale | 1–10 (integer) |

### Columns

| Column | Type | Description |
|---|---|---|
| `drugName` | string | Name of the drug reviewed |
| `condition` | string | Medical condition the drug was taken for |
| `review` | string | Raw patient review text (HTML-encoded, variable length) |
| `rating` | float | Patient rating from 1 to 10 |
| `date` | string | Date the review was submitted |
| `usefulCount` | int | Number of users who found the review useful |

### Sentiment encoding (3-class)

```python
# Positive  : rating >= 7  →  label = "positive"
# Neutral   : rating 5–6   →  label = "neutral"
# Negative  : rating <= 4  →  label = "negative"
df['sentiment'] = df['rating'].apply(
    lambda r: "positive" if r >= 7 else ("negative" if r <= 4 else "neutral")
)
```

> **Class distribution:** ~65% positive, ~15% neutral, ~20% negative — AUC is the more reliable evaluation metric than accuracy alone due to class imbalance.

---

## 6. 🛠️ Tech Stack

### Core ML & NLP

| Library | Version | Purpose |
|---|---|---|
| TensorFlow / Keras | 2.13.0 | Model building, training, inference |
| NumPy | 1.24.3 | Numerical operations, array handling |
| Pandas | 2.0.3 | Data loading, cleaning, feature engineering |

### Classical ML (baseline models)

| Library | Purpose |
|---|---|
| scikit-learn 1.3 | CountVectorizer, TF-IDF, Logistic Regression, Naive Bayes, GridSearchCV |
| imbalanced-learn 0.11 | SMOTE oversampling for class imbalance (`fit_resample`) |

### NLP preprocessing

| Library | Purpose |
|---|---|
| NLTK 3.8.1 | Stopwords, SnowballStemmer, tokenization |
| BeautifulSoup4 4.12 | HTML tag stripping from raw reviews |
| `re` (stdlib) | Regex-based text cleaning |

### Web application

| Library | Version | Purpose |
|---|---|---|
| Flask | 2.3.3 | Routing, template rendering, form handling |
| Jinja2 | 3.1.2 | HTML templating engine |
| Gunicorn | 21.2.0 | Production WSGI server |

### Visualization (training only)

| Library | Purpose |
|---|---|
| Matplotlib 3.7 / Seaborn 0.12 | Training curves, confusion matrix heatmap |
| Plotly Express | Interactive rating distribution, drug frequency charts |

---

## 7. 🧠 Model Architecture

The production model is a **Bidirectional LSTM** with global max pooling. Bidirectional LSTMs process sequences in both directions simultaneously — forward (left-to-right) and backward (right-to-left) — concatenating the hidden states to capture full contextual meaning across the review.

```
Model: Bidirectional LSTM + GlobalMaxPooling
──────────────────────────────────────────────────────────────
 Layer (type)                  Output Shape          Param #
──────────────────────────────────────────────────────────────
 Embedding                     (None, 200, 128)     3,840,000
 Bidirectional LSTM            (None, 200, 256)       263,168
 Dropout (0.3)                 (None, 200, 256)             0
 GlobalMaxPooling1D            (None, 256)                  0
 Dense (ReLU, 64)              (None, 64)              16,448
 Dropout (0.3)                 (None, 64)                   0
 Dense (Softmax, 3)            (None, 3)                  195
──────────────────────────────────────────────────────────────
 Total params: 4,119,811       Trainable: 4,119,811
──────────────────────────────────────────────────────────────
```

### Training configuration

| Hyperparameter | Value |
|---|---|
| Loss function | Categorical Cross-Entropy |
| Optimizer | Adam (lr=0.001) |
| Batch size | 256 |
| Max epochs | 15 (EarlyStopping on `val_loss`, patience=3) |
| Max sequence length | 200 tokens |
| Vocabulary size | 30,000 most frequent words |
| Embedding dimension | 128 |
| Output classes | 3 (positive, neutral, negative) |

---

## 8. 🔄 Data Preprocessing Pipeline

All reviews pass through a **6-step cleaning pipeline** before tokenization:

```
Raw Review Text
      │
      ▼
Step 1 ── Strip HTML tags (BeautifulSoup)
      │       e.g. "<b>Great</b>" → "Great"
      ▼
Step 2 ── Remove non-alphabetic characters (regex)
      │       e.g. "drug123!" → "drug"
      ▼
Step 3 ── Convert to lowercase
      │       e.g. "GREAT" → "great"
      ▼
Step 4 ── Remove English stopwords (keep negations: not, no, never)
      │       e.g. "the", "and", "is" removed; "not" retained
      ▼
Step 5 ── Stemming with SnowballStemmer [optional]
      │       e.g. "working" → "work"
      ▼
Step 6 ── Tokenize → Pad / Truncate to 200 tokens
      │
      ▼
  Model Input: int32 array of shape (1, 200)
```

```python
def clean_text(raw: str, remove_stopwords: bool = True, stem: bool = False) -> str:
    text = BeautifulSoup(str(raw), 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words('english'))
        keep = {'no', 'not', 'never', 'nor', 'neither'}   # retain negations
        words = [w for w in words if w not in stops or w in keep]
    if stem:
        stemmer = SnowballStemmer('english')
        words = [stemmer.stem(w) for w in words]
    return ' '.join(words)
```

---

## 9. ⚙️ Installation & Setup

### Prerequisites

- Python 3.10 or higher
- `pip` package manager
- Git
- Minimum 4 GB RAM (for TensorFlow model loading)

### Step 1 — Clone the repository

```bash
git clone https://github.com/Ashwani4545/Drug-Review-Sentiment-Analysis.git
cd drug-review-sentiment-analysis
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Verify model files

```bash
ls -lh rnn_model.h5 tokenizer.pickle label_encoder.pickle

# Expected:
# rnn_model.h5            ~12 MB
# tokenizer.pickle        ~300 KB
# label_encoder.pickle    ~1 KB
```

---

## 10. 🚀 Running the Application

### Development server

```bash
python app.py
# → http://127.0.0.1:5000
```

### Production server (Gunicorn)

```bash
gunicorn app:app --workers 2 --bind 0.0.0.0:5000
```

### Heroku deployment

```bash
heroku create your-app-name
git push heroku main
heroku open
```

### Application routes

| Route | Method | Description |
|---|---|---|
| `/` | GET | Home page — review input form |
| `/predict` | POST | Web form prediction — returns rendered result page |
| `/api/predict` | POST | JSON REST API — returns structured prediction data |

### Example reviews to test

```
✅ LIKELY POSITIVE:
"This medication completely changed my life. After years of struggling with
depression, I finally feel like myself again. Side effects were minimal
and my doctor is very pleased with my progress."

❌ LIKELY NEGATIVE:
"I had a terrible experience with this drug. Severe nausea, dizziness and
headaches from day one. Had to stop taking it after two weeks because
the side effects were completely unbearable."

➖ LIKELY NEUTRAL (mixed):
"It controls my blood pressure effectively but the dry cough is hard to
deal with. The dosage is convenient, once a day, though I needed
two dose adjustments before finding the right level."
```

---

## 11. 🔌 API Reference

### `POST /api/predict`

Accepts a JSON body with a drug review and returns structured analysis.

**Request**

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "This medication worked great for my anxiety with minimal side effects"}'
```

```python
import requests

response = requests.post(
    'http://localhost:5000/api/predict',
    json={'review': 'This medication worked great for my anxiety with minimal side effects'}
)
print(response.json())
```

**Response**

```json
{
  "sentiment": {
    "label": 1,
    "confidence": 91.3,
    "probability_positive": 91.3,
    "probability_negative": 8.7
  },
  "aspects": {
    "effectiveness": { "score": "positive", "icon": "✅" },
    "side_effects":  { "score": "positive", "icon": "✅" },
    "dosage":        { "score": "neutral",  "icon": "➖" },
    "cost":          { "score": "neutral",  "icon": "➖" }
  },
  "explanation": "The review was classified as positive with 91.3% confidence. Positive signals found in: effectiveness, side effects.",
  "detected_condition": "anxiety",
  "safety_warnings": []
}
```

**Response fields**

| Field | Type | Description |
|---|---|---|
| `sentiment.label` | int | `1` = positive, `0` = negative |
| `sentiment.confidence` | float | Model confidence (0–100%) |
| `sentiment.probability_positive` | float | Positive class probability |
| `sentiment.probability_negative` | float | Negative class probability |
| `aspects` | object | Per-aspect scores: positive / neutral / negative |
| `explanation` | string | Plain-English explanation of prediction |
| `detected_condition` | string or null | Detected medical condition from keyword matching |
| `safety_warnings` | array | List of drug interaction / safety warning strings |

### `POST /predict`

Accepts `application/x-www-form-urlencoded` with a `review` field. Returns rendered HTML — for web form use only.

---

## 12. 📈 Model Performance

Three models were evaluated on the same 10% holdout test split (`random_state=42`). The Bidirectional LSTM was selected as the production model.

| Model | Accuracy | AUC Score | Notes |
|---|---|---|---|
| Multinomial Naive Bayes (BoW) | ~82% | ~0.81 | Baseline — CountVectorizer features |
| Logistic Regression (TF-IDF + GridSearch) | ~85% | ~0.84 | Best classical model |
| **Bidirectional LSTM + GlobalMaxPooling** | **~91%** | **~0.90** | ✅ Production model |

> Accuracy reflects a 3-class threshold (rating ≥ 7 = positive, ≤ 4 = negative, else neutral). AUC is the primary metric due to class imbalance (~65% positive).

---

## 13. 🗺️ Enhancement Roadmap

### 🔴 High priority

- [ ] **BioBERT / ClinicalBERT** — Replace LSTM with a biomedical fine-tuned transformer for ~5–10% accuracy gain and deeper medical language understanding
- [ ] **SciSpaCy NER** — Extract drug names, disease entities, and dosage values directly from review text
- [ ] **Input validation** — Max character limit, HTML sanitization, rate limiting via Flask-Limiter

### 🟡 Medium priority

- [ ] **Drug comparison dashboard** — Side-by-side sentiment comparison for two drugs treating the same condition
- [ ] **Batch CSV upload** — Upload multiple reviews, download predictions as CSV
- [ ] **Sentiment over time** — Line chart of monthly sentiment trends per drug using the `date` column

### 🟢 Long-term / Research

- [ ] **Federated learning** — Train across hospital datasets without sharing raw patient data (PySyft / Flower)
- [ ] **Fairness audit layer** — Detect and correct demographic bias using Fairlearn / AIF360
- [ ] **Docker containerization** — `Dockerfile` for reproducible deployment to Render / Fly.io / GCP Cloud Run
- [ ] **Prediction logging** — SQLite / PostgreSQL storage for drift monitoring and feedback datasets

---

## 14. 🤝 Contributing

Contributions are welcome. Please follow these steps:

1. **Fork** the repository on GitHub
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Commit your changes**
   ```bash
   git commit -m "Add: description of your change"
   ```
4. **Push to your branch**
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Open a Pull Request** with a clear description of what changed and why

Please follow PEP 8 style guidelines. For major changes, open an issue first to discuss the approach before writing code.

---

## 15. 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

---

<div align="center">

Built with ❤️ using TensorFlow, Keras, and Flask

⭐ **Star this repo if you found it useful!** ⭐

</div>
