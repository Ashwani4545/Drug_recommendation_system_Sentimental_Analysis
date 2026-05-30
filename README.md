# 💊 Drug Review Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.8+-3572A5?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?style=for-the-badge&logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-2E7D32?style=for-the-badge)

> **NLP-powered Drug Recommendation System** — Classifies patient drug reviews as positive or negative sentiment using a Bidirectional LSTM model, deployed as a Flask web application.

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Features](#2-features)
3. [Project Structure](#3-project-structure)
4. [Dataset](#4-dataset)
5. [Tech Stack](#5-tech-stack)
6. [Model Architecture](#6-model-architecture)
7. [Data Preprocessing Pipeline](#7-data-preprocessing-pipeline)
8. [Installation & Setup](#8-installation--setup)
9. [Running the Application](#9-running-the-application)
10. [API Usage](#10-api-usage)
11. [Model Performance](#11-model-performance)
12. [Known Bugs & Fixes](#12-known-bugs--fixes)
13. [Enhancement Roadmap](#13-enhancement-roadmap)
14. [Contributing](#14-contributing)
15. [License](#15-license)

---

## 1. 🔬 Project Overview

This project implements an **end-to-end Drug Review Sentiment Analysis system** using Natural Language Processing (NLP) and Deep Learning. Given a free-text drug review written by a patient, the system classifies it as either a **positive (Good Drug)** or **negative (Bad Drug)** experience — helping pharmacists, researchers, and patients make data-driven decisions about drug efficacy and tolerability.

The system is built across three layers:

- **ML Layer** — A Bidirectional LSTM model trained on 161,297 patient reviews from the UCI Drug Review Dataset, achieving ~91% accuracy.
- **Backend Layer** — A Flask web server that loads the trained model at startup and serves predictions in real-time.
- **Frontend Layer** — A Bootstrap 4 HTML interface where users enter reviews and receive instant predictions.

| Property | Details |
|---|---|
| **Task** | Binary Sentiment Classification (Positive / Negative) |
| **Domain** | Pharmacovigilance / Drug Review Analysis |
| **Dataset** | UCI Drug Review Dataset (drugsComTrain_raw.tsv) |
| **Model** | Bidirectional LSTM (Bi-LSTM) with Dropout |
| **Framework** | TensorFlow 2.x / Keras |
| **Deployment** | Flask Web Application |
| **Input** | Free-text drug review (up to 200 tokens) |
| **Output** | Sentiment label: Good Drug (1) or Bad Drug (0) |

---

## 2. ✨ Features

- 🔮 **Real-time sentiment prediction** via web form — no setup needed for end users
- 🧠 **Bidirectional LSTM model** — captures both forward and backward context in reviews
- 🔤 **Pre-trained tokenizer** — consistent vocabulary between training and inference
- 📏 **Sequence padding** — handles variable-length reviews with `max_length=200` tokens
- 📱 **Bootstrap 4 responsive frontend** — mobile-friendly input and result pages
- ☁️ **Heroku/Gunicorn deployment-ready** — Procfile and requirements.txt included
- 📊 **Baseline models included** — Naive Bayes and Logistic Regression for benchmarking
- 📈 **Full EDA notebook** — rating distribution, drug frequency, condition analysis with Plotly

---

## 3. 📁 Project Structure

```
drug-review-sentiment-analysis/
├── app.py                                               # Flask application (routing + inference)
├── rnn_model.h5                                         # Trained Bi-LSTM model weights (Keras HDF5)
├── tokenizer.pickle                                     # Fitted Keras Tokenizer (training vocabulary)
├── Drug_Review_Sentiment_Analysis.py                    # Full ML training pipeline
├── Drug_Sentiment_Analysis_RNN_Bidirectional_lstm.ipynb # Jupyter notebook walkthrough
├── requirements.txt                                     # Python dependencies (pinned versions)
├── Procfile                                             # Heroku / gunicorn deployment config
├── runtime.txt                                          # Python version specification
├── templates/
│   ├── home.html                                        # Review input page (textarea form)
│   └── result.html                                      # Prediction result page
└── static/
    ├── css/
    │   └── styles.css                                   # Custom styles + background
    └── images/
        └── sss.jpg                                      # Background image (⚠️ currently missing)
```

| File | Description |
|---|---|
| `app.py` | Flask web application — routing, model loading, prediction logic |
| `rnn_model.h5` | Trained Bidirectional LSTM model weights (Keras HDF5 format) |
| `tokenizer.pickle` | Fitted Keras Tokenizer — preserves the exact training vocabulary |
| `Drug_Review_Sentiment_Analysis.py` | Full ML pipeline: EDA, preprocessing, model training, evaluation |
| `Drug_Sentiment_Analysis_RNN_Bidirectional_lstm.ipynb` | Jupyter notebook — interactive walkthrough of the full pipeline |
| `templates/home.html` | Input page — review textarea form with Bootstrap 4 layout |
| `templates/result.html` | Result page — displays predicted sentiment label |
| `static/css/styles.css` | Custom CSS — background image, typography, body styling |
| `requirements.txt` | Python dependencies with pinned versions |
| `Procfile` | Heroku deployment config — gunicorn server command |
| `runtime.txt` | Python runtime version specification for Heroku |

---

## 4. 📊 Dataset

### Source

**UCI ML Repository — Drug Review Dataset (Drugs.com)**
🔗 https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29

### Dataset Statistics

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

### Sentiment Encoding

```python
# Positive sentiment: rating > 6  →  label = 1
# Negative sentiment: rating ≤ 6  →  label = 0
df['Sentiment'] = np.where(df['rating'] > 6, 1, 0)
```

> **Class distribution:** ~70% positive, ~30% negative — AUC is the more reliable evaluation metric than accuracy alone.

---

## 5. 🛠️ Tech Stack

### Core ML & NLP
| Library | Version | Purpose |
|---|---|---|
| TensorFlow / Keras | 2.3.1 | Model building, training, inference |
| NumPy | 1.18.1 | Numerical operations, array handling |
| Pandas | 1.0.3 | Data loading, cleaning, feature engineering |

### Classical ML (Baseline Models)
| Library | Purpose |
|---|---|
| scikit-learn | CountVectorizer, TF-IDF, Logistic Regression, Naive Bayes, GridSearchCV |
| imbalanced-learn | SMOTE oversampling for class imbalance |

### NLP Preprocessing
| Library | Purpose |
|---|---|
| NLTK 3.4.5 | Stopwords, SnowballStemmer, tokenization |
| BeautifulSoup4 | HTML tag stripping from raw reviews |
| `re` (stdlib) | Regex-based text cleaning |

### Web Application
| Library | Version | Purpose |
|---|---|---|
| Flask | 1.1.2 | Routing, template rendering, form handling |
| Jinja2 | 2.11.2 | HTML templating engine |
| Bootstrap | 4.3.1 | Responsive frontend CSS framework |
| Gunicorn | 20.0.4 | Production WSGI server |

### Visualization (Notebook)
| Library | Purpose |
|---|---|
| Matplotlib / Seaborn | Training curves, confusion matrix |
| Plotly Express | Interactive rating distribution, drug frequency charts |

---

## 6. 🧠 Model Architecture

The production model is a **stacked Bidirectional LSTM** network. Bidirectional LSTMs process sequences in both directions simultaneously — the forward pass reads left-to-right, the backward pass reads right-to-left. Hidden states from both passes are concatenated, allowing the model to capture full contextual meaning from a review.

```
Model: Bidirectional LSTM (Stacked)
──────────────────────────────────────────────────────────────
 Layer (type)                  Output Shape          Param #
──────────────────────────────────────────────────────────────
 Embedding                     (None, 200, 128)       640,000
 Bidirectional LSTM (Layer 1)  (None, 200, 256)       263,168
 Dropout (0.5)                 (None, 200, 256)             0
 Bidirectional LSTM (Layer 2)  (None, 128)            164,352
 Dropout (0.5)                 (None, 128)                  0
 Dense — sigmoid               (None, 1)                  129
──────────────────────────────────────────────────────────────
 Total params: 1,067,649       Trainable: 1,067,649
──────────────────────────────────────────────────────────────
```

### Layer Summary

| # | Layer Type | Purpose | Output Shape |
|---|---|---|---|
| 1 | Embedding | Vocab → Dense Vectors | `vocab_size × 128` |
| 2 | Bidirectional LSTM | Forward + Backward context | `128 units × 2` |
| 3 | Dropout (0.5) | Regularization | — |
| 4 | Bidirectional LSTM | Second recurrent layer | `64 units × 2` |
| 5 | Dropout (0.5) | Regularization | — |
| 6 | Dense (sigmoid) | Binary classification output | `1 unit` |

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Loss function | Binary Cross-Entropy |
| Optimizer | Adam (lr=0.001) |
| Batch size | 64 |
| Epochs | 10 (with early stopping on `val_loss`) |
| Max sequence length | 200 tokens |
| Vocabulary size | 5,000 most frequent words |
| Embedding dimension | 128 |

---

## 7. 🔄 Data Preprocessing Pipeline

All reviews pass through a **6-step cleaning pipeline** before tokenization:

```
Raw Review Text
      │
      ▼
Step 1 ── Strip HTML tags (BeautifulSoup)
      │       e.g. "&lt;b&gt;Great&lt;/b&gt;" → "Great"
      ▼
Step 2 ── Remove non-alphabetic characters (regex)
      │       e.g. "drug123!" → "drug"
      ▼
Step 3 ── Convert to lowercase
      │       e.g. "GREAT" → "great"
      ▼
Step 4 ── Remove English stopwords [optional]
      │       e.g. "the", "and", "is" removed
      ▼
Step 5 ── Stemming with SnowballStemmer [optional]
      │       e.g. "working" → "work"
      ▼
Step 6 ── Tokenize → Pad/Truncate to 200 tokens
      │
      ▼
  Model Input: int32 array of shape (1, 200)
```

```python
def cleanData(raw_data, remove_stopwords=False, stemming=False):
    # Step 1: Strip HTML
    text = BeautifulSoup(raw_data, 'html.parser').get_text()
    # Step 2: Remove non-alpha characters
    letters_only = re.sub('[^a-zA-Z]', ' ', text)
    # Step 3: Lowercase
    words = letters_only.lower().split()
    # Step 4: Remove stopwords (optional)
    if remove_stopwords:
        stops = set(stopwords.words('english'))
        words = [w for w in words if w not in stops]
    # Step 5: Stemming (optional)
    if stemming:
        stemmer = SnowballStemmer('english')
        words = [stemmer.stem(w) for w in words]
    return ' '.join(words)
```

---

## 8. ⚙️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- `pip` package manager
- Git
- Minimum 4GB RAM (for TensorFlow model loading)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/your-username/drug-review-sentiment-analysis.git
cd drug-review-sentiment-analysis
```

### Step 2 — Create Virtual Environment

```bash
# Create environment
python -m venv venv

# Activate — macOS / Linux
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Verify Model Files

Ensure both model files are present in the project root:

```bash
ls -lh rnn_model.h5 tokenizer.pickle

# Expected output:
# -rw-r--r--  rnn_model.h5       (~12 MB)
# -rw-r--r--  tokenizer.pickle   (~300 KB)
```

---

## 9. 🚀 Running the Application

### Development Server

```bash
python app.py

# Output:
#  * Running on http://127.0.0.1:5000
#  * Debug mode: on
```

### Production Server (Gunicorn)

```bash
gunicorn app:app --workers 2 --bind 0.0.0.0:5000
```

### Heroku Deployment

```bash
heroku create your-app-name
git push heroku main
heroku open
```

### Application Routes

| Route | Method | Description |
|---|---|---|
| `/` | GET | Home page — review input form |
| `/predict` | POST | Prediction endpoint — returns result page |

### Example Reviews to Test

```
✅ LIKELY POSITIVE:
"This medication completely changed my life. After years of struggling with
depression, I finally feel like myself again. Side effects were minimal
and my doctor is very pleased with my progress."

❌ LIKELY NEGATIVE:
"I had a terrible experience with this drug. Severe nausea, dizziness and
headaches from day one. Had to stop taking it after two weeks because
the side effects were completely unbearable."
```

---

## 10. 🔌 API Usage

### `GET /`
Returns the HTML home page with the review input form.

---

### `POST /predict`
Accepts a form-encoded drug review and returns the result page with the predicted sentiment.

**Request**

```bash
# cURL
curl -X POST http://localhost:5000/predict \
  -d 'review=This drug worked great for my condition with minimal side effects'
```

```python
# Python requests
import requests

response = requests.post(
    'http://localhost:5000/predict',
    data={'review': 'This drug worked great for my condition with minimal side effects'}
)
print(response.text)  # Returns rendered HTML with prediction
```

**Response** — Rendered HTML page showing:
- `prediction == 1` → **"Good Drug — Drug is Advisable"**
- `prediction == 0` → **"Bad Drug — Drug is not Advisable"**

---

## 11. 📈 Model Performance

Three models were evaluated on the same test split (10% holdout, `random_state=0`). The Bidirectional LSTM was selected as the production model based on its superior accuracy and AUC score.

| Model | Accuracy | AUC Score | Notes |
|---|---|---|---|
| Multinomial Naive Bayes (BoW) | ~82% | ~0.81 | Baseline — CountVectorizer features |
| Logistic Regression (TF-IDF) | ~85% | ~0.84 | Grid-searched best hyperparameters |
| **Bidirectional LSTM (RNN)** | **~91%** | **~0.90** | ✅ Final production model |

> **Note:** Accuracy above reflects a binary threshold at `rating > 6`. Because the class distribution is skewed (~70% positive), AUC is the more reliable evaluation metric.

---

## 12. 🐛 Known Bugs & Required Fixes

> ⚠️ The two **CRITICAL** bugs will crash the application on TensorFlow ≥ 2.6 and must be fixed before deployment.

| Severity | Issue | Problem | Fix |
|---|---|---|---|
| 🔴 CRITICAL | `predict_classes()` removed | `model.predict_classes()` was removed in TF 2.6+. Crashes the prediction endpoint entirely. | Replace with `(model.predict(enc) > 0.5).astype(int)[0][0]` |
| 🔴 CRITICAL | `tokenizer.fit_on_texts()` at inference | Calling `fit_on_texts()` on new input mutates the tokenizer vocabulary, producing tokens different from what the model was trained on. | Remove the call — use only `tokenizer.texts_to_sequences(data)` |
| 🟠 BUG | `result.html` copy-paste error | Bad prediction message reads `"(Movie is bad)"` — leftover from a movie review template. | Update message to reflect drug domain context |
| 🟠 BUG | Missing background image | `styles.css` references `../images/sss.jpg` which does not exist. Page renders broken. | Replace with a CSS gradient or add the image to `static/images/` |
| 🟡 WARNING | Outdated dependencies | `Flask==1.1.2`, `TF==2.3.1`, `Keras==2.4.3` have known CVEs. TF 2.3 is EOL. | Upgrade to `Flask>=2.3`, `tensorflow>=2.13`, `Python 3.10` |
| 🟡 WARNING | Procfile wrong callable | `gunicorn app:Drug-Review-Sentiment-Analysis-api` references no Flask object. | Change to: `web: gunicorn app:app` |
| 🟡 WARNING | `runtime.txt` conflict | Specifies both `python 3.6` and `python 3.8` on separate lines. | Keep only `python-3.10.x` |

### ✅ Quick Fix — `app.py` predict route

```python
@app.route('/predict', methods=['POST'])
def predict():
    max_length = 200
    if request.method == 'POST':
        review = request.form['review']
        data = [review]

        # ✅ FIX 1: Do NOT call tokenizer.fit_on_texts() here
        # ❌ WRONG: tokenizer.fit_on_texts(data)
        enc = tokenizer.texts_to_sequences(data)        # ← correct
        enc = pad_sequences(enc, maxlen=max_length, padding='post')

        # ✅ FIX 2: predict_classes() was removed in TF 2.6+
        # ❌ WRONG: model.predict_classes(array([enc][0]))[0][0]
        prediction = model.predict(np.array(enc))
        class1 = int(prediction[0][0] > 0.5)            # ← correct

    return render_template('result.html', prediction=class1)
```

---

## 13. 🗺️ Enhancement Roadmap

### 🔴 High Priority
- [ ] **3-class sentiment** — Add Neutral class for mixed reviews (ratings 4–6)
- [ ] **Confidence score display** — Show prediction probability (e.g. `87% positive`)
- [ ] **Side effect NER** — Integrate SciSpaCy to extract drug/disease entities from review text
- [ ] **Input validation** — Max character limit, HTML sanitization, rate limiting with Flask-Limiter

### 🟡 Medium Priority
- [ ] **Drug comparison dashboard** — Side-by-side sentiment comparison for two drugs treating the same condition
- [ ] **Sentiment over time** — Line chart of monthly sentiment trends per drug using the date column
- [ ] **Batch CSV upload** — Upload multiple reviews, download predictions as CSV via `flask.send_file()`
- [ ] **REST JSON API** — `/api/predict` endpoint returning `{"prediction": 1, "confidence": 0.87}`

### 🟢 Long-term / Research
- [ ] **Upgrade to Bio_ClinicalBERT** — Fine-tune HuggingFace transformer on medical text (~5–10% accuracy gain over LSTM)
- [ ] **LIME explainability** — Highlight the words in each review that drove the prediction
- [ ] **Docker containerization** — `Dockerfile` for reproducible deployment to Render / Fly.io / GCP Cloud Run
- [ ] **Prediction logging** — SQLite/PostgreSQL storage of predictions for drift monitoring and feedback dataset

---

## 14. 🤝 Contributing

Contributions are welcome! Please follow these steps:

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
5. **Open a Pull Request** with a clear description of what was changed and why

> Please ensure your code follows PEP 8 style guidelines. For major changes, open an issue first to discuss the approach before writing code.

---

## 15. 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2024 [Your Name]

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
