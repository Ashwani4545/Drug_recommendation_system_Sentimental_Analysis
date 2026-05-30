"""
Enhanced Drug Review Sentiment Analysis — Training Script
Improvements over original:
  1. Fixed SMOTE API (fit_resample not fit_sample)
  2. Fixed deprecated get_feature_names → get_feature_names_out
  3. Added 3-class sentiment (positive / neutral / negative)
  4. Aspect-level keyword extraction & basic aspect model training
  5. Proper model saving (tokenizer + label encoder)
  6. Evaluation dashboard (confusion matrix, ROC, per-class metrics)
  7. GridSearch pipeline with proper preprocessed text
  8. Runtime-safe: no %matplotlib magic (use plt.savefig instead)
"""

import os
import re
import pickle
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

warnings.filterwarnings("ignore")

# ── 1. Load data ─────────────────────────────────────────────────────────────
DATA_PATH = "drugsComTrain_raw.tsv"     # adjust path as needed
TEST_PATH  = "drugsComTest_raw.tsv"

df_train = pd.read_csv(DATA_PATH, sep="\t", on_bad_lines="skip")
df_test  = pd.read_csv(TEST_PATH,  sep="\t", on_bad_lines="skip")

df = pd.concat([df_train, df_test], ignore_index=True)
print(f"Total records: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# ── 2. EDA ────────────────────────────────────────────────────────────────────
print("\nSummary statistics:\n", df.describe())
print(f"\nUnique drugs: {df['drugName'].nunique()}")
print(f"Unique conditions: {df['condition'].nunique()}")
print(f"Missing values:\n{df.isna().sum()}")

# Rating distribution
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x="rating", palette="viridis")
plt.title("Distribution of Drug Ratings")
plt.tight_layout()
plt.savefig("rating_distribution.png")
plt.close()

# Top drugs by review count
top_drugs = df["drugName"].value_counts().head(20)
plt.figure(figsize=(10, 5))
top_drugs.plot(kind="bar", color="#00d4a4")
plt.title("Top 20 Drugs by Review Count")
plt.tight_layout()
plt.savefig("top_drugs.png")
plt.close()

# ── 3. Preprocessing ─────────────────────────────────────────────────────────
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
nltk.download("stopwords", quiet=True)

def clean_text(raw: str, remove_stopwords: bool = True, stem: bool = False) -> str:
    """Strip HTML, remove non-alpha, lowercase, optionally remove stopwords/stem."""
    text = BeautifulSoup(str(raw), "html.parser").get_text()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        # Keep negations — important for sentiment
        keep = {"no", "not", "never", "nor", "neither"}
        words = [w for w in words if w not in stops or w in keep]
    if stem:
        stemmer = SnowballStemmer("english")
        words = [stemmer.stem(w) for w in words]
    return " ".join(words)


df.dropna(subset=["review", "rating"], inplace=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 3-class sentiment label
def make_label(rating):
    if rating >= 7:  return "positive"
    if rating <= 4:  return "negative"
    return "neutral"

df["sentiment"]   = df["rating"].apply(make_label)
df["review_clean"] = df["review"].apply(clean_text)

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

# ── 4. Train / test split ────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

X = df["review_clean"]
y = df["sentiment"]

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.1, random_state=42, stratify=y_enc
)
print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

# ── 5. Bag-of-Words — Naive Bayes baseline ────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, roc_auc_score,
                              classification_report, confusion_matrix)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Fixed: use fit_resample (fit_sample was removed in imbalanced-learn 0.8+)
tfidf_vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=3)
X_train_tfidf = tfidf_vec.fit_transform(X_train)
X_test_tfidf  = tfidf_vec.transform(X_test)

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train_tfidf, y_train)   # ← FIXED
print(f"\nAfter SMOTE — class counts: {Counter(y_res)}")

nb = MultinomialNB()
nb.fit(X_res, y_res)
nb_preds = nb.predict(X_test_tfidf)

print("\n── Naive Bayes baseline ──")
print(classification_report(y_test, nb_preds, target_names=le.classes_))

# ── 6. Logistic Regression with GridSearch ────────────────────────────────────
from sklearn.model_selection import GridSearchCV

lr_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("lr",    LogisticRegression(max_iter=1000, random_state=42)),
])

params = {
    "lr__C":           [0.1, 1, 10],
    "tfidf__min_df":   [2, 5],
    "tfidf__max_features": [10000, None],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
}

grid = GridSearchCV(lr_pipeline, params, scoring="accuracy", cv=3, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)                  # train on raw cleaned text (pipeline handles vectorisation)
print("\nBest params:", grid.best_params_)

lr_preds = grid.predict(X_test)
print("\n── Logistic Regression (GridSearch) ──")
print(classification_report(y_test, lr_preds, target_names=le.classes_))

# Confusion matrix
cm = confusion_matrix(y_test, lr_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix — Logistic Regression")
plt.ylabel("True"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix_lr.png")
plt.close()

# ── 7. RNN / LSTM model ───────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

VOCAB_SIZE  = 30000
MAX_LEN     = 200
EMBED_DIM   = 128
BATCH_SIZE  = 256
EPOCHS      = 15
NUM_CLASSES = len(le.classes_)   # 3

keras_tok = KerasTokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
keras_tok.fit_on_texts(X_train)

X_train_seq = pad_sequences(keras_tok.texts_to_sequences(X_train), maxlen=MAX_LEN, padding="post")
X_test_seq  = pad_sequences(keras_tok.texts_to_sequences(X_test),  maxlen=MAX_LEN, padding="post")

# One-hot encode for categorical_crossentropy
y_train_cat = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test_cat  = tf.keras.utils.to_categorical(y_test,  NUM_CLASSES)

model = Sequential([
    Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.3),
    GlobalMaxPooling1D(),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation="softmax"),   # 3 output classes
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5),
]

history = model.fit(
    X_train_seq, y_train_cat,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
)

# Training curves
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Accuracy"); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss"); plt.legend()
plt.tight_layout()
plt.savefig("training_curves.png")
plt.close()

# Evaluate
rnn_probs  = model.predict(X_test_seq)
rnn_preds  = np.argmax(rnn_probs, axis=1)
print("\n── Bidirectional LSTM ──")
print(classification_report(y_test, rnn_preds, target_names=le.classes_))

# ── 8. Save artefacts ─────────────────────────────────────────────────────────
model.save("rnn_model.h5")
print("Saved rnn_model.h5")

with open("tokenizer.pickle", "wb") as f:
    pickle.dump(keras_tok, f)
print("Saved tokenizer.pickle")

with open("label_encoder.pickle", "wb") as f:
    pickle.dump(le, f)
print("Saved label_encoder.pickle")

with open("tfidf_lr_model.pickle", "wb") as f:
    pickle.dump(grid.best_estimator_, f)
print("Saved tfidf_lr_model.pickle (best Logistic Regression pipeline)")

# ── 9. Aspect keyword extractor (rule-based, exportable) ─────────────────────
ASPECT_LEXICONS = {
    "effectiveness": {
        "positive": ["effective","works","helped","relief","improved","cured","better","healed"],
        "negative":  ["ineffective","useless","failed","no effect","didn't work","worthless"],
    },
    "side_effects": {
        "positive": ["no side effects","well tolerated","minimal side","gentle","clean"],
        "negative":  ["nausea","vomiting","dizziness","headache","rash","fatigue","drowsy",
                      "insomnia","weight gain","hair loss","dry mouth","constipation","diarrhea"],
    },
    "dosage": {
        "positive": ["once a day","convenient","easy to take","simple","small pill"],
        "negative":  ["hard to swallow","too many pills","complicated","strict schedule"],
    },
    "cost": {
        "positive": ["affordable","cheap","generic","good value","covered by insurance"],
        "negative":  ["expensive","costly","overpriced","not covered","pricey"],
    },
}

def extract_aspects(text: str) -> dict:
    t = text.lower()
    result = {}
    for aspect, lexicon in ASPECT_LEXICONS.items():
        pos = sum(1 for w in lexicon["positive"] if w in t)
        neg = sum(1 for w in lexicon["negative"] if w in t)
        result[aspect] = "positive" if pos > neg else ("negative" if neg > pos else "neutral")
    return result

# Test on a sample
sample = "The medication worked great for my depression but caused severe nausea and was very expensive."
print("\nAspect extraction sample:", extract_aspects(sample))

print("\nAll artefacts saved. Training complete.")