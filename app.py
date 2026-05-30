"""
Enhanced Drug Recommendation System
Upgrades over original:
  - Aspect-Based Sentiment Analysis (effectiveness, side effects, dosage, cost)
  - Confidence score with probability bar
  - Top drug recommendations based on condition keywords
  - XAI explanation layer (key sentiment words highlighted)
  - Proper tokenizer inference (no re-fitting on test input)
  - Deprecated predict_classes() replaced
  - REST API endpoint (/api/predict) for JSON consumers
  - Drug interaction safety warning via keyword rules
"""

from flask import Flask, render_template, url_for, request, jsonify
import numpy as np
import pickle
import re
from collections import defaultdict
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from numpy import array

app = Flask(__name__)

# ── Model & tokenizer ────────────────────────────────────────────────────────
model = load_model("rnn_model.h5")

with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

MAX_LENGTH = 200

# ── Aspect keyword lexicons ──────────────────────────────────────────────────
ASPECT_LEXICONS = {
    "effectiveness": {
        "positive": ["effective", "works", "helped", "relief", "improved", "cured",
                     "better", "amazing", "miracle", "healed", "resolved", "cleared"],
        "negative": ["ineffective", "useless", "didn't work", "no effect", "failed",
                     "worthless", "pointless", "didn't help", "no improvement"],
    },
    "side_effects": {
        "positive": ["no side effects", "no side-effects", "tolerated", "well tolerated",
                     "minimal side", "clean", "gentle"],
        "negative": ["nausea", "vomiting", "dizzy", "dizziness", "headache", "rash",
                     "fatigue", "tired", "drowsy", "insomnia", "anxiety", "depression",
                     "weight gain", "hair loss", "dry mouth", "constipation", "diarrhea",
                     "side effect", "side-effect", "adverse", "reaction"],
    },
    "dosage": {
        "positive": ["easy to take", "once a day", "convenient", "small pill",
                     "simple dosage", "easy dosing"],
        "negative": ["hard to swallow", "too many pills", "complicated", "confusing",
                     "multiple doses", "strict schedule", "missed dose"],
    },
    "cost": {
        "positive": ["affordable", "cheap", "inexpensive", "generic", "worth it",
                     "good value", "covered by insurance", "low cost"],
        "negative": ["expensive", "costly", "overpriced", "not covered", "pricey",
                     "unaffordable", "out of pocket"],
    },
}

# Positive/negative word lists for XAI keyword highlighting
POSITIVE_WORDS = {
    "great", "excellent", "wonderful", "amazing", "fantastic", "good", "helpful",
    "effective", "works", "better", "improved", "relief", "recommend", "love",
    "best", "perfect", "positive", "happy", "pleased", "comfortable", "safe",
}
NEGATIVE_WORDS = {
    "terrible", "awful", "horrible", "bad", "worst", "useless", "failed",
    "pain", "nausea", "sick", "dizzy", "headache", "rash", "severe",
    "dangerous", "harmful", "avoid", "never", "stopped", "discontinued",
}

# ── Drug condition → recommendation map ─────────────────────────────────────
CONDITION_DRUG_MAP = {
    "depression": [
        {"name": "Sertraline (Zoloft)", "rating": 8.2, "reviews": 1240, "badge": "Most reviewed"},
        {"name": "Escitalopram (Lexapro)", "rating": 8.0, "reviews": 980, "badge": "High satisfaction"},
        {"name": "Bupropion (Wellbutrin)", "rating": 7.7, "reviews": 860, "badge": None},
    ],
    "anxiety": [
        {"name": "Buspirone", "rating": 7.9, "reviews": 710, "badge": "Non-addictive"},
        {"name": "Sertraline (Zoloft)", "rating": 7.8, "reviews": 690, "badge": None},
        {"name": "Venlafaxine (Effexor)", "rating": 7.5, "reviews": 520, "badge": None},
    ],
    "pain": [
        {"name": "Ibuprofen", "rating": 8.5, "reviews": 2100, "badge": "OTC"},
        {"name": "Naproxen", "rating": 8.1, "reviews": 1340, "badge": None},
        {"name": "Celecoxib (Celebrex)", "rating": 7.9, "reviews": 780, "badge": None},
    ],
    "diabetes": [
        {"name": "Metformin", "rating": 8.7, "reviews": 1890, "badge": "First-line"},
        {"name": "Sitagliptin (Januvia)", "rating": 8.0, "reviews": 670, "badge": None},
        {"name": "Empagliflozin (Jardiance)", "rating": 8.3, "reviews": 590, "badge": "Cardio benefit"},
    ],
    "hypertension": [
        {"name": "Lisinopril", "rating": 8.4, "reviews": 1560, "badge": "Most prescribed"},
        {"name": "Amlodipine", "rating": 8.1, "reviews": 1200, "badge": None},
        {"name": "Losartan", "rating": 8.0, "reviews": 980, "badge": None},
    ],
    "acne": [
        {"name": "Doxycycline", "rating": 8.0, "reviews": 870, "badge": None},
        {"name": "Tretinoin (Retin-A)", "rating": 8.6, "reviews": 1430, "badge": "Top rated"},
        {"name": "Clindamycin topical", "rating": 7.8, "reviews": 620, "badge": None},
    ],
    "insomnia": [
        {"name": "Melatonin", "rating": 7.6, "reviews": 1100, "badge": "OTC"},
        {"name": "Zolpidem (Ambien)", "rating": 7.9, "reviews": 890, "badge": None},
        {"name": "Trazodone", "rating": 8.1, "reviews": 740, "badge": "Non-habit forming"},
    ],
}

CONDITION_KEYWORDS = {
    "depression": ["depress", "sad", "mood", "antidepressant", "ssri"],
    "anxiety": ["anxi", "panic", "worry", "stress", "phobia"],
    "pain": ["pain", "ache", "hurt", "inflamm", "arthritis", "migraine", "headache"],
    "diabetes": ["diabet", "blood sugar", "insulin", "glucose", "a1c"],
    "hypertension": ["blood pressure", "hypertens", "bp", "heart"],
    "acne": ["acne", "pimple", "breakout", "skin", "blemish"],
    "insomnia": ["sleep", "insomnia", "awake", "rest", "tired"],
}

# ── Drug interaction safety flags ───────────────────────────────────────────
INTERACTION_FLAGS = {
    "alcohol": "⚠️ Combining this drug with alcohol may cause severe drowsiness or liver damage.",
    "warfarin": "⚠️ Interaction with blood thinners (warfarin) — consult your doctor.",
    "maoi": "⚠️ Dangerous interaction with MAOIs reported. Seek medical advice immediately.",
    "pregnancy": "⚠️ Some drugs in this category may be unsafe during pregnancy.",
    "kidney": "⚠️ Dosage adjustment required for kidney/renal conditions.",
}

# ── Helper functions ─────────────────────────────────────────────────────────

def preprocess(text: str) -> np.ndarray:
    """Tokenize and pad — uses pre-fitted tokenizer vocabulary only."""
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding="post")
    return array(padded)


def predict_sentiment(text: str) -> dict:
    """Return class (0/1), probability, and label."""
    inp = preprocess(text)
    prob = float(model.predict(inp, verbose=0)[0][0])
    label = 1 if prob >= 0.5 else 0
    confidence = prob if label == 1 else (1 - prob)
    return {
        "label": label,
        "confidence": round(confidence * 100, 1),
        "probability_positive": round(prob * 100, 1),
        "probability_negative": round((1 - prob) * 100, 1),
    }


def aspect_sentiment(text: str) -> dict:
    """Score each aspect as positive, negative, or neutral."""
    text_lower = text.lower()
    results = {}
    for aspect, lexicon in ASPECT_LEXICONS.items():
        pos = sum(1 for w in lexicon["positive"] if w in text_lower)
        neg = sum(1 for w in lexicon["negative"] if w in text_lower)
        if pos > neg:
            score = "positive"
            icon = "✅"
        elif neg > pos:
            score = "negative"
            icon = "❌"
        else:
            score = "neutral"
            icon = "➖"
        results[aspect] = {"score": score, "icon": icon, "pos": pos, "neg": neg}
    return results


def highlight_keywords(text: str) -> str:
    """Wrap positive/negative sentiment words in <mark> tags for XAI display."""
    words = text.split()
    highlighted = []
    for word in words:
        clean = re.sub(r"[^a-zA-Z]", "", word).lower()
        if clean in POSITIVE_WORDS:
            highlighted.append(f'<mark class="pos-word">{word}</mark>')
        elif clean in NEGATIVE_WORDS:
            highlighted.append(f'<mark class="neg-word">{word}</mark>')
        else:
            highlighted.append(word)
    return " ".join(highlighted)


def detect_condition(text: str) -> tuple:
    """Match review text to closest medical condition."""
    text_lower = text.lower()
    scores = defaultdict(int)
    for condition, keywords in CONDITION_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[condition] += 1
    if not scores:
        return None, []
    best = max(scores, key=scores.get)
    return best, CONDITION_DRUG_MAP.get(best, [])


def check_interactions(text: str) -> list:
    """Return any relevant safety warnings based on keyword matches."""
    text_lower = text.lower()
    warnings = []
    for keyword, warning in INTERACTION_FLAGS.items():
        if keyword in text_lower:
            warnings.append(warning)
    return warnings


def build_explanation(sentiment: dict, aspects: dict, review: str) -> str:
    """Generate a plain-English explanation of the prediction."""
    label_word = "positive" if sentiment["label"] == 1 else "negative"
    conf = sentiment["confidence"]
    parts = [f"The review was classified as <strong>{label_word}</strong> with "
             f"<strong>{conf}%</strong> confidence."]
    positives = [k for k, v in aspects.items() if v["score"] == "positive"]
    negatives = [k for k, v in aspects.items() if v["score"] == "negative"]
    if positives:
        readable = [a.replace("_", " ") for a in positives]
        parts.append(f"Positive signals found in: {', '.join(readable)}.")
    if negatives:
        readable = [a.replace("_", " ") for a in negatives]
        parts.append(f"Concerns flagged in: {', '.join(readable)}.")
    return " ".join(parts)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    review = request.form.get("review", "").strip()
    if not review:
        return render_template("home.html", error="Please enter a review.")

    sentiment   = predict_sentiment(review)
    aspects     = aspect_sentiment(review)
    highlighted = highlight_keywords(review)
    explanation = build_explanation(sentiment, aspects, review)
    condition, recommendations = detect_condition(review)
    interactions = check_interactions(review)

    return render_template(
        "result.html",
        review=review,
        highlighted_review=highlighted,
        sentiment=sentiment,
        aspects=aspects,
        explanation=explanation,
        condition=condition,
        recommendations=recommendations,
        interactions=interactions,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON REST endpoint for external consumers."""
    data = request.get_json(force=True)
    review = data.get("review", "").strip()
    if not review:
        return jsonify({"error": "No review provided"}), 400

    sentiment    = predict_sentiment(review)
    aspects      = aspect_sentiment(review)
    explanation  = build_explanation(sentiment, aspects, review)
    condition, _ = detect_condition(review)
    interactions = check_interactions(review)

    return jsonify({
        "sentiment": sentiment,
        "aspects": aspects,
        "explanation": explanation,
        "detected_condition": condition,
        "safety_warnings": interactions,
    })


if __name__ == "__main__":
    app.run(debug=True)