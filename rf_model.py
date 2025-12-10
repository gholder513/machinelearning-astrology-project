"""
Random Forest pipeline for zodiac sign classification based on descriptions.
"""

from __future__ import annotations

from typing import Tuple, Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from load import load_horoscopes, clean_text
from config import (
    HOROSCOPE_CSV,
    RF_TFIDF_MAX_FEATURES,
    RF_TFIDF_NGRAM_RANGE,
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    RF_TEST_SIZE,
    RF_RANDOM_STATE,
)


def train_random_forest_from_csv(
    csv_path: str | None = None,
) -> Tuple[TfidfVectorizer, LabelEncoder, RandomForestClassifier]:
    """
    Train a Random Forest classifier on horoscope descriptions -> sign.

    Returns:
      vectorizer: fitted TfidfVectorizer
      label_encoder: fitted LabelEncoder for sign strings
      clf: trained RandomForestClassifier
    """
    # 1. Load data
    df = load_horoscopes(csv_path)
    print(f"Loaded {len(df)} rows from {HOROSCOPE_CSV}")

    # X: cleaned descriptions, y: sign labels
    X_text = df["clean_description"].tolist()
    y_str = df["sign"].astype(str).tolist()

    # 2. Vectorize text with TF-IDF
    print("Building TF-IDF features for Random Forest...")
    vectorizer = TfidfVectorizer(
        max_features=RF_TFIDF_MAX_FEATURES,
        ngram_range=RF_TFIDF_NGRAM_RANGE,
        stop_words="english",
    )
    X = vectorizer.fit_transform(X_text)

    # 3. Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)

    # 4. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=RF_TEST_SIZE,
        random_state=RF_RANDOM_STATE,
        stratify=y,
    )

    # 5. Train Random Forest
    print("Training Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # 6. Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nRandom Forest accuracy on held-out test set: {acc:.4f}\n")
    print("Classification report (by zodiac sign):")
    target_names = label_encoder.inverse_transform(sorted(set(y)))
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
    )

    return vectorizer, label_encoder, clf


def predict_sign_rf(
    text: str,
    vectorizer: TfidfVectorizer,
    label_encoder: LabelEncoder,
    clf: RandomForestClassifier,
) -> Tuple[str, Dict[str, float]]:
    """
    Predict zodiac sign for a new description using the trained Random Forest.

    Returns:
      best_sign: predicted sign string
      proba_dict: dict of sign -> probability
    """
    clean = clean_text(text)
    X_vec = vectorizer.transform([clean])
    proba = clf.predict_proba(X_vec)[0]
    classes = label_encoder.classes_

    # Map probabilities back to sign strings
    proba_dict = {cls: float(p) for cls, p in zip(classes, proba)}
    best_sign_idx = int(np.argmax(proba))
    best_sign = classes[best_sign_idx]

    return best_sign, proba_dict
