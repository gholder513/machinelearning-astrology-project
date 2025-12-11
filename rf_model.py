# rf_model.py
"""
Random Forest pipeline for zodiac sign classification based on descriptions.
"""

from __future__ import annotations

from typing import Tuple, Dict, Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from load import load_horoscopes, clean_text
from config import *


def train_random_forest_from_csv(
    csv_path: str | None = None,
) -> Tuple[
    TfidfVectorizer,
    LabelEncoder,
    RandomForestClassifier,
    Dict[str, Any],
    float,
]:
    """
    Train a Random Forest classifier on horoscope descriptions -> sign.

    returns vectorizer, label_encoder, clf,
      rf_report_dict: sklearn classification_report as a dict
      rf_accuracy: overall accuracy from the report
    """
    # Load data
    df = load_horoscopes(csv_path)
    print(f"Loaded {len(df)} rows from {HOROSCOPE_CSV}")

    # X: cleaned descriptions, y: sign labels
    X_text = df["clean_description"].tolist()
    y_str = df["sign"].astype(str).tolist()

    # Vectorize text with TF-IDF
    print("Building TF-IDF features for Random Forest...")
    vectorizer = TfidfVectorizer(
        max_features=RF_TFIDF_MAX_FEATURES,
        ngram_range=RF_TFIDF_NGRAM_RANGE,
        stop_words="english",
    )
    X = vectorizer.fit_transform(X_text)

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=RF_TEST_SIZE,
        random_state=RF_RANDOM_STATE,
        stratify=y,
    )

    # Train Random Forest
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

    # Overall accuracy from accuracy_score (for printing)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nRandom Forest accuracy on held-out test set: {acc:.4f}\n")

    # Text report for console
    target_names = label_encoder.inverse_transform(sorted(set(y)))
    print("Classification report (by zodiac sign):")
    text_report = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    print(text_report)

    # Report as a dict for return
    rf_report_dict: Dict[str, Any] = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    rf_accuracy: float = float(rf_report_dict["accuracy"])

    return vectorizer, label_encoder, clf, rf_report_dict, rf_accuracy


def predict_sign_rf(
    text: str,
    vectorizer: TfidfVectorizer,
    label_encoder: LabelEncoder,
    clf: RandomForestClassifier,
) -> Tuple[str, Dict[str, float]]:
    """
    Predict zodiac sign for a new description using the trained Random Forest.

    retruns: best_sign, proba_dict
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
