"""
Utilities for loading and cleaning the horoscope dataset.
"""

import re
import pandas as pd
from config import HOROSCOPE_CSV


def clean_text(text: str) -> str:
    """
    Light text normalizer:

    - ensure string
    - normalize fancy quotes
    - lowercase
    - collapse whitespace

    We *do not* strip punctuation or apostrophes so that
    sentence transformers have the richest possible input.
    """
    if not isinstance(text, str):
        return ""

    # Normalize some common unicode quotes
    text = (
        text.replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
    )

    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_horoscopes(csv_path: str | None = None) -> pd.DataFrame:
    """
    Load the horoscope CSV and return a DataFrame with:
      columns: ['sign', 'description', 'clean_description']

    Expects columns in the CSV: 'sign', 'description'.
    """
    if csv_path is None:
        csv_path = HOROSCOPE_CSV

    df = pd.read_csv(csv_path)
    # standardize sign names
    df["sign"] = df["sign"].astype(str).str.lower().str.strip()

    # ensure we have the expected columns
    if "description" not in df.columns or "sign" not in df.columns:
        raise ValueError("CSV must contain 'description' and 'sign' columns.")

    # Cleaned version mainly for TF-IDF; embeddings can use raw text if desired
    df["clean_description"] = df["description"].apply(clean_text)
    return df
