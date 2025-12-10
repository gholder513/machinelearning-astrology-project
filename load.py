"""
Utilities for loading and cleaning the horoscope dataset.
"""

import re
import pandas as pd
from config import HOROSCOPE_CSV


def clean_text(text: str) -> str:
    """
    Text normalizer used before embedding:
    - lowercase
    - normalize fancy apostrophes to plain '
    - keep letters, spaces, and apostrophes
    - collapse whitespace

    This preserves contractions like "don't" instead of turning them into "don".
    """
    if not isinstance(text, str):
        return ""

    # Lowercase + normalize “smart quotes”
    text = text.lower()
    text = (
        text.replace("’", "'")
            .replace("`", "'")
            .replace("‘", "'")
            .replace("“", '"')
            .replace("”", '"')
    )

    # Keep letters, spaces, and apostrophes; drop other punctuation
    text = re.sub(r"[^a-z'\s]", " ", text)
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

    # Original description with is kept in 'description'
    # also provide a cleaned version for embedding / analysis
    df["clean_description"] = df["description"].astype(str).apply(clean_text)
    return df
