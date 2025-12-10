"""
Utilities for loading the horoscope dataset.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd
from config import *


def clean_text(text: str) -> str:
    """
    Simple text normalizer for any classical methods (if you use them later).
    - lowercase
    - keep letters, spaces, apostrophes
    - collapse whitespace

    NOTE: The embedding-based pipeline uses the *raw* description text
    (with punctuation and apostrophes) so this is not used there.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # keep letters, spaces, apostrophes
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_horoscopes(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the horoscope CSV and return a DataFrame with at least:
      ['sign', 'description']

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

    # keep raw description for embeddings; optionally keep a cleaned version
    df["description"] = df["description"].astype(str)
    df["clean_description"] = df["description"].apply(clean_text)
    return df
