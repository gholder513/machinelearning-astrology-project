"""
Global configuration: paths and constants.
"""

from pathlib import Path

# Project root (directory containing this file)
PROJECT_ROOT = Path(__file__).resolve().parent

# Path to the horoscope CSV
HOROSCOPE_CSV = PROJECT_ROOT / "horoscope.csv"

# Embedding model settings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32  # used when encoding many descriptions

# How many traits per sign to extract for interpretability
N_TRAITS_PER_SIGN = 3

# How many of the most typical descriptions (per sign) to mine for traits
TOP_DESCRIPTIONS_PER_SIGN_FOR_TRAITS = 10

# Phrase constraints
MAX_PHRASE_TOKENS = 4
MIN_PHRASE_TOKENS = 1
