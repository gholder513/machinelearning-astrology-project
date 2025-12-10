"""
Global configuration: paths, constants, and model settings.
"""

from pathlib import Path

# Project root (directory containing this file)
PROJECT_ROOT = Path(__file__).resolve().parent

# Path to the horoscope CSV
HOROSCOPE_CSV = PROJECT_ROOT / "horoscope.csv"

# Embedding model / batching
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32

# How many traits per sign to extract for interpretability
N_TRAITS_PER_SIGN = 3

# How many example horoscopes to show as "closest" for explanation
N_TOP_EXAMPLES = 3
