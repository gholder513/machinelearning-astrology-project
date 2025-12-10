"""
Global configuration: paths and constants.
"""

from pathlib import Path

# Project root (directory containing this file)
PROJECT_ROOT = Path(__file__).resolve().parent

# Path to the horoscope CSV
HOROSCOPE_CSV = PROJECT_ROOT / "horoscope.csv"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# How many traits per sign to extract for interpretability
N_TRAITS_PER_SIGN = 3

# How many similar training examples to show for the predicted sign
N_TOP_EXAMPLES = 3

# How many similarity scores to display for the embedding classifier
TOP_SIMILARITY_EXPLAIN = 12

#  Generic TF-IDF defaults (if you still use them elsewhere) 
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)

# Random Forest settings 
RF_TFIDF_MAX_FEATURES = 5000
RF_TFIDF_NGRAM_RANGE = (1, 2)
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = None  # or an int if you want to regularize tree depth
RF_TEST_SIZE = 0.2   # 20% hold-out test split
RF_RANDOM_STATE = 42
