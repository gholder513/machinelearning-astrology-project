"""
Global configuration: paths and constants.
"""

from pathlib import Path
import os


# Project root (directory containing this file)
PROJECT_ROOT = Path(__file__).resolve().parent
# Path to the horoscope CSV
HOROSCOPE_CSV = PROJECT_ROOT / "horoscope.csv"

API_BASE_URL = os.environ.get("VITE_API_BASE_URL", "https://machinelearning-astrology-project.onrender.com/")

# Embedding model settings
# Name that traits.py / embedding pipeline uses
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Optional alias if some code uses EMBEDDING_MODEL instead
EMBEDDING_MODEL = EMBEDDING_MODEL_NAME

# Batch size for SentenceTransformer.encode
EMBEDDING_BATCH_SIZE = 32  # you can lower this if you hit OOM in Docker

GPT_MODEL_NAME = "gpt-4o-mini"  # or another model you have access to

# Number of evaluation rounds for the horoscope accuracy session
NUM_HOROSCOPE_ROUNDS = 10

# Traits extraction and example lookup settings
# How many traits per sign to extract for interpretability
N_TRAITS_PER_SIGN = 3

# How many similar training examples to show for the predicted sign
N_TOP_EXAMPLES = 3

# How many similarity scores to display for the embedding classifier
TOP_SIMILARITY_EXPLAIN = 12

# Phrase extraction settings (used in embedding-aware trait mining)
MIN_PHRASE_TOKENS = 1   # minimum words in a candidate phrase
MAX_PHRASE_TOKENS = 4   # maximum words in a candidate phrase

TOP_DESCRIPTIONS_PER_SIGN_FOR_TRAITS = 3  # controls how many candidate phrases to extract per sign


TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)



RF_TFIDF_MAX_FEATURES = 5000
RF_TFIDF_NGRAM_RANGE = (1, 2)
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = None  # or an int if you want to regularize tree depth
RF_TEST_SIZE = 0.3   # 30% hold-out test split
RF_RANDOM_STATE = 42

EMBEDDING_BATCH_SIZE = 32
TOP_DESCRIPTIONS_PER_SIGN_FOR_TRAITS = 3  # or however many representative examples per sign
