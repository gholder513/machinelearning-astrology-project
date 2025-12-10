"""
Global configuration: paths, constants, and model names.
"""

from pathlib import Path

# Project root (directory containing this file)
PROJECT_ROOT = Path(__file__).resolve().parent

# Path to the horoscope CSV
HOROSCOPE_CSV = PROJECT_ROOT / "horoscope.csv"

# TF-IDF settings (for trait candidate extraction) 
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)  # unigrams + bigrams

# How many traits per sign to extract for interpretability
N_TRAITS_PER_SIGN = 3

# Max number of TF-IDF candidate terms per sign before ranking with embeddings
TRAIT_CANDIDATE_MAX_TERMS = 50

#  SentenceTransformer settings 
# Good balance of speed & quality. You can swap to all-mpnet-base-v2 if you want heavier.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
