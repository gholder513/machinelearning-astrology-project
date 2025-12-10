"""
Classification logic using sign centroid embeddings and cosine similarity.
"""

from typing import Dict, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from load import clean_text


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return vec
    return vec / norm


def classify_with_embeddings(
    text: str,
    model: SentenceTransformer,
    sign_centroids: Dict[str, np.ndarray],
) -> Tuple[str | None, Dict[str, float]]:
    """
    Classify a new description using cosine similarity between the
    embedding of the text and each sign's centroid embedding.

    Returns:
      best_sign (or None if we have no centroids),
      all_similarities_dict (sign -> cosine similarity)
    """
    if not sign_centroids:
        return None, {}

    clean = clean_text(text)
    # We use a list here so encode returns shape (1, dim)
    emb = model.encode([clean], show_progress_bar=False)
    emb = l2_normalize(np.asarray(emb)[0])  # (dim,)

    sims: Dict[str, float] = {}
    for sign, centroid in sign_centroids.items():
        centroid = np.asarray(centroid)
        # assuming centroids are already normalized
        sims[sign] = float(emb @ centroid)

    if not sims:
        return None, {}

    best_sign = max(sims, key=sims.get)
    return best_sign, sims
