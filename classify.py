"""
Classification logic using sign centroid embeddings.
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


def classify_with_centroids(
    text: str,
    model: SentenceTransformer,
    sign_centroids: Dict[str, np.ndarray],
) -> Tuple[Optional[str], Dict[str, float]]:
    """
    Classify a new description using cosine similarity between the
    embedding of the text and each sign's centroid embedding.

    Returns:
      best_sign (or None if we have no centroids),
      all_similarities_dict
    """
    text = text.strip()
    if not text:
        return None, {}

    # Embed and normalize
    emb = model.encode([text], convert_to_numpy=True)[0]
    norm = np.linalg.norm(emb)
    if norm == 0.0:
        return None, {}
    emb = emb / norm

    sims: Dict[str, float] = {}
    for sign, centroid in sign_centroids.items():
        # centroid is already normalized; dot product ~= cosine similarity
        sims[sign] = float(emb @ centroid)

    if not sims:
        return None, {}

    best_sign = max(sims, key=sims.get)
    return best_sign, sims
