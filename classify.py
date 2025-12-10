"""
Classification logic using embedding-based sign centroids,
plus helper functions for interpretability.
"""

from typing import Dict, Tuple, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from load import clean_text


def embed_text(
    model: SentenceTransformer,
    text: str,
) -> np.ndarray:
    """
    Embed a single text into a 1 x D vector.
    We reuse clean_text so apostrophes are preserved and normalized.
    """
    cleaned = clean_text(text)
    emb = model.encode(
        [cleaned],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )  # shape: (1, D)
    return emb


def classify_with_embeddings(
    text: str,
    model: SentenceTransformer,
    sign_centroids: Dict[str, np.ndarray],
) -> Tuple[str | None, Dict[str, float], np.ndarray]:
    """
    Classify a new description using cosine similarity between the
    embedding of the text and each sign's centroid embedding.

    Returns:
      best_sign (or None if we have no signal),
      dict[sign] -> cosine similarity score,
      query_embedding (1 x D)
    """
    print("\nClassifying text using cosine similarity in embedding space:")
    print("  ", text)

    query_emb = embed_text(model, text)  # (1, D)

    sims: Dict[str, float] = {}
    for sign, centroid in sign_centroids.items():
        sim = float(cosine_similarity(query_emb, centroid)[0][0])
        sims[sign] = sim

    if not sims:
        return None, {}, query_emb

    max_sim = max(sims.values())
    if max_sim <= 0.0:
        # basically no meaningful alignment with any sign centroid
        return None, sims, query_emb

    best_sign = max(sims, key=sims.get)
    return best_sign, sims, query_emb


def top_examples_for_sign(
    sign: str,
    query_emb: np.ndarray,
    descriptions: List[str],
    embeddings: np.ndarray,
    signs: np.ndarray,
    k: int = 3,
) -> List[Tuple[str, float]]:
    """
    For a given sign and query embedding, return the top-k training
    descriptions for that sign, sorted by cosine similarity to the query.
    """
    idxs = np.where(signs == sign)[0]
    if len(idxs) == 0:
        return []

    sign_vecs = embeddings[idxs]  # (m, D)

    # If everything is normalized, dot product == cosine similarity
    sims = (sign_vecs @ query_emb.T).ravel()  # (m,)

    order = np.argsort(sims)[::-1][:k]
    results: List[Tuple[str, float]] = []
    for rel_idx in order:
        abs_idx = idxs[rel_idx]
        results.append((descriptions[abs_idx], float(sims[rel_idx])))

    return results
