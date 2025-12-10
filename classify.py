"""
Classification logic using sign centroid vectors and cosine similarity.
"""

from typing import Dict, List, Tuple
import numpy as np
from sentence_transformers import util

from load import clean_text


def classify_with_centroids(
    text: str,
    model,
    centroids: Dict[str, np.ndarray],
) -> Tuple[str | None, Dict[str, float], np.ndarray]:
    """
    Classify a new description using cosine similarity between the
    query embedding and each sign's centroid vector.

    Returns:
      best_sign (or None if we have no signal),
      all_similarities_dict,
      query_emb (the embedding of the input text)
    """
    clean = clean_text(text)
    query_emb = model.encode(clean, convert_to_numpy=True)

    sims: Dict[str, float] = {}
    for sign, centroid in centroids.items():
        # centroid is shape (1, D) or (D,)
        sim = float(util.cos_sim(query_emb, centroid)[0][0])
        sims[sign] = sim

    if not sims:
        return None, {}, query_emb

    max_sim = max(sims.values())
    if max_sim <= 0.0:
        # everything basically unrelated
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
    Return top-k most similar training examples for the given sign.

    Each result is (description_text, similarity_score).
    Similarity is cosine similarity between query_emb and each
    embedding of that sign.
    """
    # mask: which rows belong to this sign
    mask = signs == sign
    sign_embs = embeddings[mask]
    sign_texts = np.array(descriptions)[mask]

    if sign_embs.size == 0:
        return []

    sims = util.cos_sim(query_emb, sign_embs).cpu().numpy().ravel()
    top_idx = sims.argsort()[::-1][:k]

    results: List[Tuple[str, float]] = []
    for i in top_idx:
        results.append((sign_texts[i], float(sims[i])))
    return results
