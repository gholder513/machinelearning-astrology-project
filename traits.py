"""
Embedding-based utilities:
- load the sentence-transformers model
- compute description embeddings
- build sign centroids
- extract short "trait" phrases per sign for interpretability
"""

from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_BATCH_SIZE,
    N_TRAITS_PER_SIGN,
)
from load import clean_text

# Words that we don't want as "traits"
BORING_WORDS = {
    # generic function words / fillers
    "just", "really", "very", "quite", "thing", "things", "something",
    "anything", "everything", "nothing", "way", "time", "day", "days",
    "right", "soon", "now", "then", "back", "away", "around", "maybe",
    "bit", "little", "lot", "kind", "sort",
    # vague verbs
    "make", "makes", "making", "take", "takes", "taking",
    "get", "gets", "getting", "go", "goes", "going",
    "know", "knows", "knowing", "feel", "feels", "feeling",
    "want", "wants", "wanted", "need", "needs", "needed",
    # generic adjectives
    "good", "bad", "better", "best", "nice", "new", "old",
    "big", "small", "great",
    # pronouns / generic people words
    "someone", "everyone", "anyone", "people", "person",
    "youre", "you", "your", "they", "them", "their",
    # horoscope boilerplate
    "today", "tonight", "tomorrow", "yesterday",
    "mood", "star", "stars",
    # contractions we don't want as traits
    "dont", "don't", "doesnt", "doesn't", "cant", "can't",
    "isnt", "isn't", "wont", "won't", "hasnt", "hasn't",
}


def load_embedding_model() -> SentenceTransformer:
    """
    Load the sentence-transformers model specified in config.
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model


def compute_embeddings(
    model: SentenceTransformer,
    texts: List[str],
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute embeddings for a list of texts.

    We assume texts are already cleaned in a reasonable way (e.g., clean_text),
    but you can also pass raw descriptions if you want.
    """
    print(f"Computing embeddings for {len(texts)} descriptions...")
    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return embeddings  # shape: (N, D)


def compute_sign_centroids(
    signs: np.ndarray,
    embeddings: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[int]]]:
    """
    Build:
      - sign_centroids: dict[sign] -> centroid embedding (1 x D)
      - sign_to_indices: dict[sign] -> list of row indices in the dataset
    """
    sign_to_indices: Dict[str, List[int]] = {}
    for i, s in enumerate(signs):
        sign_to_indices.setdefault(s, []).append(i)

    sign_centroids: Dict[str, np.ndarray] = {}
    for sign, idxs in sign_to_indices.items():
        sign_vecs = embeddings[idxs]  # (k, D)
        centroid = sign_vecs.mean(axis=0, keepdims=True)  # (1, D)

        # Re-normalize centroid to unit length for clean cosine similarity
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        sign_centroids[sign] = centroid

    return sign_centroids, sign_to_indices


def _is_content_word(word: str) -> bool:
    """
    Decide if a token is "contentful" enough to be part of a trait.
    """
    word = word.strip("'")  # strip leading/trailing apostrophes for check
    if not word:
        return False
    if len(word) < 4:
        return False
    if word in BORING_WORDS:
        return False
    return True


def _candidate_phrases_from_text(text: str) -> List[str]:
    """
    Very simple phrase extractor:
    - clean text
    - build bigrams of "content" words
    - fallback to single content words if needed
    """
    cleaned = clean_text(text)
    tokens = cleaned.split()
    phrases: List[str] = []

    # bigrams first
    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i], tokens[i + 1]
        if _is_content_word(w1) and _is_content_word(w2):
            phrases.append(f"{w1} {w2}")

    # if we didn't find any bigrams, fall back to unigrams
    if not phrases:
        for w in tokens:
            if _is_content_word(w):
                phrases.append(w)

    return phrases


def extract_sign_traits(
    descriptions: List[str],
    signs: np.ndarray,
    embeddings: np.ndarray,
    sign_to_indices: Dict[str, List[int]],
    sign_centroids: Dict[str, np.ndarray],
    n_traits: int | None = None,
) -> Dict[str, List[str]]:
    """
    For each sign:
      - look at that sign's descriptions
      - rank each description by similarity to the sign centroid
      - from the most central descriptions, harvest simple n-gram "traits"

    Returns:
      dict[sign] -> list of short trait strings (len ~ n_traits)
    """
    if n_traits is None:
        n_traits = N_TRAITS_PER_SIGN

    traits_by_sign: Dict[str, List[str]] = {}

    for sign, idxs in sign_to_indices.items():
        centroid = sign_centroids[sign]  # (1, D)
        sign_vecs = embeddings[idxs]     # (k, D)

        # Since both are normalized, cosine similarity is just dot product
        sims = (sign_vecs @ centroid.T).ravel()  # (k,)

        # Sort indices of this sign by similarity descending
        order = np.argsort(sims)[::-1]

        selected_traits: List[str] = []
        seen: set[str] = set()

        # Look at the most central ~20 descriptions for this sign
        for rank_idx in order[:20]:
            abs_idx = idxs[rank_idx]
            desc = descriptions[abs_idx]

            # Generate candidate phrases from this description
            candidates = _candidate_phrases_from_text(desc)
            for phrase in candidates:
                if phrase not in seen:
                    seen.add(phrase)
                    selected_traits.append(phrase)
                    if len(selected_traits) >= n_traits:
                        break

            if len(selected_traits) >= n_traits:
                break

        traits_by_sign[sign] = selected_traits

    return traits_by_sign
