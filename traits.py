"""
Embedding-based trait extraction and sign centroids.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer

from config import *

# Load spaCy model once (for noun-phrase extraction)
# Make sure you've run: python -m spacy download en_core_web_sm
_nlp = spacy.load("en_core_web_sm")

# Tokens we don't want dominating our traits
BORING_TOKENS = {
    "today",
    "tonight",
    "tomorrow",
    "yesterday",
    "day",
    "week",
    "time",
    "things",
    "thing",
    "something",
    "anything",
    "everything",
    "nothing",
    "someone",
    "anyone",
    "everyone",
    "people",
    "person",
    "way",
    "stuff",
    "lot",
    "little",
    "bit",
    "kind",
    "sort",
    "friend",
    "friends",
    "relationship",
    "problem",
    "problems",
}


PRONOUN_POS = {"PRON", "DET"}


def load_embedding_model() -> SentenceTransformer:
    """
    Load the sentence-transformers model.
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model


def _normalize_vec(v: np.ndarray) -> np.ndarray:
    """
    L2-normalize a vector to unit length (for cosine similarity via dot product).
    """
    norm = np.linalg.norm(v)
    if norm == 0.0:
        return v
    return v / norm


def embed_descriptions(
    model: SentenceTransformer,
    texts: List[str],
) -> np.ndarray:
    """
    Embed a list of texts into a 2D array (num_texts x dim).
    We normalize embeddings so dot product ~= cosine similarity.
    """
    print(f"Computing embeddings for {len(texts)} descriptions...")
    embs = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    # normalize
    embs = np.vstack([_normalize_vec(v) for v in embs])
    return embs


def compute_sign_centroids(
    signs: np.ndarray,
    embeddings: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Given one embedding per description, compute the centroid per sign.
    """
    centroids: Dict[str, np.ndarray] = {}
    unique_signs = sorted(set(signs))

    for sign in unique_signs:
        idx = np.where(signs == sign)[0]
        sign_embs = embeddings[idx]
        centroid = sign_embs.mean(axis=0)
        centroid = _normalize_vec(centroid)
        centroids[sign] = centroid

    return centroids


def _extract_candidate_phrases(text: str) -> List[str]:
    """
    Use spaCy to extract decent noun-phrase style candidates from a description.
    We try to avoid:
      - starting with pronouns/determiners ("you", "your", "this", "that")
      - very long phrases
      - chunks where the root isn't a NOUN/PROPN
    """
    doc = _nlp(text)
    candidates: List[str] = []

    for chunk in doc.noun_chunks:
        root = chunk.root
        first = chunk[0]

        # We want "real" content-y noun phrases.
        if root.pos_ not in ("NOUN", "PROPN"):
            continue
        # Avoid pronoun/determiner-leading phrases ("you", "your", "this")
        if first.pos_ in PRONOUN_POS:
            continue

        # Token length constraints
        if not (MIN_PHRASE_TOKENS <= len(chunk) <= MAX_PHRASE_TOKENS):
            continue

        phrase = chunk.text.strip()
        if phrase:
            candidates.append(phrase)

    # deduplicate while preserving order
    seen = set()
    unique_candidates: List[str] = []
    for p in candidates:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            unique_candidates.append(p)

    return unique_candidates


def _is_good_phrase(phrase: str) -> bool:
    """
    Filter out phrases that are mostly boring tokens.
    """
    tokens = [
        t.strip(".,!?;:").lower()
        for t in phrase.split()
        if t.strip(".,!?;:")
    ]
    if not tokens:
        return False

    # At least one token must be non-boring and alphabetic
    has_content = False
    for t in tokens:
        if not any(ch.isalpha() for ch in t):
            continue
        if t not in BORING_TOKENS:
            has_content = True
            break
    return has_content


def extract_sign_traits(
    model: SentenceTransformer,
    df,
    signs: np.ndarray,
    embeddings: np.ndarray,
    centroids: Dict[str, np.ndarray],
    n_traits: int | None = None,
) -> Dict[str, List[str]]:
    """
    For each sign:
      1. Find descriptions whose embeddings are closest to that sign's centroid.
      2. From the top descriptions, extract noun-phrase candidates with spaCy.
      3. Embed each candidate phrase and score it by similarity to the centroid.
      4. Choose the top n_traits phrases as "traits" for interpretability.

    Returns: dict[sign] -> [trait1, trait2, trait3]
    """
    if n_traits is None:
        n_traits = N_TRAITS_PER_SIGN

    traits_per_sign: Dict[str, List[str]] = {}
    unique_signs = sorted(set(signs))

    for sign in unique_signs:
        print(f"  Mining traits for sign: {sign}")
        idx = np.where(signs == sign)[0]
        sign_embs = embeddings[idx]
        sign_texts = df["description"].iloc[idx].tolist()

        centroid = centroids[sign]

        # Cosine similarity via dot product (all vectors are normalized)
        sims = sign_embs @ centroid
        # Take indices of most representative descriptions
        top_k = min(TOP_DESCRIPTIONS_PER_SIGN_FOR_TRAITS, len(idx))
        top_indices = np.argsort(-sims)[:top_k]

        candidate_phrases: List[str] = []

        for local_i in top_indices:
            desc_text = sign_texts[local_i]
            phrases = _extract_candidate_phrases(desc_text)
            for p in phrases:
                if _is_good_phrase(p):
                    candidate_phrases.append(p)

        # Deduplicate while preserving order
        seen = set()
        candidate_phrases_unique: List[str] = []
        for p in candidate_phrases:
            key = p.lower()
            if key not in seen:
                seen.add(key)
                candidate_phrases_unique.append(p)

        if not candidate_phrases_unique:
            traits_per_sign[sign] = []
            continue

        # Score each phrase by similarity to centroid using embeddings
        phrase_embs = model.encode(
            candidate_phrases_unique,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        phrase_embs = np.vstack([_normalize_vec(v) for v in phrase_embs])
        phrase_scores = phrase_embs @ centroid  # cosine via dot

        scored = list(zip(candidate_phrases_unique, phrase_scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        top_traits = [p for p, _ in scored[:n_traits]]
        traits_per_sign[sign] = top_traits

    return traits_per_sign


def prepare_embeddings_and_traits(df) -> Tuple[
    SentenceTransformer,
    Dict[str, np.ndarray],
    Dict[str, List[str]],
]:
    """
    High-level helper:
      - load embedding model
      - embed all descriptions
      - compute sign centroids
      - extract traits per sign

    Returns:
      model, sign_centroids, sign_traits
    """
    model = load_embedding_model()
    texts = df["description"].astype(str).tolist()
    embeddings = embed_descriptions(model, texts)
    signs = df["sign"].values

    centroids = compute_sign_centroids(signs, embeddings)
    traits = extract_sign_traits(model, df, signs, embeddings, centroids)

    return model, centroids, traits
