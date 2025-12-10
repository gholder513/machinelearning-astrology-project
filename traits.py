"""
Build sign-level embeddings with SentenceTransformers, extract descriptive
traits per sign using a KeyBERT-style approach, and expose helpers
for main/classification.

Pipeline:
- Load embedding model
- Compute per-document embeddings
- Compute per-sign centroid embeddings
- Use TF-IDF to get candidate terms per sign
- Rank candidate terms by cosine similarity to sign centroids in embedding space
  -> pick top N_TRAITS_PER_SIGN traits per sign
"""

from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from config import (
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    N_TRAITS_PER_SIGN,
    TRAIT_CANDIDATE_MAX_TERMS,
    EMBEDDING_MODEL_NAME,
)

# Expanded list of words/phrases we don't want as traits
BORING_WORDS = {
    # generic function words / fillers
    "just", "really", "very", "quite", "thing", "things", "something",
    "anything", "everything", "nothing", "way", "time", "day", "days",
    "right", "soon", "now", "then", "back", "away", "around", "maybe",
    "bit", "little", "lot", "kind", "sort",
    # vague verbs
    "make", "makes", "making", "take", "takes", "taking",
    "get", "gets", "getting", "go", "goes", "going",
    "know", "knows", "knowing", "let know",
    "feel", "feels", "feeling", "felt",
    "want", "wants", "wanted", "need", "needs", "needed",
    "try", "tries", "trying",
    # generic adjectives
    "good", "bad", "better", "best", "nice", "new", "old",
    "big", "small", "great", "long", "short",
    "like",
    # pronouns and people words
    "someone", "everyone", "anyone", "people", "person",
    # horoscope boilerplate
    "today", "tonight", "tomorrow", "yesterday",
    "mood", "star", "stars",
}


def load_embedding_model() -> SentenceTransformer:
    """
    Load the SentenceTransformer model specified in config.
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """
    L2-normalize a vector or matrix along the last dimension.
    """
    mat = np.asarray(mat, dtype=float)
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    norms = np.where(norms == 0.0, 1e-12, norms)
    return mat / norms


def compute_document_embeddings(
    descriptions: List[str],
    model: SentenceTransformer,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Compute and L2-normalize embeddings for each description.
    Returns: (num_docs, dim) numpy array.
    """
    print(f"Computing embeddings for {len(descriptions)} descriptions...")
    embeddings = model.encode(
        descriptions,
        batch_size=batch_size,
        show_progress_bar=True,
    )
    embeddings = l2_normalize(np.asarray(embeddings))
    return embeddings


def compute_sign_centroids(
    signs: np.ndarray,
    doc_embeddings: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute centroid embedding for each sign by averaging document embeddings.
    Returns: dict[sign] -> (dim,) normalized vector.
    """
    unique_signs = sorted(set(signs))
    centroids: Dict[str, np.ndarray] = {}

    for sign in unique_signs:
        idx = np.where(signs == sign)[0]
        if len(idx) == 0:
            continue
        sign_embeds = doc_embeddings[idx]
        centroid = sign_embeds.mean(axis=0, keepdims=True)
        centroid = l2_normalize(centroid)[0]  # (dim,)
        centroids[sign] = centroid

    return centroids


def build_tfidf_vectorizer(texts: List[str]) -> Tuple[TfidfVectorizer, np.ndarray, np.ndarray]:
    """
    Build TF-IDF vectorizer on cleaned descriptions.
    Returns:
      - vectorizer
      - tfidf_matrix (num_docs x num_features)
      - feature_names (array of strings)
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
    )
    tfidf = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    return vectorizer, tfidf, feature_names


def _is_reasonable_trait(term: str) -> bool:
    """
    Heuristic filter for trait-like words/phrases:
    - 1–2 word phrase
    - alphabetic (ignoring spaces)
    - length > 3 chars
    - not in BORING_WORDS
    """
    term = term.strip()
    plain = term.replace(" ", "")
    if not plain.isalpha():
        return False
    if len(plain) < 4:
        return False
    if term in BORING_WORDS:
        return False

    # Limit to at most two tokens (avoid whole sentences)
    if len(term.split()) > 2:
        return False

    return True


def extract_sign_traits(
    signs: np.ndarray,
    tfidf_matrix,
    feature_names: np.ndarray,
    sign_centroids: Dict[str, np.ndarray],
    model: SentenceTransformer,
    n_traits: int | None = None,
) -> Dict[str, List[str]]:
    """
    For each sign:
      - compute mean TF-IDF vector over its docs
      - get top TF-IDF terms as candidate phrases
      - filter to trait-like candidates
      - embed candidates and rank by cosine similarity to sign centroid
      - keep top n_traits terms

    Returns: dict[sign] -> [trait1, trait2, trait3]
    """
    if n_traits is None:
        n_traits = N_TRAITS_PER_SIGN

    unique_signs = sorted(set(signs))
    traits: Dict[str, List[str]] = {}

    for sign in unique_signs:
        idx = np.where(signs == sign)[0]
        if len(idx) == 0:
            continue

        # Aggregate TF-IDF for this sign
        mean_vec_sparse = tfidf_matrix[idx].mean(axis=0)  # 1 x F
        mean_vec = np.asarray(mean_vec_sparse).ravel()
        scores = list(zip(feature_names, mean_vec))
        scores.sort(key=lambda x: x[1], reverse=True)

        # collect candidate terms by TF-IDF importance
        candidates: List[str] = []
        for term, score in scores:
            if score <= 0:
                break
            if _is_reasonable_trait(term):
                candidates.append(term)
            if len(candidates) >= TRAIT_CANDIDATE_MAX_TERMS:
                break

        # If filtering was too aggressive, relax and just take top TF-IDF terms
        if not candidates:
            candidates = [t for t, _ in scores[:TRAIT_CANDIDATE_MAX_TERMS]]

        # Deduplicate while preserving order
        seen = set()
        candidates = [c for c in candidates if not (c in seen or seen.add(c))]

        # Step 2: rank candidates by similarity to sign centroid in embedding space
        cand_embeddings = model.encode(
            candidates,
            show_progress_bar=False,
            batch_size=32,
        )
        cand_embeddings = l2_normalize(np.asarray(cand_embeddings))

        centroid = sign_centroids[sign]  # (dim,)
        sims = cand_embeddings @ centroid  # cosine similarity for normalized vectors

        # Pick top n_traits by similarity
        best_idx = np.argsort(-sims)[:n_traits]
        chosen = [candidates[i] for i in best_idx]
        traits[sign] = chosen

    return traits


def build_sign_model(df) -> Tuple[
    SentenceTransformer,
    Dict[str, np.ndarray],
    Dict[str, List[str]],
]:
    """
    High-level helper used by main.py:

    Given the DataFrame with columns ['sign', 'description', 'clean_description']:
      - loads the embedding model
      - computes per-document embeddings
      - computes sign centroids
      - builds TF-IDF
      - extracts traits per sign using embedding-aware ranking

    Returns:
      embedding_model,
      sign_centroids (dict[sign] -> (dim,) vector),
      sign_traits (dict[sign] -> list[str])
    """
    # Load embedding model
    model = load_embedding_model()

    # Document-level embeddings (using raw descriptions for richness)
    descriptions = df["description"].astype(str).tolist()
    doc_embeddings = compute_document_embeddings(descriptions, model)

    # Sign-level centroids
    signs = df["sign"].values
    sign_centroids = compute_sign_centroids(signs, doc_embeddings)

    # TF-IDF for candidate terms (using cleaned text)
    _, tfidf_matrix, feature_names = build_tfidf_vectorizer(
        df["clean_description"].tolist()
    )

    # Traits per sign
    sign_traits = extract_sign_traits(
        signs=signs,
        tfidf_matrix=tfidf_matrix,
        feature_names=feature_names,
        sign_centroids=sign_centroids,
        model=model,
        n_traits=N_TRAITS_PER_SIGN,
    )

    return model, sign_centroids, sign_traits
