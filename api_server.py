from __future__ import annotations

from typing import Dict, List, Optional, Any
from pathlib import Path
import os

import joblib
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

from classify import classify_with_centroids, top_examples_for_sign
from config import (
    HOROSCOPE_CSV,   
    N_TOP_EXAMPLES,
    GPT_MODEL_NAME,
    # NUM_HOROSCOPE_ROUNDS,  
    EMBEDDING_MODEL_NAME,
)


# from load import load_horoscopes
# from rf_model import train_random_forest_from_csv
# from traits import prepare_embeddings_and_traits
from rf_model import predict_sign_rf


app = FastAPI(title="Zodiac Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # or restrict to Netlify domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths to precomputed artifacts
BASE_DIR = Path(__file__).resolve().parent
EMBED_ASSETS_PATH = BASE_DIR / "embed_assets.pkl"
RF_ASSETS_PATH = BASE_DIR / "rf_assets.pkl"

# Global objects (lazy-loaded)
embedding_model = None

sign_centroids: Dict[str, np.ndarray] | None = None
sign_traits: Dict[str, List[str]] | None = None
descriptions: List[str] | None = None
signs_array: np.ndarray | None = None
description_embeddings: np.ndarray | None = None

rf_vectorizer = None
rf_label_encoder = None
rf_clf = None
rf_report_dict: Dict[str, Any] | None = None
rf_accuracy: float | None = None

openai_client: Optional[OpenAI] = None


# Pydantic models
class EmbedRequest(BaseModel):
    text: str


class TopExample(BaseModel):
    text: str
    similarity: float


class EmbedResponse(BaseModel):
    predicted_sign: Optional[str]
    similarities: Dict[str, float]
    top_examples: List[TopExample]


class RFRequest(BaseModel):
    text: str


class RFResponse(BaseModel):
    predicted_sign: str
    probabilities: Dict[str, float]


class HoroscopeRequest(BaseModel):
    sign: str
    description: str
    round_index: int


class HoroscopeResponse(BaseModel):
    horoscope: str


# lazy loading
def _init_openai_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Create a .env file with OPENAI_API_KEY=your_key_here "
            "or export it in your shell."
        )
    return OpenAI(api_key=api_key)


def _ensure_assets_loaded() -> None:
    """
    Lazy-load all ML artifacts and the embedding model into globals.
    Called at the top of each endpoint that needs them.
    """
    global embedding_model, sign_centroids, sign_traits
    global descriptions, signs_array, description_embeddings
    global rf_vectorizer, rf_label_encoder, rf_clf, rf_report_dict, rf_accuracy
    global openai_client

    # if already loaded, do nothing
    if (
        embedding_model is not None
        and sign_centroids is not None
        and sign_traits is not None
        and descriptions is not None
        and signs_array is not None
        and description_embeddings is not None
        and rf_clf is not None
        and rf_report_dict is not None
        and rf_accuracy is not None
        and openai_client is not None
    ):
        return

    print("Lazy-loading precomputed ML assets...")

    # Embedding-related assets
    embed_assets = joblib.load(EMBED_ASSETS_PATH)
    descriptions = embed_assets["descriptions"]
    signs_array = embed_assets["signs_array"]
    description_embeddings = embed_assets["description_embeddings"]
    sign_centroids = embed_assets["sign_centroids"]
    sign_traits = embed_assets["sign_traits"]

    # RF-related assets
    rf_assets = joblib.load(RF_ASSETS_PATH)
    rf_vectorizer = rf_assets["vectorizer"]
    rf_label_encoder = rf_assets["label_encoder"]
    rf_clf = rf_assets["clf"]
    rf_report_dict = rf_assets["report"]
    rf_accuracy = rf_assets["accuracy"]

    # load embedding model
    from sentence_transformers import SentenceTransformer

    print(f"Loading SentenceTransformer model '{EMBEDDING_MODEL_NAME}'...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # OpenAI client
    openai_client = _init_openai_client()

    print("ML assets and clients loaded.")


# Horoscope generation helpers 
def _pick_style(round_idx: int) -> tuple[str, str]:
    tone_options = [
        "warm and encouraging",
        "direct and practical",
        "playful and lighthearted",
        "introspective and reflective",
        "mysterious and poetic",
    ]
    focus_options = [
        "relationships and emotional patterns",
        "creative expression and hobbies",
        "daily habits and routines",
        "career, ambition, and long-term goals",
        "inner growth, mindset, and self-talk",
        "body, health, and physical environment",
    ]

    tone = tone_options[round_idx % len(tone_options)]
    focus = focus_options[(round_idx * 2) % len(focus_options)]
    return tone, focus


def _generate_horoscope(
    client: OpenAI,
    sign: str,
    user_description: str,
    similar_examples: List[tuple[str, float]],
    round_idx: int,
    model_name: str,
) -> str:
    examples_text = ""
    if similar_examples:
        examples_list = []
        for ex_text, ex_sim in similar_examples:
            snippet = ex_text.replace("\n", " ").strip()
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            examples_list.append(f"- (sim {ex_sim:.3f}) {snippet}")
        examples_text = "\n".join(examples_list)

    tone_hint, focus_hint = _pick_style(round_idx)

    system_prompt = (
        "You are an astrology assistant that writes concise, personality-focused horoscopes.\n"
        "You will be given a zodiac sign, a short description of the person, and some example "
        "horoscope snippets for that sign.\n"
        "Your job is to generate a short (3–4 sentence) horoscope that feels accurate and "
        "reflective of the person's character, not generic fluff.\n"
        "VARY your style from call to call: don't always start with 'Today' or 'This', "
        "change sentence structure, and avoid reusing distinctive phrases.\n"
        "Do not copy any training text verbatim.\n"
    )

    user_prompt = f"""
ZODIAC SIGN: {sign.upper()}
USER DESCRIPTION: "{user_description}"

Desired tone for THIS horoscope: {tone_hint}.
Primary focus for THIS horoscope: {focus_hint}.

Training horoscope snippets for this sign (and their similarity scores):
{examples_text if examples_text else "(no examples provided)"}

Write a short, natural horoscope (3–4 sentences) that:
  - Feels tailored to this person.
  - Leans into the typical traits of {sign.title()}.
  - Uses the requested tone and focus for THIS round.
  - Does NOT reuse the exact same opening lines you've used in other horoscopes.
  - Does NOT repeat all of the user's hobbies every time; sometimes highlight just one
    or two and connect them to deeper themes (emotions, growth, choices, mindset).
  - Avoids repeating the training snippets verbatim.
  - Does NOT mention that it was generated by AI or that it's an example.
"""

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.95,
        top_p=0.9,
        presence_penalty=0.7,
        frequency_penalty=0.4,
        max_tokens=300,
    )

    return resp.choices[0].message.content.strip()


# Startup hook (now lightweight)
# You can omit this entirely, or leave a small log. Crucially, we do NOT
# load all models here anymore.

@app.on_event("startup")
async def on_startup():
    print("FastAPI app startup (lazy ML loading enabled).")


# Endpoints
@app.post("/api/embed/classify", response_model=EmbedResponse)
def embed_classify(req: EmbedRequest):
    """
    Embedding-based classifier:
      input: text
      output: predicted sign, similarities, and top similar training examples.
    """
    _ensure_assets_loaded()

    best_sign, sims, query_emb = classify_with_centroids(
        req.text, embedding_model, sign_centroids  
    )

    if best_sign is None:
        return EmbedResponse(predicted_sign=None, similarities={}, top_examples=[])

    examples_raw = top_examples_for_sign(
        sign=best_sign,
        query_emb=query_emb,
        descriptions=descriptions,               
        embeddings=description_embeddings,       
        signs=signs_array,                       
        k=N_TOP_EXAMPLES,
    )
    examples = [
        TopExample(text=ex_text, similarity=ex_sim) for ex_text, ex_sim in examples_raw
    ]

    return EmbedResponse(
        predicted_sign=best_sign,
        similarities=sims,
        top_examples=examples,
    )


@app.get("/api/traits")
def get_traits():
    """
    Traits endpoint used by TraitsPanel on the frontend.
    """
    _ensure_assets_loaded()

    return {
        "source": (
            "Embedding-aware traits mined from 768 Kaggle horoscope rows "
            "using Transformers + TF-IDF."
        ),
        "traits": sign_traits,
    }


@app.get("/api/rf/metrics")
def rf_metrics():
    """
    Expose Random Forest evaluation metrics for the frontend.
    Returns:
      - accuracy: overall accuracy (float)
      - classification_report: full sklearn classification_report dict
    """
    _ensure_assets_loaded()

    if rf_report_dict is None or rf_accuracy is None:
        return {"detail": "Random Forest metrics not available."}

    return {
        "accuracy": rf_accuracy,
        "classification_report": rf_report_dict,
    }


@app.post("/api/rf/classify", response_model=RFResponse)
def rf_classify(req: RFRequest):
    """
    Random Forest classifier:
      input: text
      output: predicted sign + probability distribution.
    """
    _ensure_assets_loaded()

    best_sign, proba_dict = predict_sign_rf(
        req.text,
        rf_vectorizer,      
        rf_label_encoder,   
        rf_clf,             
    )
    return RFResponse(predicted_sign=best_sign, probabilities=proba_dict)


@app.post("/api/gpt/generate", response_model=HoroscopeResponse)
def gpt_generate(req: HoroscopeRequest):
    """
    AI-aided horoscope generator:
      input: sign, description, round_index
      output: a single horoscope text.
      Accuracy tracking stays on the frontend.
    """
    _ensure_assets_loaded()

    # compute embedding for user description to find similar examples
    user_emb = embedding_model.encode(    
        req.description,
        convert_to_numpy=True,
    )

    examples_raw = top_examples_for_sign(
        sign=req.sign.lower(),
        query_emb=user_emb,
        descriptions=descriptions,               
        embeddings=description_embeddings,       
        signs=signs_array,                       
        k=N_TOP_EXAMPLES,
    )

    horoscope = _generate_horoscope(
        client=openai_client,  
        sign=req.sign,
        user_description=req.description,
        similar_examples=examples_raw,
        round_idx=req.round_index,
        model_name=GPT_MODEL_NAME,
    )

    return HoroscopeResponse(horoscope=horoscope)
