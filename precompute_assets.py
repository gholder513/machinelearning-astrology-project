from __future__ import annotations

import joblib
import numpy as np

from config import HOROSCOPE_CSV, EMBEDDING_MODEL_NAME
from load import load_horoscopes
from traits import prepare_embeddings_and_traits
from rf_model import train_random_forest_from_csv


def main() -> None:
    print(f"Loading horoscopes from {HOROSCOPE_CSV}...")
    df = load_horoscopes(HOROSCOPE_CSV)

    
    descriptions_clean = df["clean_description"].astype(str).tolist()
    signs_array = df["sign"].values

    print("Preparing embedding model, centroids, and traits...")
    
    model, sign_centroids, sign_traits = prepare_embeddings_and_traits(df)

    print(f"Computing embeddings for {len(descriptions_clean)} clean descriptions...")
    description_embeddings = model.encode(
        descriptions_clean,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    # normalize so dot products ≈ cosine similarity
    norms = np.linalg.norm(description_embeddings, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    description_embeddings = description_embeddings / norms

    embed_assets = {
        "descriptions": descriptions_clean,          
        "signs_array": signs_array,                  
        "description_embeddings": description_embeddings,  
        "sign_centroids": sign_centroids,            
        "sign_traits": sign_traits,                  
        "embedding_model_name": EMBEDDING_MODEL_NAME,
    }
    joblib.dump(embed_assets, "embed_assets.pkl")
    print("Saved embed_assets.pkl")

    print("Training Random Forest model for text classification...")
    rf_vec, rf_le, rf_clf, rf_report, rf_acc = train_random_forest_from_csv(HOROSCOPE_CSV)
    rf_assets = {
        "vectorizer": rf_vec,
        "label_encoder": rf_le,
        "clf": rf_clf,
        "report": rf_report,
        "accuracy": rf_acc,
    }
    joblib.dump(rf_assets, "rf_assets.pkl")
    print("Saved rf_assets.pkl")

    print("Precomputation complete.")


if __name__ == "__main__":
    main()