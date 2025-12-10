"""
Entry point:
- load data
- build embeddings, sign centroids, and traits
- run interactive classification REPL
"""

from pprint import pprint

import numpy as np

from config import HOROSCOPE_CSV, N_TOP_EXAMPLES
from load import load_horoscopes
from traits import (
    load_embedding_model,
    compute_embeddings,
    compute_sign_centroids,
    extract_sign_traits,
)
from classify import (
    classify_with_embeddings,
    top_examples_for_sign,
)


def main():
    # Load the dataset
    df = load_horoscopes()
    print(f"Loaded {len(df)} rows from {HOROSCOPE_CSV}")

    descriptions = df["description"].astype(str).tolist()
    signs = df["sign"].astype(str).values
    clean_texts = df["clean_description"].astype(str).tolist()

    # 2. Load embedding model and compute embeddings
    model = load_embedding_model()
    embeddings = compute_embeddings(model, clean_texts, normalize=True)

    # 3. Compute sign centroids + index mapping
    sign_centroids, sign_to_indices = compute_sign_centroids(signs, embeddings)

    # 4. Extract descriptive traits per sign
    sign_traits = extract_sign_traits(
        descriptions=descriptions,
        signs=signs,
        embeddings=embeddings,
        sign_to_indices=sign_to_indices,
        sign_centroids=sign_centroids,
    )

    print("\n Extracted Traits Per Sign (embedding-aware) ")
    pprint(sign_traits)

    # Explain what similarities mean
    print(
        "\nNote: When you enter a description, the model embeds your text and\n"
        "compares it to the 'centroid' embedding for each sign. The decimal\n"
        "values you see are cosine similarity scores in embedding space:\n"
        "  • +1.0  = perfectly aligned (identical direction)\n"
        "  •  0.0  = unrelated / orthogonal\n"
        "  • -1.0  = opposite direction\n"
        "In practice, you'll see modest positives (e.g., 0.18–0.30). Higher\n"
        "values mean your description is more semantically similar to that sign.\n"
    )

    # Simple REPL to classify new descriptions
    print("You can now type a description to classify it into a zodiac sign.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        text = input("Enter a description: ").strip()
        if text.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        best_sign, sims, query_emb = classify_with_embeddings(
            text=text,
            model=model,
            sign_centroids=sign_centroids,
        )

        if best_sign is None:
            print("\nThe model couldn't find a strong match for any sign.\n")
            continue

        # Sort similarities descending for display
        sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)

        print(f"\nPredicted sign: {best_sign}")
        print("\nCosine similarity scores (higher = more similar):")
        for sign, score in sorted_sims:
            print(f"  {sign:11s}  {score:.4f}")

        # Explain WHY that sign was chosen
        print("\nMain factors contributing to this prediction:")
        print(
            f"- The {best_sign!r} centroid had the highest cosine similarity "
            f"to your description embedding: {sims[best_sign]:.4f}."
        )

        traits = sign_traits.get(best_sign)
        if traits:
            print(
                f"- The model has learned these key descriptive traits for {best_sign}: "
                + ", ".join(traits)
            )

        # Show a few closest training examples for that sign
        examples = top_examples_for_sign(
            sign=best_sign,
            query_emb=query_emb,
            descriptions=descriptions,
            embeddings=embeddings,
            signs=signs,
            k=N_TOP_EXAMPLES,
        )

        if examples:
            print(f"- Training horoscopes most similar to your text for {best_sign}:")
            for i, (ex_text, ex_sim) in enumerate(examples, start=1):
                snippet = ex_text.strip().replace("\n", " ")
                if len(snippet) > 200:
                    snippet = snippet[:200] + "..."
                print(f"    {i}. (sim {ex_sim:.4f}) {snippet}")

        print()  # blank line before next prompt


if __name__ == "__main__":
    main()
