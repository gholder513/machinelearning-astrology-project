"""
Entry point: load data, compute embeddings, extract traits, and run
an interactive loop to classify new descriptions.
"""

from __future__ import annotations

from pprint import pprint

from load import load_horoscopes
from traits import prepare_embeddings_and_traits
from classify import classify_with_centroids, top_examples_for_sign
from config import N_TOP_EXAMPLES


def main():
    # Load the dataset
    df = load_horoscopes()
    csv_path = getattr(df, "attrs", {}).get("filepath_or_buffer", "horoscope.csv")
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Build embeddings, centroids, and traits
    # prepare_embeddings_and_traits is assumed to:
    #   - load the embedding model
    #   - (optionally) compute internal embeddings & centroids
    # and return at least: model, sign_centroids, sign_traits
    model, sign_centroids, sign_traits = prepare_embeddings_and_traits(df)

    print("\n Extracted Traits Per Sign (embedding-aware) ")
    pprint(sign_traits)

    # For showing top similar training examples, we also need:
    #   - the cleaned descriptions
    #   - the sign labels
    #   - embeddings for each description (using the same model)
    descriptions = df["clean_description"].tolist()
    signs = df["sign"].values

    # Compute embeddings for all descriptions once here
    # (even if prepare_embeddings_and_traits computed its own internally,
    #  it's fine to recompute; these will still live in the same space.)
    print(f"\nComputing embeddings for {len(descriptions)} descriptions for example lookup...")
    embeddings = model.encode(
        descriptions,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    # Simple REPL to classify new descriptions
    print("\nYou can now type a description to classify it into a zodiac sign.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        text = input("Enter a description: ").strip()
        if text.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        # Use the sign_centroids returned by prepare_embeddings_and_traits
        best_sign, sims, query_emb = classify_with_centroids(text, model, sign_centroids)

        if best_sign is None:
            print("No meaningful similarity found.\n")
            continue

        print(f"\nPredicted sign: {best_sign}")
        print("Similarities (descending):")
        for sign, score in sorted(sims.items(), key=lambda x: x[1], reverse=True):
            print(f"  {sign:11s}  {score:.4f}")
        print("  (These decimals are cosine similarities: closer to 1.0 = more semantically similar.)")

        # Show top similar training examples for this sign
        examples = top_examples_for_sign(
            sign=best_sign,
            query_emb=query_emb,
            descriptions=descriptions,   # list of cleaned descriptions from your df
            embeddings=embeddings,       # np.array of embeddings for those descriptions
            signs=signs,                 # np.array of sign labels
            k=N_TOP_EXAMPLES,            # e.g., 3
        )

        if examples:
            print(f"\nMost similar training horoscopes for '{best_sign}':")
            for i, (ex_text, ex_sim) in enumerate(examples, start=1):
                snippet = ex_text.strip().replace("\n", " ")
                if len(snippet) > 200:
                    snippet = snippet[:200] + "..."
                print(f"  {i}. (Similarity score {ex_sim:.4f}) {snippet}")

        print()


if __name__ == "__main__":
    main()
