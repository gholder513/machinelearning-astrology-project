"""
Entry point: load data, compute embeddings, extract traits, and run
an interactive loop to classify new descriptions.
"""

from __future__ import annotations

from pprint import pprint

from load import load_horoscopes
from traits import prepare_embeddings_and_traits
from classify import classify_with_centroids


def main():
    # Load the dataset
    df = load_horoscopes()
    print(f"Loaded {len(df)} rows from {df.attrs.get('filepath_or_buffer', 'horoscope.csv') if hasattr(df, 'attrs') else 'horoscope.csv'}")

    # Build embeddings, centroids, and traits
    model, sign_centroids, sign_traits = prepare_embeddings_and_traits(df)

    print("\n Extracted Traits Per Sign (embedding-aware) ")
    pprint(sign_traits)

    # Simple REPL to classify new descriptions
    print("\nYou can now type a description to classify it into a zodiac sign.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        text = input("Enter a description: ").strip()
        if text.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        best_sign, sims = classify_with_centroids(text, model, sign_centroids)

        if best_sign is None or not sims:
            print("\nSorry, I couldn't get a meaningful signal from that input.\n")
            continue

        print(f"\nPredicted sign: {best_sign}")

        # Explain why: show that sign's traits
        traits = sign_traits.get(best_sign, [])
        if traits:
            print("\nMain factors / traits for this sign (from the training data):")
            for t in traits:
                print(f"  - {t}")
        else:
            print("\n(No extracted traits available for this sign.)")

        # Explain the decimal numbers
        print(
            "\nSimilarity scores (cosine similarity):\n"
            "These numbers measure how close your description's meaning is to each sign's\n"
            "overall 'vibe' in the training data. Values are roughly between -1 and +1.\n"
            "Higher = more semantically similar; the predicted sign has the highest score.\n"
        )

        # Sort and display similarities
        for sign, score in sorted(sims.items(), key=lambda x: x[1], reverse=True):
            print(f"  {sign:11s}  {score:.4f}")

        print()  # blank line before next prompt


if __name__ == "__main__":
    main()
