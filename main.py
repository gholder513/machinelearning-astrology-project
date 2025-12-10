"""
Entry point: load data, build embedding-based sign model,
and run a simple interactive loop to classify new descriptions.
"""

from pprint import pprint

from config import HOROSCOPE_CSV
from load import load_horoscopes
from traits import build_sign_model
from classify import classify_with_embeddings


def main():
    # Load the dataset
    df = load_horoscopes()
    print(f"Loaded {len(df)} rows from {HOROSCOPE_CSV}")

    #  Build sign model: embedding model, sign centroids, traits
    model, sign_centroids, sign_traits = build_sign_model(df)

    print("\n Extracted Traits Per Sign (embedding-aware) ")
    pprint(sign_traits)

    # Simple REPL to classify new descriptions
    print("\nYou can now type a description to classify it into a zodiac sign.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            text = input("Enter a description: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if text.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        best_sign, sims = classify_with_embeddings(text, model, sign_centroids)

        if best_sign is None or not sims:
            print("\nCould not classify (no signal). Try a longer or more specific description.\n")
            continue

        print(f"\nPredicted sign: {best_sign}")
        print("Similarities (descending):")
        for sign, score in sorted(sims.items(), key=lambda x: x[1], reverse=True):
            print(f"  {sign:11s}  {score:.4f}")
        print()


if __name__ == "__main__":
    main()
