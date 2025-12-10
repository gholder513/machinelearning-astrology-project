# rf_main.py
"""
Entry point for the Random Forest zodiac classifier.

This does NOT modify or replace your existing embedding-based main.py.
It's a parallel script you can run with:

    python rf_main.py
"""

from __future__ import annotations

from pprint import pprint

from rf_model import train_random_forest_from_csv, predict_sign_rf
from config import HOROSCOPE_CSV


def explain_rf_probabilities():
    print(
        "\nRandom Forest probability explanation:\n"
        "  The decimals shown for each sign are predicted probabilities\n"
        "  from the Random Forest model (they sum to 1.0 across signs).\n"
        "  Higher values mean the model is more confident in that sign.\n"
    )


def main():
    print(f"Training Random Forest on: {HOROSCOPE_CSV}")
    vectorizer, label_encoder, clf = train_random_forest_from_csv(HOROSCOPE_CSV)

    explain_rf_probabilities()

    print("\nYou can now type a description to classify it with the Random Forest.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        text = input("Enter a description: ").strip()
        if text.lower() in {"quit", "exit"}:
            print("Goodbye from Random Forest classifier!")
            break

        best_sign, proba_dict = predict_sign_rf(text, vectorizer, label_encoder, clf)

        print(f"\nRandom Forest predicted sign: {best_sign}")
        print("Probabilities (descending):")

        for sign, p in sorted(proba_dict.items(), key=lambda x: x[1], reverse=True):
            print(f"  {sign:11s} {p:.4f}")

        print()


if __name__ == "__main__":
    main()
