"""
Simple CLI to choose which zodiac model to run inside the container.
- Option 1: Embedding-based model (main.py)
- Option 2: Random Forest model (rf_main.py)
"""

from __future__ import annotations

import sys

from main import main as embedding_main
from rf_main import main as rf_main_main


def choose_and_run():
    while True:
        print("\nWhich zodiac model would you like to use?")
        print("  1) Embedding-based classifier (sentence-transformers)")
        print("  2) Random Forest classifier (scikit-learn)")
        print("  q) Quit")

        choice = input("Enter 1, 2, or q: ").strip().lower()

        if choice == "1":
            print("\nLaunching embedding-based classifier...\n")
            embedding_main()
            break
        elif choice == "2":
            print("\nLaunching Random Forest classifier...\n")
            rf_main_main()
            break
        elif choice in {"q", "quit", "exit"}:
            print("Exiting. Goodbye!")
            sys.exit(0)
        else:
            print("Invalid choice. Please try again.\n")


if __name__ == "__main__":
    choose_and_run()
