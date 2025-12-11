Astrology NLP Machine Learning Project

A hybrid NLP system that combines traditional machine learning, sentence-transformer embeddings, and LLM-assisted text generation to classify zodiac signs, extract descriptive traits, and generate personalized horoscopes.
The project includes a fully containerized FastAPI backend, a React frontend, and a unified Docker Compose workflow.

Overview

This project explores whether linguistic patterns in horoscope text can be leveraged to classify zodiac signs and generate user-tailored personality horoscopes. The system includes:

Embedding-based classifier using Sentence Transformers

Random Forest classifier trained on TF-IDF features

AI-aided horoscope generator using OpenAI models

Trait extraction pipeline combining embeddings and TF-IDF

Full-stack web interface wired through Docker Compose

The dataset consists of 768 horoscope entries scraped from a Kaggle dataset.

Features
1. Embedding-Based Zodiac Classifier

Uses sentence-transformers/all-MiniLM-L6-v2 to compute vector embeddings of horoscope descriptions.
Centroids are computed per zodiac sign, and cosine similarity determines predicted labels.
The classifier returns:

Predicted zodiac sign

Similarity scores for all signs

Most similar training examples

Extracted traits representative of the predicted sign

2. Random Forest Classifier

A traditional ML baseline using:

TF-IDF vectorization

Label encoding

Random Forest classification

Full sklearn classification report exposed via /api/rf/metrics

This model serves as a benchmark for accuracy, precision, recall, and f1-score.

3. AI-Aided Horoscope Generator

A ten-round evaluation system:

User inputs sign + brief description

Backend retrieves top embedding matches to anchor generation

OpenAI model generates unique horoscopes with variable tones, focuses, and phrasing

User rates each horoscope for subjective accuracy

Overall accuracy score computed across rounds

This component includes custom prompts, penalties, and stylistic rotation to reduce repetition.

4. Trait Extraction

Initial TF-IDF–based traits produced mostly filler words due to horoscope linguistic ambiguity.
The final system uses:

Sentence Transformer embeddings

Phrase-level mining

Filtering, normalization, and fallback behavior

Centroid comparison per sign

Results are considerably more interpretable (examples below):

aquarius: ['romance', "today's surprising events", 'tomorrow night']
cancer:   ['evening', 'surprises', 'fun']
leo:      ['Unsettling', 'affairs', 'dinner']

5. Full Dockerized System

A single command launches backend + frontend:

docker compose up --build


Backend: FastAPI (Python)
Frontend: React + Vite + Nginx
All models load automatically on container startup.

Why Sentence Transformers?

TF-IDF was insufficient for semantic similarity in a domain full of vague language. Example:

“Greek yogurt”

“European dairy”

These share zero overlapping words but are clearly related.
Sentence Transformers embed meaning rather than word frequency, which resolved:

Zero-similarity outputs

Trait extraction noise

Misclassification caused by adverbs and filler words

Rare-word imbalance

Embeddings enabled multi-phrase trait extraction and improved model stability.

File Structure
machinelearning-astrology-project/
│
├── api_server.py
├── classify.py
├── config.py
├── load.py
├── traits.py
├── rf_model.py
├── horoscope.csv
│
├── frontend/
│   ├── my-react-app/
│   └── styles/
│
├── Dockerfile
├── docker-compose.yml
└── README.md

Installation and Usage
Local Development (without Docker)

Install Python dependencies:

pip install -r requirements.txt


Run embedding classifier:

python main.py


Run Random Forest classifier:

python rf_main.py

Docker Build and Run

Build the standalone image:

docker build -t zodiac-classifier .


Run interactively:

docker run -it --rm zodiac-classifier


Force a specific entrypoint:

docker run -it --rm zodiac-classifier python main.py
docker run -it --rm zodiac-classifier python rf_main.py

Full Stack Launch
docker compose up --build


Backend runs at:

http://localhost:8000


Frontend runs at:

http://localhost:5173

Development Notes (Chronological Summary)
Step 1: Preprocessing

Implemented TF-IDF mapping incorrectly emphasized filler words.

Added heuristics, filtering, and apostrophe handling.

Early similarity scores were universally zero due to linguistic ambiguity and adverb saturation.

Moved to embedding-based preprocessing with Sentence Transformers.

Expanded trait extraction from three words to multi-word phrases.

Step 2: Random Forest Model

Integrated sklearn Random Forest with TF-IDF features.

Trained on horoscope descriptions only.

Exposed accuracy and classification report via API.

Implemented Docker workflows.

Step 3: AI Horoscope Evaluation

Added iterative horoscope generation using OpenAI.

Introduced temperature, top_p, presence penalty, and frequency penalty to vary style.

Added tone and focus rotation.

Implemented 10-round scoring mechanism.

Step 4: Full Stack Integration

Built React + Vite frontend.

Connected all API endpoints.

Resolved CORS issues and Nginx routing.

Added Random Forest metrics table + trait panel UI.

Containerized entire system into one unified developer experience.

Example Embedding Classifier Output

Input:
"Likes studying and playing sports"

Output:

Predicted sign: cancer
Similarities (descending):
cancer       0.2512
aries        0.2234
sagittarius  0.2188
...
gemini       0.1682


Top Training Examples:

(sim 0.2954) it's time to listen to your body and slow down...

(sim 0.2778) you'll be working hard to try to find your purpose...

(sim 0.2678) that special person you've been playing games with...

Example Random Forest Output

Accuracy: 0.0996 on held-out test set
Classification report exposed through /api/rf/metrics.

Per-sign precision, recall, f1-score, and support are displayed in the frontend.

Known Challenges

Horoscope language is inherently ambiguous and stylistically noisy.

Trait extraction from unstructured text required multiple iterations and filtering passes.

Early TF-IDF models were unstable due to adverb dominance.

LLM horoscope generator required strong stylistic constraints to avoid repetition.

Docker routing between Nginx and FastAPI required careful API path normalization.

Future Improvements

Fine-tuned transformer model for sign classification

Characterization embeddings built on sign-specific contrastive learning

Adjustable sentiment or tone controls for horoscope generator

Per-user adaptive evaluation instead of fixed 10-round system

Scalable backend with async batching for embeddings