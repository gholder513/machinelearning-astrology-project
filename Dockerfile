# Dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install spacy && \
    python -m spacy download en_core_web_sm

COPY . .

# pre-download sentence-transformers model so it doesn't download on every run
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Default command: run the model selector
CMD ["python", "model_selector.py"]
