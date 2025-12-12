# backend Dockerfile at project root
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app


# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.3.1"

# Install the rest of the dependencies (no torch here)
RUN pip install --no-cache-dir -r requirements.txt

#  Download spaCy model (spacy is already installed from requirements.txt)
RUN python -m spacy download en_core_web_sm

#  Copy the app code
COPY . .

# pre-download the sentence-transformers model to avoid doing it at runtime
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
