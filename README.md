# Astrology NLP Machine Learning Project


Embedding Model from HuggingFace



# Build the image
docker build -t zodiac-classifier .

# Run interactively (so you can see prompts, either the random forest or embedded)
docker run -it --rm zodiac-classifier


If you ever want to skip the menu and run a specific one directly, you can override the CMD at runtime:

# Force embedding model
docker run -it --rm zodiac-classifier python main.py

# Force random forest model
docker run -it --rm zodiac-classifier python rf_main.py
