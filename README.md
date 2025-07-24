# Intelligent Shopper's Assistant: A Hybrid Product Recommendation System

## Overview

This project is a comprehensive implementation of an advanced product recommendation system for e-commerce. It demonstrates a multi-layered approach, starting from simple baseline models and progressing to a sophisticated, explainable hybrid engine. The system leverages user ratings, product metadata, and the sentiment of review text to provide relevant, diverse, and transparent recommendations.

This project is designed to showcase proficiency in core machine learning concepts, natural language processing (NLP), and MLOps principles like creating reproducible environments and serving models via an API.

---

## 🚀 Features

* **Popularity-Based Model:** A baseline model that recommends the most reviewed items. This is effective for new users (solving the "cold start" problem).
* **Sentiment-Aware Collaborative Filtering:** A powerful "Users who liked this also liked..." model. It goes beyond simple ratings by analyzing the sentiment of review text using **NLTK's VADER**, creating a more nuanced "adjusted rating" that better reflects user satisfaction.
* **Content-Based Filtering:** Recommends items based on their attributes ("Because you liked a product with these features..."). It uses **TF-IDF** to vectorize product descriptions and **Cosine Similarity** to find similar items.
* **Advanced Hybrid Engine:** Intelligently combines the recommendations from all models. It prioritizes personalized results from collaborative and content-based filtering and uses the popularity model as a robust fallback, ensuring a diverse and relevant final list.
* **Explainable AI (XAI):** A key feature that builds user trust. Every recommendation is delivered with a human-readable explanation, detailing *why* the item is being suggested (e.g., "Recommended because it shares features like {'coffee', 'organic'}..." or "Recommended because users who liked product X also liked this item.").

---

## 🛠️ Tech Stack

* **Language:** Python
* **Core Libraries:**
    * **Data Manipulation:** Pandas, NumPy
    * **Machine Learning:** Scikit-learn (for TF-IDF, Cosine Similarity)
    * **Scientific Computing:** SciPy (for Sparse Matrices)
    * **Natural Language Processing (NLP):** NLTK (for VADER Sentiment Analysis)
* **Deployment:** Flask (for creating a simple REST API endpoint)
* **Development Environment:** VS Code, Virtual Environments (venv)

---

## 📂 Project Structure

/product-recommendation-system

|-- recsys_env/               # Virtual environment

|-- app.py                    # Main application, hybrid logic, and Flask API

|-- collaborative_filtering_model.py

|-- content_based_model.py

|-- popularity_model.py

|-- item2vec_model.py         # Conceptual code for a deep learning approach

|-- Reviews.csv               # The dataset

|-- README.md                 # This documentation

|-- requirements.txt          # Project dependencies


---

## 📈 Future Work

* **Implement Item2Vec:** The `item2vec_model.py` contains the conceptual code for a deep learning approach using `gensim`. This would involve training a Word2Vec model on user purchase sequences to learn latent product embeddings, potentially capturing more complex relationships.
* **Full-Fledged UI:** Develop a front-end interface (e.g., using React or Streamlit) to interact with the recommendation API in a user-friendly way.
* **Scalability:** For larger datasets, migrate the data processing and model training to a distributed computing framework like Apache Spark.
