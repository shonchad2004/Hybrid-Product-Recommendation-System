# item2vec_model.py (Conceptual Code)

import pandas as pd
from gensim.models import Word2Vec

# Note: This is a conceptual file. To run it, you would need to integrate
# the 'df_filtered' from your collaborative filtering script.

def train_item2vec_model(df_filtered):
    """
    Trains an Item2Vec model using Word2Vec.
    """
    print("Starting Item2Vec model training...")
    
    # --- 1. Prepare the data ---
    # The model needs "sentences". In our case, a sentence is the sequence of
    # products a single user has reviewed. We group our data by 'UserId' and
    # collect all the 'ProductId's they reviewed into a list.
    
    # Ensure we drop any potential duplicates for a user's session
    user_purchases = df_filtered.groupby('UserId')['ProductId'].apply(lambda products: list(products.unique()))
    
    # The Word2Vec model expects a list of lists (our sentences)
    sentences = list(user_purchases)
    
    if not sentences:
        print("Not enough data to create sentences for Item2Vec.")
        return None

    print(f"Created {len(sentences)} user 'sentences' for training.")
    
    # --- 2. Train the Word2Vec model ---
    # - sentences: Our list of user purchase histories.
    # - vector_size: The dimensionality of the product vectors (e.g., 100 dimensions).
    # - window: The max distance between the current and predicted product within a sentence.
    # - min_count: Ignores all products with a total frequency lower than this.
    # - workers: Number of CPU threads to use.
    print("Training Word2Vec model...")
    model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)
    print("Item2Vec model trained successfully.")
    
    return model

def get_item2vec_recommendations(model, product_id, num_recommendations=10):
    """
    Gets recommendations from a trained Item2Vec model.
    """
    if model is None or product_id not in model.wv:
        print(f"Product {product_id} not found in Item2Vec model.")
        return []
    
    # The most_similar function finds the top N closest items in the vector space.
    similar_products = model.wv.most_similar(product_id, topn=num_recommendations)
    
    # The result is a list of tuples (productId, similarity_score)
    recommendations = [{'productId': item[0], 'explanation': f"Recommended because it is conceptually similar to '{product_id}' (Item2Vec score: {item[1]:.2f})."} for item in similar_products]
    
    return recommendations

# --- Example of how you would use this ---
if __name__ == '__main__':
    # In a real application, you would load 'df_filtered' from the other script
    # or a saved file. For this example, we'll create a dummy DataFrame.
    
    print("\n--- Testing Item2Vec Concept ---")
    
    # Create dummy data that mimics the structure of df_filtered
    dummy_data = {
        'UserId': ['U1', 'U1', 'U1', 'U2', 'U2', 'U3', 'U3', 'U3', 'U3'],
        'ProductId': ['P1', 'P2', 'P3', 'P1', 'P4', 'P2', 'P3', 'P4', 'P5'],
        'Score': [5, 4, 5, 3, 5, 5, 4, 2, 5],
        'Text': ['great', 'good', 'love it', 'ok', 'best', 'nice', 'awesome', 'bad', 'amazing']
    }
    dummy_df = pd.DataFrame(dummy_data)
    
    # Train the model on our dummy data
    item2vec_model = train_item2vec_model(dummy_df)
    
    if item2vec_model:
        # Get recommendations for 'P2'
        # We expect 'P3' to be a strong recommendation because U1 and U3 bought them together.
        test_product = 'P2'
        recommendations = get_item2vec_recommendations(item2vec_model, test_product)
        
        print(f"\nItem2Vec recommendations for '{test_product}':")
        for rec in recommendations:
            print(f"  - Product: {rec['productId']}\n    Explanation: {rec['explanation']}")