# popularity_model.py

import pandas as pd

# --- Step 1: Load the correct dataset ---
# We are now using 'Reviews.csv' to be consistent with our other models.
print("Popularity Model: Loading data from Reviews.csv...")
df = pd.read_csv('Reviews.csv')

# The column names in this file are 'UserId', 'ProductId', 'Score', etc.
# We'll work with a sample for speed.
df_sample = df.sample(n=100000, random_state=42)
print("Popularity Model: Data loaded and sampled.")

# --- Step 2: Calculate popularity ---
# We count the number of reviews for each 'ProductId'.
counts = df_sample['ProductId'].value_counts()
popularity_df = pd.DataFrame({'ProductId': counts.index, 'rating_count': counts.values})
print("Popularity Model: Popularity calculated.")

# --- Step 3: Rank the products ---
most_popular = popularity_df.sort_values('rating_count', ascending=False)

# --- Step 4: Create the recommender function ---
def recommend_popular_products(num_recommendations=10):
    """Returns the top N most popular products."""
    # We rename the column to 'productId' for consistency in the final output.
    # This makes it easier to combine results in app.py.
    return most_popular.head(num_recommendations).rename(columns={'ProductId': 'productId'})

# --- Example Usage ---
if __name__ == '__main__':
    print("\n--- Testing our Popularity Recommender (on new data) ---")
    recommendations = recommend_popular_products(5)
    print(recommendations)