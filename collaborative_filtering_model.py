# collaborative_filtering_model.py

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# This is a one-time check. If you haven't downloaded the vader_lexicon, this will do it.
# It's good practice to have this check in your code.
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except (nltk.downloader.DownloadError, AttributeError):
    nltk.download('vader_lexicon')


# --- Step 1: Load and Prepare the Dataset ---
print("Loading dataset with review text...")
# We use the 'Reviews.csv' file which has product reviews.
# It has columns like 'ProductId', 'UserId', 'Score' (the rating), and 'Text' (the review).
df = pd.read_csv('Reviews.csv')

# The dataset is very large. We'll take a manageable sample to run on a standard laptop.
# We also MUST drop rows where 'Text' or 'ProductId' or 'UserId' is missing,
# as they are essential for our model.
df_sample = df.dropna(subset=['Text', 'ProductId', 'UserId']).sample(n=50000, random_state=42)
print(f"Dataset loaded. Working with a sample of {len(df_sample)} reviews.")


# --- Step 2: Perform Sentiment Analysis ---
# We initialize the VADER sentiment analyzer. VADER is great because it doesn't
# require training and is specifically tuned for social media/review text.
sia = SentimentIntensityAnalyzer()

# This function takes text and returns the 'compound' sentiment score.
# This score is a single float from -1.0 (most negative) to +1.0 (most positive).
def get_sentiment_score(text):
    return sia.polarity_scores(str(text))['compound']

# We apply this function to every review in our 'Text' column to create a new 'sentiment_score' column.
print("Performing sentiment analysis on review text...")
# Using .loc to avoid SettingWithCopyWarning
df_sample.loc[:, 'sentiment_score'] = df_sample['Text'].apply(get_sentiment_score)
print("Sentiment analysis complete.")


# --- Step 3: Create the "Adjusted Rating" ---
# This is the core of our advanced model. We combine the original star rating ('Score')
# with the sentiment score to create a more nuanced rating.
# **CRITICAL CORRECTION**: The formula should use the original 'Score'.
# Formula: adjusted_rating = original_rating * (1 + sentiment_score)
# A positive review (sentiment ~0.8) on a 5-star rating becomes: 5 * (1 + 0.8) = 9
# A negative review (sentiment ~-0.6) on a 5-star rating becomes: 5 * (1 - 0.6) = 2
# This correctly weighs the original rating by the sentiment of the text.
df_sample.loc[:, 'adjusted_rating'] = df_sample['Score'] * (1 + df_sample['sentiment_score'])
print("Created sentiment-adjusted ratings.")


# --- Step 4: Prepare Data for Collaborative Filtering ---
# To get reliable recommendations, we need items and users with a minimum number of interactions.
# This filtering step removes noise from users who only reviewed once or products with very few reviews.
min_product_ratings = 5 # A product should have at least 5 reviews
min_user_ratings = 5    # A user should have made at least 5 reviews

product_counts = df_sample['ProductId'].value_counts()
user_counts = df_sample['UserId'].value_counts()

popular_products = product_counts[product_counts >= min_product_ratings].index
active_users = user_counts[user_counts >= min_user_ratings].index

# We filter our main dataframe to only include these active users and popular products.
df_filtered = df_sample[df_sample['ProductId'].isin(popular_products) & df_sample['UserId'].isin(active_users)]
print(f"Original sample size: {len(df_sample)}, Filtered size for model: {len(df_filtered)}")


# --- Step 5: Build the User-Item Matrix ---
# We use a pivot_table to transform our data. Rows are users, columns are products,
# and the values are our new 'adjusted_rating'. This is the matrix our model will learn from.
# We use fillna(0) to handle cases where a user hasn't rated a product.
user_item_matrix = df_filtered.pivot_table(index='UserId', columns='ProductId', values='adjusted_rating').fillna(0)

# We convert our matrix to a sparse matrix format. This is a memory-saving trick,
# as our matrix is mostly full of zeros (sparse).
user_item_sparse_matrix = csr_matrix(user_item_matrix.values)

# We calculate the cosine similarity between items (the columns of our matrix).
# To do this, we transpose the matrix (.T) so that items become rows.
# Cosine similarity measures how similar two items are based on how users have rated them.
item_similarity = cosine_similarity(user_item_sparse_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)
print("Sentiment-aware item-item similarity matrix created.")


# --- Step 6: Build the Recommender Functions ---
def get_item_item_recommendations(product_id, num_recommendations=10):
    """
    Recommends items similar to a given product_id. This now uses the sentiment-aware model.
    """
    if product_id not in item_similarity_df.columns:
        # Return an empty list instead of a string for consistent output type
        return []

    similar_scores = item_similarity_df[product_id].sort_values(ascending=False)
    similar_items = similar_scores.drop(product_id).head(num_recommendations)
    return list(similar_items.index)

def get_item_item_recommendations_with_explanation(product_id, num_recommendations=10):
    """
    Generates recommendations with a simple explanation. This is our XAI function.
    """
    if product_id not in item_similarity_df.columns:
        return []
    
    similar_scores = item_similarity_df[product_id].sort_values(ascending=False)
    similar_items = similar_scores.drop(product_id).head(num_recommendations)
    
    recommendations = []
    for rec_id, score in similar_items.items():
        explanation = f"Recommended because users who liked '{product_id}' also liked this item (Similarity Score: {score:.2f})."
        recommendations.append({'productId': rec_id, 'explanation': explanation})
        
    return recommendations


# --- Example Usage ---
# This block will only run if you execute this file directly (python collaborative_filtering_model.py)
if __name__ == '__main__':
    print("\n--- Testing our Sentiment-Aware Collaborative Filtering Recommender ---")
    if not df_filtered.empty:
        # **CORRECTION**: We must select a specific product to test, e.g., the first one.
        test_product_id = df_filtered['ProductId'].iloc[0]
        
        # Test the new function with explanations
        recommendations = get_item_item_recommendations_with_explanation(test_product_id, 5)
        
        print(f"Recommendations with explanations for product '{test_product_id}':")
        # Pretty print the results
        for rec in recommendations:
            print(f"  - Product: {rec['productId']}\n    Explanation: {rec['explanation']}")
    else:
        print("Not enough data in the filtered set to run a test. Try relaxing the filter conditions (min_product_ratings).")