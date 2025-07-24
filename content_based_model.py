# content_based_model.py

import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# --- Step 1: Import the EXACT SAME filtered data from the collaborative model ---
# This is the key to ensuring our models know about the same products.
# We import the 'df_filtered' DataFrame that has already been cleaned and filtered.
from collaborative_filtering_model import df_filtered as df_model_data

print("Content-Based Model: Using filtered data from collaborative model.")

# --- Step 2: Get the unique products from this guaranteed data source ---
# Now, our unique_products list is guaranteed to have products that the other model knows.
unique_products = df_model_data['ProductId'].unique()

# --- Step 3: Create Mock Metadata for this specific set of products ---
# We create a "description" for each product, just like before.
categories = ['organic', 'snack', 'gluten-free', 'coffee', 'tea', 'healthy', 'treat', 'beverage']
product_metadata = pd.DataFrame({
    'productId': unique_products,
    'description': [' '.join(random.sample(categories, k=random.randint(2, 4))) for _ in unique_products]
})
print(f"Content-Based Model: Mock metadata created for {len(unique_products)} products.")

# No need to create a subset anymore, as we are already working with a small, relevant dataset.
metadata_subset = product_metadata

# --- Step 4: TF-IDF Vectorization ---
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(metadata_subset['description'])
print("Content-Based Model: TF-IDF matrix created.")

# --- Step 5: Calculate Content Similarity ---
cosine_sim_content = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create a mapping from productId to the matrix index for easy lookup
indices = pd.Series(metadata_subset.index, index=metadata_subset['productId']).drop_duplicates()
print("Content-Based Model: Similarity matrix created.")

# --- Step 6: Recommender Functions (with explanations) ---
# This logic remains the same, but it's now operating on the harmonized data.
def get_content_based_recommendations_with_explanation(product_id, num_recommendations=10):
    recommendations_with_explanation = []
    
    if product_id not in indices:
        return recommendations_with_explanation

    idx = indices[product_id]
    sim_scores = list(enumerate(cosine_sim_content[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    product_indices = [i[0] for i in sim_scores]
    recommended_product_ids = metadata_subset['productId'].iloc[product_indices]

    try:
        product_desc = metadata_subset.loc[metadata_subset['productId'] == product_id, 'description'].iloc[0]
    except IndexError:
        return []

    for rec_id in recommended_product_ids:
        rec_desc = metadata_subset.loc[metadata_subset['productId'] == rec_id, 'description'].iloc[0]
        common_features = set(product_desc.split()) & set(rec_desc.split())
        explanation = f"Recommended because it shares features like {list(common_features)} with '{product_id}'."
        recommendations_with_explanation.append({'productId': rec_id, 'explanation': explanation})
        
    return recommendations_with_explanation

# --- Example Usage ---
if __name__ == '__main__':
    print("\n--- Testing our Content-Based Recommender ---")
    if not metadata_subset.empty:
        test_product_id = metadata_subset['productId'].iloc[0]
        recommendations = get_content_based_recommendations_with_explanation(test_product_id, 5)

        if recommendations:
            print(f"Recommendations with explanations for product '{test_product_id}':")
            for rec in recommendations:
                print(f"  - Product: {rec['productId']}\n    Explanation: {rec['explanation']}")