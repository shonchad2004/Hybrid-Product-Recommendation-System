# app.py

import pandas as pd
from flask import Flask, request, jsonify

# Import the recommendation functions from our other model files
from popularity_model import recommend_popular_products
from collaborative_filtering_model import get_item_item_recommendations_with_explanation, df_filtered as df_collab_filtered
from content_based_model import get_content_based_recommendations_with_explanation, metadata_subset

# --- Initialize the Flask App ---
# This creates an instance of the Flask web application.
app = Flask(__name__)

# --- The Hybrid Recommender Logic (no changes needed here) ---
def hybrid_recommendations_with_explanations(user_id, product_id, num_recommendations=10):
    final_recommendations = {}

    # Get recommendations from BOTH personalized models
    collab_recs = get_item_item_recommendations_with_explanation(product_id, num_recommendations)
    content_recs = get_content_based_recommendations_with_explanation(product_id, num_recommendations)
    
    # Combine recommendations, prioritizing collaborative
    for rec in collab_recs:
        if rec['productId'] not in final_recommendations:
            final_recommendations[rec['productId']] = rec['explanation']
    
    for rec in content_recs:
        if rec['productId'] not in final_recommendations:
            final_recommendations[rec['productId']] = rec['explanation']

    # Fill with Popularity-Based Recommendations if needed
    if len(final_recommendations) < num_recommendations:
        needed = num_recommendations - len(final_recommendations)
        popular_recs_df = recommend_popular_products(needed * 2)
        
        for prod_id in popular_recs_df['productId']:
            if len(final_recommendations) < num_recommendations and prod_id not in final_recommendations:
                final_recommendations[prod_id] = "Recommended because it is one of a top-selling product."

    # Format and truncate the final list
    final_output = []
    for prod_id, explanation in final_recommendations.items():
        final_output.append({'productId': prod_id, 'explanation': explanation})
    
    return final_output[:num_recommendations]

# --- Define the API Endpoint ---
# The @app.route decorator tells Flask which URL should trigger our function.
# '/recommend' is the path. methods=['GET'] means it responds to GET requests.
@app.route('/recommend', methods=['GET'])
def recommend():
    """
    This function is the API endpoint. It gets triggered when a user visits the URL.
    It expects a 'product_id' as a URL parameter.
    Example: http://127.0.0.1:5000/recommend?product_id=B0030VJ8E0
    """
    # request.args.get() is how we read URL parameters.
    product_id = request.args.get('product_id')
    
    # Simple error handling if the product_id is missing.
    if not product_id:
        return jsonify({"error": "Please provide a 'product_id' parameter."}), 400
    
    # A placeholder user_id. In a real system, this would come from a login session.
    # We need to find a user who has actually reviewed the given product_id for the model to work.
    try:
        user_id = df_collab_filtered[df_collab_filtered['ProductId'] == product_id]['UserId'].iloc[0]
    except IndexError:
        return jsonify({"error": f"Product ID '{product_id}' not found in our filtered dataset."}), 404

    # Call our main hybrid function to get the recommendations.
    recommendations = hybrid_recommendations_with_explanations(user_id, product_id, 10)
    
    # jsonify() converts our Python list of dictionaries into a proper JSON response.
    return jsonify(recommendations)

# --- Run the App ---
# The if __name__ == '__main__': block ensures this code only runs
# when you execute 'python app.py' directly.
if __name__ == '__main__':
    # app.run() starts the development server.
    # debug=True allows the server to auto-reload when you save changes.
    print("Starting Flask server... To get recommendations, visit:")
    print("http://127.0.0.1:5000/recommend?product_id=YOUR_PRODUCT_ID")
    app.run(debug=True)