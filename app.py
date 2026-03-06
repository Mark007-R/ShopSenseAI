"""
Flask Web Application for Product Recommendation System
"""

from flask import Flask, request, jsonify, render_template
import os
import time

app = Flask(__name__)

# ── Load recommender on startup ──────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'product_recommendation_dataset_v2.csv')

print("⏳ Initializing recommendation engine...")
t0 = time.time()
from recommender import ProductRecommender
engine = ProductRecommender(DATA_PATH)
print(f"✅ Engine ready in {time.time() - t0:.2f}s")


# ── ROUTES ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    stats = engine.get_stats()
    users = engine.get_all_users()
    products = engine.get_all_products()
    return render_template('index.html', stats=stats, users=users, products=products)


@app.route('/api/recommend', methods=['GET'])
def recommend():
    """
    GET /api/recommend?user_id=U001&algo=hybrid&top_n=6
    Returns top-N product recommendations for a user.
    """
    user_id = request.args.get('user_id', '').strip()
    algo = request.args.get('algo', 'hybrid')
    top_n = int(request.args.get('top_n', 6))

    if not user_id:
        return jsonify({'error': 'user_id is required'}), 400

    recs = engine.get_recommendations(user_id, algo=algo, top_n=top_n)
    history = engine.get_user_history(user_id)
    return jsonify({
        'user_id': user_id,
        'algorithm': algo,
        'recommendations': recs,
        'history': history
    })


@app.route('/api/similar', methods=['GET'])
def similar_products():
    """
    GET /api/similar?product_id=P101&top_n=5
    Returns products similar to the given product (content-based).
    """
    product_id = request.args.get('product_id', '').strip()
    top_n = int(request.args.get('top_n', 5))

    if not product_id:
        return jsonify({'error': 'product_id is required'}), 400

    similar = engine.get_similar_products(product_id, top_n=top_n)
    return jsonify({
        'product_id': product_id,
        'similar_products': similar
    })


@app.route('/api/trending', methods=['GET'])
def trending():
    """GET /api/trending?top_n=6 — Returns trending products"""
    top_n = int(request.args.get('top_n', 6))
    return jsonify({'trending': engine.get_trending(top_n=top_n)})


@app.route('/api/evaluate', methods=['GET'])
def evaluate():
    """GET /api/evaluate — Returns Precision@5 for all models"""
    print("Running evaluation (may take ~30s)...")
    results = engine.get_evaluation_results()
    return jsonify({'evaluation': results})


@app.route('/api/stats', methods=['GET'])
def stats():
    """GET /api/stats — Dataset statistics"""
    return jsonify(engine.get_stats())


@app.route('/api/history/<user_id>', methods=['GET'])
def history(user_id):
    """GET /api/history/<user_id> — User's interaction history"""
    h = engine.get_user_history(user_id, top_n=10)
    return jsonify({'user_id': user_id, 'history': h})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
