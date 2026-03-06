# 🛍️ ShopSense — AI Product Recommendation System

A production-grade recommendation engine built with Flask, trained on 60,000 real user-product interactions across 5,000 users and 50 products spanning 6 categories.

---

## 🏗️ Architecture

```
Dataset (60K interactions, 37 features)
    ↓
Data Preprocessing
 • Interaction weighting (view=1, wishlist=2, cart=3, purchase=5)
 • Engagement score = interaction + rating boost − return penalty
 • Feature normalization & encoding
    ↓
User-Item Matrix (5000 × 50 engagement pivot)
    ↓
Recommendation Algorithms
 ├── SVD Matrix Factorization  (Model-based CF)
 ├── User-Based CF             (Memory-based CF)
 ├── Item-Based CF             (Memory-based CF)
 ├── Content-Based Filtering   (Category + Brand + Price similarity)
 └── Trending / Popularity     (Cold-start fallback)
    ↓
Model Training
 • SVD: k=20 latent factors on sparse matrix
 • Cosine similarity matrices for CF models
 • Content feature vectors via LabelEncoder + MinMaxScaler
    ↓
Evaluation
 • Precision@5 using Leave-One-Out
 • Catalog Coverage metric
    ↓
Flask Web App / REST API
 • GET /api/recommend
 • GET /api/similar
 • GET /api/trending
 • GET /api/evaluate
 • GET /api/stats
 • GET /api/history/<user_id>
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place dataset
```
recommendation_app/
└── data/
    └── product_recommendation_dataset_v2.csv
```

### 3. Run the app
```bash
python app.py
```
Visit: **http://localhost:5000**

---

## 📡 API Reference

### `GET /api/recommend`
Returns personalized product recommendations for a user.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `user_id` | string | required | e.g. `U00192` |
| `algo` | string | `hybrid` | `hybrid`, `svd`, `user_cf`, `item_cf` |
| `top_n` | int | `6` | Number of results |

**Response:**
```json
{
  "user_id": "U00192",
  "algorithm": "hybrid",
  "recommendations": [
    {
      "product_id": "P101",
      "product_name": "Laptop Pro 15",
      "category": "Electronics",
      "brand": "Dell",
      "price": 65999,
      "score": 4.872
    }
  ],
  "history": [...]
}
```

### `GET /api/similar`
Content-based similar product lookup.
```
GET /api/similar?product_id=P101&top_n=5
```

### `GET /api/trending`
Returns popularity-ranked products from last 90 days.
```
GET /api/trending?top_n=6
```

### `GET /api/evaluate`
Runs Precision@5 Leave-One-Out evaluation on all models (~30–60s).
```
GET /api/evaluate
```

### `GET /api/stats`
Dataset overview and distribution stats.

### `GET /api/history/<user_id>`
Full interaction history for a user.

---

## 🧠 Models Explained

| Model | Type | Strength |
|-------|------|----------|
| **SVD Matrix Factorization** | Model-based CF | Captures latent user-product relationships; handles sparsity well |
| **User-Based CF** | Memory-based CF | Finds look-alike users and borrows their preferences |
| **Item-Based CF** | Memory-based CF | Recommends items co-liked with the user's history |
| **Content-Based** | Feature similarity | Works without interaction data; great for similar-item widgets |
| **Hybrid (SVD + Item-CF)** | Ensemble | Best overall accuracy (0.6×SVD + 0.4×Item-CF weighted blend) |
| **Trending** | Popularity | Cold-start fallback for new/unknown users |

### Engagement Score Formula
```
score = interaction_weight
      + (rating × 0.5)
      + (purchase × scroll_depth / 100 × 0.3)
      - (returned × 1.5)
```

---

## 📁 Project Structure
```
recommendation_app/
├── app.py                  # Flask routes & API endpoints
├── recommender.py          # Full ML pipeline
├── requirements.txt
├── data/
│   └── product_recommendation_dataset_v2.csv
└── templates/
    └── index.html          # Interactive web dashboard
```

---

## ✨ What Would Make This Project STAND OUT

### 🔴 High Impact (implement these first)

**1. Real-Time Feedback Loop**
Track clicks on recommendations, capture implicit feedback, and retrain incrementally. A model that learns from its own suggestions creates a virtuous cycle — this is what separates Netflix-level systems from basic demos.

**2. A/B Testing Framework**
Serve different algorithms to different user cohorts and measure CTR, conversion, and revenue impact. Add a `/api/experiment` endpoint that randomly assigns users to control vs treatment. This is industry-standard and immediately credible in any interview or demo.

**3. Explainable Recommendations**
Add a `reason` field to each recommendation: *"Because you purchased Iron by Philips"* or *"Trending in Home Appliances this week."* Explainability increases user trust and click-through rates by 15–30% in literature.

### 🟡 Medium Impact (differentiation features)

**4. Contextual Bandits / Online Learning**
Use a multi-armed bandit (e.g., LinUCB) that balances exploration (showing new products) and exploitation (showing high-confidence picks). This outperforms static models in dynamic catalogs.

**5. Session-Based Recommendations**
Use an RNN or GRU to model within-session behavior: "User viewed A → B → C, next likely product is D." This is critical for retail because most sessions are anonymous.

**6. Deep Learning Embedding Model**
Train a two-tower neural network (user tower + item tower) using TensorFlow/PyTorch. Embed users and products into a shared latent space for ANN retrieval. This approach powers Amazon, YouTube, and Spotify at scale.

**7. Segment-Aware Recommendations**
The dataset has `user_segment` (tech_enthusiast, kitchen_chef, etc.). Add a segment-level prior — new users in "fitness_freak" segment immediately get fitness-weighted recommendations rather than generic trending.

**8. Price Sensitivity Modeling**
Use `clv_category` and `membership_tier` to adjust recommendations by price band. A Bronze/Low CLV user should not consistently receive ₹89,999 iPhone recommendations.

### 🟢 Polish & Production (makes it portfolio-ready)

**9. Redis Caching Layer**
Cache user recommendation vectors in Redis with a 1-hour TTL. Reduces latency from ~200ms to ~5ms. Critical for production credibility.

**10. Recommendation Diversity**
Add a Maximum Marginal Relevance (MMR) post-processing step to ensure recommended products span multiple categories. Pure score optimization leads to category collapse (all recommendations from one category).

**11. Cold-Start Onboarding Quiz**
A 3-question onboarding flow ("What's your primary interest?", "Budget range?", "Device preference?") that seeds a new user's preference vector without requiring any history. Boosts initial engagement.

**12. Monitoring Dashboard**
Track recommendation metrics over time: mean precision, catalog coverage, novelty score, click-through rate. A live dashboard demonstrates operational maturity beyond a Jupyter notebook.

---

## 📊 Evaluation Metrics Explained

| Metric | What it measures | Why it matters |
|--------|-----------------|----------------|
| **Precision@K** | Of K recommended items, how many did the user actually interact with? | Primary accuracy metric |
| **Catalog Coverage** | % of total products ever recommended across all users | Diversity — prevents popularity bias |
| **Novelty** | Average inverse popularity of recommended items | Avoids recommending only blockbusters |
| **Mean Reciprocal Rank** | How high does the first relevant item rank? | Quality of top recommendation |

---

## 💡 Business Impact

| Outcome | Industry Benchmark |
|---------|-------------------|
| Increase in average order value | +10–35% (Amazon: 35% revenue from recommendations) |
| Improvement in session duration | +20–40% |
| Reduction in bounce rate | -15–25% |
| Customer retention uplift | +8–15% with personalization |

A recommendation system that explains its suggestions, learns from feedback, and handles cold-start users is the difference between a classroom project and a production system worth deploying.

---

## 🛠️ Tech Stack
- **Backend**: Python 3.10+, Flask 2.3
- **ML**: NumPy, SciPy (SVD), Scikit-learn (cosine similarity, encoders)
- **Data**: Pandas
- **Frontend**: Vanilla JS, Google Fonts (Syne + DM Sans)
