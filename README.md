# ShopSenseAI

> 🔗 **Live demo:** https://iambatman07-shopsenseai.hf.space · [HF Space](https://huggingface.co/spaces/IamBatman07/ShopSenseAI)

ShopSenseAI is a Flask-based product recommendation web app that combines collaborative filtering, content-based similarity, and popularity fallback in one project.

It includes:
- Personalized recommendations for known users.
- Cold-start recommendations for guest users based on selected products.
- Similar product lookup.
- Model evaluation endpoint using a temporal split.
- Single-page UI for exploring all flows.

## Features

- `SVD` matrix factorization recommender.
- `User-Based CF` recommender using user-user cosine similarity.
- `Item-Based CF` recommender using item-item cosine similarity.
- `Hybrid` recommender that blends SVD and Item-CF scores.
- `Content-Based` similar-item lookup using category, brand, and normalized price.
- `Trending` model using weighted interactions from recent 90-day activity.
- Guest cold-start recommendations from a product basket plus interaction intent.
- Evaluation pipeline with Precision@5 on a temporal 80/20 split.

## Tech Stack

- Python
- Flask
- pandas
- numpy
- scikit-learn
- scipy
- HTML/CSS/Vanilla JavaScript

## How It Works

1. Load and clean interaction data.
2. Build engagement score per interaction.
3. Aggregate to a user-item matrix.
4. Train all recommendation models.
5. Serve predictions through Flask APIs.

Engagement score formula used in `recommender.py`:

```text
engagement_score = interaction_score
                 + 0.5 * rating
                 + 0.3 * (purchase * scroll_depth_pct / 100)
                 - 1.5 * returned
```

Interaction weights:
- `view`: 1
- `wishlist`: 2
- `add_to_cart`: 3
- `purchase`: 5

## Setup

### 1. Clone and enter project

```bash
git clone <your-repo-url>
cd ShopSenseAI
```

### 2. Create and activate virtual environment (recommended)

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify dataset path

Make sure this file exists:

```text
data/product_recommendation_dataset_v2.csv
```

### 5. Run the app

```bash
python app.py
```

Open:

```text
http://localhost:5000
```