"""
Central configuration for the recommendation project.

Update values in this file to test different behaviors without changing
core logic files.
"""

# Data
DATA_PATH = "product_recommendation_dataset_v2.csv"

# Recommendation defaults
DEFAULT_METHOD = "hybrid"  # Options: user_cf, item_cf, content, hybrid
DEFAULT_N_RECOMMENDATIONS = 20
DEFAULT_N_SIMILAR_USERS = 10

# Hybrid weights (used when method == hybrid)
HYBRID_WEIGHTS = {
    "collaborative_user": 0.3,
    "collaborative_item": 0.3,
    "content_based": 0.4,
}

# Evaluation and batch processing
DEFAULT_BATCH_USER_LIMIT = 100
DEFAULT_EVAL_USER_LIMIT = 50
DEFAULT_OUTPUT_PATH = "recommendations_output.csv"

# System behavior
MIN_PURCHASE_THRESHOLD = 1
