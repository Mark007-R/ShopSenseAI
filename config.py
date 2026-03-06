DATA_PATH = "datasets/product_recommendation_dataset_v2.csv"
DEFAULT_METHOD = "hybrid"
DEFAULT_N_RECOMMENDATIONS = 20
DEFAULT_N_SIMILAR_USERS = 10
HYBRID_WEIGHTS = {
    "collaborative_user": 0.3,
    "collaborative_item": 0.3,
    "content_based": 0.4,
}
DEFAULT_BATCH_USER_LIMIT = 100
DEFAULT_EVAL_USER_LIMIT = 50
DEFAULT_OUTPUT_PATH = "recommendations_output.csv"
MIN_PURCHASE_THRESHOLD = 1
