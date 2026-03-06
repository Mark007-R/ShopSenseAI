# Databricks Product Recommendation System

ALS-based collaborative filtering recommendation system for Databricks platform.

## Features

- **ALS Collaborative Filtering**: Scalable matrix factorization using Spark MLlib
- **Batch Processing**: Generate recommendations for all users efficiently
- **DBFS Integration**: Direct read/write to Databricks File System
- **Delta Lake Output**: Save recommendations in Delta format
- **PySpark Optimized**: Leverages Spark's distributed computing

## Setup

### 1. Upload Dataset to DBFS

```python
dbutils.fs.mkdirs("dbfs:/FileStore/datasets/")
dbutils.fs.cp("file:/path/to/product_recommendation_dataset_v2.csv", 
              "dbfs:/FileStore/datasets/product_recommendation_dataset_v2.csv")
```

### 2. Import Notebook

- Upload `Databricks_Recommendation_Notebook.ipynb` to your Databricks workspace
- Or upload `databricks_recommendation_system.py` and `databricks_main.py`

### 3. Run

Open the notebook and execute all cells, or run:

```python
%run ./databricks_main
```

## Configuration

Edit `config.py` for customization:

- `DATA_PATH`: DBFS path to dataset
- `OUTPUT_PATH`: DBFS path for batch recommendations
- `ALS_RANK`: Model latent factors (default: 10)
- `ALS_MAX_ITER`: Training iterations (default: 10)
- `ALS_REG_PARAM`: Regularization (default: 0.01)
- `DEFAULT_N_RECOMMENDATIONS`: Recommendations per user (default: 20)

## Usage

### Interactive Mode

```python
from databricks_recommendation_system import DatabricksRecommendationSystem
import config

rec_system = DatabricksRecommendationSystem()
df = rec_system.load_data(config.DATA_PATH)
interactions_df = rec_system.preprocess_interactions(df)
model = rec_system.build_als_model(interactions_df)

# Get recommendations for specific users
user_recs = rec_system.get_user_recommendations(["USER_12345"], 10)
display(user_recs)
```

### Batch Mode

```python
# Generate recommendations for all users
batch_recs = rec_system.generate_batch_recommendations(
    interactions_df,
    num_recommendations=20,
    output_path=config.OUTPUT_PATH
)
```

## File Structure

- `databricks_recommendation_system.py`: Core ALS recommendation engine
- `Databricks_Recommendation_Notebook.ipynb`: Interactive notebook
- `databricks_main.py`: Automated pipeline script
- `config.py`: Configuration settings
- `requirements.txt`: Dependencies (PySpark 3.3+)
- `datasets/`: Sample dataset folder

## Model Details

**ALS (Alternating Least Squares)**:
- User-product interaction matrix factorization
- Implicit feedback scoring
- Cold-start strategy: drop
- Non-negative constraints
- RMSE evaluation

## Output Format

Batch recommendations saved as Delta table with columns:
- `user_id`: User identifier
- `product_id`: Recommended product
- `recommendation_score`: Predicted rating

## Performance

- Handles 60,000+ interactions
- Distributed processing across cluster
- Delta Lake for efficient storage
- Adaptive query execution enabled
