# Product Recommendation System

## Overview

A hybrid recommendation system combining collaborative filtering and content-based filtering for e-commerce product recommendations.

## Features

- User-Based Collaborative Filtering
- Item-Based Collaborative Filtering
- Content-Based Filtering
- Hybrid Approach
- Cold-Start Handling
- Batch Processing
- Databricks Support

## Installation

```bash
pip install pandas numpy scikit-learn scipy
```

## Usage

### Quick Start

```python
from product_recommendation_system import ProductRecommendationSystem

rec_system = ProductRecommendationSystem('product_recommendation_dataset_v2.csv')
result = rec_system.get_recommendations(user_id='U00001', method='hybrid', n_recommendations=5)
print(result)
```

### Interactive Mode

```bash
python QUICKSTART.py
```

### Batch Processing

```python
rec_system.generate_batch_recommendations(
    method='hybrid',
    n_recommendations=5,
    output_path='recommendations.csv'
)
```

## Configuration

Edit `config.py` to customize:
- DEFAULT_METHOD
- DEFAULT_N_RECOMMENDATIONS
- HYBRID_WEIGHTS
- MIN_PURCHASE_THRESHOLD

## File Structure

- `product_recommendation_system.py` - Main recommendation engine
- `databricks_recommendation_system.py` - Databricks/PySpark implementation
- `config.py` - Configuration settings
- `QUICKSTART.py` - Interactive demo script
- `datasets/product_recommendation_dataset_v2.csv` - Sample dataset

## Methods

### Collaborative Filtering (User-Based)
Recommends products based on similar users' preferences.

### Collaborative Filtering (Item-Based)
Recommends products similar to those the user has purchased.

### Content-Based Filtering
Recommends products in similar categories/brands to user preferences.

### Hybrid
Combines all methods with weighted scores for best results.

## License

Open source
