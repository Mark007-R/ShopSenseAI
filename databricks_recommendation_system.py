"""
Databricks-Optimized Product Recommendation System
==================================================

This module provides a production-ready product recommendation system
optimized for distributed computing on Databricks using PySpark.

Features:
- Distributed collaborative filtering
- Apache Spark optimizations
- MLlib integration
- Delta Lake support
- Horizontal scalability

Usage in Databricks:
1. Upload product_recommendation_dataset_v2.csv to Databricks FileStore
2. Run the notebook cells in sequence
3. Use the recommendation functions for real-time and batch predictions
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, row_number, count, avg, max, min, when
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.recommendation import ALS
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
import numpy as np
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabricksRecommendationSystem:
    """
    Databricks-optimized collaborative filtering recommendation system using PySpark.
    """
    
    def __init__(self, spark: SparkSession = None):
        """
        Initialize the system.
        
        Args:
            spark: SparkSession instance (created if None)
        """
        self.spark = spark or SparkSession.builder \
            .appName("ProductRecommendationSystem") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adapter.skewJoin.enabled", "true") \
            .getOrCreate()
        
        self.als_model = None
        self.user_indexer = None
        self.product_indexer = None
        logger.info("✓ Databricks Recommendation System initialized")
    
    def load_data(self, data_path: str) -> 'pyspark.sql.DataFrame':
        """
        Load data from CSV and create Spark DataFrame.
        
        Args:
            data_path: Path to CSV file (can be /dbfs/FileStore/... on Databricks)
            
        Returns:
            Spark DataFrame
        """
        logger.info(f"📂 Loading data from {data_path}")
        
        df = self.spark.read \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .csv(data_path)
        
        # Cache for performance
        df.cache()
        
        logger.info(f"✓ Loaded {df.count()} records")
        logger.info(f"✓ Columns: {', '.join(df.columns)}")
        
        return df
    
    def preprocess_interactions(self, df: 'pyspark.sql.DataFrame') -> 'pyspark.sql.DataFrame':
        """
        Preprocess and create interaction ratings for collaborative filtering.
        
        Args:
            df: Raw data DataFrame
            
        Returns:
            Processed DataFrame with user_id, product_id, rating columns
        """
        logger.info("🔨 Preprocessing interactions...")
        
        # Create interaction score
        interaction_df = df.select(
            col('user_id'),
            col('product_id'),
            col('product_name'),
            col('category'),
            col('brand'),
            col('listed_price_inr'),
            col('interaction_type'),
            col('session_duration_sec'),
            col('pages_visited'),
            col('review_given'),
            col('rating'),
            col('purchase')
        )
        
        # Calculate interaction score (1-5 scale for ALS)
        interaction_df = interaction_df.withColumn(
            'interaction_score',
            when(col('interaction_type') == 'purchase', 5.0)
            .when(col('interaction_type') == 'add_to_cart', 3.0)
            .when(col('interaction_type') == 'wishlist', 2.0)
            .otherwise(1.0)
        )
        
        # Include review boost
        interaction_df = interaction_df.withColumn(
            'interaction_score',
            col('interaction_score') + when(col('review_given') == 1, 1.0).otherwise(0.0)
        )
        
        # Cap score at 5 for consistent rating scale
        interaction_df = interaction_df.withColumn(
            'interaction_score',
            when(col('interaction_score') > 5, 5.0).otherwise(col('interaction_score'))
        )
        
        logger.info("✓ Interaction preprocessing complete")
        
        return interaction_df
    
    def build_als_model(
        self,
        interactions_df: 'pyspark.sql.DataFrame',
        rank: int = 10,
        max_iter: int = 10,
        reg_param: float = 0.01,
        alpha: float = 1.0
    ) -> 'pyspark.ml.recommendation.ALSModel':
        """
        Build ALS (Alternating Least Squares) collaborative filtering model.
        
        Args:
            interactions_df: DataFrame with user_id, product_id, interaction_score
            rank: Number of latent factors
            max_iter: Maximum iterations
            reg_param: Regularization parameter
            alpha: Alpha for implicit feedback
            
        Returns:
            Trained ALS model
        """
        logger.info("🤖 Building ALS Collaborative Filtering model...")
        
        # Encode user and product IDs for ALS (requires integer IDs)
        self.user_indexer = StringIndexer(inputCol='user_id', outputCol='user_id_indexed')
        user_indexed = self.user_indexer.fit(interactions_df).transform(interactions_df)
        
        self.product_indexer = StringIndexer(inputCol='product_id', outputCol='product_id_indexed')
        product_indexed = self.product_indexer.fit(user_indexed).transform(user_indexed)
        
        # Initialize and fit ALS model
        als = ALS(
            maxIter=max_iter,
            regParam=reg_param,
            userCol='user_id_indexed',
            itemCol='product_id_indexed',
            ratingCol='interaction_score',
            coldStartStrategy='drop',
            nonnegative=True,
            rank=rank,
            alpha=alpha
        )
        
        self.als_model = als.fit(product_indexed)
        
        # Evaluate with RMSE
        predictions = self.als_model.transform(product_indexed)
        evaluator = RegressionEvaluator(
            metricName='rmse',
            labelCol='interaction_score',
            predictionCol='prediction'
        )
        rmse = evaluator.evaluate(predictions)
        
        logger.info(f"✓ ALS Model built (Rank={rank}, RMSE={rmse:.4f})")
        
        return self.als_model
    
    def get_user_recommendations(
        self,
        user_ids: List[str],
        num_recommendations: int = 5
    ) -> 'pyspark.sql.DataFrame':
        """
        Get recommendations for specific users.
        
        Args:
            user_ids: List of user IDs
            num_recommendations: Number of products to recommend
            
        Returns:
            DataFrame with recommendations
        """
        logger.info(f"🎯 Generating recommendations for {len(user_ids)} users...")
        
        if not self.als_model:
            raise ValueError("ALS model not trained. Call build_als_model first.")
        if not self.user_indexer:
            raise ValueError("User indexer not fitted. Call build_als_model first.")
        
        # Create DataFrame of unique user indices
        user_df = self.spark.createDataFrame(
            user_ids,
            'user_id'
        )
        user_df = self.user_indexer.transform(user_df)
        
        # Get recommendations
        recommendations = self.als_model.recommendForUserSubset(
            user_df,
            num_recommendations
        )
        
        logger.info(f"✓ Generated recommendations for {recommendations.count()} users")
        
        return recommendations
    
    def get_product_recommendations(
        self,
        product_ids: List[str],
        num_recommendations: int = 5
    ) -> 'pyspark.sql.DataFrame':
        """
        Get similar products (item-based recommendations).
        
        Args:
            product_ids: List of product IDs
            num_recommendations: Number of similar products
            
        Returns:
            DataFrame with similar products
        """
        logger.info(f"📦 Finding similar products for {len(product_ids)} items...")
        
        if not self.als_model:
            raise ValueError("ALS model not trained. Call build_als_model first.")
        if not self.product_indexer:
            raise ValueError("Product indexer not fitted. Call build_als_model first.")
        
        # Create DataFrame of unique product indices
        product_df = self.spark.createDataFrame(
            product_ids,
            'product_id'
        )
        product_df = self.product_indexer.transform(product_df)
        
        # Get similar products
        recommendations = self.als_model.recommendForItemSubset(
            product_df,
            num_recommendations
        )
        
        logger.info(f"✓ Generated recommendations for {recommendations.count()} products")
        
        return recommendations
    
    def generate_batch_recommendations(
        self,
        interactions_df: 'pyspark.sql.DataFrame',
        num_recommendations: int = 5,
        output_path: str = None
    ) -> 'pyspark.sql.DataFrame':
        """
        Generate recommendations for all users.
        
        Args:
            interactions_df: Interaction DataFrame
            num_recommendations: Number of recommendations per user
            output_path: Optional path to save (Delta Lake format)
            
        Returns:
            DataFrame with all recommendations
        """
        logger.info(f"⚡ Generating batch recommendations for all users...")
        
        if not self.als_model:
            raise ValueError("ALS model not trained. Call build_als_model first.")
        
        # Get unique users
        users = interactions_df.select('user_id').distinct()
        
        # Get recommendations
        recommendations = self.als_model.recommendForUserSubset(users, num_recommendations)
        
        # Flatten recommendations
        from pyspark.sql.functions import explode, col
        
        flattened = recommendations.select(
            col('user_id'),
            explode(col('recommendations')).alias('recommendation')
        ).select(
            col('user_id'),
            col('recommendation.product_id').alias('product_id'),
            col('recommendation.rating').alias('recommendation_score')
        )
        
        # Save to Delta Lake
        if output_path:
            flattened.write \
                .format('delta') \
                .mode('overwrite') \
                .save(output_path)
            logger.info(f"✓ Recommendations saved to {output_path}")
        
        logger.info(f"✓ Generated {flattened.count()} recommendations")
        
        return flattened
    
    def get_stats(self, df: 'pyspark.sql.DataFrame') -> Dict:
        """
        Get dataset statistics for monitoring.
        
        Args:
            df: Data DataFrame
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_records': df.count(),
            'unique_users': df.select('user_id').distinct().count(),
            'unique_products': df.select('product_id').distinct().count(),
            'avg_interaction_score': df.agg({'interaction_score': 'avg'}).collect()[0][0],
            'date_range': {
                'min': df.agg({'interaction_date': 'min'}).collect()[0][0],
                'max': df.agg({'interaction_date': 'max'}).collect()[0][0]
            }
        }
        return stats


def databricks_example():
    """
    Example usage in Databricks environment.
    
    In Databricks notebook, use:
    
    spark = spark  # Provided by Databricks
    
    # Initialize system
    rec_system = DatabricksRecommendationSystem(spark)
    
    # Load data from FileStore
    interactions = rec_system.load_data('/dbfs/FileStore/product_recommendation_dataset_v2.csv')
    
    # Preprocess
    interactions = rec_system.preprocess_interactions(interactions)
    
    # Build model
    als_model = rec_system.build_als_model(interactions, rank=10, max_iter=20)
    
    # Generate recommendations
    recommendations = rec_system.generate_batch_recommendations(
        interactions,
        num_recommendations=5,
        output_path='/dbfs/output/recommendations'
    )
    
    # Display results
    display(recommendations)
    """
    pass


if __name__ == "__main__":
    print("""
    Databricks Product Recommendation System
    ========================================
    
    This module is designed to run in Databricks environment.
    Use the functions above in a Databricks notebook.
    
    See databricks_example() for usage patterns.
    """)
