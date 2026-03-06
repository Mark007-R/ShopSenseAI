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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabricksRecommendationSystem:
    
    def __init__(self, spark: SparkSession = None):
        self.spark = spark or SparkSession.builder \
            .appName("ProductRecommendationSystem") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adapter.skewJoin.enabled", "true") \
            .getOrCreate()
        self.als_model = None
        self.user_indexer = None
        self.product_indexer = None
        logger.info("Databricks Recommendation System initialized")
    
    def load_data(self, data_path: str) -> 'pyspark.sql.DataFrame':
        logger.info(f"Loading data from {data_path}")
        df = self.spark.read \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .csv(data_path)
        df.cache()
        logger.info(f"Loaded {df.count()} records")
        logger.info(f"Columns: {', '.join(df.columns)}")
        return df
    
    def preprocess_interactions(self, df: 'pyspark.sql.DataFrame') -> 'pyspark.sql.DataFrame':
        logger.info("Preprocessing interactions...")
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
        interaction_df = interaction_df.withColumn(
            'interaction_score',
            when(col('interaction_type') == 'purchase', 5.0)
            .when(col('interaction_type') == 'add_to_cart', 3.0)
            .when(col('interaction_type') == 'wishlist', 2.0)
            .otherwise(1.0)
        )
        interaction_df = interaction_df.withColumn(
            'interaction_score',
            col('interaction_score') + when(col('review_given') == 1, 1.0).otherwise(0.0)
        )
        interaction_df = interaction_df.withColumn(
            'interaction_score',
            when(col('interaction_score') > 5, 5.0).otherwise(col('interaction_score'))
        )
        logger.info("Interaction preprocessing complete")
        return interaction_df
    
    def build_als_model(
        self,
        interactions_df: 'pyspark.sql.DataFrame',
        rank: int = 10,
        max_iter: int = 10,
        reg_param: float = 0.01,
        alpha: float = 1.0
    ) -> 'pyspark.ml.recommendation.ALSModel':
        logger.info("Building ALS Collaborative Filtering model...")
        self.user_indexer = StringIndexer(inputCol='user_id', outputCol='user_id_indexed')
        user_indexed = self.user_indexer.fit(interactions_df).transform(interactions_df)
        self.product_indexer = StringIndexer(inputCol='product_id', outputCol='product_id_indexed')
        product_indexed = self.product_indexer.fit(user_indexed).transform(user_indexed)
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
        predictions = self.als_model.transform(product_indexed)
        evaluator = RegressionEvaluator(
            metricName='rmse',
            labelCol='interaction_score',
            predictionCol='prediction'
        )
        rmse = evaluator.evaluate(predictions)
        logger.info(f"ALS Model built (Rank={rank}, RMSE={rmse:.4f})")
        return self.als_model
    
    def get_user_recommendations(
        self,
        user_ids: List[str],
        num_recommendations: int = 5
    ) -> 'pyspark.sql.DataFrame':
        logger.info(f"Generating recommendations for {len(user_ids)} users...")
        if not self.als_model:
            raise ValueError("ALS model not trained. Call build_als_model first.")
        if not self.user_indexer:
            raise ValueError("User indexer not fitted. Call build_als_model first.")
        user_df = self.spark.createDataFrame(
            user_ids,
            'user_id'
        )
        user_df = self.user_indexer.transform(user_df)
        recommendations = self.als_model.recommendForUserSubset(
            user_df,
            num_recommendations
        )
        logger.info(f"Generated recommendations for {recommendations.count()} users")
        return recommendations
    
    def get_product_recommendations(
        self,
        product_ids: List[str],
        num_recommendations: int = 5
    ) -> 'pyspark.sql.DataFrame':
        logger.info(f"Finding similar products for {len(product_ids)} items...")
        if not self.als_model:
            raise ValueError("ALS model not trained. Call build_als_model first.")
        if not self.product_indexer:
            raise ValueError("Product indexer not fitted. Call build_als_model first.")
        product_df = self.spark.createDataFrame(
            product_ids,
            'product_id'
        )
        product_df = self.product_indexer.transform(product_df)
        recommendations = self.als_model.recommendForItemSubset(
            product_df,
            num_recommendations
        )
        logger.info(f"Generated recommendations for {recommendations.count()} products")
        return recommendations
    
    def generate_batch_recommendations(
        self,
        interactions_df: 'pyspark.sql.DataFrame',
        num_recommendations: int = 5,
        output_path: str = None
    ) -> 'pyspark.sql.DataFrame':
        logger.info(f"Generating batch recommendations for all users...")
        if not self.als_model:
            raise ValueError("ALS model not trained. Call build_als_model first.")
        users = interactions_df.select('user_id').distinct()
        recommendations = self.als_model.recommendForUserSubset(users, num_recommendations)
        from pyspark.sql.functions import explode, col
        flattened = recommendations.select(
            col('user_id'),
            explode(col('recommendations')).alias('recommendation')
        ).select(
            col('user_id'),
            col('recommendation.product_id').alias('product_id'),
            col('recommendation.rating').alias('recommendation_score')
        )
        if output_path:
            flattened.write \
                .format('delta') \
                .mode('overwrite') \
                .save(output_path)
            logger.info(f"Recommendations saved to {output_path}")
        logger.info(f"Generated {flattened.count()} recommendations")
        return flattened
    
    def get_stats(self, df: 'pyspark.sql.DataFrame') -> Dict:
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


if __name__ == "__main__":
    print("Databricks Product Recommendation System - Use functions in Databricks notebook")
