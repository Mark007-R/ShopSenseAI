from databricks_recommendation_system import DatabricksRecommendationSystem
import config

def main():
    print("=" * 60)
    print("DATABRICKS PRODUCT RECOMMENDATION SYSTEM")
    print("=" * 60)
    
    rec_system = DatabricksRecommendationSystem()
    
    print("\nStep 1: Loading data...")
    df = rec_system.load_data(config.DATA_PATH)
    df.show(5)
    
    print("\nStep 2: Preprocessing interactions...")
    interactions_df = rec_system.preprocess_interactions(df)
    interactions_df.show(5)
    
    print("\nStep 3: Building ALS model...")
    model = rec_system.build_als_model(
        interactions_df,
        rank=config.ALS_RANK,
        max_iter=config.ALS_MAX_ITER,
        reg_param=config.ALS_REG_PARAM,
        alpha=config.ALS_ALPHA
    )
    
    print("\nStep 4: Getting dataset statistics...")
    stats = rec_system.get_stats(interactions_df)
    print(f"Total Records: {stats['total_records']}")
    print(f"Unique Users: {stats['unique_users']}")
    print(f"Unique Products: {stats['unique_products']}")
    print(f"Average Interaction Score: {stats['avg_interaction_score']:.2f}")
    
    print("\nStep 5: Generating recommendations for sample users...")
    sample_users = interactions_df.select('user_id').distinct().limit(5).rdd.flatMap(lambda x: x).collect()
    recommendations = rec_system.get_user_recommendations(sample_users, config.DEFAULT_N_RECOMMENDATIONS)
    recommendations.show(truncate=False)
    
    print("\nStep 6: Generating batch recommendations...")
    batch_recs = rec_system.generate_batch_recommendations(
        interactions_df,
        num_recommendations=config.DEFAULT_N_RECOMMENDATIONS,
        output_path=config.OUTPUT_PATH
    )
    batch_recs.show(10)
    
    print("\n" + "=" * 60)
    print("RECOMMENDATION SYSTEM COMPLETE")
    print(f"Batch recommendations saved to: {config.OUTPUT_PATH}")
    print("=" * 60)
    
    return rec_system, batch_recs

if __name__ == "__main__":
    rec_system, recommendations = main()
    print("\nUse 'rec_system' for interactive queries")
    print("Use 'recommendations' to view batch results")
