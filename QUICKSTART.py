from product_recommendation_system import ProductRecommendationSystem
from config import DATA_PATH, DEFAULT_METHOD, DEFAULT_N_RECOMMENDATIONS, HYBRID_WEIGHTS, MIN_PURCHASE_THRESHOLD


def main():
    print("Initializing recommendation system...")
    rec_system = ProductRecommendationSystem(data_path=DATA_PATH, min_purchase_threshold=MIN_PURCHASE_THRESHOLD)
    
    print("\nUser type options: existing, new")
    user_mode = input("Are you existing or new user? [existing]: ").strip().lower()
    user_mode = "new" if user_mode in ["new", "n"] else "existing"
    
    if user_mode == "new":
        print("\nEnter items you like (name/category/brand). Example: smart watch, headphones")
        items_input = input("Items (comma-separated): ").strip()
        seed_items = [item.strip() for item in items_input.split(",") if item.strip()] if items_input else []
        result = rec_system.recommend_for_new_user(seed_items=seed_items, n_recommendations=DEFAULT_N_RECOMMENDATIONS)
        print(f"\nMatched: {result.get('matched_items', [])}")
    else:
        sample_user = rec_system.data['user_id'].iloc[0]
        user_id = input(f"Enter user_id [{sample_user}]: ").strip() or sample_user
        
        print("\nMethods: user_cf, item_cf, content, hybrid")
        method = input(f"Choose method [{DEFAULT_METHOD}]: ").strip().lower() or DEFAULT_METHOD
        if method not in ["user_cf", "item_cf", "content", "hybrid"]:
            print(f"Invalid. Using {DEFAULT_METHOD}")
            method = DEFAULT_METHOD
        
        kwargs = {"weights": HYBRID_WEIGHTS} if method == "hybrid" else {}
        result = rec_system.get_recommendations(user_id=user_id, method=method, 
                                               n_recommendations=DEFAULT_N_RECOMMENDATIONS, **kwargs)
    
    print("\n" + "=" * 90)
    print(f"Recommendations for '{result['user_id']}' | method='{result['method']}' | count={result['count']}")
    print("=" * 90)
    if result['recommendations']:
        print(f"{'Rank':<6}{'Product ID':<15}{'Name':<42}{'Category':<20}{'Score':<8}")
        print("-" * 90)
        for idx, rec in enumerate(result['recommendations'], 1):
            name = str(rec.get('product_name', ''))[:40]
            print(f"{idx:<6}{rec.get('product_id', ''):<15}{name:<42}"
                  f"{str(rec.get('category', '')):<20}{rec.get('score', 0):<8.4f}")
    else:
        print("No recommendations found.")
    
    save_choice = input("\nSave to CSV? [y/N]: ").strip().lower()
    if save_choice == "y":
        import pandas as pd
        if user_mode == "new":
            output_df = pd.DataFrame(result.get('recommendations', []))
            output_df.to_csv("recommendations_new_user.csv", index=False)
            print("Saved to recommendations_new_user.csv")
        else:
            output_df = rec_system.generate_batch_recommendations(user_ids=[result['user_id']], 
                                                                  method=result['method'],
                                                                  n_recommendations=DEFAULT_N_RECOMMENDATIONS)
            print(f"Saved to batch_recommendations.csv")


if __name__ == "__main__":
    main()
