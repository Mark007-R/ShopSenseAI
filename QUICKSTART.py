from product_recommendation_system import ProductRecommendationSystem
from config import (
    DATA_PATH,
    DEFAULT_METHOD,
    DEFAULT_N_RECOMMENDATIONS,
    HYBRID_WEIGHTS,
    MIN_PURCHASE_THRESHOLD,
)


def _prompt_method() -> str:
    valid_methods = ["user_cf", "item_cf", "content", "hybrid"]
    print("\nAvailable methods: user_cf, item_cf, content, hybrid")
    method_input = input(f"Choose method [{DEFAULT_METHOD}]: ").strip().lower()
    if not method_input:
        return DEFAULT_METHOD
    if method_input not in valid_methods:
        print(f"Invalid method '{method_input}'. Falling back to '{DEFAULT_METHOD}'.")
        return DEFAULT_METHOD
    return method_input


def _prompt_user_mode() -> str:
    print("\nUser type options: existing, new")
    mode = input("Are you an existing user or new user? [existing]: ").strip().lower()
    if mode in ["", "existing", "e"]:
        return "existing"
    if mode in ["new", "n"]:
        return "new"
    print("Invalid input. Falling back to 'existing'.")
    return "existing"


def _prompt_user_id(rec_system: ProductRecommendationSystem) -> str:
    sample_user = rec_system.data['user_id'].iloc[0]
    user_id_input = input(f"Enter user_id [{sample_user}]: ").strip()
    return user_id_input if user_id_input else sample_user


def _print_recommendations(result: dict) -> None:
    print("\n" + "=" * 90)
    print(
        f"Recommendations for user '{result['user_id']}' | "
        f"method='{result['method']}' | top={result['count']}"
    )
    print("=" * 90)
    if not result['recommendations']:
        print("No recommendations found.")
        return
    print(f"{'Rank':<6}{'Product ID':<15}{'Product Name':<42}{'Category':<20}{'Score':<8}")
    print("-" * 90)
    for idx, rec in enumerate(result['recommendations'], 1):
        product_name = str(rec.get('product_name', ''))[:40]
        print(
            f"{idx:<6}{rec.get('product_id', ''):<15}{product_name:<42}"
            f"{str(rec.get('category', '')):<20}{rec.get('score', 0):<8.4f}"
        )


def _prompt_new_user_items() -> list:
    print("\nEnter a few items you like (product name/category/brand/product_id).")
    print("Example: smart watch, headphones, electronics")
    raw = input("Your items (comma-separated): ").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def main():
    print("Initializing recommendation system...")
    rec_system = ProductRecommendationSystem(
        data_path=DATA_PATH,
        min_purchase_threshold=MIN_PURCHASE_THRESHOLD,
    )
    print("\nInteractive recommendation mode")
    user_mode = _prompt_user_mode()
    if user_mode == "new":
        entered_items = _prompt_new_user_items()
        result = rec_system.recommend_for_new_user(
            seed_items=entered_items,
            n_recommendations=DEFAULT_N_RECOMMENDATIONS,
        )
        print(f"\nMatched input terms: {result.get('matched_items', [])}")
        print(f"Seed product IDs used: {result.get('seed_product_ids', [])}")
    else:
        user_id = _prompt_user_id(rec_system)
        method = _prompt_method()
        request_kwargs = {}
        if method == "hybrid":
            request_kwargs["weights"] = HYBRID_WEIGHTS
        result = rec_system.get_recommendations(
            user_id=user_id,
            method=method,
            n_recommendations=DEFAULT_N_RECOMMENDATIONS,
            **request_kwargs,
        )
    _print_recommendations(result)
    save_choice = input("\nSave these recommendations to CSV? [y/N]: ").strip().lower()
    if save_choice == "y":
        if user_mode == "new":
            import pandas as pd
            output_path = "recommendations_new_user.csv"
            output_df = pd.DataFrame(result.get('recommendations', []))
            output_df.to_csv(output_path, index=False)
            print(f"Saved {len(output_df)} rows to {output_path}")
        else:
            output_df = rec_system.generate_batch_recommendations(
                user_ids=[user_id],
                method=method,
                n_recommendations=DEFAULT_N_RECOMMENDATIONS,
                output_path=f"recommendations_{user_id}.csv",
            )
            print(f"Saved {len(output_df)} rows to recommendations_{user_id}.csv")


if __name__ == "__main__":
    main()
