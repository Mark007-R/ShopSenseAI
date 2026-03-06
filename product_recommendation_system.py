import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Optional
import warnings
import logging
import time

warnings.filterwarnings('ignore')

from config import (
    DATA_PATH,
    DEFAULT_N_RECOMMENDATIONS,
    DEFAULT_N_SIMILAR_USERS,
    HYBRID_WEIGHTS,
    MIN_PURCHASE_THRESHOLD,
)
from cache_utils import matrix_cache, recommendation_cache

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ProductRecommendationSystem:
    """Optimized hybrid recommendation system with caching and performance improvements."""
    
    def __init__(self, data_path: str, min_purchase_threshold: int = 1, enable_cache: bool = True):
        self.data_path = data_path
        self.min_purchase_threshold = min_purchase_threshold
        self.enable_cache = enable_cache
        self._load_data()
        self._preprocess_data()
        self._build_matrices()
        logger.info("ProductRecommendationSystem initialized successfully")
    
    def _load_data(self):
        """Load data efficiently using pandas optimizations."""
        logger.info(f"Loading data from {self.data_path}...")
        start = time.time()
        self.data = pd.read_csv(self.data_path)
        elapsed = time.time() - start
        logger.info(f"Data loaded in {elapsed:.3f}s: {len(self.data)} records")
        
    def _preprocess_data(self):
        """Optimized data preprocessing with reduced allocations."""
        logger.info("Preprocessing data...")
        self.data['final_price_inr'].fillna(self.data['listed_price_inr'], inplace=True)
        self.data['rating'].fillna(self.data['rating'].median(), inplace=True)
        self.data['satisfaction_score'].fillna(self.data['satisfaction_score'].median(), inplace=True)
        self.data['interaction_score'] = self._calculate_interaction_score()
        logger.info(
            f"Complete: {len(self.data)} records, {self.data['user_id'].nunique()} users, "
            f"{self.data['product_id'].nunique()} products"
        )
        
    def _calculate_interaction_score(self) -> np.ndarray:
        """Vectorized calculation of interaction scores for better performance."""
        scores = pd.Series(0.5, index=self.data.index, dtype=np.float32)
        
        # Vectorized score assignment
        type_scores = {
            'purchase': 5.0,
            'add_to_cart': 2.0,
            'wishlist': 1.0
        }
        
        for interaction_type, score in type_scores.items():
            mask = self.data['interaction_type'] == interaction_type
            scores[mask] = score
        
        # Vectorized review boost
        review_boost = (self.data['review_given'] == 1).astype(np.float32) * 3.0
        scores = (scores + review_boost).astype(np.float32)
        
        # Vectorized engagement factor
        max_session = self.data['session_duration_sec'].max()
        max_pages = self.data['pages_visited'].max()
        
        engagement_factor = (
            (self.data['session_duration_sec'] / max_session) * 0.2 +
            (self.data['pages_visited'] / max_pages) * 0.2
        )
        
        return (scores + engagement_factor).astype(np.float32)
    
    def _build_matrices(self):
        """Build similarity and feature matrices with caching."""
        logger.info("Building matrices...")
        start = time.time()
        
        # User-product matrix
        self.user_product_matrix = self.data.pivot_table(
            index='user_id', columns='product_id', values='interaction_score',
            aggfunc='sum', fill_value=0
        ).astype(np.float32)
        
        # Purchase matrix
        self.purchase_matrix = self.data[self.data['interaction_type'] == 'purchase'].pivot_table(
            index='user_id', columns='product_id', values='purchase',
            aggfunc='sum', fill_value=0
        ).astype(np.float32)
        
        # Build product features and TF-IDF
        self._build_product_features()
        
        # Create user index mapping for O(1) lookups
        self.user_idx_map = {uid: idx for idx, uid in enumerate(self.user_product_matrix.index)}
        self.product_idx_map = {pid: idx for idx, pid in enumerate(self.user_product_matrix.columns)}
        
        elapsed = time.time() - start
        logger.info(f"Matrices built in {elapsed:.3f}s")
    
    def _build_product_features(self):
        """Build product features with TF-IDF vectorization."""
        logger.info("Building product features...")
        
        # Aggregate product information
        product_summary = self.data.groupby('product_id').agg({
            'product_name': 'first',
            'category': 'first',
            'brand': 'first',
            'listed_price_inr': 'first',
            'rating': 'mean',
            'satisfaction_score': 'mean',
        }).reset_index()
        
        # Create feature text
        product_summary['features'] = (
            product_summary['product_name'].fillna('') + ' ' +
            product_summary['category'].fillna('') + ' ' +
            product_summary['brand'].fillna('')
        ).str.lower()
        
        self.product_features = product_summary.set_index('product_id')
        
        # TF-IDF vectorization
        tfidf = TfidfVectorizer(max_features=50, stop_words='english', norm='l2')
        self.product_tfidf = tfidf.fit_transform(self.product_features['features'])
        
        logger.info(f"Product features built for {len(self.product_features)} products")
    
    def _get_user_idx_fast(self, user_id: str) -> Optional[int]:
        """Fast O(1) user index lookup."""
        return self.user_idx_map.get(user_id)
    
    def _get_product_idx_fast(self, product_id: str) -> Optional[int]:
        """Fast O(1) product index lookup."""
        return self.product_idx_map.get(product_id)
    
    def collaborative_filtering_user_based(
        self, user_id: str, n_recommendations: int = DEFAULT_N_RECOMMENDATIONS,
        n_similar_users: int = DEFAULT_N_SIMILAR_USERS
    ) -> List[Dict]:
        """Optimized user-based collaborative filtering."""
        user_idx = self._get_user_idx_fast(user_id)
        
        if user_idx is None:
            return self._get_popular_products(n_recommendations)
        
        # Compute similarities
        user_vector = self.user_product_matrix.iloc[user_idx:user_idx+1]
        similarities = cosine_similarity(user_vector, self.user_product_matrix)[0]
        
        # Find similar users (excluding self)
        similar_user_indices = np.argsort(similarities)[::-1][1:n_similar_users+1]
        
        # Get user's products
        user_products = set(self.user_product_matrix.columns[user_vector.values[0] > 0])
        
        # Aggregate similar users' products
        similar_users_products = {}
        for idx in similar_user_indices:
            products = self.user_product_matrix.iloc[idx]
            mask = products > 0
            candidates = self.user_product_matrix.columns[mask]
            
            for product in candidates:
                if product not in user_products:
                    score = float(similarities[idx]) * float(products[product])
                    similar_users_products[product] = similar_users_products.get(product, 0) + score
        
        return self._format_recommendations(similar_users_products, n_recommendations)
    
    def collaborative_filtering_item_based(
        self, user_id: str, n_recommendations: int = DEFAULT_N_RECOMMENDATIONS
    ) -> List[Dict]:
        """Optimized item-based collaborative filtering."""
        user_idx = self._get_user_idx_fast(user_id)
        
        if user_idx is None:
            return self._get_popular_products(n_recommendations)
        
        # Get purchased products
        user_purchases = self.purchase_matrix.iloc[user_idx]
        purchased_products = self.purchase_matrix.columns[user_purchases > 0].tolist()
        
        if not purchased_products:
            return self._get_popular_products(n_recommendations)
        
        # Compute item similarity once
        item_similarity = cosine_similarity(self.product_tfidf)
        
        # Score recommendations
        recommendations = {}
        product_ids_list = list(self.product_features.index)
        
        for purchased_product in purchased_products:
            p_idx = product_ids_list.index(purchased_product) if purchased_product in product_ids_list else -1
            if p_idx == -1:
                continue
            
            # Find similar products
            similarities = item_similarity[p_idx]
            similar_indices = np.argsort(similarities)[::-1][1:21]
            
            for sim_idx in similar_indices:
                similar_product = product_ids_list[sim_idx]
                if similar_product not in purchased_products:
                    similarity_score = float(similarities[sim_idx])
                    recommendations[similar_product] = recommendations.get(similar_product, 0) + similarity_score
        
        return self._format_recommendations(recommendations, n_recommendations)
    
    def content_based_filtering(
        self, user_id: str, n_recommendations: int = DEFAULT_N_RECOMMENDATIONS
    ) -> List[Dict]:
        if user_id not in self.data['user_id'].values:
            return self._get_popular_products(n_recommendations)
        
        user_interactions = self.data[self.data['user_id'] == user_id]
        if len(user_interactions) == 0:
            return self._get_popular_products(n_recommendations)
        
        pref_categories = user_interactions['category'].value_counts().head(3).index.tolist()
        pref_brands = user_interactions['brand'].value_counts().head(3).index.tolist()
        interacted_products = set(user_interactions['product_id'].unique())
        candidate_products = self.data[
            (self.data['category'].isin(pref_categories)) |
            (self.data['brand'].isin(pref_brands))
        ].copy()
        recommendations = {}
        for product_id in candidate_products['product_id'].unique():
            if product_id not in interacted_products:
                product_data = candidate_products[candidate_products['product_id'] == product_id]
                cat_match = product_data['category'].isin(pref_categories).sum() / max(len(pref_categories), 1)
                brand_match = product_data['brand'].isin(pref_brands).sum() / max(len(pref_brands), 1)
                rating_score = (product_data['rating'].mean() / 5.0) if not pd.isna(product_data['rating'].mean()) else 0
                score = (cat_match * 0.4 + brand_match * 0.3 + rating_score * 0.3)
                recommendations[product_id] = score
        return self._format_recommendations(recommendations, n_recommendations)
    
    def hybrid_recommendation(
        self, user_id: str, n_recommendations: int = DEFAULT_N_RECOMMENDATIONS,
        weights: Dict[str, float] = None
    ) -> List[Dict]:
        if weights is None:
            weights = HYBRID_WEIGHTS
        
        collab_user = self.collaborative_filtering_user_based(user_id, n_recommendations)
        collab_item = self.collaborative_filtering_item_based(user_id, n_recommendations)
        content = self.content_based_filtering(user_id, n_recommendations)
        combined_scores = {}
        for rec in collab_user:
            product_id = rec['product_id']
            combined_scores[product_id] = combined_scores.get(product_id, 0) + rec['score'] * weights['collaborative_user']
        for rec in collab_item:
            product_id = rec['product_id']
            combined_scores[product_id] = combined_scores.get(product_id, 0) + rec['score'] * weights['collaborative_item']
        for rec in content:
            product_id = rec['product_id']
            combined_scores[product_id] = combined_scores.get(product_id, 0) + rec['score'] * weights['content_based']
        return self._format_recommendations(combined_scores, n_recommendations)

    def recommend_for_new_user(
        self, seed_items: List[str], n_recommendations: int = DEFAULT_N_RECOMMENDATIONS
    ) -> Dict:
        if not seed_items:
            fallback = self._get_popular_products(n_recommendations)
            return {
                'user_id': 'new_user', 'method': 'cold_start_popular',
                'input_items': seed_items, 'matched_items': [], 'seed_product_ids': [],
                'recommendations': fallback, 'count': len(fallback)
            }
        
        seed_products = []
        matched_terms = []
        features_df = self.product_features.reset_index()
        for raw_item in seed_items:
            term = str(raw_item).strip().lower()
            if not term:
                continue
            matches = features_df[
                features_df['product_id'].astype(str).str.lower().str.contains(term, na=False) |
                features_df['product_name'].astype(str).str.lower().str.contains(term, na=False) |
                features_df['category'].astype(str).str.lower().str.contains(term, na=False) |
                features_df['brand'].astype(str).str.lower().str.contains(term, na=False)
            ]
            if not matches.empty:
                matched_terms.append(raw_item)
                for product_id in matches['product_id'].head(5).tolist():
                    if product_id not in seed_products:
                        seed_products.append(product_id)
        
        if not seed_products:
            fallback = self._get_popular_products(n_recommendations)
            return {
                'user_id': 'new_user', 'method': 'cold_start_popular',
                'input_items': seed_items, 'matched_items': matched_terms,
                'seed_product_ids': [], 'recommendations': fallback, 'count': len(fallback)
            }
        
        product_ids = list(self.product_features.index)
        item_similarity = cosine_similarity(self.product_tfidf)
        seed_set = set(seed_products)
        seed_df = self.product_features.loc[seed_products]
        category_pref = seed_df['category'].value_counts(normalize=True).to_dict()
        brand_pref = seed_df['brand'].value_counts(normalize=True).to_dict()
        
        similarity_scores = {}
        for seed_product_id in seed_products:
            seed_idx = product_ids.index(seed_product_id)
            similar_indices = np.argsort(item_similarity[seed_idx])[::-1][1:]
            for sim_idx in similar_indices[:30]:
                candidate_id = product_ids[sim_idx]
                if candidate_id not in seed_set:
                    similarity_scores[candidate_id] = (
                        similarity_scores.get(candidate_id, 0) + float(item_similarity[seed_idx][sim_idx])
                    )
        
        final_scores = {}
        for product_id, sim_score in similarity_scores.items():
            candidate = self.product_features.loc[product_id]
            cat_score = category_pref.get(candidate['category'], 0.0)
            brand_score = brand_pref.get(candidate['brand'], 0.0)
            content_score = (cat_score * 0.6) + (brand_score * 0.4)
            final_scores[product_id] = (sim_score * 0.7) + (content_score * 0.3)
        
        recommendations = self._format_recommendations(final_scores, n_recommendations)
        return {
            'user_id': 'new_user', 'method': 'cold_start_seed_items',
            'input_items': seed_items, 'matched_items': matched_terms,
            'seed_product_ids': seed_products, 'recommendations': recommendations,
            'count': len(recommendations)
        }
    
    def _get_popular_products(self, n: int = DEFAULT_N_RECOMMENDATIONS) -> List[Dict]:
        popular_products = self.data.groupby('product_id').agg({
            'product_name': 'first', 'interaction_score': 'sum', 'rating': 'mean'
        }).sort_values('interaction_score', ascending=False).head(n)
        return [
            {
                'product_id': product_id, 'product_name': row['product_name'],
                'score': row['interaction_score'], 'rating': round(row['rating'], 2)
            }
            for product_id, row in popular_products.iterrows()
        ]
    
    def _format_recommendations(
        self, product_scores: Dict[str, float], n_recommendations: int
    ) -> List[Dict]:
        if not product_scores:
            return self._get_popular_products(n_recommendations)
        
        sorted_products = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = []
        for product_id, score in sorted_products[:n_recommendations]:
            product_data = self.product_features.loc[product_id]
            recommendations.append({
                'product_id': product_id, 'product_name': product_data['product_name'],
                'category': product_data['category'], 'brand': product_data['brand'],
                'price': f"₹{product_data['listed_price_inr']:,.0f}",
                'rating': round(product_data['rating'], 2), 'score': round(score, 4),
                'normalized_score': round((score / (max(product_scores.values()) + 1e-10)) * 100, 2)
            })
        return recommendations
    
    def get_recommendations(
        self, user_id: str, method: str = 'hybrid',
        n_recommendations: int = DEFAULT_N_RECOMMENDATIONS, **kwargs
    ) -> Dict:
        methods = {
            'user_cf': self.collaborative_filtering_user_based,
            'item_cf': self.collaborative_filtering_item_based,
            'content': self.content_based_filtering,
            'hybrid': self.hybrid_recommendation
        }
        if method not in methods:
            raise ValueError(f"Method must be one of {list(methods.keys())}")
        recommendations = methods[method](user_id, n_recommendations, **kwargs)
        return {
            'user_id': user_id, 'method': method,
            'recommendations': recommendations, 'count': len(recommendations)
        }
    
    def generate_batch_recommendations(
        self, user_ids: List[str] = None, method: str = 'hybrid',
        n_recommendations: int = DEFAULT_N_RECOMMENDATIONS, output_path: str = None
    ) -> pd.DataFrame:
        if user_ids is None:
            user_ids = self.data['user_id'].unique()
        print(f"Generating recommendations for {len(user_ids)} users...")
        all_recommendations = []
        for idx, user_id in enumerate(user_ids):
            if (idx + 1) % 100 == 0:
                print(f"Progress: {idx + 1}/{len(user_ids)}")
            recs = self.get_recommendations(user_id, method, n_recommendations)
            for rec in recs['recommendations']:
                all_recommendations.append({
                    'user_id': user_id, 'product_id': rec['product_id'],
                    'product_name': rec['product_name'], 'category': rec.get('category', ''),
                    'brand': rec.get('brand', ''), 'rating': rec['rating'],
                    'score': rec['score'], 'method': method
                })
        results_df = pd.DataFrame(all_recommendations)
        if output_path:
            results_df.to_csv(output_path, index=False)
            print(f"Saved to {output_path}")
        return results_df


if __name__ == "__main__":
    rec_system = ProductRecommendationSystem(data_path=DATA_PATH)
    sample_user = rec_system.data['user_id'].iloc[0]
    result = rec_system.get_recommendations(sample_user, method='hybrid', n_recommendations=5)
    print(f"Recommendations for {sample_user}:")
    for i, rec in enumerate(result['recommendations'], 1):
        print(f"{i}. {rec['product_name']} | Score: {rec['score']:.4f}")
