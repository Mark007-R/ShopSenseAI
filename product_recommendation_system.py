import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from config import (
    DATA_PATH,
    DEFAULT_N_RECOMMENDATIONS,
    DEFAULT_N_SIMILAR_USERS,
    HYBRID_WEIGHTS,
    MIN_PURCHASE_THRESHOLD,
)


class ProductRecommendationSystem:
    
    def __init__(self, data_path: str, min_purchase_threshold: int = 1):
        self.data = pd.read_csv(data_path)
        self.min_purchase_threshold = min_purchase_threshold
        self._preprocess_data()
        self._build_interactions_matrix()
        
    def _preprocess_data(self):
        print("Preprocessing data...")
        self.data['final_price_inr'].fillna(self.data['listed_price_inr'], inplace=True)
        self.data['rating'].fillna(self.data['rating'].median(), inplace=True)
        self.data['satisfaction_score'].fillna(self.data['satisfaction_score'].median(), inplace=True)
        self.data['interaction_score'] = self._calculate_interaction_score()
        self.product_engagement = self.data.groupby('product_id').size()
        print(f"Data preprocessing complete: {len(self.data)} records, {self.data['user_id'].nunique()} users, {self.data['product_id'].nunique()} products")
        
    def _calculate_interaction_score(self) -> np.ndarray:
        scores = pd.Series(0.5, index=self.data.index)
        scores[self.data['interaction_type'] == 'purchase'] = 5.0
        scores[self.data['interaction_type'] == 'add_to_cart'] = 2.0
        scores[self.data['interaction_type'] == 'wishlist'] = 1.0
        review_boost = (self.data['review_given'] == 1).astype(float) * 3.0
        scores = scores + review_boost
        engagement_factor = (
            (self.data['session_duration_sec'] / self.data['session_duration_sec'].max()) * 0.2 +
            (self.data['pages_visited'] / self.data['pages_visited'].max()) * 0.2
        )
        return scores + engagement_factor
    
    def _build_interactions_matrix(self):
        print("Building interaction matrices...")
        self.user_product_matrix = self.data.pivot_table(
            index='user_id',
            columns='product_id',
            values='interaction_score',
            aggfunc='sum',
            fill_value=0
        )
        self.purchase_matrix = self.data[self.data['interaction_type'] == 'purchase'].pivot_table(
            index='user_id',
            columns='product_id',
            values='purchase',
            aggfunc='sum',
            fill_value=0
        )
        print(f"User-Product Matrix: {self.user_product_matrix.shape}")
        print(f"Purchase Matrix: {self.purchase_matrix.shape}")
        self._build_product_features()
    
    def _build_product_features(self):
        product_summary = self.data.groupby('product_id').agg({
            'product_name': 'first',
            'category': 'first',
            'brand': 'first',
            'listed_price_inr': 'first',
            'rating': 'mean',
            'satisfaction_score': 'mean',
        }).reset_index()
        product_summary['features'] = (
            product_summary['product_name'] + ' ' +
            product_summary['category'] + ' ' +
            product_summary['brand']
        ).str.lower()
        self.product_features = product_summary.set_index('product_id')
        tfidf = TfidfVectorizer(max_features=50, stop_words='english')
        self.product_tfidf = tfidf.fit_transform(self.product_features['features'])
    
    def collaborative_filtering_user_based(
        self,
        user_id: str,
        n_recommendations: int = DEFAULT_N_RECOMMENDATIONS,
        n_similar_users: int = DEFAULT_N_SIMILAR_USERS
    ) -> List[Dict]:
        if user_id not in self.user_product_matrix.index:
            return self._get_popular_products(n_recommendations)
        
        user_idx = list(self.user_product_matrix.index).index(user_id)
        similarities = cosine_similarity(
            self.user_product_matrix.iloc[user_idx:user_idx+1],
            self.user_product_matrix
        )[0]
        similar_user_indices = np.argsort(similarities)[::-1][1:n_similar_users+1]
        user_products = set(
            self.user_product_matrix.columns[
                self.user_product_matrix.iloc[user_idx] > 0
            ]
        )
        similar_users_products = {}
        for idx in similar_user_indices:
            similar_user_id = self.user_product_matrix.index[idx]
            for product in self.user_product_matrix.columns:
                if (self.user_product_matrix.iloc[idx][product] > 0 and
                    product not in user_products):
                    similar_users_products[product] = (
                        similar_users_products.get(product, 0) +
                        similarities[idx] * self.user_product_matrix.iloc[idx][product]
                    )
        return self._format_recommendations(similar_users_products, n_recommendations)
    
    def collaborative_filtering_item_based(
        self,
        user_id: str,
        n_recommendations: int = DEFAULT_N_RECOMMENDATIONS
    ) -> List[Dict]:
        if user_id not in self.purchase_matrix.index:
            return self._get_popular_products(n_recommendations)
        
        user_idx = list(self.purchase_matrix.index).index(user_id)
        purchased_products = self.purchase_matrix.columns[
            self.purchase_matrix.iloc[user_idx] > 0
        ]
        if len(purchased_products) == 0:
            return self._get_popular_products(n_recommendations)
        
        item_similarity = cosine_similarity(self.product_tfidf)
        recommendations = {}
        for purchased_product in purchased_products:
            product_idx = list(self.product_features.index).index(purchased_product)
            similar_products = np.argsort(item_similarity[product_idx])[::-1][1:]
            for sim_idx in similar_products[:20]:
                similar_product = self.product_features.index[sim_idx]
                if similar_product not in purchased_products:
                    similarity_score = item_similarity[product_idx][sim_idx]
                    recommendations[similar_product] = (
                        recommendations.get(similar_product, 0) + similarity_score
                    )
        return self._format_recommendations(recommendations, n_recommendations)
    
    def content_based_filtering(
        self,
        user_id: str,
        n_recommendations: int = DEFAULT_N_RECOMMENDATIONS
    ) -> List[Dict]:
        if user_id not in self.data['user_id'].values:
            return self._get_popular_products(n_recommendations)
        
        user_interactions = self.data[self.data['user_id'] == user_id]
        if len(user_interactions) == 0:
            return self._get_popular_products(n_recommendations)
        
        preferred_categories = user_interactions['category'].value_counts().head(3).index.tolist()
        preferred_brands = user_interactions['brand'].value_counts().head(3).index.tolist()
        interacted_products = set(user_interactions['product_id'].unique())
        candidate_products = self.data[
            (self.data['category'].isin(preferred_categories)) |
            (self.data['brand'].isin(preferred_brands))
        ].copy()
        recommendations = {}
        for product_id in candidate_products['product_id'].unique():
            if product_id not in interacted_products:
                product_data = candidate_products[
                    candidate_products['product_id'] == product_id
                ]
                category_match = float(
                    product_data['category'].isin(preferred_categories).sum()
                ) / max(len(preferred_categories), 1)
                brand_match = float(
                    product_data['brand'].isin(preferred_brands).sum()
                ) / max(len(preferred_brands), 1)
                rating_score = (product_data['rating'].mean() / 5.0) if not pd.isna(product_data['rating'].mean()) else 0
                score = (category_match * 0.4 + brand_match * 0.3 + rating_score * 0.3)
                recommendations[product_id] = score
        return self._format_recommendations(recommendations, n_recommendations)
    
    def hybrid_recommendation(
        self,
        user_id: str,
        n_recommendations: int = DEFAULT_N_RECOMMENDATIONS,
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

    def _resolve_seed_products(self, seed_items: List[str]) -> Tuple[List[str], List[str]]:
        if not seed_items:
            return [], []
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
        return seed_products, matched_terms

    def recommend_for_new_user(
        self,
        seed_items: List[str],
        n_recommendations: int = DEFAULT_N_RECOMMENDATIONS
    ) -> Dict:
        seed_products, matched_terms = self._resolve_seed_products(seed_items)
        if not seed_products:
            fallback = self._get_popular_products(n_recommendations)
            return {
                'user_id': 'new_user',
                'method': 'cold_start_popular',
                'input_items': seed_items,
                'matched_items': [],
                'seed_product_ids': [],
                'recommendations': fallback,
                'count': len(fallback)
            }
        product_ids = list(self.product_features.index)
        item_similarity = cosine_similarity(self.product_tfidf)
        seed_set = set(seed_products)
        seed_df = self.product_features.loc[seed_products]
        category_pref = seed_df['category'].value_counts(normalize=True).to_dict()
        brand_pref = seed_df['brand'].value_counts(normalize=True).to_dict()
        similarity_scores = {}
        content_scores = {}
        for seed_product_id in seed_products:
            seed_idx = product_ids.index(seed_product_id)
            similar_indices = np.argsort(item_similarity[seed_idx])[::-1][1:]
            for sim_idx in similar_indices[:30]:
                candidate_id = product_ids[sim_idx]
                if candidate_id in seed_set:
                    continue
                similarity_scores[candidate_id] = (
                    similarity_scores.get(candidate_id, 0) + float(item_similarity[seed_idx][sim_idx])
                )
        for candidate_id in similarity_scores.keys():
            candidate = self.product_features.loc[candidate_id]
            cat_score = category_pref.get(candidate['category'], 0.0)
            brand_score = brand_pref.get(candidate['brand'], 0.0)
            content_scores[candidate_id] = (cat_score * 0.6) + (brand_score * 0.4)
        final_scores = {}
        for product_id, sim_score in similarity_scores.items():
            final_scores[product_id] = (sim_score * 0.7) + (content_scores.get(product_id, 0.0) * 0.3)
        recommendations = self._format_recommendations(final_scores, n_recommendations)
        return {
            'user_id': 'new_user',
            'method': 'cold_start_seed_items',
            'input_items': seed_items,
            'matched_items': matched_terms,
            'seed_product_ids': seed_products,
            'recommendations': recommendations,
            'count': len(recommendations)
        }
    
    def _get_popular_products(self, n: int = DEFAULT_N_RECOMMENDATIONS) -> List[Dict]:
        popular_products = self.data.groupby('product_id').agg({
            'product_name': 'first',
            'interaction_score': 'sum',
            'rating': 'mean'
        }).sort_values('interaction_score', ascending=False).head(n)
        return [
            {
                'product_id': product_id,
                'product_name': row['product_name'],
                'score': row['interaction_score'],
                'rating': round(row['rating'], 2),
                'reason': 'Popular Product'
            }
            for product_id, row in popular_products.iterrows()
        ]
    
    def _format_recommendations(
        self,
        product_scores: Dict[str, float],
        n_recommendations: int
    ) -> List[Dict]:
        if not product_scores:
            return self._get_popular_products(n_recommendations)
        
        sorted_products = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = []
        for product_id, score in sorted_products[:n_recommendations]:
            product_data = self.product_features.loc[product_id]
            recommendations.append({
                'product_id': product_id,
                'product_name': product_data['product_name'],
                'category': product_data['category'],
                'brand': product_data['brand'],
                'price': f"₹{product_data['listed_price_inr']:,.0f}",
                'rating': round(product_data['rating'], 2),
                'score': round(score, 4),
                'normalized_score': round((score / (max(product_scores.values()) + 1e-10)) * 100, 2)
            })
        return recommendations
    
    def get_recommendations(
        self,
        user_id: str,
        method: str = 'hybrid',
        n_recommendations: int = DEFAULT_N_RECOMMENDATIONS,
        **kwargs
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
            'user_id': user_id,
            'method': method,
            'recommendations': recommendations,
            'count': len(recommendations)
        }
    
    def generate_batch_recommendations(
        self,
        user_ids: List[str] = None,
        method: str = 'hybrid',
        n_recommendations: int = DEFAULT_N_RECOMMENDATIONS,
        output_path: str = None
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
                    'user_id': user_id,
                    'product_id': rec['product_id'],
                    'product_name': rec['product_name'],
                    'category': rec.get('category', ''),
                    'brand': rec.get('brand', ''),
                    'price': rec['price'],
                    'rating': rec['rating'],
                    'recommendation_score': rec['score'],
                    'normalized_score': rec['normalized_score'],
                    'method': method,
                    'rank': len(all_recommendations) % n_recommendations + 1
                })
        results_df = pd.DataFrame(all_recommendations)
        if output_path:
            results_df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        print(f"Generated {len(results_df)} recommendations")
        return results_df
    
    def evaluate_recommendations(self, test_users: List[str] = None) -> Dict:
        if test_users is None:
            test_users = self.data['user_id'].unique()[:100]
        print("Evaluating recommendations...")
        metrics = {
            'coverage': [],
            'diversity': [],
            'precision': []
        }
        for user_id in test_users:
            recs = self.hybrid_recommendation(user_id, n_recommendations=5)
            if recs:
                coverage = len(set([r['product_id'] for r in recs]))
                metrics['coverage'].append(coverage / len(self.product_features))
                categories = [r.get('category', '') for r in recs]
                diversity = len(set(categories)) / max(len(categories), 1)
                metrics['diversity'].append(diversity)
                ratings = [r['rating'] for r in recs]
                precision = np.mean(ratings) / 5.0 if ratings else 0
                metrics['precision'].append(precision)
        return {
            'avg_coverage': f"{np.mean(metrics['coverage']):.2%}",
            'avg_diversity': f"{np.mean(metrics['diversity']):.2%}",
            'avg_precision': f"{np.mean(metrics['precision']):.2%}",
            'evaluation_sample_size': len(test_users)
        }


if __name__ == "__main__":
    rec_system = ProductRecommendationSystem(
        data_path=DATA_PATH,
        min_purchase_threshold=MIN_PURCHASE_THRESHOLD
    )
    sample_user = rec_system.data['user_id'].iloc[0]
    result = rec_system.get_recommendations(sample_user, method='hybrid', n_recommendations=5)
    print(f"Recommendations for user {sample_user}:")
    for i, rec in enumerate(result['recommendations'], 1):
        print(f"{i}. {rec['product_name']} | Score: {rec['score']:.4f}")
