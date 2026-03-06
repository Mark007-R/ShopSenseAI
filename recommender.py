"""
Product Recommendation Engine
Pipeline: Dataset → Preprocessing → User-Item Matrix → Algorithm → Training → Evaluation
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# STEP 1: DATA PREPROCESSING
# ─────────────────────────────────────────────
class DataPreprocessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.product_meta = None

    def load_and_clean(self):
        df = pd.read_csv(self.filepath)

        # Drop duplicates
        df = df.drop_duplicates()

        # Fill missing ratings with 0 (unrated)
        df['rating'] = df['rating'].fillna(0)
        df['satisfaction_score'] = df['satisfaction_score'].fillna(0)

        # Encode interaction_type as numeric weight
        interaction_weights = {
            'view': 1,
            'wishlist': 2,
            'add_to_cart': 3,
            'purchase': 5
        }
        df['interaction_score'] = df['interaction_type'].map(interaction_weights)

        # Boost score for purchases with high rating
        df['engagement_score'] = (
            df['interaction_score'] * 1.0
            + df['rating'] * 0.5
            + (df['purchase'] * df['scroll_depth_pct'] / 100) * 0.3
            + (df['returned'].apply(lambda x: -1 if x == 1 else 0)) * 1.5
        )
        df['engagement_score'] = df['engagement_score'].clip(lower=0)

        # Parse date
        df['interaction_date'] = pd.to_datetime(df['interaction_date'])

        self.df = df

        # Build product metadata lookup
        self.product_meta = (
            df[['product_id', 'product_name', 'category', 'brand', 'listed_price_inr']]
            .drop_duplicates('product_id')
            .set_index('product_id')
        )

        return df

    def get_product_meta(self, product_id):
        if product_id in self.product_meta.index:
            return self.product_meta.loc[product_id].to_dict()
        return {}


# ─────────────────────────────────────────────
# STEP 2: USER-ITEM MATRIX
# ─────────────────────────────────────────────
class UserItemMatrix:
    def __init__(self, df):
        self.df = df
        self.matrix = None
        self.user_ids = None
        self.product_ids = None
        self.user_index = {}
        self.product_index = {}

    def build(self):
        # Aggregate engagement score per user-product pair
        agg = (
            self.df.groupby(['user_id', 'product_id'])['engagement_score']
            .sum()
            .reset_index()
        )

        pivot = agg.pivot(index='user_id', columns='product_id', values='engagement_score').fillna(0)

        self.matrix = pivot.values
        self.user_ids = list(pivot.index)
        self.product_ids = list(pivot.columns)
        self.user_index = {uid: i for i, uid in enumerate(self.user_ids)}
        self.product_index = {pid: i for i, pid in enumerate(self.product_ids)}

        return pivot

    def get_sparse(self):
        return csr_matrix(self.matrix)


# ─────────────────────────────────────────────
# STEP 3 & 4: RECOMMENDATION ALGORITHMS + TRAINING
# ─────────────────────────────────────────────
class CollaborativeFilteringModel:
    """SVD-based Matrix Factorization (Model-based CF)"""

    def __init__(self, n_factors=20):
        self.n_factors = n_factors
        self.U = None
        self.sigma = None
        self.Vt = None
        self.predicted_matrix = None

    def train(self, matrix):
        sparse = csr_matrix(matrix.astype(float))
        k = min(self.n_factors, min(sparse.shape) - 1)
        self.U, self.sigma, self.Vt = svds(sparse, k=k)
        # Reconstruct predicted ratings
        sigma_diag = np.diag(self.sigma)
        self.predicted_matrix = np.dot(np.dot(self.U, sigma_diag), self.Vt)
        return self

    def recommend(self, user_idx, product_ids, user_item_row, top_n=5):
        if self.predicted_matrix is None:
            return []
        scores = self.predicted_matrix[user_idx]
        # Exclude already interacted products
        interacted = np.where(user_item_row > 0)[0]
        scores[interacted] = -np.inf
        top_indices = np.argsort(scores)[::-1][:top_n]
        return [(product_ids[i], float(scores[i])) for i in top_indices if scores[i] > -np.inf]


class UserBasedCF:
    """User-Based Collaborative Filtering using cosine similarity"""

    def __init__(self):
        self.similarity_matrix = None
        self.matrix = None

    def train(self, matrix):
        self.matrix = matrix
        self.similarity_matrix = cosine_similarity(matrix)
        return self

    def recommend(self, user_idx, product_ids, top_n=5):
        if self.similarity_matrix is None:
            return []
        sim_scores = self.similarity_matrix[user_idx]
        sim_scores[user_idx] = 0  # exclude self
        top_users = np.argsort(sim_scores)[::-1][:20]
        # Weighted sum of neighbor interactions
        weighted = np.zeros(len(product_ids))
        weight_sum = 0
        for neighbor in top_users:
            w = sim_scores[neighbor]
            weighted += w * self.matrix[neighbor]
            weight_sum += w
        if weight_sum > 0:
            weighted /= weight_sum
        # Exclude already interacted
        interacted = np.where(self.matrix[user_idx] > 0)[0]
        weighted[interacted] = -np.inf
        top_indices = np.argsort(weighted)[::-1][:top_n]
        return [(product_ids[i], float(weighted[i])) for i in top_indices if weighted[i] > -np.inf]


class ItemBasedCF:
    """Item-Based Collaborative Filtering"""

    def __init__(self):
        self.item_similarity = None
        self.matrix = None

    def train(self, matrix):
        self.matrix = matrix
        self.item_similarity = cosine_similarity(matrix.T)
        return self

    def recommend(self, user_idx, product_ids, top_n=5):
        user_vec = self.matrix[user_idx]
        scores = self.item_similarity.T.dot(user_vec)
        interacted = np.where(user_vec > 0)[0]
        scores[interacted] = -np.inf
        top_indices = np.argsort(scores)[::-1][:top_n]
        return [(product_ids[i], float(scores[i])) for i in top_indices if scores[i] > -np.inf]


class ContentBasedModel:
    """Content-Based Filtering using product category/brand features"""

    def __init__(self, df, product_ids):
        self.df = df
        self.product_ids = product_ids
        self.feature_matrix = None
        self.product_features = None

    def train(self):
        meta = (
            self.df[['product_id', 'category', 'brand', 'listed_price_inr']]
            .drop_duplicates('product_id')
            .set_index('product_id')
        )
        meta = meta.reindex(self.product_ids)

        le_cat = LabelEncoder()
        le_brand = LabelEncoder()
        scaler = MinMaxScaler()

        cat_enc = le_cat.fit_transform(meta['category'].fillna('Unknown'))
        brand_enc = le_brand.fit_transform(meta['brand'].fillna('Unknown'))
        price_scaled = scaler.fit_transform(meta[['listed_price_inr']].fillna(0))

        self.feature_matrix = np.column_stack([cat_enc, brand_enc, price_scaled])
        self.item_similarity = cosine_similarity(self.feature_matrix)
        return self

    def recommend_similar(self, product_id, top_n=5):
        if product_id not in self.product_ids:
            return []
        idx = self.product_ids.index(product_id)
        scores = self.item_similarity[idx].copy()
        scores[idx] = -1
        top_indices = np.argsort(scores)[::-1][:top_n]
        return [(self.product_ids[i], float(scores[i])) for i in top_indices]


class TrendingModel:
    """Popularity-based fallback (trending products)"""

    def __init__(self, df):
        self.df = df
        self.trending = []

    def train(self):
        # Recent 90 days weighted score
        recent = self.df[self.df['interaction_date'] >= self.df['interaction_date'].max() - pd.Timedelta(days=90)]
        weights = {'view': 1, 'wishlist': 2, 'add_to_cart': 3, 'purchase': 5}
        recent = recent.copy()
        recent['w'] = recent['interaction_type'].map(weights)
        trend = recent.groupby('product_id')['w'].sum().sort_values(ascending=False)
        self.trending = list(trend.index)
        return self

    def recommend(self, exclude_ids=None, top_n=5):
        if exclude_ids is None:
            exclude_ids = []
        return [(pid, 0.0) for pid in self.trending if pid not in exclude_ids][:top_n]


# ─────────────────────────────────────────────
# STEP 5: EVALUATION
# ─────────────────────────────────────────────
class Evaluator:
    def __init__(self, matrix, user_ids, product_ids):
        self.matrix = matrix
        self.user_ids = user_ids
        self.product_ids = product_ids

    def precision_at_k(self, model, k=5, sample_size=200):
        """Precision@K using leave-one-out evaluation"""
        precisions = []
        sample_users = np.random.choice(len(self.user_ids), min(sample_size, len(self.user_ids)), replace=False)

        for uid in sample_users:
            interacted = np.where(self.matrix[uid] > 0)[0]
            if len(interacted) < 2:
                continue
            # Hold out one item
            held_out_idx = np.random.choice(interacted)
            held_out_pid = self.product_ids[held_out_idx]

            # Temporarily mask
            original = self.matrix[uid][held_out_idx]
            self.matrix[uid][held_out_idx] = 0

            try:
                if isinstance(model, CollaborativeFilteringModel):
                    recs = model.recommend(uid, self.product_ids, self.matrix[uid], top_n=k)
                elif isinstance(model, UserBasedCF):
                    recs = model.recommend(uid, self.product_ids, top_n=k)
                elif isinstance(model, ItemBasedCF):
                    recs = model.recommend(uid, self.product_ids, top_n=k)
                else:
                    recs = []

                rec_ids = [r[0] for r in recs]
                hit = 1 if held_out_pid in rec_ids else 0
                precisions.append(hit)
            except Exception:
                pass
            finally:
                self.matrix[uid][held_out_idx] = original

        return np.mean(precisions) if precisions else 0.0

    def coverage(self, all_recommendations, total_products):
        """Catalog coverage: % of products ever recommended"""
        recommended = set()
        for recs in all_recommendations:
            recommended.update([r[0] for r in recs])
        return len(recommended) / total_products if total_products > 0 else 0

    def evaluate_all(self, svd_model, user_cf, item_cf):
        results = {}
        np.random.seed(42)
        results['SVD (Matrix Factorization)'] = {
            'precision_at_5': round(self.precision_at_k(svd_model, k=5), 4),
            'description': 'Latent factor model via SVD decomposition'
        }
        results['User-Based CF'] = {
            'precision_at_5': round(self.precision_at_k(user_cf, k=5), 4),
            'description': 'Finds similar users and recommends their items'
        }
        results['Item-Based CF'] = {
            'precision_at_5': round(self.precision_at_k(item_cf, k=5), 4),
            'description': 'Recommends items similar to what the user liked'
        }
        return results


# ─────────────────────────────────────────────
# MASTER RECOMMENDER (orchestrates all models)
# ─────────────────────────────────────────────
class ProductRecommender:
    def __init__(self, filepath):
        self.filepath = filepath
        self.is_trained = False
        self._build()

    def _build(self):
        # Step 1: Preprocess
        self.preprocessor = DataPreprocessor(self.filepath)
        self.df = self.preprocessor.load_and_clean()

        # Step 2: Build matrix
        self.uim = UserItemMatrix(self.df)
        self.pivot = self.uim.build()
        self.matrix = self.uim.matrix
        self.user_ids = self.uim.user_ids
        self.product_ids = self.uim.product_ids

        # Step 3 & 4: Train models
        self.svd_model = CollaborativeFilteringModel(n_factors=20).train(self.matrix)
        self.user_cf = UserBasedCF().train(self.matrix)
        self.item_cf = ItemBasedCF().train(self.matrix)
        self.content_model = ContentBasedModel(self.df, self.product_ids).train()
        self.trending_model = TrendingModel(self.df).train()

        self.is_trained = True

    def get_recommendations(self, user_id, algo='hybrid', top_n=6):
        """Main recommendation endpoint"""
        if user_id not in self.uim.user_index:
            # Cold-start: return trending
            recs = self.trending_model.recommend(top_n=top_n)
            return self._enrich(recs, note='trending')

        uid = self.uim.user_index[user_id]
        user_row = self.matrix[uid]

        if algo == 'svd':
            recs = self.svd_model.recommend(uid, self.product_ids, user_row.copy(), top_n=top_n)
        elif algo == 'user_cf':
            recs = self.user_cf.recommend(uid, self.product_ids, top_n=top_n)
        elif algo == 'item_cf':
            recs = self.item_cf.recommend(uid, self.product_ids, top_n=top_n)
        elif algo == 'hybrid':
            # Weighted combination: SVD + Item-CF
            svd_recs = dict(self.svd_model.recommend(uid, self.product_ids, user_row.copy(), top_n=top_n * 2))
            item_recs = dict(self.item_cf.recommend(uid, self.product_ids, top_n=top_n * 2))
            all_pids = set(svd_recs) | set(item_recs)
            combined = {}
            for pid in all_pids:
                combined[pid] = 0.6 * svd_recs.get(pid, 0) + 0.4 * item_recs.get(pid, 0)
            recs = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_n]
        else:
            recs = self.trending_model.recommend(top_n=top_n)

        if not recs:
            # fallback
            interacted = [self.product_ids[i] for i in np.where(user_row > 0)[0]]
            recs = self.trending_model.recommend(exclude_ids=interacted, top_n=top_n)
            return self._enrich(recs, note='trending_fallback')

        return self._enrich(recs)

    def get_similar_products(self, product_id, top_n=5):
        return self._enrich(self.content_model.recommend_similar(product_id, top_n=top_n))

    def _enrich(self, recs, note=None):
        enriched = []
        for pid, score in recs:
            meta = self.preprocessor.get_product_meta(pid)
            enriched.append({
                'product_id': pid,
                'product_name': meta.get('product_name', pid),
                'category': meta.get('category', ''),
                'brand': meta.get('brand', ''),
                'price': meta.get('listed_price_inr', 0),
                'score': round(float(score), 4),
                'note': note
            })
        return enriched

    def get_user_history(self, user_id, top_n=5):
        user_df = self.df[self.df['user_id'] == user_id]
        if user_df.empty:
            return []
        recent = (user_df.sort_values('interaction_date', ascending=False)
                  [['product_id', 'product_name', 'category', 'interaction_type', 'interaction_date']]
                  .drop_duplicates('product_id').head(top_n))
        return recent.to_dict('records')

    def get_evaluation_results(self):
        evaluator = Evaluator(self.matrix.copy(), self.user_ids, self.product_ids)
        return evaluator.evaluate_all(self.svd_model, self.user_cf, self.item_cf)

    def get_stats(self):
        return {
            'total_users': len(self.user_ids),
            'total_products': len(self.product_ids),
            'total_interactions': len(self.df),
            'categories': list(self.df['category'].unique()),
            'interaction_breakdown': self.df['interaction_type'].value_counts().to_dict()
        }

    def get_all_users(self):
        return self.user_ids[:200]  # return first 200 for dropdown

    def get_all_products(self):
        return self.product_ids

    def get_trending(self, top_n=6):
        recs = self.trending_model.recommend(top_n=top_n)
        return self._enrich(recs)
