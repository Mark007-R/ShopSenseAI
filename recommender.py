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
        scores = self.predicted_matrix[user_idx].copy()   # copy — never mutate predicted_matrix
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

    def recommend(self, user_idx, product_ids, top_n=5, masked_row=None):
        if self.similarity_matrix is None:
            return []
        sim_scores = self.similarity_matrix[user_idx].copy()
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
        # Use masked_row if provided (evaluation), else fall back to trained row
        exclude_vec = masked_row if masked_row is not None else self.matrix[user_idx]
        interacted = np.where(exclude_vec > 0)[0]
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

    def recommend(self, user_idx, product_ids, top_n=5, masked_row=None):
        user_vec = masked_row if masked_row is not None else self.matrix[user_idx]
        scores = self.item_similarity.T.dot(user_vec).copy()
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
# STEP 5: EVALUATION  (Temporal Train / Test Split)
# ─────────────────────────────────────────────
class Evaluator:
    """
    Proper offline evaluation using an 80/20 temporal split.

    Why temporal split instead of leave-one-out (LOO)?
    ────────────────────────────────────────────────────
    LOO on a pre-trained similarity/SVD matrix suffers from look-ahead bias:
    the model was trained on the FULL matrix including the held-out item, so
    neighbours were partly chosen *because* they share that item — making it
    trivially easy to predict back (User-CF inflates to ~92%, meaningless).

    A temporal split avoids this completely:
      • Models are retrained on the TRAIN portion only (interactions before cutoff)
      • Ground-truth is items the user interacted with AFTER the cutoff
        that were NOT already in their training history
      • Precision@K = fraction of users for whom ≥1 recommended item
        appears in their future ground-truth set
    """

    def __init__(self, df, product_ids):
        self.df = df
        self.product_ids = product_ids
        self._build_split()

    def _build_split(self):
        df = self.df.copy()
        df['interaction_date'] = pd.to_datetime(df['interaction_date'])
        cutoff = pd.Timestamp(df['interaction_date'].quantile(0.8))
        self.cutoff = cutoff

        train_df = df[df['interaction_date'] <= cutoff]
        test_df  = df[df['interaction_date'] > cutoff]

        # Build train matrix
        agg = train_df.groupby(['user_id', 'product_id'])['engagement_score'].sum().reset_index()
        pivot = agg.pivot(index='user_id', columns='product_id',
                          values='engagement_score').fillna(0)
        pivot = pivot.reindex(columns=self.product_ids, fill_value=0)

        self.train_matrix   = pivot.values
        self.train_user_ids = list(pivot.index)
        self.train_user_idx = {u: i for i, u in enumerate(self.train_user_ids)}

        # Build test ground truth: unseen items per user after cutoff
        agg_test = test_df.groupby(['user_id', 'product_id'])['engagement_score'].sum().reset_index()
        self.test_gt = {}
        for _, row in agg_test.iterrows():
            uid, pid = row['user_id'], row['product_id']
            if uid not in self.train_user_idx or pid not in self.product_ids:
                continue
            uidx = self.train_user_idx[uid]
            pidx = self.product_ids.index(pid)
            if self.train_matrix[uidx][pidx] == 0:          # truly unseen during training
                self.test_gt.setdefault(uid, set()).add(pid)

    def _retrain_and_score(self, ModelClass, model_kwargs, k, sample_size, seed):
        """Retrain a fresh model on train-only data, then evaluate."""
        model = ModelClass(**model_kwargs).train(self.train_matrix)
        np.random.seed(seed)
        users = list(self.test_gt.keys())
        np.random.shuffle(users)
        users = users[:sample_size]

        hits = []
        for uid in users:
            uidx = self.train_user_idx[uid]
            train_row = self.train_matrix[uidx]
            ground_truth = self.test_gt[uid]

            if isinstance(model, CollaborativeFilteringModel):
                recs = model.recommend(uidx, self.product_ids, train_row.copy(), top_n=k)
            elif isinstance(model, UserBasedCF):
                recs = model.recommend(uidx, self.product_ids, top_n=k, masked_row=train_row)
            elif isinstance(model, ItemBasedCF):
                recs = model.recommend(uidx, self.product_ids, top_n=k, masked_row=train_row)
            else:
                recs = []

            rec_pids = {r[0] for r in recs}
            hits.append(1 if rec_pids & ground_truth else 0)

        return round(np.mean(hits), 4) if hits else 0.0

    def evaluate_all(self, k=5, sample_size=300, seed=42):
        results = {}
        common = dict(k=k, sample_size=sample_size, seed=seed)

        results['SVD (Matrix Factorization)'] = {
            'precision_at_5': self._retrain_and_score(
                CollaborativeFilteringModel, {'n_factors': 20}, **common),
            'description': 'Latent factor model via SVD decomposition'
        }
        results['User-Based CF'] = {
            'precision_at_5': self._retrain_and_score(
                UserBasedCF, {}, **common),
            'description': 'Finds similar users and recommends their items'
        }
        results['Item-Based CF'] = {
            'precision_at_5': self._retrain_and_score(
                ItemBasedCF, {}, **common),
            'description': 'Recommends items similar to what the user liked'
        }
        results['_meta'] = {
            'method': 'Temporal 80/20 train-test split',
            'cutoff': str(self.cutoff.date()),
            'test_users': len(self.test_gt),
            'sample_size': sample_size,
            'note': 'Models retrained on pre-cutoff data; ground truth = unseen post-cutoff interactions'
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
        evaluator = Evaluator(self.df, self.product_ids)
        return evaluator.evaluate_all(k=5, sample_size=300, seed=42)

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

    def get_guest_recommendations(self, product_list, interaction_types=None, top_n=6):
        """
        Cold-start: recommend for a brand-new user given a list of product_ids
        they indicate interest in (e.g. from a preference quiz / manual input).

        Strategy — weighted Item-Based CF:
          1. Build a pseudo user-vector from the supplied products
             (weight by interaction type if provided, else uniform 3.0)
          2. Score all other products via item-similarity dot-product
          3. Exclude the seed products from results
          4. Return top_n enriched recommendations + explain which seed drove each
        """
        if interaction_types is None:
            interaction_types = {}

        weight_map = {'view': 1.0, 'wishlist': 2.0, 'add_to_cart': 3.0, 'purchase': 5.0}

        # Build pseudo vector (length = number of catalogue products)
        pseudo_vec = np.zeros(len(self.product_ids))
        valid_seeds = []
        for pid in product_list:
            if pid in self.uim.product_index:
                idx = self.uim.product_index[pid]
                w = weight_map.get(interaction_types.get(pid, 'add_to_cart'), 3.0)
                pseudo_vec[idx] = w
                valid_seeds.append(pid)

        if not valid_seeds:
            # Nothing matched — fall back to trending
            recs = self.trending_model.recommend(top_n=top_n)
            return {
                'recommendations': self._enrich(recs, note='trending_fallback'),
                'valid_seeds': [],
                'method': 'trending_fallback'
            }

        # Item-CF scores: each candidate gets sum of similarity × seed weight
        item_sim = self.item_cf.item_similarity          # shape (n_items, n_items)
        scores = item_sim.T.dot(pseudo_vec)              # (n_items,)

        # Mask seed products out
        for pid in valid_seeds:
            idx = self.uim.product_index[pid]
            scores[idx] = -np.inf

        top_indices = np.argsort(scores)[::-1][:top_n]
        raw_recs = [(self.product_ids[i], float(scores[i])) for i in top_indices if scores[i] > -np.inf]

        # Explain: for each recommendation, find which seed product drove it most
        def top_driver(rec_idx):
            contribs = {
                pid: item_sim[self.uim.product_index[pid]][rec_idx] * pseudo_vec[self.uim.product_index[pid]]
                for pid in valid_seeds
            }
            best = max(contribs, key=contribs.get)
            return best, contribs[best]

        enriched = []
        for pid, score in raw_recs:
            meta = self.preprocessor.get_product_meta(pid)
            rec_idx = self.uim.product_index[pid]
            driver_pid, driver_score = top_driver(rec_idx)
            driver_meta = self.preprocessor.get_product_meta(driver_pid)
            enriched.append({
                'product_id': pid,
                'product_name': meta.get('product_name', pid),
                'category': meta.get('category', ''),
                'brand': meta.get('brand', ''),
                'price': meta.get('listed_price_inr', 0),
                'score': round(score, 4),
                'because_of': driver_meta.get('product_name', driver_pid),
                'because_of_id': driver_pid,
                'note': 'guest_item_cf'
            })

        return {
            'recommendations': enriched,
            'valid_seeds': [
                {**self.preprocessor.get_product_meta(p), 'product_id': p,
                 'interaction': interaction_types.get(p, 'add_to_cart')}
                for p in valid_seeds
            ],
            'method': 'item_cf_cold_start'
        }

    def get_product_catalogue(self):
        """Return full product list with metadata for the guest picker UI"""
        result = []
        for pid in self.product_ids:
            meta = self.preprocessor.get_product_meta(pid)
            result.append({
                'product_id': pid,
                'product_name': meta.get('product_name', pid),
                'category': meta.get('category', ''),
                'brand': meta.get('brand', ''),
                'price': meta.get('listed_price_inr', 0),
            })
        return sorted(result, key=lambda x: x['category'])