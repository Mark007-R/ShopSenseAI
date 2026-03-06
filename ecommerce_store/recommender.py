import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict
import random

class EcommerceRecommender:
    
    def __init__(self, products_path: str):
        self.products = pd.read_csv(products_path)
        self._prepare_features()
        self._initialize_user_history()
    
    def _prepare_features(self):
        self.products['text_features'] = (
            self.products['name'] + ' ' +
            self.products['category'] + ' ' +
            self.products['subcategory'] + ' ' +
            self.products['brand'] + ' ' +
            self.products['features'].fillna('')
        )
        
        self.tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.product_features = self.tfidf.fit_transform(self.products['text_features'])
        self.similarity_matrix = cosine_similarity(self.product_features)
    
    def _initialize_user_history(self):
        self.user_interactions = {}
        self.user_cart = {}
        self.user_wishlist = {}
    
    def get_product_by_id(self, product_id: str) -> Dict:
        product = self.products[self.products['product_id'] == product_id]
        if len(product) > 0:
            return product.iloc[0].to_dict()
        return None
    
    def get_all_products(self, category: str = None, sort_by: str = 'rating') -> pd.DataFrame:
        df = self.products.copy()
        
        if category and category != 'All':
            df = df[df['category'] == category]
        
        if sort_by == 'price_low':
            df = df.sort_values('price')
        elif sort_by == 'price_high':
            df = df.sort_values('price', ascending=False)
        elif sort_by == 'rating':
            df = df.sort_values('rating', ascending=False)
        elif sort_by == 'popular':
            df = df.sort_values('reviews_count', ascending=False)
        
        return df
    
    def search_products(self, query: str) -> pd.DataFrame:
        query_lower = query.lower()
        matches = self.products[
            self.products['name'].str.lower().str.contains(query_lower) |
            self.products['category'].str.lower().str.contains(query_lower) |
            self.products['brand'].str.lower().str.contains(query_lower) |
            self.products['description'].str.lower().str.contains(query_lower)
        ]
        return matches
    
    def get_similar_products(self, product_id: str, n: int = 6) -> List[Dict]:
        try:
            idx = self.products[self.products['product_id'] == product_id].index[0]
            sim_scores = list(enumerate(self.similarity_matrix[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:n+1]
            
            product_indices = [i[0] for i in sim_scores]
            similar_products = self.products.iloc[product_indices].copy()
            similar_products['similarity_score'] = [score[1] for score in sim_scores]
            
            return similar_products.to_dict('records')
        except:
            return []
    
    def get_personalized_recommendations(self, user_id: str, n: int = 10) -> List[Dict]:
        if user_id not in self.user_interactions or not self.user_interactions[user_id]:
            return self._get_trending_products(n)
        
        user_products = self.user_interactions[user_id]
        all_similar = []
        
        for product_id in user_products[-5:]:
            similar = self.get_similar_products(product_id, n=3)
            all_similar.extend(similar)
        
        similar_df = pd.DataFrame(all_similar)
        if len(similar_df) > 0:
            similar_df = similar_df.drop_duplicates('product_id')
            similar_df = similar_df[~similar_df['product_id'].isin(user_products)]
            similar_df = similar_df.sort_values('similarity_score', ascending=False)
            return similar_df.head(n).to_dict('records')
        
        return self._get_trending_products(n)
    
    def _get_trending_products(self, n: int = 10) -> List[Dict]:
        trending = self.products.copy()
        trending['popularity_score'] = (
            trending['rating'] * 0.5 + 
            (trending['reviews_count'] / trending['reviews_count'].max()) * 5 * 0.5
        )
        trending = trending.sort_values('popularity_score', ascending=False)
        return trending.head(n).to_dict('records')
    
    def get_frequently_bought_together(self, product_id: str, n: int = 3) -> List[Dict]:
        product = self.get_product_by_id(product_id)
        if not product:
            return []
        
        category = product['category']
        subcategory = product['subcategory']
        
        related = self.products[
            ((self.products['category'] == category) | 
             (self.products['subcategory'] == subcategory)) &
            (self.products['product_id'] != product_id)
        ].copy()
        
        if len(related) > 0:
            related['score'] = related['rating'] + (related['reviews_count'] / 1000)
            related = related.sort_values('score', ascending=False)
            return related.head(n).to_dict('records')
        
        return []
    
    def add_to_history(self, user_id: str, product_id: str):
        if user_id not in self.user_interactions:
            self.user_interactions[user_id] = []
        if product_id not in self.user_interactions[user_id]:
            self.user_interactions[user_id].append(product_id)
    
    def add_to_cart(self, user_id: str, product_id: str, quantity: int = 1):
        if user_id not in self.user_cart:
            self.user_cart[user_id] = {}
        self.user_cart[user_id][product_id] = self.user_cart[user_id].get(product_id, 0) + quantity
        self.add_to_history(user_id, product_id)
    
    def add_to_wishlist(self, user_id: str, product_id: str):
        if user_id not in self.user_wishlist:
            self.user_wishlist[user_id] = []
        if product_id not in self.user_wishlist[user_id]:
            self.user_wishlist[user_id].append(product_id)
        self.add_to_history(user_id, product_id)
    
    def get_cart(self, user_id: str) -> List[Dict]:
        if user_id not in self.user_cart:
            return []
        
        cart_items = []
        for product_id, quantity in self.user_cart[user_id].items():
            product = self.get_product_by_id(product_id)
            if product:
                product['quantity'] = quantity
                product['subtotal'] = product['price'] * quantity
                cart_items.append(product)
        return cart_items
    
    def get_wishlist(self, user_id: str) -> List[Dict]:
        if user_id not in self.user_wishlist:
            return []
        
        wishlist_items = []
        for product_id in self.user_wishlist[user_id]:
            product = self.get_product_by_id(product_id)
            if product:
                wishlist_items.append(product)
        return wishlist_items
    
    def remove_from_cart(self, user_id: str, product_id: str):
        if user_id in self.user_cart and product_id in self.user_cart[user_id]:
            del self.user_cart[user_id][product_id]
    
    def remove_from_wishlist(self, user_id: str, product_id: str):
        if user_id in self.user_wishlist and product_id in self.user_wishlist[user_id]:
            self.user_wishlist[user_id].remove(product_id)
    
    def get_categories(self) -> List[str]:
        return ['All'] + sorted(self.products['category'].unique().tolist())
    
    def get_cart_total(self, user_id: str) -> float:
        cart_items = self.get_cart(user_id)
        return sum(item['subtotal'] for item in cart_items)
    
    def get_cart_count(self, user_id: str) -> int:
        if user_id not in self.user_cart:
            return 0
        return sum(self.user_cart[user_id].values())
