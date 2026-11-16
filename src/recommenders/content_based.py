"""
Content-Based Recommendation Engine
Uses TF-IDF feature extraction & Cosine Similarity on product metadata (title, category, tags, description).
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.recommenders.base import BaseRecommender

class ContentBasedRecommender(BaseRecommender):
    def __init__(self, max_features: int = 5000):
        super().__init__(name="ContentBased_TFIDF")
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=max_features,
            sublinear_tf=True
        )
        self.products_df: Optional[pd.DataFrame] = None
        self.tfidf_matrix = None
        self.item_sim_matrix = None
        self.product_id_to_idx: Dict[str, int] = {}
        self.idx_to_product_id: Dict[int, str] = {}
        self.user_history: Dict[str, List[str]] = {}

    def fit(self, products_df: pd.DataFrame, interactions_df: Optional[pd.DataFrame] = None) -> "ContentBasedRecommender":
        """
        Fit TF-IDF on product metadata and calculate item-item cosine similarity matrix.
        """
        self.products_df = products_df.copy().reset_index(drop=True)
        self.product_id_to_idx = {pid: idx for idx, pid in enumerate(self.products_df["product_id"])}
        self.idx_to_product_id = {idx: pid for idx, pid in enumerate(self.products_df["product_id"])}
        
        # Fit TF-IDF on rich metadata soup
        self.tfidf_matrix = self.vectorizer.fit_transform(self.products_df["content_soup"].fillna(""))
        
        # Compute item-item cosine similarity
        self.item_sim_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

        # Index user historical items if interactions provided
        self.user_history = {}
        if interactions_df is not None and not interactions_df.empty:
            for user_id, group in interactions_df.groupby("user_id"):
                self.user_history[user_id] = group["product_id"].tolist()

        self.is_fitted = True
        return self

    def get_similar_items(self, product_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Find the most similar items to a target item using metadata cosine similarity."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        if product_id not in self.product_id_to_idx:
            return []

        idx = self.product_id_to_idx[product_id]
        sim_scores = self.item_sim_matrix[idx]
        
        # Sort indices descending, excluding the item itself
        similar_indices = np.argsort(sim_scores)[::-1]
        results = []
        for sim_idx in similar_indices:
            if sim_idx == idx:
                continue
            pid = self.idx_to_product_id[sim_idx]
            score = float(sim_scores[sim_idx])
            results.append({
                "product_id": pid,
                "score": round(score, 4),
                "reason": f"Content similarity to {product_id} ({int(score * 100)}%)"
            })
            if len(results) >= top_k:
                break
        return results

    def recommend(self, user_id: str, top_k: int = 10, exclude_known: bool = True) -> List[Dict[str, Any]]:
        """
        Build a synthetic user profile vector by averaging the TF-IDF vectors of interacted items,
        and score all catalog items by cosine similarity.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        known_items = self.user_history.get(user_id, [])
        known_indices = [self.product_id_to_idx[pid] for pid in known_items if pid in self.product_id_to_idx]

        if not known_indices:
            # Cold-start fallback: return top diverse products
            return []

        # Create user profile vector: average vector of user's past products
        user_vector = np.asarray(self.tfidf_matrix[known_indices].mean(axis=0))
        user_sims = cosine_similarity(user_vector, self.tfidf_matrix).flatten()

        sorted_indices = np.argsort(user_sims)[::-1]
        known_set = set(known_items) if exclude_known else set()

        results = []
        for idx in sorted_indices:
            pid = self.idx_to_product_id[idx]
            if pid in known_set:
                continue
            score = float(user_sims[idx])
            results.append({
                "product_id": pid,
                "score": round(score, 4),
                "reason": f"Matches user profile taste ({int(score * 100)}% match)"
            })
            if len(results) >= top_k:
                break
        return results
