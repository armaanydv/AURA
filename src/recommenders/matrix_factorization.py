"""
Matrix Factorization Recommendation Engine via Singular Value Decomposition (SVD)
Decomposes the User-Item interaction matrix into low-rank latent user and item factor matrices.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
from src.recommenders.base import BaseRecommender

class SVDRecommender(BaseRecommender):
    def __init__(self, n_factors: int = 20):
        super().__init__(name="MatrixFactorization_SVD")
        self.n_factors = n_factors
        self.user_item_matrix: Optional[pd.DataFrame] = None
        self.user_ids: List[str] = []
        self.item_ids: List[str] = []
        self.user_to_idx: Dict[str, int] = {}
        self.item_to_idx: Dict[str, int] = {}
        self.idx_to_item: Dict[int, str] = {}
        
        # Latent representations
        self.user_factors = None # U * sigma
        self.item_factors = None # V^T
        self.predicted_matrix: Optional[np.ndarray] = None
        self.user_means: Optional[np.ndarray] = None

    def fit(self, interactions_df: pd.DataFrame, value_col: str = "rating") -> "SVDRecommender":
        """
        Fit Truncated SVD on user-item matrix.
        interactions_df must have ['user_id', 'product_id', value_col].
        """
        # Create pivot table
        pivot = interactions_df.pivot_table(
            index="user_id", 
            columns="product_id", 
            values=value_col, 
            fill_value=0.0
        )
        self.user_item_matrix = pivot
        self.user_ids = list(pivot.index)
        self.item_ids = list(pivot.columns)
        self.user_to_idx = {uid: i for i, uid in enumerate(self.user_ids)}
        self.item_to_idx = {iid: i for i, iid in enumerate(self.item_ids)}
        self.idx_to_item = {i: iid for i, iid in enumerate(self.item_ids)}

        R = pivot.values.astype(float)
        
        # User mean centering for non-zero values
        # To avoid zero-bias, compute mean over observed entries
        mask = R > 0
        user_sums = np.sum(R, axis=1)
        user_counts = np.sum(mask, axis=1)
        user_counts[user_counts == 0] = 1 # avoid div by zero
        self.user_means = user_sums / user_counts
        
        R_demeaned = np.where(mask, R - self.user_means[:, np.newaxis], 0.0)

        # Cap n_factors to min(R.shape) - 1
        k = min(self.n_factors, min(R.shape) - 2)
        if k < 2:
            k = 2

        U, sigma, Vt = svds(R_demeaned, k=k)
        
        # Sort singular values in descending order
        idx = np.argsort(sigma)[::-1]
        sigma = sigma[idx]
        U = U[:, idx]
        Vt = Vt[idx, :]

        sigma_diag = np.diag(sigma)
        self.user_factors = np.dot(U, sigma_diag)
        self.item_factors = Vt.T # Shape: (num_items, k)
        
        # Full reconstructed matrix
        self.predicted_matrix = np.dot(self.user_factors, Vt) + self.user_means[:, np.newaxis]
        self.is_fitted = True
        return self

    def recommend(self, user_id: str, top_k: int = 10, exclude_known: bool = True) -> List[Dict[str, Any]]:
        """Recommend items for a user based on reconstructed latent factor ratings."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        if user_id not in self.user_to_idx:
            # Cold-start user
            return []

        u_idx = self.user_to_idx[user_id]
        pred_scores = self.predicted_matrix[u_idx]

        if exclude_known:
            # Mask out items the user already interacted with
            interacted_mask = self.user_item_matrix.values[u_idx] > 0
            pred_scores = np.where(interacted_mask, -np.inf, pred_scores)

        sorted_indices = np.argsort(pred_scores)[::-1]
        
        results = []
        for idx in sorted_indices:
            score = float(pred_scores[idx])
            if score == -np.inf:
                continue
            iid = self.idx_to_item[idx]
            results.append({
                "product_id": iid,
                "score": round(max(0.0, score), 3),
                "reason": f"Predicted rating {round(score, 2)}/5 via SVD latent factors"
            })
            if len(results) >= top_k:
                break
        return results

    def get_similar_items(self, product_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Find items with similar latent representations (dot product / cosine similarity in latent space)."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        if product_id not in self.item_to_idx:
            return []

        i_idx = self.item_to_idx[product_id]
        target_vec = self.item_factors[i_idx].reshape(1, -1)
        sims = cosine_similarity(target_vec, self.item_factors).flatten()

        sorted_indices = np.argsort(sims)[::-1]
        results = []
        for idx in sorted_indices:
            if idx == i_idx:
                continue
            iid = self.idx_to_item[idx]
            score = float(sims[idx])
            results.append({
                "product_id": iid,
                "score": round(score, 4),
                "reason": f"Latent factor similarity ({int(score * 100)}%)"
            })
            if len(results) >= top_k:
                break
        return results
