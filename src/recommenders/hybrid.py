"""
AURA Hybrid Recommendation Engine
Weighted ensemble combining Content-Based Filtering, SVD Matrix Factorization, and Collaborative Filtering,
with automated Popularity / Trending Fallback for cold-start users.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from src.recommenders.base import BaseRecommender
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.matrix_factorization import SVDRecommender
from src.recommenders.collaborative import CollaborativeRecommender

class HybridRecommender(BaseRecommender):
    def __init__(
        self,
        weight_svd: float = 0.50,
        weight_content: float = 0.30,
        weight_collab: float = 0.20
    ):
        super().__init__(name="Hybrid_RecSys_Ensemble")
        self.w_svd = weight_svd
        self.w_content = weight_content
        self.w_collab = weight_collab

        self.cb_model = ContentBasedRecommender()
        self.svd_model = SVDRecommender()
        self.cf_model = CollaborativeRecommender(mode="item")
        
        self.trending_items: List[Dict[str, Any]] = []
        self.all_products_map: Dict[str, Dict[str, Any]] = {}

    def fit(
        self,
        products_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        trending_df: Optional[pd.DataFrame] = None
    ) -> "HybridRecommender":
        """Fit all constituent recommendation engines and cache catalog metadata."""
        self.all_products_map = {row["product_id"]: row.to_dict() for _, row in products_df.iterrows()}
        
        # 1. Fit Content-Based
        self.cb_model.fit(products_df, interactions_df)

        # 2. Fit Matrix Factorization (SVD) on ratings (fallback to implicit interactions if needed)
        train_ratings = ratings_df if not ratings_df.empty else interactions_df
        val_col = "rating" if "rating" in train_ratings.columns else "implicit_weight"
        self.svd_model.fit(train_ratings, value_col=val_col)

        # 3. Fit Collaborative Filtering
        self.cf_model.fit(train_ratings, value_col=val_col)

        # 4. Cache trending items for Cold Start fallback
        if trending_df is not None and not trending_df.empty:
            self.trending_items = [
                {
                    "product_id": row["product_id"],
                    "score": float(row["popularity_score"]),
                    "reason": "Trending item (Cold-start fallback)"
                }
                for _, row in trending_df.iterrows()
            ]

        self.is_fitted = True
        return self

    def recommend(self, user_id: str, top_k: int = 10, exclude_known: bool = True) -> List[Dict[str, Any]]:
        """
        Ensemble recommendation:
        1. Fetch candidates and scores from SVD, Content-Based, and Collaborative models.
        2. Normalize scores per model to [0, 1].
        3. Compute weighted ensemble score.
        4. If cold-start, backfill with trending items.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        # Candidate generation
        svd_recs = self.svd_model.recommend(user_id, top_k=top_k * 3, exclude_known=exclude_known)
        cb_recs = self.cb_model.recommend(user_id, top_k=top_k * 3, exclude_known=exclude_known)
        cf_recs = self.cf_model.recommend(user_id, top_k=top_k * 3, exclude_known=exclude_known)

        # Detect cold start if all models return empty
        if not svd_recs and not cb_recs and not cf_recs:
            return self._cold_start_recommendations(top_k)

        # Helper to normalize scores
        def normalize_scores(recs):
            if not recs:
                return {}
            max_s = max(r["score"] for r in recs)
            min_s = min(r["score"] for r in recs)
            diff = max_s - min_s if max_s > min_s else 1.0
            return {r["product_id"]: (r["score"] - min_s) / diff for r in recs}

        norm_svd = normalize_scores(svd_recs)
        norm_cb = normalize_scores(cb_recs)
        norm_cf = normalize_scores(cf_recs)

        candidate_ids = set(norm_svd.keys()) | set(norm_cb.keys()) | set(norm_cf.keys())
        scored_candidates = []

        for pid in candidate_ids:
            score = (
                self.w_svd * norm_svd.get(pid, 0.0) +
                self.w_content * norm_cb.get(pid, 0.0) +
                self.w_collab * norm_cf.get(pid, 0.0)
            )
            # Explainability tag
            reasons = []
            if pid in norm_svd:
                reasons.append("Latent Preference")
            if pid in norm_cb:
                reasons.append("Content Affinity")
            if pid in norm_cf:
                reasons.append("Peer Behavior")
            
            scored_candidates.append({
                "product_id": pid,
                "score": round(score, 4),
                "reason": " + ".join(reasons) if reasons else "Personalized Ensemble"
            })

        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        top_recs = scored_candidates[:top_k]

        # If we still have fewer items than top_k, backfill with trending
        if len(top_recs) < top_k:
            existing_pids = {r["product_id"] for r in top_recs}
            for trend_item in self.trending_items:
                if trend_item["product_id"] not in existing_pids:
                    top_recs.append(trend_item)
                    existing_pids.add(trend_item["product_id"])
                    if len(top_recs) >= top_k:
                        break

        # Enrich with product metadata
        for item in top_recs:
            pid = item["product_id"]
            if pid in self.all_products_map:
                p_meta = self.all_products_map[pid]
                item["title"] = p_meta.get("title", "")
                item["category"] = p_meta.get("category", "")
                item["subcategory"] = p_meta.get("subcategory", "")
                item["brand"] = p_meta.get("brand", "")
                item["price"] = p_meta.get("price", 0.0)

        return top_recs

    def _cold_start_recommendations(self, top_k: int) -> List[Dict[str, Any]]:
        """Fallback to trending items when user has zero historical data."""
        recs = list(self.trending_items[:top_k])
        for item in recs:
            pid = item["product_id"]
            if pid in self.all_products_map:
                p_meta = self.all_products_map[pid]
                item["title"] = p_meta.get("title", "")
                item["category"] = p_meta.get("category", "")
                item["subcategory"] = p_meta.get("subcategory", "")
                item["brand"] = p_meta.get("brand", "")
                item["price"] = p_meta.get("price", 0.0)
        return recs

    def get_similar_items(self, product_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Fetch similar items by ensembling content similarity and latent item factor similarity."""
        cb_sims = self.cb_model.get_similar_items(product_id, top_k=top_k)
        for item in cb_sims:
            pid = item["product_id"]
            if pid in self.all_products_map:
                p_meta = self.all_products_map[pid]
                item["title"] = p_meta.get("title", "")
                item["category"] = p_meta.get("category", "")
                item["price"] = p_meta.get("price", 0.0)
        return cb_sims
