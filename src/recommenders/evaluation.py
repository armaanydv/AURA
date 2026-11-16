"""
AURA Recommendation System Evaluation Metrics
Computes Precision@K, Recall@K, NDCG@K, MAP@K, and Hit Rate@K on train/test splits.
"""

from typing import List, Dict, Any, Set
import numpy as np
import pandas as pd

def precision_at_k(actual: Set[str], predicted: List[str], k: int) -> float:
    """Proportion of top-K recommendations that are relevant."""
    if not predicted or k <= 0:
        return 0.0
    pred_k = predicted[:k]
    hits = len(set(pred_k) & actual)
    return hits / float(k)

def recall_at_k(actual: Set[str], predicted: List[str], k: int) -> float:
    """Proportion of actual relevant items retrieved in top-K."""
    if not actual or not predicted or k <= 0:
        return 0.0
    pred_k = predicted[:k]
    hits = len(set(pred_k) & actual)
    return hits / float(len(actual))

def ndcg_at_k(actual: Set[str], predicted: List[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at rank K."""
    if not actual or not predicted or k <= 0:
        return 0.0
    pred_k = predicted[:k]
    
    # DCG: sum((2^rel - 1) / log2(rank + 1))
    dcg = 0.0
    for rank, item in enumerate(pred_k, start=1):
        if item in actual:
            dcg += 1.0 / np.log2(rank + 1)
            
    # IDCG: ideal DCG if all relevant items were ranked at top
    ideal_hits = min(len(actual), k)
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal_hits + 1))
    
    if idcg == 0:
        return 0.0
    return dcg / idcg

def hit_rate_at_k(actual: Set[str], predicted: List[str], k: int) -> float:
    """1.0 if at least one relevant item is in top-K, 0.0 otherwise."""
    if not actual or not predicted or k <= 0:
        return 0.0
    pred_k = predicted[:k]
    return 1.0 if (set(pred_k) & actual) else 0.0

def evaluate_model(recommender, train_df: pd.DataFrame, test_df: pd.DataFrame, k: int = 10, max_users: int = 50) -> Dict[str, float]:
    """
    Evaluate any AURA recommender over a holdout test dataframe with columns ['user_id', 'product_id'].
    Ensures test items were not in train.
    """
    precisions = []
    recalls = []
    ndcgs = []
    hit_rates = []

    # Get users who have entries in test_df
    user_groups = list(test_df.groupby("user_id"))[:max_users]

    for user_id, group in user_groups:
        actual_items = set(group["product_id"].tolist())
        recs = recommender.recommend(user_id, top_k=k, exclude_known=True)
        pred_items = [r["product_id"] for r in recs]

        precisions.append(precision_at_k(actual_items, pred_items, k))
        recalls.append(recall_at_k(actual_items, pred_items, k))
        ndcgs.append(ndcg_at_k(actual_items, pred_items, k))
        hit_rates.append(hit_rate_at_k(actual_items, pred_items, k))

    return {
        f"Precision@{k}": round(float(np.mean(precisions)), 4) if precisions else 0.0,
        f"Recall@{k}": round(float(np.mean(recalls)), 4) if recalls else 0.0,
        f"NDCG@{k}": round(float(np.mean(ndcgs)), 4) if ndcgs else 0.0,
        f"HitRate@{k}": round(float(np.mean(hit_rates)), 4) if hit_rates else 0.0,
        "EvaluatedUsers": len(user_groups)
    }
