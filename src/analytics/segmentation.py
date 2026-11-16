"""
Customer Segmentation Engine using RFM Analysis & K-Means Clustering
Segments users into business personas (Champions, Loyalists, At Risk, Hibernating, etc.)
and generates 2D/3D coordinates for interactive dashboard visualization.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

SEGMENT_LABELS = {
    0: "Champions",
    1: "Loyal Customers",
    2: "Potential Loyalists",
    3: "At Risk",
    4: "Hibernating"
}

class CustomerSegmentation:
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.pca = PCA(n_components=2, random_state=random_state)
        self.is_fitted = False
        self.cluster_names: Dict[int, str] = {}

    def fit_transform(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit K-Means on RFM features (recency_days_purchase, frequency, monetary)
        and append cluster labels and 2D PCA projections.
        """
        df = rfm_df.copy()
        
        # Feature columns
        feature_cols = ["recency_days_purchase", "frequency", "monetary", "total_interactions"]
        # Impute and clean
        X = df[feature_cols].fillna(0).values
        
        # Log-transform skewed features to stabilize variance
        X_trans = np.zeros_like(X)
        X_trans[:, 0] = np.log1p(np.maximum(0, X[:, 0])) # recency
        X_trans[:, 1] = np.log1p(np.maximum(0, X[:, 1])) # frequency
        X_trans[:, 2] = np.log1p(np.maximum(0, X[:, 2])) # monetary
        X_trans[:, 3] = np.log1p(np.maximum(0, X[:, 3])) # interactions

        X_scaled = self.scaler.fit_transform(X_trans)
        clusters = self.kmeans.fit_predict(X_scaled)
        pca_coords = self.pca.fit_transform(X_scaled)

        df["cluster"] = clusters
        df["pca_x"] = pca_coords[:, 0]
        df["pca_y"] = pca_coords[:, 1]

        # Dynamically assign meaningful segment names based on cluster mean monetary & recency
        cluster_stats = df.groupby("cluster").agg({
            "recency_days_purchase": "mean",
            "frequency": "mean",
            "monetary": "mean"
        })
        
        # Rank by RFM health: High Monetary + High Frequency + Low Recency = Best
        health_score = (
            cluster_stats["monetary"] / (cluster_stats["monetary"].max() + 1e-5) * 0.4 +
            cluster_stats["frequency"] / (cluster_stats["frequency"].max() + 1e-5) * 0.3 +
            (1.0 - cluster_stats["recency_days_purchase"] / (cluster_stats["recency_days_purchase"].max() + 1e-5)) * 0.3
        )
        ranked_clusters = health_score.sort_values(ascending=False).index.tolist()

        persona_order = ["Champions", "Loyal Customers", "Potential Loyalists", "At Risk", "Hibernating"]
        self.cluster_names = {cluster_id: persona_order[i] for i, cluster_id in enumerate(ranked_clusters)}

        df["segment_name"] = df["cluster"].map(self.cluster_names)
        self.is_fitted = True
        return df

    def get_cluster_profiles(self, df_with_clusters: pd.DataFrame) -> pd.DataFrame:
        """Generate executive summary statistics per customer segment."""
        return df_with_clusters.groupby("segment_name").agg(
            user_count=("user_id", "count"),
            avg_recency=("recency_days_purchase", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_spend=("monetary", "mean"),
            total_revenue=("monetary", "sum")
        ).round(2).reset_index()
