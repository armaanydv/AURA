"""
AURA - Seed Database & Verify ML Models Execution
"""

import sys
from pathlib import Path

# Fix Windows console UTF-8 encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure workspace root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.data.pipeline import DataPipeline
from src.recommenders.hybrid import HybridRecommender
from src.recommenders.evaluation import evaluate_model
from src.analytics.segmentation import CustomerSegmentation
from src.analytics.churn import ChurnPredictor
from src.analytics.clv import CLVEngine
from src.analytics.cohorts import CohortAnalyzer

def main():
    print("🚀 [1/5] Initializing Database & Seeding E-Commerce Data...")
    pipeline = DataPipeline()
    counts = pipeline.seed_data(num_products=150, num_users=400, force=True)
    print(f"✅ Seeding Complete: {counts}")

    print("\n📊 [2/5] Loading Feature Dataframes...")
    products_df = pipeline.load_products_df()
    users_df = pipeline.load_users_df()
    ratings_df = pipeline.load_ratings_df()
    interactions_df = pipeline.load_implicit_feedback_df()
    trending_df = pipeline.get_trending_products(limit=10)
    rfm_df = pipeline.get_rfm_metrics()
    print(f"Products: {len(products_df)}, Users: {len(users_df)}, Ratings: {len(ratings_df)}, Interactions: {len(interactions_df)}")

    print("\n🧠 [3/5] Fitting Hybrid Recommender (Content-Based + SVD + Collaborative)...")
    recommender = HybridRecommender()
    recommender.fit(products_df, interactions_df, ratings_df, trending_df)
    
    # Test sample recommendations
    sample_user = users_df.iloc[0]["user_id"]
    recs = recommender.recommend(sample_user, top_k=5)
    print(f"Sample recommendations for {sample_user}:")
    for r in recs:
        print(f"  - [{r.get('score')}] {r.get('title')} ({r.get('category')}) -> {r.get('reason')}")

    # Evaluate offline metrics
    print("\n📈 Evaluating Offline RecSys Metrics...")
    test_sample = ratings_df.sample(frac=0.2, random_state=42)
    eval_metrics = evaluate_model(recommender, test_sample, k=10, max_users=30)
    print(f"Evaluation Metrics: {eval_metrics}")

    print("\n👥 [4/5] Running Customer Segmentation (RFM + K-Means)...")
    segmenter = CustomerSegmentation(n_clusters=5)
    segmented_rfm = segmenter.fit_transform(rfm_df)
    profiles = segmenter.get_cluster_profiles(segmented_rfm)
    print(profiles.to_string(index=False))

    print("\n⚠️ [5/5] Training Churn Prediction & Computing CLV...")
    churn_model = ChurnPredictor()
    churn_metrics = churn_model.fit_evaluate(rfm_df)
    print(f"Churn Model Performance: {churn_metrics}")

    clv_engine = CLVEngine()
    clv_df = clv_engine.compute_clv(segmented_rfm, churn_model.model.predict_proba(rfm_df[churn_model.feature_cols].fillna(0))[:, 1])
    print(f"CLV Tiers Distribution:\n{clv_df['clv_tier'].value_counts().to_string()}")

    print("\n🎉 ALL ENGINES INITIALIZED & VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
