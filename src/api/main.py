"""
AURA - Core FastAPI Application
Asynchronous, high-throughput REST API serving personalized recommendations,
real-time event streaming, and customer intelligence analytics.
"""

import uuid
import time
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from src.data.pipeline import DataPipeline
from src.recommenders.hybrid import HybridRecommender
from src.analytics.segmentation import CustomerSegmentation
from src.analytics.churn import ChurnPredictor
from src.analytics.clv import CLVEngine
from src.analytics.cohorts import CohortAnalyzer
from src.api.schemas import (
    ProductItem,
    RecommendationResponse,
    SimilarProductResponse,
    EventTrackRequest,
    EventTrackResponse,
    CustomerIntelligenceResponse,
    HealthResponse
)

app = FastAPI(
    title="AURA - Recommendation & Personalization Engine API",
    description="Enterprise-grade AI Recommendation System & Customer Intelligence Platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for local dashboards and frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Application State
STATE = {
    "pipeline": None,
    "recommender": None,
    "segmenter": None,
    "churn_model": None,
    "clv_engine": None,
    "cohort_analyzer": None,
    "products_df": None,
    "users_df": None,
    "rfm_df": None,
    "segmented_rfm": None,
    "clv_df": None,
    "is_ready": False
}

def load_services():
    """Initialize data and models into memory."""
    pipeline = DataPipeline()
    pipeline.seed_data(num_products=150, num_users=400)
    
    products_df = pipeline.load_products_df()
    users_df = pipeline.load_users_df()
    ratings_df = pipeline.load_ratings_df()
    interactions_df = pipeline.load_implicit_feedback_df()
    trending_df = pipeline.get_trending_products(limit=15)
    rfm_df = pipeline.get_rfm_metrics()

    # Models
    recommender = HybridRecommender()
    recommender.fit(products_df, interactions_df, ratings_df, trending_df)

    segmenter = CustomerSegmentation(n_clusters=5)
    segmented_rfm = segmenter.fit_transform(rfm_df)

    churn_model = ChurnPredictor()
    churn_model.fit_evaluate(rfm_df)

    clv_engine = CLVEngine()
    churn_probs = churn_model.model.predict_proba(rfm_df[churn_model.feature_cols].fillna(0))[:, 1]
    clv_df = clv_engine.compute_clv(segmented_rfm, churn_probs)

    conn = pipeline.get_connection()
    cohort_analyzer = CohortAnalyzer(conn)

    STATE.update({
        "pipeline": pipeline,
        "recommender": recommender,
        "segmenter": segmenter,
        "churn_model": churn_model,
        "clv_engine": clv_engine,
        "cohort_analyzer": cohort_analyzer,
        "products_df": products_df,
        "users_df": users_df,
        "rfm_df": rfm_df,
        "segmented_rfm": segmented_rfm,
        "clv_df": clv_df,
        "is_ready": True
    })

# Load state on startup
load_services()

# -------------------------------------------------------------
# System & Catalog Routes
# -------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse, tags=["System"])
def get_health():
    """Health check endpoint with dataset volume statistics."""
    p_df = STATE["products_df"]
    u_df = STATE["users_df"]
    conn = STATE["pipeline"].get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM fact_interactions")
    i_cnt = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM fact_purchases")
    p_cnt = cur.fetchone()[0]

    return HealthResponse(
        status="HEALTHY",
        version="1.0.0",
        total_products=len(p_df) if p_df is not None else 0,
        total_users=len(u_df) if u_df is not None else 0,
        total_interactions=i_cnt,
        total_purchases=p_cnt
    )

@app.get("/api/products", tags=["Catalog"])
def list_products(category: Optional[str] = None, limit: int = 50):
    """Retrieve catalog products with optional category filtering."""
    df = STATE["products_df"]
    if category:
        df = df[df["category"].str.lower() == category.lower()]
    return df.head(limit)[["product_id", "title", "category", "subcategory", "brand", "price"]].to_dict(orient="records")

@app.get("/api/users", tags=["Catalog"])
def list_users(limit: int = 50):
    """Retrieve user profiles."""
    return STATE["users_df"].head(limit)[["user_id", "name", "email", "age", "location", "preferred_category"]].to_dict(orient="records")

# -------------------------------------------------------------
# Recommendation Endpoints
# -------------------------------------------------------------
@app.get("/api/recommend/user/{user_id}", response_model=RecommendationResponse, tags=["Recommendations"])
def get_user_recommendations(
    user_id: str,
    top_k: int = Query(10, ge=1, le=50),
    model: str = Query("hybrid", pattern="^(hybrid|content|svd|collab)$")
):
    """
    Generate real-time personalized recommendations for a user.
    Supports switching between Hybrid, Content-Based, SVD, and Collaborative filtering.
    """
    recommender = STATE["recommender"]
    if not recommender:
        raise HTTPException(status_code=503, detail="Recommender model is not ready.")

    if model == "content":
        raw_recs = recommender.cb_model.recommend(user_id, top_k=top_k)
    elif model == "svd":
        raw_recs = recommender.svd_model.recommend(user_id, top_k=top_k)
    elif model == "collab":
        raw_recs = recommender.cf_model.recommend(user_id, top_k=top_k)
    else:
        raw_recs = recommender.recommend(user_id, top_k=top_k)

    # Enrich recommendations with catalog metadata
    items = []
    prod_map = recommender.all_products_map
    for r in raw_recs:
        pid = r["product_id"]
        meta = prod_map.get(pid, {})
        items.append(ProductItem(
            product_id=pid,
            title=meta.get("title", pid),
            category=meta.get("category", "General"),
            subcategory=meta.get("subcategory"),
            brand=meta.get("brand"),
            price=float(meta.get("price", 0.0)),
            score=float(r.get("score", 0.0)),
            reason=r.get("reason", "Personalized match")
        ))

    return RecommendationResponse(
        user_id=user_id,
        algorithm=f"AURA_{model.upper()}",
        total_recommendations=len(items),
        recommendations=items
    )

@app.get("/api/recommend/similar/{product_id}", response_model=SimilarProductResponse, tags=["Recommendations"])
def get_similar_products(product_id: str, top_k: int = Query(6, ge=1, le=20)):
    """Fetch similar products using metadata cosine similarity and collaborative associations."""
    recommender = STATE["recommender"]
    raw_sims = recommender.get_similar_items(product_id, top_k=top_k)
    
    items = []
    prod_map = recommender.all_products_map
    for r in raw_sims:
        pid = r["product_id"]
        meta = prod_map.get(pid, {})
        items.append(ProductItem(
            product_id=pid,
            title=meta.get("title", pid),
            category=meta.get("category", "General"),
            price=float(meta.get("price", 0.0)),
            score=float(r.get("score", 0.0)),
            reason=r.get("reason")
        ))
    return SimilarProductResponse(product_id=product_id, similar_items=items)

@app.get("/api/recommend/trending", tags=["Recommendations"])
def get_trending(limit: int = 10):
    """Fetch top trending items across the platform."""
    trending_df = STATE["pipeline"].get_trending_products(limit=limit)
    return trending_df.to_dict(orient="records")

# -------------------------------------------------------------
# Event Streaming & Ingestion
# -------------------------------------------------------------
@app.post("/api/events/track", response_model=EventTrackResponse, tags=["Events"])
def track_event(payload: EventTrackRequest):
    """
    Ingest a real-time user event (clickstream view, cart add, like, purchase).
    Persists to the SQLite warehouse for immediate personalization feedback.
    """
    int_id = f"INT_{uuid.uuid4().hex[:10]}"
    now_iso = datetime.now().isoformat()
    session_id = payload.session_id or f"SESS_{payload.user_id}_live"

    conn = STATE["pipeline"].get_connection()
    with conn:
        conn.execute("""
            INSERT INTO fact_interactions (interaction_id, user_id, product_id, event_type, session_id, timestamp, dwell_time_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (int_id, payload.user_id, payload.product_id, payload.event_type, session_id, now_iso, payload.dwell_time_seconds))

        # If purchase event, also insert to fact_purchases
        if payload.event_type == "purchase":
            purch_id = f"PUR_{uuid.uuid4().hex[:10]}"
            # Look up price
            prod_meta = STATE["recommender"].all_products_map.get(payload.product_id, {})
            price = prod_meta.get("price", 50.0)
            conn.execute("""
                INSERT INTO fact_purchases (purchase_id, user_id, product_id, quantity, unit_price, total_amount, payment_method, timestamp)
                VALUES (?, ?, ?, 1, ?, ?, 'Instant Pay', ?)
            """, (purch_id, payload.user_id, payload.product_id, price, price, now_iso))

    return EventTrackResponse(
        status="SUCCESS",
        interaction_id=int_id,
        message=f"Event '{payload.event_type}' recorded successfully."
    )

# -------------------------------------------------------------
# Customer Analytics Endpoints
# -------------------------------------------------------------
@app.get("/api/analytics/customer/{user_id}", response_model=CustomerIntelligenceResponse, tags=["Analytics"])
def get_customer_intelligence(user_id: str):
    """Retrieve full customer 360 profile: RFM Segment, Churn Probability, CLV Tier, and Actionable Guidance."""
    clv_df = STATE["clv_df"]
    user_rows = clv_df[clv_df["user_id"] == user_id]
    if user_rows.empty:
        raise HTTPException(status_code=404, detail="Customer not found")

    row = user_rows.iloc[0]
    churn_analysis = STATE["churn_model"].predict_user_churn(row)

    return CustomerIntelligenceResponse(
        user_id=row["user_id"],
        name=row["name"],
        email=row["email"],
        segment_name=row.get("segment_name", "Regular"),
        churn_probability=churn_analysis["churn_probability"],
        churn_risk_tier=churn_analysis["risk_tier"],
        actionable_recommendation=churn_analysis["actionable_recommendation"],
        historical_clv=float(row.get("historical_clv", 0.0)),
        predicted_1yr_clv=float(row.get("predicted_1yr_clv", 0.0)),
        clv_tier=row.get("clv_tier", "Silver"),
        total_orders=int(row.get("frequency", 0)),
        recency_days=int(row.get("recency_days_purchase", 0))
    )

@app.get("/api/analytics/segments", tags=["Analytics"])
def get_customer_segments_summary():
    """Retrieve high-level customer segment breakdown with monetary averages."""
    clv_df = STATE["clv_df"]
    summary = clv_df.groupby("segment_name").agg(
        user_count=("user_id", "count"),
        avg_recency=("recency_days_purchase", "mean"),
        avg_spend=("monetary", "mean"),
        total_revenue=("monetary", "sum"),
        avg_churn_prob=("churn_probability", "mean")
    ).round(2).reset_index()
    return summary.to_dict(orient="records")

@app.get("/api/analytics/cohorts", tags=["Analytics"])
def get_cohort_matrix():
    """Retrieve monthly cohort retention matrix."""
    matrix = STATE["cohort_analyzer"].build_monthly_cohort_matrix()
    return matrix.reset_index().to_dict(orient="records")
