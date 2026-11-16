"""
AURA API & Recommendation Engine Test Suite
Tests API business logic, recommendations, event tracking, and customer intelligence.
"""

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import pytest
from src.api.main import (
    get_health,
    list_products,
    list_users,
    get_user_recommendations,
    get_similar_products,
    get_trending,
    track_event,
    get_customer_intelligence,
    get_customer_segments_summary,
    get_cohort_matrix
)
from src.api.schemas import EventTrackRequest

def test_health_endpoint():
    res = get_health()
    assert res.status == "HEALTHY"
    assert res.total_products > 0
    assert res.total_users > 0
    assert res.total_interactions > 0

def test_list_products():
    res = list_products(limit=5)
    assert len(res) == 5
    assert "product_id" in res[0]
    assert "category" in res[0]

def test_list_users():
    res = list_users(limit=5)
    assert len(res) == 5
    assert "user_id" in res[0]

def test_recommend_user_hybrid():
    res = get_user_recommendations(user_id="USER_0001", top_k=5, model="hybrid")
    assert res.user_id == "USER_0001"
    assert res.algorithm == "AURA_HYBRID"
    assert len(res.recommendations) <= 5
    if len(res.recommendations) > 0:
        rec = res.recommendations[0]
        assert rec.product_id != ""
        assert rec.score >= 0.0
        assert rec.title != ""

def test_recommend_user_models():
    for m in ["content", "svd", "collab"]:
        res = get_user_recommendations(user_id="USER_0001", top_k=3, model=m)
        assert res.algorithm == f"AURA_{m.upper()}"

def test_recommend_similar_products():
    res = get_similar_products(product_id="PROD_0001", top_k=4)
    assert res.product_id == "PROD_0001"
    assert len(res.similar_items) <= 4

def test_trending_products():
    res = get_trending(limit=5)
    assert len(res) <= 5
    assert "popularity_score" in res[0]

def test_event_tracking():
    req = EventTrackRequest(
        user_id="USER_0001",
        product_id="PROD_0005",
        event_type="view",
        dwell_time_seconds=30
    )
    res = track_event(req)
    assert res.status == "SUCCESS"
    assert res.interaction_id.startswith("INT_")

def test_customer_intelligence():
    res = get_customer_intelligence(user_id="USER_0001")
    assert res.user_id == "USER_0001"
    assert res.segment_name != ""
    assert 0.0 <= res.churn_probability <= 1.0
    assert res.clv_tier in ["Diamond (VIP)", "Gold", "Silver", "Bronze"]
    assert res.actionable_recommendation != ""

def test_segments_summary():
    res = get_customer_segments_summary()
    assert len(res) > 0
    assert "segment_name" in res[0]
    assert "user_count" in res[0]

def test_cohort_matrix():
    res = get_cohort_matrix()
    assert isinstance(res, list)
