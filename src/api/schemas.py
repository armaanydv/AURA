"""
Pydantic Request & Response Schemas for AURA API
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ProductItem(BaseModel):
    product_id: str
    title: str
    category: str
    subcategory: Optional[str] = None
    brand: Optional[str] = None
    price: float
    score: Optional[float] = None
    reason: Optional[str] = None

class RecommendationResponse(BaseModel):
    user_id: str
    algorithm: str
    total_recommendations: int
    recommendations: List[ProductItem]

class SimilarProductResponse(BaseModel):
    product_id: str
    similar_items: List[ProductItem]

class EventTrackRequest(BaseModel):
    user_id: str
    product_id: str
    event_type: str = Field(..., description="view, like, add_to_cart, purchase")
    session_id: Optional[str] = None
    dwell_time_seconds: Optional[int] = 0

class EventTrackResponse(BaseModel):
    status: str
    interaction_id: str
    message: str

class CustomerIntelligenceResponse(BaseModel):
    user_id: str
    name: str
    email: str
    segment_name: str
    churn_probability: float
    churn_risk_tier: str
    actionable_recommendation: str
    historical_clv: float
    predicted_1yr_clv: float
    clv_tier: str
    total_orders: int
    recency_days: int

class HealthResponse(BaseModel):
    status: str
    version: str
    total_products: int
    total_users: int
    total_interactions: int
    total_purchases: int
