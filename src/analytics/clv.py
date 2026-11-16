"""
Customer Lifetime Value (CLV) Engine
Computes Historical Value and Predictive 1-Year Forward CLV incorporating Churn Probability and Average Margin.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd

class CLVEngine:
    def __init__(self, profit_margin: float = 0.40):
        self.profit_margin = profit_margin

    def compute_clv(self, user_rfm_df: pd.DataFrame, churn_probs: np.ndarray) -> pd.DataFrame:
        """
        Compute Historical Spend, Predicted 1-Year Forward CLV, and CLV Tier.
        """
        df = user_rfm_df.copy()
        
        # Historical CLV
        df["historical_clv"] = df["monetary"].round(2)
        df["churn_probability"] = churn_probs

        # Annualized frequency estimation (based on frequency and lifespan)
        # Annual frequency = frequency * (365 / days_since_signup)
        signup_dt = pd.to_datetime(df["signup_date"])
        ref_dt = pd.to_datetime(df["signup_date"]).max()
        days_active = np.maximum(1, (ref_dt - signup_dt).dt.days)
        annualized_freq = (df["frequency"] / days_active) * 365.0
        
        # Predicted forward value = Annual Frequency * Avg Order Value * Retention Rate * Profit Margin
        retention_rate = np.maximum(0.05, 1.0 - df["churn_probability"])
        projected_spend = annualized_freq * df["avg_order_value"] * retention_rate
        
        # Total CLV = Historical Spend + Projected Future 1-Year Value
        df["predicted_1yr_clv"] = np.round(df["historical_clv"] + (projected_spend * self.profit_margin), 2)

        # Categorize into value tiers
        quantiles = df["predicted_1yr_clv"].quantile([0.50, 0.80, 0.95]).values
        def get_tier(val):
            if val >= quantiles[2]:
                return "Diamond (VIP)"
            elif val >= quantiles[1]:
                return "Gold"
            elif val >= quantiles[0]:
                return "Silver"
            else:
                return "Bronze"

        df["clv_tier"] = df["predicted_1yr_clv"].apply(get_tier)
        return df
