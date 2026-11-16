"""
Customer Churn Prediction Engine
Trains an ensemble tree classifier to predict 30-day churn probability,
computes AUC-ROC & Feature Importance, and classifies customers into actionable risk tiers.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

class ChurnPredictor:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = GradientBoostingClassifier(
            n_estimators=80,
            learning_rate=0.08,
            max_depth=3,
            random_state=random_state
        )
        self.feature_cols = [
            "recency_days_purchase",
            "frequency",
            "monetary",
            "avg_order_value",
            "total_interactions",
            "age"
        ]
        self.metrics: Dict[str, float] = {}
        self.feature_importance_: Dict[str, float] = {}
        self.is_fitted = False

    def fit_evaluate(self, rfm_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract features, define churn target (recency_days_active > 30), train model,
        and evaluate performance metrics.
        """
        df = rfm_df.copy()
        # Churn definition: User has had zero activity in past 30 days
        df["is_churn"] = (df["recency_days_active"] > 30).astype(int)

        X = df[self.feature_cols].fillna(0)
        y = df["is_churn"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=self.random_state, stratify=y
        )

        self.model.fit(X_train, y_train)

        # Predictions
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        self.metrics = {
            "AUC_ROC": round(float(roc_auc_score(y_test, y_prob)), 4),
            "Accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "Precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "Recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "F1_Score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "Test_Samples": len(y_test)
        }

        # Feature importances
        self.feature_importance_ = {
            feat: round(float(imp), 4)
            for feat, imp in zip(self.feature_cols, self.model.feature_importances_)
        }

        self.is_fitted = True
        return self.metrics

    def predict_user_churn(self, user_rfm_row: pd.Series) -> Dict[str, Any]:
        """Predict churn probability and assign actionable risk category for an individual user."""
        if not self.is_fitted:
            raise RuntimeError("Churn model is not fitted yet.")

        feats_df = pd.DataFrame([[user_rfm_row.get(c, 0) for c in self.feature_cols]], columns=self.feature_cols)
        prob = float(self.model.predict_proba(feats_df)[0, 1])

        if prob >= 0.70:
            tier = "High Risk"
            recommendation = "Urgent win-back campaign: Special 20% discount offer or push notification."
        elif prob >= 0.35:
            tier = "Medium Risk"
            recommendation = "Re-engagement email: Showcase trending items in preferred category."
        else:
            tier = "Low Risk / Healthy"
            recommendation = "Regular loyalty perks and new arrival showcases."

        return {
            "churn_probability": round(prob, 3),
            "risk_tier": tier,
            "actionable_recommendation": recommendation
        }

    def predict_all(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """Attach churn probability and risk tier to the entire user dataset."""
        df = rfm_df.copy()
        X = df[self.feature_cols].fillna(0)
        probs = self.model.predict_proba(X)[:, 1]
        df["churn_probability"] = np.round(probs, 3)
        df["churn_risk_tier"] = pd.cut(
            df["churn_probability"],
            bins=[-0.01, 0.35, 0.70, 1.0],
            labels=["Low Risk", "Medium Risk", "High Risk"]
        )
        return df
