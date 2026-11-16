"""
AURA Data Pipeline & Database Management
Manages SQLite warehouse, schema migrations, synthetic seeding, and analytical queries.
"""

import os
import sqlite3
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import pandas as pd
from src.data.generator import ECommerceDataGenerator

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "aura_warehouse.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

class DataPipeline:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def get_connection(self) -> sqlite3.Connection:
        """Get an active SQLite connection with row factory configured."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def initialize_schema(self) -> None:
        """Create tables and indexes defined in schema.sql."""
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema_sql = f.read()
        with self.get_connection() as conn:
            conn.executescript(schema_sql)
            conn.commit()

    def seed_data(self, num_products: int = 150, num_users: int = 400, force: bool = False) -> Dict[str, int]:
        """Seed synthetic e-commerce data if tables are empty or force is True."""
        self.initialize_schema()
        
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM dim_products")
            count = cur.fetchone()[0]
            if count > 0 and not force:
                cur.execute("SELECT COUNT(*) FROM dim_users")
                u_count = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM fact_interactions")
                i_count = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM fact_purchases")
                p_count = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM fact_ratings")
                r_count = cur.fetchone()[0]
                return {
                    "products": count,
                    "users": u_count,
                    "interactions": i_count,
                    "purchases": p_count,
                    "ratings": r_count
                }

            if force:
                conn.executescript("""
                    DELETE FROM fact_ratings;
                    DELETE FROM fact_purchases;
                    DELETE FROM fact_interactions;
                    DELETE FROM dim_products;
                    DELETE FROM dim_users;
                """)

            generator = ECommerceDataGenerator(seed=42)
            products = generator.generate_products(count=num_products)
            users = generator.generate_users(count=num_users)
            activity = generator.generate_interactions_and_purchases(users, products)

            # Insert Products
            cur.executemany("""
                INSERT INTO dim_products (product_id, title, category, subcategory, brand, price, cost, description, tags, created_at)
                VALUES (:product_id, :title, :category, :subcategory, :brand, :price, :cost, :description, :tags, :created_at)
            """, products)

            # Insert Users
            cur.executemany("""
                INSERT INTO dim_users (user_id, name, email, age, gender, location, signup_date, preferred_category)
                VALUES (:user_id, :name, :email, :age, :gender, :location, :signup_date, :preferred_category)
            """, users)

            # Insert Interactions
            cur.executemany("""
                INSERT INTO fact_interactions (interaction_id, user_id, product_id, event_type, session_id, timestamp, dwell_time_seconds)
                VALUES (:interaction_id, :user_id, :product_id, :event_type, :session_id, :timestamp, :dwell_time_seconds)
            """, activity["interactions"])

            # Insert Purchases
            cur.executemany("""
                INSERT INTO fact_purchases (purchase_id, user_id, product_id, quantity, unit_price, total_amount, payment_method, timestamp)
                VALUES (:purchase_id, :user_id, :product_id, :quantity, :unit_price, :total_amount, :payment_method, :timestamp)
            """, activity["purchases"])

            # Insert Ratings
            cur.executemany("""
                INSERT INTO fact_ratings (rating_id, user_id, product_id, rating, review_text, timestamp)
                VALUES (:rating_id, :user_id, :product_id, :rating, :review_text, :timestamp)
            """, activity["ratings"])

            conn.commit()

            return {
                "products": len(products),
                "users": len(users),
                "interactions": len(activity["interactions"]),
                "purchases": len(activity["purchases"]),
                "ratings": len(activity["ratings"])
            }

    # -------------------------------------------------------------
    # Analytical & RecSys Feature Extraction Queries
    # -------------------------------------------------------------
    def load_products_df(self) -> pd.DataFrame:
        """Load all products with engineered text content for Content-Based filtering."""
        with self.get_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM dim_products", conn)
        # Combine text features for TF-IDF / embeddings
        df["content_soup"] = (
            df["title"] + " " + 
            df["category"] + " " + 
            df["subcategory"] + " " + 
            df["brand"] + " " + 
            df["tags"].str.replace(",", " ") + " " + 
            df["description"]
        )
        return df

    def load_users_df(self) -> pd.DataFrame:
        """Load all users."""
        with self.get_connection() as conn:
            return pd.read_sql_query("SELECT * FROM dim_users", conn)

    def load_ratings_df(self) -> pd.DataFrame:
        """Load explicit ratings for Collaborative Filtering."""
        with self.get_connection() as conn:
            return pd.read_sql_query("SELECT user_id, product_id, rating, timestamp FROM fact_ratings", conn)

    def load_implicit_feedback_df(self) -> pd.DataFrame:
        """
        Derive implicit feedback matrix from clickstream + purchases:
        view = 1 pt, add_to_cart = 3 pts, purchase = 5 pts.
        """
        query = """
        WITH interaction_points AS (
            SELECT 
                user_id, 
                product_id, 
                CASE 
                    WHEN event_type = 'view' THEN 1.0
                    WHEN event_type = 'add_to_cart' THEN 3.0
                    ELSE 1.0
                END as score
            FROM fact_interactions
        ),
        purchase_points AS (
            SELECT 
                user_id, 
                product_id, 
                5.0 * quantity as score
            FROM fact_purchases
        ),
        combined AS (
            SELECT * FROM interaction_points
            UNION ALL
            SELECT * FROM purchase_points
        )
        SELECT 
            user_id, 
            product_id, 
            SUM(score) as implicit_weight
        FROM combined
        GROUP BY user_id, product_id
        """
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn)

    def get_rfm_metrics(self) -> pd.DataFrame:
        """
        Calculate Recency, Frequency, Monetary (RFM) metrics for customer segmentation and churn modeling.
        """
        query = """
        WITH max_time AS (
            SELECT MAX(timestamp) as ref_time FROM fact_interactions
        ),
        user_orders AS (
            SELECT 
                p.user_id,
                MAX(p.timestamp) as last_purchase_time,
                COUNT(DISTINCT p.purchase_id) as total_orders,
                SUM(p.total_amount) as total_spend,
                AVG(p.total_amount) as avg_order_value
            FROM fact_purchases p
            GROUP BY p.user_id
        ),
        user_activity AS (
            SELECT 
                i.user_id,
                MAX(i.timestamp) as last_active_time,
                COUNT(i.interaction_id) as total_interactions
            FROM fact_interactions i
            GROUP BY i.user_id
        )
        SELECT 
            u.user_id,
            u.name,
            u.email,
            u.age,
            u.gender,
            u.location,
            u.preferred_category,
            u.signup_date,
            COALESCE(uo.total_orders, 0) as frequency,
            COALESCE(uo.total_spend, 0.0) as monetary,
            COALESCE(uo.avg_order_value, 0.0) as avg_order_value,
            COALESCE(ua.total_interactions, 0) as total_interactions,
            CAST((julianday(m.ref_time) - julianday(COALESCE(uo.last_purchase_time, u.signup_date))) AS INTEGER) as recency_days_purchase,
            CAST((julianday(m.ref_time) - julianday(COALESCE(ua.last_active_time, u.signup_date))) AS INTEGER) as recency_days_active
        FROM dim_users u
        CROSS JOIN max_time m
        LEFT JOIN user_orders uo ON u.user_id = uo.user_id
        LEFT JOIN user_activity ua ON u.user_id = ua.user_id
        """
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn)

    def get_funnel_summary(self) -> Dict[str, int]:
        """Compute the overall user conversion funnel."""
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM fact_interactions WHERE event_type = 'view'")
            views = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM fact_interactions WHERE event_type = 'add_to_cart'")
            carts = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM fact_purchases")
            purchases = cur.fetchone()[0]
            return {"views": views, "add_to_cart": carts, "purchases": purchases}

    def get_trending_products(self, limit: int = 10) -> pd.DataFrame:
        """Fetch trending products based on combined recent views and purchases."""
        query = """
        SELECT 
            p.product_id,
            p.title,
            p.category,
            p.subcategory,
            p.brand,
            p.price,
            COUNT(i.interaction_id) as interaction_count,
            COALESCE(purch.purchase_count, 0) as purchase_count,
            (COUNT(i.interaction_id) + COALESCE(purch.purchase_count, 0) * 3) as popularity_score
        FROM dim_products p
        LEFT JOIN fact_interactions i ON p.product_id = i.product_id
        LEFT JOIN (
            SELECT product_id, COUNT(*) as purchase_count FROM fact_purchases GROUP BY product_id
        ) purch ON p.product_id = purch.product_id
        GROUP BY p.product_id
        ORDER BY popularity_score DESC
        LIMIT :limit
        """
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params={"limit": limit})
