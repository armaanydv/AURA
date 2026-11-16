-- AURA - Core Database Schema (Analytical Star Schema)
-- Dimensions: dim_users, dim_products
-- Facts: fact_interactions (clickstream), fact_purchases, fact_ratings

-- Users Dimension
CREATE TABLE IF NOT EXISTS dim_users (
    user_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    age INTEGER,
    gender TEXT,
    location TEXT,
    signup_date TIMESTAMP NOT NULL,
    preferred_category TEXT
);

-- Products Dimension
CREATE TABLE IF NOT EXISTS dim_products (
    product_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    category TEXT NOT NULL,
    subcategory TEXT NOT NULL,
    brand TEXT NOT NULL,
    price REAL NOT NULL,
    cost REAL NOT NULL,
    description TEXT,
    tags TEXT,
    created_at TIMESTAMP NOT NULL
);

-- Clickstream & User Interactions Fact Table
CREATE TABLE IF NOT EXISTS fact_interactions (
    interaction_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    event_type TEXT NOT NULL, -- 'view', 'like', 'add_to_cart', 'share'
    session_id TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    dwell_time_seconds INTEGER DEFAULT 0,
    FOREIGN KEY (user_id) REFERENCES dim_users(user_id),
    FOREIGN KEY (product_id) REFERENCES dim_products(product_id)
);

-- Purchases Fact Table
CREATE TABLE IF NOT EXISTS fact_purchases (
    purchase_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    quantity INTEGER NOT NULL DEFAULT 1,
    unit_price REAL NOT NULL,
    total_amount REAL NOT NULL,
    payment_method TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    FOREIGN KEY (user_id) REFERENCES dim_users(user_id),
    FOREIGN KEY (product_id) REFERENCES dim_products(product_id)
);

-- Ratings & Reviews Fact Table
CREATE TABLE IF NOT EXISTS fact_ratings (
    rating_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    rating REAL NOT NULL, -- 1.0 to 5.0
    review_text TEXT,
    timestamp TIMESTAMP NOT NULL,
    UNIQUE(user_id, product_id),
    FOREIGN KEY (user_id) REFERENCES dim_users(user_id),
    FOREIGN KEY (product_id) REFERENCES dim_products(product_id)
);

-- Analytical Indexes for Fast SQL Queries
CREATE INDEX IF NOT EXISTS idx_interactions_user ON fact_interactions(user_id);
CREATE INDEX IF NOT EXISTS idx_interactions_product ON fact_interactions(product_id);
CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON fact_interactions(timestamp);

CREATE INDEX IF NOT EXISTS idx_purchases_user ON fact_purchases(user_id);
CREATE INDEX IF NOT EXISTS idx_purchases_timestamp ON fact_purchases(timestamp);

CREATE INDEX IF NOT EXISTS idx_ratings_user ON fact_ratings(user_id);
CREATE INDEX IF NOT EXISTS idx_ratings_product ON fact_ratings(product_id);
