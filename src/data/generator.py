"""
AURA Synthetic E-Commerce Data Generator
Generates realistic, skewed, enterprise-grade e-commerce data:
- Products catalog with diverse categories and textual descriptions
- Users with behavioral archetypes and preferences
- Clickstream event stream (views, carts, likes, dwell time)
- Purchases and transactional history spanning 180 days
- Explicit ratings for collaborative filtering
"""

import random
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np

CATEGORIES = {
    "Electronics": {
        "subcategories": ["Smartphones", "Audio", "Laptops", "Wearables", "Accessories"],
        "brands": ["Apple", "Samsung", "Sony", "Bose", "Dell", "Anker", "Logitech"],
        "price_range": (25.0, 1499.0),
        "tags": ["gadget", "bluetooth", "wireless", "battery", "usb-c", "portable", "premium"]
    },
    "Fashion": {
        "subcategories": ["Men's Apparel", "Women's Apparel", "Footwear", "Watches", "Bags"],
        "brands": ["Nike", "Adidas", "Levi's", "Zara", "Casio", "Fossil", "Patagonia"],
        "price_range": (15.0, 350.0),
        "tags": ["cotton", "sportswear", "trendy", "casual", "vintage", "eco-friendly", "waterproof"]
    },
    "Fitness & Health": {
        "subcategories": ["Gym Equipment", "Yoga", "Supplements", "Running", "Recovery"],
        "brands": ["Lululemon", "Under Armour", "Optimum Nutrition", "Garmin", "Therabody"],
        "price_range": (20.0, 450.0),
        "tags": ["workout", "fitness", "organic", "cardio", "protein", "training", "wellness"]
    },
    "Home & Living": {
        "subcategories": ["Kitchen", "Furniture", "Lighting", "Coffee", "Decor"],
        "brands": ["IKEA", "Nespresso", "Dyson", "Philips", "KitchenAid", "Breville"],
        "price_range": (30.0, 899.0),
        "tags": ["smart-home", "kitchen", "espresso", "minimalist", "ambient", "durable"]
    },
    "Books & Media": {
        "subcategories": ["Sci-Fi", "Business", "Self-Help", "Tech", "Biographies"],
        "brands": ["Penguin", "HarperCollins", "O'Reilly", "Simon & Schuster", "MIT Press"],
        "price_range": (10.0, 65.0),
        "tags": ["bestseller", "reading", "knowledge", "career", "leadership", "deep-learning"]
    }
}

FIRST_NAMES = ["Liam", "Olivia", "Noah", "Emma", "Oliver", "Charlotte", "Elijah", "Amelia", 
               "James", "Sophia", "Aarav", "Priya", "Lucas", "Mia", "Mateo", "Harper", 
               "Ethan", "Evelyn", "Daniel", "Abigail", "Arjun", "Zara", "Leo", "Maya"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", 
              "Patel", "Sharma", "Rodriguez", "Wilson", "Chen", "Kim", "Anderson", "Taylor"]
LOCATIONS = ["New York, US", "San Francisco, US", "London, UK", "Toronto, CA", "Berlin, DE", 
             "Bengaluru, IN", "Sydney, AU", "Paris, FR", "Tokyo, JP", "Singapore, SG"]

class ECommerceDataGenerator:
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.now = datetime.now()

    def generate_products(self, count: int = 150) -> List[Dict[str, Any]]:
        """Generate a realistic product catalog with rich text metadata for Content-Based filtering."""
        products = []
        for i in range(1, count + 1):
            category = random.choice(list(CATEGORIES.keys()))
            meta = CATEGORIES[category]
            subcategory = random.choice(meta["subcategories"])
            brand = random.choice(meta["brands"])
            min_p, max_p = meta["price_range"]
            price = round(random.uniform(min_p, max_p), 2)
            cost = round(price * random.uniform(0.45, 0.70), 2)
            
            selected_tags = random.sample(meta["tags"], k=random.randint(2, 4))
            product_id = f"PROD_{i:04d}"
            title = f"{brand} {subcategory[:-1] if subcategory.endswith('s') else subcategory} {random.choice(['Pro', 'Air', 'Elite', 'Max', 'Plus', 'Classic', 'Prime'])}"
            description = (f"The ultimate {title} by {brand}. Designed for superior performance in {subcategory} and {category}. "
                           f"Features top-tier {', '.join(selected_tags)} engineering.")
            
            created_days_ago = random.randint(60, 365)
            created_at = (self.now - timedelta(days=created_days_ago)).isoformat()
            
            products.append({
                "product_id": product_id,
                "title": title,
                "category": category,
                "subcategory": subcategory,
                "brand": brand,
                "price": price,
                "cost": cost,
                "description": description,
                "tags": ",".join(selected_tags),
                "created_at": created_at
            })
        return products

    def generate_users(self, count: int = 400) -> List[Dict[str, Any]]:
        """Generate diverse user profiles with behavioral segments and category affinity."""
        users = []
        categories = list(CATEGORIES.keys())
        for i in range(1, count + 1):
            first = random.choice(FIRST_NAMES)
            last = random.choice(LAST_NAMES)
            user_id = f"USER_{i:04d}"
            email = f"{first.lower()}.{last.lower()}{random.randint(10, 999)}@example.com"
            age = int(np.clip(np.random.normal(32, 10), 18, 70))
            gender = random.choice(["Male", "Female", "Non-Binary"])
            location = random.choice(LOCATIONS)
            preferred_cat = random.choice(categories)
            
            # Signup dates distributed across the last 180 days
            signup_days_ago = random.randint(10, 180)
            signup_date = (self.now - timedelta(days=signup_days_ago)).isoformat()
            
            users.append({
                "user_id": user_id,
                "name": f"{first} {last}",
                "email": email,
                "age": age,
                "gender": gender,
                "location": location,
                "signup_date": signup_date,
                "preferred_category": preferred_cat
            })
        return users

    def generate_interactions_and_purchases(
        self, users: List[Dict[str, Any]], products: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generate clickstream events, purchases, and ratings following realistic power-law user activity."""
        interactions = []
        purchases = []
        ratings = []
        
        # Product lookup map
        prod_map = {p["product_id"]: p for p in products}
        prod_by_cat = {}
        for p in products:
            prod_by_cat.setdefault(p["category"], []).append(p["product_id"])
            
        all_prod_ids = [p["product_id"] for p in products]
        
        # Activity skew: Pareto principle (20% power users generate 60% of events)
        user_activity_weights = np.random.pareto(a=1.8, size=len(users))
        user_activity_weights /= user_activity_weights.max() # scale 0 to 1

        for user, activity_score in zip(users, user_activity_weights):
            user_id = user["user_id"]
            preferred_cat = user["preferred_category"]
            signup_dt = datetime.fromisoformat(user["signup_date"])
            
            # Active user vs churned/inactive user
            # High activity users have recent sessions; inactive users have not been seen in 30+ days
            is_churning = random.random() < 0.28
            if is_churning:
                last_active_days_ago = random.randint(35, 120)
            else:
                last_active_days_ago = random.randint(0, 14)

            max_span_days = max(1, (self.now - signup_dt).days - last_active_days_ago)
            num_sessions = max(1, int(activity_score * 25) + random.randint(1, 4))
            
            user_rated_products = set()

            for s_idx in range(num_sessions):
                session_id = f"SESS_{user_id}_{s_idx+1}_{uuid.uuid4().hex[:6]}"
                session_day_offset = random.randint(last_active_days_ago, last_active_days_ago + max_span_days)
                session_time = self.now - timedelta(days=session_day_offset, hours=random.randint(0, 23), minutes=random.randint(0, 59))
                
                # Number of items browsed in this session
                session_length = random.randint(1, 8)
                
                for step in range(session_length):
                    # 70% chance of browsing preferred category, 30% random discovery
                    if random.random() < 0.70 and preferred_cat in prod_by_cat:
                        prod_id = random.choice(prod_by_cat[preferred_cat])
                    else:
                        prod_id = random.choice(all_prod_ids)
                        
                    prod = prod_map[prod_id]
                    event_time = session_time + timedelta(seconds=step * random.randint(15, 120))
                    
                    # Funnel: View -> Like/Cart -> Purchase
                    event_type = "view"
                    dwell_seconds = random.randint(10, 240)
                    
                    interactions.append({
                        "interaction_id": f"INT_{uuid.uuid4().hex[:10]}",
                        "user_id": user_id,
                        "product_id": prod_id,
                        "event_type": event_type,
                        "session_id": session_id,
                        "timestamp": event_time.isoformat(),
                        "dwell_time_seconds": dwell_seconds
                    })
                    
                    # Cart addition (18% conversion from view)
                    if random.random() < 0.18:
                        cart_time = event_time + timedelta(seconds=random.randint(20, 60))
                        interactions.append({
                            "interaction_id": f"INT_{uuid.uuid4().hex[:10]}",
                            "user_id": user_id,
                            "product_id": prod_id,
                            "event_type": "add_to_cart",
                            "session_id": session_id,
                            "timestamp": cart_time.isoformat(),
                            "dwell_time_seconds": random.randint(15, 60)
                        })
                        
                        # Purchase conversion (45% from cart)
                        if random.random() < 0.45:
                            purch_time = cart_time + timedelta(seconds=random.randint(40, 180))
                            qty = 1 if random.random() < 0.85 else random.randint(2, 3)
                            purchases.append({
                                "purchase_id": f"PUR_{uuid.uuid4().hex[:10]}",
                                "user_id": user_id,
                                "product_id": prod_id,
                                "quantity": qty,
                                "unit_price": prod["price"],
                                "total_amount": round(prod["price"] * qty, 2),
                                "payment_method": random.choice(["Credit Card", "Apple Pay", "PayPal", "Debit Card"]),
                                "timestamp": purch_time.isoformat()
                            })
                            
                            # Ratings given after purchase (65% chance)
                            if prod_id not in user_rated_products and random.random() < 0.65:
                                user_rated_products.add(prod_id)
                                rate_time = purch_time + timedelta(days=random.randint(1, 7))
                                # Higher rating if matches user's preferred category
                                base_rating = 4.0 if prod["category"] == preferred_cat else 3.2
                                rating_val = float(np.clip(np.random.normal(base_rating, 0.8), 1.0, 5.0))
                                rating_val = round(rating_val * 2) / 2 # Round to nearest 0.5
                                
                                ratings.append({
                                    "rating_id": f"RAT_{uuid.uuid4().hex[:10]}",
                                    "user_id": user_id,
                                    "product_id": prod_id,
                                    "rating": float(rating_val),
                                    "review_text": f"Rated {rating_val}/5. Great {prod['subcategory']} from {prod['brand']}.",
                                    "timestamp": rate_time.isoformat()
                                })

        return {
            "interactions": interactions,
            "purchases": purchases,
            "ratings": ratings
        }
