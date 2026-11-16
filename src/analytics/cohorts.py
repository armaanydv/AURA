"""
Cohort & Retention Analysis Engine
Builds cohort retention matrices and retention decay curves using SQL and Pandas.
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
import sqlite3

class CohortAnalyzer:
    def __init__(self, db_conn: sqlite3.Connection):
        self.conn = db_conn

    def build_monthly_cohort_matrix(self) -> pd.DataFrame:
        """
        Calculates monthly cohort retention matrix.
        Rows: Signup Month (Cohort)
        Columns: Month Offset (0, 1, 2, 3...)
        Values: Retention Rate (%)
        """
        query = """
        WITH user_cohorts AS (
            SELECT 
                user_id,
                strftime('%Y-%m', signup_date) as cohort_month,
                CAST(substr(signup_date, 1, 4) AS INTEGER) as cohort_year,
                CAST(substr(signup_date, 6, 2) AS INTEGER) as cohort_month_num
            FROM dim_users
        ),
        activity_months AS (
            SELECT 
                user_id,
                strftime('%Y-%m', timestamp) as active_month,
                CAST(substr(timestamp, 1, 4) AS INTEGER) as active_year,
                CAST(substr(timestamp, 6, 2) AS INTEGER) as active_month_num
            FROM fact_interactions
            UNION
            SELECT 
                user_id,
                strftime('%Y-%m', timestamp) as active_month,
                CAST(substr(timestamp, 1, 4) AS INTEGER) as active_year,
                CAST(substr(timestamp, 6, 2) AS INTEGER) as active_month_num
            FROM fact_purchases
        ),
        cohort_activity AS (
            SELECT 
                uc.user_id,
                uc.cohort_month,
                am.active_month,
                (am.active_year - uc.cohort_year) * 12 + (am.active_month_num - uc.cohort_month_num) as period_offset
            FROM user_cohorts uc
            JOIN activity_months am ON uc.user_id = am.user_id
            WHERE (am.active_year - uc.cohort_year) * 12 + (am.active_month_num - uc.cohort_month_num) >= 0
        )
        SELECT 
            cohort_month,
            period_offset,
            COUNT(DISTINCT user_id) as active_users
        FROM cohort_activity
        GROUP BY cohort_month, period_offset
        ORDER BY cohort_month, period_offset
        """
        df = pd.read_sql_query(query, self.conn)
        
        if df.empty:
            return pd.DataFrame()

        # Pivot to Matrix
        cohort_pivot = df.pivot(index="cohort_month", columns="period_offset", values="active_users").fillna(0)
        
        # Calculate retention percentage relative to Period 0 (or cohort size)
        # Period 0 count is the base
        base_size = cohort_pivot[0]
        retention_matrix = cohort_pivot.divide(base_size, axis=0) * 100.0
        return retention_matrix.round(1)
