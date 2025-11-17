# db.py
import os
import pandas as pd
from dotenv import load_dotenv
import mysql.connector

load_dotenv()

HOST = os.getenv("MYSQL_HOST", "localhost")
USER = os.getenv("MYSQL_USER", "root")
PASSWORD = os.getenv("MYSQL_PASSWORD", "SQL@1234$")
DB = os.getenv("MYSQL_DB", "job_recommender")

def fetch_all_jobs_from_db():
    try:
        conn = mysql.connector.connect(
            host=HOST, user=USER, password=PASSWORD, database=DB
        )
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                job_id, company_id, job_title, job_key_skills,
                job_description, job_minimum_salary, job_maximum_salary,
                job_city, job_state, job_ext_experience, job_created_at
            FROM jobs
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        cols = [
            "job_id", "company_id", "job_title", "job_key_skills",
            "job_description", "job_minimum_salary", "job_maximum_salary",
            "job_city", "job_state", "job_ext_experience", "job_created_at"
        ]

        df = pd.DataFrame(rows, columns=cols)
        df["average_salary"] = (
            pd.to_numeric(df["job_minimum_salary"], errors="coerce").fillna(0) +
            pd.to_numeric(df["job_maximum_salary"], errors="coerce").fillna(0)
        ) / 2
        return df

    except mysql.connector.Error as err:
        print(f"⚠️ Database fetch failed: {err}")
        return None
def fetch_user_by_id(user_id):
    try:
        conn = mysql.connector.connect(
            host=HOST, user=USER, password=PASSWORD, database=DB
        )
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                user_id, user_name, user_profile, user_skills,
                user_experience_ext, user_city, user_state, user_job_location
            FROM users
            WHERE user_id = %s
        """, (user_id,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if not row:
            return None

        cols = [
            "user_id", "user_name", "user_profile", "user_skills",
            "user_experience_ext", "user_city", "user_state", "user_job_location"
        ]
        return dict(zip(cols, row))

    except mysql.connector.Error as err:
        print(f"⚠️ User fetch failed: {err}")
        return None


# Fallback helper to read CSV if DB not used
def load_jobs_from_csv(path="data/jobs_cleaned.csv"):
    df = pd.read_csv(path, dtype=str)
    # ensure numeric salary columns if exist
    for c in ["job_minimum_salary", "job_maximum_salary"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["average_salary"] = ((df.get("job_minimum_salary", 0).fillna(0) + df.get("job_maximum_salary", 0).fillna(0)) / 2)
    return df

# ✅ Compatibility alias for pipeline.py
def fetch_all_jobs():
    """Wrapper for backward compatibility."""
    return load_jobs_from_csv()

