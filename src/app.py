import os
import streamlit as st
import numpy as np
import pandas as pd
import requests   # calling FastAPI backend

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Job Recommender 💼",
    page_icon="💼",
    layout="wide",
)

API_URL = "http://127.0.0.1:8000"

# =========================================================
# UI / STYLING
# =========================================================
st.markdown(
    """
    <style>
        .title {
            font-size: 42px;
            font-weight: 700;
            background: -webkit-linear-gradient(45deg, #0072ff, #00c6ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtext {
            color: #777;
            font-size: 18px;
            margin-bottom: 20px;
        }
        .job-card {
            background: rgba(255, 255, 255, 0.05);
            padding: 18px 22px;
            border-radius: 14px;
            margin-bottom: 16px;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.08);
            transition: all 0.2s ease-in-out;
        }
        .job-card:hover {
            background: rgba(255,255,255,0.1);
            box-shadow: 0 0 18px rgba(0,0,0,0.12);
        }
        .job-title {
            font-size: 20px;
            font-weight: 600;
            color: #00c6ff;
        }
        .job-location {
            color: #999;
            font-size: 15px;
        }
        .salary {
            color: #00ffb0;
            font-weight: 500;
        }
        .match {
            font-weight: 600;
            color: #ffa600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">💼 Smart Job Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Discover your perfect career match powered by AI & FAISS embeddings.</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Filters")
top_k = st.sidebar.slider("Number of recommendations", 5, 20, 10)

# =========================================================
# USER INPUT
# =========================================================
st.divider()
st.subheader("🔍 Get Personalized Job Recommendations")
user_id_input = st.text_input("👤 Enter User ID", placeholder="e.g. 777")

if st.button("🚀 Find Jobs"):

    if not user_id_input.strip():
        st.warning("⚠ Enter a valid user ID.")
        st.stop()

    user_id = int(user_id_input.strip())

    with st.spinner("🔍 Contacting AI Recommendation Engine..."):
        response = requests.get(f"{API_URL}/recommend/{user_id}?top_k={top_k}")

    if response.status_code != 200:
        st.error(f"❌ API Error: {response.text}")
        st.stop()

    data = response.json()
    results = data["results"]

    if len(results) == 0:
        st.info("No matching jobs found.")
        st.stop()

    st.success(f"🎯 Found {len(results)} matching jobs for User {user_id}")

    # display jobs
    for job in results:

        st.markdown(
            f"""
            <div class="job-card">
                <div class="job-title">{job.get('job_title')}</div>
                <div class="job-location">📍 {job.get('job_city')}</div>
                <div class="match">🔥 Match Score: {job.get('final_score', 0):.2f}</div>
                <div class="match">⭐ Priority Level: {job.get('_priority', '')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("📄 View Job Details"):
            st.markdown(f"**Job ID:** {job.get('job_id')}")
            st.markdown(f"**Skills:** {job.get('job_key_skills')}")
            st.markdown(f"**Salary:** {job.get('job_minimum_salary')} - {job.get('job_maximum_salary')}")
            st.markdown(f"**State:** {job.get('job_state')}")
            st.markdown(f"**Description:** {job.get('job_description')}")
            st.markdown("---")
