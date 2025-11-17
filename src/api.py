from fastapi import FastAPI, HTTPException
import numpy as np
import os

from .pipeline import recommend_jobs_for_user
from .database import fetch_user_by_id, fetch_all_jobs_from_db, load_jobs_from_csv
from .faiss_index import load_faiss_index

app = FastAPI(title="Job Recommendation API", version="1.0")

# ============================================================
# 📂 PATHS (absolute-safe)
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBED_DIR = os.path.join(BASE_DIR, "data", "embeddings")

EMBEDDING_PATH = os.path.join(EMBED_DIR, "jobs_embeddings.npy")
IDS_PATH = os.path.join(EMBED_DIR, "job_ids.npy")
META_PATH = os.path.join(EMBED_DIR, "job_metadatas.npy")
TITLE_EMB_PATH = os.path.join(EMBED_DIR, "job_title_embs.npy")
INDEX_PATH = os.path.join(EMBED_DIR, "faiss_index.bin")

CSV_PATH = os.path.join(BASE_DIR, "data", "jobs_cleaned.csv")

# GLOBALS
index = None
job_ids = None
job_title_embs = None
jobs_df = None


# ============================================================
# 🔄 STARTUP LOADER
# ============================================================
@app.on_event("startup")
def load_resources():
    global index, job_ids, job_title_embs, jobs_df

    print("🔄 Loading FAISS index and job dataset...")

    required_files = [
        INDEX_PATH, EMBEDDING_PATH, IDS_PATH,
        META_PATH, TITLE_EMB_PATH
    ]

    if not all(os.path.exists(p) for p in required_files):
        raise RuntimeError(
            "❌ Embeddings not found. "
            "Run pipeline.py first to generate FAISS + embeddings."
        )

    # Load FAISS index & numpy vectors
    index = load_faiss_index(INDEX_PATH)
    job_ids = np.load(IDS_PATH)
    job_title_embs = np.load(TITLE_EMB_PATH, allow_pickle=True)

    # Load jobs from MySQL, fallback to CSV
    try:
        jobs_df = fetch_all_jobs_from_db()
        print("✅ Loaded jobs from MySQL")
    except Exception as e:
        print(f"⚠️ MySQL error: {e} → using CSV fallback")
        jobs_df = load_jobs_from_csv(CSV_PATH)

    print("✅ API resources loaded successfully!")


# ============================================================
# 🔧 CLEAN JSON VALUES
# ============================================================
def clean_value(v):
    if isinstance(v, (np.float32, np.float64)):
        return float(v)
    if isinstance(v, (np.int32, np.int64)):
        return int(v)
    if isinstance(v, float) and np.isnan(v):
        return ""
    if v is None:
        return ""
    if isinstance(v, str):
        return v.encode("utf-8", "ignore").decode("utf-8", "ignore")
    return v


# ============================================================
# 🎯 RECOMMEND ENDPOINT
# ============================================================
@app.get("/recommend/{user_id}")
def get_recommendations(user_id: int, top_k: int = 10):

    user = fetch_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Pass title embeddings (important for accuracy)
    recs = recommend_jobs_for_user(
        user,
        index,
        job_ids,
        jobs_df,
        job_title_embs,
        top_k=top_k
    )

    clean_results = []
    for r in recs:
        cleaned = {k: clean_value(v) for k, v in r.items()}
        if "_final_score" in cleaned:
            cleaned["final_score"] = cleaned["_final_score"]
        clean_results.append(cleaned)

    return {
        "user_id": user_id,
        "user_name": user.get("user_name"),
        "results": clean_results
    }


# ============================================================
# 🔄 HOT RELOAD ENDPOINT (Used by incremental updater)
# ============================================================
@app.get("/reload")
def reload_resources():
    global index, job_ids, job_title_embs, jobs_df

    print("🔄 Reload request received → refreshing resources...")

    try:
        index = load_faiss_index(INDEX_PATH)
        job_ids = np.load(IDS_PATH)
        job_title_embs = np.load(TITLE_EMB_PATH, allow_pickle=True)

        # Try MySQL first
        try:
            jobs_df = fetch_all_jobs_from_db()
            print("✅ Reloaded jobs from MySQL")
        except Exception as e:
            print(f"⚠️ DB reload failed: {e} → using CSV fallback")
            jobs_df = load_jobs_from_csv(CSV_PATH)

        print("✅ API resources refreshed successfully.")
        return {"status": "success", "message": "Resources reloaded"}

    except Exception as e:
        print(f"❌ Reload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
