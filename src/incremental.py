"""
📈 Incremental Embedding Builder + Updater
First run → full embedding build
Next runs → only new DB rows
"""

import os
import sys
import time
import numpy as np
import faiss
import requests

from .database import fetch_all_jobs_from_db
from .embedding_local import generate_job_embeddings, generate_user_embedding
from .faiss_index import load_faiss_index

# Paths
ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(ROOT, ".."))
EMBED_DIR = os.path.join(BASE, "data", "embeddings")

os.makedirs(EMBED_DIR, exist_ok=True)

EMBED_PATH = os.path.join(EMBED_DIR, "jobs_embeddings.npy")
IDS_PATH = os.path.join(EMBED_DIR, "job_ids.npy")
META_PATH = os.path.join(EMBED_DIR, "job_metadatas.npy")
TITLE_EMB_PATH = os.path.join(EMBED_DIR, "job_title_embs.npy")
INDEX_PATH = os.path.join(EMBED_DIR, "faiss_index.bin")


# ==========================
# SAFE Title Embedding
# ==========================
def generate_title_embeddings(job_titles):
    """
    Safely embed job titles without crashing:
    - Blank titles → store None
    - Invalid text → store None
    """
    out = []

    for t in job_titles:
        txt = str(t or "").strip()

        # If title empty or missing → store None
        if txt == "":
            out.append(None)
            continue

        try:
            e = generate_user_embedding({"user_profile": txt})
            e = np.array(e, dtype=np.float32)
            if e.ndim == 2:
                e = e[0]
            out.append(e)
        except Exception:
            out.append(None)

    return np.array(out, dtype=object)


# ==========================
# FULL BUILD (first time)
# ==========================
def full_build():
    df = fetch_all_jobs_from_db()
    df["job_id"] = df["job_id"].astype(int)

    print(f"🔨 Full build: {len(df)} jobs")

    job_ids = df["job_id"].values
    _, embs, metas = generate_job_embeddings(df)

    # SAFE title embeddings
    title_embs = generate_title_embeddings(df["job_title"].tolist())

    # Build FAISS
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    # Save all 5 files
    np.save(EMBED_PATH, embs)
    np.save(IDS_PATH, job_ids)
    np.save(META_PATH, metas, allow_pickle=True)
    np.save(TITLE_EMB_PATH, title_embs, allow_pickle=True)
    faiss.write_index(index, INDEX_PATH)

    print("✅ Full embedding build complete.\n")


# ==========================
# INCREMENTAL UPDATE
# ==========================
def incremental_update():

    embs = np.load(EMBED_PATH)
    ids = np.load(IDS_PATH)
    metas = np.load(META_PATH, allow_pickle=True)
    title_embs = np.load(TITLE_EMB_PATH, allow_pickle=True)
    index = load_faiss_index(INDEX_PATH)

    df = fetch_all_jobs_from_db()
    df["job_id"] = df["job_id"].astype(int)

    cached_ids = set(ids.tolist())
    db_ids = set(df["job_id"].tolist())

    new_ids = sorted(list(db_ids - cached_ids))

    if not new_ids:
        print("😴 No new jobs.")
        return

    print(f"🆕 {len(new_ids)} new jobs found.")

    df_new = df[df["job_id"].isin(new_ids)]

    # Original job_ids
    new_ids_array = df_new["job_id"].values

    # Generate embeddings
    _, new_embs, new_metas = generate_job_embeddings(df_new)

    # SAFE title embeddings
    new_titles = generate_title_embeddings(df_new["job_title"].tolist())

    # Add to FAISS
    faiss.normalize_L2(new_embs)
    index.add(new_embs)

    # Append to caches
    embs = np.vstack([embs, new_embs])
    ids = np.concatenate([ids, new_ids_array])
    metas = np.concatenate([metas, new_metas])
    title_embs = np.concatenate([title_embs, new_titles])

    # Save updated files
    np.save(EMBED_PATH, embs)
    np.save(IDS_PATH, ids)
    np.save(META_PATH, metas, allow_pickle=True)
    np.save(TITLE_EMB_PATH, title_embs, allow_pickle=True)
    faiss.write_index(index, INDEX_PATH)

    print("📦 Incremental update complete.\n")


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":

    # If ANY embedding file missing → FULL BUILD
    if not all(os.path.exists(p) for p in
               [EMBED_PATH, IDS_PATH, META_PATH, TITLE_EMB_PATH, INDEX_PATH]):
        full_build()
    else:
        incremental_update()
