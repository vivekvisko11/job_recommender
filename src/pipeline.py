# ============================================
#   UPDATED PIPELINE.PY  (NO EMBEDDING BUILDING)
#   Loads existing embeddings only.
#   Full logic preserved exactly as before.
# ============================================

import os
import numpy as np
import re
from math import exp
from rapidfuzz import fuzz

from .database import fetch_all_jobs_from_db, load_jobs_from_csv, fetch_user_by_id
from .embedding_local import generate_user_embedding
from .faiss_index import load_faiss_index, search_index

# ============================================
# PATHS
# ============================================

os.makedirs("data/embeddings", exist_ok=True)

EMBED_DIR = "data/embeddings"
EMBEDDING_PATH = os.path.join(EMBED_DIR, "jobs_embeddings.npy")
IDS_PATH = os.path.join(EMBED_DIR, "job_ids.npy")
METADATA_PATH = os.path.join(EMBED_DIR, "job_metadatas.npy")
INDEX_PATH = os.path.join(EMBED_DIR, "faiss_index.bin")
TITLE_EMB_PATH = os.path.join(EMBED_DIR, "job_title_embs.npy")

# ------------------------------
# LOAD JOB EMBEDDINGS
# ------------------------------
def load_job_embeddings():
    embs = np.load(EMBEDDING_PATH)
    ids = np.load(IDS_PATH)
    metas = np.load(METADATA_PATH, allow_pickle=True)
    title_embs = np.load(TITLE_EMB_PATH, allow_pickle=True)
    index = load_faiss_index(INDEX_PATH)
    return embs, ids, metas, title_embs, index

# ============================================
# HELPERS (UNCHANGED – same logic preserved)
# ============================================

def tokenize(text):
    return set(re.sub(r"[^a-zA-Z0-9 ]+", " ", str(text or "")).lower().strip().split())

def norm_str(x):
    return str(x or "").strip().lower()

def cosine_sim(a, b):
    try:
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        return float(np.dot(a, b) / ((np.linalg.norm(a)+1e-9) * (np.linalg.norm(b)+1e-9)))
    except:
        return 0.0

def embed_text_safe(text):
    try:
        emb = generate_user_embedding({"user_profile": str(text)})
        emb = np.asarray(emb, dtype=np.float32)
        return emb[0] if emb.ndim == 2 else emb
    except:
        return None

def softmax_list(arr):
    exps = [np.exp(x) for x in arr]
    s = sum(exps)
    return [e / s for e in exps]

def top_semantic_title_matches(user_title_emb, job_title_embs, top_n=80):
    sims = []
    for emb in job_title_embs:
        sims.append(cosine_sim(user_title_emb, emb) if emb is not None else 0)
    sims = np.array(sims)
    idx = np.argsort(sims)[::-1][:top_n]
    return list(idx)

def assign_title_priority(title_sim):
    if title_sim >= 0.80: return 0
    if title_sim >= 0.55: return 1
    return 2

# ============================================
# RECOMMENDER (UNCHANGED)
# ============================================

def recommend_jobs_for_user(user_dict, index, job_ids, jobs_df, job_title_embs, top_k=10, faiss_pool=300):

    def norm(x): return str(x or "").strip().lower()

    user_city = norm(user_dict.get("user_city"))
    user_pref_locs = [norm(x) for x in str(user_dict.get("user_job_location") or "").split(",") if x.strip()]
    raw_title = norm(user_dict.get("user_profile"))
    user_title = raw_title
    user_title_emb = embed_text_safe(user_title)

    q_emb = generate_user_embedding(user_dict).astype("float32")
    faiss_idxs, faiss_scores = search_index(index, q_emb, top_k=top_k + faiss_pool)
    faiss_idxs = faiss_idxs[:faiss_pool]

    strong_idx = top_semantic_title_matches(user_title_emb, job_title_embs, top_n=80)

    sims = []
    for emb in job_title_embs:
        sims.append(cosine_sim(user_title_emb, emb) if emb is not None else 0)
    basic_idx = np.argsort(sims)[::-1][:faiss_pool]

    combined_idx = list(set(faiss_idxs) | set(strong_idx) | set(basic_idx))

    pool = []
    for i in combined_idx:
        jid = int(job_ids[i])
        row = jobs_df[jobs_df["job_id"].astype(int) == jid]
        if not row.empty:
            pool.append((jid, row.iloc[0].to_dict(), job_title_embs[i], faiss_scores[min(i, len(faiss_scores)-1)]))

    results = []
    for jid, row0, title_emb, base_score in pool:
        job_title = norm(row0.get("job_title"))
        job_city = norm(row0.get("job_city"))
        user_skills = set(norm(x) for x in str(user_dict.get("user_skills") or "").split(",") if x.strip())
        job_skills = set(norm(x) for x in str(row0.get("job_key_skills") or "").split(",") if x.strip())

        sem = cosine_sim(user_title_emb, title_emb) if title_emb is not None else 0
        fuzzy = fuzz.token_set_ratio(user_title, job_title) / 100.0
        title_sim = 0.6 * sem + 0.4 * fuzzy

        vals = []
        for u in user_skills:
            for j in job_skills:
                r = fuzz.partial_ratio(u, j) / 100.0
                if r > 0.6:
                    vals.append(r)
        skills_sim = sum(vals)/len(vals) if vals else 0

        loc = 1 if job_city == user_city else (0.7 if any(l in job_city for l in user_pref_locs) else 0.1)

        t_s, s_s, l_s = softmax_list([title_sim, skills_sim, loc])
        field_score = 0.55*t_s + 0.30*s_s + 0.10*l_s
        final_score = 0.05*float(base_score) + 0.95*field_score

        out = dict(row0)
        out["_final_score"] = round(final_score, 4)
        out["_priority"] = assign_title_priority(title_sim)

        results.append(out)

    ranked = sorted(results, key=lambda r: (r["_priority"], -r["_final_score"]))
    return ranked[:top_k]


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("🚀 Starting pipeline...\n")

    jobs_df = fetch_all_jobs_from_db()

    embs, job_ids, metas, title_embs, index = load_job_embeddings()

    user = fetch_user_by_id(1246)
    if user:
        results = recommend_jobs_for_user(user, index, job_ids, jobs_df, title_embs)
        for r in results:
            print(r["job_title"], r["job_city"], r["_final_score"])
