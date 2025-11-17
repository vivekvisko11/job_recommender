"""
embedding_local.py
-------------------------------------------------------
Handles local embedding generation for job and user data.

✅ Focuses embeddings only on semantic text (title, skills, experience positions, etc.)
✅ Skips job descriptions & numeric fields (salary, dates)
✅ Stores company, city/state, salary, created_at separately as metadata
-------------------------------------------------------
"""

import re
import numpy as np
import warnings
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================
# MODEL LOADING
# =========================================================
MODEL_NAME = "intfloat/e5-large-v2" # Excellent multilingual model for semantic similarity
model = SentenceTransformer(MODEL_NAME)



# =========================================================
#  HELPER FUNCTIONS
# =========================================================
def is_valid_text(value):
    """Return True if the value is a meaningful non-placeholder string."""
    if value is None:
        return False
    s = str(value).strip().lower()
    if s in ["", "0", "none", "nan", "null", "na", "n/a"]:
        return False
    return True


def extract_experience_positions(exp_str):
    """
    Extracts only the 'ExperiencePosition' parts from user_experience_ext.
    Removes company names and date fields.
    """
    if not exp_str:
        return ""
    text = str(exp_str)
    positions = re.findall(r"ExperiencePosition\s+([A-Za-z0-9\/\-\s&\+]+)", text)
    cleaned = [p.strip() for p in positions if p.strip()]
    return ", ".join(dict.fromkeys(cleaned))  # unique and ordered


# =========================================================
# 🧩 JOB EMBEDDING GENERATION
# =========================================================
def generate_job_embeddings(df_jobs):
    """
    Generate semantic embeddings for jobs using only core semantic fields:
    - job_title
    - job_key_skills
    - job_ext_experience

    Excludes job_description and location from the embedding text.
    City/state are stored separately as metadata.
    """
    texts, ids, metadatas = [], [], []

    for _, row in df_jobs.iterrows():
        text_parts = []

        if is_valid_text(row.get("job_title")):
            text_parts.append(f"Job Title: {row['job_title']}")
        if is_valid_text(row.get("job_key_skills")):
            text_parts.append(f"Skills Required: {row['job_key_skills']}")
        if is_valid_text(row.get("job_ext_experience")):
            text_parts.append(f"Experience Required: {row['job_ext_experience']}")

        text = ". ".join(text_parts).strip()
        if not text:
            continue

        texts.append(text)
        ids.append(row["job_id"])

        # Metadata for location/salary/timestamps
        metadatas.append({
            "job_id": row["job_id"],
            "company_id": row.get("company_id"),
            "job_city": row.get("job_city"),
            "job_state": row.get("job_state"),
            "job_minimum_salary": row.get("job_minimum_salary"),
            "job_maximum_salary": row.get("job_maximum_salary"),
            "average_salary": row.get("average_salary"),
            "job_created_at": row.get("job_created_at"),
        })

    print(f"🧠 Encoding {len(texts)} job entries using {MODEL_NAME} (no descriptions or city)...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    print(f"✅ Generated {len(embeddings)} embeddings for jobs.")
    return ids, np.array(embeddings, dtype="float32"), metadatas


# =========================================================
# 🧩 USER EMBEDDING GENERATION
# =========================================================
def generate_user_embedding(user_dict):
    """
    Generate a single user embedding from semantic fields:
    - user_profile
    - user_skills
    - ExperiencePosition (from user_experience_ext)
    - location references are skipped from text (handled via metadata separately)
    """
    text_parts = []

    if is_valid_text(user_dict.get("user_profile")):
        text_parts.append(f"Profile: {user_dict['user_profile']}")
    if is_valid_text(user_dict.get("user_skills")):
        text_parts.append(f"Skills: {user_dict['user_skills']}")

    if is_valid_text(user_dict.get("user_experience_ext")):
        exp_positions = extract_experience_positions(user_dict["user_experience_ext"])
        if exp_positions:
            text_parts.append(f"Experience: {exp_positions}")

    # City/state are excluded here — they’ll be handled separately
    text = ". ".join(text_parts).strip()

    if not text:
        raise ValueError("❌ User text empty or invalid for embedding generation.")

    embedding = model.encode([text], normalize_embeddings=True)
    return np.array(embedding[0], dtype="float32")
