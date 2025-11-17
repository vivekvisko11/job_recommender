# 💼 AI Job Recommendation System

An intelligent job recommendation engine powered by **FAISS**, **Semantic Embeddings**, **FastAPI**, and **Streamlit**.  
It matches users to jobs based on job titles, skills, descriptions, experience, and location — with ultra‑fast FAISS search.

---

## 🚀 Features

### 🔍 AI-Powered Matching
- Semantic job title matching  
- Strong autocorrect for incorrect titles (e.g., *data scienctist → data scientist*)  
- Fuzzy skill similarity  
- Location-based prioritization  
- Weighted scoring combining FAISS + semantic layers  

### ⚡ High-Speed FAISS Search
- Uses **intfloat/e5-large-v2** embeddings  
- FAISS index for vector search  
- Handles **thousands of jobs in milliseconds**

### 🔄 Incremental Embedding Updates
Runs automatically or manually using:
```bash
python -m src.incremental
python -m src.incremental --once
```
Only new rows from DB get embedded and appended to:
- `jobs_embeddings.npy`
- `job_title_embs.npy`
- `job_ids.npy`
- `job_metadatas.npy`
- `faiss_index.bin`

### 🌐 Streamlit Frontend
- Clean UI  
- Enter User ID → get job recommendations instantly  
- View match score, job details, skills, salary, etc.

### 🖥 FastAPI Backend
Main endpoint:
```
GET /recommend/{user_id}?top_k=10
```
Hot reload:
```
GET /reload
```

---

## 📁 Project Structure

```
job_recommender/
│
├── src/
│   ├── api.py               # FastAPI backend
│   ├── app.py               # Streamlit UI
│   ├── pipeline.py          # Full embedding + FAISS builder
│   ├── incremental.py       # Incremental embedding updater
│   ├── database.py          # MySQL connector
│   ├── faiss_index.py       # FAISS load/build helpers
│   ├── embedding_local.py   # Embedding generation
│
├── data/
│   ├── jobs_cleaned.csv     # (ignored)
│   └── embeddings/          # (ignored - stores FAISS + .npy)
│
├── requirements.txt
└── README.md
```

---

## 🛠 Installation

### 1️⃣ Clone the repo
```bash
git clone https://github.com/vivekvisko11/job_recommender.git
cd job_recommender
```

### 2️⃣ Create virtual environment
Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

Mac/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🔧 Setup

### 1️⃣ Prepare MySQL Tables
You need tables like:
- `jobs`
- `users`  

Or modify `database.py` to load CSV files.

### 2️⃣ Build initial embeddings
```bash
python -m src.pipeline
```
This generates:
- `jobs_embeddings.npy`
- `job_ids.npy`
- `job_title_embs.npy`
- `job_metadatas.npy`
- `faiss_index.bin`

Stored inside:
```
data/embeddings/
```

---

## 🔁 Incremental Updates

Run auto updater:
```bash
python -m src.incremental
```

Run one-time update:
```bash
python -m src.incremental --once
```

---

## 🌐 Run FastAPI Server
```bash
uvicorn src.api:app --reload --port 8000
```

Example API call:
```
http://127.0.0.1:8000/recommend/1246?top_k=10
```

---

## 🖥 Run Streamlit App
```bash
streamlit run src/app.py
```

---

## 📦 Git Upload Notes

These **will NOT be uploaded** to GitHub (ignored intentionally):

✔ `venv/`  
✔ `data/embeddings/`  
✔ `.bin` FAISS index  
✔ Raw `.csv` job data  
✔ Any file > 100MB  

---

## 🤝 Contributing
Pull requests are welcome!  

---

## ⭐ Support
If you like this project, give it a ⭐ on GitHub!

