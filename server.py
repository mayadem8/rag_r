import os
import json
import threading
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = "gpt-5.2"

BASE = os.path.dirname(os.path.abspath(__file__))
EMB_PATH = os.path.join(BASE, "embeddings.npz")
META_PATH = os.path.join(BASE, "meta.json")
CHUNKS_TEXT_PATH = os.path.join(BASE, "chunks.jsonl")

TOP_K = 5
MIN_SCORE = 0.45



app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://rag-r.vercel.app",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- globals loaded later ----
READY = False
LOAD_ERROR = None
meta = None
emb = None
texts = None
st_model = None
client = None

@app.get("/")
def root():
    # This makes port detection instant
    return {"status": "ok", "ready": READY, "error": LOAD_ERROR}

@app.get("/health")
def health():
    return {"ready": READY, "error": LOAD_ERROR}

def load_rag():
    global READY, LOAD_ERROR, meta, emb, texts, st_model, client
    try:
        print("Loading RAG system...", flush=True)
        print("BASE:", BASE, flush=True)
        print("FILES:", os.listdir(BASE), flush=True)

        if not os.path.exists(META_PATH):
            raise RuntimeError(f"meta.json not found at {META_PATH}")
        if not os.path.exists(EMB_PATH):
            raise RuntimeError(f"embeddings.npz not found at {EMB_PATH}")
        if not os.path.exists(CHUNKS_TEXT_PATH):
            raise RuntimeError(f"chunks.jsonl not found at {CHUNKS_TEXT_PATH}")

        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)

        emb = np.load(EMB_PATH)["embeddings"]
        model_name = meta["meta"]["model_name"]

        texts = []
        with open(CHUNKS_TEXT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                texts.append(json.loads(line)["text"])

        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is missing in Render env vars")

        # Heavy imports only inside loader (NOT at import time)
        from sentence_transformers import SentenceTransformer
        from openai import OpenAI

        st_model = SentenceTransformer(model_name)
        client = OpenAI(api_key=OPENAI_API_KEY)

        READY = True
        print("✅ Model loaded. Server ready.", flush=True)

    except Exception as e:
        LOAD_ERROR = str(e)
        print("❌ Load failed:", LOAD_ERROR, flush=True)

@app.on_event("startup")
def startup():
    threading.Thread(target=load_rag, daemon=True).start()

class Question(BaseModel):
    question: str

def retrieve(query: str):
    qv = st_model.encode([query]).astype(np.float32)
    qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-12)

    scores = (emb @ qv.T).reshape(-1)
    order = np.argsort(-scores)

    picked = []
    for i in order:
        s = float(scores[i])
        if s < MIN_SCORE:
            break
        picked.append(texts[int(i)])
        if len(picked) >= TOP_K:
            break
    return picked

def ask_gpt(question, chunks):
    context = "\n\n".join(chunks)

    system_prompt = (
        "You are an internal medical assistant.\n"
        "Answer using ONLY the provided CONTEXT.\n"
        "Give a short, clean answer.\n"
        "DO NOT mention sources.\n"
    )

    resp = client.responses.create(
        model=MODEL,
        instructions=system_prompt,
        input=f"CONTEXT:\n{context}\n\nQUESTION: {question}",
        max_output_tokens=200,
    )
    return resp.output_text.strip()

@app.post("/ask")
def ask(q: Question):
    if LOAD_ERROR:
        raise HTTPException(status_code=500, detail=f"Startup load failed: {LOAD_ERROR}")
    if not READY:
        raise HTTPException(status_code=503, detail="Model is still loading. Try again in a moment.")

    chunks = retrieve(q.question)
    if not chunks:
        return {"answer": "No information found."}

    return {"answer": ask_gpt(q.question, chunks)}
