import json
import os
import threading
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

BASE = os.path.dirname(os.path.abspath(__file__))

# ✅ Use the new OpenAI-embedded files
EMB_PATH = os.path.join(BASE, "embeddings_openai.npz")
META_PATH = os.path.join(BASE, "meta_openai.json")
CHUNKS_TEXT_PATH = os.path.join(BASE, "chunks.jsonl")

TOP_K = 8
MIN_SCORE = 0.25

app = FastAPI()

# ---- CORS (so Vercel can call Render) ----
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://rag-r.vercel.app",
        "https://rag-cgyabkpay-mayas-projects-549f9e33.vercel.app",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- globals loaded after startup ----
READY = False
LOAD_ERROR = None
emb = None
texts = None
client = None


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)


@app.get("/")
def root():
    return {"status": "ok", "ready": READY, "error": LOAD_ERROR}


@app.get("/health")
def health():
    return {"ready": READY, "error": LOAD_ERROR}


def load_rag():
    global READY, LOAD_ERROR, emb, texts, client
    try:
        print("Loading RAG system (OpenAI embeddings)...", flush=True)

        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is missing in Render environment variables.")

        if not os.path.exists(EMB_PATH):
            raise RuntimeError(f"Missing embeddings file: {EMB_PATH}")
        if not os.path.exists(META_PATH):
            raise RuntimeError(f"Missing meta file: {META_PATH}")
        if not os.path.exists(CHUNKS_TEXT_PATH):
            raise RuntimeError(f"Missing chunks file: {CHUNKS_TEXT_PATH}")

        # Load and normalize embeddings once
        emb_raw = np.load(EMB_PATH)["embeddings"].astype(np.float32)
        emb = _l2_normalize(emb_raw)

        # Load texts
        texts = []
        with open(CHUNKS_TEXT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                texts.append(json.loads(line)["text"])

        # Optional: verify embed model matches meta
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta_model = meta.get("meta", {}).get("model_name", "")
        if meta_model and meta_model != EMBED_MODEL:
            print(f"⚠️ meta_openai.json model_name={meta_model} but OPENAI_EMBED_MODEL={EMBED_MODEL}", flush=True)

        client = OpenAI(api_key=OPENAI_API_KEY)

        READY = True
        print("✅ Server ready.", flush=True)

    except Exception as e:
        LOAD_ERROR = str(e)
        print("❌ Load failed:", LOAD_ERROR, flush=True)


@app.on_event("startup")
def startup():
    threading.Thread(target=load_rag, daemon=True).start()


class Question(BaseModel):
    question: str


def embed_query(text: str) -> np.ndarray:
    r = client.embeddings.create(model=EMBED_MODEL, input=text)
    vec = np.array([r.data[0].embedding], dtype=np.float32)
    return _l2_normalize(vec)


def retrieve(query: str):
    qv = embed_query(query)  # (1, d)
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


def ask_gpt(question: str, chunks: list[str]) -> str:
    context = "\n\n".join(chunks)

    system_prompt = (
    "You are a helpful medical assistant.\n"
    "Answer ONLY using the provided CONTEXT.\n"
    "IMPORTANT RULES:\n"
    "- NEVER mention or hint at CONTEXT, 'your text', documents, sources, retrieval, or that you used provided text.\n"
    "- Do NOT say phrases like: 'from the text', 'from your document', 'based on context'.\n"
    "- Output plain text only (NO markdown). Do not use **, #, bullets with markdown.\n"
    "- Keep it short and clear.\n"
    "Language: reply in the user's language\n"
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
        raise HTTPException(status_code=503, detail="Server is waking up. Try again in a moment.")

    chunks = retrieve(q.question)
    if not chunks:
        return {"answer": "No information found."}

    return {"answer": ask_gpt(q.question, chunks)}
