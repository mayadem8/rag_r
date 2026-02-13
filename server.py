import json
import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from openai import OpenAI
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

print("Loading RAG system...")

# ---------- Load everything ONCE ----------
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

emb = np.load(EMB_PATH)["embeddings"]
chunks_meta = meta["chunks"]
model_name = meta["meta"]["model_name"]

texts = []
with open(CHUNKS_TEXT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        texts.append(json.loads(line)["text"])

st_model = SentenceTransformer(model_name)
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env or environment variables.")
client = OpenAI(api_key=OPENAI_API_KEY)

print("✅ Model loaded. Server ready.")

# ---------- Request schema ----------
class Question(BaseModel):
    question: str


# ---------- Retrieval ----------
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


# ---------- GPT ----------
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


# ---------- Endpoint ----------
@app.post("/ask")
def ask(q: Question):
    try:
        chunks = retrieve(q.question)

        if not chunks:
            return {"answer": "No information found."}

        answer = ask_gpt(q.question, chunks)
        return {"answer": answer}
    except Exception as err:
        print(f"/ask failed: {err}")
        raise HTTPException(status_code=500, detail=str(err))


