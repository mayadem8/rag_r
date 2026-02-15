import os, json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

BASE = os.path.dirname(os.path.abspath(__file__))

# 🔵 ORIGINAL FILES (read from)
META_PATH = os.path.join(BASE, "meta.json")
CHUNKS_TEXT_PATH = os.path.join(BASE, "chunks.jsonl")

# 🟢 NEW FILES (write to)
OUT_EMB_PATH = os.path.join(BASE, "embeddings_openai.npz")
OUT_META_PATH = os.path.join(BASE, "meta_openai.json")

EMBED_MODEL = "text-embedding-3-small"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- Load texts ----------
texts = []
with open(CHUNKS_TEXT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        texts.append(json.loads(line)["text"])

# ---------- Create embeddings ----------
embs = []
batch_size = 64

for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    r = client.embeddings.create(model=EMBED_MODEL, input=batch)
    batch_vecs = [d.embedding for d in r.data]
    embs.extend(batch_vecs)
    print(f"embedded {min(i+batch_size, len(texts))}/{len(texts)}")

emb = np.array(embs, dtype=np.float32)
np.savez_compressed(OUT_EMB_PATH, embeddings=emb)

# ---------- Create new meta file ----------
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

meta["meta"]["model_name"] = EMBED_MODEL

with open(OUT_META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("✅ Saved:", OUT_EMB_PATH)
print("✅ Saved:", OUT_META_PATH)
