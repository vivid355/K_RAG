# -*- coding: utf-8 -*-
"""Diagnose why all similarity scores come out identical."""
import hashlib
from pathlib import Path

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

BASE_DIR = Path(__file__).parent
QDRANT_PATH = BASE_DIR / "qdrant_db"
COLLECTION = "my_rag_stage1"
DENSE_MODEL = "Qwen/Qwen3-Embedding-0.6B"

embeddings = HuggingFaceEmbeddings(model_name=DENSE_MODEL)
client = QdrantClient(path=str(QDRANT_PATH))

print("collection_exists:", client.collection_exists(COLLECTION))
print("count:", client.count(COLLECTION).count)

# 1) pull every point: id -> vector + payload snippet
records = client.scroll(
    collection_name=COLLECTION,
    with_vectors=True,
    with_payload=True,
    limit=10000,
)[0]
print("scroll returned:", len(records))

vec_by_id = {}
text_by_id = {}
for r in records:
    vec_by_id[r.id] = np.asarray(r.vector, dtype=np.float64)
    src = (r.payload or {}).get("source", "?")
    txt = (r.payload or {}).get("page_content", "")[:30]
    text_by_id[r.id] = f"{src}|{txt}"

# 2) how many DISTINCT vectors are stored?
def h(v):
    return hashlib.sha256(np.ascontiguousarray(v)).hexdigest()[:16]

hash_groups = {}
for pid, v in vec_by_id.items():
    hash_groups.setdefault(h(v), []).append(pid)
print("total points:", len(vec_by_id), " distinct vectors:", len(hash_groups))
for hh, ids in list(hash_groups.items())[:12]:
    print(f"  vec_hash={hh}  n={len(ids)}  e.g. {text_by_id.get(ids[0])}")

# 3) re-embed the query, compute real cosine vs each unique stored vector
q = embeddings.embed_query("向日葵")
qv = np.asarray(q, dtype=np.float64)
qv = qv / np.linalg.norm(qv)
print("\nreal cosine scores (unique by value):")
uniq_scores = set()
for hh, ids in hash_groups.items():
    v = vec_by_id[ids[0]]
    v = v / np.linalg.norm(v)
    uniq_scores.add(round(float(qv @ v), 6))
print(sorted(uniq_scores))
