"""Build the experimental BGE/FAISS index plus the independent BM25 and E5 pooling indexes.

Outputs work/index/{bge,e5}.faiss, bm25.pkl, index_stats.json (offline cost). Usage: python 02_build_indexes.py
"""
from __future__ import annotations

import os
import pickle
import time

import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from common import load_config, passage_text, read_jsonl, work, write_json
from retrieval import bm25_tokenize

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")


def dense_index(cfg: dict, model_key: str, texts: list[str], name: str) -> dict:
    t0 = time.perf_counter()
    dev = cfg["encoders"]["device"]
    model = SentenceTransformer(cfg["encoders"][model_key], device=dev)
    emb = model.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True,
                       show_progress_bar=True).astype(np.float32)
    index = faiss.IndexFlatIP(emb.shape[1])  # exact cosine search: no ANN approximation error
    index.add(emb)
    faiss.write_index(index, str(work(cfg, "index", f"{name}.faiss")))
    del model
    if dev == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return {"model": cfg["encoders"][model_key], "dim": int(emb.shape[1]), "seconds": round(time.perf_counter() - t0, 1)}


def main() -> None:
    cfg = load_config()
    passages = read_jsonl(work(cfg, "passages.jsonl"))
    if not passages:
        raise SystemExit("Run 01_prepare_data.py first.")
    stats = {"passages": len(passages), "device": cfg["encoders"]["device"]}

    stats["bge"] = dense_index(cfg, "embedder", [passage_text(p) for p in passages], "bge")
    stats["e5"] = dense_index(cfg, "pool_dense", ["passage: " + passage_text(p) for p in passages], "e5")

    t0 = time.perf_counter()
    bm25 = BM25Okapi([bm25_tokenize(passage_text(p)) for p in passages])
    with work(cfg, "index", "bm25.pkl").open("wb") as f:
        pickle.dump(bm25, f)
    stats["bm25"] = {"seconds": round(time.perf_counter() - t0, 1)}

    write_json(work(cfg, "index", "index_stats.json"), stats)
    print(stats)


if __name__ == "__main__":
    main()
