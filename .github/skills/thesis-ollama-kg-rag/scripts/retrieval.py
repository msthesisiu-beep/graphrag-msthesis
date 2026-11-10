"""Retrieval (BGE/FAISS, E5, BM25, cross-encoder) and knowledge-graph lookup utilities."""
from __future__ import annotations

import pickle
import re
from itertools import chain

import numpy as np

from common import free_memory, passage_text, read_jsonl, work

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_BM25_TOKEN = re.compile(r"[a-z0-9]+(?:[\-_.][a-z0-9]+)*")


def bm25_tokenize(text: str) -> list[str]:
    return _BM25_TOKEN.findall(text.lower())


class Retriever:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.passages = read_jsonl(work(cfg, "passages.jsonl"))
        self.by_id = {p["id"]: p for p in self.passages}
        self.ids = [p["id"] for p in self.passages]
        self.device = cfg["encoders"]["device"]
        self._models: dict = {}
        self._indexes: dict = {}
        self._bm25 = None

    def release_models(self) -> None:
        self._models.clear()
        free_memory()

    # -------- lazy loading (keeps memory low when a stage needs only one model)
    def _st(self, key: str):
        if key not in self._models:
            from sentence_transformers import SentenceTransformer
            self._models[key] = SentenceTransformer(self.cfg["encoders"][key], device=self.device)
        return self._models[key]

    def _ce(self):
        if "reranker" not in self._models:
            from sentence_transformers import CrossEncoder
            self._models["reranker"] = CrossEncoder(self.cfg["encoders"]["reranker"], device=self.device)
        return self._models["reranker"]

    def _index(self, name: str):
        if name not in self._indexes:
            import faiss
            self._indexes[name] = faiss.read_index(str(work(self.cfg, "index", f"{name}.faiss")))
        return self._indexes[name]

    def _search(self, index_name: str, model_key: str, text: str, k: int):
        q = self._st(model_key).encode([text], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        scores, idx = self._index(index_name).search(q, k)
        return [(self.ids[i], float(s)) for s, i in zip(scores[0], idx[0]) if i >= 0]

    # -------- public API
    def search_bge(self, query: str, k: int):
        return self._search("bge", "embedder", BGE_QUERY_PREFIX + query, k)

    def search_e5(self, query: str, k: int):
        return self._search("e5", "pool_dense", "query: " + query, k)

    def search_bm25(self, query: str, k: int):
        if self._bm25 is None:
            with work(self.cfg, "index", "bm25.pkl").open("rb") as f:
                self._bm25 = pickle.load(f)
        scores = self._bm25.get_scores(bm25_tokenize(query))
        top = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in top if scores[i] > 0]

    def rerank(self, query: str, items: list[tuple[str, str]]) -> list[tuple[str, float]]:
        """items: (item_id, text). Returns (item_id, score) sorted by score desc."""
        if not items:
            return []
        scores = self._ce().predict([(query, t) for _, t in items], batch_size=16, show_progress_bar=False)
        return sorted(((i, float(s)) for (i, _), s in zip(items, scores)), key=lambda x: -x[1])

    def embed(self, texts: list[str], query: bool = False) -> np.ndarray:
        if not texts:
            return np.empty((0, 768), dtype=np.float32)
        prefix = BGE_QUERY_PREFIX if query else ""
        res = self._st("embedder").encode([prefix + t for t in texts], normalize_embeddings=True,
                                          convert_to_numpy=True, batch_size=32, show_progress_bar=False)
        return np.asarray(res, dtype=np.float32)

    def ptext(self, pid: str) -> str:
        return passage_text(self.by_id[pid])


def rrf(rankings: list[list[tuple[str, float]]], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (pid, _) in enumerate(ranking, start=1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
    return [pid for pid, _ in sorted(scores.items(), key=lambda x: -x[1])]


# ---------------------------------------------------------------- knowledge graph
def load_graph(cfg: dict):
    with work(cfg, "kg", "graph.pkl").open("rb") as f:
        return pickle.load(f)


def render_triple(label: str, d: dict) -> str:
    return (f"[{label}] {d['s_name']} --{d['relation']}--> {d['o_name']} "
            f"(source: TS {d['spec']} {d['version']} clause {d['clause']})")


class Linker:
    """Exact surface-form entity linking: abbreviations case-sensitive, long names case-insensitive."""

    _TOK = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_/]*")

    def __init__(self, G, max_ngram: int = 8):
        self.case: dict[str, set] = {}
        self.lower: dict[str, set] = {}
        for node, d in G.nodes(data=True):
            for s in {node, *d.get("names", [])}:
                toks = self._TOK.findall(s)
                if not toks or len(toks) > max_ngram:
                    continue
                key = " ".join(toks)
                if re.fullmatch(r"[A-Z0-9][A-Z0-9\-_/]{1,}[a-z]?", key):
                    self.case.setdefault(key, set()).add(node)
                elif len(key) >= 4:
                    self.lower.setdefault(key.lower(), set()).add(node)
        self.max_ngram = max_ngram

    def link(self, text: str) -> set:
        toks = self._TOK.findall(text)
        found: set = set()
        for i in range(len(toks)):
            for n in range(1, self.max_ngram + 1):
                if i + n > len(toks):
                    break
                key = " ".join(toks[i:i + n])
                found |= self.case.get(key, set())
                found |= self.lower.get(key.lower(), set())
        return found


def neighbourhood_edges(G, seeds: set, hops: int, valid_pids: set) -> list[tuple[int, dict]]:
    """Edges reachable within `hops` (direction ignored), BFS order, provenance-filtered."""
    seen_nodes, frontier, seen_edges, out = set(seeds), set(seeds), set(), []
    for h in range(1, hops + 1):
        nxt = set()
        for n in sorted(frontier):
            if n not in G:
                continue
            for u, v, key, d in chain(G.out_edges(n, keys=True, data=True), G.in_edges(n, keys=True, data=True)):
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                if d.get("passage_id") in valid_pids and d.get("verified", False):
                    out.append((h, d))
                other = v if u == n else u
                if other not in seen_nodes:
                    nxt.add(other)
        seen_nodes |= nxt
        frontier = nxt
    return out
