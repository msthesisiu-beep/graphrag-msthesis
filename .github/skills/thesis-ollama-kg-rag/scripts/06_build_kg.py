"""Gate 7: provenance-aware knowledge graph (extract -> normalise/type-check -> independent verification -> NetworkX).

Phases are resumable and use one model at a time (extractor, then verifier):
  python 06_build_kg.py extract [--limit N]
  python 06_build_kg.py verify
  python 06_build_kg.py build
Scope (config kg.scope): 'candidates' = union of BGE top-30 passages for dev+test questions; 'all' = whole corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import pickle
import re
import time
from collections import Counter

import networkx as nx
from tqdm import tqdm

from common import append_jsonl, llm, load_config, load_split, read_jsonl, warmup, work, write_json, write_jsonl
from prompts import (TRIPLE_VERIFY_SCHEMA, kg_extract_messages, kg_extract_schema, triple_verify_messages)
from retrieval import Retriever

ALIAS = re.compile(r"\b([A-Z][A-Za-z\-]+(?:\s+(?:and|of|for|the|to|in|[A-Z][A-Za-z\-]+)){1,8})\s+\(([A-Z][A-Za-z0-9\-/]{1,12})\)")
ABBREV = re.compile(r"[A-Z0-9][A-Za-z0-9\-_/]*")


def scope_passages(cfg: dict, R: Retriever) -> list[str]:
    if cfg["kg"]["scope"] == "all":
        return R.ids
    qs = {q["question_id"]: q for q in read_jsonl(work(cfg, "questions_all.jsonl"))}
    pids: set[str] = set()
    for qid in load_split(cfg, "dev") + load_split(cfg, "test"):
        pids |= {pid for pid, _ in R.search_bge(qs[qid]["question"], cfg["retrieval"]["candidates"])}
    return sorted(pids)


def alias_map(passages: list[dict]) -> dict[str, str]:
    counts: dict[str, Counter] = {}
    for p in passages:
        for long, abbr in ALIAS.findall(p["text"]):
            if sum(c.isupper() for c in abbr) >= 2:
                counts.setdefault(long.lower(), Counter())[abbr] += 1
    return {k: c.most_common(1)[0][0] for k, c in counts.items()}


def canon(name: str, alias: dict[str, str]) -> tuple[str, str]:
    n = re.sub(r"\s+", " ", name).strip().strip(".,;:'\"")
    m = re.fullmatch(r"(.+?)\s*\(([A-Z][A-Za-z0-9\-/]{1,12})\)", n)
    if m:
        return m.group(2), n
    if ABBREV.fullmatch(n) and sum(c.isupper() for c in n) >= 2:
        return n, n
    return alias.get(n.lower(), n.lower()), n


def triple_id(s: str, r: str, o: str, pid: str) -> str:
    return hashlib.sha1(f"{s}|{r}|{o}|{pid}".encode()).hexdigest()[:12]


def extract(cfg: dict, limit: int | None) -> None:
    R = Retriever(cfg)
    k = cfg["kg"]
    out = work(cfg, "kg", "extract_raw.jsonl")
    done = {r["passage_id"] for r in read_jsonl(out) if r["status"] == "ok"}
    scope = scope_passages(cfg, R)
    write_json(work(cfg, "kg", "scope.json"), {"scope": k["scope"], "passages": len(scope)})
    todo = [pid for pid in scope if pid not in done][:limit]
    print(f"KG scope {len(scope)} passages, {len(todo)} to extract with {cfg['models']['kg_extractor']}")
    if not todo:
        return
    R.release_models()
    schema = kg_extract_schema(k["entity_types"], k["relations"], k["max_triples_per_passage"])
    warmup(cfg, "kg_extractor")
    for pid in tqdm(todo):
        t0 = time.perf_counter()
        try:
            r = llm(cfg, "kg_extractor", kg_extract_messages(R.by_id[pid], k["entity_types"], k["relations"],
                                                             k["max_triples_per_passage"]), schema=schema, max_tokens=1500)
            if r["json"] is None:
                raise RuntimeError("unparsable JSON")
            rec = {"passage_id": pid, "status": "ok", "triples": r["json"].get("triples", [])[: k["max_triples_per_passage"]],
                   "in_tokens": r["in_tokens"], "out_tokens": r["out_tokens"]}
        except Exception as e:
            rec = {"passage_id": pid, "status": "error", "error": str(e)[:300]}
        rec["seconds"] = round(time.perf_counter() - t0, 2)
        append_jsonl(out, [rec])
    latest = {r["passage_id"]: r for r in read_jsonl(out)}
    errors = [pid for pid in todo if latest.get(pid, {}).get("status") != "ok"]
    if errors:
        raise SystemExit(f"{len(errors)} KG extraction errors remain; rerun extract to retry them.")


def normalise(cfg: dict, R: Retriever) -> list[dict]:
    k = cfg["kg"]
    raw = {}
    for r in read_jsonl(work(cfg, "kg", "extract_raw.jsonl")):
        if r["status"] == "ok":
            raw[r["passage_id"]] = r  # last ok record wins
    alias = alias_map([R.by_id[pid] for pid in raw])
    out, seen = [], set()
    for pid, rec in raw.items():
        p = R.by_id[pid]
        low = p["text"].lower()
        for t in rec["triples"]:
            s_node, s_name = canon(t.get("subject", ""), alias)
            o_node, o_name = canon(t.get("object", ""), alias)
            tid = triple_id(s_node, t.get("relation", ""), o_node, pid)
            if tid in seen:
                continue
            seen.add(tid)
            grounded = s_name.lower() in low and o_name.lower() in low
            type_ok = (t.get("relation") in k["relations"] and t.get("subject_type") in k["entity_types"]
                       and t.get("object_type") in k["entity_types"] and s_node and o_node and s_node != o_node
                       and len(s_name) <= 80 and len(o_name) <= 80 and grounded)
            out.append({"triple_id": tid, "subject": s_name, "relation": t.get("relation"), "object": o_name,
                        "s_node": s_node, "o_node": o_node, "s_type": t.get("subject_type"), "o_type": t.get("object_type"),
                        "passage_id": pid, "spec": p["spec"], "version": p["version"], "clause": p["clause"],
                        "type_ok": bool(type_ok), "verified": False})
    return out


def verify(cfg: dict) -> None:
    R = Retriever(cfg)
    triples = normalise(cfg, R)
    vout = work(cfg, "kg", "verify_raw.jsonl")
    verdicts: dict[str, bool] = {}
    for r in read_jsonl(vout):
        if r["status"] == "ok":
            verdicts.update(r["results"])
    by_pid: dict[str, list[dict]] = {}
    for t in triples:
        if t["type_ok"] and t["triple_id"] not in verdicts:
            by_pid.setdefault(t["passage_id"], []).append(t)
    print(f"{sum(t['type_ok'] for t in triples)} type-checked triples; {sum(map(len, by_pid.values()))} to verify "
          f"with {cfg['models']['verifier']}")
    if by_pid:
        warmup(cfg, "verifier")
    for pid, ts in tqdm(by_pid.items()):
        for i in range(0, len(ts), 10):
            batch = ts[i:i + 10]
            t0 = time.perf_counter()
            try:
                r = llm(cfg, "verifier", triple_verify_messages(R.by_id[pid], batch), schema=TRIPLE_VERIFY_SCHEMA, max_tokens=400)
                res = {x["index"]: bool(x["entailed"]) for x in (r["json"] or {}).get("results", []) if isinstance(x.get("index"), int)}
                if len(res) < len(batch):
                    raise RuntimeError("incomplete verifier output")
                append_jsonl(vout, [{"passage_id": pid, "status": "ok",
                                     "results": {t["triple_id"]: res.get(j, False) for j, t in enumerate(batch)},
                                     "seconds": round(time.perf_counter() - t0, 3),
                                     "in_tokens": r["in_tokens"], "out_tokens": r["out_tokens"]}])
            except Exception as e:
                append_jsonl(vout, [{"passage_id": pid, "status": "error", "error": str(e)[:300],
                                     "seconds": round(time.perf_counter() - t0, 3)}])
    for r in read_jsonl(vout):
        if r["status"] == "ok":
            verdicts.update(r["results"])
    missing = [t["triple_id"] for t in triples if t["type_ok"] and t["triple_id"] not in verdicts]
    if missing:
        raise SystemExit(f"{len(missing)} KG triples remain unverified; rerun verify to retry them.")
    for t in triples:
        t["verified"] = bool(t["type_ok"] and verdicts.get(t["triple_id"], False))
    write_jsonl(work(cfg, "kg", "triples_all.jsonl"), triples)
    print(f"verified {sum(t['verified'] for t in triples)} / {len(triples)} extracted triples")


def build(cfg: dict) -> None:
    triples = read_jsonl(work(cfg, "kg", "triples_all.jsonl"))
    G = nx.MultiDiGraph()
    type_votes: dict[str, Counter] = {}
    for t in triples:
        if not t["verified"]:
            continue
        for node, name, typ in ((t["s_node"], t["subject"], t["s_type"]), (t["o_node"], t["object"], t["o_type"])):
            if node not in G:
                G.add_node(node, names=set())
            G.nodes[node]["names"].add(name)
            type_votes.setdefault(node, Counter())[typ] += 1
        G.add_edge(t["s_node"], t["o_node"], key=t["triple_id"], triple_id=t["triple_id"], relation=t["relation"],
                   s_name=t["subject"], o_name=t["object"], passage_id=t["passage_id"], spec=t["spec"],
                   version=t["version"], clause=t["clause"], verified=True)
    for node, votes in type_votes.items():
        G.nodes[node]["type"] = votes.most_common(1)[0][0]
        G.nodes[node]["names"] = sorted(G.nodes[node]["names"])
    with work(cfg, "kg", "graph.pkl").open("wb") as f:
        pickle.dump(G, f)
    extract_rows = [r for r in read_jsonl(work(cfg, "kg", "extract_raw.jsonl")) if r["status"] == "ok"]
    stats = {
        "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
        "relations": Counter(d["relation"] for *_, d in G.edges(data=True)).most_common(),
        "node_types": Counter(d.get("type") for _, d in G.nodes(data=True)).most_common(),
        "extracted": len(triples), "type_ok": sum(t["type_ok"] for t in triples),
        "verified": sum(t["verified"] for t in triples),
        "source_passages": len({d["passage_id"] for *_, d in G.edges(data=True)}),
        "offline_cost": {"extract_seconds": round(sum(r["seconds"] for r in extract_rows), 1),
                         "extract_in_tokens": sum(r.get("in_tokens", 0) for r in extract_rows),
                         "extract_out_tokens": sum(r.get("out_tokens", 0) for r in extract_rows)},
        "graph_pickle_mb": round(work(cfg, "kg", "graph.pkl").stat().st_size / 1e6, 2),
    }
    write_json(work(cfg, "kg", "kg_stats.json"), stats)
    print({k: v for k, v in stats.items() if k not in ("relations", "node_types")})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["extract", "verify", "build"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config()
    if args.phase == "extract":
        extract(cfg, args.limit)
    elif args.phase == "verify":
        verify(cfg)
    else:
        build(cfg)


if __name__ == "__main__":
    main()
