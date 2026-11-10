"""Gate 10: per-question metrics (retrieval, citation, graph evidence) + RAGAS with the local evaluator model.

  python 08_evaluate.py --split test                 # deterministic metrics only
  python 08_evaluate.py --split test --ragas         # + RAGAS (faithfulness full & passage-only, relevancy, ctx P/R)
  python 08_evaluate.py --split test --judge-triples # + judged-relevant triple precision (verifier model)
Output: work/runs/<split>/per_question_metrics.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import append_jsonl, llm, load_config, read_jsonl, warmup, work
from prompts import TRIPLE_RELEVANCE_SCHEMA, triple_relevance_messages


def load_log(cfg, split) -> list[dict]:
    rows = {}
    for r in read_jsonl(work(cfg, "runs", split, "log.jsonl")):
        if r["status"] == "ok":
            rows[(r["question_id"], r["cond"], r["run"])] = r  # last successful row wins
    return list(rows.values())


def deterministic_metrics(r: dict, refs: set[str]) -> dict:
    pids = r["passage_ids"]
    first = next((i for i, p in enumerate(pids, 1) if p in refs), None)
    labels = r["labels"]
    cites = [c for c in r.get("citations", [])]
    valid = [c for c in cites if c in labels]
    supported = [c for c in valid if labels[c]["passage_id"] in refs]
    tp = r["triple_passage_ids"]
    return {
        "recall_at_k": float(first is not None) if r["config"] != "0" else np.nan,
        "full_coverage": float(bool(refs) and refs <= set(pids)) if r["config"] != "0" else np.nan,
        "mrr": (1.0 / first if first else 0.0) if r["config"] != "0" else np.nan,
        "n_citations": len(cites),
        "citation_validity": len(valid) / len(cites) if cites else np.nan,
        "citation_support": len(supported) / len(valid) if valid else np.nan,
        "graph_citations": sum(labels[c]["kind"] == "triple" for c in valid),
        "n_triples": len(tp),
        "gold_linked_precision": (sum(p in refs for p in tp) / len(tp)) if tp else np.nan,
    }


def ragas_scores(cfg, rows: list[dict], qs: dict, cache_path) -> dict:
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

    host, o = cfg["ollama"]["host"], cfg["ollama"]
    judge = LangchainLLMWrapper(ChatOllama(model=cfg["models"]["evaluator"], base_url=host, temperature=0,
                                           seed=cfg["seed"], num_ctx=o["num_ctx"], keep_alive=o["keep_alive"]))
    emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=cfg["models"]["ragas_embeddings"], base_url=host))
    run_cfg = RunConfig(timeout=o["timeout_s"], max_retries=3, max_workers=1, seed=cfg["seed"])

    cache = {(c["question_id"], c["cond"], c["run"]): c for c in read_jsonl(cache_path)}
    todo = [r for r in rows if (r["question_id"], r["cond"], r["run"]) not in cache]
    print(f"RAGAS: {len(cache)} cached, {len(todo)} to score with {cfg['models']['evaluator']}")

    def sample(r, contexts):
        q = qs[r["question_id"]]
        keyed = q["options"][q["answer"] - 1]
        resp = (f"Option {r['option']}: {q['options'][r['option'] - 1]}. " if r["option"] else "") + r["justification"]
        return SingleTurnSample(user_input=f"{q['question']}\nOptions: " + "; ".join(q["options"]),
                                response=resp, retrieved_contexts=contexts or [""],
                                reference=f"{keyed}. {q['explanation']}")

    def run(metrics, samples):
        res = evaluate(EvaluationDataset(samples=samples), metrics=metrics, llm=judge, embeddings=emb,
                       run_config=run_cfg, raise_exceptions=False, show_progress=False)
        return res.to_pandas()

    faith, relev, cprec, crec = Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()
    batch = 10
    for i in tqdm(range(0, len(todo), batch)):
        chunk = todo[i:i + batch]
        with_ctx = [r for r in chunk if r["config"] != "0"]
        out = {(r["question_id"], r["cond"], r["run"]): {"question_id": r["question_id"], "cond": r["cond"], "run": r["run"]}
               for r in chunk}
        df = run([relev], [sample(r, r["contexts_full"]) for r in chunk])
        for r, v in zip(chunk, df[relev.name]):
            out[(r["question_id"], r["cond"], r["run"])]["answer_relevancy"] = v
        if with_ctx:
            df = run([faith, cprec, crec], [sample(r, r["contexts_full"]) for r in with_ctx])
            for r, (_, s) in zip(with_ctx, df.iterrows()):
                out[(r["question_id"], r["cond"], r["run"])].update(
                    faithfulness_full=s[faith.name], context_precision=s[cprec.name], context_recall=s[crec.name])
            graph_rows = [r for r in with_ctx if r["triple_ids"]]
            if graph_rows:
                df = run([faith], [sample(r, r["contexts_passages"]) for r in graph_rows])
                for r, v in zip(graph_rows, df[faith.name]):
                    out[(r["question_id"], r["cond"], r["run"])]["faithfulness_passages"] = v
        clean = [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()} for d in out.values()]
        append_jsonl(cache_path, clean)
        cache.update({(d["question_id"], d["cond"], d["run"]): d for d in clean})
    return cache


def judge_triples(cfg, rows, qs, kg_triples, ps, cache_path) -> dict:
    cache = {(c["question_id"], c["triple_id"]): c["helpful"] for c in read_jsonl(cache_path)}
    pairs = {(r["question_id"], t) for r in rows for t in r["triple_ids"]} - set(cache)
    if pairs:
        warmup(cfg, "verifier")
    for qid, tid in tqdm(sorted(pairs), desc="judge triples"):
        t = kg_triples[tid]
        stmt = f"{t['subject']} {t['relation']} {t['object']}"
        p = ps[t["passage_id"]]
        try:
            r = llm(cfg, "verifier", triple_relevance_messages(qs[qid], stmt, f"{p['header']}\n{p['text']}"),
                    schema=TRIPLE_RELEVANCE_SCHEMA, max_tokens=32)
            helpful = bool((r["json"] or {}).get("helpful", False))
        except RuntimeError:
            continue
        append_jsonl(cache_path, [{"question_id": qid, "triple_id": tid, "helpful": helpful}])
        cache[(qid, tid)] = helpful
    return cache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--ragas", action="store_true")
    ap.add_argument("--judge-triples", action="store_true")
    ap.add_argument("--all-runs", action="store_true", help="RAGAS on every repeat (default: run 0 only)")
    args = ap.parse_args()
    cfg = load_config()
    qs = {q["question_id"]: q for q in read_jsonl(work(cfg, "questions_all.jsonl"))}
    qlog = pd.read_csv(work(cfg, "benchmark", "question_log.csv")).fillna("")
    refs = {r.question_id: {x for x in r.reference_passage_ids.split(";") if x} for r in qlog.itertuples()}
    meta = qlog.set_index("question_id")[["hop_type", "question_type", "retriever_found"]].to_dict("index")
    rows = load_log(cfg, args.split)
    kg = {t["triple_id"]: t for t in read_jsonl(work(cfg, "kg", "triples_all.jsonl"))}
    corpus_versions = {p["id"]: p["version"] for p in read_jsonl(work(cfg, "passages.jsonl"))}

    out = []
    for r in rows:
        m = deterministic_metrics(r, refs.get(r["question_id"], set()))
        prov = [kg.get(t, {}).get("passage_id") in corpus_versions and kg[t]["version"] == corpus_versions[kg[t]["passage_id"]]
                for t in r["triple_ids"]]
        m["provenance_ok"] = float(all(prov)) if prov else np.nan
        out.append({k: r.get(k) for k in ("question_id", "cond", "config", "variant", "mode", "run", "correct", "valid",
                                          "T_q", "evidence_tokens", "graph_tokens", "graph_cap", "in_tokens", "out_tokens",
                                          "t_retrieve", "t_rerank", "t_kg", "t_generate", "t_total", "truncated_risk")}
                   | m | meta.get(r["question_id"], {}))
    df = pd.DataFrame(out)

    if args.judge_triples:
        ps = {p["id"]: p for p in read_jsonl(work(cfg, "passages.jsonl"))}
        cache = judge_triples(cfg, [r for r in rows if r["triple_ids"]], qs, kg, ps, work(cfg, "runs", args.split, "triple_judgements.jsonl"))
        jr = {(r["question_id"], r["cond"], r["run"]): [cache.get((r["question_id"], t)) for t in r["triple_ids"]] for r in rows}
        df["judged_relevant_precision"] = [
            (np.mean([x for x in jr[k] if x is not None]) if any(x is not None for x in jr[k]) else np.nan)
            for k in zip(df.question_id, df.cond, df.run)]

    if args.ragas:
        target = rows if args.all_runs else [r for r in rows if r["run"] == 0]
        cache = ragas_scores(cfg, target, qs, work(cfg, "runs", args.split, "ragas.jsonl"))
        for col in ("answer_relevancy", "faithfulness_full", "faithfulness_passages", "context_precision", "context_recall"):
            df[col] = [cache.get(k, {}).get(col) for k in zip(df.question_id, df.cond, df.run)]

    path = work(cfg, "runs", args.split, "per_question_metrics.csv")
    df.to_csv(path, index=False)
    print(df.groupby("cond")[["correct", "recall_at_k", "citation_validity", "evidence_tokens", "t_total"]].mean().round(3))
    print(f"wrote {path} ({len(df)} rows); truncation-risk rows: {int(df.truncated_risk.fillna(False).sum())}")


if __name__ == "__main__":
    main()
