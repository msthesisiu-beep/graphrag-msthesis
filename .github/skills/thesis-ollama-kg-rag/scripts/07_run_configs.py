"""Gate 8-9: run configurations 0/A/B/C/D + random-triple and unfiltered-expansion controls with the local generator.

Plans:
  --plan pilot      dev split, default variant, all configs (exploratory end-to-end check; use --limit 20)
  --plan dev-grid   dev split, B + all 18 C variants (matched) -> 09_stats.py --dev-select
  --plan test-main  test split, 0, A, B, C(default + selected), D, RAND, UNF in matched and overhead modes
Utilities:
  --calibrate-threshold     set reranker threshold from dev reference passages
  --determinism-check N     run B twice on N dev questions to decide repeats (1 vs 3)
Log: work/runs/<split>/log.jsonl (one row per question x condition x run; resumable).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time

import numpy as np
from tqdm import tqdm

from common import (append_jsonl, llm, load_config, load_split, ntok, read_json, read_jsonl, stable_int,
                    trim_to_tokens, warmup, work, write_json)
from prompts import GEN_SCHEMA, PROMPT_VERSION, gen_messages
from retrieval import Linker, Retriever, load_graph, neighbourhood_edges, render_triple


def vname(v: dict) -> str:
    return f"t{v['triples']}-h{v['hops']}-g{int(round(v['share'] * 100))}"


def vparse(name: str) -> dict:
    t, h, g = name.split("-")
    return {"triples": int(t[1:]), "hops": int(h[1:]), "share": int(g[1:]) / 100}


def as_option(x) -> int | None:
    try:
        return int(str(x).strip().split()[0].rstrip(".:"))
    except (ValueError, IndexError):
        return None


class Builder:
    def __init__(self, cfg: dict, R: Retriever, G):
        self.cfg, self.R, self.G = cfg, R, G
        self.linker = Linker(G) if G is not None else None
        self.valid = set(R.by_id)
        self.all_edges = ([d for *_, d in G.edges(data=True) if d.get("verified") and d["passage_id"] in self.valid]
                          if G is not None else [])

    # ------------------------------------------------ rendering / token accounting
    def p_head(self, pid: str) -> str:
        return f"[P00] {self.R.by_id[pid]['header']}\n"

    def p_cost(self, pid: str, body: str) -> int:
        return ntok(self.cfg, self.p_head(pid) + body) + 1

    def g_cost(self, d: dict) -> int:
        return ntok(self.cfg, render_triple("G00", d)) + 1

    def render(self, items: list[tuple]) -> dict:
        parts, labels, pctx, gctx = [], {}, [], []
        np_, ng = 0, 0
        for it in items:
            if it[0] == "p":
                np_ += 1
                lab = f"P{np_}"
                s = f"[{lab}] {self.R.by_id[it[1]]['header']}\n{it[2]}"
                labels[lab] = {"kind": "passage", "id": it[1], "passage_id": it[1]}
                pctx.append(s)
            else:
                ng += 1
                lab = f"G{ng}"
                s = render_triple(lab, it[1])
                labels[lab] = {"kind": "triple", "id": it[1]["triple_id"], "passage_id": it[1]["passage_id"]}
                gctx.append(s)
            parts.append(s)
        block = "\n\n".join(parts)
        return {"block": block, "labels": labels, "contexts_full": parts, "contexts_passages": pctx,
                "evidence_tokens": ntok(self.cfg, block), "graph_tokens": ntok(self.cfg, "\n\n".join(gctx)),
                "passage_ids": [it[1] for it in items if it[0] == "p"],
                "triple_ids": [it[1]["triple_id"] for it in items if it[0] == "g"],
                "triple_passage_ids": [it[1]["passage_id"] for it in items if it[0] == "g"]}

    def full(self, pids: list[str]) -> list[tuple]:
        return [("p", pid, self.R.by_id[pid]["text"]) for pid in pids]

    def fill(self, pids: list[str], budget: int) -> list[tuple]:
        """Passages in the given order; the first one that overflows is trimmed at a sentence boundary, rest dropped."""
        out, used = [], 0
        for pid in pids:
            body = self.R.by_id[pid]["text"]
            cost = self.p_cost(pid, body)
            if used + cost <= budget:
                out.append(("p", pid, body))
                used += cost
                continue
            trimmed = trim_to_tokens(self.cfg, body, budget - used - self.p_cost(pid, ""))
            if trimmed:
                out.append(("p", pid, trimmed))
            break
        return out

    # ------------------------------------------------ graph selection
    def select(self, seeds: set, qemb: np.ndarray, v: dict, cap: int, how: str = "selective",
               rng: np.random.Generator | None = None, count: int | None = None) -> list[dict]:
        if how == "random":
            order = [self.all_edges[i] for i in rng.permutation(len(self.all_edges))]
            limit = count
        else:
            cand = neighbourhood_edges(self.G, seeds, v["hops"], self.valid)
            if how == "selective" and cand:
                texts = [f"{d['s_name']} {d['relation'].replace('_', ' ')} {d['o_name']}" for _, d in cand]
                sims = self.R.embed(texts) @ qemb
                cand = [cand[i] for i in sorted(range(len(cand)), key=lambda i: (-sims[i], cand[i][0]))]
            order = [d for _, d in cand]
            limit = v["triples"] if how == "selective" else None
        out, used, seen = [], 0, set()
        for d in order:
            if limit is not None and len(out) >= limit:
                break
            key = (d["s_name"].lower(), d["relation"], d["o_name"].lower())
            if key in seen:
                continue
            c = self.g_cost(d)
            if used + c > cap:
                break
            seen.add(key)
            out.append(d)
            used += c
        return out


def build_conditions(cfg, Bld: Builder, q, cands, B_ids, T_q, plan_variants, plan, qemb) -> list[dict]:
    R = Bld.R
    keep = cfg["retrieval"]["rerank_keep"]
    vec_ids = [pid for pid, _ in cands]
    conds = []

    def add(config, variant, mode, items, t_kg=0.0, extra=None):
        conds.append({"config": config, "variant": variant, "mode": mode, "items": items, "t_kg": t_kg,
                      "extra": extra or {}})

    if plan["0"]:
        add("0", "-", "-", None)
    if plan["A"]:
        add("A", "-", "matched", Bld.fill(vec_ids, T_q))
        if "overhead" in plan["modes"]:
            add("A", "-", "overhead", Bld.full(vec_ids[:keep]))
    add("B", "-", "-", Bld.full(B_ids))
    if Bld.G is None:
        return conds

    b_text = " ".join(R.by_id[p]["text"] for p in B_ids)
    default = cfg["kg"]["default_variant"]
    c_default_triples = None
    for vn in plan_variants:
        v = vparse(vn)
        cap = int(v["share"] * T_q)
        t0 = time.perf_counter()
        seeds = Bld.linker.link(b_text)
        triples = Bld.select(seeds, qemb, v, cap)
        t_kg = time.perf_counter() - t0
        g_items = [("g", d) for d in triples]
        meta = {"seeds": len(seeds), "graph_cap": cap}
        add("C", vn, "matched", Bld.fill(B_ids, T_q - cap) + g_items, t_kg, meta)
        if "overhead" in plan["modes"]:
            add("C", vn, "overhead", Bld.full(B_ids) + g_items, t_kg, meta)
        if vn == vname(default):
            c_default_triples = triples

    if plan["controls"] and c_default_triples is not None:
        v, vn = default, vname(default)
        cap = int(v["share"] * T_q)
        rng = np.random.default_rng(stable_int(cfg["seed"], q["question_id"]))
        t0 = time.perf_counter()
        rand = Bld.select(set(), qemb, v, cap, how="random", rng=rng, count=len(c_default_triples))
        t_r = time.perf_counter() - t0
        t0 = time.perf_counter()
        unf = Bld.select(Bld.linker.link(b_text), qemb, v, cap, how="unfiltered")
        t_u = time.perf_counter() - t0
        for name, trip, t in (("RAND", rand, t_r), ("UNF", unf, t_u)):
            g_items = [("g", d) for d in trip]
            add(name, vn, "matched", Bld.fill(B_ids, T_q - cap) + g_items, t, {"graph_cap": cap})
            if "overhead" in plan["modes"]:
                add(name, vn, "overhead", Bld.full(B_ids) + g_items, t, {"graph_cap": cap})

    if plan["D"]:
        v, vn = default, vname(default)
        cap = int(v["share"] * T_q)
        t0 = time.perf_counter()
        seeds = Bld.linker.link(" ".join(R.by_id[p]["text"] for p in vec_ids))
        triples = Bld.select(seeds, qemb, v, T_q)  # cap applied in the joint fill below
        pool = [(f"p::{pid}", R.ptext(pid)) for pid in vec_ids] + \
               [(f"g::{i}", render_triple("G", d)[4:]) for i, d in enumerate(triples)]
        scored = R.rerank(q["question"], pool)
        items, used, gused = [], 0, 0
        for iid, _ in scored:
            kind, ref = iid.split("::", 1)
            if kind == "g":
                d = triples[int(ref)]
                c = Bld.g_cost(d)
                if gused + c <= cap and used + c <= T_q:
                    items.append(("g", d))
                    used, gused = used + c, gused + c
                continue
            body = R.by_id[ref]["text"]
            c = Bld.p_cost(ref, body)
            if used + c <= T_q:
                items.append(("p", ref, body))
                used += c
                continue
            trimmed = trim_to_tokens(cfg, body, T_q - used - Bld.p_cost(ref, ""))
            if trimmed:
                items.append(("p", ref, trimmed))
            break
        add("D", vn, "matched", items, time.perf_counter() - t0, {"seeds": len(seeds), "graph_cap": cap})
    return conds


def generate(cfg, q, rendered, seed) -> dict:
    block = None if rendered is None else rendered["block"]
    r = llm(cfg, "generator", gen_messages(q, block), schema=GEN_SCHEMA, seed=seed, max_tokens=cfg["ollama"]["gen_max_tokens"])
    j = r["json"] or {}
    opt = as_option(j.get("option"))
    valid = opt is not None and 1 <= opt <= len(q["options"])
    return {"option": opt if valid else None, "valid": valid, "correct": int(valid and opt == q["answer"]),
            "justification": str(j.get("justification", ""))[:2000],
            "citations": [str(c).strip("[] ") for c in j.get("citations", []) if str(c).strip()] if isinstance(j.get("citations"), list) else [],
            "raw": r["content"][:2000] if r["json"] is None else "", "t_generate": r["wall_s"],
            "in_tokens": r["in_tokens"], "in_tokens_ollama": r["in_tokens_ollama"], "out_tokens": r["out_tokens"],
            "truncated_risk": r["truncated_risk"]}


def b_selection(cfg, R, q_text, threshold):
    keep = cfg["retrieval"]["rerank_keep"]
    t0 = time.perf_counter()
    cands = R.search_bge(q_text, cfg["retrieval"]["candidates"])
    t_ret = time.perf_counter() - t0
    t0 = time.perf_counter()
    rr = R.rerank(q_text, [(pid, R.ptext(pid)) for pid, _ in cands])
    t_rr = time.perf_counter() - t0
    top = rr[:keep]
    B = [pid for pid, s in top if threshold is None or s >= threshold] or [top[0][0]]  # never empty
    return cands, rr, B, t_ret, t_rr


def get_threshold(cfg):
    if cfg["retrieval"]["rerank_threshold"] is not None:
        return cfg["retrieval"]["rerank_threshold"]
    p = work(cfg, "runs", "dev", "rerank_threshold.json")
    return read_json(p)["threshold"] if p.exists() else None


def references(cfg) -> dict[str, list[str]]:
    import pandas as pd
    df = pd.read_csv(work(cfg, "benchmark", "question_log.csv")).fillna("")
    return {r.question_id: [x for x in r.reference_passage_ids.split(";") if x] for r in df.itertuples()}


def calibrate(cfg, R, qs) -> None:
    refs = references(cfg)
    scores = []
    for qid in tqdm(load_split(cfg, "dev")):
        _, rr, _, _, _ = b_selection(cfg, R, qs[qid]["question"], None)
        scores += [s for pid, s in rr[: cfg["retrieval"]["rerank_keep"]] if pid in refs.get(qid, [])]
    thr = float(np.percentile(scores, 5)) if scores else None
    write_json(work(cfg, "runs", "dev", "rerank_threshold.json"),
               {"threshold": thr, "rule": "5th percentile of reranker scores of reference passages in dev top-k",
                "n_scores": len(scores)})
    print(f"rerank threshold = {thr} from {len(scores)} dev reference passages")


def determinism(cfg, R, qs, n) -> None:
    warmup(cfg, "generator")
    Bld = Builder(cfg, R, None)
    same = 0
    ids = load_split(cfg, "dev")[:n]
    for qid in tqdm(ids):
        _, _, B, _, _ = b_selection(cfg, R, qs[qid]["question"], get_threshold(cfg))
        rd = Bld.render(Bld.full(B))
        a, b = generate(cfg, qs[qid], rd, cfg["seed"]), generate(cfg, qs[qid], rd, cfg["seed"])
        same += a["option"] == b["option"] and a["justification"] == b["justification"]
    res = {"n": len(ids), "identical": same, "deterministic": same == len(ids)}
    write_json(work(cfg, "runs", "determinism.json"), res)
    print(res, "-> keep run.repeats=1" if res["deterministic"] else "-> set run.repeats=3 in config")


def condition_keys(plan: dict, variants: list[str], default: str) -> set[str]:
    keys = {"B:-:-"}
    if plan["0"]:
        keys.add("0:-:-")
    if plan["A"]:
        keys.add("A:-:matched")
        if "overhead" in plan["modes"]:
            keys.add("A:-:overhead")
    for variant in variants:
        keys.add(f"C:{variant}:matched")
        if "overhead" in plan["modes"]:
            keys.add(f"C:{variant}:overhead")
    if plan["controls"] and default in variants:
        for name in ("RAND", "UNF"):
            keys.add(f"{name}:{default}:matched")
            if "overhead" in plan["modes"]:
                keys.add(f"{name}:{default}:overhead")
    if plan["D"]:
        keys.add(f"D:{default}:matched")
    return keys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=["pilot", "dev-grid", "test-main"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--default-only", action="store_true", help="test-main without dev-selected variants")
    ap.add_argument("--calibrate-threshold", action="store_true")
    ap.add_argument("--determinism-check", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config()
    R = Retriever(cfg)
    qs = {q["question_id"]: q for q in read_jsonl(work(cfg, "questions_all.jsonl"))}
    if args.calibrate_threshold:
        return calibrate(cfg, R, qs)
    if args.determinism_check:
        return determinism(cfg, R, qs, args.determinism_check)
    if not args.plan:
        ap.error("--plan is required")

    k = cfg["kg"]
    default = vname(k["default_variant"])
    if args.plan == "pilot":
        split, variants = "dev", [default]
        plan = {"0": True, "A": True, "D": True, "controls": True, "modes": ["matched", "overhead"]}
    elif args.plan == "dev-grid":
        split = "dev"
        variants = [vname({"triples": t, "hops": h, "share": s}) for t in k["grid"]["triples"]
                    for h in k["grid"]["hops"] for s in k["grid"]["share"]]
        plan = {"0": False, "A": False, "D": False, "controls": False, "modes": ["matched"]}
    else:
        split = "test"
        sel = work(cfg, "runs", "dev", "selected_variants.json")
        if not sel.exists() and not args.default_only:
            raise SystemExit("Run --plan dev-grid and 09_stats.py --split dev --dev-select first (or --default-only).")
        variants = [default] if args.default_only else read_json(sel)["variants"]
        plan = {"0": True, "A": True, "D": True, "controls": True, "modes": ["matched", "overhead"]}

    ids = load_split(cfg, split)
    order = [ids[i] for i in np.random.default_rng(cfg["seed"]).permutation(len(ids))][: args.limit]
    threshold = get_threshold(cfg)
    log = work(cfg, "runs", split, "log.jsonl")
    done = {(r["question_id"], r["cond"], r["run"]) for r in read_jsonl(log) if r["status"] == "ok"}
    repeats = cfg["run"]["repeats"]
    expected = condition_keys(plan, variants, default)
    pending = [qid for qid in order if any((qid, cond, run) not in done for cond in expected for run in range(repeats))]
    write_json(work(cfg, "runs", split, f"plan_{args.plan}.json"),
               {"plan": plan, "variants": variants, "threshold": threshold, "repeats": repeats, "questions": len(order),
                "questions_pending": len(pending),
                "prompt_version": PROMPT_VERSION, "generator": cfg["models"]["generator"],
                "config_sha256": cfg["_config_sha256"], "started": dt.datetime.now().isoformat(timespec="seconds")})
    if not pending:
        print("All planned question/condition/repeat records are complete; no model work needed.")
        return
    G = load_graph(cfg)
    Bld = Builder(cfg, R, G)
    warmup(cfg, "generator")

    for qid in tqdm(pending, desc=f"{args.plan}/{split}"):
        q = qs[qid]
        cands, _, B_ids, t_ret, t_rr = b_selection(cfg, R, q["question"], threshold)
        T_q = Bld.render(Bld.full(B_ids))["evidence_tokens"]
        qemb = R.embed([q["question"]], query=True)[0]
        conds = build_conditions(cfg, Bld, q, cands, B_ids, T_q, variants, plan, qemb)
        rng = np.random.default_rng(stable_int(cfg["seed"], qid, "order"))  # randomise condition order per question
        for ci in rng.permutation(len(conds)):
            c = conds[ci]
            cond = f"{c['config']}:{c['variant']}:{c['mode']}"
            rendered = None if c["items"] is None else Bld.render(c["items"])
            for run in range(repeats):
                if (qid, cond, run) in done:
                    continue
                seed = cfg["seed"] + run
                try:
                    g = generate(cfg, q, rendered, seed)
                    status = "ok"
                except RuntimeError as e:
                    g, status = {"error": str(e)[:300], "valid": False, "correct": 0, "t_generate": 0.0}, "error"
                uses_ret = c["config"] != "0"
                uses_rr = c["config"] not in ("0", "A")
                stages = {"t_retrieve": t_ret if uses_ret else 0.0, "t_rerank": t_rr if uses_rr else 0.0,
                          "t_kg": c["t_kg"]}
                row = {"question_id": qid, "split": split, "plan": args.plan, "cond": cond, "config": c["config"],
                       "variant": c["variant"], "mode": c["mode"], "run": run, "seed": seed, "status": status,
                       "category": q["category"], "T_q": T_q, "B_passage_ids": B_ids,
                       "reranker_saw_graph": c["config"] == "D", "prompt_version": PROMPT_VERSION,
                       "generator": cfg["models"]["generator"], **c["extra"], **stages, **g,
                       "timestamp": dt.datetime.now().isoformat(timespec="seconds")}
                if rendered is not None:
                    row.update({k2: rendered[k2] for k2 in ("labels", "contexts_full", "contexts_passages", "evidence_tokens",
                                                            "graph_tokens", "passage_ids", "triple_ids", "triple_passage_ids")})
                else:
                    row.update(labels={}, contexts_full=[], contexts_passages=[], evidence_tokens=0, graph_tokens=0,
                               passage_ids=[], triple_ids=[], triple_passage_ids=[])
                row["t_total"] = sum(stages.values()) + g.get("t_generate", 0.0)
                append_jsonl(log, [row])

    rows = read_jsonl(log)
    print(json.dumps({"rows": len(rows), "errors": sum(r["status"] != "ok" for r in rows)}))


if __name__ == "__main__":
    main()
