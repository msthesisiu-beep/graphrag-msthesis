"""Reproducible, separately labelled synthetic Release 17 question generation.

Usage: python 11_generate_questions.py plan|generate|verify|export|all --target 300
Generated candidates are checkpointed, not regenerated on resume. Human review
is required before treating the exported questions as a final benchmark.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import urllib.request
from pathlib import Path

from common import RawEvidence, append_jsonl, llm, load_config, norm_ws, quote_in_text, read_jsonl, stable_int, work, write_json, write_jsonl

CREATE_SYSTEM = (
    "Create exactly one technical multiple-choice question using ONLY the supplied specification passage. "
    "Test a specific requirement, procedure, dependency, or parameter, not generic trivia. "
    "Use four distinct plausible options and exactly one correct answer. Do not mention clause numbers "
    "or the source document in the question. Copy a contiguous supporting sentence verbatim as quote. "
    "If the passage cannot support an unambiguous question, return suitable=false. "
    "Return suitable, question, options, answer (1-4), explanation, quote as JSON."
)
CHECK_SYSTEM = (
    "Independently solve the multiple-choice question using ONLY the supplied specification passage. "
    "No proposed answer is provided. supported=true only if exactly one option is fully supported. "
    "Set ambiguous=true if another option is defensible or wording is underspecified. "
    "Copy a contiguous supporting sentence verbatim as quote. Return supported, ambiguous, "
    "answer (1-4, or 0 if unsupported), quote, explanation as JSON."
)
CREATE_SCHEMA = {"type": "object", "properties": {
    "suitable": {"type": "boolean"}, "question": {"type": "string"},
    "options": {"type": "array", "items": {"type": "string"}, "minItems": 4, "maxItems": 4},
    "answer": {"type": "integer"}, "explanation": {"type": "string"}, "quote": {"type": "string"}},
    "required": ["suitable", "question", "options", "answer", "explanation", "quote"]}
CHECK_SCHEMA = {"type": "object", "properties": {
    "supported": {"type": "boolean"}, "ambiguous": {"type": "boolean"},
    "answer": {"type": "integer"}, "quote": {"type": "string"}, "explanation": {"type": "string"}},
    "required": ["supported", "ambiguous", "answer", "quote", "explanation"]}
PROMPT_HASH = hashlib.sha256(json.dumps([CREATE_SYSTEM, CHECK_SYSTEM, CREATE_SCHEMA, CHECK_SCHEMA], sort_keys=True).encode()).hexdigest()


def output(cfg: dict, name: str) -> Path:
    return work(cfg, "synthetic", name)


def select_passages(cfg: dict, passages: list[dict], limit: int) -> list[dict]:
    groups = {}
    for spec in sorted(cfg["corpus"]["specs"]):
        eligible = [passage for passage in passages if passage.get("release") == 17 and passage["spec"] == spec
                    and 100 <= passage.get("n_tokens", 0) <= cfg["corpus"]["max_tokens"]]
        eligible.sort(key=lambda passage: (stable_int(cfg["seed"], passage["id"]), passage["id"]))
        seen_clauses = set()
        groups[spec] = []
        for passage in eligible:
            if passage["clause"] not in seen_clauses:
                groups[spec].append(passage)
                seen_clauses.add(passage["clause"])
    selected = []
    for position in range(max((len(group) for group in groups.values()), default=0)):
        for group in groups.values():
            if position < len(group):
                selected.append(group[position])
                if len(selected) == limit:
                    return selected
    return selected


def prepare(cfg: dict, target: int, source_limit: int) -> dict:
    if cfg["corpus"]["release"] != 17:
        raise ValueError("This module requires Release 17")
    if cfg["models"]["question_creator"] in {cfg["models"]["generator"], cfg["models"]["verifier"]}:
        raise ValueError("Question creator must differ from benchmark generator and verifier")
    passages_path = work(cfg, "passages.jsonl")
    passages = read_jsonl(passages_path)
    selected = select_passages(cfg, passages, source_limit)
    if not selected:
        raise ValueError("Prepare the configured corpus before generating questions")
    raw = RawEvidence(cfg)
    sources = {}
    for passage in selected:
        if passage["source"] not in sources:
            citation = raw.check(passage, "")
            sources[passage["source"]] = {key: citation[key] for key in
                                         ("md_path", "docx_path", "md_sha256", "docx_sha256", "version")}
    manifest = {"protocol": "synthetic-release17-v1", "seed": cfg["seed"], "target": target,
                "source_limit": source_limit, "corpus": cfg["corpus"], "sources": sources,
                "passages_sha256": hashlib.sha256(passages_path.read_bytes()).hexdigest(),
                "prompt_sha256": PROMPT_HASH, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "models": {role: cfg["models"][role] for role in ("question_creator", "verifier", "generator")},
                "ollama_options": {"temperature": 0, "num_ctx": cfg["ollama"]["num_ctx"]},
                "selected_passage_ids": [passage["id"] for passage in selected]}
    plan_path = output(cfg, "plan.json")
    if plan_path.exists() and json.loads(plan_path.read_text()) != manifest:
        raise ValueError("Existing synthetic plan differs; use a separate work directory, not mixed checkpoints")
    write_json(plan_path, manifest)
    return manifest


def check_models(cfg: dict) -> None:
    with urllib.request.urlopen(cfg["ollama"]["host"] + "/api/tags", timeout=10) as response:
        available = {model["name"]: model["digest"] for model in json.load(response)["models"]}
    snapshot = {role: available[cfg["models"][role]] for role in ("question_creator", "verifier")}
    path = output(cfg, "model_digests.json")
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise ValueError("Ollama model digests changed; refusing to mix runs")
    write_json(path, snapshot)


def validate_candidate(cfg: dict, passage: dict, draft: dict, raw: RawEvidence) -> dict:
    if draft.get("suitable") is not True:
        raise ValueError("unsuitable_passage")
    options = draft.get("options", [])
    if not isinstance(options, list) or len(options) != 4 or not all(isinstance(option, str) and option.strip() for option in options):
        raise ValueError("invalid_options")
    if len({norm_ws(option) for option in options}) != 4:
        raise ValueError("duplicate_options")
    if type(draft.get("answer")) is not int or not 1 <= draft["answer"] <= 4 or not str(draft.get("question", "")).strip():
        raise ValueError("invalid_question_or_answer")
    quote = draft.get("quote", "")
    citation = raw.check(passage, quote)
    if not quote_in_text(quote, passage["text"]) or not citation["source_validated"]:
        raise ValueError("quote_not_in_both_raw_formats")
    return {"question_id": "synthetic " + hashlib.sha256((passage["id"] + norm_ws(draft["question"])).encode()).hexdigest()[:20],
            "dataset": "synthetic", "release_tag": 17, "category": "Standards specifications",
            "question": draft["question"], "options": options, "answer": draft["answer"],
            "explanation": draft.get("explanation", ""), "citations": [citation], "passage_id": passage["id"],
            "question_creator": cfg["models"]["question_creator"], "prompt_sha256": PROMPT_HASH}


def generate(cfg: dict, plan: dict) -> None:
    passages = {passage["id"]: passage for passage in read_jsonl(work(cfg, "passages.jsonl"))}
    path = output(cfg, "generation.jsonl")
    existing = {row["passage_id"]: row for row in read_jsonl(path)}
    raw = RawEvidence(cfg)
    for pid in plan["selected_passage_ids"]:
        if existing.get(pid, {}).get("status") in {"ok", "rejected"}:
            continue
        passage = passages[pid]
        messages = [{"role": "system", "content": CREATE_SYSTEM},
                    {"role": "user", "content": passage["header"] + "\n" + passage["text"]}]
        try:
            response = llm(cfg, "question_creator", messages, schema=CREATE_SCHEMA,
                           seed=stable_int(cfg["seed"], pid) % (2 ** 31), max_tokens=800)
            if not response["json"] or response["truncated_risk"]:
                raise RuntimeError("Unparsable or truncated generation")
            try:
                candidate = validate_candidate(cfg, passage, response["json"], raw)
                record = {"passage_id": pid, "status": "ok", "candidate": candidate}
            except ValueError as error:
                record = {"passage_id": pid, "status": "rejected", "reason": str(error)}
            record["response"] = response
        except Exception as error:
            append_jsonl(path, [{"passage_id": pid, "status": "error", "error": str(error)}])
            raise
        append_jsonl(path, [record])
        print(f"Generation {pid}: {record['status']}", flush=True)


def verify(cfg: dict, plan: dict) -> None:
    generated = {row["passage_id"]: row for row in read_jsonl(output(cfg, "generation.jsonl"))}
    path = output(cfg, "verification.jsonl")
    existing = {row["question_id"]: row for row in read_jsonl(path)}
    passages = {passage["id"]: passage for passage in read_jsonl(work(cfg, "passages.jsonl"))}
    raw = RawEvidence(cfg)
    for pid in plan["selected_passage_ids"]:
        row = generated.get(pid, {})
        if row.get("status") != "ok":
            continue
        candidate = row["candidate"]
        qid = candidate["question_id"]
        if existing.get(qid, {}).get("status") == "ok":
            continue
        passage = passages[pid]
        options = "\n".join(f"{position}. {option}" for position, option in enumerate(candidate["options"], 1))
        messages = [{"role": "system", "content": CHECK_SYSTEM}, {"role": "user", "content":
                    f"Source:\n{passage['header']}\n{passage['text']}\nQuestion: {candidate['question']}\nOptions:\n{options}"}]
        try:
            response = llm(cfg, "verifier", messages, schema=CHECK_SCHEMA, seed=stable_int(cfg["seed"], qid) % (2 ** 31), max_tokens=800)
            result = response["json"]
            if not result or response["truncated_risk"]:
                raise RuntimeError("Unparsable or truncated verification")
            citation = raw.check(passage, result.get("quote", ""))
            accepted = (result.get("supported") is True and result.get("ambiguous") is False
                        and type(result.get("answer")) is int and result["answer"] == candidate["answer"]
                        and citation["source_validated"] and quote_in_text(result.get("quote", ""), passage["text"]))
            record = {"question_id": qid, "status": "ok", "accepted": bool(accepted), "response": response,
                      "candidate": candidate | {"citations": candidate["citations"] + [citation],
                                                "verified_explanation": result.get("explanation", ""),
                                                "verifier_model": cfg["models"]["verifier"], "human_review": "pending"}}
        except Exception as error:
            append_jsonl(path, [{"question_id": qid, "status": "error", "error": str(error)}])
            raise
        append_jsonl(path, [record])
        existing[qid] = record
        export(cfg, plan)
        print(f"Verification {qid}: accepted={accepted}", flush=True)


def export(cfg: dict, plan: dict) -> list[dict]:
    records = {row["question_id"]: row for row in read_jsonl(output(cfg, "verification.jsonl"))}
    seen = {norm_ws(question["question"]) for question in read_jsonl(work(cfg, "questions_all.jsonl"))}
    answers = []
    for record in records.values():
        if record.get("status") != "ok" or not record.get("accepted"):
            continue
        candidate = record["candidate"]
        key = norm_ws(candidate["question"])
        if key in seen:
            continue
        seen.add(key)
        answers.append(candidate)
        if len(answers) == plan["target"]:
            break
    write_jsonl(output(cfg, "questions.jsonl"), answers)
    review_path = output(cfg, "questions_review.csv")
    prior_review = {}
    if review_path.exists():
        with review_path.open(newline="") as handle:
            prior_review = {row["question_id"]: row["manual_supported"] for row in csv.DictReader(handle)}
    with review_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question_id", "question", "options", "answer", "spec", "clause",
                                                   "folder", "md_name", "docx_name", "quote", "manual_supported"])
        writer.writeheader()
        for answer in answers:
            citation = answer["citations"][-1]
            writer.writerow({"question_id": answer["question_id"], "question": answer["question"],
                             "options": json.dumps(answer["options"]), "answer": answer["answer"],
                             **{key: citation[key] for key in ("spec", "clause", "folder", "md_name", "docx_name", "quote")},
                             "manual_supported": prior_review.get(answer["question_id"], "")})
    write_json(output(cfg, "summary.json"), {"accepted_pending_human_review": len(answers), "target": plan["target"],
               "target_met": len(answers) >= plan["target"], "dataset": "synthetic", "prompt_sha256": PROMPT_HASH})
    return answers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["plan", "generate", "verify", "export", "all"])
    parser.add_argument("--target", type=int, default=300)
    parser.add_argument("--source-limit", type=int, default=900)
    args = parser.parse_args()
    if args.target < 1 or args.source_limit < args.target:
        parser.error("Require source-limit >= target >= 1")
    cfg = load_config()
    cfg["models"]["question_creator"] = cfg["models"].get("question_creator", cfg["models"]["kg_extractor"])
    plan = prepare(cfg, args.target, args.source_limit)
    print(f"Synthetic plan: {len(plan['selected_passage_ids'])} clauses; target={args.target}", flush=True)
    if args.phase in {"generate", "verify", "all"}:
        check_models(cfg)
    if args.phase in {"generate", "all"}:
        generate(cfg, plan)
    if args.phase in {"verify", "all"}:
        verify(cfg, plan)
    if args.phase in {"export", "all"}:
        answers = export(cfg, plan)
        print(f"Accepted pending human review: {len(answers)}/{args.target}", flush=True)


if __name__ == "__main__":
    main()