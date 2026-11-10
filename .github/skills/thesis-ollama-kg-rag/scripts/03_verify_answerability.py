"""Gate 4-5: retriever-independent candidate pooling (BM25 + E5, never BGE) and answerability verification.

Each question gets one record in work/benchmark/verification.jsonl with status ok|error.
Execution errors are NEVER treated as exclusions; rerun with --retry-errors until none remain.
Usage: python 03_verify_answerability.py [--limit N] [--retry-errors]
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from tqdm import tqdm

from common import RawEvidence, append_jsonl, llm, load_config, quote_in_text, read_jsonl, warmup, work, write_jsonl
from prompts import ADJUDICATE_SCHEMA, PROMPT_VERSION, VERIFY_SCHEMA, adjudicate_messages, verify_messages
from retrieval import Retriever, rrf


VALIDATION_VERSION = "raw-md-docx-v1"


class VerifierResponseError(RuntimeError):
    def __init__(self, message: str, tokens_in: int, tokens_out: int):
        super().__init__(message)
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out


def verifier_response(cfg: dict, messages: list[dict], schema: dict, max_tokens: int) -> dict:
    tokens_in, tokens_out = 0, 0
    for budget in (max_tokens, max_tokens * 2):
        response = llm(cfg, "verifier", messages, schema=schema, max_tokens=budget)
        tokens_in += response["in_tokens"]
        tokens_out += response["out_tokens"]
        result = response.get("json")
        valid = isinstance(result, dict) and all(key in result for key in schema["required"])
        if valid:
            for key, definition in schema["properties"].items():
                value = result.get(key)
                if definition["type"] == "boolean":
                    valid &= type(value) is bool
                elif definition["type"] == "string":
                    valid &= isinstance(value, str) and ("enum" not in definition or value in definition["enum"])
                elif definition["type"] == "array":
                    valid &= isinstance(value, list) and all(isinstance(item, dict)
                        and isinstance(item.get("passage_id"), str) and isinstance(item.get("quote"), str) for item in value)
        if valid:
            return response | {"in_tokens": tokens_in, "out_tokens": tokens_out}
    raise VerifierResponseError(f"Verifier returned invalid JSON after budgets {max_tokens}/{max_tokens * 2}; "
                                f"last output tokens={response['out_tokens']}", tokens_in, tokens_out)


def verify_passages(cfg: dict, question: dict, passages: list[dict]) -> list[tuple[list[dict], dict]]:
    try:
        response = verifier_response(cfg, verify_messages(question, passages), VERIFY_SCHEMA, 1500)
        return [(passages, response)]
    except VerifierResponseError as error:
        if len(passages) <= 1:
            raise
        middle = len(passages) // 2
        results = verify_passages(cfg, question, passages[:middle]) + verify_passages(cfg, question, passages[middle:])
        batch, response = results[0]
        results[0] = (batch, response | {"in_tokens": response["in_tokens"] + error.tokens_in,
                                       "out_tokens": response["out_tokens"] + error.tokens_out})
        return results


def verify_question(cfg: dict, R: Retriever, q: dict, raw_sources: RawEvidence | None = None) -> dict:
    raw_sources = raw_sources or RawEvidence(cfg)
    if q.get("release_tag") != cfg["corpus"]["release"]:
        raise ValueError("Question is outside the configured release")
    b = cfg["benchmark"]
    keyed = q["options"][q["answer"] - 1]
    query = f"{q['question']} {keyed}"  # pooling searches for support of the keyed option
    bm, e5 = R.search_bm25(query, b["pool_k"]), R.search_e5(query, b["pool_k"])
    sources: dict[str, list[str]] = {}
    for name, hits in (("bm25", bm), ("e5", e5)):
        for pid, _ in hits:
            sources.setdefault(pid, []).append(name)
    pooled = rrf([bm, e5])
    rec = {"question_id": q["question_id"], "category": q["category"], "release_tag": q["release_tag"],
           "pool_size": len(pooled), "candidate_sources": sorted({s for v in sources.values() for s in v}),
           "checked_passage_ids": pooled[: b["verify_top_n"]], "prompt_version": PROMPT_VERSION,
           "verifier_model": cfg["models"]["verifier"], "tokens_in": 0, "tokens_out": 0,
           "validation_version": VALIDATION_VERSION}
    if not pooled:
        return rec | {"status": "ok", "verifier_verdict": "no_support", "quote_check": "fail",
                      "reference_passage_ids": [], "decision": "exclude", "reason_code": "X1"}

    verdicts, conflict, evidence = [], False, []
    to_check = rec["checked_passage_ids"]
    for i in range(0, len(to_check), b["verify_batch"]):
        batch = [R.by_id[pid] for pid in to_check[i:i + b["verify_batch"]]]
        for checked_batch, response in verify_passages(cfg, q, batch):
            rec["tokens_in"] += response["in_tokens"]
            rec["tokens_out"] += response["out_tokens"]
            result = response["json"]
            verdicts.append(result["verdict"])
            conflict |= result["release_conflict"]
            allowed = {passage["id"] for passage in checked_batch}
            for item in result["evidence"]:
                pid = item["passage_id"]
                quote = item["quote"]
                checked = raw_sources.check(R.by_id[pid], quote) if pid in allowed else {}
                ok = pid in allowed and quote_in_text(quote, R.by_id[pid]["text"]) and checked.get("source_validated", False)
                evidence.append(checked | {"passage_id": pid, "quote": quote, "quote_ok": bool(ok)})

    verdict = "supports" if "supports" in verdicts else ("partial" if "partial" in verdicts else "no_support")
    refs = sorted({e["passage_id"] for e in evidence if e["quote_ok"]})
    rec.update(status="ok", verifier_verdict=verdict, release_conflict=conflict, evidence=evidence,
               quote_check="pass" if refs else "fail", reference_passage_ids=refs,
               reference_sources=sorted({s for pid in refs for s in sources.get(pid, [])}))

    off_release = q["release_tag"] is not None and q["release_tag"] != cfg["corpus"]["release"]
    if conflict or off_release:
        code = "X2"
    elif verdict == "no_support":
        code = "X2" if (conflict or off_release) else "X1"
    elif verdict == "partial":
        code = "X3"
    elif not refs:
        code = "X7"
    else:
        quotes = [e["quote"] for e in evidence if e["quote_ok"]][:5]
        a = verifier_response(cfg, adjudicate_messages(q, quotes), ADJUDICATE_SCHEMA, 64)
        rec["tokens_in"] += a["in_tokens"]
        rec["tokens_out"] += a["out_tokens"]
        aj = a["json"] or {}
        rec["adjudication"] = aj
        code = "X4" if aj.get("other_option_defensible") else ("X5" if aj.get("answerable_without_standards_text") else None)
    rec["decision"] = "exclude" if code else "include"
    rec["reason_code"] = code
    return rec


def export_answers(cfg: dict, questions: list[dict], records: list[dict]) -> None:
    by_id = {q["question_id"]: q for q in questions}
    latest = {record["question_id"]: record for record in records}
    answers = []
    for qid, record in latest.items():
        if qid not in by_id or record.get("validation_version") != VALIDATION_VERSION or record.get("decision") != "include" or record.get("status") != "ok":
            continue
        citations = [item for item in record.get("evidence", []) if item.get("quote_ok") and item.get("source_validated")]
        if not citations:
            continue
        question = by_id[qid]
        if question.get("release_tag") != cfg["corpus"]["release"]:
            continue
        answers.append(question | {"answer_text": question["options"][question["answer"] - 1],
                                   "citations": citations, "validation_version": VALIDATION_VERSION,
                                   "supported_specs": sorted({citation["spec"] for citation in citations}),
                                   "supported_series": sorted({citation["spec"].split(".")[0] for citation in citations})})
    write_jsonl(work(cfg, "benchmark", "validated_answers.jsonl"), answers)
    write_jsonl(work(cfg, "benchmark", "questions_scoped.jsonl"), answers)
    with work(cfg, "benchmark", "validated_answers.csv").open("w", newline="") as output:
        fields = ["question_id", "question", "answer", "answer_text", "release", "spec", "version", "clause",
                  "folder", "md_name", "md_path", "docx_name", "docx_path", "quote", "md_sha256", "docx_sha256"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for answer in answers:
            for citation in answer["citations"]:
                writer.writerow({field: answer.get(field, citation.get(field, "")) for field in fields})
    print(f"Raw MD + DOCX validated answers: {len(answers)}")


def filter_existing(cfg: dict, R: Retriever, questions: list[dict], records: list[dict]) -> list[dict]:
    raw_sources = RawEvidence(cfg)
    allowed_questions = {q["question_id"] for q in questions if q.get("release_tag") == cfg["corpus"]["release"]}
    filtered = []
    for record in {row["question_id"]: row for row in records}.values():
        if record["question_id"] not in allowed_questions or record.get("status") != "ok" or record.get("decision") != "include":
            continue
        if record.get("verifier_verdict") != "supports" or record.get("release_conflict"):
            continue
        evidence = []
        allowed_passages = set(record.get("checked_passage_ids", []))
        for item in record.get("evidence", []):
            passage = R.by_id.get(item.get("passage_id"))
            if not passage or passage["id"] not in allowed_passages or not item.get("quote_ok"):
                continue
            if passage.get("release") != cfg["corpus"]["release"] or passage.get("spec") not in cfg["corpus"]["specs"]:
                continue
            checked = raw_sources.check(passage, item.get("quote", ""))
            if checked["source_validated"] and quote_in_text(item.get("quote", ""), passage["text"]):
                evidence.append(checked | {"quote_ok": True})
        if evidence:
            filtered.append(record | {"evidence": evidence, "validation_version": VALIDATION_VERSION,
                                      "reference_passage_ids": sorted({item["passage_id"] for item in evidence}),
                                      "verification_reused": True})
    return filtered


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--retry-errors", action="store_true")
    ap.add_argument("--filter-existing", type=Path, help="Filter existing verifier records using the configured release/specs and both raw formats; no LLM calls")
    args = ap.parse_args()
    cfg = load_config()
    R = Retriever(cfg)
    raw_sources = RawEvidence(cfg)
    questions = read_jsonl(work(cfg, "questions_all.jsonl"))[: args.limit]
    if args.filter_existing:
        if not args.filter_existing.is_file():
            raise SystemExit(f"Verification file not found: {args.filter_existing}")
        filtered = filter_existing(cfg, R, questions, read_jsonl(args.filter_existing))
        write_jsonl(work(cfg, "benchmark", "scoped_verification.jsonl"), filtered)
        export_answers(cfg, questions, filtered)
        return
    out = work(cfg, "benchmark", "verification.jsonl")
    existing = {r["question_id"]: r for r in read_jsonl(out) if r.get("validation_version") == VALIDATION_VERSION}
    if args.retry_errors:  # drop error rows so they are re-run; ok rows are kept
        write_jsonl(out, [r for r in existing.values() if r["status"] == "ok"])
        existing = {k: v for k, v in existing.items() if v["status"] == "ok"}
    todo = [q for q in questions if q["question_id"] not in existing]
    print(f"{len(existing)} done, {len(todo)} to verify with {cfg['models']['verifier']}")
    if not todo:
        export_answers(cfg, questions, read_jsonl(out))
        return
    warmup(cfg, "verifier")
    for q in tqdm(todo):
        t0 = time.perf_counter()
        try:
            rec = verify_question(cfg, R, q, raw_sources)
        except Exception as e:
            rec = {"question_id": q["question_id"], "category": q["category"], "status": "error", "error": str(e)[:500],
                   "validation_version": VALIDATION_VERSION}
        rec["seconds"] = round(time.perf_counter() - t0, 2)
        append_jsonl(out, [rec])

    export_answers(cfg, questions, read_jsonl(out))
    rows = list({r["question_id"]: r for r in read_jsonl(out) if r.get("validation_version") == VALIDATION_VERSION}.values())
    ok = [r for r in rows if r["status"] == "ok"]
    errors = len(rows) - len(ok)
    print(f"ok={len(ok)} errors={errors} include={sum(r['decision'] == 'include' for r in ok)}")
    if errors:
        raise SystemExit(f"{errors} verification errors remain; rerun with --retry-errors.")


if __name__ == "__main__":
    main()
