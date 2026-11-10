"""Export concise answer rationales and uncalibrated evidence-check scores.

This does not change generation prompts, frozen plans, or benchmark acceptance.
"""
from __future__ import annotations

import argparse
import csv
import importlib
from pathlib import Path

from common import load_config, read_jsonl, work, write_jsonl


def annotate(candidate: dict, verification: dict | None) -> dict:
    verified = bool(verification and verification.get("status") == "ok")
    result = (verification or {}).get("response", {}).get("json") or {}
    checked = (verification or {}).get("candidate", candidate)
    citations = checked.get("citations", [])
    creator_quote = citations[0] if citations else {}
    verifier_quote = citations[-1] if verified and len(citations) > 1 else {}
    independent = bool(checked.get("verifier_model") and checked.get("question_creator")
                       and checked["verifier_model"] != checked["question_creator"])
    checks = {
        "creator_quote_in_md_and_docx": bool(creator_quote.get("raw_md_ok") and creator_quote.get("raw_docx_ok")),
        "verifier_quote_in_md_and_docx": bool(verifier_quote.get("raw_md_ok") and verifier_quote.get("raw_docx_ok")),
        "independent_verifier": independent,
        "answer_agreement": type(result.get("answer")) is int and result["answer"] == candidate["answer"],
        "supported_unambiguous": result.get("supported") is True and result.get("ambiguous") is False,
        "verifier_rationale_present": bool(str(checked.get("verified_explanation", "")).strip()),
    }
    weights = {"creator_quote_in_md_and_docx": 15, "verifier_quote_in_md_and_docx": 25,
               "independent_verifier": 15, "answer_agreement": 20, "supported_unambiguous": 20,
               "verifier_rationale_present": 5}
    score = sum(weights[key] for key, passed in checks.items() if passed) if verified else None
    options = candidate["options"]
    answer_text = options[candidate["answer"] - 1]
    rationale = checked.get("verified_explanation", "").strip() if verified else ""
    return checked | {
        "answer_text": answer_text, "answer_rationale": rationale or candidate.get("explanation", ""),
        "rationale_source": "independent_verifier" if rationale else "question_creator_unverified",
        "creator_rationale": candidate.get("explanation", ""),
        "verifier_rationale": rationale, "confidence_score": score,
        "confidence_method": "weighted_evidence_checklist_v1",
        "confidence_interpretation": "Uncalibrated review aid, not probability of correctness. A score of 100 only means all listed checks passed.",
        "confidence_checks": checks, "confidence_weights": weights,
        "verification_status": "accepted" if verified and verification.get("accepted") else ("rejected" if verified else "pending"),
        "human_review": "pending",
    }


def export_review(cfg: dict, include_candidates: bool = False) -> list[dict]:
    generated = {row["passage_id"]: row for row in read_jsonl(work(cfg, "synthetic", "generation.jsonl"))}
    records = {row["question_id"]: row for row in read_jsonl(work(cfg, "synthetic", "verification.jsonl"))}
    if include_candidates:
        candidates = [row["candidate"] for row in generated.values() if row.get("status") == "ok"]
        prefix = "candidates_review_enriched"
    else:
        candidates = read_jsonl(work(cfg, "synthetic", "questions.jsonl"))
        prefix = "questions_review_enriched"
    original = {row["candidate"]["question_id"]: row["candidate"] for row in generated.values() if row.get("status") == "ok"}
    rows = [annotate(original.get(candidate["question_id"], candidate), records.get(candidate["question_id"])) for candidate in candidates]
    path = work(cfg, "synthetic", f"{prefix}.csv")
    prior = {}
    if path.exists():
        with path.open(newline="") as handle:
            prior = {row["question_id"]: row for row in csv.DictReader(handle)}
    write_jsonl(work(cfg, "synthetic", f"{prefix}.jsonl"), rows)
    fields = ["question_id", "question", "option_1", "option_2", "option_3", "option_4", "answer", "answer_text",
              "answer_rationale", "rationale_source", "creator_rationale", "verifier_rationale", "confidence_score",
              "confidence_method", "confidence_interpretation", "verification_status", "spec", "version", "clause",
              "folder", "md_name", "docx_name", "md_path", "docx_path", "creator_quote", "verifier_quote",
              "manual_supported", "manual_confidence", "reviewer_notes"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            citations = row.get("citations", [])
            citation = citations[-1] if citations else {}
            record = {field: row.get(field, citation.get(field, "")) for field in fields}
            record.update({f"option_{number}": option for number, option in enumerate(row["options"], 1)})
            record["creator_quote"] = citations[0].get("quote", "") if citations else ""
            record["verifier_quote"] = citation.get("quote", "") if row["verification_status"] != "pending" else ""
            for field in ("manual_supported", "manual_confidence", "reviewer_notes"):
                record[field] = prior.get(row["question_id"], {}).get(field, "")
            writer.writerow(record)
    print(f"Exported {len(rows)} rationale/confidence review rows to {path}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-candidates", action="store_true", help="Preview generated candidates; pending questions have no confidence score")
    parser.add_argument("--after-pid", type=int, help="On macOS, wait for the synthetic handoff before final export")
    args = parser.parse_args()
    if args.after_pid:
        pilot = importlib.import_module("13_pilot20")
        pilot.wait_for_handoff(args.after_pid)
    cfg = load_config()
    export_review(cfg, args.include_candidates)


if __name__ == "__main__":
    main()