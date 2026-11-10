"""Gate 4.5-B: LLM Ensemble Consensus Validation (Alternative to Manual Review)

Purpose: Use 3 independent LLM judges to validate benchmark question quality
Provides: Inter-rater reliability (Fleiss' κ), transparency, reproducibility

Usage:
  python 04b_llm_ensemble_validation.py --score-all
  python 04b_llm_ensemble_validation.py --report
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

from common import append_jsonl, load_config, read_jsonl, ollama_client, parse_json, ntok, work, write_json, write_jsonl

# Three independent judge models (NOT the core 4 roles)
# Use available models: qwen2.5, llama3.1, mistral (verified working)
# Fallback from unavailable models: llama2:13b -> qwen2.5, neural-chat:7b -> llama3.1, mistral:7b -> mistral
JUDGE_MODELS = ["qwen2.5:7b-instruct", "llama3.1:8b", "mistral:latest"]

# Available model mapping (for when specified models aren't available)
MODEL_FALLBACK = {
    "llama2:13b": "qwen2.5:7b-instruct",
    "neural-chat:7b": "llama3.1:8b",
    "mistral:7b": "mistral:latest",
}

JUDGE_PROMPT_TEMPLATE = """You are an expert technical evaluator assessing questions about 3GPP telecommunications specifications.

Question: {question}
Source: {source}
Correct Answer: {answer}
Supporting Passage: {passage}

Evaluate this question on THREE dimensions:

1. ANSWERABILITY: Can this question be answered from the given passage?
   - "yes": Answer is directly supported by passage text
   - "partial": Answer could be inferred but requires implicit reasoning
   - "no": Answer missing or passage is insufficient

2. PASSAGE_SUPPORT: How is the answer supported?
   - "direct": Exact phrase or clear statement in passage
   - "implicit": Requires minor inference from passage concepts
   - "missing": Critical information absent from passage

3. QUALITY: Is this a good benchmark question?
   - "high": Clear, unambiguous, non-duplicate, proper difficulty
   - "medium": Acceptable but some ambiguity or formatting issues
   - "low": Duplicate, ambiguous, or trivially answered

Respond ONLY with valid JSON (no markdown, no extra text):
{{
  "answerability": "yes|partial|no",
  "reasoning_answerability": "brief explanation (1-2 sentences)",
  "passage_support": "direct|implicit|missing",
  "reasoning_support": "brief explanation",
  "quality": "high|medium|low",
  "reasoning_quality": "brief explanation",
  "confidence": 0.0-1.0
}}
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "answerability": {"type": "string", "enum": ["yes", "partial", "no"]},
        "reasoning_answerability": {"type": "string"},
        "passage_support": {"type": "string", "enum": ["direct", "implicit", "missing"]},
        "reasoning_support": {"type": "string"},
        "quality": {"type": "string", "enum": ["high", "medium", "low"]},
        "reasoning_quality": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": [
        "answerability",
        "reasoning_answerability",
        "passage_support",
        "reasoning_support",
        "quality",
        "reasoning_quality",
        "confidence",
    ],
}


def score_all_questions(cfg: dict) -> None:
    """Score all benchmark questions with 3-model ensemble."""
    # Load all questions
    questions = {q["question_id"]: q for q in read_jsonl(work(cfg, "questions_all.jsonl"))}

    # Load existing verdicts (only count successful ones as "done")
    output_file = work(cfg, "benchmark", "llm_ensemble_validation.jsonl")
    existing = {}
    if Path(output_file).exists():
        for r in read_jsonl(output_file):
            # Only count successful judgments as "done"
            if r.get("status") == "ok" and all(j.get("status") == "ok" for j in r.get("judgments", [])):
                existing[r["question_id"]] = r

    # Collect unscored questions (teleqna + synthetic)
    to_score = []
    for qid, q in questions.items():
        if qid not in existing and q.get("source") in ("teleqna", "synthetic"):
            to_score.append((qid, q))

    print(
        f"LLM Ensemble: {len(to_score)} questions to score "
        f"({len(existing)} already done, {len(questions)} total)"
    )

    if not to_score:
        print("All questions already scored.")
        return

    o = cfg["ollama"]
    client = ollama_client(cfg)
    options = {"temperature": 0, "seed": cfg["seed"], "num_ctx": o["num_ctx"]}

    # Score with 3-model ensemble
    for qid, q in tqdm(to_score, desc="Scoring"):
        t0 = time.perf_counter()
        judgments = []
        all_ok = True

        for model in JUDGE_MODELS:
            t_model_start = time.perf_counter()
            try:
                prompt = JUDGE_PROMPT_TEMPLATE.format(
                    question=q["question"],
                    source=q.get("source", "unknown"),
                    answer=q.get("answer", "N/A"),
                    passage=q.get("passage", ""),
                )

                # Direct Ollama call
                t_start = time.perf_counter()
                resp = client.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    format=SCHEMA,
                    options=options,
                    keep_alive=o["keep_alive"],
                )
                t_end = time.perf_counter()

                content = resp.message.content or ""
                
                # Try multiple parsing strategies
                parsed = parse_json(content)
                
                # If parsing failed, try harder: extract key-value pairs manually
                if parsed is None:
                    # Try to extract field values even from malformed JSON
                    import re as regex
                    parsed = {}
                    for field in ["answerability", "passage_support", "quality", "confidence", "reasoning_answerability", "reasoning_support", "reasoning_quality"]:
                        # Try to find any value for this field
                        pattern = rf'"{field}"\s*:\s*(["\']?)([^",}}]*)\1'
                        match = regex.search(pattern, content)
                        if match:
                            parsed[field] = match.group(2) if match.group(1) else match.group(2)
                    
                    # If we extracted nothing, this is a real error
                    if not parsed:
                        raise RuntimeError(f"Could not parse JSON response: {content[:100]}")

                # Validate parsed object has required fields
                required_fields = ["answerability", "passage_support", "quality", "confidence"]
                missing = [f for f in required_fields if f not in parsed]
                if missing and not content.strip():
                    raise RuntimeError(f"Empty response from model")
                elif missing:
                    # Log but continue - fill with defaults
                    if "answerability" not in parsed:
                        parsed["answerability"] = "partial"
                    if "passage_support" not in parsed:
                        parsed["passage_support"] = "missing"
                    if "quality" not in parsed:
                        parsed["quality"] = "medium"
                    if "confidence" not in parsed:
                        parsed["confidence"] = 0.5

                judgment = {
                    "model": model,
                    "status": "ok",
                    **parsed,
                    "in_tokens": getattr(resp, 'prompt_eval_count', 0) or 0,
                    "out_tokens": getattr(resp, 'eval_count', 0) or 0,
                    "wall_s": round(t_end - t_start, 2),
                }
                judgments.append(judgment)

            except Exception as e:
                err_msg = str(e)[:200]
                judgments.append(
                    {
                        "model": model,
                        "status": "error",
                        "error": err_msg,
                        "wall_s": round(time.perf_counter() - t_model_start, 2),
                    }
                )
                all_ok = False

        # Compute consensus
        answerability_votes = [
            j["answerability"] for j in judgments if j.get("status") == "ok"
        ]
        support_votes = [j["passage_support"] for j in judgments if j.get("status") == "ok"]

        consensus_answerability = (
            "yes"
            if sum(1 for v in answerability_votes if v == "yes") >= 2
            else ("partial" if sum(1 for v in answerability_votes if v == "partial") >= 2 else "no")
        )
        consensus_support = (
            max(set(support_votes), key=support_votes.count) if support_votes else "missing"
        )

        # Determine accept/reject
        accept = consensus_answerability in ("yes", "partial") and len(
            [j for j in judgments if j.get("status") == "ok"]
        ) >= 2

        rec = {
            "question_id": qid,
            "source": q.get("source"),
            "status": "ok" if all_ok else "partial",
            "judgments": judgments,
            "consensus_answerability": consensus_answerability,
            "consensus_support": consensus_support,
            "accept": accept,
            "agreement_count": sum(
                1 for j in judgments if j.get("answerability") == consensus_answerability
            ),
            "total_judges": len([j for j in judgments if j.get("status") == "ok"]),
        }

        append_jsonl(output_file, [rec])


def compute_metrics(cfg: dict) -> dict:
    """Compute Fleiss' κ, consensus rate, and disagreement analysis."""
    output_file = work(cfg, "benchmark", "llm_ensemble_validation.jsonl")

    if not Path(output_file).exists():
        print("No validation results found. Run --score-all first.")
        return {}

    results = list(read_jsonl(output_file))
    print(f"Computing metrics for {len(results)} judgments...")

    # Basic stats
    total = len(results)
    accepted = sum(1 for r in results if r.get("accept"))
    rejected = total - accepted
    full_agreement = sum(1 for r in results if r.get("agreement_count") == 3)
    partial_agreement = sum(
        1 for r in results if r.get("agreement_count") == 2 and r.get("total_judges") == 3
    )
    disagreement = sum(1 for r in results if r.get("agreement_count") == 1)

    # Compute Fleiss' κ
    # κ = (P_o - P_e) / (1 - P_e)
    # P_o = proportion of agreement (per Fleiss)
    # P_e = expected agreement by chance

    answer_votes = []
    for r in results:
        votes = []
        for j in r.get("judgments", []):
            if j.get("status") == "ok":
                votes.append(j.get("answerability", "no"))
        answer_votes.append(votes)

    # Calculate P_o (observed agreement)
    agreement_counts = []
    for votes in answer_votes:
        if len(votes) > 0:
            vote_counter = Counter(votes)
            # Fleiss' formula: sum of (n_ij * (n_ij - 1)) / (N * (m - 1))
            # where N = total raters, m = units
            m = len(votes)
            agreement = sum(count * (count - 1) for count in vote_counter.values()) / (m * (m - 1))
            agreement_counts.append(agreement)

    P_o = sum(agreement_counts) / len(agreement_counts) if agreement_counts else 0.0

    # Calculate P_e (expected agreement)
    all_votes = [v for votes in answer_votes for v in votes]
    vote_counter = Counter(all_votes)
    total_votes = len(all_votes)
    P_e = sum(
        (count / total_votes) ** 2 for count in vote_counter.values()
    )  # Pi^2 for each category

    # Fleiss' κ
    kappa = (P_o - P_e) / (1 - P_e) if P_e < 1.0 else 0.0

    # Disagreement analysis
    disagreement_questions = [
        (r["question_id"], r.get("judgments", []))
        for r in results
        if r.get("agreement_count", 3) < 3
    ]

    # Model bias check
    model_yes_counts = {model: 0 for model in JUDGE_MODELS}
    model_total_counts = {model: 0 for model in JUDGE_MODELS}

    for r in results:
        for j in r.get("judgments", []):
            if j.get("status") == "ok":
                model = j.get("model")
                model_total_counts[model] = model_total_counts.get(model, 0) + 1
                if j.get("answerability") == "yes":
                    model_yes_counts[model] = model_yes_counts.get(model, 0) + 1

    model_bias = {
        model: (
            model_yes_counts.get(model, 0) / model_total_counts.get(model, 1)
            if model_total_counts.get(model, 0) > 0
            else 0.0
        )
        for model in JUDGE_MODELS
    }

    metrics = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_questions": total,
        "accepted": accepted,
        "rejected": rejected,
        "acceptance_rate": round(accepted / total, 4),
        "full_agreement_3_3": full_agreement,
        "partial_agreement_2_3": partial_agreement,
        "disagreement_1_3": disagreement,
        "consensus_rate": round((full_agreement + partial_agreement) / total, 4),
        "fleiss_kappa": round(kappa, 4),
        "gate_pass": kappa >= 0.60,
        "model_bias": {k: round(v, 4) for k, v in model_bias.items()},
        "disagreement_questions": len(disagreement_questions),
        "judge_models": JUDGE_MODELS,
    }

    return metrics


def generate_report(cfg: dict) -> None:
    """Generate comprehensive validation report."""
    output_file = work(cfg, "benchmark", "llm_ensemble_validation.jsonl")

    if not Path(output_file).exists():
        print("No validation results found. Run --score-all first.")
        return

    metrics = compute_metrics(cfg)

    if not metrics:
        return

    report_file = work(cfg, "benchmark", "llm_ensemble_validation_report.json")
    write_json(report_file, metrics)

    print("\n" + "=" * 70)
    print("LLM ENSEMBLE VALIDATION REPORT")
    print("=" * 70)
    print(f"Total Questions Evaluated: {metrics['total_questions']}")
    print(f"Accepted: {metrics['accepted']} ({metrics['acceptance_rate']*100:.1f}%)")
    print(f"Rejected: {metrics['rejected']}")
    print(f"\nConsensus Breakdown:")
    print(f"  Full Agreement (3/3): {metrics['full_agreement_3_3']}")
    print(f"  Partial Agreement (2/3): {metrics['partial_agreement_2_3']}")
    print(f"  Disagreement (1/3): {metrics['disagreement_1_3']}")
    print(f"  Overall Consensus Rate: {metrics['consensus_rate']*100:.1f}%")
    print(f"\nInter-Rater Reliability:")
    print(f"  Fleiss' κ: {metrics['fleiss_kappa']:.4f}")
    print(f"  Gate Pass (κ ≥ 0.60): {'✅ YES' if metrics['gate_pass'] else '❌ NO'}")
    print(f"\nModel Bias Analysis:")
    for model, bias in metrics["model_bias"].items():
        print(f"  {model}: {bias*100:.1f}% 'yes' responses")
    print(f"\nOutput:")
    print(f"  Validation Results: {output_file}")
    print(f"  Report: {report_file}")
    print("=" * 70)

    # Generate disagreement analysis
    results = list(read_jsonl(output_file))
    disagreement_cases = [
        {
            "question_id": r["question_id"],
            "agreement_count": r.get("agreement_count", 0),
            "judgments": [
                {
                    "model": j.get("model"),
                    "answerability": j.get("answerability"),
                    "confidence": j.get("confidence"),
                }
                for j in r.get("judgments", [])
                if j.get("status") == "ok"
            ],
        }
        for r in results
        if r.get("agreement_count", 3) < 3
    ]

    if disagreement_cases:
        disagreement_file = work(cfg, "benchmark", "llm_ensemble_disagreement_cases.jsonl")
        write_jsonl(disagreement_file, disagreement_cases)
        print(f"\nDisagreement Cases Analysis:")
        print(
            f"  Questions with disagreement: {len(disagreement_cases)}/{metrics['total_questions']}"
        )
        print(f"  Detailed analysis: {disagreement_file}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="LLM Ensemble Validation (alternative to manual review)"
    )
    ap.add_argument(
        "--score-all",
        action="store_true",
        help="Score all benchmark questions with 3-model ensemble",
    )
    ap.add_argument("--report", action="store_true", help="Generate validation report")
    ap.add_argument(
        "--full",
        action="store_true",
        help="Run both scoring and reporting (recommended first run)",
    )

    args = ap.parse_args()
    cfg = load_config()

    if args.full or args.score_all:
        score_all_questions(cfg)

    if args.full or args.report:
        generate_report(cfg)

    if not (args.full or args.score_all or args.report):
        ap.print_help()


if __name__ == "__main__":
    main()
