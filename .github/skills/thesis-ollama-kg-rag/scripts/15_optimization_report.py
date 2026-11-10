"""Read-only performance/quality diagnostics; never tune on held-out test results."""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import statistics
from pathlib import Path

from common import load_config, read_jsonl, work, write_json


def finite_mean(rows: list[dict], field: str) -> float | None:
    values = []
    for row in rows:
        try:
            value = float(row[field])
        except (KeyError, ValueError, TypeError):
            continue
        if math.isfinite(value):
            values.append(value)
    return statistics.mean(values) if values else None


def recommendations(conditions: list[dict], quality: dict, budget_violations: int) -> list[str]:
    if not conditions or any(row.get("errors", 0) or row.get("missing", 0) for row in conditions):
        return ["Finish all pilot conditions and resolve execution errors before selecting accuracy-tuning parameters."]
    if budget_violations:
        return ["Matched-context evidence exceeds T_q. Repair budget enforcement in a separately versioned run before comparing or tuning accuracy."]
    if any(values.get("provenance_ok", 1) is not None and values.get("provenance_ok", 1) < 1 for values in quality.values()):
        return ["Resolve graph provenance failures before tuning graph parameters."]
    output = []
    baseline = next((row for row in conditions if row["condition"] == "B:-:-"), None)
    if baseline:
        recall = quality.get("B:-:-", {}).get("recall_at_k")
        if recall is not None and recall < 1:
            output.append("Audit missed reference passages; compare retrieval candidate counts and reranker top-k on reviewed development questions only.")
        for row in conditions:
            if row["condition"].startswith("C:") and row["condition"].endswith(":matched"):
                delta = row["accuracy"] - baseline["accuracy"]
                output.append(f"Observed C-minus-B pilot accuracy difference: {100 * delta:+.1f} percentage points. Treat as descriptive, not a significance claim.")
                output.append("After human review, use the existing 18-variant development grid for triple count, hops, and graph share; freeze the chosen variant before test-main.")
    if any(values.get("citation_validity") is not None and values["citation_validity"] < 1 for values in quality.values()):
        output.append("Inspect invalid citation labels and source mappings before changing retrieval settings; do not silently rewrite frozen prompts.")
    output.append("Compare accuracy against latency and evidence tokens as a Pareto trade-off; do not select a configuration solely for maximum accuracy on these 20 questions.")
    output.append("Use a separate reviewed development set to validate any proposed change. Leave the held-out test set untouched.")
    return output


def build_report(directory: Path) -> dict:
    profile = {}
    for name in ("extract_raw.jsonl", "verify_raw.jsonl"):
        rows = read_jsonl(directory / "kg" / name)
        field = "passage_id"
        latest = {row[field]: row for row in rows} if name == "extract_raw.jsonl" else None
        successful = [row for row in (latest.values() if latest is not None else rows) if row.get("status") == "ok"]
        profile[name] = {"records": len(rows), "successful_records": len(successful),
                         "mean_recorded_seconds": finite_mean(successful, "seconds"),
                         "mean_output_tokens": finite_mean(successful, "out_tokens"),
                         "errors_logged": sum(row.get("status") == "error" for row in rows)}
    extract = {row["passage_id"]: row for row in read_jsonl(directory / "kg" / "extract_raw.jsonl")}
    profile["mean_triples_per_successful_passage"] = statistics.mean(
        len(row.get("triples", [])) for row in extract.values() if row.get("status") == "ok"
    ) if any(row.get("status") == "ok" for row in extract.values()) else None
    accuracy_path = directory / "accuracy_summary.json"
    conditions = json.loads(accuracy_path.read_text()).get("conditions", []) if accuracy_path.exists() else []
    metrics_path = directory / "runs" / "dev" / "per_question_metrics.csv"
    grouped = {}
    if metrics_path.exists():
        with metrics_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                grouped.setdefault(row["cond"], []).append(row)
    fields = ("recall_at_k", "mrr", "citation_validity", "citation_support", "provenance_ok", "gold_linked_precision",
              "evidence_tokens", "t_retrieve", "t_rerank", "t_kg", "t_generate", "t_total")
    quality = {condition: {field: finite_mean(rows, field) for field in fields} for condition, rows in grouped.items()}
    logs = {}
    for row in read_jsonl(directory / "runs" / "dev" / "log.jsonl"):
        key = row["question_id"], row["cond"], row["run"]
        if key not in logs or row.get("status") == "ok" or logs[key].get("status") != "ok":
            logs[key] = row
    violations = sum(row.get("status") == "ok" and row.get("mode") == "matched"
                     and row.get("evidence_tokens", 0) > row.get("T_q", 0) for row in logs.values())
    return {"scope": "exploratory_pilot_only", "automatic_parameter_changes": False,
            "human_review_required": True, "profile": profile, "conditions": conditions,
            "quality_and_latency": quality, "matched_budget_violations": violations,
            "recommendations": recommendations(conditions, quality, violations),
            "safe_runtime_optimization": "KG source selection now releases encoder models before extractor warmup on subsequent extraction processes; no measured speedup claimed."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--after-pilot-pid", type=int)
    args = parser.parse_args()
    if args.after_pilot_pid:
        pilot = importlib.import_module("13_pilot20")
        pilot.wait_for_handoff(args.after_pilot_pid, expected_script="13_pilot20.py")
    cfg = load_config()
    directory = work(cfg, "pilot20", "optimization_report.json").parent
    report = build_report(directory)
    write_json(directory / "optimization_report.json", report)
    lines = ["# Pilot Optimization Diagnostics", "", "Exploratory only. No configuration was automatically changed.", "",
             f"Matched-budget violations: {report['matched_budget_violations']}", "", "## Recommendations", ""]
    lines.extend(f"- {item}" for item in report["recommendations"])
    lines += ["", "## Runtime", "", report["safe_runtime_optimization"], "",
              "Numeric phase profiles, condition accuracy, retrieval/citation metrics, and latency breakdowns are in optimization_report.json."]
    (directory / "OPTIMIZATION_REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()