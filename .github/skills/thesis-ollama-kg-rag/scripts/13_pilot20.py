"""Isolated exploratory end-to-end pilot after synthetic generation completes."""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import importlib
import json
import os
import select
import subprocess
import sys
from collections import defaultdict
from contextlib import closing
from pathlib import Path

import yaml

from common import RawEvidence, load_config, read_jsonl, sha256_file, stable_int, unload_ollama_models, work, write_json, write_jsonl


def wait_for_handoff(pid: int, expected_script: str = "12_verify_then_generate.py") -> None:
    process = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True)
    if process.returncode:
        return
    if expected_script not in process.stdout:
        raise RuntimeError("Refusing to wait for an unrelated process")
    with closing(select.kqueue()) as queue:
        event = select.kevent(pid, filter=select.KQ_FILTER_PROC,
                              flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT, fflags=select.KQ_NOTE_EXIT)
        try:
            queue.control([event], 0)
        except ProcessLookupError:
            return
        print(f"Waiting for synthetic handoff PID {pid} to exit", flush=True)
        queue.control(None, 1, None)


def completed_synthetic(cfg: dict) -> list[dict]:
    plan_path = work(cfg, "synthetic", "plan.json")
    plan = json.loads(plan_path.read_text())
    generated = {row["passage_id"]: row for row in read_jsonl(work(cfg, "synthetic", "generation.jsonl"))}
    selected = set(plan["selected_passage_ids"])
    if not selected or any(generated.get(pid, {}).get("status") not in {"ok", "rejected"} for pid in selected):
        raise RuntimeError("Synthetic creation is incomplete; pilot blocked")
    expected = {generated[pid]["candidate"]["question_id"] for pid in selected if generated[pid]["status"] == "ok"}
    verified = {row["question_id"]: row for row in read_jsonl(work(cfg, "synthetic", "verification.jsonl"))}
    if not expected or any(verified.get(qid, {}).get("status") != "ok" for qid in expected):
        raise RuntimeError("Synthetic verification is incomplete or errored; pilot blocked")
    answers = read_jsonl(work(cfg, "synthetic", "questions.jsonl"))
    if any(question["question_id"] not in expected or not verified[question["question_id"]].get("accepted") for question in answers):
        raise RuntimeError("Synthetic export contains unverified questions")
    return answers


def sample_questions(questions: list[dict], count: int, seed: int) -> list[dict]:
    groups = defaultdict(list)
    for question in questions:
        groups[question["citations"][0]["spec"]].append(question)
    for group in groups.values():
        group.sort(key=lambda question: (stable_int(seed, question["question_id"]), question["question_id"]))
    selected = []
    for position in range(max((len(group) for group in groups.values()), default=0)):
        for spec in sorted(groups):
            if position < len(groups[spec]):
                selected.append(groups[spec][position])
                if len(selected) == count:
                    return selected
    raise RuntimeError(f"Need {count} accepted synthetic questions; only {len(selected)} available")


def prepare_pilot(cfg: dict, directory: Path) -> dict:
    questions = sample_questions(completed_synthetic(cfg), 20, cfg["seed"])
    passages = read_jsonl(work(cfg, "passages.jsonl"))
    by_id = {passage["id"]: passage for passage in passages}
    raw = RawEvidence(cfg)
    for question in questions:
        if question.get("dataset") != "synthetic" or question.get("release_tag") != 17:
            raise RuntimeError("Pilot accepts only labelled synthetic Release 17 questions")
        for citation in question["citations"]:
            if not raw.check(by_id[citation["passage_id"]], citation["quote"])["source_validated"]:
                raise RuntimeError("Pilot citation failed raw MD/DOCX validation")
    directory.mkdir(parents=True, exist_ok=True)
    pilot = copy.deepcopy(cfg)
    pilot.pop("_config_path", None)
    pilot.pop("_config_sha256", None)
    pilot["paths"]["work_dir"] = str(directory)
    pilot["run"]["repeats"] = 1
    pilot["kg"]["scope"] = "candidates"
    pilot["retrieval"]["rerank_threshold"] = None
    inputs = [work(cfg, "passages.jsonl"), work(cfg, "index", "bge.faiss"),
              work(cfg, "index", "e5.faiss"), work(cfg, "index", "bm25.pkl")]
    manifest = {"protocol": "exploratory-synthetic-pilot20-v1", "human_review_required": True,
                "question_ids": [question["question_id"] for question in questions], "seed": cfg["seed"],
                "config": pilot, "source_files": {str(path): sha256_file(path) for path in inputs},
                "questions_sha256": sha256_file(work(cfg, "synthetic", "questions.jsonl")),
                "threshold_policy": "No pilot-answer calibration; rerank top-k without a threshold"}
    manifest_path = directory / "pilot_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("Pilot inputs changed; use a different pilot directory")
    for source in inputs:
        relative = source.relative_to(Path(cfg["paths"]["work_dir"]))
        destination = directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            if destination.resolve() != source.resolve():
                raise RuntimeError(f"Existing pilot link differs: {destination}")
        elif destination.exists():
            raise RuntimeError(f"Refusing to overwrite pilot input: {destination}")
        else:
            destination.symlink_to(source.resolve())
    write_json(manifest_path, manifest)
    write_jsonl(work(pilot, "questions_all.jsonl"), questions)
    work(pilot, "split", "dev_ids.txt").write_text("\n".join(manifest["question_ids"]) + "\n")
    work(pilot, "split", "test_ids.txt").write_text("")
    classify = importlib.import_module("05_split").classify
    with work(pilot, "benchmark", "question_log.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question_id", "reference_passage_ids", "question_type", "hop_type", "retriever_found"])
        writer.writeheader()
        for question in questions:
            refs = sorted({citation["passage_id"] for citation in question["citations"]})
            question_type, hop_type = classify(question, [by_id[pid] for pid in refs])
            writer.writerow({"question_id": question["question_id"], "reference_passage_ids": ";".join(refs),
                             "question_type": question_type, "hop_type": hop_type, "retriever_found": ""})
    path = directory / "pilot_config.yaml"
    path.write_text(yaml.safe_dump(pilot, sort_keys=False))
    return pilot | {"_config_path": str(path.resolve())}


def accuracy_summary(rows: list[dict], question_ids: set[str], conditions: list[str]) -> list[dict]:
    latest = {}
    for row in rows:
        if row["question_id"] in question_ids and row["cond"] in conditions and row["run"] == 0:
            key = row["question_id"], row["cond"]
            if key not in latest or row.get("status") == "ok" or latest[key].get("status") != "ok":
                latest[key] = row
    summaries = []
    for condition in conditions:
        records = [latest[(qid, condition)] for qid in sorted(question_ids) if (qid, condition) in latest]
        successes = [row for row in records if row.get("status") == "ok"]
        correct = sum(bool(row.get("valid")) and row.get("correct") == 1 for row in successes)
        summaries.append({"condition": condition, "expected": len(question_ids), "successful": len(successes),
                          "errors": sum(row.get("status") != "ok" for row in records),
                          "missing": len(question_ids) - len(records), "correct": correct,
                          "accuracy": correct / len(question_ids),
                          "invalid_answers": sum(not row.get("valid") for row in successes),
                          "mean_latency_seconds": sum(row["t_total"] for row in successes) / len(successes) if successes else None})
    return summaries


def report(cfg: dict) -> None:
    directory = Path(cfg["paths"]["work_dir"])
    manifest = json.loads((directory / "pilot_manifest.json").read_text())
    variant = importlib.import_module("07_run_configs").vname(cfg["kg"]["default_variant"])
    conditions = ["0:-:-", "A:-:matched", "A:-:overhead", "B:-:-", f"D:{variant}:matched"]
    conditions += [f"{name}:{variant}:{mode}" for name in ("C", "RAND", "UNF") for mode in ("matched", "overhead")]
    summary = accuracy_summary(read_jsonl(work(cfg, "runs", "dev", "log.jsonl")), set(manifest["question_ids"]), conditions)
    write_json(directory / "accuracy_summary.json", {"exploratory": True, "human_review_pending": True, "conditions": summary})
    lines = ["# Exploratory Synthetic Pilot: 20 Questions", "", "Not a thesis test result. Human review is pending. Each question changes accuracy by 5 percentage points.", "",
             "Thresholds are not fitted to these 20 answers. Failed and missing runs remain in the accuracy denominator.", "",
             "| Condition | Correct / 20 | Accuracy | Errors | Missing | Mean Latency (s) |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary:
        latency = f"{row['mean_latency_seconds']:.1f}" if row["mean_latency_seconds"] is not None else "n/a"
        lines.append(f"| {row['condition']} | {row['correct']} / {row['expected']} | {100 * row['accuracy']:.1f}% | {row['errors']} | {row['missing']} | {latency} |")
    lines += ["", "Retrieval, citation, evidence-budget, and provenance metrics: runs/dev/per_question_metrics.csv.",
              "These are descriptive results only; no significance or generalisation claim is supported by this pilot."]
    (directory / "PILOT_REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    if any(row["errors"] or row["missing"] for row in summary):
        raise RuntimeError("Pilot has failed or missing runs; report includes them, rerun to recover")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--after-pid", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    source = load_config()
    directory = Path(source["paths"]["work_dir"]) / "pilot20"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Pilot20 is already scheduled or running")
        if args.after_pid:
            wait_for_handoff(args.after_pid)
        cfg = prepare_pilot(source, directory)
        if args.prepare_only:
            return
        environment = os.environ | {"THESIS_CONFIG": cfg["_config_path"]}
        scripts = Path(__file__).resolve().parent
        commands = [("06_build_kg.py", "extract"), ("06_build_kg.py", "verify"), ("06_build_kg.py", "build"),
                    ("07_run_configs.py", "--plan", "pilot", "--limit", "20"), ("08_evaluate.py", "--split", "dev")]
        try:
            for name, *arguments in commands:
                unload_ollama_models(source)
                print(f"Starting pilot stage: {name} {' '.join(arguments)}", flush=True)
                subprocess.run([sys.executable, "-u", str(scripts / name), *arguments], env=environment, check=True)
        finally:
            if work(cfg, "runs", "dev", "log.jsonl").exists():
                report(cfg)


if __name__ == "__main__":
    main()