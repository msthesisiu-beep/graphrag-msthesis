"""Retry unresolved TeleQnA checks, then start the separate synthetic workflow."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import os
import shutil
import subprocess
import sys
from pathlib import Path

from common import load_config, read_jsonl, unload_ollama_models, work


def require_complete(questions: list[dict], records: list[dict]) -> None:
    expected = {question["question_id"] for question in questions}
    latest = {record["question_id"]: record for record in records}
    if not expected or expected != set(latest) or any(record.get("status") != "ok" for record in latest.values()):
        raise RuntimeError("Verification incomplete or errored; synthetic generation remains blocked")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=300)
    parser.add_argument("--source-limit", type=int, default=900)
    args = parser.parse_args()
    if args.target < 1 or args.source_limit < args.target:
        parser.error("Require source-limit >= target >= 1")
    cfg = load_config()
    environment = os.environ | {"THESIS_CONFIG": cfg["_config_path"]}
    scripts = Path(__file__).resolve().parent
    with work(cfg, "synthetic", "handoff.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A verification-to-generation handoff is already running")
        verification = work(cfg, "benchmark", "verification.jsonl")
        if verification.exists():
            stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            shutil.copy2(verification, verification.with_name(f"verification.before_handoff_{stamp}.jsonl"))
        subprocess.run([sys.executable, "-u", str(scripts / "03_verify_answerability.py"), "--retry-errors"],
                       env=environment, check=True)
        questions = read_jsonl(work(cfg, "questions_all.jsonl"))
        records = read_jsonl(verification)
        require_complete(questions, records)
        print(f"All {len(questions)} TeleQnA records successful; starting synthetic target {args.target}", flush=True)
        for phase in ("generate", "verify", "export"):
            if phase != "export":
                unload_ollama_models(cfg)
            subprocess.run([sys.executable, "-u", str(scripts / "11_generate_questions.py"), phase,
                            "--target", str(args.target), "--source-limit", str(args.source_limit)],
                           env=environment, check=True)


if __name__ == "__main__":
    main()