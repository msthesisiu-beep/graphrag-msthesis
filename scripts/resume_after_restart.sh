#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
PY="/opt/miniconda3/envs/hybrid_graphs_py311/bin/python"
SCRIPTS=".github/skills/thesis-ollama-kg-rag/scripts"
export THESIS_CONFIG="${THESIS_CONFIG:-.github/skills/thesis-ollama-kg-rag/assets/config.yaml}"
export PYTHONUNBUFFERED=1

if pgrep -f 'scripts/(03_verify_answerability|06_build_kg|07_run_configs|08_evaluate|11_generate_questions|12_verify_then_generate|13_pilot20|14_review_questions|15_optimization_report)\.py' > /dev/null; then
    printf 'A thesis worker or scheduler is already active. Do not start a duplicate workflow.\n' >&2
    exit 1
fi

printf 'Resuming TeleQnA verification and synthetic creation/verification.\n'
"$PY" "$SCRIPTS/12_verify_then_generate.py" --target 300 --source-limit 900
printf 'Exporting answer rationales, evidence-check scores, and review sheets.\n'
"$PY" "$SCRIPTS/14_review_questions.py" --include-candidates
"$PY" "$SCRIPTS/14_review_questions.py"
printf 'Starting or resuming the isolated exploratory 20-question pilot.\n'
pilot_exit=0
"$PY" "$SCRIPTS/13_pilot20.py" || pilot_exit=$?
printf 'Writing read-only performance and quality diagnostics.\n'
"$PY" "$SCRIPTS/15_optimization_report.py"
if [[ "$pilot_exit" != "0" ]]; then
    exit "$pilot_exit"
fi
printf 'Workflow finished. Human validation and final thesis experiments remain separate.\n'