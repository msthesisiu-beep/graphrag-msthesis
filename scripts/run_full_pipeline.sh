#!/bin/bash
set -euo pipefail

PY="/opt/miniconda3/envs/hybrid_graphs_py311/bin/python"
SKILL_DIR=".github/skills/thesis-ollama-kg-rag/scripts"

if [[ -n "${THESIS_CONFIG:-}" ]]; then
	CONFIG_PATH="$THESIS_CONFIG"
elif [[ -f thesis_config.yaml ]]; then
	CONFIG_PATH="thesis_config.yaml"
else
	CONFIG_PATH=".github/skills/thesis-ollama-kg-rag/assets/config.yaml"
fi
CONFIG_SHA256="$(shasum -a 256 "$CONFIG_PATH" | awk '{print $1}')"
WORK_DIR="$("$PY" -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["paths"]["work_dir"])' "$CONFIG_PATH")"
LOG="$WORK_DIR/pipeline_runner.log"
CHECKPOINT_DIR="$WORK_DIR/checkpoints/pipeline"

mkdir -p "$CHECKPOINT_DIR"

run_step() {
	local step="$1"
	shift
	local marker="$CHECKPOINT_DIR/${step}.json"
	local script_sha256=""
	if [[ -f "$2" ]]; then
		script_sha256="$(shasum -a 256 "$2" | awk '{print $1}')"
	fi
	echo ">>> Starting ${step}: $(date)" | tee -a "$LOG"
	"$@" 2>&1 | tee -a "$LOG"
	printf '{"completed_at":"%s","config_sha256":"%s","script_sha256":"%s"}\n' \
		"$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$CONFIG_SHA256" "$script_sha256" > "${marker}.tmp"
	mv "${marker}.tmp" "$marker"
	echo ">>> Completed ${step}: $(date)" | tee -a "$LOG"
}

run_cached_step() {
	local step="$1"
	shift
	local outputs=()
	while [[ "$1" != "--" ]]; do
		outputs+=("$1")
		shift
	done
	shift
	local marker="$CHECKPOINT_DIR/${step}.json"
	local script_sha256=""
	if [[ -f "$2" ]]; then
		script_sha256="$(shasum -a 256 "$2" | awk '{print $1}')"
	fi
	if [[ -f "$marker" ]] && "$PY" -c 'import json,sys; m=json.load(open(sys.argv[1])); sys.exit(0 if m.get("config_sha256")==sys.argv[2] and m.get("script_sha256")==sys.argv[3] else 1)' \
		"$marker" "$CONFIG_SHA256" "$script_sha256"; then
		local output
		local outputs_exist=1
		for output in "${outputs[@]}"; do
			if [[ ! -e "$output" ]]; then
				outputs_exist=0
				break
			fi
		done
		if [[ "$outputs_exist" == "1" ]]; then
			echo ">>> Skipping ${step}: valid checkpoint and outputs exist." | tee -a "$LOG"
			return
		fi
	fi
	run_step "$step" "$@"
}

echo "=== Pipeline Runner Started: $(date) ===" | tee -a "$LOG"

# Outputs are per-question checkpoints. Preserve them on normal runs so each
# stage can resume; set CLEAN_WORK=1 only when an intentional fresh run is needed.
if [[ "${CLEAN_WORK:-0}" == "1" ]]; then
	echo ">>> CLEAN_WORK=1: removing generated benchmark and run artifacts..." | tee -a "$LOG"
	rm -rf "$WORK_DIR/benchmark" "$WORK_DIR/kg" "$WORK_DIR/split" "$WORK_DIR/runs" \
		"$WORK_DIR/reports" "$WORK_DIR/review" "$WORK_DIR/index" "$WORK_DIR/checkpoints"
	rm -f "$WORK_DIR/questions_all.jsonl" "$WORK_DIR/passages.jsonl" "$WORK_DIR/corpus_manifest.json"
	mkdir -p "$CHECKPOINT_DIR"
fi

# Gates 2-3: use the Release-filtered TeleQnA set and configured TSpec source.
run_cached_step "00_prepare_data" "$WORK_DIR/questions_all.jsonl" "$WORK_DIR/passages.jsonl" \
	"$WORK_DIR/corpus_manifest.json" -- "$PY" "$SKILL_DIR/01_prepare_data.py"
run_cached_step "00_build_indexes" "$WORK_DIR/index/bge.faiss" "$WORK_DIR/index/e5.faiss" \
	"$WORK_DIR/index/bm25.pkl" "$WORK_DIR/index/index_stats.json" -- "$PY" "$SKILL_DIR/02_build_indexes.py"

# 1. Gate 4-5: Run answerability verification from question 1
run_step "01_verification" "$PY" "$SKILL_DIR/03_verify_answerability.py" --retry-errors

# Pause for the independent manual review before freezing a benchmark split.
if [[ ! -f "$WORK_DIR/benchmark/manual_review_scores.json" ]]; then
	if [[ ! -f "$WORK_DIR/review/questions_review.csv" ]]; then
		run_step "01a_export_question_review" "$PY" "$SKILL_DIR/04_manual_review.py" export-questions
	fi
	echo ">>> Pipeline paused: review $WORK_DIR/review/questions_review.csv, fill manual_supported, then run score-questions." | tee -a "$LOG"
	exit 0
fi
"$PY" -c 'import json,sys; s=json.load(open(sys.argv[1])); sys.exit(0 if s.get("gate_passed") else "Manual-review gate failed; do not continue to split/test stages.")' \
	"$WORK_DIR/benchmark/manual_review_scores.json"

# 2. Gate 6: Split Freezing
if [[ -f "$WORK_DIR/split/FROZEN.json" ]]; then
	THESIS_WORK_DIR="$WORK_DIR" "$PY" -c '
import hashlib, json
import os
from pathlib import Path
pdir = Path(os.environ["THESIS_WORK_DIR"])
p = pdir / "split" / "FROZEN.json"
f = json.loads(p.read_text())
if not f.get("gate", {}).get("gate_passed") or f.get("skip_gate"):
		raise SystemExit("Existing split is ungated; complete manual review before test-main.")
files = {"question_log.csv": pdir / "benchmark" / "question_log.csv",
					 "dev_ids.txt": pdir / "split" / "dev_ids.txt",
					 "test_ids.txt": pdir / "split" / "test_ids.txt"}
for name, path in files.items():
		if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != f.get("sha256", {}).get(name):
				raise SystemExit(f"Frozen split checksum mismatch: {path}")
print("Reusing validated frozen split.")
'
else
	run_step "02_split" "$PY" "$SKILL_DIR/05_split.py"
fi

# 3. Gate 7: Knowledge Graph (extract -> verify -> build)
run_step "03a_kg_extract" "$PY" "$SKILL_DIR/06_build_kg.py" extract
run_step "03b_kg_verify" "$PY" "$SKILL_DIR/06_build_kg.py" verify
run_step "03c_kg_build" "$PY" "$SKILL_DIR/06_build_kg.py" build

# 4. Gate 8-9: Execution Engine (test-main with default-only)
run_step "04a_threshold" "$PY" "$SKILL_DIR/07_run_configs.py" --calibrate-threshold
run_step "04b_determinism" "$PY" "$SKILL_DIR/07_run_configs.py" --determinism-check 5

run_step "05_test_main" "$PY" "$SKILL_DIR/07_run_configs.py" --plan test-main --default-only

# 5. Gate 10: Evaluation & Scoring
run_step "06_evaluate" "$PY" "$SKILL_DIR/08_evaluate.py" --split test

# 6. Gate 11: Statistical Testing & Report Compilation
run_step "07_stats" "$PY" "$SKILL_DIR/09_stats.py" --split test

run_step "08_tables" "$PY" "$SKILL_DIR/10_tables.py" --split test

echo "=== Pipeline Completed Successfully: $(date) ===" | tee -a "$LOG"
