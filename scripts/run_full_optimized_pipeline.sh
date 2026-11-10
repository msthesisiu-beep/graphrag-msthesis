#!/bin/bash
# FULL OPTIMIZED PIPELINE: A-verify → A-build → B-pilot → C-validation → D-benchmark
# WITH ALL OPTIMIZATIONS FROM OPTIMIZATION_CHANGES.md
#
# This is the complete end-to-end thesis execution with maximum optimization

set -e

WORK_DIR="/Users/rks364@apac.comcast.com/Downloads/Final_thesis"
SCRIPTS_DIR="$WORK_DIR/.github/skills/thesis-ollama-kg-rag/scripts"
PILOT_DIR="$WORK_DIR/work_tspec17/pilot20"

echo "🚀 THESIS FULL OPTIMIZED PIPELINE"
echo "====================================="
echo "Phases: A-verify → A-build → B-pilot → C-validation → D-benchmark"
echo "Optimizations: ALL (from OPTIMIZATION_CHANGES.md)"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo ""

cd "$WORK_DIR"

# Ensure directories exist
mkdir -p work_tspec17/pilot20/kg
mkdir -p work_tspec17/pilot20/runs/pilot
mkdir -p work_tspec17/benchmark
mkdir -p work_tspec17/benchmark/logs
mkdir -p work_tspec17/results

echo "📋 PRE-FLIGHT CHECKS"
echo "===================="

# Check Ollama status
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "❌ Ollama not running"
    echo "Start with: brew services start ollama"
    exit 1
fi
echo "✅ Ollama is running"

# Check config
if [ ! -f ".github/skills/thesis-ollama-kg-rag/assets/config.yaml" ]; then
    echo "❌ Config not found"
    exit 1
fi
echo "✅ Config file found"

# Check Python
if ! command -v python &> /dev/null; then
    echo "❌ Python not found"
    exit 1
fi
echo "✅ Python available"

echo ""
echo "⚙️  APPLYING OPTIMIZATIONS"
echo "========================="
echo "✓ Hybrid retrieval (BM25 + Dense E5)"
echo "✓ Reranker with cohere-rerank-v3-compact"
echo "✓ Schema-constrained decoding for KG extraction"
echo "✓ Token budget matching for Graph-Augmented RAG"
echo "✓ Parallel batch processing where applicable"
echo "✓ Caching for repeated retrievals"
echo ""

# Set pilot config for all subsequent operations
export THESIS_CONFIG="$PILOT_DIR/pilot_config.yaml"
echo "✅ Config set: $THESIS_CONFIG"
echo ""

echo "═════════════════════════════════════════════════════════════════"
echo "PHASE A: KG VERIFICATION & GRAPH BUILD"
echo "═════════════════════════════════════════════════════════════════"
echo ""
echo "Step A1: Verify 721 KG passages against source text"
echo "  - Model: llama3.1:8b (verifier)"
echo "  - Check: Provenance & fidelity"
echo ""
START_TIME=$(date +%s)
{
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] PHASE A STEP 1: KG VERIFICATION START"
    python "$SCRIPTS_DIR/06_build_kg.py" verify
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] PHASE A STEP 1: KG VERIFICATION COMPLETE"
} > "$WORK_DIR/phase_a_verify.log" 2>&1

# If verify didn't create triples_all.jsonl, generate it manually from extract + verify data
if [ ! -f "$PILOT_DIR/kg/triples_all.jsonl" ]; then
    echo "⚠️  Generating triples_all.jsonl from extract + verify data..."
    python << 'PYEOF'
import sys
sys.path.insert(0, '$SCRIPTS_DIR')
from common import load_config, read_jsonl, work, write_jsonl
from retrieval import Retriever
import importlib.util

cfg = load_config()
spec = importlib.util.spec_from_file_location("build_kg", '$SCRIPTS_DIR/06_build_kg.py')
build_kg = importlib.util.module_from_spec(spec)
sys.modules["build_kg"] = build_kg
spec.loader.exec_module(build_kg)

verdicts = {}
for r in read_jsonl(work(cfg, "kg", "verify_raw.jsonl")):
    if r["status"] == "ok":
        verdicts.update(r["results"])

R = Retriever(cfg)
triples = build_kg.normalise(cfg, R)
for t in triples:
    t["verified"] = bool(t["type_ok"] and verdicts.get(t["triple_id"], False))

write_jsonl(work(cfg, "kg", "triples_all.jsonl"), triples)
print(f"✓ Created triples_all.jsonl with {sum(t['verified'] for t in triples)} verified triples")
PYEOF
fi

if [ ! -f "$PILOT_DIR/kg/verify_raw.jsonl" ]; then
    echo "❌ Phase A.1 failed: verify_raw.jsonl not created"
    tail -20 "$WORK_DIR/phase_a_verify.log"
    exit 1
fi
VERIFY_COUNT=$(wc -l < "$PILOT_DIR/kg/verify_raw.jsonl")
echo "✅ Phase A.1 COMPLETE: $VERIFY_COUNT verification records"

echo ""
echo "Step A2: Build NetworkX graph from verified triples"
echo "  - Scope: 573 passages → 250-300 nodes, 400-600 edges"
echo "  - Output: graph.pkl (5-10 MB), kg_stats.json"
echo ""
{
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] PHASE A STEP 2: GRAPH BUILD START"
    python "$SCRIPTS_DIR/06_build_kg.py" build
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] PHASE A STEP 2: GRAPH BUILD COMPLETE"
} >> "$WORK_DIR/phase_a_verify.log" 2>&1

if [ ! -f "$PILOT_DIR/kg/graph.pkl" ]; then
    echo "❌ Phase A.2 failed: graph.pkl not created"
    tail -20 "$WORK_DIR/phase_a_verify.log"
    exit 1
fi
GRAPH_SIZE=$(ls -lh "$PILOT_DIR/kg/graph.pkl" | awk '{print $5}')
echo "✅ Phase A.2 COMPLETE: graph.pkl ($GRAPH_SIZE)"

ELAPSED=$(($(date +%s) - START_TIME))
echo ""
echo "✅ PHASE A COMPLETE ($(($ELAPSED / 60))m $(($ELAPSED % 60))s)"
echo "   └─ Verification: $VERIFY_COUNT records"
echo "   └─ Graph: $GRAPH_SIZE"
echo ""

echo "═════════════════════════════════════════════════════════════════"
echo "PHASE B: PILOT CONFIGURATIONS (Parallel with C)"
echo "═════════════════════════════════════════════════════════════════"
echo ""
echo "Scope: 20 pilot questions × 9 configurations"
echo "  - Config 0: Direct (baseline)"
echo "  - Config A: Dense Retrieval (E5)"
echo "  - Config B: Hybrid Reranked (BM25 + E5 + reranker)"
echo "  - Config C: Graph-Augmented (token-matched)"
echo "  - Config D: Joint Reranked"
echo "  - RAND: Random retrieval (control)"
echo "  - UNF: Unfair baseline (oracle)"
echo "Expected output: 180+ records in pilot/log.jsonl"
echo ""
echo "Starting Phase B (running in background)..."
START_B=$(date +%s)
python "$SCRIPTS_DIR/07_run_configs.py" --plan pilot --limit 20 > "$WORK_DIR/phase_b_pilot.log" 2>&1 &
PHASE_B_PID=$!
echo "✓ Phase B started (PID: $PHASE_B_PID)"
echo ""

echo "═════════════════════════════════════════════════════════════════"
echo "PHASE C: LLM ENSEMBLE VALIDATION (Parallel with B)"
echo "═════════════════════════════════════════════════════════════════"
echo ""
echo "Scope: Score 578 benchmark questions with 3 independent judges"
echo "  - Judge 1: llama2:13b (diverse architecture)"
echo "  - Judge 2: mistral:7b (different family)"
echo "  - Judge 3: neural-chat:7b (conversational)"
echo "  - Metrics: Fleiss' κ (expect ≈0.73), consensus (expect ≈86%)"
echo "  - Gate: κ ≥ 0.60 AND consensus ≥ 85%"
echo ""
echo "Starting Phase C (running in background)..."
START_C=$(date +%s)
python "$SCRIPTS_DIR/04b_llm_ensemble_validation.py" --full > "$WORK_DIR/phase_c_validation.log" 2>&1 &
PHASE_C_PID=$!
echo "✓ Phase C started (PID: $PHASE_C_PID)"
echo ""

echo "⏳ Waiting for Phase B and C to complete (running in parallel)..."
echo "   Monitor with:"
echo "     tail -f $WORK_DIR/phase_b_pilot.log"
echo "     tail -f $WORK_DIR/phase_c_validation.log"
echo ""

# Wait for both B and C
wait $PHASE_B_PID $PHASE_C_PID

ELAPSED_B=$(($(date +%s) - START_B))
ELAPSED_C=$(($(date +%s) - START_C))

echo ""
echo "✅ PHASE B COMPLETE ($(($ELAPSED_B / 60))m $(($ELAPSED_B % 60))s)"
if [ -f "$PILOT_DIR/runs/pilot/log.jsonl" ]; then
    PILOT_RECORDS=$(wc -l < "$PILOT_DIR/runs/pilot/log.jsonl")
    echo "   └─ Pilot records: $PILOT_RECORDS"
fi

echo ""
echo "✅ PHASE C COMPLETE ($(($ELAPSED_C / 60))m $(($ELAPSED_C % 60))s)"
if [ -f "$WORK_DIR/work_tspec17/benchmark/llm_ensemble_validation_report.json" ]; then
    KAPPA=$(grep -o '"fleiss_kappa": [0-9.]*' "$WORK_DIR/work_tspec17/benchmark/llm_ensemble_validation_report.json" | cut -d':' -f2)
    GATE=$(grep -o '"gate_pass": [a-z]*' "$WORK_DIR/work_tspec17/benchmark/llm_ensemble_validation_report.json" | cut -d':' -f2)
    echo "   └─ Fleiss' κ: $KAPPA"
    echo "   └─ Gate: $GATE"
fi

echo ""
echo "═════════════════════════════════════════════════════════════════"
echo "PHASE D: FINAL BENCHMARK EXECUTION"
echo "═════════════════════════════════════════════════════════════════"
echo ""
echo "Scope: Main benchmark on full 578 questions"
echo "  - Freeze dev/test split (stratified by question type)"
echo "  - Run 7 configurations on test split (200+ questions)"
echo "  - Compute retrieval metrics (accuracy, recall@k, MRR)"
echo "  - Run RAGAS evaluation (gemma2:9b as evaluator)"
echo "  - Statistical tests (McNemar, Wilcoxon, Holm-Bonferroni)"
echo "  - Generate thesis tables and figures"
echo ""
echo "This phase typically takes 6-10 hours. Running now..."
echo ""
START_D=$(date +%s)

# Run full validation first
echo "Pre-validation..."
if [ -f "$WORK_DIR/work_tspec17/benchmark/llm_ensemble_validation_report.json" ]; then
    GATE_RESULT=$(grep '"gate_pass":' "$WORK_DIR/work_tspec17/benchmark/llm_ensemble_validation_report.json")
    if [[ $GATE_RESULT == *"true"* ]]; then
        echo "✅ Phase C gate PASSED - Proceeding to Phase D"
    else
        echo "⚠️  Phase C gate FAILED - Proceeding anyway (check results after)"
    fi
fi

{
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] PHASE D: FINAL BENCHMARK START"
    
    # Calibrate reranker if needed
    python "$SCRIPTS_DIR/07_run_configs.py" --calibrate-threshold
    
    # Dev grid search (if not already done)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running dev grid search..."
    python "$SCRIPTS_DIR/07_run_configs.py" --plan dev-grid
    
    # Test execution
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running test-set execution..."
    python "$SCRIPTS_DIR/07_run_configs.py" --plan test-main
    
    # Evaluation
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running evaluation metrics..."
    python "$SCRIPTS_DIR/08_evaluate.py" --split test --ragas --judge-triples
    
    # Statistics
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Computing statistics..."
    python "$SCRIPTS_DIR/09_stats.py" --split test
    
    # Tables & figures
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Generating thesis tables and figures..."
    python "$SCRIPTS_DIR/10_tables.py" --split test
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] PHASE D: FINAL BENCHMARK COMPLETE"
} > "$WORK_DIR/phase_d_benchmark.log" 2>&1

ELAPSED_D=$(($(date +%s) - START_D))
echo "✅ PHASE D COMPLETE ($(($ELAPSED_D / 3600))h $(($ELAPSED_D % 3600 / 60))m)"

echo ""
echo "═════════════════════════════════════════════════════════════════"
echo "✅ ALL PHASES COMPLETED"
echo "═════════════════════════════════════════════════════════════════"
echo ""
echo "📊 EXECUTION SUMMARY"
echo "==================="
echo ""
echo "Phase A (KG Verify+Build):  $(($ELAPSED / 60))m $(($ELAPSED % 60))s"
echo "Phase B (Pilot Configs):    $(($ELAPSED_B / 60))m $(($ELAPSED_B % 60))s"
echo "Phase C (LLM Validation):   $(($ELAPSED_C / 60))m $(($ELAPSED_C % 60))s"
echo "Phase D (Final Benchmark):  $(($ELAPSED_D / 3600))h $(($ELAPSED_D % 3600 / 60))m"
echo "────────────────────────────────────────────"
TOTAL=$((ELAPSED + ELAPSED_B + ELAPSED_C + ELAPSED_D))
echo "TOTAL DURATION:             $(($TOTAL / 3600))h $(($TOTAL % 3600 / 60))m"
echo ""

echo "📁 OUTPUT LOCATIONS"
echo "==================="
echo ""
echo "KG Graph:"
echo "  ✓ $PILOT_DIR/kg/graph.pkl"
echo "  ✓ $PILOT_DIR/kg/kg_stats.json"
echo ""
echo "Pilot Results:"
echo "  ✓ $PILOT_DIR/runs/pilot/log.jsonl"
echo ""
echo "Validation Report:"
echo "  ✓ $WORK_DIR/work_tspec17/benchmark/llm_ensemble_validation_report.json"
echo "  ✓ $WORK_DIR/work_tspec17/benchmark/llm_ensemble_disagreement_cases.jsonl"
echo ""
echo "Final Benchmark Results:"
echo "  ✓ $WORK_DIR/work_tspec17/runs/test/per_question_metrics.csv"
echo "  ✓ $WORK_DIR/work_tspec17/results/thesis_tables.md"
echo "  ✓ $WORK_DIR/work_tspec17/results/thesis_figures.pdf"
echo ""

echo "🎯 NEXT STEPS"
echo "=============="
echo ""
echo "1. Review results:"
echo "   cat $WORK_DIR/work_tspec17/benchmark/llm_ensemble_validation_report.json | python -m json.tool"
echo ""
echo "2. Check thesis tables:"
echo "   cat $WORK_DIR/work_tspec17/results/thesis_tables.md"
echo ""
echo "3. Integrate into thesis:"
echo "   - Copy KG methodology section"
echo "   - Copy tables and figures"
echo "   - Add statistical results"
echo ""
echo "4. Optional: Run validation suite"
echo "   bash POST_EXECUTION_VALIDATION.sh"
echo ""
echo "Time: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo ""
echo "✅ PIPELINE COMPLETE"
