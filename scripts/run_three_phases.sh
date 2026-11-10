#!/bin/bash
# Master execution script: Run all 3 phases in parallel
# Phase A: KG Verification (continue from current state)
# Phase B: Pilot Configurations 
# Phase C: LLM Ensemble Validation
#
# Usage: bash run_three_phases.sh

set -e

WORK_DIR="/Users/rks364@apac.comcast.com/Downloads/Final_thesis"
SCRIPTS_DIR="$WORK_DIR/.github/skills/thesis-ollama-kg-rag/scripts"
PILOT_DIR="$WORK_DIR/work_tspec17/pilot20"

echo "🚀 THESIS 3-PHASE PARALLEL EXECUTION"
echo "====================================="
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Work Directory: $WORK_DIR"
echo ""

cd "$WORK_DIR"

# Ensure directories exist
mkdir -p work_tspec17/pilot20/kg
mkdir -p work_tspec17/benchmark

echo "📋 PRE-FLIGHT CHECKS"
echo "===================="

# Check Ollama status
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "❌ Ollama not running. Start with: brew services start ollama"
    exit 1
fi
echo "✅ Ollama is running"

# Check config
if [ ! -f ".github/skills/thesis-ollama-kg-rag/assets/config.yaml" ]; then
    echo "❌ Config not found at .github/skills/thesis-ollama-kg-rag/assets/config.yaml"
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
echo "🔄 PHASE A: KG VERIFICATION & GRAPH BUILD"
echo "=========================================="
echo "Status: Verify KG extraction on 573 passages, then build NetworkX graph"
echo "Steps: 1) Verify 721 passages, 2) Build graph.pkl"
echo ""
echo "Starting Phase A: Verify + Build"
{
  echo "Starting KG verification at $(date '+%Y-%m-%d %H:%M:%S')"
  python "$SCRIPTS_DIR/06_build_kg.py" verify
  echo "Verification complete at $(date '+%Y-%m-%d %H:%M:%S')"
  echo ""
  echo "Starting KG graph build at $(date '+%Y-%m-%d %H:%M:%S')"
  python "$SCRIPTS_DIR/06_build_kg.py" build
  echo "Graph build complete at $(date '+%Y-%m-%d %H:%M:%S')"
} > "$WORK_DIR/phase_a_verify.log" 2>&1 &
PHASE_A_PID=$!
echo "✓ Phase A started (PID: $PHASE_A_PID) - Running verify + build"

echo ""
echo "🎬 PHASE B: PILOT CONFIGURATIONS"
echo "================================"
echo "Status: Run 20-question pilot with 7 configs + 2 controls"
echo "Configs: 0, A, B, C[Matched], D[Joint], RAND, UNF"
echo ""
echo "Starting: python $SCRIPTS_DIR/07_run_configs.py --plan pilot --limit 20"
sleep 2  # Offset to avoid model load conflicts
nohup python "$SCRIPTS_DIR/07_run_configs.py" --plan pilot --limit 20 > "$WORK_DIR/phase_b_pilot.log" 2>&1 &
PHASE_B_PID=$!
echo "✓ Phase B started (PID: $PHASE_B_PID)"

echo ""
echo "🔬 PHASE C: LLM ENSEMBLE VALIDATION"
echo "==================================="
echo "Status: Score all 578 questions with 3 independent judges"
echo "Judges: llama2:13b, mistral:7b, neural-chat:7b"
echo "Gate: Fleiss' κ ≥ 0.60 + consensus ≥ 85%"
echo ""
echo "Starting: python $SCRIPTS_DIR/04b_llm_ensemble_validation.py --full"
sleep 2
nohup python "$SCRIPTS_DIR/04b_llm_ensemble_validation.py" --full > "$WORK_DIR/phase_c_validation.log" 2>&1 &
PHASE_C_PID=$!
echo "✓ Phase C started (PID: $PHASE_C_PID)"

echo ""
echo "📊 EXECUTION STATUS"
echo "==================="
echo "Phase A (KG Verify):     PID $PHASE_A_PID → $WORK_DIR/phase_a_verify.log"
echo "Phase B (Pilot Configs): PID $PHASE_B_PID → $WORK_DIR/phase_b_pilot.log"
echo "Phase C (LLM Validation):PID $PHASE_C_PID → $WORK_DIR/phase_c_validation.log"
echo ""
echo "Monitor with:"
echo "  tail -f $WORK_DIR/phase_a_verify.log"
echo "  tail -f $WORK_DIR/phase_b_pilot.log"
echo "  tail -f $WORK_DIR/phase_c_validation.log"
echo ""
echo "⏱️  ESTIMATED COMPLETION"
echo "======================="
echo "Phase A (KG Verify):      ~1.2 hours"
echo "Phase B (Pilot Configs):  ~2-4 hours"
echo "Phase C (LLM Validation): ~2 hours"
echo "Total (Parallel):         ~2-4 hours wall-clock"
echo ""
echo "Next steps after completion:"
echo "1. Check phase_c_validation.log for Fleiss' κ and gate PASS/FAIL"
echo "2. Review $WORK_DIR/work_tspec17/pilot20/kg/kg_stats.json (graph metrics)"
echo "3. Check $WORK_DIR/work_tspec17/pilot20/runs/pilot/log.jsonl (pilot results)"
echo "4. View $WORK_DIR/work_tspec17/benchmark/llm_ensemble_validation_report.json (validation report)"
echo ""

# Wait for all phases to complete
wait $PHASE_A_PID $PHASE_B_PID $PHASE_C_PID

echo "✅ ALL PHASES COMPLETED"
echo "========================"
echo "Outputs:"
echo "  KG Graph:            $WORK_DIR/work_tspec17/pilot20/kg/graph.pkl"
echo "  Pilot Results:       $WORK_DIR/work_tspec17/pilot20/runs/pilot/log.jsonl"
echo "  Validation Report:   $WORK_DIR/work_tspec17/benchmark/llm_ensemble_validation_report.json"
echo "  Disagreement Cases:  $WORK_DIR/work_tspec17/benchmark/llm_ensemble_disagreement_cases.jsonl"
echo ""
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
