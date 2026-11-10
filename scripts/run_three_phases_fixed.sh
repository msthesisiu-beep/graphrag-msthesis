#!/bin/bash
# FIXED Master execution script: Run all 3 phases with proper sequencing
# Phase A: KG Verification + Graph Build (sequential, must complete)
# Phase B: Pilot Configurations (after Phase A completes)
# Phase C: LLM Ensemble Validation (can run in parallel with Phase B)
#
# KEY FIX: Phase A now includes BOTH verify AND build steps
# Usage: bash run_three_phases_fixed.sh

set -e

WORK_DIR="/Users/rks364@apac.comcast.com/Downloads/Final_thesis"
SCRIPTS_DIR="$WORK_DIR/.github/skills/thesis-ollama-kg-rag/scripts"
PILOT_DIR="$WORK_DIR/work_tspec17/pilot20"

echo "🚀 THESIS 3-PHASE EXECUTION (FIXED VERSION)"
echo "==========================================="
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Work Directory: $WORK_DIR"
echo ""

cd "$WORK_DIR"

# Ensure directories exist
mkdir -p work_tspec17/pilot20/kg
mkdir -p work_tspec17/benchmark
mkdir -p work_tspec17/pilot20/runs/pilot

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
echo "🔄 PHASE A: KG VERIFICATION + GRAPH BUILD"
echo "=========================================="
echo "Step 1: Verify 721 KG passages against source text"
echo "Step 2: Build NetworkX graph from verified triples"
echo "Expected output: graph.pkl (5-10 MB), kg_stats.json"
echo ""
echo "Starting KG verification + build..."
{
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting KG verification"
    python "$SCRIPTS_DIR/06_build_kg.py" verify
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Verification complete"
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting graph build"
    python "$SCRIPTS_DIR/06_build_kg.py" build
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Graph build complete"
} > "$WORK_DIR/phase_a_verify_build.log" 2>&1
PHASE_A_EXIT=$?

if [ $PHASE_A_EXIT -ne 0 ]; then
    echo "❌ Phase A failed with exit code $PHASE_A_EXIT"
    echo "Check log: $WORK_DIR/phase_a_verify_build.log"
    exit 1
fi
echo "✅ Phase A COMPLETE (Verify + Build)"

# Verify Phase A outputs
if [ ! -f "$PILOT_DIR/kg/graph.pkl" ]; then
    echo "❌ Phase A failed: graph.pkl not created"
    exit 1
fi
echo "✅ Verified: graph.pkl exists"

if [ ! -f "$PILOT_DIR/kg/kg_stats.json" ]; then
    echo "⚠️  Warning: kg_stats.json not created (may be non-fatal)"
fi

echo ""
echo "🎬 PHASE B: PILOT CONFIGURATIONS (Starting now)"
echo "=============================================="
echo "Scope: 20 pilot questions × 9 configurations"
echo "Configs: Direct, Dense, Hybrid, Graph-Augmented, Joint Reranked, Random, Unfair"
echo "Expected: 180+ execution records in pilot/log.jsonl"
echo ""
echo "Starting: python $SCRIPTS_DIR/07_run_configs.py --plan pilot --limit 20"
python "$SCRIPTS_DIR/07_run_configs.py" --plan pilot --limit 20 > "$WORK_DIR/phase_b_pilot.log" 2>&1 &
PHASE_B_PID=$!
echo "✓ Phase B started (PID: $PHASE_B_PID, running in background)"

echo ""
echo "🔬 PHASE C: LLM ENSEMBLE VALIDATION (Starting in parallel)"
echo "=========================================================="
echo "Scope: Score all 578 questions with 3 independent judges"
echo "Judges: llama2:13b, mistral:7b, neural-chat:7b (different from pipeline)"
echo "Metrics: Fleiss' κ (expected 0.73), consensus % (expected 86%)"
echo "Gate: κ ≥ 0.60 AND consensus ≥ 85% → PASS or FAIL"
echo ""
echo "Starting: python $SCRIPTS_DIR/04b_llm_ensemble_validation.py --full"
python "$SCRIPTS_DIR/04b_llm_ensemble_validation.py" --full > "$WORK_DIR/phase_c_validation.log" 2>&1 &
PHASE_C_PID=$!
echo "✓ Phase C started (PID: $PHASE_C_PID, running in background)"

echo ""
echo "📊 EXECUTION STATUS"
echo "==================="
echo "Phase A (KG Verify+Build): ✅ COMPLETE"
echo "Phase B (Pilot Configs):   ⏳ Running (PID $PHASE_B_PID) → $WORK_DIR/phase_b_pilot.log"
echo "Phase C (LLM Validation):  ⏳ Running (PID $PHASE_C_PID) → $WORK_DIR/phase_c_validation.log"
echo ""
echo "Monitor B & C with:"
echo "  tail -f $WORK_DIR/phase_b_pilot.log"
echo "  tail -f $WORK_DIR/phase_c_validation.log"
echo ""
echo "⏱️  ESTIMATED REMAINING TIME"
echo "=============================="
echo "Phase B (Pilot Configs):  ~2-4 hours"
echo "Phase C (LLM Validation): ~2 hours"
echo "Total (Parallel):         ~2-4 hours from now"
echo ""

# Wait for Phase B and C to complete
echo "Waiting for Phase B and C to complete..."
wait $PHASE_B_PID $PHASE_C_PID

echo ""
echo "✅ ALL PHASES COMPLETED"
echo "========================"
echo ""
echo "📋 OUTPUTS SUMMARY"
echo "=================="
echo ""
echo "Phase A Outputs:"
echo "  ✓ KG Graph:      $PILOT_DIR/kg/graph.pkl"
echo "  ✓ KG Statistics: $PILOT_DIR/kg/kg_stats.json"
echo ""
echo "Phase B Outputs:"
echo "  ✓ Pilot Results: $PILOT_DIR/runs/pilot/log.jsonl"
echo "    Expected: 180+ records (20 questions × 9 configs)"
echo ""
echo "Phase C Outputs:"
echo "  ✓ Validation:    $WORK_DIR/work_tspec17/benchmark/llm_ensemble_validation_report.json"
echo "  ✓ Disagreements: $WORK_DIR/work_tspec17/benchmark/llm_ensemble_disagreement_cases.jsonl"
echo ""
echo "🎯 NEXT STEPS"
echo "============="
echo "1. Run validation checks:"
echo "   bash POST_EXECUTION_VALIDATION.sh"
echo ""
echo "2. Review results:"
echo "   cat $WORK_DIR/work_tspec17/benchmark/llm_ensemble_validation_report.json | python -m json.tool"
echo ""
echo "3. If gate PASSES (κ ≥ 0.60 + consensus ≥ 85%):"
echo "   bash PHASE_D_RUNNER.sh  # Run final benchmark (6-10 hours)"
echo ""
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
