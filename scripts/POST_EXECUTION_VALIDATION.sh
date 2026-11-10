#!/bin/bash

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                    POST-EXECUTION VALIDATION SUITE                        ║"
echo "║              Checking outputs of all 3 phases + Gate Determination        ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Timestamp: $(date)"
echo ""

# Initialize results
VALIDATION_PASS=true
RESULTS_FILE="/tmp/validation_results_$(date +%s).txt"

{
echo "═══════════════════════════════════════════════════════════════════════════"
echo "PHASE A: KG VERIFICATION VALIDATION"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

if [ -f "work_tspec17/pilot20/kg/graph.pkl" ]; then
    echo "✅ KG Graph File:"
    ls -lh work_tspec17/pilot20/kg/graph.pkl
    echo ""
else
    echo "❌ KG Graph NOT FOUND"
    VALIDATION_PASS=false
    echo ""
fi

if [ -f "work_tspec17/pilot20/kg/kg_stats.json" ]; then
    echo "✅ KG Statistics:"
    python3 -c "import json; d=json.load(open('work_tspec17/pilot20/kg/kg_stats.json')); print(f\"  Nodes: {d.get('num_nodes', 'N/A')}\"); print(f\"  Edges: {d.get('num_edges', 'N/A')}\"); print(f\"  Triples: {d.get('num_triples', 'N/A')}\")" 2>/dev/null || echo "  (Could not parse stats)"
    echo ""
else
    echo "❌ KG Statistics NOT FOUND"
    echo ""
fi

KG_VERIFY_COUNT=$(wc -l < work_tspec17/pilot20/kg/verify_raw.jsonl 2>/dev/null || echo "0")
echo "✅ KG Verifications: $KG_VERIFY_COUNT passages processed"
echo ""

echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "PHASE B: PILOT EXECUTION VALIDATION"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

if [ -f "work_tspec17/pilot20/runs/pilot/log.jsonl" ]; then
    PILOT_LINES=$(wc -l < work_tspec17/pilot20/runs/pilot/log.jsonl)
    echo "✅ Pilot Results File:"
    ls -lh work_tspec17/pilot20/runs/pilot/log.jsonl
    echo "   Records: $PILOT_LINES"
    
    if [ "$PILOT_LINES" -ge 180 ]; then
        echo "   Status: ✅ PASS (180+ records expected)"
    else
        echo "   Status: ⚠️  WARNING ($PILOT_LINES < 180 expected)"
    fi
    echo ""
else
    echo "❌ Pilot Results NOT FOUND"
    VALIDATION_PASS=false
    echo ""
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "PHASE C: LLM ENSEMBLE VALIDATION"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

if [ -f "work_tspec17/benchmark/llm_ensemble_validation_report.json" ]; then
    echo "✅ Validation Report Found:"
    python3 << 'PYEOF'
import json
try:
    with open('work_tspec17/benchmark/llm_ensemble_validation_report.json', 'r') as f:
        report = json.load(f)
    
    print(f"  Fleiss' κ (Inter-rater Reliability): {report.get('fleiss_kappa', 'N/A'):.3f}")
    print(f"  Consensus %: {report.get('consensus_pct', 'N/A'):.1f}%")
    print(f"  Model Bias Max: {report.get('model_bias_max_pct', 'N/A'):.1f}%")
    print(f"  Total Questions Scored: {report.get('total_questions', 'N/A')}")
    print()
    
    gate_pass = report.get('gate_pass', False)
    kappa = report.get('fleiss_kappa', 0)
    consensus = report.get('consensus_pct', 0)
    
    print("  GATE CRITERIA CHECK:")
    print(f"    κ ≥ 0.60: {kappa:.3f} → {'✅ PASS' if kappa >= 0.60 else '❌ FAIL'}")
    print(f"    Consensus ≥ 85%: {consensus:.1f}% → {'✅ PASS' if consensus >= 85 else '❌ FAIL'}")
    print()
    print(f"  FINAL GATE: {'✅ PASS' if gate_pass else '❌ FAIL'}")
    print()
    
except Exception as e:
    print(f"  Error parsing report: {e}")
PYEOF
else
    echo "❌ Validation Report NOT FOUND"
    echo ""
fi

if [ -f "work_tspec17/benchmark/llm_ensemble_disagreement_cases.jsonl" ]; then
    DISAGREE_COUNT=$(wc -l < work_tspec17/benchmark/llm_ensemble_disagreement_cases.jsonl)
    echo "✅ Disagreement Cases: $DISAGREE_COUNT"
    echo ""
else
    echo "⏳ Disagreement Cases: Not yet generated"
    echo ""
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "SUMMARY"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

if [ "$VALIDATION_PASS" = true ]; then
    echo "✅ ALL CRITICAL OUTPUTS PRESENT"
    echo ""
    echo "Ready for next steps:"
    echo "  1. Thesis methodology section integration"
    echo "  2. Phase D: Freeze main test split"
    echo "  3. Full benchmark execution"
    echo "  4. Statistical hypothesis tests"
else
    echo "⚠️  Some outputs missing - check above for details"
fi

} | tee "$RESULTS_FILE"

echo ""
echo "📄 Full validation log saved to: $RESULTS_FILE"
