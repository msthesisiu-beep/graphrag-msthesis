---
name: thesis-ollama-kg-rag
description: 'Reproduce, execute, audit, and evaluate the Master Thesis experimental pipeline comparing standard RAG vs. Knowledge-Graph-augmented RAG on 3GPP specifications and TeleQnA using local Ollama LLMs and MPS GPU encoders.'
argument-hint: '[gate-number | all | pilot | dev-grid | test-main | stats]'
user-invocable: true
disable-model-invocation: false
---

# Thesis KG-RAG Experimental Pipeline

End-to-end execution, evaluation, and statistical reporting pipeline for the Master Thesis:
**"Comparing Standard RAG vs. Knowledge Graph-Augmented RAG on 3GPP Telecommunications Specifications"**

This skill orchestrates a fully local, reproducible experimentation framework running on local Ollama models and Apple Silicon (MPS) GPU embedding/reranking models.

---

## When to Use

Use this skill when asked to:
- Set up or verify the local Ollama and Python execution environment (**Gate 0/1**).
- Ingest and chunk 3GPP Release 17 technical specifications (`23.501`, `23.502`, `24.501`, `38.300`, `38.331`) and TeleQnA standards questions (**Gate 2–3**).
- Build dense (BGE, E5) and lexical (BM25) vector indices (**Gate 3 Indexing**).
- Perform retriever-independent candidate pooling and LLM answerability verification (**Gate 4–5**).
- Run ensemble LLM consensus validation for benchmark question quality assessment (**Gate 4–5 Ensemble**).
- Extract, verify, and build the provenance-tracked 3GPP NetworkX Knowledge Graph (**Gate 7**).
- Execute RAG conditions 0 (Direct), A (Dense), B (Hybrid Reranked), C (Graph-Augmented with token budget matching), D (Joint Reranked), and controls (RAND, UNF) (**Gate 8–9**).
- Compute deterministic metrics (Recall@k, MRR, citations, provenance) and run local RAGAS evaluation with `gemma2:9b` (**Gate 10**).
- Run statistical hypothesis tests (McNemar, Wilcoxon, Holm-Bonferroni) and compile thesis camera-ready tables/figures (**Gate 11**).

---

## Non-Negotiable Thesis Invariants

Every script and execution stage strictly adheres to the following methodology constraints:

1. **Four Independent Model Roles**:
   - **Generator**: `qwen2.5:7b-instruct` (produces answer options, chain-of-thought justification, citations)
   - **Verifier**: `llama3.1:8b` (verifies question answerability and checks KG triple fidelity)
   - **Evaluator**: `gemma2:9b` (unbiased RAGAS judge for faithfulness, relevancy, and context metrics)
   - **KG Extractor**: `mistral-nemo:12b` (extracts structured (s, r, o) triples from normative text)
   *Rule: The generator, verifier, evaluator, and KG extractor must never be the same model.*

2. **Retriever-Independent Pooling**:
   - Candidates for benchmark answerability verification are pooled strictly using **BM25 + E5** (never the experimental BGE retriever), preventing retrieval bias.

3. **Strict Token Budget Matching ($T_q$)**:
   - In matched mode, Graph-Augmented RAG (Config C) is capped strictly at $T_q$ (the token footprint of Config B passages) to guarantee accuracy gains stem from graph structure rather than expanded context windows.

4. **Provenance Integrity**:
   - Every graph triple retains immutable links to its origin `(passage_id, spec, version, clause)`. Dangling or unverifiable facts are purged during Gate 7 verification.

---

## Project Structure & Key Assets

```text
.github/skills/thesis-ollama-kg-rag/
├── SKILL.md                          # This workflow guide
├── assets/
│   └── config.yaml                   # Master thesis configuration template
└── scripts/
    ├── requirements.txt              # Pinned Python package dependencies
    ├── common.py                     # Config hashing, Ollama client, token budgeting, I/O
    ├── prompts.py                    # Structured prompt templates & JSON schemas
    ├── retrieval.py                  # BM25, FAISS (BGE/E5), RRF, Reranker, Graph Linker
    ├── 00_check_env.py               # Gate 0/1: Hardware, Ollama, model role check
    ├── 01_prepare_data.py            # Gate 2-3: TeleQnA & 3GPP spec chunking in-place
    ├── 02_build_indexes.py           # Indexing: BGE, E5, BM25 indices
    ├── 03_verify_answerability.py    # Gate 4-5: BM25+E5 pooling & answerability check
    ├── 04_ensemble_validation.py     # Gate 4-5: 3-model ensemble consensus validation (Fleiss' κ)
    ├── 06_build_kg.py                # Gate 7: Extract, verify, & compile NetworkX graph
    ├── 07_run_configs.py             # Gate 8-9: Execution engine for configs 0/A/B/C/D/controls
    ├── 08_evaluate.py                # Gate 10: Retrieval/citation metrics & RAGAS scoring
    ├── 09_stats.py                   # Gate 11: Statistical hypothesis tests & grid selection
    └── 10_tables.py                  # Gate 11: Thesis camera-ready tables & figure generation
```

---

## Step-by-Step Execution Guide

All commands run from the project root (`Final_thesis/`):

### Stage 1: Environment & Dependency Setup (Gate 0/1)
```bash
# 1. Install Python virtualenv dependencies
pip install -r .github/skills/thesis-ollama-kg-rag/scripts/requirements.txt

# 2. Verify hardware (MPS acceleration), Ollama status, and model roles
python .github/skills/thesis-ollama-kg-rag/scripts/00_check_env.py --pull
```
*Outputs: `<paths.work_dir>/env.json` with system specifications and model SHA-256 digests.*

---

### Stage 2: Data Ingestion & Passage Chunking (Gate 2–3)
```bash
python .github/skills/thesis-ollama-kg-rag/scripts/01_prepare_data.py
```
*Reads `teleqna-*.zip`; TSpec Markdown comes from `paths.tspec_dir` for the configured release, or from `tspec-*.zip` when `paths.tspec_source: zip`. TeleQnA candidates without an explicit matching Release tag are excluded. Final accepted questions are supported only by the configured specification set.*
*Outputs under `paths.work_dir` (default: `work_tspec17`): `passages.jsonl`, `questions_all.jsonl`, and `corpus_manifest.json` with source-file hashes.*

---

### Stage 3: Index Construction
```bash
python .github/skills/thesis-ollama-kg-rag/scripts/02_build_indexes.py
```
*Outputs: `<paths.work_dir>/index/{bge,e5}.faiss`, `bm25.pkl`, and `index_stats.json`.*

---

### Stage 4: Answerability Verification & Quality Gate (Gate 4–5)
```bash
# 1. Pool BM25 + E5 candidates and verify with Llama 3.1
python .github/skills/thesis-ollama-kg-rag/scripts/03_verify_answerability.py

# In case of network interruption or timeouts, safely resume:
python .github/skills/thesis-ollama-kg-rag/scripts/03_verify_answerability.py --retry-errors
```
*Outputs: `<paths.work_dir>/benchmark/verification.jsonl`.*

#### Stage 5: LLM Ensemble Consensus Validation (Gate 4–5)
```bash
# Score all questions using 3-model ensemble (qwen2.5, llama3.1, mistral)
python .github/skills/thesis-ollama-kg-rag/scripts/04_ensemble_validation.py --full

# Generate validation report with inter-rater reliability metrics
# (Results already generated by --full, but can be re-run separately)
python .github/skills/thesis-ollama-kg-rag/scripts/04_ensemble_validation.py --report
```
*Gate Acceptance Criteria: Fleiss' κ ≥ 0.60 (strong inter-rater agreement across 3 LLM judges).*
*Outputs: `llm_ensemble_validation.jsonl` (per-question verdicts), `llm_ensemble_validation_report.json` (metrics), `llm_ensemble_disagreement_cases.jsonl` (analysis of disagreement).*

---

### Stage 6: Knowledge Graph Construction (Gate 7)
The KG construction is divided into three resumable, single-model phases:
```bash
# 1. Extraction with mistral-nemo:12b (scope: candidate passages for dev+test)
python .github/skills/thesis-ollama-kg-rag/scripts/06_build_kg.py extract

# 2. Independent fidelity verification with llama3.1:8b against source passage text
python .github/skills/thesis-ollama-kg-rag/scripts/06_build_kg.py verify

# 3. NetworkX MultiDiGraph compilation with canonical alias resolution
python .github/skills/thesis-ollama-kg-rag/scripts/06_build_kg.py build
```
*Outputs under `<paths.work_dir>/kg`: extraction/verification logs, triples, graph, and KG statistics.*

---

### Stage 7: Running RAG Configurations & Controls (Gate 8–9)
```bash
# 1. Calibrate reranker threshold using 5th percentile of dev reference passages
python .github/skills/thesis-ollama-kg-rag/scripts/07_run_configs.py --calibrate-threshold

# 2. Run determinism verification on N=30 dev questions
python .github/skills/thesis-ollama-kg-rag/scripts/07_run_configs.py --determinism-check 30

# 3. Pilot verification run (quick end-to-end sanity test)
python .github/skills/thesis-ollama-kg-rag/scripts/07_run_configs.py --plan pilot --limit 20

# 4. Dev grid search across all 18 C-variants (triples x hops x budget share)
python .github/skills/thesis-ollama-kg-rag/scripts/07_run_configs.py --plan dev-grid

# 5. Full test-set execution (Configs 0, A, B, C, D, RAND, UNF)
python .github/skills/thesis-ollama-kg-rag/scripts/07_run_configs.py --plan test-main
```
*Logs: `<paths.work_dir>/runs/<split>/log.jsonl` (resumable; one row per question $\times$ condition $\times$ run).*

---

### Stage 8: Evaluation & Scoring (Gate 10)
```bash
# Deterministic metrics (accuracy, recall@k, MRR, citation validity, provenance)
python .github/skills/thesis-ollama-kg-rag/scripts/08_evaluate.py --split test

# Full evaluation with RAGAS metrics (evaluator: gemma2:9b) and triple relevance
python .github/skills/thesis-ollama-kg-rag/scripts/08_evaluate.py --split test --ragas --judge-triples
```
*Outputs: `<paths.work_dir>/runs/test/per_question_metrics.csv`.*

---

### Stage 9: Statistics & Thesis Tables (Gate 11)
```bash
# Select optimal C variant from dev grid
python .github/skills/thesis-ollama-kg-rag/scripts/09_stats.py --split dev --dev-select

# Compute hypothesis tests (McNemar, Wilcoxon, Holm-Bonferroni) on test split
python .github/skills/thesis-ollama-kg-rag/scripts/09_stats.py --split test

# Generate camera-ready LaTeX/Markdown thesis tables
python .github/skills/thesis-ollama-kg-rag/scripts/10_tables.py --split test
```

---

## Troubleshooting & Recovery

| Issue | Root Cause | Resolution |
| :--- | :--- | :--- |
| **Ollama Context Truncation** | Prompt exceeds 8192 tokens | Ensure `num_ctx: 8192` is set in Ollama requests; check `truncated_risk` flag in logs. |
| **Timeout during Generation** | Large model inference on slow CPU | Check `ollama.timeout_s: 900` in `config.yaml`. Ensure MPS acceleration is active in `00_check_env.py`. |
| **Interrupted Extraction/Verification** | Power loss or network error | All scripts use append-only `.jsonl` logging with resumption. Re-run the exact command; completed IDs are skipped automatically. |
| **Out-of-Memory on MPS** | Batch size too large for GPU VRAM | In `02_build_indexes.py` and `retrieval.py`, reduce `batch_size` from 64 to 32 or 16. |

### Resuming the Full Runner

`./run_full_pipeline.sh` preserves `work/` by default. Verification, KG extraction/verification, and configuration runs append per-item results and skip successful items on rerun. Successful runner stages are recorded under `work/checkpoints/pipeline/`; a failed stage stops the runner and does not create its completion marker.

Only start over deliberately, after archiving any results you need:
```bash
CLEAN_WORK=1 ./run_full_pipeline.sh
```

**For thesis validation**: Ensure `04_ensemble_validation.py --full` passes before running test set. The per-question log files and ensemble validation report are the recovery source if a run is interrupted.
