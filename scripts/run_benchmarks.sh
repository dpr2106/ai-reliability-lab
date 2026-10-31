#!/usr/bin/env bash
# =============================================================================
# AI Reliability Lab - Reproducible Benchmark Runner
# =============================================================================
# Executes retrieval benchmark evaluation and end-to-end RAG evaluation,
# recording experiment results transactionally into the shared SQLite database.
# =============================================================================

set -euo pipefail

CORPUS="${CORPUS:-data/benchmarks/sample_corpus.jsonl}"
DATASET="${DATASET:-data/benchmarks/sample_benchmark.jsonl}"
DB_PATH="${DB_PATH:-${RELIABILITY_LAB_DB:-.reliability-lab/experiments.sqlite3}}"
TOP_K="${TOP_K:-5}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-benchmark-run}"
LLM_PROVIDER="${LLM_PROVIDER:-mock}"
MODEL="${MODEL:-gemini-2.5-flash}"
RUN_RAG_EVALUATION="${RUN_RAG_EVALUATION:-true}"
SKIP_NLI="${SKIP_NLI:-true}"

echo "============================================================"
echo "AI Reliability Lab — Benchmark & Evaluation Suite"
echo "============================================================"
echo "Corpus:           $CORPUS"
echo "Dataset:          $DATASET"
echo "Database:         $DB_PATH"
echo "Top-K:            $TOP_K"
echo "Experiment Name:  $EXPERIMENT_NAME"
echo "LLM Provider:     $LLM_PROVIDER"
echo "Model:            $MODEL"
echo "Run RAG Eval:     $RUN_RAG_EVALUATION"
echo "Skip Neural NLI:  $SKIP_NLI"
echo "============================================================"

# Validate required input paths
if [ ! -f "$CORPUS" ]; then
    echo "ERROR: Corpus file not found at: $CORPUS" >&2
    exit 1
fi

if [ ! -f "$DATASET" ]; then
    echo "ERROR: Benchmark dataset file not found at: $DATASET" >&2
    exit 1
fi

# Ensure target database directory exists
mkdir -p "$(dirname "$DB_PATH")"

# Validate LLM provider and credentials
if [ "$LLM_PROVIDER" = "gemini" ]; then
    if [ -z "${GEMINI_API_KEY:-}" ]; then
        echo "ERROR: GEMINI_API_KEY environment variable is required when LLM_PROVIDER=gemini." >&2
        echo "Set GEMINI_API_KEY in your environment or use LLM_PROVIDER=mock for offline runs." >&2
        exit 1
    fi
    echo "[PROVENANCE] Using live Gemini provider ($MODEL)."
elif [ "$LLM_PROVIDER" = "mock" ]; then
    echo "------------------------------------------------------------"
    echo "[PROVENANCE NOTICE] MOCK LLM PROVIDER IN USE"
    echo "RAG answers and judge verdicts are synthetic."
    echo "Do not represent mock benchmark results as live inference."
    echo "------------------------------------------------------------"
fi

# -----------------------------------------------------------------------------
# Stage 1: Retrieval Benchmark Evaluation
# -----------------------------------------------------------------------------
echo ""
echo ">>> [1/2] Running Retrieval Benchmark Evaluation..."
reliability-lab run-benchmark \
    --corpus "$CORPUS" \
    --dataset "$DATASET" \
    --database "$DB_PATH" \
    --top-k "$TOP_K" \
    --experiment-name "${EXPERIMENT_NAME}-retrieval"

# -----------------------------------------------------------------------------
# Stage 2: End-to-End RAG Evaluation (Generation, Groundedness, Citations, Judge)
# -----------------------------------------------------------------------------
if [ "$RUN_RAG_EVALUATION" = "true" ]; then
    echo ""
    echo ">>> [2/2] Running End-to-End RAG Evaluation..."

    EXTRA_ARGS=()
    if [ "$SKIP_NLI" = "true" ]; then
        EXTRA_ARGS+=("--skip-nli")
    fi

    reliability-lab run-rag-evaluation \
        --corpus "$CORPUS" \
        --dataset "$DATASET" \
        --database "$DB_PATH" \
        --top-k "$TOP_K" \
        --retriever bm25 \
        --llm-provider "$LLM_PROVIDER" \
        --model "$MODEL" \
        --experiment-name "${EXPERIMENT_NAME}-rag" \
        "${EXTRA_ARGS[@]}"
else
    echo ""
    echo ">>> [2/2] Skipping RAG Evaluation (RUN_RAG_EVALUATION=$RUN_RAG_EVALUATION)."
fi

echo ""
echo "============================================================"
echo "Benchmarks completed successfully."
echo "Results saved to database: $DB_PATH"
echo "============================================================"
