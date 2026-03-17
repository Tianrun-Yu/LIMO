#!/usr/bin/env bash
# ===========================================================================
# Evaluation runner: generate → extract → score → aggregate.
#
# For each (student, q1_baseline, q2_method) adapter and each benchmark,
# runs the full ACC@5 evaluation pipeline.
#
# Reads:   CKPT_ROOT/<student>/<q1_baseline>/<q2_method>/adapter_final/
#          BENCHMARK_ROOT/<benchmark>.jsonl
# Writes:  EVAL_ROOT/<student>/<q1_baseline>/<q2_method>/<benchmark>/
#            generations.jsonl  answers.jsonl  scores.jsonl  summary.json
#          RESULTS_ROOT/table3.csv  (aggregated at the end)
#
# Usage:
#   bash run_eval.sh [STUDENT] [Q1_BASELINE] [Q2_METHOD] [BENCHMARK]
#   (defaults: qwen2.5-7b, all, all, all)
#
# Examples:
#   bash run_eval.sh
#   bash run_eval.sh qwen2.5-7b random topb_B1 aime24
#   bash run_eval.sh qwen2.5-7b random all all
#   bash run_eval.sh all random topb_B1 math500
# ===========================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Paths  (edit to match your setup)
# ---------------------------------------------------------------------------
CKPT_ROOT=/home/tianruny/LIMO/results/checkpoints
EVAL_ROOT=/home/tianruny/LIMO/results/eval
RESULTS_ROOT=/home/tianruny/LIMO/results
BENCHMARK_ROOT=/home/tianruny/LIMO/data/benchmarks
EVAL_DIR=/home/tianruny/LIMO/eval

TENSOR_PARALLEL_SIZE=1   # set > 1 if using multi-GPU vLLM
MAX_NEW_TOKENS=12288

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
STUDENT_ARG="${1:-qwen2.5-7b}"
Q1_BASELINE_ARG="${2:-all}"
Q2_METHOD_ARG="${3:-all}"
BENCHMARK_ARG="${4:-all}"

ALL_STUDENTS="qwen2.5-7b qwen3-4b qwen2.5-3b qwen3-14b llama-3.1-8b llama-3.2-3b mistral-7b-v0.3 qwen3-32b"
ALL_Q1_BASELINES="random token_length rule_quality global_naturalness local_naturalness rsr topk_entropy gnorm llm_quality"
ALL_Q2_METHODS="topb_B1 topb_B5 topb_B10 \
    kl_tau0.05 kl_tau0.1 kl_tau0.5 kl_tau1 kl_tau2 kl_tau5 \
    chi2_tau0.05 chi2_tau0.1 chi2_tau0.5 chi2_tau1 chi2_tau2 chi2_tau5"
ALL_BENCHMARKS="AIME AMC CPQA MATH-L1 MATH-L2 MATH-L3 MATH-L4 MATH-L5"

[[ "$STUDENT_ARG"     == "all" ]] && STUDENTS="$ALL_STUDENTS"     || STUDENTS="${STUDENT_ARG//,/ }"
[[ "$Q1_BASELINE_ARG" == "all" ]] && Q1_BASELINES="$ALL_Q1_BASELINES" || Q1_BASELINES="$Q1_BASELINE_ARG"
[[ "$Q2_METHOD_ARG"   == "all" ]] && Q2_METHODS="$ALL_Q2_METHODS"     || Q2_METHODS="$Q2_METHOD_ARG"
[[ "$BENCHMARK_ARG"   == "all" ]] && BENCHMARKS="$ALL_BENCHMARKS"     || BENCHMARKS="$BENCHMARK_ARG"

echo "============================================================"
echo "Evaluation"
echo "  CKPT_ROOT  : $CKPT_ROOT"
echo "  EVAL_ROOT  : $EVAL_ROOT"
echo "  students   : $STUDENTS"
echo "  q1_baselines: $Q1_BASELINES"
echo "  q2_methods : $Q2_METHODS"
echo "  benchmarks : $BENCHMARKS"
echo "============================================================"

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for student in $STUDENTS; do
    for q1_baseline in $Q1_BASELINES; do
        for q2_method in $Q2_METHODS; do

            adapter_dir="$CKPT_ROOT/$student/$q1_baseline/$q2_method/adapter_final"
            if [[ ! -d "$adapter_dir" ]]; then
                echo "[SKIP] no adapter: $adapter_dir"
                continue
            fi

            for benchmark in $BENCHMARKS; do
                bm_file="$BENCHMARK_ROOT/${benchmark}.json"
                if [[ ! -f "$bm_file" ]]; then
                    echo "[SKIP] benchmark not found: $bm_file"
                    continue
                fi

                out_dir="$EVAL_ROOT/$student/$q1_baseline/$q2_method/$benchmark"

                # Skip if already fully scored
                if [[ -f "$out_dir/summary.json" ]]; then
                    echo "[SKIP] done: $student/$q1_baseline/$q2_method/$benchmark"
                    continue
                fi

                echo ""
                echo "------------------------------------------------------------"
                echo "EVAL  $student / $q1_baseline / $q2_method  →  $benchmark"
                echo "------------------------------------------------------------"

                # Step 1: Generate (skipped automatically if generations.jsonl exists)
                python "$EVAL_DIR/generate.py" \
                    --model_path             "$adapter_dir" \
                    --benchmark              "$benchmark" \
                    --benchmark_path         "$bm_file" \
                    --output_dir             "$out_dir" \
                    --prompts_dir            "$EVAL_DIR/prompts" \
                    --tensor_parallel_size   "$TENSOR_PARALLEL_SIZE" \
                    --max_new_tokens         "$MAX_NEW_TOKENS"

                # Step 2: Extract answers (skipped if answers.jsonl exists)
                python "$EVAL_DIR/extract_answer.py" \
                    --eval_dir "$out_dir"

                # Step 3: Score (skipped if summary.json exists)
                python "$EVAL_DIR/score.py" \
                    --eval_dir "$out_dir"

            done
        done
    done
done

# ---------------------------------------------------------------------------
# Aggregate all results into table3.csv
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Aggregating results …"
python "$EVAL_DIR/aggregate.py" \
    --eval_root  "$EVAL_ROOT" \
    --output_csv "$RESULTS_ROOT/table3.csv"

echo "============================================================"
echo "All done."
echo "  Eval results : $EVAL_ROOT"
echo "  Summary table: $RESULTS_ROOT/table3.csv"
echo "============================================================"
