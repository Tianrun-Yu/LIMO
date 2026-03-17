#!/usr/bin/env bash
# ===========================================================================
# LoRA SFT training runner.
#
# Trains one adapter per (student, q1_baseline, q2_method) combination.
# Reads train.jsonl from Q2_ROOT, saves adapter to CKPT_ROOT.
#
# Usage:
#   bash run_train.sh [STUDENT] [Q1_BASELINE] [Q2_METHOD]
#   (defaults: qwen2.5-7b, all, all)
#
# Examples:
#   bash run_train.sh
#   bash run_train.sh qwen2.5-7b random topb_B1
#   bash run_train.sh qwen2.5-7b random all
#   bash run_train.sh all random topb_B1
# ===========================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Paths  (edit to match your setup)
# ---------------------------------------------------------------------------
Q2_ROOT=/home/tianruny/LIMO/data/Q2
MODEL_ROOT=/home/tianruny/LIMO/models/students
CKPT_ROOT=/home/tianruny/LIMO/results/checkpoints
TRAIN_DIR=/home/tianruny/LIMO/train

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
STUDENT_ARG="${1:-qwen2.5-7b}"
Q1_BASELINE_ARG="${2:-all}"
Q2_METHOD_ARG="${3:-all}"

ALL_STUDENTS="qwen2.5-7b qwen3-4b qwen2.5-3b qwen3-14b llama-3.1-8b llama-3.2-3b mistral-7b-v0.3 qwen3-32b"
ALL_Q1_BASELINES="random token_length rule_quality global_naturalness local_naturalness rsr topk_entropy gnorm llm_quality"
ALL_Q2_METHODS="topb_B1 topb_B5 topb_B10 \
    kl_tau0.05 kl_tau0.1 kl_tau0.5 kl_tau1 kl_tau2 kl_tau5 \
    chi2_tau0.05 chi2_tau0.1 chi2_tau0.5 chi2_tau1 chi2_tau2 chi2_tau5"

[[ "$STUDENT_ARG"     == "all" ]] && STUDENTS="$ALL_STUDENTS"     || STUDENTS="${STUDENT_ARG//,/ }"
[[ "$Q1_BASELINE_ARG" == "all" ]] && Q1_BASELINES="$ALL_Q1_BASELINES" || Q1_BASELINES="$Q1_BASELINE_ARG"
[[ "$Q2_METHOD_ARG"   == "all" ]] && Q2_METHODS="$ALL_Q2_METHODS"     || Q2_METHODS="$Q2_METHOD_ARG"

echo "============================================================"
echo "LoRA SFT Training"
echo "  Q2_ROOT    : $Q2_ROOT"
echo "  MODEL_ROOT : $MODEL_ROOT"
echo "  CKPT_ROOT  : $CKPT_ROOT"
echo "  students   : $STUDENTS"
echo "  q1_baselines: $Q1_BASELINES"
echo "  q2_methods : $Q2_METHODS"
echo "============================================================"

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for student in $STUDENTS; do
    model_path="$MODEL_ROOT/$student"
    if [[ ! -d "$model_path" ]]; then
        echo "[WARN] model not found: $model_path – skipping"
        continue
    fi

    for q1_baseline in $Q1_BASELINES; do
        for q2_method in $Q2_METHODS; do

            train_data="$Q2_ROOT/$student/$q1_baseline/$q2_method/train.jsonl"
            if [[ ! -f "$train_data" ]]; then
                echo "[SKIP] train data not found: $train_data"
                continue
            fi

            out_dir="$CKPT_ROOT/$student/$q1_baseline/$q2_method"
            adapter_final="$out_dir/adapter_final/adapter_config.json"
            if [[ -f "$adapter_final" ]]; then
                echo "[SKIP] already trained: $out_dir"
                continue
            fi

            echo ""
            echo "------------------------------------------------------------"
            echo "TRAIN  $student / $q1_baseline / $q2_method"
            echo "  data : $train_data"
            echo "  out  : $out_dir"
            echo "------------------------------------------------------------"

            python "$TRAIN_DIR/train_lora.py" \
                --train_data  "$train_data" \
                --model_path  "$model_path" \
                --output_dir  "$out_dir" \
                --lora_config "$TRAIN_DIR/configs/lora_default.yaml" \
                --sft_config  "$TRAIN_DIR/configs/sft_default.yaml"

        done
    done
done

echo ""
echo "============================================================"
echo "All training done.  Adapters under: $CKPT_ROOT"
echo "============================================================"
