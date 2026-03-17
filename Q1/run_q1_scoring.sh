#!/usr/bin/env bash
# ===========================================================================
# Q1 Scoring – run all baselines for one or more student models.
#
# Each baseline writes:
#   $OUT_ROOT/<student>/<baseline>/scores.json
#
# Usage:
#   bash run_q1_scoring.sh [STUDENT] [BASELINE]
#
#   STUDENT   comma-separated list, or "all"   (default: qwen2.5-7b)
#   BASELINE  one of: random token_length rule_quality global_naturalness
#             local_naturalness rsr topk_entropy gnorm llm_quality all
#             (default: all)
#
# Examples:
#   bash run_q1_scoring.sh                              # qwen2.5-7b, all baselines
#   bash run_q1_scoring.sh qwen3-4b                    # qwen3-4b, all baselines
#   bash run_q1_scoring.sh qwen2.5-7b global_naturalness
#   bash run_q1_scoring.sh "qwen2.5-7b,qwen3-4b" rsr
#   bash run_q1_scoring.sh all token_length
# ===========================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Paths  (edit these to match your setup)
# ---------------------------------------------------------------------------
RSR_ROOT=/home/tianruny/LIMO/data/training/rsr/RSR_data
OUT_ROOT=/home/tianruny/LIMO/data/Q1
Q1_DIR=/home/tianruny/LIMO/Q1

# Student model root (each student folder lives under this directory)
STUDENT_MODEL_ROOT=/home/tianruny/LIMO/models/students

# Judge model for llm_quality
JUDGE_MODEL=/home/tianruny/LIMO/models/judges/qwen3-32b

# Common settings
N_PROBLEMS=5000
MAX_LENGTH=32768
SEED=0

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
STUDENT_ARG="${1:-qwen2.5-7b}"
BASELINE_ARG="${2:-all}"

ALL_STUDENTS="qwen2.5-7b qwen3-4b qwen2.5-3b qwen3-14b llama-3.1-8b llama-3.2-3b mistral-7b-v0.3 qwen3-32b"
ALL_BASELINES="random token_length rule_quality global_naturalness local_naturalness rsr topk_entropy gnorm llm_quality"

if [[ "$STUDENT_ARG" == "all" ]]; then
    STUDENTS_LIST="$ALL_STUDENTS"
else
    # comma → space
    STUDENTS_LIST="${STUDENT_ARG//,/ }"
fi

if [[ "$BASELINE_ARG" == "all" ]]; then
    BASELINES_LIST="$ALL_BASELINES"
else
    BASELINES_LIST="$BASELINE_ARG"
fi

echo "============================================================"
echo "Q1 Scoring"
echo "  RSR_ROOT : $RSR_ROOT"
echo "  OUT_ROOT : $OUT_ROOT"
echo "  students : $STUDENTS_LIST"
echo "  baselines: $BASELINES_LIST"
echo "============================================================"

# ---------------------------------------------------------------------------
# Helper: run a baseline for the given student(s)
# ---------------------------------------------------------------------------
run_baseline() {
    local baseline="$1"
    local students_csv="$2"       # comma-separated
    local student_model="$3"      # path to student model (empty for non-GPU)

    echo ""
    echo "------------------------------------------------------------"
    echo "BASELINE: $baseline  |  students: $students_csv"
    echo "------------------------------------------------------------"

    case "$baseline" in

    random)
        python "$Q1_DIR/score_random_baseline.py" \
            --rsr_root   "$RSR_ROOT" \
            --out_root   "$OUT_ROOT" \
            --students   "$students_csv" \
            --seed       "$SEED" \
            --n_problems "$N_PROBLEMS" \
            --no_verify_prompts
        ;;

    token_length)
        python "$Q1_DIR/score_token_length_baseline.py" \
            --rsr_root   "$RSR_ROOT" \
            --out_root   "$OUT_ROOT" \
            --students   "$students_csv" \
            --n_problems "$N_PROBLEMS" \
            --no_verify_prompts
        ;;

    rule_quality)
        python "$Q1_DIR/score_rule_quality_baseline.py" \
            --rsr_root   "$RSR_ROOT" \
            --out_root   "$OUT_ROOT" \
            --students   "$students_csv" \
            --n_problems "$N_PROBLEMS" \
            --no_verify_prompts
        ;;

    global_naturalness)
        python "$Q1_DIR/score_global_naturalness_baseline.py" \
            --rsr_root      "$RSR_ROOT" \
            --out_root      "$OUT_ROOT" \
            --student_model "$student_model" \
            --students      "$students_csv" \
            --batch_size    1 \
            --max_length    "$MAX_LENGTH" \
            --n_problems    "$N_PROBLEMS" \
            --no_verify_prompts
        ;;

    local_naturalness)
        python "$Q1_DIR/score_local_naturalness_baseline.py" \
            --rsr_root      "$RSR_ROOT" \
            --out_root      "$OUT_ROOT" \
            --student_model "$student_model" \
            --students      "$students_csv" \
            --batch_size    8 \
            --local_m       4 \
            --max_length    "$MAX_LENGTH" \
            --n_problems    "$N_PROBLEMS" \
            --no_verify_prompts
        ;;

    rsr)
        python "$Q1_DIR/score_rsr_baseline.py" \
            --rsr_root      "$RSR_ROOT" \
            --out_root      "$OUT_ROOT" \
            --student_model "$student_model" \
            --students      "$students_csv" \
            --batch_size    1 \
            --r_max         100 \
            --max_length    "$MAX_LENGTH" \
            --n_problems    "$N_PROBLEMS" \
            --no_verify_prompts
        ;;

    topk_entropy)
        python "$Q1_DIR/score_topk_entropy_baseline.py" \
            --rsr_root      "$RSR_ROOT" \
            --out_root      "$OUT_ROOT" \
            --student_model "$student_model" \
            --students      "$students_csv" \
            --top_k         50 \
            --batch_size    1 \
            --max_length    "$MAX_LENGTH" \
            --n_problems    "$N_PROBLEMS" \
            --no_verify_prompts
        ;;

    gnorm)
        python "$Q1_DIR/score_gnorm_baseline.py" \
            --rsr_root      "$RSR_ROOT" \
            --out_root      "$OUT_ROOT" \
            --student_model "$student_model" \
            --students      "$students_csv" \
            --proj_dim      4096 \
            --proj_seed     42 \
            --max_length    "$MAX_LENGTH" \
            --n_problems    "$N_PROBLEMS" \
            --no_verify_prompts
        ;;

    llm_quality)
        python "$Q1_DIR/score_llm_quality_baseline.py" \
            --rsr_root      "$RSR_ROOT" \
            --out_root      "$OUT_ROOT" \
            --judge_model   "$JUDGE_MODEL" \
            --students      "$students_csv" \
            --batch_size    32 \
            --max_new_tokens 512 \
            --max_input_len  3584 \
            --n_problems    "$N_PROBLEMS" \
            --no_verify_prompts
        ;;

    *)
        echo "[ERROR] Unknown baseline: $baseline"
        exit 1
        ;;
    esac
}

# ---------------------------------------------------------------------------
# Main loop: one student at a time for GPU baselines
# ---------------------------------------------------------------------------
# Non-GPU baselines (same scores regardless of student model) can take the
# full comma-separated list at once. GPU baselines are run per student so
# each student gets its own model checkpoint.

# Split students into array
IFS=' ' read -ra STUDENTS_ARR <<< "$STUDENTS_LIST"

for baseline in $BASELINES_LIST; do
    case "$baseline" in
    random|token_length|rule_quality|llm_quality)
        # Non-student-specific (or judge-based): run once with all students
        students_csv="$(echo "$STUDENTS_LIST" | tr ' ' ',')"
        run_baseline "$baseline" "$students_csv" ""
        ;;
    *)
        # Student-model-specific: run one student at a time
        for student in "${STUDENTS_ARR[@]}"; do
            student_model="$STUDENT_MODEL_ROOT/$student"
            if [[ ! -d "$student_model" ]]; then
                echo "[WARN] student model not found: $student_model – skipping"
                continue
            fi
            run_baseline "$baseline" "$student" "$student_model"
        done
        ;;
    esac
done

echo ""
echo "============================================================"
echo "All done.  Results under: $OUT_ROOT"
echo "============================================================"
