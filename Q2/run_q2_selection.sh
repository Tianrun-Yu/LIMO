#!/usr/bin/env bash
# ===========================================================================
# Q2 Selection – generate training sets from Q1 score matrices.
#
# Reads:   $Q1_ROOT/<student>/<q1_baseline>/scores.json
# Writes:  $Q2_ROOT/<student>/<q1_baseline>/<q2_method>/train.jsonl
#
# Usage:
#   bash run_q2_selection.sh [STUDENT] [Q1_BASELINE] [Q2_METHOD]
#
#   STUDENT      comma-separated list, or "all"  (default: qwen2.5-7b)
#   Q1_BASELINE  one of: random token_length rule_quality global_naturalness
#                local_naturalness rsr topk_entropy gnorm llm_quality all
#                (default: all)
#   Q2_METHOD    one of: topb kl chi2 all            (default: all)
#
# Examples:
#   bash run_q2_selection.sh
#   bash run_q2_selection.sh qwen2.5-7b global_naturalness topb
#   bash run_q2_selection.sh qwen3-4b rsr kl
#   bash run_q2_selection.sh "qwen2.5-7b,qwen3-4b" random chi2
#   bash run_q2_selection.sh all token_length all
# ===========================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Paths  (edit to match your setup)
# ---------------------------------------------------------------------------
RSR_ROOT=/home/tianruny/LIMO/data/training/rsr/RSR_data
Q1_ROOT=/home/tianruny/LIMO/data/Q1
Q2_ROOT=/home/tianruny/LIMO/data/Q2
Q2_DIR=/home/tianruny/LIMO/Q2

N_PROBLEMS=5000
SEED=0

# ---------------------------------------------------------------------------
# Top-B settings  (run these B values)
# ---------------------------------------------------------------------------
TOPB_B_VALUES="1 5 10"        # B=1 is argmax; B>1 uses uniform-sample

# ---------------------------------------------------------------------------
# KL temperature grid
# ---------------------------------------------------------------------------
KL_TAUS="0.05 0.1 0.5 1.0 2.0 5.0"
KL_SELECTION="sample"         # "argmax" or "sample"

# ---------------------------------------------------------------------------
# Chi² regularization grid
# ---------------------------------------------------------------------------
CHI2_TAUS="0.05 0.1 0.5 1.0 2.0 5.0"
CHI2_SELECTION="sample"       # "argmax" or "sample"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
STUDENT_ARG="${1:-qwen2.5-7b}"
Q1_BASELINE_ARG="${2:-all}"
Q2_METHOD_ARG="${3:-all}"

ALL_STUDENTS="qwen2.5-7b qwen3-4b qwen2.5-3b qwen3-14b llama-3.1-8b llama-3.2-3b mistral-7b-v0.3 qwen3-32b"
ALL_Q1_BASELINES="random token_length rule_quality global_naturalness local_naturalness rsr topk_entropy gnorm llm_quality"
ALL_Q2_METHODS="topb kl chi2"

[[ "$STUDENT_ARG"     == "all" ]] && STUDENTS_LIST="$ALL_STUDENTS"     || STUDENTS_LIST="${STUDENT_ARG//,/ }"
[[ "$Q1_BASELINE_ARG" == "all" ]] && Q1_BASELINES="$ALL_Q1_BASELINES"  || Q1_BASELINES="$Q1_BASELINE_ARG"
[[ "$Q2_METHOD_ARG"   == "all" ]] && Q2_METHODS="$ALL_Q2_METHODS"      || Q2_METHODS="$Q2_METHOD_ARG"

echo "============================================================"
echo "Q2 Selection"
echo "  Q1_ROOT    : $Q1_ROOT"
echo "  Q2_ROOT    : $Q2_ROOT"
echo "  students   : $STUDENTS_LIST"
echo "  q1_baselines: $Q1_BASELINES"
echo "  q2_methods : $Q2_METHODS"
echo "============================================================"

# ---------------------------------------------------------------------------
# Helper: check that a Q1 scores.json exists before running Q2
# ---------------------------------------------------------------------------
q1_exists() {
    local student="$1"
    local q1_baseline="$2"
    local path="$Q1_ROOT/$student/$q1_baseline/scores.json"
    if [[ ! -f "$path" ]]; then
        echo "[SKIP] Q1 scores not found: $path"
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
students_csv="$(echo "$STUDENTS_LIST" | tr ' ' ',')"

for q1_baseline in $Q1_BASELINES; do
    for q2_method in $Q2_METHODS; do

        echo ""
        echo "------------------------------------------------------------"
        echo "Q1=$q1_baseline  Q2=$q2_method  students=$students_csv"
        echo "------------------------------------------------------------"

        # Check at least one student has Q1 scores before loading the model
        any_found=0
        for student in $STUDENTS_LIST; do
            scores_path="$Q1_ROOT/$student/$q1_baseline/scores.json"
            if [[ -f "$scores_path" ]]; then
                any_found=1
                break
            fi
        done
        if [[ $any_found -eq 0 ]]; then
            echo "[SKIP] no Q1 scores found for baseline=$q1_baseline, skipping."
            continue
        fi

        case "$q2_method" in

        topb)
            for B in $TOPB_B_VALUES; do
                if [[ "$B" -eq 1 ]]; then
                    sel="argmax"
                else
                    sel="sample"
                fi
                echo "  → topb  B=$B  selection=$sel"
                # Run per student (paths are student-specific)
                for student in $STUDENTS_LIST; do
                    scores_path="$Q1_ROOT/$student/$q1_baseline/scores.json"
                    [[ ! -f "$scores_path" ]] && echo "    [SKIP] $student" && continue
                    python "$Q2_DIR/select_topb.py" \
                        --q1_root     "$Q1_ROOT" \
                        --q2_root     "$Q2_ROOT" \
                        --rsr_root    "$RSR_ROOT" \
                        --students    "$student" \
                        --q1_baseline "$q1_baseline" \
                        --B           "$B" \
                        --selection   "$sel" \
                        --seed        "$SEED" \
                        --n_problems  "$N_PROBLEMS"
                done
            done
            ;;

        kl)
            for tau in $KL_TAUS; do
                echo "  → kl  τ=$tau  selection=$KL_SELECTION"
                for student in $STUDENTS_LIST; do
                    scores_path="$Q1_ROOT/$student/$q1_baseline/scores.json"
                    [[ ! -f "$scores_path" ]] && echo "    [SKIP] $student" && continue
                    python "$Q2_DIR/select_kl.py" \
                        --q1_root     "$Q1_ROOT" \
                        --q2_root     "$Q2_ROOT" \
                        --rsr_root    "$RSR_ROOT" \
                        --students    "$student" \
                        --q1_baseline "$q1_baseline" \
                        --tau         "$tau" \
                        --selection   "$KL_SELECTION" \
                        --seed        "$SEED" \
                        --n_problems  "$N_PROBLEMS"
                done
            done
            ;;

        chi2)
            for tau in $CHI2_TAUS; do
                echo "  → chi2  τ=$tau  selection=$CHI2_SELECTION"
                for student in $STUDENTS_LIST; do
                    scores_path="$Q1_ROOT/$student/$q1_baseline/scores.json"
                    [[ ! -f "$scores_path" ]] && echo "    [SKIP] $student" && continue
                    python "$Q2_DIR/select_chi2.py" \
                        --q1_root     "$Q1_ROOT" \
                        --q2_root     "$Q2_ROOT" \
                        --rsr_root    "$RSR_ROOT" \
                        --students    "$student" \
                        --q1_baseline "$q1_baseline" \
                        --tau         "$tau" \
                        --selection   "$CHI2_SELECTION" \
                        --seed        "$SEED" \
                        --n_problems  "$N_PROBLEMS"
                done
            done
            ;;

        *)
            echo "[ERROR] Unknown Q2 method: $q2_method"
            exit 1
            ;;
        esac
    done
done

echo ""
echo "============================================================"
echo "All done.  Results under: $Q2_ROOT"
echo "============================================================"
