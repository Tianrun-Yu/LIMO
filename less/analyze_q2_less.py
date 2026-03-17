#!/usr/bin/env python3
"""
Analyze LESS gradient-similarity scores for Q2-selected datasets.

For each Q2 dataset (chi2, kl, topb), look up the LESS score of every
selected sample and compare distributions.

Usage:
  python analyze_q2_less.py \
      --less_dir  /home/tianruny/LIMO/less/scores_maxlen8192 \
      --q2_dirs   /home/tianruny/LIMO/data/Q2/qwen2.5-math-7b/rsr/chi2_tau0.5_B5 \
                  /home/tianruny/LIMO/data/Q2/qwen2.5-math-7b/rsr/kl_tau0.5_B5 \
                  /home/tianruny/LIMO/data/Q2/qwen2.5-math-7b/rsr/topb_B5 \
      --output    /home/tianruny/LIMO/less/q2_analysis
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_less_scores(less_dir: str) -> dict:
    """
    Returns dict: (teacher, file_stem) -> list[float]
    e.g. ("qwen3-8b", "short-sys_5k_gen1") -> [0.03, 0.01, ...]
    """
    lookup = {}
    less_root = Path(less_dir)
    for teacher_dir in less_root.iterdir():
        if not teacher_dir.is_dir():
            continue
        for score_file in teacher_dir.glob("*_scores.json"):
            data = json.load(open(score_file))
            file_stem = score_file.stem.replace("_scores", "")
            key = (teacher_dir.name, file_stem)
            lookup[key] = data["scores"]
    return lookup


def get_less_score(lookup, teacher, run_file):
    """Return the scores list for a (teacher, run_file) pair, or None."""
    file_stem = Path(run_file).stem          # e.g. "short-sys_5k_gen1"
    return lookup.get((teacher, file_stem))


def load_q2_less_scores(q2_path: str, lookup: dict):
    """
    For each sample in a Q2 train.jsonl, look up its LESS score.
    Returns (found_scores, missing_count).
    """
    found, missing = [], 0
    with open(q2_path) as f:
        for line in f:
            ex = json.loads(line)
            meta = ex.get("meta", {})
            teacher   = meta.get("teacher")
            run_file  = meta.get("run_file")
            prob_id   = meta.get("problem_id")
            if teacher is None or run_file is None or prob_id is None:
                missing += 1
                continue
            scores_list = get_less_score(lookup, teacher, run_file)
            if scores_list is None or prob_id >= len(scores_list):
                missing += 1
                continue
            score = scores_list[prob_id]
            if not np.isnan(score):
                found.append(score)
            else:
                missing += 1
    return found, missing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--less_dir", required=True)
    ap.add_argument("--q2_dirs",  nargs="+", required=True)
    ap.add_argument("--output",   required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print("[1/3] Loading LESS scores index...")
    lookup = load_less_scores(args.less_dir)
    teachers_available = set(t for (t, _) in lookup.keys())
    files_available    = set(f for (_, f) in lookup.keys())
    print(f"  teachers with scores : {sorted(teachers_available)}")
    print(f"  files with scores    : {sorted(files_available)}")
    print(f"  total (teacher,file) pairs: {len(lookup)}")

    print("\n[2/3] Looking up LESS scores for each Q2 dataset...")
    results = {}   # name -> list[float]

    for q2_dir in args.q2_dirs:
        name      = Path(q2_dir).name
        jsonl     = Path(q2_dir) / "train.jsonl"
        if not jsonl.exists():
            print(f"  [SKIP] {name}: train.jsonl not found")
            continue

        scores, missing = load_q2_less_scores(str(jsonl), lookup)
        total = len(scores) + missing
        hit_rate = len(scores) / total * 100 if total > 0 else 0

        print(f"\n  {name}")
        print(f"    samples     : {total}")
        print(f"    found       : {len(scores)}  ({hit_rate:.1f}%)")
        print(f"    missing     : {missing}  (teacher/file not yet scored)")

        if scores:
            arr = np.array(scores)
            print(f"    mean        : {arr.mean():.4f}")
            print(f"    std         : {arr.std():.4f}")
            print(f"    p25/p50/p75 : {np.percentile(arr,25):.4f} / "
                  f"{np.percentile(arr,50):.4f} / {np.percentile(arr,75):.4f}")
            print(f"    top10% mean : {np.mean(sorted(arr,reverse=True)[:max(1,len(arr)//10)]):.4f}")
            results[name] = arr.tolist()

    if not results:
        print("\nNo scores found yet — run less_score.py first.")
        return

    # ------------------------------------------------------------------
    # 3. Plots
    # ------------------------------------------------------------------
    print("\n[3/3] Generating plots...")

    names  = list(results.keys())
    arrays = [np.array(results[n]) for n in names]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    # --- Violin plot ---
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 2.5), 5))
    parts = ax.violinplot(arrays, positions=range(len(names)),
                          showmedians=True, showextrema=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("LESS gradient similarity score")
    ax.set_title("Q1 score distribution (LESS) per Q2 selection method")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    violin_path = Path(args.output) / "violin.png"
    plt.savefig(violin_path, dpi=150)
    plt.close()
    print(f"  Saved violin plot → {violin_path}")

    # --- KDE / histogram overlay ---
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(
        min(a.min() for a in arrays),
        max(a.max() for a in arrays),
        60
    )
    for i, (name, arr) in enumerate(zip(names, arrays)):
        ax.hist(arr, bins=bins, alpha=0.4, color=colors[i % len(colors)],
                label=f"{name} (μ={arr.mean():.3f})", density=True)
    ax.set_xlabel("LESS gradient similarity score")
    ax.set_ylabel("density")
    ax.set_title("LESS score distribution per Q2 method")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    hist_path = Path(args.output) / "histogram.png"
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"  Saved histogram      → {hist_path}")

    # --- Summary JSON ---
    summary = {}
    for name, arr in zip(names, arrays):
        arr = np.array(arr)
        summary[name] = {
            "n"            : len(arr),
            "mean"         : float(arr.mean()),
            "std"          : float(arr.std()),
            "p25"          : float(np.percentile(arr, 25)),
            "p50"          : float(np.percentile(arr, 50)),
            "p75"          : float(np.percentile(arr, 75)),
            "top10pct_mean": float(np.mean(sorted(arr, reverse=True)
                                           [:max(1, len(arr)//10)])),
        }
    summary_path = Path(args.output) / "summary.json"
    json.dump(summary, open(summary_path, "w"), indent=2)
    print(f"  Saved summary JSON   → {summary_path}")

    # Print table
    print("\n" + "="*70)
    print(f"{'Method':<25} {'n':>6} {'mean':>8} {'std':>7} "
          f"{'p50':>8} {'top10%':>8}")
    print("-"*70)
    for name in names:
        s = summary[name]
        print(f"{name:<25} {s['n']:>6} {s['mean']:>8.4f} {s['std']:>7.4f} "
              f"{s['p50']:>8.4f} {s['top10pct_mean']:>8.4f}")


if __name__ == "__main__":
    main()
