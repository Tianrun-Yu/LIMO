#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1 – Rule-based Quality scorer.

Implements the LIMO-style rule-based heuristic and outputs a full C×N score
matrix. No GPU required.

Indicators per assistant text:
  - len:             total word count
  - self_ver:        frequency of {"check", "verify"}
  - explore:         frequency of {"perhaps", "might"}
  - granularity:     frequency of {"therefore", "since"}

Composite score per (fidx, pid):
  score = 0.30*z(len) + 0.20*z(self_ver) + 0.25*z(explore) + 0.25*z(granularity)

where z-scoring is pool-wise across all C candidates for each problem.

NOTE: z-scoring requires loading all C files before any scores can be written,
so this script always loads all files first (no sharding / no GPU resume needed).

Output:
  out_root/<student>/rule_quality/scores.json

scores.json schema:
  {
    "method":          "rule_quality",
    "weights":         {"len": 0.30, "self_ver": 0.20, "explore": 0.25, "granularity": 0.25},
    "n_problems":      <int>,
    "n_candidates":    <int>,
    "candidate_files": [{"fidx": 0, "teacher": "...", "path": "..."}, ...],
    "scores":          [[float, ...], ...]   # shape [C, N], composite z-score
  }

Usage:
  python score_rule_quality_baseline.py \\
    --rsr_root /home/tianruny/LIMO/data/training/rsr/RSR_data \\
    --out_root /home/tianruny/LIMO/data/Q1 \\
    --students qwen2.5-7b
"""

import os
import json
import glob
import argparse
import hashlib
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def get_sua(messages: List[Dict]) -> Tuple[str, str, str]:
    sys_c = user_c = asst_c = ""
    for m in messages:
        r = m.get("role", "")
        c = m.get("content", "")
        if r == "system":      sys_c = c
        elif r == "user":      user_c = c
        elif r == "assistant": asst_c = c
    return sys_c, user_c, asst_c


def discover_candidate_files(rsr_root: str) -> List[Tuple[str, str]]:
    teacher_dirs = sorted(
        [p for p in glob.glob(os.path.join(rsr_root, "*")) if os.path.isdir(p)]
    )
    if not teacher_dirs:
        raise FileNotFoundError(f"No teacher dirs found under: {rsr_root}")
    cand_files: List[Tuple[str, str]] = []
    for td in teacher_dirs:
        teacher = os.path.basename(td)
        files   = sorted(glob.glob(os.path.join(td, "*.json")))
        if not files:
            raise FileNotFoundError(f"No *.json files under: {td}")
        for fp in files:
            cand_files.append((teacher, fp))
    return cand_files


def load_json_list(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_canonical_prompt_hashes(first_file: str, n_problems: int) -> List[str]:
    data = load_json_list(first_file)
    if len(data) < n_problems:
        raise ValueError(f"{first_file} has {len(data)} items, expected >= {n_problems}")
    hashes = []
    for i in range(n_problems):
        sys_c, user_c, _ = get_sua(data[i]["messages"])
        hashes.append(sha1_text(sys_c + "\n" + user_c))
    return hashes


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def word_stats(text: str) -> Tuple[float, float, float, float]:
    """Return (n_words, freq_selfver, freq_explore, freq_granularity)."""
    words = text.lower().split()
    n = len(words)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0

    def freq(keys):
        return sum(1 for w in words if w in keys) / n

    return (
        float(n),
        freq({"check", "verify"}),
        freq({"perhaps", "might"}),
        freq({"therefore", "since"}),
    )


def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score across the C dimension for one problem (shape [C])."""
    return (x - x.mean()) / (x.std() + eps)


# ---------------------------------------------------------------------------
# Main scoring routine
# ---------------------------------------------------------------------------

def score_rule_quality(
    rsr_root: str,
    out_path: str,
    n_problems: int = 5000,
    verify_prompts: bool = True,
):
    method = "rule_quality"
    w_len, w_sv, w_ex, w_gr = 0.30, 0.20, 0.25, 0.25

    cand_files = discover_candidate_files(rsr_root)
    C = len(cand_files)
    if C < 2:
        raise RuntimeError(f"Too few candidate files discovered: {C}")
    print(f"[INFO] discovered {C} candidate files")

    canonical_hashes = None
    if verify_prompts:
        canonical_hashes = build_canonical_prompt_hashes(cand_files[0][1], n_problems)
        print("[INFO] built canonical prompt hashes from:", cand_files[0][1])

    # PASS 1: raw indicators [C, N]
    lens    = np.zeros((C, n_problems), dtype=np.float32)
    selfver = np.zeros((C, n_problems), dtype=np.float32)
    explore = np.zeros((C, n_problems), dtype=np.float32)
    gran    = np.zeros((C, n_problems), dtype=np.float32)

    for fidx, (teacher, fp) in enumerate(cand_files):
        print(f"[PASS1 LOAD] {fidx:02d} {teacher} :: {os.path.basename(fp)}")
        data = load_json_list(fp)
        if len(data) < n_problems:
            raise ValueError(f"{fp} has {len(data)} items, expected >= {n_problems}")

        for pid in range(n_problems):
            ex = data[pid]
            sys_c, user_c, asst_c = get_sua(ex["messages"])

            if verify_prompts and canonical_hashes is not None:
                h = sha1_text(sys_c + "\n" + user_c)
                if h != canonical_hashes[pid]:
                    raise RuntimeError(
                        f"Prompt mismatch at pid={pid} for file={fp}\n"
                        f"Expected hash={canonical_hashes[pid]}, got hash={h}"
                    )

            n, sv, ex_, gr = word_stats(asst_c)
            lens[fidx, pid]    = n
            selfver[fidx, pid] = sv
            explore[fidx, pid] = ex_
            gran[fidx, pid]    = gr

        del data

    # PASS 1.5: pool-wise z-score per problem → composite scores [C, N]
    print("[INFO] computing pool-wise z-scores ...")
    composite = np.zeros((C, n_problems), dtype=np.float32)
    for pid in range(n_problems):
        z_len = zscore(lens[:, pid])
        z_sv  = zscore(selfver[:, pid])
        z_ex  = zscore(explore[:, pid])
        z_gr  = zscore(gran[:, pid])
        composite[:, pid] = w_len * z_len + w_sv * z_sv + w_ex * z_ex + w_gr * z_gr

    # Convert to nested Python list [C][N]
    scores = composite.tolist()

    # Save
    ensure_dir(os.path.dirname(out_path))
    out = {
        "method":          method,
        "weights":         {"len": w_len, "self_ver": w_sv, "explore": w_ex, "granularity": w_gr},
        "n_problems":      n_problems,
        "n_candidates":    C,
        "candidate_files": [
            {"fidx": i, "teacher": t, "path": p}
            for i, (t, p) in enumerate(cand_files)
        ],
        "scores": scores,   # shape [C][n_problems], pool-wise z-score composite
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    print(f"[OK] wrote scores for {C} × {n_problems} = {C * n_problems} pairs")
    print(f"[OK] {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Q1 Rule Quality baseline: pool-wise z-score composite of text indicators."
    )
    ap.add_argument("--rsr_root",  type=str, required=True,
                    help="Path to RSR_data root")
    ap.add_argument("--out_root",  type=str, required=True,
                    help="Q1 output root, e.g. /home/tianruny/LIMO/data/Q1")
    ap.add_argument("--students",  type=str, required=True,
                    help="Comma-separated student folder names, e.g. qwen2.5-7b,qwen3-4b")
    ap.add_argument("--n_problems", type=int, default=5000)
    ap.add_argument("--no_verify_prompts", action="store_true",
                    help="Skip cross-file prompt consistency check (faster).")
    args = ap.parse_args()

    students       = [s.strip() for s in args.students.split(",") if s.strip()]
    verify_prompts = not args.no_verify_prompts

    for s in students:
        out_dir  = os.path.join(args.out_root, s, "rule_quality")
        out_path = os.path.join(out_dir, "scores.json")
        print(f"\n=== student: {s} ===")
        score_rule_quality(
            rsr_root=args.rsr_root,
            out_path=out_path,
            n_problems=args.n_problems,
            verify_prompts=verify_prompts,
        )


if __name__ == "__main__":
    main()
