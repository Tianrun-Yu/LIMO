#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q1 – G-Norm (projected gradient norm) scorer.

Computes the random-projected gradient norm for every (candidate, problem)
pair and writes the full C×N score matrix to scores.json.

Energy function (paper Appendix C.1):
  g̃(τ) = R · ∇_θ ℓ(θ; x, τ)    where R ∈ R^{d×|θ|} is a fixed random projection
  score_gnorm(τ) = ‖g̃(τ)‖₂ / sqrt(d)
  (lower = less gradient signal = easier = more natural for the student)

Implementation:
  - One backward pass per trajectory (no cross-trajectory batching possible).
  - Random projection accumulated param-by-param to avoid storing the full
    |θ|-dimensional gradient vector.
  - Model stays in eval() mode; only gradient computation, no weight update.

Output:
  out_root/<student>/gnorm/scores.json

scores.json schema:
  {
    "method":          "gnorm",
    "student_model":   "...",
    "proj_dim":        4096,
    "proj_seed":       42,
    "n_problems":      <int>,
    "n_candidates":    <int>,
    "candidate_files": [{"fidx": 0, "teacher": "...", "path": "..."}, ...],
    "scores":          [[float, ...], ...]   # shape [C, N]
  }

Supports score cache (resume). No sharding (backward passes are serial).

Usage:
  python score_gnorm_baseline.py \\
    --rsr_root      /home/tianruny/LIMO/data/training/rsr/RSR_data \\
    --out_root      /home/tianruny/LIMO/data/Q1 \\
    --student_model /home/tianruny/LIMO/models/students/qwen2.5-7b \\
    --students      qwen2.5-7b \\
    --proj_dim      4096 \\
    --proj_seed     42 \\
    --max_length    32768
"""

import os
import json
import glob
import argparse
import hashlib
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


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


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_student_model(
    model_path: str,
    device: str,
    dtype: torch.dtype,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    print(f"[INFO] loading student model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        trust_remote_code=True,
        device_map=device,
    )
    model.eval()
    print(f"[INFO] model loaded on {device}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Chat template helpers
# ---------------------------------------------------------------------------

def build_prompt_text(tokenizer, sys_c: str, user_c: str) -> str:
    messages = [
        {"role": "system", "content": sys_c},
        {"role": "user",   "content": user_c},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return f"{sys_c}\n\nUser: {user_c}\nAssistant:"


def build_full_text(tokenizer, sys_c: str, user_c: str, asst_c: str) -> str:
    messages = [
        {"role": "system",    "content": sys_c},
        {"role": "user",      "content": user_c},
        {"role": "assistant", "content": asst_c},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        return build_prompt_text(tokenizer, sys_c, user_c) + asst_c


def get_prompt_length(tokenizer, sys_c: str, user_c: str, max_length: int) -> int:
    prompt_text = build_prompt_text(tokenizer, sys_c, user_c)
    ids = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
        return_tensors=None,
    )["input_ids"]
    return len(ids)


# ---------------------------------------------------------------------------
# G-Norm scoring: one trajectory at a time (backward pass required)
# ---------------------------------------------------------------------------

def score_gnorm_single(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sys_c: str,
    user_c: str,
    asst_c: str,
    proj_dim: int,
    proj_seed: int,
    max_length: int,
    device: str,
) -> float:
    prompt_len = get_prompt_length(tokenizer, sys_c, user_c, max_length)

    full_text = build_full_text(tokenizer, sys_c, user_c, asst_c)
    enc = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"].to(device)   # [1, T]
    T = input_ids.shape[1]

    asst_start = min(prompt_len, T - 1)
    n_asst = T - asst_start
    if n_asst <= 0:
        return 0.0

    model.zero_grad()
    with torch.enable_grad():
        outputs = model(input_ids)
        logits  = outputs.logits[0]            # [T, V]

        pred_start   = max(asst_start - 1, 0)
        shift_logits = logits[pred_start: T - 1]      # [n_asst, V]
        shift_labels = input_ids[0, asst_start: T]    # [n_asst]

        loss = F.cross_entropy(shift_logits.float(), shift_labels, reduction="mean")
        loss.backward()

    # Accumulated projected gradient norm (memory-efficient, no full grad vector)
    rng  = torch.Generator()
    rng.manual_seed(proj_seed)
    proj = torch.zeros(proj_dim, dtype=torch.float32)

    for p in model.parameters():
        n = p.numel()
        R_param = torch.randn(proj_dim, n, generator=rng, dtype=torch.float32)
        if p.grad is not None:
            g = p.grad.detach().float().reshape(-1).cpu()
            proj.add_(R_param.mv(g))
            p.grad = None   # free immediately

    score = (proj / math.sqrt(proj_dim)).norm().item()
    model.zero_grad()
    return score


# ---------------------------------------------------------------------------
# Main scoring routine
# ---------------------------------------------------------------------------

def score_gnorm(
    rsr_root: str,
    out_path: str,
    student_model: str,
    n_problems: int = 5000,
    proj_dim: int = 4096,
    proj_seed: int = 42,
    max_length: int = 32768,
    verify_prompts: bool = True,
    score_cache_path: Optional[str] = None,
):
    method = "gnorm"
    cand_files = discover_candidate_files(rsr_root)
    C = len(cand_files)
    if C < 2:
        raise RuntimeError(f"Too few candidate files discovered: {C}")
    print(f"[INFO] discovered {C} candidate files")
    print(f"[INFO] proj_dim={proj_dim}, proj_seed={proj_seed}")

    canonical_hashes: Optional[List[str]] = None
    if verify_prompts:
        canonical_hashes = build_canonical_prompt_hashes(cand_files[0][1], n_problems)
        print("[INFO] built canonical prompt hashes from:", cand_files[0][1])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model, tokenizer = load_student_model(student_model, device, dtype)

    # ------------------------------------------------------------------
    # Score cache (resume on failure)
    # ------------------------------------------------------------------
    NAN = float("nan")
    score_cache: Dict[str, float] = {}
    if score_cache_path and os.path.isfile(score_cache_path):
        with open(score_cache_path, "r", encoding="utf-8") as f:
            score_cache = json.load(f)
        print(f"[INFO] loaded {len(score_cache)} cached scores from {score_cache_path}")

    scores: List[List[float]] = [[NAN] * n_problems for _ in range(C)]
    for key, val in score_cache.items():
        fi, pi = key.split(":")
        fi, pi = int(fi), int(pi)
        if 0 <= fi < C and 0 <= pi < n_problems:
            scores[fi][pi] = val

    # ------------------------------------------------------------------
    # PASS 1 – score every (fidx, pid), one trajectory at a time
    # ------------------------------------------------------------------
    for fidx, (teacher, fp) in enumerate(cand_files):
        if all(scores[fidx][pid] == scores[fidx][pid] for pid in range(n_problems)):
            print(f"[PASS1 SKIP] {fidx:02d} {teacher} – all scores cached")
            continue

        print(f"[PASS1 LOAD] {fidx:02d} {teacher} :: {os.path.basename(fp)}")
        data = load_json_list(fp)
        if len(data) < n_problems:
            raise ValueError(f"{fp} has {len(data)} items, expected >= {n_problems}")

        for pid in range(n_problems):
            if scores[fidx][pid] == scores[fidx][pid]:   # cached
                continue

            ex = data[pid]
            sys_c, user_c, asst_c = get_sua(ex["messages"])

            if verify_prompts and canonical_hashes is not None:
                h = sha1_text(sys_c + "\n" + user_c)
                if h != canonical_hashes[pid]:
                    raise RuntimeError(
                        f"Prompt mismatch at pid={pid} for file={fp}\n"
                        f"Expected hash={canonical_hashes[pid]}, got hash={h}"
                    )

            sc = score_gnorm_single(
                model, tokenizer,
                sys_c, user_c, asst_c,
                proj_dim=proj_dim,
                proj_seed=proj_seed,
                max_length=max_length,
                device=device,
            )
            scores[fidx][pid] = sc
            score_cache[f"{fidx}:{pid}"] = sc

            if (pid + 1) % 5 == 0:
                pct = (pid + 1) / n_problems * 100
                print(f"  → scored {pid + 1}/{n_problems} ({pct:.1f}%)")

            # Save cache after each trajectory for safe resume
            if score_cache_path:
                ensure_dir(os.path.dirname(score_cache_path))
                with open(score_cache_path, "w", encoding="utf-8") as fw:
                    json.dump(score_cache, fw, ensure_ascii=False)

        del data
        torch.cuda.empty_cache()
        print(f"  → finished {teacher}")

    del model
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Write scores.json with full [C][N] matrix
    # ------------------------------------------------------------------
    ensure_dir(os.path.dirname(out_path))
    out = {
        "method":          method,
        "student_model":   student_model,
        "proj_dim":        proj_dim,
        "proj_seed":       proj_seed,
        "n_problems":      n_problems,
        "n_candidates":    C,
        "candidate_files": [
            {"fidx": i, "teacher": t, "path": p}
            for i, (t, p) in enumerate(cand_files)
        ],
        "scores": scores,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    n_nan = sum(1 for row in scores for v in row if v != v)
    print(f"[OK] wrote {C} × {n_problems} scores ({n_nan} NaN/unscored)")
    print(f"[OK] {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Q1 G-Norm baseline: projected gradient norm per (candidate, problem)."
    )
    ap.add_argument("--rsr_root",      type=str, required=True)
    ap.add_argument("--out_root",      type=str, required=True,
                    help="Q1 output root, e.g. /home/tianruny/LIMO/data/Q1")
    ap.add_argument("--student_model", type=str,
                    default="/home/tianruny/LIMO/models/students/qwen2.5-7b")
    ap.add_argument("--students",      type=str, required=True)
    ap.add_argument("--n_problems",    type=int, default=5000)
    ap.add_argument("--proj_dim",      type=int, default=4096)
    ap.add_argument("--proj_seed",     type=int, default=42)
    ap.add_argument("--max_length",    type=int, default=32768)
    ap.add_argument("--no_verify_prompts", action="store_true")
    ap.add_argument("--score_cache",   type=str, default=None,
                    help="Path for JSON score cache (safe resume). "
                         "Defaults to <out_dir>/score_cache.json.")
    args = ap.parse_args()

    students       = [s.strip() for s in args.students.split(",") if s.strip()]
    verify_prompts = not args.no_verify_prompts

    for s in students:
        out_dir  = os.path.join(args.out_root, s, "gnorm")
        out_path = os.path.join(out_dir, "scores.json")
        cache_path = args.score_cache or os.path.join(out_dir, "score_cache.json")

        print(f"\n=== student: {s} ===")
        score_gnorm(
            rsr_root=args.rsr_root,
            out_path=out_path,
            student_model=args.student_model,
            n_problems=args.n_problems,
            proj_dim=args.proj_dim,
            proj_seed=args.proj_seed,
            max_length=args.max_length,
            verify_prompts=verify_prompts,
            score_cache_path=cache_path,
        )


if __name__ == "__main__":
    main()
