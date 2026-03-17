#!/usr/bin/env python3
"""
LESS-style gradient similarity scoring for RSR training data.

For each training sample, computes cosine similarity between its
gradient and the eval (AIME) gradient — following:
  "LESS: Selecting Influential Data for Targeted Instruction Tuning"
  (Xia et al., 2024)

Steps:
  1. Load base model + wrap with LoRA (no training)
  2. Compute eval gradient (AIME, 30 problems, mean over samples)
  3. For each training file, compute per-sample gradient
  4. Project both to low-dim via random projection
  5. cosine_sim(train_grad, eval_grad) = score
  6. Save scores to output_dir/<model>/<file>.json

Usage:
  python less_score.py \
    --model  /path/to/qwen2.5-math-7b \
    --data   /path/to/RSR_data \
    --eval   /path/to/AIME.json \
    --output /path/to/output \
    --proj_dim 4096 \
    --n_lora_layers 8 \
    --lora_r 32 \
    --max_length 2048 \
    --device cuda
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def tokenize_train_sample(messages, tokenizer, max_length):
    """
    Tokenize a training sample (system + user + assistant with full CoT).
    Labels: -100 for system+user tokens, actual ids for assistant tokens.
    """
    full_text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_msgs = messages[:2]
    prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)

    full_ids   = tokenizer(full_text,   truncation=True, max_length=max_length)["input_ids"]
    prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_length)["input_ids"]

    prompt_len = len(prompt_ids)
    labels     = [-100] * prompt_len + full_ids[prompt_len:]
    labels     = labels[:len(full_ids)]

    return torch.tensor(full_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def tokenize_eval_sample(item, tokenizer, max_length,
                          system_prompt="Please reason step by step, and put your final answer within \\boxed{}"):
    """
    Tokenize an eval problem (question + ground-truth answer only).
    Labels: -100 for system+question tokens, actual ids for answer tokens.
    """
    messages = [
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": item["prompt"]},
        {"role": "assistant", "content": f"\\boxed{{{item['answer']}}}"},
    ]
    full_text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_msgs = messages[:2]
    prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)

    full_ids   = tokenizer(full_text,   truncation=True, max_length=max_length)["input_ids"]
    prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_length)["input_ids"]

    prompt_len = len(prompt_ids)
    labels     = [-100] * prompt_len + full_ids[prompt_len:]
    labels     = labels[:len(full_ids)]

    return torch.tensor(full_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


# ---------------------------------------------------------------------------
# Gradient extraction
# ---------------------------------------------------------------------------

def compute_gradient_vector(model, input_ids, labels, device):
    """
    Run one forward+backward pass, collect LoRA parameter gradients,
    flatten and concatenate into a single GPU tensor. Returns torch.Tensor on device.
    """
    model.zero_grad()

    input_ids = input_ids.unsqueeze(0).to(device)
    labels    = labels.unsqueeze(0).to(device)

    with torch.enable_grad():
        out  = model(input_ids=input_ids, labels=labels)
        loss = out.loss
        if torch.isnan(loss) or torch.isinf(loss):
            return None
        loss.backward()

    grads = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grads.append(param.grad.detach().float().view(-1))

    model.zero_grad()

    if not grads:
        return None
    return torch.cat(grads)   # stays on GPU


# ---------------------------------------------------------------------------
# Random projection
# ---------------------------------------------------------------------------

class RandomProjector:
    """
    Project a high-dimensional gradient vector to proj_dim dims on GPU.
    Uses a fixed random seed so train and eval projections are consistent.
    Uses the JL lemma: R has entries ±1/sqrt(proj_dim).
    Processes in row-chunks to avoid materializing the full (in_dim × proj_dim) matrix.
    """
    def __init__(self, in_dim: int, proj_dim: int, seed: int = 42, row_chunk: int = 65536):
        self.in_dim    = in_dim
        self.proj_dim  = proj_dim
        self.seed      = seed
        self.row_chunk = row_chunk   # process this many input dims at a time

    def project(self, vec: torch.Tensor) -> np.ndarray:
        """
        vec: shape (in_dim,) torch.Tensor on any device
        Returns: shape (proj_dim,) numpy float32 array
        """
        device = vec.device
        scale  = 1.0 / (self.proj_dim ** 0.5)
        result = torch.zeros(self.proj_dim, device=device, dtype=torch.float32)
        gen    = torch.Generator(device=device)
        gen.manual_seed(self.seed)

        idx = 0
        while idx < self.in_dim:
            end  = min(idx + self.row_chunk, self.in_dim)
            rows = end - idx
            # Random ±scale matrix block: (rows, proj_dim)
            R = torch.randint(0, 2, (rows, self.proj_dim),
                              generator=gen, device=device, dtype=torch.float32)
            R = R * (2 * scale) - scale   # map {0,1} → {-scale, +scale}
            result += vec[idx:end] @ R    # (rows,) @ (rows, proj_dim) → (proj_dim,)
            idx = end

        return result.cpu().numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="LESS-style gradient similarity scoring")
    ap.add_argument("--model",        required=True, help="Base model path")
    ap.add_argument("--data",         required=True, help="RSR_data root directory")
    ap.add_argument("--eval",         required=True, help="AIME.json path")
    ap.add_argument("--output",       required=True, help="Output directory for scores")
    ap.add_argument("--proj_dim",     type=int, default=4096,  help="Random projection dim")
    ap.add_argument("--n_lora_layers",type=int, default=8,     help="Number of last transformer layers to attach LoRA")
    ap.add_argument("--lora_r",       type=int, default=16,    help="LoRA rank")
    ap.add_argument("--max_length",   type=int, default=8192,  help="Max token length")
    ap.add_argument("--max_eval",     type=int, default=0,     help="Max eval samples (0=all)")
    ap.add_argument("--max_train",    type=int, default=0,     help="Max train samples per file (0=all)")
    ap.add_argument("--device",       default="cuda",          help="Device")
    ap.add_argument("--teachers",     nargs="*", default=None, help="Subset of teacher dirs to process (default: all)")
    ap.add_argument("--gpu_id",       type=int,  default=0,   help="This GPU's index (for splitting work)")
    ap.add_argument("--n_gpus",       type=int,  default=1,   help="Total number of GPUs (splits teacher list evenly)")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load model + LoRA
    # ------------------------------------------------------------------
    print("[1/4] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    # Attach LoRA only to last n_lora_layers layers
    num_layers  = base_model.config.num_hidden_layers
    first_layer = max(0, num_layers - args.n_lora_layers)
    target_mods = [
        f"model.layers.{i}.{proj}"
        for i in range(first_layer, num_layers)
        for proj in ["self_attn.q_proj", "self_attn.v_proj"]
    ]
    lora_cfg = LoraConfig(
        r              = args.lora_r,
        lora_alpha     = args.lora_r * 2,
        target_modules = [m.split(".")[-1] for m in ["q_proj", "v_proj"]],
        layers_to_transform = list(range(first_layer, num_layers)),
        lora_dropout   = 0.0,
        bias           = "none",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    # Count trainable params for random projector
    grad_dim = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] LoRA gradient dim = {grad_dim:,}  →  proj_dim = {args.proj_dim}")

    projector = RandomProjector(in_dim=grad_dim, proj_dim=args.proj_dim)

    # ------------------------------------------------------------------
    # 2. Compute eval (AIME) gradient
    # ------------------------------------------------------------------
    print("\n[2/4] Computing eval (AIME) gradient...")
    eval_problems = json.load(open(args.eval))
    if args.max_eval > 0:
        eval_problems = eval_problems[:args.max_eval]

    eval_grads = []
    for item in tqdm(eval_problems, desc="AIME eval grad"):
        ids, lbl = tokenize_eval_sample(item, tokenizer, args.max_length)
        g = compute_gradient_vector(model, ids, lbl, device)
        if g is not None:
            eval_grads.append(g)

    if not eval_grads:
        raise RuntimeError("No valid eval gradients computed.")

    # Mean gradient over all eval samples, then project
    eval_grad_mean = torch.stack(eval_grads).mean(dim=0)              # (grad_dim,) on GPU
    eval_proj      = projector.project(eval_grad_mean)                # (proj_dim,) numpy
    eval_proj_norm = eval_proj / (np.linalg.norm(eval_proj) + 1e-8)
    print(f"[INFO] Eval gradient computed from {len(eval_grads)} / {len(eval_problems)} problems")

    # ------------------------------------------------------------------
    # 3. Score each training file
    # ------------------------------------------------------------------
    data_root = Path(args.data)
    teacher_dirs = sorted([
        d for d in data_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    if args.teachers:
        teacher_dirs = [d for d in teacher_dirs if d.name in args.teachers]

    # Split teacher list across GPUs
    if args.n_gpus > 1:
        teacher_dirs = teacher_dirs[args.gpu_id::args.n_gpus]

    print(f"\n[3/4] Scoring training data ({len(teacher_dirs)} teachers, gpu_id={args.gpu_id})...")

    all_scores = {}   # {teacher/file: [scores]}

    for teacher_dir in teacher_dirs:
        teacher_name = teacher_dir.name
        json_files   = sorted(teacher_dir.glob("*.json"))
        out_teacher  = Path(args.output) / teacher_name
        out_teacher.mkdir(parents=True, exist_ok=True)

        for jf in json_files:
            out_path = out_teacher / (jf.stem + "_scores.json")
            if out_path.exists():
                print(f"  [skip] {teacher_name}/{jf.name} (already done)")
                continue

            samples = json.load(open(jf))
            if args.max_train > 0:
                samples = samples[:args.max_train]

            scores = []
            for sample in tqdm(samples, desc=f"{teacher_name}/{jf.name}", ncols=80):
                messages = sample["messages"]
                ids, lbl = tokenize_train_sample(messages, tokenizer, args.max_length)
                g = compute_gradient_vector(model, ids, lbl, device)
                if g is None:
                    scores.append(float("nan"))
                    continue
                proj  = projector.project(g.float())           # g is GPU tensor
                norm  = proj / (np.linalg.norm(proj) + 1e-8)
                score = float(np.dot(norm, eval_proj_norm))   # cosine similarity
                scores.append(score)

            # Save
            result = {
                "teacher": teacher_name,
                "file":    jf.name,
                "n":       len(scores),
                "scores":  scores,
                "mean":    float(np.nanmean(scores)),
                "std":     float(np.nanstd(scores)),
                "top10pct_mean": float(np.nanmean(sorted([s for s in scores if not np.isnan(s)],
                                                          reverse=True)[:max(1, len(scores)//10)])),
            }
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

            all_scores[f"{teacher_name}/{jf.name}"] = scores
            print(f"  ✓ {teacher_name}/{jf.name}  mean={result['mean']:.4f}  "
                  f"top10%={result['top10pct_mean']:.4f}")

    # ------------------------------------------------------------------
    # 4. Summary table
    # ------------------------------------------------------------------
    print("\n[4/4] Summary (sorted by mean score):")
    summary_rows = []
    for key, scores in all_scores.items():
        valid = [s for s in scores if not np.isnan(s)]
        if valid:
            summary_rows.append((key, np.mean(valid), np.std(valid),
                                  np.mean(sorted(valid, reverse=True)[:max(1, len(valid)//10)])))

    summary_rows.sort(key=lambda x: x[1], reverse=True)
    print(f"\n{'Dataset':<45} {'Mean':>8} {'Std':>8} {'Top10%':>8}")
    print("-" * 75)
    for name, mean, std, top10 in summary_rows:
        print(f"{name:<45} {mean:>8.4f} {std:>8.4f} {top10:>8.4f}")

    # Save summary
    summary_path = Path(args.output) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump([{"key": r[0], "mean": r[1], "std": r[2], "top10pct": r[3]}
                   for r in summary_rows], f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
