#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute cross-entropy loss of a model on arbitrary datasets WITHOUT training.

Usage:
  python compute_loss.py \
    --model  /path/to/model_or_adapter \
    --train  /path/to/data1.jsonl  /path/to/data2.jsonl \
    --eval   /path/to/AIME.json    /path/to/MATH-L1.json \
    --max_samples 200 \
    --max_length  4096 \
    --output_dir  /home/tianruny/LIMO/analysis/results

Train data format:  JSONL with {"messages": [{role, content}, ...], ...}
Eval  data format:  JSON  with [{"prompt": ..., "answer": ...}, ...]
                    (benchmark format, loss computed on answer tokens only)
"""

import os
import json
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_train_jsonl(path, tokenizer, max_length, max_samples=None):
    """
    Train data: messages format.
    Loss is computed on assistant tokens only.
    """
    records = []
    with open(path) as f:
        lines = f.readlines()
    if max_samples and len(lines) > max_samples:
        lines = random.sample(lines, max_samples)

    for line in lines:
        ex = json.loads(line.strip())
        messages = ex["messages"]

        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
        prompt_msgs = [m for m in messages if m["role"] != "assistant"]
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True)

        full_ids   = tokenizer(full_text,   truncation=True, max_length=max_length)["input_ids"]
        prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_length)["input_ids"]

        prompt_len = len(prompt_ids)
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        labels = labels[:len(full_ids)]

        # Skip if no answer tokens
        if all(l == -100 for l in labels):
            continue

        records.append({
            "input_ids":      torch.tensor(full_ids,  dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
            "labels":         torch.tensor(labels,    dtype=torch.long),
        })
    return records


def load_eval_json(path, tokenizer, max_length, max_samples=None):
    """
    Benchmark data: [{prompt, answer, ...}].
    Loss is computed on answer tokens only (\\boxed{answer}).
    """
    records = []
    with open(path) as f:
        problems = json.load(f)
    if max_samples and len(problems) > max_samples:
        problems = random.sample(problems, max_samples)

    for item in problems:
        # Use solution if available (MATH datasets), else just answer
        answer_text = item.get("solution") or f"\\boxed{{{item['answer']}}}"

        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": item["prompt"]},
            {"role": "assistant", "content": answer_text},
        ]
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
        prompt_msgs = messages[:2]
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True)

        full_ids   = tokenizer(full_text,   truncation=True, max_length=max_length)["input_ids"]
        prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_length)["input_ids"]

        prompt_len = len(prompt_ids)
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        labels = labels[:len(full_ids)]

        if all(l == -100 for l in labels):
            continue

        records.append({
            "input_ids":      torch.tensor(full_ids,  dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
            "labels":         torch.tensor(labels,    dtype=torch.long),
        })
    return records


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def compute_loss(model, records, batch_size=1, device="cuda"):
    """Average cross-entropy loss over all records."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    with torch.no_grad():
        for i in range(0, len(records), batch_size):
            batch = records[i: i + batch_size]

            # Pad to same length
            max_len = max(r["input_ids"].shape[0] for r in batch)
            pad_id  = model.config.pad_token_id or 0

            inp  = torch.full((len(batch), max_len), pad_id,  dtype=torch.long)
            mask = torch.zeros((len(batch), max_len),          dtype=torch.long)
            lbl  = torch.full((len(batch), max_len), -100,    dtype=torch.long)

            for j, r in enumerate(batch):
                L = r["input_ids"].shape[0]
                inp[j,  :L] = r["input_ids"]
                mask[j, :L] = r["attention_mask"]
                lbl[j,  :L] = r["labels"]

            inp  = inp.to(device)
            mask = mask.to(device)
            lbl  = lbl.to(device)

            out  = model(input_ids=inp, attention_mask=mask, labels=lbl)
            # out.loss is mean over non-masked tokens
            n_tok = (lbl != -100).sum().item()
            total_loss   += out.loss.item() * n_tok
            total_tokens += n_tok
            n_batches    += 1

            if n_batches % 20 == 0:
                print(f"  [{n_batches}/{(len(records)+batch_size-1)//batch_size}] "
                      f"running loss={total_loss/total_tokens:.4f}", flush=True)

    return total_loss / total_tokens if total_tokens > 0 else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Compute loss on datasets without training.")
    ap.add_argument("--model",       required=True,
                    help="Model or LoRA adapter path.")
    ap.add_argument("--train",       nargs="*", default=[],
                    help="Train JSONL files (messages format).")
    ap.add_argument("--eval",        nargs="*", default=[],
                    help="Eval JSON files (benchmark format).")
    ap.add_argument("--max_train_samples", type=int, default=200,
                    help="Max samples per train dataset (random subset).")
    ap.add_argument("--max_eval_samples",  type=int, default=None,
                    help="Max samples per eval dataset (default: use all).")
    ap.add_argument("--max_length",  type=int, default=4096,
                    help="Max token length.")
    ap.add_argument("--output_dir",  default="/home/tianruny/LIMO/analysis/results",
                    help="Directory to save results.")
    ap.add_argument("--seed",        type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device: {device}")

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    print(f"\n[INFO] loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Auto-detect LoRA adapter
    adapter_cfg = os.path.join(args.model, "adapter_config.json")
    if os.path.isfile(adapter_cfg):
        from peft import PeftModel
        with open(adapter_cfg) as f:
            base_model_path = json.load(f)["base_model_name_or_path"]
        print(f"[INFO] LoRA adapter detected, base model: {base_model_path}")
        base = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=torch.bfloat16,
            device_map={"": device}, trust_remote_code=True)
        model = PeftModel.from_pretrained(base, args.model)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16,
            device_map={"": device}, trust_remote_code=True)

    model.eval()

    # -----------------------------------------------------------------------
    # Compute losses
    # -----------------------------------------------------------------------
    results = {}

    for path in args.train:
        name = Path(path).parent.name + "/" + Path(path).name
        print(f"\n[TRAIN] {name}")
        records = load_train_jsonl(path, tokenizer, args.max_length, args.max_train_samples)
        print(f"  loaded {len(records)} samples")
        loss = compute_loss(model, records, device=device)
        results[f"train::{name}"] = loss
        print(f"  loss = {loss:.4f}")

    for path in args.eval:
        name = Path(path).name
        print(f"\n[EVAL] {name}")
        records = load_eval_json(path, tokenizer, args.max_length, args.max_eval_samples)
        print(f"  loaded {len(records)} samples")
        loss = compute_loss(model, records, device=device)
        results[f"eval::{name}"] = loss
        print(f"  loss = {loss:.4f}")

    # -----------------------------------------------------------------------
    # Print summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"{'Dataset':<45} {'Loss':>8}   {'PPL':>8}")
    print("=" * 60)
    for name, loss in results.items():
        ppl = np.exp(loss) if not np.isnan(loss) else float("nan")
        print(f"{name:<45} {loss:>8.4f}   {ppl:>8.2f}")
    print("=" * 60)

    # Save results
    model_name = Path(args.model).name
    out_path = os.path.join(args.output_dir, f"loss_{model_name}.json")
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "results": results}, f, indent=2)
    print(f"\n[OK] results saved → {out_path}")


if __name__ == "__main__":
    main()
