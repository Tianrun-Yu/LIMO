#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full parameter SFT with:
  - Flash Attention 2
  - ZeRO-3 (DeepSpeed)
  - bf16
  - Sequence Packing (greedy bin-packing + position_ids reset)
  - Liger Kernels (fused RMSNorm, SwiGLU, CrossEntropy)
  - Gradient Checkpointing
  - Per-sample train_weight support (from meta.train_weight)

Usage (4 GPUs):
  torchrun --nproc_per_node=4 train_full.py \
    --train_data  /home/tianruny/LIMO/data/Q2/qwen2.5-math-7b/rsr/chi2_B5/train.jsonl \
    --model_path  /home/tianruny/LIMO/models/students/qwen2.5-math-7b \
    --output_dir  /home/tianruny/LIMO/results/checkpoints/qwen2.5-math-7b/rsr/chi2_B5_full

Usage (8 GPUs):
  torchrun --nproc_per_node=8 train_full.py ...
"""

import os
import json
import csv
import random
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)


# ---------------------------------------------------------------------------
# Optional: Liger kernels (fused ops — must apply BEFORE model load)
# ---------------------------------------------------------------------------

def apply_liger(model_type: str = "qwen2"):
    try:
        if model_type == "qwen2":
            from liger_kernel.transformers import apply_liger_kernel_to_qwen2
            apply_liger_kernel_to_qwen2(
                rope=True,
                rms_norm=True,
                swiglu=True,
                fused_linear_cross_entropy=True,
            )
        print("[INFO] Liger kernels applied")
    except ImportError:
        print("[WARN] liger_kernel not installed, skipping")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_train_jsonl(path: str) -> List[dict]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


# ---------------------------------------------------------------------------
# Tokenize helpers
# ---------------------------------------------------------------------------

def tokenize_example(ex: dict, tokenizer, max_length: int):
    """Returns (input_ids, labels, train_weight) as lists/float."""
    messages = ex["messages"]
    train_weight = float(ex.get("meta", {}).get("train_weight", 1.0))

    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    prompt_messages = [m for m in messages if m["role"] != "assistant"]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True)

    full_ids   = tokenizer(full_text,   truncation=True, max_length=max_length)["input_ids"]
    prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_length)["input_ids"]

    prompt_len = len(prompt_ids)
    labels = [-100] * prompt_len + full_ids[prompt_len:]
    labels = labels[:len(full_ids)]

    return full_ids, labels, train_weight


# ---------------------------------------------------------------------------
# Greedy sequence packing
# ---------------------------------------------------------------------------

def pack_sequences(records: List[dict], max_length: int) -> List[dict]:
    """
    Greedy bin-packing: fills bins of max_length tokens.
    Each bin becomes one training example with:
      - input_ids:    concatenated token ids
      - labels:       concatenated labels (-100 for prompt tokens)
      - position_ids: reset to 0 at each sequence boundary
      - token_weights: per-token train_weight (replicated from source example)
    """
    # Sort by length descending for better bin utilisation
    records = sorted(records, key=lambda r: len(r["input_ids"]), reverse=True)

    bins: List[List[dict]] = []
    bin_lengths: List[int] = []

    for rec in records:
        L = len(rec["input_ids"])
        if L > max_length:
            continue  # single over-length example: skip

        placed = False
        for i, bl in enumerate(bin_lengths):
            if bl + L <= max_length:
                bins[i].append(rec)
                bin_lengths[i] += L
                placed = True
                break
        if not placed:
            bins.append([rec])
            bin_lengths.append(L)

    packed = []
    for bin_recs in bins:
        input_ids    = []
        labels       = []
        position_ids = []
        token_weights= []

        for rec in bin_recs:
            seq_len = len(rec["input_ids"])
            input_ids    += rec["input_ids"]
            labels       += rec["labels"]
            position_ids += list(range(seq_len))            # reset per sequence
            token_weights+= [rec["train_weight"]] * seq_len # broadcast weight to tokens

        packed.append({
            "input_ids":     input_ids,
            "labels":        labels,
            "position_ids":  position_ids,
            "token_weights": token_weights,
            "length":        len(input_ids),
        })

    print(f"[INFO] Packing: {len(records)} examples → {len(packed)} packed bins "
          f"(ratio {len(records)/max(1,len(packed)):.1f}x)")
    return packed


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PackedSFTDataset(TorchDataset):
    def __init__(self, examples: List[dict], tokenizer, max_length: int):
        # Tokenize
        raw_records = []
        for ex in examples:
            ids, lbl, w = tokenize_example(ex, tokenizer, max_length)
            raw_records.append({"input_ids": ids, "labels": lbl,
                                 "train_weight": w})

        # Pack
        self.records = pack_sequences(raw_records, max_length)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return {
            "input_ids":     np.array(r["input_ids"],     dtype=np.int64),
            "labels":        np.array(r["labels"],        dtype=np.int64),
            "position_ids":  np.array(r["position_ids"],  dtype=np.int64),
            "token_weights": np.array(r["token_weights"], dtype=np.float32),
            "length":        r["length"],
        }


# ---------------------------------------------------------------------------
# AIME eval dataset (unchanged — no packing needed for eval)
# ---------------------------------------------------------------------------

class AIMEEvalDataset(TorchDataset):
    def __init__(self, aime_path: str, tokenizer, max_length: int = 1024,
                 system_prompt: str = "Please reason step by step, and put your final answer within \\boxed{}"):
        self.records = []
        with open(aime_path) as f:
            problems = json.load(f)

        for item in problems:
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
            labels = [-100] * prompt_len + full_ids[prompt_len:]
            labels = labels[:len(full_ids)]

            self.records.append({
                "input_ids":     np.array(full_ids, dtype=np.int64),
                "labels":        np.array(labels,   dtype=np.int64),
                "position_ids":  np.array(list(range(len(full_ids))), dtype=np.int64),
                "token_weights": np.ones(len(full_ids), dtype=np.float32),
                "length":        len(full_ids),
            })

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


# ---------------------------------------------------------------------------
# Collator: handles variable-length packed sequences
# ---------------------------------------------------------------------------

class PackedCollator:
    def __init__(self, pad_token_id: int = 0, pad_to_multiple_of: int = 64):
        self.pad_token_id    = pad_token_id
        self.pad_to_multiple = pad_to_multiple_of

    def __call__(self, features):
        max_len = max(f["input_ids"].shape[0] for f in features)
        if self.pad_to_multiple:
            max_len = ((max_len + self.pad_to_multiple - 1)
                       // self.pad_to_multiple * self.pad_to_multiple)
        B = len(features)

        inp  = np.full((B, max_len), self.pad_token_id, dtype=np.int64)
        lbl  = np.full((B, max_len), -100,              dtype=np.int64)
        pos  = np.zeros((B, max_len),                   dtype=np.int64)
        mask = np.zeros((B, max_len),                   dtype=np.int64)
        wts  = np.zeros((B, max_len),                   dtype=np.float32)

        for i, f in enumerate(features):
            L = f["input_ids"].shape[0]
            inp[i,  :L] = f["input_ids"]
            lbl[i,  :L] = f["labels"]
            pos[i,  :L] = f["position_ids"]
            mask[i, :L] = 1
            wts[i,  :L] = f["token_weights"]

        return {
            "input_ids":      torch.from_numpy(inp),
            "attention_mask": torch.from_numpy(mask),
            "labels":         torch.from_numpy(lbl),
            "position_ids":   torch.from_numpy(pos),
            "token_weights":  torch.from_numpy(wts),
        }


# ---------------------------------------------------------------------------
# Weighted trainer: per-token weighted cross-entropy
# ---------------------------------------------------------------------------

class WeightedTrainer(Trainer):
    """
    Applies per-token weights to the CE loss.
    For uniform weights (all 1.0) behaviour is identical to standard Trainer.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        token_weights = inputs.pop("token_weights", None)   # (B, T)

        if token_weights is None or (token_weights == 1.0).all():
            outputs = model(**inputs)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        # Manual weighted cross-entropy
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits                      # (B, T, V)

        shift_logits  = logits[..., :-1, :].contiguous()
        shift_labels  = labels[..., 1:].contiguous()
        shift_weights = token_weights[..., 1:].contiguous().to(logits.device)

        B, T, V = shift_logits.shape
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        per_token_loss = loss_fct(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
        ).view(B, T)

        # Apply weights and normalise by number of non-masked tokens
        non_pad = (shift_labels != -100).float()
        weighted = per_token_loss * shift_weights * non_pad
        denom    = (shift_weights * non_pad).sum().clamp(min=1e-8)
        loss     = weighted.sum() / denom

        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Loss plot callback (same as train_lora.py)
# ---------------------------------------------------------------------------

class LossPlotCallback(TrainerCallback):
    def __init__(self, output_dir: str, train_dataset, data_collator, n_init_samples: int = 8):
        self.output_dir    = output_dir
        self.train_dataset = train_dataset
        self.data_collator = data_collator
        self.n_init_samples= n_init_samples
        os.makedirs(output_dir, exist_ok=True)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if int(os.environ.get("LOCAL_RANK", 0)) != 0:
            return
        try:
            indices = random.sample(range(len(self.train_dataset)),
                                    min(self.n_init_samples, len(self.train_dataset)))
            samples = [self.train_dataset[i] for i in indices]
            batch   = self.data_collator(samples)
            batch   = {k: v.to(model.device) for k, v in batch.items()}
            model.eval()
            with torch.no_grad():
                # temporarily remove token_weights for step-0 loss
                batch.pop("token_weights", None)
                loss = model(**batch).loss.item()
            model.train()
            state.log_history.insert(0, {"step": 0, "loss": loss})
            print(f"[INFO] step 0 train loss = {loss:.4f}")
        except Exception as e:
            print(f"[WARN] step-0 train loss failed: {e}")

    def on_evaluate(self, args, state, control, **kwargs):
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            self._save_plot(state)

    def on_log(self, args, state, control, **kwargs):
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            self._save_plot(state)

    def _save_plot(self, state):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            train_steps, train_losses = [], []
            eval_steps,  eval_losses  = [], []
            for entry in state.log_history:
                if "loss"      in entry: train_steps.append(entry["step"]); train_losses.append(entry["loss"])
                if "eval_loss" in entry: eval_steps.append(entry["step"]);  eval_losses.append(entry["eval_loss"])

            if not train_steps:
                return

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(train_steps, train_losses, label="Train Loss", alpha=0.8)
            if eval_steps:
                ax.plot(eval_steps, eval_losses, "r-o", label="AIME Eval Loss", markersize=5, linewidth=2)
            ax.set_xlabel("Steps"); ax.set_ylabel("Loss")
            ax.set_title("Training Loss vs AIME Eval Loss (Full Fine-tune)")
            ax.legend(); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "loss_curve.png"), dpi=150)
            plt.close()

            csv_path = os.path.join(self.output_dir, "loss_log.csv")
            t_dict = dict(zip(train_steps, train_losses))
            e_dict = dict(zip(eval_steps,  eval_losses))
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["step", "train_loss", "eval_loss"])
                for s in sorted(set(train_steps) | set(eval_steps)):
                    w.writerow([s, t_dict.get(s, ""), e_dict.get(s, "")])
        except Exception as e:
            print(f"[WARN] plot/csv save failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Full-parameter SFT with packing, ZeRO-3, Flash Attn, Liger")
    ap.add_argument("--train_data",  required=True)
    ap.add_argument("--model_path",  required=True)
    ap.add_argument("--output_dir",  required=True)
    ap.add_argument("--sft_config",  default=None)
    ap.add_argument("--deepspeed",   default=None, help="Path to ZeRO-3 json. Auto-detected if omitted.")
    ap.add_argument("--lr",          type=float, default=None)
    ap.add_argument("--epochs",      type=int,   default=None)
    ap.add_argument("--batch_size",  type=int,   default=None)
    ap.add_argument("--grad_accum",  type=int,   default=None)
    ap.add_argument("--max_length",  type=int,   default=None)
    ap.add_argument("--aime_data",   type=str,   default=None)
    ap.add_argument("--eval_steps",  type=int,   default=10)
    ap.add_argument("--no_liger",    action="store_true", help="Disable Liger kernels")
    args = ap.parse_args()

    configs_dir = Path(__file__).parent / "configs"

    # -----------------------------------------------------------------------
    # 1. Liger kernels (must be applied BEFORE model load)
    # -----------------------------------------------------------------------
    if not args.no_liger:
        apply_liger("qwen2")

    # -----------------------------------------------------------------------
    # 2. Config
    # -----------------------------------------------------------------------
    sft_cfg = load_yaml(args.sft_config or str(configs_dir / "sft_full.yaml"))
    if args.lr:         sft_cfg["learning_rate"]              = args.lr
    if args.epochs:     sft_cfg["num_train_epochs"]            = args.epochs
    if args.batch_size: sft_cfg["per_device_train_batch_size"] = args.batch_size
    if args.grad_accum: sft_cfg["gradient_accumulation_steps"] = args.grad_accum
    if args.max_length: sft_cfg["max_seq_length"]              = args.max_length

    max_seq_length = sft_cfg.get("max_seq_length", 16384)
    ds_path = args.deepspeed or str(configs_dir / "zero3.json")

    print(f"[INFO] train_data  : {args.train_data}")
    print(f"[INFO] model_path  : {args.model_path}")
    print(f"[INFO] output_dir  : {args.output_dir}")
    print(f"[INFO] max_seq_len : {max_seq_length}")
    print(f"[INFO] deepspeed   : {ds_path}")
    print(f"[INFO] sft_cfg     : {sft_cfg}")

    # -----------------------------------------------------------------------
    # 3. Tokenizer
    # -----------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # -----------------------------------------------------------------------
    # 4. Dataset
    # -----------------------------------------------------------------------
    raw      = load_train_jsonl(args.train_data)
    dataset  = PackedSFTDataset(raw, tokenizer, max_seq_length)
    print(f"[INFO] packed bins  : {len(dataset)}")

    aime_path    = args.aime_data or "/home/tianruny/LIMO/data/benchmarks/AIME.json"
    eval_dataset = None
    if os.path.exists(aime_path):
        eval_dataset = AIMEEvalDataset(aime_path, tokenizer)
        print(f"[INFO] AIME eval set: {len(eval_dataset)} problems")

    # -----------------------------------------------------------------------
    # 5. Model — full parameters, Flash Attention 2, bf16
    #    ZeRO-3: do NOT use device_map, let DeepSpeed shard params across GPUs
    # -----------------------------------------------------------------------
    attn_impl = "flash_attention_2"
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        print("[WARN] flash_attn not installed, falling back to sdpa")
        attn_impl = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_impl,
        # No device_map — ZeRO-3 handles placement
    )
    print(f"[INFO] attn_implementation = {attn_impl}")
    model.enable_input_require_grads()

    # -----------------------------------------------------------------------
    # 6. Training arguments
    # -----------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=sft_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=sft_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=sft_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=sft_cfg.get("learning_rate", 1e-5),
        lr_scheduler_type=sft_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=sft_cfg.get("warmup_ratio", 0.05),
        bf16=True,
        logging_steps=sft_cfg.get("logging_steps", 10),
        save_strategy=sft_cfg.get("save_strategy", "steps"),
        save_steps=sft_cfg.get("save_steps", 100),
        save_total_limit=sft_cfg.get("save_total_limit", 3),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,
        remove_unused_columns=False,
        group_by_length=True,
        length_column_name="length",
        report_to="none",
        deepspeed=ds_path,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        eval_on_start=eval_dataset is not None,
        per_device_eval_batch_size=1,
    )

    # -----------------------------------------------------------------------
    # 7. Collator + Trainer
    # -----------------------------------------------------------------------
    data_collator = PackedCollator(
        pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=64,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[LossPlotCallback(args.output_dir, dataset, data_collator)],
    )

    # Resume from checkpoint if available
    import glob
    ckpt_dirs = sorted(glob.glob(os.path.join(args.output_dir, "checkpoint-*")))
    resume = ckpt_dirs[-1] if ckpt_dirs else None
    if resume:
        print(f"[INFO] resuming from {resume}")
    trainer.train(resume_from_checkpoint=resume)

    # -----------------------------------------------------------------------
    # 8. Save final model
    # -----------------------------------------------------------------------
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[OK] model saved → {final_dir}")


if __name__ == "__main__":
    main()
