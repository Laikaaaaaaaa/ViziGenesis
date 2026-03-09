#!/usr/bin/env python3
"""
ViziGenesis — QLoRA Fine-Tuning Pipeline
=========================================
Fine-tunes an open-source LLM (Mistral-7B / Phi-3 / LLaMA-3) on
financial data using QLoRA (Quantized Low-Rank Adaptation).

QLoRA allows fine-tuning a 7B parameter model on a single RTX 4090 (24GB VRAM)
by quantizing the base model to 4-bit and training only low-rank adapters.

Requirements (install on training machine):
    pip install torch transformers accelerate peft bitsandbytes datasets trl

Run:
    python -m backend.fin_llm.train                        # defaults
    python -m backend.fin_llm.train --base_model mistralai/Mistral-7B-Instruct-v0.3
    python -m backend.fin_llm.train --base_model microsoft/Phi-3-mini-4k-instruct
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

# ──────────────────── Configuration ────────────────────

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG = {
    # Model
    "base_model": "mistralai/Mistral-7B-Instruct-v0.3",
    "model_max_length": 2048,

    # QLoRA
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],

    # Training
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 8,      # effective batch = 4 * 8 = 32
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "fp16": False,
    "bf16": True,               # Use bf16 on Ampere+ GPUs

    # Data
    "train_data": str(ROOT / "data" / "processed" / "instruction_data" / "train.jsonl"),
    "eval_data": str(ROOT / "data" / "processed" / "instruction_data" / "val.jsonl"),

    # Output
    "output_dir": str(ROOT / "models" / "fin_llm"),
    "logging_steps": 10,
    "save_steps": 200,
    "eval_steps": 200,
    "save_total_limit": 3,

    # Data quality
    "min_instruction_chars": 8,
    "min_output_chars": 24,
    "max_output_chars": 12000,
    "dedupe_examples": True,
}


def check_dependencies() -> bool:
    """Check if all required packages are installed."""
    missing = []
    for pkg in ["torch", "transformers", "peft", "bitsandbytes", "datasets", "trl", "accelerate"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    return True


def load_training_data(train_path: str, eval_path: str):
    """Load instruction-tuning data."""
    from datasets import Dataset

    def load_jsonl(path: str):
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return items

    def quality_filter(items):
        cleaned = []
        seen = set()
        dropped_empty = 0
        dropped_short = 0
        dropped_long = 0
        dropped_dupe = 0

        for it in items:
            instruction = str(it.get("instruction", "")).strip()
            input_text = str(it.get("input", "")).strip()
            output = str(it.get("output", "")).strip()

            if not instruction or not output:
                dropped_empty += 1
                continue
            if len(instruction) < DEFAULT_CONFIG["min_instruction_chars"] or len(output) < DEFAULT_CONFIG["min_output_chars"]:
                dropped_short += 1
                continue
            if len(output) > DEFAULT_CONFIG["max_output_chars"]:
                dropped_long += 1
                continue

            normalized = {
                "instruction": instruction,
                "input": input_text,
                "output": output,
            }

            if DEFAULT_CONFIG["dedupe_examples"]:
                sig = (instruction.lower(), input_text.lower(), output.lower())
                if sig in seen:
                    dropped_dupe += 1
                    continue
                seen.add(sig)

            cleaned.append(normalized)

        print(
            "Data quality filter: kept=%d, dropped_empty=%d, dropped_short=%d, dropped_long=%d, dropped_dupe=%d"
            % (len(cleaned), dropped_empty, dropped_short, dropped_long, dropped_dupe)
        )
        return cleaned

    train_items = quality_filter(load_jsonl(train_path))
    eval_items = quality_filter(load_jsonl(eval_path)) if eval_path else []

    if not train_items:
        raise ValueError(f"No training data found at {train_path}")

    print(f"Loaded {len(train_items)} training examples, {len(eval_items)} eval examples")

    train_ds = Dataset.from_list(train_items)
    eval_ds = Dataset.from_list(eval_items) if eval_items else None

    return train_ds, eval_ds


def format_instruction(example: Dict) -> str:
    """Format a single example into the instruction template."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text:
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output}"
        )
    else:
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output}"
        )
    return text


def format_chat(example: Dict, tokenizer) -> str:
    """Format as chat messages for instruct-tuned models."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    user_msg = instruction
    if input_text:
        user_msg += f"\n\n{input_text}"

    messages = [
        {"role": "system", "content": (
            "You are ViziGenesis, an expert AI financial analyst. You have deep knowledge of "
            "macroeconomics, stock markets, technical analysis, corporate fundamentals, and "
            "global finance. Provide detailed, actionable analysis with specific data points, "
            "historical context, and trading implications. Be precise with numbers and cite "
            "specific indicators, tickers, and metrics."
        )},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": output},
    ]

    # Use tokenizer's chat template if available
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        text = format_instruction(example)

    return text


def train(config: Dict) -> None:
    """Main training function."""
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    from trl import SFTTrainer

    print(f"\n{'='*60}")
    print(f"ViziGenesis Financial LLM Training")
    print(f"{'='*60}")
    print(f"Base model: {config['base_model']}")
    print(f"Output: {config['output_dir']}")

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        # TF32 improves throughput on Ampere/Hopper with minimal quality impact.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        print("WARNING: No GPU detected. Training will be extremely slow.")

    # Hardware-aware dtype selection.
    bf16_supported = bool(torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
    if config.get("bf16", True) and not bf16_supported:
        print("bf16 not supported on this GPU/runtime, falling back to fp16")
        config["bf16"] = False
        config["fp16"] = True

    # ── 1. Load tokenizer ──
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model"],
        model_max_length=config["model_max_length"],
        padding_side="right",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 2. Quantization config (4-bit QLoRA) ──
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if config["bf16"] else torch.float16,
        bnb_4bit_use_double_quant=True,  # Nested quantization
    )

    # ── 3. Load base model ──
    print("Loading base model (4-bit quantized)...")
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if config["bf16"] else torch.float16,
    )
    model = prepare_model_for_kbit_training(model)

    # ── 4. LoRA config ──
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── 5. Load data ──
    print("\nLoading training data...")
    train_ds, eval_ds = load_training_data(config["train_data"], config["eval_data"])

    # Format examples
    def format_fn(example):
        return format_chat(example, tokenizer)

    # ── 6. Training arguments ──
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        max_grad_norm=config["max_grad_norm"],
        fp16=config["fp16"],
        bf16=config["bf16"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=config["eval_steps"] if eval_ds else None,
        save_total_limit=config["save_total_limit"],
        load_best_model_at_end=True if eval_ds else False,
        report_to="none",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=2,
        group_by_length=True,
        seed=42,
    )

    # ── 7. Trainer ──
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        formatting_func=format_fn,
        max_seq_length=config["model_max_length"],
        packing=True,  # Pack short sequences together for efficiency
    )

    # ── 8. Train ──
    print(f"\nStarting training...")
    print(f"  Epochs: {config['num_train_epochs']}")
    print(f"  Batch size: {config['per_device_train_batch_size']} × {config['gradient_accumulation_steps']} = {config['per_device_train_batch_size'] * config['gradient_accumulation_steps']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  LoRA rank: {config['lora_r']}, alpha: {config['lora_alpha']}")

    trainer.train()

    # ── 9. Save ──
    print("\nSaving model...")
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Save config
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Model saved to: {final_dir}")
    print(f"To merge LoRA weights and export: python -m backend.fin_llm.merge")


def merge_and_export(config: Dict) -> None:
    """Merge LoRA adapters with base model and export full model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    adapter_dir = Path(config["output_dir"]) / "final"
    merged_dir = Path(config["output_dir"]) / "merged"

    print(f"Loading base model: {config['base_model']}")
    base_model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapters from: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {merged_dir}")
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_dir))

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(merged_dir))

    print(f"Export complete! Full model at: {merged_dir}")


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ViziGenesis Financial LLM Training")
    parser.add_argument("--action", choices=["train", "merge"], default="train")
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.base_model:
        config["base_model"] = args.base_model
    if args.epochs:
        config["num_train_epochs"] = args.epochs
    if args.lr:
        config["learning_rate"] = args.lr
    if args.batch_size:
        config["per_device_train_batch_size"] = args.batch_size
    if args.lora_r:
        config["lora_r"] = args.lora_r
    if args.max_length:
        config["model_max_length"] = args.max_length
    if args.output_dir:
        config["output_dir"] = args.output_dir

    if not check_dependencies():
        sys.exit(1)

    if args.action == "train":
        train(config)
    elif args.action == "merge":
        merge_and_export(config)


if __name__ == "__main__":
    main()
