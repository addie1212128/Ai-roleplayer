"""QLoRA supervised fine-tune (SFT) with Unsloth — the roleplay keeper.

Verified against the official Unsloth Mistral-Nemo notebook + TRL v0.22.2 source
(Oct 2026). The non-obvious, run-blocking details this encodes:

  * `from unsloth import ...` runs BEFORE trl/transformers so Unsloth's patches
    apply (they only hit not-yet-imported modules) — hence the in-function imports.
  * SFTConfig takes `max_length=`, NOT `max_seq_length=` (that field was removed
    from SFTConfig in current TRL and raises TypeError).
  * SFTTrainer takes `processing_class=` (tokenizer= was removed in TRL 0.16.0).
  * train_on_responses_only masks everything but assistant turns; the mistral
    markers are "[INST]"/"[/INST]" with NO spaces, or every assistant token is
    masked and the run dies. We decode one batch's labels to prove it worked.

Recommended path: do a cheap v0 plumbing run on llama3.1-8b first
(`--model llama3.1-8b --smoke`), then the real keeper on nemo-instruct.

Usage (inside the training env — see training/README.md for install):
    python training/train_sft.py --data training/data/train.jsonl            # keeper
    python training/train_sft.py --model llama3.1-8b --smoke                  # 60-step smoke test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rp_common as rp  # noqa: E402  (stdlib-only; safe before the heavy imports)


def run(args) -> int:
    # Fail fast on a missing dataset BEFORE the multi-minute 12B load.
    if not Path(args.data).exists():
        print(f"{args.data} not found — run prepare_data.py first "
              "(or pass --data <path>).")
        return 1
    # Unsloth MUST be imported before trl/transformers/peft.
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template, train_on_responses_only
    except ImportError:
        print("Unsloth not installed. See training/README.md → 'Install the training env'.")
        print("  (Linux/WSL)  uv pip install unsloth --torch-backend=auto")
        print("  (Windows)    irm https://unsloth.ai/install.ps1 | iex")
        return 1
    import torch  # noqa: F401  (import after unsloth)
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    preset = rp.MODEL_PRESETS.get(args.model)
    if not preset:
        print(f"Unknown --model '{args.model}'. Choices: {', '.join(rp.MODEL_PRESETS)}")
        return 1
    model_name = args.model_name or preset["model_name"]
    template = args.template or preset["template"]
    print(f"Model: {model_name}   template: {template}   max_seq_len: {args.max_seq_len}")

    # 1) Load 4-bit (QLoRA). dtype=None auto-picks bf16 on Ampere+/Ada, fp16 else.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=True,
    )

    # 2) LoRA adapters on all seven linear projections. lora_dropout MUST be 0 to
    #    keep Unsloth's Triton fast path.
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # 3) Chat template + render each record's messages into a 'text' column.
    tokenizer = get_chat_template(tokenizer, chat_template=template)

    def formatting_prompts_func(examples):
        return {"text": [
            tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
            for c in examples["messages"]
        ]}

    dataset = load_dataset("json", data_files=args.data, split="train")
    if "messages" not in dataset.column_names:
        print(f"{args.data} has no 'messages' column — run prepare_data.py first.")
        return 1
    dataset = dataset.map(formatting_prompts_func, batched=True)
    print(f"Training on {len(dataset)} records.")

    # 4) Trainer. Note max_length (NOT max_seq_length) and processing_class.
    sft_kwargs = dict(
        dataset_text_field="text",
        max_length=args.max_seq_len,
        packing=False,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=5,
        learning_rate=args.lr,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        logging_steps=1,
        seed=3407,
        output_dir=args.out,
        report_to="none",
    )
    if args.smoke:
        sft_kwargs["max_steps"] = args.max_steps   # cheap pipeline check
    else:
        sft_kwargs["num_train_epochs"] = args.epochs

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(**sft_kwargs),
    )

    # 5) Mask everything but assistant turns. Use verified markers when known,
    #    else let unsloth-zoo auto-detect them from the template.
    markers = rp.TEMPLATE_PRESETS.get(template, {})
    if markers.get("instruction_part") and markers.get("response_part"):
        trainer = train_on_responses_only(
            trainer,
            instruction_part=markers["instruction_part"],
            response_part=markers["response_part"],
        )
    else:
        trainer = train_on_responses_only(trainer)

    _verify_masking(trainer, tokenizer)

    # 6) Train + save the LoRA adapter.
    trainer.train()
    adapter_out = args.adapter_out or str(Path(args.out) / f"{args.model}-lora")
    model.save_pretrained(adapter_out)
    tokenizer.save_pretrained(adapter_out)
    print(f"\n✅ LoRA adapter saved -> {adapter_out}")
    print("Next: export to GGUF for Ollama:")
    print(f"  python training/export_gguf.py --adapter {adapter_out} "
          f"--base {preset['merge_base']} --template {template}")
    return 0


def _verify_masking(trainer, tokenizer) -> None:
    """Decode one example's labels; masked (-100) tokens render as spaces. You
    should see ONLY the assistant text. Aborts early if EVERYTHING is masked."""
    try:
        labels = trainer.train_dataset[0]["labels"]
    except Exception:
        print("(Could not read labels for the mask check — continuing.)")
        return
    space = tokenizer(" ", add_special_tokens=False).input_ids[0]
    unmasked = [x for x in labels if x != -100]
    if not unmasked:
        raise SystemExit("ABORT: every label is masked — the response markers do not "
                         "match the template. Check --template and the markers in rp_common.")
    shown = tokenizer.decode([space if x == -100 else x for x in labels])
    print("\n--- mask check (only assistant text should be visible) ---")
    print(shown[:600])
    print("--- end mask check ---\n")


def main() -> int:
    rp.safe_stdout()
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "training" / "outputs"
    ap = argparse.ArgumentParser(description="QLoRA SFT of a roleplay model with Unsloth.")
    ap.add_argument("--model", default=rp.DEFAULT_MODEL,
                    help=f"Preset: {', '.join(rp.MODEL_PRESETS)} (default {rp.DEFAULT_MODEL}).")
    ap.add_argument("--model-name", default=None, help="Override the HF model id for the preset.")
    ap.add_argument("--template", default=None, help="Override the chat template name.")
    ap.add_argument("--data", default=str(repo_root / "training" / "data" / "train.jsonl"))
    ap.add_argument("--out", default=str(out_dir), help="Trainer output_dir (checkpoints).")
    ap.add_argument("--adapter-out", default=None, help="Where to save the final LoRA adapter.")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=16, help="16 (notebook) or 32 (more RP capacity).")
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2, help="per_device_train_batch_size (20GB: up to ~4 @2048).")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1.0, help="1-3 typical; more risks overfitting.")
    ap.add_argument("--smoke", action="store_true", help="Run only --max-steps steps to test the pipeline.")
    ap.add_argument("--max-steps", type=int, default=60)
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
