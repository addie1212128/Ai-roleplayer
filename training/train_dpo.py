"""Optional preference polish (DPO / ORPO) with Unsloth — a LATER stage.

Run this AFTER you have a good SFT model and want to sharpen style / reduce
refusals. It teaches "prefer this reply over that one" from {prompt, chosen,
rejected} pairs — ideally the ones import_my_stories.py mined from YOUR swipes
(the kept swipe is `chosen`, each alternate is `rejected`).

Recommended flow (keeps SFT in the loop, avoids a double-PEFT wrap):
    1. Merge your SFT adapter to a 16-bit folder:
         python training/export_gguf.py --adapter <sft-lora> --base <base> --merge-only
    2. DPO-polish on top of that merged model:
         python training/train_dpo.py --model ./merged-nemo --data training/data/my_stories_dpo.jsonl

Verified (Oct 2026) API facts this encodes:
  * DPOConfig uses beta + max_length + truncation_mode. `max_prompt_length` was
    REMOVED from DPOConfig AND ORPOConfig — passing it raises TypeError.
  * DPOTrainer/ORPOTrainer take processing_class=, not tokenizer=.
  * ORPO is reference-free; its import path moved across TRL versions, so we try
    the top-level import first and fall back to trl.experimental.orpo.
  * PatchDPOTrainer() must be called before building the trainer (covers ORPO).
  * We do NOT pass precompute_ref_log_probs (a newer-TRL arg) since the Unsloth
    env pins an older TRL that lacks it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rp_common as rp  # noqa: E402


def _import_orpo():
    """ORPO moved to trl.experimental in newer TRL; support both layouts."""
    try:
        from trl import ORPOConfig, ORPOTrainer  # older TRL (top-level)
        return ORPOConfig, ORPOTrainer
    except ImportError:
        from trl.experimental.orpo import ORPOConfig, ORPOTrainer  # TRL 1.14+
        return ORPOConfig, ORPOTrainer


def run(args) -> int:
    try:
        from unsloth import FastLanguageModel, PatchDPOTrainer
    except ImportError:
        print("Unsloth not installed. See training/README.md → 'Install the training env'.")
        return 1
    PatchDPOTrainer()                       # BEFORE building any DPO/ORPO trainer
    import torch  # noqa: F401
    from datasets import load_dataset
    from unsloth.chat_templates import get_chat_template

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"{data_path} not found.")
        print("Mine pairs from your swipes first:  python training/import_my_stories.py")
        print("…or point --data at a public set, e.g. trl-lib/ultrafeedback_binarized "
              "or jondurbin/gutenberg-dpo-v0.1 (anti-slop).")
        return 1
    dataset = load_dataset("json", data_files=str(data_path), split="train")
    print(f"Preference pairs: {len(dataset)}  ({args.method.upper()} from {data_path})")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
        dtype=None,
    )
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
    )

    # Render preference data with the SAME (lenient) template SFT used, not the
    # raw native one — otherwise conversational prompt/chosen/rejected can trip
    # the strict alternation check. Smoke-test one pair before the long run.
    tokenizer = get_chat_template(tokenizer, chat_template=args.template)
    try:
        ex = dataset[0]
        tokenizer.apply_chat_template(ex["prompt"] + ex["chosen"], tokenize=False)
    except Exception as e:
        print(f"ABORT: a preference pair failed to render with the '{args.template}' "
              f"template ({e}). Check --template and that prompts alternate / end on a user turn.")
        return 1

    common = dict(
        output_dir=args.out,
        beta=args.beta,
        max_length=args.max_seq_len,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        optim="adamw_8bit",
        lr_scheduler_type="linear",
        warmup_ratio=0.03,
        logging_steps=1,
        seed=3407,
        bf16=True,                          # avoid fp16 for preference training
        report_to="none",                   # don't auto-upload to wandb/comet/mlflow if installed
    )

    if args.method == "dpo":
        from trl import DPOConfig, DPOTrainer
        trainer = DPOTrainer(
            model=model,
            ref_model=None,                 # None -> frozen copy of the policy
            args=DPOConfig(truncation_mode="keep_start", **common),
            train_dataset=dataset,
            processing_class=tokenizer,
        )
    else:  # orpo — reference-free (folds SFT + preference into one loss)
        ORPOConfig, ORPOTrainer = _import_orpo()
        trainer = ORPOTrainer(
            model=model,
            args=ORPOConfig(**common),
            train_dataset=dataset,
            processing_class=tokenizer,
        )

    trainer.train()
    adapter_out = args.adapter_out or str(Path(args.out) / f"{args.method}-lora")
    model.save_pretrained(adapter_out)
    tokenizer.save_pretrained(adapter_out)
    print(f"\n✅ {args.method.upper()} adapter saved -> {adapter_out}")
    print("Export it — but --base MUST be the merged SFT model this was trained on")
    print(f"(i.e. --model above, {args.model}), NOT a raw HF base id:")
    print(f"  python training/export_gguf.py --adapter {adapter_out} "
          f"--base {args.model} --template {args.template} --name my-roleplayer")
    return 0


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "training" / "outputs"
    ap = argparse.ArgumentParser(description="DPO/ORPO preference polish with Unsloth.")
    ap.add_argument("--method", choices=["dpo", "orpo"], default="dpo")
    ap.add_argument("--model", default="unsloth/Mistral-Nemo-Instruct-2407",
                    help="Your merged SFT folder (recommended) or an HF base id.")
    ap.add_argument("--data", default=str(repo_root / "training" / "data" / "my_stories_dpo.jsonl"),
                    help="{prompt,chosen,rejected} JSONL.")
    ap.add_argument("--out", default=str(out_dir))
    ap.add_argument("--adapter-out", default=None)
    ap.add_argument("--template", default="mistral",
                    help="Chat template family (match the model you trained: mistral/chatml/...).")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-6, help="Preference LR is much lower than SFT.")
    ap.add_argument("--epochs", type=float, default=1.0)
    args = ap.parse_args()
    rp.safe_stdout()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
