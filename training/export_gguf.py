"""Merge a LoRA adapter and export it to GGUF + an Ollama model.

Since Ollama v0.34.1 (Sep 2026) removed the ADAPTER instruction AND built-in
GGUF conversion / server-side quantization, this manual llama.cpp pipeline is
now the ONLY reliable path into Ollama:

    merge (bf16)  ->  convert_hf_to_gguf.py  ->  llama-quantize  ->  Modelfile  ->  ollama create

Verified (Oct 2026) gotchas this encodes:
  * Merge in 16-bit: reload the base with load_in_4bit=False BEFORE merging, or a
    QLoRA merge requantizes and degrades. (Default merge device is CPU — a 12B
    bf16 merge needs ~24-28GB; your 96GB RAM handles it comfortably.)
  * Script is convert_hf_to_gguf.py (underscores); binary is llama-quantize;
    build llama.cpp with CMake (the Makefile build was removed).
  * llama-quantize only reads a high-precision GGUF (f16/bf16) with a valid
    general.architecture header — convert first, don't feed it safetensors.
  * A bare `FROM model.gguf` inherits NO template/stops. Mistral-Nemo is
    [INST]/[/INST]+</s> (Tekken), NOT ChatML; do not put a literal <s> in the
    template (BOS is auto-added). Best: copy `ollama show --template mistral-nemo`
    — which this script does automatically when that model is already pulled.

Usage:
    python training/export_gguf.py --adapter training/outputs/nemo-instruct-lora \\
        --base mistralai/Mistral-Nemo-Instruct-2407 --template mistral \\
        --llama-cpp-dir ~/llama.cpp --quants Q4_K_M Q5_K_M --name my-roleplayer
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rp_common as rp  # noqa: E402

# Fallback Ollama templates (used only if we can't inherit one via `ollama show`).
_BUILTIN_TEMPLATES = {
    "mistral": (
        '{{- range $i, $m := .Messages }}\n'
        '{{- if eq $m.Role "user" }}[INST] {{ if and $.System (eq (len (slice $.Messages $i)) 1) }}{{ $.System }}\n\n'
        '{{ end }}{{ $m.Content }}[/INST]\n'
        '{{- else if eq $m.Role "assistant" }} {{ $m.Content }}</s>\n'
        '{{- end }}\n'
        '{{- end }}',
        ["[INST]", "[/INST]", "</s>"],
    ),
    "chatml": (
        '{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}'
        '{{ if .Prompt }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n{{ end }}'
        '<|im_start|>assistant\n{{ .Response }}<|im_end|>\n',
        ["<|im_start|>", "<|im_end|>"],
    ),
}
_BUILTIN_TEMPLATES["qwen-2.5"] = _BUILTIN_TEMPLATES["chatml"]


def _run(cmd: list[str], cwd: str | None = None) -> None:
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def _find_quantize(llama_dir: Path) -> Path:
    """Locate the llama-quantize binary across CMake layouts / OSes."""
    for rel in ("build/bin/llama-quantize", "build/bin/Release/llama-quantize.exe",
                "build/bin/llama-quantize.exe", "llama-quantize"):
        p = llama_dir / rel
        if p.exists():
            return p
    raise SystemExit(f"llama-quantize not found under {llama_dir}. Build llama.cpp first "
                     "(cmake -B build && cmake --build build --config Release -j).")


def merge_adapter(base: str, adapter: str, out: str, device: str) -> None:
    """Reload the base in bf16 and fold the LoRA in (QLoRA-safe)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    print(f"Merging {adapter} into {base} (bf16, device={device}) -> {out}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map=device)
    merged = PeftModel.from_pretrained(base_model, adapter).merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True)
    # Prefer the adapter's tokenizer (it may carry added tokens); fall back to base.
    try:
        AutoTokenizer.from_pretrained(adapter).save_pretrained(out)
    except Exception:
        AutoTokenizer.from_pretrained(base).save_pretrained(out)
    print(f"  merged -> {out}")


def _ollama_template(family: str) -> tuple[str, list[str]] | None:
    """Copy the known-good template from an already-pulled Ollama model."""
    try:
        r = subprocess.run(["ollama", "show", "--template", family],
                           capture_output=True, text=True, check=True)
    except Exception:
        return None
    tmpl = r.stdout.strip("\n")
    return (tmpl, []) if tmpl else None


def write_modelfile(merged_dir: Path, gguf_name: str, template: str, family: str,
                    system: str, num_ctx: int, temperature: float,
                    inherit: bool = True) -> Path:
    # Inheriting `ollama show --template <family>` is right only when training used
    # that family's NATIVE template. For a model trained on a DIFFERENT template
    # (e.g. a Mistral model trained with ChatML), pass --no-inherit so we use the
    # matching built-in instead of the family's (wrong) native template.
    inherited = _ollama_template(family) if inherit else None
    if inherited:
        tmpl, stops = inherited
        print(f"  using the known-good template inherited from `ollama show --template {family}`")
    elif template in _BUILTIN_TEMPLATES:
        tmpl, stops = _BUILTIN_TEMPLATES[template]
        print(f"  using the built-in '{template}' template (pull `{family}` and re-run to "
              f"inherit the exact official one)")
    else:
        tmpl, stops = "", []
        print(f"  ⚠ no template for '{template}'. Pull `{family}` in Ollama so the exact "
              f"template can be inherited, or add TEMPLATE to the Modelfile by hand.")

    lines = [f"FROM ./{gguf_name}"]
    if tmpl:
        lines.append(f'TEMPLATE """{tmpl}"""')
    for s in stops:
        lines.append(f'PARAMETER stop "{s}"')
    lines.append(f"PARAMETER num_ctx {num_ctx}")
    lines.append(f"PARAMETER temperature {temperature}")
    if system:
        lines.append(f'SYSTEM """{system}"""')
    path = merged_dir / "Modelfile"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {path}")
    return path


def run(args) -> int:
    # 1) Merge (unless reusing an existing merged dir). Validate BEFORE touching
    #    Path(args.adapter) so a bare invocation hits the friendly message, not a
    #    TypeError on Path(None).
    if args.merged_dir:
        merged = Path(args.merged_dir)
        print(f"Reusing merged model at {merged}")
    else:
        if not args.adapter or not args.base:
            print("Provide --adapter and --base (or --merged-dir to reuse a merge).")
            return 1
        merged = Path(args.adapter).parent / (Path(args.adapter).name + "-merged")
        merge_adapter(args.base, args.adapter, str(merged), args.device)
    if args.merge_only:
        print(f"\n✅ Merge complete -> {merged}  (use this as --model for train_dpo.py)")
        return 0

    llama_dir = Path(args.llama_cpp_dir).expanduser()
    convert = llama_dir / "convert_hf_to_gguf.py"
    if not convert.exists():
        print(f"{convert} not found. Clone + build llama.cpp (see training/README.md) "
              "and pass its path via --llama-cpp-dir.")
        return 1
    quantize = _find_quantize(llama_dir)

    # 2) Convert the merged HF folder to a high-precision GGUF
    hp = merged / f"model-{args.outtype}.gguf"
    print(f"\nConverting -> {hp}")
    _run([sys.executable, str(convert), str(merged), "--outfile", str(hp),
          "--outtype", args.outtype])

    # 3) Quantize to each requested type
    produced: list[Path] = []
    for q in args.quants:
        outq = merged / f"model-{q}.gguf"
        print(f"\nQuantizing -> {outq}  ({q})")
        _run([str(quantize), str(hp), str(outq), q, str(args.threads)])
        produced.append(outq)

    # 4) Modelfile (built from the FIRST quant) + ollama create.
    # Resolve the Ollama family to inherit a template from. Only auto-resolve from
    # a preset when the requested template is a family's NATIVE format; a
    # cross-family template like chatml (the 24B path, or a Qwen model) must NOT
    # silently inherit mistral-small's native [INST] template. Prefer an explicit
    # --ollama-family; otherwise fall back to the template name (a harmless miss
    # that gracefully drops to the built-in template).
    family = args.ollama_family
    if not family and args.template != "chatml":
        preset = next((p for p in rp.MODEL_PRESETS.values() if p["template"] == args.template), None)
        if preset:
            family = preset["ollama_family"]
    family = family or args.template
    primary = produced[0]
    mf = write_modelfile(merged, primary.name, args.template, family,
                         args.system, args.num_ctx, args.temperature,
                         inherit=not args.no_inherit)

    if args.skip_ollama:
        print(f"\n✅ GGUF(s) ready: {', '.join(p.name for p in produced)}")
        print(f"Create the Ollama model yourself:  ollama create {args.name} -f {mf}")
        return 0
    print(f"\nCreating Ollama model '{args.name}'")
    try:
        _run(["ollama", "create", args.name, "-f", mf.name], cwd=str(merged))
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"  ollama create failed ({e}). Is Ollama installed and running? "
              f"You can run it manually: ollama create {args.name} -f {mf}")
        return 1
    print(f"\n✅ Done. In the app, pick '{args.name}' in ⚙ Settings → model, or test:")
    print(f'   ollama run {args.name} "Set the scene in a rain-soaked tavern."')
    print("Verify the template took:  ollama show --modelfile " + args.name)
    return 0


def main() -> int:
    rp.safe_stdout()
    ap = argparse.ArgumentParser(description="Merge LoRA -> GGUF -> Ollama.")
    ap.add_argument("--adapter", default=None, help="Trained LoRA adapter dir (from train_*.py).")
    ap.add_argument("--base", default=None, help="Base model HF id you fine-tuned.")
    ap.add_argument("--merged-dir", default=None, help="Reuse an existing merged folder (skip merge).")
    ap.add_argument("--merge-only", action="store_true", help="Stop after merging (e.g. for DPO).")
    ap.add_argument("--template", default="mistral", help="Chat template family (mistral/chatml/...).")
    ap.add_argument("--ollama-family", default=None,
                    help="Ollama tag to inherit the template from (default: derived from --template).")
    ap.add_argument("--no-inherit", action="store_true",
                    help="Don't inherit a template via `ollama show`; use the built-in for --template "
                         "(use when the model was trained on a non-native template, e.g. ChatML on Mistral).")
    ap.add_argument("--llama-cpp-dir", default="~/llama.cpp")
    ap.add_argument("--outtype", default="f16", choices=["f16", "bf16", "f32"],
                    help="High-precision intermediate (use bf16 if the source is bf16).")
    ap.add_argument("--quants", nargs="+", default=["Q4_K_M"],
                    help="Quant types, e.g. Q4_K_M Q5_K_M Q8_0. First one is used for the Ollama model.")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--name", default="my-roleplayer", help="Ollama model name.")
    ap.add_argument("--device", default="cpu", help="merge device_map: cpu (safe, uses RAM) or auto.")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--system", default="You are a creative, immersive roleplay partner. "
                    "Stay in character and never break the fourth wall.")
    ap.add_argument("--skip-ollama", action="store_true", help="Produce GGUFs but don't call ollama.")
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
