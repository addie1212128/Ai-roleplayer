# 🛠️ Training toolkit — build your own roleplay model

Runnable scripts that turn an open base model into a custom roleplay **(and
teaching)** model that drops straight into this app via Ollama. This is the
*doing*; [`../TRAINING.md`](../TRAINING.md) is the *why* (strategy, costs,
sources).

> **Verified Oct 2026.** Every API/command/version here was cross-checked
> against official docs (Unsloth, TRL, llama.cpp, Ollama, Axolotl, Hugging
> Face) — including the nasty 2026 breakers: Ollama dropping `ADAPTER`, TRL
> removing `max_prompt_length`, and the Windows CPU-wheel trap.

---

## ⚡ Your next commands (after one-time setup below)

```bash
# 0) (optional, recommended) turn YOUR played stories into training data
python training/import_my_stories.py

# 1) build the blended dataset (roleplay-dominant + a teaching slice)
python training/prepare_data.py --char-name "Aria"

# 2) PLUMBING run — cheap 8B smoke test to prove the pipeline end-to-end
python training/train_sft.py --model llama3.1-8b --smoke

# 3) the KEEPER — Mistral-Nemo 12B QLoRA on your 20GB card
python training/train_sft.py

# 4) export to GGUF + register with Ollama -> shows up in the app's model dropdown
python training/export_gguf.py \
    --adapter training/outputs/nemo-instruct-lora \
    --base mistralai/Mistral-Nemo-Base-2407 \
    --template mistral --llama-cpp-dir ~/llama.cpp --name my-roleplayer
```

Then open the app, **⚙ Settings → model → `my-roleplayer`**, and play. 🎭

---

## What's here

| File | Does |
|---|---|
| `import_my_stories.py` | 🔑 Your stories → SFT data **+ free DPO pairs** (kept swipe = *chosen*, alternates = *rejected*). Pure stdlib. |
| `prepare_data.py` | Download + format PIPPA / LimaRP / CoSER + a teaching slice (Dolly / rotut / Open-Platypus), keep roleplay dominant, dedup, slop-filter → `data/train.jsonl`. |
| `train_sft.py` | Unsloth QLoRA SFT. Defaults tuned for Nemo-12B on ~20 GB. |
| `train_dpo.py` | Optional DPO/ORPO preference polish (a later stage). |
| `export_gguf.py` | Merge → GGUF → quantize → Ollama Modelfile → `ollama create`. |
| `axolotl_mistral_small_24b_fsdp2.yaml` | End-of-month 2-GPU (20+32 GB) path for the 24B keeper. |
| `rp_common.py` | Shared conventions (templates, system-prompt build, dedup, slop). |

---

## One-time setup

Three separate things, because they have conflicting dependencies. Keep them in
**their own environments** — and separate from the app's `.venv` (that one is
only for running the FastAPI server).

### 1. Ollama (you already have this for the app)
Make sure it's installed and running. The finished model plugs in here.

### 2. Install the training env (Unsloth)
> ⚠️ **The #1 silent failure on Windows:** a bare `pip install torch` installs a
> **CPU-only** wheel and training runs on the CPU with *no error*. Use one of
> these, then confirm `torch.cuda.is_available()` is `True`.

**Windows (native — no WSL required):**
```powershell
# easiest, most robust: the official installer one-liner
irm https://unsloth.ai/install.ps1 | iex
```
or manually, **CUDA torch first**:
```powershell
conda create -n rp-train python=3.12 -y ; conda activate rp-train
pip install torch --index-url https://download.pytorch.org/whl/cu128   # NOT bare `pip install torch`
pip install unsloth
pip install -U "triton-windows<3.7"
```

**Linux / WSL2 (lowest-friction, reproducible):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv rp-train --python 3.12 && source rp-train/bin/activate
uv pip install unsloth --torch-backend=auto
```

Verify the GPU + fast kernels actually loaded:
```bash
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m xformers.info          # want memory_efficient_attention AVAILABLE
python -c "from unsloth import FastLanguageModel; print('unsloth OK')"
```

### 3. Install the data env + log in to Hugging Face
```bash
pip install -r training/requirements-data.txt
huggingface-cli login            # REQUIRED: PIPPA/LimaRP mirrors + Llama-3.1 are gated
```
(You can share the training env for this; the data deps are light.)

### 4. llama.cpp (for GGUF export)
```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
python3 -m pip install -r requirements.txt
cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j   # Makefile build was removed
```
**Windows shortcut:** download the prebuilt `llama-<build>-bin-win-cuda-*.zip`
**and** the matching `cudart-llama-bin-win-cuda-*.zip` from the llama.cpp
releases, and extract **both into the same folder** (the CUDA zip has no runtime
DLLs, so the binaries won't start without cudart). `winget install llama.cpp`
gives the Vulkan build, not CUDA.

---

## The steps in detail

**Import your stories** (`import_my_stories.py`) — reads the app's `data/`
folder, rebuilds each character + persona into a system prompt, and writes
`data/my_stories_sft.jsonl` plus `data/my_stories_dpo.jsonl` (the DPO file fills
up as you *swipe* to alternate replies in the app). A few hundred of your own
exchanges personalize the model more than any public set.

**Prepare data** (`prepare_data.py`) — blends sources roleplay-dominant
(~77 % RP / ~15 % general / ~8 % tutor by default), length-filters PIPPA's short
replies, drops slop and duplicates, and prints the real token balance. Pass
`--char-name "<your character>"` so the teaching data stays in-character. Toggle
any source with `--no-pippa`, `--no-coser`, etc., and cap sizes with
`--max-pippa N`. Your imported stories are folded in automatically.

**Train SFT** (`train_sft.py`) — do the 8B `--smoke` run first (60 steps, proves
the whole pipeline cheaply), then the keeper. It auto-masks everything but
assistant turns and **prints a mask check** before the long run — you should see
only the assistant text. Bump quality with `--lora-r 32 --lora-alpha 32`; if you
OOM, drop `--batch 1` or `--max-seq-len`; watch real usage with `nvidia-smi` on
the first steps (you likely have room for `--batch 4` at 2048 ctx).

**Export** (`export_gguf.py`) — merges in bf16 (on CPU by default; your 96 GB RAM
handles the 12B merge easily), converts to GGUF, quantizes (`--quants Q4_K_M
Q5_K_M Q8_0`), writes a Modelfile, and runs `ollama create`. It even copies the
**exact** Mistral-Nemo template if you've `ollama pull mistral-nemo`'d it. For a
12B expect ~7 GB (Q4_K_M) / ~8.5 GB (Q5_K_M) / ~13 GB (Q8_0).

**Polish (optional, later)** (`train_dpo.py`) — after SFT, teach preferences
from your swipe pairs. First merge your SFT adapter, then DPO on top:
```bash
python training/export_gguf.py --adapter training/outputs/nemo-instruct-lora \
    --base mistralai/Mistral-Nemo-Base-2407 --merge-only          # -> ...-merged
python training/train_dpo.py --model training/outputs/nemo-instruct-lora-merged \
    --data training/data/my_stories_dpo.jsonl
```

**Go bigger, later** (`axolotl_mistral_small_24b_fsdp2.yaml`) — when your 32 GB
card arrives, train the 24B keeper across both GPUs with Axolotl FSDP2 (WSL2
only). See the comments in that file.

---

## Troubleshooting (the real 2026 footguns)

| Symptom | Cause / fix |
|---|---|
| Training is slow / GPU idle | Bare `pip install torch` = CPU wheel. Reinstall torch from the `cuXXX` index; check `torch.cuda.is_available()`. |
| `TypeError: ... max_seq_length` (SFTConfig) | Current TRL renamed it — `train_sft.py` already uses `max_length`. Don't hand-edit it back. |
| `RuntimeError: Dataset scripts are no longer supported … PIPPA.py` | `datasets>=4.0` removed loading scripts — `prepare_data.py` already bypasses it via `data_files`. Don't call `load_dataset("PygmalionAI/PIPPA", "pippa_deduped")`. |
| `401/403` loading PIPPA/LimaRP/Llama | Run `huggingface-cli login` and accept each dataset/model's terms on its HF page. |
| Mask check shows **nothing** (all masked) | Template ↔ marker mismatch. Confirm `--template` matches the base; `rp_common.TEMPLATE_PRESETS` has the right markers. |
| `ollama create` fails on a Modelfile with `ADAPTER` | Ollama v0.34.1 removed `ADAPTER` — you must **merge** first. `export_gguf.py` already does. |
| Ollama output leaks `[/INST]` / `<\|im_end\|>` | Wrong template. Pull the family model (`ollama pull mistral-nemo`) and re-run export so the exact template is inherited. |
| `make` fails building llama.cpp | The Makefile build was removed — use CMake (`cmake -B build && cmake --build build --config Release -j`). |
| Garbage after merge | You merged into a 4-bit base. Reload the base in bf16 (`load_in_4bit=False`) first — `export_gguf.py` already does. |

---

## Appendix — verified version notes
- Let `pip install unsloth` resolve the HF stack; it caps `trl<=1.13`, so don't
  `pip install -U trl` into this env (use a separate venv for standalone TRL 1.15).
- Python **3.11–3.12** for training (avoid 3.13/3.14 on Windows — thin wheels).
  Keep this separate from the app's `.venv`.
- Models: `unsloth/Mistral-Nemo-Instruct-2407` (keeper), `meta-llama/Llama-3.1-8B-Instruct`
  (plumbing, gated), `mistralai/Mistral-Small-24B-Base-2501` (24B, dual-GPU only).
- llama.cpp tracks master (no pin): `convert_hf_to_gguf.py` + `llama-quantize`, CMake build.
- Ollama ≥ v0.34.1: merge + convert + quantize yourself (no server-side conversion/adapters).

Full sources are in [`../TRAINING.md`](../TRAINING.md).
