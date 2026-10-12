#!/usr/bin/env bash
# build_on_cloud.sh — one-command roleplay-model build on a rented GPU box.
#
# Runs the whole toolkit end to end on a fresh Linux CUDA box (RunPod / Vast.ai /
# Lambda): install -> prepare data -> QLoRA SFT -> export to GGUF. It does NOT
# touch Ollama (that lives on your local PC) — it leaves you a .gguf + Modelfile
# to download and `ollama create` at home. See training/CLOUD.md for the full
# runbook and the download step.
#
# Usage (on the rented box, from the repo root):
#   export HF_TOKEN=hf_xxx          # a token whose account accepted the gated datasets/model
#   bash training/build_on_cloud.sh                 # the Nemo-12B keeper
#   SMOKE=1 MODEL=llama3.1-8b bash training/build_on_cloud.sh    # cheap pipeline test first
#
# Tunables (env vars): MODEL, SMOKE, CHAR_NAME, QUANTS, NAME, LLAMA_DIR.
# If any step fails, fall back to the step-by-step commands in training/CLOUD.md.
set -euo pipefail

MODEL="${MODEL:-nemo-instruct}"          # or llama3.1-8b (cheap), qwen2.5-14b
SMOKE="${SMOKE:-0}"                        # 1 = 60-step pipeline test, no real training
CHAR_NAME="${CHAR_NAME:-the character}"    # name used in the tutor system prompt
QUANTS="${QUANTS:-Q4_K_M}"                 # space-separated, e.g. "Q4_K_M Q5_K_M"
NAME="${NAME:-my-roleplayer}"              # Ollama model name for the generated Modelfile
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
OUT="$REPO/training/outputs"

echo "== preflight =="
nvidia-smi -L || { echo "ERROR: no CUDA GPU detected — rent a GPU box (see training/CLOUD.md)."; exit 1; }
: "${HF_TOKEN:?ERROR: set HF_TOKEN (a Hugging Face token whose account accepted the gated PIPPA/LimaRP datasets and the model). export HF_TOKEN=hf_...}"
export HF_TOKEN; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"   # non-interactive HF auth

echo "== install training env (uv + unsloth) =="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv venv .venv-train --python 3.12
# shellcheck disable=SC1091
source .venv-train/bin/activate
uv pip install unsloth --torch-backend=auto
uv pip install -r training/requirements-data.txt
python -c "import torch; assert torch.cuda.is_available(), 'torch cannot see the GPU'; print('torch', torch.__version__, 'CUDA OK', torch.cuda.get_device_name(0))"

echo "== prepare data (public datasets; your own stories aren't on this box — see CLOUD.md to add them) =="
python training/prepare_data.py --char-name "$CHAR_NAME"

echo "== train (QLoRA SFT): model=$MODEL smoke=$SMOKE =="
if [ "$SMOKE" = "1" ]; then
  python training/train_sft.py --model "$MODEL" --smoke
else
  python training/train_sft.py --model "$MODEL"
fi

echo "== build llama.cpp (for GGUF conversion) =="
if [ ! -x "$LLAMA_DIR/build/bin/llama-quantize" ]; then
  [ -d "$LLAMA_DIR/.git" ] || git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
  # The uv venv ships without pip, so bootstrap it before llama.cpp's requirements.
  ( cd "$LLAMA_DIR" && python3 -m ensurepip --upgrade && python3 -m pip install -r requirements.txt \
      && cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j )
fi

echo "== export to GGUF (no Ollama on this box; writes a Modelfile to download) =="
# Pull the preset's merge-base + template so we merge into the RIGHT weights.
BASE="$(python -c "import sys; sys.path.insert(0,'training'); import rp_common as r; print(r.MODEL_PRESETS['$MODEL']['merge_base'])")"
TMPL="$(python -c "import sys; sys.path.insert(0,'training'); import rp_common as r; print(r.MODEL_PRESETS['$MODEL']['template'])")"
EXTRA=()
[ "$TMPL" = "chatml" ] && EXTRA+=(--no-inherit)
python training/export_gguf.py \
  --adapter "$OUT/$MODEL-lora" --base "$BASE" --template "$TMPL" \
  --llama-cpp-dir "$LLAMA_DIR" --quants $QUANTS --name "$NAME" --skip-ollama "${EXTRA[@]}"

MERGED="$OUT/$MODEL-lora-merged"
echo
echo "========================================================================"
echo "DONE. GGUF(s) + Modelfile are in:  $MERGED"
ls -lh "$MERGED"/*.gguf "$MERGED"/Modelfile 2>/dev/null || true
echo
echo "NEXT, on your LOCAL PC (where Ollama runs):"
echo "  1) Download the .gguf AND the Modelfile from $MERGED"
echo "     (RunPod: runpodctl send / the file browser; Vast/Lambda: scp)."
echo "  2) Put them in the same folder, then:  ollama create $NAME -f Modelfile"
echo "  3) In the app: Settings -> model -> $NAME."
echo "========================================================================"
