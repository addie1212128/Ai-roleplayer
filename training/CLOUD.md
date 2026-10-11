# ☁️ Build your model on a rented GPU

The training itself needs a CUDA GPU. Renting one for a few hours is usually the
cheapest + fastest way to build the 12B keeper (and it doesn't tie up your PC).
The finished model comes back as a `.gguf` you load into your **local** Ollama,
so it still runs offline on your own machine afterward.

> **What you need:** a GPU-box provider account (RunPod / Vast.ai / Lambda) and a
> **Hugging Face token** whose account has accepted the gated datasets + model.
> I can't rent the box for you — it needs your billing + HF login.

---

## 1. Accept the gated repos (once, in a browser)
Log in at huggingface.co and click **"Agree / Access"** on each:
- `PygmalionAI/PIPPA`, `agentlans/lemonilia-LimaRP` (roleplay data; Not-For-All-Audiences)
- the model you'll train: `mistralai/Mistral-Nemo-Instruct-2407` (or `meta-llama/Llama-3.1-8B-Instruct` for the smoke run)

Then make a token at **huggingface.co/settings/tokens** (read scope is enough).

## 2. Rent a GPU box
For the **Nemo-12B keeper**, a single **24 GB** card (RTX 4090 / A5000) is plenty
and cheapest (~$0.3–0.8/hr); an **A100 40/80 GB** is faster (~$1–1.8/hr). Pick a
PyTorch/CUDA template (Ubuntu). ~1–3 hours of rental is typical.

## 3. One-command build (on the rented box)
```bash
git clone https://github.com/addie1212128/Ai-roleplayer && cd Ai-roleplayer
export HF_TOKEN=hf_xxxxxxxx                      # your token from step 1

# optional cheap dry run first — proves the whole pipeline in ~minutes:
SMOKE=1 MODEL=llama3.1-8b bash training/build_on_cloud.sh

# the real keeper:
CHAR_NAME="Aria" bash training/build_on_cloud.sh
```
The script installs Unsloth, prepares the data, QLoRA-fine-tunes, builds
llama.cpp, and exports a quantized `.gguf` + an Ollama `Modelfile`. When it
finishes it prints the output folder and the exact download + `ollama create`
steps.

> Tunables: `MODEL=` (nemo-instruct | llama3.1-8b | qwen2.5-14b), `QUANTS="Q4_K_M Q5_K_M"`,
> `NAME=my-roleplayer`, `SMOKE=1`. If a step fails, run the manual steps below so
> you can see exactly where it stopped.

## 4. (Optional) add YOUR stories to the mix
The cloud box doesn't have your local `data/`. To fold in your own played
stories (the highest-value slice):
1. On your PC, zip `data/` (or just `data/stories`, `data/characters`, `data/personas`).
2. Upload it to the box and unzip into the repo's `data/`.
3. Re-run with the importer first:
   ```bash
   python training/import_my_stories.py
   python training/prepare_data.py --char-name "Aria" \
     --my-stories training/data/my_stories_sft.jsonl
   # then train_sft.py / export_gguf.py as the script does
   ```

## 5. Bring the model home
On the **rented box** the script leaves `training/outputs/<model>-lora-merged/`
with `*.gguf` + `Modelfile`. Download **both** to your PC:
- **RunPod:** `runpodctl send <file>` (or the web file browser).
- **Vast.ai / Lambda:** `scp -P <port> root@<host>:/path/*.gguf .` and the `Modelfile`.

Then **on your PC** (Ollama installed), in the folder with those files:
```bash
ollama create my-roleplayer -f Modelfile
```
Open the app → **⚙ Settings → model → `my-roleplayer`**. Done — it now runs
locally, offline, on your own GPU. 🎭 **Stop/terminate the rented box** so it
stops billing.

---

## Manual steps (if you'd rather run stage by stage)
```bash
# env
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"
uv venv .venv-train --python 3.12 && source .venv-train/bin/activate
uv pip install unsloth --torch-backend=auto
uv pip install -r training/requirements-data.txt
export HF_TOKEN=hf_xxx HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

# data -> train -> export
python training/prepare_data.py --char-name "Aria"
python training/train_sft.py --model nemo-instruct            # add --smoke to test the pipeline
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
( cd ~/llama.cpp && pip install -r requirements.txt && cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j )
python training/export_gguf.py --adapter training/outputs/nemo-instruct-lora \
  --base mistralai/Mistral-Nemo-Instruct-2407 --template mistral \
  --llama-cpp-dir ~/llama.cpp --quants Q4_K_M --name my-roleplayer --skip-ollama
```

See [`README.md`](README.md) for what each script does and
[`../TRAINING.md`](../TRAINING.md) for the strategy, costs, and model choices.
