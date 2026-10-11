# 🧠 Train your own roleplay model (the custom-model track)

This is the plan for building a **custom AI model tuned for roleplay + learning**
that drops straight into this app. The app already talks to [Ollama](https://ollama.com),
so the finish line is simple: you end up with a `.gguf` file and an Ollama
"model" that shows up in the Settings dropdown like any other. Then every
feature you already have — lorebooks, RAG memory, group scenes, personas,
swipes — runs on *your* brain instead of a stock one.

> Researched 2026-10 from current community + vendor sources. Numbers in this
> space move fast; links are at the bottom so you can re-check prices and model
> picks before you spend money.

> 🛠️ **The runnable toolkit is built** — this doc is the *strategy*; the
> *scripts* live in **[`training/`](training/README.md)** (dataset prep, a
> one-command importer for your own stories, the QLoRA trainer tuned for your
> 20 GB card, DPO polish, and GGUF→Ollama export). Jump there when you're ready
> to actually run it.

---

## First, the honest reality check ✋

You said you're willing to run your PC for months and/or cloud-train. Here's the
truth so you don't waste either:

- **You are *not* training a model from scratch.** A from-scratch model
  (GPT/Claude-class) costs *millions of dollars* and thousands of GPUs running
  for months. That's not a home project, and you don't need it — the hard part
  (grammar, reasoning, world knowledge) is already done by open "base" models
  that smart people released for free.
- **You are *fine-tuning*** — taking one of those open base models and nudging
  its *personality, style, and behavior* toward great roleplay. This is the
  thing hobbyists actually do, and it's what every model in the app's README
  table (Cydonia, Mag-Mell, etc.) already is: a fine-tune of a base model.
- **It takes hours-to-days, not months.** A roleplay QLoRA fine-tune of a
  7B–13B model is a few GPU-hours. "Months on end" would only happen if you
  tried to full-train from scratch (don't) or kept re-running experiments (fun,
  but optional).
- **Cloud is usually *cheaper and faster* than your home PC for the actual
  training run** — a rented A100/H100 for a few hours costs ~$5–40 and finishes
  while your PC would still be warming up. Your home GPU is perfect for
  *running* the finished model 24/7; renting is better for the occasional
  training burst. You can do either; this plan supports both.

So the realistic goal: **a personalized roleplay model, trained in an afternoon
(local) or an hour (cloud), that's yours forever and runs offline.** 🎯

---

## 📍 Your setup → your exact path

**Your rig:** Ryzen 9 7950X3D (16-core), 96 GB RAM, **20 GB VRAM now → ~52 GB**
at end of month (adding a 32 GB card). You want the model to do **both** —
*teach* you things *and* adapt to you. This is a genuinely strong
home-training box; here's the plan tuned to it.

**Right now (single 20 GB GPU → use Unsloth):**
- 20 GB with QLoRA trains **7–8B** easily and handles **Mistral-Nemo 12B** with
  gradient checkpointing + batch size 1–2 + ~4–8K training context. So your
  *keeper* (12B) is home-trainable **today**, no cloud needed.
- Your **96 GB RAM** makes the GGUF export step trivial (it only wants 32–64 GB),
  and the 7950X3D chews through data prep. No bottleneck anywhere.
- Plan: v0 plumbing run on **Llama 3.1 8B** (fast/cheap to shake out bugs), then
  the real one on **Mistral-Nemo 12B**.

**End of month (20 + 32 = ~52 GB, two GPUs → switch to Axolotl + FSDP-QLoRA):**
- Dual-GPU unlocks **Mistral Small 24B** (companion-grade) QLoRA and fast 12B at
  long context. With FSDP-QLoRA you *can* even attempt a **70B** run, though it's
  tight and the mismatched 20+32 cards balance around the smaller one — so **24B
  is your sweet spot**.
- Tool swap: **Unsloth is single-GPU**, so use **Axolotl** (FSDP/DeepSpeed) to
  drive both cards. Keep Unsloth for quick single-card experiments.

**Making it do "both":** one model, two mechanisms —
- *Teaches you* → **bake it into the fine-tune**: blend ~10–30% instruction /
  educational data + a handful of tutor-style example dialogues into your RP
  training mix.
- *Adapts to you* → **don't fine-tune for this** — the app's **RAG memory**
  already learns your stories live and updates instantly. Fine-tuning to
  "remember new facts" is the catastrophic-forgetting trap (see Step 6).

➡️ **Bottom line: one fine-tune for voice + teaching, RAG for the adapting.**
Train a 12B on your 20 GB card now; graduate to 24B on the dual-GPU rig at
month's end. You may never need to rent a cloud GPU at all. 🎉

---

## The pipeline at a glance

```
┌─ 1. Pick a BASE model ──────────────┐
│  (Mistral-Nemo 12B = default)       │
└──────────────┬──────────────────────┘
               ▼
┌─ 2. Build your DATASET ─────────────┐   roleplay convos + (optional) "learning" data
│  PIPPA + LimaRP + novels + yours    │
└──────────────┬──────────────────────┘
               ▼
┌─ 3. TRAIN (QLoRA) ──────────────────┐   Unsloth, a few GPU-hours
│  SFT  →  (optional) DPO/ORPO polish │
└──────────────┬──────────────────────┘
               ▼
┌─ 4. EVALUATE ───────────────────────┐   roleplay vibe-check + capability regression
└──────────────┬──────────────────────┘
               ▼
┌─ 5. EXPORT to GGUF + Ollama ────────┐   merge → convert → quantize → Modelfile
└──────────────┬──────────────────────┘
               ▼
        ✅ shows up in THIS app's model dropdown
```

---

## Step 0 — Decide what "learning" means to you

You said "roleplay **and learning**." That word has two very different meanings,
and they need different approaches. Pick one (or both):

| If you meant… | What it is | Best approach |
|---|---|---|
| **A. A model that *teaches*** (tutor/educational — explains topics, quizzes you, stays in a teaching persona) | A roleplay *style* on top of instructional content | **Fine-tune.** Mix educational/instruction datasets into your training mix and write tutor-style example dialogues. |
| **B. A model that *adapts to you*** (remembers your stories, learns your preferences over time) | Per-user personalization | **Mostly RAG, a little fine-tune.** The app's RAG memory already does the "remembers your stories" part live. Fine-tuning bakes in *stable* taste (tone, formatting, what you like), not changing facts. |

**Recommendation:** For "learns *about me / my stories* as we go," lean on the
**RAG memory this app already has** — it's safer and updates instantly. Reserve
**fine-tuning** for *stable* things: the model's voice, its roleplay instincts,
and (if you want a tutor) its teaching behavior. Trying to make a fine-tune
"remember new facts" invites **catastrophic forgetting** (it gets worse at
everything else) — that's the #1 documented failure mode.

---

## Step 1 — Pick a base model

This is the single biggest decision. Recommendations, best first:

| Base model | Size | Fits on | License | Why |
|---|---|---|---|---|
| **⭐ Mistral-Nemo** | 12B | 24GB GPU (QLoRA) or cloud | **Apache 2.0** (cleanest) | 128K context, the current community sweet spot for RP fine-tunes. **Start here.** |
| Llama 3.1 | 8B | 16GB GPU / cheap cloud | Llama Community | Smaller, cheaper, fastest to iterate. Great for your *first* run. |
| Qwen2.5 | 14B | 24GB (tight) / cloud | mixed (Apache/Qwen) | Very strong reasoning; good if you want the "learning"/tutor angle. |
| Mistral Small | 24B | 2×24GB or cloud (48GB+) | Apache 2.0 | Best quality here; this is what the app's top "24GB" pick (Cydonia) is built on. Cloud-tune it. |

**Models to study / benchmark against** (don't reinvent — read their recipes
and dataset lists on Hugging Face): **Cydonia** (TheDrummer), **ArliAI RPMax**,
**Mag-Mell 12B**, and **Pantheon-RP** (Gryphe — a Nemo-based RP tune that even
used KTO preference training). These are your north stars for "what good looks
like."

> **Rule of thumb:** do your *first* end-to-end run on **Llama 3.1 8B** because
> it's fast and cheap to get all the plumbing working, then redo it on
> **Mistral-Nemo 12B** for the model you actually keep.

---

## Step 2 — Build your dataset (this is where quality actually comes from)

The model is only as good as what you feed it. **Data quality >> data quantity.**
A few thousand *excellent* examples beat a million mediocre ones.

### Roleplay data (the core)
| Dataset | What's in it | Notes |
|---|---|---|
| **PIPPA** (PygmalionAI) | ~1M lines / 26k conversations, 1000+ personas | The classic RP set. **Caveat: replies are short** — mix with novel-style data or your model will be terse. |
| **LimaRP** (lemonilia) | small, *hand-curated* high-quality RP | Quality over quantity — exactly the kind of data that punches above its weight. |
| **CoSER** | ~29,800 conversations from 771 famous books | Book-grounded character dialogue → richer, longer prose. |
| **Gutenberg / novel-style sets** (e.g. Gryphe's) | public-domain prose | Teaches long, descriptive, "novel" replies. |
| **Airoboros** (RP subset) | mixed instruction + RP | Good general glue. |

### "Learning" data (only if you picked tutor-mode in Step 0)
Mix in **instruction / educational** datasets (e.g. high-quality instruction
sets, Q&A, or subject material you care about) — but keep it a *minority* of the
mix (~10–30%) so the model stays a roleplayer that *can* teach, not a dry
textbook.

### Your own data (the secret weapon) 🔑
Export your best stories from this app (the **💾 export** button → JSON) and turn
good exchanges into training examples. **A few hundred examples in *your* taste**
will personalize the model more than any public set. This is the highest-value,
lowest-effort step.

### Formatting rules that matter
- Keep each base model's **chat template** (Mistral uses `[INST]`-style /
  "Tekken"; Nemo RP tunes often use ChatML — match whatever your base expects).
- **Dedup aggressively** and strip "slop" (repetitive AI-isms). The RPMax recipe
  is explicitly built around *variety + deduplication* — copy that discipline.
- Vary reply length and style so the model doesn't collapse into one voice.

---

## Step 3 — Train it (method + tool)

### Method: QLoRA, then (optionally) a preference pass
1. **SFT with QLoRA** (supervised fine-tune, 4-bit). This is 95% of the result.
   QLoRA lets you fine-tune big models on one consumer GPU by freezing the base
   in 4-bit and training tiny adapter weights.
2. *(Optional polish)* **DPO / ORPO / KTO** — preference training that teaches
   "prefer *this* kind of reply over *that* one." **ORPO** is nice because it
   folds the preference step into one stage. Pantheon-RP used **KTO**. Do this
   *after* you have a working SFT model and want to push style/reduce refusals.

### Tool: pick one
| Tool | Best for | Notes |
|---|---|---|
| **⭐ Unsloth** | Fastest single-GPU QLoRA | ~2× faster, ~70% less VRAM. **Use this on a home GPU or a single rented card.** Free tier is single-GPU. |
| **Axolotl** | YAML-reproducible / multi-GPU / all preference methods | Does SFT, LoRA, QLoRA, DPO, KTO, ORPO, GRPO; scales to multi-GPU with FSDP/DeepSpeed. Use for the serious Nemo/Mistral-Small run. |
| **LLaMA-Factory** | No-code web UI | Easiest if you don't want to touch configs; can turn on the Unsloth speedup internally. |

**Recommendation:** **Unsloth** for your first runs (fast, cheap, one GPU),
graduate to **Axolotl** if you go multi-GPU or want ORPO/KTO with reproducible
configs.

---

## Step 4 — Hardware: your PC vs the cloud

### VRAM needed (QLoRA, rough — depends on context length & batch size)
| Model size | VRAM for QLoRA | Your options |
|---|---|---|
| 7–8B | ~12 GB | Any 16GB+ GPU, or a 24GB card comfortably |
| 12–13B | fits on **24 GB** (RTX 4090/3090) | Home-trainable on a 24GB card 🎉 |
| 24B | > one 24GB card | 2×24GB (FSDP-QLoRA) or rent a 48GB+ |
| 70B | 48–80 GB (or 2×24 w/ FSDP-QLoRA) | Cloud only |

### Cloud pricing (Oct 2026 snapshot — *verify live before renting*)
| GPU | ~$/hr | Where |
|---|---|---|
| A100 80GB | **$1.05–1.79** | Vast.ai cheapest, RunPod mid |
| H100 | $1.50–2.99 | Vast.ai cheapest, Lambda most stable/pricey |

- **Vast.ai** = cheapest, marketplace, *no uptime SLA* (fine for a short run,
  checkpoint often).
- **RunPod** = middle ground, community + secure tiers.
- **Lambda** = most stable/supported, most expensive.

**A typical 7–13B roleplay QLoRA run is a few GPU-hours → roughly $5–40 total.**
That's why cloud often wins for the *training burst*: cheaper than the
electricity + time of grinding your PC for days.

**Suggested split:** rent a GPU for the *training run*, then **run the finished
model on your home GPU 24/7** (free, offline, private). Best of both.

---

## Step 5 — Export to GGUF + plug into this app

This is the payoff — turning your trained adapter into something Ollama (and
therefore this app) can run:

1. **Merge the adapter into the base** (PEFT `merge_and_unload()`), save the
   merged model + tokenizer.
2. **Convert to GGUF** with `llama.cpp`'s conversion script.
3. **Quantize** — `Q4_K_M` (small/fast, great default), `Q5_K_M` (a bit nicer),
   or `Q8_0` (highest quality, bigger). A 12B at Q4_K_M runs on ~10–12GB VRAM.
4. **Write an Ollama `Modelfile`:**
   ```
   FROM ./your-model-Q4_K_M.gguf
   TEMPLATE """{{ .System }} ... """   # match your base's chat template
   PARAMETER temperature 0.9
   PARAMETER stop "<|im_end|>"
   ```
5. **Register it:** `ollama create my-roleplayer -f Modelfile`
6. **Done** — `my-roleplayer` now appears in this app's **⚙ Settings → model**
   dropdown. Pick it and play. The app auto-applies a sampler preset; tweak in
   the advanced panel.

> Disk/RAM for conversion: keep ~32GB system RAM free (64GB for 14B+) and lots
> of free disk for the intermediate float16 files.

---

## Step 6 — Evaluate (don't skip — this is how you avoid a worse model)

Fine-tuning can quietly make the model *dumber* at general tasks (catastrophic
forgetting) even while roleplay feels better. Check both:

1. **Roleplay vibe-check** — run a fixed set of your own scenarios through the
   app, compare against the base model and a reference tune (Cydonia/Mag-Mell).
   Look for: stays in character, doesn't speak for you, varied prose, follows
   the lorebook.
2. **Capability regression** — ask it some reasoning/instruction questions it
   could answer *before* training. If it got noticeably worse, you overcooked it
   (too many epochs / learning rate too high / data too narrow). Back off.
3. **Repetition / slop check** — long scenes shouldn't collapse into repeated
   phrases.

**Common mistakes:** too many epochs (overfitting → parroting training data),
learning rate too high, dataset too small or too samey, forgetting to match the
chat template. When in doubt: *less* training, *better* data.

---

## 🗺️ Suggested roadmap (build list for the model track)

- [ ] **v0 — plumbing (a weekend):** Llama 3.1 8B + a small slice of PIPPA/LimaRP,
      QLoRA via Unsloth on a rented A100 for an hour. Export to GGUF, load in the
      app. Goal: *the whole pipeline works end to end.* (~$5–15)
- [ ] **v0.5 — your taste:** add a few hundred examples exported from your own
      best stories. Retrain. This is where it starts feeling like *yours*.
- [ ] **v1 — the keeper:** Mistral-Nemo 12B, a curated mix (PIPPA + LimaRP +
      novel-style + yours, deduped), QLoRA SFT. Export, benchmark against Cydonia.
- [ ] **v1.5 — polish (optional):** an ORPO/KTO preference pass to sharpen style
      and reduce refusals.
- [ ] **v2 — go big (optional, cloud):** Mistral Small 24B on a 48GB+ rented GPU
      for companion-grade quality.
- [ ] **tutor variant (if you want "learning"):** fork v1's dataset, blend in
      ~10–30% educational/instruction data + tutor-style examples, retrain as a
      separate model.

---

## 💵 Time & cost summary

| Path | Hardware | Time | Cost |
|---|---|---|---|
| First run (8B) | rented A100, 1 session | ~1–2 hrs | ~$5–15 |
| Keeper (12B) | 24GB home GPU **or** rented A100 | afternoon / ~2–4 hrs | $0 (home) or ~$10–40 (cloud) |
| Big (24B) | rented 48GB+ | a few hrs | ~$20–60 |
| From scratch ❌ | thousands of GPUs | months | **millions — don't** |

Running the finished model in this app afterward: **free & offline** on your
own GPU.

---

## Sources
- [Best Local LLMs for Character Roleplay (2026)](https://www.aicompanionpick.com/best-local-llms-for-character-roleplay) · [Best Open Source LLMs 2025 (Klu)](https://klu.ai/blog/open-source-llm-models)
- [PIPPA dataset](https://huggingface.co/datasets/PygmalionAI/PIPPA) · [PIPPA paper](https://arxiv.org/pdf/2308.05884) · [Dampf's dataset list](https://rentry.co/datasets-llm) · [lemonilia Roleplay-Forums](https://huggingface.co/datasets/lemonilia/Roleplay-Forums_2023-04)
- [Cydonia (TheDrummer)](https://huggingface.co/TheDrummer/Cydonia-22B-v1.2) · [ArliAI RPMax](https://huggingface.co/Statuo/MS-ArliAI-RPMax-22b-6bpw) · [Pantheon-RP (Gryphe)](https://huggingface.co/Gryphe/Pantheon-RP-1.6-12b-Nemo)
- [Unsloth vs Axolotl vs LLaMA-Factory](https://theaiengineer.substack.com/p/unsloth-vs-axolotl-vs-llama-factory) · [Framework comparison 2026 (MarkTechPost)](https://www.marktechpost.com/2026/07/22/unsloth-vs-axolotl-vs-trl-vs-llama-factory-a-fine-tuning-framework-comparison-on-speed-vram-and-multi-gpu/) · [Fine-tuning guide 2026 (Codersera)](https://codersera.com/blog/fine-tuning-llms-complete-guide-2026/)
- [GPU VRAM for fine-tuning 2026 (Spheron)](https://www.spheron.network/blog/gpu-vram-requirements-fine-tune-llm-2026/) · [QLoRA explained (Gamut)](https://www.gamut.so/blog/qlora-explained)
- [RunPod vs Vast.ai 2026](https://www.spheron.network/blog/runpod-vs-vastai-2026/) · [Live GPU pricing](https://gpuperhour.com/) · [H100 prices compared](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison)
- [When RAG isn't enough (fine-tune vs RAG)](https://bigdataboutique.com/blog/fine-tuning-llms-when-rag-isnt-enough) · [Catastrophic forgetting in LLM tuning](https://arxiv.org/pdf/2406.04836)
- [GGUF conversion script example](https://huggingface.co/spaces/hf-skills/llm-trainer/blob/main/scripts/convert_to_gguf.py)
