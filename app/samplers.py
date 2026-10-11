"""Per-model sampler presets for roleplay.

Recommended starting points differ by model family (researched from
r/LocalLLaMA / r/SillyTavernAI and model cards). A single global temperature
can't express this, so we key presets by a substring of the Ollama model tag
and merge over sensible defaults. Users can still override per story.

Notes baked in from research:
- `min_p` is the primary truncation sampler for RP and MUST be set explicitly
  (Ollama's default is 0.0 = off). It + temperature are the backbone.
- High `repeat_penalty` (>~1.15) paradoxically causes looping and corrupts
  structured output. Some models (Cydonia/Magnum 24B) want 1.0.
- Ollama's options object has NO dry_* / xtc_* keys — those samplers are not
  available; we lean on min_p + an in-prompt anti-cliché list instead.
"""
from __future__ import annotations

# Safe, universal defaults (work on every model and code path).
DEFAULTS = {
    "temperature": 1.0,
    "min_p": 0.05,
    "top_p": 0.95,
    "top_k": 40,
    "repeat_penalty": 1.08,
    "repeat_last_n": 256,
    "num_predict": 512,
    "num_ctx": 16384,
}

# (substrings, overrides). First match wins. Keep substrings lowercase.
_PRESETS = [
    (["cydonia", "magnum", "mag-mell", "magmell"],
        {"temperature": 1.0, "min_p": 0.05, "repeat_penalty": 1.0}),
    (["personalityengine", "personality-engine"],
        {"temperature": 1.0, "min_p": 0.05, "repeat_penalty": 1.0}),
    (["qwen3", "qwq"],
        {"temperature": 0.7, "min_p": 0.05, "top_k": 20, "repeat_penalty": 1.05}),
    (["gemma3", "gemma-3", "gemma2", "gemma-2"],
        {"temperature": 1.0, "min_p": 0.05}),  # Go runner may ignore penalties
    (["mistral-nemo", "nemo"],
        {"temperature": 1.0, "min_p": 0.08, "repeat_penalty": 1.08}),
    (["stheno", "llama3", "llama-3", "llama3.1", "llama-3.1"],
        {"temperature": 1.0, "min_p": 0.05, "repeat_penalty": 1.05}),
    (["dolphin"],
        {"temperature": 1.0, "min_p": 0.05, "repeat_penalty": 1.08}),
]


def preset_for(model: str) -> dict:
    """Return a full settings dict (DEFAULTS merged with any family preset)."""
    out = dict(DEFAULTS)
    name = (model or "").lower()
    for subs, override in _PRESETS:
        if any(s in name for s in subs):
            out.update(override)
            break
    return out


# Keys we forward to Ollama's `options`. `num_ctx` / `num_predict` are also here.
OLLAMA_OPTION_KEYS = [
    "temperature", "min_p", "top_p", "top_k",
    "repeat_penalty", "repeat_last_n",
    "num_ctx", "num_predict", "seed",
]


def build_options(settings: dict) -> dict:
    """Pick only the keys Ollama understands, dropping Nones."""
    opts = {}
    for k in OLLAMA_OPTION_KEYS:
        v = settings.get(k)
        if v is not None:
            opts[k] = v
    return opts
