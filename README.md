# MutinyBIM Weight-Space Mixer

Crowdsourced **weight-space programming** for [`permitzip/mutinybim`](https://github.com/permitzip/mutinybim) on **`google/gemma-4-31B-it`**.

Instead of pooling Colab credits into one shared VM, everyone adopts a construction trade on **their own Google Colab GPU**, trains a tiny (~50 MB) spec-pinned **Trade LoRA** verified deterministically against `bim-script` + `bim-lint` + `bim-calc`, pushes it to Hugging Face, and anyone can blend every contributor's adapters in one click via **Task Arithmetic**, **Clash-Vector Negation** (`-λ · ΔW_clash`), **TIES**, **DARE**, or **CMA-ES EvoMerge** — then plug the merged multi-trade brain straight into `bim-mcp` to co-design live buildings inside `bim-editor`.

---

## One-Click Colab Notebooks

> **Experimental.** The Colab badges need the GitHub repository to be published. The CPU verifier can be tested before using any GPU time. GPU availability, memory fit, and compute-unit rates are not verified; check your Colab runtime picker.

| Notebook | What it does | GPU to try | Open in Colab |
| :--- | :--- | :--- | :--- |
| **`01_train_trade_lora.ipynb`** | Pick a trade (`architecture`, `mechanical`, `plumbing`, `electrical`, `drafting`, `structural`, `fire_protection`, `clash_negation`, `coord_rlvr`) → SFT + RLVR (`GRPOTrainer`) against live MutinyBIM → push ~50 MB LoRA + scorecard to HF | **A100 (40GB / 80GB)** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/5hyycjb77y/mutinybim-mixer/blob/main/notebooks/01_train_trade_lora.ipynb) |
| **`02_weight_space_mixer.ipynb`** | Pull community trade LoRAs → blend via Linear / TIES / DARE + `-λ·ΔW_clash` → inspect 60-layer SVD diagnostics → run CMA-ES EvoMerge against `bim-lint` → optional local inference preview | **L4 (24GB)** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/5hyycjb77y/mutinybim-mixer/blob/main/notebooks/02_weight_space_mixer.ipynb) |
| **`03_vibe_architect_studio.ipynb`** | **For architects, designers & vibe coders (Mixtape Edition):** Gradio studio app (sketch/prompt or Instant Showcase → real MutinyBIM IFC 4.3 + SVG/PDF sheets + 360° orbit GIF + interactive 3D GLB + 3-track Mixtape A/B strip + 4-shot photoreal storyboard + Vision Critique + ☀️ Solar Dome & Sun Hours) | **L4 (24GB)** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/5hyycjb77y/mutinybim-mixer/blob/main/notebooks/03_vibe_architect_studio.ipynb) |
| **`04_sketch_to_bim_lora.ipynb`** | **Self-supervised Sketch → BIM trainer:** Generate infinite verified `(napkin sketch, @permitzip/bim-script)` pairs across 6 topologies and train a weight-space-compatible `vision_floorplan` LoRA | **A100 (40GB / 80GB)** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/5hyycjb77y/mutinybim-mixer/blob/main/notebooks/04_sketch_to_bim_lora.ipynb) |

---

## Architecture

```text
  Colab Contributor A ───────────▶ ΔW_arch   (HF Hub) ─┐
  Colab Contributor B ───────────▶ ΔW_mech   (HF Hub) ─┼─▶ 02_weight_space_mixer.ipynb
  Colab Contributor C ───────────▶ ΔW_plumb  (HF Hub) ─┤     ├─ Exact Dense ΔW = (α/r)(B @ A)
  Colab Contributor D ───────────▶ -λ·ΔW_clash (HF)   ─┘     ├─ TIES / DARE / Task Arithmetic
                                                               ├─ 60-Layer SVD + CMA-ES EvoMerge
                                                               └─ OpenAI Endpoint + Tunnel
                                                                         │
                                                                         ▼
  Local Machine:  npm run demo  ◀──▶  npm run mcp:http  ◀──▶  python3 -m mixer.mcp_bridge
                   (bim-editor)         (:3333/mcp)             (preflight + 1-retry repair)
```

---

## Pinned Weight-Space Contract (`spec.json`)

Every contributor's LoRA must live in the exact same linear subspace so adapters add, subtract, and SVD-decompose cleanly:

* **Base Model:** `google/gemma-4-31B-it` (`unsloth/gemma-4-31B-it`, 60 dense Transformer layers, Apache-2.0, native thinking + native vision upside)
* **LoRA Spec:** `r = 16`, `lora_alpha = 16`, `lora_dropout = 0.0`, `bias = "none"`, `target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
* **MutinyBIM Verifier Target:** [`permitzip/mutinybim`](https://github.com/permitzip/mutinybim) @ commit `fbf5f61424b64d9f99b1c896ca02328c88d9813c` (Node `>=22.13`)

### Reward Ladder (`verifier/server.mjs`)
Every generated `defineScript(...)` is evaluated in-memory against warm `@permitzip/bim-script`, `@permitzip/bim-kernel`, and `@permitzip/bim-lint`:
1. **`format_and_lint` (`0.15`)** — `lintScript` + `kitUseWarnings` (hard security gate blocks unsafe imports before execution)
2. **`compile_and_load` (`0.15`)** — `node:module` `stripTypeScriptTypes` + ESM import
3. **`executes_cleanly` (`0.25`)** — `runScript` against `sliceModel`
4. **`kernel_check_ok` (`0.15`)** — `checkScriptOutput` preflight
5. **`bim_lint_clean` (`0.20`)** — `LintEngine.check` coordination & clash count
6. **`spec_match` (`0.10`)** — expected element kinds / minimum element count

---

## Repository Layout

```text
mutinybim-mixer/
├── spec.json                         # Pinned base model, LoRA spec, MutinyBIM commit, reward weights
├── registry.json                     # Community Trade LoRA catalog (HF repo IDs + eval scorecards)
├── verifier/
│   ├── server.mjs                    # Warm in-process MutinyBIM Node HTTP verifier
│   └── verify_client.py              # Python client + Gemma 4 <channel|thought> parser + GRPO reward_func
├── mixer/
│   ├── lora_ops.py                   # Exact dense-delta LoRA arithmetic, TIES, DARE, SVD factorizer & diagnostics
│   ├── evomerge.py                   # CMA-ES (with zero-dep NES fallback) 60-layer EvoMerge optimizer
│   ├── prompts.py                    # Single-trade training prompts + 6-task MULTI_TRADE_BENCHMARK
│   ├── sketch_engine.py              # Self-supervised parametric floor plans + organic napkin-sketch generator
│   ├── solar_engine.py               # Geolocation + solstice/equinox Sun-Dial HUD + 3D Solar Dome + Direct Sun Hours
│   ├── studio_render.py              # IFC → GLB + depth/clay + turntable GIF + 4-shot storyboard + Mixtape presets
│   └── mcp_bridge.py                 # Streamable HTTP MCP bridge to live `mutinybim-mcp` & `bim-editor`
├── notebooks/
│   ├── 01_train_trade_lora.ipynb     # One-click Colab trainer (SFT + RLVR GRPO)
│   ├── 02_weight_space_mixer.ipynb   # One-click Colab weight-space mixer + SVD plots + EvoMerge + server
│   ├── 03_vibe_architect_studio.ipynb # Mixtape Edition Studio: sketch/showcase → IFC/PDF/GLB/GIF + Solar Dome
│   └── 04_sketch_to_bim_lora.ipynb   # Self-supervised Napkin Sketch -> MutinyBIM multimodal LoRA trainer
└── tests/
    └── test_mixer.py                 # Zero-dependency unit tests
```

---

## Quickstart

### 1. Adopt a Trade & Train a LoRA
Open [`notebooks/01_train_trade_lora.ipynb`](notebooks/01_train_trade_lora.ipynb) in Colab on an A100, choose a `TRADE` in Cell 3, run SFT + GRPO, and push your ~50 MB adapter + `eval_scorecard.json` to Hugging Face. Add your entry to `registry.json` via PR!

### 2. Mix Community LoRAs & Co-Design in `bim-editor`
1. Open [`notebooks/02_weight_space_mixer.ipynb`](notebooks/02_weight_space_mixer.ipynb) and try the CPU-only synthetic merge first. The registry starts empty. Synthetic adapters check the merge plumbing; do not load them into Gemma or treat their results as model metrics. Enable model evaluation only with real compatible adapters.
2. Locally, clone and start MutinyBIM:
   ```bash
   git clone https://github.com/permitzip/mutinybim.git
   cd mutinybim && npm install
   npm run demo      # Terminal 1: bim-editor UI
   npm run mcp:http  # Terminal 2: Streamable HTTP MCP on http://127.0.0.1:3333/mcp
   ```
3. For an editor integration test, run inference and the bridge on a machine you control. The notebook’s optional loopback server is not a shared Colab endpoint:
   ```bash
   python3 -m mixer.mcp_bridge \
     --llm-url http://127.0.0.1:8000/v1 \
     --mcp-url http://127.0.0.1:3333/mcp
   ```

### 3. Try the Studio

Run setup and verifier checks before allocating a GPU. Use synthetic inputs, keep public Gradio sharing off for the initial run, and try one generation with the default SDXL backend before FLUX or training. Only grant a scoped Hugging Face token when a selected model or an explicit adapter upload needs it. Never put tokens in a cell, printed output, or Git. The notebook’s optional public link is not a private workspace.

### 4. Run Local Tests
No GPU or PyTorch required for the core test suite:
```bash
node --check verifier/server.mjs
PYTHONPATH=. python3 -m unittest discover -s tests -v
# Optional CPU integration: pinned MutinyBIM checkout with npm dependencies
MIXER_TEST_MUTINYBIM=/path/to/mutinybim python3 -m unittest discover -s tests -v
```

---

## License
Apache License 2.0 — see [LICENSE](LICENSE).
