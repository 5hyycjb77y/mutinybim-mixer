# AGENTS.md — MutinyBIM Mixer

Crowdsourced weight-space LoRA mixing, parametric napkin-sketch generation, geometry-locked rendering, and passive-solar analysis for [`permitzip/mutinybim`](https://github.com/permitzip/mutinybim) on `google/gemma-4-31B-it`.

## Pinned Contract (`spec.json`)
- **Base model:** `google/gemma-4-31B-it` (`unsloth/gemma-4-31B-it`, 60 layers)
- **LoRA subspace:** `r = 16`, `lora_alpha = 16`, `lora_dropout = 0.0`, `bias = "none"`, `target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
- **MutinyBIM target:** `permitzip/mutinybim` @ `fbf5f61424b64d9f99b1c896ca02328c88d9813c` (Node `>=22.13`)

---

## Connecting Agents (MCP)

### 1. `mutinybim-mixer` Stdio MCP Server (Local Tool Server)
`python3 -m mixer.mcp_server` is a **local tool server** that reads and writes files confined to `--workspace` / `MIXER_WORKSPACE` (default: repository root; symlinks escaping the workspace are refused) and uses only the `--verifier-url` / `MIXER_VERIFIER_URL` configured at startup (loopback-only by default unless `--allow-remote-verifier` is passed). Repo-root [`.mcp.json`](.mcp.json) auto-registers it:

```bash
python3 -m mixer.mcp_server
```

With a running MutinyBIM verifier in another terminal, pass `--verifier-url` (or `MIXER_VERIFIER_URL=http://127.0.0.1:8787`):

```bash
node verifier/server.mjs --mutinybim /path/to/mutinybim --port 8787
```

**Exposed MCP tools (9):**
- `mixer_info` — read `spec.json`, `registry.json`, and available camera/art/city/date/topology presets
- `generate_sketch` — synthesize a parametric floor plan + `@permitzip/bim-script` TypeScript + napkin sketch SVG
- `solar_position_and_dial` — solar altitude, clockwise-from-North azimuth, sunrise/sunset (local solar time, ±1° approx.), and SVG Sun-Dial HUD
- `check_envelope_gates` — True-North-aware closed exterior loop check, roof/top slab check, per-facade WWR, and fail-closed `compute_passive_solar_reward`
- `solar_direct_sun_hours` — zero-dep raycaster for explicit 3D points, or full IFC workplane grid + Ladybug 3D dome (`ifcopenshell`/`numpy`/`matplotlib` optional)
- `build_studio_prompt` — compose positive/negative photoreal prompts for a vibe + art preset + camera preset
- `check_adapter_compatibility` — verify LoRA configs match the pinned rank/modules in `spec.json`
- `verify_script` — lint + compile + execute + `bim-lint` via `verifier/server.mjs`
- `export_bundle` — export IFC 4.3, SVG sheets, vector PDF, and run `envelope_gates` via `verifier/server.mjs`

### 2. Google Colab GPU Bridge (`googlecolab/colab-mcp`)
To run the Colab notebooks (`01_train_trade_lora.ipynb`, `02_weight_space_mixer.ipynb`, `03_vibe_architect_studio.ipynb`, `04_sketch_to_bim_lora.ipynb`) or GPU checks (`test_closed_box_interior_no_holes`, showcase exports, 3D Solar Dome) from an agent:

```json
{
  "mcpServers": {
    "colab-mcp": {
      "command": "uvx",
      "args": ["git+https://github.com/googlecolab/colab-mcp"]
    }
  }
}
```

### 3. Live `bim-editor` Co-Design Bridge
Connect an OpenAI-compatible Gemma endpoint directly to MutinyBIM's `npm run mcp:http` (`http://127.0.0.1:3333/mcp`):

```bash
python3 -m mixer.mcp_bridge --llm-url http://127.0.0.1:8000/v1 --mcp-url http://127.0.0.1:3333/mcp
```

---

## Guardrails & Verification
- Keep top-level imports in `mixer/*.py` 100% Python stdlib (`torch`, `numpy`, `PIL`, `ifcopenshell`, `trimesh`, `matplotlib`, `diffusers`, `cairosvg` must stay lazy-imported).
- Keep `compute_passive_solar_reward` fail-closed (required gate keyword arguments, `0.0` when `verify_ok`, `spec_match == 1.0`, `has_closed_envelope`, or `max_facade_wwr <= 0.40` fail).
- Keep `_should_skip_ifc_class` skipping `IfcOpeningElement`, `IfcOpeningStandardCase`, `IfcSpace`, `IfcVirtualElement`, `IfcAnnotation`, and `IfcGrid` in `studio_render.py`.
- Run verification after edits:
  ```bash
  node --check verifier/server.mjs
  python3 -m py_compile mixer/*.py tests/*.py
  PYTHONPATH=. python3 -m unittest discover -s tests -v
  MIXER_TEST_MUTINYBIM=/path/to/mutinybim MIXER_SKETCH_SEEDS=10 PYTHONPATH=. python3 -m unittest discover -s tests -v
  ```
