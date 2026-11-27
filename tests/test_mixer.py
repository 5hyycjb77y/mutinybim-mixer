"""Stdlib unit tests for mutinybim-mixer (no GPU or third-party packages required)."""

from __future__ import annotations

import ast
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from mixer.evomerge import EvoMergeConfig, EvoMergeOptimizer
from mixer.lora_ops import (
    check_adapter_compatibility,
    depth_band_weights,
    load_spec,
    parse_lora_key,
)
from mixer.prompts import MULTI_TRADE_BENCHMARK, TRADE_TASKS, format_task_prompt
from mixer.studio_render import (
    ART_DIRECTION_PRESETS,
    CAMERA_PRESETS,
    VIBE_INSPIRATION_CHIPS,
    build_render_prompt,
)
from verifier.verify_client import (
    ExportBundle,
    MutinyVerifierClient,
    extract_typescript,
    make_grpo_reward_func,
    strip_gemma_thinking,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestConfigs(unittest.TestCase):
    def test_spec_json_valid(self) -> None:
        spec = load_spec(REPO_ROOT / "spec.json")
        self.assertEqual(spec["base_model"]["hf_id"], "google/gemma-4-31B-it")
        self.assertEqual(spec["base_model"]["num_hidden_layers"], 60)
        self.assertEqual(spec["lora"]["r"], 16)
        self.assertEqual(spec["mutinybim"]["commit"], "efe343144c61cd8e08890b75f5f45c3541fe4abb")
        self.assertAlmostEqual(sum(spec["reward_weights"].values()), 1.0, places=6)
        for trade in (
            "architecture",
            "mechanical",
            "plumbing",
            "electrical",
            "drafting",
            "structural",
            "fire_protection",
            "clash_negation",
            "coord_rlvr",
        ):
            self.assertIn(trade, spec["trades"])

    def test_registry_json_valid(self) -> None:
        reg = json.loads((REPO_ROOT / "registry.json").read_text(encoding="utf-8"))
        self.assertEqual(reg["spec_version"], "0.1.0")
        self.assertEqual(reg["base_model"], "google/gemma-4-31B-it")
        self.assertIsInstance(reg["adapters"], list)
        self.assertIn("scorecard", reg["example_entry"])


class TestGemmaThinkingAndExtraction(unittest.TestCase):
    def test_strip_gemma_thinking_closed(self) -> None:
        raw = (
            "<|think|><|channel>thought\nLet's compute wall coordinates at 10 ft.\n<channel|>"
            "Here is the script:\n```ts\nimport { defineScript } from '@permitzip/bim-script';\nexport default defineScript({ kind: 'layout', reads: {}, parameters: {}, run() {} });\n```"
        )
        thought, answer = strip_gemma_thinking(raw)
        self.assertIn("wall coordinates", thought)
        self.assertNotIn("<|channel>", answer)
        ts = extract_typescript(raw)
        self.assertTrue(ts.startswith("import { defineScript }"))

    def test_strip_gemma_thinking_unclosed_with_code(self) -> None:
        raw = "<|channel>thought\nPlanning...\n```typescript\nimport { defineScript } from '@permitzip/bim-script';\nexport default defineScript({ kind: 'route', reads: {}, parameters: {}, run() {} });\n```"
        thought, _ = strip_gemma_thinking(raw)
        self.assertIn("Planning", thought)
        ts = extract_typescript(raw)
        self.assertIn("defineScript", ts)

    def test_extract_typescript_prefers_define_script(self) -> None:
        raw = "```ts\nconst x = 1;\n```\n\n```ts\nimport { defineScript } from '@permitzip/bim-script';\nexport default defineScript({ kind: 'part', reads: {}, parameters: {}, run() {} });\n```"
        ts = extract_typescript(raw)
        self.assertIn("defineScript", ts)
        self.assertNotIn("const x = 1", ts)


class TestMockVerifierClient(unittest.TestCase):
    server: ThreadingHTTPServer
    base_url: str

    @classmethod
    def setUpClass(cls) -> None:
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def _send(self, code: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/health":
                    self._send(200, {"status": "ok", "mutinybim_commit": "efe343144c61"})
                elif self.path.startswith("/reference"):
                    self._send(200, {"markdown": "# Ref", "declarations": "", "examples": []})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                if self.path == "/verify":
                    src = body.get("source", "")
                    has_clash = "CLASH" in src
                    ok = "defineScript" in src and not has_clash
                    self._send(
                        200,
                        {
                            "ok": ok,
                            "reward": 0.95 if ok else (0.55 if has_clash else 0.1),
                            "breakdown": {"kernel_check_ok": 1.0 if ("defineScript" in src) else 0.0},
                            "stage_failed": "bim_lint_errors" if has_clash else None,
                            "error": None,
                            "steps": 3 if ok else 0,
                            "label": "Mock run",
                            "kinds": {"wall": 4},
                            "lint_summary": {"errors": 2 if has_clash else 0, "warnings": 0, "infos": 0, "findings": []},
                            "elapsed_ms": 1.5,
                        },
                    )
                elif self.path == "/verify_batch":
                    items = body.get("items", [])
                    results = []
                    for item in items:
                        src = item.get("source", "")
                        has_clash = "CLASH" in src
                        ok = "defineScript" in src and not has_clash
                        results.append(
                            {
                                "ok": ok,
                                "reward": 0.95 if ok else (0.55 if has_clash else 0.1),
                                "breakdown": {"kernel_check_ok": 1.0 if ("defineScript" in src) else 0.0},
                                "stage_failed": "bim_lint_errors" if has_clash else None,
                                "error": None,
                                "steps": 3 if ok else 0,
                                "label": "Mock run",
                                "kinds": {"wall": 4},
                                "lint_summary": {"errors": 2 if has_clash else 0, "warnings": 0, "infos": 0, "findings": []},
                                "elapsed_ms": 1.2,
                            }
                        )
                    self._send(200, {"results": results})
                elif self.path == "/export":
                    self._send(
                        200,
                        {
                            "ok": True,
                            "reward": 0.96,
                            "breakdown": {"kernel_check_ok": 1.0},
                            "stage_failed": None,
                            "error": None,
                            "steps": 4,
                            "label": "Mock studio export",
                            "kinds": {"wall": 4, "sheet": 1},
                            "lint_summary": {"errors": 0, "warnings": 0, "infos": 0, "findings": []},
                            "elapsed_ms": 2.0,
                            "ifc_text": "ISO-10303-21;\nHEADER;\nENDSEC;\n",
                            "ifc_report": {"schema": "IFC4X3_ADD2"},
                            "sheets": [{"id": "s1", "number": "A-101", "title": "Plan", "svg": "<svg></svg>"}],
                            "pdf_base64": "JVBERi0xLjQK",
                        },
                    )

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        port = cls.server.server_address[1]
        cls.base_url = f"http://127.0.0.1:{port}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def test_client_and_grpo_reward(self) -> None:
        client = MutinyVerifierClient(self.base_url)
        self.assertEqual(client.health()["status"], "ok")
        self.assertIn("markdown", client.reference(compact=True))

        res = client.verify("export default defineScript({})")
        self.assertTrue(res.ok)
        self.assertAlmostEqual(res.reward, 0.95)

        reward_fn = make_grpo_reward_func(client)
        scores = reward_fn(
            ["p1", "p2"],
            [
                "<|channel>thought\nok<channel|>```ts\nexport default defineScript({})\n```",
                "broken output",
            ],
        )
        self.assertEqual(len(scores), 2)
        self.assertAlmostEqual(scores[0], 0.95)
        self.assertAlmostEqual(scores[1], 0.1)

        clash_reward_fn = make_grpo_reward_func(client, invert_for_clash_adapter=True)
        clash_scores = clash_reward_fn(
            ["p1", "p2"],
            [
                "```ts\n// CLASH\nexport default defineScript({})\n```",
                "```ts\nexport default defineScript({})\n```",
            ],
        )
        self.assertGreater(clash_scores[0], clash_scores[1])

        bundle = client.export_bundle("export default defineScript({})")
        self.assertIsInstance(bundle, ExportBundle)
        self.assertTrue(bundle.verify.ok)
        self.assertIn("ISO-10303-21", bundle.ifc_text or "")
        self.assertEqual(len(bundle.sheets), 1)
        self.assertEqual(bundle.sheets[0]["number"], "A-101")
        self.assertEqual(bundle.pdf_base64, "JVBERi0xLjQK")


class TestLoraOpsAndEvoMerge(unittest.TestCase):
    def test_parse_lora_key(self) -> None:
        k1 = parse_lora_key("base_model.model.model.language_model.layers.42.self_attn.q_proj.lora_A.weight")
        self.assertIsNotNone(k1)
        assert k1 is not None
        self.assertEqual(k1.layer_idx, 42)
        self.assertEqual(k1.module_name, "q_proj")
        self.assertEqual(k1.factor, "lora_A")

        k2 = parse_lora_key("model.layers.0.mlp.gate_proj.lora_B.default.weight")
        self.assertIsNotNone(k2)
        assert k2 is not None
        self.assertEqual(k2.layer_idx, 0)
        self.assertEqual(k2.module_name, "gate_proj")
        self.assertEqual(k2.factor, "lora_B")
        self.assertEqual(k2.adapter_name, "default")

    def test_depth_band_weights_60_layers(self) -> None:
        sched = depth_band_weights([0.2, 0.8, 0.4], num_layers=60)
        self.assertEqual(len(sched), 60)
        self.assertAlmostEqual(sched[0], 0.2, places=6)
        self.assertAlmostEqual(sched[-1], 0.4, places=6)
        self.assertGreater(sched[30], 0.6)

    def test_check_adapter_compatibility(self) -> None:
        spec = load_spec(REPO_ROOT / "spec.json")
        good_cfg = {
            "base_model_name_or_path": "google/gemma-4-31B-it",
            "target_modules": spec["lora"]["target_modules"],
            "bias": "none",
        }
        self.assertEqual(check_adapter_compatibility(good_cfg, spec), [])

        bad_cfg = {
            "base_model_name_or_path": "Qwen/Qwen3-14B",
            "target_modules": ["q_proj"],
            "bias": "all",
        }
        issues = check_adapter_compatibility(bad_cfg, spec)
        self.assertEqual(len(issues), 3)

    def test_evomerge_nes_convergence(self) -> None:
        cfg = EvoMergeConfig(
            mode="depth_bands",
            num_bands=6,
            num_layers=60,
            method="ties",
            population_size=10,
            generations=15,
            sigma=0.25,
            allow_negative_indices=(2,),
            seed=7,
        )
        opt = EvoMergeOptimizer(num_adapters=3, config=cfg)
        self.assertEqual(opt.dimension(), 3 * 6 + 1)

        target_w0 = 0.9
        target_w1 = 0.6
        target_w2 = -0.4

        def fitness(genome: dict[str, Any]) -> float:
            s0 = sum(genome["layer_schedules"][0]) / 60.0
            s1 = sum(genome["layer_schedules"][1]) / 60.0
            s2 = sum(genome["layer_schedules"][2]) / 60.0
            d = genome["density"]
            err = (s0 - target_w0) ** 2 + (s1 - target_w1) ** 2 + (s2 - target_w2) ** 2 + (d - 0.5) ** 2
            return 1.0 - err

        init_fit = fitness(opt.decode(opt.initial_vector()))
        res = opt.optimize(fitness)
        self.assertGreaterEqual(res.best_fitness, init_fit)
        self.assertLess(res.best_genome["layer_schedules"][2][0], 0.0)

    def test_optional_torch_merge_and_svd_roundtrip(self) -> None:
        try:
            import torch  # type: ignore
            from mixer.lora_ops import merge_deltas, refactorize_deltas_to_lora, svd_diagnostics
        except ImportError:
            self.skipTest("torch not installed in local environment")

        prefix = "base_model.model.model.language_model.layers.0.self_attn.q_proj"
        a_delta = {prefix: torch.eye(16) * 0.5}
        b_delta = {prefix: torch.eye(16) * 0.25}
        clash_delta = {prefix: torch.eye(16) * 0.25}

        merged = merge_deltas([a_delta, b_delta, clash_delta], [1.0, 1.0, -1.0], method="linear")
        # 0.5 + 0.25 - 0.25 == 0.5
        self.assertTrue(torch.allclose(merged[prefix], torch.eye(16) * 0.5, atol=1e-5))

        lora_sd, stats = refactorize_deltas_to_lora(merged, out_rank=16, lora_alpha=16)
        self.assertLess(stats["mean_relative_svd_error"], 1e-4)
        recon = lora_sd[f"{prefix}.lora_B.weight"] @ lora_sd[f"{prefix}.lora_A.weight"]
        self.assertTrue(torch.allclose(recon, merged[prefix], atol=1e-4))

        diag = svd_diagnostics(a_delta, b_delta)
        self.assertAlmostEqual(diag["global_mean_cosine_similarity"], 1.0, places=5)


class TestPromptsAndNotebooks(unittest.TestCase):
    def test_prompts_and_benchmark(self) -> None:
        for trade in ("architecture", "mechanical", "plumbing", "electrical", "drafting", "structural", "fire_protection"):
            self.assertIn(trade, TRADE_TASKS)
            self.assertGreaterEqual(len(TRADE_TASKS[trade]), 1)
        self.assertEqual(len(MULTI_TRADE_BENCHMARK), 6)
        msgs = format_task_prompt(MULTI_TRADE_BENCHMARK[0], "kit docs")
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1]["role"], "user")
        self.assertIn("kit docs", msgs[0]["content"])

    def test_notebooks_valid_nbformat_and_ast(self) -> None:
        for nb_name in (
            "01_train_trade_lora.ipynb",
            "02_weight_space_mixer.ipynb",
            "03_vibe_architect_studio.ipynb",
        ):
            nb_path = REPO_ROOT / "notebooks" / nb_name
            nb = json.loads(nb_path.read_text(encoding="utf-8"))
            self.assertEqual(nb["nbformat"], 4)
            self.assertGreaterEqual(len(nb["cells"]), 6)
            for idx, cell in enumerate(nb["cells"]):
                if cell["cell_type"] == "code":
                    lines = []
                    for line in "".join(cell["source"]).splitlines():
                        stripped = line.lstrip()
                        indent = line[: len(line) - len(stripped)]
                        if stripped.startswith("!") or stripped.startswith("%"):
                            lines.append(f"{indent}pass  # colab magic")
                        else:
                            lines.append(line)
                    ast.parse("\n".join(lines), filename=f"{nb_name}:cell_{idx}")


class TestStudioRender(unittest.TestCase):
    def test_presets_and_prompt_builder(self) -> None:
        self.assertGreaterEqual(len(ART_DIRECTION_PRESETS), 5)
        self.assertIn("isometric_cutaway", CAMERA_PRESETS)
        self.assertEqual(len(VIBE_INSPIRATION_CHIPS), 4)
        pos, neg = build_render_prompt(
            "Cozy timber tea bar",
            "ArchDaily Golden Hour Timber",
            "isometric_cutaway",
        )
        self.assertIn("Cozy timber tea bar", pos)
        self.assertIn("isometric", pos)
        self.assertTrue(len(neg) > 10)


if __name__ == "__main__":
    unittest.main()
