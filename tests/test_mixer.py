"""Stdlib unit tests for mutinybim-mixer (no GPU or third-party packages required)."""

from __future__ import annotations

import ast
import inspect
import json
import math
import subprocess
import sys
import tempfile
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
from mixer.mcp_server import TOOLS, _validate_verifier_url, handle_jsonrpc_message
from mixer.prompts import MULTI_TRADE_BENCHMARK, TRADE_TASKS, format_task_prompt
from mixer.sketch_engine import (
    SKETCH_STYLES,
    SKETCH_TOPOLOGIES,
    generate_sketch_dataset_specs,
    generate_sketch_sample,
    render_parametric_sketch_svg,
)
from mixer.solar_engine import (
    CITY_PRESETS,
    KEY_DATES,
    _format_shadow_frame_label,
    _read_ifc_source,
    _select_shadow_casters,
    build_scalar_heatmap_overlay,
    build_shadow_mesh,
    compute_direct_sun_hours_for_points,
    compute_passive_solar_reward,
    describe_solar_lighting_for_camera,
    envelope_gates,
    project_shadow_triangles_pure,
    read_ifc_true_north_deg,
    render_sun_dial_svg,
    select_timelapse_solar_hours,
    solar_day_arc,
    solar_position,
)
from mixer.studio_render import (
    ART_DIRECTION_PRESETS,
    CAMERA_PRESETS,
    MIXTAPE_PRESETS,
    VIBE_INSPIRATION_CHIPS,
    _clip_near,
    _rasterize_camera_views,
    _resolve_space_boxes,
    _should_skip_ifc_class,
    build_render_prompt,
    compute_camera_pose,
    derive_rooms_from_walls,
    first_wall_hit_distance_m,
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
        self.assertEqual(spec["mutinybim"]["commit"], "fbf5f61424b64d9f99b1c896ca02328c88d9813c")
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
            "vision_floorplan",
            "passive_solar",
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
                    self._send(200, {"status": "ok", "mutinybim_commit": "fbf5f61424b6"})
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
            "04_sketch_to_bim_lora.ipynb",
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
        self.assertEqual(len(CAMERA_PRESETS), 4)
        self.assertIn("isometric_cutaway", CAMERA_PRESETS)
        self.assertIn("opposite_interior", CAMERA_PRESETS)
        self.assertGreaterEqual(len(MIXTAPE_PRESETS), 3)
        for preset in MIXTAPE_PRESETS.values():
            self.assertIn(preset["mode"], ("style_trio", "weight_sweep"))
            self.assertEqual(len(preset["tracks"]), 3)
        self.assertEqual(len(VIBE_INSPIRATION_CHIPS), 4)
        for cam_key in CAMERA_PRESETS:
            pos, neg = build_render_prompt(
                "Cozy timber tea bar",
                "ArchDaily Golden Hour Timber",
                cam_key,
            )
            self.assertIn("Cozy timber tea bar", pos)
            self.assertTrue(len(neg) > 10)

    def test_clip_near_sutherland_hodgman(self) -> None:
        # All behind -> empty
        self.assertEqual(_clip_near([(0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.04)], near=0.05), [])
        # All in front -> unchanged 3 vertices
        front = [(0.0, 0.0, 0.1), (1.0, 0.0, 0.2), (0.0, 1.0, 0.3)]
        self.assertEqual(len(_clip_near(front, near=0.05)), 3)
        # 2 vertices behind, 1 in front -> clipped triangle (3 vertices at z >= 0.05)
        one_in = _clip_near([(0.0, 0.0, 1.05), (2.0, 0.0, -0.95), (0.0, 2.0, -0.95)], near=0.05)
        self.assertEqual(len(one_in), 3)
        self.assertTrue(all(p[2] >= 0.05 - 1e-9 for p in one_in))
        self.assertAlmostEqual(one_in[1][0], 1.0, places=5)
        # 1 vertex behind, 2 in front -> clipped quad (4 vertices at z >= 0.05)
        two_in = _clip_near([(0.0, 0.0, -0.95), (2.0, 0.0, 1.05), (0.0, 2.0, 1.05)], near=0.05)
        self.assertEqual(len(two_in), 4)
        self.assertTrue(all(p[2] >= 0.05 - 1e-9 for p in two_in))

    def test_camera_pose_eye_level_and_constant_turntable_scale(self) -> None:
        vmin = (-6.0, -4.0, 0.0)
        vmax = (6.0, 4.0, 6.4)
        for cam_key in ("eye_level_interior", "opposite_interior"):
            pose_gnd = compute_camera_pose(vmin, vmax, cam_key, storey_elevations_m=[0.0, 3.2], storey_index=0)
            self.assertAlmostEqual(pose_gnd["eye"][2], 1.6, places=4)
            self.assertAlmostEqual(pose_gnd["target"][2], 1.6, places=4)
            pose_upper = compute_camera_pose(vmin, vmax, cam_key, storey_elevations_m=[0.0, 3.2], storey_index=1)
            self.assertAlmostEqual(pose_upper["eye"][2], 4.8, places=4)

        scales = [
            compute_camera_pose(
                vmin,
                vmax,
                "isometric_cutaway",
                width=512,
                height=384,
                azimuth_override=i * 20.0,
                elevation_override=34.0,
            )["px_scale"]
            for i in range(18)
        ]
        self.assertAlmostEqual(min(scales), max(scales), places=9)

    def test_closed_box_interior_no_holes(self) -> None:
        try:
            import numpy as np  # type: ignore
            import trimesh  # type: ignore
        except ImportError:
            self.skipTest("numpy/trimesh required for full rasterizer room test")

        # Build a sealed 10m x 8m x 3m room from 6 box slabs (walls, floor, ceiling).
        panels = [
            trimesh.creation.box(extents=[10.2, 8.2, 0.2], transform=trimesh.transformations.translation_matrix([0, 0, -0.1])),
            trimesh.creation.box(extents=[10.2, 8.2, 0.2], transform=trimesh.transformations.translation_matrix([0, 0, 3.1])),
            trimesh.creation.box(extents=[10.2, 0.2, 3.0], transform=trimesh.transformations.translation_matrix([0, -4.1, 1.5])),
            trimesh.creation.box(extents=[10.2, 0.2, 3.0], transform=trimesh.transformations.translation_matrix([0, 4.1, 1.5])),
            trimesh.creation.box(extents=[0.2, 8.0, 3.0], transform=trimesh.transformations.translation_matrix([-5.1, 0, 1.5])),
            trimesh.creation.box(extents=[0.2, 8.0, 3.0], transform=trimesh.transformations.translation_matrix([5.1, 0, 1.5])),
        ]
        meshes = [(m, (220, 220, 220, 255)) for m in panels]
        for cam_key in ("eye_level_interior", "opposite_interior"):
            depth_pil, _ = _rasterize_camera_views(
                meshes,
                camera_preset=cam_key,
                width=160,
                height=120,
                storey_elevations_m=[0.0],
            )
            arr = np.asarray(depth_pil)
            self.assertEqual(int(np.sum(arr == 0)), 0, f"Background hole found in {cam_key}")

    def test_skip_ifc_classes_and_room_diagonal_interior_cameras(self) -> None:
        for cls_name in (
            "IfcOpeningElement",
            "IfcOpeningStandardCase",
            "IfcSpace",
            "IfcVirtualElement",
            "IfcAnnotation",
            "IfcGrid",
        ):
            self.assertTrue(_should_skip_ifc_class(cls_name), cls_name)
        for cls_name in ("IfcWall", "IfcWallStandardCase", "IfcDoor", "IfcWindow", "IfcSlab", "IfcRoof", "IfcSpaceHeater"):
            self.assertFalse(_should_skip_ifc_class(cls_name), cls_name)

        spaces = [
            {"xmin": 0.0, "ymin": 0.0, "xmax": 7.0, "ymax": 9.0, "zmin": 0.0},
            {"xmin": 7.0, "ymin": 0.0, "xmax": 12.0, "ymax": 4.5, "zmin": 0.0},
        ]
        p_eye = compute_camera_pose(
            (0.0, 0.0, 0.0),
            (12.0, 9.0, 7.0),
            "eye_level_interior",
            storey_elevations_m=[0.0, 3.6],
            storey_index=0,
            space_boxes=spaces,
        )
        p_opp = compute_camera_pose(
            (0.0, 0.0, 0.0),
            (12.0, 9.0, 7.0),
            "opposite_interior",
            storey_elevations_m=[0.0, 3.6],
            storey_index=0,
            space_boxes=spaces,
        )
        # Both cameras sit strictly inside the largest room (0..7 x 0..9) at 1.6 m eye height, looking diagonally across it.
        self.assertGreater(p_eye["eye"][0], 0.0)
        self.assertLess(p_eye["eye"][0], 3.5)
        self.assertAlmostEqual(p_eye["eye"][2], 1.6, places=4)
        self.assertGreater(p_opp["eye"][0], 3.5)
        self.assertLess(p_opp["eye"][0], 7.0)
        self.assertAlmostEqual(p_opp["eye"][2], 1.6, places=4)
        self.assertAlmostEqual(p_eye["eye"][0], p_opp["target"][0], places=4)
        self.assertAlmostEqual(p_opp["eye"][0], p_eye["target"][0], places=4)

    def test_derive_rooms_l_shaped_and_forward_ray_guard(self) -> None:
        # L-shaped exterior loop: notch cuts out x in (5, 10), y in (5, 9) so bbox diagonal crosses outside!
        l_walls = [
            {"id": "w1", "start": (0.0, 0.0), "end": (10.0, 0.0), "base_z_m": 0.0, "height_m": 3.0, "thickness_m": 0.2},
            {"id": "w2", "start": (10.0, 0.0), "end": (10.0, 5.0), "base_z_m": 0.0, "height_m": 3.0, "thickness_m": 0.2},
            {"id": "w3", "start": (10.0, 5.0), "end": (5.0, 5.0), "base_z_m": 0.0, "height_m": 3.0, "thickness_m": 0.2},
            {"id": "w4", "start": (5.0, 5.0), "end": (5.0, 9.0), "base_z_m": 0.0, "height_m": 3.0, "thickness_m": 0.2},
            {"id": "w5", "start": (5.0, 9.0), "end": (0.0, 9.0), "base_z_m": 0.0, "height_m": 3.0, "thickness_m": 0.2},
            {"id": "w6", "start": (0.0, 9.0), "end": (0.0, 0.0), "base_z_m": 0.0, "height_m": 3.0, "thickness_m": 0.2},
        ]
        rooms = derive_rooms_from_walls(l_walls)
        self.assertEqual(len(rooms), 1)
        pose = compute_camera_pose(
            (0.0, 0.0, 0.0),
            (10.0, 9.0, 3.0),
            "eye_level_interior",
            storey_elevations_m=[0.0],
            space_boxes=rooms,
            wall_segments=l_walls,
        )
        self.assertFalse(pose["camera_fallback"])
        hit_m = first_wall_hit_distance_m(pose["eye"], pose["forward"], l_walls)
        self.assertIsNotNone(hit_m)
        self.assertGreaterEqual(hit_m, 3.0)

        # Forward-ray guard rejects a close-wall candidate or flags fallback when all views < min_clearance_m.
        self.assertGreaterEqual(
            inspect.signature(compute_camera_pose).parameters["min_clearance_m"].default,
            3.0,
        )
        tiny_box = [{"xmin": 0.0, "ymin": 0.0, "xmax": 2.0, "ymax": 2.0, "zmin": 0.0}]
        tiny_walls = [
            {"start": (0.0, 0.0), "end": (2.0, 0.0), "base_z_m": 0.0, "height_m": 3.0, "thickness_m": 0.2},
            {"start": (2.0, 0.0), "end": (2.0, 2.0), "base_z_m": 0.0, "height_m": 3.0, "thickness_m": 0.2},
            {"start": (2.0, 2.0), "end": (0.0, 2.0), "base_z_m": 0.0, "height_m": 3.0, "thickness_m": 0.2},
            {"start": (0.0, 2.0), "end": (0.0, 0.0), "base_z_m": 0.0, "height_m": 3.0, "thickness_m": 0.2},
        ]
        # Must trigger using the default min_clearance_m (M3 mutation guard).
        tiny_pose = compute_camera_pose(
            (0.0, 0.0, 0.0),
            (2.0, 2.0, 3.0),
            "eye_level_interior",
            space_boxes=tiny_box,
            wall_segments=tiny_walls,
        )
        self.assertTrue(tiny_pose["camera_fallback"])
        self.assertEqual(tiny_pose["fallback_reason"], "nearest_wall_under_min_clearance")

        # Level 2 zmin stays exact (e.g. 12 ft = 3.6576 m), not rounded to 3.5 m.
        l2_walls = [
            {"start": (0.0, 0.0), "end": (6.0, 0.0), "base_z_m": 3.6576, "height_m": 2.7432, "thickness_m": 0.2},
            {"start": (6.0, 0.0), "end": (6.0, 5.0), "base_z_m": 3.6576, "height_m": 2.7432, "thickness_m": 0.2},
            {"start": (6.0, 5.0), "end": (0.0, 5.0), "base_z_m": 3.6576, "height_m": 2.7432, "thickness_m": 0.2},
            {"start": (0.0, 5.0), "end": (0.0, 0.0), "base_z_m": 3.6576, "height_m": 2.7432, "thickness_m": 0.2},
        ]
        l2_rooms = derive_rooms_from_walls(l2_walls)
        self.assertEqual(len(l2_rooms), 1)
        self.assertAlmostEqual(l2_rooms[0]["zmin"], 3.6576, places=4)

        empty_pose = compute_camera_pose((0.0, 0.0, 0.0), (10.0, 8.0, 3.0), "eye_level_interior", space_boxes=[])
        self.assertTrue(empty_pose["camera_fallback"])
        self.assertEqual(empty_pose["fallback_reason"], "no_rooms_detected")


class TestSolarEngine(unittest.TestCase):
    def test_solar_position_hemispheres_and_equinox(self) -> None:
        self.assertGreaterEqual(len(CITY_PRESETS), 8)
        self.assertEqual(len(KEY_DATES), 4)
        sf_summer = solar_position(37.7749, 172, 12.0)
        sf_winter = solar_position(37.7749, 355, 12.0)
        self.assertGreater(sf_summer.altitude_deg, sf_winter.altitude_deg + 40.0)
        self.assertAlmostEqual(sf_summer.azimuth_deg, 180.0, delta=1.5)

        sydney_noon = solar_position(-33.8688, 172, 12.0)
        self.assertTrue(sydney_noon.azimuth_deg < 2.0 or sydney_noon.azimuth_deg > 358.0)
        mag = sum(c * c for c in sf_summer.sun_vec) ** 0.5
        self.assertAlmostEqual(mag, 1.0, places=6)

        _, sunrise_eq, sunset_eq = solar_day_arc(37.7749, 80, step_minutes=30)
        self.assertAlmostEqual(sunrise_eq, 6.0, delta=0.25)
        self.assertAlmostEqual(sunset_eq, 18.0, delta=0.25)

    def test_sun_dial_svg_and_heatmap_schema(self) -> None:
        svg = render_sun_dial_svg(
            37.7749,
            80,
            15.0,
            city_label="San Francisco",
            date_label="Spring Equinox",
        )
        self.assertTrue(svg.startswith("<svg"))
        self.assertTrue(svg.endswith("</svg>"))
        self.assertIn("1 m pole shadow:", svg)
        self.assertIn("Sunrise", svg)

        overlay = build_scalar_heatmap_overlay(
            [{"xMm": 0, "yMm": 0, "zMm": 850, "widthMm": 1000, "depthMm": 1000, "value": 6.5}]
        )
        self.assertEqual(overlay["palette"], "thermal")
        self.assertEqual(overlay["cells"][0]["zMm"], 850)
        self.assertGreater(
            compute_passive_solar_reward(
                4.5, 5.0, verify_ok=True, spec_match=1.0, has_closed_envelope=True, max_facade_wwr=0.30
            ),
            compute_passive_solar_reward(
                1.0, 9.0, verify_ok=True, spec_match=1.0, has_closed_envelope=True, max_facade_wwr=0.30
            ),
        )

    def test_polar_day_night_and_horizon_shadows(self) -> None:
        # Svalbard (78.22°N): Jun 21 polar day, Dec 21 polar night -> no math domain errors.
        pos_day, sr_day, ss_day = solar_day_arc(78.22, 172, step_minutes=60)
        self.assertEqual(sr_day, 0.0)
        self.assertEqual(ss_day, 24.0)
        self.assertGreater(len(pos_day), 20)

        pos_night, sr_night, ss_night = solar_day_arc(78.22, 355, step_minutes=60)
        self.assertEqual(sr_night, ss_night)
        self.assertEqual(pos_night, [])

        # Horizon / night returns None instead of negative or huge shadow lengths.
        midnight = solar_position(37.7749, 80, 0.0)
        self.assertLessEqual(midnight.altitude_deg, 0.0)
        self.assertIsNone(midnight.shadow_len_1m)
        noon = solar_position(37.7749, 80, 12.0)
        self.assertIsNotNone(noon.shadow_len_1m)
        self.assertGreater(noon.shadow_len_1m, 0.0)

    def test_read_ifc_true_north_deg(self) -> None:
        ifc_30 = (
            "ISO-10303-21;\nDATA;\n"
            "#3=IFCAXIS2PLACEMENT3D(#1,$,$);\n"
            "#4=IFCDIRECTION((0.5,0.8660254));\n"
            "#5=IFCGEOMETRICREPRESENTATIONCONTEXT($,'Model',3,1.E-5,#3,#4);\n"
            "ENDSEC;\n"
        )
        self.assertAlmostEqual(read_ifc_true_north_deg(ifc_30), 30.0, places=2)

        ifc_neg45 = ifc_30.replace("0.5,0.8660254", "-0.70710678,0.70710678")
        self.assertAlmostEqual(read_ifc_true_north_deg(ifc_neg45), -45.0, places=2)

        ifc_missing = ifc_30.replace("#4)", "$)")
        self.assertEqual(read_ifc_true_north_deg(ifc_missing), 0.0)

    def test_sun_hours_physics_equinox_closed_box_wall_and_rotated_site(self) -> None:
        # 1. Unobstructed equinox in both N and S hemispheres gets ~12 hours.
        for lat in (37.7749, -33.8688):
            hrs = compute_direct_sun_hours_for_points(
                [(0.0, 0.0, 0.0)], [], lat_deg=lat, day_of_year=80, step_minutes=15
            )[0]
            self.assertGreaterEqual(hrs, 11.25)
            self.assertLessEqual(hrs, 12.25)

        def _quad(a, b, c, d):
            return [(a, b, c), (a, c, d)]

        # 2. Point inside a closed roof + 4 walls box gets 0 hours.
        closed_box = (
            _quad((-4, -4, 3), (4, -4, 3), (4, 4, 3), (-4, 4, 3))
            + _quad((-4, -4, 0), (4, -4, 0), (4, -4, 3), (-4, -4, 3))
            + _quad((4, -4, 0), (4, 4, 0), (4, 4, 3), (4, -4, 3))
            + _quad((4, 4, 0), (-4, 4, 0), (-4, 4, 3), (4, 4, 3))
            + _quad((-4, 4, 0), (-4, -4, 0), (-4, -4, 3), (-4, 4, 3))
        )
        self.assertEqual(
            compute_direct_sun_hours_for_points(
                [(0.0, 0.0, 0.85)], closed_box, lat_deg=40.0, day_of_year=172, step_minutes=30
            )[0],
            0.0,
        )

        # 3. Tall east-west wall at y=0 (x in [-10,10], z in [0,8]) in winter (N hemisphere):
        #    north side (y=+2) is shaded, south side (y=-2) receives direct sun.
        ew_wall = _quad((-10, 0, 0), (10, 0, 0), (10, 0, 8), (-10, 0, 8))
        h_north, h_south = compute_direct_sun_hours_for_points(
            [(0.0, 2.0, 0.85), (0.0, -2.0, 0.85)],
            ew_wall,
            lat_deg=40.0,
            day_of_year=355,
            true_north_deg=0.0,
            step_minutes=30,
        )
        self.assertGreater(h_south, h_north + 4.0)

        # 4. Same geometry rotated by true_north_deg = +90° (True North is plan +X, True South is plan -X):
        #    tall north-south wall at x=0 (y in [-10,10], z in [0,8]) shades +X side vs -X side.
        ns_wall = _quad((0, -10, 0), (0, 10, 0), (0, 10, 8), (0, -10, 8))
        h_tnorth, h_tsouth = compute_direct_sun_hours_for_points(
            [(2.0, 0.0, 0.85), (-2.0, 0.0, 0.85)],
            ns_wall,
            lat_deg=40.0,
            day_of_year=355,
            true_north_deg=90.0,
            step_minutes=30,
        )
        self.assertGreater(h_tsouth, h_tnorth + 4.0)

    def test_shadow_projection_pure_direction_window_stability_mutation(self) -> None:
        def _box_tris(x0: float, y0: float, z0: float, x1: float, y1: float, z1: float) -> list[Any]:
            v = [
                (x0, y0, z0),
                (x1, y0, z0),
                (x1, y1, z0),
                (x0, y1, z0),
                (x0, y0, z1),
                (x1, y0, z1),
                (x1, y1, z1),
                (x0, y1, z1),
            ]
            idx = [
                (0, 2, 1), (0, 3, 2),  # bottom
                (4, 5, 6), (4, 6, 7),  # top
                (0, 1, 5), (0, 5, 4),  # -Y
                (1, 2, 6), (1, 6, 5),  # +X
                (2, 3, 7), (2, 7, 6),  # +Y
                (3, 0, 4), (3, 4, 7),  # -X
            ]
            return [(v[a], v[b], v[c]) for a, b, c in idx]

        def _point_in_tri_2d(px: float, py: float, tri: Any) -> bool:
            a, b, c = tri
            d1 = (px - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (py - b[1])
            d2 = (px - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (py - c[1])
            d3 = (px - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (py - a[1])
            has_neg = (d1 < -1e-6) or (d2 < -1e-6) or (d3 < -1e-6)
            has_pos = (d1 > 1e-6) or (d2 > 1e-6) or (d3 > 1e-6)
            return not (has_neg and has_pos)

        def _covered_by_shadow(px: float, py: float, tris: list[Any]) -> bool:
            return any(_point_in_tri_2d(px, py, tri) for tri in tris)

        def _assert_box_shadow_matches(
            shadow_tris: list[Any],
            sun_vec: tuple[float, float, float],
            height_m: float = 3.0,
        ) -> None:
            self.assertGreater(len(shadow_tris), 0, "Shadow triangles missing")
            sx, sy, sz = sun_vec
            horiz = math.hypot(sx, sy)
            ux, uy = -sx / horiz, -sy / horiz
            max_top_proj = max(x * ux + y * uy for x in (0.0, 1.0) for y in (0.0, 1.0))
            pts = [pt for tri in shadow_tris for pt in tri]
            for a, b, c in shadow_tris:
                nz = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                self.assertGreater(nz, 0.0)
            far_edge = max(pt[0] * ux + pt[1] * uy for pt in pts) - max_top_proj
            expected_len = height_m * horiz / sz  # height / tan(alt)
            self.assertAlmostEqual(far_edge, expected_len, delta=0.15)
            # Shadow centroid also moves strictly in -sun_horizontal direction from box center (0.5, 0.5).
            cx = sum(pt[0] for pt in pts) / len(pts) - 0.5
            cy = sum(pt[1] for pt in pts) / len(pts) - 0.5
            self.assertGreater(cx * ux + cy * uy, 0.5)

        # 1. Shadow direction & length for a 1 x 1 x 3 m box (and ensure floor slab at z<=0 casts nothing).
        box_tris = _box_tris(0.0, 0.0, 0.0, 1.0, 1.0, 3.0)
        slab_tris = _box_tris(-2.0, -2.0, -0.2, 3.0, 3.0, 0.0)
        ground_rect = (-12.0, -12.0, 13.0, 13.0)
        pos_15 = solar_position(51.5, 80, 15.0)
        self.assertEqual(project_shadow_triangles_pure(slab_tris, pos_15.sun_vec, 0.0, ground_rect), [])
        box_shadows = project_shadow_triangles_pure(box_tris, pos_15.sun_vec, 0.0, ground_rect)
        _assert_box_shadow_matches(box_shadows, pos_15.sun_vec, height_m=3.0)

        # 2. Window opening light: solid piers shade floor behind them; window opening lets sunlight through.
        wall_with_window = (
            _box_tris(0.0, -0.1, 0.0, 2.0, 0.1, 3.0)   # left pier
            + _box_tris(4.0, -0.1, 0.0, 6.0, 0.1, 3.0)  # right pier
            + _box_tris(2.0, -0.1, 0.0, 4.0, 0.1, 0.9)  # sill
            + _box_tris(2.0, -0.1, 2.4, 4.0, 0.1, 3.0)  # lintel
        )
        sun_south_45 = (0.0, -math.cos(math.pi / 4.0), math.sin(math.pi / 4.0))
        win_shadows = project_shadow_triangles_pure(wall_with_window, sun_south_45, 0.0, (-7.0, -7.0, 13.0, 13.0))
        self.assertFalse(_covered_by_shadow(3.0, 1.65, win_shadows))
        self.assertTrue(_covered_by_shadow(1.0, 1.65, win_shadows))
        self.assertTrue(_covered_by_shadow(5.0, 1.65, win_shadows))

        # 3. Frame stability: px_scale identical across every frame of a day sweep with padded ground bounds.
        sweep, _, _ = solar_day_arc(51.5, 80, step_minutes=60)
        sweep = [p for p in sweep if p.altitude_deg >= 8.0]
        g_pad = (-7.0, -7.0, 8.0, 8.0)
        scales = []
        for pos in sweep:
            s_tris = project_shadow_triangles_pure(box_tris, pos.sun_vec, 0.0, g_pad)
            all_pts = [
                (g_pad[0], g_pad[1], -0.03),
                (g_pad[2], g_pad[3], -0.03),
                *[pt for tri in box_tris for pt in tri],
                *[pt for tri in s_tris for pt in tri],
            ]
            vmin = (min(p[0] for p in all_pts), min(p[1] for p in all_pts), min(p[2] for p in all_pts))
            vmax = (max(p[0] for p in all_pts), max(p[1] for p in all_pts), max(p[2] for p in all_pts))
            scales.append(compute_camera_pose(vmin, vmax, "isometric_cutaway", width=512, height=384)["px_scale"])
        self.assertGreater(len(scales), 5)
        self.assertAlmostEqual(min(scales), max(scales), places=9)

        # 4. Mutation check: empty shadow list must fail the direction & length assertion.
        with self.assertRaises(AssertionError):
            _assert_box_shadow_matches([], pos_15.sun_vec, height_m=3.0)

    def test_shadow_mesh_numpy_trimesh_direction_window_stability_mutation(self) -> None:
        try:
            import numpy as np  # type: ignore
            import trimesh  # type: ignore
        except ImportError:
            self.skipTest("numpy/trimesh required for Trimesh shadow test")

        def _box_mesh(x0: float, y0: float, z0: float, x1: float, y1: float, z1: float) -> Any:
            return trimesh.creation.box(
                extents=[x1 - x0, y1 - y0, z1 - z0],
                transform=trimesh.transformations.translation_matrix([0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1)]),
            )

        def _point_in_shadow_mesh(px: float, py: float, mesh: Any) -> bool:
            if len(mesh.faces) == 0:
                return False
            tris = mesh.vertices[mesh.faces]
            for a, b, c in tris:
                d1 = (px - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (py - b[1])
                d2 = (px - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (py - c[1])
                d3 = (px - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (py - a[1])
                if not (((d1 < -1e-6) or (d2 < -1e-6) or (d3 < -1e-6)) and ((d1 > 1e-6) or (d2 > 1e-6) or (d3 > 1e-6))):
                    return True
            return False

        def _assert_mesh_shadow_matches(mesh: Any, sun_vec: tuple[float, float, float], height_m: float = 3.0) -> None:
            self.assertGreater(len(mesh.faces), 0, "Empty shadow mesh")
            sx, sy, sz = sun_vec
            horiz = math.hypot(sx, sy)
            ux, uy = -sx / horiz, -sy / horiz
            max_top_proj = max(x * ux + y * uy for x in (0.0, 1.0) for y in (0.0, 1.0))
            pts = mesh.vertices
            far_edge = float((pts[:, 0] * ux + pts[:, 1] * uy).max()) - max_top_proj
            self.assertAlmostEqual(far_edge, height_m * horiz / sz, delta=0.15)
            self.assertTrue(np.all(mesh.face_normals[:, 2] > 0.99))

        box = _box_mesh(0.0, 0.0, 0.0, 1.0, 1.0, 3.0)
        box_tris = box.vertices[box.faces]
        pos_15 = solar_position(51.5, 80, 15.0)
        s_mesh = build_shadow_mesh(box_tris, pos_15.sun_vec, 0.0, (-12.0, -12.0, 13.0, 13.0))
        _assert_mesh_shadow_matches(s_mesh, pos_15.sun_vec, height_m=3.0)

        wall_parts = [
            _box_mesh(0.0, -0.1, 0.0, 2.0, 0.1, 3.0),
            _box_mesh(4.0, -0.1, 0.0, 6.0, 0.1, 3.0),
            _box_mesh(2.0, -0.1, 0.0, 4.0, 0.1, 0.9),
            _box_mesh(2.0, -0.1, 2.4, 4.0, 0.1, 3.0),
        ]
        wall_tris = np.concatenate([m.vertices[m.faces] for m in wall_parts], axis=0)
        sun_south_45 = (0.0, -math.cos(math.pi / 4.0), math.sin(math.pi / 4.0))
        win_mesh = build_shadow_mesh(wall_tris, sun_south_45, 0.0, (-7.0, -7.0, 13.0, 13.0))
        self.assertFalse(_point_in_shadow_mesh(3.0, 1.65, win_mesh))
        self.assertTrue(_point_in_shadow_mesh(1.0, 1.65, win_mesh))

        sweep, _, _ = solar_day_arc(51.5, 80, step_minutes=60)
        sweep = [p for p in sweep if p.altitude_deg >= 8.0]
        g_pad = (-7.0, -7.0, 8.0, 8.0)
        ground_mesh = trimesh.Trimesh(
            np.array([[g_pad[0], g_pad[1], -0.03], [g_pad[2], g_pad[1], -0.03], [g_pad[2], g_pad[3], -0.03], [g_pad[0], g_pad[3], -0.03]]),
            np.array([[0, 1, 2], [0, 2, 3]]),
            process=False,
        )
        scales = []
        for pos in sweep:
            sm = build_shadow_mesh(box_tris, pos.sun_vec, 0.0, g_pad)
            verts = np.vstack([ground_mesh.vertices, box.vertices, sm.vertices]) if len(sm.vertices) else np.vstack([ground_mesh.vertices, box.vertices])
            scales.append(
                compute_camera_pose(verts.min(0).tolist(), verts.max(0).tolist(), "isometric_cutaway", width=512, height=384)["px_scale"]
            )
        self.assertAlmostEqual(min(scales), max(scales), places=9)

        empty_mesh = trimesh.Trimesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), process=False)
        with self.assertRaises(AssertionError):
            _assert_mesh_shadow_matches(empty_mesh, pos_15.sun_vec, height_m=3.0)

    def test_describe_solar_lighting_for_camera_prompt_and_timelapse_hours(self) -> None:
        pose = compute_camera_pose((0.0, 0.0, 0.0), (12.0, 8.0, 3.5), "isometric_cutaway")
        right = pose["right"]
        # Synthetic sun along camera +right at golden hour altitude (10 deg) -> warm sun, cool shadow, screen-right.
        alt10 = math.radians(10.0)
        sun_right_pos = solar_position(51.5, 80, 17.0)
        # Construct exact SolarPosition matching +right horizontal + sin(10 deg) up:
        from mixer.solar_engine import SolarPosition
        pos_right = SolarPosition(
            hour_solar=17.0,
            altitude_deg=10.0,
            azimuth_deg=260.0,
            shadow_len_1m=round(1.0 / math.tan(alt10), 2),
            sun_vec=(right[0] * math.cos(alt10), right[1] * math.cos(alt10), math.sin(alt10)),
        )
        desc_right = describe_solar_lighting_for_camera(pos_right, pose)
        self.assertEqual(desc_right["phase"], "golden_hour")
        self.assertGreater(desc_right["sun_cam"][0], 0.5)
        self.assertEqual(desc_right["screen_dir"], "upper-right")
        self.assertIn("from screen-right", desc_right["prompt_cue"])
        self.assertGreater(desc_right["sun_tint_rgb"][0], desc_right["sun_tint_rgb"][2])
        self.assertGreater(desc_right["shadow_tint_rgb"][2], desc_right["shadow_tint_rgb"][0])

        pos_left = SolarPosition(
            hour_solar=8.0,
            altitude_deg=22.0,
            azimuth_deg=100.0,
            shadow_len_1m=2.48,
            sun_vec=(-right[0] * math.cos(math.radians(22.0)), -right[1] * math.cos(math.radians(22.0)), math.sin(math.radians(22.0))),
        )
        desc_left = describe_solar_lighting_for_camera(pos_left, pose)
        self.assertEqual(desc_left["phase"], "low_sun")
        self.assertIn("from screen-left", desc_left["prompt_cue"])

        pos_night = SolarPosition(
            hour_solar=21.0,
            altitude_deg=-5.0,
            azimuth_deg=310.0,
            shadow_len_1m=None,
            sun_vec=sun_right_pos.sun_vec,
        )
        desc_night = describe_solar_lighting_for_camera(pos_night, pose)
        self.assertEqual(desc_night["phase"], "night")

        base_pos, base_neg = build_render_prompt("Cedar atrium", "ArchDaily Golden Hour Timber", "isometric_cutaway")
        sol_pos, sol_neg = build_render_prompt(
            "Cedar atrium",
            "ArchDaily Golden Hour Timber",
            "isometric_cutaway",
            solar_lighting_cue=desc_right["prompt_cue"],
            solar_negative_cue=desc_right["negative_cue"],
        )
        self.assertNotIn("from screen-right", base_pos)
        self.assertIn(desc_right["prompt_cue"], sol_pos)
        self.assertIn(desc_right["negative_cue"], sol_neg)
        self.assertNotIn("contradictory shadow direction", base_neg)

        hrs = select_timelapse_solar_hours(51.5, 80, num_frames=4, min_altitude_deg=8.0)
        self.assertEqual(len(hrs), 4)
        self.assertEqual(hrs, sorted(hrs))
        for h in hrs:
            self.assertGreaterEqual(solar_position(51.5, 80, h).altitude_deg, 8.0)
        self.assertEqual(select_timelapse_solar_hours(78.22, 355, num_frames=4), [12.0])
        self.assertEqual(len(select_timelapse_solar_hours(78.22, 172, num_frames=4)), 4)

    def test_ifc_source_path_vs_text_and_shadow_caster_flags(self) -> None:
        # Missing str .ifc path must raise FileNotFoundError instead of being parsed as IFC text.
        with self.assertRaises(FileNotFoundError):
            _read_ifc_source("/tmp/nonexistent_mutinybim_dir_999/model.ifc")
        with self.assertRaises(FileNotFoundError):
            read_ifc_true_north_deg("/tmp/nonexistent_mutinybim_dir_999/model.ifc")

        # Valid .ifc temp str path round-trips true north, _resolve_space_boxes, and envelope_gates.
        step_ifc = """ISO-10303-21;
DATA;
#1=IFCSIUNIT(*,.LENGTHUNIT.,.MILLI.,.METRE.);
#2=IFCCARTESIANPOINT((0.,0.,0.));
#3=IFCAXIS2PLACEMENT3D(#2,$,$);
#4=IFCDIRECTION((-0.5,0.8660254));
#5=IFCGEOMETRICREPRESENTATIONCONTEXT($,'Model',3,1.E-5,#3,#4);
#10=IFCLOCALPLACEMENT($,#3);
#20=IFCCARTESIANPOINT((3000.,0.,0.));
#21=IFCAXIS2PLACEMENT2D(#20,$);
#22=IFCRECTANGLEPROFILEDEF(.AREA.,$,#21,6000.,200.);
#23=IFCEXTRUDEDAREASOLID(#22,#3,#4,3000.);
#24=IFCSHAPEREPRESENTATION($,'Body','SweptSolid',(#23));
#25=IFCPRODUCTDEFINITIONSHAPE($,$,(#24));
#30=IFCCARTESIANPOINT((0.,0.,0.));
#31=IFCDIRECTION((0.,0.,1.));
#32=IFCDIRECTION((1.,0.,0.));
#33=IFCAXIS2PLACEMENT3D(#30,#31,#32);
#34=IFCLOCALPLACEMENT(#10,#33);
#35=IFCWALL('w1',$,'Wall',$,$,#34,#25,$,$);
#40=IFCCARTESIANPOINT((6000.,0.,0.));
#41=IFCDIRECTION((0.,1.,0.));
#42=IFCAXIS2PLACEMENT3D(#40,#31,#41);
#43=IFCLOCALPLACEMENT(#10,#42);
#44=IFCCARTESIANPOINT((2500.,0.,0.));
#45=IFCAXIS2PLACEMENT2D(#44,$);
#46=IFCRECTANGLEPROFILEDEF(.AREA.,$,#45,5000.,200.);
#47=IFCEXTRUDEDAREASOLID(#46,#3,#31,3000.);
#48=IFCSHAPEREPRESENTATION($,'Body','SweptSolid',(#47));
#49=IFCPRODUCTDEFINITIONSHAPE($,$,(#48));
#50=IFCWALL('w2',$,'Wall',$,$,#43,#49,$,$);
#60=IFCCARTESIANPOINT((6000.,5000.,0.));
#61=IFCDIRECTION((-1.,0.,0.));
#62=IFCAXIS2PLACEMENT3D(#60,#31,#61);
#63=IFCLOCALPLACEMENT(#10,#62);
#64=IFCWALL('w3',$,'Wall',$,$,#63,#25,$,$);
#70=IFCCARTESIANPOINT((0.,5000.,0.));
#71=IFCDIRECTION((0.,-1.,0.));
#72=IFCAXIS2PLACEMENT3D(#70,#31,#71);
#73=IFCLOCALPLACEMENT(#10,#72);
#74=IFCWALL('w4',$,'Wall',$,$,#73,#49,$,$);
ENDSEC;
"""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "model.ifc"
            p.write_text(step_ifc, encoding="utf-8")
            p_str = str(p)
            self.assertAlmostEqual(read_ifc_true_north_deg(p_str), -30.0, places=2)
            self.assertAlmostEqual(read_ifc_true_north_deg(p), -30.0, places=2)
            rooms_str = _resolve_space_boxes([], p_str)
            self.assertEqual(len(rooms_str), 1)
            explicit = [{"xmin": 1.0, "ymin": 1.0, "xmax": 4.0, "ymax": 4.0, "zmin": 0.0}]
            self.assertEqual(_resolve_space_boxes(explicit, p_str), explicit)

        # M4 & M8 guards: window excluded from casters, roof_assumed flagged & labeled.
        class _DummyMesh:
            def __init__(self, verts, n_faces=2):
                self.vertices = verts
                self.faces = [(0, 1, 2)] * n_faces

        wall_m = _DummyMesh([(0.0, 0.0, 0.0), (6.0, 5.0, 3.0)])
        win_m = _DummyMesh([(2.0, 0.0, 0.9), (3.2, 0.2, 2.4)])
        floor_slab_m = _DummyMesh([(0.0, 0.0, -0.2), (6.0, 5.0, 0.0)])
        roof_m = _DummyMesh([(0.0, 0.0, 3.0), (6.0, 5.0, 3.2)])
        top_slab_m = _DummyMesh([(0.0, 0.0, 2.8), (6.0, 5.0, 3.0)])

        casters_no_roof, roof_assumed_no_roof, roof_spec = _select_shadow_casters(
            [("IfcWallStandardCase", wall_m, ()), ("IfcWindow", win_m, ()), ("IfcSlab", floor_slab_m, ())],
            0.0,
        )
        self.assertIn(wall_m, casters_no_roof)
        self.assertIn(floor_slab_m, casters_no_roof)
        self.assertNotIn(win_m, casters_no_roof)
        self.assertTrue(roof_assumed_no_roof)
        self.assertIsNotNone(roof_spec)
        self.assertAlmostEqual(roof_spec[4], 3.0, places=4)

        pos = solar_position(51.5, 80, 12.0)
        self.assertIn("roof assumed", _format_shadow_frame_label(pos, roof_assumed=True))
        self.assertNotIn("roof assumed", _format_shadow_frame_label(pos, roof_assumed=False))

        casters_with_roof, roof_assumed_with_roof, roof_spec_none = _select_shadow_casters(
            [("IfcWall", wall_m, ()), ("IfcWindow", win_m, ()), ("IfcRoof", roof_m, ())],
            0.0,
        )
        self.assertNotIn(win_m, casters_with_roof)
        self.assertFalse(roof_assumed_with_roof)
        self.assertIsNone(roof_spec_none)

        _, roof_assumed_top_slab, _ = _select_shadow_casters(
            [("IfcWall", wall_m, ()), ("IfcSlab", top_slab_m, ())],
            0.0,
        )
        self.assertFalse(roof_assumed_top_slab)

    def test_passive_solar_reward_fail_closed_and_gates(self) -> None:
        # Omitting required gate keyword arguments must raise TypeError (never silently return 1.0).
        with self.assertRaises(TypeError):
            compute_passive_solar_reward(4.5, 5.0)  # type: ignore[call-arg]
        with self.assertRaises(TypeError):
            compute_passive_solar_reward(4.5, 5.0, verify_ok=True, spec_match=1.0, has_closed_envelope=True)  # type: ignore[call-arg]

        # None or invalid / failing gates must return 0.0.
        self.assertEqual(
            compute_passive_solar_reward(
                5.0, 4.0, verify_ok=False, spec_match=1.0, has_closed_envelope=True, max_facade_wwr=0.25
            ),
            0.0,
        )
        self.assertEqual(
            compute_passive_solar_reward(
                5.0, 4.0, verify_ok=True, spec_match=0.8, has_closed_envelope=True, max_facade_wwr=0.25
            ),
            0.0,
        )
        self.assertEqual(
            compute_passive_solar_reward(
                5.0, 4.0, verify_ok=True, spec_match=1.0, has_closed_envelope=False, max_facade_wwr=0.25
            ),
            0.0,
        )
        self.assertEqual(
            compute_passive_solar_reward(
                5.0, 4.0, verify_ok=True, spec_match=1.0, has_closed_envelope=True, max_facade_wwr=0.55
            ),
            0.0,
        )
        self.assertEqual(
            compute_passive_solar_reward(
                5.0, 4.0, verify_ok=True, spec_match=1.0, has_closed_envelope=True, max_facade_wwr=None  # type: ignore[arg-type]
            ),
            0.0,
        )
        self.assertGreater(
            compute_passive_solar_reward(
                4.5, 4.0, verify_ok=True, spec_match=1.0, has_closed_envelope=True, max_facade_wwr=0.35
            ),
            0.8,
        )

    def test_envelope_gates_closed_open_roof_and_wwr(self) -> None:
        base_walls = [
            {"id": "wS", "start": (0.0, 0.0), "end": (10.0, 0.0), "height_m": 3.0},
            {"id": "wE", "start": (10.0, 0.0), "end": (10.0, 8.0), "height_m": 3.0},
            {"id": "wN", "start": (10.0, 8.0), "end": (0.0, 8.0), "height_m": 3.0},
            {"id": "wW", "start": (0.0, 8.0), "end": (0.0, 0.0), "height_m": 3.0},
        ]
        win_25 = [{"id": "win1", "host_wall_id": "wS", "width_m": 5.0, "height_m": 1.5}]  # 7.5 / 30 = 0.25
        floor_only = [{"id": "slab0", "base_z_m": -0.2, "top_z_m": 0.0, "is_roof_type": False}]
        roof_slab = [{"id": "roof0", "base_z_m": 3.0, "top_z_m": 3.2, "is_roof_type": True}]

        # 1. Closed loop + roof + 25% south window -> passes gates.
        ok = envelope_gates({"walls": base_walls, "windows": win_25, "roofs": [{"id": "r1"}], "slabs": floor_only})
        self.assertTrue(ok["storeys_closed"])
        self.assertTrue(ok["has_roof_or_top_slab"])
        self.assertTrue(ok["has_closed_envelope"])
        self.assertAlmostEqual(ok["facade_wwr"]["S"], 0.25, places=3)
        self.assertAlmostEqual(ok["max_facade_wwr"], 0.25, places=3)
        self.assertGreater(
            compute_passive_solar_reward(
                4.5, 4.0, verify_ok=True, spec_match=1.0,
                has_closed_envelope=ok["has_closed_envelope"], max_facade_wwr=ok["max_facade_wwr"],
            ),
            0.8,
        )

        # 2. Floor slab only (no roof / top slab) -> fails closed envelope -> reward 0.0.
        no_roof = envelope_gates({"walls": base_walls, "windows": win_25, "roofs": [], "slabs": floor_only})
        self.assertTrue(no_roof["storeys_closed"])
        self.assertFalse(no_roof["has_roof_or_top_slab"])
        self.assertFalse(no_roof["has_closed_envelope"])
        self.assertEqual(
            compute_passive_solar_reward(
                4.5, 4.0, verify_ok=True, spec_match=1.0,
                has_closed_envelope=no_roof["has_closed_envelope"], max_facade_wwr=no_roof["max_facade_wwr"],
            ),
            0.0,
        )

        # 3. Missing exterior south wall EVEN WITH a closed interior room inside -> storeys_closed=False!
        open_ext_closed_int = [
            *base_walls[1:],  # omit wS!
            {"id": "int1", "start": (2.0, 2.0), "end": (8.0, 2.0), "height_m": 3.0},
            {"id": "int2", "start": (8.0, 2.0), "end": (8.0, 6.0), "height_m": 3.0},
            {"id": "int3", "start": (8.0, 6.0), "end": (2.0, 6.0), "height_m": 3.0},
            {"id": "int4", "start": (2.0, 6.0), "end": (2.0, 2.0), "height_m": 3.0},
        ]
        open_res = envelope_gates({"walls": open_ext_closed_int, "windows": [], "roofs": [{"id": "r1"}], "slabs": floor_only})
        self.assertFalse(open_res["storeys_closed"])
        self.assertFalse(open_res["has_closed_envelope"])

        # 4. Over-glazed facade (60% WWR on south wall) -> reward fails closed with 0.0.
        win_60 = [{"id": "win_big", "host_wall_id": "wS", "width_m": 9.0, "height_m": 2.0}]  # 18 / 30 = 0.60
        over = envelope_gates({"walls": base_walls, "windows": win_60, "roofs": [], "slabs": roof_slab})
        self.assertTrue(over["has_closed_envelope"])
        self.assertAlmostEqual(over["max_facade_wwr"], 0.60, places=3)
        self.assertEqual(
            compute_passive_solar_reward(
                4.5, 4.0, verify_ok=True, spec_match=1.0,
                has_closed_envelope=over["has_closed_envelope"], max_facade_wwr=over["max_facade_wwr"],
            ),
            0.0,
        )

        # 5. Rotated site: true_north_deg = +90° (True North is Plan +X -> Plan -X outward wall wW faces True South).
        win_on_wW = [{"id": "winW", "host_wall_id": "wW", "width_m": 4.0, "height_m": 1.5}]  # 6 / 24 = 0.25
        rot = envelope_gates({"walls": base_walls, "windows": win_on_wW, "roofs": [{"id": "r1"}], "true_north_deg": 90.0})
        self.assertAlmostEqual(rot["facade_wwr"]["S"], 0.25, places=3)
        self.assertEqual(rot["facade_wwr"]["W"], 0.0)

        # 6. Synthetic IFC STEP string round-trip.
        step_ifc = """ISO-10303-21;
DATA;
#1=IFCSIUNIT(*,.LENGTHUNIT.,.MILLI.,.METRE.);
#2=IFCCARTESIANPOINT((0.,0.,0.));
#3=IFCAXIS2PLACEMENT3D(#2,$,$);
#4=IFCDIRECTION((0.,1.));
#5=IFCGEOMETRICREPRESENTATIONCONTEXT($,'Model',3,1.E-5,#3,#4);
#10=IFCLOCALPLACEMENT($,#3);
#20=IFCAXIS2PLACEMENT3D(#2,$,#4);
#21=IFCCARTESIANPOINT((5000.,0.,0.));
#22=IFCAXIS2PLACEMENT2D(#21,$);
#23=IFCRECTANGLEPROFILEDEF(.AREA.,$,#22,10000.,200.);
#24=IFCEXTRUDEDAREASOLID(#23,#3,#4,3000.);
#25=IFCSHAPEREPRESENTATION($,'Body','SweptSolid',(#24));
#26=IFCPRODUCTDEFINITIONSHAPE($,$,(#25));
#30=IFCCARTESIANPOINT((0.,0.,0.));
#31=IFCAXIS2PLACEMENT3D(#30,#32,#33);
#32=IFCDIRECTION((0.,0.,1.));
#33=IFCDIRECTION((1.,0.,0.));
#34=IFCLOCALPLACEMENT(#10,#31);
#35=IFCWALL('w1',$,'Wall',$,$,#34,#26,$,$);
#40=IFCCARTESIANPOINT((10000.,0.,0.));
#41=IFCDIRECTION((0.,1.,0.));
#42=IFCAXIS2PLACEMENT3D(#40,#32,#41);
#43=IFCLOCALPLACEMENT(#10,#42);
#44=IFCCARTESIANPOINT((4000.,0.,0.));
#45=IFCAXIS2PLACEMENT2D(#44,$);
#46=IFCRECTANGLEPROFILEDEF(.AREA.,$,#45,8000.,200.);
#47=IFCEXTRUDEDAREASOLID(#46,#3,#32,3000.);
#48=IFCSHAPEREPRESENTATION($,'Body','SweptSolid',(#47));
#49=IFCPRODUCTDEFINITIONSHAPE($,$,(#48));
#50=IFCWALL('w2',$,'Wall',$,$,#43,#49,$,$);
#60=IFCCARTESIANPOINT((10000.,8000.,0.));
#61=IFCDIRECTION((-1.,0.,0.));
#62=IFCAXIS2PLACEMENT3D(#60,#32,#61);
#63=IFCLOCALPLACEMENT(#10,#62);
#64=IFCWALL('w3',$,'Wall',$,$,#63,#26,$,$);
#70=IFCCARTESIANPOINT((0.,8000.,0.));
#71=IFCDIRECTION((0.,-1.,0.));
#72=IFCAXIS2PLACEMENT3D(#70,#32,#71);
#73=IFCLOCALPLACEMENT(#10,#72);
#74=IFCWALL('w4',$,'Wall',$,$,#73,#49,$,$);
#80=IFCOPENINGELEMENT('op1',$,$,$,$,#34,$,$);
#81=IFCRELVOIDSELEMENT('rv1',$,$,$,#35,#80);
#82=IFCWINDOW('win1',$,$,$,$,#34,$,$,1500.,4000.,$,$,$);
#83=IFCRELFILLSELEMENT('rf1',$,$,$,#80,#82);
#90=IFCROOF('r1',$,$,$,$,#10,$,$);
ENDSEC;
"""
        ifc_gates = envelope_gates(step_ifc)
        self.assertTrue(ifc_gates["has_closed_envelope"])
        self.assertEqual(ifc_gates["exterior_wall_count"], 4)
        self.assertAlmostEqual(ifc_gates["facade_wwr"]["S"], 0.2, places=3)  # 6 m² / 30 m² = 0.20

        # 7. Interior partitions inside a closed perimeter — even tagged is_external=True — do NOT count in exterior_wall_count.
        with_partitions = [
            *base_walls,
            {"id": "p1", "start": (5.0, 0.0), "end": (5.0, 8.0), "height_m": 3.0, "is_external": True},
            {"id": "p2", "start": (5.0, 4.0), "end": (10.0, 4.0), "height_m": 3.0, "is_external": True},
        ]
        part_gates = envelope_gates({"walls": with_partitions, "windows": win_25, "roofs": [{"id": "r1"}], "slabs": floor_only})
        self.assertTrue(part_gates["has_closed_envelope"])
        self.assertEqual(part_gates["exterior_wall_count"], 4)
        self.assertAlmostEqual(part_gates["facade_wwr"]["S"], 0.25, places=3)


class TestSketchEngine(unittest.TestCase):
    def test_all_topologies_generate_valid_scripts_and_bounds(self) -> None:
        self.assertEqual(len(SKETCH_TOPOLOGIES), 6)
        for idx, topo in enumerate(SKETCH_TOPOLOGIES):
            sample = generate_sketch_sample(2000 + idx, topology=topo)
            self.assertEqual(sample.topology, topo)
            self.assertIn('from "@permitzip/bim-script"', sample.ts_source)
            self.assertIn("export default defineScript(", sample.ts_source)
            self.assertIn("units.ft", sample.ts_source)
            self.assertGreaterEqual(len(sample.rooms), 2)
            self.assertGreaterEqual(len(sample.walls), 5)
            self.assertGreaterEqual(len(sample.openings), 3)
            for op in sample.openings:
                self.assertGreaterEqual(op.along_ft, 2.0)

    def test_determinism_and_batch(self) -> None:
        a = generate_sketch_sample(4242, topology="l_shaped_studio")
        b = generate_sketch_sample(4242, topology="l_shaped_studio")
        self.assertEqual(a.ts_source, b.ts_source)
        batch = generate_sketch_dataset_specs(12, start_seed=100)
        self.assertEqual(len(batch), 12)
        self.assertEqual({s.topology for s in batch}, set(SKETCH_TOPOLOGIES))

    def test_render_parametric_sketch_svg_styles(self) -> None:
        sample = generate_sketch_sample(500, topology="boutique_cafe")
        for style in SKETCH_STYLES:
            svg = render_parametric_sketch_svg(sample, style=style)
            self.assertTrue(svg.startswith("<svg"))
            self.assertTrue(svg.endswith("</svg>"))
            self.assertIn("<path", svg)
            self.assertIn("CAFE SEATING", svg)


class TestMixerMcpServer(unittest.TestCase):
    def test_mcp_json_and_agents_md_present(self) -> None:
        mcp_cfg = json.loads((REPO_ROOT / ".mcp.json").read_text(encoding="utf-8"))
        self.assertIn("mutinybim-mixer", mcp_cfg.get("mcpServers", {}))
        srv = mcp_cfg["mcpServers"]["mutinybim-mixer"]
        self.assertEqual(srv["command"], "python3")
        self.assertEqual(srv["args"], ["-m", "mixer.mcp_server"])

        agents_md = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
        self.assertIn("mixer.mcp_server", agents_md)
        self.assertIn("googlecolab/colab-mcp", agents_md)

    def test_jsonrpc_tools_list_and_offline_calls(self) -> None:
        init = handle_jsonrpc_message({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}})
        self.assertEqual(init["result"]["serverInfo"]["name"], "mutinybim-mixer")
        self.assertIsNone(handle_jsonrpc_message({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        self.assertEqual(handle_jsonrpc_message({"jsonrpc": "2.0", "id": 2, "method": "ping"})["result"], {})

        tlist = handle_jsonrpc_message({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})
        names = [t["name"] for t in tlist["result"]["tools"]]
        self.assertEqual(names, [t["name"] for t in TOOLS])
        self.assertEqual(len(names), 9)

        def _call(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            resp = handle_jsonrpc_message(
                {"jsonrpc": "2.0", "id": 10, "method": "tools/call", "params": {"name": tool_name, "arguments": arguments}}
            )
            self.assertFalse(resp["result"]["isError"], resp)
            return json.loads(resp["result"]["content"][0]["text"])

        info = _call("mixer_info", {})
        self.assertEqual(info["mutinybim"]["commit"], "fbf5f61424b64d9f99b1c896ca02328c88d9813c")
        self.assertIn("passive_solar", info["trades"])

        sk = _call("generate_sketch", {"topology": "boutique_cafe", "seed": 77, "style": "black_sharpie_napkin"})
        self.assertEqual(sk["topology"], "boutique_cafe")
        self.assertTrue(sk["svg"].startswith("<svg"))

        sol = _call("solar_position_and_dial", {"city": "sydney", "date": "summer_solstice", "hour_solar": 12.0})
        self.assertTrue(sol["azimuth_deg"] < 2.0 or sol["azimuth_deg"] > 358.0)
        self.assertIn("<svg", sol["svg"])

        gates = _call(
            "check_envelope_gates",
            {
                "model": {
                    "walls": [
                        {"id": "wS", "start": [0, 0], "end": [10, 0], "height_m": 3.0},
                        {"id": "wE", "start": [10, 0], "end": [10, 8], "height_m": 3.0},
                        {"id": "wN", "start": [10, 8], "end": [0, 8], "height_m": 3.0},
                        {"id": "wW", "start": [0, 8], "end": [0, 0], "height_m": 3.0},
                    ],
                    "windows": [{"id": "win1", "host_wall_id": "wS", "width_m": 4.0, "height_m": 1.5}],
                    "roofs": [{"id": "r1"}],
                },
                "winter_mean_hours": 4.5,
                "summer_mean_hours": 4.0,
                "verify_ok": True,
                "spec_match": 1.0,
            },
        )
        self.assertTrue(gates["has_closed_envelope"])
        self.assertAlmostEqual(gates["facade_wwr"]["S"], 0.2, places=3)
        self.assertGreater(gates["passive_solar_reward"], 0.8)

        sun_pts = _call("solar_direct_sun_hours", {"points": [[0.0, 0.0, 0.0]], "city": "san_francisco", "date": "spring_equinox", "step_minutes": 30})
        self.assertGreater(sun_pts["point_hours"][0], 11.0)

        pr = _call("build_studio_prompt", {"user_vibe": "Warm cedar library"})
        self.assertIn("Warm cedar library", pr["positive_prompt"])
        self.assertNotIn("solar_lighting", pr)

        pr_sol = _call(
            "build_studio_prompt",
            {"user_vibe": "Warm cedar library", "hour_solar": 16.75, "city": "london", "date": "spring_equinox"},
        )
        self.assertIn("solar_lighting", pr_sol)
        self.assertEqual(pr_sol["solar_lighting"]["phase"], "golden_hour")
        self.assertIn("golden-hour", pr_sol["positive_prompt"])

        compat = _call("check_adapter_compatibility", {"adapters": [{"r": 16, "lora_alpha": 16, "lora_dropout": 0.0, "bias": "none"}]})
        self.assertTrue(compat["all_compatible"])

        unk = handle_jsonrpc_message({"jsonrpc": "2.0", "id": 99, "method": "tools/call", "params": {"name": "no_such_tool"}})
        self.assertIn("error", unk)

    def test_stdio_subprocess_roundtrip(self) -> None:
        msgs = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "mixer_info", "arguments": {}}},
        ]
        stdin_payload = "\n".join(json.dumps(m) for m in msgs) + "\n"
        proc = subprocess.run(
            [sys.executable, "-m", "mixer.mcp_server"],
            cwd=str(REPO_ROOT),
            input=stdin_payload,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        lines = [json.loads(line) for line in proc.stdout.strip().splitlines() if line.strip()]
        self.assertEqual([ln["id"] for ln in lines], [1, 2, 3])
        self.assertEqual(len(lines[1]["result"]["tools"]), 9)
        info = json.loads(lines[2]["result"]["content"][0]["text"])
        self.assertEqual(info["spec_version"], "0.1.0")

    def test_workspace_confinement_blocks_outside_and_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as ws_tmp, tempfile.TemporaryDirectory() as out_tmp:
            ws = Path(ws_tmp).resolve()
            outside = Path(out_tmp).resolve()
            outside_cfg = outside / "adapter_config.json"
            outside_cfg.write_text(json.dumps({"r": 16, "lora_alpha": 16}), encoding="utf-8")
            outside_ifc = outside / "canary.ifc"
            outside_ifc.write_text("ISO-10303-21;\nDATA;\nENDSEC;\n", encoding="utf-8")
            outside_write = outside / "written-here"

            # Outside reads must return isError=True.
            r1 = handle_jsonrpc_message(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "check_adapter_compatibility", "arguments": {"adapters": [str(outside_cfg)]}},
                },
                workspace=ws,
            )
            self.assertTrue(r1["result"]["isError"])

            r2 = handle_jsonrpc_message(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": "check_envelope_gates", "arguments": {"ifc_path": str(outside_ifc)}},
                },
                workspace=ws,
            )
            self.assertTrue(r2["result"]["isError"])

            # Outside writes must fail without creating directories.
            r3 = handle_jsonrpc_message(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "export_bundle",
                        "arguments": {"ts_source": "export default 1;", "out_dir": str(outside_write)},
                    },
                },
                workspace=ws,
            )
            self.assertTrue(r3["result"]["isError"])
            self.assertFalse(outside_write.exists())

            r4 = handle_jsonrpc_message(
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "tools/call",
                    "params": {
                        "name": "solar_direct_sun_hours",
                        "arguments": {"ifc_path": "inside.ifc", "out_dir": str(outside_write)},
                    },
                },
                workspace=ws,
            )
            self.assertTrue(r4["result"]["isError"])
            self.assertFalse(outside_write.exists())

            # Symlinks inside workspace pointing outside must also be refused.
            (ws / "link_cfg.json").symlink_to(outside_cfg)
            r5 = handle_jsonrpc_message(
                {
                    "jsonrpc": "2.0",
                    "id": 5,
                    "method": "tools/call",
                    "params": {"name": "check_adapter_compatibility", "arguments": {"adapters": ["link_cfg.json"]}},
                },
                workspace=ws,
            )
            self.assertTrue(r5["result"]["isError"])

            # Valid relative path inside workspace succeeds.
            (ws / "ok_cfg.json").write_text(json.dumps({"r": 16, "lora_alpha": 16, "bias": "none"}), encoding="utf-8")
            r6 = handle_jsonrpc_message(
                {
                    "jsonrpc": "2.0",
                    "id": 6,
                    "method": "tools/call",
                    "params": {"name": "check_adapter_compatibility", "arguments": {"adapters": ["ok_cfg.json"]}},
                },
                workspace=ws,
            )
            self.assertFalse(r6["result"]["isError"])
            self.assertTrue(json.loads(r6["result"]["content"][0]["text"])["all_compatible"])

    def test_verifier_url_cannot_be_overridden_by_caller_and_loopback_enforced(self) -> None:
        by_name = {t["name"]: t for t in TOOLS}
        self.assertNotIn("verifier_url", by_name["verify_script"]["inputSchema"]["properties"])
        self.assertNotIn("verifier_url", by_name["export_bundle"]["inputSchema"]["properties"])

        self.assertEqual(_validate_verifier_url("http://127.0.0.1:8787"), "http://127.0.0.1:8787")
        self.assertEqual(_validate_verifier_url("http://localhost:8787"), "http://localhost:8787")
        with self.assertRaises(ValueError):
            _validate_verifier_url("http://example.com:8787")
        self.assertEqual(
            _validate_verifier_url("http://example.com:8787", allow_remote=True),
            "http://example.com:8787",
        )

        hits: list[str] = []

        class _CanaryHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                hits.append(f"GET {self.path}")
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

        srv = ThreadingHTTPServer(("127.0.0.1", 0), _CanaryHandler)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            canary_url = f"http://127.0.0.1:{srv.server_address[1]}"
            # Configure server with a closed loopback port (not canary_url) and ensure caller's verifier_url is ignored.
            with tempfile.TemporaryDirectory() as unused_tmp:
                resp = handle_jsonrpc_message(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {
                            "name": "verify_script",
                            "arguments": {"ts_source": "export default 1;", "verifier_url": canary_url},
                        },
                    },
                    verifier_url="http://127.0.0.1:9",
                    workspace=Path(unused_tmp),
                )
            self.assertTrue(resp["result"]["isError"])
            self.assertEqual(hits, [])
        finally:
            srv.shutdown()
            srv.server_close()

    def test_stdio_malformed_json_recovers_for_next_ping(self) -> None:
        stdin_payload = "not-valid-json\n" + json.dumps({"jsonrpc": "2.0", "id": 7, "method": "ping"}) + "\n"
        proc = subprocess.run(
            [sys.executable, "-m", "mixer.mcp_server"],
            cwd=str(REPO_ROOT),
            input=stdin_payload,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        lines = [json.loads(line) for line in proc.stdout.strip().splitlines() if line.strip()]
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["error"]["code"], -32700)
        self.assertEqual(lines[1], {"jsonrpc": "2.0", "id": 7, "result": {}})


if __name__ == "__main__":
    unittest.main()
