"""Opt-in CPU smoke of the notebook's real verifier, against the pinned repo."""

import base64
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import time
import unittest
from urllib.error import URLError

from verifier.verify_client import MutinyVerifierClient
from mixer.lora_ops import load_spec
from mixer.sketch_engine import generate_sketch_sample
from mixer.solar_engine import (
    _extract_model_elements_from_ifc,
    _split_step_args,
    compute_direct_sun_hours_for_points,
    envelope_gates,
    read_ifc_true_north_deg,
    render_solstice_shadow_gif,
)
from mixer.studio_render import (
    _resolve_space_boxes,
    compute_camera_pose,
    extract_ifc_room_boxes,
    first_wall_hit_distance_m,
)

ROOT = Path(__file__).resolve().parent.parent
CHECKOUT = os.environ.get("MIXER_TEST_MUTINYBIM")


@unittest.skipUnless(CHECKOUT, "set MIXER_TEST_MUTINYBIM to a checkout with npm dependencies")
class TestLiveVerifier(unittest.TestCase):
    def test_official_examples_and_office_export(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        with subprocess.Popen(
            ["node", str(ROOT / "verifier/server.mjs"), "--mutinybim", CHECKOUT,
             "--host", "127.0.0.1", "--port", str(port), "--spec", str(ROOT / "spec.json")],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        ) as proc:
            try:
                client = MutinyVerifierClient(f"http://127.0.0.1:{port}", timeout=30)
                for _ in range(40):
                    if proc.poll() is not None:
                        self.fail(proc.stdout.read())
                    try:
                        health = client.health()
                        break
                    except URLError:
                        time.sleep(0.25)
                else:
                    self.fail("verifier did not start")
                self.assertEqual(health["mutinybim_commit"], load_spec(ROOT / "spec.json")["mutinybim"]["commit"])
                examples = {x["name"]: x["source"] for x in client.reference(compact=True)["examples"]}
                for name in ("small-office", "rooftop-unit", "plumbing-riser", "electrical-room", "mechanical-details"):
                    with self.subTest(name=name):
                        result = client.verify(examples[name])
                        self.assertTrue(result.ok, result.error)
                bundle = client.export_bundle(examples["small-office"], project_name="Synthetic office")
                self.assertTrue(bundle.verify.ok, bundle.verify.error)
                self.assertIn("IFC4X3", bundle.ifc_text or "")
                self.assertIn("IFCGEOMETRICREPRESENTATIONCONTEXT", bundle.ifc_text or "")
                self.assertIn("IFCOPENINGELEMENT", bundle.ifc_text or "")
                # Every IfcOpeningElement at fbf5f61 has a body (non-$ Representation attribute at index 6).
                opening_lines = [ln for ln in (bundle.ifc_text or "").splitlines() if "=IFCOPENINGELEMENT(" in ln]
                self.assertGreaterEqual(len(opening_lines), 10)
                for ln in opening_lines:
                    args = _split_step_args(ln.split("(", 1)[1].rsplit(")", 1)[0])
                    self.assertTrue(len(args) >= 7 and args[6].startswith("#"), f"Opening missing body: {ln}")
                self.assertEqual(read_ifc_true_north_deg(bundle.ifc_text or ""), 0.0)
                gates = envelope_gates(bundle.ifc_text or "")
                self.assertTrue(gates["storeys_closed"], gates)
                self.assertEqual(gates["exterior_wall_count"], 8, gates)
                self.assertGreaterEqual(gates["max_facade_wwr"], 0.0)
                self.assertLessEqual(gates["max_facade_wwr"], 1.0)
                self.assertEqual(gates["true_north_deg"], 0.0)
                self.assertEqual(len(bundle.sheets), 2)
                self.assertTrue(base64.b64decode(bundle.pdf_base64 or "").startswith(b"%PDF"))

                # Real export room extraction, interior forward-ray guard, mutation check, and flush ceiling sky-slot test.
                apt = generate_sketch_sample(1042, topology="apartment_unit")
                apt_bundle = client.export_bundle(apt.ts_source, project_name="Apartment 1042")
                self.assertTrue(apt_bundle.verify.ok, apt_bundle.verify.error)
                apt_ifc = apt_bundle.ifc_text or ""
                apt_elems = _extract_model_elements_from_ifc(apt_ifc)
                apt_walls = apt_elems["walls"]
                apt_ceilings = apt_elems["ceilings"]
                self.assertEqual(len(apt_ceilings), 3)

                rooms = extract_ifc_room_boxes(apt_ifc)
                self.assertEqual(len(rooms), 3, rooms)
                # Expected rooms for apartment_unit seed 1042 (32x20 ft = 9.75x6.10 m):
                # Living (~3.96..9.75 x 0..6.10), Bedroom (~0..3.96 x 0..3.05), Bath (~0..3.96 x 3.05..6.10).
                expected = [
                    (3.96, 0.0, 9.75, 6.10),
                    (0.0, 0.0, 3.96, 3.05),
                    (0.0, 3.05, 3.96, 6.10),
                ]
                for ex0, ey0, ex1, ey1 in expected:
                    self.assertTrue(
                        any(
                            abs(r["xmin"] - ex0) <= 0.35
                            and abs(r["ymin"] - ey0) <= 0.35
                            and abs(r["xmax"] - ex1) <= 0.35
                            and abs(r["ymax"] - ey1) <= 0.35
                            for r in rooms
                        ),
                        f"Missing expected room ({ex0},{ey0})-({ex1},{ey1}) in {rooms}",
                    )

                xs_all = [w["start"][0] for w in apt_walls] + [w["end"][0] for w in apt_walls]
                ys_all = [w["start"][1] for w in apt_walls] + [w["end"][1] for w in apt_walls]
                zs_all = [w["base_z_m"] for w in apt_walls] + [w["top_z_m"] for w in apt_walls]
                vmin = (min(xs_all), min(ys_all), min(zs_all))
                vmax = (max(xs_all), max(ys_all), max(zs_all))

                for cam_key in ("eye_level_interior", "opposite_interior"):
                    pose = compute_camera_pose(
                        vmin,
                        vmax,
                        cam_key,
                        storey_elevations_m=[0.0],
                        space_boxes=rooms,
                        wall_segments=apt_walls,
                    )
                    self.assertFalse(pose["camera_fallback"], pose)
                    ex, ey, ez = pose["eye"]
                    self.assertAlmostEqual(ez, 1.6, places=2)
                    self.assertTrue(
                        any(r["xmin"] <= ex <= r["xmax"] and r["ymin"] <= ey <= r["ymax"] for r in rooms),
                        f"Eye {pose['eye']} outside detected rooms {rooms}",
                    )
                    hit_m = first_wall_hit_distance_m(pose["eye"], pose["forward"], apt_walls)
                    self.assertTrue(hit_m is None or hit_m >= 3.0, f"{cam_key} hit wall too close: {hit_m}m")

                # Mutation check: with empty space_boxes, fallback flags True and hits the partition < 3.0m.
                mut_pose = compute_camera_pose(
                    vmin,
                    vmax,
                    "eye_level_interior",
                    storey_elevations_m=[0.0],
                    space_boxes=[],
                    wall_segments=apt_walls,
                )
                self.assertTrue(mut_pose["camera_fallback"])
                mut_hit = first_wall_hit_distance_m(mut_pose["eye"], mut_pose["forward"], apt_walls)
                self.assertIsNotNone(mut_hit)
                self.assertLess(mut_hit, 3.0)

                # Flush-ceiling sky-slot solar test (London Jun 21): interior points away from windows & along x=3.96 read 0 h.
                tris = []
                for w in apt_walls:
                    sx, sy = w["start"]
                    ex, ey = w["end"]
                    bz, tz = w["base_z_m"], w["top_z_m"]
                    tris.append(((sx, sy, bz), (ex, ey, bz), (ex, ey, tz)))
                    tris.append(((sx, sy, bz), (ex, ey, tz), (sx, sy, tz)))
                for c in apt_ceilings:
                    poly = c["polygon"]
                    cz = c["base_z_m"]
                    p0 = (poly[0][0], poly[0][1], cz)
                    for k in range(1, len(poly) - 1):
                        tris.append((p0, (poly[k][0], poly[k][1], cz), (poly[k + 1][0], poly[k + 1][1], cz)))

                indoor_pts = [
                    (3.96, 1.5, 0.85),   # directly along middle bedroom/living wall junction
                    (3.96, 3.05, 0.85),  # central T-junction between all 3 rooms (old sky slot!)
                    (2.0, 4.5, 0.85),    # bath interior
                    (7.0, 3.0, 0.85),    # mid living studio
                ]
                hrs = compute_direct_sun_hours_for_points(
                    indoor_pts,
                    tris,
                    lat_deg=51.5074,
                    day_of_year=172,
                    true_north_deg=0.0,
                    step_minutes=30,
                )
                self.assertEqual(hrs, [0.0, 0.0, 0.0, 0.0], f"Sky-slot leakage detected: {hrs}")

                # Verify str path, Path object, and inline text all agree on rotated (-30 deg) apartment export.
                ctx_match = re.search(r"#\d+\s*=\s*IFCGEOMETRICREPRESENTATIONCONTEXT\(([^;]+)\)\s*;", apt_ifc, re.IGNORECASE)
                self.assertIsNotNone(ctx_match, "Missing IFCGEOMETRICREPRESENTATIONCONTEXT")
                ctx_args = _split_step_args(ctx_match.group(1))
                tn_ref = ctx_args[5].strip() if len(ctx_args) >= 6 else "$"
                if tn_ref.startswith("#"):
                    ref_id = tn_ref[1:]
                    rotated_ifc = re.sub(
                        rf"(#{ref_id}\s*=\s*IFCDIRECTION\(\()([^)]+)(\)\)\s*;)",
                        r"\g<1>-0.5,0.8660254\g<3>",
                        apt_ifc,
                        flags=re.IGNORECASE,
                    )
                else:
                    dir_line = "#999999=IFCDIRECTION((-0.5,0.8660254));\n"
                    rotated_ifc = apt_ifc.replace(
                        ctx_match.group(0),
                        ctx_match.group(0).replace(",$);", ",#999999);"),
                    ).replace("ENDSEC;", f"{dir_line}ENDSEC;", 1)

                with tempfile.TemporaryDirectory() as td:
                    tmp_ifc = Path(td) / "model.ifc"
                    tmp_ifc.write_text(rotated_ifc, encoding="utf-8")
                    gates_ref = envelope_gates(rotated_ifc)
                    for src in (str(tmp_ifc), tmp_ifc, rotated_ifc):
                        self.assertEqual(len(extract_ifc_room_boxes(src)), 3)
                        self.assertEqual(len(_resolve_space_boxes([], src)), 3)
                        self.assertAlmostEqual(read_ifc_true_north_deg(src), -30.0, delta=0.05)
                        g_src = envelope_gates(src)
                        self.assertEqual(g_src["exterior_wall_count"], gates_ref["exterior_wall_count"])
                        self.assertEqual(g_src["has_closed_envelope"], gates_ref["has_closed_envelope"])
                        self.assertEqual(g_src["facade_wwr"], gates_ref["facade_wwr"])

                    # End-to-end loader (M1) + shadow GIF (M4/M8) verification when Colab deps exist.
                    try:
                        import ifcopenshell  # noqa: F401  # type: ignore
                        import numpy  # noqa: F401  # type: ignore
                        from PIL import Image  # type: ignore
                        import trimesh  # noqa: F401  # type: ignore
                        from mixer.studio_render import _load_ifc_scene_and_metadata

                        _, _, _, loaded_spaces = _load_ifc_scene_and_metadata(str(tmp_ifc))
                        self.assertEqual(len(loaded_spaces), 3)
                        gif_out = Path(td) / "shadow.gif"
                        render_solstice_shadow_gif(
                            str(tmp_ifc),
                            str(gif_out),
                            lat_deg=51.5,
                            day_of_year=80,
                            step_minutes=240,
                        )
                        self.assertTrue(gif_out.exists())
                        with Image.open(gif_out) as im:
                            self.assertGreaterEqual(getattr(im, "n_frames", 1), 1)
                    except ImportError:
                        pass
            finally:
                if proc.poll() is None:
                    proc.terminate()
