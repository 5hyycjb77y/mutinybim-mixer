"""Opt-in CPU smoke of the notebook's real verifier, against the pinned repo."""

import base64
import os
from pathlib import Path
import socket
import subprocess
import time
import unittest
from urllib.error import URLError

from verifier.verify_client import MutinyVerifierClient
from mixer.lora_ops import load_spec
from mixer.solar_engine import _split_step_args, envelope_gates, read_ifc_true_north_deg

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
            finally:
                if proc.poll() is None:
                    proc.terminate()
