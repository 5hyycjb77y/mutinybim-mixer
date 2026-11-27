"""Opt-in: every sketch topology's ground-truth script builds in the pinned MutinyBIM, with the counts it promises."""

import os
from pathlib import Path
import socket
import subprocess
import time
import unittest
from urllib.error import URLError

from verifier.verify_client import MutinyVerifierClient
from mixer.sketch_engine import SKETCH_TOPOLOGIES, generate_sketch_sample

ROOT = Path(__file__).resolve().parent.parent
CHECKOUT = os.environ.get("MIXER_TEST_MUTINYBIM")
SEEDS_PER_TOPOLOGY = int(os.environ.get("MIXER_SKETCH_SEEDS", "5"))


@unittest.skipUnless(CHECKOUT, "set MIXER_TEST_MUTINYBIM to a checkout with npm dependencies")
class TestLiveSketchEngine(unittest.TestCase):
    def test_every_topology_builds_with_its_expected_counts(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        with subprocess.Popen(
            ["node", str(ROOT / "verifier/server.mjs"), "--mutinybim", CHECKOUT,
             "--host", "127.0.0.1", "--port", str(port), "--spec", str(ROOT / "spec.json")],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        ) as proc:
            try:
                client = MutinyVerifierClient(f"http://127.0.0.1:{port}", timeout=60)
                for _ in range(40):
                    if proc.poll() is not None:
                        self.fail(proc.stdout.read())
                    try:
                        client.health()
                        break
                    except URLError:
                        time.sleep(0.25)
                else:
                    self.fail("verifier did not start")
                for topology in SKETCH_TOPOLOGIES:
                    for n in range(SEEDS_PER_TOPOLOGY):
                        seed = 1000 + n * len(SKETCH_TOPOLOGIES)
                        sample = generate_sketch_sample(seed, topology=topology)
                        with self.subTest(topology=topology, seed=seed):
                            result = client.verify(sample.ts_source, expected_kinds=sample.expected_kinds)
                            self.assertTrue(result.ok, f"{result.stage_failed}: {result.error}")
                            self.assertEqual(result.breakdown.get("spec_match"), 1.0, f"built {result.kinds}, expected {sample.expected_kinds}")
            finally:
                if proc.poll() is None:
                    proc.terminate()
