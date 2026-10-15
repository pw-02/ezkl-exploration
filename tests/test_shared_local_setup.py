"""Regression checks for immutable local setup reuse (no native EZKL required)."""
import gzip
import shutil
import sys
import tempfile
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from zkinfer.runtime.data_plane import ArtifactRef
from zkinfer.runtime.node.cache import NodeCache
from zkinfer.backends.ezkl.ezkl_prover import EZKLProofStages


class SharedLocalSetupTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.shared = self.root / "bootstrap"
        self.shared.mkdir()
        for name in NodeCache.SETUP_FILES:
            (self.shared / name).write_bytes(b'{"run_args":{"logrows":1}}')
        self.ref = ArtifactRef(backend="filesystem", uri=str(self.shared))
        self.cache = NodeCache(str(self.root / "cache"), max_bytes=1)

    def test_complete_setup_reuses_original_and_survives_eviction_and_cleanup(self):
        original = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.shared.iterdir()}
        with patch("shutil.copy2", side_effect=AssertionError("unexpected copy")):
            with self.cache.pinned_setup("key"):
                result, metrics = self.cache.restore_setup_with_metrics(self.ref, "key")
                self.assertEqual(Path(result), self.shared)
                self.assertTrue(metrics["shared_read_only"])
                self.assertEqual(metrics["bytes_downloaded"], 0)
                self.assertGreater(metrics["bytes_reused_shared"], 0)
                self.assertEqual(self.cache.total_bytes(), 0)
        self.cache.evict_if_needed()
        shutil.rmtree(self.cache.root_dir)
        self.assertEqual(original, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.shared.iterdir()})

    def test_concurrent_reuse(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: self.cache.restore_setup(self.ref, "key"), range(12)))
        self.assertEqual(set(results), {str(self.shared)})
        self.assertEqual(self.cache.total_bytes(), 0)

    def test_incomplete_setup_remains_private(self):
        (self.shared / "pk.json").unlink()
        result, metrics = self.cache.restore_setup_with_metrics(self.ref, "key")
        self.assertNotEqual(Path(result), self.shared)
        self.assertFalse(metrics.get("shared_read_only", False))
        (Path(result) / "pk.json").write_bytes(b"new")
        self.assertFalse((self.shared / "pk.json").exists())

    def test_compressed_only_setup_still_restores_privately(self):
        (self.shared / "pk.json").unlink()
        with gzip.open(self.shared / "pk.json.gz", "wb") as file:
            file.write(b"key")
        result, metrics = self.cache.restore_setup_with_metrics(self.ref, "key")
        self.assertNotEqual(Path(result), self.shared)
        self.assertEqual((Path(result) / "pk.json").read_bytes(), b"key")
        self.assertFalse(metrics.get("shared_read_only", False))

    def test_s3_still_downloads_to_private_cache(self):
        def download(bucket, key, destination):
            Path(destination).write_bytes(b"downloaded")
        with patch("zkinfer.runtime.node.cache.s3.exists", side_effect=lambda b, k: not k.endswith(".gz")), patch(
            "zkinfer.runtime.node.cache.s3.download_file", side_effect=download
        ):
            result, metrics = self.cache.restore_setup_with_metrics(
                ArtifactRef(backend="s3", uri="setup/key", s3_bucket="bucket"), "key"
            )
        self.assertTrue(Path(result).is_relative_to(self.cache.root_dir))
        self.assertGreater(metrics["bytes_downloaded"], 0)
        self.assertFalse(metrics.get("shared_read_only", False))

    def test_read_only_prover_rejects_missing_keys_and_writes(self):
        with patch.dict(sys.modules, {"ezkl": types.ModuleType("ezkl")}):
            prover = EZKLProofStages(
                input_data_path="input", onnx_model_path="model",
                artifact_dir=str(self.root / "output"),
                proving_cache_enabled=True, proving_cache_overwrite=False,
                proving_cache_type="filesystem", proving_cache_root_dir=str(self.shared),
                proving_cache_read_only=True, local_tmp_dir=str(self.root / "tmp"),
                srs_dir=str(self.root / "srs"),
            )
        self.assertEqual(prover._try_load_keys_from_cache(), (True, 0.0))
        self.assertTrue(prover.calibrate_settings_with_fallback()[0])
        self.assertTrue(prover.compile_circuit()[0])
        with self.assertRaises(RuntimeError):
            prover._save_settings({})
        with self.assertRaises(RuntimeError):
            prover.setup_and_compute_proof()
        (self.shared / "pk.json").unlink()
        with self.assertRaises(FileNotFoundError):
            prover._try_load_keys_from_cache()
        self.assertFalse((self.shared / "pk.json").exists())


if __name__ == "__main__":
    unittest.main()
