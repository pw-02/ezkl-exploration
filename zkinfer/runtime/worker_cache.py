from __future__ import annotations

import gzip
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from zkinfer.runtime.data_plane import ArtifactRef
from zkinfer.storage import s3
from zkinfer.storage.artifact_store import sha256_file


class WorkerLocalCache:
    """Bounded node-local cache for immutable artifacts and setup outputs.

    The shared artifact store is canonical. This cache is a best-effort local
    acceleration layer with an LRU-style eviction policy. It is safe to delete
    and rebuild from shared storage.
    """

    SETUP_FILES = ("settings.json", "network.compiled", "pk.json", "vk.json")

    def __init__(
        self,
        root_dir: str,
        max_bytes: int,
        default_s3_bucket: str = "",
    ):
        self.root_dir = Path(root_dir)
        self.objects_dir = self.root_dir / "objects"
        self.setup_dir_root = self.root_dir / "setup"
        self.max_bytes = max(0, int(max_bytes or 0))
        self.default_s3_bucket = default_s3_bucket or ""
        self.last_eviction: Dict[str, Any] = {"eviction_count": 0, "evicted_bytes": 0}

        self.objects_dir.mkdir(parents=True, exist_ok=True)
        self.setup_dir_root.mkdir(parents=True, exist_ok=True)

    def total_bytes(self) -> int:
        total = 0
        if not self.root_dir.exists():
            return 0
        for path in self.root_dir.rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except FileNotFoundError:
                    continue
        return total

    def cached_setup_keys(self) -> list[str]:
        keys = []
        for setup_dir in self.setup_dir_root.iterdir() if self.setup_dir_root.exists() else []:
            if not setup_dir.is_dir():
                continue
            if (setup_dir / "settings.json").exists() or (setup_dir / "network.compiled").exists():
                keys.append(setup_dir.name)
        return sorted(set(keys))

    def setup_dir(self, setup_key: str) -> str:
        safe_key = setup_key.replace("/", "_").replace("..", "_") or "unknown"
        path = self.setup_dir_root / safe_key
        path.mkdir(parents=True, exist_ok=True)
        self._touch(path)
        return str(path)

    def materialize(self, ref: ArtifactRef, filename_hint: str = "artifact") -> str:
        path, _ = self.materialize_with_metrics(ref, filename_hint=filename_hint)
        return path

    def materialize_with_metrics(
        self,
        ref: ArtifactRef,
        filename_hint: str = "artifact",
    ) -> Tuple[str, Dict[str, Any]]:
        """Return a local cache path and transfer/cache metrics for an ArtifactRef."""
        if ref.is_empty:
            raise ValueError("Cannot materialize empty ArtifactRef")

        started = time.perf_counter()
        digest = ref.sha256 or self._stable_ref_digest(ref)
        filename = os.path.basename(ref.uri) or filename_hint
        local_path = self.objects_dir / digest[:2] / digest / filename

        metrics: Dict[str, Any] = {
            "uri": ref.uri,
            "backend": ref.backend,
            "cache_hit": False,
            "bytes_from_local_cache": 0,
            "bytes_downloaded": 0,
            "materialize_time(s)": 0.0,
            "local_path": str(local_path),
        }

        if local_path.exists():
            self._touch(local_path)
            metrics["cache_hit"] = True
            metrics["bytes_from_local_cache"] = local_path.stat().st_size
            metrics["materialize_time(s)"] = time.perf_counter() - started
            return str(local_path), metrics

        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = str(local_path) + ".tmp"

        if ref.backend == "filesystem":
            if not os.path.exists(ref.uri):
                raise FileNotFoundError(ref.uri)
            shutil.copy2(ref.uri, tmp_path)
        elif ref.backend == "s3":
            bucket = ref.s3_bucket or self.default_s3_bucket
            if not bucket:
                raise ValueError("S3 artifact ref requires a bucket")
            s3.download_file(bucket, ref.uri, tmp_path)
        else:
            raise ValueError(f"Unsupported artifact backend: {ref.backend}")

        if ref.sha256:
            actual = sha256_file(tmp_path)
            if actual != ref.sha256:
                try:
                    os.remove(tmp_path)
                finally:
                    raise ValueError(
                        f"Hash mismatch for {ref.uri}: expected {ref.sha256}, got {actual}"
                    )

        os.replace(tmp_path, local_path)
        self._touch(local_path)
        size_bytes = local_path.stat().st_size
        eviction = self.evict_if_needed(protected_paths=[local_path])

        metrics["bytes_downloaded"] = size_bytes
        metrics["eviction_count"] = eviction["eviction_count"]
        metrics["evicted_bytes"] = eviction["evicted_bytes"]
        metrics["materialize_time(s)"] = time.perf_counter() - started
        return str(local_path), metrics

    def hydrate_setup_cache(self, shared_setup_ref: ArtifactRef, setup_key: str) -> str:
        path, _ = self.hydrate_setup_cache_with_metrics(shared_setup_ref, setup_key)
        return path

    def hydrate_setup_cache_with_metrics(
        self,
        shared_setup_ref: ArtifactRef,
        setup_key: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """Populate local setup cache and report per-file cache/transfer metrics."""
        started = time.perf_counter()
        local_setup_dir = Path(self.setup_dir(setup_key))
        metrics: Dict[str, Any] = {
            "setup_key": setup_key,
            "setup_cache_dir": str(local_setup_dir),
            "shared_setup_uri": shared_setup_ref.uri,
            "files": {},
            "bytes_from_local_cache": 0,
            "bytes_downloaded": 0,
            "pk_decompress_time(s)": 0.0,
            "hydrate_time(s)": 0.0,
        }

        if shared_setup_ref.is_empty:
            metrics["hydrate_time(s)"] = time.perf_counter() - started
            return str(local_setup_dir), metrics

        for filename in self.SETUP_FILES:
            local_path = local_setup_dir / filename
            file_metrics = {
                "cache_hit": False,
                "downloaded": False,
                "bytes": 0,
                "source": "",
            }

            if local_path.exists():
                size = local_path.stat().st_size
                file_metrics.update(cache_hit=True, bytes=size, source="local_cache")
                metrics["bytes_from_local_cache"] += size
                metrics["files"][filename] = file_metrics
                continue

            if filename == "pk.json":
                gz_path = str(local_path) + ".gz"
                if self._copy_or_download_optional(shared_setup_ref, "pk.json.gz", gz_path):
                    compressed_size = os.path.getsize(gz_path)
                    decompress_started = time.perf_counter()
                    with gzip.open(gz_path, "rb") as src, open(local_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    metrics["pk_decompress_time(s)"] += time.perf_counter() - decompress_started
                    os.remove(gz_path)
                    file_metrics.update(
                        downloaded=True,
                        bytes=compressed_size,
                        source="shared_setup_compressed",
                        raw_bytes=os.path.getsize(local_path),
                    )
                    metrics["bytes_downloaded"] += compressed_size
                    metrics["files"][filename] = file_metrics
                    continue

            downloaded = self._copy_or_download_optional(shared_setup_ref, filename, str(local_path))
            if downloaded:
                size = os.path.getsize(local_path)
                file_metrics.update(downloaded=True, bytes=size, source="shared_setup")
                metrics["bytes_downloaded"] += size
            metrics["files"][filename] = file_metrics

        self._touch(local_setup_dir)
        eviction = self.evict_if_needed(protected_paths=[local_setup_dir])
        metrics["eviction_count"] = eviction["eviction_count"]
        metrics["evicted_bytes"] = eviction["evicted_bytes"]
        metrics["hydrate_time(s)"] = time.perf_counter() - started
        return str(local_setup_dir), metrics

    def publish_setup_cache(self, local_setup_dir: str, shared_setup_ref: ArtifactRef) -> None:
        self.publish_setup_cache_with_metrics(local_setup_dir, shared_setup_ref)

    def publish_setup_cache_with_metrics(
        self,
        local_setup_dir: str,
        shared_setup_ref: ArtifactRef,
    ) -> Dict[str, Any]:
        """Publish local setup artifacts back to the shared setup prefix.

        The proving key is compressed before publishing. Local hot cache keeps
        the raw pk.json because EZKL expects that format.
        """
        started = time.perf_counter()
        metrics: Dict[str, Any] = {
            "shared_setup_uri": shared_setup_ref.uri,
            "files": {},
            "bytes_uploaded": 0,
            "pk_compress_time(s)": 0.0,
            "publish_time(s)": 0.0,
        }
        if shared_setup_ref.is_empty:
            metrics["publish_time(s)"] = time.perf_counter() - started
            return metrics

        local_dir = Path(local_setup_dir)
        if not local_dir.exists():
            metrics["publish_time(s)"] = time.perf_counter() - started
            return metrics

        for filename in self.SETUP_FILES:
            source = local_dir / filename
            if not source.exists():
                metrics["files"][filename] = {"uploaded": False, "bytes": 0, "reason": "missing"}
                continue

            if filename == "pk.json":
                compressed = local_dir / "pk.json.gz"
                compress_started = time.perf_counter()
                with open(source, "rb") as raw, gzip.open(compressed, "wb", compresslevel=6) as zipped:
                    shutil.copyfileobj(raw, zipped)
                metrics["pk_compress_time(s)"] += time.perf_counter() - compress_started
                compressed_size = compressed.stat().st_size
                try:
                    self._copy_or_upload(shared_setup_ref, "pk.json.gz", str(compressed))
                    metrics["files"][filename] = {
                        "uploaded": True,
                        "bytes": compressed_size,
                        "raw_bytes": source.stat().st_size,
                        "artifact": "pk.json.gz",
                    }
                    metrics["bytes_uploaded"] += compressed_size
                finally:
                    compressed.unlink(missing_ok=True)
            else:
                size = source.stat().st_size
                self._copy_or_upload(shared_setup_ref, filename, str(source))
                metrics["files"][filename] = {"uploaded": True, "bytes": size, "artifact": filename}
                metrics["bytes_uploaded"] += size

        eviction = self.evict_if_needed(protected_paths=[Path(local_setup_dir)])
        metrics["eviction_count"] = eviction["eviction_count"]
        metrics["evicted_bytes"] = eviction["evicted_bytes"]
        metrics["publish_time(s)"] = time.perf_counter() - started
        return metrics

    def evict_if_needed(self, protected_paths: Optional[Iterable[Path]] = None) -> Dict[str, int]:
        if self.max_bytes <= 0:
            self.last_eviction = {"eviction_count": 0, "evicted_bytes": 0}
            return self.last_eviction

        protected = {p.resolve() for p in (protected_paths or [])}
        eviction_count = 0
        evicted_bytes = 0

        while self.total_bytes() > self.max_bytes:
            candidates = []
            for child in list(self.objects_dir.glob("*/*")) + list(self.setup_dir_root.glob("*")):
                if not child.exists():
                    continue
                resolved = child.resolve()
                if any(str(resolved).startswith(str(item)) for item in protected):
                    continue
                try:
                    stat = child.stat()
                except FileNotFoundError:
                    continue
                candidates.append((stat.st_mtime, child))

            if not candidates:
                break

            _, victim = min(candidates, key=lambda item: item[0])
            victim_bytes = self._path_size(victim)
            if victim.is_dir():
                shutil.rmtree(victim, ignore_errors=True)
            else:
                victim.unlink(missing_ok=True)
            eviction_count += 1
            evicted_bytes += victim_bytes

        self.last_eviction = {"eviction_count": eviction_count, "evicted_bytes": evicted_bytes}
        return self.last_eviction

    def _copy_or_download_optional(self, prefix_ref: ArtifactRef, filename: str, local_path: str) -> bool:
        if prefix_ref.backend == "filesystem":
            source = os.path.join(prefix_ref.uri, filename)
            if not os.path.exists(source):
                return False
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            shutil.copy2(source, local_path)
            return True

        if prefix_ref.backend == "s3":
            bucket = prefix_ref.s3_bucket or self.default_s3_bucket
            key = f"{prefix_ref.uri.rstrip('/')}/{filename}"
            if not bucket or not s3.exists(bucket, key):
                return False
            s3.download_file(bucket, key, local_path)
            return True

        return False

    def _copy_or_upload(self, prefix_ref: ArtifactRef, filename: str, local_path: str) -> None:
        if prefix_ref.backend == "filesystem":
            destination = os.path.join(prefix_ref.uri, filename)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copy2(local_path, destination)
            return

        if prefix_ref.backend == "s3":
            bucket = prefix_ref.s3_bucket or self.default_s3_bucket
            if not bucket:
                raise ValueError("S3 setup cache publish requires bucket")
            key = f"{prefix_ref.uri.rstrip('/')}/{filename}"
            s3.upload_file(local_path, bucket, key)
            return

        raise ValueError(f"Unsupported setup cache backend: {prefix_ref.backend}")

    @staticmethod
    def _path_size(path: Path) -> int:
        if path.is_file():
            try:
                return path.stat().st_size
            except FileNotFoundError:
                return 0
        total = 0
        for item in path.rglob("*") if path.exists() else []:
            if item.is_file():
                try:
                    total += item.stat().st_size
                except FileNotFoundError:
                    continue
        return total

    @staticmethod
    def _stable_ref_digest(ref: ArtifactRef) -> str:
        import hashlib

        raw = f"{ref.backend}|{ref.s3_bucket}|{ref.uri}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _touch(path: Path) -> None:
        now = time.time()
        try:
            os.utime(path, (now, now))
        except FileNotFoundError:
            pass
