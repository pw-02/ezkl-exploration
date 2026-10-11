from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

from zkinfer.runtime.data_plane import ArtifactRef
from zkinfer.storage import s3


_CHUNK_SIZE = 8 * 1024 * 1024


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def object_key(root: str, kind: str, digest: str, filename: str) -> str:
    kind = (kind or "blob").strip("/")
    filename = filename or "blob"
    return f"{root.rstrip('/')}/objects/{kind}/{digest[:2]}/{digest}/{filename}".lstrip("/")


def _atomic_copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=os.path.dirname(dst))
    os.close(fd)
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _atomic_write_bytes(data: bytes, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=os.path.dirname(dst))
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(data)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


class ArtifactStore:
    """Content-addressed data-plane store.

    The coordinator and workers exchange ArtifactRef metadata over gRPC.  Large
    bytes live here, keyed by SHA-256 so immutable artifacts can be reused across
    requests and verified after transfer.
    """

    def __init__(
        self,
        backend: str,
        root: str,
        s3_bucket: Optional[str] = None,
    ):
        if backend not in {"filesystem", "s3"}:
            raise ValueError(f"Unsupported artifact backend: {backend}")
        if backend == "s3" and not s3_bucket:
            raise ValueError("s3_bucket is required for S3 artifact store")

        self.backend = backend
        self.root = root.rstrip("/")
        self.s3_bucket = s3_bucket or ""

        if self.backend == "filesystem":
            Path(self.root).mkdir(parents=True, exist_ok=True)

    def put_file(
        self,
        path: str,
        kind: str,
        filename: Optional[str] = None,
        compress: bool = False,
    ) -> ArtifactRef:
        source_path = path
        cleanup_path = None
        object_filename = filename or os.path.basename(path)

        if compress:
            object_filename = f"{object_filename}.gz" if not object_filename.endswith(".gz") else object_filename
            cleanup_path = f"{path}.gz.tmp"
            with open(path, "rb") as raw, gzip.open(cleanup_path, "wb", compresslevel=6) as zipped:
                shutil.copyfileobj(raw, zipped)
            source_path = cleanup_path

        try:
            digest = sha256_file(source_path)
            size_bytes = os.path.getsize(source_path)
            uri = object_key(self.root, kind, digest, object_filename)

            if self.backend == "filesystem":
                if not os.path.exists(uri):
                    _atomic_copy(source_path, uri)
            else:
                if not s3.exists(self.s3_bucket, uri):
                    s3.upload_file(source_path, self.s3_bucket, uri)

            return ArtifactRef(
                backend=self.backend,
                uri=uri,
                s3_bucket=self.s3_bucket,
                sha256=digest,
                size_bytes=size_bytes,
            )
        finally:
            if cleanup_path and os.path.exists(cleanup_path):
                os.remove(cleanup_path)

    def put_bytes(self, data: bytes, kind: str, filename: str) -> ArtifactRef:
        digest = sha256_bytes(data)
        uri = object_key(self.root, kind, digest, filename)

        if self.backend == "filesystem":
            if not os.path.exists(uri):
                _atomic_write_bytes(data, uri)
        else:
            if not s3.exists(self.s3_bucket, uri):
                s3.put_bytes(data, self.s3_bucket, uri)

        return ArtifactRef(
            backend=self.backend,
            uri=uri,
            s3_bucket=self.s3_bucket,
            sha256=digest,
            size_bytes=len(data),
        )

    def put_json(self, obj: Any, kind: str, filename: str) -> ArtifactRef:
        data = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return self.put_bytes(data, kind=kind, filename=filename)

    def get_file(self, ref: ArtifactRef, local_path: str) -> str:
        if ref.is_empty:
            raise ValueError("Cannot materialize empty ArtifactRef")

        if ref.backend == "filesystem":
            if not os.path.exists(ref.uri):
                raise FileNotFoundError(ref.uri)
            source_path = ref.uri
            if os.path.abspath(source_path) != os.path.abspath(local_path):
                _atomic_copy(source_path, local_path)
        elif ref.backend == "s3":
            bucket = ref.s3_bucket or self.s3_bucket
            if not bucket:
                raise ValueError("S3 ArtifactRef missing bucket")
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(bucket, ref.uri, local_path)
        else:
            raise ValueError(f"Unsupported ArtifactRef backend: {ref.backend}")

        if ref.sha256:
            actual = sha256_file(local_path)
            if actual != ref.sha256:
                raise ValueError(
                    f"Artifact hash mismatch for {ref.uri}: expected {ref.sha256}, got {actual}"
                )

        return local_path

    def get_json(self, ref: ArtifactRef) -> Any:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "artifact.json")
            self.get_file(ref, path)
            with open(path, "r", encoding="utf-8") as file:
                return json.load(file)
