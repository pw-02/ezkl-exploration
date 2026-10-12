"""Shared storage organised by setup lifetime and request lifetime."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from zkinfer.runtime.data_plane import ArtifactRef
from zkinfer.storage import s3


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _component(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"Expected a single storage path component: {value!r}")
    return value


class SharedStorage:
    """Reusable setup bundles and request-owned inputs/proof outputs.

    ArtifactRef remains the control-plane transport type; it does not determine
    the lifetime or placement of the bytes it references.
    """

    def __init__(self, backend: str, setup_root: str, requests_root: str, s3_bucket: str = ""):
        if backend not in {"filesystem", "s3"}:
            raise ValueError(f"Unsupported storage backend: {backend}")
        if backend == "s3" and not s3_bucket:
            raise ValueError("S3 storage requires a bucket")
        self.backend = backend
        self.setup_root = setup_root.rstrip("/")
        self.requests_root = requests_root.rstrip("/")
        self.s3_bucket = s3_bucket or ""
        if backend == "filesystem":
            for root in (self.setup_root, self.requests_root):
                Path(root).mkdir(parents=True, exist_ok=True)

    def prefix_ref(self, uri: str) -> ArtifactRef:
        return ArtifactRef(self.backend, uri, s3_bucket=self.s3_bucket)

    def setup_ref(self, setup_key: str) -> ArtifactRef:
        return self.prefix_ref(os.path.join(self.setup_root, _component(setup_key)))

    def proof_prefix(self, request_id: str, partition_id: str) -> ArtifactRef:
        return self.prefix_ref(os.path.join(
            self.requests_root, _component(request_id), "proofs", _component(partition_id),
        ))

    def put_model(self, model_bytes: bytes) -> ArtifactRef:
        setup_key = hashlib.sha256(model_bytes).hexdigest()
        uri = os.path.join(self.setup_ref(setup_key).uri, "model.onnx")
        return self._put_bytes(model_bytes, uri)

    def put_input(self, request_id: str, partition_id: str, obj: Any) -> ArtifactRef:
        uri = os.path.join(
            self.requests_root, _component(request_id), "inputs",
            _component(partition_id) + ".json",
        )
        data = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return self._put_bytes(data, uri, validate_existing=True)

    def _put_bytes(self, data: bytes, uri: str, validate_existing: bool = False) -> ArtifactRef:
        digest = hashlib.sha256(data).hexdigest()
        if self.backend == "filesystem":
            if Path(uri).exists():
                if validate_existing and sha256_file(uri) != digest:
                    raise ValueError(f"Immutable storage object has different content: {uri}")
            else:
                Path(uri).parent.mkdir(parents=True, exist_ok=True)
                fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=Path(uri).parent)
                try:
                    with os.fdopen(fd, "wb") as file:
                        file.write(data)
                    os.replace(tmp, uri)
                finally:
                    if os.path.exists(tmp):
                        os.remove(tmp)
        elif s3.exists(self.s3_bucket, uri):
            if validate_existing and hashlib.sha256(s3.get_bytes(self.s3_bucket, uri)).hexdigest() != digest:
                raise ValueError(f"Immutable storage object has different content: {uri}")
        else:
            s3.put_bytes(data, self.s3_bucket, uri)
        return ArtifactRef(
            self.backend, uri, s3_bucket=self.s3_bucket, sha256=digest, size_bytes=len(data),
        )
