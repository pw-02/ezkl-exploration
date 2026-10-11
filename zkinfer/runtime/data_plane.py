from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ArtifactRef:
    backend: str
    uri: str
    s3_bucket: str = ""
    sha256: str = ""
    size_bytes: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.backend or not self.uri

    @classmethod
    def empty(cls) -> "ArtifactRef":
        return cls(backend="", uri="")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ArtifactRef":
        if not data:
            return cls.empty()
        return cls(
            backend=str(data.get("backend") or ""),
            uri=str(data.get("uri") or ""),
            s3_bucket=str(data.get("s3_bucket") or ""),
            sha256=str(data.get("sha256") or ""),
            size_bytes=int(data.get("size_bytes") or 0),
        )
