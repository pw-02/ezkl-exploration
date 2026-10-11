import csv
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict

import onnx
import pandas as pd

from zkinfer.storage import s3


_CSV_LOCKS: dict[str, threading.Lock] = {}
_CSV_LOCKS_GUARD = threading.Lock()


def _csv_lock(file_path: str) -> threading.Lock:
    key = str(Path(file_path).resolve())
    with _CSV_LOCKS_GUARD:
        return _CSV_LOCKS.setdefault(key, threading.Lock())


def compute_file_md5_hex(path: str) -> str:
    md5 = hashlib.md5()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def compute_bytes_md5_hex(raw_bytes: bytes) -> str:
    md5 = hashlib.md5()
    md5.update(raw_bytes)
    return md5.hexdigest()


def exists(path_or_key: str, storage_type: str = "filesystem", s3_bucket: str | None = None) -> bool:
    if storage_type == "s3":
        if not s3_bucket:
            raise ValueError("s3_bucket is required when storage_type='s3'")
        return s3.exists(s3_bucket, path_or_key)
    return os.path.exists(path_or_key)


def load_model_proto(path_or_key: str, storage_type: str = "filesystem", s3_bucket: str | None = None):
    if storage_type == "s3":
        if not s3_bucket:
            raise ValueError("s3_bucket is required when storage_type='s3'")
        raw_bytes = s3.get_bytes(s3_bucket, path_or_key)
        model_proto = onnx.ModelProto()
        model_proto.ParseFromString(raw_bytes)
        return model_proto
    return onnx.load(path_or_key)


def save_model_proto(model_proto, path_or_key: str, storage_type: str = "filesystem", s3_bucket: str | None = None) -> None:
    raw_bytes = model_proto.SerializeToString()
    if storage_type == "s3":
        if not s3_bucket:
            raise ValueError("s3_bucket is required when storage_type='s3'")
        s3.put_bytes(raw_bytes, s3_bucket, path_or_key)
        return
    parent = os.path.dirname(path_or_key)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path_or_key, "wb") as file:
        file.write(raw_bytes)


def load_json(path_or_key: str, storage_type: str = "filesystem", s3_bucket: str | None = None) -> Any:
    if storage_type == "s3":
        if not s3_bucket:
            raise ValueError("s3_bucket is required when storage_type='s3'")
        return s3.get_json(s3_bucket, path_or_key)
    with open(path_or_key, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Any, path_or_key: str, storage_type: str = "filesystem", s3_bucket: str | None = None) -> None:
    if storage_type == "s3":
        if not s3_bucket:
            raise ValueError("s3_bucket is required when storage_type='s3'")
        s3.put_json(data, s3_bucket, path_or_key)
        return
    parent = os.path.dirname(path_or_key)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path_or_key, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def remove(path_or_key: str, storage_type: str = "filesystem", s3_bucket: str | None = None) -> None:
    if storage_type == "s3":
        if not s3_bucket:
            raise ValueError("s3_bucket is required when storage_type='s3'")
        s3.delete_file(s3_bucket, path_or_key)
        return
    if os.path.exists(path_or_key):
        os.remove(path_or_key)


def write_dict_to_csv(data: Dict, file_path: str) -> None:
    """Append one row while safely allowing metrics to gain new columns.

    Stage-aware metrics are intentionally dynamic: different jobs can observe
    different stages. A plain ``DictWriter(..., fieldnames=data.keys())`` would
    corrupt an existing CSV when later rows have a different key order/schema.
    When new columns appear, rewrite the small report CSV with the union schema;
    otherwise append normally. Calls are serialized per file inside the
    coordinator process.
    """
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with _csv_lock(file_path):
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            fieldnames = list(data.keys())
            with open(file_path, "w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(data)
            return

        with open(file_path, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            existing_fields = list(reader.fieldnames or [])
            new_fields = [key for key in data.keys() if key not in existing_fields]

            if not new_fields:
                with open(file_path, "a", newline="", encoding="utf-8") as append_file:
                    writer = csv.DictWriter(
                        append_file,
                        fieldnames=existing_fields,
                        extrasaction="ignore",
                    )
                    writer.writerow(data)
                return

            rows = list(reader)

        fieldnames = existing_fields + new_fields
        tmp_path = file_path + ".rewrite.tmp"
        with open(tmp_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            writer.writerow(data)
        os.replace(tmp_path, file_path)


def read_csv_as_dict(csv_file: str) -> Dict:
    if not os.path.exists(csv_file):
        return {}
    df = pd.read_csv(csv_file)
    return df.to_dict(orient="list")
