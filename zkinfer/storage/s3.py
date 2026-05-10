import json
import os
from typing import Any, Optional

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError


DEFAULT_TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=8 * 1024 * 1024,
    max_concurrency=8,
    multipart_chunksize=8 * 1024 * 1024,
    use_threads=True,
)


def normalize_key(key: str) -> str:
    return key.replace("\\", "/")


def client():
    return boto3.client("s3")


def exists(bucket: str, key: str) -> bool:
    try:
        client().head_object(Bucket=bucket, Key=normalize_key(key))
        return True
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def upload_file(local_path: str, bucket: str, key: str) -> None:
    client().upload_file(
        local_path,
        bucket,
        normalize_key(key),
        Config=DEFAULT_TRANSFER_CONFIG,
    )


def download_file(bucket: str, key: str, local_path: str) -> None:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    client().download_file(
        bucket,
        normalize_key(key),
        local_path,
        Config=DEFAULT_TRANSFER_CONFIG,
    )


def download_if_exists(bucket: str, key: str, local_path: str) -> bool:
    if not exists(bucket, key):
        return False
    download_file(bucket, key, local_path)
    return True


def put_bytes(data: bytes, bucket: str, key: str, content_type: str = "application/octet-stream") -> None:
    client().put_object(
        Body=data,
        Bucket=bucket,
        Key=normalize_key(key),
        ContentType=content_type,
    )


def get_bytes(bucket: str, key: str) -> bytes:
    response = client().get_object(Bucket=bucket, Key=normalize_key(key))
    return response["Body"].read()


def put_json(data: Any, bucket: str, key: str) -> None:
    put_bytes(
        json.dumps(data, indent=4).encode("utf-8"),
        bucket,
        key,
        content_type="application/json",
    )


def get_json(bucket: str, key: str) -> Any:
    return json.loads(get_bytes(bucket, key).decode("utf-8"))


def delete_file(bucket: str, key: str) -> None:
    client().delete_object(Bucket=bucket, Key=normalize_key(key))


def delete_prefix(bucket: str, prefix: str) -> int:
    s3 = client()
    paginator = s3.get_paginator("list_objects_v2")
    deleted = 0

    for page in paginator.paginate(Bucket=bucket, Prefix=normalize_key(prefix)):
        objects = page.get("Contents", [])
        if not objects:
            continue

        response = s3.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": obj["Key"]} for obj in objects]},
        )
        deleted += len(response.get("Deleted", []))

    return deleted