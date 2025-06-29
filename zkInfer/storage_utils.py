
import csv
import os
import json
from typing import Dict
import boto3
from botocore.exceptions import ClientError
import pandas as pd
import onnx
from boto3.s3.transfer import TransferConfig


# def load_json(path):
#     with open(path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     return data

def compute_content_md5_hex(path):
    md5 = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5.update(chunk)
    # Hex digest is always safe for folder/file names
    return md5.hexdigest()
import hashlib

def compute_bytes_md5_hex(raw_bytes: bytes) -> str:
    """
    Compute the MD5 hash of a bytes object, returning a hex digest (safe for filesystem use).
    """
    md5 = hashlib.md5()
    md5.update(raw_bytes)
    return md5.hexdigest()



def load_model_proto(path_or_key, use_s3=False, s3_bucket=None):
    """
    Load an ONNX model from a local path or S3 key.
    If use_s3 is True, it will download the model from S3.
    """
    if use_s3:
        s3 = boto3.client("s3")
        try:
            response = s3.get_object(Bucket=s3_bucket, Key=s3_path(path_or_key))
            raw_bytes = response['Body'].read()
            model_proto = onnx.ModelProto()
            model_proto.ParseFromString(raw_bytes)
            return model_proto
        except ClientError as e:
            raise RuntimeError(f"Failed to download {path_or_key} from S3 bucket {s3_bucket}: {e}")
    else:
        return onnx.load(path_or_key)



def s3_path(key_or_prefix: str) -> str:
    """
    Converts Windows-style paths to S3-friendly keys by replacing backslashes with forward slashes.
    """
    return key_or_prefix.replace("\\", "/")

def convert_csv_to_dict(csv_file, start_timestamp = None, end_timestamp = None):
    df = pd.read_csv(csv_file)
    return df.to_dict(orient='list')

def write_dict_to_csv(data: Dict, file_path: str):
        """Write a dictionary to a CSV file."""
        with open(file_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(data)

def upload_to_s3(local_path, bucket, s3_key):
    s3 = boto3.client('s3')
    s3.upload_file(local_path, bucket, s3_path(s3_key))

def download_from_s3(bucket, s3_key, local_path, max_concurrency=8):
    s3 = boto3.client('s3')
    config = TransferConfig(
        multipart_threshold=8 * 1024 * 1024,    # Start multipart for files > 8MB
        max_concurrency=max_concurrency,        # Number of threads
        multipart_chunksize=8 * 1024 * 1024,    # Size per chunk (8MB)
        use_threads=True,
    )
    s3.download_file(bucket, s3_key, local_path, Config=config)


def file_exists_in_s3(bucket, s3_key):
    s3 = boto3.client('s3')
    try:
        s3.head_object(Bucket=bucket, Key=s3_path(s3_key))
        return True
    except:
        return False

def download_if_exists_in_s3(bucket: str, s3_key: str, local_path: str):
    """
    Downloads a file from S3 if it exists.
    """
    if file_exists_in_s3(bucket, s3_key):
        download_from_s3(bucket, s3_key, local_path)
        return True
    return False
            
def s3_file_exists(s3_bucket, key):
    s3 = boto3.client("s3")
    try:
        s3.head_object(Bucket=s3_bucket, Key=s3_path(key))
        return True
    except ClientError as e:
        # If error code is 404, the object does not exist.
        if e.response["Error"]["Code"] == "404":
            return False
        # Otherwise, re-raise (could be permissions, etc.)
        raise
    
def s3_download_file_to_string(s3_bucket, key):
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=s3_bucket, Key=s3_path(key))
        return obj['Body'].read().decode('utf-8')
    except ClientError as e:
        raise RuntimeError(f"Failed to download {key} from S3 bucket {s3_bucket}: {e}")


def file_exists(path_or_key, use_s3=False, s3_bucket=None):
    if use_s3:
        # Use boto3 or your S3 client to check
        return s3_file_exists(s3_bucket, path_or_key)
    else:
        return os.path.exists(path_or_key)
    
def save_model_proto_file(model_proto, path_or_key, use_s3=False, s3_bucket=None):
    if use_s3:
        # Serialize model_proto to bytes and upload to S3
        raw_bytes = model_proto.SerializeToString()
        s3 = boto3.client("s3")
        s3.put_object(Body=raw_bytes, Bucket=s3_bucket, Key=s3_path(path_or_key), ContentType="application/octet-stream")
    else:
        #ensure the directory exists
        os.makedirs(os.path.dirname(path_or_key), exist_ok=True)
        with open(path_or_key, "wb") as f:
            f.write(model_proto.SerializeToString())
    
def load_json_file(path_or_key, use_s3=False, s3_bucket=None):
    if use_s3:
        # Download file to memory from S3, then parse JSON
        content = s3_download_file_to_string(s3_bucket, path_or_key)
        return json.loads(content)
    else:
        with open(path_or_key, "r") as f:
            return json.load(f)
        
def upload_to_s3(local_path, bucket, s3_key, max_concurrency=8):
    s3 = boto3.client('s3')
    config = TransferConfig(
        multipart_threshold=8 * 1024 * 1024,    # Files >8MB use multipart
        max_concurrency=max_concurrency,        # Number of threads
        multipart_chunksize=8 * 1024 * 1024,    # Chunk size (8MB)
        use_threads=True,
    )
    s3.upload_file(local_path, bucket, s3_key, Config=config)


# def save_large_json_to_s3(data, bucket, key, max_concurrency=8):
#     s3 = boto3.client("s3")
#     config = TransferConfig(
#         multipart_threshold=8 * 1024 * 1024,
#         max_concurrency=max_concurrency,
#         multipart_chunksize=8 * 1024 * 1024,
#         use_threads=True,
#     )
#     with tempfile.NamedTemporaryFile("w", delete=False) as f:
#         json.dump(data, f, separators=(",", ":"))  # compact!
#         f.flush()
#         s3.upload_file(
#             f.name,
#             bucket,
#             key,
#             ExtraArgs={"ContentType": "application/json"},
#             Config=config
#         )



def save_json_file(data, path_or_key, use_s3=False, s3_bucket=None):
    if use_s3:
        # Serialize data to JSON and upload to S3
        json_str = json.dumps(data, indent=4)
        s3 = boto3.client("s3")
        s3.put_object(Body=json_str.encode('utf-8'), Bucket=s3_bucket, Key=s3_path(path_or_key), ContentType="application/json")
    else:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path_or_key), exist_ok=True)
        with open(path_or_key, "w") as f:
            json.dump(data, f, indent=4)

