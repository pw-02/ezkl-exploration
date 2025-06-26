import boto3
import json

def s3_path(key_or_prefix: str) -> str:
    """
    Converts Windows-style paths to S3-friendly keys by replacing backslashes with forward slashes.
    """
    return key_or_prefix.replace("\\", "/")

def upload_to_s3(local_path, bucket, s3_key):
    s3 = boto3.client('s3')
    s3.upload_file(local_path, bucket, s3_path(s3_key))

def download_from_s3(bucket, s3_key, local_path):
    s3 = boto3.client('s3')
    s3.download_file(bucket, s3_path(s3_key), local_path)

def file_exists_in_s3(bucket, s3_key):
    s3 = boto3.client('s3')
    try:
        s3.head_object(Bucket=bucket, Key=s3_path(s3_key))
        return True
    except:
        return False

def upload_if_not_exists(local_path, bucket, s3_key):
    if not file_exists_in_s3(bucket, s3_key):
        upload_to_s3(local_path, bucket, s3_key)
        return True
    return False

def upload_modelproto_to_s3(raw_bytes, bucket, s3_key):
    s3 = boto3.client('s3')
    s3.put_object(Body=raw_bytes, Bucket=bucket, Key=s3_path(s3_key), ContentType="application/octet-stream")

def download_modelproto_from_s3(bucket, s3_key):
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket, Key=s3_path(s3_key))
    raw_bytes = response['Body'].read()
    from onnx import ModelProto
    model_proto = ModelProto()
    model_proto.ParseFromString(raw_bytes)
    return model_proto

def upload_modelproto_if_not_exists(raw_bytes, bucket, s3_key):
    if not file_exists_in_s3(bucket, s3_key):
        upload_modelproto_to_s3(raw_bytes, bucket, s3_key)
        return True
    return False

def upload_json_to_s3(data, bucket, s3_key):
    s3 = boto3.client('s3')
    s3.put_object(Body=json.dumps(data, indent=4), Bucket=bucket, Key=s3_path(s3_key), ContentType="application/json")

def download_json_from_s3(bucket, s3_key):
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket, Key=s3_path(s3_key))
    raw_data = response['Body'].read().decode('utf-8')
    return json.loads(raw_data)

def delete_s3_prefix(s3_bucket: str, prefix: str):
    """
    Deletes all objects in S3 bucket with the given prefix ("directory").
    """
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    deleted = 0
    for page in paginator.paginate(Bucket=s3_bucket, Prefix=s3_path(prefix)):
        objects = page.get('Contents', [])
        if not objects:
            continue
        delete_keys = {'Objects': [{'Key': obj['Key']} for obj in objects]}
        response = s3.delete_objects(Bucket=s3_bucket, Delete=delete_keys)
        deleted += len(response.get('Deleted', []))

def delete_s3_file(s3_bucket: str, s3_key: str):
    """
    Deletes a single file (object) from S3.
    """
    s3 = boto3.client('s3')
    try:
        s3.delete_object(Bucket=s3_bucket, Key=s3_path(s3_key))
    except Exception as e:
        pass

def delete_non_onnx_files_from_s3(bucket: str, prefix: str):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=s3_path(prefix)):
        objects = page.get('Contents', [])
        for obj in objects:
            key = obj['Key']
            if not key.endswith(".onnx"):
                s3.delete_object(Bucket=bucket, Key=key)

def download_if_exists_in_s3(bucket: str, s3_key: str, local_path: str):
    """
    Downloads a file from S3 if it exists.
    """
    if file_exists_in_s3(bucket, s3_key):
        download_from_s3(bucket, s3_key, local_path)
        return True
    return False
