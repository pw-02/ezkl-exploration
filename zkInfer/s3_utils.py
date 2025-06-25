import boto3
import json
def upload_to_s3(local_path, bucket, s3_key):
    s3 = boto3.client('s3')
    s3.upload_file(local_path, bucket, s3_key)

def download_from_s3(bucket, s3_key, local_path):
    s3 = boto3.client('s3')
    s3.download_file(bucket, s3_key, local_path)

def file_exists_in_s3(bucket, s3_key):
    s3 = boto3.client('s3')
    try:
        s3.head_object(Bucket=bucket, Key=s3_key)
        return True
    except:
        return False

def upload_if_not_exists(local_path, bucket, s3_key):
    if not file_exists_in_s3(bucket, s3_key):
        upload_to_s3(local_path, bucket, s3_key)
        return True
    return False

def upload_modelproto_to_s3(model_proto, bucket, s3_key):
    # Serialize ModelProto to bytes
    raw_bytes = model_proto.SerializeToString()
    s3 = boto3.client('s3')
    s3.put_object(Body=raw_bytes, Bucket=bucket, Key=s3_key, ContentType="application/octet-stream")

def download_modelproto_from_s3(bucket, s3_key):
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket, Key=s3_key)
    raw_bytes = response['Body'].read()
    
    # Deserialize bytes to ModelProto
    from onnx import ModelProto
    model_proto = ModelProto()
    model_proto.ParseFromString(raw_bytes)
    
    return model_proto

def upload_modelproto_if_not_exists(model_proto, bucket, s3_key):
    if not file_exists_in_s3(bucket, s3_key):
        upload_modelproto_to_s3(model_proto, bucket, s3_key)
        return True
    return False

def upload_json_to_s3(data, bucket, s3_key):
    s3 = boto3.client('s3')
    s3.put_object(Body=json.dumps(data, indent=4), Bucket=bucket, Key=s3_key, ContentType="application/json")

def download_json_from_s3(bucket, s3_key):
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket, Key=s3_key)
    raw_data = response['Body'].read().decode('utf-8')
    return json.loads(raw_data)