import hashlib
import base64
import json

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def compute_content_md5(path):
    md5 = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5.update(chunk)
    # S3 expects base64 encoding of the raw digest
    return base64.b64encode(md5.digest()).decode('utf-8')