import csv
import hashlib
import base64
import json
import re

from pyparsing import Dict

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

def compute_bytes_md5(raw_bytes: bytes) -> str:
    """
    Compute the MD5 hash of a bytes object, returning a base64-encoded digest (like S3 Content-MD5).
    """
    md5 = hashlib.md5()
    md5.update(raw_bytes)
    return base64.b64encode(md5.digest()).decode('utf-8')

def write_dict_to_csv(data: Dict, file_path: str):
        """Write a dictionary to a CSV file."""
        with open(file_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(data)

def parse_resource_usage_file(log_path):
    max_system_mem = 0.0
    max_process_mem = 0.0

    system_pattern = re.compile(r"TYPE:system.*MEM: ([\d\.]+)GB")
    process_pattern = re.compile(r"TYPE:process.*MEM: ([\d\.]+)GB")

    with open(log_path, "r") as f:
        for line in f:
            sys_match = system_pattern.search(line)
            proc_match = process_pattern.search(line)
            if sys_match:
                mem_gb = float(sys_match.group(1))
                if mem_gb > max_system_mem:
                    max_system_mem = mem_gb
            elif proc_match:
                mem_gb = float(proc_match.group(1))
                if mem_gb > max_process_mem:
                    max_process_mem = mem_gb

    return max_process_mem, max_system_mem