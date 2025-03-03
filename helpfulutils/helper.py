import json
import numpy as np


def read_json_file_to_string(file_path):
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    return json.dumps(json_data, indent=4)  # Convert JSON object to a pretty-printed string

def read_json_file_to_dict(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


