import re
import json

import re
import json

# Function to parse the input file and convert it to a list of dictionaries
def parse_table_to_json_working(file_path):
    with open(file_path, 'r') as f:
        file_content = f.read()
    # Split the file content into lines
    lines = file_content.split('\n')
    lines = lines[1:-1]
    # Initialize variables
    data = []
    headers = None

    # Iterate through lines
    for line in lines:
        # Strip leading and trailing whitespace
        line = line.strip()

        line = line.replace(' ','')
        if not '├─────┼' in line:
  

            split_list = line.split("│")
            split_list = split_list[1:-1]
            if split_list[0] != 'idx':
                row_dict = {"idx": split_list[0],
                            "opkind": split_list[1],
                            "out_scale": split_list[2],
                            "inputs": split_list[3],
                            "out_dims": split_list[4]
                }
                data.append(row_dict)

    return data

# Function to parse the input file and convert it to a list of dictionaries
def parse_table_to_json(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # lines = lines[1:-1]
    # Initialize variables
    data = []
    headers = None

    # Process each line
    for line in lines:
        # Split each line by '|' and strip whitespace
        parts = line.split('|')
        parts = parts[1:]

        # Extract relevant information from parts
        index = parts[0].replace(' ','')

        if index != '':
            operation = parts[2].split()[0]  # Extracts the operation (->, 0/0>, etc.)
            name = parts[2].split()[1]       # Extracts the name (e.g., Conv, Const, Max, etc.)
            inputs = parts[2].split()[2]
            out_dims = parts[2].split()[4]

            # Create a dictionary for the current entry
            entry = {
                'index': index,
                'operation': operation,
                'name': name,
                'inputs': inputs,
                'out_dims': out_dims
            }
        else:
            pass

    return data

# Path to your text file
file_path = 'nnef_to_onnx/table.txt'

# Parse the table and convert it to JSON
parsed_data = parse_table_to_json(file_path)

# Convert to JSON string
json_data = json.dumps(parsed_data, indent=4)

# Print the JSON output
print(json_data)
