import json
import re

# Step 1: Read the content of the file
with open('nnef_to_onnx/new 16.txt', 'r') as file:
    data = file.read()

# Step 2: Preprocess the string to convert it into a valid JSON format

# Replace 'Graph {' with '{"Graph": {'
data = data.replace('Graph {', '{"Graph": {')

# Replace 'Node {' with '{"Node": {'
data = data.replace('Node', '')

# Replace 'op: ' with '"op": ' and other key names to be JSON compatible
data = re.sub(r'(\b\w+):', r'"\1":', data)

# Function to replace matches with quoted versions
def replace_with_quotes(match):
    return f'"{match.group(1)}"'

# Regular expression to find blocks like TypedSource { ... }
pattern = re.compile(r'(\b\w+\s*{\s*[^{}]*})')

# Replace all matches with the quoted version
data = pattern.sub(replace_with_quotes, data)


pattern2 = re.compile(r' \s*([\w]+\([^)]*\))')
data = pattern2.sub(replace_with_quotes, data)

# # Regular expression to find terms like Conv { ... } with nested braces
# pattern3 = re.compile(r'"op":\s*([\w]+\s*{[^{}]*(?:{[^{}]*}[^{}]*)*})')
# data = pattern3.sub(lambda m: f'"op": "{m.group(1)}"', data)
#Regular expression to find lists
pattern4 = re.compile(r'(\[[^\[\]]*\])')
data = pattern4.sub(lambda m: f'"{m.group(1)}"', data)

# Correct specific cases of nested structures like Conv {...}
def replace_nested_structures(match):
    content = match.group(1)
    # Replace any remaining nested braces
    content = re.sub(r'{\s*"', '{ "', content)
    return f'"{content}"'

pattern5 = re.compile(r'("op":\s*"[^"]*{[^{}]*})')
data = pattern5.sub(replace_nested_structures, data)
# Replace '>, ' with '", ' to close JSON objects properly
# data = data.replace('>,', '",')

# Replace '}' with '}', but only when it's not followed by a comma
data = re.sub(r'}(\s*[,|\]])', r'}\1', data)

# Replace > with " to close JSON keys properly
# data = data.replace('>', '"')

data = data.replace('":"', '": "')

data = data.replace('\n','')
data = data + '}'
# Step 3: Parse the preprocessed string into a JSON object
try:
    json_data = json.loads(data)
    print(json.dumps(json_data, indent=2))  # Pretty print the JSON object
except json.JSONDecodeError as e:
    print(f"Failed to parse JSON: {e}")
