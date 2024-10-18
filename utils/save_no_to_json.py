import numpy as np
import json

# Load the data from the .npy file
data = np.load('utils\data.npy', allow_pickle=True)  # Use allow_pickle=True if the data contains objects

# Convert the numpy array to a list (or appropriate format) for JSON serialization
data_list = data.tolist()  # or use data.item() if it's a single object
flattened_list = [item for sublist in data_list for item in sublist]
flattened_list = [item for sublist in flattened_list for item in sublist]
flattened_list = [item for sublist in flattened_list for item in sublist]

# Save the data to a JSON file
with open('data.json', 'w') as json_file:
    json.dump(flattened_list, json_file)

print("Data has been successfully loaded from 'data.npy' and saved to 'data.json'.")
