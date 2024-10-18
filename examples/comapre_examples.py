import os
import ezkl
import json
import time
def get_examples():
    examples_path = 'examples'
    EXAMPLES_OMIT = [
        # these are too large
        'mobilenet_large',
        'mobilenet',
        'doodles',
        'nanoGPT',
        "self_attention",
        'multihead_attention',
        'large_op_graph',
        '1l_instance_norm',
        'variable_cnn',
        'accuracy',
        'linear_regression',
        "mnist_gan",
        'residual_block',
        'residual_block_split',
        'network_split_0',
        'network_split_1',
        'network_split_2',
        'network_split_3',
        'network_split_4',
        'network_split_5',
        'network_split_6',
        'simple_cnn'
    ]
    examples = []
    for subdir, _, _ in os.walk(os.path.join(examples_path, "onnx")):
        name = subdir.split('/')[-1]
        if name in EXAMPLES_OMIT or name == "onnx":
            continue
        else:
            examples.append((
                os.path.join(subdir, "network.onnx"),
                os.path.join(subdir, "input.json"),
            ))
    return examples

def test_examples(model_file, input_file, folder_path):
    """Tests all examples in the examples folder"""
    # gen settings
    settings_path = os.path.join(folder_path, "settings.json")
    compiled_model_path = os.path.join(folder_path, 'network.ezkl')
    pk_path = os.path.join(folder_path, 'test.pk')
    vk_path = os.path.join(folder_path, 'test.vk')
    witness_path = os.path.join(folder_path, 'witness.json')
    proof_path = os.path.join(folder_path, 'proof.json')

    print("Testing example: ", model_file)
    res = ezkl.gen_settings(model_file, settings_path)
    assert res

    res = ezkl.calibrate_settings(input_file, model_file, settings_path, "resources")
    assert res

    print("Compiling example: ", model_file)
    res = ezkl.compile_circuit(model_file, compiled_model_path, settings_path)
    assert res

    with open(settings_path, 'r') as f:
        data = json.load(f)

    logrows = data["run_args"]["logrows"]
    srs_path = os.path.join(folder_path, f"srs_{logrows}")

    # generate the srs file if the path does not exist
    if not os.path.exists(srs_path):
        print("Generating srs file: ", srs_path)
        ezkl.gen_srs(srs_path, logrows)

    print("Setting up example: ", model_file)
    res = ezkl.setup(
        compiled_model_path,
        vk_path,
        pk_path,
        srs_path
    )
    assert res == True
    assert os.path.isfile(vk_path)
    assert os.path.isfile(pk_path)

    print("Generating witness for example: ", model_file)
    res = ezkl.gen_witness(input_file, compiled_model_path, witness_path)
    assert os.path.isfile(witness_path)

    print("Proving example: ", model_file)
    ezkl.prove(
        witness_path,
        compiled_model_path,
        pk_path,
        proof_path,
        "single",
        srs_path=srs_path,
    )

    assert os.path.isfile(proof_path)

    print("Verifying example: ", model_file)
    res = ezkl.verify(
        proof_path,
        settings_path,
        vk_path,
        srs_path=srs_path,
    )

    assert res == True

    # Assuming your JSON file is named 'data.json'
    with open(settings_path, 'r') as f:
        data_dict = json.load(f)
        
    return data_dict
import onnx
import numpy as np

# Function to count model parameters in an ONNX model
def count_onnx_model_parameters(onnx_model_path):
    model = onnx.load(onnx_model_path)
    model_parameters = sum(np.prod(initializer.dims) for initializer in model.graph.initializer)
    return model_parameters


def get_onnx_forward_operations(onnx_model_path):
    from collections import Counter

    model = onnx.load(onnx_model_path)
    layer_types = Counter(node.op_type for node in model.graph.node)
    return dict(layer_types)

def compute_ratio(numerator, denominator):
    if denominator == 0:
        return None
    ratio = numerator/ denominator
    return ratio

if __name__ == "__main__":
    examples = get_examples()
    for idx, example in enumerate(examples):
        onnx_model_path, input_file = example
        parent_dir = os.path.dirname(onnx_model_path)
        parent_folder_name = os.path.basename(parent_dir)
        total_params_onnx = count_onnx_model_parameters(onnx_model_path)
        print(f"Total number of parameters in the ONNX model: {total_params_onnx}")

        # Generate EZKL settings
        settings_path = f"settings_{parent_folder_name}.json"
        ezkl.gen_settings(onnx_model_path, settings_path)
         # Load EZKL settings
        with open(settings_path, 'r') as f:
            settings_data = json.load(f)
        # os.remove(settings_path)
         # Get operations and layer types in the forward pass
        onnx_forward_operations = get_onnx_forward_operations(onnx_model_path)
        parent_dir = os.path.dirname(onnx_model_path)
        parent_folder_name = os.path.basename(parent_dir)

        # Create output dictionary
        data_dict = {
            "model_name": parent_folder_name,
            # "num_model_params(pytorch)": total_params_torch,
            "num_model_params(onnx)": total_params_onnx,
            # "inference_results_match": compare_outputs(pytorch_results, onnx_results),
            "logrows_in_circuit": settings_data["run_args"]["logrows"],
            "rows_in_circuit": settings_data["num_rows"],
            "assignments_circuit": settings_data["total_assignments"],
            # "circuit_rows/onnx_model_param": compute_ratio(settings_data["num_rows"], total_params_onnx),
            # "circuit_assignments/onnx_model_param": compute_ratio(settings_data["total_assignments"], total_params_onnx),
            "inference_operations_onnx": onnx_forward_operations
        }

        csv_file = 'model_to_circuit_relationship/model_to_circuit_comparison.csv'
        file_exists = os.path.isfile(csv_file)

        import csv

        # Print the output dictionary
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data_dict.keys())
            if not file_exists:
                writer.writeheader()  # Write header only if the file is new
            writer.writerow(data_dict)  # Write data as a new row
