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


if __name__ == "__main__":
    import csv
    print('started')
    output_folder = "examples/outputs"
    os.makedirs(output_folder,exist_ok=True)
    times = {}
    examples = get_examples()
    end = time.perf_counter()
    for example in examples:
        model_File, input_file = example
        data_dict = test_examples(model_File, input_file, output_folder)
        data_dict['name'] = os.path.basename(os.path.dirname(model_File))
        data_dict['total_time'] = time.perf_counter() - end
        csv_file = 'data.csv'
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data_dict.keys())
            if not file_exists:
                writer.writeheader()  # Write header only if the file is new
            writer.writerow(data_dict)  # Write data as a new row
        end = time.perf_counter()
    print('ended')
