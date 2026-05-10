import ezkl
import os
import onnx
import json
import csv
import numpy as np

mdoels_to_analyze = [
    ('examples/onnx/mnist_classifier/network.onnx','mnist_classifier'),
    ('examples/onnx/mnist_gan/network.onnx','mnist_gan'),
    ('examples/onnx/nanoGPT/nano_gpt_4_layers_64_embd.onnx','nano_gpt_4_layers_64_embd'),
    ('examples/onnx/nanoGPT/nano_gpt_4_layers_80_embd.onnx','nano_gpt_4_layers_80_embd'),
    ('examples/onnx/nanoGPT/nano_gpt_4_layers_96_embd.onnx','nano_gpt_4_layers_96_embd'),
    ('examples/onnx/nanoGPT/nano_gpt_4_layers_112_embd.onnx','nano_gpt_4_layers_112_embd'),
    ('examples/onnx/nanoGPT/nano_gpt_4_layers_128_embd.onnx','nano_gpt_4_layers_128_embd'),
    ('examples/onnx/nanoGPT/nano_gpt_4_layers_144_embd.onnx','nano_gpt_4_layers_144_embd'),
    ('examples/onnx/nanoGPT/nano_gpt_5_layers_64_embd.onnx','nano_gpt_5_layers_64_embd'),
    ('examples/onnx/nanoGPT/nano_gpt_10_layers_64_embd.onnx','nano_gpt_10_layers_64_embd'),
    ('examples/onnx/nanoGPT/nano_gpt_15_layers_64_embd.onnx','nano_gpt_15_layers_64_embd'),
    ('examples/onnx/nanoGPT/nano_gpt_20_layers_64_embd.onnx','nano_gpt_20_layers_64_embd'),
    ('examples/onnx/nanoGPT/nano_gpt_25_layers_64_embd.onnx','nano_gpt_25_layers_64_embd'),
    ('examples/onnx/mobile_net/mobilenetv2_050_Opset18.onnx','mobilenetv2_050_Opset18'),
    ('examples/onnx/mobile_net/network.onnx','mobilenet'),
    ('examples/onnx/mobilenet_large/network.onnx','mobilenet_large'),
    ('examples/onnx/clip/network.onnx','clip'),
    ('examples/onnx/xgboost/network.onnx','xgboost'),
    ('examples/onnx/random_forest/network.onnx','random_forest'),
    ('examples/onnx/lstm_large/network.onnx','lstm_large'),
    ('examples/onnx/linear_regression/network.onnx','linear_regression'),
    ('examples/onnx/lightgbm/network.onnx','lightgbm'),
    ('examples/onnx/rnn/network.onnx','rnn'),


]

def count_weights_and_tensors_in_onnx_model(model):
    # Load the ONNX model
    if isinstance(model, str):
        model = onnx.load(model)
    
    total_weights = 0
    total_input_size = 0
    total_output_size = 0

    # Count weights in initializers
    for initializer in model.graph.initializer:
        total_weights += len(onnx.numpy_helper.to_array(initializer).flatten())

    # Count input tensor sizes
    for input_tensor in model.graph.input:
        shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        total_input_size += int(np.prod(shape))

    # Count output tensor sizes
    for output_tensor in model.graph.output:
        shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
        total_output_size += int(np.prod(shape))

    return total_weights + total_input_size + total_output_size

def analyze_model(onnx_model_path, model_name):
    data_dir = r'utils'
    tmp_file = os.path.join(data_dir, 'settings.json')

    onnx_model = onnx.load(onnx_model_path)
    num_model_ops  = len(onnx_model.graph.node)
    num_model_params =  0
    for initializer in onnx_model.graph.initializer:
        param_array = onnx.numpy_helper.to_array(initializer)
        num_model_params += param_array.size
    num_combined_params = count_weights_and_tensors_in_onnx_model(onnx_model)

    ezkl.gen_settings(onnx_model_path, tmp_file)

    
    info = {
        "name": model_name,
        "onnx_model_path": onnx_model_path,
        "num_model_ops": num_model_ops,
        "num_model_params": num_model_params,
        "num_combined_params": num_combined_params}
    
    if os.path.exists(tmp_file):
        with open(tmp_file, 'r') as f:
            ezkl_settings = json.load(f)
        info = {**info, **ezkl_settings}
    
    return info

def main(models_to_analyze):

    data_dir = r'utils'
    report_file = os.path.join(data_dir, 'report.csv')
    file_exists = os.path.isfile(report_file)
    if file_exists:
        #read in csv file and get the models that have already been analyzed
        with open(report_file, mode='r') as file:
            reader = csv.DictReader(file)
            analyzed_models = [row["name"] for row in reader]
    else:
        analyzed_models = []

    for onnx_model_path, model_name in models_to_analyze:
        if model_name in analyzed_models:
          continue
    
        info = analyze_model(onnx_model_path, model_name)
        file_exists = os.path.isfile(report_file)
        with open(report_file, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=info.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(info)
        print(f"Model {model_name} analyzed and results saved in {report_file}")

if __name__ == '__main__':
    main(mdoels_to_analyze)






