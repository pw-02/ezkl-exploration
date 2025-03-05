import ezkl
import os
import onnx
import json
import csv

mdoels_to_analyze = [
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
    ('examples/onnx/mobilenet_large/network.onnx','mobilenet_large')
    ('examples/onnx/clip/network.onnx','clip'),




]

def analyze_model(onnx_model_path, model_name):
    data_dir = r'utils'
    tmp_file = os.path.join(data_dir, 'settings.json')

    ezkl.gen_settings(onnx_model_path, tmp_file)
    onnx_model = onnx.load(onnx_model_path)
    num_model_ops  = len(onnx_model.graph.node)
    num_model_params =  0
    for initializer in onnx_model.graph.initializer:
        param_array = onnx.numpy_helper.to_array(initializer)
        num_model_params += param_array.size
    
    info = {
        "name": model_name,
        "onnx_model_path": onnx_model_path,
        "num_model_ops": num_model_ops,
        "num_model_params": num_model_params}
    
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
        with open(report_file, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=info.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(info)
        print(f"Model {model_name} analyzed and results saved in {report_file}")

if __name__ == '__main__':
    main(mdoels_to_analyze)






