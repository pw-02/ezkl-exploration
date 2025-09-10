import os
import shutil
import json
import csv
from zkInfer.onnx_splitter import (
    # collect_intermediate_inference_outputs,
    split_onnx_model,
    # save_split_models,
    get_model_info,
)
import ezkl

def clean_directory(directory):
    """Recursively delete a directory and all its contents."""
    if os.path.exists(directory):
        try:
            shutil.rmtree(directory)
        except Exception as e:
            print(f"Error removing directory {directory}: {e}")

def write_info_to_csv(report_file, info):
    header_written = os.path.exists(report_file)
    with open(report_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=info.keys())
        if not header_written:
            writer.writeheader()
        writer.writerow(info)

def gen_and_merge_settings(onnx_model_path, input_data_path, model_name, tmp_settings_file):
    info = {
        'name': model_name,
        'onnx_model_path': onnx_model_path,
        'input_data_path': input_data_path,
    }
    info.update(get_model_info(onnx_model_path))
    ezkl.gen_settings(onnx_model_path, tmp_settings_file)
    with open(tmp_settings_file, 'r') as f:
        ezkl_settings = json.load(f)
        info.update(ezkl_settings)
    return info

def get_settings_file(model_name, onnx_model_path: str, input_data_path: str, group_size: int) -> str:
    tmp_cache_directory = os.path.join("cache", "tmp")
    os.makedirs(tmp_cache_directory, exist_ok=True)
    report_file = os.path.join("ezkl_settings_report.csv")
    tmp_settings_file = os.path.join(tmp_cache_directory, 'settings.json')
    info = gen_and_merge_settings(onnx_model_path, input_data_path, model_name, tmp_settings_file)
    write_info_to_csv(report_file, info)
    # write_info_to_csv(report_file, info)
    # if group_size is None:
    #     info = gen_and_merge_settings(onnx_model_path, input_data_path, model_name, tmp_settings_file)
    #     write_info_to_csv(report_file, info)
    # else:
    #     intermediate_outputs = collect_intermediate_inference_outputs(onnx_model_path, input_data_path)
    #     sub_models = split_onnx_model(onnx_model_path, group_size)
    #     submodel_io = save_split_models(sub_models, intermediate_outputs, tmp_cache_directory)
    #     for sub_name, (input_path, model_path, meta) in submodel_io.items():
    #         info = gen_and_merge_settings(model_path, input_path, model_name, tmp_settings_file)
    #         write_info_to_csv(report_file, info)

    clean_directory(tmp_cache_directory)

if __name__ == "__main__":
    name = "bert"
    input_file = r"examples/onnx/bert/input.json"
    onnx_file = r"examples/onnx/bert/bert_tiny_squad.onnx"
    # onnx_file = r"examples/onnx/mnist_classifier/network.onnx"

    # input_file = r"examples\onnx\resnet18\input.json"
    # onnx_file = r"examples\onnx\resnet18\resnet18_cifar10.onnx"
    group_size = None  # Adjust as needed
    get_settings_file(name, onnx_file, input_file, group_size)
    print("Settings file generated successfully.")
