import logging
import os
import json
import csv
import ezkl
import shutil
def write_info_to_csv(report_file, info):
    header_written = os.path.exists(report_file)

    with open(report_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=info.keys())
        if not header_written:
            writer.writeheader()
        writer.writerow(info)


def gen_settings_file(tmp_folder,model_name, onnx_model_path, input_data_path):
    report_file = "ezkl_settings_report.csv"


    tmp_settings_file = f"{tmp_folder}/settings_{model_name}.json"

    info = {
        "name": model_name,
        "onnx_model_path": onnx_model_path,
        "input_data_path": input_data_path,
    }

    ezkl.gen_settings(onnx_model_path, tmp_settings_file)

    with open(tmp_settings_file, "r") as f:
        ezkl_settings = json.load(f)
        info.update(ezkl_settings)

    write_info_to_csv(report_file, info)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    path = "tmp/debug_split"
    error_count = 0

    tmp_folder = "ezkl_tmp"
    if os.path.exists(tmp_folder):
            shutil.rmtree(tmp_folder)
            # os.removedirs(tmp_folder)

    os.makedirs(tmp_folder, exist_ok=True)

    model_files = sorted(
        [f for f in os.listdir(path) if f.startswith("model_") and f.endswith(".onnx")],
        key=lambda x: int(x.replace("model_", "").replace(".onnx", ""))
    )

    for model_file in model_files:
        try:
            idx = model_file.replace("model_", "").replace(".onnx", "")

            input_file = f"input_{idx}.json"

            onnx_model_path = os.path.join(path, model_file)
            input_data_path = os.path.join(path, input_file)

            if not os.path.exists(input_data_path):
                raise FileNotFoundError(f"Missing input file: {input_data_path}")

            model_name = f"model_{idx}"

            gen_settings_file(
                tmp_folder=tmp_folder,
                model_name=model_name,
                onnx_model_path=onnx_model_path,
                input_data_path=input_data_path,
            )

            print(f"Settings file generated successfully for {model_name}.")

        except Exception as e:
            error_count += 1
            print(f"Error processing {model_file}: {e}")

    print(f"Total errors encountered: {error_count}")