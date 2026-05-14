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


def gen_settings_file(tmp_folder,
                      model_name, 
                      onnx_model_path, 
                      input_data_path, 
                      run_args,
                      run_calibration=False):
    
    report_file = "ezkl_settings_report.csv"

    tmp_settings_file = f"{tmp_folder}/settings_{model_name}.json"

    info = {
        "name": model_name,
        "onnx_model_path": onnx_model_path,
        "input_data_path": input_data_path,
    }
    if run_args is not None:
        ezkl.gen_settings(onnx_model_path, tmp_settings_file, py_run_args=run_args)
    else:
        ezkl.gen_settings(onnx_model_path, tmp_settings_file)
    if run_calibration:
            res = ezkl.calibrate_settings(
                input_data_path,
                onnx_model_path,
                tmp_settings_file,
                "resources",
            )

    with open(tmp_settings_file, "r") as f:
        ezkl_settings = json.load(f)
        info.update(ezkl_settings)

    write_info_to_csv(report_file, info)


def gen_settings_files_for_all_models(path, run_args, run_calibration=False):
    error_count = 0

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
                print(f"Missing input file: {input_data_path}")
                continue

            model_name = f"model_{idx}"

            gen_settings_file(
                tmp_folder="_ezkl_tmp",
                model_name=model_name,
                onnx_model_path=onnx_model_path,
                input_data_path=input_data_path,
                run_args=run_args,
                run_calibration=run_calibration
            )
            print(f"Settings file generated successfully for {model_name}.")
        except Exception as e:
            error_count += 1
            print(f"Error processing {model_file}: {e}")

    print(f"Total errors encountered: {error_count}")

def gen_settings_for_given_model_and_input(onnx_model_path, input_data_path, run_args, run_calibration=False, model_name = None):
    try:

        if not os.path.exists(input_data_path):
            print(f"Missing input file: {input_data_path}")
            return
        if model_name is None:
            model_name = os.path.basename(onnx_model_path).replace(".onnx", "")
        
        gen_settings_file(
            tmp_folder="_ezkl_tmp",
            model_name=model_name,
            onnx_model_path=onnx_model_path,
            input_data_path=input_data_path,
            run_args=run_args,
            run_calibration=run_calibration
        )
        print(f"Settings file generated successfully for {model_name}.")
    except Exception as e:
        print(f"Error processing {onnx_model_path}: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tmp_folder = "_ezkl_tmp"
    if os.path.exists(tmp_folder):
            shutil.rmtree(tmp_folder)
    os.makedirs(tmp_folder, exist_ok=True)

    # run_args = ezkl.PyRunArgs()
    # run_args.input_visibility = "private"
    # run_args.param_visibility = "private"
    # run_args.output_visibility = "public"

    run_args = None

    run_calibration = False
    gen_settings_files_for_all_models(path="_tmp/split__nanoGPT", run_args=run_args, run_calibration=run_calibration)
    # onnx_model_path = "experiments/models/llama/tiny_llama_6_layers_128_embd.onnx"
    # input_data_path = "experiments/models/llama/input.json"
    # gen_settings_for_given_model_and_input(onnx_model_path, input_data_path, run_args, run_calibration)


  
