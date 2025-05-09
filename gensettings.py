import ezkl
import os
onnx_model_path = r"C:\Users\pw\projects\ezkl-exploration\cache\residual_block\model.onnx"
settings_path = "settings.json"

output = ezkl.gen_settings(onnx_model_path, settings_path)
print(output)
print(f"Settings file generated at: {os.path.abspath(settings_path)}")