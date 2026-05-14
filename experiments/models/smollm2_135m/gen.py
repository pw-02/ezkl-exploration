from huggingface_hub import snapshot_download

# SmolLM2 135M ONNX
snapshot_download(
    repo_id="onnx-community/SmolLM2-135M-ONNX",
    local_dir="models/smollm2-135m-onnx",
    # allow_patterns=[
    #     "onnx/model_q4f16.onnx",
    #     "*.json",
    #     "*.txt",
    # ],
    local_dir_use_symlinks=False,
)

print("Downloaded ONNX models.")