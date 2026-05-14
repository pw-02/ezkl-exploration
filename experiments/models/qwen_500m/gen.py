from huggingface_hub import snapshot_download

# Qwen2.5 0.5B ONNX
snapshot_download(
    repo_id="onnx-community/Qwen2.5-0.5B",
    local_dir="models/qwen2.5-0.5b-onnx",
    # allow_patterns=[
    #     "onnx/model_q4f16.onnx",
    #     "*.json",
    #     "*.txt",
    # ],
    local_dir_use_symlinks=False,
)