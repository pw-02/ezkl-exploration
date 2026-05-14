# export_mobilellm_125m_to_onnx.py

import subprocess
from pathlib import Path

import torch
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM


MODEL_ID = "facebook/MobileLLM-125M"
OUT_DIR = Path("mobilellm_125m_onnx")


def install_note():
    print("Install deps first:")
    print("pip install -U torch transformers optimum[onnxruntime] onnx onnxruntime accelerate")


def export_to_onnx():
    OUT_DIR.mkdir(exist_ok=True)

    cmd = [
        "optimum-cli",
        "export",
        "onnx",
        "--model",
        MODEL_ID,
        "--task",
        "text-generation",
        "--trust-remote-code",
        str(OUT_DIR),
    ]

    print("Running:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)


def test_onnx():
    print("Loading exported ONNX model...")

    tokenizer = AutoTokenizer.from_pretrained(OUT_DIR)
    model = ORTModelForCausalLM.from_pretrained(OUT_DIR)

    prompt = "The answer is"
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=5,
        do_sample=False,
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Output:")
    print(text)


def main():
    install_note()
    export_to_onnx()
    test_onnx()

    print("\nDone. ONNX files are in:")
    print(OUT_DIR.resolve())


if __name__ == "__main__":
    main()