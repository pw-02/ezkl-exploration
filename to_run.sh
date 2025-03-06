#!/bin/bash

# List of commands to execute
commands=(
    "python gen_proof.py model=mnist_classifier model.split_group_size=null"
    "python gen_proof.py model=mnist_gan model.split_group_size=1"
    "python gen_proof.py model=nano_gpt_4_layers_64_embd model.split_group_size=null"
    "python gen_proof.py model=nano_gpt_4_layers_64_embd model.split_group_size=1"
    "python gen_proof.py model=nano_gpt_10_layers_64_embd model.split_group_size=null"
    "python gen_proof.py model=nano_gpt_10_layers_64_embd model.split_group_size=1"
    "python gen_proof.py model=nano_gpt_4_layers_96_embd model.split_group_size=null"
    "python gen_proof.py model=nano_gpt_4_layers_96_embd model.split_group_size=1"
    "python gen_proof.py model=nano_gpt_4_layers_128_embd model.split_group_size=null"
    "python gen_proof.py model=nano_gpt_4_layers_128_embd model.split_group_size=1"
    "python gen_proof.py model=mobilenetv2_050_Opset18 model.split_group_size=null"
    "python gen_proof.py model=mobilenetv2_050_Opset18 model.split_group_size=1"
)

# Loop over the commands
for cmd in "${commands[@]}"; do
    echo "Running: $cmd"
    
    # Run the command and check for errors
    if $cmd; then
        echo "✅ SUCCESS: $cmd"
    else
        echo "❌ ERROR: $cmd"
    fi
done

echo "Execution completed."
