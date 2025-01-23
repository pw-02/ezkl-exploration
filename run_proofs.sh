#!/bin/bash

# Execute dispatcher commands for different models
#!/bin/bash
conda activate ezkl

# Execute dispatcher commands for different models

python distributed_proving/dispatcher.py model=mnist_classifier
python distributed_proving/dispatcher.py model=mnist_gan



# python distributed_proving/dispatcher.py model=nano_gpt_5_layers_64_embd
# python distributed_proving/dispatcher.py model=nano_gpt_10_layers_64_embd
# python distributed_proving/dispatcher.py model=nano_gpt_15_layers_64_embd
# python distributed_proving/dispatcher.py model=nano_gpt_20_layers_64_embd
# python distributed_proving/dispatcher.py model=nano_gpt_4_layers_112_embd
# python distributed_proving/dispatcher.py model=nano_gpt_4_layers_128_embd
# python distributed_proving/dispatcher.py model=nano_gpt_4_layers_144_embd
# python distributed_proving/dispatcher.py model=nano_gpt_25_layers_64_embd

#commands to run the script
# chmod +x run_dispatcher.sh
# ./run_dispatcher.sh
