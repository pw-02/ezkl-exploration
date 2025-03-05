# #simple examples to check everything is working
# python gen_proof.py model=mnist_classifier model.split_group_size=null 
# python gen_proof.py model=mnist_gan model.split_group_size=1

# #complex examples
# python gen_proof.py model=nano_gpt_4_layers_64_embd model.split_group_size=null
# python gen_proof.py model=nano_gpt_4_layers_64_embd model.split_group_size=1

#nanogts

python gen_proof.py model=nano_gpt_10_layers_64_embd model.split_group_size=null
python gen_proof.py model=nano_gpt_10_layers_64_embd model.split_group_size=1

python gen_proof.py model=nano_gpt_4_layers_96_embd model.split_group_size=null
python gen_proof.py model=nano_gpt_4_layers_96_embd model.split_group_size=1

python gen_proof.py model=nano_gpt_4_layers_128_embd model.split_group_size=null
python gen_proof.py model=nano_gpt_4_layers_128_embd model.split_group_size=1

#mobilenets
python gen_proof.py model=mobilenetv2_050_Opset18 model.split_group_size=null
python gen_proof.py model=mobilenetv2_050_Opset18 model.split_group_size=1
