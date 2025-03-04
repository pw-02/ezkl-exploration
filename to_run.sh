#simple examples to check everything is working
python gen_proof.py model=mnist_classifier model.split_group_size=null 
python gen_proof.py model=mnist_gan model.split_group_size=1

#complex examples
python gen_proof.py model=nano_gpt_4_layers_64_embd model.split_group_size=null
python gen_proof.py model=nano_gpt_4_layers_64_embd model.split_group_size=1