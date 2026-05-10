# check if notebook is in colab
try:
    # install ezkl
    import google.colab
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ezkl"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "onnx"])

# rely on local installation of ezkl if the notebook is not in colab
except:
    pass


# here we create and (potentially train a model)

# make sure you have the dependencies required here already installed
from torch import nn
import ezkl
import os
import json
import torch


# Defines the model
# we got convs, we got relu, we got linear layers
# What else could one want ????

# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()

#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2)
#         self.conv2 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=2)

#         self.relu = nn.ReLU()


#         # self.d1 = nn.Linear(48, 48)
#         # self.d2 = nn.Linear(48, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.relu(x)
#         return x

# circuit = MyModel()
# # Simulate random input
# torch.manual_seed(42)  # Set a fixed seed
# # x = torch.rand(1, 3, 8, 8, requires_grad=True)
# x = torch.rand(1, *[3, 8, 8], requires_grad=True)
# #get the output of the model
# output = circuit(x)
# print(f'output:{output}, shape: {output.shape}')



# # Flips the neural net into inference mode
# circuit.eval()

#     # Export the model
# torch.onnx.export(circuit,               # model being run
#                       x,                   # model input (or a tuple for multiple inputs)
#                       model_path,            # where to save the model (can be a file or file-like object)
#                       export_params=True,        # store the trained parameter weights inside the model file
#                       opset_version=10,          # the ONNX version to export the model to
#                       do_constant_folding=True,  # whether to execute constant folding for optimization
#                       input_names = ['input'],   # the model's input names
#                       output_names = ['output'], # the model's output names
#                       dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
#                                     'output' : {0 : 'batch_size'}})

# data_array = ((x).detach().numpy()).reshape([-1]).tolist()

# data = dict(input_data = [data_array])

# # Serialize data into file:
# json.dump( data, open(data_path, 'w' ))

output_folder = "tests/single_model"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
model_path = os.path.join(output_folder, "model.pth")
compiled_model_path = os.path.join(output_folder,'network.compiled')
pk_path = os.path.join(output_folder,'test.pk')
vk_path = os.path.join(output_folder,'test.vk')
settings_path = os.path.join(output_folder,'settings.json')
witness_path = os.path.join(output_folder,'witness.json')
data_path = os.path.join(output_folder,'input.json')


py_run_args = ezkl.PyRunArgs()
py_run_args.input_visibility = "public"
py_run_args.output_visibility = "public"
py_run_args.param_visibility = "fixed" # "fixed" for params means that the committed to params are used for all proofs

res = ezkl.gen_settings(model_path, settings_path, py_run_args=py_run_args)
assert res == True

res = ezkl.compile_circuit(model_path, compiled_model_path, settings_path)
assert res == True

# srs path
res = ezkl.get_srs( settings_path)

# now generate the witness file 

res = ezkl.gen_witness(data_path, compiled_model_path, witness_path)
assert os.path.isfile(witness_path)

res = ezkl.setup(
        compiled_model_path,
        vk_path,
        pk_path,
        
    )

assert res == True
assert os.path.isfile(vk_path)
assert os.path.isfile(pk_path)
assert os.path.isfile(settings_path)

proof_path = os.path.join('test.pf')

res = ezkl.prove(
        witness_path,
        compiled_model_path,
        pk_path,
        proof_path,
        
        "single",
    )

# print(res)
assert os.path.isfile(proof_path)

res = ezkl.verify(
        proof_path,
        settings_path,
        vk_path,
        
    )

assert res == True
print("verified")