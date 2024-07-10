from torch import nn
import ezkl
import os
import json
import logging

# uncomment for more descriptive logging 
FORMAT = '%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(logging.INFO)

import torch

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=2, stride=4)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        return x
    
    def split_1(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x


circuit = MyModel()

x = torch.rand(1,*[3, 8, 8], requires_grad=True)

# Flips the neural net into inference mode
circuit.eval()

    # Export the model
torch.onnx.export(circuit,               # model being run
                      x,                   # model input (or a tuple for multiple inputs)
                      "network.onnx",            # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})


data_path = os.path.join(os.getcwd(), "input_0.json")
data = dict(input_data = [((x).detach().numpy()).reshape([-1]).tolist()])
json.dump( data, open(data_path, 'w' ))

inter_1 = circuit.split_1(x)
data_path = os.path.join(os.getcwd(), "input_1.json")
data = dict(input_data = [((inter_1).detach().numpy()).reshape([-1]).tolist()])
json.dump( data, open(data_path, 'w' ))

import onnx

input_path = "network.onnx"
output_path = "network_split_0.onnx"
input_names = ["input"]
output_names = ["/relu/Relu_output_0"]
# first model
onnx.utils.extract_model(input_path, output_path, input_names, output_names)

import onnx

input_path = "network.onnx"
output_path = "network_split_1.onnx"
input_names = ["/relu/Relu_output_0"]
output_names = ["output"]
# second model
onnx.utils.extract_model(input_path, output_path, input_names, output_names)

import ezkl


data_path = os.path.join('input.json')

run_args = ezkl.PyRunArgs()
run_args.input_visibility = "public"
run_args.param_visibility = "fixed"
run_args.output_visibility = "public"
run_args.input_scale = 2
run_args.logrows = 8

ezkl.get_srs(logrows=run_args.logrows, commitment=ezkl.PyCommitments.KZG)

# iterate over each submodel gen-settings, compile circuit and setup zkSNARK

def setup(i):
    # file names
    model_path = os.path.join('network_split_'+str(i)+'.onnx')
    settings_path = os.path.join('settings_split_'+str(i)+'.json')
    data_path =  os.path.join('input_'+str(i)+'.json')
    compiled_model_path = os.path.join('network_split_'+str(i)+'.compiled')
    pk_path = os.path.join('test_split_'+str(i)+'.pk')
    vk_path = os.path.join('test_split_'+str(i)+'.vk')
    witness_path = os.path.join('witness_split_'+str(i)+'.json')

    if i > 0:
         prev_witness_path = os.path.join('witness_split_'+str(i-1)+'.json')
         witness = json.load(open(prev_witness_path, 'r'))
         data = dict(input_data = witness['outputs'])
         # Serialize data into file:
         json.dump(data, open(data_path, 'w' ))
    else:
         data_path = os.path.join('input_0.json')

    # generate settings for the current model
    res = ezkl.gen_settings(model_path, settings_path, py_run_args=run_args)
    res = ezkl.calibrate_settings(data_path, model_path, settings_path, "resources", scales=[run_args.input_scale], max_logrows=run_args.logrows)
    assert res == True

    # load settings and print them to the console
    settings = json.load(open(settings_path, 'r'))
    settings['run_args']['logrows'] = run_args.logrows
    json.dump(settings, open(settings_path, 'w' ))

    res = ezkl.compile_circuit(model_path, compiled_model_path, settings_path)


    res = ezkl.setup(
         compiled_model_path,
         vk_path,
         pk_path,
      )

    assert res == True
    assert os.path.isfile(vk_path)
    assert os.path.isfile(pk_path)

    res = ezkl.gen_witness(data_path, compiled_model_path, witness_path, vk_path)
    run_args.input_scale = settings["model_output_scales"][0]

# for i in range(2):
#     setup(i)

# GENERATE A PROOF
def prove_model(i):
    proof_path = os.path.join('proof_split_'+str(i)+'.json')
    witness_path = os.path.join('witness_split_'+str(i)+'.json')
    compiled_model_path = os.path.join('network_split_'+str(i)+'.compiled')
    pk_path = os.path.join('test_split_'+str(i)+'.pk')
    vk_path = os.path.join('test_split_'+str(i)+'.vk')
    settings_path = os.path.join('settings_split_'+str(i)+'.json')

    res = ezkl.prove(
            witness_path,
            compiled_model_path,
            pk_path,
            proof_path,
            "for-aggr",
        )

    print(res)
    res_1_proof = res["proof"]
    assert os.path.isfile(proof_path)

    # # Verify the proof
    if i > 0:
        print("swapping commitments")
        # swap the proof commitments if we are not the first model
        prev_witness_path = os.path.join('witness_split_'+str(i-1)+'.json')
        prev_witness = json.load(open(prev_witness_path, 'r'))

        witness = json.load(open(witness_path, 'r'))

        print(prev_witness["processed_outputs"])
        print(witness["processed_inputs"])
        witness["processed_inputs"] = prev_witness["processed_outputs"]

        # now save the witness
        with open(witness_path, "w") as f:
            json.dump(witness, f)

        res = ezkl.swap_proof_commitments(proof_path, witness_path)
        print(res)
        
        # load proof and then print 
        proof = json.load(open(proof_path, 'r'))
        res_2_proof = proof["hex_proof"]
        # show diff in hex strings
        print(res_1_proof)
        print(res_2_proof)
        assert res_1_proof == res_2_proof

    res = ezkl.verify(
            proof_path,
            settings_path,
            vk_path,
        )

    assert res == True
    print("verified")

# for i in range(2):
#     prove_model(i)

import ezkl

run_args = ezkl.PyRunArgs()
run_args.input_visibility = "polycommit"
run_args.param_visibility = "fixed"
run_args.output_visibility = "polycommit"
run_args.variables = [("batch_size", 1)]
run_args.input_scale = 2
run_args.logrows = 8

for i in range(2):
    setup(i)

for i in range(2):
    prove_model(i)

# # now mock aggregate the proofs
# proofs = []
# for i in range(2):
#     proof_path = os.path.join('proof_split_'+str(i)+'.json')
#     proofs.append(proof_path)

# ezkl.mock_aggregate(proofs, logrows=22, split_proofs = True)