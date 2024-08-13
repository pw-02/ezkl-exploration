import os
from distributed_proving.log_utils import ExperimentLogger, ResourceMonitor, time_function, print_func_exec_info
import ezkl
from distributed_proving.utils import get_num_parameters
from typing import List
import json
import glob
import re
from collections import OrderedDict

#model splits, and the orgional input are needed

class SubModelConfig():
    def __init__(self, worker_dir:str):
        self.directory = worker_dir
        self.model_path =  os.path.join(self.directory, f'model.onnx')
        self.data_path = os.path.join(self.directory, f'input.json')
        self.compiled_model_path = os.path.join(self.directory, f'network.compiled')
        self.pk_path = os.path.join(self.directory, f'key.pk')
        self.vk_path = os.path.join(self.directory, f'key.vk')
        self.settings_path = os.path.join(self.directory, f'settings.json')
        self.witness_path = os.path.join(self.directory, f'witness.json')
        self.cal_path = os.path.join(self.directory, f'calibration.json')
        self.proof_path = os.path.join(self.directory, f'test.pf')
        self.exp_logger = ExperimentLogger(log_dir=self.directory)

def get_run_args():
    run_args = ezkl.PyRunArgs()
    run_args.input_visibility = "polycommit"
    run_args.param_visibility = "fixed"
    run_args.output_visibility = "polycommit"
    run_args.input_scale = 2
    # run_args.logrows = 20
    return run_args

if __name__ == '__main__':
    run_args = get_run_args()
    previous_model_0 = SubModelConfig('proof_spliting/two_inputs/network1')
    previous_model_1 = SubModelConfig('proof_spliting/two_inputs/network2')
    current_model = SubModelConfig('proof_spliting/two_inputs/network3')
    
    inputs = []
    prev_witness_path = previous_model_0.witness_path
    witness0 = json.load(open(prev_witness_path,'r'))
    inputs.append(witness0['inputs'][0])
    inputs = witness0['inputs']
    prev_witness_path = previous_model_1.witness_path
    witness1 = json.load(open(prev_witness_path,'r'))
    next_input = witness1['outputs'][0]
    inputs.append(witness1['outputs'][0])
    inputs.reverse()

    data = dict(input_data = inputs)

    json.dump(data, open(current_model.data_path, 'w' ))

    res = ezkl.gen_settings(current_model.model_path, current_model.settings_path, py_run_args=run_args)
    # res = ezkl.calibrate_settings(config.data_path, config.model_path, config.settings_path, "resources", scales=[run_args.input_scale], max_logrows=run_args.logrows)
    settings = json.load(open(current_model.settings_path, 'r'))
    settings['run_args']['logrows'] = run_args.logrows
    json.dump(settings, open(current_model.settings_path, 'w' ))
    assert res == True

    res = ezkl.compile_circuit(current_model.model_path, current_model.compiled_model_path, current_model.settings_path)
    assert res == True

    res = ezkl.setup(current_model.compiled_model_path, current_model.vk_path, current_model.pk_path)
    assert res == True
    assert os.path.isfile(current_model.vk_path)
    assert os.path.isfile(current_model.pk_path)

    #generate witness for the current model
    res = ezkl.gen_witness(current_model.data_path, current_model.compiled_model_path, current_model.witness_path, current_model.vk_path)
    run_args.input_scale = settings["model_output_scales"][0]

    print(f'Running prove for: {current_model.model_path}')
    res = ezkl.prove(current_model.witness_path,current_model.compiled_model_path,current_model.pk_path,current_model.proof_path,"for-aggr",)
    # print(res)
    res_1_proof = res["proof"]
    assert os.path.isfile(current_model.proof_path)
    print("swapping commitments")
    
    # swap the proof commitments if we are not the first model
    prev_witness_path = previous_model_1.witness_path
    prev_witness1 = json.load(open(prev_witness_path, 'r'))

    prev_witness_path = previous_model_0.witness_path
    prev_witness0 = json.load(open(prev_witness_path, 'r'))
    processed_input = prev_witness0["processed_inputs"]['polycommit']
    for input in processed_input:
        prev_witness1["processed_outputs"]['polycommit'].append(input)

    witness = json.load(open(current_model.witness_path, 'r'))

    # print(prev_witness["processed_outputs"])
    # print(witness["processed_inputs"])
    witness["processed_inputs"] = prev_witness1["processed_outputs"]

    # now save the witness
    with open(current_model.witness_path, "w") as f:
        json.dump(witness, f)

    ezkl.swap_proof_commitments(current_model.proof_path, current_model.witness_path)

    res = ezkl.verify(current_model.proof_path, current_model.settings_path, current_model.vk_path,)
    assert res == True
    # print(f"verified {current_model.model_path}")
    print(f"Verified Add_output_0")


    pass



























