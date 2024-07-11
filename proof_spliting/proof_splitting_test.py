import os
from distrubuted_proving.log_utils import ExperimentLogger, ResourceMonitor, time_function, print_func_exec_info
import ezkl
from distrubuted_proving.utils import get_num_parameters
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

def run_setup(sub_model_configs:List[SubModelConfig], run_args):
     #setup
    for idx, config in enumerate(sub_model_configs):
        print(f'Running Setup for: {config.model_path}')
        if idx > 0:
            #use the previous witness as the input for this split
            prev_witness_path = sub_model_configs[idx-1].witness_path
            witness = json.load(open(prev_witness_path,'r'))
            data = dict(input_data = witness['outputs'])
            json.dump(data, open(config.data_path, 'w' ))
        if run_args:
            res = ezkl.gen_settings(config.model_path, config.settings_path, py_run_args=run_args)
            # res = ezkl.calibrate_settings(config.data_path, config.model_path, config.settings_path, "resources", scales=[run_args.input_scale], max_logrows=run_args.logrows)
            settings = json.load(open(config.settings_path, 'r'))
            settings['run_args']['logrows'] = run_args.logrows
            json.dump(settings, open(config.settings_path, 'w' ))
        else:
            res = ezkl.gen_settings(config.model_path, config.settings_path)
            settings = json.load(open(config.settings_path, 'r'))
        assert res == True

        res = ezkl.compile_circuit(config.model_path, config.compiled_model_path, config.settings_path)
        assert res == True

        res = ezkl.setup(config.compiled_model_path, config.vk_path, config.pk_path)
        assert res == True
        assert os.path.isfile(config.vk_path)
        assert os.path.isfile(config.pk_path)

        #generate witness for the current model
        res = ezkl.gen_witness(config.data_path, config.compiled_model_path, config.witness_path, config.vk_path)
        run_args.input_scale = settings["model_output_scales"][0]

def run_proving(sub_model_configs:List[SubModelConfig]):
    #prove
    for idx, config in enumerate(sub_model_configs):
        print(f'Running prove for: {config.model_path}')

        res = ezkl.prove(config.witness_path,config.compiled_model_path,config.pk_path,config.proof_path,"for-aggr",)
        # print(res)
        res_1_proof = res["proof"]
        assert os.path.isfile(config.proof_path)

        #verify the proof
        if idx > 0:
            print("swapping commitments")
            # swap the proof commitments if we are not the first model
            prev_witness_path = sub_model_configs[idx-1].witness_path
            prev_witness = json.load(open(prev_witness_path, 'r'))

            witness = json.load(open(config.witness_path, 'r'))

            # print(prev_witness["processed_outputs"])
            # print(witness["processed_inputs"])
            witness["processed_inputs"] = prev_witness["processed_outputs"]

            # now save the witness
            with open(config.witness_path, "w") as f:
                json.dump(witness, f)

            ezkl.swap_proof_commitments(config.proof_path, config.witness_path)
            
            # load proof and then print 
            proof = json.load(open(config.proof_path, 'r'))
            res_2_proof = proof["hex_proof"]
            # show diff in hex strings
            # print(res_1_proof)
            # print(res_2_proof)
            # assert res_1_proof == res_2_proof
        
        res = ezkl.verify(config.proof_path, config.settings_path, config.vk_path,)
        assert res == True
        print(f"verified {config.model_path}")


def execute_proof_split(sub_model_configs:List[SubModelConfig]):
    run_args = get_run_args()
    run_setup(sub_model_configs, run_args)
    run_proving(sub_model_configs,run_args)

if __name__ == '__main__':
    
    configs = {}
    folder_path = 'examples/onnx/residual_block_split'
    entries = os.listdir(folder_path)
    subfolders = [entry for entry in entries if os.path.isdir(os.path.join(folder_path, entry))]
    regex_pattern = r'split_(\d+)'

    for sub_model_dir in subfolders:
         sub_model_dir = os.path.join(folder_path, sub_model_dir)
         match = re.search(regex_pattern, os.path.basename(sub_model_dir))
         idx = int(match.group(1))
         configs[idx] = SubModelConfig(sub_model_dir)
         ordered_variables = OrderedDict(sorted(configs.items(), key=lambda item: int(item[0])))

    execute_proof_split(list(ordered_variables.values()))

    pass



























