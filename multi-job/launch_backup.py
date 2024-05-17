import hydra
from omegaconf import DictConfig
import os 
import mnist_classifier
import logging
import torch
import json
import ezkl
import time
from functools import wraps
import onnx
import multiprocessing
import shutil
FORMAT = '%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT, filename='my_log.log', filemode='a')
logging.getLogger().setLevel(logging.DEBUG)

def time_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get the start time
        start_time = time.time()
        
        # Call the function
        result = func(*args, **kwargs)
        
        # Get the end time
        end_time = time.time()
        
        # Calculate the execution time
        execution_time = end_time - start_time
        
        # Print the execution time
        print(f"{func.__name__} executed in {execution_time:.4f} seconds")
        
        # Return the result of the function
        return result
    return wrapper

@time_function
@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(config: DictConfig):
    config.output_dir = os.path.join(config.output_dir, config.model.name)
    if os.path.exists(config.output_dir):
        # Delete the folder and its contents
        shutil.rmtree(config.output_dir)
        print(f"The folder '{config.output_dir}' has been deleted.")
   
    os.makedirs( config.output_dir, exist_ok=True)

    # Create processes for two ranks
    processes = []
    for rank in range(2):
        p = multiprocessing.Process(target=run_process, args=(config, rank))
        processes.append(p)

    # Start processes
    for p in processes:
        p.start()

    # Wait for processes to finish
    for p in processes:
        p.join()

@time_function
def run_process(config,rank):
    prepare(config, rank)
    run_setup(config, rank)
    proof_result = prove_model(config, rank)
    verify_model(rank, config, proof_result)

@time_function
def prepare(config: DictConfig, rank:int):
    # model: config.model 
    model_path = os.path.join(config.output_dir,f'network.onnx')
    data_path = os.path.join(config.output_dir,f'input_{rank}.json')    
    network_split_path = os.path.join(config.output_dir,f'network_split_{rank}.onnx')
    if config.model.name == 'mnist_classifier':
        model, shape, data_point =  mnist_classifier.get_model()
    export_model_and_data(model=model, x=data_point, model_path=model_path, data_path=data_path, output_path=network_split_path,rank=rank )

@time_function
def run_setup(config, rank):
    model_path = os.path.join(config.output_dir,f'network_split_{rank}.onnx')
    settings_path = os.path.join(config.output_dir,f'settings_split_{rank}.json')
    data_path = os.path.join(config.output_dir,f'input_{rank}.json')
    compiled_model_path = os.path.join(config.output_dir,f'network_{rank}.compiled')
    pk_path = os.path.join(config.output_dir,f'test_split_{rank}.pk')
    vk_path = os.path.join(config.output_dir,f'test_split_{rank}.vk')
    witness_path = os.path.join(config.output_dir,f'witness_split_{rank}.json')

    gen_settings(model_path, settings_path)
    calibrate_settings(data_path, model_path, settings_path)
    compile_circuit(model_path, compiled_model_path, settings_path)
    
    #generate the srs
    get_srs(settings_path)
    
    #runs setup
    setup(compiled_model_path,vk_path,pk_path,settings_path)

    if rank > 0:
        prev_witness_path = os.path.join(config.output_dir,'witness_split_'+str(rank-1)+'.json')
        # Wait for the previous witness file to exist
        while not os.path.exists(prev_witness_path):
            time.sleep(0.01)
        witness = json.load(open(prev_witness_path, 'r'))
        data = dict(input_data = witness['outputs'])
        # Serialize data into file:
        json.dump(data, open(data_path, 'w' ))
    
    # # generate settings for the current model
    # res = ezkl.gen_settings(model_path, settings_path, py_run_args=run_args)
    # res = ezkl.calibrate_settings(data_path, model_path, settings_path, "resources", scales=[run_args.input_scale], max_logrows=run_args.logrows)
    # assert res == True
    

    #enerate the witness file 
    gen_witness(data_path, compiled_model_path, witness_path)


def get_args():
    run_args = ezkl.PyRunArgs()
    run_args.input_visibility = "public"
    run_args.param_visibility = "fixed"
    run_args.output_visibility = "public"
    return run_args
@time_function
def prove_model(config, rank ):
    proof_path = os.path.join(config.output_dir,f'proof_split_{rank}.pf')
    witness_path = os.path.join(config.output_dir,f'witness_split_{rank}.json')
    compiled_model_path = os.path.join(config.output_dir,f'network_{rank}.compiled')
    pk_path = os.path.join(config.output_dir,f'test_split_{rank}.pk')
    res = ezkl.prove( witness_path,compiled_model_path,pk_path,proof_path,"for-aggr")
    print(res)
    assert os.path.isfile(proof_path)
    return res["proof"]
@time_function
def verify_model(rank, config,res_1_proof):
    witness_path = os.path.join(config.output_dir,f'witness_split_{rank}.json')
    proof_path = os.path.join(config.output_dir,f'proof_split_{rank}.pf')
    vk_path = os.path.join(config.output_dir,f'test_split_{rank}.vk')
    settings_path = os.path.join(config.output_dir,f'settings_split_{rank}.json')
  
    if rank > 0:
        print("swapping commitments")
        # swap the proof commitments if we are not the first model
        prev_witness_path = os.path.join(config.output_dir,f'witness_split_{rank-1}.json')
        
        # Wait for the previous witness file to exist
        while not os.path.exists(prev_witness_path):
            time.sleep(0.01)

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
    print(f"verified_{rank}")




@time_function
def setup(compiled_model_path,vk_path,pk_path,settings_path):

    res = ezkl.setup(compiled_model_path,vk_path,pk_path)
    assert res == True
    assert os.path.isfile(vk_path)
    assert os.path.isfile(pk_path)
    assert os.path.isfile(settings_path)

#@time_function    
def mock(witness_path, compiled_model_path):
    res = ezkl.mock(witness_path, compiled_model_path)
    assert res == True

#@time_function
def gen_witness(data_path, compiled_model_path, witness_path):
    res = ezkl.gen_witness(data_path, compiled_model_path, witness_path)
    assert os.path.isfile(witness_path)

#@time_function
def get_srs(settings_path):
    res = ezkl.get_srs(settings_path)

#@time_function
def compile_circuit(model_path,compiled_model_path,settings_path):
    res = ezkl.compile_circuit(model_path, compiled_model_path, settings_path)
    assert res == True

#@time_function
def calibrate_settings(data_path, model_path,settings_path):
    res = ezkl.calibrate_settings(data_path, model_path, settings_path, "resources")
    assert res == True

#@time_function
def gen_settings(model_path, settings_path):
    res = ezkl.gen_settings(model_path, settings_path)
    assert res == True

#@time_function
def gen_iput_data(shape, cal_path):
    data_array = (torch.randn(20, *shape).detach().numpy()).reshape([-1]).tolist()
    data = dict(input_data = [data_array])
    json.dump(data, open(cal_path, 'w'))
    return data


#@time_function
def export_model_and_data(model, x, model_path, data_path, rank, output_path):
    # After training, export to onnx (network.onnx) and create a data file (input.json)
    print(x)

    # Flips the neural net into inference mode
    model.eval()
    model.to('cpu')

    # Export the model
    torch.onnx.export(model,               # model being run
                        x,                   # model input (or a tuple for multiple inputs)
                        model_path,            # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=10,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['input'],   # the model's input names
                        output_names = ['output'], # the model's output names
                        dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                        'output' : {0 : 'batch_size'}})
    if rank == 0:
        data_json = dict(input_data = [((x).detach().numpy()).reshape([-1]).tolist()])
        json.dump( data_json, open(data_path, 'w' ))
        input_names = ["input"]
        output_names = ["/AveragePool_output_0"]
        # first model
        onnx.utils.extract_model(model_path, output_path, input_names, output_names)

    elif rank == 1:
        inter_1 = model.split_1(x)
        data = dict(input_data = [((inter_1).detach().numpy()).reshape([-1]).tolist()])
        json.dump( data, open(data_path, 'w' ))
        input_names = ["/AveragePool_output_0"]
        output_names = ["output"]
        onnx.utils.extract_model(model_path, output_path, input_names, output_names)

if __name__ == "__main__":
    main()


