import hydra
from omegaconf import DictConfig
import os 
import little_transformerr
import mnist_classifier
import nano_gpt
import logging
import torch
import json
import ezkl
import time
from functools import wraps


FORMAT = '%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT)
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


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(config: DictConfig):
    # model: config.model
    config.output_dir = os.path.join(config.output_dir, config.model.name)
    os.makedirs( config.output_dir, exist_ok=True)
    model_path = os.path.join(config.output_dir,'network.onnx')
    compiled_model_path = os.path.join(config.output_dir,'network.compiled')
    pk_path = os.path.join(config.output_dir,'key.pk')
    vk_path = os.path.join(config.output_dir,'key.vk')
    settings_path = os.path.join(config.output_dir,'settings.json')
    witness_path = os.path.join(config.output_dir,'witness.json')
    data_path = os.path.join(config.output_dir,'input.json')
    cal_path = os.path.join(config.output_dir,"calibration.json")
    proof_path = os.path.join(config.output_dir,'test.pf')

    if config.model.name == 'little_transformer':
        model, shape, data_point =  little_transformerr.get_model(
            seq_len=config.model.seq_len,
            batch_size=config.model.batch_size,
            max_epochs=config.model.max_epochs,
            max_value=config.model.max_value,
            layer_count=config.model.layer_count,
            embed_dim=config.model.embed_dim,
            num_heads=config.model.num_heads,
            ff_dim=config.model.ff_dim)
    

    elif config.model.name == 'mnist_classifier':
        model, shape, data_point =  mnist_classifier.get_model()
    
    elif config.model.name == 'nano_gpt':
        model, shape, data_point =  nano_gpt.get_model(num_layers=config.model.num_layers)



    # with open(data_path, 'w') as f:
    #     json.dump(example_input_data, f)
    
    export_model_and_data(model=model, x=data_point, model_path=model_path, data_path=data_path )

    gen_settings(model_path, settings_path)
    
    calibrate_settings(data_path, model_path, settings_path)

    compile_circuit(model_path, compiled_model_path, settings_path)
    
    #generate the srs
    get_srs(settings_path) 

    #generate the witness file 
    gen_witness(data_path, compiled_model_path, witness_path)

    # #move proves
    # mock(witness_path, compiled_model_path)

    #runs setup
    setup(compiled_model_path,vk_path,pk_path,settings_path)

    #GENERATE A PROOF
    prove(witness_path,compiled_model_path,pk_path,proof_path)

    #verify
    verify(proof_path,settings_path,vk_path,)


# #@time_function
def verify(proof_path,settings_path,vk_path):
    res = ezkl.verify(proof_path,settings_path,vk_path,)
    assert res == True
    print("verified")

#@time_function
def prove(witness_path,compiled_model_path,pk_path,proof_path):
    res = ezkl.prove(witness_path,compiled_model_path,pk_path,proof_path, "single",)
    print(res)
    assert os.path.isfile(proof_path)

#@time_function
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
def export_model_and_data(model, x, model_path, data_path):
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

    data_array = ((x).detach().numpy()).reshape([-1]).tolist()

    data_json = dict(input_data = [data_array])

    print(data_json)

    # Serialize data into file:
    json.dump( data_json, open(data_path, 'w' ))

if __name__ == "__main__":
    main()