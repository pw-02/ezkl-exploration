import onnx
from onnx import helper, ModelProto
from split_model import get_intermediate_outputs, split_onnx_model_at_every_node, run_inference_on_onnx_model, merge_onnx_models, run_inference_on_onnx_model_using_input_file
from utils import  analyze_onnx_model_for_zk_proving, load_onnx_model, read_json_file_to_dict

from collections import OrderedDict
import json

def group_models(dct:OrderedDict, n):
    items = list(dct.items())  # Convert dictionary items to a list of tuples
    grouped_items = [dict(items[i:i+n]) for i in range(0, len(items), n)]
    return grouped_items

def get_model_inputs(model, intermediate_values):
    input_data = []
    for input_tensor in model.graph.input:
        input_data.append(intermediate_values[input_tensor.name])
    return input_data

def test_single_model(onnx_model_path,json_input_file):
     
     combined_model_result = run_inference_on_onnx_model(onnx_model_path, json_input_file)
     print(f'Combined model result: {combined_model_result}')


def test_combine_model(onnx_model_path,json_input_file):
    full_model_result = run_inference_on_onnx_model_using_input_file(onnx_model_path, json_input_file)
    print(f'Full model result: {full_model_result}')
    
    intermediate_values = get_intermediate_outputs(onnx_model_path, json_input_file)
   
    print(f'Splitting model for distributed proving..')
    
    sub_models = split_onnx_model_at_every_node(onnx_model_path, json_input_file, intermediate_values, 'tmp')

    group_elements = group_models(sub_models, 2)
    for idx, group in enumerate(group_elements):
        print(f"Processing {idx+1}_path_to_merged_model")  
        merged_model = merge_onnx_models(group)
        settings = analyze_onnx_model_for_zk_proving(merged_model)
        input_data = get_model_inputs(merged_model, intermediate_values)
        # Save the merged model
        output_model_path = f'{idx+1}_path_to_merged_model.onnx'
        input_data_save_path = f'{idx+1}_input.json'

        onnx.save(merged_model, output_model_path)

        # proving_input = {"input_data": input_data}
        # with open(input_data_save_path, 'w') as json_file:
        #     json.dump(proving_input, json_file, indent=4)

        combined_model_result = run_inference_on_onnx_model(output_model_path, intermediate_values)
        # print(f"Global ONNX model saved to {output_model_path}")  

        print(f'Combined model result: {combined_model_result}')

if __name__ == '__main__':
    onnx_model_path = 'examples/onnx/mnist_classifier/network.onnx'
    json_input_file =  'examples/onnx/mnist_classifier/input.json'

    onnx_model_path = 'examples/onnx/mobilenet/mobilenetv2_050_Opset18.onnx'
    json_input_file =  'examples/onnx/mobilenet/input.json'

    onnx_model_path = 'examples/onnx/mnist_gan/network.onnx'
    json_input_file =  'examples/onnx/mnist_gan/input.json'
    

    test_combine_model(onnx_model_path,json_input_file)
    # test_single_model('8_path_to_merged_model.onnx','8_input.json')
