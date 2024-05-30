
import onnx

org_file = 'multi-job/nano_gpt_test/network.onnx'
output_path = 'multi-job/nano_gpt_test/network_p1.onnx'
input_names = ["input"]
output_names = ["/h.0/Add_1_output_0"]

model = onnx.load(org_file)

for node in model.graph:
    # node inputs
    for idx, node_input_name in enumerate(node.input):
        print(idx, node_input_name)
    # node outputs
    for idx, node_output_name in enumerate(node.output):
        print(idx, node_output_name)

onnx.utils.extract_model(org_file, output_path, input_names, output_names)

output_path = 'multi-job/nano_gpt_test/network_p2.onnx'
input_names = ["/Add_1_output_0"]
output_names = ["output"]

onnx.utils.extract_model(org_file, output_path, input_names, output_names)
