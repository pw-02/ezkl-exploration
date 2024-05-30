
import onnx



org_file = 'multi-job/mobilenet_test/network.onnx'
output_path1 = 'multi-job/mobilenet_test/network_0.onnx'
output_path2 = 'multi-job/mobilenet_test/network_1.onnx'

m1_input = ["data"]
m1_output = ["mobilenetv20_features_conv0_fwd"]

model = onnx.load(org_file)

for node in model.graph.initializer:
    print(node.name)
    print()


for node in model.graph.node:
    # # node inputs
    for idx, node_input_name in enumerate(node.input):
        print(idx, node_input_name)
    print()
    for idx, node_output_name in enumerate(node.output):
        print(idx, node_output_name)

onnx.utils.extract_model(org_file, output_path1, m1_input, m1_output, check_model=False)

m2_input = m1_output

for node in model.graph.initializer:
    if node.name != 'mobilenetv20_features_conv0_weight':
        m2_input.append(node.name)


m2_output = ["mobilenetv20_output_flatten0_reshape0"]
onnx.utils.extract_model(org_file, output_path2, m2_input, m2_output)

