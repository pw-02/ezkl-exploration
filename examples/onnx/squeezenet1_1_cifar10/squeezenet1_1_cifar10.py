import torch
import torchvision
import numpy as np
import json

import onnx

def remove_identity_nodes(onnx_model_path, output_path):
    model = onnx.load(onnx_model_path)
    identity_map = {}

    # First, map Identity outputs to their inputs
    nodes_to_keep = []
    for node in model.graph.node:
        if node.op_type == "Identity":
            identity_map[node.output[0]] = node.input[0]
        else:
            nodes_to_keep.append(node)

    # Patch up all node inputs to skip Identity outputs
    for node in nodes_to_keep:
        node.input[:] = [identity_map.get(i, i) for i in node.input]

    # If graph outputs reference Identity outputs, patch them too
    for out in model.graph.output:
        if out.name in identity_map:
            out.name = identity_map[out.name]

    # Replace the node list with the filtered list
    del model.graph.node[:]
    model.graph.node.extend(nodes_to_keep)

    # Save cleaned model
    onnx.save(model, output_path)


# Create SqueezeNet with 10 output classes
model = torchvision.models.resnet18(num_classes=10)

# Dummy CIFAR-10 input
dummy = torch.randn(1, 3, 32, 32)

# Export to ONNX
torch.onnx.export(
    model, dummy, r"examples\onnx\squeezenet1_1_cifar10\squeezenet1_1_cifar10.onnx",
    input_names=['input'], 
    output_names=['output'], 
    opset_version=10,
    do_constant_folding=True,
    export_params=True
)


# Usage example:
remove_identity_nodes(
    r"examples\onnx\squeezenet1_1_cifar10\squeezenet1_1_cifar10.onnx",
    r"examples\onnx\squeezenet1_1_cifar10\squeezenet1_1_cifar10_no_identity.onnx"
)

# data = np.random.rand(1, 3, 32, 32).astype(float)
# flat_data = data.flatten().tolist()
# json_obj = {"input_data": flat_data}

# with open(r"examples\onnx\vgg\vgg16_input.json", "w") as f:
#     json.dump(json_obj, f, indent=2)
