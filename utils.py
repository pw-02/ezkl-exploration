import onnx

def list_layers(onnx_model_path):
    onnx_model = onnx.load(onnx_model_path)
    layers = []
    for node in onnx_model.graph.node:
        layers.append(node.name)
    return layers

