import onnx
from collections import defaultdict

FULL = "experiments/models/mnist_classifier/mnist_classifier.onnx"
SPLIT = "_tmp/split_mnist_classifier/model_9.onnx"

def index_model(path):
    m = onnx.load(path)
    producers = {}
    consumers = defaultdict(list)

    for i, node in enumerate(m.graph.node):
        for out in node.output:
            producers[out] = (i, node)
        for inp in node.input:
            consumers[inp].append((i, node))

    return m, producers, consumers

def show_node(i, node):
    print(f"{i}: {node.name} {node.op_type}")
    print("  inputs :", list(node.input))
    print("  outputs:", list(node.output))

def inspect(path):
    m, producers, consumers = index_model(path)
    print("\n====================")
    print(path)
    print("====================")

    print("\nGraph inputs:")
    for x in m.graph.input:
        print(" ", x.name, x.type)

    print("\nNodes:")
    for i, n in enumerate(m.graph.node):
        show_node(i, n)

    print("\nInputs that have no producer inside this model:")
    for x in m.graph.input:
        name = x.name
        print("\nINPUT:", name)
        print("  produced inside split?", name in producers)
        print("  consumed by:")
        for item in consumers.get(name, []):
            show_node(*item)

inspect(FULL)
inspect(SPLIT)