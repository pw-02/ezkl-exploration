import onnx
from collections import Counter, defaultdict
import numpy as np

model_path = "onnx_graphs/bert/bert_tiny.onnx"
model = onnx.load(model_path)

# Count raw ONNX node types
op_counts = Counter(node.op_type for node in model.graph.node)

print("ONNX operator counts:")
for op, count in op_counts.most_common():
    print(f"{op:20s} {count}")

# Count parameters from initializers
initializer_sizes = {
    init.name: np.prod(init.dims) for init in model.graph.initializer
}

total_params = sum(initializer_sizes.values())
print(f"\nTotal initializer parameters: {total_params:,}")

# Attribute parameters to operators that consume initializer tensors
op_param_counts = defaultdict(int)

for node in model.graph.node:
    for inp in node.input:
        if inp in initializer_sizes:
            op_param_counts[node.op_type] += initializer_sizes[inp]

print("\nParameter-associated operators:")
for op, count in sorted(op_param_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{op:20s} {count:,}")