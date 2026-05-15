import onnx
from onnx import TensorProto

def dtype_name(t):
    return TensorProto.DataType.Name(t)

def dump(path):
    print("\n===", path, "===")
    m = onnx.load(path)

    print("\nInputs:")
    for x in m.graph.input:
        tt = x.type.tensor_type
        dims = [
            d.dim_value if d.HasField("dim_value") else d.dim_param
            for d in tt.shape.dim
        ]
        print(x.name, dtype_name(tt.elem_type), dims)

    print("\nInitializers:")
    for init in m.graph.initializer[:20]:
        print(init.name, dtype_name(init.data_type), list(init.dims))

    print("\nFirst 20 nodes:")
    for i, n in enumerate(m.graph.node[:20], 1):
        print(i, n.name or "<unnamed>", n.op_type)
        print("  inputs :", list(n.input))
        print("  outputs:", list(n.output))

dump("experiments/models/nanoGPT/nano_gpt_4_layers_64_embd.onnx")
dump("_tmp/split_nanoGPT/model_1.onnx")
import onnx

def dump_meta(path):
    m = onnx.load(path)
    print("\n==", path)
    print("ir_version:", m.ir_version)
    print("opsets:", [(o.domain, o.version) for o in m.opset_import])
    print("producer:", m.producer_name, m.producer_version)
    print("value_info count:", len(m.graph.value_info))
    print("doc:", repr(m.graph.doc_string))

    for i in m.graph.input:
        print("input raw:", i)

dump_meta("experiments/models/nanoGPT/nano_gpt_4_layers_64_embd.onnx")
dump_meta("_tmp/split_nanoGPT/model_1.onnx")

import onnx

parent = onnx.load("experiments/models/nanoGPT/nano_gpt_4_layers_64_embd.onnx")
m = onnx.load("_tmp/split_nanoGPT/model_1.onnx")

m.producer_name = parent.producer_name
m.producer_version = parent.producer_version

del m.graph.value_info[:]

onnx.checker.check_model(m)
onnx.save(m, "_tmp/split_nanoGPT/model_1_clean.onnx")