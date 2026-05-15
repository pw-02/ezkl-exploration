import onnx
from onnx import TensorProto, numpy_helper

def dtype_name(t):
    return TensorProto.DataType.Name(t)

def dump(path):
    m = onnx.load(path)
    print("\n==", path, "==")

    print("\nInputs:")
    for x in m.graph.input:
        tt = x.type.tensor_type
        dims = [
            d.dim_value if d.HasField("dim_value") else d.dim_param
            for d in tt.shape.dim
        ]
        print(x.name, dtype_name(tt.elem_type), dims)

    print("\nInitializers:")
    init_map = {i.name: i for i in m.graph.initializer}
    for i in m.graph.initializer:
        arr = numpy_helper.to_array(i)
        print(i.name, dtype_name(i.data_type), list(i.dims), arr.tolist() if arr.size <= 20 else "")

    print("\nNodes:")
    for idx, n in enumerate(m.graph.node):
        print(idx, n.name or "<unnamed>", n.op_type)
        print("  inputs :", list(n.input))
        print("  outputs:", list(n.output))
        if n.op_type == "Reshape":
            shape_name = n.input[1]
            print("  Reshape shape input:", shape_name)
            if shape_name in init_map:
                print("  shape value:", numpy_helper.to_array(init_map[shape_name]))

dump("_tmp/split_mnist_classifier/model_9.onnx")
dump("_tmp/split_mnist_classifier/model_10.onnx")