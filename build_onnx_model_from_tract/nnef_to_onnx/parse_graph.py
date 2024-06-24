import re
from typing import List, Dict, Optional, Union

class Op:
    pass

class TypedSource(Op):
    def __init__(self, fact: str):
        self.fact = fact

class Const(Op):
    def __init__(self, *values: Union[str, float]):
        self.values = values

class Conv(Op):
    def __init__(self, pool_spec: Dict, kernel_fmt: str, group: int, q_params: Optional[str]):
        self.pool_spec = pool_spec
        self.kernel_fmt = kernel_fmt
        self.group = group
        self.q_params = q_params

class TypedBinOp(Op):
    def __init__(self, op_type: str, q_params: Optional[str]):
        self.op_type = op_type
        self.q_params = q_params

class MaxPool(Op):
    def __init__(self, pool_spec: Dict, with_index_outputs: Optional[str]):
        self.pool_spec = pool_spec
        self.with_index_outputs = with_index_outputs

class Reshape(Op):
    def __init__(self, reshape_type: int, input_shape: List[int], output_shape: List[int]):
        self.reshape_type = reshape_type
        self.input_shape = input_shape
        self.output_shape = output_shape

class EinSum(Op):
    def __init__(self, equation: str):
        self.equation = equation

class Node:
    def __init__(self, id: int, name: str, inputs: List[str], op: Op, outputs: List[str]):
        self.id = id
        self.name = name
        self.inputs = inputs
        self.op = op
        self.outputs = outputs

class Graph:
    def __init__(self, nodes: List[Node], inputs: List[str], outputs: List[str], outlet_labels: Dict[str, str], properties: Dict, symbol_table: Dict):
        self.nodes = nodes
        self.inputs = inputs
        self.outputs = outputs
        self.outlet_labels = outlet_labels
        self.properties = properties
        self.symbol_table = symbol_table

def parse_op(op_str: str) -> Op:
    if op_str.startswith("TypedSource"):
        fact = re.search(r"fact: ([^}]+)", op_str).group(1)
        return TypedSource(fact)
    elif op_str.startswith("Const"):
        values = re.findall(r"([-\d.]+(?:e[-+]?\d+)?|F32|[-\w]+)", op_str)
        return Const(*values)
    elif op_str.startswith("Conv"):
        pool_spec = eval(re.search(r"pool_spec: ({[^}]+})", op_str).group(1))
        kernel_fmt = re.search(r"kernel_fmt: (\w+)", op_str).group(1)
        group = int(re.search(r"group: (\d+)", op_str).group(1))
        q_params = re.search(r"q_params: (\w+)", op_str)
        q_params = q_params.group(1) if q_params else None
        return Conv(pool_spec, kernel_fmt, group, q_params)
    elif op_str.startswith("TypedBinOp"):
        op_type = re.search(r"TypedBinOp\(([^,]+)", op_str).group(1)
        q_params = re.search(r", (None)?\)", op_str).group(1)
        return TypedBinOp(op_type, q_params)
    elif op_str.startswith("MaxPool"):
        pool_spec = eval(re.search(r"pool_spec: ({[^}]+})", op_str).group(1))
        with_index_outputs = re.search(r"with_index_outputs: (\w+)", op_str)
        with_index_outputs = with_index_outputs.group(1) if with_index_outputs else None
        return MaxPool(pool_spec, with_index_outputs)
    elif op_str.startswith("Reshape"):
        params = eval(re.search(r"Reshape\(([^)]+)\)", op_str).group(1))
        return Reshape(params[0], params[1], params[2])
    elif op_str.startswith("EinSum"):
        equation = re.search(r"EinSum (\S+)", op_str).group(1)
        return EinSum(equation)
    else:
        raise ValueError(f"Unknown op: {op_str}")

def parse_node(node_str: str) -> Node:
    # Regular expressions to extract each attribute
    id_pattern = r'id: (\d+)'
    name_pattern = r'name: "(.+?)"'
    inputs_pattern = r'inputs: \[(.*?)\]'
    op_pattern = r'op: (.+?)\}'
    outputs_pattern = r'outputs: \[(.*?)\]'

    # Extracting attributes using regex
    node_id = re.search(id_pattern, node_str).group(1)
    node_name = re.search(name_pattern, node_str).group(1)
    inputs = re.search(inputs_pattern, node_str).group(1).split(', ') if re.search(inputs_pattern, node_str) else []
    op = re.search(op_pattern, node_str).group(1).strip()
    outputs = re.search(outputs_pattern, node_str).group(1).split(', ') if re.search(outputs_pattern, node_str) else []
    return Node(node_id, node_name, inputs, op, outputs)

def parse_graph(graph_str: str) -> Graph:
    nodes_str = re.search(r"nodes: \[([^\]]+)\]", graph_str)
    pattern = r'Node \{(?:[^{}]|(?:\{[^{}]*\}))*\}'
    nodes_strs = re.findall(pattern, nodes_str.string)
    nodes = [parse_node(node_str) for node_str in nodes_strs]
    inputs = re.findall(r"inputs: \[([^\]]*)\]", graph_str)[0].split(", ") if "inputs: [" in graph_str else []
    outputs = re.findall(r"outputs: \[([^\]]*)\]", graph_str)[0].split(", ") if "outputs: [" in graph_str else []
    outlet_labels = eval(re.search(r"outlet_labels: ({[^}]+})", graph_str).group(1))
    # outlet_labels = None
    # properties = eval(re.search(r"properties: ({[^}]+})", graph_str).group(1))
    # symbol_table = eval(re.search(r"symbol_table: ({[^}]+})", graph_str).group(1))
    # return Graph(nodes, inputs, outputs, outlet_labels, properties, symbol_table)

# Example usage
with open('nnef_to_onnx/new 16.txt', 'r') as file:
    data = file.read()
graph = parse_graph(data)
pass
