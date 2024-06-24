import os
import numpy as np
import os
import shutil
import tempfile
from collections import OrderedDict
import tarfile
import nnef
import _nnef
from nnef.parser import *
from nnef.binary import read_tensor, write_tensor



def tgz_extract(file_path, dir_path):
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(dir_path)


def read_nnef(path, input_shapes=None, stdlib=None, decomposed=None, custom_shapes=None, infer_shapes=False, load_variables=True):
        compressed = os.path.splitext(path) in ['tgz', 'gz'] and not os.path.isdir(path)

        folder = None
        try:
            if compressed:
                folder = tempfile.mkdtemp(prefix="nnef_")
                tgz_extract(path, folder)
                path = folder

            if not os.path.isdir(path):
                raise IOError("NNEF model must be a (compressed) folder, but an uncompressed file was provided")

            nnef_graph = load_graph(path, stdlib=stdlib, lowered=decomposed, load_variables=load_variables)
            if infer_shapes:
                nnef.infer_shapes(nnef_graph, external_shapes=input_shapes or {}, custom_shapes=custom_shapes or {})

            # return _build_graph(nnef_graph)
        finally:
            if folder is not None:
                shutil.rmtree(folder)
        return nnef_graph



def load_graph(path, stdlib=None, lowered=None, load_variables=True):
    if os.path.isfile(path):
        return parse_file(path, stdlib=stdlib, lowered=lowered)

    graph_fn = os.path.join(path, 'graph.nnef')
    quant_fn = os.path.join(path, 'graph.quant')

    graph = parse_file(graph_fn, quant_fn if os.path.isfile(quant_fn) else None, stdlib=stdlib, lowered=lowered)

    if load_variables:
        for operation in graph.operations:
            if operation.name == 'variable':
                variable_filename = operation.attribs['label'] + '.dat'
                if variable_filename.startswith(os.path.sep):
                    variable_filename = variable_filename[1:]
                variable_filename = os.path.join(path, variable_filename)
                tensor_name = operation.outputs['output']
                with open(variable_filename) as variable_file:
                    data = read_tensor(variable_file)

                data_shape = list(data.shape)
                shape = operation.attribs['shape']
                if data_shape != shape:
                    raise _nnef.Error('shape {} in variable file does not match shape {} defined in network structure'
                                      .format(data_shape, shape))

                tensor = graph.tensors[tensor_name]
                graph.tensors[tensor_name] = _nnef.Tensor(tensor.name, tensor.dtype, data_shape, data, tensor.quantization)

    return graph

if __name__ == '__main__':
    variable_filename = 'nnef_to_onnx/simple_cnn.onnx.nnef/variable3.dat'
    with open(variable_filename) as variable_file:
        data = read_tensor(variable_file)
    ezkl_example = 'nnef_to_onnx/simple_cnn.nnef'
    working_example = 'nnef_to_onnx/simple_cnn.onnx.nnef'

    nnef_graph = read_nnef(ezkl_example)
    print()
