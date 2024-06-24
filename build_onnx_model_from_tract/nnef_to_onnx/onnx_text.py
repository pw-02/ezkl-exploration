# Copyright (c) 2020 The Khronos Group Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import nnef_tools.io.nnef as nnef_io
import nnef_tools.io.onnx as onnx_io
import nnef_tools.conversion.onnx_to_nnef as onnx_to_nnef
import nnef_tools.conversion.nnef_to_onnx as nnef_to_onnx
import nnef_tools.optimization.nnef_optimizer as nnef_opt
import nnef_tools.optimization.onnx_optimizer as onnx_opt
import numpy as np
import unittest
import tempfile
import onnx
import sys
import os
from onnx import helper, TensorProto
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE


UNITTEST_FOLDER = os.environ.get('UNITTEST_FOLDER')


class TestEnv(unittest.TestCase):

    _type_to_numpy = {
        "tensor(float)": np.float32,
        "tensor(double)": np.float64,
        "tensor(int8)": np.int8,
        "tensor(int16)": np.int16,
        "tensor(int32)": np.int32,
        "tensor(int64)": np.int64,
        "tensor(uint8)": np.uint8,
        "tensor(uint16)": np.uint16,
        "tensor(uint32)": np.uint32,
        "tensor(uint64)": np.uint64,
        "tensor(bool)": np.bool_,
    }

    _network_folder = os.path.join(UNITTEST_FOLDER, 'onnx/nets/') if UNITTEST_FOLDER else None
    _output_folder = os.path.join(UNITTEST_FOLDER, 'onnx/ops/') if UNITTEST_FOLDER else None
    _infer_shapes = False
    _optimize = True

    def setUp(self) -> None:
        self._onnx_reader = onnx_io.Reader(simplify=True)
        self._onnx_writer = onnx_io.Writer()
        self._nnef_optimizer = nnef_opt.Optimizer()
        self._onnx_optimizer = onnx_opt.Optimizer()
        self._onnx_to_nnef_converter = onnx_to_nnef.Converter(infer_shapes=self._infer_shapes)
        self._nnef_to_onnx_converter = nnef_to_onnx.Converter()
        self._nnef_reader = nnef_io.Reader(custom_shapes=self._nnef_to_onnx_converter.defined_shapes(),
                                           decomposed=self._nnef_to_onnx_converter.decomposed_operations())
        self._nnef_writer = nnef_io.Writer(fragments=self._onnx_to_nnef_converter.defined_operations(),
                                           fragment_dependencies=self._onnx_to_nnef_converter.defined_operation_dependencies())

    def tearDown(self) -> None:
        pass

    def _convert_to_nnef(self, filename):
        onnx_graph = self._onnx_reader(filename)
        if self._optimize:
            onnx_graph = self._onnx_optimizer(onnx_graph)
        nnef_graph = self._onnx_to_nnef_converter(onnx_graph)
        if self._optimize:
            nnef_graph = self._nnef_optimizer(nnef_graph)
        self._nnef_writer(nnef_graph, filename + '.nnef')

    def _convert_from_nnef(self, filename):
        nnef_graph = self._nnef_reader(filename)
        onnx_graph = self._nnef_to_onnx_converter(nnef_graph)
        self._onnx_writer(onnx_graph, filename + '.onnx')

    @staticmethod
    def _random_data(dtype, shape):
        if dtype == bool:
            return np.random.random(shape) > 0.5
        else:
            return np.random.random(shape).astype(dtype)

    @staticmethod
    def _exec_model(filename):
        import onnxruntime
        np.random.seed(0)

        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

        session = onnxruntime.InferenceSession(filename, sess_options=options,
                                               providers=['CPUExecutionProvider'])

        inputs = {input.name: TestEnv._random_data(TestEnv._type_to_numpy[input.type], input.shape)
                  for input in session.get_inputs()}
        outputs = session.run([output.name for output in session.get_outputs()], inputs)

        return outputs

    @staticmethod
    def _create_tensor(value_info, data):
        name, shape, dtype = onnx_io.reader._get_value_info(value_info)
        if data is None:
            data = TestEnv._random_data(dtype, shape)
        elif not isinstance(data, np.ndarray):
            data = np.array(data)
        return helper.make_tensor(name, NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)], shape, vals=data.flat)

    @staticmethod
    def _create_model(name, nodes, inputs, outputs, constants, values, opset_version, ir_version):
        tensors = [TestEnv._create_tensor(item, values.get(item.name)) for item in constants]
        graph_def = helper.make_graph(nodes, name, inputs, outputs, value_info=constants, initializer=tensors)
        model_def = helper.make_model(graph_def, producer_name='nnef-to-onnx-test')
        model_def.opset_import[0].version = opset_version
        model_def.ir_version = ir_version
        onnx.checker.check_model(model_def, full_check=True)
        return model_def

    @staticmethod
    def _save_model(model_def, filename):
        with open(filename, 'wb') as file:
            file.write(model_def.SerializeToString())

    def _test_conversion(self, name, nodes, inputs, outputs, constants=None, values=None, opset_version=11, ir_version=6, epsilon=1e-5):
        filename = tempfile.mktemp() if self._output_folder is None else TestEnv._output_folder + name + '.onnx'
        model_def = self._create_model('G', nodes, inputs, outputs, constants or [], values or {}, opset_version, ir_version)
        self._save_model(model_def, filename)
        self._test_conversion_from_file(filename, epsilon=epsilon)

    def _test_conversion_from_file(self, filename, epsilon=1e-5):
        self._convert_to_nnef(filename)
        self._convert_from_nnef(filename + '.nnef')

        original_outputs = self._exec_model(filename)
        converted_outputs = self._exec_model(filename + '.nnef.onnx')

        self.assertEqual(len(original_outputs), len(converted_outputs))
        for original, converted in zip(original_outputs, converted_outputs):
            if original.dtype == bool:
                self.assertTrue(np.all(original == converted))
            else:
                diff = np.max(np.abs(original - converted))
                self.assertLess(diff, epsilon)

   
class TestCases(TestEnv):
    def test_shufflenet_v2(self):
        self._test_conversion_from_file(self._network_folder + 'shufflenet_v2.onnx', epsilon=1e-4)


if __name__ == '__main__':

    env = TestEnv()
    env.setUp()
    env._convert_from_nnef('nnef_to_onnx/simple_cnn.onnx.nnef')
    # env._test_conversion_from_file('nnef_to_onnx/simple_cnn.onnx', epsilon=1e-4)
