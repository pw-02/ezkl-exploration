import nnef_tools.io.nnef as nnef_io
import nnef_tools.io.onnx as onnx_io
from nnef_to_onnx.reader import Reader
import nnef_to_onnx.converter as converter
import os
import tarfile
import gzip
import shutil
import os
import nnef
import onnx

def create_tgz(input_tar, output_tgz):
    with open(input_tar, 'rb') as f_in:
        with gzip.open(output_tgz, 'wb') as f_out:
            f_out.writelines(f_in)

def create_gz_file(input_path, output_filename):
    try:
        with open(input_path, 'rb') as f_in:
            with gzip.open(output_filename, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"Successfully created {output_filename}")
    except Exception as e:
        print(f"Failed to create {output_filename}. Error: {e}")

def main():
    nnef_model_path = "simple_cnn/graph.nnef"
    nnef_model_path = 'model.nnef.tgz'
    graph = nnef.load_graph(nnef_model_path)
    pass


def main1():
    # folder_path = 'simple_cnn'
    # input_tar = 'output.tar'  # Output .gz file name
    output_filename = 'model.nnef.tgz'

    # create_gz_file(input_tar,output_filename)
    # create_tar_gz(folder_path, output_filename)
    # create_gz_file (folder_path, output_filename)
    compressed = os.path.splitext(output_filename) in ['.tgz', '.gz'] and not os.path.isdir(output_filename)

    nnef_to_onnx_converter = converter.Converter()
    nnef_reader = Reader(custom_shapes=nnef_to_onnx_converter.defined_shapes(), decomposed=nnef_to_onnx_converter.decomposed_operations())

    nnef_graph = nnef_reader(output_filename)
    onnx_graph = nnef_to_onnx_converter(nnef_graph)
    onnx_writer = onnx_io.Writer()
    onnx_writer(onnx_graph, output_filename + '.onnx')


if __name__ == '__main__':
    main1()