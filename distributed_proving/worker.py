import grpc
import zkpservice_pb2_grpc as pb2_grpc
import zkpservice_pb2 as pb2
from concurrent import futures
import logging
from datetime import datetime
import ezkl
import os
import json
from log_utils import ExperimentLogger, time_function, ResourceMonitor
import uuid
import threading
import shutil
import pandas as pd
import csv

# Global logging configuration
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("ZKPWorker")

class EZKLProver:
    def __init__(self, worker_dir: str, log_dir: str, overwrite=False):
        self.directory = worker_dir
        self.model_path = os.path.join(self.directory, 'model.onnx')
        self.data_path = os.path.join(self.directory, 'input.json')
        self.compiled_model_path = os.path.join(self.directory, 'network.compiled')
        self.pk_path = os.path.join(self.directory, 'key.pk')
        self.vk_path = os.path.join(self.directory, 'key.vk')
        self.settings_path = os.path.join(self.directory, 'settings.json')
        self.witness_path = os.path.join(self.directory, 'witness.json')
        self.proof_path = os.path.join(self.directory, 'test.pf')
        self.exp_logger = ExperimentLogger(log_dir=log_dir)
        self.overwrite = overwrite

    @time_function
    def gen_settings(self):
        if not self.overwrite and os.path.isfile(self.settings_path):
            return True
        ezkl.gen_settings(self.model_path, self.settings_path)
        ezkl.calibrate_settings(self.data_path, self.model_path, self.settings_path, "resources")

    @time_function
    def compile_circuit(self):
        if not self.overwrite and os.path.isfile(self.compiled_model_path):
            return True
        ezkl.compile_circuit(self.model_path, self.compiled_model_path, self.settings_path)

    @time_function
    def get_srs(self):
        if not self.overwrite and os.path.isfile(self.vk_path):
            return True
        ezkl.get_srs(self.settings_path)

    @time_function
    def gen_witness(self):
        ezkl.gen_witness(self.data_path, self.compiled_model_path, self.witness_path)
        assert os.path.isfile(self.witness_path)

    @time_function
    def setup(self):
        if not self.overwrite and os.path.isfile(self.pk_path):
            return True
        ezkl.setup(self.compiled_model_path, self.vk_path, self.pk_path)
        self._cleanup_files(['halo2_ffts.csv', 'halo2_msms.csv', 'halo2_prover.csv'])
        assert os.path.isfile(self.vk_path)
        assert os.path.isfile(self.pk_path)
        assert os.path.isfile(self.settings_path)

    @time_function
    def prove(self):
        ezkl.prove(self.witness_path, self.compiled_model_path, self.pk_path, self.proof_path, "single")
        assert os.path.isfile(self.proof_path)

    @time_function
    def verify(self):
        try:
            res = ezkl.verify(self.proof_path, self.settings_path, self.vk_path)
            self.exp_logger.log_value('verified', str(res))
        except Exception as e:
            logger.exception("Error in verification: %s", e)
            self.exp_logger.log_value('verified', "False")
            return False

    def run_end_to_end_proof(self):
        with ResourceMonitor() as monitor:
            functions = [
                ('gen_settings', self.gen_settings),
                ('compile_circuit', self.compile_circuit),
                ('get_srs', self.get_srs),
                ('gen_witness', self.gen_witness),
                ('setup', self.setup),
                ('prove', self.prove),
                ('verify', self.verify)
            ]
            for func_name, func in functions:
                execution_time = func()
                self.exp_logger.log_value(f'ezkl_{func_name}_time(s)', execution_time)
            self.exp_logger.log_env_resources()

            resource_data = monitor.resource_data
            self.exp_logger.log_value("resource_usage", resource_data)

            return self.proof_path, self.exp_logger.data

    def _cleanup_files(self, files):
        for file in files:
            if os.path.isfile(file):
                os.remove(file)

class ZKPWorkerServicer(pb2_grpc.ZKPWorkerServiceServicer):
    def __init__(self):
        self.computed_proof = None
        self.log_dir = "worker_logs"
        self.requests = {}
        self.lock = threading.Lock()
        self.tmp_dir = "data"
        os.makedirs(self.tmp_dir, exist_ok=True)

    def Ping(self, request, context):
        logger.info("Received Ping Request from %s", request.message)
        return pb2.MessageResponse(message='pong', received=True)

    def ComputeProof(self, request, context):
        request_id = str(uuid.uuid4())
        logger.info("Received 'Compute Proof' request with ID %s", request_id)
        with self.lock:
            self.requests[request_id] = {'completed': False}  # Initialize request status
        threading.Thread(target=self.process_request, args=(request_id, request)).start()
        return pb2.ProofResponse(request_id=request_id, message="Request received")

    def process_request(self, request_id, request):
        working_dir = os.path.join(self.tmp_dir, request.model_id)
        os.makedirs(working_dir, exist_ok=True)

        try:
            self._save_data(request, working_dir)
            prover = EZKLProver(working_dir, self.log_dir)
            proof_path, performance_data = prover.run_end_to_end_proof()

            if not request.cache_setup_files:
                shutil.rmtree(working_dir)

            with self.lock:
                self.requests[request_id] = {
                    'completed': True,
                    'success': True,
                    'proof': 'proof'.encode('utf-8'),
                    'performance_data': performance_data
                }

            logger.info("Proof computed and verified for request ID %s", request_id)

        except Exception as e:
            logger.error(f"Failed to compute proof for request {request_id}: {e}")
            with self.lock:
                self.requests[request_id] = {
                    'completed': True,
                    'success': False,
                    'proof': None,
                    'performance_data': {'error_message': str(e)}
                }

    def _save_data(self, request, working_dir):
        with open(os.path.join(working_dir, 'model.onnx'), 'wb') as f:
            f.write(request.onnx_model)

        model_input = json.loads(request.input_data)
        with open(os.path.join(working_dir, 'input.json'), 'w') as f:
            json.dump(model_input, f)

    def CheckProofStatus(self, request, context):
        request_id = request.request_id
        with self.lock:
            request_data = self.requests.get(request_id)

        if request_data and request_data['completed']:
            if request_data['success']:
                halo2_metrics = self._get_metrics()
                request_data['performance_data'].update(halo2_metrics)
            return pb2.ProofStatusResponse(
                is_completed=request_data['completed'],
                is_success=request_data['success'],
                proof=request_data['proof'],
                performance_data=json.dumps(request_data['performance_data']),
                message="Completed"
            )

        return pb2.ProofStatusResponse(
            is_completed=request_data['completed'] if request_data else False,
            is_success=False,
            message='in progress' if request_data else 'unknown request'
        )

    def _get_metrics(self):
        halo2_metrics = {}
        if  os.path.isfile('halo2_circuit.csv'):
                #read in csv file
                with open('halo2_circuit.csv', mode='r') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        for key, value in row.items():
                            key = f"halo2_{key}"
                            halo2_metrics[key] = value
                #delete the file
                os.remove('halo2_circuit.csv')
            
        if os.path.isfile('halo2_ffts.csv'):
                halo2_metrics.update(self.get_fft_logs('halo2_ffts.csv'))
                os.remove('halo2_ffts.csv')

        if os.path.isfile('halo2_msms.csv'):
                halo2_metrics.update(self.get_msm_logs('halo2_msms.csv'))
                os.remove('halo2_msms.csv')
            
        if os.path.isfile('halo2_prover.csv'):
                halo2_metrics.update(self.get_prover_logs('halo2_prover.csv'))
                os.remove('halo2_prover.csv')
        return halo2_metrics
    


    def get_fft_logs(self, fft_file):
        fft_metrics = {}
        df = pd.read_csv(fft_file)  # Replace 'your_file.csv' with the actual file path
        # Calculate the total number of FFTs
        fft_metrics['halo2_fft_count'] = int(len(df))
        fft_metrics['halo2_largest_fft'] = int(df['size'].max())
        fft_metrics['halo2_total_fft_time(s)'] = float(df['duration(s)'].sum())
        fft_metrics['halo2_avg_fft_time(s)'] = float(df['duration(s)'].mean())
        fft_metrics['halo2_fft_device'] = str(df['device'].iloc[0])
        # Convert the DataFrame to a dictionary
        data_dict = df.to_dict(orient='records')  # 'records' format creates a list of dictionaries
        fft_data = json.dumps(data_dict)
        fft_metrics['fft_data'] = fft_data
        return fft_metrics     
    
    def get_prover_logs(self,prover_metrics_file):
        prover_metrics = {}
        with open(prover_metrics_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                for key, value in row.items():
                    key = f"halo2_{key}"
                    prover_metrics[key] = value
        return prover_metrics
    
    def get_msm_logs(self,msm_file):
        msm_metrics = {}
        df = pd.read_csv(msm_file)  # Replace 'your_file.csv' with the actual file path
        # Calculate the total number of MSMs
        msm_metrics['halo2_msm_count'] = int(len(df))
        msm_metrics['halo2_largest_msm'] = int(df['num_coeffs'].max())
        msm_metrics['halo2_total_msm_time(s)'] = float(df['duration(s)'].sum())
        msm_metrics['halo2_avg_msm_time(s)'] = float(df['duration(s)'].mean())
        msm_metrics['halo2_msm_device'] = str(df['device'].iloc[0])

        # Calculate the average duration

        # Convert the DataFrame to a dictionary
        data_dict = df.to_dict(orient='records')  # 'records' format creates a list of dictionaries
        msm_data = json.dumps(data_dict)
        msm_metrics['msm_data'] = msm_data
        return msm_metrics
    
    


def serve(port):
    try:
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        pb2_grpc.add_ZKPWorkerServiceServicer_to_server(ZKPWorkerServicer(), server)
        server.add_insecure_port(f'[::]:{port}')
        server.start()
        logging.info(f"Worker started on port {port}...")
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("Server stopped due to keyboard interrupt")
        server.stop(0)
    except Exception as e:
        logging.exception("Error in server: %s", e)
    finally:
        server.stop(0)
        # logging.info("Server stopped")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run gRPC server")
    parser.add_argument("--port", type=int, default=50051, help="Port number for the server to listen on")
    args = parser.parse_args()
    serve(args.port)

if __name__ == '__main__':
    main()
