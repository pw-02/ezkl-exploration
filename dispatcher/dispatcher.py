import grpc
import queue
import threading
import onnx
import ezkl
import json
import os
from typing import List
from utils.helpers import read_json_file_to_dict
from concurrent.futures import ThreadPoolExecutor
import generated.jobs_pb2 as jobs_pb2
import generated.jobs_pb2_grpc as jobs_pb2_grpc
from enum import Enum
import time

class JobStatus(Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class GlobalProvingJob():
    def __init__(self, job_id, job_name, input_data, onnx_model_path, num_of_splits, data_directory) ):
        self.job_id = job_id
        self.model_name = job_name
        self.input_data = input_data
        self.onnx_model_path = onnx_model_path
        self.num_of_splits = num_of_splits
        self.status = JobStatus.PENDING
        self.data_directory = data_directory
        self.model_to_prove: List[OnnxModel] = []

class OnnxModel():   
    def __init__(self, model_name, input_data:str, onnx_model_path:str, combined_splits = None, gen_info_on_init=True):
        if onnx_model_path is None:
            raise TypeError("Model path or model proto must be provided")
        self.model_name = model_name
        self.input_data = input_data
        self.onnx_model_path = onnx_model_path
        self.isprocessed = False
        self.computed_witness = None
        self.computed_proof = None
        self.combined_splits = combined_splits
        if gen_info_on_init:
            self.generate_model_info()

    def generate_model_info(self):
        tmp_settings_file = 'tmp_settings.json'
        onnx_model = onnx.load(self.onnx_model_path)
        num_model_ops = len(onnx_model.graph.node)
        num_model_params =  0
        for initializer in onnx_model.graph.initializer:
            param_array = onnx.numpy_helper.to_array(initializer)
            num_model_params += param_array.size
        #get ezkl settings
        ezkl.gen_settings(self.onnx_model_path, tmp_settings_file)
        try:
            with open(tmp_settings_file, 'r') as f:
                ezkl_settings = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error reading JSON settings file: {e}")
            ezkl_settings = {}

        self.model_info = {"num_model_ops": num_model_ops,"num_model_params": num_model_params,}
        self.model_info = {**self.model_info, **ezkl_settings}
        #delete temp file
        os.remove(tmp_settings_file)  

class DispatcherService(jobs_pb2_grpc.DispatcherServicer):
    def __init__(self):
        self.main_job_queue = queue.Queue()  # Stores main jobs submitted to dispatcher by clients
        self.sub_job_queue = queue.Queue()  # Stores sub-jobs for workers to process
        self.job_status = {}  # {job_id: (status, result)}
        self.worker_addresses = ["localhost:50052", "localhost:50053"]
        self.data_directory = "data"
        self.log_directory = "logs"
        self.lock = threading.Lock()
        threading.Thread(target=self.process_main_jobs, daemon=True).start()
        threading.Thread(target=self.process_sub_jobs, daemon=True).start()

    def SubmitJob(self, request, context):
        """Receives a job and adds it to the main job queue after verifying the inputs."""
        #generte a unique job id
        job_id = str(int(time.time()))  # Unique job ID based on timestamp
        job_name = request.job_name
        self.input_data_path = request.input_data
        model_name = request.model_name
        onnx_model_path = request.onnx_model_path
        num_of_splits = request.num_of_splits

        #verify the input data path and model path 
        if not os.path.exists(self.input_data_path):
            return jobs_pb2.JobStatus(job_id=job_id, status=JobStatus.FAILED, result="Input data path does not exist")
        if not os.path.exists(onnx_model_path):
            return jobs_pb2.JobStatus(job_id=job_id, status=JobStatus.FAILED, result="ONNX model path does not exist")
        #check if the model and input data are valid
        try:
            input_data = read_json_file_to_dict(self.input_data_path)
            onnx_model = onnx.load(onnx_model_path)
            #run inference on the model to check if it is valid
            result = run_model_inference(onnx_model, input_data)
            
        except json.JSONDecodeError:
            return jobs_pb2.JobStatus(job_id=job_id, status=JobStatus.FAILED, result="Invalid input data file")
        except Exception:
            return jobs_pb2.JobStatus(job_id=job_id, status=JobStatus.FAILED, result="Invalid ONNX model file")


    





        globaljob = GlobalProvingJob(
            job_id, model_name, self.input_data_path, onnx_model_path, num_of_splits, self.data_directory)


        self.main_job_queue.put((job_id, globaljob))  # Add job to main queue
        return jobs_pb2.JobStatus(job_id=job_id, status="PENDING", result="")

    def process_main_jobs(self):
        """Processes one main job at a time, splitting it into sub-jobs."""
        while True:
            job_id, data = self.main_job_queue.get()  # Get next main job
            self.job_status[job_id] = ("RUNNING", "")
            sub_jobs = self.split_into_sub_jobs(job_id, data)
            for sub_job in sub_jobs:
                self.sub_job_queue.put((job_id, *sub_job))  # Add sub-jobs to queue

            # Wait for all sub-jobs to finish
            while not self.all_sub_jobs_done(job_id, len(sub_jobs)):
                pass  # Busy wait (can be improved with event-based handling)

            # Combine sub-job results
            with self.lock:
                final_result = " ".join(self.job_status[job_id][1])  
                self.job_status[job_id] = ("COMPLETED", final_result)

    def split_into_sub_jobs(self, job_id, data):
        """Splits a job into sub-jobs."""
        chunks = data.split()
        return [(i, chunk) for i, chunk in enumerate(chunks)]

    def all_sub_jobs_done(self, job_id, num_sub_jobs):
        """Checks if all sub-jobs are completed."""
        return len(self.job_status.get(job_id, ("", []))[1]) == num_sub_jobs

    def process_sub_jobs(self):
        """Assigns sub-jobs to workers as they become available."""
        worker_stubs = [self.get_worker_stub(w) for w in self.worker_addresses]
        while True:
            job_id, sub_job_id, sub_data = self.sub_job_queue.get()
            for worker_stub in worker_stubs:
                response = worker_stub.ProcessSubJob(jobs_pb2.SubJobRequest(
                    job_id=job_id, sub_job_id=sub_job_id, data=sub_data
                ))
                with self.lock:
                    if job_id in self.job_status:
                        _, results = self.job_status[job_id]
                        results.append(response.result)
                        self.job_status[job_id] = ("RUNNING", results)
                break  # Assign one worker per sub-job

    def get_worker_stub(self, worker_address):
        channel = grpc.insecure_channel(worker_address)
        return jobs_pb2_grpc.WorkerStub(channel)

    def GetJobStatus(self, request, context):
        status, result = self.job_status.get(request.job_id, ("UNKNOWN", ""))
        return jobs_pb2.JobStatus(job_id=request.job_id, status=status, result=result)

def serve():
    server = grpc.server(ThreadPoolExecutor(max_workers=10))
    jobs_pb2_grpc.add_DispatcherServicer_to_server(DispatcherService(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("Dispatcher running on port 50051...")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
