import grpc
import time
import proto.zkpservice_pb2_grpc as pb2_grpc
import proto.zkpservice_pb2 as pb2
import hydra
from omegaconf import DictConfig
import os
import shutil
from concurrent import futures
import queue

class ZKPProvingService(pb2_grpc.DispatcherServicer):
    def __init__(self, config: DictConfig):

        self.worker_addresses = config.worker_addresses
        self.number_workers = len(self.worker_addresses)
        self.working_dir = config.working_dir
        self.model_onnx_file = config.onnx_file
        self.worker_queue = queue.Queue()

        if not os.path.exists(self.model_onnx_file):
            raise FileNotFoundError(f"The specified file '{self.model_onnx_file}' does not exist.")
        
        if os.path.exists(self.working_dir):
            shutil.rmtree(self.working_dir)

        os.makedirs(self.working_dir, exist_ok=True)


    def confirm_connections(self):
        for worker_address in self.worker_addresses:
             # Forward task to the worker
            try:
                with grpc.insecure_channel(worker_address) as channel:
                    stub = pb2_grpc.WorkerStub(channel)
                    response = stub.ProcessTask(request)
                    print("Worker registered:", worker_address)
                    self.worker_queue.put(worker_address)
            except:
                raise ConnectionError(f"Cannot connect to worker at {worker_address}") 
            
    
    def GenerateProof(self, request, context):
        pass
    
    def Dispatch(self, request, context):
        # Here goes the logic to dispatch tasks to workers
        print("Received task:", request)
        try:
            worker_address = self.worker_queue.get(timeout=5)  # Timeout after 5 seconds
        except queue.Empty:
            print("No available workers.")
            return pb2.Task(id="", data="No available workers.")
        
        # Forward task to the worker
        with grpc.insecure_channel(worker_address) as channel:
            stub = pb2_grpc.WorkerStub(channel)
            response = stub.ProcessTask(request)
            print("Received response from worker:", response)
            return response
        
    def RegisterWorker(self, request, context):
        worker_address = request.address
        self.worker_queue.put(worker_address)
        print("Worker registered:", worker_address)
        return pb2.WorkerRegistrationResponse(success=True, message="Worker registered successfully.")

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def serve(config: DictConfig):
    try:
        proving_service = ZKPProvingService(config)
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        pb2_grpc.add_DispatcherServicer_to_server(proving_service, server)
        server.add_insecure_port('[::]:50051')
        server.start()
        print("Dispatcher started...")
        # Keep the server running until interrupted
        server.wait_for_termination()

    except KeyboardInterrupt:
        print("Server stopped due to keyboard interrupt")
        server.stop(0)
    except Exception as e:
        print(f"Error in serve(): {e}")        

if __name__ == '__main__':
    serve()
