import grpc
import grpc
import proto.zkpservice_pb2_grpc as pb2_grpc
import proto.zkpservice_pb2 as pb2
import time
import threading
from concurrent import futures
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

class WorkerServicer(pb2_grpc.WorkerServicer):

    def ProcessTask(self, request, context):
        logging.info("Received task: %s", request)
        # Process the task
        # Simulating processing time
        time.sleep(2)
        # Return response
        return pb2.Task(id=request.id, data="Processed " + request.data)
    
    def Ping(self, request, context): 
        logging.info("Received Ping Request from %s", request.message)
        # Return response
        result = {'message': 'pong', 'received': True}
        return pb2.MessageResponse(**result)
    

def run_worker():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    pb2_grpc.add_WorkerServicer_to_server(WorkerServicer(), server)
    server.add_insecure_port('[::]:50052')
    server.start()
    logging.info("Worker started...")
    # # Register with dispatcher
    # channel = grpc.insecure_channel("localhost:50051")
    # stub = pb2_grpc.DispatcherStub(channel)
    # response = stub.RegisterWorker(pb2.WorkerAddress(address="localhost:50052"))  # Change the address as needed
    # print("Registration response:", response)
    server.wait_for_termination()


if __name__ == '__main__':
    run_worker()
