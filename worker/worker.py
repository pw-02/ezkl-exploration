import grpc
from concurrent.futures import ThreadPoolExecutor
import generated.jobs_pb2 as jobs_pb2
import generated.jobs_pb2_grpc as jobs_pb2_grpc

class WorkerService(jobs_pb2_grpc.WorkerServicer):
    def ProcessSubJob(self, request, context):
        """Processes a sub-job and returns the result."""
        job_id = request.job_id
        sub_job_id = request.sub_job_id
        data = request.data
        print(f"Worker processing sub-job {sub_job_id} for job {job_id}: {data}")
        
        # Simulate processing (e.g., reversing text)
        result = data[::-1]  
        return jobs_pb2.SubJobResponse(sub_job_id=sub_job_id, result=result)

def serve(worker_port):
    server = grpc.server(ThreadPoolExecutor(max_workers=5))
    jobs_pb2_grpc.add_WorkerServicer_to_server(WorkerService(), server)
    server.add_insecure_port(f"[::]:{worker_port}")
    server.start()
    print(f"Worker running on port {worker_port}...")
    server.wait_for_termination()

if __name__ == "__main__":
    import sys
    port = sys.argv[1] if len(sys.argv) > 1 else "50052"
    serve(port)
