import grpc
import time
import generated.jobs_pb2 as jobs_pb2
import generated.jobs_pb2_grpc as jobs_pb2_grpc

DISPATCHER_ADDRESS = "localhost:50051"

def submit_job(data):
    """Sends a job to the dispatcher."""
    channel = grpc.insecure_channel(DISPATCHER_ADDRESS)
    stub = jobs_pb2_grpc.DispatcherStub(channel)
    job_id = str(int(time.time()))  # Unique job ID based on timestamp
    response = stub.SubmitJob(jobs_pb2.JobRequest(job_id=job_id, data=data))
    print(f"Job submitted: {response.job_id} - Status: {response.status}")
    return job_id

def get_job_status(job_id):
    """Fetches job status from the dispatcher."""
    channel = grpc.insecure_channel(DISPATCHER_ADDRESS)
    stub = jobs_pb2_grpc.DispatcherStub(channel)
    response = stub.GetJobStatus(jobs_pb2.JobStatusRequest(job_id=job_id))
    return response.status, response.result

if __name__ == "__main__":
    job_id = submit_job("hello world distributed processing")
    while True:
        status, result = get_job_status(job_id)
        print(f"Job {job_id} - Status: {status}")
        if status == "COMPLETED":
            print(f"Final Result: {result}")
            break
        time.sleep(2)  # Check every 2 seconds
