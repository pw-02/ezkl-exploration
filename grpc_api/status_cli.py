import grpc
import argparse
from grpc_api import zkservice_pb2 as pb, zkservice_pb2_grpc as pb_grpc

def print_job_summary(job_id, server="localhost:50051"):
    with grpc.insecure_channel(server) as channel:
        stub = pb_grpc.ZKJobServiceStub(channel)
        response = stub.GetJobSummary(pb.JobIDRequest(job_id=job_id))

    print(f"\n🧾 Job Summary: {response.job_name} (ID: {response.job_id})")
    print(f"Status: {response.status} ({response.completed_sub_jobs}/{response.total_sub_jobs} sub-jobs complete)")

    for sj in response.sub_jobs:
        print(f"  ├─ Sub-job {sj.sub_job_id}: {pb.SubJobStatus.Name(sj.status)}"
              f" | worker={sj.worker_id} | {int(sj.progress * 100)}% | msg='{sj.message}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("job_id", help="The ID of the job to check")
    parser.add_argument("--server", default="localhost:50051", help="gRPC server address")
    args = parser.parse_args()

    print_job_summary(args.job_id, args.server)
