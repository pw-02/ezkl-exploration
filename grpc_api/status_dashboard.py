import grpc
import time
import os
from grpc_api import zkservice_pb2 as pb, zkservice_pb2_grpc as pb_grpc

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_summary(summary: pb.JobSummaryResponse):
    print(f"🧾 Job ID: {summary.job_id}")
    print(f"📝 Job Name: {summary.job_name}")
    print(f"📈 Status: {summary.status}")
    print(f"🧩 Sub-jobs: {summary.completed_sub_jobs}/{summary.total_sub_jobs} complete")
    print()

    # for s in summary.sub_jobs:
    #     print(f"  🔹 {s.sub_job_id}")
    #     print(f"     Worker:  {s.worker_id}")
    #     print(f"     Status:  {pb.SubJobStatus.Name(s.status)}")
    #     print(f"     Progress: {s.progress:.1f}%")
    #     print(f"     Last Seen: {s.last_seen}")
    #     print(f"     Message: {s.message}")
    #     print()

def monitor_job(job_id, target="localhost:50051", interval=10):
    with grpc.insecure_channel(target) as channel:
        stub = pb_grpc.ZKJobServiceStub(channel)

        try:
            while True:
                response = stub.GetJobSummary(pb.JobIDRequest(job_id=job_id))
                clear_screen()
                print_summary(response)

                if response.status == "COMPLETED":
                    print("🎉 Job is complete!")
                    break

                time.sleep(interval)

        except grpc.RpcError as e:
            print(f"❌ gRPC error: {e.details()} (code={e.code()})")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python live_status_cli.py <job_id>")
        exit(1)

    monitor_job(sys.argv[1])
