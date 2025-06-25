# submit_and_run_debug.py

import grpc
import hydra
import time
from omegaconf import DictConfig, OmegaConf
import zkservice_pb2 as pb, zkservice_pb2_grpc as pb_grpc
from worker_s3 import main as run_worker


def live_status_tracker(stub, interval=10):
    """Continuously print job status summary."""
    while True:
        try:
            resp = stub.ListActiveSubJobs(pb.StatusAck(success=True))
            if resp.statuses:
                print("\n🖥️ Live Sub-Job Statuses:")
                for sid, info in resp.statuses.items():
                    print(f"🔹 {sid} | Worker={info.worker_id} | "
                          f"Status={pb.SubJobStatus.Name(info.status)} | "
                          f"Progress={info.progress*100:.1f}% | "
                          f"Message={info.message} | Last seen: {info.last_seen}")
            else:
                print("🟡 No active sub-jobs right now.")
        except grpc.RpcError as e:
            logger.warning(f"⚠️ Failed to fetch live status: {e.details()}")
        time.sleep(interval)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    job_cfg = cfg.model
    host = cfg.dispatcher.host
    port = cfg.dispatcher.port
    target = f"{host}:{port}"
    
 
    print(f"Connecting to dispatcher at {target}")
    print("🔁 Submitting job with config:")
    print("\n" + OmegaConf.to_yaml(job_cfg))

    try:
        with grpc.insecure_channel(target) as channel:
            stub = pb_grpc.ZKJobServiceStub(channel)

            # Submit the job
            response = stub.SubmitJob(pb.SubmitJobRequest(
                job_name=job_cfg.name,
                onnx_model_path=job_cfg.onnx_file,
                input_data_path=job_cfg.input_file,
                split_mode=job_cfg.split_mode,
                ops_per_chunk=job_cfg.ops_per_chunk
            ))

            print(f"✅ Job submitted. Assigned ID: {response.job_id}")

            # Start live tracking in background
            # tracking_thread = threading.Thread(target=live_status_tracker, args=(stub,), daemon=True)
            # tracking_thread.start()

            # Launch a local worker for debugging
            run_worker(cfg)

    except grpc.RpcError as e:
        print(f"❌ gRPC error: {e.details()} (code={e.code()})")


if __name__ == "__main__":
    main()
