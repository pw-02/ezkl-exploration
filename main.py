import hydra
from omegaconf import DictConfig

from grpc_api.dispatcher import serve as serve_dispatcher
from grpc_api.worker_client import run_worker
from grpc_api.submit_job import submit_job


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    mode = cfg.get("run_mode")
    
    if mode == "dispatcher":
        serve_dispatcher(cfg.dispatcher)
    elif mode == "worker":
        run_worker(cfg.worker)
    elif mode == "submit":
        submit_job(cfg.model)
    else:
        raise ValueError(f"Unknown run_mode: {mode}")

if __name__ == "__main__":
    main()
