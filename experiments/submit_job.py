import logging
import sys

import grpc
import hydra
from omegaconf import DictConfig, OmegaConf

from zkinfer.client.api import ZKInferenceClient


PARTITION_MODE_TO_RUNTIME = {
    "whole_model": "none",
    "single_operator": "single_ops",
    "fixed_size": "fixed",
}


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] submit: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("submit")


def runtime_partition_mode(cfg: DictConfig) -> str:
    mode = str(cfg.execution.partitioning.mode)
    try:
        return PARTITION_MODE_TO_RUNTIME[mode]
    except KeyError as exc:
        choices = ", ".join(PARTITION_MODE_TO_RUNTIME)
        raise ValueError(
            f"Unsupported partitioning.mode={mode!r}; choose one of: {choices}"
        ) from exc


@hydra.main(config_path="./config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logger = setup_logger()
    target = f"{cfg.launch.coordinator_host}:{cfg.launch.coordinator_port}"

    logger.info("Connecting to coordinator at %s", target)
    logger.info("Submitting workload:\n%s", OmegaConf.to_yaml(cfg.workload))
    logger.info("Execution config:\n%s", OmegaConf.to_yaml(cfg.execution))
    logger.info("Coordinator scheduler: %s", cfg.jobs.scheduler)

    client = ZKInferenceClient(target=target, logger=logger)

    try:
        request_id = client.submit_inference_request(
            name=cfg.workload.name,
            onnx_model_path=cfg.workload.onnx_file,
            input_data_path=cfg.workload.input_file,
            split_mode=runtime_partition_mode(cfg),
            ops_per_chunk=int(cfg.execution.partitioning.operators_per_partition),
            simplify_model=cfg.execution.get("simplify_model", False),
            input_shapes=cfg.workload.get("input_shapes", None),
        )
        logger.info("Job submitted. Request ID: %s", request_id)
    except grpc.RpcError as exc:
        logger.error("gRPC error: %s code=%s", exc.details(), exc.code())
        raise


if __name__ == "__main__":
    main()
