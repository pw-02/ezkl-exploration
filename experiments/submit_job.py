import logging
import sys

import grpc
import hydra
from omegaconf import DictConfig, OmegaConf

from zkinfer.client.api import ZKInferenceClient


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] submit: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("submit")


@hydra.main(config_path="./config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logger = setup_logger()

    target = f"{cfg.launch.coordinator_host}:{cfg.launch.coordinator_port}"

    logger.info("Connecting to coordinator at %s", target)
    logger.info("Submitting workload:\n%s", OmegaConf.to_yaml(cfg.workload))
    logger.info("Execution config:\n%s", OmegaConf.to_yaml(cfg.execution))
    logger.info("Job config:\n%s", OmegaConf.to_yaml(cfg.jobs))

    client = ZKInferenceClient(
        target=target,
        logger=logger,
    )

    try:
        request_id = client.submit_inference_request(
            name=cfg.workload.name,
            onnx_model_path=cfg.workload.onnx_file,
            input_data_path=cfg.workload.input_file,
            split_mode=cfg.execution.split_mode,
            ops_per_chunk=cfg.execution.ops_per_chunk,
            scheduler=cfg.jobs.scheduler,
            simplify_model=cfg.execution.get("simplify_model", False),
            simplify_input_shapes=cfg.workload.get("input_shapes", None),
        )

        logger.info("Job submitted. Request ID: %s", request_id)

    except grpc.RpcError as exc:
        logger.error(
            "gRPC error: %s code=%s",
            exc.details(),
            exc.code(),
        )
        raise


if __name__ == "__main__":
    main()