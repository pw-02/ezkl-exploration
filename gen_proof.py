import os
import logging
import shutil
import multiprocessing

import hydra
from omegaconf import DictConfig

from zkInfer.zk_job import GlobalProvingJob
from utils.resource_monitor import log_system_usage


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("zk")

    if not os.path.exists(cfg.model.onnx_file):
        raise FileNotFoundError(f"ONNX model not found: {cfg.model.onnx_file}")
    if not os.path.exists(cfg.model.input_file):
        raise FileNotFoundError(f"Input file not found: {cfg.model.input_file}")

    job = GlobalProvingJob(
        job_name=cfg.model.name,
        input_data_path=cfg.model.input_file,
        onnx_model_path=cfg.model.onnx_file,
        split_mode=cfg.model.split_mode,
        ops_per_chunk=cfg.model.ops_per_chunk,
        cache_setup_files=cfg.get("cache_setup_files", True),
    )

    # Start system monitor in background process
    log_file = os.path.join(job.report_directory, "system_usage.log")
    logging_process = multiprocessing.Process(target=log_system_usage, args=(log_file,))
    logging_process.start()

    try:
        job.prepare_for_processing(save_ezkl_settings=True)
        job.gen_proof_for_sub_models()

        if not job.cache_setup_files:
            logger.info(f"Removing cache directory: {job.cache_directory}")
            shutil.rmtree(job.cache_directory, ignore_errors=True)

    finally:
        if logging_process.is_alive():
            logging_process.terminate()
            logging_process.join()
        logger.info("All done. Shutting down.")


if __name__ == "__main__":
    main()
