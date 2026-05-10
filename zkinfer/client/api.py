import json
import logging
from typing import Any, Dict, Optional

import grpc
from omegaconf import DictConfig, ListConfig, OmegaConf

import zkinfer.proto.zkservice_pb2 as pb
import zkinfer.proto.zkservice_pb2_grpc as pb_grpc


def _to_plain_dict(value) -> Optional[Dict[str, Any]]:
    if value is None:
        return None

    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)

    return value


class ZKInferenceClient:
    def __init__(
        self,
        target: str,
        grpc_max_message_mb: int = 64,
        logger: Optional[logging.Logger] = None,
    ):
        self.target = target
        self.grpc_max_message_bytes = grpc_max_message_mb * 1024 * 1024
        self.logger = logger or logging.getLogger(__name__)

    def submit_inference_request(
        self,
        name: str,
        onnx_model_path: str,
        input_data_path: str,
        split_mode: str,
        ops_per_chunk: int,
        scheduler: str,
        simplify_model: bool = False,
        simplify_input_shapes: Optional[Dict[str, Any]] = None,
    ) -> str:
        simplify_input_shapes = _to_plain_dict(simplify_input_shapes)

        simplify_input_shapes_json = (
            json.dumps(simplify_input_shapes)
            if simplify_input_shapes
            else ""
        )

        with grpc.insecure_channel(
            self.target,
            options=[
                ("grpc.max_send_message_length", self.grpc_max_message_bytes),
                ("grpc.max_receive_message_length", self.grpc_max_message_bytes),
            ],
        ) as channel:
            stub = pb_grpc.ZKJobServiceStub(channel)

            response = stub.SubmitInferenceRequest(
                pb.InferenceRequest(
                    name=name,
                    onnx_model_path=onnx_model_path,
                    input_data_path=input_data_path,
                    split_mode=split_mode,
                    ops_per_chunk=ops_per_chunk,
                    scheduler=scheduler,
                    simplify_model=simplify_model,
                    simplify_input_shapes_json=simplify_input_shapes_json,
                )
            )

        return response.request_id