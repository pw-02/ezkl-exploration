import logging

import grpc

import zkinfer.proto.zkservice_pb2 as pb
import zkinfer.proto.zkservice_pb2_grpc as pb_grpc


class ZKInferenceClient:
    def __init__(
        self,
        target: str,
        grpc_max_message_mb: int = 64,
        logger: logging.Logger | None = None,
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
        schedule: str,
    ) -> str:
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
                    schedule=schedule,
                )
            )

        return response.request_id