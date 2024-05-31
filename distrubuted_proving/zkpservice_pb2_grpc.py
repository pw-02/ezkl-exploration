# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

import zkpservice_pb2 as zkpservice__pb2

GRPC_GENERATED_VERSION = '1.64.0'
GRPC_VERSION = grpc.__version__
EXPECTED_ERROR_RELEASE = '1.65.0'
SCHEDULED_RELEASE_DATE = 'June 25, 2024'
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    warnings.warn(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + f' but the generated code in zkpservice_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
        + f' This warning will become an error in {EXPECTED_ERROR_RELEASE},'
        + f' scheduled for release on {SCHEDULED_RELEASE_DATE}.',
        RuntimeWarning
    )


class WorkerStub(object):
    """
    service Dispatcher {
    rpc Dispatch(Task) returns (Task) {}
    rpc RegisterWorker(WorkerAddress) returns (WorkerRegistrationResponse) {}
    }
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ProcessTask = channel.unary_unary(
                '/Worker/ProcessTask',
                request_serializer=zkpservice__pb2.Task.SerializeToString,
                response_deserializer=zkpservice__pb2.Task.FromString,
                _registered_method=True)
        self.Ping = channel.unary_unary(
                '/Worker/Ping',
                request_serializer=zkpservice__pb2.Message.SerializeToString,
                response_deserializer=zkpservice__pb2.MessageResponse.FromString,
                _registered_method=True)
        self.ComputeProof = channel.unary_unary(
                '/Worker/ComputeProof',
                request_serializer=zkpservice__pb2.ProofInfo.SerializeToString,
                response_deserializer=zkpservice__pb2.Message.FromString,
                _registered_method=True)


class WorkerServicer(object):
    """
    service Dispatcher {
    rpc Dispatch(Task) returns (Task) {}
    rpc RegisterWorker(WorkerAddress) returns (WorkerRegistrationResponse) {}
    }
    """

    def ProcessTask(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Ping(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ComputeProof(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_WorkerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ProcessTask': grpc.unary_unary_rpc_method_handler(
                    servicer.ProcessTask,
                    request_deserializer=zkpservice__pb2.Task.FromString,
                    response_serializer=zkpservice__pb2.Task.SerializeToString,
            ),
            'Ping': grpc.unary_unary_rpc_method_handler(
                    servicer.Ping,
                    request_deserializer=zkpservice__pb2.Message.FromString,
                    response_serializer=zkpservice__pb2.MessageResponse.SerializeToString,
            ),
            'ComputeProof': grpc.unary_unary_rpc_method_handler(
                    servicer.ComputeProof,
                    request_deserializer=zkpservice__pb2.ProofInfo.FromString,
                    response_serializer=zkpservice__pb2.Message.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'Worker', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('Worker', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class Worker(object):
    """
    service Dispatcher {
    rpc Dispatch(Task) returns (Task) {}
    rpc RegisterWorker(WorkerAddress) returns (WorkerRegistrationResponse) {}
    }
    """

    @staticmethod
    def ProcessTask(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/Worker/ProcessTask',
            zkpservice__pb2.Task.SerializeToString,
            zkpservice__pb2.Task.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def Ping(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/Worker/Ping',
            zkpservice__pb2.Message.SerializeToString,
            zkpservice__pb2.MessageResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def ComputeProof(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/Worker/ComputeProof',
            zkpservice__pb2.ProofInfo.SerializeToString,
            zkpservice__pb2.Message.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)