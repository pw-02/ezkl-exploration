# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: zkpservice.proto
# Protobuf Python Version: 5.26.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x10zkpservice.proto\"O\n\x13ProofStatusResponse\x12\x16\n\x0estatus_message\x18\x01 \x01(\t\x12\x11\n\tstatus_id\x18\x02 \x01(\x05\x12\r\n\x05proof\x18\x03 \x01(\x0c\"7\n\x14WorkerStatusResponse\x12\x0f\n\x07message\x18\x01 \x01(\t\x12\x0e\n\x06isbusy\x18\x02 \x01(\x08\"5\n\tProofData\x12\x13\n\x0bmodel_bytes\x18\x01 \x01(\x0c\x12\x13\n\x0bmodel_input\x18\x02 \x01(\t\"\x1a\n\x07Message\x12\x0f\n\x07message\x18\x01 \x01(\t\"4\n\x0fMessageResponse\x12\x0f\n\x07message\x18\x01 \x01(\t\x12\x10\n\x08received\x18\x02 \x01(\x08\" \n\x04Task\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0c\n\x04\x64\x61ta\x18\x02 \x01(\t2\xdf\x01\n\x06Worker\x12\x1d\n\x0bProcessTask\x12\x05.Task\x1a\x05.Task\"\x00\x12\"\n\x04Ping\x12\x08.Message\x1a\x10.MessageResponse\x12&\n\x0c\x43omputeProof\x12\n.ProofData\x1a\x08.Message\"\x00\x12\x34\n\x0fGetWorkerStatus\x12\x08.Message\x1a\x15.WorkerStatusResponse\"\x00\x12\x34\n\x10GetComputedProof\x12\x08.Message\x1a\x14.ProofStatusResponse\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'zkpservice_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_PROOFSTATUSRESPONSE']._serialized_start=20
  _globals['_PROOFSTATUSRESPONSE']._serialized_end=99
  _globals['_WORKERSTATUSRESPONSE']._serialized_start=101
  _globals['_WORKERSTATUSRESPONSE']._serialized_end=156
  _globals['_PROOFDATA']._serialized_start=158
  _globals['_PROOFDATA']._serialized_end=211
  _globals['_MESSAGE']._serialized_start=213
  _globals['_MESSAGE']._serialized_end=239
  _globals['_MESSAGERESPONSE']._serialized_start=241
  _globals['_MESSAGERESPONSE']._serialized_end=293
  _globals['_TASK']._serialized_start=295
  _globals['_TASK']._serialized_end=327
  _globals['_WORKER']._serialized_start=330
  _globals['_WORKER']._serialized_end=553
# @@protoc_insertion_point(module_scope)
