#!/usr/bin/env python
import argparse

import grpc
from tritonclient.grpc import service_pb2, service_pb2_grpc

FLAGS = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8001",
        help="Inference server URL. Default is localhost:8001.",
    )

    FLAGS = parser.parse_args()

    model_name = "ebr"
    model_version = "1"
    batch_size = 1

    # Create gRPC stub for communicating with the server
    channel = grpc.insecure_channel(FLAGS.url)
    grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)

    # Health
    try:
        request = service_pb2.ServerLiveRequest()
        response = grpc_stub.ServerLive(request)
        print("server {}".format(response))
    except Exception as ex:
        print(ex)

    request = service_pb2.ServerReadyRequest()
    response = grpc_stub.ServerReady(request)
    print("server {}".format(response))

    request = service_pb2.ModelReadyRequest(name=model_name, version=model_version)
    response = grpc_stub.ModelReady(request)
    print("model {}".format(response))

    # Metadata
    request = service_pb2.ServerMetadataRequest()
    response = grpc_stub.ServerMetadata(request)
    print("server metadata:\n{}".format(response))

    request = service_pb2.ModelMetadataRequest(name=model_name, version=model_version)
    response = grpc_stub.ModelMetadata(request)
    print("model metadata:\n{}".format(response))

    # Configuration
    request = service_pb2.ModelConfigRequest(name=model_name, version=model_version)
    response = grpc_stub.ModelConfig(request)
    print("model config:\n{}".format(response))

    # Infer
    request = service_pb2.ModelInferRequest()
    request.model_name = model_name
    request.model_version = model_version
    request.id = "my request id"

    input = service_pb2.ModelInferRequest().InferInputTensor()
    input.name = "input"
    input.datatype = "FP32"
    input.shape.extend([1, 299, 299, 3])
    request.inputs.extend([input])

    output = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
    output.name = "InceptionV3/Predictions/Softmax"
    request.outputs.extend([output])

    request.raw_input_contents.extend([bytes(1072812 * "a", "utf-8")])

    response = grpc_stub.ModelInfer(request)
    print("model infer:\n{}".format(response))

    print("PASS")