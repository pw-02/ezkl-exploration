import grpc
import proto.zkpservice_pb2 as pb2
import proto.zkpservice_pb2_grpc as pb2_grpc

def dispatch_task(dispatcher_address, task):
    with grpc.insecure_channel(dispatcher_address) as channel:
        stub = pb2_grpc.DispatcherStub(channel)
        response = stub.Dispatch(task)
        return response

if __name__ == "__main__":
    # Define the dispatcher address
    dispatcher_address = "localhost:50051"
    
    # Create a task
    task = pb2.Task(id="1", data="examples/onnx/mobilenet/mobilenetv2_050_Opset18.onnx")

    # Dispatch the task to the dispatcher
    response = dispatch_task(dispatcher_address, task)

    # Print the response
    print("Response from dispatcher:", response)
