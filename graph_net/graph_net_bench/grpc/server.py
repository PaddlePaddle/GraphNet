import grpc
from concurrent import futures
import message_pb2
import message_pb2_grpc


class SampleRemoteExecutor(message_pb2_grpc.SampleRemoteExecutorServicer):
    def Execute(self, request, context):
        print("[GraphNet] Received ExecuteRequest")
        return message_pb2.ExecutionReply(
            ret_code=0, stdout="", stderr="", rpc_output=request.rpc_input
        )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    message_pb2_grpc.add_SampleRemoteExecutorServicer_to_server(
        SampleRemoteExecutor(), server
    )

    # Listen on all interfaces (0.0.0.0) at port 50052
    server.add_insecure_port("0.0.0.0:50052")
    print("Server started on port 50052...")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
