import grpc
import message_pb2
import message_pb2_grpc


def run():
    # REPLACE 'SERVER_IP' with the actual IP address of Machine A
    server_ip = "localhost"
    channel = grpc.insecure_channel(f"{server_ip}:50052")
    stub = message_pb2_grpc.SampleRemoteExecutorStub(channel)

    request = message_pb2.ExecutionRequest(
        rpc_cmd="my-echo",
        rpc_input=message_pb2.RpcData(str_data="gooooooooooood"),
    )
    response = stub.Execute(request)
    print(f"{response.rpc_output=}")


if __name__ == "__main__":
    run()
