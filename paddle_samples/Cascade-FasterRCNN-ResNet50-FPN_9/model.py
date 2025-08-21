import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, parameter_0, parameter_1, parameter_2, parameter_3, data_0):
        # pd_op.flatten: (2048x12544xf32) <- (2048x256x7x7xf32)
        flatten_0 = paddle._C_ops.flatten(data_0, 1, 3)

        # pd_op.matmul: (2048x1024xf32) <- (2048x12544xf32, 12544x1024xf32)
        matmul_0 = paddle._C_ops.matmul(flatten_0, parameter_3, False, False)

        # pd_op.add: (2048x1024xf32) <- (2048x1024xf32, 1024xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_2)

        # pd_op.relu: (2048x1024xf32) <- (2048x1024xf32)
        relu_0 = paddle._C_ops.relu(add_0)

        # pd_op.matmul: (2048x1024xf32) <- (2048x1024xf32, 1024x1024xf32)
        matmul_1 = paddle._C_ops.matmul(relu_0, parameter_1, False, False)

        # pd_op.add: (2048x1024xf32) <- (2048x1024xf32, 1024xf32)
        add_1 = paddle._C_ops.add(matmul_1, parameter_0)

        # pd_op.relu: (2048x1024xf32) <- (2048x1024xf32)
        relu_1 = paddle._C_ops.relu(add_1)
        return flatten_0, matmul_0, relu_0, matmul_1, relu_1
