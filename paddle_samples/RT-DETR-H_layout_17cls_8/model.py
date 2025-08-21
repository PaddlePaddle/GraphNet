import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, parameter_0, parameter_1, parameter_2, parameter_3, data_0):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [2]

        # pd_op.unsqueeze: (1x-1x1x4xf32) <- (1x-1x4xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(data_0, full_int_array_0)

        # pd_op.matmul: (1x-1x512xf32) <- (1x-1x4xf32, 4x512xf32)
        matmul_0 = paddle._C_ops.matmul(data_0, parameter_3, False, False)

        # pd_op.add: (1x-1x512xf32) <- (1x-1x512xf32, 512xf32)
        add_1 = paddle._C_ops.add(matmul_0, parameter_2)

        # pd_op.relu: (1x-1x512xf32) <- (1x-1x512xf32)
        relu_0 = paddle._C_ops.relu(add_1)

        # pd_op.matmul: (1x-1x256xf32) <- (1x-1x512xf32, 512x256xf32)
        matmul_1 = paddle._C_ops.matmul(relu_0, parameter_1, False, False)

        # pd_op.add: (1x-1x256xf32) <- (1x-1x256xf32, 256xf32)
        add_0 = paddle._C_ops.add(matmul_1, parameter_0)
        return matmul_0, relu_0, matmul_1, unsqueeze_0, add_0
