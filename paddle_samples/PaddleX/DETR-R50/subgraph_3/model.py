import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        parameter_0,
        parameter_1,
        parameter_2,
        parameter_3,
        parameter_4,
        parameter_5,
        parameter_6,
        parameter_7,
        data_0,
    ):
        # pd_op.matmul: (6x4x100x5xf32) <- (6x4x100x256xf32, 256x5xf32)
        matmul_0 = paddle._C_ops.matmul(data_0, parameter_7, False, False)
        del parameter_7

        # pd_op.add: (6x4x100x5xf32) <- (6x4x100x5xf32, 5xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_6)
        del parameter_6

        # pd_op.matmul: (6x4x100x256xf32) <- (6x4x100x256xf32, 256x256xf32)
        matmul_1 = paddle._C_ops.matmul(data_0, parameter_5, False, False)
        del data_0, parameter_5

        # pd_op.add: (6x4x100x256xf32) <- (6x4x100x256xf32, 256xf32)
        add_1 = paddle._C_ops.add(matmul_1, parameter_4)
        del parameter_4

        # pd_op.relu: (6x4x100x256xf32) <- (6x4x100x256xf32)
        relu_0 = paddle._C_ops.relu(add_1)
        del add_1

        # pd_op.matmul: (6x4x100x256xf32) <- (6x4x100x256xf32, 256x256xf32)
        matmul_2 = paddle._C_ops.matmul(relu_0, parameter_3, False, False)
        del parameter_3

        # pd_op.add: (6x4x100x256xf32) <- (6x4x100x256xf32, 256xf32)
        add_2 = paddle._C_ops.add(matmul_2, parameter_2)
        del parameter_2

        # pd_op.relu: (6x4x100x256xf32) <- (6x4x100x256xf32)
        relu_1 = paddle._C_ops.relu(add_2)
        del add_2

        # pd_op.matmul: (6x4x100x4xf32) <- (6x4x100x256xf32, 256x4xf32)
        matmul_3 = paddle._C_ops.matmul(relu_1, parameter_1, False, False)
        del parameter_1

        # pd_op.add: (6x4x100x4xf32) <- (6x4x100x4xf32, 4xf32)
        add_3 = paddle._C_ops.add(matmul_3, parameter_0)
        del parameter_0

        # pd_op.sigmoid: (6x4x100x4xf32) <- (6x4x100x4xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(add_3)
        del add_3, matmul_0, matmul_1, matmul_2, matmul_3, relu_0, relu_1

        return sigmoid_0, add_0
