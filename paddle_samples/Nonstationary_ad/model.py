import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self, parameter_0, parameter_1, parameter_2, parameter_3, parameter_4, data_0
    ):
        # pd_op.matmul: (-1x32xf32) <- (-1x4xf32, 4x32xf32)
        matmul_1 = paddle._C_ops.matmul(data_0, parameter_4, False, False)

        # pd_op.add: (-1x32xf32) <- (-1x32xf32, 32xf32)
        add_0 = paddle._C_ops.add(matmul_1, parameter_3)

        # pd_op.relu: (-1x32xf32) <- (-1x32xf32)
        relu_0 = paddle._C_ops.relu(add_0)

        # pd_op.matmul: (-1x32xf32) <- (-1x32xf32, 32x32xf32)
        matmul_2 = paddle._C_ops.matmul(relu_0, parameter_2, False, False)

        # pd_op.add: (-1x32xf32) <- (-1x32xf32, 32xf32)
        add_1 = paddle._C_ops.add(matmul_2, parameter_1)

        # pd_op.relu: (-1x32xf32) <- (-1x32xf32)
        relu_1 = paddle._C_ops.relu(add_1)

        # pd_op.matmul: (-1x96xf32) <- (-1x32xf32, 32x96xf32)
        matmul_0 = paddle._C_ops.matmul(relu_1, parameter_0, False, False)
        return matmul_0
