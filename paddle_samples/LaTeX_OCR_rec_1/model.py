import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, parameter_0, parameter_1, parameter_2, parameter_3, data_0):
        # pd_op.matmul: (20x44x2048xf32) <- (20x44x256xf32, 256x2048xf32)
        matmul_0 = paddle._C_ops.matmul(data_0, parameter_3, False, False)

        # pd_op.add: (20x44x2048xf32) <- (20x44x2048xf32, 2048xf32)
        add_1 = paddle._C_ops.add(matmul_0, parameter_2)

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("2"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split_with_num: ([20x44x1024xf32, 20x44x1024xf32]) <- (20x44x2048xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(add_1, 2, full_0)

        # builtin.split: (20x44x1024xf32, 20x44x1024xf32) <- ([20x44x1024xf32, 20x44x1024xf32])
        (
            split_0,
            split_1,
        ) = split_with_num_0

        # pd_op.gelu: (20x44x1024xf32) <- (20x44x1024xf32)
        gelu_0 = paddle._C_ops.gelu(split_1, False)

        # pd_op.multiply: (20x44x1024xf32) <- (20x44x1024xf32, 20x44x1024xf32)
        multiply_0 = paddle._C_ops.multiply(split_0, gelu_0)

        # pd_op.matmul: (20x44x256xf32) <- (20x44x1024xf32, 1024x256xf32)
        matmul_1 = paddle._C_ops.matmul(multiply_0, parameter_1, False, False)

        # pd_op.add: (20x44x256xf32) <- (20x44x256xf32, 256xf32)
        add_0 = paddle._C_ops.add(matmul_1, parameter_0)
        return matmul_0, full_0, split_0, split_1, gelu_0, multiply_0, matmul_1, add_0
