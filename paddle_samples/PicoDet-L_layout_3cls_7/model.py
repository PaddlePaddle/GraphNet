import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (15362xf32) <- (15362xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(data_0, full_0, float("0"), True)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_0 = []

        # pd_op.sum: (xf32) <- (15362xf32, 0xi64)
        sum_0 = paddle._C_ops.sum(scale_0, full_int_array_0, None, False)

        # pd_op.divide: (xf32) <- (xf32, xf32)
        divide_0 = paddle._C_ops.divide(sum_0, data_1)
        return full_0, scale_0, full_int_array_0, sum_0, divide_0
