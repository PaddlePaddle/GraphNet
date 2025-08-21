import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.mean: (13xf32) <- (13x160x160xf32)
        mean_0 = paddle._C_ops.mean(data_0, [1, 2], False)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [-1]

        # pd_op.squeeze: (13xf32) <- (13x1xf32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(data_1, full_int_array_1)

        # pd_op.divide: (13xf32) <- (13xf32, 13xf32)
        divide_0 = paddle._C_ops.divide(mean_0, squeeze_0)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_0 = []

        # pd_op.sum: (xf32) <- (13xf32, 0xi64)
        sum_0 = paddle._C_ops.sum(divide_0, full_int_array_0, None, False)
        return mean_0, squeeze_0, divide_0, full_int_array_0, sum_0
