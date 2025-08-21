import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("13"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.topk: (2x12x13xf32, 2x12x13xi64) <- (2x12x5376xf32, 1xi32)
        topk_0, topk_1 = (lambda x, f: f(x))(
            paddle._C_ops.topk(data_0, full_0, -1, True, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("5376"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.one_hot: (2x12x13x5376xf32) <- (2x12x13xi64, 1xi32)
        one_hot_0 = paddle._C_ops.one_hot(
            topk_1 % paddle.cast(full_1, topk_1.dtype), full_1
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [-2]

        # pd_op.sum: (2x12x5376xf32) <- (2x12x13x5376xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(one_hot_0, full_int_array_0, None, False)

        # pd_op.multiply: (2x12x5376xf32) <- (2x12x5376xf32, 2x12x1xf32)
        multiply_0 = paddle._C_ops.multiply(sum_0, data_1)
        return multiply_0
