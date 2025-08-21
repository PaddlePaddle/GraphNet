import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.shape64: (3xi64) <- (2x-1x-1xf32)
        shape64_0 = paddle._C_ops.shape64(data_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [2]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [3]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("13"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.topk: (2x-1x13xf32, 2x-1x13xi64) <- (2x-1x-1xf32, 1xi32)
        topk_0, topk_1 = (lambda x, f: f(x))(
            paddle._C_ops.topk(data_0, full_0, -1, True, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.one_hot: (2x-1x13x-1xf32) <- (2x-1x13xi64, xi64)
        one_hot_0 = paddle._C_ops.one_hot(
            topk_1 % paddle.cast(slice_1, topk_1.dtype), slice_1
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [-2]

        # pd_op.sum: (2x-1x-1xf32) <- (2x-1x13x-1xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(one_hot_0, full_int_array_3, None, False)

        # pd_op.multiply: (2x-1x-1xf32) <- (2x-1x-1xf32, 2x-1x1xf32)
        multiply_0 = paddle._C_ops.multiply(sum_0, data_1)
        return multiply_0
