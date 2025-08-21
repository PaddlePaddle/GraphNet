import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0):
        # pd_op.shape64: (3xi64) <- (1x-1x-1xf32)
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

        # pd_op.full: (1xi64) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("-2"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.argmax: (1x-1xi64) <- (1x-1x-1xf32, 1xi64)
        argmax_0 = paddle._C_ops.argmax(data_0, full_0, False, False, paddle.int64)

        # pd_op.one_hot: (1x-1x-1xf32) <- (1x-1xi64, xi64)
        one_hot_0 = paddle._C_ops.one_hot(
            argmax_0 % paddle.cast(slice_0, argmax_0.dtype), slice_0
        )

        # pd_op.transpose: (1x-1x-1xf32) <- (1x-1x-1xf32)
        transpose_0 = paddle._C_ops.transpose(one_hot_0, [0, 2, 1])
        return transpose_0
