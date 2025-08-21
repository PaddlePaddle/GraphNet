import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0):
        # pd_op.shape64: (3xi64) <- (-1x-1x384xf32)
        shape64_0 = paddle._C_ops.shape64(data_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.transpose: (-1x384x-1xf32) <- (-1x-1x384xf32)
        transpose_0 = paddle._C_ops.transpose(data_0, [0, 2, 1])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [0, 384, 3, 80]

        # pd_op.reshape: (-1x384x3x80xf32) <- (-1x384x-1xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(transpose_0, full_int_array_3)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [3, 2]

        # pd_op.pool2d: (-1x384x1x40xf32) <- (-1x384x3x80xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            reshape_0,
            full_int_array_4,
            [3, 2],
            [0, 0],
            False,
            True,
            "NCHW",
            "avg",
            False,
            False,
            "EXPLICIT",
        )
        return pool2d_0
