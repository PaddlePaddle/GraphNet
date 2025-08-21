import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0):
        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_0 = [24, 96, -1]

        # pd_op.reshape: (24x96x256xf32) <- (24x96x16x16xf32, 3xi64)
        reshape_0 = paddle._C_ops.reshape(data_0, full_int_array_0)

        # pd_op.transpose: (24x256x96xf32) <- (24x96x256xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_0, [0, 2, 1])

        # pd_op.full: (24x256xf32) <- ()
        full_2 = paddle._C_ops.full(
            [24, 256],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (24x501x50xf32) <- ()
        full_0 = paddle._C_ops.full(
            [24, 501, 50],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (24x501x8xf32) <- ()
        full_1 = paddle._C_ops.full(
            [24, 501, 8],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )
        return full_0, full_1, transpose_0, full_2
