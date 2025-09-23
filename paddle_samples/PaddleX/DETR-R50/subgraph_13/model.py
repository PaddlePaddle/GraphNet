import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.full: (1xi64) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (1xi64) <- (1xi64)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_0,
            [1],
            paddle.int64,
            [float("4")],
            paddle.framework._current_expected_place(),
        )
        del full_0

        # pd_op.cast: (1xf32) <- (1xi64)
        cast_0 = paddle._C_ops.cast(assign_value__0, paddle.float32)
        del assign_value__0

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (1xf32) <- (1xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(cast_0, full_1, full_2)
        del cast_0, full_1, full_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [-1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [2147483647]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_1

        # pd_op.slice: (4x100x4xf32) <- (6x4x100x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_0, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del data_0

        # pd_op.slice: (4x100x5xf32) <- (6x4x100x5xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_1, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del assign_0, assign_1, data_1, full_int_array_0, full_int_array_1

        return slice_0, slice_1, clip_0
