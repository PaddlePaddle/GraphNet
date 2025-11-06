import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [-1]

        # pd_op.unsqueeze: (4x22400x1xb) <- (4x22400xb, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(data_0, full_int_array_0)
        del data_0, full_int_array_0

        # pd_op.masked_select: (-1xf32) <- (4x22400x1xf32, 4x22400x1xb)
        masked_select_0 = paddle._C_ops.masked_select(data_1, unsqueeze_0)
        del data_1

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_1 = []

        # pd_op.sum: (xf32) <- (-1xf32, 0xi64)
        sum_0 = paddle._C_ops.sum(masked_select_0, full_int_array_1, None, False)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(sum_0, full_0, float("0"), True)
        del sum_0

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.not_equal: (4x22400x1xb) <- (4x22400x1xi64, xi64)
        not_equal_0 = paddle._C_ops.not_equal(data_2, full_1)
        del data_2, full_1

        # pd_op.cast: (4x22400x1xf32) <- (4x22400x1xb)
        cast_0 = paddle._C_ops.cast(not_equal_0, paddle.float32)
        del not_equal_0

        # pd_op.sum: (xf32) <- (4x22400x1xf32, 0xi64)
        sum_1 = paddle._C_ops.sum(cast_0, full_int_array_1, None, False)
        del cast_0

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (xf32) <- (xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(sum_1, full_0, full_2)
        del full_2, sum_1

        # pd_op.add: (xf32) <- (xf32, xf32)
        add_0 = paddle._C_ops.add(scale_0, data_3)
        del data_3

        # pd_op.divide: (xf32) <- (xf32, xf32)
        divide_0 = paddle._C_ops.divide(add_0, clip_0)
        del (
            add_0,
            clip_0,
            full_0,
            full_int_array_1,
            masked_select_0,
            scale_0,
            unsqueeze_0,
        )

        return divide_0
