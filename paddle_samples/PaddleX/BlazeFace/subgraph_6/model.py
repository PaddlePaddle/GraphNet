import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(data_0, full_0, float("1"), True)
        del full_0

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [data_0]
        del data_0

        # pd_op.stack: (1xi64) <- ([xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]
        del scale_0

        # pd_op.stack: (1xi64) <- ([xi64])
        stack_1 = paddle._C_ops.stack(combine_1, 0)
        del combine_1

        # pd_op.slice: (1xf32) <- (4x1xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(data_1, [0], stack_0, stack_1, [-1], [0])
        del data_1, stack_0, stack_1

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("3"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1xf32) <- (1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_0, full_1, float("0"), True)
        del full_1

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("-3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("22400"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (1xf32) <- (1xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(scale_1, full_2, full_3)
        del full_2, full_3, scale_1

        # pd_op.full: (xf32) <- ()
        full_4 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (1xb) <- (1xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(clip_0, full_4)
        del clip_0, full_4, slice_0

        return greater_than_0
