import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([1x8400x2xf32, 1x8400x2xf32]) <- (1x8400x2xf32, 1x8400x2xf32)
        combine_0 = [data_0, data_1]

        # pd_op.concat: (1x8400x4xf32) <- ([1x8400x2xf32, 1x8400x2xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("15.99"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (1x8400x4xf32) <- (1x8400x4xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(concat_0, full_1, full_2)
        return clip_0
