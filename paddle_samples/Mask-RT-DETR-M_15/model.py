import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0):
        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (1x100x4xf32) <- (1x100x4xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(data_0, full_0, full_1)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1e-05"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (1x100x4xf32) <- (1x100x4xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(clip_0, full_2, full_3)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x100x4xf32) <- (1x100x4xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(clip_0, full_4, float("1"), True)

        # pd_op.clip: (1x100x4xf32) <- (1x100x4xf32, 1xf32, 1xf32)
        clip_2 = paddle._C_ops.clip(scale_0, full_2, full_3)

        # pd_op.divide: (1x100x4xf32) <- (1x100x4xf32, 1x100x4xf32)
        divide_0 = paddle._C_ops.divide(clip_1, clip_2)

        # pd_op.log: (1x100x4xf32) <- (1x100x4xf32)
        log_0 = paddle._C_ops.log(divide_0)
        return log_0
