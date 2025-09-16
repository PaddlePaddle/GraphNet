import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.full: (16x8x144x144xf32) <- ()
        full_0 = paddle._C_ops.full(
            [16, 8, 144, 144],
            float("-inf"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (16x8x144x144xf32) <- (16x8x144x144xf32, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            full_0, full_1, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.full_like: (16x8x144x144xf32) <- (16x8x144x144xf32, 1xf32)
        full_like_1 = paddle._C_ops.full_like(
            data_0, full_1, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.full_like: (16x1x144x144xb) <- (16x1x144x144xb, 1xf32)
        full_like_2 = paddle._C_ops.full_like(
            data_1, full_1, paddle.bool, paddle.framework._current_expected_place()
        )
        del full_1

        # pd_op.cast: (16x1x144x144xf32) <- (16x1x144x144xb)
        cast_0 = paddle._C_ops.cast(full_like_2, paddle.float32)
        del full_like_2

        # pd_op.cast: (16x1x144x144xf32) <- (16x1x144x144xb)
        cast_1 = paddle._C_ops.cast(data_1, paddle.float32)
        del data_1

        # pd_op.add: (16x8x144x144xf32) <- (16x8x144x144xf32, 16x8x144x144xf32)
        add_0 = paddle._C_ops.add(full_like_0, full_like_1)
        del full_like_0, full_like_1

        # pd_op.add: (16x8x144x144xf32) <- (16x8x144x144xf32, 16x1x144x144xf32)
        add_1 = paddle._C_ops.add(add_0, cast_0)
        del add_0, cast_0

        # pd_op.add: (16x8x144x144xf32) <- (16x8x144x144xf32, 16x8x144x144xf32)
        add_2 = paddle._C_ops.add(full_0, add_1)

        # pd_op.add: (16x8x144x144xf32) <- (16x8x144x144xf32, 16x8x144x144xf32)
        add_3 = paddle._C_ops.add(data_0, add_1)
        del data_0

        # pd_op.add: (16x8x144x144xf32) <- (16x1x144x144xf32, 16x8x144x144xf32)
        add_4 = paddle._C_ops.add(cast_1, add_1)
        del cast_1

        # pd_op.cast: (16x8x144x144xb) <- (16x8x144x144xf32)
        cast_2 = paddle._C_ops.cast(add_4, paddle.bool)
        del add_4

        # pd_op.where: (16x8x144x144xf32) <- (16x8x144x144xb, 16x8x144x144xf32, 16x8x144x144xf32)
        where_0 = paddle._C_ops.where(cast_2, add_2, add_3)
        del add_1, add_2, add_3, cast_2, full_0

        return where_0
