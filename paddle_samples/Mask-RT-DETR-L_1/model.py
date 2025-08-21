import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2):
        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 2x400x256xf32)
        add_0 = paddle._C_ops.add(data_0, data_2)

        # pd_op.full: (400x400xf32) <- ()
        full_0 = paddle._C_ops.full(
            [400, 400],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (400x400xf32) <- ()
        full_1 = paddle._C_ops.full(
            [400, 400],
            float("-inf"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (400x400xf32) <- (400x400xb, 400x400xf32, 400x400xf32)
        where_0 = paddle._C_ops.where(data_1, full_0, full_1)
        return add_0, where_0
