import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4):
        # pd_op.divide: (1xf32) <- (1xf32, xf32)
        divide_0 = paddle._C_ops.divide(data_0, data_1)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(data_2, full_0, float("0"), True)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("2.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_0 = full_1

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(data_3, full_1, float("0"), True)

        # pd_op.add: (xf32) <- (xf32, xf32)
        add_0 = paddle._C_ops.add(scale_0, scale_1)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(data_4, full_2, float("0"), True)

        # pd_op.add: (xf32) <- (xf32, xf32)
        add_1 = paddle._C_ops.add(add_0, scale_2)

        # pd_op.scale: (1xf32) <- (1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(divide_0, full_1, float("0"), True)

        # pd_op.add: (1xf32) <- (xf32, 1xf32)
        add_2 = paddle._C_ops.add(add_1, scale_3)
        return (
            full_0,
            scale_0,
            full_1,
            scale_1,
            add_0,
            full_2,
            scale_2,
            add_1,
            assign_0,
            scale_3,
            add_2,
            divide_0,
        )
