import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7):
        # pd_op.maximum: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        maximum_0 = paddle._C_ops.maximum(data_0, data_4)

        # pd_op.maximum: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        maximum_1 = paddle._C_ops.maximum(data_1, data_5)

        # pd_op.minimum: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        minimum_0 = paddle._C_ops.minimum(data_2, data_6)

        # pd_op.minimum: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        minimum_1 = paddle._C_ops.minimum(data_3, data_7)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_0 = paddle._C_ops.subtract(minimum_0, maximum_0)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (-1x1xf32) <- (-1x1xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(subtract_0, full_0, full_1)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_1 = paddle._C_ops.subtract(minimum_1, maximum_1)

        # pd_op.clip: (-1x1xf32) <- (-1x1xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(subtract_1, full_0, full_1)

        # pd_op.multiply: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        multiply_0 = paddle._C_ops.multiply(clip_0, clip_1)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_2 = paddle._C_ops.subtract(data_2, data_0)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_3 = paddle._C_ops.subtract(data_3, data_1)

        # pd_op.multiply: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        multiply_1 = paddle._C_ops.multiply(subtract_2, subtract_3)

        # pd_op.clip: (-1x1xf32) <- (-1x1xf32, 1xf32, 1xf32)
        clip_2 = paddle._C_ops.clip(multiply_1, full_0, full_1)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_4 = paddle._C_ops.subtract(data_6, data_4)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_5 = paddle._C_ops.subtract(data_7, data_5)

        # pd_op.multiply: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        multiply_2 = paddle._C_ops.multiply(subtract_4, subtract_5)

        # pd_op.clip: (-1x1xf32) <- (-1x1xf32, 1xf32, 1xf32)
        clip_3 = paddle._C_ops.clip(multiply_2, full_0, full_1)

        # pd_op.add: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        add_0 = paddle._C_ops.add(clip_2, clip_3)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_6 = paddle._C_ops.subtract(add_0, multiply_0)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(subtract_6, full_2, float("1e-09"), True)

        # pd_op.divide: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        divide_0 = paddle._C_ops.divide(multiply_0, scale_0)
        return divide_0
