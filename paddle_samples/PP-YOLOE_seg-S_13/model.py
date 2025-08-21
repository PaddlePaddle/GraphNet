import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6):
        # pd_op.greater_equal: (13x1x160xb) <- (1x1x160xf32, 13x1x1xf32)
        greater_equal_0 = paddle._C_ops.greater_equal(data_1, data_2)

        # pd_op.less_than: (13x1x160xb) <- (1x1x160xf32, 13x1x1xf32)
        less_than_0 = paddle._C_ops.less_than(data_1, data_3)

        # pd_op.multiply: (13x1x160xb) <- (13x1x160xb, 13x1x160xb)
        multiply_1 = paddle._C_ops.multiply(greater_equal_0, less_than_0)

        # pd_op.greater_equal: (13x160x1xb) <- (1x160x1xf32, 13x1x1xf32)
        greater_equal_1 = paddle._C_ops.greater_equal(data_4, data_5)

        # pd_op.multiply: (13x160x160xb) <- (13x1x160xb, 13x160x1xb)
        multiply_2 = paddle._C_ops.multiply(multiply_1, greater_equal_1)

        # pd_op.less_than: (13x160x1xb) <- (1x160x1xf32, 13x1x1xf32)
        less_than_1 = paddle._C_ops.less_than(data_4, data_6)

        # pd_op.multiply: (13x160x160xb) <- (13x160x160xb, 13x160x1xb)
        multiply_3 = paddle._C_ops.multiply(multiply_2, less_than_1)

        # pd_op.cast: (13x160x160xf32) <- (13x160x160xb)
        cast_0 = paddle._C_ops.cast(multiply_3, paddle.float32)

        # pd_op.multiply: (13x160x160xf32) <- (13x160x160xf32, 13x160x160xf32)
        multiply_0 = paddle._C_ops.multiply(data_0, cast_0)
        return cast_0, multiply_0
