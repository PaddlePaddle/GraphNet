import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.cast: (461x1xf32) <- (461x1xf64)
        cast_0 = paddle._C_ops.cast(data_1, paddle.float32)

        # pd_op.full: (xf32) <- ()
        full_0 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (461x1xb) <- (461x1xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(cast_0, full_0)

        # pd_op.cast: (461x1xf32) <- (461x1xb)
        cast_1 = paddle._C_ops.cast(greater_than_0, paddle.float32)

        # pd_op.multiply: (461x1xf32) <- (461x1xf32, 461x1xf32)
        multiply_0 = paddle._C_ops.multiply(cast_0, cast_1)

        # pd_op.subtract: (461x1xf32) <- (461x1xf32, 461x1xf32)
        subtract_0 = paddle._C_ops.subtract(data_0, cast_0)

        # pd_op.abs: (461x1xf32) <- (461x1xf32)
        abs_0 = paddle._C_ops.abs(subtract_0)

        # pd_op.pow: (461x1xf32) <- (461x1xf32)
        pow_0 = paddle._C_ops.pow(abs_0, float("2"))

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0.75"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (461x1xf32) <- (461x1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(pow_0, full_1, float("0"), True)

        # pd_op.less_equal: (461x1xb) <- (461x1xf32, xf32)
        less_equal_0 = paddle._C_ops.less_equal(cast_0, full_0)

        # pd_op.cast: (461x1xf32) <- (461x1xb)
        cast_2 = paddle._C_ops.cast(less_equal_0, paddle.float32)

        # pd_op.multiply: (461x1xf32) <- (461x1xf32, 461x1xf32)
        multiply_1 = paddle._C_ops.multiply(scale_0, cast_2)

        # pd_op.add: (461x1xf32) <- (461x1xf32, 461x1xf32)
        add_0 = paddle._C_ops.add(multiply_0, multiply_1)

        # pd_op.bce_loss: (461x1xf32) <- (461x1xf32, 461x1xf32)
        bce_loss_0 = paddle._C_ops.bce_loss(data_0, cast_0)

        # pd_op.multiply: (461x1xf32) <- (461x1xf32, 461x1xf32)
        multiply_2 = paddle._C_ops.multiply(bce_loss_0, add_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [1]

        # pd_op.sum: (461xf32) <- (461x1xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(multiply_2, full_int_array_0, None, False)
        return sum_0
