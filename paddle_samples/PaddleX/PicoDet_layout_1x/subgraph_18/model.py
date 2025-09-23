import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.cast: (5266x11xf32) <- (5266x11xf64)
        cast_0 = paddle._C_ops.cast(data_1, paddle.float32)
        del data_1

        # pd_op.full: (xf32) <- ()
        full_0 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (5266x11xb) <- (5266x11xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(cast_0, full_0)

        # pd_op.cast: (5266x11xf32) <- (5266x11xb)
        cast_1 = paddle._C_ops.cast(greater_than_0, paddle.float32)
        del greater_than_0

        # pd_op.multiply: (5266x11xf32) <- (5266x11xf32, 5266x11xf32)
        multiply_0 = paddle._C_ops.multiply(cast_0, cast_1)
        del cast_1

        # pd_op.subtract: (5266x11xf32) <- (5266x11xf32, 5266x11xf32)
        subtract_0 = paddle._C_ops.subtract(data_0, cast_0)

        # pd_op.abs: (5266x11xf32) <- (5266x11xf32)
        abs_0 = paddle._C_ops.abs(subtract_0)
        del subtract_0

        # pd_op.pow: (5266x11xf32) <- (5266x11xf32)
        pow_0 = paddle._C_ops.pow(abs_0, float("2"))
        del abs_0

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0.75"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (5266x11xf32) <- (5266x11xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(pow_0, full_1, float("0"), True)
        del full_1, pow_0

        # pd_op.less_equal: (5266x11xb) <- (5266x11xf32, xf32)
        less_equal_0 = paddle._C_ops.less_equal(cast_0, full_0)
        del full_0

        # pd_op.cast: (5266x11xf32) <- (5266x11xb)
        cast_2 = paddle._C_ops.cast(less_equal_0, paddle.float32)
        del less_equal_0

        # pd_op.multiply: (5266x11xf32) <- (5266x11xf32, 5266x11xf32)
        multiply_1 = paddle._C_ops.multiply(scale_0, cast_2)
        del cast_2, scale_0

        # pd_op.add: (5266x11xf32) <- (5266x11xf32, 5266x11xf32)
        add_0 = paddle._C_ops.add(multiply_0, multiply_1)
        del multiply_0, multiply_1

        # pd_op.bce_loss: (5266x11xf32) <- (5266x11xf32, 5266x11xf32)
        bce_loss_0 = paddle._C_ops.bce_loss(data_0, cast_0)
        del cast_0, data_0

        # pd_op.multiply: (5266x11xf32) <- (5266x11xf32, 5266x11xf32)
        multiply_2 = paddle._C_ops.multiply(bce_loss_0, add_0)
        del add_0, bce_loss_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [1]

        # pd_op.sum: (5266xf32) <- (5266x11xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(multiply_2, full_int_array_0, None, False)
        del full_int_array_0, multiply_2

        return sum_0
