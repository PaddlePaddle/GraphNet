import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2):
        # pd_op.cast: (10308x4xf32) <- (10308x4xf32)
        cast_0 = paddle._C_ops.cast(data_1, paddle.float32)

        # pd_op.full: (xf32) <- ()
        full_2 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (10308x4xb) <- (10308x4xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(cast_0, full_2)

        # pd_op.cast: (10308x4xf32) <- (10308x4xb)
        cast_2 = paddle._C_ops.cast(greater_than_0, paddle.float32)

        # pd_op.multiply: (10308x4xf32) <- (10308x4xf32, 10308x4xf32)
        multiply_0 = paddle._C_ops.multiply(cast_0, cast_2)

        # pd_op.subtract: (10308x4xf32) <- (10308x4xf32, 10308x4xf32)
        subtract_0 = paddle._C_ops.subtract(data_0, cast_0)

        # pd_op.abs: (10308x4xf32) <- (10308x4xf32)
        abs_0 = paddle._C_ops.abs(subtract_0)

        # pd_op.pow: (10308x4xf32) <- (10308x4xf32)
        pow_0 = paddle._C_ops.pow(abs_0, float("2"))

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.75"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (10308x4xf32) <- (10308x4xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(pow_0, full_0, float("0"), True)

        # pd_op.less_equal: (10308x4xb) <- (10308x4xf32, xf32)
        less_equal_0 = paddle._C_ops.less_equal(cast_0, full_2)

        # pd_op.cast: (10308x4xf32) <- (10308x4xb)
        cast_1 = paddle._C_ops.cast(less_equal_0, paddle.float32)

        # pd_op.multiply: (10308x4xf32) <- (10308x4xf32, 10308x4xf32)
        multiply_1 = paddle._C_ops.multiply(scale_0, cast_1)

        # pd_op.add: (10308x4xf32) <- (10308x4xf32, 10308x4xf32)
        add_0 = paddle._C_ops.add(multiply_0, multiply_1)

        # pd_op.bce_loss: (10308x4xf32) <- (10308x4xf32, 10308x4xf32)
        bce_loss_0 = paddle._C_ops.bce_loss(data_0, cast_0)

        # pd_op.multiply: (10308x4xf32) <- (10308x4xf32, 10308x4xf32)
        multiply_2 = paddle._C_ops.multiply(bce_loss_0, add_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [1]

        # pd_op.sum: (10308xf32) <- (10308x4xf32, 1xi64)
        sum_1 = paddle._C_ops.sum(multiply_2, full_int_array_0, None, False)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (10308xf32) <- (10308xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(sum_1, full_1, float("0"), True)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_1 = []

        # pd_op.sum: (xf32) <- (10308xf32, 0xi64)
        sum_0 = paddle._C_ops.sum(scale_1, full_int_array_1, None, False)

        # pd_op.divide: (xf32) <- (xf32, xf32)
        divide_0 = paddle._C_ops.divide(sum_0, data_2)
        return (
            cast_0,
            multiply_0,
            subtract_0,
            abs_0,
            full_0,
            scale_0,
            cast_1,
            multiply_1,
            add_0,
            bce_loss_0,
            multiply_2,
            full_int_array_0,
            full_1,
            scale_1,
            full_int_array_1,
            sum_0,
            divide_0,
        )
