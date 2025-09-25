import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3):
        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_0 = paddle._C_ops.equal(data_0, data_1)
        del data_0, data_1

        # pd_op.sigmoid: (-1x1xf32) <- (-1x1xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(data_2)

        # pd_op.cast: (-1x1xf32) <- (-1x1xf64)
        cast_0 = paddle._C_ops.cast(data_3, paddle.float32)
        del data_3

        # pd_op.full: (xf32) <- ()
        full_0 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (-1x1xb) <- (-1x1xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(cast_0, full_0)

        # pd_op.cast: (-1x1xf32) <- (-1x1xb)
        cast_1 = paddle._C_ops.cast(greater_than_0, paddle.float32)
        del greater_than_0

        # pd_op.multiply: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        multiply_0 = paddle._C_ops.multiply(cast_0, cast_1)
        del cast_1

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_0 = paddle._C_ops.subtract(sigmoid_0, cast_0)

        # pd_op.abs: (-1x1xf32) <- (-1x1xf32)
        abs_0 = paddle._C_ops.abs(subtract_0)

        # pd_op.pow: (-1x1xf32) <- (-1x1xf32)
        pow_0 = paddle._C_ops.pow(abs_0, float("2"))

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0.75"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(pow_0, full_1, float("0"), True)
        del pow_0

        # pd_op.less_equal: (-1x1xb) <- (-1x1xf32, xf32)
        less_equal_0 = paddle._C_ops.less_equal(cast_0, full_0)
        del full_0

        # pd_op.cast: (-1x1xf32) <- (-1x1xb)
        cast_2 = paddle._C_ops.cast(less_equal_0, paddle.float32)
        del less_equal_0

        # pd_op.multiply: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        multiply_1 = paddle._C_ops.multiply(scale_1, cast_2)

        # pd_op.add: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        add_0 = paddle._C_ops.add(multiply_0, multiply_1)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.sigmoid_cross_entropy_with_logits: (-1x1xf32) <- (-1x1xf32, -1x1xf32, None)
        sigmoid_cross_entropy_with_logits_0 = (
            paddle._C_ops.sigmoid_cross_entropy_with_logits(
                data_2, cast_0, None, False, -100
            )
        )
        del data_2

        # pd_op.multiply: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        multiply_2 = paddle._C_ops.multiply(sigmoid_cross_entropy_with_logits_0, add_0)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(multiply_2, full_3, float("0"), True)
        del multiply_2

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_0 = []

        # pd_op.sum: (xf32) <- (-1x1xf32, 0xi64)
        sum_0 = paddle._C_ops.sum(scale_2, full_int_array_0, None, False)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0.0625"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(sum_0, full_4, float("0"), True)
        del (
            abs_0,
            add_0,
            cast_0,
            cast_2,
            full_1,
            full_3,
            full_4,
            full_int_array_0,
            multiply_0,
            multiply_1,
            scale_1,
            scale_2,
            sigmoid_0,
            sigmoid_cross_entropy_with_logits_0,
            subtract_0,
            sum_0,
        )

        return scale_0
