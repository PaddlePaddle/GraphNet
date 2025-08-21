import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3):
        # pd_op.sigmoid: (8x300x4xf32) <- (8x300x4xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(data_0)

        # pd_op.pow: (8x300x4xf32) <- (8x300x4xf32)
        pow_0 = paddle._C_ops.pow(sigmoid_0, float("2"))

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.75"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (8x300x4xf32) <- (8x300x4xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(pow_0, full_0, float("0"), True)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (8x300x4xf32) <- (8x300x4xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(data_2, full_1, float("1"), True)

        # pd_op.multiply: (8x300x4xf32) <- (8x300x4xf32, 8x300x4xf32)
        multiply_0 = paddle._C_ops.multiply(scale_0, scale_1)

        # pd_op.multiply: (8x300x4xf32) <- (8x300x4xf32, 8x300x4xf32)
        multiply_1 = paddle._C_ops.multiply(data_1, data_2)

        # pd_op.add: (8x300x4xf32) <- (8x300x4xf32, 8x300x4xf32)
        add_0 = paddle._C_ops.add(multiply_0, multiply_1)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.sigmoid_cross_entropy_with_logits: (8x300x4xf32) <- (8x300x4xf32, 8x300x4xf32, None)
        sigmoid_cross_entropy_with_logits_0 = (
            paddle._C_ops.sigmoid_cross_entropy_with_logits(
                data_0, data_1, None, False, -100
            )
        )

        # pd_op.multiply: (8x300x4xf32) <- (8x300x4xf32, 8x300x4xf32)
        multiply_2 = paddle._C_ops.multiply(sigmoid_cross_entropy_with_logits_0, add_0)

        # pd_op.mean: (8x4xf32) <- (8x300x4xf32)
        mean_0 = paddle._C_ops.mean(multiply_2, [1], False)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_0 = []

        # pd_op.sum: (xf32) <- (8x4xf32, 0xi64)
        sum_0 = paddle._C_ops.sum(mean_0, full_int_array_0, None, False)

        # pd_op.divide: (1xf32) <- (xf32, 1xf32)
        divide_0 = paddle._C_ops.divide(sum_0, data_3)
        return (
            sigmoid_0,
            full_0,
            scale_0,
            scale_1,
            multiply_0,
            multiply_1,
            add_0,
            sigmoid_cross_entropy_with_logits_0,
            multiply_2,
            mean_0,
            full_int_array_0,
            sum_0,
            divide_0,
        )
