import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2):
        # pd_op.cast: (-1xi64) <- (-1xf32)
        cast_0 = paddle._C_ops.cast(data_1, paddle.int64)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1xi64) <- (-1xi64, 1xf32)
        scale_2 = paddle._C_ops.scale(cast_0, full_1, float("1"), True)

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_1 = paddle._C_ops.cast(scale_2, paddle.float32)

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_0 = paddle._C_ops.subtract(cast_1, data_1)

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_2 = paddle._C_ops.cast(cast_0, paddle.float32)

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_1 = paddle._C_ops.subtract(data_1, cast_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [-1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_1

        # pd_op.unsqueeze: (-1x1xi64) <- (-1xi64, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(cast_0, full_int_array_1)

        # pd_op.cross_entropy_with_softmax: (-1x8xf32, -1x1xf32) <- (-1x8xf32, -1x1xi64)
        cross_entropy_with_softmax_0, cross_entropy_with_softmax_1 = (
            lambda x, f: f(x)
        )(
            paddle._C_ops.cross_entropy_with_softmax(
                data_0, unsqueeze_0, False, True, True, -100, -1
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.squeeze: (-1xf32) <- (-1x1xf32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(
            cross_entropy_with_softmax_1, full_int_array_1
        )

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_0 = paddle._C_ops.multiply(squeeze_0, subtract_0)

        # pd_op.unsqueeze: (-1x1xi64) <- (-1xi64, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(scale_2, full_int_array_1)

        # pd_op.cross_entropy_with_softmax: (-1x8xf32, -1x1xf32) <- (-1x8xf32, -1x1xi64)
        cross_entropy_with_softmax_2, cross_entropy_with_softmax_3 = (
            lambda x, f: f(x)
        )(
            paddle._C_ops.cross_entropy_with_softmax(
                data_0, unsqueeze_1, False, True, True, -100, -1
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.squeeze: (-1xf32) <- (-1x1xf32, 1xi64)
        squeeze_1 = paddle._C_ops.squeeze(
            cross_entropy_with_softmax_3, full_int_array_1
        )

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_1 = paddle._C_ops.multiply(squeeze_1, subtract_1)

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_0 = paddle._C_ops.add(multiply_0, multiply_1)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.25"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_2 = full_0

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(add_0, full_0, float("0"), True)

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_2 = paddle._C_ops.multiply(scale_0, data_2)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_0 = []

        # pd_op.sum: (xf32) <- (-1xf32, 0xi64)
        sum_0 = paddle._C_ops.sum(multiply_2, full_int_array_0, None, False)

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(sum_0, full_0, float("0"), True)
        return (
            subtract_0,
            subtract_1,
            unsqueeze_0,
            cross_entropy_with_softmax_0,
            cross_entropy_with_softmax_1,
            assign_0,
            squeeze_0,
            multiply_0,
            unsqueeze_1,
            cross_entropy_with_softmax_2,
            cross_entropy_with_softmax_3,
            assign_1,
            squeeze_1,
            multiply_1,
            full_0,
            scale_0,
            multiply_2,
            full_int_array_0,
            assign_2,
            scale_1,
        )
