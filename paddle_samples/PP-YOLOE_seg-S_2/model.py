import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6):
        # pd_op.masked_select: (-1xf32) <- (1x8400x4xf32, 1x8400x4xb)
        masked_select_0 = paddle._C_ops.masked_select(data_0, data_1)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [-1, 4]

        # pd_op.reshape: (-1x4xf32) <- (-1xf32, 2xi64)
        reshape_0 = paddle._C_ops.reshape(masked_select_0, full_int_array_1)

        # pd_op.floor: (-1x4xf32) <- (-1x4xf32)
        floor_0 = paddle._C_ops.floor(reshape_0)

        # pd_op.cast: (-1x4xi64) <- (-1x4xf32)
        cast_0 = paddle._C_ops.cast(floor_0, paddle.int64)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x4xi64) <- (-1x4xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(cast_0, full_0, float("1"), True)

        # pd_op.cast: (-1x4xf32) <- (-1x4xi64)
        cast_1 = paddle._C_ops.cast(scale_1, paddle.float32)

        # pd_op.subtract: (-1x4xf32) <- (-1x4xf32, -1x4xf32)
        subtract_0 = paddle._C_ops.subtract(cast_1, reshape_0)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x4xf32) <- (-1x4xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(subtract_0, full_1, float("1"), True)

        # pd_op.scale: (-1x4xi64) <- (-1x4xi64, 1xf32)
        scale_2 = paddle._C_ops.scale(cast_0, full_0, float("0"), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [-1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_2

        # pd_op.unsqueeze: (-1x4x1xi64) <- (-1x4xi64, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(scale_2, full_int_array_2)

        # pd_op.cross_entropy_with_softmax: (13x4x17xf32, 13x4x1xf32) <- (13x4x17xf32, -1x4x1xi64)
        cross_entropy_with_softmax_0, cross_entropy_with_softmax_1 = (
            lambda x, f: f(x)
        )(
            paddle._C_ops.cross_entropy_with_softmax(
                data_2, unsqueeze_0, False, True, True, -100, -1
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.squeeze: (13x4xf32) <- (13x4x1xf32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(
            cross_entropy_with_softmax_1, full_int_array_2
        )

        # pd_op.multiply: (13x4xf32) <- (13x4xf32, -1x4xf32)
        multiply_0 = paddle._C_ops.multiply(squeeze_0, subtract_0)

        # pd_op.scale: (-1x4xi64) <- (-1x4xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(scale_1, full_0, float("0"), True)

        # pd_op.unsqueeze: (-1x4x1xi64) <- (-1x4xi64, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(scale_3, full_int_array_2)

        # pd_op.cross_entropy_with_softmax: (13x4x17xf32, 13x4x1xf32) <- (13x4x17xf32, -1x4x1xi64)
        cross_entropy_with_softmax_2, cross_entropy_with_softmax_3 = (
            lambda x, f: f(x)
        )(
            paddle._C_ops.cross_entropy_with_softmax(
                data_2, unsqueeze_1, False, True, True, -100, -1
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.squeeze: (13x4xf32) <- (13x4x1xf32, 1xi64)
        squeeze_1 = paddle._C_ops.squeeze(
            cross_entropy_with_softmax_3, full_int_array_2
        )

        # pd_op.multiply: (13x4xf32) <- (13x4xf32, -1x4xf32)
        multiply_1 = paddle._C_ops.multiply(squeeze_1, scale_0)

        # pd_op.add: (13x4xf32) <- (13x4xf32, 13x4xf32)
        add_0 = paddle._C_ops.add(multiply_0, multiply_1)

        # pd_op.mean: (13x1xf32) <- (13x4xf32)
        mean_0 = paddle._C_ops.mean(add_0, [-1], True)

        # pd_op.multiply: (13x1xf32) <- (13x1xf32, 13x1xf32)
        multiply_2 = paddle._C_ops.multiply(mean_0, data_3)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_0 = []

        # pd_op.sum: (xf32) <- (13x1xf32, 0xi64)
        sum_0 = paddle._C_ops.sum(multiply_2, full_int_array_0, None, False)

        # pd_op.divide: (xf32) <- (xf32, xf32)
        divide_0 = paddle._C_ops.divide(sum_0, data_4)

        # pd_op.multiply: (1x8400x4xf32) <- (1x8400x4xf32, 8400x1xf32)
        multiply_3 = paddle._C_ops.multiply(data_5, data_6)
        return (
            subtract_0,
            scale_0,
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
            add_0,
            mean_0,
            multiply_2,
            full_int_array_0,
            sum_0,
            multiply_3,
            divide_0,
        )
