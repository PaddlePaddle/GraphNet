import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6):
        # pd_op.full: (1xi64) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("-2"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.argmax: (1x8400xi64) <- (1x1x8400xf32, 1xi64)
        argmax_0 = paddle._C_ops.argmax(data_0, full_0, False, False, paddle.int64)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x1xi32) <- (1x1xi32, 1xf32)
        scale_0 = paddle._C_ops.scale(data_1, full_1, float("0"), True)

        # pd_op.cast: (1x1xi64) <- (1x1xi32)
        cast_0 = paddle._C_ops.cast(scale_0, paddle.int64)

        # pd_op.add: (1x8400xi64) <- (1x8400xi64, 1x1xi64)
        add_0 = paddle._C_ops.add(argmax_0, cast_0)

        # pd_op.flatten: (1xi32) <- (1x1x1xi32)
        flatten_0 = paddle._C_ops.flatten(data_2, 0, 2)

        # pd_op.flatten: (8400xi64) <- (1x8400xi64)
        flatten_1 = paddle._C_ops.flatten(add_0, 0, 1)

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.gather: (8400xi32) <- (1xi32, 8400xi64, 1xi32)
        gather_0 = paddle._C_ops.gather(flatten_0, flatten_1, full_2)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 8400]

        # pd_op.reshape: (1x8400xi32) <- (8400xi32, 2xi64)
        reshape_1 = paddle._C_ops.reshape(gather_0, full_int_array_0)

        # pd_op.full: (xf32) <- ()
        full_3 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (1x8400xb) <- (1x8400xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(data_3, full_3)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (1x8400xi32) <- (1x8400xi32, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            reshape_1, full_4, paddle.int32, paddle.framework._current_expected_place()
        )

        # pd_op.where: (1x8400xi32) <- (1x8400xb, 1x8400xi32, 1x8400xi32)
        where_0 = paddle._C_ops.where(greater_than_0, reshape_1, full_like_0)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [-1, 4]

        # pd_op.reshape: (1x4xf32) <- (1x1x4xf32, 2xi64)
        reshape_2 = paddle._C_ops.reshape(data_4, full_int_array_1)

        # pd_op.gather: (8400x4xf32) <- (1x4xf32, 8400xi64, 1xi32)
        gather_1 = paddle._C_ops.gather(reshape_2, flatten_1, full_2)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_2 = [1, 8400, 4]

        # pd_op.reshape: (1x8400x4xf32) <- (8400x4xf32, 3xi64)
        reshape_0 = paddle._C_ops.reshape(gather_1, full_int_array_2)

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("3"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.one_hot: (1x8400x3xf32) <- (1x8400xi32, 1xi32)
        one_hot_0 = paddle._C_ops.one_hot(
            where_0 % paddle.cast(full_5, where_0.dtype), full_5
        )

        # pd_op.full: (2xi64) <- ()
        full_6 = paddle._C_ops.full(
            [2], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (2xi64) <- (2xi64)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_6,
            [2],
            paddle.int64,
            [float("0"), float("1")],
            paddle.framework._current_expected_place(),
        )

        # pd_op.index_select: (1x8400x2xf32) <- (1x8400x3xf32, 2xi64)
        index_select_0 = paddle._C_ops.index_select(one_hot_0, assign_value__0, -1)

        # pd_op.multiply: (1x1x8400xf32) <- (1x1x8400xf32, 1x1x8400xf32)
        multiply_1 = paddle._C_ops.multiply(data_5, data_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [-1]

        # pd_op.max: (1x1x1xf32) <- (1x1x8400xf32, 1xi64)
        max_0 = paddle._C_ops.max(multiply_1, full_int_array_3, True)

        # pd_op.multiply: (1x1x8400xf32) <- (1x1x8400xf32, 1x1x8400xf32)
        multiply_2 = paddle._C_ops.multiply(data_6, data_0)

        # pd_op.max: (1x1x1xf32) <- (1x1x8400xf32, 1xi64)
        max_1 = paddle._C_ops.max(multiply_2, full_int_array_3, True)

        # pd_op.scale: (1x1x1xf32) <- (1x1x1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(max_0, full_1, float("1e-09"), True)

        # pd_op.divide: (1x1x8400xf32) <- (1x1x8400xf32, 1x1x1xf32)
        divide_0 = paddle._C_ops.divide(multiply_1, scale_1)

        # pd_op.multiply: (1x1x8400xf32) <- (1x1x8400xf32, 1x1x1xf32)
        multiply_3 = paddle._C_ops.multiply(divide_0, max_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [-2]

        # pd_op.max: (1x8400xf32) <- (1x1x8400xf32, 1xi64)
        max_2 = paddle._C_ops.max(multiply_3, full_int_array_4, False)

        # pd_op.unsqueeze: (1x8400x1xf32) <- (1x8400xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(max_2, full_int_array_3)

        # pd_op.multiply: (1x8400x2xf32) <- (1x8400x2xf32, 1x8400x1xf32)
        multiply_0 = paddle._C_ops.multiply(index_select_0, unsqueeze_0)
        return where_0, reshape_0, multiply_0, add_0
