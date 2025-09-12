import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [1]

        # pd_op.unsqueeze: (2x1x3549xf32) <- (2x3549xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(data_0, full_int_array_0)
        del data_0, full_int_array_0

        # pd_op.full: (xf32) <- ()
        full_0 = paddle._C_ops.full(
            [], float("1"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (2x1x3549xb) <- (2x1x3549xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(unsqueeze_0, full_0)
        del full_0, unsqueeze_0

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_1 = [1, 14, 1]

        # pd_op.tile: (2x14x3549xb) <- (2x1x3549xb, 3xi64)
        tile_0 = paddle._C_ops.tile(greater_than_0, full_int_array_1)
        del full_int_array_1, greater_than_0

        # pd_op.full: (1xi64) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("-2"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.argmax: (2x3549xi64) <- (2x14x3549xf32, 1xi64)
        argmax_0 = paddle._C_ops.argmax(data_1, full_1, False, False, paddle.int64)

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("14"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.one_hot: (2x3549x14xf32) <- (2x3549xi64, 1xi32)
        one_hot_0 = paddle._C_ops.one_hot(
            argmax_0 % paddle.cast(full_2, argmax_0.dtype), full_2
        )
        del argmax_0, full_2

        # pd_op.transpose: (2x14x3549xf32) <- (2x3549x14xf32)
        transpose_0 = paddle._C_ops.transpose(one_hot_0, [0, 2, 1])
        del one_hot_0

        # pd_op.where: (2x14x3549xf32) <- (2x14x3549xb, 2x14x3549xf32, 2x14x3549xf32)
        where_0 = paddle._C_ops.where(tile_0, transpose_0, data_2)
        del data_2, tile_0, transpose_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [-2]

        # pd_op.sum: (2x3549xf32) <- (2x14x3549xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(where_0, full_int_array_2, None, False)

        # pd_op.argmax: (2x3549xi64) <- (2x14x3549xf32, 1xi64)
        argmax_1 = paddle._C_ops.argmax(where_0, full_1, False, False, paddle.int64)
        del full_1

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("14"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (2x1xi32) <- (2x1xi32, 1xf32)
        scale_0 = paddle._C_ops.scale(data_3, full_3, float("0"), True)
        del data_3, full_3

        # pd_op.cast: (2x1xi64) <- (2x1xi32)
        cast_0 = paddle._C_ops.cast(scale_0, paddle.int64)
        del scale_0

        # pd_op.add: (2x3549xi64) <- (2x3549xi64, 2x1xi64)
        add_0 = paddle._C_ops.add(argmax_1, cast_0)
        del argmax_1, cast_0

        # pd_op.flatten: (28xi32) <- (2x14x1xi32)
        flatten_0 = paddle._C_ops.flatten(data_4, 0, 2)
        del data_4

        # pd_op.flatten: (7098xi64) <- (2x3549xi64)
        flatten_1 = paddle._C_ops.flatten(add_0, 0, 1)
        del add_0

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.gather: (7098xi32) <- (28xi32, 7098xi64, 1xi32)
        gather_0 = paddle._C_ops.gather(flatten_0, flatten_1, full_4)
        del flatten_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [2, 3549]

        # pd_op.reshape: (2x3549xi32) <- (7098xi32, 2xi64)
        reshape_1 = paddle._C_ops.reshape(gather_0, full_int_array_3)
        del full_int_array_3, gather_0

        # pd_op.full: (xf32) <- ()
        full_5 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (2x3549xb) <- (2x3549xf32, xf32)
        greater_than_1 = paddle._C_ops.greater_than(sum_0, full_5)
        del full_5, sum_0

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("4"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (2x3549xi32) <- (2x3549xi32, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            reshape_1, full_6, paddle.int32, paddle.framework._current_expected_place()
        )
        del full_6

        # pd_op.where: (2x3549xi32) <- (2x3549xb, 2x3549xi32, 2x3549xi32)
        where_1 = paddle._C_ops.where(greater_than_1, reshape_1, full_like_0)
        del full_like_0, greater_than_1, reshape_1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [-1, 4]

        # pd_op.reshape: (28x4xf32) <- (2x14x4xf32, 2xi64)
        reshape_2 = paddle._C_ops.reshape(data_5, full_int_array_4)
        del data_5, full_int_array_4

        # pd_op.gather: (7098x4xf32) <- (28x4xf32, 7098xi64, 1xi32)
        gather_1 = paddle._C_ops.gather(reshape_2, flatten_1, full_4)
        del flatten_1, full_4, reshape_2

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_5 = [2, 3549, 4]

        # pd_op.reshape: (2x3549x4xf32) <- (7098x4xf32, 3xi64)
        reshape_0 = paddle._C_ops.reshape(gather_1, full_int_array_5)
        del full_int_array_5, gather_1

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("5"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.one_hot: (2x3549x5xf32) <- (2x3549xi32, 1xi32)
        one_hot_1 = paddle._C_ops.one_hot(
            where_1 % paddle.cast(full_7, where_1.dtype), full_7
        )
        del full_7

        # pd_op.full: (4xi64) <- ()
        full_8 = paddle._C_ops.full(
            [4], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (4xi64) <- (4xi64)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_8,
            [4],
            paddle.int64,
            [float("0"), float("1"), float("2"), float("3")],
            paddle.framework._current_expected_place(),
        )
        del full_8

        # pd_op.index_select: (2x3549x4xf32) <- (2x3549x5xf32, 4xi64)
        index_select_0 = paddle._C_ops.index_select(one_hot_1, assign_value__0, -1)
        del assign_value__0, one_hot_1

        # pd_op.multiply: (2x14x3549xf32) <- (2x14x3549xf32, 2x14x3549xf32)
        multiply_1 = paddle._C_ops.multiply(data_6, where_0)
        del data_6

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [-1]

        # pd_op.max: (2x14x1xf32) <- (2x14x3549xf32, 1xi64)
        max_0 = paddle._C_ops.max(multiply_1, full_int_array_6, True)

        # pd_op.multiply: (2x14x3549xf32) <- (2x14x3549xf32, 2x14x3549xf32)
        multiply_2 = paddle._C_ops.multiply(data_1, where_0)
        del data_1, where_0

        # pd_op.max: (2x14x1xf32) <- (2x14x3549xf32, 1xi64)
        max_1 = paddle._C_ops.max(multiply_2, full_int_array_6, True)
        del multiply_2

        # pd_op.full: (1xf32) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (2x14x1xf32) <- (2x14x1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(max_0, full_9, float("1e-09"), True)
        del full_9, max_0

        # pd_op.divide: (2x14x3549xf32) <- (2x14x3549xf32, 2x14x1xf32)
        divide_0 = paddle._C_ops.divide(multiply_1, scale_1)
        del multiply_1, scale_1

        # pd_op.multiply: (2x14x3549xf32) <- (2x14x3549xf32, 2x14x1xf32)
        multiply_3 = paddle._C_ops.multiply(divide_0, max_1)
        del divide_0, max_1

        # pd_op.max: (2x3549xf32) <- (2x14x3549xf32, 1xi64)
        max_2 = paddle._C_ops.max(multiply_3, full_int_array_2, False)
        del full_int_array_2, multiply_3

        # pd_op.unsqueeze: (2x3549x1xf32) <- (2x3549xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(max_2, full_int_array_6)
        del full_int_array_6, max_2

        # pd_op.multiply: (2x3549x4xf32) <- (2x3549x4xf32, 2x3549x1xf32)
        multiply_0 = paddle._C_ops.multiply(index_select_0, unsqueeze_1)
        del index_select_0, unsqueeze_1, where_1

        return reshape_0, multiply_0
