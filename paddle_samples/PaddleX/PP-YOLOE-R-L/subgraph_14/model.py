import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8
    ):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [1]

        # pd_op.unsqueeze: (1x1x21504xf32) <- (1x21504xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(data_2, full_int_array_0)
        del data_2, full_int_array_0

        # pd_op.full: (xf32) <- ()
        full_0 = paddle._C_ops.full(
            [], float("1"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (1x1x21504xb) <- (1x1x21504xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(unsqueeze_0, full_0)
        del full_0, unsqueeze_0

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_0 = [full_1, data_0, full_1]
        del full_1

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.tile: (1x-1x21504xb) <- (1x1x21504xb, 3xi64)
        tile_0 = paddle._C_ops.tile(greater_than_0, stack_0)
        del greater_than_0, stack_0

        # pd_op.full: (1xi64) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("-2"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.argmax: (1x21504xi64) <- (1x-1x21504xf32, 1xi64)
        argmax_0 = paddle._C_ops.argmax(data_3, full_2, False, False, paddle.int64)

        # pd_op.one_hot: (1x21504x-1xf32) <- (1x21504xi64, xi64)
        one_hot_0 = paddle._C_ops.one_hot(
            argmax_0 % paddle.cast(data_1, argmax_0.dtype), data_1
        )
        del argmax_0, data_1

        # pd_op.transpose: (1x-1x21504xf32) <- (1x21504x-1xf32)
        transpose_0 = paddle._C_ops.transpose(one_hot_0, [0, 2, 1])
        del one_hot_0

        # pd_op.where: (1x-1x21504xf32) <- (1x-1x21504xb, 1x-1x21504xf32, 1x-1x21504xf32)
        where_0 = paddle._C_ops.where(tile_0, transpose_0, data_4)
        del data_4, tile_0, transpose_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [-2]

        # pd_op.sum: (1x21504xf32) <- (1x-1x21504xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(where_0, full_int_array_1, None, False)

        # pd_op.argmax: (1x21504xi64) <- (1x-1x21504xf32, 1xi64)
        argmax_1 = paddle._C_ops.argmax(where_0, full_2, False, False, paddle.int64)
        del full_2

        # pd_op.cast: (xi32) <- (xi64)
        cast_0 = paddle._C_ops.cast(data_0, paddle.int32)
        del data_0

        # pd_op.multiply: (1x1xi32) <- (1x1xi32, xi32)
        multiply_1 = paddle._C_ops.multiply(data_5, cast_0)
        del cast_0, data_5

        # pd_op.cast: (1x1xi64) <- (1x1xi32)
        cast_1 = paddle._C_ops.cast(multiply_1, paddle.int64)
        del multiply_1

        # pd_op.add: (1x21504xi64) <- (1x21504xi64, 1x1xi64)
        add_0 = paddle._C_ops.add(argmax_1, cast_1)
        del argmax_1, cast_1

        # pd_op.flatten: (-1xi32) <- (1x-1x1xi32)
        flatten_0 = paddle._C_ops.flatten(data_6, 0, 2)
        del data_6

        # pd_op.flatten: (21504xi64) <- (1x21504xi64)
        flatten_1 = paddle._C_ops.flatten(add_0, 0, 1)
        del add_0

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.gather: (21504xi32) <- (-1xi32, 21504xi64, 1xi32)
        gather_0 = paddle._C_ops.gather(flatten_0, flatten_1, full_3)
        del flatten_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [1, 21504]

        # pd_op.reshape: (1x21504xi32) <- (21504xi32, 2xi64)
        reshape_1 = paddle._C_ops.reshape(gather_0, full_int_array_2)
        del full_int_array_2, gather_0

        # pd_op.full: (xf32) <- ()
        full_4 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (1x21504xb) <- (1x21504xf32, xf32)
        greater_than_1 = paddle._C_ops.greater_than(sum_0, full_4)
        del full_4, sum_0

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("15"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (1x21504xi32) <- (1x21504xi32, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            reshape_1, full_5, paddle.int32, paddle.framework._current_expected_place()
        )
        del full_5

        # pd_op.where: (1x21504xi32) <- (1x21504xb, 1x21504xi32, 1x21504xi32)
        where_1 = paddle._C_ops.where(greater_than_1, reshape_1, full_like_0)
        del full_like_0, greater_than_1, reshape_1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [-1, 5]

        # pd_op.reshape: (-1x5xf32) <- (1x-1x5xf32, 2xi64)
        reshape_2 = paddle._C_ops.reshape(data_7, full_int_array_3)
        del data_7, full_int_array_3

        # pd_op.gather: (21504x5xf32) <- (-1x5xf32, 21504xi64, 1xi32)
        gather_1 = paddle._C_ops.gather(reshape_2, flatten_1, full_3)
        del flatten_1, full_3, reshape_2

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_4 = [1, 21504, 5]

        # pd_op.reshape: (1x21504x5xf32) <- (21504x5xf32, 3xi64)
        reshape_0 = paddle._C_ops.reshape(gather_1, full_int_array_4)
        del full_int_array_4, gather_1

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("16"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.one_hot: (1x21504x16xf32) <- (1x21504xi32, 1xi32)
        one_hot_1 = paddle._C_ops.one_hot(
            where_1 % paddle.cast(full_6, where_1.dtype), full_6
        )
        del full_6

        # pd_op.full: (15xi64) <- ()
        full_7 = paddle._C_ops.full(
            [15], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (15xi64) <- (15xi64)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_7,
            [15],
            paddle.int64,
            [
                float("0"),
                float("1"),
                float("2"),
                float("3"),
                float("4"),
                float("5"),
                float("6"),
                float("7"),
                float("8"),
                float("9"),
                float("10"),
                float("11"),
                float("12"),
                float("13"),
                float("14"),
            ],
            paddle.framework._current_expected_place(),
        )
        del full_7

        # pd_op.index_select: (1x21504x15xf32) <- (1x21504x16xf32, 15xi64)
        index_select_0 = paddle._C_ops.index_select(one_hot_1, assign_value__0, -1)
        del assign_value__0, one_hot_1

        # pd_op.multiply: (1x-1x21504xf32) <- (1x-1x21504xf32, 1x-1x21504xf32)
        multiply_2 = paddle._C_ops.multiply(data_8, where_0)
        del data_8

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [-1]

        # pd_op.max: (1x-1x1xf32) <- (1x-1x21504xf32, 1xi64)
        max_0 = paddle._C_ops.max(multiply_2, full_int_array_5, True)

        # pd_op.multiply: (1x-1x21504xf32) <- (1x-1x21504xf32, 1x-1x21504xf32)
        multiply_3 = paddle._C_ops.multiply(data_3, where_0)
        del data_3, where_0

        # pd_op.max: (1x-1x1xf32) <- (1x-1x21504xf32, 1xi64)
        max_1 = paddle._C_ops.max(multiply_3, full_int_array_5, True)
        del multiply_3

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x-1x1xf32) <- (1x-1x1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(max_0, full_8, float("1e-09"), True)
        del full_8, max_0

        # pd_op.divide: (1x-1x21504xf32) <- (1x-1x21504xf32, 1x-1x1xf32)
        divide_0 = paddle._C_ops.divide(multiply_2, scale_0)
        del multiply_2, scale_0

        # pd_op.multiply: (1x-1x21504xf32) <- (1x-1x21504xf32, 1x-1x1xf32)
        multiply_4 = paddle._C_ops.multiply(divide_0, max_1)
        del divide_0, max_1

        # pd_op.max: (1x21504xf32) <- (1x-1x21504xf32, 1xi64)
        max_2 = paddle._C_ops.max(multiply_4, full_int_array_1, False)
        del full_int_array_1, multiply_4

        # pd_op.unsqueeze: (1x21504x1xf32) <- (1x21504xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(max_2, full_int_array_5)
        del full_int_array_5, max_2

        # pd_op.multiply: (1x21504x15xf32) <- (1x21504x15xf32, 1x21504x1xf32)
        multiply_0 = paddle._C_ops.multiply(index_select_0, unsqueeze_1)
        del index_select_0, unsqueeze_1, where_1

        return reshape_0, multiply_0
