import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6):
        # pd_op.full: (1xi64) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("-2"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.argmax: (2x-1xi64) <- (2x-1x-1xf32, 1xi64)
        argmax_0 = paddle._C_ops.argmax(data_2, full_0, False, False, paddle.int64)
        del full_0

        # pd_op.full: (1xf64) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("2"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (2xi32) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_1, full_2, full_3, dtype="int32")
        del full_1, full_2, full_3

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [-1]

        # pd_op.unsqueeze: (2x1xi32) <- (2xi32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(arange_0, full_int_array_0)
        del arange_0

        # pd_op.cast: (xi32) <- (xi64)
        cast_0 = paddle._C_ops.cast(data_0, paddle.int32)
        del data_0

        # pd_op.multiply: (2x1xi32) <- (2x1xi32, xi32)
        multiply_1 = paddle._C_ops.multiply(unsqueeze_0, cast_0)
        del cast_0, unsqueeze_0

        # pd_op.cast: (2x1xi64) <- (2x1xi32)
        cast_1 = paddle._C_ops.cast(multiply_1, paddle.int64)
        del multiply_1

        # pd_op.add: (2x-1xi64) <- (2x-1xi64, 2x1xi64)
        add_0 = paddle._C_ops.add(argmax_0, cast_1)
        del argmax_0, cast_1

        # pd_op.flatten: (-1xi32) <- (2x-1x1xi32)
        flatten_0 = paddle._C_ops.flatten(data_3, 0, 2)
        del data_3

        # pd_op.flatten: (-1xi64) <- (2x-1xi64)
        flatten_1 = paddle._C_ops.flatten(add_0, 0, 1)
        del add_0

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.gather: (-1xi32) <- (-1xi32, -1xi64, 1xi32)
        gather_0 = paddle._C_ops.gather(flatten_0, flatten_1, full_4)
        del flatten_0

        # pd_op.full: (xi64) <- ()
        full_5 = paddle._C_ops.full(
            [], float("2"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_0 = [full_5, data_1]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.reshape: (2x-1xi32) <- (-1xi32, 2xi64)
        reshape_1 = paddle._C_ops.reshape(gather_0, stack_0)
        del gather_0, stack_0

        # pd_op.full: (xf32) <- ()
        full_6 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (2x-1xb) <- (2x-1xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(data_4, full_6)
        del data_4, full_6

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("11"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (2x-1xi32) <- (2x-1xi32, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            reshape_1, full_7, paddle.int32, paddle.framework._current_expected_place()
        )
        del full_7

        # pd_op.where: (2x-1xi32) <- (2x-1xb, 2x-1xi32, 2x-1xi32)
        where_0 = paddle._C_ops.where(greater_than_0, reshape_1, full_like_0)
        del full_like_0, greater_than_0, reshape_1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [-1, 4]

        # pd_op.reshape: (-1x4xf32) <- (2x-1x4xf32, 2xi64)
        reshape_2 = paddle._C_ops.reshape(data_5, full_int_array_1)
        del full_int_array_1

        # pd_op.gather: (-1x4xf32) <- (-1x4xf32, -1xi64, 1xi32)
        gather_1 = paddle._C_ops.gather(reshape_2, flatten_1, full_4)
        del flatten_1, full_4, reshape_2

        # pd_op.full: (xi64) <- ()
        full_8 = paddle._C_ops.full(
            [], float("4"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_1 = [full_5, data_1, full_8]
        del data_1, full_5, full_8

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_1, 0)
        del combine_1

        # pd_op.reshape: (2x-1x4xf32) <- (-1x4xf32, 3xi64)
        reshape_0 = paddle._C_ops.reshape(gather_1, stack_1)
        del gather_1, stack_1

        # pd_op.full: (1xi32) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("12"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.one_hot: (2x-1x12xf32) <- (2x-1xi32, 1xi32)
        one_hot_0 = paddle._C_ops.one_hot(
            where_0 % paddle.cast(full_9, where_0.dtype), full_9
        )
        del full_9

        # pd_op.full: (11xi64) <- ()
        full_10 = paddle._C_ops.full(
            [11], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (11xi64) <- (11xi64)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_10,
            [11],
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
            ],
            paddle.framework._current_expected_place(),
        )
        del full_10

        # pd_op.index_select: (2x-1x11xf32) <- (2x-1x12xf32, 11xi64)
        index_select_0 = paddle._C_ops.index_select(one_hot_0, assign_value__0, -1)
        del assign_value__0, one_hot_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        # pd_op.unsqueeze: (2x-1x1x4xf32) <- (2x-1x4xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(data_5, full_int_array_2)
        del data_5

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.unsqueeze: (2x1x-1x4xf32) <- (2x-1x4xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(data_6, full_int_array_3)
        del data_6, full_int_array_3

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [0]

        # pd_op.slice: (2x-1x1x2xf32) <- (2x-1x1x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            unsqueeze_1, [3], full_int_array_4, full_int_array_2, [1], []
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [2147483647]

        # pd_op.slice: (2x-1x1x2xf32) <- (2x-1x1x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            unsqueeze_1, [3], full_int_array_2, full_int_array_5, [1], []
        )
        del unsqueeze_1

        # pd_op.slice: (2x1x-1x2xf32) <- (2x1x-1x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            unsqueeze_2, [3], full_int_array_4, full_int_array_2, [1], []
        )
        del full_int_array_4

        # pd_op.slice: (2x1x-1x2xf32) <- (2x1x-1x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            unsqueeze_2, [3], full_int_array_2, full_int_array_5, [1], []
        )
        del full_int_array_2, full_int_array_5, unsqueeze_2

        # pd_op.maximum: (2x-1x-1x2xf32) <- (2x-1x1x2xf32, 2x1x-1x2xf32)
        maximum_0 = paddle._C_ops.maximum(slice_0, slice_2)

        # pd_op.minimum: (2x-1x-1x2xf32) <- (2x-1x1x2xf32, 2x1x-1x2xf32)
        minimum_0 = paddle._C_ops.minimum(slice_1, slice_3)

        # pd_op.subtract: (2x-1x-1x2xf32) <- (2x-1x-1x2xf32, 2x-1x-1x2xf32)
        subtract_0 = paddle._C_ops.subtract(minimum_0, maximum_0)
        del maximum_0, minimum_0

        # pd_op.full: (1xf32) <- ()
        full_11 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_12 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (2x-1x-1x2xf32) <- (2x-1x-1x2xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(subtract_0, full_11, full_12)
        del subtract_0

        # pd_op.prod: (2x-1x-1xf32) <- (2x-1x-1x2xf32, 1xi64)
        prod_0 = paddle._C_ops.prod(clip_0, full_int_array_0, False, False)
        del clip_0

        # pd_op.subtract: (2x-1x1x2xf32) <- (2x-1x1x2xf32, 2x-1x1x2xf32)
        subtract_1 = paddle._C_ops.subtract(slice_1, slice_0)
        del slice_0, slice_1

        # pd_op.clip: (2x-1x1x2xf32) <- (2x-1x1x2xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(subtract_1, full_11, full_12)
        del subtract_1

        # pd_op.prod: (2x-1x1xf32) <- (2x-1x1x2xf32, 1xi64)
        prod_1 = paddle._C_ops.prod(clip_1, full_int_array_0, False, False)
        del clip_1

        # pd_op.subtract: (2x1x-1x2xf32) <- (2x1x-1x2xf32, 2x1x-1x2xf32)
        subtract_2 = paddle._C_ops.subtract(slice_3, slice_2)
        del slice_2, slice_3

        # pd_op.clip: (2x1x-1x2xf32) <- (2x1x-1x2xf32, 1xf32, 1xf32)
        clip_2 = paddle._C_ops.clip(subtract_2, full_11, full_12)
        del full_11, full_12, subtract_2

        # pd_op.prod: (2x1x-1xf32) <- (2x1x-1x2xf32, 1xi64)
        prod_2 = paddle._C_ops.prod(clip_2, full_int_array_0, False, False)
        del clip_2

        # pd_op.add: (2x-1x-1xf32) <- (2x-1x1xf32, 2x1x-1xf32)
        add_1 = paddle._C_ops.add(prod_1, prod_2)
        del prod_1, prod_2

        # pd_op.subtract: (2x-1x-1xf32) <- (2x-1x-1xf32, 2x-1x-1xf32)
        subtract_3 = paddle._C_ops.subtract(add_1, prod_0)
        del add_1

        # pd_op.full: (1xf32) <- ()
        full_13 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (2x-1x-1xf32) <- (2x-1x-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(subtract_3, full_13, float("1e-09"), True)
        del full_13, subtract_3

        # pd_op.divide: (2x-1x-1xf32) <- (2x-1x-1xf32, 2x-1x-1xf32)
        divide_0 = paddle._C_ops.divide(prod_0, scale_0)
        del prod_0, scale_0

        # pd_op.multiply: (2x-1x-1xf32) <- (2x-1x-1xf32, 2x-1x-1xf32)
        multiply_2 = paddle._C_ops.multiply(divide_0, data_2)
        del data_2, divide_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [-2]

        # pd_op.max: (2x-1xf32) <- (2x-1x-1xf32, 1xi64)
        max_0 = paddle._C_ops.max(multiply_2, full_int_array_6, False)
        del full_int_array_6, multiply_2

        # pd_op.unsqueeze: (2x-1x1xf32) <- (2x-1xf32, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(max_0, full_int_array_0)
        del full_int_array_0, max_0

        # pd_op.multiply: (2x-1x11xf32) <- (2x-1x11xf32, 2x-1x1xf32)
        multiply_0 = paddle._C_ops.multiply(index_select_0, unsqueeze_3)
        del index_select_0, unsqueeze_3, where_0

        return reshape_0, multiply_0
