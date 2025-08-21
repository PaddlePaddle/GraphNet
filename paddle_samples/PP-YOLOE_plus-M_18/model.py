import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4):
        # pd_op.full: (1xi64) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("-2"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.argmax: (8x4116xi64) <- (8x1x4116xf32, 1xi64)
        argmax_0 = paddle._C_ops.argmax(data_0, full_0, False, False, paddle.int64)

        # pd_op.full: (1xf64) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("8"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (8xi32) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_1, full_2, full_3, dtype="int32")

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [-1]

        # pd_op.unsqueeze: (8x1xi32) <- (8xi32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(arange_0, full_int_array_0)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (8x1xi32) <- (8x1xi32, 1xf32)
        scale_0 = paddle._C_ops.scale(unsqueeze_0, full_4, float("0"), True)

        # pd_op.cast: (8x1xi64) <- (8x1xi32)
        cast_0 = paddle._C_ops.cast(scale_0, paddle.int64)

        # pd_op.add: (8x4116xi64) <- (8x4116xi64, 8x1xi64)
        add_0 = paddle._C_ops.add(argmax_0, cast_0)

        # pd_op.flatten: (8xi32) <- (8x1x1xi32)
        flatten_0 = paddle._C_ops.flatten(data_1, 0, 2)

        # pd_op.flatten: (32928xi64) <- (8x4116xi64)
        flatten_1 = paddle._C_ops.flatten(add_0, 0, 1)

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.gather: (32928xi32) <- (8xi32, 32928xi64, 1xi32)
        gather_0 = paddle._C_ops.gather(flatten_0, flatten_1, full_5)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [8, 4116]

        # pd_op.reshape: (8x4116xi32) <- (32928xi32, 2xi64)
        reshape_1 = paddle._C_ops.reshape(gather_0, full_int_array_1)

        # pd_op.full: (xf32) <- ()
        full_6 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (8x4116xb) <- (8x4116xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(data_2, full_6)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("4"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (8x4116xi32) <- (8x4116xi32, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            reshape_1, full_7, paddle.int32, paddle.framework._current_expected_place()
        )

        # pd_op.where: (8x4116xi32) <- (8x4116xb, 8x4116xi32, 8x4116xi32)
        where_0 = paddle._C_ops.where(greater_than_0, reshape_1, full_like_0)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [-1, 4]

        # pd_op.reshape: (8x4xf32) <- (8x1x4xf32, 2xi64)
        reshape_2 = paddle._C_ops.reshape(data_3, full_int_array_2)

        # pd_op.gather: (32928x4xf32) <- (8x4xf32, 32928xi64, 1xi32)
        gather_1 = paddle._C_ops.gather(reshape_2, flatten_1, full_5)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_3 = [8, 4116, 4]

        # pd_op.reshape: (8x4116x4xf32) <- (32928x4xf32, 3xi64)
        reshape_0 = paddle._C_ops.reshape(gather_1, full_int_array_3)

        # pd_op.full: (1xi32) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("5"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.one_hot: (8x4116x5xf32) <- (8x4116xi32, 1xi32)
        one_hot_0 = paddle._C_ops.one_hot(
            where_0 % paddle.cast(full_8, where_0.dtype), full_8
        )

        # pd_op.full: (4xi64) <- ()
        full_9 = paddle._C_ops.full(
            [4], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (4xi64) <- (4xi64)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_9,
            [4],
            paddle.int64,
            [float("0"), float("1"), float("2"), float("3")],
            paddle.framework._current_expected_place(),
        )

        # pd_op.index_select: (8x4116x4xf32) <- (8x4116x5xf32, 4xi64)
        index_select_0 = paddle._C_ops.index_select(one_hot_0, assign_value__0, -1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [2]

        # pd_op.unsqueeze: (8x1x1x4xf32) <- (8x1x4xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(data_3, full_int_array_4)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [1]

        # pd_op.unsqueeze: (8x1x4116x4xf32) <- (8x4116x4xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(data_4, full_int_array_5)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [0]

        # pd_op.slice: (8x1x1x2xf32) <- (8x1x1x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            unsqueeze_1, [3], full_int_array_6, full_int_array_4, [1], []
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [2147483647]

        # pd_op.slice: (8x1x1x2xf32) <- (8x1x1x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            unsqueeze_1, [3], full_int_array_4, full_int_array_7, [1], []
        )

        # pd_op.slice: (8x1x4116x2xf32) <- (8x1x4116x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            unsqueeze_2, [3], full_int_array_6, full_int_array_4, [1], []
        )

        # pd_op.slice: (8x1x4116x2xf32) <- (8x1x4116x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            unsqueeze_2, [3], full_int_array_4, full_int_array_7, [1], []
        )

        # pd_op.maximum: (8x1x4116x2xf32) <- (8x1x1x2xf32, 8x1x4116x2xf32)
        maximum_0 = paddle._C_ops.maximum(slice_0, slice_2)

        # pd_op.minimum: (8x1x4116x2xf32) <- (8x1x1x2xf32, 8x1x4116x2xf32)
        minimum_0 = paddle._C_ops.minimum(slice_1, slice_3)

        # pd_op.subtract: (8x1x4116x2xf32) <- (8x1x4116x2xf32, 8x1x4116x2xf32)
        subtract_0 = paddle._C_ops.subtract(minimum_0, maximum_0)

        # pd_op.full: (1xf32) <- ()
        full_10 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_11 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (8x1x4116x2xf32) <- (8x1x4116x2xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(subtract_0, full_10, full_11)

        # pd_op.prod: (8x1x4116xf32) <- (8x1x4116x2xf32, 1xi64)
        prod_0 = paddle._C_ops.prod(clip_0, full_int_array_0, False, False)

        # pd_op.subtract: (8x1x1x2xf32) <- (8x1x1x2xf32, 8x1x1x2xf32)
        subtract_1 = paddle._C_ops.subtract(slice_1, slice_0)

        # pd_op.clip: (8x1x1x2xf32) <- (8x1x1x2xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(subtract_1, full_10, full_11)

        # pd_op.prod: (8x1x1xf32) <- (8x1x1x2xf32, 1xi64)
        prod_1 = paddle._C_ops.prod(clip_1, full_int_array_0, False, False)

        # pd_op.subtract: (8x1x4116x2xf32) <- (8x1x4116x2xf32, 8x1x4116x2xf32)
        subtract_2 = paddle._C_ops.subtract(slice_3, slice_2)

        # pd_op.clip: (8x1x4116x2xf32) <- (8x1x4116x2xf32, 1xf32, 1xf32)
        clip_2 = paddle._C_ops.clip(subtract_2, full_10, full_11)

        # pd_op.prod: (8x1x4116xf32) <- (8x1x4116x2xf32, 1xi64)
        prod_2 = paddle._C_ops.prod(clip_2, full_int_array_0, False, False)

        # pd_op.add: (8x1x4116xf32) <- (8x1x1xf32, 8x1x4116xf32)
        add_1 = paddle._C_ops.add(prod_1, prod_2)

        # pd_op.subtract: (8x1x4116xf32) <- (8x1x4116xf32, 8x1x4116xf32)
        subtract_3 = paddle._C_ops.subtract(add_1, prod_0)

        # pd_op.scale: (8x1x4116xf32) <- (8x1x4116xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(subtract_3, full_4, float("1e-09"), True)

        # pd_op.divide: (8x1x4116xf32) <- (8x1x4116xf32, 8x1x4116xf32)
        divide_0 = paddle._C_ops.divide(prod_0, scale_1)

        # pd_op.multiply: (8x1x4116xf32) <- (8x1x4116xf32, 8x1x4116xf32)
        multiply_1 = paddle._C_ops.multiply(divide_0, data_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [-2]

        # pd_op.max: (8x4116xf32) <- (8x1x4116xf32, 1xi64)
        max_0 = paddle._C_ops.max(multiply_1, full_int_array_8, False)

        # pd_op.unsqueeze: (8x4116x1xf32) <- (8x4116xf32, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(max_0, full_int_array_0)

        # pd_op.multiply: (8x4116x4xf32) <- (8x4116x4xf32, 8x4116x1xf32)
        multiply_0 = paddle._C_ops.multiply(index_select_0, unsqueeze_3)
        return where_0, reshape_0, multiply_0
