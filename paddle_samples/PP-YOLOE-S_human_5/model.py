import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4, data_5):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [2]

        # pd_op.unsqueeze: (2x22x1x4xf32) <- (2x22x4xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(data_4, full_int_array_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.unsqueeze: (2x1x4116x4xf32) <- (2x4116x4xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(data_1, full_int_array_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.slice: (2x22x1x2xf32) <- (2x22x1x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            unsqueeze_1, [3], full_int_array_2, full_int_array_0, [1], []
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [2147483647]

        # pd_op.slice: (2x22x1x2xf32) <- (2x22x1x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            unsqueeze_1, [3], full_int_array_0, full_int_array_3, [1], []
        )

        # pd_op.slice: (2x1x4116x2xf32) <- (2x1x4116x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            unsqueeze_2, [3], full_int_array_2, full_int_array_0, [1], []
        )

        # pd_op.slice: (2x1x4116x2xf32) <- (2x1x4116x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            unsqueeze_2, [3], full_int_array_0, full_int_array_3, [1], []
        )

        # pd_op.maximum: (2x22x4116x2xf32) <- (2x22x1x2xf32, 2x1x4116x2xf32)
        maximum_0 = paddle._C_ops.maximum(slice_0, slice_2)

        # pd_op.minimum: (2x22x4116x2xf32) <- (2x22x1x2xf32, 2x1x4116x2xf32)
        minimum_0 = paddle._C_ops.minimum(slice_1, slice_3)

        # pd_op.subtract: (2x22x4116x2xf32) <- (2x22x4116x2xf32, 2x22x4116x2xf32)
        subtract_0 = paddle._C_ops.subtract(minimum_0, maximum_0)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (2x22x4116x2xf32) <- (2x22x4116x2xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(subtract_0, full_0, full_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [-1]

        # pd_op.prod: (2x22x4116xf32) <- (2x22x4116x2xf32, 1xi64)
        prod_0 = paddle._C_ops.prod(clip_0, full_int_array_4, False, False)

        # pd_op.subtract: (2x22x1x2xf32) <- (2x22x1x2xf32, 2x22x1x2xf32)
        subtract_1 = paddle._C_ops.subtract(slice_1, slice_0)

        # pd_op.clip: (2x22x1x2xf32) <- (2x22x1x2xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(subtract_1, full_0, full_1)

        # pd_op.prod: (2x22x1xf32) <- (2x22x1x2xf32, 1xi64)
        prod_1 = paddle._C_ops.prod(clip_1, full_int_array_4, False, False)

        # pd_op.subtract: (2x1x4116x2xf32) <- (2x1x4116x2xf32, 2x1x4116x2xf32)
        subtract_2 = paddle._C_ops.subtract(slice_3, slice_2)

        # pd_op.clip: (2x1x4116x2xf32) <- (2x1x4116x2xf32, 1xf32, 1xf32)
        clip_2 = paddle._C_ops.clip(subtract_2, full_0, full_1)

        # pd_op.prod: (2x1x4116xf32) <- (2x1x4116x2xf32, 1xi64)
        prod_2 = paddle._C_ops.prod(clip_2, full_int_array_4, False, False)

        # pd_op.add: (2x22x4116xf32) <- (2x22x1xf32, 2x1x4116xf32)
        add_0 = paddle._C_ops.add(prod_1, prod_2)

        # pd_op.subtract: (2x22x4116xf32) <- (2x22x4116xf32, 2x22x4116xf32)
        subtract_3 = paddle._C_ops.subtract(add_0, prod_0)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (2x22x4116xf32) <- (2x22x4116xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(subtract_3, full_2, float("1e-09"), True)

        # pd_op.divide: (2x22x4116xf32) <- (2x22x4116xf32, 2x22x4116xf32)
        divide_0 = paddle._C_ops.divide(prod_0, scale_0)

        # pd_op.transpose: (2x1x4116xf32) <- (2x4116x1xf32)
        transpose_0 = paddle._C_ops.transpose(data_0, [0, 2, 1])

        # pd_op.full: (1xf64) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("2"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (2xi32) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_3, full_4, full_5, dtype="int32")

        # pd_op.unsqueeze: (2x1xi32) <- (2xi32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(arange_0, full_int_array_4)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [1, 22]

        # pd_op.tile: (2x22xi32) <- (2x1xi32, 2xi64)
        tile_0 = paddle._C_ops.tile(unsqueeze_0, full_int_array_5)

        # pd_op.squeeze: (2x22xi32) <- (2x22x1xi32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(data_3, full_int_array_4)

        # builtin.combine: ([2x22xi32, 2x22xi32]) <- (2x22xi32, 2x22xi32)
        combine_0 = [tile_0, squeeze_0]

        # pd_op.stack: (2x22x2xi32) <- ([2x22xi32, 2x22xi32])
        stack_0 = paddle._C_ops.stack(combine_0, -1)

        # pd_op.gather_nd: (2x22x4116xf32) <- (2x1x4116xf32, 2x22x2xi32)
        gather_nd_0 = paddle._C_ops.gather_nd(transpose_0, stack_0)

        # pd_op.pow: (2x22x4116xf32) <- (2x22x4116xf32)
        pow_0 = paddle._C_ops.pow(gather_nd_0, float("1"))

        # pd_op.pow: (2x22x4116xf32) <- (2x22x4116xf32)
        pow_1 = paddle._C_ops.pow(divide_0, float("6"))

        # pd_op.multiply: (2x22x4116xf32) <- (2x22x4116xf32, 2x22x4116xf32)
        multiply_1 = paddle._C_ops.multiply(pow_0, pow_1)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_6 = [0, 1]

        # pd_op.unsqueeze: (1x1x4116x2xf32) <- (4116x2xf32, 2xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(data_2, full_int_array_6)

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("3"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split_with_num: ([1x1x4116x1xf32, 1x1x4116x1xf32]) <- (1x1x4116x2xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(unsqueeze_3, 2, full_6)

        # builtin.split: (1x1x4116x1xf32, 1x1x4116x1xf32) <- ([1x1x4116x1xf32, 1x1x4116x1xf32])
        (
            split_0,
            split_1,
        ) = split_with_num_0

        # pd_op.split_with_num: ([2x22x1x1xf32, 2x22x1x1xf32, 2x22x1x1xf32, 2x22x1x1xf32]) <- (2x22x1x4xf32, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(unsqueeze_1, 4, full_6)

        # builtin.split: (2x22x1x1xf32, 2x22x1x1xf32, 2x22x1x1xf32, 2x22x1x1xf32) <- ([2x22x1x1xf32, 2x22x1x1xf32, 2x22x1x1xf32, 2x22x1x1xf32])
        (
            split_2,
            split_3,
            split_4,
            split_5,
        ) = split_with_num_1

        # pd_op.subtract: (2x22x4116x1xf32) <- (1x1x4116x1xf32, 2x22x1x1xf32)
        subtract_4 = paddle._C_ops.subtract(split_0, split_2)

        # pd_op.subtract: (2x22x4116x1xf32) <- (1x1x4116x1xf32, 2x22x1x1xf32)
        subtract_5 = paddle._C_ops.subtract(split_1, split_3)

        # pd_op.subtract: (2x22x4116x1xf32) <- (2x22x1x1xf32, 1x1x4116x1xf32)
        subtract_6 = paddle._C_ops.subtract(split_4, split_0)

        # pd_op.subtract: (2x22x4116x1xf32) <- (2x22x1x1xf32, 1x1x4116x1xf32)
        subtract_7 = paddle._C_ops.subtract(split_5, split_1)

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([2x22x4116x1xf32, 2x22x4116x1xf32, 2x22x4116x1xf32, 2x22x4116x1xf32]) <- (2x22x4116x1xf32, 2x22x4116x1xf32, 2x22x4116x1xf32, 2x22x4116x1xf32)
        combine_1 = [subtract_4, subtract_5, subtract_6, subtract_7]

        # pd_op.concat: (2x22x4116x4xf32) <- ([2x22x4116x1xf32, 2x22x4116x1xf32, 2x22x4116x1xf32, 2x22x4116x1xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_1, full_7)

        # pd_op.min: (2x22x4116xf32) <- (2x22x4116x4xf32, 1xi64)
        min_0 = paddle._C_ops.min(concat_0, full_int_array_4, False)

        # pd_op.full: (xf32) <- ()
        full_8 = paddle._C_ops.full(
            [],
            float("1e-09"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.greater_than: (2x22x4116xb) <- (2x22x4116xf32, xf32)
        greater_than_1 = paddle._C_ops.greater_than(min_0, full_8)

        # pd_op.cast: (2x22x4116xf32) <- (2x22x4116xb)
        cast_0 = paddle._C_ops.cast(greater_than_1, paddle.float32)

        # pd_op.multiply: (2x22x4116xf32) <- (2x22x4116xf32, 2x22x4116xf32)
        multiply_2 = paddle._C_ops.multiply(multiply_1, cast_0)

        # pd_op.full: (1xi32) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("13"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.topk: (2x22x13xf32, 2x22x13xi64) <- (2x22x4116xf32, 1xi32)
        topk_0, topk_1 = (lambda x, f: f(x))(
            paddle._C_ops.topk(multiply_2, full_9, -1, True, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.full: (1xi32) <- ()
        full_10 = paddle._C_ops.full(
            [1], float("4116"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.one_hot: (2x22x13x4116xf32) <- (2x22x13xi64, 1xi32)
        one_hot_0 = paddle._C_ops.one_hot(
            topk_1 % paddle.cast(full_10, topk_1.dtype), full_10
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [-2]

        # pd_op.sum: (2x22x4116xf32) <- (2x22x13x4116xf32, 1xi64)
        sum_1 = paddle._C_ops.sum(one_hot_0, full_int_array_7, None, False)

        # pd_op.multiply: (2x22x4116xf32) <- (2x22x4116xf32, 2x22x1xf32)
        multiply_3 = paddle._C_ops.multiply(sum_1, data_5)

        # pd_op.multiply: (2x22x4116xf32) <- (2x22x4116xf32, 2x22x4116xf32)
        multiply_4 = paddle._C_ops.multiply(multiply_3, cast_0)

        # pd_op.multiply: (2x22x4116xf32) <- (2x22x4116xf32, 2x22x1xf32)
        multiply_0 = paddle._C_ops.multiply(multiply_4, data_5)

        # pd_op.sum: (2x4116xf32) <- (2x22x4116xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(multiply_0, full_int_array_7, None, False)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_8 = []

        # pd_op.max: (xf32) <- (2x4116xf32, 0xi64)
        max_0 = paddle._C_ops.max(sum_0, full_int_array_8, False)

        # pd_op.full: (xf32) <- ()
        full_11 = paddle._C_ops.full(
            [], float("1"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (xb) <- (xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(max_0, full_11)
        return greater_than_0, sum_0, divide_0, multiply_0, unsqueeze_0, multiply_1
