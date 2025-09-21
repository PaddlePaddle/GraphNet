import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4, data_5):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [2]

        # pd_op.unsqueeze: (8x2x1x4xf32) <- (8x2x4xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(data_4, full_int_array_0)
        del data_4

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.unsqueeze: (8x1x6885x4xf32) <- (8x6885x4xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(data_1, full_int_array_1)
        del data_1, full_int_array_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.slice: (8x2x1x2xf32) <- (8x2x1x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            unsqueeze_0, [3], full_int_array_2, full_int_array_0, [1], []
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [2147483647]

        # pd_op.slice: (8x2x1x2xf32) <- (8x2x1x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            unsqueeze_0, [3], full_int_array_0, full_int_array_3, [1], []
        )

        # pd_op.slice: (8x1x6885x2xf32) <- (8x1x6885x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            unsqueeze_1, [3], full_int_array_2, full_int_array_0, [1], []
        )
        del full_int_array_2

        # pd_op.slice: (8x1x6885x2xf32) <- (8x1x6885x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            unsqueeze_1, [3], full_int_array_0, full_int_array_3, [1], []
        )
        del full_int_array_0, full_int_array_3, unsqueeze_1

        # pd_op.maximum: (8x2x6885x2xf32) <- (8x2x1x2xf32, 8x1x6885x2xf32)
        maximum_0 = paddle._C_ops.maximum(slice_0, slice_2)

        # pd_op.minimum: (8x2x6885x2xf32) <- (8x2x1x2xf32, 8x1x6885x2xf32)
        minimum_0 = paddle._C_ops.minimum(slice_1, slice_3)

        # pd_op.subtract: (8x2x6885x2xf32) <- (8x2x6885x2xf32, 8x2x6885x2xf32)
        subtract_0 = paddle._C_ops.subtract(minimum_0, maximum_0)
        del maximum_0, minimum_0

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (8x2x6885x2xf32) <- (8x2x6885x2xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(subtract_0, full_0, full_1)
        del subtract_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [-1]

        # pd_op.prod: (8x2x6885xf32) <- (8x2x6885x2xf32, 1xi64)
        prod_0 = paddle._C_ops.prod(clip_0, full_int_array_4, False, False)
        del clip_0

        # pd_op.subtract: (8x2x1x2xf32) <- (8x2x1x2xf32, 8x2x1x2xf32)
        subtract_1 = paddle._C_ops.subtract(slice_1, slice_0)
        del slice_0, slice_1

        # pd_op.clip: (8x2x1x2xf32) <- (8x2x1x2xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(subtract_1, full_0, full_1)
        del subtract_1

        # pd_op.prod: (8x2x1xf32) <- (8x2x1x2xf32, 1xi64)
        prod_1 = paddle._C_ops.prod(clip_1, full_int_array_4, False, False)
        del clip_1

        # pd_op.subtract: (8x1x6885x2xf32) <- (8x1x6885x2xf32, 8x1x6885x2xf32)
        subtract_2 = paddle._C_ops.subtract(slice_3, slice_2)
        del slice_2, slice_3

        # pd_op.clip: (8x1x6885x2xf32) <- (8x1x6885x2xf32, 1xf32, 1xf32)
        clip_2 = paddle._C_ops.clip(subtract_2, full_0, full_1)
        del full_0, full_1, subtract_2

        # pd_op.prod: (8x1x6885xf32) <- (8x1x6885x2xf32, 1xi64)
        prod_2 = paddle._C_ops.prod(clip_2, full_int_array_4, False, False)
        del clip_2

        # pd_op.add: (8x2x6885xf32) <- (8x2x1xf32, 8x1x6885xf32)
        add_0 = paddle._C_ops.add(prod_1, prod_2)
        del prod_1, prod_2

        # pd_op.subtract: (8x2x6885xf32) <- (8x2x6885xf32, 8x2x6885xf32)
        subtract_3 = paddle._C_ops.subtract(add_0, prod_0)
        del add_0

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (8x2x6885xf32) <- (8x2x6885xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(subtract_3, full_2, float("1e-09"), True)
        del full_2, subtract_3

        # pd_op.divide: (8x2x6885xf32) <- (8x2x6885xf32, 8x2x6885xf32)
        divide_0 = paddle._C_ops.divide(prod_0, scale_0)
        del prod_0, scale_0

        # pd_op.transpose: (8x4x6885xf32) <- (8x6885x4xf32)
        transpose_0 = paddle._C_ops.transpose(data_0, [0, 2, 1])
        del data_0

        # pd_op.full: (1xf64) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("8"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (8xi32) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_3, full_4, full_5, dtype="int32")
        del full_3, full_4, full_5

        # pd_op.unsqueeze: (8x1xi32) <- (8xi32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(arange_0, full_int_array_4)
        del arange_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [1, 2]

        # pd_op.tile: (8x2xi32) <- (8x1xi32, 2xi64)
        tile_0 = paddle._C_ops.tile(unsqueeze_2, full_int_array_5)
        del full_int_array_5

        # pd_op.squeeze: (8x2xi32) <- (8x2x1xi32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(data_3, full_int_array_4)
        del data_3

        # builtin.combine: ([8x2xi32, 8x2xi32]) <- (8x2xi32, 8x2xi32)
        combine_0 = [tile_0, squeeze_0]
        del squeeze_0, tile_0

        # pd_op.stack: (8x2x2xi32) <- ([8x2xi32, 8x2xi32])
        stack_0 = paddle._C_ops.stack(combine_0, -1)
        del combine_0

        # pd_op.gather_nd: (8x2x6885xf32) <- (8x4x6885xf32, 8x2x2xi32)
        gather_nd_0 = paddle._C_ops.gather_nd(transpose_0, stack_0)
        del stack_0, transpose_0

        # pd_op.pow: (8x2x6885xf32) <- (8x2x6885xf32)
        pow_0 = paddle._C_ops.pow(gather_nd_0, float("1"))
        del gather_nd_0

        # pd_op.pow: (8x2x6885xf32) <- (8x2x6885xf32)
        pow_1 = paddle._C_ops.pow(divide_0, float("6"))

        # pd_op.multiply: (8x2x6885xf32) <- (8x2x6885xf32, 8x2x6885xf32)
        multiply_0 = paddle._C_ops.multiply(pow_0, pow_1)
        del pow_0, pow_1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_6 = [0, 1]

        # pd_op.unsqueeze: (1x1x6885x2xf32) <- (6885x2xf32, 2xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(data_2, full_int_array_6)
        del data_2, full_int_array_6

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("3"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split_with_num: ([1x1x6885x1xf32, 1x1x6885x1xf32]) <- (1x1x6885x2xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(unsqueeze_3, 2, full_6)
        del unsqueeze_3

        # builtin.split: (1x1x6885x1xf32, 1x1x6885x1xf32) <- ([1x1x6885x1xf32, 1x1x6885x1xf32])
        (
            split_0,
            split_1,
        ) = split_with_num_0
        del split_with_num_0

        # pd_op.split_with_num: ([8x2x1x1xf32, 8x2x1x1xf32, 8x2x1x1xf32, 8x2x1x1xf32]) <- (8x2x1x4xf32, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(unsqueeze_0, 4, full_6)
        del full_6, unsqueeze_0

        # builtin.split: (8x2x1x1xf32, 8x2x1x1xf32, 8x2x1x1xf32, 8x2x1x1xf32) <- ([8x2x1x1xf32, 8x2x1x1xf32, 8x2x1x1xf32, 8x2x1x1xf32])
        (
            split_2,
            split_3,
            split_4,
            split_5,
        ) = split_with_num_1
        del split_with_num_1

        # pd_op.subtract: (8x2x6885x1xf32) <- (1x1x6885x1xf32, 8x2x1x1xf32)
        subtract_4 = paddle._C_ops.subtract(split_0, split_2)
        del split_2

        # pd_op.subtract: (8x2x6885x1xf32) <- (1x1x6885x1xf32, 8x2x1x1xf32)
        subtract_5 = paddle._C_ops.subtract(split_1, split_3)
        del split_3

        # pd_op.subtract: (8x2x6885x1xf32) <- (8x2x1x1xf32, 1x1x6885x1xf32)
        subtract_6 = paddle._C_ops.subtract(split_4, split_0)
        del split_0, split_4

        # pd_op.subtract: (8x2x6885x1xf32) <- (8x2x1x1xf32, 1x1x6885x1xf32)
        subtract_7 = paddle._C_ops.subtract(split_5, split_1)
        del split_1, split_5

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([8x2x6885x1xf32, 8x2x6885x1xf32, 8x2x6885x1xf32, 8x2x6885x1xf32]) <- (8x2x6885x1xf32, 8x2x6885x1xf32, 8x2x6885x1xf32, 8x2x6885x1xf32)
        combine_1 = [subtract_4, subtract_5, subtract_6, subtract_7]
        del subtract_4, subtract_5, subtract_6, subtract_7

        # pd_op.concat: (8x2x6885x4xf32) <- ([8x2x6885x1xf32, 8x2x6885x1xf32, 8x2x6885x1xf32, 8x2x6885x1xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_1, full_7)
        del combine_1, full_7

        # pd_op.min: (8x2x6885xf32) <- (8x2x6885x4xf32, 1xi64)
        min_0 = paddle._C_ops.min(concat_0, full_int_array_4, False)
        del concat_0, full_int_array_4

        # pd_op.full: (xf32) <- ()
        full_8 = paddle._C_ops.full(
            [],
            float("1e-09"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.greater_than: (8x2x6885xb) <- (8x2x6885xf32, xf32)
        greater_than_1 = paddle._C_ops.greater_than(min_0, full_8)
        del full_8, min_0

        # pd_op.cast: (8x2x6885xf32) <- (8x2x6885xb)
        cast_0 = paddle._C_ops.cast(greater_than_1, paddle.float32)
        del greater_than_1

        # pd_op.multiply: (8x2x6885xf32) <- (8x2x6885xf32, 8x2x6885xf32)
        multiply_1 = paddle._C_ops.multiply(multiply_0, cast_0)

        # pd_op.full: (1xi32) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("13"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.topk: (8x2x13xf32, 8x2x13xi64) <- (8x2x6885xf32, 1xi32)
        topk_0, topk_1 = (lambda x, f: f(x))(
            paddle._C_ops.topk(multiply_1, full_9, -1, True, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del full_9, multiply_1

        # pd_op.full: (1xi32) <- ()
        full_10 = paddle._C_ops.full(
            [1], float("6885"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.one_hot: (8x2x13x6885xf32) <- (8x2x13xi64, 1xi32)
        one_hot_0 = paddle._C_ops.one_hot(
            topk_1 % paddle.cast(full_10, topk_1.dtype), full_10
        )
        del full_10, topk_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [-2]

        # pd_op.sum: (8x2x6885xf32) <- (8x2x13x6885xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(one_hot_0, full_int_array_7, None, False)
        del one_hot_0

        # pd_op.multiply: (8x2x6885xf32) <- (8x2x6885xf32, 8x2x1xf32)
        multiply_2 = paddle._C_ops.multiply(sum_0, data_5)
        del sum_0

        # pd_op.multiply: (8x2x6885xf32) <- (8x2x6885xf32, 8x2x6885xf32)
        multiply_3 = paddle._C_ops.multiply(multiply_2, cast_0)
        del cast_0, multiply_2

        # pd_op.multiply: (8x2x6885xf32) <- (8x2x6885xf32, 8x2x1xf32)
        multiply_4 = paddle._C_ops.multiply(multiply_3, data_5)
        del data_5, multiply_3

        # pd_op.sum: (8x6885xf32) <- (8x2x6885xf32, 1xi64)
        sum_1 = paddle._C_ops.sum(multiply_4, full_int_array_7, None, False)
        del full_int_array_7

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_8 = []

        # pd_op.max: (xf32) <- (8x6885xf32, 0xi64)
        max_0 = paddle._C_ops.max(sum_1, full_int_array_8, False)
        del full_int_array_8

        # pd_op.full: (xf32) <- ()
        full_11 = paddle._C_ops.full(
            [], float("1"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (xb) <- (xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(max_0, full_11)
        del divide_0, full_11, max_0, multiply_0, multiply_4, sum_1, unsqueeze_2

        return greater_than_0
