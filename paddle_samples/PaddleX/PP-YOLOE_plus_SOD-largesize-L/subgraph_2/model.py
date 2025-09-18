import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7):
        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_0 = paddle._C_ops.equal(data_0, full_0)

        # pd_op.cast: (xi64) <- (xb)
        cast_0 = paddle._C_ops.cast(equal_0, paddle.int64)
        del equal_0

        # pd_op.not_equal: (xb) <- (xi64, xi64)
        not_equal_0 = paddle._C_ops.not_equal(cast_0, full_0)
        del cast_0

        # pd_op.cast: (xi64) <- (xb)
        cast_1 = paddle._C_ops.cast(not_equal_0, paddle.int64)
        del not_equal_0

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_1 = paddle._C_ops.equal(cast_1, full_0)
        del cast_1, full_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [2]

        # pd_op.unsqueeze: (1x-1x1x4xf32) <- (1x-1x4xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(data_6, full_int_array_0)
        del data_6

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.unsqueeze: (1x1x-1x4xf32) <- (1x-1x4xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(data_2, full_int_array_1)
        del data_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.slice: (1x-1x1x2xf32) <- (1x-1x1x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            unsqueeze_0, [3], full_int_array_2, full_int_array_0, [1], []
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [2147483647]

        # pd_op.slice: (1x-1x1x2xf32) <- (1x-1x1x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            unsqueeze_0, [3], full_int_array_0, full_int_array_3, [1], []
        )

        # pd_op.slice: (1x1x-1x2xf32) <- (1x1x-1x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            unsqueeze_1, [3], full_int_array_2, full_int_array_0, [1], []
        )
        del full_int_array_2

        # pd_op.slice: (1x1x-1x2xf32) <- (1x1x-1x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            unsqueeze_1, [3], full_int_array_0, full_int_array_3, [1], []
        )
        del full_int_array_3, unsqueeze_1

        # pd_op.maximum: (1x-1x-1x2xf32) <- (1x-1x1x2xf32, 1x1x-1x2xf32)
        maximum_0 = paddle._C_ops.maximum(slice_0, slice_2)

        # pd_op.minimum: (1x-1x-1x2xf32) <- (1x-1x1x2xf32, 1x1x-1x2xf32)
        minimum_0 = paddle._C_ops.minimum(slice_1, slice_3)

        # pd_op.subtract: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1x-1x-1x2xf32)
        subtract_0 = paddle._C_ops.subtract(minimum_0, maximum_0)
        del maximum_0, minimum_0

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(subtract_0, full_1, full_2)
        del subtract_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [-1]

        # pd_op.prod: (1x-1x-1xf32) <- (1x-1x-1x2xf32, 1xi64)
        prod_0 = paddle._C_ops.prod(clip_0, full_int_array_4, False, False)
        del clip_0

        # pd_op.subtract: (1x-1x1x2xf32) <- (1x-1x1x2xf32, 1x-1x1x2xf32)
        subtract_1 = paddle._C_ops.subtract(slice_1, slice_0)
        del slice_0, slice_1

        # pd_op.clip: (1x-1x1x2xf32) <- (1x-1x1x2xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(subtract_1, full_1, full_2)
        del subtract_1

        # pd_op.prod: (1x-1x1xf32) <- (1x-1x1x2xf32, 1xi64)
        prod_1 = paddle._C_ops.prod(clip_1, full_int_array_4, False, False)
        del clip_1

        # pd_op.subtract: (1x1x-1x2xf32) <- (1x1x-1x2xf32, 1x1x-1x2xf32)
        subtract_2 = paddle._C_ops.subtract(slice_3, slice_2)
        del slice_2, slice_3

        # pd_op.clip: (1x1x-1x2xf32) <- (1x1x-1x2xf32, 1xf32, 1xf32)
        clip_2 = paddle._C_ops.clip(subtract_2, full_1, full_2)
        del full_2, subtract_2

        # pd_op.prod: (1x1x-1xf32) <- (1x1x-1x2xf32, 1xi64)
        prod_2 = paddle._C_ops.prod(clip_2, full_int_array_4, False, False)
        del clip_2

        # pd_op.add: (1x-1x-1xf32) <- (1x-1x1xf32, 1x1x-1xf32)
        add_0 = paddle._C_ops.add(prod_1, prod_2)
        del prod_1, prod_2

        # pd_op.subtract: (1x-1x-1xf32) <- (1x-1x-1xf32, 1x-1x-1xf32)
        subtract_3 = paddle._C_ops.subtract(add_0, prod_0)
        del add_0

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x-1x-1xf32) <- (1x-1x-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(subtract_3, full_3, float("1e-09"), True)
        del subtract_3

        # pd_op.divide: (1x-1x-1xf32) <- (1x-1x-1xf32, 1x-1x-1xf32)
        divide_0 = paddle._C_ops.divide(prod_0, scale_0)
        del prod_0, scale_0

        # pd_op.transpose: (1x10x-1xf32) <- (1x-1x10xf32)
        transpose_0 = paddle._C_ops.transpose(data_1, [0, 2, 1])
        del data_1

        # pd_op.full: (1xf64) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (1xi32) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_4, full_5, full_5, dtype="int32")
        del full_4, full_5

        # pd_op.unsqueeze: (1x1xi32) <- (1xi32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(arange_0, full_int_array_4)
        del arange_0

        # pd_op.full: (xi64) <- ()
        full_6 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_0 = [full_6, data_0]
        del data_0, full_6

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.tile: (1x-1xi32) <- (1x1xi32, 2xi64)
        tile_0 = paddle._C_ops.tile(unsqueeze_2, stack_0)
        del stack_0

        # pd_op.squeeze: (1x-1xi32) <- (1x-1x1xi32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(data_5, full_int_array_4)
        del data_5

        # builtin.combine: ([1x-1xi32, 1x-1xi32]) <- (1x-1xi32, 1x-1xi32)
        combine_1 = [tile_0, squeeze_0]
        del squeeze_0, tile_0

        # pd_op.stack: (1x-1x2xi32) <- ([1x-1xi32, 1x-1xi32])
        stack_1 = paddle._C_ops.stack(combine_1, -1)
        del combine_1

        # pd_op.gather_nd: (1x-1x-1xf32) <- (1x10x-1xf32, 1x-1x2xi32)
        gather_nd_0 = paddle._C_ops.gather_nd(transpose_0, stack_1)
        del stack_1, transpose_0

        # pd_op.pow: (1x-1x-1xf32) <- (1x-1x-1xf32)
        pow_0 = paddle._C_ops.pow(gather_nd_0, float("1"))
        del gather_nd_0

        # pd_op.pow: (1x-1x-1xf32) <- (1x-1x-1xf32)
        pow_1 = paddle._C_ops.pow(divide_0, float("6"))

        # pd_op.multiply: (1x-1x-1xf32) <- (1x-1x-1xf32, 1x-1x-1xf32)
        multiply_0 = paddle._C_ops.multiply(pow_0, pow_1)
        del pow_0, pow_1

        # pd_op.multiply: (1x-1x-1xf32) <- (1x-1x-1xf32, 1x-1x1xf32)
        multiply_1 = paddle._C_ops.multiply(multiply_0, data_7)
        del multiply_0

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(data_4, full_3, float("0"), True)
        del data_4, full_3

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [0, 1]

        # pd_op.unsqueeze: (1x1x-1x2xf32) <- (-1x2xf32, 2xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(data_3, full_int_array_5)
        del data_3

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("3"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split_with_num: ([1x1x-1x1xf32, 1x1x-1x1xf32]) <- (1x1x-1x2xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(unsqueeze_3, 2, full_7)
        del unsqueeze_3

        # builtin.split: (1x1x-1x1xf32, 1x1x-1x1xf32) <- ([1x1x-1x1xf32, 1x1x-1x1xf32])
        (
            split_0,
            split_1,
        ) = split_with_num_0
        del split_with_num_0

        # pd_op.split_with_num: ([1x-1x1x1xf32, 1x-1x1x1xf32, 1x-1x1x1xf32, 1x-1x1x1xf32]) <- (1x-1x1x4xf32, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(unsqueeze_0, 4, full_7)
        del full_7, unsqueeze_0

        # builtin.split: (1x-1x1x1xf32, 1x-1x1x1xf32, 1x-1x1x1xf32, 1x-1x1x1xf32) <- ([1x-1x1x1xf32, 1x-1x1x1xf32, 1x-1x1x1xf32, 1x-1x1x1xf32])
        (
            split_2,
            split_3,
            split_4,
            split_5,
        ) = split_with_num_1
        del split_with_num_1

        # pd_op.subtract: (1x-1x-1x1xf32) <- (1x1x-1x1xf32, 1x-1x1x1xf32)
        subtract_4 = paddle._C_ops.subtract(split_0, split_2)

        # pd_op.subtract: (1x-1x-1x1xf32) <- (1x1x-1x1xf32, 1x-1x1x1xf32)
        subtract_5 = paddle._C_ops.subtract(split_1, split_3)

        # pd_op.subtract: (1x-1x-1x1xf32) <- (1x-1x1x1xf32, 1x1x-1x1xf32)
        subtract_6 = paddle._C_ops.subtract(split_4, split_0)

        # pd_op.subtract: (1x-1x-1x1xf32) <- (1x-1x1x1xf32, 1x1x-1x1xf32)
        subtract_7 = paddle._C_ops.subtract(split_5, split_1)

        # pd_op.full: (1xi32) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([1x-1x-1x1xf32, 1x-1x-1x1xf32, 1x-1x-1x1xf32, 1x-1x-1x1xf32]) <- (1x-1x-1x1xf32, 1x-1x-1x1xf32, 1x-1x-1x1xf32, 1x-1x-1x1xf32)
        combine_2 = [subtract_4, subtract_5, subtract_6, subtract_7]
        del subtract_4, subtract_5, subtract_6, subtract_7

        # pd_op.concat: (1x-1x-1x4xf32) <- ([1x-1x-1x1xf32, 1x-1x-1x1xf32, 1x-1x-1x1xf32, 1x-1x-1x1xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_2, full_8)
        del combine_2

        # pd_op.min: (1x-1x-1xf32) <- (1x-1x-1x4xf32, 1xi64)
        min_0 = paddle._C_ops.min(concat_0, full_int_array_4, False)
        del concat_0

        # pd_op.full: (xf32) <- ()
        full_9 = paddle._C_ops.full(
            [],
            float("1e-09"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.greater_than: (1x-1x-1xb) <- (1x-1x-1xf32, xf32)
        greater_than_1 = paddle._C_ops.greater_than(min_0, full_9)
        del min_0

        # pd_op.unsqueeze: (1x1x-1x1xf32) <- (-1x1xf32, 2xi64)
        unsqueeze_4 = paddle._C_ops.unsqueeze(scale_1, full_int_array_5)
        del full_int_array_5, scale_1

        # pd_op.add: (1x-1x1x1xf32) <- (1x-1x1x1xf32, 1x-1x1x1xf32)
        add_1 = paddle._C_ops.add(split_2, split_4)
        del split_2, split_4

        # pd_op.full: (1xf32) <- ()
        full_10 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x-1x1x1xf32) <- (1x-1x1x1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(add_1, full_10, float("0"), True)
        del add_1

        # pd_op.add: (1x-1x1x1xf32) <- (1x-1x1x1xf32, 1x-1x1x1xf32)
        add_2 = paddle._C_ops.add(split_3, split_5)
        del split_3, split_5

        # pd_op.scale: (1x-1x1x1xf32) <- (1x-1x1x1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(add_2, full_10, float("0"), True)
        del add_2, full_10

        # pd_op.subtract: (1x-1x-1x1xf32) <- (1x-1x1x1xf32, 1x1x-1x1xf32)
        subtract_8 = paddle._C_ops.subtract(scale_2, unsqueeze_4)

        # pd_op.subtract: (1x-1x-1x1xf32) <- (1x1x-1x1xf32, 1x-1x-1x1xf32)
        subtract_9 = paddle._C_ops.subtract(split_0, subtract_8)
        del subtract_8

        # pd_op.subtract: (1x-1x-1x1xf32) <- (1x-1x1x1xf32, 1x1x-1x1xf32)
        subtract_10 = paddle._C_ops.subtract(scale_3, unsqueeze_4)

        # pd_op.subtract: (1x-1x-1x1xf32) <- (1x1x-1x1xf32, 1x-1x-1x1xf32)
        subtract_11 = paddle._C_ops.subtract(split_1, subtract_10)
        del subtract_10

        # pd_op.add: (1x-1x-1x1xf32) <- (1x-1x1x1xf32, 1x1x-1x1xf32)
        add_3 = paddle._C_ops.add(scale_2, unsqueeze_4)
        del scale_2

        # pd_op.subtract: (1x-1x-1x1xf32) <- (1x-1x-1x1xf32, 1x1x-1x1xf32)
        subtract_12 = paddle._C_ops.subtract(add_3, split_0)
        del add_3, split_0

        # pd_op.add: (1x-1x-1x1xf32) <- (1x-1x1x1xf32, 1x1x-1x1xf32)
        add_4 = paddle._C_ops.add(scale_3, unsqueeze_4)
        del scale_3, unsqueeze_4

        # pd_op.subtract: (1x-1x-1x1xf32) <- (1x-1x-1x1xf32, 1x1x-1x1xf32)
        subtract_13 = paddle._C_ops.subtract(add_4, split_1)
        del add_4, split_1

        # builtin.combine: ([1x-1x-1x1xf32, 1x-1x-1x1xf32, 1x-1x-1x1xf32, 1x-1x-1x1xf32]) <- (1x-1x-1x1xf32, 1x-1x-1x1xf32, 1x-1x-1x1xf32, 1x-1x-1x1xf32)
        combine_3 = [subtract_9, subtract_11, subtract_12, subtract_13]
        del subtract_11, subtract_12, subtract_13, subtract_9

        # pd_op.concat: (1x-1x-1x4xf32) <- ([1x-1x-1x1xf32, 1x-1x-1x1xf32, 1x-1x-1x1xf32, 1x-1x-1x1xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_3, full_8)
        del combine_3, full_8

        # pd_op.min: (1x-1x-1xf32) <- (1x-1x-1x4xf32, 1xi64)
        min_1 = paddle._C_ops.min(concat_1, full_int_array_4, False)
        del concat_1

        # pd_op.greater_than: (1x-1x-1xb) <- (1x-1x-1xf32, xf32)
        greater_than_2 = paddle._C_ops.greater_than(min_1, full_9)
        del full_9, min_1

        # pd_op.cast: (1x-1x-1xf32) <- (1x-1x-1xb)
        cast_2 = paddle._C_ops.cast(greater_than_1, paddle.float32)
        del greater_than_1

        # pd_op.cast: (1x-1x-1xf32) <- (1x-1x-1xb)
        cast_3 = paddle._C_ops.cast(greater_than_2, paddle.float32)
        del greater_than_2

        # pd_op.multiply: (1x-1x-1xf32) <- (1x-1x-1xf32, 1x-1x1xf32)
        multiply_2 = paddle._C_ops.multiply(cast_2, data_7)
        del cast_2

        # pd_op.multiply: (1x-1x-1xf32) <- (1x-1x-1xf32, 1x-1x1xf32)
        multiply_3 = paddle._C_ops.multiply(cast_3, data_7)
        del cast_3

        # pd_op.sum: (1x-1x1xf32) <- (1x-1x-1xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(multiply_2, full_int_array_4, None, True)
        del full_int_array_4

        # pd_op.full: (xf32) <- ()
        full_11 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.equal: (1x-1x1xb) <- (1x-1x1xf32, xf32)
        equal_2 = paddle._C_ops.equal(sum_0, full_11)
        del sum_0

        # pd_op.add: (1x-1x-1xf32) <- (1x-1x-1xf32, 1x-1x-1xf32)
        add_5 = paddle._C_ops.add(multiply_1, multiply_3)

        # pd_op.full_like: (1x-1x-1xf32) <- (1x-1x-1xf32, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            add_5, full_1, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.full_like: (1x-1x-1xf32) <- (1x-1x-1xf32, 1xf32)
        full_like_1 = paddle._C_ops.full_like(
            multiply_1,
            full_1,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_like: (1x-1x1xb) <- (1x-1x1xb, 1xf32)
        full_like_2 = paddle._C_ops.full_like(
            equal_2, full_1, paddle.bool, paddle.framework._current_expected_place()
        )
        del full_1

        # pd_op.cast: (1x-1x1xf32) <- (1x-1x1xb)
        cast_4 = paddle._C_ops.cast(full_like_2, paddle.float32)
        del full_like_2

        # pd_op.cast: (1x-1x1xf32) <- (1x-1x1xb)
        cast_5 = paddle._C_ops.cast(equal_2, paddle.float32)
        del equal_2

        # pd_op.add: (1x-1x-1xf32) <- (1x-1x-1xf32, 1x-1x-1xf32)
        add_6 = paddle._C_ops.add(full_like_0, full_like_1)
        del full_like_0, full_like_1

        # pd_op.add: (1x-1x-1xf32) <- (1x-1x-1xf32, 1x-1x1xf32)
        add_7 = paddle._C_ops.add(add_6, cast_4)
        del add_6, cast_4

        # pd_op.add: (1x-1x-1xf32) <- (1x-1x-1xf32, 1x-1x-1xf32)
        add_8 = paddle._C_ops.add(add_5, add_7)
        del add_5

        # pd_op.add: (1x-1x-1xf32) <- (1x-1x-1xf32, 1x-1x-1xf32)
        add_9 = paddle._C_ops.add(multiply_1, add_7)

        # pd_op.add: (1x-1x-1xf32) <- (1x-1x1xf32, 1x-1x-1xf32)
        add_10 = paddle._C_ops.add(cast_5, add_7)
        del add_7, cast_5

        # pd_op.cast: (1x-1x-1xb) <- (1x-1x-1xf32)
        cast_6 = paddle._C_ops.cast(add_10, paddle.bool)
        del add_10

        # pd_op.where: (1x-1x-1xf32) <- (1x-1x-1xb, 1x-1x-1xf32, 1x-1x-1xf32)
        where_0 = paddle._C_ops.where(cast_6, add_8, add_9)
        del add_8, add_9, cast_6

        # pd_op.shape64: (3xi64) <- (1x-1x-1xf32)
        shape64_0 = paddle._C_ops.shape64(where_0)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_1, full_int_array_0, [1], [0]
        )
        del full_int_array_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [3]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_0, full_int_array_6, [1], [0]
        )
        del full_int_array_0, full_int_array_6, shape64_0

        # pd_op.full: (1xi32) <- ()
        full_12 = paddle._C_ops.full(
            [1], float("13"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.topk: (1x-1x13xf32, 1x-1x13xi64) <- (1x-1x-1xf32, 1xi32)
        topk_0, topk_1 = (lambda x, f: f(x))(
            paddle._C_ops.topk(where_0, full_12, -1, True, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del full_12, where_0

        # pd_op.one_hot: (1x-1x13x-1xf32) <- (1x-1x13xi64, xi64)
        one_hot_0 = paddle._C_ops.one_hot(
            topk_1 % paddle.cast(slice_5, topk_1.dtype), slice_5
        )
        del slice_5, topk_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [-2]

        # pd_op.sum: (1x-1x-1xf32) <- (1x-1x13x-1xf32, 1xi64)
        sum_1 = paddle._C_ops.sum(one_hot_0, full_int_array_7, None, False)
        del one_hot_0

        # pd_op.multiply: (1x-1x-1xf32) <- (1x-1x-1xf32, 1x-1x1xf32)
        multiply_4 = paddle._C_ops.multiply(sum_1, data_7)
        del data_7, sum_1

        # pd_op.greater_than: (1x-1x-1xb) <- (1x-1x-1xf32, xf32)
        greater_than_3 = paddle._C_ops.greater_than(multiply_3, full_11)
        del multiply_3

        # pd_op.greater_than: (1x-1x-1xb) <- (1x-1x-1xf32, xf32)
        greater_than_4 = paddle._C_ops.greater_than(multiply_2, full_11)
        del full_11, multiply_2

        # pd_op.bitwise_or: (1x-1x-1xb) <- (1x-1x-1xb, 1x-1x-1xb)
        bitwise_or_0 = paddle._C_ops.bitwise_or(greater_than_3, greater_than_4)
        del greater_than_3, greater_than_4

        # pd_op.cast: (1x-1x-1xf32) <- (1x-1x-1xb)
        cast_7 = paddle._C_ops.cast(bitwise_or_0, paddle.float32)
        del bitwise_or_0

        # pd_op.multiply: (1x-1x-1xf32) <- (1x-1x-1xf32, 1x-1x-1xf32)
        multiply_5 = paddle._C_ops.multiply(multiply_4, cast_7)
        del cast_7, multiply_4

        # pd_op.sum: (1x-1xf32) <- (1x-1x-1xf32, 1xi64)
        sum_2 = paddle._C_ops.sum(multiply_5, full_int_array_7, None, False)
        del full_int_array_7

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_8 = []

        # pd_op.max: (xf32) <- (1x-1xf32, 0xi64)
        max_0 = paddle._C_ops.max(sum_2, full_int_array_8, False)
        del full_int_array_8

        # pd_op.full: (xf32) <- ()
        full_13 = paddle._C_ops.full(
            [], float("1"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (xb) <- (xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(max_0, full_13)
        del divide_0, full_13, max_0, multiply_1, multiply_5, sum_2, unsqueeze_2

        return greater_than_0
