import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4, data_5):
        # pd_op.full: (xf32) <- ()
        full_0 = paddle._C_ops.full(
            [], float("1"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (1x25x21504xb) <- (1x25x21504xf32, xf32)
        greater_than_1 = paddle._C_ops.greater_than(data_0, full_0)
        del full_0

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (1x25x21504xf32) <- (1x25x21504xf32, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            data_0, full_1, paddle.float32, paddle.framework._current_expected_place()
        )
        del full_1

        # pd_op.where: (1x25x21504xf32) <- (1x25x21504xb, 1x25x21504xf32, 1x25x21504xf32)
        where_0 = paddle._C_ops.where(greater_than_1, full_like_0, data_0)
        del data_0, full_like_0, greater_than_1

        # pd_op.transpose: (1x15x21504xf32) <- (1x21504x15xf32)
        transpose_0 = paddle._C_ops.transpose(data_1, [0, 2, 1])
        del data_1

        # pd_op.full: (1xf64) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (1xi32) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_2, full_3, full_3, dtype="int32")
        del full_2, full_3

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [-1]

        # pd_op.unsqueeze: (1x1xi32) <- (1xi32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(arange_0, full_int_array_0)
        del arange_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [1, 25]

        # pd_op.tile: (1x25xi32) <- (1x1xi32, 2xi64)
        tile_0 = paddle._C_ops.tile(unsqueeze_0, full_int_array_1)
        del full_int_array_1

        # pd_op.squeeze: (1x25xi32) <- (1x25x1xi32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(data_2, full_int_array_0)
        del data_2

        # builtin.combine: ([1x25xi32, 1x25xi32]) <- (1x25xi32, 1x25xi32)
        combine_0 = [tile_0, squeeze_0]
        del squeeze_0, tile_0

        # pd_op.stack: (1x25x2xi32) <- ([1x25xi32, 1x25xi32])
        stack_0 = paddle._C_ops.stack(combine_0, -1)
        del combine_0

        # pd_op.gather_nd: (1x25x21504xf32) <- (1x15x21504xf32, 1x25x2xi32)
        gather_nd_0 = paddle._C_ops.gather_nd(transpose_0, stack_0)
        del stack_0, transpose_0

        # pd_op.pow: (1x25x21504xf32) <- (1x25x21504xf32)
        pow_0 = paddle._C_ops.pow(gather_nd_0, float("1"))
        del gather_nd_0

        # pd_op.pow: (1x25x21504xf32) <- (1x25x21504xf32)
        pow_1 = paddle._C_ops.pow(where_0, float("6"))

        # pd_op.multiply: (1x25x21504xf32) <- (1x25x21504xf32, 1x25x21504xf32)
        multiply_0 = paddle._C_ops.multiply(pow_0, pow_1)
        del pow_0, pow_1

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("2"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split_with_num: ([1x25x1xf32, 1x25x1xf32, 1x25x1xf32, 1x25x1xf32, 1x25x1xf32]) <- (1x25x5xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(data_4, 5, full_4)
        del data_4

        # builtin.split: (1x25x1xf32, 1x25x1xf32, 1x25x1xf32, 1x25x1xf32, 1x25x1xf32) <- ([1x25x1xf32, 1x25x1xf32, 1x25x1xf32, 1x25x1xf32, 1x25x1xf32])
        (
            split_0,
            split_1,
            split_2,
            split_3,
            split_4,
        ) = split_with_num_0
        del split_with_num_0

        # pd_op.full: (4xf32) <- ()
        full_5 = paddle._C_ops.full(
            [4], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (4xf32) <- (4xf32)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_5,
            [4],
            paddle.float32,
            [float("0.5"), float("0.5"), float("-0.5"), float("-0.5")],
            paddle.framework._current_expected_place(),
        )
        del full_5

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_2 = [1, 1, 4]

        # pd_op.reshape: (1x1x4xf32) <- (4xf32, 3xi64)
        reshape_0 = paddle._C_ops.reshape(assign_value__0, full_int_array_2)
        del assign_value__0

        # pd_op.multiply: (1x25x4xf32) <- (1x1x4xf32, 1x25x1xf32)
        multiply_1 = paddle._C_ops.multiply(reshape_0, split_2)
        del reshape_0, split_2

        # pd_op.full: (4xf32) <- ()
        full_6 = paddle._C_ops.full(
            [4], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (4xf32) <- (4xf32)
        assign_value__1 = paddle._C_ops.assign_value_(
            full_6,
            [4],
            paddle.float32,
            [float("-0.5"), float("0.5"), float("0.5"), float("-0.5")],
            paddle.framework._current_expected_place(),
        )
        del full_6

        # pd_op.reshape: (1x1x4xf32) <- (4xf32, 3xi64)
        reshape_1 = paddle._C_ops.reshape(assign_value__1, full_int_array_2)
        del assign_value__1, full_int_array_2

        # pd_op.multiply: (1x25x4xf32) <- (1x1x4xf32, 1x25x1xf32)
        multiply_2 = paddle._C_ops.multiply(reshape_1, split_3)
        del reshape_1, split_3

        # builtin.combine: ([1x25x4xf32, 1x25x4xf32]) <- (1x25x4xf32, 1x25x4xf32)
        combine_1 = [multiply_1, multiply_2]
        del multiply_1, multiply_2

        # pd_op.stack: (1x25x4x2xf32) <- ([1x25x4xf32, 1x25x4xf32])
        stack_1 = paddle._C_ops.stack(combine_1, -1)
        del combine_1

        # pd_op.sin: (1x25x1xf32) <- (1x25x1xf32)
        sin_0 = paddle._C_ops.sin(split_4)

        # pd_op.cos: (1x25x1xf32) <- (1x25x1xf32)
        cos_0 = paddle._C_ops.cos(split_4)
        del split_4

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([1x25x1xf32, 1x25x1xf32]) <- (1x25x1xf32, 1x25x1xf32)
        combine_2 = [cos_0, sin_0]

        # pd_op.concat: (1x25x2xf32) <- ([1x25x1xf32, 1x25x1xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_2, full_7)
        del combine_2

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x25x1xf32) <- (1x25x1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(sin_0, full_8, float("0"), True)
        del full_8, sin_0

        # builtin.combine: ([1x25x1xf32, 1x25x1xf32]) <- (1x25x1xf32, 1x25x1xf32)
        combine_3 = [scale_0, cos_0]
        del cos_0, scale_0

        # pd_op.concat: (1x25x2xf32) <- ([1x25x1xf32, 1x25x1xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_3, full_7)
        del combine_3, full_7

        # builtin.combine: ([1x25x2xf32, 1x25x2xf32]) <- (1x25x2xf32, 1x25x2xf32)
        combine_4 = [concat_0, concat_1]
        del concat_0, concat_1

        # pd_op.stack: (1x25x2x2xf32) <- ([1x25x2xf32, 1x25x2xf32])
        stack_2 = paddle._C_ops.stack(combine_4, -2)
        del combine_4

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_3 = [-1, 4, 2]

        # pd_op.reshape: (25x4x2xf32) <- (1x25x4x2xf32, 3xi64)
        reshape_2 = paddle._C_ops.reshape(stack_1, full_int_array_3)
        del full_int_array_3, stack_1

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_4 = [-1, 2, 2]

        # pd_op.reshape: (25x2x2xf32) <- (1x25x2x2xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(stack_2, full_int_array_4)
        del full_int_array_4, stack_2

        # pd_op.bmm: (25x4x2xf32) <- (25x4x2xf32, 25x2x2xf32)
        bmm_0 = paddle._C_ops.bmm(reshape_2, reshape_3)
        del reshape_2, reshape_3

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_5 = [1, -1, 4, 2]

        # pd_op.reshape: (1x25x4x2xf32) <- (25x4x2xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(bmm_0, full_int_array_5)
        del bmm_0, full_int_array_5

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [1]

        # pd_op.slice: (1x25x4xf32) <- (1x25x4x2xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            reshape_4, [3], full_int_array_6, full_int_array_7, [1], [3]
        )

        # pd_op.add: (1x25x4xf32) <- (1x25x4xf32, 1x25x1xf32)
        add_0 = paddle._C_ops.add(slice_0, split_0)
        del slice_0, split_0

        # pd_op.set_value_with_tensor_: (1x25x4x2xf32) <- (1x25x4x2xf32, 1x25x4xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__0 = paddle._C_ops.set_value_with_tensor_(
            reshape_4,
            add_0,
            full_int_array_6,
            full_int_array_7,
            full_int_array_7,
            [3],
            [3],
            [],
        )
        del add_0, reshape_4

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [2]

        # pd_op.slice: (1x25x4xf32) <- (1x25x4x2xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            set_value_with_tensor__0, [3], full_int_array_7, full_int_array_8, [1], [3]
        )

        # pd_op.add: (1x25x4xf32) <- (1x25x4xf32, 1x25x1xf32)
        add_1 = paddle._C_ops.add(slice_1, split_1)
        del slice_1, split_1

        # pd_op.set_value_with_tensor_: (1x25x4x2xf32) <- (1x25x4x2xf32, 1x25x4xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__1 = paddle._C_ops.set_value_with_tensor_(
            set_value_with_tensor__0,
            add_1,
            full_int_array_7,
            full_int_array_8,
            full_int_array_7,
            [3],
            [3],
            [],
        )
        del add_1, full_int_array_7, full_int_array_8, set_value_with_tensor__0

        # pd_op.unsqueeze: (1x1x21504x2xf32) <- (1x21504x2xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(data_3, full_int_array_6)
        del data_3, full_int_array_6

        # pd_op.split_with_num: ([1x25x1x2xf32, 1x25x1x2xf32, 1x25x1x2xf32, 1x25x1x2xf32]) <- (1x25x4x2xf32, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(
            set_value_with_tensor__1, 4, full_4
        )
        del full_4

        # builtin.split: (1x25x1x2xf32, 1x25x1x2xf32, 1x25x1x2xf32, 1x25x1x2xf32) <- ([1x25x1x2xf32, 1x25x1x2xf32, 1x25x1x2xf32, 1x25x1x2xf32])
        (
            split_5,
            split_6,
            split_7,
            split_8,
        ) = split_with_num_1
        del split_with_num_1

        # pd_op.subtract: (1x25x1x2xf32) <- (1x25x1x2xf32, 1x25x1x2xf32)
        subtract_0 = paddle._C_ops.subtract(split_6, split_5)
        del split_6

        # pd_op.subtract: (1x25x1x2xf32) <- (1x25x1x2xf32, 1x25x1x2xf32)
        subtract_1 = paddle._C_ops.subtract(split_8, split_5)
        del split_8

        # pd_op.subtract: (1x25x21504x2xf32) <- (1x1x21504x2xf32, 1x25x1x2xf32)
        subtract_2 = paddle._C_ops.subtract(unsqueeze_1, split_5)
        del split_5, unsqueeze_1

        # pd_op.multiply: (1x25x1x2xf32) <- (1x25x1x2xf32, 1x25x1x2xf32)
        multiply_3 = paddle._C_ops.multiply(subtract_0, subtract_0)

        # pd_op.sum: (1x25x1xf32) <- (1x25x1x2xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(multiply_3, full_int_array_0, None, False)
        del multiply_3

        # pd_op.multiply: (1x25x1x2xf32) <- (1x25x1x2xf32, 1x25x1x2xf32)
        multiply_4 = paddle._C_ops.multiply(subtract_1, subtract_1)

        # pd_op.sum: (1x25x1xf32) <- (1x25x1x2xf32, 1xi64)
        sum_1 = paddle._C_ops.sum(multiply_4, full_int_array_0, None, False)
        del multiply_4

        # pd_op.multiply: (1x25x21504x2xf32) <- (1x25x21504x2xf32, 1x25x1x2xf32)
        multiply_5 = paddle._C_ops.multiply(subtract_2, subtract_0)
        del subtract_0

        # pd_op.sum: (1x25x21504xf32) <- (1x25x21504x2xf32, 1xi64)
        sum_2 = paddle._C_ops.sum(multiply_5, full_int_array_0, None, False)
        del multiply_5

        # pd_op.multiply: (1x25x21504x2xf32) <- (1x25x21504x2xf32, 1x25x1x2xf32)
        multiply_6 = paddle._C_ops.multiply(subtract_2, subtract_1)
        del subtract_1, subtract_2

        # pd_op.sum: (1x25x21504xf32) <- (1x25x21504x2xf32, 1xi64)
        sum_3 = paddle._C_ops.sum(multiply_6, full_int_array_0, None, False)
        del full_int_array_0, multiply_6

        # pd_op.full: (xf32) <- ()
        full_9 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_equal: (1x25x21504xb) <- (1x25x21504xf32, xf32)
        greater_equal_0 = paddle._C_ops.greater_equal(sum_2, full_9)

        # pd_op.less_equal: (1x25x21504xb) <- (1x25x21504xf32, 1x25x1xf32)
        less_equal_0 = paddle._C_ops.less_equal(sum_2, sum_0)
        del sum_0, sum_2

        # pd_op.bitwise_and: (1x25x21504xb) <- (1x25x21504xb, 1x25x21504xb)
        bitwise_and_0 = paddle._C_ops.bitwise_and(greater_equal_0, less_equal_0)
        del greater_equal_0, less_equal_0

        # pd_op.greater_equal: (1x25x21504xb) <- (1x25x21504xf32, xf32)
        greater_equal_1 = paddle._C_ops.greater_equal(sum_3, full_9)
        del full_9

        # pd_op.bitwise_and: (1x25x21504xb) <- (1x25x21504xb, 1x25x21504xb)
        bitwise_and_1 = paddle._C_ops.bitwise_and(bitwise_and_0, greater_equal_1)
        del bitwise_and_0, greater_equal_1

        # pd_op.less_equal: (1x25x21504xb) <- (1x25x21504xf32, 1x25x1xf32)
        less_equal_1 = paddle._C_ops.less_equal(sum_3, sum_1)
        del sum_1, sum_3

        # pd_op.bitwise_and: (1x25x21504xb) <- (1x25x21504xb, 1x25x21504xb)
        bitwise_and_2 = paddle._C_ops.bitwise_and(bitwise_and_1, less_equal_1)
        del bitwise_and_1, less_equal_1

        # pd_op.cast: (1x25x21504xf32) <- (1x25x21504xb)
        cast_0 = paddle._C_ops.cast(bitwise_and_2, paddle.float32)

        # pd_op.multiply: (1x25x21504xf32) <- (1x25x21504xf32, 1x25x21504xf32)
        multiply_7 = paddle._C_ops.multiply(multiply_0, cast_0)
        del cast_0

        # pd_op.full: (1xi32) <- ()
        full_10 = paddle._C_ops.full(
            [1], float("13"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.topk: (1x25x13xf32, 1x25x13xi64) <- (1x25x21504xf32, 1xi32)
        topk_0, topk_1 = (lambda x, f: f(x))(
            paddle._C_ops.topk(multiply_7, full_10, -1, True, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del full_10, multiply_7

        # pd_op.full: (1xi32) <- ()
        full_11 = paddle._C_ops.full(
            [1], float("21504"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.one_hot: (1x25x13x21504xf32) <- (1x25x13xi64, 1xi32)
        one_hot_0 = paddle._C_ops.one_hot(
            topk_1 % paddle.cast(full_11, topk_1.dtype), full_11
        )
        del full_11, topk_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [-2]

        # pd_op.sum: (1x25x21504xf32) <- (1x25x13x21504xf32, 1xi64)
        sum_4 = paddle._C_ops.sum(one_hot_0, full_int_array_9, None, False)
        del one_hot_0

        # pd_op.multiply: (1x25x21504xf32) <- (1x25x21504xf32, 1x25x1xf32)
        multiply_8 = paddle._C_ops.multiply(sum_4, data_5)
        del sum_4

        # pd_op.cast: (1x25x21504xf32) <- (1x25x21504xb)
        cast_1 = paddle._C_ops.cast(bitwise_and_2, paddle.float32)
        del bitwise_and_2

        # pd_op.multiply: (1x25x21504xf32) <- (1x25x21504xf32, 1x25x21504xf32)
        multiply_9 = paddle._C_ops.multiply(multiply_8, cast_1)
        del cast_1, multiply_8

        # pd_op.multiply: (1x25x21504xf32) <- (1x25x21504xf32, 1x25x1xf32)
        multiply_10 = paddle._C_ops.multiply(multiply_9, data_5)
        del data_5, multiply_9

        # pd_op.sum: (1x21504xf32) <- (1x25x21504xf32, 1xi64)
        sum_5 = paddle._C_ops.sum(multiply_10, full_int_array_9, None, False)
        del full_int_array_9

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_10 = []

        # pd_op.max: (xf32) <- (1x21504xf32, 0xi64)
        max_0 = paddle._C_ops.max(sum_5, full_int_array_10, False)
        del full_int_array_10

        # pd_op.full: (xf32) <- ()
        full_12 = paddle._C_ops.full(
            [], float("1"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (xb) <- (xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(max_0, full_12)
        del (
            full_12,
            max_0,
            multiply_0,
            multiply_10,
            set_value_with_tensor__1,
            sum_5,
            unsqueeze_0,
            where_0,
        )

        return greater_than_0
