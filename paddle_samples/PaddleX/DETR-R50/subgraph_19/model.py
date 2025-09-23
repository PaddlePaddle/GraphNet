import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        data_0,
        data_1,
        data_2,
        data_3,
        data_4,
        data_5,
        data_6,
        data_7,
        data_8,
        data_9,
        data_10,
        data_11,
        data_12,
        data_13,
        data_14,
        data_15,
        data_16,
        data_17,
        data_18,
        data_19,
        data_20,
        data_21,
        data_22,
    ):
        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_0 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_1 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_2 = full_0

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(data_0, full_0, float("1"), True)
        del data_0

        # pd_op.add: (xi64) <- (xi64, xi64)
        add_0 = paddle._C_ops.add(scale_1, data_1)
        del data_1, scale_1

        # pd_op.add: (xi64) <- (xi64, xi64)
        add_1 = paddle._C_ops.add(add_0, data_2)
        del add_0, data_2

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_0 = paddle._C_ops.equal(add_1, full_1)
        del add_1

        # pd_op.cast: (xi64) <- (xb)
        cast_0 = paddle._C_ops.cast(equal_0, paddle.int64)
        del equal_0

        # pd_op.not_equal: (xb) <- (xi64, xi64)
        not_equal_0 = paddle._C_ops.not_equal(cast_0, full_1)
        del cast_0

        # pd_op.cast: (xi64) <- (xb)
        cast_1 = paddle._C_ops.cast(not_equal_0, paddle.int64)
        del not_equal_0

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_1 = paddle._C_ops.equal(cast_1, full_1)
        del cast_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_3 = full_int_array_1

        # pd_op.slice: (100x4xf32) <- (4x100x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_9, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_4 = full_2

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_5 = full_2

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_6 = full_2

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_7 = full_2

        # pd_op.gather: (1x4xf32) <- (100x4xf32, 1xi64, 1xi32)
        gather_0 = paddle._C_ops.gather(slice_0, data_14, full_2)
        del data_14

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_8 = full_int_array_2

        # pd_op.slice: (100x4xf32) <- (4x100x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_9, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.greater_than: (xb) <- (xi64, xi64)
        greater_than_0 = paddle._C_ops.greater_than(data_3, full_1)
        del data_3

        # pd_op.cast: (xi64) <- (xb)
        cast_2 = paddle._C_ops.cast(greater_than_0, paddle.int64)
        del greater_than_0

        # pd_op.not_equal: (xb) <- (xi64, xi64)
        not_equal_1 = paddle._C_ops.not_equal(cast_2, full_1)
        del cast_2

        # pd_op.cast: (xi64) <- (xb)
        cast_3 = paddle._C_ops.cast(not_equal_1, paddle.int64)
        del not_equal_1

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_2 = paddle._C_ops.equal(cast_3, full_1)
        del cast_3

        # pd_op.gather: (-1x4xf32) <- (100x4xf32, -1xi64, 1xi32)
        gather_1 = paddle._C_ops.gather(slice_1, data_16, full_2)
        del data_16

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_9 = full_int_array_3

        # pd_op.slice: (100x4xf32) <- (4x100x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            data_9, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.greater_than: (xb) <- (xi64, xi64)
        greater_than_1 = paddle._C_ops.greater_than(data_4, full_1)
        del data_4

        # pd_op.cast: (xi64) <- (xb)
        cast_4 = paddle._C_ops.cast(greater_than_1, paddle.int64)
        del greater_than_1

        # pd_op.not_equal: (xb) <- (xi64, xi64)
        not_equal_2 = paddle._C_ops.not_equal(cast_4, full_1)
        del cast_4

        # pd_op.cast: (xi64) <- (xb)
        cast_5 = paddle._C_ops.cast(not_equal_2, paddle.int64)
        del not_equal_2

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_3 = paddle._C_ops.equal(cast_5, full_1)
        del cast_5

        # pd_op.gather: (-1x4xf32) <- (100x4xf32, -1xi64, 1xi32)
        gather_2 = paddle._C_ops.gather(slice_2, data_18, full_2)
        del data_18

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [4]

        # pd_op.slice: (100x4xf32) <- (4x100x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            data_9, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del data_9

        # pd_op.greater_than: (xb) <- (xi64, xi64)
        greater_than_2 = paddle._C_ops.greater_than(data_5, full_1)
        del data_5

        # pd_op.cast: (xi64) <- (xb)
        cast_6 = paddle._C_ops.cast(greater_than_2, paddle.int64)
        del greater_than_2

        # pd_op.not_equal: (xb) <- (xi64, xi64)
        not_equal_3 = paddle._C_ops.not_equal(cast_6, full_1)
        del cast_6

        # pd_op.cast: (xi64) <- (xb)
        cast_7 = paddle._C_ops.cast(not_equal_3, paddle.int64)
        del not_equal_3

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_4 = paddle._C_ops.equal(cast_7, full_1)
        del cast_7

        # pd_op.gather: (-1x4xf32) <- (100x4xf32, -1xi64, 1xi32)
        gather_3 = paddle._C_ops.gather(slice_3, data_20, full_2)
        del data_20

        # builtin.combine: ([1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32]) <- (1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32)
        combine_0 = [gather_0, gather_1, gather_2, gather_3]

        # pd_op.concat: (-1x4xf32) <- ([1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_2)
        del combine_0

        # pd_op.gather: (1x4xf32) <- (1x4xf32, 1xi64, 1xi32)
        gather_4 = paddle._C_ops.gather(data_10, data_15, full_2)
        del data_10, data_15

        # pd_op.greater_than: (xb) <- (xi64, xi64)
        greater_than_3 = paddle._C_ops.greater_than(data_6, full_1)
        del data_6

        # pd_op.cast: (xi64) <- (xb)
        cast_8 = paddle._C_ops.cast(greater_than_3, paddle.int64)
        del greater_than_3

        # pd_op.not_equal: (xb) <- (xi64, xi64)
        not_equal_4 = paddle._C_ops.not_equal(cast_8, full_1)
        del cast_8

        # pd_op.cast: (xi64) <- (xb)
        cast_9 = paddle._C_ops.cast(not_equal_4, paddle.int64)
        del not_equal_4

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_5 = paddle._C_ops.equal(cast_9, full_1)
        del cast_9

        # pd_op.gather: (-1x4xf32) <- (-1x4xf32, -1xi64, 1xi32)
        gather_5 = paddle._C_ops.gather(data_11, data_17, full_2)
        del data_11, data_17

        # pd_op.greater_than: (xb) <- (xi64, xi64)
        greater_than_4 = paddle._C_ops.greater_than(data_7, full_1)
        del data_7

        # pd_op.cast: (xi64) <- (xb)
        cast_10 = paddle._C_ops.cast(greater_than_4, paddle.int64)
        del greater_than_4

        # pd_op.not_equal: (xb) <- (xi64, xi64)
        not_equal_5 = paddle._C_ops.not_equal(cast_10, full_1)
        del cast_10

        # pd_op.cast: (xi64) <- (xb)
        cast_11 = paddle._C_ops.cast(not_equal_5, paddle.int64)
        del not_equal_5

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_6 = paddle._C_ops.equal(cast_11, full_1)
        del cast_11

        # pd_op.gather: (-1x4xf32) <- (-1x4xf32, -1xi64, 1xi32)
        gather_6 = paddle._C_ops.gather(data_12, data_19, full_2)
        del data_12, data_19

        # pd_op.greater_than: (xb) <- (xi64, xi64)
        greater_than_5 = paddle._C_ops.greater_than(data_8, full_1)
        del data_8

        # pd_op.cast: (xi64) <- (xb)
        cast_12 = paddle._C_ops.cast(greater_than_5, paddle.int64)
        del greater_than_5

        # pd_op.not_equal: (xb) <- (xi64, xi64)
        not_equal_6 = paddle._C_ops.not_equal(cast_12, full_1)
        del cast_12

        # pd_op.cast: (xi64) <- (xb)
        cast_13 = paddle._C_ops.cast(not_equal_6, paddle.int64)
        del not_equal_6

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_7 = paddle._C_ops.equal(cast_13, full_1)
        del cast_13, full_1

        # pd_op.gather: (-1x4xf32) <- (-1x4xf32, -1xi64, 1xi32)
        gather_7 = paddle._C_ops.gather(data_13, data_21, full_2)
        del data_13, data_21

        # builtin.combine: ([1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32]) <- (1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32)
        combine_1 = [gather_4, gather_5, gather_6, gather_7]
        del gather_4, gather_5, gather_6, gather_7

        # pd_op.concat: (-1x4xf32) <- ([1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_2)
        del combine_1

        # pd_op.subtract: (-1x4xf32) <- (-1x4xf32, -1x4xf32)
        subtract_0 = paddle._C_ops.subtract(concat_0, concat_1)

        # pd_op.abs: (-1x4xf32) <- (-1x4xf32)
        abs_0 = paddle._C_ops.abs(subtract_0)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_5 = []

        # pd_op.assign: (0xi64) <- (0xi64)
        assign_10 = full_int_array_5

        # pd_op.sum: (xf32) <- (-1x4xf32, 0xi64)
        sum_0 = paddle._C_ops.sum(abs_0, full_int_array_5, None, False)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(sum_0, full_3, float("0"), True)
        del sum_0

        # pd_op.divide: (1xf32) <- (xf32, 1xf32)
        divide_0 = paddle._C_ops.divide(scale_2, data_22)

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_11 = full_4

        # pd_op.split_with_num: ([-1x2xf32, -1x2xf32]) <- (-1x4xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(concat_0, 2, full_4)

        # builtin.split: (-1x2xf32, -1x2xf32) <- ([-1x2xf32, -1x2xf32])
        (
            split_0,
            split_1,
        ) = split_with_num_0
        del split_with_num_0

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_12 = full_5

        # pd_op.scale: (-1x2xf32) <- (-1x2xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(split_1, full_5, float("0"), True)
        del split_1

        # pd_op.assign: (-1x2xf32) <- (-1x2xf32)
        assign_13 = scale_3

        # pd_op.subtract: (-1x2xf32) <- (-1x2xf32, -1x2xf32)
        subtract_1 = paddle._C_ops.subtract(split_0, scale_3)

        # pd_op.add: (-1x2xf32) <- (-1x2xf32, -1x2xf32)
        add_2 = paddle._C_ops.add(split_0, scale_3)

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x2xf32, -1x2xf32]) <- (-1x2xf32, -1x2xf32)
        combine_2 = [subtract_1, add_2]

        # pd_op.concat: (-1x4xf32) <- ([-1x2xf32, -1x2xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_6)
        del combine_2

        # pd_op.split_with_num: ([-1x2xf32, -1x2xf32]) <- (-1x4xf32, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(concat_1, 2, full_4)

        # builtin.split: (-1x2xf32, -1x2xf32) <- ([-1x2xf32, -1x2xf32])
        (
            split_2,
            split_3,
        ) = split_with_num_1
        del split_with_num_1

        # pd_op.scale: (-1x2xf32) <- (-1x2xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(split_3, full_5, float("0"), True)
        del split_3

        # pd_op.subtract: (-1x2xf32) <- (-1x2xf32, -1x2xf32)
        subtract_2 = paddle._C_ops.subtract(split_2, scale_4)

        # pd_op.add: (-1x2xf32) <- (-1x2xf32, -1x2xf32)
        add_3 = paddle._C_ops.add(split_2, scale_4)
        del scale_4, split_2

        # builtin.combine: ([-1x2xf32, -1x2xf32]) <- (-1x2xf32, -1x2xf32)
        combine_3 = [subtract_2, add_3]
        del add_3, subtract_2

        # pd_op.concat: (-1x4xf32) <- ([-1x2xf32, -1x2xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_3, full_6)
        del combine_3

        # pd_op.split_with_num: ([-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32]) <- (-1x4xf32, 1xi32)
        split_with_num_2 = paddle._C_ops.split_with_num(concat_2, 4, full_4)
        del concat_2

        # builtin.split: (-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32) <- ([-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32])
        (
            split_4,
            split_5,
            split_6,
            split_7,
        ) = split_with_num_2
        del split_with_num_2

        # pd_op.split_with_num: ([-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32]) <- (-1x4xf32, 1xi32)
        split_with_num_3 = paddle._C_ops.split_with_num(concat_3, 4, full_4)
        del concat_3

        # builtin.split: (-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32) <- ([-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32])
        (
            split_8,
            split_9,
            split_10,
            split_11,
        ) = split_with_num_3
        del split_with_num_3

        # pd_op.maximum: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        maximum_0 = paddle._C_ops.maximum(split_4, split_8)

        # pd_op.maximum: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        maximum_1 = paddle._C_ops.maximum(split_5, split_9)

        # pd_op.minimum: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        minimum_0 = paddle._C_ops.minimum(split_6, split_10)

        # pd_op.minimum: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        minimum_1 = paddle._C_ops.minimum(split_7, split_11)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_3 = paddle._C_ops.subtract(minimum_0, maximum_0)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_14 = full_7

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_15 = full_8

        # pd_op.clip: (-1x1xf32) <- (-1x1xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(subtract_3, full_7, full_8)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_4 = paddle._C_ops.subtract(minimum_1, maximum_1)

        # pd_op.clip: (-1x1xf32) <- (-1x1xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(subtract_4, full_7, full_8)

        # pd_op.multiply: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        multiply_0 = paddle._C_ops.multiply(clip_0, clip_1)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_5 = paddle._C_ops.subtract(split_6, split_4)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_6 = paddle._C_ops.subtract(split_7, split_5)

        # pd_op.multiply: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        multiply_1 = paddle._C_ops.multiply(subtract_5, subtract_6)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_7 = paddle._C_ops.subtract(split_10, split_8)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_8 = paddle._C_ops.subtract(split_11, split_9)

        # pd_op.multiply: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        multiply_2 = paddle._C_ops.multiply(subtract_7, subtract_8)
        del subtract_7, subtract_8

        # pd_op.add: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        add_4 = paddle._C_ops.add(multiply_1, multiply_2)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_9 = paddle._C_ops.subtract(add_4, multiply_0)

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(subtract_9, full_0, float("1e-10"), True)
        del subtract_9

        # pd_op.divide: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        divide_1 = paddle._C_ops.divide(multiply_0, scale_5)

        # pd_op.minimum: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        minimum_2 = paddle._C_ops.minimum(split_4, split_8)

        # pd_op.minimum: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        minimum_3 = paddle._C_ops.minimum(split_5, split_9)

        # pd_op.maximum: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        maximum_2 = paddle._C_ops.maximum(split_6, split_10)

        # pd_op.maximum: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        maximum_3 = paddle._C_ops.maximum(split_7, split_11)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_10 = paddle._C_ops.subtract(maximum_2, minimum_2)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_11 = paddle._C_ops.subtract(maximum_3, minimum_3)

        # pd_op.multiply: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        multiply_3 = paddle._C_ops.multiply(subtract_10, subtract_11)

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(multiply_3, full_0, float("1e-10"), True)
        del multiply_3

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_12 = paddle._C_ops.subtract(scale_6, scale_5)

        # pd_op.divide: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        divide_2 = paddle._C_ops.divide(subtract_12, scale_6)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_13 = paddle._C_ops.subtract(divide_1, divide_2)

        # pd_op.full: (1xf32) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(subtract_13, full_9, float("1"), True)
        del subtract_13

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(scale_7, full_0, float("0"), True)
        del full_0, scale_7

        # pd_op.sum: (xf32) <- (-1x1xf32, 0xi64)
        sum_1 = paddle._C_ops.sum(scale_8, full_int_array_5, None, False)

        # pd_op.divide: (1xf32) <- (xf32, 1xf32)
        divide_3 = paddle._C_ops.divide(sum_1, data_22)
        del data_22

        # pd_op.full: (1xf32) <- ()
        full_10 = paddle._C_ops.full(
            [1], float("2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1xf32) <- (1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(divide_3, full_10, float("0"), True)
        del (
            abs_0,
            add_2,
            add_4,
            assign_0,
            assign_1,
            assign_10,
            assign_11,
            assign_12,
            assign_13,
            assign_14,
            assign_15,
            assign_2,
            assign_3,
            assign_4,
            assign_5,
            assign_6,
            assign_7,
            assign_8,
            assign_9,
            clip_0,
            clip_1,
            concat_0,
            concat_1,
            divide_1,
            divide_2,
            divide_3,
            full_10,
            full_2,
            full_3,
            full_4,
            full_5,
            full_6,
            full_7,
            full_8,
            full_9,
            full_int_array_0,
            full_int_array_1,
            full_int_array_2,
            full_int_array_3,
            full_int_array_4,
            full_int_array_5,
            gather_0,
            gather_1,
            gather_2,
            gather_3,
            maximum_0,
            maximum_1,
            maximum_2,
            maximum_3,
            minimum_0,
            minimum_1,
            minimum_2,
            minimum_3,
            multiply_0,
            multiply_1,
            multiply_2,
            scale_2,
            scale_3,
            scale_5,
            scale_6,
            scale_8,
            slice_0,
            slice_1,
            slice_2,
            slice_3,
            split_0,
            split_10,
            split_11,
            split_4,
            split_5,
            split_6,
            split_7,
            split_8,
            split_9,
            subtract_0,
            subtract_1,
            subtract_10,
            subtract_11,
            subtract_12,
            subtract_3,
            subtract_4,
            subtract_5,
            subtract_6,
            sum_1,
        )

        return divide_0, scale_0
