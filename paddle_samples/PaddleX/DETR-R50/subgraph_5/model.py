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
        data_23,
        data_24,
        data_25,
        data_26,
        data_27,
    ):
        # pd_op.full: (4x100xi64) <- ()
        full_0 = paddle._C_ops.full(
            [4, 100],
            float("4"),
            paddle.int64,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_0 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_1 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_2 = full_1

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(data_0, full_1, float("0"), True)
        del data_0

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_2 = paddle._C_ops.scale(scale_1, full_1, float("1"), True)
        del scale_1

        # pd_op.add: (xi64) <- (xi64, xi64)
        add_0 = paddle._C_ops.add(scale_2, data_1)
        del data_1, scale_2

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_3 = paddle._C_ops.scale(add_0, full_1, float("1"), True)
        del add_0

        # pd_op.full: (xi64) <- ()
        full_2 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (xb) <- (xi64, xi64)
        greater_than_0 = paddle._C_ops.greater_than(scale_3, full_2)
        del scale_3

        # pd_op.cast: (xi64) <- (xb)
        cast_0 = paddle._C_ops.cast(greater_than_0, paddle.int64)
        del greater_than_0

        # pd_op.not_equal: (xb) <- (xi64, xi64)
        not_equal_0 = paddle._C_ops.not_equal(cast_0, full_2)
        del cast_0

        # pd_op.cast: (xi64) <- (xb)
        cast_1 = paddle._C_ops.cast(not_equal_0, paddle.int64)
        del not_equal_0

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_0 = paddle._C_ops.equal(cast_1, full_2)
        del cast_1

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_3 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_4 = full_3

        # pd_op.full_like: (-1xi64) <- (-1xi64, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            data_8, full_3, paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.full_like: (1xi64) <- (1xi64, 1xf32)
        full_like_1 = paddle._C_ops.full_like(
            data_10, full_1, paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_5 = full_4

        # pd_op.full_like: (-1xi64) <- (-1xi64, 1xf32)
        full_like_2 = paddle._C_ops.full_like(
            data_12, full_4, paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("3"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (1xi64) <- (1xi64, 1xf32)
        full_like_3 = paddle._C_ops.full_like(
            data_14, full_5, paddle.int64, paddle.framework._current_expected_place()
        )
        del full_5

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_6 = full_6

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_7 = full_6

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_8 = full_6

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_9 = full_6

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_10 = full_6

        # builtin.combine: ([-1xi64, 1xi64, -1xi64, 1xi64]) <- (-1xi64, 1xi64, -1xi64, 1xi64)
        combine_0 = [full_like_0, full_like_1, full_like_2, full_like_3]
        del full_like_0, full_like_1, full_like_2, full_like_3

        # pd_op.concat: (-1xi64) <- ([-1xi64, 1xi64, -1xi64, 1xi64], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_6)
        del combine_0

        # builtin.combine: ([-1xi64, 1xi64, -1xi64, 1xi64]) <- (-1xi64, 1xi64, -1xi64, 1xi64)
        combine_1 = [data_8, data_10, data_12, data_14]

        # pd_op.concat: (-1xi64) <- ([-1xi64, 1xi64, -1xi64, 1xi64], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_6)
        del combine_1

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("100"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1xi64) <- (-1xi64, 1xf32)
        scale_4 = paddle._C_ops.scale(concat_0, full_7, float("0"), True)
        del concat_0, full_7

        # pd_op.add: (-1xi64) <- (-1xi64, -1xi64)
        add_1 = paddle._C_ops.add(concat_1, scale_4)
        del concat_1, scale_4

        # pd_op.gather: (-1x1xi32) <- (-1x1xi32, -1xi64, 1xi32)
        gather_0 = paddle._C_ops.gather(data_21, data_9, full_6)
        del data_21

        # pd_op.gather: (1x1xi32) <- (1x1xi32, 1xi64, 1xi32)
        gather_1 = paddle._C_ops.gather(data_22, data_11, full_6)
        del data_22

        # pd_op.gather: (-1x1xi32) <- (-1x1xi32, -1xi64, 1xi32)
        gather_2 = paddle._C_ops.gather(data_23, data_13, full_6)
        del data_23

        # pd_op.gather: (1x1xi32) <- (1x1xi32, 1xi64, 1xi32)
        gather_3 = paddle._C_ops.gather(data_24, data_15, full_6)
        del data_24

        # builtin.combine: ([-1x1xi32, 1x1xi32, -1x1xi32, 1x1xi32]) <- (-1x1xi32, 1x1xi32, -1x1xi32, 1x1xi32)
        combine_2 = [gather_0, gather_1, gather_2, gather_3]
        del gather_0, gather_1, gather_2, gather_3

        # pd_op.concat: (-1x1xi32) <- ([-1x1xi32, 1x1xi32, -1x1xi32, 1x1xi32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_6)
        del combine_2

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [-1, 1]

        # pd_op.reshape: (400x1xi64) <- (4x100xi64, 2xi64)
        reshape_0 = paddle._C_ops.reshape(full_0, full_int_array_0)
        del full_0, full_int_array_0

        # pd_op.cast: (-1x1xi64) <- (-1x1xi32)
        cast_2 = paddle._C_ops.cast(concat_2, paddle.int64)
        del concat_2

        # pd_op.scatter: (400x1xi64) <- (400x1xi64, -1xi64, -1x1xi64)
        scatter_0 = paddle._C_ops.scatter(reshape_0, add_1, cast_2, True)
        del add_1, cast_2, reshape_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [4, 100]

        # pd_op.reshape: (4x100xi64) <- (400x1xi64, 2xi64)
        reshape_1 = paddle._C_ops.reshape(scatter_0, full_int_array_1)
        del full_int_array_1, scatter_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [-1]

        # pd_op.unsqueeze: (4x100x1xi64) <- (4x100xi64, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(reshape_1, full_int_array_2)
        del reshape_1

        # pd_op.cross_entropy_with_softmax: (4x100x5xf32, 4x100x1xf32) <- (4x100x5xf32, 4x100x1xi64)
        cross_entropy_with_softmax_0, cross_entropy_with_softmax_1 = (
            lambda x, f: f(x)
        )(
            paddle._C_ops.cross_entropy_with_softmax(
                data_20, unsqueeze_0, False, True, True, -100, -1
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del data_20

        # pd_op.full: (xi64) <- ()
        full_8 = paddle._C_ops.full(
            [], float("-100"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.not_equal: (4x100x1xb) <- (4x100x1xi64, xi64)
        not_equal_1 = paddle._C_ops.not_equal(unsqueeze_0, full_8)
        del full_8

        # pd_op.cast: (4x100x1xi64) <- (4x100x1xb)
        cast_3 = paddle._C_ops.cast(not_equal_1, paddle.int64)

        # pd_op.multiply: (4x100x1xi64) <- (4x100x1xi64, 4x100x1xi64)
        multiply_0 = paddle._C_ops.multiply(cast_3, unsqueeze_0)
        del cast_3

        # pd_op.cast: (4x100x1xf32) <- (4x100x1xb)
        cast_4 = paddle._C_ops.cast(not_equal_1, paddle.float32)
        del not_equal_1

        # pd_op.squeeze: (4x100xf32) <- (4x100x1xf32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(cast_4, full_int_array_2)
        del cast_4, full_int_array_2

        # pd_op.gather_nd: (4x100xf32) <- (5xf32, 4x100x1xi64)
        gather_nd_0 = paddle._C_ops.gather_nd(data_27, multiply_0)
        del data_27, multiply_0

        # pd_op.multiply: (4x100xf32) <- (4x100xf32, 4x100xf32)
        multiply_1 = paddle._C_ops.multiply(gather_nd_0, squeeze_0)
        del gather_nd_0, squeeze_0

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_3 = [4, 100, 1]

        # pd_op.reshape: (4x100x1xf32) <- (4x100xf32, 3xi64)
        reshape_2 = paddle._C_ops.reshape(multiply_1, full_int_array_3)
        del full_int_array_3, multiply_1

        # pd_op.multiply: (4x100x1xf32) <- (4x100x1xf32, 4x100x1xf32)
        multiply_2 = paddle._C_ops.multiply(cross_entropy_with_softmax_1, reshape_2)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_4 = []

        # pd_op.assign: (0xi64) <- (0xi64)
        assign_11 = full_int_array_4

        # pd_op.assign: (0xi64) <- (0xi64)
        assign_12 = full_int_array_4

        # pd_op.sum: (xf32) <- (4x100x1xf32, 0xi64)
        sum_0 = paddle._C_ops.sum(multiply_2, full_int_array_4, None, False)

        # pd_op.sum: (xf32) <- (4x100x1xf32, 0xi64)
        sum_1 = paddle._C_ops.sum(reshape_2, full_int_array_4, None, False)

        # pd_op.full: (xf32) <- ()
        full_9 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.equal: (xb) <- (xf32, xf32)
        equal_1 = paddle._C_ops.equal(sum_1, full_9)
        del full_9

        # pd_op.cast: (xf32) <- (xb)
        cast_5 = paddle._C_ops.cast(equal_1, paddle.float32)
        del equal_1

        # pd_op.add: (xf32) <- (xf32, xf32)
        add_2 = paddle._C_ops.add(sum_1, cast_5)
        del cast_5, sum_1

        # pd_op.divide: (xf32) <- (xf32, xf32)
        divide_1 = paddle._C_ops.divide(sum_0, add_2)

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_5 = paddle._C_ops.scale(data_2, full_1, float("0"), True)
        del data_2

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_6 = paddle._C_ops.scale(scale_5, full_1, float("1"), True)
        del scale_5

        # pd_op.add: (xi64) <- (xi64, xi64)
        add_3 = paddle._C_ops.add(scale_6, data_3)
        del data_3, scale_6

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_7 = paddle._C_ops.scale(add_3, full_1, float("1"), True)
        del add_3

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_2 = paddle._C_ops.equal(scale_7, full_2)
        del scale_7

        # pd_op.cast: (xi64) <- (xb)
        cast_6 = paddle._C_ops.cast(equal_2, paddle.int64)
        del equal_2

        # pd_op.not_equal: (xb) <- (xi64, xi64)
        not_equal_2 = paddle._C_ops.not_equal(cast_6, full_2)
        del cast_6

        # pd_op.cast: (xi64) <- (xb)
        cast_7 = paddle._C_ops.cast(not_equal_2, paddle.int64)
        del not_equal_2

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_3 = paddle._C_ops.equal(cast_7, full_2)
        del cast_7

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_13 = full_int_array_6

        # pd_op.slice: (100x4xf32) <- (4x100x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_26, [0], full_int_array_5, full_int_array_6, [1], [0]
        )

        # pd_op.greater_than: (xb) <- (xi64, xi64)
        greater_than_1 = paddle._C_ops.greater_than(data_4, full_2)
        del data_4

        # pd_op.cast: (xi64) <- (xb)
        cast_8 = paddle._C_ops.cast(greater_than_1, paddle.int64)
        del greater_than_1

        # pd_op.not_equal: (xb) <- (xi64, xi64)
        not_equal_3 = paddle._C_ops.not_equal(cast_8, full_2)
        del cast_8

        # pd_op.cast: (xi64) <- (xb)
        cast_9 = paddle._C_ops.cast(not_equal_3, paddle.int64)
        del not_equal_3

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_4 = paddle._C_ops.equal(cast_9, full_2)
        del cast_9

        # pd_op.gather: (-1x4xf32) <- (100x4xf32, -1xi64, 1xi32)
        gather_4 = paddle._C_ops.gather(slice_0, data_8, full_6)
        del data_8

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_14 = full_int_array_7

        # pd_op.slice: (100x4xf32) <- (4x100x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_26, [0], full_int_array_6, full_int_array_7, [1], [0]
        )

        # pd_op.gather: (1x4xf32) <- (100x4xf32, 1xi64, 1xi32)
        gather_5 = paddle._C_ops.gather(slice_1, data_10, full_6)
        del data_10

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_15 = full_int_array_8

        # pd_op.slice: (100x4xf32) <- (4x100x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            data_26, [0], full_int_array_7, full_int_array_8, [1], [0]
        )

        # pd_op.greater_than: (xb) <- (xi64, xi64)
        greater_than_2 = paddle._C_ops.greater_than(data_5, full_2)
        del data_5

        # pd_op.cast: (xi64) <- (xb)
        cast_10 = paddle._C_ops.cast(greater_than_2, paddle.int64)
        del greater_than_2

        # pd_op.not_equal: (xb) <- (xi64, xi64)
        not_equal_4 = paddle._C_ops.not_equal(cast_10, full_2)
        del cast_10

        # pd_op.cast: (xi64) <- (xb)
        cast_11 = paddle._C_ops.cast(not_equal_4, paddle.int64)
        del not_equal_4

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_5 = paddle._C_ops.equal(cast_11, full_2)
        del cast_11

        # pd_op.gather: (-1x4xf32) <- (100x4xf32, -1xi64, 1xi32)
        gather_6 = paddle._C_ops.gather(slice_2, data_12, full_6)
        del data_12

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [4]

        # pd_op.slice: (100x4xf32) <- (4x100x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            data_26, [0], full_int_array_8, full_int_array_9, [1], [0]
        )
        del data_26

        # pd_op.gather: (1x4xf32) <- (100x4xf32, 1xi64, 1xi32)
        gather_7 = paddle._C_ops.gather(slice_3, data_14, full_6)
        del data_14

        # builtin.combine: ([-1x4xf32, 1x4xf32, -1x4xf32, 1x4xf32]) <- (-1x4xf32, 1x4xf32, -1x4xf32, 1x4xf32)
        combine_3 = [gather_4, gather_5, gather_6, gather_7]

        # pd_op.concat: (-1x4xf32) <- ([-1x4xf32, 1x4xf32, -1x4xf32, 1x4xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_3, full_6)
        del combine_3

        # pd_op.greater_than: (xb) <- (xi64, xi64)
        greater_than_3 = paddle._C_ops.greater_than(data_6, full_2)
        del data_6

        # pd_op.cast: (xi64) <- (xb)
        cast_12 = paddle._C_ops.cast(greater_than_3, paddle.int64)
        del greater_than_3

        # pd_op.not_equal: (xb) <- (xi64, xi64)
        not_equal_5 = paddle._C_ops.not_equal(cast_12, full_2)
        del cast_12

        # pd_op.cast: (xi64) <- (xb)
        cast_13 = paddle._C_ops.cast(not_equal_5, paddle.int64)
        del not_equal_5

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_6 = paddle._C_ops.equal(cast_13, full_2)
        del cast_13

        # pd_op.gather: (-1x4xf32) <- (-1x4xf32, -1xi64, 1xi32)
        gather_8 = paddle._C_ops.gather(data_16, data_9, full_6)
        del data_16, data_9

        # pd_op.gather: (1x4xf32) <- (1x4xf32, 1xi64, 1xi32)
        gather_9 = paddle._C_ops.gather(data_17, data_11, full_6)
        del data_11, data_17

        # pd_op.greater_than: (xb) <- (xi64, xi64)
        greater_than_4 = paddle._C_ops.greater_than(data_7, full_2)
        del data_7

        # pd_op.cast: (xi64) <- (xb)
        cast_14 = paddle._C_ops.cast(greater_than_4, paddle.int64)
        del greater_than_4

        # pd_op.not_equal: (xb) <- (xi64, xi64)
        not_equal_6 = paddle._C_ops.not_equal(cast_14, full_2)
        del cast_14

        # pd_op.cast: (xi64) <- (xb)
        cast_15 = paddle._C_ops.cast(not_equal_6, paddle.int64)
        del not_equal_6

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_7 = paddle._C_ops.equal(cast_15, full_2)
        del cast_15, full_2

        # pd_op.gather: (-1x4xf32) <- (-1x4xf32, -1xi64, 1xi32)
        gather_10 = paddle._C_ops.gather(data_18, data_13, full_6)
        del data_13, data_18

        # pd_op.gather: (1x4xf32) <- (1x4xf32, 1xi64, 1xi32)
        gather_11 = paddle._C_ops.gather(data_19, data_15, full_6)
        del data_15, data_19

        # builtin.combine: ([-1x4xf32, 1x4xf32, -1x4xf32, 1x4xf32]) <- (-1x4xf32, 1x4xf32, -1x4xf32, 1x4xf32)
        combine_4 = [gather_8, gather_9, gather_10, gather_11]
        del gather_10, gather_11, gather_8, gather_9

        # pd_op.concat: (-1x4xf32) <- ([-1x4xf32, 1x4xf32, -1x4xf32, 1x4xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_4, full_6)
        del combine_4, full_6

        # pd_op.subtract: (-1x4xf32) <- (-1x4xf32, -1x4xf32)
        subtract_0 = paddle._C_ops.subtract(concat_3, concat_4)

        # pd_op.abs: (-1x4xf32) <- (-1x4xf32)
        abs_0 = paddle._C_ops.abs(subtract_0)

        # pd_op.sum: (xf32) <- (-1x4xf32, 0xi64)
        sum_2 = paddle._C_ops.sum(abs_0, full_int_array_4, None, False)

        # pd_op.full: (1xf32) <- ()
        full_10 = paddle._C_ops.full(
            [1], float("5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(sum_2, full_10, float("0"), True)
        del sum_2

        # pd_op.divide: (1xf32) <- (xf32, 1xf32)
        divide_0 = paddle._C_ops.divide(scale_8, data_25)

        # pd_op.full: (1xi32) <- ()
        full_11 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_16 = full_11

        # pd_op.split_with_num: ([-1x2xf32, -1x2xf32]) <- (-1x4xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(concat_3, 2, full_11)

        # builtin.split: (-1x2xf32, -1x2xf32) <- ([-1x2xf32, -1x2xf32])
        (
            split_0,
            split_1,
        ) = split_with_num_0
        del split_with_num_0

        # pd_op.full: (1xf32) <- ()
        full_12 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_17 = full_12

        # pd_op.scale: (-1x2xf32) <- (-1x2xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(split_1, full_12, float("0"), True)
        del split_1

        # pd_op.assign: (-1x2xf32) <- (-1x2xf32)
        assign_18 = scale_9

        # pd_op.subtract: (-1x2xf32) <- (-1x2xf32, -1x2xf32)
        subtract_1 = paddle._C_ops.subtract(split_0, scale_9)

        # pd_op.add: (-1x2xf32) <- (-1x2xf32, -1x2xf32)
        add_4 = paddle._C_ops.add(split_0, scale_9)

        # pd_op.full: (1xi32) <- ()
        full_13 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x2xf32, -1x2xf32]) <- (-1x2xf32, -1x2xf32)
        combine_5 = [subtract_1, add_4]

        # pd_op.concat: (-1x4xf32) <- ([-1x2xf32, -1x2xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_5, full_13)
        del combine_5

        # pd_op.split_with_num: ([-1x2xf32, -1x2xf32]) <- (-1x4xf32, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(concat_4, 2, full_11)

        # builtin.split: (-1x2xf32, -1x2xf32) <- ([-1x2xf32, -1x2xf32])
        (
            split_2,
            split_3,
        ) = split_with_num_1
        del split_with_num_1

        # pd_op.scale: (-1x2xf32) <- (-1x2xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(split_3, full_12, float("0"), True)
        del split_3

        # pd_op.subtract: (-1x2xf32) <- (-1x2xf32, -1x2xf32)
        subtract_2 = paddle._C_ops.subtract(split_2, scale_10)

        # pd_op.add: (-1x2xf32) <- (-1x2xf32, -1x2xf32)
        add_5 = paddle._C_ops.add(split_2, scale_10)
        del scale_10, split_2

        # builtin.combine: ([-1x2xf32, -1x2xf32]) <- (-1x2xf32, -1x2xf32)
        combine_6 = [subtract_2, add_5]
        del add_5, subtract_2

        # pd_op.concat: (-1x4xf32) <- ([-1x2xf32, -1x2xf32], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_6, full_13)
        del combine_6

        # pd_op.split_with_num: ([-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32]) <- (-1x4xf32, 1xi32)
        split_with_num_2 = paddle._C_ops.split_with_num(concat_5, 4, full_11)
        del concat_5

        # builtin.split: (-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32) <- ([-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32])
        (
            split_4,
            split_5,
            split_6,
            split_7,
        ) = split_with_num_2
        del split_with_num_2

        # pd_op.split_with_num: ([-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32]) <- (-1x4xf32, 1xi32)
        split_with_num_3 = paddle._C_ops.split_with_num(concat_6, 4, full_11)
        del concat_6

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
        full_14 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_19 = full_14

        # pd_op.clip: (-1x1xf32) <- (-1x1xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(subtract_3, full_3, full_14)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_4 = paddle._C_ops.subtract(minimum_1, maximum_1)

        # pd_op.clip: (-1x1xf32) <- (-1x1xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(subtract_4, full_3, full_14)
        del full_3

        # pd_op.multiply: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        multiply_3 = paddle._C_ops.multiply(clip_0, clip_1)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_5 = paddle._C_ops.subtract(split_6, split_4)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_6 = paddle._C_ops.subtract(split_7, split_5)

        # pd_op.multiply: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        multiply_4 = paddle._C_ops.multiply(subtract_5, subtract_6)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_7 = paddle._C_ops.subtract(split_10, split_8)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_8 = paddle._C_ops.subtract(split_11, split_9)

        # pd_op.multiply: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        multiply_5 = paddle._C_ops.multiply(subtract_7, subtract_8)
        del subtract_7, subtract_8

        # pd_op.add: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        add_6 = paddle._C_ops.add(multiply_4, multiply_5)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_9 = paddle._C_ops.subtract(add_6, multiply_3)

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(subtract_9, full_1, float("1e-10"), True)
        del subtract_9

        # pd_op.divide: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        divide_2 = paddle._C_ops.divide(multiply_3, scale_11)

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
        multiply_6 = paddle._C_ops.multiply(subtract_10, subtract_11)

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(multiply_6, full_1, float("1e-10"), True)
        del multiply_6

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_12 = paddle._C_ops.subtract(scale_12, scale_11)

        # pd_op.divide: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        divide_3 = paddle._C_ops.divide(subtract_12, scale_12)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_13 = paddle._C_ops.subtract(divide_2, divide_3)

        # pd_op.full: (1xf32) <- ()
        full_15 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(subtract_13, full_15, float("1"), True)
        del subtract_13

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(scale_13, full_1, float("0"), True)
        del full_1, scale_13

        # pd_op.sum: (xf32) <- (-1x1xf32, 0xi64)
        sum_3 = paddle._C_ops.sum(scale_14, full_int_array_4, None, False)

        # pd_op.divide: (1xf32) <- (xf32, 1xf32)
        divide_4 = paddle._C_ops.divide(sum_3, data_25)
        del data_25

        # pd_op.scale: (1xf32) <- (1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(divide_4, full_4, float("0"), True)
        del (
            abs_0,
            add_2,
            add_4,
            add_6,
            assign_0,
            assign_1,
            assign_10,
            assign_11,
            assign_12,
            assign_13,
            assign_14,
            assign_15,
            assign_16,
            assign_17,
            assign_18,
            assign_19,
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
            concat_3,
            concat_4,
            cross_entropy_with_softmax_1,
            divide_2,
            divide_3,
            divide_4,
            full_10,
            full_11,
            full_12,
            full_13,
            full_14,
            full_15,
            full_4,
            full_int_array_4,
            full_int_array_5,
            full_int_array_6,
            full_int_array_7,
            full_int_array_8,
            full_int_array_9,
            gather_4,
            gather_5,
            gather_6,
            gather_7,
            maximum_0,
            maximum_1,
            maximum_2,
            maximum_3,
            minimum_0,
            minimum_1,
            minimum_2,
            minimum_3,
            multiply_2,
            multiply_3,
            multiply_4,
            multiply_5,
            reshape_2,
            scale_11,
            scale_12,
            scale_14,
            scale_8,
            scale_9,
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
            sum_0,
            sum_3,
            unsqueeze_0,
        )

        return cross_entropy_with_softmax_0, divide_0, scale_0, divide_1
