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
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (1xi64) <- (1xi64, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            data_5, full_1, paddle.int64, paddle.framework._current_expected_place()
        )
        del full_1

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (1xi64) <- (1xi64, 1xf32)
        full_like_1 = paddle._C_ops.full_like(
            data_7, full_2, paddle.int64, paddle.framework._current_expected_place()
        )
        del full_2

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (2xi64) <- (2xi64, 1xf32)
        full_like_2 = paddle._C_ops.full_like(
            data_9, full_3, paddle.int64, paddle.framework._current_expected_place()
        )
        del full_3

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("3"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (1xi64) <- (1xi64, 1xf32)
        full_like_3 = paddle._C_ops.full_like(
            data_11, full_4, paddle.int64, paddle.framework._current_expected_place()
        )
        del full_4

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([1xi64, 1xi64, 2xi64, 1xi64]) <- (1xi64, 1xi64, 2xi64, 1xi64)
        combine_0 = [full_like_0, full_like_1, full_like_2, full_like_3]
        del full_like_0, full_like_1, full_like_2, full_like_3

        # pd_op.concat: (5xi64) <- ([1xi64, 1xi64, 2xi64, 1xi64], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_5)
        del combine_0

        # builtin.combine: ([1xi64, 1xi64, 2xi64, 1xi64]) <- (1xi64, 1xi64, 2xi64, 1xi64)
        combine_1 = [data_5, data_7, data_9, data_11]
        del data_11, data_5, data_7, data_9

        # pd_op.concat: (5xi64) <- ([1xi64, 1xi64, 2xi64, 1xi64], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_5)
        del combine_1

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("100"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (5xi64) <- (5xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(concat_0, full_6, float("0"), True)
        del concat_0, full_6

        # pd_op.add: (5xi64) <- (5xi64, 5xi64)
        add_0 = paddle._C_ops.add(concat_1, scale_0)
        del concat_1, scale_0

        # pd_op.gather: (1x1xi32) <- (1x1xi32, 1xi64, 1xi32)
        gather_0 = paddle._C_ops.gather(data_1, data_6, full_5)
        del data_1, data_6

        # pd_op.gather: (1x1xi32) <- (1x1xi32, 1xi64, 1xi32)
        gather_1 = paddle._C_ops.gather(data_2, data_8, full_5)
        del data_2, data_8

        # pd_op.gather: (2x1xi32) <- (2x1xi32, 2xi64, 1xi32)
        gather_2 = paddle._C_ops.gather(data_3, data_10, full_5)
        del data_10, data_3

        # pd_op.gather: (1x1xi32) <- (1x1xi32, 1xi64, 1xi32)
        gather_3 = paddle._C_ops.gather(data_4, data_12, full_5)
        del data_12, data_4

        # builtin.combine: ([1x1xi32, 1x1xi32, 2x1xi32, 1x1xi32]) <- (1x1xi32, 1x1xi32, 2x1xi32, 1x1xi32)
        combine_2 = [gather_0, gather_1, gather_2, gather_3]
        del gather_0, gather_1, gather_2, gather_3

        # pd_op.concat: (5x1xi32) <- ([1x1xi32, 1x1xi32, 2x1xi32, 1x1xi32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_5)
        del combine_2, full_5

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [-1, 1]

        # pd_op.reshape: (400x1xi64) <- (4x100xi64, 2xi64)
        reshape_0 = paddle._C_ops.reshape(full_0, full_int_array_0)
        del full_0, full_int_array_0

        # pd_op.cast: (5x1xi64) <- (5x1xi32)
        cast_0 = paddle._C_ops.cast(concat_2, paddle.int64)
        del concat_2

        # pd_op.scatter: (400x1xi64) <- (400x1xi64, 5xi64, 5x1xi64)
        scatter_0 = paddle._C_ops.scatter(reshape_0, add_0, cast_0, True)
        del add_0, cast_0, reshape_0

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
                data_0, unsqueeze_0, False, True, True, -100, -1
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del data_0

        # pd_op.full: (xi64) <- ()
        full_7 = paddle._C_ops.full(
            [], float("-100"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.not_equal: (4x100x1xb) <- (4x100x1xi64, xi64)
        not_equal_0 = paddle._C_ops.not_equal(unsqueeze_0, full_7)
        del full_7

        # pd_op.cast: (4x100x1xi64) <- (4x100x1xb)
        cast_1 = paddle._C_ops.cast(not_equal_0, paddle.int64)

        # pd_op.multiply: (4x100x1xi64) <- (4x100x1xi64, 4x100x1xi64)
        multiply_0 = paddle._C_ops.multiply(cast_1, unsqueeze_0)
        del cast_1

        # pd_op.cast: (4x100x1xf32) <- (4x100x1xb)
        cast_2 = paddle._C_ops.cast(not_equal_0, paddle.float32)
        del not_equal_0

        # pd_op.squeeze: (4x100xf32) <- (4x100x1xf32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(cast_2, full_int_array_2)
        del cast_2, full_int_array_2

        # pd_op.gather_nd: (4x100xf32) <- (5xf32, 4x100x1xi64)
        gather_nd_0 = paddle._C_ops.gather_nd(data_13, multiply_0)
        del data_13, multiply_0

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

        # pd_op.sum: (xf32) <- (4x100x1xf32, 0xi64)
        sum_0 = paddle._C_ops.sum(multiply_2, full_int_array_4, None, False)

        # pd_op.sum: (xf32) <- (4x100x1xf32, 0xi64)
        sum_1 = paddle._C_ops.sum(reshape_2, full_int_array_4, None, False)

        # pd_op.full: (xf32) <- ()
        full_8 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.equal: (xb) <- (xf32, xf32)
        equal_0 = paddle._C_ops.equal(sum_1, full_8)
        del full_8

        # pd_op.cast: (xf32) <- (xb)
        cast_3 = paddle._C_ops.cast(equal_0, paddle.float32)
        del equal_0

        # pd_op.add: (xf32) <- (xf32, xf32)
        add_1 = paddle._C_ops.add(sum_1, cast_3)
        del cast_3, sum_1

        # pd_op.divide: (xf32) <- (xf32, xf32)
        divide_0 = paddle._C_ops.divide(sum_0, add_1)
        del (
            add_1,
            cross_entropy_with_softmax_1,
            full_int_array_4,
            multiply_2,
            reshape_2,
            sum_0,
            unsqueeze_0,
        )

        return cross_entropy_with_softmax_0, divide_0
