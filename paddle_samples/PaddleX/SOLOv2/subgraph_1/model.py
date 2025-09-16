import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4):
        # pd_op.argsort: (127xf32, 127xi64) <- (127xf32)
        argsort_0, argsort_1 = (lambda x, f: f(x))(
            paddle._C_ops.argsort(data_3, -1, True, False),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.gather: (127x200x232xf32) <- (127x200x232xf32, 127xi64, 1xi32)
        gather_3 = paddle._C_ops.gather(data_1, argsort_1, full_0)
        del data_1

        # pd_op.gather: (127x200x232xf32) <- (127x200x232xf32, 127xi64, 1xi32)
        gather_4 = paddle._C_ops.gather(data_0, argsort_1, full_0)
        del data_0

        # pd_op.gather: (127xf32) <- (127xf32, 127xi64, 1xi32)
        gather_5 = paddle._C_ops.gather(data_4, argsort_1, full_0)
        del data_4

        # pd_op.gather: (127xf32) <- (127xf32, 127xi64, 1xi32)
        gather_6 = paddle._C_ops.gather(data_3, argsort_1, full_0)
        del data_3

        # pd_op.gather: (127xi64) <- (127xi64, 127xi64, 1xi32)
        gather_7 = paddle._C_ops.gather(data_2, argsort_1, full_0)
        del argsort_1, data_2

        # pd_op.flatten: (127x46400xf32) <- (127x200x232xf32)
        flatten_0 = paddle._C_ops.flatten(gather_3, 1, 2)
        del gather_3

        # pd_op.transpose: (46400x127xf32) <- (127x46400xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_0, [1, 0])

        # pd_op.matmul: (127x127xf32) <- (127x46400xf32, 46400x127xf32)
        matmul_0 = paddle._C_ops.matmul(flatten_0, transpose_0, False, False)
        del flatten_0, transpose_0

        # pd_op.full: (1xi64) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (1xi64) <- (1xi64)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_1,
            [1],
            paddle.int64,
            [float("127")],
            paddle.framework._current_expected_place(),
        )
        del full_1

        # pd_op.cast: (1xi32) <- (1xi64)
        cast_0 = paddle._C_ops.cast(assign_value__0, paddle.int32)
        del assign_value__0

        # pd_op.cast: (1xi64) <- (1xi32)
        cast_1 = paddle._C_ops.cast(cast_0, paddle.int64)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_0 = []

        # pd_op.reshape: (xi64) <- (1xi64, 0xi64)
        reshape_0 = paddle._C_ops.reshape(cast_1, full_int_array_0)
        del cast_1

        # pd_op.cast: (1xi64) <- (1xi32)
        cast_2 = paddle._C_ops.cast(cast_0, paddle.int64)

        # pd_op.reshape: (xi64) <- (1xi64, 0xi64)
        reshape_1 = paddle._C_ops.reshape(cast_2, full_int_array_0)
        del cast_2

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_0 = [reshape_0, reshape_1]
        del reshape_0, reshape_1

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.expand: (-1x-1xf32) <- (127xf32, 2xi64)
        expand_0 = paddle._C_ops.expand(gather_5, stack_0)
        del gather_5, stack_0

        # pd_op.transpose: (-1x-1xf32) <- (-1x-1xf32)
        transpose_1 = paddle._C_ops.transpose(expand_0, [1, 0])

        # pd_op.add: (-1x-1xf32) <- (-1x-1xf32, -1x-1xf32)
        add_0 = paddle._C_ops.add(expand_0, transpose_1)
        del expand_0, transpose_1

        # pd_op.subtract: (127x127xf32) <- (-1x-1xf32, 127x127xf32)
        subtract_0 = paddle._C_ops.subtract(add_0, matmul_0)
        del add_0

        # pd_op.divide: (127x127xf32) <- (127x127xf32, 127x127xf32)
        divide_0 = paddle._C_ops.divide(matmul_0, subtract_0)
        del matmul_0, subtract_0

        # pd_op.triu: (127x127xf32) <- (127x127xf32)
        triu_0 = paddle._C_ops.triu(divide_0, 1)
        del divide_0

        # pd_op.cast: (1xi64) <- (1xi32)
        cast_3 = paddle._C_ops.cast(cast_0, paddle.int64)

        # pd_op.reshape: (xi64) <- (1xi64, 0xi64)
        reshape_2 = paddle._C_ops.reshape(cast_3, full_int_array_0)
        del cast_3

        # pd_op.cast: (1xi64) <- (1xi32)
        cast_4 = paddle._C_ops.cast(cast_0, paddle.int64)

        # pd_op.reshape: (xi64) <- (1xi64, 0xi64)
        reshape_3 = paddle._C_ops.reshape(cast_4, full_int_array_0)
        del cast_4

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_1 = [reshape_2, reshape_3]
        del reshape_2, reshape_3

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_1, 0)
        del combine_1

        # pd_op.expand: (-1x-1xi64) <- (127xi64, 2xi64)
        expand_1 = paddle._C_ops.expand(gather_7, stack_1)
        del stack_1

        # pd_op.transpose: (-1x-1xi64) <- (-1x-1xi64)
        transpose_2 = paddle._C_ops.transpose(expand_1, [1, 0])

        # pd_op.equal: (-1x-1xb) <- (-1x-1xi64, -1x-1xi64)
        equal_0 = paddle._C_ops.equal(expand_1, transpose_2)
        del expand_1, transpose_2

        # pd_op.cast: (-1x-1xf32) <- (-1x-1xb)
        cast_5 = paddle._C_ops.cast(equal_0, paddle.float32)
        del equal_0

        # pd_op.triu: (-1x-1xf32) <- (-1x-1xf32)
        triu_1 = paddle._C_ops.triu(cast_5, 1)
        del cast_5

        # pd_op.multiply: (127x127xf32) <- (127x127xf32, -1x-1xf32)
        multiply_0 = paddle._C_ops.multiply(triu_0, triu_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [0]

        # pd_op.max: (127xf32) <- (127x127xf32, 1xi64)
        max_0 = paddle._C_ops.max(multiply_0, full_int_array_1, False)
        del multiply_0

        # pd_op.cast: (1xi64) <- (1xi32)
        cast_6 = paddle._C_ops.cast(cast_0, paddle.int64)

        # pd_op.reshape: (xi64) <- (1xi64, 0xi64)
        reshape_4 = paddle._C_ops.reshape(cast_6, full_int_array_0)
        del cast_6

        # pd_op.cast: (1xi64) <- (1xi32)
        cast_7 = paddle._C_ops.cast(cast_0, paddle.int64)
        del cast_0

        # pd_op.reshape: (xi64) <- (1xi64, 0xi64)
        reshape_5 = paddle._C_ops.reshape(cast_7, full_int_array_0)
        del cast_7, full_int_array_0

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_2 = [reshape_4, reshape_5]
        del reshape_4, reshape_5

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_2 = paddle._C_ops.stack(combine_2, 0)
        del combine_2

        # pd_op.expand: (-1x-1xf32) <- (127xf32, 2xi64)
        expand_2 = paddle._C_ops.expand(max_0, stack_2)
        del max_0, stack_2

        # pd_op.transpose: (-1x-1xf32) <- (-1x-1xf32)
        transpose_3 = paddle._C_ops.transpose(expand_2, [1, 0])
        del expand_2

        # pd_op.multiply: (127x127xf32) <- (127x127xf32, -1x-1xf32)
        multiply_1 = paddle._C_ops.multiply(triu_0, triu_1)
        del triu_0, triu_1

        # pd_op.full: (xf32) <- ()
        full_2 = paddle._C_ops.full(
            [], float("2"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.elementwise_pow: (127x127xf32) <- (127x127xf32, xf32)
        elementwise_pow_0 = paddle._C_ops.elementwise_pow(multiply_1, full_2)
        del multiply_1

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("-2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (127x127xf32) <- (127x127xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(elementwise_pow_0, full_3, float("0"), True)
        del elementwise_pow_0

        # pd_op.exp: (127x127xf32) <- (127x127xf32)
        exp_0 = paddle._C_ops.exp(scale_0)
        del scale_0

        # pd_op.elementwise_pow: (-1x-1xf32) <- (-1x-1xf32, xf32)
        elementwise_pow_1 = paddle._C_ops.elementwise_pow(transpose_3, full_2)
        del full_2, transpose_3

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(elementwise_pow_1, full_3, float("0"), True)
        del elementwise_pow_1, full_3

        # pd_op.exp: (-1x-1xf32) <- (-1x-1xf32)
        exp_1 = paddle._C_ops.exp(scale_1)
        del scale_1

        # pd_op.divide: (127x127xf32) <- (127x127xf32, -1x-1xf32)
        divide_1 = paddle._C_ops.divide(exp_0, exp_1)
        del exp_0, exp_1

        # pd_op.min: (127xf32) <- (127x127xf32, 1xi64)
        min_0 = paddle._C_ops.min(divide_1, full_int_array_1, False)
        del divide_1, full_int_array_1

        # pd_op.multiply: (127xf32) <- (127xf32, 127xf32)
        multiply_2 = paddle._C_ops.multiply(gather_6, min_0)
        del gather_6, min_0

        # pd_op.full: (127xf32) <- ()
        full_4 = paddle._C_ops.full(
            [127],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (xf32) <- ()
        full_5 = paddle._C_ops.full(
            [],
            float("0.05"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.greater_equal: (127xb) <- (127xf32, xf32)
        greater_equal_0 = paddle._C_ops.greater_equal(multiply_2, full_5)
        del full_5

        # pd_op.where: (127xf32) <- (127xb, 127xf32, 127xf32)
        where_0 = paddle._C_ops.where(greater_equal_0, multiply_2, full_4)
        del full_4, greater_equal_0

        # pd_op.nonzero: (-1x1xi64) <- (127xf32)
        nonzero_0 = paddle._C_ops.nonzero(where_0)
        del where_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [1]

        # pd_op.squeeze: (-1xi64) <- (-1x1xi64, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(nonzero_0, full_int_array_2)
        del full_int_array_2, nonzero_0

        # pd_op.shape64: (1xi64) <- (127xf32)
        shape64_0 = paddle._C_ops.shape64(multiply_2)

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1xi64) <- (1xi64, 1xf32)
        scale_2 = paddle._C_ops.scale(shape64_0, full_6, float("-1"), True)
        del full_6, shape64_0

        # pd_op.cast: (1xi64) <- (1xi64)
        cast_8 = paddle._C_ops.cast(scale_2, paddle.int64)
        del scale_2

        # builtin.combine: ([-1xi64, 1xi64]) <- (-1xi64, 1xi64)
        combine_3 = [squeeze_0, cast_8]
        del cast_8, squeeze_0

        # pd_op.concat: (-1xi64) <- ([-1xi64, 1xi64], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_3, full_0)
        del combine_3

        # pd_op.gather: (-1x200x232xf32) <- (127x200x232xf32, -1xi64, 1xi32)
        gather_1 = paddle._C_ops.gather(gather_4, concat_0, full_0)
        del gather_4

        # pd_op.gather: (-1xf32) <- (127xf32, -1xi64, 1xi32)
        gather_0 = paddle._C_ops.gather(multiply_2, concat_0, full_0)
        del multiply_2

        # pd_op.gather: (-1xi64) <- (127xi64, -1xi64, 1xi32)
        gather_2 = paddle._C_ops.gather(gather_7, concat_0, full_0)
        del concat_0, full_0, gather_7

        return gather_0, gather_1, gather_2
