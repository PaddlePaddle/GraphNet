import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        parameter_0,
        parameter_1,
        parameter_2,
        parameter_3,
        parameter_4,
        parameter_5,
        data_0,
        data_1,
        data_2,
        data_3,
        data_4,
        data_5,
        data_6,
    ):
        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (xb) <- (xi64, xi64)
        greater_than_0 = paddle._C_ops.greater_than(data_4, full_0)

        # pd_op.cast: (xi64) <- (xb)
        cast_0 = paddle._C_ops.cast(greater_than_0, paddle.int64)
        del greater_than_0

        # pd_op.not_equal: (xb) <- (xi64, xi64)
        not_equal_0 = paddle._C_ops.not_equal(cast_0, full_0)
        del cast_0

        # pd_op.cast: (xi64) <- (xb)
        cast_1 = paddle._C_ops.cast(not_equal_0, paddle.int64)
        del not_equal_0

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_0 = paddle._C_ops.equal(cast_1, full_0)
        del cast_1

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_0 = [data_4, data_4]
        del data_4

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.roll: (2x-1x-1x-1xf32) <- (2x-1x-1x-1xf32, 2xi64)
        roll_0 = paddle._C_ops.roll(data_5, stack_0, [1, 2])
        del data_5

        # pd_op.greater_than: (xb) <- (xi64, xi64)
        greater_than_1 = paddle._C_ops.greater_than(data_0, full_0)
        del data_0

        # pd_op.cast: (xi64) <- (xb)
        cast_2 = paddle._C_ops.cast(greater_than_1, paddle.int64)
        del greater_than_1

        # pd_op.not_equal: (xb) <- (xi64, xi64)
        not_equal_1 = paddle._C_ops.not_equal(cast_2, full_0)
        del cast_2, full_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [0, 0]

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_1 = [data_1, data_2]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_1, 0)
        del combine_1

        # pd_op.slice: (2x-1x-1x-1xf32) <- (2x-1x-1x-1xf32, 2xi64, 2xi64)
        slice_0 = paddle._C_ops.slice(
            roll_0, [1, 2], full_int_array_0, stack_1, [-1, -1], []
        )

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_0 = paddle._C_ops.multiply(data_1, data_2)
        del data_1, data_2

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("-1"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_2 = [full_1, multiply_0, data_3]
        del data_3, full_1, multiply_0

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_2 = paddle._C_ops.stack(combine_2, 0)
        del combine_2

        # pd_op.reshape: (-1x-1x-1xf32) <- (2x-1x-1x-1xf32, 3xi64)
        reshape_0 = paddle._C_ops.reshape(slice_0, stack_2)
        del stack_2

        # pd_op.full: (xf64) <- ()
        full_2 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_2,
            [],
            paddle.float64,
            [float("0.972727")],
            paddle.framework._current_expected_place(),
        )
        del full_2

        # pd_op.cast: (xf32) <- (xf64)
        cast_3 = paddle._C_ops.cast(assign_value__0, paddle.float32)
        del assign_value__0

        # pd_op.shape64: (3xi64) <- (-1x-1x-1xf32)
        shape64_0 = paddle._C_ops.shape64(reshape_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del full_int_array_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [2]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [3]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del full_int_array_4, shape64_0

        # pd_op.full: (xi64) <- ()
        full_3 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_3 = [slice_1, full_3, full_3]
        del full_3, slice_1

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_3 = paddle._C_ops.stack(combine_3, 0)
        del combine_3

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_0 = paddle._C_ops.uniform(
            stack_3,
            paddle.float32,
            full_4,
            full_5,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_3

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_1 = paddle._C_ops.add(cast_3, uniform_0)
        del uniform_0

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_0 = paddle._C_ops.floor(add_1)
        del add_1

        # pd_op.divide: (-1x-1x-1xf32) <- (-1x-1x-1xf32, xf32)
        divide_0 = paddle._C_ops.divide(reshape_0, cast_3)

        # pd_op.multiply: (-1x-1x-1xf32) <- (-1x-1x-1xf32, -1x1x1xf32)
        multiply_1 = paddle._C_ops.multiply(divide_0, floor_0)

        # pd_op.add: (2x-1x-1xf32) <- (2x-1x-1xf32, -1x-1x-1xf32)
        add_2 = paddle._C_ops.add(data_6, multiply_1)
        del data_6

        # pd_op.layer_norm: (2x-1x-1xf32, 2x-1xf32, 2x-1xf32) <- (2x-1x-1xf32, 192xf32, 192xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_2, parameter_5, parameter_4, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_4, parameter_5

        # pd_op.matmul: (2x-1x768xf32) <- (2x-1x-1xf32, 192x768xf32)
        matmul_0 = paddle._C_ops.matmul(layer_norm_0, parameter_3, False, False)
        del parameter_3

        # pd_op.add: (2x-1x768xf32) <- (2x-1x768xf32, 768xf32)
        add_3 = paddle._C_ops.add(matmul_0, parameter_2)
        del parameter_2

        # pd_op.gelu: (2x-1x768xf32) <- (2x-1x768xf32)
        gelu_0 = paddle._C_ops.gelu(add_3, False)

        # pd_op.matmul: (2x-1x192xf32) <- (2x-1x768xf32, 768x192xf32)
        matmul_1 = paddle._C_ops.matmul(gelu_0, parameter_1, False, False)
        del parameter_1

        # pd_op.add: (2x-1x192xf32) <- (2x-1x192xf32, 192xf32)
        add_4 = paddle._C_ops.add(matmul_1, parameter_0)
        del parameter_0

        # pd_op.full: (xf64) <- ()
        full_6 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__1 = paddle._C_ops.assign_value_(
            full_6,
            [],
            paddle.float64,
            [float("0.972727")],
            paddle.framework._current_expected_place(),
        )
        del full_6

        # pd_op.cast: (xf32) <- (xf64)
        cast_4 = paddle._C_ops.cast(assign_value__1, paddle.float32)
        del assign_value__1

        # pd_op.shape64: (3xi64) <- (2x-1x192xf32)
        shape64_1 = paddle._C_ops.shape64(add_4)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del full_int_array_2, full_int_array_3, shape64_1

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_5 = [2, 1, 1]

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_1 = paddle._C_ops.uniform(
            full_int_array_5,
            paddle.float32,
            full_4,
            full_5,
            0,
            paddle.framework._current_expected_place(),
        )
        del full_4, full_5, full_int_array_5

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_5 = paddle._C_ops.add(cast_4, uniform_1)
        del uniform_1

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_1 = paddle._C_ops.floor(add_5)
        del add_5

        # pd_op.divide: (2x-1x192xf32) <- (2x-1x192xf32, xf32)
        divide_1 = paddle._C_ops.divide(add_4, cast_4)

        # pd_op.multiply: (2x-1x192xf32) <- (2x-1x192xf32, 2x1x1xf32)
        multiply_2 = paddle._C_ops.multiply(divide_1, floor_1)

        # pd_op.add: (2x-1x192xf32) <- (2x-1x-1xf32, 2x-1x192xf32)
        add_0 = paddle._C_ops.add(add_2, multiply_2)
        del (
            add_2,
            add_3,
            add_4,
            cast_3,
            cast_4,
            divide_0,
            divide_1,
            floor_0,
            floor_1,
            full_int_array_0,
            gelu_0,
            layer_norm_0,
            layer_norm_1,
            layer_norm_2,
            matmul_0,
            matmul_1,
            multiply_1,
            multiply_2,
            reshape_0,
            roll_0,
            slice_0,
            stack_0,
            stack_1,
        )

        return add_0
