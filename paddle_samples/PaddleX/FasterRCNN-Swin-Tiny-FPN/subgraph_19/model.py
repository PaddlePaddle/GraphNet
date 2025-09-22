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
    ):
        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [3, 3]

        # pd_op.roll: (2x21x28x768xf32) <- (2x21x28x768xf32, 2xi64)
        roll_0 = paddle._C_ops.roll(data_0, full_int_array_0, [1, 2])
        del data_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [0, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [20, 24]

        # pd_op.slice: (2x20x24x768xf32) <- (2x21x28x768xf32, 2xi64, 2xi64)
        slice_0 = paddle._C_ops.slice(
            roll_0, [1, 2], full_int_array_1, full_int_array_2, [1, 1], []
        )

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_3 = [-1, 480, 768]

        # pd_op.reshape: (2x480x768xf32) <- (2x20x24x768xf32, 3xi64)
        reshape_0 = paddle._C_ops.reshape(slice_0, full_int_array_3)
        del full_int_array_3

        # pd_op.full: (xf64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_0,
            [],
            paddle.float64,
            [float("0.9")],
            paddle.framework._current_expected_place(),
        )
        del full_0

        # pd_op.cast: (xf32) <- (xf64)
        cast_0 = paddle._C_ops.cast(assign_value__0, paddle.float32)
        del assign_value__0

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_4 = [2, 1, 1]

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_0 = paddle._C_ops.uniform(
            full_int_array_4,
            paddle.float32,
            full_1,
            full_2,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_1 = paddle._C_ops.add(cast_0, uniform_0)
        del uniform_0

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_0 = paddle._C_ops.floor(add_1)
        del add_1

        # pd_op.divide: (2x480x768xf32) <- (2x480x768xf32, xf32)
        divide_0 = paddle._C_ops.divide(reshape_0, cast_0)

        # pd_op.multiply: (2x480x768xf32) <- (2x480x768xf32, 2x1x1xf32)
        multiply_0 = paddle._C_ops.multiply(divide_0, floor_0)

        # pd_op.add: (2x480x768xf32) <- (2x480x768xf32, 2x480x768xf32)
        add_2 = paddle._C_ops.add(data_1, multiply_0)
        del data_1

        # pd_op.layer_norm: (2x480x768xf32, 2x480xf32, 2x480xf32) <- (2x480x768xf32, 768xf32, 768xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_2, parameter_5, parameter_4, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_4, parameter_5

        # pd_op.matmul: (2x480x3072xf32) <- (2x480x768xf32, 768x3072xf32)
        matmul_0 = paddle._C_ops.matmul(layer_norm_0, parameter_3, False, False)
        del parameter_3

        # pd_op.add: (2x480x3072xf32) <- (2x480x3072xf32, 3072xf32)
        add_3 = paddle._C_ops.add(matmul_0, parameter_2)
        del parameter_2

        # pd_op.gelu: (2x480x3072xf32) <- (2x480x3072xf32)
        gelu_0 = paddle._C_ops.gelu(add_3, False)

        # pd_op.matmul: (2x480x768xf32) <- (2x480x3072xf32, 3072x768xf32)
        matmul_1 = paddle._C_ops.matmul(gelu_0, parameter_1, False, False)
        del parameter_1

        # pd_op.add: (2x480x768xf32) <- (2x480x768xf32, 768xf32)
        add_4 = paddle._C_ops.add(matmul_1, parameter_0)
        del parameter_0

        # pd_op.full: (xf64) <- ()
        full_3 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__1 = paddle._C_ops.assign_value_(
            full_3,
            [],
            paddle.float64,
            [float("0.9")],
            paddle.framework._current_expected_place(),
        )
        del full_3

        # pd_op.cast: (xf32) <- (xf64)
        cast_1 = paddle._C_ops.cast(assign_value__1, paddle.float32)
        del assign_value__1

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_1 = paddle._C_ops.uniform(
            full_int_array_4,
            paddle.float32,
            full_1,
            full_2,
            0,
            paddle.framework._current_expected_place(),
        )
        del full_1, full_2, full_int_array_4

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_5 = paddle._C_ops.add(cast_1, uniform_1)
        del uniform_1

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_1 = paddle._C_ops.floor(add_5)
        del add_5

        # pd_op.divide: (2x480x768xf32) <- (2x480x768xf32, xf32)
        divide_1 = paddle._C_ops.divide(add_4, cast_1)

        # pd_op.multiply: (2x480x768xf32) <- (2x480x768xf32, 2x1x1xf32)
        multiply_1 = paddle._C_ops.multiply(divide_1, floor_1)

        # pd_op.add: (2x480x768xf32) <- (2x480x768xf32, 2x480x768xf32)
        add_0 = paddle._C_ops.add(add_2, multiply_1)
        del (
            add_2,
            add_3,
            add_4,
            cast_0,
            cast_1,
            divide_0,
            divide_1,
            floor_0,
            floor_1,
            full_int_array_0,
            full_int_array_1,
            full_int_array_2,
            gelu_0,
            layer_norm_0,
            layer_norm_1,
            layer_norm_2,
            matmul_0,
            matmul_1,
            multiply_0,
            multiply_1,
            reshape_0,
            roll_0,
            slice_0,
        )

        return add_0
