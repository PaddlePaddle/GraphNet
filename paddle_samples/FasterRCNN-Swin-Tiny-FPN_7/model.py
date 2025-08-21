import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0):
        # pd_op.full: (xf64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_0,
            [],
            paddle.float64,
            [float("0.927273")],
            paddle.framework._current_expected_place(),
        )

        # pd_op.cast: (xf32) <- (xf64)
        cast_0 = paddle._C_ops.cast(assign_value__0, paddle.float32)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_0 = [2, 1, 1]

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
            full_int_array_0,
            paddle.float32,
            full_1,
            full_2,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_0 = paddle._C_ops.add(cast_0, uniform_0)

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_0 = paddle._C_ops.floor(add_0)

        # pd_op.divide: (2x2352x384xf32) <- (2x2352x384xf32, xf32)
        divide_0 = paddle._C_ops.divide(data_0, cast_0)

        # pd_op.multiply: (2x2352x384xf32) <- (2x2352x384xf32, 2x1x1xf32)
        multiply_0 = paddle._C_ops.multiply(divide_0, floor_0)
        return cast_0, floor_0, divide_0, multiply_0
