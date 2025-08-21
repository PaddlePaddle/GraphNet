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
            [float("0.945455")],
            paddle.framework._current_expected_place(),
        )

        # pd_op.cast: (xf32) <- (xf64)
        cast_0 = paddle._C_ops.cast(assign_value__0, paddle.float32)

        # pd_op.shape64: (3xi64) <- (2x-1x384xf32)
        shape64_0 = paddle._C_ops.shape64(data_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [2]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_2 = [2, 1, 1]

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
            full_int_array_2,
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

        # pd_op.divide: (2x-1x384xf32) <- (2x-1x384xf32, xf32)
        divide_0 = paddle._C_ops.divide(data_0, cast_0)

        # pd_op.multiply: (2x-1x384xf32) <- (2x-1x384xf32, 2x1x1xf32)
        multiply_0 = paddle._C_ops.multiply(divide_0, floor_0)
        return cast_0, floor_0, divide_0, multiply_0
