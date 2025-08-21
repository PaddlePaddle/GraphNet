import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (1xf32) <- (1xf32)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_0,
            [1],
            paddle.float32,
            [float("0")],
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (4xi64) <- ()
        full_1 = paddle._C_ops.full(
            [4], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (4xi64) <- (4xi64)
        assign_value__1 = paddle._C_ops.assign_value_(
            full_1,
            [4],
            paddle.int64,
            [float("1"), float("0"), float("1"), float("0")],
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [-1]

        # pd_op.unsqueeze: (4x1xi64) <- (4xi64, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(assign_value__1, full_int_array_0)

        # pd_op.gather_nd: (4xi64) <- (2xi64, 4x1xi64)
        gather_nd_0 = paddle._C_ops.gather_nd(data_1, unsqueeze_1)

        # pd_op.cast: (4xf32) <- (4xi64)
        cast_0 = paddle._C_ops.cast(gather_nd_0, paddle.float32)

        # pd_op.divide: (1x8400x4xf32) <- (1x8400x4xf32, 4xf32)
        divide_0 = paddle._C_ops.divide(data_0, cast_0)

        # pd_op.shape64: (3xi64) <- (1x8400x4xf32)
        shape64_0 = paddle._C_ops.shape64(divide_0)

        # pd_op.empty: (1x8400x4xf32) <- (3xi64)
        empty_0 = paddle._C_ops.empty(
            shape64_0, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [1]

        # pd_op.slice: (1x8400xf32) <- (1x8400x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            divide_0, [2], full_int_array_1, full_int_array_2, [1], [2]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [3]

        # pd_op.slice: (1x8400xf32) <- (1x8400x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            divide_0, [2], full_int_array_3, full_int_array_4, [1], [2]
        )

        # pd_op.add: (1x8400xf32) <- (1x8400xf32, 1x8400xf32)
        add_0 = paddle._C_ops.add(slice_0, slice_1)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x8400xf32) <- (1x8400xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(add_0, full_2, float("0"), True)

        # pd_op.set_value_with_tensor_: (1x8400x4xf32) <- (1x8400x4xf32, 1x8400xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__1 = paddle._C_ops.set_value_with_tensor_(
            empty_0,
            scale_0,
            full_int_array_1,
            full_int_array_2,
            full_int_array_2,
            [2],
            [2],
            [],
        )

        # pd_op.slice: (1x8400xf32) <- (1x8400x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            divide_0, [2], full_int_array_2, full_int_array_3, [1], [2]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [4]

        # pd_op.slice: (1x8400xf32) <- (1x8400x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            divide_0, [2], full_int_array_4, full_int_array_5, [1], [2]
        )

        # pd_op.add: (1x8400xf32) <- (1x8400xf32, 1x8400xf32)
        add_1 = paddle._C_ops.add(slice_2, slice_3)

        # pd_op.scale: (1x8400xf32) <- (1x8400xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(add_1, full_2, float("0"), True)

        # pd_op.set_value_with_tensor_: (1x8400x4xf32) <- (1x8400x4xf32, 1x8400xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__2 = paddle._C_ops.set_value_with_tensor_(
            set_value_with_tensor__1,
            scale_1,
            full_int_array_2,
            full_int_array_3,
            full_int_array_2,
            [2],
            [2],
            [],
        )

        # pd_op.slice: (1x8400xf32) <- (1x8400x4xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            divide_0, [2], full_int_array_3, full_int_array_4, [1], [2]
        )

        # pd_op.slice: (1x8400xf32) <- (1x8400x4xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            divide_0, [2], full_int_array_1, full_int_array_2, [1], [2]
        )

        # pd_op.subtract: (1x8400xf32) <- (1x8400xf32, 1x8400xf32)
        subtract_0 = paddle._C_ops.subtract(slice_4, slice_5)

        # pd_op.set_value_with_tensor_: (1x8400x4xf32) <- (1x8400x4xf32, 1x8400xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__3 = paddle._C_ops.set_value_with_tensor_(
            set_value_with_tensor__2,
            subtract_0,
            full_int_array_3,
            full_int_array_4,
            full_int_array_2,
            [2],
            [2],
            [],
        )

        # pd_op.slice: (1x8400xf32) <- (1x8400x4xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            divide_0, [2], full_int_array_4, full_int_array_5, [1], [2]
        )

        # pd_op.slice: (1x8400xf32) <- (1x8400x4xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            divide_0, [2], full_int_array_2, full_int_array_3, [1], [2]
        )

        # pd_op.subtract: (1x8400xf32) <- (1x8400xf32, 1x8400xf32)
        subtract_1 = paddle._C_ops.subtract(slice_6, slice_7)

        # pd_op.set_value_with_tensor_: (1x8400x4xf32) <- (1x8400x4xf32, 1x8400xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__0 = paddle._C_ops.set_value_with_tensor_(
            set_value_with_tensor__3,
            subtract_1,
            full_int_array_4,
            full_int_array_5,
            full_int_array_2,
            [2],
            [2],
            [],
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [2147483647]

        # pd_op.slice: (1x8400x2xf32) <- (1x8400x4xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            set_value_with_tensor__0, [2], full_int_array_3, full_int_array_6, [1], []
        )

        # pd_op.prod: (1x8400xf32) <- (1x8400x2xf32, 1xi64)
        prod_0 = paddle._C_ops.prod(slice_8, full_int_array_3, False, False)

        # pd_op.unsqueeze: (1x8400x1xf32) <- (1x8400xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(prod_0, full_int_array_0)

        # pd_op.full: (4xi64) <- ()
        full_3 = paddle._C_ops.full(
            [4], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (4xi64) <- (4xi64)
        assign_value__2 = paddle._C_ops.assign_value_(
            full_3,
            [4],
            paddle.int64,
            [float("160"), float("160"), float("160"), float("160")],
            paddle.framework._current_expected_place(),
        )

        # pd_op.cast: (4xf32) <- (4xi64)
        cast_1 = paddle._C_ops.cast(assign_value__2, paddle.float32)

        # pd_op.multiply: (1x8400x4xf32) <- (1x8400x4xf32, 4xf32)
        multiply_0 = paddle._C_ops.multiply(divide_0, cast_1)
        return multiply_0, unsqueeze_0, assign_value__0, set_value_with_tensor__0
