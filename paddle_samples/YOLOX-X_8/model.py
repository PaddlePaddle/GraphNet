import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4, data_5):
        # pd_op.cast: (2xi32) <- (2xi64)
        cast_0 = paddle._C_ops.cast(data_5, paddle.int32)

        # pd_op.bilinear_interp: (2x3x-1x-1xf32) <- (2x3x640x640xf32, 2xi32, None, None)
        bilinear_interp_0 = paddle._C_ops.bilinear_interp(
            data_4, cast_0, None, None, "NCHW", -1, -1, -1, [], "bilinear", False, 0
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [2147483647]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        # pd_op.strided_slice: (1x2xf32) <- (1x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_0 = paddle._C_ops.strided_slice(
            data_2, [1], full_int_array_0, full_int_array_1, full_int_array_2
        )

        # pd_op.multiply: (1x2xf32) <- (1x2xf32, xf32)
        multiply_0 = paddle._C_ops.multiply(strided_slice_0, data_0)

        # pd_op.set_value_with_tensor_: (1x4xf32) <- (1x4xf32, 1x2xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__2 = paddle._C_ops.set_value_with_tensor_(
            data_2,
            multiply_0,
            full_int_array_0,
            full_int_array_1,
            full_int_array_2,
            [1],
            [],
            [],
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.strided_slice: (1x2xf32) <- (1x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_1 = paddle._C_ops.strided_slice(
            set_value_with_tensor__2,
            [1],
            full_int_array_3,
            full_int_array_1,
            full_int_array_2,
        )

        # pd_op.multiply: (1x2xf32) <- (1x2xf32, xf32)
        multiply_1 = paddle._C_ops.multiply(strided_slice_1, data_1)

        # pd_op.set_value_with_tensor_: (1x4xf32) <- (1x4xf32, 1x2xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__0 = paddle._C_ops.set_value_with_tensor_(
            set_value_with_tensor__2,
            multiply_1,
            full_int_array_3,
            full_int_array_1,
            full_int_array_2,
            [1],
            [],
            [],
        )

        # pd_op.strided_slice: (5x2xf32) <- (5x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_2 = paddle._C_ops.strided_slice(
            data_3, [1], full_int_array_0, full_int_array_1, full_int_array_2
        )

        # pd_op.multiply: (5x2xf32) <- (5x2xf32, xf32)
        multiply_2 = paddle._C_ops.multiply(strided_slice_2, data_0)

        # pd_op.set_value_with_tensor_: (5x4xf32) <- (5x4xf32, 5x2xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__3 = paddle._C_ops.set_value_with_tensor_(
            data_3,
            multiply_2,
            full_int_array_0,
            full_int_array_1,
            full_int_array_2,
            [1],
            [],
            [],
        )

        # pd_op.strided_slice: (5x2xf32) <- (5x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_3 = paddle._C_ops.strided_slice(
            set_value_with_tensor__3,
            [1],
            full_int_array_3,
            full_int_array_1,
            full_int_array_2,
        )

        # pd_op.multiply: (5x2xf32) <- (5x2xf32, xf32)
        multiply_3 = paddle._C_ops.multiply(strided_slice_3, data_1)

        # pd_op.set_value_with_tensor_: (5x4xf32) <- (5x4xf32, 5x2xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__1 = paddle._C_ops.set_value_with_tensor_(
            set_value_with_tensor__3,
            multiply_3,
            full_int_array_3,
            full_int_array_1,
            full_int_array_2,
            [1],
            [],
            [],
        )
        return bilinear_interp_0, set_value_with_tensor__0, set_value_with_tensor__1
