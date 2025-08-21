import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7):
        # pd_op.cast: (2xi32) <- (2xi64)
        cast_0 = paddle._C_ops.cast(data_7, paddle.int32)

        # pd_op.bilinear_interp: (4x3x-1x-1xf32) <- (4x3x640x640xf32, 2xi32, None, None)
        bilinear_interp_0 = paddle._C_ops.bilinear_interp(
            data_6, cast_0, None, None, "NCHW", -1, -1, -1, [], "bilinear", False, 0
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [2147483647]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        # pd_op.strided_slice: (8x2xf32) <- (8x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_0 = paddle._C_ops.strided_slice(
            data_2, [1], full_int_array_0, full_int_array_1, full_int_array_2
        )

        # pd_op.multiply: (8x2xf32) <- (8x2xf32, xf32)
        multiply_0 = paddle._C_ops.multiply(strided_slice_0, data_0)

        # pd_op.set_value_with_tensor_: (8x4xf32) <- (8x4xf32, 8x2xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__4 = paddle._C_ops.set_value_with_tensor_(
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

        # pd_op.strided_slice: (8x2xf32) <- (8x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_1 = paddle._C_ops.strided_slice(
            set_value_with_tensor__4,
            [1],
            full_int_array_3,
            full_int_array_1,
            full_int_array_2,
        )

        # pd_op.multiply: (8x2xf32) <- (8x2xf32, xf32)
        multiply_1 = paddle._C_ops.multiply(strided_slice_1, data_1)

        # pd_op.set_value_with_tensor_: (8x4xf32) <- (8x4xf32, 8x2xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__0 = paddle._C_ops.set_value_with_tensor_(
            set_value_with_tensor__4,
            multiply_1,
            full_int_array_3,
            full_int_array_1,
            full_int_array_2,
            [1],
            [],
            [],
        )

        # pd_op.strided_slice: (4x2xf32) <- (4x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_2 = paddle._C_ops.strided_slice(
            data_3, [1], full_int_array_0, full_int_array_1, full_int_array_2
        )

        # pd_op.multiply: (4x2xf32) <- (4x2xf32, xf32)
        multiply_2 = paddle._C_ops.multiply(strided_slice_2, data_0)

        # pd_op.set_value_with_tensor_: (4x4xf32) <- (4x4xf32, 4x2xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__5 = paddle._C_ops.set_value_with_tensor_(
            data_3,
            multiply_2,
            full_int_array_0,
            full_int_array_1,
            full_int_array_2,
            [1],
            [],
            [],
        )

        # pd_op.strided_slice: (4x2xf32) <- (4x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_3 = paddle._C_ops.strided_slice(
            set_value_with_tensor__5,
            [1],
            full_int_array_3,
            full_int_array_1,
            full_int_array_2,
        )

        # pd_op.multiply: (4x2xf32) <- (4x2xf32, xf32)
        multiply_3 = paddle._C_ops.multiply(strided_slice_3, data_1)

        # pd_op.set_value_with_tensor_: (4x4xf32) <- (4x4xf32, 4x2xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__1 = paddle._C_ops.set_value_with_tensor_(
            set_value_with_tensor__5,
            multiply_3,
            full_int_array_3,
            full_int_array_1,
            full_int_array_2,
            [1],
            [],
            [],
        )

        # pd_op.strided_slice: (2x2xf32) <- (2x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_4 = paddle._C_ops.strided_slice(
            data_4, [1], full_int_array_0, full_int_array_1, full_int_array_2
        )

        # pd_op.multiply: (2x2xf32) <- (2x2xf32, xf32)
        multiply_4 = paddle._C_ops.multiply(strided_slice_4, data_0)

        # pd_op.set_value_with_tensor_: (2x4xf32) <- (2x4xf32, 2x2xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__6 = paddle._C_ops.set_value_with_tensor_(
            data_4,
            multiply_4,
            full_int_array_0,
            full_int_array_1,
            full_int_array_2,
            [1],
            [],
            [],
        )

        # pd_op.strided_slice: (2x2xf32) <- (2x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_5 = paddle._C_ops.strided_slice(
            set_value_with_tensor__6,
            [1],
            full_int_array_3,
            full_int_array_1,
            full_int_array_2,
        )

        # pd_op.multiply: (2x2xf32) <- (2x2xf32, xf32)
        multiply_5 = paddle._C_ops.multiply(strided_slice_5, data_1)

        # pd_op.set_value_with_tensor_: (2x4xf32) <- (2x4xf32, 2x2xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__2 = paddle._C_ops.set_value_with_tensor_(
            set_value_with_tensor__6,
            multiply_5,
            full_int_array_3,
            full_int_array_1,
            full_int_array_2,
            [1],
            [],
            [],
        )

        # pd_op.strided_slice: (2x2xf32) <- (2x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_6 = paddle._C_ops.strided_slice(
            data_5, [1], full_int_array_0, full_int_array_1, full_int_array_2
        )

        # pd_op.multiply: (2x2xf32) <- (2x2xf32, xf32)
        multiply_6 = paddle._C_ops.multiply(strided_slice_6, data_0)

        # pd_op.set_value_with_tensor_: (2x4xf32) <- (2x4xf32, 2x2xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__7 = paddle._C_ops.set_value_with_tensor_(
            data_5,
            multiply_6,
            full_int_array_0,
            full_int_array_1,
            full_int_array_2,
            [1],
            [],
            [],
        )

        # pd_op.strided_slice: (2x2xf32) <- (2x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_7 = paddle._C_ops.strided_slice(
            set_value_with_tensor__7,
            [1],
            full_int_array_3,
            full_int_array_1,
            full_int_array_2,
        )

        # pd_op.multiply: (2x2xf32) <- (2x2xf32, xf32)
        multiply_7 = paddle._C_ops.multiply(strided_slice_7, data_1)

        # pd_op.set_value_with_tensor_: (2x4xf32) <- (2x4xf32, 2x2xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__3 = paddle._C_ops.set_value_with_tensor_(
            set_value_with_tensor__7,
            multiply_7,
            full_int_array_3,
            full_int_array_1,
            full_int_array_2,
            [1],
            [],
            [],
        )
        return (
            bilinear_interp_0,
            set_value_with_tensor__0,
            set_value_with_tensor__1,
            set_value_with_tensor__2,
            set_value_with_tensor__3,
        )
