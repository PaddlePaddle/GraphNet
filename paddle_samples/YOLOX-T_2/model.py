import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [2147483647]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        # pd_op.strided_slice: (-1x2xf32) <- (-1x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_0 = paddle._C_ops.strided_slice(
            data_0, [1], full_int_array_0, full_int_array_1, full_int_array_2
        )

        # pd_op.multiply: (-1x2xf32) <- (-1x2xf32, xf32)
        multiply_0 = paddle._C_ops.multiply(strided_slice_0, data_1)

        # pd_op.set_value_with_tensor_: (-1x4xf32) <- (-1x4xf32, -1x2xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__1 = paddle._C_ops.set_value_with_tensor_(
            data_0,
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

        # pd_op.strided_slice: (-1x2xf32) <- (-1x4xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_1 = paddle._C_ops.strided_slice(
            set_value_with_tensor__1,
            [1],
            full_int_array_3,
            full_int_array_1,
            full_int_array_2,
        )

        # pd_op.multiply: (-1x2xf32) <- (-1x2xf32, xf32)
        multiply_1 = paddle._C_ops.multiply(strided_slice_1, data_2)

        # pd_op.set_value_with_tensor_: (-1x4xf32) <- (-1x4xf32, -1x2xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__0 = paddle._C_ops.set_value_with_tensor_(
            set_value_with_tensor__1,
            multiply_1,
            full_int_array_3,
            full_int_array_1,
            full_int_array_2,
            [1],
            [],
            [],
        )
        return set_value_with_tensor__0
