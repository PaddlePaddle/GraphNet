import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0):
        # pd_op.full: (1xf64) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("128"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (128xf32) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_0, full_1, full_2, dtype="float32")
        del full_0, full_1, full_2

        # pd_op.full: (xf32) <- ()
        full_3 = paddle._C_ops.full(
            [], float("2"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.floor_divide: (128xf32) <- (128xf32, xf32)
        floor_divide_0 = paddle._C_ops.floor_divide(arange_0, full_3)
        del arange_0, full_3

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (128xf32) <- (128xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(floor_divide_0, full_4, float("0"), True)
        del floor_divide_0, full_4

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0.0078125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (128xf32) <- (128xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(scale_0, full_5, float("0"), True)
        del full_5, scale_0

        # pd_op.full: (128xf32) <- ()
        full_6 = paddle._C_ops.full(
            [128],
            float("10000"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.elementwise_pow: (128xf32) <- (128xf32, 128xf32)
        elementwise_pow_0 = paddle._C_ops.elementwise_pow(full_6, scale_1)
        del full_6, scale_1

        # pd_op.sigmoid: (1x9x4xf32) <- (1x9x4xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(data_0)
        del data_0

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("6.28319"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x9x4xf32) <- (1x9x4xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(sigmoid_0, full_7, float("0"), True)
        del full_7, sigmoid_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [3]

        # pd_op.unsqueeze: (1x9x4x1xf32) <- (1x9x4xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(scale_2, full_int_array_0)
        del full_int_array_0, scale_2

        # pd_op.divide: (1x9x4x128xf32) <- (1x9x4x1xf32, 128xf32)
        divide_0 = paddle._C_ops.divide(unsqueeze_0, elementwise_pow_0)
        del elementwise_pow_0, unsqueeze_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2147483647]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [2]

        # pd_op.strided_slice: (1x9x4x64xf32) <- (1x9x4x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_0 = paddle._C_ops.strided_slice(
            divide_0, [3], full_int_array_1, full_int_array_2, full_int_array_3
        )
        del full_int_array_1

        # pd_op.sin: (1x9x4x64xf32) <- (1x9x4x64xf32)
        sin_0 = paddle._C_ops.sin(strided_slice_0)
        del strided_slice_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [1]

        # pd_op.strided_slice: (1x9x4x64xf32) <- (1x9x4x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_1 = paddle._C_ops.strided_slice(
            divide_0, [3], full_int_array_4, full_int_array_2, full_int_array_3
        )
        del divide_0, full_int_array_2, full_int_array_3, full_int_array_4

        # pd_op.cos: (1x9x4x64xf32) <- (1x9x4x64xf32)
        cos_0 = paddle._C_ops.cos(strided_slice_1)
        del strided_slice_1

        # builtin.combine: ([1x9x4x64xf32, 1x9x4x64xf32]) <- (1x9x4x64xf32, 1x9x4x64xf32)
        combine_0 = [sin_0, cos_0]
        del cos_0, sin_0

        # pd_op.stack: (1x9x4x64x2xf32) <- ([1x9x4x64xf32, 1x9x4x64xf32])
        stack_0 = paddle._C_ops.stack(combine_0, 4)
        del combine_0

        # pd_op.flatten: (1x9x512xf32) <- (1x9x4x64x2xf32)
        flatten_0 = paddle._C_ops.flatten(stack_0, 2, 4)
        del stack_0

        return flatten_0
