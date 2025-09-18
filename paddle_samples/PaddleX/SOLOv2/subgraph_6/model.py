import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6):
        # pd_op.divide: (2xf32) <- (2xf32, xf32)
        divide_0 = paddle._C_ops.divide(data_3, data_4)
        del data_3, data_4

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (2xf32) <- (2xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(divide_0, full_0, float("0.5"), True)
        del divide_0, full_0

        # pd_op.cast: (2xi32) <- (2xf32)
        cast_1 = paddle._C_ops.cast(scale_0, paddle.int32)
        del scale_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.unsqueeze: (1x-1x-1x-1xf32) <- (-1x-1x-1xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(data_2, full_int_array_0)
        del data_2

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_0 = [data_0, data_1]
        del data_0, data_1

        # pd_op.bilinear_interp: (1x-1x-1x-1xf32) <- (1x-1x-1x-1xf32, None, [xi64, xi64], None)
        bilinear_interp_0 = paddle._C_ops.bilinear_interp(
            unsqueeze_0,
            None,
            combine_0,
            None,
            "NCHW",
            -1,
            -1,
            -1,
            [],
            "bilinear",
            False,
            0,
        )
        del combine_0, unsqueeze_0

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(data_5, paddle.int64)
        del data_5

        # pd_op.cast: (xi64) <- (xi32)
        cast_3 = paddle._C_ops.cast(data_6, paddle.int64)
        del data_6

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [0, 0]

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_1 = [cast_2, cast_3]
        del cast_2, cast_3

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_1, 0)
        del combine_1

        # pd_op.slice: (1x-1x-1x-1xf32) <- (1x-1x-1x-1xf32, 2xi64, 2xi64)
        slice_0 = paddle._C_ops.slice(
            bilinear_interp_0, [2, 3], full_int_array_1, stack_0, [-1, -1], []
        )
        del bilinear_interp_0, full_int_array_1, stack_0

        # pd_op.cast: (2xi32) <- (2xi32)
        cast_4 = paddle._C_ops.cast(cast_1, paddle.int32)
        del cast_1

        # pd_op.bilinear_interp: (1x-1x-1x-1xf32) <- (1x-1x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_1 = paddle._C_ops.bilinear_interp(
            slice_0, cast_4, None, None, "NCHW", -1, -1, -1, [], "bilinear", False, 0
        )
        del cast_4, slice_0

        # pd_op.squeeze: (-1x-1x-1xf32) <- (1x-1x-1x-1xf32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(bilinear_interp_1, full_int_array_0)
        del bilinear_interp_1, full_int_array_0

        # pd_op.full: (xf32) <- ()
        full_1 = paddle._C_ops.full(
            [], float("0.5"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (-1x-1x-1xb) <- (-1x-1x-1xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(squeeze_0, full_1)
        del full_1, squeeze_0

        # pd_op.cast: (-1x-1x-1xui8) <- (-1x-1x-1xb)
        cast_0 = paddle._C_ops.cast(greater_than_0, paddle.uint8)
        del greater_than_0

        return cast_0
