import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
    ):
        # pd_op.full: (1x853x640xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1, 853, 640],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.set_value_: (1x853x640xf32) <- (1x853x640xf32, 1xi64, 1xi64, 1xi64)
        set_value__0 = paddle._C_ops.set_value_(
            full_0,
            full_int_array_0,
            full_int_array_1,
            full_int_array_1,
            [0],
            [0],
            [],
            [1],
            [float("1")],
        )
        del full_0

        # pd_op.unsqueeze: (1x1x853x640xf32) <- (1x853x640xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(set_value__0, full_int_array_0)

        # pd_op.nearest_interp: (1x1x214x160xf32) <- (1x1x853x640xf32, None, None, None)
        nearest_interp_0 = paddle._C_ops.nearest_interp(
            unsqueeze_0, None, None, None, "NCHW", -1, 214, 160, [], "nearest", False, 0
        )
        del unsqueeze_0

        # pd_op.squeeze: (1x214x160xf32) <- (1x1x214x160xf32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(nearest_interp_0, full_int_array_0)
        del nearest_interp_0

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.cumsum: (1x214x160xf32) <- (1x214x160xf32, 1xi32)
        cumsum_0 = paddle._C_ops.cumsum(squeeze_0, full_1, False, False, False)

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("2"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.cumsum: (1x214x160xf32) <- (1x214x160xf32, 1xi32)
        cumsum_1 = paddle._C_ops.cumsum(squeeze_0, full_2, False, False, False)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x214x160xf32) <- (1x214x160xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(cumsum_0, full_3, float("0"), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [-1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [2147483647]

        # pd_op.slice: (1x1x160xf32) <- (1x214x160xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            cumsum_0, [1], full_int_array_2, full_int_array_3, [1], []
        )
        del cumsum_0

        # pd_op.scale: (1x1x160xf32) <- (1x1x160xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_0, full_3, float("1e-06"), True)
        del slice_0

        # pd_op.divide: (1x214x160xf32) <- (1x214x160xf32, 1x1x160xf32)
        divide_0 = paddle._C_ops.divide(scale_0, scale_1)
        del scale_0, scale_1

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("6.28319"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x214x160xf32) <- (1x214x160xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(divide_0, full_4, float("0"), True)
        del divide_0

        # pd_op.scale: (1x214x160xf32) <- (1x214x160xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(cumsum_1, full_3, float("0"), True)

        # pd_op.slice: (1x214x1xf32) <- (1x214x160xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            cumsum_1, [2], full_int_array_2, full_int_array_3, [1], []
        )
        del cumsum_1

        # pd_op.scale: (1x214x1xf32) <- (1x214x1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(slice_1, full_3, float("1e-06"), True)
        del slice_1

        # pd_op.divide: (1x214x160xf32) <- (1x214x160xf32, 1x214x1xf32)
        divide_1 = paddle._C_ops.divide(scale_3, scale_4)
        del scale_3, scale_4

        # pd_op.scale: (1x214x160xf32) <- (1x214x160xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(divide_1, full_4, float("0"), True)
        del divide_1

        # pd_op.full: (1xf64) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("128"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (128xi64) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_5, full_6, full_7, dtype="int64")
        del full_5, full_6, full_7

        # pd_op.full: (xi64) <- ()
        full_8 = paddle._C_ops.full(
            [], float("2"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.floor_divide: (128xi64) <- (128xi64, xi64)
        floor_divide_0 = paddle._C_ops.floor_divide(arange_0, full_8)
        del arange_0, full_8

        # pd_op.cast: (128xf32) <- (128xi64)
        cast_0 = paddle._C_ops.cast(floor_divide_0, paddle.float32)
        del floor_divide_0

        # pd_op.full: (1xf32) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (128xf32) <- (128xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(cast_0, full_9, float("0"), True)
        del cast_0, full_9

        # pd_op.full: (1xf32) <- ()
        full_10 = paddle._C_ops.full(
            [1], float("0.0078125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (128xf32) <- (128xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(scale_6, full_10, float("0"), True)
        del full_10, scale_6

        # pd_op.full: (128xf32) <- ()
        full_11 = paddle._C_ops.full(
            [128],
            float("20"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.elementwise_pow: (128xf32) <- (128xf32, 128xf32)
        elementwise_pow_0 = paddle._C_ops.elementwise_pow(full_11, scale_7)
        del full_11, scale_7

        # pd_op.unsqueeze: (1x214x160x1xf32) <- (1x214x160xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(scale_5, full_int_array_2)
        del scale_5

        # pd_op.divide: (1x214x160x128xf32) <- (1x214x160x1xf32, 128xf32)
        divide_2 = paddle._C_ops.divide(unsqueeze_1, elementwise_pow_0)
        del unsqueeze_1

        # pd_op.unsqueeze: (1x214x160x1xf32) <- (1x214x160xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(scale_2, full_int_array_2)
        del scale_2

        # pd_op.divide: (1x214x160x128xf32) <- (1x214x160x1xf32, 128xf32)
        divide_3 = paddle._C_ops.divide(unsqueeze_2, elementwise_pow_0)
        del unsqueeze_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [2]

        # pd_op.strided_slice: (1x214x160x64xf32) <- (1x214x160x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_0 = paddle._C_ops.strided_slice(
            divide_2, [3], full_int_array_0, full_int_array_3, full_int_array_4
        )

        # pd_op.sin: (1x214x160x64xf32) <- (1x214x160x64xf32)
        sin_0 = paddle._C_ops.sin(strided_slice_0)
        del strided_slice_0

        # pd_op.strided_slice: (1x214x160x64xf32) <- (1x214x160x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_1 = paddle._C_ops.strided_slice(
            divide_2, [3], full_int_array_1, full_int_array_3, full_int_array_4
        )
        del divide_2

        # pd_op.cos: (1x214x160x64xf32) <- (1x214x160x64xf32)
        cos_0 = paddle._C_ops.cos(strided_slice_1)
        del strided_slice_1

        # builtin.combine: ([1x214x160x64xf32, 1x214x160x64xf32]) <- (1x214x160x64xf32, 1x214x160x64xf32)
        combine_0 = [sin_0, cos_0]
        del cos_0, sin_0

        # pd_op.stack: (1x214x160x64x2xf32) <- ([1x214x160x64xf32, 1x214x160x64xf32])
        stack_0 = paddle._C_ops.stack(combine_0, 4)
        del combine_0

        # pd_op.flatten: (1x214x160x128xf32) <- (1x214x160x64x2xf32)
        flatten_0 = paddle._C_ops.flatten(stack_0, 3, 4)
        del stack_0

        # pd_op.strided_slice: (1x214x160x64xf32) <- (1x214x160x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_2 = paddle._C_ops.strided_slice(
            divide_3, [3], full_int_array_0, full_int_array_3, full_int_array_4
        )

        # pd_op.sin: (1x214x160x64xf32) <- (1x214x160x64xf32)
        sin_1 = paddle._C_ops.sin(strided_slice_2)
        del strided_slice_2

        # pd_op.strided_slice: (1x214x160x64xf32) <- (1x214x160x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_3 = paddle._C_ops.strided_slice(
            divide_3, [3], full_int_array_1, full_int_array_3, full_int_array_4
        )
        del divide_3

        # pd_op.cos: (1x214x160x64xf32) <- (1x214x160x64xf32)
        cos_1 = paddle._C_ops.cos(strided_slice_3)
        del strided_slice_3

        # builtin.combine: ([1x214x160x64xf32, 1x214x160x64xf32]) <- (1x214x160x64xf32, 1x214x160x64xf32)
        combine_1 = [sin_1, cos_1]
        del cos_1, sin_1

        # pd_op.stack: (1x214x160x64x2xf32) <- ([1x214x160x64xf32, 1x214x160x64xf32])
        stack_1 = paddle._C_ops.stack(combine_1, 4)
        del combine_1

        # pd_op.flatten: (1x214x160x128xf32) <- (1x214x160x64x2xf32)
        flatten_1 = paddle._C_ops.flatten(stack_1, 3, 4)
        del stack_1

        # pd_op.full: (1xi32) <- ()
        full_12 = paddle._C_ops.full(
            [1], float("3"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([1x214x160x128xf32, 1x214x160x128xf32]) <- (1x214x160x128xf32, 1x214x160x128xf32)
        combine_2 = [flatten_1, flatten_0]
        del flatten_0, flatten_1

        # pd_op.concat: (1x214x160x256xf32) <- ([1x214x160x128xf32, 1x214x160x128xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_2, full_12)
        del combine_2

        # pd_op.transpose: (1x256x214x160xf32) <- (1x214x160x256xf32)
        transpose_0 = paddle._C_ops.transpose(concat_0, [0, 3, 1, 2])
        del concat_0

        # pd_op.unsqueeze: (1x1x853x640xf32) <- (1x853x640xf32, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(set_value__0, full_int_array_0)

        # pd_op.nearest_interp: (1x1x107x80xf32) <- (1x1x853x640xf32, None, None, None)
        nearest_interp_1 = paddle._C_ops.nearest_interp(
            unsqueeze_3, None, None, None, "NCHW", -1, 107, 80, [], "nearest", False, 0
        )
        del unsqueeze_3

        # pd_op.squeeze: (1x107x80xf32) <- (1x1x107x80xf32, 1xi64)
        squeeze_1 = paddle._C_ops.squeeze(nearest_interp_1, full_int_array_0)
        del nearest_interp_1

        # pd_op.cumsum: (1x107x80xf32) <- (1x107x80xf32, 1xi32)
        cumsum_2 = paddle._C_ops.cumsum(squeeze_1, full_1, False, False, False)

        # pd_op.cumsum: (1x107x80xf32) <- (1x107x80xf32, 1xi32)
        cumsum_3 = paddle._C_ops.cumsum(squeeze_1, full_2, False, False, False)

        # pd_op.scale: (1x107x80xf32) <- (1x107x80xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(cumsum_2, full_3, float("0"), True)

        # pd_op.slice: (1x1x80xf32) <- (1x107x80xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            cumsum_2, [1], full_int_array_2, full_int_array_3, [1], []
        )
        del cumsum_2

        # pd_op.scale: (1x1x80xf32) <- (1x1x80xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(slice_2, full_3, float("1e-06"), True)
        del slice_2

        # pd_op.divide: (1x107x80xf32) <- (1x107x80xf32, 1x1x80xf32)
        divide_4 = paddle._C_ops.divide(scale_8, scale_9)
        del scale_8, scale_9

        # pd_op.scale: (1x107x80xf32) <- (1x107x80xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(divide_4, full_4, float("0"), True)
        del divide_4

        # pd_op.scale: (1x107x80xf32) <- (1x107x80xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(cumsum_3, full_3, float("0"), True)

        # pd_op.slice: (1x107x1xf32) <- (1x107x80xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            cumsum_3, [2], full_int_array_2, full_int_array_3, [1], []
        )
        del cumsum_3

        # pd_op.scale: (1x107x1xf32) <- (1x107x1xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(slice_3, full_3, float("1e-06"), True)
        del slice_3

        # pd_op.divide: (1x107x80xf32) <- (1x107x80xf32, 1x107x1xf32)
        divide_5 = paddle._C_ops.divide(scale_11, scale_12)
        del scale_11, scale_12

        # pd_op.scale: (1x107x80xf32) <- (1x107x80xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(divide_5, full_4, float("0"), True)
        del divide_5

        # pd_op.unsqueeze: (1x107x80x1xf32) <- (1x107x80xf32, 1xi64)
        unsqueeze_4 = paddle._C_ops.unsqueeze(scale_13, full_int_array_2)
        del scale_13

        # pd_op.divide: (1x107x80x128xf32) <- (1x107x80x1xf32, 128xf32)
        divide_6 = paddle._C_ops.divide(unsqueeze_4, elementwise_pow_0)
        del unsqueeze_4

        # pd_op.unsqueeze: (1x107x80x1xf32) <- (1x107x80xf32, 1xi64)
        unsqueeze_5 = paddle._C_ops.unsqueeze(scale_10, full_int_array_2)
        del scale_10

        # pd_op.divide: (1x107x80x128xf32) <- (1x107x80x1xf32, 128xf32)
        divide_7 = paddle._C_ops.divide(unsqueeze_5, elementwise_pow_0)
        del unsqueeze_5

        # pd_op.strided_slice: (1x107x80x64xf32) <- (1x107x80x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_4 = paddle._C_ops.strided_slice(
            divide_6, [3], full_int_array_0, full_int_array_3, full_int_array_4
        )

        # pd_op.sin: (1x107x80x64xf32) <- (1x107x80x64xf32)
        sin_2 = paddle._C_ops.sin(strided_slice_4)
        del strided_slice_4

        # pd_op.strided_slice: (1x107x80x64xf32) <- (1x107x80x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_5 = paddle._C_ops.strided_slice(
            divide_6, [3], full_int_array_1, full_int_array_3, full_int_array_4
        )
        del divide_6

        # pd_op.cos: (1x107x80x64xf32) <- (1x107x80x64xf32)
        cos_2 = paddle._C_ops.cos(strided_slice_5)
        del strided_slice_5

        # builtin.combine: ([1x107x80x64xf32, 1x107x80x64xf32]) <- (1x107x80x64xf32, 1x107x80x64xf32)
        combine_3 = [sin_2, cos_2]
        del cos_2, sin_2

        # pd_op.stack: (1x107x80x64x2xf32) <- ([1x107x80x64xf32, 1x107x80x64xf32])
        stack_2 = paddle._C_ops.stack(combine_3, 4)
        del combine_3

        # pd_op.flatten: (1x107x80x128xf32) <- (1x107x80x64x2xf32)
        flatten_2 = paddle._C_ops.flatten(stack_2, 3, 4)
        del stack_2

        # pd_op.strided_slice: (1x107x80x64xf32) <- (1x107x80x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_6 = paddle._C_ops.strided_slice(
            divide_7, [3], full_int_array_0, full_int_array_3, full_int_array_4
        )

        # pd_op.sin: (1x107x80x64xf32) <- (1x107x80x64xf32)
        sin_3 = paddle._C_ops.sin(strided_slice_6)
        del strided_slice_6

        # pd_op.strided_slice: (1x107x80x64xf32) <- (1x107x80x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_7 = paddle._C_ops.strided_slice(
            divide_7, [3], full_int_array_1, full_int_array_3, full_int_array_4
        )
        del divide_7

        # pd_op.cos: (1x107x80x64xf32) <- (1x107x80x64xf32)
        cos_3 = paddle._C_ops.cos(strided_slice_7)
        del strided_slice_7

        # builtin.combine: ([1x107x80x64xf32, 1x107x80x64xf32]) <- (1x107x80x64xf32, 1x107x80x64xf32)
        combine_4 = [sin_3, cos_3]
        del cos_3, sin_3

        # pd_op.stack: (1x107x80x64x2xf32) <- ([1x107x80x64xf32, 1x107x80x64xf32])
        stack_3 = paddle._C_ops.stack(combine_4, 4)
        del combine_4

        # pd_op.flatten: (1x107x80x128xf32) <- (1x107x80x64x2xf32)
        flatten_3 = paddle._C_ops.flatten(stack_3, 3, 4)
        del stack_3

        # builtin.combine: ([1x107x80x128xf32, 1x107x80x128xf32]) <- (1x107x80x128xf32, 1x107x80x128xf32)
        combine_5 = [flatten_3, flatten_2]
        del flatten_2, flatten_3

        # pd_op.concat: (1x107x80x256xf32) <- ([1x107x80x128xf32, 1x107x80x128xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_5, full_12)
        del combine_5

        # pd_op.transpose: (1x256x107x80xf32) <- (1x107x80x256xf32)
        transpose_1 = paddle._C_ops.transpose(concat_1, [0, 3, 1, 2])
        del concat_1

        # pd_op.unsqueeze: (1x1x853x640xf32) <- (1x853x640xf32, 1xi64)
        unsqueeze_6 = paddle._C_ops.unsqueeze(set_value__0, full_int_array_0)

        # pd_op.nearest_interp: (1x1x54x40xf32) <- (1x1x853x640xf32, None, None, None)
        nearest_interp_2 = paddle._C_ops.nearest_interp(
            unsqueeze_6, None, None, None, "NCHW", -1, 54, 40, [], "nearest", False, 0
        )
        del unsqueeze_6

        # pd_op.squeeze: (1x54x40xf32) <- (1x1x54x40xf32, 1xi64)
        squeeze_2 = paddle._C_ops.squeeze(nearest_interp_2, full_int_array_0)
        del nearest_interp_2

        # pd_op.cumsum: (1x54x40xf32) <- (1x54x40xf32, 1xi32)
        cumsum_4 = paddle._C_ops.cumsum(squeeze_2, full_1, False, False, False)

        # pd_op.cumsum: (1x54x40xf32) <- (1x54x40xf32, 1xi32)
        cumsum_5 = paddle._C_ops.cumsum(squeeze_2, full_2, False, False, False)

        # pd_op.scale: (1x54x40xf32) <- (1x54x40xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(cumsum_4, full_3, float("0"), True)

        # pd_op.slice: (1x1x40xf32) <- (1x54x40xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            cumsum_4, [1], full_int_array_2, full_int_array_3, [1], []
        )
        del cumsum_4

        # pd_op.scale: (1x1x40xf32) <- (1x1x40xf32, 1xf32)
        scale_15 = paddle._C_ops.scale(slice_4, full_3, float("1e-06"), True)
        del slice_4

        # pd_op.divide: (1x54x40xf32) <- (1x54x40xf32, 1x1x40xf32)
        divide_8 = paddle._C_ops.divide(scale_14, scale_15)
        del scale_14, scale_15

        # pd_op.scale: (1x54x40xf32) <- (1x54x40xf32, 1xf32)
        scale_16 = paddle._C_ops.scale(divide_8, full_4, float("0"), True)
        del divide_8

        # pd_op.scale: (1x54x40xf32) <- (1x54x40xf32, 1xf32)
        scale_17 = paddle._C_ops.scale(cumsum_5, full_3, float("0"), True)

        # pd_op.slice: (1x54x1xf32) <- (1x54x40xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            cumsum_5, [2], full_int_array_2, full_int_array_3, [1], []
        )
        del cumsum_5

        # pd_op.scale: (1x54x1xf32) <- (1x54x1xf32, 1xf32)
        scale_18 = paddle._C_ops.scale(slice_5, full_3, float("1e-06"), True)
        del slice_5

        # pd_op.divide: (1x54x40xf32) <- (1x54x40xf32, 1x54x1xf32)
        divide_9 = paddle._C_ops.divide(scale_17, scale_18)
        del scale_17, scale_18

        # pd_op.scale: (1x54x40xf32) <- (1x54x40xf32, 1xf32)
        scale_19 = paddle._C_ops.scale(divide_9, full_4, float("0"), True)
        del divide_9

        # pd_op.unsqueeze: (1x54x40x1xf32) <- (1x54x40xf32, 1xi64)
        unsqueeze_7 = paddle._C_ops.unsqueeze(scale_19, full_int_array_2)
        del scale_19

        # pd_op.divide: (1x54x40x128xf32) <- (1x54x40x1xf32, 128xf32)
        divide_10 = paddle._C_ops.divide(unsqueeze_7, elementwise_pow_0)
        del unsqueeze_7

        # pd_op.unsqueeze: (1x54x40x1xf32) <- (1x54x40xf32, 1xi64)
        unsqueeze_8 = paddle._C_ops.unsqueeze(scale_16, full_int_array_2)
        del scale_16

        # pd_op.divide: (1x54x40x128xf32) <- (1x54x40x1xf32, 128xf32)
        divide_11 = paddle._C_ops.divide(unsqueeze_8, elementwise_pow_0)
        del unsqueeze_8

        # pd_op.strided_slice: (1x54x40x64xf32) <- (1x54x40x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_8 = paddle._C_ops.strided_slice(
            divide_10, [3], full_int_array_0, full_int_array_3, full_int_array_4
        )

        # pd_op.sin: (1x54x40x64xf32) <- (1x54x40x64xf32)
        sin_4 = paddle._C_ops.sin(strided_slice_8)
        del strided_slice_8

        # pd_op.strided_slice: (1x54x40x64xf32) <- (1x54x40x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_9 = paddle._C_ops.strided_slice(
            divide_10, [3], full_int_array_1, full_int_array_3, full_int_array_4
        )
        del divide_10

        # pd_op.cos: (1x54x40x64xf32) <- (1x54x40x64xf32)
        cos_4 = paddle._C_ops.cos(strided_slice_9)
        del strided_slice_9

        # builtin.combine: ([1x54x40x64xf32, 1x54x40x64xf32]) <- (1x54x40x64xf32, 1x54x40x64xf32)
        combine_6 = [sin_4, cos_4]
        del cos_4, sin_4

        # pd_op.stack: (1x54x40x64x2xf32) <- ([1x54x40x64xf32, 1x54x40x64xf32])
        stack_4 = paddle._C_ops.stack(combine_6, 4)
        del combine_6

        # pd_op.flatten: (1x54x40x128xf32) <- (1x54x40x64x2xf32)
        flatten_4 = paddle._C_ops.flatten(stack_4, 3, 4)
        del stack_4

        # pd_op.strided_slice: (1x54x40x64xf32) <- (1x54x40x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_10 = paddle._C_ops.strided_slice(
            divide_11, [3], full_int_array_0, full_int_array_3, full_int_array_4
        )

        # pd_op.sin: (1x54x40x64xf32) <- (1x54x40x64xf32)
        sin_5 = paddle._C_ops.sin(strided_slice_10)
        del strided_slice_10

        # pd_op.strided_slice: (1x54x40x64xf32) <- (1x54x40x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_11 = paddle._C_ops.strided_slice(
            divide_11, [3], full_int_array_1, full_int_array_3, full_int_array_4
        )
        del divide_11

        # pd_op.cos: (1x54x40x64xf32) <- (1x54x40x64xf32)
        cos_5 = paddle._C_ops.cos(strided_slice_11)
        del strided_slice_11

        # builtin.combine: ([1x54x40x64xf32, 1x54x40x64xf32]) <- (1x54x40x64xf32, 1x54x40x64xf32)
        combine_7 = [sin_5, cos_5]
        del cos_5, sin_5

        # pd_op.stack: (1x54x40x64x2xf32) <- ([1x54x40x64xf32, 1x54x40x64xf32])
        stack_5 = paddle._C_ops.stack(combine_7, 4)
        del combine_7

        # pd_op.flatten: (1x54x40x128xf32) <- (1x54x40x64x2xf32)
        flatten_5 = paddle._C_ops.flatten(stack_5, 3, 4)
        del stack_5

        # builtin.combine: ([1x54x40x128xf32, 1x54x40x128xf32]) <- (1x54x40x128xf32, 1x54x40x128xf32)
        combine_8 = [flatten_5, flatten_4]
        del flatten_4, flatten_5

        # pd_op.concat: (1x54x40x256xf32) <- ([1x54x40x128xf32, 1x54x40x128xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_8, full_12)
        del combine_8

        # pd_op.transpose: (1x256x54x40xf32) <- (1x54x40x256xf32)
        transpose_2 = paddle._C_ops.transpose(concat_2, [0, 3, 1, 2])
        del concat_2

        # pd_op.unsqueeze: (1x1x853x640xf32) <- (1x853x640xf32, 1xi64)
        unsqueeze_9 = paddle._C_ops.unsqueeze(set_value__0, full_int_array_0)

        # pd_op.nearest_interp: (1x1x27x20xf32) <- (1x1x853x640xf32, None, None, None)
        nearest_interp_3 = paddle._C_ops.nearest_interp(
            unsqueeze_9, None, None, None, "NCHW", -1, 27, 20, [], "nearest", False, 0
        )
        del unsqueeze_9

        # pd_op.squeeze: (1x27x20xf32) <- (1x1x27x20xf32, 1xi64)
        squeeze_3 = paddle._C_ops.squeeze(nearest_interp_3, full_int_array_0)
        del nearest_interp_3

        # pd_op.cumsum: (1x27x20xf32) <- (1x27x20xf32, 1xi32)
        cumsum_6 = paddle._C_ops.cumsum(squeeze_3, full_1, False, False, False)

        # pd_op.cumsum: (1x27x20xf32) <- (1x27x20xf32, 1xi32)
        cumsum_7 = paddle._C_ops.cumsum(squeeze_3, full_2, False, False, False)

        # pd_op.scale: (1x27x20xf32) <- (1x27x20xf32, 1xf32)
        scale_20 = paddle._C_ops.scale(cumsum_6, full_3, float("0"), True)

        # pd_op.slice: (1x1x20xf32) <- (1x27x20xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            cumsum_6, [1], full_int_array_2, full_int_array_3, [1], []
        )
        del cumsum_6

        # pd_op.scale: (1x1x20xf32) <- (1x1x20xf32, 1xf32)
        scale_21 = paddle._C_ops.scale(slice_6, full_3, float("1e-06"), True)
        del slice_6

        # pd_op.divide: (1x27x20xf32) <- (1x27x20xf32, 1x1x20xf32)
        divide_12 = paddle._C_ops.divide(scale_20, scale_21)
        del scale_20, scale_21

        # pd_op.scale: (1x27x20xf32) <- (1x27x20xf32, 1xf32)
        scale_22 = paddle._C_ops.scale(divide_12, full_4, float("0"), True)
        del divide_12

        # pd_op.scale: (1x27x20xf32) <- (1x27x20xf32, 1xf32)
        scale_23 = paddle._C_ops.scale(cumsum_7, full_3, float("0"), True)

        # pd_op.slice: (1x27x1xf32) <- (1x27x20xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            cumsum_7, [2], full_int_array_2, full_int_array_3, [1], []
        )
        del cumsum_7

        # pd_op.scale: (1x27x1xf32) <- (1x27x1xf32, 1xf32)
        scale_24 = paddle._C_ops.scale(slice_7, full_3, float("1e-06"), True)
        del slice_7

        # pd_op.divide: (1x27x20xf32) <- (1x27x20xf32, 1x27x1xf32)
        divide_13 = paddle._C_ops.divide(scale_23, scale_24)
        del scale_23, scale_24

        # pd_op.scale: (1x27x20xf32) <- (1x27x20xf32, 1xf32)
        scale_25 = paddle._C_ops.scale(divide_13, full_4, float("0"), True)
        del divide_13

        # pd_op.unsqueeze: (1x27x20x1xf32) <- (1x27x20xf32, 1xi64)
        unsqueeze_10 = paddle._C_ops.unsqueeze(scale_25, full_int_array_2)
        del scale_25

        # pd_op.divide: (1x27x20x128xf32) <- (1x27x20x1xf32, 128xf32)
        divide_14 = paddle._C_ops.divide(unsqueeze_10, elementwise_pow_0)
        del unsqueeze_10

        # pd_op.unsqueeze: (1x27x20x1xf32) <- (1x27x20xf32, 1xi64)
        unsqueeze_11 = paddle._C_ops.unsqueeze(scale_22, full_int_array_2)
        del scale_22

        # pd_op.divide: (1x27x20x128xf32) <- (1x27x20x1xf32, 128xf32)
        divide_15 = paddle._C_ops.divide(unsqueeze_11, elementwise_pow_0)
        del unsqueeze_11

        # pd_op.strided_slice: (1x27x20x64xf32) <- (1x27x20x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_12 = paddle._C_ops.strided_slice(
            divide_14, [3], full_int_array_0, full_int_array_3, full_int_array_4
        )

        # pd_op.sin: (1x27x20x64xf32) <- (1x27x20x64xf32)
        sin_6 = paddle._C_ops.sin(strided_slice_12)
        del strided_slice_12

        # pd_op.strided_slice: (1x27x20x64xf32) <- (1x27x20x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_13 = paddle._C_ops.strided_slice(
            divide_14, [3], full_int_array_1, full_int_array_3, full_int_array_4
        )
        del divide_14

        # pd_op.cos: (1x27x20x64xf32) <- (1x27x20x64xf32)
        cos_6 = paddle._C_ops.cos(strided_slice_13)
        del strided_slice_13

        # builtin.combine: ([1x27x20x64xf32, 1x27x20x64xf32]) <- (1x27x20x64xf32, 1x27x20x64xf32)
        combine_9 = [sin_6, cos_6]
        del cos_6, sin_6

        # pd_op.stack: (1x27x20x64x2xf32) <- ([1x27x20x64xf32, 1x27x20x64xf32])
        stack_6 = paddle._C_ops.stack(combine_9, 4)
        del combine_9

        # pd_op.flatten: (1x27x20x128xf32) <- (1x27x20x64x2xf32)
        flatten_6 = paddle._C_ops.flatten(stack_6, 3, 4)
        del stack_6

        # pd_op.strided_slice: (1x27x20x64xf32) <- (1x27x20x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_14 = paddle._C_ops.strided_slice(
            divide_15, [3], full_int_array_0, full_int_array_3, full_int_array_4
        )

        # pd_op.sin: (1x27x20x64xf32) <- (1x27x20x64xf32)
        sin_7 = paddle._C_ops.sin(strided_slice_14)
        del strided_slice_14

        # pd_op.strided_slice: (1x27x20x64xf32) <- (1x27x20x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_15 = paddle._C_ops.strided_slice(
            divide_15, [3], full_int_array_1, full_int_array_3, full_int_array_4
        )
        del divide_15

        # pd_op.cos: (1x27x20x64xf32) <- (1x27x20x64xf32)
        cos_7 = paddle._C_ops.cos(strided_slice_15)
        del strided_slice_15

        # builtin.combine: ([1x27x20x64xf32, 1x27x20x64xf32]) <- (1x27x20x64xf32, 1x27x20x64xf32)
        combine_10 = [sin_7, cos_7]
        del cos_7, sin_7

        # pd_op.stack: (1x27x20x64x2xf32) <- ([1x27x20x64xf32, 1x27x20x64xf32])
        stack_7 = paddle._C_ops.stack(combine_10, 4)
        del combine_10

        # pd_op.flatten: (1x27x20x128xf32) <- (1x27x20x64x2xf32)
        flatten_7 = paddle._C_ops.flatten(stack_7, 3, 4)
        del stack_7

        # builtin.combine: ([1x27x20x128xf32, 1x27x20x128xf32]) <- (1x27x20x128xf32, 1x27x20x128xf32)
        combine_11 = [flatten_7, flatten_6]
        del flatten_6, flatten_7

        # pd_op.concat: (1x27x20x256xf32) <- ([1x27x20x128xf32, 1x27x20x128xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_11, full_12)
        del combine_11

        # pd_op.transpose: (1x256x27x20xf32) <- (1x27x20x256xf32)
        transpose_3 = paddle._C_ops.transpose(concat_3, [0, 3, 1, 2])
        del concat_3

        # pd_op.unsqueeze: (1x1x853x640xf32) <- (1x853x640xf32, 1xi64)
        unsqueeze_12 = paddle._C_ops.unsqueeze(set_value__0, full_int_array_0)

        # pd_op.nearest_interp: (1x1x14x10xf32) <- (1x1x853x640xf32, None, None, None)
        nearest_interp_4 = paddle._C_ops.nearest_interp(
            unsqueeze_12, None, None, None, "NCHW", -1, 14, 10, [], "nearest", False, 0
        )
        del unsqueeze_12

        # pd_op.squeeze: (1x14x10xf32) <- (1x1x14x10xf32, 1xi64)
        squeeze_4 = paddle._C_ops.squeeze(nearest_interp_4, full_int_array_0)
        del nearest_interp_4

        # pd_op.cumsum: (1x14x10xf32) <- (1x14x10xf32, 1xi32)
        cumsum_8 = paddle._C_ops.cumsum(squeeze_4, full_1, False, False, False)
        del full_1

        # pd_op.cumsum: (1x14x10xf32) <- (1x14x10xf32, 1xi32)
        cumsum_9 = paddle._C_ops.cumsum(squeeze_4, full_2, False, False, False)
        del full_2

        # pd_op.scale: (1x14x10xf32) <- (1x14x10xf32, 1xf32)
        scale_26 = paddle._C_ops.scale(cumsum_8, full_3, float("0"), True)

        # pd_op.slice: (1x1x10xf32) <- (1x14x10xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            cumsum_8, [1], full_int_array_2, full_int_array_3, [1], []
        )
        del cumsum_8

        # pd_op.scale: (1x1x10xf32) <- (1x1x10xf32, 1xf32)
        scale_27 = paddle._C_ops.scale(slice_8, full_3, float("1e-06"), True)
        del slice_8

        # pd_op.divide: (1x14x10xf32) <- (1x14x10xf32, 1x1x10xf32)
        divide_16 = paddle._C_ops.divide(scale_26, scale_27)
        del scale_26, scale_27

        # pd_op.scale: (1x14x10xf32) <- (1x14x10xf32, 1xf32)
        scale_28 = paddle._C_ops.scale(divide_16, full_4, float("0"), True)
        del divide_16

        # pd_op.scale: (1x14x10xf32) <- (1x14x10xf32, 1xf32)
        scale_29 = paddle._C_ops.scale(cumsum_9, full_3, float("0"), True)

        # pd_op.slice: (1x14x1xf32) <- (1x14x10xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            cumsum_9, [2], full_int_array_2, full_int_array_3, [1], []
        )
        del cumsum_9

        # pd_op.scale: (1x14x1xf32) <- (1x14x1xf32, 1xf32)
        scale_30 = paddle._C_ops.scale(slice_9, full_3, float("1e-06"), True)
        del full_3, slice_9

        # pd_op.divide: (1x14x10xf32) <- (1x14x10xf32, 1x14x1xf32)
        divide_17 = paddle._C_ops.divide(scale_29, scale_30)
        del scale_29, scale_30

        # pd_op.scale: (1x14x10xf32) <- (1x14x10xf32, 1xf32)
        scale_31 = paddle._C_ops.scale(divide_17, full_4, float("0"), True)
        del divide_17, full_4

        # pd_op.unsqueeze: (1x14x10x1xf32) <- (1x14x10xf32, 1xi64)
        unsqueeze_13 = paddle._C_ops.unsqueeze(scale_31, full_int_array_2)
        del scale_31

        # pd_op.divide: (1x14x10x128xf32) <- (1x14x10x1xf32, 128xf32)
        divide_18 = paddle._C_ops.divide(unsqueeze_13, elementwise_pow_0)
        del unsqueeze_13

        # pd_op.unsqueeze: (1x14x10x1xf32) <- (1x14x10xf32, 1xi64)
        unsqueeze_14 = paddle._C_ops.unsqueeze(scale_28, full_int_array_2)
        del full_int_array_2, scale_28

        # pd_op.divide: (1x14x10x128xf32) <- (1x14x10x1xf32, 128xf32)
        divide_19 = paddle._C_ops.divide(unsqueeze_14, elementwise_pow_0)
        del elementwise_pow_0, unsqueeze_14

        # pd_op.strided_slice: (1x14x10x64xf32) <- (1x14x10x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_16 = paddle._C_ops.strided_slice(
            divide_18, [3], full_int_array_0, full_int_array_3, full_int_array_4
        )

        # pd_op.sin: (1x14x10x64xf32) <- (1x14x10x64xf32)
        sin_8 = paddle._C_ops.sin(strided_slice_16)
        del strided_slice_16

        # pd_op.strided_slice: (1x14x10x64xf32) <- (1x14x10x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_17 = paddle._C_ops.strided_slice(
            divide_18, [3], full_int_array_1, full_int_array_3, full_int_array_4
        )
        del divide_18

        # pd_op.cos: (1x14x10x64xf32) <- (1x14x10x64xf32)
        cos_8 = paddle._C_ops.cos(strided_slice_17)
        del strided_slice_17

        # builtin.combine: ([1x14x10x64xf32, 1x14x10x64xf32]) <- (1x14x10x64xf32, 1x14x10x64xf32)
        combine_12 = [sin_8, cos_8]
        del cos_8, sin_8

        # pd_op.stack: (1x14x10x64x2xf32) <- ([1x14x10x64xf32, 1x14x10x64xf32])
        stack_8 = paddle._C_ops.stack(combine_12, 4)
        del combine_12

        # pd_op.flatten: (1x14x10x128xf32) <- (1x14x10x64x2xf32)
        flatten_8 = paddle._C_ops.flatten(stack_8, 3, 4)
        del stack_8

        # pd_op.strided_slice: (1x14x10x64xf32) <- (1x14x10x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_18 = paddle._C_ops.strided_slice(
            divide_19, [3], full_int_array_0, full_int_array_3, full_int_array_4
        )
        del full_int_array_0

        # pd_op.sin: (1x14x10x64xf32) <- (1x14x10x64xf32)
        sin_9 = paddle._C_ops.sin(strided_slice_18)
        del strided_slice_18

        # pd_op.strided_slice: (1x14x10x64xf32) <- (1x14x10x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_19 = paddle._C_ops.strided_slice(
            divide_19, [3], full_int_array_1, full_int_array_3, full_int_array_4
        )
        del divide_19, full_int_array_1, full_int_array_3, full_int_array_4

        # pd_op.cos: (1x14x10x64xf32) <- (1x14x10x64xf32)
        cos_9 = paddle._C_ops.cos(strided_slice_19)
        del strided_slice_19

        # builtin.combine: ([1x14x10x64xf32, 1x14x10x64xf32]) <- (1x14x10x64xf32, 1x14x10x64xf32)
        combine_13 = [sin_9, cos_9]
        del cos_9, sin_9

        # pd_op.stack: (1x14x10x64x2xf32) <- ([1x14x10x64xf32, 1x14x10x64xf32])
        stack_9 = paddle._C_ops.stack(combine_13, 4)
        del combine_13

        # pd_op.flatten: (1x14x10x128xf32) <- (1x14x10x64x2xf32)
        flatten_9 = paddle._C_ops.flatten(stack_9, 3, 4)
        del stack_9

        # builtin.combine: ([1x14x10x128xf32, 1x14x10x128xf32]) <- (1x14x10x128xf32, 1x14x10x128xf32)
        combine_14 = [flatten_9, flatten_8]
        del flatten_8, flatten_9

        # pd_op.concat: (1x14x10x256xf32) <- ([1x14x10x128xf32, 1x14x10x128xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_14, full_12)
        del combine_14, full_12

        # pd_op.transpose: (1x256x14x10xf32) <- (1x14x10x256xf32)
        transpose_4 = paddle._C_ops.transpose(concat_4, [0, 3, 1, 2])
        del (
            concat_4,
            set_value__0,
            squeeze_0,
            squeeze_1,
            squeeze_2,
            squeeze_3,
            squeeze_4,
        )

        return transpose_0, transpose_1, transpose_2, transpose_3, transpose_4
