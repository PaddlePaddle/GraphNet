import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, parameter_0, parameter_1, data_0, data_1):
        # pd_op.conv2d: (4x256x-1x-1xf32) <- (4x2048x-1x-1xf32, 256x2048x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_1, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_0, full_int_array_0)

        # pd_op.add: (4x256x-1x-1xf32) <- (4x256x-1x-1xf32, 1x256x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_0, reshape_0)

        # pd_op.shape64: (4xi64) <- (4x256x-1x-1xf32)
        shape64_0 = paddle._C_ops.shape64(add_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [3]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [4]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.flatten: (4x256x-1xf32) <- (4x256x-1x-1xf32)
        flatten_1 = paddle._C_ops.flatten(add_0, 2, 3)

        # pd_op.transpose: (4x-1x256xf32) <- (4x256x-1xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_1, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [0]

        # pd_op.unsqueeze: (1x4x-1x-1xf32) <- (4x-1x-1xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(data_1, full_int_array_4)

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_0 = [slice_0, slice_1]

        # pd_op.nearest_interp: (1x4x-1x-1xf32) <- (1x4x-1x-1xf32, None, [xi64, xi64], None)
        nearest_interp_0 = paddle._C_ops.nearest_interp(
            unsqueeze_0,
            None,
            combine_0,
            None,
            "NCHW",
            -1,
            -1,
            -1,
            [],
            "nearest",
            False,
            0,
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [1]

        # pd_op.slice: (4x-1x-1xf32) <- (1x4x-1x-1xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            nearest_interp_0, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.cumsum: (4x-1x-1xf32) <- (4x-1x-1xf32, 1xi32)
        cumsum_0 = paddle._C_ops.cumsum(slice_2, full_0, False, False, False)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("2"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.cumsum: (4x-1x-1xf32) <- (4x-1x-1xf32, 1xi32)
        cumsum_1 = paddle._C_ops.cumsum(slice_2, full_1, False, False, False)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (4x-1x-1xf32) <- (4x-1x-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(cumsum_0, full_2, float("0"), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [-1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [2147483647]

        # pd_op.slice: (4x-1x-1xf32) <- (4x-1x-1xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            cumsum_0, [1], full_int_array_6, full_int_array_7, [1], []
        )

        # pd_op.scale: (4x-1x-1xf32) <- (4x-1x-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_3, full_2, float("1e-06"), True)

        # pd_op.divide: (4x-1x-1xf32) <- (4x-1x-1xf32, 4x-1x-1xf32)
        divide_0 = paddle._C_ops.divide(scale_0, scale_1)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("6.28319"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (4x-1x-1xf32) <- (4x-1x-1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(divide_0, full_3, float("0"), True)

        # pd_op.scale: (4x-1x-1xf32) <- (4x-1x-1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(cumsum_1, full_2, float("0"), True)

        # pd_op.slice: (4x-1x-1xf32) <- (4x-1x-1xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            cumsum_1, [2], full_int_array_6, full_int_array_7, [1], []
        )

        # pd_op.scale: (4x-1x-1xf32) <- (4x-1x-1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(slice_4, full_2, float("1e-06"), True)

        # pd_op.divide: (4x-1x-1xf32) <- (4x-1x-1xf32, 4x-1x-1xf32)
        divide_1 = paddle._C_ops.divide(scale_3, scale_4)

        # pd_op.scale: (4x-1x-1xf32) <- (4x-1x-1xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(divide_1, full_3, float("0"), True)

        # pd_op.full: (1xf64) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("128"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (128xi64) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_4, full_5, full_6, dtype="int64")

        # pd_op.full: (xi64) <- ()
        full_7 = paddle._C_ops.full(
            [], float("2"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.floor_divide: (128xi64) <- (128xi64, xi64)
        floor_divide_0 = paddle._C_ops.floor_divide(arange_0, full_7)

        # pd_op.cast: (128xf32) <- (128xi64)
        cast_0 = paddle._C_ops.cast(floor_divide_0, paddle.float32)

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (128xf32) <- (128xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(cast_0, full_8, float("0"), True)

        # pd_op.full: (1xf32) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("0.0078125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (128xf32) <- (128xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(scale_6, full_9, float("0"), True)

        # pd_op.full: (128xf32) <- ()
        full_10 = paddle._C_ops.full(
            [128],
            float("10000"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.elementwise_pow: (128xf32) <- (128xf32, 128xf32)
        elementwise_pow_0 = paddle._C_ops.elementwise_pow(full_10, scale_7)

        # pd_op.unsqueeze: (4x-1x-1x1xf32) <- (4x-1x-1xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(scale_5, full_int_array_6)

        # pd_op.divide: (4x-1x-1x128xf32) <- (4x-1x-1x1xf32, 128xf32)
        divide_2 = paddle._C_ops.divide(unsqueeze_1, elementwise_pow_0)

        # pd_op.unsqueeze: (4x-1x-1x1xf32) <- (4x-1x-1xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(scale_2, full_int_array_6)

        # pd_op.divide: (4x-1x-1x128xf32) <- (4x-1x-1x1xf32, 128xf32)
        divide_3 = paddle._C_ops.divide(unsqueeze_2, elementwise_pow_0)

        # pd_op.strided_slice: (4x-1x-1x64xf32) <- (4x-1x-1x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_0 = paddle._C_ops.strided_slice(
            divide_2, [3], full_int_array_4, full_int_array_7, full_int_array_1
        )

        # pd_op.sin: (4x-1x-1x64xf32) <- (4x-1x-1x64xf32)
        sin_0 = paddle._C_ops.sin(strided_slice_0)

        # pd_op.strided_slice: (4x-1x-1x64xf32) <- (4x-1x-1x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_1 = paddle._C_ops.strided_slice(
            divide_2, [3], full_int_array_5, full_int_array_7, full_int_array_1
        )

        # pd_op.cos: (4x-1x-1x64xf32) <- (4x-1x-1x64xf32)
        cos_0 = paddle._C_ops.cos(strided_slice_1)

        # builtin.combine: ([4x-1x-1x64xf32, 4x-1x-1x64xf32]) <- (4x-1x-1x64xf32, 4x-1x-1x64xf32)
        combine_1 = [sin_0, cos_0]

        # pd_op.stack: (4x-1x-1x64x2xf32) <- ([4x-1x-1x64xf32, 4x-1x-1x64xf32])
        stack_0 = paddle._C_ops.stack(combine_1, 4)

        # pd_op.flatten: (4x-1x-1x128xf32) <- (4x-1x-1x64x2xf32)
        flatten_2 = paddle._C_ops.flatten(stack_0, 3, 4)

        # pd_op.strided_slice: (4x-1x-1x64xf32) <- (4x-1x-1x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_2 = paddle._C_ops.strided_slice(
            divide_3, [3], full_int_array_4, full_int_array_7, full_int_array_1
        )

        # pd_op.sin: (4x-1x-1x64xf32) <- (4x-1x-1x64xf32)
        sin_1 = paddle._C_ops.sin(strided_slice_2)

        # pd_op.strided_slice: (4x-1x-1x64xf32) <- (4x-1x-1x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_3 = paddle._C_ops.strided_slice(
            divide_3, [3], full_int_array_5, full_int_array_7, full_int_array_1
        )

        # pd_op.cos: (4x-1x-1x64xf32) <- (4x-1x-1x64xf32)
        cos_1 = paddle._C_ops.cos(strided_slice_3)

        # builtin.combine: ([4x-1x-1x64xf32, 4x-1x-1x64xf32]) <- (4x-1x-1x64xf32, 4x-1x-1x64xf32)
        combine_2 = [sin_1, cos_1]

        # pd_op.stack: (4x-1x-1x64x2xf32) <- ([4x-1x-1x64xf32, 4x-1x-1x64xf32])
        stack_1 = paddle._C_ops.stack(combine_2, 4)

        # pd_op.flatten: (4x-1x-1x128xf32) <- (4x-1x-1x64x2xf32)
        flatten_3 = paddle._C_ops.flatten(stack_1, 3, 4)

        # pd_op.full: (1xi32) <- ()
        full_11 = paddle._C_ops.full(
            [1], float("3"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([4x-1x-1x128xf32, 4x-1x-1x128xf32]) <- (4x-1x-1x128xf32, 4x-1x-1x128xf32)
        combine_3 = [flatten_3, flatten_2]

        # pd_op.concat: (4x-1x-1x256xf32) <- ([4x-1x-1x128xf32, 4x-1x-1x128xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_3, full_11)

        # pd_op.flatten: (4x-1x256xf32) <- (4x-1x-1x256xf32)
        flatten_0 = paddle._C_ops.flatten(concat_0, 1, 2)

        # pd_op.scale: (4x-1x-1xf32) <- (4x-1x-1xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(slice_2, full_2, float("-1"), True)

        # pd_op.full: (1xf32) <- ()
        full_12 = paddle._C_ops.full(
            [1], float("1e+09"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (4x-1x-1xf32) <- (4x-1x-1xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(scale_8, full_12, float("0"), True)

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_0 = paddle._C_ops.multiply(slice_0, slice_1)

        # pd_op.full: (xi64) <- ()
        full_13 = paddle._C_ops.full(
            [], float("4"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_14 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_4 = [full_13, full_14, full_14, multiply_0]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_2 = paddle._C_ops.stack(combine_4, 0)

        # pd_op.reshape: (4x1x1x-1xf32) <- (4x-1x-1xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(scale_9, stack_2)
        return (
            conv2d_0,
            reshape_0,
            transpose_0,
            reshape_1,
            flatten_0,
            slice_0,
            slice_1,
            add_0,
        )
