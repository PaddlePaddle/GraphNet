import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        parameter_0,
        parameter_1,
        parameter_2,
        parameter_3,
        parameter_4,
        parameter_5,
        parameter_6,
        parameter_7,
        parameter_8,
        parameter_9,
        parameter_10,
        parameter_11,
        parameter_12,
        parameter_13,
        parameter_14,
        parameter_15,
        parameter_16,
        parameter_17,
        parameter_18,
        parameter_19,
        parameter_20,
        parameter_21,
        parameter_22,
        parameter_23,
        parameter_24,
        parameter_25,
        parameter_26,
        parameter_27,
        parameter_28,
        parameter_29,
        parameter_30,
        parameter_31,
        parameter_32,
        parameter_33,
        parameter_34,
        parameter_35,
        parameter_36,
        parameter_37,
        parameter_38,
        parameter_39,
        parameter_40,
        parameter_41,
        data_0,
        data_1,
        data_2,
        data_3,
        data_4,
        data_5,
        data_6,
        data_7,
        data_8,
        data_9,
        data_10,
        data_11,
        data_12,
        data_13,
    ):
        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("-1"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_0 = [data_0, full_0, full_0]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)

        # pd_op.expand: (-1x1x256xf32) <- (1x1x256xf32, 3xi64)
        expand_0 = paddle._C_ops.expand(data_4, stack_0)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x1x256xf32, -1x-1x256xf32]) <- (-1x1x256xf32, -1x-1x256xf32)
        combine_1 = [expand_0, data_3]

        # pd_op.concat: (-1x-1x256xf32) <- ([-1x1x256xf32, -1x-1x256xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_1, full_1)

        # pd_op.full: (xi64) <- ()
        full_2 = paddle._C_ops.full(
            [], float("16"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        floor_divide_0 = paddle._C_ops.floor_divide(data_1, full_2)

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        floor_divide_1 = paddle._C_ops.floor_divide(data_2, full_2)

        # pd_op.full: (1xi64) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xi64) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_0 = paddle.arange(full_3, floor_divide_0, full_4, dtype="int64")

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(floor_divide_1, full_5, float("42"), True)

        # pd_op.multiply: (-1xi64) <- (-1xi64, xi64)
        multiply_0 = paddle._C_ops.multiply(arange_0, scale_0)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [-1, 1]

        # pd_op.reshape: (-1x1xi64) <- (-1xi64, 2xi64)
        reshape_0 = paddle._C_ops.reshape(multiply_0, full_int_array_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.reshape: (1xi64) <- (xi64, 1xi64)
        reshape_1 = paddle._C_ops.reshape(floor_divide_1, full_int_array_1)

        # pd_op.repeat_interleave_with_tensor_index: (-1x-1xi64) <- (-1x1xi64, 1xi64)
        repeat_interleave_with_tensor_index_0 = (
            paddle._C_ops.repeat_interleave_with_tensor_index(reshape_0, reshape_1, 1)
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [-1]

        # pd_op.reshape: (-1xi64) <- (-1x-1xi64, 1xi64)
        reshape_2 = paddle._C_ops.reshape(
            repeat_interleave_with_tensor_index_0, full_int_array_2
        )

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_1 = paddle._C_ops.multiply(floor_divide_0, floor_divide_1)

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_1 = paddle.arange(full_3, multiply_1, full_4, dtype="int64")

        # pd_op.add: (-1xi64) <- (-1xi64, -1xi64)
        add_0 = paddle._C_ops.add(reshape_2, arange_1)

        # pd_op.full: (1xi64) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1xi64) <- (-1xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(add_0, full_7, float("1"), True)

        # pd_op.full: (1xi32) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([1xi64, -1xi64]) <- (1xi64, -1xi64)
        combine_2 = [full_6, scale_1]

        # pd_op.concat: (-1xi64) <- ([1xi64, -1xi64], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_2, full_8)

        # pd_op.cast: (-1xi64) <- (-1xi64)
        cast_0 = paddle._C_ops.cast(concat_1, paddle.int64)

        # pd_op.transpose: (505x1x256xf32) <- (1x505x256xf32)
        transpose_0 = paddle._C_ops.transpose(data_5, [1, 0, 2])

        # pd_op.unsqueeze: (-1x1xi64) <- (-1xi64, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(cast_0, full_int_array_2)

        # pd_op.gather_nd: (-1x1x256xf32) <- (505x1x256xf32, -1x1xi64)
        gather_nd_0 = paddle._C_ops.gather_nd(transpose_0, unsqueeze_0)

        # pd_op.transpose: (1x-1x256xf32) <- (-1x1x256xf32)
        transpose_1 = paddle._C_ops.transpose(gather_nd_0, [1, 0, 2])

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 1x-1x256xf32)
        add_1 = paddle._C_ops.add(concat_0, transpose_1)

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_1, layer_norm_2, layer_norm_3 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_1, parameter_41, parameter_40, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.shape64: (3xi64) <- (-1x-1x256xf32)
        shape64_0 = paddle._C_ops.shape64(layer_norm_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [0]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_3, full_int_array_1, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [2]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_1, full_int_array_4, [1], [0]
        )

        # pd_op.matmul: (-1x-1x768xf32) <- (-1x-1x256xf32, 256x768xf32)
        matmul_0 = paddle._C_ops.matmul(layer_norm_1, parameter_39, False, False)

        # pd_op.add: (-1x-1x768xf32) <- (-1x-1x768xf32, 768xf32)
        add_2 = paddle._C_ops.add(matmul_0, parameter_38)

        # pd_op.full: (xi64) <- ()
        full_9 = paddle._C_ops.full(
            [], float("3"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_10 = paddle._C_ops.full(
            [], float("8"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_11 = paddle._C_ops.full(
            [], float("32"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_3 = [slice_0, slice_1, full_9, full_10, full_11]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_3, 0)

        # pd_op.reshape: (-1x-1x3x8x32xf32) <- (-1x-1x768xf32, 5xi64)
        reshape_3 = paddle._C_ops.reshape(add_2, stack_1)

        # pd_op.transpose: (3x-1x8x-1x32xf32) <- (-1x-1x3x8x32xf32)
        transpose_2 = paddle._C_ops.transpose(reshape_3, [2, 0, 3, 1, 4])

        # pd_op.unbind: ([-1x8x-1x32xf32, -1x8x-1x32xf32, -1x8x-1x32xf32]) <- (3x-1x8x-1x32xf32)
        unbind_0 = paddle._C_ops.unbind(transpose_2, 0)

        # builtin.split: (-1x8x-1x32xf32, -1x8x-1x32xf32, -1x8x-1x32xf32) <- ([-1x8x-1x32xf32, -1x8x-1x32xf32, -1x8x-1x32xf32])
        (
            split_0,
            split_1,
            split_2,
        ) = unbind_0

        # pd_op.transpose: (-1x8x32x-1xf32) <- (-1x8x-1x32xf32)
        transpose_3 = paddle._C_ops.transpose(split_1, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x32xf32, -1x8x32x-1xf32)
        matmul_1 = paddle._C_ops.matmul(split_0, transpose_3, False, False)

        # pd_op.full: (1xf32) <- ()
        full_12 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(matmul_1, full_12, float("0"), True)

        # pd_op.softmax: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32)
        softmax_0 = paddle._C_ops.softmax(scale_2, -1)

        # pd_op.matmul: (-1x8x-1x32xf32) <- (-1x8x-1x-1xf32, -1x8x-1x32xf32)
        matmul_2 = paddle._C_ops.matmul(softmax_0, split_2, False, False)

        # pd_op.transpose: (-1x-1x8x32xf32) <- (-1x8x-1x32xf32)
        transpose_4 = paddle._C_ops.transpose(matmul_2, [0, 2, 1, 3])

        # pd_op.full: (xi64) <- ()
        full_13 = paddle._C_ops.full(
            [], float("256"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_4 = [slice_0, slice_1, full_13]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_2 = paddle._C_ops.stack(combine_4, 0)

        # pd_op.reshape: (-1x-1x256xf32) <- (-1x-1x8x32xf32, 3xi64)
        reshape_4 = paddle._C_ops.reshape(transpose_4, stack_2)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x256xf32, 256x256xf32)
        matmul_3 = paddle._C_ops.matmul(reshape_4, data_6, False, True)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_3 = paddle._C_ops.add(matmul_3, data_7)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_4 = paddle._C_ops.add(add_1, add_3)

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_4, layer_norm_5, layer_norm_6 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_4, parameter_37, parameter_36, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x256xf32, 256x1024xf32)
        matmul_4 = paddle._C_ops.matmul(layer_norm_4, parameter_35, False, False)

        # pd_op.add: (-1x-1x1024xf32) <- (-1x-1x1024xf32, 1024xf32)
        add_5 = paddle._C_ops.add(matmul_4, parameter_34)

        # pd_op.gelu: (-1x-1x1024xf32) <- (-1x-1x1024xf32)
        gelu_0 = paddle._C_ops.gelu(add_5, False)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x1024xf32, 1024x256xf32)
        matmul_5 = paddle._C_ops.matmul(gelu_0, parameter_33, False, False)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_6 = paddle._C_ops.add(matmul_5, parameter_32)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_7 = paddle._C_ops.add(add_4, add_6)

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_7, layer_norm_8, layer_norm_9 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_7, parameter_31, parameter_30, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.shape64: (3xi64) <- (-1x-1x256xf32)
        shape64_1 = paddle._C_ops.shape64(layer_norm_7)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_3, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_1, full_int_array_4, [1], [0]
        )

        # pd_op.matmul: (-1x-1x768xf32) <- (-1x-1x256xf32, 256x768xf32)
        matmul_6 = paddle._C_ops.matmul(layer_norm_7, parameter_29, False, False)

        # pd_op.add: (-1x-1x768xf32) <- (-1x-1x768xf32, 768xf32)
        add_8 = paddle._C_ops.add(matmul_6, parameter_28)

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_5 = [slice_2, slice_3, full_9, full_10, full_11]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_3 = paddle._C_ops.stack(combine_5, 0)

        # pd_op.reshape: (-1x-1x3x8x32xf32) <- (-1x-1x768xf32, 5xi64)
        reshape_5 = paddle._C_ops.reshape(add_8, stack_3)

        # pd_op.transpose: (3x-1x8x-1x32xf32) <- (-1x-1x3x8x32xf32)
        transpose_5 = paddle._C_ops.transpose(reshape_5, [2, 0, 3, 1, 4])

        # pd_op.unbind: ([-1x8x-1x32xf32, -1x8x-1x32xf32, -1x8x-1x32xf32]) <- (3x-1x8x-1x32xf32)
        unbind_1 = paddle._C_ops.unbind(transpose_5, 0)

        # builtin.split: (-1x8x-1x32xf32, -1x8x-1x32xf32, -1x8x-1x32xf32) <- ([-1x8x-1x32xf32, -1x8x-1x32xf32, -1x8x-1x32xf32])
        (
            split_3,
            split_4,
            split_5,
        ) = unbind_1

        # pd_op.transpose: (-1x8x32x-1xf32) <- (-1x8x-1x32xf32)
        transpose_6 = paddle._C_ops.transpose(split_4, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x32xf32, -1x8x32x-1xf32)
        matmul_7 = paddle._C_ops.matmul(split_3, transpose_6, False, False)

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(matmul_7, full_12, float("0"), True)

        # pd_op.softmax: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32)
        softmax_1 = paddle._C_ops.softmax(scale_3, -1)

        # pd_op.matmul: (-1x8x-1x32xf32) <- (-1x8x-1x-1xf32, -1x8x-1x32xf32)
        matmul_8 = paddle._C_ops.matmul(softmax_1, split_5, False, False)

        # pd_op.transpose: (-1x-1x8x32xf32) <- (-1x8x-1x32xf32)
        transpose_7 = paddle._C_ops.transpose(matmul_8, [0, 2, 1, 3])

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_6 = [slice_2, slice_3, full_13]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_4 = paddle._C_ops.stack(combine_6, 0)

        # pd_op.reshape: (-1x-1x256xf32) <- (-1x-1x8x32xf32, 3xi64)
        reshape_6 = paddle._C_ops.reshape(transpose_7, stack_4)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x256xf32, 256x256xf32)
        matmul_9 = paddle._C_ops.matmul(reshape_6, data_8, False, True)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_9 = paddle._C_ops.add(matmul_9, data_9)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_10 = paddle._C_ops.add(add_7, add_9)

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_10, layer_norm_11, layer_norm_12 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_10, parameter_27, parameter_26, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x256xf32, 256x1024xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_10, parameter_25, False, False)

        # pd_op.add: (-1x-1x1024xf32) <- (-1x-1x1024xf32, 1024xf32)
        add_11 = paddle._C_ops.add(matmul_10, parameter_24)

        # pd_op.gelu: (-1x-1x1024xf32) <- (-1x-1x1024xf32)
        gelu_1 = paddle._C_ops.gelu(add_11, False)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x1024xf32, 1024x256xf32)
        matmul_11 = paddle._C_ops.matmul(gelu_1, parameter_23, False, False)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_12 = paddle._C_ops.add(matmul_11, parameter_22)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_13 = paddle._C_ops.add(add_10, add_12)

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_13, layer_norm_14, layer_norm_15 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_13, parameter_21, parameter_20, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.shape64: (3xi64) <- (-1x-1x256xf32)
        shape64_2 = paddle._C_ops.shape64(layer_norm_13)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_3, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_1, full_int_array_4, [1], [0]
        )

        # pd_op.matmul: (-1x-1x768xf32) <- (-1x-1x256xf32, 256x768xf32)
        matmul_12 = paddle._C_ops.matmul(layer_norm_13, parameter_19, False, False)

        # pd_op.add: (-1x-1x768xf32) <- (-1x-1x768xf32, 768xf32)
        add_14 = paddle._C_ops.add(matmul_12, parameter_18)

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_7 = [slice_4, slice_5, full_9, full_10, full_11]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_5 = paddle._C_ops.stack(combine_7, 0)

        # pd_op.reshape: (-1x-1x3x8x32xf32) <- (-1x-1x768xf32, 5xi64)
        reshape_7 = paddle._C_ops.reshape(add_14, stack_5)

        # pd_op.transpose: (3x-1x8x-1x32xf32) <- (-1x-1x3x8x32xf32)
        transpose_8 = paddle._C_ops.transpose(reshape_7, [2, 0, 3, 1, 4])

        # pd_op.unbind: ([-1x8x-1x32xf32, -1x8x-1x32xf32, -1x8x-1x32xf32]) <- (3x-1x8x-1x32xf32)
        unbind_2 = paddle._C_ops.unbind(transpose_8, 0)

        # builtin.split: (-1x8x-1x32xf32, -1x8x-1x32xf32, -1x8x-1x32xf32) <- ([-1x8x-1x32xf32, -1x8x-1x32xf32, -1x8x-1x32xf32])
        (
            split_6,
            split_7,
            split_8,
        ) = unbind_2

        # pd_op.transpose: (-1x8x32x-1xf32) <- (-1x8x-1x32xf32)
        transpose_9 = paddle._C_ops.transpose(split_7, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x32xf32, -1x8x32x-1xf32)
        matmul_13 = paddle._C_ops.matmul(split_6, transpose_9, False, False)

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(matmul_13, full_12, float("0"), True)

        # pd_op.softmax: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32)
        softmax_2 = paddle._C_ops.softmax(scale_4, -1)

        # pd_op.matmul: (-1x8x-1x32xf32) <- (-1x8x-1x-1xf32, -1x8x-1x32xf32)
        matmul_14 = paddle._C_ops.matmul(softmax_2, split_8, False, False)

        # pd_op.transpose: (-1x-1x8x32xf32) <- (-1x8x-1x32xf32)
        transpose_10 = paddle._C_ops.transpose(matmul_14, [0, 2, 1, 3])

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_8 = [slice_4, slice_5, full_13]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_6 = paddle._C_ops.stack(combine_8, 0)

        # pd_op.reshape: (-1x-1x256xf32) <- (-1x-1x8x32xf32, 3xi64)
        reshape_8 = paddle._C_ops.reshape(transpose_10, stack_6)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x256xf32, 256x256xf32)
        matmul_15 = paddle._C_ops.matmul(reshape_8, data_10, False, True)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_15 = paddle._C_ops.add(matmul_15, data_11)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_16 = paddle._C_ops.add(add_13, add_15)

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_16, layer_norm_17, layer_norm_18 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_16, parameter_17, parameter_16, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x256xf32, 256x1024xf32)
        matmul_16 = paddle._C_ops.matmul(layer_norm_16, parameter_15, False, False)

        # pd_op.add: (-1x-1x1024xf32) <- (-1x-1x1024xf32, 1024xf32)
        add_17 = paddle._C_ops.add(matmul_16, parameter_14)

        # pd_op.gelu: (-1x-1x1024xf32) <- (-1x-1x1024xf32)
        gelu_2 = paddle._C_ops.gelu(add_17, False)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x1024xf32, 1024x256xf32)
        matmul_17 = paddle._C_ops.matmul(gelu_2, parameter_13, False, False)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_18 = paddle._C_ops.add(matmul_17, parameter_12)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_19 = paddle._C_ops.add(add_16, add_18)

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_19, layer_norm_20, layer_norm_21 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_19, parameter_11, parameter_10, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.shape64: (3xi64) <- (-1x-1x256xf32)
        shape64_3 = paddle._C_ops.shape64(layer_norm_19)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_3, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_1, full_int_array_4, [1], [0]
        )

        # pd_op.matmul: (-1x-1x768xf32) <- (-1x-1x256xf32, 256x768xf32)
        matmul_18 = paddle._C_ops.matmul(layer_norm_19, parameter_9, False, False)

        # pd_op.add: (-1x-1x768xf32) <- (-1x-1x768xf32, 768xf32)
        add_20 = paddle._C_ops.add(matmul_18, parameter_8)

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_9 = [slice_6, slice_7, full_9, full_10, full_11]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_7 = paddle._C_ops.stack(combine_9, 0)

        # pd_op.reshape: (-1x-1x3x8x32xf32) <- (-1x-1x768xf32, 5xi64)
        reshape_9 = paddle._C_ops.reshape(add_20, stack_7)

        # pd_op.transpose: (3x-1x8x-1x32xf32) <- (-1x-1x3x8x32xf32)
        transpose_11 = paddle._C_ops.transpose(reshape_9, [2, 0, 3, 1, 4])

        # pd_op.unbind: ([-1x8x-1x32xf32, -1x8x-1x32xf32, -1x8x-1x32xf32]) <- (3x-1x8x-1x32xf32)
        unbind_3 = paddle._C_ops.unbind(transpose_11, 0)

        # builtin.split: (-1x8x-1x32xf32, -1x8x-1x32xf32, -1x8x-1x32xf32) <- ([-1x8x-1x32xf32, -1x8x-1x32xf32, -1x8x-1x32xf32])
        (
            split_9,
            split_10,
            split_11,
        ) = unbind_3

        # pd_op.transpose: (-1x8x32x-1xf32) <- (-1x8x-1x32xf32)
        transpose_12 = paddle._C_ops.transpose(split_10, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x32xf32, -1x8x32x-1xf32)
        matmul_19 = paddle._C_ops.matmul(split_9, transpose_12, False, False)

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(matmul_19, full_12, float("0"), True)

        # pd_op.softmax: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32)
        softmax_3 = paddle._C_ops.softmax(scale_5, -1)

        # pd_op.matmul: (-1x8x-1x32xf32) <- (-1x8x-1x-1xf32, -1x8x-1x32xf32)
        matmul_20 = paddle._C_ops.matmul(softmax_3, split_11, False, False)

        # pd_op.transpose: (-1x-1x8x32xf32) <- (-1x8x-1x32xf32)
        transpose_13 = paddle._C_ops.transpose(matmul_20, [0, 2, 1, 3])

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_10 = [slice_6, slice_7, full_13]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_8 = paddle._C_ops.stack(combine_10, 0)

        # pd_op.reshape: (-1x-1x256xf32) <- (-1x-1x8x32xf32, 3xi64)
        reshape_10 = paddle._C_ops.reshape(transpose_13, stack_8)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x256xf32, 256x256xf32)
        matmul_21 = paddle._C_ops.matmul(reshape_10, data_12, False, True)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_21 = paddle._C_ops.add(matmul_21, data_13)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_22 = paddle._C_ops.add(add_19, add_21)

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_22, layer_norm_23, layer_norm_24 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_22, parameter_7, parameter_6, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x256xf32, 256x1024xf32)
        matmul_22 = paddle._C_ops.matmul(layer_norm_22, parameter_5, False, False)

        # pd_op.add: (-1x-1x1024xf32) <- (-1x-1x1024xf32, 1024xf32)
        add_23 = paddle._C_ops.add(matmul_22, parameter_4)

        # pd_op.gelu: (-1x-1x1024xf32) <- (-1x-1x1024xf32)
        gelu_3 = paddle._C_ops.gelu(add_23, False)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x1024xf32, 1024x256xf32)
        matmul_23 = paddle._C_ops.matmul(gelu_3, parameter_3, False, False)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_24 = paddle._C_ops.add(matmul_23, parameter_2)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_25 = paddle._C_ops.add(add_22, add_24)

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_0, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_25, parameter_1, parameter_0, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        return layer_norm_0
