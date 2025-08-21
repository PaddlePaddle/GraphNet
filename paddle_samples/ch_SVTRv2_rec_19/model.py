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
        parameter_42,
        parameter_43,
        parameter_44,
        parameter_45,
        parameter_46,
        parameter_47,
        parameter_48,
        parameter_49,
        parameter_50,
        parameter_51,
        parameter_52,
        parameter_53,
        parameter_54,
        parameter_55,
        parameter_56,
        parameter_57,
        parameter_58,
        parameter_59,
        parameter_60,
        parameter_61,
        parameter_62,
        parameter_63,
        parameter_64,
        parameter_65,
        parameter_66,
        parameter_67,
        data_0,
    ):
        # pd_op.shape64: (4xi64) <- (-1x-1x-1x80xf32)
        shape64_0 = paddle._C_ops.shape64(data_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [3]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.conv2d: (-1x256x-1x80xf32) <- (-1x-1x-1x80xf32, 256x32x5x5xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_67, [1, 1], [2, 2], "EXPLICIT", [1, 1], 8, "NCHW"
        )

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_4 = [1, -1, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_66, full_int_array_4)

        # pd_op.add: (-1x256x-1x80xf32) <- (-1x256x-1x80xf32, 1x256x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_0, reshape_0)

        # pd_op.add: (-1x256x-1x80xf32) <- (-1x-1x-1x80xf32, -1x256x-1x80xf32)
        add_1 = paddle._C_ops.add(data_0, add_0)

        # pd_op.flatten: (-1x256x-1xf32) <- (-1x256x-1x80xf32)
        flatten_0 = paddle._C_ops.flatten(add_1, 2, 3)

        # pd_op.transpose: (-1x-1x256xf32) <- (-1x256x-1xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_0, [0, 2, 1])

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_1, layer_norm_2, layer_norm_3 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_0, parameter_65, parameter_64, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x256xf32, 256x1024xf32)
        matmul_0 = paddle._C_ops.matmul(layer_norm_1, parameter_63, False, False)

        # pd_op.add: (-1x-1x1024xf32) <- (-1x-1x1024xf32, 1024xf32)
        add_2 = paddle._C_ops.add(matmul_0, parameter_62)

        # pd_op.gelu: (-1x-1x1024xf32) <- (-1x-1x1024xf32)
        gelu_0 = paddle._C_ops.gelu(add_2, False)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x1024xf32, 1024x256xf32)
        matmul_1 = paddle._C_ops.matmul(gelu_0, parameter_61, False, False)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_3 = paddle._C_ops.add(matmul_1, parameter_60)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_4 = paddle._C_ops.add(layer_norm_1, add_3)

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_4, layer_norm_5, layer_norm_6 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_4, parameter_59, parameter_58, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.transpose: (-1x256x-1xf32) <- (-1x-1x256xf32)
        transpose_1 = paddle._C_ops.transpose(layer_norm_4, [0, 2, 1])

        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("80"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_0 = [full_0, slice_1, slice_2, full_1]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)

        # pd_op.reshape: (-1x-1x-1x80xf32) <- (-1x256x-1xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(transpose_1, stack_0)

        # pd_op.shape64: (4xi64) <- (-1x-1x-1x80xf32)
        shape64_1 = paddle._C_ops.shape64(reshape_1)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.conv2d: (-1x256x-1x80xf32) <- (-1x-1x-1x80xf32, 256x32x5x5xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            reshape_1, parameter_57, [1, 1], [2, 2], "EXPLICIT", [1, 1], 8, "NCHW"
        )

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(parameter_56, full_int_array_4)

        # pd_op.add: (-1x256x-1x80xf32) <- (-1x256x-1x80xf32, 1x256x1x1xf32)
        add_5 = paddle._C_ops.add(conv2d_1, reshape_2)

        # pd_op.add: (-1x256x-1x80xf32) <- (-1x-1x-1x80xf32, -1x256x-1x80xf32)
        add_6 = paddle._C_ops.add(reshape_1, add_5)

        # pd_op.flatten: (-1x256x-1xf32) <- (-1x256x-1x80xf32)
        flatten_1 = paddle._C_ops.flatten(add_6, 2, 3)

        # pd_op.transpose: (-1x-1x256xf32) <- (-1x256x-1xf32)
        transpose_2 = paddle._C_ops.transpose(flatten_1, [0, 2, 1])

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_7, layer_norm_8, layer_norm_9 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_2, parameter_55, parameter_54, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x256xf32, 256x1024xf32)
        matmul_2 = paddle._C_ops.matmul(layer_norm_7, parameter_53, False, False)

        # pd_op.add: (-1x-1x1024xf32) <- (-1x-1x1024xf32, 1024xf32)
        add_7 = paddle._C_ops.add(matmul_2, parameter_52)

        # pd_op.gelu: (-1x-1x1024xf32) <- (-1x-1x1024xf32)
        gelu_1 = paddle._C_ops.gelu(add_7, False)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x1024xf32, 1024x256xf32)
        matmul_3 = paddle._C_ops.matmul(gelu_1, parameter_51, False, False)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_8 = paddle._C_ops.add(matmul_3, parameter_50)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_9 = paddle._C_ops.add(layer_norm_7, add_8)

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_10, layer_norm_11, layer_norm_12 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_9, parameter_49, parameter_48, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.transpose: (-1x256x-1xf32) <- (-1x-1x256xf32)
        transpose_3 = paddle._C_ops.transpose(layer_norm_10, [0, 2, 1])

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_1 = [full_0, slice_4, slice_5, full_1]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_1, 0)

        # pd_op.reshape: (-1x-1x-1x80xf32) <- (-1x256x-1xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_3, stack_1)

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x80xf32)
        flatten_2 = paddle._C_ops.flatten(reshape_3, 2, 3)

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_4 = paddle._C_ops.transpose(flatten_2, [0, 2, 1])

        # pd_op.matmul: (-1x-1x768xf32) <- (-1x-1x-1xf32, 256x768xf32)
        matmul_4 = paddle._C_ops.matmul(transpose_4, parameter_47, False, False)

        # pd_op.add: (-1x-1x768xf32) <- (-1x-1x768xf32, 768xf32)
        add_10 = paddle._C_ops.add(matmul_4, parameter_46)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_5 = [0, -1, 3, 8, 32]

        # pd_op.reshape: (-1x-1x3x8x32xf32) <- (-1x-1x768xf32, 5xi64)
        reshape_4 = paddle._C_ops.reshape(add_10, full_int_array_5)

        # pd_op.transpose: (3x-1x8x-1x32xf32) <- (-1x-1x3x8x32xf32)
        transpose_5 = paddle._C_ops.transpose(reshape_4, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x-1x32xf32) <- (3x-1x8x-1x32xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            transpose_5, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x8x-1x32xf32) <- (3x-1x8x-1x32xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            transpose_5, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (-1x8x-1x32xf32) <- (3x-1x8x-1x32xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            transpose_5, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.transpose: (-1x8x32x-1xf32) <- (-1x8x-1x32xf32)
        transpose_6 = paddle._C_ops.transpose(slice_7, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x32xf32, -1x8x32x-1xf32)
        matmul_5 = paddle._C_ops.matmul(slice_6, transpose_6, False, False)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(matmul_5, full_2, float("0"), True)

        # pd_op.softmax: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32)
        softmax_0 = paddle._C_ops.softmax(scale_0, -1)

        # pd_op.matmul: (-1x8x-1x32xf32) <- (-1x8x-1x-1xf32, -1x8x-1x32xf32)
        matmul_6 = paddle._C_ops.matmul(softmax_0, slice_8, False, False)

        # pd_op.transpose: (-1x-1x8x32xf32) <- (-1x8x-1x32xf32)
        transpose_7 = paddle._C_ops.transpose(matmul_6, [0, 2, 1, 3])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_6 = [0, -1, 256]

        # pd_op.reshape: (-1x-1x256xf32) <- (-1x-1x8x32xf32, 3xi64)
        reshape_5 = paddle._C_ops.reshape(transpose_7, full_int_array_6)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x256xf32, 256x256xf32)
        matmul_7 = paddle._C_ops.matmul(reshape_5, parameter_45, False, False)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_11 = paddle._C_ops.add(matmul_7, parameter_44)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x-1xf32, -1x-1x256xf32)
        add_12 = paddle._C_ops.add(transpose_4, add_11)

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_13, layer_norm_14, layer_norm_15 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_12, parameter_43, parameter_42, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x256xf32, 256x1024xf32)
        matmul_8 = paddle._C_ops.matmul(layer_norm_13, parameter_41, False, False)

        # pd_op.add: (-1x-1x1024xf32) <- (-1x-1x1024xf32, 1024xf32)
        add_13 = paddle._C_ops.add(matmul_8, parameter_40)

        # pd_op.gelu: (-1x-1x1024xf32) <- (-1x-1x1024xf32)
        gelu_2 = paddle._C_ops.gelu(add_13, False)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x1024xf32, 1024x256xf32)
        matmul_9 = paddle._C_ops.matmul(gelu_2, parameter_39, False, False)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_14 = paddle._C_ops.add(matmul_9, parameter_38)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_15 = paddle._C_ops.add(layer_norm_13, add_14)

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_16, layer_norm_17, layer_norm_18 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_15, parameter_37, parameter_36, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x-1x768xf32) <- (-1x-1x256xf32, 256x768xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_16, parameter_35, False, False)

        # pd_op.add: (-1x-1x768xf32) <- (-1x-1x768xf32, 768xf32)
        add_16 = paddle._C_ops.add(matmul_10, parameter_34)

        # pd_op.reshape: (-1x-1x3x8x32xf32) <- (-1x-1x768xf32, 5xi64)
        reshape_6 = paddle._C_ops.reshape(add_16, full_int_array_5)

        # pd_op.transpose: (3x-1x8x-1x32xf32) <- (-1x-1x3x8x32xf32)
        transpose_8 = paddle._C_ops.transpose(reshape_6, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x-1x32xf32) <- (3x-1x8x-1x32xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            transpose_8, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x8x-1x32xf32) <- (3x-1x8x-1x32xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            transpose_8, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (-1x8x-1x32xf32) <- (3x-1x8x-1x32xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            transpose_8, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.transpose: (-1x8x32x-1xf32) <- (-1x8x-1x32xf32)
        transpose_9 = paddle._C_ops.transpose(slice_10, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x32xf32, -1x8x32x-1xf32)
        matmul_11 = paddle._C_ops.matmul(slice_9, transpose_9, False, False)

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(matmul_11, full_2, float("0"), True)

        # pd_op.softmax: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32)
        softmax_1 = paddle._C_ops.softmax(scale_1, -1)

        # pd_op.matmul: (-1x8x-1x32xf32) <- (-1x8x-1x-1xf32, -1x8x-1x32xf32)
        matmul_12 = paddle._C_ops.matmul(softmax_1, slice_11, False, False)

        # pd_op.transpose: (-1x-1x8x32xf32) <- (-1x8x-1x32xf32)
        transpose_10 = paddle._C_ops.transpose(matmul_12, [0, 2, 1, 3])

        # pd_op.reshape: (-1x-1x256xf32) <- (-1x-1x8x32xf32, 3xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_10, full_int_array_6)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x256xf32, 256x256xf32)
        matmul_13 = paddle._C_ops.matmul(reshape_7, parameter_33, False, False)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_17 = paddle._C_ops.add(matmul_13, parameter_32)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_18 = paddle._C_ops.add(layer_norm_16, add_17)

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_19, layer_norm_20, layer_norm_21 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_18, parameter_31, parameter_30, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x256xf32, 256x1024xf32)
        matmul_14 = paddle._C_ops.matmul(layer_norm_19, parameter_29, False, False)

        # pd_op.add: (-1x-1x1024xf32) <- (-1x-1x1024xf32, 1024xf32)
        add_19 = paddle._C_ops.add(matmul_14, parameter_28)

        # pd_op.gelu: (-1x-1x1024xf32) <- (-1x-1x1024xf32)
        gelu_3 = paddle._C_ops.gelu(add_19, False)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x1024xf32, 1024x256xf32)
        matmul_15 = paddle._C_ops.matmul(gelu_3, parameter_27, False, False)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_20 = paddle._C_ops.add(matmul_15, parameter_26)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_21 = paddle._C_ops.add(layer_norm_19, add_20)

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_22, layer_norm_23, layer_norm_24 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_21, parameter_25, parameter_24, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x-1x768xf32) <- (-1x-1x256xf32, 256x768xf32)
        matmul_16 = paddle._C_ops.matmul(layer_norm_22, parameter_23, False, False)

        # pd_op.add: (-1x-1x768xf32) <- (-1x-1x768xf32, 768xf32)
        add_22 = paddle._C_ops.add(matmul_16, parameter_22)

        # pd_op.reshape: (-1x-1x3x8x32xf32) <- (-1x-1x768xf32, 5xi64)
        reshape_8 = paddle._C_ops.reshape(add_22, full_int_array_5)

        # pd_op.transpose: (3x-1x8x-1x32xf32) <- (-1x-1x3x8x32xf32)
        transpose_11 = paddle._C_ops.transpose(reshape_8, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x-1x32xf32) <- (3x-1x8x-1x32xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            transpose_11, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x8x-1x32xf32) <- (3x-1x8x-1x32xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            transpose_11, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (-1x8x-1x32xf32) <- (3x-1x8x-1x32xf32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            transpose_11, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.transpose: (-1x8x32x-1xf32) <- (-1x8x-1x32xf32)
        transpose_12 = paddle._C_ops.transpose(slice_13, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x32xf32, -1x8x32x-1xf32)
        matmul_17 = paddle._C_ops.matmul(slice_12, transpose_12, False, False)

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(matmul_17, full_2, float("0"), True)

        # pd_op.softmax: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32)
        softmax_2 = paddle._C_ops.softmax(scale_2, -1)

        # pd_op.matmul: (-1x8x-1x32xf32) <- (-1x8x-1x-1xf32, -1x8x-1x32xf32)
        matmul_18 = paddle._C_ops.matmul(softmax_2, slice_14, False, False)

        # pd_op.transpose: (-1x-1x8x32xf32) <- (-1x8x-1x32xf32)
        transpose_13 = paddle._C_ops.transpose(matmul_18, [0, 2, 1, 3])

        # pd_op.reshape: (-1x-1x256xf32) <- (-1x-1x8x32xf32, 3xi64)
        reshape_9 = paddle._C_ops.reshape(transpose_13, full_int_array_6)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x256xf32, 256x256xf32)
        matmul_19 = paddle._C_ops.matmul(reshape_9, parameter_21, False, False)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_23 = paddle._C_ops.add(matmul_19, parameter_20)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_24 = paddle._C_ops.add(layer_norm_22, add_23)

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_25, layer_norm_26, layer_norm_27 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_24, parameter_19, parameter_18, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x256xf32, 256x1024xf32)
        matmul_20 = paddle._C_ops.matmul(layer_norm_25, parameter_17, False, False)

        # pd_op.add: (-1x-1x1024xf32) <- (-1x-1x1024xf32, 1024xf32)
        add_25 = paddle._C_ops.add(matmul_20, parameter_16)

        # pd_op.gelu: (-1x-1x1024xf32) <- (-1x-1x1024xf32)
        gelu_4 = paddle._C_ops.gelu(add_25, False)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x1024xf32, 1024x256xf32)
        matmul_21 = paddle._C_ops.matmul(gelu_4, parameter_15, False, False)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_26 = paddle._C_ops.add(matmul_21, parameter_14)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_27 = paddle._C_ops.add(layer_norm_25, add_26)

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_28, layer_norm_29, layer_norm_30 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_27, parameter_13, parameter_12, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x-1x768xf32) <- (-1x-1x256xf32, 256x768xf32)
        matmul_22 = paddle._C_ops.matmul(layer_norm_28, parameter_11, False, False)

        # pd_op.add: (-1x-1x768xf32) <- (-1x-1x768xf32, 768xf32)
        add_28 = paddle._C_ops.add(matmul_22, parameter_10)

        # pd_op.reshape: (-1x-1x3x8x32xf32) <- (-1x-1x768xf32, 5xi64)
        reshape_10 = paddle._C_ops.reshape(add_28, full_int_array_5)

        # pd_op.transpose: (3x-1x8x-1x32xf32) <- (-1x-1x3x8x32xf32)
        transpose_14 = paddle._C_ops.transpose(reshape_10, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x8x-1x32xf32) <- (3x-1x8x-1x32xf32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            transpose_14, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x8x-1x32xf32) <- (3x-1x8x-1x32xf32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            transpose_14, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (-1x8x-1x32xf32) <- (3x-1x8x-1x32xf32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            transpose_14, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.transpose: (-1x8x32x-1xf32) <- (-1x8x-1x32xf32)
        transpose_15 = paddle._C_ops.transpose(slice_16, [0, 1, 3, 2])

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x32xf32, -1x8x32x-1xf32)
        matmul_23 = paddle._C_ops.matmul(slice_15, transpose_15, False, False)

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(matmul_23, full_2, float("0"), True)

        # pd_op.softmax: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32)
        softmax_3 = paddle._C_ops.softmax(scale_3, -1)

        # pd_op.matmul: (-1x8x-1x32xf32) <- (-1x8x-1x-1xf32, -1x8x-1x32xf32)
        matmul_24 = paddle._C_ops.matmul(softmax_3, slice_17, False, False)

        # pd_op.transpose: (-1x-1x8x32xf32) <- (-1x8x-1x32xf32)
        transpose_16 = paddle._C_ops.transpose(matmul_24, [0, 2, 1, 3])

        # pd_op.reshape: (-1x-1x256xf32) <- (-1x-1x8x32xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_16, full_int_array_6)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x256xf32, 256x256xf32)
        matmul_25 = paddle._C_ops.matmul(reshape_11, parameter_9, False, False)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_29 = paddle._C_ops.add(matmul_25, parameter_8)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_30 = paddle._C_ops.add(layer_norm_28, add_29)

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_31, layer_norm_32, layer_norm_33 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_30, parameter_7, parameter_6, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x256xf32, 256x1024xf32)
        matmul_26 = paddle._C_ops.matmul(layer_norm_31, parameter_5, False, False)

        # pd_op.add: (-1x-1x1024xf32) <- (-1x-1x1024xf32, 1024xf32)
        add_31 = paddle._C_ops.add(matmul_26, parameter_4)

        # pd_op.gelu: (-1x-1x1024xf32) <- (-1x-1x1024xf32)
        gelu_5 = paddle._C_ops.gelu(add_31, False)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x1024xf32, 1024x256xf32)
        matmul_27 = paddle._C_ops.matmul(gelu_5, parameter_3, False, False)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_32 = paddle._C_ops.add(matmul_27, parameter_2)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_33 = paddle._C_ops.add(layer_norm_31, add_32)

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_0, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_33, parameter_1, parameter_0, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        return layer_norm_0
