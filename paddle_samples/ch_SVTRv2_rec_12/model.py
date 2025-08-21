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

        # pd_op.conv2d: (-1x128x-1x80xf32) <- (-1x-1x-1x80xf32, 128x32x5x5xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_59, [1, 1], [2, 2], "EXPLICIT", [1, 1], 4, "NCHW"
        )

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_4 = [1, -1, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_58, full_int_array_4)

        # pd_op.add: (-1x128x-1x80xf32) <- (-1x128x-1x80xf32, 1x128x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_0, reshape_1)

        # pd_op.add: (-1x128x-1x80xf32) <- (-1x-1x-1x80xf32, -1x128x-1x80xf32)
        add_1 = paddle._C_ops.add(data_0, add_0)

        # pd_op.flatten: (-1x128x-1xf32) <- (-1x128x-1x80xf32)
        flatten_0 = paddle._C_ops.flatten(add_1, 2, 3)

        # pd_op.transpose: (-1x-1x128xf32) <- (-1x128x-1xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_0, [0, 2, 1])

        # pd_op.layer_norm: (-1x-1x128xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x128xf32, 128xf32, 128xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_0, parameter_57, parameter_56, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x128xf32, 128x512xf32)
        matmul_0 = paddle._C_ops.matmul(layer_norm_0, parameter_55, False, False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_2 = paddle._C_ops.add(matmul_0, parameter_54)

        # pd_op.gelu: (-1x-1x512xf32) <- (-1x-1x512xf32)
        gelu_0 = paddle._C_ops.gelu(add_2, False)

        # pd_op.matmul: (-1x-1x128xf32) <- (-1x-1x512xf32, 512x128xf32)
        matmul_1 = paddle._C_ops.matmul(gelu_0, parameter_53, False, False)

        # pd_op.add: (-1x-1x128xf32) <- (-1x-1x128xf32, 128xf32)
        add_3 = paddle._C_ops.add(matmul_1, parameter_52)

        # pd_op.add: (-1x-1x128xf32) <- (-1x-1x128xf32, -1x-1x128xf32)
        add_4 = paddle._C_ops.add(layer_norm_0, add_3)

        # pd_op.layer_norm: (-1x-1x128xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x128xf32, 128xf32, 128xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_4, parameter_51, parameter_50, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.transpose: (-1x128x-1xf32) <- (-1x-1x128xf32)
        transpose_1 = paddle._C_ops.transpose(layer_norm_3, [0, 2, 1])

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

        # pd_op.reshape: (-1x-1x-1x80xf32) <- (-1x128x-1xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(transpose_1, stack_0)

        # pd_op.shape64: (4xi64) <- (-1x-1x-1x80xf32)
        shape64_1 = paddle._C_ops.shape64(reshape_2)

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

        # pd_op.conv2d: (-1x128x-1x80xf32) <- (-1x-1x-1x80xf32, 128x32x5x5xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            reshape_2, parameter_49, [1, 1], [2, 2], "EXPLICIT", [1, 1], 4, "NCHW"
        )

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(parameter_48, full_int_array_4)

        # pd_op.add: (-1x128x-1x80xf32) <- (-1x128x-1x80xf32, 1x128x1x1xf32)
        add_5 = paddle._C_ops.add(conv2d_1, reshape_3)

        # pd_op.add: (-1x128x-1x80xf32) <- (-1x-1x-1x80xf32, -1x128x-1x80xf32)
        add_6 = paddle._C_ops.add(reshape_2, add_5)

        # pd_op.flatten: (-1x128x-1xf32) <- (-1x128x-1x80xf32)
        flatten_1 = paddle._C_ops.flatten(add_6, 2, 3)

        # pd_op.transpose: (-1x-1x128xf32) <- (-1x128x-1xf32)
        transpose_2 = paddle._C_ops.transpose(flatten_1, [0, 2, 1])

        # pd_op.layer_norm: (-1x-1x128xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x128xf32, 128xf32, 128xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_2, parameter_47, parameter_46, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x128xf32, 128x512xf32)
        matmul_2 = paddle._C_ops.matmul(layer_norm_6, parameter_45, False, False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_7 = paddle._C_ops.add(matmul_2, parameter_44)

        # pd_op.gelu: (-1x-1x512xf32) <- (-1x-1x512xf32)
        gelu_1 = paddle._C_ops.gelu(add_7, False)

        # pd_op.matmul: (-1x-1x128xf32) <- (-1x-1x512xf32, 512x128xf32)
        matmul_3 = paddle._C_ops.matmul(gelu_1, parameter_43, False, False)

        # pd_op.add: (-1x-1x128xf32) <- (-1x-1x128xf32, 128xf32)
        add_8 = paddle._C_ops.add(matmul_3, parameter_42)

        # pd_op.add: (-1x-1x128xf32) <- (-1x-1x128xf32, -1x-1x128xf32)
        add_9 = paddle._C_ops.add(layer_norm_6, add_8)

        # pd_op.layer_norm: (-1x-1x128xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x128xf32, 128xf32, 128xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_9, parameter_41, parameter_40, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.transpose: (-1x128x-1xf32) <- (-1x-1x128xf32)
        transpose_3 = paddle._C_ops.transpose(layer_norm_9, [0, 2, 1])

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_1 = [full_0, slice_4, slice_5, full_1]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_1, 0)

        # pd_op.reshape: (-1x-1x-1x80xf32) <- (-1x128x-1xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(transpose_3, stack_1)

        # pd_op.shape64: (4xi64) <- (-1x-1x-1x80xf32)
        shape64_2 = paddle._C_ops.shape64(reshape_4)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.conv2d: (-1x128x-1x80xf32) <- (-1x-1x-1x80xf32, 128x32x5x5xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            reshape_4, parameter_39, [1, 1], [2, 2], "EXPLICIT", [1, 1], 4, "NCHW"
        )

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(parameter_38, full_int_array_4)

        # pd_op.add: (-1x128x-1x80xf32) <- (-1x128x-1x80xf32, 1x128x1x1xf32)
        add_10 = paddle._C_ops.add(conv2d_2, reshape_5)

        # pd_op.add: (-1x128x-1x80xf32) <- (-1x-1x-1x80xf32, -1x128x-1x80xf32)
        add_11 = paddle._C_ops.add(reshape_4, add_10)

        # pd_op.flatten: (-1x128x-1xf32) <- (-1x128x-1x80xf32)
        flatten_2 = paddle._C_ops.flatten(add_11, 2, 3)

        # pd_op.transpose: (-1x-1x128xf32) <- (-1x128x-1xf32)
        transpose_4 = paddle._C_ops.transpose(flatten_2, [0, 2, 1])

        # pd_op.layer_norm: (-1x-1x128xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x128xf32, 128xf32, 128xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_4, parameter_37, parameter_36, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x128xf32, 128x512xf32)
        matmul_4 = paddle._C_ops.matmul(layer_norm_12, parameter_35, False, False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_12 = paddle._C_ops.add(matmul_4, parameter_34)

        # pd_op.gelu: (-1x-1x512xf32) <- (-1x-1x512xf32)
        gelu_2 = paddle._C_ops.gelu(add_12, False)

        # pd_op.matmul: (-1x-1x128xf32) <- (-1x-1x512xf32, 512x128xf32)
        matmul_5 = paddle._C_ops.matmul(gelu_2, parameter_33, False, False)

        # pd_op.add: (-1x-1x128xf32) <- (-1x-1x128xf32, 128xf32)
        add_13 = paddle._C_ops.add(matmul_5, parameter_32)

        # pd_op.add: (-1x-1x128xf32) <- (-1x-1x128xf32, -1x-1x128xf32)
        add_14 = paddle._C_ops.add(layer_norm_12, add_13)

        # pd_op.layer_norm: (-1x-1x128xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x128xf32, 128xf32, 128xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_14, parameter_31, parameter_30, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.transpose: (-1x128x-1xf32) <- (-1x-1x128xf32)
        transpose_5 = paddle._C_ops.transpose(layer_norm_15, [0, 2, 1])

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_2 = [full_0, slice_7, slice_8, full_1]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_2 = paddle._C_ops.stack(combine_2, 0)

        # pd_op.reshape: (-1x-1x-1x80xf32) <- (-1x128x-1xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(transpose_5, stack_2)

        # pd_op.shape64: (4xi64) <- (-1x-1x-1x80xf32)
        shape64_3 = paddle._C_ops.shape64(reshape_6)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.conv2d: (-1x128x-1x80xf32) <- (-1x-1x-1x80xf32, 128x32x5x5xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            reshape_6, parameter_29, [1, 1], [2, 2], "EXPLICIT", [1, 1], 4, "NCHW"
        )

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(parameter_28, full_int_array_4)

        # pd_op.add: (-1x128x-1x80xf32) <- (-1x128x-1x80xf32, 1x128x1x1xf32)
        add_15 = paddle._C_ops.add(conv2d_3, reshape_7)

        # pd_op.add: (-1x128x-1x80xf32) <- (-1x-1x-1x80xf32, -1x128x-1x80xf32)
        add_16 = paddle._C_ops.add(reshape_6, add_15)

        # pd_op.flatten: (-1x128x-1xf32) <- (-1x128x-1x80xf32)
        flatten_3 = paddle._C_ops.flatten(add_16, 2, 3)

        # pd_op.transpose: (-1x-1x128xf32) <- (-1x128x-1xf32)
        transpose_6 = paddle._C_ops.transpose(flatten_3, [0, 2, 1])

        # pd_op.layer_norm: (-1x-1x128xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x128xf32, 128xf32, 128xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_6, parameter_27, parameter_26, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x128xf32, 128x512xf32)
        matmul_6 = paddle._C_ops.matmul(layer_norm_18, parameter_25, False, False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_17 = paddle._C_ops.add(matmul_6, parameter_24)

        # pd_op.gelu: (-1x-1x512xf32) <- (-1x-1x512xf32)
        gelu_3 = paddle._C_ops.gelu(add_17, False)

        # pd_op.matmul: (-1x-1x128xf32) <- (-1x-1x512xf32, 512x128xf32)
        matmul_7 = paddle._C_ops.matmul(gelu_3, parameter_23, False, False)

        # pd_op.add: (-1x-1x128xf32) <- (-1x-1x128xf32, 128xf32)
        add_18 = paddle._C_ops.add(matmul_7, parameter_22)

        # pd_op.add: (-1x-1x128xf32) <- (-1x-1x128xf32, -1x-1x128xf32)
        add_19 = paddle._C_ops.add(layer_norm_18, add_18)

        # pd_op.layer_norm: (-1x-1x128xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x128xf32, 128xf32, 128xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_19, parameter_21, parameter_20, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.transpose: (-1x128x-1xf32) <- (-1x-1x128xf32)
        transpose_7 = paddle._C_ops.transpose(layer_norm_21, [0, 2, 1])

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_3 = [full_0, slice_10, slice_11, full_1]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_3 = paddle._C_ops.stack(combine_3, 0)

        # pd_op.reshape: (-1x-1x-1x80xf32) <- (-1x128x-1xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(transpose_7, stack_3)

        # pd_op.shape64: (4xi64) <- (-1x-1x-1x80xf32)
        shape64_4 = paddle._C_ops.shape64(reshape_8)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            shape64_4, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            shape64_4, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            shape64_4, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.conv2d: (-1x128x-1x80xf32) <- (-1x-1x-1x80xf32, 128x32x5x5xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            reshape_8, parameter_19, [1, 1], [2, 2], "EXPLICIT", [1, 1], 4, "NCHW"
        )

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(parameter_18, full_int_array_4)

        # pd_op.add: (-1x128x-1x80xf32) <- (-1x128x-1x80xf32, 1x128x1x1xf32)
        add_20 = paddle._C_ops.add(conv2d_4, reshape_9)

        # pd_op.add: (-1x128x-1x80xf32) <- (-1x-1x-1x80xf32, -1x128x-1x80xf32)
        add_21 = paddle._C_ops.add(reshape_8, add_20)

        # pd_op.flatten: (-1x128x-1xf32) <- (-1x128x-1x80xf32)
        flatten_4 = paddle._C_ops.flatten(add_21, 2, 3)

        # pd_op.transpose: (-1x-1x128xf32) <- (-1x128x-1xf32)
        transpose_8 = paddle._C_ops.transpose(flatten_4, [0, 2, 1])

        # pd_op.layer_norm: (-1x-1x128xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x128xf32, 128xf32, 128xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_8, parameter_17, parameter_16, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x128xf32, 128x512xf32)
        matmul_8 = paddle._C_ops.matmul(layer_norm_24, parameter_15, False, False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_22 = paddle._C_ops.add(matmul_8, parameter_14)

        # pd_op.gelu: (-1x-1x512xf32) <- (-1x-1x512xf32)
        gelu_4 = paddle._C_ops.gelu(add_22, False)

        # pd_op.matmul: (-1x-1x128xf32) <- (-1x-1x512xf32, 512x128xf32)
        matmul_9 = paddle._C_ops.matmul(gelu_4, parameter_13, False, False)

        # pd_op.add: (-1x-1x128xf32) <- (-1x-1x128xf32, 128xf32)
        add_23 = paddle._C_ops.add(matmul_9, parameter_12)

        # pd_op.add: (-1x-1x128xf32) <- (-1x-1x128xf32, -1x-1x128xf32)
        add_24 = paddle._C_ops.add(layer_norm_24, add_23)

        # pd_op.layer_norm: (-1x-1x128xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x128xf32, 128xf32, 128xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_24, parameter_11, parameter_10, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.transpose: (-1x128x-1xf32) <- (-1x-1x128xf32)
        transpose_9 = paddle._C_ops.transpose(layer_norm_27, [0, 2, 1])

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_4 = [full_0, slice_13, slice_14, full_1]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_4 = paddle._C_ops.stack(combine_4, 0)

        # pd_op.reshape: (-1x-1x-1x80xf32) <- (-1x128x-1xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(transpose_9, stack_4)

        # pd_op.shape64: (4xi64) <- (-1x-1x-1x80xf32)
        shape64_5 = paddle._C_ops.shape64(reshape_10)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            shape64_5, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            shape64_5, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            shape64_5, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.conv2d: (-1x128x-1x80xf32) <- (-1x-1x-1x80xf32, 128x32x5x5xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            reshape_10, parameter_9, [1, 1], [2, 2], "EXPLICIT", [1, 1], 4, "NCHW"
        )

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_11 = paddle._C_ops.reshape(parameter_8, full_int_array_4)

        # pd_op.add: (-1x128x-1x80xf32) <- (-1x128x-1x80xf32, 1x128x1x1xf32)
        add_25 = paddle._C_ops.add(conv2d_5, reshape_11)

        # pd_op.add: (-1x128x-1x80xf32) <- (-1x-1x-1x80xf32, -1x128x-1x80xf32)
        add_26 = paddle._C_ops.add(reshape_10, add_25)

        # pd_op.flatten: (-1x128x-1xf32) <- (-1x128x-1x80xf32)
        flatten_5 = paddle._C_ops.flatten(add_26, 2, 3)

        # pd_op.transpose: (-1x-1x128xf32) <- (-1x128x-1xf32)
        transpose_10 = paddle._C_ops.transpose(flatten_5, [0, 2, 1])

        # pd_op.layer_norm: (-1x-1x128xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x128xf32, 128xf32, 128xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_10, parameter_7, parameter_6, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x128xf32, 128x512xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_30, parameter_5, False, False)

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_27 = paddle._C_ops.add(matmul_10, parameter_4)

        # pd_op.gelu: (-1x-1x512xf32) <- (-1x-1x512xf32)
        gelu_5 = paddle._C_ops.gelu(add_27, False)

        # pd_op.matmul: (-1x-1x128xf32) <- (-1x-1x512xf32, 512x128xf32)
        matmul_11 = paddle._C_ops.matmul(gelu_5, parameter_3, False, False)

        # pd_op.add: (-1x-1x128xf32) <- (-1x-1x128xf32, 128xf32)
        add_28 = paddle._C_ops.add(matmul_11, parameter_2)

        # pd_op.add: (-1x-1x128xf32) <- (-1x-1x128xf32, -1x-1x128xf32)
        add_29 = paddle._C_ops.add(layer_norm_30, add_28)

        # pd_op.layer_norm: (-1x-1x128xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x128xf32, 128xf32, 128xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_29, parameter_1, parameter_0, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.transpose: (-1x128x-1xf32) <- (-1x-1x128xf32)
        transpose_11 = paddle._C_ops.transpose(layer_norm_33, [0, 2, 1])

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_5 = [full_0, slice_16, slice_17, full_1]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_5 = paddle._C_ops.stack(combine_5, 0)

        # pd_op.reshape: (-1x-1x-1x80xf32) <- (-1x128x-1xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(transpose_11, stack_5)
        return reshape_0
