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
        parameter_68,
        parameter_69,
        parameter_70,
        parameter_71,
        data_0,
        data_1,
        data_2,
        data_3,
        data_4,
        data_5,
        data_6,
        data_7,
        data_8,
    ):
        # pd_op.conv2d: (-1x256x-1x80xf32) <- (-1x-1x-1x80xf32, 256x32x5x5xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_8, parameter_71, [1, 1], [2, 2], "EXPLICIT", [1, 1], 8, "NCHW"
        )
        del parameter_71

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_70, full_int_array_0)
        del parameter_70

        # pd_op.add: (-1x256x-1x80xf32) <- (-1x256x-1x80xf32, 1x256x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_0, reshape_0)
        del conv2d_0, reshape_0

        # pd_op.add: (-1x256x-1x80xf32) <- (-1x-1x-1x80xf32, -1x256x-1x80xf32)
        add_1 = paddle._C_ops.add(data_8, add_0)
        del add_0, data_8

        # pd_op.flatten: (-1x256x-1xf32) <- (-1x256x-1x80xf32)
        flatten_0 = paddle._C_ops.flatten(add_1, 2, 3)
        del add_1

        # pd_op.transpose: (-1x-1x256xf32) <- (-1x256x-1xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_0, [0, 2, 1])
        del flatten_0

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_1, layer_norm_2, layer_norm_3 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_0, parameter_69, parameter_68, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_68, parameter_69, transpose_0

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x256xf32, 256x1024xf32)
        matmul_0 = paddle._C_ops.matmul(layer_norm_1, parameter_67, False, False)
        del parameter_67

        # pd_op.add: (-1x-1x1024xf32) <- (-1x-1x1024xf32, 1024xf32)
        add_2 = paddle._C_ops.add(matmul_0, parameter_66)
        del matmul_0, parameter_66

        # pd_op.gelu: (-1x-1x1024xf32) <- (-1x-1x1024xf32)
        gelu_0 = paddle._C_ops.gelu(add_2, False)
        del add_2

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x1024xf32, 1024x256xf32)
        matmul_1 = paddle._C_ops.matmul(gelu_0, parameter_65, False, False)
        del gelu_0, parameter_65

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_3 = paddle._C_ops.add(matmul_1, parameter_64)
        del matmul_1, parameter_64

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_4 = paddle._C_ops.add(layer_norm_1, add_3)
        del add_3, layer_norm_1

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_4, layer_norm_5, layer_norm_6 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_4, parameter_63, parameter_62, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_4, parameter_62, parameter_63

        # pd_op.transpose: (-1x256x-1xf32) <- (-1x-1x256xf32)
        transpose_1 = paddle._C_ops.transpose(layer_norm_4, [0, 2, 1])
        del layer_norm_4

        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("80"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_0 = [full_0, data_0, data_1, full_1]
        del data_0, data_1

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.reshape: (-1x-1x-1x80xf32) <- (-1x256x-1xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(transpose_1, stack_0)
        del stack_0, transpose_1

        # pd_op.shape64: (4xi64) <- (-1x-1x-1x80xf32)
        shape64_0 = paddle._C_ops.shape64(reshape_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [1]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [2]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [3]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del shape64_0

        # pd_op.conv2d: (-1x256x-1x80xf32) <- (-1x-1x-1x80xf32, 256x32x5x5xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            reshape_1, parameter_61, [1, 1], [2, 2], "EXPLICIT", [1, 1], 8, "NCHW"
        )
        del parameter_61

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(parameter_60, full_int_array_0)
        del parameter_60

        # pd_op.add: (-1x256x-1x80xf32) <- (-1x256x-1x80xf32, 1x256x1x1xf32)
        add_5 = paddle._C_ops.add(conv2d_1, reshape_2)
        del conv2d_1, reshape_2

        # pd_op.add: (-1x256x-1x80xf32) <- (-1x-1x-1x80xf32, -1x256x-1x80xf32)
        add_6 = paddle._C_ops.add(reshape_1, add_5)
        del add_5, reshape_1

        # pd_op.flatten: (-1x256x-1xf32) <- (-1x256x-1x80xf32)
        flatten_1 = paddle._C_ops.flatten(add_6, 2, 3)
        del add_6

        # pd_op.transpose: (-1x-1x256xf32) <- (-1x256x-1xf32)
        transpose_2 = paddle._C_ops.transpose(flatten_1, [0, 2, 1])
        del flatten_1

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_7, layer_norm_8, layer_norm_9 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_2, parameter_59, parameter_58, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_58, parameter_59, transpose_2

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x256xf32, 256x1024xf32)
        matmul_2 = paddle._C_ops.matmul(layer_norm_7, parameter_57, False, False)
        del parameter_57

        # pd_op.add: (-1x-1x1024xf32) <- (-1x-1x1024xf32, 1024xf32)
        add_7 = paddle._C_ops.add(matmul_2, parameter_56)
        del matmul_2, parameter_56

        # pd_op.gelu: (-1x-1x1024xf32) <- (-1x-1x1024xf32)
        gelu_1 = paddle._C_ops.gelu(add_7, False)
        del add_7

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x1024xf32, 1024x256xf32)
        matmul_3 = paddle._C_ops.matmul(gelu_1, parameter_55, False, False)
        del gelu_1, parameter_55

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_8 = paddle._C_ops.add(matmul_3, parameter_54)
        del matmul_3, parameter_54

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_9 = paddle._C_ops.add(layer_norm_7, add_8)
        del add_8, layer_norm_7

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_10, layer_norm_11, layer_norm_12 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_9, parameter_53, parameter_52, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_9, parameter_52, parameter_53

        # pd_op.transpose: (-1x256x-1xf32) <- (-1x-1x256xf32)
        transpose_3 = paddle._C_ops.transpose(layer_norm_10, [0, 2, 1])
        del layer_norm_10

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_1 = [full_0, slice_1, slice_2, full_1]
        del full_1, slice_1, slice_2

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_1, 0)
        del combine_1

        # pd_op.reshape: (-1x-1x-1x80xf32) <- (-1x256x-1xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_3, stack_1)
        del stack_1, transpose_3

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x80xf32)
        flatten_2 = paddle._C_ops.flatten(reshape_3, 2, 3)
        del reshape_3

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_4 = paddle._C_ops.transpose(flatten_2, [0, 2, 1])
        del flatten_2

        # pd_op.matmul: (-1x-1x768xf32) <- (-1x-1x-1xf32, 256x768xf32)
        matmul_4 = paddle._C_ops.matmul(transpose_4, parameter_51, False, False)
        del parameter_51

        # pd_op.add: (-1x-1x768xf32) <- (-1x-1x768xf32, 768xf32)
        add_10 = paddle._C_ops.add(matmul_4, parameter_50)
        del matmul_4, parameter_50

        # pd_op.full: (xi64) <- ()
        full_2 = paddle._C_ops.full(
            [], float("-1"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_3 = paddle._C_ops.full(
            [], float("3"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_4 = paddle._C_ops.full(
            [], float("32"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_2 = [full_0, full_2, full_3, data_2, full_4]
        del data_2

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_2 = paddle._C_ops.stack(combine_2, 0)
        del combine_2

        # pd_op.reshape: (-1x-1x3x-1x32xf32) <- (-1x-1x768xf32, 5xi64)
        reshape_4 = paddle._C_ops.reshape(add_10, stack_2)
        del add_10, stack_2

        # pd_op.transpose: (3x-1x-1x-1x32xf32) <- (-1x-1x3x-1x32xf32)
        transpose_5 = paddle._C_ops.transpose(reshape_4, [2, 0, 3, 1, 4])
        del reshape_4

        # pd_op.slice: (-1x-1x-1x32xf32) <- (3x-1x-1x-1x32xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            transpose_5, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (-1x-1x-1x32xf32) <- (3x-1x-1x-1x32xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            transpose_5, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x-1x-1x32xf32) <- (3x-1x-1x-1x32xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            transpose_5, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del transpose_5

        # pd_op.transpose: (-1x-1x32x-1xf32) <- (-1x-1x-1x32xf32)
        transpose_6 = paddle._C_ops.transpose(slice_4, [0, 1, 3, 2])
        del slice_4

        # pd_op.matmul: (-1x-1x-1x-1xf32) <- (-1x-1x-1x32xf32, -1x-1x32x-1xf32)
        matmul_5 = paddle._C_ops.matmul(slice_3, transpose_6, False, False)
        del slice_3, transpose_6

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(matmul_5, full_5, float("0"), True)
        del matmul_5

        # pd_op.softmax: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        softmax_0 = paddle._C_ops.softmax(scale_0, -1)
        del scale_0

        # pd_op.matmul: (-1x-1x-1x32xf32) <- (-1x-1x-1x-1xf32, -1x-1x-1x32xf32)
        matmul_6 = paddle._C_ops.matmul(softmax_0, slice_5, False, False)
        del slice_5, softmax_0

        # pd_op.transpose: (-1x-1x-1x32xf32) <- (-1x-1x-1x32xf32)
        transpose_7 = paddle._C_ops.transpose(matmul_6, [0, 2, 1, 3])
        del matmul_6

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_3 = [full_0, full_2, data_3]
        del data_3

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_3 = paddle._C_ops.stack(combine_3, 0)
        del combine_3

        # pd_op.reshape: (-1x-1x-1xf32) <- (-1x-1x-1x32xf32, 3xi64)
        reshape_5 = paddle._C_ops.reshape(transpose_7, stack_3)
        del stack_3, transpose_7

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x-1xf32, 256x256xf32)
        matmul_7 = paddle._C_ops.matmul(reshape_5, parameter_49, False, False)
        del parameter_49, reshape_5

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_11 = paddle._C_ops.add(matmul_7, parameter_48)
        del matmul_7, parameter_48

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x-1xf32, -1x-1x256xf32)
        add_12 = paddle._C_ops.add(transpose_4, add_11)
        del add_11, transpose_4

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_13, layer_norm_14, layer_norm_15 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_12, parameter_47, parameter_46, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_12, parameter_46, parameter_47

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x256xf32, 256x1024xf32)
        matmul_8 = paddle._C_ops.matmul(layer_norm_13, parameter_45, False, False)
        del parameter_45

        # pd_op.add: (-1x-1x1024xf32) <- (-1x-1x1024xf32, 1024xf32)
        add_13 = paddle._C_ops.add(matmul_8, parameter_44)
        del matmul_8, parameter_44

        # pd_op.gelu: (-1x-1x1024xf32) <- (-1x-1x1024xf32)
        gelu_2 = paddle._C_ops.gelu(add_13, False)
        del add_13

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x1024xf32, 1024x256xf32)
        matmul_9 = paddle._C_ops.matmul(gelu_2, parameter_43, False, False)
        del gelu_2, parameter_43

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_14 = paddle._C_ops.add(matmul_9, parameter_42)
        del matmul_9, parameter_42

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_15 = paddle._C_ops.add(layer_norm_13, add_14)
        del add_14, layer_norm_13

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_16, layer_norm_17, layer_norm_18 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_15, parameter_41, parameter_40, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_15, parameter_40, parameter_41

        # pd_op.matmul: (-1x-1x768xf32) <- (-1x-1x256xf32, 256x768xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_16, parameter_39, False, False)
        del parameter_39

        # pd_op.add: (-1x-1x768xf32) <- (-1x-1x768xf32, 768xf32)
        add_16 = paddle._C_ops.add(matmul_10, parameter_38)
        del matmul_10, parameter_38

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_4 = [full_0, full_2, full_3, data_4, full_4]
        del data_4

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_4 = paddle._C_ops.stack(combine_4, 0)
        del combine_4

        # pd_op.reshape: (-1x-1x3x-1x32xf32) <- (-1x-1x768xf32, 5xi64)
        reshape_6 = paddle._C_ops.reshape(add_16, stack_4)
        del add_16, stack_4

        # pd_op.transpose: (3x-1x-1x-1x32xf32) <- (-1x-1x3x-1x32xf32)
        transpose_8 = paddle._C_ops.transpose(reshape_6, [2, 0, 3, 1, 4])
        del reshape_6

        # pd_op.slice: (-1x-1x-1x32xf32) <- (3x-1x-1x-1x32xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            transpose_8, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (-1x-1x-1x32xf32) <- (3x-1x-1x-1x32xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            transpose_8, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x-1x-1x32xf32) <- (3x-1x-1x-1x32xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            transpose_8, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del transpose_8

        # pd_op.transpose: (-1x-1x32x-1xf32) <- (-1x-1x-1x32xf32)
        transpose_9 = paddle._C_ops.transpose(slice_7, [0, 1, 3, 2])
        del slice_7

        # pd_op.matmul: (-1x-1x-1x-1xf32) <- (-1x-1x-1x32xf32, -1x-1x32x-1xf32)
        matmul_11 = paddle._C_ops.matmul(slice_6, transpose_9, False, False)
        del slice_6, transpose_9

        # pd_op.scale: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(matmul_11, full_5, float("0"), True)
        del matmul_11

        # pd_op.softmax: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        softmax_1 = paddle._C_ops.softmax(scale_1, -1)
        del scale_1

        # pd_op.matmul: (-1x-1x-1x32xf32) <- (-1x-1x-1x-1xf32, -1x-1x-1x32xf32)
        matmul_12 = paddle._C_ops.matmul(softmax_1, slice_8, False, False)
        del slice_8, softmax_1

        # pd_op.transpose: (-1x-1x-1x32xf32) <- (-1x-1x-1x32xf32)
        transpose_10 = paddle._C_ops.transpose(matmul_12, [0, 2, 1, 3])
        del matmul_12

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_5 = [full_0, full_2, data_5]
        del data_5

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_5 = paddle._C_ops.stack(combine_5, 0)
        del combine_5

        # pd_op.reshape: (-1x-1x-1xf32) <- (-1x-1x-1x32xf32, 3xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_10, stack_5)
        del stack_5, transpose_10

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x-1xf32, 256x256xf32)
        matmul_13 = paddle._C_ops.matmul(reshape_7, parameter_37, False, False)
        del parameter_37, reshape_7

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_17 = paddle._C_ops.add(matmul_13, parameter_36)
        del matmul_13, parameter_36

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_18 = paddle._C_ops.add(layer_norm_16, add_17)
        del add_17, layer_norm_16

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_19, layer_norm_20, layer_norm_21 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_18, parameter_35, parameter_34, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_18, parameter_34, parameter_35

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x256xf32, 256x1024xf32)
        matmul_14 = paddle._C_ops.matmul(layer_norm_19, parameter_33, False, False)
        del parameter_33

        # pd_op.add: (-1x-1x1024xf32) <- (-1x-1x1024xf32, 1024xf32)
        add_19 = paddle._C_ops.add(matmul_14, parameter_32)
        del matmul_14, parameter_32

        # pd_op.gelu: (-1x-1x1024xf32) <- (-1x-1x1024xf32)
        gelu_3 = paddle._C_ops.gelu(add_19, False)
        del add_19

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x1024xf32, 1024x256xf32)
        matmul_15 = paddle._C_ops.matmul(gelu_3, parameter_31, False, False)
        del gelu_3, parameter_31

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_20 = paddle._C_ops.add(matmul_15, parameter_30)
        del matmul_15, parameter_30

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_21 = paddle._C_ops.add(layer_norm_19, add_20)
        del add_20, layer_norm_19

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_22, layer_norm_23, layer_norm_24 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_21, parameter_29, parameter_28, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_21, parameter_28, parameter_29

        # pd_op.matmul: (-1x-1x768xf32) <- (-1x-1x256xf32, 256x768xf32)
        matmul_16 = paddle._C_ops.matmul(layer_norm_22, parameter_27, False, False)
        del parameter_27

        # pd_op.add: (-1x-1x768xf32) <- (-1x-1x768xf32, 768xf32)
        add_22 = paddle._C_ops.add(matmul_16, parameter_26)
        del matmul_16, parameter_26

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_6 = [full_0, full_2, full_3, data_6, full_4]
        del data_6, full_3, full_4

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_6 = paddle._C_ops.stack(combine_6, 0)
        del combine_6

        # pd_op.reshape: (-1x-1x3x-1x32xf32) <- (-1x-1x768xf32, 5xi64)
        reshape_8 = paddle._C_ops.reshape(add_22, stack_6)
        del add_22, stack_6

        # pd_op.transpose: (3x-1x-1x-1x32xf32) <- (-1x-1x3x-1x32xf32)
        transpose_11 = paddle._C_ops.transpose(reshape_8, [2, 0, 3, 1, 4])
        del reshape_8

        # pd_op.slice: (-1x-1x-1x32xf32) <- (3x-1x-1x-1x32xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            transpose_11, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (-1x-1x-1x32xf32) <- (3x-1x-1x-1x32xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            transpose_11, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x-1x-1x32xf32) <- (3x-1x-1x-1x32xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            transpose_11, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del transpose_11

        # pd_op.transpose: (-1x-1x32x-1xf32) <- (-1x-1x-1x32xf32)
        transpose_12 = paddle._C_ops.transpose(slice_10, [0, 1, 3, 2])
        del slice_10

        # pd_op.matmul: (-1x-1x-1x-1xf32) <- (-1x-1x-1x32xf32, -1x-1x32x-1xf32)
        matmul_17 = paddle._C_ops.matmul(slice_9, transpose_12, False, False)
        del slice_9, transpose_12

        # pd_op.scale: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(matmul_17, full_5, float("0"), True)
        del matmul_17

        # pd_op.softmax: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        softmax_2 = paddle._C_ops.softmax(scale_2, -1)
        del scale_2

        # pd_op.matmul: (-1x-1x-1x32xf32) <- (-1x-1x-1x-1xf32, -1x-1x-1x32xf32)
        matmul_18 = paddle._C_ops.matmul(softmax_2, slice_11, False, False)
        del slice_11, softmax_2

        # pd_op.transpose: (-1x-1x-1x32xf32) <- (-1x-1x-1x32xf32)
        transpose_13 = paddle._C_ops.transpose(matmul_18, [0, 2, 1, 3])
        del matmul_18

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_7 = [full_0, full_2, data_7]
        del data_7, full_0, full_2

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_7 = paddle._C_ops.stack(combine_7, 0)
        del combine_7

        # pd_op.reshape: (-1x-1x-1xf32) <- (-1x-1x-1x32xf32, 3xi64)
        reshape_9 = paddle._C_ops.reshape(transpose_13, stack_7)
        del stack_7, transpose_13

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x-1xf32, 256x256xf32)
        matmul_19 = paddle._C_ops.matmul(reshape_9, parameter_25, False, False)
        del parameter_25, reshape_9

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_23 = paddle._C_ops.add(matmul_19, parameter_24)
        del matmul_19, parameter_24

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_24 = paddle._C_ops.add(layer_norm_22, add_23)
        del add_23, layer_norm_22

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_25, layer_norm_26, layer_norm_27 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_24, parameter_23, parameter_22, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_24, parameter_22, parameter_23

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x256xf32, 256x1024xf32)
        matmul_20 = paddle._C_ops.matmul(layer_norm_25, parameter_21, False, False)
        del parameter_21

        # pd_op.add: (-1x-1x1024xf32) <- (-1x-1x1024xf32, 1024xf32)
        add_25 = paddle._C_ops.add(matmul_20, parameter_20)
        del matmul_20, parameter_20

        # pd_op.gelu: (-1x-1x1024xf32) <- (-1x-1x1024xf32)
        gelu_4 = paddle._C_ops.gelu(add_25, False)
        del add_25

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x1024xf32, 1024x256xf32)
        matmul_21 = paddle._C_ops.matmul(gelu_4, parameter_19, False, False)
        del gelu_4, parameter_19

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_26 = paddle._C_ops.add(matmul_21, parameter_18)
        del matmul_21, parameter_18

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_27 = paddle._C_ops.add(layer_norm_25, add_26)
        del add_26, layer_norm_25

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_28, layer_norm_29, layer_norm_30 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_27, parameter_17, parameter_16, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_27, parameter_16, parameter_17

        # pd_op.matmul: (-1x-1x768xf32) <- (-1x-1x256xf32, 256x768xf32)
        matmul_22 = paddle._C_ops.matmul(layer_norm_28, parameter_15, False, False)
        del parameter_15

        # pd_op.add: (-1x-1x768xf32) <- (-1x-1x768xf32, 768xf32)
        add_28 = paddle._C_ops.add(matmul_22, parameter_14)
        del matmul_22, parameter_14

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_5 = [0, -1, 3, 8, 32]

        # pd_op.reshape: (-1x-1x3x8x32xf32) <- (-1x-1x768xf32, 5xi64)
        reshape_10 = paddle._C_ops.reshape(add_28, full_int_array_5)
        del add_28, full_int_array_5

        # pd_op.transpose: (3x-1x8x-1x32xf32) <- (-1x-1x3x8x32xf32)
        transpose_14 = paddle._C_ops.transpose(reshape_10, [2, 0, 3, 1, 4])
        del reshape_10

        # pd_op.slice: (-1x8x-1x32xf32) <- (3x-1x8x-1x32xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            transpose_14, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (-1x8x-1x32xf32) <- (3x-1x8x-1x32xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            transpose_14, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x8x-1x32xf32) <- (3x-1x8x-1x32xf32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            transpose_14, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del full_int_array_4, transpose_14

        # pd_op.transpose: (-1x8x32x-1xf32) <- (-1x8x-1x32xf32)
        transpose_15 = paddle._C_ops.transpose(slice_13, [0, 1, 3, 2])
        del slice_13

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x32xf32, -1x8x32x-1xf32)
        matmul_23 = paddle._C_ops.matmul(slice_12, transpose_15, False, False)
        del slice_12, transpose_15

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(matmul_23, full_5, float("0"), True)
        del full_5, matmul_23

        # pd_op.softmax: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32)
        softmax_3 = paddle._C_ops.softmax(scale_3, -1)
        del scale_3

        # pd_op.matmul: (-1x8x-1x32xf32) <- (-1x8x-1x-1xf32, -1x8x-1x32xf32)
        matmul_24 = paddle._C_ops.matmul(softmax_3, slice_14, False, False)
        del slice_14, softmax_3

        # pd_op.transpose: (-1x-1x8x32xf32) <- (-1x8x-1x32xf32)
        transpose_16 = paddle._C_ops.transpose(matmul_24, [0, 2, 1, 3])
        del matmul_24

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_6 = [0, -1, 256]

        # pd_op.reshape: (-1x-1x256xf32) <- (-1x-1x8x32xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_16, full_int_array_6)
        del full_int_array_6, transpose_16

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x256xf32, 256x256xf32)
        matmul_25 = paddle._C_ops.matmul(reshape_11, parameter_13, False, False)
        del parameter_13, reshape_11

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_29 = paddle._C_ops.add(matmul_25, parameter_12)
        del matmul_25, parameter_12

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_30 = paddle._C_ops.add(layer_norm_28, add_29)
        del add_29, layer_norm_28

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_31, layer_norm_32, layer_norm_33 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_30, parameter_11, parameter_10, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_30, parameter_10, parameter_11

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x256xf32, 256x1024xf32)
        matmul_26 = paddle._C_ops.matmul(layer_norm_31, parameter_9, False, False)
        del parameter_9

        # pd_op.add: (-1x-1x1024xf32) <- (-1x-1x1024xf32, 1024xf32)
        add_31 = paddle._C_ops.add(matmul_26, parameter_8)
        del matmul_26, parameter_8

        # pd_op.gelu: (-1x-1x1024xf32) <- (-1x-1x1024xf32)
        gelu_5 = paddle._C_ops.gelu(add_31, False)
        del add_31

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x1024xf32, 1024x256xf32)
        matmul_27 = paddle._C_ops.matmul(gelu_5, parameter_7, False, False)
        del gelu_5, parameter_7

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_32 = paddle._C_ops.add(matmul_27, parameter_6)
        del matmul_27, parameter_6

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, -1x-1x256xf32)
        add_33 = paddle._C_ops.add(layer_norm_31, add_32)
        del add_32, layer_norm_31

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_34, layer_norm_35, layer_norm_36 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_33, parameter_5, parameter_4, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_33, parameter_4, parameter_5

        # pd_op.shape64: (3xi64) <- (-1x-1x256xf32)
        shape64_1 = paddle._C_ops.shape64(layer_norm_34)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del full_int_array_3, shape64_1

        # pd_op.transpose: (-1x256x-1xf32) <- (-1x-1x256xf32)
        transpose_17 = paddle._C_ops.transpose(layer_norm_34, [0, 2, 1])
        del layer_norm_34

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_7 = [0, 256, 6, 80]

        # pd_op.reshape: (-1x256x6x80xf32) <- (-1x256x-1xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(transpose_17, full_int_array_7)
        del full_int_array_7, transpose_17

        # pd_op.conv2d: (-1x384x3x80xf32) <- (-1x256x6x80xf32, 384x256x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            reshape_12, parameter_3, [2, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_3, reshape_12

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(parameter_2, full_int_array_0)
        del full_int_array_0, parameter_2

        # pd_op.add: (-1x384x3x80xf32) <- (-1x384x3x80xf32, 1x384x1x1xf32)
        add_34 = paddle._C_ops.add(conv2d_2, reshape_13)
        del conv2d_2, reshape_13

        # pd_op.shape64: (4xi64) <- (-1x384x3x80xf32)
        shape64_2 = paddle._C_ops.shape64(add_34)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del full_int_array_1, full_int_array_2, shape64_2

        # pd_op.flatten: (-1x384x240xf32) <- (-1x384x3x80xf32)
        flatten_3 = paddle._C_ops.flatten(add_34, 2, 3)
        del add_34

        # pd_op.transpose: (-1x240x384xf32) <- (-1x384x240xf32)
        transpose_18 = paddle._C_ops.transpose(flatten_3, [0, 2, 1])
        del flatten_3

        # pd_op.layer_norm: (-1x240x384xf32, -1x240xf32, -1x240xf32) <- (-1x240x384xf32, 384xf32, 384xf32)
        layer_norm_0, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_18, parameter_1, parameter_0, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_0, parameter_1, transpose_18

        return layer_norm_0
