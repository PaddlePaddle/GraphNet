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
    ):
        # pd_op.matmul: (-1x-1x1152xf32) <- (-1x-1x-1xf32, 384x1152xf32)
        matmul_0 = paddle._C_ops.matmul(data_6, parameter_71, False, False)
        del parameter_71

        # pd_op.add: (-1x-1x1152xf32) <- (-1x-1x1152xf32, 1152xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_70)
        del matmul_0, parameter_70

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_0 = [0, -1, 3, 12, 32]

        # pd_op.reshape: (-1x-1x3x12x32xf32) <- (-1x-1x1152xf32, 5xi64)
        reshape_0 = paddle._C_ops.reshape(add_0, full_int_array_0)
        del add_0

        # pd_op.transpose: (3x-1x12x-1x32xf32) <- (-1x-1x3x12x32xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_0, [2, 0, 3, 1, 4])
        del reshape_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [1]

        # pd_op.slice: (-1x12x-1x32xf32) <- (3x-1x12x-1x32xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            transpose_0, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [2]

        # pd_op.slice: (-1x12x-1x32xf32) <- (3x-1x12x-1x32xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            transpose_0, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [3]

        # pd_op.slice: (-1x12x-1x32xf32) <- (3x-1x12x-1x32xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            transpose_0, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del transpose_0

        # pd_op.transpose: (-1x12x32x-1xf32) <- (-1x12x-1x32xf32)
        transpose_1 = paddle._C_ops.transpose(slice_1, [0, 1, 3, 2])
        del slice_1

        # pd_op.matmul: (-1x12x-1x-1xf32) <- (-1x12x-1x32xf32, -1x12x32x-1xf32)
        matmul_1 = paddle._C_ops.matmul(slice_0, transpose_1, False, False)
        del slice_0, transpose_1

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x12x-1x-1xf32) <- (-1x12x-1x-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(matmul_1, full_0, float("0"), True)
        del matmul_1

        # pd_op.softmax: (-1x12x-1x-1xf32) <- (-1x12x-1x-1xf32)
        softmax_0 = paddle._C_ops.softmax(scale_0, -1)
        del scale_0

        # pd_op.matmul: (-1x12x-1x32xf32) <- (-1x12x-1x-1xf32, -1x12x-1x32xf32)
        matmul_2 = paddle._C_ops.matmul(softmax_0, slice_2, False, False)
        del slice_2, softmax_0

        # pd_op.transpose: (-1x-1x12x32xf32) <- (-1x12x-1x32xf32)
        transpose_2 = paddle._C_ops.transpose(matmul_2, [0, 2, 1, 3])
        del matmul_2

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_5 = [0, -1, 384]

        # pd_op.reshape: (-1x-1x384xf32) <- (-1x-1x12x32xf32, 3xi64)
        reshape_1 = paddle._C_ops.reshape(transpose_2, full_int_array_5)
        del transpose_2

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x384xf32, 384x384xf32)
        matmul_3 = paddle._C_ops.matmul(reshape_1, parameter_69, False, False)
        del parameter_69, reshape_1

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, 384xf32)
        add_1 = paddle._C_ops.add(matmul_3, parameter_68)
        del matmul_3, parameter_68

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x-1xf32, -1x-1x384xf32)
        add_2 = paddle._C_ops.add(data_6, add_1)
        del add_1, data_6

        # pd_op.layer_norm: (-1x-1x384xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x384xf32, 384xf32, 384xf32)
        layer_norm_1, layer_norm_2, layer_norm_3 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_2, parameter_67, parameter_66, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_2, parameter_66, parameter_67

        # pd_op.matmul: (-1x-1x1536xf32) <- (-1x-1x384xf32, 384x1536xf32)
        matmul_4 = paddle._C_ops.matmul(layer_norm_1, parameter_65, False, False)
        del parameter_65

        # pd_op.add: (-1x-1x1536xf32) <- (-1x-1x1536xf32, 1536xf32)
        add_3 = paddle._C_ops.add(matmul_4, parameter_64)
        del matmul_4, parameter_64

        # pd_op.gelu: (-1x-1x1536xf32) <- (-1x-1x1536xf32)
        gelu_0 = paddle._C_ops.gelu(add_3, False)
        del add_3

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x1536xf32, 1536x384xf32)
        matmul_5 = paddle._C_ops.matmul(gelu_0, parameter_63, False, False)
        del gelu_0, parameter_63

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, 384xf32)
        add_4 = paddle._C_ops.add(matmul_5, parameter_62)
        del matmul_5, parameter_62

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_5 = paddle._C_ops.add(layer_norm_1, add_4)
        del add_4, layer_norm_1

        # pd_op.layer_norm: (-1x-1x384xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x384xf32, 384xf32, 384xf32)
        layer_norm_4, layer_norm_5, layer_norm_6 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_5, parameter_61, parameter_60, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_5, parameter_60, parameter_61

        # pd_op.matmul: (-1x-1x1152xf32) <- (-1x-1x384xf32, 384x1152xf32)
        matmul_6 = paddle._C_ops.matmul(layer_norm_4, parameter_59, False, False)
        del parameter_59

        # pd_op.add: (-1x-1x1152xf32) <- (-1x-1x1152xf32, 1152xf32)
        add_6 = paddle._C_ops.add(matmul_6, parameter_58)
        del matmul_6, parameter_58

        # pd_op.reshape: (-1x-1x3x12x32xf32) <- (-1x-1x1152xf32, 5xi64)
        reshape_2 = paddle._C_ops.reshape(add_6, full_int_array_0)
        del add_6

        # pd_op.transpose: (3x-1x12x-1x32xf32) <- (-1x-1x3x12x32xf32)
        transpose_3 = paddle._C_ops.transpose(reshape_2, [2, 0, 3, 1, 4])
        del reshape_2

        # pd_op.slice: (-1x12x-1x32xf32) <- (3x-1x12x-1x32xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            transpose_3, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (-1x12x-1x32xf32) <- (3x-1x12x-1x32xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            transpose_3, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x12x-1x32xf32) <- (3x-1x12x-1x32xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            transpose_3, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del transpose_3

        # pd_op.transpose: (-1x12x32x-1xf32) <- (-1x12x-1x32xf32)
        transpose_4 = paddle._C_ops.transpose(slice_4, [0, 1, 3, 2])
        del slice_4

        # pd_op.matmul: (-1x12x-1x-1xf32) <- (-1x12x-1x32xf32, -1x12x32x-1xf32)
        matmul_7 = paddle._C_ops.matmul(slice_3, transpose_4, False, False)
        del slice_3, transpose_4

        # pd_op.scale: (-1x12x-1x-1xf32) <- (-1x12x-1x-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(matmul_7, full_0, float("0"), True)
        del matmul_7

        # pd_op.softmax: (-1x12x-1x-1xf32) <- (-1x12x-1x-1xf32)
        softmax_1 = paddle._C_ops.softmax(scale_1, -1)
        del scale_1

        # pd_op.matmul: (-1x12x-1x32xf32) <- (-1x12x-1x-1xf32, -1x12x-1x32xf32)
        matmul_8 = paddle._C_ops.matmul(softmax_1, slice_5, False, False)
        del slice_5, softmax_1

        # pd_op.transpose: (-1x-1x12x32xf32) <- (-1x12x-1x32xf32)
        transpose_5 = paddle._C_ops.transpose(matmul_8, [0, 2, 1, 3])
        del matmul_8

        # pd_op.reshape: (-1x-1x384xf32) <- (-1x-1x12x32xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_5, full_int_array_5)
        del transpose_5

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x384xf32, 384x384xf32)
        matmul_9 = paddle._C_ops.matmul(reshape_3, parameter_57, False, False)
        del parameter_57, reshape_3

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, 384xf32)
        add_7 = paddle._C_ops.add(matmul_9, parameter_56)
        del matmul_9, parameter_56

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_8 = paddle._C_ops.add(layer_norm_4, add_7)
        del add_7, layer_norm_4

        # pd_op.layer_norm: (-1x-1x384xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x384xf32, 384xf32, 384xf32)
        layer_norm_7, layer_norm_8, layer_norm_9 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_8, parameter_55, parameter_54, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_8, parameter_54, parameter_55

        # pd_op.matmul: (-1x-1x1536xf32) <- (-1x-1x384xf32, 384x1536xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_7, parameter_53, False, False)
        del parameter_53

        # pd_op.add: (-1x-1x1536xf32) <- (-1x-1x1536xf32, 1536xf32)
        add_9 = paddle._C_ops.add(matmul_10, parameter_52)
        del matmul_10, parameter_52

        # pd_op.gelu: (-1x-1x1536xf32) <- (-1x-1x1536xf32)
        gelu_1 = paddle._C_ops.gelu(add_9, False)
        del add_9

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x1536xf32, 1536x384xf32)
        matmul_11 = paddle._C_ops.matmul(gelu_1, parameter_51, False, False)
        del gelu_1, parameter_51

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, 384xf32)
        add_10 = paddle._C_ops.add(matmul_11, parameter_50)
        del matmul_11, parameter_50

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_11 = paddle._C_ops.add(layer_norm_7, add_10)
        del add_10, layer_norm_7

        # pd_op.layer_norm: (-1x-1x384xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x384xf32, 384xf32, 384xf32)
        layer_norm_10, layer_norm_11, layer_norm_12 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_11, parameter_49, parameter_48, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_11, parameter_48, parameter_49

        # pd_op.matmul: (-1x-1x1152xf32) <- (-1x-1x384xf32, 384x1152xf32)
        matmul_12 = paddle._C_ops.matmul(layer_norm_10, parameter_47, False, False)
        del parameter_47

        # pd_op.add: (-1x-1x1152xf32) <- (-1x-1x1152xf32, 1152xf32)
        add_12 = paddle._C_ops.add(matmul_12, parameter_46)
        del matmul_12, parameter_46

        # pd_op.reshape: (-1x-1x3x12x32xf32) <- (-1x-1x1152xf32, 5xi64)
        reshape_4 = paddle._C_ops.reshape(add_12, full_int_array_0)
        del add_12, full_int_array_0

        # pd_op.transpose: (3x-1x12x-1x32xf32) <- (-1x-1x3x12x32xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_4, [2, 0, 3, 1, 4])
        del reshape_4

        # pd_op.slice: (-1x12x-1x32xf32) <- (3x-1x12x-1x32xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            transpose_6, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (-1x12x-1x32xf32) <- (3x-1x12x-1x32xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            transpose_6, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x12x-1x32xf32) <- (3x-1x12x-1x32xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            transpose_6, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del transpose_6

        # pd_op.transpose: (-1x12x32x-1xf32) <- (-1x12x-1x32xf32)
        transpose_7 = paddle._C_ops.transpose(slice_7, [0, 1, 3, 2])
        del slice_7

        # pd_op.matmul: (-1x12x-1x-1xf32) <- (-1x12x-1x32xf32, -1x12x32x-1xf32)
        matmul_13 = paddle._C_ops.matmul(slice_6, transpose_7, False, False)
        del slice_6, transpose_7

        # pd_op.scale: (-1x12x-1x-1xf32) <- (-1x12x-1x-1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(matmul_13, full_0, float("0"), True)
        del matmul_13

        # pd_op.softmax: (-1x12x-1x-1xf32) <- (-1x12x-1x-1xf32)
        softmax_2 = paddle._C_ops.softmax(scale_2, -1)
        del scale_2

        # pd_op.matmul: (-1x12x-1x32xf32) <- (-1x12x-1x-1xf32, -1x12x-1x32xf32)
        matmul_14 = paddle._C_ops.matmul(softmax_2, slice_8, False, False)
        del slice_8, softmax_2

        # pd_op.transpose: (-1x-1x12x32xf32) <- (-1x12x-1x32xf32)
        transpose_8 = paddle._C_ops.transpose(matmul_14, [0, 2, 1, 3])
        del matmul_14

        # pd_op.reshape: (-1x-1x384xf32) <- (-1x-1x12x32xf32, 3xi64)
        reshape_5 = paddle._C_ops.reshape(transpose_8, full_int_array_5)
        del full_int_array_5, transpose_8

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x384xf32, 384x384xf32)
        matmul_15 = paddle._C_ops.matmul(reshape_5, parameter_45, False, False)
        del parameter_45, reshape_5

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, 384xf32)
        add_13 = paddle._C_ops.add(matmul_15, parameter_44)
        del matmul_15, parameter_44

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_14 = paddle._C_ops.add(layer_norm_10, add_13)
        del add_13, layer_norm_10

        # pd_op.layer_norm: (-1x-1x384xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x384xf32, 384xf32, 384xf32)
        layer_norm_13, layer_norm_14, layer_norm_15 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_14, parameter_43, parameter_42, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_14, parameter_42, parameter_43

        # pd_op.matmul: (-1x-1x1536xf32) <- (-1x-1x384xf32, 384x1536xf32)
        matmul_16 = paddle._C_ops.matmul(layer_norm_13, parameter_41, False, False)
        del parameter_41

        # pd_op.add: (-1x-1x1536xf32) <- (-1x-1x1536xf32, 1536xf32)
        add_15 = paddle._C_ops.add(matmul_16, parameter_40)
        del matmul_16, parameter_40

        # pd_op.gelu: (-1x-1x1536xf32) <- (-1x-1x1536xf32)
        gelu_2 = paddle._C_ops.gelu(add_15, False)
        del add_15

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x1536xf32, 1536x384xf32)
        matmul_17 = paddle._C_ops.matmul(gelu_2, parameter_39, False, False)
        del gelu_2, parameter_39

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, 384xf32)
        add_16 = paddle._C_ops.add(matmul_17, parameter_38)
        del matmul_17, parameter_38

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_17 = paddle._C_ops.add(layer_norm_13, add_16)
        del add_16, layer_norm_13

        # pd_op.layer_norm: (-1x-1x384xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x384xf32, 384xf32, 384xf32)
        layer_norm_16, layer_norm_17, layer_norm_18 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_17, parameter_37, parameter_36, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_17, parameter_36, parameter_37

        # pd_op.matmul: (-1x-1x1152xf32) <- (-1x-1x384xf32, 384x1152xf32)
        matmul_18 = paddle._C_ops.matmul(layer_norm_16, parameter_35, False, False)
        del parameter_35

        # pd_op.add: (-1x-1x1152xf32) <- (-1x-1x1152xf32, 1152xf32)
        add_18 = paddle._C_ops.add(matmul_18, parameter_34)
        del matmul_18, parameter_34

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

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
        combine_0 = [full_1, full_2, full_3, data_0, full_4]
        del data_0

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.reshape: (-1x-1x3x-1x32xf32) <- (-1x-1x1152xf32, 5xi64)
        reshape_6 = paddle._C_ops.reshape(add_18, stack_0)
        del add_18, stack_0

        # pd_op.transpose: (3x-1x-1x-1x32xf32) <- (-1x-1x3x-1x32xf32)
        transpose_9 = paddle._C_ops.transpose(reshape_6, [2, 0, 3, 1, 4])
        del reshape_6

        # pd_op.slice: (-1x-1x-1x32xf32) <- (3x-1x-1x-1x32xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            transpose_9, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (-1x-1x-1x32xf32) <- (3x-1x-1x-1x32xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            transpose_9, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x-1x-1x32xf32) <- (3x-1x-1x-1x32xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            transpose_9, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del transpose_9

        # pd_op.transpose: (-1x-1x32x-1xf32) <- (-1x-1x-1x32xf32)
        transpose_10 = paddle._C_ops.transpose(slice_10, [0, 1, 3, 2])
        del slice_10

        # pd_op.matmul: (-1x-1x-1x-1xf32) <- (-1x-1x-1x32xf32, -1x-1x32x-1xf32)
        matmul_19 = paddle._C_ops.matmul(slice_9, transpose_10, False, False)
        del slice_9, transpose_10

        # pd_op.scale: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(matmul_19, full_0, float("0"), True)
        del matmul_19

        # pd_op.softmax: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        softmax_3 = paddle._C_ops.softmax(scale_3, -1)
        del scale_3

        # pd_op.matmul: (-1x-1x-1x32xf32) <- (-1x-1x-1x-1xf32, -1x-1x-1x32xf32)
        matmul_20 = paddle._C_ops.matmul(softmax_3, slice_11, False, False)
        del slice_11, softmax_3

        # pd_op.transpose: (-1x-1x-1x32xf32) <- (-1x-1x-1x32xf32)
        transpose_11 = paddle._C_ops.transpose(matmul_20, [0, 2, 1, 3])
        del matmul_20

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_1 = [full_1, full_2, data_1]
        del data_1

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_1, 0)
        del combine_1

        # pd_op.reshape: (-1x-1x-1xf32) <- (-1x-1x-1x32xf32, 3xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_11, stack_1)
        del stack_1, transpose_11

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x-1xf32, 384x384xf32)
        matmul_21 = paddle._C_ops.matmul(reshape_7, parameter_33, False, False)
        del parameter_33, reshape_7

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, 384xf32)
        add_19 = paddle._C_ops.add(matmul_21, parameter_32)
        del matmul_21, parameter_32

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_20 = paddle._C_ops.add(layer_norm_16, add_19)
        del add_19, layer_norm_16

        # pd_op.layer_norm: (-1x-1x384xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x384xf32, 384xf32, 384xf32)
        layer_norm_19, layer_norm_20, layer_norm_21 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_20, parameter_31, parameter_30, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_20, parameter_30, parameter_31

        # pd_op.matmul: (-1x-1x1536xf32) <- (-1x-1x384xf32, 384x1536xf32)
        matmul_22 = paddle._C_ops.matmul(layer_norm_19, parameter_29, False, False)
        del parameter_29

        # pd_op.add: (-1x-1x1536xf32) <- (-1x-1x1536xf32, 1536xf32)
        add_21 = paddle._C_ops.add(matmul_22, parameter_28)
        del matmul_22, parameter_28

        # pd_op.gelu: (-1x-1x1536xf32) <- (-1x-1x1536xf32)
        gelu_3 = paddle._C_ops.gelu(add_21, False)
        del add_21

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x1536xf32, 1536x384xf32)
        matmul_23 = paddle._C_ops.matmul(gelu_3, parameter_27, False, False)
        del gelu_3, parameter_27

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, 384xf32)
        add_22 = paddle._C_ops.add(matmul_23, parameter_26)
        del matmul_23, parameter_26

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_23 = paddle._C_ops.add(layer_norm_19, add_22)
        del add_22, layer_norm_19

        # pd_op.layer_norm: (-1x-1x384xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x384xf32, 384xf32, 384xf32)
        layer_norm_22, layer_norm_23, layer_norm_24 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_23, parameter_25, parameter_24, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_23, parameter_24, parameter_25

        # pd_op.matmul: (-1x-1x1152xf32) <- (-1x-1x384xf32, 384x1152xf32)
        matmul_24 = paddle._C_ops.matmul(layer_norm_22, parameter_23, False, False)
        del parameter_23

        # pd_op.add: (-1x-1x1152xf32) <- (-1x-1x1152xf32, 1152xf32)
        add_24 = paddle._C_ops.add(matmul_24, parameter_22)
        del matmul_24, parameter_22

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_2 = [full_1, full_2, full_3, data_2, full_4]
        del data_2

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_2 = paddle._C_ops.stack(combine_2, 0)
        del combine_2

        # pd_op.reshape: (-1x-1x3x-1x32xf32) <- (-1x-1x1152xf32, 5xi64)
        reshape_8 = paddle._C_ops.reshape(add_24, stack_2)
        del add_24, stack_2

        # pd_op.transpose: (3x-1x-1x-1x32xf32) <- (-1x-1x3x-1x32xf32)
        transpose_12 = paddle._C_ops.transpose(reshape_8, [2, 0, 3, 1, 4])
        del reshape_8

        # pd_op.slice: (-1x-1x-1x32xf32) <- (3x-1x-1x-1x32xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            transpose_12, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (-1x-1x-1x32xf32) <- (3x-1x-1x-1x32xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            transpose_12, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x-1x-1x32xf32) <- (3x-1x-1x-1x32xf32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            transpose_12, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del transpose_12

        # pd_op.transpose: (-1x-1x32x-1xf32) <- (-1x-1x-1x32xf32)
        transpose_13 = paddle._C_ops.transpose(slice_13, [0, 1, 3, 2])
        del slice_13

        # pd_op.matmul: (-1x-1x-1x-1xf32) <- (-1x-1x-1x32xf32, -1x-1x32x-1xf32)
        matmul_25 = paddle._C_ops.matmul(slice_12, transpose_13, False, False)
        del slice_12, transpose_13

        # pd_op.scale: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(matmul_25, full_0, float("0"), True)
        del matmul_25

        # pd_op.softmax: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        softmax_4 = paddle._C_ops.softmax(scale_4, -1)
        del scale_4

        # pd_op.matmul: (-1x-1x-1x32xf32) <- (-1x-1x-1x-1xf32, -1x-1x-1x32xf32)
        matmul_26 = paddle._C_ops.matmul(softmax_4, slice_14, False, False)
        del slice_14, softmax_4

        # pd_op.transpose: (-1x-1x-1x32xf32) <- (-1x-1x-1x32xf32)
        transpose_14 = paddle._C_ops.transpose(matmul_26, [0, 2, 1, 3])
        del matmul_26

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_3 = [full_1, full_2, data_3]
        del data_3

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_3 = paddle._C_ops.stack(combine_3, 0)
        del combine_3

        # pd_op.reshape: (-1x-1x-1xf32) <- (-1x-1x-1x32xf32, 3xi64)
        reshape_9 = paddle._C_ops.reshape(transpose_14, stack_3)
        del stack_3, transpose_14

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x-1xf32, 384x384xf32)
        matmul_27 = paddle._C_ops.matmul(reshape_9, parameter_21, False, False)
        del parameter_21, reshape_9

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, 384xf32)
        add_25 = paddle._C_ops.add(matmul_27, parameter_20)
        del matmul_27, parameter_20

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_26 = paddle._C_ops.add(layer_norm_22, add_25)
        del add_25, layer_norm_22

        # pd_op.layer_norm: (-1x-1x384xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x384xf32, 384xf32, 384xf32)
        layer_norm_25, layer_norm_26, layer_norm_27 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_26, parameter_19, parameter_18, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_26, parameter_18, parameter_19

        # pd_op.matmul: (-1x-1x1536xf32) <- (-1x-1x384xf32, 384x1536xf32)
        matmul_28 = paddle._C_ops.matmul(layer_norm_25, parameter_17, False, False)
        del parameter_17

        # pd_op.add: (-1x-1x1536xf32) <- (-1x-1x1536xf32, 1536xf32)
        add_27 = paddle._C_ops.add(matmul_28, parameter_16)
        del matmul_28, parameter_16

        # pd_op.gelu: (-1x-1x1536xf32) <- (-1x-1x1536xf32)
        gelu_4 = paddle._C_ops.gelu(add_27, False)
        del add_27

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x1536xf32, 1536x384xf32)
        matmul_29 = paddle._C_ops.matmul(gelu_4, parameter_15, False, False)
        del gelu_4, parameter_15

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, 384xf32)
        add_28 = paddle._C_ops.add(matmul_29, parameter_14)
        del matmul_29, parameter_14

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_29 = paddle._C_ops.add(layer_norm_25, add_28)
        del add_28, layer_norm_25

        # pd_op.layer_norm: (-1x-1x384xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x384xf32, 384xf32, 384xf32)
        layer_norm_28, layer_norm_29, layer_norm_30 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_29, parameter_13, parameter_12, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_29, parameter_12, parameter_13

        # pd_op.matmul: (-1x-1x1152xf32) <- (-1x-1x384xf32, 384x1152xf32)
        matmul_30 = paddle._C_ops.matmul(layer_norm_28, parameter_11, False, False)
        del parameter_11

        # pd_op.add: (-1x-1x1152xf32) <- (-1x-1x1152xf32, 1152xf32)
        add_30 = paddle._C_ops.add(matmul_30, parameter_10)
        del matmul_30, parameter_10

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_4 = [full_1, full_2, full_3, data_4, full_4]
        del data_4, full_3, full_4

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_4 = paddle._C_ops.stack(combine_4, 0)
        del combine_4

        # pd_op.reshape: (-1x-1x3x-1x32xf32) <- (-1x-1x1152xf32, 5xi64)
        reshape_10 = paddle._C_ops.reshape(add_30, stack_4)
        del add_30, stack_4

        # pd_op.transpose: (3x-1x-1x-1x32xf32) <- (-1x-1x3x-1x32xf32)
        transpose_15 = paddle._C_ops.transpose(reshape_10, [2, 0, 3, 1, 4])
        del reshape_10

        # pd_op.slice: (-1x-1x-1x32xf32) <- (3x-1x-1x-1x32xf32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            transpose_15, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del full_int_array_1

        # pd_op.slice: (-1x-1x-1x32xf32) <- (3x-1x-1x-1x32xf32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            transpose_15, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del full_int_array_2

        # pd_op.slice: (-1x-1x-1x32xf32) <- (3x-1x-1x-1x32xf32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            transpose_15, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del full_int_array_3, full_int_array_4, transpose_15

        # pd_op.transpose: (-1x-1x32x-1xf32) <- (-1x-1x-1x32xf32)
        transpose_16 = paddle._C_ops.transpose(slice_16, [0, 1, 3, 2])
        del slice_16

        # pd_op.matmul: (-1x-1x-1x-1xf32) <- (-1x-1x-1x32xf32, -1x-1x32x-1xf32)
        matmul_31 = paddle._C_ops.matmul(slice_15, transpose_16, False, False)
        del slice_15, transpose_16

        # pd_op.scale: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(matmul_31, full_0, float("0"), True)
        del full_0, matmul_31

        # pd_op.softmax: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        softmax_5 = paddle._C_ops.softmax(scale_5, -1)
        del scale_5

        # pd_op.matmul: (-1x-1x-1x32xf32) <- (-1x-1x-1x-1xf32, -1x-1x-1x32xf32)
        matmul_32 = paddle._C_ops.matmul(softmax_5, slice_17, False, False)
        del slice_17, softmax_5

        # pd_op.transpose: (-1x-1x-1x32xf32) <- (-1x-1x-1x32xf32)
        transpose_17 = paddle._C_ops.transpose(matmul_32, [0, 2, 1, 3])
        del matmul_32

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_5 = [full_1, full_2, data_5]
        del data_5, full_1, full_2

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_5 = paddle._C_ops.stack(combine_5, 0)
        del combine_5

        # pd_op.reshape: (-1x-1x-1xf32) <- (-1x-1x-1x32xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_17, stack_5)
        del stack_5, transpose_17

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x-1xf32, 384x384xf32)
        matmul_33 = paddle._C_ops.matmul(reshape_11, parameter_9, False, False)
        del parameter_9, reshape_11

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, 384xf32)
        add_31 = paddle._C_ops.add(matmul_33, parameter_8)
        del matmul_33, parameter_8

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_32 = paddle._C_ops.add(layer_norm_28, add_31)
        del add_31, layer_norm_28

        # pd_op.layer_norm: (-1x-1x384xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x384xf32, 384xf32, 384xf32)
        layer_norm_31, layer_norm_32, layer_norm_33 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_32, parameter_7, parameter_6, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_32, parameter_6, parameter_7

        # pd_op.matmul: (-1x-1x1536xf32) <- (-1x-1x384xf32, 384x1536xf32)
        matmul_34 = paddle._C_ops.matmul(layer_norm_31, parameter_5, False, False)
        del parameter_5

        # pd_op.add: (-1x-1x1536xf32) <- (-1x-1x1536xf32, 1536xf32)
        add_33 = paddle._C_ops.add(matmul_34, parameter_4)
        del matmul_34, parameter_4

        # pd_op.gelu: (-1x-1x1536xf32) <- (-1x-1x1536xf32)
        gelu_5 = paddle._C_ops.gelu(add_33, False)
        del add_33

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x1536xf32, 1536x384xf32)
        matmul_35 = paddle._C_ops.matmul(gelu_5, parameter_3, False, False)
        del gelu_5, parameter_3

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, 384xf32)
        add_34 = paddle._C_ops.add(matmul_35, parameter_2)
        del matmul_35, parameter_2

        # pd_op.add: (-1x-1x384xf32) <- (-1x-1x384xf32, -1x-1x384xf32)
        add_35 = paddle._C_ops.add(layer_norm_31, add_34)
        del add_34, layer_norm_31

        # pd_op.layer_norm: (-1x-1x384xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x384xf32, 384xf32, 384xf32)
        layer_norm_0, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_35, parameter_1, parameter_0, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_35, parameter_0, parameter_1

        return layer_norm_0
