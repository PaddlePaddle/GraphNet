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
        parameter_72,
        parameter_73,
        parameter_74,
        parameter_75,
        parameter_76,
        parameter_77,
        parameter_78,
        parameter_79,
        parameter_80,
        parameter_81,
        parameter_82,
        data_0,
        data_1,
        data_2,
        data_3,
    ):
        # pd_op.flatten: (4x1024x40xf32) <- (4x1024x1x40xf32)
        flatten_0 = paddle._C_ops.flatten(data_0, 2, 3)
        del data_0

        # pd_op.transpose: (4x40x1024xf32) <- (4x1024x40xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_0, [0, 2, 1])
        del flatten_0

        # pd_op.matmul: (4x40x384xf32) <- (4x40x1024xf32, 1024x384xf32)
        matmul_1 = paddle._C_ops.matmul(transpose_0, parameter_82, False, False)
        del parameter_82

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_0 = []

        # pd_op.max: (xi64) <- (4xi64, 0xi64)
        max_0 = paddle._C_ops.max(data_2, full_int_array_0, False)
        del data_2, full_int_array_0

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(max_0, full_0, float("2"), True)
        del full_0, max_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [scale_0]
        del scale_0

        # pd_op.stack: (1xi64) <- ([xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.slice: (4x-1xi64) <- (4x25xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(data_1, [1], full_int_array_1, stack_0, [-1], [])
        del data_1, full_int_array_1, stack_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [-1]

        # pd_op.slice: (4x-1xi64) <- (4x-1xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            slice_0, [1], full_int_array_2, full_int_array_3, [1], []
        )
        del full_int_array_2, full_int_array_3, slice_0

        # pd_op.embedding: (4x-1x384xf32) <- (4x-1xi64, 6629x384xf32)
        embedding_0 = paddle._C_ops.embedding(slice_1, parameter_81, 0, False)
        del parameter_81

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("19.5959"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (4x-1x384xf32) <- (4x-1x384xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(embedding_0, full_1, float("0"), True)
        del embedding_0, full_1

        # pd_op.transpose: (-1x4x384xf32) <- (4x-1x384xf32)
        transpose_1 = paddle._C_ops.transpose(scale_1, [1, 0, 2])
        del scale_1

        # pd_op.shape64: (3xi64) <- (-1x4x384xf32)
        shape64_0 = paddle._C_ops.shape64(transpose_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_4, full_int_array_5, [1], [0]
        )
        del full_int_array_4, full_int_array_5, shape64_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [slice_2]
        del slice_2

        # pd_op.stack: (1xi64) <- ([xi64])
        stack_1 = paddle._C_ops.stack(combine_1, 0)
        del combine_1

        # pd_op.slice: (-1x1x384xf32) <- (5000x1x384xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(data_3, [0], full_int_array_6, stack_1, [-1], [])
        del data_3, full_int_array_6, stack_1

        # pd_op.add: (-1x4x384xf32) <- (-1x4x384xf32, -1x1x384xf32)
        add_0 = paddle._C_ops.add(transpose_1, slice_3)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (-1x4x384xf32, -1x4x384xui8) <- (-1x4x384xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_0, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_0

        # pd_op.transpose: (4x-1x384xf32) <- (-1x4x384xf32)
        transpose_2 = paddle._C_ops.transpose(dropout_0, [1, 0, 2])
        del dropout_0

        # pd_op.shape64: (3xi64) <- (4x-1x384xf32)
        shape64_1 = paddle._C_ops.shape64(transpose_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [2]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_7, full_int_array_8, [1], [0]
        )
        del full_int_array_7, full_int_array_8, shape64_1

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_2 = [slice_4, slice_4]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_2 = paddle._C_ops.stack(combine_2, 0)
        del combine_2

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_with_tensor: (-1x-1xf32) <- (1xf32, 2xi64)
        full_with_tensor_0 = paddle._C_ops.full_with_tensor(
            full_3, stack_2, paddle.float32
        )
        del full_3, stack_2

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_3 = [slice_4, slice_4]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_3 = paddle._C_ops.stack(combine_3, 0)
        del combine_3

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("-inf"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_with_tensor: (-1x-1xf32) <- (1xf32, 2xi64)
        full_with_tensor_1 = paddle._C_ops.full_with_tensor(
            full_4, stack_3, paddle.float32
        )
        del full_4, stack_3

        # pd_op.triu: (-1x-1xf32) <- (-1x-1xf32)
        triu_0 = paddle._C_ops.triu(full_with_tensor_1, 1)
        del full_with_tensor_1

        # pd_op.add: (-1x-1xf32) <- (-1x-1xf32, -1x-1xf32)
        add_1 = paddle._C_ops.add(full_with_tensor_0, triu_0)
        del full_with_tensor_0, triu_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_9 = [0, 1]

        # pd_op.unsqueeze: (1x1x-1x-1xf32) <- (-1x-1xf32, 2xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(add_1, full_int_array_9)
        del add_1, full_int_array_9

        # pd_op.matmul: (4x-1x1152xf32) <- (4x-1x384xf32, 384x1152xf32)
        matmul_2 = paddle._C_ops.matmul(transpose_2, parameter_80, False, False)
        del parameter_80

        # pd_op.add: (4x-1x1152xf32) <- (4x-1x1152xf32, 1152xf32)
        add_2 = paddle._C_ops.add(matmul_2, parameter_79)
        del parameter_79

        # pd_op.full: (xi64) <- ()
        full_5 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_6 = paddle._C_ops.full(
            [], float("3"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_7 = paddle._C_ops.full(
            [], float("12"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_8 = paddle._C_ops.full(
            [], float("32"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_4 = [full_5, slice_4, full_6, full_7, full_8]
        del full_5, full_6, full_7, full_8

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_4 = paddle._C_ops.stack(combine_4, 0)
        del combine_4

        # pd_op.reshape: (4x-1x3x12x32xf32) <- (4x-1x1152xf32, 5xi64)
        reshape_0 = paddle._C_ops.reshape(add_2, stack_4)
        del stack_4

        # pd_op.transpose: (3x4x12x-1x32xf32) <- (4x-1x3x12x32xf32)
        transpose_3 = paddle._C_ops.transpose(reshape_0, [2, 0, 3, 1, 4])
        del reshape_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_10 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_11 = [1]

        # pd_op.slice: (4x12x-1x32xf32) <- (3x4x12x-1x32xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            transpose_3, [0], full_int_array_10, full_int_array_11, [1], [0]
        )
        del full_int_array_10, full_int_array_11

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_12 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_13 = [2]

        # pd_op.slice: (4x12x-1x32xf32) <- (3x4x12x-1x32xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            transpose_3, [0], full_int_array_12, full_int_array_13, [1], [0]
        )
        del full_int_array_12, full_int_array_13

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_14 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_15 = [3]

        # pd_op.slice: (4x12x-1x32xf32) <- (3x4x12x-1x32xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            transpose_3, [0], full_int_array_14, full_int_array_15, [1], [0]
        )
        del full_int_array_14, full_int_array_15

        # pd_op.transpose: (4x12x32x-1xf32) <- (4x12x-1x32xf32)
        transpose_4 = paddle._C_ops.transpose(slice_6, [0, 1, 3, 2])
        del slice_6

        # pd_op.matmul: (4x12x-1x-1xf32) <- (4x12x-1x32xf32, 4x12x32x-1xf32)
        matmul_3 = paddle._C_ops.matmul(slice_5, transpose_4, False, False)

        # pd_op.full: (1xf32) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (4x12x-1x-1xf32) <- (4x12x-1x-1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(matmul_3, full_9, float("0"), True)
        del full_9, matmul_3

        # pd_op.add: (4x12x-1x-1xf32) <- (4x12x-1x-1xf32, 1x1x-1x-1xf32)
        add_3 = paddle._C_ops.add(scale_2, unsqueeze_0)

        # pd_op.softmax: (4x12x-1x-1xf32) <- (4x12x-1x-1xf32)
        softmax_0 = paddle._C_ops.softmax(add_3, -1)
        del add_3

        # pd_op.matmul: (4x12x-1x32xf32) <- (4x12x-1x-1xf32, 4x12x-1x32xf32)
        matmul_4 = paddle._C_ops.matmul(softmax_0, slice_7, False, False)

        # pd_op.transpose: (4x-1x12x32xf32) <- (4x12x-1x32xf32)
        transpose_5 = paddle._C_ops.transpose(matmul_4, [0, 2, 1, 3])
        del matmul_4

        # pd_op.full: (xi64) <- ()
        full_10 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_11 = paddle._C_ops.full(
            [], float("384"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_5 = [full_10, slice_4, full_11]
        del full_10, full_11, slice_4

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_5 = paddle._C_ops.stack(combine_5, 0)
        del combine_5

        # pd_op.reshape: (4x-1x384xf32) <- (4x-1x12x32xf32, 3xi64)
        reshape_1 = paddle._C_ops.reshape(transpose_5, stack_5)
        del stack_5

        # pd_op.matmul: (4x-1x384xf32) <- (4x-1x384xf32, 384x384xf32)
        matmul_5 = paddle._C_ops.matmul(reshape_1, parameter_78, False, False)
        del parameter_78

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 384xf32)
        add_4 = paddle._C_ops.add(matmul_5, parameter_77)
        del parameter_77

        # pd_op.full: (1xf32) <- ()
        full_12 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (4x-1x384xf32, 4x-1x384xui8) <- (4x-1x384xf32, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_4, None, full_12, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_4

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 4x-1x384xf32)
        add_5 = paddle._C_ops.add(transpose_2, dropout_2)

        # pd_op.layer_norm: (4x-1x384xf32, 4x-1xf32, 4x-1xf32) <- (4x-1x384xf32, 384xf32, 384xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_5, parameter_76, parameter_75, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_75, parameter_76

        # pd_op.shape64: (3xi64) <- (4x-1x384xf32)
        shape64_2 = paddle._C_ops.shape64(layer_norm_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_16 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_17 = [2]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_16, full_int_array_17, [1], [0]
        )
        del full_int_array_16, full_int_array_17, shape64_2

        # pd_op.matmul: (4x-1x384xf32) <- (4x-1x384xf32, 384x384xf32)
        matmul_6 = paddle._C_ops.matmul(layer_norm_0, parameter_74, False, False)
        del parameter_74

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 384xf32)
        add_6 = paddle._C_ops.add(matmul_6, parameter_73)
        del parameter_73

        # pd_op.full: (xi64) <- ()
        full_13 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_14 = paddle._C_ops.full(
            [], float("12"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_15 = paddle._C_ops.full(
            [], float("32"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_6 = [full_13, slice_8, full_14, full_15]
        del full_13, full_14, full_15

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_6 = paddle._C_ops.stack(combine_6, 0)
        del combine_6

        # pd_op.reshape: (4x-1x12x32xf32) <- (4x-1x384xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(add_6, stack_6)
        del stack_6

        # pd_op.transpose: (4x12x-1x32xf32) <- (4x-1x12x32xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_2, [0, 2, 1, 3])
        del reshape_2

        # pd_op.matmul: (4x40x768xf32) <- (4x40x384xf32, 384x768xf32)
        matmul_7 = paddle._C_ops.matmul(matmul_1, parameter_72, False, False)
        del parameter_72

        # pd_op.add: (4x40x768xf32) <- (4x40x768xf32, 768xf32)
        add_7 = paddle._C_ops.add(matmul_7, parameter_71)
        del parameter_71

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_18 = [0, 40, 2, 12, 32]

        # pd_op.reshape: (4x40x2x12x32xf32) <- (4x40x768xf32, 5xi64)
        reshape_3 = paddle._C_ops.reshape(add_7, full_int_array_18)
        del full_int_array_18

        # pd_op.transpose: (2x4x12x40x32xf32) <- (4x40x2x12x32xf32)
        transpose_7 = paddle._C_ops.transpose(reshape_3, [2, 0, 3, 1, 4])
        del reshape_3

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_19 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_20 = [1]

        # pd_op.slice: (4x12x40x32xf32) <- (2x4x12x40x32xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            transpose_7, [0], full_int_array_19, full_int_array_20, [1], [0]
        )
        del full_int_array_19, full_int_array_20

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_21 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_22 = [2]

        # pd_op.slice: (4x12x40x32xf32) <- (2x4x12x40x32xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            transpose_7, [0], full_int_array_21, full_int_array_22, [1], [0]
        )
        del full_int_array_21, full_int_array_22

        # pd_op.transpose: (4x12x32x40xf32) <- (4x12x40x32xf32)
        transpose_8 = paddle._C_ops.transpose(slice_9, [0, 1, 3, 2])
        del slice_9

        # pd_op.matmul: (4x12x-1x40xf32) <- (4x12x-1x32xf32, 4x12x32x40xf32)
        matmul_8 = paddle._C_ops.matmul(transpose_6, transpose_8, False, False)

        # pd_op.full: (1xf32) <- ()
        full_16 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (4x12x-1x40xf32) <- (4x12x-1x40xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(matmul_8, full_16, float("0"), True)
        del full_16, matmul_8

        # pd_op.softmax: (4x12x-1x40xf32) <- (4x12x-1x40xf32)
        softmax_1 = paddle._C_ops.softmax(scale_3, -1)
        del scale_3

        # pd_op.matmul: (4x12x-1x32xf32) <- (4x12x-1x40xf32, 4x12x40x32xf32)
        matmul_9 = paddle._C_ops.matmul(softmax_1, slice_10, False, False)

        # pd_op.transpose: (4x-1x12x32xf32) <- (4x12x-1x32xf32)
        transpose_9 = paddle._C_ops.transpose(matmul_9, [0, 2, 1, 3])
        del matmul_9

        # pd_op.full: (xi64) <- ()
        full_17 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_18 = paddle._C_ops.full(
            [], float("384"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_7 = [full_17, slice_8, full_18]
        del full_17, full_18, slice_8

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_7 = paddle._C_ops.stack(combine_7, 0)
        del combine_7

        # pd_op.reshape: (4x-1x384xf32) <- (4x-1x12x32xf32, 3xi64)
        reshape_4 = paddle._C_ops.reshape(transpose_9, stack_7)
        del stack_7

        # pd_op.matmul: (4x-1x384xf32) <- (4x-1x384xf32, 384x384xf32)
        matmul_10 = paddle._C_ops.matmul(reshape_4, parameter_70, False, False)
        del parameter_70

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 384xf32)
        add_8 = paddle._C_ops.add(matmul_10, parameter_69)
        del parameter_69

        # pd_op.full: (1xf32) <- ()
        full_19 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (4x-1x384xf32, 4x-1x384xui8) <- (4x-1x384xf32, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_8, None, full_19, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_8

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 4x-1x384xf32)
        add_9 = paddle._C_ops.add(layer_norm_0, dropout_4)

        # pd_op.layer_norm: (4x-1x384xf32, 4x-1xf32, 4x-1xf32) <- (4x-1x384xf32, 384xf32, 384xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_9, parameter_68, parameter_67, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_67, parameter_68

        # pd_op.matmul: (4x-1x1536xf32) <- (4x-1x384xf32, 384x1536xf32)
        matmul_11 = paddle._C_ops.matmul(layer_norm_3, parameter_66, False, False)
        del parameter_66

        # pd_op.add: (4x-1x1536xf32) <- (4x-1x1536xf32, 1536xf32)
        add_10 = paddle._C_ops.add(matmul_11, parameter_65)
        del parameter_65

        # pd_op.relu: (4x-1x1536xf32) <- (4x-1x1536xf32)
        relu_0 = paddle._C_ops.relu(add_10)
        del add_10

        # pd_op.full: (1xf32) <- ()
        full_20 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (4x-1x1536xf32, 4x-1x1536xui8) <- (4x-1x1536xf32, None, 1xf32)
        dropout_6, dropout_7 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                relu_0, None, full_20, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (4x-1x384xf32) <- (4x-1x1536xf32, 1536x384xf32)
        matmul_12 = paddle._C_ops.matmul(dropout_6, parameter_64, False, False)
        del parameter_64

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 384xf32)
        add_11 = paddle._C_ops.add(matmul_12, parameter_63)
        del parameter_63

        # pd_op.full: (1xf32) <- ()
        full_21 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (4x-1x384xf32, 4x-1x384xui8) <- (4x-1x384xf32, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_11, None, full_21, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_11

        # pd_op.full: (1xf32) <- ()
        full_22 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (4x-1x384xf32, 4x-1x384xui8) <- (4x-1x384xf32, None, 1xf32)
        dropout_10, dropout_11 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                dropout_8, None, full_22, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del dropout_8

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 4x-1x384xf32)
        add_12 = paddle._C_ops.add(layer_norm_3, dropout_10)

        # pd_op.layer_norm: (4x-1x384xf32, 4x-1xf32, 4x-1xf32) <- (4x-1x384xf32, 384xf32, 384xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_12, parameter_62, parameter_61, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_61, parameter_62

        # pd_op.shape64: (3xi64) <- (4x-1x384xf32)
        shape64_3 = paddle._C_ops.shape64(layer_norm_6)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_23 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_24 = [2]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_23, full_int_array_24, [1], [0]
        )
        del full_int_array_23, full_int_array_24, shape64_3

        # pd_op.matmul: (4x-1x1152xf32) <- (4x-1x384xf32, 384x1152xf32)
        matmul_13 = paddle._C_ops.matmul(layer_norm_6, parameter_60, False, False)
        del parameter_60

        # pd_op.add: (4x-1x1152xf32) <- (4x-1x1152xf32, 1152xf32)
        add_13 = paddle._C_ops.add(matmul_13, parameter_59)
        del parameter_59

        # pd_op.full: (xi64) <- ()
        full_23 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_24 = paddle._C_ops.full(
            [], float("3"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_25 = paddle._C_ops.full(
            [], float("12"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_26 = paddle._C_ops.full(
            [], float("32"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_8 = [full_23, slice_11, full_24, full_25, full_26]
        del full_23, full_24, full_25, full_26

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_8 = paddle._C_ops.stack(combine_8, 0)
        del combine_8

        # pd_op.reshape: (4x-1x3x12x32xf32) <- (4x-1x1152xf32, 5xi64)
        reshape_5 = paddle._C_ops.reshape(add_13, stack_8)
        del stack_8

        # pd_op.transpose: (3x4x12x-1x32xf32) <- (4x-1x3x12x32xf32)
        transpose_10 = paddle._C_ops.transpose(reshape_5, [2, 0, 3, 1, 4])
        del reshape_5

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_25 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_26 = [1]

        # pd_op.slice: (4x12x-1x32xf32) <- (3x4x12x-1x32xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            transpose_10, [0], full_int_array_25, full_int_array_26, [1], [0]
        )
        del full_int_array_25, full_int_array_26

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_27 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_28 = [2]

        # pd_op.slice: (4x12x-1x32xf32) <- (3x4x12x-1x32xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            transpose_10, [0], full_int_array_27, full_int_array_28, [1], [0]
        )
        del full_int_array_27, full_int_array_28

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_29 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_30 = [3]

        # pd_op.slice: (4x12x-1x32xf32) <- (3x4x12x-1x32xf32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            transpose_10, [0], full_int_array_29, full_int_array_30, [1], [0]
        )
        del full_int_array_29, full_int_array_30

        # pd_op.transpose: (4x12x32x-1xf32) <- (4x12x-1x32xf32)
        transpose_11 = paddle._C_ops.transpose(slice_13, [0, 1, 3, 2])
        del slice_13

        # pd_op.matmul: (4x12x-1x-1xf32) <- (4x12x-1x32xf32, 4x12x32x-1xf32)
        matmul_14 = paddle._C_ops.matmul(slice_12, transpose_11, False, False)

        # pd_op.full: (1xf32) <- ()
        full_27 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (4x12x-1x-1xf32) <- (4x12x-1x-1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(matmul_14, full_27, float("0"), True)
        del full_27, matmul_14

        # pd_op.add: (4x12x-1x-1xf32) <- (4x12x-1x-1xf32, 1x1x-1x-1xf32)
        add_14 = paddle._C_ops.add(scale_4, unsqueeze_0)

        # pd_op.softmax: (4x12x-1x-1xf32) <- (4x12x-1x-1xf32)
        softmax_2 = paddle._C_ops.softmax(add_14, -1)
        del add_14

        # pd_op.matmul: (4x12x-1x32xf32) <- (4x12x-1x-1xf32, 4x12x-1x32xf32)
        matmul_15 = paddle._C_ops.matmul(softmax_2, slice_14, False, False)

        # pd_op.transpose: (4x-1x12x32xf32) <- (4x12x-1x32xf32)
        transpose_12 = paddle._C_ops.transpose(matmul_15, [0, 2, 1, 3])
        del matmul_15

        # pd_op.full: (xi64) <- ()
        full_28 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_29 = paddle._C_ops.full(
            [], float("384"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_9 = [full_28, slice_11, full_29]
        del full_28, full_29, slice_11

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_9 = paddle._C_ops.stack(combine_9, 0)
        del combine_9

        # pd_op.reshape: (4x-1x384xf32) <- (4x-1x12x32xf32, 3xi64)
        reshape_6 = paddle._C_ops.reshape(transpose_12, stack_9)
        del stack_9

        # pd_op.matmul: (4x-1x384xf32) <- (4x-1x384xf32, 384x384xf32)
        matmul_16 = paddle._C_ops.matmul(reshape_6, parameter_58, False, False)
        del parameter_58

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 384xf32)
        add_15 = paddle._C_ops.add(matmul_16, parameter_57)
        del parameter_57

        # pd_op.full: (1xf32) <- ()
        full_30 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (4x-1x384xf32, 4x-1x384xui8) <- (4x-1x384xf32, None, 1xf32)
        dropout_12, dropout_13 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_15, None, full_30, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_15

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 4x-1x384xf32)
        add_16 = paddle._C_ops.add(layer_norm_6, dropout_12)

        # pd_op.layer_norm: (4x-1x384xf32, 4x-1xf32, 4x-1xf32) <- (4x-1x384xf32, 384xf32, 384xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_16, parameter_56, parameter_55, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_55, parameter_56

        # pd_op.shape64: (3xi64) <- (4x-1x384xf32)
        shape64_4 = paddle._C_ops.shape64(layer_norm_9)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_31 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_32 = [2]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            shape64_4, [0], full_int_array_31, full_int_array_32, [1], [0]
        )
        del full_int_array_31, full_int_array_32, shape64_4

        # pd_op.matmul: (4x-1x384xf32) <- (4x-1x384xf32, 384x384xf32)
        matmul_17 = paddle._C_ops.matmul(layer_norm_9, parameter_54, False, False)
        del parameter_54

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 384xf32)
        add_17 = paddle._C_ops.add(matmul_17, parameter_53)
        del parameter_53

        # pd_op.full: (xi64) <- ()
        full_31 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_32 = paddle._C_ops.full(
            [], float("12"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_33 = paddle._C_ops.full(
            [], float("32"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_10 = [full_31, slice_15, full_32, full_33]
        del full_31, full_32, full_33

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_10 = paddle._C_ops.stack(combine_10, 0)
        del combine_10

        # pd_op.reshape: (4x-1x12x32xf32) <- (4x-1x384xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(add_17, stack_10)
        del stack_10

        # pd_op.transpose: (4x12x-1x32xf32) <- (4x-1x12x32xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_7, [0, 2, 1, 3])
        del reshape_7

        # pd_op.matmul: (4x40x768xf32) <- (4x40x384xf32, 384x768xf32)
        matmul_18 = paddle._C_ops.matmul(matmul_1, parameter_52, False, False)
        del parameter_52

        # pd_op.add: (4x40x768xf32) <- (4x40x768xf32, 768xf32)
        add_18 = paddle._C_ops.add(matmul_18, parameter_51)
        del parameter_51

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_33 = [0, 40, 2, 12, 32]

        # pd_op.reshape: (4x40x2x12x32xf32) <- (4x40x768xf32, 5xi64)
        reshape_8 = paddle._C_ops.reshape(add_18, full_int_array_33)
        del full_int_array_33

        # pd_op.transpose: (2x4x12x40x32xf32) <- (4x40x2x12x32xf32)
        transpose_14 = paddle._C_ops.transpose(reshape_8, [2, 0, 3, 1, 4])
        del reshape_8

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_34 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_35 = [1]

        # pd_op.slice: (4x12x40x32xf32) <- (2x4x12x40x32xf32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            transpose_14, [0], full_int_array_34, full_int_array_35, [1], [0]
        )
        del full_int_array_34, full_int_array_35

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_36 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_37 = [2]

        # pd_op.slice: (4x12x40x32xf32) <- (2x4x12x40x32xf32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            transpose_14, [0], full_int_array_36, full_int_array_37, [1], [0]
        )
        del full_int_array_36, full_int_array_37

        # pd_op.transpose: (4x12x32x40xf32) <- (4x12x40x32xf32)
        transpose_15 = paddle._C_ops.transpose(slice_16, [0, 1, 3, 2])
        del slice_16

        # pd_op.matmul: (4x12x-1x40xf32) <- (4x12x-1x32xf32, 4x12x32x40xf32)
        matmul_19 = paddle._C_ops.matmul(transpose_13, transpose_15, False, False)

        # pd_op.full: (1xf32) <- ()
        full_34 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (4x12x-1x40xf32) <- (4x12x-1x40xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(matmul_19, full_34, float("0"), True)
        del full_34, matmul_19

        # pd_op.softmax: (4x12x-1x40xf32) <- (4x12x-1x40xf32)
        softmax_3 = paddle._C_ops.softmax(scale_5, -1)
        del scale_5

        # pd_op.matmul: (4x12x-1x32xf32) <- (4x12x-1x40xf32, 4x12x40x32xf32)
        matmul_20 = paddle._C_ops.matmul(softmax_3, slice_17, False, False)

        # pd_op.transpose: (4x-1x12x32xf32) <- (4x12x-1x32xf32)
        transpose_16 = paddle._C_ops.transpose(matmul_20, [0, 2, 1, 3])
        del matmul_20

        # pd_op.full: (xi64) <- ()
        full_35 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_36 = paddle._C_ops.full(
            [], float("384"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_11 = [full_35, slice_15, full_36]
        del full_35, full_36, slice_15

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_11 = paddle._C_ops.stack(combine_11, 0)
        del combine_11

        # pd_op.reshape: (4x-1x384xf32) <- (4x-1x12x32xf32, 3xi64)
        reshape_9 = paddle._C_ops.reshape(transpose_16, stack_11)
        del stack_11

        # pd_op.matmul: (4x-1x384xf32) <- (4x-1x384xf32, 384x384xf32)
        matmul_21 = paddle._C_ops.matmul(reshape_9, parameter_50, False, False)
        del parameter_50

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 384xf32)
        add_19 = paddle._C_ops.add(matmul_21, parameter_49)
        del parameter_49

        # pd_op.full: (1xf32) <- ()
        full_37 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (4x-1x384xf32, 4x-1x384xui8) <- (4x-1x384xf32, None, 1xf32)
        dropout_14, dropout_15 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_19, None, full_37, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_19

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 4x-1x384xf32)
        add_20 = paddle._C_ops.add(layer_norm_9, dropout_14)

        # pd_op.layer_norm: (4x-1x384xf32, 4x-1xf32, 4x-1xf32) <- (4x-1x384xf32, 384xf32, 384xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_20, parameter_48, parameter_47, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_47, parameter_48

        # pd_op.matmul: (4x-1x1536xf32) <- (4x-1x384xf32, 384x1536xf32)
        matmul_22 = paddle._C_ops.matmul(layer_norm_12, parameter_46, False, False)
        del parameter_46

        # pd_op.add: (4x-1x1536xf32) <- (4x-1x1536xf32, 1536xf32)
        add_21 = paddle._C_ops.add(matmul_22, parameter_45)
        del parameter_45

        # pd_op.relu: (4x-1x1536xf32) <- (4x-1x1536xf32)
        relu_1 = paddle._C_ops.relu(add_21)
        del add_21

        # pd_op.full: (1xf32) <- ()
        full_38 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (4x-1x1536xf32, 4x-1x1536xui8) <- (4x-1x1536xf32, None, 1xf32)
        dropout_16, dropout_17 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                relu_1, None, full_38, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (4x-1x384xf32) <- (4x-1x1536xf32, 1536x384xf32)
        matmul_23 = paddle._C_ops.matmul(dropout_16, parameter_44, False, False)
        del parameter_44

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 384xf32)
        add_22 = paddle._C_ops.add(matmul_23, parameter_43)
        del parameter_43

        # pd_op.full: (1xf32) <- ()
        full_39 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (4x-1x384xf32, 4x-1x384xui8) <- (4x-1x384xf32, None, 1xf32)
        dropout_18, dropout_19 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_22, None, full_39, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_22

        # pd_op.full: (1xf32) <- ()
        full_40 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (4x-1x384xf32, 4x-1x384xui8) <- (4x-1x384xf32, None, 1xf32)
        dropout_20, dropout_21 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                dropout_18, None, full_40, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del dropout_18

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 4x-1x384xf32)
        add_23 = paddle._C_ops.add(layer_norm_12, dropout_20)

        # pd_op.layer_norm: (4x-1x384xf32, 4x-1xf32, 4x-1xf32) <- (4x-1x384xf32, 384xf32, 384xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_23, parameter_42, parameter_41, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_41, parameter_42

        # pd_op.shape64: (3xi64) <- (4x-1x384xf32)
        shape64_5 = paddle._C_ops.shape64(layer_norm_15)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_38 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_39 = [2]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(
            shape64_5, [0], full_int_array_38, full_int_array_39, [1], [0]
        )
        del full_int_array_38, full_int_array_39, shape64_5

        # pd_op.matmul: (4x-1x1152xf32) <- (4x-1x384xf32, 384x1152xf32)
        matmul_24 = paddle._C_ops.matmul(layer_norm_15, parameter_40, False, False)
        del parameter_40

        # pd_op.add: (4x-1x1152xf32) <- (4x-1x1152xf32, 1152xf32)
        add_24 = paddle._C_ops.add(matmul_24, parameter_39)
        del parameter_39

        # pd_op.full: (xi64) <- ()
        full_41 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_42 = paddle._C_ops.full(
            [], float("3"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_43 = paddle._C_ops.full(
            [], float("12"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_44 = paddle._C_ops.full(
            [], float("32"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_12 = [full_41, slice_18, full_42, full_43, full_44]
        del full_41, full_42, full_43, full_44

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_12 = paddle._C_ops.stack(combine_12, 0)
        del combine_12

        # pd_op.reshape: (4x-1x3x12x32xf32) <- (4x-1x1152xf32, 5xi64)
        reshape_10 = paddle._C_ops.reshape(add_24, stack_12)
        del stack_12

        # pd_op.transpose: (3x4x12x-1x32xf32) <- (4x-1x3x12x32xf32)
        transpose_17 = paddle._C_ops.transpose(reshape_10, [2, 0, 3, 1, 4])
        del reshape_10

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_40 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_41 = [1]

        # pd_op.slice: (4x12x-1x32xf32) <- (3x4x12x-1x32xf32, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(
            transpose_17, [0], full_int_array_40, full_int_array_41, [1], [0]
        )
        del full_int_array_40, full_int_array_41

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_42 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_43 = [2]

        # pd_op.slice: (4x12x-1x32xf32) <- (3x4x12x-1x32xf32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(
            transpose_17, [0], full_int_array_42, full_int_array_43, [1], [0]
        )
        del full_int_array_42, full_int_array_43

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_44 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_45 = [3]

        # pd_op.slice: (4x12x-1x32xf32) <- (3x4x12x-1x32xf32, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(
            transpose_17, [0], full_int_array_44, full_int_array_45, [1], [0]
        )
        del full_int_array_44, full_int_array_45

        # pd_op.transpose: (4x12x32x-1xf32) <- (4x12x-1x32xf32)
        transpose_18 = paddle._C_ops.transpose(slice_20, [0, 1, 3, 2])
        del slice_20

        # pd_op.matmul: (4x12x-1x-1xf32) <- (4x12x-1x32xf32, 4x12x32x-1xf32)
        matmul_25 = paddle._C_ops.matmul(slice_19, transpose_18, False, False)

        # pd_op.full: (1xf32) <- ()
        full_45 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (4x12x-1x-1xf32) <- (4x12x-1x-1xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(matmul_25, full_45, float("0"), True)
        del full_45, matmul_25

        # pd_op.add: (4x12x-1x-1xf32) <- (4x12x-1x-1xf32, 1x1x-1x-1xf32)
        add_25 = paddle._C_ops.add(scale_6, unsqueeze_0)

        # pd_op.softmax: (4x12x-1x-1xf32) <- (4x12x-1x-1xf32)
        softmax_4 = paddle._C_ops.softmax(add_25, -1)
        del add_25

        # pd_op.matmul: (4x12x-1x32xf32) <- (4x12x-1x-1xf32, 4x12x-1x32xf32)
        matmul_26 = paddle._C_ops.matmul(softmax_4, slice_21, False, False)

        # pd_op.transpose: (4x-1x12x32xf32) <- (4x12x-1x32xf32)
        transpose_19 = paddle._C_ops.transpose(matmul_26, [0, 2, 1, 3])
        del matmul_26

        # pd_op.full: (xi64) <- ()
        full_46 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_47 = paddle._C_ops.full(
            [], float("384"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_13 = [full_46, slice_18, full_47]
        del full_46, full_47, slice_18

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_13 = paddle._C_ops.stack(combine_13, 0)
        del combine_13

        # pd_op.reshape: (4x-1x384xf32) <- (4x-1x12x32xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_19, stack_13)
        del stack_13

        # pd_op.matmul: (4x-1x384xf32) <- (4x-1x384xf32, 384x384xf32)
        matmul_27 = paddle._C_ops.matmul(reshape_11, parameter_38, False, False)
        del parameter_38

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 384xf32)
        add_26 = paddle._C_ops.add(matmul_27, parameter_37)
        del parameter_37

        # pd_op.full: (1xf32) <- ()
        full_48 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (4x-1x384xf32, 4x-1x384xui8) <- (4x-1x384xf32, None, 1xf32)
        dropout_22, dropout_23 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_26, None, full_48, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_26

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 4x-1x384xf32)
        add_27 = paddle._C_ops.add(layer_norm_15, dropout_22)

        # pd_op.layer_norm: (4x-1x384xf32, 4x-1xf32, 4x-1xf32) <- (4x-1x384xf32, 384xf32, 384xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_27, parameter_36, parameter_35, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_35, parameter_36

        # pd_op.shape64: (3xi64) <- (4x-1x384xf32)
        shape64_6 = paddle._C_ops.shape64(layer_norm_18)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_46 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_47 = [2]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(
            shape64_6, [0], full_int_array_46, full_int_array_47, [1], [0]
        )
        del full_int_array_46, full_int_array_47, shape64_6

        # pd_op.matmul: (4x-1x384xf32) <- (4x-1x384xf32, 384x384xf32)
        matmul_28 = paddle._C_ops.matmul(layer_norm_18, parameter_34, False, False)
        del parameter_34

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 384xf32)
        add_28 = paddle._C_ops.add(matmul_28, parameter_33)
        del parameter_33

        # pd_op.full: (xi64) <- ()
        full_49 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_50 = paddle._C_ops.full(
            [], float("12"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_51 = paddle._C_ops.full(
            [], float("32"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_14 = [full_49, slice_22, full_50, full_51]
        del full_49, full_50, full_51

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_14 = paddle._C_ops.stack(combine_14, 0)
        del combine_14

        # pd_op.reshape: (4x-1x12x32xf32) <- (4x-1x384xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(add_28, stack_14)
        del stack_14

        # pd_op.transpose: (4x12x-1x32xf32) <- (4x-1x12x32xf32)
        transpose_20 = paddle._C_ops.transpose(reshape_12, [0, 2, 1, 3])
        del reshape_12

        # pd_op.matmul: (4x40x768xf32) <- (4x40x384xf32, 384x768xf32)
        matmul_29 = paddle._C_ops.matmul(matmul_1, parameter_32, False, False)
        del parameter_32

        # pd_op.add: (4x40x768xf32) <- (4x40x768xf32, 768xf32)
        add_29 = paddle._C_ops.add(matmul_29, parameter_31)
        del parameter_31

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_48 = [0, 40, 2, 12, 32]

        # pd_op.reshape: (4x40x2x12x32xf32) <- (4x40x768xf32, 5xi64)
        reshape_13 = paddle._C_ops.reshape(add_29, full_int_array_48)
        del full_int_array_48

        # pd_op.transpose: (2x4x12x40x32xf32) <- (4x40x2x12x32xf32)
        transpose_21 = paddle._C_ops.transpose(reshape_13, [2, 0, 3, 1, 4])
        del reshape_13

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_49 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_50 = [1]

        # pd_op.slice: (4x12x40x32xf32) <- (2x4x12x40x32xf32, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(
            transpose_21, [0], full_int_array_49, full_int_array_50, [1], [0]
        )
        del full_int_array_49, full_int_array_50

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_51 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_52 = [2]

        # pd_op.slice: (4x12x40x32xf32) <- (2x4x12x40x32xf32, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(
            transpose_21, [0], full_int_array_51, full_int_array_52, [1], [0]
        )
        del full_int_array_51, full_int_array_52

        # pd_op.transpose: (4x12x32x40xf32) <- (4x12x40x32xf32)
        transpose_22 = paddle._C_ops.transpose(slice_23, [0, 1, 3, 2])
        del slice_23

        # pd_op.matmul: (4x12x-1x40xf32) <- (4x12x-1x32xf32, 4x12x32x40xf32)
        matmul_30 = paddle._C_ops.matmul(transpose_20, transpose_22, False, False)

        # pd_op.full: (1xf32) <- ()
        full_52 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (4x12x-1x40xf32) <- (4x12x-1x40xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(matmul_30, full_52, float("0"), True)
        del full_52, matmul_30

        # pd_op.softmax: (4x12x-1x40xf32) <- (4x12x-1x40xf32)
        softmax_5 = paddle._C_ops.softmax(scale_7, -1)
        del scale_7

        # pd_op.matmul: (4x12x-1x32xf32) <- (4x12x-1x40xf32, 4x12x40x32xf32)
        matmul_31 = paddle._C_ops.matmul(softmax_5, slice_24, False, False)

        # pd_op.transpose: (4x-1x12x32xf32) <- (4x12x-1x32xf32)
        transpose_23 = paddle._C_ops.transpose(matmul_31, [0, 2, 1, 3])
        del matmul_31

        # pd_op.full: (xi64) <- ()
        full_53 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_54 = paddle._C_ops.full(
            [], float("384"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_15 = [full_53, slice_22, full_54]
        del full_53, full_54, slice_22

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_15 = paddle._C_ops.stack(combine_15, 0)
        del combine_15

        # pd_op.reshape: (4x-1x384xf32) <- (4x-1x12x32xf32, 3xi64)
        reshape_14 = paddle._C_ops.reshape(transpose_23, stack_15)
        del stack_15

        # pd_op.matmul: (4x-1x384xf32) <- (4x-1x384xf32, 384x384xf32)
        matmul_32 = paddle._C_ops.matmul(reshape_14, parameter_30, False, False)
        del parameter_30

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 384xf32)
        add_30 = paddle._C_ops.add(matmul_32, parameter_29)
        del parameter_29

        # pd_op.full: (1xf32) <- ()
        full_55 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (4x-1x384xf32, 4x-1x384xui8) <- (4x-1x384xf32, None, 1xf32)
        dropout_24, dropout_25 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_30, None, full_55, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_30

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 4x-1x384xf32)
        add_31 = paddle._C_ops.add(layer_norm_18, dropout_24)

        # pd_op.layer_norm: (4x-1x384xf32, 4x-1xf32, 4x-1xf32) <- (4x-1x384xf32, 384xf32, 384xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_31, parameter_28, parameter_27, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_27, parameter_28

        # pd_op.matmul: (4x-1x1536xf32) <- (4x-1x384xf32, 384x1536xf32)
        matmul_33 = paddle._C_ops.matmul(layer_norm_21, parameter_26, False, False)
        del parameter_26

        # pd_op.add: (4x-1x1536xf32) <- (4x-1x1536xf32, 1536xf32)
        add_32 = paddle._C_ops.add(matmul_33, parameter_25)
        del parameter_25

        # pd_op.relu: (4x-1x1536xf32) <- (4x-1x1536xf32)
        relu_2 = paddle._C_ops.relu(add_32)
        del add_32

        # pd_op.full: (1xf32) <- ()
        full_56 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (4x-1x1536xf32, 4x-1x1536xui8) <- (4x-1x1536xf32, None, 1xf32)
        dropout_26, dropout_27 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                relu_2, None, full_56, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (4x-1x384xf32) <- (4x-1x1536xf32, 1536x384xf32)
        matmul_34 = paddle._C_ops.matmul(dropout_26, parameter_24, False, False)
        del parameter_24

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 384xf32)
        add_33 = paddle._C_ops.add(matmul_34, parameter_23)
        del parameter_23

        # pd_op.full: (1xf32) <- ()
        full_57 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (4x-1x384xf32, 4x-1x384xui8) <- (4x-1x384xf32, None, 1xf32)
        dropout_28, dropout_29 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_33, None, full_57, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_33

        # pd_op.full: (1xf32) <- ()
        full_58 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (4x-1x384xf32, 4x-1x384xui8) <- (4x-1x384xf32, None, 1xf32)
        dropout_30, dropout_31 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                dropout_28, None, full_58, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del dropout_28

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 4x-1x384xf32)
        add_34 = paddle._C_ops.add(layer_norm_21, dropout_30)

        # pd_op.layer_norm: (4x-1x384xf32, 4x-1xf32, 4x-1xf32) <- (4x-1x384xf32, 384xf32, 384xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_34, parameter_22, parameter_21, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_21, parameter_22

        # pd_op.shape64: (3xi64) <- (4x-1x384xf32)
        shape64_7 = paddle._C_ops.shape64(layer_norm_24)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_53 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_54 = [2]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(
            shape64_7, [0], full_int_array_53, full_int_array_54, [1], [0]
        )
        del full_int_array_53, full_int_array_54, shape64_7

        # pd_op.matmul: (4x-1x1152xf32) <- (4x-1x384xf32, 384x1152xf32)
        matmul_35 = paddle._C_ops.matmul(layer_norm_24, parameter_20, False, False)
        del parameter_20

        # pd_op.add: (4x-1x1152xf32) <- (4x-1x1152xf32, 1152xf32)
        add_35 = paddle._C_ops.add(matmul_35, parameter_19)
        del parameter_19

        # pd_op.full: (xi64) <- ()
        full_59 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_60 = paddle._C_ops.full(
            [], float("3"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_61 = paddle._C_ops.full(
            [], float("12"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_62 = paddle._C_ops.full(
            [], float("32"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_16 = [full_59, slice_25, full_60, full_61, full_62]
        del full_59, full_60, full_61, full_62

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_16 = paddle._C_ops.stack(combine_16, 0)
        del combine_16

        # pd_op.reshape: (4x-1x3x12x32xf32) <- (4x-1x1152xf32, 5xi64)
        reshape_15 = paddle._C_ops.reshape(add_35, stack_16)
        del stack_16

        # pd_op.transpose: (3x4x12x-1x32xf32) <- (4x-1x3x12x32xf32)
        transpose_24 = paddle._C_ops.transpose(reshape_15, [2, 0, 3, 1, 4])
        del reshape_15

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_55 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_56 = [1]

        # pd_op.slice: (4x12x-1x32xf32) <- (3x4x12x-1x32xf32, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(
            transpose_24, [0], full_int_array_55, full_int_array_56, [1], [0]
        )
        del full_int_array_55, full_int_array_56

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_57 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_58 = [2]

        # pd_op.slice: (4x12x-1x32xf32) <- (3x4x12x-1x32xf32, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(
            transpose_24, [0], full_int_array_57, full_int_array_58, [1], [0]
        )
        del full_int_array_57, full_int_array_58

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_59 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_60 = [3]

        # pd_op.slice: (4x12x-1x32xf32) <- (3x4x12x-1x32xf32, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(
            transpose_24, [0], full_int_array_59, full_int_array_60, [1], [0]
        )
        del full_int_array_59, full_int_array_60

        # pd_op.transpose: (4x12x32x-1xf32) <- (4x12x-1x32xf32)
        transpose_25 = paddle._C_ops.transpose(slice_27, [0, 1, 3, 2])
        del slice_27

        # pd_op.matmul: (4x12x-1x-1xf32) <- (4x12x-1x32xf32, 4x12x32x-1xf32)
        matmul_36 = paddle._C_ops.matmul(slice_26, transpose_25, False, False)

        # pd_op.full: (1xf32) <- ()
        full_63 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (4x12x-1x-1xf32) <- (4x12x-1x-1xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(matmul_36, full_63, float("0"), True)
        del full_63, matmul_36

        # pd_op.add: (4x12x-1x-1xf32) <- (4x12x-1x-1xf32, 1x1x-1x-1xf32)
        add_36 = paddle._C_ops.add(scale_8, unsqueeze_0)

        # pd_op.softmax: (4x12x-1x-1xf32) <- (4x12x-1x-1xf32)
        softmax_6 = paddle._C_ops.softmax(add_36, -1)
        del add_36

        # pd_op.matmul: (4x12x-1x32xf32) <- (4x12x-1x-1xf32, 4x12x-1x32xf32)
        matmul_37 = paddle._C_ops.matmul(softmax_6, slice_28, False, False)

        # pd_op.transpose: (4x-1x12x32xf32) <- (4x12x-1x32xf32)
        transpose_26 = paddle._C_ops.transpose(matmul_37, [0, 2, 1, 3])
        del matmul_37

        # pd_op.full: (xi64) <- ()
        full_64 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_65 = paddle._C_ops.full(
            [], float("384"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_17 = [full_64, slice_25, full_65]
        del full_64, full_65, slice_25

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_17 = paddle._C_ops.stack(combine_17, 0)
        del combine_17

        # pd_op.reshape: (4x-1x384xf32) <- (4x-1x12x32xf32, 3xi64)
        reshape_16 = paddle._C_ops.reshape(transpose_26, stack_17)
        del stack_17

        # pd_op.matmul: (4x-1x384xf32) <- (4x-1x384xf32, 384x384xf32)
        matmul_38 = paddle._C_ops.matmul(reshape_16, parameter_18, False, False)
        del parameter_18

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 384xf32)
        add_37 = paddle._C_ops.add(matmul_38, parameter_17)
        del parameter_17

        # pd_op.full: (1xf32) <- ()
        full_66 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (4x-1x384xf32, 4x-1x384xui8) <- (4x-1x384xf32, None, 1xf32)
        dropout_32, dropout_33 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_37, None, full_66, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_37

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 4x-1x384xf32)
        add_38 = paddle._C_ops.add(layer_norm_24, dropout_32)

        # pd_op.layer_norm: (4x-1x384xf32, 4x-1xf32, 4x-1xf32) <- (4x-1x384xf32, 384xf32, 384xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_38, parameter_16, parameter_15, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_15, parameter_16

        # pd_op.shape64: (3xi64) <- (4x-1x384xf32)
        shape64_8 = paddle._C_ops.shape64(layer_norm_27)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_61 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_62 = [2]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(
            shape64_8, [0], full_int_array_61, full_int_array_62, [1], [0]
        )
        del full_int_array_61, full_int_array_62, shape64_8

        # pd_op.matmul: (4x-1x384xf32) <- (4x-1x384xf32, 384x384xf32)
        matmul_39 = paddle._C_ops.matmul(layer_norm_27, parameter_14, False, False)
        del parameter_14

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 384xf32)
        add_39 = paddle._C_ops.add(matmul_39, parameter_13)
        del parameter_13

        # pd_op.full: (xi64) <- ()
        full_67 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_68 = paddle._C_ops.full(
            [], float("12"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_69 = paddle._C_ops.full(
            [], float("32"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_18 = [full_67, slice_29, full_68, full_69]
        del full_67, full_68, full_69

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_18 = paddle._C_ops.stack(combine_18, 0)
        del combine_18

        # pd_op.reshape: (4x-1x12x32xf32) <- (4x-1x384xf32, 4xi64)
        reshape_17 = paddle._C_ops.reshape(add_39, stack_18)
        del stack_18

        # pd_op.transpose: (4x12x-1x32xf32) <- (4x-1x12x32xf32)
        transpose_27 = paddle._C_ops.transpose(reshape_17, [0, 2, 1, 3])
        del reshape_17

        # pd_op.matmul: (4x40x768xf32) <- (4x40x384xf32, 384x768xf32)
        matmul_40 = paddle._C_ops.matmul(matmul_1, parameter_12, False, False)
        del parameter_12

        # pd_op.add: (4x40x768xf32) <- (4x40x768xf32, 768xf32)
        add_40 = paddle._C_ops.add(matmul_40, parameter_11)
        del parameter_11

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_63 = [0, 40, 2, 12, 32]

        # pd_op.reshape: (4x40x2x12x32xf32) <- (4x40x768xf32, 5xi64)
        reshape_18 = paddle._C_ops.reshape(add_40, full_int_array_63)
        del full_int_array_63

        # pd_op.transpose: (2x4x12x40x32xf32) <- (4x40x2x12x32xf32)
        transpose_28 = paddle._C_ops.transpose(reshape_18, [2, 0, 3, 1, 4])
        del reshape_18

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_64 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_65 = [1]

        # pd_op.slice: (4x12x40x32xf32) <- (2x4x12x40x32xf32, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(
            transpose_28, [0], full_int_array_64, full_int_array_65, [1], [0]
        )
        del full_int_array_64, full_int_array_65

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_66 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_67 = [2]

        # pd_op.slice: (4x12x40x32xf32) <- (2x4x12x40x32xf32, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(
            transpose_28, [0], full_int_array_66, full_int_array_67, [1], [0]
        )
        del full_int_array_66, full_int_array_67

        # pd_op.transpose: (4x12x32x40xf32) <- (4x12x40x32xf32)
        transpose_29 = paddle._C_ops.transpose(slice_30, [0, 1, 3, 2])
        del slice_30

        # pd_op.matmul: (4x12x-1x40xf32) <- (4x12x-1x32xf32, 4x12x32x40xf32)
        matmul_41 = paddle._C_ops.matmul(transpose_27, transpose_29, False, False)

        # pd_op.full: (1xf32) <- ()
        full_70 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (4x12x-1x40xf32) <- (4x12x-1x40xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(matmul_41, full_70, float("0"), True)
        del full_70, matmul_41

        # pd_op.softmax: (4x12x-1x40xf32) <- (4x12x-1x40xf32)
        softmax_7 = paddle._C_ops.softmax(scale_9, -1)
        del scale_9

        # pd_op.matmul: (4x12x-1x32xf32) <- (4x12x-1x40xf32, 4x12x40x32xf32)
        matmul_42 = paddle._C_ops.matmul(softmax_7, slice_31, False, False)

        # pd_op.transpose: (4x-1x12x32xf32) <- (4x12x-1x32xf32)
        transpose_30 = paddle._C_ops.transpose(matmul_42, [0, 2, 1, 3])
        del matmul_42

        # pd_op.full: (xi64) <- ()
        full_71 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_72 = paddle._C_ops.full(
            [], float("384"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_19 = [full_71, slice_29, full_72]
        del full_71, full_72, slice_29

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_19 = paddle._C_ops.stack(combine_19, 0)
        del combine_19

        # pd_op.reshape: (4x-1x384xf32) <- (4x-1x12x32xf32, 3xi64)
        reshape_19 = paddle._C_ops.reshape(transpose_30, stack_19)
        del stack_19

        # pd_op.matmul: (4x-1x384xf32) <- (4x-1x384xf32, 384x384xf32)
        matmul_43 = paddle._C_ops.matmul(reshape_19, parameter_10, False, False)
        del parameter_10

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 384xf32)
        add_41 = paddle._C_ops.add(matmul_43, parameter_9)
        del parameter_9

        # pd_op.full: (1xf32) <- ()
        full_73 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (4x-1x384xf32, 4x-1x384xui8) <- (4x-1x384xf32, None, 1xf32)
        dropout_34, dropout_35 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_41, None, full_73, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_41

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 4x-1x384xf32)
        add_42 = paddle._C_ops.add(layer_norm_27, dropout_34)

        # pd_op.layer_norm: (4x-1x384xf32, 4x-1xf32, 4x-1xf32) <- (4x-1x384xf32, 384xf32, 384xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_42, parameter_8, parameter_7, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_7, parameter_8

        # pd_op.matmul: (4x-1x1536xf32) <- (4x-1x384xf32, 384x1536xf32)
        matmul_44 = paddle._C_ops.matmul(layer_norm_30, parameter_6, False, False)
        del parameter_6

        # pd_op.add: (4x-1x1536xf32) <- (4x-1x1536xf32, 1536xf32)
        add_43 = paddle._C_ops.add(matmul_44, parameter_5)
        del parameter_5

        # pd_op.relu: (4x-1x1536xf32) <- (4x-1x1536xf32)
        relu_3 = paddle._C_ops.relu(add_43)
        del add_43

        # pd_op.full: (1xf32) <- ()
        full_74 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (4x-1x1536xf32, 4x-1x1536xui8) <- (4x-1x1536xf32, None, 1xf32)
        dropout_36, dropout_37 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                relu_3, None, full_74, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (4x-1x384xf32) <- (4x-1x1536xf32, 1536x384xf32)
        matmul_45 = paddle._C_ops.matmul(dropout_36, parameter_4, False, False)
        del parameter_4

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 384xf32)
        add_44 = paddle._C_ops.add(matmul_45, parameter_3)
        del parameter_3

        # pd_op.full: (1xf32) <- ()
        full_75 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (4x-1x384xf32, 4x-1x384xui8) <- (4x-1x384xf32, None, 1xf32)
        dropout_38, dropout_39 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_44, None, full_75, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_44

        # pd_op.full: (1xf32) <- ()
        full_76 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (4x-1x384xf32, 4x-1x384xui8) <- (4x-1x384xf32, None, 1xf32)
        dropout_40, dropout_41 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                dropout_38, None, full_76, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del dropout_38

        # pd_op.add: (4x-1x384xf32) <- (4x-1x384xf32, 4x-1x384xf32)
        add_45 = paddle._C_ops.add(layer_norm_30, dropout_40)

        # pd_op.layer_norm: (4x-1x384xf32, 4x-1xf32, 4x-1xf32) <- (4x-1x384xf32, 384xf32, 384xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_45, parameter_2, parameter_1, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_1, parameter_2

        # pd_op.matmul: (4x-1x6629xf32) <- (4x-1x384xf32, 384x6629xf32)
        matmul_0 = paddle._C_ops.matmul(layer_norm_33, parameter_0, False, False)
        del (
            add_12,
            add_13,
            add_16,
            add_17,
            add_18,
            add_2,
            add_20,
            add_23,
            add_24,
            add_27,
            add_28,
            add_29,
            add_31,
            add_34,
            add_35,
            add_38,
            add_39,
            add_40,
            add_42,
            add_45,
            add_5,
            add_6,
            add_7,
            add_9,
            dropout_1,
            dropout_10,
            dropout_11,
            dropout_12,
            dropout_13,
            dropout_14,
            dropout_15,
            dropout_16,
            dropout_17,
            dropout_19,
            dropout_2,
            dropout_20,
            dropout_21,
            dropout_22,
            dropout_23,
            dropout_24,
            dropout_25,
            dropout_26,
            dropout_27,
            dropout_29,
            dropout_3,
            dropout_30,
            dropout_31,
            dropout_32,
            dropout_33,
            dropout_34,
            dropout_35,
            dropout_36,
            dropout_37,
            dropout_39,
            dropout_4,
            dropout_40,
            dropout_41,
            dropout_5,
            dropout_6,
            dropout_7,
            dropout_9,
            full_12,
            full_19,
            full_2,
            full_20,
            full_21,
            full_22,
            full_30,
            full_37,
            full_38,
            full_39,
            full_40,
            full_48,
            full_55,
            full_56,
            full_57,
            full_58,
            full_66,
            full_73,
            full_74,
            full_75,
            full_76,
            layer_norm_0,
            layer_norm_1,
            layer_norm_10,
            layer_norm_11,
            layer_norm_12,
            layer_norm_13,
            layer_norm_14,
            layer_norm_15,
            layer_norm_16,
            layer_norm_17,
            layer_norm_18,
            layer_norm_19,
            layer_norm_2,
            layer_norm_20,
            layer_norm_21,
            layer_norm_22,
            layer_norm_23,
            layer_norm_24,
            layer_norm_25,
            layer_norm_26,
            layer_norm_27,
            layer_norm_28,
            layer_norm_29,
            layer_norm_3,
            layer_norm_30,
            layer_norm_31,
            layer_norm_32,
            layer_norm_33,
            layer_norm_34,
            layer_norm_35,
            layer_norm_4,
            layer_norm_5,
            layer_norm_6,
            layer_norm_7,
            layer_norm_8,
            layer_norm_9,
            matmul_1,
            matmul_10,
            matmul_11,
            matmul_12,
            matmul_13,
            matmul_16,
            matmul_17,
            matmul_18,
            matmul_2,
            matmul_21,
            matmul_22,
            matmul_23,
            matmul_24,
            matmul_27,
            matmul_28,
            matmul_29,
            matmul_32,
            matmul_33,
            matmul_34,
            matmul_35,
            matmul_38,
            matmul_39,
            matmul_40,
            matmul_43,
            matmul_44,
            matmul_45,
            matmul_5,
            matmul_6,
            matmul_7,
            parameter_0,
            relu_0,
            relu_1,
            relu_2,
            relu_3,
            reshape_1,
            reshape_11,
            reshape_14,
            reshape_16,
            reshape_19,
            reshape_4,
            reshape_6,
            reshape_9,
            scale_2,
            scale_4,
            scale_6,
            scale_8,
            slice_1,
            slice_10,
            slice_12,
            slice_14,
            slice_17,
            slice_19,
            slice_21,
            slice_24,
            slice_26,
            slice_28,
            slice_3,
            slice_31,
            slice_5,
            slice_7,
            softmax_0,
            softmax_1,
            softmax_2,
            softmax_3,
            softmax_4,
            softmax_5,
            softmax_6,
            softmax_7,
            transpose_0,
            transpose_1,
            transpose_10,
            transpose_11,
            transpose_12,
            transpose_13,
            transpose_14,
            transpose_15,
            transpose_16,
            transpose_17,
            transpose_18,
            transpose_19,
            transpose_2,
            transpose_20,
            transpose_21,
            transpose_22,
            transpose_23,
            transpose_24,
            transpose_25,
            transpose_26,
            transpose_27,
            transpose_28,
            transpose_29,
            transpose_3,
            transpose_30,
            transpose_4,
            transpose_5,
            transpose_6,
            transpose_7,
            transpose_8,
            transpose_9,
            unsqueeze_0,
        )

        return matmul_0
