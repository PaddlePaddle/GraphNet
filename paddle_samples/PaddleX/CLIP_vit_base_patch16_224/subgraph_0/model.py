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
        parameter_83,
        parameter_84,
        parameter_85,
        parameter_86,
        parameter_87,
        parameter_88,
        parameter_89,
        parameter_90,
        parameter_91,
        parameter_92,
        parameter_93,
        parameter_94,
        parameter_95,
        parameter_96,
        parameter_97,
        parameter_98,
        parameter_99,
        parameter_100,
        parameter_101,
        parameter_102,
        parameter_103,
        parameter_104,
        parameter_105,
        parameter_106,
        parameter_107,
        parameter_108,
        parameter_109,
        parameter_110,
        parameter_111,
        parameter_112,
        parameter_113,
        parameter_114,
        parameter_115,
        parameter_116,
        parameter_117,
        parameter_118,
        parameter_119,
        parameter_120,
        parameter_121,
        parameter_122,
        parameter_123,
        parameter_124,
        parameter_125,
        parameter_126,
        parameter_127,
        parameter_128,
        parameter_129,
        parameter_130,
        parameter_131,
        parameter_132,
        parameter_133,
        parameter_134,
        parameter_135,
        parameter_136,
        parameter_137,
        parameter_138,
        parameter_139,
        parameter_140,
        parameter_141,
        parameter_142,
        parameter_143,
        parameter_144,
        parameter_145,
        parameter_146,
        parameter_147,
        parameter_148,
        parameter_149,
        parameter_150,
        data_0,
        data_1,
        data_2,
        data_3,
    ):
        # pd_op.conv2d: (-1x768x14x14xf32) <- (-1x3x224x224xf32, 768x3x16x16xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_3, parameter_150, [16, 16], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_3, parameter_150

        # pd_op.shape64: (4xi64) <- (-1x768x14x14xf32)
        shape64_0 = paddle._C_ops.shape64(conv2d_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del full_int_array_0, full_int_array_1, shape64_0

        # pd_op.flatten: (-1x768x196xf32) <- (-1x768x14x14xf32)
        flatten_0 = paddle._C_ops.flatten(conv2d_0, 2, 3)
        del conv2d_0

        # pd_op.transpose: (-1x196x768xf32) <- (-1x768x196xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_0, [0, 2, 1])
        del flatten_0

        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("-1"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("-1"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_0 = [data_2, full_0, full_1]
        del data_2, full_0, full_1

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.expand: (-1x1x768xf32) <- (1x1x768xf32, 3xi64)
        expand_0 = paddle._C_ops.expand(data_0, stack_0)
        del data_0, stack_0

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x1x768xf32, -1x196x768xf32]) <- (-1x1x768xf32, -1x196x768xf32)
        combine_1 = [expand_0, transpose_0]
        del expand_0, transpose_0

        # pd_op.concat: (-1x197x768xf32) <- ([-1x1x768xf32, -1x196x768xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_1, full_2)
        del combine_1, full_2

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 1x197x768xf32)
        add_1 = paddle._C_ops.add(concat_0, data_1)
        del concat_0, data_1

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_1, parameter_149, parameter_148, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_1, parameter_148, parameter_149

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                layer_norm_0, parameter_147, parameter_146, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_146, parameter_147

        # pd_op.shape64: (3xi64) <- (-1x197x768xf32)
        shape64_1 = paddle._C_ops.shape64(layer_norm_3)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del full_int_array_2, full_int_array_3, shape64_1

        # pd_op.matmul: (-1x197x2304xf32) <- (-1x197x768xf32, 768x2304xf32)
        matmul_0 = paddle._C_ops.matmul(layer_norm_3, parameter_145, False, False)
        del layer_norm_3, parameter_145

        # pd_op.add: (-1x197x2304xf32) <- (-1x197x2304xf32, 2304xf32)
        add_2 = paddle._C_ops.add(matmul_0, parameter_144)
        del matmul_0, parameter_144

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_4 = [-1, 197, 3, 12, 64]

        # pd_op.reshape: (-1x197x3x12x64xf32) <- (-1x197x2304xf32, 5xi64)
        reshape_0 = paddle._C_ops.reshape(add_2, full_int_array_4)
        del add_2, full_int_array_4

        # pd_op.transpose: (3x-1x12x197x64xf32) <- (-1x197x3x12x64xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_0, [2, 0, 3, 1, 4])
        del reshape_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [1]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            transpose_1, [0], full_int_array_5, full_int_array_6, [1], [0]
        )
        del full_int_array_5, full_int_array_6

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [2]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            transpose_1, [0], full_int_array_7, full_int_array_8, [1], [0]
        )
        del full_int_array_7, full_int_array_8

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_10 = [3]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            transpose_1, [0], full_int_array_9, full_int_array_10, [1], [0]
        )
        del full_int_array_10, full_int_array_9, transpose_1

        # pd_op.transpose: (-1x12x64x197xf32) <- (-1x12x197x64xf32)
        transpose_2 = paddle._C_ops.transpose(slice_3, [0, 1, 3, 2])
        del slice_3

        # pd_op.matmul: (-1x12x197x197xf32) <- (-1x12x197x64xf32, -1x12x64x197xf32)
        matmul_1 = paddle._C_ops.matmul(slice_2, transpose_2, False, False)
        del slice_2, transpose_2

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x12x197x197xf32) <- (-1x12x197x197xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(matmul_1, full_3, float("0"), True)
        del full_3, matmul_1

        # pd_op.softmax: (-1x12x197x197xf32) <- (-1x12x197x197xf32)
        softmax_0 = paddle._C_ops.softmax(scale_0, -1)
        del scale_0

        # pd_op.matmul: (-1x12x197x64xf32) <- (-1x12x197x197xf32, -1x12x197x64xf32)
        matmul_2 = paddle._C_ops.matmul(softmax_0, slice_4, False, False)
        del slice_4, softmax_0

        # pd_op.transpose: (-1x197x12x64xf32) <- (-1x12x197x64xf32)
        transpose_3 = paddle._C_ops.transpose(matmul_2, [0, 2, 1, 3])
        del matmul_2

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_11 = [-1, 197, 768]

        # pd_op.reshape: (-1x197x768xf32) <- (-1x197x12x64xf32, 3xi64)
        reshape_1 = paddle._C_ops.reshape(transpose_3, full_int_array_11)
        del full_int_array_11, transpose_3

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x768xf32, 768x768xf32)
        matmul_3 = paddle._C_ops.matmul(reshape_1, parameter_143, False, False)
        del parameter_143, reshape_1

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_3 = paddle._C_ops.add(matmul_3, parameter_142)
        del matmul_3, parameter_142

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_4 = paddle._C_ops.add(layer_norm_0, add_3)
        del add_3, layer_norm_0

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_4, parameter_141, parameter_140, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_140, parameter_141

        # pd_op.matmul: (-1x197x3072xf32) <- (-1x197x768xf32, 768x3072xf32)
        matmul_4 = paddle._C_ops.matmul(layer_norm_6, parameter_139, False, False)
        del layer_norm_6, parameter_139

        # pd_op.add: (-1x197x3072xf32) <- (-1x197x3072xf32, 3072xf32)
        add_5 = paddle._C_ops.add(matmul_4, parameter_138)
        del matmul_4, parameter_138

        # pd_op.gelu: (-1x197x3072xf32) <- (-1x197x3072xf32)
        gelu_0 = paddle._C_ops.gelu(add_5, False)
        del add_5

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x3072xf32, 3072x768xf32)
        matmul_5 = paddle._C_ops.matmul(gelu_0, parameter_137, False, False)
        del gelu_0, parameter_137

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_6 = paddle._C_ops.add(matmul_5, parameter_136)
        del matmul_5, parameter_136

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_7 = paddle._C_ops.add(add_4, add_6)
        del add_4, add_6

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_7, parameter_135, parameter_134, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_134, parameter_135

        # pd_op.shape64: (3xi64) <- (-1x197x768xf32)
        shape64_2 = paddle._C_ops.shape64(layer_norm_9)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_12 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_13 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_12, full_int_array_13, [1], [0]
        )
        del full_int_array_12, full_int_array_13, shape64_2

        # pd_op.matmul: (-1x197x2304xf32) <- (-1x197x768xf32, 768x2304xf32)
        matmul_6 = paddle._C_ops.matmul(layer_norm_9, parameter_133, False, False)
        del layer_norm_9, parameter_133

        # pd_op.add: (-1x197x2304xf32) <- (-1x197x2304xf32, 2304xf32)
        add_8 = paddle._C_ops.add(matmul_6, parameter_132)
        del matmul_6, parameter_132

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_14 = [-1, 197, 3, 12, 64]

        # pd_op.reshape: (-1x197x3x12x64xf32) <- (-1x197x2304xf32, 5xi64)
        reshape_2 = paddle._C_ops.reshape(add_8, full_int_array_14)
        del add_8, full_int_array_14

        # pd_op.transpose: (3x-1x12x197x64xf32) <- (-1x197x3x12x64xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_2, [2, 0, 3, 1, 4])
        del reshape_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_15 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_16 = [1]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            transpose_4, [0], full_int_array_15, full_int_array_16, [1], [0]
        )
        del full_int_array_15, full_int_array_16

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_17 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_18 = [2]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            transpose_4, [0], full_int_array_17, full_int_array_18, [1], [0]
        )
        del full_int_array_17, full_int_array_18

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_19 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_20 = [3]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            transpose_4, [0], full_int_array_19, full_int_array_20, [1], [0]
        )
        del full_int_array_19, full_int_array_20, transpose_4

        # pd_op.transpose: (-1x12x64x197xf32) <- (-1x12x197x64xf32)
        transpose_5 = paddle._C_ops.transpose(slice_7, [0, 1, 3, 2])
        del slice_7

        # pd_op.matmul: (-1x12x197x197xf32) <- (-1x12x197x64xf32, -1x12x64x197xf32)
        matmul_7 = paddle._C_ops.matmul(slice_6, transpose_5, False, False)
        del slice_6, transpose_5

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x12x197x197xf32) <- (-1x12x197x197xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(matmul_7, full_4, float("0"), True)
        del full_4, matmul_7

        # pd_op.softmax: (-1x12x197x197xf32) <- (-1x12x197x197xf32)
        softmax_1 = paddle._C_ops.softmax(scale_1, -1)
        del scale_1

        # pd_op.matmul: (-1x12x197x64xf32) <- (-1x12x197x197xf32, -1x12x197x64xf32)
        matmul_8 = paddle._C_ops.matmul(softmax_1, slice_8, False, False)
        del slice_8, softmax_1

        # pd_op.transpose: (-1x197x12x64xf32) <- (-1x12x197x64xf32)
        transpose_6 = paddle._C_ops.transpose(matmul_8, [0, 2, 1, 3])
        del matmul_8

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_21 = [-1, 197, 768]

        # pd_op.reshape: (-1x197x768xf32) <- (-1x197x12x64xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_6, full_int_array_21)
        del full_int_array_21, transpose_6

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x768xf32, 768x768xf32)
        matmul_9 = paddle._C_ops.matmul(reshape_3, parameter_131, False, False)
        del parameter_131, reshape_3

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_9 = paddle._C_ops.add(matmul_9, parameter_130)
        del matmul_9, parameter_130

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_10 = paddle._C_ops.add(add_7, add_9)
        del add_7, add_9

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_10, parameter_129, parameter_128, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_128, parameter_129

        # pd_op.matmul: (-1x197x3072xf32) <- (-1x197x768xf32, 768x3072xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_12, parameter_127, False, False)
        del layer_norm_12, parameter_127

        # pd_op.add: (-1x197x3072xf32) <- (-1x197x3072xf32, 3072xf32)
        add_11 = paddle._C_ops.add(matmul_10, parameter_126)
        del matmul_10, parameter_126

        # pd_op.gelu: (-1x197x3072xf32) <- (-1x197x3072xf32)
        gelu_1 = paddle._C_ops.gelu(add_11, False)
        del add_11

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x3072xf32, 3072x768xf32)
        matmul_11 = paddle._C_ops.matmul(gelu_1, parameter_125, False, False)
        del gelu_1, parameter_125

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_12 = paddle._C_ops.add(matmul_11, parameter_124)
        del matmul_11, parameter_124

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_13 = paddle._C_ops.add(add_10, add_12)
        del add_10, add_12

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_13, parameter_123, parameter_122, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_122, parameter_123

        # pd_op.shape64: (3xi64) <- (-1x197x768xf32)
        shape64_3 = paddle._C_ops.shape64(layer_norm_15)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_22 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_23 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_22, full_int_array_23, [1], [0]
        )
        del full_int_array_22, full_int_array_23, shape64_3

        # pd_op.matmul: (-1x197x2304xf32) <- (-1x197x768xf32, 768x2304xf32)
        matmul_12 = paddle._C_ops.matmul(layer_norm_15, parameter_121, False, False)
        del layer_norm_15, parameter_121

        # pd_op.add: (-1x197x2304xf32) <- (-1x197x2304xf32, 2304xf32)
        add_14 = paddle._C_ops.add(matmul_12, parameter_120)
        del matmul_12, parameter_120

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_24 = [-1, 197, 3, 12, 64]

        # pd_op.reshape: (-1x197x3x12x64xf32) <- (-1x197x2304xf32, 5xi64)
        reshape_4 = paddle._C_ops.reshape(add_14, full_int_array_24)
        del add_14, full_int_array_24

        # pd_op.transpose: (3x-1x12x197x64xf32) <- (-1x197x3x12x64xf32)
        transpose_7 = paddle._C_ops.transpose(reshape_4, [2, 0, 3, 1, 4])
        del reshape_4

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_25 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_26 = [1]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            transpose_7, [0], full_int_array_25, full_int_array_26, [1], [0]
        )
        del full_int_array_25, full_int_array_26

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_27 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_28 = [2]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            transpose_7, [0], full_int_array_27, full_int_array_28, [1], [0]
        )
        del full_int_array_27, full_int_array_28

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_29 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_30 = [3]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            transpose_7, [0], full_int_array_29, full_int_array_30, [1], [0]
        )
        del full_int_array_29, full_int_array_30, transpose_7

        # pd_op.transpose: (-1x12x64x197xf32) <- (-1x12x197x64xf32)
        transpose_8 = paddle._C_ops.transpose(slice_11, [0, 1, 3, 2])
        del slice_11

        # pd_op.matmul: (-1x12x197x197xf32) <- (-1x12x197x64xf32, -1x12x64x197xf32)
        matmul_13 = paddle._C_ops.matmul(slice_10, transpose_8, False, False)
        del slice_10, transpose_8

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x12x197x197xf32) <- (-1x12x197x197xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(matmul_13, full_5, float("0"), True)
        del full_5, matmul_13

        # pd_op.softmax: (-1x12x197x197xf32) <- (-1x12x197x197xf32)
        softmax_2 = paddle._C_ops.softmax(scale_2, -1)
        del scale_2

        # pd_op.matmul: (-1x12x197x64xf32) <- (-1x12x197x197xf32, -1x12x197x64xf32)
        matmul_14 = paddle._C_ops.matmul(softmax_2, slice_12, False, False)
        del slice_12, softmax_2

        # pd_op.transpose: (-1x197x12x64xf32) <- (-1x12x197x64xf32)
        transpose_9 = paddle._C_ops.transpose(matmul_14, [0, 2, 1, 3])
        del matmul_14

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_31 = [-1, 197, 768]

        # pd_op.reshape: (-1x197x768xf32) <- (-1x197x12x64xf32, 3xi64)
        reshape_5 = paddle._C_ops.reshape(transpose_9, full_int_array_31)
        del full_int_array_31, transpose_9

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x768xf32, 768x768xf32)
        matmul_15 = paddle._C_ops.matmul(reshape_5, parameter_119, False, False)
        del parameter_119, reshape_5

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_15 = paddle._C_ops.add(matmul_15, parameter_118)
        del matmul_15, parameter_118

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_16 = paddle._C_ops.add(add_13, add_15)
        del add_13, add_15

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_16, parameter_117, parameter_116, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_116, parameter_117

        # pd_op.matmul: (-1x197x3072xf32) <- (-1x197x768xf32, 768x3072xf32)
        matmul_16 = paddle._C_ops.matmul(layer_norm_18, parameter_115, False, False)
        del layer_norm_18, parameter_115

        # pd_op.add: (-1x197x3072xf32) <- (-1x197x3072xf32, 3072xf32)
        add_17 = paddle._C_ops.add(matmul_16, parameter_114)
        del matmul_16, parameter_114

        # pd_op.gelu: (-1x197x3072xf32) <- (-1x197x3072xf32)
        gelu_2 = paddle._C_ops.gelu(add_17, False)
        del add_17

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x3072xf32, 3072x768xf32)
        matmul_17 = paddle._C_ops.matmul(gelu_2, parameter_113, False, False)
        del gelu_2, parameter_113

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_18 = paddle._C_ops.add(matmul_17, parameter_112)
        del matmul_17, parameter_112

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_19 = paddle._C_ops.add(add_16, add_18)
        del add_16, add_18

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_19, parameter_111, parameter_110, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_110, parameter_111

        # pd_op.shape64: (3xi64) <- (-1x197x768xf32)
        shape64_4 = paddle._C_ops.shape64(layer_norm_21)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_32 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_33 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            shape64_4, [0], full_int_array_32, full_int_array_33, [1], [0]
        )
        del full_int_array_32, full_int_array_33, shape64_4

        # pd_op.matmul: (-1x197x2304xf32) <- (-1x197x768xf32, 768x2304xf32)
        matmul_18 = paddle._C_ops.matmul(layer_norm_21, parameter_109, False, False)
        del layer_norm_21, parameter_109

        # pd_op.add: (-1x197x2304xf32) <- (-1x197x2304xf32, 2304xf32)
        add_20 = paddle._C_ops.add(matmul_18, parameter_108)
        del matmul_18, parameter_108

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_34 = [-1, 197, 3, 12, 64]

        # pd_op.reshape: (-1x197x3x12x64xf32) <- (-1x197x2304xf32, 5xi64)
        reshape_6 = paddle._C_ops.reshape(add_20, full_int_array_34)
        del add_20, full_int_array_34

        # pd_op.transpose: (3x-1x12x197x64xf32) <- (-1x197x3x12x64xf32)
        transpose_10 = paddle._C_ops.transpose(reshape_6, [2, 0, 3, 1, 4])
        del reshape_6

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_35 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_36 = [1]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            transpose_10, [0], full_int_array_35, full_int_array_36, [1], [0]
        )
        del full_int_array_35, full_int_array_36

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_37 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_38 = [2]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            transpose_10, [0], full_int_array_37, full_int_array_38, [1], [0]
        )
        del full_int_array_37, full_int_array_38

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_39 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_40 = [3]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            transpose_10, [0], full_int_array_39, full_int_array_40, [1], [0]
        )
        del full_int_array_39, full_int_array_40, transpose_10

        # pd_op.transpose: (-1x12x64x197xf32) <- (-1x12x197x64xf32)
        transpose_11 = paddle._C_ops.transpose(slice_15, [0, 1, 3, 2])
        del slice_15

        # pd_op.matmul: (-1x12x197x197xf32) <- (-1x12x197x64xf32, -1x12x64x197xf32)
        matmul_19 = paddle._C_ops.matmul(slice_14, transpose_11, False, False)
        del slice_14, transpose_11

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x12x197x197xf32) <- (-1x12x197x197xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(matmul_19, full_6, float("0"), True)
        del full_6, matmul_19

        # pd_op.softmax: (-1x12x197x197xf32) <- (-1x12x197x197xf32)
        softmax_3 = paddle._C_ops.softmax(scale_3, -1)
        del scale_3

        # pd_op.matmul: (-1x12x197x64xf32) <- (-1x12x197x197xf32, -1x12x197x64xf32)
        matmul_20 = paddle._C_ops.matmul(softmax_3, slice_16, False, False)
        del slice_16, softmax_3

        # pd_op.transpose: (-1x197x12x64xf32) <- (-1x12x197x64xf32)
        transpose_12 = paddle._C_ops.transpose(matmul_20, [0, 2, 1, 3])
        del matmul_20

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_41 = [-1, 197, 768]

        # pd_op.reshape: (-1x197x768xf32) <- (-1x197x12x64xf32, 3xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_12, full_int_array_41)
        del full_int_array_41, transpose_12

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x768xf32, 768x768xf32)
        matmul_21 = paddle._C_ops.matmul(reshape_7, parameter_107, False, False)
        del parameter_107, reshape_7

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_21 = paddle._C_ops.add(matmul_21, parameter_106)
        del matmul_21, parameter_106

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_22 = paddle._C_ops.add(add_19, add_21)
        del add_19, add_21

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_22, parameter_105, parameter_104, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_104, parameter_105

        # pd_op.matmul: (-1x197x3072xf32) <- (-1x197x768xf32, 768x3072xf32)
        matmul_22 = paddle._C_ops.matmul(layer_norm_24, parameter_103, False, False)
        del layer_norm_24, parameter_103

        # pd_op.add: (-1x197x3072xf32) <- (-1x197x3072xf32, 3072xf32)
        add_23 = paddle._C_ops.add(matmul_22, parameter_102)
        del matmul_22, parameter_102

        # pd_op.gelu: (-1x197x3072xf32) <- (-1x197x3072xf32)
        gelu_3 = paddle._C_ops.gelu(add_23, False)
        del add_23

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x3072xf32, 3072x768xf32)
        matmul_23 = paddle._C_ops.matmul(gelu_3, parameter_101, False, False)
        del gelu_3, parameter_101

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_24 = paddle._C_ops.add(matmul_23, parameter_100)
        del matmul_23, parameter_100

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_25 = paddle._C_ops.add(add_22, add_24)
        del add_22, add_24

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_25, parameter_99, parameter_98, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_98, parameter_99

        # pd_op.shape64: (3xi64) <- (-1x197x768xf32)
        shape64_5 = paddle._C_ops.shape64(layer_norm_27)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_42 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_43 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            shape64_5, [0], full_int_array_42, full_int_array_43, [1], [0]
        )
        del full_int_array_42, full_int_array_43, shape64_5

        # pd_op.matmul: (-1x197x2304xf32) <- (-1x197x768xf32, 768x2304xf32)
        matmul_24 = paddle._C_ops.matmul(layer_norm_27, parameter_97, False, False)
        del layer_norm_27, parameter_97

        # pd_op.add: (-1x197x2304xf32) <- (-1x197x2304xf32, 2304xf32)
        add_26 = paddle._C_ops.add(matmul_24, parameter_96)
        del matmul_24, parameter_96

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_44 = [-1, 197, 3, 12, 64]

        # pd_op.reshape: (-1x197x3x12x64xf32) <- (-1x197x2304xf32, 5xi64)
        reshape_8 = paddle._C_ops.reshape(add_26, full_int_array_44)
        del add_26, full_int_array_44

        # pd_op.transpose: (3x-1x12x197x64xf32) <- (-1x197x3x12x64xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_8, [2, 0, 3, 1, 4])
        del reshape_8

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_45 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_46 = [1]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(
            transpose_13, [0], full_int_array_45, full_int_array_46, [1], [0]
        )
        del full_int_array_45, full_int_array_46

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_47 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_48 = [2]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(
            transpose_13, [0], full_int_array_47, full_int_array_48, [1], [0]
        )
        del full_int_array_47, full_int_array_48

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_49 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_50 = [3]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(
            transpose_13, [0], full_int_array_49, full_int_array_50, [1], [0]
        )
        del full_int_array_49, full_int_array_50, transpose_13

        # pd_op.transpose: (-1x12x64x197xf32) <- (-1x12x197x64xf32)
        transpose_14 = paddle._C_ops.transpose(slice_19, [0, 1, 3, 2])
        del slice_19

        # pd_op.matmul: (-1x12x197x197xf32) <- (-1x12x197x64xf32, -1x12x64x197xf32)
        matmul_25 = paddle._C_ops.matmul(slice_18, transpose_14, False, False)
        del slice_18, transpose_14

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x12x197x197xf32) <- (-1x12x197x197xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(matmul_25, full_7, float("0"), True)
        del full_7, matmul_25

        # pd_op.softmax: (-1x12x197x197xf32) <- (-1x12x197x197xf32)
        softmax_4 = paddle._C_ops.softmax(scale_4, -1)
        del scale_4

        # pd_op.matmul: (-1x12x197x64xf32) <- (-1x12x197x197xf32, -1x12x197x64xf32)
        matmul_26 = paddle._C_ops.matmul(softmax_4, slice_20, False, False)
        del slice_20, softmax_4

        # pd_op.transpose: (-1x197x12x64xf32) <- (-1x12x197x64xf32)
        transpose_15 = paddle._C_ops.transpose(matmul_26, [0, 2, 1, 3])
        del matmul_26

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_51 = [-1, 197, 768]

        # pd_op.reshape: (-1x197x768xf32) <- (-1x197x12x64xf32, 3xi64)
        reshape_9 = paddle._C_ops.reshape(transpose_15, full_int_array_51)
        del full_int_array_51, transpose_15

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x768xf32, 768x768xf32)
        matmul_27 = paddle._C_ops.matmul(reshape_9, parameter_95, False, False)
        del parameter_95, reshape_9

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_27 = paddle._C_ops.add(matmul_27, parameter_94)
        del matmul_27, parameter_94

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_28 = paddle._C_ops.add(add_25, add_27)
        del add_25, add_27

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_28, parameter_93, parameter_92, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_92, parameter_93

        # pd_op.matmul: (-1x197x3072xf32) <- (-1x197x768xf32, 768x3072xf32)
        matmul_28 = paddle._C_ops.matmul(layer_norm_30, parameter_91, False, False)
        del layer_norm_30, parameter_91

        # pd_op.add: (-1x197x3072xf32) <- (-1x197x3072xf32, 3072xf32)
        add_29 = paddle._C_ops.add(matmul_28, parameter_90)
        del matmul_28, parameter_90

        # pd_op.gelu: (-1x197x3072xf32) <- (-1x197x3072xf32)
        gelu_4 = paddle._C_ops.gelu(add_29, False)
        del add_29

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x3072xf32, 3072x768xf32)
        matmul_29 = paddle._C_ops.matmul(gelu_4, parameter_89, False, False)
        del gelu_4, parameter_89

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_30 = paddle._C_ops.add(matmul_29, parameter_88)
        del matmul_29, parameter_88

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_31 = paddle._C_ops.add(add_28, add_30)
        del add_28, add_30

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_31, parameter_87, parameter_86, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_86, parameter_87

        # pd_op.shape64: (3xi64) <- (-1x197x768xf32)
        shape64_6 = paddle._C_ops.shape64(layer_norm_33)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_52 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_53 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(
            shape64_6, [0], full_int_array_52, full_int_array_53, [1], [0]
        )
        del full_int_array_52, full_int_array_53, shape64_6

        # pd_op.matmul: (-1x197x2304xf32) <- (-1x197x768xf32, 768x2304xf32)
        matmul_30 = paddle._C_ops.matmul(layer_norm_33, parameter_85, False, False)
        del layer_norm_33, parameter_85

        # pd_op.add: (-1x197x2304xf32) <- (-1x197x2304xf32, 2304xf32)
        add_32 = paddle._C_ops.add(matmul_30, parameter_84)
        del matmul_30, parameter_84

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_54 = [-1, 197, 3, 12, 64]

        # pd_op.reshape: (-1x197x3x12x64xf32) <- (-1x197x2304xf32, 5xi64)
        reshape_10 = paddle._C_ops.reshape(add_32, full_int_array_54)
        del add_32, full_int_array_54

        # pd_op.transpose: (3x-1x12x197x64xf32) <- (-1x197x3x12x64xf32)
        transpose_16 = paddle._C_ops.transpose(reshape_10, [2, 0, 3, 1, 4])
        del reshape_10

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_55 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_56 = [1]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(
            transpose_16, [0], full_int_array_55, full_int_array_56, [1], [0]
        )
        del full_int_array_55, full_int_array_56

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_57 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_58 = [2]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(
            transpose_16, [0], full_int_array_57, full_int_array_58, [1], [0]
        )
        del full_int_array_57, full_int_array_58

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_59 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_60 = [3]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(
            transpose_16, [0], full_int_array_59, full_int_array_60, [1], [0]
        )
        del full_int_array_59, full_int_array_60, transpose_16

        # pd_op.transpose: (-1x12x64x197xf32) <- (-1x12x197x64xf32)
        transpose_17 = paddle._C_ops.transpose(slice_23, [0, 1, 3, 2])
        del slice_23

        # pd_op.matmul: (-1x12x197x197xf32) <- (-1x12x197x64xf32, -1x12x64x197xf32)
        matmul_31 = paddle._C_ops.matmul(slice_22, transpose_17, False, False)
        del slice_22, transpose_17

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x12x197x197xf32) <- (-1x12x197x197xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(matmul_31, full_8, float("0"), True)
        del full_8, matmul_31

        # pd_op.softmax: (-1x12x197x197xf32) <- (-1x12x197x197xf32)
        softmax_5 = paddle._C_ops.softmax(scale_5, -1)
        del scale_5

        # pd_op.matmul: (-1x12x197x64xf32) <- (-1x12x197x197xf32, -1x12x197x64xf32)
        matmul_32 = paddle._C_ops.matmul(softmax_5, slice_24, False, False)
        del slice_24, softmax_5

        # pd_op.transpose: (-1x197x12x64xf32) <- (-1x12x197x64xf32)
        transpose_18 = paddle._C_ops.transpose(matmul_32, [0, 2, 1, 3])
        del matmul_32

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_61 = [-1, 197, 768]

        # pd_op.reshape: (-1x197x768xf32) <- (-1x197x12x64xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_18, full_int_array_61)
        del full_int_array_61, transpose_18

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x768xf32, 768x768xf32)
        matmul_33 = paddle._C_ops.matmul(reshape_11, parameter_83, False, False)
        del parameter_83, reshape_11

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_33 = paddle._C_ops.add(matmul_33, parameter_82)
        del matmul_33, parameter_82

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_34 = paddle._C_ops.add(add_31, add_33)
        del add_31, add_33

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_34, parameter_81, parameter_80, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_80, parameter_81

        # pd_op.matmul: (-1x197x3072xf32) <- (-1x197x768xf32, 768x3072xf32)
        matmul_34 = paddle._C_ops.matmul(layer_norm_36, parameter_79, False, False)
        del layer_norm_36, parameter_79

        # pd_op.add: (-1x197x3072xf32) <- (-1x197x3072xf32, 3072xf32)
        add_35 = paddle._C_ops.add(matmul_34, parameter_78)
        del matmul_34, parameter_78

        # pd_op.gelu: (-1x197x3072xf32) <- (-1x197x3072xf32)
        gelu_5 = paddle._C_ops.gelu(add_35, False)
        del add_35

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x3072xf32, 3072x768xf32)
        matmul_35 = paddle._C_ops.matmul(gelu_5, parameter_77, False, False)
        del gelu_5, parameter_77

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_36 = paddle._C_ops.add(matmul_35, parameter_76)
        del matmul_35, parameter_76

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_37 = paddle._C_ops.add(add_34, add_36)
        del add_34, add_36

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_37, parameter_75, parameter_74, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_74, parameter_75

        # pd_op.shape64: (3xi64) <- (-1x197x768xf32)
        shape64_7 = paddle._C_ops.shape64(layer_norm_39)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_62 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_63 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(
            shape64_7, [0], full_int_array_62, full_int_array_63, [1], [0]
        )
        del full_int_array_62, full_int_array_63, shape64_7

        # pd_op.matmul: (-1x197x2304xf32) <- (-1x197x768xf32, 768x2304xf32)
        matmul_36 = paddle._C_ops.matmul(layer_norm_39, parameter_73, False, False)
        del layer_norm_39, parameter_73

        # pd_op.add: (-1x197x2304xf32) <- (-1x197x2304xf32, 2304xf32)
        add_38 = paddle._C_ops.add(matmul_36, parameter_72)
        del matmul_36, parameter_72

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_64 = [-1, 197, 3, 12, 64]

        # pd_op.reshape: (-1x197x3x12x64xf32) <- (-1x197x2304xf32, 5xi64)
        reshape_12 = paddle._C_ops.reshape(add_38, full_int_array_64)
        del add_38, full_int_array_64

        # pd_op.transpose: (3x-1x12x197x64xf32) <- (-1x197x3x12x64xf32)
        transpose_19 = paddle._C_ops.transpose(reshape_12, [2, 0, 3, 1, 4])
        del reshape_12

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_65 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_66 = [1]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(
            transpose_19, [0], full_int_array_65, full_int_array_66, [1], [0]
        )
        del full_int_array_65, full_int_array_66

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_67 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_68 = [2]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(
            transpose_19, [0], full_int_array_67, full_int_array_68, [1], [0]
        )
        del full_int_array_67, full_int_array_68

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_69 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_70 = [3]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(
            transpose_19, [0], full_int_array_69, full_int_array_70, [1], [0]
        )
        del full_int_array_69, full_int_array_70, transpose_19

        # pd_op.transpose: (-1x12x64x197xf32) <- (-1x12x197x64xf32)
        transpose_20 = paddle._C_ops.transpose(slice_27, [0, 1, 3, 2])
        del slice_27

        # pd_op.matmul: (-1x12x197x197xf32) <- (-1x12x197x64xf32, -1x12x64x197xf32)
        matmul_37 = paddle._C_ops.matmul(slice_26, transpose_20, False, False)
        del slice_26, transpose_20

        # pd_op.full: (1xf32) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x12x197x197xf32) <- (-1x12x197x197xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(matmul_37, full_9, float("0"), True)
        del full_9, matmul_37

        # pd_op.softmax: (-1x12x197x197xf32) <- (-1x12x197x197xf32)
        softmax_6 = paddle._C_ops.softmax(scale_6, -1)
        del scale_6

        # pd_op.matmul: (-1x12x197x64xf32) <- (-1x12x197x197xf32, -1x12x197x64xf32)
        matmul_38 = paddle._C_ops.matmul(softmax_6, slice_28, False, False)
        del slice_28, softmax_6

        # pd_op.transpose: (-1x197x12x64xf32) <- (-1x12x197x64xf32)
        transpose_21 = paddle._C_ops.transpose(matmul_38, [0, 2, 1, 3])
        del matmul_38

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_71 = [-1, 197, 768]

        # pd_op.reshape: (-1x197x768xf32) <- (-1x197x12x64xf32, 3xi64)
        reshape_13 = paddle._C_ops.reshape(transpose_21, full_int_array_71)
        del full_int_array_71, transpose_21

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x768xf32, 768x768xf32)
        matmul_39 = paddle._C_ops.matmul(reshape_13, parameter_71, False, False)
        del parameter_71, reshape_13

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_39 = paddle._C_ops.add(matmul_39, parameter_70)
        del matmul_39, parameter_70

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_40 = paddle._C_ops.add(add_37, add_39)
        del add_37, add_39

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_40, parameter_69, parameter_68, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_68, parameter_69

        # pd_op.matmul: (-1x197x3072xf32) <- (-1x197x768xf32, 768x3072xf32)
        matmul_40 = paddle._C_ops.matmul(layer_norm_42, parameter_67, False, False)
        del layer_norm_42, parameter_67

        # pd_op.add: (-1x197x3072xf32) <- (-1x197x3072xf32, 3072xf32)
        add_41 = paddle._C_ops.add(matmul_40, parameter_66)
        del matmul_40, parameter_66

        # pd_op.gelu: (-1x197x3072xf32) <- (-1x197x3072xf32)
        gelu_6 = paddle._C_ops.gelu(add_41, False)
        del add_41

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x3072xf32, 3072x768xf32)
        matmul_41 = paddle._C_ops.matmul(gelu_6, parameter_65, False, False)
        del gelu_6, parameter_65

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_42 = paddle._C_ops.add(matmul_41, parameter_64)
        del matmul_41, parameter_64

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_43 = paddle._C_ops.add(add_40, add_42)
        del add_40, add_42

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_43, parameter_63, parameter_62, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_62, parameter_63

        # pd_op.shape64: (3xi64) <- (-1x197x768xf32)
        shape64_8 = paddle._C_ops.shape64(layer_norm_45)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_72 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_73 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(
            shape64_8, [0], full_int_array_72, full_int_array_73, [1], [0]
        )
        del full_int_array_72, full_int_array_73, shape64_8

        # pd_op.matmul: (-1x197x2304xf32) <- (-1x197x768xf32, 768x2304xf32)
        matmul_42 = paddle._C_ops.matmul(layer_norm_45, parameter_61, False, False)
        del layer_norm_45, parameter_61

        # pd_op.add: (-1x197x2304xf32) <- (-1x197x2304xf32, 2304xf32)
        add_44 = paddle._C_ops.add(matmul_42, parameter_60)
        del matmul_42, parameter_60

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_74 = [-1, 197, 3, 12, 64]

        # pd_op.reshape: (-1x197x3x12x64xf32) <- (-1x197x2304xf32, 5xi64)
        reshape_14 = paddle._C_ops.reshape(add_44, full_int_array_74)
        del add_44, full_int_array_74

        # pd_op.transpose: (3x-1x12x197x64xf32) <- (-1x197x3x12x64xf32)
        transpose_22 = paddle._C_ops.transpose(reshape_14, [2, 0, 3, 1, 4])
        del reshape_14

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_75 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_76 = [1]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(
            transpose_22, [0], full_int_array_75, full_int_array_76, [1], [0]
        )
        del full_int_array_75, full_int_array_76

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_77 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_78 = [2]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(
            transpose_22, [0], full_int_array_77, full_int_array_78, [1], [0]
        )
        del full_int_array_77, full_int_array_78

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_79 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_80 = [3]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(
            transpose_22, [0], full_int_array_79, full_int_array_80, [1], [0]
        )
        del full_int_array_79, full_int_array_80, transpose_22

        # pd_op.transpose: (-1x12x64x197xf32) <- (-1x12x197x64xf32)
        transpose_23 = paddle._C_ops.transpose(slice_31, [0, 1, 3, 2])
        del slice_31

        # pd_op.matmul: (-1x12x197x197xf32) <- (-1x12x197x64xf32, -1x12x64x197xf32)
        matmul_43 = paddle._C_ops.matmul(slice_30, transpose_23, False, False)
        del slice_30, transpose_23

        # pd_op.full: (1xf32) <- ()
        full_10 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x12x197x197xf32) <- (-1x12x197x197xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(matmul_43, full_10, float("0"), True)
        del full_10, matmul_43

        # pd_op.softmax: (-1x12x197x197xf32) <- (-1x12x197x197xf32)
        softmax_7 = paddle._C_ops.softmax(scale_7, -1)
        del scale_7

        # pd_op.matmul: (-1x12x197x64xf32) <- (-1x12x197x197xf32, -1x12x197x64xf32)
        matmul_44 = paddle._C_ops.matmul(softmax_7, slice_32, False, False)
        del slice_32, softmax_7

        # pd_op.transpose: (-1x197x12x64xf32) <- (-1x12x197x64xf32)
        transpose_24 = paddle._C_ops.transpose(matmul_44, [0, 2, 1, 3])
        del matmul_44

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_81 = [-1, 197, 768]

        # pd_op.reshape: (-1x197x768xf32) <- (-1x197x12x64xf32, 3xi64)
        reshape_15 = paddle._C_ops.reshape(transpose_24, full_int_array_81)
        del full_int_array_81, transpose_24

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x768xf32, 768x768xf32)
        matmul_45 = paddle._C_ops.matmul(reshape_15, parameter_59, False, False)
        del parameter_59, reshape_15

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_45 = paddle._C_ops.add(matmul_45, parameter_58)
        del matmul_45, parameter_58

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_46 = paddle._C_ops.add(add_43, add_45)
        del add_43, add_45

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_46, parameter_57, parameter_56, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_56, parameter_57

        # pd_op.matmul: (-1x197x3072xf32) <- (-1x197x768xf32, 768x3072xf32)
        matmul_46 = paddle._C_ops.matmul(layer_norm_48, parameter_55, False, False)
        del layer_norm_48, parameter_55

        # pd_op.add: (-1x197x3072xf32) <- (-1x197x3072xf32, 3072xf32)
        add_47 = paddle._C_ops.add(matmul_46, parameter_54)
        del matmul_46, parameter_54

        # pd_op.gelu: (-1x197x3072xf32) <- (-1x197x3072xf32)
        gelu_7 = paddle._C_ops.gelu(add_47, False)
        del add_47

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x3072xf32, 3072x768xf32)
        matmul_47 = paddle._C_ops.matmul(gelu_7, parameter_53, False, False)
        del gelu_7, parameter_53

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_48 = paddle._C_ops.add(matmul_47, parameter_52)
        del matmul_47, parameter_52

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_49 = paddle._C_ops.add(add_46, add_48)
        del add_46, add_48

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_49, parameter_51, parameter_50, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_50, parameter_51

        # pd_op.shape64: (3xi64) <- (-1x197x768xf32)
        shape64_9 = paddle._C_ops.shape64(layer_norm_51)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_82 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_83 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_33 = paddle._C_ops.slice(
            shape64_9, [0], full_int_array_82, full_int_array_83, [1], [0]
        )
        del full_int_array_82, full_int_array_83, shape64_9

        # pd_op.matmul: (-1x197x2304xf32) <- (-1x197x768xf32, 768x2304xf32)
        matmul_48 = paddle._C_ops.matmul(layer_norm_51, parameter_49, False, False)
        del layer_norm_51, parameter_49

        # pd_op.add: (-1x197x2304xf32) <- (-1x197x2304xf32, 2304xf32)
        add_50 = paddle._C_ops.add(matmul_48, parameter_48)
        del matmul_48, parameter_48

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_84 = [-1, 197, 3, 12, 64]

        # pd_op.reshape: (-1x197x3x12x64xf32) <- (-1x197x2304xf32, 5xi64)
        reshape_16 = paddle._C_ops.reshape(add_50, full_int_array_84)
        del add_50, full_int_array_84

        # pd_op.transpose: (3x-1x12x197x64xf32) <- (-1x197x3x12x64xf32)
        transpose_25 = paddle._C_ops.transpose(reshape_16, [2, 0, 3, 1, 4])
        del reshape_16

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_85 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_86 = [1]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_34 = paddle._C_ops.slice(
            transpose_25, [0], full_int_array_85, full_int_array_86, [1], [0]
        )
        del full_int_array_85, full_int_array_86

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_87 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_88 = [2]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_35 = paddle._C_ops.slice(
            transpose_25, [0], full_int_array_87, full_int_array_88, [1], [0]
        )
        del full_int_array_87, full_int_array_88

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_89 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_90 = [3]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(
            transpose_25, [0], full_int_array_89, full_int_array_90, [1], [0]
        )
        del full_int_array_89, full_int_array_90, transpose_25

        # pd_op.transpose: (-1x12x64x197xf32) <- (-1x12x197x64xf32)
        transpose_26 = paddle._C_ops.transpose(slice_35, [0, 1, 3, 2])
        del slice_35

        # pd_op.matmul: (-1x12x197x197xf32) <- (-1x12x197x64xf32, -1x12x64x197xf32)
        matmul_49 = paddle._C_ops.matmul(slice_34, transpose_26, False, False)
        del slice_34, transpose_26

        # pd_op.full: (1xf32) <- ()
        full_11 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x12x197x197xf32) <- (-1x12x197x197xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(matmul_49, full_11, float("0"), True)
        del full_11, matmul_49

        # pd_op.softmax: (-1x12x197x197xf32) <- (-1x12x197x197xf32)
        softmax_8 = paddle._C_ops.softmax(scale_8, -1)
        del scale_8

        # pd_op.matmul: (-1x12x197x64xf32) <- (-1x12x197x197xf32, -1x12x197x64xf32)
        matmul_50 = paddle._C_ops.matmul(softmax_8, slice_36, False, False)
        del slice_36, softmax_8

        # pd_op.transpose: (-1x197x12x64xf32) <- (-1x12x197x64xf32)
        transpose_27 = paddle._C_ops.transpose(matmul_50, [0, 2, 1, 3])
        del matmul_50

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_91 = [-1, 197, 768]

        # pd_op.reshape: (-1x197x768xf32) <- (-1x197x12x64xf32, 3xi64)
        reshape_17 = paddle._C_ops.reshape(transpose_27, full_int_array_91)
        del full_int_array_91, transpose_27

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x768xf32, 768x768xf32)
        matmul_51 = paddle._C_ops.matmul(reshape_17, parameter_47, False, False)
        del parameter_47, reshape_17

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_51 = paddle._C_ops.add(matmul_51, parameter_46)
        del matmul_51, parameter_46

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_52 = paddle._C_ops.add(add_49, add_51)
        del add_49, add_51

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_52, parameter_45, parameter_44, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_44, parameter_45

        # pd_op.matmul: (-1x197x3072xf32) <- (-1x197x768xf32, 768x3072xf32)
        matmul_52 = paddle._C_ops.matmul(layer_norm_54, parameter_43, False, False)
        del layer_norm_54, parameter_43

        # pd_op.add: (-1x197x3072xf32) <- (-1x197x3072xf32, 3072xf32)
        add_53 = paddle._C_ops.add(matmul_52, parameter_42)
        del matmul_52, parameter_42

        # pd_op.gelu: (-1x197x3072xf32) <- (-1x197x3072xf32)
        gelu_8 = paddle._C_ops.gelu(add_53, False)
        del add_53

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x3072xf32, 3072x768xf32)
        matmul_53 = paddle._C_ops.matmul(gelu_8, parameter_41, False, False)
        del gelu_8, parameter_41

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_54 = paddle._C_ops.add(matmul_53, parameter_40)
        del matmul_53, parameter_40

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_55 = paddle._C_ops.add(add_52, add_54)
        del add_52, add_54

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_55, parameter_39, parameter_38, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_38, parameter_39

        # pd_op.shape64: (3xi64) <- (-1x197x768xf32)
        shape64_10 = paddle._C_ops.shape64(layer_norm_57)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_92 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_93 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_37 = paddle._C_ops.slice(
            shape64_10, [0], full_int_array_92, full_int_array_93, [1], [0]
        )
        del full_int_array_92, full_int_array_93, shape64_10

        # pd_op.matmul: (-1x197x2304xf32) <- (-1x197x768xf32, 768x2304xf32)
        matmul_54 = paddle._C_ops.matmul(layer_norm_57, parameter_37, False, False)
        del layer_norm_57, parameter_37

        # pd_op.add: (-1x197x2304xf32) <- (-1x197x2304xf32, 2304xf32)
        add_56 = paddle._C_ops.add(matmul_54, parameter_36)
        del matmul_54, parameter_36

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_94 = [-1, 197, 3, 12, 64]

        # pd_op.reshape: (-1x197x3x12x64xf32) <- (-1x197x2304xf32, 5xi64)
        reshape_18 = paddle._C_ops.reshape(add_56, full_int_array_94)
        del add_56, full_int_array_94

        # pd_op.transpose: (3x-1x12x197x64xf32) <- (-1x197x3x12x64xf32)
        transpose_28 = paddle._C_ops.transpose(reshape_18, [2, 0, 3, 1, 4])
        del reshape_18

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_95 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_96 = [1]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_38 = paddle._C_ops.slice(
            transpose_28, [0], full_int_array_95, full_int_array_96, [1], [0]
        )
        del full_int_array_95, full_int_array_96

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_97 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_98 = [2]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_39 = paddle._C_ops.slice(
            transpose_28, [0], full_int_array_97, full_int_array_98, [1], [0]
        )
        del full_int_array_97, full_int_array_98

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_99 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_100 = [3]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_40 = paddle._C_ops.slice(
            transpose_28, [0], full_int_array_99, full_int_array_100, [1], [0]
        )
        del full_int_array_100, full_int_array_99, transpose_28

        # pd_op.transpose: (-1x12x64x197xf32) <- (-1x12x197x64xf32)
        transpose_29 = paddle._C_ops.transpose(slice_39, [0, 1, 3, 2])
        del slice_39

        # pd_op.matmul: (-1x12x197x197xf32) <- (-1x12x197x64xf32, -1x12x64x197xf32)
        matmul_55 = paddle._C_ops.matmul(slice_38, transpose_29, False, False)
        del slice_38, transpose_29

        # pd_op.full: (1xf32) <- ()
        full_12 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x12x197x197xf32) <- (-1x12x197x197xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(matmul_55, full_12, float("0"), True)
        del full_12, matmul_55

        # pd_op.softmax: (-1x12x197x197xf32) <- (-1x12x197x197xf32)
        softmax_9 = paddle._C_ops.softmax(scale_9, -1)
        del scale_9

        # pd_op.matmul: (-1x12x197x64xf32) <- (-1x12x197x197xf32, -1x12x197x64xf32)
        matmul_56 = paddle._C_ops.matmul(softmax_9, slice_40, False, False)
        del slice_40, softmax_9

        # pd_op.transpose: (-1x197x12x64xf32) <- (-1x12x197x64xf32)
        transpose_30 = paddle._C_ops.transpose(matmul_56, [0, 2, 1, 3])
        del matmul_56

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_101 = [-1, 197, 768]

        # pd_op.reshape: (-1x197x768xf32) <- (-1x197x12x64xf32, 3xi64)
        reshape_19 = paddle._C_ops.reshape(transpose_30, full_int_array_101)
        del full_int_array_101, transpose_30

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x768xf32, 768x768xf32)
        matmul_57 = paddle._C_ops.matmul(reshape_19, parameter_35, False, False)
        del parameter_35, reshape_19

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_57 = paddle._C_ops.add(matmul_57, parameter_34)
        del matmul_57, parameter_34

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_58 = paddle._C_ops.add(add_55, add_57)
        del add_55, add_57

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_58, parameter_33, parameter_32, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_32, parameter_33

        # pd_op.matmul: (-1x197x3072xf32) <- (-1x197x768xf32, 768x3072xf32)
        matmul_58 = paddle._C_ops.matmul(layer_norm_60, parameter_31, False, False)
        del layer_norm_60, parameter_31

        # pd_op.add: (-1x197x3072xf32) <- (-1x197x3072xf32, 3072xf32)
        add_59 = paddle._C_ops.add(matmul_58, parameter_30)
        del matmul_58, parameter_30

        # pd_op.gelu: (-1x197x3072xf32) <- (-1x197x3072xf32)
        gelu_9 = paddle._C_ops.gelu(add_59, False)
        del add_59

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x3072xf32, 3072x768xf32)
        matmul_59 = paddle._C_ops.matmul(gelu_9, parameter_29, False, False)
        del gelu_9, parameter_29

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_60 = paddle._C_ops.add(matmul_59, parameter_28)
        del matmul_59, parameter_28

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_61 = paddle._C_ops.add(add_58, add_60)
        del add_58, add_60

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_61, parameter_27, parameter_26, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_26, parameter_27

        # pd_op.shape64: (3xi64) <- (-1x197x768xf32)
        shape64_11 = paddle._C_ops.shape64(layer_norm_63)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_102 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_103 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_41 = paddle._C_ops.slice(
            shape64_11, [0], full_int_array_102, full_int_array_103, [1], [0]
        )
        del full_int_array_102, full_int_array_103, shape64_11

        # pd_op.matmul: (-1x197x2304xf32) <- (-1x197x768xf32, 768x2304xf32)
        matmul_60 = paddle._C_ops.matmul(layer_norm_63, parameter_25, False, False)
        del layer_norm_63, parameter_25

        # pd_op.add: (-1x197x2304xf32) <- (-1x197x2304xf32, 2304xf32)
        add_62 = paddle._C_ops.add(matmul_60, parameter_24)
        del matmul_60, parameter_24

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_104 = [-1, 197, 3, 12, 64]

        # pd_op.reshape: (-1x197x3x12x64xf32) <- (-1x197x2304xf32, 5xi64)
        reshape_20 = paddle._C_ops.reshape(add_62, full_int_array_104)
        del add_62, full_int_array_104

        # pd_op.transpose: (3x-1x12x197x64xf32) <- (-1x197x3x12x64xf32)
        transpose_31 = paddle._C_ops.transpose(reshape_20, [2, 0, 3, 1, 4])
        del reshape_20

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_105 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_106 = [1]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_42 = paddle._C_ops.slice(
            transpose_31, [0], full_int_array_105, full_int_array_106, [1], [0]
        )
        del full_int_array_105, full_int_array_106

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_107 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_108 = [2]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_43 = paddle._C_ops.slice(
            transpose_31, [0], full_int_array_107, full_int_array_108, [1], [0]
        )
        del full_int_array_107, full_int_array_108

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_109 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_110 = [3]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_44 = paddle._C_ops.slice(
            transpose_31, [0], full_int_array_109, full_int_array_110, [1], [0]
        )
        del full_int_array_109, full_int_array_110, transpose_31

        # pd_op.transpose: (-1x12x64x197xf32) <- (-1x12x197x64xf32)
        transpose_32 = paddle._C_ops.transpose(slice_43, [0, 1, 3, 2])
        del slice_43

        # pd_op.matmul: (-1x12x197x197xf32) <- (-1x12x197x64xf32, -1x12x64x197xf32)
        matmul_61 = paddle._C_ops.matmul(slice_42, transpose_32, False, False)
        del slice_42, transpose_32

        # pd_op.full: (1xf32) <- ()
        full_13 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x12x197x197xf32) <- (-1x12x197x197xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(matmul_61, full_13, float("0"), True)
        del full_13, matmul_61

        # pd_op.softmax: (-1x12x197x197xf32) <- (-1x12x197x197xf32)
        softmax_10 = paddle._C_ops.softmax(scale_10, -1)
        del scale_10

        # pd_op.matmul: (-1x12x197x64xf32) <- (-1x12x197x197xf32, -1x12x197x64xf32)
        matmul_62 = paddle._C_ops.matmul(softmax_10, slice_44, False, False)
        del slice_44, softmax_10

        # pd_op.transpose: (-1x197x12x64xf32) <- (-1x12x197x64xf32)
        transpose_33 = paddle._C_ops.transpose(matmul_62, [0, 2, 1, 3])
        del matmul_62

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_111 = [-1, 197, 768]

        # pd_op.reshape: (-1x197x768xf32) <- (-1x197x12x64xf32, 3xi64)
        reshape_21 = paddle._C_ops.reshape(transpose_33, full_int_array_111)
        del full_int_array_111, transpose_33

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x768xf32, 768x768xf32)
        matmul_63 = paddle._C_ops.matmul(reshape_21, parameter_23, False, False)
        del parameter_23, reshape_21

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_63 = paddle._C_ops.add(matmul_63, parameter_22)
        del matmul_63, parameter_22

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_64 = paddle._C_ops.add(add_61, add_63)
        del add_61, add_63

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_64, parameter_21, parameter_20, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_20, parameter_21

        # pd_op.matmul: (-1x197x3072xf32) <- (-1x197x768xf32, 768x3072xf32)
        matmul_64 = paddle._C_ops.matmul(layer_norm_66, parameter_19, False, False)
        del layer_norm_66, parameter_19

        # pd_op.add: (-1x197x3072xf32) <- (-1x197x3072xf32, 3072xf32)
        add_65 = paddle._C_ops.add(matmul_64, parameter_18)
        del matmul_64, parameter_18

        # pd_op.gelu: (-1x197x3072xf32) <- (-1x197x3072xf32)
        gelu_10 = paddle._C_ops.gelu(add_65, False)
        del add_65

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x3072xf32, 3072x768xf32)
        matmul_65 = paddle._C_ops.matmul(gelu_10, parameter_17, False, False)
        del gelu_10, parameter_17

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_66 = paddle._C_ops.add(matmul_65, parameter_16)
        del matmul_65, parameter_16

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_67 = paddle._C_ops.add(add_64, add_66)
        del add_64, add_66

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_67, parameter_15, parameter_14, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_14, parameter_15

        # pd_op.shape64: (3xi64) <- (-1x197x768xf32)
        shape64_12 = paddle._C_ops.shape64(layer_norm_69)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_112 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_113 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_45 = paddle._C_ops.slice(
            shape64_12, [0], full_int_array_112, full_int_array_113, [1], [0]
        )
        del full_int_array_112, full_int_array_113, shape64_12

        # pd_op.matmul: (-1x197x2304xf32) <- (-1x197x768xf32, 768x2304xf32)
        matmul_66 = paddle._C_ops.matmul(layer_norm_69, parameter_13, False, False)
        del layer_norm_69, parameter_13

        # pd_op.add: (-1x197x2304xf32) <- (-1x197x2304xf32, 2304xf32)
        add_68 = paddle._C_ops.add(matmul_66, parameter_12)
        del matmul_66, parameter_12

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_114 = [-1, 197, 3, 12, 64]

        # pd_op.reshape: (-1x197x3x12x64xf32) <- (-1x197x2304xf32, 5xi64)
        reshape_22 = paddle._C_ops.reshape(add_68, full_int_array_114)
        del add_68, full_int_array_114

        # pd_op.transpose: (3x-1x12x197x64xf32) <- (-1x197x3x12x64xf32)
        transpose_34 = paddle._C_ops.transpose(reshape_22, [2, 0, 3, 1, 4])
        del reshape_22

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_115 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_116 = [1]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_46 = paddle._C_ops.slice(
            transpose_34, [0], full_int_array_115, full_int_array_116, [1], [0]
        )
        del full_int_array_115, full_int_array_116

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_117 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_118 = [2]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_47 = paddle._C_ops.slice(
            transpose_34, [0], full_int_array_117, full_int_array_118, [1], [0]
        )
        del full_int_array_117, full_int_array_118

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_119 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_120 = [3]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_48 = paddle._C_ops.slice(
            transpose_34, [0], full_int_array_119, full_int_array_120, [1], [0]
        )
        del full_int_array_119, full_int_array_120, transpose_34

        # pd_op.transpose: (-1x12x64x197xf32) <- (-1x12x197x64xf32)
        transpose_35 = paddle._C_ops.transpose(slice_47, [0, 1, 3, 2])
        del slice_47

        # pd_op.matmul: (-1x12x197x197xf32) <- (-1x12x197x64xf32, -1x12x64x197xf32)
        matmul_67 = paddle._C_ops.matmul(slice_46, transpose_35, False, False)
        del slice_46, transpose_35

        # pd_op.full: (1xf32) <- ()
        full_14 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x12x197x197xf32) <- (-1x12x197x197xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(matmul_67, full_14, float("0"), True)
        del full_14, matmul_67

        # pd_op.softmax: (-1x12x197x197xf32) <- (-1x12x197x197xf32)
        softmax_11 = paddle._C_ops.softmax(scale_11, -1)
        del scale_11

        # pd_op.matmul: (-1x12x197x64xf32) <- (-1x12x197x197xf32, -1x12x197x64xf32)
        matmul_68 = paddle._C_ops.matmul(softmax_11, slice_48, False, False)
        del slice_48, softmax_11

        # pd_op.transpose: (-1x197x12x64xf32) <- (-1x12x197x64xf32)
        transpose_36 = paddle._C_ops.transpose(matmul_68, [0, 2, 1, 3])
        del matmul_68

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_121 = [-1, 197, 768]

        # pd_op.reshape: (-1x197x768xf32) <- (-1x197x12x64xf32, 3xi64)
        reshape_23 = paddle._C_ops.reshape(transpose_36, full_int_array_121)
        del full_int_array_121, transpose_36

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x768xf32, 768x768xf32)
        matmul_69 = paddle._C_ops.matmul(reshape_23, parameter_11, False, False)
        del parameter_11, reshape_23

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_69 = paddle._C_ops.add(matmul_69, parameter_10)
        del matmul_69, parameter_10

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_70 = paddle._C_ops.add(add_67, add_69)
        del add_67, add_69

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_72, layer_norm_73, layer_norm_74 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_70, parameter_9, parameter_8, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_8, parameter_9

        # pd_op.matmul: (-1x197x3072xf32) <- (-1x197x768xf32, 768x3072xf32)
        matmul_70 = paddle._C_ops.matmul(layer_norm_72, parameter_7, False, False)
        del layer_norm_72, parameter_7

        # pd_op.add: (-1x197x3072xf32) <- (-1x197x3072xf32, 3072xf32)
        add_71 = paddle._C_ops.add(matmul_70, parameter_6)
        del matmul_70, parameter_6

        # pd_op.gelu: (-1x197x3072xf32) <- (-1x197x3072xf32)
        gelu_11 = paddle._C_ops.gelu(add_71, False)
        del add_71

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x3072xf32, 3072x768xf32)
        matmul_71 = paddle._C_ops.matmul(gelu_11, parameter_5, False, False)
        del gelu_11, parameter_5

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_72 = paddle._C_ops.add(matmul_71, parameter_4)
        del matmul_71, parameter_4

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_73 = paddle._C_ops.add(add_70, add_72)
        del add_70, add_72

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_122 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_123 = [2147483647]

        # pd_op.slice: (-1x196x768xf32) <- (-1x197x768xf32, 1xi64, 1xi64)
        slice_49 = paddle._C_ops.slice(
            add_73, [1], full_int_array_122, full_int_array_123, [1], []
        )
        del add_73, full_int_array_122, full_int_array_123

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_124 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_125 = [1]

        # pd_op.slice: (-1x768xf32) <- (-1x196x768xf32, 1xi64, 1xi64)
        slice_50 = paddle._C_ops.slice(
            slice_49, [1], full_int_array_124, full_int_array_125, [1], [1]
        )
        del full_int_array_124, full_int_array_125

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_126 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_127 = [2147483647]

        # pd_op.slice: (-1x195x768xf32) <- (-1x196x768xf32, 1xi64, 1xi64)
        slice_51 = paddle._C_ops.slice(
            slice_49, [1], full_int_array_126, full_int_array_127, [1], []
        )
        del full_int_array_126, full_int_array_127, slice_49

        # pd_op.layer_norm: (-1x768xf32, -1xf32, -1xf32) <- (-1x768xf32, 768xf32, 768xf32)
        layer_norm_75, layer_norm_76, layer_norm_77 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                slice_50, parameter_3, parameter_2, float("1e-05"), 1
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_2, parameter_3, slice_50

        # pd_op.matmul: (-1x102xf32) <- (-1x768xf32, 768x102xf32)
        matmul_72 = paddle._C_ops.matmul(layer_norm_75, parameter_1, False, False)
        del layer_norm_75, parameter_1

        # pd_op.add: (-1x102xf32) <- (-1x102xf32, 102xf32)
        add_0 = paddle._C_ops.add(matmul_72, parameter_0)
        del matmul_72, parameter_0

        return add_0
