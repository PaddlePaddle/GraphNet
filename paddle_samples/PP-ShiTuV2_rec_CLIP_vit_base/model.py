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
        data_4,
    ):
        # pd_op.shape64: (4xi64) <- (-1x3x224x224xf32)
        shape64_0 = paddle._C_ops.shape64(data_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.conv2d: (-1x768x14x14xf32) <- (-1x3x224x224xf32, 768x3x16x16xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_2, parameter_150, [16, 16], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.shape64: (4xi64) <- (-1x768x14x14xf32)
        shape64_1 = paddle._C_ops.shape64(conv2d_0)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.flatten: (-1x768x196xf32) <- (-1x768x14x14xf32)
        flatten_0 = paddle._C_ops.flatten(conv2d_0, 2, 3)

        # pd_op.transpose: (-1x196x768xf32) <- (-1x768x196xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_0, [0, 2, 1])

        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("-1"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_0 = [slice_0, full_0, full_0]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)

        # pd_op.expand: (-1x1x768xf32) <- (1x1x768xf32, 3xi64)
        expand_0 = paddle._C_ops.expand(data_3, stack_0)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x1x768xf32, -1x196x768xf32]) <- (-1x1x768xf32, -1x196x768xf32)
        combine_1 = [expand_0, transpose_0]

        # pd_op.concat: (-1x197x768xf32) <- ([-1x1x768xf32, -1x196x768xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_1, full_1)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 1x197x768xf32)
        add_1 = paddle._C_ops.add(concat_0, data_4)

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_1, parameter_149, parameter_148, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                layer_norm_0, parameter_147, parameter_146, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.shape64: (3xi64) <- (-1x197x768xf32)
        shape64_2 = paddle._C_ops.shape64(layer_norm_3)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.matmul: (-1x197x2304xf32) <- (-1x197x768xf32, 768x2304xf32)
        matmul_1 = paddle._C_ops.matmul(layer_norm_3, parameter_145, False, False)

        # pd_op.add: (-1x197x2304xf32) <- (-1x197x2304xf32, 2304xf32)
        add_2 = paddle._C_ops.add(matmul_1, parameter_144)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_2 = [-1, 197, 3, 12, 64]

        # pd_op.reshape: (-1x197x3x12x64xf32) <- (-1x197x2304xf32, 5xi64)
        reshape_0 = paddle._C_ops.reshape(add_2, full_int_array_2)

        # pd_op.transpose: (3x-1x12x197x64xf32) <- (-1x197x3x12x64xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_0, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            transpose_1, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [2]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            transpose_1, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [3]

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            transpose_1, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.transpose: (-1x12x64x197xf32) <- (-1x12x197x64xf32)
        transpose_2 = paddle._C_ops.transpose(slice_4, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x197x197xf32) <- (-1x12x197x64xf32, -1x12x64x197xf32)
        matmul_2 = paddle._C_ops.matmul(slice_3, transpose_2, False, False)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x12x197x197xf32) <- (-1x12x197x197xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(matmul_2, full_2, float("0"), True)

        # pd_op.softmax: (-1x12x197x197xf32) <- (-1x12x197x197xf32)
        softmax_0 = paddle._C_ops.softmax(scale_0, -1)

        # pd_op.matmul: (-1x12x197x64xf32) <- (-1x12x197x197xf32, -1x12x197x64xf32)
        matmul_3 = paddle._C_ops.matmul(softmax_0, slice_5, False, False)

        # pd_op.transpose: (-1x197x12x64xf32) <- (-1x12x197x64xf32)
        transpose_3 = paddle._C_ops.transpose(matmul_3, [0, 2, 1, 3])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_5 = [-1, 197, 768]

        # pd_op.reshape: (-1x197x768xf32) <- (-1x197x12x64xf32, 3xi64)
        reshape_1 = paddle._C_ops.reshape(transpose_3, full_int_array_5)

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x768xf32, 768x768xf32)
        matmul_4 = paddle._C_ops.matmul(reshape_1, parameter_143, False, False)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_3 = paddle._C_ops.add(matmul_4, parameter_142)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_4 = paddle._C_ops.add(layer_norm_0, add_3)

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_4, parameter_141, parameter_140, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x197x3072xf32) <- (-1x197x768xf32, 768x3072xf32)
        matmul_5 = paddle._C_ops.matmul(layer_norm_6, parameter_139, False, False)

        # pd_op.add: (-1x197x3072xf32) <- (-1x197x3072xf32, 3072xf32)
        add_5 = paddle._C_ops.add(matmul_5, parameter_138)

        # pd_op.gelu: (-1x197x3072xf32) <- (-1x197x3072xf32)
        gelu_0 = paddle._C_ops.gelu(add_5, False)

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x3072xf32, 3072x768xf32)
        matmul_6 = paddle._C_ops.matmul(gelu_0, parameter_137, False, False)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_6 = paddle._C_ops.add(matmul_6, parameter_136)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_7 = paddle._C_ops.add(add_4, add_6)

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_7, parameter_135, parameter_134, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.shape64: (3xi64) <- (-1x197x768xf32)
        shape64_3 = paddle._C_ops.shape64(layer_norm_9)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.matmul: (-1x197x2304xf32) <- (-1x197x768xf32, 768x2304xf32)
        matmul_7 = paddle._C_ops.matmul(layer_norm_9, parameter_133, False, False)

        # pd_op.add: (-1x197x2304xf32) <- (-1x197x2304xf32, 2304xf32)
        add_8 = paddle._C_ops.add(matmul_7, parameter_132)

        # pd_op.reshape: (-1x197x3x12x64xf32) <- (-1x197x2304xf32, 5xi64)
        reshape_2 = paddle._C_ops.reshape(add_8, full_int_array_2)

        # pd_op.transpose: (3x-1x12x197x64xf32) <- (-1x197x3x12x64xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_2, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            transpose_4, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            transpose_4, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            transpose_4, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.transpose: (-1x12x64x197xf32) <- (-1x12x197x64xf32)
        transpose_5 = paddle._C_ops.transpose(slice_8, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x197x197xf32) <- (-1x12x197x64xf32, -1x12x64x197xf32)
        matmul_8 = paddle._C_ops.matmul(slice_7, transpose_5, False, False)

        # pd_op.scale: (-1x12x197x197xf32) <- (-1x12x197x197xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(matmul_8, full_2, float("0"), True)

        # pd_op.softmax: (-1x12x197x197xf32) <- (-1x12x197x197xf32)
        softmax_1 = paddle._C_ops.softmax(scale_1, -1)

        # pd_op.matmul: (-1x12x197x64xf32) <- (-1x12x197x197xf32, -1x12x197x64xf32)
        matmul_9 = paddle._C_ops.matmul(softmax_1, slice_9, False, False)

        # pd_op.transpose: (-1x197x12x64xf32) <- (-1x12x197x64xf32)
        transpose_6 = paddle._C_ops.transpose(matmul_9, [0, 2, 1, 3])

        # pd_op.reshape: (-1x197x768xf32) <- (-1x197x12x64xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_6, full_int_array_5)

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x768xf32, 768x768xf32)
        matmul_10 = paddle._C_ops.matmul(reshape_3, parameter_131, False, False)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_9 = paddle._C_ops.add(matmul_10, parameter_130)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_10 = paddle._C_ops.add(add_7, add_9)

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_10, parameter_129, parameter_128, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x197x3072xf32) <- (-1x197x768xf32, 768x3072xf32)
        matmul_11 = paddle._C_ops.matmul(layer_norm_12, parameter_127, False, False)

        # pd_op.add: (-1x197x3072xf32) <- (-1x197x3072xf32, 3072xf32)
        add_11 = paddle._C_ops.add(matmul_11, parameter_126)

        # pd_op.gelu: (-1x197x3072xf32) <- (-1x197x3072xf32)
        gelu_1 = paddle._C_ops.gelu(add_11, False)

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x3072xf32, 3072x768xf32)
        matmul_12 = paddle._C_ops.matmul(gelu_1, parameter_125, False, False)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_12 = paddle._C_ops.add(matmul_12, parameter_124)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_13 = paddle._C_ops.add(add_10, add_12)

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_13, parameter_123, parameter_122, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.shape64: (3xi64) <- (-1x197x768xf32)
        shape64_4 = paddle._C_ops.shape64(layer_norm_15)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            shape64_4, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.matmul: (-1x197x2304xf32) <- (-1x197x768xf32, 768x2304xf32)
        matmul_13 = paddle._C_ops.matmul(layer_norm_15, parameter_121, False, False)

        # pd_op.add: (-1x197x2304xf32) <- (-1x197x2304xf32, 2304xf32)
        add_14 = paddle._C_ops.add(matmul_13, parameter_120)

        # pd_op.reshape: (-1x197x3x12x64xf32) <- (-1x197x2304xf32, 5xi64)
        reshape_4 = paddle._C_ops.reshape(add_14, full_int_array_2)

        # pd_op.transpose: (3x-1x12x197x64xf32) <- (-1x197x3x12x64xf32)
        transpose_7 = paddle._C_ops.transpose(reshape_4, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            transpose_7, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            transpose_7, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            transpose_7, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.transpose: (-1x12x64x197xf32) <- (-1x12x197x64xf32)
        transpose_8 = paddle._C_ops.transpose(slice_12, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x197x197xf32) <- (-1x12x197x64xf32, -1x12x64x197xf32)
        matmul_14 = paddle._C_ops.matmul(slice_11, transpose_8, False, False)

        # pd_op.scale: (-1x12x197x197xf32) <- (-1x12x197x197xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(matmul_14, full_2, float("0"), True)

        # pd_op.softmax: (-1x12x197x197xf32) <- (-1x12x197x197xf32)
        softmax_2 = paddle._C_ops.softmax(scale_2, -1)

        # pd_op.matmul: (-1x12x197x64xf32) <- (-1x12x197x197xf32, -1x12x197x64xf32)
        matmul_15 = paddle._C_ops.matmul(softmax_2, slice_13, False, False)

        # pd_op.transpose: (-1x197x12x64xf32) <- (-1x12x197x64xf32)
        transpose_9 = paddle._C_ops.transpose(matmul_15, [0, 2, 1, 3])

        # pd_op.reshape: (-1x197x768xf32) <- (-1x197x12x64xf32, 3xi64)
        reshape_5 = paddle._C_ops.reshape(transpose_9, full_int_array_5)

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x768xf32, 768x768xf32)
        matmul_16 = paddle._C_ops.matmul(reshape_5, parameter_119, False, False)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_15 = paddle._C_ops.add(matmul_16, parameter_118)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_16 = paddle._C_ops.add(add_13, add_15)

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_16, parameter_117, parameter_116, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x197x3072xf32) <- (-1x197x768xf32, 768x3072xf32)
        matmul_17 = paddle._C_ops.matmul(layer_norm_18, parameter_115, False, False)

        # pd_op.add: (-1x197x3072xf32) <- (-1x197x3072xf32, 3072xf32)
        add_17 = paddle._C_ops.add(matmul_17, parameter_114)

        # pd_op.gelu: (-1x197x3072xf32) <- (-1x197x3072xf32)
        gelu_2 = paddle._C_ops.gelu(add_17, False)

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x3072xf32, 3072x768xf32)
        matmul_18 = paddle._C_ops.matmul(gelu_2, parameter_113, False, False)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_18 = paddle._C_ops.add(matmul_18, parameter_112)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_19 = paddle._C_ops.add(add_16, add_18)

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_19, parameter_111, parameter_110, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.shape64: (3xi64) <- (-1x197x768xf32)
        shape64_5 = paddle._C_ops.shape64(layer_norm_21)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            shape64_5, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.matmul: (-1x197x2304xf32) <- (-1x197x768xf32, 768x2304xf32)
        matmul_19 = paddle._C_ops.matmul(layer_norm_21, parameter_109, False, False)

        # pd_op.add: (-1x197x2304xf32) <- (-1x197x2304xf32, 2304xf32)
        add_20 = paddle._C_ops.add(matmul_19, parameter_108)

        # pd_op.reshape: (-1x197x3x12x64xf32) <- (-1x197x2304xf32, 5xi64)
        reshape_6 = paddle._C_ops.reshape(add_20, full_int_array_2)

        # pd_op.transpose: (3x-1x12x197x64xf32) <- (-1x197x3x12x64xf32)
        transpose_10 = paddle._C_ops.transpose(reshape_6, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            transpose_10, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            transpose_10, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            transpose_10, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.transpose: (-1x12x64x197xf32) <- (-1x12x197x64xf32)
        transpose_11 = paddle._C_ops.transpose(slice_16, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x197x197xf32) <- (-1x12x197x64xf32, -1x12x64x197xf32)
        matmul_20 = paddle._C_ops.matmul(slice_15, transpose_11, False, False)

        # pd_op.scale: (-1x12x197x197xf32) <- (-1x12x197x197xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(matmul_20, full_2, float("0"), True)

        # pd_op.softmax: (-1x12x197x197xf32) <- (-1x12x197x197xf32)
        softmax_3 = paddle._C_ops.softmax(scale_3, -1)

        # pd_op.matmul: (-1x12x197x64xf32) <- (-1x12x197x197xf32, -1x12x197x64xf32)
        matmul_21 = paddle._C_ops.matmul(softmax_3, slice_17, False, False)

        # pd_op.transpose: (-1x197x12x64xf32) <- (-1x12x197x64xf32)
        transpose_12 = paddle._C_ops.transpose(matmul_21, [0, 2, 1, 3])

        # pd_op.reshape: (-1x197x768xf32) <- (-1x197x12x64xf32, 3xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_12, full_int_array_5)

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x768xf32, 768x768xf32)
        matmul_22 = paddle._C_ops.matmul(reshape_7, parameter_107, False, False)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_21 = paddle._C_ops.add(matmul_22, parameter_106)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_22 = paddle._C_ops.add(add_19, add_21)

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_22, parameter_105, parameter_104, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x197x3072xf32) <- (-1x197x768xf32, 768x3072xf32)
        matmul_23 = paddle._C_ops.matmul(layer_norm_24, parameter_103, False, False)

        # pd_op.add: (-1x197x3072xf32) <- (-1x197x3072xf32, 3072xf32)
        add_23 = paddle._C_ops.add(matmul_23, parameter_102)

        # pd_op.gelu: (-1x197x3072xf32) <- (-1x197x3072xf32)
        gelu_3 = paddle._C_ops.gelu(add_23, False)

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x3072xf32, 3072x768xf32)
        matmul_24 = paddle._C_ops.matmul(gelu_3, parameter_101, False, False)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_24 = paddle._C_ops.add(matmul_24, parameter_100)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_25 = paddle._C_ops.add(add_22, add_24)

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_25, parameter_99, parameter_98, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.shape64: (3xi64) <- (-1x197x768xf32)
        shape64_6 = paddle._C_ops.shape64(layer_norm_27)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(
            shape64_6, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.matmul: (-1x197x2304xf32) <- (-1x197x768xf32, 768x2304xf32)
        matmul_25 = paddle._C_ops.matmul(layer_norm_27, parameter_97, False, False)

        # pd_op.add: (-1x197x2304xf32) <- (-1x197x2304xf32, 2304xf32)
        add_26 = paddle._C_ops.add(matmul_25, parameter_96)

        # pd_op.reshape: (-1x197x3x12x64xf32) <- (-1x197x2304xf32, 5xi64)
        reshape_8 = paddle._C_ops.reshape(add_26, full_int_array_2)

        # pd_op.transpose: (3x-1x12x197x64xf32) <- (-1x197x3x12x64xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_8, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(
            transpose_13, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(
            transpose_13, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(
            transpose_13, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.transpose: (-1x12x64x197xf32) <- (-1x12x197x64xf32)
        transpose_14 = paddle._C_ops.transpose(slice_20, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x197x197xf32) <- (-1x12x197x64xf32, -1x12x64x197xf32)
        matmul_26 = paddle._C_ops.matmul(slice_19, transpose_14, False, False)

        # pd_op.scale: (-1x12x197x197xf32) <- (-1x12x197x197xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(matmul_26, full_2, float("0"), True)

        # pd_op.softmax: (-1x12x197x197xf32) <- (-1x12x197x197xf32)
        softmax_4 = paddle._C_ops.softmax(scale_4, -1)

        # pd_op.matmul: (-1x12x197x64xf32) <- (-1x12x197x197xf32, -1x12x197x64xf32)
        matmul_27 = paddle._C_ops.matmul(softmax_4, slice_21, False, False)

        # pd_op.transpose: (-1x197x12x64xf32) <- (-1x12x197x64xf32)
        transpose_15 = paddle._C_ops.transpose(matmul_27, [0, 2, 1, 3])

        # pd_op.reshape: (-1x197x768xf32) <- (-1x197x12x64xf32, 3xi64)
        reshape_9 = paddle._C_ops.reshape(transpose_15, full_int_array_5)

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x768xf32, 768x768xf32)
        matmul_28 = paddle._C_ops.matmul(reshape_9, parameter_95, False, False)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_27 = paddle._C_ops.add(matmul_28, parameter_94)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_28 = paddle._C_ops.add(add_25, add_27)

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_28, parameter_93, parameter_92, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x197x3072xf32) <- (-1x197x768xf32, 768x3072xf32)
        matmul_29 = paddle._C_ops.matmul(layer_norm_30, parameter_91, False, False)

        # pd_op.add: (-1x197x3072xf32) <- (-1x197x3072xf32, 3072xf32)
        add_29 = paddle._C_ops.add(matmul_29, parameter_90)

        # pd_op.gelu: (-1x197x3072xf32) <- (-1x197x3072xf32)
        gelu_4 = paddle._C_ops.gelu(add_29, False)

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x3072xf32, 3072x768xf32)
        matmul_30 = paddle._C_ops.matmul(gelu_4, parameter_89, False, False)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_30 = paddle._C_ops.add(matmul_30, parameter_88)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_31 = paddle._C_ops.add(add_28, add_30)

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_31, parameter_87, parameter_86, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.shape64: (3xi64) <- (-1x197x768xf32)
        shape64_7 = paddle._C_ops.shape64(layer_norm_33)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(
            shape64_7, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.matmul: (-1x197x2304xf32) <- (-1x197x768xf32, 768x2304xf32)
        matmul_31 = paddle._C_ops.matmul(layer_norm_33, parameter_85, False, False)

        # pd_op.add: (-1x197x2304xf32) <- (-1x197x2304xf32, 2304xf32)
        add_32 = paddle._C_ops.add(matmul_31, parameter_84)

        # pd_op.reshape: (-1x197x3x12x64xf32) <- (-1x197x2304xf32, 5xi64)
        reshape_10 = paddle._C_ops.reshape(add_32, full_int_array_2)

        # pd_op.transpose: (3x-1x12x197x64xf32) <- (-1x197x3x12x64xf32)
        transpose_16 = paddle._C_ops.transpose(reshape_10, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(
            transpose_16, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(
            transpose_16, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(
            transpose_16, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.transpose: (-1x12x64x197xf32) <- (-1x12x197x64xf32)
        transpose_17 = paddle._C_ops.transpose(slice_24, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x197x197xf32) <- (-1x12x197x64xf32, -1x12x64x197xf32)
        matmul_32 = paddle._C_ops.matmul(slice_23, transpose_17, False, False)

        # pd_op.scale: (-1x12x197x197xf32) <- (-1x12x197x197xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(matmul_32, full_2, float("0"), True)

        # pd_op.softmax: (-1x12x197x197xf32) <- (-1x12x197x197xf32)
        softmax_5 = paddle._C_ops.softmax(scale_5, -1)

        # pd_op.matmul: (-1x12x197x64xf32) <- (-1x12x197x197xf32, -1x12x197x64xf32)
        matmul_33 = paddle._C_ops.matmul(softmax_5, slice_25, False, False)

        # pd_op.transpose: (-1x197x12x64xf32) <- (-1x12x197x64xf32)
        transpose_18 = paddle._C_ops.transpose(matmul_33, [0, 2, 1, 3])

        # pd_op.reshape: (-1x197x768xf32) <- (-1x197x12x64xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_18, full_int_array_5)

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x768xf32, 768x768xf32)
        matmul_34 = paddle._C_ops.matmul(reshape_11, parameter_83, False, False)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_33 = paddle._C_ops.add(matmul_34, parameter_82)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_34 = paddle._C_ops.add(add_31, add_33)

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_34, parameter_81, parameter_80, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x197x3072xf32) <- (-1x197x768xf32, 768x3072xf32)
        matmul_35 = paddle._C_ops.matmul(layer_norm_36, parameter_79, False, False)

        # pd_op.add: (-1x197x3072xf32) <- (-1x197x3072xf32, 3072xf32)
        add_35 = paddle._C_ops.add(matmul_35, parameter_78)

        # pd_op.gelu: (-1x197x3072xf32) <- (-1x197x3072xf32)
        gelu_5 = paddle._C_ops.gelu(add_35, False)

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x3072xf32, 3072x768xf32)
        matmul_36 = paddle._C_ops.matmul(gelu_5, parameter_77, False, False)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_36 = paddle._C_ops.add(matmul_36, parameter_76)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_37 = paddle._C_ops.add(add_34, add_36)

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_37, parameter_75, parameter_74, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.shape64: (3xi64) <- (-1x197x768xf32)
        shape64_8 = paddle._C_ops.shape64(layer_norm_39)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(
            shape64_8, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.matmul: (-1x197x2304xf32) <- (-1x197x768xf32, 768x2304xf32)
        matmul_37 = paddle._C_ops.matmul(layer_norm_39, parameter_73, False, False)

        # pd_op.add: (-1x197x2304xf32) <- (-1x197x2304xf32, 2304xf32)
        add_38 = paddle._C_ops.add(matmul_37, parameter_72)

        # pd_op.reshape: (-1x197x3x12x64xf32) <- (-1x197x2304xf32, 5xi64)
        reshape_12 = paddle._C_ops.reshape(add_38, full_int_array_2)

        # pd_op.transpose: (3x-1x12x197x64xf32) <- (-1x197x3x12x64xf32)
        transpose_19 = paddle._C_ops.transpose(reshape_12, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(
            transpose_19, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(
            transpose_19, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(
            transpose_19, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.transpose: (-1x12x64x197xf32) <- (-1x12x197x64xf32)
        transpose_20 = paddle._C_ops.transpose(slice_28, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x197x197xf32) <- (-1x12x197x64xf32, -1x12x64x197xf32)
        matmul_38 = paddle._C_ops.matmul(slice_27, transpose_20, False, False)

        # pd_op.scale: (-1x12x197x197xf32) <- (-1x12x197x197xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(matmul_38, full_2, float("0"), True)

        # pd_op.softmax: (-1x12x197x197xf32) <- (-1x12x197x197xf32)
        softmax_6 = paddle._C_ops.softmax(scale_6, -1)

        # pd_op.matmul: (-1x12x197x64xf32) <- (-1x12x197x197xf32, -1x12x197x64xf32)
        matmul_39 = paddle._C_ops.matmul(softmax_6, slice_29, False, False)

        # pd_op.transpose: (-1x197x12x64xf32) <- (-1x12x197x64xf32)
        transpose_21 = paddle._C_ops.transpose(matmul_39, [0, 2, 1, 3])

        # pd_op.reshape: (-1x197x768xf32) <- (-1x197x12x64xf32, 3xi64)
        reshape_13 = paddle._C_ops.reshape(transpose_21, full_int_array_5)

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x768xf32, 768x768xf32)
        matmul_40 = paddle._C_ops.matmul(reshape_13, parameter_71, False, False)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_39 = paddle._C_ops.add(matmul_40, parameter_70)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_40 = paddle._C_ops.add(add_37, add_39)

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_40, parameter_69, parameter_68, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x197x3072xf32) <- (-1x197x768xf32, 768x3072xf32)
        matmul_41 = paddle._C_ops.matmul(layer_norm_42, parameter_67, False, False)

        # pd_op.add: (-1x197x3072xf32) <- (-1x197x3072xf32, 3072xf32)
        add_41 = paddle._C_ops.add(matmul_41, parameter_66)

        # pd_op.gelu: (-1x197x3072xf32) <- (-1x197x3072xf32)
        gelu_6 = paddle._C_ops.gelu(add_41, False)

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x3072xf32, 3072x768xf32)
        matmul_42 = paddle._C_ops.matmul(gelu_6, parameter_65, False, False)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_42 = paddle._C_ops.add(matmul_42, parameter_64)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_43 = paddle._C_ops.add(add_40, add_42)

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_43, parameter_63, parameter_62, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.shape64: (3xi64) <- (-1x197x768xf32)
        shape64_9 = paddle._C_ops.shape64(layer_norm_45)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(
            shape64_9, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.matmul: (-1x197x2304xf32) <- (-1x197x768xf32, 768x2304xf32)
        matmul_43 = paddle._C_ops.matmul(layer_norm_45, parameter_61, False, False)

        # pd_op.add: (-1x197x2304xf32) <- (-1x197x2304xf32, 2304xf32)
        add_44 = paddle._C_ops.add(matmul_43, parameter_60)

        # pd_op.reshape: (-1x197x3x12x64xf32) <- (-1x197x2304xf32, 5xi64)
        reshape_14 = paddle._C_ops.reshape(add_44, full_int_array_2)

        # pd_op.transpose: (3x-1x12x197x64xf32) <- (-1x197x3x12x64xf32)
        transpose_22 = paddle._C_ops.transpose(reshape_14, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(
            transpose_22, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(
            transpose_22, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_33 = paddle._C_ops.slice(
            transpose_22, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.transpose: (-1x12x64x197xf32) <- (-1x12x197x64xf32)
        transpose_23 = paddle._C_ops.transpose(slice_32, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x197x197xf32) <- (-1x12x197x64xf32, -1x12x64x197xf32)
        matmul_44 = paddle._C_ops.matmul(slice_31, transpose_23, False, False)

        # pd_op.scale: (-1x12x197x197xf32) <- (-1x12x197x197xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(matmul_44, full_2, float("0"), True)

        # pd_op.softmax: (-1x12x197x197xf32) <- (-1x12x197x197xf32)
        softmax_7 = paddle._C_ops.softmax(scale_7, -1)

        # pd_op.matmul: (-1x12x197x64xf32) <- (-1x12x197x197xf32, -1x12x197x64xf32)
        matmul_45 = paddle._C_ops.matmul(softmax_7, slice_33, False, False)

        # pd_op.transpose: (-1x197x12x64xf32) <- (-1x12x197x64xf32)
        transpose_24 = paddle._C_ops.transpose(matmul_45, [0, 2, 1, 3])

        # pd_op.reshape: (-1x197x768xf32) <- (-1x197x12x64xf32, 3xi64)
        reshape_15 = paddle._C_ops.reshape(transpose_24, full_int_array_5)

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x768xf32, 768x768xf32)
        matmul_46 = paddle._C_ops.matmul(reshape_15, parameter_59, False, False)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_45 = paddle._C_ops.add(matmul_46, parameter_58)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_46 = paddle._C_ops.add(add_43, add_45)

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_46, parameter_57, parameter_56, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x197x3072xf32) <- (-1x197x768xf32, 768x3072xf32)
        matmul_47 = paddle._C_ops.matmul(layer_norm_48, parameter_55, False, False)

        # pd_op.add: (-1x197x3072xf32) <- (-1x197x3072xf32, 3072xf32)
        add_47 = paddle._C_ops.add(matmul_47, parameter_54)

        # pd_op.gelu: (-1x197x3072xf32) <- (-1x197x3072xf32)
        gelu_7 = paddle._C_ops.gelu(add_47, False)

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x3072xf32, 3072x768xf32)
        matmul_48 = paddle._C_ops.matmul(gelu_7, parameter_53, False, False)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_48 = paddle._C_ops.add(matmul_48, parameter_52)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_49 = paddle._C_ops.add(add_46, add_48)

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_49, parameter_51, parameter_50, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.shape64: (3xi64) <- (-1x197x768xf32)
        shape64_10 = paddle._C_ops.shape64(layer_norm_51)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_34 = paddle._C_ops.slice(
            shape64_10, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.matmul: (-1x197x2304xf32) <- (-1x197x768xf32, 768x2304xf32)
        matmul_49 = paddle._C_ops.matmul(layer_norm_51, parameter_49, False, False)

        # pd_op.add: (-1x197x2304xf32) <- (-1x197x2304xf32, 2304xf32)
        add_50 = paddle._C_ops.add(matmul_49, parameter_48)

        # pd_op.reshape: (-1x197x3x12x64xf32) <- (-1x197x2304xf32, 5xi64)
        reshape_16 = paddle._C_ops.reshape(add_50, full_int_array_2)

        # pd_op.transpose: (3x-1x12x197x64xf32) <- (-1x197x3x12x64xf32)
        transpose_25 = paddle._C_ops.transpose(reshape_16, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_35 = paddle._C_ops.slice(
            transpose_25, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(
            transpose_25, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_37 = paddle._C_ops.slice(
            transpose_25, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.transpose: (-1x12x64x197xf32) <- (-1x12x197x64xf32)
        transpose_26 = paddle._C_ops.transpose(slice_36, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x197x197xf32) <- (-1x12x197x64xf32, -1x12x64x197xf32)
        matmul_50 = paddle._C_ops.matmul(slice_35, transpose_26, False, False)

        # pd_op.scale: (-1x12x197x197xf32) <- (-1x12x197x197xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(matmul_50, full_2, float("0"), True)

        # pd_op.softmax: (-1x12x197x197xf32) <- (-1x12x197x197xf32)
        softmax_8 = paddle._C_ops.softmax(scale_8, -1)

        # pd_op.matmul: (-1x12x197x64xf32) <- (-1x12x197x197xf32, -1x12x197x64xf32)
        matmul_51 = paddle._C_ops.matmul(softmax_8, slice_37, False, False)

        # pd_op.transpose: (-1x197x12x64xf32) <- (-1x12x197x64xf32)
        transpose_27 = paddle._C_ops.transpose(matmul_51, [0, 2, 1, 3])

        # pd_op.reshape: (-1x197x768xf32) <- (-1x197x12x64xf32, 3xi64)
        reshape_17 = paddle._C_ops.reshape(transpose_27, full_int_array_5)

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x768xf32, 768x768xf32)
        matmul_52 = paddle._C_ops.matmul(reshape_17, parameter_47, False, False)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_51 = paddle._C_ops.add(matmul_52, parameter_46)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_52 = paddle._C_ops.add(add_49, add_51)

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_52, parameter_45, parameter_44, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x197x3072xf32) <- (-1x197x768xf32, 768x3072xf32)
        matmul_53 = paddle._C_ops.matmul(layer_norm_54, parameter_43, False, False)

        # pd_op.add: (-1x197x3072xf32) <- (-1x197x3072xf32, 3072xf32)
        add_53 = paddle._C_ops.add(matmul_53, parameter_42)

        # pd_op.gelu: (-1x197x3072xf32) <- (-1x197x3072xf32)
        gelu_8 = paddle._C_ops.gelu(add_53, False)

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x3072xf32, 3072x768xf32)
        matmul_54 = paddle._C_ops.matmul(gelu_8, parameter_41, False, False)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_54 = paddle._C_ops.add(matmul_54, parameter_40)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_55 = paddle._C_ops.add(add_52, add_54)

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_55, parameter_39, parameter_38, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.shape64: (3xi64) <- (-1x197x768xf32)
        shape64_11 = paddle._C_ops.shape64(layer_norm_57)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_38 = paddle._C_ops.slice(
            shape64_11, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.matmul: (-1x197x2304xf32) <- (-1x197x768xf32, 768x2304xf32)
        matmul_55 = paddle._C_ops.matmul(layer_norm_57, parameter_37, False, False)

        # pd_op.add: (-1x197x2304xf32) <- (-1x197x2304xf32, 2304xf32)
        add_56 = paddle._C_ops.add(matmul_55, parameter_36)

        # pd_op.reshape: (-1x197x3x12x64xf32) <- (-1x197x2304xf32, 5xi64)
        reshape_18 = paddle._C_ops.reshape(add_56, full_int_array_2)

        # pd_op.transpose: (3x-1x12x197x64xf32) <- (-1x197x3x12x64xf32)
        transpose_28 = paddle._C_ops.transpose(reshape_18, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_39 = paddle._C_ops.slice(
            transpose_28, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_40 = paddle._C_ops.slice(
            transpose_28, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_41 = paddle._C_ops.slice(
            transpose_28, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.transpose: (-1x12x64x197xf32) <- (-1x12x197x64xf32)
        transpose_29 = paddle._C_ops.transpose(slice_40, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x197x197xf32) <- (-1x12x197x64xf32, -1x12x64x197xf32)
        matmul_56 = paddle._C_ops.matmul(slice_39, transpose_29, False, False)

        # pd_op.scale: (-1x12x197x197xf32) <- (-1x12x197x197xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(matmul_56, full_2, float("0"), True)

        # pd_op.softmax: (-1x12x197x197xf32) <- (-1x12x197x197xf32)
        softmax_9 = paddle._C_ops.softmax(scale_9, -1)

        # pd_op.matmul: (-1x12x197x64xf32) <- (-1x12x197x197xf32, -1x12x197x64xf32)
        matmul_57 = paddle._C_ops.matmul(softmax_9, slice_41, False, False)

        # pd_op.transpose: (-1x197x12x64xf32) <- (-1x12x197x64xf32)
        transpose_30 = paddle._C_ops.transpose(matmul_57, [0, 2, 1, 3])

        # pd_op.reshape: (-1x197x768xf32) <- (-1x197x12x64xf32, 3xi64)
        reshape_19 = paddle._C_ops.reshape(transpose_30, full_int_array_5)

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x768xf32, 768x768xf32)
        matmul_58 = paddle._C_ops.matmul(reshape_19, parameter_35, False, False)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_57 = paddle._C_ops.add(matmul_58, parameter_34)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_58 = paddle._C_ops.add(add_55, add_57)

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_58, parameter_33, parameter_32, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x197x3072xf32) <- (-1x197x768xf32, 768x3072xf32)
        matmul_59 = paddle._C_ops.matmul(layer_norm_60, parameter_31, False, False)

        # pd_op.add: (-1x197x3072xf32) <- (-1x197x3072xf32, 3072xf32)
        add_59 = paddle._C_ops.add(matmul_59, parameter_30)

        # pd_op.gelu: (-1x197x3072xf32) <- (-1x197x3072xf32)
        gelu_9 = paddle._C_ops.gelu(add_59, False)

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x3072xf32, 3072x768xf32)
        matmul_60 = paddle._C_ops.matmul(gelu_9, parameter_29, False, False)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_60 = paddle._C_ops.add(matmul_60, parameter_28)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_61 = paddle._C_ops.add(add_58, add_60)

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_61, parameter_27, parameter_26, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.shape64: (3xi64) <- (-1x197x768xf32)
        shape64_12 = paddle._C_ops.shape64(layer_norm_63)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_42 = paddle._C_ops.slice(
            shape64_12, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.matmul: (-1x197x2304xf32) <- (-1x197x768xf32, 768x2304xf32)
        matmul_61 = paddle._C_ops.matmul(layer_norm_63, parameter_25, False, False)

        # pd_op.add: (-1x197x2304xf32) <- (-1x197x2304xf32, 2304xf32)
        add_62 = paddle._C_ops.add(matmul_61, parameter_24)

        # pd_op.reshape: (-1x197x3x12x64xf32) <- (-1x197x2304xf32, 5xi64)
        reshape_20 = paddle._C_ops.reshape(add_62, full_int_array_2)

        # pd_op.transpose: (3x-1x12x197x64xf32) <- (-1x197x3x12x64xf32)
        transpose_31 = paddle._C_ops.transpose(reshape_20, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_43 = paddle._C_ops.slice(
            transpose_31, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_44 = paddle._C_ops.slice(
            transpose_31, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_45 = paddle._C_ops.slice(
            transpose_31, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.transpose: (-1x12x64x197xf32) <- (-1x12x197x64xf32)
        transpose_32 = paddle._C_ops.transpose(slice_44, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x197x197xf32) <- (-1x12x197x64xf32, -1x12x64x197xf32)
        matmul_62 = paddle._C_ops.matmul(slice_43, transpose_32, False, False)

        # pd_op.scale: (-1x12x197x197xf32) <- (-1x12x197x197xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(matmul_62, full_2, float("0"), True)

        # pd_op.softmax: (-1x12x197x197xf32) <- (-1x12x197x197xf32)
        softmax_10 = paddle._C_ops.softmax(scale_10, -1)

        # pd_op.matmul: (-1x12x197x64xf32) <- (-1x12x197x197xf32, -1x12x197x64xf32)
        matmul_63 = paddle._C_ops.matmul(softmax_10, slice_45, False, False)

        # pd_op.transpose: (-1x197x12x64xf32) <- (-1x12x197x64xf32)
        transpose_33 = paddle._C_ops.transpose(matmul_63, [0, 2, 1, 3])

        # pd_op.reshape: (-1x197x768xf32) <- (-1x197x12x64xf32, 3xi64)
        reshape_21 = paddle._C_ops.reshape(transpose_33, full_int_array_5)

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x768xf32, 768x768xf32)
        matmul_64 = paddle._C_ops.matmul(reshape_21, parameter_23, False, False)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_63 = paddle._C_ops.add(matmul_64, parameter_22)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_64 = paddle._C_ops.add(add_61, add_63)

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_64, parameter_21, parameter_20, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x197x3072xf32) <- (-1x197x768xf32, 768x3072xf32)
        matmul_65 = paddle._C_ops.matmul(layer_norm_66, parameter_19, False, False)

        # pd_op.add: (-1x197x3072xf32) <- (-1x197x3072xf32, 3072xf32)
        add_65 = paddle._C_ops.add(matmul_65, parameter_18)

        # pd_op.gelu: (-1x197x3072xf32) <- (-1x197x3072xf32)
        gelu_10 = paddle._C_ops.gelu(add_65, False)

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x3072xf32, 3072x768xf32)
        matmul_66 = paddle._C_ops.matmul(gelu_10, parameter_17, False, False)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_66 = paddle._C_ops.add(matmul_66, parameter_16)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_67 = paddle._C_ops.add(add_64, add_66)

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_67, parameter_15, parameter_14, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.shape64: (3xi64) <- (-1x197x768xf32)
        shape64_13 = paddle._C_ops.shape64(layer_norm_69)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_46 = paddle._C_ops.slice(
            shape64_13, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.matmul: (-1x197x2304xf32) <- (-1x197x768xf32, 768x2304xf32)
        matmul_67 = paddle._C_ops.matmul(layer_norm_69, parameter_13, False, False)

        # pd_op.add: (-1x197x2304xf32) <- (-1x197x2304xf32, 2304xf32)
        add_68 = paddle._C_ops.add(matmul_67, parameter_12)

        # pd_op.reshape: (-1x197x3x12x64xf32) <- (-1x197x2304xf32, 5xi64)
        reshape_22 = paddle._C_ops.reshape(add_68, full_int_array_2)

        # pd_op.transpose: (3x-1x12x197x64xf32) <- (-1x197x3x12x64xf32)
        transpose_34 = paddle._C_ops.transpose(reshape_22, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_47 = paddle._C_ops.slice(
            transpose_34, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_48 = paddle._C_ops.slice(
            transpose_34, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x12x197x64xf32) <- (3x-1x12x197x64xf32, 1xi64, 1xi64)
        slice_49 = paddle._C_ops.slice(
            transpose_34, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.transpose: (-1x12x64x197xf32) <- (-1x12x197x64xf32)
        transpose_35 = paddle._C_ops.transpose(slice_48, [0, 1, 3, 2])

        # pd_op.matmul: (-1x12x197x197xf32) <- (-1x12x197x64xf32, -1x12x64x197xf32)
        matmul_68 = paddle._C_ops.matmul(slice_47, transpose_35, False, False)

        # pd_op.scale: (-1x12x197x197xf32) <- (-1x12x197x197xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(matmul_68, full_2, float("0"), True)

        # pd_op.softmax: (-1x12x197x197xf32) <- (-1x12x197x197xf32)
        softmax_11 = paddle._C_ops.softmax(scale_11, -1)

        # pd_op.matmul: (-1x12x197x64xf32) <- (-1x12x197x197xf32, -1x12x197x64xf32)
        matmul_69 = paddle._C_ops.matmul(softmax_11, slice_49, False, False)

        # pd_op.transpose: (-1x197x12x64xf32) <- (-1x12x197x64xf32)
        transpose_36 = paddle._C_ops.transpose(matmul_69, [0, 2, 1, 3])

        # pd_op.reshape: (-1x197x768xf32) <- (-1x197x12x64xf32, 3xi64)
        reshape_23 = paddle._C_ops.reshape(transpose_36, full_int_array_5)

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x768xf32, 768x768xf32)
        matmul_70 = paddle._C_ops.matmul(reshape_23, parameter_11, False, False)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_69 = paddle._C_ops.add(matmul_70, parameter_10)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_70 = paddle._C_ops.add(add_67, add_69)

        # pd_op.layer_norm: (-1x197x768xf32, -1x197xf32, -1x197xf32) <- (-1x197x768xf32, 768xf32, 768xf32)
        layer_norm_72, layer_norm_73, layer_norm_74 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_70, parameter_9, parameter_8, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x197x3072xf32) <- (-1x197x768xf32, 768x3072xf32)
        matmul_71 = paddle._C_ops.matmul(layer_norm_72, parameter_7, False, False)

        # pd_op.add: (-1x197x3072xf32) <- (-1x197x3072xf32, 3072xf32)
        add_71 = paddle._C_ops.add(matmul_71, parameter_6)

        # pd_op.gelu: (-1x197x3072xf32) <- (-1x197x3072xf32)
        gelu_11 = paddle._C_ops.gelu(add_71, False)

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x3072xf32, 3072x768xf32)
        matmul_72 = paddle._C_ops.matmul(gelu_11, parameter_5, False, False)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, 768xf32)
        add_72 = paddle._C_ops.add(matmul_72, parameter_4)

        # pd_op.add: (-1x197x768xf32) <- (-1x197x768xf32, -1x197x768xf32)
        add_73 = paddle._C_ops.add(add_70, add_72)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [2147483647]

        # pd_op.slice: (-1x196x768xf32) <- (-1x197x768xf32, 1xi64, 1xi64)
        slice_50 = paddle._C_ops.slice(
            add_73, [1], full_int_array_1, full_int_array_6, [1], []
        )

        # pd_op.layer_norm: (-1x196x768xf32, -1x196xf32, -1x196xf32) <- (-1x196x768xf32, 768xf32, 768xf32)
        layer_norm_75, layer_norm_76, layer_norm_77 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                slice_50, parameter_3, parameter_2, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x196x512xf32) <- (-1x196x768xf32, 768x512xf32)
        matmul_73 = paddle._C_ops.matmul(layer_norm_75, data_0, False, False)

        # pd_op.mean: (-1x512xf32) <- (-1x196x512xf32)
        mean_0 = paddle._C_ops.mean(matmul_73, [1], False)

        # pd_op.matmul: (-1x512xf32) <- (-1x512xf32, 512x512xf32)
        matmul_74 = paddle._C_ops.matmul(mean_0, parameter_1, False, False)

        # pd_op.add: (-1x512xf32) <- (-1x512xf32, 512xf32)
        add_0 = paddle._C_ops.add(matmul_74, parameter_0)

        # pd_op.square: (-1x512xf32) <- (-1x512xf32)
        square_0 = paddle._C_ops.square(add_0)

        # pd_op.sum: (-1x1xf32) <- (-1x512xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(square_0, full_int_array_1, None, True)

        # pd_op.sqrt: (-1x1xf32) <- (-1x1xf32)
        sqrt_0 = paddle._C_ops.sqrt(sum_0)

        # pd_op.divide: (-1x512xf32) <- (-1x512xf32, -1x1xf32)
        divide_0 = paddle._C_ops.divide(add_0, sqrt_0)

        # pd_op.square: (512x159xf32) <- (512x159xf32)
        square_1 = paddle._C_ops.square(data_1)

        # pd_op.sum: (1x159xf32) <- (512x159xf32, 1xi64)
        sum_1 = paddle._C_ops.sum(square_1, full_int_array_0, None, True)

        # pd_op.sqrt: (1x159xf32) <- (1x159xf32)
        sqrt_1 = paddle._C_ops.sqrt(sum_1)

        # pd_op.divide: (512x159xf32) <- (512x159xf32, 1x159xf32)
        divide_1 = paddle._C_ops.divide(data_1, sqrt_1)

        # pd_op.matmul: (-1x159xf32) <- (-1x512xf32, 512x159xf32)
        matmul_0 = paddle._C_ops.matmul(divide_0, divide_1, False, False)
        return mean_0, add_0, matmul_0
