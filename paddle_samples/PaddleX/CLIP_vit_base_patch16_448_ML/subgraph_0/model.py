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
        parameter_151,
        parameter_152,
        parameter_153,
        parameter_154,
        parameter_155,
        parameter_156,
        parameter_157,
        parameter_158,
        parameter_159,
        parameter_160,
        parameter_161,
        parameter_162,
        parameter_163,
        parameter_164,
        parameter_165,
        parameter_166,
        parameter_167,
        parameter_168,
        parameter_169,
        parameter_170,
        data_0,
        data_1,
        data_2,
        data_3,
        data_4,
    ):
        # pd_op.conv2d: (-1x768x28x28xf32) <- (-1x3x448x448xf32, 768x3x16x16xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_4, parameter_170, [16, 16], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_4, parameter_170

        # pd_op.shape64: (4xi64) <- (-1x768x28x28xf32)
        shape64_0 = paddle._C_ops.shape64(conv2d_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_3 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_4 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_5 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_6 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_7 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_8 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_9 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_10 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_11 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_12 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_13 = full_int_array_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_14 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_15 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_16 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_17 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_18 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_19 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_20 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_21 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_22 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_23 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_24 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_25 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_26 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_27 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_28 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_29 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_30 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_31 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_32 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_33 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_34 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_35 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_36 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_37 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_38 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_39 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_40 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_41 = full_int_array_1

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_0

        # pd_op.flatten: (-1x768x784xf32) <- (-1x768x28x28xf32)
        flatten_1 = paddle._C_ops.flatten(conv2d_0, 2, 3)

        # pd_op.transpose: (-1x784x768xf32) <- (-1x768x784xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_1, [0, 2, 1])
        del flatten_1

        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("-1"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_0 = [data_3, full_0, full_0]
        del data_3

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.expand: (-1x1x768xf32) <- (1x1x768xf32, 3xi64)
        expand_0 = paddle._C_ops.expand(data_0, stack_0)
        del data_0

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_42 = full_1

        # builtin.combine: ([-1x1x768xf32, -1x784x768xf32]) <- (-1x1x768xf32, -1x784x768xf32)
        combine_1 = [expand_0, transpose_0]

        # pd_op.concat: (-1x785x768xf32) <- ([-1x1x768xf32, -1x784x768xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_1, full_1)
        del combine_1

        # pd_op.slice: (1x1x768xf32) <- (1x197x768xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_1, [1], full_int_array_0, full_int_array_1, [1], []
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2147483647]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_43 = full_int_array_2

        # pd_op.slice: (1x196x768xf32) <- (1x197x768xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            data_1, [1], full_int_array_1, full_int_array_2, [1], []
        )
        del data_1

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [-1, 14, 14, 768]

        # pd_op.reshape: (1x14x14x768xf32) <- (1x196x768xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(slice_2, full_int_array_3)
        del full_int_array_3

        # pd_op.transpose: (1x768x14x14xf32) <- (1x14x14x768xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_0, [0, 3, 1, 2])
        del reshape_0

        # pd_op.cast: (1x768x14x14xf32) <- (1x768x14x14xf32)
        cast_0 = paddle._C_ops.cast(transpose_1, paddle.float32)
        del transpose_1

        # pd_op.bicubic_interp: (1x768x28x28xf32) <- (1x768x14x14xf32, None, None, None)
        bicubic_interp_0 = paddle._C_ops.bicubic_interp(
            cast_0, None, None, None, "NCHW", -1, 28, 28, [], "bicubic", False, 0
        )

        # pd_op.flatten: (1x768x784xf32) <- (1x768x28x28xf32)
        flatten_2 = paddle._C_ops.flatten(bicubic_interp_0, 2, 3)

        # pd_op.transpose: (1x784x768xf32) <- (1x768x784xf32)
        transpose_2 = paddle._C_ops.transpose(flatten_2, [0, 2, 1])
        del flatten_2

        # pd_op.cast: (1x784x768xf32) <- (1x784x768xf32)
        cast_1 = paddle._C_ops.cast(transpose_2, paddle.float32)
        del transpose_2

        # builtin.combine: ([1x1x768xf32, 1x784x768xf32]) <- (1x1x768xf32, 1x784x768xf32)
        combine_2 = [slice_1, cast_1]

        # pd_op.concat: (1x785x768xf32) <- ([1x1x768xf32, 1x784x768xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_2, full_1)
        del combine_2

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, 1x785x768xf32)
        add_0 = paddle._C_ops.add(concat_0, concat_1)

        # pd_op.layer_norm: (-1x785x768xf32, -1x785xf32, -1x785xf32) <- (-1x785x768xf32, 768xf32, 768xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_0, parameter_169, parameter_168, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_168, parameter_169

        # pd_op.layer_norm: (-1x785x768xf32, -1x785xf32, -1x785xf32) <- (-1x785x768xf32, 768xf32, 768xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                layer_norm_0, parameter_167, parameter_166, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_166, parameter_167

        # pd_op.shape64: (3xi64) <- (-1x785x768xf32)
        shape64_1 = paddle._C_ops.shape64(layer_norm_3)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_1

        # pd_op.matmul: (-1x785x2304xf32) <- (-1x785x768xf32, 768x2304xf32)
        matmul_0 = paddle._C_ops.matmul(layer_norm_3, parameter_165, False, False)
        del parameter_165

        # pd_op.add: (-1x785x2304xf32) <- (-1x785x2304xf32, 2304xf32)
        add_1 = paddle._C_ops.add(matmul_0, parameter_164)
        del parameter_164

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_4 = [-1, 785, 3, 12, 64]

        # pd_op.reshape: (-1x785x3x12x64xf32) <- (-1x785x2304xf32, 5xi64)
        reshape_1 = paddle._C_ops.reshape(add_1, full_int_array_4)

        # pd_op.transpose: (3x-1x12x785x64xf32) <- (-1x785x3x12x64xf32)
        transpose_3 = paddle._C_ops.transpose(reshape_1, [2, 0, 3, 1, 4])
        del reshape_1

        # pd_op.slice: (-1x12x785x64xf32) <- (3x-1x12x785x64xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            transpose_3, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_44 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_45 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_46 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_47 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_48 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_49 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_50 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_51 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_52 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_53 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_54 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_55 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_56 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_57 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_58 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_59 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_60 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_61 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_62 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_63 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_64 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_65 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_66 = full_int_array_5

        # pd_op.slice: (-1x12x785x64xf32) <- (3x-1x12x785x64xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            transpose_3, [0], full_int_array_1, full_int_array_5, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_67 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_68 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_69 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_70 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_71 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_72 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_73 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_74 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_75 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_76 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_77 = full_int_array_6

        # pd_op.slice: (-1x12x785x64xf32) <- (3x-1x12x785x64xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            transpose_3, [0], full_int_array_5, full_int_array_6, [1], [0]
        )

        # pd_op.transpose: (-1x12x64x785xf32) <- (-1x12x785x64xf32)
        transpose_4 = paddle._C_ops.transpose(slice_5, [0, 1, 3, 2])
        del slice_5

        # pd_op.matmul: (-1x12x785x785xf32) <- (-1x12x785x64xf32, -1x12x64x785xf32)
        matmul_1 = paddle._C_ops.matmul(slice_4, transpose_4, False, False)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_78 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_79 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_80 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_81 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_82 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_83 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_84 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_85 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_86 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_87 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_88 = full_2

        # pd_op.scale: (-1x12x785x785xf32) <- (-1x12x785x785xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(matmul_1, full_2, float("0"), True)
        del matmul_1

        # pd_op.softmax: (-1x12x785x785xf32) <- (-1x12x785x785xf32)
        softmax_0 = paddle._C_ops.softmax(scale_0, -1)
        del scale_0

        # pd_op.matmul: (-1x12x785x64xf32) <- (-1x12x785x785xf32, -1x12x785x64xf32)
        matmul_2 = paddle._C_ops.matmul(softmax_0, slice_6, False, False)

        # pd_op.transpose: (-1x785x12x64xf32) <- (-1x12x785x64xf32)
        transpose_5 = paddle._C_ops.transpose(matmul_2, [0, 2, 1, 3])
        del matmul_2

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_7 = [-1, 785, 768]

        # pd_op.reshape: (-1x785x768xf32) <- (-1x785x12x64xf32, 3xi64)
        reshape_2 = paddle._C_ops.reshape(transpose_5, full_int_array_7)

        # pd_op.matmul: (-1x785x768xf32) <- (-1x785x768xf32, 768x768xf32)
        matmul_3 = paddle._C_ops.matmul(reshape_2, parameter_163, False, False)
        del parameter_163

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, 768xf32)
        add_2 = paddle._C_ops.add(matmul_3, parameter_162)
        del parameter_162

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, -1x785x768xf32)
        add_3 = paddle._C_ops.add(layer_norm_0, add_2)

        # pd_op.layer_norm: (-1x785x768xf32, -1x785xf32, -1x785xf32) <- (-1x785x768xf32, 768xf32, 768xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_3, parameter_161, parameter_160, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_160, parameter_161

        # pd_op.matmul: (-1x785x3072xf32) <- (-1x785x768xf32, 768x3072xf32)
        matmul_4 = paddle._C_ops.matmul(layer_norm_6, parameter_159, False, False)
        del parameter_159

        # pd_op.add: (-1x785x3072xf32) <- (-1x785x3072xf32, 3072xf32)
        add_4 = paddle._C_ops.add(matmul_4, parameter_158)
        del parameter_158

        # pd_op.gelu: (-1x785x3072xf32) <- (-1x785x3072xf32)
        gelu_0 = paddle._C_ops.gelu(add_4, False)

        # pd_op.matmul: (-1x785x768xf32) <- (-1x785x3072xf32, 3072x768xf32)
        matmul_5 = paddle._C_ops.matmul(gelu_0, parameter_157, False, False)
        del parameter_157

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, 768xf32)
        add_5 = paddle._C_ops.add(matmul_5, parameter_156)
        del parameter_156

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, -1x785x768xf32)
        add_6 = paddle._C_ops.add(add_3, add_5)

        # pd_op.layer_norm: (-1x785x768xf32, -1x785xf32, -1x785xf32) <- (-1x785x768xf32, 768xf32, 768xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_6, parameter_155, parameter_154, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_154, parameter_155

        # pd_op.shape64: (3xi64) <- (-1x785x768xf32)
        shape64_2 = paddle._C_ops.shape64(layer_norm_9)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_2

        # pd_op.matmul: (-1x785x2304xf32) <- (-1x785x768xf32, 768x2304xf32)
        matmul_6 = paddle._C_ops.matmul(layer_norm_9, parameter_153, False, False)
        del parameter_153

        # pd_op.add: (-1x785x2304xf32) <- (-1x785x2304xf32, 2304xf32)
        add_7 = paddle._C_ops.add(matmul_6, parameter_152)
        del parameter_152

        # pd_op.reshape: (-1x785x3x12x64xf32) <- (-1x785x2304xf32, 5xi64)
        reshape_3 = paddle._C_ops.reshape(add_7, full_int_array_4)

        # pd_op.transpose: (3x-1x12x785x64xf32) <- (-1x785x3x12x64xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_3, [2, 0, 3, 1, 4])
        del reshape_3

        # pd_op.slice: (-1x12x785x64xf32) <- (3x-1x12x785x64xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            transpose_6, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x12x785x64xf32) <- (3x-1x12x785x64xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            transpose_6, [0], full_int_array_1, full_int_array_5, [1], [0]
        )

        # pd_op.slice: (-1x12x785x64xf32) <- (3x-1x12x785x64xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            transpose_6, [0], full_int_array_5, full_int_array_6, [1], [0]
        )

        # pd_op.transpose: (-1x12x64x785xf32) <- (-1x12x785x64xf32)
        transpose_7 = paddle._C_ops.transpose(slice_9, [0, 1, 3, 2])
        del slice_9

        # pd_op.matmul: (-1x12x785x785xf32) <- (-1x12x785x64xf32, -1x12x64x785xf32)
        matmul_7 = paddle._C_ops.matmul(slice_8, transpose_7, False, False)

        # pd_op.scale: (-1x12x785x785xf32) <- (-1x12x785x785xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(matmul_7, full_2, float("0"), True)
        del matmul_7

        # pd_op.softmax: (-1x12x785x785xf32) <- (-1x12x785x785xf32)
        softmax_1 = paddle._C_ops.softmax(scale_1, -1)
        del scale_1

        # pd_op.matmul: (-1x12x785x64xf32) <- (-1x12x785x785xf32, -1x12x785x64xf32)
        matmul_8 = paddle._C_ops.matmul(softmax_1, slice_10, False, False)

        # pd_op.transpose: (-1x785x12x64xf32) <- (-1x12x785x64xf32)
        transpose_8 = paddle._C_ops.transpose(matmul_8, [0, 2, 1, 3])
        del matmul_8

        # pd_op.reshape: (-1x785x768xf32) <- (-1x785x12x64xf32, 3xi64)
        reshape_4 = paddle._C_ops.reshape(transpose_8, full_int_array_7)

        # pd_op.matmul: (-1x785x768xf32) <- (-1x785x768xf32, 768x768xf32)
        matmul_9 = paddle._C_ops.matmul(reshape_4, parameter_151, False, False)
        del parameter_151

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, 768xf32)
        add_8 = paddle._C_ops.add(matmul_9, parameter_150)
        del parameter_150

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, -1x785x768xf32)
        add_9 = paddle._C_ops.add(add_6, add_8)

        # pd_op.layer_norm: (-1x785x768xf32, -1x785xf32, -1x785xf32) <- (-1x785x768xf32, 768xf32, 768xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_9, parameter_149, parameter_148, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_148, parameter_149

        # pd_op.matmul: (-1x785x3072xf32) <- (-1x785x768xf32, 768x3072xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_12, parameter_147, False, False)
        del parameter_147

        # pd_op.add: (-1x785x3072xf32) <- (-1x785x3072xf32, 3072xf32)
        add_10 = paddle._C_ops.add(matmul_10, parameter_146)
        del parameter_146

        # pd_op.gelu: (-1x785x3072xf32) <- (-1x785x3072xf32)
        gelu_1 = paddle._C_ops.gelu(add_10, False)

        # pd_op.matmul: (-1x785x768xf32) <- (-1x785x3072xf32, 3072x768xf32)
        matmul_11 = paddle._C_ops.matmul(gelu_1, parameter_145, False, False)
        del parameter_145

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, 768xf32)
        add_11 = paddle._C_ops.add(matmul_11, parameter_144)
        del parameter_144

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, -1x785x768xf32)
        add_12 = paddle._C_ops.add(add_9, add_11)

        # pd_op.layer_norm: (-1x785x768xf32, -1x785xf32, -1x785xf32) <- (-1x785x768xf32, 768xf32, 768xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_12, parameter_143, parameter_142, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_142, parameter_143

        # pd_op.shape64: (3xi64) <- (-1x785x768xf32)
        shape64_3 = paddle._C_ops.shape64(layer_norm_15)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_3

        # pd_op.matmul: (-1x785x2304xf32) <- (-1x785x768xf32, 768x2304xf32)
        matmul_12 = paddle._C_ops.matmul(layer_norm_15, parameter_141, False, False)
        del parameter_141

        # pd_op.add: (-1x785x2304xf32) <- (-1x785x2304xf32, 2304xf32)
        add_13 = paddle._C_ops.add(matmul_12, parameter_140)
        del parameter_140

        # pd_op.reshape: (-1x785x3x12x64xf32) <- (-1x785x2304xf32, 5xi64)
        reshape_5 = paddle._C_ops.reshape(add_13, full_int_array_4)

        # pd_op.transpose: (3x-1x12x785x64xf32) <- (-1x785x3x12x64xf32)
        transpose_9 = paddle._C_ops.transpose(reshape_5, [2, 0, 3, 1, 4])
        del reshape_5

        # pd_op.slice: (-1x12x785x64xf32) <- (3x-1x12x785x64xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            transpose_9, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x12x785x64xf32) <- (3x-1x12x785x64xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            transpose_9, [0], full_int_array_1, full_int_array_5, [1], [0]
        )

        # pd_op.slice: (-1x12x785x64xf32) <- (3x-1x12x785x64xf32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            transpose_9, [0], full_int_array_5, full_int_array_6, [1], [0]
        )

        # pd_op.transpose: (-1x12x64x785xf32) <- (-1x12x785x64xf32)
        transpose_10 = paddle._C_ops.transpose(slice_13, [0, 1, 3, 2])
        del slice_13

        # pd_op.matmul: (-1x12x785x785xf32) <- (-1x12x785x64xf32, -1x12x64x785xf32)
        matmul_13 = paddle._C_ops.matmul(slice_12, transpose_10, False, False)

        # pd_op.scale: (-1x12x785x785xf32) <- (-1x12x785x785xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(matmul_13, full_2, float("0"), True)
        del matmul_13

        # pd_op.softmax: (-1x12x785x785xf32) <- (-1x12x785x785xf32)
        softmax_2 = paddle._C_ops.softmax(scale_2, -1)
        del scale_2

        # pd_op.matmul: (-1x12x785x64xf32) <- (-1x12x785x785xf32, -1x12x785x64xf32)
        matmul_14 = paddle._C_ops.matmul(softmax_2, slice_14, False, False)

        # pd_op.transpose: (-1x785x12x64xf32) <- (-1x12x785x64xf32)
        transpose_11 = paddle._C_ops.transpose(matmul_14, [0, 2, 1, 3])
        del matmul_14

        # pd_op.reshape: (-1x785x768xf32) <- (-1x785x12x64xf32, 3xi64)
        reshape_6 = paddle._C_ops.reshape(transpose_11, full_int_array_7)

        # pd_op.matmul: (-1x785x768xf32) <- (-1x785x768xf32, 768x768xf32)
        matmul_15 = paddle._C_ops.matmul(reshape_6, parameter_139, False, False)
        del parameter_139

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, 768xf32)
        add_14 = paddle._C_ops.add(matmul_15, parameter_138)
        del parameter_138

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, -1x785x768xf32)
        add_15 = paddle._C_ops.add(add_12, add_14)

        # pd_op.layer_norm: (-1x785x768xf32, -1x785xf32, -1x785xf32) <- (-1x785x768xf32, 768xf32, 768xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_15, parameter_137, parameter_136, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_136, parameter_137

        # pd_op.matmul: (-1x785x3072xf32) <- (-1x785x768xf32, 768x3072xf32)
        matmul_16 = paddle._C_ops.matmul(layer_norm_18, parameter_135, False, False)
        del parameter_135

        # pd_op.add: (-1x785x3072xf32) <- (-1x785x3072xf32, 3072xf32)
        add_16 = paddle._C_ops.add(matmul_16, parameter_134)
        del parameter_134

        # pd_op.gelu: (-1x785x3072xf32) <- (-1x785x3072xf32)
        gelu_2 = paddle._C_ops.gelu(add_16, False)

        # pd_op.matmul: (-1x785x768xf32) <- (-1x785x3072xf32, 3072x768xf32)
        matmul_17 = paddle._C_ops.matmul(gelu_2, parameter_133, False, False)
        del parameter_133

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, 768xf32)
        add_17 = paddle._C_ops.add(matmul_17, parameter_132)
        del parameter_132

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, -1x785x768xf32)
        add_18 = paddle._C_ops.add(add_15, add_17)

        # pd_op.layer_norm: (-1x785x768xf32, -1x785xf32, -1x785xf32) <- (-1x785x768xf32, 768xf32, 768xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_18, parameter_131, parameter_130, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_130, parameter_131

        # pd_op.shape64: (3xi64) <- (-1x785x768xf32)
        shape64_4 = paddle._C_ops.shape64(layer_norm_21)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            shape64_4, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_4

        # pd_op.matmul: (-1x785x2304xf32) <- (-1x785x768xf32, 768x2304xf32)
        matmul_18 = paddle._C_ops.matmul(layer_norm_21, parameter_129, False, False)
        del parameter_129

        # pd_op.add: (-1x785x2304xf32) <- (-1x785x2304xf32, 2304xf32)
        add_19 = paddle._C_ops.add(matmul_18, parameter_128)
        del parameter_128

        # pd_op.reshape: (-1x785x3x12x64xf32) <- (-1x785x2304xf32, 5xi64)
        reshape_7 = paddle._C_ops.reshape(add_19, full_int_array_4)

        # pd_op.transpose: (3x-1x12x785x64xf32) <- (-1x785x3x12x64xf32)
        transpose_12 = paddle._C_ops.transpose(reshape_7, [2, 0, 3, 1, 4])
        del reshape_7

        # pd_op.slice: (-1x12x785x64xf32) <- (3x-1x12x785x64xf32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            transpose_12, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x12x785x64xf32) <- (3x-1x12x785x64xf32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            transpose_12, [0], full_int_array_1, full_int_array_5, [1], [0]
        )

        # pd_op.slice: (-1x12x785x64xf32) <- (3x-1x12x785x64xf32, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(
            transpose_12, [0], full_int_array_5, full_int_array_6, [1], [0]
        )

        # pd_op.transpose: (-1x12x64x785xf32) <- (-1x12x785x64xf32)
        transpose_13 = paddle._C_ops.transpose(slice_17, [0, 1, 3, 2])
        del slice_17

        # pd_op.matmul: (-1x12x785x785xf32) <- (-1x12x785x64xf32, -1x12x64x785xf32)
        matmul_19 = paddle._C_ops.matmul(slice_16, transpose_13, False, False)

        # pd_op.scale: (-1x12x785x785xf32) <- (-1x12x785x785xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(matmul_19, full_2, float("0"), True)
        del matmul_19

        # pd_op.softmax: (-1x12x785x785xf32) <- (-1x12x785x785xf32)
        softmax_3 = paddle._C_ops.softmax(scale_3, -1)
        del scale_3

        # pd_op.matmul: (-1x12x785x64xf32) <- (-1x12x785x785xf32, -1x12x785x64xf32)
        matmul_20 = paddle._C_ops.matmul(softmax_3, slice_18, False, False)

        # pd_op.transpose: (-1x785x12x64xf32) <- (-1x12x785x64xf32)
        transpose_14 = paddle._C_ops.transpose(matmul_20, [0, 2, 1, 3])
        del matmul_20

        # pd_op.reshape: (-1x785x768xf32) <- (-1x785x12x64xf32, 3xi64)
        reshape_8 = paddle._C_ops.reshape(transpose_14, full_int_array_7)

        # pd_op.matmul: (-1x785x768xf32) <- (-1x785x768xf32, 768x768xf32)
        matmul_21 = paddle._C_ops.matmul(reshape_8, parameter_127, False, False)
        del parameter_127

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, 768xf32)
        add_20 = paddle._C_ops.add(matmul_21, parameter_126)
        del parameter_126

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, -1x785x768xf32)
        add_21 = paddle._C_ops.add(add_18, add_20)

        # pd_op.layer_norm: (-1x785x768xf32, -1x785xf32, -1x785xf32) <- (-1x785x768xf32, 768xf32, 768xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_21, parameter_125, parameter_124, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_124, parameter_125

        # pd_op.matmul: (-1x785x3072xf32) <- (-1x785x768xf32, 768x3072xf32)
        matmul_22 = paddle._C_ops.matmul(layer_norm_24, parameter_123, False, False)
        del parameter_123

        # pd_op.add: (-1x785x3072xf32) <- (-1x785x3072xf32, 3072xf32)
        add_22 = paddle._C_ops.add(matmul_22, parameter_122)
        del parameter_122

        # pd_op.gelu: (-1x785x3072xf32) <- (-1x785x3072xf32)
        gelu_3 = paddle._C_ops.gelu(add_22, False)

        # pd_op.matmul: (-1x785x768xf32) <- (-1x785x3072xf32, 3072x768xf32)
        matmul_23 = paddle._C_ops.matmul(gelu_3, parameter_121, False, False)
        del parameter_121

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, 768xf32)
        add_23 = paddle._C_ops.add(matmul_23, parameter_120)
        del parameter_120

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, -1x785x768xf32)
        add_24 = paddle._C_ops.add(add_21, add_23)

        # pd_op.layer_norm: (-1x785x768xf32, -1x785xf32, -1x785xf32) <- (-1x785x768xf32, 768xf32, 768xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_24, parameter_119, parameter_118, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_118, parameter_119

        # pd_op.shape64: (3xi64) <- (-1x785x768xf32)
        shape64_5 = paddle._C_ops.shape64(layer_norm_27)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(
            shape64_5, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_5

        # pd_op.matmul: (-1x785x2304xf32) <- (-1x785x768xf32, 768x2304xf32)
        matmul_24 = paddle._C_ops.matmul(layer_norm_27, parameter_117, False, False)
        del parameter_117

        # pd_op.add: (-1x785x2304xf32) <- (-1x785x2304xf32, 2304xf32)
        add_25 = paddle._C_ops.add(matmul_24, parameter_116)
        del parameter_116

        # pd_op.reshape: (-1x785x3x12x64xf32) <- (-1x785x2304xf32, 5xi64)
        reshape_9 = paddle._C_ops.reshape(add_25, full_int_array_4)

        # pd_op.transpose: (3x-1x12x785x64xf32) <- (-1x785x3x12x64xf32)
        transpose_15 = paddle._C_ops.transpose(reshape_9, [2, 0, 3, 1, 4])
        del reshape_9

        # pd_op.slice: (-1x12x785x64xf32) <- (3x-1x12x785x64xf32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(
            transpose_15, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x12x785x64xf32) <- (3x-1x12x785x64xf32, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(
            transpose_15, [0], full_int_array_1, full_int_array_5, [1], [0]
        )

        # pd_op.slice: (-1x12x785x64xf32) <- (3x-1x12x785x64xf32, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(
            transpose_15, [0], full_int_array_5, full_int_array_6, [1], [0]
        )

        # pd_op.transpose: (-1x12x64x785xf32) <- (-1x12x785x64xf32)
        transpose_16 = paddle._C_ops.transpose(slice_21, [0, 1, 3, 2])
        del slice_21

        # pd_op.matmul: (-1x12x785x785xf32) <- (-1x12x785x64xf32, -1x12x64x785xf32)
        matmul_25 = paddle._C_ops.matmul(slice_20, transpose_16, False, False)

        # pd_op.scale: (-1x12x785x785xf32) <- (-1x12x785x785xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(matmul_25, full_2, float("0"), True)
        del matmul_25

        # pd_op.softmax: (-1x12x785x785xf32) <- (-1x12x785x785xf32)
        softmax_4 = paddle._C_ops.softmax(scale_4, -1)
        del scale_4

        # pd_op.matmul: (-1x12x785x64xf32) <- (-1x12x785x785xf32, -1x12x785x64xf32)
        matmul_26 = paddle._C_ops.matmul(softmax_4, slice_22, False, False)

        # pd_op.transpose: (-1x785x12x64xf32) <- (-1x12x785x64xf32)
        transpose_17 = paddle._C_ops.transpose(matmul_26, [0, 2, 1, 3])
        del matmul_26

        # pd_op.reshape: (-1x785x768xf32) <- (-1x785x12x64xf32, 3xi64)
        reshape_10 = paddle._C_ops.reshape(transpose_17, full_int_array_7)

        # pd_op.matmul: (-1x785x768xf32) <- (-1x785x768xf32, 768x768xf32)
        matmul_27 = paddle._C_ops.matmul(reshape_10, parameter_115, False, False)
        del parameter_115

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, 768xf32)
        add_26 = paddle._C_ops.add(matmul_27, parameter_114)
        del parameter_114

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, -1x785x768xf32)
        add_27 = paddle._C_ops.add(add_24, add_26)

        # pd_op.layer_norm: (-1x785x768xf32, -1x785xf32, -1x785xf32) <- (-1x785x768xf32, 768xf32, 768xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_27, parameter_113, parameter_112, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_112, parameter_113

        # pd_op.matmul: (-1x785x3072xf32) <- (-1x785x768xf32, 768x3072xf32)
        matmul_28 = paddle._C_ops.matmul(layer_norm_30, parameter_111, False, False)
        del parameter_111

        # pd_op.add: (-1x785x3072xf32) <- (-1x785x3072xf32, 3072xf32)
        add_28 = paddle._C_ops.add(matmul_28, parameter_110)
        del parameter_110

        # pd_op.gelu: (-1x785x3072xf32) <- (-1x785x3072xf32)
        gelu_4 = paddle._C_ops.gelu(add_28, False)

        # pd_op.matmul: (-1x785x768xf32) <- (-1x785x3072xf32, 3072x768xf32)
        matmul_29 = paddle._C_ops.matmul(gelu_4, parameter_109, False, False)
        del parameter_109

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, 768xf32)
        add_29 = paddle._C_ops.add(matmul_29, parameter_108)
        del parameter_108

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, -1x785x768xf32)
        add_30 = paddle._C_ops.add(add_27, add_29)

        # pd_op.layer_norm: (-1x785x768xf32, -1x785xf32, -1x785xf32) <- (-1x785x768xf32, 768xf32, 768xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_30, parameter_107, parameter_106, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_106, parameter_107

        # pd_op.shape64: (3xi64) <- (-1x785x768xf32)
        shape64_6 = paddle._C_ops.shape64(layer_norm_33)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(
            shape64_6, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_6

        # pd_op.matmul: (-1x785x2304xf32) <- (-1x785x768xf32, 768x2304xf32)
        matmul_30 = paddle._C_ops.matmul(layer_norm_33, parameter_105, False, False)
        del parameter_105

        # pd_op.add: (-1x785x2304xf32) <- (-1x785x2304xf32, 2304xf32)
        add_31 = paddle._C_ops.add(matmul_30, parameter_104)
        del parameter_104

        # pd_op.reshape: (-1x785x3x12x64xf32) <- (-1x785x2304xf32, 5xi64)
        reshape_11 = paddle._C_ops.reshape(add_31, full_int_array_4)

        # pd_op.transpose: (3x-1x12x785x64xf32) <- (-1x785x3x12x64xf32)
        transpose_18 = paddle._C_ops.transpose(reshape_11, [2, 0, 3, 1, 4])
        del reshape_11

        # pd_op.slice: (-1x12x785x64xf32) <- (3x-1x12x785x64xf32, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(
            transpose_18, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x12x785x64xf32) <- (3x-1x12x785x64xf32, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(
            transpose_18, [0], full_int_array_1, full_int_array_5, [1], [0]
        )

        # pd_op.slice: (-1x12x785x64xf32) <- (3x-1x12x785x64xf32, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(
            transpose_18, [0], full_int_array_5, full_int_array_6, [1], [0]
        )

        # pd_op.transpose: (-1x12x64x785xf32) <- (-1x12x785x64xf32)
        transpose_19 = paddle._C_ops.transpose(slice_25, [0, 1, 3, 2])
        del slice_25

        # pd_op.matmul: (-1x12x785x785xf32) <- (-1x12x785x64xf32, -1x12x64x785xf32)
        matmul_31 = paddle._C_ops.matmul(slice_24, transpose_19, False, False)

        # pd_op.scale: (-1x12x785x785xf32) <- (-1x12x785x785xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(matmul_31, full_2, float("0"), True)
        del matmul_31

        # pd_op.softmax: (-1x12x785x785xf32) <- (-1x12x785x785xf32)
        softmax_5 = paddle._C_ops.softmax(scale_5, -1)
        del scale_5

        # pd_op.matmul: (-1x12x785x64xf32) <- (-1x12x785x785xf32, -1x12x785x64xf32)
        matmul_32 = paddle._C_ops.matmul(softmax_5, slice_26, False, False)

        # pd_op.transpose: (-1x785x12x64xf32) <- (-1x12x785x64xf32)
        transpose_20 = paddle._C_ops.transpose(matmul_32, [0, 2, 1, 3])
        del matmul_32

        # pd_op.reshape: (-1x785x768xf32) <- (-1x785x12x64xf32, 3xi64)
        reshape_12 = paddle._C_ops.reshape(transpose_20, full_int_array_7)

        # pd_op.matmul: (-1x785x768xf32) <- (-1x785x768xf32, 768x768xf32)
        matmul_33 = paddle._C_ops.matmul(reshape_12, parameter_103, False, False)
        del parameter_103

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, 768xf32)
        add_32 = paddle._C_ops.add(matmul_33, parameter_102)
        del parameter_102

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, -1x785x768xf32)
        add_33 = paddle._C_ops.add(add_30, add_32)

        # pd_op.layer_norm: (-1x785x768xf32, -1x785xf32, -1x785xf32) <- (-1x785x768xf32, 768xf32, 768xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_33, parameter_101, parameter_100, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_100, parameter_101

        # pd_op.matmul: (-1x785x3072xf32) <- (-1x785x768xf32, 768x3072xf32)
        matmul_34 = paddle._C_ops.matmul(layer_norm_36, parameter_99, False, False)
        del parameter_99

        # pd_op.add: (-1x785x3072xf32) <- (-1x785x3072xf32, 3072xf32)
        add_34 = paddle._C_ops.add(matmul_34, parameter_98)
        del parameter_98

        # pd_op.gelu: (-1x785x3072xf32) <- (-1x785x3072xf32)
        gelu_5 = paddle._C_ops.gelu(add_34, False)

        # pd_op.matmul: (-1x785x768xf32) <- (-1x785x3072xf32, 3072x768xf32)
        matmul_35 = paddle._C_ops.matmul(gelu_5, parameter_97, False, False)
        del parameter_97

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, 768xf32)
        add_35 = paddle._C_ops.add(matmul_35, parameter_96)
        del parameter_96

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, -1x785x768xf32)
        add_36 = paddle._C_ops.add(add_33, add_35)

        # pd_op.layer_norm: (-1x785x768xf32, -1x785xf32, -1x785xf32) <- (-1x785x768xf32, 768xf32, 768xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_36, parameter_95, parameter_94, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_94, parameter_95

        # pd_op.shape64: (3xi64) <- (-1x785x768xf32)
        shape64_7 = paddle._C_ops.shape64(layer_norm_39)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(
            shape64_7, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_7

        # pd_op.matmul: (-1x785x2304xf32) <- (-1x785x768xf32, 768x2304xf32)
        matmul_36 = paddle._C_ops.matmul(layer_norm_39, parameter_93, False, False)
        del parameter_93

        # pd_op.add: (-1x785x2304xf32) <- (-1x785x2304xf32, 2304xf32)
        add_37 = paddle._C_ops.add(matmul_36, parameter_92)
        del parameter_92

        # pd_op.reshape: (-1x785x3x12x64xf32) <- (-1x785x2304xf32, 5xi64)
        reshape_13 = paddle._C_ops.reshape(add_37, full_int_array_4)

        # pd_op.transpose: (3x-1x12x785x64xf32) <- (-1x785x3x12x64xf32)
        transpose_21 = paddle._C_ops.transpose(reshape_13, [2, 0, 3, 1, 4])
        del reshape_13

        # pd_op.slice: (-1x12x785x64xf32) <- (3x-1x12x785x64xf32, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(
            transpose_21, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x12x785x64xf32) <- (3x-1x12x785x64xf32, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(
            transpose_21, [0], full_int_array_1, full_int_array_5, [1], [0]
        )

        # pd_op.slice: (-1x12x785x64xf32) <- (3x-1x12x785x64xf32, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(
            transpose_21, [0], full_int_array_5, full_int_array_6, [1], [0]
        )

        # pd_op.transpose: (-1x12x64x785xf32) <- (-1x12x785x64xf32)
        transpose_22 = paddle._C_ops.transpose(slice_29, [0, 1, 3, 2])
        del slice_29

        # pd_op.matmul: (-1x12x785x785xf32) <- (-1x12x785x64xf32, -1x12x64x785xf32)
        matmul_37 = paddle._C_ops.matmul(slice_28, transpose_22, False, False)

        # pd_op.scale: (-1x12x785x785xf32) <- (-1x12x785x785xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(matmul_37, full_2, float("0"), True)
        del matmul_37

        # pd_op.softmax: (-1x12x785x785xf32) <- (-1x12x785x785xf32)
        softmax_6 = paddle._C_ops.softmax(scale_6, -1)
        del scale_6

        # pd_op.matmul: (-1x12x785x64xf32) <- (-1x12x785x785xf32, -1x12x785x64xf32)
        matmul_38 = paddle._C_ops.matmul(softmax_6, slice_30, False, False)

        # pd_op.transpose: (-1x785x12x64xf32) <- (-1x12x785x64xf32)
        transpose_23 = paddle._C_ops.transpose(matmul_38, [0, 2, 1, 3])
        del matmul_38

        # pd_op.reshape: (-1x785x768xf32) <- (-1x785x12x64xf32, 3xi64)
        reshape_14 = paddle._C_ops.reshape(transpose_23, full_int_array_7)

        # pd_op.matmul: (-1x785x768xf32) <- (-1x785x768xf32, 768x768xf32)
        matmul_39 = paddle._C_ops.matmul(reshape_14, parameter_91, False, False)
        del parameter_91

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, 768xf32)
        add_38 = paddle._C_ops.add(matmul_39, parameter_90)
        del parameter_90

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, -1x785x768xf32)
        add_39 = paddle._C_ops.add(add_36, add_38)

        # pd_op.layer_norm: (-1x785x768xf32, -1x785xf32, -1x785xf32) <- (-1x785x768xf32, 768xf32, 768xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_39, parameter_89, parameter_88, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_88, parameter_89

        # pd_op.matmul: (-1x785x3072xf32) <- (-1x785x768xf32, 768x3072xf32)
        matmul_40 = paddle._C_ops.matmul(layer_norm_42, parameter_87, False, False)
        del parameter_87

        # pd_op.add: (-1x785x3072xf32) <- (-1x785x3072xf32, 3072xf32)
        add_40 = paddle._C_ops.add(matmul_40, parameter_86)
        del parameter_86

        # pd_op.gelu: (-1x785x3072xf32) <- (-1x785x3072xf32)
        gelu_6 = paddle._C_ops.gelu(add_40, False)

        # pd_op.matmul: (-1x785x768xf32) <- (-1x785x3072xf32, 3072x768xf32)
        matmul_41 = paddle._C_ops.matmul(gelu_6, parameter_85, False, False)
        del parameter_85

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, 768xf32)
        add_41 = paddle._C_ops.add(matmul_41, parameter_84)
        del parameter_84

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, -1x785x768xf32)
        add_42 = paddle._C_ops.add(add_39, add_41)

        # pd_op.layer_norm: (-1x785x768xf32, -1x785xf32, -1x785xf32) <- (-1x785x768xf32, 768xf32, 768xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_42, parameter_83, parameter_82, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_82, parameter_83

        # pd_op.shape64: (3xi64) <- (-1x785x768xf32)
        shape64_8 = paddle._C_ops.shape64(layer_norm_45)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(
            shape64_8, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_8

        # pd_op.matmul: (-1x785x2304xf32) <- (-1x785x768xf32, 768x2304xf32)
        matmul_42 = paddle._C_ops.matmul(layer_norm_45, parameter_81, False, False)
        del parameter_81

        # pd_op.add: (-1x785x2304xf32) <- (-1x785x2304xf32, 2304xf32)
        add_43 = paddle._C_ops.add(matmul_42, parameter_80)
        del parameter_80

        # pd_op.reshape: (-1x785x3x12x64xf32) <- (-1x785x2304xf32, 5xi64)
        reshape_15 = paddle._C_ops.reshape(add_43, full_int_array_4)

        # pd_op.transpose: (3x-1x12x785x64xf32) <- (-1x785x3x12x64xf32)
        transpose_24 = paddle._C_ops.transpose(reshape_15, [2, 0, 3, 1, 4])
        del reshape_15

        # pd_op.slice: (-1x12x785x64xf32) <- (3x-1x12x785x64xf32, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(
            transpose_24, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x12x785x64xf32) <- (3x-1x12x785x64xf32, 1xi64, 1xi64)
        slice_33 = paddle._C_ops.slice(
            transpose_24, [0], full_int_array_1, full_int_array_5, [1], [0]
        )

        # pd_op.slice: (-1x12x785x64xf32) <- (3x-1x12x785x64xf32, 1xi64, 1xi64)
        slice_34 = paddle._C_ops.slice(
            transpose_24, [0], full_int_array_5, full_int_array_6, [1], [0]
        )

        # pd_op.transpose: (-1x12x64x785xf32) <- (-1x12x785x64xf32)
        transpose_25 = paddle._C_ops.transpose(slice_33, [0, 1, 3, 2])
        del slice_33

        # pd_op.matmul: (-1x12x785x785xf32) <- (-1x12x785x64xf32, -1x12x64x785xf32)
        matmul_43 = paddle._C_ops.matmul(slice_32, transpose_25, False, False)

        # pd_op.scale: (-1x12x785x785xf32) <- (-1x12x785x785xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(matmul_43, full_2, float("0"), True)
        del matmul_43

        # pd_op.softmax: (-1x12x785x785xf32) <- (-1x12x785x785xf32)
        softmax_7 = paddle._C_ops.softmax(scale_7, -1)
        del scale_7

        # pd_op.matmul: (-1x12x785x64xf32) <- (-1x12x785x785xf32, -1x12x785x64xf32)
        matmul_44 = paddle._C_ops.matmul(softmax_7, slice_34, False, False)

        # pd_op.transpose: (-1x785x12x64xf32) <- (-1x12x785x64xf32)
        transpose_26 = paddle._C_ops.transpose(matmul_44, [0, 2, 1, 3])
        del matmul_44

        # pd_op.reshape: (-1x785x768xf32) <- (-1x785x12x64xf32, 3xi64)
        reshape_16 = paddle._C_ops.reshape(transpose_26, full_int_array_7)

        # pd_op.matmul: (-1x785x768xf32) <- (-1x785x768xf32, 768x768xf32)
        matmul_45 = paddle._C_ops.matmul(reshape_16, parameter_79, False, False)
        del parameter_79

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, 768xf32)
        add_44 = paddle._C_ops.add(matmul_45, parameter_78)
        del parameter_78

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, -1x785x768xf32)
        add_45 = paddle._C_ops.add(add_42, add_44)

        # pd_op.layer_norm: (-1x785x768xf32, -1x785xf32, -1x785xf32) <- (-1x785x768xf32, 768xf32, 768xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_45, parameter_77, parameter_76, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_76, parameter_77

        # pd_op.matmul: (-1x785x3072xf32) <- (-1x785x768xf32, 768x3072xf32)
        matmul_46 = paddle._C_ops.matmul(layer_norm_48, parameter_75, False, False)
        del parameter_75

        # pd_op.add: (-1x785x3072xf32) <- (-1x785x3072xf32, 3072xf32)
        add_46 = paddle._C_ops.add(matmul_46, parameter_74)
        del parameter_74

        # pd_op.gelu: (-1x785x3072xf32) <- (-1x785x3072xf32)
        gelu_7 = paddle._C_ops.gelu(add_46, False)

        # pd_op.matmul: (-1x785x768xf32) <- (-1x785x3072xf32, 3072x768xf32)
        matmul_47 = paddle._C_ops.matmul(gelu_7, parameter_73, False, False)
        del parameter_73

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, 768xf32)
        add_47 = paddle._C_ops.add(matmul_47, parameter_72)
        del parameter_72

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, -1x785x768xf32)
        add_48 = paddle._C_ops.add(add_45, add_47)

        # pd_op.layer_norm: (-1x785x768xf32, -1x785xf32, -1x785xf32) <- (-1x785x768xf32, 768xf32, 768xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_48, parameter_71, parameter_70, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_70, parameter_71

        # pd_op.shape64: (3xi64) <- (-1x785x768xf32)
        shape64_9 = paddle._C_ops.shape64(layer_norm_51)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_35 = paddle._C_ops.slice(
            shape64_9, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_9

        # pd_op.matmul: (-1x785x2304xf32) <- (-1x785x768xf32, 768x2304xf32)
        matmul_48 = paddle._C_ops.matmul(layer_norm_51, parameter_69, False, False)
        del parameter_69

        # pd_op.add: (-1x785x2304xf32) <- (-1x785x2304xf32, 2304xf32)
        add_49 = paddle._C_ops.add(matmul_48, parameter_68)
        del parameter_68

        # pd_op.reshape: (-1x785x3x12x64xf32) <- (-1x785x2304xf32, 5xi64)
        reshape_17 = paddle._C_ops.reshape(add_49, full_int_array_4)

        # pd_op.transpose: (3x-1x12x785x64xf32) <- (-1x785x3x12x64xf32)
        transpose_27 = paddle._C_ops.transpose(reshape_17, [2, 0, 3, 1, 4])
        del reshape_17

        # pd_op.slice: (-1x12x785x64xf32) <- (3x-1x12x785x64xf32, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(
            transpose_27, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x12x785x64xf32) <- (3x-1x12x785x64xf32, 1xi64, 1xi64)
        slice_37 = paddle._C_ops.slice(
            transpose_27, [0], full_int_array_1, full_int_array_5, [1], [0]
        )

        # pd_op.slice: (-1x12x785x64xf32) <- (3x-1x12x785x64xf32, 1xi64, 1xi64)
        slice_38 = paddle._C_ops.slice(
            transpose_27, [0], full_int_array_5, full_int_array_6, [1], [0]
        )

        # pd_op.transpose: (-1x12x64x785xf32) <- (-1x12x785x64xf32)
        transpose_28 = paddle._C_ops.transpose(slice_37, [0, 1, 3, 2])
        del slice_37

        # pd_op.matmul: (-1x12x785x785xf32) <- (-1x12x785x64xf32, -1x12x64x785xf32)
        matmul_49 = paddle._C_ops.matmul(slice_36, transpose_28, False, False)

        # pd_op.scale: (-1x12x785x785xf32) <- (-1x12x785x785xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(matmul_49, full_2, float("0"), True)
        del matmul_49

        # pd_op.softmax: (-1x12x785x785xf32) <- (-1x12x785x785xf32)
        softmax_8 = paddle._C_ops.softmax(scale_8, -1)
        del scale_8

        # pd_op.matmul: (-1x12x785x64xf32) <- (-1x12x785x785xf32, -1x12x785x64xf32)
        matmul_50 = paddle._C_ops.matmul(softmax_8, slice_38, False, False)

        # pd_op.transpose: (-1x785x12x64xf32) <- (-1x12x785x64xf32)
        transpose_29 = paddle._C_ops.transpose(matmul_50, [0, 2, 1, 3])
        del matmul_50

        # pd_op.reshape: (-1x785x768xf32) <- (-1x785x12x64xf32, 3xi64)
        reshape_18 = paddle._C_ops.reshape(transpose_29, full_int_array_7)

        # pd_op.matmul: (-1x785x768xf32) <- (-1x785x768xf32, 768x768xf32)
        matmul_51 = paddle._C_ops.matmul(reshape_18, parameter_67, False, False)
        del parameter_67

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, 768xf32)
        add_50 = paddle._C_ops.add(matmul_51, parameter_66)
        del parameter_66

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, -1x785x768xf32)
        add_51 = paddle._C_ops.add(add_48, add_50)

        # pd_op.layer_norm: (-1x785x768xf32, -1x785xf32, -1x785xf32) <- (-1x785x768xf32, 768xf32, 768xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_51, parameter_65, parameter_64, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_64, parameter_65

        # pd_op.matmul: (-1x785x3072xf32) <- (-1x785x768xf32, 768x3072xf32)
        matmul_52 = paddle._C_ops.matmul(layer_norm_54, parameter_63, False, False)
        del parameter_63

        # pd_op.add: (-1x785x3072xf32) <- (-1x785x3072xf32, 3072xf32)
        add_52 = paddle._C_ops.add(matmul_52, parameter_62)
        del parameter_62

        # pd_op.gelu: (-1x785x3072xf32) <- (-1x785x3072xf32)
        gelu_8 = paddle._C_ops.gelu(add_52, False)

        # pd_op.matmul: (-1x785x768xf32) <- (-1x785x3072xf32, 3072x768xf32)
        matmul_53 = paddle._C_ops.matmul(gelu_8, parameter_61, False, False)
        del parameter_61

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, 768xf32)
        add_53 = paddle._C_ops.add(matmul_53, parameter_60)
        del parameter_60

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, -1x785x768xf32)
        add_54 = paddle._C_ops.add(add_51, add_53)

        # pd_op.layer_norm: (-1x785x768xf32, -1x785xf32, -1x785xf32) <- (-1x785x768xf32, 768xf32, 768xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_54, parameter_59, parameter_58, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_58, parameter_59

        # pd_op.shape64: (3xi64) <- (-1x785x768xf32)
        shape64_10 = paddle._C_ops.shape64(layer_norm_57)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_39 = paddle._C_ops.slice(
            shape64_10, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_10

        # pd_op.matmul: (-1x785x2304xf32) <- (-1x785x768xf32, 768x2304xf32)
        matmul_54 = paddle._C_ops.matmul(layer_norm_57, parameter_57, False, False)
        del parameter_57

        # pd_op.add: (-1x785x2304xf32) <- (-1x785x2304xf32, 2304xf32)
        add_55 = paddle._C_ops.add(matmul_54, parameter_56)
        del parameter_56

        # pd_op.reshape: (-1x785x3x12x64xf32) <- (-1x785x2304xf32, 5xi64)
        reshape_19 = paddle._C_ops.reshape(add_55, full_int_array_4)

        # pd_op.transpose: (3x-1x12x785x64xf32) <- (-1x785x3x12x64xf32)
        transpose_30 = paddle._C_ops.transpose(reshape_19, [2, 0, 3, 1, 4])
        del reshape_19

        # pd_op.slice: (-1x12x785x64xf32) <- (3x-1x12x785x64xf32, 1xi64, 1xi64)
        slice_40 = paddle._C_ops.slice(
            transpose_30, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x12x785x64xf32) <- (3x-1x12x785x64xf32, 1xi64, 1xi64)
        slice_41 = paddle._C_ops.slice(
            transpose_30, [0], full_int_array_1, full_int_array_5, [1], [0]
        )

        # pd_op.slice: (-1x12x785x64xf32) <- (3x-1x12x785x64xf32, 1xi64, 1xi64)
        slice_42 = paddle._C_ops.slice(
            transpose_30, [0], full_int_array_5, full_int_array_6, [1], [0]
        )

        # pd_op.transpose: (-1x12x64x785xf32) <- (-1x12x785x64xf32)
        transpose_31 = paddle._C_ops.transpose(slice_41, [0, 1, 3, 2])
        del slice_41

        # pd_op.matmul: (-1x12x785x785xf32) <- (-1x12x785x64xf32, -1x12x64x785xf32)
        matmul_55 = paddle._C_ops.matmul(slice_40, transpose_31, False, False)

        # pd_op.scale: (-1x12x785x785xf32) <- (-1x12x785x785xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(matmul_55, full_2, float("0"), True)
        del matmul_55

        # pd_op.softmax: (-1x12x785x785xf32) <- (-1x12x785x785xf32)
        softmax_9 = paddle._C_ops.softmax(scale_9, -1)
        del scale_9

        # pd_op.matmul: (-1x12x785x64xf32) <- (-1x12x785x785xf32, -1x12x785x64xf32)
        matmul_56 = paddle._C_ops.matmul(softmax_9, slice_42, False, False)

        # pd_op.transpose: (-1x785x12x64xf32) <- (-1x12x785x64xf32)
        transpose_32 = paddle._C_ops.transpose(matmul_56, [0, 2, 1, 3])
        del matmul_56

        # pd_op.reshape: (-1x785x768xf32) <- (-1x785x12x64xf32, 3xi64)
        reshape_20 = paddle._C_ops.reshape(transpose_32, full_int_array_7)

        # pd_op.matmul: (-1x785x768xf32) <- (-1x785x768xf32, 768x768xf32)
        matmul_57 = paddle._C_ops.matmul(reshape_20, parameter_55, False, False)
        del parameter_55

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, 768xf32)
        add_56 = paddle._C_ops.add(matmul_57, parameter_54)
        del parameter_54

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, -1x785x768xf32)
        add_57 = paddle._C_ops.add(add_54, add_56)

        # pd_op.layer_norm: (-1x785x768xf32, -1x785xf32, -1x785xf32) <- (-1x785x768xf32, 768xf32, 768xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_57, parameter_53, parameter_52, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_52, parameter_53

        # pd_op.matmul: (-1x785x3072xf32) <- (-1x785x768xf32, 768x3072xf32)
        matmul_58 = paddle._C_ops.matmul(layer_norm_60, parameter_51, False, False)
        del parameter_51

        # pd_op.add: (-1x785x3072xf32) <- (-1x785x3072xf32, 3072xf32)
        add_58 = paddle._C_ops.add(matmul_58, parameter_50)
        del parameter_50

        # pd_op.gelu: (-1x785x3072xf32) <- (-1x785x3072xf32)
        gelu_9 = paddle._C_ops.gelu(add_58, False)

        # pd_op.matmul: (-1x785x768xf32) <- (-1x785x3072xf32, 3072x768xf32)
        matmul_59 = paddle._C_ops.matmul(gelu_9, parameter_49, False, False)
        del parameter_49

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, 768xf32)
        add_59 = paddle._C_ops.add(matmul_59, parameter_48)
        del parameter_48

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, -1x785x768xf32)
        add_60 = paddle._C_ops.add(add_57, add_59)

        # pd_op.layer_norm: (-1x785x768xf32, -1x785xf32, -1x785xf32) <- (-1x785x768xf32, 768xf32, 768xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_60, parameter_47, parameter_46, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_46, parameter_47

        # pd_op.shape64: (3xi64) <- (-1x785x768xf32)
        shape64_11 = paddle._C_ops.shape64(layer_norm_63)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_43 = paddle._C_ops.slice(
            shape64_11, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_11

        # pd_op.matmul: (-1x785x2304xf32) <- (-1x785x768xf32, 768x2304xf32)
        matmul_60 = paddle._C_ops.matmul(layer_norm_63, parameter_45, False, False)
        del parameter_45

        # pd_op.add: (-1x785x2304xf32) <- (-1x785x2304xf32, 2304xf32)
        add_61 = paddle._C_ops.add(matmul_60, parameter_44)
        del parameter_44

        # pd_op.reshape: (-1x785x3x12x64xf32) <- (-1x785x2304xf32, 5xi64)
        reshape_21 = paddle._C_ops.reshape(add_61, full_int_array_4)

        # pd_op.transpose: (3x-1x12x785x64xf32) <- (-1x785x3x12x64xf32)
        transpose_33 = paddle._C_ops.transpose(reshape_21, [2, 0, 3, 1, 4])
        del reshape_21

        # pd_op.slice: (-1x12x785x64xf32) <- (3x-1x12x785x64xf32, 1xi64, 1xi64)
        slice_44 = paddle._C_ops.slice(
            transpose_33, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x12x785x64xf32) <- (3x-1x12x785x64xf32, 1xi64, 1xi64)
        slice_45 = paddle._C_ops.slice(
            transpose_33, [0], full_int_array_1, full_int_array_5, [1], [0]
        )

        # pd_op.slice: (-1x12x785x64xf32) <- (3x-1x12x785x64xf32, 1xi64, 1xi64)
        slice_46 = paddle._C_ops.slice(
            transpose_33, [0], full_int_array_5, full_int_array_6, [1], [0]
        )

        # pd_op.transpose: (-1x12x64x785xf32) <- (-1x12x785x64xf32)
        transpose_34 = paddle._C_ops.transpose(slice_45, [0, 1, 3, 2])
        del slice_45

        # pd_op.matmul: (-1x12x785x785xf32) <- (-1x12x785x64xf32, -1x12x64x785xf32)
        matmul_61 = paddle._C_ops.matmul(slice_44, transpose_34, False, False)

        # pd_op.scale: (-1x12x785x785xf32) <- (-1x12x785x785xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(matmul_61, full_2, float("0"), True)
        del matmul_61

        # pd_op.softmax: (-1x12x785x785xf32) <- (-1x12x785x785xf32)
        softmax_10 = paddle._C_ops.softmax(scale_10, -1)
        del scale_10

        # pd_op.matmul: (-1x12x785x64xf32) <- (-1x12x785x785xf32, -1x12x785x64xf32)
        matmul_62 = paddle._C_ops.matmul(softmax_10, slice_46, False, False)

        # pd_op.transpose: (-1x785x12x64xf32) <- (-1x12x785x64xf32)
        transpose_35 = paddle._C_ops.transpose(matmul_62, [0, 2, 1, 3])
        del matmul_62

        # pd_op.reshape: (-1x785x768xf32) <- (-1x785x12x64xf32, 3xi64)
        reshape_22 = paddle._C_ops.reshape(transpose_35, full_int_array_7)

        # pd_op.matmul: (-1x785x768xf32) <- (-1x785x768xf32, 768x768xf32)
        matmul_63 = paddle._C_ops.matmul(reshape_22, parameter_43, False, False)
        del parameter_43

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, 768xf32)
        add_62 = paddle._C_ops.add(matmul_63, parameter_42)
        del parameter_42

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, -1x785x768xf32)
        add_63 = paddle._C_ops.add(add_60, add_62)

        # pd_op.layer_norm: (-1x785x768xf32, -1x785xf32, -1x785xf32) <- (-1x785x768xf32, 768xf32, 768xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_63, parameter_41, parameter_40, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_40, parameter_41

        # pd_op.matmul: (-1x785x3072xf32) <- (-1x785x768xf32, 768x3072xf32)
        matmul_64 = paddle._C_ops.matmul(layer_norm_66, parameter_39, False, False)
        del parameter_39

        # pd_op.add: (-1x785x3072xf32) <- (-1x785x3072xf32, 3072xf32)
        add_64 = paddle._C_ops.add(matmul_64, parameter_38)
        del parameter_38

        # pd_op.gelu: (-1x785x3072xf32) <- (-1x785x3072xf32)
        gelu_10 = paddle._C_ops.gelu(add_64, False)

        # pd_op.matmul: (-1x785x768xf32) <- (-1x785x3072xf32, 3072x768xf32)
        matmul_65 = paddle._C_ops.matmul(gelu_10, parameter_37, False, False)
        del parameter_37

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, 768xf32)
        add_65 = paddle._C_ops.add(matmul_65, parameter_36)
        del parameter_36

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, -1x785x768xf32)
        add_66 = paddle._C_ops.add(add_63, add_65)

        # pd_op.layer_norm: (-1x785x768xf32, -1x785xf32, -1x785xf32) <- (-1x785x768xf32, 768xf32, 768xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_66, parameter_35, parameter_34, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_34, parameter_35

        # pd_op.shape64: (3xi64) <- (-1x785x768xf32)
        shape64_12 = paddle._C_ops.shape64(layer_norm_69)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_47 = paddle._C_ops.slice(
            shape64_12, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_12

        # pd_op.matmul: (-1x785x2304xf32) <- (-1x785x768xf32, 768x2304xf32)
        matmul_66 = paddle._C_ops.matmul(layer_norm_69, parameter_33, False, False)
        del parameter_33

        # pd_op.add: (-1x785x2304xf32) <- (-1x785x2304xf32, 2304xf32)
        add_67 = paddle._C_ops.add(matmul_66, parameter_32)
        del parameter_32

        # pd_op.reshape: (-1x785x3x12x64xf32) <- (-1x785x2304xf32, 5xi64)
        reshape_23 = paddle._C_ops.reshape(add_67, full_int_array_4)
        del full_int_array_4

        # pd_op.transpose: (3x-1x12x785x64xf32) <- (-1x785x3x12x64xf32)
        transpose_36 = paddle._C_ops.transpose(reshape_23, [2, 0, 3, 1, 4])
        del reshape_23

        # pd_op.slice: (-1x12x785x64xf32) <- (3x-1x12x785x64xf32, 1xi64, 1xi64)
        slice_48 = paddle._C_ops.slice(
            transpose_36, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x12x785x64xf32) <- (3x-1x12x785x64xf32, 1xi64, 1xi64)
        slice_49 = paddle._C_ops.slice(
            transpose_36, [0], full_int_array_1, full_int_array_5, [1], [0]
        )

        # pd_op.slice: (-1x12x785x64xf32) <- (3x-1x12x785x64xf32, 1xi64, 1xi64)
        slice_50 = paddle._C_ops.slice(
            transpose_36, [0], full_int_array_5, full_int_array_6, [1], [0]
        )

        # pd_op.transpose: (-1x12x64x785xf32) <- (-1x12x785x64xf32)
        transpose_37 = paddle._C_ops.transpose(slice_49, [0, 1, 3, 2])
        del slice_49

        # pd_op.matmul: (-1x12x785x785xf32) <- (-1x12x785x64xf32, -1x12x64x785xf32)
        matmul_67 = paddle._C_ops.matmul(slice_48, transpose_37, False, False)

        # pd_op.scale: (-1x12x785x785xf32) <- (-1x12x785x785xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(matmul_67, full_2, float("0"), True)
        del matmul_67

        # pd_op.softmax: (-1x12x785x785xf32) <- (-1x12x785x785xf32)
        softmax_11 = paddle._C_ops.softmax(scale_11, -1)
        del scale_11

        # pd_op.matmul: (-1x12x785x64xf32) <- (-1x12x785x785xf32, -1x12x785x64xf32)
        matmul_68 = paddle._C_ops.matmul(softmax_11, slice_50, False, False)

        # pd_op.transpose: (-1x785x12x64xf32) <- (-1x12x785x64xf32)
        transpose_38 = paddle._C_ops.transpose(matmul_68, [0, 2, 1, 3])
        del matmul_68

        # pd_op.reshape: (-1x785x768xf32) <- (-1x785x12x64xf32, 3xi64)
        reshape_24 = paddle._C_ops.reshape(transpose_38, full_int_array_7)
        del full_int_array_7

        # pd_op.matmul: (-1x785x768xf32) <- (-1x785x768xf32, 768x768xf32)
        matmul_69 = paddle._C_ops.matmul(reshape_24, parameter_31, False, False)
        del parameter_31

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, 768xf32)
        add_68 = paddle._C_ops.add(matmul_69, parameter_30)
        del parameter_30

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, -1x785x768xf32)
        add_69 = paddle._C_ops.add(add_66, add_68)

        # pd_op.layer_norm: (-1x785x768xf32, -1x785xf32, -1x785xf32) <- (-1x785x768xf32, 768xf32, 768xf32)
        layer_norm_72, layer_norm_73, layer_norm_74 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_69, parameter_29, parameter_28, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_28, parameter_29

        # pd_op.matmul: (-1x785x3072xf32) <- (-1x785x768xf32, 768x3072xf32)
        matmul_70 = paddle._C_ops.matmul(layer_norm_72, parameter_27, False, False)
        del parameter_27

        # pd_op.add: (-1x785x3072xf32) <- (-1x785x3072xf32, 3072xf32)
        add_70 = paddle._C_ops.add(matmul_70, parameter_26)
        del parameter_26

        # pd_op.gelu: (-1x785x3072xf32) <- (-1x785x3072xf32)
        gelu_11 = paddle._C_ops.gelu(add_70, False)

        # pd_op.matmul: (-1x785x768xf32) <- (-1x785x3072xf32, 3072x768xf32)
        matmul_71 = paddle._C_ops.matmul(gelu_11, parameter_25, False, False)
        del parameter_25

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, 768xf32)
        add_71 = paddle._C_ops.add(matmul_71, parameter_24)
        del parameter_24

        # pd_op.add: (-1x785x768xf32) <- (-1x785x768xf32, -1x785x768xf32)
        add_72 = paddle._C_ops.add(add_69, add_71)

        # pd_op.slice: (-1x784x768xf32) <- (-1x785x768xf32, 1xi64, 1xi64)
        slice_51 = paddle._C_ops.slice(
            add_72, [1], full_int_array_1, full_int_array_2, [1], []
        )

        # pd_op.slice: (-1x768xf32) <- (-1x784x768xf32, 1xi64, 1xi64)
        slice_52 = paddle._C_ops.slice(
            slice_51, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.slice: (-1x783x768xf32) <- (-1x784x768xf32, 1xi64, 1xi64)
        slice_53 = paddle._C_ops.slice(
            slice_51, [1], full_int_array_1, full_int_array_2, [1], []
        )

        # pd_op.layer_norm: (-1x768xf32, -1xf32, -1xf32) <- (-1x768xf32, 768xf32, 768xf32)
        layer_norm_75, layer_norm_76, layer_norm_77 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                slice_52, parameter_23, parameter_22, float("1e-05"), 1
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_22, parameter_23

        # pd_op.shape64: (2xi64) <- (-1x768xf32)
        shape64_13 = paddle._C_ops.shape64(layer_norm_75)

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_54 = paddle._C_ops.slice(
            shape64_13, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_13

        # pd_op.full: (xi64) <- ()
        full_3 = paddle._C_ops.full(
            [], float("768"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_4 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_3 = [slice_54, full_3, full_0, full_4]
        del full_0, full_3, slice_54

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_3, 0)
        del combine_3

        # pd_op.reshape: (-1x768x-1x1xf32) <- (-1x768xf32, 4xi64)
        reshape_25 = paddle._C_ops.reshape(layer_norm_75, stack_1)
        del stack_1

        # pd_op.conv2d: (-1x768x-1x1xf32) <- (-1x768x-1x1xf32, 768x768x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            reshape_25, parameter_21, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_21

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_8 = [1, -1, 1, 1]

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_26 = paddle._C_ops.reshape(parameter_20, full_int_array_8)
        del parameter_20

        # pd_op.add: (-1x768x-1x1xf32) <- (-1x768x-1x1xf32, 1x768x1x1xf32)
        add_73 = paddle._C_ops.add(conv2d_1, reshape_26)

        # pd_op.relu: (-1x768x-1x1xf32) <- (-1x768x-1x1xf32)
        relu_0 = paddle._C_ops.relu(add_73)
        del add_73

        # pd_op.flatten: (-1x768x-1xf32) <- (-1x768x-1x1xf32)
        flatten_3 = paddle._C_ops.flatten(relu_0, 2, 3)

        # pd_op.transpose: (-1x-1x768xf32) <- (-1x768x-1xf32)
        transpose_39 = paddle._C_ops.transpose(flatten_3, [0, 2, 1])
        del flatten_3

        # pd_op.unsqueeze: (1x33x768xf32) <- (33x768xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(data_2, full_int_array_0)
        del data_2

        # pd_op.shape64: (4xi64) <- (-1x768x-1x1xf32)
        shape64_14 = paddle._C_ops.shape64(reshape_25)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_55 = paddle._C_ops.slice(
            shape64_14, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del full_int_array_0, full_int_array_1

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_56 = paddle._C_ops.slice(
            shape64_14, [0], full_int_array_5, full_int_array_6, [1], [0]
        )
        del shape64_14

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_4 = [slice_55, full_4, full_4]
        del full_4, slice_55

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_2 = paddle._C_ops.stack(combine_4, 0)
        del combine_4

        # pd_op.tile: (-1x33x768xf32) <- (1x33x768xf32, 3xi64)
        tile_0 = paddle._C_ops.tile(unsqueeze_0, stack_2)
        del stack_2, unsqueeze_0

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_89 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_90 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_91 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_92 = full_5

        # pd_op.dropout: (-1x33x768xf32, -1x33x768xui8) <- (-1x33x768xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                tile_0, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.add: (-1x33x768xf32) <- (-1x33x768xf32, -1x33x768xf32)
        add_74 = paddle._C_ops.add(tile_0, dropout_0)
        del dropout_0, tile_0

        # pd_op.layer_norm: (-1x33x768xf32, -1x33xf32, -1x33xf32) <- (-1x33x768xf32, 768xf32, 768xf32)
        layer_norm_78, layer_norm_79, layer_norm_80 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_74, parameter_19, parameter_18, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_18, parameter_19

        # pd_op.matmul: (-1x33x768xf32) <- (-1x33x768xf32, 768x768xf32)
        matmul_72 = paddle._C_ops.matmul(layer_norm_78, parameter_17, False, False)
        del parameter_17

        # pd_op.add: (-1x33x768xf32) <- (-1x33x768xf32, 768xf32)
        add_75 = paddle._C_ops.add(matmul_72, parameter_16)
        del parameter_16

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_9 = [0, 0, 8, 96]

        # pd_op.reshape: (-1x33x8x96xf32) <- (-1x33x768xf32, 4xi64)
        reshape_27 = paddle._C_ops.reshape(add_75, full_int_array_9)

        # pd_op.transpose: (-1x8x33x96xf32) <- (-1x33x8x96xf32)
        transpose_40 = paddle._C_ops.transpose(reshape_27, [0, 2, 1, 3])
        del reshape_27

        # pd_op.matmul: (-1x-1x768xf32) <- (-1x-1x768xf32, 768x768xf32)
        matmul_73 = paddle._C_ops.matmul(transpose_39, parameter_15, False, False)
        del parameter_15

        # pd_op.add: (-1x-1x768xf32) <- (-1x-1x768xf32, 768xf32)
        add_76 = paddle._C_ops.add(matmul_73, parameter_14)
        del parameter_14

        # pd_op.matmul: (-1x-1x768xf32) <- (-1x-1x768xf32, 768x768xf32)
        matmul_74 = paddle._C_ops.matmul(transpose_39, parameter_13, False, False)
        del parameter_13

        # pd_op.add: (-1x-1x768xf32) <- (-1x-1x768xf32, 768xf32)
        add_77 = paddle._C_ops.add(matmul_74, parameter_12)
        del parameter_12

        # pd_op.reshape: (-1x-1x8x96xf32) <- (-1x-1x768xf32, 4xi64)
        reshape_28 = paddle._C_ops.reshape(add_76, full_int_array_9)

        # pd_op.transpose: (-1x8x-1x96xf32) <- (-1x-1x8x96xf32)
        transpose_41 = paddle._C_ops.transpose(reshape_28, [0, 2, 1, 3])
        del reshape_28

        # pd_op.reshape: (-1x-1x8x96xf32) <- (-1x-1x768xf32, 4xi64)
        reshape_29 = paddle._C_ops.reshape(add_77, full_int_array_9)
        del full_int_array_9

        # pd_op.transpose: (-1x8x-1x96xf32) <- (-1x-1x8x96xf32)
        transpose_42 = paddle._C_ops.transpose(reshape_29, [0, 2, 1, 3])
        del reshape_29

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("0.102062"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x8x33x96xf32) <- (-1x8x33x96xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(transpose_40, full_6, float("0"), True)
        del transpose_40

        # pd_op.matmul: (-1x8x33x-1xf32) <- (-1x8x33x96xf32, -1x8x-1x96xf32)
        matmul_75 = paddle._C_ops.matmul(scale_12, transpose_41, False, True)

        # pd_op.softmax: (-1x8x33x-1xf32) <- (-1x8x33x-1xf32)
        softmax_12 = paddle._C_ops.softmax(matmul_75, -1)
        del matmul_75

        # pd_op.dropout: (-1x8x33x-1xf32, -1x8x33x-1xui8) <- (-1x8x33x-1xf32, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_12, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (-1x8x33x96xf32) <- (-1x8x33x-1xf32, -1x8x-1x96xf32)
        matmul_76 = paddle._C_ops.matmul(dropout_2, transpose_42, False, False)

        # pd_op.transpose: (-1x33x8x96xf32) <- (-1x8x33x96xf32)
        transpose_43 = paddle._C_ops.transpose(matmul_76, [0, 2, 1, 3])
        del matmul_76

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_10 = [0, 0, 768]

        # pd_op.reshape: (-1x33x768xf32) <- (-1x33x8x96xf32, 3xi64)
        reshape_30 = paddle._C_ops.reshape(transpose_43, full_int_array_10)
        del full_int_array_10

        # pd_op.matmul: (-1x33x768xf32) <- (-1x33x768xf32, 768x768xf32)
        matmul_77 = paddle._C_ops.matmul(reshape_30, parameter_11, False, False)
        del parameter_11

        # pd_op.add: (-1x33x768xf32) <- (-1x33x768xf32, 768xf32)
        add_78 = paddle._C_ops.add(matmul_77, parameter_10)
        del parameter_10

        # pd_op.dropout: (-1x33x768xf32, -1x33x768xui8) <- (-1x33x768xf32, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_78, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_78

        # pd_op.add: (-1x33x768xf32) <- (-1x33x768xf32, -1x33x768xf32)
        add_79 = paddle._C_ops.add(layer_norm_78, dropout_4)

        # pd_op.layer_norm: (-1x33x768xf32, -1x33xf32, -1x33xf32) <- (-1x33x768xf32, 768xf32, 768xf32)
        layer_norm_81, layer_norm_82, layer_norm_83 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_79, parameter_9, parameter_8, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_8, parameter_9

        # pd_op.matmul: (-1x33x2048xf32) <- (-1x33x768xf32, 768x2048xf32)
        matmul_78 = paddle._C_ops.matmul(layer_norm_81, parameter_7, False, False)
        del parameter_7

        # pd_op.add: (-1x33x2048xf32) <- (-1x33x2048xf32, 2048xf32)
        add_80 = paddle._C_ops.add(matmul_78, parameter_6)
        del parameter_6

        # pd_op.relu: (-1x33x2048xf32) <- (-1x33x2048xf32)
        relu_1 = paddle._C_ops.relu(add_80)
        del add_80

        # pd_op.dropout: (-1x33x2048xf32, -1x33x2048xui8) <- (-1x33x2048xf32, None, 1xf32)
        dropout_6, dropout_7 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                relu_1, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (-1x33x768xf32) <- (-1x33x2048xf32, 2048x768xf32)
        matmul_79 = paddle._C_ops.matmul(dropout_6, parameter_5, False, False)
        del parameter_5

        # pd_op.add: (-1x33x768xf32) <- (-1x33x768xf32, 768xf32)
        add_81 = paddle._C_ops.add(matmul_79, parameter_4)
        del parameter_4

        # pd_op.dropout: (-1x33x768xf32, -1x33x768xui8) <- (-1x33x768xf32, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_81, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_81, full_5

        # pd_op.add: (-1x33x768xf32) <- (-1x33x768xf32, -1x33x768xf32)
        add_82 = paddle._C_ops.add(layer_norm_81, dropout_8)

        # pd_op.layer_norm: (-1x33x768xf32, -1x33xf32, -1x33xf32) <- (-1x33x768xf32, 768xf32, 768xf32)
        layer_norm_84, layer_norm_85, layer_norm_86 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_82, parameter_3, parameter_2, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_2, parameter_3

        # pd_op.flatten: (-1x25344xf32) <- (-1x33x768xf32)
        flatten_4 = paddle._C_ops.flatten(layer_norm_84, 1, 2)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_11 = [2, 3]

        # pd_op.unsqueeze: (-1x25344x1x1xf32) <- (-1x25344xf32, 2xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(flatten_4, full_int_array_11)

        # pd_op.conv2d: (-1x33x1x1xf32) <- (-1x25344x1x1xf32, 33x768x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            unsqueeze_1, parameter_1, [1, 1], [0, 0], "EXPLICIT", [1, 1], 33, "NCHW"
        )
        del parameter_1

        # pd_op.reshape: (1x33x1x1xf32) <- (33xf32, 4xi64)
        reshape_31 = paddle._C_ops.reshape(parameter_0, full_int_array_8)
        del full_int_array_8, parameter_0

        # pd_op.add: (-1x33x1x1xf32) <- (-1x33x1x1xf32, 1x33x1x1xf32)
        add_83 = paddle._C_ops.add(conv2d_2, reshape_31)

        # pd_op.flatten: (-1x33xf32) <- (-1x33x1x1xf32)
        flatten_0 = paddle._C_ops.flatten(add_83, 1, 3)
        del (
            add_0,
            add_1,
            add_10,
            add_11,
            add_12,
            add_13,
            add_14,
            add_15,
            add_16,
            add_17,
            add_18,
            add_19,
            add_2,
            add_20,
            add_21,
            add_22,
            add_23,
            add_24,
            add_25,
            add_26,
            add_27,
            add_28,
            add_29,
            add_3,
            add_30,
            add_31,
            add_32,
            add_33,
            add_34,
            add_35,
            add_36,
            add_37,
            add_38,
            add_39,
            add_4,
            add_40,
            add_41,
            add_42,
            add_43,
            add_44,
            add_45,
            add_46,
            add_47,
            add_48,
            add_49,
            add_5,
            add_50,
            add_51,
            add_52,
            add_53,
            add_54,
            add_55,
            add_56,
            add_57,
            add_58,
            add_59,
            add_6,
            add_60,
            add_61,
            add_62,
            add_63,
            add_64,
            add_65,
            add_66,
            add_67,
            add_68,
            add_69,
            add_7,
            add_70,
            add_71,
            add_72,
            add_74,
            add_75,
            add_76,
            add_77,
            add_79,
            add_8,
            add_82,
            add_83,
            add_9,
            assign_0,
            assign_1,
            assign_10,
            assign_11,
            assign_12,
            assign_13,
            assign_14,
            assign_15,
            assign_16,
            assign_17,
            assign_18,
            assign_19,
            assign_2,
            assign_20,
            assign_21,
            assign_22,
            assign_23,
            assign_24,
            assign_25,
            assign_26,
            assign_27,
            assign_28,
            assign_29,
            assign_3,
            assign_30,
            assign_31,
            assign_32,
            assign_33,
            assign_34,
            assign_35,
            assign_36,
            assign_37,
            assign_38,
            assign_39,
            assign_4,
            assign_40,
            assign_41,
            assign_42,
            assign_43,
            assign_44,
            assign_45,
            assign_46,
            assign_47,
            assign_48,
            assign_49,
            assign_5,
            assign_50,
            assign_51,
            assign_52,
            assign_53,
            assign_54,
            assign_55,
            assign_56,
            assign_57,
            assign_58,
            assign_59,
            assign_6,
            assign_60,
            assign_61,
            assign_62,
            assign_63,
            assign_64,
            assign_65,
            assign_66,
            assign_67,
            assign_68,
            assign_69,
            assign_7,
            assign_70,
            assign_71,
            assign_72,
            assign_73,
            assign_74,
            assign_75,
            assign_76,
            assign_77,
            assign_78,
            assign_79,
            assign_8,
            assign_80,
            assign_81,
            assign_82,
            assign_83,
            assign_84,
            assign_85,
            assign_86,
            assign_87,
            assign_88,
            assign_89,
            assign_9,
            assign_90,
            assign_91,
            assign_92,
            bicubic_interp_0,
            cast_0,
            cast_1,
            concat_0,
            concat_1,
            conv2d_0,
            conv2d_1,
            conv2d_2,
            dropout_2,
            dropout_3,
            dropout_4,
            dropout_5,
            dropout_6,
            dropout_7,
            dropout_8,
            dropout_9,
            expand_0,
            flatten_4,
            full_1,
            full_2,
            full_6,
            full_int_array_11,
            full_int_array_2,
            full_int_array_5,
            full_int_array_6,
            gelu_0,
            gelu_1,
            gelu_10,
            gelu_11,
            gelu_2,
            gelu_3,
            gelu_4,
            gelu_5,
            gelu_6,
            gelu_7,
            gelu_8,
            gelu_9,
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
            layer_norm_36,
            layer_norm_37,
            layer_norm_38,
            layer_norm_39,
            layer_norm_4,
            layer_norm_40,
            layer_norm_41,
            layer_norm_42,
            layer_norm_43,
            layer_norm_44,
            layer_norm_45,
            layer_norm_46,
            layer_norm_47,
            layer_norm_48,
            layer_norm_49,
            layer_norm_5,
            layer_norm_50,
            layer_norm_51,
            layer_norm_52,
            layer_norm_53,
            layer_norm_54,
            layer_norm_55,
            layer_norm_56,
            layer_norm_57,
            layer_norm_58,
            layer_norm_59,
            layer_norm_6,
            layer_norm_60,
            layer_norm_61,
            layer_norm_62,
            layer_norm_63,
            layer_norm_64,
            layer_norm_65,
            layer_norm_66,
            layer_norm_67,
            layer_norm_68,
            layer_norm_69,
            layer_norm_7,
            layer_norm_70,
            layer_norm_71,
            layer_norm_72,
            layer_norm_73,
            layer_norm_74,
            layer_norm_75,
            layer_norm_76,
            layer_norm_77,
            layer_norm_78,
            layer_norm_79,
            layer_norm_8,
            layer_norm_80,
            layer_norm_81,
            layer_norm_82,
            layer_norm_83,
            layer_norm_84,
            layer_norm_85,
            layer_norm_86,
            layer_norm_9,
            matmul_0,
            matmul_10,
            matmul_11,
            matmul_12,
            matmul_15,
            matmul_16,
            matmul_17,
            matmul_18,
            matmul_21,
            matmul_22,
            matmul_23,
            matmul_24,
            matmul_27,
            matmul_28,
            matmul_29,
            matmul_3,
            matmul_30,
            matmul_33,
            matmul_34,
            matmul_35,
            matmul_36,
            matmul_39,
            matmul_4,
            matmul_40,
            matmul_41,
            matmul_42,
            matmul_45,
            matmul_46,
            matmul_47,
            matmul_48,
            matmul_5,
            matmul_51,
            matmul_52,
            matmul_53,
            matmul_54,
            matmul_57,
            matmul_58,
            matmul_59,
            matmul_6,
            matmul_60,
            matmul_63,
            matmul_64,
            matmul_65,
            matmul_66,
            matmul_69,
            matmul_70,
            matmul_71,
            matmul_72,
            matmul_73,
            matmul_74,
            matmul_77,
            matmul_78,
            matmul_79,
            matmul_9,
            relu_0,
            relu_1,
            reshape_10,
            reshape_12,
            reshape_14,
            reshape_16,
            reshape_18,
            reshape_2,
            reshape_20,
            reshape_22,
            reshape_24,
            reshape_25,
            reshape_26,
            reshape_30,
            reshape_31,
            reshape_4,
            reshape_6,
            reshape_8,
            scale_12,
            slice_1,
            slice_10,
            slice_12,
            slice_14,
            slice_16,
            slice_18,
            slice_2,
            slice_20,
            slice_22,
            slice_24,
            slice_26,
            slice_28,
            slice_30,
            slice_32,
            slice_34,
            slice_36,
            slice_38,
            slice_4,
            slice_40,
            slice_42,
            slice_44,
            slice_46,
            slice_48,
            slice_50,
            slice_51,
            slice_52,
            slice_6,
            slice_8,
            softmax_0,
            softmax_1,
            softmax_10,
            softmax_11,
            softmax_12,
            softmax_2,
            softmax_3,
            softmax_4,
            softmax_5,
            softmax_6,
            softmax_7,
            softmax_8,
            softmax_9,
            stack_0,
            transpose_0,
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
            transpose_31,
            transpose_32,
            transpose_33,
            transpose_34,
            transpose_35,
            transpose_36,
            transpose_37,
            transpose_38,
            transpose_39,
            transpose_4,
            transpose_41,
            transpose_42,
            transpose_43,
            transpose_5,
            transpose_6,
            transpose_7,
            transpose_8,
            transpose_9,
            unsqueeze_1,
        )

        return flatten_0
