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
        parameter_171,
        parameter_172,
        parameter_173,
        parameter_174,
        parameter_175,
        parameter_176,
        parameter_177,
        parameter_178,
        parameter_179,
        parameter_180,
        parameter_181,
        parameter_182,
        parameter_183,
        parameter_184,
        parameter_185,
        parameter_186,
        parameter_187,
        parameter_188,
        parameter_189,
        parameter_190,
        data_0,
        data_1,
        data_2,
        data_3,
    ):
        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x3x-1x-1xf32, 64x3x7x7xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_3, parameter_190, [4, 4], [3, 3], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_3, parameter_190

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_189, full_int_array_0)
        del parameter_189

        # pd_op.add: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 1x64x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_0, reshape_0)
        del conv2d_0, reshape_0

        # pd_op.shape64: (4xi64) <- (-1x64x-1x-1xf32)
        shape64_0 = paddle._C_ops.shape64(add_0)

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

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [3]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [4]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_4, full_int_array_5, [1], [0]
        )
        del shape64_0

        # pd_op.flatten: (-1x64x-1xf32) <- (-1x64x-1x-1xf32)
        flatten_0 = paddle._C_ops.flatten(add_0, 2, 3)
        del add_0

        # pd_op.transpose: (-1x-1x64xf32) <- (-1x64x-1xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_0, [0, 2, 1])
        del flatten_0

        # pd_op.layer_norm: (-1x-1x64xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x64xf32, 64xf32, 64xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_0, parameter_188, parameter_187, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_187, parameter_188, transpose_0

        # pd_op.layer_norm: (-1x-1x64xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x64xf32, 64xf32, 64xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                layer_norm_0, parameter_186, parameter_185, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_185, parameter_186

        # pd_op.shape64: (3xi64) <- (-1x-1x64xf32)
        shape64_1 = paddle._C_ops.shape64(layer_norm_3)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_1

        # pd_op.matmul: (-1x-1x64xf32) <- (-1x-1x64xf32, 64x64xf32)
        matmul_0 = paddle._C_ops.matmul(layer_norm_3, parameter_184, False, False)
        del parameter_184

        # pd_op.add: (-1x-1x64xf32) <- (-1x-1x64xf32, 64xf32)
        add_1 = paddle._C_ops.add(matmul_0, parameter_183)
        del matmul_0, parameter_183

        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("64"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_0 = [slice_3, slice_4, full_0, full_1]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.reshape: (-1x-1x1x64xf32) <- (-1x-1x64xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(add_1, stack_0)
        del add_1, stack_0

        # pd_op.transpose: (-1x1x-1x64xf32) <- (-1x-1x1x64xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_1, [0, 2, 1, 3])
        del reshape_1

        # pd_op.transpose: (-1x64x-1xf32) <- (-1x-1x64xf32)
        transpose_2 = paddle._C_ops.transpose(layer_norm_3, [0, 2, 1])
        del layer_norm_3

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_1 = [slice_3, full_1, slice_1, slice_2]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_1, 0)
        del combine_1

        # pd_op.reshape: (-1x64x-1x-1xf32) <- (-1x64x-1xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(transpose_2, stack_1)
        del stack_1, transpose_2

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x8x8xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            reshape_2, parameter_182, [8, 8], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_182, reshape_2

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(parameter_181, full_int_array_0)
        del parameter_181

        # pd_op.add: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 1x64x1x1xf32)
        add_2 = paddle._C_ops.add(conv2d_1, reshape_3)
        del conv2d_1, reshape_3

        # pd_op.full: (xi64) <- ()
        full_2 = paddle._C_ops.full(
            [], float("-1"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_2 = [slice_3, full_1, full_2]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_2 = paddle._C_ops.stack(combine_2, 0)
        del combine_2

        # pd_op.reshape: (-1x64x-1xf32) <- (-1x64x-1x-1xf32, 3xi64)
        reshape_4 = paddle._C_ops.reshape(add_2, stack_2)
        del add_2, stack_2

        # pd_op.transpose: (-1x-1x64xf32) <- (-1x64x-1xf32)
        transpose_3 = paddle._C_ops.transpose(reshape_4, [0, 2, 1])
        del reshape_4

        # pd_op.layer_norm: (-1x-1x64xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x64xf32, 64xf32, 64xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_3, parameter_180, parameter_179, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_179, parameter_180, transpose_3

        # pd_op.matmul: (-1x-1x128xf32) <- (-1x-1x64xf32, 64x128xf32)
        matmul_1 = paddle._C_ops.matmul(layer_norm_6, parameter_178, False, False)
        del layer_norm_6, parameter_178

        # pd_op.add: (-1x-1x128xf32) <- (-1x-1x128xf32, 128xf32)
        add_3 = paddle._C_ops.add(matmul_1, parameter_177)
        del matmul_1, parameter_177

        # pd_op.full: (xi64) <- ()
        full_3 = paddle._C_ops.full(
            [], float("2"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_3 = [slice_3, full_2, full_3, full_0, full_1]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_3 = paddle._C_ops.stack(combine_3, 0)
        del combine_3

        # pd_op.reshape: (-1x-1x2x1x64xf32) <- (-1x-1x128xf32, 5xi64)
        reshape_5 = paddle._C_ops.reshape(add_3, stack_3)
        del add_3, stack_3

        # pd_op.transpose: (2x-1x1x-1x64xf32) <- (-1x-1x2x1x64xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_5, [2, 0, 3, 1, 4])
        del reshape_5

        # pd_op.slice: (-1x1x-1x64xf32) <- (2x-1x1x-1x64xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            transpose_4, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (-1x1x-1x64xf32) <- (2x-1x1x-1x64xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            transpose_4, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del transpose_4

        # pd_op.transpose: (-1x1x64x-1xf32) <- (-1x1x-1x64xf32)
        transpose_5 = paddle._C_ops.transpose(slice_5, [0, 1, 3, 2])
        del slice_5

        # pd_op.matmul: (-1x1x-1x-1xf32) <- (-1x1x-1x64xf32, -1x1x64x-1xf32)
        matmul_2 = paddle._C_ops.matmul(transpose_1, transpose_5, False, False)
        del transpose_1, transpose_5

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x1x-1x-1xf32) <- (-1x1x-1x-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(matmul_2, full_4, float("0"), True)
        del matmul_2

        # pd_op.softmax: (-1x1x-1x-1xf32) <- (-1x1x-1x-1xf32)
        softmax_0 = paddle._C_ops.softmax(scale_0, -1)
        del scale_0

        # pd_op.matmul: (-1x1x-1x64xf32) <- (-1x1x-1x-1xf32, -1x1x-1x64xf32)
        matmul_3 = paddle._C_ops.matmul(softmax_0, slice_6, False, False)
        del slice_6, softmax_0

        # pd_op.transpose: (-1x-1x1x64xf32) <- (-1x1x-1x64xf32)
        transpose_6 = paddle._C_ops.transpose(matmul_3, [0, 2, 1, 3])
        del matmul_3

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_4 = [slice_3, slice_4, full_1]
        del slice_3, slice_4

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_4 = paddle._C_ops.stack(combine_4, 0)
        del combine_4

        # pd_op.reshape: (-1x-1x64xf32) <- (-1x-1x1x64xf32, 3xi64)
        reshape_6 = paddle._C_ops.reshape(transpose_6, stack_4)
        del stack_4, transpose_6

        # pd_op.matmul: (-1x-1x64xf32) <- (-1x-1x64xf32, 64x64xf32)
        matmul_4 = paddle._C_ops.matmul(reshape_6, parameter_176, False, False)
        del parameter_176, reshape_6

        # pd_op.add: (-1x-1x64xf32) <- (-1x-1x64xf32, 64xf32)
        add_4 = paddle._C_ops.add(matmul_4, parameter_175)
        del matmul_4, parameter_175

        # pd_op.add: (-1x-1x64xf32) <- (-1x-1x64xf32, -1x-1x64xf32)
        add_5 = paddle._C_ops.add(layer_norm_0, add_4)
        del add_4, layer_norm_0

        # pd_op.layer_norm: (-1x-1x64xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x64xf32, 64xf32, 64xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_5, parameter_174, parameter_173, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_173, parameter_174

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x64xf32, 64x256xf32)
        matmul_5 = paddle._C_ops.matmul(layer_norm_9, parameter_172, False, False)
        del layer_norm_9, parameter_172

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_6 = paddle._C_ops.add(matmul_5, parameter_171)
        del matmul_5, parameter_171

        # pd_op.shape64: (3xi64) <- (-1x-1x256xf32)
        shape64_2 = paddle._C_ops.shape64(add_6)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_2

        # pd_op.transpose: (-1x256x-1xf32) <- (-1x-1x256xf32)
        transpose_7 = paddle._C_ops.transpose(add_6, [0, 2, 1])
        del add_6

        # pd_op.full: (xi64) <- ()
        full_5 = paddle._C_ops.full(
            [], float("256"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_5 = [slice_7, full_5, slice_1, slice_2]
        del slice_7

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_5 = paddle._C_ops.stack(combine_5, 0)
        del combine_5

        # pd_op.reshape: (-1x256x-1x-1xf32) <- (-1x256x-1xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_7, stack_5)
        del stack_5, transpose_7

        # pd_op.depthwise_conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x1x3x3xf32)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(
            reshape_7, parameter_170, [1, 1], [1, 1], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del parameter_170, reshape_7

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(parameter_169, full_int_array_0)
        del parameter_169

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_7 = paddle._C_ops.add(depthwise_conv2d_0, reshape_8)
        del depthwise_conv2d_0, reshape_8

        # pd_op.flatten: (-1x256x-1xf32) <- (-1x256x-1x-1xf32)
        flatten_1 = paddle._C_ops.flatten(add_7, 2, 3)
        del add_7

        # pd_op.transpose: (-1x-1x256xf32) <- (-1x256x-1xf32)
        transpose_8 = paddle._C_ops.transpose(flatten_1, [0, 2, 1])
        del flatten_1

        # pd_op.gelu: (-1x-1x256xf32) <- (-1x-1x256xf32)
        gelu_0 = paddle._C_ops.gelu(transpose_8, False)
        del transpose_8

        # pd_op.matmul: (-1x-1x64xf32) <- (-1x-1x256xf32, 256x64xf32)
        matmul_6 = paddle._C_ops.matmul(gelu_0, parameter_168, False, False)
        del gelu_0, parameter_168

        # pd_op.add: (-1x-1x64xf32) <- (-1x-1x64xf32, 64xf32)
        add_8 = paddle._C_ops.add(matmul_6, parameter_167)
        del matmul_6, parameter_167

        # pd_op.add: (-1x-1x64xf32) <- (-1x-1x64xf32, -1x-1x64xf32)
        add_9 = paddle._C_ops.add(add_5, add_8)
        del add_5, add_8

        # pd_op.layer_norm: (-1x-1x64xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x64xf32, 64xf32, 64xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_9, parameter_166, parameter_165, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_165, parameter_166

        # pd_op.shape64: (3xi64) <- (-1x-1x64xf32)
        shape64_3 = paddle._C_ops.shape64(layer_norm_12)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_3

        # pd_op.matmul: (-1x-1x64xf32) <- (-1x-1x64xf32, 64x64xf32)
        matmul_7 = paddle._C_ops.matmul(layer_norm_12, parameter_164, False, False)
        del parameter_164

        # pd_op.add: (-1x-1x64xf32) <- (-1x-1x64xf32, 64xf32)
        add_10 = paddle._C_ops.add(matmul_7, parameter_163)
        del matmul_7, parameter_163

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_6 = [slice_9, slice_10, full_0, full_1]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_6 = paddle._C_ops.stack(combine_6, 0)
        del combine_6

        # pd_op.reshape: (-1x-1x1x64xf32) <- (-1x-1x64xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(add_10, stack_6)
        del add_10, stack_6

        # pd_op.transpose: (-1x1x-1x64xf32) <- (-1x-1x1x64xf32)
        transpose_9 = paddle._C_ops.transpose(reshape_9, [0, 2, 1, 3])
        del reshape_9

        # pd_op.transpose: (-1x64x-1xf32) <- (-1x-1x64xf32)
        transpose_10 = paddle._C_ops.transpose(layer_norm_12, [0, 2, 1])
        del layer_norm_12

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_7 = [slice_9, full_1, slice_1, slice_2]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_7 = paddle._C_ops.stack(combine_7, 0)
        del combine_7

        # pd_op.reshape: (-1x64x-1x-1xf32) <- (-1x64x-1xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(transpose_10, stack_7)
        del stack_7, transpose_10

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x8x8xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            reshape_10, parameter_162, [8, 8], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_162, reshape_10

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_11 = paddle._C_ops.reshape(parameter_161, full_int_array_0)
        del parameter_161

        # pd_op.add: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 1x64x1x1xf32)
        add_11 = paddle._C_ops.add(conv2d_2, reshape_11)
        del conv2d_2, reshape_11

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_8 = [slice_9, full_1, full_2]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_8 = paddle._C_ops.stack(combine_8, 0)
        del combine_8

        # pd_op.reshape: (-1x64x-1xf32) <- (-1x64x-1x-1xf32, 3xi64)
        reshape_12 = paddle._C_ops.reshape(add_11, stack_8)
        del add_11, stack_8

        # pd_op.transpose: (-1x-1x64xf32) <- (-1x64x-1xf32)
        transpose_11 = paddle._C_ops.transpose(reshape_12, [0, 2, 1])
        del reshape_12

        # pd_op.layer_norm: (-1x-1x64xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x64xf32, 64xf32, 64xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_11, parameter_160, parameter_159, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_159, parameter_160, transpose_11

        # pd_op.matmul: (-1x-1x128xf32) <- (-1x-1x64xf32, 64x128xf32)
        matmul_8 = paddle._C_ops.matmul(layer_norm_15, parameter_158, False, False)
        del layer_norm_15, parameter_158

        # pd_op.add: (-1x-1x128xf32) <- (-1x-1x128xf32, 128xf32)
        add_12 = paddle._C_ops.add(matmul_8, parameter_157)
        del matmul_8, parameter_157

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_9 = [slice_9, full_2, full_3, full_0, full_1]
        del full_0

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_9 = paddle._C_ops.stack(combine_9, 0)
        del combine_9

        # pd_op.reshape: (-1x-1x2x1x64xf32) <- (-1x-1x128xf32, 5xi64)
        reshape_13 = paddle._C_ops.reshape(add_12, stack_9)
        del add_12, stack_9

        # pd_op.transpose: (2x-1x1x-1x64xf32) <- (-1x-1x2x1x64xf32)
        transpose_12 = paddle._C_ops.transpose(reshape_13, [2, 0, 3, 1, 4])
        del reshape_13

        # pd_op.slice: (-1x1x-1x64xf32) <- (2x-1x1x-1x64xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            transpose_12, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (-1x1x-1x64xf32) <- (2x-1x1x-1x64xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            transpose_12, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del transpose_12

        # pd_op.transpose: (-1x1x64x-1xf32) <- (-1x1x-1x64xf32)
        transpose_13 = paddle._C_ops.transpose(slice_11, [0, 1, 3, 2])
        del slice_11

        # pd_op.matmul: (-1x1x-1x-1xf32) <- (-1x1x-1x64xf32, -1x1x64x-1xf32)
        matmul_9 = paddle._C_ops.matmul(transpose_9, transpose_13, False, False)
        del transpose_13, transpose_9

        # pd_op.scale: (-1x1x-1x-1xf32) <- (-1x1x-1x-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(matmul_9, full_4, float("0"), True)
        del matmul_9

        # pd_op.softmax: (-1x1x-1x-1xf32) <- (-1x1x-1x-1xf32)
        softmax_1 = paddle._C_ops.softmax(scale_1, -1)
        del scale_1

        # pd_op.matmul: (-1x1x-1x64xf32) <- (-1x1x-1x-1xf32, -1x1x-1x64xf32)
        matmul_10 = paddle._C_ops.matmul(softmax_1, slice_12, False, False)
        del slice_12, softmax_1

        # pd_op.transpose: (-1x-1x1x64xf32) <- (-1x1x-1x64xf32)
        transpose_14 = paddle._C_ops.transpose(matmul_10, [0, 2, 1, 3])
        del matmul_10

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_10 = [slice_9, slice_10, full_1]
        del slice_10, slice_9

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_10 = paddle._C_ops.stack(combine_10, 0)
        del combine_10

        # pd_op.reshape: (-1x-1x64xf32) <- (-1x-1x1x64xf32, 3xi64)
        reshape_14 = paddle._C_ops.reshape(transpose_14, stack_10)
        del stack_10, transpose_14

        # pd_op.matmul: (-1x-1x64xf32) <- (-1x-1x64xf32, 64x64xf32)
        matmul_11 = paddle._C_ops.matmul(reshape_14, parameter_156, False, False)
        del parameter_156, reshape_14

        # pd_op.add: (-1x-1x64xf32) <- (-1x-1x64xf32, 64xf32)
        add_13 = paddle._C_ops.add(matmul_11, parameter_155)
        del matmul_11, parameter_155

        # pd_op.add: (-1x-1x64xf32) <- (-1x-1x64xf32, -1x-1x64xf32)
        add_14 = paddle._C_ops.add(add_9, add_13)
        del add_13, add_9

        # pd_op.layer_norm: (-1x-1x64xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x64xf32, 64xf32, 64xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_14, parameter_154, parameter_153, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_153, parameter_154

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x64xf32, 64x256xf32)
        matmul_12 = paddle._C_ops.matmul(layer_norm_18, parameter_152, False, False)
        del layer_norm_18, parameter_152

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_15 = paddle._C_ops.add(matmul_12, parameter_151)
        del matmul_12, parameter_151

        # pd_op.shape64: (3xi64) <- (-1x-1x256xf32)
        shape64_4 = paddle._C_ops.shape64(add_15)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            shape64_4, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            shape64_4, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_4

        # pd_op.transpose: (-1x256x-1xf32) <- (-1x-1x256xf32)
        transpose_15 = paddle._C_ops.transpose(add_15, [0, 2, 1])
        del add_15

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_11 = [slice_13, full_5, slice_1, slice_2]
        del full_5, slice_13

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_11 = paddle._C_ops.stack(combine_11, 0)
        del combine_11

        # pd_op.reshape: (-1x256x-1x-1xf32) <- (-1x256x-1xf32, 4xi64)
        reshape_15 = paddle._C_ops.reshape(transpose_15, stack_11)
        del stack_11, transpose_15

        # pd_op.depthwise_conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x1x3x3xf32)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(
            reshape_15, parameter_150, [1, 1], [1, 1], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del parameter_150, reshape_15

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_16 = paddle._C_ops.reshape(parameter_149, full_int_array_0)
        del parameter_149

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_16 = paddle._C_ops.add(depthwise_conv2d_1, reshape_16)
        del depthwise_conv2d_1, reshape_16

        # pd_op.flatten: (-1x256x-1xf32) <- (-1x256x-1x-1xf32)
        flatten_2 = paddle._C_ops.flatten(add_16, 2, 3)
        del add_16

        # pd_op.transpose: (-1x-1x256xf32) <- (-1x256x-1xf32)
        transpose_16 = paddle._C_ops.transpose(flatten_2, [0, 2, 1])
        del flatten_2

        # pd_op.gelu: (-1x-1x256xf32) <- (-1x-1x256xf32)
        gelu_1 = paddle._C_ops.gelu(transpose_16, False)
        del transpose_16

        # pd_op.matmul: (-1x-1x64xf32) <- (-1x-1x256xf32, 256x64xf32)
        matmul_13 = paddle._C_ops.matmul(gelu_1, parameter_148, False, False)
        del gelu_1, parameter_148

        # pd_op.add: (-1x-1x64xf32) <- (-1x-1x64xf32, 64xf32)
        add_17 = paddle._C_ops.add(matmul_13, parameter_147)
        del matmul_13, parameter_147

        # pd_op.add: (-1x-1x64xf32) <- (-1x-1x64xf32, -1x-1x64xf32)
        add_18 = paddle._C_ops.add(add_14, add_17)
        del add_14, add_17

        # pd_op.layer_norm: (-1x-1x64xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x64xf32, 64xf32, 64xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_18, parameter_146, parameter_145, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_18, parameter_145, parameter_146

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_12 = [data_0, slice_1, slice_2, full_1]
        del slice_1, slice_2

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_12 = paddle._C_ops.stack(combine_12, 0)
        del combine_12

        # pd_op.reshape: (-1x-1x-1x64xf32) <- (-1x-1x64xf32, 4xi64)
        reshape_17 = paddle._C_ops.reshape(layer_norm_21, stack_12)
        del layer_norm_21, stack_12

        # pd_op.transpose: (-1x64x-1x-1xf32) <- (-1x-1x-1x64xf32)
        transpose_17 = paddle._C_ops.transpose(reshape_17, [0, 3, 1, 2])
        del reshape_17

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x64x-1x-1xf32, 128x64x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            transpose_17, parameter_144, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_144

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_18 = paddle._C_ops.reshape(parameter_143, full_int_array_0)
        del parameter_143

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_19 = paddle._C_ops.add(conv2d_3, reshape_18)
        del conv2d_3, reshape_18

        # pd_op.shape64: (4xi64) <- (-1x128x-1x-1xf32)
        shape64_5 = paddle._C_ops.shape64(add_19)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            shape64_5, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            shape64_5, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            shape64_5, [0], full_int_array_4, full_int_array_5, [1], [0]
        )
        del shape64_5

        # pd_op.flatten: (-1x128x-1xf32) <- (-1x128x-1x-1xf32)
        flatten_3 = paddle._C_ops.flatten(add_19, 2, 3)
        del add_19

        # pd_op.transpose: (-1x-1x128xf32) <- (-1x128x-1xf32)
        transpose_18 = paddle._C_ops.transpose(flatten_3, [0, 2, 1])
        del flatten_3

        # pd_op.layer_norm: (-1x-1x128xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x128xf32, 128xf32, 128xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_18, parameter_142, parameter_141, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_141, parameter_142, transpose_18

        # pd_op.layer_norm: (-1x-1x128xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x128xf32, 128xf32, 128xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                layer_norm_24, parameter_140, parameter_139, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_139, parameter_140

        # pd_op.shape64: (3xi64) <- (-1x-1x128xf32)
        shape64_6 = paddle._C_ops.shape64(layer_norm_27)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(
            shape64_6, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(
            shape64_6, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_6

        # pd_op.matmul: (-1x-1x128xf32) <- (-1x-1x128xf32, 128x128xf32)
        matmul_14 = paddle._C_ops.matmul(layer_norm_27, parameter_138, False, False)
        del parameter_138

        # pd_op.add: (-1x-1x128xf32) <- (-1x-1x128xf32, 128xf32)
        add_20 = paddle._C_ops.add(matmul_14, parameter_137)
        del matmul_14, parameter_137

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_13 = [slice_18, slice_19, full_3, full_1]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_13 = paddle._C_ops.stack(combine_13, 0)
        del combine_13

        # pd_op.reshape: (-1x-1x2x64xf32) <- (-1x-1x128xf32, 4xi64)
        reshape_19 = paddle._C_ops.reshape(add_20, stack_13)
        del add_20, stack_13

        # pd_op.transpose: (-1x2x-1x64xf32) <- (-1x-1x2x64xf32)
        transpose_19 = paddle._C_ops.transpose(reshape_19, [0, 2, 1, 3])
        del reshape_19

        # pd_op.transpose: (-1x128x-1xf32) <- (-1x-1x128xf32)
        transpose_20 = paddle._C_ops.transpose(layer_norm_27, [0, 2, 1])
        del layer_norm_27

        # pd_op.full: (xi64) <- ()
        full_6 = paddle._C_ops.full(
            [], float("128"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_14 = [slice_18, full_6, slice_16, slice_17]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_14 = paddle._C_ops.stack(combine_14, 0)
        del combine_14

        # pd_op.reshape: (-1x128x-1x-1xf32) <- (-1x128x-1xf32, 4xi64)
        reshape_20 = paddle._C_ops.reshape(transpose_20, stack_14)
        del stack_14, transpose_20

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x128x4x4xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            reshape_20, parameter_136, [4, 4], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_136, reshape_20

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_21 = paddle._C_ops.reshape(parameter_135, full_int_array_0)
        del parameter_135

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_21 = paddle._C_ops.add(conv2d_4, reshape_21)
        del conv2d_4, reshape_21

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_15 = [slice_18, full_6, full_2]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_15 = paddle._C_ops.stack(combine_15, 0)
        del combine_15

        # pd_op.reshape: (-1x128x-1xf32) <- (-1x128x-1x-1xf32, 3xi64)
        reshape_22 = paddle._C_ops.reshape(add_21, stack_15)
        del add_21, stack_15

        # pd_op.transpose: (-1x-1x128xf32) <- (-1x128x-1xf32)
        transpose_21 = paddle._C_ops.transpose(reshape_22, [0, 2, 1])
        del reshape_22

        # pd_op.layer_norm: (-1x-1x128xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x128xf32, 128xf32, 128xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_21, parameter_134, parameter_133, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_133, parameter_134, transpose_21

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x128xf32, 128x256xf32)
        matmul_15 = paddle._C_ops.matmul(layer_norm_30, parameter_132, False, False)
        del layer_norm_30, parameter_132

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_22 = paddle._C_ops.add(matmul_15, parameter_131)
        del matmul_15, parameter_131

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_16 = [slice_18, full_2, full_3, full_3, full_1]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_16 = paddle._C_ops.stack(combine_16, 0)
        del combine_16

        # pd_op.reshape: (-1x-1x2x2x64xf32) <- (-1x-1x256xf32, 5xi64)
        reshape_23 = paddle._C_ops.reshape(add_22, stack_16)
        del add_22, stack_16

        # pd_op.transpose: (2x-1x2x-1x64xf32) <- (-1x-1x2x2x64xf32)
        transpose_22 = paddle._C_ops.transpose(reshape_23, [2, 0, 3, 1, 4])
        del reshape_23

        # pd_op.slice: (-1x2x-1x64xf32) <- (2x-1x2x-1x64xf32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(
            transpose_22, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (-1x2x-1x64xf32) <- (2x-1x2x-1x64xf32, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(
            transpose_22, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del transpose_22

        # pd_op.transpose: (-1x2x64x-1xf32) <- (-1x2x-1x64xf32)
        transpose_23 = paddle._C_ops.transpose(slice_20, [0, 1, 3, 2])
        del slice_20

        # pd_op.matmul: (-1x2x-1x-1xf32) <- (-1x2x-1x64xf32, -1x2x64x-1xf32)
        matmul_16 = paddle._C_ops.matmul(transpose_19, transpose_23, False, False)
        del transpose_19, transpose_23

        # pd_op.scale: (-1x2x-1x-1xf32) <- (-1x2x-1x-1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(matmul_16, full_4, float("0"), True)
        del matmul_16

        # pd_op.softmax: (-1x2x-1x-1xf32) <- (-1x2x-1x-1xf32)
        softmax_2 = paddle._C_ops.softmax(scale_2, -1)
        del scale_2

        # pd_op.matmul: (-1x2x-1x64xf32) <- (-1x2x-1x-1xf32, -1x2x-1x64xf32)
        matmul_17 = paddle._C_ops.matmul(softmax_2, slice_21, False, False)
        del slice_21, softmax_2

        # pd_op.transpose: (-1x-1x2x64xf32) <- (-1x2x-1x64xf32)
        transpose_24 = paddle._C_ops.transpose(matmul_17, [0, 2, 1, 3])
        del matmul_17

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_17 = [slice_18, slice_19, full_6]
        del slice_18, slice_19

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_17 = paddle._C_ops.stack(combine_17, 0)
        del combine_17

        # pd_op.reshape: (-1x-1x128xf32) <- (-1x-1x2x64xf32, 3xi64)
        reshape_24 = paddle._C_ops.reshape(transpose_24, stack_17)
        del stack_17, transpose_24

        # pd_op.matmul: (-1x-1x128xf32) <- (-1x-1x128xf32, 128x128xf32)
        matmul_18 = paddle._C_ops.matmul(reshape_24, parameter_130, False, False)
        del parameter_130, reshape_24

        # pd_op.add: (-1x-1x128xf32) <- (-1x-1x128xf32, 128xf32)
        add_23 = paddle._C_ops.add(matmul_18, parameter_129)
        del matmul_18, parameter_129

        # pd_op.add: (-1x-1x128xf32) <- (-1x-1x128xf32, -1x-1x128xf32)
        add_24 = paddle._C_ops.add(layer_norm_24, add_23)
        del add_23, layer_norm_24

        # pd_op.layer_norm: (-1x-1x128xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x128xf32, 128xf32, 128xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_24, parameter_128, parameter_127, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_127, parameter_128

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x128xf32, 128x512xf32)
        matmul_19 = paddle._C_ops.matmul(layer_norm_33, parameter_126, False, False)
        del layer_norm_33, parameter_126

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_25 = paddle._C_ops.add(matmul_19, parameter_125)
        del matmul_19, parameter_125

        # pd_op.shape64: (3xi64) <- (-1x-1x512xf32)
        shape64_7 = paddle._C_ops.shape64(add_25)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(
            shape64_7, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(
            shape64_7, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_7

        # pd_op.transpose: (-1x512x-1xf32) <- (-1x-1x512xf32)
        transpose_25 = paddle._C_ops.transpose(add_25, [0, 2, 1])
        del add_25

        # pd_op.full: (xi64) <- ()
        full_7 = paddle._C_ops.full(
            [], float("512"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_18 = [slice_22, full_7, slice_16, slice_17]
        del slice_22

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_18 = paddle._C_ops.stack(combine_18, 0)
        del combine_18

        # pd_op.reshape: (-1x512x-1x-1xf32) <- (-1x512x-1xf32, 4xi64)
        reshape_25 = paddle._C_ops.reshape(transpose_25, stack_18)
        del stack_18, transpose_25

        # pd_op.depthwise_conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x1x3x3xf32)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(
            reshape_25, parameter_124, [1, 1], [1, 1], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del parameter_124, reshape_25

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_26 = paddle._C_ops.reshape(parameter_123, full_int_array_0)
        del parameter_123

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 1x512x1x1xf32)
        add_26 = paddle._C_ops.add(depthwise_conv2d_2, reshape_26)
        del depthwise_conv2d_2, reshape_26

        # pd_op.flatten: (-1x512x-1xf32) <- (-1x512x-1x-1xf32)
        flatten_4 = paddle._C_ops.flatten(add_26, 2, 3)
        del add_26

        # pd_op.transpose: (-1x-1x512xf32) <- (-1x512x-1xf32)
        transpose_26 = paddle._C_ops.transpose(flatten_4, [0, 2, 1])
        del flatten_4

        # pd_op.gelu: (-1x-1x512xf32) <- (-1x-1x512xf32)
        gelu_2 = paddle._C_ops.gelu(transpose_26, False)
        del transpose_26

        # pd_op.matmul: (-1x-1x128xf32) <- (-1x-1x512xf32, 512x128xf32)
        matmul_20 = paddle._C_ops.matmul(gelu_2, parameter_122, False, False)
        del gelu_2, parameter_122

        # pd_op.add: (-1x-1x128xf32) <- (-1x-1x128xf32, 128xf32)
        add_27 = paddle._C_ops.add(matmul_20, parameter_121)
        del matmul_20, parameter_121

        # pd_op.add: (-1x-1x128xf32) <- (-1x-1x128xf32, -1x-1x128xf32)
        add_28 = paddle._C_ops.add(add_24, add_27)
        del add_24, add_27

        # pd_op.layer_norm: (-1x-1x128xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x128xf32, 128xf32, 128xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_28, parameter_120, parameter_119, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_119, parameter_120

        # pd_op.shape64: (3xi64) <- (-1x-1x128xf32)
        shape64_8 = paddle._C_ops.shape64(layer_norm_36)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(
            shape64_8, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(
            shape64_8, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_8

        # pd_op.matmul: (-1x-1x128xf32) <- (-1x-1x128xf32, 128x128xf32)
        matmul_21 = paddle._C_ops.matmul(layer_norm_36, parameter_118, False, False)
        del parameter_118

        # pd_op.add: (-1x-1x128xf32) <- (-1x-1x128xf32, 128xf32)
        add_29 = paddle._C_ops.add(matmul_21, parameter_117)
        del matmul_21, parameter_117

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_19 = [slice_24, slice_25, full_3, full_1]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_19 = paddle._C_ops.stack(combine_19, 0)
        del combine_19

        # pd_op.reshape: (-1x-1x2x64xf32) <- (-1x-1x128xf32, 4xi64)
        reshape_27 = paddle._C_ops.reshape(add_29, stack_19)
        del add_29, stack_19

        # pd_op.transpose: (-1x2x-1x64xf32) <- (-1x-1x2x64xf32)
        transpose_27 = paddle._C_ops.transpose(reshape_27, [0, 2, 1, 3])
        del reshape_27

        # pd_op.transpose: (-1x128x-1xf32) <- (-1x-1x128xf32)
        transpose_28 = paddle._C_ops.transpose(layer_norm_36, [0, 2, 1])
        del layer_norm_36

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_20 = [slice_24, full_6, slice_16, slice_17]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_20 = paddle._C_ops.stack(combine_20, 0)
        del combine_20

        # pd_op.reshape: (-1x128x-1x-1xf32) <- (-1x128x-1xf32, 4xi64)
        reshape_28 = paddle._C_ops.reshape(transpose_28, stack_20)
        del stack_20, transpose_28

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x128x4x4xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            reshape_28, parameter_116, [4, 4], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_116, reshape_28

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_29 = paddle._C_ops.reshape(parameter_115, full_int_array_0)
        del parameter_115

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_30 = paddle._C_ops.add(conv2d_5, reshape_29)
        del conv2d_5, reshape_29

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_21 = [slice_24, full_6, full_2]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_21 = paddle._C_ops.stack(combine_21, 0)
        del combine_21

        # pd_op.reshape: (-1x128x-1xf32) <- (-1x128x-1x-1xf32, 3xi64)
        reshape_30 = paddle._C_ops.reshape(add_30, stack_21)
        del add_30, stack_21

        # pd_op.transpose: (-1x-1x128xf32) <- (-1x128x-1xf32)
        transpose_29 = paddle._C_ops.transpose(reshape_30, [0, 2, 1])
        del reshape_30

        # pd_op.layer_norm: (-1x-1x128xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x128xf32, 128xf32, 128xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_29, parameter_114, parameter_113, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_113, parameter_114, transpose_29

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x128xf32, 128x256xf32)
        matmul_22 = paddle._C_ops.matmul(layer_norm_39, parameter_112, False, False)
        del layer_norm_39, parameter_112

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_31 = paddle._C_ops.add(matmul_22, parameter_111)
        del matmul_22, parameter_111

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_22 = [slice_24, full_2, full_3, full_3, full_1]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_22 = paddle._C_ops.stack(combine_22, 0)
        del combine_22

        # pd_op.reshape: (-1x-1x2x2x64xf32) <- (-1x-1x256xf32, 5xi64)
        reshape_31 = paddle._C_ops.reshape(add_31, stack_22)
        del add_31, stack_22

        # pd_op.transpose: (2x-1x2x-1x64xf32) <- (-1x-1x2x2x64xf32)
        transpose_30 = paddle._C_ops.transpose(reshape_31, [2, 0, 3, 1, 4])
        del reshape_31

        # pd_op.slice: (-1x2x-1x64xf32) <- (2x-1x2x-1x64xf32, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(
            transpose_30, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (-1x2x-1x64xf32) <- (2x-1x2x-1x64xf32, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(
            transpose_30, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del transpose_30

        # pd_op.transpose: (-1x2x64x-1xf32) <- (-1x2x-1x64xf32)
        transpose_31 = paddle._C_ops.transpose(slice_26, [0, 1, 3, 2])
        del slice_26

        # pd_op.matmul: (-1x2x-1x-1xf32) <- (-1x2x-1x64xf32, -1x2x64x-1xf32)
        matmul_23 = paddle._C_ops.matmul(transpose_27, transpose_31, False, False)
        del transpose_27, transpose_31

        # pd_op.scale: (-1x2x-1x-1xf32) <- (-1x2x-1x-1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(matmul_23, full_4, float("0"), True)
        del matmul_23

        # pd_op.softmax: (-1x2x-1x-1xf32) <- (-1x2x-1x-1xf32)
        softmax_3 = paddle._C_ops.softmax(scale_3, -1)
        del scale_3

        # pd_op.matmul: (-1x2x-1x64xf32) <- (-1x2x-1x-1xf32, -1x2x-1x64xf32)
        matmul_24 = paddle._C_ops.matmul(softmax_3, slice_27, False, False)
        del slice_27, softmax_3

        # pd_op.transpose: (-1x-1x2x64xf32) <- (-1x2x-1x64xf32)
        transpose_32 = paddle._C_ops.transpose(matmul_24, [0, 2, 1, 3])
        del matmul_24

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_23 = [slice_24, slice_25, full_6]
        del slice_24, slice_25

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_23 = paddle._C_ops.stack(combine_23, 0)
        del combine_23

        # pd_op.reshape: (-1x-1x128xf32) <- (-1x-1x2x64xf32, 3xi64)
        reshape_32 = paddle._C_ops.reshape(transpose_32, stack_23)
        del stack_23, transpose_32

        # pd_op.matmul: (-1x-1x128xf32) <- (-1x-1x128xf32, 128x128xf32)
        matmul_25 = paddle._C_ops.matmul(reshape_32, parameter_110, False, False)
        del parameter_110, reshape_32

        # pd_op.add: (-1x-1x128xf32) <- (-1x-1x128xf32, 128xf32)
        add_32 = paddle._C_ops.add(matmul_25, parameter_109)
        del matmul_25, parameter_109

        # pd_op.add: (-1x-1x128xf32) <- (-1x-1x128xf32, -1x-1x128xf32)
        add_33 = paddle._C_ops.add(add_28, add_32)
        del add_28, add_32

        # pd_op.layer_norm: (-1x-1x128xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x128xf32, 128xf32, 128xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_33, parameter_108, parameter_107, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_107, parameter_108

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x128xf32, 128x512xf32)
        matmul_26 = paddle._C_ops.matmul(layer_norm_42, parameter_106, False, False)
        del layer_norm_42, parameter_106

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_34 = paddle._C_ops.add(matmul_26, parameter_105)
        del matmul_26, parameter_105

        # pd_op.shape64: (3xi64) <- (-1x-1x512xf32)
        shape64_9 = paddle._C_ops.shape64(add_34)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(
            shape64_9, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(
            shape64_9, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_9

        # pd_op.transpose: (-1x512x-1xf32) <- (-1x-1x512xf32)
        transpose_33 = paddle._C_ops.transpose(add_34, [0, 2, 1])
        del add_34

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_24 = [slice_28, full_7, slice_16, slice_17]
        del slice_28

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_24 = paddle._C_ops.stack(combine_24, 0)
        del combine_24

        # pd_op.reshape: (-1x512x-1x-1xf32) <- (-1x512x-1xf32, 4xi64)
        reshape_33 = paddle._C_ops.reshape(transpose_33, stack_24)
        del stack_24, transpose_33

        # pd_op.depthwise_conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x1x3x3xf32)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(
            reshape_33, parameter_104, [1, 1], [1, 1], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del parameter_104, reshape_33

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_34 = paddle._C_ops.reshape(parameter_103, full_int_array_0)
        del parameter_103

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 1x512x1x1xf32)
        add_35 = paddle._C_ops.add(depthwise_conv2d_3, reshape_34)
        del depthwise_conv2d_3, reshape_34

        # pd_op.flatten: (-1x512x-1xf32) <- (-1x512x-1x-1xf32)
        flatten_5 = paddle._C_ops.flatten(add_35, 2, 3)
        del add_35

        # pd_op.transpose: (-1x-1x512xf32) <- (-1x512x-1xf32)
        transpose_34 = paddle._C_ops.transpose(flatten_5, [0, 2, 1])
        del flatten_5

        # pd_op.gelu: (-1x-1x512xf32) <- (-1x-1x512xf32)
        gelu_3 = paddle._C_ops.gelu(transpose_34, False)
        del transpose_34

        # pd_op.matmul: (-1x-1x128xf32) <- (-1x-1x512xf32, 512x128xf32)
        matmul_27 = paddle._C_ops.matmul(gelu_3, parameter_102, False, False)
        del gelu_3, parameter_102

        # pd_op.add: (-1x-1x128xf32) <- (-1x-1x128xf32, 128xf32)
        add_36 = paddle._C_ops.add(matmul_27, parameter_101)
        del matmul_27, parameter_101

        # pd_op.add: (-1x-1x128xf32) <- (-1x-1x128xf32, -1x-1x128xf32)
        add_37 = paddle._C_ops.add(add_33, add_36)
        del add_33, add_36

        # pd_op.layer_norm: (-1x-1x128xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x128xf32, 128xf32, 128xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_37, parameter_100, parameter_99, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_37, parameter_100, parameter_99

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_25 = [data_0, slice_16, slice_17, full_6]
        del full_6, slice_16, slice_17

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_25 = paddle._C_ops.stack(combine_25, 0)
        del combine_25

        # pd_op.reshape: (-1x-1x-1x128xf32) <- (-1x-1x128xf32, 4xi64)
        reshape_35 = paddle._C_ops.reshape(layer_norm_45, stack_25)
        del layer_norm_45, stack_25

        # pd_op.transpose: (-1x128x-1x-1xf32) <- (-1x-1x-1x128xf32)
        transpose_35 = paddle._C_ops.transpose(reshape_35, [0, 3, 1, 2])
        del reshape_35

        # pd_op.conv2d: (-1x320x-1x-1xf32) <- (-1x128x-1x-1xf32, 320x128x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            transpose_35, parameter_98, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_98

        # pd_op.reshape: (1x320x1x1xf32) <- (320xf32, 4xi64)
        reshape_36 = paddle._C_ops.reshape(parameter_97, full_int_array_0)
        del parameter_97

        # pd_op.add: (-1x320x-1x-1xf32) <- (-1x320x-1x-1xf32, 1x320x1x1xf32)
        add_38 = paddle._C_ops.add(conv2d_6, reshape_36)
        del conv2d_6, reshape_36

        # pd_op.shape64: (4xi64) <- (-1x320x-1x-1xf32)
        shape64_10 = paddle._C_ops.shape64(add_38)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(
            shape64_10, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(
            shape64_10, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(
            shape64_10, [0], full_int_array_4, full_int_array_5, [1], [0]
        )
        del shape64_10

        # pd_op.flatten: (-1x320x-1xf32) <- (-1x320x-1x-1xf32)
        flatten_6 = paddle._C_ops.flatten(add_38, 2, 3)
        del add_38

        # pd_op.transpose: (-1x-1x320xf32) <- (-1x320x-1xf32)
        transpose_36 = paddle._C_ops.transpose(flatten_6, [0, 2, 1])
        del flatten_6

        # pd_op.layer_norm: (-1x-1x320xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x320xf32, 320xf32, 320xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_36, parameter_96, parameter_95, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_95, parameter_96, transpose_36

        # pd_op.layer_norm: (-1x-1x320xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x320xf32, 320xf32, 320xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                layer_norm_48, parameter_94, parameter_93, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_93, parameter_94

        # pd_op.shape64: (3xi64) <- (-1x-1x320xf32)
        shape64_11 = paddle._C_ops.shape64(layer_norm_51)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_33 = paddle._C_ops.slice(
            shape64_11, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_34 = paddle._C_ops.slice(
            shape64_11, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_11

        # pd_op.matmul: (-1x-1x320xf32) <- (-1x-1x320xf32, 320x320xf32)
        matmul_28 = paddle._C_ops.matmul(layer_norm_51, parameter_92, False, False)
        del parameter_92

        # pd_op.add: (-1x-1x320xf32) <- (-1x-1x320xf32, 320xf32)
        add_39 = paddle._C_ops.add(matmul_28, parameter_91)
        del matmul_28, parameter_91

        # pd_op.full: (xi64) <- ()
        full_8 = paddle._C_ops.full(
            [], float("5"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_26 = [slice_33, slice_34, full_8, full_1]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_26 = paddle._C_ops.stack(combine_26, 0)
        del combine_26

        # pd_op.reshape: (-1x-1x5x64xf32) <- (-1x-1x320xf32, 4xi64)
        reshape_37 = paddle._C_ops.reshape(add_39, stack_26)
        del add_39, stack_26

        # pd_op.transpose: (-1x5x-1x64xf32) <- (-1x-1x5x64xf32)
        transpose_37 = paddle._C_ops.transpose(reshape_37, [0, 2, 1, 3])
        del reshape_37

        # pd_op.transpose: (-1x320x-1xf32) <- (-1x-1x320xf32)
        transpose_38 = paddle._C_ops.transpose(layer_norm_51, [0, 2, 1])
        del layer_norm_51

        # pd_op.full: (xi64) <- ()
        full_9 = paddle._C_ops.full(
            [], float("320"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_27 = [slice_33, full_9, slice_31, slice_32]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_27 = paddle._C_ops.stack(combine_27, 0)
        del combine_27

        # pd_op.reshape: (-1x320x-1x-1xf32) <- (-1x320x-1xf32, 4xi64)
        reshape_38 = paddle._C_ops.reshape(transpose_38, stack_27)
        del stack_27, transpose_38

        # pd_op.conv2d: (-1x320x-1x-1xf32) <- (-1x320x-1x-1xf32, 320x320x2x2xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            reshape_38, parameter_90, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_90, reshape_38

        # pd_op.reshape: (1x320x1x1xf32) <- (320xf32, 4xi64)
        reshape_39 = paddle._C_ops.reshape(parameter_89, full_int_array_0)
        del parameter_89

        # pd_op.add: (-1x320x-1x-1xf32) <- (-1x320x-1x-1xf32, 1x320x1x1xf32)
        add_40 = paddle._C_ops.add(conv2d_7, reshape_39)
        del conv2d_7, reshape_39

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_28 = [slice_33, full_9, full_2]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_28 = paddle._C_ops.stack(combine_28, 0)
        del combine_28

        # pd_op.reshape: (-1x320x-1xf32) <- (-1x320x-1x-1xf32, 3xi64)
        reshape_40 = paddle._C_ops.reshape(add_40, stack_28)
        del add_40, stack_28

        # pd_op.transpose: (-1x-1x320xf32) <- (-1x320x-1xf32)
        transpose_39 = paddle._C_ops.transpose(reshape_40, [0, 2, 1])
        del reshape_40

        # pd_op.layer_norm: (-1x-1x320xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x320xf32, 320xf32, 320xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_39, parameter_88, parameter_87, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_87, parameter_88, transpose_39

        # pd_op.matmul: (-1x-1x640xf32) <- (-1x-1x320xf32, 320x640xf32)
        matmul_29 = paddle._C_ops.matmul(layer_norm_54, parameter_86, False, False)
        del layer_norm_54, parameter_86

        # pd_op.add: (-1x-1x640xf32) <- (-1x-1x640xf32, 640xf32)
        add_41 = paddle._C_ops.add(matmul_29, parameter_85)
        del matmul_29, parameter_85

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_29 = [slice_33, full_2, full_3, full_8, full_1]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_29 = paddle._C_ops.stack(combine_29, 0)
        del combine_29

        # pd_op.reshape: (-1x-1x2x5x64xf32) <- (-1x-1x640xf32, 5xi64)
        reshape_41 = paddle._C_ops.reshape(add_41, stack_29)
        del add_41, stack_29

        # pd_op.transpose: (2x-1x5x-1x64xf32) <- (-1x-1x2x5x64xf32)
        transpose_40 = paddle._C_ops.transpose(reshape_41, [2, 0, 3, 1, 4])
        del reshape_41

        # pd_op.slice: (-1x5x-1x64xf32) <- (2x-1x5x-1x64xf32, 1xi64, 1xi64)
        slice_35 = paddle._C_ops.slice(
            transpose_40, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (-1x5x-1x64xf32) <- (2x-1x5x-1x64xf32, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(
            transpose_40, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del transpose_40

        # pd_op.transpose: (-1x5x64x-1xf32) <- (-1x5x-1x64xf32)
        transpose_41 = paddle._C_ops.transpose(slice_35, [0, 1, 3, 2])
        del slice_35

        # pd_op.matmul: (-1x5x-1x-1xf32) <- (-1x5x-1x64xf32, -1x5x64x-1xf32)
        matmul_30 = paddle._C_ops.matmul(transpose_37, transpose_41, False, False)
        del transpose_37, transpose_41

        # pd_op.scale: (-1x5x-1x-1xf32) <- (-1x5x-1x-1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(matmul_30, full_4, float("0"), True)
        del matmul_30

        # pd_op.softmax: (-1x5x-1x-1xf32) <- (-1x5x-1x-1xf32)
        softmax_4 = paddle._C_ops.softmax(scale_4, -1)
        del scale_4

        # pd_op.matmul: (-1x5x-1x64xf32) <- (-1x5x-1x-1xf32, -1x5x-1x64xf32)
        matmul_31 = paddle._C_ops.matmul(softmax_4, slice_36, False, False)
        del slice_36, softmax_4

        # pd_op.transpose: (-1x-1x5x64xf32) <- (-1x5x-1x64xf32)
        transpose_42 = paddle._C_ops.transpose(matmul_31, [0, 2, 1, 3])
        del matmul_31

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_30 = [slice_33, slice_34, full_9]
        del slice_33, slice_34

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_30 = paddle._C_ops.stack(combine_30, 0)
        del combine_30

        # pd_op.reshape: (-1x-1x320xf32) <- (-1x-1x5x64xf32, 3xi64)
        reshape_42 = paddle._C_ops.reshape(transpose_42, stack_30)
        del stack_30, transpose_42

        # pd_op.matmul: (-1x-1x320xf32) <- (-1x-1x320xf32, 320x320xf32)
        matmul_32 = paddle._C_ops.matmul(reshape_42, parameter_84, False, False)
        del parameter_84, reshape_42

        # pd_op.add: (-1x-1x320xf32) <- (-1x-1x320xf32, 320xf32)
        add_42 = paddle._C_ops.add(matmul_32, parameter_83)
        del matmul_32, parameter_83

        # pd_op.add: (-1x-1x320xf32) <- (-1x-1x320xf32, -1x-1x320xf32)
        add_43 = paddle._C_ops.add(layer_norm_48, add_42)
        del add_42, layer_norm_48

        # pd_op.layer_norm: (-1x-1x320xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x320xf32, 320xf32, 320xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_43, parameter_82, parameter_81, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_81, parameter_82

        # pd_op.matmul: (-1x-1x1280xf32) <- (-1x-1x320xf32, 320x1280xf32)
        matmul_33 = paddle._C_ops.matmul(layer_norm_57, parameter_80, False, False)
        del layer_norm_57, parameter_80

        # pd_op.add: (-1x-1x1280xf32) <- (-1x-1x1280xf32, 1280xf32)
        add_44 = paddle._C_ops.add(matmul_33, parameter_79)
        del matmul_33, parameter_79

        # pd_op.shape64: (3xi64) <- (-1x-1x1280xf32)
        shape64_12 = paddle._C_ops.shape64(add_44)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_37 = paddle._C_ops.slice(
            shape64_12, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_38 = paddle._C_ops.slice(
            shape64_12, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_12

        # pd_op.transpose: (-1x1280x-1xf32) <- (-1x-1x1280xf32)
        transpose_43 = paddle._C_ops.transpose(add_44, [0, 2, 1])
        del add_44

        # pd_op.full: (xi64) <- ()
        full_10 = paddle._C_ops.full(
            [], float("1280"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_31 = [slice_37, full_10, slice_31, slice_32]
        del slice_37

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_31 = paddle._C_ops.stack(combine_31, 0)
        del combine_31

        # pd_op.reshape: (-1x1280x-1x-1xf32) <- (-1x1280x-1xf32, 4xi64)
        reshape_43 = paddle._C_ops.reshape(transpose_43, stack_31)
        del stack_31, transpose_43

        # pd_op.depthwise_conv2d: (-1x1280x-1x-1xf32) <- (-1x1280x-1x-1xf32, 1280x1x3x3xf32)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(
            reshape_43, parameter_78, [1, 1], [1, 1], "EXPLICIT", 1280, [1, 1], "NCHW"
        )
        del parameter_78, reshape_43

        # pd_op.reshape: (1x1280x1x1xf32) <- (1280xf32, 4xi64)
        reshape_44 = paddle._C_ops.reshape(parameter_77, full_int_array_0)
        del parameter_77

        # pd_op.add: (-1x1280x-1x-1xf32) <- (-1x1280x-1x-1xf32, 1x1280x1x1xf32)
        add_45 = paddle._C_ops.add(depthwise_conv2d_4, reshape_44)
        del depthwise_conv2d_4, reshape_44

        # pd_op.flatten: (-1x1280x-1xf32) <- (-1x1280x-1x-1xf32)
        flatten_7 = paddle._C_ops.flatten(add_45, 2, 3)
        del add_45

        # pd_op.transpose: (-1x-1x1280xf32) <- (-1x1280x-1xf32)
        transpose_44 = paddle._C_ops.transpose(flatten_7, [0, 2, 1])
        del flatten_7

        # pd_op.gelu: (-1x-1x1280xf32) <- (-1x-1x1280xf32)
        gelu_4 = paddle._C_ops.gelu(transpose_44, False)
        del transpose_44

        # pd_op.matmul: (-1x-1x320xf32) <- (-1x-1x1280xf32, 1280x320xf32)
        matmul_34 = paddle._C_ops.matmul(gelu_4, parameter_76, False, False)
        del gelu_4, parameter_76

        # pd_op.add: (-1x-1x320xf32) <- (-1x-1x320xf32, 320xf32)
        add_46 = paddle._C_ops.add(matmul_34, parameter_75)
        del matmul_34, parameter_75

        # pd_op.add: (-1x-1x320xf32) <- (-1x-1x320xf32, -1x-1x320xf32)
        add_47 = paddle._C_ops.add(add_43, add_46)
        del add_43, add_46

        # pd_op.layer_norm: (-1x-1x320xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x320xf32, 320xf32, 320xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_47, parameter_74, parameter_73, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_73, parameter_74

        # pd_op.shape64: (3xi64) <- (-1x-1x320xf32)
        shape64_13 = paddle._C_ops.shape64(layer_norm_60)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_39 = paddle._C_ops.slice(
            shape64_13, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_40 = paddle._C_ops.slice(
            shape64_13, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_13

        # pd_op.matmul: (-1x-1x320xf32) <- (-1x-1x320xf32, 320x320xf32)
        matmul_35 = paddle._C_ops.matmul(layer_norm_60, parameter_72, False, False)
        del parameter_72

        # pd_op.add: (-1x-1x320xf32) <- (-1x-1x320xf32, 320xf32)
        add_48 = paddle._C_ops.add(matmul_35, parameter_71)
        del matmul_35, parameter_71

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_32 = [slice_39, slice_40, full_8, full_1]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_32 = paddle._C_ops.stack(combine_32, 0)
        del combine_32

        # pd_op.reshape: (-1x-1x5x64xf32) <- (-1x-1x320xf32, 4xi64)
        reshape_45 = paddle._C_ops.reshape(add_48, stack_32)
        del add_48, stack_32

        # pd_op.transpose: (-1x5x-1x64xf32) <- (-1x-1x5x64xf32)
        transpose_45 = paddle._C_ops.transpose(reshape_45, [0, 2, 1, 3])
        del reshape_45

        # pd_op.transpose: (-1x320x-1xf32) <- (-1x-1x320xf32)
        transpose_46 = paddle._C_ops.transpose(layer_norm_60, [0, 2, 1])
        del layer_norm_60

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_33 = [slice_39, full_9, slice_31, slice_32]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_33 = paddle._C_ops.stack(combine_33, 0)
        del combine_33

        # pd_op.reshape: (-1x320x-1x-1xf32) <- (-1x320x-1xf32, 4xi64)
        reshape_46 = paddle._C_ops.reshape(transpose_46, stack_33)
        del stack_33, transpose_46

        # pd_op.conv2d: (-1x320x-1x-1xf32) <- (-1x320x-1x-1xf32, 320x320x2x2xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            reshape_46, parameter_70, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_70, reshape_46

        # pd_op.reshape: (1x320x1x1xf32) <- (320xf32, 4xi64)
        reshape_47 = paddle._C_ops.reshape(parameter_69, full_int_array_0)
        del parameter_69

        # pd_op.add: (-1x320x-1x-1xf32) <- (-1x320x-1x-1xf32, 1x320x1x1xf32)
        add_49 = paddle._C_ops.add(conv2d_8, reshape_47)
        del conv2d_8, reshape_47

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_34 = [slice_39, full_9, full_2]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_34 = paddle._C_ops.stack(combine_34, 0)
        del combine_34

        # pd_op.reshape: (-1x320x-1xf32) <- (-1x320x-1x-1xf32, 3xi64)
        reshape_48 = paddle._C_ops.reshape(add_49, stack_34)
        del add_49, stack_34

        # pd_op.transpose: (-1x-1x320xf32) <- (-1x320x-1xf32)
        transpose_47 = paddle._C_ops.transpose(reshape_48, [0, 2, 1])
        del reshape_48

        # pd_op.layer_norm: (-1x-1x320xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x320xf32, 320xf32, 320xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_47, parameter_68, parameter_67, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_67, parameter_68, transpose_47

        # pd_op.matmul: (-1x-1x640xf32) <- (-1x-1x320xf32, 320x640xf32)
        matmul_36 = paddle._C_ops.matmul(layer_norm_63, parameter_66, False, False)
        del layer_norm_63, parameter_66

        # pd_op.add: (-1x-1x640xf32) <- (-1x-1x640xf32, 640xf32)
        add_50 = paddle._C_ops.add(matmul_36, parameter_65)
        del matmul_36, parameter_65

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_35 = [slice_39, full_2, full_3, full_8, full_1]
        del full_8

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_35 = paddle._C_ops.stack(combine_35, 0)
        del combine_35

        # pd_op.reshape: (-1x-1x2x5x64xf32) <- (-1x-1x640xf32, 5xi64)
        reshape_49 = paddle._C_ops.reshape(add_50, stack_35)
        del add_50, stack_35

        # pd_op.transpose: (2x-1x5x-1x64xf32) <- (-1x-1x2x5x64xf32)
        transpose_48 = paddle._C_ops.transpose(reshape_49, [2, 0, 3, 1, 4])
        del reshape_49

        # pd_op.slice: (-1x5x-1x64xf32) <- (2x-1x5x-1x64xf32, 1xi64, 1xi64)
        slice_41 = paddle._C_ops.slice(
            transpose_48, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (-1x5x-1x64xf32) <- (2x-1x5x-1x64xf32, 1xi64, 1xi64)
        slice_42 = paddle._C_ops.slice(
            transpose_48, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del transpose_48

        # pd_op.transpose: (-1x5x64x-1xf32) <- (-1x5x-1x64xf32)
        transpose_49 = paddle._C_ops.transpose(slice_41, [0, 1, 3, 2])
        del slice_41

        # pd_op.matmul: (-1x5x-1x-1xf32) <- (-1x5x-1x64xf32, -1x5x64x-1xf32)
        matmul_37 = paddle._C_ops.matmul(transpose_45, transpose_49, False, False)
        del transpose_45, transpose_49

        # pd_op.scale: (-1x5x-1x-1xf32) <- (-1x5x-1x-1xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(matmul_37, full_4, float("0"), True)
        del matmul_37

        # pd_op.softmax: (-1x5x-1x-1xf32) <- (-1x5x-1x-1xf32)
        softmax_5 = paddle._C_ops.softmax(scale_5, -1)
        del scale_5

        # pd_op.matmul: (-1x5x-1x64xf32) <- (-1x5x-1x-1xf32, -1x5x-1x64xf32)
        matmul_38 = paddle._C_ops.matmul(softmax_5, slice_42, False, False)
        del slice_42, softmax_5

        # pd_op.transpose: (-1x-1x5x64xf32) <- (-1x5x-1x64xf32)
        transpose_50 = paddle._C_ops.transpose(matmul_38, [0, 2, 1, 3])
        del matmul_38

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_36 = [slice_39, slice_40, full_9]
        del slice_39, slice_40

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_36 = paddle._C_ops.stack(combine_36, 0)
        del combine_36

        # pd_op.reshape: (-1x-1x320xf32) <- (-1x-1x5x64xf32, 3xi64)
        reshape_50 = paddle._C_ops.reshape(transpose_50, stack_36)
        del stack_36, transpose_50

        # pd_op.matmul: (-1x-1x320xf32) <- (-1x-1x320xf32, 320x320xf32)
        matmul_39 = paddle._C_ops.matmul(reshape_50, parameter_64, False, False)
        del parameter_64, reshape_50

        # pd_op.add: (-1x-1x320xf32) <- (-1x-1x320xf32, 320xf32)
        add_51 = paddle._C_ops.add(matmul_39, parameter_63)
        del matmul_39, parameter_63

        # pd_op.add: (-1x-1x320xf32) <- (-1x-1x320xf32, -1x-1x320xf32)
        add_52 = paddle._C_ops.add(add_47, add_51)
        del add_47, add_51

        # pd_op.layer_norm: (-1x-1x320xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x320xf32, 320xf32, 320xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_52, parameter_62, parameter_61, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_61, parameter_62

        # pd_op.matmul: (-1x-1x1280xf32) <- (-1x-1x320xf32, 320x1280xf32)
        matmul_40 = paddle._C_ops.matmul(layer_norm_66, parameter_60, False, False)
        del layer_norm_66, parameter_60

        # pd_op.add: (-1x-1x1280xf32) <- (-1x-1x1280xf32, 1280xf32)
        add_53 = paddle._C_ops.add(matmul_40, parameter_59)
        del matmul_40, parameter_59

        # pd_op.shape64: (3xi64) <- (-1x-1x1280xf32)
        shape64_14 = paddle._C_ops.shape64(add_53)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_43 = paddle._C_ops.slice(
            shape64_14, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_44 = paddle._C_ops.slice(
            shape64_14, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_14

        # pd_op.transpose: (-1x1280x-1xf32) <- (-1x-1x1280xf32)
        transpose_51 = paddle._C_ops.transpose(add_53, [0, 2, 1])
        del add_53

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_37 = [slice_43, full_10, slice_31, slice_32]
        del full_10, slice_43

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_37 = paddle._C_ops.stack(combine_37, 0)
        del combine_37

        # pd_op.reshape: (-1x1280x-1x-1xf32) <- (-1x1280x-1xf32, 4xi64)
        reshape_51 = paddle._C_ops.reshape(transpose_51, stack_37)
        del stack_37, transpose_51

        # pd_op.depthwise_conv2d: (-1x1280x-1x-1xf32) <- (-1x1280x-1x-1xf32, 1280x1x3x3xf32)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(
            reshape_51, parameter_58, [1, 1], [1, 1], "EXPLICIT", 1280, [1, 1], "NCHW"
        )
        del parameter_58, reshape_51

        # pd_op.reshape: (1x1280x1x1xf32) <- (1280xf32, 4xi64)
        reshape_52 = paddle._C_ops.reshape(parameter_57, full_int_array_0)
        del parameter_57

        # pd_op.add: (-1x1280x-1x-1xf32) <- (-1x1280x-1x-1xf32, 1x1280x1x1xf32)
        add_54 = paddle._C_ops.add(depthwise_conv2d_5, reshape_52)
        del depthwise_conv2d_5, reshape_52

        # pd_op.flatten: (-1x1280x-1xf32) <- (-1x1280x-1x-1xf32)
        flatten_8 = paddle._C_ops.flatten(add_54, 2, 3)
        del add_54

        # pd_op.transpose: (-1x-1x1280xf32) <- (-1x1280x-1xf32)
        transpose_52 = paddle._C_ops.transpose(flatten_8, [0, 2, 1])
        del flatten_8

        # pd_op.gelu: (-1x-1x1280xf32) <- (-1x-1x1280xf32)
        gelu_5 = paddle._C_ops.gelu(transpose_52, False)
        del transpose_52

        # pd_op.matmul: (-1x-1x320xf32) <- (-1x-1x1280xf32, 1280x320xf32)
        matmul_41 = paddle._C_ops.matmul(gelu_5, parameter_56, False, False)
        del gelu_5, parameter_56

        # pd_op.add: (-1x-1x320xf32) <- (-1x-1x320xf32, 320xf32)
        add_55 = paddle._C_ops.add(matmul_41, parameter_55)
        del matmul_41, parameter_55

        # pd_op.add: (-1x-1x320xf32) <- (-1x-1x320xf32, -1x-1x320xf32)
        add_56 = paddle._C_ops.add(add_52, add_55)
        del add_52, add_55

        # pd_op.layer_norm: (-1x-1x320xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x320xf32, 320xf32, 320xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_56, parameter_54, parameter_53, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_56, parameter_53, parameter_54

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_38 = [data_0, slice_31, slice_32, full_9]
        del full_9, slice_31, slice_32

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_38 = paddle._C_ops.stack(combine_38, 0)
        del combine_38

        # pd_op.reshape: (-1x-1x-1x320xf32) <- (-1x-1x320xf32, 4xi64)
        reshape_53 = paddle._C_ops.reshape(layer_norm_69, stack_38)
        del layer_norm_69, stack_38

        # pd_op.transpose: (-1x320x-1x-1xf32) <- (-1x-1x-1x320xf32)
        transpose_53 = paddle._C_ops.transpose(reshape_53, [0, 3, 1, 2])
        del reshape_53

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x320x-1x-1xf32, 512x320x3x3xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            transpose_53, parameter_52, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_52

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_54 = paddle._C_ops.reshape(parameter_51, full_int_array_0)
        del parameter_51

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 1x512x1x1xf32)
        add_57 = paddle._C_ops.add(conv2d_9, reshape_54)
        del conv2d_9, reshape_54

        # pd_op.shape64: (4xi64) <- (-1x512x-1x-1xf32)
        shape64_15 = paddle._C_ops.shape64(add_57)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_45 = paddle._C_ops.slice(
            shape64_15, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_46 = paddle._C_ops.slice(
            shape64_15, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_47 = paddle._C_ops.slice(
            shape64_15, [0], full_int_array_4, full_int_array_5, [1], [0]
        )
        del shape64_15

        # pd_op.flatten: (-1x512x-1xf32) <- (-1x512x-1x-1xf32)
        flatten_9 = paddle._C_ops.flatten(add_57, 2, 3)
        del add_57

        # pd_op.transpose: (-1x-1x512xf32) <- (-1x512x-1xf32)
        transpose_54 = paddle._C_ops.transpose(flatten_9, [0, 2, 1])
        del flatten_9

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_72, layer_norm_73, layer_norm_74 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_54, parameter_50, parameter_49, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_49, parameter_50, transpose_54

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_75, layer_norm_76, layer_norm_77 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                layer_norm_72, parameter_48, parameter_47, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_47, parameter_48

        # pd_op.shape64: (3xi64) <- (-1x-1x512xf32)
        shape64_16 = paddle._C_ops.shape64(layer_norm_75)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_48 = paddle._C_ops.slice(
            shape64_16, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_49 = paddle._C_ops.slice(
            shape64_16, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_16

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x512xf32, 512x512xf32)
        matmul_42 = paddle._C_ops.matmul(layer_norm_75, parameter_46, False, False)
        del parameter_46

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_58 = paddle._C_ops.add(matmul_42, parameter_45)
        del matmul_42, parameter_45

        # pd_op.full: (xi64) <- ()
        full_11 = paddle._C_ops.full(
            [], float("8"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_39 = [slice_48, slice_49, full_11, full_1]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_39 = paddle._C_ops.stack(combine_39, 0)
        del combine_39

        # pd_op.reshape: (-1x-1x8x64xf32) <- (-1x-1x512xf32, 4xi64)
        reshape_55 = paddle._C_ops.reshape(add_58, stack_39)
        del add_58, stack_39

        # pd_op.transpose: (-1x8x-1x64xf32) <- (-1x-1x8x64xf32)
        transpose_55 = paddle._C_ops.transpose(reshape_55, [0, 2, 1, 3])
        del reshape_55

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x512xf32, 512x1024xf32)
        matmul_43 = paddle._C_ops.matmul(layer_norm_75, parameter_44, False, False)
        del layer_norm_75, parameter_44

        # pd_op.add: (-1x-1x1024xf32) <- (-1x-1x1024xf32, 1024xf32)
        add_59 = paddle._C_ops.add(matmul_43, parameter_43)
        del matmul_43, parameter_43

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_40 = [slice_48, full_2, full_3, full_11, full_1]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_40 = paddle._C_ops.stack(combine_40, 0)
        del combine_40

        # pd_op.reshape: (-1x-1x2x8x64xf32) <- (-1x-1x1024xf32, 5xi64)
        reshape_56 = paddle._C_ops.reshape(add_59, stack_40)
        del add_59, stack_40

        # pd_op.transpose: (2x-1x8x-1x64xf32) <- (-1x-1x2x8x64xf32)
        transpose_56 = paddle._C_ops.transpose(reshape_56, [2, 0, 3, 1, 4])
        del reshape_56

        # pd_op.slice: (-1x8x-1x64xf32) <- (2x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_50 = paddle._C_ops.slice(
            transpose_56, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (-1x8x-1x64xf32) <- (2x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_51 = paddle._C_ops.slice(
            transpose_56, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del transpose_56

        # pd_op.transpose: (-1x8x64x-1xf32) <- (-1x8x-1x64xf32)
        transpose_57 = paddle._C_ops.transpose(slice_50, [0, 1, 3, 2])
        del slice_50

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x64xf32, -1x8x64x-1xf32)
        matmul_44 = paddle._C_ops.matmul(transpose_55, transpose_57, False, False)
        del transpose_55, transpose_57

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(matmul_44, full_4, float("0"), True)
        del matmul_44

        # pd_op.softmax: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32)
        softmax_6 = paddle._C_ops.softmax(scale_6, -1)
        del scale_6

        # pd_op.matmul: (-1x8x-1x64xf32) <- (-1x8x-1x-1xf32, -1x8x-1x64xf32)
        matmul_45 = paddle._C_ops.matmul(softmax_6, slice_51, False, False)
        del slice_51, softmax_6

        # pd_op.transpose: (-1x-1x8x64xf32) <- (-1x8x-1x64xf32)
        transpose_58 = paddle._C_ops.transpose(matmul_45, [0, 2, 1, 3])
        del matmul_45

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_41 = [slice_48, slice_49, full_7]
        del slice_48, slice_49

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_41 = paddle._C_ops.stack(combine_41, 0)
        del combine_41

        # pd_op.reshape: (-1x-1x512xf32) <- (-1x-1x8x64xf32, 3xi64)
        reshape_57 = paddle._C_ops.reshape(transpose_58, stack_41)
        del stack_41, transpose_58

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x512xf32, 512x512xf32)
        matmul_46 = paddle._C_ops.matmul(reshape_57, parameter_42, False, False)
        del parameter_42, reshape_57

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_60 = paddle._C_ops.add(matmul_46, parameter_41)
        del matmul_46, parameter_41

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, -1x-1x512xf32)
        add_61 = paddle._C_ops.add(layer_norm_72, add_60)
        del add_60, layer_norm_72

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_78, layer_norm_79, layer_norm_80 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_61, parameter_40, parameter_39, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_39, parameter_40

        # pd_op.matmul: (-1x-1x2048xf32) <- (-1x-1x512xf32, 512x2048xf32)
        matmul_47 = paddle._C_ops.matmul(layer_norm_78, parameter_38, False, False)
        del layer_norm_78, parameter_38

        # pd_op.add: (-1x-1x2048xf32) <- (-1x-1x2048xf32, 2048xf32)
        add_62 = paddle._C_ops.add(matmul_47, parameter_37)
        del matmul_47, parameter_37

        # pd_op.shape64: (3xi64) <- (-1x-1x2048xf32)
        shape64_17 = paddle._C_ops.shape64(add_62)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_52 = paddle._C_ops.slice(
            shape64_17, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_53 = paddle._C_ops.slice(
            shape64_17, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_17

        # pd_op.transpose: (-1x2048x-1xf32) <- (-1x-1x2048xf32)
        transpose_59 = paddle._C_ops.transpose(add_62, [0, 2, 1])
        del add_62

        # pd_op.full: (xi64) <- ()
        full_12 = paddle._C_ops.full(
            [], float("2048"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_42 = [slice_52, full_12, slice_46, slice_47]
        del slice_52

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_42 = paddle._C_ops.stack(combine_42, 0)
        del combine_42

        # pd_op.reshape: (-1x2048x-1x-1xf32) <- (-1x2048x-1xf32, 4xi64)
        reshape_58 = paddle._C_ops.reshape(transpose_59, stack_42)
        del stack_42, transpose_59

        # pd_op.depthwise_conv2d: (-1x2048x-1x-1xf32) <- (-1x2048x-1x-1xf32, 2048x1x3x3xf32)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(
            reshape_58, parameter_36, [1, 1], [1, 1], "EXPLICIT", 2048, [1, 1], "NCHW"
        )
        del parameter_36, reshape_58

        # pd_op.reshape: (1x2048x1x1xf32) <- (2048xf32, 4xi64)
        reshape_59 = paddle._C_ops.reshape(parameter_35, full_int_array_0)
        del parameter_35

        # pd_op.add: (-1x2048x-1x-1xf32) <- (-1x2048x-1x-1xf32, 1x2048x1x1xf32)
        add_63 = paddle._C_ops.add(depthwise_conv2d_6, reshape_59)
        del depthwise_conv2d_6, reshape_59

        # pd_op.flatten: (-1x2048x-1xf32) <- (-1x2048x-1x-1xf32)
        flatten_10 = paddle._C_ops.flatten(add_63, 2, 3)
        del add_63

        # pd_op.transpose: (-1x-1x2048xf32) <- (-1x2048x-1xf32)
        transpose_60 = paddle._C_ops.transpose(flatten_10, [0, 2, 1])
        del flatten_10

        # pd_op.gelu: (-1x-1x2048xf32) <- (-1x-1x2048xf32)
        gelu_6 = paddle._C_ops.gelu(transpose_60, False)
        del transpose_60

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x2048xf32, 2048x512xf32)
        matmul_48 = paddle._C_ops.matmul(gelu_6, parameter_34, False, False)
        del gelu_6, parameter_34

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_64 = paddle._C_ops.add(matmul_48, parameter_33)
        del matmul_48, parameter_33

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, -1x-1x512xf32)
        add_65 = paddle._C_ops.add(add_61, add_64)
        del add_61, add_64

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_81, layer_norm_82, layer_norm_83 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_65, parameter_32, parameter_31, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_31, parameter_32

        # pd_op.shape64: (3xi64) <- (-1x-1x512xf32)
        shape64_18 = paddle._C_ops.shape64(layer_norm_81)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_54 = paddle._C_ops.slice(
            shape64_18, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_55 = paddle._C_ops.slice(
            shape64_18, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_18

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x512xf32, 512x512xf32)
        matmul_49 = paddle._C_ops.matmul(layer_norm_81, parameter_30, False, False)
        del parameter_30

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_66 = paddle._C_ops.add(matmul_49, parameter_29)
        del matmul_49, parameter_29

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_43 = [slice_54, slice_55, full_11, full_1]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_43 = paddle._C_ops.stack(combine_43, 0)
        del combine_43

        # pd_op.reshape: (-1x-1x8x64xf32) <- (-1x-1x512xf32, 4xi64)
        reshape_60 = paddle._C_ops.reshape(add_66, stack_43)
        del add_66, stack_43

        # pd_op.transpose: (-1x8x-1x64xf32) <- (-1x-1x8x64xf32)
        transpose_61 = paddle._C_ops.transpose(reshape_60, [0, 2, 1, 3])
        del reshape_60

        # pd_op.matmul: (-1x-1x1024xf32) <- (-1x-1x512xf32, 512x1024xf32)
        matmul_50 = paddle._C_ops.matmul(layer_norm_81, parameter_28, False, False)
        del layer_norm_81, parameter_28

        # pd_op.add: (-1x-1x1024xf32) <- (-1x-1x1024xf32, 1024xf32)
        add_67 = paddle._C_ops.add(matmul_50, parameter_27)
        del matmul_50, parameter_27

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_44 = [slice_54, full_2, full_3, full_11, full_1]
        del full_1, full_11, full_2, full_3

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_44 = paddle._C_ops.stack(combine_44, 0)
        del combine_44

        # pd_op.reshape: (-1x-1x2x8x64xf32) <- (-1x-1x1024xf32, 5xi64)
        reshape_61 = paddle._C_ops.reshape(add_67, stack_44)
        del add_67, stack_44

        # pd_op.transpose: (2x-1x8x-1x64xf32) <- (-1x-1x2x8x64xf32)
        transpose_62 = paddle._C_ops.transpose(reshape_61, [2, 0, 3, 1, 4])
        del reshape_61

        # pd_op.slice: (-1x8x-1x64xf32) <- (2x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_56 = paddle._C_ops.slice(
            transpose_62, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (-1x8x-1x64xf32) <- (2x-1x8x-1x64xf32, 1xi64, 1xi64)
        slice_57 = paddle._C_ops.slice(
            transpose_62, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del transpose_62

        # pd_op.transpose: (-1x8x64x-1xf32) <- (-1x8x-1x64xf32)
        transpose_63 = paddle._C_ops.transpose(slice_56, [0, 1, 3, 2])
        del slice_56

        # pd_op.matmul: (-1x8x-1x-1xf32) <- (-1x8x-1x64xf32, -1x8x64x-1xf32)
        matmul_51 = paddle._C_ops.matmul(transpose_61, transpose_63, False, False)
        del transpose_61, transpose_63

        # pd_op.scale: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(matmul_51, full_4, float("0"), True)
        del full_4, matmul_51

        # pd_op.softmax: (-1x8x-1x-1xf32) <- (-1x8x-1x-1xf32)
        softmax_7 = paddle._C_ops.softmax(scale_7, -1)
        del scale_7

        # pd_op.matmul: (-1x8x-1x64xf32) <- (-1x8x-1x-1xf32, -1x8x-1x64xf32)
        matmul_52 = paddle._C_ops.matmul(softmax_7, slice_57, False, False)
        del slice_57, softmax_7

        # pd_op.transpose: (-1x-1x8x64xf32) <- (-1x8x-1x64xf32)
        transpose_64 = paddle._C_ops.transpose(matmul_52, [0, 2, 1, 3])
        del matmul_52

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_45 = [slice_54, slice_55, full_7]
        del slice_54, slice_55

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_45 = paddle._C_ops.stack(combine_45, 0)
        del combine_45

        # pd_op.reshape: (-1x-1x512xf32) <- (-1x-1x8x64xf32, 3xi64)
        reshape_62 = paddle._C_ops.reshape(transpose_64, stack_45)
        del stack_45, transpose_64

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x512xf32, 512x512xf32)
        matmul_53 = paddle._C_ops.matmul(reshape_62, parameter_26, False, False)
        del parameter_26, reshape_62

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_68 = paddle._C_ops.add(matmul_53, parameter_25)
        del matmul_53, parameter_25

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, -1x-1x512xf32)
        add_69 = paddle._C_ops.add(add_65, add_68)
        del add_65, add_68

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_84, layer_norm_85, layer_norm_86 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_69, parameter_24, parameter_23, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_23, parameter_24

        # pd_op.matmul: (-1x-1x2048xf32) <- (-1x-1x512xf32, 512x2048xf32)
        matmul_54 = paddle._C_ops.matmul(layer_norm_84, parameter_22, False, False)
        del layer_norm_84, parameter_22

        # pd_op.add: (-1x-1x2048xf32) <- (-1x-1x2048xf32, 2048xf32)
        add_70 = paddle._C_ops.add(matmul_54, parameter_21)
        del matmul_54, parameter_21

        # pd_op.shape64: (3xi64) <- (-1x-1x2048xf32)
        shape64_19 = paddle._C_ops.shape64(add_70)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_58 = paddle._C_ops.slice(
            shape64_19, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_59 = paddle._C_ops.slice(
            shape64_19, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_19

        # pd_op.transpose: (-1x2048x-1xf32) <- (-1x-1x2048xf32)
        transpose_65 = paddle._C_ops.transpose(add_70, [0, 2, 1])
        del add_70

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_46 = [slice_58, full_12, slice_46, slice_47]
        del full_12, slice_58

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_46 = paddle._C_ops.stack(combine_46, 0)
        del combine_46

        # pd_op.reshape: (-1x2048x-1x-1xf32) <- (-1x2048x-1xf32, 4xi64)
        reshape_63 = paddle._C_ops.reshape(transpose_65, stack_46)
        del stack_46, transpose_65

        # pd_op.depthwise_conv2d: (-1x2048x-1x-1xf32) <- (-1x2048x-1x-1xf32, 2048x1x3x3xf32)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(
            reshape_63, parameter_20, [1, 1], [1, 1], "EXPLICIT", 2048, [1, 1], "NCHW"
        )
        del parameter_20, reshape_63

        # pd_op.reshape: (1x2048x1x1xf32) <- (2048xf32, 4xi64)
        reshape_64 = paddle._C_ops.reshape(parameter_19, full_int_array_0)
        del parameter_19

        # pd_op.add: (-1x2048x-1x-1xf32) <- (-1x2048x-1x-1xf32, 1x2048x1x1xf32)
        add_71 = paddle._C_ops.add(depthwise_conv2d_7, reshape_64)
        del depthwise_conv2d_7, reshape_64

        # pd_op.flatten: (-1x2048x-1xf32) <- (-1x2048x-1x-1xf32)
        flatten_11 = paddle._C_ops.flatten(add_71, 2, 3)
        del add_71

        # pd_op.transpose: (-1x-1x2048xf32) <- (-1x2048x-1xf32)
        transpose_66 = paddle._C_ops.transpose(flatten_11, [0, 2, 1])
        del flatten_11

        # pd_op.gelu: (-1x-1x2048xf32) <- (-1x-1x2048xf32)
        gelu_7 = paddle._C_ops.gelu(transpose_66, False)
        del transpose_66

        # pd_op.matmul: (-1x-1x512xf32) <- (-1x-1x2048xf32, 2048x512xf32)
        matmul_55 = paddle._C_ops.matmul(gelu_7, parameter_18, False, False)
        del gelu_7, parameter_18

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, 512xf32)
        add_72 = paddle._C_ops.add(matmul_55, parameter_17)
        del matmul_55, parameter_17

        # pd_op.add: (-1x-1x512xf32) <- (-1x-1x512xf32, -1x-1x512xf32)
        add_73 = paddle._C_ops.add(add_69, add_72)
        del add_69, add_72

        # pd_op.layer_norm: (-1x-1x512xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x512xf32, 512xf32, 512xf32)
        layer_norm_87, layer_norm_88, layer_norm_89 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_73, parameter_16, parameter_15, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_73, parameter_15, parameter_16

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_47 = [data_0, slice_46, slice_47, full_7]
        del data_0, full_7, slice_46, slice_47

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_47 = paddle._C_ops.stack(combine_47, 0)
        del combine_47

        # pd_op.reshape: (-1x-1x-1x512xf32) <- (-1x-1x512xf32, 4xi64)
        reshape_65 = paddle._C_ops.reshape(layer_norm_87, stack_47)
        del layer_norm_87, stack_47

        # pd_op.transpose: (-1x512x-1x-1xf32) <- (-1x-1x-1x512xf32)
        transpose_67 = paddle._C_ops.transpose(reshape_65, [0, 3, 1, 2])
        del reshape_65

        # pd_op.shape64: (4xi64) <- (-1x64x-1x-1xf32)
        shape64_20 = paddle._C_ops.shape64(transpose_17)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_60 = paddle._C_ops.slice(
            shape64_20, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_61 = paddle._C_ops.slice(
            shape64_20, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_62 = paddle._C_ops.slice(
            shape64_20, [0], full_int_array_4, full_int_array_5, [1], [0]
        )
        del shape64_20

        # pd_op.shape64: (4xi64) <- (-1x128x-1x-1xf32)
        shape64_21 = paddle._C_ops.shape64(transpose_35)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_63 = paddle._C_ops.slice(
            shape64_21, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_64 = paddle._C_ops.slice(
            shape64_21, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_65 = paddle._C_ops.slice(
            shape64_21, [0], full_int_array_4, full_int_array_5, [1], [0]
        )
        del shape64_21

        # pd_op.shape64: (4xi64) <- (-1x320x-1x-1xf32)
        shape64_22 = paddle._C_ops.shape64(transpose_53)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_66 = paddle._C_ops.slice(
            shape64_22, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_67 = paddle._C_ops.slice(
            shape64_22, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_68 = paddle._C_ops.slice(
            shape64_22, [0], full_int_array_4, full_int_array_5, [1], [0]
        )
        del shape64_22

        # pd_op.shape64: (4xi64) <- (-1x512x-1x-1xf32)
        shape64_23 = paddle._C_ops.shape64(transpose_67)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_69 = paddle._C_ops.slice(
            shape64_23, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del full_int_array_1, full_int_array_2

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_70 = paddle._C_ops.slice(
            shape64_23, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del full_int_array_3

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_71 = paddle._C_ops.slice(
            shape64_23, [0], full_int_array_4, full_int_array_5, [1], [0]
        )
        del full_int_array_4, full_int_array_5, shape64_23

        # pd_op.flatten: (-1x512x-1xf32) <- (-1x512x-1x-1xf32)
        flatten_12 = paddle._C_ops.flatten(transpose_67, 2, 3)
        del transpose_67

        # pd_op.transpose: (-1x-1x512xf32) <- (-1x512x-1xf32)
        transpose_68 = paddle._C_ops.transpose(flatten_12, [0, 2, 1])
        del flatten_12

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x512xf32, 512x256xf32)
        matmul_56 = paddle._C_ops.matmul(transpose_68, parameter_14, False, False)
        del parameter_14, transpose_68

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_74 = paddle._C_ops.add(matmul_56, parameter_13)
        del matmul_56, parameter_13

        # pd_op.transpose: (-1x256x-1xf32) <- (-1x-1x256xf32)
        transpose_69 = paddle._C_ops.transpose(add_74, [0, 2, 1])
        del add_74

        # pd_op.full: (xi64) <- ()
        full_13 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_48 = [full_13, full_13, slice_70, slice_71]
        del slice_70, slice_71

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_48 = paddle._C_ops.stack(combine_48, 0)
        del combine_48

        # pd_op.reshape: (-1x256x-1x-1xf32) <- (-1x256x-1xf32, 4xi64)
        reshape_66 = paddle._C_ops.reshape(transpose_69, stack_48)
        del stack_48, transpose_69

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_49 = [slice_61, slice_62]

        # pd_op.bilinear_interp: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, None, [xi64, xi64], None)
        bilinear_interp_1 = paddle._C_ops.bilinear_interp(
            reshape_66,
            None,
            combine_49,
            None,
            "NCHW",
            -1,
            -1,
            -1,
            [],
            "bilinear",
            False,
            0,
        )
        del reshape_66

        # pd_op.flatten: (-1x320x-1xf32) <- (-1x320x-1x-1xf32)
        flatten_13 = paddle._C_ops.flatten(transpose_53, 2, 3)
        del transpose_53

        # pd_op.transpose: (-1x-1x320xf32) <- (-1x320x-1xf32)
        transpose_70 = paddle._C_ops.transpose(flatten_13, [0, 2, 1])
        del flatten_13

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x320xf32, 320x256xf32)
        matmul_57 = paddle._C_ops.matmul(transpose_70, parameter_12, False, False)
        del parameter_12, transpose_70

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_75 = paddle._C_ops.add(matmul_57, parameter_11)
        del matmul_57, parameter_11

        # pd_op.transpose: (-1x256x-1xf32) <- (-1x-1x256xf32)
        transpose_71 = paddle._C_ops.transpose(add_75, [0, 2, 1])
        del add_75

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_50 = [full_13, full_13, slice_67, slice_68]
        del slice_67, slice_68

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_49 = paddle._C_ops.stack(combine_50, 0)
        del combine_50

        # pd_op.reshape: (-1x256x-1x-1xf32) <- (-1x256x-1xf32, 4xi64)
        reshape_67 = paddle._C_ops.reshape(transpose_71, stack_49)
        del stack_49, transpose_71

        # pd_op.bilinear_interp: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, None, [xi64, xi64], None)
        bilinear_interp_2 = paddle._C_ops.bilinear_interp(
            reshape_67,
            None,
            combine_49,
            None,
            "NCHW",
            -1,
            -1,
            -1,
            [],
            "bilinear",
            False,
            0,
        )
        del reshape_67

        # pd_op.flatten: (-1x128x-1xf32) <- (-1x128x-1x-1xf32)
        flatten_14 = paddle._C_ops.flatten(transpose_35, 2, 3)
        del transpose_35

        # pd_op.transpose: (-1x-1x128xf32) <- (-1x128x-1xf32)
        transpose_72 = paddle._C_ops.transpose(flatten_14, [0, 2, 1])
        del flatten_14

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x128xf32, 128x256xf32)
        matmul_58 = paddle._C_ops.matmul(transpose_72, parameter_10, False, False)
        del parameter_10, transpose_72

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_76 = paddle._C_ops.add(matmul_58, parameter_9)
        del matmul_58, parameter_9

        # pd_op.transpose: (-1x256x-1xf32) <- (-1x-1x256xf32)
        transpose_73 = paddle._C_ops.transpose(add_76, [0, 2, 1])
        del add_76

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_51 = [full_13, full_13, slice_64, slice_65]
        del slice_64, slice_65

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_50 = paddle._C_ops.stack(combine_51, 0)
        del combine_51

        # pd_op.reshape: (-1x256x-1x-1xf32) <- (-1x256x-1xf32, 4xi64)
        reshape_68 = paddle._C_ops.reshape(transpose_73, stack_50)
        del stack_50, transpose_73

        # pd_op.bilinear_interp: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, None, [xi64, xi64], None)
        bilinear_interp_3 = paddle._C_ops.bilinear_interp(
            reshape_68,
            None,
            combine_49,
            None,
            "NCHW",
            -1,
            -1,
            -1,
            [],
            "bilinear",
            False,
            0,
        )
        del combine_49, reshape_68

        # pd_op.flatten: (-1x64x-1xf32) <- (-1x64x-1x-1xf32)
        flatten_15 = paddle._C_ops.flatten(transpose_17, 2, 3)
        del transpose_17

        # pd_op.transpose: (-1x-1x64xf32) <- (-1x64x-1xf32)
        transpose_74 = paddle._C_ops.transpose(flatten_15, [0, 2, 1])
        del flatten_15

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x64xf32, 64x256xf32)
        matmul_59 = paddle._C_ops.matmul(transpose_74, parameter_8, False, False)
        del parameter_8, transpose_74

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_77 = paddle._C_ops.add(matmul_59, parameter_7)
        del matmul_59, parameter_7

        # pd_op.transpose: (-1x256x-1xf32) <- (-1x-1x256xf32)
        transpose_75 = paddle._C_ops.transpose(add_77, [0, 2, 1])
        del add_77

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_52 = [full_13, full_13, slice_61, slice_62]
        del full_13, slice_61, slice_62

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_51 = paddle._C_ops.stack(combine_52, 0)
        del combine_52

        # pd_op.reshape: (-1x256x-1x-1xf32) <- (-1x256x-1xf32, 4xi64)
        reshape_69 = paddle._C_ops.reshape(transpose_75, stack_51)
        del stack_51, transpose_75

        # pd_op.full: (1xi32) <- ()
        full_14 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x256x-1x-1xf32, -1x256x-1x-1xf32, -1x256x-1x-1xf32, -1x256x-1x-1xf32]) <- (-1x256x-1x-1xf32, -1x256x-1x-1xf32, -1x256x-1x-1xf32, -1x256x-1x-1xf32)
        combine_53 = [
            bilinear_interp_1,
            bilinear_interp_2,
            bilinear_interp_3,
            reshape_69,
        ]
        del bilinear_interp_1, bilinear_interp_2, bilinear_interp_3, reshape_69

        # pd_op.concat: (-1x1024x-1x-1xf32) <- ([-1x256x-1x-1xf32, -1x256x-1x-1xf32, -1x256x-1x-1xf32, -1x256x-1x-1xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_53, full_14)
        del combine_53, full_14

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x1024x-1x-1xf32, 256x1024x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            concat_0, parameter_6, [1, 1], [0, 0], "SAME", [1, 1], 1, "NCHW"
        )
        del concat_0, parameter_6

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
            batch_norm__5,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_10,
                parameter_5,
                parameter_4,
                parameter_3,
                parameter_2,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_10, parameter_2, parameter_3, parameter_4, parameter_5

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_0 = paddle._C_ops.relu(batch_norm__0)
        del batch_norm__0

        # pd_op.conv2d: (-1x2x-1x-1xf32) <- (-1x256x-1x-1xf32, 2x256x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            relu_0, parameter_1, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_1, relu_0

        # pd_op.reshape: (1x2x1x1xf32) <- (2xf32, 4xi64)
        reshape_70 = paddle._C_ops.reshape(parameter_0, full_int_array_0)
        del full_int_array_0, parameter_0

        # pd_op.add: (-1x2x-1x-1xf32) <- (-1x2x-1x-1xf32, 1x2x1x1xf32)
        add_78 = paddle._C_ops.add(conv2d_11, reshape_70)
        del conv2d_11, reshape_70

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_54 = [data_1, data_2]
        del data_1, data_2

        # pd_op.bilinear_interp: (-1x2x-1x-1xf32) <- (-1x2x-1x-1xf32, None, [xi64, xi64], None)
        bilinear_interp_0 = paddle._C_ops.bilinear_interp(
            add_78, None, combine_54, None, "NCHW", -1, -1, -1, [], "bilinear", False, 0
        )
        del add_78, combine_54

        return bilinear_interp_0
