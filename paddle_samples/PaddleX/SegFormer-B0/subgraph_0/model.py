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
    ):
        # pd_op.conv2d: (2x32x128x256xf32) <- (2x3x512x1024xf32, 32x3x7x7xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_190, [4, 4], [3, 3], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_0, parameter_190

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_189, full_int_array_0)
        del parameter_189

        # pd_op.add: (2x32x128x256xf32) <- (2x32x128x256xf32, 1x32x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_0, reshape_0)

        # pd_op.flatten: (2x32x32768xf32) <- (2x32x128x256xf32)
        flatten_0 = paddle._C_ops.flatten(add_0, 2, 3)

        # pd_op.transpose: (2x32768x32xf32) <- (2x32x32768xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_0, [0, 2, 1])
        del flatten_0

        # pd_op.layer_norm: (2x32768x32xf32, 2x32768xf32, 2x32768xf32) <- (2x32768x32xf32, 32xf32, 32xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_0, parameter_188, parameter_187, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_187, parameter_188

        # pd_op.layer_norm: (2x32768x32xf32, 2x32768xf32, 2x32768xf32) <- (2x32768x32xf32, 32xf32, 32xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                layer_norm_0, parameter_186, parameter_185, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_185, parameter_186

        # pd_op.matmul: (2x32768x32xf32) <- (2x32768x32xf32, 32x32xf32)
        matmul_0 = paddle._C_ops.matmul(layer_norm_3, parameter_184, False, False)
        del parameter_184

        # pd_op.add: (2x32768x32xf32) <- (2x32768x32xf32, 32xf32)
        add_1 = paddle._C_ops.add(matmul_0, parameter_183)
        del parameter_183

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [2, 32768, 1, 32]

        # pd_op.reshape: (2x32768x1x32xf32) <- (2x32768x32xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(add_1, full_int_array_1)

        # pd_op.transpose: (2x1x32768x32xf32) <- (2x32768x1x32xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_1, [0, 2, 1, 3])
        del reshape_1

        # pd_op.transpose: (2x32x32768xf32) <- (2x32768x32xf32)
        transpose_2 = paddle._C_ops.transpose(layer_norm_3, [0, 2, 1])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_2 = [2, 32, 128, 256]

        # pd_op.reshape: (2x32x128x256xf32) <- (2x32x32768xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(transpose_2, full_int_array_2)

        # pd_op.conv2d: (2x32x16x32xf32) <- (2x32x128x256xf32, 32x32x8x8xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            reshape_2, parameter_182, [8, 8], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_182

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(parameter_181, full_int_array_0)
        del parameter_181

        # pd_op.add: (2x32x16x32xf32) <- (2x32x16x32xf32, 1x32x1x1xf32)
        add_2 = paddle._C_ops.add(conv2d_1, reshape_3)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_3 = [2, 32, -1]

        # pd_op.reshape: (2x32x512xf32) <- (2x32x16x32xf32, 3xi64)
        reshape_4 = paddle._C_ops.reshape(add_2, full_int_array_3)

        # pd_op.transpose: (2x512x32xf32) <- (2x32x512xf32)
        transpose_3 = paddle._C_ops.transpose(reshape_4, [0, 2, 1])
        del reshape_4

        # pd_op.layer_norm: (2x512x32xf32, 2x512xf32, 2x512xf32) <- (2x512x32xf32, 32xf32, 32xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_3, parameter_180, parameter_179, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_179, parameter_180

        # pd_op.matmul: (2x512x64xf32) <- (2x512x32xf32, 32x64xf32)
        matmul_1 = paddle._C_ops.matmul(layer_norm_6, parameter_178, False, False)
        del parameter_178

        # pd_op.add: (2x512x64xf32) <- (2x512x64xf32, 64xf32)
        add_3 = paddle._C_ops.add(matmul_1, parameter_177)
        del parameter_177

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_4 = [2, -1, 2, 1, 32]

        # pd_op.reshape: (2x512x2x1x32xf32) <- (2x512x64xf32, 5xi64)
        reshape_5 = paddle._C_ops.reshape(add_3, full_int_array_4)

        # pd_op.transpose: (2x2x1x512x32xf32) <- (2x512x2x1x32xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_5, [2, 0, 3, 1, 4])
        del reshape_5

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_3 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_4 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_5 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_6 = full_int_array_5

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_7 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_8 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_9 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_10 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_11 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_12 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_13 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_14 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_15 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_16 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_17 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_18 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_19 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_20 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_21 = full_int_array_6

        # pd_op.slice: (2x1x512x32xf32) <- (2x2x1x512x32xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            transpose_4, [0], full_int_array_5, full_int_array_6, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_22 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_23 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_24 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_25 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_26 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_27 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_28 = full_int_array_7

        # pd_op.slice: (2x1x512x32xf32) <- (2x2x1x512x32xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            transpose_4, [0], full_int_array_6, full_int_array_7, [1], [0]
        )

        # pd_op.transpose: (2x1x32x512xf32) <- (2x1x512x32xf32)
        transpose_5 = paddle._C_ops.transpose(slice_0, [0, 1, 3, 2])
        del slice_0

        # pd_op.matmul: (2x1x32768x512xf32) <- (2x1x32768x32xf32, 2x1x32x512xf32)
        matmul_2 = paddle._C_ops.matmul(transpose_1, transpose_5, False, False)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_29 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_30 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_31 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_32 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_33 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_34 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_35 = full_0

        # pd_op.scale: (2x1x32768x512xf32) <- (2x1x32768x512xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(matmul_2, full_0, float("0"), True)
        del matmul_2

        # pd_op.softmax: (2x1x32768x512xf32) <- (2x1x32768x512xf32)
        softmax_0 = paddle._C_ops.softmax(scale_0, -1)
        del scale_0

        # pd_op.matmul: (2x1x32768x32xf32) <- (2x1x32768x512xf32, 2x1x512x32xf32)
        matmul_3 = paddle._C_ops.matmul(softmax_0, slice_1, False, False)

        # pd_op.transpose: (2x32768x1x32xf32) <- (2x1x32768x32xf32)
        transpose_6 = paddle._C_ops.transpose(matmul_3, [0, 2, 1, 3])
        del matmul_3

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_8 = [2, 32768, 32]

        # pd_op.reshape: (2x32768x32xf32) <- (2x32768x1x32xf32, 3xi64)
        reshape_6 = paddle._C_ops.reshape(transpose_6, full_int_array_8)

        # pd_op.matmul: (2x32768x32xf32) <- (2x32768x32xf32, 32x32xf32)
        matmul_4 = paddle._C_ops.matmul(reshape_6, parameter_176, False, False)
        del parameter_176

        # pd_op.add: (2x32768x32xf32) <- (2x32768x32xf32, 32xf32)
        add_4 = paddle._C_ops.add(matmul_4, parameter_175)
        del parameter_175

        # pd_op.add: (2x32768x32xf32) <- (2x32768x32xf32, 2x32768x32xf32)
        add_5 = paddle._C_ops.add(layer_norm_0, add_4)

        # pd_op.layer_norm: (2x32768x32xf32, 2x32768xf32, 2x32768xf32) <- (2x32768x32xf32, 32xf32, 32xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_5, parameter_174, parameter_173, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_173, parameter_174

        # pd_op.matmul: (2x32768x128xf32) <- (2x32768x32xf32, 32x128xf32)
        matmul_5 = paddle._C_ops.matmul(layer_norm_9, parameter_172, False, False)
        del parameter_172

        # pd_op.add: (2x32768x128xf32) <- (2x32768x128xf32, 128xf32)
        add_6 = paddle._C_ops.add(matmul_5, parameter_171)
        del parameter_171

        # pd_op.transpose: (2x128x32768xf32) <- (2x32768x128xf32)
        transpose_7 = paddle._C_ops.transpose(add_6, [0, 2, 1])
        del add_6

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_9 = [2, 128, 128, 256]

        # pd_op.reshape: (2x128x128x256xf32) <- (2x128x32768xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_7, full_int_array_9)

        # pd_op.depthwise_conv2d: (2x128x128x256xf32) <- (2x128x128x256xf32, 128x1x3x3xf32)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(
            reshape_7, parameter_170, [1, 1], [1, 1], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del parameter_170

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(parameter_169, full_int_array_0)
        del parameter_169

        # pd_op.add: (2x128x128x256xf32) <- (2x128x128x256xf32, 1x128x1x1xf32)
        add_7 = paddle._C_ops.add(depthwise_conv2d_0, reshape_8)

        # pd_op.flatten: (2x128x32768xf32) <- (2x128x128x256xf32)
        flatten_1 = paddle._C_ops.flatten(add_7, 2, 3)

        # pd_op.transpose: (2x32768x128xf32) <- (2x128x32768xf32)
        transpose_8 = paddle._C_ops.transpose(flatten_1, [0, 2, 1])
        del flatten_1

        # pd_op.gelu: (2x32768x128xf32) <- (2x32768x128xf32)
        gelu_0 = paddle._C_ops.gelu(transpose_8, False)

        # pd_op.matmul: (2x32768x32xf32) <- (2x32768x128xf32, 128x32xf32)
        matmul_6 = paddle._C_ops.matmul(gelu_0, parameter_168, False, False)
        del parameter_168

        # pd_op.add: (2x32768x32xf32) <- (2x32768x32xf32, 32xf32)
        add_8 = paddle._C_ops.add(matmul_6, parameter_167)
        del parameter_167

        # pd_op.add: (2x32768x32xf32) <- (2x32768x32xf32, 2x32768x32xf32)
        add_9 = paddle._C_ops.add(add_5, add_8)

        # pd_op.layer_norm: (2x32768x32xf32, 2x32768xf32, 2x32768xf32) <- (2x32768x32xf32, 32xf32, 32xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_9, parameter_166, parameter_165, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_165, parameter_166

        # pd_op.matmul: (2x32768x32xf32) <- (2x32768x32xf32, 32x32xf32)
        matmul_7 = paddle._C_ops.matmul(layer_norm_12, parameter_164, False, False)
        del parameter_164

        # pd_op.add: (2x32768x32xf32) <- (2x32768x32xf32, 32xf32)
        add_10 = paddle._C_ops.add(matmul_7, parameter_163)
        del parameter_163

        # pd_op.reshape: (2x32768x1x32xf32) <- (2x32768x32xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(add_10, full_int_array_1)
        del full_int_array_1

        # pd_op.transpose: (2x1x32768x32xf32) <- (2x32768x1x32xf32)
        transpose_9 = paddle._C_ops.transpose(reshape_9, [0, 2, 1, 3])
        del reshape_9

        # pd_op.transpose: (2x32x32768xf32) <- (2x32768x32xf32)
        transpose_10 = paddle._C_ops.transpose(layer_norm_12, [0, 2, 1])

        # pd_op.reshape: (2x32x128x256xf32) <- (2x32x32768xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(transpose_10, full_int_array_2)
        del full_int_array_2

        # pd_op.conv2d: (2x32x16x32xf32) <- (2x32x128x256xf32, 32x32x8x8xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            reshape_10, parameter_162, [8, 8], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_162

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        reshape_11 = paddle._C_ops.reshape(parameter_161, full_int_array_0)
        del parameter_161

        # pd_op.add: (2x32x16x32xf32) <- (2x32x16x32xf32, 1x32x1x1xf32)
        add_11 = paddle._C_ops.add(conv2d_2, reshape_11)

        # pd_op.reshape: (2x32x512xf32) <- (2x32x16x32xf32, 3xi64)
        reshape_12 = paddle._C_ops.reshape(add_11, full_int_array_3)
        del full_int_array_3

        # pd_op.transpose: (2x512x32xf32) <- (2x32x512xf32)
        transpose_11 = paddle._C_ops.transpose(reshape_12, [0, 2, 1])
        del reshape_12

        # pd_op.layer_norm: (2x512x32xf32, 2x512xf32, 2x512xf32) <- (2x512x32xf32, 32xf32, 32xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_11, parameter_160, parameter_159, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_159, parameter_160

        # pd_op.matmul: (2x512x64xf32) <- (2x512x32xf32, 32x64xf32)
        matmul_8 = paddle._C_ops.matmul(layer_norm_15, parameter_158, False, False)
        del parameter_158

        # pd_op.add: (2x512x64xf32) <- (2x512x64xf32, 64xf32)
        add_12 = paddle._C_ops.add(matmul_8, parameter_157)
        del parameter_157

        # pd_op.reshape: (2x512x2x1x32xf32) <- (2x512x64xf32, 5xi64)
        reshape_13 = paddle._C_ops.reshape(add_12, full_int_array_4)
        del full_int_array_4

        # pd_op.transpose: (2x2x1x512x32xf32) <- (2x512x2x1x32xf32)
        transpose_12 = paddle._C_ops.transpose(reshape_13, [2, 0, 3, 1, 4])
        del reshape_13

        # pd_op.slice: (2x1x512x32xf32) <- (2x2x1x512x32xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            transpose_12, [0], full_int_array_5, full_int_array_6, [1], [0]
        )

        # pd_op.slice: (2x1x512x32xf32) <- (2x2x1x512x32xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            transpose_12, [0], full_int_array_6, full_int_array_7, [1], [0]
        )

        # pd_op.transpose: (2x1x32x512xf32) <- (2x1x512x32xf32)
        transpose_13 = paddle._C_ops.transpose(slice_2, [0, 1, 3, 2])
        del slice_2

        # pd_op.matmul: (2x1x32768x512xf32) <- (2x1x32768x32xf32, 2x1x32x512xf32)
        matmul_9 = paddle._C_ops.matmul(transpose_9, transpose_13, False, False)

        # pd_op.scale: (2x1x32768x512xf32) <- (2x1x32768x512xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(matmul_9, full_0, float("0"), True)
        del matmul_9

        # pd_op.softmax: (2x1x32768x512xf32) <- (2x1x32768x512xf32)
        softmax_1 = paddle._C_ops.softmax(scale_1, -1)
        del scale_1

        # pd_op.matmul: (2x1x32768x32xf32) <- (2x1x32768x512xf32, 2x1x512x32xf32)
        matmul_10 = paddle._C_ops.matmul(softmax_1, slice_3, False, False)

        # pd_op.transpose: (2x32768x1x32xf32) <- (2x1x32768x32xf32)
        transpose_14 = paddle._C_ops.transpose(matmul_10, [0, 2, 1, 3])
        del matmul_10

        # pd_op.reshape: (2x32768x32xf32) <- (2x32768x1x32xf32, 3xi64)
        reshape_14 = paddle._C_ops.reshape(transpose_14, full_int_array_8)
        del full_int_array_8

        # pd_op.matmul: (2x32768x32xf32) <- (2x32768x32xf32, 32x32xf32)
        matmul_11 = paddle._C_ops.matmul(reshape_14, parameter_156, False, False)
        del parameter_156

        # pd_op.add: (2x32768x32xf32) <- (2x32768x32xf32, 32xf32)
        add_13 = paddle._C_ops.add(matmul_11, parameter_155)
        del parameter_155

        # pd_op.full: (xf32) <- ()
        full_1 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_1,
            [],
            paddle.float32,
            [float("0.985714")],
            paddle.framework._current_expected_place(),
        )
        del full_1

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_10 = [2, 1, 1]

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_0 = paddle._C_ops.uniform(
            full_int_array_10,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_14 = paddle._C_ops.add(assign_value__0, uniform_0)
        del uniform_0

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_0 = paddle._C_ops.floor(add_14)
        del add_14

        # pd_op.divide: (2x32768x32xf32) <- (2x32768x32xf32, xf32)
        divide_0 = paddle._C_ops.divide(add_13, assign_value__0)

        # pd_op.multiply: (2x32768x32xf32) <- (2x32768x32xf32, 2x1x1xf32)
        multiply_0 = paddle._C_ops.multiply(divide_0, floor_0)

        # pd_op.add: (2x32768x32xf32) <- (2x32768x32xf32, 2x32768x32xf32)
        add_15 = paddle._C_ops.add(add_9, multiply_0)

        # pd_op.layer_norm: (2x32768x32xf32, 2x32768xf32, 2x32768xf32) <- (2x32768x32xf32, 32xf32, 32xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_15, parameter_154, parameter_153, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_153, parameter_154

        # pd_op.matmul: (2x32768x128xf32) <- (2x32768x32xf32, 32x128xf32)
        matmul_12 = paddle._C_ops.matmul(layer_norm_18, parameter_152, False, False)
        del parameter_152

        # pd_op.add: (2x32768x128xf32) <- (2x32768x128xf32, 128xf32)
        add_16 = paddle._C_ops.add(matmul_12, parameter_151)
        del parameter_151

        # pd_op.transpose: (2x128x32768xf32) <- (2x32768x128xf32)
        transpose_15 = paddle._C_ops.transpose(add_16, [0, 2, 1])
        del add_16

        # pd_op.reshape: (2x128x128x256xf32) <- (2x128x32768xf32, 4xi64)
        reshape_15 = paddle._C_ops.reshape(transpose_15, full_int_array_9)
        del full_int_array_9

        # pd_op.depthwise_conv2d: (2x128x128x256xf32) <- (2x128x128x256xf32, 128x1x3x3xf32)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(
            reshape_15, parameter_150, [1, 1], [1, 1], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del parameter_150

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_16 = paddle._C_ops.reshape(parameter_149, full_int_array_0)
        del parameter_149

        # pd_op.add: (2x128x128x256xf32) <- (2x128x128x256xf32, 1x128x1x1xf32)
        add_17 = paddle._C_ops.add(depthwise_conv2d_1, reshape_16)

        # pd_op.flatten: (2x128x32768xf32) <- (2x128x128x256xf32)
        flatten_2 = paddle._C_ops.flatten(add_17, 2, 3)

        # pd_op.transpose: (2x32768x128xf32) <- (2x128x32768xf32)
        transpose_16 = paddle._C_ops.transpose(flatten_2, [0, 2, 1])
        del flatten_2

        # pd_op.gelu: (2x32768x128xf32) <- (2x32768x128xf32)
        gelu_1 = paddle._C_ops.gelu(transpose_16, False)

        # pd_op.matmul: (2x32768x32xf32) <- (2x32768x128xf32, 128x32xf32)
        matmul_13 = paddle._C_ops.matmul(gelu_1, parameter_148, False, False)
        del parameter_148

        # pd_op.add: (2x32768x32xf32) <- (2x32768x32xf32, 32xf32)
        add_18 = paddle._C_ops.add(matmul_13, parameter_147)
        del parameter_147

        # pd_op.full: (xf32) <- ()
        full_4 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__1 = paddle._C_ops.assign_value_(
            full_4,
            [],
            paddle.float32,
            [float("0.985714")],
            paddle.framework._current_expected_place(),
        )
        del full_4

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_1 = paddle._C_ops.uniform(
            full_int_array_10,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_19 = paddle._C_ops.add(assign_value__1, uniform_1)
        del uniform_1

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_1 = paddle._C_ops.floor(add_19)
        del add_19

        # pd_op.divide: (2x32768x32xf32) <- (2x32768x32xf32, xf32)
        divide_1 = paddle._C_ops.divide(add_18, assign_value__1)

        # pd_op.multiply: (2x32768x32xf32) <- (2x32768x32xf32, 2x1x1xf32)
        multiply_1 = paddle._C_ops.multiply(divide_1, floor_1)

        # pd_op.add: (2x32768x32xf32) <- (2x32768x32xf32, 2x32768x32xf32)
        add_20 = paddle._C_ops.add(add_15, multiply_1)

        # pd_op.layer_norm: (2x32768x32xf32, 2x32768xf32, 2x32768xf32) <- (2x32768x32xf32, 32xf32, 32xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_20, parameter_146, parameter_145, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_145, parameter_146

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_11 = [2, 128, 256, 32]

        # pd_op.reshape: (2x128x256x32xf32) <- (2x32768x32xf32, 4xi64)
        reshape_17 = paddle._C_ops.reshape(layer_norm_21, full_int_array_11)
        del full_int_array_11

        # pd_op.transpose: (2x32x128x256xf32) <- (2x128x256x32xf32)
        transpose_17 = paddle._C_ops.transpose(reshape_17, [0, 3, 1, 2])
        del reshape_17

        # pd_op.conv2d: (2x64x64x128xf32) <- (2x32x128x256xf32, 64x32x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            transpose_17, parameter_144, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_144

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_18 = paddle._C_ops.reshape(parameter_143, full_int_array_0)
        del parameter_143

        # pd_op.add: (2x64x64x128xf32) <- (2x64x64x128xf32, 1x64x1x1xf32)
        add_21 = paddle._C_ops.add(conv2d_3, reshape_18)

        # pd_op.flatten: (2x64x8192xf32) <- (2x64x64x128xf32)
        flatten_3 = paddle._C_ops.flatten(add_21, 2, 3)

        # pd_op.transpose: (2x8192x64xf32) <- (2x64x8192xf32)
        transpose_18 = paddle._C_ops.transpose(flatten_3, [0, 2, 1])
        del flatten_3

        # pd_op.layer_norm: (2x8192x64xf32, 2x8192xf32, 2x8192xf32) <- (2x8192x64xf32, 64xf32, 64xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_18, parameter_142, parameter_141, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_141, parameter_142

        # pd_op.layer_norm: (2x8192x64xf32, 2x8192xf32, 2x8192xf32) <- (2x8192x64xf32, 64xf32, 64xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                layer_norm_24, parameter_140, parameter_139, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_139, parameter_140

        # pd_op.matmul: (2x8192x64xf32) <- (2x8192x64xf32, 64x64xf32)
        matmul_14 = paddle._C_ops.matmul(layer_norm_27, parameter_138, False, False)
        del parameter_138

        # pd_op.add: (2x8192x64xf32) <- (2x8192x64xf32, 64xf32)
        add_22 = paddle._C_ops.add(matmul_14, parameter_137)
        del parameter_137

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_12 = [2, 8192, 2, 32]

        # pd_op.reshape: (2x8192x2x32xf32) <- (2x8192x64xf32, 4xi64)
        reshape_19 = paddle._C_ops.reshape(add_22, full_int_array_12)

        # pd_op.transpose: (2x2x8192x32xf32) <- (2x8192x2x32xf32)
        transpose_19 = paddle._C_ops.transpose(reshape_19, [0, 2, 1, 3])
        del reshape_19

        # pd_op.transpose: (2x64x8192xf32) <- (2x8192x64xf32)
        transpose_20 = paddle._C_ops.transpose(layer_norm_27, [0, 2, 1])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_13 = [2, 64, 64, 128]

        # pd_op.reshape: (2x64x64x128xf32) <- (2x64x8192xf32, 4xi64)
        reshape_20 = paddle._C_ops.reshape(transpose_20, full_int_array_13)

        # pd_op.conv2d: (2x64x16x32xf32) <- (2x64x64x128xf32, 64x64x4x4xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            reshape_20, parameter_136, [4, 4], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_136

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_21 = paddle._C_ops.reshape(parameter_135, full_int_array_0)
        del parameter_135

        # pd_op.add: (2x64x16x32xf32) <- (2x64x16x32xf32, 1x64x1x1xf32)
        add_23 = paddle._C_ops.add(conv2d_4, reshape_21)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_14 = [2, 64, -1]

        # pd_op.reshape: (2x64x512xf32) <- (2x64x16x32xf32, 3xi64)
        reshape_22 = paddle._C_ops.reshape(add_23, full_int_array_14)

        # pd_op.transpose: (2x512x64xf32) <- (2x64x512xf32)
        transpose_21 = paddle._C_ops.transpose(reshape_22, [0, 2, 1])
        del reshape_22

        # pd_op.layer_norm: (2x512x64xf32, 2x512xf32, 2x512xf32) <- (2x512x64xf32, 64xf32, 64xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_21, parameter_134, parameter_133, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_133, parameter_134

        # pd_op.matmul: (2x512x128xf32) <- (2x512x64xf32, 64x128xf32)
        matmul_15 = paddle._C_ops.matmul(layer_norm_30, parameter_132, False, False)
        del parameter_132

        # pd_op.add: (2x512x128xf32) <- (2x512x128xf32, 128xf32)
        add_24 = paddle._C_ops.add(matmul_15, parameter_131)
        del parameter_131

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_15 = [2, -1, 2, 2, 32]

        # pd_op.reshape: (2x512x2x2x32xf32) <- (2x512x128xf32, 5xi64)
        reshape_23 = paddle._C_ops.reshape(add_24, full_int_array_15)

        # pd_op.transpose: (2x2x2x512x32xf32) <- (2x512x2x2x32xf32)
        transpose_22 = paddle._C_ops.transpose(reshape_23, [2, 0, 3, 1, 4])
        del reshape_23

        # pd_op.slice: (2x2x512x32xf32) <- (2x2x2x512x32xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            transpose_22, [0], full_int_array_5, full_int_array_6, [1], [0]
        )

        # pd_op.slice: (2x2x512x32xf32) <- (2x2x2x512x32xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            transpose_22, [0], full_int_array_6, full_int_array_7, [1], [0]
        )

        # pd_op.transpose: (2x2x32x512xf32) <- (2x2x512x32xf32)
        transpose_23 = paddle._C_ops.transpose(slice_4, [0, 1, 3, 2])
        del slice_4

        # pd_op.matmul: (2x2x8192x512xf32) <- (2x2x8192x32xf32, 2x2x32x512xf32)
        matmul_16 = paddle._C_ops.matmul(transpose_19, transpose_23, False, False)

        # pd_op.scale: (2x2x8192x512xf32) <- (2x2x8192x512xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(matmul_16, full_0, float("0"), True)
        del matmul_16

        # pd_op.softmax: (2x2x8192x512xf32) <- (2x2x8192x512xf32)
        softmax_2 = paddle._C_ops.softmax(scale_2, -1)
        del scale_2

        # pd_op.matmul: (2x2x8192x32xf32) <- (2x2x8192x512xf32, 2x2x512x32xf32)
        matmul_17 = paddle._C_ops.matmul(softmax_2, slice_5, False, False)

        # pd_op.transpose: (2x8192x2x32xf32) <- (2x2x8192x32xf32)
        transpose_24 = paddle._C_ops.transpose(matmul_17, [0, 2, 1, 3])
        del matmul_17

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_16 = [2, 8192, 64]

        # pd_op.reshape: (2x8192x64xf32) <- (2x8192x2x32xf32, 3xi64)
        reshape_24 = paddle._C_ops.reshape(transpose_24, full_int_array_16)

        # pd_op.matmul: (2x8192x64xf32) <- (2x8192x64xf32, 64x64xf32)
        matmul_18 = paddle._C_ops.matmul(reshape_24, parameter_130, False, False)
        del parameter_130

        # pd_op.add: (2x8192x64xf32) <- (2x8192x64xf32, 64xf32)
        add_25 = paddle._C_ops.add(matmul_18, parameter_129)
        del parameter_129

        # pd_op.full: (xf32) <- ()
        full_5 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__2 = paddle._C_ops.assign_value_(
            full_5,
            [],
            paddle.float32,
            [float("0.971429")],
            paddle.framework._current_expected_place(),
        )
        del full_5

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_2 = paddle._C_ops.uniform(
            full_int_array_10,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_26 = paddle._C_ops.add(assign_value__2, uniform_2)
        del uniform_2

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_2 = paddle._C_ops.floor(add_26)
        del add_26

        # pd_op.divide: (2x8192x64xf32) <- (2x8192x64xf32, xf32)
        divide_2 = paddle._C_ops.divide(add_25, assign_value__2)

        # pd_op.multiply: (2x8192x64xf32) <- (2x8192x64xf32, 2x1x1xf32)
        multiply_2 = paddle._C_ops.multiply(divide_2, floor_2)

        # pd_op.add: (2x8192x64xf32) <- (2x8192x64xf32, 2x8192x64xf32)
        add_27 = paddle._C_ops.add(layer_norm_24, multiply_2)

        # pd_op.layer_norm: (2x8192x64xf32, 2x8192xf32, 2x8192xf32) <- (2x8192x64xf32, 64xf32, 64xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_27, parameter_128, parameter_127, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_127, parameter_128

        # pd_op.matmul: (2x8192x256xf32) <- (2x8192x64xf32, 64x256xf32)
        matmul_19 = paddle._C_ops.matmul(layer_norm_33, parameter_126, False, False)
        del parameter_126

        # pd_op.add: (2x8192x256xf32) <- (2x8192x256xf32, 256xf32)
        add_28 = paddle._C_ops.add(matmul_19, parameter_125)
        del parameter_125

        # pd_op.transpose: (2x256x8192xf32) <- (2x8192x256xf32)
        transpose_25 = paddle._C_ops.transpose(add_28, [0, 2, 1])
        del add_28

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_17 = [2, 256, 64, 128]

        # pd_op.reshape: (2x256x64x128xf32) <- (2x256x8192xf32, 4xi64)
        reshape_25 = paddle._C_ops.reshape(transpose_25, full_int_array_17)

        # pd_op.depthwise_conv2d: (2x256x64x128xf32) <- (2x256x64x128xf32, 256x1x3x3xf32)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(
            reshape_25, parameter_124, [1, 1], [1, 1], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del parameter_124

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_26 = paddle._C_ops.reshape(parameter_123, full_int_array_0)
        del parameter_123

        # pd_op.add: (2x256x64x128xf32) <- (2x256x64x128xf32, 1x256x1x1xf32)
        add_29 = paddle._C_ops.add(depthwise_conv2d_2, reshape_26)

        # pd_op.flatten: (2x256x8192xf32) <- (2x256x64x128xf32)
        flatten_4 = paddle._C_ops.flatten(add_29, 2, 3)

        # pd_op.transpose: (2x8192x256xf32) <- (2x256x8192xf32)
        transpose_26 = paddle._C_ops.transpose(flatten_4, [0, 2, 1])
        del flatten_4

        # pd_op.gelu: (2x8192x256xf32) <- (2x8192x256xf32)
        gelu_2 = paddle._C_ops.gelu(transpose_26, False)

        # pd_op.matmul: (2x8192x64xf32) <- (2x8192x256xf32, 256x64xf32)
        matmul_20 = paddle._C_ops.matmul(gelu_2, parameter_122, False, False)
        del parameter_122

        # pd_op.add: (2x8192x64xf32) <- (2x8192x64xf32, 64xf32)
        add_30 = paddle._C_ops.add(matmul_20, parameter_121)
        del parameter_121

        # pd_op.full: (xf32) <- ()
        full_6 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__3 = paddle._C_ops.assign_value_(
            full_6,
            [],
            paddle.float32,
            [float("0.971429")],
            paddle.framework._current_expected_place(),
        )
        del full_6

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_3 = paddle._C_ops.uniform(
            full_int_array_10,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_31 = paddle._C_ops.add(assign_value__3, uniform_3)
        del uniform_3

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_3 = paddle._C_ops.floor(add_31)
        del add_31

        # pd_op.divide: (2x8192x64xf32) <- (2x8192x64xf32, xf32)
        divide_3 = paddle._C_ops.divide(add_30, assign_value__3)

        # pd_op.multiply: (2x8192x64xf32) <- (2x8192x64xf32, 2x1x1xf32)
        multiply_3 = paddle._C_ops.multiply(divide_3, floor_3)

        # pd_op.add: (2x8192x64xf32) <- (2x8192x64xf32, 2x8192x64xf32)
        add_32 = paddle._C_ops.add(add_27, multiply_3)

        # pd_op.layer_norm: (2x8192x64xf32, 2x8192xf32, 2x8192xf32) <- (2x8192x64xf32, 64xf32, 64xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_32, parameter_120, parameter_119, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_119, parameter_120

        # pd_op.matmul: (2x8192x64xf32) <- (2x8192x64xf32, 64x64xf32)
        matmul_21 = paddle._C_ops.matmul(layer_norm_36, parameter_118, False, False)
        del parameter_118

        # pd_op.add: (2x8192x64xf32) <- (2x8192x64xf32, 64xf32)
        add_33 = paddle._C_ops.add(matmul_21, parameter_117)
        del parameter_117

        # pd_op.reshape: (2x8192x2x32xf32) <- (2x8192x64xf32, 4xi64)
        reshape_27 = paddle._C_ops.reshape(add_33, full_int_array_12)
        del full_int_array_12

        # pd_op.transpose: (2x2x8192x32xf32) <- (2x8192x2x32xf32)
        transpose_27 = paddle._C_ops.transpose(reshape_27, [0, 2, 1, 3])
        del reshape_27

        # pd_op.transpose: (2x64x8192xf32) <- (2x8192x64xf32)
        transpose_28 = paddle._C_ops.transpose(layer_norm_36, [0, 2, 1])

        # pd_op.reshape: (2x64x64x128xf32) <- (2x64x8192xf32, 4xi64)
        reshape_28 = paddle._C_ops.reshape(transpose_28, full_int_array_13)
        del full_int_array_13

        # pd_op.conv2d: (2x64x16x32xf32) <- (2x64x64x128xf32, 64x64x4x4xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            reshape_28, parameter_116, [4, 4], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_116

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_29 = paddle._C_ops.reshape(parameter_115, full_int_array_0)
        del parameter_115

        # pd_op.add: (2x64x16x32xf32) <- (2x64x16x32xf32, 1x64x1x1xf32)
        add_34 = paddle._C_ops.add(conv2d_5, reshape_29)

        # pd_op.reshape: (2x64x512xf32) <- (2x64x16x32xf32, 3xi64)
        reshape_30 = paddle._C_ops.reshape(add_34, full_int_array_14)
        del full_int_array_14

        # pd_op.transpose: (2x512x64xf32) <- (2x64x512xf32)
        transpose_29 = paddle._C_ops.transpose(reshape_30, [0, 2, 1])
        del reshape_30

        # pd_op.layer_norm: (2x512x64xf32, 2x512xf32, 2x512xf32) <- (2x512x64xf32, 64xf32, 64xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_29, parameter_114, parameter_113, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_113, parameter_114

        # pd_op.matmul: (2x512x128xf32) <- (2x512x64xf32, 64x128xf32)
        matmul_22 = paddle._C_ops.matmul(layer_norm_39, parameter_112, False, False)
        del parameter_112

        # pd_op.add: (2x512x128xf32) <- (2x512x128xf32, 128xf32)
        add_35 = paddle._C_ops.add(matmul_22, parameter_111)
        del parameter_111

        # pd_op.reshape: (2x512x2x2x32xf32) <- (2x512x128xf32, 5xi64)
        reshape_31 = paddle._C_ops.reshape(add_35, full_int_array_15)
        del full_int_array_15

        # pd_op.transpose: (2x2x2x512x32xf32) <- (2x512x2x2x32xf32)
        transpose_30 = paddle._C_ops.transpose(reshape_31, [2, 0, 3, 1, 4])
        del reshape_31

        # pd_op.slice: (2x2x512x32xf32) <- (2x2x2x512x32xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            transpose_30, [0], full_int_array_5, full_int_array_6, [1], [0]
        )

        # pd_op.slice: (2x2x512x32xf32) <- (2x2x2x512x32xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            transpose_30, [0], full_int_array_6, full_int_array_7, [1], [0]
        )

        # pd_op.transpose: (2x2x32x512xf32) <- (2x2x512x32xf32)
        transpose_31 = paddle._C_ops.transpose(slice_6, [0, 1, 3, 2])
        del slice_6

        # pd_op.matmul: (2x2x8192x512xf32) <- (2x2x8192x32xf32, 2x2x32x512xf32)
        matmul_23 = paddle._C_ops.matmul(transpose_27, transpose_31, False, False)

        # pd_op.scale: (2x2x8192x512xf32) <- (2x2x8192x512xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(matmul_23, full_0, float("0"), True)
        del matmul_23

        # pd_op.softmax: (2x2x8192x512xf32) <- (2x2x8192x512xf32)
        softmax_3 = paddle._C_ops.softmax(scale_3, -1)
        del scale_3

        # pd_op.matmul: (2x2x8192x32xf32) <- (2x2x8192x512xf32, 2x2x512x32xf32)
        matmul_24 = paddle._C_ops.matmul(softmax_3, slice_7, False, False)

        # pd_op.transpose: (2x8192x2x32xf32) <- (2x2x8192x32xf32)
        transpose_32 = paddle._C_ops.transpose(matmul_24, [0, 2, 1, 3])
        del matmul_24

        # pd_op.reshape: (2x8192x64xf32) <- (2x8192x2x32xf32, 3xi64)
        reshape_32 = paddle._C_ops.reshape(transpose_32, full_int_array_16)
        del full_int_array_16

        # pd_op.matmul: (2x8192x64xf32) <- (2x8192x64xf32, 64x64xf32)
        matmul_25 = paddle._C_ops.matmul(reshape_32, parameter_110, False, False)
        del parameter_110

        # pd_op.add: (2x8192x64xf32) <- (2x8192x64xf32, 64xf32)
        add_36 = paddle._C_ops.add(matmul_25, parameter_109)
        del parameter_109

        # pd_op.full: (xf32) <- ()
        full_7 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__4 = paddle._C_ops.assign_value_(
            full_7,
            [],
            paddle.float32,
            [float("0.957143")],
            paddle.framework._current_expected_place(),
        )
        del full_7

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_4 = paddle._C_ops.uniform(
            full_int_array_10,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_37 = paddle._C_ops.add(assign_value__4, uniform_4)
        del uniform_4

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_4 = paddle._C_ops.floor(add_37)
        del add_37

        # pd_op.divide: (2x8192x64xf32) <- (2x8192x64xf32, xf32)
        divide_4 = paddle._C_ops.divide(add_36, assign_value__4)

        # pd_op.multiply: (2x8192x64xf32) <- (2x8192x64xf32, 2x1x1xf32)
        multiply_4 = paddle._C_ops.multiply(divide_4, floor_4)

        # pd_op.add: (2x8192x64xf32) <- (2x8192x64xf32, 2x8192x64xf32)
        add_38 = paddle._C_ops.add(add_32, multiply_4)

        # pd_op.layer_norm: (2x8192x64xf32, 2x8192xf32, 2x8192xf32) <- (2x8192x64xf32, 64xf32, 64xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_38, parameter_108, parameter_107, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_107, parameter_108

        # pd_op.matmul: (2x8192x256xf32) <- (2x8192x64xf32, 64x256xf32)
        matmul_26 = paddle._C_ops.matmul(layer_norm_42, parameter_106, False, False)
        del parameter_106

        # pd_op.add: (2x8192x256xf32) <- (2x8192x256xf32, 256xf32)
        add_39 = paddle._C_ops.add(matmul_26, parameter_105)
        del parameter_105

        # pd_op.transpose: (2x256x8192xf32) <- (2x8192x256xf32)
        transpose_33 = paddle._C_ops.transpose(add_39, [0, 2, 1])
        del add_39

        # pd_op.reshape: (2x256x64x128xf32) <- (2x256x8192xf32, 4xi64)
        reshape_33 = paddle._C_ops.reshape(transpose_33, full_int_array_17)
        del full_int_array_17

        # pd_op.depthwise_conv2d: (2x256x64x128xf32) <- (2x256x64x128xf32, 256x1x3x3xf32)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(
            reshape_33, parameter_104, [1, 1], [1, 1], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del parameter_104

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_34 = paddle._C_ops.reshape(parameter_103, full_int_array_0)
        del parameter_103

        # pd_op.add: (2x256x64x128xf32) <- (2x256x64x128xf32, 1x256x1x1xf32)
        add_40 = paddle._C_ops.add(depthwise_conv2d_3, reshape_34)

        # pd_op.flatten: (2x256x8192xf32) <- (2x256x64x128xf32)
        flatten_5 = paddle._C_ops.flatten(add_40, 2, 3)

        # pd_op.transpose: (2x8192x256xf32) <- (2x256x8192xf32)
        transpose_34 = paddle._C_ops.transpose(flatten_5, [0, 2, 1])
        del flatten_5

        # pd_op.gelu: (2x8192x256xf32) <- (2x8192x256xf32)
        gelu_3 = paddle._C_ops.gelu(transpose_34, False)

        # pd_op.matmul: (2x8192x64xf32) <- (2x8192x256xf32, 256x64xf32)
        matmul_27 = paddle._C_ops.matmul(gelu_3, parameter_102, False, False)
        del parameter_102

        # pd_op.add: (2x8192x64xf32) <- (2x8192x64xf32, 64xf32)
        add_41 = paddle._C_ops.add(matmul_27, parameter_101)
        del parameter_101

        # pd_op.full: (xf32) <- ()
        full_8 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__5 = paddle._C_ops.assign_value_(
            full_8,
            [],
            paddle.float32,
            [float("0.957143")],
            paddle.framework._current_expected_place(),
        )
        del full_8

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_5 = paddle._C_ops.uniform(
            full_int_array_10,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_42 = paddle._C_ops.add(assign_value__5, uniform_5)
        del uniform_5

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_5 = paddle._C_ops.floor(add_42)
        del add_42

        # pd_op.divide: (2x8192x64xf32) <- (2x8192x64xf32, xf32)
        divide_5 = paddle._C_ops.divide(add_41, assign_value__5)

        # pd_op.multiply: (2x8192x64xf32) <- (2x8192x64xf32, 2x1x1xf32)
        multiply_5 = paddle._C_ops.multiply(divide_5, floor_5)

        # pd_op.add: (2x8192x64xf32) <- (2x8192x64xf32, 2x8192x64xf32)
        add_43 = paddle._C_ops.add(add_38, multiply_5)

        # pd_op.layer_norm: (2x8192x64xf32, 2x8192xf32, 2x8192xf32) <- (2x8192x64xf32, 64xf32, 64xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_43, parameter_100, parameter_99, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_100, parameter_99

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_18 = [2, 64, 128, 64]

        # pd_op.reshape: (2x64x128x64xf32) <- (2x8192x64xf32, 4xi64)
        reshape_35 = paddle._C_ops.reshape(layer_norm_45, full_int_array_18)
        del full_int_array_18

        # pd_op.transpose: (2x64x64x128xf32) <- (2x64x128x64xf32)
        transpose_35 = paddle._C_ops.transpose(reshape_35, [0, 3, 1, 2])
        del reshape_35

        # pd_op.conv2d: (2x160x32x64xf32) <- (2x64x64x128xf32, 160x64x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            transpose_35, parameter_98, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_98

        # pd_op.reshape: (1x160x1x1xf32) <- (160xf32, 4xi64)
        reshape_36 = paddle._C_ops.reshape(parameter_97, full_int_array_0)
        del parameter_97

        # pd_op.add: (2x160x32x64xf32) <- (2x160x32x64xf32, 1x160x1x1xf32)
        add_44 = paddle._C_ops.add(conv2d_6, reshape_36)

        # pd_op.flatten: (2x160x2048xf32) <- (2x160x32x64xf32)
        flatten_6 = paddle._C_ops.flatten(add_44, 2, 3)

        # pd_op.transpose: (2x2048x160xf32) <- (2x160x2048xf32)
        transpose_36 = paddle._C_ops.transpose(flatten_6, [0, 2, 1])
        del flatten_6

        # pd_op.layer_norm: (2x2048x160xf32, 2x2048xf32, 2x2048xf32) <- (2x2048x160xf32, 160xf32, 160xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_36, parameter_96, parameter_95, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_95, parameter_96

        # pd_op.layer_norm: (2x2048x160xf32, 2x2048xf32, 2x2048xf32) <- (2x2048x160xf32, 160xf32, 160xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                layer_norm_48, parameter_94, parameter_93, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_93, parameter_94

        # pd_op.matmul: (2x2048x160xf32) <- (2x2048x160xf32, 160x160xf32)
        matmul_28 = paddle._C_ops.matmul(layer_norm_51, parameter_92, False, False)
        del parameter_92

        # pd_op.add: (2x2048x160xf32) <- (2x2048x160xf32, 160xf32)
        add_45 = paddle._C_ops.add(matmul_28, parameter_91)
        del parameter_91

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_19 = [2, 2048, 5, 32]

        # pd_op.reshape: (2x2048x5x32xf32) <- (2x2048x160xf32, 4xi64)
        reshape_37 = paddle._C_ops.reshape(add_45, full_int_array_19)

        # pd_op.transpose: (2x5x2048x32xf32) <- (2x2048x5x32xf32)
        transpose_37 = paddle._C_ops.transpose(reshape_37, [0, 2, 1, 3])
        del reshape_37

        # pd_op.transpose: (2x160x2048xf32) <- (2x2048x160xf32)
        transpose_38 = paddle._C_ops.transpose(layer_norm_51, [0, 2, 1])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_20 = [2, 160, 32, 64]

        # pd_op.reshape: (2x160x32x64xf32) <- (2x160x2048xf32, 4xi64)
        reshape_38 = paddle._C_ops.reshape(transpose_38, full_int_array_20)

        # pd_op.conv2d: (2x160x16x32xf32) <- (2x160x32x64xf32, 160x160x2x2xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            reshape_38, parameter_90, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_90

        # pd_op.reshape: (1x160x1x1xf32) <- (160xf32, 4xi64)
        reshape_39 = paddle._C_ops.reshape(parameter_89, full_int_array_0)
        del parameter_89

        # pd_op.add: (2x160x16x32xf32) <- (2x160x16x32xf32, 1x160x1x1xf32)
        add_46 = paddle._C_ops.add(conv2d_7, reshape_39)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_21 = [2, 160, -1]

        # pd_op.reshape: (2x160x512xf32) <- (2x160x16x32xf32, 3xi64)
        reshape_40 = paddle._C_ops.reshape(add_46, full_int_array_21)

        # pd_op.transpose: (2x512x160xf32) <- (2x160x512xf32)
        transpose_39 = paddle._C_ops.transpose(reshape_40, [0, 2, 1])
        del reshape_40

        # pd_op.layer_norm: (2x512x160xf32, 2x512xf32, 2x512xf32) <- (2x512x160xf32, 160xf32, 160xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_39, parameter_88, parameter_87, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_87, parameter_88

        # pd_op.matmul: (2x512x320xf32) <- (2x512x160xf32, 160x320xf32)
        matmul_29 = paddle._C_ops.matmul(layer_norm_54, parameter_86, False, False)
        del parameter_86

        # pd_op.add: (2x512x320xf32) <- (2x512x320xf32, 320xf32)
        add_47 = paddle._C_ops.add(matmul_29, parameter_85)
        del parameter_85

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_22 = [2, -1, 2, 5, 32]

        # pd_op.reshape: (2x512x2x5x32xf32) <- (2x512x320xf32, 5xi64)
        reshape_41 = paddle._C_ops.reshape(add_47, full_int_array_22)

        # pd_op.transpose: (2x2x5x512x32xf32) <- (2x512x2x5x32xf32)
        transpose_40 = paddle._C_ops.transpose(reshape_41, [2, 0, 3, 1, 4])
        del reshape_41

        # pd_op.slice: (2x5x512x32xf32) <- (2x2x5x512x32xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            transpose_40, [0], full_int_array_5, full_int_array_6, [1], [0]
        )

        # pd_op.slice: (2x5x512x32xf32) <- (2x2x5x512x32xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            transpose_40, [0], full_int_array_6, full_int_array_7, [1], [0]
        )

        # pd_op.transpose: (2x5x32x512xf32) <- (2x5x512x32xf32)
        transpose_41 = paddle._C_ops.transpose(slice_8, [0, 1, 3, 2])
        del slice_8

        # pd_op.matmul: (2x5x2048x512xf32) <- (2x5x2048x32xf32, 2x5x32x512xf32)
        matmul_30 = paddle._C_ops.matmul(transpose_37, transpose_41, False, False)

        # pd_op.scale: (2x5x2048x512xf32) <- (2x5x2048x512xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(matmul_30, full_0, float("0"), True)
        del matmul_30

        # pd_op.softmax: (2x5x2048x512xf32) <- (2x5x2048x512xf32)
        softmax_4 = paddle._C_ops.softmax(scale_4, -1)
        del scale_4

        # pd_op.matmul: (2x5x2048x32xf32) <- (2x5x2048x512xf32, 2x5x512x32xf32)
        matmul_31 = paddle._C_ops.matmul(softmax_4, slice_9, False, False)

        # pd_op.transpose: (2x2048x5x32xf32) <- (2x5x2048x32xf32)
        transpose_42 = paddle._C_ops.transpose(matmul_31, [0, 2, 1, 3])
        del matmul_31

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_23 = [2, 2048, 160]

        # pd_op.reshape: (2x2048x160xf32) <- (2x2048x5x32xf32, 3xi64)
        reshape_42 = paddle._C_ops.reshape(transpose_42, full_int_array_23)

        # pd_op.matmul: (2x2048x160xf32) <- (2x2048x160xf32, 160x160xf32)
        matmul_32 = paddle._C_ops.matmul(reshape_42, parameter_84, False, False)
        del parameter_84

        # pd_op.add: (2x2048x160xf32) <- (2x2048x160xf32, 160xf32)
        add_48 = paddle._C_ops.add(matmul_32, parameter_83)
        del parameter_83

        # pd_op.full: (xf32) <- ()
        full_9 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__6 = paddle._C_ops.assign_value_(
            full_9,
            [],
            paddle.float32,
            [float("0.942857")],
            paddle.framework._current_expected_place(),
        )
        del full_9

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_6 = paddle._C_ops.uniform(
            full_int_array_10,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_49 = paddle._C_ops.add(assign_value__6, uniform_6)
        del uniform_6

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_6 = paddle._C_ops.floor(add_49)
        del add_49

        # pd_op.divide: (2x2048x160xf32) <- (2x2048x160xf32, xf32)
        divide_6 = paddle._C_ops.divide(add_48, assign_value__6)

        # pd_op.multiply: (2x2048x160xf32) <- (2x2048x160xf32, 2x1x1xf32)
        multiply_6 = paddle._C_ops.multiply(divide_6, floor_6)

        # pd_op.add: (2x2048x160xf32) <- (2x2048x160xf32, 2x2048x160xf32)
        add_50 = paddle._C_ops.add(layer_norm_48, multiply_6)

        # pd_op.layer_norm: (2x2048x160xf32, 2x2048xf32, 2x2048xf32) <- (2x2048x160xf32, 160xf32, 160xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_50, parameter_82, parameter_81, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_81, parameter_82

        # pd_op.matmul: (2x2048x640xf32) <- (2x2048x160xf32, 160x640xf32)
        matmul_33 = paddle._C_ops.matmul(layer_norm_57, parameter_80, False, False)
        del parameter_80

        # pd_op.add: (2x2048x640xf32) <- (2x2048x640xf32, 640xf32)
        add_51 = paddle._C_ops.add(matmul_33, parameter_79)
        del parameter_79

        # pd_op.transpose: (2x640x2048xf32) <- (2x2048x640xf32)
        transpose_43 = paddle._C_ops.transpose(add_51, [0, 2, 1])
        del add_51

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_24 = [2, 640, 32, 64]

        # pd_op.reshape: (2x640x32x64xf32) <- (2x640x2048xf32, 4xi64)
        reshape_43 = paddle._C_ops.reshape(transpose_43, full_int_array_24)

        # pd_op.depthwise_conv2d: (2x640x32x64xf32) <- (2x640x32x64xf32, 640x1x3x3xf32)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(
            reshape_43, parameter_78, [1, 1], [1, 1], "EXPLICIT", 640, [1, 1], "NCHW"
        )
        del parameter_78

        # pd_op.reshape: (1x640x1x1xf32) <- (640xf32, 4xi64)
        reshape_44 = paddle._C_ops.reshape(parameter_77, full_int_array_0)
        del parameter_77

        # pd_op.add: (2x640x32x64xf32) <- (2x640x32x64xf32, 1x640x1x1xf32)
        add_52 = paddle._C_ops.add(depthwise_conv2d_4, reshape_44)

        # pd_op.flatten: (2x640x2048xf32) <- (2x640x32x64xf32)
        flatten_7 = paddle._C_ops.flatten(add_52, 2, 3)

        # pd_op.transpose: (2x2048x640xf32) <- (2x640x2048xf32)
        transpose_44 = paddle._C_ops.transpose(flatten_7, [0, 2, 1])
        del flatten_7

        # pd_op.gelu: (2x2048x640xf32) <- (2x2048x640xf32)
        gelu_4 = paddle._C_ops.gelu(transpose_44, False)

        # pd_op.matmul: (2x2048x160xf32) <- (2x2048x640xf32, 640x160xf32)
        matmul_34 = paddle._C_ops.matmul(gelu_4, parameter_76, False, False)
        del parameter_76

        # pd_op.add: (2x2048x160xf32) <- (2x2048x160xf32, 160xf32)
        add_53 = paddle._C_ops.add(matmul_34, parameter_75)
        del parameter_75

        # pd_op.full: (xf32) <- ()
        full_10 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__7 = paddle._C_ops.assign_value_(
            full_10,
            [],
            paddle.float32,
            [float("0.942857")],
            paddle.framework._current_expected_place(),
        )
        del full_10

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_7 = paddle._C_ops.uniform(
            full_int_array_10,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_54 = paddle._C_ops.add(assign_value__7, uniform_7)
        del uniform_7

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_7 = paddle._C_ops.floor(add_54)
        del add_54

        # pd_op.divide: (2x2048x160xf32) <- (2x2048x160xf32, xf32)
        divide_7 = paddle._C_ops.divide(add_53, assign_value__7)

        # pd_op.multiply: (2x2048x160xf32) <- (2x2048x160xf32, 2x1x1xf32)
        multiply_7 = paddle._C_ops.multiply(divide_7, floor_7)

        # pd_op.add: (2x2048x160xf32) <- (2x2048x160xf32, 2x2048x160xf32)
        add_55 = paddle._C_ops.add(add_50, multiply_7)

        # pd_op.layer_norm: (2x2048x160xf32, 2x2048xf32, 2x2048xf32) <- (2x2048x160xf32, 160xf32, 160xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_55, parameter_74, parameter_73, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_73, parameter_74

        # pd_op.matmul: (2x2048x160xf32) <- (2x2048x160xf32, 160x160xf32)
        matmul_35 = paddle._C_ops.matmul(layer_norm_60, parameter_72, False, False)
        del parameter_72

        # pd_op.add: (2x2048x160xf32) <- (2x2048x160xf32, 160xf32)
        add_56 = paddle._C_ops.add(matmul_35, parameter_71)
        del parameter_71

        # pd_op.reshape: (2x2048x5x32xf32) <- (2x2048x160xf32, 4xi64)
        reshape_45 = paddle._C_ops.reshape(add_56, full_int_array_19)
        del full_int_array_19

        # pd_op.transpose: (2x5x2048x32xf32) <- (2x2048x5x32xf32)
        transpose_45 = paddle._C_ops.transpose(reshape_45, [0, 2, 1, 3])
        del reshape_45

        # pd_op.transpose: (2x160x2048xf32) <- (2x2048x160xf32)
        transpose_46 = paddle._C_ops.transpose(layer_norm_60, [0, 2, 1])

        # pd_op.reshape: (2x160x32x64xf32) <- (2x160x2048xf32, 4xi64)
        reshape_46 = paddle._C_ops.reshape(transpose_46, full_int_array_20)
        del full_int_array_20

        # pd_op.conv2d: (2x160x16x32xf32) <- (2x160x32x64xf32, 160x160x2x2xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            reshape_46, parameter_70, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_70

        # pd_op.reshape: (1x160x1x1xf32) <- (160xf32, 4xi64)
        reshape_47 = paddle._C_ops.reshape(parameter_69, full_int_array_0)
        del parameter_69

        # pd_op.add: (2x160x16x32xf32) <- (2x160x16x32xf32, 1x160x1x1xf32)
        add_57 = paddle._C_ops.add(conv2d_8, reshape_47)

        # pd_op.reshape: (2x160x512xf32) <- (2x160x16x32xf32, 3xi64)
        reshape_48 = paddle._C_ops.reshape(add_57, full_int_array_21)
        del full_int_array_21

        # pd_op.transpose: (2x512x160xf32) <- (2x160x512xf32)
        transpose_47 = paddle._C_ops.transpose(reshape_48, [0, 2, 1])
        del reshape_48

        # pd_op.layer_norm: (2x512x160xf32, 2x512xf32, 2x512xf32) <- (2x512x160xf32, 160xf32, 160xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_47, parameter_68, parameter_67, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_67, parameter_68

        # pd_op.matmul: (2x512x320xf32) <- (2x512x160xf32, 160x320xf32)
        matmul_36 = paddle._C_ops.matmul(layer_norm_63, parameter_66, False, False)
        del parameter_66

        # pd_op.add: (2x512x320xf32) <- (2x512x320xf32, 320xf32)
        add_58 = paddle._C_ops.add(matmul_36, parameter_65)
        del parameter_65

        # pd_op.reshape: (2x512x2x5x32xf32) <- (2x512x320xf32, 5xi64)
        reshape_49 = paddle._C_ops.reshape(add_58, full_int_array_22)
        del full_int_array_22

        # pd_op.transpose: (2x2x5x512x32xf32) <- (2x512x2x5x32xf32)
        transpose_48 = paddle._C_ops.transpose(reshape_49, [2, 0, 3, 1, 4])
        del reshape_49

        # pd_op.slice: (2x5x512x32xf32) <- (2x2x5x512x32xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            transpose_48, [0], full_int_array_5, full_int_array_6, [1], [0]
        )

        # pd_op.slice: (2x5x512x32xf32) <- (2x2x5x512x32xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            transpose_48, [0], full_int_array_6, full_int_array_7, [1], [0]
        )

        # pd_op.transpose: (2x5x32x512xf32) <- (2x5x512x32xf32)
        transpose_49 = paddle._C_ops.transpose(slice_10, [0, 1, 3, 2])
        del slice_10

        # pd_op.matmul: (2x5x2048x512xf32) <- (2x5x2048x32xf32, 2x5x32x512xf32)
        matmul_37 = paddle._C_ops.matmul(transpose_45, transpose_49, False, False)

        # pd_op.scale: (2x5x2048x512xf32) <- (2x5x2048x512xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(matmul_37, full_0, float("0"), True)
        del matmul_37

        # pd_op.softmax: (2x5x2048x512xf32) <- (2x5x2048x512xf32)
        softmax_5 = paddle._C_ops.softmax(scale_5, -1)
        del scale_5

        # pd_op.matmul: (2x5x2048x32xf32) <- (2x5x2048x512xf32, 2x5x512x32xf32)
        matmul_38 = paddle._C_ops.matmul(softmax_5, slice_11, False, False)

        # pd_op.transpose: (2x2048x5x32xf32) <- (2x5x2048x32xf32)
        transpose_50 = paddle._C_ops.transpose(matmul_38, [0, 2, 1, 3])
        del matmul_38

        # pd_op.reshape: (2x2048x160xf32) <- (2x2048x5x32xf32, 3xi64)
        reshape_50 = paddle._C_ops.reshape(transpose_50, full_int_array_23)
        del full_int_array_23

        # pd_op.matmul: (2x2048x160xf32) <- (2x2048x160xf32, 160x160xf32)
        matmul_39 = paddle._C_ops.matmul(reshape_50, parameter_64, False, False)
        del parameter_64

        # pd_op.add: (2x2048x160xf32) <- (2x2048x160xf32, 160xf32)
        add_59 = paddle._C_ops.add(matmul_39, parameter_63)
        del parameter_63

        # pd_op.full: (xf32) <- ()
        full_11 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__8 = paddle._C_ops.assign_value_(
            full_11,
            [],
            paddle.float32,
            [float("0.928571")],
            paddle.framework._current_expected_place(),
        )
        del full_11

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_8 = paddle._C_ops.uniform(
            full_int_array_10,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_60 = paddle._C_ops.add(assign_value__8, uniform_8)
        del uniform_8

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_8 = paddle._C_ops.floor(add_60)
        del add_60

        # pd_op.divide: (2x2048x160xf32) <- (2x2048x160xf32, xf32)
        divide_8 = paddle._C_ops.divide(add_59, assign_value__8)

        # pd_op.multiply: (2x2048x160xf32) <- (2x2048x160xf32, 2x1x1xf32)
        multiply_8 = paddle._C_ops.multiply(divide_8, floor_8)

        # pd_op.add: (2x2048x160xf32) <- (2x2048x160xf32, 2x2048x160xf32)
        add_61 = paddle._C_ops.add(add_55, multiply_8)

        # pd_op.layer_norm: (2x2048x160xf32, 2x2048xf32, 2x2048xf32) <- (2x2048x160xf32, 160xf32, 160xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_61, parameter_62, parameter_61, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_61, parameter_62

        # pd_op.matmul: (2x2048x640xf32) <- (2x2048x160xf32, 160x640xf32)
        matmul_40 = paddle._C_ops.matmul(layer_norm_66, parameter_60, False, False)
        del parameter_60

        # pd_op.add: (2x2048x640xf32) <- (2x2048x640xf32, 640xf32)
        add_62 = paddle._C_ops.add(matmul_40, parameter_59)
        del parameter_59

        # pd_op.transpose: (2x640x2048xf32) <- (2x2048x640xf32)
        transpose_51 = paddle._C_ops.transpose(add_62, [0, 2, 1])
        del add_62

        # pd_op.reshape: (2x640x32x64xf32) <- (2x640x2048xf32, 4xi64)
        reshape_51 = paddle._C_ops.reshape(transpose_51, full_int_array_24)
        del full_int_array_24

        # pd_op.depthwise_conv2d: (2x640x32x64xf32) <- (2x640x32x64xf32, 640x1x3x3xf32)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(
            reshape_51, parameter_58, [1, 1], [1, 1], "EXPLICIT", 640, [1, 1], "NCHW"
        )
        del parameter_58

        # pd_op.reshape: (1x640x1x1xf32) <- (640xf32, 4xi64)
        reshape_52 = paddle._C_ops.reshape(parameter_57, full_int_array_0)
        del parameter_57

        # pd_op.add: (2x640x32x64xf32) <- (2x640x32x64xf32, 1x640x1x1xf32)
        add_63 = paddle._C_ops.add(depthwise_conv2d_5, reshape_52)

        # pd_op.flatten: (2x640x2048xf32) <- (2x640x32x64xf32)
        flatten_8 = paddle._C_ops.flatten(add_63, 2, 3)

        # pd_op.transpose: (2x2048x640xf32) <- (2x640x2048xf32)
        transpose_52 = paddle._C_ops.transpose(flatten_8, [0, 2, 1])
        del flatten_8

        # pd_op.gelu: (2x2048x640xf32) <- (2x2048x640xf32)
        gelu_5 = paddle._C_ops.gelu(transpose_52, False)

        # pd_op.matmul: (2x2048x160xf32) <- (2x2048x640xf32, 640x160xf32)
        matmul_41 = paddle._C_ops.matmul(gelu_5, parameter_56, False, False)
        del parameter_56

        # pd_op.add: (2x2048x160xf32) <- (2x2048x160xf32, 160xf32)
        add_64 = paddle._C_ops.add(matmul_41, parameter_55)
        del parameter_55

        # pd_op.full: (xf32) <- ()
        full_12 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__9 = paddle._C_ops.assign_value_(
            full_12,
            [],
            paddle.float32,
            [float("0.928571")],
            paddle.framework._current_expected_place(),
        )
        del full_12

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_9 = paddle._C_ops.uniform(
            full_int_array_10,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_65 = paddle._C_ops.add(assign_value__9, uniform_9)
        del uniform_9

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_9 = paddle._C_ops.floor(add_65)
        del add_65

        # pd_op.divide: (2x2048x160xf32) <- (2x2048x160xf32, xf32)
        divide_9 = paddle._C_ops.divide(add_64, assign_value__9)

        # pd_op.multiply: (2x2048x160xf32) <- (2x2048x160xf32, 2x1x1xf32)
        multiply_9 = paddle._C_ops.multiply(divide_9, floor_9)

        # pd_op.add: (2x2048x160xf32) <- (2x2048x160xf32, 2x2048x160xf32)
        add_66 = paddle._C_ops.add(add_61, multiply_9)

        # pd_op.layer_norm: (2x2048x160xf32, 2x2048xf32, 2x2048xf32) <- (2x2048x160xf32, 160xf32, 160xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_66, parameter_54, parameter_53, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_53, parameter_54

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_25 = [2, 32, 64, 160]

        # pd_op.reshape: (2x32x64x160xf32) <- (2x2048x160xf32, 4xi64)
        reshape_53 = paddle._C_ops.reshape(layer_norm_69, full_int_array_25)
        del full_int_array_25

        # pd_op.transpose: (2x160x32x64xf32) <- (2x32x64x160xf32)
        transpose_53 = paddle._C_ops.transpose(reshape_53, [0, 3, 1, 2])
        del reshape_53

        # pd_op.conv2d: (2x256x16x32xf32) <- (2x160x32x64xf32, 256x160x3x3xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            transpose_53, parameter_52, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_52

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_54 = paddle._C_ops.reshape(parameter_51, full_int_array_0)
        del parameter_51

        # pd_op.add: (2x256x16x32xf32) <- (2x256x16x32xf32, 1x256x1x1xf32)
        add_67 = paddle._C_ops.add(conv2d_9, reshape_54)

        # pd_op.flatten: (2x256x512xf32) <- (2x256x16x32xf32)
        flatten_9 = paddle._C_ops.flatten(add_67, 2, 3)

        # pd_op.transpose: (2x512x256xf32) <- (2x256x512xf32)
        transpose_54 = paddle._C_ops.transpose(flatten_9, [0, 2, 1])
        del flatten_9

        # pd_op.layer_norm: (2x512x256xf32, 2x512xf32, 2x512xf32) <- (2x512x256xf32, 256xf32, 256xf32)
        layer_norm_72, layer_norm_73, layer_norm_74 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_54, parameter_50, parameter_49, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_49, parameter_50

        # pd_op.layer_norm: (2x512x256xf32, 2x512xf32, 2x512xf32) <- (2x512x256xf32, 256xf32, 256xf32)
        layer_norm_75, layer_norm_76, layer_norm_77 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                layer_norm_72, parameter_48, parameter_47, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_47, parameter_48

        # pd_op.matmul: (2x512x256xf32) <- (2x512x256xf32, 256x256xf32)
        matmul_42 = paddle._C_ops.matmul(layer_norm_75, parameter_46, False, False)
        del parameter_46

        # pd_op.add: (2x512x256xf32) <- (2x512x256xf32, 256xf32)
        add_68 = paddle._C_ops.add(matmul_42, parameter_45)
        del parameter_45

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_26 = [2, 512, 8, 32]

        # pd_op.reshape: (2x512x8x32xf32) <- (2x512x256xf32, 4xi64)
        reshape_55 = paddle._C_ops.reshape(add_68, full_int_array_26)

        # pd_op.transpose: (2x8x512x32xf32) <- (2x512x8x32xf32)
        transpose_55 = paddle._C_ops.transpose(reshape_55, [0, 2, 1, 3])
        del reshape_55

        # pd_op.matmul: (2x512x512xf32) <- (2x512x256xf32, 256x512xf32)
        matmul_43 = paddle._C_ops.matmul(layer_norm_75, parameter_44, False, False)
        del parameter_44

        # pd_op.add: (2x512x512xf32) <- (2x512x512xf32, 512xf32)
        add_69 = paddle._C_ops.add(matmul_43, parameter_43)
        del parameter_43

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_27 = [2, -1, 2, 8, 32]

        # pd_op.reshape: (2x512x2x8x32xf32) <- (2x512x512xf32, 5xi64)
        reshape_56 = paddle._C_ops.reshape(add_69, full_int_array_27)

        # pd_op.transpose: (2x2x8x512x32xf32) <- (2x512x2x8x32xf32)
        transpose_56 = paddle._C_ops.transpose(reshape_56, [2, 0, 3, 1, 4])
        del reshape_56

        # pd_op.slice: (2x8x512x32xf32) <- (2x2x8x512x32xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            transpose_56, [0], full_int_array_5, full_int_array_6, [1], [0]
        )

        # pd_op.slice: (2x8x512x32xf32) <- (2x2x8x512x32xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            transpose_56, [0], full_int_array_6, full_int_array_7, [1], [0]
        )

        # pd_op.transpose: (2x8x32x512xf32) <- (2x8x512x32xf32)
        transpose_57 = paddle._C_ops.transpose(slice_12, [0, 1, 3, 2])
        del slice_12

        # pd_op.matmul: (2x8x512x512xf32) <- (2x8x512x32xf32, 2x8x32x512xf32)
        matmul_44 = paddle._C_ops.matmul(transpose_55, transpose_57, False, False)

        # pd_op.scale: (2x8x512x512xf32) <- (2x8x512x512xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(matmul_44, full_0, float("0"), True)
        del matmul_44

        # pd_op.softmax: (2x8x512x512xf32) <- (2x8x512x512xf32)
        softmax_6 = paddle._C_ops.softmax(scale_6, -1)
        del scale_6

        # pd_op.matmul: (2x8x512x32xf32) <- (2x8x512x512xf32, 2x8x512x32xf32)
        matmul_45 = paddle._C_ops.matmul(softmax_6, slice_13, False, False)

        # pd_op.transpose: (2x512x8x32xf32) <- (2x8x512x32xf32)
        transpose_58 = paddle._C_ops.transpose(matmul_45, [0, 2, 1, 3])
        del matmul_45

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_28 = [2, 512, 256]

        # pd_op.reshape: (2x512x256xf32) <- (2x512x8x32xf32, 3xi64)
        reshape_57 = paddle._C_ops.reshape(transpose_58, full_int_array_28)

        # pd_op.matmul: (2x512x256xf32) <- (2x512x256xf32, 256x256xf32)
        matmul_46 = paddle._C_ops.matmul(reshape_57, parameter_42, False, False)
        del parameter_42

        # pd_op.add: (2x512x256xf32) <- (2x512x256xf32, 256xf32)
        add_70 = paddle._C_ops.add(matmul_46, parameter_41)
        del parameter_41

        # pd_op.full: (xf32) <- ()
        full_13 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__10 = paddle._C_ops.assign_value_(
            full_13,
            [],
            paddle.float32,
            [float("0.914286")],
            paddle.framework._current_expected_place(),
        )
        del full_13

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_10 = paddle._C_ops.uniform(
            full_int_array_10,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_71 = paddle._C_ops.add(assign_value__10, uniform_10)
        del uniform_10

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_10 = paddle._C_ops.floor(add_71)
        del add_71

        # pd_op.divide: (2x512x256xf32) <- (2x512x256xf32, xf32)
        divide_10 = paddle._C_ops.divide(add_70, assign_value__10)

        # pd_op.multiply: (2x512x256xf32) <- (2x512x256xf32, 2x1x1xf32)
        multiply_10 = paddle._C_ops.multiply(divide_10, floor_10)

        # pd_op.add: (2x512x256xf32) <- (2x512x256xf32, 2x512x256xf32)
        add_72 = paddle._C_ops.add(layer_norm_72, multiply_10)

        # pd_op.layer_norm: (2x512x256xf32, 2x512xf32, 2x512xf32) <- (2x512x256xf32, 256xf32, 256xf32)
        layer_norm_78, layer_norm_79, layer_norm_80 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_72, parameter_40, parameter_39, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_39, parameter_40

        # pd_op.matmul: (2x512x1024xf32) <- (2x512x256xf32, 256x1024xf32)
        matmul_47 = paddle._C_ops.matmul(layer_norm_78, parameter_38, False, False)
        del parameter_38

        # pd_op.add: (2x512x1024xf32) <- (2x512x1024xf32, 1024xf32)
        add_73 = paddle._C_ops.add(matmul_47, parameter_37)
        del parameter_37

        # pd_op.transpose: (2x1024x512xf32) <- (2x512x1024xf32)
        transpose_59 = paddle._C_ops.transpose(add_73, [0, 2, 1])
        del add_73

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_29 = [2, 1024, 16, 32]

        # pd_op.reshape: (2x1024x16x32xf32) <- (2x1024x512xf32, 4xi64)
        reshape_58 = paddle._C_ops.reshape(transpose_59, full_int_array_29)

        # pd_op.depthwise_conv2d: (2x1024x16x32xf32) <- (2x1024x16x32xf32, 1024x1x3x3xf32)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(
            reshape_58, parameter_36, [1, 1], [1, 1], "EXPLICIT", 1024, [1, 1], "NCHW"
        )
        del parameter_36

        # pd_op.reshape: (1x1024x1x1xf32) <- (1024xf32, 4xi64)
        reshape_59 = paddle._C_ops.reshape(parameter_35, full_int_array_0)
        del parameter_35

        # pd_op.add: (2x1024x16x32xf32) <- (2x1024x16x32xf32, 1x1024x1x1xf32)
        add_74 = paddle._C_ops.add(depthwise_conv2d_6, reshape_59)

        # pd_op.flatten: (2x1024x512xf32) <- (2x1024x16x32xf32)
        flatten_10 = paddle._C_ops.flatten(add_74, 2, 3)

        # pd_op.transpose: (2x512x1024xf32) <- (2x1024x512xf32)
        transpose_60 = paddle._C_ops.transpose(flatten_10, [0, 2, 1])
        del flatten_10

        # pd_op.gelu: (2x512x1024xf32) <- (2x512x1024xf32)
        gelu_6 = paddle._C_ops.gelu(transpose_60, False)

        # pd_op.matmul: (2x512x256xf32) <- (2x512x1024xf32, 1024x256xf32)
        matmul_48 = paddle._C_ops.matmul(gelu_6, parameter_34, False, False)
        del parameter_34

        # pd_op.add: (2x512x256xf32) <- (2x512x256xf32, 256xf32)
        add_75 = paddle._C_ops.add(matmul_48, parameter_33)
        del parameter_33

        # pd_op.full: (xf32) <- ()
        full_14 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__11 = paddle._C_ops.assign_value_(
            full_14,
            [],
            paddle.float32,
            [float("0.914286")],
            paddle.framework._current_expected_place(),
        )
        del full_14

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_11 = paddle._C_ops.uniform(
            full_int_array_10,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_76 = paddle._C_ops.add(assign_value__11, uniform_11)
        del uniform_11

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_11 = paddle._C_ops.floor(add_76)
        del add_76

        # pd_op.divide: (2x512x256xf32) <- (2x512x256xf32, xf32)
        divide_11 = paddle._C_ops.divide(add_75, assign_value__11)

        # pd_op.multiply: (2x512x256xf32) <- (2x512x256xf32, 2x1x1xf32)
        multiply_11 = paddle._C_ops.multiply(divide_11, floor_11)

        # pd_op.add: (2x512x256xf32) <- (2x512x256xf32, 2x512x256xf32)
        add_77 = paddle._C_ops.add(add_72, multiply_11)

        # pd_op.layer_norm: (2x512x256xf32, 2x512xf32, 2x512xf32) <- (2x512x256xf32, 256xf32, 256xf32)
        layer_norm_81, layer_norm_82, layer_norm_83 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_77, parameter_32, parameter_31, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_31, parameter_32

        # pd_op.matmul: (2x512x256xf32) <- (2x512x256xf32, 256x256xf32)
        matmul_49 = paddle._C_ops.matmul(layer_norm_81, parameter_30, False, False)
        del parameter_30

        # pd_op.add: (2x512x256xf32) <- (2x512x256xf32, 256xf32)
        add_78 = paddle._C_ops.add(matmul_49, parameter_29)
        del parameter_29

        # pd_op.reshape: (2x512x8x32xf32) <- (2x512x256xf32, 4xi64)
        reshape_60 = paddle._C_ops.reshape(add_78, full_int_array_26)
        del full_int_array_26

        # pd_op.transpose: (2x8x512x32xf32) <- (2x512x8x32xf32)
        transpose_61 = paddle._C_ops.transpose(reshape_60, [0, 2, 1, 3])
        del reshape_60

        # pd_op.matmul: (2x512x512xf32) <- (2x512x256xf32, 256x512xf32)
        matmul_50 = paddle._C_ops.matmul(layer_norm_81, parameter_28, False, False)
        del parameter_28

        # pd_op.add: (2x512x512xf32) <- (2x512x512xf32, 512xf32)
        add_79 = paddle._C_ops.add(matmul_50, parameter_27)
        del parameter_27

        # pd_op.reshape: (2x512x2x8x32xf32) <- (2x512x512xf32, 5xi64)
        reshape_61 = paddle._C_ops.reshape(add_79, full_int_array_27)
        del full_int_array_27

        # pd_op.transpose: (2x2x8x512x32xf32) <- (2x512x2x8x32xf32)
        transpose_62 = paddle._C_ops.transpose(reshape_61, [2, 0, 3, 1, 4])
        del reshape_61

        # pd_op.slice: (2x8x512x32xf32) <- (2x2x8x512x32xf32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            transpose_62, [0], full_int_array_5, full_int_array_6, [1], [0]
        )

        # pd_op.slice: (2x8x512x32xf32) <- (2x2x8x512x32xf32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            transpose_62, [0], full_int_array_6, full_int_array_7, [1], [0]
        )

        # pd_op.transpose: (2x8x32x512xf32) <- (2x8x512x32xf32)
        transpose_63 = paddle._C_ops.transpose(slice_14, [0, 1, 3, 2])
        del slice_14

        # pd_op.matmul: (2x8x512x512xf32) <- (2x8x512x32xf32, 2x8x32x512xf32)
        matmul_51 = paddle._C_ops.matmul(transpose_61, transpose_63, False, False)

        # pd_op.scale: (2x8x512x512xf32) <- (2x8x512x512xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(matmul_51, full_0, float("0"), True)
        del matmul_51

        # pd_op.softmax: (2x8x512x512xf32) <- (2x8x512x512xf32)
        softmax_7 = paddle._C_ops.softmax(scale_7, -1)
        del scale_7

        # pd_op.matmul: (2x8x512x32xf32) <- (2x8x512x512xf32, 2x8x512x32xf32)
        matmul_52 = paddle._C_ops.matmul(softmax_7, slice_15, False, False)

        # pd_op.transpose: (2x512x8x32xf32) <- (2x8x512x32xf32)
        transpose_64 = paddle._C_ops.transpose(matmul_52, [0, 2, 1, 3])
        del matmul_52

        # pd_op.reshape: (2x512x256xf32) <- (2x512x8x32xf32, 3xi64)
        reshape_62 = paddle._C_ops.reshape(transpose_64, full_int_array_28)
        del full_int_array_28

        # pd_op.matmul: (2x512x256xf32) <- (2x512x256xf32, 256x256xf32)
        matmul_53 = paddle._C_ops.matmul(reshape_62, parameter_26, False, False)
        del parameter_26

        # pd_op.add: (2x512x256xf32) <- (2x512x256xf32, 256xf32)
        add_80 = paddle._C_ops.add(matmul_53, parameter_25)
        del parameter_25

        # pd_op.full: (xf32) <- ()
        full_15 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__12 = paddle._C_ops.assign_value_(
            full_15,
            [],
            paddle.float32,
            [float("0.9")],
            paddle.framework._current_expected_place(),
        )
        del full_15

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_12 = paddle._C_ops.uniform(
            full_int_array_10,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_81 = paddle._C_ops.add(assign_value__12, uniform_12)
        del uniform_12

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_12 = paddle._C_ops.floor(add_81)
        del add_81

        # pd_op.divide: (2x512x256xf32) <- (2x512x256xf32, xf32)
        divide_12 = paddle._C_ops.divide(add_80, assign_value__12)

        # pd_op.multiply: (2x512x256xf32) <- (2x512x256xf32, 2x1x1xf32)
        multiply_12 = paddle._C_ops.multiply(divide_12, floor_12)

        # pd_op.add: (2x512x256xf32) <- (2x512x256xf32, 2x512x256xf32)
        add_82 = paddle._C_ops.add(add_77, multiply_12)

        # pd_op.layer_norm: (2x512x256xf32, 2x512xf32, 2x512xf32) <- (2x512x256xf32, 256xf32, 256xf32)
        layer_norm_84, layer_norm_85, layer_norm_86 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_82, parameter_24, parameter_23, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_23, parameter_24

        # pd_op.matmul: (2x512x1024xf32) <- (2x512x256xf32, 256x1024xf32)
        matmul_54 = paddle._C_ops.matmul(layer_norm_84, parameter_22, False, False)
        del parameter_22

        # pd_op.add: (2x512x1024xf32) <- (2x512x1024xf32, 1024xf32)
        add_83 = paddle._C_ops.add(matmul_54, parameter_21)
        del parameter_21

        # pd_op.transpose: (2x1024x512xf32) <- (2x512x1024xf32)
        transpose_65 = paddle._C_ops.transpose(add_83, [0, 2, 1])
        del add_83

        # pd_op.reshape: (2x1024x16x32xf32) <- (2x1024x512xf32, 4xi64)
        reshape_63 = paddle._C_ops.reshape(transpose_65, full_int_array_29)
        del full_int_array_29

        # pd_op.depthwise_conv2d: (2x1024x16x32xf32) <- (2x1024x16x32xf32, 1024x1x3x3xf32)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(
            reshape_63, parameter_20, [1, 1], [1, 1], "EXPLICIT", 1024, [1, 1], "NCHW"
        )
        del parameter_20

        # pd_op.reshape: (1x1024x1x1xf32) <- (1024xf32, 4xi64)
        reshape_64 = paddle._C_ops.reshape(parameter_19, full_int_array_0)
        del parameter_19

        # pd_op.add: (2x1024x16x32xf32) <- (2x1024x16x32xf32, 1x1024x1x1xf32)
        add_84 = paddle._C_ops.add(depthwise_conv2d_7, reshape_64)

        # pd_op.flatten: (2x1024x512xf32) <- (2x1024x16x32xf32)
        flatten_11 = paddle._C_ops.flatten(add_84, 2, 3)

        # pd_op.transpose: (2x512x1024xf32) <- (2x1024x512xf32)
        transpose_66 = paddle._C_ops.transpose(flatten_11, [0, 2, 1])
        del flatten_11

        # pd_op.gelu: (2x512x1024xf32) <- (2x512x1024xf32)
        gelu_7 = paddle._C_ops.gelu(transpose_66, False)

        # pd_op.matmul: (2x512x256xf32) <- (2x512x1024xf32, 1024x256xf32)
        matmul_55 = paddle._C_ops.matmul(gelu_7, parameter_18, False, False)
        del parameter_18

        # pd_op.add: (2x512x256xf32) <- (2x512x256xf32, 256xf32)
        add_85 = paddle._C_ops.add(matmul_55, parameter_17)
        del parameter_17

        # pd_op.full: (xf32) <- ()
        full_16 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__13 = paddle._C_ops.assign_value_(
            full_16,
            [],
            paddle.float32,
            [float("0.9")],
            paddle.framework._current_expected_place(),
        )
        del full_16

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_13 = paddle._C_ops.uniform(
            full_int_array_10,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del full_int_array_10

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_86 = paddle._C_ops.add(assign_value__13, uniform_13)
        del uniform_13

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_13 = paddle._C_ops.floor(add_86)
        del add_86

        # pd_op.divide: (2x512x256xf32) <- (2x512x256xf32, xf32)
        divide_13 = paddle._C_ops.divide(add_85, assign_value__13)

        # pd_op.multiply: (2x512x256xf32) <- (2x512x256xf32, 2x1x1xf32)
        multiply_13 = paddle._C_ops.multiply(divide_13, floor_13)

        # pd_op.add: (2x512x256xf32) <- (2x512x256xf32, 2x512x256xf32)
        add_87 = paddle._C_ops.add(add_82, multiply_13)

        # pd_op.layer_norm: (2x512x256xf32, 2x512xf32, 2x512xf32) <- (2x512x256xf32, 256xf32, 256xf32)
        layer_norm_87, layer_norm_88, layer_norm_89 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_87, parameter_16, parameter_15, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_15, parameter_16

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_30 = [2, 16, 32, 256]

        # pd_op.reshape: (2x16x32x256xf32) <- (2x512x256xf32, 4xi64)
        reshape_65 = paddle._C_ops.reshape(layer_norm_87, full_int_array_30)
        del full_int_array_30

        # pd_op.transpose: (2x256x16x32xf32) <- (2x16x32x256xf32)
        transpose_67 = paddle._C_ops.transpose(reshape_65, [0, 3, 1, 2])
        del reshape_65

        # pd_op.flatten: (2x256x512xf32) <- (2x256x16x32xf32)
        flatten_12 = paddle._C_ops.flatten(transpose_67, 2, 3)

        # pd_op.transpose: (2x512x256xf32) <- (2x256x512xf32)
        transpose_68 = paddle._C_ops.transpose(flatten_12, [0, 2, 1])
        del flatten_12

        # pd_op.matmul: (2x512x256xf32) <- (2x512x256xf32, 256x256xf32)
        matmul_56 = paddle._C_ops.matmul(transpose_68, parameter_14, False, False)
        del parameter_14

        # pd_op.add: (2x512x256xf32) <- (2x512x256xf32, 256xf32)
        add_88 = paddle._C_ops.add(matmul_56, parameter_13)
        del parameter_13

        # pd_op.transpose: (2x256x512xf32) <- (2x512x256xf32)
        transpose_69 = paddle._C_ops.transpose(add_88, [0, 2, 1])
        del add_88

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_31 = [0, 0, 16, 32]

        # pd_op.reshape: (2x256x16x32xf32) <- (2x256x512xf32, 4xi64)
        reshape_66 = paddle._C_ops.reshape(transpose_69, full_int_array_31)
        del full_int_array_31

        # pd_op.bilinear_interp: (2x256x128x256xf32) <- (2x256x16x32xf32, None, None, None)
        bilinear_interp_1 = paddle._C_ops.bilinear_interp(
            reshape_66, None, None, None, "NCHW", -1, 128, 256, [], "bilinear", False, 0
        )

        # pd_op.flatten: (2x160x2048xf32) <- (2x160x32x64xf32)
        flatten_13 = paddle._C_ops.flatten(transpose_53, 2, 3)

        # pd_op.transpose: (2x2048x160xf32) <- (2x160x2048xf32)
        transpose_70 = paddle._C_ops.transpose(flatten_13, [0, 2, 1])
        del flatten_13

        # pd_op.matmul: (2x2048x256xf32) <- (2x2048x160xf32, 160x256xf32)
        matmul_57 = paddle._C_ops.matmul(transpose_70, parameter_12, False, False)
        del parameter_12

        # pd_op.add: (2x2048x256xf32) <- (2x2048x256xf32, 256xf32)
        add_89 = paddle._C_ops.add(matmul_57, parameter_11)
        del parameter_11

        # pd_op.transpose: (2x256x2048xf32) <- (2x2048x256xf32)
        transpose_71 = paddle._C_ops.transpose(add_89, [0, 2, 1])
        del add_89

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_32 = [0, 0, 32, 64]

        # pd_op.reshape: (2x256x32x64xf32) <- (2x256x2048xf32, 4xi64)
        reshape_67 = paddle._C_ops.reshape(transpose_71, full_int_array_32)
        del full_int_array_32

        # pd_op.bilinear_interp: (2x256x128x256xf32) <- (2x256x32x64xf32, None, None, None)
        bilinear_interp_2 = paddle._C_ops.bilinear_interp(
            reshape_67, None, None, None, "NCHW", -1, 128, 256, [], "bilinear", False, 0
        )

        # pd_op.flatten: (2x64x8192xf32) <- (2x64x64x128xf32)
        flatten_14 = paddle._C_ops.flatten(transpose_35, 2, 3)

        # pd_op.transpose: (2x8192x64xf32) <- (2x64x8192xf32)
        transpose_72 = paddle._C_ops.transpose(flatten_14, [0, 2, 1])
        del flatten_14

        # pd_op.matmul: (2x8192x256xf32) <- (2x8192x64xf32, 64x256xf32)
        matmul_58 = paddle._C_ops.matmul(transpose_72, parameter_10, False, False)
        del parameter_10

        # pd_op.add: (2x8192x256xf32) <- (2x8192x256xf32, 256xf32)
        add_90 = paddle._C_ops.add(matmul_58, parameter_9)
        del parameter_9

        # pd_op.transpose: (2x256x8192xf32) <- (2x8192x256xf32)
        transpose_73 = paddle._C_ops.transpose(add_90, [0, 2, 1])
        del add_90

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_33 = [0, 0, 64, 128]

        # pd_op.reshape: (2x256x64x128xf32) <- (2x256x8192xf32, 4xi64)
        reshape_68 = paddle._C_ops.reshape(transpose_73, full_int_array_33)
        del full_int_array_33

        # pd_op.bilinear_interp: (2x256x128x256xf32) <- (2x256x64x128xf32, None, None, None)
        bilinear_interp_3 = paddle._C_ops.bilinear_interp(
            reshape_68, None, None, None, "NCHW", -1, 128, 256, [], "bilinear", False, 0
        )

        # pd_op.flatten: (2x32x32768xf32) <- (2x32x128x256xf32)
        flatten_15 = paddle._C_ops.flatten(transpose_17, 2, 3)

        # pd_op.transpose: (2x32768x32xf32) <- (2x32x32768xf32)
        transpose_74 = paddle._C_ops.transpose(flatten_15, [0, 2, 1])
        del flatten_15

        # pd_op.matmul: (2x32768x256xf32) <- (2x32768x32xf32, 32x256xf32)
        matmul_59 = paddle._C_ops.matmul(transpose_74, parameter_8, False, False)
        del parameter_8

        # pd_op.add: (2x32768x256xf32) <- (2x32768x256xf32, 256xf32)
        add_91 = paddle._C_ops.add(matmul_59, parameter_7)
        del parameter_7

        # pd_op.transpose: (2x256x32768xf32) <- (2x32768x256xf32)
        transpose_75 = paddle._C_ops.transpose(add_91, [0, 2, 1])
        del add_91

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_34 = [0, 0, 128, 256]

        # pd_op.reshape: (2x256x128x256xf32) <- (2x256x32768xf32, 4xi64)
        reshape_69 = paddle._C_ops.reshape(transpose_75, full_int_array_34)
        del full_int_array_34

        # pd_op.full: (1xi32) <- ()
        full_17 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([2x256x128x256xf32, 2x256x128x256xf32, 2x256x128x256xf32, 2x256x128x256xf32]) <- (2x256x128x256xf32, 2x256x128x256xf32, 2x256x128x256xf32, 2x256x128x256xf32)
        combine_0 = [
            bilinear_interp_1,
            bilinear_interp_2,
            bilinear_interp_3,
            reshape_69,
        ]

        # pd_op.concat: (2x1024x128x256xf32) <- ([2x256x128x256xf32, 2x256x128x256xf32, 2x256x128x256xf32, 2x256x128x256xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_17)
        del combine_0

        # pd_op.conv2d: (2x256x128x256xf32) <- (2x1024x128x256xf32, 256x1024x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            concat_0, parameter_6, [1, 1], [0, 0], "SAME", [1, 1], 1, "NCHW"
        )
        del parameter_6

        # pd_op.batch_norm_: (2x256x128x256xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x128x256xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_2, parameter_3, parameter_4, parameter_5

        # pd_op.relu: (2x256x128x256xf32) <- (2x256x128x256xf32)
        relu_0 = paddle._C_ops.relu(batch_norm__0)
        del batch_norm__0

        # pd_op.full: (1xf32) <- ()
        full_18 = paddle._C_ops.full(
            [1], float("1.11111"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (2x256x128x256xf32) <- (2x256x128x256xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(relu_0, full_18, float("0"), True)

        # pd_op.shape64: (4xi64) <- (2x256x128x256xf32)
        shape64_0 = paddle._C_ops.shape64(relu_0)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_5, full_int_array_6, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_6, full_int_array_7, [1], [0]
        )
        del shape64_0

        # pd_op.full: (xi64) <- ()
        full_19 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_1 = [slice_16, slice_17, full_19, full_19]
        del full_19, slice_16, slice_17

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_1, 0)
        del combine_1

        # pd_op.uniform: (-1x-1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_14 = paddle._C_ops.uniform(
            stack_0,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del full_2, full_3, stack_0

        # pd_op.full: (1xf32) <- ()
        full_20 = paddle._C_ops.full(
            [1],
            float("0.1"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.greater_equal: (-1x-1x1x1xb) <- (-1x-1x1x1xf32, 1xf32)
        greater_equal_0 = paddle._C_ops.greater_equal(uniform_14, full_20)
        del full_20, uniform_14

        # pd_op.cast: (2x256x128x256xf32) <- (2x256x128x256xf32)
        cast_0 = paddle._C_ops.cast(scale_8, paddle.float32)
        del scale_8

        # pd_op.cast: (-1x-1x1x1xf32) <- (-1x-1x1x1xb)
        cast_1 = paddle._C_ops.cast(greater_equal_0, paddle.float32)
        del greater_equal_0

        # pd_op.multiply: (2x256x128x256xf32) <- (2x256x128x256xf32, -1x-1x1x1xf32)
        multiply_14 = paddle._C_ops.multiply(cast_0, cast_1)

        # pd_op.conv2d: (2x2x128x256xf32) <- (2x256x128x256xf32, 2x256x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            multiply_14, parameter_1, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_1

        # pd_op.reshape: (1x2x1x1xf32) <- (2xf32, 4xi64)
        reshape_70 = paddle._C_ops.reshape(parameter_0, full_int_array_0)
        del full_int_array_0, parameter_0

        # pd_op.add: (2x2x128x256xf32) <- (2x2x128x256xf32, 1x2x1x1xf32)
        add_92 = paddle._C_ops.add(conv2d_11, reshape_70)

        # pd_op.bilinear_interp: (2x2x512x1024xf32) <- (2x2x128x256xf32, None, None, None)
        bilinear_interp_0 = paddle._C_ops.bilinear_interp(
            add_92, None, None, None, "NCHW", -1, 512, 1024, [], "bilinear", False, 0
        )
        del (
            add_0,
            add_1,
            add_10,
            add_11,
            add_12,
            add_13,
            add_15,
            add_17,
            add_18,
            add_2,
            add_20,
            add_21,
            add_22,
            add_23,
            add_24,
            add_25,
            add_27,
            add_29,
            add_3,
            add_30,
            add_32,
            add_33,
            add_34,
            add_35,
            add_36,
            add_38,
            add_4,
            add_40,
            add_41,
            add_43,
            add_44,
            add_45,
            add_46,
            add_47,
            add_48,
            add_5,
            add_50,
            add_52,
            add_53,
            add_55,
            add_56,
            add_57,
            add_58,
            add_59,
            add_61,
            add_63,
            add_64,
            add_66,
            add_67,
            add_68,
            add_69,
            add_7,
            add_70,
            add_72,
            add_74,
            add_75,
            add_77,
            add_78,
            add_79,
            add_8,
            add_80,
            add_82,
            add_84,
            add_85,
            add_87,
            add_9,
            add_92,
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
            assign_4,
            assign_5,
            assign_6,
            assign_7,
            assign_8,
            assign_9,
            assign_value__0,
            assign_value__1,
            assign_value__10,
            assign_value__11,
            assign_value__12,
            assign_value__13,
            assign_value__2,
            assign_value__3,
            assign_value__4,
            assign_value__5,
            assign_value__6,
            assign_value__7,
            assign_value__8,
            assign_value__9,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
            batch_norm__5,
            bilinear_interp_1,
            bilinear_interp_2,
            bilinear_interp_3,
            cast_0,
            cast_1,
            concat_0,
            conv2d_0,
            conv2d_1,
            conv2d_10,
            conv2d_11,
            conv2d_2,
            conv2d_3,
            conv2d_4,
            conv2d_5,
            conv2d_6,
            conv2d_7,
            conv2d_8,
            conv2d_9,
            depthwise_conv2d_0,
            depthwise_conv2d_1,
            depthwise_conv2d_2,
            depthwise_conv2d_3,
            depthwise_conv2d_4,
            depthwise_conv2d_5,
            depthwise_conv2d_6,
            depthwise_conv2d_7,
            divide_0,
            divide_1,
            divide_10,
            divide_11,
            divide_12,
            divide_13,
            divide_2,
            divide_3,
            divide_4,
            divide_5,
            divide_6,
            divide_7,
            divide_8,
            divide_9,
            floor_0,
            floor_1,
            floor_10,
            floor_11,
            floor_12,
            floor_13,
            floor_2,
            floor_3,
            floor_4,
            floor_5,
            floor_6,
            floor_7,
            floor_8,
            floor_9,
            full_0,
            full_17,
            full_18,
            full_int_array_5,
            full_int_array_6,
            full_int_array_7,
            gelu_0,
            gelu_1,
            gelu_2,
            gelu_3,
            gelu_4,
            gelu_5,
            gelu_6,
            gelu_7,
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
            layer_norm_87,
            layer_norm_88,
            layer_norm_89,
            layer_norm_9,
            matmul_0,
            matmul_1,
            matmul_11,
            matmul_12,
            matmul_13,
            matmul_14,
            matmul_15,
            matmul_18,
            matmul_19,
            matmul_20,
            matmul_21,
            matmul_22,
            matmul_25,
            matmul_26,
            matmul_27,
            matmul_28,
            matmul_29,
            matmul_32,
            matmul_33,
            matmul_34,
            matmul_35,
            matmul_36,
            matmul_39,
            matmul_4,
            matmul_40,
            matmul_41,
            matmul_42,
            matmul_43,
            matmul_46,
            matmul_47,
            matmul_48,
            matmul_49,
            matmul_5,
            matmul_50,
            matmul_53,
            matmul_54,
            matmul_55,
            matmul_56,
            matmul_57,
            matmul_58,
            matmul_59,
            matmul_6,
            matmul_7,
            matmul_8,
            multiply_0,
            multiply_1,
            multiply_10,
            multiply_11,
            multiply_12,
            multiply_13,
            multiply_14,
            multiply_2,
            multiply_3,
            multiply_4,
            multiply_5,
            multiply_6,
            multiply_7,
            multiply_8,
            multiply_9,
            relu_0,
            reshape_0,
            reshape_10,
            reshape_11,
            reshape_14,
            reshape_15,
            reshape_16,
            reshape_18,
            reshape_2,
            reshape_20,
            reshape_21,
            reshape_24,
            reshape_25,
            reshape_26,
            reshape_28,
            reshape_29,
            reshape_3,
            reshape_32,
            reshape_33,
            reshape_34,
            reshape_36,
            reshape_38,
            reshape_39,
            reshape_42,
            reshape_43,
            reshape_44,
            reshape_46,
            reshape_47,
            reshape_50,
            reshape_51,
            reshape_52,
            reshape_54,
            reshape_57,
            reshape_58,
            reshape_59,
            reshape_6,
            reshape_62,
            reshape_63,
            reshape_64,
            reshape_66,
            reshape_67,
            reshape_68,
            reshape_69,
            reshape_7,
            reshape_70,
            reshape_8,
            slice_1,
            slice_11,
            slice_13,
            slice_15,
            slice_3,
            slice_5,
            slice_7,
            slice_9,
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
            transpose_40,
            transpose_41,
            transpose_42,
            transpose_43,
            transpose_44,
            transpose_45,
            transpose_46,
            transpose_47,
            transpose_48,
            transpose_49,
            transpose_5,
            transpose_50,
            transpose_51,
            transpose_52,
            transpose_53,
            transpose_54,
            transpose_55,
            transpose_56,
            transpose_57,
            transpose_58,
            transpose_59,
            transpose_6,
            transpose_60,
            transpose_61,
            transpose_62,
            transpose_63,
            transpose_64,
            transpose_65,
            transpose_66,
            transpose_67,
            transpose_68,
            transpose_69,
            transpose_7,
            transpose_70,
            transpose_71,
            transpose_72,
            transpose_73,
            transpose_74,
            transpose_75,
            transpose_8,
            transpose_9,
        )

        return bilinear_interp_0
