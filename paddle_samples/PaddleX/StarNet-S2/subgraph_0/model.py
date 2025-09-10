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
        parameter_191,
        parameter_192,
        parameter_193,
        parameter_194,
        parameter_195,
        parameter_196,
        parameter_197,
        parameter_198,
        parameter_199,
        parameter_200,
        parameter_201,
        parameter_202,
        parameter_203,
        parameter_204,
        parameter_205,
        parameter_206,
        parameter_207,
        parameter_208,
        parameter_209,
        parameter_210,
        parameter_211,
        parameter_212,
        parameter_213,
        parameter_214,
        parameter_215,
        parameter_216,
        parameter_217,
        parameter_218,
        parameter_219,
        parameter_220,
        parameter_221,
        parameter_222,
        parameter_223,
        parameter_224,
        parameter_225,
        parameter_226,
        parameter_227,
        parameter_228,
        parameter_229,
        parameter_230,
        parameter_231,
        parameter_232,
        parameter_233,
        data_0,
    ):
        # pd_op.conv2d: (-1x32x112x112xf32) <- (-1x3x224x224xf32, 32x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_233, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_0, parameter_233

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_232, full_int_array_0)
        del parameter_232

        # pd_op.add: (-1x32x112x112xf32) <- (-1x32x112x112xf32, 1x32x1x1xf32)
        add_1 = paddle._C_ops.add(conv2d_0, reshape_0)
        del conv2d_0, reshape_0

        # pd_op.batch_norm_: (-1x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        (
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
            batch_norm__5,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_1,
                parameter_231,
                parameter_230,
                parameter_229,
                parameter_228,
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
        del add_1, parameter_228, parameter_229, parameter_230, parameter_231

        # pd_op.relu6: (-1x32x112x112xf32) <- (-1x32x112x112xf32)
        relu6_0 = paddle._C_ops.relu6(batch_norm__0)
        del batch_norm__0

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x32x112x112xf32, 32x32x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            relu6_0, parameter_227, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_227, relu6_0

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_226, full_int_array_0)
        del parameter_226

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        add_2 = paddle._C_ops.add(conv2d_1, reshape_1)
        del conv2d_1, reshape_1

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        (
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            batch_norm__10,
            batch_norm__11,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_2,
                parameter_225,
                parameter_224,
                parameter_223,
                parameter_222,
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
        del add_2, parameter_222, parameter_223, parameter_224, parameter_225

        # pd_op.depthwise_conv2d: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 32x1x7x7xf32)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(
            batch_norm__6, parameter_221, [1, 1], [3, 3], "EXPLICIT", 32, [1, 1], "NCHW"
        )
        del parameter_221

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(parameter_220, full_int_array_0)
        del parameter_220

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        add_3 = paddle._C_ops.add(depthwise_conv2d_0, reshape_2)
        del depthwise_conv2d_0, reshape_2

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        (
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            batch_norm__16,
            batch_norm__17,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_3,
                parameter_219,
                parameter_218,
                parameter_217,
                parameter_216,
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
        del add_3, parameter_216, parameter_217, parameter_218, parameter_219

        # pd_op.conv2d: (-1x128x56x56xf32) <- (-1x32x56x56xf32, 128x32x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            batch_norm__12, parameter_215, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_215

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(parameter_214, full_int_array_0)
        del parameter_214

        # pd_op.add: (-1x128x56x56xf32) <- (-1x128x56x56xf32, 1x128x1x1xf32)
        add_4 = paddle._C_ops.add(conv2d_2, reshape_3)
        del conv2d_2, reshape_3

        # pd_op.conv2d: (-1x128x56x56xf32) <- (-1x32x56x56xf32, 128x32x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            batch_norm__12, parameter_213, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del batch_norm__12, parameter_213

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(parameter_212, full_int_array_0)
        del parameter_212

        # pd_op.add: (-1x128x56x56xf32) <- (-1x128x56x56xf32, 1x128x1x1xf32)
        add_5 = paddle._C_ops.add(conv2d_3, reshape_4)
        del conv2d_3, reshape_4

        # pd_op.relu6: (-1x128x56x56xf32) <- (-1x128x56x56xf32)
        relu6_1 = paddle._C_ops.relu6(add_4)
        del add_4

        # pd_op.multiply: (-1x128x56x56xf32) <- (-1x128x56x56xf32, -1x128x56x56xf32)
        multiply_0 = paddle._C_ops.multiply(relu6_1, add_5)
        del add_5, relu6_1

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x128x56x56xf32, 32x128x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            multiply_0, parameter_211, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_0, parameter_211

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(parameter_210, full_int_array_0)
        del parameter_210

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        add_6 = paddle._C_ops.add(conv2d_4, reshape_5)
        del conv2d_4, reshape_5

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        (
            batch_norm__18,
            batch_norm__19,
            batch_norm__20,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_6,
                parameter_209,
                parameter_208,
                parameter_207,
                parameter_206,
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
        del add_6, parameter_206, parameter_207, parameter_208, parameter_209

        # pd_op.depthwise_conv2d: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 32x1x7x7xf32)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(
            batch_norm__18,
            parameter_205,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            32,
            [1, 1],
            "NCHW",
        )
        del batch_norm__18, parameter_205

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(parameter_204, full_int_array_0)
        del parameter_204

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        add_7 = paddle._C_ops.add(depthwise_conv2d_1, reshape_6)
        del depthwise_conv2d_1, reshape_6

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, -1x32x56x56xf32)
        add_8 = paddle._C_ops.add(batch_norm__6, add_7)
        del add_7, batch_norm__6

        # pd_op.conv2d: (-1x64x28x28xf32) <- (-1x32x56x56xf32, 64x32x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            add_8, parameter_203, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_8, parameter_203

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(parameter_202, full_int_array_0)
        del parameter_202

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        add_9 = paddle._C_ops.add(conv2d_5, reshape_7)
        del conv2d_5, reshape_7

        # pd_op.batch_norm_: (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_9,
                parameter_201,
                parameter_200,
                parameter_199,
                parameter_198,
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
        del add_9, parameter_198, parameter_199, parameter_200, parameter_201

        # pd_op.depthwise_conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x1x7x7xf32)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(
            batch_norm__24,
            parameter_197,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            64,
            [1, 1],
            "NCHW",
        )
        del parameter_197

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(parameter_196, full_int_array_0)
        del parameter_196

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        add_10 = paddle._C_ops.add(depthwise_conv2d_2, reshape_8)
        del depthwise_conv2d_2, reshape_8

        # pd_op.batch_norm_: (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_10,
                parameter_195,
                parameter_194,
                parameter_193,
                parameter_192,
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
        del add_10, parameter_192, parameter_193, parameter_194, parameter_195

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x64x28x28xf32, 256x64x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            batch_norm__30, parameter_191, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_191

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(parameter_190, full_int_array_0)
        del parameter_190

        # pd_op.add: (-1x256x28x28xf32) <- (-1x256x28x28xf32, 1x256x1x1xf32)
        add_11 = paddle._C_ops.add(conv2d_6, reshape_9)
        del conv2d_6, reshape_9

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x64x28x28xf32, 256x64x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            batch_norm__30, parameter_189, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del batch_norm__30, parameter_189

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(parameter_188, full_int_array_0)
        del parameter_188

        # pd_op.add: (-1x256x28x28xf32) <- (-1x256x28x28xf32, 1x256x1x1xf32)
        add_12 = paddle._C_ops.add(conv2d_7, reshape_10)
        del conv2d_7, reshape_10

        # pd_op.relu6: (-1x256x28x28xf32) <- (-1x256x28x28xf32)
        relu6_2 = paddle._C_ops.relu6(add_11)
        del add_11

        # pd_op.multiply: (-1x256x28x28xf32) <- (-1x256x28x28xf32, -1x256x28x28xf32)
        multiply_1 = paddle._C_ops.multiply(relu6_2, add_12)
        del add_12, relu6_2

        # pd_op.conv2d: (-1x64x28x28xf32) <- (-1x256x28x28xf32, 64x256x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            multiply_1, parameter_187, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_1, parameter_187

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_11 = paddle._C_ops.reshape(parameter_186, full_int_array_0)
        del parameter_186

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        add_13 = paddle._C_ops.add(conv2d_8, reshape_11)
        del conv2d_8, reshape_11

        # pd_op.batch_norm_: (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__36,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_13,
                parameter_185,
                parameter_184,
                parameter_183,
                parameter_182,
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
        del add_13, parameter_182, parameter_183, parameter_184, parameter_185

        # pd_op.depthwise_conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x1x7x7xf32)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(
            batch_norm__36,
            parameter_181,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            64,
            [1, 1],
            "NCHW",
        )
        del batch_norm__36, parameter_181

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(parameter_180, full_int_array_0)
        del parameter_180

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        add_14 = paddle._C_ops.add(depthwise_conv2d_3, reshape_12)
        del depthwise_conv2d_3, reshape_12

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, -1x64x28x28xf32)
        add_15 = paddle._C_ops.add(batch_norm__24, add_14)
        del add_14, batch_norm__24

        # pd_op.depthwise_conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x1x7x7xf32)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(
            add_15, parameter_179, [1, 1], [3, 3], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del parameter_179

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(parameter_178, full_int_array_0)
        del parameter_178

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        add_16 = paddle._C_ops.add(depthwise_conv2d_4, reshape_13)
        del depthwise_conv2d_4, reshape_13

        # pd_op.batch_norm_: (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__42,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_16,
                parameter_177,
                parameter_176,
                parameter_175,
                parameter_174,
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
        del add_16, parameter_174, parameter_175, parameter_176, parameter_177

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x64x28x28xf32, 256x64x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            batch_norm__42, parameter_173, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_173

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(parameter_172, full_int_array_0)
        del parameter_172

        # pd_op.add: (-1x256x28x28xf32) <- (-1x256x28x28xf32, 1x256x1x1xf32)
        add_17 = paddle._C_ops.add(conv2d_9, reshape_14)
        del conv2d_9, reshape_14

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x64x28x28xf32, 256x64x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            batch_norm__42, parameter_171, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del batch_norm__42, parameter_171

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_15 = paddle._C_ops.reshape(parameter_170, full_int_array_0)
        del parameter_170

        # pd_op.add: (-1x256x28x28xf32) <- (-1x256x28x28xf32, 1x256x1x1xf32)
        add_18 = paddle._C_ops.add(conv2d_10, reshape_15)
        del conv2d_10, reshape_15

        # pd_op.relu6: (-1x256x28x28xf32) <- (-1x256x28x28xf32)
        relu6_3 = paddle._C_ops.relu6(add_17)
        del add_17

        # pd_op.multiply: (-1x256x28x28xf32) <- (-1x256x28x28xf32, -1x256x28x28xf32)
        multiply_2 = paddle._C_ops.multiply(relu6_3, add_18)
        del add_18, relu6_3

        # pd_op.conv2d: (-1x64x28x28xf32) <- (-1x256x28x28xf32, 64x256x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            multiply_2, parameter_169, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_2, parameter_169

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_16 = paddle._C_ops.reshape(parameter_168, full_int_array_0)
        del parameter_168

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        add_19 = paddle._C_ops.add(conv2d_11, reshape_16)
        del conv2d_11, reshape_16

        # pd_op.batch_norm_: (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__48,
            batch_norm__49,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_19,
                parameter_167,
                parameter_166,
                parameter_165,
                parameter_164,
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
        del add_19, parameter_164, parameter_165, parameter_166, parameter_167

        # pd_op.depthwise_conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x1x7x7xf32)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(
            batch_norm__48,
            parameter_163,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            64,
            [1, 1],
            "NCHW",
        )
        del batch_norm__48, parameter_163

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_17 = paddle._C_ops.reshape(parameter_162, full_int_array_0)
        del parameter_162

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        add_20 = paddle._C_ops.add(depthwise_conv2d_5, reshape_17)
        del depthwise_conv2d_5, reshape_17

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, -1x64x28x28xf32)
        add_21 = paddle._C_ops.add(add_15, add_20)
        del add_15, add_20

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x64x28x28xf32, 128x64x3x3xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            add_21, parameter_161, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_21, parameter_161

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_18 = paddle._C_ops.reshape(parameter_160, full_int_array_0)
        del parameter_160

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_22 = paddle._C_ops.add(conv2d_12, reshape_18)
        del conv2d_12, reshape_18

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_22,
                parameter_159,
                parameter_158,
                parameter_157,
                parameter_156,
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
        del add_22, parameter_156, parameter_157, parameter_158, parameter_159

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(
            batch_norm__54,
            parameter_155,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            128,
            [1, 1],
            "NCHW",
        )
        del parameter_155

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_19 = paddle._C_ops.reshape(parameter_154, full_int_array_0)
        del parameter_154

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_23 = paddle._C_ops.add(depthwise_conv2d_6, reshape_19)
        del depthwise_conv2d_6, reshape_19

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_23,
                parameter_153,
                parameter_152,
                parameter_151,
                parameter_150,
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
        del add_23, parameter_150, parameter_151, parameter_152, parameter_153

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            batch_norm__60, parameter_149, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_149

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_20 = paddle._C_ops.reshape(parameter_148, full_int_array_0)
        del parameter_148

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        add_24 = paddle._C_ops.add(conv2d_13, reshape_20)
        del conv2d_13, reshape_20

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            batch_norm__60, parameter_147, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del batch_norm__60, parameter_147

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_21 = paddle._C_ops.reshape(parameter_146, full_int_array_0)
        del parameter_146

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        add_25 = paddle._C_ops.add(conv2d_14, reshape_21)
        del conv2d_14, reshape_21

        # pd_op.relu6: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu6_4 = paddle._C_ops.relu6(add_24)
        del add_24

        # pd_op.multiply: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        multiply_3 = paddle._C_ops.multiply(relu6_4, add_25)
        del add_25, relu6_4

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x512x14x14xf32, 128x512x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            multiply_3, parameter_145, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_3, parameter_145

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_22 = paddle._C_ops.reshape(parameter_144, full_int_array_0)
        del parameter_144

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_26 = paddle._C_ops.add(conv2d_15, reshape_22)
        del conv2d_15, reshape_22

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__66,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            batch_norm__70,
            batch_norm__71,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_26,
                parameter_143,
                parameter_142,
                parameter_141,
                parameter_140,
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
        del add_26, parameter_140, parameter_141, parameter_142, parameter_143

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(
            batch_norm__66,
            parameter_139,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            128,
            [1, 1],
            "NCHW",
        )
        del batch_norm__66, parameter_139

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_23 = paddle._C_ops.reshape(parameter_138, full_int_array_0)
        del parameter_138

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_27 = paddle._C_ops.add(depthwise_conv2d_7, reshape_23)
        del depthwise_conv2d_7, reshape_23

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x14x14xf32)
        add_28 = paddle._C_ops.add(batch_norm__54, add_27)
        del add_27, batch_norm__54

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        depthwise_conv2d_8 = paddle._C_ops.depthwise_conv2d(
            add_28, parameter_137, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del parameter_137

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_24 = paddle._C_ops.reshape(parameter_136, full_int_array_0)
        del parameter_136

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_29 = paddle._C_ops.add(depthwise_conv2d_8, reshape_24)
        del depthwise_conv2d_8, reshape_24

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__72,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_29,
                parameter_135,
                parameter_134,
                parameter_133,
                parameter_132,
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
        del add_29, parameter_132, parameter_133, parameter_134, parameter_135

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            batch_norm__72, parameter_131, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_131

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_25 = paddle._C_ops.reshape(parameter_130, full_int_array_0)
        del parameter_130

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        add_30 = paddle._C_ops.add(conv2d_16, reshape_25)
        del conv2d_16, reshape_25

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            batch_norm__72, parameter_129, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del batch_norm__72, parameter_129

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_26 = paddle._C_ops.reshape(parameter_128, full_int_array_0)
        del parameter_128

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        add_31 = paddle._C_ops.add(conv2d_17, reshape_26)
        del conv2d_17, reshape_26

        # pd_op.relu6: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu6_5 = paddle._C_ops.relu6(add_30)
        del add_30

        # pd_op.multiply: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        multiply_4 = paddle._C_ops.multiply(relu6_5, add_31)
        del add_31, relu6_5

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x512x14x14xf32, 128x512x1x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            multiply_4, parameter_127, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_4, parameter_127

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_27 = paddle._C_ops.reshape(parameter_126, full_int_array_0)
        del parameter_126

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_32 = paddle._C_ops.add(conv2d_18, reshape_27)
        del conv2d_18, reshape_27

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__78,
            batch_norm__79,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
            batch_norm__83,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_32,
                parameter_125,
                parameter_124,
                parameter_123,
                parameter_122,
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
        del add_32, parameter_122, parameter_123, parameter_124, parameter_125

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        depthwise_conv2d_9 = paddle._C_ops.depthwise_conv2d(
            batch_norm__78,
            parameter_121,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            128,
            [1, 1],
            "NCHW",
        )
        del batch_norm__78, parameter_121

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_28 = paddle._C_ops.reshape(parameter_120, full_int_array_0)
        del parameter_120

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_33 = paddle._C_ops.add(depthwise_conv2d_9, reshape_28)
        del depthwise_conv2d_9, reshape_28

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x14x14xf32)
        add_34 = paddle._C_ops.add(add_28, add_33)
        del add_28, add_33

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        depthwise_conv2d_10 = paddle._C_ops.depthwise_conv2d(
            add_34, parameter_119, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del parameter_119

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_29 = paddle._C_ops.reshape(parameter_118, full_int_array_0)
        del parameter_118

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_35 = paddle._C_ops.add(depthwise_conv2d_10, reshape_29)
        del depthwise_conv2d_10, reshape_29

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__84,
            batch_norm__85,
            batch_norm__86,
            batch_norm__87,
            batch_norm__88,
            batch_norm__89,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_35,
                parameter_117,
                parameter_116,
                parameter_115,
                parameter_114,
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
        del add_35, parameter_114, parameter_115, parameter_116, parameter_117

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            batch_norm__84, parameter_113, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_113

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_30 = paddle._C_ops.reshape(parameter_112, full_int_array_0)
        del parameter_112

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        add_36 = paddle._C_ops.add(conv2d_19, reshape_30)
        del conv2d_19, reshape_30

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            batch_norm__84, parameter_111, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del batch_norm__84, parameter_111

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_31 = paddle._C_ops.reshape(parameter_110, full_int_array_0)
        del parameter_110

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        add_37 = paddle._C_ops.add(conv2d_20, reshape_31)
        del conv2d_20, reshape_31

        # pd_op.relu6: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu6_6 = paddle._C_ops.relu6(add_36)
        del add_36

        # pd_op.multiply: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        multiply_5 = paddle._C_ops.multiply(relu6_6, add_37)
        del add_37, relu6_6

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x512x14x14xf32, 128x512x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            multiply_5, parameter_109, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_5, parameter_109

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_32 = paddle._C_ops.reshape(parameter_108, full_int_array_0)
        del parameter_108

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_38 = paddle._C_ops.add(conv2d_21, reshape_32)
        del conv2d_21, reshape_32

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__90,
            batch_norm__91,
            batch_norm__92,
            batch_norm__93,
            batch_norm__94,
            batch_norm__95,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_38,
                parameter_107,
                parameter_106,
                parameter_105,
                parameter_104,
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
        del add_38, parameter_104, parameter_105, parameter_106, parameter_107

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        depthwise_conv2d_11 = paddle._C_ops.depthwise_conv2d(
            batch_norm__90,
            parameter_103,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            128,
            [1, 1],
            "NCHW",
        )
        del batch_norm__90, parameter_103

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_33 = paddle._C_ops.reshape(parameter_102, full_int_array_0)
        del parameter_102

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_39 = paddle._C_ops.add(depthwise_conv2d_11, reshape_33)
        del depthwise_conv2d_11, reshape_33

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x14x14xf32)
        add_40 = paddle._C_ops.add(add_34, add_39)
        del add_34, add_39

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        depthwise_conv2d_12 = paddle._C_ops.depthwise_conv2d(
            add_40, parameter_101, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del parameter_101

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_34 = paddle._C_ops.reshape(parameter_100, full_int_array_0)
        del parameter_100

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_41 = paddle._C_ops.add(depthwise_conv2d_12, reshape_34)
        del depthwise_conv2d_12, reshape_34

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__96,
            batch_norm__97,
            batch_norm__98,
            batch_norm__99,
            batch_norm__100,
            batch_norm__101,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_41,
                parameter_99,
                parameter_98,
                parameter_97,
                parameter_96,
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
        del add_41, parameter_96, parameter_97, parameter_98, parameter_99

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(
            batch_norm__96, parameter_95, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_95

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_35 = paddle._C_ops.reshape(parameter_94, full_int_array_0)
        del parameter_94

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        add_42 = paddle._C_ops.add(conv2d_22, reshape_35)
        del conv2d_22, reshape_35

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(
            batch_norm__96, parameter_93, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del batch_norm__96, parameter_93

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_36 = paddle._C_ops.reshape(parameter_92, full_int_array_0)
        del parameter_92

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        add_43 = paddle._C_ops.add(conv2d_23, reshape_36)
        del conv2d_23, reshape_36

        # pd_op.relu6: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu6_7 = paddle._C_ops.relu6(add_42)
        del add_42

        # pd_op.multiply: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        multiply_6 = paddle._C_ops.multiply(relu6_7, add_43)
        del add_43, relu6_7

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x512x14x14xf32, 128x512x1x1xf32)
        conv2d_24 = paddle._C_ops.conv2d(
            multiply_6, parameter_91, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_6, parameter_91

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_37 = paddle._C_ops.reshape(parameter_90, full_int_array_0)
        del parameter_90

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_44 = paddle._C_ops.add(conv2d_24, reshape_37)
        del conv2d_24, reshape_37

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__102,
            batch_norm__103,
            batch_norm__104,
            batch_norm__105,
            batch_norm__106,
            batch_norm__107,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_44,
                parameter_89,
                parameter_88,
                parameter_87,
                parameter_86,
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
        del add_44, parameter_86, parameter_87, parameter_88, parameter_89

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        depthwise_conv2d_13 = paddle._C_ops.depthwise_conv2d(
            batch_norm__102,
            parameter_85,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            128,
            [1, 1],
            "NCHW",
        )
        del batch_norm__102, parameter_85

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_38 = paddle._C_ops.reshape(parameter_84, full_int_array_0)
        del parameter_84

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_45 = paddle._C_ops.add(depthwise_conv2d_13, reshape_38)
        del depthwise_conv2d_13, reshape_38

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x14x14xf32)
        add_46 = paddle._C_ops.add(add_40, add_45)
        del add_40, add_45

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        depthwise_conv2d_14 = paddle._C_ops.depthwise_conv2d(
            add_46, parameter_83, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del parameter_83

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_39 = paddle._C_ops.reshape(parameter_82, full_int_array_0)
        del parameter_82

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_47 = paddle._C_ops.add(depthwise_conv2d_14, reshape_39)
        del depthwise_conv2d_14, reshape_39

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__108,
            batch_norm__109,
            batch_norm__110,
            batch_norm__111,
            batch_norm__112,
            batch_norm__113,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_47,
                parameter_81,
                parameter_80,
                parameter_79,
                parameter_78,
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
        del add_47, parameter_78, parameter_79, parameter_80, parameter_81

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(
            batch_norm__108, parameter_77, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_77

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_40 = paddle._C_ops.reshape(parameter_76, full_int_array_0)
        del parameter_76

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        add_48 = paddle._C_ops.add(conv2d_25, reshape_40)
        del conv2d_25, reshape_40

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(
            batch_norm__108, parameter_75, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del batch_norm__108, parameter_75

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_41 = paddle._C_ops.reshape(parameter_74, full_int_array_0)
        del parameter_74

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        add_49 = paddle._C_ops.add(conv2d_26, reshape_41)
        del conv2d_26, reshape_41

        # pd_op.relu6: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu6_8 = paddle._C_ops.relu6(add_48)
        del add_48

        # pd_op.multiply: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        multiply_7 = paddle._C_ops.multiply(relu6_8, add_49)
        del add_49, relu6_8

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x512x14x14xf32, 128x512x1x1xf32)
        conv2d_27 = paddle._C_ops.conv2d(
            multiply_7, parameter_73, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_7, parameter_73

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_42 = paddle._C_ops.reshape(parameter_72, full_int_array_0)
        del parameter_72

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_50 = paddle._C_ops.add(conv2d_27, reshape_42)
        del conv2d_27, reshape_42

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__114,
            batch_norm__115,
            batch_norm__116,
            batch_norm__117,
            batch_norm__118,
            batch_norm__119,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_50,
                parameter_71,
                parameter_70,
                parameter_69,
                parameter_68,
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
        del add_50, parameter_68, parameter_69, parameter_70, parameter_71

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        depthwise_conv2d_15 = paddle._C_ops.depthwise_conv2d(
            batch_norm__114,
            parameter_67,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            128,
            [1, 1],
            "NCHW",
        )
        del batch_norm__114, parameter_67

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_43 = paddle._C_ops.reshape(parameter_66, full_int_array_0)
        del parameter_66

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_51 = paddle._C_ops.add(depthwise_conv2d_15, reshape_43)
        del depthwise_conv2d_15, reshape_43

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x14x14xf32)
        add_52 = paddle._C_ops.add(add_46, add_51)
        del add_46, add_51

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        depthwise_conv2d_16 = paddle._C_ops.depthwise_conv2d(
            add_52, parameter_65, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del parameter_65

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_44 = paddle._C_ops.reshape(parameter_64, full_int_array_0)
        del parameter_64

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_53 = paddle._C_ops.add(depthwise_conv2d_16, reshape_44)
        del depthwise_conv2d_16, reshape_44

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__120,
            batch_norm__121,
            batch_norm__122,
            batch_norm__123,
            batch_norm__124,
            batch_norm__125,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_53,
                parameter_63,
                parameter_62,
                parameter_61,
                parameter_60,
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
        del add_53, parameter_60, parameter_61, parameter_62, parameter_63

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(
            batch_norm__120, parameter_59, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_59

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_45 = paddle._C_ops.reshape(parameter_58, full_int_array_0)
        del parameter_58

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        add_54 = paddle._C_ops.add(conv2d_28, reshape_45)
        del conv2d_28, reshape_45

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(
            batch_norm__120, parameter_57, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del batch_norm__120, parameter_57

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_46 = paddle._C_ops.reshape(parameter_56, full_int_array_0)
        del parameter_56

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        add_55 = paddle._C_ops.add(conv2d_29, reshape_46)
        del conv2d_29, reshape_46

        # pd_op.relu6: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu6_9 = paddle._C_ops.relu6(add_54)
        del add_54

        # pd_op.multiply: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        multiply_8 = paddle._C_ops.multiply(relu6_9, add_55)
        del add_55, relu6_9

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x512x14x14xf32, 128x512x1x1xf32)
        conv2d_30 = paddle._C_ops.conv2d(
            multiply_8, parameter_55, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_8, parameter_55

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_47 = paddle._C_ops.reshape(parameter_54, full_int_array_0)
        del parameter_54

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_56 = paddle._C_ops.add(conv2d_30, reshape_47)
        del conv2d_30, reshape_47

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__126,
            batch_norm__127,
            batch_norm__128,
            batch_norm__129,
            batch_norm__130,
            batch_norm__131,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_56,
                parameter_53,
                parameter_52,
                parameter_51,
                parameter_50,
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
        del add_56, parameter_50, parameter_51, parameter_52, parameter_53

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        depthwise_conv2d_17 = paddle._C_ops.depthwise_conv2d(
            batch_norm__126,
            parameter_49,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            128,
            [1, 1],
            "NCHW",
        )
        del batch_norm__126, parameter_49

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_48 = paddle._C_ops.reshape(parameter_48, full_int_array_0)
        del parameter_48

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_57 = paddle._C_ops.add(depthwise_conv2d_17, reshape_48)
        del depthwise_conv2d_17, reshape_48

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x14x14xf32)
        add_58 = paddle._C_ops.add(add_52, add_57)
        del add_52, add_57

        # pd_op.conv2d: (-1x256x7x7xf32) <- (-1x128x14x14xf32, 256x128x3x3xf32)
        conv2d_31 = paddle._C_ops.conv2d(
            add_58, parameter_47, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_58, parameter_47

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_49 = paddle._C_ops.reshape(parameter_46, full_int_array_0)
        del parameter_46

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        add_59 = paddle._C_ops.add(conv2d_31, reshape_49)
        del conv2d_31, reshape_49

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__132,
            batch_norm__133,
            batch_norm__134,
            batch_norm__135,
            batch_norm__136,
            batch_norm__137,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_59,
                parameter_45,
                parameter_44,
                parameter_43,
                parameter_42,
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
        del add_59, parameter_42, parameter_43, parameter_44, parameter_45

        # pd_op.depthwise_conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x1x7x7xf32)
        depthwise_conv2d_18 = paddle._C_ops.depthwise_conv2d(
            batch_norm__132,
            parameter_41,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            256,
            [1, 1],
            "NCHW",
        )
        del parameter_41

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_50 = paddle._C_ops.reshape(parameter_40, full_int_array_0)
        del parameter_40

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        add_60 = paddle._C_ops.add(depthwise_conv2d_18, reshape_50)
        del depthwise_conv2d_18, reshape_50

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__138,
            batch_norm__139,
            batch_norm__140,
            batch_norm__141,
            batch_norm__142,
            batch_norm__143,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_60,
                parameter_39,
                parameter_38,
                parameter_37,
                parameter_36,
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
        del add_60, parameter_36, parameter_37, parameter_38, parameter_39

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x256x7x7xf32, 1024x256x1x1xf32)
        conv2d_32 = paddle._C_ops.conv2d(
            batch_norm__138, parameter_35, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_35

        # pd_op.reshape: (1x1024x1x1xf32) <- (1024xf32, 4xi64)
        reshape_51 = paddle._C_ops.reshape(parameter_34, full_int_array_0)
        del parameter_34

        # pd_op.add: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1x1024x1x1xf32)
        add_61 = paddle._C_ops.add(conv2d_32, reshape_51)
        del conv2d_32, reshape_51

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x256x7x7xf32, 1024x256x1x1xf32)
        conv2d_33 = paddle._C_ops.conv2d(
            batch_norm__138, parameter_33, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del batch_norm__138, parameter_33

        # pd_op.reshape: (1x1024x1x1xf32) <- (1024xf32, 4xi64)
        reshape_52 = paddle._C_ops.reshape(parameter_32, full_int_array_0)
        del parameter_32

        # pd_op.add: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1x1024x1x1xf32)
        add_62 = paddle._C_ops.add(conv2d_33, reshape_52)
        del conv2d_33, reshape_52

        # pd_op.relu6: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32)
        relu6_10 = paddle._C_ops.relu6(add_61)
        del add_61

        # pd_op.multiply: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, -1x1024x7x7xf32)
        multiply_9 = paddle._C_ops.multiply(relu6_10, add_62)
        del add_62, relu6_10

        # pd_op.conv2d: (-1x256x7x7xf32) <- (-1x1024x7x7xf32, 256x1024x1x1xf32)
        conv2d_34 = paddle._C_ops.conv2d(
            multiply_9, parameter_31, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_9, parameter_31

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_53 = paddle._C_ops.reshape(parameter_30, full_int_array_0)
        del parameter_30

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        add_63 = paddle._C_ops.add(conv2d_34, reshape_53)
        del conv2d_34, reshape_53

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__144,
            batch_norm__145,
            batch_norm__146,
            batch_norm__147,
            batch_norm__148,
            batch_norm__149,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_63,
                parameter_29,
                parameter_28,
                parameter_27,
                parameter_26,
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
        del add_63, parameter_26, parameter_27, parameter_28, parameter_29

        # pd_op.depthwise_conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x1x7x7xf32)
        depthwise_conv2d_19 = paddle._C_ops.depthwise_conv2d(
            batch_norm__144,
            parameter_25,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            256,
            [1, 1],
            "NCHW",
        )
        del batch_norm__144, parameter_25

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_54 = paddle._C_ops.reshape(parameter_24, full_int_array_0)
        del parameter_24

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        add_64 = paddle._C_ops.add(depthwise_conv2d_19, reshape_54)
        del depthwise_conv2d_19, reshape_54

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x256x7x7xf32)
        add_65 = paddle._C_ops.add(batch_norm__132, add_64)
        del add_64, batch_norm__132

        # pd_op.depthwise_conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x1x7x7xf32)
        depthwise_conv2d_20 = paddle._C_ops.depthwise_conv2d(
            add_65, parameter_23, [1, 1], [3, 3], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del parameter_23

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_55 = paddle._C_ops.reshape(parameter_22, full_int_array_0)
        del parameter_22

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        add_66 = paddle._C_ops.add(depthwise_conv2d_20, reshape_55)
        del depthwise_conv2d_20, reshape_55

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__150,
            batch_norm__151,
            batch_norm__152,
            batch_norm__153,
            batch_norm__154,
            batch_norm__155,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_66,
                parameter_21,
                parameter_20,
                parameter_19,
                parameter_18,
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
        del add_66, parameter_18, parameter_19, parameter_20, parameter_21

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x256x7x7xf32, 1024x256x1x1xf32)
        conv2d_35 = paddle._C_ops.conv2d(
            batch_norm__150, parameter_17, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_17

        # pd_op.reshape: (1x1024x1x1xf32) <- (1024xf32, 4xi64)
        reshape_56 = paddle._C_ops.reshape(parameter_16, full_int_array_0)
        del parameter_16

        # pd_op.add: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1x1024x1x1xf32)
        add_67 = paddle._C_ops.add(conv2d_35, reshape_56)
        del conv2d_35, reshape_56

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x256x7x7xf32, 1024x256x1x1xf32)
        conv2d_36 = paddle._C_ops.conv2d(
            batch_norm__150, parameter_15, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del batch_norm__150, parameter_15

        # pd_op.reshape: (1x1024x1x1xf32) <- (1024xf32, 4xi64)
        reshape_57 = paddle._C_ops.reshape(parameter_14, full_int_array_0)
        del parameter_14

        # pd_op.add: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1x1024x1x1xf32)
        add_68 = paddle._C_ops.add(conv2d_36, reshape_57)
        del conv2d_36, reshape_57

        # pd_op.relu6: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32)
        relu6_11 = paddle._C_ops.relu6(add_67)
        del add_67

        # pd_op.multiply: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, -1x1024x7x7xf32)
        multiply_10 = paddle._C_ops.multiply(relu6_11, add_68)
        del add_68, relu6_11

        # pd_op.conv2d: (-1x256x7x7xf32) <- (-1x1024x7x7xf32, 256x1024x1x1xf32)
        conv2d_37 = paddle._C_ops.conv2d(
            multiply_10, parameter_13, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_10, parameter_13

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_58 = paddle._C_ops.reshape(parameter_12, full_int_array_0)
        del parameter_12

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        add_69 = paddle._C_ops.add(conv2d_37, reshape_58)
        del conv2d_37, reshape_58

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__156,
            batch_norm__157,
            batch_norm__158,
            batch_norm__159,
            batch_norm__160,
            batch_norm__161,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_69,
                parameter_11,
                parameter_10,
                parameter_9,
                parameter_8,
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
        del add_69, parameter_10, parameter_11, parameter_8, parameter_9

        # pd_op.depthwise_conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x1x7x7xf32)
        depthwise_conv2d_21 = paddle._C_ops.depthwise_conv2d(
            batch_norm__156,
            parameter_7,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            256,
            [1, 1],
            "NCHW",
        )
        del batch_norm__156, parameter_7

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_59 = paddle._C_ops.reshape(parameter_6, full_int_array_0)
        del full_int_array_0, parameter_6

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        add_70 = paddle._C_ops.add(depthwise_conv2d_21, reshape_59)
        del depthwise_conv2d_21, reshape_59

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x256x7x7xf32)
        add_71 = paddle._C_ops.add(add_65, add_70)
        del add_65, add_70

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__162,
            batch_norm__163,
            batch_norm__164,
            batch_norm__165,
            batch_norm__166,
            batch_norm__167,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_71,
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
        del add_71, parameter_2, parameter_3, parameter_4, parameter_5

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf32) <- (-1x256x7x7xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            batch_norm__162,
            full_int_array_1,
            [1, 1],
            [0, 0],
            False,
            True,
            "NCHW",
            "avg",
            False,
            True,
            "EXPLICIT",
        )
        del batch_norm__162, full_int_array_1

        # pd_op.flatten: (-1x256xf32) <- (-1x256x1x1xf32)
        flatten_0 = paddle._C_ops.flatten(pool2d_0, 1, 3)
        del pool2d_0

        # pd_op.matmul: (-1x102xf32) <- (-1x256xf32, 256x102xf32)
        matmul_0 = paddle._C_ops.matmul(flatten_0, parameter_1, False, False)
        del flatten_0, parameter_1

        # pd_op.add: (-1x102xf32) <- (-1x102xf32, 102xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_0)
        del matmul_0, parameter_0

        return add_0
