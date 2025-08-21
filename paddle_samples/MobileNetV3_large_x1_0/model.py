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
        parameter_234,
        parameter_235,
        parameter_236,
        parameter_237,
        parameter_238,
        parameter_239,
        parameter_240,
        parameter_241,
        parameter_242,
        parameter_243,
        parameter_244,
        parameter_245,
        parameter_246,
        parameter_247,
        parameter_248,
        parameter_249,
        parameter_250,
        parameter_251,
        parameter_252,
        parameter_253,
        parameter_254,
        parameter_255,
        parameter_256,
        parameter_257,
        parameter_258,
        parameter_259,
        parameter_260,
        parameter_261,
        parameter_262,
        parameter_263,
        parameter_264,
        parameter_265,
        parameter_266,
        parameter_267,
        parameter_268,
        parameter_269,
        data_0,
    ):
        # pd_op.conv2d: (128x16x112x112xf32) <- (128x3x224x224xf32, 16x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_269, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x16x112x112xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (128x16x112x112xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        (
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
            batch_norm__5,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_0,
                parameter_268,
                parameter_267,
                parameter_266,
                parameter_265,
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

        # pd_op.hardswish: (128x16x112x112xf32) <- (128x16x112x112xf32)
        hardswish_0 = paddle._C_ops.hardswish(batch_norm__0)

        # pd_op.conv2d: (128x16x112x112xf32) <- (128x16x112x112xf32, 16x16x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            hardswish_0, parameter_264, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x16x112x112xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (128x16x112x112xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        (
            batch_norm__270,
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            batch_norm__10,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_1,
                parameter_263,
                parameter_262,
                parameter_261,
                parameter_260,
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

        # pd_op.relu: (128x16x112x112xf32) <- (128x16x112x112xf32)
        relu_0 = paddle._C_ops.relu(batch_norm__270)

        # pd_op.depthwise_conv2d: (128x16x112x112xf32) <- (128x16x112x112xf32, 16x1x3x3xf32)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(
            relu_0, parameter_259, [1, 1], [1, 1], "EXPLICIT", 16, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (128x16x112x112xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (128x16x112x112xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        (
            batch_norm__271,
            batch_norm__11,
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_0,
                parameter_258,
                parameter_257,
                parameter_256,
                parameter_255,
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

        # pd_op.relu: (128x16x112x112xf32) <- (128x16x112x112xf32)
        relu_1 = paddle._C_ops.relu(batch_norm__271)

        # pd_op.conv2d: (128x16x112x112xf32) <- (128x16x112x112xf32, 16x16x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            relu_1, parameter_254, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x16x112x112xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (128x16x112x112xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        (
            batch_norm__16,
            batch_norm__17,
            batch_norm__18,
            batch_norm__19,
            batch_norm__20,
            batch_norm__21,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_2,
                parameter_253,
                parameter_252,
                parameter_251,
                parameter_250,
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

        # pd_op.add: (128x16x112x112xf32) <- (128x16x112x112xf32, 128x16x112x112xf32)
        add_0 = paddle._C_ops.add(hardswish_0, batch_norm__16)

        # pd_op.conv2d: (128x64x112x112xf32) <- (128x16x112x112xf32, 64x16x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            add_0, parameter_249, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x64x112x112xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (128x64x112x112xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__272,
            batch_norm__22,
            batch_norm__23,
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_3,
                parameter_248,
                parameter_247,
                parameter_246,
                parameter_245,
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

        # pd_op.relu: (128x64x112x112xf32) <- (128x64x112x112xf32)
        relu_2 = paddle._C_ops.relu(batch_norm__272)

        # pd_op.depthwise_conv2d: (128x64x56x56xf32) <- (128x64x112x112xf32, 64x1x3x3xf32)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(
            relu_2, parameter_244, [2, 2], [1, 1], "EXPLICIT", 64, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (128x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (128x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__273,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
            batch_norm__30,
            batch_norm__31,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_1,
                parameter_243,
                parameter_242,
                parameter_241,
                parameter_240,
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

        # pd_op.relu: (128x64x56x56xf32) <- (128x64x56x56xf32)
        relu_3 = paddle._C_ops.relu(batch_norm__273)

        # pd_op.conv2d: (128x24x56x56xf32) <- (128x64x56x56xf32, 24x64x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            relu_3, parameter_239, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (128x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        (
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
            batch_norm__36,
            batch_norm__37,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_4,
                parameter_238,
                parameter_237,
                parameter_236,
                parameter_235,
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

        # pd_op.conv2d: (128x72x56x56xf32) <- (128x24x56x56xf32, 72x24x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            batch_norm__32, parameter_234, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x72x56x56xf32, 72xf32, 72xf32, 72xf32, 72xf32, -1xui8) <- (128x72x56x56xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        (
            batch_norm__274,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
            batch_norm__42,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_5,
                parameter_233,
                parameter_232,
                parameter_231,
                parameter_230,
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

        # pd_op.relu: (128x72x56x56xf32) <- (128x72x56x56xf32)
        relu_4 = paddle._C_ops.relu(batch_norm__274)

        # pd_op.depthwise_conv2d: (128x72x56x56xf32) <- (128x72x56x56xf32, 72x1x3x3xf32)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(
            relu_4, parameter_229, [1, 1], [1, 1], "EXPLICIT", 72, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (128x72x56x56xf32, 72xf32, 72xf32, 72xf32, 72xf32, -1xui8) <- (128x72x56x56xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        (
            batch_norm__275,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_2,
                parameter_228,
                parameter_227,
                parameter_226,
                parameter_225,
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

        # pd_op.relu: (128x72x56x56xf32) <- (128x72x56x56xf32)
        relu_5 = paddle._C_ops.relu(batch_norm__275)

        # pd_op.conv2d: (128x24x56x56xf32) <- (128x72x56x56xf32, 24x72x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            relu_5, parameter_224, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (128x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        (
            batch_norm__48,
            batch_norm__49,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_6,
                parameter_223,
                parameter_222,
                parameter_221,
                parameter_220,
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

        # pd_op.add: (128x24x56x56xf32) <- (128x24x56x56xf32, 128x24x56x56xf32)
        add_1 = paddle._C_ops.add(batch_norm__32, batch_norm__48)

        # pd_op.conv2d: (128x72x56x56xf32) <- (128x24x56x56xf32, 72x24x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            add_1, parameter_219, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x72x56x56xf32, 72xf32, 72xf32, 72xf32, 72xf32, -1xui8) <- (128x72x56x56xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        (
            batch_norm__276,
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_7,
                parameter_218,
                parameter_217,
                parameter_216,
                parameter_215,
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

        # pd_op.relu: (128x72x56x56xf32) <- (128x72x56x56xf32)
        relu_6 = paddle._C_ops.relu(batch_norm__276)

        # pd_op.depthwise_conv2d: (128x72x28x28xf32) <- (128x72x56x56xf32, 72x1x5x5xf32)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(
            relu_6, parameter_214, [2, 2], [2, 2], "EXPLICIT", 72, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (128x72x28x28xf32, 72xf32, 72xf32, 72xf32, 72xf32, -1xui8) <- (128x72x28x28xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        (
            batch_norm__277,
            batch_norm__59,
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_3,
                parameter_213,
                parameter_212,
                parameter_211,
                parameter_210,
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

        # pd_op.relu: (128x72x28x28xf32) <- (128x72x28x28xf32)
        relu_7 = paddle._C_ops.relu(batch_norm__277)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_7 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_6 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_5 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_4 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_3 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_2 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_1 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_0 = full_int_array_0

        # pd_op.pool2d: (128x72x1x1xf32) <- (128x72x28x28xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            relu_7,
            full_int_array_0,
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

        # pd_op.conv2d: (128x18x1x1xf32) <- (128x72x1x1xf32, 18x72x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            pool2d_0, parameter_209, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [1, -1, 1, 1]

        # pd_op.reshape: (1x18x1x1xf32) <- (18xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_208, full_int_array_1)

        # pd_op.add: (128x18x1x1xf32) <- (128x18x1x1xf32, 1x18x1x1xf32)
        add_11 = paddle._C_ops.add(conv2d_8, reshape_0)

        # pd_op.relu: (128x18x1x1xf32) <- (128x18x1x1xf32)
        relu_8 = paddle._C_ops.relu(add_11)

        # pd_op.conv2d: (128x72x1x1xf32) <- (128x18x1x1xf32, 72x18x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            relu_8, parameter_207, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x72x1x1xf32) <- (72xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_206, full_int_array_1)

        # pd_op.add: (128x72x1x1xf32) <- (128x72x1x1xf32, 1x72x1x1xf32)
        add_12 = paddle._C_ops.add(conv2d_9, reshape_1)

        # pd_op.hardsigmoid: (128x72x1x1xf32) <- (128x72x1x1xf32)
        hardsigmoid_0 = paddle._C_ops.hardsigmoid(add_12, float("0.2"), float("0.5"))

        # pd_op.multiply: (128x72x28x28xf32) <- (128x72x28x28xf32, 128x72x1x1xf32)
        multiply_0 = paddle._C_ops.multiply(relu_7, hardsigmoid_0)

        # pd_op.conv2d: (128x40x28x28xf32) <- (128x72x28x28xf32, 40x72x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            multiply_0, parameter_205, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x40x28x28xf32, 40xf32, 40xf32, 40xf32, 40xf32, -1xui8) <- (128x40x28x28xf32, 40xf32, 40xf32, 40xf32, 40xf32)
        (
            batch_norm__64,
            batch_norm__65,
            batch_norm__66,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_10,
                parameter_204,
                parameter_203,
                parameter_202,
                parameter_201,
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

        # pd_op.conv2d: (128x120x28x28xf32) <- (128x40x28x28xf32, 120x40x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            batch_norm__64, parameter_200, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x120x28x28xf32, 120xf32, 120xf32, 120xf32, 120xf32, -1xui8) <- (128x120x28x28xf32, 120xf32, 120xf32, 120xf32, 120xf32)
        (
            batch_norm__278,
            batch_norm__70,
            batch_norm__71,
            batch_norm__72,
            batch_norm__73,
            batch_norm__74,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_11,
                parameter_199,
                parameter_198,
                parameter_197,
                parameter_196,
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

        # pd_op.relu: (128x120x28x28xf32) <- (128x120x28x28xf32)
        relu_9 = paddle._C_ops.relu(batch_norm__278)

        # pd_op.depthwise_conv2d: (128x120x28x28xf32) <- (128x120x28x28xf32, 120x1x5x5xf32)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(
            relu_9, parameter_195, [1, 1], [2, 2], "EXPLICIT", 120, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (128x120x28x28xf32, 120xf32, 120xf32, 120xf32, 120xf32, -1xui8) <- (128x120x28x28xf32, 120xf32, 120xf32, 120xf32, 120xf32)
        (
            batch_norm__279,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
            batch_norm__78,
            batch_norm__79,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_4,
                parameter_194,
                parameter_193,
                parameter_192,
                parameter_191,
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

        # pd_op.relu: (128x120x28x28xf32) <- (128x120x28x28xf32)
        relu_10 = paddle._C_ops.relu(batch_norm__279)

        # pd_op.pool2d: (128x120x1x1xf32) <- (128x120x28x28xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(
            relu_10,
            full_int_array_0,
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

        # pd_op.conv2d: (128x30x1x1xf32) <- (128x120x1x1xf32, 30x120x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            pool2d_1, parameter_190, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x30x1x1xf32) <- (30xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(parameter_189, full_int_array_1)

        # pd_op.add: (128x30x1x1xf32) <- (128x30x1x1xf32, 1x30x1x1xf32)
        add_13 = paddle._C_ops.add(conv2d_12, reshape_2)

        # pd_op.relu: (128x30x1x1xf32) <- (128x30x1x1xf32)
        relu_11 = paddle._C_ops.relu(add_13)

        # pd_op.conv2d: (128x120x1x1xf32) <- (128x30x1x1xf32, 120x30x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            relu_11, parameter_188, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x120x1x1xf32) <- (120xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(parameter_187, full_int_array_1)

        # pd_op.add: (128x120x1x1xf32) <- (128x120x1x1xf32, 1x120x1x1xf32)
        add_14 = paddle._C_ops.add(conv2d_13, reshape_3)

        # pd_op.hardsigmoid: (128x120x1x1xf32) <- (128x120x1x1xf32)
        hardsigmoid_1 = paddle._C_ops.hardsigmoid(add_14, float("0.2"), float("0.5"))

        # pd_op.multiply: (128x120x28x28xf32) <- (128x120x28x28xf32, 128x120x1x1xf32)
        multiply_1 = paddle._C_ops.multiply(relu_10, hardsigmoid_1)

        # pd_op.conv2d: (128x40x28x28xf32) <- (128x120x28x28xf32, 40x120x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            multiply_1, parameter_186, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x40x28x28xf32, 40xf32, 40xf32, 40xf32, 40xf32, -1xui8) <- (128x40x28x28xf32, 40xf32, 40xf32, 40xf32, 40xf32)
        (
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
            batch_norm__83,
            batch_norm__84,
            batch_norm__85,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_14,
                parameter_185,
                parameter_184,
                parameter_183,
                parameter_182,
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

        # pd_op.add: (128x40x28x28xf32) <- (128x40x28x28xf32, 128x40x28x28xf32)
        add_2 = paddle._C_ops.add(batch_norm__64, batch_norm__80)

        # pd_op.conv2d: (128x120x28x28xf32) <- (128x40x28x28xf32, 120x40x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            add_2, parameter_181, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x120x28x28xf32, 120xf32, 120xf32, 120xf32, 120xf32, -1xui8) <- (128x120x28x28xf32, 120xf32, 120xf32, 120xf32, 120xf32)
        (
            batch_norm__280,
            batch_norm__86,
            batch_norm__87,
            batch_norm__88,
            batch_norm__89,
            batch_norm__90,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_15,
                parameter_180,
                parameter_179,
                parameter_178,
                parameter_177,
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

        # pd_op.relu: (128x120x28x28xf32) <- (128x120x28x28xf32)
        relu_12 = paddle._C_ops.relu(batch_norm__280)

        # pd_op.depthwise_conv2d: (128x120x28x28xf32) <- (128x120x28x28xf32, 120x1x5x5xf32)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(
            relu_12, parameter_176, [1, 1], [2, 2], "EXPLICIT", 120, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (128x120x28x28xf32, 120xf32, 120xf32, 120xf32, 120xf32, -1xui8) <- (128x120x28x28xf32, 120xf32, 120xf32, 120xf32, 120xf32)
        (
            batch_norm__281,
            batch_norm__91,
            batch_norm__92,
            batch_norm__93,
            batch_norm__94,
            batch_norm__95,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_5,
                parameter_175,
                parameter_174,
                parameter_173,
                parameter_172,
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

        # pd_op.relu: (128x120x28x28xf32) <- (128x120x28x28xf32)
        relu_13 = paddle._C_ops.relu(batch_norm__281)

        # pd_op.pool2d: (128x120x1x1xf32) <- (128x120x28x28xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(
            relu_13,
            full_int_array_0,
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

        # pd_op.conv2d: (128x30x1x1xf32) <- (128x120x1x1xf32, 30x120x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            pool2d_2, parameter_171, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x30x1x1xf32) <- (30xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(parameter_170, full_int_array_1)

        # pd_op.add: (128x30x1x1xf32) <- (128x30x1x1xf32, 1x30x1x1xf32)
        add_15 = paddle._C_ops.add(conv2d_16, reshape_4)

        # pd_op.relu: (128x30x1x1xf32) <- (128x30x1x1xf32)
        relu_14 = paddle._C_ops.relu(add_15)

        # pd_op.conv2d: (128x120x1x1xf32) <- (128x30x1x1xf32, 120x30x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            relu_14, parameter_169, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x120x1x1xf32) <- (120xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(parameter_168, full_int_array_1)

        # pd_op.add: (128x120x1x1xf32) <- (128x120x1x1xf32, 1x120x1x1xf32)
        add_16 = paddle._C_ops.add(conv2d_17, reshape_5)

        # pd_op.hardsigmoid: (128x120x1x1xf32) <- (128x120x1x1xf32)
        hardsigmoid_2 = paddle._C_ops.hardsigmoid(add_16, float("0.2"), float("0.5"))

        # pd_op.multiply: (128x120x28x28xf32) <- (128x120x28x28xf32, 128x120x1x1xf32)
        multiply_2 = paddle._C_ops.multiply(relu_13, hardsigmoid_2)

        # pd_op.conv2d: (128x40x28x28xf32) <- (128x120x28x28xf32, 40x120x1x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            multiply_2, parameter_167, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x40x28x28xf32, 40xf32, 40xf32, 40xf32, 40xf32, -1xui8) <- (128x40x28x28xf32, 40xf32, 40xf32, 40xf32, 40xf32)
        (
            batch_norm__96,
            batch_norm__97,
            batch_norm__98,
            batch_norm__99,
            batch_norm__100,
            batch_norm__101,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_18,
                parameter_166,
                parameter_165,
                parameter_164,
                parameter_163,
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

        # pd_op.add: (128x40x28x28xf32) <- (128x40x28x28xf32, 128x40x28x28xf32)
        add_3 = paddle._C_ops.add(add_2, batch_norm__96)

        # pd_op.conv2d: (128x240x28x28xf32) <- (128x40x28x28xf32, 240x40x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            add_3, parameter_162, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x240x28x28xf32, 240xf32, 240xf32, 240xf32, 240xf32, -1xui8) <- (128x240x28x28xf32, 240xf32, 240xf32, 240xf32, 240xf32)
        (
            batch_norm__102,
            batch_norm__103,
            batch_norm__104,
            batch_norm__105,
            batch_norm__106,
            batch_norm__107,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_19,
                parameter_161,
                parameter_160,
                parameter_159,
                parameter_158,
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

        # pd_op.hardswish: (128x240x28x28xf32) <- (128x240x28x28xf32)
        hardswish_1 = paddle._C_ops.hardswish(batch_norm__102)

        # pd_op.depthwise_conv2d: (128x240x14x14xf32) <- (128x240x28x28xf32, 240x1x3x3xf32)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(
            hardswish_1, parameter_157, [2, 2], [1, 1], "EXPLICIT", 240, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (128x240x14x14xf32, 240xf32, 240xf32, 240xf32, 240xf32, -1xui8) <- (128x240x14x14xf32, 240xf32, 240xf32, 240xf32, 240xf32)
        (
            batch_norm__108,
            batch_norm__109,
            batch_norm__110,
            batch_norm__111,
            batch_norm__112,
            batch_norm__113,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_6,
                parameter_156,
                parameter_155,
                parameter_154,
                parameter_153,
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

        # pd_op.hardswish: (128x240x14x14xf32) <- (128x240x14x14xf32)
        hardswish_2 = paddle._C_ops.hardswish(batch_norm__108)

        # pd_op.conv2d: (128x80x14x14xf32) <- (128x240x14x14xf32, 80x240x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            hardswish_2, parameter_152, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x80x14x14xf32, 80xf32, 80xf32, 80xf32, 80xf32, -1xui8) <- (128x80x14x14xf32, 80xf32, 80xf32, 80xf32, 80xf32)
        (
            batch_norm__114,
            batch_norm__115,
            batch_norm__116,
            batch_norm__117,
            batch_norm__118,
            batch_norm__119,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_20,
                parameter_151,
                parameter_150,
                parameter_149,
                parameter_148,
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

        # pd_op.conv2d: (128x200x14x14xf32) <- (128x80x14x14xf32, 200x80x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            batch_norm__114,
            parameter_147,
            [1, 1],
            [0, 0],
            "EXPLICIT",
            [1, 1],
            1,
            "NCHW",
        )

        # pd_op.batch_norm_: (128x200x14x14xf32, 200xf32, 200xf32, 200xf32, 200xf32, -1xui8) <- (128x200x14x14xf32, 200xf32, 200xf32, 200xf32, 200xf32)
        (
            batch_norm__120,
            batch_norm__121,
            batch_norm__122,
            batch_norm__123,
            batch_norm__124,
            batch_norm__125,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_21,
                parameter_146,
                parameter_145,
                parameter_144,
                parameter_143,
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

        # pd_op.hardswish: (128x200x14x14xf32) <- (128x200x14x14xf32)
        hardswish_3 = paddle._C_ops.hardswish(batch_norm__120)

        # pd_op.depthwise_conv2d: (128x200x14x14xf32) <- (128x200x14x14xf32, 200x1x3x3xf32)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(
            hardswish_3, parameter_142, [1, 1], [1, 1], "EXPLICIT", 200, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (128x200x14x14xf32, 200xf32, 200xf32, 200xf32, 200xf32, -1xui8) <- (128x200x14x14xf32, 200xf32, 200xf32, 200xf32, 200xf32)
        (
            batch_norm__126,
            batch_norm__127,
            batch_norm__128,
            batch_norm__129,
            batch_norm__130,
            batch_norm__131,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_7,
                parameter_141,
                parameter_140,
                parameter_139,
                parameter_138,
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

        # pd_op.hardswish: (128x200x14x14xf32) <- (128x200x14x14xf32)
        hardswish_4 = paddle._C_ops.hardswish(batch_norm__126)

        # pd_op.conv2d: (128x80x14x14xf32) <- (128x200x14x14xf32, 80x200x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(
            hardswish_4, parameter_137, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x80x14x14xf32, 80xf32, 80xf32, 80xf32, 80xf32, -1xui8) <- (128x80x14x14xf32, 80xf32, 80xf32, 80xf32, 80xf32)
        (
            batch_norm__132,
            batch_norm__133,
            batch_norm__134,
            batch_norm__135,
            batch_norm__136,
            batch_norm__137,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_22,
                parameter_136,
                parameter_135,
                parameter_134,
                parameter_133,
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

        # pd_op.add: (128x80x14x14xf32) <- (128x80x14x14xf32, 128x80x14x14xf32)
        add_4 = paddle._C_ops.add(batch_norm__114, batch_norm__132)

        # pd_op.conv2d: (128x184x14x14xf32) <- (128x80x14x14xf32, 184x80x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(
            add_4, parameter_132, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x184x14x14xf32, 184xf32, 184xf32, 184xf32, 184xf32, -1xui8) <- (128x184x14x14xf32, 184xf32, 184xf32, 184xf32, 184xf32)
        (
            batch_norm__138,
            batch_norm__139,
            batch_norm__140,
            batch_norm__141,
            batch_norm__142,
            batch_norm__143,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_23,
                parameter_131,
                parameter_130,
                parameter_129,
                parameter_128,
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

        # pd_op.hardswish: (128x184x14x14xf32) <- (128x184x14x14xf32)
        hardswish_5 = paddle._C_ops.hardswish(batch_norm__138)

        # pd_op.depthwise_conv2d: (128x184x14x14xf32) <- (128x184x14x14xf32, 184x1x3x3xf32)
        depthwise_conv2d_8 = paddle._C_ops.depthwise_conv2d(
            hardswish_5, parameter_127, [1, 1], [1, 1], "EXPLICIT", 184, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (128x184x14x14xf32, 184xf32, 184xf32, 184xf32, 184xf32, -1xui8) <- (128x184x14x14xf32, 184xf32, 184xf32, 184xf32, 184xf32)
        (
            batch_norm__144,
            batch_norm__145,
            batch_norm__146,
            batch_norm__147,
            batch_norm__148,
            batch_norm__149,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_8,
                parameter_126,
                parameter_125,
                parameter_124,
                parameter_123,
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

        # pd_op.hardswish: (128x184x14x14xf32) <- (128x184x14x14xf32)
        hardswish_6 = paddle._C_ops.hardswish(batch_norm__144)

        # pd_op.conv2d: (128x80x14x14xf32) <- (128x184x14x14xf32, 80x184x1x1xf32)
        conv2d_24 = paddle._C_ops.conv2d(
            hardswish_6, parameter_122, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x80x14x14xf32, 80xf32, 80xf32, 80xf32, 80xf32, -1xui8) <- (128x80x14x14xf32, 80xf32, 80xf32, 80xf32, 80xf32)
        (
            batch_norm__150,
            batch_norm__151,
            batch_norm__152,
            batch_norm__153,
            batch_norm__154,
            batch_norm__155,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_24,
                parameter_121,
                parameter_120,
                parameter_119,
                parameter_118,
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

        # pd_op.add: (128x80x14x14xf32) <- (128x80x14x14xf32, 128x80x14x14xf32)
        add_5 = paddle._C_ops.add(add_4, batch_norm__150)

        # pd_op.conv2d: (128x184x14x14xf32) <- (128x80x14x14xf32, 184x80x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(
            add_5, parameter_117, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x184x14x14xf32, 184xf32, 184xf32, 184xf32, 184xf32, -1xui8) <- (128x184x14x14xf32, 184xf32, 184xf32, 184xf32, 184xf32)
        (
            batch_norm__156,
            batch_norm__157,
            batch_norm__158,
            batch_norm__159,
            batch_norm__160,
            batch_norm__161,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_25,
                parameter_116,
                parameter_115,
                parameter_114,
                parameter_113,
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

        # pd_op.hardswish: (128x184x14x14xf32) <- (128x184x14x14xf32)
        hardswish_7 = paddle._C_ops.hardswish(batch_norm__156)

        # pd_op.depthwise_conv2d: (128x184x14x14xf32) <- (128x184x14x14xf32, 184x1x3x3xf32)
        depthwise_conv2d_9 = paddle._C_ops.depthwise_conv2d(
            hardswish_7, parameter_112, [1, 1], [1, 1], "EXPLICIT", 184, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (128x184x14x14xf32, 184xf32, 184xf32, 184xf32, 184xf32, -1xui8) <- (128x184x14x14xf32, 184xf32, 184xf32, 184xf32, 184xf32)
        (
            batch_norm__162,
            batch_norm__163,
            batch_norm__164,
            batch_norm__165,
            batch_norm__166,
            batch_norm__167,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_9,
                parameter_111,
                parameter_110,
                parameter_109,
                parameter_108,
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

        # pd_op.hardswish: (128x184x14x14xf32) <- (128x184x14x14xf32)
        hardswish_8 = paddle._C_ops.hardswish(batch_norm__162)

        # pd_op.conv2d: (128x80x14x14xf32) <- (128x184x14x14xf32, 80x184x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(
            hardswish_8, parameter_107, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x80x14x14xf32, 80xf32, 80xf32, 80xf32, 80xf32, -1xui8) <- (128x80x14x14xf32, 80xf32, 80xf32, 80xf32, 80xf32)
        (
            batch_norm__168,
            batch_norm__169,
            batch_norm__170,
            batch_norm__171,
            batch_norm__172,
            batch_norm__173,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_26,
                parameter_106,
                parameter_105,
                parameter_104,
                parameter_103,
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

        # pd_op.add: (128x80x14x14xf32) <- (128x80x14x14xf32, 128x80x14x14xf32)
        add_6 = paddle._C_ops.add(add_5, batch_norm__168)

        # pd_op.conv2d: (128x480x14x14xf32) <- (128x80x14x14xf32, 480x80x1x1xf32)
        conv2d_27 = paddle._C_ops.conv2d(
            add_6, parameter_102, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x480x14x14xf32, 480xf32, 480xf32, 480xf32, 480xf32, -1xui8) <- (128x480x14x14xf32, 480xf32, 480xf32, 480xf32, 480xf32)
        (
            batch_norm__174,
            batch_norm__175,
            batch_norm__176,
            batch_norm__177,
            batch_norm__178,
            batch_norm__179,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_27,
                parameter_101,
                parameter_100,
                parameter_99,
                parameter_98,
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

        # pd_op.hardswish: (128x480x14x14xf32) <- (128x480x14x14xf32)
        hardswish_9 = paddle._C_ops.hardswish(batch_norm__174)

        # pd_op.depthwise_conv2d: (128x480x14x14xf32) <- (128x480x14x14xf32, 480x1x3x3xf32)
        depthwise_conv2d_10 = paddle._C_ops.depthwise_conv2d(
            hardswish_9, parameter_97, [1, 1], [1, 1], "EXPLICIT", 480, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (128x480x14x14xf32, 480xf32, 480xf32, 480xf32, 480xf32, -1xui8) <- (128x480x14x14xf32, 480xf32, 480xf32, 480xf32, 480xf32)
        (
            batch_norm__180,
            batch_norm__181,
            batch_norm__182,
            batch_norm__183,
            batch_norm__184,
            batch_norm__185,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_10,
                parameter_96,
                parameter_95,
                parameter_94,
                parameter_93,
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

        # pd_op.hardswish: (128x480x14x14xf32) <- (128x480x14x14xf32)
        hardswish_10 = paddle._C_ops.hardswish(batch_norm__180)

        # pd_op.pool2d: (128x480x1x1xf32) <- (128x480x14x14xf32, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(
            hardswish_10,
            full_int_array_0,
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

        # pd_op.conv2d: (128x120x1x1xf32) <- (128x480x1x1xf32, 120x480x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(
            pool2d_3, parameter_92, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x120x1x1xf32) <- (120xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(parameter_91, full_int_array_1)

        # pd_op.add: (128x120x1x1xf32) <- (128x120x1x1xf32, 1x120x1x1xf32)
        add_17 = paddle._C_ops.add(conv2d_28, reshape_6)

        # pd_op.relu: (128x120x1x1xf32) <- (128x120x1x1xf32)
        relu_15 = paddle._C_ops.relu(add_17)

        # pd_op.conv2d: (128x480x1x1xf32) <- (128x120x1x1xf32, 480x120x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(
            relu_15, parameter_90, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x480x1x1xf32) <- (480xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(parameter_89, full_int_array_1)

        # pd_op.add: (128x480x1x1xf32) <- (128x480x1x1xf32, 1x480x1x1xf32)
        add_18 = paddle._C_ops.add(conv2d_29, reshape_7)

        # pd_op.hardsigmoid: (128x480x1x1xf32) <- (128x480x1x1xf32)
        hardsigmoid_3 = paddle._C_ops.hardsigmoid(add_18, float("0.2"), float("0.5"))

        # pd_op.multiply: (128x480x14x14xf32) <- (128x480x14x14xf32, 128x480x1x1xf32)
        multiply_3 = paddle._C_ops.multiply(hardswish_10, hardsigmoid_3)

        # pd_op.conv2d: (128x112x14x14xf32) <- (128x480x14x14xf32, 112x480x1x1xf32)
        conv2d_30 = paddle._C_ops.conv2d(
            multiply_3, parameter_88, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x112x14x14xf32, 112xf32, 112xf32, 112xf32, 112xf32, -1xui8) <- (128x112x14x14xf32, 112xf32, 112xf32, 112xf32, 112xf32)
        (
            batch_norm__186,
            batch_norm__187,
            batch_norm__188,
            batch_norm__189,
            batch_norm__190,
            batch_norm__191,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_30,
                parameter_87,
                parameter_86,
                parameter_85,
                parameter_84,
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

        # pd_op.conv2d: (128x672x14x14xf32) <- (128x112x14x14xf32, 672x112x1x1xf32)
        conv2d_31 = paddle._C_ops.conv2d(
            batch_norm__186, parameter_83, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x672x14x14xf32, 672xf32, 672xf32, 672xf32, 672xf32, -1xui8) <- (128x672x14x14xf32, 672xf32, 672xf32, 672xf32, 672xf32)
        (
            batch_norm__192,
            batch_norm__193,
            batch_norm__194,
            batch_norm__195,
            batch_norm__196,
            batch_norm__197,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_31,
                parameter_82,
                parameter_81,
                parameter_80,
                parameter_79,
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

        # pd_op.hardswish: (128x672x14x14xf32) <- (128x672x14x14xf32)
        hardswish_11 = paddle._C_ops.hardswish(batch_norm__192)

        # pd_op.depthwise_conv2d: (128x672x14x14xf32) <- (128x672x14x14xf32, 672x1x3x3xf32)
        depthwise_conv2d_11 = paddle._C_ops.depthwise_conv2d(
            hardswish_11, parameter_78, [1, 1], [1, 1], "EXPLICIT", 672, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (128x672x14x14xf32, 672xf32, 672xf32, 672xf32, 672xf32, -1xui8) <- (128x672x14x14xf32, 672xf32, 672xf32, 672xf32, 672xf32)
        (
            batch_norm__198,
            batch_norm__199,
            batch_norm__200,
            batch_norm__201,
            batch_norm__202,
            batch_norm__203,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_11,
                parameter_77,
                parameter_76,
                parameter_75,
                parameter_74,
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

        # pd_op.hardswish: (128x672x14x14xf32) <- (128x672x14x14xf32)
        hardswish_12 = paddle._C_ops.hardswish(batch_norm__198)

        # pd_op.pool2d: (128x672x1x1xf32) <- (128x672x14x14xf32, 2xi64)
        pool2d_4 = paddle._C_ops.pool2d(
            hardswish_12,
            full_int_array_0,
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

        # pd_op.conv2d: (128x168x1x1xf32) <- (128x672x1x1xf32, 168x672x1x1xf32)
        conv2d_32 = paddle._C_ops.conv2d(
            pool2d_4, parameter_73, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x168x1x1xf32) <- (168xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(parameter_72, full_int_array_1)

        # pd_op.add: (128x168x1x1xf32) <- (128x168x1x1xf32, 1x168x1x1xf32)
        add_19 = paddle._C_ops.add(conv2d_32, reshape_8)

        # pd_op.relu: (128x168x1x1xf32) <- (128x168x1x1xf32)
        relu_16 = paddle._C_ops.relu(add_19)

        # pd_op.conv2d: (128x672x1x1xf32) <- (128x168x1x1xf32, 672x168x1x1xf32)
        conv2d_33 = paddle._C_ops.conv2d(
            relu_16, parameter_71, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x672x1x1xf32) <- (672xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(parameter_70, full_int_array_1)

        # pd_op.add: (128x672x1x1xf32) <- (128x672x1x1xf32, 1x672x1x1xf32)
        add_20 = paddle._C_ops.add(conv2d_33, reshape_9)

        # pd_op.hardsigmoid: (128x672x1x1xf32) <- (128x672x1x1xf32)
        hardsigmoid_4 = paddle._C_ops.hardsigmoid(add_20, float("0.2"), float("0.5"))

        # pd_op.multiply: (128x672x14x14xf32) <- (128x672x14x14xf32, 128x672x1x1xf32)
        multiply_4 = paddle._C_ops.multiply(hardswish_12, hardsigmoid_4)

        # pd_op.conv2d: (128x112x14x14xf32) <- (128x672x14x14xf32, 112x672x1x1xf32)
        conv2d_34 = paddle._C_ops.conv2d(
            multiply_4, parameter_69, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x112x14x14xf32, 112xf32, 112xf32, 112xf32, 112xf32, -1xui8) <- (128x112x14x14xf32, 112xf32, 112xf32, 112xf32, 112xf32)
        (
            batch_norm__204,
            batch_norm__205,
            batch_norm__206,
            batch_norm__207,
            batch_norm__208,
            batch_norm__209,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_34,
                parameter_68,
                parameter_67,
                parameter_66,
                parameter_65,
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

        # pd_op.add: (128x112x14x14xf32) <- (128x112x14x14xf32, 128x112x14x14xf32)
        add_7 = paddle._C_ops.add(batch_norm__186, batch_norm__204)

        # pd_op.conv2d: (128x672x14x14xf32) <- (128x112x14x14xf32, 672x112x1x1xf32)
        conv2d_35 = paddle._C_ops.conv2d(
            add_7, parameter_64, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x672x14x14xf32, 672xf32, 672xf32, 672xf32, 672xf32, -1xui8) <- (128x672x14x14xf32, 672xf32, 672xf32, 672xf32, 672xf32)
        (
            batch_norm__210,
            batch_norm__211,
            batch_norm__212,
            batch_norm__213,
            batch_norm__214,
            batch_norm__215,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_35,
                parameter_63,
                parameter_62,
                parameter_61,
                parameter_60,
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

        # pd_op.hardswish: (128x672x14x14xf32) <- (128x672x14x14xf32)
        hardswish_13 = paddle._C_ops.hardswish(batch_norm__210)

        # pd_op.depthwise_conv2d: (128x672x7x7xf32) <- (128x672x14x14xf32, 672x1x5x5xf32)
        depthwise_conv2d_12 = paddle._C_ops.depthwise_conv2d(
            hardswish_13, parameter_59, [2, 2], [2, 2], "EXPLICIT", 672, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (128x672x7x7xf32, 672xf32, 672xf32, 672xf32, 672xf32, -1xui8) <- (128x672x7x7xf32, 672xf32, 672xf32, 672xf32, 672xf32)
        (
            batch_norm__216,
            batch_norm__217,
            batch_norm__218,
            batch_norm__219,
            batch_norm__220,
            batch_norm__221,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_12,
                parameter_58,
                parameter_57,
                parameter_56,
                parameter_55,
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

        # pd_op.hardswish: (128x672x7x7xf32) <- (128x672x7x7xf32)
        hardswish_14 = paddle._C_ops.hardswish(batch_norm__216)

        # pd_op.pool2d: (128x672x1x1xf32) <- (128x672x7x7xf32, 2xi64)
        pool2d_5 = paddle._C_ops.pool2d(
            hardswish_14,
            full_int_array_0,
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

        # pd_op.conv2d: (128x168x1x1xf32) <- (128x672x1x1xf32, 168x672x1x1xf32)
        conv2d_36 = paddle._C_ops.conv2d(
            pool2d_5, parameter_54, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x168x1x1xf32) <- (168xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(parameter_53, full_int_array_1)

        # pd_op.add: (128x168x1x1xf32) <- (128x168x1x1xf32, 1x168x1x1xf32)
        add_21 = paddle._C_ops.add(conv2d_36, reshape_10)

        # pd_op.relu: (128x168x1x1xf32) <- (128x168x1x1xf32)
        relu_17 = paddle._C_ops.relu(add_21)

        # pd_op.conv2d: (128x672x1x1xf32) <- (128x168x1x1xf32, 672x168x1x1xf32)
        conv2d_37 = paddle._C_ops.conv2d(
            relu_17, parameter_52, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x672x1x1xf32) <- (672xf32, 4xi64)
        reshape_11 = paddle._C_ops.reshape(parameter_51, full_int_array_1)

        # pd_op.add: (128x672x1x1xf32) <- (128x672x1x1xf32, 1x672x1x1xf32)
        add_22 = paddle._C_ops.add(conv2d_37, reshape_11)

        # pd_op.hardsigmoid: (128x672x1x1xf32) <- (128x672x1x1xf32)
        hardsigmoid_5 = paddle._C_ops.hardsigmoid(add_22, float("0.2"), float("0.5"))

        # pd_op.multiply: (128x672x7x7xf32) <- (128x672x7x7xf32, 128x672x1x1xf32)
        multiply_5 = paddle._C_ops.multiply(hardswish_14, hardsigmoid_5)

        # pd_op.conv2d: (128x160x7x7xf32) <- (128x672x7x7xf32, 160x672x1x1xf32)
        conv2d_38 = paddle._C_ops.conv2d(
            multiply_5, parameter_50, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x160x7x7xf32, 160xf32, 160xf32, 160xf32, 160xf32, -1xui8) <- (128x160x7x7xf32, 160xf32, 160xf32, 160xf32, 160xf32)
        (
            batch_norm__222,
            batch_norm__223,
            batch_norm__224,
            batch_norm__225,
            batch_norm__226,
            batch_norm__227,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_38,
                parameter_49,
                parameter_48,
                parameter_47,
                parameter_46,
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

        # pd_op.conv2d: (128x960x7x7xf32) <- (128x160x7x7xf32, 960x160x1x1xf32)
        conv2d_39 = paddle._C_ops.conv2d(
            batch_norm__222, parameter_45, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x960x7x7xf32, 960xf32, 960xf32, 960xf32, 960xf32, -1xui8) <- (128x960x7x7xf32, 960xf32, 960xf32, 960xf32, 960xf32)
        (
            batch_norm__228,
            batch_norm__229,
            batch_norm__230,
            batch_norm__231,
            batch_norm__232,
            batch_norm__233,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_39,
                parameter_44,
                parameter_43,
                parameter_42,
                parameter_41,
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

        # pd_op.hardswish: (128x960x7x7xf32) <- (128x960x7x7xf32)
        hardswish_15 = paddle._C_ops.hardswish(batch_norm__228)

        # pd_op.depthwise_conv2d: (128x960x7x7xf32) <- (128x960x7x7xf32, 960x1x5x5xf32)
        depthwise_conv2d_13 = paddle._C_ops.depthwise_conv2d(
            hardswish_15, parameter_40, [1, 1], [2, 2], "EXPLICIT", 960, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (128x960x7x7xf32, 960xf32, 960xf32, 960xf32, 960xf32, -1xui8) <- (128x960x7x7xf32, 960xf32, 960xf32, 960xf32, 960xf32)
        (
            batch_norm__234,
            batch_norm__235,
            batch_norm__236,
            batch_norm__237,
            batch_norm__238,
            batch_norm__239,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_13,
                parameter_39,
                parameter_38,
                parameter_37,
                parameter_36,
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

        # pd_op.hardswish: (128x960x7x7xf32) <- (128x960x7x7xf32)
        hardswish_16 = paddle._C_ops.hardswish(batch_norm__234)

        # pd_op.pool2d: (128x960x1x1xf32) <- (128x960x7x7xf32, 2xi64)
        pool2d_6 = paddle._C_ops.pool2d(
            hardswish_16,
            full_int_array_0,
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

        # pd_op.conv2d: (128x240x1x1xf32) <- (128x960x1x1xf32, 240x960x1x1xf32)
        conv2d_40 = paddle._C_ops.conv2d(
            pool2d_6, parameter_35, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x240x1x1xf32) <- (240xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(parameter_34, full_int_array_1)

        # pd_op.add: (128x240x1x1xf32) <- (128x240x1x1xf32, 1x240x1x1xf32)
        add_23 = paddle._C_ops.add(conv2d_40, reshape_12)

        # pd_op.relu: (128x240x1x1xf32) <- (128x240x1x1xf32)
        relu_18 = paddle._C_ops.relu(add_23)

        # pd_op.conv2d: (128x960x1x1xf32) <- (128x240x1x1xf32, 960x240x1x1xf32)
        conv2d_41 = paddle._C_ops.conv2d(
            relu_18, parameter_33, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x960x1x1xf32) <- (960xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(parameter_32, full_int_array_1)

        # pd_op.add: (128x960x1x1xf32) <- (128x960x1x1xf32, 1x960x1x1xf32)
        add_24 = paddle._C_ops.add(conv2d_41, reshape_13)

        # pd_op.hardsigmoid: (128x960x1x1xf32) <- (128x960x1x1xf32)
        hardsigmoid_6 = paddle._C_ops.hardsigmoid(add_24, float("0.2"), float("0.5"))

        # pd_op.multiply: (128x960x7x7xf32) <- (128x960x7x7xf32, 128x960x1x1xf32)
        multiply_6 = paddle._C_ops.multiply(hardswish_16, hardsigmoid_6)

        # pd_op.conv2d: (128x160x7x7xf32) <- (128x960x7x7xf32, 160x960x1x1xf32)
        conv2d_42 = paddle._C_ops.conv2d(
            multiply_6, parameter_31, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x160x7x7xf32, 160xf32, 160xf32, 160xf32, 160xf32, -1xui8) <- (128x160x7x7xf32, 160xf32, 160xf32, 160xf32, 160xf32)
        (
            batch_norm__240,
            batch_norm__241,
            batch_norm__242,
            batch_norm__243,
            batch_norm__244,
            batch_norm__245,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_42,
                parameter_30,
                parameter_29,
                parameter_28,
                parameter_27,
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

        # pd_op.add: (128x160x7x7xf32) <- (128x160x7x7xf32, 128x160x7x7xf32)
        add_8 = paddle._C_ops.add(batch_norm__222, batch_norm__240)

        # pd_op.conv2d: (128x960x7x7xf32) <- (128x160x7x7xf32, 960x160x1x1xf32)
        conv2d_43 = paddle._C_ops.conv2d(
            add_8, parameter_26, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x960x7x7xf32, 960xf32, 960xf32, 960xf32, 960xf32, -1xui8) <- (128x960x7x7xf32, 960xf32, 960xf32, 960xf32, 960xf32)
        (
            batch_norm__246,
            batch_norm__247,
            batch_norm__248,
            batch_norm__249,
            batch_norm__250,
            batch_norm__251,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_43,
                parameter_25,
                parameter_24,
                parameter_23,
                parameter_22,
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

        # pd_op.hardswish: (128x960x7x7xf32) <- (128x960x7x7xf32)
        hardswish_17 = paddle._C_ops.hardswish(batch_norm__246)

        # pd_op.depthwise_conv2d: (128x960x7x7xf32) <- (128x960x7x7xf32, 960x1x5x5xf32)
        depthwise_conv2d_14 = paddle._C_ops.depthwise_conv2d(
            hardswish_17, parameter_21, [1, 1], [2, 2], "EXPLICIT", 960, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (128x960x7x7xf32, 960xf32, 960xf32, 960xf32, 960xf32, -1xui8) <- (128x960x7x7xf32, 960xf32, 960xf32, 960xf32, 960xf32)
        (
            batch_norm__252,
            batch_norm__253,
            batch_norm__254,
            batch_norm__255,
            batch_norm__256,
            batch_norm__257,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_14,
                parameter_20,
                parameter_19,
                parameter_18,
                parameter_17,
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

        # pd_op.hardswish: (128x960x7x7xf32) <- (128x960x7x7xf32)
        hardswish_18 = paddle._C_ops.hardswish(batch_norm__252)

        # pd_op.pool2d: (128x960x1x1xf32) <- (128x960x7x7xf32, 2xi64)
        pool2d_7 = paddle._C_ops.pool2d(
            hardswish_18,
            full_int_array_0,
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

        # pd_op.conv2d: (128x240x1x1xf32) <- (128x960x1x1xf32, 240x960x1x1xf32)
        conv2d_44 = paddle._C_ops.conv2d(
            pool2d_7, parameter_16, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x240x1x1xf32) <- (240xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(parameter_15, full_int_array_1)

        # pd_op.add: (128x240x1x1xf32) <- (128x240x1x1xf32, 1x240x1x1xf32)
        add_25 = paddle._C_ops.add(conv2d_44, reshape_14)

        # pd_op.relu: (128x240x1x1xf32) <- (128x240x1x1xf32)
        relu_19 = paddle._C_ops.relu(add_25)

        # pd_op.conv2d: (128x960x1x1xf32) <- (128x240x1x1xf32, 960x240x1x1xf32)
        conv2d_45 = paddle._C_ops.conv2d(
            relu_19, parameter_14, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x960x1x1xf32) <- (960xf32, 4xi64)
        reshape_15 = paddle._C_ops.reshape(parameter_13, full_int_array_1)

        # pd_op.add: (128x960x1x1xf32) <- (128x960x1x1xf32, 1x960x1x1xf32)
        add_26 = paddle._C_ops.add(conv2d_45, reshape_15)

        # pd_op.hardsigmoid: (128x960x1x1xf32) <- (128x960x1x1xf32)
        hardsigmoid_7 = paddle._C_ops.hardsigmoid(add_26, float("0.2"), float("0.5"))

        # pd_op.multiply: (128x960x7x7xf32) <- (128x960x7x7xf32, 128x960x1x1xf32)
        multiply_7 = paddle._C_ops.multiply(hardswish_18, hardsigmoid_7)

        # pd_op.conv2d: (128x160x7x7xf32) <- (128x960x7x7xf32, 160x960x1x1xf32)
        conv2d_46 = paddle._C_ops.conv2d(
            multiply_7, parameter_12, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x160x7x7xf32, 160xf32, 160xf32, 160xf32, 160xf32, -1xui8) <- (128x160x7x7xf32, 160xf32, 160xf32, 160xf32, 160xf32)
        (
            batch_norm__258,
            batch_norm__259,
            batch_norm__260,
            batch_norm__261,
            batch_norm__262,
            batch_norm__263,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_46,
                parameter_11,
                parameter_10,
                parameter_9,
                parameter_8,
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

        # pd_op.add: (128x160x7x7xf32) <- (128x160x7x7xf32, 128x160x7x7xf32)
        add_9 = paddle._C_ops.add(add_8, batch_norm__258)

        # pd_op.conv2d: (128x960x7x7xf32) <- (128x160x7x7xf32, 960x160x1x1xf32)
        conv2d_47 = paddle._C_ops.conv2d(
            add_9, parameter_7, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x960x7x7xf32, 960xf32, 960xf32, 960xf32, 960xf32, -1xui8) <- (128x960x7x7xf32, 960xf32, 960xf32, 960xf32, 960xf32)
        (
            batch_norm__264,
            batch_norm__265,
            batch_norm__266,
            batch_norm__267,
            batch_norm__268,
            batch_norm__269,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_47,
                parameter_6,
                parameter_5,
                parameter_4,
                parameter_3,
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

        # pd_op.hardswish: (128x960x7x7xf32) <- (128x960x7x7xf32)
        hardswish_19 = paddle._C_ops.hardswish(batch_norm__264)

        # pd_op.pool2d: (128x960x1x1xf32) <- (128x960x7x7xf32, 2xi64)
        pool2d_8 = paddle._C_ops.pool2d(
            hardswish_19,
            full_int_array_0,
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

        # pd_op.conv2d: (128x1280x1x1xf32) <- (128x960x1x1xf32, 1280x960x1x1xf32)
        conv2d_48 = paddle._C_ops.conv2d(
            pool2d_8, parameter_2, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.hardswish: (128x1280x1x1xf32) <- (128x1280x1x1xf32)
        hardswish_20 = paddle._C_ops.hardswish(conv2d_48)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (128x1280x1x1xf32, 128x1280x1x1xui8) <- (128x1280x1x1xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                hardswish_20, None, full_0, False, "downgrade_in_infer", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.flatten: (128x1280xf32) <- (128x1280x1x1xf32)
        flatten_0 = paddle._C_ops.flatten(dropout_0, 1, 3)

        # pd_op.matmul: (128x102xf32) <- (128x1280xf32, 1280x102xf32)
        matmul_0 = paddle._C_ops.matmul(flatten_0, parameter_1, False, False)

        # pd_op.add: (128x102xf32) <- (128x102xf32, 102xf32)
        add_10 = paddle._C_ops.add(matmul_0, parameter_0)
        return (
            conv2d_0,
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
            batch_norm__5,
            hardswish_0,
            conv2d_1,
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            batch_norm__10,
            relu_0,
            depthwise_conv2d_0,
            batch_norm__11,
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            relu_1,
            conv2d_2,
            batch_norm__16,
            batch_norm__17,
            batch_norm__18,
            batch_norm__19,
            batch_norm__20,
            batch_norm__21,
            add_0,
            conv2d_3,
            batch_norm__22,
            batch_norm__23,
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
            relu_2,
            depthwise_conv2d_1,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
            batch_norm__30,
            batch_norm__31,
            relu_3,
            conv2d_4,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
            batch_norm__36,
            batch_norm__37,
            conv2d_5,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
            batch_norm__42,
            relu_4,
            depthwise_conv2d_2,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
            relu_5,
            conv2d_6,
            batch_norm__48,
            batch_norm__49,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
            add_1,
            conv2d_7,
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            relu_6,
            depthwise_conv2d_3,
            batch_norm__59,
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            relu_7,
            full_int_array_0,
            pool2d_0,
            conv2d_8,
            reshape_0,
            relu_8,
            conv2d_9,
            reshape_1,
            hardsigmoid_0,
            multiply_0,
            conv2d_10,
            batch_norm__64,
            batch_norm__65,
            batch_norm__66,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            conv2d_11,
            batch_norm__70,
            batch_norm__71,
            batch_norm__72,
            batch_norm__73,
            batch_norm__74,
            relu_9,
            depthwise_conv2d_4,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
            batch_norm__78,
            batch_norm__79,
            relu_10,
            assign_0,
            pool2d_1,
            conv2d_12,
            reshape_2,
            relu_11,
            conv2d_13,
            reshape_3,
            hardsigmoid_1,
            multiply_1,
            conv2d_14,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
            batch_norm__83,
            batch_norm__84,
            batch_norm__85,
            add_2,
            conv2d_15,
            batch_norm__86,
            batch_norm__87,
            batch_norm__88,
            batch_norm__89,
            batch_norm__90,
            relu_12,
            depthwise_conv2d_5,
            batch_norm__91,
            batch_norm__92,
            batch_norm__93,
            batch_norm__94,
            batch_norm__95,
            relu_13,
            assign_1,
            pool2d_2,
            conv2d_16,
            reshape_4,
            relu_14,
            conv2d_17,
            reshape_5,
            hardsigmoid_2,
            multiply_2,
            conv2d_18,
            batch_norm__96,
            batch_norm__97,
            batch_norm__98,
            batch_norm__99,
            batch_norm__100,
            batch_norm__101,
            add_3,
            conv2d_19,
            batch_norm__102,
            batch_norm__103,
            batch_norm__104,
            batch_norm__105,
            batch_norm__106,
            batch_norm__107,
            hardswish_1,
            depthwise_conv2d_6,
            batch_norm__108,
            batch_norm__109,
            batch_norm__110,
            batch_norm__111,
            batch_norm__112,
            batch_norm__113,
            hardswish_2,
            conv2d_20,
            batch_norm__114,
            batch_norm__115,
            batch_norm__116,
            batch_norm__117,
            batch_norm__118,
            batch_norm__119,
            conv2d_21,
            batch_norm__120,
            batch_norm__121,
            batch_norm__122,
            batch_norm__123,
            batch_norm__124,
            batch_norm__125,
            hardswish_3,
            depthwise_conv2d_7,
            batch_norm__126,
            batch_norm__127,
            batch_norm__128,
            batch_norm__129,
            batch_norm__130,
            batch_norm__131,
            hardswish_4,
            conv2d_22,
            batch_norm__132,
            batch_norm__133,
            batch_norm__134,
            batch_norm__135,
            batch_norm__136,
            batch_norm__137,
            add_4,
            conv2d_23,
            batch_norm__138,
            batch_norm__139,
            batch_norm__140,
            batch_norm__141,
            batch_norm__142,
            batch_norm__143,
            hardswish_5,
            depthwise_conv2d_8,
            batch_norm__144,
            batch_norm__145,
            batch_norm__146,
            batch_norm__147,
            batch_norm__148,
            batch_norm__149,
            hardswish_6,
            conv2d_24,
            batch_norm__150,
            batch_norm__151,
            batch_norm__152,
            batch_norm__153,
            batch_norm__154,
            batch_norm__155,
            add_5,
            conv2d_25,
            batch_norm__156,
            batch_norm__157,
            batch_norm__158,
            batch_norm__159,
            batch_norm__160,
            batch_norm__161,
            hardswish_7,
            depthwise_conv2d_9,
            batch_norm__162,
            batch_norm__163,
            batch_norm__164,
            batch_norm__165,
            batch_norm__166,
            batch_norm__167,
            hardswish_8,
            conv2d_26,
            batch_norm__168,
            batch_norm__169,
            batch_norm__170,
            batch_norm__171,
            batch_norm__172,
            batch_norm__173,
            add_6,
            conv2d_27,
            batch_norm__174,
            batch_norm__175,
            batch_norm__176,
            batch_norm__177,
            batch_norm__178,
            batch_norm__179,
            hardswish_9,
            depthwise_conv2d_10,
            batch_norm__180,
            batch_norm__181,
            batch_norm__182,
            batch_norm__183,
            batch_norm__184,
            batch_norm__185,
            hardswish_10,
            assign_2,
            pool2d_3,
            conv2d_28,
            reshape_6,
            relu_15,
            conv2d_29,
            reshape_7,
            hardsigmoid_3,
            multiply_3,
            conv2d_30,
            batch_norm__186,
            batch_norm__187,
            batch_norm__188,
            batch_norm__189,
            batch_norm__190,
            batch_norm__191,
            conv2d_31,
            batch_norm__192,
            batch_norm__193,
            batch_norm__194,
            batch_norm__195,
            batch_norm__196,
            batch_norm__197,
            hardswish_11,
            depthwise_conv2d_11,
            batch_norm__198,
            batch_norm__199,
            batch_norm__200,
            batch_norm__201,
            batch_norm__202,
            batch_norm__203,
            hardswish_12,
            assign_3,
            pool2d_4,
            conv2d_32,
            reshape_8,
            relu_16,
            conv2d_33,
            reshape_9,
            hardsigmoid_4,
            multiply_4,
            conv2d_34,
            batch_norm__204,
            batch_norm__205,
            batch_norm__206,
            batch_norm__207,
            batch_norm__208,
            batch_norm__209,
            add_7,
            conv2d_35,
            batch_norm__210,
            batch_norm__211,
            batch_norm__212,
            batch_norm__213,
            batch_norm__214,
            batch_norm__215,
            hardswish_13,
            depthwise_conv2d_12,
            batch_norm__216,
            batch_norm__217,
            batch_norm__218,
            batch_norm__219,
            batch_norm__220,
            batch_norm__221,
            hardswish_14,
            assign_4,
            pool2d_5,
            conv2d_36,
            reshape_10,
            relu_17,
            conv2d_37,
            reshape_11,
            hardsigmoid_5,
            multiply_5,
            conv2d_38,
            batch_norm__222,
            batch_norm__223,
            batch_norm__224,
            batch_norm__225,
            batch_norm__226,
            batch_norm__227,
            conv2d_39,
            batch_norm__228,
            batch_norm__229,
            batch_norm__230,
            batch_norm__231,
            batch_norm__232,
            batch_norm__233,
            hardswish_15,
            depthwise_conv2d_13,
            batch_norm__234,
            batch_norm__235,
            batch_norm__236,
            batch_norm__237,
            batch_norm__238,
            batch_norm__239,
            hardswish_16,
            assign_5,
            pool2d_6,
            conv2d_40,
            reshape_12,
            relu_18,
            conv2d_41,
            reshape_13,
            hardsigmoid_6,
            multiply_6,
            conv2d_42,
            batch_norm__240,
            batch_norm__241,
            batch_norm__242,
            batch_norm__243,
            batch_norm__244,
            batch_norm__245,
            add_8,
            conv2d_43,
            batch_norm__246,
            batch_norm__247,
            batch_norm__248,
            batch_norm__249,
            batch_norm__250,
            batch_norm__251,
            hardswish_17,
            depthwise_conv2d_14,
            batch_norm__252,
            batch_norm__253,
            batch_norm__254,
            batch_norm__255,
            batch_norm__256,
            batch_norm__257,
            hardswish_18,
            assign_6,
            pool2d_7,
            conv2d_44,
            reshape_14,
            relu_19,
            conv2d_45,
            reshape_15,
            hardsigmoid_7,
            multiply_7,
            conv2d_46,
            batch_norm__258,
            batch_norm__259,
            batch_norm__260,
            batch_norm__261,
            batch_norm__262,
            batch_norm__263,
            add_9,
            conv2d_47,
            batch_norm__264,
            batch_norm__265,
            batch_norm__266,
            batch_norm__267,
            batch_norm__268,
            batch_norm__269,
            hardswish_19,
            assign_7,
            pool2d_8,
            conv2d_48,
            full_0,
            dropout_0,
            dropout_1,
            flatten_0,
            matmul_0,
            add_10,
        )
