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
        data_0,
    ):
        # pd_op.conv2d: (-1x48x112x112xf32) <- (-1x3x224x224xf32, 48x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_266, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_0, parameter_266

        # pd_op.batch_norm_: (-1x48x112x112xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (-1x48x112x112xf32, 48xf32, 48xf32, 48xf32, 48xf32)
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
                parameter_265,
                parameter_264,
                parameter_263,
                parameter_262,
                True,
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
        del conv2d_0, parameter_262, parameter_263, parameter_264, parameter_265

        # pd_op.relu6: (-1x48x112x112xf32) <- (-1x48x112x112xf32)
        relu6_0 = paddle._C_ops.relu6(batch_norm__0)
        del batch_norm__0

        # pd_op.conv2d: (-1x48x112x112xf32) <- (-1x48x112x112xf32, 48x48x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            relu6_0, parameter_261, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_261, relu6_0

        # pd_op.batch_norm_: (-1x48x112x112xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (-1x48x112x112xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        (
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            batch_norm__10,
            batch_norm__11,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_1,
                parameter_260,
                parameter_259,
                parameter_258,
                parameter_257,
                True,
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
        del conv2d_1, parameter_257, parameter_258, parameter_259, parameter_260

        # pd_op.relu6: (-1x48x112x112xf32) <- (-1x48x112x112xf32)
        relu6_1 = paddle._C_ops.relu6(batch_norm__6)
        del batch_norm__6

        # pd_op.depthwise_conv2d: (-1x48x112x112xf32) <- (-1x48x112x112xf32, 48x1x3x3xf32)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(
            relu6_1, parameter_256, [1, 1], [1, 1], "EXPLICIT", 48, [1, 1], "NCHW"
        )
        del parameter_256, relu6_1

        # pd_op.batch_norm_: (-1x48x112x112xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (-1x48x112x112xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        (
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            batch_norm__16,
            batch_norm__17,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_0,
                parameter_255,
                parameter_254,
                parameter_253,
                parameter_252,
                True,
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
        del (
            depthwise_conv2d_0,
            parameter_252,
            parameter_253,
            parameter_254,
            parameter_255,
        )

        # pd_op.relu6: (-1x48x112x112xf32) <- (-1x48x112x112xf32)
        relu6_2 = paddle._C_ops.relu6(batch_norm__12)
        del batch_norm__12

        # pd_op.conv2d: (-1x24x112x112xf32) <- (-1x48x112x112xf32, 24x48x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            relu6_2, parameter_251, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_251, relu6_2

        # pd_op.batch_norm_: (-1x24x112x112xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (-1x24x112x112xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        (
            batch_norm__18,
            batch_norm__19,
            batch_norm__20,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_2,
                parameter_250,
                parameter_249,
                parameter_248,
                parameter_247,
                True,
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
        del conv2d_2, parameter_247, parameter_248, parameter_249, parameter_250

        # pd_op.conv2d: (-1x144x112x112xf32) <- (-1x24x112x112xf32, 144x24x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            batch_norm__18, parameter_246, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del batch_norm__18, parameter_246

        # pd_op.batch_norm_: (-1x144x112x112xf32, 144xf32, 144xf32, 144xf32, 144xf32, -1xui8) <- (-1x144x112x112xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        (
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_3,
                parameter_245,
                parameter_244,
                parameter_243,
                parameter_242,
                True,
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
        del conv2d_3, parameter_242, parameter_243, parameter_244, parameter_245

        # pd_op.relu6: (-1x144x112x112xf32) <- (-1x144x112x112xf32)
        relu6_3 = paddle._C_ops.relu6(batch_norm__24)
        del batch_norm__24

        # pd_op.depthwise_conv2d: (-1x144x56x56xf32) <- (-1x144x112x112xf32, 144x1x3x3xf32)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(
            relu6_3, parameter_241, [2, 2], [1, 1], "EXPLICIT", 144, [1, 1], "NCHW"
        )
        del parameter_241, relu6_3

        # pd_op.batch_norm_: (-1x144x56x56xf32, 144xf32, 144xf32, 144xf32, 144xf32, -1xui8) <- (-1x144x56x56xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        (
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_1,
                parameter_240,
                parameter_239,
                parameter_238,
                parameter_237,
                True,
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
        del (
            depthwise_conv2d_1,
            parameter_237,
            parameter_238,
            parameter_239,
            parameter_240,
        )

        # pd_op.relu6: (-1x144x56x56xf32) <- (-1x144x56x56xf32)
        relu6_4 = paddle._C_ops.relu6(batch_norm__30)
        del batch_norm__30

        # pd_op.conv2d: (-1x36x56x56xf32) <- (-1x144x56x56xf32, 36x144x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            relu6_4, parameter_236, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_236, relu6_4

        # pd_op.batch_norm_: (-1x36x56x56xf32, 36xf32, 36xf32, 36xf32, 36xf32, -1xui8) <- (-1x36x56x56xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        (
            batch_norm__36,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_4,
                parameter_235,
                parameter_234,
                parameter_233,
                parameter_232,
                True,
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
        del conv2d_4, parameter_232, parameter_233, parameter_234, parameter_235

        # pd_op.conv2d: (-1x216x56x56xf32) <- (-1x36x56x56xf32, 216x36x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            batch_norm__36, parameter_231, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_231

        # pd_op.batch_norm_: (-1x216x56x56xf32, 216xf32, 216xf32, 216xf32, 216xf32, -1xui8) <- (-1x216x56x56xf32, 216xf32, 216xf32, 216xf32, 216xf32)
        (
            batch_norm__42,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_5,
                parameter_230,
                parameter_229,
                parameter_228,
                parameter_227,
                True,
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
        del conv2d_5, parameter_227, parameter_228, parameter_229, parameter_230

        # pd_op.relu6: (-1x216x56x56xf32) <- (-1x216x56x56xf32)
        relu6_5 = paddle._C_ops.relu6(batch_norm__42)
        del batch_norm__42

        # pd_op.depthwise_conv2d: (-1x216x56x56xf32) <- (-1x216x56x56xf32, 216x1x3x3xf32)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(
            relu6_5, parameter_226, [1, 1], [1, 1], "EXPLICIT", 216, [1, 1], "NCHW"
        )
        del parameter_226, relu6_5

        # pd_op.batch_norm_: (-1x216x56x56xf32, 216xf32, 216xf32, 216xf32, 216xf32, -1xui8) <- (-1x216x56x56xf32, 216xf32, 216xf32, 216xf32, 216xf32)
        (
            batch_norm__48,
            batch_norm__49,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_2,
                parameter_225,
                parameter_224,
                parameter_223,
                parameter_222,
                True,
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
        del (
            depthwise_conv2d_2,
            parameter_222,
            parameter_223,
            parameter_224,
            parameter_225,
        )

        # pd_op.relu6: (-1x216x56x56xf32) <- (-1x216x56x56xf32)
        relu6_6 = paddle._C_ops.relu6(batch_norm__48)
        del batch_norm__48

        # pd_op.conv2d: (-1x36x56x56xf32) <- (-1x216x56x56xf32, 36x216x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            relu6_6, parameter_221, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_221, relu6_6

        # pd_op.batch_norm_: (-1x36x56x56xf32, 36xf32, 36xf32, 36xf32, 36xf32, -1xui8) <- (-1x36x56x56xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        (
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_6,
                parameter_220,
                parameter_219,
                parameter_218,
                parameter_217,
                True,
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
        del conv2d_6, parameter_217, parameter_218, parameter_219, parameter_220

        # pd_op.add: (-1x36x56x56xf32) <- (-1x36x56x56xf32, -1x36x56x56xf32)
        add_1 = paddle._C_ops.add(batch_norm__36, batch_norm__54)
        del batch_norm__36, batch_norm__54

        # pd_op.conv2d: (-1x216x56x56xf32) <- (-1x36x56x56xf32, 216x36x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            add_1, parameter_216, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_1, parameter_216

        # pd_op.batch_norm_: (-1x216x56x56xf32, 216xf32, 216xf32, 216xf32, 216xf32, -1xui8) <- (-1x216x56x56xf32, 216xf32, 216xf32, 216xf32, 216xf32)
        (
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_7,
                parameter_215,
                parameter_214,
                parameter_213,
                parameter_212,
                True,
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
        del conv2d_7, parameter_212, parameter_213, parameter_214, parameter_215

        # pd_op.relu6: (-1x216x56x56xf32) <- (-1x216x56x56xf32)
        relu6_7 = paddle._C_ops.relu6(batch_norm__60)
        del batch_norm__60

        # pd_op.depthwise_conv2d: (-1x216x28x28xf32) <- (-1x216x56x56xf32, 216x1x3x3xf32)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(
            relu6_7, parameter_211, [2, 2], [1, 1], "EXPLICIT", 216, [1, 1], "NCHW"
        )
        del parameter_211, relu6_7

        # pd_op.batch_norm_: (-1x216x28x28xf32, 216xf32, 216xf32, 216xf32, 216xf32, -1xui8) <- (-1x216x28x28xf32, 216xf32, 216xf32, 216xf32, 216xf32)
        (
            batch_norm__66,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            batch_norm__70,
            batch_norm__71,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_3,
                parameter_210,
                parameter_209,
                parameter_208,
                parameter_207,
                True,
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
        del (
            depthwise_conv2d_3,
            parameter_207,
            parameter_208,
            parameter_209,
            parameter_210,
        )

        # pd_op.relu6: (-1x216x28x28xf32) <- (-1x216x28x28xf32)
        relu6_8 = paddle._C_ops.relu6(batch_norm__66)
        del batch_norm__66

        # pd_op.conv2d: (-1x48x28x28xf32) <- (-1x216x28x28xf32, 48x216x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            relu6_8, parameter_206, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_206, relu6_8

        # pd_op.batch_norm_: (-1x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (-1x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        (
            batch_norm__72,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_8,
                parameter_205,
                parameter_204,
                parameter_203,
                parameter_202,
                True,
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
        del conv2d_8, parameter_202, parameter_203, parameter_204, parameter_205

        # pd_op.conv2d: (-1x288x28x28xf32) <- (-1x48x28x28xf32, 288x48x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            batch_norm__72, parameter_201, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_201

        # pd_op.batch_norm_: (-1x288x28x28xf32, 288xf32, 288xf32, 288xf32, 288xf32, -1xui8) <- (-1x288x28x28xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        (
            batch_norm__78,
            batch_norm__79,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
            batch_norm__83,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_9,
                parameter_200,
                parameter_199,
                parameter_198,
                parameter_197,
                True,
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
        del conv2d_9, parameter_197, parameter_198, parameter_199, parameter_200

        # pd_op.relu6: (-1x288x28x28xf32) <- (-1x288x28x28xf32)
        relu6_9 = paddle._C_ops.relu6(batch_norm__78)
        del batch_norm__78

        # pd_op.depthwise_conv2d: (-1x288x28x28xf32) <- (-1x288x28x28xf32, 288x1x3x3xf32)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(
            relu6_9, parameter_196, [1, 1], [1, 1], "EXPLICIT", 288, [1, 1], "NCHW"
        )
        del parameter_196, relu6_9

        # pd_op.batch_norm_: (-1x288x28x28xf32, 288xf32, 288xf32, 288xf32, 288xf32, -1xui8) <- (-1x288x28x28xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        (
            batch_norm__84,
            batch_norm__85,
            batch_norm__86,
            batch_norm__87,
            batch_norm__88,
            batch_norm__89,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_4,
                parameter_195,
                parameter_194,
                parameter_193,
                parameter_192,
                True,
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
        del (
            depthwise_conv2d_4,
            parameter_192,
            parameter_193,
            parameter_194,
            parameter_195,
        )

        # pd_op.relu6: (-1x288x28x28xf32) <- (-1x288x28x28xf32)
        relu6_10 = paddle._C_ops.relu6(batch_norm__84)
        del batch_norm__84

        # pd_op.conv2d: (-1x48x28x28xf32) <- (-1x288x28x28xf32, 48x288x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            relu6_10, parameter_191, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_191, relu6_10

        # pd_op.batch_norm_: (-1x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (-1x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        (
            batch_norm__90,
            batch_norm__91,
            batch_norm__92,
            batch_norm__93,
            batch_norm__94,
            batch_norm__95,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_10,
                parameter_190,
                parameter_189,
                parameter_188,
                parameter_187,
                True,
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
        del conv2d_10, parameter_187, parameter_188, parameter_189, parameter_190

        # pd_op.add: (-1x48x28x28xf32) <- (-1x48x28x28xf32, -1x48x28x28xf32)
        add_2 = paddle._C_ops.add(batch_norm__72, batch_norm__90)
        del batch_norm__72, batch_norm__90

        # pd_op.conv2d: (-1x288x28x28xf32) <- (-1x48x28x28xf32, 288x48x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            add_2, parameter_186, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_186

        # pd_op.batch_norm_: (-1x288x28x28xf32, 288xf32, 288xf32, 288xf32, 288xf32, -1xui8) <- (-1x288x28x28xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        (
            batch_norm__96,
            batch_norm__97,
            batch_norm__98,
            batch_norm__99,
            batch_norm__100,
            batch_norm__101,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_11,
                parameter_185,
                parameter_184,
                parameter_183,
                parameter_182,
                True,
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
        del conv2d_11, parameter_182, parameter_183, parameter_184, parameter_185

        # pd_op.relu6: (-1x288x28x28xf32) <- (-1x288x28x28xf32)
        relu6_11 = paddle._C_ops.relu6(batch_norm__96)
        del batch_norm__96

        # pd_op.depthwise_conv2d: (-1x288x28x28xf32) <- (-1x288x28x28xf32, 288x1x3x3xf32)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(
            relu6_11, parameter_181, [1, 1], [1, 1], "EXPLICIT", 288, [1, 1], "NCHW"
        )
        del parameter_181, relu6_11

        # pd_op.batch_norm_: (-1x288x28x28xf32, 288xf32, 288xf32, 288xf32, 288xf32, -1xui8) <- (-1x288x28x28xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        (
            batch_norm__102,
            batch_norm__103,
            batch_norm__104,
            batch_norm__105,
            batch_norm__106,
            batch_norm__107,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_5,
                parameter_180,
                parameter_179,
                parameter_178,
                parameter_177,
                True,
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
        del (
            depthwise_conv2d_5,
            parameter_177,
            parameter_178,
            parameter_179,
            parameter_180,
        )

        # pd_op.relu6: (-1x288x28x28xf32) <- (-1x288x28x28xf32)
        relu6_12 = paddle._C_ops.relu6(batch_norm__102)
        del batch_norm__102

        # pd_op.conv2d: (-1x48x28x28xf32) <- (-1x288x28x28xf32, 48x288x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            relu6_12, parameter_176, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_176, relu6_12

        # pd_op.batch_norm_: (-1x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (-1x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        (
            batch_norm__108,
            batch_norm__109,
            batch_norm__110,
            batch_norm__111,
            batch_norm__112,
            batch_norm__113,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_12,
                parameter_175,
                parameter_174,
                parameter_173,
                parameter_172,
                True,
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
        del conv2d_12, parameter_172, parameter_173, parameter_174, parameter_175

        # pd_op.add: (-1x48x28x28xf32) <- (-1x48x28x28xf32, -1x48x28x28xf32)
        add_3 = paddle._C_ops.add(add_2, batch_norm__108)
        del add_2, batch_norm__108

        # pd_op.conv2d: (-1x288x28x28xf32) <- (-1x48x28x28xf32, 288x48x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            add_3, parameter_171, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_3, parameter_171

        # pd_op.batch_norm_: (-1x288x28x28xf32, 288xf32, 288xf32, 288xf32, 288xf32, -1xui8) <- (-1x288x28x28xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        (
            batch_norm__114,
            batch_norm__115,
            batch_norm__116,
            batch_norm__117,
            batch_norm__118,
            batch_norm__119,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_13,
                parameter_170,
                parameter_169,
                parameter_168,
                parameter_167,
                True,
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
        del conv2d_13, parameter_167, parameter_168, parameter_169, parameter_170

        # pd_op.relu6: (-1x288x28x28xf32) <- (-1x288x28x28xf32)
        relu6_13 = paddle._C_ops.relu6(batch_norm__114)
        del batch_norm__114

        # pd_op.depthwise_conv2d: (-1x288x14x14xf32) <- (-1x288x28x28xf32, 288x1x3x3xf32)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(
            relu6_13, parameter_166, [2, 2], [1, 1], "EXPLICIT", 288, [1, 1], "NCHW"
        )
        del parameter_166, relu6_13

        # pd_op.batch_norm_: (-1x288x14x14xf32, 288xf32, 288xf32, 288xf32, 288xf32, -1xui8) <- (-1x288x14x14xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        (
            batch_norm__120,
            batch_norm__121,
            batch_norm__122,
            batch_norm__123,
            batch_norm__124,
            batch_norm__125,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_6,
                parameter_165,
                parameter_164,
                parameter_163,
                parameter_162,
                True,
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
        del (
            depthwise_conv2d_6,
            parameter_162,
            parameter_163,
            parameter_164,
            parameter_165,
        )

        # pd_op.relu6: (-1x288x14x14xf32) <- (-1x288x14x14xf32)
        relu6_14 = paddle._C_ops.relu6(batch_norm__120)
        del batch_norm__120

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x288x14x14xf32, 96x288x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            relu6_14, parameter_161, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_161, relu6_14

        # pd_op.batch_norm_: (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__126,
            batch_norm__127,
            batch_norm__128,
            batch_norm__129,
            batch_norm__130,
            batch_norm__131,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_14,
                parameter_160,
                parameter_159,
                parameter_158,
                parameter_157,
                True,
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
        del conv2d_14, parameter_157, parameter_158, parameter_159, parameter_160

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x96x14x14xf32, 576x96x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            batch_norm__126,
            parameter_156,
            [1, 1],
            [0, 0],
            "EXPLICIT",
            [1, 1],
            1,
            "NCHW",
        )
        del parameter_156

        # pd_op.batch_norm_: (-1x576x14x14xf32, 576xf32, 576xf32, 576xf32, 576xf32, -1xui8) <- (-1x576x14x14xf32, 576xf32, 576xf32, 576xf32, 576xf32)
        (
            batch_norm__132,
            batch_norm__133,
            batch_norm__134,
            batch_norm__135,
            batch_norm__136,
            batch_norm__137,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_15,
                parameter_155,
                parameter_154,
                parameter_153,
                parameter_152,
                True,
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
        del conv2d_15, parameter_152, parameter_153, parameter_154, parameter_155

        # pd_op.relu6: (-1x576x14x14xf32) <- (-1x576x14x14xf32)
        relu6_15 = paddle._C_ops.relu6(batch_norm__132)
        del batch_norm__132

        # pd_op.depthwise_conv2d: (-1x576x14x14xf32) <- (-1x576x14x14xf32, 576x1x3x3xf32)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(
            relu6_15, parameter_151, [1, 1], [1, 1], "EXPLICIT", 576, [1, 1], "NCHW"
        )
        del parameter_151, relu6_15

        # pd_op.batch_norm_: (-1x576x14x14xf32, 576xf32, 576xf32, 576xf32, 576xf32, -1xui8) <- (-1x576x14x14xf32, 576xf32, 576xf32, 576xf32, 576xf32)
        (
            batch_norm__138,
            batch_norm__139,
            batch_norm__140,
            batch_norm__141,
            batch_norm__142,
            batch_norm__143,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_7,
                parameter_150,
                parameter_149,
                parameter_148,
                parameter_147,
                True,
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
        del (
            depthwise_conv2d_7,
            parameter_147,
            parameter_148,
            parameter_149,
            parameter_150,
        )

        # pd_op.relu6: (-1x576x14x14xf32) <- (-1x576x14x14xf32)
        relu6_16 = paddle._C_ops.relu6(batch_norm__138)
        del batch_norm__138

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x576x14x14xf32, 96x576x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            relu6_16, parameter_146, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_146, relu6_16

        # pd_op.batch_norm_: (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__144,
            batch_norm__145,
            batch_norm__146,
            batch_norm__147,
            batch_norm__148,
            batch_norm__149,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_16,
                parameter_145,
                parameter_144,
                parameter_143,
                parameter_142,
                True,
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
        del conv2d_16, parameter_142, parameter_143, parameter_144, parameter_145

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, -1x96x14x14xf32)
        add_4 = paddle._C_ops.add(batch_norm__126, batch_norm__144)
        del batch_norm__126, batch_norm__144

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x96x14x14xf32, 576x96x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            add_4, parameter_141, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_141

        # pd_op.batch_norm_: (-1x576x14x14xf32, 576xf32, 576xf32, 576xf32, 576xf32, -1xui8) <- (-1x576x14x14xf32, 576xf32, 576xf32, 576xf32, 576xf32)
        (
            batch_norm__150,
            batch_norm__151,
            batch_norm__152,
            batch_norm__153,
            batch_norm__154,
            batch_norm__155,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_17,
                parameter_140,
                parameter_139,
                parameter_138,
                parameter_137,
                True,
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
        del conv2d_17, parameter_137, parameter_138, parameter_139, parameter_140

        # pd_op.relu6: (-1x576x14x14xf32) <- (-1x576x14x14xf32)
        relu6_17 = paddle._C_ops.relu6(batch_norm__150)
        del batch_norm__150

        # pd_op.depthwise_conv2d: (-1x576x14x14xf32) <- (-1x576x14x14xf32, 576x1x3x3xf32)
        depthwise_conv2d_8 = paddle._C_ops.depthwise_conv2d(
            relu6_17, parameter_136, [1, 1], [1, 1], "EXPLICIT", 576, [1, 1], "NCHW"
        )
        del parameter_136, relu6_17

        # pd_op.batch_norm_: (-1x576x14x14xf32, 576xf32, 576xf32, 576xf32, 576xf32, -1xui8) <- (-1x576x14x14xf32, 576xf32, 576xf32, 576xf32, 576xf32)
        (
            batch_norm__156,
            batch_norm__157,
            batch_norm__158,
            batch_norm__159,
            batch_norm__160,
            batch_norm__161,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_8,
                parameter_135,
                parameter_134,
                parameter_133,
                parameter_132,
                True,
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
        del (
            depthwise_conv2d_8,
            parameter_132,
            parameter_133,
            parameter_134,
            parameter_135,
        )

        # pd_op.relu6: (-1x576x14x14xf32) <- (-1x576x14x14xf32)
        relu6_18 = paddle._C_ops.relu6(batch_norm__156)
        del batch_norm__156

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x576x14x14xf32, 96x576x1x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            relu6_18, parameter_131, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_131, relu6_18

        # pd_op.batch_norm_: (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__162,
            batch_norm__163,
            batch_norm__164,
            batch_norm__165,
            batch_norm__166,
            batch_norm__167,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_18,
                parameter_130,
                parameter_129,
                parameter_128,
                parameter_127,
                True,
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
        del conv2d_18, parameter_127, parameter_128, parameter_129, parameter_130

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, -1x96x14x14xf32)
        add_5 = paddle._C_ops.add(add_4, batch_norm__162)
        del add_4, batch_norm__162

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x96x14x14xf32, 576x96x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            add_5, parameter_126, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_126

        # pd_op.batch_norm_: (-1x576x14x14xf32, 576xf32, 576xf32, 576xf32, 576xf32, -1xui8) <- (-1x576x14x14xf32, 576xf32, 576xf32, 576xf32, 576xf32)
        (
            batch_norm__168,
            batch_norm__169,
            batch_norm__170,
            batch_norm__171,
            batch_norm__172,
            batch_norm__173,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_19,
                parameter_125,
                parameter_124,
                parameter_123,
                parameter_122,
                True,
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
        del conv2d_19, parameter_122, parameter_123, parameter_124, parameter_125

        # pd_op.relu6: (-1x576x14x14xf32) <- (-1x576x14x14xf32)
        relu6_19 = paddle._C_ops.relu6(batch_norm__168)
        del batch_norm__168

        # pd_op.depthwise_conv2d: (-1x576x14x14xf32) <- (-1x576x14x14xf32, 576x1x3x3xf32)
        depthwise_conv2d_9 = paddle._C_ops.depthwise_conv2d(
            relu6_19, parameter_121, [1, 1], [1, 1], "EXPLICIT", 576, [1, 1], "NCHW"
        )
        del parameter_121, relu6_19

        # pd_op.batch_norm_: (-1x576x14x14xf32, 576xf32, 576xf32, 576xf32, 576xf32, -1xui8) <- (-1x576x14x14xf32, 576xf32, 576xf32, 576xf32, 576xf32)
        (
            batch_norm__174,
            batch_norm__175,
            batch_norm__176,
            batch_norm__177,
            batch_norm__178,
            batch_norm__179,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_9,
                parameter_120,
                parameter_119,
                parameter_118,
                parameter_117,
                True,
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
        del (
            depthwise_conv2d_9,
            parameter_117,
            parameter_118,
            parameter_119,
            parameter_120,
        )

        # pd_op.relu6: (-1x576x14x14xf32) <- (-1x576x14x14xf32)
        relu6_20 = paddle._C_ops.relu6(batch_norm__174)
        del batch_norm__174

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x576x14x14xf32, 96x576x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            relu6_20, parameter_116, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_116, relu6_20

        # pd_op.batch_norm_: (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__180,
            batch_norm__181,
            batch_norm__182,
            batch_norm__183,
            batch_norm__184,
            batch_norm__185,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_20,
                parameter_115,
                parameter_114,
                parameter_113,
                parameter_112,
                True,
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
        del conv2d_20, parameter_112, parameter_113, parameter_114, parameter_115

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, -1x96x14x14xf32)
        add_6 = paddle._C_ops.add(add_5, batch_norm__180)
        del add_5, batch_norm__180

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x96x14x14xf32, 576x96x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            add_6, parameter_111, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_6, parameter_111

        # pd_op.batch_norm_: (-1x576x14x14xf32, 576xf32, 576xf32, 576xf32, 576xf32, -1xui8) <- (-1x576x14x14xf32, 576xf32, 576xf32, 576xf32, 576xf32)
        (
            batch_norm__186,
            batch_norm__187,
            batch_norm__188,
            batch_norm__189,
            batch_norm__190,
            batch_norm__191,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_21,
                parameter_110,
                parameter_109,
                parameter_108,
                parameter_107,
                True,
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
        del conv2d_21, parameter_107, parameter_108, parameter_109, parameter_110

        # pd_op.relu6: (-1x576x14x14xf32) <- (-1x576x14x14xf32)
        relu6_21 = paddle._C_ops.relu6(batch_norm__186)
        del batch_norm__186

        # pd_op.depthwise_conv2d: (-1x576x14x14xf32) <- (-1x576x14x14xf32, 576x1x3x3xf32)
        depthwise_conv2d_10 = paddle._C_ops.depthwise_conv2d(
            relu6_21, parameter_106, [1, 1], [1, 1], "EXPLICIT", 576, [1, 1], "NCHW"
        )
        del parameter_106, relu6_21

        # pd_op.batch_norm_: (-1x576x14x14xf32, 576xf32, 576xf32, 576xf32, 576xf32, -1xui8) <- (-1x576x14x14xf32, 576xf32, 576xf32, 576xf32, 576xf32)
        (
            batch_norm__192,
            batch_norm__193,
            batch_norm__194,
            batch_norm__195,
            batch_norm__196,
            batch_norm__197,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_10,
                parameter_105,
                parameter_104,
                parameter_103,
                parameter_102,
                True,
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
        del (
            depthwise_conv2d_10,
            parameter_102,
            parameter_103,
            parameter_104,
            parameter_105,
        )

        # pd_op.relu6: (-1x576x14x14xf32) <- (-1x576x14x14xf32)
        relu6_22 = paddle._C_ops.relu6(batch_norm__192)
        del batch_norm__192

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x576x14x14xf32, 144x576x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(
            relu6_22, parameter_101, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_101, relu6_22

        # pd_op.batch_norm_: (-1x144x14x14xf32, 144xf32, 144xf32, 144xf32, 144xf32, -1xui8) <- (-1x144x14x14xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        (
            batch_norm__198,
            batch_norm__199,
            batch_norm__200,
            batch_norm__201,
            batch_norm__202,
            batch_norm__203,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_22,
                parameter_100,
                parameter_99,
                parameter_98,
                parameter_97,
                True,
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
        del conv2d_22, parameter_100, parameter_97, parameter_98, parameter_99

        # pd_op.conv2d: (-1x864x14x14xf32) <- (-1x144x14x14xf32, 864x144x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(
            batch_norm__198, parameter_96, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_96

        # pd_op.batch_norm_: (-1x864x14x14xf32, 864xf32, 864xf32, 864xf32, 864xf32, -1xui8) <- (-1x864x14x14xf32, 864xf32, 864xf32, 864xf32, 864xf32)
        (
            batch_norm__204,
            batch_norm__205,
            batch_norm__206,
            batch_norm__207,
            batch_norm__208,
            batch_norm__209,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_23,
                parameter_95,
                parameter_94,
                parameter_93,
                parameter_92,
                True,
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
        del conv2d_23, parameter_92, parameter_93, parameter_94, parameter_95

        # pd_op.relu6: (-1x864x14x14xf32) <- (-1x864x14x14xf32)
        relu6_23 = paddle._C_ops.relu6(batch_norm__204)
        del batch_norm__204

        # pd_op.depthwise_conv2d: (-1x864x14x14xf32) <- (-1x864x14x14xf32, 864x1x3x3xf32)
        depthwise_conv2d_11 = paddle._C_ops.depthwise_conv2d(
            relu6_23, parameter_91, [1, 1], [1, 1], "EXPLICIT", 864, [1, 1], "NCHW"
        )
        del parameter_91, relu6_23

        # pd_op.batch_norm_: (-1x864x14x14xf32, 864xf32, 864xf32, 864xf32, 864xf32, -1xui8) <- (-1x864x14x14xf32, 864xf32, 864xf32, 864xf32, 864xf32)
        (
            batch_norm__210,
            batch_norm__211,
            batch_norm__212,
            batch_norm__213,
            batch_norm__214,
            batch_norm__215,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_11,
                parameter_90,
                parameter_89,
                parameter_88,
                parameter_87,
                True,
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
        del depthwise_conv2d_11, parameter_87, parameter_88, parameter_89, parameter_90

        # pd_op.relu6: (-1x864x14x14xf32) <- (-1x864x14x14xf32)
        relu6_24 = paddle._C_ops.relu6(batch_norm__210)
        del batch_norm__210

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x864x14x14xf32, 144x864x1x1xf32)
        conv2d_24 = paddle._C_ops.conv2d(
            relu6_24, parameter_86, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_86, relu6_24

        # pd_op.batch_norm_: (-1x144x14x14xf32, 144xf32, 144xf32, 144xf32, 144xf32, -1xui8) <- (-1x144x14x14xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        (
            batch_norm__216,
            batch_norm__217,
            batch_norm__218,
            batch_norm__219,
            batch_norm__220,
            batch_norm__221,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_24,
                parameter_85,
                parameter_84,
                parameter_83,
                parameter_82,
                True,
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
        del conv2d_24, parameter_82, parameter_83, parameter_84, parameter_85

        # pd_op.add: (-1x144x14x14xf32) <- (-1x144x14x14xf32, -1x144x14x14xf32)
        add_7 = paddle._C_ops.add(batch_norm__198, batch_norm__216)
        del batch_norm__198, batch_norm__216

        # pd_op.conv2d: (-1x864x14x14xf32) <- (-1x144x14x14xf32, 864x144x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(
            add_7, parameter_81, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_81

        # pd_op.batch_norm_: (-1x864x14x14xf32, 864xf32, 864xf32, 864xf32, 864xf32, -1xui8) <- (-1x864x14x14xf32, 864xf32, 864xf32, 864xf32, 864xf32)
        (
            batch_norm__222,
            batch_norm__223,
            batch_norm__224,
            batch_norm__225,
            batch_norm__226,
            batch_norm__227,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_25,
                parameter_80,
                parameter_79,
                parameter_78,
                parameter_77,
                True,
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
        del conv2d_25, parameter_77, parameter_78, parameter_79, parameter_80

        # pd_op.relu6: (-1x864x14x14xf32) <- (-1x864x14x14xf32)
        relu6_25 = paddle._C_ops.relu6(batch_norm__222)
        del batch_norm__222

        # pd_op.depthwise_conv2d: (-1x864x14x14xf32) <- (-1x864x14x14xf32, 864x1x3x3xf32)
        depthwise_conv2d_12 = paddle._C_ops.depthwise_conv2d(
            relu6_25, parameter_76, [1, 1], [1, 1], "EXPLICIT", 864, [1, 1], "NCHW"
        )
        del parameter_76, relu6_25

        # pd_op.batch_norm_: (-1x864x14x14xf32, 864xf32, 864xf32, 864xf32, 864xf32, -1xui8) <- (-1x864x14x14xf32, 864xf32, 864xf32, 864xf32, 864xf32)
        (
            batch_norm__228,
            batch_norm__229,
            batch_norm__230,
            batch_norm__231,
            batch_norm__232,
            batch_norm__233,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_12,
                parameter_75,
                parameter_74,
                parameter_73,
                parameter_72,
                True,
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
        del depthwise_conv2d_12, parameter_72, parameter_73, parameter_74, parameter_75

        # pd_op.relu6: (-1x864x14x14xf32) <- (-1x864x14x14xf32)
        relu6_26 = paddle._C_ops.relu6(batch_norm__228)
        del batch_norm__228

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x864x14x14xf32, 144x864x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(
            relu6_26, parameter_71, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_71, relu6_26

        # pd_op.batch_norm_: (-1x144x14x14xf32, 144xf32, 144xf32, 144xf32, 144xf32, -1xui8) <- (-1x144x14x14xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        (
            batch_norm__234,
            batch_norm__235,
            batch_norm__236,
            batch_norm__237,
            batch_norm__238,
            batch_norm__239,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_26,
                parameter_70,
                parameter_69,
                parameter_68,
                parameter_67,
                True,
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
        del conv2d_26, parameter_67, parameter_68, parameter_69, parameter_70

        # pd_op.add: (-1x144x14x14xf32) <- (-1x144x14x14xf32, -1x144x14x14xf32)
        add_8 = paddle._C_ops.add(add_7, batch_norm__234)
        del add_7, batch_norm__234

        # pd_op.conv2d: (-1x864x14x14xf32) <- (-1x144x14x14xf32, 864x144x1x1xf32)
        conv2d_27 = paddle._C_ops.conv2d(
            add_8, parameter_66, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_8, parameter_66

        # pd_op.batch_norm_: (-1x864x14x14xf32, 864xf32, 864xf32, 864xf32, 864xf32, -1xui8) <- (-1x864x14x14xf32, 864xf32, 864xf32, 864xf32, 864xf32)
        (
            batch_norm__240,
            batch_norm__241,
            batch_norm__242,
            batch_norm__243,
            batch_norm__244,
            batch_norm__245,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_27,
                parameter_65,
                parameter_64,
                parameter_63,
                parameter_62,
                True,
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
        del conv2d_27, parameter_62, parameter_63, parameter_64, parameter_65

        # pd_op.relu6: (-1x864x14x14xf32) <- (-1x864x14x14xf32)
        relu6_27 = paddle._C_ops.relu6(batch_norm__240)
        del batch_norm__240

        # pd_op.depthwise_conv2d: (-1x864x7x7xf32) <- (-1x864x14x14xf32, 864x1x3x3xf32)
        depthwise_conv2d_13 = paddle._C_ops.depthwise_conv2d(
            relu6_27, parameter_61, [2, 2], [1, 1], "EXPLICIT", 864, [1, 1], "NCHW"
        )
        del parameter_61, relu6_27

        # pd_op.batch_norm_: (-1x864x7x7xf32, 864xf32, 864xf32, 864xf32, 864xf32, -1xui8) <- (-1x864x7x7xf32, 864xf32, 864xf32, 864xf32, 864xf32)
        (
            batch_norm__246,
            batch_norm__247,
            batch_norm__248,
            batch_norm__249,
            batch_norm__250,
            batch_norm__251,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_13,
                parameter_60,
                parameter_59,
                parameter_58,
                parameter_57,
                True,
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
        del depthwise_conv2d_13, parameter_57, parameter_58, parameter_59, parameter_60

        # pd_op.relu6: (-1x864x7x7xf32) <- (-1x864x7x7xf32)
        relu6_28 = paddle._C_ops.relu6(batch_norm__246)
        del batch_norm__246

        # pd_op.conv2d: (-1x240x7x7xf32) <- (-1x864x7x7xf32, 240x864x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(
            relu6_28, parameter_56, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_56, relu6_28

        # pd_op.batch_norm_: (-1x240x7x7xf32, 240xf32, 240xf32, 240xf32, 240xf32, -1xui8) <- (-1x240x7x7xf32, 240xf32, 240xf32, 240xf32, 240xf32)
        (
            batch_norm__252,
            batch_norm__253,
            batch_norm__254,
            batch_norm__255,
            batch_norm__256,
            batch_norm__257,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_28,
                parameter_55,
                parameter_54,
                parameter_53,
                parameter_52,
                True,
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
        del conv2d_28, parameter_52, parameter_53, parameter_54, parameter_55

        # pd_op.conv2d: (-1x1440x7x7xf32) <- (-1x240x7x7xf32, 1440x240x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(
            batch_norm__252, parameter_51, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_51

        # pd_op.batch_norm_: (-1x1440x7x7xf32, 1440xf32, 1440xf32, 1440xf32, 1440xf32, -1xui8) <- (-1x1440x7x7xf32, 1440xf32, 1440xf32, 1440xf32, 1440xf32)
        (
            batch_norm__258,
            batch_norm__259,
            batch_norm__260,
            batch_norm__261,
            batch_norm__262,
            batch_norm__263,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_29,
                parameter_50,
                parameter_49,
                parameter_48,
                parameter_47,
                True,
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
        del conv2d_29, parameter_47, parameter_48, parameter_49, parameter_50

        # pd_op.relu6: (-1x1440x7x7xf32) <- (-1x1440x7x7xf32)
        relu6_29 = paddle._C_ops.relu6(batch_norm__258)
        del batch_norm__258

        # pd_op.depthwise_conv2d: (-1x1440x7x7xf32) <- (-1x1440x7x7xf32, 1440x1x3x3xf32)
        depthwise_conv2d_14 = paddle._C_ops.depthwise_conv2d(
            relu6_29, parameter_46, [1, 1], [1, 1], "EXPLICIT", 1440, [1, 1], "NCHW"
        )
        del parameter_46, relu6_29

        # pd_op.batch_norm_: (-1x1440x7x7xf32, 1440xf32, 1440xf32, 1440xf32, 1440xf32, -1xui8) <- (-1x1440x7x7xf32, 1440xf32, 1440xf32, 1440xf32, 1440xf32)
        (
            batch_norm__264,
            batch_norm__265,
            batch_norm__266,
            batch_norm__267,
            batch_norm__268,
            batch_norm__269,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_14,
                parameter_45,
                parameter_44,
                parameter_43,
                parameter_42,
                True,
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
        del depthwise_conv2d_14, parameter_42, parameter_43, parameter_44, parameter_45

        # pd_op.relu6: (-1x1440x7x7xf32) <- (-1x1440x7x7xf32)
        relu6_30 = paddle._C_ops.relu6(batch_norm__264)
        del batch_norm__264

        # pd_op.conv2d: (-1x240x7x7xf32) <- (-1x1440x7x7xf32, 240x1440x1x1xf32)
        conv2d_30 = paddle._C_ops.conv2d(
            relu6_30, parameter_41, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_41, relu6_30

        # pd_op.batch_norm_: (-1x240x7x7xf32, 240xf32, 240xf32, 240xf32, 240xf32, -1xui8) <- (-1x240x7x7xf32, 240xf32, 240xf32, 240xf32, 240xf32)
        (
            batch_norm__270,
            batch_norm__271,
            batch_norm__272,
            batch_norm__273,
            batch_norm__274,
            batch_norm__275,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_30,
                parameter_40,
                parameter_39,
                parameter_38,
                parameter_37,
                True,
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
        del conv2d_30, parameter_37, parameter_38, parameter_39, parameter_40

        # pd_op.add: (-1x240x7x7xf32) <- (-1x240x7x7xf32, -1x240x7x7xf32)
        add_9 = paddle._C_ops.add(batch_norm__252, batch_norm__270)
        del batch_norm__252, batch_norm__270

        # pd_op.conv2d: (-1x1440x7x7xf32) <- (-1x240x7x7xf32, 1440x240x1x1xf32)
        conv2d_31 = paddle._C_ops.conv2d(
            add_9, parameter_36, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_36

        # pd_op.batch_norm_: (-1x1440x7x7xf32, 1440xf32, 1440xf32, 1440xf32, 1440xf32, -1xui8) <- (-1x1440x7x7xf32, 1440xf32, 1440xf32, 1440xf32, 1440xf32)
        (
            batch_norm__276,
            batch_norm__277,
            batch_norm__278,
            batch_norm__279,
            batch_norm__280,
            batch_norm__281,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_31,
                parameter_35,
                parameter_34,
                parameter_33,
                parameter_32,
                True,
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
        del conv2d_31, parameter_32, parameter_33, parameter_34, parameter_35

        # pd_op.relu6: (-1x1440x7x7xf32) <- (-1x1440x7x7xf32)
        relu6_31 = paddle._C_ops.relu6(batch_norm__276)
        del batch_norm__276

        # pd_op.depthwise_conv2d: (-1x1440x7x7xf32) <- (-1x1440x7x7xf32, 1440x1x3x3xf32)
        depthwise_conv2d_15 = paddle._C_ops.depthwise_conv2d(
            relu6_31, parameter_31, [1, 1], [1, 1], "EXPLICIT", 1440, [1, 1], "NCHW"
        )
        del parameter_31, relu6_31

        # pd_op.batch_norm_: (-1x1440x7x7xf32, 1440xf32, 1440xf32, 1440xf32, 1440xf32, -1xui8) <- (-1x1440x7x7xf32, 1440xf32, 1440xf32, 1440xf32, 1440xf32)
        (
            batch_norm__282,
            batch_norm__283,
            batch_norm__284,
            batch_norm__285,
            batch_norm__286,
            batch_norm__287,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_15,
                parameter_30,
                parameter_29,
                parameter_28,
                parameter_27,
                True,
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
        del depthwise_conv2d_15, parameter_27, parameter_28, parameter_29, parameter_30

        # pd_op.relu6: (-1x1440x7x7xf32) <- (-1x1440x7x7xf32)
        relu6_32 = paddle._C_ops.relu6(batch_norm__282)
        del batch_norm__282

        # pd_op.conv2d: (-1x240x7x7xf32) <- (-1x1440x7x7xf32, 240x1440x1x1xf32)
        conv2d_32 = paddle._C_ops.conv2d(
            relu6_32, parameter_26, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_26, relu6_32

        # pd_op.batch_norm_: (-1x240x7x7xf32, 240xf32, 240xf32, 240xf32, 240xf32, -1xui8) <- (-1x240x7x7xf32, 240xf32, 240xf32, 240xf32, 240xf32)
        (
            batch_norm__288,
            batch_norm__289,
            batch_norm__290,
            batch_norm__291,
            batch_norm__292,
            batch_norm__293,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_32,
                parameter_25,
                parameter_24,
                parameter_23,
                parameter_22,
                True,
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
        del conv2d_32, parameter_22, parameter_23, parameter_24, parameter_25

        # pd_op.add: (-1x240x7x7xf32) <- (-1x240x7x7xf32, -1x240x7x7xf32)
        add_10 = paddle._C_ops.add(add_9, batch_norm__288)
        del add_9, batch_norm__288

        # pd_op.conv2d: (-1x1440x7x7xf32) <- (-1x240x7x7xf32, 1440x240x1x1xf32)
        conv2d_33 = paddle._C_ops.conv2d(
            add_10, parameter_21, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_10, parameter_21

        # pd_op.batch_norm_: (-1x1440x7x7xf32, 1440xf32, 1440xf32, 1440xf32, 1440xf32, -1xui8) <- (-1x1440x7x7xf32, 1440xf32, 1440xf32, 1440xf32, 1440xf32)
        (
            batch_norm__294,
            batch_norm__295,
            batch_norm__296,
            batch_norm__297,
            batch_norm__298,
            batch_norm__299,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_33,
                parameter_20,
                parameter_19,
                parameter_18,
                parameter_17,
                True,
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
        del conv2d_33, parameter_17, parameter_18, parameter_19, parameter_20

        # pd_op.relu6: (-1x1440x7x7xf32) <- (-1x1440x7x7xf32)
        relu6_33 = paddle._C_ops.relu6(batch_norm__294)
        del batch_norm__294

        # pd_op.depthwise_conv2d: (-1x1440x7x7xf32) <- (-1x1440x7x7xf32, 1440x1x3x3xf32)
        depthwise_conv2d_16 = paddle._C_ops.depthwise_conv2d(
            relu6_33, parameter_16, [1, 1], [1, 1], "EXPLICIT", 1440, [1, 1], "NCHW"
        )
        del parameter_16, relu6_33

        # pd_op.batch_norm_: (-1x1440x7x7xf32, 1440xf32, 1440xf32, 1440xf32, 1440xf32, -1xui8) <- (-1x1440x7x7xf32, 1440xf32, 1440xf32, 1440xf32, 1440xf32)
        (
            batch_norm__300,
            batch_norm__301,
            batch_norm__302,
            batch_norm__303,
            batch_norm__304,
            batch_norm__305,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_16,
                parameter_15,
                parameter_14,
                parameter_13,
                parameter_12,
                True,
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
        del depthwise_conv2d_16, parameter_12, parameter_13, parameter_14, parameter_15

        # pd_op.relu6: (-1x1440x7x7xf32) <- (-1x1440x7x7xf32)
        relu6_34 = paddle._C_ops.relu6(batch_norm__300)
        del batch_norm__300

        # pd_op.conv2d: (-1x480x7x7xf32) <- (-1x1440x7x7xf32, 480x1440x1x1xf32)
        conv2d_34 = paddle._C_ops.conv2d(
            relu6_34, parameter_11, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_11, relu6_34

        # pd_op.batch_norm_: (-1x480x7x7xf32, 480xf32, 480xf32, 480xf32, 480xf32, -1xui8) <- (-1x480x7x7xf32, 480xf32, 480xf32, 480xf32, 480xf32)
        (
            batch_norm__306,
            batch_norm__307,
            batch_norm__308,
            batch_norm__309,
            batch_norm__310,
            batch_norm__311,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_34,
                parameter_10,
                parameter_9,
                parameter_8,
                parameter_7,
                True,
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
        del conv2d_34, parameter_10, parameter_7, parameter_8, parameter_9

        # pd_op.conv2d: (-1x1920x7x7xf32) <- (-1x480x7x7xf32, 1920x480x1x1xf32)
        conv2d_35 = paddle._C_ops.conv2d(
            batch_norm__306, parameter_6, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del batch_norm__306, parameter_6

        # pd_op.batch_norm_: (-1x1920x7x7xf32, 1920xf32, 1920xf32, 1920xf32, 1920xf32, -1xui8) <- (-1x1920x7x7xf32, 1920xf32, 1920xf32, 1920xf32, 1920xf32)
        (
            batch_norm__312,
            batch_norm__313,
            batch_norm__314,
            batch_norm__315,
            batch_norm__316,
            batch_norm__317,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_35,
                parameter_5,
                parameter_4,
                parameter_3,
                parameter_2,
                True,
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
        del conv2d_35, parameter_2, parameter_3, parameter_4, parameter_5

        # pd_op.relu6: (-1x1920x7x7xf32) <- (-1x1920x7x7xf32)
        relu6_35 = paddle._C_ops.relu6(batch_norm__312)
        del batch_norm__312

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.pool2d: (-1x1920x1x1xf32) <- (-1x1920x7x7xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            relu6_35,
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
        del full_int_array_0, relu6_35

        # pd_op.flatten: (-1x1920xf32) <- (-1x1920x1x1xf32)
        flatten_0 = paddle._C_ops.flatten(pool2d_0, 1, 3)
        del pool2d_0

        # pd_op.matmul: (-1x102xf32) <- (-1x1920xf32, 1920x102xf32)
        matmul_0 = paddle._C_ops.matmul(flatten_0, parameter_1, False, False)
        del flatten_0, parameter_1

        # pd_op.add: (-1x102xf32) <- (-1x102xf32, 102xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_0)
        del matmul_0, parameter_0

        return add_0
