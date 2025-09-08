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
        data_0,
    ):
        # pd_op.conv2d: (-1x32x-1x-1xf32) <- (-1x3x-1x-1xf32, 32x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_261, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_0, parameter_261

        # pd_op.batch_norm_: (-1x32x-1x-1xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x-1x-1xf32, 32xf32, 32xf32, 32xf32, 32xf32)
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
                parameter_260,
                parameter_259,
                parameter_258,
                parameter_257,
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
        del conv2d_0, parameter_257, parameter_258, parameter_259, parameter_260

        # pd_op.relu: (-1x32x-1x-1xf32) <- (-1x32x-1x-1xf32)
        relu_0 = paddle._C_ops.relu(batch_norm__0)
        del batch_norm__0

        # pd_op.depthwise_conv2d: (-1x32x-1x-1xf32) <- (-1x32x-1x-1xf32, 32x1x3x3xf32)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(
            relu_0, parameter_256, [1, 1], [1, 1], "EXPLICIT", 32, [1, 1], "NCHW"
        )
        del parameter_256, relu_0

        # pd_op.batch_norm_: (-1x32x-1x-1xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x-1x-1xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        (
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            batch_norm__10,
            batch_norm__11,
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
                True,
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

        # pd_op.relu: (-1x32x-1x-1xf32) <- (-1x32x-1x-1xf32)
        relu_1 = paddle._C_ops.relu(batch_norm__6)
        del batch_norm__6

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x32x-1x-1xf32, 64x32x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            relu_1, parameter_251, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_251, relu_1

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            batch_norm__16,
            batch_norm__17,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_1,
                parameter_250,
                parameter_249,
                parameter_248,
                parameter_247,
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
        del conv2d_1, parameter_247, parameter_248, parameter_249, parameter_250

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_2 = paddle._C_ops.relu(batch_norm__12)
        del batch_norm__12

        # pd_op.depthwise_conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x1x3x3xf32)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(
            relu_2, parameter_246, [2, 2], [1, 1], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del parameter_246, relu_2

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__18,
            batch_norm__19,
            batch_norm__20,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_1,
                parameter_245,
                parameter_244,
                parameter_243,
                parameter_242,
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
        del (
            depthwise_conv2d_1,
            parameter_242,
            parameter_243,
            parameter_244,
            parameter_245,
        )

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_3 = paddle._C_ops.relu(batch_norm__18)
        del batch_norm__18

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x64x-1x-1xf32, 128x64x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            relu_3, parameter_241, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_241, relu_3

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_2,
                parameter_240,
                parameter_239,
                parameter_238,
                parameter_237,
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
        del conv2d_2, parameter_237, parameter_238, parameter_239, parameter_240

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_4 = paddle._C_ops.relu(batch_norm__24)
        del batch_norm__24

        # pd_op.depthwise_conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x1x3x3xf32)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(
            relu_4, parameter_236, [1, 1], [1, 1], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del parameter_236, relu_4

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_2,
                parameter_235,
                parameter_234,
                parameter_233,
                parameter_232,
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
        del (
            depthwise_conv2d_2,
            parameter_232,
            parameter_233,
            parameter_234,
            parameter_235,
        )

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_5 = paddle._C_ops.relu(batch_norm__30)
        del batch_norm__30

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x128x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            relu_5, parameter_231, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_231, relu_5

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__36,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_3,
                parameter_230,
                parameter_229,
                parameter_228,
                parameter_227,
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
        del conv2d_3, parameter_227, parameter_228, parameter_229, parameter_230

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_6 = paddle._C_ops.relu(batch_norm__36)
        del batch_norm__36

        # pd_op.depthwise_conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x1x3x3xf32)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(
            relu_6, parameter_226, [2, 2], [1, 1], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del parameter_226, relu_6

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__42,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_3,
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
        del (
            depthwise_conv2d_3,
            parameter_222,
            parameter_223,
            parameter_224,
            parameter_225,
        )

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_7 = paddle._C_ops.relu(batch_norm__42)
        del batch_norm__42

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x128x-1x-1xf32, 256x128x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            relu_7, parameter_221, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_221, relu_7

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__48,
            batch_norm__49,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_4,
                parameter_220,
                parameter_219,
                parameter_218,
                parameter_217,
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
        del conv2d_4, parameter_217, parameter_218, parameter_219, parameter_220

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_8 = paddle._C_ops.relu(batch_norm__48)
        del batch_norm__48

        # pd_op.depthwise_conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x1x3x3xf32)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(
            relu_8, parameter_216, [1, 1], [1, 1], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del parameter_216, relu_8

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_4,
                parameter_215,
                parameter_214,
                parameter_213,
                parameter_212,
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
        del (
            depthwise_conv2d_4,
            parameter_212,
            parameter_213,
            parameter_214,
            parameter_215,
        )

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_9 = paddle._C_ops.relu(batch_norm__54)
        del batch_norm__54

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            relu_9, parameter_211, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_211, relu_9

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_5,
                parameter_210,
                parameter_209,
                parameter_208,
                parameter_207,
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
        del conv2d_5, parameter_207, parameter_208, parameter_209, parameter_210

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_10 = paddle._C_ops.relu(batch_norm__60)
        del batch_norm__60

        # pd_op.depthwise_conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x1x5x5xf32)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(
            relu_10, parameter_206, [2, 2], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del parameter_206

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__66,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            batch_norm__70,
            batch_norm__71,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_5,
                parameter_205,
                parameter_204,
                parameter_203,
                parameter_202,
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
        del (
            depthwise_conv2d_5,
            parameter_202,
            parameter_203,
            parameter_204,
            parameter_205,
        )

        # pd_op.depthwise_conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x1x3x3xf32)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(
            relu_10, parameter_201, [2, 2], [1, 1], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del parameter_201, relu_10

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__72,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_6,
                parameter_200,
                parameter_199,
                parameter_198,
                parameter_197,
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
        del (
            depthwise_conv2d_6,
            parameter_197,
            parameter_198,
            parameter_199,
            parameter_200,
        )

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, -1x256x-1x-1xf32)
        add_1 = paddle._C_ops.add(batch_norm__66, batch_norm__72)
        del batch_norm__66, batch_norm__72

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_11 = paddle._C_ops.relu(add_1)
        del add_1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf32) <- (-1x256x-1x-1xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            relu_11,
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

        # pd_op.conv2d: (-1x64x1x1xf32) <- (-1x256x1x1xf32, 64x256x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            pool2d_0, parameter_196, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_196, pool2d_0

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [1, -1, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_195, full_int_array_1)
        del parameter_195

        # pd_op.add: (-1x64x1x1xf32) <- (-1x64x1x1xf32, 1x64x1x1xf32)
        add_2 = paddle._C_ops.add(conv2d_6, reshape_0)
        del conv2d_6, reshape_0

        # pd_op.relu: (-1x64x1x1xf32) <- (-1x64x1x1xf32)
        relu_12 = paddle._C_ops.relu(add_2)
        del add_2

        # pd_op.conv2d: (-1x256x1x1xf32) <- (-1x64x1x1xf32, 256x64x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            relu_12, parameter_194, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_194, relu_12

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_193, full_int_array_1)
        del parameter_193

        # pd_op.add: (-1x256x1x1xf32) <- (-1x256x1x1xf32, 1x256x1x1xf32)
        add_3 = paddle._C_ops.add(conv2d_7, reshape_1)
        del conv2d_7, reshape_1

        # pd_op.sigmoid: (-1x256x1x1xf32) <- (-1x256x1x1xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(add_3)
        del add_3

        # pd_op.multiply: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, -1x256x1x1xf32)
        multiply_0 = paddle._C_ops.multiply(relu_11, sigmoid_0)
        del relu_11, sigmoid_0

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            multiply_0, parameter_192, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_0, parameter_192

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__78,
            batch_norm__79,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
            batch_norm__83,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_8,
                parameter_191,
                parameter_190,
                parameter_189,
                parameter_188,
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
        del conv2d_8, parameter_188, parameter_189, parameter_190, parameter_191

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_13 = paddle._C_ops.relu(batch_norm__78)
        del batch_norm__78

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x256x-1x-1xf32, 512x256x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            relu_13, parameter_187, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_187, relu_13

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__84,
            batch_norm__85,
            batch_norm__86,
            batch_norm__87,
            batch_norm__88,
            batch_norm__89,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_9,
                parameter_186,
                parameter_185,
                parameter_184,
                parameter_183,
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
        del conv2d_9, parameter_183, parameter_184, parameter_185, parameter_186

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_14 = paddle._C_ops.relu(batch_norm__84)
        del batch_norm__84

        # pd_op.depthwise_conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x1x5x5xf32)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(
            relu_14, parameter_182, [1, 1], [2, 2], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del parameter_182

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__90,
            batch_norm__91,
            batch_norm__92,
            batch_norm__93,
            batch_norm__94,
            batch_norm__95,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_7,
                parameter_181,
                parameter_180,
                parameter_179,
                parameter_178,
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
        del (
            depthwise_conv2d_7,
            parameter_178,
            parameter_179,
            parameter_180,
            parameter_181,
        )

        # pd_op.depthwise_conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x1x3x3xf32)
        depthwise_conv2d_8 = paddle._C_ops.depthwise_conv2d(
            relu_14, parameter_177, [1, 1], [1, 1], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del parameter_177

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__96,
            batch_norm__97,
            batch_norm__98,
            batch_norm__99,
            batch_norm__100,
            batch_norm__101,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_8,
                parameter_176,
                parameter_175,
                parameter_174,
                parameter_173,
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
        del (
            depthwise_conv2d_8,
            parameter_173,
            parameter_174,
            parameter_175,
            parameter_176,
        )

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, -1x512x-1x-1xf32)
        add_4 = paddle._C_ops.add(batch_norm__90, batch_norm__96)
        del batch_norm__90, batch_norm__96

        # pd_op.depthwise_conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x1x1x1xf32)
        depthwise_conv2d_9 = paddle._C_ops.depthwise_conv2d(
            relu_14, parameter_172, [1, 1], [0, 0], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del parameter_172, relu_14

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__102,
            batch_norm__103,
            batch_norm__104,
            batch_norm__105,
            batch_norm__106,
            batch_norm__107,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_9,
                parameter_171,
                parameter_170,
                parameter_169,
                parameter_168,
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
        del (
            depthwise_conv2d_9,
            parameter_168,
            parameter_169,
            parameter_170,
            parameter_171,
        )

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, -1x512x-1x-1xf32)
        add_5 = paddle._C_ops.add(add_4, batch_norm__102)
        del add_4, batch_norm__102

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_15 = paddle._C_ops.relu(add_5)
        del add_5

        # pd_op.pool2d: (-1x512x1x1xf32) <- (-1x512x-1x-1xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(
            relu_15,
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

        # pd_op.conv2d: (-1x128x1x1xf32) <- (-1x512x1x1xf32, 128x512x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            pool2d_1, parameter_167, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_167, pool2d_1

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(parameter_166, full_int_array_1)
        del parameter_166

        # pd_op.add: (-1x128x1x1xf32) <- (-1x128x1x1xf32, 1x128x1x1xf32)
        add_6 = paddle._C_ops.add(conv2d_10, reshape_2)
        del conv2d_10, reshape_2

        # pd_op.relu: (-1x128x1x1xf32) <- (-1x128x1x1xf32)
        relu_16 = paddle._C_ops.relu(add_6)
        del add_6

        # pd_op.conv2d: (-1x512x1x1xf32) <- (-1x128x1x1xf32, 512x128x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            relu_16, parameter_165, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_165, relu_16

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(parameter_164, full_int_array_1)
        del parameter_164

        # pd_op.add: (-1x512x1x1xf32) <- (-1x512x1x1xf32, 1x512x1x1xf32)
        add_7 = paddle._C_ops.add(conv2d_11, reshape_3)
        del conv2d_11, reshape_3

        # pd_op.sigmoid: (-1x512x1x1xf32) <- (-1x512x1x1xf32)
        sigmoid_1 = paddle._C_ops.sigmoid(add_7)
        del add_7

        # pd_op.multiply: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, -1x512x1x1xf32)
        multiply_1 = paddle._C_ops.multiply(relu_15, sigmoid_1)
        del relu_15, sigmoid_1

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x512x-1x-1xf32, 256x512x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            multiply_1, parameter_163, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_1, parameter_163

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_162,
                parameter_161,
                parameter_160,
                parameter_159,
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
        del conv2d_12, parameter_159, parameter_160, parameter_161, parameter_162

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_17 = paddle._C_ops.relu(batch_norm__108)
        del batch_norm__108

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x256x-1x-1xf32, 512x256x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            relu_17, parameter_158, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_158, relu_17

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_157,
                parameter_156,
                parameter_155,
                parameter_154,
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
        del conv2d_13, parameter_154, parameter_155, parameter_156, parameter_157

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_18 = paddle._C_ops.relu(batch_norm__114)
        del batch_norm__114

        # pd_op.depthwise_conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x1x5x5xf32)
        depthwise_conv2d_10 = paddle._C_ops.depthwise_conv2d(
            relu_18, parameter_153, [1, 1], [2, 2], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del parameter_153

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__120,
            batch_norm__121,
            batch_norm__122,
            batch_norm__123,
            batch_norm__124,
            batch_norm__125,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_10,
                parameter_152,
                parameter_151,
                parameter_150,
                parameter_149,
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
        del (
            depthwise_conv2d_10,
            parameter_149,
            parameter_150,
            parameter_151,
            parameter_152,
        )

        # pd_op.depthwise_conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x1x3x3xf32)
        depthwise_conv2d_11 = paddle._C_ops.depthwise_conv2d(
            relu_18, parameter_148, [1, 1], [1, 1], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del parameter_148

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__126,
            batch_norm__127,
            batch_norm__128,
            batch_norm__129,
            batch_norm__130,
            batch_norm__131,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_11,
                parameter_147,
                parameter_146,
                parameter_145,
                parameter_144,
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
        del (
            depthwise_conv2d_11,
            parameter_144,
            parameter_145,
            parameter_146,
            parameter_147,
        )

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, -1x512x-1x-1xf32)
        add_8 = paddle._C_ops.add(batch_norm__120, batch_norm__126)
        del batch_norm__120, batch_norm__126

        # pd_op.depthwise_conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x1x1x1xf32)
        depthwise_conv2d_12 = paddle._C_ops.depthwise_conv2d(
            relu_18, parameter_143, [1, 1], [0, 0], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del parameter_143, relu_18

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__132,
            batch_norm__133,
            batch_norm__134,
            batch_norm__135,
            batch_norm__136,
            batch_norm__137,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_12,
                parameter_142,
                parameter_141,
                parameter_140,
                parameter_139,
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
        del (
            depthwise_conv2d_12,
            parameter_139,
            parameter_140,
            parameter_141,
            parameter_142,
        )

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, -1x512x-1x-1xf32)
        add_9 = paddle._C_ops.add(add_8, batch_norm__132)
        del add_8, batch_norm__132

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_19 = paddle._C_ops.relu(add_9)
        del add_9

        # pd_op.pool2d: (-1x512x1x1xf32) <- (-1x512x-1x-1xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(
            relu_19,
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

        # pd_op.conv2d: (-1x128x1x1xf32) <- (-1x512x1x1xf32, 128x512x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            pool2d_2, parameter_138, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_138, pool2d_2

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(parameter_137, full_int_array_1)
        del parameter_137

        # pd_op.add: (-1x128x1x1xf32) <- (-1x128x1x1xf32, 1x128x1x1xf32)
        add_10 = paddle._C_ops.add(conv2d_14, reshape_4)
        del conv2d_14, reshape_4

        # pd_op.relu: (-1x128x1x1xf32) <- (-1x128x1x1xf32)
        relu_20 = paddle._C_ops.relu(add_10)
        del add_10

        # pd_op.conv2d: (-1x512x1x1xf32) <- (-1x128x1x1xf32, 512x128x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            relu_20, parameter_136, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_136, relu_20

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(parameter_135, full_int_array_1)
        del parameter_135

        # pd_op.add: (-1x512x1x1xf32) <- (-1x512x1x1xf32, 1x512x1x1xf32)
        add_11 = paddle._C_ops.add(conv2d_15, reshape_5)
        del conv2d_15, reshape_5

        # pd_op.sigmoid: (-1x512x1x1xf32) <- (-1x512x1x1xf32)
        sigmoid_2 = paddle._C_ops.sigmoid(add_11)
        del add_11

        # pd_op.multiply: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, -1x512x1x1xf32)
        multiply_2 = paddle._C_ops.multiply(relu_19, sigmoid_2)
        del relu_19, sigmoid_2

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x512x-1x-1xf32, 256x512x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            multiply_2, parameter_134, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_2, parameter_134

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__138,
            batch_norm__139,
            batch_norm__140,
            batch_norm__141,
            batch_norm__142,
            batch_norm__143,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_16,
                parameter_133,
                parameter_132,
                parameter_131,
                parameter_130,
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
        del conv2d_16, parameter_130, parameter_131, parameter_132, parameter_133

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_21 = paddle._C_ops.relu(batch_norm__138)
        del batch_norm__138

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x256x-1x-1xf32, 512x256x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            relu_21, parameter_129, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_129, relu_21

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__144,
            batch_norm__145,
            batch_norm__146,
            batch_norm__147,
            batch_norm__148,
            batch_norm__149,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_17,
                parameter_128,
                parameter_127,
                parameter_126,
                parameter_125,
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
        del conv2d_17, parameter_125, parameter_126, parameter_127, parameter_128

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_22 = paddle._C_ops.relu(batch_norm__144)
        del batch_norm__144

        # pd_op.depthwise_conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x1x5x5xf32)
        depthwise_conv2d_13 = paddle._C_ops.depthwise_conv2d(
            relu_22, parameter_124, [1, 1], [2, 2], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del parameter_124

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__150,
            batch_norm__151,
            batch_norm__152,
            batch_norm__153,
            batch_norm__154,
            batch_norm__155,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_13,
                parameter_123,
                parameter_122,
                parameter_121,
                parameter_120,
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
        del (
            depthwise_conv2d_13,
            parameter_120,
            parameter_121,
            parameter_122,
            parameter_123,
        )

        # pd_op.depthwise_conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x1x3x3xf32)
        depthwise_conv2d_14 = paddle._C_ops.depthwise_conv2d(
            relu_22, parameter_119, [1, 1], [1, 1], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del parameter_119

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__156,
            batch_norm__157,
            batch_norm__158,
            batch_norm__159,
            batch_norm__160,
            batch_norm__161,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_14,
                parameter_118,
                parameter_117,
                parameter_116,
                parameter_115,
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
        del (
            depthwise_conv2d_14,
            parameter_115,
            parameter_116,
            parameter_117,
            parameter_118,
        )

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, -1x512x-1x-1xf32)
        add_12 = paddle._C_ops.add(batch_norm__150, batch_norm__156)
        del batch_norm__150, batch_norm__156

        # pd_op.depthwise_conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x1x1x1xf32)
        depthwise_conv2d_15 = paddle._C_ops.depthwise_conv2d(
            relu_22, parameter_114, [1, 1], [0, 0], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del parameter_114, relu_22

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__162,
            batch_norm__163,
            batch_norm__164,
            batch_norm__165,
            batch_norm__166,
            batch_norm__167,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_15,
                parameter_113,
                parameter_112,
                parameter_111,
                parameter_110,
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
        del (
            depthwise_conv2d_15,
            parameter_110,
            parameter_111,
            parameter_112,
            parameter_113,
        )

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, -1x512x-1x-1xf32)
        add_13 = paddle._C_ops.add(add_12, batch_norm__162)
        del add_12, batch_norm__162

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_23 = paddle._C_ops.relu(add_13)
        del add_13

        # pd_op.pool2d: (-1x512x1x1xf32) <- (-1x512x-1x-1xf32, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(
            relu_23,
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

        # pd_op.conv2d: (-1x128x1x1xf32) <- (-1x512x1x1xf32, 128x512x1x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            pool2d_3, parameter_109, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_109, pool2d_3

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(parameter_108, full_int_array_1)
        del parameter_108

        # pd_op.add: (-1x128x1x1xf32) <- (-1x128x1x1xf32, 1x128x1x1xf32)
        add_14 = paddle._C_ops.add(conv2d_18, reshape_6)
        del conv2d_18, reshape_6

        # pd_op.relu: (-1x128x1x1xf32) <- (-1x128x1x1xf32)
        relu_24 = paddle._C_ops.relu(add_14)
        del add_14

        # pd_op.conv2d: (-1x512x1x1xf32) <- (-1x128x1x1xf32, 512x128x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            relu_24, parameter_107, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_107, relu_24

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(parameter_106, full_int_array_1)
        del parameter_106

        # pd_op.add: (-1x512x1x1xf32) <- (-1x512x1x1xf32, 1x512x1x1xf32)
        add_15 = paddle._C_ops.add(conv2d_19, reshape_7)
        del conv2d_19, reshape_7

        # pd_op.sigmoid: (-1x512x1x1xf32) <- (-1x512x1x1xf32)
        sigmoid_3 = paddle._C_ops.sigmoid(add_15)
        del add_15

        # pd_op.multiply: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, -1x512x1x1xf32)
        multiply_3 = paddle._C_ops.multiply(relu_23, sigmoid_3)
        del relu_23, sigmoid_3

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x512x-1x-1xf32, 256x512x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            multiply_3, parameter_105, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_3, parameter_105

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__168,
            batch_norm__169,
            batch_norm__170,
            batch_norm__171,
            batch_norm__172,
            batch_norm__173,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_20,
                parameter_104,
                parameter_103,
                parameter_102,
                parameter_101,
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
        del conv2d_20, parameter_101, parameter_102, parameter_103, parameter_104

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_25 = paddle._C_ops.relu(batch_norm__168)
        del batch_norm__168

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x256x-1x-1xf32, 512x256x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            relu_25, parameter_100, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_100, relu_25

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__174,
            batch_norm__175,
            batch_norm__176,
            batch_norm__177,
            batch_norm__178,
            batch_norm__179,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_21,
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
        del conv2d_21, parameter_96, parameter_97, parameter_98, parameter_99

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_26 = paddle._C_ops.relu(batch_norm__174)
        del batch_norm__174

        # pd_op.depthwise_conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x1x5x5xf32)
        depthwise_conv2d_16 = paddle._C_ops.depthwise_conv2d(
            relu_26, parameter_95, [1, 1], [2, 2], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del parameter_95

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__180,
            batch_norm__181,
            batch_norm__182,
            batch_norm__183,
            batch_norm__184,
            batch_norm__185,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_16,
                parameter_94,
                parameter_93,
                parameter_92,
                parameter_91,
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
        del depthwise_conv2d_16, parameter_91, parameter_92, parameter_93, parameter_94

        # pd_op.depthwise_conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x1x3x3xf32)
        depthwise_conv2d_17 = paddle._C_ops.depthwise_conv2d(
            relu_26, parameter_90, [1, 1], [1, 1], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del parameter_90

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__186,
            batch_norm__187,
            batch_norm__188,
            batch_norm__189,
            batch_norm__190,
            batch_norm__191,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_17,
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
        del depthwise_conv2d_17, parameter_86, parameter_87, parameter_88, parameter_89

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, -1x512x-1x-1xf32)
        add_16 = paddle._C_ops.add(batch_norm__180, batch_norm__186)
        del batch_norm__180, batch_norm__186

        # pd_op.depthwise_conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x1x1x1xf32)
        depthwise_conv2d_18 = paddle._C_ops.depthwise_conv2d(
            relu_26, parameter_85, [1, 1], [0, 0], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del parameter_85, relu_26

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__192,
            batch_norm__193,
            batch_norm__194,
            batch_norm__195,
            batch_norm__196,
            batch_norm__197,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_18,
                parameter_84,
                parameter_83,
                parameter_82,
                parameter_81,
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
        del depthwise_conv2d_18, parameter_81, parameter_82, parameter_83, parameter_84

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, -1x512x-1x-1xf32)
        add_17 = paddle._C_ops.add(add_16, batch_norm__192)
        del add_16, batch_norm__192

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_27 = paddle._C_ops.relu(add_17)
        del add_17

        # pd_op.pool2d: (-1x512x1x1xf32) <- (-1x512x-1x-1xf32, 2xi64)
        pool2d_4 = paddle._C_ops.pool2d(
            relu_27,
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

        # pd_op.conv2d: (-1x128x1x1xf32) <- (-1x512x1x1xf32, 128x512x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(
            pool2d_4, parameter_80, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_80, pool2d_4

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(parameter_79, full_int_array_1)
        del parameter_79

        # pd_op.add: (-1x128x1x1xf32) <- (-1x128x1x1xf32, 1x128x1x1xf32)
        add_18 = paddle._C_ops.add(conv2d_22, reshape_8)
        del conv2d_22, reshape_8

        # pd_op.relu: (-1x128x1x1xf32) <- (-1x128x1x1xf32)
        relu_28 = paddle._C_ops.relu(add_18)
        del add_18

        # pd_op.conv2d: (-1x512x1x1xf32) <- (-1x128x1x1xf32, 512x128x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(
            relu_28, parameter_78, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_78, relu_28

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(parameter_77, full_int_array_1)
        del parameter_77

        # pd_op.add: (-1x512x1x1xf32) <- (-1x512x1x1xf32, 1x512x1x1xf32)
        add_19 = paddle._C_ops.add(conv2d_23, reshape_9)
        del conv2d_23, reshape_9

        # pd_op.sigmoid: (-1x512x1x1xf32) <- (-1x512x1x1xf32)
        sigmoid_4 = paddle._C_ops.sigmoid(add_19)
        del add_19

        # pd_op.multiply: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, -1x512x1x1xf32)
        multiply_4 = paddle._C_ops.multiply(relu_27, sigmoid_4)
        del relu_27, sigmoid_4

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x512x-1x-1xf32, 256x512x1x1xf32)
        conv2d_24 = paddle._C_ops.conv2d(
            multiply_4, parameter_76, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_4, parameter_76

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__198,
            batch_norm__199,
            batch_norm__200,
            batch_norm__201,
            batch_norm__202,
            batch_norm__203,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_24,
                parameter_75,
                parameter_74,
                parameter_73,
                parameter_72,
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
        del conv2d_24, parameter_72, parameter_73, parameter_74, parameter_75

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_29 = paddle._C_ops.relu(batch_norm__198)
        del batch_norm__198

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x256x-1x-1xf32, 512x256x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(
            relu_29, parameter_71, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_71, relu_29

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__204,
            batch_norm__205,
            batch_norm__206,
            batch_norm__207,
            batch_norm__208,
            batch_norm__209,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_25,
                parameter_70,
                parameter_69,
                parameter_68,
                parameter_67,
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
        del conv2d_25, parameter_67, parameter_68, parameter_69, parameter_70

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_30 = paddle._C_ops.relu(batch_norm__204)
        del batch_norm__204

        # pd_op.depthwise_conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x1x5x5xf32)
        depthwise_conv2d_19 = paddle._C_ops.depthwise_conv2d(
            relu_30, parameter_66, [1, 1], [2, 2], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del parameter_66

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__210,
            batch_norm__211,
            batch_norm__212,
            batch_norm__213,
            batch_norm__214,
            batch_norm__215,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_19,
                parameter_65,
                parameter_64,
                parameter_63,
                parameter_62,
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
        del depthwise_conv2d_19, parameter_62, parameter_63, parameter_64, parameter_65

        # pd_op.depthwise_conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x1x3x3xf32)
        depthwise_conv2d_20 = paddle._C_ops.depthwise_conv2d(
            relu_30, parameter_61, [1, 1], [1, 1], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del parameter_61

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__216,
            batch_norm__217,
            batch_norm__218,
            batch_norm__219,
            batch_norm__220,
            batch_norm__221,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_20,
                parameter_60,
                parameter_59,
                parameter_58,
                parameter_57,
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
        del depthwise_conv2d_20, parameter_57, parameter_58, parameter_59, parameter_60

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, -1x512x-1x-1xf32)
        add_20 = paddle._C_ops.add(batch_norm__210, batch_norm__216)
        del batch_norm__210, batch_norm__216

        # pd_op.depthwise_conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x1x1x1xf32)
        depthwise_conv2d_21 = paddle._C_ops.depthwise_conv2d(
            relu_30, parameter_56, [1, 1], [0, 0], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del parameter_56, relu_30

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__222,
            batch_norm__223,
            batch_norm__224,
            batch_norm__225,
            batch_norm__226,
            batch_norm__227,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_21,
                parameter_55,
                parameter_54,
                parameter_53,
                parameter_52,
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
        del depthwise_conv2d_21, parameter_52, parameter_53, parameter_54, parameter_55

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, -1x512x-1x-1xf32)
        add_21 = paddle._C_ops.add(add_20, batch_norm__222)
        del add_20, batch_norm__222

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_31 = paddle._C_ops.relu(add_21)
        del add_21

        # pd_op.pool2d: (-1x512x1x1xf32) <- (-1x512x-1x-1xf32, 2xi64)
        pool2d_5 = paddle._C_ops.pool2d(
            relu_31,
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

        # pd_op.conv2d: (-1x128x1x1xf32) <- (-1x512x1x1xf32, 128x512x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(
            pool2d_5, parameter_51, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_51, pool2d_5

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(parameter_50, full_int_array_1)
        del parameter_50

        # pd_op.add: (-1x128x1x1xf32) <- (-1x128x1x1xf32, 1x128x1x1xf32)
        add_22 = paddle._C_ops.add(conv2d_26, reshape_10)
        del conv2d_26, reshape_10

        # pd_op.relu: (-1x128x1x1xf32) <- (-1x128x1x1xf32)
        relu_32 = paddle._C_ops.relu(add_22)
        del add_22

        # pd_op.conv2d: (-1x512x1x1xf32) <- (-1x128x1x1xf32, 512x128x1x1xf32)
        conv2d_27 = paddle._C_ops.conv2d(
            relu_32, parameter_49, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_49, relu_32

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_11 = paddle._C_ops.reshape(parameter_48, full_int_array_1)
        del full_int_array_1, parameter_48

        # pd_op.add: (-1x512x1x1xf32) <- (-1x512x1x1xf32, 1x512x1x1xf32)
        add_23 = paddle._C_ops.add(conv2d_27, reshape_11)
        del conv2d_27, reshape_11

        # pd_op.sigmoid: (-1x512x1x1xf32) <- (-1x512x1x1xf32)
        sigmoid_5 = paddle._C_ops.sigmoid(add_23)
        del add_23

        # pd_op.multiply: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, -1x512x1x1xf32)
        multiply_5 = paddle._C_ops.multiply(relu_31, sigmoid_5)
        del relu_31, sigmoid_5

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x512x-1x-1xf32, 256x512x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(
            multiply_5, parameter_47, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_5, parameter_47

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__228,
            batch_norm__229,
            batch_norm__230,
            batch_norm__231,
            batch_norm__232,
            batch_norm__233,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_28,
                parameter_46,
                parameter_45,
                parameter_44,
                parameter_43,
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
        del conv2d_28, parameter_43, parameter_44, parameter_45, parameter_46

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_33 = paddle._C_ops.relu(batch_norm__228)
        del batch_norm__228

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x256x-1x-1xf32, 512x256x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(
            relu_33, parameter_42, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_42, relu_33

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__234,
            batch_norm__235,
            batch_norm__236,
            batch_norm__237,
            batch_norm__238,
            batch_norm__239,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_29,
                parameter_41,
                parameter_40,
                parameter_39,
                parameter_38,
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
        del conv2d_29, parameter_38, parameter_39, parameter_40, parameter_41

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_34 = paddle._C_ops.relu(batch_norm__234)
        del batch_norm__234

        # pd_op.depthwise_conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x1x5x5xf32)
        depthwise_conv2d_22 = paddle._C_ops.depthwise_conv2d(
            relu_34, parameter_37, [2, 2], [2, 2], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del parameter_37

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__240,
            batch_norm__241,
            batch_norm__242,
            batch_norm__243,
            batch_norm__244,
            batch_norm__245,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_22,
                parameter_36,
                parameter_35,
                parameter_34,
                parameter_33,
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
        del depthwise_conv2d_22, parameter_33, parameter_34, parameter_35, parameter_36

        # pd_op.depthwise_conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x1x3x3xf32)
        depthwise_conv2d_23 = paddle._C_ops.depthwise_conv2d(
            relu_34, parameter_32, [2, 2], [1, 1], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del parameter_32, relu_34

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__246,
            batch_norm__247,
            batch_norm__248,
            batch_norm__249,
            batch_norm__250,
            batch_norm__251,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_23,
                parameter_31,
                parameter_30,
                parameter_29,
                parameter_28,
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
        del depthwise_conv2d_23, parameter_28, parameter_29, parameter_30, parameter_31

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, -1x512x-1x-1xf32)
        add_24 = paddle._C_ops.add(batch_norm__240, batch_norm__246)
        del batch_norm__240, batch_norm__246

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_35 = paddle._C_ops.relu(add_24)
        del add_24

        # pd_op.conv2d: (-1x1024x-1x-1xf32) <- (-1x512x-1x-1xf32, 1024x512x1x1xf32)
        conv2d_30 = paddle._C_ops.conv2d(
            relu_35, parameter_27, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_27, relu_35

        # pd_op.batch_norm_: (-1x1024x-1x-1xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x-1x-1xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        (
            batch_norm__252,
            batch_norm__253,
            batch_norm__254,
            batch_norm__255,
            batch_norm__256,
            batch_norm__257,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_30,
                parameter_26,
                parameter_25,
                parameter_24,
                parameter_23,
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
        del conv2d_30, parameter_23, parameter_24, parameter_25, parameter_26

        # pd_op.relu: (-1x1024x-1x-1xf32) <- (-1x1024x-1x-1xf32)
        relu_36 = paddle._C_ops.relu(batch_norm__252)
        del batch_norm__252

        # pd_op.depthwise_conv2d: (-1x1024x-1x-1xf32) <- (-1x1024x-1x-1xf32, 1024x1x5x5xf32)
        depthwise_conv2d_24 = paddle._C_ops.depthwise_conv2d(
            relu_36, parameter_22, [1, 1], [2, 2], "EXPLICIT", 1024, [1, 1], "NCHW"
        )
        del parameter_22

        # pd_op.batch_norm_: (-1x1024x-1x-1xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x-1x-1xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        (
            batch_norm__258,
            batch_norm__259,
            batch_norm__260,
            batch_norm__261,
            batch_norm__262,
            batch_norm__263,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_24,
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
        del depthwise_conv2d_24, parameter_18, parameter_19, parameter_20, parameter_21

        # pd_op.depthwise_conv2d: (-1x1024x-1x-1xf32) <- (-1x1024x-1x-1xf32, 1024x1x3x3xf32)
        depthwise_conv2d_25 = paddle._C_ops.depthwise_conv2d(
            relu_36, parameter_17, [1, 1], [1, 1], "EXPLICIT", 1024, [1, 1], "NCHW"
        )
        del parameter_17

        # pd_op.batch_norm_: (-1x1024x-1x-1xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x-1x-1xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        (
            batch_norm__264,
            batch_norm__265,
            batch_norm__266,
            batch_norm__267,
            batch_norm__268,
            batch_norm__269,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_25,
                parameter_16,
                parameter_15,
                parameter_14,
                parameter_13,
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
        del depthwise_conv2d_25, parameter_13, parameter_14, parameter_15, parameter_16

        # pd_op.add: (-1x1024x-1x-1xf32) <- (-1x1024x-1x-1xf32, -1x1024x-1x-1xf32)
        add_25 = paddle._C_ops.add(batch_norm__258, batch_norm__264)
        del batch_norm__258, batch_norm__264

        # pd_op.depthwise_conv2d: (-1x1024x-1x-1xf32) <- (-1x1024x-1x-1xf32, 1024x1x1x1xf32)
        depthwise_conv2d_26 = paddle._C_ops.depthwise_conv2d(
            relu_36, parameter_12, [1, 1], [0, 0], "EXPLICIT", 1024, [1, 1], "NCHW"
        )
        del parameter_12

        # pd_op.batch_norm_: (-1x1024x-1x-1xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x-1x-1xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        (
            batch_norm__270,
            batch_norm__271,
            batch_norm__272,
            batch_norm__273,
            batch_norm__274,
            batch_norm__275,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_26,
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
        del depthwise_conv2d_26, parameter_10, parameter_11, parameter_8, parameter_9

        # pd_op.add: (-1x1024x-1x-1xf32) <- (-1x1024x-1x-1xf32, -1x1024x-1x-1xf32)
        add_26 = paddle._C_ops.add(add_25, batch_norm__270)
        del add_25, batch_norm__270

        # pd_op.relu: (-1x1024x-1x-1xf32) <- (-1x1024x-1x-1xf32)
        relu_37 = paddle._C_ops.relu(add_26)
        del add_26

        # pd_op.conv2d: (-1x1024x-1x-1xf32) <- (-1x1024x-1x-1xf32, 1024x1024x1x1xf32)
        conv2d_31 = paddle._C_ops.conv2d(
            relu_37, parameter_7, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_7, relu_37

        # pd_op.batch_norm_: (-1x1024x-1x-1xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x-1x-1xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
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
                parameter_6,
                parameter_5,
                parameter_4,
                parameter_3,
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
        del conv2d_31, parameter_3, parameter_4, parameter_5, parameter_6

        # pd_op.relu: (-1x1024x-1x-1xf32) <- (-1x1024x-1x-1xf32)
        relu_38 = paddle._C_ops.relu(batch_norm__276)
        del batch_norm__276

        # pd_op.add: (-1x1024x-1x-1xf32) <- (-1x1024x-1x-1xf32, -1x1024x-1x-1xf32)
        add_27 = paddle._C_ops.add(relu_38, relu_36)
        del relu_36, relu_38

        # pd_op.pool2d: (-1x1024x1x1xf32) <- (-1x1024x-1x-1xf32, 2xi64)
        pool2d_6 = paddle._C_ops.pool2d(
            add_27,
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
        del add_27, full_int_array_0

        # pd_op.conv2d: (-1x1280x1x1xf32) <- (-1x1024x1x1xf32, 1280x1024x1x1xf32)
        conv2d_32 = paddle._C_ops.conv2d(
            pool2d_6, parameter_2, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_2, pool2d_6

        # pd_op.relu: (-1x1280x1x1xf32) <- (-1x1280x1x1xf32)
        relu_39 = paddle._C_ops.relu(conv2d_32)
        del conv2d_32

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (-1x1280x1x1xf32, -1x1280x1x1xui8) <- (-1x1280x1x1xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                relu_39, None, full_0, True, "downgrade_in_infer", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del full_0, relu_39

        # pd_op.flatten: (-1x1280xf32) <- (-1x1280x1x1xf32)
        flatten_0 = paddle._C_ops.flatten(dropout_0, 1, 3)
        del dropout_0

        # pd_op.matmul: (-1x102xf32) <- (-1x1280xf32, 1280x102xf32)
        matmul_0 = paddle._C_ops.matmul(flatten_0, parameter_1, False, False)
        del flatten_0, parameter_1

        # pd_op.add: (-1x102xf32) <- (-1x102xf32, 102xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_0)
        del matmul_0, parameter_0

        return add_0
