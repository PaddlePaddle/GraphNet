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
        parameter_270,
        parameter_271,
        parameter_272,
        parameter_273,
        parameter_274,
        parameter_275,
        parameter_276,
        parameter_277,
        parameter_278,
        parameter_279,
        parameter_280,
        parameter_281,
        parameter_282,
        data_0,
        data_1,
        data_2,
    ):
        # pd_op.conv2d: (128x64x56x56xf32) <- (128x3x112x112xf32, 64x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_1, parameter_282, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_1, parameter_282

        # pd_op.batch_norm_: (128x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (128x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_281,
                parameter_280,
                parameter_279,
                parameter_278,
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
        del parameter_278, parameter_279, parameter_280, parameter_281

        # pd_op.prelu: (128x64x56x56xf32) <- (128x64x56x56xf32, 64xf32)
        prelu_0 = paddle._C_ops.prelu(batch_norm__0, parameter_277, "NCHW", "channel")
        del parameter_277

        # pd_op.depthwise_conv2d: (128x64x56x56xf32) <- (128x64x56x56xf32, 64x1x3x3xf32)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(
            prelu_0, parameter_276, [1, 1], [1, 1], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del parameter_276

        # pd_op.batch_norm_: (128x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (128x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_275,
                parameter_274,
                parameter_273,
                parameter_272,
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
        del parameter_272, parameter_273, parameter_274, parameter_275

        # pd_op.prelu: (128x64x56x56xf32) <- (128x64x56x56xf32, 64xf32)
        prelu_1 = paddle._C_ops.prelu(batch_norm__6, parameter_271, "NCHW", "channel")
        del parameter_271

        # pd_op.conv2d: (128x128x56x56xf32) <- (128x64x56x56xf32, 128x64x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            prelu_1, parameter_270, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_270

        # pd_op.batch_norm_: (128x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (128x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_269,
                parameter_268,
                parameter_267,
                parameter_266,
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
        del parameter_266, parameter_267, parameter_268, parameter_269

        # pd_op.prelu: (128x128x56x56xf32) <- (128x128x56x56xf32, 128xf32)
        prelu_2 = paddle._C_ops.prelu(batch_norm__12, parameter_265, "NCHW", "channel")
        del parameter_265

        # pd_op.depthwise_conv2d: (128x128x28x28xf32) <- (128x128x56x56xf32, 128x1x3x3xf32)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(
            prelu_2, parameter_264, [2, 2], [1, 1], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del parameter_264

        # pd_op.batch_norm_: (128x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (128x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
        del parameter_260, parameter_261, parameter_262, parameter_263

        # pd_op.prelu: (128x128x28x28xf32) <- (128x128x28x28xf32, 128xf32)
        prelu_3 = paddle._C_ops.prelu(batch_norm__18, parameter_259, "NCHW", "channel")
        del parameter_259

        # pd_op.conv2d: (128x64x28x28xf32) <- (128x128x28x28xf32, 64x128x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            prelu_3, parameter_258, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_258

        # pd_op.batch_norm_: (128x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (128x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_257,
                parameter_256,
                parameter_255,
                parameter_254,
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
        del parameter_254, parameter_255, parameter_256, parameter_257

        # pd_op.conv2d: (128x128x28x28xf32) <- (128x64x28x28xf32, 128x64x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            batch_norm__24, parameter_253, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_253

        # pd_op.batch_norm_: (128x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (128x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_3,
                parameter_252,
                parameter_251,
                parameter_250,
                parameter_249,
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
        del parameter_249, parameter_250, parameter_251, parameter_252

        # pd_op.prelu: (128x128x28x28xf32) <- (128x128x28x28xf32, 128xf32)
        prelu_4 = paddle._C_ops.prelu(batch_norm__30, parameter_248, "NCHW", "channel")
        del parameter_248

        # pd_op.depthwise_conv2d: (128x128x28x28xf32) <- (128x128x28x28xf32, 128x1x3x3xf32)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(
            prelu_4, parameter_247, [1, 1], [1, 1], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del parameter_247

        # pd_op.batch_norm_: (128x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (128x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__36,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_2,
                parameter_246,
                parameter_245,
                parameter_244,
                parameter_243,
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
        del parameter_243, parameter_244, parameter_245, parameter_246

        # pd_op.prelu: (128x128x28x28xf32) <- (128x128x28x28xf32, 128xf32)
        prelu_5 = paddle._C_ops.prelu(batch_norm__36, parameter_242, "NCHW", "channel")
        del parameter_242

        # pd_op.conv2d: (128x64x28x28xf32) <- (128x128x28x28xf32, 64x128x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            prelu_5, parameter_241, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_241

        # pd_op.batch_norm_: (128x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (128x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__42,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_4,
                parameter_240,
                parameter_239,
                parameter_238,
                parameter_237,
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
        del parameter_237, parameter_238, parameter_239, parameter_240

        # pd_op.add: (128x64x28x28xf32) <- (128x64x28x28xf32, 128x64x28x28xf32)
        add_0 = paddle._C_ops.add(batch_norm__24, batch_norm__42)

        # pd_op.conv2d: (128x128x28x28xf32) <- (128x64x28x28xf32, 128x64x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            add_0, parameter_236, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_236

        # pd_op.batch_norm_: (128x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (128x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__48,
            batch_norm__49,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_5,
                parameter_235,
                parameter_234,
                parameter_233,
                parameter_232,
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
        del parameter_232, parameter_233, parameter_234, parameter_235

        # pd_op.prelu: (128x128x28x28xf32) <- (128x128x28x28xf32, 128xf32)
        prelu_6 = paddle._C_ops.prelu(batch_norm__48, parameter_231, "NCHW", "channel")
        del parameter_231

        # pd_op.depthwise_conv2d: (128x128x28x28xf32) <- (128x128x28x28xf32, 128x1x3x3xf32)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(
            prelu_6, parameter_230, [1, 1], [1, 1], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del parameter_230

        # pd_op.batch_norm_: (128x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (128x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_3,
                parameter_229,
                parameter_228,
                parameter_227,
                parameter_226,
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
        del parameter_226, parameter_227, parameter_228, parameter_229

        # pd_op.prelu: (128x128x28x28xf32) <- (128x128x28x28xf32, 128xf32)
        prelu_7 = paddle._C_ops.prelu(batch_norm__54, parameter_225, "NCHW", "channel")
        del parameter_225

        # pd_op.conv2d: (128x64x28x28xf32) <- (128x128x28x28xf32, 64x128x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            prelu_7, parameter_224, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_224

        # pd_op.batch_norm_: (128x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (128x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
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
        del parameter_220, parameter_221, parameter_222, parameter_223

        # pd_op.add: (128x64x28x28xf32) <- (128x64x28x28xf32, 128x64x28x28xf32)
        add_1 = paddle._C_ops.add(add_0, batch_norm__60)

        # pd_op.conv2d: (128x128x28x28xf32) <- (128x64x28x28xf32, 128x64x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            add_1, parameter_219, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_219

        # pd_op.batch_norm_: (128x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (128x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__66,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            batch_norm__70,
            batch_norm__71,
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
        del parameter_215, parameter_216, parameter_217, parameter_218

        # pd_op.prelu: (128x128x28x28xf32) <- (128x128x28x28xf32, 128xf32)
        prelu_8 = paddle._C_ops.prelu(batch_norm__66, parameter_214, "NCHW", "channel")
        del parameter_214

        # pd_op.depthwise_conv2d: (128x128x28x28xf32) <- (128x128x28x28xf32, 128x1x3x3xf32)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(
            prelu_8, parameter_213, [1, 1], [1, 1], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del parameter_213

        # pd_op.batch_norm_: (128x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (128x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__72,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_4,
                parameter_212,
                parameter_211,
                parameter_210,
                parameter_209,
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
        del parameter_209, parameter_210, parameter_211, parameter_212

        # pd_op.prelu: (128x128x28x28xf32) <- (128x128x28x28xf32, 128xf32)
        prelu_9 = paddle._C_ops.prelu(batch_norm__72, parameter_208, "NCHW", "channel")
        del parameter_208

        # pd_op.conv2d: (128x64x28x28xf32) <- (128x128x28x28xf32, 64x128x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            prelu_9, parameter_207, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_207

        # pd_op.batch_norm_: (128x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (128x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_206,
                parameter_205,
                parameter_204,
                parameter_203,
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
        del parameter_203, parameter_204, parameter_205, parameter_206

        # pd_op.add: (128x64x28x28xf32) <- (128x64x28x28xf32, 128x64x28x28xf32)
        add_2 = paddle._C_ops.add(add_1, batch_norm__78)

        # pd_op.conv2d: (128x128x28x28xf32) <- (128x64x28x28xf32, 128x64x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            add_2, parameter_202, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_202

        # pd_op.batch_norm_: (128x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (128x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_201,
                parameter_200,
                parameter_199,
                parameter_198,
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
        del parameter_198, parameter_199, parameter_200, parameter_201

        # pd_op.prelu: (128x128x28x28xf32) <- (128x128x28x28xf32, 128xf32)
        prelu_10 = paddle._C_ops.prelu(batch_norm__84, parameter_197, "NCHW", "channel")
        del parameter_197

        # pd_op.depthwise_conv2d: (128x128x28x28xf32) <- (128x128x28x28xf32, 128x1x3x3xf32)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(
            prelu_10, parameter_196, [1, 1], [1, 1], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del parameter_196

        # pd_op.batch_norm_: (128x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (128x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__90,
            batch_norm__91,
            batch_norm__92,
            batch_norm__93,
            batch_norm__94,
            batch_norm__95,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_5,
                parameter_195,
                parameter_194,
                parameter_193,
                parameter_192,
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
        del parameter_192, parameter_193, parameter_194, parameter_195

        # pd_op.prelu: (128x128x28x28xf32) <- (128x128x28x28xf32, 128xf32)
        prelu_11 = paddle._C_ops.prelu(batch_norm__90, parameter_191, "NCHW", "channel")
        del parameter_191

        # pd_op.conv2d: (128x64x28x28xf32) <- (128x128x28x28xf32, 64x128x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            prelu_11, parameter_190, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_190

        # pd_op.batch_norm_: (128x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (128x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__96,
            batch_norm__97,
            batch_norm__98,
            batch_norm__99,
            batch_norm__100,
            batch_norm__101,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_10,
                parameter_189,
                parameter_188,
                parameter_187,
                parameter_186,
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
        del parameter_186, parameter_187, parameter_188, parameter_189

        # pd_op.add: (128x64x28x28xf32) <- (128x64x28x28xf32, 128x64x28x28xf32)
        add_3 = paddle._C_ops.add(add_2, batch_norm__96)

        # pd_op.conv2d: (128x256x28x28xf32) <- (128x64x28x28xf32, 256x64x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            add_3, parameter_185, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_185

        # pd_op.batch_norm_: (128x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (128x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__102,
            batch_norm__103,
            batch_norm__104,
            batch_norm__105,
            batch_norm__106,
            batch_norm__107,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_11,
                parameter_184,
                parameter_183,
                parameter_182,
                parameter_181,
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
        del parameter_181, parameter_182, parameter_183, parameter_184

        # pd_op.prelu: (128x256x28x28xf32) <- (128x256x28x28xf32, 256xf32)
        prelu_12 = paddle._C_ops.prelu(
            batch_norm__102, parameter_180, "NCHW", "channel"
        )
        del parameter_180

        # pd_op.depthwise_conv2d: (128x256x14x14xf32) <- (128x256x28x28xf32, 256x1x3x3xf32)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(
            prelu_12, parameter_179, [2, 2], [1, 1], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del parameter_179

        # pd_op.batch_norm_: (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_178,
                parameter_177,
                parameter_176,
                parameter_175,
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
        del parameter_175, parameter_176, parameter_177, parameter_178

        # pd_op.prelu: (128x256x14x14xf32) <- (128x256x14x14xf32, 256xf32)
        prelu_13 = paddle._C_ops.prelu(
            batch_norm__108, parameter_174, "NCHW", "channel"
        )
        del parameter_174

        # pd_op.conv2d: (128x128x14x14xf32) <- (128x256x14x14xf32, 128x256x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            prelu_13, parameter_173, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_173

        # pd_op.batch_norm_: (128x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (128x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__114,
            batch_norm__115,
            batch_norm__116,
            batch_norm__117,
            batch_norm__118,
            batch_norm__119,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_12,
                parameter_172,
                parameter_171,
                parameter_170,
                parameter_169,
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
        del parameter_169, parameter_170, parameter_171, parameter_172

        # pd_op.conv2d: (128x256x14x14xf32) <- (128x128x14x14xf32, 256x128x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            batch_norm__114,
            parameter_168,
            [1, 1],
            [0, 0],
            "EXPLICIT",
            [1, 1],
            1,
            "NCHW",
        )
        del parameter_168

        # pd_op.batch_norm_: (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__120,
            batch_norm__121,
            batch_norm__122,
            batch_norm__123,
            batch_norm__124,
            batch_norm__125,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_13,
                parameter_167,
                parameter_166,
                parameter_165,
                parameter_164,
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
        del parameter_164, parameter_165, parameter_166, parameter_167

        # pd_op.prelu: (128x256x14x14xf32) <- (128x256x14x14xf32, 256xf32)
        prelu_14 = paddle._C_ops.prelu(
            batch_norm__120, parameter_163, "NCHW", "channel"
        )
        del parameter_163

        # pd_op.depthwise_conv2d: (128x256x14x14xf32) <- (128x256x14x14xf32, 256x1x3x3xf32)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(
            prelu_14, parameter_162, [1, 1], [1, 1], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del parameter_162

        # pd_op.batch_norm_: (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
        del parameter_158, parameter_159, parameter_160, parameter_161

        # pd_op.prelu: (128x256x14x14xf32) <- (128x256x14x14xf32, 256xf32)
        prelu_15 = paddle._C_ops.prelu(
            batch_norm__126, parameter_157, "NCHW", "channel"
        )
        del parameter_157

        # pd_op.conv2d: (128x128x14x14xf32) <- (128x256x14x14xf32, 128x256x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            prelu_15, parameter_156, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_156

        # pd_op.batch_norm_: (128x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (128x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__132,
            batch_norm__133,
            batch_norm__134,
            batch_norm__135,
            batch_norm__136,
            batch_norm__137,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_14,
                parameter_155,
                parameter_154,
                parameter_153,
                parameter_152,
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
        del parameter_152, parameter_153, parameter_154, parameter_155

        # pd_op.add: (128x128x14x14xf32) <- (128x128x14x14xf32, 128x128x14x14xf32)
        add_4 = paddle._C_ops.add(batch_norm__114, batch_norm__132)

        # pd_op.conv2d: (128x256x14x14xf32) <- (128x128x14x14xf32, 256x128x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            add_4, parameter_151, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_151

        # pd_op.batch_norm_: (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__138,
            batch_norm__139,
            batch_norm__140,
            batch_norm__141,
            batch_norm__142,
            batch_norm__143,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_15,
                parameter_150,
                parameter_149,
                parameter_148,
                parameter_147,
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
        del parameter_147, parameter_148, parameter_149, parameter_150

        # pd_op.prelu: (128x256x14x14xf32) <- (128x256x14x14xf32, 256xf32)
        prelu_16 = paddle._C_ops.prelu(
            batch_norm__138, parameter_146, "NCHW", "channel"
        )
        del parameter_146

        # pd_op.depthwise_conv2d: (128x256x14x14xf32) <- (128x256x14x14xf32, 256x1x3x3xf32)
        depthwise_conv2d_8 = paddle._C_ops.depthwise_conv2d(
            prelu_16, parameter_145, [1, 1], [1, 1], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del parameter_145

        # pd_op.batch_norm_: (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_144,
                parameter_143,
                parameter_142,
                parameter_141,
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
        del parameter_141, parameter_142, parameter_143, parameter_144

        # pd_op.prelu: (128x256x14x14xf32) <- (128x256x14x14xf32, 256xf32)
        prelu_17 = paddle._C_ops.prelu(
            batch_norm__144, parameter_140, "NCHW", "channel"
        )
        del parameter_140

        # pd_op.conv2d: (128x128x14x14xf32) <- (128x256x14x14xf32, 128x256x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            prelu_17, parameter_139, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_139

        # pd_op.batch_norm_: (128x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (128x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__150,
            batch_norm__151,
            batch_norm__152,
            batch_norm__153,
            batch_norm__154,
            batch_norm__155,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_16,
                parameter_138,
                parameter_137,
                parameter_136,
                parameter_135,
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
        del parameter_135, parameter_136, parameter_137, parameter_138

        # pd_op.add: (128x128x14x14xf32) <- (128x128x14x14xf32, 128x128x14x14xf32)
        add_5 = paddle._C_ops.add(add_4, batch_norm__150)

        # pd_op.conv2d: (128x256x14x14xf32) <- (128x128x14x14xf32, 256x128x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            add_5, parameter_134, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_134

        # pd_op.batch_norm_: (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__156,
            batch_norm__157,
            batch_norm__158,
            batch_norm__159,
            batch_norm__160,
            batch_norm__161,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_17,
                parameter_133,
                parameter_132,
                parameter_131,
                parameter_130,
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
        del parameter_130, parameter_131, parameter_132, parameter_133

        # pd_op.prelu: (128x256x14x14xf32) <- (128x256x14x14xf32, 256xf32)
        prelu_18 = paddle._C_ops.prelu(
            batch_norm__156, parameter_129, "NCHW", "channel"
        )
        del parameter_129

        # pd_op.depthwise_conv2d: (128x256x14x14xf32) <- (128x256x14x14xf32, 256x1x3x3xf32)
        depthwise_conv2d_9 = paddle._C_ops.depthwise_conv2d(
            prelu_18, parameter_128, [1, 1], [1, 1], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del parameter_128

        # pd_op.batch_norm_: (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_127,
                parameter_126,
                parameter_125,
                parameter_124,
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
        del parameter_124, parameter_125, parameter_126, parameter_127

        # pd_op.prelu: (128x256x14x14xf32) <- (128x256x14x14xf32, 256xf32)
        prelu_19 = paddle._C_ops.prelu(
            batch_norm__162, parameter_123, "NCHW", "channel"
        )
        del parameter_123

        # pd_op.conv2d: (128x128x14x14xf32) <- (128x256x14x14xf32, 128x256x1x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            prelu_19, parameter_122, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_122

        # pd_op.batch_norm_: (128x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (128x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__168,
            batch_norm__169,
            batch_norm__170,
            batch_norm__171,
            batch_norm__172,
            batch_norm__173,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_18,
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
        del parameter_118, parameter_119, parameter_120, parameter_121

        # pd_op.add: (128x128x14x14xf32) <- (128x128x14x14xf32, 128x128x14x14xf32)
        add_6 = paddle._C_ops.add(add_5, batch_norm__168)

        # pd_op.conv2d: (128x256x14x14xf32) <- (128x128x14x14xf32, 256x128x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            add_6, parameter_117, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_117

        # pd_op.batch_norm_: (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__174,
            batch_norm__175,
            batch_norm__176,
            batch_norm__177,
            batch_norm__178,
            batch_norm__179,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_19,
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
        del parameter_113, parameter_114, parameter_115, parameter_116

        # pd_op.prelu: (128x256x14x14xf32) <- (128x256x14x14xf32, 256xf32)
        prelu_20 = paddle._C_ops.prelu(
            batch_norm__174, parameter_112, "NCHW", "channel"
        )
        del parameter_112

        # pd_op.depthwise_conv2d: (128x256x14x14xf32) <- (128x256x14x14xf32, 256x1x3x3xf32)
        depthwise_conv2d_10 = paddle._C_ops.depthwise_conv2d(
            prelu_20, parameter_111, [1, 1], [1, 1], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del parameter_111

        # pd_op.batch_norm_: (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_110,
                parameter_109,
                parameter_108,
                parameter_107,
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
        del parameter_107, parameter_108, parameter_109, parameter_110

        # pd_op.prelu: (128x256x14x14xf32) <- (128x256x14x14xf32, 256xf32)
        prelu_21 = paddle._C_ops.prelu(
            batch_norm__180, parameter_106, "NCHW", "channel"
        )
        del parameter_106

        # pd_op.conv2d: (128x128x14x14xf32) <- (128x256x14x14xf32, 128x256x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            prelu_21, parameter_105, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_105

        # pd_op.batch_norm_: (128x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (128x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__186,
            batch_norm__187,
            batch_norm__188,
            batch_norm__189,
            batch_norm__190,
            batch_norm__191,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_20,
                parameter_104,
                parameter_103,
                parameter_102,
                parameter_101,
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
        del parameter_101, parameter_102, parameter_103, parameter_104

        # pd_op.add: (128x128x14x14xf32) <- (128x128x14x14xf32, 128x128x14x14xf32)
        add_7 = paddle._C_ops.add(add_6, batch_norm__186)

        # pd_op.conv2d: (128x256x14x14xf32) <- (128x128x14x14xf32, 256x128x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            add_7, parameter_100, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_100

        # pd_op.batch_norm_: (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__192,
            batch_norm__193,
            batch_norm__194,
            batch_norm__195,
            batch_norm__196,
            batch_norm__197,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_21,
                parameter_99,
                parameter_98,
                parameter_97,
                parameter_96,
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
        del parameter_96, parameter_97, parameter_98, parameter_99

        # pd_op.prelu: (128x256x14x14xf32) <- (128x256x14x14xf32, 256xf32)
        prelu_22 = paddle._C_ops.prelu(batch_norm__192, parameter_95, "NCHW", "channel")
        del parameter_95

        # pd_op.depthwise_conv2d: (128x256x14x14xf32) <- (128x256x14x14xf32, 256x1x3x3xf32)
        depthwise_conv2d_11 = paddle._C_ops.depthwise_conv2d(
            prelu_22, parameter_94, [1, 1], [1, 1], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del parameter_94

        # pd_op.batch_norm_: (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_93,
                parameter_92,
                parameter_91,
                parameter_90,
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
        del parameter_90, parameter_91, parameter_92, parameter_93

        # pd_op.prelu: (128x256x14x14xf32) <- (128x256x14x14xf32, 256xf32)
        prelu_23 = paddle._C_ops.prelu(batch_norm__198, parameter_89, "NCHW", "channel")
        del parameter_89

        # pd_op.conv2d: (128x128x14x14xf32) <- (128x256x14x14xf32, 128x256x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(
            prelu_23, parameter_88, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_88

        # pd_op.batch_norm_: (128x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (128x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__204,
            batch_norm__205,
            batch_norm__206,
            batch_norm__207,
            batch_norm__208,
            batch_norm__209,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_22,
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
        del parameter_84, parameter_85, parameter_86, parameter_87

        # pd_op.add: (128x128x14x14xf32) <- (128x128x14x14xf32, 128x128x14x14xf32)
        add_8 = paddle._C_ops.add(add_7, batch_norm__204)

        # pd_op.conv2d: (128x256x14x14xf32) <- (128x128x14x14xf32, 256x128x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(
            add_8, parameter_83, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_83

        # pd_op.batch_norm_: (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__210,
            batch_norm__211,
            batch_norm__212,
            batch_norm__213,
            batch_norm__214,
            batch_norm__215,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_23,
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
        del parameter_79, parameter_80, parameter_81, parameter_82

        # pd_op.prelu: (128x256x14x14xf32) <- (128x256x14x14xf32, 256xf32)
        prelu_24 = paddle._C_ops.prelu(batch_norm__210, parameter_78, "NCHW", "channel")
        del parameter_78

        # pd_op.depthwise_conv2d: (128x256x14x14xf32) <- (128x256x14x14xf32, 256x1x3x3xf32)
        depthwise_conv2d_12 = paddle._C_ops.depthwise_conv2d(
            prelu_24, parameter_77, [1, 1], [1, 1], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del parameter_77

        # pd_op.batch_norm_: (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (128x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_76,
                parameter_75,
                parameter_74,
                parameter_73,
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
        del parameter_73, parameter_74, parameter_75, parameter_76

        # pd_op.prelu: (128x256x14x14xf32) <- (128x256x14x14xf32, 256xf32)
        prelu_25 = paddle._C_ops.prelu(batch_norm__216, parameter_72, "NCHW", "channel")
        del parameter_72

        # pd_op.conv2d: (128x128x14x14xf32) <- (128x256x14x14xf32, 128x256x1x1xf32)
        conv2d_24 = paddle._C_ops.conv2d(
            prelu_25, parameter_71, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_71

        # pd_op.batch_norm_: (128x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (128x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__222,
            batch_norm__223,
            batch_norm__224,
            batch_norm__225,
            batch_norm__226,
            batch_norm__227,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_24,
                parameter_70,
                parameter_69,
                parameter_68,
                parameter_67,
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
        del parameter_67, parameter_68, parameter_69, parameter_70

        # pd_op.add: (128x128x14x14xf32) <- (128x128x14x14xf32, 128x128x14x14xf32)
        add_9 = paddle._C_ops.add(add_8, batch_norm__222)

        # pd_op.conv2d: (128x512x14x14xf32) <- (128x128x14x14xf32, 512x128x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(
            add_9, parameter_66, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_66

        # pd_op.batch_norm_: (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__228,
            batch_norm__229,
            batch_norm__230,
            batch_norm__231,
            batch_norm__232,
            batch_norm__233,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_25,
                parameter_65,
                parameter_64,
                parameter_63,
                parameter_62,
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
        del parameter_62, parameter_63, parameter_64, parameter_65

        # pd_op.prelu: (128x512x14x14xf32) <- (128x512x14x14xf32, 512xf32)
        prelu_26 = paddle._C_ops.prelu(batch_norm__228, parameter_61, "NCHW", "channel")
        del parameter_61

        # pd_op.depthwise_conv2d: (128x512x7x7xf32) <- (128x512x14x14xf32, 512x1x3x3xf32)
        depthwise_conv2d_13 = paddle._C_ops.depthwise_conv2d(
            prelu_26, parameter_60, [2, 2], [1, 1], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del parameter_60

        # pd_op.batch_norm_: (128x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_59,
                parameter_58,
                parameter_57,
                parameter_56,
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
        del parameter_56, parameter_57, parameter_58, parameter_59

        # pd_op.prelu: (128x512x7x7xf32) <- (128x512x7x7xf32, 512xf32)
        prelu_27 = paddle._C_ops.prelu(batch_norm__234, parameter_55, "NCHW", "channel")
        del parameter_55

        # pd_op.conv2d: (128x128x7x7xf32) <- (128x512x7x7xf32, 128x512x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(
            prelu_27, parameter_54, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_54

        # pd_op.batch_norm_: (128x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (128x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__240,
            batch_norm__241,
            batch_norm__242,
            batch_norm__243,
            batch_norm__244,
            batch_norm__245,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_26,
                parameter_53,
                parameter_52,
                parameter_51,
                parameter_50,
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
        del parameter_50, parameter_51, parameter_52, parameter_53

        # pd_op.conv2d: (128x256x7x7xf32) <- (128x128x7x7xf32, 256x128x1x1xf32)
        conv2d_27 = paddle._C_ops.conv2d(
            batch_norm__240, parameter_49, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_49

        # pd_op.batch_norm_: (128x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (128x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__246,
            batch_norm__247,
            batch_norm__248,
            batch_norm__249,
            batch_norm__250,
            batch_norm__251,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_27,
                parameter_48,
                parameter_47,
                parameter_46,
                parameter_45,
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
        del parameter_45, parameter_46, parameter_47, parameter_48

        # pd_op.prelu: (128x256x7x7xf32) <- (128x256x7x7xf32, 256xf32)
        prelu_28 = paddle._C_ops.prelu(batch_norm__246, parameter_44, "NCHW", "channel")
        del parameter_44

        # pd_op.depthwise_conv2d: (128x256x7x7xf32) <- (128x256x7x7xf32, 256x1x3x3xf32)
        depthwise_conv2d_14 = paddle._C_ops.depthwise_conv2d(
            prelu_28, parameter_43, [1, 1], [1, 1], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del parameter_43

        # pd_op.batch_norm_: (128x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (128x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_42,
                parameter_41,
                parameter_40,
                parameter_39,
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
        del parameter_39, parameter_40, parameter_41, parameter_42

        # pd_op.prelu: (128x256x7x7xf32) <- (128x256x7x7xf32, 256xf32)
        prelu_29 = paddle._C_ops.prelu(batch_norm__252, parameter_38, "NCHW", "channel")
        del parameter_38

        # pd_op.conv2d: (128x128x7x7xf32) <- (128x256x7x7xf32, 128x256x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(
            prelu_29, parameter_37, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_37

        # pd_op.batch_norm_: (128x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (128x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__258,
            batch_norm__259,
            batch_norm__260,
            batch_norm__261,
            batch_norm__262,
            batch_norm__263,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_28,
                parameter_36,
                parameter_35,
                parameter_34,
                parameter_33,
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
        del parameter_33, parameter_34, parameter_35, parameter_36

        # pd_op.add: (128x128x7x7xf32) <- (128x128x7x7xf32, 128x128x7x7xf32)
        add_10 = paddle._C_ops.add(batch_norm__240, batch_norm__258)

        # pd_op.conv2d: (128x256x7x7xf32) <- (128x128x7x7xf32, 256x128x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(
            add_10, parameter_32, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_32

        # pd_op.batch_norm_: (128x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (128x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__264,
            batch_norm__265,
            batch_norm__266,
            batch_norm__267,
            batch_norm__268,
            batch_norm__269,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_29,
                parameter_31,
                parameter_30,
                parameter_29,
                parameter_28,
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
        del parameter_28, parameter_29, parameter_30, parameter_31

        # pd_op.prelu: (128x256x7x7xf32) <- (128x256x7x7xf32, 256xf32)
        prelu_30 = paddle._C_ops.prelu(batch_norm__264, parameter_27, "NCHW", "channel")
        del parameter_27

        # pd_op.depthwise_conv2d: (128x256x7x7xf32) <- (128x256x7x7xf32, 256x1x3x3xf32)
        depthwise_conv2d_15 = paddle._C_ops.depthwise_conv2d(
            prelu_30, parameter_26, [1, 1], [1, 1], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del parameter_26

        # pd_op.batch_norm_: (128x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (128x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__270,
            batch_norm__271,
            batch_norm__272,
            batch_norm__273,
            batch_norm__274,
            batch_norm__275,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_15,
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
        del parameter_22, parameter_23, parameter_24, parameter_25

        # pd_op.prelu: (128x256x7x7xf32) <- (128x256x7x7xf32, 256xf32)
        prelu_31 = paddle._C_ops.prelu(batch_norm__270, parameter_21, "NCHW", "channel")
        del parameter_21

        # pd_op.conv2d: (128x128x7x7xf32) <- (128x256x7x7xf32, 128x256x1x1xf32)
        conv2d_30 = paddle._C_ops.conv2d(
            prelu_31, parameter_20, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_20

        # pd_op.batch_norm_: (128x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (128x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__276,
            batch_norm__277,
            batch_norm__278,
            batch_norm__279,
            batch_norm__280,
            batch_norm__281,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_30,
                parameter_19,
                parameter_18,
                parameter_17,
                parameter_16,
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
        del parameter_16, parameter_17, parameter_18, parameter_19

        # pd_op.add: (128x128x7x7xf32) <- (128x128x7x7xf32, 128x128x7x7xf32)
        add_11 = paddle._C_ops.add(add_10, batch_norm__276)

        # pd_op.conv2d: (128x512x7x7xf32) <- (128x128x7x7xf32, 512x128x1x1xf32)
        conv2d_31 = paddle._C_ops.conv2d(
            add_11, parameter_15, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_15

        # pd_op.batch_norm_: (128x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__282,
            batch_norm__283,
            batch_norm__284,
            batch_norm__285,
            batch_norm__286,
            batch_norm__287,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_31,
                parameter_14,
                parameter_13,
                parameter_12,
                parameter_11,
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
        del parameter_11, parameter_12, parameter_13, parameter_14

        # pd_op.prelu: (128x512x7x7xf32) <- (128x512x7x7xf32, 512xf32)
        prelu_32 = paddle._C_ops.prelu(batch_norm__282, parameter_10, "NCHW", "channel")
        del parameter_10

        # pd_op.depthwise_conv2d: (128x512x1x1xf32) <- (128x512x7x7xf32, 512x1x7x7xf32)
        depthwise_conv2d_16 = paddle._C_ops.depthwise_conv2d(
            prelu_32, parameter_9, [1, 1], [0, 0], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del parameter_9

        # pd_op.batch_norm_: (128x512x1x1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x1x1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__288,
            batch_norm__289,
            batch_norm__290,
            batch_norm__291,
            batch_norm__292,
            batch_norm__293,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_16,
                parameter_8,
                parameter_7,
                parameter_6,
                parameter_5,
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
        del parameter_5, parameter_6, parameter_7, parameter_8

        # pd_op.conv2d: (128x128x1x1xf32) <- (128x512x1x1xf32, 128x512x1x1xf32)
        conv2d_32 = paddle._C_ops.conv2d(
            batch_norm__288, parameter_4, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_4

        # pd_op.batch_norm_: (128x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (128x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__294,
            batch_norm__295,
            batch_norm__296,
            batch_norm__297,
            batch_norm__298,
            batch_norm__299,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_32,
                parameter_3,
                parameter_2,
                parameter_1,
                parameter_0,
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
        del parameter_0, parameter_1, parameter_2, parameter_3

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [128, 128]

        # pd_op.reshape: (128x128xf32) <- (128x128x1x1xf32, 2xi64)
        reshape_0 = paddle._C_ops.reshape(batch_norm__294, full_int_array_0)
        del full_int_array_0

        # pd_op.square: (128x128xf32) <- (128x128xf32)
        square_0 = paddle._C_ops.square(reshape_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.sum: (128x1xf32) <- (128x128xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(square_0, full_int_array_1, None, True)

        # pd_op.sqrt: (128x1xf32) <- (128x1xf32)
        sqrt_0 = paddle._C_ops.sqrt(sum_0)
        del sum_0

        # pd_op.divide: (128x128xf32) <- (128x128xf32, 128x1xf32)
        divide_0 = paddle._C_ops.divide(reshape_0, sqrt_0)

        # pd_op.square: (128x93431xf32) <- (128x93431xf32)
        square_1 = paddle._C_ops.square(data_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.sum: (1x93431xf32) <- (128x93431xf32, 1xi64)
        sum_1 = paddle._C_ops.sum(square_1, full_int_array_2, None, True)

        # pd_op.sqrt: (1x93431xf32) <- (1x93431xf32)
        sqrt_1 = paddle._C_ops.sqrt(sum_1)
        del sum_1

        # pd_op.divide: (128x93431xf32) <- (128x93431xf32, 1x93431xf32)
        divide_1 = paddle._C_ops.divide(data_0, sqrt_1)
        del data_0

        # pd_op.matmul: (128x93431xf32) <- (128x128xf32, 128x93431xf32)
        matmul_0 = paddle._C_ops.matmul(divide_0, divide_1, False, False)

        # pd_op.square: (128x93431xf32) <- (128x93431xf32)
        square_2 = paddle._C_ops.square(matmul_0)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (128x93431xf32) <- (128x93431xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(square_2, full_0, float("1"), True)
        del square_2

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_0 = full_1

        # pd_op.scale: (128x93431xf32) <- (128x93431xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(scale_1, full_1, float("1e-06"), True)
        del scale_1

        # pd_op.sqrt: (128x93431xf32) <- (128x93431xf32)
        sqrt_2 = paddle._C_ops.sqrt(scale_2)
        del scale_2

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0.877583"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (128x93431xf32) <- (128x93431xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(matmul_0, full_2, float("0"), True)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0.479426"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (128x93431xf32) <- (128x93431xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(sqrt_2, full_3, float("0"), True)

        # pd_op.subtract: (128x93431xf32) <- (128x93431xf32, 128x93431xf32)
        subtract_0 = paddle._C_ops.subtract(scale_3, scale_4)

        # pd_op.scale: (128x93431xf32) <- (128x93431xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(matmul_0, full_1, float("-0.239713"), True)

        # pd_op.full: (xf32) <- ()
        full_4 = paddle._C_ops.full(
            [],
            float("-0.877583"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.greater_than: (128x93431xb) <- (128x93431xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(matmul_0, full_4)
        del full_4

        # pd_op.cast: (128x93431xf32) <- (128x93431xb)
        cast_0 = paddle._C_ops.cast(greater_than_0, paddle.float32)
        del greater_than_0

        # pd_op.multiply: (128x93431xf32) <- (128x93431xf32, 128x93431xf32)
        multiply_0 = paddle._C_ops.multiply(cast_0, subtract_0)

        # pd_op.scale: (128x93431xf32) <- (128x93431xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(cast_0, full_0, float("1"), True)

        # pd_op.multiply: (128x93431xf32) <- (128x93431xf32, 128x93431xf32)
        multiply_1 = paddle._C_ops.multiply(scale_6, scale_5)

        # pd_op.add: (128x93431xf32) <- (128x93431xf32, 128x93431xf32)
        add_12 = paddle._C_ops.add(multiply_0, multiply_1)

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("93431"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.one_hot: (128x1x93431xf32) <- (128x1xi64, 1xi32)
        one_hot_0 = paddle._C_ops.one_hot(
            data_2 % paddle.cast(full_5, data_2.dtype), full_5
        )
        del data_2, full_5

        # pd_op.squeeze: (128x93431xf32) <- (128x1x93431xf32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(one_hot_0, full_int_array_1)
        del one_hot_0

        # pd_op.multiply: (128x93431xf32) <- (128x93431xf32, 128x93431xf32)
        multiply_2 = paddle._C_ops.multiply(squeeze_0, add_12)

        # pd_op.scale: (128x93431xf32) <- (128x93431xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(squeeze_0, full_0, float("1"), True)

        # pd_op.multiply: (128x93431xf32) <- (128x93431xf32, 128x93431xf32)
        multiply_3 = paddle._C_ops.multiply(scale_7, matmul_0)

        # pd_op.add: (128x93431xf32) <- (128x93431xf32, 128x93431xf32)
        add_13 = paddle._C_ops.add(multiply_2, multiply_3)

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("64"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (128x93431xf32) <- (128x93431xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(add_13, full_6, float("0"), True)
        del (
            add_0,
            add_1,
            add_10,
            add_11,
            add_12,
            add_13,
            add_2,
            add_3,
            add_4,
            add_5,
            add_6,
            add_7,
            add_8,
            add_9,
            assign_0,
            batch_norm__0,
            batch_norm__1,
            batch_norm__10,
            batch_norm__100,
            batch_norm__101,
            batch_norm__102,
            batch_norm__103,
            batch_norm__104,
            batch_norm__105,
            batch_norm__106,
            batch_norm__107,
            batch_norm__108,
            batch_norm__109,
            batch_norm__11,
            batch_norm__110,
            batch_norm__111,
            batch_norm__112,
            batch_norm__113,
            batch_norm__114,
            batch_norm__115,
            batch_norm__116,
            batch_norm__117,
            batch_norm__118,
            batch_norm__119,
            batch_norm__12,
            batch_norm__120,
            batch_norm__121,
            batch_norm__122,
            batch_norm__123,
            batch_norm__124,
            batch_norm__125,
            batch_norm__126,
            batch_norm__127,
            batch_norm__128,
            batch_norm__129,
            batch_norm__13,
            batch_norm__130,
            batch_norm__131,
            batch_norm__132,
            batch_norm__133,
            batch_norm__134,
            batch_norm__135,
            batch_norm__136,
            batch_norm__137,
            batch_norm__138,
            batch_norm__139,
            batch_norm__14,
            batch_norm__140,
            batch_norm__141,
            batch_norm__142,
            batch_norm__143,
            batch_norm__144,
            batch_norm__145,
            batch_norm__146,
            batch_norm__147,
            batch_norm__148,
            batch_norm__149,
            batch_norm__15,
            batch_norm__150,
            batch_norm__151,
            batch_norm__152,
            batch_norm__153,
            batch_norm__154,
            batch_norm__155,
            batch_norm__156,
            batch_norm__157,
            batch_norm__158,
            batch_norm__159,
            batch_norm__16,
            batch_norm__160,
            batch_norm__161,
            batch_norm__162,
            batch_norm__163,
            batch_norm__164,
            batch_norm__165,
            batch_norm__166,
            batch_norm__167,
            batch_norm__168,
            batch_norm__169,
            batch_norm__17,
            batch_norm__170,
            batch_norm__171,
            batch_norm__172,
            batch_norm__173,
            batch_norm__174,
            batch_norm__175,
            batch_norm__176,
            batch_norm__177,
            batch_norm__178,
            batch_norm__179,
            batch_norm__18,
            batch_norm__180,
            batch_norm__181,
            batch_norm__182,
            batch_norm__183,
            batch_norm__184,
            batch_norm__185,
            batch_norm__186,
            batch_norm__187,
            batch_norm__188,
            batch_norm__189,
            batch_norm__19,
            batch_norm__190,
            batch_norm__191,
            batch_norm__192,
            batch_norm__193,
            batch_norm__194,
            batch_norm__195,
            batch_norm__196,
            batch_norm__197,
            batch_norm__198,
            batch_norm__199,
            batch_norm__2,
            batch_norm__20,
            batch_norm__200,
            batch_norm__201,
            batch_norm__202,
            batch_norm__203,
            batch_norm__204,
            batch_norm__205,
            batch_norm__206,
            batch_norm__207,
            batch_norm__208,
            batch_norm__209,
            batch_norm__21,
            batch_norm__210,
            batch_norm__211,
            batch_norm__212,
            batch_norm__213,
            batch_norm__214,
            batch_norm__215,
            batch_norm__216,
            batch_norm__217,
            batch_norm__218,
            batch_norm__219,
            batch_norm__22,
            batch_norm__220,
            batch_norm__221,
            batch_norm__222,
            batch_norm__223,
            batch_norm__224,
            batch_norm__225,
            batch_norm__226,
            batch_norm__227,
            batch_norm__228,
            batch_norm__229,
            batch_norm__23,
            batch_norm__230,
            batch_norm__231,
            batch_norm__232,
            batch_norm__233,
            batch_norm__234,
            batch_norm__235,
            batch_norm__236,
            batch_norm__237,
            batch_norm__238,
            batch_norm__239,
            batch_norm__24,
            batch_norm__240,
            batch_norm__241,
            batch_norm__242,
            batch_norm__243,
            batch_norm__244,
            batch_norm__245,
            batch_norm__246,
            batch_norm__247,
            batch_norm__248,
            batch_norm__249,
            batch_norm__25,
            batch_norm__250,
            batch_norm__251,
            batch_norm__252,
            batch_norm__253,
            batch_norm__254,
            batch_norm__255,
            batch_norm__256,
            batch_norm__257,
            batch_norm__258,
            batch_norm__259,
            batch_norm__26,
            batch_norm__260,
            batch_norm__261,
            batch_norm__262,
            batch_norm__263,
            batch_norm__264,
            batch_norm__265,
            batch_norm__266,
            batch_norm__267,
            batch_norm__268,
            batch_norm__269,
            batch_norm__27,
            batch_norm__270,
            batch_norm__271,
            batch_norm__272,
            batch_norm__273,
            batch_norm__274,
            batch_norm__275,
            batch_norm__276,
            batch_norm__277,
            batch_norm__278,
            batch_norm__279,
            batch_norm__28,
            batch_norm__280,
            batch_norm__281,
            batch_norm__282,
            batch_norm__283,
            batch_norm__284,
            batch_norm__285,
            batch_norm__286,
            batch_norm__287,
            batch_norm__288,
            batch_norm__289,
            batch_norm__29,
            batch_norm__290,
            batch_norm__291,
            batch_norm__292,
            batch_norm__293,
            batch_norm__294,
            batch_norm__295,
            batch_norm__296,
            batch_norm__297,
            batch_norm__298,
            batch_norm__299,
            batch_norm__3,
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
            batch_norm__36,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__4,
            batch_norm__40,
            batch_norm__41,
            batch_norm__42,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
            batch_norm__48,
            batch_norm__49,
            batch_norm__5,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
            batch_norm__6,
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
            batch_norm__66,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            batch_norm__7,
            batch_norm__70,
            batch_norm__71,
            batch_norm__72,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
            batch_norm__78,
            batch_norm__79,
            batch_norm__8,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
            batch_norm__83,
            batch_norm__84,
            batch_norm__85,
            batch_norm__86,
            batch_norm__87,
            batch_norm__88,
            batch_norm__89,
            batch_norm__9,
            batch_norm__90,
            batch_norm__91,
            batch_norm__92,
            batch_norm__93,
            batch_norm__94,
            batch_norm__95,
            batch_norm__96,
            batch_norm__97,
            batch_norm__98,
            batch_norm__99,
            cast_0,
            conv2d_0,
            conv2d_1,
            conv2d_10,
            conv2d_11,
            conv2d_12,
            conv2d_13,
            conv2d_14,
            conv2d_15,
            conv2d_16,
            conv2d_17,
            conv2d_18,
            conv2d_19,
            conv2d_2,
            conv2d_20,
            conv2d_21,
            conv2d_22,
            conv2d_23,
            conv2d_24,
            conv2d_25,
            conv2d_26,
            conv2d_27,
            conv2d_28,
            conv2d_29,
            conv2d_3,
            conv2d_30,
            conv2d_31,
            conv2d_32,
            conv2d_4,
            conv2d_5,
            conv2d_6,
            conv2d_7,
            conv2d_8,
            conv2d_9,
            depthwise_conv2d_0,
            depthwise_conv2d_1,
            depthwise_conv2d_10,
            depthwise_conv2d_11,
            depthwise_conv2d_12,
            depthwise_conv2d_13,
            depthwise_conv2d_14,
            depthwise_conv2d_15,
            depthwise_conv2d_16,
            depthwise_conv2d_2,
            depthwise_conv2d_3,
            depthwise_conv2d_4,
            depthwise_conv2d_5,
            depthwise_conv2d_6,
            depthwise_conv2d_7,
            depthwise_conv2d_8,
            depthwise_conv2d_9,
            divide_0,
            divide_1,
            full_0,
            full_1,
            full_2,
            full_3,
            full_6,
            full_int_array_1,
            full_int_array_2,
            matmul_0,
            multiply_0,
            multiply_1,
            multiply_2,
            multiply_3,
            prelu_0,
            prelu_1,
            prelu_10,
            prelu_11,
            prelu_12,
            prelu_13,
            prelu_14,
            prelu_15,
            prelu_16,
            prelu_17,
            prelu_18,
            prelu_19,
            prelu_2,
            prelu_20,
            prelu_21,
            prelu_22,
            prelu_23,
            prelu_24,
            prelu_25,
            prelu_26,
            prelu_27,
            prelu_28,
            prelu_29,
            prelu_3,
            prelu_30,
            prelu_31,
            prelu_32,
            prelu_4,
            prelu_5,
            prelu_6,
            prelu_7,
            prelu_8,
            prelu_9,
            reshape_0,
            scale_3,
            scale_4,
            scale_5,
            scale_6,
            scale_7,
            sqrt_0,
            sqrt_1,
            sqrt_2,
            square_0,
            square_1,
            squeeze_0,
            subtract_0,
        )

        return scale_0
