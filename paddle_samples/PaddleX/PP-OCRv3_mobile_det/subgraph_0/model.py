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
        parameter_283,
        parameter_284,
        parameter_285,
        parameter_286,
        parameter_287,
        parameter_288,
        parameter_289,
        parameter_290,
        parameter_291,
        parameter_292,
        parameter_293,
        parameter_294,
        parameter_295,
        parameter_296,
        parameter_297,
        parameter_298,
        parameter_299,
        parameter_300,
        data_0,
    ):
        # pd_op.conv2d: (4x8x480x480xf32) <- (4x3x960x960xf32, 8x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_300, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_0, parameter_300

        # pd_op.batch_norm_: (4x8x480x480xf32, 8xf32, 8xf32, 8xf32, 8xf32, -1xui8) <- (4x8x480x480xf32, 8xf32, 8xf32, 8xf32, 8xf32)
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
                parameter_299,
                parameter_298,
                parameter_297,
                parameter_296,
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
        del parameter_296, parameter_297, parameter_298, parameter_299

        # pd_op.hardswish: (4x8x480x480xf32) <- (4x8x480x480xf32)
        hardswish_0 = paddle._C_ops.hardswish(batch_norm__0)

        # pd_op.conv2d: (4x8x480x480xf32) <- (4x8x480x480xf32, 8x8x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            hardswish_0, parameter_295, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_295

        # pd_op.batch_norm_: (4x8x480x480xf32, 8xf32, 8xf32, 8xf32, 8xf32, -1xui8) <- (4x8x480x480xf32, 8xf32, 8xf32, 8xf32, 8xf32)
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
                parameter_294,
                parameter_293,
                parameter_292,
                parameter_291,
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
        del parameter_291, parameter_292, parameter_293, parameter_294

        # pd_op.relu: (4x8x480x480xf32) <- (4x8x480x480xf32)
        relu_0 = paddle._C_ops.relu(batch_norm__6)
        del batch_norm__6

        # pd_op.depthwise_conv2d: (4x8x480x480xf32) <- (4x8x480x480xf32, 8x1x3x3xf32)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(
            relu_0, parameter_290, [1, 1], [1, 1], "EXPLICIT", 8, [1, 1], "NCHW"
        )
        del parameter_290

        # pd_op.batch_norm_: (4x8x480x480xf32, 8xf32, 8xf32, 8xf32, 8xf32, -1xui8) <- (4x8x480x480xf32, 8xf32, 8xf32, 8xf32, 8xf32)
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
                parameter_289,
                parameter_288,
                parameter_287,
                parameter_286,
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
        del parameter_286, parameter_287, parameter_288, parameter_289

        # pd_op.relu: (4x8x480x480xf32) <- (4x8x480x480xf32)
        relu_1 = paddle._C_ops.relu(batch_norm__12)
        del batch_norm__12

        # pd_op.conv2d: (4x8x480x480xf32) <- (4x8x480x480xf32, 8x8x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            relu_1, parameter_285, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_285

        # pd_op.batch_norm_: (4x8x480x480xf32, 8xf32, 8xf32, 8xf32, 8xf32, -1xui8) <- (4x8x480x480xf32, 8xf32, 8xf32, 8xf32, 8xf32)
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
                parameter_284,
                parameter_283,
                parameter_282,
                parameter_281,
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
        del parameter_281, parameter_282, parameter_283, parameter_284

        # pd_op.add: (4x8x480x480xf32) <- (4x8x480x480xf32, 4x8x480x480xf32)
        add_0 = paddle._C_ops.add(hardswish_0, batch_norm__18)

        # pd_op.conv2d: (4x32x480x480xf32) <- (4x8x480x480xf32, 32x8x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            add_0, parameter_280, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_280

        # pd_op.batch_norm_: (4x32x480x480xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (4x32x480x480xf32, 32xf32, 32xf32, 32xf32, 32xf32)
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
                parameter_279,
                parameter_278,
                parameter_277,
                parameter_276,
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
        del parameter_276, parameter_277, parameter_278, parameter_279

        # pd_op.relu: (4x32x480x480xf32) <- (4x32x480x480xf32)
        relu_2 = paddle._C_ops.relu(batch_norm__24)
        del batch_norm__24

        # pd_op.depthwise_conv2d: (4x32x240x240xf32) <- (4x32x480x480xf32, 32x1x3x3xf32)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(
            relu_2, parameter_275, [2, 2], [1, 1], "EXPLICIT", 32, [1, 1], "NCHW"
        )
        del parameter_275

        # pd_op.batch_norm_: (4x32x240x240xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (4x32x240x240xf32, 32xf32, 32xf32, 32xf32, 32xf32)
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
                parameter_274,
                parameter_273,
                parameter_272,
                parameter_271,
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
        del parameter_271, parameter_272, parameter_273, parameter_274

        # pd_op.relu: (4x32x240x240xf32) <- (4x32x240x240xf32)
        relu_3 = paddle._C_ops.relu(batch_norm__30)
        del batch_norm__30

        # pd_op.conv2d: (4x16x240x240xf32) <- (4x32x240x240xf32, 16x32x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            relu_3, parameter_270, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_270

        # pd_op.batch_norm_: (4x16x240x240xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (4x16x240x240xf32, 16xf32, 16xf32, 16xf32, 16xf32)
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

        # pd_op.conv2d: (4x40x240x240xf32) <- (4x16x240x240xf32, 40x16x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            batch_norm__36, parameter_265, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_265

        # pd_op.batch_norm_: (4x40x240x240xf32, 40xf32, 40xf32, 40xf32, 40xf32, -1xui8) <- (4x40x240x240xf32, 40xf32, 40xf32, 40xf32, 40xf32)
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
                parameter_264,
                parameter_263,
                parameter_262,
                parameter_261,
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
        del parameter_261, parameter_262, parameter_263, parameter_264

        # pd_op.relu: (4x40x240x240xf32) <- (4x40x240x240xf32)
        relu_4 = paddle._C_ops.relu(batch_norm__42)
        del batch_norm__42

        # pd_op.depthwise_conv2d: (4x40x240x240xf32) <- (4x40x240x240xf32, 40x1x3x3xf32)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(
            relu_4, parameter_260, [1, 1], [1, 1], "EXPLICIT", 40, [1, 1], "NCHW"
        )
        del parameter_260

        # pd_op.batch_norm_: (4x40x240x240xf32, 40xf32, 40xf32, 40xf32, 40xf32, -1xui8) <- (4x40x240x240xf32, 40xf32, 40xf32, 40xf32, 40xf32)
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
                parameter_259,
                parameter_258,
                parameter_257,
                parameter_256,
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
        del parameter_256, parameter_257, parameter_258, parameter_259

        # pd_op.relu: (4x40x240x240xf32) <- (4x40x240x240xf32)
        relu_5 = paddle._C_ops.relu(batch_norm__48)
        del batch_norm__48

        # pd_op.conv2d: (4x16x240x240xf32) <- (4x40x240x240xf32, 16x40x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            relu_5, parameter_255, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_255

        # pd_op.batch_norm_: (4x16x240x240xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (4x16x240x240xf32, 16xf32, 16xf32, 16xf32, 16xf32)
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
                parameter_254,
                parameter_253,
                parameter_252,
                parameter_251,
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
        del parameter_251, parameter_252, parameter_253, parameter_254

        # pd_op.add: (4x16x240x240xf32) <- (4x16x240x240xf32, 4x16x240x240xf32)
        add_1 = paddle._C_ops.add(batch_norm__36, batch_norm__54)

        # pd_op.conv2d: (4x40x240x240xf32) <- (4x16x240x240xf32, 40x16x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            add_1, parameter_250, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_250

        # pd_op.batch_norm_: (4x40x240x240xf32, 40xf32, 40xf32, 40xf32, 40xf32, -1xui8) <- (4x40x240x240xf32, 40xf32, 40xf32, 40xf32, 40xf32)
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
                parameter_249,
                parameter_248,
                parameter_247,
                parameter_246,
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
        del parameter_246, parameter_247, parameter_248, parameter_249

        # pd_op.relu: (4x40x240x240xf32) <- (4x40x240x240xf32)
        relu_6 = paddle._C_ops.relu(batch_norm__60)
        del batch_norm__60

        # pd_op.depthwise_conv2d: (4x40x120x120xf32) <- (4x40x240x240xf32, 40x1x5x5xf32)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(
            relu_6, parameter_245, [2, 2], [2, 2], "EXPLICIT", 40, [1, 1], "NCHW"
        )
        del parameter_245

        # pd_op.batch_norm_: (4x40x120x120xf32, 40xf32, 40xf32, 40xf32, 40xf32, -1xui8) <- (4x40x120x120xf32, 40xf32, 40xf32, 40xf32, 40xf32)
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
                parameter_244,
                parameter_243,
                parameter_242,
                parameter_241,
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
        del parameter_241, parameter_242, parameter_243, parameter_244

        # pd_op.relu: (4x40x120x120xf32) <- (4x40x120x120xf32)
        relu_7 = paddle._C_ops.relu(batch_norm__66)
        del batch_norm__66

        # pd_op.conv2d: (4x24x120x120xf32) <- (4x40x120x120xf32, 24x40x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            relu_7, parameter_240, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_240

        # pd_op.batch_norm_: (4x24x120x120xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (4x24x120x120xf32, 24xf32, 24xf32, 24xf32, 24xf32)
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
                parameter_239,
                parameter_238,
                parameter_237,
                parameter_236,
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
        del parameter_236, parameter_237, parameter_238, parameter_239

        # pd_op.conv2d: (4x64x120x120xf32) <- (4x24x120x120xf32, 64x24x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            batch_norm__72, parameter_235, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_235

        # pd_op.batch_norm_: (4x64x120x120xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x120x120xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_234,
                parameter_233,
                parameter_232,
                parameter_231,
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
        del parameter_231, parameter_232, parameter_233, parameter_234

        # pd_op.relu: (4x64x120x120xf32) <- (4x64x120x120xf32)
        relu_8 = paddle._C_ops.relu(batch_norm__78)
        del batch_norm__78

        # pd_op.depthwise_conv2d: (4x64x120x120xf32) <- (4x64x120x120xf32, 64x1x5x5xf32)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(
            relu_8, parameter_230, [1, 1], [2, 2], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del parameter_230

        # pd_op.batch_norm_: (4x64x120x120xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x120x120xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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

        # pd_op.relu: (4x64x120x120xf32) <- (4x64x120x120xf32)
        relu_9 = paddle._C_ops.relu(batch_norm__84)
        del batch_norm__84

        # pd_op.conv2d: (4x24x120x120xf32) <- (4x64x120x120xf32, 24x64x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            relu_9, parameter_225, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_225

        # pd_op.batch_norm_: (4x24x120x120xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (4x24x120x120xf32, 24xf32, 24xf32, 24xf32, 24xf32)
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
                parameter_224,
                parameter_223,
                parameter_222,
                parameter_221,
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
        del parameter_221, parameter_222, parameter_223, parameter_224

        # pd_op.add: (4x24x120x120xf32) <- (4x24x120x120xf32, 4x24x120x120xf32)
        add_2 = paddle._C_ops.add(batch_norm__72, batch_norm__90)

        # pd_op.conv2d: (4x64x120x120xf32) <- (4x24x120x120xf32, 64x24x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            add_2, parameter_220, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_220

        # pd_op.batch_norm_: (4x64x120x120xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x120x120xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_219,
                parameter_218,
                parameter_217,
                parameter_216,
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
        del parameter_216, parameter_217, parameter_218, parameter_219

        # pd_op.relu: (4x64x120x120xf32) <- (4x64x120x120xf32)
        relu_10 = paddle._C_ops.relu(batch_norm__96)
        del batch_norm__96

        # pd_op.depthwise_conv2d: (4x64x120x120xf32) <- (4x64x120x120xf32, 64x1x5x5xf32)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(
            relu_10, parameter_215, [1, 1], [2, 2], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del parameter_215

        # pd_op.batch_norm_: (4x64x120x120xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x120x120xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_214,
                parameter_213,
                parameter_212,
                parameter_211,
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
        del parameter_211, parameter_212, parameter_213, parameter_214

        # pd_op.relu: (4x64x120x120xf32) <- (4x64x120x120xf32)
        relu_11 = paddle._C_ops.relu(batch_norm__102)
        del batch_norm__102

        # pd_op.conv2d: (4x24x120x120xf32) <- (4x64x120x120xf32, 24x64x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            relu_11, parameter_210, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_210

        # pd_op.batch_norm_: (4x24x120x120xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (4x24x120x120xf32, 24xf32, 24xf32, 24xf32, 24xf32)
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
                parameter_209,
                parameter_208,
                parameter_207,
                parameter_206,
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
        del parameter_206, parameter_207, parameter_208, parameter_209

        # pd_op.add: (4x24x120x120xf32) <- (4x24x120x120xf32, 4x24x120x120xf32)
        add_3 = paddle._C_ops.add(add_2, batch_norm__108)

        # pd_op.conv2d: (4x120x120x120xf32) <- (4x24x120x120xf32, 120x24x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            add_3, parameter_205, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_205

        # pd_op.batch_norm_: (4x120x120x120xf32, 120xf32, 120xf32, 120xf32, 120xf32, -1xui8) <- (4x120x120x120xf32, 120xf32, 120xf32, 120xf32, 120xf32)
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
        del parameter_201, parameter_202, parameter_203, parameter_204

        # pd_op.hardswish: (4x120x120x120xf32) <- (4x120x120x120xf32)
        hardswish_1 = paddle._C_ops.hardswish(batch_norm__114)

        # pd_op.depthwise_conv2d: (4x120x60x60xf32) <- (4x120x120x120xf32, 120x1x3x3xf32)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(
            hardswish_1, parameter_200, [2, 2], [1, 1], "EXPLICIT", 120, [1, 1], "NCHW"
        )
        del parameter_200

        # pd_op.batch_norm_: (4x120x60x60xf32, 120xf32, 120xf32, 120xf32, 120xf32, -1xui8) <- (4x120x60x60xf32, 120xf32, 120xf32, 120xf32, 120xf32)
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
        del parameter_196, parameter_197, parameter_198, parameter_199

        # pd_op.hardswish: (4x120x60x60xf32) <- (4x120x60x60xf32)
        hardswish_2 = paddle._C_ops.hardswish(batch_norm__120)

        # pd_op.conv2d: (4x40x60x60xf32) <- (4x120x60x60xf32, 40x120x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            hardswish_2, parameter_195, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_195

        # pd_op.batch_norm_: (4x40x60x60xf32, 40xf32, 40xf32, 40xf32, 40xf32, -1xui8) <- (4x40x60x60xf32, 40xf32, 40xf32, 40xf32, 40xf32)
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
        del parameter_191, parameter_192, parameter_193, parameter_194

        # pd_op.conv2d: (4x104x60x60xf32) <- (4x40x60x60xf32, 104x40x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            batch_norm__126,
            parameter_190,
            [1, 1],
            [0, 0],
            "EXPLICIT",
            [1, 1],
            1,
            "NCHW",
        )
        del parameter_190

        # pd_op.batch_norm_: (4x104x60x60xf32, 104xf32, 104xf32, 104xf32, 104xf32, -1xui8) <- (4x104x60x60xf32, 104xf32, 104xf32, 104xf32, 104xf32)
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

        # pd_op.hardswish: (4x104x60x60xf32) <- (4x104x60x60xf32)
        hardswish_3 = paddle._C_ops.hardswish(batch_norm__132)

        # pd_op.depthwise_conv2d: (4x104x60x60xf32) <- (4x104x60x60xf32, 104x1x3x3xf32)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(
            hardswish_3, parameter_185, [1, 1], [1, 1], "EXPLICIT", 104, [1, 1], "NCHW"
        )
        del parameter_185

        # pd_op.batch_norm_: (4x104x60x60xf32, 104xf32, 104xf32, 104xf32, 104xf32, -1xui8) <- (4x104x60x60xf32, 104xf32, 104xf32, 104xf32, 104xf32)
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

        # pd_op.hardswish: (4x104x60x60xf32) <- (4x104x60x60xf32)
        hardswish_4 = paddle._C_ops.hardswish(batch_norm__138)

        # pd_op.conv2d: (4x40x60x60xf32) <- (4x104x60x60xf32, 40x104x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            hardswish_4, parameter_180, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_180

        # pd_op.batch_norm_: (4x40x60x60xf32, 40xf32, 40xf32, 40xf32, 40xf32, -1xui8) <- (4x40x60x60xf32, 40xf32, 40xf32, 40xf32, 40xf32)
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
                parameter_179,
                parameter_178,
                parameter_177,
                parameter_176,
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
        del parameter_176, parameter_177, parameter_178, parameter_179

        # pd_op.add: (4x40x60x60xf32) <- (4x40x60x60xf32, 4x40x60x60xf32)
        add_4 = paddle._C_ops.add(batch_norm__126, batch_norm__144)

        # pd_op.conv2d: (4x96x60x60xf32) <- (4x40x60x60xf32, 96x40x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            add_4, parameter_175, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_175

        # pd_op.batch_norm_: (4x96x60x60xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (4x96x60x60xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
                parameter_174,
                parameter_173,
                parameter_172,
                parameter_171,
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
        del parameter_171, parameter_172, parameter_173, parameter_174

        # pd_op.hardswish: (4x96x60x60xf32) <- (4x96x60x60xf32)
        hardswish_5 = paddle._C_ops.hardswish(batch_norm__150)

        # pd_op.depthwise_conv2d: (4x96x60x60xf32) <- (4x96x60x60xf32, 96x1x3x3xf32)
        depthwise_conv2d_8 = paddle._C_ops.depthwise_conv2d(
            hardswish_5, parameter_170, [1, 1], [1, 1], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del parameter_170

        # pd_op.batch_norm_: (4x96x60x60xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (4x96x60x60xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
                parameter_169,
                parameter_168,
                parameter_167,
                parameter_166,
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
        del parameter_166, parameter_167, parameter_168, parameter_169

        # pd_op.hardswish: (4x96x60x60xf32) <- (4x96x60x60xf32)
        hardswish_6 = paddle._C_ops.hardswish(batch_norm__156)

        # pd_op.conv2d: (4x40x60x60xf32) <- (4x96x60x60xf32, 40x96x1x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            hardswish_6, parameter_165, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_165

        # pd_op.batch_norm_: (4x40x60x60xf32, 40xf32, 40xf32, 40xf32, 40xf32, -1xui8) <- (4x40x60x60xf32, 40xf32, 40xf32, 40xf32, 40xf32)
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
                parameter_164,
                parameter_163,
                parameter_162,
                parameter_161,
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
        del parameter_161, parameter_162, parameter_163, parameter_164

        # pd_op.add: (4x40x60x60xf32) <- (4x40x60x60xf32, 4x40x60x60xf32)
        add_5 = paddle._C_ops.add(add_4, batch_norm__162)

        # pd_op.conv2d: (4x96x60x60xf32) <- (4x40x60x60xf32, 96x40x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            add_5, parameter_160, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_160

        # pd_op.batch_norm_: (4x96x60x60xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (4x96x60x60xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
                parameter_159,
                parameter_158,
                parameter_157,
                parameter_156,
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
        del parameter_156, parameter_157, parameter_158, parameter_159

        # pd_op.hardswish: (4x96x60x60xf32) <- (4x96x60x60xf32)
        hardswish_7 = paddle._C_ops.hardswish(batch_norm__168)

        # pd_op.depthwise_conv2d: (4x96x60x60xf32) <- (4x96x60x60xf32, 96x1x3x3xf32)
        depthwise_conv2d_9 = paddle._C_ops.depthwise_conv2d(
            hardswish_7, parameter_155, [1, 1], [1, 1], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del parameter_155

        # pd_op.batch_norm_: (4x96x60x60xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (4x96x60x60xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
                parameter_154,
                parameter_153,
                parameter_152,
                parameter_151,
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
        del parameter_151, parameter_152, parameter_153, parameter_154

        # pd_op.hardswish: (4x96x60x60xf32) <- (4x96x60x60xf32)
        hardswish_8 = paddle._C_ops.hardswish(batch_norm__174)

        # pd_op.conv2d: (4x40x60x60xf32) <- (4x96x60x60xf32, 40x96x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            hardswish_8, parameter_150, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_150

        # pd_op.batch_norm_: (4x40x60x60xf32, 40xf32, 40xf32, 40xf32, 40xf32, -1xui8) <- (4x40x60x60xf32, 40xf32, 40xf32, 40xf32, 40xf32)
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
                parameter_149,
                parameter_148,
                parameter_147,
                parameter_146,
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
        del parameter_146, parameter_147, parameter_148, parameter_149

        # pd_op.add: (4x40x60x60xf32) <- (4x40x60x60xf32, 4x40x60x60xf32)
        add_6 = paddle._C_ops.add(add_5, batch_norm__180)

        # pd_op.conv2d: (4x240x60x60xf32) <- (4x40x60x60xf32, 240x40x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            add_6, parameter_145, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_145

        # pd_op.batch_norm_: (4x240x60x60xf32, 240xf32, 240xf32, 240xf32, 240xf32, -1xui8) <- (4x240x60x60xf32, 240xf32, 240xf32, 240xf32, 240xf32)
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

        # pd_op.hardswish: (4x240x60x60xf32) <- (4x240x60x60xf32)
        hardswish_9 = paddle._C_ops.hardswish(batch_norm__186)

        # pd_op.depthwise_conv2d: (4x240x60x60xf32) <- (4x240x60x60xf32, 240x1x3x3xf32)
        depthwise_conv2d_10 = paddle._C_ops.depthwise_conv2d(
            hardswish_9, parameter_140, [1, 1], [1, 1], "EXPLICIT", 240, [1, 1], "NCHW"
        )
        del parameter_140

        # pd_op.batch_norm_: (4x240x60x60xf32, 240xf32, 240xf32, 240xf32, 240xf32, -1xui8) <- (4x240x60x60xf32, 240xf32, 240xf32, 240xf32, 240xf32)
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
                parameter_139,
                parameter_138,
                parameter_137,
                parameter_136,
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
        del parameter_136, parameter_137, parameter_138, parameter_139

        # pd_op.hardswish: (4x240x60x60xf32) <- (4x240x60x60xf32)
        hardswish_10 = paddle._C_ops.hardswish(batch_norm__192)

        # pd_op.conv2d: (4x56x60x60xf32) <- (4x240x60x60xf32, 56x240x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(
            hardswish_10, parameter_135, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_135

        # pd_op.batch_norm_: (4x56x60x60xf32, 56xf32, 56xf32, 56xf32, 56xf32, -1xui8) <- (4x56x60x60xf32, 56xf32, 56xf32, 56xf32, 56xf32)
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
                parameter_134,
                parameter_133,
                parameter_132,
                parameter_131,
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
        del parameter_131, parameter_132, parameter_133, parameter_134

        # pd_op.conv2d: (4x336x60x60xf32) <- (4x56x60x60xf32, 336x56x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(
            batch_norm__198,
            parameter_130,
            [1, 1],
            [0, 0],
            "EXPLICIT",
            [1, 1],
            1,
            "NCHW",
        )
        del parameter_130

        # pd_op.batch_norm_: (4x336x60x60xf32, 336xf32, 336xf32, 336xf32, 336xf32, -1xui8) <- (4x336x60x60xf32, 336xf32, 336xf32, 336xf32, 336xf32)
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
                parameter_129,
                parameter_128,
                parameter_127,
                parameter_126,
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
        del parameter_126, parameter_127, parameter_128, parameter_129

        # pd_op.hardswish: (4x336x60x60xf32) <- (4x336x60x60xf32)
        hardswish_11 = paddle._C_ops.hardswish(batch_norm__204)

        # pd_op.depthwise_conv2d: (4x336x60x60xf32) <- (4x336x60x60xf32, 336x1x3x3xf32)
        depthwise_conv2d_11 = paddle._C_ops.depthwise_conv2d(
            hardswish_11, parameter_125, [1, 1], [1, 1], "EXPLICIT", 336, [1, 1], "NCHW"
        )
        del parameter_125

        # pd_op.batch_norm_: (4x336x60x60xf32, 336xf32, 336xf32, 336xf32, 336xf32, -1xui8) <- (4x336x60x60xf32, 336xf32, 336xf32, 336xf32, 336xf32)
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
                parameter_124,
                parameter_123,
                parameter_122,
                parameter_121,
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
        del parameter_121, parameter_122, parameter_123, parameter_124

        # pd_op.hardswish: (4x336x60x60xf32) <- (4x336x60x60xf32)
        hardswish_12 = paddle._C_ops.hardswish(batch_norm__210)

        # pd_op.conv2d: (4x56x60x60xf32) <- (4x336x60x60xf32, 56x336x1x1xf32)
        conv2d_24 = paddle._C_ops.conv2d(
            hardswish_12, parameter_120, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_120

        # pd_op.batch_norm_: (4x56x60x60xf32, 56xf32, 56xf32, 56xf32, 56xf32, -1xui8) <- (4x56x60x60xf32, 56xf32, 56xf32, 56xf32, 56xf32)
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
                parameter_119,
                parameter_118,
                parameter_117,
                parameter_116,
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
        del parameter_116, parameter_117, parameter_118, parameter_119

        # pd_op.add: (4x56x60x60xf32) <- (4x56x60x60xf32, 4x56x60x60xf32)
        add_7 = paddle._C_ops.add(batch_norm__198, batch_norm__216)

        # pd_op.conv2d: (4x336x60x60xf32) <- (4x56x60x60xf32, 336x56x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(
            add_7, parameter_115, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_115

        # pd_op.batch_norm_: (4x336x60x60xf32, 336xf32, 336xf32, 336xf32, 336xf32, -1xui8) <- (4x336x60x60xf32, 336xf32, 336xf32, 336xf32, 336xf32)
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
                parameter_114,
                parameter_113,
                parameter_112,
                parameter_111,
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
        del parameter_111, parameter_112, parameter_113, parameter_114

        # pd_op.hardswish: (4x336x60x60xf32) <- (4x336x60x60xf32)
        hardswish_13 = paddle._C_ops.hardswish(batch_norm__222)

        # pd_op.depthwise_conv2d: (4x336x30x30xf32) <- (4x336x60x60xf32, 336x1x5x5xf32)
        depthwise_conv2d_12 = paddle._C_ops.depthwise_conv2d(
            hardswish_13, parameter_110, [2, 2], [2, 2], "EXPLICIT", 336, [1, 1], "NCHW"
        )
        del parameter_110

        # pd_op.batch_norm_: (4x336x30x30xf32, 336xf32, 336xf32, 336xf32, 336xf32, -1xui8) <- (4x336x30x30xf32, 336xf32, 336xf32, 336xf32, 336xf32)
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
                parameter_109,
                parameter_108,
                parameter_107,
                parameter_106,
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
        del parameter_106, parameter_107, parameter_108, parameter_109

        # pd_op.hardswish: (4x336x30x30xf32) <- (4x336x30x30xf32)
        hardswish_14 = paddle._C_ops.hardswish(batch_norm__228)

        # pd_op.conv2d: (4x80x30x30xf32) <- (4x336x30x30xf32, 80x336x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(
            hardswish_14, parameter_105, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_105

        # pd_op.batch_norm_: (4x80x30x30xf32, 80xf32, 80xf32, 80xf32, 80xf32, -1xui8) <- (4x80x30x30xf32, 80xf32, 80xf32, 80xf32, 80xf32)
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

        # pd_op.conv2d: (4x480x30x30xf32) <- (4x80x30x30xf32, 480x80x1x1xf32)
        conv2d_27 = paddle._C_ops.conv2d(
            batch_norm__234,
            parameter_100,
            [1, 1],
            [0, 0],
            "EXPLICIT",
            [1, 1],
            1,
            "NCHW",
        )
        del parameter_100

        # pd_op.batch_norm_: (4x480x30x30xf32, 480xf32, 480xf32, 480xf32, 480xf32, -1xui8) <- (4x480x30x30xf32, 480xf32, 480xf32, 480xf32, 480xf32)
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

        # pd_op.hardswish: (4x480x30x30xf32) <- (4x480x30x30xf32)
        hardswish_15 = paddle._C_ops.hardswish(batch_norm__240)

        # pd_op.depthwise_conv2d: (4x480x30x30xf32) <- (4x480x30x30xf32, 480x1x5x5xf32)
        depthwise_conv2d_13 = paddle._C_ops.depthwise_conv2d(
            hardswish_15, parameter_95, [1, 1], [2, 2], "EXPLICIT", 480, [1, 1], "NCHW"
        )
        del parameter_95

        # pd_op.batch_norm_: (4x480x30x30xf32, 480xf32, 480xf32, 480xf32, 480xf32, -1xui8) <- (4x480x30x30xf32, 480xf32, 480xf32, 480xf32, 480xf32)
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
                parameter_94,
                parameter_93,
                parameter_92,
                parameter_91,
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
        del parameter_91, parameter_92, parameter_93, parameter_94

        # pd_op.hardswish: (4x480x30x30xf32) <- (4x480x30x30xf32)
        hardswish_16 = paddle._C_ops.hardswish(batch_norm__246)

        # pd_op.conv2d: (4x80x30x30xf32) <- (4x480x30x30xf32, 80x480x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(
            hardswish_16, parameter_90, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_90

        # pd_op.batch_norm_: (4x80x30x30xf32, 80xf32, 80xf32, 80xf32, 80xf32, -1xui8) <- (4x80x30x30xf32, 80xf32, 80xf32, 80xf32, 80xf32)
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
                parameter_89,
                parameter_88,
                parameter_87,
                parameter_86,
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
        del parameter_86, parameter_87, parameter_88, parameter_89

        # pd_op.add: (4x80x30x30xf32) <- (4x80x30x30xf32, 4x80x30x30xf32)
        add_8 = paddle._C_ops.add(batch_norm__234, batch_norm__252)

        # pd_op.conv2d: (4x480x30x30xf32) <- (4x80x30x30xf32, 480x80x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(
            add_8, parameter_85, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_85

        # pd_op.batch_norm_: (4x480x30x30xf32, 480xf32, 480xf32, 480xf32, 480xf32, -1xui8) <- (4x480x30x30xf32, 480xf32, 480xf32, 480xf32, 480xf32)
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
                parameter_84,
                parameter_83,
                parameter_82,
                parameter_81,
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
        del parameter_81, parameter_82, parameter_83, parameter_84

        # pd_op.hardswish: (4x480x30x30xf32) <- (4x480x30x30xf32)
        hardswish_17 = paddle._C_ops.hardswish(batch_norm__258)

        # pd_op.depthwise_conv2d: (4x480x30x30xf32) <- (4x480x30x30xf32, 480x1x5x5xf32)
        depthwise_conv2d_14 = paddle._C_ops.depthwise_conv2d(
            hardswish_17, parameter_80, [1, 1], [2, 2], "EXPLICIT", 480, [1, 1], "NCHW"
        )
        del parameter_80

        # pd_op.batch_norm_: (4x480x30x30xf32, 480xf32, 480xf32, 480xf32, 480xf32, -1xui8) <- (4x480x30x30xf32, 480xf32, 480xf32, 480xf32, 480xf32)
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
                parameter_79,
                parameter_78,
                parameter_77,
                parameter_76,
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
        del parameter_76, parameter_77, parameter_78, parameter_79

        # pd_op.hardswish: (4x480x30x30xf32) <- (4x480x30x30xf32)
        hardswish_18 = paddle._C_ops.hardswish(batch_norm__264)

        # pd_op.conv2d: (4x80x30x30xf32) <- (4x480x30x30xf32, 80x480x1x1xf32)
        conv2d_30 = paddle._C_ops.conv2d(
            hardswish_18, parameter_75, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_75

        # pd_op.batch_norm_: (4x80x30x30xf32, 80xf32, 80xf32, 80xf32, 80xf32, -1xui8) <- (4x80x30x30xf32, 80xf32, 80xf32, 80xf32, 80xf32)
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
                parameter_74,
                parameter_73,
                parameter_72,
                parameter_71,
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
        del parameter_71, parameter_72, parameter_73, parameter_74

        # pd_op.add: (4x80x30x30xf32) <- (4x80x30x30xf32, 4x80x30x30xf32)
        add_9 = paddle._C_ops.add(add_8, batch_norm__270)

        # pd_op.conv2d: (4x480x30x30xf32) <- (4x80x30x30xf32, 480x80x1x1xf32)
        conv2d_31 = paddle._C_ops.conv2d(
            add_9, parameter_70, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_70

        # pd_op.batch_norm_: (4x480x30x30xf32, 480xf32, 480xf32, 480xf32, 480xf32, -1xui8) <- (4x480x30x30xf32, 480xf32, 480xf32, 480xf32, 480xf32)
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
                parameter_69,
                parameter_68,
                parameter_67,
                parameter_66,
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
        del parameter_66, parameter_67, parameter_68, parameter_69

        # pd_op.hardswish: (4x480x30x30xf32) <- (4x480x30x30xf32)
        hardswish_19 = paddle._C_ops.hardswish(batch_norm__276)

        # pd_op.conv2d: (4x96x30x30xf32) <- (4x480x30x30xf32, 96x480x1x1xf32)
        conv2d_32 = paddle._C_ops.conv2d(
            hardswish_19, parameter_65, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_65

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_0 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_1 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_2 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_3 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_4 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_5 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_6 = full_int_array_0

        # pd_op.pool2d: (4x96x1x1xf32) <- (4x96x30x30xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            conv2d_32,
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

        # pd_op.conv2d: (4x24x1x1xf32) <- (4x96x1x1xf32, 24x96x1x1xf32)
        conv2d_33 = paddle._C_ops.conv2d(
            pool2d_0, parameter_64, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_64

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [1, -1, 1, 1]

        # pd_op.reshape: (1x24x1x1xf32) <- (24xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_63, full_int_array_1)
        del parameter_63

        # pd_op.add: (4x24x1x1xf32) <- (4x24x1x1xf32, 1x24x1x1xf32)
        add_10 = paddle._C_ops.add(conv2d_33, reshape_0)

        # pd_op.relu: (4x24x1x1xf32) <- (4x24x1x1xf32)
        relu_12 = paddle._C_ops.relu(add_10)
        del add_10

        # pd_op.conv2d: (4x96x1x1xf32) <- (4x24x1x1xf32, 96x24x1x1xf32)
        conv2d_34 = paddle._C_ops.conv2d(
            relu_12, parameter_62, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_62

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_61, full_int_array_1)
        del parameter_61

        # pd_op.add: (4x96x1x1xf32) <- (4x96x1x1xf32, 1x96x1x1xf32)
        add_11 = paddle._C_ops.add(conv2d_34, reshape_1)

        # pd_op.hardsigmoid: (4x96x1x1xf32) <- (4x96x1x1xf32)
        hardsigmoid_0 = paddle._C_ops.hardsigmoid(add_11, float("0.2"), float("0.5"))
        del add_11

        # pd_op.multiply: (4x96x30x30xf32) <- (4x96x30x30xf32, 4x96x1x1xf32)
        multiply_0 = paddle._C_ops.multiply(conv2d_32, hardsigmoid_0)

        # pd_op.add: (4x96x30x30xf32) <- (4x96x30x30xf32, 4x96x30x30xf32)
        add_12 = paddle._C_ops.add(conv2d_32, multiply_0)

        # pd_op.conv2d: (4x96x60x60xf32) <- (4x56x60x60xf32, 96x56x1x1xf32)
        conv2d_35 = paddle._C_ops.conv2d(
            add_7, parameter_60, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_60

        # pd_op.pool2d: (4x96x1x1xf32) <- (4x96x60x60xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(
            conv2d_35,
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

        # pd_op.conv2d: (4x24x1x1xf32) <- (4x96x1x1xf32, 24x96x1x1xf32)
        conv2d_36 = paddle._C_ops.conv2d(
            pool2d_1, parameter_59, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_59

        # pd_op.reshape: (1x24x1x1xf32) <- (24xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(parameter_58, full_int_array_1)
        del parameter_58

        # pd_op.add: (4x24x1x1xf32) <- (4x24x1x1xf32, 1x24x1x1xf32)
        add_13 = paddle._C_ops.add(conv2d_36, reshape_2)

        # pd_op.relu: (4x24x1x1xf32) <- (4x24x1x1xf32)
        relu_13 = paddle._C_ops.relu(add_13)
        del add_13

        # pd_op.conv2d: (4x96x1x1xf32) <- (4x24x1x1xf32, 96x24x1x1xf32)
        conv2d_37 = paddle._C_ops.conv2d(
            relu_13, parameter_57, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_57

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(parameter_56, full_int_array_1)
        del parameter_56

        # pd_op.add: (4x96x1x1xf32) <- (4x96x1x1xf32, 1x96x1x1xf32)
        add_14 = paddle._C_ops.add(conv2d_37, reshape_3)

        # pd_op.hardsigmoid: (4x96x1x1xf32) <- (4x96x1x1xf32)
        hardsigmoid_1 = paddle._C_ops.hardsigmoid(add_14, float("0.2"), float("0.5"))
        del add_14

        # pd_op.multiply: (4x96x60x60xf32) <- (4x96x60x60xf32, 4x96x1x1xf32)
        multiply_1 = paddle._C_ops.multiply(conv2d_35, hardsigmoid_1)

        # pd_op.add: (4x96x60x60xf32) <- (4x96x60x60xf32, 4x96x60x60xf32)
        add_15 = paddle._C_ops.add(conv2d_35, multiply_1)

        # pd_op.conv2d: (4x96x120x120xf32) <- (4x24x120x120xf32, 96x24x1x1xf32)
        conv2d_38 = paddle._C_ops.conv2d(
            add_3, parameter_55, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_55

        # pd_op.pool2d: (4x96x1x1xf32) <- (4x96x120x120xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(
            conv2d_38,
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

        # pd_op.conv2d: (4x24x1x1xf32) <- (4x96x1x1xf32, 24x96x1x1xf32)
        conv2d_39 = paddle._C_ops.conv2d(
            pool2d_2, parameter_54, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_54

        # pd_op.reshape: (1x24x1x1xf32) <- (24xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(parameter_53, full_int_array_1)
        del parameter_53

        # pd_op.add: (4x24x1x1xf32) <- (4x24x1x1xf32, 1x24x1x1xf32)
        add_16 = paddle._C_ops.add(conv2d_39, reshape_4)

        # pd_op.relu: (4x24x1x1xf32) <- (4x24x1x1xf32)
        relu_14 = paddle._C_ops.relu(add_16)
        del add_16

        # pd_op.conv2d: (4x96x1x1xf32) <- (4x24x1x1xf32, 96x24x1x1xf32)
        conv2d_40 = paddle._C_ops.conv2d(
            relu_14, parameter_52, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_52

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(parameter_51, full_int_array_1)
        del parameter_51

        # pd_op.add: (4x96x1x1xf32) <- (4x96x1x1xf32, 1x96x1x1xf32)
        add_17 = paddle._C_ops.add(conv2d_40, reshape_5)

        # pd_op.hardsigmoid: (4x96x1x1xf32) <- (4x96x1x1xf32)
        hardsigmoid_2 = paddle._C_ops.hardsigmoid(add_17, float("0.2"), float("0.5"))
        del add_17

        # pd_op.multiply: (4x96x120x120xf32) <- (4x96x120x120xf32, 4x96x1x1xf32)
        multiply_2 = paddle._C_ops.multiply(conv2d_38, hardsigmoid_2)

        # pd_op.add: (4x96x120x120xf32) <- (4x96x120x120xf32, 4x96x120x120xf32)
        add_18 = paddle._C_ops.add(conv2d_38, multiply_2)

        # pd_op.conv2d: (4x96x240x240xf32) <- (4x16x240x240xf32, 96x16x1x1xf32)
        conv2d_41 = paddle._C_ops.conv2d(
            add_1, parameter_50, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_50

        # pd_op.pool2d: (4x96x1x1xf32) <- (4x96x240x240xf32, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(
            conv2d_41,
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

        # pd_op.conv2d: (4x24x1x1xf32) <- (4x96x1x1xf32, 24x96x1x1xf32)
        conv2d_42 = paddle._C_ops.conv2d(
            pool2d_3, parameter_49, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_49

        # pd_op.reshape: (1x24x1x1xf32) <- (24xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(parameter_48, full_int_array_1)
        del parameter_48

        # pd_op.add: (4x24x1x1xf32) <- (4x24x1x1xf32, 1x24x1x1xf32)
        add_19 = paddle._C_ops.add(conv2d_42, reshape_6)

        # pd_op.relu: (4x24x1x1xf32) <- (4x24x1x1xf32)
        relu_15 = paddle._C_ops.relu(add_19)
        del add_19

        # pd_op.conv2d: (4x96x1x1xf32) <- (4x24x1x1xf32, 96x24x1x1xf32)
        conv2d_43 = paddle._C_ops.conv2d(
            relu_15, parameter_47, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_47

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(parameter_46, full_int_array_1)
        del parameter_46

        # pd_op.add: (4x96x1x1xf32) <- (4x96x1x1xf32, 1x96x1x1xf32)
        add_20 = paddle._C_ops.add(conv2d_43, reshape_7)

        # pd_op.hardsigmoid: (4x96x1x1xf32) <- (4x96x1x1xf32)
        hardsigmoid_3 = paddle._C_ops.hardsigmoid(add_20, float("0.2"), float("0.5"))
        del add_20

        # pd_op.multiply: (4x96x240x240xf32) <- (4x96x240x240xf32, 4x96x1x1xf32)
        multiply_3 = paddle._C_ops.multiply(conv2d_41, hardsigmoid_3)

        # pd_op.add: (4x96x240x240xf32) <- (4x96x240x240xf32, 4x96x240x240xf32)
        add_21 = paddle._C_ops.add(conv2d_41, multiply_3)

        # pd_op.nearest_interp: (4x96x60x60xf32) <- (4x96x30x30xf32, None, None, None)
        nearest_interp_0 = paddle._C_ops.nearest_interp(
            add_12,
            None,
            None,
            None,
            "NCHW",
            -1,
            -1,
            -1,
            [float("2"), float("2")],
            "nearest",
            False,
            1,
        )

        # pd_op.add: (4x96x60x60xf32) <- (4x96x60x60xf32, 4x96x60x60xf32)
        add_22 = paddle._C_ops.add(add_15, nearest_interp_0)

        # pd_op.nearest_interp: (4x96x120x120xf32) <- (4x96x60x60xf32, None, None, None)
        nearest_interp_1 = paddle._C_ops.nearest_interp(
            add_22,
            None,
            None,
            None,
            "NCHW",
            -1,
            -1,
            -1,
            [float("2"), float("2")],
            "nearest",
            False,
            1,
        )

        # pd_op.add: (4x96x120x120xf32) <- (4x96x120x120xf32, 4x96x120x120xf32)
        add_23 = paddle._C_ops.add(add_18, nearest_interp_1)

        # pd_op.nearest_interp: (4x96x240x240xf32) <- (4x96x120x120xf32, None, None, None)
        nearest_interp_2 = paddle._C_ops.nearest_interp(
            add_23,
            None,
            None,
            None,
            "NCHW",
            -1,
            -1,
            -1,
            [float("2"), float("2")],
            "nearest",
            False,
            1,
        )

        # pd_op.add: (4x96x240x240xf32) <- (4x96x240x240xf32, 4x96x240x240xf32)
        add_24 = paddle._C_ops.add(add_21, nearest_interp_2)

        # pd_op.conv2d: (4x24x30x30xf32) <- (4x96x30x30xf32, 24x96x3x3xf32)
        conv2d_44 = paddle._C_ops.conv2d(
            add_12, parameter_45, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_45

        # pd_op.pool2d: (4x24x1x1xf32) <- (4x24x30x30xf32, 2xi64)
        pool2d_4 = paddle._C_ops.pool2d(
            conv2d_44,
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

        # pd_op.conv2d: (4x6x1x1xf32) <- (4x24x1x1xf32, 6x24x1x1xf32)
        conv2d_45 = paddle._C_ops.conv2d(
            pool2d_4, parameter_44, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_44

        # pd_op.reshape: (1x6x1x1xf32) <- (6xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(parameter_43, full_int_array_1)
        del parameter_43

        # pd_op.add: (4x6x1x1xf32) <- (4x6x1x1xf32, 1x6x1x1xf32)
        add_25 = paddle._C_ops.add(conv2d_45, reshape_8)

        # pd_op.relu: (4x6x1x1xf32) <- (4x6x1x1xf32)
        relu_16 = paddle._C_ops.relu(add_25)
        del add_25

        # pd_op.conv2d: (4x24x1x1xf32) <- (4x6x1x1xf32, 24x6x1x1xf32)
        conv2d_46 = paddle._C_ops.conv2d(
            relu_16, parameter_42, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_42

        # pd_op.reshape: (1x24x1x1xf32) <- (24xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(parameter_41, full_int_array_1)
        del parameter_41

        # pd_op.add: (4x24x1x1xf32) <- (4x24x1x1xf32, 1x24x1x1xf32)
        add_26 = paddle._C_ops.add(conv2d_46, reshape_9)

        # pd_op.hardsigmoid: (4x24x1x1xf32) <- (4x24x1x1xf32)
        hardsigmoid_4 = paddle._C_ops.hardsigmoid(add_26, float("0.2"), float("0.5"))
        del add_26

        # pd_op.multiply: (4x24x30x30xf32) <- (4x24x30x30xf32, 4x24x1x1xf32)
        multiply_4 = paddle._C_ops.multiply(conv2d_44, hardsigmoid_4)

        # pd_op.add: (4x24x30x30xf32) <- (4x24x30x30xf32, 4x24x30x30xf32)
        add_27 = paddle._C_ops.add(conv2d_44, multiply_4)

        # pd_op.conv2d: (4x24x60x60xf32) <- (4x96x60x60xf32, 24x96x3x3xf32)
        conv2d_47 = paddle._C_ops.conv2d(
            add_22, parameter_40, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_40

        # pd_op.pool2d: (4x24x1x1xf32) <- (4x24x60x60xf32, 2xi64)
        pool2d_5 = paddle._C_ops.pool2d(
            conv2d_47,
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

        # pd_op.conv2d: (4x6x1x1xf32) <- (4x24x1x1xf32, 6x24x1x1xf32)
        conv2d_48 = paddle._C_ops.conv2d(
            pool2d_5, parameter_39, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_39

        # pd_op.reshape: (1x6x1x1xf32) <- (6xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(parameter_38, full_int_array_1)
        del parameter_38

        # pd_op.add: (4x6x1x1xf32) <- (4x6x1x1xf32, 1x6x1x1xf32)
        add_28 = paddle._C_ops.add(conv2d_48, reshape_10)

        # pd_op.relu: (4x6x1x1xf32) <- (4x6x1x1xf32)
        relu_17 = paddle._C_ops.relu(add_28)
        del add_28

        # pd_op.conv2d: (4x24x1x1xf32) <- (4x6x1x1xf32, 24x6x1x1xf32)
        conv2d_49 = paddle._C_ops.conv2d(
            relu_17, parameter_37, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_37

        # pd_op.reshape: (1x24x1x1xf32) <- (24xf32, 4xi64)
        reshape_11 = paddle._C_ops.reshape(parameter_36, full_int_array_1)
        del parameter_36

        # pd_op.add: (4x24x1x1xf32) <- (4x24x1x1xf32, 1x24x1x1xf32)
        add_29 = paddle._C_ops.add(conv2d_49, reshape_11)

        # pd_op.hardsigmoid: (4x24x1x1xf32) <- (4x24x1x1xf32)
        hardsigmoid_5 = paddle._C_ops.hardsigmoid(add_29, float("0.2"), float("0.5"))
        del add_29

        # pd_op.multiply: (4x24x60x60xf32) <- (4x24x60x60xf32, 4x24x1x1xf32)
        multiply_5 = paddle._C_ops.multiply(conv2d_47, hardsigmoid_5)

        # pd_op.add: (4x24x60x60xf32) <- (4x24x60x60xf32, 4x24x60x60xf32)
        add_30 = paddle._C_ops.add(conv2d_47, multiply_5)

        # pd_op.conv2d: (4x24x120x120xf32) <- (4x96x120x120xf32, 24x96x3x3xf32)
        conv2d_50 = paddle._C_ops.conv2d(
            add_23, parameter_35, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_35

        # pd_op.pool2d: (4x24x1x1xf32) <- (4x24x120x120xf32, 2xi64)
        pool2d_6 = paddle._C_ops.pool2d(
            conv2d_50,
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

        # pd_op.conv2d: (4x6x1x1xf32) <- (4x24x1x1xf32, 6x24x1x1xf32)
        conv2d_51 = paddle._C_ops.conv2d(
            pool2d_6, parameter_34, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_34

        # pd_op.reshape: (1x6x1x1xf32) <- (6xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(parameter_33, full_int_array_1)
        del parameter_33

        # pd_op.add: (4x6x1x1xf32) <- (4x6x1x1xf32, 1x6x1x1xf32)
        add_31 = paddle._C_ops.add(conv2d_51, reshape_12)

        # pd_op.relu: (4x6x1x1xf32) <- (4x6x1x1xf32)
        relu_18 = paddle._C_ops.relu(add_31)
        del add_31

        # pd_op.conv2d: (4x24x1x1xf32) <- (4x6x1x1xf32, 24x6x1x1xf32)
        conv2d_52 = paddle._C_ops.conv2d(
            relu_18, parameter_32, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_32

        # pd_op.reshape: (1x24x1x1xf32) <- (24xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(parameter_31, full_int_array_1)
        del parameter_31

        # pd_op.add: (4x24x1x1xf32) <- (4x24x1x1xf32, 1x24x1x1xf32)
        add_32 = paddle._C_ops.add(conv2d_52, reshape_13)

        # pd_op.hardsigmoid: (4x24x1x1xf32) <- (4x24x1x1xf32)
        hardsigmoid_6 = paddle._C_ops.hardsigmoid(add_32, float("0.2"), float("0.5"))
        del add_32

        # pd_op.multiply: (4x24x120x120xf32) <- (4x24x120x120xf32, 4x24x1x1xf32)
        multiply_6 = paddle._C_ops.multiply(conv2d_50, hardsigmoid_6)

        # pd_op.add: (4x24x120x120xf32) <- (4x24x120x120xf32, 4x24x120x120xf32)
        add_33 = paddle._C_ops.add(conv2d_50, multiply_6)

        # pd_op.conv2d: (4x24x240x240xf32) <- (4x96x240x240xf32, 24x96x3x3xf32)
        conv2d_53 = paddle._C_ops.conv2d(
            add_24, parameter_30, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_30

        # pd_op.pool2d: (4x24x1x1xf32) <- (4x24x240x240xf32, 2xi64)
        pool2d_7 = paddle._C_ops.pool2d(
            conv2d_53,
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

        # pd_op.conv2d: (4x6x1x1xf32) <- (4x24x1x1xf32, 6x24x1x1xf32)
        conv2d_54 = paddle._C_ops.conv2d(
            pool2d_7, parameter_29, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_29

        # pd_op.reshape: (1x6x1x1xf32) <- (6xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(parameter_28, full_int_array_1)
        del parameter_28

        # pd_op.add: (4x6x1x1xf32) <- (4x6x1x1xf32, 1x6x1x1xf32)
        add_34 = paddle._C_ops.add(conv2d_54, reshape_14)

        # pd_op.relu: (4x6x1x1xf32) <- (4x6x1x1xf32)
        relu_19 = paddle._C_ops.relu(add_34)
        del add_34

        # pd_op.conv2d: (4x24x1x1xf32) <- (4x6x1x1xf32, 24x6x1x1xf32)
        conv2d_55 = paddle._C_ops.conv2d(
            relu_19, parameter_27, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_27

        # pd_op.reshape: (1x24x1x1xf32) <- (24xf32, 4xi64)
        reshape_15 = paddle._C_ops.reshape(parameter_26, full_int_array_1)
        del full_int_array_1, parameter_26

        # pd_op.add: (4x24x1x1xf32) <- (4x24x1x1xf32, 1x24x1x1xf32)
        add_35 = paddle._C_ops.add(conv2d_55, reshape_15)

        # pd_op.hardsigmoid: (4x24x1x1xf32) <- (4x24x1x1xf32)
        hardsigmoid_7 = paddle._C_ops.hardsigmoid(add_35, float("0.2"), float("0.5"))
        del add_35

        # pd_op.multiply: (4x24x240x240xf32) <- (4x24x240x240xf32, 4x24x1x1xf32)
        multiply_7 = paddle._C_ops.multiply(conv2d_53, hardsigmoid_7)

        # pd_op.add: (4x24x240x240xf32) <- (4x24x240x240xf32, 4x24x240x240xf32)
        add_36 = paddle._C_ops.add(conv2d_53, multiply_7)

        # pd_op.nearest_interp: (4x24x240x240xf32) <- (4x24x30x30xf32, None, None, None)
        nearest_interp_3 = paddle._C_ops.nearest_interp(
            add_27,
            None,
            None,
            None,
            "NCHW",
            -1,
            -1,
            -1,
            [float("8"), float("8")],
            "nearest",
            False,
            1,
        )

        # pd_op.nearest_interp: (4x24x240x240xf32) <- (4x24x60x60xf32, None, None, None)
        nearest_interp_4 = paddle._C_ops.nearest_interp(
            add_30,
            None,
            None,
            None,
            "NCHW",
            -1,
            -1,
            -1,
            [float("4"), float("4")],
            "nearest",
            False,
            1,
        )

        # pd_op.nearest_interp: (4x24x240x240xf32) <- (4x24x120x120xf32, None, None, None)
        nearest_interp_5 = paddle._C_ops.nearest_interp(
            add_33,
            None,
            None,
            None,
            "NCHW",
            -1,
            -1,
            -1,
            [float("2"), float("2")],
            "nearest",
            False,
            1,
        )

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_7 = full_0

        # builtin.combine: ([4x24x240x240xf32, 4x24x240x240xf32, 4x24x240x240xf32, 4x24x240x240xf32]) <- (4x24x240x240xf32, 4x24x240x240xf32, 4x24x240x240xf32, 4x24x240x240xf32)
        combine_0 = [nearest_interp_3, nearest_interp_4, nearest_interp_5, add_36]

        # pd_op.concat: (4x96x240x240xf32) <- ([4x24x240x240xf32, 4x24x240x240xf32, 4x24x240x240xf32, 4x24x240x240xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_0, full_0)
        del combine_0

        # pd_op.conv2d: (4x24x240x240xf32) <- (4x96x240x240xf32, 24x96x3x3xf32)
        conv2d_56 = paddle._C_ops.conv2d(
            concat_1, parameter_25, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_25

        # pd_op.batch_norm_: (4x24x240x240xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (4x24x240x240xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        (
            batch_norm__282,
            batch_norm__283,
            batch_norm__284,
            batch_norm__285,
            batch_norm__286,
            batch_norm__287,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_56,
                parameter_24,
                parameter_23,
                parameter_22,
                parameter_21,
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
        del parameter_21, parameter_22, parameter_23, parameter_24

        # pd_op.relu: (4x24x240x240xf32) <- (4x24x240x240xf32)
        relu_20 = paddle._C_ops.relu(batch_norm__282)
        del batch_norm__282

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_2 = []

        # pd_op.assign: (0xi64) <- (0xi64)
        assign_8 = full_int_array_2

        # pd_op.assign: (0xi64) <- (0xi64)
        assign_9 = full_int_array_2

        # pd_op.assign: (0xi64) <- (0xi64)
        assign_10 = full_int_array_2

        # pd_op.conv2d_transpose: (4x24x480x480xf32) <- (4x24x240x240xf32, 24x24x2x2xf32, 0xi64)
        conv2d_transpose_0 = paddle._C_ops.conv2d_transpose(
            relu_20,
            parameter_20,
            [2, 2],
            [0, 0],
            [],
            full_int_array_2,
            "EXPLICIT",
            1,
            [1, 1],
            "NCHW",
        )
        del parameter_20

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [1, 24, 1, 1]

        # pd_op.reshape: (1x24x1x1xf32) <- (24xf32, 4xi64)
        reshape_16 = paddle._C_ops.reshape(parameter_19, full_int_array_3)
        del parameter_19

        # pd_op.add: (4x24x480x480xf32) <- (4x24x480x480xf32, 1x24x1x1xf32)
        add_37 = paddle._C_ops.add(conv2d_transpose_0, reshape_16)

        # pd_op.batch_norm_: (4x24x480x480xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (4x24x480x480xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        (
            batch_norm__288,
            batch_norm__289,
            batch_norm__290,
            batch_norm__291,
            batch_norm__292,
            batch_norm__293,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_37,
                parameter_18,
                parameter_17,
                parameter_16,
                parameter_15,
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
        del parameter_15, parameter_16, parameter_17, parameter_18

        # pd_op.relu: (4x24x480x480xf32) <- (4x24x480x480xf32)
        relu_21 = paddle._C_ops.relu(batch_norm__288)
        del batch_norm__288

        # pd_op.conv2d_transpose: (4x1x960x960xf32) <- (4x24x480x480xf32, 24x1x2x2xf32, 0xi64)
        conv2d_transpose_1 = paddle._C_ops.conv2d_transpose(
            relu_21,
            parameter_14,
            [2, 2],
            [0, 0],
            [],
            full_int_array_2,
            "EXPLICIT",
            1,
            [1, 1],
            "NCHW",
        )
        del parameter_14

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_4 = [1, 1, 1, 1]

        # pd_op.reshape: (1x1x1x1xf32) <- (1xf32, 4xi64)
        reshape_17 = paddle._C_ops.reshape(parameter_13, full_int_array_4)
        del parameter_13

        # pd_op.add: (4x1x960x960xf32) <- (4x1x960x960xf32, 1x1x1x1xf32)
        add_38 = paddle._C_ops.add(conv2d_transpose_1, reshape_17)

        # pd_op.sigmoid: (4x1x960x960xf32) <- (4x1x960x960xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(add_38)
        del add_38

        # pd_op.conv2d: (4x24x240x240xf32) <- (4x96x240x240xf32, 24x96x3x3xf32)
        conv2d_57 = paddle._C_ops.conv2d(
            concat_1, parameter_12, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_12

        # pd_op.batch_norm_: (4x24x240x240xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (4x24x240x240xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        (
            batch_norm__294,
            batch_norm__295,
            batch_norm__296,
            batch_norm__297,
            batch_norm__298,
            batch_norm__299,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_57,
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
        del parameter_10, parameter_11, parameter_8, parameter_9

        # pd_op.relu: (4x24x240x240xf32) <- (4x24x240x240xf32)
        relu_22 = paddle._C_ops.relu(batch_norm__294)
        del batch_norm__294

        # pd_op.conv2d_transpose: (4x24x480x480xf32) <- (4x24x240x240xf32, 24x24x2x2xf32, 0xi64)
        conv2d_transpose_2 = paddle._C_ops.conv2d_transpose(
            relu_22,
            parameter_7,
            [2, 2],
            [0, 0],
            [],
            full_int_array_2,
            "EXPLICIT",
            1,
            [1, 1],
            "NCHW",
        )
        del parameter_7

        # pd_op.reshape: (1x24x1x1xf32) <- (24xf32, 4xi64)
        reshape_18 = paddle._C_ops.reshape(parameter_6, full_int_array_3)
        del full_int_array_3, parameter_6

        # pd_op.add: (4x24x480x480xf32) <- (4x24x480x480xf32, 1x24x1x1xf32)
        add_39 = paddle._C_ops.add(conv2d_transpose_2, reshape_18)

        # pd_op.batch_norm_: (4x24x480x480xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (4x24x480x480xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        (
            batch_norm__300,
            batch_norm__301,
            batch_norm__302,
            batch_norm__303,
            batch_norm__304,
            batch_norm__305,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_39,
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

        # pd_op.relu: (4x24x480x480xf32) <- (4x24x480x480xf32)
        relu_23 = paddle._C_ops.relu(batch_norm__300)
        del batch_norm__300

        # pd_op.conv2d_transpose: (4x1x960x960xf32) <- (4x24x480x480xf32, 24x1x2x2xf32, 0xi64)
        conv2d_transpose_3 = paddle._C_ops.conv2d_transpose(
            relu_23,
            parameter_1,
            [2, 2],
            [0, 0],
            [],
            full_int_array_2,
            "EXPLICIT",
            1,
            [1, 1],
            "NCHW",
        )
        del parameter_1

        # pd_op.reshape: (1x1x1x1xf32) <- (1xf32, 4xi64)
        reshape_19 = paddle._C_ops.reshape(parameter_0, full_int_array_4)
        del full_int_array_4, parameter_0

        # pd_op.add: (4x1x960x960xf32) <- (4x1x960x960xf32, 1x1x1x1xf32)
        add_40 = paddle._C_ops.add(conv2d_transpose_3, reshape_19)

        # pd_op.sigmoid: (4x1x960x960xf32) <- (4x1x960x960xf32)
        sigmoid_1 = paddle._C_ops.sigmoid(add_40)
        del add_40

        # pd_op.subtract: (4x1x960x960xf32) <- (4x1x960x960xf32, 4x1x960x960xf32)
        subtract_0 = paddle._C_ops.subtract(sigmoid_0, sigmoid_1)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("-50"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (4x1x960x960xf32) <- (4x1x960x960xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(subtract_0, full_1, float("0"), True)
        del subtract_0

        # pd_op.exp: (4x1x960x960xf32) <- (4x1x960x960xf32)
        exp_0 = paddle._C_ops.exp(scale_0)
        del scale_0

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (4x1x960x960xf32) <- (4x1x960x960xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(exp_0, full_2, float("1"), True)

        # pd_op.reciprocal: (4x1x960x960xf32) <- (4x1x960x960xf32)
        reciprocal_0 = paddle._C_ops.reciprocal(scale_1)
        del scale_1

        # builtin.combine: ([4x1x960x960xf32, 4x1x960x960xf32, 4x1x960x960xf32]) <- (4x1x960x960xf32, 4x1x960x960xf32, 4x1x960x960xf32)
        combine_1 = [sigmoid_0, sigmoid_1, reciprocal_0]

        # pd_op.concat: (4x3x960x960xf32) <- ([4x1x960x960xf32, 4x1x960x960xf32, 4x1x960x960xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_1, full_0)
        del (
            add_0,
            add_1,
            add_12,
            add_15,
            add_18,
            add_2,
            add_21,
            add_22,
            add_23,
            add_24,
            add_27,
            add_3,
            add_30,
            add_33,
            add_36,
            add_37,
            add_39,
            add_4,
            add_5,
            add_6,
            add_7,
            add_8,
            add_9,
            assign_0,
            assign_1,
            assign_10,
            assign_2,
            assign_3,
            assign_4,
            assign_5,
            assign_6,
            assign_7,
            assign_8,
            assign_9,
            batch_norm__0,
            batch_norm__1,
            batch_norm__10,
            batch_norm__100,
            batch_norm__101,
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
            batch_norm__283,
            batch_norm__284,
            batch_norm__285,
            batch_norm__286,
            batch_norm__287,
            batch_norm__289,
            batch_norm__29,
            batch_norm__290,
            batch_norm__291,
            batch_norm__292,
            batch_norm__293,
            batch_norm__295,
            batch_norm__296,
            batch_norm__297,
            batch_norm__298,
            batch_norm__299,
            batch_norm__3,
            batch_norm__301,
            batch_norm__302,
            batch_norm__303,
            batch_norm__304,
            batch_norm__305,
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
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
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
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
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
            batch_norm__79,
            batch_norm__8,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
            batch_norm__83,
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
            batch_norm__97,
            batch_norm__98,
            batch_norm__99,
            combine_1,
            concat_1,
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
            conv2d_33,
            conv2d_34,
            conv2d_35,
            conv2d_36,
            conv2d_37,
            conv2d_38,
            conv2d_39,
            conv2d_4,
            conv2d_40,
            conv2d_41,
            conv2d_42,
            conv2d_43,
            conv2d_44,
            conv2d_45,
            conv2d_46,
            conv2d_47,
            conv2d_48,
            conv2d_49,
            conv2d_5,
            conv2d_50,
            conv2d_51,
            conv2d_52,
            conv2d_53,
            conv2d_54,
            conv2d_55,
            conv2d_56,
            conv2d_57,
            conv2d_6,
            conv2d_7,
            conv2d_8,
            conv2d_9,
            conv2d_transpose_0,
            conv2d_transpose_1,
            conv2d_transpose_2,
            conv2d_transpose_3,
            depthwise_conv2d_0,
            depthwise_conv2d_1,
            depthwise_conv2d_10,
            depthwise_conv2d_11,
            depthwise_conv2d_12,
            depthwise_conv2d_13,
            depthwise_conv2d_14,
            depthwise_conv2d_2,
            depthwise_conv2d_3,
            depthwise_conv2d_4,
            depthwise_conv2d_5,
            depthwise_conv2d_6,
            depthwise_conv2d_7,
            depthwise_conv2d_8,
            depthwise_conv2d_9,
            exp_0,
            full_0,
            full_1,
            full_2,
            full_int_array_0,
            full_int_array_2,
            hardsigmoid_0,
            hardsigmoid_1,
            hardsigmoid_2,
            hardsigmoid_3,
            hardsigmoid_4,
            hardsigmoid_5,
            hardsigmoid_6,
            hardsigmoid_7,
            hardswish_0,
            hardswish_1,
            hardswish_10,
            hardswish_11,
            hardswish_12,
            hardswish_13,
            hardswish_14,
            hardswish_15,
            hardswish_16,
            hardswish_17,
            hardswish_18,
            hardswish_19,
            hardswish_2,
            hardswish_3,
            hardswish_4,
            hardswish_5,
            hardswish_6,
            hardswish_7,
            hardswish_8,
            hardswish_9,
            multiply_0,
            multiply_1,
            multiply_2,
            multiply_3,
            multiply_4,
            multiply_5,
            multiply_6,
            multiply_7,
            nearest_interp_0,
            nearest_interp_1,
            nearest_interp_2,
            nearest_interp_3,
            nearest_interp_4,
            nearest_interp_5,
            pool2d_0,
            pool2d_1,
            pool2d_2,
            pool2d_3,
            pool2d_4,
            pool2d_5,
            pool2d_6,
            pool2d_7,
            reciprocal_0,
            relu_0,
            relu_1,
            relu_10,
            relu_11,
            relu_12,
            relu_13,
            relu_14,
            relu_15,
            relu_16,
            relu_17,
            relu_18,
            relu_19,
            relu_2,
            relu_20,
            relu_21,
            relu_22,
            relu_23,
            relu_3,
            relu_4,
            relu_5,
            relu_6,
            relu_7,
            relu_8,
            relu_9,
            reshape_0,
            reshape_1,
            reshape_10,
            reshape_11,
            reshape_12,
            reshape_13,
            reshape_14,
            reshape_15,
            reshape_16,
            reshape_17,
            reshape_18,
            reshape_19,
            reshape_2,
            reshape_3,
            reshape_4,
            reshape_5,
            reshape_6,
            reshape_7,
            reshape_8,
            reshape_9,
            sigmoid_0,
            sigmoid_1,
        )

        return concat_0
