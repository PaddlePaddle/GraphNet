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
        data_0,
        data_1,
        data_2,
        data_3,
    ):
        # pd_op.conv2d: (16x96x36x36xf32) <- (16x96x36x36xf32, 96x96x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_298, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (16x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
                parameter_297,
                parameter_296,
                parameter_295,
                parameter_294,
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

        # pd_op.hardswish: (16x96x36x36xf32) <- (16x96x36x36xf32)
        hardswish_0 = paddle._C_ops.hardswish(batch_norm__0)

        # pd_op.conv2d: (16x96x18x18xf32) <- (16x192x18x18xf32, 96x192x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            data_1, parameter_293, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x96x18x18xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (16x96x18x18xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
                parameter_292,
                parameter_291,
                parameter_290,
                parameter_289,
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

        # pd_op.hardswish: (16x96x18x18xf32) <- (16x96x18x18xf32)
        hardswish_1 = paddle._C_ops.hardswish(batch_norm__6)

        # pd_op.conv2d: (16x96x9x9xf32) <- (16x384x9x9xf32, 96x384x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            data_2, parameter_288, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x96x9x9xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (16x96x9x9xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            batch_norm__16,
            batch_norm__17,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_2,
                parameter_287,
                parameter_286,
                parameter_285,
                parameter_284,
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

        # pd_op.hardswish: (16x96x9x9xf32) <- (16x96x9x9xf32)
        hardswish_2 = paddle._C_ops.hardswish(batch_norm__12)

        # pd_op.nearest_interp: (16x96x18x18xf32) <- (16x96x9x9xf32, None, None, None)
        nearest_interp_0 = paddle._C_ops.nearest_interp(
            hardswish_2,
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
            0,
        )

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_24 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_23 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_22 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_2 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_1 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_0 = full_0

        # builtin.combine: ([16x96x18x18xf32, 16x96x18x18xf32]) <- (16x96x18x18xf32, 16x96x18x18xf32)
        combine_0 = [nearest_interp_0, hardswish_1]

        # pd_op.concat: (16x192x18x18xf32) <- ([16x96x18x18xf32, 16x96x18x18xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        # pd_op.depthwise_conv2d: (16x192x18x18xf32) <- (16x192x18x18xf32, 192x1x5x5xf32)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(
            concat_0, parameter_283, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (16x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (16x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__18,
            batch_norm__19,
            batch_norm__20,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_0,
                parameter_282,
                parameter_281,
                parameter_280,
                parameter_279,
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

        # pd_op.hardswish: (16x192x18x18xf32) <- (16x192x18x18xf32)
        hardswish_3 = paddle._C_ops.hardswish(batch_norm__18)

        # pd_op.conv2d: (16x192x18x18xf32) <- (16x192x18x18xf32, 192x192x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            hardswish_3, parameter_278, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (16x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32)
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
                parameter_277,
                parameter_276,
                parameter_275,
                parameter_274,
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

        # pd_op.hardswish: (16x192x18x18xf32) <- (16x192x18x18xf32)
        hardswish_4 = paddle._C_ops.hardswish(batch_norm__24)

        # pd_op.depthwise_conv2d: (16x192x18x18xf32) <- (16x192x18x18xf32, 192x1x5x5xf32)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(
            hardswish_4, parameter_273, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (16x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (16x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32)
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
                parameter_272,
                parameter_271,
                parameter_270,
                parameter_269,
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

        # pd_op.hardswish: (16x192x18x18xf32) <- (16x192x18x18xf32)
        hardswish_5 = paddle._C_ops.hardswish(batch_norm__30)

        # pd_op.conv2d: (16x96x18x18xf32) <- (16x192x18x18xf32, 96x192x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            hardswish_5, parameter_268, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x96x18x18xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (16x96x18x18xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
                parameter_267,
                parameter_266,
                parameter_265,
                parameter_264,
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

        # pd_op.hardswish: (16x96x18x18xf32) <- (16x96x18x18xf32)
        hardswish_6 = paddle._C_ops.hardswish(batch_norm__36)

        # pd_op.nearest_interp: (16x96x36x36xf32) <- (16x96x18x18xf32, None, None, None)
        nearest_interp_1 = paddle._C_ops.nearest_interp(
            hardswish_6,
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
            0,
        )

        # builtin.combine: ([16x96x36x36xf32, 16x96x36x36xf32]) <- (16x96x36x36xf32, 16x96x36x36xf32)
        combine_1 = [nearest_interp_1, hardswish_0]

        # pd_op.concat: (16x192x36x36xf32) <- ([16x96x36x36xf32, 16x96x36x36xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_0)

        # pd_op.depthwise_conv2d: (16x192x36x36xf32) <- (16x192x36x36xf32, 192x1x5x5xf32)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(
            concat_1, parameter_263, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (16x192x36x36xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (16x192x36x36xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__42,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_2,
                parameter_262,
                parameter_261,
                parameter_260,
                parameter_259,
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

        # pd_op.hardswish: (16x192x36x36xf32) <- (16x192x36x36xf32)
        hardswish_7 = paddle._C_ops.hardswish(batch_norm__42)

        # pd_op.conv2d: (16x192x36x36xf32) <- (16x192x36x36xf32, 192x192x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            hardswish_7, parameter_258, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x192x36x36xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (16x192x36x36xf32, 192xf32, 192xf32, 192xf32, 192xf32)
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

        # pd_op.hardswish: (16x192x36x36xf32) <- (16x192x36x36xf32)
        hardswish_8 = paddle._C_ops.hardswish(batch_norm__48)

        # pd_op.depthwise_conv2d: (16x192x36x36xf32) <- (16x192x36x36xf32, 192x1x5x5xf32)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(
            hardswish_8, parameter_253, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (16x192x36x36xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (16x192x36x36xf32, 192xf32, 192xf32, 192xf32, 192xf32)
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

        # pd_op.hardswish: (16x192x36x36xf32) <- (16x192x36x36xf32)
        hardswish_9 = paddle._C_ops.hardswish(batch_norm__54)

        # pd_op.conv2d: (16x96x36x36xf32) <- (16x192x36x36xf32, 96x192x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            hardswish_9, parameter_248, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (16x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
                parameter_247,
                parameter_246,
                parameter_245,
                parameter_244,
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

        # pd_op.hardswish: (16x96x36x36xf32) <- (16x96x36x36xf32)
        hardswish_48 = paddle._C_ops.hardswish(batch_norm__60)

        # pd_op.depthwise_conv2d: (16x96x18x18xf32) <- (16x96x36x36xf32, 96x1x5x5xf32)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(
            hardswish_48, parameter_243, [2, 2], [2, 2], "EXPLICIT", 96, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (16x96x18x18xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (16x96x18x18xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__66,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            batch_norm__70,
            batch_norm__71,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_4,
                parameter_242,
                parameter_241,
                parameter_240,
                parameter_239,
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

        # pd_op.hardswish: (16x96x18x18xf32) <- (16x96x18x18xf32)
        hardswish_10 = paddle._C_ops.hardswish(batch_norm__66)

        # pd_op.conv2d: (16x96x18x18xf32) <- (16x96x18x18xf32, 96x96x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            hardswish_10, parameter_238, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x96x18x18xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (16x96x18x18xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__72,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_7,
                parameter_237,
                parameter_236,
                parameter_235,
                parameter_234,
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

        # pd_op.hardswish: (16x96x18x18xf32) <- (16x96x18x18xf32)
        hardswish_11 = paddle._C_ops.hardswish(batch_norm__72)

        # builtin.combine: ([16x96x18x18xf32, 16x96x18x18xf32]) <- (16x96x18x18xf32, 16x96x18x18xf32)
        combine_2 = [hardswish_11, hardswish_6]

        # pd_op.concat: (16x192x18x18xf32) <- ([16x96x18x18xf32, 16x96x18x18xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_0)

        # pd_op.depthwise_conv2d: (16x192x18x18xf32) <- (16x192x18x18xf32, 192x1x5x5xf32)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(
            concat_2, parameter_233, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (16x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (16x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__78,
            batch_norm__79,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
            batch_norm__83,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_5,
                parameter_232,
                parameter_231,
                parameter_230,
                parameter_229,
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

        # pd_op.hardswish: (16x192x18x18xf32) <- (16x192x18x18xf32)
        hardswish_12 = paddle._C_ops.hardswish(batch_norm__78)

        # pd_op.conv2d: (16x192x18x18xf32) <- (16x192x18x18xf32, 192x192x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            hardswish_12, parameter_228, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (16x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__84,
            batch_norm__85,
            batch_norm__86,
            batch_norm__87,
            batch_norm__88,
            batch_norm__89,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_8,
                parameter_227,
                parameter_226,
                parameter_225,
                parameter_224,
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

        # pd_op.hardswish: (16x192x18x18xf32) <- (16x192x18x18xf32)
        hardswish_13 = paddle._C_ops.hardswish(batch_norm__84)

        # pd_op.depthwise_conv2d: (16x192x18x18xf32) <- (16x192x18x18xf32, 192x1x5x5xf32)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(
            hardswish_13, parameter_223, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (16x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (16x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__90,
            batch_norm__91,
            batch_norm__92,
            batch_norm__93,
            batch_norm__94,
            batch_norm__95,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_6,
                parameter_222,
                parameter_221,
                parameter_220,
                parameter_219,
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

        # pd_op.hardswish: (16x192x18x18xf32) <- (16x192x18x18xf32)
        hardswish_14 = paddle._C_ops.hardswish(batch_norm__90)

        # pd_op.conv2d: (16x96x18x18xf32) <- (16x192x18x18xf32, 96x192x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            hardswish_14, parameter_218, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x96x18x18xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (16x96x18x18xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__96,
            batch_norm__97,
            batch_norm__98,
            batch_norm__99,
            batch_norm__100,
            batch_norm__101,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_9,
                parameter_217,
                parameter_216,
                parameter_215,
                parameter_214,
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

        # pd_op.hardswish: (16x96x18x18xf32) <- (16x96x18x18xf32)
        hardswish_49 = paddle._C_ops.hardswish(batch_norm__96)

        # pd_op.depthwise_conv2d: (16x96x9x9xf32) <- (16x96x18x18xf32, 96x1x5x5xf32)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(
            hardswish_49, parameter_213, [2, 2], [2, 2], "EXPLICIT", 96, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (16x96x9x9xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (16x96x9x9xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__102,
            batch_norm__103,
            batch_norm__104,
            batch_norm__105,
            batch_norm__106,
            batch_norm__107,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_7,
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

        # pd_op.hardswish: (16x96x9x9xf32) <- (16x96x9x9xf32)
        hardswish_15 = paddle._C_ops.hardswish(batch_norm__102)

        # pd_op.conv2d: (16x96x9x9xf32) <- (16x96x9x9xf32, 96x96x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            hardswish_15, parameter_208, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x96x9x9xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (16x96x9x9xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__108,
            batch_norm__109,
            batch_norm__110,
            batch_norm__111,
            batch_norm__112,
            batch_norm__113,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_10,
                parameter_207,
                parameter_206,
                parameter_205,
                parameter_204,
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

        # pd_op.hardswish: (16x96x9x9xf32) <- (16x96x9x9xf32)
        hardswish_16 = paddle._C_ops.hardswish(batch_norm__108)

        # builtin.combine: ([16x96x9x9xf32, 16x96x9x9xf32]) <- (16x96x9x9xf32, 16x96x9x9xf32)
        combine_3 = [hardswish_16, hardswish_2]

        # pd_op.concat: (16x192x9x9xf32) <- ([16x96x9x9xf32, 16x96x9x9xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_3, full_0)

        # pd_op.depthwise_conv2d: (16x192x9x9xf32) <- (16x192x9x9xf32, 192x1x5x5xf32)
        depthwise_conv2d_8 = paddle._C_ops.depthwise_conv2d(
            concat_3, parameter_203, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (16x192x9x9xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (16x192x9x9xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__114,
            batch_norm__115,
            batch_norm__116,
            batch_norm__117,
            batch_norm__118,
            batch_norm__119,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_8,
                parameter_202,
                parameter_201,
                parameter_200,
                parameter_199,
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

        # pd_op.hardswish: (16x192x9x9xf32) <- (16x192x9x9xf32)
        hardswish_17 = paddle._C_ops.hardswish(batch_norm__114)

        # pd_op.conv2d: (16x192x9x9xf32) <- (16x192x9x9xf32, 192x192x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            hardswish_17, parameter_198, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x192x9x9xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (16x192x9x9xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__120,
            batch_norm__121,
            batch_norm__122,
            batch_norm__123,
            batch_norm__124,
            batch_norm__125,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_11,
                parameter_197,
                parameter_196,
                parameter_195,
                parameter_194,
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

        # pd_op.hardswish: (16x192x9x9xf32) <- (16x192x9x9xf32)
        hardswish_18 = paddle._C_ops.hardswish(batch_norm__120)

        # pd_op.depthwise_conv2d: (16x192x9x9xf32) <- (16x192x9x9xf32, 192x1x5x5xf32)
        depthwise_conv2d_9 = paddle._C_ops.depthwise_conv2d(
            hardswish_18, parameter_193, [1, 1], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (16x192x9x9xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (16x192x9x9xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__126,
            batch_norm__127,
            batch_norm__128,
            batch_norm__129,
            batch_norm__130,
            batch_norm__131,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_9,
                parameter_192,
                parameter_191,
                parameter_190,
                parameter_189,
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

        # pd_op.hardswish: (16x192x9x9xf32) <- (16x192x9x9xf32)
        hardswish_19 = paddle._C_ops.hardswish(batch_norm__126)

        # pd_op.conv2d: (16x96x9x9xf32) <- (16x192x9x9xf32, 96x192x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            hardswish_19, parameter_188, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x96x9x9xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (16x96x9x9xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__132,
            batch_norm__133,
            batch_norm__134,
            batch_norm__135,
            batch_norm__136,
            batch_norm__137,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_12,
                parameter_187,
                parameter_186,
                parameter_185,
                parameter_184,
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

        # pd_op.hardswish: (16x96x9x9xf32) <- (16x96x9x9xf32)
        hardswish_50 = paddle._C_ops.hardswish(batch_norm__132)

        # pd_op.depthwise_conv2d: (16x96x5x5xf32) <- (16x96x9x9xf32, 96x1x5x5xf32)
        depthwise_conv2d_10 = paddle._C_ops.depthwise_conv2d(
            hardswish_2, parameter_183, [2, 2], [2, 2], "EXPLICIT", 96, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (16x96x5x5xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (16x96x5x5xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__138,
            batch_norm__139,
            batch_norm__140,
            batch_norm__141,
            batch_norm__142,
            batch_norm__143,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_10,
                parameter_182,
                parameter_181,
                parameter_180,
                parameter_179,
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

        # pd_op.hardswish: (16x96x5x5xf32) <- (16x96x5x5xf32)
        hardswish_20 = paddle._C_ops.hardswish(batch_norm__138)

        # pd_op.conv2d: (16x96x5x5xf32) <- (16x96x5x5xf32, 96x96x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            hardswish_20, parameter_178, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x96x5x5xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (16x96x5x5xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__144,
            batch_norm__145,
            batch_norm__146,
            batch_norm__147,
            batch_norm__148,
            batch_norm__149,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_13,
                parameter_177,
                parameter_176,
                parameter_175,
                parameter_174,
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

        # pd_op.hardswish: (16x96x5x5xf32) <- (16x96x5x5xf32)
        hardswish_21 = paddle._C_ops.hardswish(batch_norm__144)

        # pd_op.depthwise_conv2d: (16x96x5x5xf32) <- (16x96x9x9xf32, 96x1x5x5xf32)
        depthwise_conv2d_11 = paddle._C_ops.depthwise_conv2d(
            hardswish_50, parameter_173, [2, 2], [2, 2], "EXPLICIT", 96, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (16x96x5x5xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (16x96x5x5xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__150,
            batch_norm__151,
            batch_norm__152,
            batch_norm__153,
            batch_norm__154,
            batch_norm__155,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_11,
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

        # pd_op.hardswish: (16x96x5x5xf32) <- (16x96x5x5xf32)
        hardswish_22 = paddle._C_ops.hardswish(batch_norm__150)

        # pd_op.conv2d: (16x96x5x5xf32) <- (16x96x5x5xf32, 96x96x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            hardswish_22, parameter_168, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x96x5x5xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (16x96x5x5xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__156,
            batch_norm__157,
            batch_norm__158,
            batch_norm__159,
            batch_norm__160,
            batch_norm__161,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_14,
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

        # pd_op.hardswish: (16x96x5x5xf32) <- (16x96x5x5xf32)
        hardswish_23 = paddle._C_ops.hardswish(batch_norm__156)

        # pd_op.add: (16x96x5x5xf32) <- (16x96x5x5xf32, 16x96x5x5xf32)
        add_12 = paddle._C_ops.add(hardswish_21, hardswish_23)

        # pd_op.shape64: (4xi64) <- (16x96x36x36xf32)
        shape64_0 = paddle._C_ops.shape64(hardswish_48)

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

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [3]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [4]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.depthwise_conv2d: (16x96x36x36xf32) <- (16x96x36x36xf32, 96x1x5x5xf32)
        depthwise_conv2d_12 = paddle._C_ops.depthwise_conv2d(
            hardswish_48, parameter_163, [1, 1], [2, 2], "EXPLICIT", 96, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (16x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (16x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__162,
            batch_norm__163,
            batch_norm__164,
            batch_norm__165,
            batch_norm__166,
            batch_norm__167,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_12,
                parameter_162,
                parameter_161,
                parameter_160,
                parameter_159,
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

        # pd_op.hardswish: (16x96x36x36xf32) <- (16x96x36x36xf32)
        hardswish_24 = paddle._C_ops.hardswish(batch_norm__162)

        # pd_op.conv2d: (16x96x36x36xf32) <- (16x96x36x36xf32, 96x96x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            hardswish_24, parameter_158, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (16x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__168,
            batch_norm__169,
            batch_norm__170,
            batch_norm__171,
            batch_norm__172,
            batch_norm__173,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_15,
                parameter_157,
                parameter_156,
                parameter_155,
                parameter_154,
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

        # pd_op.hardswish: (16x96x36x36xf32) <- (16x96x36x36xf32)
        hardswish_25 = paddle._C_ops.hardswish(batch_norm__168)

        # pd_op.depthwise_conv2d: (16x96x36x36xf32) <- (16x96x36x36xf32, 96x1x5x5xf32)
        depthwise_conv2d_13 = paddle._C_ops.depthwise_conv2d(
            hardswish_25, parameter_153, [1, 1], [2, 2], "EXPLICIT", 96, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (16x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (16x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__174,
            batch_norm__175,
            batch_norm__176,
            batch_norm__177,
            batch_norm__178,
            batch_norm__179,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_13,
                parameter_152,
                parameter_151,
                parameter_150,
                parameter_149,
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

        # pd_op.hardswish: (16x96x36x36xf32) <- (16x96x36x36xf32)
        hardswish_26 = paddle._C_ops.hardswish(batch_norm__174)

        # pd_op.conv2d: (16x96x36x36xf32) <- (16x96x36x36xf32, 96x96x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            hardswish_26, parameter_148, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (16x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__180,
            batch_norm__181,
            batch_norm__182,
            batch_norm__183,
            batch_norm__184,
            batch_norm__185,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_16,
                parameter_147,
                parameter_146,
                parameter_145,
                parameter_144,
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

        # pd_op.hardswish: (16x96x36x36xf32) <- (16x96x36x36xf32)
        hardswish_27 = paddle._C_ops.hardswish(batch_norm__180)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_16 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_10 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_4 = full_int_array_0

        # pd_op.pool2d: (16x96x1x1xf32) <- (16x96x36x36xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            hardswish_27,
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

        # pd_op.conv2d: (16x96x1x1xf32) <- (16x96x1x1xf32, 96x96x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            pool2d_0, parameter_143, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_6 = [1, -1, 1, 1]

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_142, full_int_array_6)

        # pd_op.add: (16x96x1x1xf32) <- (16x96x1x1xf32, 1x96x1x1xf32)
        add_13 = paddle._C_ops.add(conv2d_17, reshape_0)

        # pd_op.sigmoid: (16x96x1x1xf32) <- (16x96x1x1xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(add_13)

        # pd_op.multiply: (16x96x36x36xf32) <- (16x96x36x36xf32, 16x96x1x1xf32)
        multiply_0 = paddle._C_ops.multiply(hardswish_27, sigmoid_0)

        # pd_op.conv2d: (16x96x36x36xf32) <- (16x96x36x36xf32, 96x96x1x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            multiply_0, parameter_141, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (16x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__186,
            batch_norm__187,
            batch_norm__188,
            batch_norm__189,
            batch_norm__190,
            batch_norm__191,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_18,
                parameter_140,
                parameter_139,
                parameter_138,
                parameter_137,
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

        # pd_op.hardswish: (16x96x36x36xf32) <- (16x96x36x36xf32)
        hardswish_28 = paddle._C_ops.hardswish(batch_norm__186)

        # pd_op.conv2d: (16x4x36x36xf32) <- (16x96x36x36xf32, 4x96x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            hardswish_28, parameter_136, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x4x1x1xf32) <- (4xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_135, full_int_array_6)

        # pd_op.add: (16x4x36x36xf32) <- (16x4x36x36xf32, 1x4x1x1xf32)
        add_14 = paddle._C_ops.add(conv2d_19, reshape_1)

        # pd_op.conv2d: (16x32x36x36xf32) <- (16x96x36x36xf32, 32x96x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            hardswish_28, parameter_134, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(parameter_133, full_int_array_6)

        # pd_op.add: (16x32x36x36xf32) <- (16x32x36x36xf32, 1x32x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_20, reshape_2)

        # pd_op.conv2d: (16x1x36x36xf32) <- (16x96x36x36xf32, 1x96x5x5xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            hardswish_27, parameter_132, [1, 1], [2, 2], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x1x36x36xf32, 1xf32, 1xf32, 1xf32, 1xf32, -1xui8) <- (16x1x36x36xf32, 1xf32, 1xf32, 1xf32, 1xf32)
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

        # pd_op.hardswish: (16x1x36x36xf32) <- (16x1x36x36xf32)
        hardswish_29 = paddle._C_ops.hardswish(batch_norm__192)

        # pd_op.conv2d: (16x1x36x36xf32) <- (16x1x36x36xf32, 1x1x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(
            hardswish_29, parameter_127, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x1x36x36xf32, 1xf32, 1xf32, 1xf32, 1xf32, -1xui8) <- (16x1x36x36xf32, 1xf32, 1xf32, 1xf32, 1xf32)
        (
            batch_norm__326,
            batch_norm__198,
            batch_norm__199,
            batch_norm__200,
            batch_norm__201,
            batch_norm__202,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_22,
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

        # pd_op.sigmoid: (16x1x36x36xf32) <- (16x1x36x36xf32)
        sigmoid_1 = paddle._C_ops.sigmoid(batch_norm__326)

        # pd_op.sigmoid: (16x4x36x36xf32) <- (16x4x36x36xf32)
        sigmoid_2 = paddle._C_ops.sigmoid(add_14)

        # pd_op.multiply: (16x4x36x36xf32) <- (16x4x36x36xf32, 16x1x36x36xf32)
        multiply_4 = paddle._C_ops.multiply(sigmoid_2, sigmoid_1)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_17 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_11 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_5 = full_1

        # pd_op.scale: (16x4x36x36xf32) <- (16x4x36x36xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(multiply_4, full_1, float("1e-09"), True)

        # pd_op.sqrt: (16x4x36x36xf32) <- (16x4x36x36xf32)
        sqrt_0 = paddle._C_ops.sqrt(scale_12)

        # pd_op.transpose: (16x36x36x4xf32) <- (16x4x36x36xf32)
        transpose_12 = paddle._C_ops.transpose(sqrt_0, [0, 2, 3, 1])

        # pd_op.transpose: (16x36x36x32xf32) <- (16x32x36x36xf32)
        transpose_0 = paddle._C_ops.transpose(add_0, [0, 2, 3, 1])

        # pd_op.full: (1xf64) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_10 = paddle._C_ops.full(
            [1], float("36"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_11 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (36xf32) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_9, full_10, full_11, dtype="float32")

        # pd_op.scale: (36xf32) <- (36xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(arange_0, full_1, float("0.5"), True)

        # pd_op.full: (1xf32) <- ()
        full_12 = paddle._C_ops.full(
            [1], float("8"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_3 = full_12

        # pd_op.scale: (36xf32) <- (36xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(scale_13, full_12, float("0"), True)

        # builtin.combine: ([36xf32, 36xf32]) <- (36xf32, 36xf32)
        combine_4 = [scale_14, scale_14]

        # pd_op.meshgrid: ([36x36xf32, 36x36xf32]) <- ([36xf32, 36xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_4)

        # builtin.split: (36x36xf32, 36x36xf32) <- ([36x36xf32, 36x36xf32])
        (
            split_4,
            split_5,
        ) = meshgrid_0

        # pd_op.flatten: (1296xf32) <- (36x36xf32)
        flatten_0 = paddle._C_ops.flatten(split_4, 0, 1)

        # pd_op.flatten: (1296xf32) <- (36x36xf32)
        flatten_1 = paddle._C_ops.flatten(split_5, 0, 1)

        # builtin.combine: ([1296xf32, 1296xf32]) <- (1296xf32, 1296xf32)
        combine_5 = [flatten_1, flatten_0]

        # pd_op.stack: (1296x2xf32) <- ([1296xf32, 1296xf32])
        stack_0 = paddle._C_ops.stack(combine_5, -1)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_7 = [16, -1, 4]

        # pd_op.reshape: (16x1296x4xf32) <- (16x36x36x4xf32, 3xi64)
        reshape_12 = paddle._C_ops.reshape(transpose_12, full_int_array_7)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_8 = [-1, 8]

        # pd_op.reshape: (82944x8xf32) <- (16x36x36x32xf32, 2xi64)
        reshape_13 = paddle._C_ops.reshape(transpose_0, full_int_array_8)

        # pd_op.softmax: (82944x8xf32) <- (82944x8xf32)
        softmax_0 = paddle._C_ops.softmax(reshape_13, 1)

        # pd_op.matmul: (82944xf32) <- (82944x8xf32, 8xf32)
        matmul_0 = paddle._C_ops.matmul(softmax_0, data_3, False, False)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_9 = [-1, 4]

        # pd_op.reshape: (20736x4xf32) <- (82944xf32, 2xi64)
        reshape_14 = paddle._C_ops.reshape(matmul_0, full_int_array_9)

        # pd_op.scale: (20736x4xf32) <- (20736x4xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(reshape_14, full_12, float("0"), True)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_10 = [16, 1296, 4]

        # pd_op.reshape: (16x1296x4xf32) <- (20736x4xf32, 3xi64)
        reshape_15 = paddle._C_ops.reshape(scale_0, full_int_array_10)

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("2"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_19 = full_2

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_13 = full_2

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_7 = full_2

        # pd_op.split_with_num: ([16x1296x2xf32, 16x1296x2xf32]) <- (16x1296x4xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(reshape_15, 2, full_2)

        # builtin.split: (16x1296x2xf32, 16x1296x2xf32) <- ([16x1296x2xf32, 16x1296x2xf32])
        (
            split_6,
            split_0,
        ) = split_with_num_0

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_20 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_14 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_8 = full_3

        # pd_op.scale: (16x1296x2xf32) <- (16x1296x2xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(split_6, full_3, float("0"), True)

        # pd_op.add: (16x1296x2xf32) <- (16x1296x2xf32, 1296x2xf32)
        add_1 = paddle._C_ops.add(scale_1, stack_0)

        # pd_op.add: (16x1296x2xf32) <- (16x1296x2xf32, 1296x2xf32)
        add_2 = paddle._C_ops.add(split_0, stack_0)

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_21 = full_4

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_15 = full_4

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_9 = full_4

        # builtin.combine: ([16x1296x2xf32, 16x1296x2xf32]) <- (16x1296x2xf32, 16x1296x2xf32)
        combine_6 = [add_1, add_2]

        # pd_op.concat: (16x1296x4xf32) <- ([16x1296x2xf32, 16x1296x2xf32], 1xi32)
        concat_7 = paddle._C_ops.concat(combine_6, full_4)

        # pd_op.flatten: (16x4x1296xf32) <- (16x4x36x36xf32)
        flatten_2 = paddle._C_ops.flatten(sqrt_0, 2, 3)

        # pd_op.transpose: (16x1296x4xf32) <- (16x4x1296xf32)
        transpose_1 = paddle._C_ops.transpose(flatten_2, [0, 2, 1])

        # pd_op.flatten: (16x32x1296xf32) <- (16x32x36x36xf32)
        flatten_3 = paddle._C_ops.flatten(add_0, 2, 3)

        # pd_op.transpose: (16x1296x32xf32) <- (16x32x1296xf32)
        transpose_2 = paddle._C_ops.transpose(flatten_3, [0, 2, 1])

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (16x1296x4xf32) <- (16x1296x4xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(concat_7, full_5, float("0"), True)

        # pd_op.shape64: (4xi64) <- (16x96x18x18xf32)
        shape64_1 = paddle._C_ops.shape64(hardswish_49)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.depthwise_conv2d: (16x96x18x18xf32) <- (16x96x18x18xf32, 96x1x5x5xf32)
        depthwise_conv2d_14 = paddle._C_ops.depthwise_conv2d(
            hardswish_49, parameter_122, [1, 1], [2, 2], "EXPLICIT", 96, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (16x96x18x18xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (16x96x18x18xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__203,
            batch_norm__204,
            batch_norm__205,
            batch_norm__206,
            batch_norm__207,
            batch_norm__208,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_14,
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

        # pd_op.hardswish: (16x96x18x18xf32) <- (16x96x18x18xf32)
        hardswish_30 = paddle._C_ops.hardswish(batch_norm__203)

        # pd_op.conv2d: (16x96x18x18xf32) <- (16x96x18x18xf32, 96x96x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(
            hardswish_30, parameter_117, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x96x18x18xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (16x96x18x18xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__209,
            batch_norm__210,
            batch_norm__211,
            batch_norm__212,
            batch_norm__213,
            batch_norm__214,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_23,
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

        # pd_op.hardswish: (16x96x18x18xf32) <- (16x96x18x18xf32)
        hardswish_31 = paddle._C_ops.hardswish(batch_norm__209)

        # pd_op.depthwise_conv2d: (16x96x18x18xf32) <- (16x96x18x18xf32, 96x1x5x5xf32)
        depthwise_conv2d_15 = paddle._C_ops.depthwise_conv2d(
            hardswish_31, parameter_112, [1, 1], [2, 2], "EXPLICIT", 96, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (16x96x18x18xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (16x96x18x18xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__215,
            batch_norm__216,
            batch_norm__217,
            batch_norm__218,
            batch_norm__219,
            batch_norm__220,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_15,
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

        # pd_op.hardswish: (16x96x18x18xf32) <- (16x96x18x18xf32)
        hardswish_32 = paddle._C_ops.hardswish(batch_norm__215)

        # pd_op.conv2d: (16x96x18x18xf32) <- (16x96x18x18xf32, 96x96x1x1xf32)
        conv2d_24 = paddle._C_ops.conv2d(
            hardswish_32, parameter_107, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x96x18x18xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (16x96x18x18xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__221,
            batch_norm__222,
            batch_norm__223,
            batch_norm__224,
            batch_norm__225,
            batch_norm__226,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_24,
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

        # pd_op.hardswish: (16x96x18x18xf32) <- (16x96x18x18xf32)
        hardswish_33 = paddle._C_ops.hardswish(batch_norm__221)

        # pd_op.pool2d: (16x96x1x1xf32) <- (16x96x18x18xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(
            hardswish_33,
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

        # pd_op.conv2d: (16x96x1x1xf32) <- (16x96x1x1xf32, 96x96x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(
            pool2d_1, parameter_102, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(parameter_101, full_int_array_6)

        # pd_op.add: (16x96x1x1xf32) <- (16x96x1x1xf32, 1x96x1x1xf32)
        add_15 = paddle._C_ops.add(conv2d_25, reshape_3)

        # pd_op.sigmoid: (16x96x1x1xf32) <- (16x96x1x1xf32)
        sigmoid_3 = paddle._C_ops.sigmoid(add_15)

        # pd_op.multiply: (16x96x18x18xf32) <- (16x96x18x18xf32, 16x96x1x1xf32)
        multiply_1 = paddle._C_ops.multiply(hardswish_33, sigmoid_3)

        # pd_op.conv2d: (16x96x18x18xf32) <- (16x96x18x18xf32, 96x96x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(
            multiply_1, parameter_100, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x96x18x18xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (16x96x18x18xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__227,
            batch_norm__228,
            batch_norm__229,
            batch_norm__230,
            batch_norm__231,
            batch_norm__232,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_26,
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

        # pd_op.hardswish: (16x96x18x18xf32) <- (16x96x18x18xf32)
        hardswish_34 = paddle._C_ops.hardswish(batch_norm__227)

        # pd_op.conv2d: (16x4x18x18xf32) <- (16x96x18x18xf32, 4x96x1x1xf32)
        conv2d_27 = paddle._C_ops.conv2d(
            hardswish_34, parameter_95, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x4x1x1xf32) <- (4xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(parameter_94, full_int_array_6)

        # pd_op.add: (16x4x18x18xf32) <- (16x4x18x18xf32, 1x4x1x1xf32)
        add_16 = paddle._C_ops.add(conv2d_27, reshape_4)

        # pd_op.conv2d: (16x32x18x18xf32) <- (16x96x18x18xf32, 32x96x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(
            hardswish_34, parameter_93, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(parameter_92, full_int_array_6)

        # pd_op.add: (16x32x18x18xf32) <- (16x32x18x18xf32, 1x32x1x1xf32)
        add_3 = paddle._C_ops.add(conv2d_28, reshape_5)

        # pd_op.conv2d: (16x1x18x18xf32) <- (16x96x18x18xf32, 1x96x5x5xf32)
        conv2d_29 = paddle._C_ops.conv2d(
            hardswish_33, parameter_91, [1, 1], [2, 2], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x1x18x18xf32, 1xf32, 1xf32, 1xf32, 1xf32, -1xui8) <- (16x1x18x18xf32, 1xf32, 1xf32, 1xf32, 1xf32)
        (
            batch_norm__233,
            batch_norm__234,
            batch_norm__235,
            batch_norm__236,
            batch_norm__237,
            batch_norm__238,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_29,
                parameter_90,
                parameter_89,
                parameter_88,
                parameter_87,
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

        # pd_op.hardswish: (16x1x18x18xf32) <- (16x1x18x18xf32)
        hardswish_35 = paddle._C_ops.hardswish(batch_norm__233)

        # pd_op.conv2d: (16x1x18x18xf32) <- (16x1x18x18xf32, 1x1x1x1xf32)
        conv2d_30 = paddle._C_ops.conv2d(
            hardswish_35, parameter_86, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x1x18x18xf32, 1xf32, 1xf32, 1xf32, 1xf32, -1xui8) <- (16x1x18x18xf32, 1xf32, 1xf32, 1xf32, 1xf32)
        (
            batch_norm__327,
            batch_norm__239,
            batch_norm__240,
            batch_norm__241,
            batch_norm__242,
            batch_norm__243,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_30,
                parameter_85,
                parameter_84,
                parameter_83,
                parameter_82,
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

        # pd_op.sigmoid: (16x1x18x18xf32) <- (16x1x18x18xf32)
        sigmoid_4 = paddle._C_ops.sigmoid(batch_norm__327)

        # pd_op.sigmoid: (16x4x18x18xf32) <- (16x4x18x18xf32)
        sigmoid_5 = paddle._C_ops.sigmoid(add_16)

        # pd_op.multiply: (16x4x18x18xf32) <- (16x4x18x18xf32, 16x1x18x18xf32)
        multiply_5 = paddle._C_ops.multiply(sigmoid_5, sigmoid_4)

        # pd_op.scale: (16x4x18x18xf32) <- (16x4x18x18xf32, 1xf32)
        scale_15 = paddle._C_ops.scale(multiply_5, full_1, float("1e-09"), True)

        # pd_op.sqrt: (16x4x18x18xf32) <- (16x4x18x18xf32)
        sqrt_1 = paddle._C_ops.sqrt(scale_15)

        # pd_op.transpose: (16x18x18x4xf32) <- (16x4x18x18xf32)
        transpose_13 = paddle._C_ops.transpose(sqrt_1, [0, 2, 3, 1])

        # pd_op.transpose: (16x18x18x32xf32) <- (16x32x18x18xf32)
        transpose_3 = paddle._C_ops.transpose(add_3, [0, 2, 3, 1])

        # pd_op.full: (1xf64) <- ()
        full_13 = paddle._C_ops.full(
            [1], float("18"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (18xf32) <- (1xf64, 1xf64, 1xf64)
        arange_1 = paddle.arange(full_9, full_13, full_11, dtype="float32")

        # pd_op.scale: (18xf32) <- (18xf32, 1xf32)
        scale_16 = paddle._C_ops.scale(arange_1, full_1, float("0.5"), True)

        # pd_op.full: (1xf32) <- ()
        full_14 = paddle._C_ops.full(
            [1], float("16"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_6 = full_14

        # pd_op.scale: (18xf32) <- (18xf32, 1xf32)
        scale_17 = paddle._C_ops.scale(scale_16, full_14, float("0"), True)

        # builtin.combine: ([18xf32, 18xf32]) <- (18xf32, 18xf32)
        combine_7 = [scale_17, scale_17]

        # pd_op.meshgrid: ([18x18xf32, 18x18xf32]) <- ([18xf32, 18xf32])
        meshgrid_1 = paddle._C_ops.meshgrid(combine_7)

        # builtin.split: (18x18xf32, 18x18xf32) <- ([18x18xf32, 18x18xf32])
        (
            split_7,
            split_8,
        ) = meshgrid_1

        # pd_op.flatten: (324xf32) <- (18x18xf32)
        flatten_4 = paddle._C_ops.flatten(split_7, 0, 1)

        # pd_op.flatten: (324xf32) <- (18x18xf32)
        flatten_5 = paddle._C_ops.flatten(split_8, 0, 1)

        # builtin.combine: ([324xf32, 324xf32]) <- (324xf32, 324xf32)
        combine_8 = [flatten_5, flatten_4]

        # pd_op.stack: (324x2xf32) <- ([324xf32, 324xf32])
        stack_1 = paddle._C_ops.stack(combine_8, -1)

        # pd_op.reshape: (16x324x4xf32) <- (16x18x18x4xf32, 3xi64)
        reshape_16 = paddle._C_ops.reshape(transpose_13, full_int_array_7)

        # pd_op.reshape: (20736x8xf32) <- (16x18x18x32xf32, 2xi64)
        reshape_17 = paddle._C_ops.reshape(transpose_3, full_int_array_8)

        # pd_op.softmax: (20736x8xf32) <- (20736x8xf32)
        softmax_1 = paddle._C_ops.softmax(reshape_17, 1)

        # pd_op.matmul: (20736xf32) <- (20736x8xf32, 8xf32)
        matmul_1 = paddle._C_ops.matmul(softmax_1, data_3, False, False)

        # pd_op.reshape: (5184x4xf32) <- (20736xf32, 2xi64)
        reshape_18 = paddle._C_ops.reshape(matmul_1, full_int_array_9)

        # pd_op.scale: (5184x4xf32) <- (5184x4xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(reshape_18, full_14, float("0"), True)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_11 = [16, 324, 4]

        # pd_op.reshape: (16x324x4xf32) <- (5184x4xf32, 3xi64)
        reshape_19 = paddle._C_ops.reshape(scale_3, full_int_array_11)

        # pd_op.split_with_num: ([16x324x2xf32, 16x324x2xf32]) <- (16x324x4xf32, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(reshape_19, 2, full_2)

        # builtin.split: (16x324x2xf32, 16x324x2xf32) <- ([16x324x2xf32, 16x324x2xf32])
        (
            split_9,
            split_1,
        ) = split_with_num_1

        # pd_op.scale: (16x324x2xf32) <- (16x324x2xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(split_9, full_3, float("0"), True)

        # pd_op.add: (16x324x2xf32) <- (16x324x2xf32, 324x2xf32)
        add_4 = paddle._C_ops.add(scale_4, stack_1)

        # pd_op.add: (16x324x2xf32) <- (16x324x2xf32, 324x2xf32)
        add_5 = paddle._C_ops.add(split_1, stack_1)

        # builtin.combine: ([16x324x2xf32, 16x324x2xf32]) <- (16x324x2xf32, 16x324x2xf32)
        combine_9 = [add_4, add_5]

        # pd_op.concat: (16x324x4xf32) <- ([16x324x2xf32, 16x324x2xf32], 1xi32)
        concat_8 = paddle._C_ops.concat(combine_9, full_4)

        # pd_op.flatten: (16x4x324xf32) <- (16x4x18x18xf32)
        flatten_6 = paddle._C_ops.flatten(sqrt_1, 2, 3)

        # pd_op.transpose: (16x324x4xf32) <- (16x4x324xf32)
        transpose_4 = paddle._C_ops.transpose(flatten_6, [0, 2, 1])

        # pd_op.flatten: (16x32x324xf32) <- (16x32x18x18xf32)
        flatten_7 = paddle._C_ops.flatten(add_3, 2, 3)

        # pd_op.transpose: (16x324x32xf32) <- (16x32x324xf32)
        transpose_5 = paddle._C_ops.transpose(flatten_7, [0, 2, 1])

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("0.0625"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (16x324x4xf32) <- (16x324x4xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(concat_8, full_6, float("0"), True)

        # pd_op.shape64: (4xi64) <- (16x96x9x9xf32)
        shape64_2 = paddle._C_ops.shape64(hardswish_50)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.depthwise_conv2d: (16x96x9x9xf32) <- (16x96x9x9xf32, 96x1x5x5xf32)
        depthwise_conv2d_16 = paddle._C_ops.depthwise_conv2d(
            hardswish_50, parameter_81, [1, 1], [2, 2], "EXPLICIT", 96, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (16x96x9x9xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (16x96x9x9xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__244,
            batch_norm__245,
            batch_norm__246,
            batch_norm__247,
            batch_norm__248,
            batch_norm__249,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_16,
                parameter_80,
                parameter_79,
                parameter_78,
                parameter_77,
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

        # pd_op.hardswish: (16x96x9x9xf32) <- (16x96x9x9xf32)
        hardswish_36 = paddle._C_ops.hardswish(batch_norm__244)

        # pd_op.conv2d: (16x96x9x9xf32) <- (16x96x9x9xf32, 96x96x1x1xf32)
        conv2d_31 = paddle._C_ops.conv2d(
            hardswish_36, parameter_76, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x96x9x9xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (16x96x9x9xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__250,
            batch_norm__251,
            batch_norm__252,
            batch_norm__253,
            batch_norm__254,
            batch_norm__255,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_31,
                parameter_75,
                parameter_74,
                parameter_73,
                parameter_72,
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

        # pd_op.hardswish: (16x96x9x9xf32) <- (16x96x9x9xf32)
        hardswish_37 = paddle._C_ops.hardswish(batch_norm__250)

        # pd_op.depthwise_conv2d: (16x96x9x9xf32) <- (16x96x9x9xf32, 96x1x5x5xf32)
        depthwise_conv2d_17 = paddle._C_ops.depthwise_conv2d(
            hardswish_37, parameter_71, [1, 1], [2, 2], "EXPLICIT", 96, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (16x96x9x9xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (16x96x9x9xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__256,
            batch_norm__257,
            batch_norm__258,
            batch_norm__259,
            batch_norm__260,
            batch_norm__261,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_17,
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

        # pd_op.hardswish: (16x96x9x9xf32) <- (16x96x9x9xf32)
        hardswish_38 = paddle._C_ops.hardswish(batch_norm__256)

        # pd_op.conv2d: (16x96x9x9xf32) <- (16x96x9x9xf32, 96x96x1x1xf32)
        conv2d_32 = paddle._C_ops.conv2d(
            hardswish_38, parameter_66, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x96x9x9xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (16x96x9x9xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__262,
            batch_norm__263,
            batch_norm__264,
            batch_norm__265,
            batch_norm__266,
            batch_norm__267,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_32,
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

        # pd_op.hardswish: (16x96x9x9xf32) <- (16x96x9x9xf32)
        hardswish_39 = paddle._C_ops.hardswish(batch_norm__262)

        # pd_op.pool2d: (16x96x1x1xf32) <- (16x96x9x9xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(
            hardswish_39,
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

        # pd_op.conv2d: (16x96x1x1xf32) <- (16x96x1x1xf32, 96x96x1x1xf32)
        conv2d_33 = paddle._C_ops.conv2d(
            pool2d_2, parameter_61, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(parameter_60, full_int_array_6)

        # pd_op.add: (16x96x1x1xf32) <- (16x96x1x1xf32, 1x96x1x1xf32)
        add_17 = paddle._C_ops.add(conv2d_33, reshape_6)

        # pd_op.sigmoid: (16x96x1x1xf32) <- (16x96x1x1xf32)
        sigmoid_6 = paddle._C_ops.sigmoid(add_17)

        # pd_op.multiply: (16x96x9x9xf32) <- (16x96x9x9xf32, 16x96x1x1xf32)
        multiply_2 = paddle._C_ops.multiply(hardswish_39, sigmoid_6)

        # pd_op.conv2d: (16x96x9x9xf32) <- (16x96x9x9xf32, 96x96x1x1xf32)
        conv2d_34 = paddle._C_ops.conv2d(
            multiply_2, parameter_59, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x96x9x9xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (16x96x9x9xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__268,
            batch_norm__269,
            batch_norm__270,
            batch_norm__271,
            batch_norm__272,
            batch_norm__273,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_34,
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

        # pd_op.hardswish: (16x96x9x9xf32) <- (16x96x9x9xf32)
        hardswish_40 = paddle._C_ops.hardswish(batch_norm__268)

        # pd_op.conv2d: (16x4x9x9xf32) <- (16x96x9x9xf32, 4x96x1x1xf32)
        conv2d_35 = paddle._C_ops.conv2d(
            hardswish_40, parameter_54, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x4x1x1xf32) <- (4xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(parameter_53, full_int_array_6)

        # pd_op.add: (16x4x9x9xf32) <- (16x4x9x9xf32, 1x4x1x1xf32)
        add_18 = paddle._C_ops.add(conv2d_35, reshape_7)

        # pd_op.conv2d: (16x32x9x9xf32) <- (16x96x9x9xf32, 32x96x1x1xf32)
        conv2d_36 = paddle._C_ops.conv2d(
            hardswish_40, parameter_52, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(parameter_51, full_int_array_6)

        # pd_op.add: (16x32x9x9xf32) <- (16x32x9x9xf32, 1x32x1x1xf32)
        add_6 = paddle._C_ops.add(conv2d_36, reshape_8)

        # pd_op.conv2d: (16x1x9x9xf32) <- (16x96x9x9xf32, 1x96x5x5xf32)
        conv2d_37 = paddle._C_ops.conv2d(
            hardswish_39, parameter_50, [1, 1], [2, 2], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x1x9x9xf32, 1xf32, 1xf32, 1xf32, 1xf32, -1xui8) <- (16x1x9x9xf32, 1xf32, 1xf32, 1xf32, 1xf32)
        (
            batch_norm__274,
            batch_norm__275,
            batch_norm__276,
            batch_norm__277,
            batch_norm__278,
            batch_norm__279,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_37,
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

        # pd_op.hardswish: (16x1x9x9xf32) <- (16x1x9x9xf32)
        hardswish_41 = paddle._C_ops.hardswish(batch_norm__274)

        # pd_op.conv2d: (16x1x9x9xf32) <- (16x1x9x9xf32, 1x1x1x1xf32)
        conv2d_38 = paddle._C_ops.conv2d(
            hardswish_41, parameter_45, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x1x9x9xf32, 1xf32, 1xf32, 1xf32, 1xf32, -1xui8) <- (16x1x9x9xf32, 1xf32, 1xf32, 1xf32, 1xf32)
        (
            batch_norm__328,
            batch_norm__280,
            batch_norm__281,
            batch_norm__282,
            batch_norm__283,
            batch_norm__284,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_38,
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

        # pd_op.sigmoid: (16x1x9x9xf32) <- (16x1x9x9xf32)
        sigmoid_7 = paddle._C_ops.sigmoid(batch_norm__328)

        # pd_op.sigmoid: (16x4x9x9xf32) <- (16x4x9x9xf32)
        sigmoid_8 = paddle._C_ops.sigmoid(add_18)

        # pd_op.multiply: (16x4x9x9xf32) <- (16x4x9x9xf32, 16x1x9x9xf32)
        multiply_6 = paddle._C_ops.multiply(sigmoid_8, sigmoid_7)

        # pd_op.scale: (16x4x9x9xf32) <- (16x4x9x9xf32, 1xf32)
        scale_18 = paddle._C_ops.scale(multiply_6, full_1, float("1e-09"), True)

        # pd_op.sqrt: (16x4x9x9xf32) <- (16x4x9x9xf32)
        sqrt_2 = paddle._C_ops.sqrt(scale_18)

        # pd_op.transpose: (16x9x9x4xf32) <- (16x4x9x9xf32)
        transpose_14 = paddle._C_ops.transpose(sqrt_2, [0, 2, 3, 1])

        # pd_op.transpose: (16x9x9x32xf32) <- (16x32x9x9xf32)
        transpose_6 = paddle._C_ops.transpose(add_6, [0, 2, 3, 1])

        # pd_op.full: (1xf64) <- ()
        full_15 = paddle._C_ops.full(
            [1], float("9"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (9xf32) <- (1xf64, 1xf64, 1xf64)
        arange_2 = paddle.arange(full_9, full_15, full_11, dtype="float32")

        # pd_op.scale: (9xf32) <- (9xf32, 1xf32)
        scale_19 = paddle._C_ops.scale(arange_2, full_1, float("0.5"), True)

        # pd_op.full: (1xf32) <- ()
        full_16 = paddle._C_ops.full(
            [1], float("32"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_12 = full_16

        # pd_op.scale: (9xf32) <- (9xf32, 1xf32)
        scale_20 = paddle._C_ops.scale(scale_19, full_16, float("0"), True)

        # builtin.combine: ([9xf32, 9xf32]) <- (9xf32, 9xf32)
        combine_10 = [scale_20, scale_20]

        # pd_op.meshgrid: ([9x9xf32, 9x9xf32]) <- ([9xf32, 9xf32])
        meshgrid_2 = paddle._C_ops.meshgrid(combine_10)

        # builtin.split: (9x9xf32, 9x9xf32) <- ([9x9xf32, 9x9xf32])
        (
            split_10,
            split_11,
        ) = meshgrid_2

        # pd_op.flatten: (81xf32) <- (9x9xf32)
        flatten_8 = paddle._C_ops.flatten(split_10, 0, 1)

        # pd_op.flatten: (81xf32) <- (9x9xf32)
        flatten_9 = paddle._C_ops.flatten(split_11, 0, 1)

        # builtin.combine: ([81xf32, 81xf32]) <- (81xf32, 81xf32)
        combine_11 = [flatten_9, flatten_8]

        # pd_op.stack: (81x2xf32) <- ([81xf32, 81xf32])
        stack_2 = paddle._C_ops.stack(combine_11, -1)

        # pd_op.reshape: (16x81x4xf32) <- (16x9x9x4xf32, 3xi64)
        reshape_20 = paddle._C_ops.reshape(transpose_14, full_int_array_7)

        # pd_op.reshape: (5184x8xf32) <- (16x9x9x32xf32, 2xi64)
        reshape_21 = paddle._C_ops.reshape(transpose_6, full_int_array_8)

        # pd_op.softmax: (5184x8xf32) <- (5184x8xf32)
        softmax_2 = paddle._C_ops.softmax(reshape_21, 1)

        # pd_op.matmul: (5184xf32) <- (5184x8xf32, 8xf32)
        matmul_2 = paddle._C_ops.matmul(softmax_2, data_3, False, False)

        # pd_op.reshape: (1296x4xf32) <- (5184xf32, 2xi64)
        reshape_22 = paddle._C_ops.reshape(matmul_2, full_int_array_9)

        # pd_op.scale: (1296x4xf32) <- (1296x4xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(reshape_22, full_16, float("0"), True)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_12 = [16, 81, 4]

        # pd_op.reshape: (16x81x4xf32) <- (1296x4xf32, 3xi64)
        reshape_23 = paddle._C_ops.reshape(scale_6, full_int_array_12)

        # pd_op.split_with_num: ([16x81x2xf32, 16x81x2xf32]) <- (16x81x4xf32, 1xi32)
        split_with_num_2 = paddle._C_ops.split_with_num(reshape_23, 2, full_2)

        # builtin.split: (16x81x2xf32, 16x81x2xf32) <- ([16x81x2xf32, 16x81x2xf32])
        (
            split_12,
            split_2,
        ) = split_with_num_2

        # pd_op.scale: (16x81x2xf32) <- (16x81x2xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(split_12, full_3, float("0"), True)

        # pd_op.add: (16x81x2xf32) <- (16x81x2xf32, 81x2xf32)
        add_7 = paddle._C_ops.add(scale_7, stack_2)

        # pd_op.add: (16x81x2xf32) <- (16x81x2xf32, 81x2xf32)
        add_8 = paddle._C_ops.add(split_2, stack_2)

        # builtin.combine: ([16x81x2xf32, 16x81x2xf32]) <- (16x81x2xf32, 16x81x2xf32)
        combine_12 = [add_7, add_8]

        # pd_op.concat: (16x81x4xf32) <- ([16x81x2xf32, 16x81x2xf32], 1xi32)
        concat_9 = paddle._C_ops.concat(combine_12, full_4)

        # pd_op.flatten: (16x4x81xf32) <- (16x4x9x9xf32)
        flatten_10 = paddle._C_ops.flatten(sqrt_2, 2, 3)

        # pd_op.transpose: (16x81x4xf32) <- (16x4x81xf32)
        transpose_7 = paddle._C_ops.transpose(flatten_10, [0, 2, 1])

        # pd_op.flatten: (16x32x81xf32) <- (16x32x9x9xf32)
        flatten_11 = paddle._C_ops.flatten(add_6, 2, 3)

        # pd_op.transpose: (16x81x32xf32) <- (16x32x81xf32)
        transpose_8 = paddle._C_ops.transpose(flatten_11, [0, 2, 1])

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("0.03125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (16x81x4xf32) <- (16x81x4xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(concat_9, full_7, float("0"), True)

        # pd_op.shape64: (4xi64) <- (16x96x5x5xf32)
        shape64_3 = paddle._C_ops.shape64(add_12)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.depthwise_conv2d: (16x96x5x5xf32) <- (16x96x5x5xf32, 96x1x5x5xf32)
        depthwise_conv2d_18 = paddle._C_ops.depthwise_conv2d(
            add_12, parameter_40, [1, 1], [2, 2], "EXPLICIT", 96, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (16x96x5x5xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (16x96x5x5xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__285,
            batch_norm__286,
            batch_norm__287,
            batch_norm__288,
            batch_norm__289,
            batch_norm__290,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_18,
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

        # pd_op.hardswish: (16x96x5x5xf32) <- (16x96x5x5xf32)
        hardswish_42 = paddle._C_ops.hardswish(batch_norm__285)

        # pd_op.conv2d: (16x96x5x5xf32) <- (16x96x5x5xf32, 96x96x1x1xf32)
        conv2d_39 = paddle._C_ops.conv2d(
            hardswish_42, parameter_35, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x96x5x5xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (16x96x5x5xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__291,
            batch_norm__292,
            batch_norm__293,
            batch_norm__294,
            batch_norm__295,
            batch_norm__296,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_39,
                parameter_34,
                parameter_33,
                parameter_32,
                parameter_31,
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

        # pd_op.hardswish: (16x96x5x5xf32) <- (16x96x5x5xf32)
        hardswish_43 = paddle._C_ops.hardswish(batch_norm__291)

        # pd_op.depthwise_conv2d: (16x96x5x5xf32) <- (16x96x5x5xf32, 96x1x5x5xf32)
        depthwise_conv2d_19 = paddle._C_ops.depthwise_conv2d(
            hardswish_43, parameter_30, [1, 1], [2, 2], "EXPLICIT", 96, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (16x96x5x5xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (16x96x5x5xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__297,
            batch_norm__298,
            batch_norm__299,
            batch_norm__300,
            batch_norm__301,
            batch_norm__302,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_19,
                parameter_29,
                parameter_28,
                parameter_27,
                parameter_26,
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

        # pd_op.hardswish: (16x96x5x5xf32) <- (16x96x5x5xf32)
        hardswish_44 = paddle._C_ops.hardswish(batch_norm__297)

        # pd_op.conv2d: (16x96x5x5xf32) <- (16x96x5x5xf32, 96x96x1x1xf32)
        conv2d_40 = paddle._C_ops.conv2d(
            hardswish_44, parameter_25, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x96x5x5xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (16x96x5x5xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__303,
            batch_norm__304,
            batch_norm__305,
            batch_norm__306,
            batch_norm__307,
            batch_norm__308,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_40,
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

        # pd_op.hardswish: (16x96x5x5xf32) <- (16x96x5x5xf32)
        hardswish_45 = paddle._C_ops.hardswish(batch_norm__303)

        # pd_op.pool2d: (16x96x1x1xf32) <- (16x96x5x5xf32, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(
            hardswish_45,
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

        # pd_op.conv2d: (16x96x1x1xf32) <- (16x96x1x1xf32, 96x96x1x1xf32)
        conv2d_41 = paddle._C_ops.conv2d(
            pool2d_3, parameter_20, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(parameter_19, full_int_array_6)

        # pd_op.add: (16x96x1x1xf32) <- (16x96x1x1xf32, 1x96x1x1xf32)
        add_19 = paddle._C_ops.add(conv2d_41, reshape_9)

        # pd_op.sigmoid: (16x96x1x1xf32) <- (16x96x1x1xf32)
        sigmoid_9 = paddle._C_ops.sigmoid(add_19)

        # pd_op.multiply: (16x96x5x5xf32) <- (16x96x5x5xf32, 16x96x1x1xf32)
        multiply_3 = paddle._C_ops.multiply(hardswish_45, sigmoid_9)

        # pd_op.conv2d: (16x96x5x5xf32) <- (16x96x5x5xf32, 96x96x1x1xf32)
        conv2d_42 = paddle._C_ops.conv2d(
            multiply_3, parameter_18, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x96x5x5xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (16x96x5x5xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__309,
            batch_norm__310,
            batch_norm__311,
            batch_norm__312,
            batch_norm__313,
            batch_norm__314,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_42,
                parameter_17,
                parameter_16,
                parameter_15,
                parameter_14,
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

        # pd_op.hardswish: (16x96x5x5xf32) <- (16x96x5x5xf32)
        hardswish_46 = paddle._C_ops.hardswish(batch_norm__309)

        # pd_op.conv2d: (16x4x5x5xf32) <- (16x96x5x5xf32, 4x96x1x1xf32)
        conv2d_43 = paddle._C_ops.conv2d(
            hardswish_46, parameter_13, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x4x1x1xf32) <- (4xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(parameter_12, full_int_array_6)

        # pd_op.add: (16x4x5x5xf32) <- (16x4x5x5xf32, 1x4x1x1xf32)
        add_20 = paddle._C_ops.add(conv2d_43, reshape_10)

        # pd_op.conv2d: (16x32x5x5xf32) <- (16x96x5x5xf32, 32x96x1x1xf32)
        conv2d_44 = paddle._C_ops.conv2d(
            hardswish_46, parameter_11, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        reshape_11 = paddle._C_ops.reshape(parameter_10, full_int_array_6)

        # pd_op.add: (16x32x5x5xf32) <- (16x32x5x5xf32, 1x32x1x1xf32)
        add_9 = paddle._C_ops.add(conv2d_44, reshape_11)

        # pd_op.conv2d: (16x1x5x5xf32) <- (16x96x5x5xf32, 1x96x5x5xf32)
        conv2d_45 = paddle._C_ops.conv2d(
            hardswish_45, parameter_9, [1, 1], [2, 2], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x1x5x5xf32, 1xf32, 1xf32, 1xf32, 1xf32, -1xui8) <- (16x1x5x5xf32, 1xf32, 1xf32, 1xf32, 1xf32)
        (
            batch_norm__315,
            batch_norm__316,
            batch_norm__317,
            batch_norm__318,
            batch_norm__319,
            batch_norm__320,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_45,
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

        # pd_op.hardswish: (16x1x5x5xf32) <- (16x1x5x5xf32)
        hardswish_47 = paddle._C_ops.hardswish(batch_norm__315)

        # pd_op.conv2d: (16x1x5x5xf32) <- (16x1x5x5xf32, 1x1x1x1xf32)
        conv2d_46 = paddle._C_ops.conv2d(
            hardswish_47, parameter_4, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x1x5x5xf32, 1xf32, 1xf32, 1xf32, 1xf32, -1xui8) <- (16x1x5x5xf32, 1xf32, 1xf32, 1xf32, 1xf32)
        (
            batch_norm__329,
            batch_norm__321,
            batch_norm__322,
            batch_norm__323,
            batch_norm__324,
            batch_norm__325,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_46,
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

        # pd_op.sigmoid: (16x1x5x5xf32) <- (16x1x5x5xf32)
        sigmoid_10 = paddle._C_ops.sigmoid(batch_norm__329)

        # pd_op.sigmoid: (16x4x5x5xf32) <- (16x4x5x5xf32)
        sigmoid_11 = paddle._C_ops.sigmoid(add_20)

        # pd_op.multiply: (16x4x5x5xf32) <- (16x4x5x5xf32, 16x1x5x5xf32)
        multiply_7 = paddle._C_ops.multiply(sigmoid_11, sigmoid_10)

        # pd_op.scale: (16x4x5x5xf32) <- (16x4x5x5xf32, 1xf32)
        scale_21 = paddle._C_ops.scale(multiply_7, full_1, float("1e-09"), True)

        # pd_op.sqrt: (16x4x5x5xf32) <- (16x4x5x5xf32)
        sqrt_3 = paddle._C_ops.sqrt(scale_21)

        # pd_op.transpose: (16x5x5x4xf32) <- (16x4x5x5xf32)
        transpose_15 = paddle._C_ops.transpose(sqrt_3, [0, 2, 3, 1])

        # pd_op.transpose: (16x5x5x32xf32) <- (16x32x5x5xf32)
        transpose_9 = paddle._C_ops.transpose(add_9, [0, 2, 3, 1])

        # pd_op.full: (1xf64) <- ()
        full_17 = paddle._C_ops.full(
            [1], float("5"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (5xf32) <- (1xf64, 1xf64, 1xf64)
        arange_3 = paddle.arange(full_9, full_17, full_11, dtype="float32")

        # pd_op.scale: (5xf32) <- (5xf32, 1xf32)
        scale_22 = paddle._C_ops.scale(arange_3, full_1, float("0.5"), True)

        # pd_op.full: (1xf32) <- ()
        full_18 = paddle._C_ops.full(
            [1], float("64"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_18 = full_18

        # pd_op.scale: (5xf32) <- (5xf32, 1xf32)
        scale_23 = paddle._C_ops.scale(scale_22, full_18, float("0"), True)

        # builtin.combine: ([5xf32, 5xf32]) <- (5xf32, 5xf32)
        combine_13 = [scale_23, scale_23]

        # pd_op.meshgrid: ([5x5xf32, 5x5xf32]) <- ([5xf32, 5xf32])
        meshgrid_3 = paddle._C_ops.meshgrid(combine_13)

        # builtin.split: (5x5xf32, 5x5xf32) <- ([5x5xf32, 5x5xf32])
        (
            split_13,
            split_14,
        ) = meshgrid_3

        # pd_op.flatten: (25xf32) <- (5x5xf32)
        flatten_12 = paddle._C_ops.flatten(split_13, 0, 1)

        # pd_op.flatten: (25xf32) <- (5x5xf32)
        flatten_13 = paddle._C_ops.flatten(split_14, 0, 1)

        # builtin.combine: ([25xf32, 25xf32]) <- (25xf32, 25xf32)
        combine_14 = [flatten_13, flatten_12]

        # pd_op.stack: (25x2xf32) <- ([25xf32, 25xf32])
        stack_3 = paddle._C_ops.stack(combine_14, -1)

        # pd_op.reshape: (16x25x4xf32) <- (16x5x5x4xf32, 3xi64)
        reshape_24 = paddle._C_ops.reshape(transpose_15, full_int_array_7)

        # pd_op.reshape: (1600x8xf32) <- (16x5x5x32xf32, 2xi64)
        reshape_25 = paddle._C_ops.reshape(transpose_9, full_int_array_8)

        # pd_op.softmax: (1600x8xf32) <- (1600x8xf32)
        softmax_3 = paddle._C_ops.softmax(reshape_25, 1)

        # pd_op.matmul: (1600xf32) <- (1600x8xf32, 8xf32)
        matmul_3 = paddle._C_ops.matmul(softmax_3, data_3, False, False)

        # pd_op.reshape: (400x4xf32) <- (1600xf32, 2xi64)
        reshape_26 = paddle._C_ops.reshape(matmul_3, full_int_array_9)

        # pd_op.scale: (400x4xf32) <- (400x4xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(reshape_26, full_18, float("0"), True)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_13 = [16, 25, 4]

        # pd_op.reshape: (16x25x4xf32) <- (400x4xf32, 3xi64)
        reshape_27 = paddle._C_ops.reshape(scale_9, full_int_array_13)

        # pd_op.split_with_num: ([16x25x2xf32, 16x25x2xf32]) <- (16x25x4xf32, 1xi32)
        split_with_num_3 = paddle._C_ops.split_with_num(reshape_27, 2, full_2)

        # builtin.split: (16x25x2xf32, 16x25x2xf32) <- ([16x25x2xf32, 16x25x2xf32])
        (
            split_15,
            split_3,
        ) = split_with_num_3

        # pd_op.scale: (16x25x2xf32) <- (16x25x2xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(split_15, full_3, float("0"), True)

        # pd_op.add: (16x25x2xf32) <- (16x25x2xf32, 25x2xf32)
        add_10 = paddle._C_ops.add(scale_10, stack_3)

        # pd_op.add: (16x25x2xf32) <- (16x25x2xf32, 25x2xf32)
        add_11 = paddle._C_ops.add(split_3, stack_3)

        # builtin.combine: ([16x25x2xf32, 16x25x2xf32]) <- (16x25x2xf32, 16x25x2xf32)
        combine_15 = [add_10, add_11]

        # pd_op.concat: (16x25x4xf32) <- ([16x25x2xf32, 16x25x2xf32], 1xi32)
        concat_10 = paddle._C_ops.concat(combine_15, full_4)

        # pd_op.flatten: (16x4x25xf32) <- (16x4x5x5xf32)
        flatten_14 = paddle._C_ops.flatten(sqrt_3, 2, 3)

        # pd_op.transpose: (16x25x4xf32) <- (16x4x25xf32)
        transpose_10 = paddle._C_ops.transpose(flatten_14, [0, 2, 1])

        # pd_op.flatten: (16x32x25xf32) <- (16x32x5x5xf32)
        flatten_15 = paddle._C_ops.flatten(add_9, 2, 3)

        # pd_op.transpose: (16x25x32xf32) <- (16x32x25xf32)
        transpose_11 = paddle._C_ops.transpose(flatten_15, [0, 2, 1])

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("0.015625"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (16x25x4xf32) <- (16x25x4xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(concat_10, full_8, float("0"), True)

        # builtin.combine: ([16x1296x4xf32, 16x324x4xf32, 16x81x4xf32, 16x25x4xf32]) <- (16x1296x4xf32, 16x324x4xf32, 16x81x4xf32, 16x25x4xf32)
        combine_16 = [transpose_1, transpose_4, transpose_7, transpose_10]

        # pd_op.concat: (16x1726x4xf32) <- ([16x1296x4xf32, 16x324x4xf32, 16x81x4xf32, 16x25x4xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_16, full_0)

        # builtin.combine: ([16x1296x4xf32, 16x324x4xf32, 16x81x4xf32, 16x25x4xf32]) <- (16x1296x4xf32, 16x324x4xf32, 16x81x4xf32, 16x25x4xf32)
        combine_17 = [scale_2, scale_5, scale_8, scale_11]

        # pd_op.concat: (16x1726x4xf32) <- ([16x1296x4xf32, 16x324x4xf32, 16x81x4xf32, 16x25x4xf32], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_17, full_0)

        # builtin.combine: ([16x1296x32xf32, 16x324x32xf32, 16x81x32xf32, 16x25x32xf32]) <- (16x1296x32xf32, 16x324x32xf32, 16x81x32xf32, 16x25x32xf32)
        combine_18 = [transpose_2, transpose_5, transpose_8, transpose_11]

        # pd_op.concat: (16x1726x32xf32) <- ([16x1296x32xf32, 16x324x32xf32, 16x81x32xf32, 16x25x32xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_18, full_0)
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
            batch_norm__11,
            hardswish_1,
            conv2d_2,
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            batch_norm__16,
            batch_norm__17,
            hardswish_2,
            nearest_interp_0,
            full_0,
            concat_0,
            depthwise_conv2d_0,
            batch_norm__18,
            batch_norm__19,
            batch_norm__20,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
            hardswish_3,
            conv2d_3,
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
            hardswish_4,
            depthwise_conv2d_1,
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
            hardswish_5,
            conv2d_4,
            batch_norm__36,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
            hardswish_6,
            nearest_interp_1,
            assign_0,
            concat_1,
            depthwise_conv2d_2,
            batch_norm__42,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
            hardswish_7,
            conv2d_5,
            batch_norm__48,
            batch_norm__49,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
            hardswish_8,
            depthwise_conv2d_3,
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
            hardswish_9,
            conv2d_6,
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
            depthwise_conv2d_4,
            batch_norm__66,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            batch_norm__70,
            batch_norm__71,
            hardswish_10,
            conv2d_7,
            batch_norm__72,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
            hardswish_11,
            assign_1,
            concat_2,
            depthwise_conv2d_5,
            batch_norm__78,
            batch_norm__79,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
            batch_norm__83,
            hardswish_12,
            conv2d_8,
            batch_norm__84,
            batch_norm__85,
            batch_norm__86,
            batch_norm__87,
            batch_norm__88,
            batch_norm__89,
            hardswish_13,
            depthwise_conv2d_6,
            batch_norm__90,
            batch_norm__91,
            batch_norm__92,
            batch_norm__93,
            batch_norm__94,
            batch_norm__95,
            hardswish_14,
            conv2d_9,
            batch_norm__96,
            batch_norm__97,
            batch_norm__98,
            batch_norm__99,
            batch_norm__100,
            batch_norm__101,
            depthwise_conv2d_7,
            batch_norm__102,
            batch_norm__103,
            batch_norm__104,
            batch_norm__105,
            batch_norm__106,
            batch_norm__107,
            hardswish_15,
            conv2d_10,
            batch_norm__108,
            batch_norm__109,
            batch_norm__110,
            batch_norm__111,
            batch_norm__112,
            batch_norm__113,
            hardswish_16,
            assign_2,
            concat_3,
            depthwise_conv2d_8,
            batch_norm__114,
            batch_norm__115,
            batch_norm__116,
            batch_norm__117,
            batch_norm__118,
            batch_norm__119,
            hardswish_17,
            conv2d_11,
            batch_norm__120,
            batch_norm__121,
            batch_norm__122,
            batch_norm__123,
            batch_norm__124,
            batch_norm__125,
            hardswish_18,
            depthwise_conv2d_9,
            batch_norm__126,
            batch_norm__127,
            batch_norm__128,
            batch_norm__129,
            batch_norm__130,
            batch_norm__131,
            hardswish_19,
            conv2d_12,
            batch_norm__132,
            batch_norm__133,
            batch_norm__134,
            batch_norm__135,
            batch_norm__136,
            batch_norm__137,
            depthwise_conv2d_10,
            batch_norm__138,
            batch_norm__139,
            batch_norm__140,
            batch_norm__141,
            batch_norm__142,
            batch_norm__143,
            hardswish_20,
            conv2d_13,
            batch_norm__144,
            batch_norm__145,
            batch_norm__146,
            batch_norm__147,
            batch_norm__148,
            batch_norm__149,
            hardswish_21,
            depthwise_conv2d_11,
            batch_norm__150,
            batch_norm__151,
            batch_norm__152,
            batch_norm__153,
            batch_norm__154,
            batch_norm__155,
            hardswish_22,
            conv2d_14,
            batch_norm__156,
            batch_norm__157,
            batch_norm__158,
            batch_norm__159,
            batch_norm__160,
            batch_norm__161,
            hardswish_23,
            depthwise_conv2d_12,
            batch_norm__162,
            batch_norm__163,
            batch_norm__164,
            batch_norm__165,
            batch_norm__166,
            batch_norm__167,
            hardswish_24,
            conv2d_15,
            batch_norm__168,
            batch_norm__169,
            batch_norm__170,
            batch_norm__171,
            batch_norm__172,
            batch_norm__173,
            hardswish_25,
            depthwise_conv2d_13,
            batch_norm__174,
            batch_norm__175,
            batch_norm__176,
            batch_norm__177,
            batch_norm__178,
            batch_norm__179,
            hardswish_26,
            conv2d_16,
            batch_norm__180,
            batch_norm__181,
            batch_norm__182,
            batch_norm__183,
            batch_norm__184,
            batch_norm__185,
            hardswish_27,
            full_int_array_0,
            pool2d_0,
            conv2d_17,
            reshape_0,
            sigmoid_0,
            multiply_0,
            conv2d_18,
            batch_norm__186,
            batch_norm__187,
            batch_norm__188,
            batch_norm__189,
            batch_norm__190,
            batch_norm__191,
            hardswish_28,
            conv2d_19,
            reshape_1,
            conv2d_20,
            reshape_2,
            add_0,
            conv2d_21,
            batch_norm__192,
            batch_norm__193,
            batch_norm__194,
            batch_norm__195,
            batch_norm__196,
            batch_norm__197,
            hardswish_29,
            conv2d_22,
            batch_norm__198,
            batch_norm__199,
            batch_norm__200,
            batch_norm__201,
            batch_norm__202,
            sigmoid_1,
            sigmoid_2,
            full_1,
            sqrt_0,
            transpose_0,
            stack_0,
            softmax_0,
            matmul_0,
            assign_3,
            scale_0,
            full_2,
            split_0,
            full_3,
            scale_1,
            add_1,
            add_2,
            full_4,
            transpose_1,
            transpose_2,
            full_5,
            scale_2,
            depthwise_conv2d_14,
            batch_norm__203,
            batch_norm__204,
            batch_norm__205,
            batch_norm__206,
            batch_norm__207,
            batch_norm__208,
            hardswish_30,
            conv2d_23,
            batch_norm__209,
            batch_norm__210,
            batch_norm__211,
            batch_norm__212,
            batch_norm__213,
            batch_norm__214,
            hardswish_31,
            depthwise_conv2d_15,
            batch_norm__215,
            batch_norm__216,
            batch_norm__217,
            batch_norm__218,
            batch_norm__219,
            batch_norm__220,
            hardswish_32,
            conv2d_24,
            batch_norm__221,
            batch_norm__222,
            batch_norm__223,
            batch_norm__224,
            batch_norm__225,
            batch_norm__226,
            hardswish_33,
            assign_4,
            pool2d_1,
            conv2d_25,
            reshape_3,
            sigmoid_3,
            multiply_1,
            conv2d_26,
            batch_norm__227,
            batch_norm__228,
            batch_norm__229,
            batch_norm__230,
            batch_norm__231,
            batch_norm__232,
            hardswish_34,
            conv2d_27,
            reshape_4,
            conv2d_28,
            reshape_5,
            add_3,
            conv2d_29,
            batch_norm__233,
            batch_norm__234,
            batch_norm__235,
            batch_norm__236,
            batch_norm__237,
            batch_norm__238,
            hardswish_35,
            conv2d_30,
            batch_norm__239,
            batch_norm__240,
            batch_norm__241,
            batch_norm__242,
            batch_norm__243,
            sigmoid_4,
            sigmoid_5,
            assign_5,
            sqrt_1,
            transpose_3,
            stack_1,
            softmax_1,
            matmul_1,
            assign_6,
            scale_3,
            assign_7,
            split_1,
            assign_8,
            scale_4,
            add_4,
            add_5,
            assign_9,
            transpose_4,
            transpose_5,
            full_6,
            scale_5,
            depthwise_conv2d_16,
            batch_norm__244,
            batch_norm__245,
            batch_norm__246,
            batch_norm__247,
            batch_norm__248,
            batch_norm__249,
            hardswish_36,
            conv2d_31,
            batch_norm__250,
            batch_norm__251,
            batch_norm__252,
            batch_norm__253,
            batch_norm__254,
            batch_norm__255,
            hardswish_37,
            depthwise_conv2d_17,
            batch_norm__256,
            batch_norm__257,
            batch_norm__258,
            batch_norm__259,
            batch_norm__260,
            batch_norm__261,
            hardswish_38,
            conv2d_32,
            batch_norm__262,
            batch_norm__263,
            batch_norm__264,
            batch_norm__265,
            batch_norm__266,
            batch_norm__267,
            hardswish_39,
            assign_10,
            pool2d_2,
            conv2d_33,
            reshape_6,
            sigmoid_6,
            multiply_2,
            conv2d_34,
            batch_norm__268,
            batch_norm__269,
            batch_norm__270,
            batch_norm__271,
            batch_norm__272,
            batch_norm__273,
            hardswish_40,
            conv2d_35,
            reshape_7,
            conv2d_36,
            reshape_8,
            add_6,
            conv2d_37,
            batch_norm__274,
            batch_norm__275,
            batch_norm__276,
            batch_norm__277,
            batch_norm__278,
            batch_norm__279,
            hardswish_41,
            conv2d_38,
            batch_norm__280,
            batch_norm__281,
            batch_norm__282,
            batch_norm__283,
            batch_norm__284,
            sigmoid_7,
            sigmoid_8,
            assign_11,
            sqrt_2,
            transpose_6,
            stack_2,
            softmax_2,
            matmul_2,
            assign_12,
            scale_6,
            assign_13,
            split_2,
            assign_14,
            scale_7,
            add_7,
            add_8,
            assign_15,
            transpose_7,
            transpose_8,
            full_7,
            scale_8,
            depthwise_conv2d_18,
            batch_norm__285,
            batch_norm__286,
            batch_norm__287,
            batch_norm__288,
            batch_norm__289,
            batch_norm__290,
            hardswish_42,
            conv2d_39,
            batch_norm__291,
            batch_norm__292,
            batch_norm__293,
            batch_norm__294,
            batch_norm__295,
            batch_norm__296,
            hardswish_43,
            depthwise_conv2d_19,
            batch_norm__297,
            batch_norm__298,
            batch_norm__299,
            batch_norm__300,
            batch_norm__301,
            batch_norm__302,
            hardswish_44,
            conv2d_40,
            batch_norm__303,
            batch_norm__304,
            batch_norm__305,
            batch_norm__306,
            batch_norm__307,
            batch_norm__308,
            hardswish_45,
            assign_16,
            pool2d_3,
            conv2d_41,
            reshape_9,
            sigmoid_9,
            multiply_3,
            conv2d_42,
            batch_norm__309,
            batch_norm__310,
            batch_norm__311,
            batch_norm__312,
            batch_norm__313,
            batch_norm__314,
            hardswish_46,
            conv2d_43,
            reshape_10,
            conv2d_44,
            reshape_11,
            add_9,
            conv2d_45,
            batch_norm__315,
            batch_norm__316,
            batch_norm__317,
            batch_norm__318,
            batch_norm__319,
            batch_norm__320,
            hardswish_47,
            conv2d_46,
            batch_norm__321,
            batch_norm__322,
            batch_norm__323,
            batch_norm__324,
            batch_norm__325,
            sigmoid_10,
            sigmoid_11,
            assign_17,
            sqrt_3,
            transpose_9,
            stack_3,
            softmax_3,
            matmul_3,
            assign_18,
            scale_9,
            assign_19,
            split_3,
            assign_20,
            scale_10,
            add_10,
            add_11,
            assign_21,
            transpose_10,
            transpose_11,
            full_8,
            scale_11,
            assign_22,
            assign_23,
            assign_24,
            concat_4,
            concat_5,
            concat_6,
            hardswish_48,
            hardswish_49,
            hardswish_50,
            add_12,
        )
