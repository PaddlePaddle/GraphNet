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
        parameter_301,
        parameter_302,
        parameter_303,
        parameter_304,
        parameter_305,
        parameter_306,
        parameter_307,
        parameter_308,
        parameter_309,
        parameter_310,
        parameter_311,
        parameter_312,
        parameter_313,
        parameter_314,
        parameter_315,
        parameter_316,
        parameter_317,
        parameter_318,
        parameter_319,
        parameter_320,
        parameter_321,
        parameter_322,
        parameter_323,
        parameter_324,
        data_0,
        data_1,
        data_2,
    ):
        # pd_op.transpose: (1x1024x2704xf32) <- (1x2704x1024xf32)
        transpose_0 = paddle._C_ops.transpose(data_0, [0, 2, 1])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [1, 1024, 52, 52]

        # pd_op.reshape: (1x1024x52x52xf32) <- (1x1024x2704xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(transpose_0, full_int_array_3)

        # pd_op.conv2d: (1x384x52x52xf32) <- (1x1024x52x52xf32, 384x1024x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            reshape_0, parameter_324, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32)
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
                parameter_323,
                parameter_322,
                parameter_321,
                parameter_320,
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

        # pd_op.swish: (1x384x52x52xf32) <- (1x384x52x52xf32)
        swish_0 = paddle._C_ops.swish(batch_norm__0)

        # pd_op.conv2d: (1x384x52x52xf32) <- (1x1024x52x52xf32, 384x1024x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            reshape_0, parameter_319, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32)
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
                parameter_318,
                parameter_317,
                parameter_316,
                parameter_315,
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

        # pd_op.swish: (1x384x52x52xf32) <- (1x384x52x52xf32)
        swish_1 = paddle._C_ops.swish(batch_norm__6)

        # pd_op.conv2d: (1x384x52x52xf32) <- (1x384x52x52xf32, 384x384x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            swish_1, parameter_314, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32)
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
                parameter_313,
                parameter_312,
                parameter_311,
                parameter_310,
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

        # pd_op.swish: (1x384x52x52xf32) <- (1x384x52x52xf32)
        swish_2 = paddle._C_ops.swish(batch_norm__12)

        # pd_op.conv2d: (1x384x52x52xf32) <- (1x384x52x52xf32, 384x384x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            swish_2, parameter_309, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__18,
            batch_norm__19,
            batch_norm__20,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_3,
                parameter_308,
                parameter_307,
                parameter_306,
                parameter_305,
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

        # pd_op.conv2d: (1x384x52x52xf32) <- (1x384x52x52xf32, 384x384x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            swish_2, parameter_304, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_4,
                parameter_303,
                parameter_302,
                parameter_301,
                parameter_300,
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

        # pd_op.add: (1x384x52x52xf32) <- (1x384x52x52xf32, 1x384x52x52xf32)
        add_0 = paddle._C_ops.add(batch_norm__18, batch_norm__24)

        # pd_op.swish: (1x384x52x52xf32) <- (1x384x52x52xf32)
        swish_3 = paddle._C_ops.swish(add_0)

        # pd_op.conv2d: (1x384x52x52xf32) <- (1x384x52x52xf32, 384x384x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            swish_3, parameter_299, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_5,
                parameter_298,
                parameter_297,
                parameter_296,
                parameter_295,
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

        # pd_op.swish: (1x384x52x52xf32) <- (1x384x52x52xf32)
        swish_4 = paddle._C_ops.swish(batch_norm__30)

        # pd_op.conv2d: (1x384x52x52xf32) <- (1x384x52x52xf32, 384x384x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            swish_4, parameter_294, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__36,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_6,
                parameter_293,
                parameter_292,
                parameter_291,
                parameter_290,
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

        # pd_op.conv2d: (1x384x52x52xf32) <- (1x384x52x52xf32, 384x384x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            swish_4, parameter_289, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__42,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_7,
                parameter_288,
                parameter_287,
                parameter_286,
                parameter_285,
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

        # pd_op.add: (1x384x52x52xf32) <- (1x384x52x52xf32, 1x384x52x52xf32)
        add_1 = paddle._C_ops.add(batch_norm__36, batch_norm__42)

        # pd_op.swish: (1x384x52x52xf32) <- (1x384x52x52xf32)
        swish_5 = paddle._C_ops.swish(add_1)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [5, 5]

        # pd_op.pool2d: (1x384x52x52xf32) <- (1x384x52x52xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            swish_5,
            full_int_array_0,
            [1, 1],
            [2, 2],
            False,
            True,
            "NCHW",
            "max",
            False,
            False,
            "EXPLICIT",
        )

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [9, 9]

        # pd_op.pool2d: (1x384x52x52xf32) <- (1x384x52x52xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(
            swish_5,
            full_int_array_1,
            [1, 1],
            [4, 4],
            False,
            True,
            "NCHW",
            "max",
            False,
            False,
            "EXPLICIT",
        )

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [13, 13]

        # pd_op.pool2d: (1x384x52x52xf32) <- (1x384x52x52xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(
            swish_5,
            full_int_array_2,
            [1, 1],
            [6, 6],
            False,
            True,
            "NCHW",
            "max",
            False,
            False,
            "EXPLICIT",
        )

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_8 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_7 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_6 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_5 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_4 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_3 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_2 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_1 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_0 = full_0

        # builtin.combine: ([1x384x52x52xf32, 1x384x52x52xf32, 1x384x52x52xf32, 1x384x52x52xf32]) <- (1x384x52x52xf32, 1x384x52x52xf32, 1x384x52x52xf32, 1x384x52x52xf32)
        combine_0 = [swish_5, pool2d_0, pool2d_1, pool2d_2]

        # pd_op.concat: (1x1536x52x52xf32) <- ([1x384x52x52xf32, 1x384x52x52xf32, 1x384x52x52xf32, 1x384x52x52xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        # pd_op.conv2d: (1x384x52x52xf32) <- (1x1536x52x52xf32, 384x1536x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            concat_0, parameter_284, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__48,
            batch_norm__49,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_8,
                parameter_283,
                parameter_282,
                parameter_281,
                parameter_280,
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

        # pd_op.swish: (1x384x52x52xf32) <- (1x384x52x52xf32)
        swish_6 = paddle._C_ops.swish(batch_norm__48)

        # pd_op.conv2d: (1x384x52x52xf32) <- (1x384x52x52xf32, 384x384x3x3xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            swish_6, parameter_279, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_9,
                parameter_278,
                parameter_277,
                parameter_276,
                parameter_275,
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

        # pd_op.swish: (1x384x52x52xf32) <- (1x384x52x52xf32)
        swish_7 = paddle._C_ops.swish(batch_norm__54)

        # pd_op.conv2d: (1x384x52x52xf32) <- (1x384x52x52xf32, 384x384x3x3xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            swish_7, parameter_274, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_10,
                parameter_273,
                parameter_272,
                parameter_271,
                parameter_270,
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

        # pd_op.conv2d: (1x384x52x52xf32) <- (1x384x52x52xf32, 384x384x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            swish_7, parameter_269, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__66,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            batch_norm__70,
            batch_norm__71,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_11,
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

        # pd_op.add: (1x384x52x52xf32) <- (1x384x52x52xf32, 1x384x52x52xf32)
        add_2 = paddle._C_ops.add(batch_norm__60, batch_norm__66)

        # pd_op.swish: (1x384x52x52xf32) <- (1x384x52x52xf32)
        swish_8 = paddle._C_ops.swish(add_2)

        # builtin.combine: ([1x384x52x52xf32, 1x384x52x52xf32]) <- (1x384x52x52xf32, 1x384x52x52xf32)
        combine_1 = [swish_0, swish_8]

        # pd_op.concat: (1x768x52x52xf32) <- ([1x384x52x52xf32, 1x384x52x52xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_0)

        # pd_op.conv2d: (1x768x52x52xf32) <- (1x768x52x52xf32, 768x768x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            concat_1, parameter_264, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x768x52x52xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (1x768x52x52xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        (
            batch_norm__72,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_12,
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

        # pd_op.swish: (1x768x52x52xf32) <- (1x768x52x52xf32)
        swish_9 = paddle._C_ops.swish(batch_norm__72)

        # pd_op.conv2d: (1x384x52x52xf32) <- (1x768x52x52xf32, 384x768x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            swish_9, parameter_259, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__78,
            batch_norm__79,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
            batch_norm__83,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_13,
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

        # pd_op.swish: (1x384x52x52xf32) <- (1x384x52x52xf32)
        swish_10 = paddle._C_ops.swish(batch_norm__78)

        # pd_op.nearest_interp: (1x384x104x104xf32) <- (1x384x52x52xf32, None, None, None)
        nearest_interp_0 = paddle._C_ops.nearest_interp(
            swish_10,
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

        # builtin.combine: ([1x384x104x104xf32, 1x512x104x104xf32]) <- (1x384x104x104xf32, 1x512x104x104xf32)
        combine_2 = [nearest_interp_0, data_2]

        # pd_op.concat: (1x896x104x104xf32) <- ([1x384x104x104xf32, 1x512x104x104xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_0)

        # pd_op.conv2d: (1x192x104x104xf32) <- (1x896x104x104xf32, 192x896x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            concat_2, parameter_254, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__84,
            batch_norm__85,
            batch_norm__86,
            batch_norm__87,
            batch_norm__88,
            batch_norm__89,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_14,
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

        # pd_op.swish: (1x192x104x104xf32) <- (1x192x104x104xf32)
        swish_11 = paddle._C_ops.swish(batch_norm__84)

        # pd_op.conv2d: (1x192x104x104xf32) <- (1x896x104x104xf32, 192x896x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            concat_2, parameter_249, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__90,
            batch_norm__91,
            batch_norm__92,
            batch_norm__93,
            batch_norm__94,
            batch_norm__95,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_15,
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

        # pd_op.swish: (1x192x104x104xf32) <- (1x192x104x104xf32)
        swish_12 = paddle._C_ops.swish(batch_norm__90)

        # pd_op.conv2d: (1x192x104x104xf32) <- (1x192x104x104xf32, 192x192x3x3xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            swish_12, parameter_244, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__96,
            batch_norm__97,
            batch_norm__98,
            batch_norm__99,
            batch_norm__100,
            batch_norm__101,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_16,
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

        # pd_op.swish: (1x192x104x104xf32) <- (1x192x104x104xf32)
        swish_13 = paddle._C_ops.swish(batch_norm__96)

        # pd_op.conv2d: (1x192x104x104xf32) <- (1x192x104x104xf32, 192x192x3x3xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            swish_13, parameter_239, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__102,
            batch_norm__103,
            batch_norm__104,
            batch_norm__105,
            batch_norm__106,
            batch_norm__107,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_17,
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

        # pd_op.conv2d: (1x192x104x104xf32) <- (1x192x104x104xf32, 192x192x1x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            swish_13, parameter_234, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__108,
            batch_norm__109,
            batch_norm__110,
            batch_norm__111,
            batch_norm__112,
            batch_norm__113,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_18,
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

        # pd_op.add: (1x192x104x104xf32) <- (1x192x104x104xf32, 1x192x104x104xf32)
        add_3 = paddle._C_ops.add(batch_norm__102, batch_norm__108)

        # pd_op.swish: (1x192x104x104xf32) <- (1x192x104x104xf32)
        swish_14 = paddle._C_ops.swish(add_3)

        # pd_op.conv2d: (1x192x104x104xf32) <- (1x192x104x104xf32, 192x192x3x3xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            swish_14, parameter_229, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__114,
            batch_norm__115,
            batch_norm__116,
            batch_norm__117,
            batch_norm__118,
            batch_norm__119,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_19,
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

        # pd_op.swish: (1x192x104x104xf32) <- (1x192x104x104xf32)
        swish_15 = paddle._C_ops.swish(batch_norm__114)

        # pd_op.conv2d: (1x192x104x104xf32) <- (1x192x104x104xf32, 192x192x3x3xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            swish_15, parameter_224, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__120,
            batch_norm__121,
            batch_norm__122,
            batch_norm__123,
            batch_norm__124,
            batch_norm__125,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_20,
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

        # pd_op.conv2d: (1x192x104x104xf32) <- (1x192x104x104xf32, 192x192x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            swish_15, parameter_219, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__126,
            batch_norm__127,
            batch_norm__128,
            batch_norm__129,
            batch_norm__130,
            batch_norm__131,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_21,
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

        # pd_op.add: (1x192x104x104xf32) <- (1x192x104x104xf32, 1x192x104x104xf32)
        add_4 = paddle._C_ops.add(batch_norm__120, batch_norm__126)

        # pd_op.swish: (1x192x104x104xf32) <- (1x192x104x104xf32)
        swish_16 = paddle._C_ops.swish(add_4)

        # pd_op.conv2d: (1x192x104x104xf32) <- (1x192x104x104xf32, 192x192x3x3xf32)
        conv2d_22 = paddle._C_ops.conv2d(
            swish_16, parameter_214, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32)
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

        # pd_op.swish: (1x192x104x104xf32) <- (1x192x104x104xf32)
        swish_17 = paddle._C_ops.swish(batch_norm__132)

        # pd_op.conv2d: (1x192x104x104xf32) <- (1x192x104x104xf32, 192x192x3x3xf32)
        conv2d_23 = paddle._C_ops.conv2d(
            swish_17, parameter_209, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32)
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
                parameter_208,
                parameter_207,
                parameter_206,
                parameter_205,
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

        # pd_op.conv2d: (1x192x104x104xf32) <- (1x192x104x104xf32, 192x192x1x1xf32)
        conv2d_24 = paddle._C_ops.conv2d(
            swish_17, parameter_204, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__144,
            batch_norm__145,
            batch_norm__146,
            batch_norm__147,
            batch_norm__148,
            batch_norm__149,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_24,
                parameter_203,
                parameter_202,
                parameter_201,
                parameter_200,
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

        # pd_op.add: (1x192x104x104xf32) <- (1x192x104x104xf32, 1x192x104x104xf32)
        add_5 = paddle._C_ops.add(batch_norm__138, batch_norm__144)

        # pd_op.swish: (1x192x104x104xf32) <- (1x192x104x104xf32)
        swish_18 = paddle._C_ops.swish(add_5)

        # builtin.combine: ([1x192x104x104xf32, 1x192x104x104xf32]) <- (1x192x104x104xf32, 1x192x104x104xf32)
        combine_3 = [swish_11, swish_18]

        # pd_op.concat: (1x384x104x104xf32) <- ([1x192x104x104xf32, 1x192x104x104xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_3, full_0)

        # pd_op.conv2d: (1x384x104x104xf32) <- (1x384x104x104xf32, 384x384x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(
            concat_3, parameter_199, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x384x104x104xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (1x384x104x104xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__150,
            batch_norm__151,
            batch_norm__152,
            batch_norm__153,
            batch_norm__154,
            batch_norm__155,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_25,
                parameter_198,
                parameter_197,
                parameter_196,
                parameter_195,
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

        # pd_op.swish: (1x384x104x104xf32) <- (1x384x104x104xf32)
        swish_19 = paddle._C_ops.swish(batch_norm__150)

        # pd_op.conv2d: (1x192x104x104xf32) <- (1x384x104x104xf32, 192x384x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(
            swish_19, parameter_194, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__156,
            batch_norm__157,
            batch_norm__158,
            batch_norm__159,
            batch_norm__160,
            batch_norm__161,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_26,
                parameter_193,
                parameter_192,
                parameter_191,
                parameter_190,
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

        # pd_op.swish: (1x192x104x104xf32) <- (1x192x104x104xf32)
        swish_20 = paddle._C_ops.swish(batch_norm__156)

        # pd_op.nearest_interp: (1x192x208x208xf32) <- (1x192x104x104xf32, None, None, None)
        nearest_interp_1 = paddle._C_ops.nearest_interp(
            swish_20,
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

        # builtin.combine: ([1x192x208x208xf32, 1x256x208x208xf32]) <- (1x192x208x208xf32, 1x256x208x208xf32)
        combine_4 = [nearest_interp_1, data_1]

        # pd_op.concat: (1x448x208x208xf32) <- ([1x192x208x208xf32, 1x256x208x208xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_4, full_0)

        # pd_op.conv2d: (1x96x208x208xf32) <- (1x448x208x208xf32, 96x448x1x1xf32)
        conv2d_27 = paddle._C_ops.conv2d(
            concat_4, parameter_189, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x96x208x208xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (1x96x208x208xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__162,
            batch_norm__163,
            batch_norm__164,
            batch_norm__165,
            batch_norm__166,
            batch_norm__167,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_27,
                parameter_188,
                parameter_187,
                parameter_186,
                parameter_185,
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

        # pd_op.swish: (1x96x208x208xf32) <- (1x96x208x208xf32)
        swish_21 = paddle._C_ops.swish(batch_norm__162)

        # pd_op.conv2d: (1x96x208x208xf32) <- (1x448x208x208xf32, 96x448x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(
            concat_4, parameter_184, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x96x208x208xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (1x96x208x208xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__168,
            batch_norm__169,
            batch_norm__170,
            batch_norm__171,
            batch_norm__172,
            batch_norm__173,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_28,
                parameter_183,
                parameter_182,
                parameter_181,
                parameter_180,
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

        # pd_op.swish: (1x96x208x208xf32) <- (1x96x208x208xf32)
        swish_22 = paddle._C_ops.swish(batch_norm__168)

        # pd_op.conv2d: (1x96x208x208xf32) <- (1x96x208x208xf32, 96x96x3x3xf32)
        conv2d_29 = paddle._C_ops.conv2d(
            swish_22, parameter_179, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x96x208x208xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (1x96x208x208xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__174,
            batch_norm__175,
            batch_norm__176,
            batch_norm__177,
            batch_norm__178,
            batch_norm__179,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_29,
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

        # pd_op.swish: (1x96x208x208xf32) <- (1x96x208x208xf32)
        swish_23 = paddle._C_ops.swish(batch_norm__174)

        # pd_op.conv2d: (1x96x208x208xf32) <- (1x96x208x208xf32, 96x96x3x3xf32)
        conv2d_30 = paddle._C_ops.conv2d(
            swish_23, parameter_174, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x96x208x208xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (1x96x208x208xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__180,
            batch_norm__181,
            batch_norm__182,
            batch_norm__183,
            batch_norm__184,
            batch_norm__185,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_30,
                parameter_173,
                parameter_172,
                parameter_171,
                parameter_170,
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

        # pd_op.conv2d: (1x96x208x208xf32) <- (1x96x208x208xf32, 96x96x1x1xf32)
        conv2d_31 = paddle._C_ops.conv2d(
            swish_23, parameter_169, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x96x208x208xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (1x96x208x208xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__186,
            batch_norm__187,
            batch_norm__188,
            batch_norm__189,
            batch_norm__190,
            batch_norm__191,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_31,
                parameter_168,
                parameter_167,
                parameter_166,
                parameter_165,
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

        # pd_op.add: (1x96x208x208xf32) <- (1x96x208x208xf32, 1x96x208x208xf32)
        add_6 = paddle._C_ops.add(batch_norm__180, batch_norm__186)

        # pd_op.swish: (1x96x208x208xf32) <- (1x96x208x208xf32)
        swish_24 = paddle._C_ops.swish(add_6)

        # pd_op.conv2d: (1x96x208x208xf32) <- (1x96x208x208xf32, 96x96x3x3xf32)
        conv2d_32 = paddle._C_ops.conv2d(
            swish_24, parameter_164, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x96x208x208xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (1x96x208x208xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__192,
            batch_norm__193,
            batch_norm__194,
            batch_norm__195,
            batch_norm__196,
            batch_norm__197,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_32,
                parameter_163,
                parameter_162,
                parameter_161,
                parameter_160,
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

        # pd_op.swish: (1x96x208x208xf32) <- (1x96x208x208xf32)
        swish_25 = paddle._C_ops.swish(batch_norm__192)

        # pd_op.conv2d: (1x96x208x208xf32) <- (1x96x208x208xf32, 96x96x3x3xf32)
        conv2d_33 = paddle._C_ops.conv2d(
            swish_25, parameter_159, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x96x208x208xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (1x96x208x208xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__198,
            batch_norm__199,
            batch_norm__200,
            batch_norm__201,
            batch_norm__202,
            batch_norm__203,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_33,
                parameter_158,
                parameter_157,
                parameter_156,
                parameter_155,
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

        # pd_op.conv2d: (1x96x208x208xf32) <- (1x96x208x208xf32, 96x96x1x1xf32)
        conv2d_34 = paddle._C_ops.conv2d(
            swish_25, parameter_154, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x96x208x208xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (1x96x208x208xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
                parameter_153,
                parameter_152,
                parameter_151,
                parameter_150,
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

        # pd_op.add: (1x96x208x208xf32) <- (1x96x208x208xf32, 1x96x208x208xf32)
        add_7 = paddle._C_ops.add(batch_norm__198, batch_norm__204)

        # pd_op.swish: (1x96x208x208xf32) <- (1x96x208x208xf32)
        swish_26 = paddle._C_ops.swish(add_7)

        # pd_op.conv2d: (1x96x208x208xf32) <- (1x96x208x208xf32, 96x96x3x3xf32)
        conv2d_35 = paddle._C_ops.conv2d(
            swish_26, parameter_149, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x96x208x208xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (1x96x208x208xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
                parameter_148,
                parameter_147,
                parameter_146,
                parameter_145,
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

        # pd_op.swish: (1x96x208x208xf32) <- (1x96x208x208xf32)
        swish_27 = paddle._C_ops.swish(batch_norm__210)

        # pd_op.conv2d: (1x96x208x208xf32) <- (1x96x208x208xf32, 96x96x3x3xf32)
        conv2d_36 = paddle._C_ops.conv2d(
            swish_27, parameter_144, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x96x208x208xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (1x96x208x208xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__216,
            batch_norm__217,
            batch_norm__218,
            batch_norm__219,
            batch_norm__220,
            batch_norm__221,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_36,
                parameter_143,
                parameter_142,
                parameter_141,
                parameter_140,
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

        # pd_op.conv2d: (1x96x208x208xf32) <- (1x96x208x208xf32, 96x96x1x1xf32)
        conv2d_37 = paddle._C_ops.conv2d(
            swish_27, parameter_139, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x96x208x208xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (1x96x208x208xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__222,
            batch_norm__223,
            batch_norm__224,
            batch_norm__225,
            batch_norm__226,
            batch_norm__227,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_37,
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

        # pd_op.add: (1x96x208x208xf32) <- (1x96x208x208xf32, 1x96x208x208xf32)
        add_8 = paddle._C_ops.add(batch_norm__216, batch_norm__222)

        # pd_op.swish: (1x96x208x208xf32) <- (1x96x208x208xf32)
        swish_28 = paddle._C_ops.swish(add_8)

        # builtin.combine: ([1x96x208x208xf32, 1x96x208x208xf32]) <- (1x96x208x208xf32, 1x96x208x208xf32)
        combine_5 = [swish_21, swish_28]

        # pd_op.concat: (1x192x208x208xf32) <- ([1x96x208x208xf32, 1x96x208x208xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_5, full_0)

        # pd_op.conv2d: (1x192x208x208xf32) <- (1x192x208x208xf32, 192x192x1x1xf32)
        conv2d_38 = paddle._C_ops.conv2d(
            concat_5, parameter_134, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x192x208x208xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (1x192x208x208xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__228,
            batch_norm__229,
            batch_norm__230,
            batch_norm__231,
            batch_norm__232,
            batch_norm__233,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_38,
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

        # pd_op.swish: (1x192x208x208xf32) <- (1x192x208x208xf32)
        swish_49 = paddle._C_ops.swish(batch_norm__228)

        # pd_op.conv2d: (1x192x104x104xf32) <- (1x192x208x208xf32, 192x192x3x3xf32)
        conv2d_39 = paddle._C_ops.conv2d(
            swish_49, parameter_129, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__234,
            batch_norm__235,
            batch_norm__236,
            batch_norm__237,
            batch_norm__238,
            batch_norm__239,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_39,
                parameter_128,
                parameter_127,
                parameter_126,
                parameter_125,
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

        # pd_op.swish: (1x192x104x104xf32) <- (1x192x104x104xf32)
        swish_29 = paddle._C_ops.swish(batch_norm__234)

        # builtin.combine: ([1x192x104x104xf32, 1x384x104x104xf32]) <- (1x192x104x104xf32, 1x384x104x104xf32)
        combine_6 = [swish_29, swish_19]

        # pd_op.concat: (1x576x104x104xf32) <- ([1x192x104x104xf32, 1x384x104x104xf32], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_6, full_0)

        # pd_op.conv2d: (1x192x104x104xf32) <- (1x576x104x104xf32, 192x576x1x1xf32)
        conv2d_40 = paddle._C_ops.conv2d(
            concat_6, parameter_124, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__240,
            batch_norm__241,
            batch_norm__242,
            batch_norm__243,
            batch_norm__244,
            batch_norm__245,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_40,
                parameter_123,
                parameter_122,
                parameter_121,
                parameter_120,
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

        # pd_op.swish: (1x192x104x104xf32) <- (1x192x104x104xf32)
        swish_30 = paddle._C_ops.swish(batch_norm__240)

        # pd_op.conv2d: (1x192x104x104xf32) <- (1x576x104x104xf32, 192x576x1x1xf32)
        conv2d_41 = paddle._C_ops.conv2d(
            concat_6, parameter_119, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__246,
            batch_norm__247,
            batch_norm__248,
            batch_norm__249,
            batch_norm__250,
            batch_norm__251,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_41,
                parameter_118,
                parameter_117,
                parameter_116,
                parameter_115,
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

        # pd_op.swish: (1x192x104x104xf32) <- (1x192x104x104xf32)
        swish_31 = paddle._C_ops.swish(batch_norm__246)

        # pd_op.conv2d: (1x192x104x104xf32) <- (1x192x104x104xf32, 192x192x3x3xf32)
        conv2d_42 = paddle._C_ops.conv2d(
            swish_31, parameter_114, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__252,
            batch_norm__253,
            batch_norm__254,
            batch_norm__255,
            batch_norm__256,
            batch_norm__257,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_42,
                parameter_113,
                parameter_112,
                parameter_111,
                parameter_110,
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

        # pd_op.swish: (1x192x104x104xf32) <- (1x192x104x104xf32)
        swish_32 = paddle._C_ops.swish(batch_norm__252)

        # pd_op.conv2d: (1x192x104x104xf32) <- (1x192x104x104xf32, 192x192x3x3xf32)
        conv2d_43 = paddle._C_ops.conv2d(
            swish_32, parameter_109, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__258,
            batch_norm__259,
            batch_norm__260,
            batch_norm__261,
            batch_norm__262,
            batch_norm__263,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_43,
                parameter_108,
                parameter_107,
                parameter_106,
                parameter_105,
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

        # pd_op.conv2d: (1x192x104x104xf32) <- (1x192x104x104xf32, 192x192x1x1xf32)
        conv2d_44 = paddle._C_ops.conv2d(
            swish_32, parameter_104, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__264,
            batch_norm__265,
            batch_norm__266,
            batch_norm__267,
            batch_norm__268,
            batch_norm__269,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_44,
                parameter_103,
                parameter_102,
                parameter_101,
                parameter_100,
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

        # pd_op.add: (1x192x104x104xf32) <- (1x192x104x104xf32, 1x192x104x104xf32)
        add_9 = paddle._C_ops.add(batch_norm__258, batch_norm__264)

        # pd_op.swish: (1x192x104x104xf32) <- (1x192x104x104xf32)
        swish_33 = paddle._C_ops.swish(add_9)

        # pd_op.conv2d: (1x192x104x104xf32) <- (1x192x104x104xf32, 192x192x3x3xf32)
        conv2d_45 = paddle._C_ops.conv2d(
            swish_33, parameter_99, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__270,
            batch_norm__271,
            batch_norm__272,
            batch_norm__273,
            batch_norm__274,
            batch_norm__275,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_45,
                parameter_98,
                parameter_97,
                parameter_96,
                parameter_95,
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

        # pd_op.swish: (1x192x104x104xf32) <- (1x192x104x104xf32)
        swish_34 = paddle._C_ops.swish(batch_norm__270)

        # pd_op.conv2d: (1x192x104x104xf32) <- (1x192x104x104xf32, 192x192x3x3xf32)
        conv2d_46 = paddle._C_ops.conv2d(
            swish_34, parameter_94, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__276,
            batch_norm__277,
            batch_norm__278,
            batch_norm__279,
            batch_norm__280,
            batch_norm__281,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_46,
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

        # pd_op.conv2d: (1x192x104x104xf32) <- (1x192x104x104xf32, 192x192x1x1xf32)
        conv2d_47 = paddle._C_ops.conv2d(
            swish_34, parameter_89, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__282,
            batch_norm__283,
            batch_norm__284,
            batch_norm__285,
            batch_norm__286,
            batch_norm__287,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_47,
                parameter_88,
                parameter_87,
                parameter_86,
                parameter_85,
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

        # pd_op.add: (1x192x104x104xf32) <- (1x192x104x104xf32, 1x192x104x104xf32)
        add_10 = paddle._C_ops.add(batch_norm__276, batch_norm__282)

        # pd_op.swish: (1x192x104x104xf32) <- (1x192x104x104xf32)
        swish_35 = paddle._C_ops.swish(add_10)

        # pd_op.conv2d: (1x192x104x104xf32) <- (1x192x104x104xf32, 192x192x3x3xf32)
        conv2d_48 = paddle._C_ops.conv2d(
            swish_35, parameter_84, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__288,
            batch_norm__289,
            batch_norm__290,
            batch_norm__291,
            batch_norm__292,
            batch_norm__293,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_48,
                parameter_83,
                parameter_82,
                parameter_81,
                parameter_80,
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

        # pd_op.swish: (1x192x104x104xf32) <- (1x192x104x104xf32)
        swish_36 = paddle._C_ops.swish(batch_norm__288)

        # pd_op.conv2d: (1x192x104x104xf32) <- (1x192x104x104xf32, 192x192x3x3xf32)
        conv2d_49 = paddle._C_ops.conv2d(
            swish_36, parameter_79, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__294,
            batch_norm__295,
            batch_norm__296,
            batch_norm__297,
            batch_norm__298,
            batch_norm__299,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_49,
                parameter_78,
                parameter_77,
                parameter_76,
                parameter_75,
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

        # pd_op.conv2d: (1x192x104x104xf32) <- (1x192x104x104xf32, 192x192x1x1xf32)
        conv2d_50 = paddle._C_ops.conv2d(
            swish_36, parameter_74, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (1x192x104x104xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__300,
            batch_norm__301,
            batch_norm__302,
            batch_norm__303,
            batch_norm__304,
            batch_norm__305,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_50,
                parameter_73,
                parameter_72,
                parameter_71,
                parameter_70,
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

        # pd_op.add: (1x192x104x104xf32) <- (1x192x104x104xf32, 1x192x104x104xf32)
        add_11 = paddle._C_ops.add(batch_norm__294, batch_norm__300)

        # pd_op.swish: (1x192x104x104xf32) <- (1x192x104x104xf32)
        swish_37 = paddle._C_ops.swish(add_11)

        # builtin.combine: ([1x192x104x104xf32, 1x192x104x104xf32]) <- (1x192x104x104xf32, 1x192x104x104xf32)
        combine_7 = [swish_30, swish_37]

        # pd_op.concat: (1x384x104x104xf32) <- ([1x192x104x104xf32, 1x192x104x104xf32], 1xi32)
        concat_7 = paddle._C_ops.concat(combine_7, full_0)

        # pd_op.conv2d: (1x384x104x104xf32) <- (1x384x104x104xf32, 384x384x1x1xf32)
        conv2d_51 = paddle._C_ops.conv2d(
            concat_7, parameter_69, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x384x104x104xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (1x384x104x104xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__306,
            batch_norm__307,
            batch_norm__308,
            batch_norm__309,
            batch_norm__310,
            batch_norm__311,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_51,
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

        # pd_op.swish: (1x384x104x104xf32) <- (1x384x104x104xf32)
        swish_48 = paddle._C_ops.swish(batch_norm__306)

        # pd_op.conv2d: (1x384x52x52xf32) <- (1x384x104x104xf32, 384x384x3x3xf32)
        conv2d_52 = paddle._C_ops.conv2d(
            swish_48, parameter_64, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__312,
            batch_norm__313,
            batch_norm__314,
            batch_norm__315,
            batch_norm__316,
            batch_norm__317,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_52,
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

        # pd_op.swish: (1x384x52x52xf32) <- (1x384x52x52xf32)
        swish_38 = paddle._C_ops.swish(batch_norm__312)

        # builtin.combine: ([1x384x52x52xf32, 1x768x52x52xf32]) <- (1x384x52x52xf32, 1x768x52x52xf32)
        combine_8 = [swish_38, swish_9]

        # pd_op.concat: (1x1152x52x52xf32) <- ([1x384x52x52xf32, 1x768x52x52xf32], 1xi32)
        concat_8 = paddle._C_ops.concat(combine_8, full_0)

        # pd_op.conv2d: (1x384x52x52xf32) <- (1x1152x52x52xf32, 384x1152x1x1xf32)
        conv2d_53 = paddle._C_ops.conv2d(
            concat_8, parameter_59, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__318,
            batch_norm__319,
            batch_norm__320,
            batch_norm__321,
            batch_norm__322,
            batch_norm__323,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_53,
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

        # pd_op.swish: (1x384x52x52xf32) <- (1x384x52x52xf32)
        swish_39 = paddle._C_ops.swish(batch_norm__318)

        # pd_op.conv2d: (1x384x52x52xf32) <- (1x1152x52x52xf32, 384x1152x1x1xf32)
        conv2d_54 = paddle._C_ops.conv2d(
            concat_8, parameter_54, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__324,
            batch_norm__325,
            batch_norm__326,
            batch_norm__327,
            batch_norm__328,
            batch_norm__329,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_54,
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

        # pd_op.swish: (1x384x52x52xf32) <- (1x384x52x52xf32)
        swish_40 = paddle._C_ops.swish(batch_norm__324)

        # pd_op.conv2d: (1x384x52x52xf32) <- (1x384x52x52xf32, 384x384x3x3xf32)
        conv2d_55 = paddle._C_ops.conv2d(
            swish_40, parameter_49, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__330,
            batch_norm__331,
            batch_norm__332,
            batch_norm__333,
            batch_norm__334,
            batch_norm__335,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_55,
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

        # pd_op.swish: (1x384x52x52xf32) <- (1x384x52x52xf32)
        swish_41 = paddle._C_ops.swish(batch_norm__330)

        # pd_op.conv2d: (1x384x52x52xf32) <- (1x384x52x52xf32, 384x384x3x3xf32)
        conv2d_56 = paddle._C_ops.conv2d(
            swish_41, parameter_44, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__336,
            batch_norm__337,
            batch_norm__338,
            batch_norm__339,
            batch_norm__340,
            batch_norm__341,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_56,
                parameter_43,
                parameter_42,
                parameter_41,
                parameter_40,
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

        # pd_op.conv2d: (1x384x52x52xf32) <- (1x384x52x52xf32, 384x384x1x1xf32)
        conv2d_57 = paddle._C_ops.conv2d(
            swish_41, parameter_39, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__342,
            batch_norm__343,
            batch_norm__344,
            batch_norm__345,
            batch_norm__346,
            batch_norm__347,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_57,
                parameter_38,
                parameter_37,
                parameter_36,
                parameter_35,
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

        # pd_op.add: (1x384x52x52xf32) <- (1x384x52x52xf32, 1x384x52x52xf32)
        add_12 = paddle._C_ops.add(batch_norm__336, batch_norm__342)

        # pd_op.swish: (1x384x52x52xf32) <- (1x384x52x52xf32)
        swish_42 = paddle._C_ops.swish(add_12)

        # pd_op.conv2d: (1x384x52x52xf32) <- (1x384x52x52xf32, 384x384x3x3xf32)
        conv2d_58 = paddle._C_ops.conv2d(
            swish_42, parameter_34, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__348,
            batch_norm__349,
            batch_norm__350,
            batch_norm__351,
            batch_norm__352,
            batch_norm__353,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_58,
                parameter_33,
                parameter_32,
                parameter_31,
                parameter_30,
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

        # pd_op.swish: (1x384x52x52xf32) <- (1x384x52x52xf32)
        swish_43 = paddle._C_ops.swish(batch_norm__348)

        # pd_op.conv2d: (1x384x52x52xf32) <- (1x384x52x52xf32, 384x384x3x3xf32)
        conv2d_59 = paddle._C_ops.conv2d(
            swish_43, parameter_29, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__354,
            batch_norm__355,
            batch_norm__356,
            batch_norm__357,
            batch_norm__358,
            batch_norm__359,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_59,
                parameter_28,
                parameter_27,
                parameter_26,
                parameter_25,
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

        # pd_op.conv2d: (1x384x52x52xf32) <- (1x384x52x52xf32, 384x384x1x1xf32)
        conv2d_60 = paddle._C_ops.conv2d(
            swish_43, parameter_24, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__360,
            batch_norm__361,
            batch_norm__362,
            batch_norm__363,
            batch_norm__364,
            batch_norm__365,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_60,
                parameter_23,
                parameter_22,
                parameter_21,
                parameter_20,
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

        # pd_op.add: (1x384x52x52xf32) <- (1x384x52x52xf32, 1x384x52x52xf32)
        add_13 = paddle._C_ops.add(batch_norm__354, batch_norm__360)

        # pd_op.swish: (1x384x52x52xf32) <- (1x384x52x52xf32)
        swish_44 = paddle._C_ops.swish(add_13)

        # pd_op.conv2d: (1x384x52x52xf32) <- (1x384x52x52xf32, 384x384x3x3xf32)
        conv2d_61 = paddle._C_ops.conv2d(
            swish_44, parameter_19, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__366,
            batch_norm__367,
            batch_norm__368,
            batch_norm__369,
            batch_norm__370,
            batch_norm__371,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_61,
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

        # pd_op.swish: (1x384x52x52xf32) <- (1x384x52x52xf32)
        swish_45 = paddle._C_ops.swish(batch_norm__366)

        # pd_op.conv2d: (1x384x52x52xf32) <- (1x384x52x52xf32, 384x384x3x3xf32)
        conv2d_62 = paddle._C_ops.conv2d(
            swish_45, parameter_14, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__372,
            batch_norm__373,
            batch_norm__374,
            batch_norm__375,
            batch_norm__376,
            batch_norm__377,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_62,
                parameter_13,
                parameter_12,
                parameter_11,
                parameter_10,
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

        # pd_op.conv2d: (1x384x52x52xf32) <- (1x384x52x52xf32, 384x384x1x1xf32)
        conv2d_63 = paddle._C_ops.conv2d(
            swish_45, parameter_9, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (1x384x52x52xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__378,
            batch_norm__379,
            batch_norm__380,
            batch_norm__381,
            batch_norm__382,
            batch_norm__383,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_63,
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

        # pd_op.add: (1x384x52x52xf32) <- (1x384x52x52xf32, 1x384x52x52xf32)
        add_14 = paddle._C_ops.add(batch_norm__372, batch_norm__378)

        # pd_op.swish: (1x384x52x52xf32) <- (1x384x52x52xf32)
        swish_46 = paddle._C_ops.swish(add_14)

        # builtin.combine: ([1x384x52x52xf32, 1x384x52x52xf32]) <- (1x384x52x52xf32, 1x384x52x52xf32)
        combine_9 = [swish_39, swish_46]

        # pd_op.concat: (1x768x52x52xf32) <- ([1x384x52x52xf32, 1x384x52x52xf32], 1xi32)
        concat_9 = paddle._C_ops.concat(combine_9, full_0)

        # pd_op.conv2d: (1x768x52x52xf32) <- (1x768x52x52xf32, 768x768x1x1xf32)
        conv2d_64 = paddle._C_ops.conv2d(
            concat_9, parameter_4, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x768x52x52xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (1x768x52x52xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        (
            batch_norm__384,
            batch_norm__385,
            batch_norm__386,
            batch_norm__387,
            batch_norm__388,
            batch_norm__389,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_64,
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

        # pd_op.swish: (1x768x52x52xf32) <- (1x768x52x52xf32)
        swish_47 = paddle._C_ops.swish(batch_norm__384)
        return (
            transpose_0,
            conv2d_0,
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
            batch_norm__5,
            swish_0,
            conv2d_1,
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            batch_norm__10,
            batch_norm__11,
            swish_1,
            conv2d_2,
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            batch_norm__16,
            batch_norm__17,
            swish_2,
            conv2d_3,
            batch_norm__18,
            batch_norm__19,
            batch_norm__20,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
            conv2d_4,
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
            add_0,
            swish_3,
            conv2d_5,
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
            swish_4,
            conv2d_6,
            batch_norm__36,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
            conv2d_7,
            batch_norm__42,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
            add_1,
            swish_5,
            full_int_array_0,
            pool2d_0,
            full_int_array_1,
            pool2d_1,
            full_int_array_2,
            pool2d_2,
            full_0,
            concat_0,
            conv2d_8,
            batch_norm__48,
            batch_norm__49,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
            swish_6,
            conv2d_9,
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
            swish_7,
            conv2d_10,
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
            conv2d_11,
            batch_norm__66,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            batch_norm__70,
            batch_norm__71,
            add_2,
            swish_8,
            assign_0,
            concat_1,
            conv2d_12,
            batch_norm__72,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
            swish_9,
            conv2d_13,
            batch_norm__78,
            batch_norm__79,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
            batch_norm__83,
            swish_10,
            nearest_interp_0,
            assign_1,
            concat_2,
            conv2d_14,
            batch_norm__84,
            batch_norm__85,
            batch_norm__86,
            batch_norm__87,
            batch_norm__88,
            batch_norm__89,
            swish_11,
            conv2d_15,
            batch_norm__90,
            batch_norm__91,
            batch_norm__92,
            batch_norm__93,
            batch_norm__94,
            batch_norm__95,
            swish_12,
            conv2d_16,
            batch_norm__96,
            batch_norm__97,
            batch_norm__98,
            batch_norm__99,
            batch_norm__100,
            batch_norm__101,
            swish_13,
            conv2d_17,
            batch_norm__102,
            batch_norm__103,
            batch_norm__104,
            batch_norm__105,
            batch_norm__106,
            batch_norm__107,
            conv2d_18,
            batch_norm__108,
            batch_norm__109,
            batch_norm__110,
            batch_norm__111,
            batch_norm__112,
            batch_norm__113,
            add_3,
            swish_14,
            conv2d_19,
            batch_norm__114,
            batch_norm__115,
            batch_norm__116,
            batch_norm__117,
            batch_norm__118,
            batch_norm__119,
            swish_15,
            conv2d_20,
            batch_norm__120,
            batch_norm__121,
            batch_norm__122,
            batch_norm__123,
            batch_norm__124,
            batch_norm__125,
            conv2d_21,
            batch_norm__126,
            batch_norm__127,
            batch_norm__128,
            batch_norm__129,
            batch_norm__130,
            batch_norm__131,
            add_4,
            swish_16,
            conv2d_22,
            batch_norm__132,
            batch_norm__133,
            batch_norm__134,
            batch_norm__135,
            batch_norm__136,
            batch_norm__137,
            swish_17,
            conv2d_23,
            batch_norm__138,
            batch_norm__139,
            batch_norm__140,
            batch_norm__141,
            batch_norm__142,
            batch_norm__143,
            conv2d_24,
            batch_norm__144,
            batch_norm__145,
            batch_norm__146,
            batch_norm__147,
            batch_norm__148,
            batch_norm__149,
            add_5,
            swish_18,
            assign_2,
            concat_3,
            conv2d_25,
            batch_norm__150,
            batch_norm__151,
            batch_norm__152,
            batch_norm__153,
            batch_norm__154,
            batch_norm__155,
            swish_19,
            conv2d_26,
            batch_norm__156,
            batch_norm__157,
            batch_norm__158,
            batch_norm__159,
            batch_norm__160,
            batch_norm__161,
            swish_20,
            nearest_interp_1,
            assign_3,
            concat_4,
            conv2d_27,
            batch_norm__162,
            batch_norm__163,
            batch_norm__164,
            batch_norm__165,
            batch_norm__166,
            batch_norm__167,
            swish_21,
            conv2d_28,
            batch_norm__168,
            batch_norm__169,
            batch_norm__170,
            batch_norm__171,
            batch_norm__172,
            batch_norm__173,
            swish_22,
            conv2d_29,
            batch_norm__174,
            batch_norm__175,
            batch_norm__176,
            batch_norm__177,
            batch_norm__178,
            batch_norm__179,
            swish_23,
            conv2d_30,
            batch_norm__180,
            batch_norm__181,
            batch_norm__182,
            batch_norm__183,
            batch_norm__184,
            batch_norm__185,
            conv2d_31,
            batch_norm__186,
            batch_norm__187,
            batch_norm__188,
            batch_norm__189,
            batch_norm__190,
            batch_norm__191,
            add_6,
            swish_24,
            conv2d_32,
            batch_norm__192,
            batch_norm__193,
            batch_norm__194,
            batch_norm__195,
            batch_norm__196,
            batch_norm__197,
            swish_25,
            conv2d_33,
            batch_norm__198,
            batch_norm__199,
            batch_norm__200,
            batch_norm__201,
            batch_norm__202,
            batch_norm__203,
            conv2d_34,
            batch_norm__204,
            batch_norm__205,
            batch_norm__206,
            batch_norm__207,
            batch_norm__208,
            batch_norm__209,
            add_7,
            swish_26,
            conv2d_35,
            batch_norm__210,
            batch_norm__211,
            batch_norm__212,
            batch_norm__213,
            batch_norm__214,
            batch_norm__215,
            swish_27,
            conv2d_36,
            batch_norm__216,
            batch_norm__217,
            batch_norm__218,
            batch_norm__219,
            batch_norm__220,
            batch_norm__221,
            conv2d_37,
            batch_norm__222,
            batch_norm__223,
            batch_norm__224,
            batch_norm__225,
            batch_norm__226,
            batch_norm__227,
            add_8,
            swish_28,
            assign_4,
            concat_5,
            conv2d_38,
            batch_norm__228,
            batch_norm__229,
            batch_norm__230,
            batch_norm__231,
            batch_norm__232,
            batch_norm__233,
            conv2d_39,
            batch_norm__234,
            batch_norm__235,
            batch_norm__236,
            batch_norm__237,
            batch_norm__238,
            batch_norm__239,
            swish_29,
            assign_5,
            concat_6,
            conv2d_40,
            batch_norm__240,
            batch_norm__241,
            batch_norm__242,
            batch_norm__243,
            batch_norm__244,
            batch_norm__245,
            swish_30,
            conv2d_41,
            batch_norm__246,
            batch_norm__247,
            batch_norm__248,
            batch_norm__249,
            batch_norm__250,
            batch_norm__251,
            swish_31,
            conv2d_42,
            batch_norm__252,
            batch_norm__253,
            batch_norm__254,
            batch_norm__255,
            batch_norm__256,
            batch_norm__257,
            swish_32,
            conv2d_43,
            batch_norm__258,
            batch_norm__259,
            batch_norm__260,
            batch_norm__261,
            batch_norm__262,
            batch_norm__263,
            conv2d_44,
            batch_norm__264,
            batch_norm__265,
            batch_norm__266,
            batch_norm__267,
            batch_norm__268,
            batch_norm__269,
            add_9,
            swish_33,
            conv2d_45,
            batch_norm__270,
            batch_norm__271,
            batch_norm__272,
            batch_norm__273,
            batch_norm__274,
            batch_norm__275,
            swish_34,
            conv2d_46,
            batch_norm__276,
            batch_norm__277,
            batch_norm__278,
            batch_norm__279,
            batch_norm__280,
            batch_norm__281,
            conv2d_47,
            batch_norm__282,
            batch_norm__283,
            batch_norm__284,
            batch_norm__285,
            batch_norm__286,
            batch_norm__287,
            add_10,
            swish_35,
            conv2d_48,
            batch_norm__288,
            batch_norm__289,
            batch_norm__290,
            batch_norm__291,
            batch_norm__292,
            batch_norm__293,
            swish_36,
            conv2d_49,
            batch_norm__294,
            batch_norm__295,
            batch_norm__296,
            batch_norm__297,
            batch_norm__298,
            batch_norm__299,
            conv2d_50,
            batch_norm__300,
            batch_norm__301,
            batch_norm__302,
            batch_norm__303,
            batch_norm__304,
            batch_norm__305,
            add_11,
            swish_37,
            assign_6,
            concat_7,
            conv2d_51,
            batch_norm__306,
            batch_norm__307,
            batch_norm__308,
            batch_norm__309,
            batch_norm__310,
            batch_norm__311,
            conv2d_52,
            batch_norm__312,
            batch_norm__313,
            batch_norm__314,
            batch_norm__315,
            batch_norm__316,
            batch_norm__317,
            swish_38,
            assign_7,
            concat_8,
            conv2d_53,
            batch_norm__318,
            batch_norm__319,
            batch_norm__320,
            batch_norm__321,
            batch_norm__322,
            batch_norm__323,
            swish_39,
            conv2d_54,
            batch_norm__324,
            batch_norm__325,
            batch_norm__326,
            batch_norm__327,
            batch_norm__328,
            batch_norm__329,
            swish_40,
            conv2d_55,
            batch_norm__330,
            batch_norm__331,
            batch_norm__332,
            batch_norm__333,
            batch_norm__334,
            batch_norm__335,
            swish_41,
            conv2d_56,
            batch_norm__336,
            batch_norm__337,
            batch_norm__338,
            batch_norm__339,
            batch_norm__340,
            batch_norm__341,
            conv2d_57,
            batch_norm__342,
            batch_norm__343,
            batch_norm__344,
            batch_norm__345,
            batch_norm__346,
            batch_norm__347,
            add_12,
            swish_42,
            conv2d_58,
            batch_norm__348,
            batch_norm__349,
            batch_norm__350,
            batch_norm__351,
            batch_norm__352,
            batch_norm__353,
            swish_43,
            conv2d_59,
            batch_norm__354,
            batch_norm__355,
            batch_norm__356,
            batch_norm__357,
            batch_norm__358,
            batch_norm__359,
            conv2d_60,
            batch_norm__360,
            batch_norm__361,
            batch_norm__362,
            batch_norm__363,
            batch_norm__364,
            batch_norm__365,
            add_13,
            swish_44,
            conv2d_61,
            batch_norm__366,
            batch_norm__367,
            batch_norm__368,
            batch_norm__369,
            batch_norm__370,
            batch_norm__371,
            swish_45,
            conv2d_62,
            batch_norm__372,
            batch_norm__373,
            batch_norm__374,
            batch_norm__375,
            batch_norm__376,
            batch_norm__377,
            conv2d_63,
            batch_norm__378,
            batch_norm__379,
            batch_norm__380,
            batch_norm__381,
            batch_norm__382,
            batch_norm__383,
            add_14,
            swish_46,
            assign_8,
            concat_9,
            conv2d_64,
            batch_norm__384,
            batch_norm__385,
            batch_norm__386,
            batch_norm__387,
            batch_norm__388,
            batch_norm__389,
            swish_47,
            swish_48,
            swish_49,
            reshape_0,
        )
