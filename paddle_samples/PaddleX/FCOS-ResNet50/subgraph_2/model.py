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
        data_0,
        data_1,
        data_2,
        data_3,
        data_4,
        data_5,
    ):
        # pd_op.conv2d: (1x64x640x448xf32) <- (1x3x1280x896xf32, 64x3x7x7xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_5, parameter_318, [2, 2], [3, 3], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_5, parameter_318

        # pd_op.batch_norm_: (1x64x640x448xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (1x64x640x448xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_317,
                parameter_316,
                parameter_315,
                parameter_314,
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
        del conv2d_0, parameter_314, parameter_315, parameter_316, parameter_317

        # pd_op.relu: (1x64x640x448xf32) <- (1x64x640x448xf32)
        relu_0 = paddle._C_ops.relu(batch_norm__0)
        del batch_norm__0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [3, 3]

        # pd_op.pool2d: (1x64x320x224xf32) <- (1x64x640x448xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            relu_0,
            full_int_array_0,
            [2, 2],
            [1, 1],
            False,
            True,
            "NCHW",
            "max",
            False,
            False,
            "EXPLICIT",
        )
        del full_int_array_0, relu_0

        # pd_op.conv2d: (1x64x320x224xf32) <- (1x64x320x224xf32, 64x64x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            pool2d_0, parameter_313, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_313

        # pd_op.batch_norm_: (1x64x320x224xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (1x64x320x224xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_312,
                parameter_311,
                parameter_310,
                parameter_309,
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
        del conv2d_1, parameter_309, parameter_310, parameter_311, parameter_312

        # pd_op.relu: (1x64x320x224xf32) <- (1x64x320x224xf32)
        relu_1 = paddle._C_ops.relu(batch_norm__6)
        del batch_norm__6

        # pd_op.conv2d: (1x64x320x224xf32) <- (1x64x320x224xf32, 64x64x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            relu_1, parameter_308, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_308, relu_1

        # pd_op.batch_norm_: (1x64x320x224xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (1x64x320x224xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_307,
                parameter_306,
                parameter_305,
                parameter_304,
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
        del conv2d_2, parameter_304, parameter_305, parameter_306, parameter_307

        # pd_op.relu: (1x64x320x224xf32) <- (1x64x320x224xf32)
        relu_2 = paddle._C_ops.relu(batch_norm__12)
        del batch_norm__12

        # pd_op.conv2d: (1x256x320x224xf32) <- (1x64x320x224xf32, 256x64x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            relu_2, parameter_303, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_303, relu_2

        # pd_op.batch_norm_: (1x256x320x224xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x320x224xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_302,
                parameter_301,
                parameter_300,
                parameter_299,
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
        del conv2d_3, parameter_299, parameter_300, parameter_301, parameter_302

        # pd_op.conv2d: (1x256x320x224xf32) <- (1x64x320x224xf32, 256x64x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            pool2d_0, parameter_298, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_298, pool2d_0

        # pd_op.batch_norm_: (1x256x320x224xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x320x224xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_297,
                parameter_296,
                parameter_295,
                parameter_294,
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
        del conv2d_4, parameter_294, parameter_295, parameter_296, parameter_297

        # pd_op.add: (1x256x320x224xf32) <- (1x256x320x224xf32, 1x256x320x224xf32)
        add_10 = paddle._C_ops.add(batch_norm__18, batch_norm__24)
        del batch_norm__18, batch_norm__24

        # pd_op.relu: (1x256x320x224xf32) <- (1x256x320x224xf32)
        relu_3 = paddle._C_ops.relu(add_10)
        del add_10

        # pd_op.conv2d: (1x64x320x224xf32) <- (1x256x320x224xf32, 64x256x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            relu_3, parameter_293, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_293

        # pd_op.batch_norm_: (1x64x320x224xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (1x64x320x224xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_292,
                parameter_291,
                parameter_290,
                parameter_289,
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
        del conv2d_5, parameter_289, parameter_290, parameter_291, parameter_292

        # pd_op.relu: (1x64x320x224xf32) <- (1x64x320x224xf32)
        relu_4 = paddle._C_ops.relu(batch_norm__30)
        del batch_norm__30

        # pd_op.conv2d: (1x64x320x224xf32) <- (1x64x320x224xf32, 64x64x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            relu_4, parameter_288, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_288, relu_4

        # pd_op.batch_norm_: (1x64x320x224xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (1x64x320x224xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_287,
                parameter_286,
                parameter_285,
                parameter_284,
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
        del conv2d_6, parameter_284, parameter_285, parameter_286, parameter_287

        # pd_op.relu: (1x64x320x224xf32) <- (1x64x320x224xf32)
        relu_5 = paddle._C_ops.relu(batch_norm__36)
        del batch_norm__36

        # pd_op.conv2d: (1x256x320x224xf32) <- (1x64x320x224xf32, 256x64x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            relu_5, parameter_283, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_283, relu_5

        # pd_op.batch_norm_: (1x256x320x224xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x320x224xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_282,
                parameter_281,
                parameter_280,
                parameter_279,
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
        del conv2d_7, parameter_279, parameter_280, parameter_281, parameter_282

        # pd_op.add: (1x256x320x224xf32) <- (1x256x320x224xf32, 1x256x320x224xf32)
        add_11 = paddle._C_ops.add(batch_norm__42, relu_3)
        del batch_norm__42, relu_3

        # pd_op.relu: (1x256x320x224xf32) <- (1x256x320x224xf32)
        relu_6 = paddle._C_ops.relu(add_11)
        del add_11

        # pd_op.conv2d: (1x64x320x224xf32) <- (1x256x320x224xf32, 64x256x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            relu_6, parameter_278, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_278

        # pd_op.batch_norm_: (1x64x320x224xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (1x64x320x224xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_277,
                parameter_276,
                parameter_275,
                parameter_274,
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
        del conv2d_8, parameter_274, parameter_275, parameter_276, parameter_277

        # pd_op.relu: (1x64x320x224xf32) <- (1x64x320x224xf32)
        relu_7 = paddle._C_ops.relu(batch_norm__48)
        del batch_norm__48

        # pd_op.conv2d: (1x64x320x224xf32) <- (1x64x320x224xf32, 64x64x3x3xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            relu_7, parameter_273, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_273, relu_7

        # pd_op.batch_norm_: (1x64x320x224xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (1x64x320x224xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_272,
                parameter_271,
                parameter_270,
                parameter_269,
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
        del conv2d_9, parameter_269, parameter_270, parameter_271, parameter_272

        # pd_op.relu: (1x64x320x224xf32) <- (1x64x320x224xf32)
        relu_8 = paddle._C_ops.relu(batch_norm__54)
        del batch_norm__54

        # pd_op.conv2d: (1x256x320x224xf32) <- (1x64x320x224xf32, 256x64x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            relu_8, parameter_268, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_268, relu_8

        # pd_op.batch_norm_: (1x256x320x224xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x320x224xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_267,
                parameter_266,
                parameter_265,
                parameter_264,
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
        del conv2d_10, parameter_264, parameter_265, parameter_266, parameter_267

        # pd_op.add: (1x256x320x224xf32) <- (1x256x320x224xf32, 1x256x320x224xf32)
        add_12 = paddle._C_ops.add(batch_norm__60, relu_6)
        del batch_norm__60, relu_6

        # pd_op.relu: (1x256x320x224xf32) <- (1x256x320x224xf32)
        relu_9 = paddle._C_ops.relu(add_12)
        del add_12

        # pd_op.conv2d: (1x128x320x224xf32) <- (1x256x320x224xf32, 128x256x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            relu_9, parameter_263, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_263

        # pd_op.batch_norm_: (1x128x320x224xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (1x128x320x224xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_262,
                parameter_261,
                parameter_260,
                parameter_259,
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
        del conv2d_11, parameter_259, parameter_260, parameter_261, parameter_262

        # pd_op.relu: (1x128x320x224xf32) <- (1x128x320x224xf32)
        relu_10 = paddle._C_ops.relu(batch_norm__66)
        del batch_norm__66

        # pd_op.conv2d: (1x128x160x112xf32) <- (1x128x320x224xf32, 128x128x3x3xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            relu_10, parameter_258, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_258, relu_10

        # pd_op.batch_norm_: (1x128x160x112xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (1x128x160x112xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_257,
                parameter_256,
                parameter_255,
                parameter_254,
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
        del conv2d_12, parameter_254, parameter_255, parameter_256, parameter_257

        # pd_op.relu: (1x128x160x112xf32) <- (1x128x160x112xf32)
        relu_11 = paddle._C_ops.relu(batch_norm__72)
        del batch_norm__72

        # pd_op.conv2d: (1x512x160x112xf32) <- (1x128x160x112xf32, 512x128x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            relu_11, parameter_253, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_253, relu_11

        # pd_op.batch_norm_: (1x512x160x112xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (1x512x160x112xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_252,
                parameter_251,
                parameter_250,
                parameter_249,
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
        del conv2d_13, parameter_249, parameter_250, parameter_251, parameter_252

        # pd_op.conv2d: (1x512x160x112xf32) <- (1x256x320x224xf32, 512x256x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            relu_9, parameter_248, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_248, relu_9

        # pd_op.batch_norm_: (1x512x160x112xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (1x512x160x112xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_247,
                parameter_246,
                parameter_245,
                parameter_244,
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
        del conv2d_14, parameter_244, parameter_245, parameter_246, parameter_247

        # pd_op.add: (1x512x160x112xf32) <- (1x512x160x112xf32, 1x512x160x112xf32)
        add_13 = paddle._C_ops.add(batch_norm__78, batch_norm__84)
        del batch_norm__78, batch_norm__84

        # pd_op.relu: (1x512x160x112xf32) <- (1x512x160x112xf32)
        relu_12 = paddle._C_ops.relu(add_13)
        del add_13

        # pd_op.conv2d: (1x128x160x112xf32) <- (1x512x160x112xf32, 128x512x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            relu_12, parameter_243, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_243

        # pd_op.batch_norm_: (1x128x160x112xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (1x128x160x112xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_242,
                parameter_241,
                parameter_240,
                parameter_239,
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
        del conv2d_15, parameter_239, parameter_240, parameter_241, parameter_242

        # pd_op.relu: (1x128x160x112xf32) <- (1x128x160x112xf32)
        relu_13 = paddle._C_ops.relu(batch_norm__90)
        del batch_norm__90

        # pd_op.conv2d: (1x128x160x112xf32) <- (1x128x160x112xf32, 128x128x3x3xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            relu_13, parameter_238, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_238, relu_13

        # pd_op.batch_norm_: (1x128x160x112xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (1x128x160x112xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_237,
                parameter_236,
                parameter_235,
                parameter_234,
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
        del conv2d_16, parameter_234, parameter_235, parameter_236, parameter_237

        # pd_op.relu: (1x128x160x112xf32) <- (1x128x160x112xf32)
        relu_14 = paddle._C_ops.relu(batch_norm__96)
        del batch_norm__96

        # pd_op.conv2d: (1x512x160x112xf32) <- (1x128x160x112xf32, 512x128x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            relu_14, parameter_233, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_233, relu_14

        # pd_op.batch_norm_: (1x512x160x112xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (1x512x160x112xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_232,
                parameter_231,
                parameter_230,
                parameter_229,
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
        del conv2d_17, parameter_229, parameter_230, parameter_231, parameter_232

        # pd_op.add: (1x512x160x112xf32) <- (1x512x160x112xf32, 1x512x160x112xf32)
        add_14 = paddle._C_ops.add(batch_norm__102, relu_12)
        del batch_norm__102, relu_12

        # pd_op.relu: (1x512x160x112xf32) <- (1x512x160x112xf32)
        relu_15 = paddle._C_ops.relu(add_14)
        del add_14

        # pd_op.conv2d: (1x128x160x112xf32) <- (1x512x160x112xf32, 128x512x1x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            relu_15, parameter_228, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_228

        # pd_op.batch_norm_: (1x128x160x112xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (1x128x160x112xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_227,
                parameter_226,
                parameter_225,
                parameter_224,
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
        del conv2d_18, parameter_224, parameter_225, parameter_226, parameter_227

        # pd_op.relu: (1x128x160x112xf32) <- (1x128x160x112xf32)
        relu_16 = paddle._C_ops.relu(batch_norm__108)
        del batch_norm__108

        # pd_op.conv2d: (1x128x160x112xf32) <- (1x128x160x112xf32, 128x128x3x3xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            relu_16, parameter_223, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_223, relu_16

        # pd_op.batch_norm_: (1x128x160x112xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (1x128x160x112xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_222,
                parameter_221,
                parameter_220,
                parameter_219,
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
        del conv2d_19, parameter_219, parameter_220, parameter_221, parameter_222

        # pd_op.relu: (1x128x160x112xf32) <- (1x128x160x112xf32)
        relu_17 = paddle._C_ops.relu(batch_norm__114)
        del batch_norm__114

        # pd_op.conv2d: (1x512x160x112xf32) <- (1x128x160x112xf32, 512x128x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            relu_17, parameter_218, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_218, relu_17

        # pd_op.batch_norm_: (1x512x160x112xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (1x512x160x112xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_217,
                parameter_216,
                parameter_215,
                parameter_214,
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
        del conv2d_20, parameter_214, parameter_215, parameter_216, parameter_217

        # pd_op.add: (1x512x160x112xf32) <- (1x512x160x112xf32, 1x512x160x112xf32)
        add_15 = paddle._C_ops.add(batch_norm__120, relu_15)
        del batch_norm__120, relu_15

        # pd_op.relu: (1x512x160x112xf32) <- (1x512x160x112xf32)
        relu_18 = paddle._C_ops.relu(add_15)
        del add_15

        # pd_op.conv2d: (1x128x160x112xf32) <- (1x512x160x112xf32, 128x512x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            relu_18, parameter_213, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_213

        # pd_op.batch_norm_: (1x128x160x112xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (1x128x160x112xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_212,
                parameter_211,
                parameter_210,
                parameter_209,
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
        del conv2d_21, parameter_209, parameter_210, parameter_211, parameter_212

        # pd_op.relu: (1x128x160x112xf32) <- (1x128x160x112xf32)
        relu_19 = paddle._C_ops.relu(batch_norm__126)
        del batch_norm__126

        # pd_op.conv2d: (1x128x160x112xf32) <- (1x128x160x112xf32, 128x128x3x3xf32)
        conv2d_22 = paddle._C_ops.conv2d(
            relu_19, parameter_208, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_208, relu_19

        # pd_op.batch_norm_: (1x128x160x112xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (1x128x160x112xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_207,
                parameter_206,
                parameter_205,
                parameter_204,
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
        del conv2d_22, parameter_204, parameter_205, parameter_206, parameter_207

        # pd_op.relu: (1x128x160x112xf32) <- (1x128x160x112xf32)
        relu_20 = paddle._C_ops.relu(batch_norm__132)
        del batch_norm__132

        # pd_op.conv2d: (1x512x160x112xf32) <- (1x128x160x112xf32, 512x128x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(
            relu_20, parameter_203, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_203, relu_20

        # pd_op.batch_norm_: (1x512x160x112xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (1x512x160x112xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_202,
                parameter_201,
                parameter_200,
                parameter_199,
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
        del conv2d_23, parameter_199, parameter_200, parameter_201, parameter_202

        # pd_op.add: (1x512x160x112xf32) <- (1x512x160x112xf32, 1x512x160x112xf32)
        add_16 = paddle._C_ops.add(batch_norm__138, relu_18)
        del batch_norm__138, relu_18

        # pd_op.relu: (1x512x160x112xf32) <- (1x512x160x112xf32)
        relu_21 = paddle._C_ops.relu(add_16)
        del add_16

        # pd_op.conv2d: (1x256x160x112xf32) <- (1x512x160x112xf32, 256x512x1x1xf32)
        conv2d_24 = paddle._C_ops.conv2d(
            relu_21, parameter_198, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_198

        # pd_op.batch_norm_: (1x256x160x112xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x160x112xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_197,
                parameter_196,
                parameter_195,
                parameter_194,
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
        del conv2d_24, parameter_194, parameter_195, parameter_196, parameter_197

        # pd_op.relu: (1x256x160x112xf32) <- (1x256x160x112xf32)
        relu_22 = paddle._C_ops.relu(batch_norm__144)
        del batch_norm__144

        # pd_op.conv2d: (1x256x80x56xf32) <- (1x256x160x112xf32, 256x256x3x3xf32)
        conv2d_25 = paddle._C_ops.conv2d(
            relu_22, parameter_193, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_193, relu_22

        # pd_op.batch_norm_: (1x256x80x56xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x80x56xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_192,
                parameter_191,
                parameter_190,
                parameter_189,
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
        del conv2d_25, parameter_189, parameter_190, parameter_191, parameter_192

        # pd_op.relu: (1x256x80x56xf32) <- (1x256x80x56xf32)
        relu_23 = paddle._C_ops.relu(batch_norm__150)
        del batch_norm__150

        # pd_op.conv2d: (1x1024x80x56xf32) <- (1x256x80x56xf32, 1024x256x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(
            relu_23, parameter_188, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_188, relu_23

        # pd_op.batch_norm_: (1x1024x80x56xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (1x1024x80x56xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
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
                parameter_187,
                parameter_186,
                parameter_185,
                parameter_184,
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
        del conv2d_26, parameter_184, parameter_185, parameter_186, parameter_187

        # pd_op.conv2d: (1x1024x80x56xf32) <- (1x512x160x112xf32, 1024x512x1x1xf32)
        conv2d_27 = paddle._C_ops.conv2d(
            relu_21, parameter_183, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_183

        # pd_op.batch_norm_: (1x1024x80x56xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (1x1024x80x56xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
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
                parameter_182,
                parameter_181,
                parameter_180,
                parameter_179,
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
        del conv2d_27, parameter_179, parameter_180, parameter_181, parameter_182

        # pd_op.add: (1x1024x80x56xf32) <- (1x1024x80x56xf32, 1x1024x80x56xf32)
        add_17 = paddle._C_ops.add(batch_norm__156, batch_norm__162)
        del batch_norm__156, batch_norm__162

        # pd_op.relu: (1x1024x80x56xf32) <- (1x1024x80x56xf32)
        relu_24 = paddle._C_ops.relu(add_17)
        del add_17

        # pd_op.conv2d: (1x256x80x56xf32) <- (1x1024x80x56xf32, 256x1024x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(
            relu_24, parameter_178, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_178

        # pd_op.batch_norm_: (1x256x80x56xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x80x56xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
        del conv2d_28, parameter_174, parameter_175, parameter_176, parameter_177

        # pd_op.relu: (1x256x80x56xf32) <- (1x256x80x56xf32)
        relu_25 = paddle._C_ops.relu(batch_norm__168)
        del batch_norm__168

        # pd_op.conv2d: (1x256x80x56xf32) <- (1x256x80x56xf32, 256x256x3x3xf32)
        conv2d_29 = paddle._C_ops.conv2d(
            relu_25, parameter_173, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_173, relu_25

        # pd_op.batch_norm_: (1x256x80x56xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x80x56xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_172,
                parameter_171,
                parameter_170,
                parameter_169,
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
        del conv2d_29, parameter_169, parameter_170, parameter_171, parameter_172

        # pd_op.relu: (1x256x80x56xf32) <- (1x256x80x56xf32)
        relu_26 = paddle._C_ops.relu(batch_norm__174)
        del batch_norm__174

        # pd_op.conv2d: (1x1024x80x56xf32) <- (1x256x80x56xf32, 1024x256x1x1xf32)
        conv2d_30 = paddle._C_ops.conv2d(
            relu_26, parameter_168, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_168, relu_26

        # pd_op.batch_norm_: (1x1024x80x56xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (1x1024x80x56xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
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
        del conv2d_30, parameter_164, parameter_165, parameter_166, parameter_167

        # pd_op.add: (1x1024x80x56xf32) <- (1x1024x80x56xf32, 1x1024x80x56xf32)
        add_18 = paddle._C_ops.add(batch_norm__180, relu_24)
        del batch_norm__180, relu_24

        # pd_op.relu: (1x1024x80x56xf32) <- (1x1024x80x56xf32)
        relu_27 = paddle._C_ops.relu(add_18)
        del add_18

        # pd_op.conv2d: (1x256x80x56xf32) <- (1x1024x80x56xf32, 256x1024x1x1xf32)
        conv2d_31 = paddle._C_ops.conv2d(
            relu_27, parameter_163, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_163

        # pd_op.batch_norm_: (1x256x80x56xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x80x56xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
        del conv2d_31, parameter_159, parameter_160, parameter_161, parameter_162

        # pd_op.relu: (1x256x80x56xf32) <- (1x256x80x56xf32)
        relu_28 = paddle._C_ops.relu(batch_norm__186)
        del batch_norm__186

        # pd_op.conv2d: (1x256x80x56xf32) <- (1x256x80x56xf32, 256x256x3x3xf32)
        conv2d_32 = paddle._C_ops.conv2d(
            relu_28, parameter_158, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_158, relu_28

        # pd_op.batch_norm_: (1x256x80x56xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x80x56xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
        del conv2d_32, parameter_154, parameter_155, parameter_156, parameter_157

        # pd_op.relu: (1x256x80x56xf32) <- (1x256x80x56xf32)
        relu_29 = paddle._C_ops.relu(batch_norm__192)
        del batch_norm__192

        # pd_op.conv2d: (1x1024x80x56xf32) <- (1x256x80x56xf32, 1024x256x1x1xf32)
        conv2d_33 = paddle._C_ops.conv2d(
            relu_29, parameter_153, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_153, relu_29

        # pd_op.batch_norm_: (1x1024x80x56xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (1x1024x80x56xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
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
        del conv2d_33, parameter_149, parameter_150, parameter_151, parameter_152

        # pd_op.add: (1x1024x80x56xf32) <- (1x1024x80x56xf32, 1x1024x80x56xf32)
        add_19 = paddle._C_ops.add(batch_norm__198, relu_27)
        del batch_norm__198, relu_27

        # pd_op.relu: (1x1024x80x56xf32) <- (1x1024x80x56xf32)
        relu_30 = paddle._C_ops.relu(add_19)
        del add_19

        # pd_op.conv2d: (1x256x80x56xf32) <- (1x1024x80x56xf32, 256x1024x1x1xf32)
        conv2d_34 = paddle._C_ops.conv2d(
            relu_30, parameter_148, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_148

        # pd_op.batch_norm_: (1x256x80x56xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x80x56xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
        del conv2d_34, parameter_144, parameter_145, parameter_146, parameter_147

        # pd_op.relu: (1x256x80x56xf32) <- (1x256x80x56xf32)
        relu_31 = paddle._C_ops.relu(batch_norm__204)
        del batch_norm__204

        # pd_op.conv2d: (1x256x80x56xf32) <- (1x256x80x56xf32, 256x256x3x3xf32)
        conv2d_35 = paddle._C_ops.conv2d(
            relu_31, parameter_143, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_143, relu_31

        # pd_op.batch_norm_: (1x256x80x56xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x80x56xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
        del conv2d_35, parameter_139, parameter_140, parameter_141, parameter_142

        # pd_op.relu: (1x256x80x56xf32) <- (1x256x80x56xf32)
        relu_32 = paddle._C_ops.relu(batch_norm__210)
        del batch_norm__210

        # pd_op.conv2d: (1x1024x80x56xf32) <- (1x256x80x56xf32, 1024x256x1x1xf32)
        conv2d_36 = paddle._C_ops.conv2d(
            relu_32, parameter_138, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_138, relu_32

        # pd_op.batch_norm_: (1x1024x80x56xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (1x1024x80x56xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
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
                parameter_137,
                parameter_136,
                parameter_135,
                parameter_134,
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
        del conv2d_36, parameter_134, parameter_135, parameter_136, parameter_137

        # pd_op.add: (1x1024x80x56xf32) <- (1x1024x80x56xf32, 1x1024x80x56xf32)
        add_20 = paddle._C_ops.add(batch_norm__216, relu_30)
        del batch_norm__216, relu_30

        # pd_op.relu: (1x1024x80x56xf32) <- (1x1024x80x56xf32)
        relu_33 = paddle._C_ops.relu(add_20)
        del add_20

        # pd_op.conv2d: (1x256x80x56xf32) <- (1x1024x80x56xf32, 256x1024x1x1xf32)
        conv2d_37 = paddle._C_ops.conv2d(
            relu_33, parameter_133, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_133

        # pd_op.batch_norm_: (1x256x80x56xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x80x56xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_132,
                parameter_131,
                parameter_130,
                parameter_129,
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
        del conv2d_37, parameter_129, parameter_130, parameter_131, parameter_132

        # pd_op.relu: (1x256x80x56xf32) <- (1x256x80x56xf32)
        relu_34 = paddle._C_ops.relu(batch_norm__222)
        del batch_norm__222

        # pd_op.conv2d: (1x256x80x56xf32) <- (1x256x80x56xf32, 256x256x3x3xf32)
        conv2d_38 = paddle._C_ops.conv2d(
            relu_34, parameter_128, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_128, relu_34

        # pd_op.batch_norm_: (1x256x80x56xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x80x56xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_127,
                parameter_126,
                parameter_125,
                parameter_124,
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
        del conv2d_38, parameter_124, parameter_125, parameter_126, parameter_127

        # pd_op.relu: (1x256x80x56xf32) <- (1x256x80x56xf32)
        relu_35 = paddle._C_ops.relu(batch_norm__228)
        del batch_norm__228

        # pd_op.conv2d: (1x1024x80x56xf32) <- (1x256x80x56xf32, 1024x256x1x1xf32)
        conv2d_39 = paddle._C_ops.conv2d(
            relu_35, parameter_123, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_123, relu_35

        # pd_op.batch_norm_: (1x1024x80x56xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (1x1024x80x56xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
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
                parameter_122,
                parameter_121,
                parameter_120,
                parameter_119,
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
        del conv2d_39, parameter_119, parameter_120, parameter_121, parameter_122

        # pd_op.add: (1x1024x80x56xf32) <- (1x1024x80x56xf32, 1x1024x80x56xf32)
        add_21 = paddle._C_ops.add(batch_norm__234, relu_33)
        del batch_norm__234, relu_33

        # pd_op.relu: (1x1024x80x56xf32) <- (1x1024x80x56xf32)
        relu_36 = paddle._C_ops.relu(add_21)
        del add_21

        # pd_op.conv2d: (1x256x80x56xf32) <- (1x1024x80x56xf32, 256x1024x1x1xf32)
        conv2d_40 = paddle._C_ops.conv2d(
            relu_36, parameter_118, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_118

        # pd_op.batch_norm_: (1x256x80x56xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x80x56xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
        del conv2d_40, parameter_114, parameter_115, parameter_116, parameter_117

        # pd_op.relu: (1x256x80x56xf32) <- (1x256x80x56xf32)
        relu_37 = paddle._C_ops.relu(batch_norm__240)
        del batch_norm__240

        # pd_op.conv2d: (1x256x80x56xf32) <- (1x256x80x56xf32, 256x256x3x3xf32)
        conv2d_41 = paddle._C_ops.conv2d(
            relu_37, parameter_113, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_113, relu_37

        # pd_op.batch_norm_: (1x256x80x56xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x80x56xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_112,
                parameter_111,
                parameter_110,
                parameter_109,
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
        del conv2d_41, parameter_109, parameter_110, parameter_111, parameter_112

        # pd_op.relu: (1x256x80x56xf32) <- (1x256x80x56xf32)
        relu_38 = paddle._C_ops.relu(batch_norm__246)
        del batch_norm__246

        # pd_op.conv2d: (1x1024x80x56xf32) <- (1x256x80x56xf32, 1024x256x1x1xf32)
        conv2d_42 = paddle._C_ops.conv2d(
            relu_38, parameter_108, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_108, relu_38

        # pd_op.batch_norm_: (1x1024x80x56xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (1x1024x80x56xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
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
        del conv2d_42, parameter_104, parameter_105, parameter_106, parameter_107

        # pd_op.add: (1x1024x80x56xf32) <- (1x1024x80x56xf32, 1x1024x80x56xf32)
        add_22 = paddle._C_ops.add(batch_norm__252, relu_36)
        del batch_norm__252, relu_36

        # pd_op.relu: (1x1024x80x56xf32) <- (1x1024x80x56xf32)
        relu_39 = paddle._C_ops.relu(add_22)
        del add_22

        # pd_op.conv2d: (1x512x80x56xf32) <- (1x1024x80x56xf32, 512x1024x1x1xf32)
        conv2d_43 = paddle._C_ops.conv2d(
            relu_39, parameter_103, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_103

        # pd_op.batch_norm_: (1x512x80x56xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (1x512x80x56xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_102,
                parameter_101,
                parameter_100,
                parameter_99,
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
        del conv2d_43, parameter_100, parameter_101, parameter_102, parameter_99

        # pd_op.relu: (1x512x80x56xf32) <- (1x512x80x56xf32)
        relu_40 = paddle._C_ops.relu(batch_norm__258)
        del batch_norm__258

        # pd_op.conv2d: (1x512x40x28xf32) <- (1x512x80x56xf32, 512x512x3x3xf32)
        conv2d_44 = paddle._C_ops.conv2d(
            relu_40, parameter_98, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_98, relu_40

        # pd_op.batch_norm_: (1x512x40x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (1x512x40x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_97,
                parameter_96,
                parameter_95,
                parameter_94,
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
        del conv2d_44, parameter_94, parameter_95, parameter_96, parameter_97

        # pd_op.relu: (1x512x40x28xf32) <- (1x512x40x28xf32)
        relu_41 = paddle._C_ops.relu(batch_norm__264)
        del batch_norm__264

        # pd_op.conv2d: (1x2048x40x28xf32) <- (1x512x40x28xf32, 2048x512x1x1xf32)
        conv2d_45 = paddle._C_ops.conv2d(
            relu_41, parameter_93, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_93, relu_41

        # pd_op.batch_norm_: (1x2048x40x28xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (1x2048x40x28xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
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
                parameter_92,
                parameter_91,
                parameter_90,
                parameter_89,
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
        del conv2d_45, parameter_89, parameter_90, parameter_91, parameter_92

        # pd_op.conv2d: (1x2048x40x28xf32) <- (1x1024x80x56xf32, 2048x1024x1x1xf32)
        conv2d_46 = paddle._C_ops.conv2d(
            relu_39, parameter_88, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_88

        # pd_op.batch_norm_: (1x2048x40x28xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (1x2048x40x28xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
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
                parameter_87,
                parameter_86,
                parameter_85,
                parameter_84,
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
        del conv2d_46, parameter_84, parameter_85, parameter_86, parameter_87

        # pd_op.add: (1x2048x40x28xf32) <- (1x2048x40x28xf32, 1x2048x40x28xf32)
        add_23 = paddle._C_ops.add(batch_norm__270, batch_norm__276)
        del batch_norm__270, batch_norm__276

        # pd_op.relu: (1x2048x40x28xf32) <- (1x2048x40x28xf32)
        relu_42 = paddle._C_ops.relu(add_23)
        del add_23

        # pd_op.conv2d: (1x512x40x28xf32) <- (1x2048x40x28xf32, 512x2048x1x1xf32)
        conv2d_47 = paddle._C_ops.conv2d(
            relu_42, parameter_83, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_83

        # pd_op.batch_norm_: (1x512x40x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (1x512x40x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_82,
                parameter_81,
                parameter_80,
                parameter_79,
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
        del conv2d_47, parameter_79, parameter_80, parameter_81, parameter_82

        # pd_op.relu: (1x512x40x28xf32) <- (1x512x40x28xf32)
        relu_43 = paddle._C_ops.relu(batch_norm__282)
        del batch_norm__282

        # pd_op.conv2d: (1x512x40x28xf32) <- (1x512x40x28xf32, 512x512x3x3xf32)
        conv2d_48 = paddle._C_ops.conv2d(
            relu_43, parameter_78, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_78, relu_43

        # pd_op.batch_norm_: (1x512x40x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (1x512x40x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_77,
                parameter_76,
                parameter_75,
                parameter_74,
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
        del conv2d_48, parameter_74, parameter_75, parameter_76, parameter_77

        # pd_op.relu: (1x512x40x28xf32) <- (1x512x40x28xf32)
        relu_44 = paddle._C_ops.relu(batch_norm__288)
        del batch_norm__288

        # pd_op.conv2d: (1x2048x40x28xf32) <- (1x512x40x28xf32, 2048x512x1x1xf32)
        conv2d_49 = paddle._C_ops.conv2d(
            relu_44, parameter_73, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_73, relu_44

        # pd_op.batch_norm_: (1x2048x40x28xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (1x2048x40x28xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
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
                parameter_72,
                parameter_71,
                parameter_70,
                parameter_69,
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
        del conv2d_49, parameter_69, parameter_70, parameter_71, parameter_72

        # pd_op.add: (1x2048x40x28xf32) <- (1x2048x40x28xf32, 1x2048x40x28xf32)
        add_24 = paddle._C_ops.add(batch_norm__294, relu_42)
        del batch_norm__294, relu_42

        # pd_op.relu: (1x2048x40x28xf32) <- (1x2048x40x28xf32)
        relu_45 = paddle._C_ops.relu(add_24)
        del add_24

        # pd_op.conv2d: (1x512x40x28xf32) <- (1x2048x40x28xf32, 512x2048x1x1xf32)
        conv2d_50 = paddle._C_ops.conv2d(
            relu_45, parameter_68, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_68

        # pd_op.batch_norm_: (1x512x40x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (1x512x40x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_67,
                parameter_66,
                parameter_65,
                parameter_64,
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
        del conv2d_50, parameter_64, parameter_65, parameter_66, parameter_67

        # pd_op.relu: (1x512x40x28xf32) <- (1x512x40x28xf32)
        relu_46 = paddle._C_ops.relu(batch_norm__300)
        del batch_norm__300

        # pd_op.conv2d: (1x512x40x28xf32) <- (1x512x40x28xf32, 512x512x3x3xf32)
        conv2d_51 = paddle._C_ops.conv2d(
            relu_46, parameter_63, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_63, relu_46

        # pd_op.batch_norm_: (1x512x40x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (1x512x40x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_62,
                parameter_61,
                parameter_60,
                parameter_59,
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
        del conv2d_51, parameter_59, parameter_60, parameter_61, parameter_62

        # pd_op.relu: (1x512x40x28xf32) <- (1x512x40x28xf32)
        relu_47 = paddle._C_ops.relu(batch_norm__306)
        del batch_norm__306

        # pd_op.conv2d: (1x2048x40x28xf32) <- (1x512x40x28xf32, 2048x512x1x1xf32)
        conv2d_52 = paddle._C_ops.conv2d(
            relu_47, parameter_58, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_58, relu_47

        # pd_op.batch_norm_: (1x2048x40x28xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (1x2048x40x28xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
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
                parameter_57,
                parameter_56,
                parameter_55,
                parameter_54,
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
        del conv2d_52, parameter_54, parameter_55, parameter_56, parameter_57

        # pd_op.add: (1x2048x40x28xf32) <- (1x2048x40x28xf32, 1x2048x40x28xf32)
        add_25 = paddle._C_ops.add(batch_norm__312, relu_45)
        del batch_norm__312, relu_45

        # pd_op.relu: (1x2048x40x28xf32) <- (1x2048x40x28xf32)
        relu_48 = paddle._C_ops.relu(add_25)
        del add_25

        # pd_op.conv2d: (1x256x160x112xf32) <- (1x512x160x112xf32, 256x512x1x1xf32)
        conv2d_53 = paddle._C_ops.conv2d(
            relu_21, parameter_53, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_53, relu_21

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [1, -1, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_52, full_int_array_1)
        del parameter_52

        # pd_op.add: (1x256x160x112xf32) <- (1x256x160x112xf32, 1x256x1x1xf32)
        add_26 = paddle._C_ops.add(conv2d_53, reshape_0)
        del conv2d_53, reshape_0

        # pd_op.conv2d: (1x256x80x56xf32) <- (1x1024x80x56xf32, 256x1024x1x1xf32)
        conv2d_54 = paddle._C_ops.conv2d(
            relu_39, parameter_51, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_51, relu_39

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_50, full_int_array_1)
        del parameter_50

        # pd_op.add: (1x256x80x56xf32) <- (1x256x80x56xf32, 1x256x1x1xf32)
        add_27 = paddle._C_ops.add(conv2d_54, reshape_1)
        del conv2d_54, reshape_1

        # pd_op.conv2d: (1x256x40x28xf32) <- (1x2048x40x28xf32, 256x2048x1x1xf32)
        conv2d_55 = paddle._C_ops.conv2d(
            relu_48, parameter_49, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_49, relu_48

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(parameter_48, full_int_array_1)
        del parameter_48

        # pd_op.add: (1x256x40x28xf32) <- (1x256x40x28xf32, 1x256x1x1xf32)
        add_28 = paddle._C_ops.add(conv2d_55, reshape_2)
        del conv2d_55, reshape_2

        # pd_op.nearest_interp: (1x256x80x56xf32) <- (1x256x40x28xf32, None, None, None)
        nearest_interp_0 = paddle._C_ops.nearest_interp(
            add_28,
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

        # pd_op.add: (1x256x80x56xf32) <- (1x256x80x56xf32, 1x256x80x56xf32)
        add_29 = paddle._C_ops.add(add_27, nearest_interp_0)
        del add_27, nearest_interp_0

        # pd_op.nearest_interp: (1x256x160x112xf32) <- (1x256x80x56xf32, None, None, None)
        nearest_interp_1 = paddle._C_ops.nearest_interp(
            add_29,
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

        # pd_op.add: (1x256x160x112xf32) <- (1x256x160x112xf32, 1x256x160x112xf32)
        add_30 = paddle._C_ops.add(add_26, nearest_interp_1)
        del add_26, nearest_interp_1

        # pd_op.conv2d: (1x256x160x112xf32) <- (1x256x160x112xf32, 256x256x3x3xf32)
        conv2d_56 = paddle._C_ops.conv2d(
            add_30, parameter_47, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_30, parameter_47

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(parameter_46, full_int_array_1)
        del parameter_46

        # pd_op.add: (1x256x160x112xf32) <- (1x256x160x112xf32, 1x256x1x1xf32)
        add_31 = paddle._C_ops.add(conv2d_56, reshape_3)
        del conv2d_56, reshape_3

        # pd_op.conv2d: (1x256x80x56xf32) <- (1x256x80x56xf32, 256x256x3x3xf32)
        conv2d_57 = paddle._C_ops.conv2d(
            add_29, parameter_45, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_29, parameter_45

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(parameter_44, full_int_array_1)
        del parameter_44

        # pd_op.add: (1x256x80x56xf32) <- (1x256x80x56xf32, 1x256x1x1xf32)
        add_32 = paddle._C_ops.add(conv2d_57, reshape_4)
        del conv2d_57, reshape_4

        # pd_op.conv2d: (1x256x40x28xf32) <- (1x256x40x28xf32, 256x256x3x3xf32)
        conv2d_58 = paddle._C_ops.conv2d(
            add_28, parameter_43, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_28, parameter_43

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(parameter_42, full_int_array_1)
        del parameter_42

        # pd_op.add: (1x256x40x28xf32) <- (1x256x40x28xf32, 1x256x1x1xf32)
        add_33 = paddle._C_ops.add(conv2d_58, reshape_5)
        del conv2d_58, reshape_5

        # pd_op.conv2d: (1x256x20x14xf32) <- (1x256x40x28xf32, 256x256x3x3xf32)
        conv2d_59 = paddle._C_ops.conv2d(
            add_33, parameter_41, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_41

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(parameter_40, full_int_array_1)
        del parameter_40

        # pd_op.add: (1x256x20x14xf32) <- (1x256x20x14xf32, 1x256x1x1xf32)
        add_34 = paddle._C_ops.add(conv2d_59, reshape_6)
        del conv2d_59, reshape_6

        # pd_op.relu: (1x256x20x14xf32) <- (1x256x20x14xf32)
        relu_49 = paddle._C_ops.relu(add_34)

        # pd_op.conv2d: (1x256x10x7xf32) <- (1x256x20x14xf32, 256x256x3x3xf32)
        conv2d_60 = paddle._C_ops.conv2d(
            relu_49, parameter_39, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_39, relu_49

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(parameter_38, full_int_array_1)
        del parameter_38

        # pd_op.add: (1x256x10x7xf32) <- (1x256x10x7xf32, 1x256x1x1xf32)
        add_35 = paddle._C_ops.add(conv2d_60, reshape_7)
        del conv2d_60, reshape_7

        # pd_op.conv2d: (1x256x160x112xf32) <- (1x256x160x112xf32, 256x256x3x3xf32)
        conv2d_61 = paddle._C_ops.conv2d(
            add_31, parameter_37, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(parameter_36, full_int_array_1)
        del parameter_36

        # pd_op.add: (1x256x160x112xf32) <- (1x256x160x112xf32, 1x256x1x1xf32)
        add_36 = paddle._C_ops.add(conv2d_61, reshape_8)
        del conv2d_61

        # pd_op.group_norm: (1x256x160x112xf32, 1x32xf32, 1x32xf32) <- (1x256x160x112xf32, 256xf32, 256xf32)
        group_norm_0, group_norm_1, group_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_36, parameter_35, parameter_34, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_36

        # pd_op.relu: (1x256x160x112xf32) <- (1x256x160x112xf32)
        relu_50 = paddle._C_ops.relu(group_norm_0)
        del group_norm_0

        # pd_op.conv2d: (1x256x160x112xf32) <- (1x256x160x112xf32, 256x256x3x3xf32)
        conv2d_62 = paddle._C_ops.conv2d(
            add_31, parameter_33, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_31

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(parameter_32, full_int_array_1)
        del parameter_32

        # pd_op.add: (1x256x160x112xf32) <- (1x256x160x112xf32, 1x256x1x1xf32)
        add_37 = paddle._C_ops.add(conv2d_62, reshape_9)
        del conv2d_62

        # pd_op.group_norm: (1x256x160x112xf32, 1x32xf32, 1x32xf32) <- (1x256x160x112xf32, 256xf32, 256xf32)
        group_norm_3, group_norm_4, group_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_37, parameter_31, parameter_30, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_37

        # pd_op.relu: (1x256x160x112xf32) <- (1x256x160x112xf32)
        relu_51 = paddle._C_ops.relu(group_norm_3)
        del group_norm_3

        # pd_op.conv2d: (1x256x160x112xf32) <- (1x256x160x112xf32, 256x256x3x3xf32)
        conv2d_63 = paddle._C_ops.conv2d(
            relu_50, parameter_29, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del relu_50

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(parameter_28, full_int_array_1)
        del parameter_28

        # pd_op.add: (1x256x160x112xf32) <- (1x256x160x112xf32, 1x256x1x1xf32)
        add_38 = paddle._C_ops.add(conv2d_63, reshape_10)
        del conv2d_63

        # pd_op.group_norm: (1x256x160x112xf32, 1x32xf32, 1x32xf32) <- (1x256x160x112xf32, 256xf32, 256xf32)
        group_norm_6, group_norm_7, group_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_38, parameter_27, parameter_26, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_38

        # pd_op.relu: (1x256x160x112xf32) <- (1x256x160x112xf32)
        relu_52 = paddle._C_ops.relu(group_norm_6)
        del group_norm_6

        # pd_op.conv2d: (1x256x160x112xf32) <- (1x256x160x112xf32, 256x256x3x3xf32)
        conv2d_64 = paddle._C_ops.conv2d(
            relu_51, parameter_25, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del relu_51

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_11 = paddle._C_ops.reshape(parameter_24, full_int_array_1)
        del parameter_24

        # pd_op.add: (1x256x160x112xf32) <- (1x256x160x112xf32, 1x256x1x1xf32)
        add_39 = paddle._C_ops.add(conv2d_64, reshape_11)
        del conv2d_64

        # pd_op.group_norm: (1x256x160x112xf32, 1x32xf32, 1x32xf32) <- (1x256x160x112xf32, 256xf32, 256xf32)
        group_norm_9, group_norm_10, group_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_39, parameter_23, parameter_22, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_39

        # pd_op.relu: (1x256x160x112xf32) <- (1x256x160x112xf32)
        relu_53 = paddle._C_ops.relu(group_norm_9)
        del group_norm_9

        # pd_op.conv2d: (1x256x160x112xf32) <- (1x256x160x112xf32, 256x256x3x3xf32)
        conv2d_65 = paddle._C_ops.conv2d(
            relu_52, parameter_21, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del relu_52

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(parameter_20, full_int_array_1)
        del parameter_20

        # pd_op.add: (1x256x160x112xf32) <- (1x256x160x112xf32, 1x256x1x1xf32)
        add_40 = paddle._C_ops.add(conv2d_65, reshape_12)
        del conv2d_65

        # pd_op.group_norm: (1x256x160x112xf32, 1x32xf32, 1x32xf32) <- (1x256x160x112xf32, 256xf32, 256xf32)
        group_norm_12, group_norm_13, group_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_40, parameter_19, parameter_18, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_40

        # pd_op.relu: (1x256x160x112xf32) <- (1x256x160x112xf32)
        relu_54 = paddle._C_ops.relu(group_norm_12)
        del group_norm_12

        # pd_op.conv2d: (1x256x160x112xf32) <- (1x256x160x112xf32, 256x256x3x3xf32)
        conv2d_66 = paddle._C_ops.conv2d(
            relu_53, parameter_17, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del relu_53

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(parameter_16, full_int_array_1)
        del parameter_16

        # pd_op.add: (1x256x160x112xf32) <- (1x256x160x112xf32, 1x256x1x1xf32)
        add_41 = paddle._C_ops.add(conv2d_66, reshape_13)
        del conv2d_66

        # pd_op.group_norm: (1x256x160x112xf32, 1x32xf32, 1x32xf32) <- (1x256x160x112xf32, 256xf32, 256xf32)
        group_norm_15, group_norm_16, group_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_41, parameter_15, parameter_14, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_41

        # pd_op.relu: (1x256x160x112xf32) <- (1x256x160x112xf32)
        relu_55 = paddle._C_ops.relu(group_norm_15)
        del group_norm_15

        # pd_op.conv2d: (1x256x160x112xf32) <- (1x256x160x112xf32, 256x256x3x3xf32)
        conv2d_67 = paddle._C_ops.conv2d(
            relu_54, parameter_13, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del relu_54

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(parameter_12, full_int_array_1)
        del parameter_12

        # pd_op.add: (1x256x160x112xf32) <- (1x256x160x112xf32, 1x256x1x1xf32)
        add_42 = paddle._C_ops.add(conv2d_67, reshape_14)
        del conv2d_67

        # pd_op.group_norm: (1x256x160x112xf32, 1x32xf32, 1x32xf32) <- (1x256x160x112xf32, 256xf32, 256xf32)
        group_norm_18, group_norm_19, group_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_42, parameter_11, parameter_10, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_42

        # pd_op.relu: (1x256x160x112xf32) <- (1x256x160x112xf32)
        relu_56 = paddle._C_ops.relu(group_norm_18)
        del group_norm_18

        # pd_op.conv2d: (1x256x160x112xf32) <- (1x256x160x112xf32, 256x256x3x3xf32)
        conv2d_68 = paddle._C_ops.conv2d(
            relu_55, parameter_9, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del relu_55

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_15 = paddle._C_ops.reshape(parameter_8, full_int_array_1)
        del parameter_8

        # pd_op.add: (1x256x160x112xf32) <- (1x256x160x112xf32, 1x256x1x1xf32)
        add_43 = paddle._C_ops.add(conv2d_68, reshape_15)
        del conv2d_68

        # pd_op.group_norm: (1x256x160x112xf32, 1x32xf32, 1x32xf32) <- (1x256x160x112xf32, 256xf32, 256xf32)
        group_norm_21, group_norm_22, group_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_43, parameter_7, parameter_6, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_43

        # pd_op.relu: (1x256x160x112xf32) <- (1x256x160x112xf32)
        relu_57 = paddle._C_ops.relu(group_norm_21)
        del group_norm_21

        # pd_op.conv2d: (1x4x160x112xf32) <- (1x256x160x112xf32, 4x256x3x3xf32)
        conv2d_69 = paddle._C_ops.conv2d(
            relu_56, parameter_5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del relu_56

        # pd_op.reshape: (1x4x1x1xf32) <- (4xf32, 4xi64)
        reshape_16 = paddle._C_ops.reshape(parameter_4, full_int_array_1)
        del parameter_4

        # pd_op.add: (1x4x160x112xf32) <- (1x4x160x112xf32, 1x4x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_69, reshape_16)
        del conv2d_69

        # pd_op.conv2d: (1x4x160x112xf32) <- (1x256x160x112xf32, 4x256x3x3xf32)
        conv2d_70 = paddle._C_ops.conv2d(
            relu_57, parameter_3, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x4x1x1xf32) <- (4xf32, 4xi64)
        reshape_17 = paddle._C_ops.reshape(parameter_2, full_int_array_1)
        del parameter_2

        # pd_op.add: (1x4x160x112xf32) <- (1x4x160x112xf32, 1x4x1x1xf32)
        add_44 = paddle._C_ops.add(conv2d_70, reshape_17)
        del conv2d_70

        # pd_op.multiply: (1x4x160x112xf32) <- (1x4x160x112xf32, 1xf32)
        multiply_0 = paddle._C_ops.multiply(add_44, data_0)
        del add_44, data_0

        # pd_op.conv2d: (1x1x160x112xf32) <- (1x256x160x112xf32, 1x256x3x3xf32)
        conv2d_71 = paddle._C_ops.conv2d(
            relu_57, parameter_1, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del relu_57

        # pd_op.reshape: (1x1x1x1xf32) <- (1xf32, 4xi64)
        reshape_18 = paddle._C_ops.reshape(parameter_0, full_int_array_1)
        del full_int_array_1, parameter_0

        # pd_op.add: (1x1x160x112xf32) <- (1x1x160x112xf32, 1x1x1x1xf32)
        add_5 = paddle._C_ops.add(conv2d_71, reshape_18)
        del conv2d_71

        # pd_op.relu: (1x4x160x112xf32) <- (1x4x160x112xf32)
        relu_58 = paddle._C_ops.relu(multiply_0)
        del multiply_0

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("8"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x4x160x112xf32) <- (1x4x160x112xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(relu_58, full_0, float("0"), True)
        del full_0, relu_58

        # pd_op.conv2d: (1x256x80x56xf32) <- (1x256x80x56xf32, 256x256x3x3xf32)
        conv2d_72 = paddle._C_ops.conv2d(
            add_32, parameter_37, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (1x256x80x56xf32) <- (1x256x80x56xf32, 1x256x1x1xf32)
        add_45 = paddle._C_ops.add(conv2d_72, reshape_8)
        del conv2d_72

        # pd_op.group_norm: (1x256x80x56xf32, 1x32xf32, 1x32xf32) <- (1x256x80x56xf32, 256xf32, 256xf32)
        group_norm_24, group_norm_25, group_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_45, parameter_35, parameter_34, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_45

        # pd_op.relu: (1x256x80x56xf32) <- (1x256x80x56xf32)
        relu_59 = paddle._C_ops.relu(group_norm_24)
        del group_norm_24

        # pd_op.conv2d: (1x256x80x56xf32) <- (1x256x80x56xf32, 256x256x3x3xf32)
        conv2d_73 = paddle._C_ops.conv2d(
            add_32, parameter_33, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_32

        # pd_op.add: (1x256x80x56xf32) <- (1x256x80x56xf32, 1x256x1x1xf32)
        add_46 = paddle._C_ops.add(conv2d_73, reshape_9)
        del conv2d_73

        # pd_op.group_norm: (1x256x80x56xf32, 1x32xf32, 1x32xf32) <- (1x256x80x56xf32, 256xf32, 256xf32)
        group_norm_27, group_norm_28, group_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_46, parameter_31, parameter_30, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_46

        # pd_op.relu: (1x256x80x56xf32) <- (1x256x80x56xf32)
        relu_60 = paddle._C_ops.relu(group_norm_27)
        del group_norm_27

        # pd_op.conv2d: (1x256x80x56xf32) <- (1x256x80x56xf32, 256x256x3x3xf32)
        conv2d_74 = paddle._C_ops.conv2d(
            relu_59, parameter_29, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del relu_59

        # pd_op.add: (1x256x80x56xf32) <- (1x256x80x56xf32, 1x256x1x1xf32)
        add_47 = paddle._C_ops.add(conv2d_74, reshape_10)
        del conv2d_74

        # pd_op.group_norm: (1x256x80x56xf32, 1x32xf32, 1x32xf32) <- (1x256x80x56xf32, 256xf32, 256xf32)
        group_norm_30, group_norm_31, group_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_47, parameter_27, parameter_26, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_47

        # pd_op.relu: (1x256x80x56xf32) <- (1x256x80x56xf32)
        relu_61 = paddle._C_ops.relu(group_norm_30)
        del group_norm_30

        # pd_op.conv2d: (1x256x80x56xf32) <- (1x256x80x56xf32, 256x256x3x3xf32)
        conv2d_75 = paddle._C_ops.conv2d(
            relu_60, parameter_25, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del relu_60

        # pd_op.add: (1x256x80x56xf32) <- (1x256x80x56xf32, 1x256x1x1xf32)
        add_48 = paddle._C_ops.add(conv2d_75, reshape_11)
        del conv2d_75

        # pd_op.group_norm: (1x256x80x56xf32, 1x32xf32, 1x32xf32) <- (1x256x80x56xf32, 256xf32, 256xf32)
        group_norm_33, group_norm_34, group_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_48, parameter_23, parameter_22, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_48

        # pd_op.relu: (1x256x80x56xf32) <- (1x256x80x56xf32)
        relu_62 = paddle._C_ops.relu(group_norm_33)
        del group_norm_33

        # pd_op.conv2d: (1x256x80x56xf32) <- (1x256x80x56xf32, 256x256x3x3xf32)
        conv2d_76 = paddle._C_ops.conv2d(
            relu_61, parameter_21, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del relu_61

        # pd_op.add: (1x256x80x56xf32) <- (1x256x80x56xf32, 1x256x1x1xf32)
        add_49 = paddle._C_ops.add(conv2d_76, reshape_12)
        del conv2d_76

        # pd_op.group_norm: (1x256x80x56xf32, 1x32xf32, 1x32xf32) <- (1x256x80x56xf32, 256xf32, 256xf32)
        group_norm_36, group_norm_37, group_norm_38 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_49, parameter_19, parameter_18, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_49

        # pd_op.relu: (1x256x80x56xf32) <- (1x256x80x56xf32)
        relu_63 = paddle._C_ops.relu(group_norm_36)
        del group_norm_36

        # pd_op.conv2d: (1x256x80x56xf32) <- (1x256x80x56xf32, 256x256x3x3xf32)
        conv2d_77 = paddle._C_ops.conv2d(
            relu_62, parameter_17, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del relu_62

        # pd_op.add: (1x256x80x56xf32) <- (1x256x80x56xf32, 1x256x1x1xf32)
        add_50 = paddle._C_ops.add(conv2d_77, reshape_13)
        del conv2d_77

        # pd_op.group_norm: (1x256x80x56xf32, 1x32xf32, 1x32xf32) <- (1x256x80x56xf32, 256xf32, 256xf32)
        group_norm_39, group_norm_40, group_norm_41 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_50, parameter_15, parameter_14, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_50

        # pd_op.relu: (1x256x80x56xf32) <- (1x256x80x56xf32)
        relu_64 = paddle._C_ops.relu(group_norm_39)
        del group_norm_39

        # pd_op.conv2d: (1x256x80x56xf32) <- (1x256x80x56xf32, 256x256x3x3xf32)
        conv2d_78 = paddle._C_ops.conv2d(
            relu_63, parameter_13, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del relu_63

        # pd_op.add: (1x256x80x56xf32) <- (1x256x80x56xf32, 1x256x1x1xf32)
        add_51 = paddle._C_ops.add(conv2d_78, reshape_14)
        del conv2d_78

        # pd_op.group_norm: (1x256x80x56xf32, 1x32xf32, 1x32xf32) <- (1x256x80x56xf32, 256xf32, 256xf32)
        group_norm_42, group_norm_43, group_norm_44 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_51, parameter_11, parameter_10, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_51

        # pd_op.relu: (1x256x80x56xf32) <- (1x256x80x56xf32)
        relu_65 = paddle._C_ops.relu(group_norm_42)
        del group_norm_42

        # pd_op.conv2d: (1x256x80x56xf32) <- (1x256x80x56xf32, 256x256x3x3xf32)
        conv2d_79 = paddle._C_ops.conv2d(
            relu_64, parameter_9, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del relu_64

        # pd_op.add: (1x256x80x56xf32) <- (1x256x80x56xf32, 1x256x1x1xf32)
        add_52 = paddle._C_ops.add(conv2d_79, reshape_15)
        del conv2d_79

        # pd_op.group_norm: (1x256x80x56xf32, 1x32xf32, 1x32xf32) <- (1x256x80x56xf32, 256xf32, 256xf32)
        group_norm_45, group_norm_46, group_norm_47 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_52, parameter_7, parameter_6, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_52

        # pd_op.relu: (1x256x80x56xf32) <- (1x256x80x56xf32)
        relu_66 = paddle._C_ops.relu(group_norm_45)
        del group_norm_45

        # pd_op.conv2d: (1x4x80x56xf32) <- (1x256x80x56xf32, 4x256x3x3xf32)
        conv2d_80 = paddle._C_ops.conv2d(
            relu_65, parameter_5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del relu_65

        # pd_op.add: (1x4x80x56xf32) <- (1x4x80x56xf32, 1x4x1x1xf32)
        add_1 = paddle._C_ops.add(conv2d_80, reshape_16)
        del conv2d_80

        # pd_op.conv2d: (1x4x80x56xf32) <- (1x256x80x56xf32, 4x256x3x3xf32)
        conv2d_81 = paddle._C_ops.conv2d(
            relu_66, parameter_3, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (1x4x80x56xf32) <- (1x4x80x56xf32, 1x4x1x1xf32)
        add_53 = paddle._C_ops.add(conv2d_81, reshape_17)
        del conv2d_81

        # pd_op.multiply: (1x4x80x56xf32) <- (1x4x80x56xf32, 1xf32)
        multiply_1 = paddle._C_ops.multiply(add_53, data_1)
        del add_53, data_1

        # pd_op.conv2d: (1x1x80x56xf32) <- (1x256x80x56xf32, 1x256x3x3xf32)
        conv2d_82 = paddle._C_ops.conv2d(
            relu_66, parameter_1, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del relu_66

        # pd_op.add: (1x1x80x56xf32) <- (1x1x80x56xf32, 1x1x1x1xf32)
        add_6 = paddle._C_ops.add(conv2d_82, reshape_18)
        del conv2d_82

        # pd_op.relu: (1x4x80x56xf32) <- (1x4x80x56xf32)
        relu_67 = paddle._C_ops.relu(multiply_1)
        del multiply_1

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("16"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x4x80x56xf32) <- (1x4x80x56xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(relu_67, full_1, float("0"), True)
        del full_1, relu_67

        # pd_op.conv2d: (1x256x40x28xf32) <- (1x256x40x28xf32, 256x256x3x3xf32)
        conv2d_83 = paddle._C_ops.conv2d(
            add_33, parameter_37, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (1x256x40x28xf32) <- (1x256x40x28xf32, 1x256x1x1xf32)
        add_54 = paddle._C_ops.add(conv2d_83, reshape_8)
        del conv2d_83

        # pd_op.group_norm: (1x256x40x28xf32, 1x32xf32, 1x32xf32) <- (1x256x40x28xf32, 256xf32, 256xf32)
        group_norm_48, group_norm_49, group_norm_50 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_54, parameter_35, parameter_34, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_54

        # pd_op.relu: (1x256x40x28xf32) <- (1x256x40x28xf32)
        relu_68 = paddle._C_ops.relu(group_norm_48)
        del group_norm_48

        # pd_op.conv2d: (1x256x40x28xf32) <- (1x256x40x28xf32, 256x256x3x3xf32)
        conv2d_84 = paddle._C_ops.conv2d(
            add_33, parameter_33, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_33

        # pd_op.add: (1x256x40x28xf32) <- (1x256x40x28xf32, 1x256x1x1xf32)
        add_55 = paddle._C_ops.add(conv2d_84, reshape_9)
        del conv2d_84

        # pd_op.group_norm: (1x256x40x28xf32, 1x32xf32, 1x32xf32) <- (1x256x40x28xf32, 256xf32, 256xf32)
        group_norm_51, group_norm_52, group_norm_53 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_55, parameter_31, parameter_30, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_55

        # pd_op.relu: (1x256x40x28xf32) <- (1x256x40x28xf32)
        relu_69 = paddle._C_ops.relu(group_norm_51)
        del group_norm_51

        # pd_op.conv2d: (1x256x40x28xf32) <- (1x256x40x28xf32, 256x256x3x3xf32)
        conv2d_85 = paddle._C_ops.conv2d(
            relu_68, parameter_29, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del relu_68

        # pd_op.add: (1x256x40x28xf32) <- (1x256x40x28xf32, 1x256x1x1xf32)
        add_56 = paddle._C_ops.add(conv2d_85, reshape_10)
        del conv2d_85

        # pd_op.group_norm: (1x256x40x28xf32, 1x32xf32, 1x32xf32) <- (1x256x40x28xf32, 256xf32, 256xf32)
        group_norm_54, group_norm_55, group_norm_56 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_56, parameter_27, parameter_26, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_56

        # pd_op.relu: (1x256x40x28xf32) <- (1x256x40x28xf32)
        relu_70 = paddle._C_ops.relu(group_norm_54)
        del group_norm_54

        # pd_op.conv2d: (1x256x40x28xf32) <- (1x256x40x28xf32, 256x256x3x3xf32)
        conv2d_86 = paddle._C_ops.conv2d(
            relu_69, parameter_25, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del relu_69

        # pd_op.add: (1x256x40x28xf32) <- (1x256x40x28xf32, 1x256x1x1xf32)
        add_57 = paddle._C_ops.add(conv2d_86, reshape_11)
        del conv2d_86

        # pd_op.group_norm: (1x256x40x28xf32, 1x32xf32, 1x32xf32) <- (1x256x40x28xf32, 256xf32, 256xf32)
        group_norm_57, group_norm_58, group_norm_59 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_57, parameter_23, parameter_22, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_57

        # pd_op.relu: (1x256x40x28xf32) <- (1x256x40x28xf32)
        relu_71 = paddle._C_ops.relu(group_norm_57)
        del group_norm_57

        # pd_op.conv2d: (1x256x40x28xf32) <- (1x256x40x28xf32, 256x256x3x3xf32)
        conv2d_87 = paddle._C_ops.conv2d(
            relu_70, parameter_21, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del relu_70

        # pd_op.add: (1x256x40x28xf32) <- (1x256x40x28xf32, 1x256x1x1xf32)
        add_58 = paddle._C_ops.add(conv2d_87, reshape_12)
        del conv2d_87

        # pd_op.group_norm: (1x256x40x28xf32, 1x32xf32, 1x32xf32) <- (1x256x40x28xf32, 256xf32, 256xf32)
        group_norm_60, group_norm_61, group_norm_62 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_58, parameter_19, parameter_18, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_58

        # pd_op.relu: (1x256x40x28xf32) <- (1x256x40x28xf32)
        relu_72 = paddle._C_ops.relu(group_norm_60)
        del group_norm_60

        # pd_op.conv2d: (1x256x40x28xf32) <- (1x256x40x28xf32, 256x256x3x3xf32)
        conv2d_88 = paddle._C_ops.conv2d(
            relu_71, parameter_17, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del relu_71

        # pd_op.add: (1x256x40x28xf32) <- (1x256x40x28xf32, 1x256x1x1xf32)
        add_59 = paddle._C_ops.add(conv2d_88, reshape_13)
        del conv2d_88

        # pd_op.group_norm: (1x256x40x28xf32, 1x32xf32, 1x32xf32) <- (1x256x40x28xf32, 256xf32, 256xf32)
        group_norm_63, group_norm_64, group_norm_65 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_59, parameter_15, parameter_14, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_59

        # pd_op.relu: (1x256x40x28xf32) <- (1x256x40x28xf32)
        relu_73 = paddle._C_ops.relu(group_norm_63)
        del group_norm_63

        # pd_op.conv2d: (1x256x40x28xf32) <- (1x256x40x28xf32, 256x256x3x3xf32)
        conv2d_89 = paddle._C_ops.conv2d(
            relu_72, parameter_13, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del relu_72

        # pd_op.add: (1x256x40x28xf32) <- (1x256x40x28xf32, 1x256x1x1xf32)
        add_60 = paddle._C_ops.add(conv2d_89, reshape_14)
        del conv2d_89

        # pd_op.group_norm: (1x256x40x28xf32, 1x32xf32, 1x32xf32) <- (1x256x40x28xf32, 256xf32, 256xf32)
        group_norm_66, group_norm_67, group_norm_68 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_60, parameter_11, parameter_10, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_60

        # pd_op.relu: (1x256x40x28xf32) <- (1x256x40x28xf32)
        relu_74 = paddle._C_ops.relu(group_norm_66)
        del group_norm_66

        # pd_op.conv2d: (1x256x40x28xf32) <- (1x256x40x28xf32, 256x256x3x3xf32)
        conv2d_90 = paddle._C_ops.conv2d(
            relu_73, parameter_9, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del relu_73

        # pd_op.add: (1x256x40x28xf32) <- (1x256x40x28xf32, 1x256x1x1xf32)
        add_61 = paddle._C_ops.add(conv2d_90, reshape_15)
        del conv2d_90

        # pd_op.group_norm: (1x256x40x28xf32, 1x32xf32, 1x32xf32) <- (1x256x40x28xf32, 256xf32, 256xf32)
        group_norm_69, group_norm_70, group_norm_71 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_61, parameter_7, parameter_6, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_61

        # pd_op.relu: (1x256x40x28xf32) <- (1x256x40x28xf32)
        relu_75 = paddle._C_ops.relu(group_norm_69)
        del group_norm_69

        # pd_op.conv2d: (1x4x40x28xf32) <- (1x256x40x28xf32, 4x256x3x3xf32)
        conv2d_91 = paddle._C_ops.conv2d(
            relu_74, parameter_5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del relu_74

        # pd_op.add: (1x4x40x28xf32) <- (1x4x40x28xf32, 1x4x1x1xf32)
        add_2 = paddle._C_ops.add(conv2d_91, reshape_16)
        del conv2d_91

        # pd_op.conv2d: (1x4x40x28xf32) <- (1x256x40x28xf32, 4x256x3x3xf32)
        conv2d_92 = paddle._C_ops.conv2d(
            relu_75, parameter_3, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (1x4x40x28xf32) <- (1x4x40x28xf32, 1x4x1x1xf32)
        add_62 = paddle._C_ops.add(conv2d_92, reshape_17)
        del conv2d_92

        # pd_op.multiply: (1x4x40x28xf32) <- (1x4x40x28xf32, 1xf32)
        multiply_2 = paddle._C_ops.multiply(add_62, data_2)
        del add_62, data_2

        # pd_op.conv2d: (1x1x40x28xf32) <- (1x256x40x28xf32, 1x256x3x3xf32)
        conv2d_93 = paddle._C_ops.conv2d(
            relu_75, parameter_1, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del relu_75

        # pd_op.add: (1x1x40x28xf32) <- (1x1x40x28xf32, 1x1x1x1xf32)
        add_7 = paddle._C_ops.add(conv2d_93, reshape_18)
        del conv2d_93

        # pd_op.relu: (1x4x40x28xf32) <- (1x4x40x28xf32)
        relu_76 = paddle._C_ops.relu(multiply_2)
        del multiply_2

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("32"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x4x40x28xf32) <- (1x4x40x28xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(relu_76, full_2, float("0"), True)
        del full_2, relu_76

        # pd_op.conv2d: (1x256x20x14xf32) <- (1x256x20x14xf32, 256x256x3x3xf32)
        conv2d_94 = paddle._C_ops.conv2d(
            add_34, parameter_37, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (1x256x20x14xf32) <- (1x256x20x14xf32, 1x256x1x1xf32)
        add_63 = paddle._C_ops.add(conv2d_94, reshape_8)
        del conv2d_94

        # pd_op.group_norm: (1x256x20x14xf32, 1x32xf32, 1x32xf32) <- (1x256x20x14xf32, 256xf32, 256xf32)
        group_norm_72, group_norm_73, group_norm_74 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_63, parameter_35, parameter_34, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_63

        # pd_op.relu: (1x256x20x14xf32) <- (1x256x20x14xf32)
        relu_77 = paddle._C_ops.relu(group_norm_72)
        del group_norm_72

        # pd_op.conv2d: (1x256x20x14xf32) <- (1x256x20x14xf32, 256x256x3x3xf32)
        conv2d_95 = paddle._C_ops.conv2d(
            add_34, parameter_33, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_34

        # pd_op.add: (1x256x20x14xf32) <- (1x256x20x14xf32, 1x256x1x1xf32)
        add_64 = paddle._C_ops.add(conv2d_95, reshape_9)
        del conv2d_95

        # pd_op.group_norm: (1x256x20x14xf32, 1x32xf32, 1x32xf32) <- (1x256x20x14xf32, 256xf32, 256xf32)
        group_norm_75, group_norm_76, group_norm_77 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_64, parameter_31, parameter_30, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_64

        # pd_op.relu: (1x256x20x14xf32) <- (1x256x20x14xf32)
        relu_78 = paddle._C_ops.relu(group_norm_75)
        del group_norm_75

        # pd_op.conv2d: (1x256x20x14xf32) <- (1x256x20x14xf32, 256x256x3x3xf32)
        conv2d_96 = paddle._C_ops.conv2d(
            relu_77, parameter_29, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del relu_77

        # pd_op.add: (1x256x20x14xf32) <- (1x256x20x14xf32, 1x256x1x1xf32)
        add_65 = paddle._C_ops.add(conv2d_96, reshape_10)
        del conv2d_96

        # pd_op.group_norm: (1x256x20x14xf32, 1x32xf32, 1x32xf32) <- (1x256x20x14xf32, 256xf32, 256xf32)
        group_norm_78, group_norm_79, group_norm_80 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_65, parameter_27, parameter_26, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_65

        # pd_op.relu: (1x256x20x14xf32) <- (1x256x20x14xf32)
        relu_79 = paddle._C_ops.relu(group_norm_78)
        del group_norm_78

        # pd_op.conv2d: (1x256x20x14xf32) <- (1x256x20x14xf32, 256x256x3x3xf32)
        conv2d_97 = paddle._C_ops.conv2d(
            relu_78, parameter_25, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del relu_78

        # pd_op.add: (1x256x20x14xf32) <- (1x256x20x14xf32, 1x256x1x1xf32)
        add_66 = paddle._C_ops.add(conv2d_97, reshape_11)
        del conv2d_97

        # pd_op.group_norm: (1x256x20x14xf32, 1x32xf32, 1x32xf32) <- (1x256x20x14xf32, 256xf32, 256xf32)
        group_norm_81, group_norm_82, group_norm_83 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_66, parameter_23, parameter_22, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_66

        # pd_op.relu: (1x256x20x14xf32) <- (1x256x20x14xf32)
        relu_80 = paddle._C_ops.relu(group_norm_81)
        del group_norm_81

        # pd_op.conv2d: (1x256x20x14xf32) <- (1x256x20x14xf32, 256x256x3x3xf32)
        conv2d_98 = paddle._C_ops.conv2d(
            relu_79, parameter_21, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del relu_79

        # pd_op.add: (1x256x20x14xf32) <- (1x256x20x14xf32, 1x256x1x1xf32)
        add_67 = paddle._C_ops.add(conv2d_98, reshape_12)
        del conv2d_98

        # pd_op.group_norm: (1x256x20x14xf32, 1x32xf32, 1x32xf32) <- (1x256x20x14xf32, 256xf32, 256xf32)
        group_norm_84, group_norm_85, group_norm_86 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_67, parameter_19, parameter_18, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_67

        # pd_op.relu: (1x256x20x14xf32) <- (1x256x20x14xf32)
        relu_81 = paddle._C_ops.relu(group_norm_84)
        del group_norm_84

        # pd_op.conv2d: (1x256x20x14xf32) <- (1x256x20x14xf32, 256x256x3x3xf32)
        conv2d_99 = paddle._C_ops.conv2d(
            relu_80, parameter_17, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del relu_80

        # pd_op.add: (1x256x20x14xf32) <- (1x256x20x14xf32, 1x256x1x1xf32)
        add_68 = paddle._C_ops.add(conv2d_99, reshape_13)
        del conv2d_99

        # pd_op.group_norm: (1x256x20x14xf32, 1x32xf32, 1x32xf32) <- (1x256x20x14xf32, 256xf32, 256xf32)
        group_norm_87, group_norm_88, group_norm_89 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_68, parameter_15, parameter_14, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_68

        # pd_op.relu: (1x256x20x14xf32) <- (1x256x20x14xf32)
        relu_82 = paddle._C_ops.relu(group_norm_87)
        del group_norm_87

        # pd_op.conv2d: (1x256x20x14xf32) <- (1x256x20x14xf32, 256x256x3x3xf32)
        conv2d_100 = paddle._C_ops.conv2d(
            relu_81, parameter_13, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del relu_81

        # pd_op.add: (1x256x20x14xf32) <- (1x256x20x14xf32, 1x256x1x1xf32)
        add_69 = paddle._C_ops.add(conv2d_100, reshape_14)
        del conv2d_100

        # pd_op.group_norm: (1x256x20x14xf32, 1x32xf32, 1x32xf32) <- (1x256x20x14xf32, 256xf32, 256xf32)
        group_norm_90, group_norm_91, group_norm_92 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_69, parameter_11, parameter_10, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_69

        # pd_op.relu: (1x256x20x14xf32) <- (1x256x20x14xf32)
        relu_83 = paddle._C_ops.relu(group_norm_90)
        del group_norm_90

        # pd_op.conv2d: (1x256x20x14xf32) <- (1x256x20x14xf32, 256x256x3x3xf32)
        conv2d_101 = paddle._C_ops.conv2d(
            relu_82, parameter_9, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del relu_82

        # pd_op.add: (1x256x20x14xf32) <- (1x256x20x14xf32, 1x256x1x1xf32)
        add_70 = paddle._C_ops.add(conv2d_101, reshape_15)
        del conv2d_101

        # pd_op.group_norm: (1x256x20x14xf32, 1x32xf32, 1x32xf32) <- (1x256x20x14xf32, 256xf32, 256xf32)
        group_norm_93, group_norm_94, group_norm_95 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_70, parameter_7, parameter_6, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_70

        # pd_op.relu: (1x256x20x14xf32) <- (1x256x20x14xf32)
        relu_84 = paddle._C_ops.relu(group_norm_93)
        del group_norm_93

        # pd_op.conv2d: (1x4x20x14xf32) <- (1x256x20x14xf32, 4x256x3x3xf32)
        conv2d_102 = paddle._C_ops.conv2d(
            relu_83, parameter_5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del relu_83

        # pd_op.add: (1x4x20x14xf32) <- (1x4x20x14xf32, 1x4x1x1xf32)
        add_3 = paddle._C_ops.add(conv2d_102, reshape_16)
        del conv2d_102

        # pd_op.conv2d: (1x4x20x14xf32) <- (1x256x20x14xf32, 4x256x3x3xf32)
        conv2d_103 = paddle._C_ops.conv2d(
            relu_84, parameter_3, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (1x4x20x14xf32) <- (1x4x20x14xf32, 1x4x1x1xf32)
        add_71 = paddle._C_ops.add(conv2d_103, reshape_17)
        del conv2d_103

        # pd_op.multiply: (1x4x20x14xf32) <- (1x4x20x14xf32, 1xf32)
        multiply_3 = paddle._C_ops.multiply(add_71, data_3)
        del add_71, data_3

        # pd_op.conv2d: (1x1x20x14xf32) <- (1x256x20x14xf32, 1x256x3x3xf32)
        conv2d_104 = paddle._C_ops.conv2d(
            relu_84, parameter_1, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del relu_84

        # pd_op.add: (1x1x20x14xf32) <- (1x1x20x14xf32, 1x1x1x1xf32)
        add_8 = paddle._C_ops.add(conv2d_104, reshape_18)
        del conv2d_104

        # pd_op.relu: (1x4x20x14xf32) <- (1x4x20x14xf32)
        relu_85 = paddle._C_ops.relu(multiply_3)
        del multiply_3

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("64"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x4x20x14xf32) <- (1x4x20x14xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(relu_85, full_3, float("0"), True)
        del full_3, relu_85

        # pd_op.conv2d: (1x256x10x7xf32) <- (1x256x10x7xf32, 256x256x3x3xf32)
        conv2d_105 = paddle._C_ops.conv2d(
            add_35, parameter_37, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_37

        # pd_op.add: (1x256x10x7xf32) <- (1x256x10x7xf32, 1x256x1x1xf32)
        add_72 = paddle._C_ops.add(conv2d_105, reshape_8)
        del conv2d_105, reshape_8

        # pd_op.group_norm: (1x256x10x7xf32, 1x32xf32, 1x32xf32) <- (1x256x10x7xf32, 256xf32, 256xf32)
        group_norm_96, group_norm_97, group_norm_98 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_72, parameter_35, parameter_34, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_72, parameter_34, parameter_35

        # pd_op.relu: (1x256x10x7xf32) <- (1x256x10x7xf32)
        relu_86 = paddle._C_ops.relu(group_norm_96)
        del group_norm_96

        # pd_op.conv2d: (1x256x10x7xf32) <- (1x256x10x7xf32, 256x256x3x3xf32)
        conv2d_106 = paddle._C_ops.conv2d(
            add_35, parameter_33, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_35, parameter_33

        # pd_op.add: (1x256x10x7xf32) <- (1x256x10x7xf32, 1x256x1x1xf32)
        add_73 = paddle._C_ops.add(conv2d_106, reshape_9)
        del conv2d_106, reshape_9

        # pd_op.group_norm: (1x256x10x7xf32, 1x32xf32, 1x32xf32) <- (1x256x10x7xf32, 256xf32, 256xf32)
        group_norm_99, group_norm_100, group_norm_101 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_73, parameter_31, parameter_30, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_73, parameter_30, parameter_31

        # pd_op.relu: (1x256x10x7xf32) <- (1x256x10x7xf32)
        relu_87 = paddle._C_ops.relu(group_norm_99)
        del group_norm_99

        # pd_op.conv2d: (1x256x10x7xf32) <- (1x256x10x7xf32, 256x256x3x3xf32)
        conv2d_107 = paddle._C_ops.conv2d(
            relu_86, parameter_29, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_29, relu_86

        # pd_op.add: (1x256x10x7xf32) <- (1x256x10x7xf32, 1x256x1x1xf32)
        add_74 = paddle._C_ops.add(conv2d_107, reshape_10)
        del conv2d_107, reshape_10

        # pd_op.group_norm: (1x256x10x7xf32, 1x32xf32, 1x32xf32) <- (1x256x10x7xf32, 256xf32, 256xf32)
        group_norm_102, group_norm_103, group_norm_104 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_74, parameter_27, parameter_26, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_74, parameter_26, parameter_27

        # pd_op.relu: (1x256x10x7xf32) <- (1x256x10x7xf32)
        relu_88 = paddle._C_ops.relu(group_norm_102)
        del group_norm_102

        # pd_op.conv2d: (1x256x10x7xf32) <- (1x256x10x7xf32, 256x256x3x3xf32)
        conv2d_108 = paddle._C_ops.conv2d(
            relu_87, parameter_25, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_25, relu_87

        # pd_op.add: (1x256x10x7xf32) <- (1x256x10x7xf32, 1x256x1x1xf32)
        add_75 = paddle._C_ops.add(conv2d_108, reshape_11)
        del conv2d_108, reshape_11

        # pd_op.group_norm: (1x256x10x7xf32, 1x32xf32, 1x32xf32) <- (1x256x10x7xf32, 256xf32, 256xf32)
        group_norm_105, group_norm_106, group_norm_107 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_75, parameter_23, parameter_22, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_75, parameter_22, parameter_23

        # pd_op.relu: (1x256x10x7xf32) <- (1x256x10x7xf32)
        relu_89 = paddle._C_ops.relu(group_norm_105)
        del group_norm_105

        # pd_op.conv2d: (1x256x10x7xf32) <- (1x256x10x7xf32, 256x256x3x3xf32)
        conv2d_109 = paddle._C_ops.conv2d(
            relu_88, parameter_21, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_21, relu_88

        # pd_op.add: (1x256x10x7xf32) <- (1x256x10x7xf32, 1x256x1x1xf32)
        add_76 = paddle._C_ops.add(conv2d_109, reshape_12)
        del conv2d_109, reshape_12

        # pd_op.group_norm: (1x256x10x7xf32, 1x32xf32, 1x32xf32) <- (1x256x10x7xf32, 256xf32, 256xf32)
        group_norm_108, group_norm_109, group_norm_110 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_76, parameter_19, parameter_18, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_76, parameter_18, parameter_19

        # pd_op.relu: (1x256x10x7xf32) <- (1x256x10x7xf32)
        relu_90 = paddle._C_ops.relu(group_norm_108)
        del group_norm_108

        # pd_op.conv2d: (1x256x10x7xf32) <- (1x256x10x7xf32, 256x256x3x3xf32)
        conv2d_110 = paddle._C_ops.conv2d(
            relu_89, parameter_17, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_17, relu_89

        # pd_op.add: (1x256x10x7xf32) <- (1x256x10x7xf32, 1x256x1x1xf32)
        add_77 = paddle._C_ops.add(conv2d_110, reshape_13)
        del conv2d_110, reshape_13

        # pd_op.group_norm: (1x256x10x7xf32, 1x32xf32, 1x32xf32) <- (1x256x10x7xf32, 256xf32, 256xf32)
        group_norm_111, group_norm_112, group_norm_113 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_77, parameter_15, parameter_14, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_77, parameter_14, parameter_15

        # pd_op.relu: (1x256x10x7xf32) <- (1x256x10x7xf32)
        relu_91 = paddle._C_ops.relu(group_norm_111)
        del group_norm_111

        # pd_op.conv2d: (1x256x10x7xf32) <- (1x256x10x7xf32, 256x256x3x3xf32)
        conv2d_111 = paddle._C_ops.conv2d(
            relu_90, parameter_13, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_13, relu_90

        # pd_op.add: (1x256x10x7xf32) <- (1x256x10x7xf32, 1x256x1x1xf32)
        add_78 = paddle._C_ops.add(conv2d_111, reshape_14)
        del conv2d_111, reshape_14

        # pd_op.group_norm: (1x256x10x7xf32, 1x32xf32, 1x32xf32) <- (1x256x10x7xf32, 256xf32, 256xf32)
        group_norm_114, group_norm_115, group_norm_116 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_78, parameter_11, parameter_10, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_78, parameter_10, parameter_11

        # pd_op.relu: (1x256x10x7xf32) <- (1x256x10x7xf32)
        relu_92 = paddle._C_ops.relu(group_norm_114)
        del group_norm_114

        # pd_op.conv2d: (1x256x10x7xf32) <- (1x256x10x7xf32, 256x256x3x3xf32)
        conv2d_112 = paddle._C_ops.conv2d(
            relu_91, parameter_9, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_9, relu_91

        # pd_op.add: (1x256x10x7xf32) <- (1x256x10x7xf32, 1x256x1x1xf32)
        add_79 = paddle._C_ops.add(conv2d_112, reshape_15)
        del conv2d_112, reshape_15

        # pd_op.group_norm: (1x256x10x7xf32, 1x32xf32, 1x32xf32) <- (1x256x10x7xf32, 256xf32, 256xf32)
        group_norm_117, group_norm_118, group_norm_119 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                add_79, parameter_7, parameter_6, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_79, parameter_6, parameter_7

        # pd_op.relu: (1x256x10x7xf32) <- (1x256x10x7xf32)
        relu_93 = paddle._C_ops.relu(group_norm_117)
        del group_norm_117

        # pd_op.conv2d: (1x4x10x7xf32) <- (1x256x10x7xf32, 4x256x3x3xf32)
        conv2d_113 = paddle._C_ops.conv2d(
            relu_92, parameter_5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_5, relu_92

        # pd_op.add: (1x4x10x7xf32) <- (1x4x10x7xf32, 1x4x1x1xf32)
        add_4 = paddle._C_ops.add(conv2d_113, reshape_16)
        del conv2d_113, reshape_16

        # pd_op.conv2d: (1x4x10x7xf32) <- (1x256x10x7xf32, 4x256x3x3xf32)
        conv2d_114 = paddle._C_ops.conv2d(
            relu_93, parameter_3, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_3

        # pd_op.add: (1x4x10x7xf32) <- (1x4x10x7xf32, 1x4x1x1xf32)
        add_80 = paddle._C_ops.add(conv2d_114, reshape_17)
        del conv2d_114, reshape_17

        # pd_op.multiply: (1x4x10x7xf32) <- (1x4x10x7xf32, 1xf32)
        multiply_4 = paddle._C_ops.multiply(add_80, data_4)
        del add_80, data_4

        # pd_op.conv2d: (1x1x10x7xf32) <- (1x256x10x7xf32, 1x256x3x3xf32)
        conv2d_115 = paddle._C_ops.conv2d(
            relu_93, parameter_1, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_1, relu_93

        # pd_op.add: (1x1x10x7xf32) <- (1x1x10x7xf32, 1x1x1x1xf32)
        add_9 = paddle._C_ops.add(conv2d_115, reshape_18)
        del conv2d_115, reshape_18

        # pd_op.relu: (1x4x10x7xf32) <- (1x4x10x7xf32)
        relu_94 = paddle._C_ops.relu(multiply_4)
        del multiply_4

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("128"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x4x10x7xf32) <- (1x4x10x7xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(relu_94, full_4, float("0"), True)
        del full_4, relu_94

        # pd_op.full: (1xf64) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("896"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("8"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (112xi64) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_5, full_6, full_7, dtype="int64")

        # pd_op.full: (1xf64) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("1280"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (160xi64) <- (1xf64, 1xf64, 1xf64)
        arange_1 = paddle.arange(full_5, full_8, full_7, dtype="int64")
        del full_7

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.unsqueeze: (1x112xi64) <- (112xi64, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(arange_0, full_int_array_2)
        del arange_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.unsqueeze: (160x1xi64) <- (160xi64, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(arange_1, full_int_array_3)
        del arange_1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [160, 112]

        # pd_op.expand: (160x112xi64) <- (1x112xi64, 2xi64)
        expand_0 = paddle._C_ops.expand(unsqueeze_0, full_int_array_4)
        del unsqueeze_0

        # pd_op.expand: (160x112xi64) <- (160x1xi64, 2xi64)
        expand_1 = paddle._C_ops.expand(unsqueeze_1, full_int_array_4)
        del full_int_array_4, unsqueeze_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [-1]

        # pd_op.reshape: (17920xi64) <- (160x112xi64, 1xi64)
        reshape_19 = paddle._C_ops.reshape(expand_0, full_int_array_5)
        del expand_0

        # pd_op.reshape: (17920xi64) <- (160x112xi64, 1xi64)
        reshape_20 = paddle._C_ops.reshape(expand_1, full_int_array_5)
        del expand_1

        # builtin.combine: ([17920xi64, 17920xi64]) <- (17920xi64, 17920xi64)
        combine_0 = [reshape_19, reshape_20]
        del reshape_19, reshape_20

        # pd_op.stack: (17920x2xi64) <- ([17920xi64, 17920xi64])
        stack_0 = paddle._C_ops.stack(combine_0, -1)
        del combine_0

        # pd_op.cast: (17920x2xf32) <- (17920x2xi64)
        cast_0 = paddle._C_ops.cast(stack_0, paddle.float32)
        del stack_0

        # pd_op.full: (1xf32) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (17920x2xf32) <- (17920x2xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(cast_0, full_9, float("4"), True)
        del cast_0

        # pd_op.full: (1xf64) <- ()
        full_10 = paddle._C_ops.full(
            [1], float("16"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (56xi64) <- (1xf64, 1xf64, 1xf64)
        arange_2 = paddle.arange(full_5, full_6, full_10, dtype="int64")

        # pd_op.arange: (80xi64) <- (1xf64, 1xf64, 1xf64)
        arange_3 = paddle.arange(full_5, full_8, full_10, dtype="int64")
        del full_10

        # pd_op.unsqueeze: (1x56xi64) <- (56xi64, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(arange_2, full_int_array_2)
        del arange_2

        # pd_op.unsqueeze: (80x1xi64) <- (80xi64, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(arange_3, full_int_array_3)
        del arange_3

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_6 = [80, 56]

        # pd_op.expand: (80x56xi64) <- (1x56xi64, 2xi64)
        expand_2 = paddle._C_ops.expand(unsqueeze_2, full_int_array_6)
        del unsqueeze_2

        # pd_op.expand: (80x56xi64) <- (80x1xi64, 2xi64)
        expand_3 = paddle._C_ops.expand(unsqueeze_3, full_int_array_6)
        del full_int_array_6, unsqueeze_3

        # pd_op.reshape: (4480xi64) <- (80x56xi64, 1xi64)
        reshape_21 = paddle._C_ops.reshape(expand_2, full_int_array_5)
        del expand_2

        # pd_op.reshape: (4480xi64) <- (80x56xi64, 1xi64)
        reshape_22 = paddle._C_ops.reshape(expand_3, full_int_array_5)
        del expand_3

        # builtin.combine: ([4480xi64, 4480xi64]) <- (4480xi64, 4480xi64)
        combine_1 = [reshape_21, reshape_22]
        del reshape_21, reshape_22

        # pd_op.stack: (4480x2xi64) <- ([4480xi64, 4480xi64])
        stack_1 = paddle._C_ops.stack(combine_1, -1)
        del combine_1

        # pd_op.cast: (4480x2xf32) <- (4480x2xi64)
        cast_1 = paddle._C_ops.cast(stack_1, paddle.float32)
        del stack_1

        # pd_op.scale: (4480x2xf32) <- (4480x2xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(cast_1, full_9, float("8"), True)
        del cast_1

        # pd_op.full: (1xf64) <- ()
        full_11 = paddle._C_ops.full(
            [1], float("32"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (28xi64) <- (1xf64, 1xf64, 1xf64)
        arange_4 = paddle.arange(full_5, full_6, full_11, dtype="int64")

        # pd_op.arange: (40xi64) <- (1xf64, 1xf64, 1xf64)
        arange_5 = paddle.arange(full_5, full_8, full_11, dtype="int64")
        del full_11

        # pd_op.unsqueeze: (1x28xi64) <- (28xi64, 1xi64)
        unsqueeze_4 = paddle._C_ops.unsqueeze(arange_4, full_int_array_2)
        del arange_4

        # pd_op.unsqueeze: (40x1xi64) <- (40xi64, 1xi64)
        unsqueeze_5 = paddle._C_ops.unsqueeze(arange_5, full_int_array_3)
        del arange_5

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_7 = [40, 28]

        # pd_op.expand: (40x28xi64) <- (1x28xi64, 2xi64)
        expand_4 = paddle._C_ops.expand(unsqueeze_4, full_int_array_7)
        del unsqueeze_4

        # pd_op.expand: (40x28xi64) <- (40x1xi64, 2xi64)
        expand_5 = paddle._C_ops.expand(unsqueeze_5, full_int_array_7)
        del full_int_array_7, unsqueeze_5

        # pd_op.reshape: (1120xi64) <- (40x28xi64, 1xi64)
        reshape_23 = paddle._C_ops.reshape(expand_4, full_int_array_5)
        del expand_4

        # pd_op.reshape: (1120xi64) <- (40x28xi64, 1xi64)
        reshape_24 = paddle._C_ops.reshape(expand_5, full_int_array_5)
        del expand_5

        # builtin.combine: ([1120xi64, 1120xi64]) <- (1120xi64, 1120xi64)
        combine_2 = [reshape_23, reshape_24]
        del reshape_23, reshape_24

        # pd_op.stack: (1120x2xi64) <- ([1120xi64, 1120xi64])
        stack_2 = paddle._C_ops.stack(combine_2, -1)
        del combine_2

        # pd_op.cast: (1120x2xf32) <- (1120x2xi64)
        cast_2 = paddle._C_ops.cast(stack_2, paddle.float32)
        del stack_2

        # pd_op.scale: (1120x2xf32) <- (1120x2xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(cast_2, full_9, float("16"), True)
        del cast_2

        # pd_op.full: (1xf64) <- ()
        full_12 = paddle._C_ops.full(
            [1], float("64"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (14xi64) <- (1xf64, 1xf64, 1xf64)
        arange_6 = paddle.arange(full_5, full_6, full_12, dtype="int64")

        # pd_op.arange: (20xi64) <- (1xf64, 1xf64, 1xf64)
        arange_7 = paddle.arange(full_5, full_8, full_12, dtype="int64")
        del full_12

        # pd_op.unsqueeze: (1x14xi64) <- (14xi64, 1xi64)
        unsqueeze_6 = paddle._C_ops.unsqueeze(arange_6, full_int_array_2)
        del arange_6

        # pd_op.unsqueeze: (20x1xi64) <- (20xi64, 1xi64)
        unsqueeze_7 = paddle._C_ops.unsqueeze(arange_7, full_int_array_3)
        del arange_7

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_8 = [20, 14]

        # pd_op.expand: (20x14xi64) <- (1x14xi64, 2xi64)
        expand_6 = paddle._C_ops.expand(unsqueeze_6, full_int_array_8)
        del unsqueeze_6

        # pd_op.expand: (20x14xi64) <- (20x1xi64, 2xi64)
        expand_7 = paddle._C_ops.expand(unsqueeze_7, full_int_array_8)
        del full_int_array_8, unsqueeze_7

        # pd_op.reshape: (280xi64) <- (20x14xi64, 1xi64)
        reshape_25 = paddle._C_ops.reshape(expand_6, full_int_array_5)
        del expand_6

        # pd_op.reshape: (280xi64) <- (20x14xi64, 1xi64)
        reshape_26 = paddle._C_ops.reshape(expand_7, full_int_array_5)
        del expand_7

        # builtin.combine: ([280xi64, 280xi64]) <- (280xi64, 280xi64)
        combine_3 = [reshape_25, reshape_26]
        del reshape_25, reshape_26

        # pd_op.stack: (280x2xi64) <- ([280xi64, 280xi64])
        stack_3 = paddle._C_ops.stack(combine_3, -1)
        del combine_3

        # pd_op.cast: (280x2xf32) <- (280x2xi64)
        cast_3 = paddle._C_ops.cast(stack_3, paddle.float32)
        del stack_3

        # pd_op.scale: (280x2xf32) <- (280x2xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(cast_3, full_9, float("32"), True)
        del cast_3

        # pd_op.full: (1xf64) <- ()
        full_13 = paddle._C_ops.full(
            [1], float("128"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (7xi64) <- (1xf64, 1xf64, 1xf64)
        arange_8 = paddle.arange(full_5, full_6, full_13, dtype="int64")
        del full_6

        # pd_op.arange: (10xi64) <- (1xf64, 1xf64, 1xf64)
        arange_9 = paddle.arange(full_5, full_8, full_13, dtype="int64")
        del full_13, full_5, full_8

        # pd_op.unsqueeze: (1x7xi64) <- (7xi64, 1xi64)
        unsqueeze_8 = paddle._C_ops.unsqueeze(arange_8, full_int_array_2)
        del arange_8, full_int_array_2

        # pd_op.unsqueeze: (10x1xi64) <- (10xi64, 1xi64)
        unsqueeze_9 = paddle._C_ops.unsqueeze(arange_9, full_int_array_3)
        del arange_9, full_int_array_3

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_9 = [10, 7]

        # pd_op.expand: (10x7xi64) <- (1x7xi64, 2xi64)
        expand_8 = paddle._C_ops.expand(unsqueeze_8, full_int_array_9)
        del unsqueeze_8

        # pd_op.expand: (10x7xi64) <- (10x1xi64, 2xi64)
        expand_9 = paddle._C_ops.expand(unsqueeze_9, full_int_array_9)
        del full_int_array_9, unsqueeze_9

        # pd_op.reshape: (70xi64) <- (10x7xi64, 1xi64)
        reshape_27 = paddle._C_ops.reshape(expand_8, full_int_array_5)
        del expand_8

        # pd_op.reshape: (70xi64) <- (10x7xi64, 1xi64)
        reshape_28 = paddle._C_ops.reshape(expand_9, full_int_array_5)
        del expand_9, full_int_array_5

        # builtin.combine: ([70xi64, 70xi64]) <- (70xi64, 70xi64)
        combine_4 = [reshape_27, reshape_28]
        del reshape_27, reshape_28

        # pd_op.stack: (70x2xi64) <- ([70xi64, 70xi64])
        stack_4 = paddle._C_ops.stack(combine_4, -1)
        del combine_4

        # pd_op.cast: (70x2xf32) <- (70x2xi64)
        cast_4 = paddle._C_ops.cast(stack_4, paddle.float32)
        del stack_4

        # pd_op.scale: (70x2xf32) <- (70x2xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(cast_4, full_9, float("64"), True)
        del cast_4, full_9

        return (
            scale_0,
            scale_1,
            scale_2,
            scale_3,
            scale_4,
            add_0,
            add_1,
            add_2,
            add_3,
            add_4,
            scale_5,
            scale_6,
            scale_7,
            scale_8,
            scale_9,
            add_5,
            add_6,
            add_7,
            add_8,
            add_9,
        )
