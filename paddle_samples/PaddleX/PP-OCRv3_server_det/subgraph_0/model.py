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
        data_0,
    ):
        # pd_op.conv2d: (-1x32x-1x-1xf32) <- (-1x3x-1x-1xf32, 32x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_302, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_0, parameter_302

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
                parameter_301,
                parameter_300,
                parameter_299,
                parameter_298,
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
        del conv2d_0, parameter_298, parameter_299, parameter_300, parameter_301

        # pd_op.relu: (-1x32x-1x-1xf32) <- (-1x32x-1x-1xf32)
        relu_0 = paddle._C_ops.relu(batch_norm__0)
        del batch_norm__0

        # pd_op.conv2d: (-1x32x-1x-1xf32) <- (-1x32x-1x-1xf32, 32x32x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            relu_0, parameter_297, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_297, relu_0

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
                conv2d_1,
                parameter_296,
                parameter_295,
                parameter_294,
                parameter_293,
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
        del conv2d_1, parameter_293, parameter_294, parameter_295, parameter_296

        # pd_op.relu: (-1x32x-1x-1xf32) <- (-1x32x-1x-1xf32)
        relu_1 = paddle._C_ops.relu(batch_norm__6)
        del batch_norm__6

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x32x-1x-1xf32, 64x32x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            relu_1, parameter_292, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_292, relu_1

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
                conv2d_2,
                parameter_291,
                parameter_290,
                parameter_289,
                parameter_288,
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
        del conv2d_2, parameter_288, parameter_289, parameter_290, parameter_291

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_2 = paddle._C_ops.relu(batch_norm__12)
        del batch_norm__12

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [3, 3]

        # pd_op.pool2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            relu_2,
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
        del full_int_array_0, relu_2

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            pool2d_0, parameter_287, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_287

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
                conv2d_3,
                parameter_286,
                parameter_285,
                parameter_284,
                parameter_283,
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
        del conv2d_3, parameter_283, parameter_284, parameter_285, parameter_286

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_3 = paddle._C_ops.relu(batch_norm__18)
        del batch_norm__18

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x3x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            relu_3, parameter_282, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_282, relu_3

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_281,
                parameter_280,
                parameter_279,
                parameter_278,
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
        del conv2d_4, parameter_278, parameter_279, parameter_280, parameter_281

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_4 = paddle._C_ops.relu(batch_norm__24)
        del batch_norm__24

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x64x-1x-1xf32, 256x64x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            relu_4, parameter_277, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_277, relu_4

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_276,
                parameter_275,
                parameter_274,
                parameter_273,
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
        del conv2d_5, parameter_273, parameter_274, parameter_275, parameter_276

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x64x-1x-1xf32, 256x64x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            pool2d_0, parameter_272, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_272, pool2d_0

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_271,
                parameter_270,
                parameter_269,
                parameter_268,
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
        del conv2d_6, parameter_268, parameter_269, parameter_270, parameter_271

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, -1x256x-1x-1xf32)
        add_0 = paddle._C_ops.add(batch_norm__36, batch_norm__30)
        del batch_norm__30, batch_norm__36

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_5 = paddle._C_ops.relu(add_0)
        del add_0

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x256x-1x-1xf32, 64x256x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            relu_5, parameter_267, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_267

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_266,
                parameter_265,
                parameter_264,
                parameter_263,
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
        del conv2d_7, parameter_263, parameter_264, parameter_265, parameter_266

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_6 = paddle._C_ops.relu(batch_norm__42)
        del batch_norm__42

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            relu_6, parameter_262, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_262, relu_6

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_261,
                parameter_260,
                parameter_259,
                parameter_258,
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
        del conv2d_8, parameter_258, parameter_259, parameter_260, parameter_261

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_7 = paddle._C_ops.relu(batch_norm__48)
        del batch_norm__48

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x64x-1x-1xf32, 256x64x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            relu_7, parameter_257, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_257, relu_7

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
                conv2d_9,
                parameter_256,
                parameter_255,
                parameter_254,
                parameter_253,
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
        del conv2d_9, parameter_253, parameter_254, parameter_255, parameter_256

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, -1x256x-1x-1xf32)
        add_1 = paddle._C_ops.add(relu_5, batch_norm__54)
        del batch_norm__54, relu_5

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_8 = paddle._C_ops.relu(add_1)
        del add_1

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x256x-1x-1xf32, 64x256x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            relu_8, parameter_252, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_252

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_251,
                parameter_250,
                parameter_249,
                parameter_248,
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
        del conv2d_10, parameter_248, parameter_249, parameter_250, parameter_251

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_9 = paddle._C_ops.relu(batch_norm__60)
        del batch_norm__60

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            relu_9, parameter_247, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_247, relu_9

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_246,
                parameter_245,
                parameter_244,
                parameter_243,
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
        del conv2d_11, parameter_243, parameter_244, parameter_245, parameter_246

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_10 = paddle._C_ops.relu(batch_norm__66)
        del batch_norm__66

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x64x-1x-1xf32, 256x64x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            relu_10, parameter_242, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_242, relu_10

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
                conv2d_12,
                parameter_241,
                parameter_240,
                parameter_239,
                parameter_238,
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
        del conv2d_12, parameter_238, parameter_239, parameter_240, parameter_241

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, -1x256x-1x-1xf32)
        add_2 = paddle._C_ops.add(relu_8, batch_norm__72)
        del batch_norm__72, relu_8

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_11 = paddle._C_ops.relu(add_2)
        del add_2

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x256x-1x-1xf32, 128x256x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            relu_11, parameter_237, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_237

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_236,
                parameter_235,
                parameter_234,
                parameter_233,
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
        del conv2d_13, parameter_233, parameter_234, parameter_235, parameter_236

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_12 = paddle._C_ops.relu(batch_norm__78)
        del batch_norm__78

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x128x3x3xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            relu_12, parameter_232, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_232, relu_12

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_231,
                parameter_230,
                parameter_229,
                parameter_228,
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
        del conv2d_14, parameter_228, parameter_229, parameter_230, parameter_231

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_13 = paddle._C_ops.relu(batch_norm__84)
        del batch_norm__84

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x128x-1x-1xf32, 512x128x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            relu_13, parameter_227, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_227, relu_13

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
                conv2d_15,
                parameter_226,
                parameter_225,
                parameter_224,
                parameter_223,
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
        del conv2d_15, parameter_223, parameter_224, parameter_225, parameter_226

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [2, 2]

        # pd_op.pool2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(
            relu_11,
            full_int_array_1,
            [2, 2],
            [0, 0],
            True,
            True,
            "NCHW",
            "avg",
            False,
            False,
            "EXPLICIT",
        )

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x256x-1x-1xf32, 512x256x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            pool2d_1, parameter_222, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_222, pool2d_1

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
                conv2d_16,
                parameter_221,
                parameter_220,
                parameter_219,
                parameter_218,
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
        del conv2d_16, parameter_218, parameter_219, parameter_220, parameter_221

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, -1x512x-1x-1xf32)
        add_3 = paddle._C_ops.add(batch_norm__96, batch_norm__90)
        del batch_norm__90, batch_norm__96

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_14 = paddle._C_ops.relu(add_3)
        del add_3

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x512x-1x-1xf32, 128x512x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            relu_14, parameter_217, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_217

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_216,
                parameter_215,
                parameter_214,
                parameter_213,
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
        del conv2d_17, parameter_213, parameter_214, parameter_215, parameter_216

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_15 = paddle._C_ops.relu(batch_norm__102)
        del batch_norm__102

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x128x3x3xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            relu_15, parameter_212, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_212, relu_15

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_211,
                parameter_210,
                parameter_209,
                parameter_208,
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
        del conv2d_18, parameter_208, parameter_209, parameter_210, parameter_211

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_16 = paddle._C_ops.relu(batch_norm__108)
        del batch_norm__108

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x128x-1x-1xf32, 512x128x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            relu_16, parameter_207, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_207, relu_16

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
                conv2d_19,
                parameter_206,
                parameter_205,
                parameter_204,
                parameter_203,
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
        del conv2d_19, parameter_203, parameter_204, parameter_205, parameter_206

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, -1x512x-1x-1xf32)
        add_4 = paddle._C_ops.add(relu_14, batch_norm__114)
        del batch_norm__114, relu_14

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_17 = paddle._C_ops.relu(add_4)
        del add_4

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x512x-1x-1xf32, 128x512x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            relu_17, parameter_202, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_202

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_201,
                parameter_200,
                parameter_199,
                parameter_198,
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
        del conv2d_20, parameter_198, parameter_199, parameter_200, parameter_201

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_18 = paddle._C_ops.relu(batch_norm__120)
        del batch_norm__120

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x128x3x3xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            relu_18, parameter_197, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_197, relu_18

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_196,
                parameter_195,
                parameter_194,
                parameter_193,
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
        del conv2d_21, parameter_193, parameter_194, parameter_195, parameter_196

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_19 = paddle._C_ops.relu(batch_norm__126)
        del batch_norm__126

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x128x-1x-1xf32, 512x128x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(
            relu_19, parameter_192, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_192, relu_19

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
                conv2d_22,
                parameter_191,
                parameter_190,
                parameter_189,
                parameter_188,
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
        del conv2d_22, parameter_188, parameter_189, parameter_190, parameter_191

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, -1x512x-1x-1xf32)
        add_5 = paddle._C_ops.add(relu_17, batch_norm__132)
        del batch_norm__132, relu_17

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_20 = paddle._C_ops.relu(add_5)
        del add_5

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x512x-1x-1xf32, 128x512x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(
            relu_20, parameter_187, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_187

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_186,
                parameter_185,
                parameter_184,
                parameter_183,
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
        del conv2d_23, parameter_183, parameter_184, parameter_185, parameter_186

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_21 = paddle._C_ops.relu(batch_norm__138)
        del batch_norm__138

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x128x3x3xf32)
        conv2d_24 = paddle._C_ops.conv2d(
            relu_21, parameter_182, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_182, relu_21

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_181,
                parameter_180,
                parameter_179,
                parameter_178,
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
        del conv2d_24, parameter_178, parameter_179, parameter_180, parameter_181

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_22 = paddle._C_ops.relu(batch_norm__144)
        del batch_norm__144

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x128x-1x-1xf32, 512x128x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(
            relu_22, parameter_177, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_177, relu_22

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
                conv2d_25,
                parameter_176,
                parameter_175,
                parameter_174,
                parameter_173,
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
        del conv2d_25, parameter_173, parameter_174, parameter_175, parameter_176

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, -1x512x-1x-1xf32)
        add_6 = paddle._C_ops.add(relu_20, batch_norm__150)
        del batch_norm__150, relu_20

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_23 = paddle._C_ops.relu(add_6)
        del add_6

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x512x-1x-1xf32, 256x512x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(
            relu_23, parameter_172, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_172

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_171,
                parameter_170,
                parameter_169,
                parameter_168,
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
        del conv2d_26, parameter_168, parameter_169, parameter_170, parameter_171

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_24 = paddle._C_ops.relu(batch_norm__156)
        del batch_norm__156

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_27 = paddle._C_ops.conv2d(
            relu_24, parameter_167, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_167, relu_24

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_166,
                parameter_165,
                parameter_164,
                parameter_163,
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
        del conv2d_27, parameter_163, parameter_164, parameter_165, parameter_166

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_25 = paddle._C_ops.relu(batch_norm__162)
        del batch_norm__162

        # pd_op.conv2d: (-1x1024x-1x-1xf32) <- (-1x256x-1x-1xf32, 1024x256x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(
            relu_25, parameter_162, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_162, relu_25

        # pd_op.batch_norm_: (-1x1024x-1x-1xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x-1x-1xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
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
                parameter_161,
                parameter_160,
                parameter_159,
                parameter_158,
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
        del conv2d_28, parameter_158, parameter_159, parameter_160, parameter_161

        # pd_op.pool2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(
            relu_23,
            full_int_array_1,
            [2, 2],
            [0, 0],
            True,
            True,
            "NCHW",
            "avg",
            False,
            False,
            "EXPLICIT",
        )

        # pd_op.conv2d: (-1x1024x-1x-1xf32) <- (-1x512x-1x-1xf32, 1024x512x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(
            pool2d_2, parameter_157, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_157, pool2d_2

        # pd_op.batch_norm_: (-1x1024x-1x-1xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x-1x-1xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
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
                parameter_156,
                parameter_155,
                parameter_154,
                parameter_153,
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
        del conv2d_29, parameter_153, parameter_154, parameter_155, parameter_156

        # pd_op.add: (-1x1024x-1x-1xf32) <- (-1x1024x-1x-1xf32, -1x1024x-1x-1xf32)
        add_7 = paddle._C_ops.add(batch_norm__174, batch_norm__168)
        del batch_norm__168, batch_norm__174

        # pd_op.relu: (-1x1024x-1x-1xf32) <- (-1x1024x-1x-1xf32)
        relu_26 = paddle._C_ops.relu(add_7)
        del add_7

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x1024x-1x-1xf32, 256x1024x1x1xf32)
        conv2d_30 = paddle._C_ops.conv2d(
            relu_26, parameter_152, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_152

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_151,
                parameter_150,
                parameter_149,
                parameter_148,
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
        del conv2d_30, parameter_148, parameter_149, parameter_150, parameter_151

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_27 = paddle._C_ops.relu(batch_norm__180)
        del batch_norm__180

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_31 = paddle._C_ops.conv2d(
            relu_27, parameter_147, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_147, relu_27

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_146,
                parameter_145,
                parameter_144,
                parameter_143,
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
        del conv2d_31, parameter_143, parameter_144, parameter_145, parameter_146

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_28 = paddle._C_ops.relu(batch_norm__186)
        del batch_norm__186

        # pd_op.conv2d: (-1x1024x-1x-1xf32) <- (-1x256x-1x-1xf32, 1024x256x1x1xf32)
        conv2d_32 = paddle._C_ops.conv2d(
            relu_28, parameter_142, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_142, relu_28

        # pd_op.batch_norm_: (-1x1024x-1x-1xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x-1x-1xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
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
                parameter_141,
                parameter_140,
                parameter_139,
                parameter_138,
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
        del conv2d_32, parameter_138, parameter_139, parameter_140, parameter_141

        # pd_op.add: (-1x1024x-1x-1xf32) <- (-1x1024x-1x-1xf32, -1x1024x-1x-1xf32)
        add_8 = paddle._C_ops.add(relu_26, batch_norm__192)
        del batch_norm__192, relu_26

        # pd_op.relu: (-1x1024x-1x-1xf32) <- (-1x1024x-1x-1xf32)
        relu_29 = paddle._C_ops.relu(add_8)
        del add_8

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x1024x-1x-1xf32, 256x1024x1x1xf32)
        conv2d_33 = paddle._C_ops.conv2d(
            relu_29, parameter_137, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_137

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
                conv2d_33,
                parameter_136,
                parameter_135,
                parameter_134,
                parameter_133,
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
        del conv2d_33, parameter_133, parameter_134, parameter_135, parameter_136

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_30 = paddle._C_ops.relu(batch_norm__198)
        del batch_norm__198

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_34 = paddle._C_ops.conv2d(
            relu_30, parameter_132, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_132, relu_30

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_131,
                parameter_130,
                parameter_129,
                parameter_128,
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
        del conv2d_34, parameter_128, parameter_129, parameter_130, parameter_131

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_31 = paddle._C_ops.relu(batch_norm__204)
        del batch_norm__204

        # pd_op.conv2d: (-1x1024x-1x-1xf32) <- (-1x256x-1x-1xf32, 1024x256x1x1xf32)
        conv2d_35 = paddle._C_ops.conv2d(
            relu_31, parameter_127, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_127, relu_31

        # pd_op.batch_norm_: (-1x1024x-1x-1xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x-1x-1xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
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
                parameter_126,
                parameter_125,
                parameter_124,
                parameter_123,
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
        del conv2d_35, parameter_123, parameter_124, parameter_125, parameter_126

        # pd_op.add: (-1x1024x-1x-1xf32) <- (-1x1024x-1x-1xf32, -1x1024x-1x-1xf32)
        add_9 = paddle._C_ops.add(relu_29, batch_norm__210)
        del batch_norm__210, relu_29

        # pd_op.relu: (-1x1024x-1x-1xf32) <- (-1x1024x-1x-1xf32)
        relu_32 = paddle._C_ops.relu(add_9)
        del add_9

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x1024x-1x-1xf32, 256x1024x1x1xf32)
        conv2d_36 = paddle._C_ops.conv2d(
            relu_32, parameter_122, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_122

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_121,
                parameter_120,
                parameter_119,
                parameter_118,
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
        del conv2d_36, parameter_118, parameter_119, parameter_120, parameter_121

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_33 = paddle._C_ops.relu(batch_norm__216)
        del batch_norm__216

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_37 = paddle._C_ops.conv2d(
            relu_33, parameter_117, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_117, relu_33

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_116,
                parameter_115,
                parameter_114,
                parameter_113,
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
        del conv2d_37, parameter_113, parameter_114, parameter_115, parameter_116

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_34 = paddle._C_ops.relu(batch_norm__222)
        del batch_norm__222

        # pd_op.conv2d: (-1x1024x-1x-1xf32) <- (-1x256x-1x-1xf32, 1024x256x1x1xf32)
        conv2d_38 = paddle._C_ops.conv2d(
            relu_34, parameter_112, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_112, relu_34

        # pd_op.batch_norm_: (-1x1024x-1x-1xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x-1x-1xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
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
                parameter_111,
                parameter_110,
                parameter_109,
                parameter_108,
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
        del conv2d_38, parameter_108, parameter_109, parameter_110, parameter_111

        # pd_op.add: (-1x1024x-1x-1xf32) <- (-1x1024x-1x-1xf32, -1x1024x-1x-1xf32)
        add_10 = paddle._C_ops.add(relu_32, batch_norm__228)
        del batch_norm__228, relu_32

        # pd_op.relu: (-1x1024x-1x-1xf32) <- (-1x1024x-1x-1xf32)
        relu_35 = paddle._C_ops.relu(add_10)
        del add_10

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x1024x-1x-1xf32, 256x1024x1x1xf32)
        conv2d_39 = paddle._C_ops.conv2d(
            relu_35, parameter_107, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_107

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_106,
                parameter_105,
                parameter_104,
                parameter_103,
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
        del conv2d_39, parameter_103, parameter_104, parameter_105, parameter_106

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_36 = paddle._C_ops.relu(batch_norm__234)
        del batch_norm__234

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_40 = paddle._C_ops.conv2d(
            relu_36, parameter_102, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_102, relu_36

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_101,
                parameter_100,
                parameter_99,
                parameter_98,
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
        del conv2d_40, parameter_100, parameter_101, parameter_98, parameter_99

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_37 = paddle._C_ops.relu(batch_norm__240)
        del batch_norm__240

        # pd_op.conv2d: (-1x1024x-1x-1xf32) <- (-1x256x-1x-1xf32, 1024x256x1x1xf32)
        conv2d_41 = paddle._C_ops.conv2d(
            relu_37, parameter_97, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_97, relu_37

        # pd_op.batch_norm_: (-1x1024x-1x-1xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x-1x-1xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
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
                parameter_96,
                parameter_95,
                parameter_94,
                parameter_93,
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
        del conv2d_41, parameter_93, parameter_94, parameter_95, parameter_96

        # pd_op.add: (-1x1024x-1x-1xf32) <- (-1x1024x-1x-1xf32, -1x1024x-1x-1xf32)
        add_11 = paddle._C_ops.add(relu_35, batch_norm__246)
        del batch_norm__246, relu_35

        # pd_op.relu: (-1x1024x-1x-1xf32) <- (-1x1024x-1x-1xf32)
        relu_38 = paddle._C_ops.relu(add_11)
        del add_11

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x1024x-1x-1xf32, 256x1024x1x1xf32)
        conv2d_42 = paddle._C_ops.conv2d(
            relu_38, parameter_92, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_92

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_91,
                parameter_90,
                parameter_89,
                parameter_88,
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
        del conv2d_42, parameter_88, parameter_89, parameter_90, parameter_91

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_39 = paddle._C_ops.relu(batch_norm__252)
        del batch_norm__252

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_43 = paddle._C_ops.conv2d(
            relu_39, parameter_87, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_87, relu_39

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_86,
                parameter_85,
                parameter_84,
                parameter_83,
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
        del conv2d_43, parameter_83, parameter_84, parameter_85, parameter_86

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_40 = paddle._C_ops.relu(batch_norm__258)
        del batch_norm__258

        # pd_op.conv2d: (-1x1024x-1x-1xf32) <- (-1x256x-1x-1xf32, 1024x256x1x1xf32)
        conv2d_44 = paddle._C_ops.conv2d(
            relu_40, parameter_82, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_82, relu_40

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
                conv2d_44,
                parameter_81,
                parameter_80,
                parameter_79,
                parameter_78,
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
        del conv2d_44, parameter_78, parameter_79, parameter_80, parameter_81

        # pd_op.add: (-1x1024x-1x-1xf32) <- (-1x1024x-1x-1xf32, -1x1024x-1x-1xf32)
        add_12 = paddle._C_ops.add(relu_38, batch_norm__264)
        del batch_norm__264, relu_38

        # pd_op.relu: (-1x1024x-1x-1xf32) <- (-1x1024x-1x-1xf32)
        relu_41 = paddle._C_ops.relu(add_12)
        del add_12

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x1024x-1x-1xf32, 512x1024x1x1xf32)
        conv2d_45 = paddle._C_ops.conv2d(
            relu_41, parameter_77, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_77

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_76,
                parameter_75,
                parameter_74,
                parameter_73,
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
        del conv2d_45, parameter_73, parameter_74, parameter_75, parameter_76

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_42 = paddle._C_ops.relu(batch_norm__270)
        del batch_norm__270

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x512x3x3xf32)
        conv2d_46 = paddle._C_ops.conv2d(
            relu_42, parameter_72, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_72, relu_42

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_71,
                parameter_70,
                parameter_69,
                parameter_68,
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
        del conv2d_46, parameter_68, parameter_69, parameter_70, parameter_71

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_43 = paddle._C_ops.relu(batch_norm__276)
        del batch_norm__276

        # pd_op.conv2d: (-1x2048x-1x-1xf32) <- (-1x512x-1x-1xf32, 2048x512x1x1xf32)
        conv2d_47 = paddle._C_ops.conv2d(
            relu_43, parameter_67, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_67, relu_43

        # pd_op.batch_norm_: (-1x2048x-1x-1xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (-1x2048x-1x-1xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
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
                parameter_66,
                parameter_65,
                parameter_64,
                parameter_63,
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
        del conv2d_47, parameter_63, parameter_64, parameter_65, parameter_66

        # pd_op.pool2d: (-1x1024x-1x-1xf32) <- (-1x1024x-1x-1xf32, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(
            relu_41,
            full_int_array_1,
            [2, 2],
            [0, 0],
            True,
            True,
            "NCHW",
            "avg",
            False,
            False,
            "EXPLICIT",
        )
        del full_int_array_1

        # pd_op.conv2d: (-1x2048x-1x-1xf32) <- (-1x1024x-1x-1xf32, 2048x1024x1x1xf32)
        conv2d_48 = paddle._C_ops.conv2d(
            pool2d_3, parameter_62, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_62, pool2d_3

        # pd_op.batch_norm_: (-1x2048x-1x-1xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (-1x2048x-1x-1xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
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
                parameter_61,
                parameter_60,
                parameter_59,
                parameter_58,
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
        del conv2d_48, parameter_58, parameter_59, parameter_60, parameter_61

        # pd_op.add: (-1x2048x-1x-1xf32) <- (-1x2048x-1x-1xf32, -1x2048x-1x-1xf32)
        add_13 = paddle._C_ops.add(batch_norm__288, batch_norm__282)
        del batch_norm__282, batch_norm__288

        # pd_op.relu: (-1x2048x-1x-1xf32) <- (-1x2048x-1x-1xf32)
        relu_44 = paddle._C_ops.relu(add_13)
        del add_13

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x2048x-1x-1xf32, 512x2048x1x1xf32)
        conv2d_49 = paddle._C_ops.conv2d(
            relu_44, parameter_57, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_57

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_56,
                parameter_55,
                parameter_54,
                parameter_53,
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
        del conv2d_49, parameter_53, parameter_54, parameter_55, parameter_56

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_45 = paddle._C_ops.relu(batch_norm__294)
        del batch_norm__294

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x512x3x3xf32)
        conv2d_50 = paddle._C_ops.conv2d(
            relu_45, parameter_52, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_52, relu_45

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_51,
                parameter_50,
                parameter_49,
                parameter_48,
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
        del conv2d_50, parameter_48, parameter_49, parameter_50, parameter_51

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_46 = paddle._C_ops.relu(batch_norm__300)
        del batch_norm__300

        # pd_op.conv2d: (-1x2048x-1x-1xf32) <- (-1x512x-1x-1xf32, 2048x512x1x1xf32)
        conv2d_51 = paddle._C_ops.conv2d(
            relu_46, parameter_47, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_47, relu_46

        # pd_op.batch_norm_: (-1x2048x-1x-1xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (-1x2048x-1x-1xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
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
                parameter_46,
                parameter_45,
                parameter_44,
                parameter_43,
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
        del conv2d_51, parameter_43, parameter_44, parameter_45, parameter_46

        # pd_op.add: (-1x2048x-1x-1xf32) <- (-1x2048x-1x-1xf32, -1x2048x-1x-1xf32)
        add_14 = paddle._C_ops.add(relu_44, batch_norm__306)
        del batch_norm__306, relu_44

        # pd_op.relu: (-1x2048x-1x-1xf32) <- (-1x2048x-1x-1xf32)
        relu_47 = paddle._C_ops.relu(add_14)
        del add_14

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x2048x-1x-1xf32, 512x2048x1x1xf32)
        conv2d_52 = paddle._C_ops.conv2d(
            relu_47, parameter_42, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_42

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_41,
                parameter_40,
                parameter_39,
                parameter_38,
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
        del conv2d_52, parameter_38, parameter_39, parameter_40, parameter_41

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_48 = paddle._C_ops.relu(batch_norm__312)
        del batch_norm__312

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x512x3x3xf32)
        conv2d_53 = paddle._C_ops.conv2d(
            relu_48, parameter_37, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_37, relu_48

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_36,
                parameter_35,
                parameter_34,
                parameter_33,
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
        del conv2d_53, parameter_33, parameter_34, parameter_35, parameter_36

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_49 = paddle._C_ops.relu(batch_norm__318)
        del batch_norm__318

        # pd_op.conv2d: (-1x2048x-1x-1xf32) <- (-1x512x-1x-1xf32, 2048x512x1x1xf32)
        conv2d_54 = paddle._C_ops.conv2d(
            relu_49, parameter_32, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_32, relu_49

        # pd_op.batch_norm_: (-1x2048x-1x-1xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (-1x2048x-1x-1xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
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
                parameter_31,
                parameter_30,
                parameter_29,
                parameter_28,
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
        del conv2d_54, parameter_28, parameter_29, parameter_30, parameter_31

        # pd_op.add: (-1x2048x-1x-1xf32) <- (-1x2048x-1x-1xf32, -1x2048x-1x-1xf32)
        add_15 = paddle._C_ops.add(relu_47, batch_norm__324)
        del batch_norm__324, relu_47

        # pd_op.relu: (-1x2048x-1x-1xf32) <- (-1x2048x-1x-1xf32)
        relu_50 = paddle._C_ops.relu(add_15)
        del add_15

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x2048x-1x-1xf32, 256x2048x1x1xf32)
        conv2d_55 = paddle._C_ops.conv2d(
            relu_50, parameter_27, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_27

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x1024x-1x-1xf32, 256x1024x1x1xf32)
        conv2d_56 = paddle._C_ops.conv2d(
            relu_41, parameter_26, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_26

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x512x-1x-1xf32, 256x512x1x1xf32)
        conv2d_57 = paddle._C_ops.conv2d(
            relu_23, parameter_25, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_25

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x1x1xf32)
        conv2d_58 = paddle._C_ops.conv2d(
            relu_11, parameter_24, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_24

        # pd_op.nearest_interp: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, None, None, None)
        nearest_interp_0 = paddle._C_ops.nearest_interp(
            conv2d_55,
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

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, -1x256x-1x-1xf32)
        add_16 = paddle._C_ops.add(conv2d_56, nearest_interp_0)
        del conv2d_56, nearest_interp_0

        # pd_op.nearest_interp: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, None, None, None)
        nearest_interp_1 = paddle._C_ops.nearest_interp(
            add_16,
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

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, -1x256x-1x-1xf32)
        add_17 = paddle._C_ops.add(conv2d_57, nearest_interp_1)
        del conv2d_57, nearest_interp_1

        # pd_op.nearest_interp: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, None, None, None)
        nearest_interp_2 = paddle._C_ops.nearest_interp(
            add_17,
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

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, -1x256x-1x-1xf32)
        add_18 = paddle._C_ops.add(conv2d_58, nearest_interp_2)
        del conv2d_58, nearest_interp_2

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x256x-1x-1xf32, 64x256x9x9xf32)
        conv2d_59 = paddle._C_ops.conv2d(
            conv2d_55, parameter_23, [1, 1], [4, 4], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del conv2d_55, parameter_23

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x256x-1x-1xf32, 64x256x9x9xf32)
        conv2d_60 = paddle._C_ops.conv2d(
            add_16, parameter_22, [1, 1], [4, 4], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_16, parameter_22

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x256x-1x-1xf32, 64x256x9x9xf32)
        conv2d_61 = paddle._C_ops.conv2d(
            add_17, parameter_21, [1, 1], [4, 4], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_17, parameter_21

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x256x-1x-1xf32, 64x256x9x9xf32)
        conv2d_62 = paddle._C_ops.conv2d(
            add_18, parameter_20, [1, 1], [4, 4], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_18, parameter_20

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x3x3xf32)
        conv2d_63 = paddle._C_ops.conv2d(
            conv2d_62, parameter_19, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_19

        # pd_op.add: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, -1x64x-1x-1xf32)
        add_19 = paddle._C_ops.add(conv2d_61, conv2d_63)
        del conv2d_61, conv2d_63

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x3x3xf32)
        conv2d_64 = paddle._C_ops.conv2d(
            add_19, parameter_18, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_18

        # pd_op.add: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, -1x64x-1x-1xf32)
        add_20 = paddle._C_ops.add(conv2d_60, conv2d_64)
        del conv2d_60, conv2d_64

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x3x3xf32)
        conv2d_65 = paddle._C_ops.conv2d(
            add_20, parameter_17, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_17

        # pd_op.add: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, -1x64x-1x-1xf32)
        add_21 = paddle._C_ops.add(conv2d_59, conv2d_65)
        del conv2d_59, conv2d_65

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x9x9xf32)
        conv2d_66 = paddle._C_ops.conv2d(
            conv2d_62, parameter_16, [1, 1], [4, 4], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del conv2d_62, parameter_16

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x9x9xf32)
        conv2d_67 = paddle._C_ops.conv2d(
            add_19, parameter_15, [1, 1], [4, 4], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_19, parameter_15

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x9x9xf32)
        conv2d_68 = paddle._C_ops.conv2d(
            add_20, parameter_14, [1, 1], [4, 4], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_20, parameter_14

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x9x9xf32)
        conv2d_69 = paddle._C_ops.conv2d(
            add_21, parameter_13, [1, 1], [4, 4], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_21, parameter_13

        # pd_op.nearest_interp: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, None, None, None)
        nearest_interp_3 = paddle._C_ops.nearest_interp(
            conv2d_69,
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
        del conv2d_69

        # pd_op.nearest_interp: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, None, None, None)
        nearest_interp_4 = paddle._C_ops.nearest_interp(
            conv2d_68,
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
        del conv2d_68

        # pd_op.nearest_interp: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, None, None, None)
        nearest_interp_5 = paddle._C_ops.nearest_interp(
            conv2d_67,
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
        del conv2d_67

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x64x-1x-1xf32, -1x64x-1x-1xf32, -1x64x-1x-1xf32, -1x64x-1x-1xf32]) <- (-1x64x-1x-1xf32, -1x64x-1x-1xf32, -1x64x-1x-1xf32, -1x64x-1x-1xf32)
        combine_0 = [nearest_interp_3, nearest_interp_4, nearest_interp_5, conv2d_66]
        del conv2d_66, nearest_interp_3, nearest_interp_4, nearest_interp_5

        # pd_op.concat: (-1x256x-1x-1xf32) <- ([-1x64x-1x-1xf32, -1x64x-1x-1xf32, -1x64x-1x-1xf32, -1x64x-1x-1xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)
        del combine_0, full_0

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x256x-1x-1xf32, 64x256x7x7xf32)
        conv2d_70 = paddle._C_ops.conv2d(
            concat_0, parameter_12, [1, 1], [3, 3], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_12

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__330,
            batch_norm__331,
            batch_norm__332,
            batch_norm__333,
            batch_norm__334,
            batch_norm__335,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_70,
                parameter_11,
                parameter_10,
                parameter_9,
                parameter_8,
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
        del conv2d_70, parameter_10, parameter_11, parameter_8, parameter_9

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_51 = paddle._C_ops.relu(batch_norm__330)
        del batch_norm__330

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_2 = []

        # pd_op.conv2d_transpose: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x2x2xf32, 0xi64)
        conv2d_transpose_0 = paddle._C_ops.conv2d_transpose(
            relu_51,
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
        del parameter_7, relu_51

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_6, full_int_array_3)
        del full_int_array_3, parameter_6

        # pd_op.add: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 1x64x1x1xf32)
        add_22 = paddle._C_ops.add(conv2d_transpose_0, reshape_0)
        del conv2d_transpose_0, reshape_0

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__336,
            batch_norm__337,
            batch_norm__338,
            batch_norm__339,
            batch_norm__340,
            batch_norm__341,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_22,
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
        del add_22, parameter_2, parameter_3, parameter_4, parameter_5

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_52 = paddle._C_ops.relu(batch_norm__336)
        del batch_norm__336

        # pd_op.conv2d_transpose: (-1x1x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x1x2x2xf32, 0xi64)
        conv2d_transpose_1 = paddle._C_ops.conv2d_transpose(
            relu_52,
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
        del full_int_array_2, parameter_1, relu_52

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_4 = [1, 1, 1, 1]

        # pd_op.reshape: (1x1x1x1xf32) <- (1xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_0, full_int_array_4)
        del full_int_array_4, parameter_0

        # pd_op.add: (-1x1x-1x-1xf32) <- (-1x1x-1x-1xf32, 1x1x1x1xf32)
        add_23 = paddle._C_ops.add(conv2d_transpose_1, reshape_1)
        del conv2d_transpose_1, reshape_1

        # pd_op.sigmoid: (-1x1x-1x-1xf32) <- (-1x1x-1x-1xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(add_23)
        del add_23, concat_0, relu_11, relu_23, relu_41, relu_50

        return sigmoid_0
