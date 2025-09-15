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
        data_0,
    ):
        # pd_op.conv2d: (-1x32x112x112xf32) <- (-1x3x224x224xf32, 32x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_305, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_0, parameter_305

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_304, full_int_array_0)
        del parameter_304

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
                parameter_303,
                parameter_302,
                parameter_301,
                parameter_300,
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
        del add_1, parameter_300, parameter_301, parameter_302, parameter_303

        # pd_op.relu6: (-1x32x112x112xf32) <- (-1x32x112x112xf32)
        relu6_0 = paddle._C_ops.relu6(batch_norm__0)
        del batch_norm__0

        # pd_op.conv2d: (-1x24x56x56xf32) <- (-1x32x112x112xf32, 24x32x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            relu6_0, parameter_299, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_299, relu6_0

        # pd_op.reshape: (1x24x1x1xf32) <- (24xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_298, full_int_array_0)
        del parameter_298

        # pd_op.add: (-1x24x56x56xf32) <- (-1x24x56x56xf32, 1x24x1x1xf32)
        add_2 = paddle._C_ops.add(conv2d_1, reshape_1)
        del conv2d_1, reshape_1

        # pd_op.batch_norm_: (-1x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (-1x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32)
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
        del add_2, parameter_294, parameter_295, parameter_296, parameter_297

        # pd_op.depthwise_conv2d: (-1x24x56x56xf32) <- (-1x24x56x56xf32, 24x1x7x7xf32)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(
            batch_norm__6, parameter_293, [1, 1], [3, 3], "EXPLICIT", 24, [1, 1], "NCHW"
        )
        del parameter_293

        # pd_op.reshape: (1x24x1x1xf32) <- (24xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(parameter_292, full_int_array_0)
        del parameter_292

        # pd_op.add: (-1x24x56x56xf32) <- (-1x24x56x56xf32, 1x24x1x1xf32)
        add_3 = paddle._C_ops.add(depthwise_conv2d_0, reshape_2)
        del depthwise_conv2d_0, reshape_2

        # pd_op.batch_norm_: (-1x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (-1x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32)
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
                parameter_291,
                parameter_290,
                parameter_289,
                parameter_288,
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
        del add_3, parameter_288, parameter_289, parameter_290, parameter_291

        # pd_op.conv2d: (-1x96x56x56xf32) <- (-1x24x56x56xf32, 96x24x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            batch_norm__12, parameter_287, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_287

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(parameter_286, full_int_array_0)
        del parameter_286

        # pd_op.add: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 1x96x1x1xf32)
        add_4 = paddle._C_ops.add(conv2d_2, reshape_3)
        del conv2d_2, reshape_3

        # pd_op.conv2d: (-1x96x56x56xf32) <- (-1x24x56x56xf32, 96x24x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            batch_norm__12, parameter_285, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del batch_norm__12, parameter_285

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(parameter_284, full_int_array_0)
        del parameter_284

        # pd_op.add: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 1x96x1x1xf32)
        add_5 = paddle._C_ops.add(conv2d_3, reshape_4)
        del conv2d_3, reshape_4

        # pd_op.relu6: (-1x96x56x56xf32) <- (-1x96x56x56xf32)
        relu6_1 = paddle._C_ops.relu6(add_4)
        del add_4

        # pd_op.multiply: (-1x96x56x56xf32) <- (-1x96x56x56xf32, -1x96x56x56xf32)
        multiply_0 = paddle._C_ops.multiply(relu6_1, add_5)
        del add_5, relu6_1

        # pd_op.conv2d: (-1x24x56x56xf32) <- (-1x96x56x56xf32, 24x96x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            multiply_0, parameter_283, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_0, parameter_283

        # pd_op.reshape: (1x24x1x1xf32) <- (24xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(parameter_282, full_int_array_0)
        del parameter_282

        # pd_op.add: (-1x24x56x56xf32) <- (-1x24x56x56xf32, 1x24x1x1xf32)
        add_6 = paddle._C_ops.add(conv2d_4, reshape_5)
        del conv2d_4, reshape_5

        # pd_op.batch_norm_: (-1x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (-1x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32)
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
                parameter_281,
                parameter_280,
                parameter_279,
                parameter_278,
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
        del add_6, parameter_278, parameter_279, parameter_280, parameter_281

        # pd_op.depthwise_conv2d: (-1x24x56x56xf32) <- (-1x24x56x56xf32, 24x1x7x7xf32)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(
            batch_norm__18,
            parameter_277,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            24,
            [1, 1],
            "NCHW",
        )
        del batch_norm__18, parameter_277

        # pd_op.reshape: (1x24x1x1xf32) <- (24xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(parameter_276, full_int_array_0)
        del parameter_276

        # pd_op.add: (-1x24x56x56xf32) <- (-1x24x56x56xf32, 1x24x1x1xf32)
        add_7 = paddle._C_ops.add(depthwise_conv2d_1, reshape_6)
        del depthwise_conv2d_1, reshape_6

        # pd_op.add: (-1x24x56x56xf32) <- (-1x24x56x56xf32, -1x24x56x56xf32)
        add_8 = paddle._C_ops.add(batch_norm__6, add_7)
        del add_7, batch_norm__6

        # pd_op.depthwise_conv2d: (-1x24x56x56xf32) <- (-1x24x56x56xf32, 24x1x7x7xf32)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(
            add_8, parameter_275, [1, 1], [3, 3], "EXPLICIT", 24, [1, 1], "NCHW"
        )
        del parameter_275

        # pd_op.reshape: (1x24x1x1xf32) <- (24xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(parameter_274, full_int_array_0)
        del parameter_274

        # pd_op.add: (-1x24x56x56xf32) <- (-1x24x56x56xf32, 1x24x1x1xf32)
        add_9 = paddle._C_ops.add(depthwise_conv2d_2, reshape_7)
        del depthwise_conv2d_2, reshape_7

        # pd_op.batch_norm_: (-1x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (-1x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32)
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
                parameter_273,
                parameter_272,
                parameter_271,
                parameter_270,
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
        del add_9, parameter_270, parameter_271, parameter_272, parameter_273

        # pd_op.conv2d: (-1x96x56x56xf32) <- (-1x24x56x56xf32, 96x24x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            batch_norm__24, parameter_269, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_269

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(parameter_268, full_int_array_0)
        del parameter_268

        # pd_op.add: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 1x96x1x1xf32)
        add_10 = paddle._C_ops.add(conv2d_5, reshape_8)
        del conv2d_5, reshape_8

        # pd_op.conv2d: (-1x96x56x56xf32) <- (-1x24x56x56xf32, 96x24x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            batch_norm__24, parameter_267, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del batch_norm__24, parameter_267

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(parameter_266, full_int_array_0)
        del parameter_266

        # pd_op.add: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 1x96x1x1xf32)
        add_11 = paddle._C_ops.add(conv2d_6, reshape_9)
        del conv2d_6, reshape_9

        # pd_op.relu6: (-1x96x56x56xf32) <- (-1x96x56x56xf32)
        relu6_2 = paddle._C_ops.relu6(add_10)
        del add_10

        # pd_op.multiply: (-1x96x56x56xf32) <- (-1x96x56x56xf32, -1x96x56x56xf32)
        multiply_1 = paddle._C_ops.multiply(relu6_2, add_11)
        del add_11, relu6_2

        # pd_op.conv2d: (-1x24x56x56xf32) <- (-1x96x56x56xf32, 24x96x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            multiply_1, parameter_265, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_1, parameter_265

        # pd_op.reshape: (1x24x1x1xf32) <- (24xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(parameter_264, full_int_array_0)
        del parameter_264

        # pd_op.add: (-1x24x56x56xf32) <- (-1x24x56x56xf32, 1x24x1x1xf32)
        add_12 = paddle._C_ops.add(conv2d_7, reshape_10)
        del conv2d_7, reshape_10

        # pd_op.batch_norm_: (-1x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (-1x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        (
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_12,
                parameter_263,
                parameter_262,
                parameter_261,
                parameter_260,
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
        del add_12, parameter_260, parameter_261, parameter_262, parameter_263

        # pd_op.depthwise_conv2d: (-1x24x56x56xf32) <- (-1x24x56x56xf32, 24x1x7x7xf32)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(
            batch_norm__30,
            parameter_259,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            24,
            [1, 1],
            "NCHW",
        )
        del batch_norm__30, parameter_259

        # pd_op.reshape: (1x24x1x1xf32) <- (24xf32, 4xi64)
        reshape_11 = paddle._C_ops.reshape(parameter_258, full_int_array_0)
        del parameter_258

        # pd_op.add: (-1x24x56x56xf32) <- (-1x24x56x56xf32, 1x24x1x1xf32)
        add_13 = paddle._C_ops.add(depthwise_conv2d_3, reshape_11)
        del depthwise_conv2d_3, reshape_11

        # pd_op.add: (-1x24x56x56xf32) <- (-1x24x56x56xf32, -1x24x56x56xf32)
        add_14 = paddle._C_ops.add(add_8, add_13)
        del add_13, add_8

        # pd_op.conv2d: (-1x48x28x28xf32) <- (-1x24x56x56xf32, 48x24x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            add_14, parameter_257, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_14, parameter_257

        # pd_op.reshape: (1x48x1x1xf32) <- (48xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(parameter_256, full_int_array_0)
        del parameter_256

        # pd_op.add: (-1x48x28x28xf32) <- (-1x48x28x28xf32, 1x48x1x1xf32)
        add_15 = paddle._C_ops.add(conv2d_8, reshape_12)
        del conv2d_8, reshape_12

        # pd_op.batch_norm_: (-1x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (-1x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        (
            batch_norm__36,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_15,
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
        del add_15, parameter_252, parameter_253, parameter_254, parameter_255

        # pd_op.depthwise_conv2d: (-1x48x28x28xf32) <- (-1x48x28x28xf32, 48x1x7x7xf32)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(
            batch_norm__36,
            parameter_251,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            48,
            [1, 1],
            "NCHW",
        )
        del parameter_251

        # pd_op.reshape: (1x48x1x1xf32) <- (48xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(parameter_250, full_int_array_0)
        del parameter_250

        # pd_op.add: (-1x48x28x28xf32) <- (-1x48x28x28xf32, 1x48x1x1xf32)
        add_16 = paddle._C_ops.add(depthwise_conv2d_4, reshape_13)
        del depthwise_conv2d_4, reshape_13

        # pd_op.batch_norm_: (-1x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (-1x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32)
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
                parameter_249,
                parameter_248,
                parameter_247,
                parameter_246,
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
        del add_16, parameter_246, parameter_247, parameter_248, parameter_249

        # pd_op.conv2d: (-1x192x28x28xf32) <- (-1x48x28x28xf32, 192x48x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            batch_norm__42, parameter_245, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_245

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(parameter_244, full_int_array_0)
        del parameter_244

        # pd_op.add: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 1x192x1x1xf32)
        add_17 = paddle._C_ops.add(conv2d_9, reshape_14)
        del conv2d_9, reshape_14

        # pd_op.conv2d: (-1x192x28x28xf32) <- (-1x48x28x28xf32, 192x48x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            batch_norm__42, parameter_243, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del batch_norm__42, parameter_243

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        reshape_15 = paddle._C_ops.reshape(parameter_242, full_int_array_0)
        del parameter_242

        # pd_op.add: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 1x192x1x1xf32)
        add_18 = paddle._C_ops.add(conv2d_10, reshape_15)
        del conv2d_10, reshape_15

        # pd_op.relu6: (-1x192x28x28xf32) <- (-1x192x28x28xf32)
        relu6_3 = paddle._C_ops.relu6(add_17)
        del add_17

        # pd_op.multiply: (-1x192x28x28xf32) <- (-1x192x28x28xf32, -1x192x28x28xf32)
        multiply_2 = paddle._C_ops.multiply(relu6_3, add_18)
        del add_18, relu6_3

        # pd_op.conv2d: (-1x48x28x28xf32) <- (-1x192x28x28xf32, 48x192x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            multiply_2, parameter_241, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_2, parameter_241

        # pd_op.reshape: (1x48x1x1xf32) <- (48xf32, 4xi64)
        reshape_16 = paddle._C_ops.reshape(parameter_240, full_int_array_0)
        del parameter_240

        # pd_op.add: (-1x48x28x28xf32) <- (-1x48x28x28xf32, 1x48x1x1xf32)
        add_19 = paddle._C_ops.add(conv2d_11, reshape_16)
        del conv2d_11, reshape_16

        # pd_op.batch_norm_: (-1x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (-1x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32)
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
                parameter_239,
                parameter_238,
                parameter_237,
                parameter_236,
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
        del add_19, parameter_236, parameter_237, parameter_238, parameter_239

        # pd_op.depthwise_conv2d: (-1x48x28x28xf32) <- (-1x48x28x28xf32, 48x1x7x7xf32)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(
            batch_norm__48,
            parameter_235,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            48,
            [1, 1],
            "NCHW",
        )
        del batch_norm__48, parameter_235

        # pd_op.reshape: (1x48x1x1xf32) <- (48xf32, 4xi64)
        reshape_17 = paddle._C_ops.reshape(parameter_234, full_int_array_0)
        del parameter_234

        # pd_op.add: (-1x48x28x28xf32) <- (-1x48x28x28xf32, 1x48x1x1xf32)
        add_20 = paddle._C_ops.add(depthwise_conv2d_5, reshape_17)
        del depthwise_conv2d_5, reshape_17

        # pd_op.add: (-1x48x28x28xf32) <- (-1x48x28x28xf32, -1x48x28x28xf32)
        add_21 = paddle._C_ops.add(batch_norm__36, add_20)
        del add_20, batch_norm__36

        # pd_op.depthwise_conv2d: (-1x48x28x28xf32) <- (-1x48x28x28xf32, 48x1x7x7xf32)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(
            add_21, parameter_233, [1, 1], [3, 3], "EXPLICIT", 48, [1, 1], "NCHW"
        )
        del parameter_233

        # pd_op.reshape: (1x48x1x1xf32) <- (48xf32, 4xi64)
        reshape_18 = paddle._C_ops.reshape(parameter_232, full_int_array_0)
        del parameter_232

        # pd_op.add: (-1x48x28x28xf32) <- (-1x48x28x28xf32, 1x48x1x1xf32)
        add_22 = paddle._C_ops.add(depthwise_conv2d_6, reshape_18)
        del depthwise_conv2d_6, reshape_18

        # pd_op.batch_norm_: (-1x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (-1x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32)
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
        del add_22, parameter_228, parameter_229, parameter_230, parameter_231

        # pd_op.conv2d: (-1x192x28x28xf32) <- (-1x48x28x28xf32, 192x48x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            batch_norm__54, parameter_227, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_227

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        reshape_19 = paddle._C_ops.reshape(parameter_226, full_int_array_0)
        del parameter_226

        # pd_op.add: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 1x192x1x1xf32)
        add_23 = paddle._C_ops.add(conv2d_12, reshape_19)
        del conv2d_12, reshape_19

        # pd_op.conv2d: (-1x192x28x28xf32) <- (-1x48x28x28xf32, 192x48x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            batch_norm__54, parameter_225, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del batch_norm__54, parameter_225

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        reshape_20 = paddle._C_ops.reshape(parameter_224, full_int_array_0)
        del parameter_224

        # pd_op.add: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 1x192x1x1xf32)
        add_24 = paddle._C_ops.add(conv2d_13, reshape_20)
        del conv2d_13, reshape_20

        # pd_op.relu6: (-1x192x28x28xf32) <- (-1x192x28x28xf32)
        relu6_4 = paddle._C_ops.relu6(add_23)
        del add_23

        # pd_op.multiply: (-1x192x28x28xf32) <- (-1x192x28x28xf32, -1x192x28x28xf32)
        multiply_3 = paddle._C_ops.multiply(relu6_4, add_24)
        del add_24, relu6_4

        # pd_op.conv2d: (-1x48x28x28xf32) <- (-1x192x28x28xf32, 48x192x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            multiply_3, parameter_223, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_3, parameter_223

        # pd_op.reshape: (1x48x1x1xf32) <- (48xf32, 4xi64)
        reshape_21 = paddle._C_ops.reshape(parameter_222, full_int_array_0)
        del parameter_222

        # pd_op.add: (-1x48x28x28xf32) <- (-1x48x28x28xf32, 1x48x1x1xf32)
        add_25 = paddle._C_ops.add(conv2d_14, reshape_21)
        del conv2d_14, reshape_21

        # pd_op.batch_norm_: (-1x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (-1x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        (
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_25,
                parameter_221,
                parameter_220,
                parameter_219,
                parameter_218,
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
        del add_25, parameter_218, parameter_219, parameter_220, parameter_221

        # pd_op.depthwise_conv2d: (-1x48x28x28xf32) <- (-1x48x28x28xf32, 48x1x7x7xf32)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(
            batch_norm__60,
            parameter_217,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            48,
            [1, 1],
            "NCHW",
        )
        del batch_norm__60, parameter_217

        # pd_op.reshape: (1x48x1x1xf32) <- (48xf32, 4xi64)
        reshape_22 = paddle._C_ops.reshape(parameter_216, full_int_array_0)
        del parameter_216

        # pd_op.add: (-1x48x28x28xf32) <- (-1x48x28x28xf32, 1x48x1x1xf32)
        add_26 = paddle._C_ops.add(depthwise_conv2d_7, reshape_22)
        del depthwise_conv2d_7, reshape_22

        # pd_op.add: (-1x48x28x28xf32) <- (-1x48x28x28xf32, -1x48x28x28xf32)
        add_27 = paddle._C_ops.add(add_21, add_26)
        del add_21, add_26

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x48x28x28xf32, 96x48x3x3xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            add_27, parameter_215, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_27, parameter_215

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_23 = paddle._C_ops.reshape(parameter_214, full_int_array_0)
        del parameter_214

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        add_28 = paddle._C_ops.add(conv2d_15, reshape_23)
        del conv2d_15, reshape_23

        # pd_op.batch_norm_: (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__66,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            batch_norm__70,
            batch_norm__71,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_28,
                parameter_213,
                parameter_212,
                parameter_211,
                parameter_210,
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
        del add_28, parameter_210, parameter_211, parameter_212, parameter_213

        # pd_op.depthwise_conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x1x7x7xf32)
        depthwise_conv2d_8 = paddle._C_ops.depthwise_conv2d(
            batch_norm__66,
            parameter_209,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            96,
            [1, 1],
            "NCHW",
        )
        del parameter_209

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_24 = paddle._C_ops.reshape(parameter_208, full_int_array_0)
        del parameter_208

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        add_29 = paddle._C_ops.add(depthwise_conv2d_8, reshape_24)
        del depthwise_conv2d_8, reshape_24

        # pd_op.batch_norm_: (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
        del add_29, parameter_204, parameter_205, parameter_206, parameter_207

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x96x14x14xf32, 384x96x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            batch_norm__72, parameter_203, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_203

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_25 = paddle._C_ops.reshape(parameter_202, full_int_array_0)
        del parameter_202

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        add_30 = paddle._C_ops.add(conv2d_16, reshape_25)
        del conv2d_16, reshape_25

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x96x14x14xf32, 384x96x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            batch_norm__72, parameter_201, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del batch_norm__72, parameter_201

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_26 = paddle._C_ops.reshape(parameter_200, full_int_array_0)
        del parameter_200

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        add_31 = paddle._C_ops.add(conv2d_17, reshape_26)
        del conv2d_17, reshape_26

        # pd_op.relu6: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        relu6_5 = paddle._C_ops.relu6(add_30)
        del add_30

        # pd_op.multiply: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        multiply_4 = paddle._C_ops.multiply(relu6_5, add_31)
        del add_31, relu6_5

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x384x14x14xf32, 96x384x1x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            multiply_4, parameter_199, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_4, parameter_199

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_27 = paddle._C_ops.reshape(parameter_198, full_int_array_0)
        del parameter_198

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        add_32 = paddle._C_ops.add(conv2d_18, reshape_27)
        del conv2d_18, reshape_27

        # pd_op.batch_norm_: (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
        del add_32, parameter_194, parameter_195, parameter_196, parameter_197

        # pd_op.depthwise_conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x1x7x7xf32)
        depthwise_conv2d_9 = paddle._C_ops.depthwise_conv2d(
            batch_norm__78,
            parameter_193,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            96,
            [1, 1],
            "NCHW",
        )
        del batch_norm__78, parameter_193

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_28 = paddle._C_ops.reshape(parameter_192, full_int_array_0)
        del parameter_192

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        add_33 = paddle._C_ops.add(depthwise_conv2d_9, reshape_28)
        del depthwise_conv2d_9, reshape_28

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, -1x96x14x14xf32)
        add_34 = paddle._C_ops.add(batch_norm__66, add_33)
        del add_33, batch_norm__66

        # pd_op.depthwise_conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x1x7x7xf32)
        depthwise_conv2d_10 = paddle._C_ops.depthwise_conv2d(
            add_34, parameter_191, [1, 1], [3, 3], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del parameter_191

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_29 = paddle._C_ops.reshape(parameter_190, full_int_array_0)
        del parameter_190

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        add_35 = paddle._C_ops.add(depthwise_conv2d_10, reshape_29)
        del depthwise_conv2d_10, reshape_29

        # pd_op.batch_norm_: (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
                parameter_189,
                parameter_188,
                parameter_187,
                parameter_186,
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
        del add_35, parameter_186, parameter_187, parameter_188, parameter_189

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x96x14x14xf32, 384x96x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            batch_norm__84, parameter_185, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_185

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_30 = paddle._C_ops.reshape(parameter_184, full_int_array_0)
        del parameter_184

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        add_36 = paddle._C_ops.add(conv2d_19, reshape_30)
        del conv2d_19, reshape_30

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x96x14x14xf32, 384x96x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            batch_norm__84, parameter_183, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del batch_norm__84, parameter_183

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_31 = paddle._C_ops.reshape(parameter_182, full_int_array_0)
        del parameter_182

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        add_37 = paddle._C_ops.add(conv2d_20, reshape_31)
        del conv2d_20, reshape_31

        # pd_op.relu6: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        relu6_6 = paddle._C_ops.relu6(add_36)
        del add_36

        # pd_op.multiply: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        multiply_5 = paddle._C_ops.multiply(relu6_6, add_37)
        del add_37, relu6_6

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x384x14x14xf32, 96x384x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            multiply_5, parameter_181, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_5, parameter_181

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_32 = paddle._C_ops.reshape(parameter_180, full_int_array_0)
        del parameter_180

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        add_38 = paddle._C_ops.add(conv2d_21, reshape_32)
        del conv2d_21, reshape_32

        # pd_op.batch_norm_: (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
                parameter_179,
                parameter_178,
                parameter_177,
                parameter_176,
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
        del add_38, parameter_176, parameter_177, parameter_178, parameter_179

        # pd_op.depthwise_conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x1x7x7xf32)
        depthwise_conv2d_11 = paddle._C_ops.depthwise_conv2d(
            batch_norm__90,
            parameter_175,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            96,
            [1, 1],
            "NCHW",
        )
        del batch_norm__90, parameter_175

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_33 = paddle._C_ops.reshape(parameter_174, full_int_array_0)
        del parameter_174

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        add_39 = paddle._C_ops.add(depthwise_conv2d_11, reshape_33)
        del depthwise_conv2d_11, reshape_33

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, -1x96x14x14xf32)
        add_40 = paddle._C_ops.add(add_34, add_39)
        del add_34, add_39

        # pd_op.depthwise_conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x1x7x7xf32)
        depthwise_conv2d_12 = paddle._C_ops.depthwise_conv2d(
            add_40, parameter_173, [1, 1], [3, 3], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del parameter_173

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_34 = paddle._C_ops.reshape(parameter_172, full_int_array_0)
        del parameter_172

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        add_41 = paddle._C_ops.add(depthwise_conv2d_12, reshape_34)
        del depthwise_conv2d_12, reshape_34

        # pd_op.batch_norm_: (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
        del add_41, parameter_168, parameter_169, parameter_170, parameter_171

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x96x14x14xf32, 384x96x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(
            batch_norm__96, parameter_167, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_167

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_35 = paddle._C_ops.reshape(parameter_166, full_int_array_0)
        del parameter_166

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        add_42 = paddle._C_ops.add(conv2d_22, reshape_35)
        del conv2d_22, reshape_35

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x96x14x14xf32, 384x96x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(
            batch_norm__96, parameter_165, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del batch_norm__96, parameter_165

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_36 = paddle._C_ops.reshape(parameter_164, full_int_array_0)
        del parameter_164

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        add_43 = paddle._C_ops.add(conv2d_23, reshape_36)
        del conv2d_23, reshape_36

        # pd_op.relu6: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        relu6_7 = paddle._C_ops.relu6(add_42)
        del add_42

        # pd_op.multiply: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        multiply_6 = paddle._C_ops.multiply(relu6_7, add_43)
        del add_43, relu6_7

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x384x14x14xf32, 96x384x1x1xf32)
        conv2d_24 = paddle._C_ops.conv2d(
            multiply_6, parameter_163, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_6, parameter_163

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_37 = paddle._C_ops.reshape(parameter_162, full_int_array_0)
        del parameter_162

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        add_44 = paddle._C_ops.add(conv2d_24, reshape_37)
        del conv2d_24, reshape_37

        # pd_op.batch_norm_: (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
                parameter_161,
                parameter_160,
                parameter_159,
                parameter_158,
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
        del add_44, parameter_158, parameter_159, parameter_160, parameter_161

        # pd_op.depthwise_conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x1x7x7xf32)
        depthwise_conv2d_13 = paddle._C_ops.depthwise_conv2d(
            batch_norm__102,
            parameter_157,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            96,
            [1, 1],
            "NCHW",
        )
        del batch_norm__102, parameter_157

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_38 = paddle._C_ops.reshape(parameter_156, full_int_array_0)
        del parameter_156

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        add_45 = paddle._C_ops.add(depthwise_conv2d_13, reshape_38)
        del depthwise_conv2d_13, reshape_38

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, -1x96x14x14xf32)
        add_46 = paddle._C_ops.add(add_40, add_45)
        del add_40, add_45

        # pd_op.depthwise_conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x1x7x7xf32)
        depthwise_conv2d_14 = paddle._C_ops.depthwise_conv2d(
            add_46, parameter_155, [1, 1], [3, 3], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del parameter_155

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_39 = paddle._C_ops.reshape(parameter_154, full_int_array_0)
        del parameter_154

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        add_47 = paddle._C_ops.add(depthwise_conv2d_14, reshape_39)
        del depthwise_conv2d_14, reshape_39

        # pd_op.batch_norm_: (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
        del add_47, parameter_150, parameter_151, parameter_152, parameter_153

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x96x14x14xf32, 384x96x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(
            batch_norm__108,
            parameter_149,
            [1, 1],
            [0, 0],
            "EXPLICIT",
            [1, 1],
            1,
            "NCHW",
        )
        del parameter_149

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_40 = paddle._C_ops.reshape(parameter_148, full_int_array_0)
        del parameter_148

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        add_48 = paddle._C_ops.add(conv2d_25, reshape_40)
        del conv2d_25, reshape_40

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x96x14x14xf32, 384x96x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(
            batch_norm__108,
            parameter_147,
            [1, 1],
            [0, 0],
            "EXPLICIT",
            [1, 1],
            1,
            "NCHW",
        )
        del batch_norm__108, parameter_147

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_41 = paddle._C_ops.reshape(parameter_146, full_int_array_0)
        del parameter_146

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        add_49 = paddle._C_ops.add(conv2d_26, reshape_41)
        del conv2d_26, reshape_41

        # pd_op.relu6: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        relu6_8 = paddle._C_ops.relu6(add_48)
        del add_48

        # pd_op.multiply: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        multiply_7 = paddle._C_ops.multiply(relu6_8, add_49)
        del add_49, relu6_8

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x384x14x14xf32, 96x384x1x1xf32)
        conv2d_27 = paddle._C_ops.conv2d(
            multiply_7, parameter_145, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_7, parameter_145

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_42 = paddle._C_ops.reshape(parameter_144, full_int_array_0)
        del parameter_144

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        add_50 = paddle._C_ops.add(conv2d_27, reshape_42)
        del conv2d_27, reshape_42

        # pd_op.batch_norm_: (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
        del add_50, parameter_140, parameter_141, parameter_142, parameter_143

        # pd_op.depthwise_conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x1x7x7xf32)
        depthwise_conv2d_15 = paddle._C_ops.depthwise_conv2d(
            batch_norm__114,
            parameter_139,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            96,
            [1, 1],
            "NCHW",
        )
        del batch_norm__114, parameter_139

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_43 = paddle._C_ops.reshape(parameter_138, full_int_array_0)
        del parameter_138

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        add_51 = paddle._C_ops.add(depthwise_conv2d_15, reshape_43)
        del depthwise_conv2d_15, reshape_43

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, -1x96x14x14xf32)
        add_52 = paddle._C_ops.add(add_46, add_51)
        del add_46, add_51

        # pd_op.depthwise_conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x1x7x7xf32)
        depthwise_conv2d_16 = paddle._C_ops.depthwise_conv2d(
            add_52, parameter_137, [1, 1], [3, 3], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del parameter_137

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_44 = paddle._C_ops.reshape(parameter_136, full_int_array_0)
        del parameter_136

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        add_53 = paddle._C_ops.add(depthwise_conv2d_16, reshape_44)
        del depthwise_conv2d_16, reshape_44

        # pd_op.batch_norm_: (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
        del add_53, parameter_132, parameter_133, parameter_134, parameter_135

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x96x14x14xf32, 384x96x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(
            batch_norm__120,
            parameter_131,
            [1, 1],
            [0, 0],
            "EXPLICIT",
            [1, 1],
            1,
            "NCHW",
        )
        del parameter_131

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_45 = paddle._C_ops.reshape(parameter_130, full_int_array_0)
        del parameter_130

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        add_54 = paddle._C_ops.add(conv2d_28, reshape_45)
        del conv2d_28, reshape_45

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x96x14x14xf32, 384x96x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(
            batch_norm__120,
            parameter_129,
            [1, 1],
            [0, 0],
            "EXPLICIT",
            [1, 1],
            1,
            "NCHW",
        )
        del batch_norm__120, parameter_129

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_46 = paddle._C_ops.reshape(parameter_128, full_int_array_0)
        del parameter_128

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        add_55 = paddle._C_ops.add(conv2d_29, reshape_46)
        del conv2d_29, reshape_46

        # pd_op.relu6: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        relu6_9 = paddle._C_ops.relu6(add_54)
        del add_54

        # pd_op.multiply: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        multiply_8 = paddle._C_ops.multiply(relu6_9, add_55)
        del add_55, relu6_9

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x384x14x14xf32, 96x384x1x1xf32)
        conv2d_30 = paddle._C_ops.conv2d(
            multiply_8, parameter_127, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_8, parameter_127

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_47 = paddle._C_ops.reshape(parameter_126, full_int_array_0)
        del parameter_126

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        add_56 = paddle._C_ops.add(conv2d_30, reshape_47)
        del conv2d_30, reshape_47

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
                add_56,
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
        del add_56, parameter_122, parameter_123, parameter_124, parameter_125

        # pd_op.depthwise_conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x1x7x7xf32)
        depthwise_conv2d_17 = paddle._C_ops.depthwise_conv2d(
            batch_norm__126,
            parameter_121,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            96,
            [1, 1],
            "NCHW",
        )
        del batch_norm__126, parameter_121

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_48 = paddle._C_ops.reshape(parameter_120, full_int_array_0)
        del parameter_120

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        add_57 = paddle._C_ops.add(depthwise_conv2d_17, reshape_48)
        del depthwise_conv2d_17, reshape_48

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, -1x96x14x14xf32)
        add_58 = paddle._C_ops.add(add_52, add_57)
        del add_52, add_57

        # pd_op.depthwise_conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x1x7x7xf32)
        depthwise_conv2d_18 = paddle._C_ops.depthwise_conv2d(
            add_58, parameter_119, [1, 1], [3, 3], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del parameter_119

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_49 = paddle._C_ops.reshape(parameter_118, full_int_array_0)
        del parameter_118

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        add_59 = paddle._C_ops.add(depthwise_conv2d_18, reshape_49)
        del depthwise_conv2d_18, reshape_49

        # pd_op.batch_norm_: (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
        del add_59, parameter_114, parameter_115, parameter_116, parameter_117

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x96x14x14xf32, 384x96x1x1xf32)
        conv2d_31 = paddle._C_ops.conv2d(
            batch_norm__132,
            parameter_113,
            [1, 1],
            [0, 0],
            "EXPLICIT",
            [1, 1],
            1,
            "NCHW",
        )
        del parameter_113

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_50 = paddle._C_ops.reshape(parameter_112, full_int_array_0)
        del parameter_112

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        add_60 = paddle._C_ops.add(conv2d_31, reshape_50)
        del conv2d_31, reshape_50

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x96x14x14xf32, 384x96x1x1xf32)
        conv2d_32 = paddle._C_ops.conv2d(
            batch_norm__132,
            parameter_111,
            [1, 1],
            [0, 0],
            "EXPLICIT",
            [1, 1],
            1,
            "NCHW",
        )
        del batch_norm__132, parameter_111

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_51 = paddle._C_ops.reshape(parameter_110, full_int_array_0)
        del parameter_110

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        add_61 = paddle._C_ops.add(conv2d_32, reshape_51)
        del conv2d_32, reshape_51

        # pd_op.relu6: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        relu6_10 = paddle._C_ops.relu6(add_60)
        del add_60

        # pd_op.multiply: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        multiply_9 = paddle._C_ops.multiply(relu6_10, add_61)
        del add_61, relu6_10

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x384x14x14xf32, 96x384x1x1xf32)
        conv2d_33 = paddle._C_ops.conv2d(
            multiply_9, parameter_109, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_9, parameter_109

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_52 = paddle._C_ops.reshape(parameter_108, full_int_array_0)
        del parameter_108

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        add_62 = paddle._C_ops.add(conv2d_33, reshape_52)
        del conv2d_33, reshape_52

        # pd_op.batch_norm_: (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__138,
            batch_norm__139,
            batch_norm__140,
            batch_norm__141,
            batch_norm__142,
            batch_norm__143,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_62,
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
        del add_62, parameter_104, parameter_105, parameter_106, parameter_107

        # pd_op.depthwise_conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x1x7x7xf32)
        depthwise_conv2d_19 = paddle._C_ops.depthwise_conv2d(
            batch_norm__138,
            parameter_103,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            96,
            [1, 1],
            "NCHW",
        )
        del batch_norm__138, parameter_103

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_53 = paddle._C_ops.reshape(parameter_102, full_int_array_0)
        del parameter_102

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        add_63 = paddle._C_ops.add(depthwise_conv2d_19, reshape_53)
        del depthwise_conv2d_19, reshape_53

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, -1x96x14x14xf32)
        add_64 = paddle._C_ops.add(add_58, add_63)
        del add_58, add_63

        # pd_op.depthwise_conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x1x7x7xf32)
        depthwise_conv2d_20 = paddle._C_ops.depthwise_conv2d(
            add_64, parameter_101, [1, 1], [3, 3], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del parameter_101

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_54 = paddle._C_ops.reshape(parameter_100, full_int_array_0)
        del parameter_100

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        add_65 = paddle._C_ops.add(depthwise_conv2d_20, reshape_54)
        del depthwise_conv2d_20, reshape_54

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
                add_65,
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
        del add_65, parameter_96, parameter_97, parameter_98, parameter_99

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x96x14x14xf32, 384x96x1x1xf32)
        conv2d_34 = paddle._C_ops.conv2d(
            batch_norm__144, parameter_95, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_95

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_55 = paddle._C_ops.reshape(parameter_94, full_int_array_0)
        del parameter_94

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        add_66 = paddle._C_ops.add(conv2d_34, reshape_55)
        del conv2d_34, reshape_55

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x96x14x14xf32, 384x96x1x1xf32)
        conv2d_35 = paddle._C_ops.conv2d(
            batch_norm__144, parameter_93, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del batch_norm__144, parameter_93

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_56 = paddle._C_ops.reshape(parameter_92, full_int_array_0)
        del parameter_92

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        add_67 = paddle._C_ops.add(conv2d_35, reshape_56)
        del conv2d_35, reshape_56

        # pd_op.relu6: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        relu6_11 = paddle._C_ops.relu6(add_66)
        del add_66

        # pd_op.multiply: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        multiply_10 = paddle._C_ops.multiply(relu6_11, add_67)
        del add_67, relu6_11

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x384x14x14xf32, 96x384x1x1xf32)
        conv2d_36 = paddle._C_ops.conv2d(
            multiply_10, parameter_91, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_10, parameter_91

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_57 = paddle._C_ops.reshape(parameter_90, full_int_array_0)
        del parameter_90

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        add_68 = paddle._C_ops.add(conv2d_36, reshape_57)
        del conv2d_36, reshape_57

        # pd_op.batch_norm_: (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__150,
            batch_norm__151,
            batch_norm__152,
            batch_norm__153,
            batch_norm__154,
            batch_norm__155,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_68,
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
        del add_68, parameter_86, parameter_87, parameter_88, parameter_89

        # pd_op.depthwise_conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x1x7x7xf32)
        depthwise_conv2d_21 = paddle._C_ops.depthwise_conv2d(
            batch_norm__150,
            parameter_85,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            96,
            [1, 1],
            "NCHW",
        )
        del batch_norm__150, parameter_85

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_58 = paddle._C_ops.reshape(parameter_84, full_int_array_0)
        del parameter_84

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        add_69 = paddle._C_ops.add(depthwise_conv2d_21, reshape_58)
        del depthwise_conv2d_21, reshape_58

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, -1x96x14x14xf32)
        add_70 = paddle._C_ops.add(add_64, add_69)
        del add_64, add_69

        # pd_op.depthwise_conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x1x7x7xf32)
        depthwise_conv2d_22 = paddle._C_ops.depthwise_conv2d(
            add_70, parameter_83, [1, 1], [3, 3], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del parameter_83

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_59 = paddle._C_ops.reshape(parameter_82, full_int_array_0)
        del parameter_82

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        add_71 = paddle._C_ops.add(depthwise_conv2d_22, reshape_59)
        del depthwise_conv2d_22, reshape_59

        # pd_op.batch_norm_: (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__156,
            batch_norm__157,
            batch_norm__158,
            batch_norm__159,
            batch_norm__160,
            batch_norm__161,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_71,
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
        del add_71, parameter_78, parameter_79, parameter_80, parameter_81

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x96x14x14xf32, 384x96x1x1xf32)
        conv2d_37 = paddle._C_ops.conv2d(
            batch_norm__156, parameter_77, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_77

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_60 = paddle._C_ops.reshape(parameter_76, full_int_array_0)
        del parameter_76

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        add_72 = paddle._C_ops.add(conv2d_37, reshape_60)
        del conv2d_37, reshape_60

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x96x14x14xf32, 384x96x1x1xf32)
        conv2d_38 = paddle._C_ops.conv2d(
            batch_norm__156, parameter_75, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del batch_norm__156, parameter_75

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_61 = paddle._C_ops.reshape(parameter_74, full_int_array_0)
        del parameter_74

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        add_73 = paddle._C_ops.add(conv2d_38, reshape_61)
        del conv2d_38, reshape_61

        # pd_op.relu6: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        relu6_12 = paddle._C_ops.relu6(add_72)
        del add_72

        # pd_op.multiply: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        multiply_11 = paddle._C_ops.multiply(relu6_12, add_73)
        del add_73, relu6_12

        # pd_op.conv2d: (-1x96x14x14xf32) <- (-1x384x14x14xf32, 96x384x1x1xf32)
        conv2d_39 = paddle._C_ops.conv2d(
            multiply_11, parameter_73, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_11, parameter_73

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_62 = paddle._C_ops.reshape(parameter_72, full_int_array_0)
        del parameter_72

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        add_74 = paddle._C_ops.add(conv2d_39, reshape_62)
        del conv2d_39, reshape_62

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
                add_74,
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
        del add_74, parameter_68, parameter_69, parameter_70, parameter_71

        # pd_op.depthwise_conv2d: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 96x1x7x7xf32)
        depthwise_conv2d_23 = paddle._C_ops.depthwise_conv2d(
            batch_norm__162,
            parameter_67,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            96,
            [1, 1],
            "NCHW",
        )
        del batch_norm__162, parameter_67

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_63 = paddle._C_ops.reshape(parameter_66, full_int_array_0)
        del parameter_66

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, 1x96x1x1xf32)
        add_75 = paddle._C_ops.add(depthwise_conv2d_23, reshape_63)
        del depthwise_conv2d_23, reshape_63

        # pd_op.add: (-1x96x14x14xf32) <- (-1x96x14x14xf32, -1x96x14x14xf32)
        add_76 = paddle._C_ops.add(add_70, add_75)
        del add_70, add_75

        # pd_op.conv2d: (-1x192x7x7xf32) <- (-1x96x14x14xf32, 192x96x3x3xf32)
        conv2d_40 = paddle._C_ops.conv2d(
            add_76, parameter_65, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_76, parameter_65

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        reshape_64 = paddle._C_ops.reshape(parameter_64, full_int_array_0)
        del parameter_64

        # pd_op.add: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 1x192x1x1xf32)
        add_77 = paddle._C_ops.add(conv2d_40, reshape_64)
        del conv2d_40, reshape_64

        # pd_op.batch_norm_: (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__168,
            batch_norm__169,
            batch_norm__170,
            batch_norm__171,
            batch_norm__172,
            batch_norm__173,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_77,
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
        del add_77, parameter_60, parameter_61, parameter_62, parameter_63

        # pd_op.depthwise_conv2d: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 192x1x7x7xf32)
        depthwise_conv2d_24 = paddle._C_ops.depthwise_conv2d(
            batch_norm__168,
            parameter_59,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            192,
            [1, 1],
            "NCHW",
        )
        del parameter_59

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        reshape_65 = paddle._C_ops.reshape(parameter_58, full_int_array_0)
        del parameter_58

        # pd_op.add: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 1x192x1x1xf32)
        add_78 = paddle._C_ops.add(depthwise_conv2d_24, reshape_65)
        del depthwise_conv2d_24, reshape_65

        # pd_op.batch_norm_: (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__174,
            batch_norm__175,
            batch_norm__176,
            batch_norm__177,
            batch_norm__178,
            batch_norm__179,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_78,
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
        del add_78, parameter_54, parameter_55, parameter_56, parameter_57

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x192x7x7xf32, 768x192x1x1xf32)
        conv2d_41 = paddle._C_ops.conv2d(
            batch_norm__174, parameter_53, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_53

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_66 = paddle._C_ops.reshape(parameter_52, full_int_array_0)
        del parameter_52

        # pd_op.add: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 1x768x1x1xf32)
        add_79 = paddle._C_ops.add(conv2d_41, reshape_66)
        del conv2d_41, reshape_66

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x192x7x7xf32, 768x192x1x1xf32)
        conv2d_42 = paddle._C_ops.conv2d(
            batch_norm__174, parameter_51, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del batch_norm__174, parameter_51

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_67 = paddle._C_ops.reshape(parameter_50, full_int_array_0)
        del parameter_50

        # pd_op.add: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 1x768x1x1xf32)
        add_80 = paddle._C_ops.add(conv2d_42, reshape_67)
        del conv2d_42, reshape_67

        # pd_op.relu6: (-1x768x7x7xf32) <- (-1x768x7x7xf32)
        relu6_13 = paddle._C_ops.relu6(add_79)
        del add_79

        # pd_op.multiply: (-1x768x7x7xf32) <- (-1x768x7x7xf32, -1x768x7x7xf32)
        multiply_12 = paddle._C_ops.multiply(relu6_13, add_80)
        del add_80, relu6_13

        # pd_op.conv2d: (-1x192x7x7xf32) <- (-1x768x7x7xf32, 192x768x1x1xf32)
        conv2d_43 = paddle._C_ops.conv2d(
            multiply_12, parameter_49, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_12, parameter_49

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        reshape_68 = paddle._C_ops.reshape(parameter_48, full_int_array_0)
        del parameter_48

        # pd_op.add: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 1x192x1x1xf32)
        add_81 = paddle._C_ops.add(conv2d_43, reshape_68)
        del conv2d_43, reshape_68

        # pd_op.batch_norm_: (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__180,
            batch_norm__181,
            batch_norm__182,
            batch_norm__183,
            batch_norm__184,
            batch_norm__185,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_81,
                parameter_47,
                parameter_46,
                parameter_45,
                parameter_44,
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
        del add_81, parameter_44, parameter_45, parameter_46, parameter_47

        # pd_op.depthwise_conv2d: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 192x1x7x7xf32)
        depthwise_conv2d_25 = paddle._C_ops.depthwise_conv2d(
            batch_norm__180,
            parameter_43,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            192,
            [1, 1],
            "NCHW",
        )
        del batch_norm__180, parameter_43

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        reshape_69 = paddle._C_ops.reshape(parameter_42, full_int_array_0)
        del parameter_42

        # pd_op.add: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 1x192x1x1xf32)
        add_82 = paddle._C_ops.add(depthwise_conv2d_25, reshape_69)
        del depthwise_conv2d_25, reshape_69

        # pd_op.add: (-1x192x7x7xf32) <- (-1x192x7x7xf32, -1x192x7x7xf32)
        add_83 = paddle._C_ops.add(batch_norm__168, add_82)
        del add_82, batch_norm__168

        # pd_op.depthwise_conv2d: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 192x1x7x7xf32)
        depthwise_conv2d_26 = paddle._C_ops.depthwise_conv2d(
            add_83, parameter_41, [1, 1], [3, 3], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del parameter_41

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        reshape_70 = paddle._C_ops.reshape(parameter_40, full_int_array_0)
        del parameter_40

        # pd_op.add: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 1x192x1x1xf32)
        add_84 = paddle._C_ops.add(depthwise_conv2d_26, reshape_70)
        del depthwise_conv2d_26, reshape_70

        # pd_op.batch_norm_: (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__186,
            batch_norm__187,
            batch_norm__188,
            batch_norm__189,
            batch_norm__190,
            batch_norm__191,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_84,
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
        del add_84, parameter_36, parameter_37, parameter_38, parameter_39

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x192x7x7xf32, 768x192x1x1xf32)
        conv2d_44 = paddle._C_ops.conv2d(
            batch_norm__186, parameter_35, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_35

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_71 = paddle._C_ops.reshape(parameter_34, full_int_array_0)
        del parameter_34

        # pd_op.add: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 1x768x1x1xf32)
        add_85 = paddle._C_ops.add(conv2d_44, reshape_71)
        del conv2d_44, reshape_71

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x192x7x7xf32, 768x192x1x1xf32)
        conv2d_45 = paddle._C_ops.conv2d(
            batch_norm__186, parameter_33, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del batch_norm__186, parameter_33

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_72 = paddle._C_ops.reshape(parameter_32, full_int_array_0)
        del parameter_32

        # pd_op.add: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 1x768x1x1xf32)
        add_86 = paddle._C_ops.add(conv2d_45, reshape_72)
        del conv2d_45, reshape_72

        # pd_op.relu6: (-1x768x7x7xf32) <- (-1x768x7x7xf32)
        relu6_14 = paddle._C_ops.relu6(add_85)
        del add_85

        # pd_op.multiply: (-1x768x7x7xf32) <- (-1x768x7x7xf32, -1x768x7x7xf32)
        multiply_13 = paddle._C_ops.multiply(relu6_14, add_86)
        del add_86, relu6_14

        # pd_op.conv2d: (-1x192x7x7xf32) <- (-1x768x7x7xf32, 192x768x1x1xf32)
        conv2d_46 = paddle._C_ops.conv2d(
            multiply_13, parameter_31, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_13, parameter_31

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        reshape_73 = paddle._C_ops.reshape(parameter_30, full_int_array_0)
        del parameter_30

        # pd_op.add: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 1x192x1x1xf32)
        add_87 = paddle._C_ops.add(conv2d_46, reshape_73)
        del conv2d_46, reshape_73

        # pd_op.batch_norm_: (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__192,
            batch_norm__193,
            batch_norm__194,
            batch_norm__195,
            batch_norm__196,
            batch_norm__197,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_87,
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
        del add_87, parameter_26, parameter_27, parameter_28, parameter_29

        # pd_op.depthwise_conv2d: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 192x1x7x7xf32)
        depthwise_conv2d_27 = paddle._C_ops.depthwise_conv2d(
            batch_norm__192,
            parameter_25,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            192,
            [1, 1],
            "NCHW",
        )
        del batch_norm__192, parameter_25

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        reshape_74 = paddle._C_ops.reshape(parameter_24, full_int_array_0)
        del parameter_24

        # pd_op.add: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 1x192x1x1xf32)
        add_88 = paddle._C_ops.add(depthwise_conv2d_27, reshape_74)
        del depthwise_conv2d_27, reshape_74

        # pd_op.add: (-1x192x7x7xf32) <- (-1x192x7x7xf32, -1x192x7x7xf32)
        add_89 = paddle._C_ops.add(add_83, add_88)
        del add_83, add_88

        # pd_op.depthwise_conv2d: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 192x1x7x7xf32)
        depthwise_conv2d_28 = paddle._C_ops.depthwise_conv2d(
            add_89, parameter_23, [1, 1], [3, 3], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del parameter_23

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        reshape_75 = paddle._C_ops.reshape(parameter_22, full_int_array_0)
        del parameter_22

        # pd_op.add: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 1x192x1x1xf32)
        add_90 = paddle._C_ops.add(depthwise_conv2d_28, reshape_75)
        del depthwise_conv2d_28, reshape_75

        # pd_op.batch_norm_: (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__198,
            batch_norm__199,
            batch_norm__200,
            batch_norm__201,
            batch_norm__202,
            batch_norm__203,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_90,
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
        del add_90, parameter_18, parameter_19, parameter_20, parameter_21

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x192x7x7xf32, 768x192x1x1xf32)
        conv2d_47 = paddle._C_ops.conv2d(
            batch_norm__198, parameter_17, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_17

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_76 = paddle._C_ops.reshape(parameter_16, full_int_array_0)
        del parameter_16

        # pd_op.add: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 1x768x1x1xf32)
        add_91 = paddle._C_ops.add(conv2d_47, reshape_76)
        del conv2d_47, reshape_76

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x192x7x7xf32, 768x192x1x1xf32)
        conv2d_48 = paddle._C_ops.conv2d(
            batch_norm__198, parameter_15, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del batch_norm__198, parameter_15

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_77 = paddle._C_ops.reshape(parameter_14, full_int_array_0)
        del parameter_14

        # pd_op.add: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 1x768x1x1xf32)
        add_92 = paddle._C_ops.add(conv2d_48, reshape_77)
        del conv2d_48, reshape_77

        # pd_op.relu6: (-1x768x7x7xf32) <- (-1x768x7x7xf32)
        relu6_15 = paddle._C_ops.relu6(add_91)
        del add_91

        # pd_op.multiply: (-1x768x7x7xf32) <- (-1x768x7x7xf32, -1x768x7x7xf32)
        multiply_14 = paddle._C_ops.multiply(relu6_15, add_92)
        del add_92, relu6_15

        # pd_op.conv2d: (-1x192x7x7xf32) <- (-1x768x7x7xf32, 192x768x1x1xf32)
        conv2d_49 = paddle._C_ops.conv2d(
            multiply_14, parameter_13, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_14, parameter_13

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        reshape_78 = paddle._C_ops.reshape(parameter_12, full_int_array_0)
        del parameter_12

        # pd_op.add: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 1x192x1x1xf32)
        add_93 = paddle._C_ops.add(conv2d_49, reshape_78)
        del conv2d_49, reshape_78

        # pd_op.batch_norm_: (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__204,
            batch_norm__205,
            batch_norm__206,
            batch_norm__207,
            batch_norm__208,
            batch_norm__209,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_93,
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
        del add_93, parameter_10, parameter_11, parameter_8, parameter_9

        # pd_op.depthwise_conv2d: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 192x1x7x7xf32)
        depthwise_conv2d_29 = paddle._C_ops.depthwise_conv2d(
            batch_norm__204,
            parameter_7,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            192,
            [1, 1],
            "NCHW",
        )
        del batch_norm__204, parameter_7

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        reshape_79 = paddle._C_ops.reshape(parameter_6, full_int_array_0)
        del full_int_array_0, parameter_6

        # pd_op.add: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 1x192x1x1xf32)
        add_94 = paddle._C_ops.add(depthwise_conv2d_29, reshape_79)
        del depthwise_conv2d_29, reshape_79

        # pd_op.add: (-1x192x7x7xf32) <- (-1x192x7x7xf32, -1x192x7x7xf32)
        add_95 = paddle._C_ops.add(add_89, add_94)
        del add_89, add_94

        # pd_op.batch_norm_: (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__210,
            batch_norm__211,
            batch_norm__212,
            batch_norm__213,
            batch_norm__214,
            batch_norm__215,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_95,
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
        del add_95, parameter_2, parameter_3, parameter_4, parameter_5

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [1, 1]

        # pd_op.pool2d: (-1x192x1x1xf32) <- (-1x192x7x7xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            batch_norm__210,
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
        del batch_norm__210, full_int_array_1

        # pd_op.flatten: (-1x192xf32) <- (-1x192x1x1xf32)
        flatten_0 = paddle._C_ops.flatten(pool2d_0, 1, 3)
        del pool2d_0

        # pd_op.matmul: (-1x102xf32) <- (-1x192xf32, 192x102xf32)
        matmul_0 = paddle._C_ops.matmul(flatten_0, parameter_1, False, False)
        del flatten_0, parameter_1

        # pd_op.add: (-1x102xf32) <- (-1x102xf32, 102xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_0)
        del matmul_0, parameter_0

        return add_0
