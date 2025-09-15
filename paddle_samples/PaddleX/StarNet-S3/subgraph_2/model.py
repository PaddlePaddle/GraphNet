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
        data_0,
    ):
        # pd_op.conv2d: (-1x32x112x112xf32) <- (-1x3x224x224xf32, 32x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_323, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_0, parameter_323

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_322, full_int_array_0)
        del parameter_322

        # pd_op.add: (-1x32x112x112xf32) <- (-1x32x112x112xf32, 1x32x1x1xf32)
        add_1 = paddle._C_ops.add(conv2d_0, reshape_0)

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
                parameter_321,
                parameter_320,
                parameter_319,
                parameter_318,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_318, parameter_319, parameter_320, parameter_321

        # pd_op.relu6: (-1x32x112x112xf32) <- (-1x32x112x112xf32)
        relu6_0 = paddle._C_ops.relu6(batch_norm__0)
        del batch_norm__0

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x32x112x112xf32, 32x32x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            relu6_0, parameter_317, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_317

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_316, full_int_array_0)
        del parameter_316

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        add_2 = paddle._C_ops.add(conv2d_1, reshape_1)

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
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
                parameter_315,
                parameter_314,
                parameter_313,
                parameter_312,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_312, parameter_313, parameter_314, parameter_315

        # pd_op.depthwise_conv2d: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 32x1x7x7xf32)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(
            batch_norm__6, parameter_311, [1, 1], [3, 3], "EXPLICIT", 32, [1, 1], "NCHW"
        )
        del parameter_311

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(parameter_310, full_int_array_0)
        del parameter_310

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        add_3 = paddle._C_ops.add(depthwise_conv2d_0, reshape_2)

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
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
                parameter_309,
                parameter_308,
                parameter_307,
                parameter_306,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_306, parameter_307, parameter_308, parameter_309

        # pd_op.conv2d: (-1x128x56x56xf32) <- (-1x32x56x56xf32, 128x32x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            batch_norm__12, parameter_305, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_305

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(parameter_304, full_int_array_0)
        del parameter_304

        # pd_op.add: (-1x128x56x56xf32) <- (-1x128x56x56xf32, 1x128x1x1xf32)
        add_4 = paddle._C_ops.add(conv2d_2, reshape_3)

        # pd_op.conv2d: (-1x128x56x56xf32) <- (-1x32x56x56xf32, 128x32x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            batch_norm__12, parameter_303, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_303

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(parameter_302, full_int_array_0)
        del parameter_302

        # pd_op.add: (-1x128x56x56xf32) <- (-1x128x56x56xf32, 1x128x1x1xf32)
        add_5 = paddle._C_ops.add(conv2d_3, reshape_4)

        # pd_op.relu6: (-1x128x56x56xf32) <- (-1x128x56x56xf32)
        relu6_1 = paddle._C_ops.relu6(add_4)
        del add_4

        # pd_op.multiply: (-1x128x56x56xf32) <- (-1x128x56x56xf32, -1x128x56x56xf32)
        multiply_0 = paddle._C_ops.multiply(relu6_1, add_5)

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x128x56x56xf32, 32x128x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            multiply_0, parameter_301, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_301

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(parameter_300, full_int_array_0)
        del parameter_300

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        add_6 = paddle._C_ops.add(conv2d_4, reshape_5)

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
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

        # pd_op.depthwise_conv2d: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 32x1x7x7xf32)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(
            batch_norm__18,
            parameter_295,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            32,
            [1, 1],
            "NCHW",
        )
        del parameter_295

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(parameter_294, full_int_array_0)
        del parameter_294

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        add_7 = paddle._C_ops.add(depthwise_conv2d_1, reshape_6)

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, -1x32x56x56xf32)
        add_8 = paddle._C_ops.add(batch_norm__6, add_7)

        # pd_op.depthwise_conv2d: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 32x1x7x7xf32)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(
            add_8, parameter_293, [1, 1], [3, 3], "EXPLICIT", 32, [1, 1], "NCHW"
        )
        del parameter_293

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(parameter_292, full_int_array_0)
        del parameter_292

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        add_9 = paddle._C_ops.add(depthwise_conv2d_2, reshape_7)

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
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
                parameter_291,
                parameter_290,
                parameter_289,
                parameter_288,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_288, parameter_289, parameter_290, parameter_291

        # pd_op.conv2d: (-1x128x56x56xf32) <- (-1x32x56x56xf32, 128x32x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            batch_norm__24, parameter_287, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_287

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(parameter_286, full_int_array_0)
        del parameter_286

        # pd_op.add: (-1x128x56x56xf32) <- (-1x128x56x56xf32, 1x128x1x1xf32)
        add_10 = paddle._C_ops.add(conv2d_5, reshape_8)

        # pd_op.conv2d: (-1x128x56x56xf32) <- (-1x32x56x56xf32, 128x32x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            batch_norm__24, parameter_285, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_285

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(parameter_284, full_int_array_0)
        del parameter_284

        # pd_op.add: (-1x128x56x56xf32) <- (-1x128x56x56xf32, 1x128x1x1xf32)
        add_11 = paddle._C_ops.add(conv2d_6, reshape_9)

        # pd_op.relu6: (-1x128x56x56xf32) <- (-1x128x56x56xf32)
        relu6_2 = paddle._C_ops.relu6(add_10)
        del add_10

        # pd_op.multiply: (-1x128x56x56xf32) <- (-1x128x56x56xf32, -1x128x56x56xf32)
        multiply_1 = paddle._C_ops.multiply(relu6_2, add_11)

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x128x56x56xf32, 32x128x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            multiply_1, parameter_283, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_283

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(parameter_282, full_int_array_0)
        del parameter_282

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        add_12 = paddle._C_ops.add(conv2d_7, reshape_10)

        # pd_op.batch_norm_: (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
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

        # pd_op.depthwise_conv2d: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 32x1x7x7xf32)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(
            batch_norm__30,
            parameter_277,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            32,
            [1, 1],
            "NCHW",
        )
        del parameter_277

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        reshape_11 = paddle._C_ops.reshape(parameter_276, full_int_array_0)
        del parameter_276

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 1x32x1x1xf32)
        add_13 = paddle._C_ops.add(depthwise_conv2d_3, reshape_11)

        # pd_op.full: (xf32) <- ()
        full_0 = paddle._C_ops.full(
            [],
            float("0.999333"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x32x56x56xf32)
        shape64_0 = paddle._C_ops.shape64(add_13)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [1]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_0

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_0 = [slice_0, full_1, full_1, full_1]
        del slice_0

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_0 = paddle._C_ops.uniform(
            stack_0,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_0

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_14 = paddle._C_ops.add(full_0, uniform_0)
        del uniform_0

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_0 = paddle._C_ops.floor(add_14)
        del add_14

        # pd_op.divide: (-1x32x56x56xf32) <- (-1x32x56x56xf32, xf32)
        divide_0 = paddle._C_ops.divide(add_13, full_0)

        # pd_op.multiply: (-1x32x56x56xf32) <- (-1x32x56x56xf32, -1x1x1x1xf32)
        multiply_2 = paddle._C_ops.multiply(divide_0, floor_0)

        # pd_op.add: (-1x32x56x56xf32) <- (-1x32x56x56xf32, -1x32x56x56xf32)
        add_15 = paddle._C_ops.add(add_8, multiply_2)

        # pd_op.conv2d: (-1x64x28x28xf32) <- (-1x32x56x56xf32, 64x32x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            add_15, parameter_275, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_275

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(parameter_274, full_int_array_0)
        del parameter_274

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        add_16 = paddle._C_ops.add(conv2d_8, reshape_12)

        # pd_op.batch_norm_: (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__36,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_16,
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
        del parameter_270, parameter_271, parameter_272, parameter_273

        # pd_op.depthwise_conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x1x7x7xf32)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(
            batch_norm__36,
            parameter_269,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            64,
            [1, 1],
            "NCHW",
        )
        del parameter_269

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(parameter_268, full_int_array_0)
        del parameter_268

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        add_17 = paddle._C_ops.add(depthwise_conv2d_4, reshape_13)

        # pd_op.batch_norm_: (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__42,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_17,
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
        del parameter_264, parameter_265, parameter_266, parameter_267

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x64x28x28xf32, 256x64x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            batch_norm__42, parameter_263, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_263

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(parameter_262, full_int_array_0)
        del parameter_262

        # pd_op.add: (-1x256x28x28xf32) <- (-1x256x28x28xf32, 1x256x1x1xf32)
        add_18 = paddle._C_ops.add(conv2d_9, reshape_14)

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x64x28x28xf32, 256x64x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            batch_norm__42, parameter_261, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_261

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_15 = paddle._C_ops.reshape(parameter_260, full_int_array_0)
        del parameter_260

        # pd_op.add: (-1x256x28x28xf32) <- (-1x256x28x28xf32, 1x256x1x1xf32)
        add_19 = paddle._C_ops.add(conv2d_10, reshape_15)

        # pd_op.relu6: (-1x256x28x28xf32) <- (-1x256x28x28xf32)
        relu6_3 = paddle._C_ops.relu6(add_18)
        del add_18

        # pd_op.multiply: (-1x256x28x28xf32) <- (-1x256x28x28xf32, -1x256x28x28xf32)
        multiply_3 = paddle._C_ops.multiply(relu6_3, add_19)

        # pd_op.conv2d: (-1x64x28x28xf32) <- (-1x256x28x28xf32, 64x256x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            multiply_3, parameter_259, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_259

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_16 = paddle._C_ops.reshape(parameter_258, full_int_array_0)
        del parameter_258

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        add_20 = paddle._C_ops.add(conv2d_11, reshape_16)

        # pd_op.batch_norm_: (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__48,
            batch_norm__49,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_20,
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

        # pd_op.depthwise_conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x1x7x7xf32)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(
            batch_norm__48,
            parameter_253,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            64,
            [1, 1],
            "NCHW",
        )
        del parameter_253

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_17 = paddle._C_ops.reshape(parameter_252, full_int_array_0)
        del parameter_252

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        add_21 = paddle._C_ops.add(depthwise_conv2d_5, reshape_17)

        # pd_op.full: (xf32) <- ()
        full_4 = paddle._C_ops.full(
            [],
            float("0.998667"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x64x28x28xf32)
        shape64_1 = paddle._C_ops.shape64(add_21)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_1

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_1 = [slice_1, full_1, full_1, full_1]
        del slice_1

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_1, 0)
        del combine_1

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_1 = paddle._C_ops.uniform(
            stack_1,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_1

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_22 = paddle._C_ops.add(full_4, uniform_1)
        del uniform_1

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_1 = paddle._C_ops.floor(add_22)
        del add_22

        # pd_op.divide: (-1x64x28x28xf32) <- (-1x64x28x28xf32, xf32)
        divide_1 = paddle._C_ops.divide(add_21, full_4)

        # pd_op.multiply: (-1x64x28x28xf32) <- (-1x64x28x28xf32, -1x1x1x1xf32)
        multiply_4 = paddle._C_ops.multiply(divide_1, floor_1)

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, -1x64x28x28xf32)
        add_23 = paddle._C_ops.add(batch_norm__36, multiply_4)

        # pd_op.depthwise_conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x1x7x7xf32)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(
            add_23, parameter_251, [1, 1], [3, 3], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del parameter_251

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_18 = paddle._C_ops.reshape(parameter_250, full_int_array_0)
        del parameter_250

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        add_24 = paddle._C_ops.add(depthwise_conv2d_6, reshape_18)

        # pd_op.batch_norm_: (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_24,
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

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x64x28x28xf32, 256x64x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            batch_norm__54, parameter_245, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_245

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_19 = paddle._C_ops.reshape(parameter_244, full_int_array_0)
        del parameter_244

        # pd_op.add: (-1x256x28x28xf32) <- (-1x256x28x28xf32, 1x256x1x1xf32)
        add_25 = paddle._C_ops.add(conv2d_12, reshape_19)

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x64x28x28xf32, 256x64x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            batch_norm__54, parameter_243, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_243

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_20 = paddle._C_ops.reshape(parameter_242, full_int_array_0)
        del parameter_242

        # pd_op.add: (-1x256x28x28xf32) <- (-1x256x28x28xf32, 1x256x1x1xf32)
        add_26 = paddle._C_ops.add(conv2d_13, reshape_20)

        # pd_op.relu6: (-1x256x28x28xf32) <- (-1x256x28x28xf32)
        relu6_4 = paddle._C_ops.relu6(add_25)
        del add_25

        # pd_op.multiply: (-1x256x28x28xf32) <- (-1x256x28x28xf32, -1x256x28x28xf32)
        multiply_5 = paddle._C_ops.multiply(relu6_4, add_26)

        # pd_op.conv2d: (-1x64x28x28xf32) <- (-1x256x28x28xf32, 64x256x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            multiply_5, parameter_241, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_241

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_21 = paddle._C_ops.reshape(parameter_240, full_int_array_0)
        del parameter_240

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        add_27 = paddle._C_ops.add(conv2d_14, reshape_21)

        # pd_op.batch_norm_: (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_27,
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

        # pd_op.depthwise_conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x1x7x7xf32)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(
            batch_norm__60,
            parameter_235,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            64,
            [1, 1],
            "NCHW",
        )
        del parameter_235

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_22 = paddle._C_ops.reshape(parameter_234, full_int_array_0)
        del parameter_234

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 1x64x1x1xf32)
        add_28 = paddle._C_ops.add(depthwise_conv2d_7, reshape_22)

        # pd_op.full: (xf32) <- ()
        full_5 = paddle._C_ops.full(
            [],
            float("0.998"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x64x28x28xf32)
        shape64_2 = paddle._C_ops.shape64(add_28)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_2

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_2 = [slice_2, full_1, full_1, full_1]
        del slice_2

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_2 = paddle._C_ops.stack(combine_2, 0)
        del combine_2

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_2 = paddle._C_ops.uniform(
            stack_2,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_2

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_29 = paddle._C_ops.add(full_5, uniform_2)
        del uniform_2

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_2 = paddle._C_ops.floor(add_29)
        del add_29

        # pd_op.divide: (-1x64x28x28xf32) <- (-1x64x28x28xf32, xf32)
        divide_2 = paddle._C_ops.divide(add_28, full_5)

        # pd_op.multiply: (-1x64x28x28xf32) <- (-1x64x28x28xf32, -1x1x1x1xf32)
        multiply_6 = paddle._C_ops.multiply(divide_2, floor_2)

        # pd_op.add: (-1x64x28x28xf32) <- (-1x64x28x28xf32, -1x64x28x28xf32)
        add_30 = paddle._C_ops.add(add_23, multiply_6)

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x64x28x28xf32, 128x64x3x3xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            add_30, parameter_233, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_233

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_23 = paddle._C_ops.reshape(parameter_232, full_int_array_0)
        del parameter_232

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_31 = paddle._C_ops.add(conv2d_15, reshape_23)

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__66,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            batch_norm__70,
            batch_norm__71,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_31,
                parameter_231,
                parameter_230,
                parameter_229,
                parameter_228,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_228, parameter_229, parameter_230, parameter_231

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        depthwise_conv2d_8 = paddle._C_ops.depthwise_conv2d(
            batch_norm__66,
            parameter_227,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            128,
            [1, 1],
            "NCHW",
        )
        del parameter_227

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_24 = paddle._C_ops.reshape(parameter_226, full_int_array_0)
        del parameter_226

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_32 = paddle._C_ops.add(depthwise_conv2d_8, reshape_24)

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__72,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_32,
                parameter_225,
                parameter_224,
                parameter_223,
                parameter_222,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_222, parameter_223, parameter_224, parameter_225

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            batch_norm__72, parameter_221, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_221

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_25 = paddle._C_ops.reshape(parameter_220, full_int_array_0)
        del parameter_220

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        add_33 = paddle._C_ops.add(conv2d_16, reshape_25)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            batch_norm__72, parameter_219, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_219

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_26 = paddle._C_ops.reshape(parameter_218, full_int_array_0)
        del parameter_218

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        add_34 = paddle._C_ops.add(conv2d_17, reshape_26)

        # pd_op.relu6: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu6_5 = paddle._C_ops.relu6(add_33)
        del add_33

        # pd_op.multiply: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        multiply_7 = paddle._C_ops.multiply(relu6_5, add_34)

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x512x14x14xf32, 128x512x1x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            multiply_7, parameter_217, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_217

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_27 = paddle._C_ops.reshape(parameter_216, full_int_array_0)
        del parameter_216

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_35 = paddle._C_ops.add(conv2d_18, reshape_27)

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__78,
            batch_norm__79,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
            batch_norm__83,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_35,
                parameter_215,
                parameter_214,
                parameter_213,
                parameter_212,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_212, parameter_213, parameter_214, parameter_215

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        depthwise_conv2d_9 = paddle._C_ops.depthwise_conv2d(
            batch_norm__78,
            parameter_211,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            128,
            [1, 1],
            "NCHW",
        )
        del parameter_211

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_28 = paddle._C_ops.reshape(parameter_210, full_int_array_0)
        del parameter_210

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_36 = paddle._C_ops.add(depthwise_conv2d_9, reshape_28)

        # pd_op.full: (xf32) <- ()
        full_6 = paddle._C_ops.full(
            [],
            float("0.997333"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x128x14x14xf32)
        shape64_3 = paddle._C_ops.shape64(add_36)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_3

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_3 = [slice_3, full_1, full_1, full_1]
        del slice_3

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_3 = paddle._C_ops.stack(combine_3, 0)
        del combine_3

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_3 = paddle._C_ops.uniform(
            stack_3,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_3

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_37 = paddle._C_ops.add(full_6, uniform_3)
        del uniform_3

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_3 = paddle._C_ops.floor(add_37)
        del add_37

        # pd_op.divide: (-1x128x14x14xf32) <- (-1x128x14x14xf32, xf32)
        divide_3 = paddle._C_ops.divide(add_36, full_6)

        # pd_op.multiply: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x1x1x1xf32)
        multiply_8 = paddle._C_ops.multiply(divide_3, floor_3)

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x14x14xf32)
        add_38 = paddle._C_ops.add(batch_norm__66, multiply_8)

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        depthwise_conv2d_10 = paddle._C_ops.depthwise_conv2d(
            add_38, parameter_209, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del parameter_209

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_29 = paddle._C_ops.reshape(parameter_208, full_int_array_0)
        del parameter_208

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_39 = paddle._C_ops.add(depthwise_conv2d_10, reshape_29)

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__84,
            batch_norm__85,
            batch_norm__86,
            batch_norm__87,
            batch_norm__88,
            batch_norm__89,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_39,
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
        del parameter_204, parameter_205, parameter_206, parameter_207

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            batch_norm__84, parameter_203, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_203

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_30 = paddle._C_ops.reshape(parameter_202, full_int_array_0)
        del parameter_202

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        add_40 = paddle._C_ops.add(conv2d_19, reshape_30)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            batch_norm__84, parameter_201, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_201

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_31 = paddle._C_ops.reshape(parameter_200, full_int_array_0)
        del parameter_200

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        add_41 = paddle._C_ops.add(conv2d_20, reshape_31)

        # pd_op.relu6: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu6_6 = paddle._C_ops.relu6(add_40)
        del add_40

        # pd_op.multiply: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        multiply_9 = paddle._C_ops.multiply(relu6_6, add_41)

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x512x14x14xf32, 128x512x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            multiply_9, parameter_199, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_199

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_32 = paddle._C_ops.reshape(parameter_198, full_int_array_0)
        del parameter_198

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_42 = paddle._C_ops.add(conv2d_21, reshape_32)

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__90,
            batch_norm__91,
            batch_norm__92,
            batch_norm__93,
            batch_norm__94,
            batch_norm__95,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_42,
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
        del parameter_194, parameter_195, parameter_196, parameter_197

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        depthwise_conv2d_11 = paddle._C_ops.depthwise_conv2d(
            batch_norm__90,
            parameter_193,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            128,
            [1, 1],
            "NCHW",
        )
        del parameter_193

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_33 = paddle._C_ops.reshape(parameter_192, full_int_array_0)
        del parameter_192

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_43 = paddle._C_ops.add(depthwise_conv2d_11, reshape_33)

        # pd_op.full: (xf32) <- ()
        full_7 = paddle._C_ops.full(
            [],
            float("0.996667"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x128x14x14xf32)
        shape64_4 = paddle._C_ops.shape64(add_43)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            shape64_4, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_4

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_4 = [slice_4, full_1, full_1, full_1]
        del slice_4

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_4 = paddle._C_ops.stack(combine_4, 0)
        del combine_4

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_4 = paddle._C_ops.uniform(
            stack_4,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_4

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_44 = paddle._C_ops.add(full_7, uniform_4)
        del uniform_4

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_4 = paddle._C_ops.floor(add_44)
        del add_44

        # pd_op.divide: (-1x128x14x14xf32) <- (-1x128x14x14xf32, xf32)
        divide_4 = paddle._C_ops.divide(add_43, full_7)

        # pd_op.multiply: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x1x1x1xf32)
        multiply_10 = paddle._C_ops.multiply(divide_4, floor_4)

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x14x14xf32)
        add_45 = paddle._C_ops.add(add_38, multiply_10)

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        depthwise_conv2d_12 = paddle._C_ops.depthwise_conv2d(
            add_45, parameter_191, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del parameter_191

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_34 = paddle._C_ops.reshape(parameter_190, full_int_array_0)
        del parameter_190

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_46 = paddle._C_ops.add(depthwise_conv2d_12, reshape_34)

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__96,
            batch_norm__97,
            batch_norm__98,
            batch_norm__99,
            batch_norm__100,
            batch_norm__101,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_46,
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

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(
            batch_norm__96, parameter_185, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_185

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_35 = paddle._C_ops.reshape(parameter_184, full_int_array_0)
        del parameter_184

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        add_47 = paddle._C_ops.add(conv2d_22, reshape_35)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(
            batch_norm__96, parameter_183, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_183

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_36 = paddle._C_ops.reshape(parameter_182, full_int_array_0)
        del parameter_182

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        add_48 = paddle._C_ops.add(conv2d_23, reshape_36)

        # pd_op.relu6: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu6_7 = paddle._C_ops.relu6(add_47)
        del add_47

        # pd_op.multiply: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        multiply_11 = paddle._C_ops.multiply(relu6_7, add_48)

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x512x14x14xf32, 128x512x1x1xf32)
        conv2d_24 = paddle._C_ops.conv2d(
            multiply_11, parameter_181, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_181

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_37 = paddle._C_ops.reshape(parameter_180, full_int_array_0)
        del parameter_180

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_49 = paddle._C_ops.add(conv2d_24, reshape_37)

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__102,
            batch_norm__103,
            batch_norm__104,
            batch_norm__105,
            batch_norm__106,
            batch_norm__107,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_49,
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

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        depthwise_conv2d_13 = paddle._C_ops.depthwise_conv2d(
            batch_norm__102,
            parameter_175,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            128,
            [1, 1],
            "NCHW",
        )
        del parameter_175

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_38 = paddle._C_ops.reshape(parameter_174, full_int_array_0)
        del parameter_174

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_50 = paddle._C_ops.add(depthwise_conv2d_13, reshape_38)

        # pd_op.full: (xf32) <- ()
        full_8 = paddle._C_ops.full(
            [],
            float("0.996"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x128x14x14xf32)
        shape64_5 = paddle._C_ops.shape64(add_50)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            shape64_5, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_5

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_5 = [slice_5, full_1, full_1, full_1]
        del slice_5

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_5 = paddle._C_ops.stack(combine_5, 0)
        del combine_5

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_5 = paddle._C_ops.uniform(
            stack_5,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_5

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_51 = paddle._C_ops.add(full_8, uniform_5)
        del uniform_5

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_5 = paddle._C_ops.floor(add_51)
        del add_51

        # pd_op.divide: (-1x128x14x14xf32) <- (-1x128x14x14xf32, xf32)
        divide_5 = paddle._C_ops.divide(add_50, full_8)

        # pd_op.multiply: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x1x1x1xf32)
        multiply_12 = paddle._C_ops.multiply(divide_5, floor_5)

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x14x14xf32)
        add_52 = paddle._C_ops.add(add_45, multiply_12)

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        depthwise_conv2d_14 = paddle._C_ops.depthwise_conv2d(
            add_52, parameter_173, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del parameter_173

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_39 = paddle._C_ops.reshape(parameter_172, full_int_array_0)
        del parameter_172

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_53 = paddle._C_ops.add(depthwise_conv2d_14, reshape_39)

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__108,
            batch_norm__109,
            batch_norm__110,
            batch_norm__111,
            batch_norm__112,
            batch_norm__113,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_53,
                parameter_171,
                parameter_170,
                parameter_169,
                parameter_168,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_168, parameter_169, parameter_170, parameter_171

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(
            batch_norm__108,
            parameter_167,
            [1, 1],
            [0, 0],
            "EXPLICIT",
            [1, 1],
            1,
            "NCHW",
        )
        del parameter_167

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_40 = paddle._C_ops.reshape(parameter_166, full_int_array_0)
        del parameter_166

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        add_54 = paddle._C_ops.add(conv2d_25, reshape_40)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(
            batch_norm__108,
            parameter_165,
            [1, 1],
            [0, 0],
            "EXPLICIT",
            [1, 1],
            1,
            "NCHW",
        )
        del parameter_165

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_41 = paddle._C_ops.reshape(parameter_164, full_int_array_0)
        del parameter_164

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        add_55 = paddle._C_ops.add(conv2d_26, reshape_41)

        # pd_op.relu6: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu6_8 = paddle._C_ops.relu6(add_54)
        del add_54

        # pd_op.multiply: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        multiply_13 = paddle._C_ops.multiply(relu6_8, add_55)

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x512x14x14xf32, 128x512x1x1xf32)
        conv2d_27 = paddle._C_ops.conv2d(
            multiply_13, parameter_163, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_163

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_42 = paddle._C_ops.reshape(parameter_162, full_int_array_0)
        del parameter_162

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_56 = paddle._C_ops.add(conv2d_27, reshape_42)

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__114,
            batch_norm__115,
            batch_norm__116,
            batch_norm__117,
            batch_norm__118,
            batch_norm__119,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_56,
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

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        depthwise_conv2d_15 = paddle._C_ops.depthwise_conv2d(
            batch_norm__114,
            parameter_157,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            128,
            [1, 1],
            "NCHW",
        )
        del parameter_157

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_43 = paddle._C_ops.reshape(parameter_156, full_int_array_0)
        del parameter_156

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_57 = paddle._C_ops.add(depthwise_conv2d_15, reshape_43)

        # pd_op.full: (xf32) <- ()
        full_9 = paddle._C_ops.full(
            [],
            float("0.995333"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x128x14x14xf32)
        shape64_6 = paddle._C_ops.shape64(add_57)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            shape64_6, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_6

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_6 = [slice_6, full_1, full_1, full_1]
        del slice_6

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_6 = paddle._C_ops.stack(combine_6, 0)
        del combine_6

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_6 = paddle._C_ops.uniform(
            stack_6,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_6

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_58 = paddle._C_ops.add(full_9, uniform_6)
        del uniform_6

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_6 = paddle._C_ops.floor(add_58)
        del add_58

        # pd_op.divide: (-1x128x14x14xf32) <- (-1x128x14x14xf32, xf32)
        divide_6 = paddle._C_ops.divide(add_57, full_9)

        # pd_op.multiply: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x1x1x1xf32)
        multiply_14 = paddle._C_ops.multiply(divide_6, floor_6)

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x14x14xf32)
        add_59 = paddle._C_ops.add(add_52, multiply_14)

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        depthwise_conv2d_16 = paddle._C_ops.depthwise_conv2d(
            add_59, parameter_155, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del parameter_155

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_44 = paddle._C_ops.reshape(parameter_154, full_int_array_0)
        del parameter_154

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_60 = paddle._C_ops.add(depthwise_conv2d_16, reshape_44)

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__120,
            batch_norm__121,
            batch_norm__122,
            batch_norm__123,
            batch_norm__124,
            batch_norm__125,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_60,
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
        del parameter_150, parameter_151, parameter_152, parameter_153

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(
            batch_norm__120,
            parameter_149,
            [1, 1],
            [0, 0],
            "EXPLICIT",
            [1, 1],
            1,
            "NCHW",
        )
        del parameter_149

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_45 = paddle._C_ops.reshape(parameter_148, full_int_array_0)
        del parameter_148

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        add_61 = paddle._C_ops.add(conv2d_28, reshape_45)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(
            batch_norm__120,
            parameter_147,
            [1, 1],
            [0, 0],
            "EXPLICIT",
            [1, 1],
            1,
            "NCHW",
        )
        del parameter_147

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_46 = paddle._C_ops.reshape(parameter_146, full_int_array_0)
        del parameter_146

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        add_62 = paddle._C_ops.add(conv2d_29, reshape_46)

        # pd_op.relu6: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu6_9 = paddle._C_ops.relu6(add_61)
        del add_61

        # pd_op.multiply: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        multiply_15 = paddle._C_ops.multiply(relu6_9, add_62)

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x512x14x14xf32, 128x512x1x1xf32)
        conv2d_30 = paddle._C_ops.conv2d(
            multiply_15, parameter_145, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_145

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_47 = paddle._C_ops.reshape(parameter_144, full_int_array_0)
        del parameter_144

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_63 = paddle._C_ops.add(conv2d_30, reshape_47)

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__126,
            batch_norm__127,
            batch_norm__128,
            batch_norm__129,
            batch_norm__130,
            batch_norm__131,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_63,
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
        del parameter_140, parameter_141, parameter_142, parameter_143

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        depthwise_conv2d_17 = paddle._C_ops.depthwise_conv2d(
            batch_norm__126,
            parameter_139,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            128,
            [1, 1],
            "NCHW",
        )
        del parameter_139

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_48 = paddle._C_ops.reshape(parameter_138, full_int_array_0)
        del parameter_138

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_64 = paddle._C_ops.add(depthwise_conv2d_17, reshape_48)

        # pd_op.full: (xf32) <- ()
        full_10 = paddle._C_ops.full(
            [],
            float("0.994667"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x128x14x14xf32)
        shape64_7 = paddle._C_ops.shape64(add_64)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            shape64_7, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_7

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_7 = [slice_7, full_1, full_1, full_1]
        del slice_7

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_7 = paddle._C_ops.stack(combine_7, 0)
        del combine_7

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_7 = paddle._C_ops.uniform(
            stack_7,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_7

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_65 = paddle._C_ops.add(full_10, uniform_7)
        del uniform_7

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_7 = paddle._C_ops.floor(add_65)
        del add_65

        # pd_op.divide: (-1x128x14x14xf32) <- (-1x128x14x14xf32, xf32)
        divide_7 = paddle._C_ops.divide(add_64, full_10)

        # pd_op.multiply: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x1x1x1xf32)
        multiply_16 = paddle._C_ops.multiply(divide_7, floor_7)

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x14x14xf32)
        add_66 = paddle._C_ops.add(add_59, multiply_16)

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        depthwise_conv2d_18 = paddle._C_ops.depthwise_conv2d(
            add_66, parameter_137, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del parameter_137

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_49 = paddle._C_ops.reshape(parameter_136, full_int_array_0)
        del parameter_136

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_67 = paddle._C_ops.add(depthwise_conv2d_18, reshape_49)

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__132,
            batch_norm__133,
            batch_norm__134,
            batch_norm__135,
            batch_norm__136,
            batch_norm__137,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_67,
                parameter_135,
                parameter_134,
                parameter_133,
                parameter_132,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_132, parameter_133, parameter_134, parameter_135

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        conv2d_31 = paddle._C_ops.conv2d(
            batch_norm__132,
            parameter_131,
            [1, 1],
            [0, 0],
            "EXPLICIT",
            [1, 1],
            1,
            "NCHW",
        )
        del parameter_131

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_50 = paddle._C_ops.reshape(parameter_130, full_int_array_0)
        del parameter_130

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        add_68 = paddle._C_ops.add(conv2d_31, reshape_50)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        conv2d_32 = paddle._C_ops.conv2d(
            batch_norm__132,
            parameter_129,
            [1, 1],
            [0, 0],
            "EXPLICIT",
            [1, 1],
            1,
            "NCHW",
        )
        del parameter_129

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_51 = paddle._C_ops.reshape(parameter_128, full_int_array_0)
        del parameter_128

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        add_69 = paddle._C_ops.add(conv2d_32, reshape_51)

        # pd_op.relu6: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu6_10 = paddle._C_ops.relu6(add_68)
        del add_68

        # pd_op.multiply: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        multiply_17 = paddle._C_ops.multiply(relu6_10, add_69)

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x512x14x14xf32, 128x512x1x1xf32)
        conv2d_33 = paddle._C_ops.conv2d(
            multiply_17, parameter_127, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_127

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_52 = paddle._C_ops.reshape(parameter_126, full_int_array_0)
        del parameter_126

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_70 = paddle._C_ops.add(conv2d_33, reshape_52)

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__138,
            batch_norm__139,
            batch_norm__140,
            batch_norm__141,
            batch_norm__142,
            batch_norm__143,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_70,
                parameter_125,
                parameter_124,
                parameter_123,
                parameter_122,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_122, parameter_123, parameter_124, parameter_125

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        depthwise_conv2d_19 = paddle._C_ops.depthwise_conv2d(
            batch_norm__138,
            parameter_121,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            128,
            [1, 1],
            "NCHW",
        )
        del parameter_121

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_53 = paddle._C_ops.reshape(parameter_120, full_int_array_0)
        del parameter_120

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_71 = paddle._C_ops.add(depthwise_conv2d_19, reshape_53)

        # pd_op.full: (xf32) <- ()
        full_11 = paddle._C_ops.full(
            [],
            float("0.994"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x128x14x14xf32)
        shape64_8 = paddle._C_ops.shape64(add_71)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            shape64_8, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_8

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_8 = [slice_8, full_1, full_1, full_1]
        del slice_8

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_8 = paddle._C_ops.stack(combine_8, 0)
        del combine_8

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_8 = paddle._C_ops.uniform(
            stack_8,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_8

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_72 = paddle._C_ops.add(full_11, uniform_8)
        del uniform_8

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_8 = paddle._C_ops.floor(add_72)
        del add_72

        # pd_op.divide: (-1x128x14x14xf32) <- (-1x128x14x14xf32, xf32)
        divide_8 = paddle._C_ops.divide(add_71, full_11)

        # pd_op.multiply: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x1x1x1xf32)
        multiply_18 = paddle._C_ops.multiply(divide_8, floor_8)

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x14x14xf32)
        add_73 = paddle._C_ops.add(add_66, multiply_18)

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        depthwise_conv2d_20 = paddle._C_ops.depthwise_conv2d(
            add_73, parameter_119, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del parameter_119

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_54 = paddle._C_ops.reshape(parameter_118, full_int_array_0)
        del parameter_118

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_74 = paddle._C_ops.add(depthwise_conv2d_20, reshape_54)

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__144,
            batch_norm__145,
            batch_norm__146,
            batch_norm__147,
            batch_norm__148,
            batch_norm__149,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_74,
                parameter_117,
                parameter_116,
                parameter_115,
                parameter_114,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_114, parameter_115, parameter_116, parameter_117

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        conv2d_34 = paddle._C_ops.conv2d(
            batch_norm__144,
            parameter_113,
            [1, 1],
            [0, 0],
            "EXPLICIT",
            [1, 1],
            1,
            "NCHW",
        )
        del parameter_113

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_55 = paddle._C_ops.reshape(parameter_112, full_int_array_0)
        del parameter_112

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        add_75 = paddle._C_ops.add(conv2d_34, reshape_55)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        conv2d_35 = paddle._C_ops.conv2d(
            batch_norm__144,
            parameter_111,
            [1, 1],
            [0, 0],
            "EXPLICIT",
            [1, 1],
            1,
            "NCHW",
        )
        del parameter_111

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_56 = paddle._C_ops.reshape(parameter_110, full_int_array_0)
        del parameter_110

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        add_76 = paddle._C_ops.add(conv2d_35, reshape_56)

        # pd_op.relu6: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu6_11 = paddle._C_ops.relu6(add_75)
        del add_75

        # pd_op.multiply: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        multiply_19 = paddle._C_ops.multiply(relu6_11, add_76)

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x512x14x14xf32, 128x512x1x1xf32)
        conv2d_36 = paddle._C_ops.conv2d(
            multiply_19, parameter_109, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_109

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_57 = paddle._C_ops.reshape(parameter_108, full_int_array_0)
        del parameter_108

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_77 = paddle._C_ops.add(conv2d_36, reshape_57)

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__150,
            batch_norm__151,
            batch_norm__152,
            batch_norm__153,
            batch_norm__154,
            batch_norm__155,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_77,
                parameter_107,
                parameter_106,
                parameter_105,
                parameter_104,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_104, parameter_105, parameter_106, parameter_107

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        depthwise_conv2d_21 = paddle._C_ops.depthwise_conv2d(
            batch_norm__150,
            parameter_103,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            128,
            [1, 1],
            "NCHW",
        )
        del parameter_103

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_58 = paddle._C_ops.reshape(parameter_102, full_int_array_0)
        del parameter_102

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_78 = paddle._C_ops.add(depthwise_conv2d_21, reshape_58)

        # pd_op.full: (xf32) <- ()
        full_12 = paddle._C_ops.full(
            [],
            float("0.993333"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x128x14x14xf32)
        shape64_9 = paddle._C_ops.shape64(add_78)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            shape64_9, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_9

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_9 = [slice_9, full_1, full_1, full_1]
        del slice_9

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_9 = paddle._C_ops.stack(combine_9, 0)
        del combine_9

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_9 = paddle._C_ops.uniform(
            stack_9,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_9

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_79 = paddle._C_ops.add(full_12, uniform_9)
        del uniform_9

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_9 = paddle._C_ops.floor(add_79)
        del add_79

        # pd_op.divide: (-1x128x14x14xf32) <- (-1x128x14x14xf32, xf32)
        divide_9 = paddle._C_ops.divide(add_78, full_12)

        # pd_op.multiply: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x1x1x1xf32)
        multiply_20 = paddle._C_ops.multiply(divide_9, floor_9)

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x14x14xf32)
        add_80 = paddle._C_ops.add(add_73, multiply_20)

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        depthwise_conv2d_22 = paddle._C_ops.depthwise_conv2d(
            add_80, parameter_101, [1, 1], [3, 3], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del parameter_101

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_59 = paddle._C_ops.reshape(parameter_100, full_int_array_0)
        del parameter_100

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_81 = paddle._C_ops.add(depthwise_conv2d_22, reshape_59)

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__156,
            batch_norm__157,
            batch_norm__158,
            batch_norm__159,
            batch_norm__160,
            batch_norm__161,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_81,
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

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        conv2d_37 = paddle._C_ops.conv2d(
            batch_norm__156, parameter_95, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_95

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_60 = paddle._C_ops.reshape(parameter_94, full_int_array_0)
        del parameter_94

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        add_82 = paddle._C_ops.add(conv2d_37, reshape_60)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x128x14x14xf32, 512x128x1x1xf32)
        conv2d_38 = paddle._C_ops.conv2d(
            batch_norm__156, parameter_93, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_93

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_61 = paddle._C_ops.reshape(parameter_92, full_int_array_0)
        del parameter_92

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, 1x512x1x1xf32)
        add_83 = paddle._C_ops.add(conv2d_38, reshape_61)

        # pd_op.relu6: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu6_12 = paddle._C_ops.relu6(add_82)
        del add_82

        # pd_op.multiply: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        multiply_21 = paddle._C_ops.multiply(relu6_12, add_83)

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x512x14x14xf32, 128x512x1x1xf32)
        conv2d_39 = paddle._C_ops.conv2d(
            multiply_21, parameter_91, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_91

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_62 = paddle._C_ops.reshape(parameter_90, full_int_array_0)
        del parameter_90

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_84 = paddle._C_ops.add(conv2d_39, reshape_62)

        # pd_op.batch_norm_: (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x14x14xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__162,
            batch_norm__163,
            batch_norm__164,
            batch_norm__165,
            batch_norm__166,
            batch_norm__167,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_84,
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

        # pd_op.depthwise_conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x1x7x7xf32)
        depthwise_conv2d_23 = paddle._C_ops.depthwise_conv2d(
            batch_norm__162,
            parameter_85,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            128,
            [1, 1],
            "NCHW",
        )
        del parameter_85

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_63 = paddle._C_ops.reshape(parameter_84, full_int_array_0)
        del parameter_84

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 1x128x1x1xf32)
        add_85 = paddle._C_ops.add(depthwise_conv2d_23, reshape_63)

        # pd_op.full: (xf32) <- ()
        full_13 = paddle._C_ops.full(
            [],
            float("0.992667"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x128x14x14xf32)
        shape64_10 = paddle._C_ops.shape64(add_85)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            shape64_10, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_10

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_10 = [slice_10, full_1, full_1, full_1]
        del slice_10

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_10 = paddle._C_ops.stack(combine_10, 0)
        del combine_10

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_10 = paddle._C_ops.uniform(
            stack_10,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_10

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_86 = paddle._C_ops.add(full_13, uniform_10)
        del uniform_10

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_10 = paddle._C_ops.floor(add_86)
        del add_86

        # pd_op.divide: (-1x128x14x14xf32) <- (-1x128x14x14xf32, xf32)
        divide_10 = paddle._C_ops.divide(add_85, full_13)

        # pd_op.multiply: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x1x1x1xf32)
        multiply_22 = paddle._C_ops.multiply(divide_10, floor_10)

        # pd_op.add: (-1x128x14x14xf32) <- (-1x128x14x14xf32, -1x128x14x14xf32)
        add_87 = paddle._C_ops.add(add_80, multiply_22)

        # pd_op.conv2d: (-1x256x7x7xf32) <- (-1x128x14x14xf32, 256x128x3x3xf32)
        conv2d_40 = paddle._C_ops.conv2d(
            add_87, parameter_83, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_83

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_64 = paddle._C_ops.reshape(parameter_82, full_int_array_0)
        del parameter_82

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        add_88 = paddle._C_ops.add(conv2d_40, reshape_64)

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__168,
            batch_norm__169,
            batch_norm__170,
            batch_norm__171,
            batch_norm__172,
            batch_norm__173,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_88,
                parameter_81,
                parameter_80,
                parameter_79,
                parameter_78,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_78, parameter_79, parameter_80, parameter_81

        # pd_op.depthwise_conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x1x7x7xf32)
        depthwise_conv2d_24 = paddle._C_ops.depthwise_conv2d(
            batch_norm__168,
            parameter_77,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            256,
            [1, 1],
            "NCHW",
        )
        del parameter_77

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_65 = paddle._C_ops.reshape(parameter_76, full_int_array_0)
        del parameter_76

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        add_89 = paddle._C_ops.add(depthwise_conv2d_24, reshape_65)

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__174,
            batch_norm__175,
            batch_norm__176,
            batch_norm__177,
            batch_norm__178,
            batch_norm__179,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_89,
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
        del parameter_72, parameter_73, parameter_74, parameter_75

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x256x7x7xf32, 1024x256x1x1xf32)
        conv2d_41 = paddle._C_ops.conv2d(
            batch_norm__174, parameter_71, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_71

        # pd_op.reshape: (1x1024x1x1xf32) <- (1024xf32, 4xi64)
        reshape_66 = paddle._C_ops.reshape(parameter_70, full_int_array_0)
        del parameter_70

        # pd_op.add: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1x1024x1x1xf32)
        add_90 = paddle._C_ops.add(conv2d_41, reshape_66)

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x256x7x7xf32, 1024x256x1x1xf32)
        conv2d_42 = paddle._C_ops.conv2d(
            batch_norm__174, parameter_69, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_69

        # pd_op.reshape: (1x1024x1x1xf32) <- (1024xf32, 4xi64)
        reshape_67 = paddle._C_ops.reshape(parameter_68, full_int_array_0)
        del parameter_68

        # pd_op.add: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1x1024x1x1xf32)
        add_91 = paddle._C_ops.add(conv2d_42, reshape_67)

        # pd_op.relu6: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32)
        relu6_13 = paddle._C_ops.relu6(add_90)
        del add_90

        # pd_op.multiply: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, -1x1024x7x7xf32)
        multiply_23 = paddle._C_ops.multiply(relu6_13, add_91)

        # pd_op.conv2d: (-1x256x7x7xf32) <- (-1x1024x7x7xf32, 256x1024x1x1xf32)
        conv2d_43 = paddle._C_ops.conv2d(
            multiply_23, parameter_67, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_67

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_68 = paddle._C_ops.reshape(parameter_66, full_int_array_0)
        del parameter_66

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        add_92 = paddle._C_ops.add(conv2d_43, reshape_68)

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__180,
            batch_norm__181,
            batch_norm__182,
            batch_norm__183,
            batch_norm__184,
            batch_norm__185,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_92,
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

        # pd_op.depthwise_conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x1x7x7xf32)
        depthwise_conv2d_25 = paddle._C_ops.depthwise_conv2d(
            batch_norm__180,
            parameter_61,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            256,
            [1, 1],
            "NCHW",
        )
        del parameter_61

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_69 = paddle._C_ops.reshape(parameter_60, full_int_array_0)
        del parameter_60

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        add_93 = paddle._C_ops.add(depthwise_conv2d_25, reshape_69)

        # pd_op.full: (xf32) <- ()
        full_14 = paddle._C_ops.full(
            [],
            float("0.992"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x256x7x7xf32)
        shape64_11 = paddle._C_ops.shape64(add_93)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            shape64_11, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_11

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_11 = [slice_11, full_1, full_1, full_1]
        del slice_11

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_11 = paddle._C_ops.stack(combine_11, 0)
        del combine_11

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_11 = paddle._C_ops.uniform(
            stack_11,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_11

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_94 = paddle._C_ops.add(full_14, uniform_11)
        del uniform_11

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_11 = paddle._C_ops.floor(add_94)
        del add_94

        # pd_op.divide: (-1x256x7x7xf32) <- (-1x256x7x7xf32, xf32)
        divide_11 = paddle._C_ops.divide(add_93, full_14)

        # pd_op.multiply: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x1x1x1xf32)
        multiply_24 = paddle._C_ops.multiply(divide_11, floor_11)

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x256x7x7xf32)
        add_95 = paddle._C_ops.add(batch_norm__168, multiply_24)

        # pd_op.depthwise_conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x1x7x7xf32)
        depthwise_conv2d_26 = paddle._C_ops.depthwise_conv2d(
            add_95, parameter_59, [1, 1], [3, 3], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del parameter_59

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_70 = paddle._C_ops.reshape(parameter_58, full_int_array_0)
        del parameter_58

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        add_96 = paddle._C_ops.add(depthwise_conv2d_26, reshape_70)

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__186,
            batch_norm__187,
            batch_norm__188,
            batch_norm__189,
            batch_norm__190,
            batch_norm__191,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_96,
                parameter_57,
                parameter_56,
                parameter_55,
                parameter_54,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_54, parameter_55, parameter_56, parameter_57

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x256x7x7xf32, 1024x256x1x1xf32)
        conv2d_44 = paddle._C_ops.conv2d(
            batch_norm__186, parameter_53, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_53

        # pd_op.reshape: (1x1024x1x1xf32) <- (1024xf32, 4xi64)
        reshape_71 = paddle._C_ops.reshape(parameter_52, full_int_array_0)
        del parameter_52

        # pd_op.add: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1x1024x1x1xf32)
        add_97 = paddle._C_ops.add(conv2d_44, reshape_71)

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x256x7x7xf32, 1024x256x1x1xf32)
        conv2d_45 = paddle._C_ops.conv2d(
            batch_norm__186, parameter_51, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_51

        # pd_op.reshape: (1x1024x1x1xf32) <- (1024xf32, 4xi64)
        reshape_72 = paddle._C_ops.reshape(parameter_50, full_int_array_0)
        del parameter_50

        # pd_op.add: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1x1024x1x1xf32)
        add_98 = paddle._C_ops.add(conv2d_45, reshape_72)

        # pd_op.relu6: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32)
        relu6_14 = paddle._C_ops.relu6(add_97)
        del add_97

        # pd_op.multiply: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, -1x1024x7x7xf32)
        multiply_25 = paddle._C_ops.multiply(relu6_14, add_98)

        # pd_op.conv2d: (-1x256x7x7xf32) <- (-1x1024x7x7xf32, 256x1024x1x1xf32)
        conv2d_46 = paddle._C_ops.conv2d(
            multiply_25, parameter_49, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_49

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_73 = paddle._C_ops.reshape(parameter_48, full_int_array_0)
        del parameter_48

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        add_99 = paddle._C_ops.add(conv2d_46, reshape_73)

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__192,
            batch_norm__193,
            batch_norm__194,
            batch_norm__195,
            batch_norm__196,
            batch_norm__197,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_99,
                parameter_47,
                parameter_46,
                parameter_45,
                parameter_44,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_44, parameter_45, parameter_46, parameter_47

        # pd_op.depthwise_conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x1x7x7xf32)
        depthwise_conv2d_27 = paddle._C_ops.depthwise_conv2d(
            batch_norm__192,
            parameter_43,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            256,
            [1, 1],
            "NCHW",
        )
        del parameter_43

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_74 = paddle._C_ops.reshape(parameter_42, full_int_array_0)
        del parameter_42

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        add_100 = paddle._C_ops.add(depthwise_conv2d_27, reshape_74)

        # pd_op.full: (xf32) <- ()
        full_15 = paddle._C_ops.full(
            [],
            float("0.991333"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x256x7x7xf32)
        shape64_12 = paddle._C_ops.shape64(add_100)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            shape64_12, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_12

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_12 = [slice_12, full_1, full_1, full_1]
        del slice_12

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_12 = paddle._C_ops.stack(combine_12, 0)
        del combine_12

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_12 = paddle._C_ops.uniform(
            stack_12,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_12

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_101 = paddle._C_ops.add(full_15, uniform_12)
        del uniform_12

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_12 = paddle._C_ops.floor(add_101)
        del add_101

        # pd_op.divide: (-1x256x7x7xf32) <- (-1x256x7x7xf32, xf32)
        divide_12 = paddle._C_ops.divide(add_100, full_15)

        # pd_op.multiply: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x1x1x1xf32)
        multiply_26 = paddle._C_ops.multiply(divide_12, floor_12)

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x256x7x7xf32)
        add_102 = paddle._C_ops.add(add_95, multiply_26)

        # pd_op.depthwise_conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x1x7x7xf32)
        depthwise_conv2d_28 = paddle._C_ops.depthwise_conv2d(
            add_102, parameter_41, [1, 1], [3, 3], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del parameter_41

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_75 = paddle._C_ops.reshape(parameter_40, full_int_array_0)
        del parameter_40

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        add_103 = paddle._C_ops.add(depthwise_conv2d_28, reshape_75)

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__198,
            batch_norm__199,
            batch_norm__200,
            batch_norm__201,
            batch_norm__202,
            batch_norm__203,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_103,
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
        del parameter_36, parameter_37, parameter_38, parameter_39

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x256x7x7xf32, 1024x256x1x1xf32)
        conv2d_47 = paddle._C_ops.conv2d(
            batch_norm__198, parameter_35, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_35

        # pd_op.reshape: (1x1024x1x1xf32) <- (1024xf32, 4xi64)
        reshape_76 = paddle._C_ops.reshape(parameter_34, full_int_array_0)
        del parameter_34

        # pd_op.add: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1x1024x1x1xf32)
        add_104 = paddle._C_ops.add(conv2d_47, reshape_76)

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x256x7x7xf32, 1024x256x1x1xf32)
        conv2d_48 = paddle._C_ops.conv2d(
            batch_norm__198, parameter_33, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_33

        # pd_op.reshape: (1x1024x1x1xf32) <- (1024xf32, 4xi64)
        reshape_77 = paddle._C_ops.reshape(parameter_32, full_int_array_0)
        del parameter_32

        # pd_op.add: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1x1024x1x1xf32)
        add_105 = paddle._C_ops.add(conv2d_48, reshape_77)

        # pd_op.relu6: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32)
        relu6_15 = paddle._C_ops.relu6(add_104)
        del add_104

        # pd_op.multiply: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, -1x1024x7x7xf32)
        multiply_27 = paddle._C_ops.multiply(relu6_15, add_105)

        # pd_op.conv2d: (-1x256x7x7xf32) <- (-1x1024x7x7xf32, 256x1024x1x1xf32)
        conv2d_49 = paddle._C_ops.conv2d(
            multiply_27, parameter_31, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_31

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_78 = paddle._C_ops.reshape(parameter_30, full_int_array_0)
        del parameter_30

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        add_106 = paddle._C_ops.add(conv2d_49, reshape_78)

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__204,
            batch_norm__205,
            batch_norm__206,
            batch_norm__207,
            batch_norm__208,
            batch_norm__209,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_106,
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
        del parameter_26, parameter_27, parameter_28, parameter_29

        # pd_op.depthwise_conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x1x7x7xf32)
        depthwise_conv2d_29 = paddle._C_ops.depthwise_conv2d(
            batch_norm__204,
            parameter_25,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            256,
            [1, 1],
            "NCHW",
        )
        del parameter_25

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_79 = paddle._C_ops.reshape(parameter_24, full_int_array_0)
        del parameter_24

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        add_107 = paddle._C_ops.add(depthwise_conv2d_29, reshape_79)

        # pd_op.full: (xf32) <- ()
        full_16 = paddle._C_ops.full(
            [],
            float("0.990667"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x256x7x7xf32)
        shape64_13 = paddle._C_ops.shape64(add_107)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            shape64_13, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_13

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_13 = [slice_13, full_1, full_1, full_1]
        del slice_13

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_13 = paddle._C_ops.stack(combine_13, 0)
        del combine_13

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_13 = paddle._C_ops.uniform(
            stack_13,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_13

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_108 = paddle._C_ops.add(full_16, uniform_13)
        del uniform_13

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_13 = paddle._C_ops.floor(add_108)
        del add_108

        # pd_op.divide: (-1x256x7x7xf32) <- (-1x256x7x7xf32, xf32)
        divide_13 = paddle._C_ops.divide(add_107, full_16)

        # pd_op.multiply: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x1x1x1xf32)
        multiply_28 = paddle._C_ops.multiply(divide_13, floor_13)

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x256x7x7xf32)
        add_109 = paddle._C_ops.add(add_102, multiply_28)

        # pd_op.depthwise_conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x1x7x7xf32)
        depthwise_conv2d_30 = paddle._C_ops.depthwise_conv2d(
            add_109, parameter_23, [1, 1], [3, 3], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del parameter_23

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_80 = paddle._C_ops.reshape(parameter_22, full_int_array_0)
        del parameter_22

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        add_110 = paddle._C_ops.add(depthwise_conv2d_30, reshape_80)

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__210,
            batch_norm__211,
            batch_norm__212,
            batch_norm__213,
            batch_norm__214,
            batch_norm__215,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_110,
                parameter_21,
                parameter_20,
                parameter_19,
                parameter_18,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_18, parameter_19, parameter_20, parameter_21

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x256x7x7xf32, 1024x256x1x1xf32)
        conv2d_50 = paddle._C_ops.conv2d(
            batch_norm__210, parameter_17, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_17

        # pd_op.reshape: (1x1024x1x1xf32) <- (1024xf32, 4xi64)
        reshape_81 = paddle._C_ops.reshape(parameter_16, full_int_array_0)
        del parameter_16

        # pd_op.add: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1x1024x1x1xf32)
        add_111 = paddle._C_ops.add(conv2d_50, reshape_81)

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x256x7x7xf32, 1024x256x1x1xf32)
        conv2d_51 = paddle._C_ops.conv2d(
            batch_norm__210, parameter_15, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_15

        # pd_op.reshape: (1x1024x1x1xf32) <- (1024xf32, 4xi64)
        reshape_82 = paddle._C_ops.reshape(parameter_14, full_int_array_0)
        del parameter_14

        # pd_op.add: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, 1x1024x1x1xf32)
        add_112 = paddle._C_ops.add(conv2d_51, reshape_82)

        # pd_op.relu6: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32)
        relu6_16 = paddle._C_ops.relu6(add_111)
        del add_111

        # pd_op.multiply: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, -1x1024x7x7xf32)
        multiply_29 = paddle._C_ops.multiply(relu6_16, add_112)

        # pd_op.conv2d: (-1x256x7x7xf32) <- (-1x1024x7x7xf32, 256x1024x1x1xf32)
        conv2d_52 = paddle._C_ops.conv2d(
            multiply_29, parameter_13, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_13

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_83 = paddle._C_ops.reshape(parameter_12, full_int_array_0)
        del parameter_12

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        add_113 = paddle._C_ops.add(conv2d_52, reshape_83)

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__216,
            batch_norm__217,
            batch_norm__218,
            batch_norm__219,
            batch_norm__220,
            batch_norm__221,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_113,
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

        # pd_op.depthwise_conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x1x7x7xf32)
        depthwise_conv2d_31 = paddle._C_ops.depthwise_conv2d(
            batch_norm__216,
            parameter_7,
            [1, 1],
            [3, 3],
            "EXPLICIT",
            256,
            [1, 1],
            "NCHW",
        )
        del parameter_7

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_84 = paddle._C_ops.reshape(parameter_6, full_int_array_0)
        del full_int_array_0, parameter_6

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 1x256x1x1xf32)
        add_114 = paddle._C_ops.add(depthwise_conv2d_31, reshape_84)

        # pd_op.full: (xf32) <- ()
        full_17 = paddle._C_ops.full(
            [],
            float("0.99"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x256x7x7xf32)
        shape64_14 = paddle._C_ops.shape64(add_114)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            shape64_14, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del full_int_array_1, full_int_array_2, shape64_14

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_14 = [slice_14, full_1, full_1, full_1]
        del full_1, slice_14

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_14 = paddle._C_ops.stack(combine_14, 0)
        del combine_14

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_14 = paddle._C_ops.uniform(
            stack_14,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del full_2, full_3, stack_14

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_115 = paddle._C_ops.add(full_17, uniform_14)
        del uniform_14

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_14 = paddle._C_ops.floor(add_115)
        del add_115

        # pd_op.divide: (-1x256x7x7xf32) <- (-1x256x7x7xf32, xf32)
        divide_14 = paddle._C_ops.divide(add_114, full_17)

        # pd_op.multiply: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x1x1x1xf32)
        multiply_30 = paddle._C_ops.multiply(divide_14, floor_14)

        # pd_op.add: (-1x256x7x7xf32) <- (-1x256x7x7xf32, -1x256x7x7xf32)
        add_116 = paddle._C_ops.add(add_109, multiply_30)

        # pd_op.batch_norm_: (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x7x7xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__222,
            batch_norm__223,
            batch_norm__224,
            batch_norm__225,
            batch_norm__226,
            batch_norm__227,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_116,
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

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf32) <- (-1x256x7x7xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            batch_norm__222,
            full_int_array_3,
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

        # pd_op.flatten: (-1x256xf32) <- (-1x256x1x1xf32)
        flatten_0 = paddle._C_ops.flatten(pool2d_0, 1, 3)

        # pd_op.matmul: (-1x102xf32) <- (-1x256xf32, 256x102xf32)
        matmul_0 = paddle._C_ops.matmul(flatten_0, parameter_1, False, False)
        del parameter_1

        # pd_op.add: (-1x102xf32) <- (-1x102xf32, 102xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_0)
        del (
            add_1,
            add_100,
            add_102,
            add_103,
            add_105,
            add_106,
            add_107,
            add_109,
            add_11,
            add_110,
            add_112,
            add_113,
            add_114,
            add_116,
            add_12,
            add_13,
            add_15,
            add_16,
            add_17,
            add_19,
            add_2,
            add_20,
            add_21,
            add_23,
            add_24,
            add_26,
            add_27,
            add_28,
            add_3,
            add_30,
            add_31,
            add_32,
            add_34,
            add_35,
            add_36,
            add_38,
            add_39,
            add_41,
            add_42,
            add_43,
            add_45,
            add_46,
            add_48,
            add_49,
            add_5,
            add_50,
            add_52,
            add_53,
            add_55,
            add_56,
            add_57,
            add_59,
            add_6,
            add_60,
            add_62,
            add_63,
            add_64,
            add_66,
            add_67,
            add_69,
            add_7,
            add_70,
            add_71,
            add_73,
            add_74,
            add_76,
            add_77,
            add_78,
            add_8,
            add_80,
            add_81,
            add_83,
            add_84,
            add_85,
            add_87,
            add_88,
            add_89,
            add_9,
            add_91,
            add_92,
            add_93,
            add_95,
            add_96,
            add_98,
            add_99,
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
            batch_norm__23,
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
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
            depthwise_conv2d_17,
            depthwise_conv2d_18,
            depthwise_conv2d_19,
            depthwise_conv2d_2,
            depthwise_conv2d_20,
            depthwise_conv2d_21,
            depthwise_conv2d_22,
            depthwise_conv2d_23,
            depthwise_conv2d_24,
            depthwise_conv2d_25,
            depthwise_conv2d_26,
            depthwise_conv2d_27,
            depthwise_conv2d_28,
            depthwise_conv2d_29,
            depthwise_conv2d_3,
            depthwise_conv2d_30,
            depthwise_conv2d_31,
            depthwise_conv2d_4,
            depthwise_conv2d_5,
            depthwise_conv2d_6,
            depthwise_conv2d_7,
            depthwise_conv2d_8,
            depthwise_conv2d_9,
            divide_0,
            divide_1,
            divide_10,
            divide_11,
            divide_12,
            divide_13,
            divide_14,
            divide_2,
            divide_3,
            divide_4,
            divide_5,
            divide_6,
            divide_7,
            divide_8,
            divide_9,
            flatten_0,
            floor_0,
            floor_1,
            floor_10,
            floor_11,
            floor_12,
            floor_13,
            floor_14,
            floor_2,
            floor_3,
            floor_4,
            floor_5,
            floor_6,
            floor_7,
            floor_8,
            floor_9,
            full_0,
            full_10,
            full_11,
            full_12,
            full_13,
            full_14,
            full_15,
            full_16,
            full_17,
            full_4,
            full_5,
            full_6,
            full_7,
            full_8,
            full_9,
            full_int_array_3,
            matmul_0,
            multiply_0,
            multiply_1,
            multiply_10,
            multiply_11,
            multiply_12,
            multiply_13,
            multiply_14,
            multiply_15,
            multiply_16,
            multiply_17,
            multiply_18,
            multiply_19,
            multiply_2,
            multiply_20,
            multiply_21,
            multiply_22,
            multiply_23,
            multiply_24,
            multiply_25,
            multiply_26,
            multiply_27,
            multiply_28,
            multiply_29,
            multiply_3,
            multiply_30,
            multiply_4,
            multiply_5,
            multiply_6,
            multiply_7,
            multiply_8,
            multiply_9,
            parameter_0,
            pool2d_0,
            relu6_0,
            relu6_1,
            relu6_10,
            relu6_11,
            relu6_12,
            relu6_13,
            relu6_14,
            relu6_15,
            relu6_16,
            relu6_2,
            relu6_3,
            relu6_4,
            relu6_5,
            relu6_6,
            relu6_7,
            relu6_8,
            relu6_9,
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
            reshape_20,
            reshape_21,
            reshape_22,
            reshape_23,
            reshape_24,
            reshape_25,
            reshape_26,
            reshape_27,
            reshape_28,
            reshape_29,
            reshape_3,
            reshape_30,
            reshape_31,
            reshape_32,
            reshape_33,
            reshape_34,
            reshape_35,
            reshape_36,
            reshape_37,
            reshape_38,
            reshape_39,
            reshape_4,
            reshape_40,
            reshape_41,
            reshape_42,
            reshape_43,
            reshape_44,
            reshape_45,
            reshape_46,
            reshape_47,
            reshape_48,
            reshape_49,
            reshape_5,
            reshape_50,
            reshape_51,
            reshape_52,
            reshape_53,
            reshape_54,
            reshape_55,
            reshape_56,
            reshape_57,
            reshape_58,
            reshape_59,
            reshape_6,
            reshape_60,
            reshape_61,
            reshape_62,
            reshape_63,
            reshape_64,
            reshape_65,
            reshape_66,
            reshape_67,
            reshape_68,
            reshape_69,
            reshape_7,
            reshape_70,
            reshape_71,
            reshape_72,
            reshape_73,
            reshape_74,
            reshape_75,
            reshape_76,
            reshape_77,
            reshape_78,
            reshape_79,
            reshape_8,
            reshape_80,
            reshape_81,
            reshape_82,
            reshape_83,
            reshape_84,
            reshape_9,
        )

        return add_0
