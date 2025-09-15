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
    ):
        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.equal: (1x21xb) <- (1x21xi64, xi64)
        equal_0 = paddle._C_ops.equal(data_0, full_0)
        del full_0

        # pd_op.cast: (1x21xf32) <- (1x21xb)
        cast_0 = paddle._C_ops.cast(equal_0, paddle.float32)
        del equal_0

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("-10000"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x21xf32) <- (1x21xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(cast_0, full_1, float("0"), True)
        del cast_0, full_1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 2]

        # pd_op.unsqueeze: (1x1x1x21xf32) <- (1x21xf32, 2xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(scale_0, full_int_array_0)
        del full_int_array_0, scale_0

        # pd_op.embedding: (1x21x768xf32) <- (1x21xi64, 30522x768xf32)
        embedding_0 = paddle._C_ops.embedding(data_0, parameter_282, 0, False)
        del data_0, parameter_282

        # pd_op.full: (1x21xi64) <- ()
        full_2 = paddle._C_ops.full(
            [1, 21],
            float("1"),
            paddle.int64,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.cumsum: (1x21xi64) <- (1x21xi64, 1xi32)
        cumsum_0 = paddle._C_ops.cumsum(full_2, full_3, False, False, False)
        del full_3

        # pd_op.subtract: (1x21xi64) <- (1x21xi64, 1x21xi64)
        subtract_0 = paddle._C_ops.subtract(cumsum_0, full_2)
        del cumsum_0, full_2

        # pd_op.embedding: (1x21x768xf32) <- (1x21xi64, 512x768xf32)
        embedding_1 = paddle._C_ops.embedding(subtract_0, parameter_281, -1, False)
        del parameter_281

        # pd_op.embedding: (1x21x768xf32) <- (1x21xi64, 2x768xf32)
        embedding_2 = paddle._C_ops.embedding(data_1, parameter_280, -1, False)
        del data_1, parameter_280

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_0 = paddle._C_ops.add(embedding_0, embedding_1)

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_1 = paddle._C_ops.add(add_0, embedding_2)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_1, parameter_279, parameter_278, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_278, parameter_279

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_0 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_1 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_2 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_3 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_4 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_5 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_6 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_7 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_8 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_9 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_10 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_11 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_12 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_13 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_14 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_15 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_16 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_17 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_18 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_19 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_20 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_21 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_22 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_23 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_24 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_25 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_26 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_27 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_28 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_29 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_30 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_31 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_32 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_33 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_34 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_35 = full_4

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                layer_norm_0, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del layer_norm_0

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_0 = paddle._C_ops.matmul(dropout_0, parameter_277, False, False)
        del parameter_277

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_2 = paddle._C_ops.add(matmul_0, parameter_276)
        del parameter_276

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_1 = paddle._C_ops.matmul(dropout_0, parameter_275, False, False)
        del parameter_275

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_3 = paddle._C_ops.add(matmul_1, parameter_274)
        del parameter_274

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_2 = paddle._C_ops.matmul(dropout_0, parameter_273, False, False)
        del parameter_273

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_4 = paddle._C_ops.add(matmul_2, parameter_272)
        del parameter_272

        # pd_op.assign: (768x1x9xf32) <- (768x1x9xf32)
        assign_36 = parameter_268
        del parameter_268

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [-2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_37 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_38 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_39 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_40 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_41 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_42 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_43 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_44 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_45 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_46 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_47 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_48 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_49 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_50 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_51 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_52 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_53 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_54 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_55 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_56 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_57 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_58 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_59 = full_int_array_1

        # pd_op.unsqueeze: (768x1x1x9xf32) <- (768x1x9xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(assign_36, full_int_array_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [-3]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_60 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_61 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_62 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_63 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_64 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_65 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_66 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_67 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_68 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_69 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_70 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_71 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_72 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_73 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_74 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_75 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_76 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_77 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_78 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_79 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_80 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_81 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_82 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_83 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_84 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_85 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_86 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_87 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_88 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_89 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_90 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_91 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_92 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_93 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_94 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_95 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_96 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_97 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_98 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_99 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_100 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_101 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_102 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_103 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_104 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_105 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_106 = full_int_array_2

        # pd_op.unsqueeze: (1x1x21x768xf32) <- (1x21x768xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(dropout_0, full_int_array_2)

        # pd_op.depthwise_conv2d: (1x1x21x768xf32) <- (1x1x21x768xf32, 768x1x1x9xf32)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(
            unsqueeze_2, unsqueeze_1, [1, 1], [0, 4], "EXPLICIT", 768, [1, 1], "NHWC"
        )

        # pd_op.squeeze: (1x21x768xf32) <- (1x1x21x768xf32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(depthwise_conv2d_0, full_int_array_2)

        # pd_op.assign: (384x768x1xf32) <- (384x768x1xf32)
        assign_107 = parameter_267
        del parameter_267

        # pd_op.unsqueeze: (384x768x1x1xf32) <- (384x768x1xf32, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(assign_107, full_int_array_1)

        # pd_op.unsqueeze: (1x1x21x768xf32) <- (1x21x768xf32, 1xi64)
        unsqueeze_4 = paddle._C_ops.unsqueeze(squeeze_0, full_int_array_2)

        # pd_op.conv2d: (1x1x21x384xf32) <- (1x1x21x768xf32, 384x768x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            unsqueeze_4, unsqueeze_3, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NHWC"
        )

        # pd_op.squeeze: (1x21x384xf32) <- (1x1x21x384xf32, 1xi64)
        squeeze_1 = paddle._C_ops.squeeze(conv2d_0, full_int_array_2)

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_5 = paddle._C_ops.add(squeeze_1, parameter_269)
        del parameter_269

        # pd_op.multiply: (1x21x384xf32) <- (1x21x384xf32, 1x21x384xf32)
        multiply_0 = paddle._C_ops.multiply(add_5, add_2)

        # pd_op.matmul: (1x21x54xf32) <- (1x21x384xf32, 384x54xf32)
        matmul_3 = paddle._C_ops.matmul(multiply_0, parameter_266, False, False)
        del parameter_266

        # pd_op.add: (1x21x54xf32) <- (1x21x54xf32, 54xf32)
        add_6 = paddle._C_ops.add(matmul_3, parameter_265)
        del parameter_265

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_3 = [-1, 9, 1]

        # pd_op.reshape: (126x9x1xf32) <- (1x21x54xf32, 3xi64)
        reshape_0 = paddle._C_ops.reshape(add_6, full_int_array_3)

        # pd_op.softmax: (126x9x1xf32) <- (126x9x1xf32)
        softmax_0 = paddle._C_ops.softmax(reshape_0, 1)
        del reshape_0

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_4 = paddle._C_ops.matmul(dropout_0, parameter_264, False, False)
        del parameter_264

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_7 = paddle._C_ops.add(matmul_4, parameter_263)
        del parameter_263

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [2, 3]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_108 = full_int_array_4

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_109 = full_int_array_4

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_110 = full_int_array_4

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_111 = full_int_array_4

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_112 = full_int_array_4

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_113 = full_int_array_4

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_114 = full_int_array_4

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_115 = full_int_array_4

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_116 = full_int_array_4

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_117 = full_int_array_4

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_118 = full_int_array_4

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_119 = full_int_array_4

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_120 = full_int_array_4

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_121 = full_int_array_4

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_122 = full_int_array_4

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_123 = full_int_array_4

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_124 = full_int_array_4

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_125 = full_int_array_4

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_126 = full_int_array_4

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_127 = full_int_array_4

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_128 = full_int_array_4

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_129 = full_int_array_4

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_130 = full_int_array_4

        # pd_op.unsqueeze: (1x21x1x1x384xf32) <- (1x21x384xf32, 2xi64)
        unsqueeze_5 = paddle._C_ops.unsqueeze(add_7, full_int_array_4)

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_5 = [0, 0, 0, 0, 4, 4]

        # pd_op.assign: (6xi64) <- (6xi64)
        assign_131 = full_int_array_5

        # pd_op.assign: (6xi64) <- (6xi64)
        assign_132 = full_int_array_5

        # pd_op.assign: (6xi64) <- (6xi64)
        assign_133 = full_int_array_5

        # pd_op.assign: (6xi64) <- (6xi64)
        assign_134 = full_int_array_5

        # pd_op.assign: (6xi64) <- (6xi64)
        assign_135 = full_int_array_5

        # pd_op.assign: (6xi64) <- (6xi64)
        assign_136 = full_int_array_5

        # pd_op.assign: (6xi64) <- (6xi64)
        assign_137 = full_int_array_5

        # pd_op.assign: (6xi64) <- (6xi64)
        assign_138 = full_int_array_5

        # pd_op.assign: (6xi64) <- (6xi64)
        assign_139 = full_int_array_5

        # pd_op.assign: (6xi64) <- (6xi64)
        assign_140 = full_int_array_5

        # pd_op.assign: (6xi64) <- (6xi64)
        assign_141 = full_int_array_5

        # pd_op.pad3d: (1x29x1x1x384xf32) <- (1x21x1x1x384xf32, 6xi64)
        pad3d_0 = paddle._C_ops.pad3d(
            unsqueeze_5, full_int_array_5, "constant", float("0"), "NDHWC"
        )

        # pd_op.squeeze: (1x29x384xf32) <- (1x29x1x1x384xf32, 2xi64)
        squeeze_2 = paddle._C_ops.squeeze(pad3d_0, full_int_array_4)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_142 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_143 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_144 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_145 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_146 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_147 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_148 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_149 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_150 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_151 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_152 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_153 = full_int_array_6

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [21]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_154 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_155 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_156 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_157 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_158 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_159 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_160 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_161 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_162 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_163 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_164 = full_int_array_7

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            squeeze_2, [1], full_int_array_6, full_int_array_7, [1], []
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_165 = full_int_array_8

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_166 = full_int_array_8

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_167 = full_int_array_8

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_168 = full_int_array_8

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_169 = full_int_array_8

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_170 = full_int_array_8

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_171 = full_int_array_8

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_172 = full_int_array_8

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_173 = full_int_array_8

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_174 = full_int_array_8

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_175 = full_int_array_8

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_176 = full_int_array_8

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [22]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_177 = full_int_array_9

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_178 = full_int_array_9

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_179 = full_int_array_9

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_180 = full_int_array_9

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_181 = full_int_array_9

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_182 = full_int_array_9

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_183 = full_int_array_9

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_184 = full_int_array_9

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_185 = full_int_array_9

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_186 = full_int_array_9

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_187 = full_int_array_9

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            squeeze_2, [1], full_int_array_8, full_int_array_9, [1], []
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_10 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_188 = full_int_array_10

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_189 = full_int_array_10

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_190 = full_int_array_10

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_191 = full_int_array_10

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_192 = full_int_array_10

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_193 = full_int_array_10

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_194 = full_int_array_10

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_195 = full_int_array_10

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_196 = full_int_array_10

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_197 = full_int_array_10

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_198 = full_int_array_10

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_11 = [23]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_199 = full_int_array_11

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_200 = full_int_array_11

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_201 = full_int_array_11

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_202 = full_int_array_11

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_203 = full_int_array_11

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_204 = full_int_array_11

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_205 = full_int_array_11

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_206 = full_int_array_11

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_207 = full_int_array_11

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_208 = full_int_array_11

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_209 = full_int_array_11

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            squeeze_2, [1], full_int_array_10, full_int_array_11, [1], []
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_12 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_210 = full_int_array_12

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_211 = full_int_array_12

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_212 = full_int_array_12

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_213 = full_int_array_12

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_214 = full_int_array_12

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_215 = full_int_array_12

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_216 = full_int_array_12

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_217 = full_int_array_12

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_218 = full_int_array_12

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_219 = full_int_array_12

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_220 = full_int_array_12

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_13 = [24]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_221 = full_int_array_13

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_222 = full_int_array_13

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_223 = full_int_array_13

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_224 = full_int_array_13

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_225 = full_int_array_13

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_226 = full_int_array_13

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_227 = full_int_array_13

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_228 = full_int_array_13

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_229 = full_int_array_13

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_230 = full_int_array_13

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_231 = full_int_array_13

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            squeeze_2, [1], full_int_array_12, full_int_array_13, [1], []
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_14 = [4]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_232 = full_int_array_14

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_233 = full_int_array_14

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_234 = full_int_array_14

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_235 = full_int_array_14

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_236 = full_int_array_14

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_237 = full_int_array_14

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_238 = full_int_array_14

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_239 = full_int_array_14

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_240 = full_int_array_14

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_241 = full_int_array_14

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_242 = full_int_array_14

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_15 = [25]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_243 = full_int_array_15

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_244 = full_int_array_15

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_245 = full_int_array_15

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_246 = full_int_array_15

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_247 = full_int_array_15

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_248 = full_int_array_15

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_249 = full_int_array_15

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_250 = full_int_array_15

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_251 = full_int_array_15

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_252 = full_int_array_15

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_253 = full_int_array_15

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            squeeze_2, [1], full_int_array_14, full_int_array_15, [1], []
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_16 = [5]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_254 = full_int_array_16

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_255 = full_int_array_16

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_256 = full_int_array_16

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_257 = full_int_array_16

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_258 = full_int_array_16

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_259 = full_int_array_16

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_260 = full_int_array_16

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_261 = full_int_array_16

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_262 = full_int_array_16

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_263 = full_int_array_16

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_264 = full_int_array_16

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_17 = [26]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_265 = full_int_array_17

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_266 = full_int_array_17

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_267 = full_int_array_17

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_268 = full_int_array_17

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_269 = full_int_array_17

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_270 = full_int_array_17

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_271 = full_int_array_17

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_272 = full_int_array_17

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_273 = full_int_array_17

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_274 = full_int_array_17

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_275 = full_int_array_17

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            squeeze_2, [1], full_int_array_16, full_int_array_17, [1], []
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_18 = [6]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_276 = full_int_array_18

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_277 = full_int_array_18

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_278 = full_int_array_18

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_279 = full_int_array_18

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_280 = full_int_array_18

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_281 = full_int_array_18

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_282 = full_int_array_18

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_283 = full_int_array_18

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_284 = full_int_array_18

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_285 = full_int_array_18

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_286 = full_int_array_18

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_19 = [27]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_287 = full_int_array_19

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_288 = full_int_array_19

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_289 = full_int_array_19

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_290 = full_int_array_19

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_291 = full_int_array_19

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_292 = full_int_array_19

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_293 = full_int_array_19

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_294 = full_int_array_19

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_295 = full_int_array_19

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_296 = full_int_array_19

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_297 = full_int_array_19

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            squeeze_2, [1], full_int_array_18, full_int_array_19, [1], []
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_20 = [7]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_298 = full_int_array_20

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_299 = full_int_array_20

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_300 = full_int_array_20

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_301 = full_int_array_20

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_302 = full_int_array_20

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_303 = full_int_array_20

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_304 = full_int_array_20

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_305 = full_int_array_20

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_306 = full_int_array_20

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_307 = full_int_array_20

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_308 = full_int_array_20

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_21 = [28]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_309 = full_int_array_21

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_310 = full_int_array_21

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_311 = full_int_array_21

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_312 = full_int_array_21

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_313 = full_int_array_21

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_314 = full_int_array_21

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_315 = full_int_array_21

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_316 = full_int_array_21

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_317 = full_int_array_21

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_318 = full_int_array_21

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_319 = full_int_array_21

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            squeeze_2, [1], full_int_array_20, full_int_array_21, [1], []
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_22 = [8]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_320 = full_int_array_22

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_321 = full_int_array_22

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_322 = full_int_array_22

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_323 = full_int_array_22

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_324 = full_int_array_22

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_325 = full_int_array_22

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_326 = full_int_array_22

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_327 = full_int_array_22

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_328 = full_int_array_22

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_329 = full_int_array_22

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_330 = full_int_array_22

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_23 = [29]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_331 = full_int_array_23

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_332 = full_int_array_23

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_333 = full_int_array_23

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_334 = full_int_array_23

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_335 = full_int_array_23

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_336 = full_int_array_23

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_337 = full_int_array_23

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_338 = full_int_array_23

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_339 = full_int_array_23

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_340 = full_int_array_23

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_341 = full_int_array_23

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            squeeze_2, [1], full_int_array_22, full_int_array_23, [1], []
        )

        # builtin.combine: ([1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32]) <- (1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32)
        combine_0 = [
            slice_0,
            slice_1,
            slice_2,
            slice_3,
            slice_4,
            slice_5,
            slice_6,
            slice_7,
            slice_8,
        ]

        # pd_op.stack: (1x21x384x9xf32) <- ([1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32])
        stack_0 = paddle._C_ops.stack(combine_0, -1)
        del combine_0

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_24 = [-1, 64, 9]

        # pd_op.reshape: (126x64x9xf32) <- (1x21x384x9xf32, 3xi64)
        reshape_1 = paddle._C_ops.reshape(stack_0, full_int_array_24)

        # pd_op.matmul: (126x64x1xf32) <- (126x64x9xf32, 126x9x1xf32)
        matmul_5 = paddle._C_ops.matmul(reshape_1, softmax_0, False, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_25 = [1, 21, 6, 64]

        # pd_op.reshape: (1x21x6x64xf32) <- (126x64x1xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(matmul_5, full_int_array_25)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_26 = [0, 0, 6, 64]

        # pd_op.reshape: (1x21x6x64xf32) <- (1x21x384xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(add_2, full_int_array_26)

        # pd_op.transpose: (1x6x21x64xf32) <- (1x21x6x64xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_3, [0, 2, 1, 3])
        del reshape_3

        # pd_op.reshape: (1x21x6x64xf32) <- (1x21x384xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(add_3, full_int_array_26)

        # pd_op.transpose: (1x6x21x64xf32) <- (1x21x6x64xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_4, [0, 2, 1, 3])
        del reshape_4

        # pd_op.reshape: (1x21x6x64xf32) <- (1x21x384xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(add_4, full_int_array_26)

        # pd_op.transpose: (1x6x21x64xf32) <- (1x21x6x64xf32)
        transpose_2 = paddle._C_ops.transpose(reshape_5, [0, 2, 1, 3])
        del reshape_5

        # pd_op.matmul: (1x6x21x21xf32) <- (1x6x21x64xf32, 1x6x21x64xf32)
        matmul_6 = paddle._C_ops.matmul(transpose_0, transpose_1, False, True)

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_342 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_343 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_344 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_345 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_346 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_347 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_348 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_349 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_350 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_351 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_352 = full_5

        # pd_op.scale: (1x6x21x21xf32) <- (1x6x21x21xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(matmul_6, full_5, float("0"), True)
        del matmul_6

        # pd_op.add: (1x6x21x21xf32) <- (1x6x21x21xf32, 1x1x1x21xf32)
        add_8 = paddle._C_ops.add(scale_1, unsqueeze_0)

        # pd_op.softmax: (1x6x21x21xf32) <- (1x6x21x21xf32)
        softmax_1 = paddle._C_ops.softmax(add_8, -1)
        del add_8

        # pd_op.dropout: (1x6x21x21xf32, 1x6x21x21xui8) <- (1x6x21x21xf32, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_1, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x6x21x64xf32) <- (1x6x21x21xf32, 1x6x21x64xf32)
        matmul_7 = paddle._C_ops.matmul(dropout_2, transpose_2, False, False)

        # pd_op.transpose: (1x21x6x64xf32) <- (1x6x21x64xf32)
        transpose_3 = paddle._C_ops.transpose(matmul_7, [0, 2, 1, 3])
        del matmul_7

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("2"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_353 = full_6

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_354 = full_6

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_355 = full_6

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_356 = full_6

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_357 = full_6

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_358 = full_6

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_359 = full_6

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_360 = full_6

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_361 = full_6

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_362 = full_6

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_363 = full_6

        # builtin.combine: ([1x21x6x64xf32, 1x21x6x64xf32]) <- (1x21x6x64xf32, 1x21x6x64xf32)
        combine_1 = [transpose_3, reshape_2]

        # pd_op.concat: (1x21x12x64xf32) <- ([1x21x6x64xf32, 1x21x6x64xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_1, full_6)
        del combine_1

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_27 = [0, 0, 768]

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_6 = paddle._C_ops.reshape(concat_0, full_int_array_27)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_8 = paddle._C_ops.matmul(reshape_6, parameter_271, False, False)
        del parameter_271

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_9 = paddle._C_ops.add(matmul_8, parameter_270)
        del parameter_270

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_9, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_9

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_10 = paddle._C_ops.add(dropout_0, dropout_4)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_10, parameter_258, parameter_257, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_257, parameter_258

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_9 = paddle._C_ops.matmul(layer_norm_3, parameter_262, False, False)
        del parameter_262

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_11 = paddle._C_ops.add(matmul_9, parameter_261)
        del parameter_261

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_0 = paddle._C_ops.gelu(add_11, False)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_10 = paddle._C_ops.matmul(gelu_0, parameter_260, False, False)
        del parameter_260

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_12 = paddle._C_ops.add(matmul_10, parameter_259)
        del parameter_259

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_6, dropout_7 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_12, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_12

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_13 = paddle._C_ops.add(layer_norm_3, dropout_6)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_13, parameter_256, parameter_255, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_255, parameter_256

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_11 = paddle._C_ops.matmul(layer_norm_6, parameter_254, False, False)
        del parameter_254

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_14 = paddle._C_ops.add(matmul_11, parameter_253)
        del parameter_253

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_12 = paddle._C_ops.matmul(layer_norm_6, parameter_252, False, False)
        del parameter_252

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_15 = paddle._C_ops.add(matmul_12, parameter_251)
        del parameter_251

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_13 = paddle._C_ops.matmul(layer_norm_6, parameter_250, False, False)
        del parameter_250

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_16 = paddle._C_ops.add(matmul_13, parameter_249)
        del parameter_249

        # pd_op.assign: (768x1x9xf32) <- (768x1x9xf32)
        assign_364 = parameter_245
        del parameter_245

        # pd_op.unsqueeze: (768x1x1x9xf32) <- (768x1x9xf32, 1xi64)
        unsqueeze_6 = paddle._C_ops.unsqueeze(assign_364, full_int_array_1)

        # pd_op.unsqueeze: (1x1x21x768xf32) <- (1x21x768xf32, 1xi64)
        unsqueeze_7 = paddle._C_ops.unsqueeze(layer_norm_6, full_int_array_2)

        # pd_op.depthwise_conv2d: (1x1x21x768xf32) <- (1x1x21x768xf32, 768x1x1x9xf32)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(
            unsqueeze_7, unsqueeze_6, [1, 1], [0, 4], "EXPLICIT", 768, [1, 1], "NHWC"
        )

        # pd_op.squeeze: (1x21x768xf32) <- (1x1x21x768xf32, 1xi64)
        squeeze_3 = paddle._C_ops.squeeze(depthwise_conv2d_1, full_int_array_2)

        # pd_op.assign: (384x768x1xf32) <- (384x768x1xf32)
        assign_365 = parameter_244
        del parameter_244

        # pd_op.unsqueeze: (384x768x1x1xf32) <- (384x768x1xf32, 1xi64)
        unsqueeze_8 = paddle._C_ops.unsqueeze(assign_365, full_int_array_1)

        # pd_op.unsqueeze: (1x1x21x768xf32) <- (1x21x768xf32, 1xi64)
        unsqueeze_9 = paddle._C_ops.unsqueeze(squeeze_3, full_int_array_2)

        # pd_op.conv2d: (1x1x21x384xf32) <- (1x1x21x768xf32, 384x768x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            unsqueeze_9, unsqueeze_8, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NHWC"
        )

        # pd_op.squeeze: (1x21x384xf32) <- (1x1x21x384xf32, 1xi64)
        squeeze_4 = paddle._C_ops.squeeze(conv2d_1, full_int_array_2)

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_17 = paddle._C_ops.add(squeeze_4, parameter_246)
        del parameter_246

        # pd_op.multiply: (1x21x384xf32) <- (1x21x384xf32, 1x21x384xf32)
        multiply_1 = paddle._C_ops.multiply(add_17, add_14)

        # pd_op.matmul: (1x21x54xf32) <- (1x21x384xf32, 384x54xf32)
        matmul_14 = paddle._C_ops.matmul(multiply_1, parameter_243, False, False)
        del parameter_243

        # pd_op.add: (1x21x54xf32) <- (1x21x54xf32, 54xf32)
        add_18 = paddle._C_ops.add(matmul_14, parameter_242)
        del parameter_242

        # pd_op.reshape: (126x9x1xf32) <- (1x21x54xf32, 3xi64)
        reshape_7 = paddle._C_ops.reshape(add_18, full_int_array_3)

        # pd_op.softmax: (126x9x1xf32) <- (126x9x1xf32)
        softmax_2 = paddle._C_ops.softmax(reshape_7, 1)
        del reshape_7

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_15 = paddle._C_ops.matmul(layer_norm_6, parameter_241, False, False)
        del parameter_241

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_19 = paddle._C_ops.add(matmul_15, parameter_240)
        del parameter_240

        # pd_op.unsqueeze: (1x21x1x1x384xf32) <- (1x21x384xf32, 2xi64)
        unsqueeze_10 = paddle._C_ops.unsqueeze(add_19, full_int_array_4)

        # pd_op.pad3d: (1x29x1x1x384xf32) <- (1x21x1x1x384xf32, 6xi64)
        pad3d_1 = paddle._C_ops.pad3d(
            unsqueeze_10, full_int_array_5, "constant", float("0"), "NDHWC"
        )

        # pd_op.squeeze: (1x29x384xf32) <- (1x29x1x1x384xf32, 2xi64)
        squeeze_5 = paddle._C_ops.squeeze(pad3d_1, full_int_array_4)

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            squeeze_5, [1], full_int_array_6, full_int_array_7, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            squeeze_5, [1], full_int_array_8, full_int_array_9, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            squeeze_5, [1], full_int_array_10, full_int_array_11, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            squeeze_5, [1], full_int_array_12, full_int_array_13, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            squeeze_5, [1], full_int_array_14, full_int_array_15, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            squeeze_5, [1], full_int_array_16, full_int_array_17, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            squeeze_5, [1], full_int_array_18, full_int_array_19, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            squeeze_5, [1], full_int_array_20, full_int_array_21, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            squeeze_5, [1], full_int_array_22, full_int_array_23, [1], []
        )

        # builtin.combine: ([1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32]) <- (1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32)
        combine_2 = [
            slice_9,
            slice_10,
            slice_11,
            slice_12,
            slice_13,
            slice_14,
            slice_15,
            slice_16,
            slice_17,
        ]

        # pd_op.stack: (1x21x384x9xf32) <- ([1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32])
        stack_1 = paddle._C_ops.stack(combine_2, -1)
        del combine_2

        # pd_op.reshape: (126x64x9xf32) <- (1x21x384x9xf32, 3xi64)
        reshape_8 = paddle._C_ops.reshape(stack_1, full_int_array_24)

        # pd_op.matmul: (126x64x1xf32) <- (126x64x9xf32, 126x9x1xf32)
        matmul_16 = paddle._C_ops.matmul(reshape_8, softmax_2, False, False)

        # pd_op.reshape: (1x21x6x64xf32) <- (126x64x1xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(matmul_16, full_int_array_25)

        # pd_op.reshape: (1x21x6x64xf32) <- (1x21x384xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(add_14, full_int_array_26)

        # pd_op.transpose: (1x6x21x64xf32) <- (1x21x6x64xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_10, [0, 2, 1, 3])
        del reshape_10

        # pd_op.reshape: (1x21x6x64xf32) <- (1x21x384xf32, 4xi64)
        reshape_11 = paddle._C_ops.reshape(add_15, full_int_array_26)

        # pd_op.transpose: (1x6x21x64xf32) <- (1x21x6x64xf32)
        transpose_5 = paddle._C_ops.transpose(reshape_11, [0, 2, 1, 3])
        del reshape_11

        # pd_op.reshape: (1x21x6x64xf32) <- (1x21x384xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(add_16, full_int_array_26)

        # pd_op.transpose: (1x6x21x64xf32) <- (1x21x6x64xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_12, [0, 2, 1, 3])
        del reshape_12

        # pd_op.matmul: (1x6x21x21xf32) <- (1x6x21x64xf32, 1x6x21x64xf32)
        matmul_17 = paddle._C_ops.matmul(transpose_4, transpose_5, False, True)

        # pd_op.scale: (1x6x21x21xf32) <- (1x6x21x21xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(matmul_17, full_5, float("0"), True)
        del matmul_17

        # pd_op.add: (1x6x21x21xf32) <- (1x6x21x21xf32, 1x1x1x21xf32)
        add_20 = paddle._C_ops.add(scale_2, unsqueeze_0)

        # pd_op.softmax: (1x6x21x21xf32) <- (1x6x21x21xf32)
        softmax_3 = paddle._C_ops.softmax(add_20, -1)
        del add_20

        # pd_op.dropout: (1x6x21x21xf32, 1x6x21x21xui8) <- (1x6x21x21xf32, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_3, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x6x21x64xf32) <- (1x6x21x21xf32, 1x6x21x64xf32)
        matmul_18 = paddle._C_ops.matmul(dropout_8, transpose_6, False, False)

        # pd_op.transpose: (1x21x6x64xf32) <- (1x6x21x64xf32)
        transpose_7 = paddle._C_ops.transpose(matmul_18, [0, 2, 1, 3])
        del matmul_18

        # builtin.combine: ([1x21x6x64xf32, 1x21x6x64xf32]) <- (1x21x6x64xf32, 1x21x6x64xf32)
        combine_3 = [transpose_7, reshape_9]

        # pd_op.concat: (1x21x12x64xf32) <- ([1x21x6x64xf32, 1x21x6x64xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_3, full_6)
        del combine_3

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_13 = paddle._C_ops.reshape(concat_1, full_int_array_27)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_19 = paddle._C_ops.matmul(reshape_13, parameter_248, False, False)
        del parameter_248

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_21 = paddle._C_ops.add(matmul_19, parameter_247)
        del parameter_247

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_10, dropout_11 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_21, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_21

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_22 = paddle._C_ops.add(layer_norm_6, dropout_10)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_22, parameter_235, parameter_234, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_234, parameter_235

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_20 = paddle._C_ops.matmul(layer_norm_9, parameter_239, False, False)
        del parameter_239

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_23 = paddle._C_ops.add(matmul_20, parameter_238)
        del parameter_238

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_1 = paddle._C_ops.gelu(add_23, False)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_21 = paddle._C_ops.matmul(gelu_1, parameter_237, False, False)
        del parameter_237

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_24 = paddle._C_ops.add(matmul_21, parameter_236)
        del parameter_236

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_12, dropout_13 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_24, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_24

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_25 = paddle._C_ops.add(layer_norm_9, dropout_12)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_25, parameter_233, parameter_232, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_232, parameter_233

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_22 = paddle._C_ops.matmul(layer_norm_12, parameter_231, False, False)
        del parameter_231

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_26 = paddle._C_ops.add(matmul_22, parameter_230)
        del parameter_230

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_23 = paddle._C_ops.matmul(layer_norm_12, parameter_229, False, False)
        del parameter_229

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_27 = paddle._C_ops.add(matmul_23, parameter_228)
        del parameter_228

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_24 = paddle._C_ops.matmul(layer_norm_12, parameter_227, False, False)
        del parameter_227

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_28 = paddle._C_ops.add(matmul_24, parameter_226)
        del parameter_226

        # pd_op.assign: (768x1x9xf32) <- (768x1x9xf32)
        assign_366 = parameter_222
        del parameter_222

        # pd_op.unsqueeze: (768x1x1x9xf32) <- (768x1x9xf32, 1xi64)
        unsqueeze_11 = paddle._C_ops.unsqueeze(assign_366, full_int_array_1)

        # pd_op.unsqueeze: (1x1x21x768xf32) <- (1x21x768xf32, 1xi64)
        unsqueeze_12 = paddle._C_ops.unsqueeze(layer_norm_12, full_int_array_2)

        # pd_op.depthwise_conv2d: (1x1x21x768xf32) <- (1x1x21x768xf32, 768x1x1x9xf32)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(
            unsqueeze_12, unsqueeze_11, [1, 1], [0, 4], "EXPLICIT", 768, [1, 1], "NHWC"
        )

        # pd_op.squeeze: (1x21x768xf32) <- (1x1x21x768xf32, 1xi64)
        squeeze_6 = paddle._C_ops.squeeze(depthwise_conv2d_2, full_int_array_2)

        # pd_op.assign: (384x768x1xf32) <- (384x768x1xf32)
        assign_367 = parameter_221
        del parameter_221

        # pd_op.unsqueeze: (384x768x1x1xf32) <- (384x768x1xf32, 1xi64)
        unsqueeze_13 = paddle._C_ops.unsqueeze(assign_367, full_int_array_1)

        # pd_op.unsqueeze: (1x1x21x768xf32) <- (1x21x768xf32, 1xi64)
        unsqueeze_14 = paddle._C_ops.unsqueeze(squeeze_6, full_int_array_2)

        # pd_op.conv2d: (1x1x21x384xf32) <- (1x1x21x768xf32, 384x768x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            unsqueeze_14, unsqueeze_13, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NHWC"
        )

        # pd_op.squeeze: (1x21x384xf32) <- (1x1x21x384xf32, 1xi64)
        squeeze_7 = paddle._C_ops.squeeze(conv2d_2, full_int_array_2)

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_29 = paddle._C_ops.add(squeeze_7, parameter_223)
        del parameter_223

        # pd_op.multiply: (1x21x384xf32) <- (1x21x384xf32, 1x21x384xf32)
        multiply_2 = paddle._C_ops.multiply(add_29, add_26)

        # pd_op.matmul: (1x21x54xf32) <- (1x21x384xf32, 384x54xf32)
        matmul_25 = paddle._C_ops.matmul(multiply_2, parameter_220, False, False)
        del parameter_220

        # pd_op.add: (1x21x54xf32) <- (1x21x54xf32, 54xf32)
        add_30 = paddle._C_ops.add(matmul_25, parameter_219)
        del parameter_219

        # pd_op.reshape: (126x9x1xf32) <- (1x21x54xf32, 3xi64)
        reshape_14 = paddle._C_ops.reshape(add_30, full_int_array_3)

        # pd_op.softmax: (126x9x1xf32) <- (126x9x1xf32)
        softmax_4 = paddle._C_ops.softmax(reshape_14, 1)
        del reshape_14

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_26 = paddle._C_ops.matmul(layer_norm_12, parameter_218, False, False)
        del parameter_218

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_31 = paddle._C_ops.add(matmul_26, parameter_217)
        del parameter_217

        # pd_op.unsqueeze: (1x21x1x1x384xf32) <- (1x21x384xf32, 2xi64)
        unsqueeze_15 = paddle._C_ops.unsqueeze(add_31, full_int_array_4)

        # pd_op.pad3d: (1x29x1x1x384xf32) <- (1x21x1x1x384xf32, 6xi64)
        pad3d_2 = paddle._C_ops.pad3d(
            unsqueeze_15, full_int_array_5, "constant", float("0"), "NDHWC"
        )

        # pd_op.squeeze: (1x29x384xf32) <- (1x29x1x1x384xf32, 2xi64)
        squeeze_8 = paddle._C_ops.squeeze(pad3d_2, full_int_array_4)

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(
            squeeze_8, [1], full_int_array_6, full_int_array_7, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(
            squeeze_8, [1], full_int_array_8, full_int_array_9, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(
            squeeze_8, [1], full_int_array_10, full_int_array_11, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(
            squeeze_8, [1], full_int_array_12, full_int_array_13, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(
            squeeze_8, [1], full_int_array_14, full_int_array_15, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(
            squeeze_8, [1], full_int_array_16, full_int_array_17, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(
            squeeze_8, [1], full_int_array_18, full_int_array_19, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(
            squeeze_8, [1], full_int_array_20, full_int_array_21, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(
            squeeze_8, [1], full_int_array_22, full_int_array_23, [1], []
        )

        # builtin.combine: ([1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32]) <- (1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32)
        combine_4 = [
            slice_18,
            slice_19,
            slice_20,
            slice_21,
            slice_22,
            slice_23,
            slice_24,
            slice_25,
            slice_26,
        ]

        # pd_op.stack: (1x21x384x9xf32) <- ([1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32])
        stack_2 = paddle._C_ops.stack(combine_4, -1)
        del combine_4

        # pd_op.reshape: (126x64x9xf32) <- (1x21x384x9xf32, 3xi64)
        reshape_15 = paddle._C_ops.reshape(stack_2, full_int_array_24)

        # pd_op.matmul: (126x64x1xf32) <- (126x64x9xf32, 126x9x1xf32)
        matmul_27 = paddle._C_ops.matmul(reshape_15, softmax_4, False, False)

        # pd_op.reshape: (1x21x6x64xf32) <- (126x64x1xf32, 4xi64)
        reshape_16 = paddle._C_ops.reshape(matmul_27, full_int_array_25)

        # pd_op.reshape: (1x21x6x64xf32) <- (1x21x384xf32, 4xi64)
        reshape_17 = paddle._C_ops.reshape(add_26, full_int_array_26)

        # pd_op.transpose: (1x6x21x64xf32) <- (1x21x6x64xf32)
        transpose_8 = paddle._C_ops.transpose(reshape_17, [0, 2, 1, 3])
        del reshape_17

        # pd_op.reshape: (1x21x6x64xf32) <- (1x21x384xf32, 4xi64)
        reshape_18 = paddle._C_ops.reshape(add_27, full_int_array_26)

        # pd_op.transpose: (1x6x21x64xf32) <- (1x21x6x64xf32)
        transpose_9 = paddle._C_ops.transpose(reshape_18, [0, 2, 1, 3])
        del reshape_18

        # pd_op.reshape: (1x21x6x64xf32) <- (1x21x384xf32, 4xi64)
        reshape_19 = paddle._C_ops.reshape(add_28, full_int_array_26)

        # pd_op.transpose: (1x6x21x64xf32) <- (1x21x6x64xf32)
        transpose_10 = paddle._C_ops.transpose(reshape_19, [0, 2, 1, 3])
        del reshape_19

        # pd_op.matmul: (1x6x21x21xf32) <- (1x6x21x64xf32, 1x6x21x64xf32)
        matmul_28 = paddle._C_ops.matmul(transpose_8, transpose_9, False, True)

        # pd_op.scale: (1x6x21x21xf32) <- (1x6x21x21xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(matmul_28, full_5, float("0"), True)
        del matmul_28

        # pd_op.add: (1x6x21x21xf32) <- (1x6x21x21xf32, 1x1x1x21xf32)
        add_32 = paddle._C_ops.add(scale_3, unsqueeze_0)

        # pd_op.softmax: (1x6x21x21xf32) <- (1x6x21x21xf32)
        softmax_5 = paddle._C_ops.softmax(add_32, -1)
        del add_32

        # pd_op.dropout: (1x6x21x21xf32, 1x6x21x21xui8) <- (1x6x21x21xf32, None, 1xf32)
        dropout_14, dropout_15 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_5, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x6x21x64xf32) <- (1x6x21x21xf32, 1x6x21x64xf32)
        matmul_29 = paddle._C_ops.matmul(dropout_14, transpose_10, False, False)

        # pd_op.transpose: (1x21x6x64xf32) <- (1x6x21x64xf32)
        transpose_11 = paddle._C_ops.transpose(matmul_29, [0, 2, 1, 3])
        del matmul_29

        # builtin.combine: ([1x21x6x64xf32, 1x21x6x64xf32]) <- (1x21x6x64xf32, 1x21x6x64xf32)
        combine_5 = [transpose_11, reshape_16]

        # pd_op.concat: (1x21x12x64xf32) <- ([1x21x6x64xf32, 1x21x6x64xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_5, full_6)
        del combine_5

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_20 = paddle._C_ops.reshape(concat_2, full_int_array_27)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_30 = paddle._C_ops.matmul(reshape_20, parameter_225, False, False)
        del parameter_225

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_33 = paddle._C_ops.add(matmul_30, parameter_224)
        del parameter_224

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_16, dropout_17 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_33, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_33

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_34 = paddle._C_ops.add(layer_norm_12, dropout_16)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_34, parameter_212, parameter_211, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_211, parameter_212

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_31 = paddle._C_ops.matmul(layer_norm_15, parameter_216, False, False)
        del parameter_216

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_35 = paddle._C_ops.add(matmul_31, parameter_215)
        del parameter_215

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_2 = paddle._C_ops.gelu(add_35, False)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_32 = paddle._C_ops.matmul(gelu_2, parameter_214, False, False)
        del parameter_214

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_36 = paddle._C_ops.add(matmul_32, parameter_213)
        del parameter_213

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_18, dropout_19 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_36, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_36

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_37 = paddle._C_ops.add(layer_norm_15, dropout_18)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_37, parameter_210, parameter_209, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_209, parameter_210

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_33 = paddle._C_ops.matmul(layer_norm_18, parameter_208, False, False)
        del parameter_208

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_38 = paddle._C_ops.add(matmul_33, parameter_207)
        del parameter_207

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_34 = paddle._C_ops.matmul(layer_norm_18, parameter_206, False, False)
        del parameter_206

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_39 = paddle._C_ops.add(matmul_34, parameter_205)
        del parameter_205

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_35 = paddle._C_ops.matmul(layer_norm_18, parameter_204, False, False)
        del parameter_204

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_40 = paddle._C_ops.add(matmul_35, parameter_203)
        del parameter_203

        # pd_op.assign: (768x1x9xf32) <- (768x1x9xf32)
        assign_368 = parameter_199
        del parameter_199

        # pd_op.unsqueeze: (768x1x1x9xf32) <- (768x1x9xf32, 1xi64)
        unsqueeze_16 = paddle._C_ops.unsqueeze(assign_368, full_int_array_1)

        # pd_op.unsqueeze: (1x1x21x768xf32) <- (1x21x768xf32, 1xi64)
        unsqueeze_17 = paddle._C_ops.unsqueeze(layer_norm_18, full_int_array_2)

        # pd_op.depthwise_conv2d: (1x1x21x768xf32) <- (1x1x21x768xf32, 768x1x1x9xf32)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(
            unsqueeze_17, unsqueeze_16, [1, 1], [0, 4], "EXPLICIT", 768, [1, 1], "NHWC"
        )

        # pd_op.squeeze: (1x21x768xf32) <- (1x1x21x768xf32, 1xi64)
        squeeze_9 = paddle._C_ops.squeeze(depthwise_conv2d_3, full_int_array_2)

        # pd_op.assign: (384x768x1xf32) <- (384x768x1xf32)
        assign_369 = parameter_198
        del parameter_198

        # pd_op.unsqueeze: (384x768x1x1xf32) <- (384x768x1xf32, 1xi64)
        unsqueeze_18 = paddle._C_ops.unsqueeze(assign_369, full_int_array_1)

        # pd_op.unsqueeze: (1x1x21x768xf32) <- (1x21x768xf32, 1xi64)
        unsqueeze_19 = paddle._C_ops.unsqueeze(squeeze_9, full_int_array_2)

        # pd_op.conv2d: (1x1x21x384xf32) <- (1x1x21x768xf32, 384x768x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            unsqueeze_19, unsqueeze_18, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NHWC"
        )

        # pd_op.squeeze: (1x21x384xf32) <- (1x1x21x384xf32, 1xi64)
        squeeze_10 = paddle._C_ops.squeeze(conv2d_3, full_int_array_2)

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_41 = paddle._C_ops.add(squeeze_10, parameter_200)
        del parameter_200

        # pd_op.multiply: (1x21x384xf32) <- (1x21x384xf32, 1x21x384xf32)
        multiply_3 = paddle._C_ops.multiply(add_41, add_38)

        # pd_op.matmul: (1x21x54xf32) <- (1x21x384xf32, 384x54xf32)
        matmul_36 = paddle._C_ops.matmul(multiply_3, parameter_197, False, False)
        del parameter_197

        # pd_op.add: (1x21x54xf32) <- (1x21x54xf32, 54xf32)
        add_42 = paddle._C_ops.add(matmul_36, parameter_196)
        del parameter_196

        # pd_op.reshape: (126x9x1xf32) <- (1x21x54xf32, 3xi64)
        reshape_21 = paddle._C_ops.reshape(add_42, full_int_array_3)

        # pd_op.softmax: (126x9x1xf32) <- (126x9x1xf32)
        softmax_6 = paddle._C_ops.softmax(reshape_21, 1)
        del reshape_21

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_37 = paddle._C_ops.matmul(layer_norm_18, parameter_195, False, False)
        del parameter_195

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_43 = paddle._C_ops.add(matmul_37, parameter_194)
        del parameter_194

        # pd_op.unsqueeze: (1x21x1x1x384xf32) <- (1x21x384xf32, 2xi64)
        unsqueeze_20 = paddle._C_ops.unsqueeze(add_43, full_int_array_4)

        # pd_op.pad3d: (1x29x1x1x384xf32) <- (1x21x1x1x384xf32, 6xi64)
        pad3d_3 = paddle._C_ops.pad3d(
            unsqueeze_20, full_int_array_5, "constant", float("0"), "NDHWC"
        )

        # pd_op.squeeze: (1x29x384xf32) <- (1x29x1x1x384xf32, 2xi64)
        squeeze_11 = paddle._C_ops.squeeze(pad3d_3, full_int_array_4)

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(
            squeeze_11, [1], full_int_array_6, full_int_array_7, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(
            squeeze_11, [1], full_int_array_8, full_int_array_9, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(
            squeeze_11, [1], full_int_array_10, full_int_array_11, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(
            squeeze_11, [1], full_int_array_12, full_int_array_13, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(
            squeeze_11, [1], full_int_array_14, full_int_array_15, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(
            squeeze_11, [1], full_int_array_16, full_int_array_17, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_33 = paddle._C_ops.slice(
            squeeze_11, [1], full_int_array_18, full_int_array_19, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_34 = paddle._C_ops.slice(
            squeeze_11, [1], full_int_array_20, full_int_array_21, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_35 = paddle._C_ops.slice(
            squeeze_11, [1], full_int_array_22, full_int_array_23, [1], []
        )

        # builtin.combine: ([1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32]) <- (1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32)
        combine_6 = [
            slice_27,
            slice_28,
            slice_29,
            slice_30,
            slice_31,
            slice_32,
            slice_33,
            slice_34,
            slice_35,
        ]

        # pd_op.stack: (1x21x384x9xf32) <- ([1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32])
        stack_3 = paddle._C_ops.stack(combine_6, -1)
        del combine_6

        # pd_op.reshape: (126x64x9xf32) <- (1x21x384x9xf32, 3xi64)
        reshape_22 = paddle._C_ops.reshape(stack_3, full_int_array_24)

        # pd_op.matmul: (126x64x1xf32) <- (126x64x9xf32, 126x9x1xf32)
        matmul_38 = paddle._C_ops.matmul(reshape_22, softmax_6, False, False)

        # pd_op.reshape: (1x21x6x64xf32) <- (126x64x1xf32, 4xi64)
        reshape_23 = paddle._C_ops.reshape(matmul_38, full_int_array_25)

        # pd_op.reshape: (1x21x6x64xf32) <- (1x21x384xf32, 4xi64)
        reshape_24 = paddle._C_ops.reshape(add_38, full_int_array_26)

        # pd_op.transpose: (1x6x21x64xf32) <- (1x21x6x64xf32)
        transpose_12 = paddle._C_ops.transpose(reshape_24, [0, 2, 1, 3])
        del reshape_24

        # pd_op.reshape: (1x21x6x64xf32) <- (1x21x384xf32, 4xi64)
        reshape_25 = paddle._C_ops.reshape(add_39, full_int_array_26)

        # pd_op.transpose: (1x6x21x64xf32) <- (1x21x6x64xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_25, [0, 2, 1, 3])
        del reshape_25

        # pd_op.reshape: (1x21x6x64xf32) <- (1x21x384xf32, 4xi64)
        reshape_26 = paddle._C_ops.reshape(add_40, full_int_array_26)

        # pd_op.transpose: (1x6x21x64xf32) <- (1x21x6x64xf32)
        transpose_14 = paddle._C_ops.transpose(reshape_26, [0, 2, 1, 3])
        del reshape_26

        # pd_op.matmul: (1x6x21x21xf32) <- (1x6x21x64xf32, 1x6x21x64xf32)
        matmul_39 = paddle._C_ops.matmul(transpose_12, transpose_13, False, True)

        # pd_op.scale: (1x6x21x21xf32) <- (1x6x21x21xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(matmul_39, full_5, float("0"), True)
        del matmul_39

        # pd_op.add: (1x6x21x21xf32) <- (1x6x21x21xf32, 1x1x1x21xf32)
        add_44 = paddle._C_ops.add(scale_4, unsqueeze_0)

        # pd_op.softmax: (1x6x21x21xf32) <- (1x6x21x21xf32)
        softmax_7 = paddle._C_ops.softmax(add_44, -1)
        del add_44

        # pd_op.dropout: (1x6x21x21xf32, 1x6x21x21xui8) <- (1x6x21x21xf32, None, 1xf32)
        dropout_20, dropout_21 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_7, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x6x21x64xf32) <- (1x6x21x21xf32, 1x6x21x64xf32)
        matmul_40 = paddle._C_ops.matmul(dropout_20, transpose_14, False, False)

        # pd_op.transpose: (1x21x6x64xf32) <- (1x6x21x64xf32)
        transpose_15 = paddle._C_ops.transpose(matmul_40, [0, 2, 1, 3])
        del matmul_40

        # builtin.combine: ([1x21x6x64xf32, 1x21x6x64xf32]) <- (1x21x6x64xf32, 1x21x6x64xf32)
        combine_7 = [transpose_15, reshape_23]

        # pd_op.concat: (1x21x12x64xf32) <- ([1x21x6x64xf32, 1x21x6x64xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_7, full_6)
        del combine_7

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_27 = paddle._C_ops.reshape(concat_3, full_int_array_27)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_41 = paddle._C_ops.matmul(reshape_27, parameter_202, False, False)
        del parameter_202

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_45 = paddle._C_ops.add(matmul_41, parameter_201)
        del parameter_201

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_22, dropout_23 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_45, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_45

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_46 = paddle._C_ops.add(layer_norm_18, dropout_22)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_46, parameter_189, parameter_188, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_188, parameter_189

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_42 = paddle._C_ops.matmul(layer_norm_21, parameter_193, False, False)
        del parameter_193

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_47 = paddle._C_ops.add(matmul_42, parameter_192)
        del parameter_192

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_3 = paddle._C_ops.gelu(add_47, False)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_43 = paddle._C_ops.matmul(gelu_3, parameter_191, False, False)
        del parameter_191

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_48 = paddle._C_ops.add(matmul_43, parameter_190)
        del parameter_190

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_24, dropout_25 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_48, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_48

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_49 = paddle._C_ops.add(layer_norm_21, dropout_24)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_49, parameter_187, parameter_186, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_186, parameter_187

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_44 = paddle._C_ops.matmul(layer_norm_24, parameter_185, False, False)
        del parameter_185

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_50 = paddle._C_ops.add(matmul_44, parameter_184)
        del parameter_184

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_45 = paddle._C_ops.matmul(layer_norm_24, parameter_183, False, False)
        del parameter_183

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_51 = paddle._C_ops.add(matmul_45, parameter_182)
        del parameter_182

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_46 = paddle._C_ops.matmul(layer_norm_24, parameter_181, False, False)
        del parameter_181

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_52 = paddle._C_ops.add(matmul_46, parameter_180)
        del parameter_180

        # pd_op.assign: (768x1x9xf32) <- (768x1x9xf32)
        assign_370 = parameter_176
        del parameter_176

        # pd_op.unsqueeze: (768x1x1x9xf32) <- (768x1x9xf32, 1xi64)
        unsqueeze_21 = paddle._C_ops.unsqueeze(assign_370, full_int_array_1)

        # pd_op.unsqueeze: (1x1x21x768xf32) <- (1x21x768xf32, 1xi64)
        unsqueeze_22 = paddle._C_ops.unsqueeze(layer_norm_24, full_int_array_2)

        # pd_op.depthwise_conv2d: (1x1x21x768xf32) <- (1x1x21x768xf32, 768x1x1x9xf32)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(
            unsqueeze_22, unsqueeze_21, [1, 1], [0, 4], "EXPLICIT", 768, [1, 1], "NHWC"
        )

        # pd_op.squeeze: (1x21x768xf32) <- (1x1x21x768xf32, 1xi64)
        squeeze_12 = paddle._C_ops.squeeze(depthwise_conv2d_4, full_int_array_2)

        # pd_op.assign: (384x768x1xf32) <- (384x768x1xf32)
        assign_371 = parameter_175
        del parameter_175

        # pd_op.unsqueeze: (384x768x1x1xf32) <- (384x768x1xf32, 1xi64)
        unsqueeze_23 = paddle._C_ops.unsqueeze(assign_371, full_int_array_1)

        # pd_op.unsqueeze: (1x1x21x768xf32) <- (1x21x768xf32, 1xi64)
        unsqueeze_24 = paddle._C_ops.unsqueeze(squeeze_12, full_int_array_2)

        # pd_op.conv2d: (1x1x21x384xf32) <- (1x1x21x768xf32, 384x768x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            unsqueeze_24, unsqueeze_23, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NHWC"
        )

        # pd_op.squeeze: (1x21x384xf32) <- (1x1x21x384xf32, 1xi64)
        squeeze_13 = paddle._C_ops.squeeze(conv2d_4, full_int_array_2)

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_53 = paddle._C_ops.add(squeeze_13, parameter_177)
        del parameter_177

        # pd_op.multiply: (1x21x384xf32) <- (1x21x384xf32, 1x21x384xf32)
        multiply_4 = paddle._C_ops.multiply(add_53, add_50)

        # pd_op.matmul: (1x21x54xf32) <- (1x21x384xf32, 384x54xf32)
        matmul_47 = paddle._C_ops.matmul(multiply_4, parameter_174, False, False)
        del parameter_174

        # pd_op.add: (1x21x54xf32) <- (1x21x54xf32, 54xf32)
        add_54 = paddle._C_ops.add(matmul_47, parameter_173)
        del parameter_173

        # pd_op.reshape: (126x9x1xf32) <- (1x21x54xf32, 3xi64)
        reshape_28 = paddle._C_ops.reshape(add_54, full_int_array_3)

        # pd_op.softmax: (126x9x1xf32) <- (126x9x1xf32)
        softmax_8 = paddle._C_ops.softmax(reshape_28, 1)
        del reshape_28

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_48 = paddle._C_ops.matmul(layer_norm_24, parameter_172, False, False)
        del parameter_172

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_55 = paddle._C_ops.add(matmul_48, parameter_171)
        del parameter_171

        # pd_op.unsqueeze: (1x21x1x1x384xf32) <- (1x21x384xf32, 2xi64)
        unsqueeze_25 = paddle._C_ops.unsqueeze(add_55, full_int_array_4)

        # pd_op.pad3d: (1x29x1x1x384xf32) <- (1x21x1x1x384xf32, 6xi64)
        pad3d_4 = paddle._C_ops.pad3d(
            unsqueeze_25, full_int_array_5, "constant", float("0"), "NDHWC"
        )

        # pd_op.squeeze: (1x29x384xf32) <- (1x29x1x1x384xf32, 2xi64)
        squeeze_14 = paddle._C_ops.squeeze(pad3d_4, full_int_array_4)

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(
            squeeze_14, [1], full_int_array_6, full_int_array_7, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_37 = paddle._C_ops.slice(
            squeeze_14, [1], full_int_array_8, full_int_array_9, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_38 = paddle._C_ops.slice(
            squeeze_14, [1], full_int_array_10, full_int_array_11, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_39 = paddle._C_ops.slice(
            squeeze_14, [1], full_int_array_12, full_int_array_13, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_40 = paddle._C_ops.slice(
            squeeze_14, [1], full_int_array_14, full_int_array_15, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_41 = paddle._C_ops.slice(
            squeeze_14, [1], full_int_array_16, full_int_array_17, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_42 = paddle._C_ops.slice(
            squeeze_14, [1], full_int_array_18, full_int_array_19, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_43 = paddle._C_ops.slice(
            squeeze_14, [1], full_int_array_20, full_int_array_21, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_44 = paddle._C_ops.slice(
            squeeze_14, [1], full_int_array_22, full_int_array_23, [1], []
        )

        # builtin.combine: ([1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32]) <- (1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32)
        combine_8 = [
            slice_36,
            slice_37,
            slice_38,
            slice_39,
            slice_40,
            slice_41,
            slice_42,
            slice_43,
            slice_44,
        ]

        # pd_op.stack: (1x21x384x9xf32) <- ([1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32])
        stack_4 = paddle._C_ops.stack(combine_8, -1)
        del combine_8

        # pd_op.reshape: (126x64x9xf32) <- (1x21x384x9xf32, 3xi64)
        reshape_29 = paddle._C_ops.reshape(stack_4, full_int_array_24)

        # pd_op.matmul: (126x64x1xf32) <- (126x64x9xf32, 126x9x1xf32)
        matmul_49 = paddle._C_ops.matmul(reshape_29, softmax_8, False, False)

        # pd_op.reshape: (1x21x6x64xf32) <- (126x64x1xf32, 4xi64)
        reshape_30 = paddle._C_ops.reshape(matmul_49, full_int_array_25)

        # pd_op.reshape: (1x21x6x64xf32) <- (1x21x384xf32, 4xi64)
        reshape_31 = paddle._C_ops.reshape(add_50, full_int_array_26)

        # pd_op.transpose: (1x6x21x64xf32) <- (1x21x6x64xf32)
        transpose_16 = paddle._C_ops.transpose(reshape_31, [0, 2, 1, 3])
        del reshape_31

        # pd_op.reshape: (1x21x6x64xf32) <- (1x21x384xf32, 4xi64)
        reshape_32 = paddle._C_ops.reshape(add_51, full_int_array_26)

        # pd_op.transpose: (1x6x21x64xf32) <- (1x21x6x64xf32)
        transpose_17 = paddle._C_ops.transpose(reshape_32, [0, 2, 1, 3])
        del reshape_32

        # pd_op.reshape: (1x21x6x64xf32) <- (1x21x384xf32, 4xi64)
        reshape_33 = paddle._C_ops.reshape(add_52, full_int_array_26)

        # pd_op.transpose: (1x6x21x64xf32) <- (1x21x6x64xf32)
        transpose_18 = paddle._C_ops.transpose(reshape_33, [0, 2, 1, 3])
        del reshape_33

        # pd_op.matmul: (1x6x21x21xf32) <- (1x6x21x64xf32, 1x6x21x64xf32)
        matmul_50 = paddle._C_ops.matmul(transpose_16, transpose_17, False, True)

        # pd_op.scale: (1x6x21x21xf32) <- (1x6x21x21xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(matmul_50, full_5, float("0"), True)
        del matmul_50

        # pd_op.add: (1x6x21x21xf32) <- (1x6x21x21xf32, 1x1x1x21xf32)
        add_56 = paddle._C_ops.add(scale_5, unsqueeze_0)

        # pd_op.softmax: (1x6x21x21xf32) <- (1x6x21x21xf32)
        softmax_9 = paddle._C_ops.softmax(add_56, -1)
        del add_56

        # pd_op.dropout: (1x6x21x21xf32, 1x6x21x21xui8) <- (1x6x21x21xf32, None, 1xf32)
        dropout_26, dropout_27 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_9, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x6x21x64xf32) <- (1x6x21x21xf32, 1x6x21x64xf32)
        matmul_51 = paddle._C_ops.matmul(dropout_26, transpose_18, False, False)

        # pd_op.transpose: (1x21x6x64xf32) <- (1x6x21x64xf32)
        transpose_19 = paddle._C_ops.transpose(matmul_51, [0, 2, 1, 3])
        del matmul_51

        # builtin.combine: ([1x21x6x64xf32, 1x21x6x64xf32]) <- (1x21x6x64xf32, 1x21x6x64xf32)
        combine_9 = [transpose_19, reshape_30]

        # pd_op.concat: (1x21x12x64xf32) <- ([1x21x6x64xf32, 1x21x6x64xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_9, full_6)
        del combine_9

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_34 = paddle._C_ops.reshape(concat_4, full_int_array_27)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_52 = paddle._C_ops.matmul(reshape_34, parameter_179, False, False)
        del parameter_179

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_57 = paddle._C_ops.add(matmul_52, parameter_178)
        del parameter_178

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_28, dropout_29 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_57, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_57

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_58 = paddle._C_ops.add(layer_norm_24, dropout_28)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_58, parameter_166, parameter_165, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_165, parameter_166

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_53 = paddle._C_ops.matmul(layer_norm_27, parameter_170, False, False)
        del parameter_170

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_59 = paddle._C_ops.add(matmul_53, parameter_169)
        del parameter_169

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_4 = paddle._C_ops.gelu(add_59, False)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_54 = paddle._C_ops.matmul(gelu_4, parameter_168, False, False)
        del parameter_168

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_60 = paddle._C_ops.add(matmul_54, parameter_167)
        del parameter_167

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_30, dropout_31 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_60, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_60

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_61 = paddle._C_ops.add(layer_norm_27, dropout_30)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_61, parameter_164, parameter_163, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_163, parameter_164

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_55 = paddle._C_ops.matmul(layer_norm_30, parameter_162, False, False)
        del parameter_162

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_62 = paddle._C_ops.add(matmul_55, parameter_161)
        del parameter_161

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_56 = paddle._C_ops.matmul(layer_norm_30, parameter_160, False, False)
        del parameter_160

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_63 = paddle._C_ops.add(matmul_56, parameter_159)
        del parameter_159

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_57 = paddle._C_ops.matmul(layer_norm_30, parameter_158, False, False)
        del parameter_158

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_64 = paddle._C_ops.add(matmul_57, parameter_157)
        del parameter_157

        # pd_op.assign: (768x1x9xf32) <- (768x1x9xf32)
        assign_372 = parameter_153
        del parameter_153

        # pd_op.unsqueeze: (768x1x1x9xf32) <- (768x1x9xf32, 1xi64)
        unsqueeze_26 = paddle._C_ops.unsqueeze(assign_372, full_int_array_1)

        # pd_op.unsqueeze: (1x1x21x768xf32) <- (1x21x768xf32, 1xi64)
        unsqueeze_27 = paddle._C_ops.unsqueeze(layer_norm_30, full_int_array_2)

        # pd_op.depthwise_conv2d: (1x1x21x768xf32) <- (1x1x21x768xf32, 768x1x1x9xf32)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(
            unsqueeze_27, unsqueeze_26, [1, 1], [0, 4], "EXPLICIT", 768, [1, 1], "NHWC"
        )

        # pd_op.squeeze: (1x21x768xf32) <- (1x1x21x768xf32, 1xi64)
        squeeze_15 = paddle._C_ops.squeeze(depthwise_conv2d_5, full_int_array_2)

        # pd_op.assign: (384x768x1xf32) <- (384x768x1xf32)
        assign_373 = parameter_152
        del parameter_152

        # pd_op.unsqueeze: (384x768x1x1xf32) <- (384x768x1xf32, 1xi64)
        unsqueeze_28 = paddle._C_ops.unsqueeze(assign_373, full_int_array_1)

        # pd_op.unsqueeze: (1x1x21x768xf32) <- (1x21x768xf32, 1xi64)
        unsqueeze_29 = paddle._C_ops.unsqueeze(squeeze_15, full_int_array_2)

        # pd_op.conv2d: (1x1x21x384xf32) <- (1x1x21x768xf32, 384x768x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            unsqueeze_29, unsqueeze_28, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NHWC"
        )

        # pd_op.squeeze: (1x21x384xf32) <- (1x1x21x384xf32, 1xi64)
        squeeze_16 = paddle._C_ops.squeeze(conv2d_5, full_int_array_2)

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_65 = paddle._C_ops.add(squeeze_16, parameter_154)
        del parameter_154

        # pd_op.multiply: (1x21x384xf32) <- (1x21x384xf32, 1x21x384xf32)
        multiply_5 = paddle._C_ops.multiply(add_65, add_62)

        # pd_op.matmul: (1x21x54xf32) <- (1x21x384xf32, 384x54xf32)
        matmul_58 = paddle._C_ops.matmul(multiply_5, parameter_151, False, False)
        del parameter_151

        # pd_op.add: (1x21x54xf32) <- (1x21x54xf32, 54xf32)
        add_66 = paddle._C_ops.add(matmul_58, parameter_150)
        del parameter_150

        # pd_op.reshape: (126x9x1xf32) <- (1x21x54xf32, 3xi64)
        reshape_35 = paddle._C_ops.reshape(add_66, full_int_array_3)

        # pd_op.softmax: (126x9x1xf32) <- (126x9x1xf32)
        softmax_10 = paddle._C_ops.softmax(reshape_35, 1)
        del reshape_35

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_59 = paddle._C_ops.matmul(layer_norm_30, parameter_149, False, False)
        del parameter_149

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_67 = paddle._C_ops.add(matmul_59, parameter_148)
        del parameter_148

        # pd_op.unsqueeze: (1x21x1x1x384xf32) <- (1x21x384xf32, 2xi64)
        unsqueeze_30 = paddle._C_ops.unsqueeze(add_67, full_int_array_4)

        # pd_op.pad3d: (1x29x1x1x384xf32) <- (1x21x1x1x384xf32, 6xi64)
        pad3d_5 = paddle._C_ops.pad3d(
            unsqueeze_30, full_int_array_5, "constant", float("0"), "NDHWC"
        )

        # pd_op.squeeze: (1x29x384xf32) <- (1x29x1x1x384xf32, 2xi64)
        squeeze_17 = paddle._C_ops.squeeze(pad3d_5, full_int_array_4)

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_45 = paddle._C_ops.slice(
            squeeze_17, [1], full_int_array_6, full_int_array_7, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_46 = paddle._C_ops.slice(
            squeeze_17, [1], full_int_array_8, full_int_array_9, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_47 = paddle._C_ops.slice(
            squeeze_17, [1], full_int_array_10, full_int_array_11, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_48 = paddle._C_ops.slice(
            squeeze_17, [1], full_int_array_12, full_int_array_13, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_49 = paddle._C_ops.slice(
            squeeze_17, [1], full_int_array_14, full_int_array_15, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_50 = paddle._C_ops.slice(
            squeeze_17, [1], full_int_array_16, full_int_array_17, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_51 = paddle._C_ops.slice(
            squeeze_17, [1], full_int_array_18, full_int_array_19, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_52 = paddle._C_ops.slice(
            squeeze_17, [1], full_int_array_20, full_int_array_21, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_53 = paddle._C_ops.slice(
            squeeze_17, [1], full_int_array_22, full_int_array_23, [1], []
        )

        # builtin.combine: ([1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32]) <- (1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32)
        combine_10 = [
            slice_45,
            slice_46,
            slice_47,
            slice_48,
            slice_49,
            slice_50,
            slice_51,
            slice_52,
            slice_53,
        ]

        # pd_op.stack: (1x21x384x9xf32) <- ([1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32])
        stack_5 = paddle._C_ops.stack(combine_10, -1)
        del combine_10

        # pd_op.reshape: (126x64x9xf32) <- (1x21x384x9xf32, 3xi64)
        reshape_36 = paddle._C_ops.reshape(stack_5, full_int_array_24)

        # pd_op.matmul: (126x64x1xf32) <- (126x64x9xf32, 126x9x1xf32)
        matmul_60 = paddle._C_ops.matmul(reshape_36, softmax_10, False, False)

        # pd_op.reshape: (1x21x6x64xf32) <- (126x64x1xf32, 4xi64)
        reshape_37 = paddle._C_ops.reshape(matmul_60, full_int_array_25)

        # pd_op.reshape: (1x21x6x64xf32) <- (1x21x384xf32, 4xi64)
        reshape_38 = paddle._C_ops.reshape(add_62, full_int_array_26)

        # pd_op.transpose: (1x6x21x64xf32) <- (1x21x6x64xf32)
        transpose_20 = paddle._C_ops.transpose(reshape_38, [0, 2, 1, 3])
        del reshape_38

        # pd_op.reshape: (1x21x6x64xf32) <- (1x21x384xf32, 4xi64)
        reshape_39 = paddle._C_ops.reshape(add_63, full_int_array_26)

        # pd_op.transpose: (1x6x21x64xf32) <- (1x21x6x64xf32)
        transpose_21 = paddle._C_ops.transpose(reshape_39, [0, 2, 1, 3])
        del reshape_39

        # pd_op.reshape: (1x21x6x64xf32) <- (1x21x384xf32, 4xi64)
        reshape_40 = paddle._C_ops.reshape(add_64, full_int_array_26)

        # pd_op.transpose: (1x6x21x64xf32) <- (1x21x6x64xf32)
        transpose_22 = paddle._C_ops.transpose(reshape_40, [0, 2, 1, 3])
        del reshape_40

        # pd_op.matmul: (1x6x21x21xf32) <- (1x6x21x64xf32, 1x6x21x64xf32)
        matmul_61 = paddle._C_ops.matmul(transpose_20, transpose_21, False, True)

        # pd_op.scale: (1x6x21x21xf32) <- (1x6x21x21xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(matmul_61, full_5, float("0"), True)
        del matmul_61

        # pd_op.add: (1x6x21x21xf32) <- (1x6x21x21xf32, 1x1x1x21xf32)
        add_68 = paddle._C_ops.add(scale_6, unsqueeze_0)

        # pd_op.softmax: (1x6x21x21xf32) <- (1x6x21x21xf32)
        softmax_11 = paddle._C_ops.softmax(add_68, -1)
        del add_68

        # pd_op.dropout: (1x6x21x21xf32, 1x6x21x21xui8) <- (1x6x21x21xf32, None, 1xf32)
        dropout_32, dropout_33 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_11, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x6x21x64xf32) <- (1x6x21x21xf32, 1x6x21x64xf32)
        matmul_62 = paddle._C_ops.matmul(dropout_32, transpose_22, False, False)

        # pd_op.transpose: (1x21x6x64xf32) <- (1x6x21x64xf32)
        transpose_23 = paddle._C_ops.transpose(matmul_62, [0, 2, 1, 3])
        del matmul_62

        # builtin.combine: ([1x21x6x64xf32, 1x21x6x64xf32]) <- (1x21x6x64xf32, 1x21x6x64xf32)
        combine_11 = [transpose_23, reshape_37]

        # pd_op.concat: (1x21x12x64xf32) <- ([1x21x6x64xf32, 1x21x6x64xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_11, full_6)
        del combine_11

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_41 = paddle._C_ops.reshape(concat_5, full_int_array_27)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_63 = paddle._C_ops.matmul(reshape_41, parameter_156, False, False)
        del parameter_156

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_69 = paddle._C_ops.add(matmul_63, parameter_155)
        del parameter_155

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_34, dropout_35 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_69, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_69

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_70 = paddle._C_ops.add(layer_norm_30, dropout_34)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_70, parameter_143, parameter_142, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_142, parameter_143

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_64 = paddle._C_ops.matmul(layer_norm_33, parameter_147, False, False)
        del parameter_147

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_71 = paddle._C_ops.add(matmul_64, parameter_146)
        del parameter_146

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_5 = paddle._C_ops.gelu(add_71, False)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_65 = paddle._C_ops.matmul(gelu_5, parameter_145, False, False)
        del parameter_145

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_72 = paddle._C_ops.add(matmul_65, parameter_144)
        del parameter_144

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_36, dropout_37 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_72, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_72

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_73 = paddle._C_ops.add(layer_norm_33, dropout_36)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_73, parameter_141, parameter_140, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_140, parameter_141

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_66 = paddle._C_ops.matmul(layer_norm_36, parameter_139, False, False)
        del parameter_139

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_74 = paddle._C_ops.add(matmul_66, parameter_138)
        del parameter_138

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_67 = paddle._C_ops.matmul(layer_norm_36, parameter_137, False, False)
        del parameter_137

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_75 = paddle._C_ops.add(matmul_67, parameter_136)
        del parameter_136

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_68 = paddle._C_ops.matmul(layer_norm_36, parameter_135, False, False)
        del parameter_135

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_76 = paddle._C_ops.add(matmul_68, parameter_134)
        del parameter_134

        # pd_op.assign: (768x1x9xf32) <- (768x1x9xf32)
        assign_374 = parameter_130
        del parameter_130

        # pd_op.unsqueeze: (768x1x1x9xf32) <- (768x1x9xf32, 1xi64)
        unsqueeze_31 = paddle._C_ops.unsqueeze(assign_374, full_int_array_1)

        # pd_op.unsqueeze: (1x1x21x768xf32) <- (1x21x768xf32, 1xi64)
        unsqueeze_32 = paddle._C_ops.unsqueeze(layer_norm_36, full_int_array_2)

        # pd_op.depthwise_conv2d: (1x1x21x768xf32) <- (1x1x21x768xf32, 768x1x1x9xf32)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(
            unsqueeze_32, unsqueeze_31, [1, 1], [0, 4], "EXPLICIT", 768, [1, 1], "NHWC"
        )

        # pd_op.squeeze: (1x21x768xf32) <- (1x1x21x768xf32, 1xi64)
        squeeze_18 = paddle._C_ops.squeeze(depthwise_conv2d_6, full_int_array_2)

        # pd_op.assign: (384x768x1xf32) <- (384x768x1xf32)
        assign_375 = parameter_129
        del parameter_129

        # pd_op.unsqueeze: (384x768x1x1xf32) <- (384x768x1xf32, 1xi64)
        unsqueeze_33 = paddle._C_ops.unsqueeze(assign_375, full_int_array_1)

        # pd_op.unsqueeze: (1x1x21x768xf32) <- (1x21x768xf32, 1xi64)
        unsqueeze_34 = paddle._C_ops.unsqueeze(squeeze_18, full_int_array_2)

        # pd_op.conv2d: (1x1x21x384xf32) <- (1x1x21x768xf32, 384x768x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            unsqueeze_34, unsqueeze_33, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NHWC"
        )

        # pd_op.squeeze: (1x21x384xf32) <- (1x1x21x384xf32, 1xi64)
        squeeze_19 = paddle._C_ops.squeeze(conv2d_6, full_int_array_2)

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_77 = paddle._C_ops.add(squeeze_19, parameter_131)
        del parameter_131

        # pd_op.multiply: (1x21x384xf32) <- (1x21x384xf32, 1x21x384xf32)
        multiply_6 = paddle._C_ops.multiply(add_77, add_74)

        # pd_op.matmul: (1x21x54xf32) <- (1x21x384xf32, 384x54xf32)
        matmul_69 = paddle._C_ops.matmul(multiply_6, parameter_128, False, False)
        del parameter_128

        # pd_op.add: (1x21x54xf32) <- (1x21x54xf32, 54xf32)
        add_78 = paddle._C_ops.add(matmul_69, parameter_127)
        del parameter_127

        # pd_op.reshape: (126x9x1xf32) <- (1x21x54xf32, 3xi64)
        reshape_42 = paddle._C_ops.reshape(add_78, full_int_array_3)

        # pd_op.softmax: (126x9x1xf32) <- (126x9x1xf32)
        softmax_12 = paddle._C_ops.softmax(reshape_42, 1)
        del reshape_42

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_70 = paddle._C_ops.matmul(layer_norm_36, parameter_126, False, False)
        del parameter_126

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_79 = paddle._C_ops.add(matmul_70, parameter_125)
        del parameter_125

        # pd_op.unsqueeze: (1x21x1x1x384xf32) <- (1x21x384xf32, 2xi64)
        unsqueeze_35 = paddle._C_ops.unsqueeze(add_79, full_int_array_4)

        # pd_op.pad3d: (1x29x1x1x384xf32) <- (1x21x1x1x384xf32, 6xi64)
        pad3d_6 = paddle._C_ops.pad3d(
            unsqueeze_35, full_int_array_5, "constant", float("0"), "NDHWC"
        )

        # pd_op.squeeze: (1x29x384xf32) <- (1x29x1x1x384xf32, 2xi64)
        squeeze_20 = paddle._C_ops.squeeze(pad3d_6, full_int_array_4)

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_54 = paddle._C_ops.slice(
            squeeze_20, [1], full_int_array_6, full_int_array_7, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_55 = paddle._C_ops.slice(
            squeeze_20, [1], full_int_array_8, full_int_array_9, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_56 = paddle._C_ops.slice(
            squeeze_20, [1], full_int_array_10, full_int_array_11, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_57 = paddle._C_ops.slice(
            squeeze_20, [1], full_int_array_12, full_int_array_13, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_58 = paddle._C_ops.slice(
            squeeze_20, [1], full_int_array_14, full_int_array_15, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_59 = paddle._C_ops.slice(
            squeeze_20, [1], full_int_array_16, full_int_array_17, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_60 = paddle._C_ops.slice(
            squeeze_20, [1], full_int_array_18, full_int_array_19, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_61 = paddle._C_ops.slice(
            squeeze_20, [1], full_int_array_20, full_int_array_21, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_62 = paddle._C_ops.slice(
            squeeze_20, [1], full_int_array_22, full_int_array_23, [1], []
        )

        # builtin.combine: ([1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32]) <- (1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32)
        combine_12 = [
            slice_54,
            slice_55,
            slice_56,
            slice_57,
            slice_58,
            slice_59,
            slice_60,
            slice_61,
            slice_62,
        ]

        # pd_op.stack: (1x21x384x9xf32) <- ([1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32])
        stack_6 = paddle._C_ops.stack(combine_12, -1)
        del combine_12

        # pd_op.reshape: (126x64x9xf32) <- (1x21x384x9xf32, 3xi64)
        reshape_43 = paddle._C_ops.reshape(stack_6, full_int_array_24)

        # pd_op.matmul: (126x64x1xf32) <- (126x64x9xf32, 126x9x1xf32)
        matmul_71 = paddle._C_ops.matmul(reshape_43, softmax_12, False, False)

        # pd_op.reshape: (1x21x6x64xf32) <- (126x64x1xf32, 4xi64)
        reshape_44 = paddle._C_ops.reshape(matmul_71, full_int_array_25)

        # pd_op.reshape: (1x21x6x64xf32) <- (1x21x384xf32, 4xi64)
        reshape_45 = paddle._C_ops.reshape(add_74, full_int_array_26)

        # pd_op.transpose: (1x6x21x64xf32) <- (1x21x6x64xf32)
        transpose_24 = paddle._C_ops.transpose(reshape_45, [0, 2, 1, 3])
        del reshape_45

        # pd_op.reshape: (1x21x6x64xf32) <- (1x21x384xf32, 4xi64)
        reshape_46 = paddle._C_ops.reshape(add_75, full_int_array_26)

        # pd_op.transpose: (1x6x21x64xf32) <- (1x21x6x64xf32)
        transpose_25 = paddle._C_ops.transpose(reshape_46, [0, 2, 1, 3])
        del reshape_46

        # pd_op.reshape: (1x21x6x64xf32) <- (1x21x384xf32, 4xi64)
        reshape_47 = paddle._C_ops.reshape(add_76, full_int_array_26)

        # pd_op.transpose: (1x6x21x64xf32) <- (1x21x6x64xf32)
        transpose_26 = paddle._C_ops.transpose(reshape_47, [0, 2, 1, 3])
        del reshape_47

        # pd_op.matmul: (1x6x21x21xf32) <- (1x6x21x64xf32, 1x6x21x64xf32)
        matmul_72 = paddle._C_ops.matmul(transpose_24, transpose_25, False, True)

        # pd_op.scale: (1x6x21x21xf32) <- (1x6x21x21xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(matmul_72, full_5, float("0"), True)
        del matmul_72

        # pd_op.add: (1x6x21x21xf32) <- (1x6x21x21xf32, 1x1x1x21xf32)
        add_80 = paddle._C_ops.add(scale_7, unsqueeze_0)

        # pd_op.softmax: (1x6x21x21xf32) <- (1x6x21x21xf32)
        softmax_13 = paddle._C_ops.softmax(add_80, -1)
        del add_80

        # pd_op.dropout: (1x6x21x21xf32, 1x6x21x21xui8) <- (1x6x21x21xf32, None, 1xf32)
        dropout_38, dropout_39 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_13, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x6x21x64xf32) <- (1x6x21x21xf32, 1x6x21x64xf32)
        matmul_73 = paddle._C_ops.matmul(dropout_38, transpose_26, False, False)

        # pd_op.transpose: (1x21x6x64xf32) <- (1x6x21x64xf32)
        transpose_27 = paddle._C_ops.transpose(matmul_73, [0, 2, 1, 3])
        del matmul_73

        # builtin.combine: ([1x21x6x64xf32, 1x21x6x64xf32]) <- (1x21x6x64xf32, 1x21x6x64xf32)
        combine_13 = [transpose_27, reshape_44]

        # pd_op.concat: (1x21x12x64xf32) <- ([1x21x6x64xf32, 1x21x6x64xf32], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_13, full_6)
        del combine_13

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_48 = paddle._C_ops.reshape(concat_6, full_int_array_27)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_74 = paddle._C_ops.matmul(reshape_48, parameter_133, False, False)
        del parameter_133

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_81 = paddle._C_ops.add(matmul_74, parameter_132)
        del parameter_132

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_40, dropout_41 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_81, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_81

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_82 = paddle._C_ops.add(layer_norm_36, dropout_40)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_82, parameter_120, parameter_119, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_119, parameter_120

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_75 = paddle._C_ops.matmul(layer_norm_39, parameter_124, False, False)
        del parameter_124

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_83 = paddle._C_ops.add(matmul_75, parameter_123)
        del parameter_123

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_6 = paddle._C_ops.gelu(add_83, False)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_76 = paddle._C_ops.matmul(gelu_6, parameter_122, False, False)
        del parameter_122

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_84 = paddle._C_ops.add(matmul_76, parameter_121)
        del parameter_121

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_42, dropout_43 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_84, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_84

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_85 = paddle._C_ops.add(layer_norm_39, dropout_42)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_85, parameter_118, parameter_117, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_117, parameter_118

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_77 = paddle._C_ops.matmul(layer_norm_42, parameter_116, False, False)
        del parameter_116

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_86 = paddle._C_ops.add(matmul_77, parameter_115)
        del parameter_115

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_78 = paddle._C_ops.matmul(layer_norm_42, parameter_114, False, False)
        del parameter_114

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_87 = paddle._C_ops.add(matmul_78, parameter_113)
        del parameter_113

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_79 = paddle._C_ops.matmul(layer_norm_42, parameter_112, False, False)
        del parameter_112

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_88 = paddle._C_ops.add(matmul_79, parameter_111)
        del parameter_111

        # pd_op.assign: (768x1x9xf32) <- (768x1x9xf32)
        assign_376 = parameter_107
        del parameter_107

        # pd_op.unsqueeze: (768x1x1x9xf32) <- (768x1x9xf32, 1xi64)
        unsqueeze_36 = paddle._C_ops.unsqueeze(assign_376, full_int_array_1)

        # pd_op.unsqueeze: (1x1x21x768xf32) <- (1x21x768xf32, 1xi64)
        unsqueeze_37 = paddle._C_ops.unsqueeze(layer_norm_42, full_int_array_2)

        # pd_op.depthwise_conv2d: (1x1x21x768xf32) <- (1x1x21x768xf32, 768x1x1x9xf32)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(
            unsqueeze_37, unsqueeze_36, [1, 1], [0, 4], "EXPLICIT", 768, [1, 1], "NHWC"
        )

        # pd_op.squeeze: (1x21x768xf32) <- (1x1x21x768xf32, 1xi64)
        squeeze_21 = paddle._C_ops.squeeze(depthwise_conv2d_7, full_int_array_2)

        # pd_op.assign: (384x768x1xf32) <- (384x768x1xf32)
        assign_377 = parameter_106
        del parameter_106

        # pd_op.unsqueeze: (384x768x1x1xf32) <- (384x768x1xf32, 1xi64)
        unsqueeze_38 = paddle._C_ops.unsqueeze(assign_377, full_int_array_1)

        # pd_op.unsqueeze: (1x1x21x768xf32) <- (1x21x768xf32, 1xi64)
        unsqueeze_39 = paddle._C_ops.unsqueeze(squeeze_21, full_int_array_2)

        # pd_op.conv2d: (1x1x21x384xf32) <- (1x1x21x768xf32, 384x768x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            unsqueeze_39, unsqueeze_38, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NHWC"
        )

        # pd_op.squeeze: (1x21x384xf32) <- (1x1x21x384xf32, 1xi64)
        squeeze_22 = paddle._C_ops.squeeze(conv2d_7, full_int_array_2)

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_89 = paddle._C_ops.add(squeeze_22, parameter_108)
        del parameter_108

        # pd_op.multiply: (1x21x384xf32) <- (1x21x384xf32, 1x21x384xf32)
        multiply_7 = paddle._C_ops.multiply(add_89, add_86)

        # pd_op.matmul: (1x21x54xf32) <- (1x21x384xf32, 384x54xf32)
        matmul_80 = paddle._C_ops.matmul(multiply_7, parameter_105, False, False)
        del parameter_105

        # pd_op.add: (1x21x54xf32) <- (1x21x54xf32, 54xf32)
        add_90 = paddle._C_ops.add(matmul_80, parameter_104)
        del parameter_104

        # pd_op.reshape: (126x9x1xf32) <- (1x21x54xf32, 3xi64)
        reshape_49 = paddle._C_ops.reshape(add_90, full_int_array_3)

        # pd_op.softmax: (126x9x1xf32) <- (126x9x1xf32)
        softmax_14 = paddle._C_ops.softmax(reshape_49, 1)
        del reshape_49

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_81 = paddle._C_ops.matmul(layer_norm_42, parameter_103, False, False)
        del parameter_103

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_91 = paddle._C_ops.add(matmul_81, parameter_102)
        del parameter_102

        # pd_op.unsqueeze: (1x21x1x1x384xf32) <- (1x21x384xf32, 2xi64)
        unsqueeze_40 = paddle._C_ops.unsqueeze(add_91, full_int_array_4)

        # pd_op.pad3d: (1x29x1x1x384xf32) <- (1x21x1x1x384xf32, 6xi64)
        pad3d_7 = paddle._C_ops.pad3d(
            unsqueeze_40, full_int_array_5, "constant", float("0"), "NDHWC"
        )

        # pd_op.squeeze: (1x29x384xf32) <- (1x29x1x1x384xf32, 2xi64)
        squeeze_23 = paddle._C_ops.squeeze(pad3d_7, full_int_array_4)

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_63 = paddle._C_ops.slice(
            squeeze_23, [1], full_int_array_6, full_int_array_7, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_64 = paddle._C_ops.slice(
            squeeze_23, [1], full_int_array_8, full_int_array_9, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_65 = paddle._C_ops.slice(
            squeeze_23, [1], full_int_array_10, full_int_array_11, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_66 = paddle._C_ops.slice(
            squeeze_23, [1], full_int_array_12, full_int_array_13, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_67 = paddle._C_ops.slice(
            squeeze_23, [1], full_int_array_14, full_int_array_15, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_68 = paddle._C_ops.slice(
            squeeze_23, [1], full_int_array_16, full_int_array_17, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_69 = paddle._C_ops.slice(
            squeeze_23, [1], full_int_array_18, full_int_array_19, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_70 = paddle._C_ops.slice(
            squeeze_23, [1], full_int_array_20, full_int_array_21, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_71 = paddle._C_ops.slice(
            squeeze_23, [1], full_int_array_22, full_int_array_23, [1], []
        )

        # builtin.combine: ([1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32]) <- (1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32)
        combine_14 = [
            slice_63,
            slice_64,
            slice_65,
            slice_66,
            slice_67,
            slice_68,
            slice_69,
            slice_70,
            slice_71,
        ]

        # pd_op.stack: (1x21x384x9xf32) <- ([1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32])
        stack_7 = paddle._C_ops.stack(combine_14, -1)
        del combine_14

        # pd_op.reshape: (126x64x9xf32) <- (1x21x384x9xf32, 3xi64)
        reshape_50 = paddle._C_ops.reshape(stack_7, full_int_array_24)

        # pd_op.matmul: (126x64x1xf32) <- (126x64x9xf32, 126x9x1xf32)
        matmul_82 = paddle._C_ops.matmul(reshape_50, softmax_14, False, False)

        # pd_op.reshape: (1x21x6x64xf32) <- (126x64x1xf32, 4xi64)
        reshape_51 = paddle._C_ops.reshape(matmul_82, full_int_array_25)

        # pd_op.reshape: (1x21x6x64xf32) <- (1x21x384xf32, 4xi64)
        reshape_52 = paddle._C_ops.reshape(add_86, full_int_array_26)

        # pd_op.transpose: (1x6x21x64xf32) <- (1x21x6x64xf32)
        transpose_28 = paddle._C_ops.transpose(reshape_52, [0, 2, 1, 3])
        del reshape_52

        # pd_op.reshape: (1x21x6x64xf32) <- (1x21x384xf32, 4xi64)
        reshape_53 = paddle._C_ops.reshape(add_87, full_int_array_26)

        # pd_op.transpose: (1x6x21x64xf32) <- (1x21x6x64xf32)
        transpose_29 = paddle._C_ops.transpose(reshape_53, [0, 2, 1, 3])
        del reshape_53

        # pd_op.reshape: (1x21x6x64xf32) <- (1x21x384xf32, 4xi64)
        reshape_54 = paddle._C_ops.reshape(add_88, full_int_array_26)

        # pd_op.transpose: (1x6x21x64xf32) <- (1x21x6x64xf32)
        transpose_30 = paddle._C_ops.transpose(reshape_54, [0, 2, 1, 3])
        del reshape_54

        # pd_op.matmul: (1x6x21x21xf32) <- (1x6x21x64xf32, 1x6x21x64xf32)
        matmul_83 = paddle._C_ops.matmul(transpose_28, transpose_29, False, True)

        # pd_op.scale: (1x6x21x21xf32) <- (1x6x21x21xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(matmul_83, full_5, float("0"), True)
        del matmul_83

        # pd_op.add: (1x6x21x21xf32) <- (1x6x21x21xf32, 1x1x1x21xf32)
        add_92 = paddle._C_ops.add(scale_8, unsqueeze_0)

        # pd_op.softmax: (1x6x21x21xf32) <- (1x6x21x21xf32)
        softmax_15 = paddle._C_ops.softmax(add_92, -1)
        del add_92

        # pd_op.dropout: (1x6x21x21xf32, 1x6x21x21xui8) <- (1x6x21x21xf32, None, 1xf32)
        dropout_44, dropout_45 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_15, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x6x21x64xf32) <- (1x6x21x21xf32, 1x6x21x64xf32)
        matmul_84 = paddle._C_ops.matmul(dropout_44, transpose_30, False, False)

        # pd_op.transpose: (1x21x6x64xf32) <- (1x6x21x64xf32)
        transpose_31 = paddle._C_ops.transpose(matmul_84, [0, 2, 1, 3])
        del matmul_84

        # builtin.combine: ([1x21x6x64xf32, 1x21x6x64xf32]) <- (1x21x6x64xf32, 1x21x6x64xf32)
        combine_15 = [transpose_31, reshape_51]

        # pd_op.concat: (1x21x12x64xf32) <- ([1x21x6x64xf32, 1x21x6x64xf32], 1xi32)
        concat_7 = paddle._C_ops.concat(combine_15, full_6)
        del combine_15

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_55 = paddle._C_ops.reshape(concat_7, full_int_array_27)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_85 = paddle._C_ops.matmul(reshape_55, parameter_110, False, False)
        del parameter_110

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_93 = paddle._C_ops.add(matmul_85, parameter_109)
        del parameter_109

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_46, dropout_47 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_93, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_93

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_94 = paddle._C_ops.add(layer_norm_42, dropout_46)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_94, parameter_97, parameter_96, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_96, parameter_97

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_86 = paddle._C_ops.matmul(layer_norm_45, parameter_101, False, False)
        del parameter_101

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_95 = paddle._C_ops.add(matmul_86, parameter_100)
        del parameter_100

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_7 = paddle._C_ops.gelu(add_95, False)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_87 = paddle._C_ops.matmul(gelu_7, parameter_99, False, False)
        del parameter_99

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_96 = paddle._C_ops.add(matmul_87, parameter_98)
        del parameter_98

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_48, dropout_49 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_96, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_96

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_97 = paddle._C_ops.add(layer_norm_45, dropout_48)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_97, parameter_95, parameter_94, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_94, parameter_95

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_88 = paddle._C_ops.matmul(layer_norm_48, parameter_93, False, False)
        del parameter_93

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_98 = paddle._C_ops.add(matmul_88, parameter_92)
        del parameter_92

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_89 = paddle._C_ops.matmul(layer_norm_48, parameter_91, False, False)
        del parameter_91

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_99 = paddle._C_ops.add(matmul_89, parameter_90)
        del parameter_90

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_90 = paddle._C_ops.matmul(layer_norm_48, parameter_89, False, False)
        del parameter_89

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_100 = paddle._C_ops.add(matmul_90, parameter_88)
        del parameter_88

        # pd_op.assign: (768x1x9xf32) <- (768x1x9xf32)
        assign_378 = parameter_84
        del parameter_84

        # pd_op.unsqueeze: (768x1x1x9xf32) <- (768x1x9xf32, 1xi64)
        unsqueeze_41 = paddle._C_ops.unsqueeze(assign_378, full_int_array_1)

        # pd_op.unsqueeze: (1x1x21x768xf32) <- (1x21x768xf32, 1xi64)
        unsqueeze_42 = paddle._C_ops.unsqueeze(layer_norm_48, full_int_array_2)

        # pd_op.depthwise_conv2d: (1x1x21x768xf32) <- (1x1x21x768xf32, 768x1x1x9xf32)
        depthwise_conv2d_8 = paddle._C_ops.depthwise_conv2d(
            unsqueeze_42, unsqueeze_41, [1, 1], [0, 4], "EXPLICIT", 768, [1, 1], "NHWC"
        )

        # pd_op.squeeze: (1x21x768xf32) <- (1x1x21x768xf32, 1xi64)
        squeeze_24 = paddle._C_ops.squeeze(depthwise_conv2d_8, full_int_array_2)

        # pd_op.assign: (384x768x1xf32) <- (384x768x1xf32)
        assign_379 = parameter_83
        del parameter_83

        # pd_op.unsqueeze: (384x768x1x1xf32) <- (384x768x1xf32, 1xi64)
        unsqueeze_43 = paddle._C_ops.unsqueeze(assign_379, full_int_array_1)

        # pd_op.unsqueeze: (1x1x21x768xf32) <- (1x21x768xf32, 1xi64)
        unsqueeze_44 = paddle._C_ops.unsqueeze(squeeze_24, full_int_array_2)

        # pd_op.conv2d: (1x1x21x384xf32) <- (1x1x21x768xf32, 384x768x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            unsqueeze_44, unsqueeze_43, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NHWC"
        )

        # pd_op.squeeze: (1x21x384xf32) <- (1x1x21x384xf32, 1xi64)
        squeeze_25 = paddle._C_ops.squeeze(conv2d_8, full_int_array_2)

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_101 = paddle._C_ops.add(squeeze_25, parameter_85)
        del parameter_85

        # pd_op.multiply: (1x21x384xf32) <- (1x21x384xf32, 1x21x384xf32)
        multiply_8 = paddle._C_ops.multiply(add_101, add_98)

        # pd_op.matmul: (1x21x54xf32) <- (1x21x384xf32, 384x54xf32)
        matmul_91 = paddle._C_ops.matmul(multiply_8, parameter_82, False, False)
        del parameter_82

        # pd_op.add: (1x21x54xf32) <- (1x21x54xf32, 54xf32)
        add_102 = paddle._C_ops.add(matmul_91, parameter_81)
        del parameter_81

        # pd_op.reshape: (126x9x1xf32) <- (1x21x54xf32, 3xi64)
        reshape_56 = paddle._C_ops.reshape(add_102, full_int_array_3)

        # pd_op.softmax: (126x9x1xf32) <- (126x9x1xf32)
        softmax_16 = paddle._C_ops.softmax(reshape_56, 1)
        del reshape_56

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_92 = paddle._C_ops.matmul(layer_norm_48, parameter_80, False, False)
        del parameter_80

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_103 = paddle._C_ops.add(matmul_92, parameter_79)
        del parameter_79

        # pd_op.unsqueeze: (1x21x1x1x384xf32) <- (1x21x384xf32, 2xi64)
        unsqueeze_45 = paddle._C_ops.unsqueeze(add_103, full_int_array_4)

        # pd_op.pad3d: (1x29x1x1x384xf32) <- (1x21x1x1x384xf32, 6xi64)
        pad3d_8 = paddle._C_ops.pad3d(
            unsqueeze_45, full_int_array_5, "constant", float("0"), "NDHWC"
        )

        # pd_op.squeeze: (1x29x384xf32) <- (1x29x1x1x384xf32, 2xi64)
        squeeze_26 = paddle._C_ops.squeeze(pad3d_8, full_int_array_4)

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_72 = paddle._C_ops.slice(
            squeeze_26, [1], full_int_array_6, full_int_array_7, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_73 = paddle._C_ops.slice(
            squeeze_26, [1], full_int_array_8, full_int_array_9, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_74 = paddle._C_ops.slice(
            squeeze_26, [1], full_int_array_10, full_int_array_11, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_75 = paddle._C_ops.slice(
            squeeze_26, [1], full_int_array_12, full_int_array_13, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_76 = paddle._C_ops.slice(
            squeeze_26, [1], full_int_array_14, full_int_array_15, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_77 = paddle._C_ops.slice(
            squeeze_26, [1], full_int_array_16, full_int_array_17, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_78 = paddle._C_ops.slice(
            squeeze_26, [1], full_int_array_18, full_int_array_19, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_79 = paddle._C_ops.slice(
            squeeze_26, [1], full_int_array_20, full_int_array_21, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_80 = paddle._C_ops.slice(
            squeeze_26, [1], full_int_array_22, full_int_array_23, [1], []
        )

        # builtin.combine: ([1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32]) <- (1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32)
        combine_16 = [
            slice_72,
            slice_73,
            slice_74,
            slice_75,
            slice_76,
            slice_77,
            slice_78,
            slice_79,
            slice_80,
        ]

        # pd_op.stack: (1x21x384x9xf32) <- ([1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32])
        stack_8 = paddle._C_ops.stack(combine_16, -1)
        del combine_16

        # pd_op.reshape: (126x64x9xf32) <- (1x21x384x9xf32, 3xi64)
        reshape_57 = paddle._C_ops.reshape(stack_8, full_int_array_24)

        # pd_op.matmul: (126x64x1xf32) <- (126x64x9xf32, 126x9x1xf32)
        matmul_93 = paddle._C_ops.matmul(reshape_57, softmax_16, False, False)

        # pd_op.reshape: (1x21x6x64xf32) <- (126x64x1xf32, 4xi64)
        reshape_58 = paddle._C_ops.reshape(matmul_93, full_int_array_25)

        # pd_op.reshape: (1x21x6x64xf32) <- (1x21x384xf32, 4xi64)
        reshape_59 = paddle._C_ops.reshape(add_98, full_int_array_26)

        # pd_op.transpose: (1x6x21x64xf32) <- (1x21x6x64xf32)
        transpose_32 = paddle._C_ops.transpose(reshape_59, [0, 2, 1, 3])
        del reshape_59

        # pd_op.reshape: (1x21x6x64xf32) <- (1x21x384xf32, 4xi64)
        reshape_60 = paddle._C_ops.reshape(add_99, full_int_array_26)

        # pd_op.transpose: (1x6x21x64xf32) <- (1x21x6x64xf32)
        transpose_33 = paddle._C_ops.transpose(reshape_60, [0, 2, 1, 3])
        del reshape_60

        # pd_op.reshape: (1x21x6x64xf32) <- (1x21x384xf32, 4xi64)
        reshape_61 = paddle._C_ops.reshape(add_100, full_int_array_26)

        # pd_op.transpose: (1x6x21x64xf32) <- (1x21x6x64xf32)
        transpose_34 = paddle._C_ops.transpose(reshape_61, [0, 2, 1, 3])
        del reshape_61

        # pd_op.matmul: (1x6x21x21xf32) <- (1x6x21x64xf32, 1x6x21x64xf32)
        matmul_94 = paddle._C_ops.matmul(transpose_32, transpose_33, False, True)

        # pd_op.scale: (1x6x21x21xf32) <- (1x6x21x21xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(matmul_94, full_5, float("0"), True)
        del matmul_94

        # pd_op.add: (1x6x21x21xf32) <- (1x6x21x21xf32, 1x1x1x21xf32)
        add_104 = paddle._C_ops.add(scale_9, unsqueeze_0)

        # pd_op.softmax: (1x6x21x21xf32) <- (1x6x21x21xf32)
        softmax_17 = paddle._C_ops.softmax(add_104, -1)
        del add_104

        # pd_op.dropout: (1x6x21x21xf32, 1x6x21x21xui8) <- (1x6x21x21xf32, None, 1xf32)
        dropout_50, dropout_51 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_17, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x6x21x64xf32) <- (1x6x21x21xf32, 1x6x21x64xf32)
        matmul_95 = paddle._C_ops.matmul(dropout_50, transpose_34, False, False)

        # pd_op.transpose: (1x21x6x64xf32) <- (1x6x21x64xf32)
        transpose_35 = paddle._C_ops.transpose(matmul_95, [0, 2, 1, 3])
        del matmul_95

        # builtin.combine: ([1x21x6x64xf32, 1x21x6x64xf32]) <- (1x21x6x64xf32, 1x21x6x64xf32)
        combine_17 = [transpose_35, reshape_58]

        # pd_op.concat: (1x21x12x64xf32) <- ([1x21x6x64xf32, 1x21x6x64xf32], 1xi32)
        concat_8 = paddle._C_ops.concat(combine_17, full_6)
        del combine_17

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_62 = paddle._C_ops.reshape(concat_8, full_int_array_27)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_96 = paddle._C_ops.matmul(reshape_62, parameter_87, False, False)
        del parameter_87

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_105 = paddle._C_ops.add(matmul_96, parameter_86)
        del parameter_86

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_52, dropout_53 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_105, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_105

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_106 = paddle._C_ops.add(layer_norm_48, dropout_52)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_106, parameter_74, parameter_73, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_73, parameter_74

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_97 = paddle._C_ops.matmul(layer_norm_51, parameter_78, False, False)
        del parameter_78

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_107 = paddle._C_ops.add(matmul_97, parameter_77)
        del parameter_77

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_8 = paddle._C_ops.gelu(add_107, False)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_98 = paddle._C_ops.matmul(gelu_8, parameter_76, False, False)
        del parameter_76

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_108 = paddle._C_ops.add(matmul_98, parameter_75)
        del parameter_75

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_54, dropout_55 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_108, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_108

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_109 = paddle._C_ops.add(layer_norm_51, dropout_54)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_109, parameter_72, parameter_71, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_71, parameter_72

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_99 = paddle._C_ops.matmul(layer_norm_54, parameter_70, False, False)
        del parameter_70

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_110 = paddle._C_ops.add(matmul_99, parameter_69)
        del parameter_69

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_100 = paddle._C_ops.matmul(layer_norm_54, parameter_68, False, False)
        del parameter_68

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_111 = paddle._C_ops.add(matmul_100, parameter_67)
        del parameter_67

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_101 = paddle._C_ops.matmul(layer_norm_54, parameter_66, False, False)
        del parameter_66

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_112 = paddle._C_ops.add(matmul_101, parameter_65)
        del parameter_65

        # pd_op.assign: (768x1x9xf32) <- (768x1x9xf32)
        assign_380 = parameter_61
        del parameter_61

        # pd_op.unsqueeze: (768x1x1x9xf32) <- (768x1x9xf32, 1xi64)
        unsqueeze_46 = paddle._C_ops.unsqueeze(assign_380, full_int_array_1)

        # pd_op.unsqueeze: (1x1x21x768xf32) <- (1x21x768xf32, 1xi64)
        unsqueeze_47 = paddle._C_ops.unsqueeze(layer_norm_54, full_int_array_2)

        # pd_op.depthwise_conv2d: (1x1x21x768xf32) <- (1x1x21x768xf32, 768x1x1x9xf32)
        depthwise_conv2d_9 = paddle._C_ops.depthwise_conv2d(
            unsqueeze_47, unsqueeze_46, [1, 1], [0, 4], "EXPLICIT", 768, [1, 1], "NHWC"
        )

        # pd_op.squeeze: (1x21x768xf32) <- (1x1x21x768xf32, 1xi64)
        squeeze_27 = paddle._C_ops.squeeze(depthwise_conv2d_9, full_int_array_2)

        # pd_op.assign: (384x768x1xf32) <- (384x768x1xf32)
        assign_381 = parameter_60
        del parameter_60

        # pd_op.unsqueeze: (384x768x1x1xf32) <- (384x768x1xf32, 1xi64)
        unsqueeze_48 = paddle._C_ops.unsqueeze(assign_381, full_int_array_1)

        # pd_op.unsqueeze: (1x1x21x768xf32) <- (1x21x768xf32, 1xi64)
        unsqueeze_49 = paddle._C_ops.unsqueeze(squeeze_27, full_int_array_2)

        # pd_op.conv2d: (1x1x21x384xf32) <- (1x1x21x768xf32, 384x768x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            unsqueeze_49, unsqueeze_48, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NHWC"
        )

        # pd_op.squeeze: (1x21x384xf32) <- (1x1x21x384xf32, 1xi64)
        squeeze_28 = paddle._C_ops.squeeze(conv2d_9, full_int_array_2)

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_113 = paddle._C_ops.add(squeeze_28, parameter_62)
        del parameter_62

        # pd_op.multiply: (1x21x384xf32) <- (1x21x384xf32, 1x21x384xf32)
        multiply_9 = paddle._C_ops.multiply(add_113, add_110)

        # pd_op.matmul: (1x21x54xf32) <- (1x21x384xf32, 384x54xf32)
        matmul_102 = paddle._C_ops.matmul(multiply_9, parameter_59, False, False)
        del parameter_59

        # pd_op.add: (1x21x54xf32) <- (1x21x54xf32, 54xf32)
        add_114 = paddle._C_ops.add(matmul_102, parameter_58)
        del parameter_58

        # pd_op.reshape: (126x9x1xf32) <- (1x21x54xf32, 3xi64)
        reshape_63 = paddle._C_ops.reshape(add_114, full_int_array_3)

        # pd_op.softmax: (126x9x1xf32) <- (126x9x1xf32)
        softmax_18 = paddle._C_ops.softmax(reshape_63, 1)
        del reshape_63

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_103 = paddle._C_ops.matmul(layer_norm_54, parameter_57, False, False)
        del parameter_57

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_115 = paddle._C_ops.add(matmul_103, parameter_56)
        del parameter_56

        # pd_op.unsqueeze: (1x21x1x1x384xf32) <- (1x21x384xf32, 2xi64)
        unsqueeze_50 = paddle._C_ops.unsqueeze(add_115, full_int_array_4)

        # pd_op.pad3d: (1x29x1x1x384xf32) <- (1x21x1x1x384xf32, 6xi64)
        pad3d_9 = paddle._C_ops.pad3d(
            unsqueeze_50, full_int_array_5, "constant", float("0"), "NDHWC"
        )

        # pd_op.squeeze: (1x29x384xf32) <- (1x29x1x1x384xf32, 2xi64)
        squeeze_29 = paddle._C_ops.squeeze(pad3d_9, full_int_array_4)

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_81 = paddle._C_ops.slice(
            squeeze_29, [1], full_int_array_6, full_int_array_7, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_82 = paddle._C_ops.slice(
            squeeze_29, [1], full_int_array_8, full_int_array_9, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_83 = paddle._C_ops.slice(
            squeeze_29, [1], full_int_array_10, full_int_array_11, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_84 = paddle._C_ops.slice(
            squeeze_29, [1], full_int_array_12, full_int_array_13, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_85 = paddle._C_ops.slice(
            squeeze_29, [1], full_int_array_14, full_int_array_15, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_86 = paddle._C_ops.slice(
            squeeze_29, [1], full_int_array_16, full_int_array_17, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_87 = paddle._C_ops.slice(
            squeeze_29, [1], full_int_array_18, full_int_array_19, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_88 = paddle._C_ops.slice(
            squeeze_29, [1], full_int_array_20, full_int_array_21, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_89 = paddle._C_ops.slice(
            squeeze_29, [1], full_int_array_22, full_int_array_23, [1], []
        )

        # builtin.combine: ([1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32]) <- (1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32)
        combine_18 = [
            slice_81,
            slice_82,
            slice_83,
            slice_84,
            slice_85,
            slice_86,
            slice_87,
            slice_88,
            slice_89,
        ]

        # pd_op.stack: (1x21x384x9xf32) <- ([1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32])
        stack_9 = paddle._C_ops.stack(combine_18, -1)
        del combine_18

        # pd_op.reshape: (126x64x9xf32) <- (1x21x384x9xf32, 3xi64)
        reshape_64 = paddle._C_ops.reshape(stack_9, full_int_array_24)

        # pd_op.matmul: (126x64x1xf32) <- (126x64x9xf32, 126x9x1xf32)
        matmul_104 = paddle._C_ops.matmul(reshape_64, softmax_18, False, False)

        # pd_op.reshape: (1x21x6x64xf32) <- (126x64x1xf32, 4xi64)
        reshape_65 = paddle._C_ops.reshape(matmul_104, full_int_array_25)

        # pd_op.reshape: (1x21x6x64xf32) <- (1x21x384xf32, 4xi64)
        reshape_66 = paddle._C_ops.reshape(add_110, full_int_array_26)

        # pd_op.transpose: (1x6x21x64xf32) <- (1x21x6x64xf32)
        transpose_36 = paddle._C_ops.transpose(reshape_66, [0, 2, 1, 3])
        del reshape_66

        # pd_op.reshape: (1x21x6x64xf32) <- (1x21x384xf32, 4xi64)
        reshape_67 = paddle._C_ops.reshape(add_111, full_int_array_26)

        # pd_op.transpose: (1x6x21x64xf32) <- (1x21x6x64xf32)
        transpose_37 = paddle._C_ops.transpose(reshape_67, [0, 2, 1, 3])
        del reshape_67

        # pd_op.reshape: (1x21x6x64xf32) <- (1x21x384xf32, 4xi64)
        reshape_68 = paddle._C_ops.reshape(add_112, full_int_array_26)

        # pd_op.transpose: (1x6x21x64xf32) <- (1x21x6x64xf32)
        transpose_38 = paddle._C_ops.transpose(reshape_68, [0, 2, 1, 3])
        del reshape_68

        # pd_op.matmul: (1x6x21x21xf32) <- (1x6x21x64xf32, 1x6x21x64xf32)
        matmul_105 = paddle._C_ops.matmul(transpose_36, transpose_37, False, True)

        # pd_op.scale: (1x6x21x21xf32) <- (1x6x21x21xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(matmul_105, full_5, float("0"), True)
        del matmul_105

        # pd_op.add: (1x6x21x21xf32) <- (1x6x21x21xf32, 1x1x1x21xf32)
        add_116 = paddle._C_ops.add(scale_10, unsqueeze_0)

        # pd_op.softmax: (1x6x21x21xf32) <- (1x6x21x21xf32)
        softmax_19 = paddle._C_ops.softmax(add_116, -1)
        del add_116

        # pd_op.dropout: (1x6x21x21xf32, 1x6x21x21xui8) <- (1x6x21x21xf32, None, 1xf32)
        dropout_56, dropout_57 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_19, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x6x21x64xf32) <- (1x6x21x21xf32, 1x6x21x64xf32)
        matmul_106 = paddle._C_ops.matmul(dropout_56, transpose_38, False, False)

        # pd_op.transpose: (1x21x6x64xf32) <- (1x6x21x64xf32)
        transpose_39 = paddle._C_ops.transpose(matmul_106, [0, 2, 1, 3])
        del matmul_106

        # builtin.combine: ([1x21x6x64xf32, 1x21x6x64xf32]) <- (1x21x6x64xf32, 1x21x6x64xf32)
        combine_19 = [transpose_39, reshape_65]

        # pd_op.concat: (1x21x12x64xf32) <- ([1x21x6x64xf32, 1x21x6x64xf32], 1xi32)
        concat_9 = paddle._C_ops.concat(combine_19, full_6)
        del combine_19

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_69 = paddle._C_ops.reshape(concat_9, full_int_array_27)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_107 = paddle._C_ops.matmul(reshape_69, parameter_64, False, False)
        del parameter_64

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_117 = paddle._C_ops.add(matmul_107, parameter_63)
        del parameter_63

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_58, dropout_59 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_117, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_117

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_118 = paddle._C_ops.add(layer_norm_54, dropout_58)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_118, parameter_51, parameter_50, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_50, parameter_51

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_108 = paddle._C_ops.matmul(layer_norm_57, parameter_55, False, False)
        del parameter_55

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_119 = paddle._C_ops.add(matmul_108, parameter_54)
        del parameter_54

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_9 = paddle._C_ops.gelu(add_119, False)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_109 = paddle._C_ops.matmul(gelu_9, parameter_53, False, False)
        del parameter_53

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_120 = paddle._C_ops.add(matmul_109, parameter_52)
        del parameter_52

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_60, dropout_61 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_120, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_120

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_121 = paddle._C_ops.add(layer_norm_57, dropout_60)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_121, parameter_49, parameter_48, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_48, parameter_49

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_110 = paddle._C_ops.matmul(layer_norm_60, parameter_47, False, False)
        del parameter_47

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_122 = paddle._C_ops.add(matmul_110, parameter_46)
        del parameter_46

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_111 = paddle._C_ops.matmul(layer_norm_60, parameter_45, False, False)
        del parameter_45

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_123 = paddle._C_ops.add(matmul_111, parameter_44)
        del parameter_44

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_112 = paddle._C_ops.matmul(layer_norm_60, parameter_43, False, False)
        del parameter_43

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_124 = paddle._C_ops.add(matmul_112, parameter_42)
        del parameter_42

        # pd_op.assign: (768x1x9xf32) <- (768x1x9xf32)
        assign_382 = parameter_38
        del parameter_38

        # pd_op.unsqueeze: (768x1x1x9xf32) <- (768x1x9xf32, 1xi64)
        unsqueeze_51 = paddle._C_ops.unsqueeze(assign_382, full_int_array_1)

        # pd_op.unsqueeze: (1x1x21x768xf32) <- (1x21x768xf32, 1xi64)
        unsqueeze_52 = paddle._C_ops.unsqueeze(layer_norm_60, full_int_array_2)

        # pd_op.depthwise_conv2d: (1x1x21x768xf32) <- (1x1x21x768xf32, 768x1x1x9xf32)
        depthwise_conv2d_10 = paddle._C_ops.depthwise_conv2d(
            unsqueeze_52, unsqueeze_51, [1, 1], [0, 4], "EXPLICIT", 768, [1, 1], "NHWC"
        )

        # pd_op.squeeze: (1x21x768xf32) <- (1x1x21x768xf32, 1xi64)
        squeeze_30 = paddle._C_ops.squeeze(depthwise_conv2d_10, full_int_array_2)

        # pd_op.assign: (384x768x1xf32) <- (384x768x1xf32)
        assign_383 = parameter_37
        del parameter_37

        # pd_op.unsqueeze: (384x768x1x1xf32) <- (384x768x1xf32, 1xi64)
        unsqueeze_53 = paddle._C_ops.unsqueeze(assign_383, full_int_array_1)

        # pd_op.unsqueeze: (1x1x21x768xf32) <- (1x21x768xf32, 1xi64)
        unsqueeze_54 = paddle._C_ops.unsqueeze(squeeze_30, full_int_array_2)

        # pd_op.conv2d: (1x1x21x384xf32) <- (1x1x21x768xf32, 384x768x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            unsqueeze_54, unsqueeze_53, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NHWC"
        )

        # pd_op.squeeze: (1x21x384xf32) <- (1x1x21x384xf32, 1xi64)
        squeeze_31 = paddle._C_ops.squeeze(conv2d_10, full_int_array_2)

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_125 = paddle._C_ops.add(squeeze_31, parameter_39)
        del parameter_39

        # pd_op.multiply: (1x21x384xf32) <- (1x21x384xf32, 1x21x384xf32)
        multiply_10 = paddle._C_ops.multiply(add_125, add_122)

        # pd_op.matmul: (1x21x54xf32) <- (1x21x384xf32, 384x54xf32)
        matmul_113 = paddle._C_ops.matmul(multiply_10, parameter_36, False, False)
        del parameter_36

        # pd_op.add: (1x21x54xf32) <- (1x21x54xf32, 54xf32)
        add_126 = paddle._C_ops.add(matmul_113, parameter_35)
        del parameter_35

        # pd_op.reshape: (126x9x1xf32) <- (1x21x54xf32, 3xi64)
        reshape_70 = paddle._C_ops.reshape(add_126, full_int_array_3)

        # pd_op.softmax: (126x9x1xf32) <- (126x9x1xf32)
        softmax_20 = paddle._C_ops.softmax(reshape_70, 1)
        del reshape_70

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_114 = paddle._C_ops.matmul(layer_norm_60, parameter_34, False, False)
        del parameter_34

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_127 = paddle._C_ops.add(matmul_114, parameter_33)
        del parameter_33

        # pd_op.unsqueeze: (1x21x1x1x384xf32) <- (1x21x384xf32, 2xi64)
        unsqueeze_55 = paddle._C_ops.unsqueeze(add_127, full_int_array_4)

        # pd_op.pad3d: (1x29x1x1x384xf32) <- (1x21x1x1x384xf32, 6xi64)
        pad3d_10 = paddle._C_ops.pad3d(
            unsqueeze_55, full_int_array_5, "constant", float("0"), "NDHWC"
        )

        # pd_op.squeeze: (1x29x384xf32) <- (1x29x1x1x384xf32, 2xi64)
        squeeze_32 = paddle._C_ops.squeeze(pad3d_10, full_int_array_4)

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_90 = paddle._C_ops.slice(
            squeeze_32, [1], full_int_array_6, full_int_array_7, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_91 = paddle._C_ops.slice(
            squeeze_32, [1], full_int_array_8, full_int_array_9, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_92 = paddle._C_ops.slice(
            squeeze_32, [1], full_int_array_10, full_int_array_11, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_93 = paddle._C_ops.slice(
            squeeze_32, [1], full_int_array_12, full_int_array_13, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_94 = paddle._C_ops.slice(
            squeeze_32, [1], full_int_array_14, full_int_array_15, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_95 = paddle._C_ops.slice(
            squeeze_32, [1], full_int_array_16, full_int_array_17, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_96 = paddle._C_ops.slice(
            squeeze_32, [1], full_int_array_18, full_int_array_19, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_97 = paddle._C_ops.slice(
            squeeze_32, [1], full_int_array_20, full_int_array_21, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_98 = paddle._C_ops.slice(
            squeeze_32, [1], full_int_array_22, full_int_array_23, [1], []
        )

        # builtin.combine: ([1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32]) <- (1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32)
        combine_20 = [
            slice_90,
            slice_91,
            slice_92,
            slice_93,
            slice_94,
            slice_95,
            slice_96,
            slice_97,
            slice_98,
        ]

        # pd_op.stack: (1x21x384x9xf32) <- ([1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32])
        stack_10 = paddle._C_ops.stack(combine_20, -1)
        del combine_20

        # pd_op.reshape: (126x64x9xf32) <- (1x21x384x9xf32, 3xi64)
        reshape_71 = paddle._C_ops.reshape(stack_10, full_int_array_24)

        # pd_op.matmul: (126x64x1xf32) <- (126x64x9xf32, 126x9x1xf32)
        matmul_115 = paddle._C_ops.matmul(reshape_71, softmax_20, False, False)

        # pd_op.reshape: (1x21x6x64xf32) <- (126x64x1xf32, 4xi64)
        reshape_72 = paddle._C_ops.reshape(matmul_115, full_int_array_25)

        # pd_op.reshape: (1x21x6x64xf32) <- (1x21x384xf32, 4xi64)
        reshape_73 = paddle._C_ops.reshape(add_122, full_int_array_26)

        # pd_op.transpose: (1x6x21x64xf32) <- (1x21x6x64xf32)
        transpose_40 = paddle._C_ops.transpose(reshape_73, [0, 2, 1, 3])
        del reshape_73

        # pd_op.reshape: (1x21x6x64xf32) <- (1x21x384xf32, 4xi64)
        reshape_74 = paddle._C_ops.reshape(add_123, full_int_array_26)

        # pd_op.transpose: (1x6x21x64xf32) <- (1x21x6x64xf32)
        transpose_41 = paddle._C_ops.transpose(reshape_74, [0, 2, 1, 3])
        del reshape_74

        # pd_op.reshape: (1x21x6x64xf32) <- (1x21x384xf32, 4xi64)
        reshape_75 = paddle._C_ops.reshape(add_124, full_int_array_26)

        # pd_op.transpose: (1x6x21x64xf32) <- (1x21x6x64xf32)
        transpose_42 = paddle._C_ops.transpose(reshape_75, [0, 2, 1, 3])
        del reshape_75

        # pd_op.matmul: (1x6x21x21xf32) <- (1x6x21x64xf32, 1x6x21x64xf32)
        matmul_116 = paddle._C_ops.matmul(transpose_40, transpose_41, False, True)

        # pd_op.scale: (1x6x21x21xf32) <- (1x6x21x21xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(matmul_116, full_5, float("0"), True)
        del matmul_116

        # pd_op.add: (1x6x21x21xf32) <- (1x6x21x21xf32, 1x1x1x21xf32)
        add_128 = paddle._C_ops.add(scale_11, unsqueeze_0)

        # pd_op.softmax: (1x6x21x21xf32) <- (1x6x21x21xf32)
        softmax_21 = paddle._C_ops.softmax(add_128, -1)
        del add_128

        # pd_op.dropout: (1x6x21x21xf32, 1x6x21x21xui8) <- (1x6x21x21xf32, None, 1xf32)
        dropout_62, dropout_63 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_21, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x6x21x64xf32) <- (1x6x21x21xf32, 1x6x21x64xf32)
        matmul_117 = paddle._C_ops.matmul(dropout_62, transpose_42, False, False)

        # pd_op.transpose: (1x21x6x64xf32) <- (1x6x21x64xf32)
        transpose_43 = paddle._C_ops.transpose(matmul_117, [0, 2, 1, 3])
        del matmul_117

        # builtin.combine: ([1x21x6x64xf32, 1x21x6x64xf32]) <- (1x21x6x64xf32, 1x21x6x64xf32)
        combine_21 = [transpose_43, reshape_72]

        # pd_op.concat: (1x21x12x64xf32) <- ([1x21x6x64xf32, 1x21x6x64xf32], 1xi32)
        concat_10 = paddle._C_ops.concat(combine_21, full_6)
        del combine_21

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_76 = paddle._C_ops.reshape(concat_10, full_int_array_27)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_118 = paddle._C_ops.matmul(reshape_76, parameter_41, False, False)
        del parameter_41

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_129 = paddle._C_ops.add(matmul_118, parameter_40)
        del parameter_40

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_64, dropout_65 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_129, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_129

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_130 = paddle._C_ops.add(layer_norm_60, dropout_64)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_130, parameter_28, parameter_27, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_27, parameter_28

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_119 = paddle._C_ops.matmul(layer_norm_63, parameter_32, False, False)
        del parameter_32

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_131 = paddle._C_ops.add(matmul_119, parameter_31)
        del parameter_31

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_10 = paddle._C_ops.gelu(add_131, False)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_120 = paddle._C_ops.matmul(gelu_10, parameter_30, False, False)
        del parameter_30

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_132 = paddle._C_ops.add(matmul_120, parameter_29)
        del parameter_29

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_66, dropout_67 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_132, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_132

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_133 = paddle._C_ops.add(layer_norm_63, dropout_66)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_133, parameter_26, parameter_25, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_25, parameter_26

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_121 = paddle._C_ops.matmul(layer_norm_66, parameter_24, False, False)
        del parameter_24

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_134 = paddle._C_ops.add(matmul_121, parameter_23)
        del parameter_23

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_122 = paddle._C_ops.matmul(layer_norm_66, parameter_22, False, False)
        del parameter_22

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_135 = paddle._C_ops.add(matmul_122, parameter_21)
        del parameter_21

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_123 = paddle._C_ops.matmul(layer_norm_66, parameter_20, False, False)
        del parameter_20

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_136 = paddle._C_ops.add(matmul_123, parameter_19)
        del parameter_19

        # pd_op.assign: (768x1x9xf32) <- (768x1x9xf32)
        assign_384 = parameter_15
        del parameter_15

        # pd_op.unsqueeze: (768x1x1x9xf32) <- (768x1x9xf32, 1xi64)
        unsqueeze_56 = paddle._C_ops.unsqueeze(assign_384, full_int_array_1)

        # pd_op.unsqueeze: (1x1x21x768xf32) <- (1x21x768xf32, 1xi64)
        unsqueeze_57 = paddle._C_ops.unsqueeze(layer_norm_66, full_int_array_2)

        # pd_op.depthwise_conv2d: (1x1x21x768xf32) <- (1x1x21x768xf32, 768x1x1x9xf32)
        depthwise_conv2d_11 = paddle._C_ops.depthwise_conv2d(
            unsqueeze_57, unsqueeze_56, [1, 1], [0, 4], "EXPLICIT", 768, [1, 1], "NHWC"
        )

        # pd_op.squeeze: (1x21x768xf32) <- (1x1x21x768xf32, 1xi64)
        squeeze_33 = paddle._C_ops.squeeze(depthwise_conv2d_11, full_int_array_2)

        # pd_op.assign: (384x768x1xf32) <- (384x768x1xf32)
        assign_385 = parameter_14
        del parameter_14

        # pd_op.unsqueeze: (384x768x1x1xf32) <- (384x768x1xf32, 1xi64)
        unsqueeze_58 = paddle._C_ops.unsqueeze(assign_385, full_int_array_1)

        # pd_op.unsqueeze: (1x1x21x768xf32) <- (1x21x768xf32, 1xi64)
        unsqueeze_59 = paddle._C_ops.unsqueeze(squeeze_33, full_int_array_2)

        # pd_op.conv2d: (1x1x21x384xf32) <- (1x1x21x768xf32, 384x768x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            unsqueeze_59, unsqueeze_58, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NHWC"
        )

        # pd_op.squeeze: (1x21x384xf32) <- (1x1x21x384xf32, 1xi64)
        squeeze_34 = paddle._C_ops.squeeze(conv2d_11, full_int_array_2)

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_137 = paddle._C_ops.add(squeeze_34, parameter_16)
        del parameter_16

        # pd_op.multiply: (1x21x384xf32) <- (1x21x384xf32, 1x21x384xf32)
        multiply_11 = paddle._C_ops.multiply(add_137, add_134)

        # pd_op.matmul: (1x21x54xf32) <- (1x21x384xf32, 384x54xf32)
        matmul_124 = paddle._C_ops.matmul(multiply_11, parameter_13, False, False)
        del parameter_13

        # pd_op.add: (1x21x54xf32) <- (1x21x54xf32, 54xf32)
        add_138 = paddle._C_ops.add(matmul_124, parameter_12)
        del parameter_12

        # pd_op.reshape: (126x9x1xf32) <- (1x21x54xf32, 3xi64)
        reshape_77 = paddle._C_ops.reshape(add_138, full_int_array_3)
        del full_int_array_3

        # pd_op.softmax: (126x9x1xf32) <- (126x9x1xf32)
        softmax_22 = paddle._C_ops.softmax(reshape_77, 1)
        del reshape_77

        # pd_op.matmul: (1x21x384xf32) <- (1x21x768xf32, 768x384xf32)
        matmul_125 = paddle._C_ops.matmul(layer_norm_66, parameter_11, False, False)
        del parameter_11

        # pd_op.add: (1x21x384xf32) <- (1x21x384xf32, 384xf32)
        add_139 = paddle._C_ops.add(matmul_125, parameter_10)
        del parameter_10

        # pd_op.unsqueeze: (1x21x1x1x384xf32) <- (1x21x384xf32, 2xi64)
        unsqueeze_60 = paddle._C_ops.unsqueeze(add_139, full_int_array_4)

        # pd_op.pad3d: (1x29x1x1x384xf32) <- (1x21x1x1x384xf32, 6xi64)
        pad3d_11 = paddle._C_ops.pad3d(
            unsqueeze_60, full_int_array_5, "constant", float("0"), "NDHWC"
        )

        # pd_op.squeeze: (1x29x384xf32) <- (1x29x1x1x384xf32, 2xi64)
        squeeze_35 = paddle._C_ops.squeeze(pad3d_11, full_int_array_4)

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_99 = paddle._C_ops.slice(
            squeeze_35, [1], full_int_array_6, full_int_array_7, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_100 = paddle._C_ops.slice(
            squeeze_35, [1], full_int_array_8, full_int_array_9, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_101 = paddle._C_ops.slice(
            squeeze_35, [1], full_int_array_10, full_int_array_11, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_102 = paddle._C_ops.slice(
            squeeze_35, [1], full_int_array_12, full_int_array_13, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_103 = paddle._C_ops.slice(
            squeeze_35, [1], full_int_array_14, full_int_array_15, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_104 = paddle._C_ops.slice(
            squeeze_35, [1], full_int_array_16, full_int_array_17, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_105 = paddle._C_ops.slice(
            squeeze_35, [1], full_int_array_18, full_int_array_19, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_106 = paddle._C_ops.slice(
            squeeze_35, [1], full_int_array_20, full_int_array_21, [1], []
        )

        # pd_op.slice: (1x21x384xf32) <- (1x29x384xf32, 1xi64, 1xi64)
        slice_107 = paddle._C_ops.slice(
            squeeze_35, [1], full_int_array_22, full_int_array_23, [1], []
        )

        # builtin.combine: ([1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32]) <- (1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32)
        combine_22 = [
            slice_99,
            slice_100,
            slice_101,
            slice_102,
            slice_103,
            slice_104,
            slice_105,
            slice_106,
            slice_107,
        ]

        # pd_op.stack: (1x21x384x9xf32) <- ([1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32, 1x21x384xf32])
        stack_11 = paddle._C_ops.stack(combine_22, -1)
        del combine_22

        # pd_op.reshape: (126x64x9xf32) <- (1x21x384x9xf32, 3xi64)
        reshape_78 = paddle._C_ops.reshape(stack_11, full_int_array_24)
        del full_int_array_24

        # pd_op.matmul: (126x64x1xf32) <- (126x64x9xf32, 126x9x1xf32)
        matmul_126 = paddle._C_ops.matmul(reshape_78, softmax_22, False, False)

        # pd_op.reshape: (1x21x6x64xf32) <- (126x64x1xf32, 4xi64)
        reshape_79 = paddle._C_ops.reshape(matmul_126, full_int_array_25)
        del full_int_array_25

        # pd_op.reshape: (1x21x6x64xf32) <- (1x21x384xf32, 4xi64)
        reshape_80 = paddle._C_ops.reshape(add_134, full_int_array_26)

        # pd_op.transpose: (1x6x21x64xf32) <- (1x21x6x64xf32)
        transpose_44 = paddle._C_ops.transpose(reshape_80, [0, 2, 1, 3])
        del reshape_80

        # pd_op.reshape: (1x21x6x64xf32) <- (1x21x384xf32, 4xi64)
        reshape_81 = paddle._C_ops.reshape(add_135, full_int_array_26)

        # pd_op.transpose: (1x6x21x64xf32) <- (1x21x6x64xf32)
        transpose_45 = paddle._C_ops.transpose(reshape_81, [0, 2, 1, 3])
        del reshape_81

        # pd_op.reshape: (1x21x6x64xf32) <- (1x21x384xf32, 4xi64)
        reshape_82 = paddle._C_ops.reshape(add_136, full_int_array_26)
        del full_int_array_26

        # pd_op.transpose: (1x6x21x64xf32) <- (1x21x6x64xf32)
        transpose_46 = paddle._C_ops.transpose(reshape_82, [0, 2, 1, 3])
        del reshape_82

        # pd_op.matmul: (1x6x21x21xf32) <- (1x6x21x64xf32, 1x6x21x64xf32)
        matmul_127 = paddle._C_ops.matmul(transpose_44, transpose_45, False, True)

        # pd_op.scale: (1x6x21x21xf32) <- (1x6x21x21xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(matmul_127, full_5, float("0"), True)
        del matmul_127

        # pd_op.add: (1x6x21x21xf32) <- (1x6x21x21xf32, 1x1x1x21xf32)
        add_140 = paddle._C_ops.add(scale_12, unsqueeze_0)

        # pd_op.softmax: (1x6x21x21xf32) <- (1x6x21x21xf32)
        softmax_23 = paddle._C_ops.softmax(add_140, -1)
        del add_140

        # pd_op.dropout: (1x6x21x21xf32, 1x6x21x21xui8) <- (1x6x21x21xf32, None, 1xf32)
        dropout_68, dropout_69 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_23, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x6x21x64xf32) <- (1x6x21x21xf32, 1x6x21x64xf32)
        matmul_128 = paddle._C_ops.matmul(dropout_68, transpose_46, False, False)

        # pd_op.transpose: (1x21x6x64xf32) <- (1x6x21x64xf32)
        transpose_47 = paddle._C_ops.transpose(matmul_128, [0, 2, 1, 3])
        del matmul_128

        # builtin.combine: ([1x21x6x64xf32, 1x21x6x64xf32]) <- (1x21x6x64xf32, 1x21x6x64xf32)
        combine_23 = [transpose_47, reshape_79]

        # pd_op.concat: (1x21x12x64xf32) <- ([1x21x6x64xf32, 1x21x6x64xf32], 1xi32)
        concat_11 = paddle._C_ops.concat(combine_23, full_6)
        del combine_23

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_83 = paddle._C_ops.reshape(concat_11, full_int_array_27)
        del full_int_array_27

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_129 = paddle._C_ops.matmul(reshape_83, parameter_18, False, False)
        del parameter_18

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_141 = paddle._C_ops.add(matmul_129, parameter_17)
        del parameter_17

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_70, dropout_71 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_141, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_141

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_142 = paddle._C_ops.add(layer_norm_66, dropout_70)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_142, parameter_5, parameter_4, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_4, parameter_5

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_130 = paddle._C_ops.matmul(layer_norm_69, parameter_9, False, False)
        del parameter_9

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_143 = paddle._C_ops.add(matmul_130, parameter_8)
        del parameter_8

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_11 = paddle._C_ops.gelu(add_143, False)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_131 = paddle._C_ops.matmul(gelu_11, parameter_7, False, False)
        del parameter_7

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_144 = paddle._C_ops.add(matmul_131, parameter_6)
        del parameter_6

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_72, dropout_73 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_144, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_144

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_145 = paddle._C_ops.add(layer_norm_69, dropout_72)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_72, layer_norm_73, layer_norm_74 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_145, parameter_3, parameter_2, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_2, parameter_3

        # pd_op.slice: (1x768xf32) <- (1x21x768xf32, 1xi64, 1xi64)
        slice_108 = paddle._C_ops.slice(
            layer_norm_72, [1], full_int_array_6, full_int_array_8, [1], [1]
        )

        # pd_op.matmul: (1x768xf32) <- (1x768xf32, 768x768xf32)
        matmul_132 = paddle._C_ops.matmul(slice_108, parameter_1, False, False)
        del parameter_1

        # pd_op.add: (1x768xf32) <- (1x768xf32, 768xf32)
        add_146 = paddle._C_ops.add(matmul_132, parameter_0)
        del parameter_0

        # pd_op.tanh: (1x768xf32) <- (1x768xf32)
        tanh_0 = paddle._C_ops.tanh(add_146)
        del (
            add_0,
            add_1,
            add_10,
            add_100,
            add_101,
            add_102,
            add_103,
            add_106,
            add_107,
            add_109,
            add_11,
            add_110,
            add_111,
            add_112,
            add_113,
            add_114,
            add_115,
            add_118,
            add_119,
            add_121,
            add_122,
            add_123,
            add_124,
            add_125,
            add_126,
            add_127,
            add_13,
            add_130,
            add_131,
            add_133,
            add_134,
            add_135,
            add_136,
            add_137,
            add_138,
            add_139,
            add_14,
            add_142,
            add_143,
            add_145,
            add_146,
            add_15,
            add_16,
            add_17,
            add_18,
            add_19,
            add_2,
            add_22,
            add_23,
            add_25,
            add_26,
            add_27,
            add_28,
            add_29,
            add_3,
            add_30,
            add_31,
            add_34,
            add_35,
            add_37,
            add_38,
            add_39,
            add_4,
            add_40,
            add_41,
            add_42,
            add_43,
            add_46,
            add_47,
            add_49,
            add_5,
            add_50,
            add_51,
            add_52,
            add_53,
            add_54,
            add_55,
            add_58,
            add_59,
            add_6,
            add_61,
            add_62,
            add_63,
            add_64,
            add_65,
            add_66,
            add_67,
            add_7,
            add_70,
            add_71,
            add_73,
            add_74,
            add_75,
            add_76,
            add_77,
            add_78,
            add_79,
            add_82,
            add_83,
            add_85,
            add_86,
            add_87,
            add_88,
            add_89,
            add_90,
            add_91,
            add_94,
            add_95,
            add_97,
            add_98,
            add_99,
            assign_0,
            assign_1,
            assign_10,
            assign_100,
            assign_101,
            assign_102,
            assign_103,
            assign_104,
            assign_105,
            assign_106,
            assign_107,
            assign_108,
            assign_109,
            assign_11,
            assign_110,
            assign_111,
            assign_112,
            assign_113,
            assign_114,
            assign_115,
            assign_116,
            assign_117,
            assign_118,
            assign_119,
            assign_12,
            assign_120,
            assign_121,
            assign_122,
            assign_123,
            assign_124,
            assign_125,
            assign_126,
            assign_127,
            assign_128,
            assign_129,
            assign_13,
            assign_130,
            assign_131,
            assign_132,
            assign_133,
            assign_134,
            assign_135,
            assign_136,
            assign_137,
            assign_138,
            assign_139,
            assign_14,
            assign_140,
            assign_141,
            assign_142,
            assign_143,
            assign_144,
            assign_145,
            assign_146,
            assign_147,
            assign_148,
            assign_149,
            assign_15,
            assign_150,
            assign_151,
            assign_152,
            assign_153,
            assign_154,
            assign_155,
            assign_156,
            assign_157,
            assign_158,
            assign_159,
            assign_16,
            assign_160,
            assign_161,
            assign_162,
            assign_163,
            assign_164,
            assign_165,
            assign_166,
            assign_167,
            assign_168,
            assign_169,
            assign_17,
            assign_170,
            assign_171,
            assign_172,
            assign_173,
            assign_174,
            assign_175,
            assign_176,
            assign_177,
            assign_178,
            assign_179,
            assign_18,
            assign_180,
            assign_181,
            assign_182,
            assign_183,
            assign_184,
            assign_185,
            assign_186,
            assign_187,
            assign_188,
            assign_189,
            assign_19,
            assign_190,
            assign_191,
            assign_192,
            assign_193,
            assign_194,
            assign_195,
            assign_196,
            assign_197,
            assign_198,
            assign_199,
            assign_2,
            assign_20,
            assign_200,
            assign_201,
            assign_202,
            assign_203,
            assign_204,
            assign_205,
            assign_206,
            assign_207,
            assign_208,
            assign_209,
            assign_21,
            assign_210,
            assign_211,
            assign_212,
            assign_213,
            assign_214,
            assign_215,
            assign_216,
            assign_217,
            assign_218,
            assign_219,
            assign_22,
            assign_220,
            assign_221,
            assign_222,
            assign_223,
            assign_224,
            assign_225,
            assign_226,
            assign_227,
            assign_228,
            assign_229,
            assign_23,
            assign_230,
            assign_231,
            assign_232,
            assign_233,
            assign_234,
            assign_235,
            assign_236,
            assign_237,
            assign_238,
            assign_239,
            assign_24,
            assign_240,
            assign_241,
            assign_242,
            assign_243,
            assign_244,
            assign_245,
            assign_246,
            assign_247,
            assign_248,
            assign_249,
            assign_25,
            assign_250,
            assign_251,
            assign_252,
            assign_253,
            assign_254,
            assign_255,
            assign_256,
            assign_257,
            assign_258,
            assign_259,
            assign_26,
            assign_260,
            assign_261,
            assign_262,
            assign_263,
            assign_264,
            assign_265,
            assign_266,
            assign_267,
            assign_268,
            assign_269,
            assign_27,
            assign_270,
            assign_271,
            assign_272,
            assign_273,
            assign_274,
            assign_275,
            assign_276,
            assign_277,
            assign_278,
            assign_279,
            assign_28,
            assign_280,
            assign_281,
            assign_282,
            assign_283,
            assign_284,
            assign_285,
            assign_286,
            assign_287,
            assign_288,
            assign_289,
            assign_29,
            assign_290,
            assign_291,
            assign_292,
            assign_293,
            assign_294,
            assign_295,
            assign_296,
            assign_297,
            assign_298,
            assign_299,
            assign_3,
            assign_30,
            assign_300,
            assign_301,
            assign_302,
            assign_303,
            assign_304,
            assign_305,
            assign_306,
            assign_307,
            assign_308,
            assign_309,
            assign_31,
            assign_310,
            assign_311,
            assign_312,
            assign_313,
            assign_314,
            assign_315,
            assign_316,
            assign_317,
            assign_318,
            assign_319,
            assign_32,
            assign_320,
            assign_321,
            assign_322,
            assign_323,
            assign_324,
            assign_325,
            assign_326,
            assign_327,
            assign_328,
            assign_329,
            assign_33,
            assign_330,
            assign_331,
            assign_332,
            assign_333,
            assign_334,
            assign_335,
            assign_336,
            assign_337,
            assign_338,
            assign_339,
            assign_34,
            assign_340,
            assign_341,
            assign_342,
            assign_343,
            assign_344,
            assign_345,
            assign_346,
            assign_347,
            assign_348,
            assign_349,
            assign_35,
            assign_350,
            assign_351,
            assign_352,
            assign_353,
            assign_354,
            assign_355,
            assign_356,
            assign_357,
            assign_358,
            assign_359,
            assign_36,
            assign_360,
            assign_361,
            assign_362,
            assign_363,
            assign_364,
            assign_365,
            assign_366,
            assign_367,
            assign_368,
            assign_369,
            assign_37,
            assign_370,
            assign_371,
            assign_372,
            assign_373,
            assign_374,
            assign_375,
            assign_376,
            assign_377,
            assign_378,
            assign_379,
            assign_38,
            assign_380,
            assign_381,
            assign_382,
            assign_383,
            assign_384,
            assign_385,
            assign_39,
            assign_4,
            assign_40,
            assign_41,
            assign_42,
            assign_43,
            assign_44,
            assign_45,
            assign_46,
            assign_47,
            assign_48,
            assign_49,
            assign_5,
            assign_50,
            assign_51,
            assign_52,
            assign_53,
            assign_54,
            assign_55,
            assign_56,
            assign_57,
            assign_58,
            assign_59,
            assign_6,
            assign_60,
            assign_61,
            assign_62,
            assign_63,
            assign_64,
            assign_65,
            assign_66,
            assign_67,
            assign_68,
            assign_69,
            assign_7,
            assign_70,
            assign_71,
            assign_72,
            assign_73,
            assign_74,
            assign_75,
            assign_76,
            assign_77,
            assign_78,
            assign_79,
            assign_8,
            assign_80,
            assign_81,
            assign_82,
            assign_83,
            assign_84,
            assign_85,
            assign_86,
            assign_87,
            assign_88,
            assign_89,
            assign_9,
            assign_90,
            assign_91,
            assign_92,
            assign_93,
            assign_94,
            assign_95,
            assign_96,
            assign_97,
            assign_98,
            assign_99,
            concat_0,
            concat_1,
            concat_10,
            concat_11,
            concat_2,
            concat_3,
            concat_4,
            concat_5,
            concat_6,
            concat_7,
            concat_8,
            concat_9,
            conv2d_0,
            conv2d_1,
            conv2d_10,
            conv2d_11,
            conv2d_2,
            conv2d_3,
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
            depthwise_conv2d_2,
            depthwise_conv2d_3,
            depthwise_conv2d_4,
            depthwise_conv2d_5,
            depthwise_conv2d_6,
            depthwise_conv2d_7,
            depthwise_conv2d_8,
            depthwise_conv2d_9,
            dropout_0,
            dropout_1,
            dropout_10,
            dropout_11,
            dropout_12,
            dropout_13,
            dropout_14,
            dropout_15,
            dropout_16,
            dropout_17,
            dropout_18,
            dropout_19,
            dropout_2,
            dropout_20,
            dropout_21,
            dropout_22,
            dropout_23,
            dropout_24,
            dropout_25,
            dropout_26,
            dropout_27,
            dropout_28,
            dropout_29,
            dropout_3,
            dropout_30,
            dropout_31,
            dropout_32,
            dropout_33,
            dropout_34,
            dropout_35,
            dropout_36,
            dropout_37,
            dropout_38,
            dropout_39,
            dropout_4,
            dropout_40,
            dropout_41,
            dropout_42,
            dropout_43,
            dropout_44,
            dropout_45,
            dropout_46,
            dropout_47,
            dropout_48,
            dropout_49,
            dropout_5,
            dropout_50,
            dropout_51,
            dropout_52,
            dropout_53,
            dropout_54,
            dropout_55,
            dropout_56,
            dropout_57,
            dropout_58,
            dropout_59,
            dropout_6,
            dropout_60,
            dropout_61,
            dropout_62,
            dropout_63,
            dropout_64,
            dropout_65,
            dropout_66,
            dropout_67,
            dropout_68,
            dropout_69,
            dropout_7,
            dropout_70,
            dropout_71,
            dropout_72,
            dropout_73,
            dropout_8,
            dropout_9,
            embedding_0,
            embedding_1,
            embedding_2,
            full_4,
            full_5,
            full_6,
            full_int_array_1,
            full_int_array_10,
            full_int_array_11,
            full_int_array_12,
            full_int_array_13,
            full_int_array_14,
            full_int_array_15,
            full_int_array_16,
            full_int_array_17,
            full_int_array_18,
            full_int_array_19,
            full_int_array_2,
            full_int_array_20,
            full_int_array_21,
            full_int_array_22,
            full_int_array_23,
            full_int_array_4,
            full_int_array_5,
            full_int_array_6,
            full_int_array_7,
            full_int_array_8,
            full_int_array_9,
            gelu_0,
            gelu_1,
            gelu_10,
            gelu_11,
            gelu_2,
            gelu_3,
            gelu_4,
            gelu_5,
            gelu_6,
            gelu_7,
            gelu_8,
            gelu_9,
            layer_norm_1,
            layer_norm_10,
            layer_norm_11,
            layer_norm_12,
            layer_norm_13,
            layer_norm_14,
            layer_norm_15,
            layer_norm_16,
            layer_norm_17,
            layer_norm_18,
            layer_norm_19,
            layer_norm_2,
            layer_norm_20,
            layer_norm_21,
            layer_norm_22,
            layer_norm_23,
            layer_norm_24,
            layer_norm_25,
            layer_norm_26,
            layer_norm_27,
            layer_norm_28,
            layer_norm_29,
            layer_norm_3,
            layer_norm_30,
            layer_norm_31,
            layer_norm_32,
            layer_norm_33,
            layer_norm_34,
            layer_norm_35,
            layer_norm_36,
            layer_norm_37,
            layer_norm_38,
            layer_norm_39,
            layer_norm_4,
            layer_norm_40,
            layer_norm_41,
            layer_norm_42,
            layer_norm_43,
            layer_norm_44,
            layer_norm_45,
            layer_norm_46,
            layer_norm_47,
            layer_norm_48,
            layer_norm_49,
            layer_norm_5,
            layer_norm_50,
            layer_norm_51,
            layer_norm_52,
            layer_norm_53,
            layer_norm_54,
            layer_norm_55,
            layer_norm_56,
            layer_norm_57,
            layer_norm_58,
            layer_norm_59,
            layer_norm_6,
            layer_norm_60,
            layer_norm_61,
            layer_norm_62,
            layer_norm_63,
            layer_norm_64,
            layer_norm_65,
            layer_norm_66,
            layer_norm_67,
            layer_norm_68,
            layer_norm_69,
            layer_norm_7,
            layer_norm_70,
            layer_norm_71,
            layer_norm_72,
            layer_norm_73,
            layer_norm_74,
            layer_norm_8,
            layer_norm_9,
            matmul_0,
            matmul_1,
            matmul_10,
            matmul_100,
            matmul_101,
            matmul_102,
            matmul_103,
            matmul_104,
            matmul_107,
            matmul_108,
            matmul_109,
            matmul_11,
            matmul_110,
            matmul_111,
            matmul_112,
            matmul_113,
            matmul_114,
            matmul_115,
            matmul_118,
            matmul_119,
            matmul_12,
            matmul_120,
            matmul_121,
            matmul_122,
            matmul_123,
            matmul_124,
            matmul_125,
            matmul_126,
            matmul_129,
            matmul_13,
            matmul_130,
            matmul_131,
            matmul_132,
            matmul_14,
            matmul_15,
            matmul_16,
            matmul_19,
            matmul_2,
            matmul_20,
            matmul_21,
            matmul_22,
            matmul_23,
            matmul_24,
            matmul_25,
            matmul_26,
            matmul_27,
            matmul_3,
            matmul_30,
            matmul_31,
            matmul_32,
            matmul_33,
            matmul_34,
            matmul_35,
            matmul_36,
            matmul_37,
            matmul_38,
            matmul_4,
            matmul_41,
            matmul_42,
            matmul_43,
            matmul_44,
            matmul_45,
            matmul_46,
            matmul_47,
            matmul_48,
            matmul_49,
            matmul_5,
            matmul_52,
            matmul_53,
            matmul_54,
            matmul_55,
            matmul_56,
            matmul_57,
            matmul_58,
            matmul_59,
            matmul_60,
            matmul_63,
            matmul_64,
            matmul_65,
            matmul_66,
            matmul_67,
            matmul_68,
            matmul_69,
            matmul_70,
            matmul_71,
            matmul_74,
            matmul_75,
            matmul_76,
            matmul_77,
            matmul_78,
            matmul_79,
            matmul_8,
            matmul_80,
            matmul_81,
            matmul_82,
            matmul_85,
            matmul_86,
            matmul_87,
            matmul_88,
            matmul_89,
            matmul_9,
            matmul_90,
            matmul_91,
            matmul_92,
            matmul_93,
            matmul_96,
            matmul_97,
            matmul_98,
            matmul_99,
            multiply_0,
            multiply_1,
            multiply_10,
            multiply_11,
            multiply_2,
            multiply_3,
            multiply_4,
            multiply_5,
            multiply_6,
            multiply_7,
            multiply_8,
            multiply_9,
            pad3d_0,
            pad3d_1,
            pad3d_10,
            pad3d_11,
            pad3d_2,
            pad3d_3,
            pad3d_4,
            pad3d_5,
            pad3d_6,
            pad3d_7,
            pad3d_8,
            pad3d_9,
            reshape_1,
            reshape_13,
            reshape_15,
            reshape_16,
            reshape_2,
            reshape_20,
            reshape_22,
            reshape_23,
            reshape_27,
            reshape_29,
            reshape_30,
            reshape_34,
            reshape_36,
            reshape_37,
            reshape_41,
            reshape_43,
            reshape_44,
            reshape_48,
            reshape_50,
            reshape_51,
            reshape_55,
            reshape_57,
            reshape_58,
            reshape_6,
            reshape_62,
            reshape_64,
            reshape_65,
            reshape_69,
            reshape_71,
            reshape_72,
            reshape_76,
            reshape_78,
            reshape_79,
            reshape_8,
            reshape_83,
            reshape_9,
            scale_1,
            scale_10,
            scale_11,
            scale_12,
            scale_2,
            scale_3,
            scale_4,
            scale_5,
            scale_6,
            scale_7,
            scale_8,
            scale_9,
            slice_0,
            slice_1,
            slice_10,
            slice_100,
            slice_101,
            slice_102,
            slice_103,
            slice_104,
            slice_105,
            slice_106,
            slice_107,
            slice_108,
            slice_11,
            slice_12,
            slice_13,
            slice_14,
            slice_15,
            slice_16,
            slice_17,
            slice_18,
            slice_19,
            slice_2,
            slice_20,
            slice_21,
            slice_22,
            slice_23,
            slice_24,
            slice_25,
            slice_26,
            slice_27,
            slice_28,
            slice_29,
            slice_3,
            slice_30,
            slice_31,
            slice_32,
            slice_33,
            slice_34,
            slice_35,
            slice_36,
            slice_37,
            slice_38,
            slice_39,
            slice_4,
            slice_40,
            slice_41,
            slice_42,
            slice_43,
            slice_44,
            slice_45,
            slice_46,
            slice_47,
            slice_48,
            slice_49,
            slice_5,
            slice_50,
            slice_51,
            slice_52,
            slice_53,
            slice_54,
            slice_55,
            slice_56,
            slice_57,
            slice_58,
            slice_59,
            slice_6,
            slice_60,
            slice_61,
            slice_62,
            slice_63,
            slice_64,
            slice_65,
            slice_66,
            slice_67,
            slice_68,
            slice_69,
            slice_7,
            slice_70,
            slice_71,
            slice_72,
            slice_73,
            slice_74,
            slice_75,
            slice_76,
            slice_77,
            slice_78,
            slice_79,
            slice_8,
            slice_80,
            slice_81,
            slice_82,
            slice_83,
            slice_84,
            slice_85,
            slice_86,
            slice_87,
            slice_88,
            slice_89,
            slice_9,
            slice_90,
            slice_91,
            slice_92,
            slice_93,
            slice_94,
            slice_95,
            slice_96,
            slice_97,
            slice_98,
            slice_99,
            softmax_0,
            softmax_1,
            softmax_10,
            softmax_11,
            softmax_12,
            softmax_13,
            softmax_14,
            softmax_15,
            softmax_16,
            softmax_17,
            softmax_18,
            softmax_19,
            softmax_2,
            softmax_20,
            softmax_21,
            softmax_22,
            softmax_23,
            softmax_3,
            softmax_4,
            softmax_5,
            softmax_6,
            softmax_7,
            softmax_8,
            softmax_9,
            squeeze_0,
            squeeze_1,
            squeeze_10,
            squeeze_11,
            squeeze_12,
            squeeze_13,
            squeeze_14,
            squeeze_15,
            squeeze_16,
            squeeze_17,
            squeeze_18,
            squeeze_19,
            squeeze_2,
            squeeze_20,
            squeeze_21,
            squeeze_22,
            squeeze_23,
            squeeze_24,
            squeeze_25,
            squeeze_26,
            squeeze_27,
            squeeze_28,
            squeeze_29,
            squeeze_3,
            squeeze_30,
            squeeze_31,
            squeeze_32,
            squeeze_33,
            squeeze_34,
            squeeze_35,
            squeeze_4,
            squeeze_5,
            squeeze_6,
            squeeze_7,
            squeeze_8,
            squeeze_9,
            stack_0,
            stack_1,
            stack_10,
            stack_11,
            stack_2,
            stack_3,
            stack_4,
            stack_5,
            stack_6,
            stack_7,
            stack_8,
            stack_9,
            subtract_0,
            transpose_0,
            transpose_1,
            transpose_10,
            transpose_11,
            transpose_12,
            transpose_13,
            transpose_14,
            transpose_15,
            transpose_16,
            transpose_17,
            transpose_18,
            transpose_19,
            transpose_2,
            transpose_20,
            transpose_21,
            transpose_22,
            transpose_23,
            transpose_24,
            transpose_25,
            transpose_26,
            transpose_27,
            transpose_28,
            transpose_29,
            transpose_3,
            transpose_30,
            transpose_31,
            transpose_32,
            transpose_33,
            transpose_34,
            transpose_35,
            transpose_36,
            transpose_37,
            transpose_38,
            transpose_39,
            transpose_4,
            transpose_40,
            transpose_41,
            transpose_42,
            transpose_43,
            transpose_44,
            transpose_45,
            transpose_46,
            transpose_47,
            transpose_5,
            transpose_6,
            transpose_7,
            transpose_8,
            transpose_9,
            unsqueeze_0,
            unsqueeze_1,
            unsqueeze_10,
            unsqueeze_11,
            unsqueeze_12,
            unsqueeze_13,
            unsqueeze_14,
            unsqueeze_15,
            unsqueeze_16,
            unsqueeze_17,
            unsqueeze_18,
            unsqueeze_19,
            unsqueeze_2,
            unsqueeze_20,
            unsqueeze_21,
            unsqueeze_22,
            unsqueeze_23,
            unsqueeze_24,
            unsqueeze_25,
            unsqueeze_26,
            unsqueeze_27,
            unsqueeze_28,
            unsqueeze_29,
            unsqueeze_3,
            unsqueeze_30,
            unsqueeze_31,
            unsqueeze_32,
            unsqueeze_33,
            unsqueeze_34,
            unsqueeze_35,
            unsqueeze_36,
            unsqueeze_37,
            unsqueeze_38,
            unsqueeze_39,
            unsqueeze_4,
            unsqueeze_40,
            unsqueeze_41,
            unsqueeze_42,
            unsqueeze_43,
            unsqueeze_44,
            unsqueeze_45,
            unsqueeze_46,
            unsqueeze_47,
            unsqueeze_48,
            unsqueeze_49,
            unsqueeze_5,
            unsqueeze_50,
            unsqueeze_51,
            unsqueeze_52,
            unsqueeze_53,
            unsqueeze_54,
            unsqueeze_55,
            unsqueeze_56,
            unsqueeze_57,
            unsqueeze_58,
            unsqueeze_59,
            unsqueeze_6,
            unsqueeze_60,
            unsqueeze_7,
            unsqueeze_8,
            unsqueeze_9,
        )

        return tanh_0
