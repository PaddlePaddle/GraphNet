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
        data_0,
        data_1,
        data_2,
        data_3,
        data_4,
        data_5,
    ):
        # pd_op.conv2d: (16x1024x16x16xf32) <- (16x3x224x224xf32, 1024x3x14x14xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_4, parameter_294, [14, 14], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_4, parameter_294

        # pd_op.flatten: (16x1024x256xf32) <- (16x1024x16x16xf32)
        flatten_0 = paddle._C_ops.flatten(conv2d_0, 2, 3)

        # pd_op.transpose: (16x256x1024xf32) <- (16x1024x256xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_0, [0, 2, 1])
        del flatten_0

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_0 = [16, -1, -1]

        # pd_op.expand: (16x1x1024xf32) <- (1x1x1024xf32, 3xi64)
        expand_0 = paddle._C_ops.expand(data_0, full_int_array_0)
        del data_0

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([16x1x1024xf32, 16x256x1024xf32]) <- (16x1x1024xf32, 16x256x1024xf32)
        combine_0 = [expand_0, transpose_0]

        # pd_op.concat: (16x257x1024xf32) <- ([16x1x1024xf32, 16x256x1024xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)
        del combine_0

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1x257x1024xf32)
        add_0 = paddle._C_ops.add(concat_0, data_1)
        del data_1

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_0, parameter_293, parameter_292, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_292, parameter_293

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                layer_norm_0, parameter_291, parameter_290, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_290, parameter_291

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        matmul_0 = paddle._C_ops.matmul(layer_norm_3, parameter_289, False, False)
        del parameter_289

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        add_1 = paddle._C_ops.add(matmul_0, parameter_288)
        del parameter_288

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_1 = [-1, 257, 3, 16, 64]

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        reshape_0 = paddle._C_ops.reshape(add_1, full_int_array_1)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_0, [2, 0, 3, 1, 4])
        del reshape_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_3 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_4 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_5 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_6 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_7 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_8 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_9 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_10 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_11 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_12 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_13 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_14 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_15 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_16 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_17 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_18 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_19 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_20 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_21 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_22 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_23 = full_int_array_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_24 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_25 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_26 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_27 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_28 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_29 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_30 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_31 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_32 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_33 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_34 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_35 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_36 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_37 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_38 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_39 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_40 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_41 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_42 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_43 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_44 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_45 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_46 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_47 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_48 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_49 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_50 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_51 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_52 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_53 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_54 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_55 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_56 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_57 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_58 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_59 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_60 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_61 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_62 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_63 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_64 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_65 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_66 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_67 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_68 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_69 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_70 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_71 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_72 = full_int_array_3

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            transpose_1, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_73 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_74 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_75 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_76 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_77 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_78 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_79 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_80 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_81 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_82 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_83 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_84 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_85 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_86 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_87 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_88 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_89 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_90 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_91 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_92 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_93 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_94 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_95 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_96 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_97 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_98 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_99 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_100 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_101 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_102 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_103 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_104 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_105 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_106 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_107 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_108 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_109 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_110 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_111 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_112 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_113 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_114 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_115 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_116 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_117 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_118 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_119 = full_int_array_4

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            transpose_1, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_120 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_121 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_122 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_123 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_124 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_125 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_126 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_127 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_128 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_129 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_130 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_131 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_132 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_133 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_134 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_135 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_136 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_137 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_138 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_139 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_140 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_141 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_142 = full_int_array_5

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            transpose_1, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        transpose_2 = paddle._C_ops.transpose(slice_1, [0, 1, 3, 2])
        del slice_1

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        matmul_1 = paddle._C_ops.matmul(slice_0, transpose_2, False, False)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_143 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_144 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_145 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_146 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_147 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_148 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_149 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_150 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_151 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_152 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_153 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_154 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_155 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_156 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_157 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_158 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_159 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_160 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_161 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_162 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_163 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_164 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_165 = full_1

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(matmul_1, full_1, float("0"), True)
        del matmul_1

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        softmax_0 = paddle._C_ops.softmax(scale_1, -1)
        del scale_1

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        matmul_2 = paddle._C_ops.matmul(softmax_0, slice_2, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        transpose_3 = paddle._C_ops.transpose(matmul_2, [0, 2, 1, 3])
        del matmul_2

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_6 = [-1, 257, 1024]

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        reshape_1 = paddle._C_ops.reshape(transpose_3, full_int_array_6)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        matmul_3 = paddle._C_ops.matmul(reshape_1, parameter_287, False, False)
        del parameter_287

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_2 = paddle._C_ops.add(matmul_3, parameter_286)
        del parameter_286

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_3 = paddle._C_ops.add(layer_norm_0, add_2)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_3, parameter_285, parameter_284, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_284, parameter_285

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        matmul_4 = paddle._C_ops.matmul(layer_norm_6, parameter_283, False, False)
        del parameter_283

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        add_4 = paddle._C_ops.add(matmul_4, parameter_282)
        del parameter_282

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        gelu_0 = paddle._C_ops.gelu(add_4, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        matmul_5 = paddle._C_ops.matmul(gelu_0, parameter_281, False, False)
        del parameter_281

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_5 = paddle._C_ops.add(matmul_5, parameter_280)
        del parameter_280

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_6 = paddle._C_ops.add(add_3, add_5)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_6, parameter_279, parameter_278, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_278, parameter_279

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        matmul_6 = paddle._C_ops.matmul(layer_norm_9, parameter_277, False, False)
        del parameter_277

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        add_7 = paddle._C_ops.add(matmul_6, parameter_276)
        del parameter_276

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        reshape_2 = paddle._C_ops.reshape(add_7, full_int_array_1)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_2, [2, 0, 3, 1, 4])
        del reshape_2

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            transpose_4, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            transpose_4, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            transpose_4, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        transpose_5 = paddle._C_ops.transpose(slice_4, [0, 1, 3, 2])
        del slice_4

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        matmul_7 = paddle._C_ops.matmul(slice_3, transpose_5, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(matmul_7, full_1, float("0"), True)
        del matmul_7

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        softmax_1 = paddle._C_ops.softmax(scale_2, -1)
        del scale_2

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        matmul_8 = paddle._C_ops.matmul(softmax_1, slice_5, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        transpose_6 = paddle._C_ops.transpose(matmul_8, [0, 2, 1, 3])
        del matmul_8

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_6, full_int_array_6)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        matmul_9 = paddle._C_ops.matmul(reshape_3, parameter_275, False, False)
        del parameter_275

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_8 = paddle._C_ops.add(matmul_9, parameter_274)
        del parameter_274

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_9 = paddle._C_ops.add(add_6, add_8)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_9, parameter_273, parameter_272, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_272, parameter_273

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_12, parameter_271, False, False)
        del parameter_271

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        add_10 = paddle._C_ops.add(matmul_10, parameter_270)
        del parameter_270

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        gelu_1 = paddle._C_ops.gelu(add_10, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        matmul_11 = paddle._C_ops.matmul(gelu_1, parameter_269, False, False)
        del parameter_269

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_11 = paddle._C_ops.add(matmul_11, parameter_268)
        del parameter_268

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_12 = paddle._C_ops.add(add_9, add_11)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_12, parameter_267, parameter_266, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_266, parameter_267

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        matmul_12 = paddle._C_ops.matmul(layer_norm_15, parameter_265, False, False)
        del parameter_265

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        add_13 = paddle._C_ops.add(matmul_12, parameter_264)
        del parameter_264

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        reshape_4 = paddle._C_ops.reshape(add_13, full_int_array_1)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        transpose_7 = paddle._C_ops.transpose(reshape_4, [2, 0, 3, 1, 4])
        del reshape_4

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            transpose_7, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            transpose_7, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            transpose_7, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        transpose_8 = paddle._C_ops.transpose(slice_7, [0, 1, 3, 2])
        del slice_7

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        matmul_13 = paddle._C_ops.matmul(slice_6, transpose_8, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(matmul_13, full_1, float("0"), True)
        del matmul_13

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        softmax_2 = paddle._C_ops.softmax(scale_3, -1)
        del scale_3

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        matmul_14 = paddle._C_ops.matmul(softmax_2, slice_8, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        transpose_9 = paddle._C_ops.transpose(matmul_14, [0, 2, 1, 3])
        del matmul_14

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        reshape_5 = paddle._C_ops.reshape(transpose_9, full_int_array_6)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        matmul_15 = paddle._C_ops.matmul(reshape_5, parameter_263, False, False)
        del parameter_263

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_14 = paddle._C_ops.add(matmul_15, parameter_262)
        del parameter_262

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_15 = paddle._C_ops.add(add_12, add_14)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_15, parameter_261, parameter_260, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_260, parameter_261

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        matmul_16 = paddle._C_ops.matmul(layer_norm_18, parameter_259, False, False)
        del parameter_259

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        add_16 = paddle._C_ops.add(matmul_16, parameter_258)
        del parameter_258

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        gelu_2 = paddle._C_ops.gelu(add_16, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        matmul_17 = paddle._C_ops.matmul(gelu_2, parameter_257, False, False)
        del parameter_257

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_17 = paddle._C_ops.add(matmul_17, parameter_256)
        del parameter_256

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_18 = paddle._C_ops.add(add_15, add_17)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_18, parameter_255, parameter_254, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_254, parameter_255

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        matmul_18 = paddle._C_ops.matmul(layer_norm_21, parameter_253, False, False)
        del parameter_253

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        add_19 = paddle._C_ops.add(matmul_18, parameter_252)
        del parameter_252

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        reshape_6 = paddle._C_ops.reshape(add_19, full_int_array_1)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        transpose_10 = paddle._C_ops.transpose(reshape_6, [2, 0, 3, 1, 4])
        del reshape_6

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            transpose_10, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            transpose_10, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            transpose_10, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        transpose_11 = paddle._C_ops.transpose(slice_10, [0, 1, 3, 2])
        del slice_10

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        matmul_19 = paddle._C_ops.matmul(slice_9, transpose_11, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(matmul_19, full_1, float("0"), True)
        del matmul_19

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        softmax_3 = paddle._C_ops.softmax(scale_4, -1)
        del scale_4

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        matmul_20 = paddle._C_ops.matmul(softmax_3, slice_11, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        transpose_12 = paddle._C_ops.transpose(matmul_20, [0, 2, 1, 3])
        del matmul_20

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_12, full_int_array_6)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        matmul_21 = paddle._C_ops.matmul(reshape_7, parameter_251, False, False)
        del parameter_251

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_20 = paddle._C_ops.add(matmul_21, parameter_250)
        del parameter_250

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_21 = paddle._C_ops.add(add_18, add_20)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_21, parameter_249, parameter_248, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_248, parameter_249

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        matmul_22 = paddle._C_ops.matmul(layer_norm_24, parameter_247, False, False)
        del parameter_247

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        add_22 = paddle._C_ops.add(matmul_22, parameter_246)
        del parameter_246

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        gelu_3 = paddle._C_ops.gelu(add_22, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        matmul_23 = paddle._C_ops.matmul(gelu_3, parameter_245, False, False)
        del parameter_245

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_23 = paddle._C_ops.add(matmul_23, parameter_244)
        del parameter_244

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_24 = paddle._C_ops.add(add_21, add_23)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_24, parameter_243, parameter_242, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_242, parameter_243

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        matmul_24 = paddle._C_ops.matmul(layer_norm_27, parameter_241, False, False)
        del parameter_241

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        add_25 = paddle._C_ops.add(matmul_24, parameter_240)
        del parameter_240

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        reshape_8 = paddle._C_ops.reshape(add_25, full_int_array_1)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_8, [2, 0, 3, 1, 4])
        del reshape_8

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            transpose_13, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            transpose_13, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            transpose_13, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        transpose_14 = paddle._C_ops.transpose(slice_13, [0, 1, 3, 2])
        del slice_13

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        matmul_25 = paddle._C_ops.matmul(slice_12, transpose_14, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(matmul_25, full_1, float("0"), True)
        del matmul_25

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        softmax_4 = paddle._C_ops.softmax(scale_5, -1)
        del scale_5

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        matmul_26 = paddle._C_ops.matmul(softmax_4, slice_14, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        transpose_15 = paddle._C_ops.transpose(matmul_26, [0, 2, 1, 3])
        del matmul_26

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        reshape_9 = paddle._C_ops.reshape(transpose_15, full_int_array_6)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        matmul_27 = paddle._C_ops.matmul(reshape_9, parameter_239, False, False)
        del parameter_239

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_26 = paddle._C_ops.add(matmul_27, parameter_238)
        del parameter_238

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_27 = paddle._C_ops.add(add_24, add_26)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_27, parameter_237, parameter_236, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_236, parameter_237

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        matmul_28 = paddle._C_ops.matmul(layer_norm_30, parameter_235, False, False)
        del parameter_235

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        add_28 = paddle._C_ops.add(matmul_28, parameter_234)
        del parameter_234

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        gelu_4 = paddle._C_ops.gelu(add_28, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        matmul_29 = paddle._C_ops.matmul(gelu_4, parameter_233, False, False)
        del parameter_233

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_29 = paddle._C_ops.add(matmul_29, parameter_232)
        del parameter_232

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_30 = paddle._C_ops.add(add_27, add_29)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_30, parameter_231, parameter_230, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_230, parameter_231

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        matmul_30 = paddle._C_ops.matmul(layer_norm_33, parameter_229, False, False)
        del parameter_229

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        add_31 = paddle._C_ops.add(matmul_30, parameter_228)
        del parameter_228

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        reshape_10 = paddle._C_ops.reshape(add_31, full_int_array_1)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        transpose_16 = paddle._C_ops.transpose(reshape_10, [2, 0, 3, 1, 4])
        del reshape_10

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            transpose_16, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            transpose_16, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            transpose_16, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        transpose_17 = paddle._C_ops.transpose(slice_16, [0, 1, 3, 2])
        del slice_16

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        matmul_31 = paddle._C_ops.matmul(slice_15, transpose_17, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(matmul_31, full_1, float("0"), True)
        del matmul_31

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        softmax_5 = paddle._C_ops.softmax(scale_6, -1)
        del scale_6

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        matmul_32 = paddle._C_ops.matmul(softmax_5, slice_17, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        transpose_18 = paddle._C_ops.transpose(matmul_32, [0, 2, 1, 3])
        del matmul_32

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_18, full_int_array_6)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        matmul_33 = paddle._C_ops.matmul(reshape_11, parameter_227, False, False)
        del parameter_227

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_32 = paddle._C_ops.add(matmul_33, parameter_226)
        del parameter_226

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_33 = paddle._C_ops.add(add_30, add_32)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_33, parameter_225, parameter_224, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_224, parameter_225

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        matmul_34 = paddle._C_ops.matmul(layer_norm_36, parameter_223, False, False)
        del parameter_223

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        add_34 = paddle._C_ops.add(matmul_34, parameter_222)
        del parameter_222

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        gelu_5 = paddle._C_ops.gelu(add_34, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        matmul_35 = paddle._C_ops.matmul(gelu_5, parameter_221, False, False)
        del parameter_221

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_35 = paddle._C_ops.add(matmul_35, parameter_220)
        del parameter_220

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_36 = paddle._C_ops.add(add_33, add_35)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_36, parameter_219, parameter_218, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_218, parameter_219

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        matmul_36 = paddle._C_ops.matmul(layer_norm_39, parameter_217, False, False)
        del parameter_217

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        add_37 = paddle._C_ops.add(matmul_36, parameter_216)
        del parameter_216

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        reshape_12 = paddle._C_ops.reshape(add_37, full_int_array_1)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        transpose_19 = paddle._C_ops.transpose(reshape_12, [2, 0, 3, 1, 4])
        del reshape_12

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(
            transpose_19, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(
            transpose_19, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(
            transpose_19, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        transpose_20 = paddle._C_ops.transpose(slice_19, [0, 1, 3, 2])
        del slice_19

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        matmul_37 = paddle._C_ops.matmul(slice_18, transpose_20, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(matmul_37, full_1, float("0"), True)
        del matmul_37

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        softmax_6 = paddle._C_ops.softmax(scale_7, -1)
        del scale_7

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        matmul_38 = paddle._C_ops.matmul(softmax_6, slice_20, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        transpose_21 = paddle._C_ops.transpose(matmul_38, [0, 2, 1, 3])
        del matmul_38

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        reshape_13 = paddle._C_ops.reshape(transpose_21, full_int_array_6)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        matmul_39 = paddle._C_ops.matmul(reshape_13, parameter_215, False, False)
        del parameter_215

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_38 = paddle._C_ops.add(matmul_39, parameter_214)
        del parameter_214

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_39 = paddle._C_ops.add(add_36, add_38)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_39, parameter_213, parameter_212, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_212, parameter_213

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        matmul_40 = paddle._C_ops.matmul(layer_norm_42, parameter_211, False, False)
        del parameter_211

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        add_40 = paddle._C_ops.add(matmul_40, parameter_210)
        del parameter_210

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        gelu_6 = paddle._C_ops.gelu(add_40, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        matmul_41 = paddle._C_ops.matmul(gelu_6, parameter_209, False, False)
        del parameter_209

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_41 = paddle._C_ops.add(matmul_41, parameter_208)
        del parameter_208

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_42 = paddle._C_ops.add(add_39, add_41)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_42, parameter_207, parameter_206, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_206, parameter_207

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        matmul_42 = paddle._C_ops.matmul(layer_norm_45, parameter_205, False, False)
        del parameter_205

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        add_43 = paddle._C_ops.add(matmul_42, parameter_204)
        del parameter_204

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        reshape_14 = paddle._C_ops.reshape(add_43, full_int_array_1)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        transpose_22 = paddle._C_ops.transpose(reshape_14, [2, 0, 3, 1, 4])
        del reshape_14

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(
            transpose_22, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(
            transpose_22, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(
            transpose_22, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        transpose_23 = paddle._C_ops.transpose(slice_22, [0, 1, 3, 2])
        del slice_22

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        matmul_43 = paddle._C_ops.matmul(slice_21, transpose_23, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(matmul_43, full_1, float("0"), True)
        del matmul_43

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        softmax_7 = paddle._C_ops.softmax(scale_8, -1)
        del scale_8

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        matmul_44 = paddle._C_ops.matmul(softmax_7, slice_23, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        transpose_24 = paddle._C_ops.transpose(matmul_44, [0, 2, 1, 3])
        del matmul_44

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        reshape_15 = paddle._C_ops.reshape(transpose_24, full_int_array_6)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        matmul_45 = paddle._C_ops.matmul(reshape_15, parameter_203, False, False)
        del parameter_203

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_44 = paddle._C_ops.add(matmul_45, parameter_202)
        del parameter_202

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_45 = paddle._C_ops.add(add_42, add_44)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_45, parameter_201, parameter_200, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_200, parameter_201

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        matmul_46 = paddle._C_ops.matmul(layer_norm_48, parameter_199, False, False)
        del parameter_199

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        add_46 = paddle._C_ops.add(matmul_46, parameter_198)
        del parameter_198

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        gelu_7 = paddle._C_ops.gelu(add_46, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        matmul_47 = paddle._C_ops.matmul(gelu_7, parameter_197, False, False)
        del parameter_197

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_47 = paddle._C_ops.add(matmul_47, parameter_196)
        del parameter_196

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_48 = paddle._C_ops.add(add_45, add_47)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_48, parameter_195, parameter_194, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_194, parameter_195

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        matmul_48 = paddle._C_ops.matmul(layer_norm_51, parameter_193, False, False)
        del parameter_193

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        add_49 = paddle._C_ops.add(matmul_48, parameter_192)
        del parameter_192

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        reshape_16 = paddle._C_ops.reshape(add_49, full_int_array_1)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        transpose_25 = paddle._C_ops.transpose(reshape_16, [2, 0, 3, 1, 4])
        del reshape_16

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(
            transpose_25, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(
            transpose_25, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(
            transpose_25, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        transpose_26 = paddle._C_ops.transpose(slice_25, [0, 1, 3, 2])
        del slice_25

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        matmul_49 = paddle._C_ops.matmul(slice_24, transpose_26, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(matmul_49, full_1, float("0"), True)
        del matmul_49

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        softmax_8 = paddle._C_ops.softmax(scale_9, -1)
        del scale_9

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        matmul_50 = paddle._C_ops.matmul(softmax_8, slice_26, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        transpose_27 = paddle._C_ops.transpose(matmul_50, [0, 2, 1, 3])
        del matmul_50

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        reshape_17 = paddle._C_ops.reshape(transpose_27, full_int_array_6)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        matmul_51 = paddle._C_ops.matmul(reshape_17, parameter_191, False, False)
        del parameter_191

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_50 = paddle._C_ops.add(matmul_51, parameter_190)
        del parameter_190

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_51 = paddle._C_ops.add(add_48, add_50)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_51, parameter_189, parameter_188, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_188, parameter_189

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        matmul_52 = paddle._C_ops.matmul(layer_norm_54, parameter_187, False, False)
        del parameter_187

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        add_52 = paddle._C_ops.add(matmul_52, parameter_186)
        del parameter_186

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        gelu_8 = paddle._C_ops.gelu(add_52, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        matmul_53 = paddle._C_ops.matmul(gelu_8, parameter_185, False, False)
        del parameter_185

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_53 = paddle._C_ops.add(matmul_53, parameter_184)
        del parameter_184

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_54 = paddle._C_ops.add(add_51, add_53)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_54, parameter_183, parameter_182, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_182, parameter_183

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        matmul_54 = paddle._C_ops.matmul(layer_norm_57, parameter_181, False, False)
        del parameter_181

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        add_55 = paddle._C_ops.add(matmul_54, parameter_180)
        del parameter_180

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        reshape_18 = paddle._C_ops.reshape(add_55, full_int_array_1)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        transpose_28 = paddle._C_ops.transpose(reshape_18, [2, 0, 3, 1, 4])
        del reshape_18

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(
            transpose_28, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(
            transpose_28, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(
            transpose_28, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        transpose_29 = paddle._C_ops.transpose(slice_28, [0, 1, 3, 2])
        del slice_28

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        matmul_55 = paddle._C_ops.matmul(slice_27, transpose_29, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(matmul_55, full_1, float("0"), True)
        del matmul_55

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        softmax_9 = paddle._C_ops.softmax(scale_10, -1)
        del scale_10

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        matmul_56 = paddle._C_ops.matmul(softmax_9, slice_29, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        transpose_30 = paddle._C_ops.transpose(matmul_56, [0, 2, 1, 3])
        del matmul_56

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        reshape_19 = paddle._C_ops.reshape(transpose_30, full_int_array_6)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        matmul_57 = paddle._C_ops.matmul(reshape_19, parameter_179, False, False)
        del parameter_179

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_56 = paddle._C_ops.add(matmul_57, parameter_178)
        del parameter_178

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_57 = paddle._C_ops.add(add_54, add_56)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_57, parameter_177, parameter_176, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_176, parameter_177

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        matmul_58 = paddle._C_ops.matmul(layer_norm_60, parameter_175, False, False)
        del parameter_175

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        add_58 = paddle._C_ops.add(matmul_58, parameter_174)
        del parameter_174

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        gelu_9 = paddle._C_ops.gelu(add_58, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        matmul_59 = paddle._C_ops.matmul(gelu_9, parameter_173, False, False)
        del parameter_173

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_59 = paddle._C_ops.add(matmul_59, parameter_172)
        del parameter_172

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_60 = paddle._C_ops.add(add_57, add_59)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_60, parameter_171, parameter_170, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_170, parameter_171

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        matmul_60 = paddle._C_ops.matmul(layer_norm_63, parameter_169, False, False)
        del parameter_169

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        add_61 = paddle._C_ops.add(matmul_60, parameter_168)
        del parameter_168

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        reshape_20 = paddle._C_ops.reshape(add_61, full_int_array_1)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        transpose_31 = paddle._C_ops.transpose(reshape_20, [2, 0, 3, 1, 4])
        del reshape_20

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(
            transpose_31, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(
            transpose_31, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(
            transpose_31, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        transpose_32 = paddle._C_ops.transpose(slice_31, [0, 1, 3, 2])
        del slice_31

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        matmul_61 = paddle._C_ops.matmul(slice_30, transpose_32, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(matmul_61, full_1, float("0"), True)
        del matmul_61

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        softmax_10 = paddle._C_ops.softmax(scale_11, -1)
        del scale_11

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        matmul_62 = paddle._C_ops.matmul(softmax_10, slice_32, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        transpose_33 = paddle._C_ops.transpose(matmul_62, [0, 2, 1, 3])
        del matmul_62

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        reshape_21 = paddle._C_ops.reshape(transpose_33, full_int_array_6)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        matmul_63 = paddle._C_ops.matmul(reshape_21, parameter_167, False, False)
        del parameter_167

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_62 = paddle._C_ops.add(matmul_63, parameter_166)
        del parameter_166

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_63 = paddle._C_ops.add(add_60, add_62)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_63, parameter_165, parameter_164, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_164, parameter_165

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        matmul_64 = paddle._C_ops.matmul(layer_norm_66, parameter_163, False, False)
        del parameter_163

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        add_64 = paddle._C_ops.add(matmul_64, parameter_162)
        del parameter_162

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        gelu_10 = paddle._C_ops.gelu(add_64, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        matmul_65 = paddle._C_ops.matmul(gelu_10, parameter_161, False, False)
        del parameter_161

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_65 = paddle._C_ops.add(matmul_65, parameter_160)
        del parameter_160

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_66 = paddle._C_ops.add(add_63, add_65)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_66, parameter_159, parameter_158, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_158, parameter_159

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        matmul_66 = paddle._C_ops.matmul(layer_norm_69, parameter_157, False, False)
        del parameter_157

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        add_67 = paddle._C_ops.add(matmul_66, parameter_156)
        del parameter_156

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        reshape_22 = paddle._C_ops.reshape(add_67, full_int_array_1)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        transpose_34 = paddle._C_ops.transpose(reshape_22, [2, 0, 3, 1, 4])
        del reshape_22

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_33 = paddle._C_ops.slice(
            transpose_34, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_34 = paddle._C_ops.slice(
            transpose_34, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_35 = paddle._C_ops.slice(
            transpose_34, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        transpose_35 = paddle._C_ops.transpose(slice_34, [0, 1, 3, 2])
        del slice_34

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        matmul_67 = paddle._C_ops.matmul(slice_33, transpose_35, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(matmul_67, full_1, float("0"), True)
        del matmul_67

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        softmax_11 = paddle._C_ops.softmax(scale_12, -1)
        del scale_12

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        matmul_68 = paddle._C_ops.matmul(softmax_11, slice_35, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        transpose_36 = paddle._C_ops.transpose(matmul_68, [0, 2, 1, 3])
        del matmul_68

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        reshape_23 = paddle._C_ops.reshape(transpose_36, full_int_array_6)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        matmul_69 = paddle._C_ops.matmul(reshape_23, parameter_155, False, False)
        del parameter_155

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_68 = paddle._C_ops.add(matmul_69, parameter_154)
        del parameter_154

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_69 = paddle._C_ops.add(add_66, add_68)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_72, layer_norm_73, layer_norm_74 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_69, parameter_153, parameter_152, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_152, parameter_153

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        matmul_70 = paddle._C_ops.matmul(layer_norm_72, parameter_151, False, False)
        del parameter_151

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        add_70 = paddle._C_ops.add(matmul_70, parameter_150)
        del parameter_150

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        gelu_11 = paddle._C_ops.gelu(add_70, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        matmul_71 = paddle._C_ops.matmul(gelu_11, parameter_149, False, False)
        del parameter_149

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_71 = paddle._C_ops.add(matmul_71, parameter_148)
        del parameter_148

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_72 = paddle._C_ops.add(add_69, add_71)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_75, layer_norm_76, layer_norm_77 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_72, parameter_147, parameter_146, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_146, parameter_147

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        matmul_72 = paddle._C_ops.matmul(layer_norm_75, parameter_145, False, False)
        del parameter_145

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        add_73 = paddle._C_ops.add(matmul_72, parameter_144)
        del parameter_144

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        reshape_24 = paddle._C_ops.reshape(add_73, full_int_array_1)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        transpose_37 = paddle._C_ops.transpose(reshape_24, [2, 0, 3, 1, 4])
        del reshape_24

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(
            transpose_37, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_37 = paddle._C_ops.slice(
            transpose_37, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_38 = paddle._C_ops.slice(
            transpose_37, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        transpose_38 = paddle._C_ops.transpose(slice_37, [0, 1, 3, 2])
        del slice_37

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        matmul_73 = paddle._C_ops.matmul(slice_36, transpose_38, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(matmul_73, full_1, float("0"), True)
        del matmul_73

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        softmax_12 = paddle._C_ops.softmax(scale_13, -1)
        del scale_13

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        matmul_74 = paddle._C_ops.matmul(softmax_12, slice_38, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        transpose_39 = paddle._C_ops.transpose(matmul_74, [0, 2, 1, 3])
        del matmul_74

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        reshape_25 = paddle._C_ops.reshape(transpose_39, full_int_array_6)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        matmul_75 = paddle._C_ops.matmul(reshape_25, parameter_143, False, False)
        del parameter_143

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_74 = paddle._C_ops.add(matmul_75, parameter_142)
        del parameter_142

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_75 = paddle._C_ops.add(add_72, add_74)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_78, layer_norm_79, layer_norm_80 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_75, parameter_141, parameter_140, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_140, parameter_141

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        matmul_76 = paddle._C_ops.matmul(layer_norm_78, parameter_139, False, False)
        del parameter_139

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        add_76 = paddle._C_ops.add(matmul_76, parameter_138)
        del parameter_138

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        gelu_12 = paddle._C_ops.gelu(add_76, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        matmul_77 = paddle._C_ops.matmul(gelu_12, parameter_137, False, False)
        del parameter_137

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_77 = paddle._C_ops.add(matmul_77, parameter_136)
        del parameter_136

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_78 = paddle._C_ops.add(add_75, add_77)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_81, layer_norm_82, layer_norm_83 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_78, parameter_135, parameter_134, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_134, parameter_135

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        matmul_78 = paddle._C_ops.matmul(layer_norm_81, parameter_133, False, False)
        del parameter_133

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        add_79 = paddle._C_ops.add(matmul_78, parameter_132)
        del parameter_132

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        reshape_26 = paddle._C_ops.reshape(add_79, full_int_array_1)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        transpose_40 = paddle._C_ops.transpose(reshape_26, [2, 0, 3, 1, 4])
        del reshape_26

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_39 = paddle._C_ops.slice(
            transpose_40, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_40 = paddle._C_ops.slice(
            transpose_40, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_41 = paddle._C_ops.slice(
            transpose_40, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        transpose_41 = paddle._C_ops.transpose(slice_40, [0, 1, 3, 2])
        del slice_40

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        matmul_79 = paddle._C_ops.matmul(slice_39, transpose_41, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(matmul_79, full_1, float("0"), True)
        del matmul_79

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        softmax_13 = paddle._C_ops.softmax(scale_14, -1)
        del scale_14

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        matmul_80 = paddle._C_ops.matmul(softmax_13, slice_41, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        transpose_42 = paddle._C_ops.transpose(matmul_80, [0, 2, 1, 3])
        del matmul_80

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        reshape_27 = paddle._C_ops.reshape(transpose_42, full_int_array_6)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        matmul_81 = paddle._C_ops.matmul(reshape_27, parameter_131, False, False)
        del parameter_131

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_80 = paddle._C_ops.add(matmul_81, parameter_130)
        del parameter_130

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_81 = paddle._C_ops.add(add_78, add_80)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_84, layer_norm_85, layer_norm_86 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_81, parameter_129, parameter_128, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_128, parameter_129

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        matmul_82 = paddle._C_ops.matmul(layer_norm_84, parameter_127, False, False)
        del parameter_127

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        add_82 = paddle._C_ops.add(matmul_82, parameter_126)
        del parameter_126

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        gelu_13 = paddle._C_ops.gelu(add_82, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        matmul_83 = paddle._C_ops.matmul(gelu_13, parameter_125, False, False)
        del parameter_125

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_83 = paddle._C_ops.add(matmul_83, parameter_124)
        del parameter_124

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_84 = paddle._C_ops.add(add_81, add_83)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_87, layer_norm_88, layer_norm_89 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_84, parameter_123, parameter_122, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_122, parameter_123

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        matmul_84 = paddle._C_ops.matmul(layer_norm_87, parameter_121, False, False)
        del parameter_121

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        add_85 = paddle._C_ops.add(matmul_84, parameter_120)
        del parameter_120

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        reshape_28 = paddle._C_ops.reshape(add_85, full_int_array_1)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        transpose_43 = paddle._C_ops.transpose(reshape_28, [2, 0, 3, 1, 4])
        del reshape_28

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_42 = paddle._C_ops.slice(
            transpose_43, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_43 = paddle._C_ops.slice(
            transpose_43, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_44 = paddle._C_ops.slice(
            transpose_43, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        transpose_44 = paddle._C_ops.transpose(slice_43, [0, 1, 3, 2])
        del slice_43

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        matmul_85 = paddle._C_ops.matmul(slice_42, transpose_44, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        scale_15 = paddle._C_ops.scale(matmul_85, full_1, float("0"), True)
        del matmul_85

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        softmax_14 = paddle._C_ops.softmax(scale_15, -1)
        del scale_15

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        matmul_86 = paddle._C_ops.matmul(softmax_14, slice_44, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        transpose_45 = paddle._C_ops.transpose(matmul_86, [0, 2, 1, 3])
        del matmul_86

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        reshape_29 = paddle._C_ops.reshape(transpose_45, full_int_array_6)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        matmul_87 = paddle._C_ops.matmul(reshape_29, parameter_119, False, False)
        del parameter_119

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_86 = paddle._C_ops.add(matmul_87, parameter_118)
        del parameter_118

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_87 = paddle._C_ops.add(add_84, add_86)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_90, layer_norm_91, layer_norm_92 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_87, parameter_117, parameter_116, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_116, parameter_117

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        matmul_88 = paddle._C_ops.matmul(layer_norm_90, parameter_115, False, False)
        del parameter_115

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        add_88 = paddle._C_ops.add(matmul_88, parameter_114)
        del parameter_114

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        gelu_14 = paddle._C_ops.gelu(add_88, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        matmul_89 = paddle._C_ops.matmul(gelu_14, parameter_113, False, False)
        del parameter_113

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_89 = paddle._C_ops.add(matmul_89, parameter_112)
        del parameter_112

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_90 = paddle._C_ops.add(add_87, add_89)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_93, layer_norm_94, layer_norm_95 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_90, parameter_111, parameter_110, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_110, parameter_111

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        matmul_90 = paddle._C_ops.matmul(layer_norm_93, parameter_109, False, False)
        del parameter_109

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        add_91 = paddle._C_ops.add(matmul_90, parameter_108)
        del parameter_108

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        reshape_30 = paddle._C_ops.reshape(add_91, full_int_array_1)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        transpose_46 = paddle._C_ops.transpose(reshape_30, [2, 0, 3, 1, 4])
        del reshape_30

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_45 = paddle._C_ops.slice(
            transpose_46, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_46 = paddle._C_ops.slice(
            transpose_46, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_47 = paddle._C_ops.slice(
            transpose_46, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        transpose_47 = paddle._C_ops.transpose(slice_46, [0, 1, 3, 2])
        del slice_46

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        matmul_91 = paddle._C_ops.matmul(slice_45, transpose_47, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        scale_16 = paddle._C_ops.scale(matmul_91, full_1, float("0"), True)
        del matmul_91

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        softmax_15 = paddle._C_ops.softmax(scale_16, -1)
        del scale_16

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        matmul_92 = paddle._C_ops.matmul(softmax_15, slice_47, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        transpose_48 = paddle._C_ops.transpose(matmul_92, [0, 2, 1, 3])
        del matmul_92

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        reshape_31 = paddle._C_ops.reshape(transpose_48, full_int_array_6)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        matmul_93 = paddle._C_ops.matmul(reshape_31, parameter_107, False, False)
        del parameter_107

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_92 = paddle._C_ops.add(matmul_93, parameter_106)
        del parameter_106

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_93 = paddle._C_ops.add(add_90, add_92)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_96, layer_norm_97, layer_norm_98 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_93, parameter_105, parameter_104, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_104, parameter_105

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        matmul_94 = paddle._C_ops.matmul(layer_norm_96, parameter_103, False, False)
        del parameter_103

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        add_94 = paddle._C_ops.add(matmul_94, parameter_102)
        del parameter_102

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        gelu_15 = paddle._C_ops.gelu(add_94, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        matmul_95 = paddle._C_ops.matmul(gelu_15, parameter_101, False, False)
        del parameter_101

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_95 = paddle._C_ops.add(matmul_95, parameter_100)
        del parameter_100

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_96 = paddle._C_ops.add(add_93, add_95)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_99, layer_norm_100, layer_norm_101 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_96, parameter_99, parameter_98, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_98, parameter_99

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        matmul_96 = paddle._C_ops.matmul(layer_norm_99, parameter_97, False, False)
        del parameter_97

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        add_97 = paddle._C_ops.add(matmul_96, parameter_96)
        del parameter_96

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        reshape_32 = paddle._C_ops.reshape(add_97, full_int_array_1)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        transpose_49 = paddle._C_ops.transpose(reshape_32, [2, 0, 3, 1, 4])
        del reshape_32

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_48 = paddle._C_ops.slice(
            transpose_49, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_49 = paddle._C_ops.slice(
            transpose_49, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_50 = paddle._C_ops.slice(
            transpose_49, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        transpose_50 = paddle._C_ops.transpose(slice_49, [0, 1, 3, 2])
        del slice_49

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        matmul_97 = paddle._C_ops.matmul(slice_48, transpose_50, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        scale_17 = paddle._C_ops.scale(matmul_97, full_1, float("0"), True)
        del matmul_97

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        softmax_16 = paddle._C_ops.softmax(scale_17, -1)
        del scale_17

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        matmul_98 = paddle._C_ops.matmul(softmax_16, slice_50, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        transpose_51 = paddle._C_ops.transpose(matmul_98, [0, 2, 1, 3])
        del matmul_98

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        reshape_33 = paddle._C_ops.reshape(transpose_51, full_int_array_6)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        matmul_99 = paddle._C_ops.matmul(reshape_33, parameter_95, False, False)
        del parameter_95

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_98 = paddle._C_ops.add(matmul_99, parameter_94)
        del parameter_94

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_99 = paddle._C_ops.add(add_96, add_98)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_102, layer_norm_103, layer_norm_104 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_99, parameter_93, parameter_92, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_92, parameter_93

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        matmul_100 = paddle._C_ops.matmul(layer_norm_102, parameter_91, False, False)
        del parameter_91

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        add_100 = paddle._C_ops.add(matmul_100, parameter_90)
        del parameter_90

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        gelu_16 = paddle._C_ops.gelu(add_100, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        matmul_101 = paddle._C_ops.matmul(gelu_16, parameter_89, False, False)
        del parameter_89

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_101 = paddle._C_ops.add(matmul_101, parameter_88)
        del parameter_88

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_102 = paddle._C_ops.add(add_99, add_101)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_105, layer_norm_106, layer_norm_107 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_102, parameter_87, parameter_86, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_86, parameter_87

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        matmul_102 = paddle._C_ops.matmul(layer_norm_105, parameter_85, False, False)
        del parameter_85

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        add_103 = paddle._C_ops.add(matmul_102, parameter_84)
        del parameter_84

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        reshape_34 = paddle._C_ops.reshape(add_103, full_int_array_1)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        transpose_52 = paddle._C_ops.transpose(reshape_34, [2, 0, 3, 1, 4])
        del reshape_34

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_51 = paddle._C_ops.slice(
            transpose_52, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_52 = paddle._C_ops.slice(
            transpose_52, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_53 = paddle._C_ops.slice(
            transpose_52, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        transpose_53 = paddle._C_ops.transpose(slice_52, [0, 1, 3, 2])
        del slice_52

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        matmul_103 = paddle._C_ops.matmul(slice_51, transpose_53, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        scale_18 = paddle._C_ops.scale(matmul_103, full_1, float("0"), True)
        del matmul_103

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        softmax_17 = paddle._C_ops.softmax(scale_18, -1)
        del scale_18

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        matmul_104 = paddle._C_ops.matmul(softmax_17, slice_53, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        transpose_54 = paddle._C_ops.transpose(matmul_104, [0, 2, 1, 3])
        del matmul_104

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        reshape_35 = paddle._C_ops.reshape(transpose_54, full_int_array_6)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        matmul_105 = paddle._C_ops.matmul(reshape_35, parameter_83, False, False)
        del parameter_83

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_104 = paddle._C_ops.add(matmul_105, parameter_82)
        del parameter_82

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_105 = paddle._C_ops.add(add_102, add_104)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_108, layer_norm_109, layer_norm_110 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_105, parameter_81, parameter_80, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_80, parameter_81

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        matmul_106 = paddle._C_ops.matmul(layer_norm_108, parameter_79, False, False)
        del parameter_79

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        add_106 = paddle._C_ops.add(matmul_106, parameter_78)
        del parameter_78

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        gelu_17 = paddle._C_ops.gelu(add_106, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        matmul_107 = paddle._C_ops.matmul(gelu_17, parameter_77, False, False)
        del parameter_77

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_107 = paddle._C_ops.add(matmul_107, parameter_76)
        del parameter_76

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_108 = paddle._C_ops.add(add_105, add_107)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_111, layer_norm_112, layer_norm_113 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_108, parameter_75, parameter_74, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_74, parameter_75

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        matmul_108 = paddle._C_ops.matmul(layer_norm_111, parameter_73, False, False)
        del parameter_73

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        add_109 = paddle._C_ops.add(matmul_108, parameter_72)
        del parameter_72

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        reshape_36 = paddle._C_ops.reshape(add_109, full_int_array_1)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        transpose_55 = paddle._C_ops.transpose(reshape_36, [2, 0, 3, 1, 4])
        del reshape_36

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_54 = paddle._C_ops.slice(
            transpose_55, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_55 = paddle._C_ops.slice(
            transpose_55, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_56 = paddle._C_ops.slice(
            transpose_55, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        transpose_56 = paddle._C_ops.transpose(slice_55, [0, 1, 3, 2])
        del slice_55

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        matmul_109 = paddle._C_ops.matmul(slice_54, transpose_56, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        scale_19 = paddle._C_ops.scale(matmul_109, full_1, float("0"), True)
        del matmul_109

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        softmax_18 = paddle._C_ops.softmax(scale_19, -1)
        del scale_19

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        matmul_110 = paddle._C_ops.matmul(softmax_18, slice_56, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        transpose_57 = paddle._C_ops.transpose(matmul_110, [0, 2, 1, 3])
        del matmul_110

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        reshape_37 = paddle._C_ops.reshape(transpose_57, full_int_array_6)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        matmul_111 = paddle._C_ops.matmul(reshape_37, parameter_71, False, False)
        del parameter_71

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_110 = paddle._C_ops.add(matmul_111, parameter_70)
        del parameter_70

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_111 = paddle._C_ops.add(add_108, add_110)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_114, layer_norm_115, layer_norm_116 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_111, parameter_69, parameter_68, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_68, parameter_69

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        matmul_112 = paddle._C_ops.matmul(layer_norm_114, parameter_67, False, False)
        del parameter_67

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        add_112 = paddle._C_ops.add(matmul_112, parameter_66)
        del parameter_66

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        gelu_18 = paddle._C_ops.gelu(add_112, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        matmul_113 = paddle._C_ops.matmul(gelu_18, parameter_65, False, False)
        del parameter_65

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_113 = paddle._C_ops.add(matmul_113, parameter_64)
        del parameter_64

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_114 = paddle._C_ops.add(add_111, add_113)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_117, layer_norm_118, layer_norm_119 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_114, parameter_63, parameter_62, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_62, parameter_63

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        matmul_114 = paddle._C_ops.matmul(layer_norm_117, parameter_61, False, False)
        del parameter_61

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        add_115 = paddle._C_ops.add(matmul_114, parameter_60)
        del parameter_60

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        reshape_38 = paddle._C_ops.reshape(add_115, full_int_array_1)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        transpose_58 = paddle._C_ops.transpose(reshape_38, [2, 0, 3, 1, 4])
        del reshape_38

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_57 = paddle._C_ops.slice(
            transpose_58, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_58 = paddle._C_ops.slice(
            transpose_58, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_59 = paddle._C_ops.slice(
            transpose_58, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        transpose_59 = paddle._C_ops.transpose(slice_58, [0, 1, 3, 2])
        del slice_58

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        matmul_115 = paddle._C_ops.matmul(slice_57, transpose_59, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        scale_20 = paddle._C_ops.scale(matmul_115, full_1, float("0"), True)
        del matmul_115

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        softmax_19 = paddle._C_ops.softmax(scale_20, -1)
        del scale_20

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        matmul_116 = paddle._C_ops.matmul(softmax_19, slice_59, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        transpose_60 = paddle._C_ops.transpose(matmul_116, [0, 2, 1, 3])
        del matmul_116

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        reshape_39 = paddle._C_ops.reshape(transpose_60, full_int_array_6)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        matmul_117 = paddle._C_ops.matmul(reshape_39, parameter_59, False, False)
        del parameter_59

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_116 = paddle._C_ops.add(matmul_117, parameter_58)
        del parameter_58

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_117 = paddle._C_ops.add(add_114, add_116)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_120, layer_norm_121, layer_norm_122 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_117, parameter_57, parameter_56, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_56, parameter_57

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        matmul_118 = paddle._C_ops.matmul(layer_norm_120, parameter_55, False, False)
        del parameter_55

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        add_118 = paddle._C_ops.add(matmul_118, parameter_54)
        del parameter_54

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        gelu_19 = paddle._C_ops.gelu(add_118, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        matmul_119 = paddle._C_ops.matmul(gelu_19, parameter_53, False, False)
        del parameter_53

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_119 = paddle._C_ops.add(matmul_119, parameter_52)
        del parameter_52

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_120 = paddle._C_ops.add(add_117, add_119)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_123, layer_norm_124, layer_norm_125 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_120, parameter_51, parameter_50, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_50, parameter_51

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        matmul_120 = paddle._C_ops.matmul(layer_norm_123, parameter_49, False, False)
        del parameter_49

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        add_121 = paddle._C_ops.add(matmul_120, parameter_48)
        del parameter_48

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        reshape_40 = paddle._C_ops.reshape(add_121, full_int_array_1)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        transpose_61 = paddle._C_ops.transpose(reshape_40, [2, 0, 3, 1, 4])
        del reshape_40

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_60 = paddle._C_ops.slice(
            transpose_61, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_61 = paddle._C_ops.slice(
            transpose_61, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_62 = paddle._C_ops.slice(
            transpose_61, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        transpose_62 = paddle._C_ops.transpose(slice_61, [0, 1, 3, 2])
        del slice_61

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        matmul_121 = paddle._C_ops.matmul(slice_60, transpose_62, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        scale_21 = paddle._C_ops.scale(matmul_121, full_1, float("0"), True)
        del matmul_121

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        softmax_20 = paddle._C_ops.softmax(scale_21, -1)
        del scale_21

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        matmul_122 = paddle._C_ops.matmul(softmax_20, slice_62, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        transpose_63 = paddle._C_ops.transpose(matmul_122, [0, 2, 1, 3])
        del matmul_122

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        reshape_41 = paddle._C_ops.reshape(transpose_63, full_int_array_6)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        matmul_123 = paddle._C_ops.matmul(reshape_41, parameter_47, False, False)
        del parameter_47

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_122 = paddle._C_ops.add(matmul_123, parameter_46)
        del parameter_46

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_123 = paddle._C_ops.add(add_120, add_122)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_126, layer_norm_127, layer_norm_128 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_123, parameter_45, parameter_44, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_44, parameter_45

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        matmul_124 = paddle._C_ops.matmul(layer_norm_126, parameter_43, False, False)
        del parameter_43

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        add_124 = paddle._C_ops.add(matmul_124, parameter_42)
        del parameter_42

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        gelu_20 = paddle._C_ops.gelu(add_124, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        matmul_125 = paddle._C_ops.matmul(gelu_20, parameter_41, False, False)
        del parameter_41

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_125 = paddle._C_ops.add(matmul_125, parameter_40)
        del parameter_40

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_126 = paddle._C_ops.add(add_123, add_125)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_129, layer_norm_130, layer_norm_131 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_126, parameter_39, parameter_38, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_38, parameter_39

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        matmul_126 = paddle._C_ops.matmul(layer_norm_129, parameter_37, False, False)
        del parameter_37

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        add_127 = paddle._C_ops.add(matmul_126, parameter_36)
        del parameter_36

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        reshape_42 = paddle._C_ops.reshape(add_127, full_int_array_1)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        transpose_64 = paddle._C_ops.transpose(reshape_42, [2, 0, 3, 1, 4])
        del reshape_42

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_63 = paddle._C_ops.slice(
            transpose_64, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_64 = paddle._C_ops.slice(
            transpose_64, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_65 = paddle._C_ops.slice(
            transpose_64, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        transpose_65 = paddle._C_ops.transpose(slice_64, [0, 1, 3, 2])
        del slice_64

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        matmul_127 = paddle._C_ops.matmul(slice_63, transpose_65, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        scale_22 = paddle._C_ops.scale(matmul_127, full_1, float("0"), True)
        del matmul_127

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        softmax_21 = paddle._C_ops.softmax(scale_22, -1)
        del scale_22

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        matmul_128 = paddle._C_ops.matmul(softmax_21, slice_65, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        transpose_66 = paddle._C_ops.transpose(matmul_128, [0, 2, 1, 3])
        del matmul_128

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        reshape_43 = paddle._C_ops.reshape(transpose_66, full_int_array_6)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        matmul_129 = paddle._C_ops.matmul(reshape_43, parameter_35, False, False)
        del parameter_35

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_128 = paddle._C_ops.add(matmul_129, parameter_34)
        del parameter_34

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_129 = paddle._C_ops.add(add_126, add_128)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_132, layer_norm_133, layer_norm_134 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_129, parameter_33, parameter_32, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_32, parameter_33

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        matmul_130 = paddle._C_ops.matmul(layer_norm_132, parameter_31, False, False)
        del parameter_31

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        add_130 = paddle._C_ops.add(matmul_130, parameter_30)
        del parameter_30

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        gelu_21 = paddle._C_ops.gelu(add_130, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        matmul_131 = paddle._C_ops.matmul(gelu_21, parameter_29, False, False)
        del parameter_29

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_131 = paddle._C_ops.add(matmul_131, parameter_28)
        del parameter_28

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_132 = paddle._C_ops.add(add_129, add_131)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_135, layer_norm_136, layer_norm_137 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_132, parameter_27, parameter_26, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_26, parameter_27

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        matmul_132 = paddle._C_ops.matmul(layer_norm_135, parameter_25, False, False)
        del parameter_25

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        add_133 = paddle._C_ops.add(matmul_132, parameter_24)
        del parameter_24

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        reshape_44 = paddle._C_ops.reshape(add_133, full_int_array_1)

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        transpose_67 = paddle._C_ops.transpose(reshape_44, [2, 0, 3, 1, 4])
        del reshape_44

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_66 = paddle._C_ops.slice(
            transpose_67, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_67 = paddle._C_ops.slice(
            transpose_67, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_68 = paddle._C_ops.slice(
            transpose_67, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        transpose_68 = paddle._C_ops.transpose(slice_67, [0, 1, 3, 2])
        del slice_67

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        matmul_133 = paddle._C_ops.matmul(slice_66, transpose_68, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        scale_23 = paddle._C_ops.scale(matmul_133, full_1, float("0"), True)
        del matmul_133

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        softmax_22 = paddle._C_ops.softmax(scale_23, -1)
        del scale_23

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        matmul_134 = paddle._C_ops.matmul(softmax_22, slice_68, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        transpose_69 = paddle._C_ops.transpose(matmul_134, [0, 2, 1, 3])
        del matmul_134

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        reshape_45 = paddle._C_ops.reshape(transpose_69, full_int_array_6)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        matmul_135 = paddle._C_ops.matmul(reshape_45, parameter_23, False, False)
        del parameter_23

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_134 = paddle._C_ops.add(matmul_135, parameter_22)
        del parameter_22

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_135 = paddle._C_ops.add(add_132, add_134)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_138, layer_norm_139, layer_norm_140 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_135, parameter_21, parameter_20, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_20, parameter_21

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        matmul_136 = paddle._C_ops.matmul(layer_norm_138, parameter_19, False, False)
        del parameter_19

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        add_136 = paddle._C_ops.add(matmul_136, parameter_18)
        del parameter_18

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        gelu_22 = paddle._C_ops.gelu(add_136, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        matmul_137 = paddle._C_ops.matmul(gelu_22, parameter_17, False, False)
        del parameter_17

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_137 = paddle._C_ops.add(matmul_137, parameter_16)
        del parameter_16

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_138 = paddle._C_ops.add(add_135, add_137)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_141, layer_norm_142, layer_norm_143 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_138, parameter_15, parameter_14, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_14, parameter_15

        # pd_op.matmul: (16x257x3072xf32) <- (16x257x1024xf32, 1024x3072xf32)
        matmul_138 = paddle._C_ops.matmul(layer_norm_141, parameter_13, False, False)
        del parameter_13

        # pd_op.add: (16x257x3072xf32) <- (16x257x3072xf32, 3072xf32)
        add_139 = paddle._C_ops.add(matmul_138, parameter_12)
        del parameter_12

        # pd_op.reshape: (16x257x3x16x64xf32) <- (16x257x3072xf32, 5xi64)
        reshape_46 = paddle._C_ops.reshape(add_139, full_int_array_1)
        del full_int_array_1

        # pd_op.transpose: (3x16x16x257x64xf32) <- (16x257x3x16x64xf32)
        transpose_70 = paddle._C_ops.transpose(reshape_46, [2, 0, 3, 1, 4])
        del reshape_46

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_69 = paddle._C_ops.slice(
            transpose_70, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_70 = paddle._C_ops.slice(
            transpose_70, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (16x16x257x64xf32) <- (3x16x16x257x64xf32, 1xi64, 1xi64)
        slice_71 = paddle._C_ops.slice(
            transpose_70, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.transpose: (16x16x64x257xf32) <- (16x16x257x64xf32)
        transpose_71 = paddle._C_ops.transpose(slice_70, [0, 1, 3, 2])
        del slice_70

        # pd_op.matmul: (16x16x257x257xf32) <- (16x16x257x64xf32, 16x16x64x257xf32)
        matmul_139 = paddle._C_ops.matmul(slice_69, transpose_71, False, False)

        # pd_op.scale: (16x16x257x257xf32) <- (16x16x257x257xf32, 1xf32)
        scale_24 = paddle._C_ops.scale(matmul_139, full_1, float("0"), True)
        del matmul_139

        # pd_op.softmax: (16x16x257x257xf32) <- (16x16x257x257xf32)
        softmax_23 = paddle._C_ops.softmax(scale_24, -1)
        del scale_24

        # pd_op.matmul: (16x16x257x64xf32) <- (16x16x257x257xf32, 16x16x257x64xf32)
        matmul_140 = paddle._C_ops.matmul(softmax_23, slice_71, False, False)

        # pd_op.transpose: (16x257x16x64xf32) <- (16x16x257x64xf32)
        transpose_72 = paddle._C_ops.transpose(matmul_140, [0, 2, 1, 3])
        del matmul_140

        # pd_op.reshape: (16x257x1024xf32) <- (16x257x16x64xf32, 3xi64)
        reshape_47 = paddle._C_ops.reshape(transpose_72, full_int_array_6)
        del full_int_array_6

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x1024xf32, 1024x1024xf32)
        matmul_141 = paddle._C_ops.matmul(reshape_47, parameter_11, False, False)
        del parameter_11

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_140 = paddle._C_ops.add(matmul_141, parameter_10)
        del parameter_10

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_141 = paddle._C_ops.add(add_138, add_140)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_144, layer_norm_145, layer_norm_146 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_141, parameter_9, parameter_8, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_8, parameter_9

        # pd_op.matmul: (16x257x4096xf32) <- (16x257x1024xf32, 1024x4096xf32)
        matmul_142 = paddle._C_ops.matmul(layer_norm_144, parameter_7, False, False)
        del parameter_7

        # pd_op.add: (16x257x4096xf32) <- (16x257x4096xf32, 4096xf32)
        add_142 = paddle._C_ops.add(matmul_142, parameter_6)
        del parameter_6

        # pd_op.gelu: (16x257x4096xf32) <- (16x257x4096xf32)
        gelu_23 = paddle._C_ops.gelu(add_142, False)

        # pd_op.matmul: (16x257x1024xf32) <- (16x257x4096xf32, 4096x1024xf32)
        matmul_143 = paddle._C_ops.matmul(gelu_23, parameter_5, False, False)
        del parameter_5

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 1024xf32)
        add_143 = paddle._C_ops.add(matmul_143, parameter_4)
        del parameter_4

        # pd_op.add: (16x257x1024xf32) <- (16x257x1024xf32, 16x257x1024xf32)
        add_144 = paddle._C_ops.add(add_141, add_143)

        # pd_op.layer_norm: (16x257x1024xf32, 16x257xf32, 16x257xf32) <- (16x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_147, layer_norm_148, layer_norm_149 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_144, parameter_3, parameter_2, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_2, parameter_3

        # pd_op.matmul: (16x257x512xf32) <- (16x257x1024xf32, 1024x512xf32)
        matmul_144 = paddle._C_ops.matmul(layer_norm_147, data_2, False, False)
        del data_2

        # pd_op.mean: (16x512xf32) <- (16x257x512xf32, 1xi64)
        mean_0 = paddle._C_ops.mean(matmul_144, full_int_array_3, False)

        # pd_op.matmul: (16x512xf32) <- (16x512xf32, 512x512xf32)
        matmul_145 = paddle._C_ops.matmul(mean_0, parameter_1, False, False)
        del parameter_1

        # pd_op.add: (16x512xf32) <- (16x512xf32, 512xf32)
        add_145 = paddle._C_ops.add(matmul_145, parameter_0)
        del parameter_0

        # pd_op.square: (16x512xf32) <- (16x512xf32)
        square_0 = paddle._C_ops.square(add_145)

        # pd_op.sum: (16x1xf32) <- (16x512xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(square_0, full_int_array_3, None, True)

        # pd_op.sqrt: (16x1xf32) <- (16x1xf32)
        sqrt_0 = paddle._C_ops.sqrt(sum_0)
        del sum_0

        # pd_op.divide: (16x512xf32) <- (16x512xf32, 16x1xf32)
        divide_0 = paddle._C_ops.divide(add_145, sqrt_0)

        # pd_op.square: (512x159xf32) <- (512x159xf32)
        square_1 = paddle._C_ops.square(data_3)

        # pd_op.sum: (1x159xf32) <- (512x159xf32, 1xi64)
        sum_1 = paddle._C_ops.sum(square_1, full_int_array_2, None, True)

        # pd_op.sqrt: (1x159xf32) <- (1x159xf32)
        sqrt_1 = paddle._C_ops.sqrt(sum_1)
        del sum_1

        # pd_op.divide: (512x159xf32) <- (512x159xf32, 1x159xf32)
        divide_1 = paddle._C_ops.divide(data_3, sqrt_1)
        del data_3

        # pd_op.matmul: (16x159xf32) <- (16x512xf32, 512x159xf32)
        matmul_146 = paddle._C_ops.matmul(divide_0, divide_1, False, False)

        # pd_op.square: (16x159xf32) <- (16x159xf32)
        square_2 = paddle._C_ops.square(matmul_146)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (16x159xf32) <- (16x159xf32, 1xf32)
        scale_25 = paddle._C_ops.scale(square_2, full_2, float("1"), True)
        del square_2

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_166 = full_3

        # pd_op.scale: (16x159xf32) <- (16x159xf32, 1xf32)
        scale_26 = paddle._C_ops.scale(scale_25, full_3, float("1e-06"), True)
        del scale_25

        # pd_op.sqrt: (16x159xf32) <- (16x159xf32)
        sqrt_2 = paddle._C_ops.sqrt(scale_26)
        del scale_26

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0.980067"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (16x159xf32) <- (16x159xf32, 1xf32)
        scale_27 = paddle._C_ops.scale(matmul_146, full_4, float("0"), True)

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0.198669"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (16x159xf32) <- (16x159xf32, 1xf32)
        scale_28 = paddle._C_ops.scale(sqrt_2, full_5, float("0"), True)

        # pd_op.subtract: (16x159xf32) <- (16x159xf32, 16x159xf32)
        subtract_0 = paddle._C_ops.subtract(scale_27, scale_28)

        # pd_op.scale: (16x159xf32) <- (16x159xf32, 1xf32)
        scale_29 = paddle._C_ops.scale(matmul_146, full_3, float("-0.0397339"), True)

        # pd_op.full: (xf32) <- ()
        full_6 = paddle._C_ops.full(
            [],
            float("-0.980067"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.greater_than: (16x159xb) <- (16x159xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(matmul_146, full_6)
        del full_6

        # pd_op.cast: (16x159xf32) <- (16x159xb)
        cast_0 = paddle._C_ops.cast(greater_than_0, paddle.float32)
        del greater_than_0

        # pd_op.multiply: (16x159xf32) <- (16x159xf32, 16x159xf32)
        multiply_0 = paddle._C_ops.multiply(cast_0, subtract_0)

        # pd_op.scale: (16x159xf32) <- (16x159xf32, 1xf32)
        scale_30 = paddle._C_ops.scale(cast_0, full_2, float("1"), True)

        # pd_op.multiply: (16x159xf32) <- (16x159xf32, 16x159xf32)
        multiply_1 = paddle._C_ops.multiply(scale_30, scale_29)

        # pd_op.add: (16x159xf32) <- (16x159xf32, 16x159xf32)
        add_146 = paddle._C_ops.add(multiply_0, multiply_1)

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("159"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.one_hot: (16x1x159xf32) <- (16x1xi64, 1xi32)
        one_hot_0 = paddle._C_ops.one_hot(
            data_5 % paddle.cast(full_7, data_5.dtype), full_7
        )
        del data_5, full_7

        # pd_op.squeeze: (16x159xf32) <- (16x1x159xf32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(one_hot_0, full_int_array_3)
        del one_hot_0

        # pd_op.multiply: (16x159xf32) <- (16x159xf32, 16x159xf32)
        multiply_2 = paddle._C_ops.multiply(squeeze_0, add_146)

        # pd_op.scale: (16x159xf32) <- (16x159xf32, 1xf32)
        scale_31 = paddle._C_ops.scale(squeeze_0, full_2, float("1"), True)

        # pd_op.multiply: (16x159xf32) <- (16x159xf32, 16x159xf32)
        multiply_3 = paddle._C_ops.multiply(scale_31, matmul_146)

        # pd_op.add: (16x159xf32) <- (16x159xf32, 16x159xf32)
        add_147 = paddle._C_ops.add(multiply_2, multiply_3)

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("30"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (16x159xf32) <- (16x159xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(add_147, full_8, float("0"), True)
        del (
            add_0,
            add_1,
            add_10,
            add_100,
            add_101,
            add_102,
            add_103,
            add_104,
            add_105,
            add_106,
            add_107,
            add_108,
            add_109,
            add_11,
            add_110,
            add_111,
            add_112,
            add_113,
            add_114,
            add_115,
            add_116,
            add_117,
            add_118,
            add_119,
            add_12,
            add_120,
            add_121,
            add_122,
            add_123,
            add_124,
            add_125,
            add_126,
            add_127,
            add_128,
            add_129,
            add_13,
            add_130,
            add_131,
            add_132,
            add_133,
            add_134,
            add_135,
            add_136,
            add_137,
            add_138,
            add_139,
            add_14,
            add_140,
            add_141,
            add_142,
            add_143,
            add_144,
            add_145,
            add_146,
            add_147,
            add_15,
            add_16,
            add_17,
            add_18,
            add_19,
            add_2,
            add_20,
            add_21,
            add_22,
            add_23,
            add_24,
            add_25,
            add_26,
            add_27,
            add_28,
            add_29,
            add_3,
            add_30,
            add_31,
            add_32,
            add_33,
            add_34,
            add_35,
            add_36,
            add_37,
            add_38,
            add_39,
            add_4,
            add_40,
            add_41,
            add_42,
            add_43,
            add_44,
            add_45,
            add_46,
            add_47,
            add_48,
            add_49,
            add_5,
            add_50,
            add_51,
            add_52,
            add_53,
            add_54,
            add_55,
            add_56,
            add_57,
            add_58,
            add_59,
            add_6,
            add_60,
            add_61,
            add_62,
            add_63,
            add_64,
            add_65,
            add_66,
            add_67,
            add_68,
            add_69,
            add_7,
            add_70,
            add_71,
            add_72,
            add_73,
            add_74,
            add_75,
            add_76,
            add_77,
            add_78,
            add_79,
            add_8,
            add_80,
            add_81,
            add_82,
            add_83,
            add_84,
            add_85,
            add_86,
            add_87,
            add_88,
            add_89,
            add_9,
            add_90,
            add_91,
            add_92,
            add_93,
            add_94,
            add_95,
            add_96,
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
            assign_17,
            assign_18,
            assign_19,
            assign_2,
            assign_20,
            assign_21,
            assign_22,
            assign_23,
            assign_24,
            assign_25,
            assign_26,
            assign_27,
            assign_28,
            assign_29,
            assign_3,
            assign_30,
            assign_31,
            assign_32,
            assign_33,
            assign_34,
            assign_35,
            assign_36,
            assign_37,
            assign_38,
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
            cast_0,
            concat_0,
            conv2d_0,
            divide_0,
            divide_1,
            expand_0,
            full_0,
            full_1,
            full_2,
            full_3,
            full_4,
            full_5,
            full_8,
            full_int_array_0,
            full_int_array_2,
            full_int_array_3,
            full_int_array_4,
            full_int_array_5,
            gelu_0,
            gelu_1,
            gelu_10,
            gelu_11,
            gelu_12,
            gelu_13,
            gelu_14,
            gelu_15,
            gelu_16,
            gelu_17,
            gelu_18,
            gelu_19,
            gelu_2,
            gelu_20,
            gelu_21,
            gelu_22,
            gelu_23,
            gelu_3,
            gelu_4,
            gelu_5,
            gelu_6,
            gelu_7,
            gelu_8,
            gelu_9,
            layer_norm_0,
            layer_norm_1,
            layer_norm_10,
            layer_norm_100,
            layer_norm_101,
            layer_norm_102,
            layer_norm_103,
            layer_norm_104,
            layer_norm_105,
            layer_norm_106,
            layer_norm_107,
            layer_norm_108,
            layer_norm_109,
            layer_norm_11,
            layer_norm_110,
            layer_norm_111,
            layer_norm_112,
            layer_norm_113,
            layer_norm_114,
            layer_norm_115,
            layer_norm_116,
            layer_norm_117,
            layer_norm_118,
            layer_norm_119,
            layer_norm_12,
            layer_norm_120,
            layer_norm_121,
            layer_norm_122,
            layer_norm_123,
            layer_norm_124,
            layer_norm_125,
            layer_norm_126,
            layer_norm_127,
            layer_norm_128,
            layer_norm_129,
            layer_norm_13,
            layer_norm_130,
            layer_norm_131,
            layer_norm_132,
            layer_norm_133,
            layer_norm_134,
            layer_norm_135,
            layer_norm_136,
            layer_norm_137,
            layer_norm_138,
            layer_norm_139,
            layer_norm_14,
            layer_norm_140,
            layer_norm_141,
            layer_norm_142,
            layer_norm_143,
            layer_norm_144,
            layer_norm_145,
            layer_norm_146,
            layer_norm_147,
            layer_norm_148,
            layer_norm_149,
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
            layer_norm_75,
            layer_norm_76,
            layer_norm_77,
            layer_norm_78,
            layer_norm_79,
            layer_norm_8,
            layer_norm_80,
            layer_norm_81,
            layer_norm_82,
            layer_norm_83,
            layer_norm_84,
            layer_norm_85,
            layer_norm_86,
            layer_norm_87,
            layer_norm_88,
            layer_norm_89,
            layer_norm_9,
            layer_norm_90,
            layer_norm_91,
            layer_norm_92,
            layer_norm_93,
            layer_norm_94,
            layer_norm_95,
            layer_norm_96,
            layer_norm_97,
            layer_norm_98,
            layer_norm_99,
            matmul_0,
            matmul_10,
            matmul_100,
            matmul_101,
            matmul_102,
            matmul_105,
            matmul_106,
            matmul_107,
            matmul_108,
            matmul_11,
            matmul_111,
            matmul_112,
            matmul_113,
            matmul_114,
            matmul_117,
            matmul_118,
            matmul_119,
            matmul_12,
            matmul_120,
            matmul_123,
            matmul_124,
            matmul_125,
            matmul_126,
            matmul_129,
            matmul_130,
            matmul_131,
            matmul_132,
            matmul_135,
            matmul_136,
            matmul_137,
            matmul_138,
            matmul_141,
            matmul_142,
            matmul_143,
            matmul_144,
            matmul_145,
            matmul_146,
            matmul_15,
            matmul_16,
            matmul_17,
            matmul_18,
            matmul_21,
            matmul_22,
            matmul_23,
            matmul_24,
            matmul_27,
            matmul_28,
            matmul_29,
            matmul_3,
            matmul_30,
            matmul_33,
            matmul_34,
            matmul_35,
            matmul_36,
            matmul_39,
            matmul_4,
            matmul_40,
            matmul_41,
            matmul_42,
            matmul_45,
            matmul_46,
            matmul_47,
            matmul_48,
            matmul_5,
            matmul_51,
            matmul_52,
            matmul_53,
            matmul_54,
            matmul_57,
            matmul_58,
            matmul_59,
            matmul_6,
            matmul_60,
            matmul_63,
            matmul_64,
            matmul_65,
            matmul_66,
            matmul_69,
            matmul_70,
            matmul_71,
            matmul_72,
            matmul_75,
            matmul_76,
            matmul_77,
            matmul_78,
            matmul_81,
            matmul_82,
            matmul_83,
            matmul_84,
            matmul_87,
            matmul_88,
            matmul_89,
            matmul_9,
            matmul_90,
            matmul_93,
            matmul_94,
            matmul_95,
            matmul_96,
            matmul_99,
            mean_0,
            multiply_0,
            multiply_1,
            multiply_2,
            multiply_3,
            reshape_1,
            reshape_11,
            reshape_13,
            reshape_15,
            reshape_17,
            reshape_19,
            reshape_21,
            reshape_23,
            reshape_25,
            reshape_27,
            reshape_29,
            reshape_3,
            reshape_31,
            reshape_33,
            reshape_35,
            reshape_37,
            reshape_39,
            reshape_41,
            reshape_43,
            reshape_45,
            reshape_47,
            reshape_5,
            reshape_7,
            reshape_9,
            scale_27,
            scale_28,
            scale_29,
            scale_30,
            scale_31,
            slice_0,
            slice_11,
            slice_12,
            slice_14,
            slice_15,
            slice_17,
            slice_18,
            slice_2,
            slice_20,
            slice_21,
            slice_23,
            slice_24,
            slice_26,
            slice_27,
            slice_29,
            slice_3,
            slice_30,
            slice_32,
            slice_33,
            slice_35,
            slice_36,
            slice_38,
            slice_39,
            slice_41,
            slice_42,
            slice_44,
            slice_45,
            slice_47,
            slice_48,
            slice_5,
            slice_50,
            slice_51,
            slice_53,
            slice_54,
            slice_56,
            slice_57,
            slice_59,
            slice_6,
            slice_60,
            slice_62,
            slice_63,
            slice_65,
            slice_66,
            slice_68,
            slice_69,
            slice_71,
            slice_8,
            slice_9,
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
            sqrt_0,
            sqrt_1,
            sqrt_2,
            square_0,
            square_1,
            squeeze_0,
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
            transpose_48,
            transpose_49,
            transpose_5,
            transpose_50,
            transpose_51,
            transpose_52,
            transpose_53,
            transpose_54,
            transpose_55,
            transpose_56,
            transpose_57,
            transpose_58,
            transpose_59,
            transpose_6,
            transpose_60,
            transpose_61,
            transpose_62,
            transpose_63,
            transpose_64,
            transpose_65,
            transpose_66,
            transpose_67,
            transpose_68,
            transpose_69,
            transpose_7,
            transpose_70,
            transpose_71,
            transpose_72,
            transpose_8,
            transpose_9,
        )

        return scale_0
