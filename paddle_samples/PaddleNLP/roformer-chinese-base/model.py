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
        data_0,
        data_1,
    ):
        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.equal: (1x8xb) <- (1x8xi64, xi64)
        equal_0 = paddle._C_ops.equal(data_0, full_0)
        del full_0

        # pd_op.cast: (1x8xf32) <- (1x8xb)
        cast_0 = paddle._C_ops.cast(equal_0, paddle.float32)
        del equal_0

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("-10000"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x8xf32) <- (1x8xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(cast_0, full_1, float("0"), True)
        del cast_0, full_1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 2]

        # pd_op.unsqueeze: (1x1x1x8xf32) <- (1x8xf32, 2xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(scale_0, full_int_array_0)
        del full_int_array_0, scale_0

        # pd_op.embedding: (1x8x768xf32) <- (1x8xi64, 50000x768xf32)
        embedding_0 = paddle._C_ops.embedding(data_0, parameter_221, -1, False)
        del data_0, parameter_221

        # pd_op.embedding: (1x8x768xf32) <- (1x8xi64, 2x768xf32)
        embedding_1 = paddle._C_ops.embedding(data_1, parameter_220, -1, False)
        del data_1, parameter_220

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 1x8x768xf32)
        add_0 = paddle._C_ops.add(embedding_0, embedding_1)

        # pd_op.layer_norm: (1x8x768xf32, 1x8xf32, 1x8xf32) <- (1x8x768xf32, 768xf32, 768xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_0, parameter_219, parameter_218, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_218, parameter_219

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_0 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_1 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_2 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_3 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_4 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_5 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_6 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_7 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_8 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_9 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_10 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_11 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_12 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_13 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_14 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_15 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_16 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_17 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_18 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_19 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_20 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_21 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_22 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_23 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_24 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_25 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_26 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_27 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_28 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_29 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_30 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_31 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_32 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_33 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_34 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_35 = full_2

        # pd_op.dropout: (1x8x768xf32, 1x8x768xui8) <- (1x8x768xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                layer_norm_0, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del layer_norm_0

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_0 = paddle._C_ops.matmul(dropout_0, parameter_217, False, False)
        del parameter_217

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_1 = paddle._C_ops.add(matmul_0, parameter_216)
        del parameter_216

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [0, 0, 12, 64]

        # pd_op.reshape: (1x8x12x64xf32) <- (1x8x768xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(add_1, full_int_array_1)

        # pd_op.transpose: (1x12x8x64xf32) <- (1x8x12x64xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_0, [0, 2, 1, 3])
        del reshape_0

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_1 = paddle._C_ops.matmul(dropout_0, parameter_215, False, False)
        del parameter_215

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_2 = paddle._C_ops.add(matmul_1, parameter_214)
        del parameter_214

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_2 = paddle._C_ops.matmul(dropout_0, parameter_213, False, False)
        del parameter_213

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_3 = paddle._C_ops.add(matmul_2, parameter_212)
        del parameter_212

        # pd_op.reshape: (1x8x12x64xf32) <- (1x8x768xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(add_2, full_int_array_1)

        # pd_op.transpose: (1x12x8x64xf32) <- (1x8x12x64xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_1, [0, 2, 1, 3])
        del reshape_1

        # pd_op.reshape: (1x8x12x64xf32) <- (1x8x768xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(add_3, full_int_array_1)

        # pd_op.transpose: (1x12x8x64xf32) <- (1x8x12x64xf32)
        transpose_2 = paddle._C_ops.transpose(reshape_2, [0, 2, 1, 3])
        del reshape_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_36 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_37 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_38 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_39 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_40 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_41 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_42 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_43 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_44 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_45 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_46 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_47 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_48 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_49 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_50 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_51 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_52 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_53 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_54 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_55 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_56 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_57 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_58 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_59 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_60 = full_int_array_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [8]

        # pd_op.slice: (8x32xf32) <- (1536x32xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            parameter_23, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_23

        # pd_op.assign: (8x32xf32) <- (8x32xf32)
        assign_61 = slice_0

        # pd_op.slice: (8x32xf32) <- (1536x32xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            parameter_22, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_22

        # pd_op.assign: (8x32xf32) <- (8x32xf32)
        assign_62 = slice_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [2147483647]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_63 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_64 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_65 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_66 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_67 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_68 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_69 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_70 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_71 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_72 = full_int_array_4

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

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_110 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_111 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_112 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_113 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_114 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_115 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_116 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_117 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_118 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_119 = full_int_array_5

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

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_143 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_144 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_145 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_146 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_147 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_148 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_149 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_150 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_151 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_152 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_153 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_154 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_155 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_156 = full_int_array_5

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_0 = paddle._C_ops.strided_slice(
            transpose_0, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_157 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_158 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_159 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_160 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_161 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_162 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_163 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_164 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_165 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_166 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_167 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_168 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_169 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_170 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_171 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_172 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_173 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_174 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_175 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_176 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_177 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_178 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_179 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_180 = full_int_array_6

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_1 = paddle._C_ops.strided_slice(
            transpose_0, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_0 = paddle._C_ops.multiply(strided_slice_0, slice_1)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_1 = paddle._C_ops.multiply(strided_slice_1, slice_0)

        # pd_op.subtract: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        subtract_0 = paddle._C_ops.subtract(multiply_0, multiply_1)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_2 = paddle._C_ops.multiply(strided_slice_0, slice_0)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_3 = paddle._C_ops.multiply(strided_slice_1, slice_1)

        # pd_op.add: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        add_4 = paddle._C_ops.add(multiply_2, multiply_3)

        # builtin.combine: ([1x12x8x32xf32, 1x12x8x32xf32]) <- (1x12x8x32xf32, 1x12x8x32xf32)
        combine_0 = [subtract_0, add_4]

        # pd_op.stack: (1x12x8x32x2xf32) <- ([1x12x8x32xf32, 1x12x8x32xf32])
        stack_0 = paddle._C_ops.stack(combine_0, -1)
        del combine_0

        # pd_op.flatten: (1x12x8x64xf32) <- (1x12x8x32x2xf32)
        flatten_0 = paddle._C_ops.flatten(stack_0, 3, 4)

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_2 = paddle._C_ops.strided_slice(
            transpose_1, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_3 = paddle._C_ops.strided_slice(
            transpose_1, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_4 = paddle._C_ops.multiply(strided_slice_2, slice_1)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_5 = paddle._C_ops.multiply(strided_slice_3, slice_0)

        # pd_op.subtract: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        subtract_1 = paddle._C_ops.subtract(multiply_4, multiply_5)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_6 = paddle._C_ops.multiply(strided_slice_2, slice_0)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_7 = paddle._C_ops.multiply(strided_slice_3, slice_1)

        # pd_op.add: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        add_5 = paddle._C_ops.add(multiply_6, multiply_7)

        # builtin.combine: ([1x12x8x32xf32, 1x12x8x32xf32]) <- (1x12x8x32xf32, 1x12x8x32xf32)
        combine_1 = [subtract_1, add_5]

        # pd_op.stack: (1x12x8x32x2xf32) <- ([1x12x8x32xf32, 1x12x8x32xf32])
        stack_1 = paddle._C_ops.stack(combine_1, -1)
        del combine_1

        # pd_op.flatten: (1x12x8x64xf32) <- (1x12x8x32x2xf32)
        flatten_1 = paddle._C_ops.flatten(stack_1, 3, 4)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_181 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_182 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_183 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_184 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_185 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_186 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_187 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_188 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_189 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_190 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_191 = full_3

        # pd_op.scale: (1x12x8x64xf32) <- (1x12x8x64xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(flatten_0, full_3, float("0"), True)
        del flatten_0

        # pd_op.matmul: (1x12x8x8xf32) <- (1x12x8x64xf32, 1x12x8x64xf32)
        matmul_3 = paddle._C_ops.matmul(scale_1, flatten_1, False, True)

        # pd_op.add: (1x12x8x8xf32) <- (1x12x8x8xf32, 1x1x1x8xf32)
        add_6 = paddle._C_ops.add(matmul_3, unsqueeze_0)

        # pd_op.softmax: (1x12x8x8xf32) <- (1x12x8x8xf32)
        softmax_0 = paddle._C_ops.softmax(add_6, -1)
        del add_6

        # pd_op.dropout: (1x12x8x8xf32, 1x12x8x8xui8) <- (1x12x8x8xf32, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_0, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x8x64xf32) <- (1x12x8x8xf32, 1x12x8x64xf32)
        matmul_4 = paddle._C_ops.matmul(dropout_2, transpose_2, False, False)

        # pd_op.transpose: (1x8x12x64xf32) <- (1x12x8x64xf32)
        transpose_3 = paddle._C_ops.transpose(matmul_4, [0, 2, 1, 3])
        del matmul_4

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_7 = [0, 0, 768]

        # pd_op.reshape: (1x8x768xf32) <- (1x8x12x64xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_3, full_int_array_7)

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_5 = paddle._C_ops.matmul(reshape_3, parameter_211, False, False)
        del parameter_211

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_7 = paddle._C_ops.add(matmul_5, parameter_210)
        del parameter_210

        # pd_op.dropout: (1x8x768xf32, 1x8x768xui8) <- (1x8x768xf32, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_7, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_7

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 1x8x768xf32)
        add_8 = paddle._C_ops.add(dropout_0, dropout_4)

        # pd_op.layer_norm: (1x8x768xf32, 1x8xf32, 1x8xf32) <- (1x8x768xf32, 768xf32, 768xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_8, parameter_205, parameter_204, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_204, parameter_205

        # pd_op.matmul: (1x8x3072xf32) <- (1x8x768xf32, 768x3072xf32)
        matmul_6 = paddle._C_ops.matmul(layer_norm_3, parameter_209, False, False)
        del parameter_209

        # pd_op.add: (1x8x3072xf32) <- (1x8x3072xf32, 3072xf32)
        add_9 = paddle._C_ops.add(matmul_6, parameter_208)
        del parameter_208

        # pd_op.gelu: (1x8x3072xf32) <- (1x8x3072xf32)
        gelu_0 = paddle._C_ops.gelu(add_9, False)

        # pd_op.matmul: (1x8x768xf32) <- (1x8x3072xf32, 3072x768xf32)
        matmul_7 = paddle._C_ops.matmul(gelu_0, parameter_207, False, False)
        del parameter_207

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_10 = paddle._C_ops.add(matmul_7, parameter_206)
        del parameter_206

        # pd_op.dropout: (1x8x768xf32, 1x8x768xui8) <- (1x8x768xf32, None, 1xf32)
        dropout_6, dropout_7 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_10, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_10

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 1x8x768xf32)
        add_11 = paddle._C_ops.add(layer_norm_3, dropout_6)

        # pd_op.layer_norm: (1x8x768xf32, 1x8xf32, 1x8xf32) <- (1x8x768xf32, 768xf32, 768xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_11, parameter_203, parameter_202, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_202, parameter_203

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_8 = paddle._C_ops.matmul(layer_norm_6, parameter_201, False, False)
        del parameter_201

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_12 = paddle._C_ops.add(matmul_8, parameter_200)
        del parameter_200

        # pd_op.reshape: (1x8x12x64xf32) <- (1x8x768xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(add_12, full_int_array_1)

        # pd_op.transpose: (1x12x8x64xf32) <- (1x8x12x64xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_4, [0, 2, 1, 3])
        del reshape_4

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_9 = paddle._C_ops.matmul(layer_norm_6, parameter_199, False, False)
        del parameter_199

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_13 = paddle._C_ops.add(matmul_9, parameter_198)
        del parameter_198

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_6, parameter_197, False, False)
        del parameter_197

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_14 = paddle._C_ops.add(matmul_10, parameter_196)
        del parameter_196

        # pd_op.reshape: (1x8x12x64xf32) <- (1x8x768xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(add_13, full_int_array_1)

        # pd_op.transpose: (1x12x8x64xf32) <- (1x8x12x64xf32)
        transpose_5 = paddle._C_ops.transpose(reshape_5, [0, 2, 1, 3])
        del reshape_5

        # pd_op.reshape: (1x8x12x64xf32) <- (1x8x768xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(add_14, full_int_array_1)

        # pd_op.transpose: (1x12x8x64xf32) <- (1x8x12x64xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_6, [0, 2, 1, 3])
        del reshape_6

        # pd_op.slice: (8x32xf32) <- (1536x32xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            parameter_21, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_21

        # pd_op.assign: (8x32xf32) <- (8x32xf32)
        assign_192 = slice_2

        # pd_op.slice: (8x32xf32) <- (1536x32xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            parameter_20, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_20

        # pd_op.assign: (8x32xf32) <- (8x32xf32)
        assign_193 = slice_3

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_4 = paddle._C_ops.strided_slice(
            transpose_4, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_5 = paddle._C_ops.strided_slice(
            transpose_4, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_8 = paddle._C_ops.multiply(strided_slice_4, slice_3)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_9 = paddle._C_ops.multiply(strided_slice_5, slice_2)

        # pd_op.subtract: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        subtract_2 = paddle._C_ops.subtract(multiply_8, multiply_9)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_10 = paddle._C_ops.multiply(strided_slice_4, slice_2)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_11 = paddle._C_ops.multiply(strided_slice_5, slice_3)

        # pd_op.add: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        add_15 = paddle._C_ops.add(multiply_10, multiply_11)

        # builtin.combine: ([1x12x8x32xf32, 1x12x8x32xf32]) <- (1x12x8x32xf32, 1x12x8x32xf32)
        combine_2 = [subtract_2, add_15]

        # pd_op.stack: (1x12x8x32x2xf32) <- ([1x12x8x32xf32, 1x12x8x32xf32])
        stack_2 = paddle._C_ops.stack(combine_2, -1)
        del combine_2

        # pd_op.flatten: (1x12x8x64xf32) <- (1x12x8x32x2xf32)
        flatten_2 = paddle._C_ops.flatten(stack_2, 3, 4)

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_6 = paddle._C_ops.strided_slice(
            transpose_5, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_7 = paddle._C_ops.strided_slice(
            transpose_5, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_12 = paddle._C_ops.multiply(strided_slice_6, slice_3)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_13 = paddle._C_ops.multiply(strided_slice_7, slice_2)

        # pd_op.subtract: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        subtract_3 = paddle._C_ops.subtract(multiply_12, multiply_13)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_14 = paddle._C_ops.multiply(strided_slice_6, slice_2)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_15 = paddle._C_ops.multiply(strided_slice_7, slice_3)

        # pd_op.add: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        add_16 = paddle._C_ops.add(multiply_14, multiply_15)

        # builtin.combine: ([1x12x8x32xf32, 1x12x8x32xf32]) <- (1x12x8x32xf32, 1x12x8x32xf32)
        combine_3 = [subtract_3, add_16]

        # pd_op.stack: (1x12x8x32x2xf32) <- ([1x12x8x32xf32, 1x12x8x32xf32])
        stack_3 = paddle._C_ops.stack(combine_3, -1)
        del combine_3

        # pd_op.flatten: (1x12x8x64xf32) <- (1x12x8x32x2xf32)
        flatten_3 = paddle._C_ops.flatten(stack_3, 3, 4)

        # pd_op.scale: (1x12x8x64xf32) <- (1x12x8x64xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(flatten_2, full_3, float("0"), True)
        del flatten_2

        # pd_op.matmul: (1x12x8x8xf32) <- (1x12x8x64xf32, 1x12x8x64xf32)
        matmul_11 = paddle._C_ops.matmul(scale_2, flatten_3, False, True)

        # pd_op.add: (1x12x8x8xf32) <- (1x12x8x8xf32, 1x1x1x8xf32)
        add_17 = paddle._C_ops.add(matmul_11, unsqueeze_0)

        # pd_op.softmax: (1x12x8x8xf32) <- (1x12x8x8xf32)
        softmax_1 = paddle._C_ops.softmax(add_17, -1)
        del add_17

        # pd_op.dropout: (1x12x8x8xf32, 1x12x8x8xui8) <- (1x12x8x8xf32, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_1, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x8x64xf32) <- (1x12x8x8xf32, 1x12x8x64xf32)
        matmul_12 = paddle._C_ops.matmul(dropout_8, transpose_6, False, False)

        # pd_op.transpose: (1x8x12x64xf32) <- (1x12x8x64xf32)
        transpose_7 = paddle._C_ops.transpose(matmul_12, [0, 2, 1, 3])
        del matmul_12

        # pd_op.reshape: (1x8x768xf32) <- (1x8x12x64xf32, 3xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_7, full_int_array_7)

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_13 = paddle._C_ops.matmul(reshape_7, parameter_195, False, False)
        del parameter_195

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_18 = paddle._C_ops.add(matmul_13, parameter_194)
        del parameter_194

        # pd_op.dropout: (1x8x768xf32, 1x8x768xui8) <- (1x8x768xf32, None, 1xf32)
        dropout_10, dropout_11 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_18, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_18

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 1x8x768xf32)
        add_19 = paddle._C_ops.add(layer_norm_6, dropout_10)

        # pd_op.layer_norm: (1x8x768xf32, 1x8xf32, 1x8xf32) <- (1x8x768xf32, 768xf32, 768xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_19, parameter_189, parameter_188, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_188, parameter_189

        # pd_op.matmul: (1x8x3072xf32) <- (1x8x768xf32, 768x3072xf32)
        matmul_14 = paddle._C_ops.matmul(layer_norm_9, parameter_193, False, False)
        del parameter_193

        # pd_op.add: (1x8x3072xf32) <- (1x8x3072xf32, 3072xf32)
        add_20 = paddle._C_ops.add(matmul_14, parameter_192)
        del parameter_192

        # pd_op.gelu: (1x8x3072xf32) <- (1x8x3072xf32)
        gelu_1 = paddle._C_ops.gelu(add_20, False)

        # pd_op.matmul: (1x8x768xf32) <- (1x8x3072xf32, 3072x768xf32)
        matmul_15 = paddle._C_ops.matmul(gelu_1, parameter_191, False, False)
        del parameter_191

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_21 = paddle._C_ops.add(matmul_15, parameter_190)
        del parameter_190

        # pd_op.dropout: (1x8x768xf32, 1x8x768xui8) <- (1x8x768xf32, None, 1xf32)
        dropout_12, dropout_13 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_21, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_21

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 1x8x768xf32)
        add_22 = paddle._C_ops.add(layer_norm_9, dropout_12)

        # pd_op.layer_norm: (1x8x768xf32, 1x8xf32, 1x8xf32) <- (1x8x768xf32, 768xf32, 768xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_22, parameter_187, parameter_186, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_186, parameter_187

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_16 = paddle._C_ops.matmul(layer_norm_12, parameter_185, False, False)
        del parameter_185

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_23 = paddle._C_ops.add(matmul_16, parameter_184)
        del parameter_184

        # pd_op.reshape: (1x8x12x64xf32) <- (1x8x768xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(add_23, full_int_array_1)

        # pd_op.transpose: (1x12x8x64xf32) <- (1x8x12x64xf32)
        transpose_8 = paddle._C_ops.transpose(reshape_8, [0, 2, 1, 3])
        del reshape_8

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_17 = paddle._C_ops.matmul(layer_norm_12, parameter_183, False, False)
        del parameter_183

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_24 = paddle._C_ops.add(matmul_17, parameter_182)
        del parameter_182

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_18 = paddle._C_ops.matmul(layer_norm_12, parameter_181, False, False)
        del parameter_181

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_25 = paddle._C_ops.add(matmul_18, parameter_180)
        del parameter_180

        # pd_op.reshape: (1x8x12x64xf32) <- (1x8x768xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(add_24, full_int_array_1)

        # pd_op.transpose: (1x12x8x64xf32) <- (1x8x12x64xf32)
        transpose_9 = paddle._C_ops.transpose(reshape_9, [0, 2, 1, 3])
        del reshape_9

        # pd_op.reshape: (1x8x12x64xf32) <- (1x8x768xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(add_25, full_int_array_1)

        # pd_op.transpose: (1x12x8x64xf32) <- (1x8x12x64xf32)
        transpose_10 = paddle._C_ops.transpose(reshape_10, [0, 2, 1, 3])
        del reshape_10

        # pd_op.slice: (8x32xf32) <- (1536x32xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            parameter_19, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_19

        # pd_op.assign: (8x32xf32) <- (8x32xf32)
        assign_194 = slice_4

        # pd_op.slice: (8x32xf32) <- (1536x32xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            parameter_18, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_18

        # pd_op.assign: (8x32xf32) <- (8x32xf32)
        assign_195 = slice_5

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_8 = paddle._C_ops.strided_slice(
            transpose_8, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_9 = paddle._C_ops.strided_slice(
            transpose_8, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_16 = paddle._C_ops.multiply(strided_slice_8, slice_5)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_17 = paddle._C_ops.multiply(strided_slice_9, slice_4)

        # pd_op.subtract: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        subtract_4 = paddle._C_ops.subtract(multiply_16, multiply_17)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_18 = paddle._C_ops.multiply(strided_slice_8, slice_4)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_19 = paddle._C_ops.multiply(strided_slice_9, slice_5)

        # pd_op.add: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        add_26 = paddle._C_ops.add(multiply_18, multiply_19)

        # builtin.combine: ([1x12x8x32xf32, 1x12x8x32xf32]) <- (1x12x8x32xf32, 1x12x8x32xf32)
        combine_4 = [subtract_4, add_26]

        # pd_op.stack: (1x12x8x32x2xf32) <- ([1x12x8x32xf32, 1x12x8x32xf32])
        stack_4 = paddle._C_ops.stack(combine_4, -1)
        del combine_4

        # pd_op.flatten: (1x12x8x64xf32) <- (1x12x8x32x2xf32)
        flatten_4 = paddle._C_ops.flatten(stack_4, 3, 4)

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_10 = paddle._C_ops.strided_slice(
            transpose_9, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_11 = paddle._C_ops.strided_slice(
            transpose_9, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_20 = paddle._C_ops.multiply(strided_slice_10, slice_5)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_21 = paddle._C_ops.multiply(strided_slice_11, slice_4)

        # pd_op.subtract: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        subtract_5 = paddle._C_ops.subtract(multiply_20, multiply_21)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_22 = paddle._C_ops.multiply(strided_slice_10, slice_4)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_23 = paddle._C_ops.multiply(strided_slice_11, slice_5)

        # pd_op.add: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        add_27 = paddle._C_ops.add(multiply_22, multiply_23)

        # builtin.combine: ([1x12x8x32xf32, 1x12x8x32xf32]) <- (1x12x8x32xf32, 1x12x8x32xf32)
        combine_5 = [subtract_5, add_27]

        # pd_op.stack: (1x12x8x32x2xf32) <- ([1x12x8x32xf32, 1x12x8x32xf32])
        stack_5 = paddle._C_ops.stack(combine_5, -1)
        del combine_5

        # pd_op.flatten: (1x12x8x64xf32) <- (1x12x8x32x2xf32)
        flatten_5 = paddle._C_ops.flatten(stack_5, 3, 4)

        # pd_op.scale: (1x12x8x64xf32) <- (1x12x8x64xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(flatten_4, full_3, float("0"), True)
        del flatten_4

        # pd_op.matmul: (1x12x8x8xf32) <- (1x12x8x64xf32, 1x12x8x64xf32)
        matmul_19 = paddle._C_ops.matmul(scale_3, flatten_5, False, True)

        # pd_op.add: (1x12x8x8xf32) <- (1x12x8x8xf32, 1x1x1x8xf32)
        add_28 = paddle._C_ops.add(matmul_19, unsqueeze_0)

        # pd_op.softmax: (1x12x8x8xf32) <- (1x12x8x8xf32)
        softmax_2 = paddle._C_ops.softmax(add_28, -1)
        del add_28

        # pd_op.dropout: (1x12x8x8xf32, 1x12x8x8xui8) <- (1x12x8x8xf32, None, 1xf32)
        dropout_14, dropout_15 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_2, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x8x64xf32) <- (1x12x8x8xf32, 1x12x8x64xf32)
        matmul_20 = paddle._C_ops.matmul(dropout_14, transpose_10, False, False)

        # pd_op.transpose: (1x8x12x64xf32) <- (1x12x8x64xf32)
        transpose_11 = paddle._C_ops.transpose(matmul_20, [0, 2, 1, 3])
        del matmul_20

        # pd_op.reshape: (1x8x768xf32) <- (1x8x12x64xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_11, full_int_array_7)

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_21 = paddle._C_ops.matmul(reshape_11, parameter_179, False, False)
        del parameter_179

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_29 = paddle._C_ops.add(matmul_21, parameter_178)
        del parameter_178

        # pd_op.dropout: (1x8x768xf32, 1x8x768xui8) <- (1x8x768xf32, None, 1xf32)
        dropout_16, dropout_17 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_29, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_29

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 1x8x768xf32)
        add_30 = paddle._C_ops.add(layer_norm_12, dropout_16)

        # pd_op.layer_norm: (1x8x768xf32, 1x8xf32, 1x8xf32) <- (1x8x768xf32, 768xf32, 768xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_30, parameter_173, parameter_172, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_172, parameter_173

        # pd_op.matmul: (1x8x3072xf32) <- (1x8x768xf32, 768x3072xf32)
        matmul_22 = paddle._C_ops.matmul(layer_norm_15, parameter_177, False, False)
        del parameter_177

        # pd_op.add: (1x8x3072xf32) <- (1x8x3072xf32, 3072xf32)
        add_31 = paddle._C_ops.add(matmul_22, parameter_176)
        del parameter_176

        # pd_op.gelu: (1x8x3072xf32) <- (1x8x3072xf32)
        gelu_2 = paddle._C_ops.gelu(add_31, False)

        # pd_op.matmul: (1x8x768xf32) <- (1x8x3072xf32, 3072x768xf32)
        matmul_23 = paddle._C_ops.matmul(gelu_2, parameter_175, False, False)
        del parameter_175

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_32 = paddle._C_ops.add(matmul_23, parameter_174)
        del parameter_174

        # pd_op.dropout: (1x8x768xf32, 1x8x768xui8) <- (1x8x768xf32, None, 1xf32)
        dropout_18, dropout_19 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_32, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_32

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 1x8x768xf32)
        add_33 = paddle._C_ops.add(layer_norm_15, dropout_18)

        # pd_op.layer_norm: (1x8x768xf32, 1x8xf32, 1x8xf32) <- (1x8x768xf32, 768xf32, 768xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_33, parameter_171, parameter_170, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_170, parameter_171

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_24 = paddle._C_ops.matmul(layer_norm_18, parameter_169, False, False)
        del parameter_169

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_34 = paddle._C_ops.add(matmul_24, parameter_168)
        del parameter_168

        # pd_op.reshape: (1x8x12x64xf32) <- (1x8x768xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(add_34, full_int_array_1)

        # pd_op.transpose: (1x12x8x64xf32) <- (1x8x12x64xf32)
        transpose_12 = paddle._C_ops.transpose(reshape_12, [0, 2, 1, 3])
        del reshape_12

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_25 = paddle._C_ops.matmul(layer_norm_18, parameter_167, False, False)
        del parameter_167

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_35 = paddle._C_ops.add(matmul_25, parameter_166)
        del parameter_166

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_26 = paddle._C_ops.matmul(layer_norm_18, parameter_165, False, False)
        del parameter_165

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_36 = paddle._C_ops.add(matmul_26, parameter_164)
        del parameter_164

        # pd_op.reshape: (1x8x12x64xf32) <- (1x8x768xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(add_35, full_int_array_1)

        # pd_op.transpose: (1x12x8x64xf32) <- (1x8x12x64xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_13, [0, 2, 1, 3])
        del reshape_13

        # pd_op.reshape: (1x8x12x64xf32) <- (1x8x768xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(add_36, full_int_array_1)

        # pd_op.transpose: (1x12x8x64xf32) <- (1x8x12x64xf32)
        transpose_14 = paddle._C_ops.transpose(reshape_14, [0, 2, 1, 3])
        del reshape_14

        # pd_op.slice: (8x32xf32) <- (1536x32xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            parameter_17, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_17

        # pd_op.assign: (8x32xf32) <- (8x32xf32)
        assign_196 = slice_6

        # pd_op.slice: (8x32xf32) <- (1536x32xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            parameter_16, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_16

        # pd_op.assign: (8x32xf32) <- (8x32xf32)
        assign_197 = slice_7

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_12 = paddle._C_ops.strided_slice(
            transpose_12, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_13 = paddle._C_ops.strided_slice(
            transpose_12, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_24 = paddle._C_ops.multiply(strided_slice_12, slice_7)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_25 = paddle._C_ops.multiply(strided_slice_13, slice_6)

        # pd_op.subtract: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        subtract_6 = paddle._C_ops.subtract(multiply_24, multiply_25)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_26 = paddle._C_ops.multiply(strided_slice_12, slice_6)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_27 = paddle._C_ops.multiply(strided_slice_13, slice_7)

        # pd_op.add: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        add_37 = paddle._C_ops.add(multiply_26, multiply_27)

        # builtin.combine: ([1x12x8x32xf32, 1x12x8x32xf32]) <- (1x12x8x32xf32, 1x12x8x32xf32)
        combine_6 = [subtract_6, add_37]

        # pd_op.stack: (1x12x8x32x2xf32) <- ([1x12x8x32xf32, 1x12x8x32xf32])
        stack_6 = paddle._C_ops.stack(combine_6, -1)
        del combine_6

        # pd_op.flatten: (1x12x8x64xf32) <- (1x12x8x32x2xf32)
        flatten_6 = paddle._C_ops.flatten(stack_6, 3, 4)

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_14 = paddle._C_ops.strided_slice(
            transpose_13, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_15 = paddle._C_ops.strided_slice(
            transpose_13, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_28 = paddle._C_ops.multiply(strided_slice_14, slice_7)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_29 = paddle._C_ops.multiply(strided_slice_15, slice_6)

        # pd_op.subtract: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        subtract_7 = paddle._C_ops.subtract(multiply_28, multiply_29)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_30 = paddle._C_ops.multiply(strided_slice_14, slice_6)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_31 = paddle._C_ops.multiply(strided_slice_15, slice_7)

        # pd_op.add: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        add_38 = paddle._C_ops.add(multiply_30, multiply_31)

        # builtin.combine: ([1x12x8x32xf32, 1x12x8x32xf32]) <- (1x12x8x32xf32, 1x12x8x32xf32)
        combine_7 = [subtract_7, add_38]

        # pd_op.stack: (1x12x8x32x2xf32) <- ([1x12x8x32xf32, 1x12x8x32xf32])
        stack_7 = paddle._C_ops.stack(combine_7, -1)
        del combine_7

        # pd_op.flatten: (1x12x8x64xf32) <- (1x12x8x32x2xf32)
        flatten_7 = paddle._C_ops.flatten(stack_7, 3, 4)

        # pd_op.scale: (1x12x8x64xf32) <- (1x12x8x64xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(flatten_6, full_3, float("0"), True)
        del flatten_6

        # pd_op.matmul: (1x12x8x8xf32) <- (1x12x8x64xf32, 1x12x8x64xf32)
        matmul_27 = paddle._C_ops.matmul(scale_4, flatten_7, False, True)

        # pd_op.add: (1x12x8x8xf32) <- (1x12x8x8xf32, 1x1x1x8xf32)
        add_39 = paddle._C_ops.add(matmul_27, unsqueeze_0)

        # pd_op.softmax: (1x12x8x8xf32) <- (1x12x8x8xf32)
        softmax_3 = paddle._C_ops.softmax(add_39, -1)
        del add_39

        # pd_op.dropout: (1x12x8x8xf32, 1x12x8x8xui8) <- (1x12x8x8xf32, None, 1xf32)
        dropout_20, dropout_21 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_3, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x8x64xf32) <- (1x12x8x8xf32, 1x12x8x64xf32)
        matmul_28 = paddle._C_ops.matmul(dropout_20, transpose_14, False, False)

        # pd_op.transpose: (1x8x12x64xf32) <- (1x12x8x64xf32)
        transpose_15 = paddle._C_ops.transpose(matmul_28, [0, 2, 1, 3])
        del matmul_28

        # pd_op.reshape: (1x8x768xf32) <- (1x8x12x64xf32, 3xi64)
        reshape_15 = paddle._C_ops.reshape(transpose_15, full_int_array_7)

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_29 = paddle._C_ops.matmul(reshape_15, parameter_163, False, False)
        del parameter_163

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_40 = paddle._C_ops.add(matmul_29, parameter_162)
        del parameter_162

        # pd_op.dropout: (1x8x768xf32, 1x8x768xui8) <- (1x8x768xf32, None, 1xf32)
        dropout_22, dropout_23 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_40, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_40

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 1x8x768xf32)
        add_41 = paddle._C_ops.add(layer_norm_18, dropout_22)

        # pd_op.layer_norm: (1x8x768xf32, 1x8xf32, 1x8xf32) <- (1x8x768xf32, 768xf32, 768xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_41, parameter_157, parameter_156, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_156, parameter_157

        # pd_op.matmul: (1x8x3072xf32) <- (1x8x768xf32, 768x3072xf32)
        matmul_30 = paddle._C_ops.matmul(layer_norm_21, parameter_161, False, False)
        del parameter_161

        # pd_op.add: (1x8x3072xf32) <- (1x8x3072xf32, 3072xf32)
        add_42 = paddle._C_ops.add(matmul_30, parameter_160)
        del parameter_160

        # pd_op.gelu: (1x8x3072xf32) <- (1x8x3072xf32)
        gelu_3 = paddle._C_ops.gelu(add_42, False)

        # pd_op.matmul: (1x8x768xf32) <- (1x8x3072xf32, 3072x768xf32)
        matmul_31 = paddle._C_ops.matmul(gelu_3, parameter_159, False, False)
        del parameter_159

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_43 = paddle._C_ops.add(matmul_31, parameter_158)
        del parameter_158

        # pd_op.dropout: (1x8x768xf32, 1x8x768xui8) <- (1x8x768xf32, None, 1xf32)
        dropout_24, dropout_25 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_43, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_43

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 1x8x768xf32)
        add_44 = paddle._C_ops.add(layer_norm_21, dropout_24)

        # pd_op.layer_norm: (1x8x768xf32, 1x8xf32, 1x8xf32) <- (1x8x768xf32, 768xf32, 768xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_44, parameter_155, parameter_154, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_154, parameter_155

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_32 = paddle._C_ops.matmul(layer_norm_24, parameter_153, False, False)
        del parameter_153

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_45 = paddle._C_ops.add(matmul_32, parameter_152)
        del parameter_152

        # pd_op.reshape: (1x8x12x64xf32) <- (1x8x768xf32, 4xi64)
        reshape_16 = paddle._C_ops.reshape(add_45, full_int_array_1)

        # pd_op.transpose: (1x12x8x64xf32) <- (1x8x12x64xf32)
        transpose_16 = paddle._C_ops.transpose(reshape_16, [0, 2, 1, 3])
        del reshape_16

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_33 = paddle._C_ops.matmul(layer_norm_24, parameter_151, False, False)
        del parameter_151

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_46 = paddle._C_ops.add(matmul_33, parameter_150)
        del parameter_150

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_34 = paddle._C_ops.matmul(layer_norm_24, parameter_149, False, False)
        del parameter_149

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_47 = paddle._C_ops.add(matmul_34, parameter_148)
        del parameter_148

        # pd_op.reshape: (1x8x12x64xf32) <- (1x8x768xf32, 4xi64)
        reshape_17 = paddle._C_ops.reshape(add_46, full_int_array_1)

        # pd_op.transpose: (1x12x8x64xf32) <- (1x8x12x64xf32)
        transpose_17 = paddle._C_ops.transpose(reshape_17, [0, 2, 1, 3])
        del reshape_17

        # pd_op.reshape: (1x8x12x64xf32) <- (1x8x768xf32, 4xi64)
        reshape_18 = paddle._C_ops.reshape(add_47, full_int_array_1)

        # pd_op.transpose: (1x12x8x64xf32) <- (1x8x12x64xf32)
        transpose_18 = paddle._C_ops.transpose(reshape_18, [0, 2, 1, 3])
        del reshape_18

        # pd_op.slice: (8x32xf32) <- (1536x32xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            parameter_15, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_15

        # pd_op.assign: (8x32xf32) <- (8x32xf32)
        assign_198 = slice_8

        # pd_op.slice: (8x32xf32) <- (1536x32xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            parameter_14, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_14

        # pd_op.assign: (8x32xf32) <- (8x32xf32)
        assign_199 = slice_9

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_16 = paddle._C_ops.strided_slice(
            transpose_16, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_17 = paddle._C_ops.strided_slice(
            transpose_16, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_32 = paddle._C_ops.multiply(strided_slice_16, slice_9)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_33 = paddle._C_ops.multiply(strided_slice_17, slice_8)

        # pd_op.subtract: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        subtract_8 = paddle._C_ops.subtract(multiply_32, multiply_33)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_34 = paddle._C_ops.multiply(strided_slice_16, slice_8)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_35 = paddle._C_ops.multiply(strided_slice_17, slice_9)

        # pd_op.add: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        add_48 = paddle._C_ops.add(multiply_34, multiply_35)

        # builtin.combine: ([1x12x8x32xf32, 1x12x8x32xf32]) <- (1x12x8x32xf32, 1x12x8x32xf32)
        combine_8 = [subtract_8, add_48]

        # pd_op.stack: (1x12x8x32x2xf32) <- ([1x12x8x32xf32, 1x12x8x32xf32])
        stack_8 = paddle._C_ops.stack(combine_8, -1)
        del combine_8

        # pd_op.flatten: (1x12x8x64xf32) <- (1x12x8x32x2xf32)
        flatten_8 = paddle._C_ops.flatten(stack_8, 3, 4)

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_18 = paddle._C_ops.strided_slice(
            transpose_17, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_19 = paddle._C_ops.strided_slice(
            transpose_17, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_36 = paddle._C_ops.multiply(strided_slice_18, slice_9)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_37 = paddle._C_ops.multiply(strided_slice_19, slice_8)

        # pd_op.subtract: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        subtract_9 = paddle._C_ops.subtract(multiply_36, multiply_37)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_38 = paddle._C_ops.multiply(strided_slice_18, slice_8)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_39 = paddle._C_ops.multiply(strided_slice_19, slice_9)

        # pd_op.add: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        add_49 = paddle._C_ops.add(multiply_38, multiply_39)

        # builtin.combine: ([1x12x8x32xf32, 1x12x8x32xf32]) <- (1x12x8x32xf32, 1x12x8x32xf32)
        combine_9 = [subtract_9, add_49]

        # pd_op.stack: (1x12x8x32x2xf32) <- ([1x12x8x32xf32, 1x12x8x32xf32])
        stack_9 = paddle._C_ops.stack(combine_9, -1)
        del combine_9

        # pd_op.flatten: (1x12x8x64xf32) <- (1x12x8x32x2xf32)
        flatten_9 = paddle._C_ops.flatten(stack_9, 3, 4)

        # pd_op.scale: (1x12x8x64xf32) <- (1x12x8x64xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(flatten_8, full_3, float("0"), True)
        del flatten_8

        # pd_op.matmul: (1x12x8x8xf32) <- (1x12x8x64xf32, 1x12x8x64xf32)
        matmul_35 = paddle._C_ops.matmul(scale_5, flatten_9, False, True)

        # pd_op.add: (1x12x8x8xf32) <- (1x12x8x8xf32, 1x1x1x8xf32)
        add_50 = paddle._C_ops.add(matmul_35, unsqueeze_0)

        # pd_op.softmax: (1x12x8x8xf32) <- (1x12x8x8xf32)
        softmax_4 = paddle._C_ops.softmax(add_50, -1)
        del add_50

        # pd_op.dropout: (1x12x8x8xf32, 1x12x8x8xui8) <- (1x12x8x8xf32, None, 1xf32)
        dropout_26, dropout_27 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_4, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x8x64xf32) <- (1x12x8x8xf32, 1x12x8x64xf32)
        matmul_36 = paddle._C_ops.matmul(dropout_26, transpose_18, False, False)

        # pd_op.transpose: (1x8x12x64xf32) <- (1x12x8x64xf32)
        transpose_19 = paddle._C_ops.transpose(matmul_36, [0, 2, 1, 3])
        del matmul_36

        # pd_op.reshape: (1x8x768xf32) <- (1x8x12x64xf32, 3xi64)
        reshape_19 = paddle._C_ops.reshape(transpose_19, full_int_array_7)

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_37 = paddle._C_ops.matmul(reshape_19, parameter_147, False, False)
        del parameter_147

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_51 = paddle._C_ops.add(matmul_37, parameter_146)
        del parameter_146

        # pd_op.dropout: (1x8x768xf32, 1x8x768xui8) <- (1x8x768xf32, None, 1xf32)
        dropout_28, dropout_29 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_51, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_51

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 1x8x768xf32)
        add_52 = paddle._C_ops.add(layer_norm_24, dropout_28)

        # pd_op.layer_norm: (1x8x768xf32, 1x8xf32, 1x8xf32) <- (1x8x768xf32, 768xf32, 768xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_52, parameter_141, parameter_140, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_140, parameter_141

        # pd_op.matmul: (1x8x3072xf32) <- (1x8x768xf32, 768x3072xf32)
        matmul_38 = paddle._C_ops.matmul(layer_norm_27, parameter_145, False, False)
        del parameter_145

        # pd_op.add: (1x8x3072xf32) <- (1x8x3072xf32, 3072xf32)
        add_53 = paddle._C_ops.add(matmul_38, parameter_144)
        del parameter_144

        # pd_op.gelu: (1x8x3072xf32) <- (1x8x3072xf32)
        gelu_4 = paddle._C_ops.gelu(add_53, False)

        # pd_op.matmul: (1x8x768xf32) <- (1x8x3072xf32, 3072x768xf32)
        matmul_39 = paddle._C_ops.matmul(gelu_4, parameter_143, False, False)
        del parameter_143

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_54 = paddle._C_ops.add(matmul_39, parameter_142)
        del parameter_142

        # pd_op.dropout: (1x8x768xf32, 1x8x768xui8) <- (1x8x768xf32, None, 1xf32)
        dropout_30, dropout_31 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_54, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_54

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 1x8x768xf32)
        add_55 = paddle._C_ops.add(layer_norm_27, dropout_30)

        # pd_op.layer_norm: (1x8x768xf32, 1x8xf32, 1x8xf32) <- (1x8x768xf32, 768xf32, 768xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_55, parameter_139, parameter_138, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_138, parameter_139

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_40 = paddle._C_ops.matmul(layer_norm_30, parameter_137, False, False)
        del parameter_137

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_56 = paddle._C_ops.add(matmul_40, parameter_136)
        del parameter_136

        # pd_op.reshape: (1x8x12x64xf32) <- (1x8x768xf32, 4xi64)
        reshape_20 = paddle._C_ops.reshape(add_56, full_int_array_1)

        # pd_op.transpose: (1x12x8x64xf32) <- (1x8x12x64xf32)
        transpose_20 = paddle._C_ops.transpose(reshape_20, [0, 2, 1, 3])
        del reshape_20

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_41 = paddle._C_ops.matmul(layer_norm_30, parameter_135, False, False)
        del parameter_135

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_57 = paddle._C_ops.add(matmul_41, parameter_134)
        del parameter_134

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_42 = paddle._C_ops.matmul(layer_norm_30, parameter_133, False, False)
        del parameter_133

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_58 = paddle._C_ops.add(matmul_42, parameter_132)
        del parameter_132

        # pd_op.reshape: (1x8x12x64xf32) <- (1x8x768xf32, 4xi64)
        reshape_21 = paddle._C_ops.reshape(add_57, full_int_array_1)

        # pd_op.transpose: (1x12x8x64xf32) <- (1x8x12x64xf32)
        transpose_21 = paddle._C_ops.transpose(reshape_21, [0, 2, 1, 3])
        del reshape_21

        # pd_op.reshape: (1x8x12x64xf32) <- (1x8x768xf32, 4xi64)
        reshape_22 = paddle._C_ops.reshape(add_58, full_int_array_1)

        # pd_op.transpose: (1x12x8x64xf32) <- (1x8x12x64xf32)
        transpose_22 = paddle._C_ops.transpose(reshape_22, [0, 2, 1, 3])
        del reshape_22

        # pd_op.slice: (8x32xf32) <- (1536x32xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            parameter_13, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_13

        # pd_op.assign: (8x32xf32) <- (8x32xf32)
        assign_200 = slice_10

        # pd_op.slice: (8x32xf32) <- (1536x32xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            parameter_12, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_12

        # pd_op.assign: (8x32xf32) <- (8x32xf32)
        assign_201 = slice_11

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_20 = paddle._C_ops.strided_slice(
            transpose_20, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_21 = paddle._C_ops.strided_slice(
            transpose_20, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_40 = paddle._C_ops.multiply(strided_slice_20, slice_11)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_41 = paddle._C_ops.multiply(strided_slice_21, slice_10)

        # pd_op.subtract: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        subtract_10 = paddle._C_ops.subtract(multiply_40, multiply_41)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_42 = paddle._C_ops.multiply(strided_slice_20, slice_10)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_43 = paddle._C_ops.multiply(strided_slice_21, slice_11)

        # pd_op.add: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        add_59 = paddle._C_ops.add(multiply_42, multiply_43)

        # builtin.combine: ([1x12x8x32xf32, 1x12x8x32xf32]) <- (1x12x8x32xf32, 1x12x8x32xf32)
        combine_10 = [subtract_10, add_59]

        # pd_op.stack: (1x12x8x32x2xf32) <- ([1x12x8x32xf32, 1x12x8x32xf32])
        stack_10 = paddle._C_ops.stack(combine_10, -1)
        del combine_10

        # pd_op.flatten: (1x12x8x64xf32) <- (1x12x8x32x2xf32)
        flatten_10 = paddle._C_ops.flatten(stack_10, 3, 4)

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_22 = paddle._C_ops.strided_slice(
            transpose_21, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_23 = paddle._C_ops.strided_slice(
            transpose_21, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_44 = paddle._C_ops.multiply(strided_slice_22, slice_11)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_45 = paddle._C_ops.multiply(strided_slice_23, slice_10)

        # pd_op.subtract: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        subtract_11 = paddle._C_ops.subtract(multiply_44, multiply_45)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_46 = paddle._C_ops.multiply(strided_slice_22, slice_10)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_47 = paddle._C_ops.multiply(strided_slice_23, slice_11)

        # pd_op.add: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        add_60 = paddle._C_ops.add(multiply_46, multiply_47)

        # builtin.combine: ([1x12x8x32xf32, 1x12x8x32xf32]) <- (1x12x8x32xf32, 1x12x8x32xf32)
        combine_11 = [subtract_11, add_60]

        # pd_op.stack: (1x12x8x32x2xf32) <- ([1x12x8x32xf32, 1x12x8x32xf32])
        stack_11 = paddle._C_ops.stack(combine_11, -1)
        del combine_11

        # pd_op.flatten: (1x12x8x64xf32) <- (1x12x8x32x2xf32)
        flatten_11 = paddle._C_ops.flatten(stack_11, 3, 4)

        # pd_op.scale: (1x12x8x64xf32) <- (1x12x8x64xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(flatten_10, full_3, float("0"), True)
        del flatten_10

        # pd_op.matmul: (1x12x8x8xf32) <- (1x12x8x64xf32, 1x12x8x64xf32)
        matmul_43 = paddle._C_ops.matmul(scale_6, flatten_11, False, True)

        # pd_op.add: (1x12x8x8xf32) <- (1x12x8x8xf32, 1x1x1x8xf32)
        add_61 = paddle._C_ops.add(matmul_43, unsqueeze_0)

        # pd_op.softmax: (1x12x8x8xf32) <- (1x12x8x8xf32)
        softmax_5 = paddle._C_ops.softmax(add_61, -1)
        del add_61

        # pd_op.dropout: (1x12x8x8xf32, 1x12x8x8xui8) <- (1x12x8x8xf32, None, 1xf32)
        dropout_32, dropout_33 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_5, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x8x64xf32) <- (1x12x8x8xf32, 1x12x8x64xf32)
        matmul_44 = paddle._C_ops.matmul(dropout_32, transpose_22, False, False)

        # pd_op.transpose: (1x8x12x64xf32) <- (1x12x8x64xf32)
        transpose_23 = paddle._C_ops.transpose(matmul_44, [0, 2, 1, 3])
        del matmul_44

        # pd_op.reshape: (1x8x768xf32) <- (1x8x12x64xf32, 3xi64)
        reshape_23 = paddle._C_ops.reshape(transpose_23, full_int_array_7)

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_45 = paddle._C_ops.matmul(reshape_23, parameter_131, False, False)
        del parameter_131

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_62 = paddle._C_ops.add(matmul_45, parameter_130)
        del parameter_130

        # pd_op.dropout: (1x8x768xf32, 1x8x768xui8) <- (1x8x768xf32, None, 1xf32)
        dropout_34, dropout_35 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_62, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_62

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 1x8x768xf32)
        add_63 = paddle._C_ops.add(layer_norm_30, dropout_34)

        # pd_op.layer_norm: (1x8x768xf32, 1x8xf32, 1x8xf32) <- (1x8x768xf32, 768xf32, 768xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_63, parameter_125, parameter_124, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_124, parameter_125

        # pd_op.matmul: (1x8x3072xf32) <- (1x8x768xf32, 768x3072xf32)
        matmul_46 = paddle._C_ops.matmul(layer_norm_33, parameter_129, False, False)
        del parameter_129

        # pd_op.add: (1x8x3072xf32) <- (1x8x3072xf32, 3072xf32)
        add_64 = paddle._C_ops.add(matmul_46, parameter_128)
        del parameter_128

        # pd_op.gelu: (1x8x3072xf32) <- (1x8x3072xf32)
        gelu_5 = paddle._C_ops.gelu(add_64, False)

        # pd_op.matmul: (1x8x768xf32) <- (1x8x3072xf32, 3072x768xf32)
        matmul_47 = paddle._C_ops.matmul(gelu_5, parameter_127, False, False)
        del parameter_127

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_65 = paddle._C_ops.add(matmul_47, parameter_126)
        del parameter_126

        # pd_op.dropout: (1x8x768xf32, 1x8x768xui8) <- (1x8x768xf32, None, 1xf32)
        dropout_36, dropout_37 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_65, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_65

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 1x8x768xf32)
        add_66 = paddle._C_ops.add(layer_norm_33, dropout_36)

        # pd_op.layer_norm: (1x8x768xf32, 1x8xf32, 1x8xf32) <- (1x8x768xf32, 768xf32, 768xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_66, parameter_123, parameter_122, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_122, parameter_123

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_48 = paddle._C_ops.matmul(layer_norm_36, parameter_121, False, False)
        del parameter_121

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_67 = paddle._C_ops.add(matmul_48, parameter_120)
        del parameter_120

        # pd_op.reshape: (1x8x12x64xf32) <- (1x8x768xf32, 4xi64)
        reshape_24 = paddle._C_ops.reshape(add_67, full_int_array_1)

        # pd_op.transpose: (1x12x8x64xf32) <- (1x8x12x64xf32)
        transpose_24 = paddle._C_ops.transpose(reshape_24, [0, 2, 1, 3])
        del reshape_24

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_49 = paddle._C_ops.matmul(layer_norm_36, parameter_119, False, False)
        del parameter_119

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_68 = paddle._C_ops.add(matmul_49, parameter_118)
        del parameter_118

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_50 = paddle._C_ops.matmul(layer_norm_36, parameter_117, False, False)
        del parameter_117

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_69 = paddle._C_ops.add(matmul_50, parameter_116)
        del parameter_116

        # pd_op.reshape: (1x8x12x64xf32) <- (1x8x768xf32, 4xi64)
        reshape_25 = paddle._C_ops.reshape(add_68, full_int_array_1)

        # pd_op.transpose: (1x12x8x64xf32) <- (1x8x12x64xf32)
        transpose_25 = paddle._C_ops.transpose(reshape_25, [0, 2, 1, 3])
        del reshape_25

        # pd_op.reshape: (1x8x12x64xf32) <- (1x8x768xf32, 4xi64)
        reshape_26 = paddle._C_ops.reshape(add_69, full_int_array_1)

        # pd_op.transpose: (1x12x8x64xf32) <- (1x8x12x64xf32)
        transpose_26 = paddle._C_ops.transpose(reshape_26, [0, 2, 1, 3])
        del reshape_26

        # pd_op.slice: (8x32xf32) <- (1536x32xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            parameter_11, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_11

        # pd_op.assign: (8x32xf32) <- (8x32xf32)
        assign_202 = slice_12

        # pd_op.slice: (8x32xf32) <- (1536x32xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            parameter_10, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_10

        # pd_op.assign: (8x32xf32) <- (8x32xf32)
        assign_203 = slice_13

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_24 = paddle._C_ops.strided_slice(
            transpose_24, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_25 = paddle._C_ops.strided_slice(
            transpose_24, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_48 = paddle._C_ops.multiply(strided_slice_24, slice_13)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_49 = paddle._C_ops.multiply(strided_slice_25, slice_12)

        # pd_op.subtract: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        subtract_12 = paddle._C_ops.subtract(multiply_48, multiply_49)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_50 = paddle._C_ops.multiply(strided_slice_24, slice_12)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_51 = paddle._C_ops.multiply(strided_slice_25, slice_13)

        # pd_op.add: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        add_70 = paddle._C_ops.add(multiply_50, multiply_51)

        # builtin.combine: ([1x12x8x32xf32, 1x12x8x32xf32]) <- (1x12x8x32xf32, 1x12x8x32xf32)
        combine_12 = [subtract_12, add_70]

        # pd_op.stack: (1x12x8x32x2xf32) <- ([1x12x8x32xf32, 1x12x8x32xf32])
        stack_12 = paddle._C_ops.stack(combine_12, -1)
        del combine_12

        # pd_op.flatten: (1x12x8x64xf32) <- (1x12x8x32x2xf32)
        flatten_12 = paddle._C_ops.flatten(stack_12, 3, 4)

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_26 = paddle._C_ops.strided_slice(
            transpose_25, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_27 = paddle._C_ops.strided_slice(
            transpose_25, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_52 = paddle._C_ops.multiply(strided_slice_26, slice_13)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_53 = paddle._C_ops.multiply(strided_slice_27, slice_12)

        # pd_op.subtract: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        subtract_13 = paddle._C_ops.subtract(multiply_52, multiply_53)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_54 = paddle._C_ops.multiply(strided_slice_26, slice_12)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_55 = paddle._C_ops.multiply(strided_slice_27, slice_13)

        # pd_op.add: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        add_71 = paddle._C_ops.add(multiply_54, multiply_55)

        # builtin.combine: ([1x12x8x32xf32, 1x12x8x32xf32]) <- (1x12x8x32xf32, 1x12x8x32xf32)
        combine_13 = [subtract_13, add_71]

        # pd_op.stack: (1x12x8x32x2xf32) <- ([1x12x8x32xf32, 1x12x8x32xf32])
        stack_13 = paddle._C_ops.stack(combine_13, -1)
        del combine_13

        # pd_op.flatten: (1x12x8x64xf32) <- (1x12x8x32x2xf32)
        flatten_13 = paddle._C_ops.flatten(stack_13, 3, 4)

        # pd_op.scale: (1x12x8x64xf32) <- (1x12x8x64xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(flatten_12, full_3, float("0"), True)
        del flatten_12

        # pd_op.matmul: (1x12x8x8xf32) <- (1x12x8x64xf32, 1x12x8x64xf32)
        matmul_51 = paddle._C_ops.matmul(scale_7, flatten_13, False, True)

        # pd_op.add: (1x12x8x8xf32) <- (1x12x8x8xf32, 1x1x1x8xf32)
        add_72 = paddle._C_ops.add(matmul_51, unsqueeze_0)

        # pd_op.softmax: (1x12x8x8xf32) <- (1x12x8x8xf32)
        softmax_6 = paddle._C_ops.softmax(add_72, -1)
        del add_72

        # pd_op.dropout: (1x12x8x8xf32, 1x12x8x8xui8) <- (1x12x8x8xf32, None, 1xf32)
        dropout_38, dropout_39 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_6, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x8x64xf32) <- (1x12x8x8xf32, 1x12x8x64xf32)
        matmul_52 = paddle._C_ops.matmul(dropout_38, transpose_26, False, False)

        # pd_op.transpose: (1x8x12x64xf32) <- (1x12x8x64xf32)
        transpose_27 = paddle._C_ops.transpose(matmul_52, [0, 2, 1, 3])
        del matmul_52

        # pd_op.reshape: (1x8x768xf32) <- (1x8x12x64xf32, 3xi64)
        reshape_27 = paddle._C_ops.reshape(transpose_27, full_int_array_7)

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_53 = paddle._C_ops.matmul(reshape_27, parameter_115, False, False)
        del parameter_115

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_73 = paddle._C_ops.add(matmul_53, parameter_114)
        del parameter_114

        # pd_op.dropout: (1x8x768xf32, 1x8x768xui8) <- (1x8x768xf32, None, 1xf32)
        dropout_40, dropout_41 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_73, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_73

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 1x8x768xf32)
        add_74 = paddle._C_ops.add(layer_norm_36, dropout_40)

        # pd_op.layer_norm: (1x8x768xf32, 1x8xf32, 1x8xf32) <- (1x8x768xf32, 768xf32, 768xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_74, parameter_109, parameter_108, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_108, parameter_109

        # pd_op.matmul: (1x8x3072xf32) <- (1x8x768xf32, 768x3072xf32)
        matmul_54 = paddle._C_ops.matmul(layer_norm_39, parameter_113, False, False)
        del parameter_113

        # pd_op.add: (1x8x3072xf32) <- (1x8x3072xf32, 3072xf32)
        add_75 = paddle._C_ops.add(matmul_54, parameter_112)
        del parameter_112

        # pd_op.gelu: (1x8x3072xf32) <- (1x8x3072xf32)
        gelu_6 = paddle._C_ops.gelu(add_75, False)

        # pd_op.matmul: (1x8x768xf32) <- (1x8x3072xf32, 3072x768xf32)
        matmul_55 = paddle._C_ops.matmul(gelu_6, parameter_111, False, False)
        del parameter_111

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_76 = paddle._C_ops.add(matmul_55, parameter_110)
        del parameter_110

        # pd_op.dropout: (1x8x768xf32, 1x8x768xui8) <- (1x8x768xf32, None, 1xf32)
        dropout_42, dropout_43 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_76, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_76

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 1x8x768xf32)
        add_77 = paddle._C_ops.add(layer_norm_39, dropout_42)

        # pd_op.layer_norm: (1x8x768xf32, 1x8xf32, 1x8xf32) <- (1x8x768xf32, 768xf32, 768xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_77, parameter_107, parameter_106, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_106, parameter_107

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_56 = paddle._C_ops.matmul(layer_norm_42, parameter_105, False, False)
        del parameter_105

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_78 = paddle._C_ops.add(matmul_56, parameter_104)
        del parameter_104

        # pd_op.reshape: (1x8x12x64xf32) <- (1x8x768xf32, 4xi64)
        reshape_28 = paddle._C_ops.reshape(add_78, full_int_array_1)

        # pd_op.transpose: (1x12x8x64xf32) <- (1x8x12x64xf32)
        transpose_28 = paddle._C_ops.transpose(reshape_28, [0, 2, 1, 3])
        del reshape_28

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_57 = paddle._C_ops.matmul(layer_norm_42, parameter_103, False, False)
        del parameter_103

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_79 = paddle._C_ops.add(matmul_57, parameter_102)
        del parameter_102

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_58 = paddle._C_ops.matmul(layer_norm_42, parameter_101, False, False)
        del parameter_101

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_80 = paddle._C_ops.add(matmul_58, parameter_100)
        del parameter_100

        # pd_op.reshape: (1x8x12x64xf32) <- (1x8x768xf32, 4xi64)
        reshape_29 = paddle._C_ops.reshape(add_79, full_int_array_1)

        # pd_op.transpose: (1x12x8x64xf32) <- (1x8x12x64xf32)
        transpose_29 = paddle._C_ops.transpose(reshape_29, [0, 2, 1, 3])
        del reshape_29

        # pd_op.reshape: (1x8x12x64xf32) <- (1x8x768xf32, 4xi64)
        reshape_30 = paddle._C_ops.reshape(add_80, full_int_array_1)

        # pd_op.transpose: (1x12x8x64xf32) <- (1x8x12x64xf32)
        transpose_30 = paddle._C_ops.transpose(reshape_30, [0, 2, 1, 3])
        del reshape_30

        # pd_op.slice: (8x32xf32) <- (1536x32xf32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            parameter_9, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_9

        # pd_op.assign: (8x32xf32) <- (8x32xf32)
        assign_204 = slice_14

        # pd_op.slice: (8x32xf32) <- (1536x32xf32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            parameter_8, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_8

        # pd_op.assign: (8x32xf32) <- (8x32xf32)
        assign_205 = slice_15

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_28 = paddle._C_ops.strided_slice(
            transpose_28, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_29 = paddle._C_ops.strided_slice(
            transpose_28, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_56 = paddle._C_ops.multiply(strided_slice_28, slice_15)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_57 = paddle._C_ops.multiply(strided_slice_29, slice_14)

        # pd_op.subtract: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        subtract_14 = paddle._C_ops.subtract(multiply_56, multiply_57)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_58 = paddle._C_ops.multiply(strided_slice_28, slice_14)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_59 = paddle._C_ops.multiply(strided_slice_29, slice_15)

        # pd_op.add: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        add_81 = paddle._C_ops.add(multiply_58, multiply_59)

        # builtin.combine: ([1x12x8x32xf32, 1x12x8x32xf32]) <- (1x12x8x32xf32, 1x12x8x32xf32)
        combine_14 = [subtract_14, add_81]

        # pd_op.stack: (1x12x8x32x2xf32) <- ([1x12x8x32xf32, 1x12x8x32xf32])
        stack_14 = paddle._C_ops.stack(combine_14, -1)
        del combine_14

        # pd_op.flatten: (1x12x8x64xf32) <- (1x12x8x32x2xf32)
        flatten_14 = paddle._C_ops.flatten(stack_14, 3, 4)

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_30 = paddle._C_ops.strided_slice(
            transpose_29, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_31 = paddle._C_ops.strided_slice(
            transpose_29, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_60 = paddle._C_ops.multiply(strided_slice_30, slice_15)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_61 = paddle._C_ops.multiply(strided_slice_31, slice_14)

        # pd_op.subtract: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        subtract_15 = paddle._C_ops.subtract(multiply_60, multiply_61)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_62 = paddle._C_ops.multiply(strided_slice_30, slice_14)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_63 = paddle._C_ops.multiply(strided_slice_31, slice_15)

        # pd_op.add: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        add_82 = paddle._C_ops.add(multiply_62, multiply_63)

        # builtin.combine: ([1x12x8x32xf32, 1x12x8x32xf32]) <- (1x12x8x32xf32, 1x12x8x32xf32)
        combine_15 = [subtract_15, add_82]

        # pd_op.stack: (1x12x8x32x2xf32) <- ([1x12x8x32xf32, 1x12x8x32xf32])
        stack_15 = paddle._C_ops.stack(combine_15, -1)
        del combine_15

        # pd_op.flatten: (1x12x8x64xf32) <- (1x12x8x32x2xf32)
        flatten_15 = paddle._C_ops.flatten(stack_15, 3, 4)

        # pd_op.scale: (1x12x8x64xf32) <- (1x12x8x64xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(flatten_14, full_3, float("0"), True)
        del flatten_14

        # pd_op.matmul: (1x12x8x8xf32) <- (1x12x8x64xf32, 1x12x8x64xf32)
        matmul_59 = paddle._C_ops.matmul(scale_8, flatten_15, False, True)

        # pd_op.add: (1x12x8x8xf32) <- (1x12x8x8xf32, 1x1x1x8xf32)
        add_83 = paddle._C_ops.add(matmul_59, unsqueeze_0)

        # pd_op.softmax: (1x12x8x8xf32) <- (1x12x8x8xf32)
        softmax_7 = paddle._C_ops.softmax(add_83, -1)
        del add_83

        # pd_op.dropout: (1x12x8x8xf32, 1x12x8x8xui8) <- (1x12x8x8xf32, None, 1xf32)
        dropout_44, dropout_45 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_7, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x8x64xf32) <- (1x12x8x8xf32, 1x12x8x64xf32)
        matmul_60 = paddle._C_ops.matmul(dropout_44, transpose_30, False, False)

        # pd_op.transpose: (1x8x12x64xf32) <- (1x12x8x64xf32)
        transpose_31 = paddle._C_ops.transpose(matmul_60, [0, 2, 1, 3])
        del matmul_60

        # pd_op.reshape: (1x8x768xf32) <- (1x8x12x64xf32, 3xi64)
        reshape_31 = paddle._C_ops.reshape(transpose_31, full_int_array_7)

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_61 = paddle._C_ops.matmul(reshape_31, parameter_99, False, False)
        del parameter_99

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_84 = paddle._C_ops.add(matmul_61, parameter_98)
        del parameter_98

        # pd_op.dropout: (1x8x768xf32, 1x8x768xui8) <- (1x8x768xf32, None, 1xf32)
        dropout_46, dropout_47 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_84, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_84

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 1x8x768xf32)
        add_85 = paddle._C_ops.add(layer_norm_42, dropout_46)

        # pd_op.layer_norm: (1x8x768xf32, 1x8xf32, 1x8xf32) <- (1x8x768xf32, 768xf32, 768xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_85, parameter_93, parameter_92, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_92, parameter_93

        # pd_op.matmul: (1x8x3072xf32) <- (1x8x768xf32, 768x3072xf32)
        matmul_62 = paddle._C_ops.matmul(layer_norm_45, parameter_97, False, False)
        del parameter_97

        # pd_op.add: (1x8x3072xf32) <- (1x8x3072xf32, 3072xf32)
        add_86 = paddle._C_ops.add(matmul_62, parameter_96)
        del parameter_96

        # pd_op.gelu: (1x8x3072xf32) <- (1x8x3072xf32)
        gelu_7 = paddle._C_ops.gelu(add_86, False)

        # pd_op.matmul: (1x8x768xf32) <- (1x8x3072xf32, 3072x768xf32)
        matmul_63 = paddle._C_ops.matmul(gelu_7, parameter_95, False, False)
        del parameter_95

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_87 = paddle._C_ops.add(matmul_63, parameter_94)
        del parameter_94

        # pd_op.dropout: (1x8x768xf32, 1x8x768xui8) <- (1x8x768xf32, None, 1xf32)
        dropout_48, dropout_49 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_87, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_87

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 1x8x768xf32)
        add_88 = paddle._C_ops.add(layer_norm_45, dropout_48)

        # pd_op.layer_norm: (1x8x768xf32, 1x8xf32, 1x8xf32) <- (1x8x768xf32, 768xf32, 768xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_88, parameter_91, parameter_90, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_90, parameter_91

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_64 = paddle._C_ops.matmul(layer_norm_48, parameter_89, False, False)
        del parameter_89

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_89 = paddle._C_ops.add(matmul_64, parameter_88)
        del parameter_88

        # pd_op.reshape: (1x8x12x64xf32) <- (1x8x768xf32, 4xi64)
        reshape_32 = paddle._C_ops.reshape(add_89, full_int_array_1)

        # pd_op.transpose: (1x12x8x64xf32) <- (1x8x12x64xf32)
        transpose_32 = paddle._C_ops.transpose(reshape_32, [0, 2, 1, 3])
        del reshape_32

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_65 = paddle._C_ops.matmul(layer_norm_48, parameter_87, False, False)
        del parameter_87

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_90 = paddle._C_ops.add(matmul_65, parameter_86)
        del parameter_86

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_66 = paddle._C_ops.matmul(layer_norm_48, parameter_85, False, False)
        del parameter_85

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_91 = paddle._C_ops.add(matmul_66, parameter_84)
        del parameter_84

        # pd_op.reshape: (1x8x12x64xf32) <- (1x8x768xf32, 4xi64)
        reshape_33 = paddle._C_ops.reshape(add_90, full_int_array_1)

        # pd_op.transpose: (1x12x8x64xf32) <- (1x8x12x64xf32)
        transpose_33 = paddle._C_ops.transpose(reshape_33, [0, 2, 1, 3])
        del reshape_33

        # pd_op.reshape: (1x8x12x64xf32) <- (1x8x768xf32, 4xi64)
        reshape_34 = paddle._C_ops.reshape(add_91, full_int_array_1)

        # pd_op.transpose: (1x12x8x64xf32) <- (1x8x12x64xf32)
        transpose_34 = paddle._C_ops.transpose(reshape_34, [0, 2, 1, 3])
        del reshape_34

        # pd_op.slice: (8x32xf32) <- (1536x32xf32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            parameter_7, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_7

        # pd_op.assign: (8x32xf32) <- (8x32xf32)
        assign_206 = slice_16

        # pd_op.slice: (8x32xf32) <- (1536x32xf32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            parameter_6, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_6

        # pd_op.assign: (8x32xf32) <- (8x32xf32)
        assign_207 = slice_17

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_32 = paddle._C_ops.strided_slice(
            transpose_32, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_33 = paddle._C_ops.strided_slice(
            transpose_32, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_64 = paddle._C_ops.multiply(strided_slice_32, slice_17)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_65 = paddle._C_ops.multiply(strided_slice_33, slice_16)

        # pd_op.subtract: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        subtract_16 = paddle._C_ops.subtract(multiply_64, multiply_65)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_66 = paddle._C_ops.multiply(strided_slice_32, slice_16)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_67 = paddle._C_ops.multiply(strided_slice_33, slice_17)

        # pd_op.add: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        add_92 = paddle._C_ops.add(multiply_66, multiply_67)

        # builtin.combine: ([1x12x8x32xf32, 1x12x8x32xf32]) <- (1x12x8x32xf32, 1x12x8x32xf32)
        combine_16 = [subtract_16, add_92]

        # pd_op.stack: (1x12x8x32x2xf32) <- ([1x12x8x32xf32, 1x12x8x32xf32])
        stack_16 = paddle._C_ops.stack(combine_16, -1)
        del combine_16

        # pd_op.flatten: (1x12x8x64xf32) <- (1x12x8x32x2xf32)
        flatten_16 = paddle._C_ops.flatten(stack_16, 3, 4)

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_34 = paddle._C_ops.strided_slice(
            transpose_33, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_35 = paddle._C_ops.strided_slice(
            transpose_33, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_68 = paddle._C_ops.multiply(strided_slice_34, slice_17)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_69 = paddle._C_ops.multiply(strided_slice_35, slice_16)

        # pd_op.subtract: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        subtract_17 = paddle._C_ops.subtract(multiply_68, multiply_69)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_70 = paddle._C_ops.multiply(strided_slice_34, slice_16)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_71 = paddle._C_ops.multiply(strided_slice_35, slice_17)

        # pd_op.add: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        add_93 = paddle._C_ops.add(multiply_70, multiply_71)

        # builtin.combine: ([1x12x8x32xf32, 1x12x8x32xf32]) <- (1x12x8x32xf32, 1x12x8x32xf32)
        combine_17 = [subtract_17, add_93]

        # pd_op.stack: (1x12x8x32x2xf32) <- ([1x12x8x32xf32, 1x12x8x32xf32])
        stack_17 = paddle._C_ops.stack(combine_17, -1)
        del combine_17

        # pd_op.flatten: (1x12x8x64xf32) <- (1x12x8x32x2xf32)
        flatten_17 = paddle._C_ops.flatten(stack_17, 3, 4)

        # pd_op.scale: (1x12x8x64xf32) <- (1x12x8x64xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(flatten_16, full_3, float("0"), True)
        del flatten_16

        # pd_op.matmul: (1x12x8x8xf32) <- (1x12x8x64xf32, 1x12x8x64xf32)
        matmul_67 = paddle._C_ops.matmul(scale_9, flatten_17, False, True)

        # pd_op.add: (1x12x8x8xf32) <- (1x12x8x8xf32, 1x1x1x8xf32)
        add_94 = paddle._C_ops.add(matmul_67, unsqueeze_0)

        # pd_op.softmax: (1x12x8x8xf32) <- (1x12x8x8xf32)
        softmax_8 = paddle._C_ops.softmax(add_94, -1)
        del add_94

        # pd_op.dropout: (1x12x8x8xf32, 1x12x8x8xui8) <- (1x12x8x8xf32, None, 1xf32)
        dropout_50, dropout_51 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_8, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x8x64xf32) <- (1x12x8x8xf32, 1x12x8x64xf32)
        matmul_68 = paddle._C_ops.matmul(dropout_50, transpose_34, False, False)

        # pd_op.transpose: (1x8x12x64xf32) <- (1x12x8x64xf32)
        transpose_35 = paddle._C_ops.transpose(matmul_68, [0, 2, 1, 3])
        del matmul_68

        # pd_op.reshape: (1x8x768xf32) <- (1x8x12x64xf32, 3xi64)
        reshape_35 = paddle._C_ops.reshape(transpose_35, full_int_array_7)

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_69 = paddle._C_ops.matmul(reshape_35, parameter_83, False, False)
        del parameter_83

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_95 = paddle._C_ops.add(matmul_69, parameter_82)
        del parameter_82

        # pd_op.dropout: (1x8x768xf32, 1x8x768xui8) <- (1x8x768xf32, None, 1xf32)
        dropout_52, dropout_53 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_95, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_95

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 1x8x768xf32)
        add_96 = paddle._C_ops.add(layer_norm_48, dropout_52)

        # pd_op.layer_norm: (1x8x768xf32, 1x8xf32, 1x8xf32) <- (1x8x768xf32, 768xf32, 768xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_96, parameter_77, parameter_76, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_76, parameter_77

        # pd_op.matmul: (1x8x3072xf32) <- (1x8x768xf32, 768x3072xf32)
        matmul_70 = paddle._C_ops.matmul(layer_norm_51, parameter_81, False, False)
        del parameter_81

        # pd_op.add: (1x8x3072xf32) <- (1x8x3072xf32, 3072xf32)
        add_97 = paddle._C_ops.add(matmul_70, parameter_80)
        del parameter_80

        # pd_op.gelu: (1x8x3072xf32) <- (1x8x3072xf32)
        gelu_8 = paddle._C_ops.gelu(add_97, False)

        # pd_op.matmul: (1x8x768xf32) <- (1x8x3072xf32, 3072x768xf32)
        matmul_71 = paddle._C_ops.matmul(gelu_8, parameter_79, False, False)
        del parameter_79

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_98 = paddle._C_ops.add(matmul_71, parameter_78)
        del parameter_78

        # pd_op.dropout: (1x8x768xf32, 1x8x768xui8) <- (1x8x768xf32, None, 1xf32)
        dropout_54, dropout_55 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_98, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_98

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 1x8x768xf32)
        add_99 = paddle._C_ops.add(layer_norm_51, dropout_54)

        # pd_op.layer_norm: (1x8x768xf32, 1x8xf32, 1x8xf32) <- (1x8x768xf32, 768xf32, 768xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_99, parameter_75, parameter_74, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_74, parameter_75

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_72 = paddle._C_ops.matmul(layer_norm_54, parameter_73, False, False)
        del parameter_73

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_100 = paddle._C_ops.add(matmul_72, parameter_72)
        del parameter_72

        # pd_op.reshape: (1x8x12x64xf32) <- (1x8x768xf32, 4xi64)
        reshape_36 = paddle._C_ops.reshape(add_100, full_int_array_1)

        # pd_op.transpose: (1x12x8x64xf32) <- (1x8x12x64xf32)
        transpose_36 = paddle._C_ops.transpose(reshape_36, [0, 2, 1, 3])
        del reshape_36

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_73 = paddle._C_ops.matmul(layer_norm_54, parameter_71, False, False)
        del parameter_71

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_101 = paddle._C_ops.add(matmul_73, parameter_70)
        del parameter_70

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_74 = paddle._C_ops.matmul(layer_norm_54, parameter_69, False, False)
        del parameter_69

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_102 = paddle._C_ops.add(matmul_74, parameter_68)
        del parameter_68

        # pd_op.reshape: (1x8x12x64xf32) <- (1x8x768xf32, 4xi64)
        reshape_37 = paddle._C_ops.reshape(add_101, full_int_array_1)

        # pd_op.transpose: (1x12x8x64xf32) <- (1x8x12x64xf32)
        transpose_37 = paddle._C_ops.transpose(reshape_37, [0, 2, 1, 3])
        del reshape_37

        # pd_op.reshape: (1x8x12x64xf32) <- (1x8x768xf32, 4xi64)
        reshape_38 = paddle._C_ops.reshape(add_102, full_int_array_1)

        # pd_op.transpose: (1x12x8x64xf32) <- (1x8x12x64xf32)
        transpose_38 = paddle._C_ops.transpose(reshape_38, [0, 2, 1, 3])
        del reshape_38

        # pd_op.slice: (8x32xf32) <- (1536x32xf32, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(
            parameter_5, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_5

        # pd_op.assign: (8x32xf32) <- (8x32xf32)
        assign_208 = slice_18

        # pd_op.slice: (8x32xf32) <- (1536x32xf32, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(
            parameter_4, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_4

        # pd_op.assign: (8x32xf32) <- (8x32xf32)
        assign_209 = slice_19

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_36 = paddle._C_ops.strided_slice(
            transpose_36, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_37 = paddle._C_ops.strided_slice(
            transpose_36, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_72 = paddle._C_ops.multiply(strided_slice_36, slice_19)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_73 = paddle._C_ops.multiply(strided_slice_37, slice_18)

        # pd_op.subtract: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        subtract_18 = paddle._C_ops.subtract(multiply_72, multiply_73)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_74 = paddle._C_ops.multiply(strided_slice_36, slice_18)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_75 = paddle._C_ops.multiply(strided_slice_37, slice_19)

        # pd_op.add: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        add_103 = paddle._C_ops.add(multiply_74, multiply_75)

        # builtin.combine: ([1x12x8x32xf32, 1x12x8x32xf32]) <- (1x12x8x32xf32, 1x12x8x32xf32)
        combine_18 = [subtract_18, add_103]

        # pd_op.stack: (1x12x8x32x2xf32) <- ([1x12x8x32xf32, 1x12x8x32xf32])
        stack_18 = paddle._C_ops.stack(combine_18, -1)
        del combine_18

        # pd_op.flatten: (1x12x8x64xf32) <- (1x12x8x32x2xf32)
        flatten_18 = paddle._C_ops.flatten(stack_18, 3, 4)

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_38 = paddle._C_ops.strided_slice(
            transpose_37, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_39 = paddle._C_ops.strided_slice(
            transpose_37, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_76 = paddle._C_ops.multiply(strided_slice_38, slice_19)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_77 = paddle._C_ops.multiply(strided_slice_39, slice_18)

        # pd_op.subtract: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        subtract_19 = paddle._C_ops.subtract(multiply_76, multiply_77)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_78 = paddle._C_ops.multiply(strided_slice_38, slice_18)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_79 = paddle._C_ops.multiply(strided_slice_39, slice_19)

        # pd_op.add: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        add_104 = paddle._C_ops.add(multiply_78, multiply_79)

        # builtin.combine: ([1x12x8x32xf32, 1x12x8x32xf32]) <- (1x12x8x32xf32, 1x12x8x32xf32)
        combine_19 = [subtract_19, add_104]

        # pd_op.stack: (1x12x8x32x2xf32) <- ([1x12x8x32xf32, 1x12x8x32xf32])
        stack_19 = paddle._C_ops.stack(combine_19, -1)
        del combine_19

        # pd_op.flatten: (1x12x8x64xf32) <- (1x12x8x32x2xf32)
        flatten_19 = paddle._C_ops.flatten(stack_19, 3, 4)

        # pd_op.scale: (1x12x8x64xf32) <- (1x12x8x64xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(flatten_18, full_3, float("0"), True)
        del flatten_18

        # pd_op.matmul: (1x12x8x8xf32) <- (1x12x8x64xf32, 1x12x8x64xf32)
        matmul_75 = paddle._C_ops.matmul(scale_10, flatten_19, False, True)

        # pd_op.add: (1x12x8x8xf32) <- (1x12x8x8xf32, 1x1x1x8xf32)
        add_105 = paddle._C_ops.add(matmul_75, unsqueeze_0)

        # pd_op.softmax: (1x12x8x8xf32) <- (1x12x8x8xf32)
        softmax_9 = paddle._C_ops.softmax(add_105, -1)
        del add_105

        # pd_op.dropout: (1x12x8x8xf32, 1x12x8x8xui8) <- (1x12x8x8xf32, None, 1xf32)
        dropout_56, dropout_57 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_9, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x8x64xf32) <- (1x12x8x8xf32, 1x12x8x64xf32)
        matmul_76 = paddle._C_ops.matmul(dropout_56, transpose_38, False, False)

        # pd_op.transpose: (1x8x12x64xf32) <- (1x12x8x64xf32)
        transpose_39 = paddle._C_ops.transpose(matmul_76, [0, 2, 1, 3])
        del matmul_76

        # pd_op.reshape: (1x8x768xf32) <- (1x8x12x64xf32, 3xi64)
        reshape_39 = paddle._C_ops.reshape(transpose_39, full_int_array_7)

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_77 = paddle._C_ops.matmul(reshape_39, parameter_67, False, False)
        del parameter_67

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_106 = paddle._C_ops.add(matmul_77, parameter_66)
        del parameter_66

        # pd_op.dropout: (1x8x768xf32, 1x8x768xui8) <- (1x8x768xf32, None, 1xf32)
        dropout_58, dropout_59 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_106, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_106

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 1x8x768xf32)
        add_107 = paddle._C_ops.add(layer_norm_54, dropout_58)

        # pd_op.layer_norm: (1x8x768xf32, 1x8xf32, 1x8xf32) <- (1x8x768xf32, 768xf32, 768xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_107, parameter_61, parameter_60, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_60, parameter_61

        # pd_op.matmul: (1x8x3072xf32) <- (1x8x768xf32, 768x3072xf32)
        matmul_78 = paddle._C_ops.matmul(layer_norm_57, parameter_65, False, False)
        del parameter_65

        # pd_op.add: (1x8x3072xf32) <- (1x8x3072xf32, 3072xf32)
        add_108 = paddle._C_ops.add(matmul_78, parameter_64)
        del parameter_64

        # pd_op.gelu: (1x8x3072xf32) <- (1x8x3072xf32)
        gelu_9 = paddle._C_ops.gelu(add_108, False)

        # pd_op.matmul: (1x8x768xf32) <- (1x8x3072xf32, 3072x768xf32)
        matmul_79 = paddle._C_ops.matmul(gelu_9, parameter_63, False, False)
        del parameter_63

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_109 = paddle._C_ops.add(matmul_79, parameter_62)
        del parameter_62

        # pd_op.dropout: (1x8x768xf32, 1x8x768xui8) <- (1x8x768xf32, None, 1xf32)
        dropout_60, dropout_61 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_109, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_109

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 1x8x768xf32)
        add_110 = paddle._C_ops.add(layer_norm_57, dropout_60)

        # pd_op.layer_norm: (1x8x768xf32, 1x8xf32, 1x8xf32) <- (1x8x768xf32, 768xf32, 768xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_110, parameter_59, parameter_58, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_58, parameter_59

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_80 = paddle._C_ops.matmul(layer_norm_60, parameter_57, False, False)
        del parameter_57

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_111 = paddle._C_ops.add(matmul_80, parameter_56)
        del parameter_56

        # pd_op.reshape: (1x8x12x64xf32) <- (1x8x768xf32, 4xi64)
        reshape_40 = paddle._C_ops.reshape(add_111, full_int_array_1)

        # pd_op.transpose: (1x12x8x64xf32) <- (1x8x12x64xf32)
        transpose_40 = paddle._C_ops.transpose(reshape_40, [0, 2, 1, 3])
        del reshape_40

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_81 = paddle._C_ops.matmul(layer_norm_60, parameter_55, False, False)
        del parameter_55

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_112 = paddle._C_ops.add(matmul_81, parameter_54)
        del parameter_54

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_82 = paddle._C_ops.matmul(layer_norm_60, parameter_53, False, False)
        del parameter_53

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_113 = paddle._C_ops.add(matmul_82, parameter_52)
        del parameter_52

        # pd_op.reshape: (1x8x12x64xf32) <- (1x8x768xf32, 4xi64)
        reshape_41 = paddle._C_ops.reshape(add_112, full_int_array_1)

        # pd_op.transpose: (1x12x8x64xf32) <- (1x8x12x64xf32)
        transpose_41 = paddle._C_ops.transpose(reshape_41, [0, 2, 1, 3])
        del reshape_41

        # pd_op.reshape: (1x8x12x64xf32) <- (1x8x768xf32, 4xi64)
        reshape_42 = paddle._C_ops.reshape(add_113, full_int_array_1)

        # pd_op.transpose: (1x12x8x64xf32) <- (1x8x12x64xf32)
        transpose_42 = paddle._C_ops.transpose(reshape_42, [0, 2, 1, 3])
        del reshape_42

        # pd_op.slice: (8x32xf32) <- (1536x32xf32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(
            parameter_3, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_3

        # pd_op.assign: (8x32xf32) <- (8x32xf32)
        assign_210 = slice_20

        # pd_op.slice: (8x32xf32) <- (1536x32xf32, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(
            parameter_2, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_2

        # pd_op.assign: (8x32xf32) <- (8x32xf32)
        assign_211 = slice_21

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_40 = paddle._C_ops.strided_slice(
            transpose_40, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_41 = paddle._C_ops.strided_slice(
            transpose_40, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_80 = paddle._C_ops.multiply(strided_slice_40, slice_21)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_81 = paddle._C_ops.multiply(strided_slice_41, slice_20)

        # pd_op.subtract: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        subtract_20 = paddle._C_ops.subtract(multiply_80, multiply_81)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_82 = paddle._C_ops.multiply(strided_slice_40, slice_20)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_83 = paddle._C_ops.multiply(strided_slice_41, slice_21)

        # pd_op.add: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        add_114 = paddle._C_ops.add(multiply_82, multiply_83)

        # builtin.combine: ([1x12x8x32xf32, 1x12x8x32xf32]) <- (1x12x8x32xf32, 1x12x8x32xf32)
        combine_20 = [subtract_20, add_114]

        # pd_op.stack: (1x12x8x32x2xf32) <- ([1x12x8x32xf32, 1x12x8x32xf32])
        stack_20 = paddle._C_ops.stack(combine_20, -1)
        del combine_20

        # pd_op.flatten: (1x12x8x64xf32) <- (1x12x8x32x2xf32)
        flatten_20 = paddle._C_ops.flatten(stack_20, 3, 4)

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_42 = paddle._C_ops.strided_slice(
            transpose_41, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_43 = paddle._C_ops.strided_slice(
            transpose_41, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_84 = paddle._C_ops.multiply(strided_slice_42, slice_21)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_85 = paddle._C_ops.multiply(strided_slice_43, slice_20)

        # pd_op.subtract: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        subtract_21 = paddle._C_ops.subtract(multiply_84, multiply_85)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_86 = paddle._C_ops.multiply(strided_slice_42, slice_20)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_87 = paddle._C_ops.multiply(strided_slice_43, slice_21)

        # pd_op.add: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        add_115 = paddle._C_ops.add(multiply_86, multiply_87)

        # builtin.combine: ([1x12x8x32xf32, 1x12x8x32xf32]) <- (1x12x8x32xf32, 1x12x8x32xf32)
        combine_21 = [subtract_21, add_115]

        # pd_op.stack: (1x12x8x32x2xf32) <- ([1x12x8x32xf32, 1x12x8x32xf32])
        stack_21 = paddle._C_ops.stack(combine_21, -1)
        del combine_21

        # pd_op.flatten: (1x12x8x64xf32) <- (1x12x8x32x2xf32)
        flatten_21 = paddle._C_ops.flatten(stack_21, 3, 4)

        # pd_op.scale: (1x12x8x64xf32) <- (1x12x8x64xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(flatten_20, full_3, float("0"), True)
        del flatten_20

        # pd_op.matmul: (1x12x8x8xf32) <- (1x12x8x64xf32, 1x12x8x64xf32)
        matmul_83 = paddle._C_ops.matmul(scale_11, flatten_21, False, True)

        # pd_op.add: (1x12x8x8xf32) <- (1x12x8x8xf32, 1x1x1x8xf32)
        add_116 = paddle._C_ops.add(matmul_83, unsqueeze_0)

        # pd_op.softmax: (1x12x8x8xf32) <- (1x12x8x8xf32)
        softmax_10 = paddle._C_ops.softmax(add_116, -1)
        del add_116

        # pd_op.dropout: (1x12x8x8xf32, 1x12x8x8xui8) <- (1x12x8x8xf32, None, 1xf32)
        dropout_62, dropout_63 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_10, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x8x64xf32) <- (1x12x8x8xf32, 1x12x8x64xf32)
        matmul_84 = paddle._C_ops.matmul(dropout_62, transpose_42, False, False)

        # pd_op.transpose: (1x8x12x64xf32) <- (1x12x8x64xf32)
        transpose_43 = paddle._C_ops.transpose(matmul_84, [0, 2, 1, 3])
        del matmul_84

        # pd_op.reshape: (1x8x768xf32) <- (1x8x12x64xf32, 3xi64)
        reshape_43 = paddle._C_ops.reshape(transpose_43, full_int_array_7)

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_85 = paddle._C_ops.matmul(reshape_43, parameter_51, False, False)
        del parameter_51

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_117 = paddle._C_ops.add(matmul_85, parameter_50)
        del parameter_50

        # pd_op.dropout: (1x8x768xf32, 1x8x768xui8) <- (1x8x768xf32, None, 1xf32)
        dropout_64, dropout_65 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_117, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_117

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 1x8x768xf32)
        add_118 = paddle._C_ops.add(layer_norm_60, dropout_64)

        # pd_op.layer_norm: (1x8x768xf32, 1x8xf32, 1x8xf32) <- (1x8x768xf32, 768xf32, 768xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_118, parameter_45, parameter_44, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_44, parameter_45

        # pd_op.matmul: (1x8x3072xf32) <- (1x8x768xf32, 768x3072xf32)
        matmul_86 = paddle._C_ops.matmul(layer_norm_63, parameter_49, False, False)
        del parameter_49

        # pd_op.add: (1x8x3072xf32) <- (1x8x3072xf32, 3072xf32)
        add_119 = paddle._C_ops.add(matmul_86, parameter_48)
        del parameter_48

        # pd_op.gelu: (1x8x3072xf32) <- (1x8x3072xf32)
        gelu_10 = paddle._C_ops.gelu(add_119, False)

        # pd_op.matmul: (1x8x768xf32) <- (1x8x3072xf32, 3072x768xf32)
        matmul_87 = paddle._C_ops.matmul(gelu_10, parameter_47, False, False)
        del parameter_47

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_120 = paddle._C_ops.add(matmul_87, parameter_46)
        del parameter_46

        # pd_op.dropout: (1x8x768xf32, 1x8x768xui8) <- (1x8x768xf32, None, 1xf32)
        dropout_66, dropout_67 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_120, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_120

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 1x8x768xf32)
        add_121 = paddle._C_ops.add(layer_norm_63, dropout_66)

        # pd_op.layer_norm: (1x8x768xf32, 1x8xf32, 1x8xf32) <- (1x8x768xf32, 768xf32, 768xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_121, parameter_43, parameter_42, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_42, parameter_43

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_88 = paddle._C_ops.matmul(layer_norm_66, parameter_41, False, False)
        del parameter_41

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_122 = paddle._C_ops.add(matmul_88, parameter_40)
        del parameter_40

        # pd_op.reshape: (1x8x12x64xf32) <- (1x8x768xf32, 4xi64)
        reshape_44 = paddle._C_ops.reshape(add_122, full_int_array_1)

        # pd_op.transpose: (1x12x8x64xf32) <- (1x8x12x64xf32)
        transpose_44 = paddle._C_ops.transpose(reshape_44, [0, 2, 1, 3])
        del reshape_44

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_89 = paddle._C_ops.matmul(layer_norm_66, parameter_39, False, False)
        del parameter_39

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_123 = paddle._C_ops.add(matmul_89, parameter_38)
        del parameter_38

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_90 = paddle._C_ops.matmul(layer_norm_66, parameter_37, False, False)
        del parameter_37

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_124 = paddle._C_ops.add(matmul_90, parameter_36)
        del parameter_36

        # pd_op.reshape: (1x8x12x64xf32) <- (1x8x768xf32, 4xi64)
        reshape_45 = paddle._C_ops.reshape(add_123, full_int_array_1)

        # pd_op.transpose: (1x12x8x64xf32) <- (1x8x12x64xf32)
        transpose_45 = paddle._C_ops.transpose(reshape_45, [0, 2, 1, 3])
        del reshape_45

        # pd_op.reshape: (1x8x12x64xf32) <- (1x8x768xf32, 4xi64)
        reshape_46 = paddle._C_ops.reshape(add_124, full_int_array_1)
        del full_int_array_1

        # pd_op.transpose: (1x12x8x64xf32) <- (1x8x12x64xf32)
        transpose_46 = paddle._C_ops.transpose(reshape_46, [0, 2, 1, 3])
        del reshape_46

        # pd_op.slice: (8x32xf32) <- (1536x32xf32, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(
            parameter_1, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_1

        # pd_op.assign: (8x32xf32) <- (8x32xf32)
        assign_212 = slice_22

        # pd_op.slice: (8x32xf32) <- (1536x32xf32, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(
            parameter_0, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del full_int_array_3, parameter_0

        # pd_op.assign: (8x32xf32) <- (8x32xf32)
        assign_213 = slice_23

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_44 = paddle._C_ops.strided_slice(
            transpose_44, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_45 = paddle._C_ops.strided_slice(
            transpose_44, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_88 = paddle._C_ops.multiply(strided_slice_44, slice_23)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_89 = paddle._C_ops.multiply(strided_slice_45, slice_22)

        # pd_op.subtract: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        subtract_22 = paddle._C_ops.subtract(multiply_88, multiply_89)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_90 = paddle._C_ops.multiply(strided_slice_44, slice_22)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_91 = paddle._C_ops.multiply(strided_slice_45, slice_23)

        # pd_op.add: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        add_125 = paddle._C_ops.add(multiply_90, multiply_91)

        # builtin.combine: ([1x12x8x32xf32, 1x12x8x32xf32]) <- (1x12x8x32xf32, 1x12x8x32xf32)
        combine_22 = [subtract_22, add_125]

        # pd_op.stack: (1x12x8x32x2xf32) <- ([1x12x8x32xf32, 1x12x8x32xf32])
        stack_22 = paddle._C_ops.stack(combine_22, -1)
        del combine_22

        # pd_op.flatten: (1x12x8x64xf32) <- (1x12x8x32x2xf32)
        flatten_22 = paddle._C_ops.flatten(stack_22, 3, 4)

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_46 = paddle._C_ops.strided_slice(
            transpose_45, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x12x8x32xf32) <- (1x12x8x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_47 = paddle._C_ops.strided_slice(
            transpose_45, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_92 = paddle._C_ops.multiply(strided_slice_46, slice_23)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_93 = paddle._C_ops.multiply(strided_slice_47, slice_22)

        # pd_op.subtract: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        subtract_23 = paddle._C_ops.subtract(multiply_92, multiply_93)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_94 = paddle._C_ops.multiply(strided_slice_46, slice_22)

        # pd_op.multiply: (1x12x8x32xf32) <- (1x12x8x32xf32, 8x32xf32)
        multiply_95 = paddle._C_ops.multiply(strided_slice_47, slice_23)

        # pd_op.add: (1x12x8x32xf32) <- (1x12x8x32xf32, 1x12x8x32xf32)
        add_126 = paddle._C_ops.add(multiply_94, multiply_95)

        # builtin.combine: ([1x12x8x32xf32, 1x12x8x32xf32]) <- (1x12x8x32xf32, 1x12x8x32xf32)
        combine_23 = [subtract_23, add_126]

        # pd_op.stack: (1x12x8x32x2xf32) <- ([1x12x8x32xf32, 1x12x8x32xf32])
        stack_23 = paddle._C_ops.stack(combine_23, -1)
        del combine_23

        # pd_op.flatten: (1x12x8x64xf32) <- (1x12x8x32x2xf32)
        flatten_23 = paddle._C_ops.flatten(stack_23, 3, 4)

        # pd_op.scale: (1x12x8x64xf32) <- (1x12x8x64xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(flatten_22, full_3, float("0"), True)
        del flatten_22

        # pd_op.matmul: (1x12x8x8xf32) <- (1x12x8x64xf32, 1x12x8x64xf32)
        matmul_91 = paddle._C_ops.matmul(scale_12, flatten_23, False, True)

        # pd_op.add: (1x12x8x8xf32) <- (1x12x8x8xf32, 1x1x1x8xf32)
        add_127 = paddle._C_ops.add(matmul_91, unsqueeze_0)

        # pd_op.softmax: (1x12x8x8xf32) <- (1x12x8x8xf32)
        softmax_11 = paddle._C_ops.softmax(add_127, -1)
        del add_127

        # pd_op.dropout: (1x12x8x8xf32, 1x12x8x8xui8) <- (1x12x8x8xf32, None, 1xf32)
        dropout_68, dropout_69 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_11, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x8x64xf32) <- (1x12x8x8xf32, 1x12x8x64xf32)
        matmul_92 = paddle._C_ops.matmul(dropout_68, transpose_46, False, False)

        # pd_op.transpose: (1x8x12x64xf32) <- (1x12x8x64xf32)
        transpose_47 = paddle._C_ops.transpose(matmul_92, [0, 2, 1, 3])
        del matmul_92

        # pd_op.reshape: (1x8x768xf32) <- (1x8x12x64xf32, 3xi64)
        reshape_47 = paddle._C_ops.reshape(transpose_47, full_int_array_7)
        del full_int_array_7

        # pd_op.matmul: (1x8x768xf32) <- (1x8x768xf32, 768x768xf32)
        matmul_93 = paddle._C_ops.matmul(reshape_47, parameter_35, False, False)
        del parameter_35

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_128 = paddle._C_ops.add(matmul_93, parameter_34)
        del parameter_34

        # pd_op.dropout: (1x8x768xf32, 1x8x768xui8) <- (1x8x768xf32, None, 1xf32)
        dropout_70, dropout_71 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_128, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_128

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 1x8x768xf32)
        add_129 = paddle._C_ops.add(layer_norm_66, dropout_70)

        # pd_op.layer_norm: (1x8x768xf32, 1x8xf32, 1x8xf32) <- (1x8x768xf32, 768xf32, 768xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_129, parameter_29, parameter_28, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_28, parameter_29

        # pd_op.matmul: (1x8x3072xf32) <- (1x8x768xf32, 768x3072xf32)
        matmul_94 = paddle._C_ops.matmul(layer_norm_69, parameter_33, False, False)
        del parameter_33

        # pd_op.add: (1x8x3072xf32) <- (1x8x3072xf32, 3072xf32)
        add_130 = paddle._C_ops.add(matmul_94, parameter_32)
        del parameter_32

        # pd_op.gelu: (1x8x3072xf32) <- (1x8x3072xf32)
        gelu_11 = paddle._C_ops.gelu(add_130, False)

        # pd_op.matmul: (1x8x768xf32) <- (1x8x3072xf32, 3072x768xf32)
        matmul_95 = paddle._C_ops.matmul(gelu_11, parameter_31, False, False)
        del parameter_31

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 768xf32)
        add_131 = paddle._C_ops.add(matmul_95, parameter_30)
        del parameter_30

        # pd_op.dropout: (1x8x768xf32, 1x8x768xui8) <- (1x8x768xf32, None, 1xf32)
        dropout_72, dropout_73 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_131, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_131

        # pd_op.add: (1x8x768xf32) <- (1x8x768xf32, 1x8x768xf32)
        add_132 = paddle._C_ops.add(layer_norm_69, dropout_72)

        # pd_op.layer_norm: (1x8x768xf32, 1x8xf32, 1x8xf32) <- (1x8x768xf32, 768xf32, 768xf32)
        layer_norm_72, layer_norm_73, layer_norm_74 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_132, parameter_27, parameter_26, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_26, parameter_27

        # pd_op.slice: (1x768xf32) <- (1x8x768xf32, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(
            layer_norm_72, [1], full_int_array_2, full_int_array_6, [1], [1]
        )
        del full_int_array_2

        # pd_op.matmul: (1x768xf32) <- (1x768xf32, 768x768xf32)
        matmul_96 = paddle._C_ops.matmul(slice_24, parameter_25, False, False)
        del parameter_25

        # pd_op.add: (1x768xf32) <- (1x768xf32, 768xf32)
        add_133 = paddle._C_ops.add(matmul_96, parameter_24)
        del parameter_24

        # pd_op.tanh: (1x768xf32) <- (1x768xf32)
        tanh_0 = paddle._C_ops.tanh(add_133)
        del (
            add_0,
            add_1,
            add_100,
            add_101,
            add_102,
            add_103,
            add_104,
            add_107,
            add_108,
            add_11,
            add_110,
            add_111,
            add_112,
            add_113,
            add_114,
            add_115,
            add_118,
            add_119,
            add_12,
            add_121,
            add_122,
            add_123,
            add_124,
            add_125,
            add_126,
            add_129,
            add_13,
            add_130,
            add_132,
            add_133,
            add_14,
            add_15,
            add_16,
            add_19,
            add_2,
            add_20,
            add_22,
            add_23,
            add_24,
            add_25,
            add_26,
            add_27,
            add_3,
            add_30,
            add_31,
            add_33,
            add_34,
            add_35,
            add_36,
            add_37,
            add_38,
            add_4,
            add_41,
            add_42,
            add_44,
            add_45,
            add_46,
            add_47,
            add_48,
            add_49,
            add_5,
            add_52,
            add_53,
            add_55,
            add_56,
            add_57,
            add_58,
            add_59,
            add_60,
            add_63,
            add_64,
            add_66,
            add_67,
            add_68,
            add_69,
            add_70,
            add_71,
            add_74,
            add_75,
            add_77,
            add_78,
            add_79,
            add_8,
            add_80,
            add_81,
            add_82,
            add_85,
            add_86,
            add_88,
            add_89,
            add_9,
            add_90,
            add_91,
            add_92,
            add_93,
            add_96,
            add_97,
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
            flatten_1,
            flatten_11,
            flatten_13,
            flatten_15,
            flatten_17,
            flatten_19,
            flatten_21,
            flatten_23,
            flatten_3,
            flatten_5,
            flatten_7,
            flatten_9,
            full_2,
            full_3,
            full_int_array_4,
            full_int_array_5,
            full_int_array_6,
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
            matmul_11,
            matmul_13,
            matmul_14,
            matmul_15,
            matmul_16,
            matmul_17,
            matmul_18,
            matmul_19,
            matmul_2,
            matmul_21,
            matmul_22,
            matmul_23,
            matmul_24,
            matmul_25,
            matmul_26,
            matmul_27,
            matmul_29,
            matmul_3,
            matmul_30,
            matmul_31,
            matmul_32,
            matmul_33,
            matmul_34,
            matmul_35,
            matmul_37,
            matmul_38,
            matmul_39,
            matmul_40,
            matmul_41,
            matmul_42,
            matmul_43,
            matmul_45,
            matmul_46,
            matmul_47,
            matmul_48,
            matmul_49,
            matmul_5,
            matmul_50,
            matmul_51,
            matmul_53,
            matmul_54,
            matmul_55,
            matmul_56,
            matmul_57,
            matmul_58,
            matmul_59,
            matmul_6,
            matmul_61,
            matmul_62,
            matmul_63,
            matmul_64,
            matmul_65,
            matmul_66,
            matmul_67,
            matmul_69,
            matmul_7,
            matmul_70,
            matmul_71,
            matmul_72,
            matmul_73,
            matmul_74,
            matmul_75,
            matmul_77,
            matmul_78,
            matmul_79,
            matmul_8,
            matmul_80,
            matmul_81,
            matmul_82,
            matmul_83,
            matmul_85,
            matmul_86,
            matmul_87,
            matmul_88,
            matmul_89,
            matmul_9,
            matmul_90,
            matmul_91,
            matmul_93,
            matmul_94,
            matmul_95,
            matmul_96,
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
            multiply_31,
            multiply_32,
            multiply_33,
            multiply_34,
            multiply_35,
            multiply_36,
            multiply_37,
            multiply_38,
            multiply_39,
            multiply_4,
            multiply_40,
            multiply_41,
            multiply_42,
            multiply_43,
            multiply_44,
            multiply_45,
            multiply_46,
            multiply_47,
            multiply_48,
            multiply_49,
            multiply_5,
            multiply_50,
            multiply_51,
            multiply_52,
            multiply_53,
            multiply_54,
            multiply_55,
            multiply_56,
            multiply_57,
            multiply_58,
            multiply_59,
            multiply_6,
            multiply_60,
            multiply_61,
            multiply_62,
            multiply_63,
            multiply_64,
            multiply_65,
            multiply_66,
            multiply_67,
            multiply_68,
            multiply_69,
            multiply_7,
            multiply_70,
            multiply_71,
            multiply_72,
            multiply_73,
            multiply_74,
            multiply_75,
            multiply_76,
            multiply_77,
            multiply_78,
            multiply_79,
            multiply_8,
            multiply_80,
            multiply_81,
            multiply_82,
            multiply_83,
            multiply_84,
            multiply_85,
            multiply_86,
            multiply_87,
            multiply_88,
            multiply_89,
            multiply_9,
            multiply_90,
            multiply_91,
            multiply_92,
            multiply_93,
            multiply_94,
            multiply_95,
            reshape_11,
            reshape_15,
            reshape_19,
            reshape_23,
            reshape_27,
            reshape_3,
            reshape_31,
            reshape_35,
            reshape_39,
            reshape_43,
            reshape_47,
            reshape_7,
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
            slice_3,
            slice_4,
            slice_5,
            slice_6,
            slice_7,
            slice_8,
            slice_9,
            softmax_0,
            softmax_1,
            softmax_10,
            softmax_11,
            softmax_2,
            softmax_3,
            softmax_4,
            softmax_5,
            softmax_6,
            softmax_7,
            softmax_8,
            softmax_9,
            stack_0,
            stack_1,
            stack_10,
            stack_11,
            stack_12,
            stack_13,
            stack_14,
            stack_15,
            stack_16,
            stack_17,
            stack_18,
            stack_19,
            stack_2,
            stack_20,
            stack_21,
            stack_22,
            stack_23,
            stack_3,
            stack_4,
            stack_5,
            stack_6,
            stack_7,
            stack_8,
            stack_9,
            strided_slice_0,
            strided_slice_1,
            strided_slice_10,
            strided_slice_11,
            strided_slice_12,
            strided_slice_13,
            strided_slice_14,
            strided_slice_15,
            strided_slice_16,
            strided_slice_17,
            strided_slice_18,
            strided_slice_19,
            strided_slice_2,
            strided_slice_20,
            strided_slice_21,
            strided_slice_22,
            strided_slice_23,
            strided_slice_24,
            strided_slice_25,
            strided_slice_26,
            strided_slice_27,
            strided_slice_28,
            strided_slice_29,
            strided_slice_3,
            strided_slice_30,
            strided_slice_31,
            strided_slice_32,
            strided_slice_33,
            strided_slice_34,
            strided_slice_35,
            strided_slice_36,
            strided_slice_37,
            strided_slice_38,
            strided_slice_39,
            strided_slice_4,
            strided_slice_40,
            strided_slice_41,
            strided_slice_42,
            strided_slice_43,
            strided_slice_44,
            strided_slice_45,
            strided_slice_46,
            strided_slice_47,
            strided_slice_5,
            strided_slice_6,
            strided_slice_7,
            strided_slice_8,
            strided_slice_9,
            subtract_0,
            subtract_1,
            subtract_10,
            subtract_11,
            subtract_12,
            subtract_13,
            subtract_14,
            subtract_15,
            subtract_16,
            subtract_17,
            subtract_18,
            subtract_19,
            subtract_2,
            subtract_20,
            subtract_21,
            subtract_22,
            subtract_23,
            subtract_3,
            subtract_4,
            subtract_5,
            subtract_6,
            subtract_7,
            subtract_8,
            subtract_9,
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
        )

        return tanh_0
