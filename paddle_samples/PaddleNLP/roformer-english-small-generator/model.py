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

        # pd_op.embedding: (1x21x128xf32) <- (1x21xi64, 30522x128xf32)
        embedding_0 = paddle._C_ops.embedding(data_0, parameter_221, -1, False)
        del data_0, parameter_221

        # pd_op.embedding: (1x21x128xf32) <- (1x21xi64, 2x128xf32)
        embedding_1 = paddle._C_ops.embedding(data_1, parameter_220, -1, False)
        del data_1, parameter_220

        # pd_op.add: (1x21x128xf32) <- (1x21x128xf32, 1x21x128xf32)
        add_0 = paddle._C_ops.add(embedding_0, embedding_1)

        # pd_op.layer_norm: (1x21x128xf32, 1x21xf32, 1x21xf32) <- (1x21x128xf32, 128xf32, 128xf32)
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

        # pd_op.dropout: (1x21x128xf32, 1x21x128xui8) <- (1x21x128xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                layer_norm_0, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del layer_norm_0

        # pd_op.matmul: (1x21x64xf32) <- (1x21x128xf32, 128x64xf32)
        matmul_0 = paddle._C_ops.matmul(dropout_0, parameter_223, False, False)
        del parameter_223

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_1 = paddle._C_ops.add(matmul_0, parameter_222)
        del parameter_222

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_1 = paddle._C_ops.matmul(add_1, parameter_217, False, False)
        del parameter_217

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_2 = paddle._C_ops.add(matmul_1, parameter_216)
        del parameter_216

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [0, 0, 1, 64]

        # pd_op.reshape: (1x21x1x64xf32) <- (1x21x64xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(add_2, full_int_array_1)

        # pd_op.transpose: (1x1x21x64xf32) <- (1x21x1x64xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_0, [0, 2, 1, 3])
        del reshape_0

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_2 = paddle._C_ops.matmul(add_1, parameter_215, False, False)
        del parameter_215

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_3 = paddle._C_ops.add(matmul_2, parameter_214)
        del parameter_214

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_3 = paddle._C_ops.matmul(add_1, parameter_213, False, False)
        del parameter_213

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_4 = paddle._C_ops.add(matmul_3, parameter_212)
        del parameter_212

        # pd_op.reshape: (1x21x1x64xf32) <- (1x21x64xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(add_3, full_int_array_1)

        # pd_op.transpose: (1x1x21x64xf32) <- (1x21x1x64xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_1, [0, 2, 1, 3])
        del reshape_1

        # pd_op.reshape: (1x21x1x64xf32) <- (1x21x64xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(add_4, full_int_array_1)

        # pd_op.transpose: (1x1x21x64xf32) <- (1x21x1x64xf32)
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

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [21]

        # pd_op.slice: (21x32xf32) <- (128x32xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            parameter_23, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_23

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_73 = slice_0

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_74 = slice_0

        # pd_op.slice: (21x32xf32) <- (128x32xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            parameter_22, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_22

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_75 = slice_1

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_76 = slice_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [2147483647]

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

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_120 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_121 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_122 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_123 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_124 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_125 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_126 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_127 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_128 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_129 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_130 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_131 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_132 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_133 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_134 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_135 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_136 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_137 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_138 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_139 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_140 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_141 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_142 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_143 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_144 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_145 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_146 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_147 = full_int_array_4

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [2]

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

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_157 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_158 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_159 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_160 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_161 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_162 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_163 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_164 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_165 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_166 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_167 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_168 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_169 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_170 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_171 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_172 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_173 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_174 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_175 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_176 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_177 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_178 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_179 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_180 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_181 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_182 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_183 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_184 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_185 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_186 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_187 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_188 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_189 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_190 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_191 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_192 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_193 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_194 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_195 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_196 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_197 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_198 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_199 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_200 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_201 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_202 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_203 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_204 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_205 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_206 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_207 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_208 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_209 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_210 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_211 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_212 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_213 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_214 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_215 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_216 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_217 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_218 = full_int_array_5

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_0 = paddle._C_ops.strided_slice(
            transpose_0, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_219 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_220 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_221 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_222 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_223 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_224 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_225 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_226 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_227 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_228 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_229 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_230 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_231 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_232 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_233 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_234 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_235 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_236 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_237 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_238 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_239 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_240 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_241 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_242 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_243 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_244 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_245 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_246 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_247 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_248 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_249 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_250 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_251 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_252 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_253 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_254 = full_int_array_6

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_1 = paddle._C_ops.strided_slice(
            transpose_0, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_0 = paddle._C_ops.multiply(strided_slice_0, slice_1)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_1 = paddle._C_ops.multiply(strided_slice_1, slice_0)

        # pd_op.subtract: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        subtract_0 = paddle._C_ops.subtract(multiply_0, multiply_1)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_2 = paddle._C_ops.multiply(strided_slice_0, slice_0)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_3 = paddle._C_ops.multiply(strided_slice_1, slice_1)

        # pd_op.add: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        add_5 = paddle._C_ops.add(multiply_2, multiply_3)

        # builtin.combine: ([1x1x21x32xf32, 1x1x21x32xf32]) <- (1x1x21x32xf32, 1x1x21x32xf32)
        combine_0 = [subtract_0, add_5]

        # pd_op.stack: (1x1x21x32x2xf32) <- ([1x1x21x32xf32, 1x1x21x32xf32])
        stack_0 = paddle._C_ops.stack(combine_0, -1)
        del combine_0

        # pd_op.flatten: (1x1x21x64xf32) <- (1x1x21x32x2xf32)
        flatten_0 = paddle._C_ops.flatten(stack_0, 3, 4)

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_2 = paddle._C_ops.strided_slice(
            transpose_1, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_3 = paddle._C_ops.strided_slice(
            transpose_1, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_4 = paddle._C_ops.multiply(strided_slice_2, slice_1)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_5 = paddle._C_ops.multiply(strided_slice_3, slice_0)

        # pd_op.subtract: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        subtract_1 = paddle._C_ops.subtract(multiply_4, multiply_5)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_6 = paddle._C_ops.multiply(strided_slice_2, slice_0)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_7 = paddle._C_ops.multiply(strided_slice_3, slice_1)

        # pd_op.add: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        add_6 = paddle._C_ops.add(multiply_6, multiply_7)

        # builtin.combine: ([1x1x21x32xf32, 1x1x21x32xf32]) <- (1x1x21x32xf32, 1x1x21x32xf32)
        combine_1 = [subtract_1, add_6]

        # pd_op.stack: (1x1x21x32x2xf32) <- ([1x1x21x32xf32, 1x1x21x32xf32])
        stack_1 = paddle._C_ops.stack(combine_1, -1)
        del combine_1

        # pd_op.flatten: (1x1x21x64xf32) <- (1x1x21x32x2xf32)
        flatten_1 = paddle._C_ops.flatten(stack_1, 3, 4)

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_4 = paddle._C_ops.strided_slice(
            transpose_2, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_5 = paddle._C_ops.strided_slice(
            transpose_2, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_8 = paddle._C_ops.multiply(strided_slice_4, slice_1)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_9 = paddle._C_ops.multiply(strided_slice_5, slice_0)

        # pd_op.subtract: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        subtract_2 = paddle._C_ops.subtract(multiply_8, multiply_9)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_10 = paddle._C_ops.multiply(strided_slice_4, slice_0)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_11 = paddle._C_ops.multiply(strided_slice_5, slice_1)

        # pd_op.add: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        add_7 = paddle._C_ops.add(multiply_10, multiply_11)

        # builtin.combine: ([1x1x21x32xf32, 1x1x21x32xf32]) <- (1x1x21x32xf32, 1x1x21x32xf32)
        combine_2 = [subtract_2, add_7]

        # pd_op.stack: (1x1x21x32x2xf32) <- ([1x1x21x32xf32, 1x1x21x32xf32])
        stack_2 = paddle._C_ops.stack(combine_2, -1)
        del combine_2

        # pd_op.flatten: (1x1x21x64xf32) <- (1x1x21x32x2xf32)
        flatten_2 = paddle._C_ops.flatten(stack_2, 3, 4)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_255 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_256 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_257 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_258 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_259 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_260 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_261 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_262 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_263 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_264 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_265 = full_3

        # pd_op.scale: (1x1x21x64xf32) <- (1x1x21x64xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(flatten_0, full_3, float("0"), True)
        del flatten_0

        # pd_op.matmul: (1x1x21x21xf32) <- (1x1x21x64xf32, 1x1x21x64xf32)
        matmul_4 = paddle._C_ops.matmul(scale_1, flatten_1, False, True)

        # pd_op.add: (1x1x21x21xf32) <- (1x1x21x21xf32, 1x1x1x21xf32)
        add_8 = paddle._C_ops.add(matmul_4, unsqueeze_0)

        # pd_op.softmax: (1x1x21x21xf32) <- (1x1x21x21xf32)
        softmax_0 = paddle._C_ops.softmax(add_8, -1)
        del add_8

        # pd_op.dropout: (1x1x21x21xf32, 1x1x21x21xui8) <- (1x1x21x21xf32, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_0, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x1x21x64xf32) <- (1x1x21x21xf32, 1x1x21x64xf32)
        matmul_5 = paddle._C_ops.matmul(dropout_2, flatten_2, False, False)

        # pd_op.transpose: (1x21x1x64xf32) <- (1x1x21x64xf32)
        transpose_3 = paddle._C_ops.transpose(matmul_5, [0, 2, 1, 3])
        del matmul_5

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_7 = [0, 0, 64]

        # pd_op.reshape: (1x21x64xf32) <- (1x21x1x64xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_3, full_int_array_7)

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_6 = paddle._C_ops.matmul(reshape_3, parameter_211, False, False)
        del parameter_211

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_9 = paddle._C_ops.add(matmul_6, parameter_210)
        del parameter_210

        # pd_op.dropout: (1x21x64xf32, 1x21x64xui8) <- (1x21x64xf32, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_9, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_9

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 1x21x64xf32)
        add_10 = paddle._C_ops.add(add_1, dropout_4)

        # pd_op.layer_norm: (1x21x64xf32, 1x21xf32, 1x21xf32) <- (1x21x64xf32, 64xf32, 64xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_10, parameter_205, parameter_204, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_204, parameter_205

        # pd_op.matmul: (1x21x256xf32) <- (1x21x64xf32, 64x256xf32)
        matmul_7 = paddle._C_ops.matmul(layer_norm_3, parameter_209, False, False)
        del parameter_209

        # pd_op.add: (1x21x256xf32) <- (1x21x256xf32, 256xf32)
        add_11 = paddle._C_ops.add(matmul_7, parameter_208)
        del parameter_208

        # pd_op.gelu: (1x21x256xf32) <- (1x21x256xf32)
        gelu_0 = paddle._C_ops.gelu(add_11, False)

        # pd_op.matmul: (1x21x64xf32) <- (1x21x256xf32, 256x64xf32)
        matmul_8 = paddle._C_ops.matmul(gelu_0, parameter_207, False, False)
        del parameter_207

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_12 = paddle._C_ops.add(matmul_8, parameter_206)
        del parameter_206

        # pd_op.dropout: (1x21x64xf32, 1x21x64xui8) <- (1x21x64xf32, None, 1xf32)
        dropout_6, dropout_7 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_12, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_12

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 1x21x64xf32)
        add_13 = paddle._C_ops.add(layer_norm_3, dropout_6)

        # pd_op.layer_norm: (1x21x64xf32, 1x21xf32, 1x21xf32) <- (1x21x64xf32, 64xf32, 64xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_13, parameter_203, parameter_202, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_202, parameter_203

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_9 = paddle._C_ops.matmul(layer_norm_6, parameter_201, False, False)
        del parameter_201

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_14 = paddle._C_ops.add(matmul_9, parameter_200)
        del parameter_200

        # pd_op.reshape: (1x21x1x64xf32) <- (1x21x64xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(add_14, full_int_array_1)

        # pd_op.transpose: (1x1x21x64xf32) <- (1x21x1x64xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_4, [0, 2, 1, 3])
        del reshape_4

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_6, parameter_199, False, False)
        del parameter_199

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_15 = paddle._C_ops.add(matmul_10, parameter_198)
        del parameter_198

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_11 = paddle._C_ops.matmul(layer_norm_6, parameter_197, False, False)
        del parameter_197

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_16 = paddle._C_ops.add(matmul_11, parameter_196)
        del parameter_196

        # pd_op.reshape: (1x21x1x64xf32) <- (1x21x64xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(add_15, full_int_array_1)

        # pd_op.transpose: (1x1x21x64xf32) <- (1x21x1x64xf32)
        transpose_5 = paddle._C_ops.transpose(reshape_5, [0, 2, 1, 3])
        del reshape_5

        # pd_op.reshape: (1x21x1x64xf32) <- (1x21x64xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(add_16, full_int_array_1)

        # pd_op.transpose: (1x1x21x64xf32) <- (1x21x1x64xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_6, [0, 2, 1, 3])
        del reshape_6

        # pd_op.slice: (21x32xf32) <- (128x32xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            parameter_21, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_21

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_266 = slice_2

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_267 = slice_2

        # pd_op.slice: (21x32xf32) <- (128x32xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            parameter_20, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_20

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_268 = slice_3

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_269 = slice_3

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_6 = paddle._C_ops.strided_slice(
            transpose_4, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_7 = paddle._C_ops.strided_slice(
            transpose_4, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_12 = paddle._C_ops.multiply(strided_slice_6, slice_3)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_13 = paddle._C_ops.multiply(strided_slice_7, slice_2)

        # pd_op.subtract: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        subtract_3 = paddle._C_ops.subtract(multiply_12, multiply_13)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_14 = paddle._C_ops.multiply(strided_slice_6, slice_2)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_15 = paddle._C_ops.multiply(strided_slice_7, slice_3)

        # pd_op.add: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        add_17 = paddle._C_ops.add(multiply_14, multiply_15)

        # builtin.combine: ([1x1x21x32xf32, 1x1x21x32xf32]) <- (1x1x21x32xf32, 1x1x21x32xf32)
        combine_3 = [subtract_3, add_17]

        # pd_op.stack: (1x1x21x32x2xf32) <- ([1x1x21x32xf32, 1x1x21x32xf32])
        stack_3 = paddle._C_ops.stack(combine_3, -1)
        del combine_3

        # pd_op.flatten: (1x1x21x64xf32) <- (1x1x21x32x2xf32)
        flatten_3 = paddle._C_ops.flatten(stack_3, 3, 4)

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_8 = paddle._C_ops.strided_slice(
            transpose_5, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_9 = paddle._C_ops.strided_slice(
            transpose_5, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_16 = paddle._C_ops.multiply(strided_slice_8, slice_3)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_17 = paddle._C_ops.multiply(strided_slice_9, slice_2)

        # pd_op.subtract: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        subtract_4 = paddle._C_ops.subtract(multiply_16, multiply_17)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_18 = paddle._C_ops.multiply(strided_slice_8, slice_2)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_19 = paddle._C_ops.multiply(strided_slice_9, slice_3)

        # pd_op.add: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        add_18 = paddle._C_ops.add(multiply_18, multiply_19)

        # builtin.combine: ([1x1x21x32xf32, 1x1x21x32xf32]) <- (1x1x21x32xf32, 1x1x21x32xf32)
        combine_4 = [subtract_4, add_18]

        # pd_op.stack: (1x1x21x32x2xf32) <- ([1x1x21x32xf32, 1x1x21x32xf32])
        stack_4 = paddle._C_ops.stack(combine_4, -1)
        del combine_4

        # pd_op.flatten: (1x1x21x64xf32) <- (1x1x21x32x2xf32)
        flatten_4 = paddle._C_ops.flatten(stack_4, 3, 4)

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_10 = paddle._C_ops.strided_slice(
            transpose_6, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_11 = paddle._C_ops.strided_slice(
            transpose_6, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_20 = paddle._C_ops.multiply(strided_slice_10, slice_3)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_21 = paddle._C_ops.multiply(strided_slice_11, slice_2)

        # pd_op.subtract: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        subtract_5 = paddle._C_ops.subtract(multiply_20, multiply_21)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_22 = paddle._C_ops.multiply(strided_slice_10, slice_2)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_23 = paddle._C_ops.multiply(strided_slice_11, slice_3)

        # pd_op.add: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        add_19 = paddle._C_ops.add(multiply_22, multiply_23)

        # builtin.combine: ([1x1x21x32xf32, 1x1x21x32xf32]) <- (1x1x21x32xf32, 1x1x21x32xf32)
        combine_5 = [subtract_5, add_19]

        # pd_op.stack: (1x1x21x32x2xf32) <- ([1x1x21x32xf32, 1x1x21x32xf32])
        stack_5 = paddle._C_ops.stack(combine_5, -1)
        del combine_5

        # pd_op.flatten: (1x1x21x64xf32) <- (1x1x21x32x2xf32)
        flatten_5 = paddle._C_ops.flatten(stack_5, 3, 4)

        # pd_op.scale: (1x1x21x64xf32) <- (1x1x21x64xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(flatten_3, full_3, float("0"), True)
        del flatten_3

        # pd_op.matmul: (1x1x21x21xf32) <- (1x1x21x64xf32, 1x1x21x64xf32)
        matmul_12 = paddle._C_ops.matmul(scale_2, flatten_4, False, True)

        # pd_op.add: (1x1x21x21xf32) <- (1x1x21x21xf32, 1x1x1x21xf32)
        add_20 = paddle._C_ops.add(matmul_12, unsqueeze_0)

        # pd_op.softmax: (1x1x21x21xf32) <- (1x1x21x21xf32)
        softmax_1 = paddle._C_ops.softmax(add_20, -1)
        del add_20

        # pd_op.dropout: (1x1x21x21xf32, 1x1x21x21xui8) <- (1x1x21x21xf32, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_1, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x1x21x64xf32) <- (1x1x21x21xf32, 1x1x21x64xf32)
        matmul_13 = paddle._C_ops.matmul(dropout_8, flatten_5, False, False)

        # pd_op.transpose: (1x21x1x64xf32) <- (1x1x21x64xf32)
        transpose_7 = paddle._C_ops.transpose(matmul_13, [0, 2, 1, 3])
        del matmul_13

        # pd_op.reshape: (1x21x64xf32) <- (1x21x1x64xf32, 3xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_7, full_int_array_7)

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_14 = paddle._C_ops.matmul(reshape_7, parameter_195, False, False)
        del parameter_195

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_21 = paddle._C_ops.add(matmul_14, parameter_194)
        del parameter_194

        # pd_op.dropout: (1x21x64xf32, 1x21x64xui8) <- (1x21x64xf32, None, 1xf32)
        dropout_10, dropout_11 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_21, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_21

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 1x21x64xf32)
        add_22 = paddle._C_ops.add(layer_norm_6, dropout_10)

        # pd_op.layer_norm: (1x21x64xf32, 1x21xf32, 1x21xf32) <- (1x21x64xf32, 64xf32, 64xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_22, parameter_189, parameter_188, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_188, parameter_189

        # pd_op.matmul: (1x21x256xf32) <- (1x21x64xf32, 64x256xf32)
        matmul_15 = paddle._C_ops.matmul(layer_norm_9, parameter_193, False, False)
        del parameter_193

        # pd_op.add: (1x21x256xf32) <- (1x21x256xf32, 256xf32)
        add_23 = paddle._C_ops.add(matmul_15, parameter_192)
        del parameter_192

        # pd_op.gelu: (1x21x256xf32) <- (1x21x256xf32)
        gelu_1 = paddle._C_ops.gelu(add_23, False)

        # pd_op.matmul: (1x21x64xf32) <- (1x21x256xf32, 256x64xf32)
        matmul_16 = paddle._C_ops.matmul(gelu_1, parameter_191, False, False)
        del parameter_191

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_24 = paddle._C_ops.add(matmul_16, parameter_190)
        del parameter_190

        # pd_op.dropout: (1x21x64xf32, 1x21x64xui8) <- (1x21x64xf32, None, 1xf32)
        dropout_12, dropout_13 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_24, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_24

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 1x21x64xf32)
        add_25 = paddle._C_ops.add(layer_norm_9, dropout_12)

        # pd_op.layer_norm: (1x21x64xf32, 1x21xf32, 1x21xf32) <- (1x21x64xf32, 64xf32, 64xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_25, parameter_187, parameter_186, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_186, parameter_187

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_17 = paddle._C_ops.matmul(layer_norm_12, parameter_185, False, False)
        del parameter_185

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_26 = paddle._C_ops.add(matmul_17, parameter_184)
        del parameter_184

        # pd_op.reshape: (1x21x1x64xf32) <- (1x21x64xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(add_26, full_int_array_1)

        # pd_op.transpose: (1x1x21x64xf32) <- (1x21x1x64xf32)
        transpose_8 = paddle._C_ops.transpose(reshape_8, [0, 2, 1, 3])
        del reshape_8

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_18 = paddle._C_ops.matmul(layer_norm_12, parameter_183, False, False)
        del parameter_183

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_27 = paddle._C_ops.add(matmul_18, parameter_182)
        del parameter_182

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_19 = paddle._C_ops.matmul(layer_norm_12, parameter_181, False, False)
        del parameter_181

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_28 = paddle._C_ops.add(matmul_19, parameter_180)
        del parameter_180

        # pd_op.reshape: (1x21x1x64xf32) <- (1x21x64xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(add_27, full_int_array_1)

        # pd_op.transpose: (1x1x21x64xf32) <- (1x21x1x64xf32)
        transpose_9 = paddle._C_ops.transpose(reshape_9, [0, 2, 1, 3])
        del reshape_9

        # pd_op.reshape: (1x21x1x64xf32) <- (1x21x64xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(add_28, full_int_array_1)

        # pd_op.transpose: (1x1x21x64xf32) <- (1x21x1x64xf32)
        transpose_10 = paddle._C_ops.transpose(reshape_10, [0, 2, 1, 3])
        del reshape_10

        # pd_op.slice: (21x32xf32) <- (128x32xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            parameter_19, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_19

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_270 = slice_4

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_271 = slice_4

        # pd_op.slice: (21x32xf32) <- (128x32xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            parameter_18, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_18

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_272 = slice_5

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_273 = slice_5

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_12 = paddle._C_ops.strided_slice(
            transpose_8, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_13 = paddle._C_ops.strided_slice(
            transpose_8, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_24 = paddle._C_ops.multiply(strided_slice_12, slice_5)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_25 = paddle._C_ops.multiply(strided_slice_13, slice_4)

        # pd_op.subtract: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        subtract_6 = paddle._C_ops.subtract(multiply_24, multiply_25)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_26 = paddle._C_ops.multiply(strided_slice_12, slice_4)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_27 = paddle._C_ops.multiply(strided_slice_13, slice_5)

        # pd_op.add: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        add_29 = paddle._C_ops.add(multiply_26, multiply_27)

        # builtin.combine: ([1x1x21x32xf32, 1x1x21x32xf32]) <- (1x1x21x32xf32, 1x1x21x32xf32)
        combine_6 = [subtract_6, add_29]

        # pd_op.stack: (1x1x21x32x2xf32) <- ([1x1x21x32xf32, 1x1x21x32xf32])
        stack_6 = paddle._C_ops.stack(combine_6, -1)
        del combine_6

        # pd_op.flatten: (1x1x21x64xf32) <- (1x1x21x32x2xf32)
        flatten_6 = paddle._C_ops.flatten(stack_6, 3, 4)

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_14 = paddle._C_ops.strided_slice(
            transpose_9, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_15 = paddle._C_ops.strided_slice(
            transpose_9, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_28 = paddle._C_ops.multiply(strided_slice_14, slice_5)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_29 = paddle._C_ops.multiply(strided_slice_15, slice_4)

        # pd_op.subtract: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        subtract_7 = paddle._C_ops.subtract(multiply_28, multiply_29)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_30 = paddle._C_ops.multiply(strided_slice_14, slice_4)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_31 = paddle._C_ops.multiply(strided_slice_15, slice_5)

        # pd_op.add: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        add_30 = paddle._C_ops.add(multiply_30, multiply_31)

        # builtin.combine: ([1x1x21x32xf32, 1x1x21x32xf32]) <- (1x1x21x32xf32, 1x1x21x32xf32)
        combine_7 = [subtract_7, add_30]

        # pd_op.stack: (1x1x21x32x2xf32) <- ([1x1x21x32xf32, 1x1x21x32xf32])
        stack_7 = paddle._C_ops.stack(combine_7, -1)
        del combine_7

        # pd_op.flatten: (1x1x21x64xf32) <- (1x1x21x32x2xf32)
        flatten_7 = paddle._C_ops.flatten(stack_7, 3, 4)

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_16 = paddle._C_ops.strided_slice(
            transpose_10, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_17 = paddle._C_ops.strided_slice(
            transpose_10, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_32 = paddle._C_ops.multiply(strided_slice_16, slice_5)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_33 = paddle._C_ops.multiply(strided_slice_17, slice_4)

        # pd_op.subtract: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        subtract_8 = paddle._C_ops.subtract(multiply_32, multiply_33)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_34 = paddle._C_ops.multiply(strided_slice_16, slice_4)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_35 = paddle._C_ops.multiply(strided_slice_17, slice_5)

        # pd_op.add: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        add_31 = paddle._C_ops.add(multiply_34, multiply_35)

        # builtin.combine: ([1x1x21x32xf32, 1x1x21x32xf32]) <- (1x1x21x32xf32, 1x1x21x32xf32)
        combine_8 = [subtract_8, add_31]

        # pd_op.stack: (1x1x21x32x2xf32) <- ([1x1x21x32xf32, 1x1x21x32xf32])
        stack_8 = paddle._C_ops.stack(combine_8, -1)
        del combine_8

        # pd_op.flatten: (1x1x21x64xf32) <- (1x1x21x32x2xf32)
        flatten_8 = paddle._C_ops.flatten(stack_8, 3, 4)

        # pd_op.scale: (1x1x21x64xf32) <- (1x1x21x64xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(flatten_6, full_3, float("0"), True)
        del flatten_6

        # pd_op.matmul: (1x1x21x21xf32) <- (1x1x21x64xf32, 1x1x21x64xf32)
        matmul_20 = paddle._C_ops.matmul(scale_3, flatten_7, False, True)

        # pd_op.add: (1x1x21x21xf32) <- (1x1x21x21xf32, 1x1x1x21xf32)
        add_32 = paddle._C_ops.add(matmul_20, unsqueeze_0)

        # pd_op.softmax: (1x1x21x21xf32) <- (1x1x21x21xf32)
        softmax_2 = paddle._C_ops.softmax(add_32, -1)
        del add_32

        # pd_op.dropout: (1x1x21x21xf32, 1x1x21x21xui8) <- (1x1x21x21xf32, None, 1xf32)
        dropout_14, dropout_15 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_2, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x1x21x64xf32) <- (1x1x21x21xf32, 1x1x21x64xf32)
        matmul_21 = paddle._C_ops.matmul(dropout_14, flatten_8, False, False)

        # pd_op.transpose: (1x21x1x64xf32) <- (1x1x21x64xf32)
        transpose_11 = paddle._C_ops.transpose(matmul_21, [0, 2, 1, 3])
        del matmul_21

        # pd_op.reshape: (1x21x64xf32) <- (1x21x1x64xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_11, full_int_array_7)

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_22 = paddle._C_ops.matmul(reshape_11, parameter_179, False, False)
        del parameter_179

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_33 = paddle._C_ops.add(matmul_22, parameter_178)
        del parameter_178

        # pd_op.dropout: (1x21x64xf32, 1x21x64xui8) <- (1x21x64xf32, None, 1xf32)
        dropout_16, dropout_17 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_33, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_33

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 1x21x64xf32)
        add_34 = paddle._C_ops.add(layer_norm_12, dropout_16)

        # pd_op.layer_norm: (1x21x64xf32, 1x21xf32, 1x21xf32) <- (1x21x64xf32, 64xf32, 64xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_34, parameter_173, parameter_172, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_172, parameter_173

        # pd_op.matmul: (1x21x256xf32) <- (1x21x64xf32, 64x256xf32)
        matmul_23 = paddle._C_ops.matmul(layer_norm_15, parameter_177, False, False)
        del parameter_177

        # pd_op.add: (1x21x256xf32) <- (1x21x256xf32, 256xf32)
        add_35 = paddle._C_ops.add(matmul_23, parameter_176)
        del parameter_176

        # pd_op.gelu: (1x21x256xf32) <- (1x21x256xf32)
        gelu_2 = paddle._C_ops.gelu(add_35, False)

        # pd_op.matmul: (1x21x64xf32) <- (1x21x256xf32, 256x64xf32)
        matmul_24 = paddle._C_ops.matmul(gelu_2, parameter_175, False, False)
        del parameter_175

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_36 = paddle._C_ops.add(matmul_24, parameter_174)
        del parameter_174

        # pd_op.dropout: (1x21x64xf32, 1x21x64xui8) <- (1x21x64xf32, None, 1xf32)
        dropout_18, dropout_19 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_36, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_36

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 1x21x64xf32)
        add_37 = paddle._C_ops.add(layer_norm_15, dropout_18)

        # pd_op.layer_norm: (1x21x64xf32, 1x21xf32, 1x21xf32) <- (1x21x64xf32, 64xf32, 64xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_37, parameter_171, parameter_170, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_170, parameter_171

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_25 = paddle._C_ops.matmul(layer_norm_18, parameter_169, False, False)
        del parameter_169

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_38 = paddle._C_ops.add(matmul_25, parameter_168)
        del parameter_168

        # pd_op.reshape: (1x21x1x64xf32) <- (1x21x64xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(add_38, full_int_array_1)

        # pd_op.transpose: (1x1x21x64xf32) <- (1x21x1x64xf32)
        transpose_12 = paddle._C_ops.transpose(reshape_12, [0, 2, 1, 3])
        del reshape_12

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_26 = paddle._C_ops.matmul(layer_norm_18, parameter_167, False, False)
        del parameter_167

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_39 = paddle._C_ops.add(matmul_26, parameter_166)
        del parameter_166

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_27 = paddle._C_ops.matmul(layer_norm_18, parameter_165, False, False)
        del parameter_165

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_40 = paddle._C_ops.add(matmul_27, parameter_164)
        del parameter_164

        # pd_op.reshape: (1x21x1x64xf32) <- (1x21x64xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(add_39, full_int_array_1)

        # pd_op.transpose: (1x1x21x64xf32) <- (1x21x1x64xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_13, [0, 2, 1, 3])
        del reshape_13

        # pd_op.reshape: (1x21x1x64xf32) <- (1x21x64xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(add_40, full_int_array_1)

        # pd_op.transpose: (1x1x21x64xf32) <- (1x21x1x64xf32)
        transpose_14 = paddle._C_ops.transpose(reshape_14, [0, 2, 1, 3])
        del reshape_14

        # pd_op.slice: (21x32xf32) <- (128x32xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            parameter_17, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_17

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_274 = slice_6

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_275 = slice_6

        # pd_op.slice: (21x32xf32) <- (128x32xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            parameter_16, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_16

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_276 = slice_7

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_277 = slice_7

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_18 = paddle._C_ops.strided_slice(
            transpose_12, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_19 = paddle._C_ops.strided_slice(
            transpose_12, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_36 = paddle._C_ops.multiply(strided_slice_18, slice_7)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_37 = paddle._C_ops.multiply(strided_slice_19, slice_6)

        # pd_op.subtract: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        subtract_9 = paddle._C_ops.subtract(multiply_36, multiply_37)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_38 = paddle._C_ops.multiply(strided_slice_18, slice_6)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_39 = paddle._C_ops.multiply(strided_slice_19, slice_7)

        # pd_op.add: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        add_41 = paddle._C_ops.add(multiply_38, multiply_39)

        # builtin.combine: ([1x1x21x32xf32, 1x1x21x32xf32]) <- (1x1x21x32xf32, 1x1x21x32xf32)
        combine_9 = [subtract_9, add_41]

        # pd_op.stack: (1x1x21x32x2xf32) <- ([1x1x21x32xf32, 1x1x21x32xf32])
        stack_9 = paddle._C_ops.stack(combine_9, -1)
        del combine_9

        # pd_op.flatten: (1x1x21x64xf32) <- (1x1x21x32x2xf32)
        flatten_9 = paddle._C_ops.flatten(stack_9, 3, 4)

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_20 = paddle._C_ops.strided_slice(
            transpose_13, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_21 = paddle._C_ops.strided_slice(
            transpose_13, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_40 = paddle._C_ops.multiply(strided_slice_20, slice_7)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_41 = paddle._C_ops.multiply(strided_slice_21, slice_6)

        # pd_op.subtract: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        subtract_10 = paddle._C_ops.subtract(multiply_40, multiply_41)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_42 = paddle._C_ops.multiply(strided_slice_20, slice_6)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_43 = paddle._C_ops.multiply(strided_slice_21, slice_7)

        # pd_op.add: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        add_42 = paddle._C_ops.add(multiply_42, multiply_43)

        # builtin.combine: ([1x1x21x32xf32, 1x1x21x32xf32]) <- (1x1x21x32xf32, 1x1x21x32xf32)
        combine_10 = [subtract_10, add_42]

        # pd_op.stack: (1x1x21x32x2xf32) <- ([1x1x21x32xf32, 1x1x21x32xf32])
        stack_10 = paddle._C_ops.stack(combine_10, -1)
        del combine_10

        # pd_op.flatten: (1x1x21x64xf32) <- (1x1x21x32x2xf32)
        flatten_10 = paddle._C_ops.flatten(stack_10, 3, 4)

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_22 = paddle._C_ops.strided_slice(
            transpose_14, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_23 = paddle._C_ops.strided_slice(
            transpose_14, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_44 = paddle._C_ops.multiply(strided_slice_22, slice_7)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_45 = paddle._C_ops.multiply(strided_slice_23, slice_6)

        # pd_op.subtract: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        subtract_11 = paddle._C_ops.subtract(multiply_44, multiply_45)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_46 = paddle._C_ops.multiply(strided_slice_22, slice_6)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_47 = paddle._C_ops.multiply(strided_slice_23, slice_7)

        # pd_op.add: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        add_43 = paddle._C_ops.add(multiply_46, multiply_47)

        # builtin.combine: ([1x1x21x32xf32, 1x1x21x32xf32]) <- (1x1x21x32xf32, 1x1x21x32xf32)
        combine_11 = [subtract_11, add_43]

        # pd_op.stack: (1x1x21x32x2xf32) <- ([1x1x21x32xf32, 1x1x21x32xf32])
        stack_11 = paddle._C_ops.stack(combine_11, -1)
        del combine_11

        # pd_op.flatten: (1x1x21x64xf32) <- (1x1x21x32x2xf32)
        flatten_11 = paddle._C_ops.flatten(stack_11, 3, 4)

        # pd_op.scale: (1x1x21x64xf32) <- (1x1x21x64xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(flatten_9, full_3, float("0"), True)
        del flatten_9

        # pd_op.matmul: (1x1x21x21xf32) <- (1x1x21x64xf32, 1x1x21x64xf32)
        matmul_28 = paddle._C_ops.matmul(scale_4, flatten_10, False, True)

        # pd_op.add: (1x1x21x21xf32) <- (1x1x21x21xf32, 1x1x1x21xf32)
        add_44 = paddle._C_ops.add(matmul_28, unsqueeze_0)

        # pd_op.softmax: (1x1x21x21xf32) <- (1x1x21x21xf32)
        softmax_3 = paddle._C_ops.softmax(add_44, -1)
        del add_44

        # pd_op.dropout: (1x1x21x21xf32, 1x1x21x21xui8) <- (1x1x21x21xf32, None, 1xf32)
        dropout_20, dropout_21 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_3, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x1x21x64xf32) <- (1x1x21x21xf32, 1x1x21x64xf32)
        matmul_29 = paddle._C_ops.matmul(dropout_20, flatten_11, False, False)

        # pd_op.transpose: (1x21x1x64xf32) <- (1x1x21x64xf32)
        transpose_15 = paddle._C_ops.transpose(matmul_29, [0, 2, 1, 3])
        del matmul_29

        # pd_op.reshape: (1x21x64xf32) <- (1x21x1x64xf32, 3xi64)
        reshape_15 = paddle._C_ops.reshape(transpose_15, full_int_array_7)

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_30 = paddle._C_ops.matmul(reshape_15, parameter_163, False, False)
        del parameter_163

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_45 = paddle._C_ops.add(matmul_30, parameter_162)
        del parameter_162

        # pd_op.dropout: (1x21x64xf32, 1x21x64xui8) <- (1x21x64xf32, None, 1xf32)
        dropout_22, dropout_23 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_45, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_45

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 1x21x64xf32)
        add_46 = paddle._C_ops.add(layer_norm_18, dropout_22)

        # pd_op.layer_norm: (1x21x64xf32, 1x21xf32, 1x21xf32) <- (1x21x64xf32, 64xf32, 64xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_46, parameter_157, parameter_156, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_156, parameter_157

        # pd_op.matmul: (1x21x256xf32) <- (1x21x64xf32, 64x256xf32)
        matmul_31 = paddle._C_ops.matmul(layer_norm_21, parameter_161, False, False)
        del parameter_161

        # pd_op.add: (1x21x256xf32) <- (1x21x256xf32, 256xf32)
        add_47 = paddle._C_ops.add(matmul_31, parameter_160)
        del parameter_160

        # pd_op.gelu: (1x21x256xf32) <- (1x21x256xf32)
        gelu_3 = paddle._C_ops.gelu(add_47, False)

        # pd_op.matmul: (1x21x64xf32) <- (1x21x256xf32, 256x64xf32)
        matmul_32 = paddle._C_ops.matmul(gelu_3, parameter_159, False, False)
        del parameter_159

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_48 = paddle._C_ops.add(matmul_32, parameter_158)
        del parameter_158

        # pd_op.dropout: (1x21x64xf32, 1x21x64xui8) <- (1x21x64xf32, None, 1xf32)
        dropout_24, dropout_25 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_48, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_48

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 1x21x64xf32)
        add_49 = paddle._C_ops.add(layer_norm_21, dropout_24)

        # pd_op.layer_norm: (1x21x64xf32, 1x21xf32, 1x21xf32) <- (1x21x64xf32, 64xf32, 64xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_49, parameter_155, parameter_154, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_154, parameter_155

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_33 = paddle._C_ops.matmul(layer_norm_24, parameter_153, False, False)
        del parameter_153

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_50 = paddle._C_ops.add(matmul_33, parameter_152)
        del parameter_152

        # pd_op.reshape: (1x21x1x64xf32) <- (1x21x64xf32, 4xi64)
        reshape_16 = paddle._C_ops.reshape(add_50, full_int_array_1)

        # pd_op.transpose: (1x1x21x64xf32) <- (1x21x1x64xf32)
        transpose_16 = paddle._C_ops.transpose(reshape_16, [0, 2, 1, 3])
        del reshape_16

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_34 = paddle._C_ops.matmul(layer_norm_24, parameter_151, False, False)
        del parameter_151

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_51 = paddle._C_ops.add(matmul_34, parameter_150)
        del parameter_150

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_35 = paddle._C_ops.matmul(layer_norm_24, parameter_149, False, False)
        del parameter_149

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_52 = paddle._C_ops.add(matmul_35, parameter_148)
        del parameter_148

        # pd_op.reshape: (1x21x1x64xf32) <- (1x21x64xf32, 4xi64)
        reshape_17 = paddle._C_ops.reshape(add_51, full_int_array_1)

        # pd_op.transpose: (1x1x21x64xf32) <- (1x21x1x64xf32)
        transpose_17 = paddle._C_ops.transpose(reshape_17, [0, 2, 1, 3])
        del reshape_17

        # pd_op.reshape: (1x21x1x64xf32) <- (1x21x64xf32, 4xi64)
        reshape_18 = paddle._C_ops.reshape(add_52, full_int_array_1)

        # pd_op.transpose: (1x1x21x64xf32) <- (1x21x1x64xf32)
        transpose_18 = paddle._C_ops.transpose(reshape_18, [0, 2, 1, 3])
        del reshape_18

        # pd_op.slice: (21x32xf32) <- (128x32xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            parameter_15, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_15

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_278 = slice_8

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_279 = slice_8

        # pd_op.slice: (21x32xf32) <- (128x32xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            parameter_14, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_14

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_280 = slice_9

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_281 = slice_9

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_24 = paddle._C_ops.strided_slice(
            transpose_16, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_25 = paddle._C_ops.strided_slice(
            transpose_16, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_48 = paddle._C_ops.multiply(strided_slice_24, slice_9)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_49 = paddle._C_ops.multiply(strided_slice_25, slice_8)

        # pd_op.subtract: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        subtract_12 = paddle._C_ops.subtract(multiply_48, multiply_49)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_50 = paddle._C_ops.multiply(strided_slice_24, slice_8)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_51 = paddle._C_ops.multiply(strided_slice_25, slice_9)

        # pd_op.add: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        add_53 = paddle._C_ops.add(multiply_50, multiply_51)

        # builtin.combine: ([1x1x21x32xf32, 1x1x21x32xf32]) <- (1x1x21x32xf32, 1x1x21x32xf32)
        combine_12 = [subtract_12, add_53]

        # pd_op.stack: (1x1x21x32x2xf32) <- ([1x1x21x32xf32, 1x1x21x32xf32])
        stack_12 = paddle._C_ops.stack(combine_12, -1)
        del combine_12

        # pd_op.flatten: (1x1x21x64xf32) <- (1x1x21x32x2xf32)
        flatten_12 = paddle._C_ops.flatten(stack_12, 3, 4)

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_26 = paddle._C_ops.strided_slice(
            transpose_17, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_27 = paddle._C_ops.strided_slice(
            transpose_17, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_52 = paddle._C_ops.multiply(strided_slice_26, slice_9)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_53 = paddle._C_ops.multiply(strided_slice_27, slice_8)

        # pd_op.subtract: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        subtract_13 = paddle._C_ops.subtract(multiply_52, multiply_53)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_54 = paddle._C_ops.multiply(strided_slice_26, slice_8)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_55 = paddle._C_ops.multiply(strided_slice_27, slice_9)

        # pd_op.add: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        add_54 = paddle._C_ops.add(multiply_54, multiply_55)

        # builtin.combine: ([1x1x21x32xf32, 1x1x21x32xf32]) <- (1x1x21x32xf32, 1x1x21x32xf32)
        combine_13 = [subtract_13, add_54]

        # pd_op.stack: (1x1x21x32x2xf32) <- ([1x1x21x32xf32, 1x1x21x32xf32])
        stack_13 = paddle._C_ops.stack(combine_13, -1)
        del combine_13

        # pd_op.flatten: (1x1x21x64xf32) <- (1x1x21x32x2xf32)
        flatten_13 = paddle._C_ops.flatten(stack_13, 3, 4)

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_28 = paddle._C_ops.strided_slice(
            transpose_18, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_29 = paddle._C_ops.strided_slice(
            transpose_18, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_56 = paddle._C_ops.multiply(strided_slice_28, slice_9)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_57 = paddle._C_ops.multiply(strided_slice_29, slice_8)

        # pd_op.subtract: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        subtract_14 = paddle._C_ops.subtract(multiply_56, multiply_57)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_58 = paddle._C_ops.multiply(strided_slice_28, slice_8)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_59 = paddle._C_ops.multiply(strided_slice_29, slice_9)

        # pd_op.add: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        add_55 = paddle._C_ops.add(multiply_58, multiply_59)

        # builtin.combine: ([1x1x21x32xf32, 1x1x21x32xf32]) <- (1x1x21x32xf32, 1x1x21x32xf32)
        combine_14 = [subtract_14, add_55]

        # pd_op.stack: (1x1x21x32x2xf32) <- ([1x1x21x32xf32, 1x1x21x32xf32])
        stack_14 = paddle._C_ops.stack(combine_14, -1)
        del combine_14

        # pd_op.flatten: (1x1x21x64xf32) <- (1x1x21x32x2xf32)
        flatten_14 = paddle._C_ops.flatten(stack_14, 3, 4)

        # pd_op.scale: (1x1x21x64xf32) <- (1x1x21x64xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(flatten_12, full_3, float("0"), True)
        del flatten_12

        # pd_op.matmul: (1x1x21x21xf32) <- (1x1x21x64xf32, 1x1x21x64xf32)
        matmul_36 = paddle._C_ops.matmul(scale_5, flatten_13, False, True)

        # pd_op.add: (1x1x21x21xf32) <- (1x1x21x21xf32, 1x1x1x21xf32)
        add_56 = paddle._C_ops.add(matmul_36, unsqueeze_0)

        # pd_op.softmax: (1x1x21x21xf32) <- (1x1x21x21xf32)
        softmax_4 = paddle._C_ops.softmax(add_56, -1)
        del add_56

        # pd_op.dropout: (1x1x21x21xf32, 1x1x21x21xui8) <- (1x1x21x21xf32, None, 1xf32)
        dropout_26, dropout_27 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_4, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x1x21x64xf32) <- (1x1x21x21xf32, 1x1x21x64xf32)
        matmul_37 = paddle._C_ops.matmul(dropout_26, flatten_14, False, False)

        # pd_op.transpose: (1x21x1x64xf32) <- (1x1x21x64xf32)
        transpose_19 = paddle._C_ops.transpose(matmul_37, [0, 2, 1, 3])
        del matmul_37

        # pd_op.reshape: (1x21x64xf32) <- (1x21x1x64xf32, 3xi64)
        reshape_19 = paddle._C_ops.reshape(transpose_19, full_int_array_7)

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_38 = paddle._C_ops.matmul(reshape_19, parameter_147, False, False)
        del parameter_147

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_57 = paddle._C_ops.add(matmul_38, parameter_146)
        del parameter_146

        # pd_op.dropout: (1x21x64xf32, 1x21x64xui8) <- (1x21x64xf32, None, 1xf32)
        dropout_28, dropout_29 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_57, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_57

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 1x21x64xf32)
        add_58 = paddle._C_ops.add(layer_norm_24, dropout_28)

        # pd_op.layer_norm: (1x21x64xf32, 1x21xf32, 1x21xf32) <- (1x21x64xf32, 64xf32, 64xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_58, parameter_141, parameter_140, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_140, parameter_141

        # pd_op.matmul: (1x21x256xf32) <- (1x21x64xf32, 64x256xf32)
        matmul_39 = paddle._C_ops.matmul(layer_norm_27, parameter_145, False, False)
        del parameter_145

        # pd_op.add: (1x21x256xf32) <- (1x21x256xf32, 256xf32)
        add_59 = paddle._C_ops.add(matmul_39, parameter_144)
        del parameter_144

        # pd_op.gelu: (1x21x256xf32) <- (1x21x256xf32)
        gelu_4 = paddle._C_ops.gelu(add_59, False)

        # pd_op.matmul: (1x21x64xf32) <- (1x21x256xf32, 256x64xf32)
        matmul_40 = paddle._C_ops.matmul(gelu_4, parameter_143, False, False)
        del parameter_143

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_60 = paddle._C_ops.add(matmul_40, parameter_142)
        del parameter_142

        # pd_op.dropout: (1x21x64xf32, 1x21x64xui8) <- (1x21x64xf32, None, 1xf32)
        dropout_30, dropout_31 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_60, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_60

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 1x21x64xf32)
        add_61 = paddle._C_ops.add(layer_norm_27, dropout_30)

        # pd_op.layer_norm: (1x21x64xf32, 1x21xf32, 1x21xf32) <- (1x21x64xf32, 64xf32, 64xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_61, parameter_139, parameter_138, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_138, parameter_139

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_41 = paddle._C_ops.matmul(layer_norm_30, parameter_137, False, False)
        del parameter_137

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_62 = paddle._C_ops.add(matmul_41, parameter_136)
        del parameter_136

        # pd_op.reshape: (1x21x1x64xf32) <- (1x21x64xf32, 4xi64)
        reshape_20 = paddle._C_ops.reshape(add_62, full_int_array_1)

        # pd_op.transpose: (1x1x21x64xf32) <- (1x21x1x64xf32)
        transpose_20 = paddle._C_ops.transpose(reshape_20, [0, 2, 1, 3])
        del reshape_20

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_42 = paddle._C_ops.matmul(layer_norm_30, parameter_135, False, False)
        del parameter_135

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_63 = paddle._C_ops.add(matmul_42, parameter_134)
        del parameter_134

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_43 = paddle._C_ops.matmul(layer_norm_30, parameter_133, False, False)
        del parameter_133

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_64 = paddle._C_ops.add(matmul_43, parameter_132)
        del parameter_132

        # pd_op.reshape: (1x21x1x64xf32) <- (1x21x64xf32, 4xi64)
        reshape_21 = paddle._C_ops.reshape(add_63, full_int_array_1)

        # pd_op.transpose: (1x1x21x64xf32) <- (1x21x1x64xf32)
        transpose_21 = paddle._C_ops.transpose(reshape_21, [0, 2, 1, 3])
        del reshape_21

        # pd_op.reshape: (1x21x1x64xf32) <- (1x21x64xf32, 4xi64)
        reshape_22 = paddle._C_ops.reshape(add_64, full_int_array_1)

        # pd_op.transpose: (1x1x21x64xf32) <- (1x21x1x64xf32)
        transpose_22 = paddle._C_ops.transpose(reshape_22, [0, 2, 1, 3])
        del reshape_22

        # pd_op.slice: (21x32xf32) <- (128x32xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            parameter_13, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_13

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_282 = slice_10

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_283 = slice_10

        # pd_op.slice: (21x32xf32) <- (128x32xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            parameter_12, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_12

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_284 = slice_11

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_285 = slice_11

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_30 = paddle._C_ops.strided_slice(
            transpose_20, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_31 = paddle._C_ops.strided_slice(
            transpose_20, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_60 = paddle._C_ops.multiply(strided_slice_30, slice_11)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_61 = paddle._C_ops.multiply(strided_slice_31, slice_10)

        # pd_op.subtract: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        subtract_15 = paddle._C_ops.subtract(multiply_60, multiply_61)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_62 = paddle._C_ops.multiply(strided_slice_30, slice_10)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_63 = paddle._C_ops.multiply(strided_slice_31, slice_11)

        # pd_op.add: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        add_65 = paddle._C_ops.add(multiply_62, multiply_63)

        # builtin.combine: ([1x1x21x32xf32, 1x1x21x32xf32]) <- (1x1x21x32xf32, 1x1x21x32xf32)
        combine_15 = [subtract_15, add_65]

        # pd_op.stack: (1x1x21x32x2xf32) <- ([1x1x21x32xf32, 1x1x21x32xf32])
        stack_15 = paddle._C_ops.stack(combine_15, -1)
        del combine_15

        # pd_op.flatten: (1x1x21x64xf32) <- (1x1x21x32x2xf32)
        flatten_15 = paddle._C_ops.flatten(stack_15, 3, 4)

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_32 = paddle._C_ops.strided_slice(
            transpose_21, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_33 = paddle._C_ops.strided_slice(
            transpose_21, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_64 = paddle._C_ops.multiply(strided_slice_32, slice_11)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_65 = paddle._C_ops.multiply(strided_slice_33, slice_10)

        # pd_op.subtract: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        subtract_16 = paddle._C_ops.subtract(multiply_64, multiply_65)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_66 = paddle._C_ops.multiply(strided_slice_32, slice_10)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_67 = paddle._C_ops.multiply(strided_slice_33, slice_11)

        # pd_op.add: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        add_66 = paddle._C_ops.add(multiply_66, multiply_67)

        # builtin.combine: ([1x1x21x32xf32, 1x1x21x32xf32]) <- (1x1x21x32xf32, 1x1x21x32xf32)
        combine_16 = [subtract_16, add_66]

        # pd_op.stack: (1x1x21x32x2xf32) <- ([1x1x21x32xf32, 1x1x21x32xf32])
        stack_16 = paddle._C_ops.stack(combine_16, -1)
        del combine_16

        # pd_op.flatten: (1x1x21x64xf32) <- (1x1x21x32x2xf32)
        flatten_16 = paddle._C_ops.flatten(stack_16, 3, 4)

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_34 = paddle._C_ops.strided_slice(
            transpose_22, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_35 = paddle._C_ops.strided_slice(
            transpose_22, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_68 = paddle._C_ops.multiply(strided_slice_34, slice_11)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_69 = paddle._C_ops.multiply(strided_slice_35, slice_10)

        # pd_op.subtract: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        subtract_17 = paddle._C_ops.subtract(multiply_68, multiply_69)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_70 = paddle._C_ops.multiply(strided_slice_34, slice_10)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_71 = paddle._C_ops.multiply(strided_slice_35, slice_11)

        # pd_op.add: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        add_67 = paddle._C_ops.add(multiply_70, multiply_71)

        # builtin.combine: ([1x1x21x32xf32, 1x1x21x32xf32]) <- (1x1x21x32xf32, 1x1x21x32xf32)
        combine_17 = [subtract_17, add_67]

        # pd_op.stack: (1x1x21x32x2xf32) <- ([1x1x21x32xf32, 1x1x21x32xf32])
        stack_17 = paddle._C_ops.stack(combine_17, -1)
        del combine_17

        # pd_op.flatten: (1x1x21x64xf32) <- (1x1x21x32x2xf32)
        flatten_17 = paddle._C_ops.flatten(stack_17, 3, 4)

        # pd_op.scale: (1x1x21x64xf32) <- (1x1x21x64xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(flatten_15, full_3, float("0"), True)
        del flatten_15

        # pd_op.matmul: (1x1x21x21xf32) <- (1x1x21x64xf32, 1x1x21x64xf32)
        matmul_44 = paddle._C_ops.matmul(scale_6, flatten_16, False, True)

        # pd_op.add: (1x1x21x21xf32) <- (1x1x21x21xf32, 1x1x1x21xf32)
        add_68 = paddle._C_ops.add(matmul_44, unsqueeze_0)

        # pd_op.softmax: (1x1x21x21xf32) <- (1x1x21x21xf32)
        softmax_5 = paddle._C_ops.softmax(add_68, -1)
        del add_68

        # pd_op.dropout: (1x1x21x21xf32, 1x1x21x21xui8) <- (1x1x21x21xf32, None, 1xf32)
        dropout_32, dropout_33 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_5, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x1x21x64xf32) <- (1x1x21x21xf32, 1x1x21x64xf32)
        matmul_45 = paddle._C_ops.matmul(dropout_32, flatten_17, False, False)

        # pd_op.transpose: (1x21x1x64xf32) <- (1x1x21x64xf32)
        transpose_23 = paddle._C_ops.transpose(matmul_45, [0, 2, 1, 3])
        del matmul_45

        # pd_op.reshape: (1x21x64xf32) <- (1x21x1x64xf32, 3xi64)
        reshape_23 = paddle._C_ops.reshape(transpose_23, full_int_array_7)

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_46 = paddle._C_ops.matmul(reshape_23, parameter_131, False, False)
        del parameter_131

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_69 = paddle._C_ops.add(matmul_46, parameter_130)
        del parameter_130

        # pd_op.dropout: (1x21x64xf32, 1x21x64xui8) <- (1x21x64xf32, None, 1xf32)
        dropout_34, dropout_35 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_69, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_69

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 1x21x64xf32)
        add_70 = paddle._C_ops.add(layer_norm_30, dropout_34)

        # pd_op.layer_norm: (1x21x64xf32, 1x21xf32, 1x21xf32) <- (1x21x64xf32, 64xf32, 64xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_70, parameter_125, parameter_124, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_124, parameter_125

        # pd_op.matmul: (1x21x256xf32) <- (1x21x64xf32, 64x256xf32)
        matmul_47 = paddle._C_ops.matmul(layer_norm_33, parameter_129, False, False)
        del parameter_129

        # pd_op.add: (1x21x256xf32) <- (1x21x256xf32, 256xf32)
        add_71 = paddle._C_ops.add(matmul_47, parameter_128)
        del parameter_128

        # pd_op.gelu: (1x21x256xf32) <- (1x21x256xf32)
        gelu_5 = paddle._C_ops.gelu(add_71, False)

        # pd_op.matmul: (1x21x64xf32) <- (1x21x256xf32, 256x64xf32)
        matmul_48 = paddle._C_ops.matmul(gelu_5, parameter_127, False, False)
        del parameter_127

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_72 = paddle._C_ops.add(matmul_48, parameter_126)
        del parameter_126

        # pd_op.dropout: (1x21x64xf32, 1x21x64xui8) <- (1x21x64xf32, None, 1xf32)
        dropout_36, dropout_37 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_72, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_72

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 1x21x64xf32)
        add_73 = paddle._C_ops.add(layer_norm_33, dropout_36)

        # pd_op.layer_norm: (1x21x64xf32, 1x21xf32, 1x21xf32) <- (1x21x64xf32, 64xf32, 64xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_73, parameter_123, parameter_122, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_122, parameter_123

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_49 = paddle._C_ops.matmul(layer_norm_36, parameter_121, False, False)
        del parameter_121

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_74 = paddle._C_ops.add(matmul_49, parameter_120)
        del parameter_120

        # pd_op.reshape: (1x21x1x64xf32) <- (1x21x64xf32, 4xi64)
        reshape_24 = paddle._C_ops.reshape(add_74, full_int_array_1)

        # pd_op.transpose: (1x1x21x64xf32) <- (1x21x1x64xf32)
        transpose_24 = paddle._C_ops.transpose(reshape_24, [0, 2, 1, 3])
        del reshape_24

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_50 = paddle._C_ops.matmul(layer_norm_36, parameter_119, False, False)
        del parameter_119

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_75 = paddle._C_ops.add(matmul_50, parameter_118)
        del parameter_118

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_51 = paddle._C_ops.matmul(layer_norm_36, parameter_117, False, False)
        del parameter_117

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_76 = paddle._C_ops.add(matmul_51, parameter_116)
        del parameter_116

        # pd_op.reshape: (1x21x1x64xf32) <- (1x21x64xf32, 4xi64)
        reshape_25 = paddle._C_ops.reshape(add_75, full_int_array_1)

        # pd_op.transpose: (1x1x21x64xf32) <- (1x21x1x64xf32)
        transpose_25 = paddle._C_ops.transpose(reshape_25, [0, 2, 1, 3])
        del reshape_25

        # pd_op.reshape: (1x21x1x64xf32) <- (1x21x64xf32, 4xi64)
        reshape_26 = paddle._C_ops.reshape(add_76, full_int_array_1)

        # pd_op.transpose: (1x1x21x64xf32) <- (1x21x1x64xf32)
        transpose_26 = paddle._C_ops.transpose(reshape_26, [0, 2, 1, 3])
        del reshape_26

        # pd_op.slice: (21x32xf32) <- (128x32xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            parameter_11, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_11

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_286 = slice_12

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_287 = slice_12

        # pd_op.slice: (21x32xf32) <- (128x32xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            parameter_10, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_10

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_288 = slice_13

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_289 = slice_13

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_36 = paddle._C_ops.strided_slice(
            transpose_24, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_37 = paddle._C_ops.strided_slice(
            transpose_24, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_72 = paddle._C_ops.multiply(strided_slice_36, slice_13)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_73 = paddle._C_ops.multiply(strided_slice_37, slice_12)

        # pd_op.subtract: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        subtract_18 = paddle._C_ops.subtract(multiply_72, multiply_73)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_74 = paddle._C_ops.multiply(strided_slice_36, slice_12)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_75 = paddle._C_ops.multiply(strided_slice_37, slice_13)

        # pd_op.add: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        add_77 = paddle._C_ops.add(multiply_74, multiply_75)

        # builtin.combine: ([1x1x21x32xf32, 1x1x21x32xf32]) <- (1x1x21x32xf32, 1x1x21x32xf32)
        combine_18 = [subtract_18, add_77]

        # pd_op.stack: (1x1x21x32x2xf32) <- ([1x1x21x32xf32, 1x1x21x32xf32])
        stack_18 = paddle._C_ops.stack(combine_18, -1)
        del combine_18

        # pd_op.flatten: (1x1x21x64xf32) <- (1x1x21x32x2xf32)
        flatten_18 = paddle._C_ops.flatten(stack_18, 3, 4)

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_38 = paddle._C_ops.strided_slice(
            transpose_25, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_39 = paddle._C_ops.strided_slice(
            transpose_25, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_76 = paddle._C_ops.multiply(strided_slice_38, slice_13)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_77 = paddle._C_ops.multiply(strided_slice_39, slice_12)

        # pd_op.subtract: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        subtract_19 = paddle._C_ops.subtract(multiply_76, multiply_77)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_78 = paddle._C_ops.multiply(strided_slice_38, slice_12)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_79 = paddle._C_ops.multiply(strided_slice_39, slice_13)

        # pd_op.add: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        add_78 = paddle._C_ops.add(multiply_78, multiply_79)

        # builtin.combine: ([1x1x21x32xf32, 1x1x21x32xf32]) <- (1x1x21x32xf32, 1x1x21x32xf32)
        combine_19 = [subtract_19, add_78]

        # pd_op.stack: (1x1x21x32x2xf32) <- ([1x1x21x32xf32, 1x1x21x32xf32])
        stack_19 = paddle._C_ops.stack(combine_19, -1)
        del combine_19

        # pd_op.flatten: (1x1x21x64xf32) <- (1x1x21x32x2xf32)
        flatten_19 = paddle._C_ops.flatten(stack_19, 3, 4)

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_40 = paddle._C_ops.strided_slice(
            transpose_26, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_41 = paddle._C_ops.strided_slice(
            transpose_26, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_80 = paddle._C_ops.multiply(strided_slice_40, slice_13)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_81 = paddle._C_ops.multiply(strided_slice_41, slice_12)

        # pd_op.subtract: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        subtract_20 = paddle._C_ops.subtract(multiply_80, multiply_81)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_82 = paddle._C_ops.multiply(strided_slice_40, slice_12)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_83 = paddle._C_ops.multiply(strided_slice_41, slice_13)

        # pd_op.add: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        add_79 = paddle._C_ops.add(multiply_82, multiply_83)

        # builtin.combine: ([1x1x21x32xf32, 1x1x21x32xf32]) <- (1x1x21x32xf32, 1x1x21x32xf32)
        combine_20 = [subtract_20, add_79]

        # pd_op.stack: (1x1x21x32x2xf32) <- ([1x1x21x32xf32, 1x1x21x32xf32])
        stack_20 = paddle._C_ops.stack(combine_20, -1)
        del combine_20

        # pd_op.flatten: (1x1x21x64xf32) <- (1x1x21x32x2xf32)
        flatten_20 = paddle._C_ops.flatten(stack_20, 3, 4)

        # pd_op.scale: (1x1x21x64xf32) <- (1x1x21x64xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(flatten_18, full_3, float("0"), True)
        del flatten_18

        # pd_op.matmul: (1x1x21x21xf32) <- (1x1x21x64xf32, 1x1x21x64xf32)
        matmul_52 = paddle._C_ops.matmul(scale_7, flatten_19, False, True)

        # pd_op.add: (1x1x21x21xf32) <- (1x1x21x21xf32, 1x1x1x21xf32)
        add_80 = paddle._C_ops.add(matmul_52, unsqueeze_0)

        # pd_op.softmax: (1x1x21x21xf32) <- (1x1x21x21xf32)
        softmax_6 = paddle._C_ops.softmax(add_80, -1)
        del add_80

        # pd_op.dropout: (1x1x21x21xf32, 1x1x21x21xui8) <- (1x1x21x21xf32, None, 1xf32)
        dropout_38, dropout_39 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_6, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x1x21x64xf32) <- (1x1x21x21xf32, 1x1x21x64xf32)
        matmul_53 = paddle._C_ops.matmul(dropout_38, flatten_20, False, False)

        # pd_op.transpose: (1x21x1x64xf32) <- (1x1x21x64xf32)
        transpose_27 = paddle._C_ops.transpose(matmul_53, [0, 2, 1, 3])
        del matmul_53

        # pd_op.reshape: (1x21x64xf32) <- (1x21x1x64xf32, 3xi64)
        reshape_27 = paddle._C_ops.reshape(transpose_27, full_int_array_7)

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_54 = paddle._C_ops.matmul(reshape_27, parameter_115, False, False)
        del parameter_115

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_81 = paddle._C_ops.add(matmul_54, parameter_114)
        del parameter_114

        # pd_op.dropout: (1x21x64xf32, 1x21x64xui8) <- (1x21x64xf32, None, 1xf32)
        dropout_40, dropout_41 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_81, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_81

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 1x21x64xf32)
        add_82 = paddle._C_ops.add(layer_norm_36, dropout_40)

        # pd_op.layer_norm: (1x21x64xf32, 1x21xf32, 1x21xf32) <- (1x21x64xf32, 64xf32, 64xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_82, parameter_109, parameter_108, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_108, parameter_109

        # pd_op.matmul: (1x21x256xf32) <- (1x21x64xf32, 64x256xf32)
        matmul_55 = paddle._C_ops.matmul(layer_norm_39, parameter_113, False, False)
        del parameter_113

        # pd_op.add: (1x21x256xf32) <- (1x21x256xf32, 256xf32)
        add_83 = paddle._C_ops.add(matmul_55, parameter_112)
        del parameter_112

        # pd_op.gelu: (1x21x256xf32) <- (1x21x256xf32)
        gelu_6 = paddle._C_ops.gelu(add_83, False)

        # pd_op.matmul: (1x21x64xf32) <- (1x21x256xf32, 256x64xf32)
        matmul_56 = paddle._C_ops.matmul(gelu_6, parameter_111, False, False)
        del parameter_111

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_84 = paddle._C_ops.add(matmul_56, parameter_110)
        del parameter_110

        # pd_op.dropout: (1x21x64xf32, 1x21x64xui8) <- (1x21x64xf32, None, 1xf32)
        dropout_42, dropout_43 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_84, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_84

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 1x21x64xf32)
        add_85 = paddle._C_ops.add(layer_norm_39, dropout_42)

        # pd_op.layer_norm: (1x21x64xf32, 1x21xf32, 1x21xf32) <- (1x21x64xf32, 64xf32, 64xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_85, parameter_107, parameter_106, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_106, parameter_107

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_57 = paddle._C_ops.matmul(layer_norm_42, parameter_105, False, False)
        del parameter_105

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_86 = paddle._C_ops.add(matmul_57, parameter_104)
        del parameter_104

        # pd_op.reshape: (1x21x1x64xf32) <- (1x21x64xf32, 4xi64)
        reshape_28 = paddle._C_ops.reshape(add_86, full_int_array_1)

        # pd_op.transpose: (1x1x21x64xf32) <- (1x21x1x64xf32)
        transpose_28 = paddle._C_ops.transpose(reshape_28, [0, 2, 1, 3])
        del reshape_28

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_58 = paddle._C_ops.matmul(layer_norm_42, parameter_103, False, False)
        del parameter_103

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_87 = paddle._C_ops.add(matmul_58, parameter_102)
        del parameter_102

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_59 = paddle._C_ops.matmul(layer_norm_42, parameter_101, False, False)
        del parameter_101

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_88 = paddle._C_ops.add(matmul_59, parameter_100)
        del parameter_100

        # pd_op.reshape: (1x21x1x64xf32) <- (1x21x64xf32, 4xi64)
        reshape_29 = paddle._C_ops.reshape(add_87, full_int_array_1)

        # pd_op.transpose: (1x1x21x64xf32) <- (1x21x1x64xf32)
        transpose_29 = paddle._C_ops.transpose(reshape_29, [0, 2, 1, 3])
        del reshape_29

        # pd_op.reshape: (1x21x1x64xf32) <- (1x21x64xf32, 4xi64)
        reshape_30 = paddle._C_ops.reshape(add_88, full_int_array_1)

        # pd_op.transpose: (1x1x21x64xf32) <- (1x21x1x64xf32)
        transpose_30 = paddle._C_ops.transpose(reshape_30, [0, 2, 1, 3])
        del reshape_30

        # pd_op.slice: (21x32xf32) <- (128x32xf32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            parameter_9, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_9

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_290 = slice_14

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_291 = slice_14

        # pd_op.slice: (21x32xf32) <- (128x32xf32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            parameter_8, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_8

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_292 = slice_15

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_293 = slice_15

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_42 = paddle._C_ops.strided_slice(
            transpose_28, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_43 = paddle._C_ops.strided_slice(
            transpose_28, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_84 = paddle._C_ops.multiply(strided_slice_42, slice_15)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_85 = paddle._C_ops.multiply(strided_slice_43, slice_14)

        # pd_op.subtract: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        subtract_21 = paddle._C_ops.subtract(multiply_84, multiply_85)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_86 = paddle._C_ops.multiply(strided_slice_42, slice_14)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_87 = paddle._C_ops.multiply(strided_slice_43, slice_15)

        # pd_op.add: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        add_89 = paddle._C_ops.add(multiply_86, multiply_87)

        # builtin.combine: ([1x1x21x32xf32, 1x1x21x32xf32]) <- (1x1x21x32xf32, 1x1x21x32xf32)
        combine_21 = [subtract_21, add_89]

        # pd_op.stack: (1x1x21x32x2xf32) <- ([1x1x21x32xf32, 1x1x21x32xf32])
        stack_21 = paddle._C_ops.stack(combine_21, -1)
        del combine_21

        # pd_op.flatten: (1x1x21x64xf32) <- (1x1x21x32x2xf32)
        flatten_21 = paddle._C_ops.flatten(stack_21, 3, 4)

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_44 = paddle._C_ops.strided_slice(
            transpose_29, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_45 = paddle._C_ops.strided_slice(
            transpose_29, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_88 = paddle._C_ops.multiply(strided_slice_44, slice_15)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_89 = paddle._C_ops.multiply(strided_slice_45, slice_14)

        # pd_op.subtract: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        subtract_22 = paddle._C_ops.subtract(multiply_88, multiply_89)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_90 = paddle._C_ops.multiply(strided_slice_44, slice_14)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_91 = paddle._C_ops.multiply(strided_slice_45, slice_15)

        # pd_op.add: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        add_90 = paddle._C_ops.add(multiply_90, multiply_91)

        # builtin.combine: ([1x1x21x32xf32, 1x1x21x32xf32]) <- (1x1x21x32xf32, 1x1x21x32xf32)
        combine_22 = [subtract_22, add_90]

        # pd_op.stack: (1x1x21x32x2xf32) <- ([1x1x21x32xf32, 1x1x21x32xf32])
        stack_22 = paddle._C_ops.stack(combine_22, -1)
        del combine_22

        # pd_op.flatten: (1x1x21x64xf32) <- (1x1x21x32x2xf32)
        flatten_22 = paddle._C_ops.flatten(stack_22, 3, 4)

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_46 = paddle._C_ops.strided_slice(
            transpose_30, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_47 = paddle._C_ops.strided_slice(
            transpose_30, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_92 = paddle._C_ops.multiply(strided_slice_46, slice_15)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_93 = paddle._C_ops.multiply(strided_slice_47, slice_14)

        # pd_op.subtract: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        subtract_23 = paddle._C_ops.subtract(multiply_92, multiply_93)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_94 = paddle._C_ops.multiply(strided_slice_46, slice_14)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_95 = paddle._C_ops.multiply(strided_slice_47, slice_15)

        # pd_op.add: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        add_91 = paddle._C_ops.add(multiply_94, multiply_95)

        # builtin.combine: ([1x1x21x32xf32, 1x1x21x32xf32]) <- (1x1x21x32xf32, 1x1x21x32xf32)
        combine_23 = [subtract_23, add_91]

        # pd_op.stack: (1x1x21x32x2xf32) <- ([1x1x21x32xf32, 1x1x21x32xf32])
        stack_23 = paddle._C_ops.stack(combine_23, -1)
        del combine_23

        # pd_op.flatten: (1x1x21x64xf32) <- (1x1x21x32x2xf32)
        flatten_23 = paddle._C_ops.flatten(stack_23, 3, 4)

        # pd_op.scale: (1x1x21x64xf32) <- (1x1x21x64xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(flatten_21, full_3, float("0"), True)
        del flatten_21

        # pd_op.matmul: (1x1x21x21xf32) <- (1x1x21x64xf32, 1x1x21x64xf32)
        matmul_60 = paddle._C_ops.matmul(scale_8, flatten_22, False, True)

        # pd_op.add: (1x1x21x21xf32) <- (1x1x21x21xf32, 1x1x1x21xf32)
        add_92 = paddle._C_ops.add(matmul_60, unsqueeze_0)

        # pd_op.softmax: (1x1x21x21xf32) <- (1x1x21x21xf32)
        softmax_7 = paddle._C_ops.softmax(add_92, -1)
        del add_92

        # pd_op.dropout: (1x1x21x21xf32, 1x1x21x21xui8) <- (1x1x21x21xf32, None, 1xf32)
        dropout_44, dropout_45 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_7, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x1x21x64xf32) <- (1x1x21x21xf32, 1x1x21x64xf32)
        matmul_61 = paddle._C_ops.matmul(dropout_44, flatten_23, False, False)

        # pd_op.transpose: (1x21x1x64xf32) <- (1x1x21x64xf32)
        transpose_31 = paddle._C_ops.transpose(matmul_61, [0, 2, 1, 3])
        del matmul_61

        # pd_op.reshape: (1x21x64xf32) <- (1x21x1x64xf32, 3xi64)
        reshape_31 = paddle._C_ops.reshape(transpose_31, full_int_array_7)

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_62 = paddle._C_ops.matmul(reshape_31, parameter_99, False, False)
        del parameter_99

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_93 = paddle._C_ops.add(matmul_62, parameter_98)
        del parameter_98

        # pd_op.dropout: (1x21x64xf32, 1x21x64xui8) <- (1x21x64xf32, None, 1xf32)
        dropout_46, dropout_47 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_93, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_93

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 1x21x64xf32)
        add_94 = paddle._C_ops.add(layer_norm_42, dropout_46)

        # pd_op.layer_norm: (1x21x64xf32, 1x21xf32, 1x21xf32) <- (1x21x64xf32, 64xf32, 64xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_94, parameter_93, parameter_92, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_92, parameter_93

        # pd_op.matmul: (1x21x256xf32) <- (1x21x64xf32, 64x256xf32)
        matmul_63 = paddle._C_ops.matmul(layer_norm_45, parameter_97, False, False)
        del parameter_97

        # pd_op.add: (1x21x256xf32) <- (1x21x256xf32, 256xf32)
        add_95 = paddle._C_ops.add(matmul_63, parameter_96)
        del parameter_96

        # pd_op.gelu: (1x21x256xf32) <- (1x21x256xf32)
        gelu_7 = paddle._C_ops.gelu(add_95, False)

        # pd_op.matmul: (1x21x64xf32) <- (1x21x256xf32, 256x64xf32)
        matmul_64 = paddle._C_ops.matmul(gelu_7, parameter_95, False, False)
        del parameter_95

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_96 = paddle._C_ops.add(matmul_64, parameter_94)
        del parameter_94

        # pd_op.dropout: (1x21x64xf32, 1x21x64xui8) <- (1x21x64xf32, None, 1xf32)
        dropout_48, dropout_49 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_96, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_96

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 1x21x64xf32)
        add_97 = paddle._C_ops.add(layer_norm_45, dropout_48)

        # pd_op.layer_norm: (1x21x64xf32, 1x21xf32, 1x21xf32) <- (1x21x64xf32, 64xf32, 64xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_97, parameter_91, parameter_90, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_90, parameter_91

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_65 = paddle._C_ops.matmul(layer_norm_48, parameter_89, False, False)
        del parameter_89

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_98 = paddle._C_ops.add(matmul_65, parameter_88)
        del parameter_88

        # pd_op.reshape: (1x21x1x64xf32) <- (1x21x64xf32, 4xi64)
        reshape_32 = paddle._C_ops.reshape(add_98, full_int_array_1)

        # pd_op.transpose: (1x1x21x64xf32) <- (1x21x1x64xf32)
        transpose_32 = paddle._C_ops.transpose(reshape_32, [0, 2, 1, 3])
        del reshape_32

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_66 = paddle._C_ops.matmul(layer_norm_48, parameter_87, False, False)
        del parameter_87

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_99 = paddle._C_ops.add(matmul_66, parameter_86)
        del parameter_86

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_67 = paddle._C_ops.matmul(layer_norm_48, parameter_85, False, False)
        del parameter_85

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_100 = paddle._C_ops.add(matmul_67, parameter_84)
        del parameter_84

        # pd_op.reshape: (1x21x1x64xf32) <- (1x21x64xf32, 4xi64)
        reshape_33 = paddle._C_ops.reshape(add_99, full_int_array_1)

        # pd_op.transpose: (1x1x21x64xf32) <- (1x21x1x64xf32)
        transpose_33 = paddle._C_ops.transpose(reshape_33, [0, 2, 1, 3])
        del reshape_33

        # pd_op.reshape: (1x21x1x64xf32) <- (1x21x64xf32, 4xi64)
        reshape_34 = paddle._C_ops.reshape(add_100, full_int_array_1)

        # pd_op.transpose: (1x1x21x64xf32) <- (1x21x1x64xf32)
        transpose_34 = paddle._C_ops.transpose(reshape_34, [0, 2, 1, 3])
        del reshape_34

        # pd_op.slice: (21x32xf32) <- (128x32xf32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            parameter_7, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_7

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_294 = slice_16

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_295 = slice_16

        # pd_op.slice: (21x32xf32) <- (128x32xf32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            parameter_6, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_6

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_296 = slice_17

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_297 = slice_17

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_48 = paddle._C_ops.strided_slice(
            transpose_32, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_49 = paddle._C_ops.strided_slice(
            transpose_32, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_96 = paddle._C_ops.multiply(strided_slice_48, slice_17)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_97 = paddle._C_ops.multiply(strided_slice_49, slice_16)

        # pd_op.subtract: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        subtract_24 = paddle._C_ops.subtract(multiply_96, multiply_97)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_98 = paddle._C_ops.multiply(strided_slice_48, slice_16)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_99 = paddle._C_ops.multiply(strided_slice_49, slice_17)

        # pd_op.add: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        add_101 = paddle._C_ops.add(multiply_98, multiply_99)

        # builtin.combine: ([1x1x21x32xf32, 1x1x21x32xf32]) <- (1x1x21x32xf32, 1x1x21x32xf32)
        combine_24 = [subtract_24, add_101]

        # pd_op.stack: (1x1x21x32x2xf32) <- ([1x1x21x32xf32, 1x1x21x32xf32])
        stack_24 = paddle._C_ops.stack(combine_24, -1)
        del combine_24

        # pd_op.flatten: (1x1x21x64xf32) <- (1x1x21x32x2xf32)
        flatten_24 = paddle._C_ops.flatten(stack_24, 3, 4)

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_50 = paddle._C_ops.strided_slice(
            transpose_33, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_51 = paddle._C_ops.strided_slice(
            transpose_33, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_100 = paddle._C_ops.multiply(strided_slice_50, slice_17)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_101 = paddle._C_ops.multiply(strided_slice_51, slice_16)

        # pd_op.subtract: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        subtract_25 = paddle._C_ops.subtract(multiply_100, multiply_101)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_102 = paddle._C_ops.multiply(strided_slice_50, slice_16)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_103 = paddle._C_ops.multiply(strided_slice_51, slice_17)

        # pd_op.add: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        add_102 = paddle._C_ops.add(multiply_102, multiply_103)

        # builtin.combine: ([1x1x21x32xf32, 1x1x21x32xf32]) <- (1x1x21x32xf32, 1x1x21x32xf32)
        combine_25 = [subtract_25, add_102]

        # pd_op.stack: (1x1x21x32x2xf32) <- ([1x1x21x32xf32, 1x1x21x32xf32])
        stack_25 = paddle._C_ops.stack(combine_25, -1)
        del combine_25

        # pd_op.flatten: (1x1x21x64xf32) <- (1x1x21x32x2xf32)
        flatten_25 = paddle._C_ops.flatten(stack_25, 3, 4)

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_52 = paddle._C_ops.strided_slice(
            transpose_34, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_53 = paddle._C_ops.strided_slice(
            transpose_34, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_104 = paddle._C_ops.multiply(strided_slice_52, slice_17)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_105 = paddle._C_ops.multiply(strided_slice_53, slice_16)

        # pd_op.subtract: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        subtract_26 = paddle._C_ops.subtract(multiply_104, multiply_105)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_106 = paddle._C_ops.multiply(strided_slice_52, slice_16)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_107 = paddle._C_ops.multiply(strided_slice_53, slice_17)

        # pd_op.add: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        add_103 = paddle._C_ops.add(multiply_106, multiply_107)

        # builtin.combine: ([1x1x21x32xf32, 1x1x21x32xf32]) <- (1x1x21x32xf32, 1x1x21x32xf32)
        combine_26 = [subtract_26, add_103]

        # pd_op.stack: (1x1x21x32x2xf32) <- ([1x1x21x32xf32, 1x1x21x32xf32])
        stack_26 = paddle._C_ops.stack(combine_26, -1)
        del combine_26

        # pd_op.flatten: (1x1x21x64xf32) <- (1x1x21x32x2xf32)
        flatten_26 = paddle._C_ops.flatten(stack_26, 3, 4)

        # pd_op.scale: (1x1x21x64xf32) <- (1x1x21x64xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(flatten_24, full_3, float("0"), True)
        del flatten_24

        # pd_op.matmul: (1x1x21x21xf32) <- (1x1x21x64xf32, 1x1x21x64xf32)
        matmul_68 = paddle._C_ops.matmul(scale_9, flatten_25, False, True)

        # pd_op.add: (1x1x21x21xf32) <- (1x1x21x21xf32, 1x1x1x21xf32)
        add_104 = paddle._C_ops.add(matmul_68, unsqueeze_0)

        # pd_op.softmax: (1x1x21x21xf32) <- (1x1x21x21xf32)
        softmax_8 = paddle._C_ops.softmax(add_104, -1)
        del add_104

        # pd_op.dropout: (1x1x21x21xf32, 1x1x21x21xui8) <- (1x1x21x21xf32, None, 1xf32)
        dropout_50, dropout_51 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_8, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x1x21x64xf32) <- (1x1x21x21xf32, 1x1x21x64xf32)
        matmul_69 = paddle._C_ops.matmul(dropout_50, flatten_26, False, False)

        # pd_op.transpose: (1x21x1x64xf32) <- (1x1x21x64xf32)
        transpose_35 = paddle._C_ops.transpose(matmul_69, [0, 2, 1, 3])
        del matmul_69

        # pd_op.reshape: (1x21x64xf32) <- (1x21x1x64xf32, 3xi64)
        reshape_35 = paddle._C_ops.reshape(transpose_35, full_int_array_7)

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_70 = paddle._C_ops.matmul(reshape_35, parameter_83, False, False)
        del parameter_83

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_105 = paddle._C_ops.add(matmul_70, parameter_82)
        del parameter_82

        # pd_op.dropout: (1x21x64xf32, 1x21x64xui8) <- (1x21x64xf32, None, 1xf32)
        dropout_52, dropout_53 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_105, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_105

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 1x21x64xf32)
        add_106 = paddle._C_ops.add(layer_norm_48, dropout_52)

        # pd_op.layer_norm: (1x21x64xf32, 1x21xf32, 1x21xf32) <- (1x21x64xf32, 64xf32, 64xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_106, parameter_77, parameter_76, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_76, parameter_77

        # pd_op.matmul: (1x21x256xf32) <- (1x21x64xf32, 64x256xf32)
        matmul_71 = paddle._C_ops.matmul(layer_norm_51, parameter_81, False, False)
        del parameter_81

        # pd_op.add: (1x21x256xf32) <- (1x21x256xf32, 256xf32)
        add_107 = paddle._C_ops.add(matmul_71, parameter_80)
        del parameter_80

        # pd_op.gelu: (1x21x256xf32) <- (1x21x256xf32)
        gelu_8 = paddle._C_ops.gelu(add_107, False)

        # pd_op.matmul: (1x21x64xf32) <- (1x21x256xf32, 256x64xf32)
        matmul_72 = paddle._C_ops.matmul(gelu_8, parameter_79, False, False)
        del parameter_79

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_108 = paddle._C_ops.add(matmul_72, parameter_78)
        del parameter_78

        # pd_op.dropout: (1x21x64xf32, 1x21x64xui8) <- (1x21x64xf32, None, 1xf32)
        dropout_54, dropout_55 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_108, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_108

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 1x21x64xf32)
        add_109 = paddle._C_ops.add(layer_norm_51, dropout_54)

        # pd_op.layer_norm: (1x21x64xf32, 1x21xf32, 1x21xf32) <- (1x21x64xf32, 64xf32, 64xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_109, parameter_75, parameter_74, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_74, parameter_75

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_73 = paddle._C_ops.matmul(layer_norm_54, parameter_73, False, False)
        del parameter_73

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_110 = paddle._C_ops.add(matmul_73, parameter_72)
        del parameter_72

        # pd_op.reshape: (1x21x1x64xf32) <- (1x21x64xf32, 4xi64)
        reshape_36 = paddle._C_ops.reshape(add_110, full_int_array_1)

        # pd_op.transpose: (1x1x21x64xf32) <- (1x21x1x64xf32)
        transpose_36 = paddle._C_ops.transpose(reshape_36, [0, 2, 1, 3])
        del reshape_36

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_74 = paddle._C_ops.matmul(layer_norm_54, parameter_71, False, False)
        del parameter_71

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_111 = paddle._C_ops.add(matmul_74, parameter_70)
        del parameter_70

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_75 = paddle._C_ops.matmul(layer_norm_54, parameter_69, False, False)
        del parameter_69

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_112 = paddle._C_ops.add(matmul_75, parameter_68)
        del parameter_68

        # pd_op.reshape: (1x21x1x64xf32) <- (1x21x64xf32, 4xi64)
        reshape_37 = paddle._C_ops.reshape(add_111, full_int_array_1)

        # pd_op.transpose: (1x1x21x64xf32) <- (1x21x1x64xf32)
        transpose_37 = paddle._C_ops.transpose(reshape_37, [0, 2, 1, 3])
        del reshape_37

        # pd_op.reshape: (1x21x1x64xf32) <- (1x21x64xf32, 4xi64)
        reshape_38 = paddle._C_ops.reshape(add_112, full_int_array_1)

        # pd_op.transpose: (1x1x21x64xf32) <- (1x21x1x64xf32)
        transpose_38 = paddle._C_ops.transpose(reshape_38, [0, 2, 1, 3])
        del reshape_38

        # pd_op.slice: (21x32xf32) <- (128x32xf32, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(
            parameter_5, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_5

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_298 = slice_18

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_299 = slice_18

        # pd_op.slice: (21x32xf32) <- (128x32xf32, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(
            parameter_4, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_4

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_300 = slice_19

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_301 = slice_19

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_54 = paddle._C_ops.strided_slice(
            transpose_36, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_55 = paddle._C_ops.strided_slice(
            transpose_36, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_108 = paddle._C_ops.multiply(strided_slice_54, slice_19)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_109 = paddle._C_ops.multiply(strided_slice_55, slice_18)

        # pd_op.subtract: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        subtract_27 = paddle._C_ops.subtract(multiply_108, multiply_109)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_110 = paddle._C_ops.multiply(strided_slice_54, slice_18)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_111 = paddle._C_ops.multiply(strided_slice_55, slice_19)

        # pd_op.add: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        add_113 = paddle._C_ops.add(multiply_110, multiply_111)

        # builtin.combine: ([1x1x21x32xf32, 1x1x21x32xf32]) <- (1x1x21x32xf32, 1x1x21x32xf32)
        combine_27 = [subtract_27, add_113]

        # pd_op.stack: (1x1x21x32x2xf32) <- ([1x1x21x32xf32, 1x1x21x32xf32])
        stack_27 = paddle._C_ops.stack(combine_27, -1)
        del combine_27

        # pd_op.flatten: (1x1x21x64xf32) <- (1x1x21x32x2xf32)
        flatten_27 = paddle._C_ops.flatten(stack_27, 3, 4)

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_56 = paddle._C_ops.strided_slice(
            transpose_37, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_57 = paddle._C_ops.strided_slice(
            transpose_37, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_112 = paddle._C_ops.multiply(strided_slice_56, slice_19)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_113 = paddle._C_ops.multiply(strided_slice_57, slice_18)

        # pd_op.subtract: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        subtract_28 = paddle._C_ops.subtract(multiply_112, multiply_113)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_114 = paddle._C_ops.multiply(strided_slice_56, slice_18)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_115 = paddle._C_ops.multiply(strided_slice_57, slice_19)

        # pd_op.add: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        add_114 = paddle._C_ops.add(multiply_114, multiply_115)

        # builtin.combine: ([1x1x21x32xf32, 1x1x21x32xf32]) <- (1x1x21x32xf32, 1x1x21x32xf32)
        combine_28 = [subtract_28, add_114]

        # pd_op.stack: (1x1x21x32x2xf32) <- ([1x1x21x32xf32, 1x1x21x32xf32])
        stack_28 = paddle._C_ops.stack(combine_28, -1)
        del combine_28

        # pd_op.flatten: (1x1x21x64xf32) <- (1x1x21x32x2xf32)
        flatten_28 = paddle._C_ops.flatten(stack_28, 3, 4)

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_58 = paddle._C_ops.strided_slice(
            transpose_38, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_59 = paddle._C_ops.strided_slice(
            transpose_38, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_116 = paddle._C_ops.multiply(strided_slice_58, slice_19)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_117 = paddle._C_ops.multiply(strided_slice_59, slice_18)

        # pd_op.subtract: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        subtract_29 = paddle._C_ops.subtract(multiply_116, multiply_117)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_118 = paddle._C_ops.multiply(strided_slice_58, slice_18)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_119 = paddle._C_ops.multiply(strided_slice_59, slice_19)

        # pd_op.add: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        add_115 = paddle._C_ops.add(multiply_118, multiply_119)

        # builtin.combine: ([1x1x21x32xf32, 1x1x21x32xf32]) <- (1x1x21x32xf32, 1x1x21x32xf32)
        combine_29 = [subtract_29, add_115]

        # pd_op.stack: (1x1x21x32x2xf32) <- ([1x1x21x32xf32, 1x1x21x32xf32])
        stack_29 = paddle._C_ops.stack(combine_29, -1)
        del combine_29

        # pd_op.flatten: (1x1x21x64xf32) <- (1x1x21x32x2xf32)
        flatten_29 = paddle._C_ops.flatten(stack_29, 3, 4)

        # pd_op.scale: (1x1x21x64xf32) <- (1x1x21x64xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(flatten_27, full_3, float("0"), True)
        del flatten_27

        # pd_op.matmul: (1x1x21x21xf32) <- (1x1x21x64xf32, 1x1x21x64xf32)
        matmul_76 = paddle._C_ops.matmul(scale_10, flatten_28, False, True)

        # pd_op.add: (1x1x21x21xf32) <- (1x1x21x21xf32, 1x1x1x21xf32)
        add_116 = paddle._C_ops.add(matmul_76, unsqueeze_0)

        # pd_op.softmax: (1x1x21x21xf32) <- (1x1x21x21xf32)
        softmax_9 = paddle._C_ops.softmax(add_116, -1)
        del add_116

        # pd_op.dropout: (1x1x21x21xf32, 1x1x21x21xui8) <- (1x1x21x21xf32, None, 1xf32)
        dropout_56, dropout_57 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_9, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x1x21x64xf32) <- (1x1x21x21xf32, 1x1x21x64xf32)
        matmul_77 = paddle._C_ops.matmul(dropout_56, flatten_29, False, False)

        # pd_op.transpose: (1x21x1x64xf32) <- (1x1x21x64xf32)
        transpose_39 = paddle._C_ops.transpose(matmul_77, [0, 2, 1, 3])
        del matmul_77

        # pd_op.reshape: (1x21x64xf32) <- (1x21x1x64xf32, 3xi64)
        reshape_39 = paddle._C_ops.reshape(transpose_39, full_int_array_7)

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_78 = paddle._C_ops.matmul(reshape_39, parameter_67, False, False)
        del parameter_67

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_117 = paddle._C_ops.add(matmul_78, parameter_66)
        del parameter_66

        # pd_op.dropout: (1x21x64xf32, 1x21x64xui8) <- (1x21x64xf32, None, 1xf32)
        dropout_58, dropout_59 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_117, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_117

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 1x21x64xf32)
        add_118 = paddle._C_ops.add(layer_norm_54, dropout_58)

        # pd_op.layer_norm: (1x21x64xf32, 1x21xf32, 1x21xf32) <- (1x21x64xf32, 64xf32, 64xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_118, parameter_61, parameter_60, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_60, parameter_61

        # pd_op.matmul: (1x21x256xf32) <- (1x21x64xf32, 64x256xf32)
        matmul_79 = paddle._C_ops.matmul(layer_norm_57, parameter_65, False, False)
        del parameter_65

        # pd_op.add: (1x21x256xf32) <- (1x21x256xf32, 256xf32)
        add_119 = paddle._C_ops.add(matmul_79, parameter_64)
        del parameter_64

        # pd_op.gelu: (1x21x256xf32) <- (1x21x256xf32)
        gelu_9 = paddle._C_ops.gelu(add_119, False)

        # pd_op.matmul: (1x21x64xf32) <- (1x21x256xf32, 256x64xf32)
        matmul_80 = paddle._C_ops.matmul(gelu_9, parameter_63, False, False)
        del parameter_63

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_120 = paddle._C_ops.add(matmul_80, parameter_62)
        del parameter_62

        # pd_op.dropout: (1x21x64xf32, 1x21x64xui8) <- (1x21x64xf32, None, 1xf32)
        dropout_60, dropout_61 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_120, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_120

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 1x21x64xf32)
        add_121 = paddle._C_ops.add(layer_norm_57, dropout_60)

        # pd_op.layer_norm: (1x21x64xf32, 1x21xf32, 1x21xf32) <- (1x21x64xf32, 64xf32, 64xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_121, parameter_59, parameter_58, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_58, parameter_59

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_81 = paddle._C_ops.matmul(layer_norm_60, parameter_57, False, False)
        del parameter_57

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_122 = paddle._C_ops.add(matmul_81, parameter_56)
        del parameter_56

        # pd_op.reshape: (1x21x1x64xf32) <- (1x21x64xf32, 4xi64)
        reshape_40 = paddle._C_ops.reshape(add_122, full_int_array_1)

        # pd_op.transpose: (1x1x21x64xf32) <- (1x21x1x64xf32)
        transpose_40 = paddle._C_ops.transpose(reshape_40, [0, 2, 1, 3])
        del reshape_40

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_82 = paddle._C_ops.matmul(layer_norm_60, parameter_55, False, False)
        del parameter_55

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_123 = paddle._C_ops.add(matmul_82, parameter_54)
        del parameter_54

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_83 = paddle._C_ops.matmul(layer_norm_60, parameter_53, False, False)
        del parameter_53

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_124 = paddle._C_ops.add(matmul_83, parameter_52)
        del parameter_52

        # pd_op.reshape: (1x21x1x64xf32) <- (1x21x64xf32, 4xi64)
        reshape_41 = paddle._C_ops.reshape(add_123, full_int_array_1)

        # pd_op.transpose: (1x1x21x64xf32) <- (1x21x1x64xf32)
        transpose_41 = paddle._C_ops.transpose(reshape_41, [0, 2, 1, 3])
        del reshape_41

        # pd_op.reshape: (1x21x1x64xf32) <- (1x21x64xf32, 4xi64)
        reshape_42 = paddle._C_ops.reshape(add_124, full_int_array_1)

        # pd_op.transpose: (1x1x21x64xf32) <- (1x21x1x64xf32)
        transpose_42 = paddle._C_ops.transpose(reshape_42, [0, 2, 1, 3])
        del reshape_42

        # pd_op.slice: (21x32xf32) <- (128x32xf32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(
            parameter_3, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_3

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_302 = slice_20

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_303 = slice_20

        # pd_op.slice: (21x32xf32) <- (128x32xf32, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(
            parameter_2, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_2

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_304 = slice_21

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_305 = slice_21

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_60 = paddle._C_ops.strided_slice(
            transpose_40, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_61 = paddle._C_ops.strided_slice(
            transpose_40, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_120 = paddle._C_ops.multiply(strided_slice_60, slice_21)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_121 = paddle._C_ops.multiply(strided_slice_61, slice_20)

        # pd_op.subtract: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        subtract_30 = paddle._C_ops.subtract(multiply_120, multiply_121)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_122 = paddle._C_ops.multiply(strided_slice_60, slice_20)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_123 = paddle._C_ops.multiply(strided_slice_61, slice_21)

        # pd_op.add: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        add_125 = paddle._C_ops.add(multiply_122, multiply_123)

        # builtin.combine: ([1x1x21x32xf32, 1x1x21x32xf32]) <- (1x1x21x32xf32, 1x1x21x32xf32)
        combine_30 = [subtract_30, add_125]

        # pd_op.stack: (1x1x21x32x2xf32) <- ([1x1x21x32xf32, 1x1x21x32xf32])
        stack_30 = paddle._C_ops.stack(combine_30, -1)
        del combine_30

        # pd_op.flatten: (1x1x21x64xf32) <- (1x1x21x32x2xf32)
        flatten_30 = paddle._C_ops.flatten(stack_30, 3, 4)

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_62 = paddle._C_ops.strided_slice(
            transpose_41, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_63 = paddle._C_ops.strided_slice(
            transpose_41, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_124 = paddle._C_ops.multiply(strided_slice_62, slice_21)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_125 = paddle._C_ops.multiply(strided_slice_63, slice_20)

        # pd_op.subtract: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        subtract_31 = paddle._C_ops.subtract(multiply_124, multiply_125)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_126 = paddle._C_ops.multiply(strided_slice_62, slice_20)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_127 = paddle._C_ops.multiply(strided_slice_63, slice_21)

        # pd_op.add: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        add_126 = paddle._C_ops.add(multiply_126, multiply_127)

        # builtin.combine: ([1x1x21x32xf32, 1x1x21x32xf32]) <- (1x1x21x32xf32, 1x1x21x32xf32)
        combine_31 = [subtract_31, add_126]

        # pd_op.stack: (1x1x21x32x2xf32) <- ([1x1x21x32xf32, 1x1x21x32xf32])
        stack_31 = paddle._C_ops.stack(combine_31, -1)
        del combine_31

        # pd_op.flatten: (1x1x21x64xf32) <- (1x1x21x32x2xf32)
        flatten_31 = paddle._C_ops.flatten(stack_31, 3, 4)

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_64 = paddle._C_ops.strided_slice(
            transpose_42, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_65 = paddle._C_ops.strided_slice(
            transpose_42, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_128 = paddle._C_ops.multiply(strided_slice_64, slice_21)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_129 = paddle._C_ops.multiply(strided_slice_65, slice_20)

        # pd_op.subtract: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        subtract_32 = paddle._C_ops.subtract(multiply_128, multiply_129)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_130 = paddle._C_ops.multiply(strided_slice_64, slice_20)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_131 = paddle._C_ops.multiply(strided_slice_65, slice_21)

        # pd_op.add: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        add_127 = paddle._C_ops.add(multiply_130, multiply_131)

        # builtin.combine: ([1x1x21x32xf32, 1x1x21x32xf32]) <- (1x1x21x32xf32, 1x1x21x32xf32)
        combine_32 = [subtract_32, add_127]

        # pd_op.stack: (1x1x21x32x2xf32) <- ([1x1x21x32xf32, 1x1x21x32xf32])
        stack_32 = paddle._C_ops.stack(combine_32, -1)
        del combine_32

        # pd_op.flatten: (1x1x21x64xf32) <- (1x1x21x32x2xf32)
        flatten_32 = paddle._C_ops.flatten(stack_32, 3, 4)

        # pd_op.scale: (1x1x21x64xf32) <- (1x1x21x64xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(flatten_30, full_3, float("0"), True)
        del flatten_30

        # pd_op.matmul: (1x1x21x21xf32) <- (1x1x21x64xf32, 1x1x21x64xf32)
        matmul_84 = paddle._C_ops.matmul(scale_11, flatten_31, False, True)

        # pd_op.add: (1x1x21x21xf32) <- (1x1x21x21xf32, 1x1x1x21xf32)
        add_128 = paddle._C_ops.add(matmul_84, unsqueeze_0)

        # pd_op.softmax: (1x1x21x21xf32) <- (1x1x21x21xf32)
        softmax_10 = paddle._C_ops.softmax(add_128, -1)
        del add_128

        # pd_op.dropout: (1x1x21x21xf32, 1x1x21x21xui8) <- (1x1x21x21xf32, None, 1xf32)
        dropout_62, dropout_63 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_10, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x1x21x64xf32) <- (1x1x21x21xf32, 1x1x21x64xf32)
        matmul_85 = paddle._C_ops.matmul(dropout_62, flatten_32, False, False)

        # pd_op.transpose: (1x21x1x64xf32) <- (1x1x21x64xf32)
        transpose_43 = paddle._C_ops.transpose(matmul_85, [0, 2, 1, 3])
        del matmul_85

        # pd_op.reshape: (1x21x64xf32) <- (1x21x1x64xf32, 3xi64)
        reshape_43 = paddle._C_ops.reshape(transpose_43, full_int_array_7)

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_86 = paddle._C_ops.matmul(reshape_43, parameter_51, False, False)
        del parameter_51

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_129 = paddle._C_ops.add(matmul_86, parameter_50)
        del parameter_50

        # pd_op.dropout: (1x21x64xf32, 1x21x64xui8) <- (1x21x64xf32, None, 1xf32)
        dropout_64, dropout_65 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_129, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_129

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 1x21x64xf32)
        add_130 = paddle._C_ops.add(layer_norm_60, dropout_64)

        # pd_op.layer_norm: (1x21x64xf32, 1x21xf32, 1x21xf32) <- (1x21x64xf32, 64xf32, 64xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_130, parameter_45, parameter_44, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_44, parameter_45

        # pd_op.matmul: (1x21x256xf32) <- (1x21x64xf32, 64x256xf32)
        matmul_87 = paddle._C_ops.matmul(layer_norm_63, parameter_49, False, False)
        del parameter_49

        # pd_op.add: (1x21x256xf32) <- (1x21x256xf32, 256xf32)
        add_131 = paddle._C_ops.add(matmul_87, parameter_48)
        del parameter_48

        # pd_op.gelu: (1x21x256xf32) <- (1x21x256xf32)
        gelu_10 = paddle._C_ops.gelu(add_131, False)

        # pd_op.matmul: (1x21x64xf32) <- (1x21x256xf32, 256x64xf32)
        matmul_88 = paddle._C_ops.matmul(gelu_10, parameter_47, False, False)
        del parameter_47

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_132 = paddle._C_ops.add(matmul_88, parameter_46)
        del parameter_46

        # pd_op.dropout: (1x21x64xf32, 1x21x64xui8) <- (1x21x64xf32, None, 1xf32)
        dropout_66, dropout_67 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_132, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_132

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 1x21x64xf32)
        add_133 = paddle._C_ops.add(layer_norm_63, dropout_66)

        # pd_op.layer_norm: (1x21x64xf32, 1x21xf32, 1x21xf32) <- (1x21x64xf32, 64xf32, 64xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_133, parameter_43, parameter_42, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_42, parameter_43

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_89 = paddle._C_ops.matmul(layer_norm_66, parameter_41, False, False)
        del parameter_41

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_134 = paddle._C_ops.add(matmul_89, parameter_40)
        del parameter_40

        # pd_op.reshape: (1x21x1x64xf32) <- (1x21x64xf32, 4xi64)
        reshape_44 = paddle._C_ops.reshape(add_134, full_int_array_1)

        # pd_op.transpose: (1x1x21x64xf32) <- (1x21x1x64xf32)
        transpose_44 = paddle._C_ops.transpose(reshape_44, [0, 2, 1, 3])
        del reshape_44

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_90 = paddle._C_ops.matmul(layer_norm_66, parameter_39, False, False)
        del parameter_39

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_135 = paddle._C_ops.add(matmul_90, parameter_38)
        del parameter_38

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_91 = paddle._C_ops.matmul(layer_norm_66, parameter_37, False, False)
        del parameter_37

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_136 = paddle._C_ops.add(matmul_91, parameter_36)
        del parameter_36

        # pd_op.reshape: (1x21x1x64xf32) <- (1x21x64xf32, 4xi64)
        reshape_45 = paddle._C_ops.reshape(add_135, full_int_array_1)

        # pd_op.transpose: (1x1x21x64xf32) <- (1x21x1x64xf32)
        transpose_45 = paddle._C_ops.transpose(reshape_45, [0, 2, 1, 3])
        del reshape_45

        # pd_op.reshape: (1x21x1x64xf32) <- (1x21x64xf32, 4xi64)
        reshape_46 = paddle._C_ops.reshape(add_136, full_int_array_1)
        del full_int_array_1

        # pd_op.transpose: (1x1x21x64xf32) <- (1x21x1x64xf32)
        transpose_46 = paddle._C_ops.transpose(reshape_46, [0, 2, 1, 3])
        del reshape_46

        # pd_op.slice: (21x32xf32) <- (128x32xf32, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(
            parameter_1, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del parameter_1

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_306 = slice_22

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_307 = slice_22

        # pd_op.slice: (21x32xf32) <- (128x32xf32, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(
            parameter_0, [0], full_int_array_2, full_int_array_3, [1], []
        )
        del full_int_array_3, parameter_0

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_308 = slice_23

        # pd_op.assign: (21x32xf32) <- (21x32xf32)
        assign_309 = slice_23

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_66 = paddle._C_ops.strided_slice(
            transpose_44, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_67 = paddle._C_ops.strided_slice(
            transpose_44, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_132 = paddle._C_ops.multiply(strided_slice_66, slice_23)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_133 = paddle._C_ops.multiply(strided_slice_67, slice_22)

        # pd_op.subtract: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        subtract_33 = paddle._C_ops.subtract(multiply_132, multiply_133)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_134 = paddle._C_ops.multiply(strided_slice_66, slice_22)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_135 = paddle._C_ops.multiply(strided_slice_67, slice_23)

        # pd_op.add: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        add_137 = paddle._C_ops.add(multiply_134, multiply_135)

        # builtin.combine: ([1x1x21x32xf32, 1x1x21x32xf32]) <- (1x1x21x32xf32, 1x1x21x32xf32)
        combine_33 = [subtract_33, add_137]

        # pd_op.stack: (1x1x21x32x2xf32) <- ([1x1x21x32xf32, 1x1x21x32xf32])
        stack_33 = paddle._C_ops.stack(combine_33, -1)
        del combine_33

        # pd_op.flatten: (1x1x21x64xf32) <- (1x1x21x32x2xf32)
        flatten_33 = paddle._C_ops.flatten(stack_33, 3, 4)

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_68 = paddle._C_ops.strided_slice(
            transpose_45, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_69 = paddle._C_ops.strided_slice(
            transpose_45, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_136 = paddle._C_ops.multiply(strided_slice_68, slice_23)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_137 = paddle._C_ops.multiply(strided_slice_69, slice_22)

        # pd_op.subtract: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        subtract_34 = paddle._C_ops.subtract(multiply_136, multiply_137)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_138 = paddle._C_ops.multiply(strided_slice_68, slice_22)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_139 = paddle._C_ops.multiply(strided_slice_69, slice_23)

        # pd_op.add: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        add_138 = paddle._C_ops.add(multiply_138, multiply_139)

        # builtin.combine: ([1x1x21x32xf32, 1x1x21x32xf32]) <- (1x1x21x32xf32, 1x1x21x32xf32)
        combine_34 = [subtract_34, add_138]

        # pd_op.stack: (1x1x21x32x2xf32) <- ([1x1x21x32xf32, 1x1x21x32xf32])
        stack_34 = paddle._C_ops.stack(combine_34, -1)
        del combine_34

        # pd_op.flatten: (1x1x21x64xf32) <- (1x1x21x32x2xf32)
        flatten_34 = paddle._C_ops.flatten(stack_34, 3, 4)

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_70 = paddle._C_ops.strided_slice(
            transpose_46, [3], full_int_array_2, full_int_array_4, full_int_array_5
        )

        # pd_op.strided_slice: (1x1x21x32xf32) <- (1x1x21x64xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_71 = paddle._C_ops.strided_slice(
            transpose_46, [3], full_int_array_6, full_int_array_4, full_int_array_5
        )

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_140 = paddle._C_ops.multiply(strided_slice_70, slice_23)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_141 = paddle._C_ops.multiply(strided_slice_71, slice_22)

        # pd_op.subtract: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        subtract_35 = paddle._C_ops.subtract(multiply_140, multiply_141)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_142 = paddle._C_ops.multiply(strided_slice_70, slice_22)

        # pd_op.multiply: (1x1x21x32xf32) <- (1x1x21x32xf32, 21x32xf32)
        multiply_143 = paddle._C_ops.multiply(strided_slice_71, slice_23)

        # pd_op.add: (1x1x21x32xf32) <- (1x1x21x32xf32, 1x1x21x32xf32)
        add_139 = paddle._C_ops.add(multiply_142, multiply_143)

        # builtin.combine: ([1x1x21x32xf32, 1x1x21x32xf32]) <- (1x1x21x32xf32, 1x1x21x32xf32)
        combine_35 = [subtract_35, add_139]

        # pd_op.stack: (1x1x21x32x2xf32) <- ([1x1x21x32xf32, 1x1x21x32xf32])
        stack_35 = paddle._C_ops.stack(combine_35, -1)
        del combine_35

        # pd_op.flatten: (1x1x21x64xf32) <- (1x1x21x32x2xf32)
        flatten_35 = paddle._C_ops.flatten(stack_35, 3, 4)

        # pd_op.scale: (1x1x21x64xf32) <- (1x1x21x64xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(flatten_33, full_3, float("0"), True)
        del flatten_33

        # pd_op.matmul: (1x1x21x21xf32) <- (1x1x21x64xf32, 1x1x21x64xf32)
        matmul_92 = paddle._C_ops.matmul(scale_12, flatten_34, False, True)

        # pd_op.add: (1x1x21x21xf32) <- (1x1x21x21xf32, 1x1x1x21xf32)
        add_140 = paddle._C_ops.add(matmul_92, unsqueeze_0)

        # pd_op.softmax: (1x1x21x21xf32) <- (1x1x21x21xf32)
        softmax_11 = paddle._C_ops.softmax(add_140, -1)
        del add_140

        # pd_op.dropout: (1x1x21x21xf32, 1x1x21x21xui8) <- (1x1x21x21xf32, None, 1xf32)
        dropout_68, dropout_69 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_11, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x1x21x64xf32) <- (1x1x21x21xf32, 1x1x21x64xf32)
        matmul_93 = paddle._C_ops.matmul(dropout_68, flatten_35, False, False)

        # pd_op.transpose: (1x21x1x64xf32) <- (1x1x21x64xf32)
        transpose_47 = paddle._C_ops.transpose(matmul_93, [0, 2, 1, 3])
        del matmul_93

        # pd_op.reshape: (1x21x64xf32) <- (1x21x1x64xf32, 3xi64)
        reshape_47 = paddle._C_ops.reshape(transpose_47, full_int_array_7)
        del full_int_array_7

        # pd_op.matmul: (1x21x64xf32) <- (1x21x64xf32, 64x64xf32)
        matmul_94 = paddle._C_ops.matmul(reshape_47, parameter_35, False, False)
        del parameter_35

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_141 = paddle._C_ops.add(matmul_94, parameter_34)
        del parameter_34

        # pd_op.dropout: (1x21x64xf32, 1x21x64xui8) <- (1x21x64xf32, None, 1xf32)
        dropout_70, dropout_71 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_141, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_141

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 1x21x64xf32)
        add_142 = paddle._C_ops.add(layer_norm_66, dropout_70)

        # pd_op.layer_norm: (1x21x64xf32, 1x21xf32, 1x21xf32) <- (1x21x64xf32, 64xf32, 64xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_142, parameter_29, parameter_28, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_28, parameter_29

        # pd_op.matmul: (1x21x256xf32) <- (1x21x64xf32, 64x256xf32)
        matmul_95 = paddle._C_ops.matmul(layer_norm_69, parameter_33, False, False)
        del parameter_33

        # pd_op.add: (1x21x256xf32) <- (1x21x256xf32, 256xf32)
        add_143 = paddle._C_ops.add(matmul_95, parameter_32)
        del parameter_32

        # pd_op.gelu: (1x21x256xf32) <- (1x21x256xf32)
        gelu_11 = paddle._C_ops.gelu(add_143, False)

        # pd_op.matmul: (1x21x64xf32) <- (1x21x256xf32, 256x64xf32)
        matmul_96 = paddle._C_ops.matmul(gelu_11, parameter_31, False, False)
        del parameter_31

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 64xf32)
        add_144 = paddle._C_ops.add(matmul_96, parameter_30)
        del parameter_30

        # pd_op.dropout: (1x21x64xf32, 1x21x64xui8) <- (1x21x64xf32, None, 1xf32)
        dropout_72, dropout_73 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_144, None, full_2, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_144

        # pd_op.add: (1x21x64xf32) <- (1x21x64xf32, 1x21x64xf32)
        add_145 = paddle._C_ops.add(layer_norm_69, dropout_72)

        # pd_op.layer_norm: (1x21x64xf32, 1x21xf32, 1x21xf32) <- (1x21x64xf32, 64xf32, 64xf32)
        layer_norm_72, layer_norm_73, layer_norm_74 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_145, parameter_27, parameter_26, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_26, parameter_27

        # pd_op.slice: (1x64xf32) <- (1x21x64xf32, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(
            layer_norm_72, [1], full_int_array_2, full_int_array_6, [1], [1]
        )
        del full_int_array_2

        # pd_op.matmul: (1x64xf32) <- (1x64xf32, 64x64xf32)
        matmul_97 = paddle._C_ops.matmul(slice_24, parameter_25, False, False)
        del parameter_25

        # pd_op.add: (1x64xf32) <- (1x64xf32, 64xf32)
        add_146 = paddle._C_ops.add(matmul_97, parameter_24)
        del parameter_24

        # pd_op.tanh: (1x64xf32) <- (1x64xf32)
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
            flatten_10,
            flatten_11,
            flatten_13,
            flatten_14,
            flatten_16,
            flatten_17,
            flatten_19,
            flatten_2,
            flatten_20,
            flatten_22,
            flatten_23,
            flatten_25,
            flatten_26,
            flatten_28,
            flatten_29,
            flatten_31,
            flatten_32,
            flatten_34,
            flatten_35,
            flatten_4,
            flatten_5,
            flatten_7,
            flatten_8,
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
            matmul_12,
            matmul_14,
            matmul_15,
            matmul_16,
            matmul_17,
            matmul_18,
            matmul_19,
            matmul_2,
            matmul_20,
            matmul_22,
            matmul_23,
            matmul_24,
            matmul_25,
            matmul_26,
            matmul_27,
            matmul_28,
            matmul_3,
            matmul_30,
            matmul_31,
            matmul_32,
            matmul_33,
            matmul_34,
            matmul_35,
            matmul_36,
            matmul_38,
            matmul_39,
            matmul_4,
            matmul_40,
            matmul_41,
            matmul_42,
            matmul_43,
            matmul_44,
            matmul_46,
            matmul_47,
            matmul_48,
            matmul_49,
            matmul_50,
            matmul_51,
            matmul_52,
            matmul_54,
            matmul_55,
            matmul_56,
            matmul_57,
            matmul_58,
            matmul_59,
            matmul_6,
            matmul_60,
            matmul_62,
            matmul_63,
            matmul_64,
            matmul_65,
            matmul_66,
            matmul_67,
            matmul_68,
            matmul_7,
            matmul_70,
            matmul_71,
            matmul_72,
            matmul_73,
            matmul_74,
            matmul_75,
            matmul_76,
            matmul_78,
            matmul_79,
            matmul_8,
            matmul_80,
            matmul_81,
            matmul_82,
            matmul_83,
            matmul_84,
            matmul_86,
            matmul_87,
            matmul_88,
            matmul_89,
            matmul_9,
            matmul_90,
            matmul_91,
            matmul_92,
            matmul_94,
            matmul_95,
            matmul_96,
            matmul_97,
            multiply_0,
            multiply_1,
            multiply_10,
            multiply_100,
            multiply_101,
            multiply_102,
            multiply_103,
            multiply_104,
            multiply_105,
            multiply_106,
            multiply_107,
            multiply_108,
            multiply_109,
            multiply_11,
            multiply_110,
            multiply_111,
            multiply_112,
            multiply_113,
            multiply_114,
            multiply_115,
            multiply_116,
            multiply_117,
            multiply_118,
            multiply_119,
            multiply_12,
            multiply_120,
            multiply_121,
            multiply_122,
            multiply_123,
            multiply_124,
            multiply_125,
            multiply_126,
            multiply_127,
            multiply_128,
            multiply_129,
            multiply_13,
            multiply_130,
            multiply_131,
            multiply_132,
            multiply_133,
            multiply_134,
            multiply_135,
            multiply_136,
            multiply_137,
            multiply_138,
            multiply_139,
            multiply_14,
            multiply_140,
            multiply_141,
            multiply_142,
            multiply_143,
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
            multiply_96,
            multiply_97,
            multiply_98,
            multiply_99,
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
            stack_24,
            stack_25,
            stack_26,
            stack_27,
            stack_28,
            stack_29,
            stack_3,
            stack_30,
            stack_31,
            stack_32,
            stack_33,
            stack_34,
            stack_35,
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
            strided_slice_48,
            strided_slice_49,
            strided_slice_5,
            strided_slice_50,
            strided_slice_51,
            strided_slice_52,
            strided_slice_53,
            strided_slice_54,
            strided_slice_55,
            strided_slice_56,
            strided_slice_57,
            strided_slice_58,
            strided_slice_59,
            strided_slice_6,
            strided_slice_60,
            strided_slice_61,
            strided_slice_62,
            strided_slice_63,
            strided_slice_64,
            strided_slice_65,
            strided_slice_66,
            strided_slice_67,
            strided_slice_68,
            strided_slice_69,
            strided_slice_7,
            strided_slice_70,
            strided_slice_71,
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
            subtract_24,
            subtract_25,
            subtract_26,
            subtract_27,
            subtract_28,
            subtract_29,
            subtract_3,
            subtract_30,
            subtract_31,
            subtract_32,
            subtract_33,
            subtract_34,
            subtract_35,
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
