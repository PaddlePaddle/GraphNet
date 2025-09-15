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

        # pd_op.embedding: (1x21x768xf32) <- (1x21xi64, 39981x768xf32)
        embedding_0 = paddle._C_ops.embedding(data_0, parameter_198, 0, False)
        del data_0, parameter_198

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

        # pd_op.embedding: (1x21x768xf32) <- (1x21xi64, 2048x768xf32)
        embedding_1 = paddle._C_ops.embedding(subtract_0, parameter_197, -1, False)
        del parameter_197

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_0 = paddle._C_ops.add(embedding_0, embedding_1)

        # pd_op.embedding: (1x21x768xf32) <- (1x21xi64, 4x768xf32)
        embedding_2 = paddle._C_ops.embedding(data_1, parameter_196, -1, False)
        del data_1, parameter_196

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_1 = paddle._C_ops.add(add_0, embedding_2)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_1, parameter_195, parameter_194, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_194, parameter_195

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

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_0 = paddle._C_ops.matmul(dropout_0, parameter_193, False, False)
        del parameter_193

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_2 = paddle._C_ops.add(matmul_0, parameter_192)
        del parameter_192

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [0, 0, 12, 64]

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(add_2, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_0, [0, 2, 1, 3])
        del reshape_0

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_1 = paddle._C_ops.matmul(dropout_0, parameter_191, False, False)
        del parameter_191

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_3 = paddle._C_ops.add(matmul_1, parameter_190)
        del parameter_190

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_2 = paddle._C_ops.matmul(dropout_0, parameter_189, False, False)
        del parameter_189

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_4 = paddle._C_ops.add(matmul_2, parameter_188)
        del parameter_188

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(add_3, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_1, [0, 2, 1, 3])
        del reshape_1

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(add_4, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_2 = paddle._C_ops.transpose(reshape_2, [0, 2, 1, 3])
        del reshape_2

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_36 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_37 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_38 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_39 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_40 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_41 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_42 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_43 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_44 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_45 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_46 = full_5

        # pd_op.scale: (1x12x21x64xf32) <- (1x12x21x64xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(transpose_0, full_5, float("0"), True)
        del transpose_0

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x21x64xf32)
        matmul_3 = paddle._C_ops.matmul(scale_1, transpose_1, False, True)

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_5 = paddle._C_ops.add(matmul_3, unsqueeze_0)

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_0 = paddle._C_ops.softmax(add_5, -1)
        del add_5

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_0, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_4 = paddle._C_ops.matmul(dropout_2, transpose_2, False, False)

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_3 = paddle._C_ops.transpose(matmul_4, [0, 2, 1, 3])
        del matmul_4

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_2 = [0, 0, 768]

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_3, full_int_array_2)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_5 = paddle._C_ops.matmul(reshape_3, parameter_187, False, False)
        del parameter_187

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_6 = paddle._C_ops.add(matmul_5, parameter_186)
        del parameter_186

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_6, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_6

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_7 = paddle._C_ops.add(dropout_0, dropout_4)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_7, parameter_181, parameter_180, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_180, parameter_181

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_6 = paddle._C_ops.matmul(layer_norm_3, parameter_185, False, False)
        del parameter_185

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_8 = paddle._C_ops.add(matmul_6, parameter_184)
        del parameter_184

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_0 = paddle._C_ops.gelu(add_8, False)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_7 = paddle._C_ops.matmul(gelu_0, parameter_183, False, False)
        del parameter_183

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_9 = paddle._C_ops.add(matmul_7, parameter_182)
        del parameter_182

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_6, dropout_7 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_9, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_9

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_10 = paddle._C_ops.add(layer_norm_3, dropout_6)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_10, parameter_179, parameter_178, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_178, parameter_179

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_8 = paddle._C_ops.matmul(layer_norm_6, parameter_177, False, False)
        del parameter_177

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_11 = paddle._C_ops.add(matmul_8, parameter_176)
        del parameter_176

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(add_11, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_4, [0, 2, 1, 3])
        del reshape_4

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_9 = paddle._C_ops.matmul(layer_norm_6, parameter_175, False, False)
        del parameter_175

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_12 = paddle._C_ops.add(matmul_9, parameter_174)
        del parameter_174

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_6, parameter_173, False, False)
        del parameter_173

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_13 = paddle._C_ops.add(matmul_10, parameter_172)
        del parameter_172

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(add_12, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_5 = paddle._C_ops.transpose(reshape_5, [0, 2, 1, 3])
        del reshape_5

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(add_13, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_6, [0, 2, 1, 3])
        del reshape_6

        # pd_op.scale: (1x12x21x64xf32) <- (1x12x21x64xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(transpose_4, full_5, float("0"), True)
        del transpose_4

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x21x64xf32)
        matmul_11 = paddle._C_ops.matmul(scale_2, transpose_5, False, True)

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_14 = paddle._C_ops.add(matmul_11, unsqueeze_0)

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_1 = paddle._C_ops.softmax(add_14, -1)
        del add_14

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_1, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_12 = paddle._C_ops.matmul(dropout_8, transpose_6, False, False)

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_7 = paddle._C_ops.transpose(matmul_12, [0, 2, 1, 3])
        del matmul_12

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_7, full_int_array_2)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_13 = paddle._C_ops.matmul(reshape_7, parameter_171, False, False)
        del parameter_171

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_15 = paddle._C_ops.add(matmul_13, parameter_170)
        del parameter_170

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_10, dropout_11 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_15, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_15

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_16 = paddle._C_ops.add(layer_norm_6, dropout_10)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_16, parameter_165, parameter_164, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_164, parameter_165

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_14 = paddle._C_ops.matmul(layer_norm_9, parameter_169, False, False)
        del parameter_169

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_17 = paddle._C_ops.add(matmul_14, parameter_168)
        del parameter_168

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_1 = paddle._C_ops.gelu(add_17, False)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_15 = paddle._C_ops.matmul(gelu_1, parameter_167, False, False)
        del parameter_167

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_18 = paddle._C_ops.add(matmul_15, parameter_166)
        del parameter_166

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_12, dropout_13 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_18, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_18

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_19 = paddle._C_ops.add(layer_norm_9, dropout_12)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_19, parameter_163, parameter_162, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_162, parameter_163

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_16 = paddle._C_ops.matmul(layer_norm_12, parameter_161, False, False)
        del parameter_161

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_20 = paddle._C_ops.add(matmul_16, parameter_160)
        del parameter_160

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(add_20, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_8 = paddle._C_ops.transpose(reshape_8, [0, 2, 1, 3])
        del reshape_8

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_17 = paddle._C_ops.matmul(layer_norm_12, parameter_159, False, False)
        del parameter_159

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_21 = paddle._C_ops.add(matmul_17, parameter_158)
        del parameter_158

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_18 = paddle._C_ops.matmul(layer_norm_12, parameter_157, False, False)
        del parameter_157

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_22 = paddle._C_ops.add(matmul_18, parameter_156)
        del parameter_156

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(add_21, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_9 = paddle._C_ops.transpose(reshape_9, [0, 2, 1, 3])
        del reshape_9

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(add_22, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_10 = paddle._C_ops.transpose(reshape_10, [0, 2, 1, 3])
        del reshape_10

        # pd_op.scale: (1x12x21x64xf32) <- (1x12x21x64xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(transpose_8, full_5, float("0"), True)
        del transpose_8

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x21x64xf32)
        matmul_19 = paddle._C_ops.matmul(scale_3, transpose_9, False, True)

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_23 = paddle._C_ops.add(matmul_19, unsqueeze_0)

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_2 = paddle._C_ops.softmax(add_23, -1)
        del add_23

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_14, dropout_15 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_2, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_20 = paddle._C_ops.matmul(dropout_14, transpose_10, False, False)

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_11 = paddle._C_ops.transpose(matmul_20, [0, 2, 1, 3])
        del matmul_20

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_11, full_int_array_2)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_21 = paddle._C_ops.matmul(reshape_11, parameter_155, False, False)
        del parameter_155

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_24 = paddle._C_ops.add(matmul_21, parameter_154)
        del parameter_154

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_16, dropout_17 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_24, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_24

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_25 = paddle._C_ops.add(layer_norm_12, dropout_16)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_25, parameter_149, parameter_148, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_148, parameter_149

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_22 = paddle._C_ops.matmul(layer_norm_15, parameter_153, False, False)
        del parameter_153

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_26 = paddle._C_ops.add(matmul_22, parameter_152)
        del parameter_152

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_2 = paddle._C_ops.gelu(add_26, False)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_23 = paddle._C_ops.matmul(gelu_2, parameter_151, False, False)
        del parameter_151

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_27 = paddle._C_ops.add(matmul_23, parameter_150)
        del parameter_150

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_18, dropout_19 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_27, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_27

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_28 = paddle._C_ops.add(layer_norm_15, dropout_18)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_28, parameter_147, parameter_146, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_146, parameter_147

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_24 = paddle._C_ops.matmul(layer_norm_18, parameter_145, False, False)
        del parameter_145

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_29 = paddle._C_ops.add(matmul_24, parameter_144)
        del parameter_144

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(add_29, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_12 = paddle._C_ops.transpose(reshape_12, [0, 2, 1, 3])
        del reshape_12

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_25 = paddle._C_ops.matmul(layer_norm_18, parameter_143, False, False)
        del parameter_143

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_30 = paddle._C_ops.add(matmul_25, parameter_142)
        del parameter_142

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_26 = paddle._C_ops.matmul(layer_norm_18, parameter_141, False, False)
        del parameter_141

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_31 = paddle._C_ops.add(matmul_26, parameter_140)
        del parameter_140

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(add_30, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_13, [0, 2, 1, 3])
        del reshape_13

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(add_31, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_14 = paddle._C_ops.transpose(reshape_14, [0, 2, 1, 3])
        del reshape_14

        # pd_op.scale: (1x12x21x64xf32) <- (1x12x21x64xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(transpose_12, full_5, float("0"), True)
        del transpose_12

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x21x64xf32)
        matmul_27 = paddle._C_ops.matmul(scale_4, transpose_13, False, True)

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_32 = paddle._C_ops.add(matmul_27, unsqueeze_0)

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_3 = paddle._C_ops.softmax(add_32, -1)
        del add_32

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_20, dropout_21 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_3, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_28 = paddle._C_ops.matmul(dropout_20, transpose_14, False, False)

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_15 = paddle._C_ops.transpose(matmul_28, [0, 2, 1, 3])
        del matmul_28

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_15 = paddle._C_ops.reshape(transpose_15, full_int_array_2)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_29 = paddle._C_ops.matmul(reshape_15, parameter_139, False, False)
        del parameter_139

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_33 = paddle._C_ops.add(matmul_29, parameter_138)
        del parameter_138

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_22, dropout_23 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_33, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_33

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_34 = paddle._C_ops.add(layer_norm_18, dropout_22)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_34, parameter_133, parameter_132, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_132, parameter_133

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_30 = paddle._C_ops.matmul(layer_norm_21, parameter_137, False, False)
        del parameter_137

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_35 = paddle._C_ops.add(matmul_30, parameter_136)
        del parameter_136

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_3 = paddle._C_ops.gelu(add_35, False)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_31 = paddle._C_ops.matmul(gelu_3, parameter_135, False, False)
        del parameter_135

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_36 = paddle._C_ops.add(matmul_31, parameter_134)
        del parameter_134

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_24, dropout_25 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_36, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_36

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_37 = paddle._C_ops.add(layer_norm_21, dropout_24)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_37, parameter_131, parameter_130, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_130, parameter_131

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_32 = paddle._C_ops.matmul(layer_norm_24, parameter_129, False, False)
        del parameter_129

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_38 = paddle._C_ops.add(matmul_32, parameter_128)
        del parameter_128

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_16 = paddle._C_ops.reshape(add_38, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_16 = paddle._C_ops.transpose(reshape_16, [0, 2, 1, 3])
        del reshape_16

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_33 = paddle._C_ops.matmul(layer_norm_24, parameter_127, False, False)
        del parameter_127

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_39 = paddle._C_ops.add(matmul_33, parameter_126)
        del parameter_126

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_34 = paddle._C_ops.matmul(layer_norm_24, parameter_125, False, False)
        del parameter_125

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_40 = paddle._C_ops.add(matmul_34, parameter_124)
        del parameter_124

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_17 = paddle._C_ops.reshape(add_39, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_17 = paddle._C_ops.transpose(reshape_17, [0, 2, 1, 3])
        del reshape_17

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_18 = paddle._C_ops.reshape(add_40, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_18 = paddle._C_ops.transpose(reshape_18, [0, 2, 1, 3])
        del reshape_18

        # pd_op.scale: (1x12x21x64xf32) <- (1x12x21x64xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(transpose_16, full_5, float("0"), True)
        del transpose_16

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x21x64xf32)
        matmul_35 = paddle._C_ops.matmul(scale_5, transpose_17, False, True)

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_41 = paddle._C_ops.add(matmul_35, unsqueeze_0)

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_4 = paddle._C_ops.softmax(add_41, -1)
        del add_41

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_26, dropout_27 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_4, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_36 = paddle._C_ops.matmul(dropout_26, transpose_18, False, False)

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_19 = paddle._C_ops.transpose(matmul_36, [0, 2, 1, 3])
        del matmul_36

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_19 = paddle._C_ops.reshape(transpose_19, full_int_array_2)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_37 = paddle._C_ops.matmul(reshape_19, parameter_123, False, False)
        del parameter_123

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_42 = paddle._C_ops.add(matmul_37, parameter_122)
        del parameter_122

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_28, dropout_29 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_42, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_42

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_43 = paddle._C_ops.add(layer_norm_24, dropout_28)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_43, parameter_117, parameter_116, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_116, parameter_117

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_38 = paddle._C_ops.matmul(layer_norm_27, parameter_121, False, False)
        del parameter_121

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_44 = paddle._C_ops.add(matmul_38, parameter_120)
        del parameter_120

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_4 = paddle._C_ops.gelu(add_44, False)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_39 = paddle._C_ops.matmul(gelu_4, parameter_119, False, False)
        del parameter_119

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_45 = paddle._C_ops.add(matmul_39, parameter_118)
        del parameter_118

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_30, dropout_31 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_45, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_45

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_46 = paddle._C_ops.add(layer_norm_27, dropout_30)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_46, parameter_115, parameter_114, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_114, parameter_115

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_40 = paddle._C_ops.matmul(layer_norm_30, parameter_113, False, False)
        del parameter_113

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_47 = paddle._C_ops.add(matmul_40, parameter_112)
        del parameter_112

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_20 = paddle._C_ops.reshape(add_47, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_20 = paddle._C_ops.transpose(reshape_20, [0, 2, 1, 3])
        del reshape_20

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_41 = paddle._C_ops.matmul(layer_norm_30, parameter_111, False, False)
        del parameter_111

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_48 = paddle._C_ops.add(matmul_41, parameter_110)
        del parameter_110

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_42 = paddle._C_ops.matmul(layer_norm_30, parameter_109, False, False)
        del parameter_109

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_49 = paddle._C_ops.add(matmul_42, parameter_108)
        del parameter_108

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_21 = paddle._C_ops.reshape(add_48, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_21 = paddle._C_ops.transpose(reshape_21, [0, 2, 1, 3])
        del reshape_21

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_22 = paddle._C_ops.reshape(add_49, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_22 = paddle._C_ops.transpose(reshape_22, [0, 2, 1, 3])
        del reshape_22

        # pd_op.scale: (1x12x21x64xf32) <- (1x12x21x64xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(transpose_20, full_5, float("0"), True)
        del transpose_20

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x21x64xf32)
        matmul_43 = paddle._C_ops.matmul(scale_6, transpose_21, False, True)

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_50 = paddle._C_ops.add(matmul_43, unsqueeze_0)

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_5 = paddle._C_ops.softmax(add_50, -1)
        del add_50

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_32, dropout_33 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_5, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_44 = paddle._C_ops.matmul(dropout_32, transpose_22, False, False)

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_23 = paddle._C_ops.transpose(matmul_44, [0, 2, 1, 3])
        del matmul_44

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_23 = paddle._C_ops.reshape(transpose_23, full_int_array_2)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_45 = paddle._C_ops.matmul(reshape_23, parameter_107, False, False)
        del parameter_107

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_51 = paddle._C_ops.add(matmul_45, parameter_106)
        del parameter_106

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_34, dropout_35 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_51, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_51

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_52 = paddle._C_ops.add(layer_norm_30, dropout_34)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_52, parameter_101, parameter_100, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_100, parameter_101

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_46 = paddle._C_ops.matmul(layer_norm_33, parameter_105, False, False)
        del parameter_105

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_53 = paddle._C_ops.add(matmul_46, parameter_104)
        del parameter_104

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_5 = paddle._C_ops.gelu(add_53, False)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_47 = paddle._C_ops.matmul(gelu_5, parameter_103, False, False)
        del parameter_103

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_54 = paddle._C_ops.add(matmul_47, parameter_102)
        del parameter_102

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_36, dropout_37 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_54, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_54

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_55 = paddle._C_ops.add(layer_norm_33, dropout_36)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_55, parameter_99, parameter_98, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_98, parameter_99

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_48 = paddle._C_ops.matmul(layer_norm_36, parameter_97, False, False)
        del parameter_97

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_56 = paddle._C_ops.add(matmul_48, parameter_96)
        del parameter_96

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_24 = paddle._C_ops.reshape(add_56, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_24 = paddle._C_ops.transpose(reshape_24, [0, 2, 1, 3])
        del reshape_24

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_49 = paddle._C_ops.matmul(layer_norm_36, parameter_95, False, False)
        del parameter_95

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_57 = paddle._C_ops.add(matmul_49, parameter_94)
        del parameter_94

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_50 = paddle._C_ops.matmul(layer_norm_36, parameter_93, False, False)
        del parameter_93

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_58 = paddle._C_ops.add(matmul_50, parameter_92)
        del parameter_92

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_25 = paddle._C_ops.reshape(add_57, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_25 = paddle._C_ops.transpose(reshape_25, [0, 2, 1, 3])
        del reshape_25

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_26 = paddle._C_ops.reshape(add_58, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_26 = paddle._C_ops.transpose(reshape_26, [0, 2, 1, 3])
        del reshape_26

        # pd_op.scale: (1x12x21x64xf32) <- (1x12x21x64xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(transpose_24, full_5, float("0"), True)
        del transpose_24

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x21x64xf32)
        matmul_51 = paddle._C_ops.matmul(scale_7, transpose_25, False, True)

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_59 = paddle._C_ops.add(matmul_51, unsqueeze_0)

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_6 = paddle._C_ops.softmax(add_59, -1)
        del add_59

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_38, dropout_39 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_6, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_52 = paddle._C_ops.matmul(dropout_38, transpose_26, False, False)

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_27 = paddle._C_ops.transpose(matmul_52, [0, 2, 1, 3])
        del matmul_52

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_27 = paddle._C_ops.reshape(transpose_27, full_int_array_2)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_53 = paddle._C_ops.matmul(reshape_27, parameter_91, False, False)
        del parameter_91

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_60 = paddle._C_ops.add(matmul_53, parameter_90)
        del parameter_90

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_40, dropout_41 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_60, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_60

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_61 = paddle._C_ops.add(layer_norm_36, dropout_40)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_61, parameter_85, parameter_84, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_84, parameter_85

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_54 = paddle._C_ops.matmul(layer_norm_39, parameter_89, False, False)
        del parameter_89

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_62 = paddle._C_ops.add(matmul_54, parameter_88)
        del parameter_88

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_6 = paddle._C_ops.gelu(add_62, False)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_55 = paddle._C_ops.matmul(gelu_6, parameter_87, False, False)
        del parameter_87

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_63 = paddle._C_ops.add(matmul_55, parameter_86)
        del parameter_86

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_42, dropout_43 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_63, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_63

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_64 = paddle._C_ops.add(layer_norm_39, dropout_42)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_64, parameter_83, parameter_82, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_82, parameter_83

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_56 = paddle._C_ops.matmul(layer_norm_42, parameter_81, False, False)
        del parameter_81

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_65 = paddle._C_ops.add(matmul_56, parameter_80)
        del parameter_80

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_28 = paddle._C_ops.reshape(add_65, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_28 = paddle._C_ops.transpose(reshape_28, [0, 2, 1, 3])
        del reshape_28

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_57 = paddle._C_ops.matmul(layer_norm_42, parameter_79, False, False)
        del parameter_79

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_66 = paddle._C_ops.add(matmul_57, parameter_78)
        del parameter_78

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_58 = paddle._C_ops.matmul(layer_norm_42, parameter_77, False, False)
        del parameter_77

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_67 = paddle._C_ops.add(matmul_58, parameter_76)
        del parameter_76

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_29 = paddle._C_ops.reshape(add_66, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_29 = paddle._C_ops.transpose(reshape_29, [0, 2, 1, 3])
        del reshape_29

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_30 = paddle._C_ops.reshape(add_67, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_30 = paddle._C_ops.transpose(reshape_30, [0, 2, 1, 3])
        del reshape_30

        # pd_op.scale: (1x12x21x64xf32) <- (1x12x21x64xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(transpose_28, full_5, float("0"), True)
        del transpose_28

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x21x64xf32)
        matmul_59 = paddle._C_ops.matmul(scale_8, transpose_29, False, True)

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_68 = paddle._C_ops.add(matmul_59, unsqueeze_0)

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_7 = paddle._C_ops.softmax(add_68, -1)
        del add_68

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_44, dropout_45 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_7, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_60 = paddle._C_ops.matmul(dropout_44, transpose_30, False, False)

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_31 = paddle._C_ops.transpose(matmul_60, [0, 2, 1, 3])
        del matmul_60

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_31 = paddle._C_ops.reshape(transpose_31, full_int_array_2)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_61 = paddle._C_ops.matmul(reshape_31, parameter_75, False, False)
        del parameter_75

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_69 = paddle._C_ops.add(matmul_61, parameter_74)
        del parameter_74

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_46, dropout_47 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_69, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_69

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_70 = paddle._C_ops.add(layer_norm_42, dropout_46)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_70, parameter_69, parameter_68, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_68, parameter_69

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_62 = paddle._C_ops.matmul(layer_norm_45, parameter_73, False, False)
        del parameter_73

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_71 = paddle._C_ops.add(matmul_62, parameter_72)
        del parameter_72

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_7 = paddle._C_ops.gelu(add_71, False)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_63 = paddle._C_ops.matmul(gelu_7, parameter_71, False, False)
        del parameter_71

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_72 = paddle._C_ops.add(matmul_63, parameter_70)
        del parameter_70

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_48, dropout_49 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_72, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_72

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_73 = paddle._C_ops.add(layer_norm_45, dropout_48)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_73, parameter_67, parameter_66, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_66, parameter_67

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_64 = paddle._C_ops.matmul(layer_norm_48, parameter_65, False, False)
        del parameter_65

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_74 = paddle._C_ops.add(matmul_64, parameter_64)
        del parameter_64

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_32 = paddle._C_ops.reshape(add_74, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_32 = paddle._C_ops.transpose(reshape_32, [0, 2, 1, 3])
        del reshape_32

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_65 = paddle._C_ops.matmul(layer_norm_48, parameter_63, False, False)
        del parameter_63

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_75 = paddle._C_ops.add(matmul_65, parameter_62)
        del parameter_62

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_66 = paddle._C_ops.matmul(layer_norm_48, parameter_61, False, False)
        del parameter_61

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_76 = paddle._C_ops.add(matmul_66, parameter_60)
        del parameter_60

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_33 = paddle._C_ops.reshape(add_75, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_33 = paddle._C_ops.transpose(reshape_33, [0, 2, 1, 3])
        del reshape_33

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_34 = paddle._C_ops.reshape(add_76, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_34 = paddle._C_ops.transpose(reshape_34, [0, 2, 1, 3])
        del reshape_34

        # pd_op.scale: (1x12x21x64xf32) <- (1x12x21x64xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(transpose_32, full_5, float("0"), True)
        del transpose_32

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x21x64xf32)
        matmul_67 = paddle._C_ops.matmul(scale_9, transpose_33, False, True)

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_77 = paddle._C_ops.add(matmul_67, unsqueeze_0)

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_8 = paddle._C_ops.softmax(add_77, -1)
        del add_77

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_50, dropout_51 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_8, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_68 = paddle._C_ops.matmul(dropout_50, transpose_34, False, False)

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_35 = paddle._C_ops.transpose(matmul_68, [0, 2, 1, 3])
        del matmul_68

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_35 = paddle._C_ops.reshape(transpose_35, full_int_array_2)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_69 = paddle._C_ops.matmul(reshape_35, parameter_59, False, False)
        del parameter_59

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_78 = paddle._C_ops.add(matmul_69, parameter_58)
        del parameter_58

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_52, dropout_53 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_78, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_78

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_79 = paddle._C_ops.add(layer_norm_48, dropout_52)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_79, parameter_53, parameter_52, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_52, parameter_53

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_70 = paddle._C_ops.matmul(layer_norm_51, parameter_57, False, False)
        del parameter_57

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_80 = paddle._C_ops.add(matmul_70, parameter_56)
        del parameter_56

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_8 = paddle._C_ops.gelu(add_80, False)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_71 = paddle._C_ops.matmul(gelu_8, parameter_55, False, False)
        del parameter_55

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_81 = paddle._C_ops.add(matmul_71, parameter_54)
        del parameter_54

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_54, dropout_55 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_81, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_81

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_82 = paddle._C_ops.add(layer_norm_51, dropout_54)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_82, parameter_51, parameter_50, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_50, parameter_51

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_72 = paddle._C_ops.matmul(layer_norm_54, parameter_49, False, False)
        del parameter_49

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_83 = paddle._C_ops.add(matmul_72, parameter_48)
        del parameter_48

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_36 = paddle._C_ops.reshape(add_83, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_36 = paddle._C_ops.transpose(reshape_36, [0, 2, 1, 3])
        del reshape_36

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_73 = paddle._C_ops.matmul(layer_norm_54, parameter_47, False, False)
        del parameter_47

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_84 = paddle._C_ops.add(matmul_73, parameter_46)
        del parameter_46

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_74 = paddle._C_ops.matmul(layer_norm_54, parameter_45, False, False)
        del parameter_45

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_85 = paddle._C_ops.add(matmul_74, parameter_44)
        del parameter_44

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_37 = paddle._C_ops.reshape(add_84, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_37 = paddle._C_ops.transpose(reshape_37, [0, 2, 1, 3])
        del reshape_37

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_38 = paddle._C_ops.reshape(add_85, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_38 = paddle._C_ops.transpose(reshape_38, [0, 2, 1, 3])
        del reshape_38

        # pd_op.scale: (1x12x21x64xf32) <- (1x12x21x64xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(transpose_36, full_5, float("0"), True)
        del transpose_36

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x21x64xf32)
        matmul_75 = paddle._C_ops.matmul(scale_10, transpose_37, False, True)

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_86 = paddle._C_ops.add(matmul_75, unsqueeze_0)

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_9 = paddle._C_ops.softmax(add_86, -1)
        del add_86

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_56, dropout_57 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_9, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_76 = paddle._C_ops.matmul(dropout_56, transpose_38, False, False)

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_39 = paddle._C_ops.transpose(matmul_76, [0, 2, 1, 3])
        del matmul_76

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_39 = paddle._C_ops.reshape(transpose_39, full_int_array_2)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_77 = paddle._C_ops.matmul(reshape_39, parameter_43, False, False)
        del parameter_43

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_87 = paddle._C_ops.add(matmul_77, parameter_42)
        del parameter_42

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_58, dropout_59 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_87, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_87

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_88 = paddle._C_ops.add(layer_norm_54, dropout_58)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_88, parameter_37, parameter_36, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_36, parameter_37

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_78 = paddle._C_ops.matmul(layer_norm_57, parameter_41, False, False)
        del parameter_41

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_89 = paddle._C_ops.add(matmul_78, parameter_40)
        del parameter_40

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_9 = paddle._C_ops.gelu(add_89, False)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_79 = paddle._C_ops.matmul(gelu_9, parameter_39, False, False)
        del parameter_39

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_90 = paddle._C_ops.add(matmul_79, parameter_38)
        del parameter_38

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_60, dropout_61 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_90, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_90

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_91 = paddle._C_ops.add(layer_norm_57, dropout_60)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_91, parameter_35, parameter_34, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_34, parameter_35

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_80 = paddle._C_ops.matmul(layer_norm_60, parameter_33, False, False)
        del parameter_33

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_92 = paddle._C_ops.add(matmul_80, parameter_32)
        del parameter_32

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_40 = paddle._C_ops.reshape(add_92, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_40 = paddle._C_ops.transpose(reshape_40, [0, 2, 1, 3])
        del reshape_40

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_81 = paddle._C_ops.matmul(layer_norm_60, parameter_31, False, False)
        del parameter_31

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_93 = paddle._C_ops.add(matmul_81, parameter_30)
        del parameter_30

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_82 = paddle._C_ops.matmul(layer_norm_60, parameter_29, False, False)
        del parameter_29

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_94 = paddle._C_ops.add(matmul_82, parameter_28)
        del parameter_28

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_41 = paddle._C_ops.reshape(add_93, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_41 = paddle._C_ops.transpose(reshape_41, [0, 2, 1, 3])
        del reshape_41

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_42 = paddle._C_ops.reshape(add_94, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_42 = paddle._C_ops.transpose(reshape_42, [0, 2, 1, 3])
        del reshape_42

        # pd_op.scale: (1x12x21x64xf32) <- (1x12x21x64xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(transpose_40, full_5, float("0"), True)
        del transpose_40

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x21x64xf32)
        matmul_83 = paddle._C_ops.matmul(scale_11, transpose_41, False, True)

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_95 = paddle._C_ops.add(matmul_83, unsqueeze_0)

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_10 = paddle._C_ops.softmax(add_95, -1)
        del add_95

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_62, dropout_63 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_10, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_84 = paddle._C_ops.matmul(dropout_62, transpose_42, False, False)

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_43 = paddle._C_ops.transpose(matmul_84, [0, 2, 1, 3])
        del matmul_84

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_43 = paddle._C_ops.reshape(transpose_43, full_int_array_2)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_85 = paddle._C_ops.matmul(reshape_43, parameter_27, False, False)
        del parameter_27

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_96 = paddle._C_ops.add(matmul_85, parameter_26)
        del parameter_26

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_64, dropout_65 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_96, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_96

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_97 = paddle._C_ops.add(layer_norm_60, dropout_64)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_97, parameter_21, parameter_20, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_20, parameter_21

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_86 = paddle._C_ops.matmul(layer_norm_63, parameter_25, False, False)
        del parameter_25

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_98 = paddle._C_ops.add(matmul_86, parameter_24)
        del parameter_24

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_10 = paddle._C_ops.gelu(add_98, False)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_87 = paddle._C_ops.matmul(gelu_10, parameter_23, False, False)
        del parameter_23

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_99 = paddle._C_ops.add(matmul_87, parameter_22)
        del parameter_22

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_66, dropout_67 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_99, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_99

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_100 = paddle._C_ops.add(layer_norm_63, dropout_66)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_100, parameter_19, parameter_18, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_18, parameter_19

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_88 = paddle._C_ops.matmul(layer_norm_66, parameter_17, False, False)
        del parameter_17

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_101 = paddle._C_ops.add(matmul_88, parameter_16)
        del parameter_16

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_44 = paddle._C_ops.reshape(add_101, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_44 = paddle._C_ops.transpose(reshape_44, [0, 2, 1, 3])
        del reshape_44

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_89 = paddle._C_ops.matmul(layer_norm_66, parameter_15, False, False)
        del parameter_15

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_102 = paddle._C_ops.add(matmul_89, parameter_14)
        del parameter_14

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_90 = paddle._C_ops.matmul(layer_norm_66, parameter_13, False, False)
        del parameter_13

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_103 = paddle._C_ops.add(matmul_90, parameter_12)
        del parameter_12

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_45 = paddle._C_ops.reshape(add_102, full_int_array_1)

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_45 = paddle._C_ops.transpose(reshape_45, [0, 2, 1, 3])
        del reshape_45

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_46 = paddle._C_ops.reshape(add_103, full_int_array_1)
        del full_int_array_1

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_46 = paddle._C_ops.transpose(reshape_46, [0, 2, 1, 3])
        del reshape_46

        # pd_op.scale: (1x12x21x64xf32) <- (1x12x21x64xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(transpose_44, full_5, float("0"), True)
        del transpose_44

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x21x64xf32)
        matmul_91 = paddle._C_ops.matmul(scale_12, transpose_45, False, True)

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_104 = paddle._C_ops.add(matmul_91, unsqueeze_0)

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_11 = paddle._C_ops.softmax(add_104, -1)
        del add_104

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_68, dropout_69 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_11, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_92 = paddle._C_ops.matmul(dropout_68, transpose_46, False, False)

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_47 = paddle._C_ops.transpose(matmul_92, [0, 2, 1, 3])
        del matmul_92

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_47 = paddle._C_ops.reshape(transpose_47, full_int_array_2)
        del full_int_array_2

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_93 = paddle._C_ops.matmul(reshape_47, parameter_11, False, False)
        del parameter_11

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_105 = paddle._C_ops.add(matmul_93, parameter_10)
        del parameter_10

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_70, dropout_71 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_105, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_105

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_106 = paddle._C_ops.add(layer_norm_66, dropout_70)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_106, parameter_5, parameter_4, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_4, parameter_5

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_94 = paddle._C_ops.matmul(layer_norm_69, parameter_9, False, False)
        del parameter_9

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_107 = paddle._C_ops.add(matmul_94, parameter_8)
        del parameter_8

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_11 = paddle._C_ops.gelu(add_107, False)

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_95 = paddle._C_ops.matmul(gelu_11, parameter_7, False, False)
        del parameter_7

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_108 = paddle._C_ops.add(matmul_95, parameter_6)
        del parameter_6

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_72, dropout_73 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_108, None, full_4, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_108

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_109 = paddle._C_ops.add(layer_norm_69, dropout_72)

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_72, layer_norm_73, layer_norm_74 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_109, parameter_3, parameter_2, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_2, parameter_3

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [1]

        # pd_op.slice: (1x768xf32) <- (1x21x768xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            layer_norm_72, [1], full_int_array_3, full_int_array_4, [1], [1]
        )

        # pd_op.matmul: (1x768xf32) <- (1x768xf32, 768x768xf32)
        matmul_96 = paddle._C_ops.matmul(slice_0, parameter_1, False, False)
        del parameter_1

        # pd_op.add: (1x768xf32) <- (1x768xf32, 768xf32)
        add_110 = paddle._C_ops.add(matmul_96, parameter_0)
        del parameter_0

        # pd_op.tanh: (1x768xf32) <- (1x768xf32)
        tanh_0 = paddle._C_ops.tanh(add_110)
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
            add_12,
            add_13,
            add_16,
            add_17,
            add_19,
            add_2,
            add_20,
            add_21,
            add_22,
            add_25,
            add_26,
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
            add_43,
            add_44,
            add_46,
            add_47,
            add_48,
            add_49,
            add_52,
            add_53,
            add_55,
            add_56,
            add_57,
            add_58,
            add_61,
            add_62,
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
            add_79,
            add_8,
            add_80,
            add_82,
            add_83,
            add_84,
            add_85,
            add_88,
            add_89,
            add_91,
            add_92,
            add_93,
            add_94,
            add_97,
            add_98,
            assign_0,
            assign_1,
            assign_10,
            assign_11,
            assign_12,
            assign_13,
            assign_14,
            assign_15,
            assign_16,
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
            assign_5,
            assign_6,
            assign_7,
            assign_8,
            assign_9,
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
            full_int_array_3,
            full_int_array_4,
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
            subtract_0,
            transpose_1,
            transpose_10,
            transpose_11,
            transpose_13,
            transpose_14,
            transpose_15,
            transpose_17,
            transpose_18,
            transpose_19,
            transpose_2,
            transpose_21,
            transpose_22,
            transpose_23,
            transpose_25,
            transpose_26,
            transpose_27,
            transpose_29,
            transpose_3,
            transpose_30,
            transpose_31,
            transpose_33,
            transpose_34,
            transpose_35,
            transpose_37,
            transpose_38,
            transpose_39,
            transpose_41,
            transpose_42,
            transpose_43,
            transpose_45,
            transpose_46,
            transpose_47,
            transpose_5,
            transpose_6,
            transpose_7,
            transpose_9,
            unsqueeze_0,
        )

        return tanh_0
