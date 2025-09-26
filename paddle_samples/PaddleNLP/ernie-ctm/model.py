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
        data_0,
        data_1,
    ):
        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.equal: (1x93xb) <- (1x93xi64, xi64)
        equal_0 = paddle._C_ops.equal(data_0, full_0)
        del full_0

        # pd_op.cast: (1x93xf32) <- (1x93xb)
        cast_0 = paddle._C_ops.cast(equal_0, paddle.float32)
        del equal_0

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("-10000"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x93xf32) <- (1x93xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(cast_0, full_1, float("0"), True)
        del cast_0, full_1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 2]

        # pd_op.unsqueeze: (1x1x1x93xf32) <- (1x93xf32, 2xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(scale_0, full_int_array_0)
        del full_int_array_0, scale_0

        # pd_op.full: (2xi64) <- ()
        full_2 = paddle._C_ops.full(
            [2], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.full: (1xi64) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xi64) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("91"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("91"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.linspace: (-1xi64) <- (1xi64, 1xi64, 1xi32)
        linspace_0 = paddle._C_ops.linspace(
            full_3,
            full_4,
            full_5,
            paddle.int64,
            paddle.framework._current_expected_place(),
        )
        del full_3, full_4, full_5

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([2xi64, -1xi64]) <- (2xi64, -1xi64)
        combine_0 = [full_2, linspace_0]
        del full_2, linspace_0

        # pd_op.concat: (-1xi64) <- ([2xi64, -1xi64], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_6)
        del combine_0, full_6

        # pd_op.embedding: (1x93x128xf32) <- (1x93xi64, 23000x128xf32)
        embedding_0 = paddle._C_ops.embedding(data_0, parameter_204, 0, False)
        del data_0, parameter_204

        # pd_op.embedding: (-1x128xf32) <- (-1xi64, 512x128xf32)
        embedding_1 = paddle._C_ops.embedding(concat_0, parameter_203, -1, False)
        del concat_0, parameter_203

        # pd_op.embedding: (1x93x128xf32) <- (1x93xi64, 2x128xf32)
        embedding_2 = paddle._C_ops.embedding(data_1, parameter_202, -1, False)
        del data_1, parameter_202

        # pd_op.add: (1x93x128xf32) <- (1x93x128xf32, 1x93x128xf32)
        add_1 = paddle._C_ops.add(embedding_0, embedding_2)
        del embedding_0, embedding_2

        # pd_op.add: (1x93x128xf32) <- (1x93x128xf32, -1x128xf32)
        add_2 = paddle._C_ops.add(add_1, embedding_1)
        del add_1, embedding_1

        # pd_op.layer_norm: (1x93x128xf32, 1x93xf32, 1x93xf32) <- (1x93x128xf32, 128xf32, 128xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_2, parameter_201, parameter_200, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_2, parameter_200, parameter_201

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (1x93x128xf32, 1x93x128xui8) <- (1x93x128xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                layer_norm_0, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del layer_norm_0

        # pd_op.matmul: (1x93x768xf32) <- (1x93x128xf32, 128x768xf32)
        matmul_0 = paddle._C_ops.matmul(dropout_0, parameter_199, False, False)
        del dropout_0, parameter_199

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_3 = paddle._C_ops.add(matmul_0, parameter_198)
        del matmul_0, parameter_198

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_1 = paddle._C_ops.matmul(add_3, parameter_197, False, False)
        del parameter_197

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_4 = paddle._C_ops.add(matmul_1, parameter_196)
        del matmul_1, parameter_196

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [0, 0, 12, 64]

        # pd_op.reshape: (1x93x12x64xf32) <- (1x93x768xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(add_4, full_int_array_1)
        del add_4

        # pd_op.transpose: (1x12x93x64xf32) <- (1x93x12x64xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_0, [0, 2, 1, 3])
        del reshape_0

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_2 = paddle._C_ops.matmul(add_3, parameter_195, False, False)
        del parameter_195

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_5 = paddle._C_ops.add(matmul_2, parameter_194)
        del matmul_2, parameter_194

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_3 = paddle._C_ops.matmul(add_3, parameter_193, False, False)
        del parameter_193

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_6 = paddle._C_ops.add(matmul_3, parameter_192)
        del matmul_3, parameter_192

        # pd_op.reshape: (1x93x12x64xf32) <- (1x93x768xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(add_5, full_int_array_1)
        del add_5

        # pd_op.transpose: (1x12x93x64xf32) <- (1x93x12x64xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_1, [0, 2, 1, 3])
        del reshape_1

        # pd_op.reshape: (1x93x12x64xf32) <- (1x93x768xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(add_6, full_int_array_1)
        del add_6

        # pd_op.transpose: (1x12x93x64xf32) <- (1x93x12x64xf32)
        transpose_2 = paddle._C_ops.transpose(reshape_2, [0, 2, 1, 3])
        del reshape_2

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x12x93x64xf32) <- (1x12x93x64xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(transpose_0, full_8, float("0"), True)
        del transpose_0

        # pd_op.matmul: (1x12x93x93xf32) <- (1x12x93x64xf32, 1x12x93x64xf32)
        matmul_4 = paddle._C_ops.matmul(scale_1, transpose_1, False, True)
        del scale_1, transpose_1

        # pd_op.add: (1x12x93x93xf32) <- (1x12x93x93xf32, 1x1x1x93xf32)
        add_7 = paddle._C_ops.add(matmul_4, unsqueeze_0)
        del matmul_4

        # pd_op.softmax: (1x12x93x93xf32) <- (1x12x93x93xf32)
        softmax_0 = paddle._C_ops.softmax(add_7, -1)
        del add_7

        # pd_op.dropout: (1x12x93x93xf32, 1x12x93x93xui8) <- (1x12x93x93xf32, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_0, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_0

        # pd_op.matmul: (1x12x93x64xf32) <- (1x12x93x93xf32, 1x12x93x64xf32)
        matmul_5 = paddle._C_ops.matmul(dropout_2, transpose_2, False, False)
        del dropout_2, transpose_2

        # pd_op.transpose: (1x93x12x64xf32) <- (1x12x93x64xf32)
        transpose_3 = paddle._C_ops.transpose(matmul_5, [0, 2, 1, 3])
        del matmul_5

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_2 = [0, 0, 768]

        # pd_op.reshape: (1x93x768xf32) <- (1x93x12x64xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_3, full_int_array_2)
        del transpose_3

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_6 = paddle._C_ops.matmul(reshape_3, parameter_191, False, False)
        del parameter_191, reshape_3

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_8 = paddle._C_ops.add(matmul_6, parameter_190)
        del matmul_6, parameter_190

        # pd_op.dropout: (1x93x768xf32, 1x93x768xui8) <- (1x93x768xf32, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_8, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_8

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 1x93x768xf32)
        add_9 = paddle._C_ops.add(add_3, dropout_4)
        del add_3, dropout_4

        # pd_op.layer_norm: (1x93x768xf32, 1x93xf32, 1x93xf32) <- (1x93x768xf32, 768xf32, 768xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_9, parameter_185, parameter_184, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_9, parameter_184, parameter_185

        # pd_op.matmul: (1x93x3072xf32) <- (1x93x768xf32, 768x3072xf32)
        matmul_7 = paddle._C_ops.matmul(layer_norm_3, parameter_189, False, False)
        del parameter_189

        # pd_op.add: (1x93x3072xf32) <- (1x93x3072xf32, 3072xf32)
        add_10 = paddle._C_ops.add(matmul_7, parameter_188)
        del matmul_7, parameter_188

        # pd_op.gelu: (1x93x3072xf32) <- (1x93x3072xf32)
        gelu_0 = paddle._C_ops.gelu(add_10, True)
        del add_10

        # pd_op.matmul: (1x93x768xf32) <- (1x93x3072xf32, 3072x768xf32)
        matmul_8 = paddle._C_ops.matmul(gelu_0, parameter_187, False, False)
        del gelu_0, parameter_187

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_11 = paddle._C_ops.add(matmul_8, parameter_186)
        del matmul_8, parameter_186

        # pd_op.dropout: (1x93x768xf32, 1x93x768xui8) <- (1x93x768xf32, None, 1xf32)
        dropout_6, dropout_7 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_11, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_11

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 1x93x768xf32)
        add_12 = paddle._C_ops.add(layer_norm_3, dropout_6)
        del dropout_6, layer_norm_3

        # pd_op.layer_norm: (1x93x768xf32, 1x93xf32, 1x93xf32) <- (1x93x768xf32, 768xf32, 768xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_12, parameter_183, parameter_182, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_12, parameter_182, parameter_183

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_9 = paddle._C_ops.matmul(layer_norm_6, parameter_181, False, False)
        del parameter_181

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_13 = paddle._C_ops.add(matmul_9, parameter_180)
        del matmul_9, parameter_180

        # pd_op.reshape: (1x93x12x64xf32) <- (1x93x768xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(add_13, full_int_array_1)
        del add_13

        # pd_op.transpose: (1x12x93x64xf32) <- (1x93x12x64xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_4, [0, 2, 1, 3])
        del reshape_4

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_6, parameter_179, False, False)
        del parameter_179

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_14 = paddle._C_ops.add(matmul_10, parameter_178)
        del matmul_10, parameter_178

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_11 = paddle._C_ops.matmul(layer_norm_6, parameter_177, False, False)
        del parameter_177

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_15 = paddle._C_ops.add(matmul_11, parameter_176)
        del matmul_11, parameter_176

        # pd_op.reshape: (1x93x12x64xf32) <- (1x93x768xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(add_14, full_int_array_1)
        del add_14

        # pd_op.transpose: (1x12x93x64xf32) <- (1x93x12x64xf32)
        transpose_5 = paddle._C_ops.transpose(reshape_5, [0, 2, 1, 3])
        del reshape_5

        # pd_op.reshape: (1x93x12x64xf32) <- (1x93x768xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(add_15, full_int_array_1)
        del add_15

        # pd_op.transpose: (1x12x93x64xf32) <- (1x93x12x64xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_6, [0, 2, 1, 3])
        del reshape_6

        # pd_op.scale: (1x12x93x64xf32) <- (1x12x93x64xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(transpose_4, full_8, float("0"), True)
        del transpose_4

        # pd_op.matmul: (1x12x93x93xf32) <- (1x12x93x64xf32, 1x12x93x64xf32)
        matmul_12 = paddle._C_ops.matmul(scale_2, transpose_5, False, True)
        del scale_2, transpose_5

        # pd_op.add: (1x12x93x93xf32) <- (1x12x93x93xf32, 1x1x1x93xf32)
        add_16 = paddle._C_ops.add(matmul_12, unsqueeze_0)
        del matmul_12

        # pd_op.softmax: (1x12x93x93xf32) <- (1x12x93x93xf32)
        softmax_1 = paddle._C_ops.softmax(add_16, -1)
        del add_16

        # pd_op.dropout: (1x12x93x93xf32, 1x12x93x93xui8) <- (1x12x93x93xf32, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_1, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_1

        # pd_op.matmul: (1x12x93x64xf32) <- (1x12x93x93xf32, 1x12x93x64xf32)
        matmul_13 = paddle._C_ops.matmul(dropout_8, transpose_6, False, False)
        del dropout_8, transpose_6

        # pd_op.transpose: (1x93x12x64xf32) <- (1x12x93x64xf32)
        transpose_7 = paddle._C_ops.transpose(matmul_13, [0, 2, 1, 3])
        del matmul_13

        # pd_op.reshape: (1x93x768xf32) <- (1x93x12x64xf32, 3xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_7, full_int_array_2)
        del transpose_7

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_14 = paddle._C_ops.matmul(reshape_7, parameter_175, False, False)
        del parameter_175, reshape_7

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_17 = paddle._C_ops.add(matmul_14, parameter_174)
        del matmul_14, parameter_174

        # pd_op.dropout: (1x93x768xf32, 1x93x768xui8) <- (1x93x768xf32, None, 1xf32)
        dropout_10, dropout_11 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_17, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_17

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 1x93x768xf32)
        add_18 = paddle._C_ops.add(layer_norm_6, dropout_10)
        del dropout_10, layer_norm_6

        # pd_op.layer_norm: (1x93x768xf32, 1x93xf32, 1x93xf32) <- (1x93x768xf32, 768xf32, 768xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_18, parameter_169, parameter_168, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_18, parameter_168, parameter_169

        # pd_op.matmul: (1x93x3072xf32) <- (1x93x768xf32, 768x3072xf32)
        matmul_15 = paddle._C_ops.matmul(layer_norm_9, parameter_173, False, False)
        del parameter_173

        # pd_op.add: (1x93x3072xf32) <- (1x93x3072xf32, 3072xf32)
        add_19 = paddle._C_ops.add(matmul_15, parameter_172)
        del matmul_15, parameter_172

        # pd_op.gelu: (1x93x3072xf32) <- (1x93x3072xf32)
        gelu_1 = paddle._C_ops.gelu(add_19, False)
        del add_19

        # pd_op.matmul: (1x93x768xf32) <- (1x93x3072xf32, 3072x768xf32)
        matmul_16 = paddle._C_ops.matmul(gelu_1, parameter_171, False, False)
        del gelu_1, parameter_171

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_20 = paddle._C_ops.add(matmul_16, parameter_170)
        del matmul_16, parameter_170

        # pd_op.dropout: (1x93x768xf32, 1x93x768xui8) <- (1x93x768xf32, None, 1xf32)
        dropout_12, dropout_13 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_20, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_20

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 1x93x768xf32)
        add_21 = paddle._C_ops.add(layer_norm_9, dropout_12)
        del dropout_12, layer_norm_9

        # pd_op.layer_norm: (1x93x768xf32, 1x93xf32, 1x93xf32) <- (1x93x768xf32, 768xf32, 768xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_21, parameter_167, parameter_166, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_21, parameter_166, parameter_167

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_17 = paddle._C_ops.matmul(layer_norm_12, parameter_165, False, False)
        del parameter_165

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_22 = paddle._C_ops.add(matmul_17, parameter_164)
        del matmul_17, parameter_164

        # pd_op.reshape: (1x93x12x64xf32) <- (1x93x768xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(add_22, full_int_array_1)
        del add_22

        # pd_op.transpose: (1x12x93x64xf32) <- (1x93x12x64xf32)
        transpose_8 = paddle._C_ops.transpose(reshape_8, [0, 2, 1, 3])
        del reshape_8

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_18 = paddle._C_ops.matmul(layer_norm_12, parameter_163, False, False)
        del parameter_163

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_23 = paddle._C_ops.add(matmul_18, parameter_162)
        del matmul_18, parameter_162

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_19 = paddle._C_ops.matmul(layer_norm_12, parameter_161, False, False)
        del parameter_161

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_24 = paddle._C_ops.add(matmul_19, parameter_160)
        del matmul_19, parameter_160

        # pd_op.reshape: (1x93x12x64xf32) <- (1x93x768xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(add_23, full_int_array_1)
        del add_23

        # pd_op.transpose: (1x12x93x64xf32) <- (1x93x12x64xf32)
        transpose_9 = paddle._C_ops.transpose(reshape_9, [0, 2, 1, 3])
        del reshape_9

        # pd_op.reshape: (1x93x12x64xf32) <- (1x93x768xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(add_24, full_int_array_1)
        del add_24

        # pd_op.transpose: (1x12x93x64xf32) <- (1x93x12x64xf32)
        transpose_10 = paddle._C_ops.transpose(reshape_10, [0, 2, 1, 3])
        del reshape_10

        # pd_op.scale: (1x12x93x64xf32) <- (1x12x93x64xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(transpose_8, full_8, float("0"), True)
        del transpose_8

        # pd_op.matmul: (1x12x93x93xf32) <- (1x12x93x64xf32, 1x12x93x64xf32)
        matmul_20 = paddle._C_ops.matmul(scale_3, transpose_9, False, True)
        del scale_3, transpose_9

        # pd_op.add: (1x12x93x93xf32) <- (1x12x93x93xf32, 1x1x1x93xf32)
        add_25 = paddle._C_ops.add(matmul_20, unsqueeze_0)
        del matmul_20

        # pd_op.softmax: (1x12x93x93xf32) <- (1x12x93x93xf32)
        softmax_2 = paddle._C_ops.softmax(add_25, -1)
        del add_25

        # pd_op.dropout: (1x12x93x93xf32, 1x12x93x93xui8) <- (1x12x93x93xf32, None, 1xf32)
        dropout_14, dropout_15 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_2, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_2

        # pd_op.matmul: (1x12x93x64xf32) <- (1x12x93x93xf32, 1x12x93x64xf32)
        matmul_21 = paddle._C_ops.matmul(dropout_14, transpose_10, False, False)
        del dropout_14, transpose_10

        # pd_op.transpose: (1x93x12x64xf32) <- (1x12x93x64xf32)
        transpose_11 = paddle._C_ops.transpose(matmul_21, [0, 2, 1, 3])
        del matmul_21

        # pd_op.reshape: (1x93x768xf32) <- (1x93x12x64xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_11, full_int_array_2)
        del transpose_11

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_22 = paddle._C_ops.matmul(reshape_11, parameter_159, False, False)
        del parameter_159, reshape_11

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_26 = paddle._C_ops.add(matmul_22, parameter_158)
        del matmul_22, parameter_158

        # pd_op.dropout: (1x93x768xf32, 1x93x768xui8) <- (1x93x768xf32, None, 1xf32)
        dropout_16, dropout_17 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_26, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_26

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 1x93x768xf32)
        add_27 = paddle._C_ops.add(layer_norm_12, dropout_16)
        del dropout_16, layer_norm_12

        # pd_op.layer_norm: (1x93x768xf32, 1x93xf32, 1x93xf32) <- (1x93x768xf32, 768xf32, 768xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_27, parameter_153, parameter_152, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_27, parameter_152, parameter_153

        # pd_op.matmul: (1x93x3072xf32) <- (1x93x768xf32, 768x3072xf32)
        matmul_23 = paddle._C_ops.matmul(layer_norm_15, parameter_157, False, False)
        del parameter_157

        # pd_op.add: (1x93x3072xf32) <- (1x93x3072xf32, 3072xf32)
        add_28 = paddle._C_ops.add(matmul_23, parameter_156)
        del matmul_23, parameter_156

        # pd_op.gelu: (1x93x3072xf32) <- (1x93x3072xf32)
        gelu_2 = paddle._C_ops.gelu(add_28, False)
        del add_28

        # pd_op.matmul: (1x93x768xf32) <- (1x93x3072xf32, 3072x768xf32)
        matmul_24 = paddle._C_ops.matmul(gelu_2, parameter_155, False, False)
        del gelu_2, parameter_155

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_29 = paddle._C_ops.add(matmul_24, parameter_154)
        del matmul_24, parameter_154

        # pd_op.dropout: (1x93x768xf32, 1x93x768xui8) <- (1x93x768xf32, None, 1xf32)
        dropout_18, dropout_19 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_29, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_29

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 1x93x768xf32)
        add_30 = paddle._C_ops.add(layer_norm_15, dropout_18)
        del dropout_18, layer_norm_15

        # pd_op.layer_norm: (1x93x768xf32, 1x93xf32, 1x93xf32) <- (1x93x768xf32, 768xf32, 768xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_30, parameter_151, parameter_150, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_30, parameter_150, parameter_151

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_25 = paddle._C_ops.matmul(layer_norm_18, parameter_149, False, False)
        del parameter_149

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_31 = paddle._C_ops.add(matmul_25, parameter_148)
        del matmul_25, parameter_148

        # pd_op.reshape: (1x93x12x64xf32) <- (1x93x768xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(add_31, full_int_array_1)
        del add_31

        # pd_op.transpose: (1x12x93x64xf32) <- (1x93x12x64xf32)
        transpose_12 = paddle._C_ops.transpose(reshape_12, [0, 2, 1, 3])
        del reshape_12

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_26 = paddle._C_ops.matmul(layer_norm_18, parameter_147, False, False)
        del parameter_147

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_32 = paddle._C_ops.add(matmul_26, parameter_146)
        del matmul_26, parameter_146

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_27 = paddle._C_ops.matmul(layer_norm_18, parameter_145, False, False)
        del parameter_145

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_33 = paddle._C_ops.add(matmul_27, parameter_144)
        del matmul_27, parameter_144

        # pd_op.reshape: (1x93x12x64xf32) <- (1x93x768xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(add_32, full_int_array_1)
        del add_32

        # pd_op.transpose: (1x12x93x64xf32) <- (1x93x12x64xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_13, [0, 2, 1, 3])
        del reshape_13

        # pd_op.reshape: (1x93x12x64xf32) <- (1x93x768xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(add_33, full_int_array_1)
        del add_33

        # pd_op.transpose: (1x12x93x64xf32) <- (1x93x12x64xf32)
        transpose_14 = paddle._C_ops.transpose(reshape_14, [0, 2, 1, 3])
        del reshape_14

        # pd_op.scale: (1x12x93x64xf32) <- (1x12x93x64xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(transpose_12, full_8, float("0"), True)
        del transpose_12

        # pd_op.matmul: (1x12x93x93xf32) <- (1x12x93x64xf32, 1x12x93x64xf32)
        matmul_28 = paddle._C_ops.matmul(scale_4, transpose_13, False, True)
        del scale_4, transpose_13

        # pd_op.add: (1x12x93x93xf32) <- (1x12x93x93xf32, 1x1x1x93xf32)
        add_34 = paddle._C_ops.add(matmul_28, unsqueeze_0)
        del matmul_28

        # pd_op.softmax: (1x12x93x93xf32) <- (1x12x93x93xf32)
        softmax_3 = paddle._C_ops.softmax(add_34, -1)
        del add_34

        # pd_op.dropout: (1x12x93x93xf32, 1x12x93x93xui8) <- (1x12x93x93xf32, None, 1xf32)
        dropout_20, dropout_21 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_3, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_3

        # pd_op.matmul: (1x12x93x64xf32) <- (1x12x93x93xf32, 1x12x93x64xf32)
        matmul_29 = paddle._C_ops.matmul(dropout_20, transpose_14, False, False)
        del dropout_20, transpose_14

        # pd_op.transpose: (1x93x12x64xf32) <- (1x12x93x64xf32)
        transpose_15 = paddle._C_ops.transpose(matmul_29, [0, 2, 1, 3])
        del matmul_29

        # pd_op.reshape: (1x93x768xf32) <- (1x93x12x64xf32, 3xi64)
        reshape_15 = paddle._C_ops.reshape(transpose_15, full_int_array_2)
        del transpose_15

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_30 = paddle._C_ops.matmul(reshape_15, parameter_143, False, False)
        del parameter_143, reshape_15

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_35 = paddle._C_ops.add(matmul_30, parameter_142)
        del matmul_30, parameter_142

        # pd_op.dropout: (1x93x768xf32, 1x93x768xui8) <- (1x93x768xf32, None, 1xf32)
        dropout_22, dropout_23 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_35, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_35

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 1x93x768xf32)
        add_36 = paddle._C_ops.add(layer_norm_18, dropout_22)
        del dropout_22, layer_norm_18

        # pd_op.layer_norm: (1x93x768xf32, 1x93xf32, 1x93xf32) <- (1x93x768xf32, 768xf32, 768xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_36, parameter_137, parameter_136, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_36, parameter_136, parameter_137

        # pd_op.matmul: (1x93x3072xf32) <- (1x93x768xf32, 768x3072xf32)
        matmul_31 = paddle._C_ops.matmul(layer_norm_21, parameter_141, False, False)
        del parameter_141

        # pd_op.add: (1x93x3072xf32) <- (1x93x3072xf32, 3072xf32)
        add_37 = paddle._C_ops.add(matmul_31, parameter_140)
        del matmul_31, parameter_140

        # pd_op.gelu: (1x93x3072xf32) <- (1x93x3072xf32)
        gelu_3 = paddle._C_ops.gelu(add_37, False)
        del add_37

        # pd_op.matmul: (1x93x768xf32) <- (1x93x3072xf32, 3072x768xf32)
        matmul_32 = paddle._C_ops.matmul(gelu_3, parameter_139, False, False)
        del gelu_3, parameter_139

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_38 = paddle._C_ops.add(matmul_32, parameter_138)
        del matmul_32, parameter_138

        # pd_op.dropout: (1x93x768xf32, 1x93x768xui8) <- (1x93x768xf32, None, 1xf32)
        dropout_24, dropout_25 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_38, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_38

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 1x93x768xf32)
        add_39 = paddle._C_ops.add(layer_norm_21, dropout_24)
        del dropout_24, layer_norm_21

        # pd_op.layer_norm: (1x93x768xf32, 1x93xf32, 1x93xf32) <- (1x93x768xf32, 768xf32, 768xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_39, parameter_135, parameter_134, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_39, parameter_134, parameter_135

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_33 = paddle._C_ops.matmul(layer_norm_24, parameter_133, False, False)
        del parameter_133

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_40 = paddle._C_ops.add(matmul_33, parameter_132)
        del matmul_33, parameter_132

        # pd_op.reshape: (1x93x12x64xf32) <- (1x93x768xf32, 4xi64)
        reshape_16 = paddle._C_ops.reshape(add_40, full_int_array_1)
        del add_40

        # pd_op.transpose: (1x12x93x64xf32) <- (1x93x12x64xf32)
        transpose_16 = paddle._C_ops.transpose(reshape_16, [0, 2, 1, 3])
        del reshape_16

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_34 = paddle._C_ops.matmul(layer_norm_24, parameter_131, False, False)
        del parameter_131

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_41 = paddle._C_ops.add(matmul_34, parameter_130)
        del matmul_34, parameter_130

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_35 = paddle._C_ops.matmul(layer_norm_24, parameter_129, False, False)
        del parameter_129

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_42 = paddle._C_ops.add(matmul_35, parameter_128)
        del matmul_35, parameter_128

        # pd_op.reshape: (1x93x12x64xf32) <- (1x93x768xf32, 4xi64)
        reshape_17 = paddle._C_ops.reshape(add_41, full_int_array_1)
        del add_41

        # pd_op.transpose: (1x12x93x64xf32) <- (1x93x12x64xf32)
        transpose_17 = paddle._C_ops.transpose(reshape_17, [0, 2, 1, 3])
        del reshape_17

        # pd_op.reshape: (1x93x12x64xf32) <- (1x93x768xf32, 4xi64)
        reshape_18 = paddle._C_ops.reshape(add_42, full_int_array_1)
        del add_42

        # pd_op.transpose: (1x12x93x64xf32) <- (1x93x12x64xf32)
        transpose_18 = paddle._C_ops.transpose(reshape_18, [0, 2, 1, 3])
        del reshape_18

        # pd_op.scale: (1x12x93x64xf32) <- (1x12x93x64xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(transpose_16, full_8, float("0"), True)
        del transpose_16

        # pd_op.matmul: (1x12x93x93xf32) <- (1x12x93x64xf32, 1x12x93x64xf32)
        matmul_36 = paddle._C_ops.matmul(scale_5, transpose_17, False, True)
        del scale_5, transpose_17

        # pd_op.add: (1x12x93x93xf32) <- (1x12x93x93xf32, 1x1x1x93xf32)
        add_43 = paddle._C_ops.add(matmul_36, unsqueeze_0)
        del matmul_36

        # pd_op.softmax: (1x12x93x93xf32) <- (1x12x93x93xf32)
        softmax_4 = paddle._C_ops.softmax(add_43, -1)
        del add_43

        # pd_op.dropout: (1x12x93x93xf32, 1x12x93x93xui8) <- (1x12x93x93xf32, None, 1xf32)
        dropout_26, dropout_27 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_4, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_4

        # pd_op.matmul: (1x12x93x64xf32) <- (1x12x93x93xf32, 1x12x93x64xf32)
        matmul_37 = paddle._C_ops.matmul(dropout_26, transpose_18, False, False)
        del dropout_26, transpose_18

        # pd_op.transpose: (1x93x12x64xf32) <- (1x12x93x64xf32)
        transpose_19 = paddle._C_ops.transpose(matmul_37, [0, 2, 1, 3])
        del matmul_37

        # pd_op.reshape: (1x93x768xf32) <- (1x93x12x64xf32, 3xi64)
        reshape_19 = paddle._C_ops.reshape(transpose_19, full_int_array_2)
        del transpose_19

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_38 = paddle._C_ops.matmul(reshape_19, parameter_127, False, False)
        del parameter_127, reshape_19

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_44 = paddle._C_ops.add(matmul_38, parameter_126)
        del matmul_38, parameter_126

        # pd_op.dropout: (1x93x768xf32, 1x93x768xui8) <- (1x93x768xf32, None, 1xf32)
        dropout_28, dropout_29 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_44, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_44

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 1x93x768xf32)
        add_45 = paddle._C_ops.add(layer_norm_24, dropout_28)
        del dropout_28, layer_norm_24

        # pd_op.layer_norm: (1x93x768xf32, 1x93xf32, 1x93xf32) <- (1x93x768xf32, 768xf32, 768xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_45, parameter_121, parameter_120, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_45, parameter_120, parameter_121

        # pd_op.matmul: (1x93x3072xf32) <- (1x93x768xf32, 768x3072xf32)
        matmul_39 = paddle._C_ops.matmul(layer_norm_27, parameter_125, False, False)
        del parameter_125

        # pd_op.add: (1x93x3072xf32) <- (1x93x3072xf32, 3072xf32)
        add_46 = paddle._C_ops.add(matmul_39, parameter_124)
        del matmul_39, parameter_124

        # pd_op.gelu: (1x93x3072xf32) <- (1x93x3072xf32)
        gelu_4 = paddle._C_ops.gelu(add_46, False)
        del add_46

        # pd_op.matmul: (1x93x768xf32) <- (1x93x3072xf32, 3072x768xf32)
        matmul_40 = paddle._C_ops.matmul(gelu_4, parameter_123, False, False)
        del gelu_4, parameter_123

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_47 = paddle._C_ops.add(matmul_40, parameter_122)
        del matmul_40, parameter_122

        # pd_op.dropout: (1x93x768xf32, 1x93x768xui8) <- (1x93x768xf32, None, 1xf32)
        dropout_30, dropout_31 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_47, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_47

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 1x93x768xf32)
        add_48 = paddle._C_ops.add(layer_norm_27, dropout_30)
        del dropout_30, layer_norm_27

        # pd_op.layer_norm: (1x93x768xf32, 1x93xf32, 1x93xf32) <- (1x93x768xf32, 768xf32, 768xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_48, parameter_119, parameter_118, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_48, parameter_118, parameter_119

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_41 = paddle._C_ops.matmul(layer_norm_30, parameter_117, False, False)
        del parameter_117

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_49 = paddle._C_ops.add(matmul_41, parameter_116)
        del matmul_41, parameter_116

        # pd_op.reshape: (1x93x12x64xf32) <- (1x93x768xf32, 4xi64)
        reshape_20 = paddle._C_ops.reshape(add_49, full_int_array_1)
        del add_49

        # pd_op.transpose: (1x12x93x64xf32) <- (1x93x12x64xf32)
        transpose_20 = paddle._C_ops.transpose(reshape_20, [0, 2, 1, 3])
        del reshape_20

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_42 = paddle._C_ops.matmul(layer_norm_30, parameter_115, False, False)
        del parameter_115

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_50 = paddle._C_ops.add(matmul_42, parameter_114)
        del matmul_42, parameter_114

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_43 = paddle._C_ops.matmul(layer_norm_30, parameter_113, False, False)
        del parameter_113

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_51 = paddle._C_ops.add(matmul_43, parameter_112)
        del matmul_43, parameter_112

        # pd_op.reshape: (1x93x12x64xf32) <- (1x93x768xf32, 4xi64)
        reshape_21 = paddle._C_ops.reshape(add_50, full_int_array_1)
        del add_50

        # pd_op.transpose: (1x12x93x64xf32) <- (1x93x12x64xf32)
        transpose_21 = paddle._C_ops.transpose(reshape_21, [0, 2, 1, 3])
        del reshape_21

        # pd_op.reshape: (1x93x12x64xf32) <- (1x93x768xf32, 4xi64)
        reshape_22 = paddle._C_ops.reshape(add_51, full_int_array_1)
        del add_51

        # pd_op.transpose: (1x12x93x64xf32) <- (1x93x12x64xf32)
        transpose_22 = paddle._C_ops.transpose(reshape_22, [0, 2, 1, 3])
        del reshape_22

        # pd_op.scale: (1x12x93x64xf32) <- (1x12x93x64xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(transpose_20, full_8, float("0"), True)
        del transpose_20

        # pd_op.matmul: (1x12x93x93xf32) <- (1x12x93x64xf32, 1x12x93x64xf32)
        matmul_44 = paddle._C_ops.matmul(scale_6, transpose_21, False, True)
        del scale_6, transpose_21

        # pd_op.add: (1x12x93x93xf32) <- (1x12x93x93xf32, 1x1x1x93xf32)
        add_52 = paddle._C_ops.add(matmul_44, unsqueeze_0)
        del matmul_44

        # pd_op.softmax: (1x12x93x93xf32) <- (1x12x93x93xf32)
        softmax_5 = paddle._C_ops.softmax(add_52, -1)
        del add_52

        # pd_op.dropout: (1x12x93x93xf32, 1x12x93x93xui8) <- (1x12x93x93xf32, None, 1xf32)
        dropout_32, dropout_33 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_5, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_5

        # pd_op.matmul: (1x12x93x64xf32) <- (1x12x93x93xf32, 1x12x93x64xf32)
        matmul_45 = paddle._C_ops.matmul(dropout_32, transpose_22, False, False)
        del dropout_32, transpose_22

        # pd_op.transpose: (1x93x12x64xf32) <- (1x12x93x64xf32)
        transpose_23 = paddle._C_ops.transpose(matmul_45, [0, 2, 1, 3])
        del matmul_45

        # pd_op.reshape: (1x93x768xf32) <- (1x93x12x64xf32, 3xi64)
        reshape_23 = paddle._C_ops.reshape(transpose_23, full_int_array_2)
        del transpose_23

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_46 = paddle._C_ops.matmul(reshape_23, parameter_111, False, False)
        del parameter_111, reshape_23

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_53 = paddle._C_ops.add(matmul_46, parameter_110)
        del matmul_46, parameter_110

        # pd_op.dropout: (1x93x768xf32, 1x93x768xui8) <- (1x93x768xf32, None, 1xf32)
        dropout_34, dropout_35 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_53, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_53

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 1x93x768xf32)
        add_54 = paddle._C_ops.add(layer_norm_30, dropout_34)
        del dropout_34, layer_norm_30

        # pd_op.layer_norm: (1x93x768xf32, 1x93xf32, 1x93xf32) <- (1x93x768xf32, 768xf32, 768xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_54, parameter_105, parameter_104, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_54, parameter_104, parameter_105

        # pd_op.matmul: (1x93x3072xf32) <- (1x93x768xf32, 768x3072xf32)
        matmul_47 = paddle._C_ops.matmul(layer_norm_33, parameter_109, False, False)
        del parameter_109

        # pd_op.add: (1x93x3072xf32) <- (1x93x3072xf32, 3072xf32)
        add_55 = paddle._C_ops.add(matmul_47, parameter_108)
        del matmul_47, parameter_108

        # pd_op.gelu: (1x93x3072xf32) <- (1x93x3072xf32)
        gelu_5 = paddle._C_ops.gelu(add_55, False)
        del add_55

        # pd_op.matmul: (1x93x768xf32) <- (1x93x3072xf32, 3072x768xf32)
        matmul_48 = paddle._C_ops.matmul(gelu_5, parameter_107, False, False)
        del gelu_5, parameter_107

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_56 = paddle._C_ops.add(matmul_48, parameter_106)
        del matmul_48, parameter_106

        # pd_op.dropout: (1x93x768xf32, 1x93x768xui8) <- (1x93x768xf32, None, 1xf32)
        dropout_36, dropout_37 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_56, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_56

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 1x93x768xf32)
        add_57 = paddle._C_ops.add(layer_norm_33, dropout_36)
        del dropout_36, layer_norm_33

        # pd_op.layer_norm: (1x93x768xf32, 1x93xf32, 1x93xf32) <- (1x93x768xf32, 768xf32, 768xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_57, parameter_103, parameter_102, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_57, parameter_102, parameter_103

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_49 = paddle._C_ops.matmul(layer_norm_36, parameter_101, False, False)
        del parameter_101

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_58 = paddle._C_ops.add(matmul_49, parameter_100)
        del matmul_49, parameter_100

        # pd_op.reshape: (1x93x12x64xf32) <- (1x93x768xf32, 4xi64)
        reshape_24 = paddle._C_ops.reshape(add_58, full_int_array_1)
        del add_58

        # pd_op.transpose: (1x12x93x64xf32) <- (1x93x12x64xf32)
        transpose_24 = paddle._C_ops.transpose(reshape_24, [0, 2, 1, 3])
        del reshape_24

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_50 = paddle._C_ops.matmul(layer_norm_36, parameter_99, False, False)
        del parameter_99

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_59 = paddle._C_ops.add(matmul_50, parameter_98)
        del matmul_50, parameter_98

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_51 = paddle._C_ops.matmul(layer_norm_36, parameter_97, False, False)
        del parameter_97

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_60 = paddle._C_ops.add(matmul_51, parameter_96)
        del matmul_51, parameter_96

        # pd_op.reshape: (1x93x12x64xf32) <- (1x93x768xf32, 4xi64)
        reshape_25 = paddle._C_ops.reshape(add_59, full_int_array_1)
        del add_59

        # pd_op.transpose: (1x12x93x64xf32) <- (1x93x12x64xf32)
        transpose_25 = paddle._C_ops.transpose(reshape_25, [0, 2, 1, 3])
        del reshape_25

        # pd_op.reshape: (1x93x12x64xf32) <- (1x93x768xf32, 4xi64)
        reshape_26 = paddle._C_ops.reshape(add_60, full_int_array_1)
        del add_60

        # pd_op.transpose: (1x12x93x64xf32) <- (1x93x12x64xf32)
        transpose_26 = paddle._C_ops.transpose(reshape_26, [0, 2, 1, 3])
        del reshape_26

        # pd_op.scale: (1x12x93x64xf32) <- (1x12x93x64xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(transpose_24, full_8, float("0"), True)
        del transpose_24

        # pd_op.matmul: (1x12x93x93xf32) <- (1x12x93x64xf32, 1x12x93x64xf32)
        matmul_52 = paddle._C_ops.matmul(scale_7, transpose_25, False, True)
        del scale_7, transpose_25

        # pd_op.add: (1x12x93x93xf32) <- (1x12x93x93xf32, 1x1x1x93xf32)
        add_61 = paddle._C_ops.add(matmul_52, unsqueeze_0)
        del matmul_52

        # pd_op.softmax: (1x12x93x93xf32) <- (1x12x93x93xf32)
        softmax_6 = paddle._C_ops.softmax(add_61, -1)
        del add_61

        # pd_op.dropout: (1x12x93x93xf32, 1x12x93x93xui8) <- (1x12x93x93xf32, None, 1xf32)
        dropout_38, dropout_39 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_6, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_6

        # pd_op.matmul: (1x12x93x64xf32) <- (1x12x93x93xf32, 1x12x93x64xf32)
        matmul_53 = paddle._C_ops.matmul(dropout_38, transpose_26, False, False)
        del dropout_38, transpose_26

        # pd_op.transpose: (1x93x12x64xf32) <- (1x12x93x64xf32)
        transpose_27 = paddle._C_ops.transpose(matmul_53, [0, 2, 1, 3])
        del matmul_53

        # pd_op.reshape: (1x93x768xf32) <- (1x93x12x64xf32, 3xi64)
        reshape_27 = paddle._C_ops.reshape(transpose_27, full_int_array_2)
        del transpose_27

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_54 = paddle._C_ops.matmul(reshape_27, parameter_95, False, False)
        del parameter_95, reshape_27

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_62 = paddle._C_ops.add(matmul_54, parameter_94)
        del matmul_54, parameter_94

        # pd_op.dropout: (1x93x768xf32, 1x93x768xui8) <- (1x93x768xf32, None, 1xf32)
        dropout_40, dropout_41 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_62, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_62

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 1x93x768xf32)
        add_63 = paddle._C_ops.add(layer_norm_36, dropout_40)
        del dropout_40, layer_norm_36

        # pd_op.layer_norm: (1x93x768xf32, 1x93xf32, 1x93xf32) <- (1x93x768xf32, 768xf32, 768xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_63, parameter_89, parameter_88, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_63, parameter_88, parameter_89

        # pd_op.matmul: (1x93x3072xf32) <- (1x93x768xf32, 768x3072xf32)
        matmul_55 = paddle._C_ops.matmul(layer_norm_39, parameter_93, False, False)
        del parameter_93

        # pd_op.add: (1x93x3072xf32) <- (1x93x3072xf32, 3072xf32)
        add_64 = paddle._C_ops.add(matmul_55, parameter_92)
        del matmul_55, parameter_92

        # pd_op.gelu: (1x93x3072xf32) <- (1x93x3072xf32)
        gelu_6 = paddle._C_ops.gelu(add_64, False)
        del add_64

        # pd_op.matmul: (1x93x768xf32) <- (1x93x3072xf32, 3072x768xf32)
        matmul_56 = paddle._C_ops.matmul(gelu_6, parameter_91, False, False)
        del gelu_6, parameter_91

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_65 = paddle._C_ops.add(matmul_56, parameter_90)
        del matmul_56, parameter_90

        # pd_op.dropout: (1x93x768xf32, 1x93x768xui8) <- (1x93x768xf32, None, 1xf32)
        dropout_42, dropout_43 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_65, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_65

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 1x93x768xf32)
        add_66 = paddle._C_ops.add(layer_norm_39, dropout_42)
        del dropout_42, layer_norm_39

        # pd_op.layer_norm: (1x93x768xf32, 1x93xf32, 1x93xf32) <- (1x93x768xf32, 768xf32, 768xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_66, parameter_87, parameter_86, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_66, parameter_86, parameter_87

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_57 = paddle._C_ops.matmul(layer_norm_42, parameter_85, False, False)
        del parameter_85

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_67 = paddle._C_ops.add(matmul_57, parameter_84)
        del matmul_57, parameter_84

        # pd_op.reshape: (1x93x12x64xf32) <- (1x93x768xf32, 4xi64)
        reshape_28 = paddle._C_ops.reshape(add_67, full_int_array_1)
        del add_67

        # pd_op.transpose: (1x12x93x64xf32) <- (1x93x12x64xf32)
        transpose_28 = paddle._C_ops.transpose(reshape_28, [0, 2, 1, 3])
        del reshape_28

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_58 = paddle._C_ops.matmul(layer_norm_42, parameter_83, False, False)
        del parameter_83

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_68 = paddle._C_ops.add(matmul_58, parameter_82)
        del matmul_58, parameter_82

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_59 = paddle._C_ops.matmul(layer_norm_42, parameter_81, False, False)
        del parameter_81

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_69 = paddle._C_ops.add(matmul_59, parameter_80)
        del matmul_59, parameter_80

        # pd_op.reshape: (1x93x12x64xf32) <- (1x93x768xf32, 4xi64)
        reshape_29 = paddle._C_ops.reshape(add_68, full_int_array_1)
        del add_68

        # pd_op.transpose: (1x12x93x64xf32) <- (1x93x12x64xf32)
        transpose_29 = paddle._C_ops.transpose(reshape_29, [0, 2, 1, 3])
        del reshape_29

        # pd_op.reshape: (1x93x12x64xf32) <- (1x93x768xf32, 4xi64)
        reshape_30 = paddle._C_ops.reshape(add_69, full_int_array_1)
        del add_69

        # pd_op.transpose: (1x12x93x64xf32) <- (1x93x12x64xf32)
        transpose_30 = paddle._C_ops.transpose(reshape_30, [0, 2, 1, 3])
        del reshape_30

        # pd_op.scale: (1x12x93x64xf32) <- (1x12x93x64xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(transpose_28, full_8, float("0"), True)
        del transpose_28

        # pd_op.matmul: (1x12x93x93xf32) <- (1x12x93x64xf32, 1x12x93x64xf32)
        matmul_60 = paddle._C_ops.matmul(scale_8, transpose_29, False, True)
        del scale_8, transpose_29

        # pd_op.add: (1x12x93x93xf32) <- (1x12x93x93xf32, 1x1x1x93xf32)
        add_70 = paddle._C_ops.add(matmul_60, unsqueeze_0)
        del matmul_60

        # pd_op.softmax: (1x12x93x93xf32) <- (1x12x93x93xf32)
        softmax_7 = paddle._C_ops.softmax(add_70, -1)
        del add_70

        # pd_op.dropout: (1x12x93x93xf32, 1x12x93x93xui8) <- (1x12x93x93xf32, None, 1xf32)
        dropout_44, dropout_45 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_7, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_7

        # pd_op.matmul: (1x12x93x64xf32) <- (1x12x93x93xf32, 1x12x93x64xf32)
        matmul_61 = paddle._C_ops.matmul(dropout_44, transpose_30, False, False)
        del dropout_44, transpose_30

        # pd_op.transpose: (1x93x12x64xf32) <- (1x12x93x64xf32)
        transpose_31 = paddle._C_ops.transpose(matmul_61, [0, 2, 1, 3])
        del matmul_61

        # pd_op.reshape: (1x93x768xf32) <- (1x93x12x64xf32, 3xi64)
        reshape_31 = paddle._C_ops.reshape(transpose_31, full_int_array_2)
        del transpose_31

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_62 = paddle._C_ops.matmul(reshape_31, parameter_79, False, False)
        del parameter_79, reshape_31

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_71 = paddle._C_ops.add(matmul_62, parameter_78)
        del matmul_62, parameter_78

        # pd_op.dropout: (1x93x768xf32, 1x93x768xui8) <- (1x93x768xf32, None, 1xf32)
        dropout_46, dropout_47 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_71, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_71

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 1x93x768xf32)
        add_72 = paddle._C_ops.add(layer_norm_42, dropout_46)
        del dropout_46, layer_norm_42

        # pd_op.layer_norm: (1x93x768xf32, 1x93xf32, 1x93xf32) <- (1x93x768xf32, 768xf32, 768xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_72, parameter_73, parameter_72, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_72, parameter_72, parameter_73

        # pd_op.matmul: (1x93x3072xf32) <- (1x93x768xf32, 768x3072xf32)
        matmul_63 = paddle._C_ops.matmul(layer_norm_45, parameter_77, False, False)
        del parameter_77

        # pd_op.add: (1x93x3072xf32) <- (1x93x3072xf32, 3072xf32)
        add_73 = paddle._C_ops.add(matmul_63, parameter_76)
        del matmul_63, parameter_76

        # pd_op.gelu: (1x93x3072xf32) <- (1x93x3072xf32)
        gelu_7 = paddle._C_ops.gelu(add_73, False)
        del add_73

        # pd_op.matmul: (1x93x768xf32) <- (1x93x3072xf32, 3072x768xf32)
        matmul_64 = paddle._C_ops.matmul(gelu_7, parameter_75, False, False)
        del gelu_7, parameter_75

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_74 = paddle._C_ops.add(matmul_64, parameter_74)
        del matmul_64, parameter_74

        # pd_op.dropout: (1x93x768xf32, 1x93x768xui8) <- (1x93x768xf32, None, 1xf32)
        dropout_48, dropout_49 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_74, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_74

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 1x93x768xf32)
        add_75 = paddle._C_ops.add(layer_norm_45, dropout_48)
        del dropout_48, layer_norm_45

        # pd_op.layer_norm: (1x93x768xf32, 1x93xf32, 1x93xf32) <- (1x93x768xf32, 768xf32, 768xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_75, parameter_71, parameter_70, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_75, parameter_70, parameter_71

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_65 = paddle._C_ops.matmul(layer_norm_48, parameter_69, False, False)
        del parameter_69

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_76 = paddle._C_ops.add(matmul_65, parameter_68)
        del matmul_65, parameter_68

        # pd_op.reshape: (1x93x12x64xf32) <- (1x93x768xf32, 4xi64)
        reshape_32 = paddle._C_ops.reshape(add_76, full_int_array_1)
        del add_76

        # pd_op.transpose: (1x12x93x64xf32) <- (1x93x12x64xf32)
        transpose_32 = paddle._C_ops.transpose(reshape_32, [0, 2, 1, 3])
        del reshape_32

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_66 = paddle._C_ops.matmul(layer_norm_48, parameter_67, False, False)
        del parameter_67

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_77 = paddle._C_ops.add(matmul_66, parameter_66)
        del matmul_66, parameter_66

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_67 = paddle._C_ops.matmul(layer_norm_48, parameter_65, False, False)
        del parameter_65

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_78 = paddle._C_ops.add(matmul_67, parameter_64)
        del matmul_67, parameter_64

        # pd_op.reshape: (1x93x12x64xf32) <- (1x93x768xf32, 4xi64)
        reshape_33 = paddle._C_ops.reshape(add_77, full_int_array_1)
        del add_77

        # pd_op.transpose: (1x12x93x64xf32) <- (1x93x12x64xf32)
        transpose_33 = paddle._C_ops.transpose(reshape_33, [0, 2, 1, 3])
        del reshape_33

        # pd_op.reshape: (1x93x12x64xf32) <- (1x93x768xf32, 4xi64)
        reshape_34 = paddle._C_ops.reshape(add_78, full_int_array_1)
        del add_78

        # pd_op.transpose: (1x12x93x64xf32) <- (1x93x12x64xf32)
        transpose_34 = paddle._C_ops.transpose(reshape_34, [0, 2, 1, 3])
        del reshape_34

        # pd_op.scale: (1x12x93x64xf32) <- (1x12x93x64xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(transpose_32, full_8, float("0"), True)
        del transpose_32

        # pd_op.matmul: (1x12x93x93xf32) <- (1x12x93x64xf32, 1x12x93x64xf32)
        matmul_68 = paddle._C_ops.matmul(scale_9, transpose_33, False, True)
        del scale_9, transpose_33

        # pd_op.add: (1x12x93x93xf32) <- (1x12x93x93xf32, 1x1x1x93xf32)
        add_79 = paddle._C_ops.add(matmul_68, unsqueeze_0)
        del matmul_68

        # pd_op.softmax: (1x12x93x93xf32) <- (1x12x93x93xf32)
        softmax_8 = paddle._C_ops.softmax(add_79, -1)
        del add_79

        # pd_op.dropout: (1x12x93x93xf32, 1x12x93x93xui8) <- (1x12x93x93xf32, None, 1xf32)
        dropout_50, dropout_51 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_8, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_8

        # pd_op.matmul: (1x12x93x64xf32) <- (1x12x93x93xf32, 1x12x93x64xf32)
        matmul_69 = paddle._C_ops.matmul(dropout_50, transpose_34, False, False)
        del dropout_50, transpose_34

        # pd_op.transpose: (1x93x12x64xf32) <- (1x12x93x64xf32)
        transpose_35 = paddle._C_ops.transpose(matmul_69, [0, 2, 1, 3])
        del matmul_69

        # pd_op.reshape: (1x93x768xf32) <- (1x93x12x64xf32, 3xi64)
        reshape_35 = paddle._C_ops.reshape(transpose_35, full_int_array_2)
        del transpose_35

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_70 = paddle._C_ops.matmul(reshape_35, parameter_63, False, False)
        del parameter_63, reshape_35

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_80 = paddle._C_ops.add(matmul_70, parameter_62)
        del matmul_70, parameter_62

        # pd_op.dropout: (1x93x768xf32, 1x93x768xui8) <- (1x93x768xf32, None, 1xf32)
        dropout_52, dropout_53 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_80, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_80

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 1x93x768xf32)
        add_81 = paddle._C_ops.add(layer_norm_48, dropout_52)
        del dropout_52, layer_norm_48

        # pd_op.layer_norm: (1x93x768xf32, 1x93xf32, 1x93xf32) <- (1x93x768xf32, 768xf32, 768xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_81, parameter_57, parameter_56, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_81, parameter_56, parameter_57

        # pd_op.matmul: (1x93x3072xf32) <- (1x93x768xf32, 768x3072xf32)
        matmul_71 = paddle._C_ops.matmul(layer_norm_51, parameter_61, False, False)
        del parameter_61

        # pd_op.add: (1x93x3072xf32) <- (1x93x3072xf32, 3072xf32)
        add_82 = paddle._C_ops.add(matmul_71, parameter_60)
        del matmul_71, parameter_60

        # pd_op.gelu: (1x93x3072xf32) <- (1x93x3072xf32)
        gelu_8 = paddle._C_ops.gelu(add_82, False)
        del add_82

        # pd_op.matmul: (1x93x768xf32) <- (1x93x3072xf32, 3072x768xf32)
        matmul_72 = paddle._C_ops.matmul(gelu_8, parameter_59, False, False)
        del gelu_8, parameter_59

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_83 = paddle._C_ops.add(matmul_72, parameter_58)
        del matmul_72, parameter_58

        # pd_op.dropout: (1x93x768xf32, 1x93x768xui8) <- (1x93x768xf32, None, 1xf32)
        dropout_54, dropout_55 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_83, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_83

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 1x93x768xf32)
        add_84 = paddle._C_ops.add(layer_norm_51, dropout_54)
        del dropout_54, layer_norm_51

        # pd_op.layer_norm: (1x93x768xf32, 1x93xf32, 1x93xf32) <- (1x93x768xf32, 768xf32, 768xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_84, parameter_55, parameter_54, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_84, parameter_54, parameter_55

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_73 = paddle._C_ops.matmul(layer_norm_54, parameter_53, False, False)
        del parameter_53

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_85 = paddle._C_ops.add(matmul_73, parameter_52)
        del matmul_73, parameter_52

        # pd_op.reshape: (1x93x12x64xf32) <- (1x93x768xf32, 4xi64)
        reshape_36 = paddle._C_ops.reshape(add_85, full_int_array_1)
        del add_85

        # pd_op.transpose: (1x12x93x64xf32) <- (1x93x12x64xf32)
        transpose_36 = paddle._C_ops.transpose(reshape_36, [0, 2, 1, 3])
        del reshape_36

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_74 = paddle._C_ops.matmul(layer_norm_54, parameter_51, False, False)
        del parameter_51

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_86 = paddle._C_ops.add(matmul_74, parameter_50)
        del matmul_74, parameter_50

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_75 = paddle._C_ops.matmul(layer_norm_54, parameter_49, False, False)
        del parameter_49

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_87 = paddle._C_ops.add(matmul_75, parameter_48)
        del matmul_75, parameter_48

        # pd_op.reshape: (1x93x12x64xf32) <- (1x93x768xf32, 4xi64)
        reshape_37 = paddle._C_ops.reshape(add_86, full_int_array_1)
        del add_86

        # pd_op.transpose: (1x12x93x64xf32) <- (1x93x12x64xf32)
        transpose_37 = paddle._C_ops.transpose(reshape_37, [0, 2, 1, 3])
        del reshape_37

        # pd_op.reshape: (1x93x12x64xf32) <- (1x93x768xf32, 4xi64)
        reshape_38 = paddle._C_ops.reshape(add_87, full_int_array_1)
        del add_87

        # pd_op.transpose: (1x12x93x64xf32) <- (1x93x12x64xf32)
        transpose_38 = paddle._C_ops.transpose(reshape_38, [0, 2, 1, 3])
        del reshape_38

        # pd_op.scale: (1x12x93x64xf32) <- (1x12x93x64xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(transpose_36, full_8, float("0"), True)
        del transpose_36

        # pd_op.matmul: (1x12x93x93xf32) <- (1x12x93x64xf32, 1x12x93x64xf32)
        matmul_76 = paddle._C_ops.matmul(scale_10, transpose_37, False, True)
        del scale_10, transpose_37

        # pd_op.add: (1x12x93x93xf32) <- (1x12x93x93xf32, 1x1x1x93xf32)
        add_88 = paddle._C_ops.add(matmul_76, unsqueeze_0)
        del matmul_76

        # pd_op.softmax: (1x12x93x93xf32) <- (1x12x93x93xf32)
        softmax_9 = paddle._C_ops.softmax(add_88, -1)
        del add_88

        # pd_op.dropout: (1x12x93x93xf32, 1x12x93x93xui8) <- (1x12x93x93xf32, None, 1xf32)
        dropout_56, dropout_57 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_9, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_9

        # pd_op.matmul: (1x12x93x64xf32) <- (1x12x93x93xf32, 1x12x93x64xf32)
        matmul_77 = paddle._C_ops.matmul(dropout_56, transpose_38, False, False)
        del dropout_56, transpose_38

        # pd_op.transpose: (1x93x12x64xf32) <- (1x12x93x64xf32)
        transpose_39 = paddle._C_ops.transpose(matmul_77, [0, 2, 1, 3])
        del matmul_77

        # pd_op.reshape: (1x93x768xf32) <- (1x93x12x64xf32, 3xi64)
        reshape_39 = paddle._C_ops.reshape(transpose_39, full_int_array_2)
        del transpose_39

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_78 = paddle._C_ops.matmul(reshape_39, parameter_47, False, False)
        del parameter_47, reshape_39

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_89 = paddle._C_ops.add(matmul_78, parameter_46)
        del matmul_78, parameter_46

        # pd_op.dropout: (1x93x768xf32, 1x93x768xui8) <- (1x93x768xf32, None, 1xf32)
        dropout_58, dropout_59 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_89, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_89

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 1x93x768xf32)
        add_90 = paddle._C_ops.add(layer_norm_54, dropout_58)
        del dropout_58, layer_norm_54

        # pd_op.layer_norm: (1x93x768xf32, 1x93xf32, 1x93xf32) <- (1x93x768xf32, 768xf32, 768xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_90, parameter_41, parameter_40, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_90, parameter_40, parameter_41

        # pd_op.matmul: (1x93x3072xf32) <- (1x93x768xf32, 768x3072xf32)
        matmul_79 = paddle._C_ops.matmul(layer_norm_57, parameter_45, False, False)
        del parameter_45

        # pd_op.add: (1x93x3072xf32) <- (1x93x3072xf32, 3072xf32)
        add_91 = paddle._C_ops.add(matmul_79, parameter_44)
        del matmul_79, parameter_44

        # pd_op.gelu: (1x93x3072xf32) <- (1x93x3072xf32)
        gelu_9 = paddle._C_ops.gelu(add_91, False)
        del add_91

        # pd_op.matmul: (1x93x768xf32) <- (1x93x3072xf32, 3072x768xf32)
        matmul_80 = paddle._C_ops.matmul(gelu_9, parameter_43, False, False)
        del gelu_9, parameter_43

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_92 = paddle._C_ops.add(matmul_80, parameter_42)
        del matmul_80, parameter_42

        # pd_op.dropout: (1x93x768xf32, 1x93x768xui8) <- (1x93x768xf32, None, 1xf32)
        dropout_60, dropout_61 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_92, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_92

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 1x93x768xf32)
        add_93 = paddle._C_ops.add(layer_norm_57, dropout_60)
        del dropout_60, layer_norm_57

        # pd_op.layer_norm: (1x93x768xf32, 1x93xf32, 1x93xf32) <- (1x93x768xf32, 768xf32, 768xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_93, parameter_39, parameter_38, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_93, parameter_38, parameter_39

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_81 = paddle._C_ops.matmul(layer_norm_60, parameter_37, False, False)
        del parameter_37

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_94 = paddle._C_ops.add(matmul_81, parameter_36)
        del matmul_81, parameter_36

        # pd_op.reshape: (1x93x12x64xf32) <- (1x93x768xf32, 4xi64)
        reshape_40 = paddle._C_ops.reshape(add_94, full_int_array_1)
        del add_94

        # pd_op.transpose: (1x12x93x64xf32) <- (1x93x12x64xf32)
        transpose_40 = paddle._C_ops.transpose(reshape_40, [0, 2, 1, 3])
        del reshape_40

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_82 = paddle._C_ops.matmul(layer_norm_60, parameter_35, False, False)
        del parameter_35

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_95 = paddle._C_ops.add(matmul_82, parameter_34)
        del matmul_82, parameter_34

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_83 = paddle._C_ops.matmul(layer_norm_60, parameter_33, False, False)
        del parameter_33

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_96 = paddle._C_ops.add(matmul_83, parameter_32)
        del matmul_83, parameter_32

        # pd_op.reshape: (1x93x12x64xf32) <- (1x93x768xf32, 4xi64)
        reshape_41 = paddle._C_ops.reshape(add_95, full_int_array_1)
        del add_95

        # pd_op.transpose: (1x12x93x64xf32) <- (1x93x12x64xf32)
        transpose_41 = paddle._C_ops.transpose(reshape_41, [0, 2, 1, 3])
        del reshape_41

        # pd_op.reshape: (1x93x12x64xf32) <- (1x93x768xf32, 4xi64)
        reshape_42 = paddle._C_ops.reshape(add_96, full_int_array_1)
        del add_96

        # pd_op.transpose: (1x12x93x64xf32) <- (1x93x12x64xf32)
        transpose_42 = paddle._C_ops.transpose(reshape_42, [0, 2, 1, 3])
        del reshape_42

        # pd_op.scale: (1x12x93x64xf32) <- (1x12x93x64xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(transpose_40, full_8, float("0"), True)
        del transpose_40

        # pd_op.matmul: (1x12x93x93xf32) <- (1x12x93x64xf32, 1x12x93x64xf32)
        matmul_84 = paddle._C_ops.matmul(scale_11, transpose_41, False, True)
        del scale_11, transpose_41

        # pd_op.add: (1x12x93x93xf32) <- (1x12x93x93xf32, 1x1x1x93xf32)
        add_97 = paddle._C_ops.add(matmul_84, unsqueeze_0)
        del matmul_84

        # pd_op.softmax: (1x12x93x93xf32) <- (1x12x93x93xf32)
        softmax_10 = paddle._C_ops.softmax(add_97, -1)
        del add_97

        # pd_op.dropout: (1x12x93x93xf32, 1x12x93x93xui8) <- (1x12x93x93xf32, None, 1xf32)
        dropout_62, dropout_63 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_10, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_10

        # pd_op.matmul: (1x12x93x64xf32) <- (1x12x93x93xf32, 1x12x93x64xf32)
        matmul_85 = paddle._C_ops.matmul(dropout_62, transpose_42, False, False)
        del dropout_62, transpose_42

        # pd_op.transpose: (1x93x12x64xf32) <- (1x12x93x64xf32)
        transpose_43 = paddle._C_ops.transpose(matmul_85, [0, 2, 1, 3])
        del matmul_85

        # pd_op.reshape: (1x93x768xf32) <- (1x93x12x64xf32, 3xi64)
        reshape_43 = paddle._C_ops.reshape(transpose_43, full_int_array_2)
        del transpose_43

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_86 = paddle._C_ops.matmul(reshape_43, parameter_31, False, False)
        del parameter_31, reshape_43

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_98 = paddle._C_ops.add(matmul_86, parameter_30)
        del matmul_86, parameter_30

        # pd_op.dropout: (1x93x768xf32, 1x93x768xui8) <- (1x93x768xf32, None, 1xf32)
        dropout_64, dropout_65 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_98, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_98

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 1x93x768xf32)
        add_99 = paddle._C_ops.add(layer_norm_60, dropout_64)
        del dropout_64, layer_norm_60

        # pd_op.layer_norm: (1x93x768xf32, 1x93xf32, 1x93xf32) <- (1x93x768xf32, 768xf32, 768xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_99, parameter_25, parameter_24, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_99, parameter_24, parameter_25

        # pd_op.matmul: (1x93x3072xf32) <- (1x93x768xf32, 768x3072xf32)
        matmul_87 = paddle._C_ops.matmul(layer_norm_63, parameter_29, False, False)
        del parameter_29

        # pd_op.add: (1x93x3072xf32) <- (1x93x3072xf32, 3072xf32)
        add_100 = paddle._C_ops.add(matmul_87, parameter_28)
        del matmul_87, parameter_28

        # pd_op.gelu: (1x93x3072xf32) <- (1x93x3072xf32)
        gelu_10 = paddle._C_ops.gelu(add_100, False)
        del add_100

        # pd_op.matmul: (1x93x768xf32) <- (1x93x3072xf32, 3072x768xf32)
        matmul_88 = paddle._C_ops.matmul(gelu_10, parameter_27, False, False)
        del gelu_10, parameter_27

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_101 = paddle._C_ops.add(matmul_88, parameter_26)
        del matmul_88, parameter_26

        # pd_op.dropout: (1x93x768xf32, 1x93x768xui8) <- (1x93x768xf32, None, 1xf32)
        dropout_66, dropout_67 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_101, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_101

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 1x93x768xf32)
        add_102 = paddle._C_ops.add(layer_norm_63, dropout_66)
        del dropout_66, layer_norm_63

        # pd_op.layer_norm: (1x93x768xf32, 1x93xf32, 1x93xf32) <- (1x93x768xf32, 768xf32, 768xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_102, parameter_23, parameter_22, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_102, parameter_22, parameter_23

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_89 = paddle._C_ops.matmul(layer_norm_66, parameter_21, False, False)
        del parameter_21

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_103 = paddle._C_ops.add(matmul_89, parameter_20)
        del matmul_89, parameter_20

        # pd_op.reshape: (1x93x12x64xf32) <- (1x93x768xf32, 4xi64)
        reshape_44 = paddle._C_ops.reshape(add_103, full_int_array_1)
        del add_103

        # pd_op.transpose: (1x12x93x64xf32) <- (1x93x12x64xf32)
        transpose_44 = paddle._C_ops.transpose(reshape_44, [0, 2, 1, 3])
        del reshape_44

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_90 = paddle._C_ops.matmul(layer_norm_66, parameter_19, False, False)
        del parameter_19

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_104 = paddle._C_ops.add(matmul_90, parameter_18)
        del matmul_90, parameter_18

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_91 = paddle._C_ops.matmul(layer_norm_66, parameter_17, False, False)
        del parameter_17

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_105 = paddle._C_ops.add(matmul_91, parameter_16)
        del matmul_91, parameter_16

        # pd_op.reshape: (1x93x12x64xf32) <- (1x93x768xf32, 4xi64)
        reshape_45 = paddle._C_ops.reshape(add_104, full_int_array_1)
        del add_104

        # pd_op.transpose: (1x12x93x64xf32) <- (1x93x12x64xf32)
        transpose_45 = paddle._C_ops.transpose(reshape_45, [0, 2, 1, 3])
        del reshape_45

        # pd_op.reshape: (1x93x12x64xf32) <- (1x93x768xf32, 4xi64)
        reshape_46 = paddle._C_ops.reshape(add_105, full_int_array_1)
        del add_105, full_int_array_1

        # pd_op.transpose: (1x12x93x64xf32) <- (1x93x12x64xf32)
        transpose_46 = paddle._C_ops.transpose(reshape_46, [0, 2, 1, 3])
        del reshape_46

        # pd_op.scale: (1x12x93x64xf32) <- (1x12x93x64xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(transpose_44, full_8, float("0"), True)
        del full_8, transpose_44

        # pd_op.matmul: (1x12x93x93xf32) <- (1x12x93x64xf32, 1x12x93x64xf32)
        matmul_92 = paddle._C_ops.matmul(scale_12, transpose_45, False, True)
        del scale_12, transpose_45

        # pd_op.add: (1x12x93x93xf32) <- (1x12x93x93xf32, 1x1x1x93xf32)
        add_106 = paddle._C_ops.add(matmul_92, unsqueeze_0)
        del matmul_92, unsqueeze_0

        # pd_op.softmax: (1x12x93x93xf32) <- (1x12x93x93xf32)
        softmax_11 = paddle._C_ops.softmax(add_106, -1)
        del add_106

        # pd_op.dropout: (1x12x93x93xf32, 1x12x93x93xui8) <- (1x12x93x93xf32, None, 1xf32)
        dropout_68, dropout_69 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_11, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_11

        # pd_op.matmul: (1x12x93x64xf32) <- (1x12x93x93xf32, 1x12x93x64xf32)
        matmul_93 = paddle._C_ops.matmul(dropout_68, transpose_46, False, False)
        del dropout_68, transpose_46

        # pd_op.transpose: (1x93x12x64xf32) <- (1x12x93x64xf32)
        transpose_47 = paddle._C_ops.transpose(matmul_93, [0, 2, 1, 3])
        del matmul_93

        # pd_op.reshape: (1x93x768xf32) <- (1x93x12x64xf32, 3xi64)
        reshape_47 = paddle._C_ops.reshape(transpose_47, full_int_array_2)
        del full_int_array_2, transpose_47

        # pd_op.matmul: (1x93x768xf32) <- (1x93x768xf32, 768x768xf32)
        matmul_94 = paddle._C_ops.matmul(reshape_47, parameter_15, False, False)
        del parameter_15, reshape_47

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_107 = paddle._C_ops.add(matmul_94, parameter_14)
        del matmul_94, parameter_14

        # pd_op.dropout: (1x93x768xf32, 1x93x768xui8) <- (1x93x768xf32, None, 1xf32)
        dropout_70, dropout_71 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_107, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_107

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 1x93x768xf32)
        add_108 = paddle._C_ops.add(layer_norm_66, dropout_70)
        del dropout_70, layer_norm_66

        # pd_op.layer_norm: (1x93x768xf32, 1x93xf32, 1x93xf32) <- (1x93x768xf32, 768xf32, 768xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_108, parameter_9, parameter_8, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_108, parameter_8, parameter_9

        # pd_op.matmul: (1x93x3072xf32) <- (1x93x768xf32, 768x3072xf32)
        matmul_95 = paddle._C_ops.matmul(layer_norm_69, parameter_13, False, False)
        del parameter_13

        # pd_op.add: (1x93x3072xf32) <- (1x93x3072xf32, 3072xf32)
        add_109 = paddle._C_ops.add(matmul_95, parameter_12)
        del matmul_95, parameter_12

        # pd_op.gelu: (1x93x3072xf32) <- (1x93x3072xf32)
        gelu_11 = paddle._C_ops.gelu(add_109, False)
        del add_109

        # pd_op.matmul: (1x93x768xf32) <- (1x93x3072xf32, 3072x768xf32)
        matmul_96 = paddle._C_ops.matmul(gelu_11, parameter_11, False, False)
        del gelu_11, parameter_11

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_110 = paddle._C_ops.add(matmul_96, parameter_10)
        del matmul_96, parameter_10

        # pd_op.dropout: (1x93x768xf32, 1x93x768xui8) <- (1x93x768xf32, None, 1xf32)
        dropout_72, dropout_73 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_110, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_110, full_7

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 1x93x768xf32)
        add_111 = paddle._C_ops.add(layer_norm_69, dropout_72)
        del dropout_72, layer_norm_69

        # pd_op.layer_norm: (1x93x768xf32, 1x93xf32, 1x93xf32) <- (1x93x768xf32, 768xf32, 768xf32)
        layer_norm_72, layer_norm_73, layer_norm_74 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_111, parameter_7, parameter_6, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_111, parameter_6, parameter_7

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [1]

        # pd_op.slice: (1x768xf32) <- (1x93x768xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            layer_norm_72, [1], full_int_array_3, full_int_array_4, [1], [1]
        )
        del full_int_array_3

        # pd_op.matmul: (1x768xf32) <- (1x768xf32, 768x768xf32)
        matmul_97 = paddle._C_ops.matmul(slice_0, parameter_5, False, False)
        del parameter_5, slice_0

        # pd_op.add: (1x768xf32) <- (1x768xf32, 768xf32)
        add_112 = paddle._C_ops.add(matmul_97, parameter_4)
        del matmul_97, parameter_4

        # pd_op.tanh: (1x768xf32) <- (1x768xf32)
        tanh_0 = paddle._C_ops.tanh(add_112)
        del add_112

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [2]

        # pd_op.slice: (1x768xf32) <- (1x93x768xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            layer_norm_72, [1], full_int_array_4, full_int_array_5, [1], [1]
        )
        del full_int_array_5

        # pd_op.unsqueeze: (1x1x768xf32) <- (1x768xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(slice_1, full_int_array_4)
        del full_int_array_4, slice_1

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_6 = [1, 93, 768]

        # pd_op.expand: (1x93x768xf32) <- (1x1x768xf32, 3xi64)
        expand_0 = paddle._C_ops.expand(unsqueeze_1, full_int_array_6)
        del full_int_array_6, unsqueeze_1

        # pd_op.full: (1xi32) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("2"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([1x93x768xf32, 1x93x768xf32]) <- (1x93x768xf32, 1x93x768xf32)
        combine_1 = [layer_norm_72, expand_0]
        del layer_norm_72

        # pd_op.concat: (1x93x1536xf32) <- ([1x93x768xf32, 1x93x768xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_9)
        del combine_1, full_9

        # pd_op.matmul: (1x93x3072xf32) <- (1x93x1536xf32, 1536x3072xf32)
        matmul_98 = paddle._C_ops.matmul(concat_1, parameter_3, False, False)
        del concat_1, parameter_3

        # pd_op.add: (1x93x3072xf32) <- (1x93x3072xf32, 3072xf32)
        add_113 = paddle._C_ops.add(matmul_98, parameter_2)
        del matmul_98, parameter_2

        # pd_op.matmul: (1x93x768xf32) <- (1x93x3072xf32, 3072x768xf32)
        matmul_99 = paddle._C_ops.matmul(add_113, parameter_1, False, False)
        del add_113, parameter_1

        # pd_op.add: (1x93x768xf32) <- (1x93x768xf32, 768xf32)
        add_0 = paddle._C_ops.add(matmul_99, parameter_0)
        del expand_0, matmul_99, parameter_0

        return add_0, tanh_0
