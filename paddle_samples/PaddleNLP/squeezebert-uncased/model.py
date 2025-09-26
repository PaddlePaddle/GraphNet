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
        data_0,
        data_1,
        data_2,
    ):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [21]

        # pd_op.slice: (1x21xi64) <- (1x512xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            parameter_0, [1], full_int_array_0, full_int_array_1, [1], []
        )
        del full_int_array_1, parameter_0

        # pd_op.embedding: (1x21x768xf32) <- (1x21xi64, 30528x768xf32)
        embedding_0 = paddle._C_ops.embedding(data_0, parameter_199, -1, False)
        del data_0, parameter_199

        # pd_op.embedding: (1x21x768xf32) <- (1x21xi64, 512x768xf32)
        embedding_1 = paddle._C_ops.embedding(slice_0, parameter_198, -1, False)
        del parameter_198, slice_0

        # pd_op.embedding: (1x21x768xf32) <- (1x21xi64, 2x768xf32)
        embedding_2 = paddle._C_ops.embedding(data_2, parameter_197, -1, False)
        del data_2, parameter_197

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_0 = paddle._C_ops.add(embedding_0, embedding_1)
        del embedding_0, embedding_1

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_1 = paddle._C_ops.add(add_0, embedding_2)
        del add_0, embedding_2

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_1, parameter_196, parameter_195, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_1, parameter_195, parameter_196

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                layer_norm_0, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del layer_norm_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [1]

        # pd_op.unsqueeze: (1x1x21xi64) <- (1x21xi64, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(data_1, full_int_array_2)
        del data_1

        # pd_op.unsqueeze: (1x1x1x21xi64) <- (1x1x21xi64, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(unsqueeze_0, full_int_array_2)
        del unsqueeze_0

        # pd_op.cast: (1x1x1x21xf32) <- (1x1x1x21xi64)
        cast_0 = paddle._C_ops.cast(unsqueeze_1, paddle.float32)
        del unsqueeze_1

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x1x1x21xf32) <- (1x1x1x21xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(cast_0, full_1, float("1"), True)
        del cast_0, full_1

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("-10000"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x1x1x21xf32) <- (1x1x1x21xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(scale_0, full_2, float("0"), True)
        del full_2, scale_0

        # pd_op.transpose: (1x768x21xf32) <- (1x21x768xf32)
        transpose_0 = paddle._C_ops.transpose(dropout_0, [0, 2, 1])
        del dropout_0

        # pd_op.assign: (768x192x1xf32) <- (768x192x1xf32)
        assign_0 = parameter_194
        del parameter_194

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [-2]

        # pd_op.unsqueeze: (768x192x1x1xf32) <- (768x192x1xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(assign_0, full_int_array_3)
        del assign_0

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(transpose_0, full_int_array_3)

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x192x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            unsqueeze_3, unsqueeze_2, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_2, unsqueeze_3

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_4 = [1, 768, 1, 1]

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_193, full_int_array_4)
        del parameter_193

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_2 = paddle._C_ops.add(conv2d_0, reshape_0)
        del conv2d_0, reshape_0

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(add_2, full_int_array_3)
        del add_2

        # pd_op.assign: (768x192x1xf32) <- (768x192x1xf32)
        assign_1 = parameter_192
        del parameter_192

        # pd_op.unsqueeze: (768x192x1x1xf32) <- (768x192x1xf32, 1xi64)
        unsqueeze_4 = paddle._C_ops.unsqueeze(assign_1, full_int_array_3)
        del assign_1

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_5 = paddle._C_ops.unsqueeze(transpose_0, full_int_array_3)

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x192x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            unsqueeze_5, unsqueeze_4, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_4, unsqueeze_5

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_191, full_int_array_4)
        del parameter_191

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_3 = paddle._C_ops.add(conv2d_1, reshape_1)
        del conv2d_1, reshape_1

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_1 = paddle._C_ops.squeeze(add_3, full_int_array_3)
        del add_3

        # pd_op.assign: (768x192x1xf32) <- (768x192x1xf32)
        assign_2 = parameter_190
        del parameter_190

        # pd_op.unsqueeze: (768x192x1x1xf32) <- (768x192x1xf32, 1xi64)
        unsqueeze_6 = paddle._C_ops.unsqueeze(assign_2, full_int_array_3)
        del assign_2

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_7 = paddle._C_ops.unsqueeze(transpose_0, full_int_array_3)

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x192x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            unsqueeze_7, unsqueeze_6, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_6, unsqueeze_7

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(parameter_189, full_int_array_4)
        del parameter_189

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_4 = paddle._C_ops.add(conv2d_2, reshape_2)
        del conv2d_2, reshape_2

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_2 = paddle._C_ops.squeeze(add_4, full_int_array_3)
        del add_4

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_5 = [1, 12, 64, 21]

        # pd_op.reshape: (1x12x64x21xf32) <- (1x768x21xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(squeeze_0, full_int_array_5)
        del squeeze_0

        # pd_op.transpose: (1x12x21x64xf32) <- (1x12x64x21xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_3, [0, 1, 3, 2])
        del reshape_3

        # pd_op.reshape: (1x12x64x21xf32) <- (1x768x21xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(squeeze_1, full_int_array_5)
        del squeeze_1

        # pd_op.reshape: (1x12x64x21xf32) <- (1x768x21xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(squeeze_2, full_int_array_5)
        del squeeze_2

        # pd_op.transpose: (1x12x21x64xf32) <- (1x12x64x21xf32)
        transpose_2 = paddle._C_ops.transpose(reshape_5, [0, 1, 3, 2])
        del reshape_5

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x64x21xf32)
        matmul_0 = paddle._C_ops.matmul(transpose_1, reshape_4, False, False)
        del reshape_4, transpose_1

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x12x21x21xf32) <- (1x12x21x21xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(matmul_0, full_3, float("0"), True)
        del matmul_0

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_5 = paddle._C_ops.add(scale_2, scale_1)
        del scale_2

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_0 = paddle._C_ops.softmax(add_5, -1)
        del add_5

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_0, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_0

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_1 = paddle._C_ops.matmul(dropout_2, transpose_2, False, False)
        del dropout_2, transpose_2

        # pd_op.transpose: (1x12x64x21xf32) <- (1x12x21x64xf32)
        transpose_3 = paddle._C_ops.transpose(matmul_1, [0, 1, 3, 2])
        del matmul_1

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_6 = [1, 768, 21]

        # pd_op.reshape: (1x768x21xf32) <- (1x12x64x21xf32, 3xi64)
        reshape_6 = paddle._C_ops.reshape(transpose_3, full_int_array_6)
        del transpose_3

        # pd_op.assign: (768x768x1xf32) <- (768x768x1xf32)
        assign_3 = parameter_188
        del parameter_188

        # pd_op.unsqueeze: (768x768x1x1xf32) <- (768x768x1xf32, 1xi64)
        unsqueeze_8 = paddle._C_ops.unsqueeze(assign_3, full_int_array_3)
        del assign_3

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_9 = paddle._C_ops.unsqueeze(reshape_6, full_int_array_3)
        del reshape_6

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x768x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            unsqueeze_9, unsqueeze_8, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del unsqueeze_8, unsqueeze_9

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(parameter_187, full_int_array_4)
        del parameter_187

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_6 = paddle._C_ops.add(conv2d_3, reshape_7)
        del conv2d_3, reshape_7

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_3 = paddle._C_ops.squeeze(add_6, full_int_array_3)
        del add_6

        # pd_op.dropout: (1x768x21xf32, 1x768x21xui8) <- (1x768x21xf32, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                squeeze_3, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del squeeze_3

        # pd_op.add: (1x768x21xf32) <- (1x768x21xf32, 1x768x21xf32)
        add_7 = paddle._C_ops.add(dropout_4, transpose_0)
        del dropout_4, transpose_0

        # pd_op.transpose: (1x21x768xf32) <- (1x768x21xf32)
        transpose_4 = paddle._C_ops.transpose(add_7, [0, 2, 1])
        del add_7

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_4, parameter_186, parameter_185, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_185, parameter_186, transpose_4

        # pd_op.transpose: (1x768x21xf32) <- (1x21x768xf32)
        transpose_5 = paddle._C_ops.transpose(layer_norm_3, [0, 2, 1])
        del layer_norm_3

        # pd_op.assign: (3072x192x1xf32) <- (3072x192x1xf32)
        assign_4 = parameter_184
        del parameter_184

        # pd_op.unsqueeze: (3072x192x1x1xf32) <- (3072x192x1xf32, 1xi64)
        unsqueeze_10 = paddle._C_ops.unsqueeze(assign_4, full_int_array_3)
        del assign_4

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_11 = paddle._C_ops.unsqueeze(transpose_5, full_int_array_3)

        # pd_op.conv2d: (1x3072x1x21xf32) <- (1x768x1x21xf32, 3072x192x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            unsqueeze_11, unsqueeze_10, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_10, unsqueeze_11

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_7 = [1, 3072, 1, 1]

        # pd_op.reshape: (1x3072x1x1xf32) <- (3072xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(parameter_183, full_int_array_7)
        del parameter_183

        # pd_op.add: (1x3072x1x21xf32) <- (1x3072x1x21xf32, 1x3072x1x1xf32)
        add_8 = paddle._C_ops.add(conv2d_4, reshape_8)
        del conv2d_4, reshape_8

        # pd_op.squeeze: (1x3072x21xf32) <- (1x3072x1x21xf32, 1xi64)
        squeeze_4 = paddle._C_ops.squeeze(add_8, full_int_array_3)
        del add_8

        # pd_op.gelu: (1x3072x21xf32) <- (1x3072x21xf32)
        gelu_0 = paddle._C_ops.gelu(squeeze_4, False)
        del squeeze_4

        # pd_op.assign: (768x768x1xf32) <- (768x768x1xf32)
        assign_5 = parameter_182
        del parameter_182

        # pd_op.unsqueeze: (768x768x1x1xf32) <- (768x768x1xf32, 1xi64)
        unsqueeze_12 = paddle._C_ops.unsqueeze(assign_5, full_int_array_3)
        del assign_5

        # pd_op.unsqueeze: (1x3072x1x21xf32) <- (1x3072x21xf32, 1xi64)
        unsqueeze_13 = paddle._C_ops.unsqueeze(gelu_0, full_int_array_3)
        del gelu_0

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x3072x1x21xf32, 768x768x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            unsqueeze_13, unsqueeze_12, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_12, unsqueeze_13

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(parameter_181, full_int_array_4)
        del parameter_181

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_9 = paddle._C_ops.add(conv2d_5, reshape_9)
        del conv2d_5, reshape_9

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_5 = paddle._C_ops.squeeze(add_9, full_int_array_3)
        del add_9

        # pd_op.dropout: (1x768x21xf32, 1x768x21xui8) <- (1x768x21xf32, None, 1xf32)
        dropout_6, dropout_7 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                squeeze_5, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del squeeze_5

        # pd_op.add: (1x768x21xf32) <- (1x768x21xf32, 1x768x21xf32)
        add_10 = paddle._C_ops.add(dropout_6, transpose_5)
        del dropout_6, transpose_5

        # pd_op.transpose: (1x21x768xf32) <- (1x768x21xf32)
        transpose_6 = paddle._C_ops.transpose(add_10, [0, 2, 1])
        del add_10

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_6, parameter_180, parameter_179, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_179, parameter_180, transpose_6

        # pd_op.transpose: (1x768x21xf32) <- (1x21x768xf32)
        transpose_7 = paddle._C_ops.transpose(layer_norm_6, [0, 2, 1])
        del layer_norm_6

        # pd_op.assign: (768x192x1xf32) <- (768x192x1xf32)
        assign_6 = parameter_178
        del parameter_178

        # pd_op.unsqueeze: (768x192x1x1xf32) <- (768x192x1xf32, 1xi64)
        unsqueeze_14 = paddle._C_ops.unsqueeze(assign_6, full_int_array_3)
        del assign_6

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_15 = paddle._C_ops.unsqueeze(transpose_7, full_int_array_3)

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x192x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            unsqueeze_15, unsqueeze_14, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_14, unsqueeze_15

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(parameter_177, full_int_array_4)
        del parameter_177

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_11 = paddle._C_ops.add(conv2d_6, reshape_10)
        del conv2d_6, reshape_10

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_6 = paddle._C_ops.squeeze(add_11, full_int_array_3)
        del add_11

        # pd_op.assign: (768x192x1xf32) <- (768x192x1xf32)
        assign_7 = parameter_176
        del parameter_176

        # pd_op.unsqueeze: (768x192x1x1xf32) <- (768x192x1xf32, 1xi64)
        unsqueeze_16 = paddle._C_ops.unsqueeze(assign_7, full_int_array_3)
        del assign_7

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_17 = paddle._C_ops.unsqueeze(transpose_7, full_int_array_3)

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x192x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            unsqueeze_17, unsqueeze_16, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_16, unsqueeze_17

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_11 = paddle._C_ops.reshape(parameter_175, full_int_array_4)
        del parameter_175

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_12 = paddle._C_ops.add(conv2d_7, reshape_11)
        del conv2d_7, reshape_11

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_7 = paddle._C_ops.squeeze(add_12, full_int_array_3)
        del add_12

        # pd_op.assign: (768x192x1xf32) <- (768x192x1xf32)
        assign_8 = parameter_174
        del parameter_174

        # pd_op.unsqueeze: (768x192x1x1xf32) <- (768x192x1xf32, 1xi64)
        unsqueeze_18 = paddle._C_ops.unsqueeze(assign_8, full_int_array_3)
        del assign_8

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_19 = paddle._C_ops.unsqueeze(transpose_7, full_int_array_3)

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x192x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            unsqueeze_19, unsqueeze_18, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_18, unsqueeze_19

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(parameter_173, full_int_array_4)
        del parameter_173

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_13 = paddle._C_ops.add(conv2d_8, reshape_12)
        del conv2d_8, reshape_12

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_8 = paddle._C_ops.squeeze(add_13, full_int_array_3)
        del add_13

        # pd_op.reshape: (1x12x64x21xf32) <- (1x768x21xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(squeeze_6, full_int_array_5)
        del squeeze_6

        # pd_op.transpose: (1x12x21x64xf32) <- (1x12x64x21xf32)
        transpose_8 = paddle._C_ops.transpose(reshape_13, [0, 1, 3, 2])
        del reshape_13

        # pd_op.reshape: (1x12x64x21xf32) <- (1x768x21xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(squeeze_7, full_int_array_5)
        del squeeze_7

        # pd_op.reshape: (1x12x64x21xf32) <- (1x768x21xf32, 4xi64)
        reshape_15 = paddle._C_ops.reshape(squeeze_8, full_int_array_5)
        del squeeze_8

        # pd_op.transpose: (1x12x21x64xf32) <- (1x12x64x21xf32)
        transpose_9 = paddle._C_ops.transpose(reshape_15, [0, 1, 3, 2])
        del reshape_15

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x64x21xf32)
        matmul_2 = paddle._C_ops.matmul(transpose_8, reshape_14, False, False)
        del reshape_14, transpose_8

        # pd_op.scale: (1x12x21x21xf32) <- (1x12x21x21xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(matmul_2, full_3, float("0"), True)
        del matmul_2

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_14 = paddle._C_ops.add(scale_3, scale_1)
        del scale_3

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_1 = paddle._C_ops.softmax(add_14, -1)
        del add_14

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_1, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_1

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_3 = paddle._C_ops.matmul(dropout_8, transpose_9, False, False)
        del dropout_8, transpose_9

        # pd_op.transpose: (1x12x64x21xf32) <- (1x12x21x64xf32)
        transpose_10 = paddle._C_ops.transpose(matmul_3, [0, 1, 3, 2])
        del matmul_3

        # pd_op.reshape: (1x768x21xf32) <- (1x12x64x21xf32, 3xi64)
        reshape_16 = paddle._C_ops.reshape(transpose_10, full_int_array_6)
        del transpose_10

        # pd_op.assign: (768x768x1xf32) <- (768x768x1xf32)
        assign_9 = parameter_172
        del parameter_172

        # pd_op.unsqueeze: (768x768x1x1xf32) <- (768x768x1xf32, 1xi64)
        unsqueeze_20 = paddle._C_ops.unsqueeze(assign_9, full_int_array_3)
        del assign_9

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_21 = paddle._C_ops.unsqueeze(reshape_16, full_int_array_3)
        del reshape_16

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x768x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            unsqueeze_21, unsqueeze_20, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del unsqueeze_20, unsqueeze_21

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_17 = paddle._C_ops.reshape(parameter_171, full_int_array_4)
        del parameter_171

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_15 = paddle._C_ops.add(conv2d_9, reshape_17)
        del conv2d_9, reshape_17

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_9 = paddle._C_ops.squeeze(add_15, full_int_array_3)
        del add_15

        # pd_op.dropout: (1x768x21xf32, 1x768x21xui8) <- (1x768x21xf32, None, 1xf32)
        dropout_10, dropout_11 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                squeeze_9, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del squeeze_9

        # pd_op.add: (1x768x21xf32) <- (1x768x21xf32, 1x768x21xf32)
        add_16 = paddle._C_ops.add(dropout_10, transpose_7)
        del dropout_10, transpose_7

        # pd_op.transpose: (1x21x768xf32) <- (1x768x21xf32)
        transpose_11 = paddle._C_ops.transpose(add_16, [0, 2, 1])
        del add_16

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_11, parameter_170, parameter_169, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_169, parameter_170, transpose_11

        # pd_op.transpose: (1x768x21xf32) <- (1x21x768xf32)
        transpose_12 = paddle._C_ops.transpose(layer_norm_9, [0, 2, 1])
        del layer_norm_9

        # pd_op.assign: (3072x192x1xf32) <- (3072x192x1xf32)
        assign_10 = parameter_168
        del parameter_168

        # pd_op.unsqueeze: (3072x192x1x1xf32) <- (3072x192x1xf32, 1xi64)
        unsqueeze_22 = paddle._C_ops.unsqueeze(assign_10, full_int_array_3)
        del assign_10

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_23 = paddle._C_ops.unsqueeze(transpose_12, full_int_array_3)

        # pd_op.conv2d: (1x3072x1x21xf32) <- (1x768x1x21xf32, 3072x192x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            unsqueeze_23, unsqueeze_22, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_22, unsqueeze_23

        # pd_op.reshape: (1x3072x1x1xf32) <- (3072xf32, 4xi64)
        reshape_18 = paddle._C_ops.reshape(parameter_167, full_int_array_7)
        del parameter_167

        # pd_op.add: (1x3072x1x21xf32) <- (1x3072x1x21xf32, 1x3072x1x1xf32)
        add_17 = paddle._C_ops.add(conv2d_10, reshape_18)
        del conv2d_10, reshape_18

        # pd_op.squeeze: (1x3072x21xf32) <- (1x3072x1x21xf32, 1xi64)
        squeeze_10 = paddle._C_ops.squeeze(add_17, full_int_array_3)
        del add_17

        # pd_op.gelu: (1x3072x21xf32) <- (1x3072x21xf32)
        gelu_1 = paddle._C_ops.gelu(squeeze_10, False)
        del squeeze_10

        # pd_op.assign: (768x768x1xf32) <- (768x768x1xf32)
        assign_11 = parameter_166
        del parameter_166

        # pd_op.unsqueeze: (768x768x1x1xf32) <- (768x768x1xf32, 1xi64)
        unsqueeze_24 = paddle._C_ops.unsqueeze(assign_11, full_int_array_3)
        del assign_11

        # pd_op.unsqueeze: (1x3072x1x21xf32) <- (1x3072x21xf32, 1xi64)
        unsqueeze_25 = paddle._C_ops.unsqueeze(gelu_1, full_int_array_3)
        del gelu_1

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x3072x1x21xf32, 768x768x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            unsqueeze_25, unsqueeze_24, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_24, unsqueeze_25

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_19 = paddle._C_ops.reshape(parameter_165, full_int_array_4)
        del parameter_165

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_18 = paddle._C_ops.add(conv2d_11, reshape_19)
        del conv2d_11, reshape_19

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_11 = paddle._C_ops.squeeze(add_18, full_int_array_3)
        del add_18

        # pd_op.dropout: (1x768x21xf32, 1x768x21xui8) <- (1x768x21xf32, None, 1xf32)
        dropout_12, dropout_13 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                squeeze_11, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del squeeze_11

        # pd_op.add: (1x768x21xf32) <- (1x768x21xf32, 1x768x21xf32)
        add_19 = paddle._C_ops.add(dropout_12, transpose_12)
        del dropout_12, transpose_12

        # pd_op.transpose: (1x21x768xf32) <- (1x768x21xf32)
        transpose_13 = paddle._C_ops.transpose(add_19, [0, 2, 1])
        del add_19

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_13, parameter_164, parameter_163, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_163, parameter_164, transpose_13

        # pd_op.transpose: (1x768x21xf32) <- (1x21x768xf32)
        transpose_14 = paddle._C_ops.transpose(layer_norm_12, [0, 2, 1])
        del layer_norm_12

        # pd_op.assign: (768x192x1xf32) <- (768x192x1xf32)
        assign_12 = parameter_162
        del parameter_162

        # pd_op.unsqueeze: (768x192x1x1xf32) <- (768x192x1xf32, 1xi64)
        unsqueeze_26 = paddle._C_ops.unsqueeze(assign_12, full_int_array_3)
        del assign_12

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_27 = paddle._C_ops.unsqueeze(transpose_14, full_int_array_3)

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x192x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            unsqueeze_27, unsqueeze_26, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_26, unsqueeze_27

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_20 = paddle._C_ops.reshape(parameter_161, full_int_array_4)
        del parameter_161

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_20 = paddle._C_ops.add(conv2d_12, reshape_20)
        del conv2d_12, reshape_20

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_12 = paddle._C_ops.squeeze(add_20, full_int_array_3)
        del add_20

        # pd_op.assign: (768x192x1xf32) <- (768x192x1xf32)
        assign_13 = parameter_160
        del parameter_160

        # pd_op.unsqueeze: (768x192x1x1xf32) <- (768x192x1xf32, 1xi64)
        unsqueeze_28 = paddle._C_ops.unsqueeze(assign_13, full_int_array_3)
        del assign_13

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_29 = paddle._C_ops.unsqueeze(transpose_14, full_int_array_3)

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x192x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            unsqueeze_29, unsqueeze_28, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_28, unsqueeze_29

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_21 = paddle._C_ops.reshape(parameter_159, full_int_array_4)
        del parameter_159

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_21 = paddle._C_ops.add(conv2d_13, reshape_21)
        del conv2d_13, reshape_21

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_13 = paddle._C_ops.squeeze(add_21, full_int_array_3)
        del add_21

        # pd_op.assign: (768x192x1xf32) <- (768x192x1xf32)
        assign_14 = parameter_158
        del parameter_158

        # pd_op.unsqueeze: (768x192x1x1xf32) <- (768x192x1xf32, 1xi64)
        unsqueeze_30 = paddle._C_ops.unsqueeze(assign_14, full_int_array_3)
        del assign_14

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_31 = paddle._C_ops.unsqueeze(transpose_14, full_int_array_3)

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x192x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            unsqueeze_31, unsqueeze_30, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_30, unsqueeze_31

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_22 = paddle._C_ops.reshape(parameter_157, full_int_array_4)
        del parameter_157

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_22 = paddle._C_ops.add(conv2d_14, reshape_22)
        del conv2d_14, reshape_22

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_14 = paddle._C_ops.squeeze(add_22, full_int_array_3)
        del add_22

        # pd_op.reshape: (1x12x64x21xf32) <- (1x768x21xf32, 4xi64)
        reshape_23 = paddle._C_ops.reshape(squeeze_12, full_int_array_5)
        del squeeze_12

        # pd_op.transpose: (1x12x21x64xf32) <- (1x12x64x21xf32)
        transpose_15 = paddle._C_ops.transpose(reshape_23, [0, 1, 3, 2])
        del reshape_23

        # pd_op.reshape: (1x12x64x21xf32) <- (1x768x21xf32, 4xi64)
        reshape_24 = paddle._C_ops.reshape(squeeze_13, full_int_array_5)
        del squeeze_13

        # pd_op.reshape: (1x12x64x21xf32) <- (1x768x21xf32, 4xi64)
        reshape_25 = paddle._C_ops.reshape(squeeze_14, full_int_array_5)
        del squeeze_14

        # pd_op.transpose: (1x12x21x64xf32) <- (1x12x64x21xf32)
        transpose_16 = paddle._C_ops.transpose(reshape_25, [0, 1, 3, 2])
        del reshape_25

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x64x21xf32)
        matmul_4 = paddle._C_ops.matmul(transpose_15, reshape_24, False, False)
        del reshape_24, transpose_15

        # pd_op.scale: (1x12x21x21xf32) <- (1x12x21x21xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(matmul_4, full_3, float("0"), True)
        del matmul_4

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_23 = paddle._C_ops.add(scale_4, scale_1)
        del scale_4

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_2 = paddle._C_ops.softmax(add_23, -1)
        del add_23

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_14, dropout_15 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_2, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_2

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_5 = paddle._C_ops.matmul(dropout_14, transpose_16, False, False)
        del dropout_14, transpose_16

        # pd_op.transpose: (1x12x64x21xf32) <- (1x12x21x64xf32)
        transpose_17 = paddle._C_ops.transpose(matmul_5, [0, 1, 3, 2])
        del matmul_5

        # pd_op.reshape: (1x768x21xf32) <- (1x12x64x21xf32, 3xi64)
        reshape_26 = paddle._C_ops.reshape(transpose_17, full_int_array_6)
        del transpose_17

        # pd_op.assign: (768x768x1xf32) <- (768x768x1xf32)
        assign_15 = parameter_156
        del parameter_156

        # pd_op.unsqueeze: (768x768x1x1xf32) <- (768x768x1xf32, 1xi64)
        unsqueeze_32 = paddle._C_ops.unsqueeze(assign_15, full_int_array_3)
        del assign_15

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_33 = paddle._C_ops.unsqueeze(reshape_26, full_int_array_3)
        del reshape_26

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x768x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            unsqueeze_33, unsqueeze_32, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del unsqueeze_32, unsqueeze_33

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_27 = paddle._C_ops.reshape(parameter_155, full_int_array_4)
        del parameter_155

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_24 = paddle._C_ops.add(conv2d_15, reshape_27)
        del conv2d_15, reshape_27

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_15 = paddle._C_ops.squeeze(add_24, full_int_array_3)
        del add_24

        # pd_op.dropout: (1x768x21xf32, 1x768x21xui8) <- (1x768x21xf32, None, 1xf32)
        dropout_16, dropout_17 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                squeeze_15, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del squeeze_15

        # pd_op.add: (1x768x21xf32) <- (1x768x21xf32, 1x768x21xf32)
        add_25 = paddle._C_ops.add(dropout_16, transpose_14)
        del dropout_16, transpose_14

        # pd_op.transpose: (1x21x768xf32) <- (1x768x21xf32)
        transpose_18 = paddle._C_ops.transpose(add_25, [0, 2, 1])
        del add_25

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_18, parameter_154, parameter_153, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_153, parameter_154, transpose_18

        # pd_op.transpose: (1x768x21xf32) <- (1x21x768xf32)
        transpose_19 = paddle._C_ops.transpose(layer_norm_15, [0, 2, 1])
        del layer_norm_15

        # pd_op.assign: (3072x192x1xf32) <- (3072x192x1xf32)
        assign_16 = parameter_152
        del parameter_152

        # pd_op.unsqueeze: (3072x192x1x1xf32) <- (3072x192x1xf32, 1xi64)
        unsqueeze_34 = paddle._C_ops.unsqueeze(assign_16, full_int_array_3)
        del assign_16

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_35 = paddle._C_ops.unsqueeze(transpose_19, full_int_array_3)

        # pd_op.conv2d: (1x3072x1x21xf32) <- (1x768x1x21xf32, 3072x192x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            unsqueeze_35, unsqueeze_34, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_34, unsqueeze_35

        # pd_op.reshape: (1x3072x1x1xf32) <- (3072xf32, 4xi64)
        reshape_28 = paddle._C_ops.reshape(parameter_151, full_int_array_7)
        del parameter_151

        # pd_op.add: (1x3072x1x21xf32) <- (1x3072x1x21xf32, 1x3072x1x1xf32)
        add_26 = paddle._C_ops.add(conv2d_16, reshape_28)
        del conv2d_16, reshape_28

        # pd_op.squeeze: (1x3072x21xf32) <- (1x3072x1x21xf32, 1xi64)
        squeeze_16 = paddle._C_ops.squeeze(add_26, full_int_array_3)
        del add_26

        # pd_op.gelu: (1x3072x21xf32) <- (1x3072x21xf32)
        gelu_2 = paddle._C_ops.gelu(squeeze_16, False)
        del squeeze_16

        # pd_op.assign: (768x768x1xf32) <- (768x768x1xf32)
        assign_17 = parameter_150
        del parameter_150

        # pd_op.unsqueeze: (768x768x1x1xf32) <- (768x768x1xf32, 1xi64)
        unsqueeze_36 = paddle._C_ops.unsqueeze(assign_17, full_int_array_3)
        del assign_17

        # pd_op.unsqueeze: (1x3072x1x21xf32) <- (1x3072x21xf32, 1xi64)
        unsqueeze_37 = paddle._C_ops.unsqueeze(gelu_2, full_int_array_3)
        del gelu_2

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x3072x1x21xf32, 768x768x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            unsqueeze_37, unsqueeze_36, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_36, unsqueeze_37

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_29 = paddle._C_ops.reshape(parameter_149, full_int_array_4)
        del parameter_149

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_27 = paddle._C_ops.add(conv2d_17, reshape_29)
        del conv2d_17, reshape_29

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_17 = paddle._C_ops.squeeze(add_27, full_int_array_3)
        del add_27

        # pd_op.dropout: (1x768x21xf32, 1x768x21xui8) <- (1x768x21xf32, None, 1xf32)
        dropout_18, dropout_19 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                squeeze_17, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del squeeze_17

        # pd_op.add: (1x768x21xf32) <- (1x768x21xf32, 1x768x21xf32)
        add_28 = paddle._C_ops.add(dropout_18, transpose_19)
        del dropout_18, transpose_19

        # pd_op.transpose: (1x21x768xf32) <- (1x768x21xf32)
        transpose_20 = paddle._C_ops.transpose(add_28, [0, 2, 1])
        del add_28

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_20, parameter_148, parameter_147, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_147, parameter_148, transpose_20

        # pd_op.transpose: (1x768x21xf32) <- (1x21x768xf32)
        transpose_21 = paddle._C_ops.transpose(layer_norm_18, [0, 2, 1])
        del layer_norm_18

        # pd_op.assign: (768x192x1xf32) <- (768x192x1xf32)
        assign_18 = parameter_146
        del parameter_146

        # pd_op.unsqueeze: (768x192x1x1xf32) <- (768x192x1xf32, 1xi64)
        unsqueeze_38 = paddle._C_ops.unsqueeze(assign_18, full_int_array_3)
        del assign_18

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_39 = paddle._C_ops.unsqueeze(transpose_21, full_int_array_3)

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x192x1x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            unsqueeze_39, unsqueeze_38, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_38, unsqueeze_39

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_30 = paddle._C_ops.reshape(parameter_145, full_int_array_4)
        del parameter_145

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_29 = paddle._C_ops.add(conv2d_18, reshape_30)
        del conv2d_18, reshape_30

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_18 = paddle._C_ops.squeeze(add_29, full_int_array_3)
        del add_29

        # pd_op.assign: (768x192x1xf32) <- (768x192x1xf32)
        assign_19 = parameter_144
        del parameter_144

        # pd_op.unsqueeze: (768x192x1x1xf32) <- (768x192x1xf32, 1xi64)
        unsqueeze_40 = paddle._C_ops.unsqueeze(assign_19, full_int_array_3)
        del assign_19

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_41 = paddle._C_ops.unsqueeze(transpose_21, full_int_array_3)

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x192x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            unsqueeze_41, unsqueeze_40, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_40, unsqueeze_41

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_31 = paddle._C_ops.reshape(parameter_143, full_int_array_4)
        del parameter_143

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_30 = paddle._C_ops.add(conv2d_19, reshape_31)
        del conv2d_19, reshape_31

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_19 = paddle._C_ops.squeeze(add_30, full_int_array_3)
        del add_30

        # pd_op.assign: (768x192x1xf32) <- (768x192x1xf32)
        assign_20 = parameter_142
        del parameter_142

        # pd_op.unsqueeze: (768x192x1x1xf32) <- (768x192x1xf32, 1xi64)
        unsqueeze_42 = paddle._C_ops.unsqueeze(assign_20, full_int_array_3)
        del assign_20

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_43 = paddle._C_ops.unsqueeze(transpose_21, full_int_array_3)

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x192x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            unsqueeze_43, unsqueeze_42, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_42, unsqueeze_43

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_32 = paddle._C_ops.reshape(parameter_141, full_int_array_4)
        del parameter_141

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_31 = paddle._C_ops.add(conv2d_20, reshape_32)
        del conv2d_20, reshape_32

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_20 = paddle._C_ops.squeeze(add_31, full_int_array_3)
        del add_31

        # pd_op.reshape: (1x12x64x21xf32) <- (1x768x21xf32, 4xi64)
        reshape_33 = paddle._C_ops.reshape(squeeze_18, full_int_array_5)
        del squeeze_18

        # pd_op.transpose: (1x12x21x64xf32) <- (1x12x64x21xf32)
        transpose_22 = paddle._C_ops.transpose(reshape_33, [0, 1, 3, 2])
        del reshape_33

        # pd_op.reshape: (1x12x64x21xf32) <- (1x768x21xf32, 4xi64)
        reshape_34 = paddle._C_ops.reshape(squeeze_19, full_int_array_5)
        del squeeze_19

        # pd_op.reshape: (1x12x64x21xf32) <- (1x768x21xf32, 4xi64)
        reshape_35 = paddle._C_ops.reshape(squeeze_20, full_int_array_5)
        del squeeze_20

        # pd_op.transpose: (1x12x21x64xf32) <- (1x12x64x21xf32)
        transpose_23 = paddle._C_ops.transpose(reshape_35, [0, 1, 3, 2])
        del reshape_35

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x64x21xf32)
        matmul_6 = paddle._C_ops.matmul(transpose_22, reshape_34, False, False)
        del reshape_34, transpose_22

        # pd_op.scale: (1x12x21x21xf32) <- (1x12x21x21xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(matmul_6, full_3, float("0"), True)
        del matmul_6

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_32 = paddle._C_ops.add(scale_5, scale_1)
        del scale_5

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_3 = paddle._C_ops.softmax(add_32, -1)
        del add_32

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_20, dropout_21 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_3, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_3

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_7 = paddle._C_ops.matmul(dropout_20, transpose_23, False, False)
        del dropout_20, transpose_23

        # pd_op.transpose: (1x12x64x21xf32) <- (1x12x21x64xf32)
        transpose_24 = paddle._C_ops.transpose(matmul_7, [0, 1, 3, 2])
        del matmul_7

        # pd_op.reshape: (1x768x21xf32) <- (1x12x64x21xf32, 3xi64)
        reshape_36 = paddle._C_ops.reshape(transpose_24, full_int_array_6)
        del transpose_24

        # pd_op.assign: (768x768x1xf32) <- (768x768x1xf32)
        assign_21 = parameter_140
        del parameter_140

        # pd_op.unsqueeze: (768x768x1x1xf32) <- (768x768x1xf32, 1xi64)
        unsqueeze_44 = paddle._C_ops.unsqueeze(assign_21, full_int_array_3)
        del assign_21

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_45 = paddle._C_ops.unsqueeze(reshape_36, full_int_array_3)
        del reshape_36

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x768x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            unsqueeze_45, unsqueeze_44, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del unsqueeze_44, unsqueeze_45

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_37 = paddle._C_ops.reshape(parameter_139, full_int_array_4)
        del parameter_139

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_33 = paddle._C_ops.add(conv2d_21, reshape_37)
        del conv2d_21, reshape_37

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_21 = paddle._C_ops.squeeze(add_33, full_int_array_3)
        del add_33

        # pd_op.dropout: (1x768x21xf32, 1x768x21xui8) <- (1x768x21xf32, None, 1xf32)
        dropout_22, dropout_23 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                squeeze_21, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del squeeze_21

        # pd_op.add: (1x768x21xf32) <- (1x768x21xf32, 1x768x21xf32)
        add_34 = paddle._C_ops.add(dropout_22, transpose_21)
        del dropout_22, transpose_21

        # pd_op.transpose: (1x21x768xf32) <- (1x768x21xf32)
        transpose_25 = paddle._C_ops.transpose(add_34, [0, 2, 1])
        del add_34

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_25, parameter_138, parameter_137, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_137, parameter_138, transpose_25

        # pd_op.transpose: (1x768x21xf32) <- (1x21x768xf32)
        transpose_26 = paddle._C_ops.transpose(layer_norm_21, [0, 2, 1])
        del layer_norm_21

        # pd_op.assign: (3072x192x1xf32) <- (3072x192x1xf32)
        assign_22 = parameter_136
        del parameter_136

        # pd_op.unsqueeze: (3072x192x1x1xf32) <- (3072x192x1xf32, 1xi64)
        unsqueeze_46 = paddle._C_ops.unsqueeze(assign_22, full_int_array_3)
        del assign_22

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_47 = paddle._C_ops.unsqueeze(transpose_26, full_int_array_3)

        # pd_op.conv2d: (1x3072x1x21xf32) <- (1x768x1x21xf32, 3072x192x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(
            unsqueeze_47, unsqueeze_46, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_46, unsqueeze_47

        # pd_op.reshape: (1x3072x1x1xf32) <- (3072xf32, 4xi64)
        reshape_38 = paddle._C_ops.reshape(parameter_135, full_int_array_7)
        del parameter_135

        # pd_op.add: (1x3072x1x21xf32) <- (1x3072x1x21xf32, 1x3072x1x1xf32)
        add_35 = paddle._C_ops.add(conv2d_22, reshape_38)
        del conv2d_22, reshape_38

        # pd_op.squeeze: (1x3072x21xf32) <- (1x3072x1x21xf32, 1xi64)
        squeeze_22 = paddle._C_ops.squeeze(add_35, full_int_array_3)
        del add_35

        # pd_op.gelu: (1x3072x21xf32) <- (1x3072x21xf32)
        gelu_3 = paddle._C_ops.gelu(squeeze_22, False)
        del squeeze_22

        # pd_op.assign: (768x768x1xf32) <- (768x768x1xf32)
        assign_23 = parameter_134
        del parameter_134

        # pd_op.unsqueeze: (768x768x1x1xf32) <- (768x768x1xf32, 1xi64)
        unsqueeze_48 = paddle._C_ops.unsqueeze(assign_23, full_int_array_3)
        del assign_23

        # pd_op.unsqueeze: (1x3072x1x21xf32) <- (1x3072x21xf32, 1xi64)
        unsqueeze_49 = paddle._C_ops.unsqueeze(gelu_3, full_int_array_3)
        del gelu_3

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x3072x1x21xf32, 768x768x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(
            unsqueeze_49, unsqueeze_48, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_48, unsqueeze_49

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_39 = paddle._C_ops.reshape(parameter_133, full_int_array_4)
        del parameter_133

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_36 = paddle._C_ops.add(conv2d_23, reshape_39)
        del conv2d_23, reshape_39

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_23 = paddle._C_ops.squeeze(add_36, full_int_array_3)
        del add_36

        # pd_op.dropout: (1x768x21xf32, 1x768x21xui8) <- (1x768x21xf32, None, 1xf32)
        dropout_24, dropout_25 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                squeeze_23, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del squeeze_23

        # pd_op.add: (1x768x21xf32) <- (1x768x21xf32, 1x768x21xf32)
        add_37 = paddle._C_ops.add(dropout_24, transpose_26)
        del dropout_24, transpose_26

        # pd_op.transpose: (1x21x768xf32) <- (1x768x21xf32)
        transpose_27 = paddle._C_ops.transpose(add_37, [0, 2, 1])
        del add_37

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_27, parameter_132, parameter_131, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_131, parameter_132, transpose_27

        # pd_op.transpose: (1x768x21xf32) <- (1x21x768xf32)
        transpose_28 = paddle._C_ops.transpose(layer_norm_24, [0, 2, 1])
        del layer_norm_24

        # pd_op.assign: (768x192x1xf32) <- (768x192x1xf32)
        assign_24 = parameter_130
        del parameter_130

        # pd_op.unsqueeze: (768x192x1x1xf32) <- (768x192x1xf32, 1xi64)
        unsqueeze_50 = paddle._C_ops.unsqueeze(assign_24, full_int_array_3)
        del assign_24

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_51 = paddle._C_ops.unsqueeze(transpose_28, full_int_array_3)

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x192x1x1xf32)
        conv2d_24 = paddle._C_ops.conv2d(
            unsqueeze_51, unsqueeze_50, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_50, unsqueeze_51

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_40 = paddle._C_ops.reshape(parameter_129, full_int_array_4)
        del parameter_129

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_38 = paddle._C_ops.add(conv2d_24, reshape_40)
        del conv2d_24, reshape_40

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_24 = paddle._C_ops.squeeze(add_38, full_int_array_3)
        del add_38

        # pd_op.assign: (768x192x1xf32) <- (768x192x1xf32)
        assign_25 = parameter_128
        del parameter_128

        # pd_op.unsqueeze: (768x192x1x1xf32) <- (768x192x1xf32, 1xi64)
        unsqueeze_52 = paddle._C_ops.unsqueeze(assign_25, full_int_array_3)
        del assign_25

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_53 = paddle._C_ops.unsqueeze(transpose_28, full_int_array_3)

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x192x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(
            unsqueeze_53, unsqueeze_52, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_52, unsqueeze_53

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_41 = paddle._C_ops.reshape(parameter_127, full_int_array_4)
        del parameter_127

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_39 = paddle._C_ops.add(conv2d_25, reshape_41)
        del conv2d_25, reshape_41

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_25 = paddle._C_ops.squeeze(add_39, full_int_array_3)
        del add_39

        # pd_op.assign: (768x192x1xf32) <- (768x192x1xf32)
        assign_26 = parameter_126
        del parameter_126

        # pd_op.unsqueeze: (768x192x1x1xf32) <- (768x192x1xf32, 1xi64)
        unsqueeze_54 = paddle._C_ops.unsqueeze(assign_26, full_int_array_3)
        del assign_26

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_55 = paddle._C_ops.unsqueeze(transpose_28, full_int_array_3)

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x192x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(
            unsqueeze_55, unsqueeze_54, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_54, unsqueeze_55

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_42 = paddle._C_ops.reshape(parameter_125, full_int_array_4)
        del parameter_125

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_40 = paddle._C_ops.add(conv2d_26, reshape_42)
        del conv2d_26, reshape_42

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_26 = paddle._C_ops.squeeze(add_40, full_int_array_3)
        del add_40

        # pd_op.reshape: (1x12x64x21xf32) <- (1x768x21xf32, 4xi64)
        reshape_43 = paddle._C_ops.reshape(squeeze_24, full_int_array_5)
        del squeeze_24

        # pd_op.transpose: (1x12x21x64xf32) <- (1x12x64x21xf32)
        transpose_29 = paddle._C_ops.transpose(reshape_43, [0, 1, 3, 2])
        del reshape_43

        # pd_op.reshape: (1x12x64x21xf32) <- (1x768x21xf32, 4xi64)
        reshape_44 = paddle._C_ops.reshape(squeeze_25, full_int_array_5)
        del squeeze_25

        # pd_op.reshape: (1x12x64x21xf32) <- (1x768x21xf32, 4xi64)
        reshape_45 = paddle._C_ops.reshape(squeeze_26, full_int_array_5)
        del squeeze_26

        # pd_op.transpose: (1x12x21x64xf32) <- (1x12x64x21xf32)
        transpose_30 = paddle._C_ops.transpose(reshape_45, [0, 1, 3, 2])
        del reshape_45

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x64x21xf32)
        matmul_8 = paddle._C_ops.matmul(transpose_29, reshape_44, False, False)
        del reshape_44, transpose_29

        # pd_op.scale: (1x12x21x21xf32) <- (1x12x21x21xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(matmul_8, full_3, float("0"), True)
        del matmul_8

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_41 = paddle._C_ops.add(scale_6, scale_1)
        del scale_6

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_4 = paddle._C_ops.softmax(add_41, -1)
        del add_41

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_26, dropout_27 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_4, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_4

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_9 = paddle._C_ops.matmul(dropout_26, transpose_30, False, False)
        del dropout_26, transpose_30

        # pd_op.transpose: (1x12x64x21xf32) <- (1x12x21x64xf32)
        transpose_31 = paddle._C_ops.transpose(matmul_9, [0, 1, 3, 2])
        del matmul_9

        # pd_op.reshape: (1x768x21xf32) <- (1x12x64x21xf32, 3xi64)
        reshape_46 = paddle._C_ops.reshape(transpose_31, full_int_array_6)
        del transpose_31

        # pd_op.assign: (768x768x1xf32) <- (768x768x1xf32)
        assign_27 = parameter_124
        del parameter_124

        # pd_op.unsqueeze: (768x768x1x1xf32) <- (768x768x1xf32, 1xi64)
        unsqueeze_56 = paddle._C_ops.unsqueeze(assign_27, full_int_array_3)
        del assign_27

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_57 = paddle._C_ops.unsqueeze(reshape_46, full_int_array_3)
        del reshape_46

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x768x1x1xf32)
        conv2d_27 = paddle._C_ops.conv2d(
            unsqueeze_57, unsqueeze_56, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del unsqueeze_56, unsqueeze_57

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_47 = paddle._C_ops.reshape(parameter_123, full_int_array_4)
        del parameter_123

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_42 = paddle._C_ops.add(conv2d_27, reshape_47)
        del conv2d_27, reshape_47

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_27 = paddle._C_ops.squeeze(add_42, full_int_array_3)
        del add_42

        # pd_op.dropout: (1x768x21xf32, 1x768x21xui8) <- (1x768x21xf32, None, 1xf32)
        dropout_28, dropout_29 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                squeeze_27, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del squeeze_27

        # pd_op.add: (1x768x21xf32) <- (1x768x21xf32, 1x768x21xf32)
        add_43 = paddle._C_ops.add(dropout_28, transpose_28)
        del dropout_28, transpose_28

        # pd_op.transpose: (1x21x768xf32) <- (1x768x21xf32)
        transpose_32 = paddle._C_ops.transpose(add_43, [0, 2, 1])
        del add_43

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_32, parameter_122, parameter_121, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_121, parameter_122, transpose_32

        # pd_op.transpose: (1x768x21xf32) <- (1x21x768xf32)
        transpose_33 = paddle._C_ops.transpose(layer_norm_27, [0, 2, 1])
        del layer_norm_27

        # pd_op.assign: (3072x192x1xf32) <- (3072x192x1xf32)
        assign_28 = parameter_120
        del parameter_120

        # pd_op.unsqueeze: (3072x192x1x1xf32) <- (3072x192x1xf32, 1xi64)
        unsqueeze_58 = paddle._C_ops.unsqueeze(assign_28, full_int_array_3)
        del assign_28

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_59 = paddle._C_ops.unsqueeze(transpose_33, full_int_array_3)

        # pd_op.conv2d: (1x3072x1x21xf32) <- (1x768x1x21xf32, 3072x192x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(
            unsqueeze_59, unsqueeze_58, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_58, unsqueeze_59

        # pd_op.reshape: (1x3072x1x1xf32) <- (3072xf32, 4xi64)
        reshape_48 = paddle._C_ops.reshape(parameter_119, full_int_array_7)
        del parameter_119

        # pd_op.add: (1x3072x1x21xf32) <- (1x3072x1x21xf32, 1x3072x1x1xf32)
        add_44 = paddle._C_ops.add(conv2d_28, reshape_48)
        del conv2d_28, reshape_48

        # pd_op.squeeze: (1x3072x21xf32) <- (1x3072x1x21xf32, 1xi64)
        squeeze_28 = paddle._C_ops.squeeze(add_44, full_int_array_3)
        del add_44

        # pd_op.gelu: (1x3072x21xf32) <- (1x3072x21xf32)
        gelu_4 = paddle._C_ops.gelu(squeeze_28, False)
        del squeeze_28

        # pd_op.assign: (768x768x1xf32) <- (768x768x1xf32)
        assign_29 = parameter_118
        del parameter_118

        # pd_op.unsqueeze: (768x768x1x1xf32) <- (768x768x1xf32, 1xi64)
        unsqueeze_60 = paddle._C_ops.unsqueeze(assign_29, full_int_array_3)
        del assign_29

        # pd_op.unsqueeze: (1x3072x1x21xf32) <- (1x3072x21xf32, 1xi64)
        unsqueeze_61 = paddle._C_ops.unsqueeze(gelu_4, full_int_array_3)
        del gelu_4

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x3072x1x21xf32, 768x768x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(
            unsqueeze_61, unsqueeze_60, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_60, unsqueeze_61

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_49 = paddle._C_ops.reshape(parameter_117, full_int_array_4)
        del parameter_117

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_45 = paddle._C_ops.add(conv2d_29, reshape_49)
        del conv2d_29, reshape_49

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_29 = paddle._C_ops.squeeze(add_45, full_int_array_3)
        del add_45

        # pd_op.dropout: (1x768x21xf32, 1x768x21xui8) <- (1x768x21xf32, None, 1xf32)
        dropout_30, dropout_31 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                squeeze_29, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del squeeze_29

        # pd_op.add: (1x768x21xf32) <- (1x768x21xf32, 1x768x21xf32)
        add_46 = paddle._C_ops.add(dropout_30, transpose_33)
        del dropout_30, transpose_33

        # pd_op.transpose: (1x21x768xf32) <- (1x768x21xf32)
        transpose_34 = paddle._C_ops.transpose(add_46, [0, 2, 1])
        del add_46

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_34, parameter_116, parameter_115, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_115, parameter_116, transpose_34

        # pd_op.transpose: (1x768x21xf32) <- (1x21x768xf32)
        transpose_35 = paddle._C_ops.transpose(layer_norm_30, [0, 2, 1])
        del layer_norm_30

        # pd_op.assign: (768x192x1xf32) <- (768x192x1xf32)
        assign_30 = parameter_114
        del parameter_114

        # pd_op.unsqueeze: (768x192x1x1xf32) <- (768x192x1xf32, 1xi64)
        unsqueeze_62 = paddle._C_ops.unsqueeze(assign_30, full_int_array_3)
        del assign_30

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_63 = paddle._C_ops.unsqueeze(transpose_35, full_int_array_3)

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x192x1x1xf32)
        conv2d_30 = paddle._C_ops.conv2d(
            unsqueeze_63, unsqueeze_62, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_62, unsqueeze_63

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_50 = paddle._C_ops.reshape(parameter_113, full_int_array_4)
        del parameter_113

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_47 = paddle._C_ops.add(conv2d_30, reshape_50)
        del conv2d_30, reshape_50

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_30 = paddle._C_ops.squeeze(add_47, full_int_array_3)
        del add_47

        # pd_op.assign: (768x192x1xf32) <- (768x192x1xf32)
        assign_31 = parameter_112
        del parameter_112

        # pd_op.unsqueeze: (768x192x1x1xf32) <- (768x192x1xf32, 1xi64)
        unsqueeze_64 = paddle._C_ops.unsqueeze(assign_31, full_int_array_3)
        del assign_31

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_65 = paddle._C_ops.unsqueeze(transpose_35, full_int_array_3)

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x192x1x1xf32)
        conv2d_31 = paddle._C_ops.conv2d(
            unsqueeze_65, unsqueeze_64, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_64, unsqueeze_65

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_51 = paddle._C_ops.reshape(parameter_111, full_int_array_4)
        del parameter_111

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_48 = paddle._C_ops.add(conv2d_31, reshape_51)
        del conv2d_31, reshape_51

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_31 = paddle._C_ops.squeeze(add_48, full_int_array_3)
        del add_48

        # pd_op.assign: (768x192x1xf32) <- (768x192x1xf32)
        assign_32 = parameter_110
        del parameter_110

        # pd_op.unsqueeze: (768x192x1x1xf32) <- (768x192x1xf32, 1xi64)
        unsqueeze_66 = paddle._C_ops.unsqueeze(assign_32, full_int_array_3)
        del assign_32

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_67 = paddle._C_ops.unsqueeze(transpose_35, full_int_array_3)

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x192x1x1xf32)
        conv2d_32 = paddle._C_ops.conv2d(
            unsqueeze_67, unsqueeze_66, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_66, unsqueeze_67

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_52 = paddle._C_ops.reshape(parameter_109, full_int_array_4)
        del parameter_109

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_49 = paddle._C_ops.add(conv2d_32, reshape_52)
        del conv2d_32, reshape_52

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_32 = paddle._C_ops.squeeze(add_49, full_int_array_3)
        del add_49

        # pd_op.reshape: (1x12x64x21xf32) <- (1x768x21xf32, 4xi64)
        reshape_53 = paddle._C_ops.reshape(squeeze_30, full_int_array_5)
        del squeeze_30

        # pd_op.transpose: (1x12x21x64xf32) <- (1x12x64x21xf32)
        transpose_36 = paddle._C_ops.transpose(reshape_53, [0, 1, 3, 2])
        del reshape_53

        # pd_op.reshape: (1x12x64x21xf32) <- (1x768x21xf32, 4xi64)
        reshape_54 = paddle._C_ops.reshape(squeeze_31, full_int_array_5)
        del squeeze_31

        # pd_op.reshape: (1x12x64x21xf32) <- (1x768x21xf32, 4xi64)
        reshape_55 = paddle._C_ops.reshape(squeeze_32, full_int_array_5)
        del squeeze_32

        # pd_op.transpose: (1x12x21x64xf32) <- (1x12x64x21xf32)
        transpose_37 = paddle._C_ops.transpose(reshape_55, [0, 1, 3, 2])
        del reshape_55

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x64x21xf32)
        matmul_10 = paddle._C_ops.matmul(transpose_36, reshape_54, False, False)
        del reshape_54, transpose_36

        # pd_op.scale: (1x12x21x21xf32) <- (1x12x21x21xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(matmul_10, full_3, float("0"), True)
        del matmul_10

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_50 = paddle._C_ops.add(scale_7, scale_1)
        del scale_7

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_5 = paddle._C_ops.softmax(add_50, -1)
        del add_50

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_32, dropout_33 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_5, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_5

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_11 = paddle._C_ops.matmul(dropout_32, transpose_37, False, False)
        del dropout_32, transpose_37

        # pd_op.transpose: (1x12x64x21xf32) <- (1x12x21x64xf32)
        transpose_38 = paddle._C_ops.transpose(matmul_11, [0, 1, 3, 2])
        del matmul_11

        # pd_op.reshape: (1x768x21xf32) <- (1x12x64x21xf32, 3xi64)
        reshape_56 = paddle._C_ops.reshape(transpose_38, full_int_array_6)
        del transpose_38

        # pd_op.assign: (768x768x1xf32) <- (768x768x1xf32)
        assign_33 = parameter_108
        del parameter_108

        # pd_op.unsqueeze: (768x768x1x1xf32) <- (768x768x1xf32, 1xi64)
        unsqueeze_68 = paddle._C_ops.unsqueeze(assign_33, full_int_array_3)
        del assign_33

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_69 = paddle._C_ops.unsqueeze(reshape_56, full_int_array_3)
        del reshape_56

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x768x1x1xf32)
        conv2d_33 = paddle._C_ops.conv2d(
            unsqueeze_69, unsqueeze_68, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del unsqueeze_68, unsqueeze_69

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_57 = paddle._C_ops.reshape(parameter_107, full_int_array_4)
        del parameter_107

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_51 = paddle._C_ops.add(conv2d_33, reshape_57)
        del conv2d_33, reshape_57

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_33 = paddle._C_ops.squeeze(add_51, full_int_array_3)
        del add_51

        # pd_op.dropout: (1x768x21xf32, 1x768x21xui8) <- (1x768x21xf32, None, 1xf32)
        dropout_34, dropout_35 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                squeeze_33, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del squeeze_33

        # pd_op.add: (1x768x21xf32) <- (1x768x21xf32, 1x768x21xf32)
        add_52 = paddle._C_ops.add(dropout_34, transpose_35)
        del dropout_34, transpose_35

        # pd_op.transpose: (1x21x768xf32) <- (1x768x21xf32)
        transpose_39 = paddle._C_ops.transpose(add_52, [0, 2, 1])
        del add_52

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_39, parameter_106, parameter_105, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_105, parameter_106, transpose_39

        # pd_op.transpose: (1x768x21xf32) <- (1x21x768xf32)
        transpose_40 = paddle._C_ops.transpose(layer_norm_33, [0, 2, 1])
        del layer_norm_33

        # pd_op.assign: (3072x192x1xf32) <- (3072x192x1xf32)
        assign_34 = parameter_104
        del parameter_104

        # pd_op.unsqueeze: (3072x192x1x1xf32) <- (3072x192x1xf32, 1xi64)
        unsqueeze_70 = paddle._C_ops.unsqueeze(assign_34, full_int_array_3)
        del assign_34

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_71 = paddle._C_ops.unsqueeze(transpose_40, full_int_array_3)

        # pd_op.conv2d: (1x3072x1x21xf32) <- (1x768x1x21xf32, 3072x192x1x1xf32)
        conv2d_34 = paddle._C_ops.conv2d(
            unsqueeze_71, unsqueeze_70, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_70, unsqueeze_71

        # pd_op.reshape: (1x3072x1x1xf32) <- (3072xf32, 4xi64)
        reshape_58 = paddle._C_ops.reshape(parameter_103, full_int_array_7)
        del parameter_103

        # pd_op.add: (1x3072x1x21xf32) <- (1x3072x1x21xf32, 1x3072x1x1xf32)
        add_53 = paddle._C_ops.add(conv2d_34, reshape_58)
        del conv2d_34, reshape_58

        # pd_op.squeeze: (1x3072x21xf32) <- (1x3072x1x21xf32, 1xi64)
        squeeze_34 = paddle._C_ops.squeeze(add_53, full_int_array_3)
        del add_53

        # pd_op.gelu: (1x3072x21xf32) <- (1x3072x21xf32)
        gelu_5 = paddle._C_ops.gelu(squeeze_34, False)
        del squeeze_34

        # pd_op.assign: (768x768x1xf32) <- (768x768x1xf32)
        assign_35 = parameter_102
        del parameter_102

        # pd_op.unsqueeze: (768x768x1x1xf32) <- (768x768x1xf32, 1xi64)
        unsqueeze_72 = paddle._C_ops.unsqueeze(assign_35, full_int_array_3)
        del assign_35

        # pd_op.unsqueeze: (1x3072x1x21xf32) <- (1x3072x21xf32, 1xi64)
        unsqueeze_73 = paddle._C_ops.unsqueeze(gelu_5, full_int_array_3)
        del gelu_5

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x3072x1x21xf32, 768x768x1x1xf32)
        conv2d_35 = paddle._C_ops.conv2d(
            unsqueeze_73, unsqueeze_72, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_72, unsqueeze_73

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_59 = paddle._C_ops.reshape(parameter_101, full_int_array_4)
        del parameter_101

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_54 = paddle._C_ops.add(conv2d_35, reshape_59)
        del conv2d_35, reshape_59

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_35 = paddle._C_ops.squeeze(add_54, full_int_array_3)
        del add_54

        # pd_op.dropout: (1x768x21xf32, 1x768x21xui8) <- (1x768x21xf32, None, 1xf32)
        dropout_36, dropout_37 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                squeeze_35, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del squeeze_35

        # pd_op.add: (1x768x21xf32) <- (1x768x21xf32, 1x768x21xf32)
        add_55 = paddle._C_ops.add(dropout_36, transpose_40)
        del dropout_36, transpose_40

        # pd_op.transpose: (1x21x768xf32) <- (1x768x21xf32)
        transpose_41 = paddle._C_ops.transpose(add_55, [0, 2, 1])
        del add_55

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_41, parameter_100, parameter_99, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_100, parameter_99, transpose_41

        # pd_op.transpose: (1x768x21xf32) <- (1x21x768xf32)
        transpose_42 = paddle._C_ops.transpose(layer_norm_36, [0, 2, 1])
        del layer_norm_36

        # pd_op.assign: (768x192x1xf32) <- (768x192x1xf32)
        assign_36 = parameter_98
        del parameter_98

        # pd_op.unsqueeze: (768x192x1x1xf32) <- (768x192x1xf32, 1xi64)
        unsqueeze_74 = paddle._C_ops.unsqueeze(assign_36, full_int_array_3)
        del assign_36

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_75 = paddle._C_ops.unsqueeze(transpose_42, full_int_array_3)

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x192x1x1xf32)
        conv2d_36 = paddle._C_ops.conv2d(
            unsqueeze_75, unsqueeze_74, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_74, unsqueeze_75

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_60 = paddle._C_ops.reshape(parameter_97, full_int_array_4)
        del parameter_97

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_56 = paddle._C_ops.add(conv2d_36, reshape_60)
        del conv2d_36, reshape_60

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_36 = paddle._C_ops.squeeze(add_56, full_int_array_3)
        del add_56

        # pd_op.assign: (768x192x1xf32) <- (768x192x1xf32)
        assign_37 = parameter_96
        del parameter_96

        # pd_op.unsqueeze: (768x192x1x1xf32) <- (768x192x1xf32, 1xi64)
        unsqueeze_76 = paddle._C_ops.unsqueeze(assign_37, full_int_array_3)
        del assign_37

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_77 = paddle._C_ops.unsqueeze(transpose_42, full_int_array_3)

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x192x1x1xf32)
        conv2d_37 = paddle._C_ops.conv2d(
            unsqueeze_77, unsqueeze_76, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_76, unsqueeze_77

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_61 = paddle._C_ops.reshape(parameter_95, full_int_array_4)
        del parameter_95

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_57 = paddle._C_ops.add(conv2d_37, reshape_61)
        del conv2d_37, reshape_61

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_37 = paddle._C_ops.squeeze(add_57, full_int_array_3)
        del add_57

        # pd_op.assign: (768x192x1xf32) <- (768x192x1xf32)
        assign_38 = parameter_94
        del parameter_94

        # pd_op.unsqueeze: (768x192x1x1xf32) <- (768x192x1xf32, 1xi64)
        unsqueeze_78 = paddle._C_ops.unsqueeze(assign_38, full_int_array_3)
        del assign_38

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_79 = paddle._C_ops.unsqueeze(transpose_42, full_int_array_3)

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x192x1x1xf32)
        conv2d_38 = paddle._C_ops.conv2d(
            unsqueeze_79, unsqueeze_78, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_78, unsqueeze_79

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_62 = paddle._C_ops.reshape(parameter_93, full_int_array_4)
        del parameter_93

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_58 = paddle._C_ops.add(conv2d_38, reshape_62)
        del conv2d_38, reshape_62

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_38 = paddle._C_ops.squeeze(add_58, full_int_array_3)
        del add_58

        # pd_op.reshape: (1x12x64x21xf32) <- (1x768x21xf32, 4xi64)
        reshape_63 = paddle._C_ops.reshape(squeeze_36, full_int_array_5)
        del squeeze_36

        # pd_op.transpose: (1x12x21x64xf32) <- (1x12x64x21xf32)
        transpose_43 = paddle._C_ops.transpose(reshape_63, [0, 1, 3, 2])
        del reshape_63

        # pd_op.reshape: (1x12x64x21xf32) <- (1x768x21xf32, 4xi64)
        reshape_64 = paddle._C_ops.reshape(squeeze_37, full_int_array_5)
        del squeeze_37

        # pd_op.reshape: (1x12x64x21xf32) <- (1x768x21xf32, 4xi64)
        reshape_65 = paddle._C_ops.reshape(squeeze_38, full_int_array_5)
        del squeeze_38

        # pd_op.transpose: (1x12x21x64xf32) <- (1x12x64x21xf32)
        transpose_44 = paddle._C_ops.transpose(reshape_65, [0, 1, 3, 2])
        del reshape_65

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x64x21xf32)
        matmul_12 = paddle._C_ops.matmul(transpose_43, reshape_64, False, False)
        del reshape_64, transpose_43

        # pd_op.scale: (1x12x21x21xf32) <- (1x12x21x21xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(matmul_12, full_3, float("0"), True)
        del matmul_12

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_59 = paddle._C_ops.add(scale_8, scale_1)
        del scale_8

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_6 = paddle._C_ops.softmax(add_59, -1)
        del add_59

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_38, dropout_39 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_6, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_6

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_13 = paddle._C_ops.matmul(dropout_38, transpose_44, False, False)
        del dropout_38, transpose_44

        # pd_op.transpose: (1x12x64x21xf32) <- (1x12x21x64xf32)
        transpose_45 = paddle._C_ops.transpose(matmul_13, [0, 1, 3, 2])
        del matmul_13

        # pd_op.reshape: (1x768x21xf32) <- (1x12x64x21xf32, 3xi64)
        reshape_66 = paddle._C_ops.reshape(transpose_45, full_int_array_6)
        del transpose_45

        # pd_op.assign: (768x768x1xf32) <- (768x768x1xf32)
        assign_39 = parameter_92
        del parameter_92

        # pd_op.unsqueeze: (768x768x1x1xf32) <- (768x768x1xf32, 1xi64)
        unsqueeze_80 = paddle._C_ops.unsqueeze(assign_39, full_int_array_3)
        del assign_39

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_81 = paddle._C_ops.unsqueeze(reshape_66, full_int_array_3)
        del reshape_66

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x768x1x1xf32)
        conv2d_39 = paddle._C_ops.conv2d(
            unsqueeze_81, unsqueeze_80, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del unsqueeze_80, unsqueeze_81

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_67 = paddle._C_ops.reshape(parameter_91, full_int_array_4)
        del parameter_91

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_60 = paddle._C_ops.add(conv2d_39, reshape_67)
        del conv2d_39, reshape_67

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_39 = paddle._C_ops.squeeze(add_60, full_int_array_3)
        del add_60

        # pd_op.dropout: (1x768x21xf32, 1x768x21xui8) <- (1x768x21xf32, None, 1xf32)
        dropout_40, dropout_41 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                squeeze_39, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del squeeze_39

        # pd_op.add: (1x768x21xf32) <- (1x768x21xf32, 1x768x21xf32)
        add_61 = paddle._C_ops.add(dropout_40, transpose_42)
        del dropout_40, transpose_42

        # pd_op.transpose: (1x21x768xf32) <- (1x768x21xf32)
        transpose_46 = paddle._C_ops.transpose(add_61, [0, 2, 1])
        del add_61

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_46, parameter_90, parameter_89, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_89, parameter_90, transpose_46

        # pd_op.transpose: (1x768x21xf32) <- (1x21x768xf32)
        transpose_47 = paddle._C_ops.transpose(layer_norm_39, [0, 2, 1])
        del layer_norm_39

        # pd_op.assign: (3072x192x1xf32) <- (3072x192x1xf32)
        assign_40 = parameter_88
        del parameter_88

        # pd_op.unsqueeze: (3072x192x1x1xf32) <- (3072x192x1xf32, 1xi64)
        unsqueeze_82 = paddle._C_ops.unsqueeze(assign_40, full_int_array_3)
        del assign_40

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_83 = paddle._C_ops.unsqueeze(transpose_47, full_int_array_3)

        # pd_op.conv2d: (1x3072x1x21xf32) <- (1x768x1x21xf32, 3072x192x1x1xf32)
        conv2d_40 = paddle._C_ops.conv2d(
            unsqueeze_83, unsqueeze_82, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_82, unsqueeze_83

        # pd_op.reshape: (1x3072x1x1xf32) <- (3072xf32, 4xi64)
        reshape_68 = paddle._C_ops.reshape(parameter_87, full_int_array_7)
        del parameter_87

        # pd_op.add: (1x3072x1x21xf32) <- (1x3072x1x21xf32, 1x3072x1x1xf32)
        add_62 = paddle._C_ops.add(conv2d_40, reshape_68)
        del conv2d_40, reshape_68

        # pd_op.squeeze: (1x3072x21xf32) <- (1x3072x1x21xf32, 1xi64)
        squeeze_40 = paddle._C_ops.squeeze(add_62, full_int_array_3)
        del add_62

        # pd_op.gelu: (1x3072x21xf32) <- (1x3072x21xf32)
        gelu_6 = paddle._C_ops.gelu(squeeze_40, False)
        del squeeze_40

        # pd_op.assign: (768x768x1xf32) <- (768x768x1xf32)
        assign_41 = parameter_86
        del parameter_86

        # pd_op.unsqueeze: (768x768x1x1xf32) <- (768x768x1xf32, 1xi64)
        unsqueeze_84 = paddle._C_ops.unsqueeze(assign_41, full_int_array_3)
        del assign_41

        # pd_op.unsqueeze: (1x3072x1x21xf32) <- (1x3072x21xf32, 1xi64)
        unsqueeze_85 = paddle._C_ops.unsqueeze(gelu_6, full_int_array_3)
        del gelu_6

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x3072x1x21xf32, 768x768x1x1xf32)
        conv2d_41 = paddle._C_ops.conv2d(
            unsqueeze_85, unsqueeze_84, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_84, unsqueeze_85

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_69 = paddle._C_ops.reshape(parameter_85, full_int_array_4)
        del parameter_85

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_63 = paddle._C_ops.add(conv2d_41, reshape_69)
        del conv2d_41, reshape_69

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_41 = paddle._C_ops.squeeze(add_63, full_int_array_3)
        del add_63

        # pd_op.dropout: (1x768x21xf32, 1x768x21xui8) <- (1x768x21xf32, None, 1xf32)
        dropout_42, dropout_43 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                squeeze_41, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del squeeze_41

        # pd_op.add: (1x768x21xf32) <- (1x768x21xf32, 1x768x21xf32)
        add_64 = paddle._C_ops.add(dropout_42, transpose_47)
        del dropout_42, transpose_47

        # pd_op.transpose: (1x21x768xf32) <- (1x768x21xf32)
        transpose_48 = paddle._C_ops.transpose(add_64, [0, 2, 1])
        del add_64

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_48, parameter_84, parameter_83, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_83, parameter_84, transpose_48

        # pd_op.transpose: (1x768x21xf32) <- (1x21x768xf32)
        transpose_49 = paddle._C_ops.transpose(layer_norm_42, [0, 2, 1])
        del layer_norm_42

        # pd_op.assign: (768x192x1xf32) <- (768x192x1xf32)
        assign_42 = parameter_82
        del parameter_82

        # pd_op.unsqueeze: (768x192x1x1xf32) <- (768x192x1xf32, 1xi64)
        unsqueeze_86 = paddle._C_ops.unsqueeze(assign_42, full_int_array_3)
        del assign_42

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_87 = paddle._C_ops.unsqueeze(transpose_49, full_int_array_3)

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x192x1x1xf32)
        conv2d_42 = paddle._C_ops.conv2d(
            unsqueeze_87, unsqueeze_86, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_86, unsqueeze_87

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_70 = paddle._C_ops.reshape(parameter_81, full_int_array_4)
        del parameter_81

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_65 = paddle._C_ops.add(conv2d_42, reshape_70)
        del conv2d_42, reshape_70

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_42 = paddle._C_ops.squeeze(add_65, full_int_array_3)
        del add_65

        # pd_op.assign: (768x192x1xf32) <- (768x192x1xf32)
        assign_43 = parameter_80
        del parameter_80

        # pd_op.unsqueeze: (768x192x1x1xf32) <- (768x192x1xf32, 1xi64)
        unsqueeze_88 = paddle._C_ops.unsqueeze(assign_43, full_int_array_3)
        del assign_43

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_89 = paddle._C_ops.unsqueeze(transpose_49, full_int_array_3)

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x192x1x1xf32)
        conv2d_43 = paddle._C_ops.conv2d(
            unsqueeze_89, unsqueeze_88, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_88, unsqueeze_89

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_71 = paddle._C_ops.reshape(parameter_79, full_int_array_4)
        del parameter_79

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_66 = paddle._C_ops.add(conv2d_43, reshape_71)
        del conv2d_43, reshape_71

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_43 = paddle._C_ops.squeeze(add_66, full_int_array_3)
        del add_66

        # pd_op.assign: (768x192x1xf32) <- (768x192x1xf32)
        assign_44 = parameter_78
        del parameter_78

        # pd_op.unsqueeze: (768x192x1x1xf32) <- (768x192x1xf32, 1xi64)
        unsqueeze_90 = paddle._C_ops.unsqueeze(assign_44, full_int_array_3)
        del assign_44

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_91 = paddle._C_ops.unsqueeze(transpose_49, full_int_array_3)

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x192x1x1xf32)
        conv2d_44 = paddle._C_ops.conv2d(
            unsqueeze_91, unsqueeze_90, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_90, unsqueeze_91

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_72 = paddle._C_ops.reshape(parameter_77, full_int_array_4)
        del parameter_77

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_67 = paddle._C_ops.add(conv2d_44, reshape_72)
        del conv2d_44, reshape_72

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_44 = paddle._C_ops.squeeze(add_67, full_int_array_3)
        del add_67

        # pd_op.reshape: (1x12x64x21xf32) <- (1x768x21xf32, 4xi64)
        reshape_73 = paddle._C_ops.reshape(squeeze_42, full_int_array_5)
        del squeeze_42

        # pd_op.transpose: (1x12x21x64xf32) <- (1x12x64x21xf32)
        transpose_50 = paddle._C_ops.transpose(reshape_73, [0, 1, 3, 2])
        del reshape_73

        # pd_op.reshape: (1x12x64x21xf32) <- (1x768x21xf32, 4xi64)
        reshape_74 = paddle._C_ops.reshape(squeeze_43, full_int_array_5)
        del squeeze_43

        # pd_op.reshape: (1x12x64x21xf32) <- (1x768x21xf32, 4xi64)
        reshape_75 = paddle._C_ops.reshape(squeeze_44, full_int_array_5)
        del squeeze_44

        # pd_op.transpose: (1x12x21x64xf32) <- (1x12x64x21xf32)
        transpose_51 = paddle._C_ops.transpose(reshape_75, [0, 1, 3, 2])
        del reshape_75

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x64x21xf32)
        matmul_14 = paddle._C_ops.matmul(transpose_50, reshape_74, False, False)
        del reshape_74, transpose_50

        # pd_op.scale: (1x12x21x21xf32) <- (1x12x21x21xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(matmul_14, full_3, float("0"), True)
        del matmul_14

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_68 = paddle._C_ops.add(scale_9, scale_1)
        del scale_9

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_7 = paddle._C_ops.softmax(add_68, -1)
        del add_68

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_44, dropout_45 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_7, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_7

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_15 = paddle._C_ops.matmul(dropout_44, transpose_51, False, False)
        del dropout_44, transpose_51

        # pd_op.transpose: (1x12x64x21xf32) <- (1x12x21x64xf32)
        transpose_52 = paddle._C_ops.transpose(matmul_15, [0, 1, 3, 2])
        del matmul_15

        # pd_op.reshape: (1x768x21xf32) <- (1x12x64x21xf32, 3xi64)
        reshape_76 = paddle._C_ops.reshape(transpose_52, full_int_array_6)
        del transpose_52

        # pd_op.assign: (768x768x1xf32) <- (768x768x1xf32)
        assign_45 = parameter_76
        del parameter_76

        # pd_op.unsqueeze: (768x768x1x1xf32) <- (768x768x1xf32, 1xi64)
        unsqueeze_92 = paddle._C_ops.unsqueeze(assign_45, full_int_array_3)
        del assign_45

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_93 = paddle._C_ops.unsqueeze(reshape_76, full_int_array_3)
        del reshape_76

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x768x1x1xf32)
        conv2d_45 = paddle._C_ops.conv2d(
            unsqueeze_93, unsqueeze_92, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del unsqueeze_92, unsqueeze_93

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_77 = paddle._C_ops.reshape(parameter_75, full_int_array_4)
        del parameter_75

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_69 = paddle._C_ops.add(conv2d_45, reshape_77)
        del conv2d_45, reshape_77

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_45 = paddle._C_ops.squeeze(add_69, full_int_array_3)
        del add_69

        # pd_op.dropout: (1x768x21xf32, 1x768x21xui8) <- (1x768x21xf32, None, 1xf32)
        dropout_46, dropout_47 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                squeeze_45, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del squeeze_45

        # pd_op.add: (1x768x21xf32) <- (1x768x21xf32, 1x768x21xf32)
        add_70 = paddle._C_ops.add(dropout_46, transpose_49)
        del dropout_46, transpose_49

        # pd_op.transpose: (1x21x768xf32) <- (1x768x21xf32)
        transpose_53 = paddle._C_ops.transpose(add_70, [0, 2, 1])
        del add_70

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_53, parameter_74, parameter_73, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_73, parameter_74, transpose_53

        # pd_op.transpose: (1x768x21xf32) <- (1x21x768xf32)
        transpose_54 = paddle._C_ops.transpose(layer_norm_45, [0, 2, 1])
        del layer_norm_45

        # pd_op.assign: (3072x192x1xf32) <- (3072x192x1xf32)
        assign_46 = parameter_72
        del parameter_72

        # pd_op.unsqueeze: (3072x192x1x1xf32) <- (3072x192x1xf32, 1xi64)
        unsqueeze_94 = paddle._C_ops.unsqueeze(assign_46, full_int_array_3)
        del assign_46

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_95 = paddle._C_ops.unsqueeze(transpose_54, full_int_array_3)

        # pd_op.conv2d: (1x3072x1x21xf32) <- (1x768x1x21xf32, 3072x192x1x1xf32)
        conv2d_46 = paddle._C_ops.conv2d(
            unsqueeze_95, unsqueeze_94, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_94, unsqueeze_95

        # pd_op.reshape: (1x3072x1x1xf32) <- (3072xf32, 4xi64)
        reshape_78 = paddle._C_ops.reshape(parameter_71, full_int_array_7)
        del parameter_71

        # pd_op.add: (1x3072x1x21xf32) <- (1x3072x1x21xf32, 1x3072x1x1xf32)
        add_71 = paddle._C_ops.add(conv2d_46, reshape_78)
        del conv2d_46, reshape_78

        # pd_op.squeeze: (1x3072x21xf32) <- (1x3072x1x21xf32, 1xi64)
        squeeze_46 = paddle._C_ops.squeeze(add_71, full_int_array_3)
        del add_71

        # pd_op.gelu: (1x3072x21xf32) <- (1x3072x21xf32)
        gelu_7 = paddle._C_ops.gelu(squeeze_46, False)
        del squeeze_46

        # pd_op.assign: (768x768x1xf32) <- (768x768x1xf32)
        assign_47 = parameter_70
        del parameter_70

        # pd_op.unsqueeze: (768x768x1x1xf32) <- (768x768x1xf32, 1xi64)
        unsqueeze_96 = paddle._C_ops.unsqueeze(assign_47, full_int_array_3)
        del assign_47

        # pd_op.unsqueeze: (1x3072x1x21xf32) <- (1x3072x21xf32, 1xi64)
        unsqueeze_97 = paddle._C_ops.unsqueeze(gelu_7, full_int_array_3)
        del gelu_7

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x3072x1x21xf32, 768x768x1x1xf32)
        conv2d_47 = paddle._C_ops.conv2d(
            unsqueeze_97, unsqueeze_96, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_96, unsqueeze_97

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_79 = paddle._C_ops.reshape(parameter_69, full_int_array_4)
        del parameter_69

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_72 = paddle._C_ops.add(conv2d_47, reshape_79)
        del conv2d_47, reshape_79

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_47 = paddle._C_ops.squeeze(add_72, full_int_array_3)
        del add_72

        # pd_op.dropout: (1x768x21xf32, 1x768x21xui8) <- (1x768x21xf32, None, 1xf32)
        dropout_48, dropout_49 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                squeeze_47, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del squeeze_47

        # pd_op.add: (1x768x21xf32) <- (1x768x21xf32, 1x768x21xf32)
        add_73 = paddle._C_ops.add(dropout_48, transpose_54)
        del dropout_48, transpose_54

        # pd_op.transpose: (1x21x768xf32) <- (1x768x21xf32)
        transpose_55 = paddle._C_ops.transpose(add_73, [0, 2, 1])
        del add_73

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_55, parameter_68, parameter_67, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_67, parameter_68, transpose_55

        # pd_op.transpose: (1x768x21xf32) <- (1x21x768xf32)
        transpose_56 = paddle._C_ops.transpose(layer_norm_48, [0, 2, 1])
        del layer_norm_48

        # pd_op.assign: (768x192x1xf32) <- (768x192x1xf32)
        assign_48 = parameter_66
        del parameter_66

        # pd_op.unsqueeze: (768x192x1x1xf32) <- (768x192x1xf32, 1xi64)
        unsqueeze_98 = paddle._C_ops.unsqueeze(assign_48, full_int_array_3)
        del assign_48

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_99 = paddle._C_ops.unsqueeze(transpose_56, full_int_array_3)

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x192x1x1xf32)
        conv2d_48 = paddle._C_ops.conv2d(
            unsqueeze_99, unsqueeze_98, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_98, unsqueeze_99

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_80 = paddle._C_ops.reshape(parameter_65, full_int_array_4)
        del parameter_65

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_74 = paddle._C_ops.add(conv2d_48, reshape_80)
        del conv2d_48, reshape_80

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_48 = paddle._C_ops.squeeze(add_74, full_int_array_3)
        del add_74

        # pd_op.assign: (768x192x1xf32) <- (768x192x1xf32)
        assign_49 = parameter_64
        del parameter_64

        # pd_op.unsqueeze: (768x192x1x1xf32) <- (768x192x1xf32, 1xi64)
        unsqueeze_100 = paddle._C_ops.unsqueeze(assign_49, full_int_array_3)
        del assign_49

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_101 = paddle._C_ops.unsqueeze(transpose_56, full_int_array_3)

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x192x1x1xf32)
        conv2d_49 = paddle._C_ops.conv2d(
            unsqueeze_101, unsqueeze_100, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_100, unsqueeze_101

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_81 = paddle._C_ops.reshape(parameter_63, full_int_array_4)
        del parameter_63

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_75 = paddle._C_ops.add(conv2d_49, reshape_81)
        del conv2d_49, reshape_81

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_49 = paddle._C_ops.squeeze(add_75, full_int_array_3)
        del add_75

        # pd_op.assign: (768x192x1xf32) <- (768x192x1xf32)
        assign_50 = parameter_62
        del parameter_62

        # pd_op.unsqueeze: (768x192x1x1xf32) <- (768x192x1xf32, 1xi64)
        unsqueeze_102 = paddle._C_ops.unsqueeze(assign_50, full_int_array_3)
        del assign_50

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_103 = paddle._C_ops.unsqueeze(transpose_56, full_int_array_3)

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x192x1x1xf32)
        conv2d_50 = paddle._C_ops.conv2d(
            unsqueeze_103, unsqueeze_102, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_102, unsqueeze_103

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_82 = paddle._C_ops.reshape(parameter_61, full_int_array_4)
        del parameter_61

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_76 = paddle._C_ops.add(conv2d_50, reshape_82)
        del conv2d_50, reshape_82

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_50 = paddle._C_ops.squeeze(add_76, full_int_array_3)
        del add_76

        # pd_op.reshape: (1x12x64x21xf32) <- (1x768x21xf32, 4xi64)
        reshape_83 = paddle._C_ops.reshape(squeeze_48, full_int_array_5)
        del squeeze_48

        # pd_op.transpose: (1x12x21x64xf32) <- (1x12x64x21xf32)
        transpose_57 = paddle._C_ops.transpose(reshape_83, [0, 1, 3, 2])
        del reshape_83

        # pd_op.reshape: (1x12x64x21xf32) <- (1x768x21xf32, 4xi64)
        reshape_84 = paddle._C_ops.reshape(squeeze_49, full_int_array_5)
        del squeeze_49

        # pd_op.reshape: (1x12x64x21xf32) <- (1x768x21xf32, 4xi64)
        reshape_85 = paddle._C_ops.reshape(squeeze_50, full_int_array_5)
        del squeeze_50

        # pd_op.transpose: (1x12x21x64xf32) <- (1x12x64x21xf32)
        transpose_58 = paddle._C_ops.transpose(reshape_85, [0, 1, 3, 2])
        del reshape_85

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x64x21xf32)
        matmul_16 = paddle._C_ops.matmul(transpose_57, reshape_84, False, False)
        del reshape_84, transpose_57

        # pd_op.scale: (1x12x21x21xf32) <- (1x12x21x21xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(matmul_16, full_3, float("0"), True)
        del matmul_16

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_77 = paddle._C_ops.add(scale_10, scale_1)
        del scale_10

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_8 = paddle._C_ops.softmax(add_77, -1)
        del add_77

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_50, dropout_51 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_8, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_8

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_17 = paddle._C_ops.matmul(dropout_50, transpose_58, False, False)
        del dropout_50, transpose_58

        # pd_op.transpose: (1x12x64x21xf32) <- (1x12x21x64xf32)
        transpose_59 = paddle._C_ops.transpose(matmul_17, [0, 1, 3, 2])
        del matmul_17

        # pd_op.reshape: (1x768x21xf32) <- (1x12x64x21xf32, 3xi64)
        reshape_86 = paddle._C_ops.reshape(transpose_59, full_int_array_6)
        del transpose_59

        # pd_op.assign: (768x768x1xf32) <- (768x768x1xf32)
        assign_51 = parameter_60
        del parameter_60

        # pd_op.unsqueeze: (768x768x1x1xf32) <- (768x768x1xf32, 1xi64)
        unsqueeze_104 = paddle._C_ops.unsqueeze(assign_51, full_int_array_3)
        del assign_51

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_105 = paddle._C_ops.unsqueeze(reshape_86, full_int_array_3)
        del reshape_86

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x768x1x1xf32)
        conv2d_51 = paddle._C_ops.conv2d(
            unsqueeze_105, unsqueeze_104, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del unsqueeze_104, unsqueeze_105

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_87 = paddle._C_ops.reshape(parameter_59, full_int_array_4)
        del parameter_59

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_78 = paddle._C_ops.add(conv2d_51, reshape_87)
        del conv2d_51, reshape_87

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_51 = paddle._C_ops.squeeze(add_78, full_int_array_3)
        del add_78

        # pd_op.dropout: (1x768x21xf32, 1x768x21xui8) <- (1x768x21xf32, None, 1xf32)
        dropout_52, dropout_53 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                squeeze_51, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del squeeze_51

        # pd_op.add: (1x768x21xf32) <- (1x768x21xf32, 1x768x21xf32)
        add_79 = paddle._C_ops.add(dropout_52, transpose_56)
        del dropout_52, transpose_56

        # pd_op.transpose: (1x21x768xf32) <- (1x768x21xf32)
        transpose_60 = paddle._C_ops.transpose(add_79, [0, 2, 1])
        del add_79

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_60, parameter_58, parameter_57, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_57, parameter_58, transpose_60

        # pd_op.transpose: (1x768x21xf32) <- (1x21x768xf32)
        transpose_61 = paddle._C_ops.transpose(layer_norm_51, [0, 2, 1])
        del layer_norm_51

        # pd_op.assign: (3072x192x1xf32) <- (3072x192x1xf32)
        assign_52 = parameter_56
        del parameter_56

        # pd_op.unsqueeze: (3072x192x1x1xf32) <- (3072x192x1xf32, 1xi64)
        unsqueeze_106 = paddle._C_ops.unsqueeze(assign_52, full_int_array_3)
        del assign_52

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_107 = paddle._C_ops.unsqueeze(transpose_61, full_int_array_3)

        # pd_op.conv2d: (1x3072x1x21xf32) <- (1x768x1x21xf32, 3072x192x1x1xf32)
        conv2d_52 = paddle._C_ops.conv2d(
            unsqueeze_107, unsqueeze_106, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_106, unsqueeze_107

        # pd_op.reshape: (1x3072x1x1xf32) <- (3072xf32, 4xi64)
        reshape_88 = paddle._C_ops.reshape(parameter_55, full_int_array_7)
        del parameter_55

        # pd_op.add: (1x3072x1x21xf32) <- (1x3072x1x21xf32, 1x3072x1x1xf32)
        add_80 = paddle._C_ops.add(conv2d_52, reshape_88)
        del conv2d_52, reshape_88

        # pd_op.squeeze: (1x3072x21xf32) <- (1x3072x1x21xf32, 1xi64)
        squeeze_52 = paddle._C_ops.squeeze(add_80, full_int_array_3)
        del add_80

        # pd_op.gelu: (1x3072x21xf32) <- (1x3072x21xf32)
        gelu_8 = paddle._C_ops.gelu(squeeze_52, False)
        del squeeze_52

        # pd_op.assign: (768x768x1xf32) <- (768x768x1xf32)
        assign_53 = parameter_54
        del parameter_54

        # pd_op.unsqueeze: (768x768x1x1xf32) <- (768x768x1xf32, 1xi64)
        unsqueeze_108 = paddle._C_ops.unsqueeze(assign_53, full_int_array_3)
        del assign_53

        # pd_op.unsqueeze: (1x3072x1x21xf32) <- (1x3072x21xf32, 1xi64)
        unsqueeze_109 = paddle._C_ops.unsqueeze(gelu_8, full_int_array_3)
        del gelu_8

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x3072x1x21xf32, 768x768x1x1xf32)
        conv2d_53 = paddle._C_ops.conv2d(
            unsqueeze_109, unsqueeze_108, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_108, unsqueeze_109

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_89 = paddle._C_ops.reshape(parameter_53, full_int_array_4)
        del parameter_53

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_81 = paddle._C_ops.add(conv2d_53, reshape_89)
        del conv2d_53, reshape_89

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_53 = paddle._C_ops.squeeze(add_81, full_int_array_3)
        del add_81

        # pd_op.dropout: (1x768x21xf32, 1x768x21xui8) <- (1x768x21xf32, None, 1xf32)
        dropout_54, dropout_55 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                squeeze_53, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del squeeze_53

        # pd_op.add: (1x768x21xf32) <- (1x768x21xf32, 1x768x21xf32)
        add_82 = paddle._C_ops.add(dropout_54, transpose_61)
        del dropout_54, transpose_61

        # pd_op.transpose: (1x21x768xf32) <- (1x768x21xf32)
        transpose_62 = paddle._C_ops.transpose(add_82, [0, 2, 1])
        del add_82

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_62, parameter_52, parameter_51, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_51, parameter_52, transpose_62

        # pd_op.transpose: (1x768x21xf32) <- (1x21x768xf32)
        transpose_63 = paddle._C_ops.transpose(layer_norm_54, [0, 2, 1])
        del layer_norm_54

        # pd_op.assign: (768x192x1xf32) <- (768x192x1xf32)
        assign_54 = parameter_50
        del parameter_50

        # pd_op.unsqueeze: (768x192x1x1xf32) <- (768x192x1xf32, 1xi64)
        unsqueeze_110 = paddle._C_ops.unsqueeze(assign_54, full_int_array_3)
        del assign_54

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_111 = paddle._C_ops.unsqueeze(transpose_63, full_int_array_3)

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x192x1x1xf32)
        conv2d_54 = paddle._C_ops.conv2d(
            unsqueeze_111, unsqueeze_110, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_110, unsqueeze_111

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_90 = paddle._C_ops.reshape(parameter_49, full_int_array_4)
        del parameter_49

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_83 = paddle._C_ops.add(conv2d_54, reshape_90)
        del conv2d_54, reshape_90

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_54 = paddle._C_ops.squeeze(add_83, full_int_array_3)
        del add_83

        # pd_op.assign: (768x192x1xf32) <- (768x192x1xf32)
        assign_55 = parameter_48
        del parameter_48

        # pd_op.unsqueeze: (768x192x1x1xf32) <- (768x192x1xf32, 1xi64)
        unsqueeze_112 = paddle._C_ops.unsqueeze(assign_55, full_int_array_3)
        del assign_55

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_113 = paddle._C_ops.unsqueeze(transpose_63, full_int_array_3)

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x192x1x1xf32)
        conv2d_55 = paddle._C_ops.conv2d(
            unsqueeze_113, unsqueeze_112, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_112, unsqueeze_113

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_91 = paddle._C_ops.reshape(parameter_47, full_int_array_4)
        del parameter_47

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_84 = paddle._C_ops.add(conv2d_55, reshape_91)
        del conv2d_55, reshape_91

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_55 = paddle._C_ops.squeeze(add_84, full_int_array_3)
        del add_84

        # pd_op.assign: (768x192x1xf32) <- (768x192x1xf32)
        assign_56 = parameter_46
        del parameter_46

        # pd_op.unsqueeze: (768x192x1x1xf32) <- (768x192x1xf32, 1xi64)
        unsqueeze_114 = paddle._C_ops.unsqueeze(assign_56, full_int_array_3)
        del assign_56

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_115 = paddle._C_ops.unsqueeze(transpose_63, full_int_array_3)

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x192x1x1xf32)
        conv2d_56 = paddle._C_ops.conv2d(
            unsqueeze_115, unsqueeze_114, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_114, unsqueeze_115

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_92 = paddle._C_ops.reshape(parameter_45, full_int_array_4)
        del parameter_45

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_85 = paddle._C_ops.add(conv2d_56, reshape_92)
        del conv2d_56, reshape_92

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_56 = paddle._C_ops.squeeze(add_85, full_int_array_3)
        del add_85

        # pd_op.reshape: (1x12x64x21xf32) <- (1x768x21xf32, 4xi64)
        reshape_93 = paddle._C_ops.reshape(squeeze_54, full_int_array_5)
        del squeeze_54

        # pd_op.transpose: (1x12x21x64xf32) <- (1x12x64x21xf32)
        transpose_64 = paddle._C_ops.transpose(reshape_93, [0, 1, 3, 2])
        del reshape_93

        # pd_op.reshape: (1x12x64x21xf32) <- (1x768x21xf32, 4xi64)
        reshape_94 = paddle._C_ops.reshape(squeeze_55, full_int_array_5)
        del squeeze_55

        # pd_op.reshape: (1x12x64x21xf32) <- (1x768x21xf32, 4xi64)
        reshape_95 = paddle._C_ops.reshape(squeeze_56, full_int_array_5)
        del squeeze_56

        # pd_op.transpose: (1x12x21x64xf32) <- (1x12x64x21xf32)
        transpose_65 = paddle._C_ops.transpose(reshape_95, [0, 1, 3, 2])
        del reshape_95

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x64x21xf32)
        matmul_18 = paddle._C_ops.matmul(transpose_64, reshape_94, False, False)
        del reshape_94, transpose_64

        # pd_op.scale: (1x12x21x21xf32) <- (1x12x21x21xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(matmul_18, full_3, float("0"), True)
        del matmul_18

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_86 = paddle._C_ops.add(scale_11, scale_1)
        del scale_11

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_9 = paddle._C_ops.softmax(add_86, -1)
        del add_86

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_56, dropout_57 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_9, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_9

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_19 = paddle._C_ops.matmul(dropout_56, transpose_65, False, False)
        del dropout_56, transpose_65

        # pd_op.transpose: (1x12x64x21xf32) <- (1x12x21x64xf32)
        transpose_66 = paddle._C_ops.transpose(matmul_19, [0, 1, 3, 2])
        del matmul_19

        # pd_op.reshape: (1x768x21xf32) <- (1x12x64x21xf32, 3xi64)
        reshape_96 = paddle._C_ops.reshape(transpose_66, full_int_array_6)
        del transpose_66

        # pd_op.assign: (768x768x1xf32) <- (768x768x1xf32)
        assign_57 = parameter_44
        del parameter_44

        # pd_op.unsqueeze: (768x768x1x1xf32) <- (768x768x1xf32, 1xi64)
        unsqueeze_116 = paddle._C_ops.unsqueeze(assign_57, full_int_array_3)
        del assign_57

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_117 = paddle._C_ops.unsqueeze(reshape_96, full_int_array_3)
        del reshape_96

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x768x1x1xf32)
        conv2d_57 = paddle._C_ops.conv2d(
            unsqueeze_117, unsqueeze_116, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del unsqueeze_116, unsqueeze_117

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_97 = paddle._C_ops.reshape(parameter_43, full_int_array_4)
        del parameter_43

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_87 = paddle._C_ops.add(conv2d_57, reshape_97)
        del conv2d_57, reshape_97

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_57 = paddle._C_ops.squeeze(add_87, full_int_array_3)
        del add_87

        # pd_op.dropout: (1x768x21xf32, 1x768x21xui8) <- (1x768x21xf32, None, 1xf32)
        dropout_58, dropout_59 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                squeeze_57, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del squeeze_57

        # pd_op.add: (1x768x21xf32) <- (1x768x21xf32, 1x768x21xf32)
        add_88 = paddle._C_ops.add(dropout_58, transpose_63)
        del dropout_58, transpose_63

        # pd_op.transpose: (1x21x768xf32) <- (1x768x21xf32)
        transpose_67 = paddle._C_ops.transpose(add_88, [0, 2, 1])
        del add_88

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_67, parameter_42, parameter_41, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_41, parameter_42, transpose_67

        # pd_op.transpose: (1x768x21xf32) <- (1x21x768xf32)
        transpose_68 = paddle._C_ops.transpose(layer_norm_57, [0, 2, 1])
        del layer_norm_57

        # pd_op.assign: (3072x192x1xf32) <- (3072x192x1xf32)
        assign_58 = parameter_40
        del parameter_40

        # pd_op.unsqueeze: (3072x192x1x1xf32) <- (3072x192x1xf32, 1xi64)
        unsqueeze_118 = paddle._C_ops.unsqueeze(assign_58, full_int_array_3)
        del assign_58

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_119 = paddle._C_ops.unsqueeze(transpose_68, full_int_array_3)

        # pd_op.conv2d: (1x3072x1x21xf32) <- (1x768x1x21xf32, 3072x192x1x1xf32)
        conv2d_58 = paddle._C_ops.conv2d(
            unsqueeze_119, unsqueeze_118, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_118, unsqueeze_119

        # pd_op.reshape: (1x3072x1x1xf32) <- (3072xf32, 4xi64)
        reshape_98 = paddle._C_ops.reshape(parameter_39, full_int_array_7)
        del parameter_39

        # pd_op.add: (1x3072x1x21xf32) <- (1x3072x1x21xf32, 1x3072x1x1xf32)
        add_89 = paddle._C_ops.add(conv2d_58, reshape_98)
        del conv2d_58, reshape_98

        # pd_op.squeeze: (1x3072x21xf32) <- (1x3072x1x21xf32, 1xi64)
        squeeze_58 = paddle._C_ops.squeeze(add_89, full_int_array_3)
        del add_89

        # pd_op.gelu: (1x3072x21xf32) <- (1x3072x21xf32)
        gelu_9 = paddle._C_ops.gelu(squeeze_58, False)
        del squeeze_58

        # pd_op.assign: (768x768x1xf32) <- (768x768x1xf32)
        assign_59 = parameter_38
        del parameter_38

        # pd_op.unsqueeze: (768x768x1x1xf32) <- (768x768x1xf32, 1xi64)
        unsqueeze_120 = paddle._C_ops.unsqueeze(assign_59, full_int_array_3)
        del assign_59

        # pd_op.unsqueeze: (1x3072x1x21xf32) <- (1x3072x21xf32, 1xi64)
        unsqueeze_121 = paddle._C_ops.unsqueeze(gelu_9, full_int_array_3)
        del gelu_9

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x3072x1x21xf32, 768x768x1x1xf32)
        conv2d_59 = paddle._C_ops.conv2d(
            unsqueeze_121, unsqueeze_120, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_120, unsqueeze_121

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_99 = paddle._C_ops.reshape(parameter_37, full_int_array_4)
        del parameter_37

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_90 = paddle._C_ops.add(conv2d_59, reshape_99)
        del conv2d_59, reshape_99

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_59 = paddle._C_ops.squeeze(add_90, full_int_array_3)
        del add_90

        # pd_op.dropout: (1x768x21xf32, 1x768x21xui8) <- (1x768x21xf32, None, 1xf32)
        dropout_60, dropout_61 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                squeeze_59, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del squeeze_59

        # pd_op.add: (1x768x21xf32) <- (1x768x21xf32, 1x768x21xf32)
        add_91 = paddle._C_ops.add(dropout_60, transpose_68)
        del dropout_60, transpose_68

        # pd_op.transpose: (1x21x768xf32) <- (1x768x21xf32)
        transpose_69 = paddle._C_ops.transpose(add_91, [0, 2, 1])
        del add_91

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_69, parameter_36, parameter_35, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_35, parameter_36, transpose_69

        # pd_op.transpose: (1x768x21xf32) <- (1x21x768xf32)
        transpose_70 = paddle._C_ops.transpose(layer_norm_60, [0, 2, 1])
        del layer_norm_60

        # pd_op.assign: (768x192x1xf32) <- (768x192x1xf32)
        assign_60 = parameter_34
        del parameter_34

        # pd_op.unsqueeze: (768x192x1x1xf32) <- (768x192x1xf32, 1xi64)
        unsqueeze_122 = paddle._C_ops.unsqueeze(assign_60, full_int_array_3)
        del assign_60

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_123 = paddle._C_ops.unsqueeze(transpose_70, full_int_array_3)

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x192x1x1xf32)
        conv2d_60 = paddle._C_ops.conv2d(
            unsqueeze_123, unsqueeze_122, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_122, unsqueeze_123

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_100 = paddle._C_ops.reshape(parameter_33, full_int_array_4)
        del parameter_33

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_92 = paddle._C_ops.add(conv2d_60, reshape_100)
        del conv2d_60, reshape_100

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_60 = paddle._C_ops.squeeze(add_92, full_int_array_3)
        del add_92

        # pd_op.assign: (768x192x1xf32) <- (768x192x1xf32)
        assign_61 = parameter_32
        del parameter_32

        # pd_op.unsqueeze: (768x192x1x1xf32) <- (768x192x1xf32, 1xi64)
        unsqueeze_124 = paddle._C_ops.unsqueeze(assign_61, full_int_array_3)
        del assign_61

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_125 = paddle._C_ops.unsqueeze(transpose_70, full_int_array_3)

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x192x1x1xf32)
        conv2d_61 = paddle._C_ops.conv2d(
            unsqueeze_125, unsqueeze_124, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_124, unsqueeze_125

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_101 = paddle._C_ops.reshape(parameter_31, full_int_array_4)
        del parameter_31

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_93 = paddle._C_ops.add(conv2d_61, reshape_101)
        del conv2d_61, reshape_101

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_61 = paddle._C_ops.squeeze(add_93, full_int_array_3)
        del add_93

        # pd_op.assign: (768x192x1xf32) <- (768x192x1xf32)
        assign_62 = parameter_30
        del parameter_30

        # pd_op.unsqueeze: (768x192x1x1xf32) <- (768x192x1xf32, 1xi64)
        unsqueeze_126 = paddle._C_ops.unsqueeze(assign_62, full_int_array_3)
        del assign_62

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_127 = paddle._C_ops.unsqueeze(transpose_70, full_int_array_3)

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x192x1x1xf32)
        conv2d_62 = paddle._C_ops.conv2d(
            unsqueeze_127, unsqueeze_126, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_126, unsqueeze_127

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_102 = paddle._C_ops.reshape(parameter_29, full_int_array_4)
        del parameter_29

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_94 = paddle._C_ops.add(conv2d_62, reshape_102)
        del conv2d_62, reshape_102

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_62 = paddle._C_ops.squeeze(add_94, full_int_array_3)
        del add_94

        # pd_op.reshape: (1x12x64x21xf32) <- (1x768x21xf32, 4xi64)
        reshape_103 = paddle._C_ops.reshape(squeeze_60, full_int_array_5)
        del squeeze_60

        # pd_op.transpose: (1x12x21x64xf32) <- (1x12x64x21xf32)
        transpose_71 = paddle._C_ops.transpose(reshape_103, [0, 1, 3, 2])
        del reshape_103

        # pd_op.reshape: (1x12x64x21xf32) <- (1x768x21xf32, 4xi64)
        reshape_104 = paddle._C_ops.reshape(squeeze_61, full_int_array_5)
        del squeeze_61

        # pd_op.reshape: (1x12x64x21xf32) <- (1x768x21xf32, 4xi64)
        reshape_105 = paddle._C_ops.reshape(squeeze_62, full_int_array_5)
        del squeeze_62

        # pd_op.transpose: (1x12x21x64xf32) <- (1x12x64x21xf32)
        transpose_72 = paddle._C_ops.transpose(reshape_105, [0, 1, 3, 2])
        del reshape_105

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x64x21xf32)
        matmul_20 = paddle._C_ops.matmul(transpose_71, reshape_104, False, False)
        del reshape_104, transpose_71

        # pd_op.scale: (1x12x21x21xf32) <- (1x12x21x21xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(matmul_20, full_3, float("0"), True)
        del matmul_20

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_95 = paddle._C_ops.add(scale_12, scale_1)
        del scale_12

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_10 = paddle._C_ops.softmax(add_95, -1)
        del add_95

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_62, dropout_63 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_10, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_10

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_21 = paddle._C_ops.matmul(dropout_62, transpose_72, False, False)
        del dropout_62, transpose_72

        # pd_op.transpose: (1x12x64x21xf32) <- (1x12x21x64xf32)
        transpose_73 = paddle._C_ops.transpose(matmul_21, [0, 1, 3, 2])
        del matmul_21

        # pd_op.reshape: (1x768x21xf32) <- (1x12x64x21xf32, 3xi64)
        reshape_106 = paddle._C_ops.reshape(transpose_73, full_int_array_6)
        del transpose_73

        # pd_op.assign: (768x768x1xf32) <- (768x768x1xf32)
        assign_63 = parameter_28
        del parameter_28

        # pd_op.unsqueeze: (768x768x1x1xf32) <- (768x768x1xf32, 1xi64)
        unsqueeze_128 = paddle._C_ops.unsqueeze(assign_63, full_int_array_3)
        del assign_63

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_129 = paddle._C_ops.unsqueeze(reshape_106, full_int_array_3)
        del reshape_106

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x768x1x1xf32)
        conv2d_63 = paddle._C_ops.conv2d(
            unsqueeze_129, unsqueeze_128, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del unsqueeze_128, unsqueeze_129

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_107 = paddle._C_ops.reshape(parameter_27, full_int_array_4)
        del parameter_27

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_96 = paddle._C_ops.add(conv2d_63, reshape_107)
        del conv2d_63, reshape_107

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_63 = paddle._C_ops.squeeze(add_96, full_int_array_3)
        del add_96

        # pd_op.dropout: (1x768x21xf32, 1x768x21xui8) <- (1x768x21xf32, None, 1xf32)
        dropout_64, dropout_65 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                squeeze_63, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del squeeze_63

        # pd_op.add: (1x768x21xf32) <- (1x768x21xf32, 1x768x21xf32)
        add_97 = paddle._C_ops.add(dropout_64, transpose_70)
        del dropout_64, transpose_70

        # pd_op.transpose: (1x21x768xf32) <- (1x768x21xf32)
        transpose_74 = paddle._C_ops.transpose(add_97, [0, 2, 1])
        del add_97

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_74, parameter_26, parameter_25, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_25, parameter_26, transpose_74

        # pd_op.transpose: (1x768x21xf32) <- (1x21x768xf32)
        transpose_75 = paddle._C_ops.transpose(layer_norm_63, [0, 2, 1])
        del layer_norm_63

        # pd_op.assign: (3072x192x1xf32) <- (3072x192x1xf32)
        assign_64 = parameter_24
        del parameter_24

        # pd_op.unsqueeze: (3072x192x1x1xf32) <- (3072x192x1xf32, 1xi64)
        unsqueeze_130 = paddle._C_ops.unsqueeze(assign_64, full_int_array_3)
        del assign_64

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_131 = paddle._C_ops.unsqueeze(transpose_75, full_int_array_3)

        # pd_op.conv2d: (1x3072x1x21xf32) <- (1x768x1x21xf32, 3072x192x1x1xf32)
        conv2d_64 = paddle._C_ops.conv2d(
            unsqueeze_131, unsqueeze_130, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_130, unsqueeze_131

        # pd_op.reshape: (1x3072x1x1xf32) <- (3072xf32, 4xi64)
        reshape_108 = paddle._C_ops.reshape(parameter_23, full_int_array_7)
        del parameter_23

        # pd_op.add: (1x3072x1x21xf32) <- (1x3072x1x21xf32, 1x3072x1x1xf32)
        add_98 = paddle._C_ops.add(conv2d_64, reshape_108)
        del conv2d_64, reshape_108

        # pd_op.squeeze: (1x3072x21xf32) <- (1x3072x1x21xf32, 1xi64)
        squeeze_64 = paddle._C_ops.squeeze(add_98, full_int_array_3)
        del add_98

        # pd_op.gelu: (1x3072x21xf32) <- (1x3072x21xf32)
        gelu_10 = paddle._C_ops.gelu(squeeze_64, False)
        del squeeze_64

        # pd_op.assign: (768x768x1xf32) <- (768x768x1xf32)
        assign_65 = parameter_22
        del parameter_22

        # pd_op.unsqueeze: (768x768x1x1xf32) <- (768x768x1xf32, 1xi64)
        unsqueeze_132 = paddle._C_ops.unsqueeze(assign_65, full_int_array_3)
        del assign_65

        # pd_op.unsqueeze: (1x3072x1x21xf32) <- (1x3072x21xf32, 1xi64)
        unsqueeze_133 = paddle._C_ops.unsqueeze(gelu_10, full_int_array_3)
        del gelu_10

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x3072x1x21xf32, 768x768x1x1xf32)
        conv2d_65 = paddle._C_ops.conv2d(
            unsqueeze_133, unsqueeze_132, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_132, unsqueeze_133

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_109 = paddle._C_ops.reshape(parameter_21, full_int_array_4)
        del parameter_21

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_99 = paddle._C_ops.add(conv2d_65, reshape_109)
        del conv2d_65, reshape_109

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_65 = paddle._C_ops.squeeze(add_99, full_int_array_3)
        del add_99

        # pd_op.dropout: (1x768x21xf32, 1x768x21xui8) <- (1x768x21xf32, None, 1xf32)
        dropout_66, dropout_67 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                squeeze_65, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del squeeze_65

        # pd_op.add: (1x768x21xf32) <- (1x768x21xf32, 1x768x21xf32)
        add_100 = paddle._C_ops.add(dropout_66, transpose_75)
        del dropout_66, transpose_75

        # pd_op.transpose: (1x21x768xf32) <- (1x768x21xf32)
        transpose_76 = paddle._C_ops.transpose(add_100, [0, 2, 1])
        del add_100

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_76, parameter_20, parameter_19, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_19, parameter_20, transpose_76

        # pd_op.transpose: (1x768x21xf32) <- (1x21x768xf32)
        transpose_77 = paddle._C_ops.transpose(layer_norm_66, [0, 2, 1])
        del layer_norm_66

        # pd_op.assign: (768x192x1xf32) <- (768x192x1xf32)
        assign_66 = parameter_18
        del parameter_18

        # pd_op.unsqueeze: (768x192x1x1xf32) <- (768x192x1xf32, 1xi64)
        unsqueeze_134 = paddle._C_ops.unsqueeze(assign_66, full_int_array_3)
        del assign_66

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_135 = paddle._C_ops.unsqueeze(transpose_77, full_int_array_3)

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x192x1x1xf32)
        conv2d_66 = paddle._C_ops.conv2d(
            unsqueeze_135, unsqueeze_134, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_134, unsqueeze_135

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_110 = paddle._C_ops.reshape(parameter_17, full_int_array_4)
        del parameter_17

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_101 = paddle._C_ops.add(conv2d_66, reshape_110)
        del conv2d_66, reshape_110

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_66 = paddle._C_ops.squeeze(add_101, full_int_array_3)
        del add_101

        # pd_op.assign: (768x192x1xf32) <- (768x192x1xf32)
        assign_67 = parameter_16
        del parameter_16

        # pd_op.unsqueeze: (768x192x1x1xf32) <- (768x192x1xf32, 1xi64)
        unsqueeze_136 = paddle._C_ops.unsqueeze(assign_67, full_int_array_3)
        del assign_67

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_137 = paddle._C_ops.unsqueeze(transpose_77, full_int_array_3)

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x192x1x1xf32)
        conv2d_67 = paddle._C_ops.conv2d(
            unsqueeze_137, unsqueeze_136, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_136, unsqueeze_137

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_111 = paddle._C_ops.reshape(parameter_15, full_int_array_4)
        del parameter_15

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_102 = paddle._C_ops.add(conv2d_67, reshape_111)
        del conv2d_67, reshape_111

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_67 = paddle._C_ops.squeeze(add_102, full_int_array_3)
        del add_102

        # pd_op.assign: (768x192x1xf32) <- (768x192x1xf32)
        assign_68 = parameter_14
        del parameter_14

        # pd_op.unsqueeze: (768x192x1x1xf32) <- (768x192x1xf32, 1xi64)
        unsqueeze_138 = paddle._C_ops.unsqueeze(assign_68, full_int_array_3)
        del assign_68

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_139 = paddle._C_ops.unsqueeze(transpose_77, full_int_array_3)

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x192x1x1xf32)
        conv2d_68 = paddle._C_ops.conv2d(
            unsqueeze_139, unsqueeze_138, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_138, unsqueeze_139

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_112 = paddle._C_ops.reshape(parameter_13, full_int_array_4)
        del parameter_13

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_103 = paddle._C_ops.add(conv2d_68, reshape_112)
        del conv2d_68, reshape_112

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_68 = paddle._C_ops.squeeze(add_103, full_int_array_3)
        del add_103

        # pd_op.reshape: (1x12x64x21xf32) <- (1x768x21xf32, 4xi64)
        reshape_113 = paddle._C_ops.reshape(squeeze_66, full_int_array_5)
        del squeeze_66

        # pd_op.transpose: (1x12x21x64xf32) <- (1x12x64x21xf32)
        transpose_78 = paddle._C_ops.transpose(reshape_113, [0, 1, 3, 2])
        del reshape_113

        # pd_op.reshape: (1x12x64x21xf32) <- (1x768x21xf32, 4xi64)
        reshape_114 = paddle._C_ops.reshape(squeeze_67, full_int_array_5)
        del squeeze_67

        # pd_op.reshape: (1x12x64x21xf32) <- (1x768x21xf32, 4xi64)
        reshape_115 = paddle._C_ops.reshape(squeeze_68, full_int_array_5)
        del full_int_array_5, squeeze_68

        # pd_op.transpose: (1x12x21x64xf32) <- (1x12x64x21xf32)
        transpose_79 = paddle._C_ops.transpose(reshape_115, [0, 1, 3, 2])
        del reshape_115

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x64x21xf32)
        matmul_22 = paddle._C_ops.matmul(transpose_78, reshape_114, False, False)
        del reshape_114, transpose_78

        # pd_op.scale: (1x12x21x21xf32) <- (1x12x21x21xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(matmul_22, full_3, float("0"), True)
        del full_3, matmul_22

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_104 = paddle._C_ops.add(scale_13, scale_1)
        del scale_1, scale_13

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_11 = paddle._C_ops.softmax(add_104, -1)
        del add_104

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_68, dropout_69 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_11, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_11

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_23 = paddle._C_ops.matmul(dropout_68, transpose_79, False, False)
        del dropout_68, transpose_79

        # pd_op.transpose: (1x12x64x21xf32) <- (1x12x21x64xf32)
        transpose_80 = paddle._C_ops.transpose(matmul_23, [0, 1, 3, 2])
        del matmul_23

        # pd_op.reshape: (1x768x21xf32) <- (1x12x64x21xf32, 3xi64)
        reshape_116 = paddle._C_ops.reshape(transpose_80, full_int_array_6)
        del full_int_array_6, transpose_80

        # pd_op.assign: (768x768x1xf32) <- (768x768x1xf32)
        assign_69 = parameter_12
        del parameter_12

        # pd_op.unsqueeze: (768x768x1x1xf32) <- (768x768x1xf32, 1xi64)
        unsqueeze_140 = paddle._C_ops.unsqueeze(assign_69, full_int_array_3)
        del assign_69

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_141 = paddle._C_ops.unsqueeze(reshape_116, full_int_array_3)
        del reshape_116

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x768x1x21xf32, 768x768x1x1xf32)
        conv2d_69 = paddle._C_ops.conv2d(
            unsqueeze_141, unsqueeze_140, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del unsqueeze_140, unsqueeze_141

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_117 = paddle._C_ops.reshape(parameter_11, full_int_array_4)
        del parameter_11

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_105 = paddle._C_ops.add(conv2d_69, reshape_117)
        del conv2d_69, reshape_117

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_69 = paddle._C_ops.squeeze(add_105, full_int_array_3)
        del add_105

        # pd_op.dropout: (1x768x21xf32, 1x768x21xui8) <- (1x768x21xf32, None, 1xf32)
        dropout_70, dropout_71 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                squeeze_69, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del squeeze_69

        # pd_op.add: (1x768x21xf32) <- (1x768x21xf32, 1x768x21xf32)
        add_106 = paddle._C_ops.add(dropout_70, transpose_77)
        del dropout_70, transpose_77

        # pd_op.transpose: (1x21x768xf32) <- (1x768x21xf32)
        transpose_81 = paddle._C_ops.transpose(add_106, [0, 2, 1])
        del add_106

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_81, parameter_10, parameter_9, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_10, parameter_9, transpose_81

        # pd_op.transpose: (1x768x21xf32) <- (1x21x768xf32)
        transpose_82 = paddle._C_ops.transpose(layer_norm_69, [0, 2, 1])
        del layer_norm_69

        # pd_op.assign: (3072x192x1xf32) <- (3072x192x1xf32)
        assign_70 = parameter_8
        del parameter_8

        # pd_op.unsqueeze: (3072x192x1x1xf32) <- (3072x192x1xf32, 1xi64)
        unsqueeze_142 = paddle._C_ops.unsqueeze(assign_70, full_int_array_3)
        del assign_70

        # pd_op.unsqueeze: (1x768x1x21xf32) <- (1x768x21xf32, 1xi64)
        unsqueeze_143 = paddle._C_ops.unsqueeze(transpose_82, full_int_array_3)

        # pd_op.conv2d: (1x3072x1x21xf32) <- (1x768x1x21xf32, 3072x192x1x1xf32)
        conv2d_70 = paddle._C_ops.conv2d(
            unsqueeze_143, unsqueeze_142, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_142, unsqueeze_143

        # pd_op.reshape: (1x3072x1x1xf32) <- (3072xf32, 4xi64)
        reshape_118 = paddle._C_ops.reshape(parameter_7, full_int_array_7)
        del full_int_array_7, parameter_7

        # pd_op.add: (1x3072x1x21xf32) <- (1x3072x1x21xf32, 1x3072x1x1xf32)
        add_107 = paddle._C_ops.add(conv2d_70, reshape_118)
        del conv2d_70, reshape_118

        # pd_op.squeeze: (1x3072x21xf32) <- (1x3072x1x21xf32, 1xi64)
        squeeze_70 = paddle._C_ops.squeeze(add_107, full_int_array_3)
        del add_107

        # pd_op.gelu: (1x3072x21xf32) <- (1x3072x21xf32)
        gelu_11 = paddle._C_ops.gelu(squeeze_70, False)
        del squeeze_70

        # pd_op.assign: (768x768x1xf32) <- (768x768x1xf32)
        assign_71 = parameter_6
        del parameter_6

        # pd_op.unsqueeze: (768x768x1x1xf32) <- (768x768x1xf32, 1xi64)
        unsqueeze_144 = paddle._C_ops.unsqueeze(assign_71, full_int_array_3)
        del assign_71

        # pd_op.unsqueeze: (1x3072x1x21xf32) <- (1x3072x21xf32, 1xi64)
        unsqueeze_145 = paddle._C_ops.unsqueeze(gelu_11, full_int_array_3)
        del gelu_11

        # pd_op.conv2d: (1x768x1x21xf32) <- (1x3072x1x21xf32, 768x768x1x1xf32)
        conv2d_71 = paddle._C_ops.conv2d(
            unsqueeze_145, unsqueeze_144, [1, 1], [0, 0], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del unsqueeze_144, unsqueeze_145

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_119 = paddle._C_ops.reshape(parameter_5, full_int_array_4)
        del full_int_array_4, parameter_5

        # pd_op.add: (1x768x1x21xf32) <- (1x768x1x21xf32, 1x768x1x1xf32)
        add_108 = paddle._C_ops.add(conv2d_71, reshape_119)
        del conv2d_71, reshape_119

        # pd_op.squeeze: (1x768x21xf32) <- (1x768x1x21xf32, 1xi64)
        squeeze_71 = paddle._C_ops.squeeze(add_108, full_int_array_3)
        del add_108, full_int_array_3

        # pd_op.dropout: (1x768x21xf32, 1x768x21xui8) <- (1x768x21xf32, None, 1xf32)
        dropout_72, dropout_73 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                squeeze_71, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del full_0, squeeze_71

        # pd_op.add: (1x768x21xf32) <- (1x768x21xf32, 1x768x21xf32)
        add_109 = paddle._C_ops.add(dropout_72, transpose_82)
        del dropout_72, transpose_82

        # pd_op.transpose: (1x21x768xf32) <- (1x768x21xf32)
        transpose_83 = paddle._C_ops.transpose(add_109, [0, 2, 1])
        del add_109

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_72, layer_norm_73, layer_norm_74 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_83, parameter_4, parameter_3, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_3, parameter_4, transpose_83

        # pd_op.transpose: (1x768x21xf32) <- (1x21x768xf32)
        transpose_84 = paddle._C_ops.transpose(layer_norm_72, [0, 2, 1])
        del layer_norm_72

        # pd_op.transpose: (1x21x768xf32) <- (1x768x21xf32)
        transpose_85 = paddle._C_ops.transpose(transpose_84, [0, 2, 1])
        del transpose_84

        # pd_op.slice: (1x768xf32) <- (1x21x768xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            transpose_85, [1], full_int_array_0, full_int_array_2, [1], [1]
        )
        del full_int_array_0, full_int_array_2

        # pd_op.matmul: (1x768xf32) <- (1x768xf32, 768x768xf32)
        matmul_24 = paddle._C_ops.matmul(slice_1, parameter_2, False, False)
        del parameter_2, slice_1

        # pd_op.add: (1x768xf32) <- (1x768xf32, 768xf32)
        add_110 = paddle._C_ops.add(matmul_24, parameter_1)
        del matmul_24, parameter_1

        # pd_op.tanh: (1x768xf32) <- (1x768xf32)
        tanh_0 = paddle._C_ops.tanh(add_110)
        del add_110, transpose_85

        return tanh_0
