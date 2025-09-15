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
        data_0,
        data_1,
    ):
        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (1x11xi64) <- (1x11xi64, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            data_0, full_0, paddle.int64, paddle.framework._current_expected_place()
        )
        del full_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_0

        # pd_op.unsqueeze: (1x1x11xi64) <- (1x11xi64, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(full_like_0, full_int_array_0)
        del full_like_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [2]

        # pd_op.unsqueeze: (1x1x1x11xi64) <- (1x1x11xi64, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(unsqueeze_0, full_int_array_1)
        del full_int_array_1, unsqueeze_0

        # pd_op.cast: (1x1x1x11xf32) <- (1x1x1x11xi64)
        cast_0 = paddle._C_ops.cast(unsqueeze_1, paddle.float32)
        del unsqueeze_1

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x1x1x11xf32) <- (1x1x1x11xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(cast_0, full_1, float("1"), True)
        del cast_0, full_1

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("-10000"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x1x1x11xf32) <- (1x1x1x11xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(scale_0, full_2, float("0"), True)
        del full_2, scale_0

        # pd_op.embedding: (1x11x768xf32) <- (1x11xi64, 21128x768xf32)
        embedding_0 = paddle._C_ops.embedding(data_0, parameter_209, -1, False)
        del data_0, parameter_209

        # pd_op.full: (1x11xi64) <- ()
        full_3 = paddle._C_ops.full(
            [1, 11],
            float("1"),
            paddle.int64,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.cumsum: (1x11xi64) <- (1x11xi64, 1xi32)
        cumsum_0 = paddle._C_ops.cumsum(full_3, full_4, False, False, False)
        del full_4

        # pd_op.subtract: (1x11xi64) <- (1x11xi64, 1x11xi64)
        subtract_0 = paddle._C_ops.subtract(cumsum_0, full_3)
        del cumsum_0, full_3

        # pd_op.embedding: (1x11x768xf32) <- (1x11xi64, 2x768xf32)
        embedding_1 = paddle._C_ops.embedding(data_1, parameter_208, -1, False)
        del data_1, parameter_208

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_0 = paddle._C_ops.add(embedding_0, embedding_1)

        # pd_op.layer_norm: (1x11x768xf32, 1x11xf32, 1x11xf32) <- (1x11x768xf32, 768xf32, 768xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_0, parameter_207, parameter_206, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_206, parameter_207

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_1 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_2 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_3 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_4 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_5 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_6 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_7 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_8 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_9 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_10 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_11 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_12 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_13 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_14 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_15 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_16 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_17 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_18 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_19 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_20 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_21 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_22 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_23 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_24 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_25 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_26 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_27 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_28 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_29 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_30 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_31 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_32 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_33 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_34 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_35 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_36 = full_5

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                layer_norm_0, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del layer_norm_0

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_0 = paddle._C_ops.matmul(dropout_0, parameter_203, False, False)
        del parameter_203

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_1 = paddle._C_ops.add(matmul_0, parameter_202)
        del parameter_202

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_1 = paddle._C_ops.matmul(dropout_0, parameter_201, False, False)
        del parameter_201

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_2 = paddle._C_ops.add(matmul_1, parameter_200)
        del parameter_200

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_2 = paddle._C_ops.matmul(dropout_0, parameter_199, False, False)
        del parameter_199

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_3 = paddle._C_ops.add(matmul_2, parameter_198)
        del parameter_198

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_2 = [1, 11, 12, 64]

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(add_1, full_int_array_2)

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_0, [0, 2, 1, 3])
        del reshape_0

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(add_2, full_int_array_2)

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_1, [0, 2, 1, 3])
        del reshape_1

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(add_3, full_int_array_2)

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_2 = paddle._C_ops.transpose(reshape_2, [0, 2, 1, 3])
        del reshape_2

        # pd_op.transpose: (1x12x64x11xf32) <- (1x12x11x64xf32)
        transpose_3 = paddle._C_ops.transpose(transpose_1, [0, 1, 3, 2])
        del transpose_1

        # pd_op.matmul: (1x12x11x11xf32) <- (1x12x11x64xf32, 1x12x64x11xf32)
        matmul_3 = paddle._C_ops.matmul(transpose_0, transpose_3, False, False)

        # pd_op.share_data_: (512x512x64xf32) <- (512x512x64xf32)
        share_data__0 = parameter_11.detach()

        # pd_op.assign: (512x512x64xf32) <- (512x512x64xf32)
        assign_37 = share_data__0
        del share_data__0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [0, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [11, 11]

        # pd_op.slice: (11x11x64xf32) <- (512x512x64xf32, 2xi64, 2xi64)
        slice_0 = paddle._C_ops.slice(
            assign_37, [0, 1], full_int_array_3, full_int_array_4, [1, 1], []
        )
        del assign_37

        # pd_op.transpose: (11x1x12x64xf32) <- (1x12x11x64xf32)
        transpose_4 = paddle._C_ops.transpose(transpose_0, [2, 0, 1, 3])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_5 = [11, 12, 64]

        # pd_op.reshape: (11x12x64xf32) <- (11x1x12x64xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_4, full_int_array_5)

        # pd_op.transpose: (11x64x11xf32) <- (11x11x64xf32)
        transpose_5 = paddle._C_ops.transpose(slice_0, [0, 2, 1])
        del slice_0

        # pd_op.matmul: (11x12x11xf32) <- (11x12x64xf32, 11x64x11xf32)
        matmul_4 = paddle._C_ops.matmul(reshape_3, transpose_5, False, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_6 = [11, 1, 12, 11]

        # pd_op.reshape: (11x1x12x11xf32) <- (11x12x11xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(matmul_4, full_int_array_6)

        # pd_op.transpose: (1x12x11x11xf32) <- (11x1x12x11xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_4, [1, 2, 0, 3])
        del reshape_4

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x12x11x11xf32)
        add_4 = paddle._C_ops.add(matmul_3, transpose_6)

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_38 = full_6

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_39 = full_6

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_40 = full_6

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_41 = full_6

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_42 = full_6

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_43 = full_6

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_44 = full_6

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_45 = full_6

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_46 = full_6

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_47 = full_6

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_48 = full_6

        # pd_op.scale: (1x12x11x11xf32) <- (1x12x11x11xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(add_4, full_6, float("0"), True)
        del add_4

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x1x1x11xf32)
        add_5 = paddle._C_ops.add(scale_2, scale_1)

        # pd_op.softmax: (1x12x11x11xf32) <- (1x12x11x11xf32)
        softmax_0 = paddle._C_ops.softmax(add_5, -1)
        del add_5

        # pd_op.dropout: (1x12x11x11xf32, 1x12x11x11xui8) <- (1x12x11x11xf32, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_0, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x11x64xf32) <- (1x12x11x11xf32, 1x12x11x64xf32)
        matmul_5 = paddle._C_ops.matmul(dropout_2, transpose_2, False, False)

        # pd_op.assign: (512x512x64xf32) <- (512x512x64xf32)
        assign_49 = parameter_11
        del parameter_11

        # pd_op.slice: (11x11x64xf32) <- (512x512x64xf32, 2xi64, 2xi64)
        slice_1 = paddle._C_ops.slice(
            assign_49, [0, 1], full_int_array_3, full_int_array_4, [1, 1], []
        )
        del assign_49

        # pd_op.transpose: (11x1x12x11xf32) <- (1x12x11x11xf32)
        transpose_7 = paddle._C_ops.transpose(dropout_2, [2, 0, 1, 3])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_7 = [11, 12, 11]

        # pd_op.reshape: (11x12x11xf32) <- (11x1x12x11xf32, 3xi64)
        reshape_5 = paddle._C_ops.reshape(transpose_7, full_int_array_7)

        # pd_op.matmul: (11x12x64xf32) <- (11x12x11xf32, 11x11x64xf32)
        matmul_6 = paddle._C_ops.matmul(reshape_5, slice_1, False, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_8 = [11, 1, 12, 64]

        # pd_op.reshape: (11x1x12x64xf32) <- (11x12x64xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(matmul_6, full_int_array_8)

        # pd_op.transpose: (1x12x11x64xf32) <- (11x1x12x64xf32)
        transpose_8 = paddle._C_ops.transpose(reshape_6, [1, 2, 0, 3])
        del reshape_6

        # pd_op.add: (1x12x11x64xf32) <- (1x12x11x64xf32, 1x12x11x64xf32)
        add_6 = paddle._C_ops.add(matmul_5, transpose_8)

        # pd_op.transpose: (1x11x12x64xf32) <- (1x12x11x64xf32)
        transpose_9 = paddle._C_ops.transpose(add_6, [0, 2, 1, 3])
        del add_6

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_9 = [1, 11, 768]

        # pd_op.reshape: (1x11x768xf32) <- (1x11x12x64xf32, 3xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_9, full_int_array_9)

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_7 = paddle._C_ops.matmul(reshape_7, parameter_197, False, False)
        del parameter_197

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_7 = paddle._C_ops.add(matmul_7, parameter_196)
        del parameter_196

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_7, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_7

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_8 = paddle._C_ops.add(dropout_0, dropout_4)

        # pd_op.layer_norm: (1x11x768xf32, 1x11xf32, 1x11xf32) <- (1x11x768xf32, 768xf32, 768xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_8, parameter_195, parameter_194, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_194, parameter_195

        # pd_op.matmul: (1x11x3072xf32) <- (1x11x768xf32, 768x3072xf32)
        matmul_8 = paddle._C_ops.matmul(layer_norm_3, parameter_193, False, False)
        del parameter_193

        # pd_op.add: (1x11x3072xf32) <- (1x11x3072xf32, 3072xf32)
        add_9 = paddle._C_ops.add(matmul_8, parameter_192)
        del parameter_192

        # pd_op.gelu: (1x11x3072xf32) <- (1x11x3072xf32)
        gelu_0 = paddle._C_ops.gelu(add_9, False)

        # pd_op.matmul: (1x11x768xf32) <- (1x11x3072xf32, 3072x768xf32)
        matmul_9 = paddle._C_ops.matmul(gelu_0, parameter_191, False, False)
        del parameter_191

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_10 = paddle._C_ops.add(matmul_9, parameter_190)
        del parameter_190

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_6, dropout_7 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_10, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_10

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_11 = paddle._C_ops.add(dropout_6, layer_norm_3)

        # pd_op.layer_norm: (1x11x768xf32, 1x11xf32, 1x11xf32) <- (1x11x768xf32, 768xf32, 768xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_11, parameter_205, parameter_204, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_204, parameter_205

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_6, parameter_187, False, False)
        del parameter_187

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_12 = paddle._C_ops.add(matmul_10, parameter_186)
        del parameter_186

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_11 = paddle._C_ops.matmul(layer_norm_6, parameter_185, False, False)
        del parameter_185

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_13 = paddle._C_ops.add(matmul_11, parameter_184)
        del parameter_184

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_12 = paddle._C_ops.matmul(layer_norm_6, parameter_183, False, False)
        del parameter_183

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_14 = paddle._C_ops.add(matmul_12, parameter_182)
        del parameter_182

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(add_12, full_int_array_2)

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_10 = paddle._C_ops.transpose(reshape_8, [0, 2, 1, 3])
        del reshape_8

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(add_13, full_int_array_2)

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_11 = paddle._C_ops.transpose(reshape_9, [0, 2, 1, 3])
        del reshape_9

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(add_14, full_int_array_2)

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_12 = paddle._C_ops.transpose(reshape_10, [0, 2, 1, 3])
        del reshape_10

        # pd_op.transpose: (1x12x64x11xf32) <- (1x12x11x64xf32)
        transpose_13 = paddle._C_ops.transpose(transpose_11, [0, 1, 3, 2])
        del transpose_11

        # pd_op.matmul: (1x12x11x11xf32) <- (1x12x11x64xf32, 1x12x64x11xf32)
        matmul_13 = paddle._C_ops.matmul(transpose_10, transpose_13, False, False)

        # pd_op.share_data_: (512x512x64xf32) <- (512x512x64xf32)
        share_data__1 = parameter_10.detach()

        # pd_op.assign: (512x512x64xf32) <- (512x512x64xf32)
        assign_50 = share_data__1
        del share_data__1

        # pd_op.slice: (11x11x64xf32) <- (512x512x64xf32, 2xi64, 2xi64)
        slice_2 = paddle._C_ops.slice(
            assign_50, [0, 1], full_int_array_3, full_int_array_4, [1, 1], []
        )
        del assign_50

        # pd_op.transpose: (11x1x12x64xf32) <- (1x12x11x64xf32)
        transpose_14 = paddle._C_ops.transpose(transpose_10, [2, 0, 1, 3])

        # pd_op.reshape: (11x12x64xf32) <- (11x1x12x64xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_14, full_int_array_5)

        # pd_op.transpose: (11x64x11xf32) <- (11x11x64xf32)
        transpose_15 = paddle._C_ops.transpose(slice_2, [0, 2, 1])
        del slice_2

        # pd_op.matmul: (11x12x11xf32) <- (11x12x64xf32, 11x64x11xf32)
        matmul_14 = paddle._C_ops.matmul(reshape_11, transpose_15, False, False)

        # pd_op.reshape: (11x1x12x11xf32) <- (11x12x11xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(matmul_14, full_int_array_6)

        # pd_op.transpose: (1x12x11x11xf32) <- (11x1x12x11xf32)
        transpose_16 = paddle._C_ops.transpose(reshape_12, [1, 2, 0, 3])
        del reshape_12

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x12x11x11xf32)
        add_15 = paddle._C_ops.add(matmul_13, transpose_16)

        # pd_op.scale: (1x12x11x11xf32) <- (1x12x11x11xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(add_15, full_6, float("0"), True)
        del add_15

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x1x1x11xf32)
        add_16 = paddle._C_ops.add(scale_3, scale_1)

        # pd_op.softmax: (1x12x11x11xf32) <- (1x12x11x11xf32)
        softmax_1 = paddle._C_ops.softmax(add_16, -1)
        del add_16

        # pd_op.dropout: (1x12x11x11xf32, 1x12x11x11xui8) <- (1x12x11x11xf32, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_1, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x11x64xf32) <- (1x12x11x11xf32, 1x12x11x64xf32)
        matmul_15 = paddle._C_ops.matmul(dropout_8, transpose_12, False, False)

        # pd_op.assign: (512x512x64xf32) <- (512x512x64xf32)
        assign_51 = parameter_10
        del parameter_10

        # pd_op.slice: (11x11x64xf32) <- (512x512x64xf32, 2xi64, 2xi64)
        slice_3 = paddle._C_ops.slice(
            assign_51, [0, 1], full_int_array_3, full_int_array_4, [1, 1], []
        )
        del assign_51

        # pd_op.transpose: (11x1x12x11xf32) <- (1x12x11x11xf32)
        transpose_17 = paddle._C_ops.transpose(dropout_8, [2, 0, 1, 3])

        # pd_op.reshape: (11x12x11xf32) <- (11x1x12x11xf32, 3xi64)
        reshape_13 = paddle._C_ops.reshape(transpose_17, full_int_array_7)

        # pd_op.matmul: (11x12x64xf32) <- (11x12x11xf32, 11x11x64xf32)
        matmul_16 = paddle._C_ops.matmul(reshape_13, slice_3, False, False)

        # pd_op.reshape: (11x1x12x64xf32) <- (11x12x64xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(matmul_16, full_int_array_8)

        # pd_op.transpose: (1x12x11x64xf32) <- (11x1x12x64xf32)
        transpose_18 = paddle._C_ops.transpose(reshape_14, [1, 2, 0, 3])
        del reshape_14

        # pd_op.add: (1x12x11x64xf32) <- (1x12x11x64xf32, 1x12x11x64xf32)
        add_17 = paddle._C_ops.add(matmul_15, transpose_18)

        # pd_op.transpose: (1x11x12x64xf32) <- (1x12x11x64xf32)
        transpose_19 = paddle._C_ops.transpose(add_17, [0, 2, 1, 3])
        del add_17

        # pd_op.reshape: (1x11x768xf32) <- (1x11x12x64xf32, 3xi64)
        reshape_15 = paddle._C_ops.reshape(transpose_19, full_int_array_9)

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_17 = paddle._C_ops.matmul(reshape_15, parameter_181, False, False)
        del parameter_181

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_18 = paddle._C_ops.add(matmul_17, parameter_180)
        del parameter_180

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_10, dropout_11 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_18, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_18

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_19 = paddle._C_ops.add(layer_norm_6, dropout_10)

        # pd_op.layer_norm: (1x11x768xf32, 1x11xf32, 1x11xf32) <- (1x11x768xf32, 768xf32, 768xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_19, parameter_179, parameter_178, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_178, parameter_179

        # pd_op.matmul: (1x11x3072xf32) <- (1x11x768xf32, 768x3072xf32)
        matmul_18 = paddle._C_ops.matmul(layer_norm_9, parameter_177, False, False)
        del parameter_177

        # pd_op.add: (1x11x3072xf32) <- (1x11x3072xf32, 3072xf32)
        add_20 = paddle._C_ops.add(matmul_18, parameter_176)
        del parameter_176

        # pd_op.gelu: (1x11x3072xf32) <- (1x11x3072xf32)
        gelu_1 = paddle._C_ops.gelu(add_20, False)

        # pd_op.matmul: (1x11x768xf32) <- (1x11x3072xf32, 3072x768xf32)
        matmul_19 = paddle._C_ops.matmul(gelu_1, parameter_175, False, False)
        del parameter_175

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_21 = paddle._C_ops.add(matmul_19, parameter_174)
        del parameter_174

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_12, dropout_13 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_21, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_21

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_22 = paddle._C_ops.add(dropout_12, layer_norm_9)

        # pd_op.layer_norm: (1x11x768xf32, 1x11xf32, 1x11xf32) <- (1x11x768xf32, 768xf32, 768xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_22, parameter_189, parameter_188, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_188, parameter_189

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_20 = paddle._C_ops.matmul(layer_norm_12, parameter_171, False, False)
        del parameter_171

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_23 = paddle._C_ops.add(matmul_20, parameter_170)
        del parameter_170

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_21 = paddle._C_ops.matmul(layer_norm_12, parameter_169, False, False)
        del parameter_169

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_24 = paddle._C_ops.add(matmul_21, parameter_168)
        del parameter_168

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_22 = paddle._C_ops.matmul(layer_norm_12, parameter_167, False, False)
        del parameter_167

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_25 = paddle._C_ops.add(matmul_22, parameter_166)
        del parameter_166

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_16 = paddle._C_ops.reshape(add_23, full_int_array_2)

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_20 = paddle._C_ops.transpose(reshape_16, [0, 2, 1, 3])
        del reshape_16

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_17 = paddle._C_ops.reshape(add_24, full_int_array_2)

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_21 = paddle._C_ops.transpose(reshape_17, [0, 2, 1, 3])
        del reshape_17

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_18 = paddle._C_ops.reshape(add_25, full_int_array_2)

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_22 = paddle._C_ops.transpose(reshape_18, [0, 2, 1, 3])
        del reshape_18

        # pd_op.transpose: (1x12x64x11xf32) <- (1x12x11x64xf32)
        transpose_23 = paddle._C_ops.transpose(transpose_21, [0, 1, 3, 2])
        del transpose_21

        # pd_op.matmul: (1x12x11x11xf32) <- (1x12x11x64xf32, 1x12x64x11xf32)
        matmul_23 = paddle._C_ops.matmul(transpose_20, transpose_23, False, False)

        # pd_op.share_data_: (512x512x64xf32) <- (512x512x64xf32)
        share_data__2 = parameter_9.detach()

        # pd_op.assign: (512x512x64xf32) <- (512x512x64xf32)
        assign_52 = share_data__2
        del share_data__2

        # pd_op.slice: (11x11x64xf32) <- (512x512x64xf32, 2xi64, 2xi64)
        slice_4 = paddle._C_ops.slice(
            assign_52, [0, 1], full_int_array_3, full_int_array_4, [1, 1], []
        )
        del assign_52

        # pd_op.transpose: (11x1x12x64xf32) <- (1x12x11x64xf32)
        transpose_24 = paddle._C_ops.transpose(transpose_20, [2, 0, 1, 3])

        # pd_op.reshape: (11x12x64xf32) <- (11x1x12x64xf32, 3xi64)
        reshape_19 = paddle._C_ops.reshape(transpose_24, full_int_array_5)

        # pd_op.transpose: (11x64x11xf32) <- (11x11x64xf32)
        transpose_25 = paddle._C_ops.transpose(slice_4, [0, 2, 1])
        del slice_4

        # pd_op.matmul: (11x12x11xf32) <- (11x12x64xf32, 11x64x11xf32)
        matmul_24 = paddle._C_ops.matmul(reshape_19, transpose_25, False, False)

        # pd_op.reshape: (11x1x12x11xf32) <- (11x12x11xf32, 4xi64)
        reshape_20 = paddle._C_ops.reshape(matmul_24, full_int_array_6)

        # pd_op.transpose: (1x12x11x11xf32) <- (11x1x12x11xf32)
        transpose_26 = paddle._C_ops.transpose(reshape_20, [1, 2, 0, 3])
        del reshape_20

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x12x11x11xf32)
        add_26 = paddle._C_ops.add(matmul_23, transpose_26)

        # pd_op.scale: (1x12x11x11xf32) <- (1x12x11x11xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(add_26, full_6, float("0"), True)
        del add_26

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x1x1x11xf32)
        add_27 = paddle._C_ops.add(scale_4, scale_1)

        # pd_op.softmax: (1x12x11x11xf32) <- (1x12x11x11xf32)
        softmax_2 = paddle._C_ops.softmax(add_27, -1)
        del add_27

        # pd_op.dropout: (1x12x11x11xf32, 1x12x11x11xui8) <- (1x12x11x11xf32, None, 1xf32)
        dropout_14, dropout_15 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_2, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x11x64xf32) <- (1x12x11x11xf32, 1x12x11x64xf32)
        matmul_25 = paddle._C_ops.matmul(dropout_14, transpose_22, False, False)

        # pd_op.assign: (512x512x64xf32) <- (512x512x64xf32)
        assign_53 = parameter_9
        del parameter_9

        # pd_op.slice: (11x11x64xf32) <- (512x512x64xf32, 2xi64, 2xi64)
        slice_5 = paddle._C_ops.slice(
            assign_53, [0, 1], full_int_array_3, full_int_array_4, [1, 1], []
        )
        del assign_53

        # pd_op.transpose: (11x1x12x11xf32) <- (1x12x11x11xf32)
        transpose_27 = paddle._C_ops.transpose(dropout_14, [2, 0, 1, 3])

        # pd_op.reshape: (11x12x11xf32) <- (11x1x12x11xf32, 3xi64)
        reshape_21 = paddle._C_ops.reshape(transpose_27, full_int_array_7)

        # pd_op.matmul: (11x12x64xf32) <- (11x12x11xf32, 11x11x64xf32)
        matmul_26 = paddle._C_ops.matmul(reshape_21, slice_5, False, False)

        # pd_op.reshape: (11x1x12x64xf32) <- (11x12x64xf32, 4xi64)
        reshape_22 = paddle._C_ops.reshape(matmul_26, full_int_array_8)

        # pd_op.transpose: (1x12x11x64xf32) <- (11x1x12x64xf32)
        transpose_28 = paddle._C_ops.transpose(reshape_22, [1, 2, 0, 3])
        del reshape_22

        # pd_op.add: (1x12x11x64xf32) <- (1x12x11x64xf32, 1x12x11x64xf32)
        add_28 = paddle._C_ops.add(matmul_25, transpose_28)

        # pd_op.transpose: (1x11x12x64xf32) <- (1x12x11x64xf32)
        transpose_29 = paddle._C_ops.transpose(add_28, [0, 2, 1, 3])
        del add_28

        # pd_op.reshape: (1x11x768xf32) <- (1x11x12x64xf32, 3xi64)
        reshape_23 = paddle._C_ops.reshape(transpose_29, full_int_array_9)

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_27 = paddle._C_ops.matmul(reshape_23, parameter_165, False, False)
        del parameter_165

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_29 = paddle._C_ops.add(matmul_27, parameter_164)
        del parameter_164

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_16, dropout_17 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_29, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_29

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_30 = paddle._C_ops.add(layer_norm_12, dropout_16)

        # pd_op.layer_norm: (1x11x768xf32, 1x11xf32, 1x11xf32) <- (1x11x768xf32, 768xf32, 768xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_30, parameter_163, parameter_162, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_162, parameter_163

        # pd_op.matmul: (1x11x3072xf32) <- (1x11x768xf32, 768x3072xf32)
        matmul_28 = paddle._C_ops.matmul(layer_norm_15, parameter_161, False, False)
        del parameter_161

        # pd_op.add: (1x11x3072xf32) <- (1x11x3072xf32, 3072xf32)
        add_31 = paddle._C_ops.add(matmul_28, parameter_160)
        del parameter_160

        # pd_op.gelu: (1x11x3072xf32) <- (1x11x3072xf32)
        gelu_2 = paddle._C_ops.gelu(add_31, False)

        # pd_op.matmul: (1x11x768xf32) <- (1x11x3072xf32, 3072x768xf32)
        matmul_29 = paddle._C_ops.matmul(gelu_2, parameter_159, False, False)
        del parameter_159

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_32 = paddle._C_ops.add(matmul_29, parameter_158)
        del parameter_158

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_18, dropout_19 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_32, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_32

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_33 = paddle._C_ops.add(dropout_18, layer_norm_15)

        # pd_op.layer_norm: (1x11x768xf32, 1x11xf32, 1x11xf32) <- (1x11x768xf32, 768xf32, 768xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_33, parameter_173, parameter_172, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_172, parameter_173

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_30 = paddle._C_ops.matmul(layer_norm_18, parameter_155, False, False)
        del parameter_155

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_34 = paddle._C_ops.add(matmul_30, parameter_154)
        del parameter_154

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_31 = paddle._C_ops.matmul(layer_norm_18, parameter_153, False, False)
        del parameter_153

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_35 = paddle._C_ops.add(matmul_31, parameter_152)
        del parameter_152

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_32 = paddle._C_ops.matmul(layer_norm_18, parameter_151, False, False)
        del parameter_151

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_36 = paddle._C_ops.add(matmul_32, parameter_150)
        del parameter_150

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_24 = paddle._C_ops.reshape(add_34, full_int_array_2)

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_30 = paddle._C_ops.transpose(reshape_24, [0, 2, 1, 3])
        del reshape_24

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_25 = paddle._C_ops.reshape(add_35, full_int_array_2)

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_31 = paddle._C_ops.transpose(reshape_25, [0, 2, 1, 3])
        del reshape_25

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_26 = paddle._C_ops.reshape(add_36, full_int_array_2)

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_32 = paddle._C_ops.transpose(reshape_26, [0, 2, 1, 3])
        del reshape_26

        # pd_op.transpose: (1x12x64x11xf32) <- (1x12x11x64xf32)
        transpose_33 = paddle._C_ops.transpose(transpose_31, [0, 1, 3, 2])
        del transpose_31

        # pd_op.matmul: (1x12x11x11xf32) <- (1x12x11x64xf32, 1x12x64x11xf32)
        matmul_33 = paddle._C_ops.matmul(transpose_30, transpose_33, False, False)

        # pd_op.share_data_: (512x512x64xf32) <- (512x512x64xf32)
        share_data__3 = parameter_8.detach()

        # pd_op.assign: (512x512x64xf32) <- (512x512x64xf32)
        assign_54 = share_data__3
        del share_data__3

        # pd_op.slice: (11x11x64xf32) <- (512x512x64xf32, 2xi64, 2xi64)
        slice_6 = paddle._C_ops.slice(
            assign_54, [0, 1], full_int_array_3, full_int_array_4, [1, 1], []
        )
        del assign_54

        # pd_op.transpose: (11x1x12x64xf32) <- (1x12x11x64xf32)
        transpose_34 = paddle._C_ops.transpose(transpose_30, [2, 0, 1, 3])

        # pd_op.reshape: (11x12x64xf32) <- (11x1x12x64xf32, 3xi64)
        reshape_27 = paddle._C_ops.reshape(transpose_34, full_int_array_5)

        # pd_op.transpose: (11x64x11xf32) <- (11x11x64xf32)
        transpose_35 = paddle._C_ops.transpose(slice_6, [0, 2, 1])
        del slice_6

        # pd_op.matmul: (11x12x11xf32) <- (11x12x64xf32, 11x64x11xf32)
        matmul_34 = paddle._C_ops.matmul(reshape_27, transpose_35, False, False)

        # pd_op.reshape: (11x1x12x11xf32) <- (11x12x11xf32, 4xi64)
        reshape_28 = paddle._C_ops.reshape(matmul_34, full_int_array_6)

        # pd_op.transpose: (1x12x11x11xf32) <- (11x1x12x11xf32)
        transpose_36 = paddle._C_ops.transpose(reshape_28, [1, 2, 0, 3])
        del reshape_28

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x12x11x11xf32)
        add_37 = paddle._C_ops.add(matmul_33, transpose_36)

        # pd_op.scale: (1x12x11x11xf32) <- (1x12x11x11xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(add_37, full_6, float("0"), True)
        del add_37

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x1x1x11xf32)
        add_38 = paddle._C_ops.add(scale_5, scale_1)

        # pd_op.softmax: (1x12x11x11xf32) <- (1x12x11x11xf32)
        softmax_3 = paddle._C_ops.softmax(add_38, -1)
        del add_38

        # pd_op.dropout: (1x12x11x11xf32, 1x12x11x11xui8) <- (1x12x11x11xf32, None, 1xf32)
        dropout_20, dropout_21 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_3, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x11x64xf32) <- (1x12x11x11xf32, 1x12x11x64xf32)
        matmul_35 = paddle._C_ops.matmul(dropout_20, transpose_32, False, False)

        # pd_op.assign: (512x512x64xf32) <- (512x512x64xf32)
        assign_55 = parameter_8
        del parameter_8

        # pd_op.slice: (11x11x64xf32) <- (512x512x64xf32, 2xi64, 2xi64)
        slice_7 = paddle._C_ops.slice(
            assign_55, [0, 1], full_int_array_3, full_int_array_4, [1, 1], []
        )
        del assign_55

        # pd_op.transpose: (11x1x12x11xf32) <- (1x12x11x11xf32)
        transpose_37 = paddle._C_ops.transpose(dropout_20, [2, 0, 1, 3])

        # pd_op.reshape: (11x12x11xf32) <- (11x1x12x11xf32, 3xi64)
        reshape_29 = paddle._C_ops.reshape(transpose_37, full_int_array_7)

        # pd_op.matmul: (11x12x64xf32) <- (11x12x11xf32, 11x11x64xf32)
        matmul_36 = paddle._C_ops.matmul(reshape_29, slice_7, False, False)

        # pd_op.reshape: (11x1x12x64xf32) <- (11x12x64xf32, 4xi64)
        reshape_30 = paddle._C_ops.reshape(matmul_36, full_int_array_8)

        # pd_op.transpose: (1x12x11x64xf32) <- (11x1x12x64xf32)
        transpose_38 = paddle._C_ops.transpose(reshape_30, [1, 2, 0, 3])
        del reshape_30

        # pd_op.add: (1x12x11x64xf32) <- (1x12x11x64xf32, 1x12x11x64xf32)
        add_39 = paddle._C_ops.add(matmul_35, transpose_38)

        # pd_op.transpose: (1x11x12x64xf32) <- (1x12x11x64xf32)
        transpose_39 = paddle._C_ops.transpose(add_39, [0, 2, 1, 3])
        del add_39

        # pd_op.reshape: (1x11x768xf32) <- (1x11x12x64xf32, 3xi64)
        reshape_31 = paddle._C_ops.reshape(transpose_39, full_int_array_9)

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_37 = paddle._C_ops.matmul(reshape_31, parameter_149, False, False)
        del parameter_149

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_40 = paddle._C_ops.add(matmul_37, parameter_148)
        del parameter_148

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_22, dropout_23 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_40, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_40

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_41 = paddle._C_ops.add(layer_norm_18, dropout_22)

        # pd_op.layer_norm: (1x11x768xf32, 1x11xf32, 1x11xf32) <- (1x11x768xf32, 768xf32, 768xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_41, parameter_147, parameter_146, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_146, parameter_147

        # pd_op.matmul: (1x11x3072xf32) <- (1x11x768xf32, 768x3072xf32)
        matmul_38 = paddle._C_ops.matmul(layer_norm_21, parameter_145, False, False)
        del parameter_145

        # pd_op.add: (1x11x3072xf32) <- (1x11x3072xf32, 3072xf32)
        add_42 = paddle._C_ops.add(matmul_38, parameter_144)
        del parameter_144

        # pd_op.gelu: (1x11x3072xf32) <- (1x11x3072xf32)
        gelu_3 = paddle._C_ops.gelu(add_42, False)

        # pd_op.matmul: (1x11x768xf32) <- (1x11x3072xf32, 3072x768xf32)
        matmul_39 = paddle._C_ops.matmul(gelu_3, parameter_143, False, False)
        del parameter_143

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_43 = paddle._C_ops.add(matmul_39, parameter_142)
        del parameter_142

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_24, dropout_25 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_43, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_43

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_44 = paddle._C_ops.add(dropout_24, layer_norm_21)

        # pd_op.layer_norm: (1x11x768xf32, 1x11xf32, 1x11xf32) <- (1x11x768xf32, 768xf32, 768xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_44, parameter_157, parameter_156, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_156, parameter_157

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_40 = paddle._C_ops.matmul(layer_norm_24, parameter_139, False, False)
        del parameter_139

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_45 = paddle._C_ops.add(matmul_40, parameter_138)
        del parameter_138

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_41 = paddle._C_ops.matmul(layer_norm_24, parameter_137, False, False)
        del parameter_137

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_46 = paddle._C_ops.add(matmul_41, parameter_136)
        del parameter_136

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_42 = paddle._C_ops.matmul(layer_norm_24, parameter_135, False, False)
        del parameter_135

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_47 = paddle._C_ops.add(matmul_42, parameter_134)
        del parameter_134

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_32 = paddle._C_ops.reshape(add_45, full_int_array_2)

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_40 = paddle._C_ops.transpose(reshape_32, [0, 2, 1, 3])
        del reshape_32

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_33 = paddle._C_ops.reshape(add_46, full_int_array_2)

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_41 = paddle._C_ops.transpose(reshape_33, [0, 2, 1, 3])
        del reshape_33

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_34 = paddle._C_ops.reshape(add_47, full_int_array_2)

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_42 = paddle._C_ops.transpose(reshape_34, [0, 2, 1, 3])
        del reshape_34

        # pd_op.transpose: (1x12x64x11xf32) <- (1x12x11x64xf32)
        transpose_43 = paddle._C_ops.transpose(transpose_41, [0, 1, 3, 2])
        del transpose_41

        # pd_op.matmul: (1x12x11x11xf32) <- (1x12x11x64xf32, 1x12x64x11xf32)
        matmul_43 = paddle._C_ops.matmul(transpose_40, transpose_43, False, False)

        # pd_op.share_data_: (512x512x64xf32) <- (512x512x64xf32)
        share_data__4 = parameter_7.detach()

        # pd_op.assign: (512x512x64xf32) <- (512x512x64xf32)
        assign_56 = share_data__4
        del share_data__4

        # pd_op.slice: (11x11x64xf32) <- (512x512x64xf32, 2xi64, 2xi64)
        slice_8 = paddle._C_ops.slice(
            assign_56, [0, 1], full_int_array_3, full_int_array_4, [1, 1], []
        )
        del assign_56

        # pd_op.transpose: (11x1x12x64xf32) <- (1x12x11x64xf32)
        transpose_44 = paddle._C_ops.transpose(transpose_40, [2, 0, 1, 3])

        # pd_op.reshape: (11x12x64xf32) <- (11x1x12x64xf32, 3xi64)
        reshape_35 = paddle._C_ops.reshape(transpose_44, full_int_array_5)

        # pd_op.transpose: (11x64x11xf32) <- (11x11x64xf32)
        transpose_45 = paddle._C_ops.transpose(slice_8, [0, 2, 1])
        del slice_8

        # pd_op.matmul: (11x12x11xf32) <- (11x12x64xf32, 11x64x11xf32)
        matmul_44 = paddle._C_ops.matmul(reshape_35, transpose_45, False, False)

        # pd_op.reshape: (11x1x12x11xf32) <- (11x12x11xf32, 4xi64)
        reshape_36 = paddle._C_ops.reshape(matmul_44, full_int_array_6)

        # pd_op.transpose: (1x12x11x11xf32) <- (11x1x12x11xf32)
        transpose_46 = paddle._C_ops.transpose(reshape_36, [1, 2, 0, 3])
        del reshape_36

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x12x11x11xf32)
        add_48 = paddle._C_ops.add(matmul_43, transpose_46)

        # pd_op.scale: (1x12x11x11xf32) <- (1x12x11x11xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(add_48, full_6, float("0"), True)
        del add_48

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x1x1x11xf32)
        add_49 = paddle._C_ops.add(scale_6, scale_1)

        # pd_op.softmax: (1x12x11x11xf32) <- (1x12x11x11xf32)
        softmax_4 = paddle._C_ops.softmax(add_49, -1)
        del add_49

        # pd_op.dropout: (1x12x11x11xf32, 1x12x11x11xui8) <- (1x12x11x11xf32, None, 1xf32)
        dropout_26, dropout_27 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_4, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x11x64xf32) <- (1x12x11x11xf32, 1x12x11x64xf32)
        matmul_45 = paddle._C_ops.matmul(dropout_26, transpose_42, False, False)

        # pd_op.assign: (512x512x64xf32) <- (512x512x64xf32)
        assign_57 = parameter_7
        del parameter_7

        # pd_op.slice: (11x11x64xf32) <- (512x512x64xf32, 2xi64, 2xi64)
        slice_9 = paddle._C_ops.slice(
            assign_57, [0, 1], full_int_array_3, full_int_array_4, [1, 1], []
        )
        del assign_57

        # pd_op.transpose: (11x1x12x11xf32) <- (1x12x11x11xf32)
        transpose_47 = paddle._C_ops.transpose(dropout_26, [2, 0, 1, 3])

        # pd_op.reshape: (11x12x11xf32) <- (11x1x12x11xf32, 3xi64)
        reshape_37 = paddle._C_ops.reshape(transpose_47, full_int_array_7)

        # pd_op.matmul: (11x12x64xf32) <- (11x12x11xf32, 11x11x64xf32)
        matmul_46 = paddle._C_ops.matmul(reshape_37, slice_9, False, False)

        # pd_op.reshape: (11x1x12x64xf32) <- (11x12x64xf32, 4xi64)
        reshape_38 = paddle._C_ops.reshape(matmul_46, full_int_array_8)

        # pd_op.transpose: (1x12x11x64xf32) <- (11x1x12x64xf32)
        transpose_48 = paddle._C_ops.transpose(reshape_38, [1, 2, 0, 3])
        del reshape_38

        # pd_op.add: (1x12x11x64xf32) <- (1x12x11x64xf32, 1x12x11x64xf32)
        add_50 = paddle._C_ops.add(matmul_45, transpose_48)

        # pd_op.transpose: (1x11x12x64xf32) <- (1x12x11x64xf32)
        transpose_49 = paddle._C_ops.transpose(add_50, [0, 2, 1, 3])
        del add_50

        # pd_op.reshape: (1x11x768xf32) <- (1x11x12x64xf32, 3xi64)
        reshape_39 = paddle._C_ops.reshape(transpose_49, full_int_array_9)

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_47 = paddle._C_ops.matmul(reshape_39, parameter_133, False, False)
        del parameter_133

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_51 = paddle._C_ops.add(matmul_47, parameter_132)
        del parameter_132

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_28, dropout_29 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_51, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_51

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_52 = paddle._C_ops.add(layer_norm_24, dropout_28)

        # pd_op.layer_norm: (1x11x768xf32, 1x11xf32, 1x11xf32) <- (1x11x768xf32, 768xf32, 768xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_52, parameter_131, parameter_130, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_130, parameter_131

        # pd_op.matmul: (1x11x3072xf32) <- (1x11x768xf32, 768x3072xf32)
        matmul_48 = paddle._C_ops.matmul(layer_norm_27, parameter_129, False, False)
        del parameter_129

        # pd_op.add: (1x11x3072xf32) <- (1x11x3072xf32, 3072xf32)
        add_53 = paddle._C_ops.add(matmul_48, parameter_128)
        del parameter_128

        # pd_op.gelu: (1x11x3072xf32) <- (1x11x3072xf32)
        gelu_4 = paddle._C_ops.gelu(add_53, False)

        # pd_op.matmul: (1x11x768xf32) <- (1x11x3072xf32, 3072x768xf32)
        matmul_49 = paddle._C_ops.matmul(gelu_4, parameter_127, False, False)
        del parameter_127

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_54 = paddle._C_ops.add(matmul_49, parameter_126)
        del parameter_126

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_30, dropout_31 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_54, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_54

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_55 = paddle._C_ops.add(dropout_30, layer_norm_27)

        # pd_op.layer_norm: (1x11x768xf32, 1x11xf32, 1x11xf32) <- (1x11x768xf32, 768xf32, 768xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_55, parameter_141, parameter_140, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_140, parameter_141

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_50 = paddle._C_ops.matmul(layer_norm_30, parameter_123, False, False)
        del parameter_123

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_56 = paddle._C_ops.add(matmul_50, parameter_122)
        del parameter_122

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_51 = paddle._C_ops.matmul(layer_norm_30, parameter_121, False, False)
        del parameter_121

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_57 = paddle._C_ops.add(matmul_51, parameter_120)
        del parameter_120

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_52 = paddle._C_ops.matmul(layer_norm_30, parameter_119, False, False)
        del parameter_119

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_58 = paddle._C_ops.add(matmul_52, parameter_118)
        del parameter_118

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_40 = paddle._C_ops.reshape(add_56, full_int_array_2)

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_50 = paddle._C_ops.transpose(reshape_40, [0, 2, 1, 3])
        del reshape_40

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_41 = paddle._C_ops.reshape(add_57, full_int_array_2)

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_51 = paddle._C_ops.transpose(reshape_41, [0, 2, 1, 3])
        del reshape_41

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_42 = paddle._C_ops.reshape(add_58, full_int_array_2)

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_52 = paddle._C_ops.transpose(reshape_42, [0, 2, 1, 3])
        del reshape_42

        # pd_op.transpose: (1x12x64x11xf32) <- (1x12x11x64xf32)
        transpose_53 = paddle._C_ops.transpose(transpose_51, [0, 1, 3, 2])
        del transpose_51

        # pd_op.matmul: (1x12x11x11xf32) <- (1x12x11x64xf32, 1x12x64x11xf32)
        matmul_53 = paddle._C_ops.matmul(transpose_50, transpose_53, False, False)

        # pd_op.share_data_: (512x512x64xf32) <- (512x512x64xf32)
        share_data__5 = parameter_6.detach()

        # pd_op.assign: (512x512x64xf32) <- (512x512x64xf32)
        assign_58 = share_data__5
        del share_data__5

        # pd_op.slice: (11x11x64xf32) <- (512x512x64xf32, 2xi64, 2xi64)
        slice_10 = paddle._C_ops.slice(
            assign_58, [0, 1], full_int_array_3, full_int_array_4, [1, 1], []
        )
        del assign_58

        # pd_op.transpose: (11x1x12x64xf32) <- (1x12x11x64xf32)
        transpose_54 = paddle._C_ops.transpose(transpose_50, [2, 0, 1, 3])

        # pd_op.reshape: (11x12x64xf32) <- (11x1x12x64xf32, 3xi64)
        reshape_43 = paddle._C_ops.reshape(transpose_54, full_int_array_5)

        # pd_op.transpose: (11x64x11xf32) <- (11x11x64xf32)
        transpose_55 = paddle._C_ops.transpose(slice_10, [0, 2, 1])
        del slice_10

        # pd_op.matmul: (11x12x11xf32) <- (11x12x64xf32, 11x64x11xf32)
        matmul_54 = paddle._C_ops.matmul(reshape_43, transpose_55, False, False)

        # pd_op.reshape: (11x1x12x11xf32) <- (11x12x11xf32, 4xi64)
        reshape_44 = paddle._C_ops.reshape(matmul_54, full_int_array_6)

        # pd_op.transpose: (1x12x11x11xf32) <- (11x1x12x11xf32)
        transpose_56 = paddle._C_ops.transpose(reshape_44, [1, 2, 0, 3])
        del reshape_44

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x12x11x11xf32)
        add_59 = paddle._C_ops.add(matmul_53, transpose_56)

        # pd_op.scale: (1x12x11x11xf32) <- (1x12x11x11xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(add_59, full_6, float("0"), True)
        del add_59

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x1x1x11xf32)
        add_60 = paddle._C_ops.add(scale_7, scale_1)

        # pd_op.softmax: (1x12x11x11xf32) <- (1x12x11x11xf32)
        softmax_5 = paddle._C_ops.softmax(add_60, -1)
        del add_60

        # pd_op.dropout: (1x12x11x11xf32, 1x12x11x11xui8) <- (1x12x11x11xf32, None, 1xf32)
        dropout_32, dropout_33 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_5, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x11x64xf32) <- (1x12x11x11xf32, 1x12x11x64xf32)
        matmul_55 = paddle._C_ops.matmul(dropout_32, transpose_52, False, False)

        # pd_op.assign: (512x512x64xf32) <- (512x512x64xf32)
        assign_59 = parameter_6
        del parameter_6

        # pd_op.slice: (11x11x64xf32) <- (512x512x64xf32, 2xi64, 2xi64)
        slice_11 = paddle._C_ops.slice(
            assign_59, [0, 1], full_int_array_3, full_int_array_4, [1, 1], []
        )
        del assign_59

        # pd_op.transpose: (11x1x12x11xf32) <- (1x12x11x11xf32)
        transpose_57 = paddle._C_ops.transpose(dropout_32, [2, 0, 1, 3])

        # pd_op.reshape: (11x12x11xf32) <- (11x1x12x11xf32, 3xi64)
        reshape_45 = paddle._C_ops.reshape(transpose_57, full_int_array_7)

        # pd_op.matmul: (11x12x64xf32) <- (11x12x11xf32, 11x11x64xf32)
        matmul_56 = paddle._C_ops.matmul(reshape_45, slice_11, False, False)

        # pd_op.reshape: (11x1x12x64xf32) <- (11x12x64xf32, 4xi64)
        reshape_46 = paddle._C_ops.reshape(matmul_56, full_int_array_8)

        # pd_op.transpose: (1x12x11x64xf32) <- (11x1x12x64xf32)
        transpose_58 = paddle._C_ops.transpose(reshape_46, [1, 2, 0, 3])
        del reshape_46

        # pd_op.add: (1x12x11x64xf32) <- (1x12x11x64xf32, 1x12x11x64xf32)
        add_61 = paddle._C_ops.add(matmul_55, transpose_58)

        # pd_op.transpose: (1x11x12x64xf32) <- (1x12x11x64xf32)
        transpose_59 = paddle._C_ops.transpose(add_61, [0, 2, 1, 3])
        del add_61

        # pd_op.reshape: (1x11x768xf32) <- (1x11x12x64xf32, 3xi64)
        reshape_47 = paddle._C_ops.reshape(transpose_59, full_int_array_9)

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_57 = paddle._C_ops.matmul(reshape_47, parameter_117, False, False)
        del parameter_117

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_62 = paddle._C_ops.add(matmul_57, parameter_116)
        del parameter_116

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_34, dropout_35 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_62, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_62

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_63 = paddle._C_ops.add(layer_norm_30, dropout_34)

        # pd_op.layer_norm: (1x11x768xf32, 1x11xf32, 1x11xf32) <- (1x11x768xf32, 768xf32, 768xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_63, parameter_115, parameter_114, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_114, parameter_115

        # pd_op.matmul: (1x11x3072xf32) <- (1x11x768xf32, 768x3072xf32)
        matmul_58 = paddle._C_ops.matmul(layer_norm_33, parameter_113, False, False)
        del parameter_113

        # pd_op.add: (1x11x3072xf32) <- (1x11x3072xf32, 3072xf32)
        add_64 = paddle._C_ops.add(matmul_58, parameter_112)
        del parameter_112

        # pd_op.gelu: (1x11x3072xf32) <- (1x11x3072xf32)
        gelu_5 = paddle._C_ops.gelu(add_64, False)

        # pd_op.matmul: (1x11x768xf32) <- (1x11x3072xf32, 3072x768xf32)
        matmul_59 = paddle._C_ops.matmul(gelu_5, parameter_111, False, False)
        del parameter_111

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_65 = paddle._C_ops.add(matmul_59, parameter_110)
        del parameter_110

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_36, dropout_37 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_65, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_65

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_66 = paddle._C_ops.add(dropout_36, layer_norm_33)

        # pd_op.layer_norm: (1x11x768xf32, 1x11xf32, 1x11xf32) <- (1x11x768xf32, 768xf32, 768xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_66, parameter_125, parameter_124, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_124, parameter_125

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_60 = paddle._C_ops.matmul(layer_norm_36, parameter_107, False, False)
        del parameter_107

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_67 = paddle._C_ops.add(matmul_60, parameter_106)
        del parameter_106

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_61 = paddle._C_ops.matmul(layer_norm_36, parameter_105, False, False)
        del parameter_105

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_68 = paddle._C_ops.add(matmul_61, parameter_104)
        del parameter_104

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_62 = paddle._C_ops.matmul(layer_norm_36, parameter_103, False, False)
        del parameter_103

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_69 = paddle._C_ops.add(matmul_62, parameter_102)
        del parameter_102

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_48 = paddle._C_ops.reshape(add_67, full_int_array_2)

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_60 = paddle._C_ops.transpose(reshape_48, [0, 2, 1, 3])
        del reshape_48

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_49 = paddle._C_ops.reshape(add_68, full_int_array_2)

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_61 = paddle._C_ops.transpose(reshape_49, [0, 2, 1, 3])
        del reshape_49

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_50 = paddle._C_ops.reshape(add_69, full_int_array_2)

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_62 = paddle._C_ops.transpose(reshape_50, [0, 2, 1, 3])
        del reshape_50

        # pd_op.transpose: (1x12x64x11xf32) <- (1x12x11x64xf32)
        transpose_63 = paddle._C_ops.transpose(transpose_61, [0, 1, 3, 2])
        del transpose_61

        # pd_op.matmul: (1x12x11x11xf32) <- (1x12x11x64xf32, 1x12x64x11xf32)
        matmul_63 = paddle._C_ops.matmul(transpose_60, transpose_63, False, False)

        # pd_op.share_data_: (512x512x64xf32) <- (512x512x64xf32)
        share_data__6 = parameter_5.detach()

        # pd_op.assign: (512x512x64xf32) <- (512x512x64xf32)
        assign_60 = share_data__6
        del share_data__6

        # pd_op.slice: (11x11x64xf32) <- (512x512x64xf32, 2xi64, 2xi64)
        slice_12 = paddle._C_ops.slice(
            assign_60, [0, 1], full_int_array_3, full_int_array_4, [1, 1], []
        )
        del assign_60

        # pd_op.transpose: (11x1x12x64xf32) <- (1x12x11x64xf32)
        transpose_64 = paddle._C_ops.transpose(transpose_60, [2, 0, 1, 3])

        # pd_op.reshape: (11x12x64xf32) <- (11x1x12x64xf32, 3xi64)
        reshape_51 = paddle._C_ops.reshape(transpose_64, full_int_array_5)

        # pd_op.transpose: (11x64x11xf32) <- (11x11x64xf32)
        transpose_65 = paddle._C_ops.transpose(slice_12, [0, 2, 1])
        del slice_12

        # pd_op.matmul: (11x12x11xf32) <- (11x12x64xf32, 11x64x11xf32)
        matmul_64 = paddle._C_ops.matmul(reshape_51, transpose_65, False, False)

        # pd_op.reshape: (11x1x12x11xf32) <- (11x12x11xf32, 4xi64)
        reshape_52 = paddle._C_ops.reshape(matmul_64, full_int_array_6)

        # pd_op.transpose: (1x12x11x11xf32) <- (11x1x12x11xf32)
        transpose_66 = paddle._C_ops.transpose(reshape_52, [1, 2, 0, 3])
        del reshape_52

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x12x11x11xf32)
        add_70 = paddle._C_ops.add(matmul_63, transpose_66)

        # pd_op.scale: (1x12x11x11xf32) <- (1x12x11x11xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(add_70, full_6, float("0"), True)
        del add_70

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x1x1x11xf32)
        add_71 = paddle._C_ops.add(scale_8, scale_1)

        # pd_op.softmax: (1x12x11x11xf32) <- (1x12x11x11xf32)
        softmax_6 = paddle._C_ops.softmax(add_71, -1)
        del add_71

        # pd_op.dropout: (1x12x11x11xf32, 1x12x11x11xui8) <- (1x12x11x11xf32, None, 1xf32)
        dropout_38, dropout_39 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_6, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x11x64xf32) <- (1x12x11x11xf32, 1x12x11x64xf32)
        matmul_65 = paddle._C_ops.matmul(dropout_38, transpose_62, False, False)

        # pd_op.assign: (512x512x64xf32) <- (512x512x64xf32)
        assign_61 = parameter_5
        del parameter_5

        # pd_op.slice: (11x11x64xf32) <- (512x512x64xf32, 2xi64, 2xi64)
        slice_13 = paddle._C_ops.slice(
            assign_61, [0, 1], full_int_array_3, full_int_array_4, [1, 1], []
        )
        del assign_61

        # pd_op.transpose: (11x1x12x11xf32) <- (1x12x11x11xf32)
        transpose_67 = paddle._C_ops.transpose(dropout_38, [2, 0, 1, 3])

        # pd_op.reshape: (11x12x11xf32) <- (11x1x12x11xf32, 3xi64)
        reshape_53 = paddle._C_ops.reshape(transpose_67, full_int_array_7)

        # pd_op.matmul: (11x12x64xf32) <- (11x12x11xf32, 11x11x64xf32)
        matmul_66 = paddle._C_ops.matmul(reshape_53, slice_13, False, False)

        # pd_op.reshape: (11x1x12x64xf32) <- (11x12x64xf32, 4xi64)
        reshape_54 = paddle._C_ops.reshape(matmul_66, full_int_array_8)

        # pd_op.transpose: (1x12x11x64xf32) <- (11x1x12x64xf32)
        transpose_68 = paddle._C_ops.transpose(reshape_54, [1, 2, 0, 3])
        del reshape_54

        # pd_op.add: (1x12x11x64xf32) <- (1x12x11x64xf32, 1x12x11x64xf32)
        add_72 = paddle._C_ops.add(matmul_65, transpose_68)

        # pd_op.transpose: (1x11x12x64xf32) <- (1x12x11x64xf32)
        transpose_69 = paddle._C_ops.transpose(add_72, [0, 2, 1, 3])
        del add_72

        # pd_op.reshape: (1x11x768xf32) <- (1x11x12x64xf32, 3xi64)
        reshape_55 = paddle._C_ops.reshape(transpose_69, full_int_array_9)

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_67 = paddle._C_ops.matmul(reshape_55, parameter_101, False, False)
        del parameter_101

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_73 = paddle._C_ops.add(matmul_67, parameter_100)
        del parameter_100

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_40, dropout_41 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_73, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_73

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_74 = paddle._C_ops.add(layer_norm_36, dropout_40)

        # pd_op.layer_norm: (1x11x768xf32, 1x11xf32, 1x11xf32) <- (1x11x768xf32, 768xf32, 768xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_74, parameter_99, parameter_98, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_98, parameter_99

        # pd_op.matmul: (1x11x3072xf32) <- (1x11x768xf32, 768x3072xf32)
        matmul_68 = paddle._C_ops.matmul(layer_norm_39, parameter_97, False, False)
        del parameter_97

        # pd_op.add: (1x11x3072xf32) <- (1x11x3072xf32, 3072xf32)
        add_75 = paddle._C_ops.add(matmul_68, parameter_96)
        del parameter_96

        # pd_op.gelu: (1x11x3072xf32) <- (1x11x3072xf32)
        gelu_6 = paddle._C_ops.gelu(add_75, False)

        # pd_op.matmul: (1x11x768xf32) <- (1x11x3072xf32, 3072x768xf32)
        matmul_69 = paddle._C_ops.matmul(gelu_6, parameter_95, False, False)
        del parameter_95

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_76 = paddle._C_ops.add(matmul_69, parameter_94)
        del parameter_94

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_42, dropout_43 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_76, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_76

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_77 = paddle._C_ops.add(dropout_42, layer_norm_39)

        # pd_op.layer_norm: (1x11x768xf32, 1x11xf32, 1x11xf32) <- (1x11x768xf32, 768xf32, 768xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_77, parameter_109, parameter_108, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_108, parameter_109

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_70 = paddle._C_ops.matmul(layer_norm_42, parameter_91, False, False)
        del parameter_91

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_78 = paddle._C_ops.add(matmul_70, parameter_90)
        del parameter_90

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_71 = paddle._C_ops.matmul(layer_norm_42, parameter_89, False, False)
        del parameter_89

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_79 = paddle._C_ops.add(matmul_71, parameter_88)
        del parameter_88

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_72 = paddle._C_ops.matmul(layer_norm_42, parameter_87, False, False)
        del parameter_87

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_80 = paddle._C_ops.add(matmul_72, parameter_86)
        del parameter_86

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_56 = paddle._C_ops.reshape(add_78, full_int_array_2)

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_70 = paddle._C_ops.transpose(reshape_56, [0, 2, 1, 3])
        del reshape_56

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_57 = paddle._C_ops.reshape(add_79, full_int_array_2)

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_71 = paddle._C_ops.transpose(reshape_57, [0, 2, 1, 3])
        del reshape_57

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_58 = paddle._C_ops.reshape(add_80, full_int_array_2)

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_72 = paddle._C_ops.transpose(reshape_58, [0, 2, 1, 3])
        del reshape_58

        # pd_op.transpose: (1x12x64x11xf32) <- (1x12x11x64xf32)
        transpose_73 = paddle._C_ops.transpose(transpose_71, [0, 1, 3, 2])
        del transpose_71

        # pd_op.matmul: (1x12x11x11xf32) <- (1x12x11x64xf32, 1x12x64x11xf32)
        matmul_73 = paddle._C_ops.matmul(transpose_70, transpose_73, False, False)

        # pd_op.share_data_: (512x512x64xf32) <- (512x512x64xf32)
        share_data__7 = parameter_4.detach()

        # pd_op.assign: (512x512x64xf32) <- (512x512x64xf32)
        assign_62 = share_data__7
        del share_data__7

        # pd_op.slice: (11x11x64xf32) <- (512x512x64xf32, 2xi64, 2xi64)
        slice_14 = paddle._C_ops.slice(
            assign_62, [0, 1], full_int_array_3, full_int_array_4, [1, 1], []
        )
        del assign_62

        # pd_op.transpose: (11x1x12x64xf32) <- (1x12x11x64xf32)
        transpose_74 = paddle._C_ops.transpose(transpose_70, [2, 0, 1, 3])

        # pd_op.reshape: (11x12x64xf32) <- (11x1x12x64xf32, 3xi64)
        reshape_59 = paddle._C_ops.reshape(transpose_74, full_int_array_5)

        # pd_op.transpose: (11x64x11xf32) <- (11x11x64xf32)
        transpose_75 = paddle._C_ops.transpose(slice_14, [0, 2, 1])
        del slice_14

        # pd_op.matmul: (11x12x11xf32) <- (11x12x64xf32, 11x64x11xf32)
        matmul_74 = paddle._C_ops.matmul(reshape_59, transpose_75, False, False)

        # pd_op.reshape: (11x1x12x11xf32) <- (11x12x11xf32, 4xi64)
        reshape_60 = paddle._C_ops.reshape(matmul_74, full_int_array_6)

        # pd_op.transpose: (1x12x11x11xf32) <- (11x1x12x11xf32)
        transpose_76 = paddle._C_ops.transpose(reshape_60, [1, 2, 0, 3])
        del reshape_60

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x12x11x11xf32)
        add_81 = paddle._C_ops.add(matmul_73, transpose_76)

        # pd_op.scale: (1x12x11x11xf32) <- (1x12x11x11xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(add_81, full_6, float("0"), True)
        del add_81

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x1x1x11xf32)
        add_82 = paddle._C_ops.add(scale_9, scale_1)

        # pd_op.softmax: (1x12x11x11xf32) <- (1x12x11x11xf32)
        softmax_7 = paddle._C_ops.softmax(add_82, -1)
        del add_82

        # pd_op.dropout: (1x12x11x11xf32, 1x12x11x11xui8) <- (1x12x11x11xf32, None, 1xf32)
        dropout_44, dropout_45 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_7, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x11x64xf32) <- (1x12x11x11xf32, 1x12x11x64xf32)
        matmul_75 = paddle._C_ops.matmul(dropout_44, transpose_72, False, False)

        # pd_op.assign: (512x512x64xf32) <- (512x512x64xf32)
        assign_63 = parameter_4
        del parameter_4

        # pd_op.slice: (11x11x64xf32) <- (512x512x64xf32, 2xi64, 2xi64)
        slice_15 = paddle._C_ops.slice(
            assign_63, [0, 1], full_int_array_3, full_int_array_4, [1, 1], []
        )
        del assign_63

        # pd_op.transpose: (11x1x12x11xf32) <- (1x12x11x11xf32)
        transpose_77 = paddle._C_ops.transpose(dropout_44, [2, 0, 1, 3])

        # pd_op.reshape: (11x12x11xf32) <- (11x1x12x11xf32, 3xi64)
        reshape_61 = paddle._C_ops.reshape(transpose_77, full_int_array_7)

        # pd_op.matmul: (11x12x64xf32) <- (11x12x11xf32, 11x11x64xf32)
        matmul_76 = paddle._C_ops.matmul(reshape_61, slice_15, False, False)

        # pd_op.reshape: (11x1x12x64xf32) <- (11x12x64xf32, 4xi64)
        reshape_62 = paddle._C_ops.reshape(matmul_76, full_int_array_8)

        # pd_op.transpose: (1x12x11x64xf32) <- (11x1x12x64xf32)
        transpose_78 = paddle._C_ops.transpose(reshape_62, [1, 2, 0, 3])
        del reshape_62

        # pd_op.add: (1x12x11x64xf32) <- (1x12x11x64xf32, 1x12x11x64xf32)
        add_83 = paddle._C_ops.add(matmul_75, transpose_78)

        # pd_op.transpose: (1x11x12x64xf32) <- (1x12x11x64xf32)
        transpose_79 = paddle._C_ops.transpose(add_83, [0, 2, 1, 3])
        del add_83

        # pd_op.reshape: (1x11x768xf32) <- (1x11x12x64xf32, 3xi64)
        reshape_63 = paddle._C_ops.reshape(transpose_79, full_int_array_9)

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_77 = paddle._C_ops.matmul(reshape_63, parameter_85, False, False)
        del parameter_85

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_84 = paddle._C_ops.add(matmul_77, parameter_84)
        del parameter_84

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_46, dropout_47 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_84, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_84

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_85 = paddle._C_ops.add(layer_norm_42, dropout_46)

        # pd_op.layer_norm: (1x11x768xf32, 1x11xf32, 1x11xf32) <- (1x11x768xf32, 768xf32, 768xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_85, parameter_83, parameter_82, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_82, parameter_83

        # pd_op.matmul: (1x11x3072xf32) <- (1x11x768xf32, 768x3072xf32)
        matmul_78 = paddle._C_ops.matmul(layer_norm_45, parameter_81, False, False)
        del parameter_81

        # pd_op.add: (1x11x3072xf32) <- (1x11x3072xf32, 3072xf32)
        add_86 = paddle._C_ops.add(matmul_78, parameter_80)
        del parameter_80

        # pd_op.gelu: (1x11x3072xf32) <- (1x11x3072xf32)
        gelu_7 = paddle._C_ops.gelu(add_86, False)

        # pd_op.matmul: (1x11x768xf32) <- (1x11x3072xf32, 3072x768xf32)
        matmul_79 = paddle._C_ops.matmul(gelu_7, parameter_79, False, False)
        del parameter_79

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_87 = paddle._C_ops.add(matmul_79, parameter_78)
        del parameter_78

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_48, dropout_49 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_87, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_87

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_88 = paddle._C_ops.add(dropout_48, layer_norm_45)

        # pd_op.layer_norm: (1x11x768xf32, 1x11xf32, 1x11xf32) <- (1x11x768xf32, 768xf32, 768xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_88, parameter_93, parameter_92, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_92, parameter_93

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_80 = paddle._C_ops.matmul(layer_norm_48, parameter_75, False, False)
        del parameter_75

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_89 = paddle._C_ops.add(matmul_80, parameter_74)
        del parameter_74

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_81 = paddle._C_ops.matmul(layer_norm_48, parameter_73, False, False)
        del parameter_73

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_90 = paddle._C_ops.add(matmul_81, parameter_72)
        del parameter_72

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_82 = paddle._C_ops.matmul(layer_norm_48, parameter_71, False, False)
        del parameter_71

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_91 = paddle._C_ops.add(matmul_82, parameter_70)
        del parameter_70

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_64 = paddle._C_ops.reshape(add_89, full_int_array_2)

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_80 = paddle._C_ops.transpose(reshape_64, [0, 2, 1, 3])
        del reshape_64

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_65 = paddle._C_ops.reshape(add_90, full_int_array_2)

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_81 = paddle._C_ops.transpose(reshape_65, [0, 2, 1, 3])
        del reshape_65

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_66 = paddle._C_ops.reshape(add_91, full_int_array_2)

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_82 = paddle._C_ops.transpose(reshape_66, [0, 2, 1, 3])
        del reshape_66

        # pd_op.transpose: (1x12x64x11xf32) <- (1x12x11x64xf32)
        transpose_83 = paddle._C_ops.transpose(transpose_81, [0, 1, 3, 2])
        del transpose_81

        # pd_op.matmul: (1x12x11x11xf32) <- (1x12x11x64xf32, 1x12x64x11xf32)
        matmul_83 = paddle._C_ops.matmul(transpose_80, transpose_83, False, False)

        # pd_op.share_data_: (512x512x64xf32) <- (512x512x64xf32)
        share_data__8 = parameter_3.detach()

        # pd_op.assign: (512x512x64xf32) <- (512x512x64xf32)
        assign_64 = share_data__8
        del share_data__8

        # pd_op.slice: (11x11x64xf32) <- (512x512x64xf32, 2xi64, 2xi64)
        slice_16 = paddle._C_ops.slice(
            assign_64, [0, 1], full_int_array_3, full_int_array_4, [1, 1], []
        )
        del assign_64

        # pd_op.transpose: (11x1x12x64xf32) <- (1x12x11x64xf32)
        transpose_84 = paddle._C_ops.transpose(transpose_80, [2, 0, 1, 3])

        # pd_op.reshape: (11x12x64xf32) <- (11x1x12x64xf32, 3xi64)
        reshape_67 = paddle._C_ops.reshape(transpose_84, full_int_array_5)

        # pd_op.transpose: (11x64x11xf32) <- (11x11x64xf32)
        transpose_85 = paddle._C_ops.transpose(slice_16, [0, 2, 1])
        del slice_16

        # pd_op.matmul: (11x12x11xf32) <- (11x12x64xf32, 11x64x11xf32)
        matmul_84 = paddle._C_ops.matmul(reshape_67, transpose_85, False, False)

        # pd_op.reshape: (11x1x12x11xf32) <- (11x12x11xf32, 4xi64)
        reshape_68 = paddle._C_ops.reshape(matmul_84, full_int_array_6)

        # pd_op.transpose: (1x12x11x11xf32) <- (11x1x12x11xf32)
        transpose_86 = paddle._C_ops.transpose(reshape_68, [1, 2, 0, 3])
        del reshape_68

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x12x11x11xf32)
        add_92 = paddle._C_ops.add(matmul_83, transpose_86)

        # pd_op.scale: (1x12x11x11xf32) <- (1x12x11x11xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(add_92, full_6, float("0"), True)
        del add_92

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x1x1x11xf32)
        add_93 = paddle._C_ops.add(scale_10, scale_1)

        # pd_op.softmax: (1x12x11x11xf32) <- (1x12x11x11xf32)
        softmax_8 = paddle._C_ops.softmax(add_93, -1)
        del add_93

        # pd_op.dropout: (1x12x11x11xf32, 1x12x11x11xui8) <- (1x12x11x11xf32, None, 1xf32)
        dropout_50, dropout_51 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_8, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x11x64xf32) <- (1x12x11x11xf32, 1x12x11x64xf32)
        matmul_85 = paddle._C_ops.matmul(dropout_50, transpose_82, False, False)

        # pd_op.assign: (512x512x64xf32) <- (512x512x64xf32)
        assign_65 = parameter_3
        del parameter_3

        # pd_op.slice: (11x11x64xf32) <- (512x512x64xf32, 2xi64, 2xi64)
        slice_17 = paddle._C_ops.slice(
            assign_65, [0, 1], full_int_array_3, full_int_array_4, [1, 1], []
        )
        del assign_65

        # pd_op.transpose: (11x1x12x11xf32) <- (1x12x11x11xf32)
        transpose_87 = paddle._C_ops.transpose(dropout_50, [2, 0, 1, 3])

        # pd_op.reshape: (11x12x11xf32) <- (11x1x12x11xf32, 3xi64)
        reshape_69 = paddle._C_ops.reshape(transpose_87, full_int_array_7)

        # pd_op.matmul: (11x12x64xf32) <- (11x12x11xf32, 11x11x64xf32)
        matmul_86 = paddle._C_ops.matmul(reshape_69, slice_17, False, False)

        # pd_op.reshape: (11x1x12x64xf32) <- (11x12x64xf32, 4xi64)
        reshape_70 = paddle._C_ops.reshape(matmul_86, full_int_array_8)

        # pd_op.transpose: (1x12x11x64xf32) <- (11x1x12x64xf32)
        transpose_88 = paddle._C_ops.transpose(reshape_70, [1, 2, 0, 3])
        del reshape_70

        # pd_op.add: (1x12x11x64xf32) <- (1x12x11x64xf32, 1x12x11x64xf32)
        add_94 = paddle._C_ops.add(matmul_85, transpose_88)

        # pd_op.transpose: (1x11x12x64xf32) <- (1x12x11x64xf32)
        transpose_89 = paddle._C_ops.transpose(add_94, [0, 2, 1, 3])
        del add_94

        # pd_op.reshape: (1x11x768xf32) <- (1x11x12x64xf32, 3xi64)
        reshape_71 = paddle._C_ops.reshape(transpose_89, full_int_array_9)

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_87 = paddle._C_ops.matmul(reshape_71, parameter_69, False, False)
        del parameter_69

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_95 = paddle._C_ops.add(matmul_87, parameter_68)
        del parameter_68

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_52, dropout_53 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_95, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_95

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_96 = paddle._C_ops.add(layer_norm_48, dropout_52)

        # pd_op.layer_norm: (1x11x768xf32, 1x11xf32, 1x11xf32) <- (1x11x768xf32, 768xf32, 768xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_96, parameter_67, parameter_66, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_66, parameter_67

        # pd_op.matmul: (1x11x3072xf32) <- (1x11x768xf32, 768x3072xf32)
        matmul_88 = paddle._C_ops.matmul(layer_norm_51, parameter_65, False, False)
        del parameter_65

        # pd_op.add: (1x11x3072xf32) <- (1x11x3072xf32, 3072xf32)
        add_97 = paddle._C_ops.add(matmul_88, parameter_64)
        del parameter_64

        # pd_op.gelu: (1x11x3072xf32) <- (1x11x3072xf32)
        gelu_8 = paddle._C_ops.gelu(add_97, False)

        # pd_op.matmul: (1x11x768xf32) <- (1x11x3072xf32, 3072x768xf32)
        matmul_89 = paddle._C_ops.matmul(gelu_8, parameter_63, False, False)
        del parameter_63

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_98 = paddle._C_ops.add(matmul_89, parameter_62)
        del parameter_62

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_54, dropout_55 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_98, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_98

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_99 = paddle._C_ops.add(dropout_54, layer_norm_51)

        # pd_op.layer_norm: (1x11x768xf32, 1x11xf32, 1x11xf32) <- (1x11x768xf32, 768xf32, 768xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_99, parameter_77, parameter_76, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_76, parameter_77

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_90 = paddle._C_ops.matmul(layer_norm_54, parameter_59, False, False)
        del parameter_59

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_100 = paddle._C_ops.add(matmul_90, parameter_58)
        del parameter_58

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_91 = paddle._C_ops.matmul(layer_norm_54, parameter_57, False, False)
        del parameter_57

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_101 = paddle._C_ops.add(matmul_91, parameter_56)
        del parameter_56

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_92 = paddle._C_ops.matmul(layer_norm_54, parameter_55, False, False)
        del parameter_55

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_102 = paddle._C_ops.add(matmul_92, parameter_54)
        del parameter_54

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_72 = paddle._C_ops.reshape(add_100, full_int_array_2)

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_90 = paddle._C_ops.transpose(reshape_72, [0, 2, 1, 3])
        del reshape_72

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_73 = paddle._C_ops.reshape(add_101, full_int_array_2)

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_91 = paddle._C_ops.transpose(reshape_73, [0, 2, 1, 3])
        del reshape_73

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_74 = paddle._C_ops.reshape(add_102, full_int_array_2)

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_92 = paddle._C_ops.transpose(reshape_74, [0, 2, 1, 3])
        del reshape_74

        # pd_op.transpose: (1x12x64x11xf32) <- (1x12x11x64xf32)
        transpose_93 = paddle._C_ops.transpose(transpose_91, [0, 1, 3, 2])
        del transpose_91

        # pd_op.matmul: (1x12x11x11xf32) <- (1x12x11x64xf32, 1x12x64x11xf32)
        matmul_93 = paddle._C_ops.matmul(transpose_90, transpose_93, False, False)

        # pd_op.share_data_: (512x512x64xf32) <- (512x512x64xf32)
        share_data__9 = parameter_2.detach()

        # pd_op.assign: (512x512x64xf32) <- (512x512x64xf32)
        assign_66 = share_data__9
        del share_data__9

        # pd_op.slice: (11x11x64xf32) <- (512x512x64xf32, 2xi64, 2xi64)
        slice_18 = paddle._C_ops.slice(
            assign_66, [0, 1], full_int_array_3, full_int_array_4, [1, 1], []
        )
        del assign_66

        # pd_op.transpose: (11x1x12x64xf32) <- (1x12x11x64xf32)
        transpose_94 = paddle._C_ops.transpose(transpose_90, [2, 0, 1, 3])

        # pd_op.reshape: (11x12x64xf32) <- (11x1x12x64xf32, 3xi64)
        reshape_75 = paddle._C_ops.reshape(transpose_94, full_int_array_5)

        # pd_op.transpose: (11x64x11xf32) <- (11x11x64xf32)
        transpose_95 = paddle._C_ops.transpose(slice_18, [0, 2, 1])
        del slice_18

        # pd_op.matmul: (11x12x11xf32) <- (11x12x64xf32, 11x64x11xf32)
        matmul_94 = paddle._C_ops.matmul(reshape_75, transpose_95, False, False)

        # pd_op.reshape: (11x1x12x11xf32) <- (11x12x11xf32, 4xi64)
        reshape_76 = paddle._C_ops.reshape(matmul_94, full_int_array_6)

        # pd_op.transpose: (1x12x11x11xf32) <- (11x1x12x11xf32)
        transpose_96 = paddle._C_ops.transpose(reshape_76, [1, 2, 0, 3])
        del reshape_76

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x12x11x11xf32)
        add_103 = paddle._C_ops.add(matmul_93, transpose_96)

        # pd_op.scale: (1x12x11x11xf32) <- (1x12x11x11xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(add_103, full_6, float("0"), True)
        del add_103

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x1x1x11xf32)
        add_104 = paddle._C_ops.add(scale_11, scale_1)

        # pd_op.softmax: (1x12x11x11xf32) <- (1x12x11x11xf32)
        softmax_9 = paddle._C_ops.softmax(add_104, -1)
        del add_104

        # pd_op.dropout: (1x12x11x11xf32, 1x12x11x11xui8) <- (1x12x11x11xf32, None, 1xf32)
        dropout_56, dropout_57 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_9, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x11x64xf32) <- (1x12x11x11xf32, 1x12x11x64xf32)
        matmul_95 = paddle._C_ops.matmul(dropout_56, transpose_92, False, False)

        # pd_op.assign: (512x512x64xf32) <- (512x512x64xf32)
        assign_67 = parameter_2
        del parameter_2

        # pd_op.slice: (11x11x64xf32) <- (512x512x64xf32, 2xi64, 2xi64)
        slice_19 = paddle._C_ops.slice(
            assign_67, [0, 1], full_int_array_3, full_int_array_4, [1, 1], []
        )
        del assign_67

        # pd_op.transpose: (11x1x12x11xf32) <- (1x12x11x11xf32)
        transpose_97 = paddle._C_ops.transpose(dropout_56, [2, 0, 1, 3])

        # pd_op.reshape: (11x12x11xf32) <- (11x1x12x11xf32, 3xi64)
        reshape_77 = paddle._C_ops.reshape(transpose_97, full_int_array_7)

        # pd_op.matmul: (11x12x64xf32) <- (11x12x11xf32, 11x11x64xf32)
        matmul_96 = paddle._C_ops.matmul(reshape_77, slice_19, False, False)

        # pd_op.reshape: (11x1x12x64xf32) <- (11x12x64xf32, 4xi64)
        reshape_78 = paddle._C_ops.reshape(matmul_96, full_int_array_8)

        # pd_op.transpose: (1x12x11x64xf32) <- (11x1x12x64xf32)
        transpose_98 = paddle._C_ops.transpose(reshape_78, [1, 2, 0, 3])
        del reshape_78

        # pd_op.add: (1x12x11x64xf32) <- (1x12x11x64xf32, 1x12x11x64xf32)
        add_105 = paddle._C_ops.add(matmul_95, transpose_98)

        # pd_op.transpose: (1x11x12x64xf32) <- (1x12x11x64xf32)
        transpose_99 = paddle._C_ops.transpose(add_105, [0, 2, 1, 3])
        del add_105

        # pd_op.reshape: (1x11x768xf32) <- (1x11x12x64xf32, 3xi64)
        reshape_79 = paddle._C_ops.reshape(transpose_99, full_int_array_9)

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_97 = paddle._C_ops.matmul(reshape_79, parameter_53, False, False)
        del parameter_53

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_106 = paddle._C_ops.add(matmul_97, parameter_52)
        del parameter_52

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_58, dropout_59 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_106, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_106

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_107 = paddle._C_ops.add(layer_norm_54, dropout_58)

        # pd_op.layer_norm: (1x11x768xf32, 1x11xf32, 1x11xf32) <- (1x11x768xf32, 768xf32, 768xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_107, parameter_51, parameter_50, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_50, parameter_51

        # pd_op.matmul: (1x11x3072xf32) <- (1x11x768xf32, 768x3072xf32)
        matmul_98 = paddle._C_ops.matmul(layer_norm_57, parameter_49, False, False)
        del parameter_49

        # pd_op.add: (1x11x3072xf32) <- (1x11x3072xf32, 3072xf32)
        add_108 = paddle._C_ops.add(matmul_98, parameter_48)
        del parameter_48

        # pd_op.gelu: (1x11x3072xf32) <- (1x11x3072xf32)
        gelu_9 = paddle._C_ops.gelu(add_108, False)

        # pd_op.matmul: (1x11x768xf32) <- (1x11x3072xf32, 3072x768xf32)
        matmul_99 = paddle._C_ops.matmul(gelu_9, parameter_47, False, False)
        del parameter_47

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_109 = paddle._C_ops.add(matmul_99, parameter_46)
        del parameter_46

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_60, dropout_61 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_109, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_109

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_110 = paddle._C_ops.add(dropout_60, layer_norm_57)

        # pd_op.layer_norm: (1x11x768xf32, 1x11xf32, 1x11xf32) <- (1x11x768xf32, 768xf32, 768xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_110, parameter_61, parameter_60, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_60, parameter_61

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_100 = paddle._C_ops.matmul(layer_norm_60, parameter_43, False, False)
        del parameter_43

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_111 = paddle._C_ops.add(matmul_100, parameter_42)
        del parameter_42

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_101 = paddle._C_ops.matmul(layer_norm_60, parameter_41, False, False)
        del parameter_41

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_112 = paddle._C_ops.add(matmul_101, parameter_40)
        del parameter_40

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_102 = paddle._C_ops.matmul(layer_norm_60, parameter_39, False, False)
        del parameter_39

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_113 = paddle._C_ops.add(matmul_102, parameter_38)
        del parameter_38

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_80 = paddle._C_ops.reshape(add_111, full_int_array_2)

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_100 = paddle._C_ops.transpose(reshape_80, [0, 2, 1, 3])
        del reshape_80

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_81 = paddle._C_ops.reshape(add_112, full_int_array_2)

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_101 = paddle._C_ops.transpose(reshape_81, [0, 2, 1, 3])
        del reshape_81

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_82 = paddle._C_ops.reshape(add_113, full_int_array_2)

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_102 = paddle._C_ops.transpose(reshape_82, [0, 2, 1, 3])
        del reshape_82

        # pd_op.transpose: (1x12x64x11xf32) <- (1x12x11x64xf32)
        transpose_103 = paddle._C_ops.transpose(transpose_101, [0, 1, 3, 2])
        del transpose_101

        # pd_op.matmul: (1x12x11x11xf32) <- (1x12x11x64xf32, 1x12x64x11xf32)
        matmul_103 = paddle._C_ops.matmul(transpose_100, transpose_103, False, False)

        # pd_op.share_data_: (512x512x64xf32) <- (512x512x64xf32)
        share_data__10 = parameter_1.detach()

        # pd_op.assign: (512x512x64xf32) <- (512x512x64xf32)
        assign_68 = share_data__10
        del share_data__10

        # pd_op.slice: (11x11x64xf32) <- (512x512x64xf32, 2xi64, 2xi64)
        slice_20 = paddle._C_ops.slice(
            assign_68, [0, 1], full_int_array_3, full_int_array_4, [1, 1], []
        )
        del assign_68

        # pd_op.transpose: (11x1x12x64xf32) <- (1x12x11x64xf32)
        transpose_104 = paddle._C_ops.transpose(transpose_100, [2, 0, 1, 3])

        # pd_op.reshape: (11x12x64xf32) <- (11x1x12x64xf32, 3xi64)
        reshape_83 = paddle._C_ops.reshape(transpose_104, full_int_array_5)

        # pd_op.transpose: (11x64x11xf32) <- (11x11x64xf32)
        transpose_105 = paddle._C_ops.transpose(slice_20, [0, 2, 1])
        del slice_20

        # pd_op.matmul: (11x12x11xf32) <- (11x12x64xf32, 11x64x11xf32)
        matmul_104 = paddle._C_ops.matmul(reshape_83, transpose_105, False, False)

        # pd_op.reshape: (11x1x12x11xf32) <- (11x12x11xf32, 4xi64)
        reshape_84 = paddle._C_ops.reshape(matmul_104, full_int_array_6)

        # pd_op.transpose: (1x12x11x11xf32) <- (11x1x12x11xf32)
        transpose_106 = paddle._C_ops.transpose(reshape_84, [1, 2, 0, 3])
        del reshape_84

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x12x11x11xf32)
        add_114 = paddle._C_ops.add(matmul_103, transpose_106)

        # pd_op.scale: (1x12x11x11xf32) <- (1x12x11x11xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(add_114, full_6, float("0"), True)
        del add_114

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x1x1x11xf32)
        add_115 = paddle._C_ops.add(scale_12, scale_1)

        # pd_op.softmax: (1x12x11x11xf32) <- (1x12x11x11xf32)
        softmax_10 = paddle._C_ops.softmax(add_115, -1)
        del add_115

        # pd_op.dropout: (1x12x11x11xf32, 1x12x11x11xui8) <- (1x12x11x11xf32, None, 1xf32)
        dropout_62, dropout_63 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_10, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x11x64xf32) <- (1x12x11x11xf32, 1x12x11x64xf32)
        matmul_105 = paddle._C_ops.matmul(dropout_62, transpose_102, False, False)

        # pd_op.assign: (512x512x64xf32) <- (512x512x64xf32)
        assign_69 = parameter_1
        del parameter_1

        # pd_op.slice: (11x11x64xf32) <- (512x512x64xf32, 2xi64, 2xi64)
        slice_21 = paddle._C_ops.slice(
            assign_69, [0, 1], full_int_array_3, full_int_array_4, [1, 1], []
        )
        del assign_69

        # pd_op.transpose: (11x1x12x11xf32) <- (1x12x11x11xf32)
        transpose_107 = paddle._C_ops.transpose(dropout_62, [2, 0, 1, 3])

        # pd_op.reshape: (11x12x11xf32) <- (11x1x12x11xf32, 3xi64)
        reshape_85 = paddle._C_ops.reshape(transpose_107, full_int_array_7)

        # pd_op.matmul: (11x12x64xf32) <- (11x12x11xf32, 11x11x64xf32)
        matmul_106 = paddle._C_ops.matmul(reshape_85, slice_21, False, False)

        # pd_op.reshape: (11x1x12x64xf32) <- (11x12x64xf32, 4xi64)
        reshape_86 = paddle._C_ops.reshape(matmul_106, full_int_array_8)

        # pd_op.transpose: (1x12x11x64xf32) <- (11x1x12x64xf32)
        transpose_108 = paddle._C_ops.transpose(reshape_86, [1, 2, 0, 3])
        del reshape_86

        # pd_op.add: (1x12x11x64xf32) <- (1x12x11x64xf32, 1x12x11x64xf32)
        add_116 = paddle._C_ops.add(matmul_105, transpose_108)

        # pd_op.transpose: (1x11x12x64xf32) <- (1x12x11x64xf32)
        transpose_109 = paddle._C_ops.transpose(add_116, [0, 2, 1, 3])
        del add_116

        # pd_op.reshape: (1x11x768xf32) <- (1x11x12x64xf32, 3xi64)
        reshape_87 = paddle._C_ops.reshape(transpose_109, full_int_array_9)

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_107 = paddle._C_ops.matmul(reshape_87, parameter_37, False, False)
        del parameter_37

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_117 = paddle._C_ops.add(matmul_107, parameter_36)
        del parameter_36

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_64, dropout_65 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_117, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_117

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_118 = paddle._C_ops.add(layer_norm_60, dropout_64)

        # pd_op.layer_norm: (1x11x768xf32, 1x11xf32, 1x11xf32) <- (1x11x768xf32, 768xf32, 768xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_118, parameter_35, parameter_34, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_34, parameter_35

        # pd_op.matmul: (1x11x3072xf32) <- (1x11x768xf32, 768x3072xf32)
        matmul_108 = paddle._C_ops.matmul(layer_norm_63, parameter_33, False, False)
        del parameter_33

        # pd_op.add: (1x11x3072xf32) <- (1x11x3072xf32, 3072xf32)
        add_119 = paddle._C_ops.add(matmul_108, parameter_32)
        del parameter_32

        # pd_op.gelu: (1x11x3072xf32) <- (1x11x3072xf32)
        gelu_10 = paddle._C_ops.gelu(add_119, False)

        # pd_op.matmul: (1x11x768xf32) <- (1x11x3072xf32, 3072x768xf32)
        matmul_109 = paddle._C_ops.matmul(gelu_10, parameter_31, False, False)
        del parameter_31

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_120 = paddle._C_ops.add(matmul_109, parameter_30)
        del parameter_30

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_66, dropout_67 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_120, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_120

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_121 = paddle._C_ops.add(dropout_66, layer_norm_63)

        # pd_op.layer_norm: (1x11x768xf32, 1x11xf32, 1x11xf32) <- (1x11x768xf32, 768xf32, 768xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_121, parameter_45, parameter_44, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_44, parameter_45

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_110 = paddle._C_ops.matmul(layer_norm_66, parameter_27, False, False)
        del parameter_27

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_122 = paddle._C_ops.add(matmul_110, parameter_26)
        del parameter_26

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_111 = paddle._C_ops.matmul(layer_norm_66, parameter_25, False, False)
        del parameter_25

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_123 = paddle._C_ops.add(matmul_111, parameter_24)
        del parameter_24

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_112 = paddle._C_ops.matmul(layer_norm_66, parameter_23, False, False)
        del parameter_23

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_124 = paddle._C_ops.add(matmul_112, parameter_22)
        del parameter_22

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_88 = paddle._C_ops.reshape(add_122, full_int_array_2)

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_110 = paddle._C_ops.transpose(reshape_88, [0, 2, 1, 3])
        del reshape_88

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_89 = paddle._C_ops.reshape(add_123, full_int_array_2)

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_111 = paddle._C_ops.transpose(reshape_89, [0, 2, 1, 3])
        del reshape_89

        # pd_op.reshape: (1x11x12x64xf32) <- (1x11x768xf32, 4xi64)
        reshape_90 = paddle._C_ops.reshape(add_124, full_int_array_2)
        del full_int_array_2

        # pd_op.transpose: (1x12x11x64xf32) <- (1x11x12x64xf32)
        transpose_112 = paddle._C_ops.transpose(reshape_90, [0, 2, 1, 3])
        del reshape_90

        # pd_op.transpose: (1x12x64x11xf32) <- (1x12x11x64xf32)
        transpose_113 = paddle._C_ops.transpose(transpose_111, [0, 1, 3, 2])
        del transpose_111

        # pd_op.matmul: (1x12x11x11xf32) <- (1x12x11x64xf32, 1x12x64x11xf32)
        matmul_113 = paddle._C_ops.matmul(transpose_110, transpose_113, False, False)

        # pd_op.share_data_: (512x512x64xf32) <- (512x512x64xf32)
        share_data__11 = parameter_0.detach()

        # pd_op.assign: (512x512x64xf32) <- (512x512x64xf32)
        assign_70 = share_data__11
        del share_data__11

        # pd_op.slice: (11x11x64xf32) <- (512x512x64xf32, 2xi64, 2xi64)
        slice_22 = paddle._C_ops.slice(
            assign_70, [0, 1], full_int_array_3, full_int_array_4, [1, 1], []
        )
        del assign_70

        # pd_op.transpose: (11x1x12x64xf32) <- (1x12x11x64xf32)
        transpose_114 = paddle._C_ops.transpose(transpose_110, [2, 0, 1, 3])

        # pd_op.reshape: (11x12x64xf32) <- (11x1x12x64xf32, 3xi64)
        reshape_91 = paddle._C_ops.reshape(transpose_114, full_int_array_5)
        del full_int_array_5

        # pd_op.transpose: (11x64x11xf32) <- (11x11x64xf32)
        transpose_115 = paddle._C_ops.transpose(slice_22, [0, 2, 1])
        del slice_22

        # pd_op.matmul: (11x12x11xf32) <- (11x12x64xf32, 11x64x11xf32)
        matmul_114 = paddle._C_ops.matmul(reshape_91, transpose_115, False, False)

        # pd_op.reshape: (11x1x12x11xf32) <- (11x12x11xf32, 4xi64)
        reshape_92 = paddle._C_ops.reshape(matmul_114, full_int_array_6)
        del full_int_array_6

        # pd_op.transpose: (1x12x11x11xf32) <- (11x1x12x11xf32)
        transpose_116 = paddle._C_ops.transpose(reshape_92, [1, 2, 0, 3])
        del reshape_92

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x12x11x11xf32)
        add_125 = paddle._C_ops.add(matmul_113, transpose_116)

        # pd_op.scale: (1x12x11x11xf32) <- (1x12x11x11xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(add_125, full_6, float("0"), True)
        del add_125

        # pd_op.add: (1x12x11x11xf32) <- (1x12x11x11xf32, 1x1x1x11xf32)
        add_126 = paddle._C_ops.add(scale_13, scale_1)

        # pd_op.softmax: (1x12x11x11xf32) <- (1x12x11x11xf32)
        softmax_11 = paddle._C_ops.softmax(add_126, -1)
        del add_126

        # pd_op.dropout: (1x12x11x11xf32, 1x12x11x11xui8) <- (1x12x11x11xf32, None, 1xf32)
        dropout_68, dropout_69 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_11, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (1x12x11x64xf32) <- (1x12x11x11xf32, 1x12x11x64xf32)
        matmul_115 = paddle._C_ops.matmul(dropout_68, transpose_112, False, False)

        # pd_op.assign: (512x512x64xf32) <- (512x512x64xf32)
        assign_71 = parameter_0
        del parameter_0

        # pd_op.slice: (11x11x64xf32) <- (512x512x64xf32, 2xi64, 2xi64)
        slice_23 = paddle._C_ops.slice(
            assign_71, [0, 1], full_int_array_3, full_int_array_4, [1, 1], []
        )
        del assign_71, full_int_array_3, full_int_array_4

        # pd_op.transpose: (11x1x12x11xf32) <- (1x12x11x11xf32)
        transpose_117 = paddle._C_ops.transpose(dropout_68, [2, 0, 1, 3])

        # pd_op.reshape: (11x12x11xf32) <- (11x1x12x11xf32, 3xi64)
        reshape_93 = paddle._C_ops.reshape(transpose_117, full_int_array_7)
        del full_int_array_7

        # pd_op.matmul: (11x12x64xf32) <- (11x12x11xf32, 11x11x64xf32)
        matmul_116 = paddle._C_ops.matmul(reshape_93, slice_23, False, False)

        # pd_op.reshape: (11x1x12x64xf32) <- (11x12x64xf32, 4xi64)
        reshape_94 = paddle._C_ops.reshape(matmul_116, full_int_array_8)
        del full_int_array_8

        # pd_op.transpose: (1x12x11x64xf32) <- (11x1x12x64xf32)
        transpose_118 = paddle._C_ops.transpose(reshape_94, [1, 2, 0, 3])
        del reshape_94

        # pd_op.add: (1x12x11x64xf32) <- (1x12x11x64xf32, 1x12x11x64xf32)
        add_127 = paddle._C_ops.add(matmul_115, transpose_118)

        # pd_op.transpose: (1x11x12x64xf32) <- (1x12x11x64xf32)
        transpose_119 = paddle._C_ops.transpose(add_127, [0, 2, 1, 3])
        del add_127

        # pd_op.reshape: (1x11x768xf32) <- (1x11x12x64xf32, 3xi64)
        reshape_95 = paddle._C_ops.reshape(transpose_119, full_int_array_9)
        del full_int_array_9

        # pd_op.matmul: (1x11x768xf32) <- (1x11x768xf32, 768x768xf32)
        matmul_117 = paddle._C_ops.matmul(reshape_95, parameter_21, False, False)
        del parameter_21

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_128 = paddle._C_ops.add(matmul_117, parameter_20)
        del parameter_20

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_70, dropout_71 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_128, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_128

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_129 = paddle._C_ops.add(layer_norm_66, dropout_70)

        # pd_op.layer_norm: (1x11x768xf32, 1x11xf32, 1x11xf32) <- (1x11x768xf32, 768xf32, 768xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_129, parameter_19, parameter_18, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_18, parameter_19

        # pd_op.matmul: (1x11x3072xf32) <- (1x11x768xf32, 768x3072xf32)
        matmul_118 = paddle._C_ops.matmul(layer_norm_69, parameter_17, False, False)
        del parameter_17

        # pd_op.add: (1x11x3072xf32) <- (1x11x3072xf32, 3072xf32)
        add_130 = paddle._C_ops.add(matmul_118, parameter_16)
        del parameter_16

        # pd_op.gelu: (1x11x3072xf32) <- (1x11x3072xf32)
        gelu_11 = paddle._C_ops.gelu(add_130, False)

        # pd_op.matmul: (1x11x768xf32) <- (1x11x3072xf32, 3072x768xf32)
        matmul_119 = paddle._C_ops.matmul(gelu_11, parameter_15, False, False)
        del parameter_15

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 768xf32)
        add_131 = paddle._C_ops.add(matmul_119, parameter_14)
        del parameter_14

        # pd_op.dropout: (1x11x768xf32, 1x11x768xui8) <- (1x11x768xf32, None, 1xf32)
        dropout_72, dropout_73 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_131, None, full_5, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_131

        # pd_op.add: (1x11x768xf32) <- (1x11x768xf32, 1x11x768xf32)
        add_132 = paddle._C_ops.add(dropout_72, layer_norm_69)

        # pd_op.layer_norm: (1x11x768xf32, 1x11xf32, 1x11xf32) <- (1x11x768xf32, 768xf32, 768xf32)
        layer_norm_72, layer_norm_73, layer_norm_74 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_132, parameter_29, parameter_28, float("1e-12"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_28, parameter_29

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_10 = [0]

        # pd_op.slice: (1x768xf32) <- (1x11x768xf32, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(
            layer_norm_72, [1], full_int_array_10, full_int_array_0, [1], [1]
        )
        del full_int_array_0

        # pd_op.matmul: (1x768xf32) <- (1x768xf32, 768x768xf32)
        matmul_120 = paddle._C_ops.matmul(slice_24, parameter_13, False, False)
        del parameter_13

        # pd_op.add: (1x768xf32) <- (1x768xf32, 768xf32)
        add_133 = paddle._C_ops.add(matmul_120, parameter_12)
        del parameter_12

        # pd_op.tanh: (1x768xf32) <- (1x768xf32)
        tanh_0 = paddle._C_ops.tanh(add_133)
        del (
            add_0,
            add_1,
            add_100,
            add_101,
            add_102,
            add_107,
            add_108,
            add_11,
            add_110,
            add_111,
            add_112,
            add_113,
            add_118,
            add_119,
            add_12,
            add_121,
            add_122,
            add_123,
            add_124,
            add_129,
            add_13,
            add_130,
            add_132,
            add_133,
            add_14,
            add_19,
            add_2,
            add_20,
            add_22,
            add_23,
            add_24,
            add_25,
            add_3,
            add_30,
            add_31,
            add_33,
            add_34,
            add_35,
            add_36,
            add_41,
            add_42,
            add_44,
            add_45,
            add_46,
            add_47,
            add_52,
            add_53,
            add_55,
            add_56,
            add_57,
            add_58,
            add_63,
            add_64,
            add_66,
            add_67,
            add_68,
            add_69,
            add_74,
            add_75,
            add_77,
            add_78,
            add_79,
            add_8,
            add_80,
            add_85,
            add_86,
            add_88,
            add_89,
            add_9,
            add_90,
            add_91,
            add_96,
            add_97,
            add_99,
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
            full_5,
            full_6,
            full_int_array_10,
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
            matmul_105,
            matmul_106,
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
            matmul_116,
            matmul_117,
            matmul_118,
            matmul_119,
            matmul_12,
            matmul_120,
            matmul_13,
            matmul_14,
            matmul_15,
            matmul_16,
            matmul_17,
            matmul_18,
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
            matmul_28,
            matmul_29,
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
            matmul_39,
            matmul_4,
            matmul_40,
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
            matmul_50,
            matmul_51,
            matmul_52,
            matmul_53,
            matmul_54,
            matmul_55,
            matmul_56,
            matmul_57,
            matmul_58,
            matmul_59,
            matmul_6,
            matmul_60,
            matmul_61,
            matmul_62,
            matmul_63,
            matmul_64,
            matmul_65,
            matmul_66,
            matmul_67,
            matmul_68,
            matmul_69,
            matmul_7,
            matmul_70,
            matmul_71,
            matmul_72,
            matmul_73,
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
            matmul_83,
            matmul_84,
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
            matmul_94,
            matmul_95,
            matmul_96,
            matmul_97,
            matmul_98,
            matmul_99,
            reshape_11,
            reshape_13,
            reshape_15,
            reshape_19,
            reshape_21,
            reshape_23,
            reshape_27,
            reshape_29,
            reshape_3,
            reshape_31,
            reshape_35,
            reshape_37,
            reshape_39,
            reshape_43,
            reshape_45,
            reshape_47,
            reshape_5,
            reshape_51,
            reshape_53,
            reshape_55,
            reshape_59,
            reshape_61,
            reshape_63,
            reshape_67,
            reshape_69,
            reshape_7,
            reshape_71,
            reshape_75,
            reshape_77,
            reshape_79,
            reshape_83,
            reshape_85,
            reshape_87,
            reshape_91,
            reshape_93,
            reshape_95,
            scale_1,
            scale_10,
            scale_11,
            scale_12,
            scale_13,
            scale_2,
            scale_3,
            scale_4,
            scale_5,
            scale_6,
            scale_7,
            scale_8,
            scale_9,
            slice_1,
            slice_11,
            slice_13,
            slice_15,
            slice_17,
            slice_19,
            slice_21,
            slice_23,
            slice_24,
            slice_3,
            slice_5,
            slice_7,
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
            transpose_0,
            transpose_10,
            transpose_100,
            transpose_102,
            transpose_103,
            transpose_104,
            transpose_105,
            transpose_106,
            transpose_107,
            transpose_108,
            transpose_109,
            transpose_110,
            transpose_112,
            transpose_113,
            transpose_114,
            transpose_115,
            transpose_116,
            transpose_117,
            transpose_118,
            transpose_119,
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
            transpose_72,
            transpose_73,
            transpose_74,
            transpose_75,
            transpose_76,
            transpose_77,
            transpose_78,
            transpose_79,
            transpose_8,
            transpose_80,
            transpose_82,
            transpose_83,
            transpose_84,
            transpose_85,
            transpose_86,
            transpose_87,
            transpose_88,
            transpose_89,
            transpose_9,
            transpose_90,
            transpose_92,
            transpose_93,
            transpose_94,
            transpose_95,
            transpose_96,
            transpose_97,
            transpose_98,
            transpose_99,
        )

        return tanh_0
