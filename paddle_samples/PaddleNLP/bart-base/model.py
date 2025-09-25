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
        data_0,
    ):
        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (1x21xi64) <- (1x21xi64, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            data_0, full_0, paddle.int64, paddle.framework._current_expected_place()
        )
        del full_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [-1]

        # pd_op.slice: (1x20xi64) <- (1x21xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_0, [1], full_int_array_0, full_int_array_1, [1], []
        )

        # pd_op.assign: (1x20xi64) <- (1x20xi64)
        assign_0 = slice_0
        del slice_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [2147483647]

        # pd_op.set_value_with_tensor_: (1x21xi64) <- (1x21xi64, 1x20xi64, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__0 = paddle._C_ops.set_value_with_tensor_(
            full_like_0,
            assign_0,
            full_int_array_2,
            full_int_array_3,
            full_int_array_2,
            [1],
            [],
            [],
        )
        del assign_0, full_like_0

        # pd_op.set_value_: (1x21xi64) <- (1x21xi64, 1xi64, 1xi64, 1xi64)
        set_value__0 = paddle._C_ops.set_value_(
            set_value_with_tensor__0,
            full_int_array_0,
            full_int_array_2,
            full_int_array_2,
            [1],
            [1],
            [],
            [1],
            [float("2")],
        )
        del full_int_array_0, full_int_array_2, set_value_with_tensor__0

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.equal: (1x21xb) <- (1x21xi64, xi64)
        equal_0 = paddle._C_ops.equal(data_0, full_1)
        del full_1

        # pd_op.cast: (1x21xf32) <- (1x21xb)
        cast_0 = paddle._C_ops.cast(equal_0, paddle.float32)
        del equal_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [1, 2]

        # pd_op.unsqueeze: (1x1x1x21xf32) <- (1x21xf32, 2xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(cast_0, full_int_array_4)
        del cast_0, full_int_array_4

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("-10000"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x1x1x21xf32) <- (1x1x1x21xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(unsqueeze_0, full_2, float("0"), True)
        del full_2, unsqueeze_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [-1, 21]

        # pd_op.reshape: (1x21xi64) <- (1x21xi64, 2xi64)
        reshape_0 = paddle._C_ops.reshape(data_0, full_int_array_5)
        del data_0

        # pd_op.embedding: (1x21x768xf32) <- (1x21xi64, 50265x768xf32)
        embedding_0 = paddle._C_ops.embedding(reshape_0, parameter_258, -1, False)
        del reshape_0

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x21x768xf32) <- (1x21x768xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(embedding_0, full_3, float("0"), True)
        del embedding_0

        # pd_op.full: (1xf64) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("21"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (21xi64) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_4, full_5, full_6, dtype="int64")
        del full_4, full_5, full_6

        # pd_op.scale: (21xi64) <- (21xi64, 1xf32)
        scale_2 = paddle._C_ops.scale(arange_0, full_3, float("2"), True)
        del arange_0

        # pd_op.embedding: (21x768xf32) <- (21xi64, 1026x768xf32)
        embedding_1 = paddle._C_ops.embedding(scale_2, parameter_257, -1, False)
        del parameter_257

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 21x768xf32)
        add_0 = paddle._C_ops.add(scale_1, embedding_1)
        del embedding_1, scale_1

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_1, layer_norm_2, layer_norm_3 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_0, parameter_256, parameter_255, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_0, parameter_255, parameter_256

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                layer_norm_1, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del layer_norm_1

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_0 = paddle._C_ops.matmul(dropout_0, parameter_254, False, False)
        del parameter_254

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_1 = paddle._C_ops.add(matmul_0, parameter_253)
        del matmul_0, parameter_253

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_6 = [0, 0, 12, 64]

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(add_1, full_int_array_6)
        del add_1

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_1, [0, 2, 1, 3])
        del reshape_1

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_1 = paddle._C_ops.matmul(dropout_0, parameter_252, False, False)
        del parameter_252

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_2 = paddle._C_ops.add(matmul_1, parameter_251)
        del matmul_1, parameter_251

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_2 = paddle._C_ops.matmul(dropout_0, parameter_250, False, False)
        del parameter_250

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_3 = paddle._C_ops.add(matmul_2, parameter_249)
        del matmul_2, parameter_249

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(add_2, full_int_array_6)
        del add_2

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_2, [0, 2, 1, 3])
        del reshape_2

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(add_3, full_int_array_6)
        del add_3

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_2 = paddle._C_ops.transpose(reshape_3, [0, 2, 1, 3])
        del reshape_3

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x12x21x64xf32) <- (1x12x21x64xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(transpose_0, full_8, float("0"), True)
        del transpose_0

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x21x64xf32)
        matmul_3 = paddle._C_ops.matmul(scale_3, transpose_1, False, True)
        del scale_3, transpose_1

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_4 = paddle._C_ops.add(matmul_3, scale_0)
        del matmul_3

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_0 = paddle._C_ops.softmax(add_4, -1)
        del add_4

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_0, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_0

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_4 = paddle._C_ops.matmul(dropout_2, transpose_2, False, False)
        del dropout_2, transpose_2

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_3 = paddle._C_ops.transpose(matmul_4, [0, 2, 1, 3])
        del matmul_4

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_7 = [0, 0, 768]

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_4 = paddle._C_ops.reshape(transpose_3, full_int_array_7)
        del transpose_3

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_5 = paddle._C_ops.matmul(reshape_4, parameter_248, False, False)
        del parameter_248, reshape_4

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_5 = paddle._C_ops.add(matmul_5, parameter_247)
        del matmul_5, parameter_247

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_5, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_5

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_6 = paddle._C_ops.add(dropout_0, dropout_4)
        del dropout_0, dropout_4

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_4, layer_norm_5, layer_norm_6 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_6, parameter_242, parameter_241, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_6, parameter_241, parameter_242

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_6 = paddle._C_ops.matmul(layer_norm_4, parameter_246, False, False)
        del parameter_246

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_7 = paddle._C_ops.add(matmul_6, parameter_245)
        del matmul_6, parameter_245

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_0 = paddle._C_ops.gelu(add_7, False)
        del add_7

        # pd_op.dropout: (1x21x3072xf32, 1x21x3072xui8) <- (1x21x3072xf32, None, 1xf32)
        dropout_6, dropout_7 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                gelu_0, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del gelu_0

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_7 = paddle._C_ops.matmul(dropout_6, parameter_244, False, False)
        del dropout_6, parameter_244

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_8 = paddle._C_ops.add(matmul_7, parameter_243)
        del matmul_7, parameter_243

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_8, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_8

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_9 = paddle._C_ops.add(layer_norm_4, dropout_8)
        del dropout_8, layer_norm_4

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_7, layer_norm_8, layer_norm_9 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_9, parameter_240, parameter_239, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_9, parameter_239, parameter_240

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_8 = paddle._C_ops.matmul(layer_norm_7, parameter_238, False, False)
        del parameter_238

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_10 = paddle._C_ops.add(matmul_8, parameter_237)
        del matmul_8, parameter_237

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(add_10, full_int_array_6)
        del add_10

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_5, [0, 2, 1, 3])
        del reshape_5

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_9 = paddle._C_ops.matmul(layer_norm_7, parameter_236, False, False)
        del parameter_236

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_11 = paddle._C_ops.add(matmul_9, parameter_235)
        del matmul_9, parameter_235

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_7, parameter_234, False, False)
        del parameter_234

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_12 = paddle._C_ops.add(matmul_10, parameter_233)
        del matmul_10, parameter_233

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(add_11, full_int_array_6)
        del add_11

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_5 = paddle._C_ops.transpose(reshape_6, [0, 2, 1, 3])
        del reshape_6

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(add_12, full_int_array_6)
        del add_12

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_7, [0, 2, 1, 3])
        del reshape_7

        # pd_op.scale: (1x12x21x64xf32) <- (1x12x21x64xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(transpose_4, full_8, float("0"), True)
        del transpose_4

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x21x64xf32)
        matmul_11 = paddle._C_ops.matmul(scale_4, transpose_5, False, True)
        del scale_4, transpose_5

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_13 = paddle._C_ops.add(matmul_11, scale_0)
        del matmul_11

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_1 = paddle._C_ops.softmax(add_13, -1)
        del add_13

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_10, dropout_11 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_1, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_1

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_12 = paddle._C_ops.matmul(dropout_10, transpose_6, False, False)
        del dropout_10, transpose_6

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_7 = paddle._C_ops.transpose(matmul_12, [0, 2, 1, 3])
        del matmul_12

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_8 = paddle._C_ops.reshape(transpose_7, full_int_array_7)
        del transpose_7

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_13 = paddle._C_ops.matmul(reshape_8, parameter_232, False, False)
        del parameter_232, reshape_8

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_14 = paddle._C_ops.add(matmul_13, parameter_231)
        del matmul_13, parameter_231

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_12, dropout_13 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_14, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_14

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_15 = paddle._C_ops.add(layer_norm_7, dropout_12)
        del dropout_12, layer_norm_7

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_10, layer_norm_11, layer_norm_12 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_15, parameter_226, parameter_225, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_15, parameter_225, parameter_226

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_14 = paddle._C_ops.matmul(layer_norm_10, parameter_230, False, False)
        del parameter_230

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_16 = paddle._C_ops.add(matmul_14, parameter_229)
        del matmul_14, parameter_229

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_1 = paddle._C_ops.gelu(add_16, False)
        del add_16

        # pd_op.dropout: (1x21x3072xf32, 1x21x3072xui8) <- (1x21x3072xf32, None, 1xf32)
        dropout_14, dropout_15 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                gelu_1, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del gelu_1

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_15 = paddle._C_ops.matmul(dropout_14, parameter_228, False, False)
        del dropout_14, parameter_228

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_17 = paddle._C_ops.add(matmul_15, parameter_227)
        del matmul_15, parameter_227

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_16, dropout_17 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_17, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_17

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_18 = paddle._C_ops.add(layer_norm_10, dropout_16)
        del dropout_16, layer_norm_10

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_13, layer_norm_14, layer_norm_15 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_18, parameter_224, parameter_223, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_18, parameter_223, parameter_224

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_16 = paddle._C_ops.matmul(layer_norm_13, parameter_222, False, False)
        del parameter_222

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_19 = paddle._C_ops.add(matmul_16, parameter_221)
        del matmul_16, parameter_221

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(add_19, full_int_array_6)
        del add_19

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_8 = paddle._C_ops.transpose(reshape_9, [0, 2, 1, 3])
        del reshape_9

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_17 = paddle._C_ops.matmul(layer_norm_13, parameter_220, False, False)
        del parameter_220

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_20 = paddle._C_ops.add(matmul_17, parameter_219)
        del matmul_17, parameter_219

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_18 = paddle._C_ops.matmul(layer_norm_13, parameter_218, False, False)
        del parameter_218

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_21 = paddle._C_ops.add(matmul_18, parameter_217)
        del matmul_18, parameter_217

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(add_20, full_int_array_6)
        del add_20

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_9 = paddle._C_ops.transpose(reshape_10, [0, 2, 1, 3])
        del reshape_10

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_11 = paddle._C_ops.reshape(add_21, full_int_array_6)
        del add_21

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_10 = paddle._C_ops.transpose(reshape_11, [0, 2, 1, 3])
        del reshape_11

        # pd_op.scale: (1x12x21x64xf32) <- (1x12x21x64xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(transpose_8, full_8, float("0"), True)
        del transpose_8

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x21x64xf32)
        matmul_19 = paddle._C_ops.matmul(scale_5, transpose_9, False, True)
        del scale_5, transpose_9

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_22 = paddle._C_ops.add(matmul_19, scale_0)
        del matmul_19

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_2 = paddle._C_ops.softmax(add_22, -1)
        del add_22

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_18, dropout_19 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_2, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_2

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_20 = paddle._C_ops.matmul(dropout_18, transpose_10, False, False)
        del dropout_18, transpose_10

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_11 = paddle._C_ops.transpose(matmul_20, [0, 2, 1, 3])
        del matmul_20

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_12 = paddle._C_ops.reshape(transpose_11, full_int_array_7)
        del transpose_11

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_21 = paddle._C_ops.matmul(reshape_12, parameter_216, False, False)
        del parameter_216, reshape_12

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_23 = paddle._C_ops.add(matmul_21, parameter_215)
        del matmul_21, parameter_215

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_20, dropout_21 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_23, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_23

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_24 = paddle._C_ops.add(layer_norm_13, dropout_20)
        del dropout_20, layer_norm_13

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_16, layer_norm_17, layer_norm_18 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_24, parameter_210, parameter_209, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_24, parameter_209, parameter_210

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_22 = paddle._C_ops.matmul(layer_norm_16, parameter_214, False, False)
        del parameter_214

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_25 = paddle._C_ops.add(matmul_22, parameter_213)
        del matmul_22, parameter_213

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_2 = paddle._C_ops.gelu(add_25, False)
        del add_25

        # pd_op.dropout: (1x21x3072xf32, 1x21x3072xui8) <- (1x21x3072xf32, None, 1xf32)
        dropout_22, dropout_23 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                gelu_2, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del gelu_2

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_23 = paddle._C_ops.matmul(dropout_22, parameter_212, False, False)
        del dropout_22, parameter_212

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_26 = paddle._C_ops.add(matmul_23, parameter_211)
        del matmul_23, parameter_211

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_24, dropout_25 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_26, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_26

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_27 = paddle._C_ops.add(layer_norm_16, dropout_24)
        del dropout_24, layer_norm_16

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_19, layer_norm_20, layer_norm_21 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_27, parameter_208, parameter_207, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_27, parameter_207, parameter_208

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_24 = paddle._C_ops.matmul(layer_norm_19, parameter_206, False, False)
        del parameter_206

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_28 = paddle._C_ops.add(matmul_24, parameter_205)
        del matmul_24, parameter_205

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(add_28, full_int_array_6)
        del add_28

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_12 = paddle._C_ops.transpose(reshape_13, [0, 2, 1, 3])
        del reshape_13

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_25 = paddle._C_ops.matmul(layer_norm_19, parameter_204, False, False)
        del parameter_204

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_29 = paddle._C_ops.add(matmul_25, parameter_203)
        del matmul_25, parameter_203

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_26 = paddle._C_ops.matmul(layer_norm_19, parameter_202, False, False)
        del parameter_202

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_30 = paddle._C_ops.add(matmul_26, parameter_201)
        del matmul_26, parameter_201

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(add_29, full_int_array_6)
        del add_29

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_14, [0, 2, 1, 3])
        del reshape_14

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_15 = paddle._C_ops.reshape(add_30, full_int_array_6)
        del add_30

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_14 = paddle._C_ops.transpose(reshape_15, [0, 2, 1, 3])
        del reshape_15

        # pd_op.scale: (1x12x21x64xf32) <- (1x12x21x64xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(transpose_12, full_8, float("0"), True)
        del transpose_12

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x21x64xf32)
        matmul_27 = paddle._C_ops.matmul(scale_6, transpose_13, False, True)
        del scale_6, transpose_13

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_31 = paddle._C_ops.add(matmul_27, scale_0)
        del matmul_27

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_3 = paddle._C_ops.softmax(add_31, -1)
        del add_31

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_26, dropout_27 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_3, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_3

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_28 = paddle._C_ops.matmul(dropout_26, transpose_14, False, False)
        del dropout_26, transpose_14

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_15 = paddle._C_ops.transpose(matmul_28, [0, 2, 1, 3])
        del matmul_28

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_16 = paddle._C_ops.reshape(transpose_15, full_int_array_7)
        del transpose_15

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_29 = paddle._C_ops.matmul(reshape_16, parameter_200, False, False)
        del parameter_200, reshape_16

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_32 = paddle._C_ops.add(matmul_29, parameter_199)
        del matmul_29, parameter_199

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_28, dropout_29 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_32, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_32

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_33 = paddle._C_ops.add(layer_norm_19, dropout_28)
        del dropout_28, layer_norm_19

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_22, layer_norm_23, layer_norm_24 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_33, parameter_194, parameter_193, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_33, parameter_193, parameter_194

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_30 = paddle._C_ops.matmul(layer_norm_22, parameter_198, False, False)
        del parameter_198

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_34 = paddle._C_ops.add(matmul_30, parameter_197)
        del matmul_30, parameter_197

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_3 = paddle._C_ops.gelu(add_34, False)
        del add_34

        # pd_op.dropout: (1x21x3072xf32, 1x21x3072xui8) <- (1x21x3072xf32, None, 1xf32)
        dropout_30, dropout_31 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                gelu_3, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del gelu_3

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_31 = paddle._C_ops.matmul(dropout_30, parameter_196, False, False)
        del dropout_30, parameter_196

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_35 = paddle._C_ops.add(matmul_31, parameter_195)
        del matmul_31, parameter_195

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_32, dropout_33 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_35, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_35

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_36 = paddle._C_ops.add(layer_norm_22, dropout_32)
        del dropout_32, layer_norm_22

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_25, layer_norm_26, layer_norm_27 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_36, parameter_192, parameter_191, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_36, parameter_191, parameter_192

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_32 = paddle._C_ops.matmul(layer_norm_25, parameter_190, False, False)
        del parameter_190

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_37 = paddle._C_ops.add(matmul_32, parameter_189)
        del matmul_32, parameter_189

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_17 = paddle._C_ops.reshape(add_37, full_int_array_6)
        del add_37

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_16 = paddle._C_ops.transpose(reshape_17, [0, 2, 1, 3])
        del reshape_17

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_33 = paddle._C_ops.matmul(layer_norm_25, parameter_188, False, False)
        del parameter_188

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_38 = paddle._C_ops.add(matmul_33, parameter_187)
        del matmul_33, parameter_187

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_34 = paddle._C_ops.matmul(layer_norm_25, parameter_186, False, False)
        del parameter_186

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_39 = paddle._C_ops.add(matmul_34, parameter_185)
        del matmul_34, parameter_185

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_18 = paddle._C_ops.reshape(add_38, full_int_array_6)
        del add_38

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_17 = paddle._C_ops.transpose(reshape_18, [0, 2, 1, 3])
        del reshape_18

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_19 = paddle._C_ops.reshape(add_39, full_int_array_6)
        del add_39

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_18 = paddle._C_ops.transpose(reshape_19, [0, 2, 1, 3])
        del reshape_19

        # pd_op.scale: (1x12x21x64xf32) <- (1x12x21x64xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(transpose_16, full_8, float("0"), True)
        del transpose_16

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x21x64xf32)
        matmul_35 = paddle._C_ops.matmul(scale_7, transpose_17, False, True)
        del scale_7, transpose_17

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_40 = paddle._C_ops.add(matmul_35, scale_0)
        del matmul_35

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_4 = paddle._C_ops.softmax(add_40, -1)
        del add_40

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_34, dropout_35 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_4, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_4

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_36 = paddle._C_ops.matmul(dropout_34, transpose_18, False, False)
        del dropout_34, transpose_18

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_19 = paddle._C_ops.transpose(matmul_36, [0, 2, 1, 3])
        del matmul_36

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_20 = paddle._C_ops.reshape(transpose_19, full_int_array_7)
        del transpose_19

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_37 = paddle._C_ops.matmul(reshape_20, parameter_184, False, False)
        del parameter_184, reshape_20

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_41 = paddle._C_ops.add(matmul_37, parameter_183)
        del matmul_37, parameter_183

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_36, dropout_37 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_41, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_41

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_42 = paddle._C_ops.add(layer_norm_25, dropout_36)
        del dropout_36, layer_norm_25

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_28, layer_norm_29, layer_norm_30 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_42, parameter_178, parameter_177, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_42, parameter_177, parameter_178

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_38 = paddle._C_ops.matmul(layer_norm_28, parameter_182, False, False)
        del parameter_182

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_43 = paddle._C_ops.add(matmul_38, parameter_181)
        del matmul_38, parameter_181

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_4 = paddle._C_ops.gelu(add_43, False)
        del add_43

        # pd_op.dropout: (1x21x3072xf32, 1x21x3072xui8) <- (1x21x3072xf32, None, 1xf32)
        dropout_38, dropout_39 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                gelu_4, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del gelu_4

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_39 = paddle._C_ops.matmul(dropout_38, parameter_180, False, False)
        del dropout_38, parameter_180

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_44 = paddle._C_ops.add(matmul_39, parameter_179)
        del matmul_39, parameter_179

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_40, dropout_41 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_44, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_44

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_45 = paddle._C_ops.add(layer_norm_28, dropout_40)
        del dropout_40, layer_norm_28

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_31, layer_norm_32, layer_norm_33 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_45, parameter_176, parameter_175, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_45, parameter_175, parameter_176

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_40 = paddle._C_ops.matmul(layer_norm_31, parameter_174, False, False)
        del parameter_174

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_46 = paddle._C_ops.add(matmul_40, parameter_173)
        del matmul_40, parameter_173

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_21 = paddle._C_ops.reshape(add_46, full_int_array_6)
        del add_46

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_20 = paddle._C_ops.transpose(reshape_21, [0, 2, 1, 3])
        del reshape_21

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_41 = paddle._C_ops.matmul(layer_norm_31, parameter_172, False, False)
        del parameter_172

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_47 = paddle._C_ops.add(matmul_41, parameter_171)
        del matmul_41, parameter_171

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_42 = paddle._C_ops.matmul(layer_norm_31, parameter_170, False, False)
        del parameter_170

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_48 = paddle._C_ops.add(matmul_42, parameter_169)
        del matmul_42, parameter_169

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_22 = paddle._C_ops.reshape(add_47, full_int_array_6)
        del add_47

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_21 = paddle._C_ops.transpose(reshape_22, [0, 2, 1, 3])
        del reshape_22

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_23 = paddle._C_ops.reshape(add_48, full_int_array_6)
        del add_48

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_22 = paddle._C_ops.transpose(reshape_23, [0, 2, 1, 3])
        del reshape_23

        # pd_op.scale: (1x12x21x64xf32) <- (1x12x21x64xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(transpose_20, full_8, float("0"), True)
        del transpose_20

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x21x64xf32)
        matmul_43 = paddle._C_ops.matmul(scale_8, transpose_21, False, True)
        del scale_8, transpose_21

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_49 = paddle._C_ops.add(matmul_43, scale_0)
        del matmul_43

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_5 = paddle._C_ops.softmax(add_49, -1)
        del add_49

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_42, dropout_43 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_5, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_5

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_44 = paddle._C_ops.matmul(dropout_42, transpose_22, False, False)
        del dropout_42, transpose_22

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_23 = paddle._C_ops.transpose(matmul_44, [0, 2, 1, 3])
        del matmul_44

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_24 = paddle._C_ops.reshape(transpose_23, full_int_array_7)
        del transpose_23

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_45 = paddle._C_ops.matmul(reshape_24, parameter_168, False, False)
        del parameter_168, reshape_24

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_50 = paddle._C_ops.add(matmul_45, parameter_167)
        del matmul_45, parameter_167

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_44, dropout_45 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_50, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_50

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_51 = paddle._C_ops.add(layer_norm_31, dropout_44)
        del dropout_44, layer_norm_31

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_34, layer_norm_35, layer_norm_36 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_51, parameter_162, parameter_161, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_51, parameter_161, parameter_162

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_46 = paddle._C_ops.matmul(layer_norm_34, parameter_166, False, False)
        del parameter_166

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_52 = paddle._C_ops.add(matmul_46, parameter_165)
        del matmul_46, parameter_165

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_5 = paddle._C_ops.gelu(add_52, False)
        del add_52

        # pd_op.dropout: (1x21x3072xf32, 1x21x3072xui8) <- (1x21x3072xf32, None, 1xf32)
        dropout_46, dropout_47 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                gelu_5, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del gelu_5

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_47 = paddle._C_ops.matmul(dropout_46, parameter_164, False, False)
        del dropout_46, parameter_164

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_53 = paddle._C_ops.add(matmul_47, parameter_163)
        del matmul_47, parameter_163

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_48, dropout_49 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_53, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_53

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_54 = paddle._C_ops.add(layer_norm_34, dropout_48)
        del dropout_48, layer_norm_34

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_37, layer_norm_38, layer_norm_39 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_54, parameter_160, parameter_159, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_54, parameter_159, parameter_160

        # pd_op.slice: (1x1x1x21xf32) <- (1x1x1x21xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            scale_0, [2], full_int_array_1, full_int_array_3, [1], []
        )
        del full_int_array_1, full_int_array_3, scale_0

        # pd_op.reshape: (1x21xi64) <- (1x21xi64, 2xi64)
        reshape_25 = paddle._C_ops.reshape(set_value__0, full_int_array_5)
        del full_int_array_5, set_value__0

        # pd_op.full: (21x21xf32) <- ()
        full_9 = paddle._C_ops.full(
            [21, 21],
            float("-inf"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.triu: (21x21xf32) <- (21x21xf32)
        triu_0 = paddle._C_ops.triu(full_9, 1)
        del full_9

        # pd_op.embedding: (1x21x768xf32) <- (1x21xi64, 50265x768xf32)
        embedding_2 = paddle._C_ops.embedding(reshape_25, parameter_258, -1, False)
        del parameter_258, reshape_25

        # pd_op.scale: (1x21x768xf32) <- (1x21x768xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(embedding_2, full_3, float("0"), True)
        del embedding_2, full_3

        # pd_op.embedding: (21x768xf32) <- (21xi64, 1026x768xf32)
        embedding_3 = paddle._C_ops.embedding(scale_2, parameter_158, -1, False)
        del parameter_158, scale_2

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 21x768xf32)
        add_55 = paddle._C_ops.add(scale_9, embedding_3)
        del embedding_3, scale_9

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_40, layer_norm_41, layer_norm_42 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_55, parameter_157, parameter_156, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_55, parameter_156, parameter_157

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_50, dropout_51 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                layer_norm_40, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del layer_norm_40

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_48 = paddle._C_ops.matmul(dropout_50, parameter_155, False, False)
        del parameter_155

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_56 = paddle._C_ops.add(matmul_48, parameter_154)
        del matmul_48, parameter_154

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_26 = paddle._C_ops.reshape(add_56, full_int_array_6)
        del add_56

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_24 = paddle._C_ops.transpose(reshape_26, [0, 2, 1, 3])
        del reshape_26

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_49 = paddle._C_ops.matmul(dropout_50, parameter_153, False, False)
        del parameter_153

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_57 = paddle._C_ops.add(matmul_49, parameter_152)
        del matmul_49, parameter_152

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_50 = paddle._C_ops.matmul(dropout_50, parameter_151, False, False)
        del parameter_151

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_58 = paddle._C_ops.add(matmul_50, parameter_150)
        del matmul_50, parameter_150

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_27 = paddle._C_ops.reshape(add_57, full_int_array_6)
        del add_57

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_25 = paddle._C_ops.transpose(reshape_27, [0, 2, 1, 3])
        del reshape_27

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_28 = paddle._C_ops.reshape(add_58, full_int_array_6)
        del add_58

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_26 = paddle._C_ops.transpose(reshape_28, [0, 2, 1, 3])
        del reshape_28

        # pd_op.scale: (1x12x21x64xf32) <- (1x12x21x64xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(transpose_24, full_8, float("0"), True)
        del transpose_24

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x21x64xf32)
        matmul_51 = paddle._C_ops.matmul(scale_10, transpose_25, False, True)
        del scale_10, transpose_25

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 21x21xf32)
        add_59 = paddle._C_ops.add(matmul_51, triu_0)
        del matmul_51

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_6 = paddle._C_ops.softmax(add_59, -1)
        del add_59

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_52, dropout_53 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_6, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_6

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_52 = paddle._C_ops.matmul(dropout_52, transpose_26, False, False)
        del dropout_52, transpose_26

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_27 = paddle._C_ops.transpose(matmul_52, [0, 2, 1, 3])
        del matmul_52

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_29 = paddle._C_ops.reshape(transpose_27, full_int_array_7)
        del transpose_27

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_53 = paddle._C_ops.matmul(reshape_29, parameter_149, False, False)
        del parameter_149, reshape_29

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_60 = paddle._C_ops.add(matmul_53, parameter_148)
        del matmul_53, parameter_148

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_54, dropout_55 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_60, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_60

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_61 = paddle._C_ops.add(dropout_50, dropout_54)
        del dropout_50, dropout_54

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_43, layer_norm_44, layer_norm_45 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_61, parameter_135, parameter_134, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_61, parameter_134, parameter_135

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_54 = paddle._C_ops.matmul(layer_norm_43, parameter_147, False, False)
        del parameter_147

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_62 = paddle._C_ops.add(matmul_54, parameter_146)
        del matmul_54, parameter_146

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_30 = paddle._C_ops.reshape(add_62, full_int_array_6)
        del add_62

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_28 = paddle._C_ops.transpose(reshape_30, [0, 2, 1, 3])
        del reshape_30

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_55 = paddle._C_ops.matmul(layer_norm_37, parameter_145, False, False)
        del parameter_145

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_63 = paddle._C_ops.add(matmul_55, parameter_144)
        del matmul_55, parameter_144

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_56 = paddle._C_ops.matmul(layer_norm_37, parameter_143, False, False)
        del parameter_143

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_64 = paddle._C_ops.add(matmul_56, parameter_142)
        del matmul_56, parameter_142

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_31 = paddle._C_ops.reshape(add_63, full_int_array_6)
        del add_63

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_29 = paddle._C_ops.transpose(reshape_31, [0, 2, 1, 3])
        del reshape_31

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_32 = paddle._C_ops.reshape(add_64, full_int_array_6)
        del add_64

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_30 = paddle._C_ops.transpose(reshape_32, [0, 2, 1, 3])
        del reshape_32

        # pd_op.scale: (1x12x21x64xf32) <- (1x12x21x64xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(transpose_28, full_8, float("0"), True)
        del transpose_28

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x21x64xf32)
        matmul_57 = paddle._C_ops.matmul(scale_11, transpose_29, False, True)
        del scale_11, transpose_29

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_65 = paddle._C_ops.add(matmul_57, slice_1)
        del matmul_57

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_7 = paddle._C_ops.softmax(add_65, -1)
        del add_65

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_56, dropout_57 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_7, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_7

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_58 = paddle._C_ops.matmul(dropout_56, transpose_30, False, False)
        del dropout_56, transpose_30

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_31 = paddle._C_ops.transpose(matmul_58, [0, 2, 1, 3])
        del matmul_58

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_33 = paddle._C_ops.reshape(transpose_31, full_int_array_7)
        del transpose_31

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_59 = paddle._C_ops.matmul(reshape_33, parameter_141, False, False)
        del parameter_141, reshape_33

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_66 = paddle._C_ops.add(matmul_59, parameter_140)
        del matmul_59, parameter_140

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_58, dropout_59 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_66, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_66

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_67 = paddle._C_ops.add(layer_norm_43, dropout_58)
        del dropout_58, layer_norm_43

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_46, layer_norm_47, layer_norm_48 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_67, parameter_133, parameter_132, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_67, parameter_132, parameter_133

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_60 = paddle._C_ops.matmul(layer_norm_46, parameter_139, False, False)
        del parameter_139

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_68 = paddle._C_ops.add(matmul_60, parameter_138)
        del matmul_60, parameter_138

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_6 = paddle._C_ops.gelu(add_68, False)
        del add_68

        # pd_op.dropout: (1x21x3072xf32, 1x21x3072xui8) <- (1x21x3072xf32, None, 1xf32)
        dropout_60, dropout_61 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                gelu_6, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del gelu_6

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_61 = paddle._C_ops.matmul(dropout_60, parameter_137, False, False)
        del dropout_60, parameter_137

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_69 = paddle._C_ops.add(matmul_61, parameter_136)
        del matmul_61, parameter_136

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_62, dropout_63 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_69, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_69

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_70 = paddle._C_ops.add(layer_norm_46, dropout_62)
        del dropout_62, layer_norm_46

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_49, layer_norm_50, layer_norm_51 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_70, parameter_131, parameter_130, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_70, parameter_130, parameter_131

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_62 = paddle._C_ops.matmul(layer_norm_49, parameter_129, False, False)
        del parameter_129

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_71 = paddle._C_ops.add(matmul_62, parameter_128)
        del matmul_62, parameter_128

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_34 = paddle._C_ops.reshape(add_71, full_int_array_6)
        del add_71

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_32 = paddle._C_ops.transpose(reshape_34, [0, 2, 1, 3])
        del reshape_34

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_63 = paddle._C_ops.matmul(layer_norm_49, parameter_127, False, False)
        del parameter_127

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_72 = paddle._C_ops.add(matmul_63, parameter_126)
        del matmul_63, parameter_126

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_64 = paddle._C_ops.matmul(layer_norm_49, parameter_125, False, False)
        del parameter_125

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_73 = paddle._C_ops.add(matmul_64, parameter_124)
        del matmul_64, parameter_124

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_35 = paddle._C_ops.reshape(add_72, full_int_array_6)
        del add_72

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_33 = paddle._C_ops.transpose(reshape_35, [0, 2, 1, 3])
        del reshape_35

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_36 = paddle._C_ops.reshape(add_73, full_int_array_6)
        del add_73

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_34 = paddle._C_ops.transpose(reshape_36, [0, 2, 1, 3])
        del reshape_36

        # pd_op.scale: (1x12x21x64xf32) <- (1x12x21x64xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(transpose_32, full_8, float("0"), True)
        del transpose_32

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x21x64xf32)
        matmul_65 = paddle._C_ops.matmul(scale_12, transpose_33, False, True)
        del scale_12, transpose_33

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 21x21xf32)
        add_74 = paddle._C_ops.add(matmul_65, triu_0)
        del matmul_65

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_8 = paddle._C_ops.softmax(add_74, -1)
        del add_74

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_64, dropout_65 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_8, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_8

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_66 = paddle._C_ops.matmul(dropout_64, transpose_34, False, False)
        del dropout_64, transpose_34

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_35 = paddle._C_ops.transpose(matmul_66, [0, 2, 1, 3])
        del matmul_66

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_37 = paddle._C_ops.reshape(transpose_35, full_int_array_7)
        del transpose_35

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_67 = paddle._C_ops.matmul(reshape_37, parameter_123, False, False)
        del parameter_123, reshape_37

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_75 = paddle._C_ops.add(matmul_67, parameter_122)
        del matmul_67, parameter_122

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_66, dropout_67 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_75, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_75

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_76 = paddle._C_ops.add(layer_norm_49, dropout_66)
        del dropout_66, layer_norm_49

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_52, layer_norm_53, layer_norm_54 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_76, parameter_109, parameter_108, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_76, parameter_108, parameter_109

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_68 = paddle._C_ops.matmul(layer_norm_52, parameter_121, False, False)
        del parameter_121

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_77 = paddle._C_ops.add(matmul_68, parameter_120)
        del matmul_68, parameter_120

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_38 = paddle._C_ops.reshape(add_77, full_int_array_6)
        del add_77

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_36 = paddle._C_ops.transpose(reshape_38, [0, 2, 1, 3])
        del reshape_38

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_69 = paddle._C_ops.matmul(layer_norm_37, parameter_119, False, False)
        del parameter_119

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_78 = paddle._C_ops.add(matmul_69, parameter_118)
        del matmul_69, parameter_118

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_70 = paddle._C_ops.matmul(layer_norm_37, parameter_117, False, False)
        del parameter_117

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_79 = paddle._C_ops.add(matmul_70, parameter_116)
        del matmul_70, parameter_116

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_39 = paddle._C_ops.reshape(add_78, full_int_array_6)
        del add_78

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_37 = paddle._C_ops.transpose(reshape_39, [0, 2, 1, 3])
        del reshape_39

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_40 = paddle._C_ops.reshape(add_79, full_int_array_6)
        del add_79

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_38 = paddle._C_ops.transpose(reshape_40, [0, 2, 1, 3])
        del reshape_40

        # pd_op.scale: (1x12x21x64xf32) <- (1x12x21x64xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(transpose_36, full_8, float("0"), True)
        del transpose_36

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x21x64xf32)
        matmul_71 = paddle._C_ops.matmul(scale_13, transpose_37, False, True)
        del scale_13, transpose_37

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_80 = paddle._C_ops.add(matmul_71, slice_1)
        del matmul_71

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_9 = paddle._C_ops.softmax(add_80, -1)
        del add_80

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_68, dropout_69 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_9, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_9

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_72 = paddle._C_ops.matmul(dropout_68, transpose_38, False, False)
        del dropout_68, transpose_38

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_39 = paddle._C_ops.transpose(matmul_72, [0, 2, 1, 3])
        del matmul_72

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_41 = paddle._C_ops.reshape(transpose_39, full_int_array_7)
        del transpose_39

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_73 = paddle._C_ops.matmul(reshape_41, parameter_115, False, False)
        del parameter_115, reshape_41

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_81 = paddle._C_ops.add(matmul_73, parameter_114)
        del matmul_73, parameter_114

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_70, dropout_71 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_81, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_81

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_82 = paddle._C_ops.add(layer_norm_52, dropout_70)
        del dropout_70, layer_norm_52

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_55, layer_norm_56, layer_norm_57 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_82, parameter_107, parameter_106, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_82, parameter_106, parameter_107

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_74 = paddle._C_ops.matmul(layer_norm_55, parameter_113, False, False)
        del parameter_113

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_83 = paddle._C_ops.add(matmul_74, parameter_112)
        del matmul_74, parameter_112

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_7 = paddle._C_ops.gelu(add_83, False)
        del add_83

        # pd_op.dropout: (1x21x3072xf32, 1x21x3072xui8) <- (1x21x3072xf32, None, 1xf32)
        dropout_72, dropout_73 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                gelu_7, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del gelu_7

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_75 = paddle._C_ops.matmul(dropout_72, parameter_111, False, False)
        del dropout_72, parameter_111

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_84 = paddle._C_ops.add(matmul_75, parameter_110)
        del matmul_75, parameter_110

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_74, dropout_75 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_84, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_84

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_85 = paddle._C_ops.add(layer_norm_55, dropout_74)
        del dropout_74, layer_norm_55

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_58, layer_norm_59, layer_norm_60 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_85, parameter_105, parameter_104, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_85, parameter_104, parameter_105

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_76 = paddle._C_ops.matmul(layer_norm_58, parameter_103, False, False)
        del parameter_103

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_86 = paddle._C_ops.add(matmul_76, parameter_102)
        del matmul_76, parameter_102

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_42 = paddle._C_ops.reshape(add_86, full_int_array_6)
        del add_86

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_40 = paddle._C_ops.transpose(reshape_42, [0, 2, 1, 3])
        del reshape_42

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_77 = paddle._C_ops.matmul(layer_norm_58, parameter_101, False, False)
        del parameter_101

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_87 = paddle._C_ops.add(matmul_77, parameter_100)
        del matmul_77, parameter_100

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_78 = paddle._C_ops.matmul(layer_norm_58, parameter_99, False, False)
        del parameter_99

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_88 = paddle._C_ops.add(matmul_78, parameter_98)
        del matmul_78, parameter_98

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_43 = paddle._C_ops.reshape(add_87, full_int_array_6)
        del add_87

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_41 = paddle._C_ops.transpose(reshape_43, [0, 2, 1, 3])
        del reshape_43

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_44 = paddle._C_ops.reshape(add_88, full_int_array_6)
        del add_88

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_42 = paddle._C_ops.transpose(reshape_44, [0, 2, 1, 3])
        del reshape_44

        # pd_op.scale: (1x12x21x64xf32) <- (1x12x21x64xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(transpose_40, full_8, float("0"), True)
        del transpose_40

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x21x64xf32)
        matmul_79 = paddle._C_ops.matmul(scale_14, transpose_41, False, True)
        del scale_14, transpose_41

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 21x21xf32)
        add_89 = paddle._C_ops.add(matmul_79, triu_0)
        del matmul_79

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_10 = paddle._C_ops.softmax(add_89, -1)
        del add_89

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_76, dropout_77 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_10, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_10

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_80 = paddle._C_ops.matmul(dropout_76, transpose_42, False, False)
        del dropout_76, transpose_42

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_43 = paddle._C_ops.transpose(matmul_80, [0, 2, 1, 3])
        del matmul_80

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_45 = paddle._C_ops.reshape(transpose_43, full_int_array_7)
        del transpose_43

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_81 = paddle._C_ops.matmul(reshape_45, parameter_97, False, False)
        del parameter_97, reshape_45

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_90 = paddle._C_ops.add(matmul_81, parameter_96)
        del matmul_81, parameter_96

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_78, dropout_79 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_90, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_90

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_91 = paddle._C_ops.add(layer_norm_58, dropout_78)
        del dropout_78, layer_norm_58

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_61, layer_norm_62, layer_norm_63 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_91, parameter_83, parameter_82, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_91, parameter_82, parameter_83

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_82 = paddle._C_ops.matmul(layer_norm_61, parameter_95, False, False)
        del parameter_95

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_92 = paddle._C_ops.add(matmul_82, parameter_94)
        del matmul_82, parameter_94

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_46 = paddle._C_ops.reshape(add_92, full_int_array_6)
        del add_92

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_44 = paddle._C_ops.transpose(reshape_46, [0, 2, 1, 3])
        del reshape_46

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_83 = paddle._C_ops.matmul(layer_norm_37, parameter_93, False, False)
        del parameter_93

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_93 = paddle._C_ops.add(matmul_83, parameter_92)
        del matmul_83, parameter_92

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_84 = paddle._C_ops.matmul(layer_norm_37, parameter_91, False, False)
        del parameter_91

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_94 = paddle._C_ops.add(matmul_84, parameter_90)
        del matmul_84, parameter_90

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_47 = paddle._C_ops.reshape(add_93, full_int_array_6)
        del add_93

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_45 = paddle._C_ops.transpose(reshape_47, [0, 2, 1, 3])
        del reshape_47

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_48 = paddle._C_ops.reshape(add_94, full_int_array_6)
        del add_94

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_46 = paddle._C_ops.transpose(reshape_48, [0, 2, 1, 3])
        del reshape_48

        # pd_op.scale: (1x12x21x64xf32) <- (1x12x21x64xf32, 1xf32)
        scale_15 = paddle._C_ops.scale(transpose_44, full_8, float("0"), True)
        del transpose_44

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x21x64xf32)
        matmul_85 = paddle._C_ops.matmul(scale_15, transpose_45, False, True)
        del scale_15, transpose_45

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_95 = paddle._C_ops.add(matmul_85, slice_1)
        del matmul_85

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_11 = paddle._C_ops.softmax(add_95, -1)
        del add_95

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_80, dropout_81 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_11, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_11

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_86 = paddle._C_ops.matmul(dropout_80, transpose_46, False, False)
        del dropout_80, transpose_46

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_47 = paddle._C_ops.transpose(matmul_86, [0, 2, 1, 3])
        del matmul_86

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_49 = paddle._C_ops.reshape(transpose_47, full_int_array_7)
        del transpose_47

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_87 = paddle._C_ops.matmul(reshape_49, parameter_89, False, False)
        del parameter_89, reshape_49

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_96 = paddle._C_ops.add(matmul_87, parameter_88)
        del matmul_87, parameter_88

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_82, dropout_83 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_96, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_96

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_97 = paddle._C_ops.add(layer_norm_61, dropout_82)
        del dropout_82, layer_norm_61

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_64, layer_norm_65, layer_norm_66 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_97, parameter_81, parameter_80, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_97, parameter_80, parameter_81

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_88 = paddle._C_ops.matmul(layer_norm_64, parameter_87, False, False)
        del parameter_87

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_98 = paddle._C_ops.add(matmul_88, parameter_86)
        del matmul_88, parameter_86

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_8 = paddle._C_ops.gelu(add_98, False)
        del add_98

        # pd_op.dropout: (1x21x3072xf32, 1x21x3072xui8) <- (1x21x3072xf32, None, 1xf32)
        dropout_84, dropout_85 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                gelu_8, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del gelu_8

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_89 = paddle._C_ops.matmul(dropout_84, parameter_85, False, False)
        del dropout_84, parameter_85

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_99 = paddle._C_ops.add(matmul_89, parameter_84)
        del matmul_89, parameter_84

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_86, dropout_87 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_99, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_99

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_100 = paddle._C_ops.add(layer_norm_64, dropout_86)
        del dropout_86, layer_norm_64

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_67, layer_norm_68, layer_norm_69 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_100, parameter_79, parameter_78, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_100, parameter_78, parameter_79

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_90 = paddle._C_ops.matmul(layer_norm_67, parameter_77, False, False)
        del parameter_77

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_101 = paddle._C_ops.add(matmul_90, parameter_76)
        del matmul_90, parameter_76

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_50 = paddle._C_ops.reshape(add_101, full_int_array_6)
        del add_101

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_48 = paddle._C_ops.transpose(reshape_50, [0, 2, 1, 3])
        del reshape_50

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_91 = paddle._C_ops.matmul(layer_norm_67, parameter_75, False, False)
        del parameter_75

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_102 = paddle._C_ops.add(matmul_91, parameter_74)
        del matmul_91, parameter_74

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_92 = paddle._C_ops.matmul(layer_norm_67, parameter_73, False, False)
        del parameter_73

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_103 = paddle._C_ops.add(matmul_92, parameter_72)
        del matmul_92, parameter_72

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_51 = paddle._C_ops.reshape(add_102, full_int_array_6)
        del add_102

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_49 = paddle._C_ops.transpose(reshape_51, [0, 2, 1, 3])
        del reshape_51

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_52 = paddle._C_ops.reshape(add_103, full_int_array_6)
        del add_103

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_50 = paddle._C_ops.transpose(reshape_52, [0, 2, 1, 3])
        del reshape_52

        # pd_op.scale: (1x12x21x64xf32) <- (1x12x21x64xf32, 1xf32)
        scale_16 = paddle._C_ops.scale(transpose_48, full_8, float("0"), True)
        del transpose_48

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x21x64xf32)
        matmul_93 = paddle._C_ops.matmul(scale_16, transpose_49, False, True)
        del scale_16, transpose_49

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 21x21xf32)
        add_104 = paddle._C_ops.add(matmul_93, triu_0)
        del matmul_93

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_12 = paddle._C_ops.softmax(add_104, -1)
        del add_104

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_88, dropout_89 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_12, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_12

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_94 = paddle._C_ops.matmul(dropout_88, transpose_50, False, False)
        del dropout_88, transpose_50

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_51 = paddle._C_ops.transpose(matmul_94, [0, 2, 1, 3])
        del matmul_94

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_53 = paddle._C_ops.reshape(transpose_51, full_int_array_7)
        del transpose_51

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_95 = paddle._C_ops.matmul(reshape_53, parameter_71, False, False)
        del parameter_71, reshape_53

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_105 = paddle._C_ops.add(matmul_95, parameter_70)
        del matmul_95, parameter_70

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_90, dropout_91 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_105, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_105

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_106 = paddle._C_ops.add(layer_norm_67, dropout_90)
        del dropout_90, layer_norm_67

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_70, layer_norm_71, layer_norm_72 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_106, parameter_57, parameter_56, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_106, parameter_56, parameter_57

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_96 = paddle._C_ops.matmul(layer_norm_70, parameter_69, False, False)
        del parameter_69

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_107 = paddle._C_ops.add(matmul_96, parameter_68)
        del matmul_96, parameter_68

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_54 = paddle._C_ops.reshape(add_107, full_int_array_6)
        del add_107

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_52 = paddle._C_ops.transpose(reshape_54, [0, 2, 1, 3])
        del reshape_54

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_97 = paddle._C_ops.matmul(layer_norm_37, parameter_67, False, False)
        del parameter_67

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_108 = paddle._C_ops.add(matmul_97, parameter_66)
        del matmul_97, parameter_66

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_98 = paddle._C_ops.matmul(layer_norm_37, parameter_65, False, False)
        del parameter_65

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_109 = paddle._C_ops.add(matmul_98, parameter_64)
        del matmul_98, parameter_64

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_55 = paddle._C_ops.reshape(add_108, full_int_array_6)
        del add_108

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_53 = paddle._C_ops.transpose(reshape_55, [0, 2, 1, 3])
        del reshape_55

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_56 = paddle._C_ops.reshape(add_109, full_int_array_6)
        del add_109

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_54 = paddle._C_ops.transpose(reshape_56, [0, 2, 1, 3])
        del reshape_56

        # pd_op.scale: (1x12x21x64xf32) <- (1x12x21x64xf32, 1xf32)
        scale_17 = paddle._C_ops.scale(transpose_52, full_8, float("0"), True)
        del transpose_52

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x21x64xf32)
        matmul_99 = paddle._C_ops.matmul(scale_17, transpose_53, False, True)
        del scale_17, transpose_53

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_110 = paddle._C_ops.add(matmul_99, slice_1)
        del matmul_99

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_13 = paddle._C_ops.softmax(add_110, -1)
        del add_110

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_92, dropout_93 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_13, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_13

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_100 = paddle._C_ops.matmul(dropout_92, transpose_54, False, False)
        del dropout_92, transpose_54

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_55 = paddle._C_ops.transpose(matmul_100, [0, 2, 1, 3])
        del matmul_100

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_57 = paddle._C_ops.reshape(transpose_55, full_int_array_7)
        del transpose_55

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_101 = paddle._C_ops.matmul(reshape_57, parameter_63, False, False)
        del parameter_63, reshape_57

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_111 = paddle._C_ops.add(matmul_101, parameter_62)
        del matmul_101, parameter_62

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_94, dropout_95 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_111, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_111

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_112 = paddle._C_ops.add(layer_norm_70, dropout_94)
        del dropout_94, layer_norm_70

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_73, layer_norm_74, layer_norm_75 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_112, parameter_55, parameter_54, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_112, parameter_54, parameter_55

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_102 = paddle._C_ops.matmul(layer_norm_73, parameter_61, False, False)
        del parameter_61

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_113 = paddle._C_ops.add(matmul_102, parameter_60)
        del matmul_102, parameter_60

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_9 = paddle._C_ops.gelu(add_113, False)
        del add_113

        # pd_op.dropout: (1x21x3072xf32, 1x21x3072xui8) <- (1x21x3072xf32, None, 1xf32)
        dropout_96, dropout_97 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                gelu_9, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del gelu_9

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_103 = paddle._C_ops.matmul(dropout_96, parameter_59, False, False)
        del dropout_96, parameter_59

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_114 = paddle._C_ops.add(matmul_103, parameter_58)
        del matmul_103, parameter_58

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_98, dropout_99 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_114, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_114

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_115 = paddle._C_ops.add(layer_norm_73, dropout_98)
        del dropout_98, layer_norm_73

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_76, layer_norm_77, layer_norm_78 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_115, parameter_53, parameter_52, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_115, parameter_52, parameter_53

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_104 = paddle._C_ops.matmul(layer_norm_76, parameter_51, False, False)
        del parameter_51

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_116 = paddle._C_ops.add(matmul_104, parameter_50)
        del matmul_104, parameter_50

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_58 = paddle._C_ops.reshape(add_116, full_int_array_6)
        del add_116

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_56 = paddle._C_ops.transpose(reshape_58, [0, 2, 1, 3])
        del reshape_58

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_105 = paddle._C_ops.matmul(layer_norm_76, parameter_49, False, False)
        del parameter_49

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_117 = paddle._C_ops.add(matmul_105, parameter_48)
        del matmul_105, parameter_48

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_106 = paddle._C_ops.matmul(layer_norm_76, parameter_47, False, False)
        del parameter_47

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_118 = paddle._C_ops.add(matmul_106, parameter_46)
        del matmul_106, parameter_46

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_59 = paddle._C_ops.reshape(add_117, full_int_array_6)
        del add_117

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_57 = paddle._C_ops.transpose(reshape_59, [0, 2, 1, 3])
        del reshape_59

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_60 = paddle._C_ops.reshape(add_118, full_int_array_6)
        del add_118

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_58 = paddle._C_ops.transpose(reshape_60, [0, 2, 1, 3])
        del reshape_60

        # pd_op.scale: (1x12x21x64xf32) <- (1x12x21x64xf32, 1xf32)
        scale_18 = paddle._C_ops.scale(transpose_56, full_8, float("0"), True)
        del transpose_56

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x21x64xf32)
        matmul_107 = paddle._C_ops.matmul(scale_18, transpose_57, False, True)
        del scale_18, transpose_57

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 21x21xf32)
        add_119 = paddle._C_ops.add(matmul_107, triu_0)
        del matmul_107

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_14 = paddle._C_ops.softmax(add_119, -1)
        del add_119

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_100, dropout_101 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_14, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_14

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_108 = paddle._C_ops.matmul(dropout_100, transpose_58, False, False)
        del dropout_100, transpose_58

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_59 = paddle._C_ops.transpose(matmul_108, [0, 2, 1, 3])
        del matmul_108

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_61 = paddle._C_ops.reshape(transpose_59, full_int_array_7)
        del transpose_59

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_109 = paddle._C_ops.matmul(reshape_61, parameter_45, False, False)
        del parameter_45, reshape_61

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_120 = paddle._C_ops.add(matmul_109, parameter_44)
        del matmul_109, parameter_44

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_102, dropout_103 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_120, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_120

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_121 = paddle._C_ops.add(layer_norm_76, dropout_102)
        del dropout_102, layer_norm_76

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_79, layer_norm_80, layer_norm_81 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_121, parameter_31, parameter_30, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_121, parameter_30, parameter_31

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_110 = paddle._C_ops.matmul(layer_norm_79, parameter_43, False, False)
        del parameter_43

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_122 = paddle._C_ops.add(matmul_110, parameter_42)
        del matmul_110, parameter_42

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_62 = paddle._C_ops.reshape(add_122, full_int_array_6)
        del add_122

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_60 = paddle._C_ops.transpose(reshape_62, [0, 2, 1, 3])
        del reshape_62

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_111 = paddle._C_ops.matmul(layer_norm_37, parameter_41, False, False)
        del parameter_41

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_123 = paddle._C_ops.add(matmul_111, parameter_40)
        del matmul_111, parameter_40

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_112 = paddle._C_ops.matmul(layer_norm_37, parameter_39, False, False)
        del parameter_39

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_124 = paddle._C_ops.add(matmul_112, parameter_38)
        del matmul_112, parameter_38

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_63 = paddle._C_ops.reshape(add_123, full_int_array_6)
        del add_123

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_61 = paddle._C_ops.transpose(reshape_63, [0, 2, 1, 3])
        del reshape_63

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_64 = paddle._C_ops.reshape(add_124, full_int_array_6)
        del add_124

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_62 = paddle._C_ops.transpose(reshape_64, [0, 2, 1, 3])
        del reshape_64

        # pd_op.scale: (1x12x21x64xf32) <- (1x12x21x64xf32, 1xf32)
        scale_19 = paddle._C_ops.scale(transpose_60, full_8, float("0"), True)
        del transpose_60

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x21x64xf32)
        matmul_113 = paddle._C_ops.matmul(scale_19, transpose_61, False, True)
        del scale_19, transpose_61

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_125 = paddle._C_ops.add(matmul_113, slice_1)
        del matmul_113

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_15 = paddle._C_ops.softmax(add_125, -1)
        del add_125

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_104, dropout_105 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_15, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_15

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_114 = paddle._C_ops.matmul(dropout_104, transpose_62, False, False)
        del dropout_104, transpose_62

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_63 = paddle._C_ops.transpose(matmul_114, [0, 2, 1, 3])
        del matmul_114

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_65 = paddle._C_ops.reshape(transpose_63, full_int_array_7)
        del transpose_63

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_115 = paddle._C_ops.matmul(reshape_65, parameter_37, False, False)
        del parameter_37, reshape_65

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_126 = paddle._C_ops.add(matmul_115, parameter_36)
        del matmul_115, parameter_36

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_106, dropout_107 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_126, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_126

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_127 = paddle._C_ops.add(layer_norm_79, dropout_106)
        del dropout_106, layer_norm_79

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_82, layer_norm_83, layer_norm_84 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_127, parameter_29, parameter_28, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_127, parameter_28, parameter_29

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_116 = paddle._C_ops.matmul(layer_norm_82, parameter_35, False, False)
        del parameter_35

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_128 = paddle._C_ops.add(matmul_116, parameter_34)
        del matmul_116, parameter_34

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_10 = paddle._C_ops.gelu(add_128, False)
        del add_128

        # pd_op.dropout: (1x21x3072xf32, 1x21x3072xui8) <- (1x21x3072xf32, None, 1xf32)
        dropout_108, dropout_109 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                gelu_10, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del gelu_10

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_117 = paddle._C_ops.matmul(dropout_108, parameter_33, False, False)
        del dropout_108, parameter_33

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_129 = paddle._C_ops.add(matmul_117, parameter_32)
        del matmul_117, parameter_32

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_110, dropout_111 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_129, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_129

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_130 = paddle._C_ops.add(layer_norm_82, dropout_110)
        del dropout_110, layer_norm_82

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_85, layer_norm_86, layer_norm_87 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_130, parameter_27, parameter_26, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_130, parameter_26, parameter_27

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_118 = paddle._C_ops.matmul(layer_norm_85, parameter_25, False, False)
        del parameter_25

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_131 = paddle._C_ops.add(matmul_118, parameter_24)
        del matmul_118, parameter_24

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_66 = paddle._C_ops.reshape(add_131, full_int_array_6)
        del add_131

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_64 = paddle._C_ops.transpose(reshape_66, [0, 2, 1, 3])
        del reshape_66

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_119 = paddle._C_ops.matmul(layer_norm_85, parameter_23, False, False)
        del parameter_23

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_132 = paddle._C_ops.add(matmul_119, parameter_22)
        del matmul_119, parameter_22

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_120 = paddle._C_ops.matmul(layer_norm_85, parameter_21, False, False)
        del parameter_21

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_133 = paddle._C_ops.add(matmul_120, parameter_20)
        del matmul_120, parameter_20

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_67 = paddle._C_ops.reshape(add_132, full_int_array_6)
        del add_132

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_65 = paddle._C_ops.transpose(reshape_67, [0, 2, 1, 3])
        del reshape_67

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_68 = paddle._C_ops.reshape(add_133, full_int_array_6)
        del add_133

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_66 = paddle._C_ops.transpose(reshape_68, [0, 2, 1, 3])
        del reshape_68

        # pd_op.scale: (1x12x21x64xf32) <- (1x12x21x64xf32, 1xf32)
        scale_20 = paddle._C_ops.scale(transpose_64, full_8, float("0"), True)
        del transpose_64

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x21x64xf32)
        matmul_121 = paddle._C_ops.matmul(scale_20, transpose_65, False, True)
        del scale_20, transpose_65

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 21x21xf32)
        add_134 = paddle._C_ops.add(matmul_121, triu_0)
        del matmul_121, triu_0

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_16 = paddle._C_ops.softmax(add_134, -1)
        del add_134

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_112, dropout_113 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_16, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_16

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_122 = paddle._C_ops.matmul(dropout_112, transpose_66, False, False)
        del dropout_112, transpose_66

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_67 = paddle._C_ops.transpose(matmul_122, [0, 2, 1, 3])
        del matmul_122

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_69 = paddle._C_ops.reshape(transpose_67, full_int_array_7)
        del transpose_67

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_123 = paddle._C_ops.matmul(reshape_69, parameter_19, False, False)
        del parameter_19, reshape_69

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_135 = paddle._C_ops.add(matmul_123, parameter_18)
        del matmul_123, parameter_18

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_114, dropout_115 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_135, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_135

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_136 = paddle._C_ops.add(layer_norm_85, dropout_114)
        del dropout_114, layer_norm_85

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_88, layer_norm_89, layer_norm_90 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_136, parameter_5, parameter_4, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_136, parameter_4, parameter_5

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_124 = paddle._C_ops.matmul(layer_norm_88, parameter_17, False, False)
        del parameter_17

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_137 = paddle._C_ops.add(matmul_124, parameter_16)
        del matmul_124, parameter_16

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_70 = paddle._C_ops.reshape(add_137, full_int_array_6)
        del add_137

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_68 = paddle._C_ops.transpose(reshape_70, [0, 2, 1, 3])
        del reshape_70

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_125 = paddle._C_ops.matmul(layer_norm_37, parameter_15, False, False)
        del parameter_15

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_138 = paddle._C_ops.add(matmul_125, parameter_14)
        del matmul_125, parameter_14

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_126 = paddle._C_ops.matmul(layer_norm_37, parameter_13, False, False)
        del parameter_13

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_139 = paddle._C_ops.add(matmul_126, parameter_12)
        del matmul_126, parameter_12

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_71 = paddle._C_ops.reshape(add_138, full_int_array_6)
        del add_138

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_69 = paddle._C_ops.transpose(reshape_71, [0, 2, 1, 3])
        del reshape_71

        # pd_op.reshape: (1x21x12x64xf32) <- (1x21x768xf32, 4xi64)
        reshape_72 = paddle._C_ops.reshape(add_139, full_int_array_6)
        del add_139, full_int_array_6

        # pd_op.transpose: (1x12x21x64xf32) <- (1x21x12x64xf32)
        transpose_70 = paddle._C_ops.transpose(reshape_72, [0, 2, 1, 3])
        del reshape_72

        # pd_op.scale: (1x12x21x64xf32) <- (1x12x21x64xf32, 1xf32)
        scale_21 = paddle._C_ops.scale(transpose_68, full_8, float("0"), True)
        del full_8, transpose_68

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x21x64xf32)
        matmul_127 = paddle._C_ops.matmul(scale_21, transpose_69, False, True)
        del scale_21, transpose_69

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x1x21xf32)
        add_140 = paddle._C_ops.add(matmul_127, slice_1)
        del matmul_127, slice_1

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_17 = paddle._C_ops.softmax(add_140, -1)
        del add_140

        # pd_op.dropout: (1x12x21x21xf32, 1x12x21x21xui8) <- (1x12x21x21xf32, None, 1xf32)
        dropout_116, dropout_117 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_17, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_17

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_128 = paddle._C_ops.matmul(dropout_116, transpose_70, False, False)
        del dropout_116, transpose_70

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_71 = paddle._C_ops.transpose(matmul_128, [0, 2, 1, 3])
        del matmul_128

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_73 = paddle._C_ops.reshape(transpose_71, full_int_array_7)
        del full_int_array_7, transpose_71

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_129 = paddle._C_ops.matmul(reshape_73, parameter_11, False, False)
        del parameter_11, reshape_73

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_141 = paddle._C_ops.add(matmul_129, parameter_10)
        del matmul_129, parameter_10

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_118, dropout_119 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_141, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_141

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_142 = paddle._C_ops.add(layer_norm_88, dropout_118)
        del dropout_118, layer_norm_88

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_91, layer_norm_92, layer_norm_93 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_142, parameter_3, parameter_2, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_142, parameter_2, parameter_3

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_130 = paddle._C_ops.matmul(layer_norm_91, parameter_9, False, False)
        del parameter_9

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_143 = paddle._C_ops.add(matmul_130, parameter_8)
        del matmul_130, parameter_8

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_11 = paddle._C_ops.gelu(add_143, False)
        del add_143

        # pd_op.dropout: (1x21x3072xf32, 1x21x3072xui8) <- (1x21x3072xf32, None, 1xf32)
        dropout_120, dropout_121 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                gelu_11, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del gelu_11

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_131 = paddle._C_ops.matmul(dropout_120, parameter_7, False, False)
        del dropout_120, parameter_7

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_144 = paddle._C_ops.add(matmul_131, parameter_6)
        del matmul_131, parameter_6

        # pd_op.dropout: (1x21x768xf32, 1x21x768xui8) <- (1x21x768xf32, None, 1xf32)
        dropout_122, dropout_123 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_144, None, full_7, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_144, full_7

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_145 = paddle._C_ops.add(layer_norm_91, dropout_122)
        del dropout_122, layer_norm_91

        # pd_op.layer_norm: (1x21x768xf32, 1x21xf32, 1x21xf32) <- (1x21x768xf32, 768xf32, 768xf32)
        layer_norm_0, layer_norm_94, layer_norm_95 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_145, parameter_1, parameter_0, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_145, layer_norm_37, parameter_0, parameter_1

        return layer_norm_0
