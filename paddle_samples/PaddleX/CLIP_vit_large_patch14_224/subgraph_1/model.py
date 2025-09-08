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
    ):
        # pd_op.conv2d: (-1x1024x16x16xf32) <- (-1x3x224x224xf32, 1024x3x14x14xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_3, parameter_294, [14, 14], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_3, parameter_294

        # pd_op.shape64: (4xi64) <- (-1x1024x16x16xf32)
        shape64_0 = paddle._C_ops.shape64(conv2d_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del full_int_array_0, full_int_array_1, shape64_0

        # pd_op.flatten: (-1x1024x256xf32) <- (-1x1024x16x16xf32)
        flatten_0 = paddle._C_ops.flatten(conv2d_0, 2, 3)
        del conv2d_0

        # pd_op.transpose: (-1x256x1024xf32) <- (-1x1024x256xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_0, [0, 2, 1])
        del flatten_0

        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("-1"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("-1"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_0 = [data_2, full_0, full_1]
        del data_2, full_0, full_1

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.expand: (-1x1x1024xf32) <- (1x1x1024xf32, 3xi64)
        expand_0 = paddle._C_ops.expand(data_0, stack_0)
        del data_0, stack_0

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x1x1024xf32, -1x256x1024xf32]) <- (-1x1x1024xf32, -1x256x1024xf32)
        combine_1 = [expand_0, transpose_0]
        del expand_0, transpose_0

        # pd_op.concat: (-1x257x1024xf32) <- ([-1x1x1024xf32, -1x256x1024xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_1, full_2)
        del combine_1, full_2

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1x257x1024xf32)
        add_1 = paddle._C_ops.add(concat_0, data_1)
        del concat_0, data_1

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_1, parameter_293, parameter_292, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_1, parameter_292, parameter_293

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                layer_norm_0, parameter_291, parameter_290, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_290, parameter_291

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_1 = paddle._C_ops.shape64(layer_norm_3)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del full_int_array_2, full_int_array_3, shape64_1

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_0 = paddle._C_ops.matmul(layer_norm_3, parameter_289, False, False)
        del layer_norm_3, parameter_289

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_2 = paddle._C_ops.add(matmul_0, parameter_288)
        del matmul_0, parameter_288

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_4 = [-1, 257, 3, 16, 64]

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_0 = paddle._C_ops.reshape(add_2, full_int_array_4)
        del add_2, full_int_array_4

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_0, [2, 0, 3, 1, 4])
        del reshape_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [1]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            transpose_1, [0], full_int_array_5, full_int_array_6, [1], [0]
        )
        del full_int_array_5, full_int_array_6

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [2]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            transpose_1, [0], full_int_array_7, full_int_array_8, [1], [0]
        )
        del full_int_array_7, full_int_array_8

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_10 = [3]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            transpose_1, [0], full_int_array_9, full_int_array_10, [1], [0]
        )
        del full_int_array_10, full_int_array_9, transpose_1

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_2 = paddle._C_ops.transpose(slice_3, [0, 1, 3, 2])
        del slice_3

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_1 = paddle._C_ops.matmul(slice_2, transpose_2, False, False)
        del slice_2, transpose_2

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(matmul_1, full_3, float("0"), True)
        del full_3, matmul_1

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_0 = paddle._C_ops.softmax(scale_0, -1)
        del scale_0

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_2 = paddle._C_ops.matmul(softmax_0, slice_4, False, False)
        del slice_4, softmax_0

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_3 = paddle._C_ops.transpose(matmul_2, [0, 2, 1, 3])
        del matmul_2

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_11 = [-1, 257, 1024]

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_1 = paddle._C_ops.reshape(transpose_3, full_int_array_11)
        del full_int_array_11, transpose_3

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_3 = paddle._C_ops.matmul(reshape_1, parameter_287, False, False)
        del parameter_287, reshape_1

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_3 = paddle._C_ops.add(matmul_3, parameter_286)
        del matmul_3, parameter_286

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_4 = paddle._C_ops.add(layer_norm_0, add_3)
        del add_3, layer_norm_0

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_4, parameter_285, parameter_284, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_284, parameter_285

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_4 = paddle._C_ops.matmul(layer_norm_6, parameter_283, False, False)
        del layer_norm_6, parameter_283

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_5 = paddle._C_ops.add(matmul_4, parameter_282)
        del matmul_4, parameter_282

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_0 = paddle._C_ops.gelu(add_5, False)
        del add_5

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_5 = paddle._C_ops.matmul(gelu_0, parameter_281, False, False)
        del gelu_0, parameter_281

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_6 = paddle._C_ops.add(matmul_5, parameter_280)
        del matmul_5, parameter_280

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_7 = paddle._C_ops.add(add_4, add_6)
        del add_4, add_6

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_7, parameter_279, parameter_278, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_278, parameter_279

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_2 = paddle._C_ops.shape64(layer_norm_9)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_12 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_13 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_12, full_int_array_13, [1], [0]
        )
        del full_int_array_12, full_int_array_13, shape64_2

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_6 = paddle._C_ops.matmul(layer_norm_9, parameter_277, False, False)
        del layer_norm_9, parameter_277

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_8 = paddle._C_ops.add(matmul_6, parameter_276)
        del matmul_6, parameter_276

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_14 = [-1, 257, 3, 16, 64]

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_2 = paddle._C_ops.reshape(add_8, full_int_array_14)
        del add_8, full_int_array_14

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_2, [2, 0, 3, 1, 4])
        del reshape_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_15 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_16 = [1]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            transpose_4, [0], full_int_array_15, full_int_array_16, [1], [0]
        )
        del full_int_array_15, full_int_array_16

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_17 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_18 = [2]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            transpose_4, [0], full_int_array_17, full_int_array_18, [1], [0]
        )
        del full_int_array_17, full_int_array_18

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_19 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_20 = [3]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            transpose_4, [0], full_int_array_19, full_int_array_20, [1], [0]
        )
        del full_int_array_19, full_int_array_20, transpose_4

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_5 = paddle._C_ops.transpose(slice_7, [0, 1, 3, 2])
        del slice_7

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_7 = paddle._C_ops.matmul(slice_6, transpose_5, False, False)
        del slice_6, transpose_5

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(matmul_7, full_4, float("0"), True)
        del full_4, matmul_7

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_1 = paddle._C_ops.softmax(scale_1, -1)
        del scale_1

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_8 = paddle._C_ops.matmul(softmax_1, slice_8, False, False)
        del slice_8, softmax_1

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_6 = paddle._C_ops.transpose(matmul_8, [0, 2, 1, 3])
        del matmul_8

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_21 = [-1, 257, 1024]

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_6, full_int_array_21)
        del full_int_array_21, transpose_6

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_9 = paddle._C_ops.matmul(reshape_3, parameter_275, False, False)
        del parameter_275, reshape_3

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_9 = paddle._C_ops.add(matmul_9, parameter_274)
        del matmul_9, parameter_274

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_10 = paddle._C_ops.add(add_7, add_9)
        del add_7, add_9

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_10, parameter_273, parameter_272, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_272, parameter_273

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_12, parameter_271, False, False)
        del layer_norm_12, parameter_271

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_11 = paddle._C_ops.add(matmul_10, parameter_270)
        del matmul_10, parameter_270

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_1 = paddle._C_ops.gelu(add_11, False)
        del add_11

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_11 = paddle._C_ops.matmul(gelu_1, parameter_269, False, False)
        del gelu_1, parameter_269

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_12 = paddle._C_ops.add(matmul_11, parameter_268)
        del matmul_11, parameter_268

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_13 = paddle._C_ops.add(add_10, add_12)
        del add_10, add_12

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_13, parameter_267, parameter_266, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_266, parameter_267

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_3 = paddle._C_ops.shape64(layer_norm_15)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_22 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_23 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_22, full_int_array_23, [1], [0]
        )
        del full_int_array_22, full_int_array_23, shape64_3

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_12 = paddle._C_ops.matmul(layer_norm_15, parameter_265, False, False)
        del layer_norm_15, parameter_265

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_14 = paddle._C_ops.add(matmul_12, parameter_264)
        del matmul_12, parameter_264

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_24 = [-1, 257, 3, 16, 64]

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_4 = paddle._C_ops.reshape(add_14, full_int_array_24)
        del add_14, full_int_array_24

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_7 = paddle._C_ops.transpose(reshape_4, [2, 0, 3, 1, 4])
        del reshape_4

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_25 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_26 = [1]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            transpose_7, [0], full_int_array_25, full_int_array_26, [1], [0]
        )
        del full_int_array_25, full_int_array_26

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_27 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_28 = [2]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            transpose_7, [0], full_int_array_27, full_int_array_28, [1], [0]
        )
        del full_int_array_27, full_int_array_28

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_29 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_30 = [3]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            transpose_7, [0], full_int_array_29, full_int_array_30, [1], [0]
        )
        del full_int_array_29, full_int_array_30, transpose_7

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_8 = paddle._C_ops.transpose(slice_11, [0, 1, 3, 2])
        del slice_11

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_13 = paddle._C_ops.matmul(slice_10, transpose_8, False, False)
        del slice_10, transpose_8

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(matmul_13, full_5, float("0"), True)
        del full_5, matmul_13

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_2 = paddle._C_ops.softmax(scale_2, -1)
        del scale_2

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_14 = paddle._C_ops.matmul(softmax_2, slice_12, False, False)
        del slice_12, softmax_2

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_9 = paddle._C_ops.transpose(matmul_14, [0, 2, 1, 3])
        del matmul_14

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_31 = [-1, 257, 1024]

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_5 = paddle._C_ops.reshape(transpose_9, full_int_array_31)
        del full_int_array_31, transpose_9

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_15 = paddle._C_ops.matmul(reshape_5, parameter_263, False, False)
        del parameter_263, reshape_5

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_15 = paddle._C_ops.add(matmul_15, parameter_262)
        del matmul_15, parameter_262

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_16 = paddle._C_ops.add(add_13, add_15)
        del add_13, add_15

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_16, parameter_261, parameter_260, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_260, parameter_261

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_16 = paddle._C_ops.matmul(layer_norm_18, parameter_259, False, False)
        del layer_norm_18, parameter_259

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_17 = paddle._C_ops.add(matmul_16, parameter_258)
        del matmul_16, parameter_258

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_2 = paddle._C_ops.gelu(add_17, False)
        del add_17

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_17 = paddle._C_ops.matmul(gelu_2, parameter_257, False, False)
        del gelu_2, parameter_257

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_18 = paddle._C_ops.add(matmul_17, parameter_256)
        del matmul_17, parameter_256

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_19 = paddle._C_ops.add(add_16, add_18)
        del add_16, add_18

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_19, parameter_255, parameter_254, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_254, parameter_255

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_4 = paddle._C_ops.shape64(layer_norm_21)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_32 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_33 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            shape64_4, [0], full_int_array_32, full_int_array_33, [1], [0]
        )
        del full_int_array_32, full_int_array_33, shape64_4

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_18 = paddle._C_ops.matmul(layer_norm_21, parameter_253, False, False)
        del layer_norm_21, parameter_253

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_20 = paddle._C_ops.add(matmul_18, parameter_252)
        del matmul_18, parameter_252

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_34 = [-1, 257, 3, 16, 64]

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_6 = paddle._C_ops.reshape(add_20, full_int_array_34)
        del add_20, full_int_array_34

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_10 = paddle._C_ops.transpose(reshape_6, [2, 0, 3, 1, 4])
        del reshape_6

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_35 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_36 = [1]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            transpose_10, [0], full_int_array_35, full_int_array_36, [1], [0]
        )
        del full_int_array_35, full_int_array_36

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_37 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_38 = [2]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            transpose_10, [0], full_int_array_37, full_int_array_38, [1], [0]
        )
        del full_int_array_37, full_int_array_38

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_39 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_40 = [3]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            transpose_10, [0], full_int_array_39, full_int_array_40, [1], [0]
        )
        del full_int_array_39, full_int_array_40, transpose_10

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_11 = paddle._C_ops.transpose(slice_15, [0, 1, 3, 2])
        del slice_15

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_19 = paddle._C_ops.matmul(slice_14, transpose_11, False, False)
        del slice_14, transpose_11

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(matmul_19, full_6, float("0"), True)
        del full_6, matmul_19

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_3 = paddle._C_ops.softmax(scale_3, -1)
        del scale_3

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_20 = paddle._C_ops.matmul(softmax_3, slice_16, False, False)
        del slice_16, softmax_3

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_12 = paddle._C_ops.transpose(matmul_20, [0, 2, 1, 3])
        del matmul_20

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_41 = [-1, 257, 1024]

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_12, full_int_array_41)
        del full_int_array_41, transpose_12

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_21 = paddle._C_ops.matmul(reshape_7, parameter_251, False, False)
        del parameter_251, reshape_7

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_21 = paddle._C_ops.add(matmul_21, parameter_250)
        del matmul_21, parameter_250

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_22 = paddle._C_ops.add(add_19, add_21)
        del add_19, add_21

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_22, parameter_249, parameter_248, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_248, parameter_249

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_22 = paddle._C_ops.matmul(layer_norm_24, parameter_247, False, False)
        del layer_norm_24, parameter_247

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_23 = paddle._C_ops.add(matmul_22, parameter_246)
        del matmul_22, parameter_246

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_3 = paddle._C_ops.gelu(add_23, False)
        del add_23

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_23 = paddle._C_ops.matmul(gelu_3, parameter_245, False, False)
        del gelu_3, parameter_245

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_24 = paddle._C_ops.add(matmul_23, parameter_244)
        del matmul_23, parameter_244

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_25 = paddle._C_ops.add(add_22, add_24)
        del add_22, add_24

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_25, parameter_243, parameter_242, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_242, parameter_243

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_5 = paddle._C_ops.shape64(layer_norm_27)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_42 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_43 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            shape64_5, [0], full_int_array_42, full_int_array_43, [1], [0]
        )
        del full_int_array_42, full_int_array_43, shape64_5

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_24 = paddle._C_ops.matmul(layer_norm_27, parameter_241, False, False)
        del layer_norm_27, parameter_241

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_26 = paddle._C_ops.add(matmul_24, parameter_240)
        del matmul_24, parameter_240

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_44 = [-1, 257, 3, 16, 64]

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_8 = paddle._C_ops.reshape(add_26, full_int_array_44)
        del add_26, full_int_array_44

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_8, [2, 0, 3, 1, 4])
        del reshape_8

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_45 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_46 = [1]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(
            transpose_13, [0], full_int_array_45, full_int_array_46, [1], [0]
        )
        del full_int_array_45, full_int_array_46

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_47 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_48 = [2]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(
            transpose_13, [0], full_int_array_47, full_int_array_48, [1], [0]
        )
        del full_int_array_47, full_int_array_48

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_49 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_50 = [3]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(
            transpose_13, [0], full_int_array_49, full_int_array_50, [1], [0]
        )
        del full_int_array_49, full_int_array_50, transpose_13

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_14 = paddle._C_ops.transpose(slice_19, [0, 1, 3, 2])
        del slice_19

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_25 = paddle._C_ops.matmul(slice_18, transpose_14, False, False)
        del slice_18, transpose_14

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(matmul_25, full_7, float("0"), True)
        del full_7, matmul_25

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_4 = paddle._C_ops.softmax(scale_4, -1)
        del scale_4

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_26 = paddle._C_ops.matmul(softmax_4, slice_20, False, False)
        del slice_20, softmax_4

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_15 = paddle._C_ops.transpose(matmul_26, [0, 2, 1, 3])
        del matmul_26

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_51 = [-1, 257, 1024]

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_9 = paddle._C_ops.reshape(transpose_15, full_int_array_51)
        del full_int_array_51, transpose_15

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_27 = paddle._C_ops.matmul(reshape_9, parameter_239, False, False)
        del parameter_239, reshape_9

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_27 = paddle._C_ops.add(matmul_27, parameter_238)
        del matmul_27, parameter_238

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_28 = paddle._C_ops.add(add_25, add_27)
        del add_25, add_27

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_28, parameter_237, parameter_236, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_236, parameter_237

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_28 = paddle._C_ops.matmul(layer_norm_30, parameter_235, False, False)
        del layer_norm_30, parameter_235

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_29 = paddle._C_ops.add(matmul_28, parameter_234)
        del matmul_28, parameter_234

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_4 = paddle._C_ops.gelu(add_29, False)
        del add_29

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_29 = paddle._C_ops.matmul(gelu_4, parameter_233, False, False)
        del gelu_4, parameter_233

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_30 = paddle._C_ops.add(matmul_29, parameter_232)
        del matmul_29, parameter_232

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_31 = paddle._C_ops.add(add_28, add_30)
        del add_28, add_30

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_31, parameter_231, parameter_230, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_230, parameter_231

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_6 = paddle._C_ops.shape64(layer_norm_33)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_52 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_53 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(
            shape64_6, [0], full_int_array_52, full_int_array_53, [1], [0]
        )
        del full_int_array_52, full_int_array_53, shape64_6

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_30 = paddle._C_ops.matmul(layer_norm_33, parameter_229, False, False)
        del layer_norm_33, parameter_229

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_32 = paddle._C_ops.add(matmul_30, parameter_228)
        del matmul_30, parameter_228

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_54 = [-1, 257, 3, 16, 64]

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_10 = paddle._C_ops.reshape(add_32, full_int_array_54)
        del add_32, full_int_array_54

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_16 = paddle._C_ops.transpose(reshape_10, [2, 0, 3, 1, 4])
        del reshape_10

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_55 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_56 = [1]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(
            transpose_16, [0], full_int_array_55, full_int_array_56, [1], [0]
        )
        del full_int_array_55, full_int_array_56

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_57 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_58 = [2]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(
            transpose_16, [0], full_int_array_57, full_int_array_58, [1], [0]
        )
        del full_int_array_57, full_int_array_58

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_59 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_60 = [3]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(
            transpose_16, [0], full_int_array_59, full_int_array_60, [1], [0]
        )
        del full_int_array_59, full_int_array_60, transpose_16

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_17 = paddle._C_ops.transpose(slice_23, [0, 1, 3, 2])
        del slice_23

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_31 = paddle._C_ops.matmul(slice_22, transpose_17, False, False)
        del slice_22, transpose_17

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(matmul_31, full_8, float("0"), True)
        del full_8, matmul_31

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_5 = paddle._C_ops.softmax(scale_5, -1)
        del scale_5

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_32 = paddle._C_ops.matmul(softmax_5, slice_24, False, False)
        del slice_24, softmax_5

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_18 = paddle._C_ops.transpose(matmul_32, [0, 2, 1, 3])
        del matmul_32

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_61 = [-1, 257, 1024]

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_18, full_int_array_61)
        del full_int_array_61, transpose_18

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_33 = paddle._C_ops.matmul(reshape_11, parameter_227, False, False)
        del parameter_227, reshape_11

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_33 = paddle._C_ops.add(matmul_33, parameter_226)
        del matmul_33, parameter_226

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_34 = paddle._C_ops.add(add_31, add_33)
        del add_31, add_33

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_34, parameter_225, parameter_224, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_224, parameter_225

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_34 = paddle._C_ops.matmul(layer_norm_36, parameter_223, False, False)
        del layer_norm_36, parameter_223

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_35 = paddle._C_ops.add(matmul_34, parameter_222)
        del matmul_34, parameter_222

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_5 = paddle._C_ops.gelu(add_35, False)
        del add_35

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_35 = paddle._C_ops.matmul(gelu_5, parameter_221, False, False)
        del gelu_5, parameter_221

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_36 = paddle._C_ops.add(matmul_35, parameter_220)
        del matmul_35, parameter_220

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_37 = paddle._C_ops.add(add_34, add_36)
        del add_34, add_36

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_37, parameter_219, parameter_218, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_218, parameter_219

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_7 = paddle._C_ops.shape64(layer_norm_39)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_62 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_63 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(
            shape64_7, [0], full_int_array_62, full_int_array_63, [1], [0]
        )
        del full_int_array_62, full_int_array_63, shape64_7

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_36 = paddle._C_ops.matmul(layer_norm_39, parameter_217, False, False)
        del layer_norm_39, parameter_217

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_38 = paddle._C_ops.add(matmul_36, parameter_216)
        del matmul_36, parameter_216

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_64 = [-1, 257, 3, 16, 64]

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_12 = paddle._C_ops.reshape(add_38, full_int_array_64)
        del add_38, full_int_array_64

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_19 = paddle._C_ops.transpose(reshape_12, [2, 0, 3, 1, 4])
        del reshape_12

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_65 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_66 = [1]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(
            transpose_19, [0], full_int_array_65, full_int_array_66, [1], [0]
        )
        del full_int_array_65, full_int_array_66

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_67 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_68 = [2]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(
            transpose_19, [0], full_int_array_67, full_int_array_68, [1], [0]
        )
        del full_int_array_67, full_int_array_68

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_69 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_70 = [3]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(
            transpose_19, [0], full_int_array_69, full_int_array_70, [1], [0]
        )
        del full_int_array_69, full_int_array_70, transpose_19

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_20 = paddle._C_ops.transpose(slice_27, [0, 1, 3, 2])
        del slice_27

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_37 = paddle._C_ops.matmul(slice_26, transpose_20, False, False)
        del slice_26, transpose_20

        # pd_op.full: (1xf32) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(matmul_37, full_9, float("0"), True)
        del full_9, matmul_37

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_6 = paddle._C_ops.softmax(scale_6, -1)
        del scale_6

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_38 = paddle._C_ops.matmul(softmax_6, slice_28, False, False)
        del slice_28, softmax_6

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_21 = paddle._C_ops.transpose(matmul_38, [0, 2, 1, 3])
        del matmul_38

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_71 = [-1, 257, 1024]

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_13 = paddle._C_ops.reshape(transpose_21, full_int_array_71)
        del full_int_array_71, transpose_21

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_39 = paddle._C_ops.matmul(reshape_13, parameter_215, False, False)
        del parameter_215, reshape_13

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_39 = paddle._C_ops.add(matmul_39, parameter_214)
        del matmul_39, parameter_214

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_40 = paddle._C_ops.add(add_37, add_39)
        del add_37, add_39

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_40, parameter_213, parameter_212, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_212, parameter_213

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_40 = paddle._C_ops.matmul(layer_norm_42, parameter_211, False, False)
        del layer_norm_42, parameter_211

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_41 = paddle._C_ops.add(matmul_40, parameter_210)
        del matmul_40, parameter_210

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_6 = paddle._C_ops.gelu(add_41, False)
        del add_41

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_41 = paddle._C_ops.matmul(gelu_6, parameter_209, False, False)
        del gelu_6, parameter_209

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_42 = paddle._C_ops.add(matmul_41, parameter_208)
        del matmul_41, parameter_208

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_43 = paddle._C_ops.add(add_40, add_42)
        del add_40, add_42

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_43, parameter_207, parameter_206, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_206, parameter_207

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_8 = paddle._C_ops.shape64(layer_norm_45)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_72 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_73 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(
            shape64_8, [0], full_int_array_72, full_int_array_73, [1], [0]
        )
        del full_int_array_72, full_int_array_73, shape64_8

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_42 = paddle._C_ops.matmul(layer_norm_45, parameter_205, False, False)
        del layer_norm_45, parameter_205

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_44 = paddle._C_ops.add(matmul_42, parameter_204)
        del matmul_42, parameter_204

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_74 = [-1, 257, 3, 16, 64]

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_14 = paddle._C_ops.reshape(add_44, full_int_array_74)
        del add_44, full_int_array_74

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_22 = paddle._C_ops.transpose(reshape_14, [2, 0, 3, 1, 4])
        del reshape_14

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_75 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_76 = [1]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(
            transpose_22, [0], full_int_array_75, full_int_array_76, [1], [0]
        )
        del full_int_array_75, full_int_array_76

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_77 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_78 = [2]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(
            transpose_22, [0], full_int_array_77, full_int_array_78, [1], [0]
        )
        del full_int_array_77, full_int_array_78

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_79 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_80 = [3]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(
            transpose_22, [0], full_int_array_79, full_int_array_80, [1], [0]
        )
        del full_int_array_79, full_int_array_80, transpose_22

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_23 = paddle._C_ops.transpose(slice_31, [0, 1, 3, 2])
        del slice_31

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_43 = paddle._C_ops.matmul(slice_30, transpose_23, False, False)
        del slice_30, transpose_23

        # pd_op.full: (1xf32) <- ()
        full_10 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(matmul_43, full_10, float("0"), True)
        del full_10, matmul_43

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_7 = paddle._C_ops.softmax(scale_7, -1)
        del scale_7

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_44 = paddle._C_ops.matmul(softmax_7, slice_32, False, False)
        del slice_32, softmax_7

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_24 = paddle._C_ops.transpose(matmul_44, [0, 2, 1, 3])
        del matmul_44

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_81 = [-1, 257, 1024]

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_15 = paddle._C_ops.reshape(transpose_24, full_int_array_81)
        del full_int_array_81, transpose_24

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_45 = paddle._C_ops.matmul(reshape_15, parameter_203, False, False)
        del parameter_203, reshape_15

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_45 = paddle._C_ops.add(matmul_45, parameter_202)
        del matmul_45, parameter_202

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_46 = paddle._C_ops.add(add_43, add_45)
        del add_43, add_45

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_46, parameter_201, parameter_200, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_200, parameter_201

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_46 = paddle._C_ops.matmul(layer_norm_48, parameter_199, False, False)
        del layer_norm_48, parameter_199

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_47 = paddle._C_ops.add(matmul_46, parameter_198)
        del matmul_46, parameter_198

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_7 = paddle._C_ops.gelu(add_47, False)
        del add_47

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_47 = paddle._C_ops.matmul(gelu_7, parameter_197, False, False)
        del gelu_7, parameter_197

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_48 = paddle._C_ops.add(matmul_47, parameter_196)
        del matmul_47, parameter_196

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_49 = paddle._C_ops.add(add_46, add_48)
        del add_46, add_48

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_49, parameter_195, parameter_194, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_194, parameter_195

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_9 = paddle._C_ops.shape64(layer_norm_51)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_82 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_83 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_33 = paddle._C_ops.slice(
            shape64_9, [0], full_int_array_82, full_int_array_83, [1], [0]
        )
        del full_int_array_82, full_int_array_83, shape64_9

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_48 = paddle._C_ops.matmul(layer_norm_51, parameter_193, False, False)
        del layer_norm_51, parameter_193

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_50 = paddle._C_ops.add(matmul_48, parameter_192)
        del matmul_48, parameter_192

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_84 = [-1, 257, 3, 16, 64]

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_16 = paddle._C_ops.reshape(add_50, full_int_array_84)
        del add_50, full_int_array_84

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_25 = paddle._C_ops.transpose(reshape_16, [2, 0, 3, 1, 4])
        del reshape_16

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_85 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_86 = [1]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_34 = paddle._C_ops.slice(
            transpose_25, [0], full_int_array_85, full_int_array_86, [1], [0]
        )
        del full_int_array_85, full_int_array_86

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_87 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_88 = [2]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_35 = paddle._C_ops.slice(
            transpose_25, [0], full_int_array_87, full_int_array_88, [1], [0]
        )
        del full_int_array_87, full_int_array_88

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_89 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_90 = [3]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(
            transpose_25, [0], full_int_array_89, full_int_array_90, [1], [0]
        )
        del full_int_array_89, full_int_array_90, transpose_25

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_26 = paddle._C_ops.transpose(slice_35, [0, 1, 3, 2])
        del slice_35

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_49 = paddle._C_ops.matmul(slice_34, transpose_26, False, False)
        del slice_34, transpose_26

        # pd_op.full: (1xf32) <- ()
        full_11 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(matmul_49, full_11, float("0"), True)
        del full_11, matmul_49

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_8 = paddle._C_ops.softmax(scale_8, -1)
        del scale_8

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_50 = paddle._C_ops.matmul(softmax_8, slice_36, False, False)
        del slice_36, softmax_8

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_27 = paddle._C_ops.transpose(matmul_50, [0, 2, 1, 3])
        del matmul_50

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_91 = [-1, 257, 1024]

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_17 = paddle._C_ops.reshape(transpose_27, full_int_array_91)
        del full_int_array_91, transpose_27

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_51 = paddle._C_ops.matmul(reshape_17, parameter_191, False, False)
        del parameter_191, reshape_17

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_51 = paddle._C_ops.add(matmul_51, parameter_190)
        del matmul_51, parameter_190

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_52 = paddle._C_ops.add(add_49, add_51)
        del add_49, add_51

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_52, parameter_189, parameter_188, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_188, parameter_189

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_52 = paddle._C_ops.matmul(layer_norm_54, parameter_187, False, False)
        del layer_norm_54, parameter_187

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_53 = paddle._C_ops.add(matmul_52, parameter_186)
        del matmul_52, parameter_186

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_8 = paddle._C_ops.gelu(add_53, False)
        del add_53

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_53 = paddle._C_ops.matmul(gelu_8, parameter_185, False, False)
        del gelu_8, parameter_185

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_54 = paddle._C_ops.add(matmul_53, parameter_184)
        del matmul_53, parameter_184

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_55 = paddle._C_ops.add(add_52, add_54)
        del add_52, add_54

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_55, parameter_183, parameter_182, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_182, parameter_183

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_10 = paddle._C_ops.shape64(layer_norm_57)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_92 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_93 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_37 = paddle._C_ops.slice(
            shape64_10, [0], full_int_array_92, full_int_array_93, [1], [0]
        )
        del full_int_array_92, full_int_array_93, shape64_10

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_54 = paddle._C_ops.matmul(layer_norm_57, parameter_181, False, False)
        del layer_norm_57, parameter_181

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_56 = paddle._C_ops.add(matmul_54, parameter_180)
        del matmul_54, parameter_180

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_94 = [-1, 257, 3, 16, 64]

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_18 = paddle._C_ops.reshape(add_56, full_int_array_94)
        del add_56, full_int_array_94

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_28 = paddle._C_ops.transpose(reshape_18, [2, 0, 3, 1, 4])
        del reshape_18

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_95 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_96 = [1]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_38 = paddle._C_ops.slice(
            transpose_28, [0], full_int_array_95, full_int_array_96, [1], [0]
        )
        del full_int_array_95, full_int_array_96

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_97 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_98 = [2]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_39 = paddle._C_ops.slice(
            transpose_28, [0], full_int_array_97, full_int_array_98, [1], [0]
        )
        del full_int_array_97, full_int_array_98

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_99 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_100 = [3]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_40 = paddle._C_ops.slice(
            transpose_28, [0], full_int_array_99, full_int_array_100, [1], [0]
        )
        del full_int_array_100, full_int_array_99, transpose_28

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_29 = paddle._C_ops.transpose(slice_39, [0, 1, 3, 2])
        del slice_39

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_55 = paddle._C_ops.matmul(slice_38, transpose_29, False, False)
        del slice_38, transpose_29

        # pd_op.full: (1xf32) <- ()
        full_12 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(matmul_55, full_12, float("0"), True)
        del full_12, matmul_55

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_9 = paddle._C_ops.softmax(scale_9, -1)
        del scale_9

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_56 = paddle._C_ops.matmul(softmax_9, slice_40, False, False)
        del slice_40, softmax_9

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_30 = paddle._C_ops.transpose(matmul_56, [0, 2, 1, 3])
        del matmul_56

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_101 = [-1, 257, 1024]

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_19 = paddle._C_ops.reshape(transpose_30, full_int_array_101)
        del full_int_array_101, transpose_30

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_57 = paddle._C_ops.matmul(reshape_19, parameter_179, False, False)
        del parameter_179, reshape_19

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_57 = paddle._C_ops.add(matmul_57, parameter_178)
        del matmul_57, parameter_178

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_58 = paddle._C_ops.add(add_55, add_57)
        del add_55, add_57

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_58, parameter_177, parameter_176, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_176, parameter_177

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_58 = paddle._C_ops.matmul(layer_norm_60, parameter_175, False, False)
        del layer_norm_60, parameter_175

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_59 = paddle._C_ops.add(matmul_58, parameter_174)
        del matmul_58, parameter_174

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_9 = paddle._C_ops.gelu(add_59, False)
        del add_59

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_59 = paddle._C_ops.matmul(gelu_9, parameter_173, False, False)
        del gelu_9, parameter_173

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_60 = paddle._C_ops.add(matmul_59, parameter_172)
        del matmul_59, parameter_172

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_61 = paddle._C_ops.add(add_58, add_60)
        del add_58, add_60

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_61, parameter_171, parameter_170, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_170, parameter_171

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_11 = paddle._C_ops.shape64(layer_norm_63)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_102 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_103 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_41 = paddle._C_ops.slice(
            shape64_11, [0], full_int_array_102, full_int_array_103, [1], [0]
        )
        del full_int_array_102, full_int_array_103, shape64_11

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_60 = paddle._C_ops.matmul(layer_norm_63, parameter_169, False, False)
        del layer_norm_63, parameter_169

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_62 = paddle._C_ops.add(matmul_60, parameter_168)
        del matmul_60, parameter_168

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_104 = [-1, 257, 3, 16, 64]

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_20 = paddle._C_ops.reshape(add_62, full_int_array_104)
        del add_62, full_int_array_104

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_31 = paddle._C_ops.transpose(reshape_20, [2, 0, 3, 1, 4])
        del reshape_20

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_105 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_106 = [1]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_42 = paddle._C_ops.slice(
            transpose_31, [0], full_int_array_105, full_int_array_106, [1], [0]
        )
        del full_int_array_105, full_int_array_106

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_107 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_108 = [2]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_43 = paddle._C_ops.slice(
            transpose_31, [0], full_int_array_107, full_int_array_108, [1], [0]
        )
        del full_int_array_107, full_int_array_108

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_109 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_110 = [3]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_44 = paddle._C_ops.slice(
            transpose_31, [0], full_int_array_109, full_int_array_110, [1], [0]
        )
        del full_int_array_109, full_int_array_110, transpose_31

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_32 = paddle._C_ops.transpose(slice_43, [0, 1, 3, 2])
        del slice_43

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_61 = paddle._C_ops.matmul(slice_42, transpose_32, False, False)
        del slice_42, transpose_32

        # pd_op.full: (1xf32) <- ()
        full_13 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(matmul_61, full_13, float("0"), True)
        del full_13, matmul_61

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_10 = paddle._C_ops.softmax(scale_10, -1)
        del scale_10

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_62 = paddle._C_ops.matmul(softmax_10, slice_44, False, False)
        del slice_44, softmax_10

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_33 = paddle._C_ops.transpose(matmul_62, [0, 2, 1, 3])
        del matmul_62

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_111 = [-1, 257, 1024]

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_21 = paddle._C_ops.reshape(transpose_33, full_int_array_111)
        del full_int_array_111, transpose_33

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_63 = paddle._C_ops.matmul(reshape_21, parameter_167, False, False)
        del parameter_167, reshape_21

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_63 = paddle._C_ops.add(matmul_63, parameter_166)
        del matmul_63, parameter_166

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_64 = paddle._C_ops.add(add_61, add_63)
        del add_61, add_63

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_64, parameter_165, parameter_164, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_164, parameter_165

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_64 = paddle._C_ops.matmul(layer_norm_66, parameter_163, False, False)
        del layer_norm_66, parameter_163

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_65 = paddle._C_ops.add(matmul_64, parameter_162)
        del matmul_64, parameter_162

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_10 = paddle._C_ops.gelu(add_65, False)
        del add_65

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_65 = paddle._C_ops.matmul(gelu_10, parameter_161, False, False)
        del gelu_10, parameter_161

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_66 = paddle._C_ops.add(matmul_65, parameter_160)
        del matmul_65, parameter_160

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_67 = paddle._C_ops.add(add_64, add_66)
        del add_64, add_66

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_67, parameter_159, parameter_158, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_158, parameter_159

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_12 = paddle._C_ops.shape64(layer_norm_69)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_112 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_113 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_45 = paddle._C_ops.slice(
            shape64_12, [0], full_int_array_112, full_int_array_113, [1], [0]
        )
        del full_int_array_112, full_int_array_113, shape64_12

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_66 = paddle._C_ops.matmul(layer_norm_69, parameter_157, False, False)
        del layer_norm_69, parameter_157

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_68 = paddle._C_ops.add(matmul_66, parameter_156)
        del matmul_66, parameter_156

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_114 = [-1, 257, 3, 16, 64]

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_22 = paddle._C_ops.reshape(add_68, full_int_array_114)
        del add_68, full_int_array_114

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_34 = paddle._C_ops.transpose(reshape_22, [2, 0, 3, 1, 4])
        del reshape_22

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_115 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_116 = [1]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_46 = paddle._C_ops.slice(
            transpose_34, [0], full_int_array_115, full_int_array_116, [1], [0]
        )
        del full_int_array_115, full_int_array_116

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_117 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_118 = [2]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_47 = paddle._C_ops.slice(
            transpose_34, [0], full_int_array_117, full_int_array_118, [1], [0]
        )
        del full_int_array_117, full_int_array_118

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_119 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_120 = [3]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_48 = paddle._C_ops.slice(
            transpose_34, [0], full_int_array_119, full_int_array_120, [1], [0]
        )
        del full_int_array_119, full_int_array_120, transpose_34

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_35 = paddle._C_ops.transpose(slice_47, [0, 1, 3, 2])
        del slice_47

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_67 = paddle._C_ops.matmul(slice_46, transpose_35, False, False)
        del slice_46, transpose_35

        # pd_op.full: (1xf32) <- ()
        full_14 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(matmul_67, full_14, float("0"), True)
        del full_14, matmul_67

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_11 = paddle._C_ops.softmax(scale_11, -1)
        del scale_11

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_68 = paddle._C_ops.matmul(softmax_11, slice_48, False, False)
        del slice_48, softmax_11

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_36 = paddle._C_ops.transpose(matmul_68, [0, 2, 1, 3])
        del matmul_68

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_121 = [-1, 257, 1024]

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_23 = paddle._C_ops.reshape(transpose_36, full_int_array_121)
        del full_int_array_121, transpose_36

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_69 = paddle._C_ops.matmul(reshape_23, parameter_155, False, False)
        del parameter_155, reshape_23

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_69 = paddle._C_ops.add(matmul_69, parameter_154)
        del matmul_69, parameter_154

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_70 = paddle._C_ops.add(add_67, add_69)
        del add_67, add_69

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_72, layer_norm_73, layer_norm_74 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_70, parameter_153, parameter_152, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_152, parameter_153

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_70 = paddle._C_ops.matmul(layer_norm_72, parameter_151, False, False)
        del layer_norm_72, parameter_151

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_71 = paddle._C_ops.add(matmul_70, parameter_150)
        del matmul_70, parameter_150

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_11 = paddle._C_ops.gelu(add_71, False)
        del add_71

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_71 = paddle._C_ops.matmul(gelu_11, parameter_149, False, False)
        del gelu_11, parameter_149

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_72 = paddle._C_ops.add(matmul_71, parameter_148)
        del matmul_71, parameter_148

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_73 = paddle._C_ops.add(add_70, add_72)
        del add_70, add_72

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_75, layer_norm_76, layer_norm_77 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_73, parameter_147, parameter_146, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_146, parameter_147

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_13 = paddle._C_ops.shape64(layer_norm_75)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_122 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_123 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_49 = paddle._C_ops.slice(
            shape64_13, [0], full_int_array_122, full_int_array_123, [1], [0]
        )
        del full_int_array_122, full_int_array_123, shape64_13

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_72 = paddle._C_ops.matmul(layer_norm_75, parameter_145, False, False)
        del layer_norm_75, parameter_145

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_74 = paddle._C_ops.add(matmul_72, parameter_144)
        del matmul_72, parameter_144

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_124 = [-1, 257, 3, 16, 64]

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_24 = paddle._C_ops.reshape(add_74, full_int_array_124)
        del add_74, full_int_array_124

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_37 = paddle._C_ops.transpose(reshape_24, [2, 0, 3, 1, 4])
        del reshape_24

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_125 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_126 = [1]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_50 = paddle._C_ops.slice(
            transpose_37, [0], full_int_array_125, full_int_array_126, [1], [0]
        )
        del full_int_array_125, full_int_array_126

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_127 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_128 = [2]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_51 = paddle._C_ops.slice(
            transpose_37, [0], full_int_array_127, full_int_array_128, [1], [0]
        )
        del full_int_array_127, full_int_array_128

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_129 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_130 = [3]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_52 = paddle._C_ops.slice(
            transpose_37, [0], full_int_array_129, full_int_array_130, [1], [0]
        )
        del full_int_array_129, full_int_array_130, transpose_37

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_38 = paddle._C_ops.transpose(slice_51, [0, 1, 3, 2])
        del slice_51

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_73 = paddle._C_ops.matmul(slice_50, transpose_38, False, False)
        del slice_50, transpose_38

        # pd_op.full: (1xf32) <- ()
        full_15 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(matmul_73, full_15, float("0"), True)
        del full_15, matmul_73

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_12 = paddle._C_ops.softmax(scale_12, -1)
        del scale_12

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_74 = paddle._C_ops.matmul(softmax_12, slice_52, False, False)
        del slice_52, softmax_12

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_39 = paddle._C_ops.transpose(matmul_74, [0, 2, 1, 3])
        del matmul_74

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_131 = [-1, 257, 1024]

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_25 = paddle._C_ops.reshape(transpose_39, full_int_array_131)
        del full_int_array_131, transpose_39

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_75 = paddle._C_ops.matmul(reshape_25, parameter_143, False, False)
        del parameter_143, reshape_25

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_75 = paddle._C_ops.add(matmul_75, parameter_142)
        del matmul_75, parameter_142

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_76 = paddle._C_ops.add(add_73, add_75)
        del add_73, add_75

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_78, layer_norm_79, layer_norm_80 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_76, parameter_141, parameter_140, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_140, parameter_141

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_76 = paddle._C_ops.matmul(layer_norm_78, parameter_139, False, False)
        del layer_norm_78, parameter_139

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_77 = paddle._C_ops.add(matmul_76, parameter_138)
        del matmul_76, parameter_138

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_12 = paddle._C_ops.gelu(add_77, False)
        del add_77

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_77 = paddle._C_ops.matmul(gelu_12, parameter_137, False, False)
        del gelu_12, parameter_137

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_78 = paddle._C_ops.add(matmul_77, parameter_136)
        del matmul_77, parameter_136

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_79 = paddle._C_ops.add(add_76, add_78)
        del add_76, add_78

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_81, layer_norm_82, layer_norm_83 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_79, parameter_135, parameter_134, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_134, parameter_135

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_14 = paddle._C_ops.shape64(layer_norm_81)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_132 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_133 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_53 = paddle._C_ops.slice(
            shape64_14, [0], full_int_array_132, full_int_array_133, [1], [0]
        )
        del full_int_array_132, full_int_array_133, shape64_14

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_78 = paddle._C_ops.matmul(layer_norm_81, parameter_133, False, False)
        del layer_norm_81, parameter_133

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_80 = paddle._C_ops.add(matmul_78, parameter_132)
        del matmul_78, parameter_132

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_134 = [-1, 257, 3, 16, 64]

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_26 = paddle._C_ops.reshape(add_80, full_int_array_134)
        del add_80, full_int_array_134

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_40 = paddle._C_ops.transpose(reshape_26, [2, 0, 3, 1, 4])
        del reshape_26

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_135 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_136 = [1]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_54 = paddle._C_ops.slice(
            transpose_40, [0], full_int_array_135, full_int_array_136, [1], [0]
        )
        del full_int_array_135, full_int_array_136

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_137 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_138 = [2]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_55 = paddle._C_ops.slice(
            transpose_40, [0], full_int_array_137, full_int_array_138, [1], [0]
        )
        del full_int_array_137, full_int_array_138

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_139 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_140 = [3]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_56 = paddle._C_ops.slice(
            transpose_40, [0], full_int_array_139, full_int_array_140, [1], [0]
        )
        del full_int_array_139, full_int_array_140, transpose_40

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_41 = paddle._C_ops.transpose(slice_55, [0, 1, 3, 2])
        del slice_55

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_79 = paddle._C_ops.matmul(slice_54, transpose_41, False, False)
        del slice_54, transpose_41

        # pd_op.full: (1xf32) <- ()
        full_16 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(matmul_79, full_16, float("0"), True)
        del full_16, matmul_79

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_13 = paddle._C_ops.softmax(scale_13, -1)
        del scale_13

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_80 = paddle._C_ops.matmul(softmax_13, slice_56, False, False)
        del slice_56, softmax_13

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_42 = paddle._C_ops.transpose(matmul_80, [0, 2, 1, 3])
        del matmul_80

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_141 = [-1, 257, 1024]

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_27 = paddle._C_ops.reshape(transpose_42, full_int_array_141)
        del full_int_array_141, transpose_42

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_81 = paddle._C_ops.matmul(reshape_27, parameter_131, False, False)
        del parameter_131, reshape_27

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_81 = paddle._C_ops.add(matmul_81, parameter_130)
        del matmul_81, parameter_130

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_82 = paddle._C_ops.add(add_79, add_81)
        del add_79, add_81

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_84, layer_norm_85, layer_norm_86 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_82, parameter_129, parameter_128, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_128, parameter_129

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_82 = paddle._C_ops.matmul(layer_norm_84, parameter_127, False, False)
        del layer_norm_84, parameter_127

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_83 = paddle._C_ops.add(matmul_82, parameter_126)
        del matmul_82, parameter_126

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_13 = paddle._C_ops.gelu(add_83, False)
        del add_83

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_83 = paddle._C_ops.matmul(gelu_13, parameter_125, False, False)
        del gelu_13, parameter_125

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_84 = paddle._C_ops.add(matmul_83, parameter_124)
        del matmul_83, parameter_124

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_85 = paddle._C_ops.add(add_82, add_84)
        del add_82, add_84

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_87, layer_norm_88, layer_norm_89 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_85, parameter_123, parameter_122, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_122, parameter_123

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_15 = paddle._C_ops.shape64(layer_norm_87)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_142 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_143 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_57 = paddle._C_ops.slice(
            shape64_15, [0], full_int_array_142, full_int_array_143, [1], [0]
        )
        del full_int_array_142, full_int_array_143, shape64_15

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_84 = paddle._C_ops.matmul(layer_norm_87, parameter_121, False, False)
        del layer_norm_87, parameter_121

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_86 = paddle._C_ops.add(matmul_84, parameter_120)
        del matmul_84, parameter_120

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_144 = [-1, 257, 3, 16, 64]

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_28 = paddle._C_ops.reshape(add_86, full_int_array_144)
        del add_86, full_int_array_144

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_43 = paddle._C_ops.transpose(reshape_28, [2, 0, 3, 1, 4])
        del reshape_28

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_145 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_146 = [1]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_58 = paddle._C_ops.slice(
            transpose_43, [0], full_int_array_145, full_int_array_146, [1], [0]
        )
        del full_int_array_145, full_int_array_146

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_147 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_148 = [2]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_59 = paddle._C_ops.slice(
            transpose_43, [0], full_int_array_147, full_int_array_148, [1], [0]
        )
        del full_int_array_147, full_int_array_148

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_149 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_150 = [3]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_60 = paddle._C_ops.slice(
            transpose_43, [0], full_int_array_149, full_int_array_150, [1], [0]
        )
        del full_int_array_149, full_int_array_150, transpose_43

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_44 = paddle._C_ops.transpose(slice_59, [0, 1, 3, 2])
        del slice_59

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_85 = paddle._C_ops.matmul(slice_58, transpose_44, False, False)
        del slice_58, transpose_44

        # pd_op.full: (1xf32) <- ()
        full_17 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(matmul_85, full_17, float("0"), True)
        del full_17, matmul_85

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_14 = paddle._C_ops.softmax(scale_14, -1)
        del scale_14

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_86 = paddle._C_ops.matmul(softmax_14, slice_60, False, False)
        del slice_60, softmax_14

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_45 = paddle._C_ops.transpose(matmul_86, [0, 2, 1, 3])
        del matmul_86

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_151 = [-1, 257, 1024]

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_29 = paddle._C_ops.reshape(transpose_45, full_int_array_151)
        del full_int_array_151, transpose_45

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_87 = paddle._C_ops.matmul(reshape_29, parameter_119, False, False)
        del parameter_119, reshape_29

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_87 = paddle._C_ops.add(matmul_87, parameter_118)
        del matmul_87, parameter_118

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_88 = paddle._C_ops.add(add_85, add_87)
        del add_85, add_87

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_90, layer_norm_91, layer_norm_92 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_88, parameter_117, parameter_116, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_116, parameter_117

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_88 = paddle._C_ops.matmul(layer_norm_90, parameter_115, False, False)
        del layer_norm_90, parameter_115

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_89 = paddle._C_ops.add(matmul_88, parameter_114)
        del matmul_88, parameter_114

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_14 = paddle._C_ops.gelu(add_89, False)
        del add_89

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_89 = paddle._C_ops.matmul(gelu_14, parameter_113, False, False)
        del gelu_14, parameter_113

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_90 = paddle._C_ops.add(matmul_89, parameter_112)
        del matmul_89, parameter_112

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_91 = paddle._C_ops.add(add_88, add_90)
        del add_88, add_90

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_93, layer_norm_94, layer_norm_95 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_91, parameter_111, parameter_110, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_110, parameter_111

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_16 = paddle._C_ops.shape64(layer_norm_93)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_152 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_153 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_61 = paddle._C_ops.slice(
            shape64_16, [0], full_int_array_152, full_int_array_153, [1], [0]
        )
        del full_int_array_152, full_int_array_153, shape64_16

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_90 = paddle._C_ops.matmul(layer_norm_93, parameter_109, False, False)
        del layer_norm_93, parameter_109

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_92 = paddle._C_ops.add(matmul_90, parameter_108)
        del matmul_90, parameter_108

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_154 = [-1, 257, 3, 16, 64]

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_30 = paddle._C_ops.reshape(add_92, full_int_array_154)
        del add_92, full_int_array_154

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_46 = paddle._C_ops.transpose(reshape_30, [2, 0, 3, 1, 4])
        del reshape_30

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_155 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_156 = [1]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_62 = paddle._C_ops.slice(
            transpose_46, [0], full_int_array_155, full_int_array_156, [1], [0]
        )
        del full_int_array_155, full_int_array_156

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_157 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_158 = [2]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_63 = paddle._C_ops.slice(
            transpose_46, [0], full_int_array_157, full_int_array_158, [1], [0]
        )
        del full_int_array_157, full_int_array_158

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_159 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_160 = [3]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_64 = paddle._C_ops.slice(
            transpose_46, [0], full_int_array_159, full_int_array_160, [1], [0]
        )
        del full_int_array_159, full_int_array_160, transpose_46

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_47 = paddle._C_ops.transpose(slice_63, [0, 1, 3, 2])
        del slice_63

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_91 = paddle._C_ops.matmul(slice_62, transpose_47, False, False)
        del slice_62, transpose_47

        # pd_op.full: (1xf32) <- ()
        full_18 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_15 = paddle._C_ops.scale(matmul_91, full_18, float("0"), True)
        del full_18, matmul_91

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_15 = paddle._C_ops.softmax(scale_15, -1)
        del scale_15

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_92 = paddle._C_ops.matmul(softmax_15, slice_64, False, False)
        del slice_64, softmax_15

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_48 = paddle._C_ops.transpose(matmul_92, [0, 2, 1, 3])
        del matmul_92

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_161 = [-1, 257, 1024]

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_31 = paddle._C_ops.reshape(transpose_48, full_int_array_161)
        del full_int_array_161, transpose_48

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_93 = paddle._C_ops.matmul(reshape_31, parameter_107, False, False)
        del parameter_107, reshape_31

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_93 = paddle._C_ops.add(matmul_93, parameter_106)
        del matmul_93, parameter_106

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_94 = paddle._C_ops.add(add_91, add_93)
        del add_91, add_93

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_96, layer_norm_97, layer_norm_98 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_94, parameter_105, parameter_104, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_104, parameter_105

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_94 = paddle._C_ops.matmul(layer_norm_96, parameter_103, False, False)
        del layer_norm_96, parameter_103

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_95 = paddle._C_ops.add(matmul_94, parameter_102)
        del matmul_94, parameter_102

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_15 = paddle._C_ops.gelu(add_95, False)
        del add_95

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_95 = paddle._C_ops.matmul(gelu_15, parameter_101, False, False)
        del gelu_15, parameter_101

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_96 = paddle._C_ops.add(matmul_95, parameter_100)
        del matmul_95, parameter_100

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_97 = paddle._C_ops.add(add_94, add_96)
        del add_94, add_96

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_99, layer_norm_100, layer_norm_101 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_97, parameter_99, parameter_98, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_98, parameter_99

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_17 = paddle._C_ops.shape64(layer_norm_99)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_162 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_163 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_65 = paddle._C_ops.slice(
            shape64_17, [0], full_int_array_162, full_int_array_163, [1], [0]
        )
        del full_int_array_162, full_int_array_163, shape64_17

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_96 = paddle._C_ops.matmul(layer_norm_99, parameter_97, False, False)
        del layer_norm_99, parameter_97

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_98 = paddle._C_ops.add(matmul_96, parameter_96)
        del matmul_96, parameter_96

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_164 = [-1, 257, 3, 16, 64]

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_32 = paddle._C_ops.reshape(add_98, full_int_array_164)
        del add_98, full_int_array_164

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_49 = paddle._C_ops.transpose(reshape_32, [2, 0, 3, 1, 4])
        del reshape_32

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_165 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_166 = [1]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_66 = paddle._C_ops.slice(
            transpose_49, [0], full_int_array_165, full_int_array_166, [1], [0]
        )
        del full_int_array_165, full_int_array_166

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_167 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_168 = [2]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_67 = paddle._C_ops.slice(
            transpose_49, [0], full_int_array_167, full_int_array_168, [1], [0]
        )
        del full_int_array_167, full_int_array_168

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_169 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_170 = [3]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_68 = paddle._C_ops.slice(
            transpose_49, [0], full_int_array_169, full_int_array_170, [1], [0]
        )
        del full_int_array_169, full_int_array_170, transpose_49

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_50 = paddle._C_ops.transpose(slice_67, [0, 1, 3, 2])
        del slice_67

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_97 = paddle._C_ops.matmul(slice_66, transpose_50, False, False)
        del slice_66, transpose_50

        # pd_op.full: (1xf32) <- ()
        full_19 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_16 = paddle._C_ops.scale(matmul_97, full_19, float("0"), True)
        del full_19, matmul_97

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_16 = paddle._C_ops.softmax(scale_16, -1)
        del scale_16

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_98 = paddle._C_ops.matmul(softmax_16, slice_68, False, False)
        del slice_68, softmax_16

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_51 = paddle._C_ops.transpose(matmul_98, [0, 2, 1, 3])
        del matmul_98

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_171 = [-1, 257, 1024]

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_33 = paddle._C_ops.reshape(transpose_51, full_int_array_171)
        del full_int_array_171, transpose_51

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_99 = paddle._C_ops.matmul(reshape_33, parameter_95, False, False)
        del parameter_95, reshape_33

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_99 = paddle._C_ops.add(matmul_99, parameter_94)
        del matmul_99, parameter_94

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_100 = paddle._C_ops.add(add_97, add_99)
        del add_97, add_99

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_102, layer_norm_103, layer_norm_104 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_100, parameter_93, parameter_92, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_92, parameter_93

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_100 = paddle._C_ops.matmul(layer_norm_102, parameter_91, False, False)
        del layer_norm_102, parameter_91

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_101 = paddle._C_ops.add(matmul_100, parameter_90)
        del matmul_100, parameter_90

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_16 = paddle._C_ops.gelu(add_101, False)
        del add_101

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_101 = paddle._C_ops.matmul(gelu_16, parameter_89, False, False)
        del gelu_16, parameter_89

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_102 = paddle._C_ops.add(matmul_101, parameter_88)
        del matmul_101, parameter_88

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_103 = paddle._C_ops.add(add_100, add_102)
        del add_100, add_102

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_105, layer_norm_106, layer_norm_107 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_103, parameter_87, parameter_86, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_86, parameter_87

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_18 = paddle._C_ops.shape64(layer_norm_105)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_172 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_173 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_69 = paddle._C_ops.slice(
            shape64_18, [0], full_int_array_172, full_int_array_173, [1], [0]
        )
        del full_int_array_172, full_int_array_173, shape64_18

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_102 = paddle._C_ops.matmul(layer_norm_105, parameter_85, False, False)
        del layer_norm_105, parameter_85

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_104 = paddle._C_ops.add(matmul_102, parameter_84)
        del matmul_102, parameter_84

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_174 = [-1, 257, 3, 16, 64]

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_34 = paddle._C_ops.reshape(add_104, full_int_array_174)
        del add_104, full_int_array_174

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_52 = paddle._C_ops.transpose(reshape_34, [2, 0, 3, 1, 4])
        del reshape_34

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_175 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_176 = [1]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_70 = paddle._C_ops.slice(
            transpose_52, [0], full_int_array_175, full_int_array_176, [1], [0]
        )
        del full_int_array_175, full_int_array_176

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_177 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_178 = [2]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_71 = paddle._C_ops.slice(
            transpose_52, [0], full_int_array_177, full_int_array_178, [1], [0]
        )
        del full_int_array_177, full_int_array_178

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_179 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_180 = [3]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_72 = paddle._C_ops.slice(
            transpose_52, [0], full_int_array_179, full_int_array_180, [1], [0]
        )
        del full_int_array_179, full_int_array_180, transpose_52

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_53 = paddle._C_ops.transpose(slice_71, [0, 1, 3, 2])
        del slice_71

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_103 = paddle._C_ops.matmul(slice_70, transpose_53, False, False)
        del slice_70, transpose_53

        # pd_op.full: (1xf32) <- ()
        full_20 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_17 = paddle._C_ops.scale(matmul_103, full_20, float("0"), True)
        del full_20, matmul_103

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_17 = paddle._C_ops.softmax(scale_17, -1)
        del scale_17

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_104 = paddle._C_ops.matmul(softmax_17, slice_72, False, False)
        del slice_72, softmax_17

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_54 = paddle._C_ops.transpose(matmul_104, [0, 2, 1, 3])
        del matmul_104

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_181 = [-1, 257, 1024]

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_35 = paddle._C_ops.reshape(transpose_54, full_int_array_181)
        del full_int_array_181, transpose_54

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_105 = paddle._C_ops.matmul(reshape_35, parameter_83, False, False)
        del parameter_83, reshape_35

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_105 = paddle._C_ops.add(matmul_105, parameter_82)
        del matmul_105, parameter_82

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_106 = paddle._C_ops.add(add_103, add_105)
        del add_103, add_105

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_108, layer_norm_109, layer_norm_110 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_106, parameter_81, parameter_80, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_80, parameter_81

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_106 = paddle._C_ops.matmul(layer_norm_108, parameter_79, False, False)
        del layer_norm_108, parameter_79

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_107 = paddle._C_ops.add(matmul_106, parameter_78)
        del matmul_106, parameter_78

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_17 = paddle._C_ops.gelu(add_107, False)
        del add_107

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_107 = paddle._C_ops.matmul(gelu_17, parameter_77, False, False)
        del gelu_17, parameter_77

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_108 = paddle._C_ops.add(matmul_107, parameter_76)
        del matmul_107, parameter_76

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_109 = paddle._C_ops.add(add_106, add_108)
        del add_106, add_108

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_111, layer_norm_112, layer_norm_113 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_109, parameter_75, parameter_74, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_74, parameter_75

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_19 = paddle._C_ops.shape64(layer_norm_111)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_182 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_183 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_73 = paddle._C_ops.slice(
            shape64_19, [0], full_int_array_182, full_int_array_183, [1], [0]
        )
        del full_int_array_182, full_int_array_183, shape64_19

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_108 = paddle._C_ops.matmul(layer_norm_111, parameter_73, False, False)
        del layer_norm_111, parameter_73

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_110 = paddle._C_ops.add(matmul_108, parameter_72)
        del matmul_108, parameter_72

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_184 = [-1, 257, 3, 16, 64]

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_36 = paddle._C_ops.reshape(add_110, full_int_array_184)
        del add_110, full_int_array_184

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_55 = paddle._C_ops.transpose(reshape_36, [2, 0, 3, 1, 4])
        del reshape_36

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_185 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_186 = [1]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_74 = paddle._C_ops.slice(
            transpose_55, [0], full_int_array_185, full_int_array_186, [1], [0]
        )
        del full_int_array_185, full_int_array_186

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_187 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_188 = [2]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_75 = paddle._C_ops.slice(
            transpose_55, [0], full_int_array_187, full_int_array_188, [1], [0]
        )
        del full_int_array_187, full_int_array_188

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_189 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_190 = [3]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_76 = paddle._C_ops.slice(
            transpose_55, [0], full_int_array_189, full_int_array_190, [1], [0]
        )
        del full_int_array_189, full_int_array_190, transpose_55

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_56 = paddle._C_ops.transpose(slice_75, [0, 1, 3, 2])
        del slice_75

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_109 = paddle._C_ops.matmul(slice_74, transpose_56, False, False)
        del slice_74, transpose_56

        # pd_op.full: (1xf32) <- ()
        full_21 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_18 = paddle._C_ops.scale(matmul_109, full_21, float("0"), True)
        del full_21, matmul_109

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_18 = paddle._C_ops.softmax(scale_18, -1)
        del scale_18

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_110 = paddle._C_ops.matmul(softmax_18, slice_76, False, False)
        del slice_76, softmax_18

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_57 = paddle._C_ops.transpose(matmul_110, [0, 2, 1, 3])
        del matmul_110

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_191 = [-1, 257, 1024]

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_37 = paddle._C_ops.reshape(transpose_57, full_int_array_191)
        del full_int_array_191, transpose_57

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_111 = paddle._C_ops.matmul(reshape_37, parameter_71, False, False)
        del parameter_71, reshape_37

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_111 = paddle._C_ops.add(matmul_111, parameter_70)
        del matmul_111, parameter_70

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_112 = paddle._C_ops.add(add_109, add_111)
        del add_109, add_111

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_114, layer_norm_115, layer_norm_116 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_112, parameter_69, parameter_68, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_68, parameter_69

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_112 = paddle._C_ops.matmul(layer_norm_114, parameter_67, False, False)
        del layer_norm_114, parameter_67

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_113 = paddle._C_ops.add(matmul_112, parameter_66)
        del matmul_112, parameter_66

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_18 = paddle._C_ops.gelu(add_113, False)
        del add_113

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_113 = paddle._C_ops.matmul(gelu_18, parameter_65, False, False)
        del gelu_18, parameter_65

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_114 = paddle._C_ops.add(matmul_113, parameter_64)
        del matmul_113, parameter_64

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_115 = paddle._C_ops.add(add_112, add_114)
        del add_112, add_114

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_117, layer_norm_118, layer_norm_119 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_115, parameter_63, parameter_62, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_62, parameter_63

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_20 = paddle._C_ops.shape64(layer_norm_117)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_192 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_193 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_77 = paddle._C_ops.slice(
            shape64_20, [0], full_int_array_192, full_int_array_193, [1], [0]
        )
        del full_int_array_192, full_int_array_193, shape64_20

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_114 = paddle._C_ops.matmul(layer_norm_117, parameter_61, False, False)
        del layer_norm_117, parameter_61

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_116 = paddle._C_ops.add(matmul_114, parameter_60)
        del matmul_114, parameter_60

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_194 = [-1, 257, 3, 16, 64]

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_38 = paddle._C_ops.reshape(add_116, full_int_array_194)
        del add_116, full_int_array_194

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_58 = paddle._C_ops.transpose(reshape_38, [2, 0, 3, 1, 4])
        del reshape_38

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_195 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_196 = [1]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_78 = paddle._C_ops.slice(
            transpose_58, [0], full_int_array_195, full_int_array_196, [1], [0]
        )
        del full_int_array_195, full_int_array_196

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_197 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_198 = [2]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_79 = paddle._C_ops.slice(
            transpose_58, [0], full_int_array_197, full_int_array_198, [1], [0]
        )
        del full_int_array_197, full_int_array_198

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_199 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_200 = [3]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_80 = paddle._C_ops.slice(
            transpose_58, [0], full_int_array_199, full_int_array_200, [1], [0]
        )
        del full_int_array_199, full_int_array_200, transpose_58

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_59 = paddle._C_ops.transpose(slice_79, [0, 1, 3, 2])
        del slice_79

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_115 = paddle._C_ops.matmul(slice_78, transpose_59, False, False)
        del slice_78, transpose_59

        # pd_op.full: (1xf32) <- ()
        full_22 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_19 = paddle._C_ops.scale(matmul_115, full_22, float("0"), True)
        del full_22, matmul_115

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_19 = paddle._C_ops.softmax(scale_19, -1)
        del scale_19

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_116 = paddle._C_ops.matmul(softmax_19, slice_80, False, False)
        del slice_80, softmax_19

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_60 = paddle._C_ops.transpose(matmul_116, [0, 2, 1, 3])
        del matmul_116

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_201 = [-1, 257, 1024]

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_39 = paddle._C_ops.reshape(transpose_60, full_int_array_201)
        del full_int_array_201, transpose_60

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_117 = paddle._C_ops.matmul(reshape_39, parameter_59, False, False)
        del parameter_59, reshape_39

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_117 = paddle._C_ops.add(matmul_117, parameter_58)
        del matmul_117, parameter_58

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_118 = paddle._C_ops.add(add_115, add_117)
        del add_115, add_117

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_120, layer_norm_121, layer_norm_122 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_118, parameter_57, parameter_56, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_56, parameter_57

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_118 = paddle._C_ops.matmul(layer_norm_120, parameter_55, False, False)
        del layer_norm_120, parameter_55

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_119 = paddle._C_ops.add(matmul_118, parameter_54)
        del matmul_118, parameter_54

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_19 = paddle._C_ops.gelu(add_119, False)
        del add_119

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_119 = paddle._C_ops.matmul(gelu_19, parameter_53, False, False)
        del gelu_19, parameter_53

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_120 = paddle._C_ops.add(matmul_119, parameter_52)
        del matmul_119, parameter_52

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_121 = paddle._C_ops.add(add_118, add_120)
        del add_118, add_120

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_123, layer_norm_124, layer_norm_125 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_121, parameter_51, parameter_50, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_50, parameter_51

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_21 = paddle._C_ops.shape64(layer_norm_123)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_202 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_203 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_81 = paddle._C_ops.slice(
            shape64_21, [0], full_int_array_202, full_int_array_203, [1], [0]
        )
        del full_int_array_202, full_int_array_203, shape64_21

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_120 = paddle._C_ops.matmul(layer_norm_123, parameter_49, False, False)
        del layer_norm_123, parameter_49

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_122 = paddle._C_ops.add(matmul_120, parameter_48)
        del matmul_120, parameter_48

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_204 = [-1, 257, 3, 16, 64]

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_40 = paddle._C_ops.reshape(add_122, full_int_array_204)
        del add_122, full_int_array_204

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_61 = paddle._C_ops.transpose(reshape_40, [2, 0, 3, 1, 4])
        del reshape_40

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_205 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_206 = [1]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_82 = paddle._C_ops.slice(
            transpose_61, [0], full_int_array_205, full_int_array_206, [1], [0]
        )
        del full_int_array_205, full_int_array_206

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_207 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_208 = [2]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_83 = paddle._C_ops.slice(
            transpose_61, [0], full_int_array_207, full_int_array_208, [1], [0]
        )
        del full_int_array_207, full_int_array_208

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_209 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_210 = [3]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_84 = paddle._C_ops.slice(
            transpose_61, [0], full_int_array_209, full_int_array_210, [1], [0]
        )
        del full_int_array_209, full_int_array_210, transpose_61

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_62 = paddle._C_ops.transpose(slice_83, [0, 1, 3, 2])
        del slice_83

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_121 = paddle._C_ops.matmul(slice_82, transpose_62, False, False)
        del slice_82, transpose_62

        # pd_op.full: (1xf32) <- ()
        full_23 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_20 = paddle._C_ops.scale(matmul_121, full_23, float("0"), True)
        del full_23, matmul_121

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_20 = paddle._C_ops.softmax(scale_20, -1)
        del scale_20

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_122 = paddle._C_ops.matmul(softmax_20, slice_84, False, False)
        del slice_84, softmax_20

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_63 = paddle._C_ops.transpose(matmul_122, [0, 2, 1, 3])
        del matmul_122

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_211 = [-1, 257, 1024]

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_41 = paddle._C_ops.reshape(transpose_63, full_int_array_211)
        del full_int_array_211, transpose_63

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_123 = paddle._C_ops.matmul(reshape_41, parameter_47, False, False)
        del parameter_47, reshape_41

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_123 = paddle._C_ops.add(matmul_123, parameter_46)
        del matmul_123, parameter_46

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_124 = paddle._C_ops.add(add_121, add_123)
        del add_121, add_123

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_126, layer_norm_127, layer_norm_128 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_124, parameter_45, parameter_44, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_44, parameter_45

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_124 = paddle._C_ops.matmul(layer_norm_126, parameter_43, False, False)
        del layer_norm_126, parameter_43

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_125 = paddle._C_ops.add(matmul_124, parameter_42)
        del matmul_124, parameter_42

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_20 = paddle._C_ops.gelu(add_125, False)
        del add_125

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_125 = paddle._C_ops.matmul(gelu_20, parameter_41, False, False)
        del gelu_20, parameter_41

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_126 = paddle._C_ops.add(matmul_125, parameter_40)
        del matmul_125, parameter_40

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_127 = paddle._C_ops.add(add_124, add_126)
        del add_124, add_126

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_129, layer_norm_130, layer_norm_131 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_127, parameter_39, parameter_38, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_38, parameter_39

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_22 = paddle._C_ops.shape64(layer_norm_129)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_212 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_213 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_85 = paddle._C_ops.slice(
            shape64_22, [0], full_int_array_212, full_int_array_213, [1], [0]
        )
        del full_int_array_212, full_int_array_213, shape64_22

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_126 = paddle._C_ops.matmul(layer_norm_129, parameter_37, False, False)
        del layer_norm_129, parameter_37

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_128 = paddle._C_ops.add(matmul_126, parameter_36)
        del matmul_126, parameter_36

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_214 = [-1, 257, 3, 16, 64]

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_42 = paddle._C_ops.reshape(add_128, full_int_array_214)
        del add_128, full_int_array_214

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_64 = paddle._C_ops.transpose(reshape_42, [2, 0, 3, 1, 4])
        del reshape_42

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_215 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_216 = [1]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_86 = paddle._C_ops.slice(
            transpose_64, [0], full_int_array_215, full_int_array_216, [1], [0]
        )
        del full_int_array_215, full_int_array_216

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_217 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_218 = [2]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_87 = paddle._C_ops.slice(
            transpose_64, [0], full_int_array_217, full_int_array_218, [1], [0]
        )
        del full_int_array_217, full_int_array_218

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_219 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_220 = [3]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_88 = paddle._C_ops.slice(
            transpose_64, [0], full_int_array_219, full_int_array_220, [1], [0]
        )
        del full_int_array_219, full_int_array_220, transpose_64

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_65 = paddle._C_ops.transpose(slice_87, [0, 1, 3, 2])
        del slice_87

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_127 = paddle._C_ops.matmul(slice_86, transpose_65, False, False)
        del slice_86, transpose_65

        # pd_op.full: (1xf32) <- ()
        full_24 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_21 = paddle._C_ops.scale(matmul_127, full_24, float("0"), True)
        del full_24, matmul_127

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_21 = paddle._C_ops.softmax(scale_21, -1)
        del scale_21

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_128 = paddle._C_ops.matmul(softmax_21, slice_88, False, False)
        del slice_88, softmax_21

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_66 = paddle._C_ops.transpose(matmul_128, [0, 2, 1, 3])
        del matmul_128

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_221 = [-1, 257, 1024]

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_43 = paddle._C_ops.reshape(transpose_66, full_int_array_221)
        del full_int_array_221, transpose_66

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_129 = paddle._C_ops.matmul(reshape_43, parameter_35, False, False)
        del parameter_35, reshape_43

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_129 = paddle._C_ops.add(matmul_129, parameter_34)
        del matmul_129, parameter_34

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_130 = paddle._C_ops.add(add_127, add_129)
        del add_127, add_129

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_132, layer_norm_133, layer_norm_134 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_130, parameter_33, parameter_32, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_32, parameter_33

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_130 = paddle._C_ops.matmul(layer_norm_132, parameter_31, False, False)
        del layer_norm_132, parameter_31

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_131 = paddle._C_ops.add(matmul_130, parameter_30)
        del matmul_130, parameter_30

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_21 = paddle._C_ops.gelu(add_131, False)
        del add_131

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_131 = paddle._C_ops.matmul(gelu_21, parameter_29, False, False)
        del gelu_21, parameter_29

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_132 = paddle._C_ops.add(matmul_131, parameter_28)
        del matmul_131, parameter_28

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_133 = paddle._C_ops.add(add_130, add_132)
        del add_130, add_132

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_135, layer_norm_136, layer_norm_137 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_133, parameter_27, parameter_26, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_26, parameter_27

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_23 = paddle._C_ops.shape64(layer_norm_135)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_222 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_223 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_89 = paddle._C_ops.slice(
            shape64_23, [0], full_int_array_222, full_int_array_223, [1], [0]
        )
        del full_int_array_222, full_int_array_223, shape64_23

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_132 = paddle._C_ops.matmul(layer_norm_135, parameter_25, False, False)
        del layer_norm_135, parameter_25

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_134 = paddle._C_ops.add(matmul_132, parameter_24)
        del matmul_132, parameter_24

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_224 = [-1, 257, 3, 16, 64]

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_44 = paddle._C_ops.reshape(add_134, full_int_array_224)
        del add_134, full_int_array_224

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_67 = paddle._C_ops.transpose(reshape_44, [2, 0, 3, 1, 4])
        del reshape_44

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_225 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_226 = [1]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_90 = paddle._C_ops.slice(
            transpose_67, [0], full_int_array_225, full_int_array_226, [1], [0]
        )
        del full_int_array_225, full_int_array_226

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_227 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_228 = [2]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_91 = paddle._C_ops.slice(
            transpose_67, [0], full_int_array_227, full_int_array_228, [1], [0]
        )
        del full_int_array_227, full_int_array_228

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_229 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_230 = [3]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_92 = paddle._C_ops.slice(
            transpose_67, [0], full_int_array_229, full_int_array_230, [1], [0]
        )
        del full_int_array_229, full_int_array_230, transpose_67

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_68 = paddle._C_ops.transpose(slice_91, [0, 1, 3, 2])
        del slice_91

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_133 = paddle._C_ops.matmul(slice_90, transpose_68, False, False)
        del slice_90, transpose_68

        # pd_op.full: (1xf32) <- ()
        full_25 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_22 = paddle._C_ops.scale(matmul_133, full_25, float("0"), True)
        del full_25, matmul_133

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_22 = paddle._C_ops.softmax(scale_22, -1)
        del scale_22

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_134 = paddle._C_ops.matmul(softmax_22, slice_92, False, False)
        del slice_92, softmax_22

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_69 = paddle._C_ops.transpose(matmul_134, [0, 2, 1, 3])
        del matmul_134

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_231 = [-1, 257, 1024]

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_45 = paddle._C_ops.reshape(transpose_69, full_int_array_231)
        del full_int_array_231, transpose_69

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_135 = paddle._C_ops.matmul(reshape_45, parameter_23, False, False)
        del parameter_23, reshape_45

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_135 = paddle._C_ops.add(matmul_135, parameter_22)
        del matmul_135, parameter_22

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_136 = paddle._C_ops.add(add_133, add_135)
        del add_133, add_135

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_138, layer_norm_139, layer_norm_140 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_136, parameter_21, parameter_20, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_20, parameter_21

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_136 = paddle._C_ops.matmul(layer_norm_138, parameter_19, False, False)
        del layer_norm_138, parameter_19

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_137 = paddle._C_ops.add(matmul_136, parameter_18)
        del matmul_136, parameter_18

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_22 = paddle._C_ops.gelu(add_137, False)
        del add_137

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_137 = paddle._C_ops.matmul(gelu_22, parameter_17, False, False)
        del gelu_22, parameter_17

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_138 = paddle._C_ops.add(matmul_137, parameter_16)
        del matmul_137, parameter_16

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_139 = paddle._C_ops.add(add_136, add_138)
        del add_136, add_138

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_141, layer_norm_142, layer_norm_143 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_139, parameter_15, parameter_14, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_14, parameter_15

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_24 = paddle._C_ops.shape64(layer_norm_141)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_232 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_233 = [1]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_93 = paddle._C_ops.slice(
            shape64_24, [0], full_int_array_232, full_int_array_233, [1], [0]
        )
        del full_int_array_232, full_int_array_233, shape64_24

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_138 = paddle._C_ops.matmul(layer_norm_141, parameter_13, False, False)
        del layer_norm_141, parameter_13

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_140 = paddle._C_ops.add(matmul_138, parameter_12)
        del matmul_138, parameter_12

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_234 = [-1, 257, 3, 16, 64]

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_46 = paddle._C_ops.reshape(add_140, full_int_array_234)
        del add_140, full_int_array_234

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_70 = paddle._C_ops.transpose(reshape_46, [2, 0, 3, 1, 4])
        del reshape_46

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_235 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_236 = [1]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_94 = paddle._C_ops.slice(
            transpose_70, [0], full_int_array_235, full_int_array_236, [1], [0]
        )
        del full_int_array_235, full_int_array_236

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_237 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_238 = [2]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_95 = paddle._C_ops.slice(
            transpose_70, [0], full_int_array_237, full_int_array_238, [1], [0]
        )
        del full_int_array_237, full_int_array_238

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_239 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_240 = [3]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_96 = paddle._C_ops.slice(
            transpose_70, [0], full_int_array_239, full_int_array_240, [1], [0]
        )
        del full_int_array_239, full_int_array_240, transpose_70

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_71 = paddle._C_ops.transpose(slice_95, [0, 1, 3, 2])
        del slice_95

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_139 = paddle._C_ops.matmul(slice_94, transpose_71, False, False)
        del slice_94, transpose_71

        # pd_op.full: (1xf32) <- ()
        full_26 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_23 = paddle._C_ops.scale(matmul_139, full_26, float("0"), True)
        del full_26, matmul_139

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_23 = paddle._C_ops.softmax(scale_23, -1)
        del scale_23

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_140 = paddle._C_ops.matmul(softmax_23, slice_96, False, False)
        del slice_96, softmax_23

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_72 = paddle._C_ops.transpose(matmul_140, [0, 2, 1, 3])
        del matmul_140

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_241 = [-1, 257, 1024]

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_47 = paddle._C_ops.reshape(transpose_72, full_int_array_241)
        del full_int_array_241, transpose_72

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_141 = paddle._C_ops.matmul(reshape_47, parameter_11, False, False)
        del parameter_11, reshape_47

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_141 = paddle._C_ops.add(matmul_141, parameter_10)
        del matmul_141, parameter_10

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_142 = paddle._C_ops.add(add_139, add_141)
        del add_139, add_141

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_144, layer_norm_145, layer_norm_146 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_142, parameter_9, parameter_8, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_8, parameter_9

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_142 = paddle._C_ops.matmul(layer_norm_144, parameter_7, False, False)
        del layer_norm_144, parameter_7

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_143 = paddle._C_ops.add(matmul_142, parameter_6)
        del matmul_142, parameter_6

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_23 = paddle._C_ops.gelu(add_143, False)
        del add_143

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_143 = paddle._C_ops.matmul(gelu_23, parameter_5, False, False)
        del gelu_23, parameter_5

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_144 = paddle._C_ops.add(matmul_143, parameter_4)
        del matmul_143, parameter_4

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_145 = paddle._C_ops.add(add_142, add_144)
        del add_142, add_144

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_242 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_243 = [1]

        # pd_op.slice: (-1x1024xf32) <- (-1x257x1024xf32, 1xi64, 1xi64)
        slice_97 = paddle._C_ops.slice(
            add_145, [1], full_int_array_242, full_int_array_243, [1], [1]
        )
        del full_int_array_242, full_int_array_243

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_244 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_245 = [2147483647]

        # pd_op.slice: (-1x256x1024xf32) <- (-1x257x1024xf32, 1xi64, 1xi64)
        slice_98 = paddle._C_ops.slice(
            add_145, [1], full_int_array_244, full_int_array_245, [1], []
        )
        del add_145, full_int_array_244, full_int_array_245

        # pd_op.layer_norm: (-1x1024xf32, -1xf32, -1xf32) <- (-1x1024xf32, 1024xf32, 1024xf32)
        layer_norm_147, layer_norm_148, layer_norm_149 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                slice_97, parameter_3, parameter_2, float("1e-05"), 1
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_2, parameter_3, slice_97

        # pd_op.matmul: (-1x102xf32) <- (-1x1024xf32, 1024x102xf32)
        matmul_144 = paddle._C_ops.matmul(layer_norm_147, parameter_1, False, False)
        del layer_norm_147, parameter_1

        # pd_op.add: (-1x102xf32) <- (-1x102xf32, 102xf32)
        add_0 = paddle._C_ops.add(matmul_144, parameter_0)
        del matmul_144, parameter_0

        return add_0
