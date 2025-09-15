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
        # pd_op.conv2d: (-1x1024x16x16xf32) <- (-1x3x224x224xf32, 1024x3x14x14xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_5, parameter_294, [14, 14], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_5, parameter_294

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
        del shape64_0

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

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_0 = [data_4, full_0, full_0]
        del data_4, full_0

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.expand: (-1x1x1024xf32) <- (1x1x1024xf32, 3xi64)
        expand_0 = paddle._C_ops.expand(data_2, stack_0)
        del data_2, stack_0

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x1x1024xf32, -1x256x1024xf32]) <- (-1x1x1024xf32, -1x256x1024xf32)
        combine_1 = [expand_0, transpose_0]
        del expand_0, transpose_0

        # pd_op.concat: (-1x257x1024xf32) <- ([-1x1x1024xf32, -1x256x1024xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_1, full_1)
        del combine_1, full_1

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1x257x1024xf32)
        add_0 = paddle._C_ops.add(concat_0, data_3)
        del concat_0, data_3

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_0, parameter_293, parameter_292, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_0, parameter_292, parameter_293

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

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_1

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_1 = paddle._C_ops.matmul(layer_norm_3, parameter_289, False, False)
        del layer_norm_3, parameter_289

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_1 = paddle._C_ops.add(matmul_1, parameter_288)
        del matmul_1, parameter_288

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_2 = [-1, 257, 3, 16, 64]

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_0 = paddle._C_ops.reshape(add_1, full_int_array_2)
        del add_1

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_0, [2, 0, 3, 1, 4])
        del reshape_0

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            transpose_1, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [2]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            transpose_1, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [3]

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            transpose_1, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del transpose_1

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_2 = paddle._C_ops.transpose(slice_3, [0, 1, 3, 2])
        del slice_3

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_2 = paddle._C_ops.matmul(slice_2, transpose_2, False, False)
        del slice_2, transpose_2

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0.125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(matmul_2, full_2, float("0"), True)
        del matmul_2

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_0 = paddle._C_ops.softmax(scale_0, -1)
        del scale_0

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_3 = paddle._C_ops.matmul(softmax_0, slice_4, False, False)
        del slice_4, softmax_0

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_3 = paddle._C_ops.transpose(matmul_3, [0, 2, 1, 3])
        del matmul_3

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_5 = [-1, 257, 1024]

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_1 = paddle._C_ops.reshape(transpose_3, full_int_array_5)
        del transpose_3

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_4 = paddle._C_ops.matmul(reshape_1, parameter_287, False, False)
        del parameter_287, reshape_1

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_2 = paddle._C_ops.add(matmul_4, parameter_286)
        del matmul_4, parameter_286

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_3 = paddle._C_ops.add(layer_norm_0, add_2)
        del add_2, layer_norm_0

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_3, parameter_285, parameter_284, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_284, parameter_285

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_5 = paddle._C_ops.matmul(layer_norm_6, parameter_283, False, False)
        del layer_norm_6, parameter_283

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_4 = paddle._C_ops.add(matmul_5, parameter_282)
        del matmul_5, parameter_282

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_0 = paddle._C_ops.gelu(add_4, False)
        del add_4

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_6 = paddle._C_ops.matmul(gelu_0, parameter_281, False, False)
        del gelu_0, parameter_281

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_5 = paddle._C_ops.add(matmul_6, parameter_280)
        del matmul_6, parameter_280

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_6 = paddle._C_ops.add(add_3, add_5)
        del add_3, add_5

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_6, parameter_279, parameter_278, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_278, parameter_279

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_2 = paddle._C_ops.shape64(layer_norm_9)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_2

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_7 = paddle._C_ops.matmul(layer_norm_9, parameter_277, False, False)
        del layer_norm_9, parameter_277

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_7 = paddle._C_ops.add(matmul_7, parameter_276)
        del matmul_7, parameter_276

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_2 = paddle._C_ops.reshape(add_7, full_int_array_2)
        del add_7

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_2, [2, 0, 3, 1, 4])
        del reshape_2

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            transpose_4, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            transpose_4, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            transpose_4, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del transpose_4

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_5 = paddle._C_ops.transpose(slice_7, [0, 1, 3, 2])
        del slice_7

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_8 = paddle._C_ops.matmul(slice_6, transpose_5, False, False)
        del slice_6, transpose_5

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(matmul_8, full_2, float("0"), True)
        del matmul_8

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_1 = paddle._C_ops.softmax(scale_1, -1)
        del scale_1

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_9 = paddle._C_ops.matmul(softmax_1, slice_8, False, False)
        del slice_8, softmax_1

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_6 = paddle._C_ops.transpose(matmul_9, [0, 2, 1, 3])
        del matmul_9

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_6, full_int_array_5)
        del transpose_6

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_10 = paddle._C_ops.matmul(reshape_3, parameter_275, False, False)
        del parameter_275, reshape_3

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_8 = paddle._C_ops.add(matmul_10, parameter_274)
        del matmul_10, parameter_274

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_9 = paddle._C_ops.add(add_6, add_8)
        del add_6, add_8

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_9, parameter_273, parameter_272, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_272, parameter_273

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_11 = paddle._C_ops.matmul(layer_norm_12, parameter_271, False, False)
        del layer_norm_12, parameter_271

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_10 = paddle._C_ops.add(matmul_11, parameter_270)
        del matmul_11, parameter_270

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_1 = paddle._C_ops.gelu(add_10, False)
        del add_10

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_12 = paddle._C_ops.matmul(gelu_1, parameter_269, False, False)
        del gelu_1, parameter_269

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_11 = paddle._C_ops.add(matmul_12, parameter_268)
        del matmul_12, parameter_268

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_12 = paddle._C_ops.add(add_9, add_11)
        del add_11, add_9

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_12, parameter_267, parameter_266, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_266, parameter_267

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_3 = paddle._C_ops.shape64(layer_norm_15)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_3

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_13 = paddle._C_ops.matmul(layer_norm_15, parameter_265, False, False)
        del layer_norm_15, parameter_265

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_13 = paddle._C_ops.add(matmul_13, parameter_264)
        del matmul_13, parameter_264

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_4 = paddle._C_ops.reshape(add_13, full_int_array_2)
        del add_13

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_7 = paddle._C_ops.transpose(reshape_4, [2, 0, 3, 1, 4])
        del reshape_4

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            transpose_7, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            transpose_7, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            transpose_7, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del transpose_7

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_8 = paddle._C_ops.transpose(slice_11, [0, 1, 3, 2])
        del slice_11

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_14 = paddle._C_ops.matmul(slice_10, transpose_8, False, False)
        del slice_10, transpose_8

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(matmul_14, full_2, float("0"), True)
        del matmul_14

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_2 = paddle._C_ops.softmax(scale_2, -1)
        del scale_2

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_15 = paddle._C_ops.matmul(softmax_2, slice_12, False, False)
        del slice_12, softmax_2

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_9 = paddle._C_ops.transpose(matmul_15, [0, 2, 1, 3])
        del matmul_15

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_5 = paddle._C_ops.reshape(transpose_9, full_int_array_5)
        del transpose_9

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_16 = paddle._C_ops.matmul(reshape_5, parameter_263, False, False)
        del parameter_263, reshape_5

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_14 = paddle._C_ops.add(matmul_16, parameter_262)
        del matmul_16, parameter_262

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_15 = paddle._C_ops.add(add_12, add_14)
        del add_12, add_14

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_15, parameter_261, parameter_260, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_260, parameter_261

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_17 = paddle._C_ops.matmul(layer_norm_18, parameter_259, False, False)
        del layer_norm_18, parameter_259

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_16 = paddle._C_ops.add(matmul_17, parameter_258)
        del matmul_17, parameter_258

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_2 = paddle._C_ops.gelu(add_16, False)
        del add_16

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_18 = paddle._C_ops.matmul(gelu_2, parameter_257, False, False)
        del gelu_2, parameter_257

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_17 = paddle._C_ops.add(matmul_18, parameter_256)
        del matmul_18, parameter_256

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_18 = paddle._C_ops.add(add_15, add_17)
        del add_15, add_17

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_18, parameter_255, parameter_254, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_254, parameter_255

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_4 = paddle._C_ops.shape64(layer_norm_21)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            shape64_4, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_4

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_19 = paddle._C_ops.matmul(layer_norm_21, parameter_253, False, False)
        del layer_norm_21, parameter_253

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_19 = paddle._C_ops.add(matmul_19, parameter_252)
        del matmul_19, parameter_252

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_6 = paddle._C_ops.reshape(add_19, full_int_array_2)
        del add_19

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_10 = paddle._C_ops.transpose(reshape_6, [2, 0, 3, 1, 4])
        del reshape_6

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            transpose_10, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            transpose_10, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            transpose_10, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del transpose_10

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_11 = paddle._C_ops.transpose(slice_15, [0, 1, 3, 2])
        del slice_15

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_20 = paddle._C_ops.matmul(slice_14, transpose_11, False, False)
        del slice_14, transpose_11

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(matmul_20, full_2, float("0"), True)
        del matmul_20

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_3 = paddle._C_ops.softmax(scale_3, -1)
        del scale_3

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_21 = paddle._C_ops.matmul(softmax_3, slice_16, False, False)
        del slice_16, softmax_3

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_12 = paddle._C_ops.transpose(matmul_21, [0, 2, 1, 3])
        del matmul_21

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_12, full_int_array_5)
        del transpose_12

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_22 = paddle._C_ops.matmul(reshape_7, parameter_251, False, False)
        del parameter_251, reshape_7

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_20 = paddle._C_ops.add(matmul_22, parameter_250)
        del matmul_22, parameter_250

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_21 = paddle._C_ops.add(add_18, add_20)
        del add_18, add_20

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_21, parameter_249, parameter_248, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_248, parameter_249

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_23 = paddle._C_ops.matmul(layer_norm_24, parameter_247, False, False)
        del layer_norm_24, parameter_247

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_22 = paddle._C_ops.add(matmul_23, parameter_246)
        del matmul_23, parameter_246

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_3 = paddle._C_ops.gelu(add_22, False)
        del add_22

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_24 = paddle._C_ops.matmul(gelu_3, parameter_245, False, False)
        del gelu_3, parameter_245

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_23 = paddle._C_ops.add(matmul_24, parameter_244)
        del matmul_24, parameter_244

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_24 = paddle._C_ops.add(add_21, add_23)
        del add_21, add_23

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_24, parameter_243, parameter_242, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_242, parameter_243

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_5 = paddle._C_ops.shape64(layer_norm_27)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            shape64_5, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_5

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_25 = paddle._C_ops.matmul(layer_norm_27, parameter_241, False, False)
        del layer_norm_27, parameter_241

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_25 = paddle._C_ops.add(matmul_25, parameter_240)
        del matmul_25, parameter_240

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_8 = paddle._C_ops.reshape(add_25, full_int_array_2)
        del add_25

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_8, [2, 0, 3, 1, 4])
        del reshape_8

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(
            transpose_13, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(
            transpose_13, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(
            transpose_13, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del transpose_13

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_14 = paddle._C_ops.transpose(slice_19, [0, 1, 3, 2])
        del slice_19

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_26 = paddle._C_ops.matmul(slice_18, transpose_14, False, False)
        del slice_18, transpose_14

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(matmul_26, full_2, float("0"), True)
        del matmul_26

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_4 = paddle._C_ops.softmax(scale_4, -1)
        del scale_4

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_27 = paddle._C_ops.matmul(softmax_4, slice_20, False, False)
        del slice_20, softmax_4

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_15 = paddle._C_ops.transpose(matmul_27, [0, 2, 1, 3])
        del matmul_27

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_9 = paddle._C_ops.reshape(transpose_15, full_int_array_5)
        del transpose_15

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_28 = paddle._C_ops.matmul(reshape_9, parameter_239, False, False)
        del parameter_239, reshape_9

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_26 = paddle._C_ops.add(matmul_28, parameter_238)
        del matmul_28, parameter_238

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_27 = paddle._C_ops.add(add_24, add_26)
        del add_24, add_26

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_27, parameter_237, parameter_236, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_236, parameter_237

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_29 = paddle._C_ops.matmul(layer_norm_30, parameter_235, False, False)
        del layer_norm_30, parameter_235

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_28 = paddle._C_ops.add(matmul_29, parameter_234)
        del matmul_29, parameter_234

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_4 = paddle._C_ops.gelu(add_28, False)
        del add_28

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_30 = paddle._C_ops.matmul(gelu_4, parameter_233, False, False)
        del gelu_4, parameter_233

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_29 = paddle._C_ops.add(matmul_30, parameter_232)
        del matmul_30, parameter_232

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_30 = paddle._C_ops.add(add_27, add_29)
        del add_27, add_29

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_30, parameter_231, parameter_230, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_230, parameter_231

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_6 = paddle._C_ops.shape64(layer_norm_33)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(
            shape64_6, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_6

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_31 = paddle._C_ops.matmul(layer_norm_33, parameter_229, False, False)
        del layer_norm_33, parameter_229

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_31 = paddle._C_ops.add(matmul_31, parameter_228)
        del matmul_31, parameter_228

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_10 = paddle._C_ops.reshape(add_31, full_int_array_2)
        del add_31

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_16 = paddle._C_ops.transpose(reshape_10, [2, 0, 3, 1, 4])
        del reshape_10

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(
            transpose_16, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(
            transpose_16, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(
            transpose_16, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del transpose_16

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_17 = paddle._C_ops.transpose(slice_23, [0, 1, 3, 2])
        del slice_23

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_32 = paddle._C_ops.matmul(slice_22, transpose_17, False, False)
        del slice_22, transpose_17

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(matmul_32, full_2, float("0"), True)
        del matmul_32

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_5 = paddle._C_ops.softmax(scale_5, -1)
        del scale_5

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_33 = paddle._C_ops.matmul(softmax_5, slice_24, False, False)
        del slice_24, softmax_5

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_18 = paddle._C_ops.transpose(matmul_33, [0, 2, 1, 3])
        del matmul_33

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_18, full_int_array_5)
        del transpose_18

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_34 = paddle._C_ops.matmul(reshape_11, parameter_227, False, False)
        del parameter_227, reshape_11

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_32 = paddle._C_ops.add(matmul_34, parameter_226)
        del matmul_34, parameter_226

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_33 = paddle._C_ops.add(add_30, add_32)
        del add_30, add_32

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_33, parameter_225, parameter_224, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_224, parameter_225

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_35 = paddle._C_ops.matmul(layer_norm_36, parameter_223, False, False)
        del layer_norm_36, parameter_223

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_34 = paddle._C_ops.add(matmul_35, parameter_222)
        del matmul_35, parameter_222

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_5 = paddle._C_ops.gelu(add_34, False)
        del add_34

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_36 = paddle._C_ops.matmul(gelu_5, parameter_221, False, False)
        del gelu_5, parameter_221

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_35 = paddle._C_ops.add(matmul_36, parameter_220)
        del matmul_36, parameter_220

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_36 = paddle._C_ops.add(add_33, add_35)
        del add_33, add_35

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_36, parameter_219, parameter_218, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_218, parameter_219

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_7 = paddle._C_ops.shape64(layer_norm_39)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(
            shape64_7, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_7

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_37 = paddle._C_ops.matmul(layer_norm_39, parameter_217, False, False)
        del layer_norm_39, parameter_217

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_37 = paddle._C_ops.add(matmul_37, parameter_216)
        del matmul_37, parameter_216

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_12 = paddle._C_ops.reshape(add_37, full_int_array_2)
        del add_37

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_19 = paddle._C_ops.transpose(reshape_12, [2, 0, 3, 1, 4])
        del reshape_12

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(
            transpose_19, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(
            transpose_19, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(
            transpose_19, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del transpose_19

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_20 = paddle._C_ops.transpose(slice_27, [0, 1, 3, 2])
        del slice_27

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_38 = paddle._C_ops.matmul(slice_26, transpose_20, False, False)
        del slice_26, transpose_20

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(matmul_38, full_2, float("0"), True)
        del matmul_38

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_6 = paddle._C_ops.softmax(scale_6, -1)
        del scale_6

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_39 = paddle._C_ops.matmul(softmax_6, slice_28, False, False)
        del slice_28, softmax_6

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_21 = paddle._C_ops.transpose(matmul_39, [0, 2, 1, 3])
        del matmul_39

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_13 = paddle._C_ops.reshape(transpose_21, full_int_array_5)
        del transpose_21

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_40 = paddle._C_ops.matmul(reshape_13, parameter_215, False, False)
        del parameter_215, reshape_13

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_38 = paddle._C_ops.add(matmul_40, parameter_214)
        del matmul_40, parameter_214

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_39 = paddle._C_ops.add(add_36, add_38)
        del add_36, add_38

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_39, parameter_213, parameter_212, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_212, parameter_213

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_41 = paddle._C_ops.matmul(layer_norm_42, parameter_211, False, False)
        del layer_norm_42, parameter_211

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_40 = paddle._C_ops.add(matmul_41, parameter_210)
        del matmul_41, parameter_210

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_6 = paddle._C_ops.gelu(add_40, False)
        del add_40

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_42 = paddle._C_ops.matmul(gelu_6, parameter_209, False, False)
        del gelu_6, parameter_209

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_41 = paddle._C_ops.add(matmul_42, parameter_208)
        del matmul_42, parameter_208

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_42 = paddle._C_ops.add(add_39, add_41)
        del add_39, add_41

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_42, parameter_207, parameter_206, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_206, parameter_207

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_8 = paddle._C_ops.shape64(layer_norm_45)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(
            shape64_8, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_8

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_43 = paddle._C_ops.matmul(layer_norm_45, parameter_205, False, False)
        del layer_norm_45, parameter_205

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_43 = paddle._C_ops.add(matmul_43, parameter_204)
        del matmul_43, parameter_204

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_14 = paddle._C_ops.reshape(add_43, full_int_array_2)
        del add_43

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_22 = paddle._C_ops.transpose(reshape_14, [2, 0, 3, 1, 4])
        del reshape_14

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(
            transpose_22, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(
            transpose_22, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(
            transpose_22, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del transpose_22

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_23 = paddle._C_ops.transpose(slice_31, [0, 1, 3, 2])
        del slice_31

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_44 = paddle._C_ops.matmul(slice_30, transpose_23, False, False)
        del slice_30, transpose_23

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(matmul_44, full_2, float("0"), True)
        del matmul_44

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_7 = paddle._C_ops.softmax(scale_7, -1)
        del scale_7

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_45 = paddle._C_ops.matmul(softmax_7, slice_32, False, False)
        del slice_32, softmax_7

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_24 = paddle._C_ops.transpose(matmul_45, [0, 2, 1, 3])
        del matmul_45

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_15 = paddle._C_ops.reshape(transpose_24, full_int_array_5)
        del transpose_24

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_46 = paddle._C_ops.matmul(reshape_15, parameter_203, False, False)
        del parameter_203, reshape_15

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_44 = paddle._C_ops.add(matmul_46, parameter_202)
        del matmul_46, parameter_202

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_45 = paddle._C_ops.add(add_42, add_44)
        del add_42, add_44

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_45, parameter_201, parameter_200, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_200, parameter_201

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_47 = paddle._C_ops.matmul(layer_norm_48, parameter_199, False, False)
        del layer_norm_48, parameter_199

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_46 = paddle._C_ops.add(matmul_47, parameter_198)
        del matmul_47, parameter_198

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_7 = paddle._C_ops.gelu(add_46, False)
        del add_46

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_48 = paddle._C_ops.matmul(gelu_7, parameter_197, False, False)
        del gelu_7, parameter_197

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_47 = paddle._C_ops.add(matmul_48, parameter_196)
        del matmul_48, parameter_196

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_48 = paddle._C_ops.add(add_45, add_47)
        del add_45, add_47

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_48, parameter_195, parameter_194, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_194, parameter_195

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_9 = paddle._C_ops.shape64(layer_norm_51)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_33 = paddle._C_ops.slice(
            shape64_9, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_9

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_49 = paddle._C_ops.matmul(layer_norm_51, parameter_193, False, False)
        del layer_norm_51, parameter_193

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_49 = paddle._C_ops.add(matmul_49, parameter_192)
        del matmul_49, parameter_192

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_16 = paddle._C_ops.reshape(add_49, full_int_array_2)
        del add_49

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_25 = paddle._C_ops.transpose(reshape_16, [2, 0, 3, 1, 4])
        del reshape_16

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_34 = paddle._C_ops.slice(
            transpose_25, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_35 = paddle._C_ops.slice(
            transpose_25, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(
            transpose_25, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del transpose_25

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_26 = paddle._C_ops.transpose(slice_35, [0, 1, 3, 2])
        del slice_35

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_50 = paddle._C_ops.matmul(slice_34, transpose_26, False, False)
        del slice_34, transpose_26

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(matmul_50, full_2, float("0"), True)
        del matmul_50

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_8 = paddle._C_ops.softmax(scale_8, -1)
        del scale_8

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_51 = paddle._C_ops.matmul(softmax_8, slice_36, False, False)
        del slice_36, softmax_8

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_27 = paddle._C_ops.transpose(matmul_51, [0, 2, 1, 3])
        del matmul_51

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_17 = paddle._C_ops.reshape(transpose_27, full_int_array_5)
        del transpose_27

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_52 = paddle._C_ops.matmul(reshape_17, parameter_191, False, False)
        del parameter_191, reshape_17

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_50 = paddle._C_ops.add(matmul_52, parameter_190)
        del matmul_52, parameter_190

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_51 = paddle._C_ops.add(add_48, add_50)
        del add_48, add_50

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_51, parameter_189, parameter_188, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_188, parameter_189

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_53 = paddle._C_ops.matmul(layer_norm_54, parameter_187, False, False)
        del layer_norm_54, parameter_187

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_52 = paddle._C_ops.add(matmul_53, parameter_186)
        del matmul_53, parameter_186

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_8 = paddle._C_ops.gelu(add_52, False)
        del add_52

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_54 = paddle._C_ops.matmul(gelu_8, parameter_185, False, False)
        del gelu_8, parameter_185

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_53 = paddle._C_ops.add(matmul_54, parameter_184)
        del matmul_54, parameter_184

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_54 = paddle._C_ops.add(add_51, add_53)
        del add_51, add_53

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_54, parameter_183, parameter_182, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_182, parameter_183

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_10 = paddle._C_ops.shape64(layer_norm_57)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_37 = paddle._C_ops.slice(
            shape64_10, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_10

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_55 = paddle._C_ops.matmul(layer_norm_57, parameter_181, False, False)
        del layer_norm_57, parameter_181

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_55 = paddle._C_ops.add(matmul_55, parameter_180)
        del matmul_55, parameter_180

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_18 = paddle._C_ops.reshape(add_55, full_int_array_2)
        del add_55

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_28 = paddle._C_ops.transpose(reshape_18, [2, 0, 3, 1, 4])
        del reshape_18

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_38 = paddle._C_ops.slice(
            transpose_28, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_39 = paddle._C_ops.slice(
            transpose_28, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_40 = paddle._C_ops.slice(
            transpose_28, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del transpose_28

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_29 = paddle._C_ops.transpose(slice_39, [0, 1, 3, 2])
        del slice_39

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_56 = paddle._C_ops.matmul(slice_38, transpose_29, False, False)
        del slice_38, transpose_29

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(matmul_56, full_2, float("0"), True)
        del matmul_56

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_9 = paddle._C_ops.softmax(scale_9, -1)
        del scale_9

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_57 = paddle._C_ops.matmul(softmax_9, slice_40, False, False)
        del slice_40, softmax_9

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_30 = paddle._C_ops.transpose(matmul_57, [0, 2, 1, 3])
        del matmul_57

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_19 = paddle._C_ops.reshape(transpose_30, full_int_array_5)
        del transpose_30

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_58 = paddle._C_ops.matmul(reshape_19, parameter_179, False, False)
        del parameter_179, reshape_19

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_56 = paddle._C_ops.add(matmul_58, parameter_178)
        del matmul_58, parameter_178

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_57 = paddle._C_ops.add(add_54, add_56)
        del add_54, add_56

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_57, parameter_177, parameter_176, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_176, parameter_177

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_59 = paddle._C_ops.matmul(layer_norm_60, parameter_175, False, False)
        del layer_norm_60, parameter_175

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_58 = paddle._C_ops.add(matmul_59, parameter_174)
        del matmul_59, parameter_174

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_9 = paddle._C_ops.gelu(add_58, False)
        del add_58

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_60 = paddle._C_ops.matmul(gelu_9, parameter_173, False, False)
        del gelu_9, parameter_173

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_59 = paddle._C_ops.add(matmul_60, parameter_172)
        del matmul_60, parameter_172

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_60 = paddle._C_ops.add(add_57, add_59)
        del add_57, add_59

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_60, parameter_171, parameter_170, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_170, parameter_171

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_11 = paddle._C_ops.shape64(layer_norm_63)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_41 = paddle._C_ops.slice(
            shape64_11, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_11

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_61 = paddle._C_ops.matmul(layer_norm_63, parameter_169, False, False)
        del layer_norm_63, parameter_169

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_61 = paddle._C_ops.add(matmul_61, parameter_168)
        del matmul_61, parameter_168

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_20 = paddle._C_ops.reshape(add_61, full_int_array_2)
        del add_61

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_31 = paddle._C_ops.transpose(reshape_20, [2, 0, 3, 1, 4])
        del reshape_20

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_42 = paddle._C_ops.slice(
            transpose_31, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_43 = paddle._C_ops.slice(
            transpose_31, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_44 = paddle._C_ops.slice(
            transpose_31, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del transpose_31

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_32 = paddle._C_ops.transpose(slice_43, [0, 1, 3, 2])
        del slice_43

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_62 = paddle._C_ops.matmul(slice_42, transpose_32, False, False)
        del slice_42, transpose_32

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(matmul_62, full_2, float("0"), True)
        del matmul_62

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_10 = paddle._C_ops.softmax(scale_10, -1)
        del scale_10

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_63 = paddle._C_ops.matmul(softmax_10, slice_44, False, False)
        del slice_44, softmax_10

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_33 = paddle._C_ops.transpose(matmul_63, [0, 2, 1, 3])
        del matmul_63

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_21 = paddle._C_ops.reshape(transpose_33, full_int_array_5)
        del transpose_33

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_64 = paddle._C_ops.matmul(reshape_21, parameter_167, False, False)
        del parameter_167, reshape_21

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_62 = paddle._C_ops.add(matmul_64, parameter_166)
        del matmul_64, parameter_166

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_63 = paddle._C_ops.add(add_60, add_62)
        del add_60, add_62

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_63, parameter_165, parameter_164, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_164, parameter_165

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_65 = paddle._C_ops.matmul(layer_norm_66, parameter_163, False, False)
        del layer_norm_66, parameter_163

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_64 = paddle._C_ops.add(matmul_65, parameter_162)
        del matmul_65, parameter_162

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_10 = paddle._C_ops.gelu(add_64, False)
        del add_64

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_66 = paddle._C_ops.matmul(gelu_10, parameter_161, False, False)
        del gelu_10, parameter_161

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_65 = paddle._C_ops.add(matmul_66, parameter_160)
        del matmul_66, parameter_160

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_66 = paddle._C_ops.add(add_63, add_65)
        del add_63, add_65

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_66, parameter_159, parameter_158, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_158, parameter_159

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_12 = paddle._C_ops.shape64(layer_norm_69)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_45 = paddle._C_ops.slice(
            shape64_12, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_12

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_67 = paddle._C_ops.matmul(layer_norm_69, parameter_157, False, False)
        del layer_norm_69, parameter_157

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_67 = paddle._C_ops.add(matmul_67, parameter_156)
        del matmul_67, parameter_156

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_22 = paddle._C_ops.reshape(add_67, full_int_array_2)
        del add_67

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_34 = paddle._C_ops.transpose(reshape_22, [2, 0, 3, 1, 4])
        del reshape_22

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_46 = paddle._C_ops.slice(
            transpose_34, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_47 = paddle._C_ops.slice(
            transpose_34, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_48 = paddle._C_ops.slice(
            transpose_34, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del transpose_34

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_35 = paddle._C_ops.transpose(slice_47, [0, 1, 3, 2])
        del slice_47

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_68 = paddle._C_ops.matmul(slice_46, transpose_35, False, False)
        del slice_46, transpose_35

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(matmul_68, full_2, float("0"), True)
        del matmul_68

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_11 = paddle._C_ops.softmax(scale_11, -1)
        del scale_11

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_69 = paddle._C_ops.matmul(softmax_11, slice_48, False, False)
        del slice_48, softmax_11

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_36 = paddle._C_ops.transpose(matmul_69, [0, 2, 1, 3])
        del matmul_69

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_23 = paddle._C_ops.reshape(transpose_36, full_int_array_5)
        del transpose_36

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_70 = paddle._C_ops.matmul(reshape_23, parameter_155, False, False)
        del parameter_155, reshape_23

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_68 = paddle._C_ops.add(matmul_70, parameter_154)
        del matmul_70, parameter_154

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_69 = paddle._C_ops.add(add_66, add_68)
        del add_66, add_68

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_72, layer_norm_73, layer_norm_74 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_69, parameter_153, parameter_152, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_152, parameter_153

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_71 = paddle._C_ops.matmul(layer_norm_72, parameter_151, False, False)
        del layer_norm_72, parameter_151

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_70 = paddle._C_ops.add(matmul_71, parameter_150)
        del matmul_71, parameter_150

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_11 = paddle._C_ops.gelu(add_70, False)
        del add_70

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_72 = paddle._C_ops.matmul(gelu_11, parameter_149, False, False)
        del gelu_11, parameter_149

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_71 = paddle._C_ops.add(matmul_72, parameter_148)
        del matmul_72, parameter_148

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_72 = paddle._C_ops.add(add_69, add_71)
        del add_69, add_71

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_75, layer_norm_76, layer_norm_77 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_72, parameter_147, parameter_146, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_146, parameter_147

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_13 = paddle._C_ops.shape64(layer_norm_75)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_49 = paddle._C_ops.slice(
            shape64_13, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_13

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_73 = paddle._C_ops.matmul(layer_norm_75, parameter_145, False, False)
        del layer_norm_75, parameter_145

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_73 = paddle._C_ops.add(matmul_73, parameter_144)
        del matmul_73, parameter_144

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_24 = paddle._C_ops.reshape(add_73, full_int_array_2)
        del add_73

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_37 = paddle._C_ops.transpose(reshape_24, [2, 0, 3, 1, 4])
        del reshape_24

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_50 = paddle._C_ops.slice(
            transpose_37, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_51 = paddle._C_ops.slice(
            transpose_37, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_52 = paddle._C_ops.slice(
            transpose_37, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del transpose_37

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_38 = paddle._C_ops.transpose(slice_51, [0, 1, 3, 2])
        del slice_51

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_74 = paddle._C_ops.matmul(slice_50, transpose_38, False, False)
        del slice_50, transpose_38

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(matmul_74, full_2, float("0"), True)
        del matmul_74

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_12 = paddle._C_ops.softmax(scale_12, -1)
        del scale_12

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_75 = paddle._C_ops.matmul(softmax_12, slice_52, False, False)
        del slice_52, softmax_12

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_39 = paddle._C_ops.transpose(matmul_75, [0, 2, 1, 3])
        del matmul_75

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_25 = paddle._C_ops.reshape(transpose_39, full_int_array_5)
        del transpose_39

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_76 = paddle._C_ops.matmul(reshape_25, parameter_143, False, False)
        del parameter_143, reshape_25

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_74 = paddle._C_ops.add(matmul_76, parameter_142)
        del matmul_76, parameter_142

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_75 = paddle._C_ops.add(add_72, add_74)
        del add_72, add_74

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_78, layer_norm_79, layer_norm_80 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_75, parameter_141, parameter_140, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_140, parameter_141

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_77 = paddle._C_ops.matmul(layer_norm_78, parameter_139, False, False)
        del layer_norm_78, parameter_139

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_76 = paddle._C_ops.add(matmul_77, parameter_138)
        del matmul_77, parameter_138

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_12 = paddle._C_ops.gelu(add_76, False)
        del add_76

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_78 = paddle._C_ops.matmul(gelu_12, parameter_137, False, False)
        del gelu_12, parameter_137

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_77 = paddle._C_ops.add(matmul_78, parameter_136)
        del matmul_78, parameter_136

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_78 = paddle._C_ops.add(add_75, add_77)
        del add_75, add_77

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_81, layer_norm_82, layer_norm_83 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_78, parameter_135, parameter_134, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_134, parameter_135

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_14 = paddle._C_ops.shape64(layer_norm_81)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_53 = paddle._C_ops.slice(
            shape64_14, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_14

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_79 = paddle._C_ops.matmul(layer_norm_81, parameter_133, False, False)
        del layer_norm_81, parameter_133

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_79 = paddle._C_ops.add(matmul_79, parameter_132)
        del matmul_79, parameter_132

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_26 = paddle._C_ops.reshape(add_79, full_int_array_2)
        del add_79

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_40 = paddle._C_ops.transpose(reshape_26, [2, 0, 3, 1, 4])
        del reshape_26

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_54 = paddle._C_ops.slice(
            transpose_40, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_55 = paddle._C_ops.slice(
            transpose_40, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_56 = paddle._C_ops.slice(
            transpose_40, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del transpose_40

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_41 = paddle._C_ops.transpose(slice_55, [0, 1, 3, 2])
        del slice_55

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_80 = paddle._C_ops.matmul(slice_54, transpose_41, False, False)
        del slice_54, transpose_41

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(matmul_80, full_2, float("0"), True)
        del matmul_80

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_13 = paddle._C_ops.softmax(scale_13, -1)
        del scale_13

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_81 = paddle._C_ops.matmul(softmax_13, slice_56, False, False)
        del slice_56, softmax_13

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_42 = paddle._C_ops.transpose(matmul_81, [0, 2, 1, 3])
        del matmul_81

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_27 = paddle._C_ops.reshape(transpose_42, full_int_array_5)
        del transpose_42

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_82 = paddle._C_ops.matmul(reshape_27, parameter_131, False, False)
        del parameter_131, reshape_27

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_80 = paddle._C_ops.add(matmul_82, parameter_130)
        del matmul_82, parameter_130

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_81 = paddle._C_ops.add(add_78, add_80)
        del add_78, add_80

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_84, layer_norm_85, layer_norm_86 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_81, parameter_129, parameter_128, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_128, parameter_129

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_83 = paddle._C_ops.matmul(layer_norm_84, parameter_127, False, False)
        del layer_norm_84, parameter_127

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_82 = paddle._C_ops.add(matmul_83, parameter_126)
        del matmul_83, parameter_126

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_13 = paddle._C_ops.gelu(add_82, False)
        del add_82

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_84 = paddle._C_ops.matmul(gelu_13, parameter_125, False, False)
        del gelu_13, parameter_125

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_83 = paddle._C_ops.add(matmul_84, parameter_124)
        del matmul_84, parameter_124

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_84 = paddle._C_ops.add(add_81, add_83)
        del add_81, add_83

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_87, layer_norm_88, layer_norm_89 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_84, parameter_123, parameter_122, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_122, parameter_123

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_15 = paddle._C_ops.shape64(layer_norm_87)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_57 = paddle._C_ops.slice(
            shape64_15, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_15

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_85 = paddle._C_ops.matmul(layer_norm_87, parameter_121, False, False)
        del layer_norm_87, parameter_121

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_85 = paddle._C_ops.add(matmul_85, parameter_120)
        del matmul_85, parameter_120

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_28 = paddle._C_ops.reshape(add_85, full_int_array_2)
        del add_85

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_43 = paddle._C_ops.transpose(reshape_28, [2, 0, 3, 1, 4])
        del reshape_28

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_58 = paddle._C_ops.slice(
            transpose_43, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_59 = paddle._C_ops.slice(
            transpose_43, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_60 = paddle._C_ops.slice(
            transpose_43, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del transpose_43

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_44 = paddle._C_ops.transpose(slice_59, [0, 1, 3, 2])
        del slice_59

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_86 = paddle._C_ops.matmul(slice_58, transpose_44, False, False)
        del slice_58, transpose_44

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(matmul_86, full_2, float("0"), True)
        del matmul_86

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_14 = paddle._C_ops.softmax(scale_14, -1)
        del scale_14

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_87 = paddle._C_ops.matmul(softmax_14, slice_60, False, False)
        del slice_60, softmax_14

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_45 = paddle._C_ops.transpose(matmul_87, [0, 2, 1, 3])
        del matmul_87

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_29 = paddle._C_ops.reshape(transpose_45, full_int_array_5)
        del transpose_45

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_88 = paddle._C_ops.matmul(reshape_29, parameter_119, False, False)
        del parameter_119, reshape_29

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_86 = paddle._C_ops.add(matmul_88, parameter_118)
        del matmul_88, parameter_118

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_87 = paddle._C_ops.add(add_84, add_86)
        del add_84, add_86

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_90, layer_norm_91, layer_norm_92 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_87, parameter_117, parameter_116, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_116, parameter_117

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_89 = paddle._C_ops.matmul(layer_norm_90, parameter_115, False, False)
        del layer_norm_90, parameter_115

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_88 = paddle._C_ops.add(matmul_89, parameter_114)
        del matmul_89, parameter_114

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_14 = paddle._C_ops.gelu(add_88, False)
        del add_88

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_90 = paddle._C_ops.matmul(gelu_14, parameter_113, False, False)
        del gelu_14, parameter_113

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_89 = paddle._C_ops.add(matmul_90, parameter_112)
        del matmul_90, parameter_112

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_90 = paddle._C_ops.add(add_87, add_89)
        del add_87, add_89

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_93, layer_norm_94, layer_norm_95 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_90, parameter_111, parameter_110, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_110, parameter_111

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_16 = paddle._C_ops.shape64(layer_norm_93)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_61 = paddle._C_ops.slice(
            shape64_16, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_16

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_91 = paddle._C_ops.matmul(layer_norm_93, parameter_109, False, False)
        del layer_norm_93, parameter_109

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_91 = paddle._C_ops.add(matmul_91, parameter_108)
        del matmul_91, parameter_108

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_30 = paddle._C_ops.reshape(add_91, full_int_array_2)
        del add_91

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_46 = paddle._C_ops.transpose(reshape_30, [2, 0, 3, 1, 4])
        del reshape_30

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_62 = paddle._C_ops.slice(
            transpose_46, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_63 = paddle._C_ops.slice(
            transpose_46, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_64 = paddle._C_ops.slice(
            transpose_46, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del transpose_46

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_47 = paddle._C_ops.transpose(slice_63, [0, 1, 3, 2])
        del slice_63

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_92 = paddle._C_ops.matmul(slice_62, transpose_47, False, False)
        del slice_62, transpose_47

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_15 = paddle._C_ops.scale(matmul_92, full_2, float("0"), True)
        del matmul_92

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_15 = paddle._C_ops.softmax(scale_15, -1)
        del scale_15

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_93 = paddle._C_ops.matmul(softmax_15, slice_64, False, False)
        del slice_64, softmax_15

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_48 = paddle._C_ops.transpose(matmul_93, [0, 2, 1, 3])
        del matmul_93

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_31 = paddle._C_ops.reshape(transpose_48, full_int_array_5)
        del transpose_48

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_94 = paddle._C_ops.matmul(reshape_31, parameter_107, False, False)
        del parameter_107, reshape_31

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_92 = paddle._C_ops.add(matmul_94, parameter_106)
        del matmul_94, parameter_106

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_93 = paddle._C_ops.add(add_90, add_92)
        del add_90, add_92

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_96, layer_norm_97, layer_norm_98 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_93, parameter_105, parameter_104, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_104, parameter_105

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_95 = paddle._C_ops.matmul(layer_norm_96, parameter_103, False, False)
        del layer_norm_96, parameter_103

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_94 = paddle._C_ops.add(matmul_95, parameter_102)
        del matmul_95, parameter_102

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_15 = paddle._C_ops.gelu(add_94, False)
        del add_94

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_96 = paddle._C_ops.matmul(gelu_15, parameter_101, False, False)
        del gelu_15, parameter_101

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_95 = paddle._C_ops.add(matmul_96, parameter_100)
        del matmul_96, parameter_100

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_96 = paddle._C_ops.add(add_93, add_95)
        del add_93, add_95

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_99, layer_norm_100, layer_norm_101 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_96, parameter_99, parameter_98, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_98, parameter_99

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_17 = paddle._C_ops.shape64(layer_norm_99)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_65 = paddle._C_ops.slice(
            shape64_17, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_17

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_97 = paddle._C_ops.matmul(layer_norm_99, parameter_97, False, False)
        del layer_norm_99, parameter_97

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_97 = paddle._C_ops.add(matmul_97, parameter_96)
        del matmul_97, parameter_96

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_32 = paddle._C_ops.reshape(add_97, full_int_array_2)
        del add_97

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_49 = paddle._C_ops.transpose(reshape_32, [2, 0, 3, 1, 4])
        del reshape_32

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_66 = paddle._C_ops.slice(
            transpose_49, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_67 = paddle._C_ops.slice(
            transpose_49, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_68 = paddle._C_ops.slice(
            transpose_49, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del transpose_49

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_50 = paddle._C_ops.transpose(slice_67, [0, 1, 3, 2])
        del slice_67

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_98 = paddle._C_ops.matmul(slice_66, transpose_50, False, False)
        del slice_66, transpose_50

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_16 = paddle._C_ops.scale(matmul_98, full_2, float("0"), True)
        del matmul_98

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_16 = paddle._C_ops.softmax(scale_16, -1)
        del scale_16

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_99 = paddle._C_ops.matmul(softmax_16, slice_68, False, False)
        del slice_68, softmax_16

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_51 = paddle._C_ops.transpose(matmul_99, [0, 2, 1, 3])
        del matmul_99

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_33 = paddle._C_ops.reshape(transpose_51, full_int_array_5)
        del transpose_51

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_100 = paddle._C_ops.matmul(reshape_33, parameter_95, False, False)
        del parameter_95, reshape_33

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_98 = paddle._C_ops.add(matmul_100, parameter_94)
        del matmul_100, parameter_94

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_99 = paddle._C_ops.add(add_96, add_98)
        del add_96, add_98

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_102, layer_norm_103, layer_norm_104 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_99, parameter_93, parameter_92, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_92, parameter_93

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_101 = paddle._C_ops.matmul(layer_norm_102, parameter_91, False, False)
        del layer_norm_102, parameter_91

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_100 = paddle._C_ops.add(matmul_101, parameter_90)
        del matmul_101, parameter_90

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_16 = paddle._C_ops.gelu(add_100, False)
        del add_100

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_102 = paddle._C_ops.matmul(gelu_16, parameter_89, False, False)
        del gelu_16, parameter_89

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_101 = paddle._C_ops.add(matmul_102, parameter_88)
        del matmul_102, parameter_88

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_102 = paddle._C_ops.add(add_99, add_101)
        del add_101, add_99

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_105, layer_norm_106, layer_norm_107 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_102, parameter_87, parameter_86, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_86, parameter_87

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_18 = paddle._C_ops.shape64(layer_norm_105)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_69 = paddle._C_ops.slice(
            shape64_18, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_18

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_103 = paddle._C_ops.matmul(layer_norm_105, parameter_85, False, False)
        del layer_norm_105, parameter_85

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_103 = paddle._C_ops.add(matmul_103, parameter_84)
        del matmul_103, parameter_84

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_34 = paddle._C_ops.reshape(add_103, full_int_array_2)
        del add_103

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_52 = paddle._C_ops.transpose(reshape_34, [2, 0, 3, 1, 4])
        del reshape_34

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_70 = paddle._C_ops.slice(
            transpose_52, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_71 = paddle._C_ops.slice(
            transpose_52, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_72 = paddle._C_ops.slice(
            transpose_52, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del transpose_52

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_53 = paddle._C_ops.transpose(slice_71, [0, 1, 3, 2])
        del slice_71

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_104 = paddle._C_ops.matmul(slice_70, transpose_53, False, False)
        del slice_70, transpose_53

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_17 = paddle._C_ops.scale(matmul_104, full_2, float("0"), True)
        del matmul_104

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_17 = paddle._C_ops.softmax(scale_17, -1)
        del scale_17

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_105 = paddle._C_ops.matmul(softmax_17, slice_72, False, False)
        del slice_72, softmax_17

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_54 = paddle._C_ops.transpose(matmul_105, [0, 2, 1, 3])
        del matmul_105

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_35 = paddle._C_ops.reshape(transpose_54, full_int_array_5)
        del transpose_54

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_106 = paddle._C_ops.matmul(reshape_35, parameter_83, False, False)
        del parameter_83, reshape_35

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_104 = paddle._C_ops.add(matmul_106, parameter_82)
        del matmul_106, parameter_82

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_105 = paddle._C_ops.add(add_102, add_104)
        del add_102, add_104

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_108, layer_norm_109, layer_norm_110 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_105, parameter_81, parameter_80, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_80, parameter_81

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_107 = paddle._C_ops.matmul(layer_norm_108, parameter_79, False, False)
        del layer_norm_108, parameter_79

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_106 = paddle._C_ops.add(matmul_107, parameter_78)
        del matmul_107, parameter_78

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_17 = paddle._C_ops.gelu(add_106, False)
        del add_106

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_108 = paddle._C_ops.matmul(gelu_17, parameter_77, False, False)
        del gelu_17, parameter_77

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_107 = paddle._C_ops.add(matmul_108, parameter_76)
        del matmul_108, parameter_76

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_108 = paddle._C_ops.add(add_105, add_107)
        del add_105, add_107

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_111, layer_norm_112, layer_norm_113 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_108, parameter_75, parameter_74, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_74, parameter_75

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_19 = paddle._C_ops.shape64(layer_norm_111)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_73 = paddle._C_ops.slice(
            shape64_19, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_19

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_109 = paddle._C_ops.matmul(layer_norm_111, parameter_73, False, False)
        del layer_norm_111, parameter_73

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_109 = paddle._C_ops.add(matmul_109, parameter_72)
        del matmul_109, parameter_72

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_36 = paddle._C_ops.reshape(add_109, full_int_array_2)
        del add_109

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_55 = paddle._C_ops.transpose(reshape_36, [2, 0, 3, 1, 4])
        del reshape_36

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_74 = paddle._C_ops.slice(
            transpose_55, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_75 = paddle._C_ops.slice(
            transpose_55, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_76 = paddle._C_ops.slice(
            transpose_55, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del transpose_55

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_56 = paddle._C_ops.transpose(slice_75, [0, 1, 3, 2])
        del slice_75

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_110 = paddle._C_ops.matmul(slice_74, transpose_56, False, False)
        del slice_74, transpose_56

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_18 = paddle._C_ops.scale(matmul_110, full_2, float("0"), True)
        del matmul_110

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_18 = paddle._C_ops.softmax(scale_18, -1)
        del scale_18

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_111 = paddle._C_ops.matmul(softmax_18, slice_76, False, False)
        del slice_76, softmax_18

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_57 = paddle._C_ops.transpose(matmul_111, [0, 2, 1, 3])
        del matmul_111

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_37 = paddle._C_ops.reshape(transpose_57, full_int_array_5)
        del transpose_57

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_112 = paddle._C_ops.matmul(reshape_37, parameter_71, False, False)
        del parameter_71, reshape_37

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_110 = paddle._C_ops.add(matmul_112, parameter_70)
        del matmul_112, parameter_70

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_111 = paddle._C_ops.add(add_108, add_110)
        del add_108, add_110

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_114, layer_norm_115, layer_norm_116 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_111, parameter_69, parameter_68, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_68, parameter_69

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_113 = paddle._C_ops.matmul(layer_norm_114, parameter_67, False, False)
        del layer_norm_114, parameter_67

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_112 = paddle._C_ops.add(matmul_113, parameter_66)
        del matmul_113, parameter_66

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_18 = paddle._C_ops.gelu(add_112, False)
        del add_112

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_114 = paddle._C_ops.matmul(gelu_18, parameter_65, False, False)
        del gelu_18, parameter_65

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_113 = paddle._C_ops.add(matmul_114, parameter_64)
        del matmul_114, parameter_64

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_114 = paddle._C_ops.add(add_111, add_113)
        del add_111, add_113

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_117, layer_norm_118, layer_norm_119 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_114, parameter_63, parameter_62, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_62, parameter_63

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_20 = paddle._C_ops.shape64(layer_norm_117)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_77 = paddle._C_ops.slice(
            shape64_20, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_20

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_115 = paddle._C_ops.matmul(layer_norm_117, parameter_61, False, False)
        del layer_norm_117, parameter_61

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_115 = paddle._C_ops.add(matmul_115, parameter_60)
        del matmul_115, parameter_60

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_38 = paddle._C_ops.reshape(add_115, full_int_array_2)
        del add_115

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_58 = paddle._C_ops.transpose(reshape_38, [2, 0, 3, 1, 4])
        del reshape_38

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_78 = paddle._C_ops.slice(
            transpose_58, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_79 = paddle._C_ops.slice(
            transpose_58, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_80 = paddle._C_ops.slice(
            transpose_58, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del transpose_58

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_59 = paddle._C_ops.transpose(slice_79, [0, 1, 3, 2])
        del slice_79

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_116 = paddle._C_ops.matmul(slice_78, transpose_59, False, False)
        del slice_78, transpose_59

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_19 = paddle._C_ops.scale(matmul_116, full_2, float("0"), True)
        del matmul_116

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_19 = paddle._C_ops.softmax(scale_19, -1)
        del scale_19

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_117 = paddle._C_ops.matmul(softmax_19, slice_80, False, False)
        del slice_80, softmax_19

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_60 = paddle._C_ops.transpose(matmul_117, [0, 2, 1, 3])
        del matmul_117

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_39 = paddle._C_ops.reshape(transpose_60, full_int_array_5)
        del transpose_60

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_118 = paddle._C_ops.matmul(reshape_39, parameter_59, False, False)
        del parameter_59, reshape_39

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_116 = paddle._C_ops.add(matmul_118, parameter_58)
        del matmul_118, parameter_58

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_117 = paddle._C_ops.add(add_114, add_116)
        del add_114, add_116

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_120, layer_norm_121, layer_norm_122 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_117, parameter_57, parameter_56, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_56, parameter_57

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_119 = paddle._C_ops.matmul(layer_norm_120, parameter_55, False, False)
        del layer_norm_120, parameter_55

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_118 = paddle._C_ops.add(matmul_119, parameter_54)
        del matmul_119, parameter_54

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_19 = paddle._C_ops.gelu(add_118, False)
        del add_118

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_120 = paddle._C_ops.matmul(gelu_19, parameter_53, False, False)
        del gelu_19, parameter_53

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_119 = paddle._C_ops.add(matmul_120, parameter_52)
        del matmul_120, parameter_52

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_120 = paddle._C_ops.add(add_117, add_119)
        del add_117, add_119

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_123, layer_norm_124, layer_norm_125 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_120, parameter_51, parameter_50, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_50, parameter_51

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_21 = paddle._C_ops.shape64(layer_norm_123)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_81 = paddle._C_ops.slice(
            shape64_21, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_21

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_121 = paddle._C_ops.matmul(layer_norm_123, parameter_49, False, False)
        del layer_norm_123, parameter_49

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_121 = paddle._C_ops.add(matmul_121, parameter_48)
        del matmul_121, parameter_48

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_40 = paddle._C_ops.reshape(add_121, full_int_array_2)
        del add_121

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_61 = paddle._C_ops.transpose(reshape_40, [2, 0, 3, 1, 4])
        del reshape_40

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_82 = paddle._C_ops.slice(
            transpose_61, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_83 = paddle._C_ops.slice(
            transpose_61, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_84 = paddle._C_ops.slice(
            transpose_61, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del transpose_61

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_62 = paddle._C_ops.transpose(slice_83, [0, 1, 3, 2])
        del slice_83

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_122 = paddle._C_ops.matmul(slice_82, transpose_62, False, False)
        del slice_82, transpose_62

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_20 = paddle._C_ops.scale(matmul_122, full_2, float("0"), True)
        del matmul_122

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_20 = paddle._C_ops.softmax(scale_20, -1)
        del scale_20

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_123 = paddle._C_ops.matmul(softmax_20, slice_84, False, False)
        del slice_84, softmax_20

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_63 = paddle._C_ops.transpose(matmul_123, [0, 2, 1, 3])
        del matmul_123

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_41 = paddle._C_ops.reshape(transpose_63, full_int_array_5)
        del transpose_63

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_124 = paddle._C_ops.matmul(reshape_41, parameter_47, False, False)
        del parameter_47, reshape_41

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_122 = paddle._C_ops.add(matmul_124, parameter_46)
        del matmul_124, parameter_46

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_123 = paddle._C_ops.add(add_120, add_122)
        del add_120, add_122

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_126, layer_norm_127, layer_norm_128 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_123, parameter_45, parameter_44, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_44, parameter_45

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_125 = paddle._C_ops.matmul(layer_norm_126, parameter_43, False, False)
        del layer_norm_126, parameter_43

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_124 = paddle._C_ops.add(matmul_125, parameter_42)
        del matmul_125, parameter_42

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_20 = paddle._C_ops.gelu(add_124, False)
        del add_124

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_126 = paddle._C_ops.matmul(gelu_20, parameter_41, False, False)
        del gelu_20, parameter_41

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_125 = paddle._C_ops.add(matmul_126, parameter_40)
        del matmul_126, parameter_40

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_126 = paddle._C_ops.add(add_123, add_125)
        del add_123, add_125

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_129, layer_norm_130, layer_norm_131 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_126, parameter_39, parameter_38, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_38, parameter_39

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_22 = paddle._C_ops.shape64(layer_norm_129)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_85 = paddle._C_ops.slice(
            shape64_22, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_22

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_127 = paddle._C_ops.matmul(layer_norm_129, parameter_37, False, False)
        del layer_norm_129, parameter_37

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_127 = paddle._C_ops.add(matmul_127, parameter_36)
        del matmul_127, parameter_36

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_42 = paddle._C_ops.reshape(add_127, full_int_array_2)
        del add_127

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_64 = paddle._C_ops.transpose(reshape_42, [2, 0, 3, 1, 4])
        del reshape_42

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_86 = paddle._C_ops.slice(
            transpose_64, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_87 = paddle._C_ops.slice(
            transpose_64, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_88 = paddle._C_ops.slice(
            transpose_64, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del transpose_64

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_65 = paddle._C_ops.transpose(slice_87, [0, 1, 3, 2])
        del slice_87

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_128 = paddle._C_ops.matmul(slice_86, transpose_65, False, False)
        del slice_86, transpose_65

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_21 = paddle._C_ops.scale(matmul_128, full_2, float("0"), True)
        del matmul_128

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_21 = paddle._C_ops.softmax(scale_21, -1)
        del scale_21

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_129 = paddle._C_ops.matmul(softmax_21, slice_88, False, False)
        del slice_88, softmax_21

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_66 = paddle._C_ops.transpose(matmul_129, [0, 2, 1, 3])
        del matmul_129

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_43 = paddle._C_ops.reshape(transpose_66, full_int_array_5)
        del transpose_66

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_130 = paddle._C_ops.matmul(reshape_43, parameter_35, False, False)
        del parameter_35, reshape_43

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_128 = paddle._C_ops.add(matmul_130, parameter_34)
        del matmul_130, parameter_34

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_129 = paddle._C_ops.add(add_126, add_128)
        del add_126, add_128

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_132, layer_norm_133, layer_norm_134 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_129, parameter_33, parameter_32, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_32, parameter_33

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_131 = paddle._C_ops.matmul(layer_norm_132, parameter_31, False, False)
        del layer_norm_132, parameter_31

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_130 = paddle._C_ops.add(matmul_131, parameter_30)
        del matmul_131, parameter_30

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_21 = paddle._C_ops.gelu(add_130, False)
        del add_130

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_132 = paddle._C_ops.matmul(gelu_21, parameter_29, False, False)
        del gelu_21, parameter_29

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_131 = paddle._C_ops.add(matmul_132, parameter_28)
        del matmul_132, parameter_28

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_132 = paddle._C_ops.add(add_129, add_131)
        del add_129, add_131

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_135, layer_norm_136, layer_norm_137 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_132, parameter_27, parameter_26, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_26, parameter_27

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_23 = paddle._C_ops.shape64(layer_norm_135)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_89 = paddle._C_ops.slice(
            shape64_23, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_23

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_133 = paddle._C_ops.matmul(layer_norm_135, parameter_25, False, False)
        del layer_norm_135, parameter_25

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_133 = paddle._C_ops.add(matmul_133, parameter_24)
        del matmul_133, parameter_24

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_44 = paddle._C_ops.reshape(add_133, full_int_array_2)
        del add_133

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_67 = paddle._C_ops.transpose(reshape_44, [2, 0, 3, 1, 4])
        del reshape_44

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_90 = paddle._C_ops.slice(
            transpose_67, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_91 = paddle._C_ops.slice(
            transpose_67, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_92 = paddle._C_ops.slice(
            transpose_67, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del transpose_67

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_68 = paddle._C_ops.transpose(slice_91, [0, 1, 3, 2])
        del slice_91

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_134 = paddle._C_ops.matmul(slice_90, transpose_68, False, False)
        del slice_90, transpose_68

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_22 = paddle._C_ops.scale(matmul_134, full_2, float("0"), True)
        del matmul_134

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_22 = paddle._C_ops.softmax(scale_22, -1)
        del scale_22

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_135 = paddle._C_ops.matmul(softmax_22, slice_92, False, False)
        del slice_92, softmax_22

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_69 = paddle._C_ops.transpose(matmul_135, [0, 2, 1, 3])
        del matmul_135

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_45 = paddle._C_ops.reshape(transpose_69, full_int_array_5)
        del transpose_69

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_136 = paddle._C_ops.matmul(reshape_45, parameter_23, False, False)
        del parameter_23, reshape_45

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_134 = paddle._C_ops.add(matmul_136, parameter_22)
        del matmul_136, parameter_22

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_135 = paddle._C_ops.add(add_132, add_134)
        del add_132, add_134

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_138, layer_norm_139, layer_norm_140 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_135, parameter_21, parameter_20, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_20, parameter_21

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_137 = paddle._C_ops.matmul(layer_norm_138, parameter_19, False, False)
        del layer_norm_138, parameter_19

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_136 = paddle._C_ops.add(matmul_137, parameter_18)
        del matmul_137, parameter_18

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_22 = paddle._C_ops.gelu(add_136, False)
        del add_136

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_138 = paddle._C_ops.matmul(gelu_22, parameter_17, False, False)
        del gelu_22, parameter_17

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_137 = paddle._C_ops.add(matmul_138, parameter_16)
        del matmul_138, parameter_16

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_138 = paddle._C_ops.add(add_135, add_137)
        del add_135, add_137

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_141, layer_norm_142, layer_norm_143 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_138, parameter_15, parameter_14, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_14, parameter_15

        # pd_op.shape64: (3xi64) <- (-1x257x1024xf32)
        shape64_24 = paddle._C_ops.shape64(layer_norm_141)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_93 = paddle._C_ops.slice(
            shape64_24, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_24

        # pd_op.matmul: (-1x257x3072xf32) <- (-1x257x1024xf32, 1024x3072xf32)
        matmul_139 = paddle._C_ops.matmul(layer_norm_141, parameter_13, False, False)
        del layer_norm_141, parameter_13

        # pd_op.add: (-1x257x3072xf32) <- (-1x257x3072xf32, 3072xf32)
        add_139 = paddle._C_ops.add(matmul_139, parameter_12)
        del matmul_139, parameter_12

        # pd_op.reshape: (-1x257x3x16x64xf32) <- (-1x257x3072xf32, 5xi64)
        reshape_46 = paddle._C_ops.reshape(add_139, full_int_array_2)
        del add_139, full_int_array_2

        # pd_op.transpose: (3x-1x16x257x64xf32) <- (-1x257x3x16x64xf32)
        transpose_70 = paddle._C_ops.transpose(reshape_46, [2, 0, 3, 1, 4])
        del reshape_46

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_94 = paddle._C_ops.slice(
            transpose_70, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_95 = paddle._C_ops.slice(
            transpose_70, [0], full_int_array_1, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (-1x16x257x64xf32) <- (3x-1x16x257x64xf32, 1xi64, 1xi64)
        slice_96 = paddle._C_ops.slice(
            transpose_70, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del full_int_array_3, full_int_array_4, transpose_70

        # pd_op.transpose: (-1x16x64x257xf32) <- (-1x16x257x64xf32)
        transpose_71 = paddle._C_ops.transpose(slice_95, [0, 1, 3, 2])
        del slice_95

        # pd_op.matmul: (-1x16x257x257xf32) <- (-1x16x257x64xf32, -1x16x64x257xf32)
        matmul_140 = paddle._C_ops.matmul(slice_94, transpose_71, False, False)
        del slice_94, transpose_71

        # pd_op.scale: (-1x16x257x257xf32) <- (-1x16x257x257xf32, 1xf32)
        scale_23 = paddle._C_ops.scale(matmul_140, full_2, float("0"), True)
        del full_2, matmul_140

        # pd_op.softmax: (-1x16x257x257xf32) <- (-1x16x257x257xf32)
        softmax_23 = paddle._C_ops.softmax(scale_23, -1)
        del scale_23

        # pd_op.matmul: (-1x16x257x64xf32) <- (-1x16x257x257xf32, -1x16x257x64xf32)
        matmul_141 = paddle._C_ops.matmul(softmax_23, slice_96, False, False)
        del slice_96, softmax_23

        # pd_op.transpose: (-1x257x16x64xf32) <- (-1x16x257x64xf32)
        transpose_72 = paddle._C_ops.transpose(matmul_141, [0, 2, 1, 3])
        del matmul_141

        # pd_op.reshape: (-1x257x1024xf32) <- (-1x257x16x64xf32, 3xi64)
        reshape_47 = paddle._C_ops.reshape(transpose_72, full_int_array_5)
        del full_int_array_5, transpose_72

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024x1024xf32)
        matmul_142 = paddle._C_ops.matmul(reshape_47, parameter_11, False, False)
        del parameter_11, reshape_47

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_140 = paddle._C_ops.add(matmul_142, parameter_10)
        del matmul_142, parameter_10

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_141 = paddle._C_ops.add(add_138, add_140)
        del add_138, add_140

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_144, layer_norm_145, layer_norm_146 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_141, parameter_9, parameter_8, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_8, parameter_9

        # pd_op.matmul: (-1x257x4096xf32) <- (-1x257x1024xf32, 1024x4096xf32)
        matmul_143 = paddle._C_ops.matmul(layer_norm_144, parameter_7, False, False)
        del layer_norm_144, parameter_7

        # pd_op.add: (-1x257x4096xf32) <- (-1x257x4096xf32, 4096xf32)
        add_142 = paddle._C_ops.add(matmul_143, parameter_6)
        del matmul_143, parameter_6

        # pd_op.gelu: (-1x257x4096xf32) <- (-1x257x4096xf32)
        gelu_23 = paddle._C_ops.gelu(add_142, False)
        del add_142

        # pd_op.matmul: (-1x257x1024xf32) <- (-1x257x4096xf32, 4096x1024xf32)
        matmul_144 = paddle._C_ops.matmul(gelu_23, parameter_5, False, False)
        del gelu_23, parameter_5

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, 1024xf32)
        add_143 = paddle._C_ops.add(matmul_144, parameter_4)
        del matmul_144, parameter_4

        # pd_op.add: (-1x257x1024xf32) <- (-1x257x1024xf32, -1x257x1024xf32)
        add_144 = paddle._C_ops.add(add_141, add_143)
        del add_141, add_143

        # pd_op.layer_norm: (-1x257x1024xf32, -1x257xf32, -1x257xf32) <- (-1x257x1024xf32, 1024xf32, 1024xf32)
        layer_norm_147, layer_norm_148, layer_norm_149 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_144, parameter_3, parameter_2, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_144, parameter_2, parameter_3

        # pd_op.matmul: (-1x257x512xf32) <- (-1x257x1024xf32, 1024x512xf32)
        matmul_145 = paddle._C_ops.matmul(layer_norm_147, data_0, False, False)
        del data_0, layer_norm_147

        # pd_op.mean: (-1x512xf32) <- (-1x257x512xf32, 1xi64)
        mean_0 = paddle._C_ops.mean(matmul_145, full_int_array_1, False)
        del matmul_145

        # pd_op.matmul: (-1x512xf32) <- (-1x512xf32, 512x512xf32)
        matmul_146 = paddle._C_ops.matmul(mean_0, parameter_1, False, False)
        del parameter_1

        # pd_op.add: (-1x512xf32) <- (-1x512xf32, 512xf32)
        add_145 = paddle._C_ops.add(matmul_146, parameter_0)
        del matmul_146, parameter_0

        # pd_op.square: (-1x512xf32) <- (-1x512xf32)
        square_0 = paddle._C_ops.square(add_145)

        # pd_op.sum: (-1x1xf32) <- (-1x512xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(square_0, full_int_array_1, None, True)
        del full_int_array_1, square_0

        # pd_op.sqrt: (-1x1xf32) <- (-1x1xf32)
        sqrt_0 = paddle._C_ops.sqrt(sum_0)
        del sum_0

        # pd_op.divide: (-1x512xf32) <- (-1x512xf32, -1x1xf32)
        divide_0 = paddle._C_ops.divide(add_145, sqrt_0)
        del sqrt_0

        # pd_op.square: (512x159xf32) <- (512x159xf32)
        square_1 = paddle._C_ops.square(data_1)

        # pd_op.sum: (1x159xf32) <- (512x159xf32, 1xi64)
        sum_1 = paddle._C_ops.sum(square_1, full_int_array_0, None, True)
        del full_int_array_0, square_1

        # pd_op.sqrt: (1x159xf32) <- (1x159xf32)
        sqrt_1 = paddle._C_ops.sqrt(sum_1)
        del sum_1

        # pd_op.divide: (512x159xf32) <- (512x159xf32, 1x159xf32)
        divide_1 = paddle._C_ops.divide(data_1, sqrt_1)
        del data_1, sqrt_1

        # pd_op.matmul: (-1x159xf32) <- (-1x512xf32, 512x159xf32)
        matmul_0 = paddle._C_ops.matmul(divide_0, divide_1, False, False)
        del add_145, divide_0, divide_1, mean_0

        return matmul_0
