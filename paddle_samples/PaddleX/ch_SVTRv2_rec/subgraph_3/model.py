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
        data_0,
    ):
        # pd_op.conv2d: (4x64x32x160xf32) <- (4x3x64x320xf32, 64x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_219, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_0, parameter_219

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_218, full_int_array_0)
        del parameter_218

        # pd_op.add: (4x64x32x160xf32) <- (4x64x32x160xf32, 1x64x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_0, reshape_0)

        # pd_op.batch_norm_: (4x64x32x160xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x32x160xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
            batch_norm__5,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_0,
                parameter_217,
                parameter_216,
                parameter_215,
                parameter_214,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_214, parameter_215, parameter_216, parameter_217

        # pd_op.gelu: (4x64x32x160xf32) <- (4x64x32x160xf32)
        gelu_0 = paddle._C_ops.gelu(batch_norm__0, False)

        # pd_op.conv2d: (4x128x16x80xf32) <- (4x64x32x160xf32, 128x64x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            gelu_0, parameter_213, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_213

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_212, full_int_array_0)
        del parameter_212

        # pd_op.add: (4x128x16x80xf32) <- (4x128x16x80xf32, 1x128x1x1xf32)
        add_1 = paddle._C_ops.add(conv2d_1, reshape_1)

        # pd_op.batch_norm_: (4x128x16x80xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (4x128x16x80xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            batch_norm__10,
            batch_norm__11,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_1,
                parameter_211,
                parameter_210,
                parameter_209,
                parameter_208,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_208, parameter_209, parameter_210, parameter_211

        # pd_op.gelu: (4x128x16x80xf32) <- (4x128x16x80xf32)
        gelu_1 = paddle._C_ops.gelu(batch_norm__6, False)

        # pd_op.conv2d: (4x128x16x80xf32) <- (4x128x16x80xf32, 128x32x5x5xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            gelu_1, parameter_207, [1, 1], [2, 2], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del parameter_207

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(parameter_206, full_int_array_0)
        del parameter_206

        # pd_op.add: (4x128x16x80xf32) <- (4x128x16x80xf32, 1x128x1x1xf32)
        add_2 = paddle._C_ops.add(conv2d_2, reshape_2)

        # pd_op.add: (4x128x16x80xf32) <- (4x128x16x80xf32, 4x128x16x80xf32)
        add_3 = paddle._C_ops.add(gelu_1, add_2)

        # pd_op.flatten: (4x128x1280xf32) <- (4x128x16x80xf32)
        flatten_0 = paddle._C_ops.flatten(add_3, 2, 3)

        # pd_op.transpose: (4x1280x128xf32) <- (4x128x1280xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_0, [0, 2, 1])
        del flatten_0

        # pd_op.layer_norm: (4x1280x128xf32, 4x1280xf32, 4x1280xf32) <- (4x1280x128xf32, 128xf32, 128xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_0, parameter_205, parameter_204, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_204, parameter_205

        # pd_op.matmul: (4x1280x512xf32) <- (4x1280x128xf32, 128x512xf32)
        matmul_0 = paddle._C_ops.matmul(layer_norm_0, parameter_203, False, False)
        del parameter_203

        # pd_op.add: (4x1280x512xf32) <- (4x1280x512xf32, 512xf32)
        add_4 = paddle._C_ops.add(matmul_0, parameter_202)
        del parameter_202

        # pd_op.gelu: (4x1280x512xf32) <- (4x1280x512xf32)
        gelu_2 = paddle._C_ops.gelu(add_4, False)

        # pd_op.matmul: (4x1280x128xf32) <- (4x1280x512xf32, 512x128xf32)
        matmul_1 = paddle._C_ops.matmul(gelu_2, parameter_201, False, False)
        del parameter_201

        # pd_op.add: (4x1280x128xf32) <- (4x1280x128xf32, 128xf32)
        add_5 = paddle._C_ops.add(matmul_1, parameter_200)
        del parameter_200

        # pd_op.add: (4x1280x128xf32) <- (4x1280x128xf32, 4x1280x128xf32)
        add_6 = paddle._C_ops.add(layer_norm_0, add_5)

        # pd_op.layer_norm: (4x1280x128xf32, 4x1280xf32, 4x1280xf32) <- (4x1280x128xf32, 128xf32, 128xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_6, parameter_199, parameter_198, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_198, parameter_199

        # pd_op.transpose: (4x128x1280xf32) <- (4x1280x128xf32)
        transpose_1 = paddle._C_ops.transpose(layer_norm_3, [0, 2, 1])
        del layer_norm_3

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [0, 128, 16, 80]

        # pd_op.reshape: (4x128x16x80xf32) <- (4x128x1280xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_1, full_int_array_1)

        # pd_op.conv2d: (4x128x16x80xf32) <- (4x128x16x80xf32, 128x32x5x5xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            reshape_3, parameter_197, [1, 1], [2, 2], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del parameter_197

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(parameter_196, full_int_array_0)
        del parameter_196

        # pd_op.add: (4x128x16x80xf32) <- (4x128x16x80xf32, 1x128x1x1xf32)
        add_7 = paddle._C_ops.add(conv2d_3, reshape_4)

        # pd_op.full: (xf64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_0,
            [],
            paddle.float64,
            [float("0.994118")],
            paddle.framework._current_expected_place(),
        )
        del full_0

        # pd_op.cast: (xf32) <- (xf64)
        cast_0 = paddle._C_ops.cast(assign_value__0, paddle.float32)
        del assign_value__0

        # pd_op.shape64: (4xi64) <- (4x128x16x80xf32)
        shape64_0 = paddle._C_ops.shape64(add_7)

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

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_10 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_11 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_12 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_13 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_14 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_15 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_16 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_17 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_18 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_19 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_20 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_21 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_22 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_23 = full_int_array_3

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

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_0

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_0 = [slice_0, full_1, full_1, full_1]
        del slice_0

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_0 = paddle._C_ops.uniform(
            stack_0,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_0

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_8 = paddle._C_ops.add(cast_0, uniform_0)
        del uniform_0

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_0 = paddle._C_ops.floor(add_8)
        del add_8

        # pd_op.divide: (4x128x16x80xf32) <- (4x128x16x80xf32, xf32)
        divide_0 = paddle._C_ops.divide(add_7, cast_0)

        # pd_op.multiply: (4x128x16x80xf32) <- (4x128x16x80xf32, -1x1x1x1xf32)
        multiply_0 = paddle._C_ops.multiply(divide_0, floor_0)

        # pd_op.add: (4x128x16x80xf32) <- (4x128x16x80xf32, 4x128x16x80xf32)
        add_9 = paddle._C_ops.add(reshape_3, multiply_0)

        # pd_op.flatten: (4x128x1280xf32) <- (4x128x16x80xf32)
        flatten_1 = paddle._C_ops.flatten(add_9, 2, 3)

        # pd_op.transpose: (4x1280x128xf32) <- (4x128x1280xf32)
        transpose_2 = paddle._C_ops.transpose(flatten_1, [0, 2, 1])
        del flatten_1

        # pd_op.layer_norm: (4x1280x128xf32, 4x1280xf32, 4x1280xf32) <- (4x1280x128xf32, 128xf32, 128xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_2, parameter_195, parameter_194, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_194, parameter_195

        # pd_op.matmul: (4x1280x512xf32) <- (4x1280x128xf32, 128x512xf32)
        matmul_2 = paddle._C_ops.matmul(layer_norm_6, parameter_193, False, False)
        del parameter_193

        # pd_op.add: (4x1280x512xf32) <- (4x1280x512xf32, 512xf32)
        add_10 = paddle._C_ops.add(matmul_2, parameter_192)
        del parameter_192

        # pd_op.gelu: (4x1280x512xf32) <- (4x1280x512xf32)
        gelu_3 = paddle._C_ops.gelu(add_10, False)

        # pd_op.matmul: (4x1280x128xf32) <- (4x1280x512xf32, 512x128xf32)
        matmul_3 = paddle._C_ops.matmul(gelu_3, parameter_191, False, False)
        del parameter_191

        # pd_op.add: (4x1280x128xf32) <- (4x1280x128xf32, 128xf32)
        add_11 = paddle._C_ops.add(matmul_3, parameter_190)
        del parameter_190

        # pd_op.full: (xf64) <- ()
        full_4 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__1 = paddle._C_ops.assign_value_(
            full_4,
            [],
            paddle.float64,
            [float("0.994118")],
            paddle.framework._current_expected_place(),
        )
        del full_4

        # pd_op.cast: (xf32) <- (xf64)
        cast_1 = paddle._C_ops.cast(assign_value__1, paddle.float32)
        del assign_value__1

        # pd_op.shape64: (3xi64) <- (4x1280x128xf32)
        shape64_1 = paddle._C_ops.shape64(add_11)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_1

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_1 = [slice_1, full_1, full_1]
        del slice_1

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_1, 0)
        del combine_1

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_1 = paddle._C_ops.uniform(
            stack_1,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_1

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_12 = paddle._C_ops.add(cast_1, uniform_1)
        del uniform_1

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_1 = paddle._C_ops.floor(add_12)
        del add_12

        # pd_op.divide: (4x1280x128xf32) <- (4x1280x128xf32, xf32)
        divide_1 = paddle._C_ops.divide(add_11, cast_1)

        # pd_op.multiply: (4x1280x128xf32) <- (4x1280x128xf32, -1x1x1xf32)
        multiply_1 = paddle._C_ops.multiply(divide_1, floor_1)

        # pd_op.add: (4x1280x128xf32) <- (4x1280x128xf32, 4x1280x128xf32)
        add_13 = paddle._C_ops.add(layer_norm_6, multiply_1)

        # pd_op.layer_norm: (4x1280x128xf32, 4x1280xf32, 4x1280xf32) <- (4x1280x128xf32, 128xf32, 128xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_13, parameter_189, parameter_188, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_188, parameter_189

        # pd_op.transpose: (4x128x1280xf32) <- (4x1280x128xf32)
        transpose_3 = paddle._C_ops.transpose(layer_norm_9, [0, 2, 1])
        del layer_norm_9

        # pd_op.reshape: (4x128x16x80xf32) <- (4x128x1280xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(transpose_3, full_int_array_1)

        # pd_op.conv2d: (4x128x16x80xf32) <- (4x128x16x80xf32, 128x32x5x5xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            reshape_5, parameter_187, [1, 1], [2, 2], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del parameter_187

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(parameter_186, full_int_array_0)
        del parameter_186

        # pd_op.add: (4x128x16x80xf32) <- (4x128x16x80xf32, 1x128x1x1xf32)
        add_14 = paddle._C_ops.add(conv2d_4, reshape_6)

        # pd_op.full: (xf64) <- ()
        full_5 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__2 = paddle._C_ops.assign_value_(
            full_5,
            [],
            paddle.float64,
            [float("0.988235")],
            paddle.framework._current_expected_place(),
        )
        del full_5

        # pd_op.cast: (xf32) <- (xf64)
        cast_2 = paddle._C_ops.cast(assign_value__2, paddle.float32)
        del assign_value__2

        # pd_op.shape64: (4xi64) <- (4x128x16x80xf32)
        shape64_2 = paddle._C_ops.shape64(add_14)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_2

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_2 = [slice_2, full_1, full_1, full_1]
        del slice_2

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_2 = paddle._C_ops.stack(combine_2, 0)
        del combine_2

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_2 = paddle._C_ops.uniform(
            stack_2,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_2

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_15 = paddle._C_ops.add(cast_2, uniform_2)
        del uniform_2

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_2 = paddle._C_ops.floor(add_15)
        del add_15

        # pd_op.divide: (4x128x16x80xf32) <- (4x128x16x80xf32, xf32)
        divide_2 = paddle._C_ops.divide(add_14, cast_2)

        # pd_op.multiply: (4x128x16x80xf32) <- (4x128x16x80xf32, -1x1x1x1xf32)
        multiply_2 = paddle._C_ops.multiply(divide_2, floor_2)

        # pd_op.add: (4x128x16x80xf32) <- (4x128x16x80xf32, 4x128x16x80xf32)
        add_16 = paddle._C_ops.add(reshape_5, multiply_2)

        # pd_op.flatten: (4x128x1280xf32) <- (4x128x16x80xf32)
        flatten_2 = paddle._C_ops.flatten(add_16, 2, 3)

        # pd_op.transpose: (4x1280x128xf32) <- (4x128x1280xf32)
        transpose_4 = paddle._C_ops.transpose(flatten_2, [0, 2, 1])
        del flatten_2

        # pd_op.layer_norm: (4x1280x128xf32, 4x1280xf32, 4x1280xf32) <- (4x1280x128xf32, 128xf32, 128xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_4, parameter_185, parameter_184, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_184, parameter_185

        # pd_op.matmul: (4x1280x512xf32) <- (4x1280x128xf32, 128x512xf32)
        matmul_4 = paddle._C_ops.matmul(layer_norm_12, parameter_183, False, False)
        del parameter_183

        # pd_op.add: (4x1280x512xf32) <- (4x1280x512xf32, 512xf32)
        add_17 = paddle._C_ops.add(matmul_4, parameter_182)
        del parameter_182

        # pd_op.gelu: (4x1280x512xf32) <- (4x1280x512xf32)
        gelu_4 = paddle._C_ops.gelu(add_17, False)

        # pd_op.matmul: (4x1280x128xf32) <- (4x1280x512xf32, 512x128xf32)
        matmul_5 = paddle._C_ops.matmul(gelu_4, parameter_181, False, False)
        del parameter_181

        # pd_op.add: (4x1280x128xf32) <- (4x1280x128xf32, 128xf32)
        add_18 = paddle._C_ops.add(matmul_5, parameter_180)
        del parameter_180

        # pd_op.full: (xf64) <- ()
        full_6 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__3 = paddle._C_ops.assign_value_(
            full_6,
            [],
            paddle.float64,
            [float("0.988235")],
            paddle.framework._current_expected_place(),
        )
        del full_6

        # pd_op.cast: (xf32) <- (xf64)
        cast_3 = paddle._C_ops.cast(assign_value__3, paddle.float32)
        del assign_value__3

        # pd_op.shape64: (3xi64) <- (4x1280x128xf32)
        shape64_3 = paddle._C_ops.shape64(add_18)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_3

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_3 = [slice_3, full_1, full_1]
        del slice_3

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_3 = paddle._C_ops.stack(combine_3, 0)
        del combine_3

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_3 = paddle._C_ops.uniform(
            stack_3,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_3

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_19 = paddle._C_ops.add(cast_3, uniform_3)
        del uniform_3

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_3 = paddle._C_ops.floor(add_19)
        del add_19

        # pd_op.divide: (4x1280x128xf32) <- (4x1280x128xf32, xf32)
        divide_3 = paddle._C_ops.divide(add_18, cast_3)

        # pd_op.multiply: (4x1280x128xf32) <- (4x1280x128xf32, -1x1x1xf32)
        multiply_3 = paddle._C_ops.multiply(divide_3, floor_3)

        # pd_op.add: (4x1280x128xf32) <- (4x1280x128xf32, 4x1280x128xf32)
        add_20 = paddle._C_ops.add(layer_norm_12, multiply_3)

        # pd_op.layer_norm: (4x1280x128xf32, 4x1280xf32, 4x1280xf32) <- (4x1280x128xf32, 128xf32, 128xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_20, parameter_179, parameter_178, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_178, parameter_179

        # pd_op.transpose: (4x128x1280xf32) <- (4x1280x128xf32)
        transpose_5 = paddle._C_ops.transpose(layer_norm_15, [0, 2, 1])
        del layer_norm_15

        # pd_op.reshape: (4x128x16x80xf32) <- (4x128x1280xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_5, full_int_array_1)

        # pd_op.conv2d: (4x128x16x80xf32) <- (4x128x16x80xf32, 128x32x5x5xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            reshape_7, parameter_177, [1, 1], [2, 2], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del parameter_177

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(parameter_176, full_int_array_0)
        del parameter_176

        # pd_op.add: (4x128x16x80xf32) <- (4x128x16x80xf32, 1x128x1x1xf32)
        add_21 = paddle._C_ops.add(conv2d_5, reshape_8)

        # pd_op.full: (xf64) <- ()
        full_7 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__4 = paddle._C_ops.assign_value_(
            full_7,
            [],
            paddle.float64,
            [float("0.982353")],
            paddle.framework._current_expected_place(),
        )
        del full_7

        # pd_op.cast: (xf32) <- (xf64)
        cast_4 = paddle._C_ops.cast(assign_value__4, paddle.float32)
        del assign_value__4

        # pd_op.shape64: (4xi64) <- (4x128x16x80xf32)
        shape64_4 = paddle._C_ops.shape64(add_21)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            shape64_4, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_4

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_4 = [slice_4, full_1, full_1, full_1]
        del slice_4

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_4 = paddle._C_ops.stack(combine_4, 0)
        del combine_4

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_4 = paddle._C_ops.uniform(
            stack_4,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_4

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_22 = paddle._C_ops.add(cast_4, uniform_4)
        del uniform_4

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_4 = paddle._C_ops.floor(add_22)
        del add_22

        # pd_op.divide: (4x128x16x80xf32) <- (4x128x16x80xf32, xf32)
        divide_4 = paddle._C_ops.divide(add_21, cast_4)

        # pd_op.multiply: (4x128x16x80xf32) <- (4x128x16x80xf32, -1x1x1x1xf32)
        multiply_4 = paddle._C_ops.multiply(divide_4, floor_4)

        # pd_op.add: (4x128x16x80xf32) <- (4x128x16x80xf32, 4x128x16x80xf32)
        add_23 = paddle._C_ops.add(reshape_7, multiply_4)

        # pd_op.flatten: (4x128x1280xf32) <- (4x128x16x80xf32)
        flatten_3 = paddle._C_ops.flatten(add_23, 2, 3)

        # pd_op.transpose: (4x1280x128xf32) <- (4x128x1280xf32)
        transpose_6 = paddle._C_ops.transpose(flatten_3, [0, 2, 1])
        del flatten_3

        # pd_op.layer_norm: (4x1280x128xf32, 4x1280xf32, 4x1280xf32) <- (4x1280x128xf32, 128xf32, 128xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_6, parameter_175, parameter_174, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_174, parameter_175

        # pd_op.matmul: (4x1280x512xf32) <- (4x1280x128xf32, 128x512xf32)
        matmul_6 = paddle._C_ops.matmul(layer_norm_18, parameter_173, False, False)
        del parameter_173

        # pd_op.add: (4x1280x512xf32) <- (4x1280x512xf32, 512xf32)
        add_24 = paddle._C_ops.add(matmul_6, parameter_172)
        del parameter_172

        # pd_op.gelu: (4x1280x512xf32) <- (4x1280x512xf32)
        gelu_5 = paddle._C_ops.gelu(add_24, False)

        # pd_op.matmul: (4x1280x128xf32) <- (4x1280x512xf32, 512x128xf32)
        matmul_7 = paddle._C_ops.matmul(gelu_5, parameter_171, False, False)
        del parameter_171

        # pd_op.add: (4x1280x128xf32) <- (4x1280x128xf32, 128xf32)
        add_25 = paddle._C_ops.add(matmul_7, parameter_170)
        del parameter_170

        # pd_op.full: (xf64) <- ()
        full_8 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__5 = paddle._C_ops.assign_value_(
            full_8,
            [],
            paddle.float64,
            [float("0.982353")],
            paddle.framework._current_expected_place(),
        )
        del full_8

        # pd_op.cast: (xf32) <- (xf64)
        cast_5 = paddle._C_ops.cast(assign_value__5, paddle.float32)
        del assign_value__5

        # pd_op.shape64: (3xi64) <- (4x1280x128xf32)
        shape64_5 = paddle._C_ops.shape64(add_25)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            shape64_5, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_5

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_5 = [slice_5, full_1, full_1]
        del slice_5

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_5 = paddle._C_ops.stack(combine_5, 0)
        del combine_5

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_5 = paddle._C_ops.uniform(
            stack_5,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_5

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_26 = paddle._C_ops.add(cast_5, uniform_5)
        del uniform_5

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_5 = paddle._C_ops.floor(add_26)
        del add_26

        # pd_op.divide: (4x1280x128xf32) <- (4x1280x128xf32, xf32)
        divide_5 = paddle._C_ops.divide(add_25, cast_5)

        # pd_op.multiply: (4x1280x128xf32) <- (4x1280x128xf32, -1x1x1xf32)
        multiply_5 = paddle._C_ops.multiply(divide_5, floor_5)

        # pd_op.add: (4x1280x128xf32) <- (4x1280x128xf32, 4x1280x128xf32)
        add_27 = paddle._C_ops.add(layer_norm_18, multiply_5)

        # pd_op.layer_norm: (4x1280x128xf32, 4x1280xf32, 4x1280xf32) <- (4x1280x128xf32, 128xf32, 128xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_27, parameter_169, parameter_168, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_168, parameter_169

        # pd_op.transpose: (4x128x1280xf32) <- (4x1280x128xf32)
        transpose_7 = paddle._C_ops.transpose(layer_norm_21, [0, 2, 1])
        del layer_norm_21

        # pd_op.reshape: (4x128x16x80xf32) <- (4x128x1280xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(transpose_7, full_int_array_1)

        # pd_op.conv2d: (4x128x16x80xf32) <- (4x128x16x80xf32, 128x32x5x5xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            reshape_9, parameter_167, [1, 1], [2, 2], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del parameter_167

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(parameter_166, full_int_array_0)
        del parameter_166

        # pd_op.add: (4x128x16x80xf32) <- (4x128x16x80xf32, 1x128x1x1xf32)
        add_28 = paddle._C_ops.add(conv2d_6, reshape_10)

        # pd_op.full: (xf64) <- ()
        full_9 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__6 = paddle._C_ops.assign_value_(
            full_9,
            [],
            paddle.float64,
            [float("0.976471")],
            paddle.framework._current_expected_place(),
        )
        del full_9

        # pd_op.cast: (xf32) <- (xf64)
        cast_6 = paddle._C_ops.cast(assign_value__6, paddle.float32)
        del assign_value__6

        # pd_op.shape64: (4xi64) <- (4x128x16x80xf32)
        shape64_6 = paddle._C_ops.shape64(add_28)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            shape64_6, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_6

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_6 = [slice_6, full_1, full_1, full_1]
        del slice_6

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_6 = paddle._C_ops.stack(combine_6, 0)
        del combine_6

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_6 = paddle._C_ops.uniform(
            stack_6,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_6

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_29 = paddle._C_ops.add(cast_6, uniform_6)
        del uniform_6

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_6 = paddle._C_ops.floor(add_29)
        del add_29

        # pd_op.divide: (4x128x16x80xf32) <- (4x128x16x80xf32, xf32)
        divide_6 = paddle._C_ops.divide(add_28, cast_6)

        # pd_op.multiply: (4x128x16x80xf32) <- (4x128x16x80xf32, -1x1x1x1xf32)
        multiply_6 = paddle._C_ops.multiply(divide_6, floor_6)

        # pd_op.add: (4x128x16x80xf32) <- (4x128x16x80xf32, 4x128x16x80xf32)
        add_30 = paddle._C_ops.add(reshape_9, multiply_6)

        # pd_op.flatten: (4x128x1280xf32) <- (4x128x16x80xf32)
        flatten_4 = paddle._C_ops.flatten(add_30, 2, 3)

        # pd_op.transpose: (4x1280x128xf32) <- (4x128x1280xf32)
        transpose_8 = paddle._C_ops.transpose(flatten_4, [0, 2, 1])
        del flatten_4

        # pd_op.layer_norm: (4x1280x128xf32, 4x1280xf32, 4x1280xf32) <- (4x1280x128xf32, 128xf32, 128xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_8, parameter_165, parameter_164, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_164, parameter_165

        # pd_op.matmul: (4x1280x512xf32) <- (4x1280x128xf32, 128x512xf32)
        matmul_8 = paddle._C_ops.matmul(layer_norm_24, parameter_163, False, False)
        del parameter_163

        # pd_op.add: (4x1280x512xf32) <- (4x1280x512xf32, 512xf32)
        add_31 = paddle._C_ops.add(matmul_8, parameter_162)
        del parameter_162

        # pd_op.gelu: (4x1280x512xf32) <- (4x1280x512xf32)
        gelu_6 = paddle._C_ops.gelu(add_31, False)

        # pd_op.matmul: (4x1280x128xf32) <- (4x1280x512xf32, 512x128xf32)
        matmul_9 = paddle._C_ops.matmul(gelu_6, parameter_161, False, False)
        del parameter_161

        # pd_op.add: (4x1280x128xf32) <- (4x1280x128xf32, 128xf32)
        add_32 = paddle._C_ops.add(matmul_9, parameter_160)
        del parameter_160

        # pd_op.full: (xf64) <- ()
        full_10 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__7 = paddle._C_ops.assign_value_(
            full_10,
            [],
            paddle.float64,
            [float("0.976471")],
            paddle.framework._current_expected_place(),
        )
        del full_10

        # pd_op.cast: (xf32) <- (xf64)
        cast_7 = paddle._C_ops.cast(assign_value__7, paddle.float32)
        del assign_value__7

        # pd_op.shape64: (3xi64) <- (4x1280x128xf32)
        shape64_7 = paddle._C_ops.shape64(add_32)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            shape64_7, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_7

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_7 = [slice_7, full_1, full_1]
        del slice_7

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_7 = paddle._C_ops.stack(combine_7, 0)
        del combine_7

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_7 = paddle._C_ops.uniform(
            stack_7,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_7

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_33 = paddle._C_ops.add(cast_7, uniform_7)
        del uniform_7

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_7 = paddle._C_ops.floor(add_33)
        del add_33

        # pd_op.divide: (4x1280x128xf32) <- (4x1280x128xf32, xf32)
        divide_7 = paddle._C_ops.divide(add_32, cast_7)

        # pd_op.multiply: (4x1280x128xf32) <- (4x1280x128xf32, -1x1x1xf32)
        multiply_7 = paddle._C_ops.multiply(divide_7, floor_7)

        # pd_op.add: (4x1280x128xf32) <- (4x1280x128xf32, 4x1280x128xf32)
        add_34 = paddle._C_ops.add(layer_norm_24, multiply_7)

        # pd_op.layer_norm: (4x1280x128xf32, 4x1280xf32, 4x1280xf32) <- (4x1280x128xf32, 128xf32, 128xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_34, parameter_159, parameter_158, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_158, parameter_159

        # pd_op.transpose: (4x128x1280xf32) <- (4x1280x128xf32)
        transpose_9 = paddle._C_ops.transpose(layer_norm_27, [0, 2, 1])
        del layer_norm_27

        # pd_op.reshape: (4x128x16x80xf32) <- (4x128x1280xf32, 4xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_9, full_int_array_1)

        # pd_op.conv2d: (4x128x16x80xf32) <- (4x128x16x80xf32, 128x32x5x5xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            reshape_11, parameter_157, [1, 1], [2, 2], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del parameter_157

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(parameter_156, full_int_array_0)
        del parameter_156

        # pd_op.add: (4x128x16x80xf32) <- (4x128x16x80xf32, 1x128x1x1xf32)
        add_35 = paddle._C_ops.add(conv2d_7, reshape_12)

        # pd_op.full: (xf64) <- ()
        full_11 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__8 = paddle._C_ops.assign_value_(
            full_11,
            [],
            paddle.float64,
            [float("0.970588")],
            paddle.framework._current_expected_place(),
        )
        del full_11

        # pd_op.cast: (xf32) <- (xf64)
        cast_8 = paddle._C_ops.cast(assign_value__8, paddle.float32)
        del assign_value__8

        # pd_op.shape64: (4xi64) <- (4x128x16x80xf32)
        shape64_8 = paddle._C_ops.shape64(add_35)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            shape64_8, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_8

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_8 = [slice_8, full_1, full_1, full_1]
        del slice_8

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_8 = paddle._C_ops.stack(combine_8, 0)
        del combine_8

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_8 = paddle._C_ops.uniform(
            stack_8,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_8

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_36 = paddle._C_ops.add(cast_8, uniform_8)
        del uniform_8

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_8 = paddle._C_ops.floor(add_36)
        del add_36

        # pd_op.divide: (4x128x16x80xf32) <- (4x128x16x80xf32, xf32)
        divide_8 = paddle._C_ops.divide(add_35, cast_8)

        # pd_op.multiply: (4x128x16x80xf32) <- (4x128x16x80xf32, -1x1x1x1xf32)
        multiply_8 = paddle._C_ops.multiply(divide_8, floor_8)

        # pd_op.add: (4x128x16x80xf32) <- (4x128x16x80xf32, 4x128x16x80xf32)
        add_37 = paddle._C_ops.add(reshape_11, multiply_8)

        # pd_op.flatten: (4x128x1280xf32) <- (4x128x16x80xf32)
        flatten_5 = paddle._C_ops.flatten(add_37, 2, 3)

        # pd_op.transpose: (4x1280x128xf32) <- (4x128x1280xf32)
        transpose_10 = paddle._C_ops.transpose(flatten_5, [0, 2, 1])
        del flatten_5

        # pd_op.layer_norm: (4x1280x128xf32, 4x1280xf32, 4x1280xf32) <- (4x1280x128xf32, 128xf32, 128xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_10, parameter_155, parameter_154, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_154, parameter_155

        # pd_op.matmul: (4x1280x512xf32) <- (4x1280x128xf32, 128x512xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_30, parameter_153, False, False)
        del parameter_153

        # pd_op.add: (4x1280x512xf32) <- (4x1280x512xf32, 512xf32)
        add_38 = paddle._C_ops.add(matmul_10, parameter_152)
        del parameter_152

        # pd_op.gelu: (4x1280x512xf32) <- (4x1280x512xf32)
        gelu_7 = paddle._C_ops.gelu(add_38, False)

        # pd_op.matmul: (4x1280x128xf32) <- (4x1280x512xf32, 512x128xf32)
        matmul_11 = paddle._C_ops.matmul(gelu_7, parameter_151, False, False)
        del parameter_151

        # pd_op.add: (4x1280x128xf32) <- (4x1280x128xf32, 128xf32)
        add_39 = paddle._C_ops.add(matmul_11, parameter_150)
        del parameter_150

        # pd_op.full: (xf64) <- ()
        full_12 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__9 = paddle._C_ops.assign_value_(
            full_12,
            [],
            paddle.float64,
            [float("0.970588")],
            paddle.framework._current_expected_place(),
        )
        del full_12

        # pd_op.cast: (xf32) <- (xf64)
        cast_9 = paddle._C_ops.cast(assign_value__9, paddle.float32)
        del assign_value__9

        # pd_op.shape64: (3xi64) <- (4x1280x128xf32)
        shape64_9 = paddle._C_ops.shape64(add_39)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            shape64_9, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_9

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_9 = [slice_9, full_1, full_1]
        del slice_9

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_9 = paddle._C_ops.stack(combine_9, 0)
        del combine_9

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_9 = paddle._C_ops.uniform(
            stack_9,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_9

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_40 = paddle._C_ops.add(cast_9, uniform_9)
        del uniform_9

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_9 = paddle._C_ops.floor(add_40)
        del add_40

        # pd_op.divide: (4x1280x128xf32) <- (4x1280x128xf32, xf32)
        divide_9 = paddle._C_ops.divide(add_39, cast_9)

        # pd_op.multiply: (4x1280x128xf32) <- (4x1280x128xf32, -1x1x1xf32)
        multiply_9 = paddle._C_ops.multiply(divide_9, floor_9)

        # pd_op.add: (4x1280x128xf32) <- (4x1280x128xf32, 4x1280x128xf32)
        add_41 = paddle._C_ops.add(layer_norm_30, multiply_9)

        # pd_op.layer_norm: (4x1280x128xf32, 4x1280xf32, 4x1280xf32) <- (4x1280x128xf32, 128xf32, 128xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_41, parameter_149, parameter_148, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_148, parameter_149

        # pd_op.transpose: (4x128x1280xf32) <- (4x1280x128xf32)
        transpose_11 = paddle._C_ops.transpose(layer_norm_33, [0, 2, 1])
        del layer_norm_33

        # pd_op.reshape: (4x128x16x80xf32) <- (4x128x1280xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(transpose_11, full_int_array_1)
        del full_int_array_1

        # pd_op.conv2d: (4x256x8x80xf32) <- (4x128x16x80xf32, 256x128x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            reshape_13, parameter_147, [2, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_147

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(parameter_146, full_int_array_0)
        del parameter_146

        # pd_op.add: (4x256x8x80xf32) <- (4x256x8x80xf32, 1x256x1x1xf32)
        add_42 = paddle._C_ops.add(conv2d_8, reshape_14)

        # pd_op.flatten: (4x256x640xf32) <- (4x256x8x80xf32)
        flatten_6 = paddle._C_ops.flatten(add_42, 2, 3)

        # pd_op.transpose: (4x640x256xf32) <- (4x256x640xf32)
        transpose_12 = paddle._C_ops.transpose(flatten_6, [0, 2, 1])
        del flatten_6

        # pd_op.layer_norm: (4x640x256xf32, 4x640xf32, 4x640xf32) <- (4x640x256xf32, 256xf32, 256xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_12, parameter_145, parameter_144, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_144, parameter_145

        # pd_op.transpose: (4x256x640xf32) <- (4x640x256xf32)
        transpose_13 = paddle._C_ops.transpose(layer_norm_36, [0, 2, 1])
        del layer_norm_36

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_4 = [0, 256, 8, 80]

        # pd_op.reshape: (4x256x8x80xf32) <- (4x256x640xf32, 4xi64)
        reshape_15 = paddle._C_ops.reshape(transpose_13, full_int_array_4)

        # pd_op.conv2d: (4x256x8x80xf32) <- (4x256x8x80xf32, 256x32x5x5xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            reshape_15, parameter_143, [1, 1], [2, 2], "EXPLICIT", [1, 1], 8, "NCHW"
        )
        del parameter_143

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_16 = paddle._C_ops.reshape(parameter_142, full_int_array_0)
        del parameter_142

        # pd_op.add: (4x256x8x80xf32) <- (4x256x8x80xf32, 1x256x1x1xf32)
        add_43 = paddle._C_ops.add(conv2d_9, reshape_16)

        # pd_op.full: (xf64) <- ()
        full_13 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__10 = paddle._C_ops.assign_value_(
            full_13,
            [],
            paddle.float64,
            [float("0.964706")],
            paddle.framework._current_expected_place(),
        )
        del full_13

        # pd_op.cast: (xf32) <- (xf64)
        cast_10 = paddle._C_ops.cast(assign_value__10, paddle.float32)
        del assign_value__10

        # pd_op.shape64: (4xi64) <- (4x256x8x80xf32)
        shape64_10 = paddle._C_ops.shape64(add_43)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            shape64_10, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_10

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_10 = [slice_10, full_1, full_1, full_1]
        del slice_10

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_10 = paddle._C_ops.stack(combine_10, 0)
        del combine_10

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_10 = paddle._C_ops.uniform(
            stack_10,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_10

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_44 = paddle._C_ops.add(cast_10, uniform_10)
        del uniform_10

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_10 = paddle._C_ops.floor(add_44)
        del add_44

        # pd_op.divide: (4x256x8x80xf32) <- (4x256x8x80xf32, xf32)
        divide_10 = paddle._C_ops.divide(add_43, cast_10)

        # pd_op.multiply: (4x256x8x80xf32) <- (4x256x8x80xf32, -1x1x1x1xf32)
        multiply_10 = paddle._C_ops.multiply(divide_10, floor_10)

        # pd_op.add: (4x256x8x80xf32) <- (4x256x8x80xf32, 4x256x8x80xf32)
        add_45 = paddle._C_ops.add(reshape_15, multiply_10)

        # pd_op.flatten: (4x256x640xf32) <- (4x256x8x80xf32)
        flatten_7 = paddle._C_ops.flatten(add_45, 2, 3)

        # pd_op.transpose: (4x640x256xf32) <- (4x256x640xf32)
        transpose_14 = paddle._C_ops.transpose(flatten_7, [0, 2, 1])
        del flatten_7

        # pd_op.layer_norm: (4x640x256xf32, 4x640xf32, 4x640xf32) <- (4x640x256xf32, 256xf32, 256xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_14, parameter_141, parameter_140, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_140, parameter_141

        # pd_op.matmul: (4x640x1024xf32) <- (4x640x256xf32, 256x1024xf32)
        matmul_12 = paddle._C_ops.matmul(layer_norm_39, parameter_139, False, False)
        del parameter_139

        # pd_op.add: (4x640x1024xf32) <- (4x640x1024xf32, 1024xf32)
        add_46 = paddle._C_ops.add(matmul_12, parameter_138)
        del parameter_138

        # pd_op.gelu: (4x640x1024xf32) <- (4x640x1024xf32)
        gelu_8 = paddle._C_ops.gelu(add_46, False)

        # pd_op.matmul: (4x640x256xf32) <- (4x640x1024xf32, 1024x256xf32)
        matmul_13 = paddle._C_ops.matmul(gelu_8, parameter_137, False, False)
        del parameter_137

        # pd_op.add: (4x640x256xf32) <- (4x640x256xf32, 256xf32)
        add_47 = paddle._C_ops.add(matmul_13, parameter_136)
        del parameter_136

        # pd_op.full: (xf64) <- ()
        full_14 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__11 = paddle._C_ops.assign_value_(
            full_14,
            [],
            paddle.float64,
            [float("0.964706")],
            paddle.framework._current_expected_place(),
        )
        del full_14

        # pd_op.cast: (xf32) <- (xf64)
        cast_11 = paddle._C_ops.cast(assign_value__11, paddle.float32)
        del assign_value__11

        # pd_op.shape64: (3xi64) <- (4x640x256xf32)
        shape64_11 = paddle._C_ops.shape64(add_47)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            shape64_11, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_11

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_11 = [slice_11, full_1, full_1]
        del slice_11

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_11 = paddle._C_ops.stack(combine_11, 0)
        del combine_11

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_11 = paddle._C_ops.uniform(
            stack_11,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_11

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_48 = paddle._C_ops.add(cast_11, uniform_11)
        del uniform_11

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_11 = paddle._C_ops.floor(add_48)
        del add_48

        # pd_op.divide: (4x640x256xf32) <- (4x640x256xf32, xf32)
        divide_11 = paddle._C_ops.divide(add_47, cast_11)

        # pd_op.multiply: (4x640x256xf32) <- (4x640x256xf32, -1x1x1xf32)
        multiply_11 = paddle._C_ops.multiply(divide_11, floor_11)

        # pd_op.add: (4x640x256xf32) <- (4x640x256xf32, 4x640x256xf32)
        add_49 = paddle._C_ops.add(layer_norm_39, multiply_11)

        # pd_op.layer_norm: (4x640x256xf32, 4x640xf32, 4x640xf32) <- (4x640x256xf32, 256xf32, 256xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_49, parameter_135, parameter_134, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_134, parameter_135

        # pd_op.transpose: (4x256x640xf32) <- (4x640x256xf32)
        transpose_15 = paddle._C_ops.transpose(layer_norm_42, [0, 2, 1])
        del layer_norm_42

        # pd_op.reshape: (4x256x8x80xf32) <- (4x256x640xf32, 4xi64)
        reshape_17 = paddle._C_ops.reshape(transpose_15, full_int_array_4)

        # pd_op.conv2d: (4x256x8x80xf32) <- (4x256x8x80xf32, 256x32x5x5xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            reshape_17, parameter_133, [1, 1], [2, 2], "EXPLICIT", [1, 1], 8, "NCHW"
        )
        del parameter_133

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_18 = paddle._C_ops.reshape(parameter_132, full_int_array_0)
        del parameter_132

        # pd_op.add: (4x256x8x80xf32) <- (4x256x8x80xf32, 1x256x1x1xf32)
        add_50 = paddle._C_ops.add(conv2d_10, reshape_18)

        # pd_op.full: (xf64) <- ()
        full_15 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__12 = paddle._C_ops.assign_value_(
            full_15,
            [],
            paddle.float64,
            [float("0.958824")],
            paddle.framework._current_expected_place(),
        )
        del full_15

        # pd_op.cast: (xf32) <- (xf64)
        cast_12 = paddle._C_ops.cast(assign_value__12, paddle.float32)
        del assign_value__12

        # pd_op.shape64: (4xi64) <- (4x256x8x80xf32)
        shape64_12 = paddle._C_ops.shape64(add_50)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            shape64_12, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_12

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_12 = [slice_12, full_1, full_1, full_1]
        del slice_12

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_12 = paddle._C_ops.stack(combine_12, 0)
        del combine_12

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_12 = paddle._C_ops.uniform(
            stack_12,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_12

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_51 = paddle._C_ops.add(cast_12, uniform_12)
        del uniform_12

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_12 = paddle._C_ops.floor(add_51)
        del add_51

        # pd_op.divide: (4x256x8x80xf32) <- (4x256x8x80xf32, xf32)
        divide_12 = paddle._C_ops.divide(add_50, cast_12)

        # pd_op.multiply: (4x256x8x80xf32) <- (4x256x8x80xf32, -1x1x1x1xf32)
        multiply_12 = paddle._C_ops.multiply(divide_12, floor_12)

        # pd_op.add: (4x256x8x80xf32) <- (4x256x8x80xf32, 4x256x8x80xf32)
        add_52 = paddle._C_ops.add(reshape_17, multiply_12)

        # pd_op.flatten: (4x256x640xf32) <- (4x256x8x80xf32)
        flatten_8 = paddle._C_ops.flatten(add_52, 2, 3)

        # pd_op.transpose: (4x640x256xf32) <- (4x256x640xf32)
        transpose_16 = paddle._C_ops.transpose(flatten_8, [0, 2, 1])
        del flatten_8

        # pd_op.layer_norm: (4x640x256xf32, 4x640xf32, 4x640xf32) <- (4x640x256xf32, 256xf32, 256xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_16, parameter_131, parameter_130, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_130, parameter_131

        # pd_op.matmul: (4x640x1024xf32) <- (4x640x256xf32, 256x1024xf32)
        matmul_14 = paddle._C_ops.matmul(layer_norm_45, parameter_129, False, False)
        del parameter_129

        # pd_op.add: (4x640x1024xf32) <- (4x640x1024xf32, 1024xf32)
        add_53 = paddle._C_ops.add(matmul_14, parameter_128)
        del parameter_128

        # pd_op.gelu: (4x640x1024xf32) <- (4x640x1024xf32)
        gelu_9 = paddle._C_ops.gelu(add_53, False)

        # pd_op.matmul: (4x640x256xf32) <- (4x640x1024xf32, 1024x256xf32)
        matmul_15 = paddle._C_ops.matmul(gelu_9, parameter_127, False, False)
        del parameter_127

        # pd_op.add: (4x640x256xf32) <- (4x640x256xf32, 256xf32)
        add_54 = paddle._C_ops.add(matmul_15, parameter_126)
        del parameter_126

        # pd_op.full: (xf64) <- ()
        full_16 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__13 = paddle._C_ops.assign_value_(
            full_16,
            [],
            paddle.float64,
            [float("0.958824")],
            paddle.framework._current_expected_place(),
        )
        del full_16

        # pd_op.cast: (xf32) <- (xf64)
        cast_13 = paddle._C_ops.cast(assign_value__13, paddle.float32)
        del assign_value__13

        # pd_op.shape64: (3xi64) <- (4x640x256xf32)
        shape64_13 = paddle._C_ops.shape64(add_54)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            shape64_13, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_13

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_13 = [slice_13, full_1, full_1]
        del slice_13

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_13 = paddle._C_ops.stack(combine_13, 0)
        del combine_13

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_13 = paddle._C_ops.uniform(
            stack_13,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_13

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_55 = paddle._C_ops.add(cast_13, uniform_13)
        del uniform_13

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_13 = paddle._C_ops.floor(add_55)
        del add_55

        # pd_op.divide: (4x640x256xf32) <- (4x640x256xf32, xf32)
        divide_13 = paddle._C_ops.divide(add_54, cast_13)

        # pd_op.multiply: (4x640x256xf32) <- (4x640x256xf32, -1x1x1xf32)
        multiply_13 = paddle._C_ops.multiply(divide_13, floor_13)

        # pd_op.add: (4x640x256xf32) <- (4x640x256xf32, 4x640x256xf32)
        add_56 = paddle._C_ops.add(layer_norm_45, multiply_13)

        # pd_op.layer_norm: (4x640x256xf32, 4x640xf32, 4x640xf32) <- (4x640x256xf32, 256xf32, 256xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_56, parameter_125, parameter_124, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_124, parameter_125

        # pd_op.transpose: (4x256x640xf32) <- (4x640x256xf32)
        transpose_17 = paddle._C_ops.transpose(layer_norm_48, [0, 2, 1])
        del layer_norm_48

        # pd_op.reshape: (4x256x8x80xf32) <- (4x256x640xf32, 4xi64)
        reshape_19 = paddle._C_ops.reshape(transpose_17, full_int_array_4)

        # pd_op.flatten: (4x256x640xf32) <- (4x256x8x80xf32)
        flatten_9 = paddle._C_ops.flatten(reshape_19, 2, 3)

        # pd_op.transpose: (4x640x256xf32) <- (4x256x640xf32)
        transpose_18 = paddle._C_ops.transpose(flatten_9, [0, 2, 1])
        del flatten_9

        # pd_op.matmul: (4x640x768xf32) <- (4x640x256xf32, 256x768xf32)
        matmul_16 = paddle._C_ops.matmul(transpose_18, parameter_123, False, False)
        del parameter_123

        # pd_op.add: (4x640x768xf32) <- (4x640x768xf32, 768xf32)
        add_57 = paddle._C_ops.add(matmul_16, parameter_122)
        del parameter_122

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_5 = [0, -1, 3, 8, 32]

        # pd_op.reshape: (4x640x3x8x32xf32) <- (4x640x768xf32, 5xi64)
        reshape_20 = paddle._C_ops.reshape(add_57, full_int_array_5)

        # pd_op.transpose: (3x4x8x640x32xf32) <- (4x640x3x8x32xf32)
        transpose_19 = paddle._C_ops.transpose(reshape_20, [2, 0, 3, 1, 4])
        del reshape_20

        # pd_op.slice: (4x8x640x32xf32) <- (3x4x8x640x32xf32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            transpose_19, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_30 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_31 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_32 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_33 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_34 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_35 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_36 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_37 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_38 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_39 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_40 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_41 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_42 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_43 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_44 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_45 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_46 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_47 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_48 = full_int_array_6

        # pd_op.slice: (4x8x640x32xf32) <- (3x4x8x640x32xf32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            transpose_19, [0], full_int_array_3, full_int_array_6, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_49 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_50 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_51 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_52 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_53 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_54 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_55 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_56 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_57 = full_int_array_7

        # pd_op.slice: (4x8x640x32xf32) <- (3x4x8x640x32xf32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            transpose_19, [0], full_int_array_6, full_int_array_7, [1], [0]
        )

        # pd_op.transpose: (4x8x32x640xf32) <- (4x8x640x32xf32)
        transpose_20 = paddle._C_ops.transpose(slice_15, [0, 1, 3, 2])
        del slice_15

        # pd_op.matmul: (4x8x640x640xf32) <- (4x8x640x32xf32, 4x8x32x640xf32)
        matmul_17 = paddle._C_ops.matmul(slice_14, transpose_20, False, False)

        # pd_op.full: (1xf32) <- ()
        full_17 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_58 = full_17

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_59 = full_17

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_60 = full_17

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_61 = full_17

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_62 = full_17

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_63 = full_17

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_64 = full_17

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_65 = full_17

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_66 = full_17

        # pd_op.scale: (4x8x640x640xf32) <- (4x8x640x640xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(matmul_17, full_17, float("0"), True)
        del matmul_17

        # pd_op.softmax: (4x8x640x640xf32) <- (4x8x640x640xf32)
        softmax_0 = paddle._C_ops.softmax(scale_0, -1)
        del scale_0

        # pd_op.matmul: (4x8x640x32xf32) <- (4x8x640x640xf32, 4x8x640x32xf32)
        matmul_18 = paddle._C_ops.matmul(softmax_0, slice_16, False, False)

        # pd_op.transpose: (4x640x8x32xf32) <- (4x8x640x32xf32)
        transpose_21 = paddle._C_ops.transpose(matmul_18, [0, 2, 1, 3])
        del matmul_18

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_8 = [0, -1, 256]

        # pd_op.reshape: (4x640x256xf32) <- (4x640x8x32xf32, 3xi64)
        reshape_21 = paddle._C_ops.reshape(transpose_21, full_int_array_8)

        # pd_op.matmul: (4x640x256xf32) <- (4x640x256xf32, 256x256xf32)
        matmul_19 = paddle._C_ops.matmul(reshape_21, parameter_121, False, False)
        del parameter_121

        # pd_op.add: (4x640x256xf32) <- (4x640x256xf32, 256xf32)
        add_58 = paddle._C_ops.add(matmul_19, parameter_120)
        del parameter_120

        # pd_op.full: (xf64) <- ()
        full_18 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__14 = paddle._C_ops.assign_value_(
            full_18,
            [],
            paddle.float64,
            [float("0.952941")],
            paddle.framework._current_expected_place(),
        )
        del full_18

        # pd_op.cast: (xf32) <- (xf64)
        cast_14 = paddle._C_ops.cast(assign_value__14, paddle.float32)
        del assign_value__14

        # pd_op.shape64: (3xi64) <- (4x640x256xf32)
        shape64_14 = paddle._C_ops.shape64(add_58)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            shape64_14, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_14

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_14 = [slice_17, full_1, full_1]
        del slice_17

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_14 = paddle._C_ops.stack(combine_14, 0)
        del combine_14

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_14 = paddle._C_ops.uniform(
            stack_14,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_14

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_59 = paddle._C_ops.add(cast_14, uniform_14)
        del uniform_14

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_14 = paddle._C_ops.floor(add_59)
        del add_59

        # pd_op.divide: (4x640x256xf32) <- (4x640x256xf32, xf32)
        divide_14 = paddle._C_ops.divide(add_58, cast_14)

        # pd_op.multiply: (4x640x256xf32) <- (4x640x256xf32, -1x1x1xf32)
        multiply_14 = paddle._C_ops.multiply(divide_14, floor_14)

        # pd_op.add: (4x640x256xf32) <- (4x640x256xf32, 4x640x256xf32)
        add_60 = paddle._C_ops.add(transpose_18, multiply_14)

        # pd_op.layer_norm: (4x640x256xf32, 4x640xf32, 4x640xf32) <- (4x640x256xf32, 256xf32, 256xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_60, parameter_119, parameter_118, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_118, parameter_119

        # pd_op.matmul: (4x640x1024xf32) <- (4x640x256xf32, 256x1024xf32)
        matmul_20 = paddle._C_ops.matmul(layer_norm_51, parameter_117, False, False)
        del parameter_117

        # pd_op.add: (4x640x1024xf32) <- (4x640x1024xf32, 1024xf32)
        add_61 = paddle._C_ops.add(matmul_20, parameter_116)
        del parameter_116

        # pd_op.gelu: (4x640x1024xf32) <- (4x640x1024xf32)
        gelu_10 = paddle._C_ops.gelu(add_61, False)

        # pd_op.matmul: (4x640x256xf32) <- (4x640x1024xf32, 1024x256xf32)
        matmul_21 = paddle._C_ops.matmul(gelu_10, parameter_115, False, False)
        del parameter_115

        # pd_op.add: (4x640x256xf32) <- (4x640x256xf32, 256xf32)
        add_62 = paddle._C_ops.add(matmul_21, parameter_114)
        del parameter_114

        # pd_op.full: (xf64) <- ()
        full_19 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__15 = paddle._C_ops.assign_value_(
            full_19,
            [],
            paddle.float64,
            [float("0.952941")],
            paddle.framework._current_expected_place(),
        )
        del full_19

        # pd_op.cast: (xf32) <- (xf64)
        cast_15 = paddle._C_ops.cast(assign_value__15, paddle.float32)
        del assign_value__15

        # pd_op.shape64: (3xi64) <- (4x640x256xf32)
        shape64_15 = paddle._C_ops.shape64(add_62)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(
            shape64_15, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_15

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_15 = [slice_18, full_1, full_1]
        del slice_18

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_15 = paddle._C_ops.stack(combine_15, 0)
        del combine_15

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_15 = paddle._C_ops.uniform(
            stack_15,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_15

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_63 = paddle._C_ops.add(cast_15, uniform_15)
        del uniform_15

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_15 = paddle._C_ops.floor(add_63)
        del add_63

        # pd_op.divide: (4x640x256xf32) <- (4x640x256xf32, xf32)
        divide_15 = paddle._C_ops.divide(add_62, cast_15)

        # pd_op.multiply: (4x640x256xf32) <- (4x640x256xf32, -1x1x1xf32)
        multiply_15 = paddle._C_ops.multiply(divide_15, floor_15)

        # pd_op.add: (4x640x256xf32) <- (4x640x256xf32, 4x640x256xf32)
        add_64 = paddle._C_ops.add(layer_norm_51, multiply_15)

        # pd_op.layer_norm: (4x640x256xf32, 4x640xf32, 4x640xf32) <- (4x640x256xf32, 256xf32, 256xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_64, parameter_113, parameter_112, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_112, parameter_113

        # pd_op.matmul: (4x640x768xf32) <- (4x640x256xf32, 256x768xf32)
        matmul_22 = paddle._C_ops.matmul(layer_norm_54, parameter_111, False, False)
        del parameter_111

        # pd_op.add: (4x640x768xf32) <- (4x640x768xf32, 768xf32)
        add_65 = paddle._C_ops.add(matmul_22, parameter_110)
        del parameter_110

        # pd_op.reshape: (4x640x3x8x32xf32) <- (4x640x768xf32, 5xi64)
        reshape_22 = paddle._C_ops.reshape(add_65, full_int_array_5)

        # pd_op.transpose: (3x4x8x640x32xf32) <- (4x640x3x8x32xf32)
        transpose_22 = paddle._C_ops.transpose(reshape_22, [2, 0, 3, 1, 4])
        del reshape_22

        # pd_op.slice: (4x8x640x32xf32) <- (3x4x8x640x32xf32, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(
            transpose_22, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (4x8x640x32xf32) <- (3x4x8x640x32xf32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(
            transpose_22, [0], full_int_array_3, full_int_array_6, [1], [0]
        )

        # pd_op.slice: (4x8x640x32xf32) <- (3x4x8x640x32xf32, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(
            transpose_22, [0], full_int_array_6, full_int_array_7, [1], [0]
        )

        # pd_op.transpose: (4x8x32x640xf32) <- (4x8x640x32xf32)
        transpose_23 = paddle._C_ops.transpose(slice_20, [0, 1, 3, 2])
        del slice_20

        # pd_op.matmul: (4x8x640x640xf32) <- (4x8x640x32xf32, 4x8x32x640xf32)
        matmul_23 = paddle._C_ops.matmul(slice_19, transpose_23, False, False)

        # pd_op.scale: (4x8x640x640xf32) <- (4x8x640x640xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(matmul_23, full_17, float("0"), True)
        del matmul_23

        # pd_op.softmax: (4x8x640x640xf32) <- (4x8x640x640xf32)
        softmax_1 = paddle._C_ops.softmax(scale_1, -1)
        del scale_1

        # pd_op.matmul: (4x8x640x32xf32) <- (4x8x640x640xf32, 4x8x640x32xf32)
        matmul_24 = paddle._C_ops.matmul(softmax_1, slice_21, False, False)

        # pd_op.transpose: (4x640x8x32xf32) <- (4x8x640x32xf32)
        transpose_24 = paddle._C_ops.transpose(matmul_24, [0, 2, 1, 3])
        del matmul_24

        # pd_op.reshape: (4x640x256xf32) <- (4x640x8x32xf32, 3xi64)
        reshape_23 = paddle._C_ops.reshape(transpose_24, full_int_array_8)

        # pd_op.matmul: (4x640x256xf32) <- (4x640x256xf32, 256x256xf32)
        matmul_25 = paddle._C_ops.matmul(reshape_23, parameter_109, False, False)
        del parameter_109

        # pd_op.add: (4x640x256xf32) <- (4x640x256xf32, 256xf32)
        add_66 = paddle._C_ops.add(matmul_25, parameter_108)
        del parameter_108

        # pd_op.full: (xf64) <- ()
        full_20 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__16 = paddle._C_ops.assign_value_(
            full_20,
            [],
            paddle.float64,
            [float("0.947059")],
            paddle.framework._current_expected_place(),
        )
        del full_20

        # pd_op.cast: (xf32) <- (xf64)
        cast_16 = paddle._C_ops.cast(assign_value__16, paddle.float32)
        del assign_value__16

        # pd_op.shape64: (3xi64) <- (4x640x256xf32)
        shape64_16 = paddle._C_ops.shape64(add_66)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(
            shape64_16, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_16

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_16 = [slice_22, full_1, full_1]
        del slice_22

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_16 = paddle._C_ops.stack(combine_16, 0)
        del combine_16

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_16 = paddle._C_ops.uniform(
            stack_16,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_16

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_67 = paddle._C_ops.add(cast_16, uniform_16)
        del uniform_16

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_16 = paddle._C_ops.floor(add_67)
        del add_67

        # pd_op.divide: (4x640x256xf32) <- (4x640x256xf32, xf32)
        divide_16 = paddle._C_ops.divide(add_66, cast_16)

        # pd_op.multiply: (4x640x256xf32) <- (4x640x256xf32, -1x1x1xf32)
        multiply_16 = paddle._C_ops.multiply(divide_16, floor_16)

        # pd_op.add: (4x640x256xf32) <- (4x640x256xf32, 4x640x256xf32)
        add_68 = paddle._C_ops.add(layer_norm_54, multiply_16)

        # pd_op.layer_norm: (4x640x256xf32, 4x640xf32, 4x640xf32) <- (4x640x256xf32, 256xf32, 256xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_68, parameter_107, parameter_106, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_106, parameter_107

        # pd_op.matmul: (4x640x1024xf32) <- (4x640x256xf32, 256x1024xf32)
        matmul_26 = paddle._C_ops.matmul(layer_norm_57, parameter_105, False, False)
        del parameter_105

        # pd_op.add: (4x640x1024xf32) <- (4x640x1024xf32, 1024xf32)
        add_69 = paddle._C_ops.add(matmul_26, parameter_104)
        del parameter_104

        # pd_op.gelu: (4x640x1024xf32) <- (4x640x1024xf32)
        gelu_11 = paddle._C_ops.gelu(add_69, False)

        # pd_op.matmul: (4x640x256xf32) <- (4x640x1024xf32, 1024x256xf32)
        matmul_27 = paddle._C_ops.matmul(gelu_11, parameter_103, False, False)
        del parameter_103

        # pd_op.add: (4x640x256xf32) <- (4x640x256xf32, 256xf32)
        add_70 = paddle._C_ops.add(matmul_27, parameter_102)
        del parameter_102

        # pd_op.full: (xf64) <- ()
        full_21 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__17 = paddle._C_ops.assign_value_(
            full_21,
            [],
            paddle.float64,
            [float("0.947059")],
            paddle.framework._current_expected_place(),
        )
        del full_21

        # pd_op.cast: (xf32) <- (xf64)
        cast_17 = paddle._C_ops.cast(assign_value__17, paddle.float32)
        del assign_value__17

        # pd_op.shape64: (3xi64) <- (4x640x256xf32)
        shape64_17 = paddle._C_ops.shape64(add_70)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(
            shape64_17, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_17

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_17 = [slice_23, full_1, full_1]
        del slice_23

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_17 = paddle._C_ops.stack(combine_17, 0)
        del combine_17

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_17 = paddle._C_ops.uniform(
            stack_17,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_17

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_71 = paddle._C_ops.add(cast_17, uniform_17)
        del uniform_17

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_17 = paddle._C_ops.floor(add_71)
        del add_71

        # pd_op.divide: (4x640x256xf32) <- (4x640x256xf32, xf32)
        divide_17 = paddle._C_ops.divide(add_70, cast_17)

        # pd_op.multiply: (4x640x256xf32) <- (4x640x256xf32, -1x1x1xf32)
        multiply_17 = paddle._C_ops.multiply(divide_17, floor_17)

        # pd_op.add: (4x640x256xf32) <- (4x640x256xf32, 4x640x256xf32)
        add_72 = paddle._C_ops.add(layer_norm_57, multiply_17)

        # pd_op.layer_norm: (4x640x256xf32, 4x640xf32, 4x640xf32) <- (4x640x256xf32, 256xf32, 256xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_72, parameter_101, parameter_100, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_100, parameter_101

        # pd_op.matmul: (4x640x768xf32) <- (4x640x256xf32, 256x768xf32)
        matmul_28 = paddle._C_ops.matmul(layer_norm_60, parameter_99, False, False)
        del parameter_99

        # pd_op.add: (4x640x768xf32) <- (4x640x768xf32, 768xf32)
        add_73 = paddle._C_ops.add(matmul_28, parameter_98)
        del parameter_98

        # pd_op.reshape: (4x640x3x8x32xf32) <- (4x640x768xf32, 5xi64)
        reshape_24 = paddle._C_ops.reshape(add_73, full_int_array_5)

        # pd_op.transpose: (3x4x8x640x32xf32) <- (4x640x3x8x32xf32)
        transpose_25 = paddle._C_ops.transpose(reshape_24, [2, 0, 3, 1, 4])
        del reshape_24

        # pd_op.slice: (4x8x640x32xf32) <- (3x4x8x640x32xf32, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(
            transpose_25, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (4x8x640x32xf32) <- (3x4x8x640x32xf32, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(
            transpose_25, [0], full_int_array_3, full_int_array_6, [1], [0]
        )

        # pd_op.slice: (4x8x640x32xf32) <- (3x4x8x640x32xf32, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(
            transpose_25, [0], full_int_array_6, full_int_array_7, [1], [0]
        )

        # pd_op.transpose: (4x8x32x640xf32) <- (4x8x640x32xf32)
        transpose_26 = paddle._C_ops.transpose(slice_25, [0, 1, 3, 2])
        del slice_25

        # pd_op.matmul: (4x8x640x640xf32) <- (4x8x640x32xf32, 4x8x32x640xf32)
        matmul_29 = paddle._C_ops.matmul(slice_24, transpose_26, False, False)

        # pd_op.scale: (4x8x640x640xf32) <- (4x8x640x640xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(matmul_29, full_17, float("0"), True)
        del matmul_29

        # pd_op.softmax: (4x8x640x640xf32) <- (4x8x640x640xf32)
        softmax_2 = paddle._C_ops.softmax(scale_2, -1)
        del scale_2

        # pd_op.matmul: (4x8x640x32xf32) <- (4x8x640x640xf32, 4x8x640x32xf32)
        matmul_30 = paddle._C_ops.matmul(softmax_2, slice_26, False, False)

        # pd_op.transpose: (4x640x8x32xf32) <- (4x8x640x32xf32)
        transpose_27 = paddle._C_ops.transpose(matmul_30, [0, 2, 1, 3])
        del matmul_30

        # pd_op.reshape: (4x640x256xf32) <- (4x640x8x32xf32, 3xi64)
        reshape_25 = paddle._C_ops.reshape(transpose_27, full_int_array_8)

        # pd_op.matmul: (4x640x256xf32) <- (4x640x256xf32, 256x256xf32)
        matmul_31 = paddle._C_ops.matmul(reshape_25, parameter_97, False, False)
        del parameter_97

        # pd_op.add: (4x640x256xf32) <- (4x640x256xf32, 256xf32)
        add_74 = paddle._C_ops.add(matmul_31, parameter_96)
        del parameter_96

        # pd_op.full: (xf64) <- ()
        full_22 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__18 = paddle._C_ops.assign_value_(
            full_22,
            [],
            paddle.float64,
            [float("0.941176")],
            paddle.framework._current_expected_place(),
        )
        del full_22

        # pd_op.cast: (xf32) <- (xf64)
        cast_18 = paddle._C_ops.cast(assign_value__18, paddle.float32)
        del assign_value__18

        # pd_op.shape64: (3xi64) <- (4x640x256xf32)
        shape64_18 = paddle._C_ops.shape64(add_74)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(
            shape64_18, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_18

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_18 = [slice_27, full_1, full_1]
        del slice_27

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_18 = paddle._C_ops.stack(combine_18, 0)
        del combine_18

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_18 = paddle._C_ops.uniform(
            stack_18,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_18

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_75 = paddle._C_ops.add(cast_18, uniform_18)
        del uniform_18

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_18 = paddle._C_ops.floor(add_75)
        del add_75

        # pd_op.divide: (4x640x256xf32) <- (4x640x256xf32, xf32)
        divide_18 = paddle._C_ops.divide(add_74, cast_18)

        # pd_op.multiply: (4x640x256xf32) <- (4x640x256xf32, -1x1x1xf32)
        multiply_18 = paddle._C_ops.multiply(divide_18, floor_18)

        # pd_op.add: (4x640x256xf32) <- (4x640x256xf32, 4x640x256xf32)
        add_76 = paddle._C_ops.add(layer_norm_60, multiply_18)

        # pd_op.layer_norm: (4x640x256xf32, 4x640xf32, 4x640xf32) <- (4x640x256xf32, 256xf32, 256xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_76, parameter_95, parameter_94, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_94, parameter_95

        # pd_op.matmul: (4x640x1024xf32) <- (4x640x256xf32, 256x1024xf32)
        matmul_32 = paddle._C_ops.matmul(layer_norm_63, parameter_93, False, False)
        del parameter_93

        # pd_op.add: (4x640x1024xf32) <- (4x640x1024xf32, 1024xf32)
        add_77 = paddle._C_ops.add(matmul_32, parameter_92)
        del parameter_92

        # pd_op.gelu: (4x640x1024xf32) <- (4x640x1024xf32)
        gelu_12 = paddle._C_ops.gelu(add_77, False)

        # pd_op.matmul: (4x640x256xf32) <- (4x640x1024xf32, 1024x256xf32)
        matmul_33 = paddle._C_ops.matmul(gelu_12, parameter_91, False, False)
        del parameter_91

        # pd_op.add: (4x640x256xf32) <- (4x640x256xf32, 256xf32)
        add_78 = paddle._C_ops.add(matmul_33, parameter_90)
        del parameter_90

        # pd_op.full: (xf64) <- ()
        full_23 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__19 = paddle._C_ops.assign_value_(
            full_23,
            [],
            paddle.float64,
            [float("0.941176")],
            paddle.framework._current_expected_place(),
        )
        del full_23

        # pd_op.cast: (xf32) <- (xf64)
        cast_19 = paddle._C_ops.cast(assign_value__19, paddle.float32)
        del assign_value__19

        # pd_op.shape64: (3xi64) <- (4x640x256xf32)
        shape64_19 = paddle._C_ops.shape64(add_78)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(
            shape64_19, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_19

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_19 = [slice_28, full_1, full_1]
        del slice_28

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_19 = paddle._C_ops.stack(combine_19, 0)
        del combine_19

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_19 = paddle._C_ops.uniform(
            stack_19,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_19

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_79 = paddle._C_ops.add(cast_19, uniform_19)
        del uniform_19

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_19 = paddle._C_ops.floor(add_79)
        del add_79

        # pd_op.divide: (4x640x256xf32) <- (4x640x256xf32, xf32)
        divide_19 = paddle._C_ops.divide(add_78, cast_19)

        # pd_op.multiply: (4x640x256xf32) <- (4x640x256xf32, -1x1x1xf32)
        multiply_19 = paddle._C_ops.multiply(divide_19, floor_19)

        # pd_op.add: (4x640x256xf32) <- (4x640x256xf32, 4x640x256xf32)
        add_80 = paddle._C_ops.add(layer_norm_63, multiply_19)

        # pd_op.layer_norm: (4x640x256xf32, 4x640xf32, 4x640xf32) <- (4x640x256xf32, 256xf32, 256xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_80, parameter_89, parameter_88, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_88, parameter_89

        # pd_op.matmul: (4x640x768xf32) <- (4x640x256xf32, 256x768xf32)
        matmul_34 = paddle._C_ops.matmul(layer_norm_66, parameter_87, False, False)
        del parameter_87

        # pd_op.add: (4x640x768xf32) <- (4x640x768xf32, 768xf32)
        add_81 = paddle._C_ops.add(matmul_34, parameter_86)
        del parameter_86

        # pd_op.reshape: (4x640x3x8x32xf32) <- (4x640x768xf32, 5xi64)
        reshape_26 = paddle._C_ops.reshape(add_81, full_int_array_5)
        del full_int_array_5

        # pd_op.transpose: (3x4x8x640x32xf32) <- (4x640x3x8x32xf32)
        transpose_28 = paddle._C_ops.transpose(reshape_26, [2, 0, 3, 1, 4])
        del reshape_26

        # pd_op.slice: (4x8x640x32xf32) <- (3x4x8x640x32xf32, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(
            transpose_28, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (4x8x640x32xf32) <- (3x4x8x640x32xf32, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(
            transpose_28, [0], full_int_array_3, full_int_array_6, [1], [0]
        )

        # pd_op.slice: (4x8x640x32xf32) <- (3x4x8x640x32xf32, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(
            transpose_28, [0], full_int_array_6, full_int_array_7, [1], [0]
        )

        # pd_op.transpose: (4x8x32x640xf32) <- (4x8x640x32xf32)
        transpose_29 = paddle._C_ops.transpose(slice_30, [0, 1, 3, 2])
        del slice_30

        # pd_op.matmul: (4x8x640x640xf32) <- (4x8x640x32xf32, 4x8x32x640xf32)
        matmul_35 = paddle._C_ops.matmul(slice_29, transpose_29, False, False)

        # pd_op.scale: (4x8x640x640xf32) <- (4x8x640x640xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(matmul_35, full_17, float("0"), True)
        del matmul_35

        # pd_op.softmax: (4x8x640x640xf32) <- (4x8x640x640xf32)
        softmax_3 = paddle._C_ops.softmax(scale_3, -1)
        del scale_3

        # pd_op.matmul: (4x8x640x32xf32) <- (4x8x640x640xf32, 4x8x640x32xf32)
        matmul_36 = paddle._C_ops.matmul(softmax_3, slice_31, False, False)

        # pd_op.transpose: (4x640x8x32xf32) <- (4x8x640x32xf32)
        transpose_30 = paddle._C_ops.transpose(matmul_36, [0, 2, 1, 3])
        del matmul_36

        # pd_op.reshape: (4x640x256xf32) <- (4x640x8x32xf32, 3xi64)
        reshape_27 = paddle._C_ops.reshape(transpose_30, full_int_array_8)
        del full_int_array_8

        # pd_op.matmul: (4x640x256xf32) <- (4x640x256xf32, 256x256xf32)
        matmul_37 = paddle._C_ops.matmul(reshape_27, parameter_85, False, False)
        del parameter_85

        # pd_op.add: (4x640x256xf32) <- (4x640x256xf32, 256xf32)
        add_82 = paddle._C_ops.add(matmul_37, parameter_84)
        del parameter_84

        # pd_op.full: (xf64) <- ()
        full_24 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__20 = paddle._C_ops.assign_value_(
            full_24,
            [],
            paddle.float64,
            [float("0.935294")],
            paddle.framework._current_expected_place(),
        )
        del full_24

        # pd_op.cast: (xf32) <- (xf64)
        cast_20 = paddle._C_ops.cast(assign_value__20, paddle.float32)
        del assign_value__20

        # pd_op.shape64: (3xi64) <- (4x640x256xf32)
        shape64_20 = paddle._C_ops.shape64(add_82)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(
            shape64_20, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_20

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_20 = [slice_32, full_1, full_1]
        del slice_32

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_20 = paddle._C_ops.stack(combine_20, 0)
        del combine_20

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_20 = paddle._C_ops.uniform(
            stack_20,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_20

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_83 = paddle._C_ops.add(cast_20, uniform_20)
        del uniform_20

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_20 = paddle._C_ops.floor(add_83)
        del add_83

        # pd_op.divide: (4x640x256xf32) <- (4x640x256xf32, xf32)
        divide_20 = paddle._C_ops.divide(add_82, cast_20)

        # pd_op.multiply: (4x640x256xf32) <- (4x640x256xf32, -1x1x1xf32)
        multiply_20 = paddle._C_ops.multiply(divide_20, floor_20)

        # pd_op.add: (4x640x256xf32) <- (4x640x256xf32, 4x640x256xf32)
        add_84 = paddle._C_ops.add(layer_norm_66, multiply_20)

        # pd_op.layer_norm: (4x640x256xf32, 4x640xf32, 4x640xf32) <- (4x640x256xf32, 256xf32, 256xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_84, parameter_83, parameter_82, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_82, parameter_83

        # pd_op.matmul: (4x640x1024xf32) <- (4x640x256xf32, 256x1024xf32)
        matmul_38 = paddle._C_ops.matmul(layer_norm_69, parameter_81, False, False)
        del parameter_81

        # pd_op.add: (4x640x1024xf32) <- (4x640x1024xf32, 1024xf32)
        add_85 = paddle._C_ops.add(matmul_38, parameter_80)
        del parameter_80

        # pd_op.gelu: (4x640x1024xf32) <- (4x640x1024xf32)
        gelu_13 = paddle._C_ops.gelu(add_85, False)

        # pd_op.matmul: (4x640x256xf32) <- (4x640x1024xf32, 1024x256xf32)
        matmul_39 = paddle._C_ops.matmul(gelu_13, parameter_79, False, False)
        del parameter_79

        # pd_op.add: (4x640x256xf32) <- (4x640x256xf32, 256xf32)
        add_86 = paddle._C_ops.add(matmul_39, parameter_78)
        del parameter_78

        # pd_op.full: (xf64) <- ()
        full_25 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__21 = paddle._C_ops.assign_value_(
            full_25,
            [],
            paddle.float64,
            [float("0.935294")],
            paddle.framework._current_expected_place(),
        )
        del full_25

        # pd_op.cast: (xf32) <- (xf64)
        cast_21 = paddle._C_ops.cast(assign_value__21, paddle.float32)
        del assign_value__21

        # pd_op.shape64: (3xi64) <- (4x640x256xf32)
        shape64_21 = paddle._C_ops.shape64(add_86)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_33 = paddle._C_ops.slice(
            shape64_21, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_21

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_21 = [slice_33, full_1, full_1]
        del slice_33

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_21 = paddle._C_ops.stack(combine_21, 0)
        del combine_21

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_21 = paddle._C_ops.uniform(
            stack_21,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_21

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_87 = paddle._C_ops.add(cast_21, uniform_21)
        del uniform_21

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_21 = paddle._C_ops.floor(add_87)
        del add_87

        # pd_op.divide: (4x640x256xf32) <- (4x640x256xf32, xf32)
        divide_21 = paddle._C_ops.divide(add_86, cast_21)

        # pd_op.multiply: (4x640x256xf32) <- (4x640x256xf32, -1x1x1xf32)
        multiply_21 = paddle._C_ops.multiply(divide_21, floor_21)

        # pd_op.add: (4x640x256xf32) <- (4x640x256xf32, 4x640x256xf32)
        add_88 = paddle._C_ops.add(layer_norm_69, multiply_21)

        # pd_op.layer_norm: (4x640x256xf32, 4x640xf32, 4x640xf32) <- (4x640x256xf32, 256xf32, 256xf32)
        layer_norm_72, layer_norm_73, layer_norm_74 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_88, parameter_77, parameter_76, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_76, parameter_77

        # pd_op.transpose: (4x256x640xf32) <- (4x640x256xf32)
        transpose_31 = paddle._C_ops.transpose(layer_norm_72, [0, 2, 1])
        del layer_norm_72

        # pd_op.reshape: (4x256x8x80xf32) <- (4x256x640xf32, 4xi64)
        reshape_28 = paddle._C_ops.reshape(transpose_31, full_int_array_4)
        del full_int_array_4

        # pd_op.conv2d: (4x384x4x80xf32) <- (4x256x8x80xf32, 384x256x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            reshape_28, parameter_75, [2, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_75

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_29 = paddle._C_ops.reshape(parameter_74, full_int_array_0)
        del full_int_array_0, parameter_74

        # pd_op.add: (4x384x4x80xf32) <- (4x384x4x80xf32, 1x384x1x1xf32)
        add_89 = paddle._C_ops.add(conv2d_11, reshape_29)

        # pd_op.flatten: (4x384x320xf32) <- (4x384x4x80xf32)
        flatten_10 = paddle._C_ops.flatten(add_89, 2, 3)

        # pd_op.transpose: (4x320x384xf32) <- (4x384x320xf32)
        transpose_32 = paddle._C_ops.transpose(flatten_10, [0, 2, 1])
        del flatten_10

        # pd_op.layer_norm: (4x320x384xf32, 4x320xf32, 4x320xf32) <- (4x320x384xf32, 384xf32, 384xf32)
        layer_norm_75, layer_norm_76, layer_norm_77 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_32, parameter_73, parameter_72, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_72, parameter_73

        # pd_op.matmul: (4x320x1152xf32) <- (4x320x384xf32, 384x1152xf32)
        matmul_40 = paddle._C_ops.matmul(layer_norm_75, parameter_71, False, False)
        del parameter_71

        # pd_op.add: (4x320x1152xf32) <- (4x320x1152xf32, 1152xf32)
        add_90 = paddle._C_ops.add(matmul_40, parameter_70)
        del parameter_70

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_9 = [0, -1, 3, 12, 32]

        # pd_op.reshape: (4x320x3x12x32xf32) <- (4x320x1152xf32, 5xi64)
        reshape_30 = paddle._C_ops.reshape(add_90, full_int_array_9)

        # pd_op.transpose: (3x4x12x320x32xf32) <- (4x320x3x12x32xf32)
        transpose_33 = paddle._C_ops.transpose(reshape_30, [2, 0, 3, 1, 4])
        del reshape_30

        # pd_op.slice: (4x12x320x32xf32) <- (3x4x12x320x32xf32, 1xi64, 1xi64)
        slice_34 = paddle._C_ops.slice(
            transpose_33, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (4x12x320x32xf32) <- (3x4x12x320x32xf32, 1xi64, 1xi64)
        slice_35 = paddle._C_ops.slice(
            transpose_33, [0], full_int_array_3, full_int_array_6, [1], [0]
        )

        # pd_op.slice: (4x12x320x32xf32) <- (3x4x12x320x32xf32, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(
            transpose_33, [0], full_int_array_6, full_int_array_7, [1], [0]
        )

        # pd_op.transpose: (4x12x32x320xf32) <- (4x12x320x32xf32)
        transpose_34 = paddle._C_ops.transpose(slice_35, [0, 1, 3, 2])
        del slice_35

        # pd_op.matmul: (4x12x320x320xf32) <- (4x12x320x32xf32, 4x12x32x320xf32)
        matmul_41 = paddle._C_ops.matmul(slice_34, transpose_34, False, False)

        # pd_op.scale: (4x12x320x320xf32) <- (4x12x320x320xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(matmul_41, full_17, float("0"), True)
        del matmul_41

        # pd_op.softmax: (4x12x320x320xf32) <- (4x12x320x320xf32)
        softmax_4 = paddle._C_ops.softmax(scale_4, -1)
        del scale_4

        # pd_op.matmul: (4x12x320x32xf32) <- (4x12x320x320xf32, 4x12x320x32xf32)
        matmul_42 = paddle._C_ops.matmul(softmax_4, slice_36, False, False)

        # pd_op.transpose: (4x320x12x32xf32) <- (4x12x320x32xf32)
        transpose_35 = paddle._C_ops.transpose(matmul_42, [0, 2, 1, 3])
        del matmul_42

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_10 = [0, -1, 384]

        # pd_op.reshape: (4x320x384xf32) <- (4x320x12x32xf32, 3xi64)
        reshape_31 = paddle._C_ops.reshape(transpose_35, full_int_array_10)

        # pd_op.matmul: (4x320x384xf32) <- (4x320x384xf32, 384x384xf32)
        matmul_43 = paddle._C_ops.matmul(reshape_31, parameter_69, False, False)
        del parameter_69

        # pd_op.add: (4x320x384xf32) <- (4x320x384xf32, 384xf32)
        add_91 = paddle._C_ops.add(matmul_43, parameter_68)
        del parameter_68

        # pd_op.full: (xf64) <- ()
        full_26 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__22 = paddle._C_ops.assign_value_(
            full_26,
            [],
            paddle.float64,
            [float("0.929412")],
            paddle.framework._current_expected_place(),
        )
        del full_26

        # pd_op.cast: (xf32) <- (xf64)
        cast_22 = paddle._C_ops.cast(assign_value__22, paddle.float32)
        del assign_value__22

        # pd_op.shape64: (3xi64) <- (4x320x384xf32)
        shape64_22 = paddle._C_ops.shape64(add_91)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_37 = paddle._C_ops.slice(
            shape64_22, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_22

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_22 = [slice_37, full_1, full_1]
        del slice_37

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_22 = paddle._C_ops.stack(combine_22, 0)
        del combine_22

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_22 = paddle._C_ops.uniform(
            stack_22,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_22

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_92 = paddle._C_ops.add(cast_22, uniform_22)
        del uniform_22

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_22 = paddle._C_ops.floor(add_92)
        del add_92

        # pd_op.divide: (4x320x384xf32) <- (4x320x384xf32, xf32)
        divide_22 = paddle._C_ops.divide(add_91, cast_22)

        # pd_op.multiply: (4x320x384xf32) <- (4x320x384xf32, -1x1x1xf32)
        multiply_22 = paddle._C_ops.multiply(divide_22, floor_22)

        # pd_op.add: (4x320x384xf32) <- (4x320x384xf32, 4x320x384xf32)
        add_93 = paddle._C_ops.add(layer_norm_75, multiply_22)

        # pd_op.layer_norm: (4x320x384xf32, 4x320xf32, 4x320xf32) <- (4x320x384xf32, 384xf32, 384xf32)
        layer_norm_78, layer_norm_79, layer_norm_80 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_93, parameter_67, parameter_66, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_66, parameter_67

        # pd_op.matmul: (4x320x1536xf32) <- (4x320x384xf32, 384x1536xf32)
        matmul_44 = paddle._C_ops.matmul(layer_norm_78, parameter_65, False, False)
        del parameter_65

        # pd_op.add: (4x320x1536xf32) <- (4x320x1536xf32, 1536xf32)
        add_94 = paddle._C_ops.add(matmul_44, parameter_64)
        del parameter_64

        # pd_op.gelu: (4x320x1536xf32) <- (4x320x1536xf32)
        gelu_14 = paddle._C_ops.gelu(add_94, False)

        # pd_op.matmul: (4x320x384xf32) <- (4x320x1536xf32, 1536x384xf32)
        matmul_45 = paddle._C_ops.matmul(gelu_14, parameter_63, False, False)
        del parameter_63

        # pd_op.add: (4x320x384xf32) <- (4x320x384xf32, 384xf32)
        add_95 = paddle._C_ops.add(matmul_45, parameter_62)
        del parameter_62

        # pd_op.full: (xf64) <- ()
        full_27 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__23 = paddle._C_ops.assign_value_(
            full_27,
            [],
            paddle.float64,
            [float("0.929412")],
            paddle.framework._current_expected_place(),
        )
        del full_27

        # pd_op.cast: (xf32) <- (xf64)
        cast_23 = paddle._C_ops.cast(assign_value__23, paddle.float32)
        del assign_value__23

        # pd_op.shape64: (3xi64) <- (4x320x384xf32)
        shape64_23 = paddle._C_ops.shape64(add_95)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_38 = paddle._C_ops.slice(
            shape64_23, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_23

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_23 = [slice_38, full_1, full_1]
        del slice_38

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_23 = paddle._C_ops.stack(combine_23, 0)
        del combine_23

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_23 = paddle._C_ops.uniform(
            stack_23,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_23

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_96 = paddle._C_ops.add(cast_23, uniform_23)
        del uniform_23

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_23 = paddle._C_ops.floor(add_96)
        del add_96

        # pd_op.divide: (4x320x384xf32) <- (4x320x384xf32, xf32)
        divide_23 = paddle._C_ops.divide(add_95, cast_23)

        # pd_op.multiply: (4x320x384xf32) <- (4x320x384xf32, -1x1x1xf32)
        multiply_23 = paddle._C_ops.multiply(divide_23, floor_23)

        # pd_op.add: (4x320x384xf32) <- (4x320x384xf32, 4x320x384xf32)
        add_97 = paddle._C_ops.add(layer_norm_78, multiply_23)

        # pd_op.layer_norm: (4x320x384xf32, 4x320xf32, 4x320xf32) <- (4x320x384xf32, 384xf32, 384xf32)
        layer_norm_81, layer_norm_82, layer_norm_83 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_97, parameter_61, parameter_60, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_60, parameter_61

        # pd_op.matmul: (4x320x1152xf32) <- (4x320x384xf32, 384x1152xf32)
        matmul_46 = paddle._C_ops.matmul(layer_norm_81, parameter_59, False, False)
        del parameter_59

        # pd_op.add: (4x320x1152xf32) <- (4x320x1152xf32, 1152xf32)
        add_98 = paddle._C_ops.add(matmul_46, parameter_58)
        del parameter_58

        # pd_op.reshape: (4x320x3x12x32xf32) <- (4x320x1152xf32, 5xi64)
        reshape_32 = paddle._C_ops.reshape(add_98, full_int_array_9)

        # pd_op.transpose: (3x4x12x320x32xf32) <- (4x320x3x12x32xf32)
        transpose_36 = paddle._C_ops.transpose(reshape_32, [2, 0, 3, 1, 4])
        del reshape_32

        # pd_op.slice: (4x12x320x32xf32) <- (3x4x12x320x32xf32, 1xi64, 1xi64)
        slice_39 = paddle._C_ops.slice(
            transpose_36, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (4x12x320x32xf32) <- (3x4x12x320x32xf32, 1xi64, 1xi64)
        slice_40 = paddle._C_ops.slice(
            transpose_36, [0], full_int_array_3, full_int_array_6, [1], [0]
        )

        # pd_op.slice: (4x12x320x32xf32) <- (3x4x12x320x32xf32, 1xi64, 1xi64)
        slice_41 = paddle._C_ops.slice(
            transpose_36, [0], full_int_array_6, full_int_array_7, [1], [0]
        )

        # pd_op.transpose: (4x12x32x320xf32) <- (4x12x320x32xf32)
        transpose_37 = paddle._C_ops.transpose(slice_40, [0, 1, 3, 2])
        del slice_40

        # pd_op.matmul: (4x12x320x320xf32) <- (4x12x320x32xf32, 4x12x32x320xf32)
        matmul_47 = paddle._C_ops.matmul(slice_39, transpose_37, False, False)

        # pd_op.scale: (4x12x320x320xf32) <- (4x12x320x320xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(matmul_47, full_17, float("0"), True)
        del matmul_47

        # pd_op.softmax: (4x12x320x320xf32) <- (4x12x320x320xf32)
        softmax_5 = paddle._C_ops.softmax(scale_5, -1)
        del scale_5

        # pd_op.matmul: (4x12x320x32xf32) <- (4x12x320x320xf32, 4x12x320x32xf32)
        matmul_48 = paddle._C_ops.matmul(softmax_5, slice_41, False, False)

        # pd_op.transpose: (4x320x12x32xf32) <- (4x12x320x32xf32)
        transpose_38 = paddle._C_ops.transpose(matmul_48, [0, 2, 1, 3])
        del matmul_48

        # pd_op.reshape: (4x320x384xf32) <- (4x320x12x32xf32, 3xi64)
        reshape_33 = paddle._C_ops.reshape(transpose_38, full_int_array_10)

        # pd_op.matmul: (4x320x384xf32) <- (4x320x384xf32, 384x384xf32)
        matmul_49 = paddle._C_ops.matmul(reshape_33, parameter_57, False, False)
        del parameter_57

        # pd_op.add: (4x320x384xf32) <- (4x320x384xf32, 384xf32)
        add_99 = paddle._C_ops.add(matmul_49, parameter_56)
        del parameter_56

        # pd_op.full: (xf64) <- ()
        full_28 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__24 = paddle._C_ops.assign_value_(
            full_28,
            [],
            paddle.float64,
            [float("0.923529")],
            paddle.framework._current_expected_place(),
        )
        del full_28

        # pd_op.cast: (xf32) <- (xf64)
        cast_24 = paddle._C_ops.cast(assign_value__24, paddle.float32)
        del assign_value__24

        # pd_op.shape64: (3xi64) <- (4x320x384xf32)
        shape64_24 = paddle._C_ops.shape64(add_99)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_42 = paddle._C_ops.slice(
            shape64_24, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_24

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_24 = [slice_42, full_1, full_1]
        del slice_42

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_24 = paddle._C_ops.stack(combine_24, 0)
        del combine_24

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_24 = paddle._C_ops.uniform(
            stack_24,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_24

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_100 = paddle._C_ops.add(cast_24, uniform_24)
        del uniform_24

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_24 = paddle._C_ops.floor(add_100)
        del add_100

        # pd_op.divide: (4x320x384xf32) <- (4x320x384xf32, xf32)
        divide_24 = paddle._C_ops.divide(add_99, cast_24)

        # pd_op.multiply: (4x320x384xf32) <- (4x320x384xf32, -1x1x1xf32)
        multiply_24 = paddle._C_ops.multiply(divide_24, floor_24)

        # pd_op.add: (4x320x384xf32) <- (4x320x384xf32, 4x320x384xf32)
        add_101 = paddle._C_ops.add(layer_norm_81, multiply_24)

        # pd_op.layer_norm: (4x320x384xf32, 4x320xf32, 4x320xf32) <- (4x320x384xf32, 384xf32, 384xf32)
        layer_norm_84, layer_norm_85, layer_norm_86 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_101, parameter_55, parameter_54, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_54, parameter_55

        # pd_op.matmul: (4x320x1536xf32) <- (4x320x384xf32, 384x1536xf32)
        matmul_50 = paddle._C_ops.matmul(layer_norm_84, parameter_53, False, False)
        del parameter_53

        # pd_op.add: (4x320x1536xf32) <- (4x320x1536xf32, 1536xf32)
        add_102 = paddle._C_ops.add(matmul_50, parameter_52)
        del parameter_52

        # pd_op.gelu: (4x320x1536xf32) <- (4x320x1536xf32)
        gelu_15 = paddle._C_ops.gelu(add_102, False)

        # pd_op.matmul: (4x320x384xf32) <- (4x320x1536xf32, 1536x384xf32)
        matmul_51 = paddle._C_ops.matmul(gelu_15, parameter_51, False, False)
        del parameter_51

        # pd_op.add: (4x320x384xf32) <- (4x320x384xf32, 384xf32)
        add_103 = paddle._C_ops.add(matmul_51, parameter_50)
        del parameter_50

        # pd_op.full: (xf64) <- ()
        full_29 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__25 = paddle._C_ops.assign_value_(
            full_29,
            [],
            paddle.float64,
            [float("0.923529")],
            paddle.framework._current_expected_place(),
        )
        del full_29

        # pd_op.cast: (xf32) <- (xf64)
        cast_25 = paddle._C_ops.cast(assign_value__25, paddle.float32)
        del assign_value__25

        # pd_op.shape64: (3xi64) <- (4x320x384xf32)
        shape64_25 = paddle._C_ops.shape64(add_103)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_43 = paddle._C_ops.slice(
            shape64_25, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_25

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_25 = [slice_43, full_1, full_1]
        del slice_43

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_25 = paddle._C_ops.stack(combine_25, 0)
        del combine_25

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_25 = paddle._C_ops.uniform(
            stack_25,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_25

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_104 = paddle._C_ops.add(cast_25, uniform_25)
        del uniform_25

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_25 = paddle._C_ops.floor(add_104)
        del add_104

        # pd_op.divide: (4x320x384xf32) <- (4x320x384xf32, xf32)
        divide_25 = paddle._C_ops.divide(add_103, cast_25)

        # pd_op.multiply: (4x320x384xf32) <- (4x320x384xf32, -1x1x1xf32)
        multiply_25 = paddle._C_ops.multiply(divide_25, floor_25)

        # pd_op.add: (4x320x384xf32) <- (4x320x384xf32, 4x320x384xf32)
        add_105 = paddle._C_ops.add(layer_norm_84, multiply_25)

        # pd_op.layer_norm: (4x320x384xf32, 4x320xf32, 4x320xf32) <- (4x320x384xf32, 384xf32, 384xf32)
        layer_norm_87, layer_norm_88, layer_norm_89 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_105, parameter_49, parameter_48, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_48, parameter_49

        # pd_op.matmul: (4x320x1152xf32) <- (4x320x384xf32, 384x1152xf32)
        matmul_52 = paddle._C_ops.matmul(layer_norm_87, parameter_47, False, False)
        del parameter_47

        # pd_op.add: (4x320x1152xf32) <- (4x320x1152xf32, 1152xf32)
        add_106 = paddle._C_ops.add(matmul_52, parameter_46)
        del parameter_46

        # pd_op.reshape: (4x320x3x12x32xf32) <- (4x320x1152xf32, 5xi64)
        reshape_34 = paddle._C_ops.reshape(add_106, full_int_array_9)

        # pd_op.transpose: (3x4x12x320x32xf32) <- (4x320x3x12x32xf32)
        transpose_39 = paddle._C_ops.transpose(reshape_34, [2, 0, 3, 1, 4])
        del reshape_34

        # pd_op.slice: (4x12x320x32xf32) <- (3x4x12x320x32xf32, 1xi64, 1xi64)
        slice_44 = paddle._C_ops.slice(
            transpose_39, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (4x12x320x32xf32) <- (3x4x12x320x32xf32, 1xi64, 1xi64)
        slice_45 = paddle._C_ops.slice(
            transpose_39, [0], full_int_array_3, full_int_array_6, [1], [0]
        )

        # pd_op.slice: (4x12x320x32xf32) <- (3x4x12x320x32xf32, 1xi64, 1xi64)
        slice_46 = paddle._C_ops.slice(
            transpose_39, [0], full_int_array_6, full_int_array_7, [1], [0]
        )

        # pd_op.transpose: (4x12x32x320xf32) <- (4x12x320x32xf32)
        transpose_40 = paddle._C_ops.transpose(slice_45, [0, 1, 3, 2])
        del slice_45

        # pd_op.matmul: (4x12x320x320xf32) <- (4x12x320x32xf32, 4x12x32x320xf32)
        matmul_53 = paddle._C_ops.matmul(slice_44, transpose_40, False, False)

        # pd_op.scale: (4x12x320x320xf32) <- (4x12x320x320xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(matmul_53, full_17, float("0"), True)
        del matmul_53

        # pd_op.softmax: (4x12x320x320xf32) <- (4x12x320x320xf32)
        softmax_6 = paddle._C_ops.softmax(scale_6, -1)
        del scale_6

        # pd_op.matmul: (4x12x320x32xf32) <- (4x12x320x320xf32, 4x12x320x32xf32)
        matmul_54 = paddle._C_ops.matmul(softmax_6, slice_46, False, False)

        # pd_op.transpose: (4x320x12x32xf32) <- (4x12x320x32xf32)
        transpose_41 = paddle._C_ops.transpose(matmul_54, [0, 2, 1, 3])
        del matmul_54

        # pd_op.reshape: (4x320x384xf32) <- (4x320x12x32xf32, 3xi64)
        reshape_35 = paddle._C_ops.reshape(transpose_41, full_int_array_10)

        # pd_op.matmul: (4x320x384xf32) <- (4x320x384xf32, 384x384xf32)
        matmul_55 = paddle._C_ops.matmul(reshape_35, parameter_45, False, False)
        del parameter_45

        # pd_op.add: (4x320x384xf32) <- (4x320x384xf32, 384xf32)
        add_107 = paddle._C_ops.add(matmul_55, parameter_44)
        del parameter_44

        # pd_op.full: (xf64) <- ()
        full_30 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__26 = paddle._C_ops.assign_value_(
            full_30,
            [],
            paddle.float64,
            [float("0.917647")],
            paddle.framework._current_expected_place(),
        )
        del full_30

        # pd_op.cast: (xf32) <- (xf64)
        cast_26 = paddle._C_ops.cast(assign_value__26, paddle.float32)
        del assign_value__26

        # pd_op.shape64: (3xi64) <- (4x320x384xf32)
        shape64_26 = paddle._C_ops.shape64(add_107)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_47 = paddle._C_ops.slice(
            shape64_26, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_26

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_26 = [slice_47, full_1, full_1]
        del slice_47

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_26 = paddle._C_ops.stack(combine_26, 0)
        del combine_26

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_26 = paddle._C_ops.uniform(
            stack_26,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_26

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_108 = paddle._C_ops.add(cast_26, uniform_26)
        del uniform_26

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_26 = paddle._C_ops.floor(add_108)
        del add_108

        # pd_op.divide: (4x320x384xf32) <- (4x320x384xf32, xf32)
        divide_26 = paddle._C_ops.divide(add_107, cast_26)

        # pd_op.multiply: (4x320x384xf32) <- (4x320x384xf32, -1x1x1xf32)
        multiply_26 = paddle._C_ops.multiply(divide_26, floor_26)

        # pd_op.add: (4x320x384xf32) <- (4x320x384xf32, 4x320x384xf32)
        add_109 = paddle._C_ops.add(layer_norm_87, multiply_26)

        # pd_op.layer_norm: (4x320x384xf32, 4x320xf32, 4x320xf32) <- (4x320x384xf32, 384xf32, 384xf32)
        layer_norm_90, layer_norm_91, layer_norm_92 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_109, parameter_43, parameter_42, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_42, parameter_43

        # pd_op.matmul: (4x320x1536xf32) <- (4x320x384xf32, 384x1536xf32)
        matmul_56 = paddle._C_ops.matmul(layer_norm_90, parameter_41, False, False)
        del parameter_41

        # pd_op.add: (4x320x1536xf32) <- (4x320x1536xf32, 1536xf32)
        add_110 = paddle._C_ops.add(matmul_56, parameter_40)
        del parameter_40

        # pd_op.gelu: (4x320x1536xf32) <- (4x320x1536xf32)
        gelu_16 = paddle._C_ops.gelu(add_110, False)

        # pd_op.matmul: (4x320x384xf32) <- (4x320x1536xf32, 1536x384xf32)
        matmul_57 = paddle._C_ops.matmul(gelu_16, parameter_39, False, False)
        del parameter_39

        # pd_op.add: (4x320x384xf32) <- (4x320x384xf32, 384xf32)
        add_111 = paddle._C_ops.add(matmul_57, parameter_38)
        del parameter_38

        # pd_op.full: (xf64) <- ()
        full_31 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__27 = paddle._C_ops.assign_value_(
            full_31,
            [],
            paddle.float64,
            [float("0.917647")],
            paddle.framework._current_expected_place(),
        )
        del full_31

        # pd_op.cast: (xf32) <- (xf64)
        cast_27 = paddle._C_ops.cast(assign_value__27, paddle.float32)
        del assign_value__27

        # pd_op.shape64: (3xi64) <- (4x320x384xf32)
        shape64_27 = paddle._C_ops.shape64(add_111)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_48 = paddle._C_ops.slice(
            shape64_27, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_27

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_27 = [slice_48, full_1, full_1]
        del slice_48

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_27 = paddle._C_ops.stack(combine_27, 0)
        del combine_27

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_27 = paddle._C_ops.uniform(
            stack_27,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_27

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_112 = paddle._C_ops.add(cast_27, uniform_27)
        del uniform_27

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_27 = paddle._C_ops.floor(add_112)
        del add_112

        # pd_op.divide: (4x320x384xf32) <- (4x320x384xf32, xf32)
        divide_27 = paddle._C_ops.divide(add_111, cast_27)

        # pd_op.multiply: (4x320x384xf32) <- (4x320x384xf32, -1x1x1xf32)
        multiply_27 = paddle._C_ops.multiply(divide_27, floor_27)

        # pd_op.add: (4x320x384xf32) <- (4x320x384xf32, 4x320x384xf32)
        add_113 = paddle._C_ops.add(layer_norm_90, multiply_27)

        # pd_op.layer_norm: (4x320x384xf32, 4x320xf32, 4x320xf32) <- (4x320x384xf32, 384xf32, 384xf32)
        layer_norm_93, layer_norm_94, layer_norm_95 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_113, parameter_37, parameter_36, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_36, parameter_37

        # pd_op.matmul: (4x320x1152xf32) <- (4x320x384xf32, 384x1152xf32)
        matmul_58 = paddle._C_ops.matmul(layer_norm_93, parameter_35, False, False)
        del parameter_35

        # pd_op.add: (4x320x1152xf32) <- (4x320x1152xf32, 1152xf32)
        add_114 = paddle._C_ops.add(matmul_58, parameter_34)
        del parameter_34

        # pd_op.reshape: (4x320x3x12x32xf32) <- (4x320x1152xf32, 5xi64)
        reshape_36 = paddle._C_ops.reshape(add_114, full_int_array_9)

        # pd_op.transpose: (3x4x12x320x32xf32) <- (4x320x3x12x32xf32)
        transpose_42 = paddle._C_ops.transpose(reshape_36, [2, 0, 3, 1, 4])
        del reshape_36

        # pd_op.slice: (4x12x320x32xf32) <- (3x4x12x320x32xf32, 1xi64, 1xi64)
        slice_49 = paddle._C_ops.slice(
            transpose_42, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (4x12x320x32xf32) <- (3x4x12x320x32xf32, 1xi64, 1xi64)
        slice_50 = paddle._C_ops.slice(
            transpose_42, [0], full_int_array_3, full_int_array_6, [1], [0]
        )

        # pd_op.slice: (4x12x320x32xf32) <- (3x4x12x320x32xf32, 1xi64, 1xi64)
        slice_51 = paddle._C_ops.slice(
            transpose_42, [0], full_int_array_6, full_int_array_7, [1], [0]
        )

        # pd_op.transpose: (4x12x32x320xf32) <- (4x12x320x32xf32)
        transpose_43 = paddle._C_ops.transpose(slice_50, [0, 1, 3, 2])
        del slice_50

        # pd_op.matmul: (4x12x320x320xf32) <- (4x12x320x32xf32, 4x12x32x320xf32)
        matmul_59 = paddle._C_ops.matmul(slice_49, transpose_43, False, False)

        # pd_op.scale: (4x12x320x320xf32) <- (4x12x320x320xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(matmul_59, full_17, float("0"), True)
        del matmul_59

        # pd_op.softmax: (4x12x320x320xf32) <- (4x12x320x320xf32)
        softmax_7 = paddle._C_ops.softmax(scale_7, -1)
        del scale_7

        # pd_op.matmul: (4x12x320x32xf32) <- (4x12x320x320xf32, 4x12x320x32xf32)
        matmul_60 = paddle._C_ops.matmul(softmax_7, slice_51, False, False)

        # pd_op.transpose: (4x320x12x32xf32) <- (4x12x320x32xf32)
        transpose_44 = paddle._C_ops.transpose(matmul_60, [0, 2, 1, 3])
        del matmul_60

        # pd_op.reshape: (4x320x384xf32) <- (4x320x12x32xf32, 3xi64)
        reshape_37 = paddle._C_ops.reshape(transpose_44, full_int_array_10)

        # pd_op.matmul: (4x320x384xf32) <- (4x320x384xf32, 384x384xf32)
        matmul_61 = paddle._C_ops.matmul(reshape_37, parameter_33, False, False)
        del parameter_33

        # pd_op.add: (4x320x384xf32) <- (4x320x384xf32, 384xf32)
        add_115 = paddle._C_ops.add(matmul_61, parameter_32)
        del parameter_32

        # pd_op.full: (xf64) <- ()
        full_32 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__28 = paddle._C_ops.assign_value_(
            full_32,
            [],
            paddle.float64,
            [float("0.911765")],
            paddle.framework._current_expected_place(),
        )
        del full_32

        # pd_op.cast: (xf32) <- (xf64)
        cast_28 = paddle._C_ops.cast(assign_value__28, paddle.float32)
        del assign_value__28

        # pd_op.shape64: (3xi64) <- (4x320x384xf32)
        shape64_28 = paddle._C_ops.shape64(add_115)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_52 = paddle._C_ops.slice(
            shape64_28, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_28

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_28 = [slice_52, full_1, full_1]
        del slice_52

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_28 = paddle._C_ops.stack(combine_28, 0)
        del combine_28

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_28 = paddle._C_ops.uniform(
            stack_28,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_28

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_116 = paddle._C_ops.add(cast_28, uniform_28)
        del uniform_28

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_28 = paddle._C_ops.floor(add_116)
        del add_116

        # pd_op.divide: (4x320x384xf32) <- (4x320x384xf32, xf32)
        divide_28 = paddle._C_ops.divide(add_115, cast_28)

        # pd_op.multiply: (4x320x384xf32) <- (4x320x384xf32, -1x1x1xf32)
        multiply_28 = paddle._C_ops.multiply(divide_28, floor_28)

        # pd_op.add: (4x320x384xf32) <- (4x320x384xf32, 4x320x384xf32)
        add_117 = paddle._C_ops.add(layer_norm_93, multiply_28)

        # pd_op.layer_norm: (4x320x384xf32, 4x320xf32, 4x320xf32) <- (4x320x384xf32, 384xf32, 384xf32)
        layer_norm_96, layer_norm_97, layer_norm_98 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_117, parameter_31, parameter_30, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_30, parameter_31

        # pd_op.matmul: (4x320x1536xf32) <- (4x320x384xf32, 384x1536xf32)
        matmul_62 = paddle._C_ops.matmul(layer_norm_96, parameter_29, False, False)
        del parameter_29

        # pd_op.add: (4x320x1536xf32) <- (4x320x1536xf32, 1536xf32)
        add_118 = paddle._C_ops.add(matmul_62, parameter_28)
        del parameter_28

        # pd_op.gelu: (4x320x1536xf32) <- (4x320x1536xf32)
        gelu_17 = paddle._C_ops.gelu(add_118, False)

        # pd_op.matmul: (4x320x384xf32) <- (4x320x1536xf32, 1536x384xf32)
        matmul_63 = paddle._C_ops.matmul(gelu_17, parameter_27, False, False)
        del parameter_27

        # pd_op.add: (4x320x384xf32) <- (4x320x384xf32, 384xf32)
        add_119 = paddle._C_ops.add(matmul_63, parameter_26)
        del parameter_26

        # pd_op.full: (xf64) <- ()
        full_33 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__29 = paddle._C_ops.assign_value_(
            full_33,
            [],
            paddle.float64,
            [float("0.911765")],
            paddle.framework._current_expected_place(),
        )
        del full_33

        # pd_op.cast: (xf32) <- (xf64)
        cast_29 = paddle._C_ops.cast(assign_value__29, paddle.float32)
        del assign_value__29

        # pd_op.shape64: (3xi64) <- (4x320x384xf32)
        shape64_29 = paddle._C_ops.shape64(add_119)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_53 = paddle._C_ops.slice(
            shape64_29, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_29

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_29 = [slice_53, full_1, full_1]
        del slice_53

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_29 = paddle._C_ops.stack(combine_29, 0)
        del combine_29

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_29 = paddle._C_ops.uniform(
            stack_29,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_29

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_120 = paddle._C_ops.add(cast_29, uniform_29)
        del uniform_29

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_29 = paddle._C_ops.floor(add_120)
        del add_120

        # pd_op.divide: (4x320x384xf32) <- (4x320x384xf32, xf32)
        divide_29 = paddle._C_ops.divide(add_119, cast_29)

        # pd_op.multiply: (4x320x384xf32) <- (4x320x384xf32, -1x1x1xf32)
        multiply_29 = paddle._C_ops.multiply(divide_29, floor_29)

        # pd_op.add: (4x320x384xf32) <- (4x320x384xf32, 4x320x384xf32)
        add_121 = paddle._C_ops.add(layer_norm_96, multiply_29)

        # pd_op.layer_norm: (4x320x384xf32, 4x320xf32, 4x320xf32) <- (4x320x384xf32, 384xf32, 384xf32)
        layer_norm_99, layer_norm_100, layer_norm_101 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_121, parameter_25, parameter_24, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_24, parameter_25

        # pd_op.matmul: (4x320x1152xf32) <- (4x320x384xf32, 384x1152xf32)
        matmul_64 = paddle._C_ops.matmul(layer_norm_99, parameter_23, False, False)
        del parameter_23

        # pd_op.add: (4x320x1152xf32) <- (4x320x1152xf32, 1152xf32)
        add_122 = paddle._C_ops.add(matmul_64, parameter_22)
        del parameter_22

        # pd_op.reshape: (4x320x3x12x32xf32) <- (4x320x1152xf32, 5xi64)
        reshape_38 = paddle._C_ops.reshape(add_122, full_int_array_9)

        # pd_op.transpose: (3x4x12x320x32xf32) <- (4x320x3x12x32xf32)
        transpose_45 = paddle._C_ops.transpose(reshape_38, [2, 0, 3, 1, 4])
        del reshape_38

        # pd_op.slice: (4x12x320x32xf32) <- (3x4x12x320x32xf32, 1xi64, 1xi64)
        slice_54 = paddle._C_ops.slice(
            transpose_45, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (4x12x320x32xf32) <- (3x4x12x320x32xf32, 1xi64, 1xi64)
        slice_55 = paddle._C_ops.slice(
            transpose_45, [0], full_int_array_3, full_int_array_6, [1], [0]
        )

        # pd_op.slice: (4x12x320x32xf32) <- (3x4x12x320x32xf32, 1xi64, 1xi64)
        slice_56 = paddle._C_ops.slice(
            transpose_45, [0], full_int_array_6, full_int_array_7, [1], [0]
        )

        # pd_op.transpose: (4x12x32x320xf32) <- (4x12x320x32xf32)
        transpose_46 = paddle._C_ops.transpose(slice_55, [0, 1, 3, 2])
        del slice_55

        # pd_op.matmul: (4x12x320x320xf32) <- (4x12x320x32xf32, 4x12x32x320xf32)
        matmul_65 = paddle._C_ops.matmul(slice_54, transpose_46, False, False)

        # pd_op.scale: (4x12x320x320xf32) <- (4x12x320x320xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(matmul_65, full_17, float("0"), True)
        del matmul_65

        # pd_op.softmax: (4x12x320x320xf32) <- (4x12x320x320xf32)
        softmax_8 = paddle._C_ops.softmax(scale_8, -1)
        del scale_8

        # pd_op.matmul: (4x12x320x32xf32) <- (4x12x320x320xf32, 4x12x320x32xf32)
        matmul_66 = paddle._C_ops.matmul(softmax_8, slice_56, False, False)

        # pd_op.transpose: (4x320x12x32xf32) <- (4x12x320x32xf32)
        transpose_47 = paddle._C_ops.transpose(matmul_66, [0, 2, 1, 3])
        del matmul_66

        # pd_op.reshape: (4x320x384xf32) <- (4x320x12x32xf32, 3xi64)
        reshape_39 = paddle._C_ops.reshape(transpose_47, full_int_array_10)

        # pd_op.matmul: (4x320x384xf32) <- (4x320x384xf32, 384x384xf32)
        matmul_67 = paddle._C_ops.matmul(reshape_39, parameter_21, False, False)
        del parameter_21

        # pd_op.add: (4x320x384xf32) <- (4x320x384xf32, 384xf32)
        add_123 = paddle._C_ops.add(matmul_67, parameter_20)
        del parameter_20

        # pd_op.full: (xf64) <- ()
        full_34 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__30 = paddle._C_ops.assign_value_(
            full_34,
            [],
            paddle.float64,
            [float("0.905882")],
            paddle.framework._current_expected_place(),
        )
        del full_34

        # pd_op.cast: (xf32) <- (xf64)
        cast_30 = paddle._C_ops.cast(assign_value__30, paddle.float32)
        del assign_value__30

        # pd_op.shape64: (3xi64) <- (4x320x384xf32)
        shape64_30 = paddle._C_ops.shape64(add_123)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_57 = paddle._C_ops.slice(
            shape64_30, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_30

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_30 = [slice_57, full_1, full_1]
        del slice_57

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_30 = paddle._C_ops.stack(combine_30, 0)
        del combine_30

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_30 = paddle._C_ops.uniform(
            stack_30,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_30

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_124 = paddle._C_ops.add(cast_30, uniform_30)
        del uniform_30

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_30 = paddle._C_ops.floor(add_124)
        del add_124

        # pd_op.divide: (4x320x384xf32) <- (4x320x384xf32, xf32)
        divide_30 = paddle._C_ops.divide(add_123, cast_30)

        # pd_op.multiply: (4x320x384xf32) <- (4x320x384xf32, -1x1x1xf32)
        multiply_30 = paddle._C_ops.multiply(divide_30, floor_30)

        # pd_op.add: (4x320x384xf32) <- (4x320x384xf32, 4x320x384xf32)
        add_125 = paddle._C_ops.add(layer_norm_99, multiply_30)

        # pd_op.layer_norm: (4x320x384xf32, 4x320xf32, 4x320xf32) <- (4x320x384xf32, 384xf32, 384xf32)
        layer_norm_102, layer_norm_103, layer_norm_104 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_125, parameter_19, parameter_18, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_18, parameter_19

        # pd_op.matmul: (4x320x1536xf32) <- (4x320x384xf32, 384x1536xf32)
        matmul_68 = paddle._C_ops.matmul(layer_norm_102, parameter_17, False, False)
        del parameter_17

        # pd_op.add: (4x320x1536xf32) <- (4x320x1536xf32, 1536xf32)
        add_126 = paddle._C_ops.add(matmul_68, parameter_16)
        del parameter_16

        # pd_op.gelu: (4x320x1536xf32) <- (4x320x1536xf32)
        gelu_18 = paddle._C_ops.gelu(add_126, False)

        # pd_op.matmul: (4x320x384xf32) <- (4x320x1536xf32, 1536x384xf32)
        matmul_69 = paddle._C_ops.matmul(gelu_18, parameter_15, False, False)
        del parameter_15

        # pd_op.add: (4x320x384xf32) <- (4x320x384xf32, 384xf32)
        add_127 = paddle._C_ops.add(matmul_69, parameter_14)
        del parameter_14

        # pd_op.full: (xf64) <- ()
        full_35 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__31 = paddle._C_ops.assign_value_(
            full_35,
            [],
            paddle.float64,
            [float("0.905882")],
            paddle.framework._current_expected_place(),
        )
        del full_35

        # pd_op.cast: (xf32) <- (xf64)
        cast_31 = paddle._C_ops.cast(assign_value__31, paddle.float32)
        del assign_value__31

        # pd_op.shape64: (3xi64) <- (4x320x384xf32)
        shape64_31 = paddle._C_ops.shape64(add_127)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_58 = paddle._C_ops.slice(
            shape64_31, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_31

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_31 = [slice_58, full_1, full_1]
        del slice_58

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_31 = paddle._C_ops.stack(combine_31, 0)
        del combine_31

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_31 = paddle._C_ops.uniform(
            stack_31,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_31

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_128 = paddle._C_ops.add(cast_31, uniform_31)
        del uniform_31

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_31 = paddle._C_ops.floor(add_128)
        del add_128

        # pd_op.divide: (4x320x384xf32) <- (4x320x384xf32, xf32)
        divide_31 = paddle._C_ops.divide(add_127, cast_31)

        # pd_op.multiply: (4x320x384xf32) <- (4x320x384xf32, -1x1x1xf32)
        multiply_31 = paddle._C_ops.multiply(divide_31, floor_31)

        # pd_op.add: (4x320x384xf32) <- (4x320x384xf32, 4x320x384xf32)
        add_129 = paddle._C_ops.add(layer_norm_102, multiply_31)

        # pd_op.layer_norm: (4x320x384xf32, 4x320xf32, 4x320xf32) <- (4x320x384xf32, 384xf32, 384xf32)
        layer_norm_105, layer_norm_106, layer_norm_107 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_129, parameter_13, parameter_12, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_12, parameter_13

        # pd_op.matmul: (4x320x1152xf32) <- (4x320x384xf32, 384x1152xf32)
        matmul_70 = paddle._C_ops.matmul(layer_norm_105, parameter_11, False, False)
        del parameter_11

        # pd_op.add: (4x320x1152xf32) <- (4x320x1152xf32, 1152xf32)
        add_130 = paddle._C_ops.add(matmul_70, parameter_10)
        del parameter_10

        # pd_op.reshape: (4x320x3x12x32xf32) <- (4x320x1152xf32, 5xi64)
        reshape_40 = paddle._C_ops.reshape(add_130, full_int_array_9)
        del full_int_array_9

        # pd_op.transpose: (3x4x12x320x32xf32) <- (4x320x3x12x32xf32)
        transpose_48 = paddle._C_ops.transpose(reshape_40, [2, 0, 3, 1, 4])
        del reshape_40

        # pd_op.slice: (4x12x320x32xf32) <- (3x4x12x320x32xf32, 1xi64, 1xi64)
        slice_59 = paddle._C_ops.slice(
            transpose_48, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (4x12x320x32xf32) <- (3x4x12x320x32xf32, 1xi64, 1xi64)
        slice_60 = paddle._C_ops.slice(
            transpose_48, [0], full_int_array_3, full_int_array_6, [1], [0]
        )

        # pd_op.slice: (4x12x320x32xf32) <- (3x4x12x320x32xf32, 1xi64, 1xi64)
        slice_61 = paddle._C_ops.slice(
            transpose_48, [0], full_int_array_6, full_int_array_7, [1], [0]
        )

        # pd_op.transpose: (4x12x32x320xf32) <- (4x12x320x32xf32)
        transpose_49 = paddle._C_ops.transpose(slice_60, [0, 1, 3, 2])
        del slice_60

        # pd_op.matmul: (4x12x320x320xf32) <- (4x12x320x32xf32, 4x12x32x320xf32)
        matmul_71 = paddle._C_ops.matmul(slice_59, transpose_49, False, False)

        # pd_op.scale: (4x12x320x320xf32) <- (4x12x320x320xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(matmul_71, full_17, float("0"), True)
        del matmul_71

        # pd_op.softmax: (4x12x320x320xf32) <- (4x12x320x320xf32)
        softmax_9 = paddle._C_ops.softmax(scale_9, -1)
        del scale_9

        # pd_op.matmul: (4x12x320x32xf32) <- (4x12x320x320xf32, 4x12x320x32xf32)
        matmul_72 = paddle._C_ops.matmul(softmax_9, slice_61, False, False)

        # pd_op.transpose: (4x320x12x32xf32) <- (4x12x320x32xf32)
        transpose_50 = paddle._C_ops.transpose(matmul_72, [0, 2, 1, 3])
        del matmul_72

        # pd_op.reshape: (4x320x384xf32) <- (4x320x12x32xf32, 3xi64)
        reshape_41 = paddle._C_ops.reshape(transpose_50, full_int_array_10)
        del full_int_array_10

        # pd_op.matmul: (4x320x384xf32) <- (4x320x384xf32, 384x384xf32)
        matmul_73 = paddle._C_ops.matmul(reshape_41, parameter_9, False, False)
        del parameter_9

        # pd_op.add: (4x320x384xf32) <- (4x320x384xf32, 384xf32)
        add_131 = paddle._C_ops.add(matmul_73, parameter_8)
        del parameter_8

        # pd_op.full: (xf64) <- ()
        full_36 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__32 = paddle._C_ops.assign_value_(
            full_36,
            [],
            paddle.float64,
            [float("0.9")],
            paddle.framework._current_expected_place(),
        )
        del full_36

        # pd_op.cast: (xf32) <- (xf64)
        cast_32 = paddle._C_ops.cast(assign_value__32, paddle.float32)
        del assign_value__32

        # pd_op.shape64: (3xi64) <- (4x320x384xf32)
        shape64_32 = paddle._C_ops.shape64(add_131)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_62 = paddle._C_ops.slice(
            shape64_32, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_32

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_32 = [slice_62, full_1, full_1]
        del slice_62

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_32 = paddle._C_ops.stack(combine_32, 0)
        del combine_32

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_32 = paddle._C_ops.uniform(
            stack_32,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_32

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_132 = paddle._C_ops.add(cast_32, uniform_32)
        del uniform_32

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_32 = paddle._C_ops.floor(add_132)
        del add_132

        # pd_op.divide: (4x320x384xf32) <- (4x320x384xf32, xf32)
        divide_32 = paddle._C_ops.divide(add_131, cast_32)

        # pd_op.multiply: (4x320x384xf32) <- (4x320x384xf32, -1x1x1xf32)
        multiply_32 = paddle._C_ops.multiply(divide_32, floor_32)

        # pd_op.add: (4x320x384xf32) <- (4x320x384xf32, 4x320x384xf32)
        add_133 = paddle._C_ops.add(layer_norm_105, multiply_32)

        # pd_op.layer_norm: (4x320x384xf32, 4x320xf32, 4x320xf32) <- (4x320x384xf32, 384xf32, 384xf32)
        layer_norm_108, layer_norm_109, layer_norm_110 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_133, parameter_7, parameter_6, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_6, parameter_7

        # pd_op.matmul: (4x320x1536xf32) <- (4x320x384xf32, 384x1536xf32)
        matmul_74 = paddle._C_ops.matmul(layer_norm_108, parameter_5, False, False)
        del parameter_5

        # pd_op.add: (4x320x1536xf32) <- (4x320x1536xf32, 1536xf32)
        add_134 = paddle._C_ops.add(matmul_74, parameter_4)
        del parameter_4

        # pd_op.gelu: (4x320x1536xf32) <- (4x320x1536xf32)
        gelu_19 = paddle._C_ops.gelu(add_134, False)

        # pd_op.matmul: (4x320x384xf32) <- (4x320x1536xf32, 1536x384xf32)
        matmul_75 = paddle._C_ops.matmul(gelu_19, parameter_3, False, False)
        del parameter_3

        # pd_op.add: (4x320x384xf32) <- (4x320x384xf32, 384xf32)
        add_135 = paddle._C_ops.add(matmul_75, parameter_2)
        del parameter_2

        # pd_op.full: (xf64) <- ()
        full_37 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__33 = paddle._C_ops.assign_value_(
            full_37,
            [],
            paddle.float64,
            [float("0.9")],
            paddle.framework._current_expected_place(),
        )
        del full_37

        # pd_op.cast: (xf32) <- (xf64)
        cast_33 = paddle._C_ops.cast(assign_value__33, paddle.float32)
        del assign_value__33

        # pd_op.shape64: (3xi64) <- (4x320x384xf32)
        shape64_33 = paddle._C_ops.shape64(add_135)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_63 = paddle._C_ops.slice(
            shape64_33, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del full_int_array_2, full_int_array_3, shape64_33

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_33 = [slice_63, full_1, full_1]
        del full_1, slice_63

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_33 = paddle._C_ops.stack(combine_33, 0)
        del combine_33

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_33 = paddle._C_ops.uniform(
            stack_33,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del full_2, full_3, stack_33

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_136 = paddle._C_ops.add(cast_33, uniform_33)
        del uniform_33

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_33 = paddle._C_ops.floor(add_136)
        del add_136

        # pd_op.divide: (4x320x384xf32) <- (4x320x384xf32, xf32)
        divide_33 = paddle._C_ops.divide(add_135, cast_33)

        # pd_op.multiply: (4x320x384xf32) <- (4x320x384xf32, -1x1x1xf32)
        multiply_33 = paddle._C_ops.multiply(divide_33, floor_33)

        # pd_op.add: (4x320x384xf32) <- (4x320x384xf32, 4x320x384xf32)
        add_137 = paddle._C_ops.add(layer_norm_108, multiply_33)

        # pd_op.layer_norm: (4x320x384xf32, 4x320xf32, 4x320xf32) <- (4x320x384xf32, 384xf32, 384xf32)
        layer_norm_111, layer_norm_112, layer_norm_113 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_137, parameter_1, parameter_0, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_0, parameter_1

        # pd_op.transpose: (4x384x320xf32) <- (4x320x384xf32)
        transpose_51 = paddle._C_ops.transpose(layer_norm_111, [0, 2, 1])
        del layer_norm_111

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_11 = [0, 384, 4, 80]

        # pd_op.reshape: (4x384x4x80xf32) <- (4x384x320xf32, 4xi64)
        reshape_42 = paddle._C_ops.reshape(transpose_51, full_int_array_11)
        del full_int_array_11

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_12 = [4, 2]

        # pd_op.pool2d: (4x384x1x40xf32) <- (4x384x4x80xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            reshape_42,
            full_int_array_12,
            [4, 2],
            [0, 0],
            False,
            True,
            "NCHW",
            "avg",
            False,
            False,
            "EXPLICIT",
        )
        del (
            add_0,
            add_1,
            add_10,
            add_101,
            add_102,
            add_103,
            add_105,
            add_106,
            add_107,
            add_109,
            add_11,
            add_110,
            add_111,
            add_113,
            add_114,
            add_115,
            add_117,
            add_118,
            add_119,
            add_121,
            add_122,
            add_123,
            add_125,
            add_126,
            add_127,
            add_129,
            add_13,
            add_130,
            add_131,
            add_133,
            add_134,
            add_135,
            add_137,
            add_14,
            add_16,
            add_17,
            add_18,
            add_2,
            add_20,
            add_21,
            add_23,
            add_24,
            add_25,
            add_27,
            add_28,
            add_3,
            add_30,
            add_31,
            add_32,
            add_34,
            add_35,
            add_37,
            add_38,
            add_39,
            add_4,
            add_41,
            add_42,
            add_43,
            add_45,
            add_46,
            add_47,
            add_49,
            add_5,
            add_50,
            add_52,
            add_53,
            add_54,
            add_56,
            add_57,
            add_58,
            add_6,
            add_60,
            add_61,
            add_62,
            add_64,
            add_65,
            add_66,
            add_68,
            add_69,
            add_7,
            add_70,
            add_72,
            add_73,
            add_74,
            add_76,
            add_77,
            add_78,
            add_80,
            add_81,
            add_82,
            add_84,
            add_85,
            add_86,
            add_88,
            add_89,
            add_9,
            add_90,
            add_91,
            add_93,
            add_94,
            add_95,
            add_97,
            add_98,
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
            assign_7,
            assign_8,
            assign_9,
            batch_norm__0,
            batch_norm__1,
            batch_norm__10,
            batch_norm__11,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
            batch_norm__5,
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            cast_0,
            cast_1,
            cast_10,
            cast_11,
            cast_12,
            cast_13,
            cast_14,
            cast_15,
            cast_16,
            cast_17,
            cast_18,
            cast_19,
            cast_2,
            cast_20,
            cast_21,
            cast_22,
            cast_23,
            cast_24,
            cast_25,
            cast_26,
            cast_27,
            cast_28,
            cast_29,
            cast_3,
            cast_30,
            cast_31,
            cast_32,
            cast_33,
            cast_4,
            cast_5,
            cast_6,
            cast_7,
            cast_8,
            cast_9,
            conv2d_0,
            conv2d_1,
            conv2d_10,
            conv2d_11,
            conv2d_2,
            conv2d_3,
            conv2d_4,
            conv2d_5,
            conv2d_6,
            conv2d_7,
            conv2d_8,
            conv2d_9,
            divide_0,
            divide_1,
            divide_10,
            divide_11,
            divide_12,
            divide_13,
            divide_14,
            divide_15,
            divide_16,
            divide_17,
            divide_18,
            divide_19,
            divide_2,
            divide_20,
            divide_21,
            divide_22,
            divide_23,
            divide_24,
            divide_25,
            divide_26,
            divide_27,
            divide_28,
            divide_29,
            divide_3,
            divide_30,
            divide_31,
            divide_32,
            divide_33,
            divide_4,
            divide_5,
            divide_6,
            divide_7,
            divide_8,
            divide_9,
            floor_0,
            floor_1,
            floor_10,
            floor_11,
            floor_12,
            floor_13,
            floor_14,
            floor_15,
            floor_16,
            floor_17,
            floor_18,
            floor_19,
            floor_2,
            floor_20,
            floor_21,
            floor_22,
            floor_23,
            floor_24,
            floor_25,
            floor_26,
            floor_27,
            floor_28,
            floor_29,
            floor_3,
            floor_30,
            floor_31,
            floor_32,
            floor_33,
            floor_4,
            floor_5,
            floor_6,
            floor_7,
            floor_8,
            floor_9,
            full_17,
            full_int_array_12,
            full_int_array_6,
            full_int_array_7,
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
            layer_norm_112,
            layer_norm_113,
            layer_norm_12,
            layer_norm_13,
            layer_norm_14,
            layer_norm_16,
            layer_norm_17,
            layer_norm_18,
            layer_norm_19,
            layer_norm_2,
            layer_norm_20,
            layer_norm_22,
            layer_norm_23,
            layer_norm_24,
            layer_norm_25,
            layer_norm_26,
            layer_norm_28,
            layer_norm_29,
            layer_norm_30,
            layer_norm_31,
            layer_norm_32,
            layer_norm_34,
            layer_norm_35,
            layer_norm_37,
            layer_norm_38,
            layer_norm_39,
            layer_norm_4,
            layer_norm_40,
            layer_norm_41,
            layer_norm_43,
            layer_norm_44,
            layer_norm_45,
            layer_norm_46,
            layer_norm_47,
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
            matmul_1,
            matmul_10,
            matmul_11,
            matmul_12,
            matmul_13,
            matmul_14,
            matmul_15,
            matmul_16,
            matmul_19,
            matmul_2,
            matmul_20,
            matmul_21,
            matmul_22,
            matmul_25,
            matmul_26,
            matmul_27,
            matmul_28,
            matmul_3,
            matmul_31,
            matmul_32,
            matmul_33,
            matmul_34,
            matmul_37,
            matmul_38,
            matmul_39,
            matmul_4,
            matmul_40,
            matmul_43,
            matmul_44,
            matmul_45,
            matmul_46,
            matmul_49,
            matmul_5,
            matmul_50,
            matmul_51,
            matmul_52,
            matmul_55,
            matmul_56,
            matmul_57,
            matmul_58,
            matmul_6,
            matmul_61,
            matmul_62,
            matmul_63,
            matmul_64,
            matmul_67,
            matmul_68,
            matmul_69,
            matmul_7,
            matmul_70,
            matmul_73,
            matmul_74,
            matmul_75,
            matmul_8,
            matmul_9,
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
            multiply_4,
            multiply_5,
            multiply_6,
            multiply_7,
            multiply_8,
            multiply_9,
            reshape_0,
            reshape_1,
            reshape_10,
            reshape_11,
            reshape_12,
            reshape_13,
            reshape_14,
            reshape_15,
            reshape_16,
            reshape_17,
            reshape_18,
            reshape_19,
            reshape_2,
            reshape_21,
            reshape_23,
            reshape_25,
            reshape_27,
            reshape_28,
            reshape_29,
            reshape_3,
            reshape_31,
            reshape_33,
            reshape_35,
            reshape_37,
            reshape_39,
            reshape_4,
            reshape_41,
            reshape_42,
            reshape_5,
            reshape_6,
            reshape_7,
            reshape_8,
            reshape_9,
            slice_14,
            slice_16,
            slice_19,
            slice_21,
            slice_24,
            slice_26,
            slice_29,
            slice_31,
            slice_34,
            slice_36,
            slice_39,
            slice_41,
            slice_44,
            slice_46,
            slice_49,
            slice_51,
            slice_54,
            slice_56,
            slice_59,
            slice_61,
            softmax_0,
            softmax_1,
            softmax_2,
            softmax_3,
            softmax_4,
            softmax_5,
            softmax_6,
            softmax_7,
            softmax_8,
            softmax_9,
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
            transpose_6,
            transpose_7,
            transpose_8,
            transpose_9,
        )

        return pool2d_0
