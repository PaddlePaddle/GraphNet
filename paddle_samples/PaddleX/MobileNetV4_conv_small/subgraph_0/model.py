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
        data_0,
    ):
        # pd_op.conv2d: (128x32x112x112xf32) <- (128x3x224x224xf32, 32x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_231, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_0, parameter_231

        # pd_op.batch_norm_: (128x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (128x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        (
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
            batch_norm__5,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_0,
                parameter_230,
                parameter_229,
                parameter_228,
                parameter_227,
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
        del parameter_227, parameter_228, parameter_229, parameter_230

        # pd_op.relu: (128x32x112x112xf32) <- (128x32x112x112xf32)
        relu_0 = paddle._C_ops.relu(batch_norm__0)
        del batch_norm__0

        # pd_op.conv2d: (128x32x56x56xf32) <- (128x32x112x112xf32, 32x32x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            relu_0, parameter_226, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_226

        # pd_op.batch_norm_: (128x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (128x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        (
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            batch_norm__10,
            batch_norm__11,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_1,
                parameter_225,
                parameter_224,
                parameter_223,
                parameter_222,
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
        del parameter_222, parameter_223, parameter_224, parameter_225

        # pd_op.relu: (128x32x56x56xf32) <- (128x32x56x56xf32)
        relu_1 = paddle._C_ops.relu(batch_norm__6)
        del batch_norm__6

        # pd_op.conv2d: (128x32x56x56xf32) <- (128x32x56x56xf32, 32x32x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            relu_1, parameter_221, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_221

        # pd_op.batch_norm_: (128x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (128x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        (
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            batch_norm__16,
            batch_norm__17,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_2,
                parameter_220,
                parameter_219,
                parameter_218,
                parameter_217,
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
        del parameter_217, parameter_218, parameter_219, parameter_220

        # pd_op.relu: (128x32x56x56xf32) <- (128x32x56x56xf32)
        relu_2 = paddle._C_ops.relu(batch_norm__12)
        del batch_norm__12

        # pd_op.full: (xf32) <- ()
        full_0 = paddle._C_ops.full(
            [],
            float("0.998235"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [128, 1, 1, 1]

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_0 = paddle._C_ops.uniform(
            full_int_array_0,
            paddle.float32,
            full_1,
            full_2,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_1 = paddle._C_ops.add(full_0, uniform_0)
        del uniform_0

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_0 = paddle._C_ops.floor(add_1)
        del add_1

        # pd_op.divide: (128x32x56x56xf32) <- (128x32x56x56xf32, xf32)
        divide_0 = paddle._C_ops.divide(relu_2, full_0)

        # pd_op.multiply: (128x32x56x56xf32) <- (128x32x56x56xf32, 128x1x1x1xf32)
        multiply_0 = paddle._C_ops.multiply(divide_0, floor_0)

        # pd_op.conv2d: (128x96x28x28xf32) <- (128x32x56x56xf32, 96x32x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            multiply_0, parameter_216, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_216

        # pd_op.batch_norm_: (128x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (128x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__18,
            batch_norm__19,
            batch_norm__20,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_3,
                parameter_215,
                parameter_214,
                parameter_213,
                parameter_212,
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
        del parameter_212, parameter_213, parameter_214, parameter_215

        # pd_op.relu: (128x96x28x28xf32) <- (128x96x28x28xf32)
        relu_3 = paddle._C_ops.relu(batch_norm__18)
        del batch_norm__18

        # pd_op.full: (xf32) <- ()
        full_3 = paddle._C_ops.full(
            [],
            float("0.996471"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_1 = paddle._C_ops.uniform(
            full_int_array_0,
            paddle.float32,
            full_1,
            full_2,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_2 = paddle._C_ops.add(full_3, uniform_1)
        del uniform_1

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_1 = paddle._C_ops.floor(add_2)
        del add_2

        # pd_op.divide: (128x96x28x28xf32) <- (128x96x28x28xf32, xf32)
        divide_1 = paddle._C_ops.divide(relu_3, full_3)

        # pd_op.multiply: (128x96x28x28xf32) <- (128x96x28x28xf32, 128x1x1x1xf32)
        multiply_1 = paddle._C_ops.multiply(divide_1, floor_1)

        # pd_op.conv2d: (128x64x28x28xf32) <- (128x96x28x28xf32, 64x96x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            multiply_1, parameter_211, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_211

        # pd_op.batch_norm_: (128x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (128x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_4,
                parameter_210,
                parameter_209,
                parameter_208,
                parameter_207,
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
        del parameter_207, parameter_208, parameter_209, parameter_210

        # pd_op.relu: (128x64x28x28xf32) <- (128x64x28x28xf32)
        relu_4 = paddle._C_ops.relu(batch_norm__24)
        del batch_norm__24

        # pd_op.full: (xf32) <- ()
        full_4 = paddle._C_ops.full(
            [],
            float("0.994706"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_2 = paddle._C_ops.uniform(
            full_int_array_0,
            paddle.float32,
            full_1,
            full_2,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_3 = paddle._C_ops.add(full_4, uniform_2)
        del uniform_2

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_2 = paddle._C_ops.floor(add_3)
        del add_3

        # pd_op.divide: (128x64x28x28xf32) <- (128x64x28x28xf32, xf32)
        divide_2 = paddle._C_ops.divide(relu_4, full_4)

        # pd_op.multiply: (128x64x28x28xf32) <- (128x64x28x28xf32, 128x1x1x1xf32)
        multiply_2 = paddle._C_ops.multiply(divide_2, floor_2)

        # pd_op.depthwise_conv2d: (128x64x28x28xf32) <- (128x64x28x28xf32, 64x1x5x5xf32)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(
            multiply_2, parameter_206, [1, 1], [2, 2], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del parameter_206

        # pd_op.batch_norm_: (128x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (128x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_0,
                parameter_205,
                parameter_204,
                parameter_203,
                parameter_202,
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
        del parameter_202, parameter_203, parameter_204, parameter_205

        # pd_op.conv2d: (128x192x28x28xf32) <- (128x64x28x28xf32, 192x64x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            batch_norm__30, parameter_201, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_201

        # pd_op.batch_norm_: (128x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (128x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__36,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_5,
                parameter_200,
                parameter_199,
                parameter_198,
                parameter_197,
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
        del parameter_197, parameter_198, parameter_199, parameter_200

        # pd_op.relu: (128x192x28x28xf32) <- (128x192x28x28xf32)
        relu_5 = paddle._C_ops.relu(batch_norm__36)
        del batch_norm__36

        # pd_op.depthwise_conv2d: (128x192x14x14xf32) <- (128x192x28x28xf32, 192x1x5x5xf32)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(
            relu_5, parameter_196, [2, 2], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del parameter_196

        # pd_op.batch_norm_: (128x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (128x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__42,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_1,
                parameter_195,
                parameter_194,
                parameter_193,
                parameter_192,
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
        del parameter_192, parameter_193, parameter_194, parameter_195

        # pd_op.relu: (128x192x14x14xf32) <- (128x192x14x14xf32)
        relu_6 = paddle._C_ops.relu(batch_norm__42)
        del batch_norm__42

        # pd_op.conv2d: (128x96x14x14xf32) <- (128x192x14x14xf32, 96x192x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            relu_6, parameter_191, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_191

        # pd_op.batch_norm_: (128x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (128x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__48,
            batch_norm__49,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_6,
                parameter_190,
                parameter_189,
                parameter_188,
                parameter_187,
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
        del parameter_187, parameter_188, parameter_189, parameter_190

        # pd_op.conv2d: (128x192x14x14xf32) <- (128x96x14x14xf32, 192x96x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            batch_norm__48, parameter_186, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_186

        # pd_op.batch_norm_: (128x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (128x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_7,
                parameter_185,
                parameter_184,
                parameter_183,
                parameter_182,
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
        del parameter_182, parameter_183, parameter_184, parameter_185

        # pd_op.relu: (128x192x14x14xf32) <- (128x192x14x14xf32)
        relu_7 = paddle._C_ops.relu(batch_norm__54)
        del batch_norm__54

        # pd_op.depthwise_conv2d: (128x192x14x14xf32) <- (128x192x14x14xf32, 192x1x3x3xf32)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(
            relu_7, parameter_181, [1, 1], [1, 1], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del parameter_181

        # pd_op.batch_norm_: (128x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (128x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_2,
                parameter_180,
                parameter_179,
                parameter_178,
                parameter_177,
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
        del parameter_177, parameter_178, parameter_179, parameter_180

        # pd_op.relu: (128x192x14x14xf32) <- (128x192x14x14xf32)
        relu_8 = paddle._C_ops.relu(batch_norm__60)
        del batch_norm__60

        # pd_op.conv2d: (128x96x14x14xf32) <- (128x192x14x14xf32, 96x192x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            relu_8, parameter_176, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_176

        # pd_op.batch_norm_: (128x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (128x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__66,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            batch_norm__70,
            batch_norm__71,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_8,
                parameter_175,
                parameter_174,
                parameter_173,
                parameter_172,
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
        del parameter_172, parameter_173, parameter_174, parameter_175

        # pd_op.full: (xf32) <- ()
        full_5 = paddle._C_ops.full(
            [],
            float("0.991176"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_3 = paddle._C_ops.uniform(
            full_int_array_0,
            paddle.float32,
            full_1,
            full_2,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_4 = paddle._C_ops.add(full_5, uniform_3)
        del uniform_3

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_3 = paddle._C_ops.floor(add_4)
        del add_4

        # pd_op.divide: (128x96x14x14xf32) <- (128x96x14x14xf32, xf32)
        divide_3 = paddle._C_ops.divide(batch_norm__66, full_5)

        # pd_op.multiply: (128x96x14x14xf32) <- (128x96x14x14xf32, 128x1x1x1xf32)
        multiply_3 = paddle._C_ops.multiply(divide_3, floor_3)

        # pd_op.add: (128x96x14x14xf32) <- (128x96x14x14xf32, 128x96x14x14xf32)
        add_5 = paddle._C_ops.add(batch_norm__48, multiply_3)

        # pd_op.conv2d: (128x192x14x14xf32) <- (128x96x14x14xf32, 192x96x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            add_5, parameter_171, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_171

        # pd_op.batch_norm_: (128x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (128x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__72,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_9,
                parameter_170,
                parameter_169,
                parameter_168,
                parameter_167,
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
        del parameter_167, parameter_168, parameter_169, parameter_170

        # pd_op.relu: (128x192x14x14xf32) <- (128x192x14x14xf32)
        relu_9 = paddle._C_ops.relu(batch_norm__72)
        del batch_norm__72

        # pd_op.depthwise_conv2d: (128x192x14x14xf32) <- (128x192x14x14xf32, 192x1x3x3xf32)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(
            relu_9, parameter_166, [1, 1], [1, 1], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del parameter_166

        # pd_op.batch_norm_: (128x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (128x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__78,
            batch_norm__79,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
            batch_norm__83,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_3,
                parameter_165,
                parameter_164,
                parameter_163,
                parameter_162,
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
        del parameter_162, parameter_163, parameter_164, parameter_165

        # pd_op.relu: (128x192x14x14xf32) <- (128x192x14x14xf32)
        relu_10 = paddle._C_ops.relu(batch_norm__78)
        del batch_norm__78

        # pd_op.conv2d: (128x96x14x14xf32) <- (128x192x14x14xf32, 96x192x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            relu_10, parameter_161, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_161

        # pd_op.batch_norm_: (128x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (128x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__84,
            batch_norm__85,
            batch_norm__86,
            batch_norm__87,
            batch_norm__88,
            batch_norm__89,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_10,
                parameter_160,
                parameter_159,
                parameter_158,
                parameter_157,
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
        del parameter_157, parameter_158, parameter_159, parameter_160

        # pd_op.full: (xf32) <- ()
        full_6 = paddle._C_ops.full(
            [],
            float("0.989412"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_4 = paddle._C_ops.uniform(
            full_int_array_0,
            paddle.float32,
            full_1,
            full_2,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_6 = paddle._C_ops.add(full_6, uniform_4)
        del uniform_4

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_4 = paddle._C_ops.floor(add_6)
        del add_6

        # pd_op.divide: (128x96x14x14xf32) <- (128x96x14x14xf32, xf32)
        divide_4 = paddle._C_ops.divide(batch_norm__84, full_6)

        # pd_op.multiply: (128x96x14x14xf32) <- (128x96x14x14xf32, 128x1x1x1xf32)
        multiply_4 = paddle._C_ops.multiply(divide_4, floor_4)

        # pd_op.add: (128x96x14x14xf32) <- (128x96x14x14xf32, 128x96x14x14xf32)
        add_7 = paddle._C_ops.add(add_5, multiply_4)

        # pd_op.conv2d: (128x192x14x14xf32) <- (128x96x14x14xf32, 192x96x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            add_7, parameter_156, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_156

        # pd_op.batch_norm_: (128x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (128x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__90,
            batch_norm__91,
            batch_norm__92,
            batch_norm__93,
            batch_norm__94,
            batch_norm__95,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_11,
                parameter_155,
                parameter_154,
                parameter_153,
                parameter_152,
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
        del parameter_152, parameter_153, parameter_154, parameter_155

        # pd_op.relu: (128x192x14x14xf32) <- (128x192x14x14xf32)
        relu_11 = paddle._C_ops.relu(batch_norm__90)
        del batch_norm__90

        # pd_op.depthwise_conv2d: (128x192x14x14xf32) <- (128x192x14x14xf32, 192x1x3x3xf32)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(
            relu_11, parameter_151, [1, 1], [1, 1], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del parameter_151

        # pd_op.batch_norm_: (128x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (128x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__96,
            batch_norm__97,
            batch_norm__98,
            batch_norm__99,
            batch_norm__100,
            batch_norm__101,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_4,
                parameter_150,
                parameter_149,
                parameter_148,
                parameter_147,
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
        del parameter_147, parameter_148, parameter_149, parameter_150

        # pd_op.relu: (128x192x14x14xf32) <- (128x192x14x14xf32)
        relu_12 = paddle._C_ops.relu(batch_norm__96)
        del batch_norm__96

        # pd_op.conv2d: (128x96x14x14xf32) <- (128x192x14x14xf32, 96x192x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            relu_12, parameter_146, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_146

        # pd_op.batch_norm_: (128x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (128x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__102,
            batch_norm__103,
            batch_norm__104,
            batch_norm__105,
            batch_norm__106,
            batch_norm__107,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_12,
                parameter_145,
                parameter_144,
                parameter_143,
                parameter_142,
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
        del parameter_142, parameter_143, parameter_144, parameter_145

        # pd_op.full: (xf32) <- ()
        full_7 = paddle._C_ops.full(
            [],
            float("0.987647"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_5 = paddle._C_ops.uniform(
            full_int_array_0,
            paddle.float32,
            full_1,
            full_2,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_8 = paddle._C_ops.add(full_7, uniform_5)
        del uniform_5

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_5 = paddle._C_ops.floor(add_8)
        del add_8

        # pd_op.divide: (128x96x14x14xf32) <- (128x96x14x14xf32, xf32)
        divide_5 = paddle._C_ops.divide(batch_norm__102, full_7)

        # pd_op.multiply: (128x96x14x14xf32) <- (128x96x14x14xf32, 128x1x1x1xf32)
        multiply_5 = paddle._C_ops.multiply(divide_5, floor_5)

        # pd_op.add: (128x96x14x14xf32) <- (128x96x14x14xf32, 128x96x14x14xf32)
        add_9 = paddle._C_ops.add(add_7, multiply_5)

        # pd_op.conv2d: (128x192x14x14xf32) <- (128x96x14x14xf32, 192x96x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            add_9, parameter_141, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_141

        # pd_op.batch_norm_: (128x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (128x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__108,
            batch_norm__109,
            batch_norm__110,
            batch_norm__111,
            batch_norm__112,
            batch_norm__113,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_13,
                parameter_140,
                parameter_139,
                parameter_138,
                parameter_137,
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
        del parameter_137, parameter_138, parameter_139, parameter_140

        # pd_op.relu: (128x192x14x14xf32) <- (128x192x14x14xf32)
        relu_13 = paddle._C_ops.relu(batch_norm__108)
        del batch_norm__108

        # pd_op.depthwise_conv2d: (128x192x14x14xf32) <- (128x192x14x14xf32, 192x1x3x3xf32)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(
            relu_13, parameter_136, [1, 1], [1, 1], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del parameter_136

        # pd_op.batch_norm_: (128x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (128x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__114,
            batch_norm__115,
            batch_norm__116,
            batch_norm__117,
            batch_norm__118,
            batch_norm__119,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_5,
                parameter_135,
                parameter_134,
                parameter_133,
                parameter_132,
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
        del parameter_132, parameter_133, parameter_134, parameter_135

        # pd_op.relu: (128x192x14x14xf32) <- (128x192x14x14xf32)
        relu_14 = paddle._C_ops.relu(batch_norm__114)
        del batch_norm__114

        # pd_op.conv2d: (128x96x14x14xf32) <- (128x192x14x14xf32, 96x192x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            relu_14, parameter_131, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_131

        # pd_op.batch_norm_: (128x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (128x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__120,
            batch_norm__121,
            batch_norm__122,
            batch_norm__123,
            batch_norm__124,
            batch_norm__125,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_14,
                parameter_130,
                parameter_129,
                parameter_128,
                parameter_127,
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
        del parameter_127, parameter_128, parameter_129, parameter_130

        # pd_op.full: (xf32) <- ()
        full_8 = paddle._C_ops.full(
            [],
            float("0.985882"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_6 = paddle._C_ops.uniform(
            full_int_array_0,
            paddle.float32,
            full_1,
            full_2,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_10 = paddle._C_ops.add(full_8, uniform_6)
        del uniform_6

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_6 = paddle._C_ops.floor(add_10)
        del add_10

        # pd_op.divide: (128x96x14x14xf32) <- (128x96x14x14xf32, xf32)
        divide_6 = paddle._C_ops.divide(batch_norm__120, full_8)

        # pd_op.multiply: (128x96x14x14xf32) <- (128x96x14x14xf32, 128x1x1x1xf32)
        multiply_6 = paddle._C_ops.multiply(divide_6, floor_6)

        # pd_op.add: (128x96x14x14xf32) <- (128x96x14x14xf32, 128x96x14x14xf32)
        add_11 = paddle._C_ops.add(add_9, multiply_6)

        # pd_op.depthwise_conv2d: (128x96x14x14xf32) <- (128x96x14x14xf32, 96x1x3x3xf32)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(
            add_11, parameter_126, [1, 1], [1, 1], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del parameter_126

        # pd_op.batch_norm_: (128x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (128x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__126,
            batch_norm__127,
            batch_norm__128,
            batch_norm__129,
            batch_norm__130,
            batch_norm__131,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_6,
                parameter_125,
                parameter_124,
                parameter_123,
                parameter_122,
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
        del parameter_122, parameter_123, parameter_124, parameter_125

        # pd_op.conv2d: (128x384x14x14xf32) <- (128x96x14x14xf32, 384x96x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            batch_norm__126,
            parameter_121,
            [1, 1],
            [0, 0],
            "EXPLICIT",
            [1, 1],
            1,
            "NCHW",
        )
        del parameter_121

        # pd_op.batch_norm_: (128x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (128x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__132,
            batch_norm__133,
            batch_norm__134,
            batch_norm__135,
            batch_norm__136,
            batch_norm__137,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_15,
                parameter_120,
                parameter_119,
                parameter_118,
                parameter_117,
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
        del parameter_117, parameter_118, parameter_119, parameter_120

        # pd_op.relu: (128x384x14x14xf32) <- (128x384x14x14xf32)
        relu_15 = paddle._C_ops.relu(batch_norm__132)
        del batch_norm__132

        # pd_op.conv2d: (128x96x14x14xf32) <- (128x384x14x14xf32, 96x384x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            relu_15, parameter_116, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_116

        # pd_op.batch_norm_: (128x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (128x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__138,
            batch_norm__139,
            batch_norm__140,
            batch_norm__141,
            batch_norm__142,
            batch_norm__143,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_16,
                parameter_115,
                parameter_114,
                parameter_113,
                parameter_112,
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
        del parameter_112, parameter_113, parameter_114, parameter_115

        # pd_op.full: (xf32) <- ()
        full_9 = paddle._C_ops.full(
            [],
            float("0.984118"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_7 = paddle._C_ops.uniform(
            full_int_array_0,
            paddle.float32,
            full_1,
            full_2,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_12 = paddle._C_ops.add(full_9, uniform_7)
        del uniform_7

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_7 = paddle._C_ops.floor(add_12)
        del add_12

        # pd_op.divide: (128x96x14x14xf32) <- (128x96x14x14xf32, xf32)
        divide_7 = paddle._C_ops.divide(batch_norm__138, full_9)

        # pd_op.multiply: (128x96x14x14xf32) <- (128x96x14x14xf32, 128x1x1x1xf32)
        multiply_7 = paddle._C_ops.multiply(divide_7, floor_7)

        # pd_op.add: (128x96x14x14xf32) <- (128x96x14x14xf32, 128x96x14x14xf32)
        add_13 = paddle._C_ops.add(add_11, multiply_7)

        # pd_op.depthwise_conv2d: (128x96x14x14xf32) <- (128x96x14x14xf32, 96x1x3x3xf32)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(
            add_13, parameter_111, [1, 1], [1, 1], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del parameter_111

        # pd_op.batch_norm_: (128x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (128x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__144,
            batch_norm__145,
            batch_norm__146,
            batch_norm__147,
            batch_norm__148,
            batch_norm__149,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_7,
                parameter_110,
                parameter_109,
                parameter_108,
                parameter_107,
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
        del parameter_107, parameter_108, parameter_109, parameter_110

        # pd_op.conv2d: (128x576x14x14xf32) <- (128x96x14x14xf32, 576x96x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            batch_norm__144,
            parameter_106,
            [1, 1],
            [0, 0],
            "EXPLICIT",
            [1, 1],
            1,
            "NCHW",
        )
        del parameter_106

        # pd_op.batch_norm_: (128x576x14x14xf32, 576xf32, 576xf32, 576xf32, 576xf32, -1xui8) <- (128x576x14x14xf32, 576xf32, 576xf32, 576xf32, 576xf32)
        (
            batch_norm__150,
            batch_norm__151,
            batch_norm__152,
            batch_norm__153,
            batch_norm__154,
            batch_norm__155,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_17,
                parameter_105,
                parameter_104,
                parameter_103,
                parameter_102,
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
        del parameter_102, parameter_103, parameter_104, parameter_105

        # pd_op.relu: (128x576x14x14xf32) <- (128x576x14x14xf32)
        relu_16 = paddle._C_ops.relu(batch_norm__150)
        del batch_norm__150

        # pd_op.depthwise_conv2d: (128x576x7x7xf32) <- (128x576x14x14xf32, 576x1x3x3xf32)
        depthwise_conv2d_8 = paddle._C_ops.depthwise_conv2d(
            relu_16, parameter_101, [2, 2], [1, 1], "EXPLICIT", 576, [1, 1], "NCHW"
        )
        del parameter_101

        # pd_op.batch_norm_: (128x576x7x7xf32, 576xf32, 576xf32, 576xf32, 576xf32, -1xui8) <- (128x576x7x7xf32, 576xf32, 576xf32, 576xf32, 576xf32)
        (
            batch_norm__156,
            batch_norm__157,
            batch_norm__158,
            batch_norm__159,
            batch_norm__160,
            batch_norm__161,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_8,
                parameter_100,
                parameter_99,
                parameter_98,
                parameter_97,
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
        del parameter_100, parameter_97, parameter_98, parameter_99

        # pd_op.relu: (128x576x7x7xf32) <- (128x576x7x7xf32)
        relu_17 = paddle._C_ops.relu(batch_norm__156)
        del batch_norm__156

        # pd_op.conv2d: (128x128x7x7xf32) <- (128x576x7x7xf32, 128x576x1x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            relu_17, parameter_96, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_96

        # pd_op.batch_norm_: (128x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (128x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__162,
            batch_norm__163,
            batch_norm__164,
            batch_norm__165,
            batch_norm__166,
            batch_norm__167,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_18,
                parameter_95,
                parameter_94,
                parameter_93,
                parameter_92,
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
        del parameter_92, parameter_93, parameter_94, parameter_95

        # pd_op.depthwise_conv2d: (128x128x7x7xf32) <- (128x128x7x7xf32, 128x1x5x5xf32)
        depthwise_conv2d_9 = paddle._C_ops.depthwise_conv2d(
            batch_norm__162,
            parameter_91,
            [1, 1],
            [2, 2],
            "EXPLICIT",
            128,
            [1, 1],
            "NCHW",
        )
        del parameter_91

        # pd_op.batch_norm_: (128x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (128x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__168,
            batch_norm__169,
            batch_norm__170,
            batch_norm__171,
            batch_norm__172,
            batch_norm__173,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_9,
                parameter_90,
                parameter_89,
                parameter_88,
                parameter_87,
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
        del parameter_87, parameter_88, parameter_89, parameter_90

        # pd_op.conv2d: (128x512x7x7xf32) <- (128x128x7x7xf32, 512x128x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            batch_norm__168, parameter_86, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_86

        # pd_op.batch_norm_: (128x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__174,
            batch_norm__175,
            batch_norm__176,
            batch_norm__177,
            batch_norm__178,
            batch_norm__179,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_19,
                parameter_85,
                parameter_84,
                parameter_83,
                parameter_82,
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
        del parameter_82, parameter_83, parameter_84, parameter_85

        # pd_op.relu: (128x512x7x7xf32) <- (128x512x7x7xf32)
        relu_18 = paddle._C_ops.relu(batch_norm__174)
        del batch_norm__174

        # pd_op.depthwise_conv2d: (128x512x7x7xf32) <- (128x512x7x7xf32, 512x1x5x5xf32)
        depthwise_conv2d_10 = paddle._C_ops.depthwise_conv2d(
            relu_18, parameter_81, [1, 1], [2, 2], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del parameter_81

        # pd_op.batch_norm_: (128x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__180,
            batch_norm__181,
            batch_norm__182,
            batch_norm__183,
            batch_norm__184,
            batch_norm__185,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_10,
                parameter_80,
                parameter_79,
                parameter_78,
                parameter_77,
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
        del parameter_77, parameter_78, parameter_79, parameter_80

        # pd_op.relu: (128x512x7x7xf32) <- (128x512x7x7xf32)
        relu_19 = paddle._C_ops.relu(batch_norm__180)
        del batch_norm__180

        # pd_op.conv2d: (128x128x7x7xf32) <- (128x512x7x7xf32, 128x512x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            relu_19, parameter_76, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_76

        # pd_op.batch_norm_: (128x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (128x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__186,
            batch_norm__187,
            batch_norm__188,
            batch_norm__189,
            batch_norm__190,
            batch_norm__191,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_20,
                parameter_75,
                parameter_74,
                parameter_73,
                parameter_72,
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
        del parameter_72, parameter_73, parameter_74, parameter_75

        # pd_op.full: (xf32) <- ()
        full_10 = paddle._C_ops.full(
            [],
            float("0.980588"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_8 = paddle._C_ops.uniform(
            full_int_array_0,
            paddle.float32,
            full_1,
            full_2,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_14 = paddle._C_ops.add(full_10, uniform_8)
        del uniform_8

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_8 = paddle._C_ops.floor(add_14)
        del add_14

        # pd_op.divide: (128x128x7x7xf32) <- (128x128x7x7xf32, xf32)
        divide_8 = paddle._C_ops.divide(batch_norm__186, full_10)

        # pd_op.multiply: (128x128x7x7xf32) <- (128x128x7x7xf32, 128x1x1x1xf32)
        multiply_8 = paddle._C_ops.multiply(divide_8, floor_8)

        # pd_op.add: (128x128x7x7xf32) <- (128x128x7x7xf32, 128x128x7x7xf32)
        add_15 = paddle._C_ops.add(batch_norm__162, multiply_8)

        # pd_op.conv2d: (128x512x7x7xf32) <- (128x128x7x7xf32, 512x128x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            add_15, parameter_71, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_71

        # pd_op.batch_norm_: (128x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__192,
            batch_norm__193,
            batch_norm__194,
            batch_norm__195,
            batch_norm__196,
            batch_norm__197,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_21,
                parameter_70,
                parameter_69,
                parameter_68,
                parameter_67,
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
        del parameter_67, parameter_68, parameter_69, parameter_70

        # pd_op.relu: (128x512x7x7xf32) <- (128x512x7x7xf32)
        relu_20 = paddle._C_ops.relu(batch_norm__192)
        del batch_norm__192

        # pd_op.depthwise_conv2d: (128x512x7x7xf32) <- (128x512x7x7xf32, 512x1x5x5xf32)
        depthwise_conv2d_11 = paddle._C_ops.depthwise_conv2d(
            relu_20, parameter_66, [1, 1], [2, 2], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del parameter_66

        # pd_op.batch_norm_: (128x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__198,
            batch_norm__199,
            batch_norm__200,
            batch_norm__201,
            batch_norm__202,
            batch_norm__203,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_11,
                parameter_65,
                parameter_64,
                parameter_63,
                parameter_62,
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
        del parameter_62, parameter_63, parameter_64, parameter_65

        # pd_op.relu: (128x512x7x7xf32) <- (128x512x7x7xf32)
        relu_21 = paddle._C_ops.relu(batch_norm__198)
        del batch_norm__198

        # pd_op.conv2d: (128x128x7x7xf32) <- (128x512x7x7xf32, 128x512x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(
            relu_21, parameter_61, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_61

        # pd_op.batch_norm_: (128x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (128x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__204,
            batch_norm__205,
            batch_norm__206,
            batch_norm__207,
            batch_norm__208,
            batch_norm__209,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_22,
                parameter_60,
                parameter_59,
                parameter_58,
                parameter_57,
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
        del parameter_57, parameter_58, parameter_59, parameter_60

        # pd_op.full: (xf32) <- ()
        full_11 = paddle._C_ops.full(
            [],
            float("0.978824"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_9 = paddle._C_ops.uniform(
            full_int_array_0,
            paddle.float32,
            full_1,
            full_2,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_16 = paddle._C_ops.add(full_11, uniform_9)
        del uniform_9

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_9 = paddle._C_ops.floor(add_16)
        del add_16

        # pd_op.divide: (128x128x7x7xf32) <- (128x128x7x7xf32, xf32)
        divide_9 = paddle._C_ops.divide(batch_norm__204, full_11)

        # pd_op.multiply: (128x128x7x7xf32) <- (128x128x7x7xf32, 128x1x1x1xf32)
        multiply_9 = paddle._C_ops.multiply(divide_9, floor_9)

        # pd_op.add: (128x128x7x7xf32) <- (128x128x7x7xf32, 128x128x7x7xf32)
        add_17 = paddle._C_ops.add(add_15, multiply_9)

        # pd_op.conv2d: (128x384x7x7xf32) <- (128x128x7x7xf32, 384x128x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(
            add_17, parameter_56, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_56

        # pd_op.batch_norm_: (128x384x7x7xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (128x384x7x7xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__210,
            batch_norm__211,
            batch_norm__212,
            batch_norm__213,
            batch_norm__214,
            batch_norm__215,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_23,
                parameter_55,
                parameter_54,
                parameter_53,
                parameter_52,
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
        del parameter_52, parameter_53, parameter_54, parameter_55

        # pd_op.relu: (128x384x7x7xf32) <- (128x384x7x7xf32)
        relu_22 = paddle._C_ops.relu(batch_norm__210)
        del batch_norm__210

        # pd_op.depthwise_conv2d: (128x384x7x7xf32) <- (128x384x7x7xf32, 384x1x5x5xf32)
        depthwise_conv2d_12 = paddle._C_ops.depthwise_conv2d(
            relu_22, parameter_51, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del parameter_51

        # pd_op.batch_norm_: (128x384x7x7xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (128x384x7x7xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__216,
            batch_norm__217,
            batch_norm__218,
            batch_norm__219,
            batch_norm__220,
            batch_norm__221,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_12,
                parameter_50,
                parameter_49,
                parameter_48,
                parameter_47,
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
        del parameter_47, parameter_48, parameter_49, parameter_50

        # pd_op.relu: (128x384x7x7xf32) <- (128x384x7x7xf32)
        relu_23 = paddle._C_ops.relu(batch_norm__216)
        del batch_norm__216

        # pd_op.conv2d: (128x128x7x7xf32) <- (128x384x7x7xf32, 128x384x1x1xf32)
        conv2d_24 = paddle._C_ops.conv2d(
            relu_23, parameter_46, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_46

        # pd_op.batch_norm_: (128x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (128x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__222,
            batch_norm__223,
            batch_norm__224,
            batch_norm__225,
            batch_norm__226,
            batch_norm__227,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_24,
                parameter_45,
                parameter_44,
                parameter_43,
                parameter_42,
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
        del parameter_42, parameter_43, parameter_44, parameter_45

        # pd_op.full: (xf32) <- ()
        full_12 = paddle._C_ops.full(
            [],
            float("0.977059"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_10 = paddle._C_ops.uniform(
            full_int_array_0,
            paddle.float32,
            full_1,
            full_2,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_18 = paddle._C_ops.add(full_12, uniform_10)
        del uniform_10

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_10 = paddle._C_ops.floor(add_18)
        del add_18

        # pd_op.divide: (128x128x7x7xf32) <- (128x128x7x7xf32, xf32)
        divide_10 = paddle._C_ops.divide(batch_norm__222, full_12)

        # pd_op.multiply: (128x128x7x7xf32) <- (128x128x7x7xf32, 128x1x1x1xf32)
        multiply_10 = paddle._C_ops.multiply(divide_10, floor_10)

        # pd_op.add: (128x128x7x7xf32) <- (128x128x7x7xf32, 128x128x7x7xf32)
        add_19 = paddle._C_ops.add(add_17, multiply_10)

        # pd_op.conv2d: (128x512x7x7xf32) <- (128x128x7x7xf32, 512x128x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(
            add_19, parameter_41, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_41

        # pd_op.batch_norm_: (128x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__228,
            batch_norm__229,
            batch_norm__230,
            batch_norm__231,
            batch_norm__232,
            batch_norm__233,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_25,
                parameter_40,
                parameter_39,
                parameter_38,
                parameter_37,
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
        del parameter_37, parameter_38, parameter_39, parameter_40

        # pd_op.relu: (128x512x7x7xf32) <- (128x512x7x7xf32)
        relu_24 = paddle._C_ops.relu(batch_norm__228)
        del batch_norm__228

        # pd_op.depthwise_conv2d: (128x512x7x7xf32) <- (128x512x7x7xf32, 512x1x3x3xf32)
        depthwise_conv2d_13 = paddle._C_ops.depthwise_conv2d(
            relu_24, parameter_36, [1, 1], [1, 1], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del parameter_36

        # pd_op.batch_norm_: (128x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__234,
            batch_norm__235,
            batch_norm__236,
            batch_norm__237,
            batch_norm__238,
            batch_norm__239,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_13,
                parameter_35,
                parameter_34,
                parameter_33,
                parameter_32,
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
        del parameter_32, parameter_33, parameter_34, parameter_35

        # pd_op.relu: (128x512x7x7xf32) <- (128x512x7x7xf32)
        relu_25 = paddle._C_ops.relu(batch_norm__234)
        del batch_norm__234

        # pd_op.conv2d: (128x128x7x7xf32) <- (128x512x7x7xf32, 128x512x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(
            relu_25, parameter_31, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_31

        # pd_op.batch_norm_: (128x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (128x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__240,
            batch_norm__241,
            batch_norm__242,
            batch_norm__243,
            batch_norm__244,
            batch_norm__245,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_26,
                parameter_30,
                parameter_29,
                parameter_28,
                parameter_27,
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
        del parameter_27, parameter_28, parameter_29, parameter_30

        # pd_op.full: (xf32) <- ()
        full_13 = paddle._C_ops.full(
            [],
            float("0.975294"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_11 = paddle._C_ops.uniform(
            full_int_array_0,
            paddle.float32,
            full_1,
            full_2,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_20 = paddle._C_ops.add(full_13, uniform_11)
        del uniform_11

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_11 = paddle._C_ops.floor(add_20)
        del add_20

        # pd_op.divide: (128x128x7x7xf32) <- (128x128x7x7xf32, xf32)
        divide_11 = paddle._C_ops.divide(batch_norm__240, full_13)

        # pd_op.multiply: (128x128x7x7xf32) <- (128x128x7x7xf32, 128x1x1x1xf32)
        multiply_11 = paddle._C_ops.multiply(divide_11, floor_11)

        # pd_op.add: (128x128x7x7xf32) <- (128x128x7x7xf32, 128x128x7x7xf32)
        add_21 = paddle._C_ops.add(add_19, multiply_11)

        # pd_op.conv2d: (128x512x7x7xf32) <- (128x128x7x7xf32, 512x128x1x1xf32)
        conv2d_27 = paddle._C_ops.conv2d(
            add_21, parameter_26, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_26

        # pd_op.batch_norm_: (128x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__246,
            batch_norm__247,
            batch_norm__248,
            batch_norm__249,
            batch_norm__250,
            batch_norm__251,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_27,
                parameter_25,
                parameter_24,
                parameter_23,
                parameter_22,
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
        del parameter_22, parameter_23, parameter_24, parameter_25

        # pd_op.relu: (128x512x7x7xf32) <- (128x512x7x7xf32)
        relu_26 = paddle._C_ops.relu(batch_norm__246)
        del batch_norm__246

        # pd_op.depthwise_conv2d: (128x512x7x7xf32) <- (128x512x7x7xf32, 512x1x3x3xf32)
        depthwise_conv2d_14 = paddle._C_ops.depthwise_conv2d(
            relu_26, parameter_21, [1, 1], [1, 1], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del parameter_21

        # pd_op.batch_norm_: (128x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__252,
            batch_norm__253,
            batch_norm__254,
            batch_norm__255,
            batch_norm__256,
            batch_norm__257,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_14,
                parameter_20,
                parameter_19,
                parameter_18,
                parameter_17,
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
        del parameter_17, parameter_18, parameter_19, parameter_20

        # pd_op.relu: (128x512x7x7xf32) <- (128x512x7x7xf32)
        relu_27 = paddle._C_ops.relu(batch_norm__252)
        del batch_norm__252

        # pd_op.conv2d: (128x128x7x7xf32) <- (128x512x7x7xf32, 128x512x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(
            relu_27, parameter_16, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_16

        # pd_op.batch_norm_: (128x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (128x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__258,
            batch_norm__259,
            batch_norm__260,
            batch_norm__261,
            batch_norm__262,
            batch_norm__263,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_28,
                parameter_15,
                parameter_14,
                parameter_13,
                parameter_12,
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
        del parameter_12, parameter_13, parameter_14, parameter_15

        # pd_op.full: (xf32) <- ()
        full_14 = paddle._C_ops.full(
            [],
            float("0.973529"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_12 = paddle._C_ops.uniform(
            full_int_array_0,
            paddle.float32,
            full_1,
            full_2,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_22 = paddle._C_ops.add(full_14, uniform_12)
        del uniform_12

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_12 = paddle._C_ops.floor(add_22)
        del add_22

        # pd_op.divide: (128x128x7x7xf32) <- (128x128x7x7xf32, xf32)
        divide_12 = paddle._C_ops.divide(batch_norm__258, full_14)

        # pd_op.multiply: (128x128x7x7xf32) <- (128x128x7x7xf32, 128x1x1x1xf32)
        multiply_12 = paddle._C_ops.multiply(divide_12, floor_12)

        # pd_op.add: (128x128x7x7xf32) <- (128x128x7x7xf32, 128x128x7x7xf32)
        add_23 = paddle._C_ops.add(add_21, multiply_12)

        # pd_op.conv2d: (128x960x7x7xf32) <- (128x128x7x7xf32, 960x128x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(
            add_23, parameter_11, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_11

        # pd_op.batch_norm_: (128x960x7x7xf32, 960xf32, 960xf32, 960xf32, 960xf32, -1xui8) <- (128x960x7x7xf32, 960xf32, 960xf32, 960xf32, 960xf32)
        (
            batch_norm__264,
            batch_norm__265,
            batch_norm__266,
            batch_norm__267,
            batch_norm__268,
            batch_norm__269,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_29,
                parameter_10,
                parameter_9,
                parameter_8,
                parameter_7,
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
        del parameter_10, parameter_7, parameter_8, parameter_9

        # pd_op.relu: (128x960x7x7xf32) <- (128x960x7x7xf32)
        relu_28 = paddle._C_ops.relu(batch_norm__264)
        del batch_norm__264

        # pd_op.full: (xf32) <- ()
        full_15 = paddle._C_ops.full(
            [],
            float("0.971765"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_13 = paddle._C_ops.uniform(
            full_int_array_0,
            paddle.float32,
            full_1,
            full_2,
            0,
            paddle.framework._current_expected_place(),
        )
        del full_1, full_2, full_int_array_0

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_24 = paddle._C_ops.add(full_15, uniform_13)
        del uniform_13

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_13 = paddle._C_ops.floor(add_24)
        del add_24

        # pd_op.divide: (128x960x7x7xf32) <- (128x960x7x7xf32, xf32)
        divide_13 = paddle._C_ops.divide(relu_28, full_15)

        # pd_op.multiply: (128x960x7x7xf32) <- (128x960x7x7xf32, 128x1x1x1xf32)
        multiply_13 = paddle._C_ops.multiply(divide_13, floor_13)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [1, 1]

        # pd_op.pool2d: (128x960x1x1xf32) <- (128x960x7x7xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            multiply_13,
            full_int_array_1,
            [1, 1],
            [0, 0],
            False,
            True,
            "NCHW",
            "avg",
            False,
            True,
            "EXPLICIT",
        )

        # pd_op.conv2d: (128x1280x1x1xf32) <- (128x960x1x1xf32, 1280x960x1x1xf32)
        conv2d_30 = paddle._C_ops.conv2d(
            pool2d_0, parameter_6, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_6

        # pd_op.batch_norm_: (128x1280x1x1xf32, 1280xf32, 1280xf32, 1280xf32, 1280xf32, -1xui8) <- (128x1280x1x1xf32, 1280xf32, 1280xf32, 1280xf32, 1280xf32)
        (
            batch_norm__270,
            batch_norm__271,
            batch_norm__272,
            batch_norm__273,
            batch_norm__274,
            batch_norm__275,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_30,
                parameter_5,
                parameter_4,
                parameter_3,
                parameter_2,
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
        del parameter_2, parameter_3, parameter_4, parameter_5

        # pd_op.relu: (128x1280x1x1xf32) <- (128x1280x1x1xf32)
        relu_29 = paddle._C_ops.relu(batch_norm__270)
        del batch_norm__270

        # pd_op.flatten: (128x1280xf32) <- (128x1280x1x1xf32)
        flatten_0 = paddle._C_ops.flatten(relu_29, 1, 3)

        # pd_op.matmul: (128x102xf32) <- (128x1280xf32, 1280x102xf32)
        matmul_0 = paddle._C_ops.matmul(flatten_0, parameter_1, False, False)
        del parameter_1

        # pd_op.add: (128x102xf32) <- (128x102xf32, 102xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_0)
        del (
            add_11,
            add_13,
            add_15,
            add_17,
            add_19,
            add_21,
            add_23,
            add_5,
            add_7,
            add_9,
            batch_norm__1,
            batch_norm__10,
            batch_norm__100,
            batch_norm__101,
            batch_norm__102,
            batch_norm__103,
            batch_norm__104,
            batch_norm__105,
            batch_norm__106,
            batch_norm__107,
            batch_norm__109,
            batch_norm__11,
            batch_norm__110,
            batch_norm__111,
            batch_norm__112,
            batch_norm__113,
            batch_norm__115,
            batch_norm__116,
            batch_norm__117,
            batch_norm__118,
            batch_norm__119,
            batch_norm__120,
            batch_norm__121,
            batch_norm__122,
            batch_norm__123,
            batch_norm__124,
            batch_norm__125,
            batch_norm__126,
            batch_norm__127,
            batch_norm__128,
            batch_norm__129,
            batch_norm__13,
            batch_norm__130,
            batch_norm__131,
            batch_norm__133,
            batch_norm__134,
            batch_norm__135,
            batch_norm__136,
            batch_norm__137,
            batch_norm__138,
            batch_norm__139,
            batch_norm__14,
            batch_norm__140,
            batch_norm__141,
            batch_norm__142,
            batch_norm__143,
            batch_norm__144,
            batch_norm__145,
            batch_norm__146,
            batch_norm__147,
            batch_norm__148,
            batch_norm__149,
            batch_norm__15,
            batch_norm__151,
            batch_norm__152,
            batch_norm__153,
            batch_norm__154,
            batch_norm__155,
            batch_norm__157,
            batch_norm__158,
            batch_norm__159,
            batch_norm__16,
            batch_norm__160,
            batch_norm__161,
            batch_norm__162,
            batch_norm__163,
            batch_norm__164,
            batch_norm__165,
            batch_norm__166,
            batch_norm__167,
            batch_norm__168,
            batch_norm__169,
            batch_norm__17,
            batch_norm__170,
            batch_norm__171,
            batch_norm__172,
            batch_norm__173,
            batch_norm__175,
            batch_norm__176,
            batch_norm__177,
            batch_norm__178,
            batch_norm__179,
            batch_norm__181,
            batch_norm__182,
            batch_norm__183,
            batch_norm__184,
            batch_norm__185,
            batch_norm__186,
            batch_norm__187,
            batch_norm__188,
            batch_norm__189,
            batch_norm__19,
            batch_norm__190,
            batch_norm__191,
            batch_norm__193,
            batch_norm__194,
            batch_norm__195,
            batch_norm__196,
            batch_norm__197,
            batch_norm__199,
            batch_norm__2,
            batch_norm__20,
            batch_norm__200,
            batch_norm__201,
            batch_norm__202,
            batch_norm__203,
            batch_norm__204,
            batch_norm__205,
            batch_norm__206,
            batch_norm__207,
            batch_norm__208,
            batch_norm__209,
            batch_norm__21,
            batch_norm__211,
            batch_norm__212,
            batch_norm__213,
            batch_norm__214,
            batch_norm__215,
            batch_norm__217,
            batch_norm__218,
            batch_norm__219,
            batch_norm__22,
            batch_norm__220,
            batch_norm__221,
            batch_norm__222,
            batch_norm__223,
            batch_norm__224,
            batch_norm__225,
            batch_norm__226,
            batch_norm__227,
            batch_norm__229,
            batch_norm__23,
            batch_norm__230,
            batch_norm__231,
            batch_norm__232,
            batch_norm__233,
            batch_norm__235,
            batch_norm__236,
            batch_norm__237,
            batch_norm__238,
            batch_norm__239,
            batch_norm__240,
            batch_norm__241,
            batch_norm__242,
            batch_norm__243,
            batch_norm__244,
            batch_norm__245,
            batch_norm__247,
            batch_norm__248,
            batch_norm__249,
            batch_norm__25,
            batch_norm__250,
            batch_norm__251,
            batch_norm__253,
            batch_norm__254,
            batch_norm__255,
            batch_norm__256,
            batch_norm__257,
            batch_norm__258,
            batch_norm__259,
            batch_norm__26,
            batch_norm__260,
            batch_norm__261,
            batch_norm__262,
            batch_norm__263,
            batch_norm__265,
            batch_norm__266,
            batch_norm__267,
            batch_norm__268,
            batch_norm__269,
            batch_norm__27,
            batch_norm__271,
            batch_norm__272,
            batch_norm__273,
            batch_norm__274,
            batch_norm__275,
            batch_norm__28,
            batch_norm__29,
            batch_norm__3,
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__4,
            batch_norm__40,
            batch_norm__41,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
            batch_norm__48,
            batch_norm__49,
            batch_norm__5,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
            batch_norm__66,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            batch_norm__7,
            batch_norm__70,
            batch_norm__71,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
            batch_norm__79,
            batch_norm__8,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
            batch_norm__83,
            batch_norm__84,
            batch_norm__85,
            batch_norm__86,
            batch_norm__87,
            batch_norm__88,
            batch_norm__89,
            batch_norm__9,
            batch_norm__91,
            batch_norm__92,
            batch_norm__93,
            batch_norm__94,
            batch_norm__95,
            batch_norm__97,
            batch_norm__98,
            batch_norm__99,
            conv2d_0,
            conv2d_1,
            conv2d_10,
            conv2d_11,
            conv2d_12,
            conv2d_13,
            conv2d_14,
            conv2d_15,
            conv2d_16,
            conv2d_17,
            conv2d_18,
            conv2d_19,
            conv2d_2,
            conv2d_20,
            conv2d_21,
            conv2d_22,
            conv2d_23,
            conv2d_24,
            conv2d_25,
            conv2d_26,
            conv2d_27,
            conv2d_28,
            conv2d_29,
            conv2d_3,
            conv2d_30,
            conv2d_4,
            conv2d_5,
            conv2d_6,
            conv2d_7,
            conv2d_8,
            conv2d_9,
            depthwise_conv2d_0,
            depthwise_conv2d_1,
            depthwise_conv2d_10,
            depthwise_conv2d_11,
            depthwise_conv2d_12,
            depthwise_conv2d_13,
            depthwise_conv2d_14,
            depthwise_conv2d_2,
            depthwise_conv2d_3,
            depthwise_conv2d_4,
            depthwise_conv2d_5,
            depthwise_conv2d_6,
            depthwise_conv2d_7,
            depthwise_conv2d_8,
            depthwise_conv2d_9,
            divide_0,
            divide_1,
            divide_10,
            divide_11,
            divide_12,
            divide_13,
            divide_2,
            divide_3,
            divide_4,
            divide_5,
            divide_6,
            divide_7,
            divide_8,
            divide_9,
            flatten_0,
            floor_0,
            floor_1,
            floor_10,
            floor_11,
            floor_12,
            floor_13,
            floor_2,
            floor_3,
            floor_4,
            floor_5,
            floor_6,
            floor_7,
            floor_8,
            floor_9,
            full_0,
            full_10,
            full_11,
            full_12,
            full_13,
            full_14,
            full_15,
            full_3,
            full_4,
            full_5,
            full_6,
            full_7,
            full_8,
            full_9,
            full_int_array_1,
            matmul_0,
            multiply_0,
            multiply_1,
            multiply_10,
            multiply_11,
            multiply_12,
            multiply_13,
            multiply_2,
            multiply_3,
            multiply_4,
            multiply_5,
            multiply_6,
            multiply_7,
            multiply_8,
            multiply_9,
            parameter_0,
            pool2d_0,
            relu_0,
            relu_1,
            relu_10,
            relu_11,
            relu_12,
            relu_13,
            relu_14,
            relu_15,
            relu_16,
            relu_17,
            relu_18,
            relu_19,
            relu_2,
            relu_20,
            relu_21,
            relu_22,
            relu_23,
            relu_24,
            relu_25,
            relu_26,
            relu_27,
            relu_28,
            relu_29,
            relu_3,
            relu_4,
            relu_5,
            relu_6,
            relu_7,
            relu_8,
            relu_9,
        )

        return add_0
