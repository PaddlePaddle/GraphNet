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
        data_0,
    ):
        # pd_op.conv2d: (-1x144x56x56xf32) <- (-1x3x224x224xf32, 144x3x4x4xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_218, [4, 4], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_0, parameter_218

        # pd_op.batch_norm_: (-1x144x56x56xf32, 144xf32, 144xf32, 144xf32, 144xf32, -1xui8) <- (-1x144x56x56xf32, 144xf32, 144xf32, 144xf32, 144xf32)
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
                parameter_217,
                parameter_216,
                parameter_215,
                parameter_214,
                True,
                float("0.1"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_0, parameter_214, parameter_215, parameter_216, parameter_217

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [36, 108]

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split: ([-1x36x56x56xf32, -1x108x56x56xf32]) <- (-1x144x56x56xf32, 2xi64, 1xi32)
        split_0 = paddle._C_ops.split(batch_norm__0, full_int_array_0, full_0)

        # builtin.split: (-1x36x56x56xf32, -1x108x56x56xf32) <- ([-1x36x56x56xf32, -1x108x56x56xf32])
        (
            split_1,
            split_2,
        ) = split_0
        del split_0

        # pd_op.conv2d: (-1x36x56x56xf32) <- (-1x36x56x56xf32, 36x36x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            split_1, parameter_213, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_213, split_1

        # builtin.combine: ([-1x36x56x56xf32, -1x108x56x56xf32]) <- (-1x36x56x56xf32, -1x108x56x56xf32)
        combine_0 = [conv2d_1, split_2]
        del conv2d_1, split_2

        # pd_op.concat: (-1x144x56x56xf32) <- ([-1x36x56x56xf32, -1x108x56x56xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)
        del combine_0

        # pd_op.conv2d: (-1x288x56x56xf32) <- (-1x144x56x56xf32, 288x144x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            concat_0, parameter_212, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_0, parameter_212

        # pd_op.batch_norm_: (-1x288x56x56xf32, 288xf32, 288xf32, 288xf32, 288xf32, -1xui8) <- (-1x288x56x56xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        (
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            batch_norm__10,
            batch_norm__11,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_2,
                parameter_211,
                parameter_210,
                parameter_209,
                parameter_208,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_2, parameter_208, parameter_209, parameter_210, parameter_211

        # pd_op.relu: (-1x288x56x56xf32) <- (-1x288x56x56xf32)
        relu_0 = paddle._C_ops.relu(batch_norm__6)
        del batch_norm__6

        # pd_op.conv2d: (-1x144x56x56xf32) <- (-1x288x56x56xf32, 144x288x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            relu_0, parameter_207, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_207, relu_0

        # pd_op.add: (-1x144x56x56xf32) <- (-1x144x56x56xf32, -1x144x56x56xf32)
        add_1 = paddle._C_ops.add(batch_norm__0, conv2d_3)
        del batch_norm__0, conv2d_3

        # pd_op.split: ([-1x36x56x56xf32, -1x108x56x56xf32]) <- (-1x144x56x56xf32, 2xi64, 1xi32)
        split_3 = paddle._C_ops.split(add_1, full_int_array_0, full_0)

        # builtin.split: (-1x36x56x56xf32, -1x108x56x56xf32) <- ([-1x36x56x56xf32, -1x108x56x56xf32])
        (
            split_4,
            split_5,
        ) = split_3
        del split_3

        # pd_op.conv2d: (-1x36x56x56xf32) <- (-1x36x56x56xf32, 36x36x3x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            split_4, parameter_206, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_206, split_4

        # builtin.combine: ([-1x36x56x56xf32, -1x108x56x56xf32]) <- (-1x36x56x56xf32, -1x108x56x56xf32)
        combine_1 = [conv2d_4, split_5]
        del conv2d_4, split_5

        # pd_op.concat: (-1x144x56x56xf32) <- ([-1x36x56x56xf32, -1x108x56x56xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_0)
        del combine_1

        # pd_op.conv2d: (-1x288x56x56xf32) <- (-1x144x56x56xf32, 288x144x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            concat_1, parameter_205, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_1, parameter_205

        # pd_op.batch_norm_: (-1x288x56x56xf32, 288xf32, 288xf32, 288xf32, 288xf32, -1xui8) <- (-1x288x56x56xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        (
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            batch_norm__16,
            batch_norm__17,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_5,
                parameter_204,
                parameter_203,
                parameter_202,
                parameter_201,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_5, parameter_201, parameter_202, parameter_203, parameter_204

        # pd_op.relu: (-1x288x56x56xf32) <- (-1x288x56x56xf32)
        relu_1 = paddle._C_ops.relu(batch_norm__12)
        del batch_norm__12

        # pd_op.conv2d: (-1x144x56x56xf32) <- (-1x288x56x56xf32, 144x288x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            relu_1, parameter_200, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_200, relu_1

        # pd_op.add: (-1x144x56x56xf32) <- (-1x144x56x56xf32, -1x144x56x56xf32)
        add_2 = paddle._C_ops.add(add_1, conv2d_6)
        del add_1, conv2d_6

        # pd_op.split: ([-1x36x56x56xf32, -1x108x56x56xf32]) <- (-1x144x56x56xf32, 2xi64, 1xi32)
        split_6 = paddle._C_ops.split(add_2, full_int_array_0, full_0)
        del full_int_array_0

        # builtin.split: (-1x36x56x56xf32, -1x108x56x56xf32) <- ([-1x36x56x56xf32, -1x108x56x56xf32])
        (
            split_7,
            split_8,
        ) = split_6
        del split_6

        # pd_op.conv2d: (-1x36x56x56xf32) <- (-1x36x56x56xf32, 36x36x3x3xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            split_7, parameter_199, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_199, split_7

        # builtin.combine: ([-1x36x56x56xf32, -1x108x56x56xf32]) <- (-1x36x56x56xf32, -1x108x56x56xf32)
        combine_2 = [conv2d_7, split_8]
        del conv2d_7, split_8

        # pd_op.concat: (-1x144x56x56xf32) <- ([-1x36x56x56xf32, -1x108x56x56xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_0)
        del combine_2

        # pd_op.conv2d: (-1x288x56x56xf32) <- (-1x144x56x56xf32, 288x144x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            concat_2, parameter_198, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_2, parameter_198

        # pd_op.batch_norm_: (-1x288x56x56xf32, 288xf32, 288xf32, 288xf32, 288xf32, -1xui8) <- (-1x288x56x56xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        (
            batch_norm__18,
            batch_norm__19,
            batch_norm__20,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_8,
                parameter_197,
                parameter_196,
                parameter_195,
                parameter_194,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_8, parameter_194, parameter_195, parameter_196, parameter_197

        # pd_op.relu: (-1x288x56x56xf32) <- (-1x288x56x56xf32)
        relu_2 = paddle._C_ops.relu(batch_norm__18)
        del batch_norm__18

        # pd_op.conv2d: (-1x144x56x56xf32) <- (-1x288x56x56xf32, 144x288x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            relu_2, parameter_193, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_193, relu_2

        # pd_op.add: (-1x144x56x56xf32) <- (-1x144x56x56xf32, -1x144x56x56xf32)
        add_3 = paddle._C_ops.add(add_2, conv2d_9)
        del add_2, conv2d_9

        # pd_op.conv2d: (-1x288x28x28xf32) <- (-1x144x56x56xf32, 288x144x2x2xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            add_3, parameter_192, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_3, parameter_192

        # pd_op.batch_norm_: (-1x288x28x28xf32, 288xf32, 288xf32, 288xf32, 288xf32, -1xui8) <- (-1x288x28x28xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        (
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_10,
                parameter_191,
                parameter_190,
                parameter_189,
                parameter_188,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_10, parameter_188, parameter_189, parameter_190, parameter_191

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [72, 216]

        # pd_op.split: ([-1x72x28x28xf32, -1x216x28x28xf32]) <- (-1x288x28x28xf32, 2xi64, 1xi32)
        split_9 = paddle._C_ops.split(batch_norm__24, full_int_array_1, full_0)

        # builtin.split: (-1x72x28x28xf32, -1x216x28x28xf32) <- ([-1x72x28x28xf32, -1x216x28x28xf32])
        (
            split_10,
            split_11,
        ) = split_9
        del split_9

        # pd_op.conv2d: (-1x72x28x28xf32) <- (-1x72x28x28xf32, 72x72x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            split_10, parameter_187, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_187, split_10

        # builtin.combine: ([-1x72x28x28xf32, -1x216x28x28xf32]) <- (-1x72x28x28xf32, -1x216x28x28xf32)
        combine_3 = [conv2d_11, split_11]
        del conv2d_11, split_11

        # pd_op.concat: (-1x288x28x28xf32) <- ([-1x72x28x28xf32, -1x216x28x28xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_3, full_0)
        del combine_3

        # pd_op.conv2d: (-1x576x28x28xf32) <- (-1x288x28x28xf32, 576x288x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            concat_3, parameter_186, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_3, parameter_186

        # pd_op.batch_norm_: (-1x576x28x28xf32, 576xf32, 576xf32, 576xf32, 576xf32, -1xui8) <- (-1x576x28x28xf32, 576xf32, 576xf32, 576xf32, 576xf32)
        (
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_12,
                parameter_185,
                parameter_184,
                parameter_183,
                parameter_182,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_12, parameter_182, parameter_183, parameter_184, parameter_185

        # pd_op.relu: (-1x576x28x28xf32) <- (-1x576x28x28xf32)
        relu_3 = paddle._C_ops.relu(batch_norm__30)
        del batch_norm__30

        # pd_op.conv2d: (-1x288x28x28xf32) <- (-1x576x28x28xf32, 288x576x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            relu_3, parameter_181, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_181, relu_3

        # pd_op.add: (-1x288x28x28xf32) <- (-1x288x28x28xf32, -1x288x28x28xf32)
        add_4 = paddle._C_ops.add(batch_norm__24, conv2d_13)
        del batch_norm__24, conv2d_13

        # pd_op.split: ([-1x72x28x28xf32, -1x216x28x28xf32]) <- (-1x288x28x28xf32, 2xi64, 1xi32)
        split_12 = paddle._C_ops.split(add_4, full_int_array_1, full_0)

        # builtin.split: (-1x72x28x28xf32, -1x216x28x28xf32) <- ([-1x72x28x28xf32, -1x216x28x28xf32])
        (
            split_13,
            split_14,
        ) = split_12
        del split_12

        # pd_op.conv2d: (-1x72x28x28xf32) <- (-1x72x28x28xf32, 72x72x3x3xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            split_13, parameter_180, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_180, split_13

        # builtin.combine: ([-1x72x28x28xf32, -1x216x28x28xf32]) <- (-1x72x28x28xf32, -1x216x28x28xf32)
        combine_4 = [conv2d_14, split_14]
        del conv2d_14, split_14

        # pd_op.concat: (-1x288x28x28xf32) <- ([-1x72x28x28xf32, -1x216x28x28xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_4, full_0)
        del combine_4

        # pd_op.conv2d: (-1x576x28x28xf32) <- (-1x288x28x28xf32, 576x288x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            concat_4, parameter_179, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_4, parameter_179

        # pd_op.batch_norm_: (-1x576x28x28xf32, 576xf32, 576xf32, 576xf32, 576xf32, -1xui8) <- (-1x576x28x28xf32, 576xf32, 576xf32, 576xf32, 576xf32)
        (
            batch_norm__36,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_15,
                parameter_178,
                parameter_177,
                parameter_176,
                parameter_175,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_15, parameter_175, parameter_176, parameter_177, parameter_178

        # pd_op.relu: (-1x576x28x28xf32) <- (-1x576x28x28xf32)
        relu_4 = paddle._C_ops.relu(batch_norm__36)
        del batch_norm__36

        # pd_op.conv2d: (-1x288x28x28xf32) <- (-1x576x28x28xf32, 288x576x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            relu_4, parameter_174, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_174, relu_4

        # pd_op.add: (-1x288x28x28xf32) <- (-1x288x28x28xf32, -1x288x28x28xf32)
        add_5 = paddle._C_ops.add(add_4, conv2d_16)
        del add_4, conv2d_16

        # pd_op.split: ([-1x72x28x28xf32, -1x216x28x28xf32]) <- (-1x288x28x28xf32, 2xi64, 1xi32)
        split_15 = paddle._C_ops.split(add_5, full_int_array_1, full_0)

        # builtin.split: (-1x72x28x28xf32, -1x216x28x28xf32) <- ([-1x72x28x28xf32, -1x216x28x28xf32])
        (
            split_16,
            split_17,
        ) = split_15
        del split_15

        # pd_op.conv2d: (-1x72x28x28xf32) <- (-1x72x28x28xf32, 72x72x3x3xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            split_16, parameter_173, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_173, split_16

        # builtin.combine: ([-1x72x28x28xf32, -1x216x28x28xf32]) <- (-1x72x28x28xf32, -1x216x28x28xf32)
        combine_5 = [conv2d_17, split_17]
        del conv2d_17, split_17

        # pd_op.concat: (-1x288x28x28xf32) <- ([-1x72x28x28xf32, -1x216x28x28xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_5, full_0)
        del combine_5

        # pd_op.conv2d: (-1x576x28x28xf32) <- (-1x288x28x28xf32, 576x288x1x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            concat_5, parameter_172, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_5, parameter_172

        # pd_op.batch_norm_: (-1x576x28x28xf32, 576xf32, 576xf32, 576xf32, 576xf32, -1xui8) <- (-1x576x28x28xf32, 576xf32, 576xf32, 576xf32, 576xf32)
        (
            batch_norm__42,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_18,
                parameter_171,
                parameter_170,
                parameter_169,
                parameter_168,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_18, parameter_168, parameter_169, parameter_170, parameter_171

        # pd_op.relu: (-1x576x28x28xf32) <- (-1x576x28x28xf32)
        relu_5 = paddle._C_ops.relu(batch_norm__42)
        del batch_norm__42

        # pd_op.conv2d: (-1x288x28x28xf32) <- (-1x576x28x28xf32, 288x576x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            relu_5, parameter_167, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_167, relu_5

        # pd_op.add: (-1x288x28x28xf32) <- (-1x288x28x28xf32, -1x288x28x28xf32)
        add_6 = paddle._C_ops.add(add_5, conv2d_19)
        del add_5, conv2d_19

        # pd_op.split: ([-1x72x28x28xf32, -1x216x28x28xf32]) <- (-1x288x28x28xf32, 2xi64, 1xi32)
        split_18 = paddle._C_ops.split(add_6, full_int_array_1, full_0)
        del full_int_array_1

        # builtin.split: (-1x72x28x28xf32, -1x216x28x28xf32) <- ([-1x72x28x28xf32, -1x216x28x28xf32])
        (
            split_19,
            split_20,
        ) = split_18
        del split_18

        # pd_op.conv2d: (-1x72x28x28xf32) <- (-1x72x28x28xf32, 72x72x3x3xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            split_19, parameter_166, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_166, split_19

        # builtin.combine: ([-1x72x28x28xf32, -1x216x28x28xf32]) <- (-1x72x28x28xf32, -1x216x28x28xf32)
        combine_6 = [conv2d_20, split_20]
        del conv2d_20, split_20

        # pd_op.concat: (-1x288x28x28xf32) <- ([-1x72x28x28xf32, -1x216x28x28xf32], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_6, full_0)
        del combine_6

        # pd_op.conv2d: (-1x576x28x28xf32) <- (-1x288x28x28xf32, 576x288x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            concat_6, parameter_165, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_6, parameter_165

        # pd_op.batch_norm_: (-1x576x28x28xf32, 576xf32, 576xf32, 576xf32, 576xf32, -1xui8) <- (-1x576x28x28xf32, 576xf32, 576xf32, 576xf32, 576xf32)
        (
            batch_norm__48,
            batch_norm__49,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_21,
                parameter_164,
                parameter_163,
                parameter_162,
                parameter_161,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_21, parameter_161, parameter_162, parameter_163, parameter_164

        # pd_op.relu: (-1x576x28x28xf32) <- (-1x576x28x28xf32)
        relu_6 = paddle._C_ops.relu(batch_norm__48)
        del batch_norm__48

        # pd_op.conv2d: (-1x288x28x28xf32) <- (-1x576x28x28xf32, 288x576x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(
            relu_6, parameter_160, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_160, relu_6

        # pd_op.add: (-1x288x28x28xf32) <- (-1x288x28x28xf32, -1x288x28x28xf32)
        add_7 = paddle._C_ops.add(add_6, conv2d_22)
        del add_6, conv2d_22

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x288x28x28xf32, 576x288x2x2xf32)
        conv2d_23 = paddle._C_ops.conv2d(
            add_7, parameter_159, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_7, parameter_159

        # pd_op.batch_norm_: (-1x576x14x14xf32, 576xf32, 576xf32, 576xf32, 576xf32, -1xui8) <- (-1x576x14x14xf32, 576xf32, 576xf32, 576xf32, 576xf32)
        (
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_23,
                parameter_158,
                parameter_157,
                parameter_156,
                parameter_155,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_23, parameter_155, parameter_156, parameter_157, parameter_158

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [144, 432]

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        split_21 = paddle._C_ops.split(batch_norm__54, full_int_array_2, full_0)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            split_22,
            split_23,
        ) = split_21
        del split_21

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        conv2d_24 = paddle._C_ops.conv2d(
            split_22, parameter_154, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_154, split_22

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        combine_7 = [conv2d_24, split_23]
        del conv2d_24, split_23

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        concat_7 = paddle._C_ops.concat(combine_7, full_0)
        del combine_7

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(
            concat_7, parameter_153, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_7, parameter_153

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_25,
                parameter_152,
                parameter_151,
                parameter_150,
                parameter_149,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_25, parameter_149, parameter_150, parameter_151, parameter_152

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        relu_7 = paddle._C_ops.relu(batch_norm__60)
        del batch_norm__60

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(
            relu_7, parameter_148, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_148, relu_7

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        add_8 = paddle._C_ops.add(batch_norm__54, conv2d_26)
        del batch_norm__54, conv2d_26

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        split_24 = paddle._C_ops.split(add_8, full_int_array_2, full_0)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            split_25,
            split_26,
        ) = split_24
        del split_24

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        conv2d_27 = paddle._C_ops.conv2d(
            split_25, parameter_147, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_147, split_25

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        combine_8 = [conv2d_27, split_26]
        del conv2d_27, split_26

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        concat_8 = paddle._C_ops.concat(combine_8, full_0)
        del combine_8

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(
            concat_8, parameter_146, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_8, parameter_146

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__66,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            batch_norm__70,
            batch_norm__71,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_28,
                parameter_145,
                parameter_144,
                parameter_143,
                parameter_142,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_28, parameter_142, parameter_143, parameter_144, parameter_145

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        relu_8 = paddle._C_ops.relu(batch_norm__66)
        del batch_norm__66

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(
            relu_8, parameter_141, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_141, relu_8

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        add_9 = paddle._C_ops.add(add_8, conv2d_29)
        del add_8, conv2d_29

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        split_27 = paddle._C_ops.split(add_9, full_int_array_2, full_0)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            split_28,
            split_29,
        ) = split_27
        del split_27

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        conv2d_30 = paddle._C_ops.conv2d(
            split_28, parameter_140, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_140, split_28

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        combine_9 = [conv2d_30, split_29]
        del conv2d_30, split_29

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        concat_9 = paddle._C_ops.concat(combine_9, full_0)
        del combine_9

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        conv2d_31 = paddle._C_ops.conv2d(
            concat_9, parameter_139, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_9, parameter_139

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__72,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_31,
                parameter_138,
                parameter_137,
                parameter_136,
                parameter_135,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_31, parameter_135, parameter_136, parameter_137, parameter_138

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        relu_9 = paddle._C_ops.relu(batch_norm__72)
        del batch_norm__72

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        conv2d_32 = paddle._C_ops.conv2d(
            relu_9, parameter_134, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_134, relu_9

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        add_10 = paddle._C_ops.add(add_9, conv2d_32)
        del add_9, conv2d_32

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        split_30 = paddle._C_ops.split(add_10, full_int_array_2, full_0)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            split_31,
            split_32,
        ) = split_30
        del split_30

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        conv2d_33 = paddle._C_ops.conv2d(
            split_31, parameter_133, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_133, split_31

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        combine_10 = [conv2d_33, split_32]
        del conv2d_33, split_32

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        concat_10 = paddle._C_ops.concat(combine_10, full_0)
        del combine_10

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        conv2d_34 = paddle._C_ops.conv2d(
            concat_10, parameter_132, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_10, parameter_132

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__78,
            batch_norm__79,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
            batch_norm__83,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_34,
                parameter_131,
                parameter_130,
                parameter_129,
                parameter_128,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_34, parameter_128, parameter_129, parameter_130, parameter_131

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        relu_10 = paddle._C_ops.relu(batch_norm__78)
        del batch_norm__78

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        conv2d_35 = paddle._C_ops.conv2d(
            relu_10, parameter_127, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_127, relu_10

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        add_11 = paddle._C_ops.add(add_10, conv2d_35)
        del add_10, conv2d_35

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        split_33 = paddle._C_ops.split(add_11, full_int_array_2, full_0)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            split_34,
            split_35,
        ) = split_33
        del split_33

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        conv2d_36 = paddle._C_ops.conv2d(
            split_34, parameter_126, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_126, split_34

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        combine_11 = [conv2d_36, split_35]
        del conv2d_36, split_35

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        concat_11 = paddle._C_ops.concat(combine_11, full_0)
        del combine_11

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        conv2d_37 = paddle._C_ops.conv2d(
            concat_11, parameter_125, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_11, parameter_125

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__84,
            batch_norm__85,
            batch_norm__86,
            batch_norm__87,
            batch_norm__88,
            batch_norm__89,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_37,
                parameter_124,
                parameter_123,
                parameter_122,
                parameter_121,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_37, parameter_121, parameter_122, parameter_123, parameter_124

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        relu_11 = paddle._C_ops.relu(batch_norm__84)
        del batch_norm__84

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        conv2d_38 = paddle._C_ops.conv2d(
            relu_11, parameter_120, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_120, relu_11

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        add_12 = paddle._C_ops.add(add_11, conv2d_38)
        del add_11, conv2d_38

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        split_36 = paddle._C_ops.split(add_12, full_int_array_2, full_0)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            split_37,
            split_38,
        ) = split_36
        del split_36

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        conv2d_39 = paddle._C_ops.conv2d(
            split_37, parameter_119, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_119, split_37

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        combine_12 = [conv2d_39, split_38]
        del conv2d_39, split_38

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        concat_12 = paddle._C_ops.concat(combine_12, full_0)
        del combine_12

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        conv2d_40 = paddle._C_ops.conv2d(
            concat_12, parameter_118, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_12, parameter_118

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__90,
            batch_norm__91,
            batch_norm__92,
            batch_norm__93,
            batch_norm__94,
            batch_norm__95,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_40,
                parameter_117,
                parameter_116,
                parameter_115,
                parameter_114,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_40, parameter_114, parameter_115, parameter_116, parameter_117

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        relu_12 = paddle._C_ops.relu(batch_norm__90)
        del batch_norm__90

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        conv2d_41 = paddle._C_ops.conv2d(
            relu_12, parameter_113, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_113, relu_12

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        add_13 = paddle._C_ops.add(add_12, conv2d_41)
        del add_12, conv2d_41

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        split_39 = paddle._C_ops.split(add_13, full_int_array_2, full_0)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            split_40,
            split_41,
        ) = split_39
        del split_39

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        conv2d_42 = paddle._C_ops.conv2d(
            split_40, parameter_112, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_112, split_40

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        combine_13 = [conv2d_42, split_41]
        del conv2d_42, split_41

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        concat_13 = paddle._C_ops.concat(combine_13, full_0)
        del combine_13

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        conv2d_43 = paddle._C_ops.conv2d(
            concat_13, parameter_111, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_13, parameter_111

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__96,
            batch_norm__97,
            batch_norm__98,
            batch_norm__99,
            batch_norm__100,
            batch_norm__101,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_43,
                parameter_110,
                parameter_109,
                parameter_108,
                parameter_107,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_43, parameter_107, parameter_108, parameter_109, parameter_110

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        relu_13 = paddle._C_ops.relu(batch_norm__96)
        del batch_norm__96

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        conv2d_44 = paddle._C_ops.conv2d(
            relu_13, parameter_106, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_106, relu_13

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        add_14 = paddle._C_ops.add(add_13, conv2d_44)
        del add_13, conv2d_44

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        split_42 = paddle._C_ops.split(add_14, full_int_array_2, full_0)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            split_43,
            split_44,
        ) = split_42
        del split_42

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        conv2d_45 = paddle._C_ops.conv2d(
            split_43, parameter_105, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_105, split_43

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        combine_14 = [conv2d_45, split_44]
        del conv2d_45, split_44

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        concat_14 = paddle._C_ops.concat(combine_14, full_0)
        del combine_14

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        conv2d_46 = paddle._C_ops.conv2d(
            concat_14, parameter_104, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_14, parameter_104

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__102,
            batch_norm__103,
            batch_norm__104,
            batch_norm__105,
            batch_norm__106,
            batch_norm__107,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_46,
                parameter_103,
                parameter_102,
                parameter_101,
                parameter_100,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_46, parameter_100, parameter_101, parameter_102, parameter_103

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        relu_14 = paddle._C_ops.relu(batch_norm__102)
        del batch_norm__102

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        conv2d_47 = paddle._C_ops.conv2d(
            relu_14, parameter_99, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_99, relu_14

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        add_15 = paddle._C_ops.add(add_14, conv2d_47)
        del add_14, conv2d_47

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        split_45 = paddle._C_ops.split(add_15, full_int_array_2, full_0)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            split_46,
            split_47,
        ) = split_45
        del split_45

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        conv2d_48 = paddle._C_ops.conv2d(
            split_46, parameter_98, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_98, split_46

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        combine_15 = [conv2d_48, split_47]
        del conv2d_48, split_47

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        concat_15 = paddle._C_ops.concat(combine_15, full_0)
        del combine_15

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        conv2d_49 = paddle._C_ops.conv2d(
            concat_15, parameter_97, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_15, parameter_97

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__108,
            batch_norm__109,
            batch_norm__110,
            batch_norm__111,
            batch_norm__112,
            batch_norm__113,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_49,
                parameter_96,
                parameter_95,
                parameter_94,
                parameter_93,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_49, parameter_93, parameter_94, parameter_95, parameter_96

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        relu_15 = paddle._C_ops.relu(batch_norm__108)
        del batch_norm__108

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        conv2d_50 = paddle._C_ops.conv2d(
            relu_15, parameter_92, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_92, relu_15

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        add_16 = paddle._C_ops.add(add_15, conv2d_50)
        del add_15, conv2d_50

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        split_48 = paddle._C_ops.split(add_16, full_int_array_2, full_0)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            split_49,
            split_50,
        ) = split_48
        del split_48

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        conv2d_51 = paddle._C_ops.conv2d(
            split_49, parameter_91, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_91, split_49

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        combine_16 = [conv2d_51, split_50]
        del conv2d_51, split_50

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        concat_16 = paddle._C_ops.concat(combine_16, full_0)
        del combine_16

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        conv2d_52 = paddle._C_ops.conv2d(
            concat_16, parameter_90, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_16, parameter_90

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__114,
            batch_norm__115,
            batch_norm__116,
            batch_norm__117,
            batch_norm__118,
            batch_norm__119,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_52,
                parameter_89,
                parameter_88,
                parameter_87,
                parameter_86,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_52, parameter_86, parameter_87, parameter_88, parameter_89

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        relu_16 = paddle._C_ops.relu(batch_norm__114)
        del batch_norm__114

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        conv2d_53 = paddle._C_ops.conv2d(
            relu_16, parameter_85, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_85, relu_16

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        add_17 = paddle._C_ops.add(add_16, conv2d_53)
        del add_16, conv2d_53

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        split_51 = paddle._C_ops.split(add_17, full_int_array_2, full_0)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            split_52,
            split_53,
        ) = split_51
        del split_51

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        conv2d_54 = paddle._C_ops.conv2d(
            split_52, parameter_84, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_84, split_52

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        combine_17 = [conv2d_54, split_53]
        del conv2d_54, split_53

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        concat_17 = paddle._C_ops.concat(combine_17, full_0)
        del combine_17

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        conv2d_55 = paddle._C_ops.conv2d(
            concat_17, parameter_83, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_17, parameter_83

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__120,
            batch_norm__121,
            batch_norm__122,
            batch_norm__123,
            batch_norm__124,
            batch_norm__125,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_55,
                parameter_82,
                parameter_81,
                parameter_80,
                parameter_79,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_55, parameter_79, parameter_80, parameter_81, parameter_82

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        relu_17 = paddle._C_ops.relu(batch_norm__120)
        del batch_norm__120

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        conv2d_56 = paddle._C_ops.conv2d(
            relu_17, parameter_78, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_78, relu_17

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        add_18 = paddle._C_ops.add(add_17, conv2d_56)
        del add_17, conv2d_56

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        split_54 = paddle._C_ops.split(add_18, full_int_array_2, full_0)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            split_55,
            split_56,
        ) = split_54
        del split_54

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        conv2d_57 = paddle._C_ops.conv2d(
            split_55, parameter_77, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_77, split_55

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        combine_18 = [conv2d_57, split_56]
        del conv2d_57, split_56

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        concat_18 = paddle._C_ops.concat(combine_18, full_0)
        del combine_18

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        conv2d_58 = paddle._C_ops.conv2d(
            concat_18, parameter_76, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_18, parameter_76

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__126,
            batch_norm__127,
            batch_norm__128,
            batch_norm__129,
            batch_norm__130,
            batch_norm__131,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_58,
                parameter_75,
                parameter_74,
                parameter_73,
                parameter_72,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_58, parameter_72, parameter_73, parameter_74, parameter_75

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        relu_18 = paddle._C_ops.relu(batch_norm__126)
        del batch_norm__126

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        conv2d_59 = paddle._C_ops.conv2d(
            relu_18, parameter_71, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_71, relu_18

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        add_19 = paddle._C_ops.add(add_18, conv2d_59)
        del add_18, conv2d_59

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        split_57 = paddle._C_ops.split(add_19, full_int_array_2, full_0)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            split_58,
            split_59,
        ) = split_57
        del split_57

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        conv2d_60 = paddle._C_ops.conv2d(
            split_58, parameter_70, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_70, split_58

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        combine_19 = [conv2d_60, split_59]
        del conv2d_60, split_59

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        concat_19 = paddle._C_ops.concat(combine_19, full_0)
        del combine_19

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        conv2d_61 = paddle._C_ops.conv2d(
            concat_19, parameter_69, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_19, parameter_69

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__132,
            batch_norm__133,
            batch_norm__134,
            batch_norm__135,
            batch_norm__136,
            batch_norm__137,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_61,
                parameter_68,
                parameter_67,
                parameter_66,
                parameter_65,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_61, parameter_65, parameter_66, parameter_67, parameter_68

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        relu_19 = paddle._C_ops.relu(batch_norm__132)
        del batch_norm__132

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        conv2d_62 = paddle._C_ops.conv2d(
            relu_19, parameter_64, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_64, relu_19

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        add_20 = paddle._C_ops.add(add_19, conv2d_62)
        del add_19, conv2d_62

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        split_60 = paddle._C_ops.split(add_20, full_int_array_2, full_0)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            split_61,
            split_62,
        ) = split_60
        del split_60

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        conv2d_63 = paddle._C_ops.conv2d(
            split_61, parameter_63, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_63, split_61

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        combine_20 = [conv2d_63, split_62]
        del conv2d_63, split_62

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        concat_20 = paddle._C_ops.concat(combine_20, full_0)
        del combine_20

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        conv2d_64 = paddle._C_ops.conv2d(
            concat_20, parameter_62, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_20, parameter_62

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__138,
            batch_norm__139,
            batch_norm__140,
            batch_norm__141,
            batch_norm__142,
            batch_norm__143,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_64,
                parameter_61,
                parameter_60,
                parameter_59,
                parameter_58,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_64, parameter_58, parameter_59, parameter_60, parameter_61

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        relu_20 = paddle._C_ops.relu(batch_norm__138)
        del batch_norm__138

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        conv2d_65 = paddle._C_ops.conv2d(
            relu_20, parameter_57, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_57, relu_20

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        add_21 = paddle._C_ops.add(add_20, conv2d_65)
        del add_20, conv2d_65

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        split_63 = paddle._C_ops.split(add_21, full_int_array_2, full_0)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            split_64,
            split_65,
        ) = split_63
        del split_63

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        conv2d_66 = paddle._C_ops.conv2d(
            split_64, parameter_56, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_56, split_64

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        combine_21 = [conv2d_66, split_65]
        del conv2d_66, split_65

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        concat_21 = paddle._C_ops.concat(combine_21, full_0)
        del combine_21

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        conv2d_67 = paddle._C_ops.conv2d(
            concat_21, parameter_55, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_21, parameter_55

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__144,
            batch_norm__145,
            batch_norm__146,
            batch_norm__147,
            batch_norm__148,
            batch_norm__149,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_67,
                parameter_54,
                parameter_53,
                parameter_52,
                parameter_51,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_67, parameter_51, parameter_52, parameter_53, parameter_54

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        relu_21 = paddle._C_ops.relu(batch_norm__144)
        del batch_norm__144

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        conv2d_68 = paddle._C_ops.conv2d(
            relu_21, parameter_50, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_50, relu_21

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        add_22 = paddle._C_ops.add(add_21, conv2d_68)
        del add_21, conv2d_68

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        split_66 = paddle._C_ops.split(add_22, full_int_array_2, full_0)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            split_67,
            split_68,
        ) = split_66
        del split_66

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        conv2d_69 = paddle._C_ops.conv2d(
            split_67, parameter_49, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_49, split_67

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        combine_22 = [conv2d_69, split_68]
        del conv2d_69, split_68

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        concat_22 = paddle._C_ops.concat(combine_22, full_0)
        del combine_22

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        conv2d_70 = paddle._C_ops.conv2d(
            concat_22, parameter_48, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_22, parameter_48

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__150,
            batch_norm__151,
            batch_norm__152,
            batch_norm__153,
            batch_norm__154,
            batch_norm__155,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_70,
                parameter_47,
                parameter_46,
                parameter_45,
                parameter_44,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_70, parameter_44, parameter_45, parameter_46, parameter_47

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        relu_22 = paddle._C_ops.relu(batch_norm__150)
        del batch_norm__150

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        conv2d_71 = paddle._C_ops.conv2d(
            relu_22, parameter_43, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_43, relu_22

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        add_23 = paddle._C_ops.add(add_22, conv2d_71)
        del add_22, conv2d_71

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        split_69 = paddle._C_ops.split(add_23, full_int_array_2, full_0)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            split_70,
            split_71,
        ) = split_69
        del split_69

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        conv2d_72 = paddle._C_ops.conv2d(
            split_70, parameter_42, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_42, split_70

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        combine_23 = [conv2d_72, split_71]
        del conv2d_72, split_71

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        concat_23 = paddle._C_ops.concat(combine_23, full_0)
        del combine_23

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        conv2d_73 = paddle._C_ops.conv2d(
            concat_23, parameter_41, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_23, parameter_41

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__156,
            batch_norm__157,
            batch_norm__158,
            batch_norm__159,
            batch_norm__160,
            batch_norm__161,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_73,
                parameter_40,
                parameter_39,
                parameter_38,
                parameter_37,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_73, parameter_37, parameter_38, parameter_39, parameter_40

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        relu_23 = paddle._C_ops.relu(batch_norm__156)
        del batch_norm__156

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        conv2d_74 = paddle._C_ops.conv2d(
            relu_23, parameter_36, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_36, relu_23

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        add_24 = paddle._C_ops.add(add_23, conv2d_74)
        del add_23, conv2d_74

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        split_72 = paddle._C_ops.split(add_24, full_int_array_2, full_0)
        del full_int_array_2

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            split_73,
            split_74,
        ) = split_72
        del split_72

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        conv2d_75 = paddle._C_ops.conv2d(
            split_73, parameter_35, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_35, split_73

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        combine_24 = [conv2d_75, split_74]
        del conv2d_75, split_74

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        concat_24 = paddle._C_ops.concat(combine_24, full_0)
        del combine_24

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        conv2d_76 = paddle._C_ops.conv2d(
            concat_24, parameter_34, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_24, parameter_34

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__162,
            batch_norm__163,
            batch_norm__164,
            batch_norm__165,
            batch_norm__166,
            batch_norm__167,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_76,
                parameter_33,
                parameter_32,
                parameter_31,
                parameter_30,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_76, parameter_30, parameter_31, parameter_32, parameter_33

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        relu_24 = paddle._C_ops.relu(batch_norm__162)
        del batch_norm__162

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        conv2d_77 = paddle._C_ops.conv2d(
            relu_24, parameter_29, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_29, relu_24

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        add_25 = paddle._C_ops.add(add_24, conv2d_77)
        del add_24, conv2d_77

        # pd_op.conv2d: (-1x1152x7x7xf32) <- (-1x576x14x14xf32, 1152x576x2x2xf32)
        conv2d_78 = paddle._C_ops.conv2d(
            add_25, parameter_28, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_25, parameter_28

        # pd_op.batch_norm_: (-1x1152x7x7xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x7x7xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__168,
            batch_norm__169,
            batch_norm__170,
            batch_norm__171,
            batch_norm__172,
            batch_norm__173,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_78,
                parameter_27,
                parameter_26,
                parameter_25,
                parameter_24,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_78, parameter_24, parameter_25, parameter_26, parameter_27

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [288, 864]

        # pd_op.split: ([-1x288x7x7xf32, -1x864x7x7xf32]) <- (-1x1152x7x7xf32, 2xi64, 1xi32)
        split_75 = paddle._C_ops.split(batch_norm__168, full_int_array_3, full_0)

        # builtin.split: (-1x288x7x7xf32, -1x864x7x7xf32) <- ([-1x288x7x7xf32, -1x864x7x7xf32])
        (
            split_76,
            split_77,
        ) = split_75
        del split_75

        # pd_op.conv2d: (-1x288x7x7xf32) <- (-1x288x7x7xf32, 288x288x3x3xf32)
        conv2d_79 = paddle._C_ops.conv2d(
            split_76, parameter_23, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_23, split_76

        # builtin.combine: ([-1x288x7x7xf32, -1x864x7x7xf32]) <- (-1x288x7x7xf32, -1x864x7x7xf32)
        combine_25 = [conv2d_79, split_77]
        del conv2d_79, split_77

        # pd_op.concat: (-1x1152x7x7xf32) <- ([-1x288x7x7xf32, -1x864x7x7xf32], 1xi32)
        concat_25 = paddle._C_ops.concat(combine_25, full_0)
        del combine_25

        # pd_op.conv2d: (-1x2304x7x7xf32) <- (-1x1152x7x7xf32, 2304x1152x1x1xf32)
        conv2d_80 = paddle._C_ops.conv2d(
            concat_25, parameter_22, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_25, parameter_22

        # pd_op.batch_norm_: (-1x2304x7x7xf32, 2304xf32, 2304xf32, 2304xf32, 2304xf32, -1xui8) <- (-1x2304x7x7xf32, 2304xf32, 2304xf32, 2304xf32, 2304xf32)
        (
            batch_norm__174,
            batch_norm__175,
            batch_norm__176,
            batch_norm__177,
            batch_norm__178,
            batch_norm__179,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_80,
                parameter_21,
                parameter_20,
                parameter_19,
                parameter_18,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_80, parameter_18, parameter_19, parameter_20, parameter_21

        # pd_op.relu: (-1x2304x7x7xf32) <- (-1x2304x7x7xf32)
        relu_25 = paddle._C_ops.relu(batch_norm__174)
        del batch_norm__174

        # pd_op.conv2d: (-1x1152x7x7xf32) <- (-1x2304x7x7xf32, 1152x2304x1x1xf32)
        conv2d_81 = paddle._C_ops.conv2d(
            relu_25, parameter_17, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_17, relu_25

        # pd_op.add: (-1x1152x7x7xf32) <- (-1x1152x7x7xf32, -1x1152x7x7xf32)
        add_26 = paddle._C_ops.add(batch_norm__168, conv2d_81)
        del batch_norm__168, conv2d_81

        # pd_op.split: ([-1x288x7x7xf32, -1x864x7x7xf32]) <- (-1x1152x7x7xf32, 2xi64, 1xi32)
        split_78 = paddle._C_ops.split(add_26, full_int_array_3, full_0)

        # builtin.split: (-1x288x7x7xf32, -1x864x7x7xf32) <- ([-1x288x7x7xf32, -1x864x7x7xf32])
        (
            split_79,
            split_80,
        ) = split_78
        del split_78

        # pd_op.conv2d: (-1x288x7x7xf32) <- (-1x288x7x7xf32, 288x288x3x3xf32)
        conv2d_82 = paddle._C_ops.conv2d(
            split_79, parameter_16, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_16, split_79

        # builtin.combine: ([-1x288x7x7xf32, -1x864x7x7xf32]) <- (-1x288x7x7xf32, -1x864x7x7xf32)
        combine_26 = [conv2d_82, split_80]
        del conv2d_82, split_80

        # pd_op.concat: (-1x1152x7x7xf32) <- ([-1x288x7x7xf32, -1x864x7x7xf32], 1xi32)
        concat_26 = paddle._C_ops.concat(combine_26, full_0)
        del combine_26

        # pd_op.conv2d: (-1x2304x7x7xf32) <- (-1x1152x7x7xf32, 2304x1152x1x1xf32)
        conv2d_83 = paddle._C_ops.conv2d(
            concat_26, parameter_15, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_26, parameter_15

        # pd_op.batch_norm_: (-1x2304x7x7xf32, 2304xf32, 2304xf32, 2304xf32, 2304xf32, -1xui8) <- (-1x2304x7x7xf32, 2304xf32, 2304xf32, 2304xf32, 2304xf32)
        (
            batch_norm__180,
            batch_norm__181,
            batch_norm__182,
            batch_norm__183,
            batch_norm__184,
            batch_norm__185,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_83,
                parameter_14,
                parameter_13,
                parameter_12,
                parameter_11,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_83, parameter_11, parameter_12, parameter_13, parameter_14

        # pd_op.relu: (-1x2304x7x7xf32) <- (-1x2304x7x7xf32)
        relu_26 = paddle._C_ops.relu(batch_norm__180)
        del batch_norm__180

        # pd_op.conv2d: (-1x1152x7x7xf32) <- (-1x2304x7x7xf32, 1152x2304x1x1xf32)
        conv2d_84 = paddle._C_ops.conv2d(
            relu_26, parameter_10, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_10, relu_26

        # pd_op.add: (-1x1152x7x7xf32) <- (-1x1152x7x7xf32, -1x1152x7x7xf32)
        add_27 = paddle._C_ops.add(add_26, conv2d_84)
        del add_26, conv2d_84

        # pd_op.split: ([-1x288x7x7xf32, -1x864x7x7xf32]) <- (-1x1152x7x7xf32, 2xi64, 1xi32)
        split_81 = paddle._C_ops.split(add_27, full_int_array_3, full_0)
        del full_int_array_3

        # builtin.split: (-1x288x7x7xf32, -1x864x7x7xf32) <- ([-1x288x7x7xf32, -1x864x7x7xf32])
        (
            split_82,
            split_83,
        ) = split_81
        del split_81

        # pd_op.conv2d: (-1x288x7x7xf32) <- (-1x288x7x7xf32, 288x288x3x3xf32)
        conv2d_85 = paddle._C_ops.conv2d(
            split_82, parameter_9, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_9, split_82

        # builtin.combine: ([-1x288x7x7xf32, -1x864x7x7xf32]) <- (-1x288x7x7xf32, -1x864x7x7xf32)
        combine_27 = [conv2d_85, split_83]
        del conv2d_85, split_83

        # pd_op.concat: (-1x1152x7x7xf32) <- ([-1x288x7x7xf32, -1x864x7x7xf32], 1xi32)
        concat_27 = paddle._C_ops.concat(combine_27, full_0)
        del combine_27, full_0

        # pd_op.conv2d: (-1x2304x7x7xf32) <- (-1x1152x7x7xf32, 2304x1152x1x1xf32)
        conv2d_86 = paddle._C_ops.conv2d(
            concat_27, parameter_8, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_27, parameter_8

        # pd_op.batch_norm_: (-1x2304x7x7xf32, 2304xf32, 2304xf32, 2304xf32, 2304xf32, -1xui8) <- (-1x2304x7x7xf32, 2304xf32, 2304xf32, 2304xf32, 2304xf32)
        (
            batch_norm__186,
            batch_norm__187,
            batch_norm__188,
            batch_norm__189,
            batch_norm__190,
            batch_norm__191,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_86,
                parameter_7,
                parameter_6,
                parameter_5,
                parameter_4,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_86, parameter_4, parameter_5, parameter_6, parameter_7

        # pd_op.relu: (-1x2304x7x7xf32) <- (-1x2304x7x7xf32)
        relu_27 = paddle._C_ops.relu(batch_norm__186)
        del batch_norm__186

        # pd_op.conv2d: (-1x1152x7x7xf32) <- (-1x2304x7x7xf32, 1152x2304x1x1xf32)
        conv2d_87 = paddle._C_ops.conv2d(
            relu_27, parameter_3, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_3, relu_27

        # pd_op.add: (-1x1152x7x7xf32) <- (-1x1152x7x7xf32, -1x1152x7x7xf32)
        add_28 = paddle._C_ops.add(add_27, conv2d_87)
        del add_27, conv2d_87

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [1, 1]

        # pd_op.pool2d: (-1x1152x1x1xf32) <- (-1x1152x7x7xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            add_28,
            full_int_array_4,
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
        del add_28, full_int_array_4

        # pd_op.conv2d: (-1x1280x1x1xf32) <- (-1x1152x1x1xf32, 1280x1152x1x1xf32)
        conv2d_88 = paddle._C_ops.conv2d(
            pool2d_0, parameter_2, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_2, pool2d_0

        # pd_op.relu: (-1x1280x1x1xf32) <- (-1x1280x1x1xf32)
        relu_28 = paddle._C_ops.relu(conv2d_88)
        del conv2d_88

        # pd_op.flatten: (-1x1280xf32) <- (-1x1280x1x1xf32)
        flatten_0 = paddle._C_ops.flatten(relu_28, 1, 3)
        del relu_28

        # pd_op.matmul: (-1x102xf32) <- (-1x1280xf32, 1280x102xf32)
        matmul_0 = paddle._C_ops.matmul(flatten_0, parameter_1, False, False)
        del flatten_0, parameter_1

        # pd_op.add: (-1x102xf32) <- (-1x102xf32, 102xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_0)
        del matmul_0, parameter_0

        return add_0
