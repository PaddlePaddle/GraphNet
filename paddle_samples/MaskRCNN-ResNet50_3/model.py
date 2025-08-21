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
        data_0,
    ):
        # pd_op.conv2d: (1x64x320x480xf32) <- (1x3x640x960xf32, 64x3x7x7xf32)
        conv2d_32 = paddle._C_ops.conv2d(
            data_0, parameter_214, [2, 2], [3, 3], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x64x320x480xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (1x64x320x480xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__172,
            batch_norm__173,
            batch_norm__174,
            batch_norm__175,
            batch_norm__176,
            batch_norm__177,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_32,
                parameter_213,
                parameter_212,
                parameter_211,
                parameter_210,
                False,
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

        # pd_op.relu: (1x64x320x480xf32) <- (1x64x320x480xf32)
        relu_31 = paddle._C_ops.relu(batch_norm__172)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [3, 3]

        # pd_op.pool2d: (1x64x160x240xf32) <- (1x64x320x480xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            relu_31,
            full_int_array_0,
            [2, 2],
            [1, 1],
            False,
            True,
            "NCHW",
            "max",
            False,
            False,
            "EXPLICIT",
        )

        # pd_op.conv2d: (1x64x160x240xf32) <- (1x64x160x240xf32, 64x64x1x1xf32)
        conv2d_33 = paddle._C_ops.conv2d(
            pool2d_0, parameter_209, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x64x160x240xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (1x64x160x240xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__178,
            batch_norm__179,
            batch_norm__180,
            batch_norm__181,
            batch_norm__182,
            batch_norm__183,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_33,
                parameter_208,
                parameter_207,
                parameter_206,
                parameter_205,
                False,
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

        # pd_op.relu: (1x64x160x240xf32) <- (1x64x160x240xf32)
        relu_32 = paddle._C_ops.relu(batch_norm__178)

        # pd_op.conv2d: (1x64x160x240xf32) <- (1x64x160x240xf32, 64x64x3x3xf32)
        conv2d_34 = paddle._C_ops.conv2d(
            relu_32, parameter_204, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x64x160x240xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (1x64x160x240xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__184,
            batch_norm__185,
            batch_norm__186,
            batch_norm__187,
            batch_norm__188,
            batch_norm__189,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_34,
                parameter_203,
                parameter_202,
                parameter_201,
                parameter_200,
                False,
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

        # pd_op.relu: (1x64x160x240xf32) <- (1x64x160x240xf32)
        relu_33 = paddle._C_ops.relu(batch_norm__184)

        # pd_op.conv2d: (1x256x160x240xf32) <- (1x64x160x240xf32, 256x64x1x1xf32)
        conv2d_35 = paddle._C_ops.conv2d(
            relu_33, parameter_199, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x256x160x240xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x160x240xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__190,
            batch_norm__191,
            batch_norm__192,
            batch_norm__193,
            batch_norm__194,
            batch_norm__195,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_35,
                parameter_198,
                parameter_197,
                parameter_196,
                parameter_195,
                False,
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

        # pd_op.conv2d: (1x256x160x240xf32) <- (1x64x160x240xf32, 256x64x1x1xf32)
        conv2d_36 = paddle._C_ops.conv2d(
            pool2d_0, parameter_194, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x256x160x240xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x160x240xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__196,
            batch_norm__197,
            batch_norm__198,
            batch_norm__199,
            batch_norm__200,
            batch_norm__201,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_36,
                parameter_193,
                parameter_192,
                parameter_191,
                parameter_190,
                False,
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

        # pd_op.add: (1x256x160x240xf32) <- (1x256x160x240xf32, 1x256x160x240xf32)
        add_0 = paddle._C_ops.add(batch_norm__190, batch_norm__196)

        # pd_op.relu: (1x256x160x240xf32) <- (1x256x160x240xf32)
        relu_34 = paddle._C_ops.relu(add_0)

        # pd_op.conv2d: (1x64x160x240xf32) <- (1x256x160x240xf32, 64x256x1x1xf32)
        conv2d_37 = paddle._C_ops.conv2d(
            relu_34, parameter_189, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x64x160x240xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (1x64x160x240xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__202,
            batch_norm__203,
            batch_norm__204,
            batch_norm__205,
            batch_norm__206,
            batch_norm__207,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_37,
                parameter_188,
                parameter_187,
                parameter_186,
                parameter_185,
                False,
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

        # pd_op.relu: (1x64x160x240xf32) <- (1x64x160x240xf32)
        relu_35 = paddle._C_ops.relu(batch_norm__202)

        # pd_op.conv2d: (1x64x160x240xf32) <- (1x64x160x240xf32, 64x64x3x3xf32)
        conv2d_38 = paddle._C_ops.conv2d(
            relu_35, parameter_184, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x64x160x240xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (1x64x160x240xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__208,
            batch_norm__209,
            batch_norm__210,
            batch_norm__211,
            batch_norm__212,
            batch_norm__213,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_38,
                parameter_183,
                parameter_182,
                parameter_181,
                parameter_180,
                False,
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

        # pd_op.relu: (1x64x160x240xf32) <- (1x64x160x240xf32)
        relu_36 = paddle._C_ops.relu(batch_norm__208)

        # pd_op.conv2d: (1x256x160x240xf32) <- (1x64x160x240xf32, 256x64x1x1xf32)
        conv2d_39 = paddle._C_ops.conv2d(
            relu_36, parameter_179, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x256x160x240xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x160x240xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__214,
            batch_norm__215,
            batch_norm__216,
            batch_norm__217,
            batch_norm__218,
            batch_norm__219,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_39,
                parameter_178,
                parameter_177,
                parameter_176,
                parameter_175,
                False,
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

        # pd_op.add: (1x256x160x240xf32) <- (1x256x160x240xf32, 1x256x160x240xf32)
        add_1 = paddle._C_ops.add(batch_norm__214, relu_34)

        # pd_op.relu: (1x256x160x240xf32) <- (1x256x160x240xf32)
        relu_37 = paddle._C_ops.relu(add_1)

        # pd_op.conv2d: (1x64x160x240xf32) <- (1x256x160x240xf32, 64x256x1x1xf32)
        conv2d_40 = paddle._C_ops.conv2d(
            relu_37, parameter_174, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x64x160x240xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (1x64x160x240xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__220,
            batch_norm__221,
            batch_norm__222,
            batch_norm__223,
            batch_norm__224,
            batch_norm__225,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_40,
                parameter_173,
                parameter_172,
                parameter_171,
                parameter_170,
                False,
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

        # pd_op.relu: (1x64x160x240xf32) <- (1x64x160x240xf32)
        relu_38 = paddle._C_ops.relu(batch_norm__220)

        # pd_op.conv2d: (1x64x160x240xf32) <- (1x64x160x240xf32, 64x64x3x3xf32)
        conv2d_41 = paddle._C_ops.conv2d(
            relu_38, parameter_169, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x64x160x240xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (1x64x160x240xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__226,
            batch_norm__227,
            batch_norm__228,
            batch_norm__229,
            batch_norm__230,
            batch_norm__231,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_41,
                parameter_168,
                parameter_167,
                parameter_166,
                parameter_165,
                False,
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

        # pd_op.relu: (1x64x160x240xf32) <- (1x64x160x240xf32)
        relu_39 = paddle._C_ops.relu(batch_norm__226)

        # pd_op.conv2d: (1x256x160x240xf32) <- (1x64x160x240xf32, 256x64x1x1xf32)
        conv2d_42 = paddle._C_ops.conv2d(
            relu_39, parameter_164, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x256x160x240xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x160x240xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__232,
            batch_norm__233,
            batch_norm__234,
            batch_norm__235,
            batch_norm__236,
            batch_norm__237,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_42,
                parameter_163,
                parameter_162,
                parameter_161,
                parameter_160,
                False,
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

        # pd_op.add: (1x256x160x240xf32) <- (1x256x160x240xf32, 1x256x160x240xf32)
        add_2 = paddle._C_ops.add(batch_norm__232, relu_37)

        # pd_op.relu: (1x256x160x240xf32) <- (1x256x160x240xf32)
        relu_0 = paddle._C_ops.relu(add_2)

        # pd_op.conv2d: (1x128x160x240xf32) <- (1x256x160x240xf32, 128x256x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            relu_0, parameter_159, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x128x160x240xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (1x128x160x240xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__238,
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_0,
                parameter_158,
                parameter_157,
                parameter_156,
                parameter_155,
                False,
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

        # pd_op.relu: (1x128x160x240xf32) <- (1x128x160x240xf32)
        relu_1 = paddle._C_ops.relu(batch_norm__238)

        # pd_op.conv2d: (1x128x80x120xf32) <- (1x128x160x240xf32, 128x128x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            relu_1, parameter_154, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x128x80x120xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (1x128x80x120xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__239,
            batch_norm__5,
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_1,
                parameter_153,
                parameter_152,
                parameter_151,
                parameter_150,
                False,
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

        # pd_op.relu: (1x128x80x120xf32) <- (1x128x80x120xf32)
        relu_2 = paddle._C_ops.relu(batch_norm__239)

        # pd_op.conv2d: (1x512x80x120xf32) <- (1x128x80x120xf32, 512x128x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            relu_2, parameter_149, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x512x80x120xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (1x512x80x120xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__10,
            batch_norm__11,
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_2,
                parameter_148,
                parameter_147,
                parameter_146,
                parameter_145,
                False,
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

        # pd_op.conv2d: (1x512x80x120xf32) <- (1x256x160x240xf32, 512x256x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            relu_0, parameter_144, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x512x80x120xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (1x512x80x120xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__16,
            batch_norm__17,
            batch_norm__18,
            batch_norm__19,
            batch_norm__20,
            batch_norm__21,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_3,
                parameter_143,
                parameter_142,
                parameter_141,
                parameter_140,
                False,
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

        # pd_op.add: (1x512x80x120xf32) <- (1x512x80x120xf32, 1x512x80x120xf32)
        add_3 = paddle._C_ops.add(batch_norm__10, batch_norm__16)

        # pd_op.relu: (1x512x80x120xf32) <- (1x512x80x120xf32)
        relu_3 = paddle._C_ops.relu(add_3)

        # pd_op.conv2d: (1x128x80x120xf32) <- (1x512x80x120xf32, 128x512x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            relu_3, parameter_139, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x128x80x120xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (1x128x80x120xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__240,
            batch_norm__22,
            batch_norm__23,
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_4,
                parameter_138,
                parameter_137,
                parameter_136,
                parameter_135,
                False,
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

        # pd_op.relu: (1x128x80x120xf32) <- (1x128x80x120xf32)
        relu_4 = paddle._C_ops.relu(batch_norm__240)

        # pd_op.conv2d: (1x128x80x120xf32) <- (1x128x80x120xf32, 128x128x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            relu_4, parameter_134, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x128x80x120xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (1x128x80x120xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__241,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
            batch_norm__30,
            batch_norm__31,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_5,
                parameter_133,
                parameter_132,
                parameter_131,
                parameter_130,
                False,
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

        # pd_op.relu: (1x128x80x120xf32) <- (1x128x80x120xf32)
        relu_5 = paddle._C_ops.relu(batch_norm__241)

        # pd_op.conv2d: (1x512x80x120xf32) <- (1x128x80x120xf32, 512x128x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            relu_5, parameter_129, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x512x80x120xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (1x512x80x120xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
            batch_norm__36,
            batch_norm__37,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_6,
                parameter_128,
                parameter_127,
                parameter_126,
                parameter_125,
                False,
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

        # pd_op.add: (1x512x80x120xf32) <- (1x512x80x120xf32, 1x512x80x120xf32)
        add_4 = paddle._C_ops.add(batch_norm__32, relu_3)

        # pd_op.relu: (1x512x80x120xf32) <- (1x512x80x120xf32)
        relu_6 = paddle._C_ops.relu(add_4)

        # pd_op.conv2d: (1x128x80x120xf32) <- (1x512x80x120xf32, 128x512x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            relu_6, parameter_124, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x128x80x120xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (1x128x80x120xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__242,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
            batch_norm__42,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_7,
                parameter_123,
                parameter_122,
                parameter_121,
                parameter_120,
                False,
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

        # pd_op.relu: (1x128x80x120xf32) <- (1x128x80x120xf32)
        relu_7 = paddle._C_ops.relu(batch_norm__242)

        # pd_op.conv2d: (1x128x80x120xf32) <- (1x128x80x120xf32, 128x128x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            relu_7, parameter_119, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x128x80x120xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (1x128x80x120xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__243,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_8,
                parameter_118,
                parameter_117,
                parameter_116,
                parameter_115,
                False,
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

        # pd_op.relu: (1x128x80x120xf32) <- (1x128x80x120xf32)
        relu_8 = paddle._C_ops.relu(batch_norm__243)

        # pd_op.conv2d: (1x512x80x120xf32) <- (1x128x80x120xf32, 512x128x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            relu_8, parameter_114, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x512x80x120xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (1x512x80x120xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__48,
            batch_norm__49,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_9,
                parameter_113,
                parameter_112,
                parameter_111,
                parameter_110,
                False,
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

        # pd_op.add: (1x512x80x120xf32) <- (1x512x80x120xf32, 1x512x80x120xf32)
        add_5 = paddle._C_ops.add(batch_norm__48, relu_6)

        # pd_op.relu: (1x512x80x120xf32) <- (1x512x80x120xf32)
        relu_9 = paddle._C_ops.relu(add_5)

        # pd_op.conv2d: (1x128x80x120xf32) <- (1x512x80x120xf32, 128x512x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            relu_9, parameter_109, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x128x80x120xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (1x128x80x120xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__244,
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_10,
                parameter_108,
                parameter_107,
                parameter_106,
                parameter_105,
                False,
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

        # pd_op.relu: (1x128x80x120xf32) <- (1x128x80x120xf32)
        relu_10 = paddle._C_ops.relu(batch_norm__244)

        # pd_op.conv2d: (1x128x80x120xf32) <- (1x128x80x120xf32, 128x128x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            relu_10, parameter_104, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x128x80x120xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (1x128x80x120xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__245,
            batch_norm__59,
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_11,
                parameter_103,
                parameter_102,
                parameter_101,
                parameter_100,
                False,
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

        # pd_op.relu: (1x128x80x120xf32) <- (1x128x80x120xf32)
        relu_11 = paddle._C_ops.relu(batch_norm__245)

        # pd_op.conv2d: (1x512x80x120xf32) <- (1x128x80x120xf32, 512x128x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            relu_11, parameter_99, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x512x80x120xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (1x512x80x120xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__64,
            batch_norm__65,
            batch_norm__66,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_12,
                parameter_98,
                parameter_97,
                parameter_96,
                parameter_95,
                False,
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

        # pd_op.add: (1x512x80x120xf32) <- (1x512x80x120xf32, 1x512x80x120xf32)
        add_6 = paddle._C_ops.add(batch_norm__64, relu_9)

        # pd_op.relu: (1x512x80x120xf32) <- (1x512x80x120xf32)
        relu_12 = paddle._C_ops.relu(add_6)

        # pd_op.conv2d: (1x256x80x120xf32) <- (1x512x80x120xf32, 256x512x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            relu_12, parameter_94, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x256x80x120xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x80x120xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__246,
            batch_norm__70,
            batch_norm__71,
            batch_norm__72,
            batch_norm__73,
            batch_norm__74,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_13,
                parameter_93,
                parameter_92,
                parameter_91,
                parameter_90,
                False,
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

        # pd_op.relu: (1x256x80x120xf32) <- (1x256x80x120xf32)
        relu_13 = paddle._C_ops.relu(batch_norm__246)

        # pd_op.conv2d: (1x256x40x60xf32) <- (1x256x80x120xf32, 256x256x3x3xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            relu_13, parameter_89, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x256x40x60xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x40x60xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__247,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
            batch_norm__78,
            batch_norm__79,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_14,
                parameter_88,
                parameter_87,
                parameter_86,
                parameter_85,
                False,
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

        # pd_op.relu: (1x256x40x60xf32) <- (1x256x40x60xf32)
        relu_14 = paddle._C_ops.relu(batch_norm__247)

        # pd_op.conv2d: (1x1024x40x60xf32) <- (1x256x40x60xf32, 1024x256x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            relu_14, parameter_84, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x1024x40x60xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (1x1024x40x60xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        (
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
            batch_norm__83,
            batch_norm__84,
            batch_norm__85,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_15,
                parameter_83,
                parameter_82,
                parameter_81,
                parameter_80,
                False,
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

        # pd_op.conv2d: (1x1024x40x60xf32) <- (1x512x80x120xf32, 1024x512x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            relu_12, parameter_79, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x1024x40x60xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (1x1024x40x60xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        (
            batch_norm__86,
            batch_norm__87,
            batch_norm__88,
            batch_norm__89,
            batch_norm__90,
            batch_norm__91,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_16,
                parameter_78,
                parameter_77,
                parameter_76,
                parameter_75,
                False,
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

        # pd_op.add: (1x1024x40x60xf32) <- (1x1024x40x60xf32, 1x1024x40x60xf32)
        add_7 = paddle._C_ops.add(batch_norm__80, batch_norm__86)

        # pd_op.relu: (1x1024x40x60xf32) <- (1x1024x40x60xf32)
        relu_15 = paddle._C_ops.relu(add_7)

        # pd_op.conv2d: (1x256x40x60xf32) <- (1x1024x40x60xf32, 256x1024x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            relu_15, parameter_74, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x256x40x60xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x40x60xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__248,
            batch_norm__92,
            batch_norm__93,
            batch_norm__94,
            batch_norm__95,
            batch_norm__96,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_17,
                parameter_73,
                parameter_72,
                parameter_71,
                parameter_70,
                False,
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

        # pd_op.relu: (1x256x40x60xf32) <- (1x256x40x60xf32)
        relu_16 = paddle._C_ops.relu(batch_norm__248)

        # pd_op.conv2d: (1x256x40x60xf32) <- (1x256x40x60xf32, 256x256x3x3xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            relu_16, parameter_69, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x256x40x60xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x40x60xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__249,
            batch_norm__97,
            batch_norm__98,
            batch_norm__99,
            batch_norm__100,
            batch_norm__101,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_18,
                parameter_68,
                parameter_67,
                parameter_66,
                parameter_65,
                False,
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

        # pd_op.relu: (1x256x40x60xf32) <- (1x256x40x60xf32)
        relu_17 = paddle._C_ops.relu(batch_norm__249)

        # pd_op.conv2d: (1x1024x40x60xf32) <- (1x256x40x60xf32, 1024x256x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            relu_17, parameter_64, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x1024x40x60xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (1x1024x40x60xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        (
            batch_norm__102,
            batch_norm__103,
            batch_norm__104,
            batch_norm__105,
            batch_norm__106,
            batch_norm__107,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_19,
                parameter_63,
                parameter_62,
                parameter_61,
                parameter_60,
                False,
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

        # pd_op.add: (1x1024x40x60xf32) <- (1x1024x40x60xf32, 1x1024x40x60xf32)
        add_8 = paddle._C_ops.add(batch_norm__102, relu_15)

        # pd_op.relu: (1x1024x40x60xf32) <- (1x1024x40x60xf32)
        relu_18 = paddle._C_ops.relu(add_8)

        # pd_op.conv2d: (1x256x40x60xf32) <- (1x1024x40x60xf32, 256x1024x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            relu_18, parameter_59, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x256x40x60xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x40x60xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__250,
            batch_norm__108,
            batch_norm__109,
            batch_norm__110,
            batch_norm__111,
            batch_norm__112,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_20,
                parameter_58,
                parameter_57,
                parameter_56,
                parameter_55,
                False,
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

        # pd_op.relu: (1x256x40x60xf32) <- (1x256x40x60xf32)
        relu_19 = paddle._C_ops.relu(batch_norm__250)

        # pd_op.conv2d: (1x256x40x60xf32) <- (1x256x40x60xf32, 256x256x3x3xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            relu_19, parameter_54, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x256x40x60xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x40x60xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__251,
            batch_norm__113,
            batch_norm__114,
            batch_norm__115,
            batch_norm__116,
            batch_norm__117,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_21,
                parameter_53,
                parameter_52,
                parameter_51,
                parameter_50,
                False,
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

        # pd_op.relu: (1x256x40x60xf32) <- (1x256x40x60xf32)
        relu_20 = paddle._C_ops.relu(batch_norm__251)

        # pd_op.conv2d: (1x1024x40x60xf32) <- (1x256x40x60xf32, 1024x256x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(
            relu_20, parameter_49, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x1024x40x60xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (1x1024x40x60xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        (
            batch_norm__118,
            batch_norm__119,
            batch_norm__120,
            batch_norm__121,
            batch_norm__122,
            batch_norm__123,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_22,
                parameter_48,
                parameter_47,
                parameter_46,
                parameter_45,
                False,
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

        # pd_op.add: (1x1024x40x60xf32) <- (1x1024x40x60xf32, 1x1024x40x60xf32)
        add_9 = paddle._C_ops.add(batch_norm__118, relu_18)

        # pd_op.relu: (1x1024x40x60xf32) <- (1x1024x40x60xf32)
        relu_21 = paddle._C_ops.relu(add_9)

        # pd_op.conv2d: (1x256x40x60xf32) <- (1x1024x40x60xf32, 256x1024x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(
            relu_21, parameter_44, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x256x40x60xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x40x60xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__252,
            batch_norm__124,
            batch_norm__125,
            batch_norm__126,
            batch_norm__127,
            batch_norm__128,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_23,
                parameter_43,
                parameter_42,
                parameter_41,
                parameter_40,
                False,
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

        # pd_op.relu: (1x256x40x60xf32) <- (1x256x40x60xf32)
        relu_22 = paddle._C_ops.relu(batch_norm__252)

        # pd_op.conv2d: (1x256x40x60xf32) <- (1x256x40x60xf32, 256x256x3x3xf32)
        conv2d_24 = paddle._C_ops.conv2d(
            relu_22, parameter_39, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x256x40x60xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x40x60xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__253,
            batch_norm__129,
            batch_norm__130,
            batch_norm__131,
            batch_norm__132,
            batch_norm__133,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_24,
                parameter_38,
                parameter_37,
                parameter_36,
                parameter_35,
                False,
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

        # pd_op.relu: (1x256x40x60xf32) <- (1x256x40x60xf32)
        relu_23 = paddle._C_ops.relu(batch_norm__253)

        # pd_op.conv2d: (1x1024x40x60xf32) <- (1x256x40x60xf32, 1024x256x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(
            relu_23, parameter_34, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x1024x40x60xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (1x1024x40x60xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        (
            batch_norm__134,
            batch_norm__135,
            batch_norm__136,
            batch_norm__137,
            batch_norm__138,
            batch_norm__139,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_25,
                parameter_33,
                parameter_32,
                parameter_31,
                parameter_30,
                False,
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

        # pd_op.add: (1x1024x40x60xf32) <- (1x1024x40x60xf32, 1x1024x40x60xf32)
        add_10 = paddle._C_ops.add(batch_norm__134, relu_21)

        # pd_op.relu: (1x1024x40x60xf32) <- (1x1024x40x60xf32)
        relu_24 = paddle._C_ops.relu(add_10)

        # pd_op.conv2d: (1x256x40x60xf32) <- (1x1024x40x60xf32, 256x1024x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(
            relu_24, parameter_29, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x256x40x60xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x40x60xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__254,
            batch_norm__140,
            batch_norm__141,
            batch_norm__142,
            batch_norm__143,
            batch_norm__144,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_26,
                parameter_28,
                parameter_27,
                parameter_26,
                parameter_25,
                False,
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

        # pd_op.relu: (1x256x40x60xf32) <- (1x256x40x60xf32)
        relu_25 = paddle._C_ops.relu(batch_norm__254)

        # pd_op.conv2d: (1x256x40x60xf32) <- (1x256x40x60xf32, 256x256x3x3xf32)
        conv2d_27 = paddle._C_ops.conv2d(
            relu_25, parameter_24, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x256x40x60xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x40x60xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__255,
            batch_norm__145,
            batch_norm__146,
            batch_norm__147,
            batch_norm__148,
            batch_norm__149,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_27,
                parameter_23,
                parameter_22,
                parameter_21,
                parameter_20,
                False,
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

        # pd_op.relu: (1x256x40x60xf32) <- (1x256x40x60xf32)
        relu_26 = paddle._C_ops.relu(batch_norm__255)

        # pd_op.conv2d: (1x1024x40x60xf32) <- (1x256x40x60xf32, 1024x256x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(
            relu_26, parameter_19, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x1024x40x60xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (1x1024x40x60xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        (
            batch_norm__150,
            batch_norm__151,
            batch_norm__152,
            batch_norm__153,
            batch_norm__154,
            batch_norm__155,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_28,
                parameter_18,
                parameter_17,
                parameter_16,
                parameter_15,
                False,
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

        # pd_op.add: (1x1024x40x60xf32) <- (1x1024x40x60xf32, 1x1024x40x60xf32)
        add_11 = paddle._C_ops.add(batch_norm__150, relu_24)

        # pd_op.relu: (1x1024x40x60xf32) <- (1x1024x40x60xf32)
        relu_27 = paddle._C_ops.relu(add_11)

        # pd_op.conv2d: (1x256x40x60xf32) <- (1x1024x40x60xf32, 256x1024x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(
            relu_27, parameter_14, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x256x40x60xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x40x60xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__256,
            batch_norm__156,
            batch_norm__157,
            batch_norm__158,
            batch_norm__159,
            batch_norm__160,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_29,
                parameter_13,
                parameter_12,
                parameter_11,
                parameter_10,
                False,
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

        # pd_op.relu: (1x256x40x60xf32) <- (1x256x40x60xf32)
        relu_28 = paddle._C_ops.relu(batch_norm__256)

        # pd_op.conv2d: (1x256x40x60xf32) <- (1x256x40x60xf32, 256x256x3x3xf32)
        conv2d_30 = paddle._C_ops.conv2d(
            relu_28, parameter_9, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x256x40x60xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x40x60xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__257,
            batch_norm__161,
            batch_norm__162,
            batch_norm__163,
            batch_norm__164,
            batch_norm__165,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_30,
                parameter_8,
                parameter_7,
                parameter_6,
                parameter_5,
                False,
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

        # pd_op.relu: (1x256x40x60xf32) <- (1x256x40x60xf32)
        relu_29 = paddle._C_ops.relu(batch_norm__257)

        # pd_op.conv2d: (1x1024x40x60xf32) <- (1x256x40x60xf32, 1024x256x1x1xf32)
        conv2d_31 = paddle._C_ops.conv2d(
            relu_29, parameter_4, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x1024x40x60xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (1x1024x40x60xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        (
            batch_norm__166,
            batch_norm__167,
            batch_norm__168,
            batch_norm__169,
            batch_norm__170,
            batch_norm__171,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_31,
                parameter_3,
                parameter_2,
                parameter_1,
                parameter_0,
                False,
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

        # pd_op.add: (1x1024x40x60xf32) <- (1x1024x40x60xf32, 1x1024x40x60xf32)
        add_12 = paddle._C_ops.add(batch_norm__166, relu_27)

        # pd_op.relu: (1x1024x40x60xf32) <- (1x1024x40x60xf32)
        relu_30 = paddle._C_ops.relu(add_12)
        return (
            relu_0,
            conv2d_0,
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
            relu_1,
            conv2d_1,
            batch_norm__5,
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            relu_2,
            conv2d_2,
            batch_norm__10,
            batch_norm__11,
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            conv2d_3,
            batch_norm__16,
            batch_norm__17,
            batch_norm__18,
            batch_norm__19,
            batch_norm__20,
            batch_norm__21,
            relu_3,
            conv2d_4,
            batch_norm__22,
            batch_norm__23,
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
            relu_4,
            conv2d_5,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
            batch_norm__30,
            batch_norm__31,
            relu_5,
            conv2d_6,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
            batch_norm__36,
            batch_norm__37,
            relu_6,
            conv2d_7,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
            batch_norm__42,
            relu_7,
            conv2d_8,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
            relu_8,
            conv2d_9,
            batch_norm__48,
            batch_norm__49,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
            relu_9,
            conv2d_10,
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            relu_10,
            conv2d_11,
            batch_norm__59,
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            relu_11,
            conv2d_12,
            batch_norm__64,
            batch_norm__65,
            batch_norm__66,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            relu_12,
            conv2d_13,
            batch_norm__70,
            batch_norm__71,
            batch_norm__72,
            batch_norm__73,
            batch_norm__74,
            relu_13,
            conv2d_14,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
            batch_norm__78,
            batch_norm__79,
            relu_14,
            conv2d_15,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
            batch_norm__83,
            batch_norm__84,
            batch_norm__85,
            conv2d_16,
            batch_norm__86,
            batch_norm__87,
            batch_norm__88,
            batch_norm__89,
            batch_norm__90,
            batch_norm__91,
            relu_15,
            conv2d_17,
            batch_norm__92,
            batch_norm__93,
            batch_norm__94,
            batch_norm__95,
            batch_norm__96,
            relu_16,
            conv2d_18,
            batch_norm__97,
            batch_norm__98,
            batch_norm__99,
            batch_norm__100,
            batch_norm__101,
            relu_17,
            conv2d_19,
            batch_norm__102,
            batch_norm__103,
            batch_norm__104,
            batch_norm__105,
            batch_norm__106,
            batch_norm__107,
            relu_18,
            conv2d_20,
            batch_norm__108,
            batch_norm__109,
            batch_norm__110,
            batch_norm__111,
            batch_norm__112,
            relu_19,
            conv2d_21,
            batch_norm__113,
            batch_norm__114,
            batch_norm__115,
            batch_norm__116,
            batch_norm__117,
            relu_20,
            conv2d_22,
            batch_norm__118,
            batch_norm__119,
            batch_norm__120,
            batch_norm__121,
            batch_norm__122,
            batch_norm__123,
            relu_21,
            conv2d_23,
            batch_norm__124,
            batch_norm__125,
            batch_norm__126,
            batch_norm__127,
            batch_norm__128,
            relu_22,
            conv2d_24,
            batch_norm__129,
            batch_norm__130,
            batch_norm__131,
            batch_norm__132,
            batch_norm__133,
            relu_23,
            conv2d_25,
            batch_norm__134,
            batch_norm__135,
            batch_norm__136,
            batch_norm__137,
            batch_norm__138,
            batch_norm__139,
            relu_24,
            conv2d_26,
            batch_norm__140,
            batch_norm__141,
            batch_norm__142,
            batch_norm__143,
            batch_norm__144,
            relu_25,
            conv2d_27,
            batch_norm__145,
            batch_norm__146,
            batch_norm__147,
            batch_norm__148,
            batch_norm__149,
            relu_26,
            conv2d_28,
            batch_norm__150,
            batch_norm__151,
            batch_norm__152,
            batch_norm__153,
            batch_norm__154,
            batch_norm__155,
            relu_27,
            conv2d_29,
            batch_norm__156,
            batch_norm__157,
            batch_norm__158,
            batch_norm__159,
            batch_norm__160,
            relu_28,
            conv2d_30,
            batch_norm__161,
            batch_norm__162,
            batch_norm__163,
            batch_norm__164,
            batch_norm__165,
            relu_29,
            conv2d_31,
            batch_norm__166,
            batch_norm__167,
            batch_norm__168,
            batch_norm__169,
            batch_norm__170,
            batch_norm__171,
            relu_30,
        )
