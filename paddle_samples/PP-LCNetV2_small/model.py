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
        data_0,
    ):
        # pd_op.conv2d: (-1x24x-1x-1xf32) <- (-1x3x-1x-1xf32, 24x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_203, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (-1x24x-1x-1xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (-1x24x-1x-1xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        (
            batch_norm__201,
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_0,
                parameter_202,
                parameter_201,
                parameter_200,
                parameter_199,
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

        # pd_op.relu: (-1x24x-1x-1xf32) <- (-1x24x-1x-1xf32)
        relu_0 = paddle._C_ops.relu(batch_norm__201)

        # pd_op.depthwise_conv2d: (-1x24x-1x-1xf32) <- (-1x24x-1x-1xf32, 24x1x3x3xf32)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(
            relu_0, parameter_198, [1, 1], [1, 1], "EXPLICIT", 24, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (-1x24x-1x-1xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (-1x24x-1x-1xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        (
            batch_norm__202,
            batch_norm__5,
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_0,
                parameter_197,
                parameter_196,
                parameter_195,
                parameter_194,
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

        # pd_op.relu: (-1x24x-1x-1xf32) <- (-1x24x-1x-1xf32)
        relu_1 = paddle._C_ops.relu(batch_norm__202)

        # pd_op.conv2d: (-1x48x-1x-1xf32) <- (-1x24x-1x-1xf32, 48x24x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            relu_1, parameter_193, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (-1x48x-1x-1xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (-1x48x-1x-1xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        (
            batch_norm__203,
            batch_norm__10,
            batch_norm__11,
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_1,
                parameter_192,
                parameter_191,
                parameter_190,
                parameter_189,
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

        # pd_op.relu: (-1x48x-1x-1xf32) <- (-1x48x-1x-1xf32)
        relu_2 = paddle._C_ops.relu(batch_norm__203)

        # pd_op.depthwise_conv2d: (-1x48x-1x-1xf32) <- (-1x48x-1x-1xf32, 48x1x3x3xf32)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(
            relu_2, parameter_188, [2, 2], [1, 1], "EXPLICIT", 48, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (-1x48x-1x-1xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (-1x48x-1x-1xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        (
            batch_norm__204,
            batch_norm__15,
            batch_norm__16,
            batch_norm__17,
            batch_norm__18,
            batch_norm__19,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_1,
                parameter_187,
                parameter_186,
                parameter_185,
                parameter_184,
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

        # pd_op.relu: (-1x48x-1x-1xf32) <- (-1x48x-1x-1xf32)
        relu_3 = paddle._C_ops.relu(batch_norm__204)

        # pd_op.conv2d: (-1x96x-1x-1xf32) <- (-1x48x-1x-1xf32, 96x48x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            relu_3, parameter_183, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (-1x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__205,
            batch_norm__20,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
            batch_norm__24,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_2,
                parameter_182,
                parameter_181,
                parameter_180,
                parameter_179,
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

        # pd_op.relu: (-1x96x-1x-1xf32) <- (-1x96x-1x-1xf32)
        relu_4 = paddle._C_ops.relu(batch_norm__205)

        # pd_op.depthwise_conv2d: (-1x96x-1x-1xf32) <- (-1x96x-1x-1xf32, 96x1x3x3xf32)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(
            relu_4, parameter_178, [1, 1], [1, 1], "EXPLICIT", 96, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (-1x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__206,
            batch_norm__25,
            batch_norm__26,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_2,
                parameter_177,
                parameter_176,
                parameter_175,
                parameter_174,
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

        # pd_op.relu: (-1x96x-1x-1xf32) <- (-1x96x-1x-1xf32)
        relu_5 = paddle._C_ops.relu(batch_norm__206)

        # pd_op.conv2d: (-1x96x-1x-1xf32) <- (-1x96x-1x-1xf32, 96x96x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            relu_5, parameter_173, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (-1x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__207,
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_3,
                parameter_172,
                parameter_171,
                parameter_170,
                parameter_169,
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

        # pd_op.relu: (-1x96x-1x-1xf32) <- (-1x96x-1x-1xf32)
        relu_6 = paddle._C_ops.relu(batch_norm__207)

        # pd_op.depthwise_conv2d: (-1x96x-1x-1xf32) <- (-1x96x-1x-1xf32, 96x1x3x3xf32)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(
            relu_6, parameter_168, [2, 2], [1, 1], "EXPLICIT", 96, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (-1x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__208,
            batch_norm__35,
            batch_norm__36,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_3,
                parameter_167,
                parameter_166,
                parameter_165,
                parameter_164,
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

        # pd_op.relu: (-1x96x-1x-1xf32) <- (-1x96x-1x-1xf32)
        relu_7 = paddle._C_ops.relu(batch_norm__208)

        # pd_op.conv2d: (-1x192x-1x-1xf32) <- (-1x96x-1x-1xf32, 192x96x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            relu_7, parameter_163, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (-1x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__209,
            batch_norm__40,
            batch_norm__41,
            batch_norm__42,
            batch_norm__43,
            batch_norm__44,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_4,
                parameter_162,
                parameter_161,
                parameter_160,
                parameter_159,
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

        # pd_op.relu: (-1x192x-1x-1xf32) <- (-1x192x-1x-1xf32)
        relu_8 = paddle._C_ops.relu(batch_norm__209)

        # pd_op.depthwise_conv2d: (-1x192x-1x-1xf32) <- (-1x192x-1x-1xf32, 192x1x3x3xf32)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(
            relu_8, parameter_158, [1, 1], [1, 1], "EXPLICIT", 192, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (-1x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__210,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
            batch_norm__48,
            batch_norm__49,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_4,
                parameter_157,
                parameter_156,
                parameter_155,
                parameter_154,
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

        # pd_op.relu: (-1x192x-1x-1xf32) <- (-1x192x-1x-1xf32)
        relu_9 = paddle._C_ops.relu(batch_norm__210)

        # pd_op.conv2d: (-1x192x-1x-1xf32) <- (-1x192x-1x-1xf32, 192x192x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            relu_9, parameter_153, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (-1x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__211,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
            batch_norm__54,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_5,
                parameter_152,
                parameter_151,
                parameter_150,
                parameter_149,
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

        # pd_op.relu: (-1x192x-1x-1xf32) <- (-1x192x-1x-1xf32)
        relu_10 = paddle._C_ops.relu(batch_norm__211)

        # pd_op.depthwise_conv2d: (-1x192x-1x-1xf32) <- (-1x192x-1x-1xf32, 192x1x5x5xf32)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(
            relu_10, parameter_148, [2, 2], [2, 2], "EXPLICIT", 192, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (-1x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
            batch_norm__60,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_5,
                parameter_147,
                parameter_146,
                parameter_145,
                parameter_144,
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

        # pd_op.depthwise_conv2d: (-1x192x-1x-1xf32) <- (-1x192x-1x-1xf32, 192x1x3x3xf32)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(
            relu_10, parameter_143, [2, 2], [1, 1], "EXPLICIT", 192, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (-1x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
            batch_norm__66,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_6,
                parameter_142,
                parameter_141,
                parameter_140,
                parameter_139,
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

        # pd_op.add: (-1x192x-1x-1xf32) <- (-1x192x-1x-1xf32, -1x192x-1x-1xf32)
        add_6 = paddle._C_ops.add(batch_norm__55, batch_norm__61)

        # pd_op.relu: (-1x192x-1x-1xf32) <- (-1x192x-1x-1xf32)
        relu_11 = paddle._C_ops.relu(add_6)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_3 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_2 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_1 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_0 = full_int_array_0

        # pd_op.pool2d: (-1x192x1x1xf32) <- (-1x192x-1x-1xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            relu_11,
            full_int_array_0,
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

        # pd_op.conv2d: (-1x48x1x1xf32) <- (-1x192x1x1xf32, 48x192x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            pool2d_0, parameter_138, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [1, -1, 1, 1]

        # pd_op.reshape: (1x48x1x1xf32) <- (48xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_137, full_int_array_1)

        # pd_op.add: (-1x48x1x1xf32) <- (-1x48x1x1xf32, 1x48x1x1xf32)
        add_7 = paddle._C_ops.add(conv2d_6, reshape_0)

        # pd_op.relu: (-1x48x1x1xf32) <- (-1x48x1x1xf32)
        relu_12 = paddle._C_ops.relu(add_7)

        # pd_op.conv2d: (-1x192x1x1xf32) <- (-1x48x1x1xf32, 192x48x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            relu_12, parameter_136, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_135, full_int_array_1)

        # pd_op.add: (-1x192x1x1xf32) <- (-1x192x1x1xf32, 1x192x1x1xf32)
        add_8 = paddle._C_ops.add(conv2d_7, reshape_1)

        # pd_op.sigmoid: (-1x192x1x1xf32) <- (-1x192x1x1xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(add_8)

        # pd_op.multiply: (-1x192x-1x-1xf32) <- (-1x192x-1x-1xf32, -1x192x1x1xf32)
        multiply_0 = paddle._C_ops.multiply(relu_11, sigmoid_0)

        # pd_op.conv2d: (-1x192x-1x-1xf32) <- (-1x192x-1x-1xf32, 192x192x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            multiply_0, parameter_134, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (-1x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__212,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            batch_norm__70,
            batch_norm__71,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_8,
                parameter_133,
                parameter_132,
                parameter_131,
                parameter_130,
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

        # pd_op.relu: (-1x192x-1x-1xf32) <- (-1x192x-1x-1xf32)
        relu_13 = paddle._C_ops.relu(batch_norm__212)

        # pd_op.conv2d: (-1x384x-1x-1xf32) <- (-1x192x-1x-1xf32, 384x192x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            relu_13, parameter_129, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (-1x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__213,
            batch_norm__72,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            batch_norm__76,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_9,
                parameter_128,
                parameter_127,
                parameter_126,
                parameter_125,
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

        # pd_op.relu: (-1x384x-1x-1xf32) <- (-1x384x-1x-1xf32)
        relu_14 = paddle._C_ops.relu(batch_norm__213)

        # pd_op.depthwise_conv2d: (-1x384x-1x-1xf32) <- (-1x384x-1x-1xf32, 384x1x5x5xf32)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(
            relu_14, parameter_124, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (-1x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__77,
            batch_norm__78,
            batch_norm__79,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_7,
                parameter_123,
                parameter_122,
                parameter_121,
                parameter_120,
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

        # pd_op.depthwise_conv2d: (-1x384x-1x-1xf32) <- (-1x384x-1x-1xf32, 384x1x3x3xf32)
        depthwise_conv2d_8 = paddle._C_ops.depthwise_conv2d(
            relu_14, parameter_119, [1, 1], [1, 1], "EXPLICIT", 384, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (-1x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__83,
            batch_norm__84,
            batch_norm__85,
            batch_norm__86,
            batch_norm__87,
            batch_norm__88,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_8,
                parameter_118,
                parameter_117,
                parameter_116,
                parameter_115,
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

        # pd_op.add: (-1x384x-1x-1xf32) <- (-1x384x-1x-1xf32, -1x384x-1x-1xf32)
        add_0 = paddle._C_ops.add(batch_norm__77, batch_norm__83)

        # pd_op.depthwise_conv2d: (-1x384x-1x-1xf32) <- (-1x384x-1x-1xf32, 384x1x1x1xf32)
        depthwise_conv2d_9 = paddle._C_ops.depthwise_conv2d(
            relu_14, parameter_114, [1, 1], [0, 0], "EXPLICIT", 384, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (-1x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__89,
            batch_norm__90,
            batch_norm__91,
            batch_norm__92,
            batch_norm__93,
            batch_norm__94,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_9,
                parameter_113,
                parameter_112,
                parameter_111,
                parameter_110,
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

        # pd_op.add: (-1x384x-1x-1xf32) <- (-1x384x-1x-1xf32, -1x384x-1x-1xf32)
        add_9 = paddle._C_ops.add(add_0, batch_norm__89)

        # pd_op.relu: (-1x384x-1x-1xf32) <- (-1x384x-1x-1xf32)
        relu_15 = paddle._C_ops.relu(add_9)

        # pd_op.pool2d: (-1x384x1x1xf32) <- (-1x384x-1x-1xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(
            relu_15,
            full_int_array_0,
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

        # pd_op.conv2d: (-1x96x1x1xf32) <- (-1x384x1x1xf32, 96x384x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            pool2d_1, parameter_109, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(parameter_108, full_int_array_1)

        # pd_op.add: (-1x96x1x1xf32) <- (-1x96x1x1xf32, 1x96x1x1xf32)
        add_10 = paddle._C_ops.add(conv2d_10, reshape_2)

        # pd_op.relu: (-1x96x1x1xf32) <- (-1x96x1x1xf32)
        relu_16 = paddle._C_ops.relu(add_10)

        # pd_op.conv2d: (-1x384x1x1xf32) <- (-1x96x1x1xf32, 384x96x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            relu_16, parameter_107, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(parameter_106, full_int_array_1)

        # pd_op.add: (-1x384x1x1xf32) <- (-1x384x1x1xf32, 1x384x1x1xf32)
        add_11 = paddle._C_ops.add(conv2d_11, reshape_3)

        # pd_op.sigmoid: (-1x384x1x1xf32) <- (-1x384x1x1xf32)
        sigmoid_1 = paddle._C_ops.sigmoid(add_11)

        # pd_op.multiply: (-1x384x-1x-1xf32) <- (-1x384x-1x-1xf32, -1x384x1x1xf32)
        multiply_1 = paddle._C_ops.multiply(relu_15, sigmoid_1)

        # pd_op.conv2d: (-1x192x-1x-1xf32) <- (-1x384x-1x-1xf32, 192x384x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            multiply_1, parameter_105, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (-1x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__214,
            batch_norm__95,
            batch_norm__96,
            batch_norm__97,
            batch_norm__98,
            batch_norm__99,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_12,
                parameter_104,
                parameter_103,
                parameter_102,
                parameter_101,
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

        # pd_op.relu: (-1x192x-1x-1xf32) <- (-1x192x-1x-1xf32)
        relu_17 = paddle._C_ops.relu(batch_norm__214)

        # pd_op.conv2d: (-1x384x-1x-1xf32) <- (-1x192x-1x-1xf32, 384x192x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            relu_17, parameter_100, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (-1x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__215,
            batch_norm__100,
            batch_norm__101,
            batch_norm__102,
            batch_norm__103,
            batch_norm__104,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_13,
                parameter_99,
                parameter_98,
                parameter_97,
                parameter_96,
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

        # pd_op.relu: (-1x384x-1x-1xf32) <- (-1x384x-1x-1xf32)
        relu_18 = paddle._C_ops.relu(batch_norm__215)

        # pd_op.depthwise_conv2d: (-1x384x-1x-1xf32) <- (-1x384x-1x-1xf32, 384x1x5x5xf32)
        depthwise_conv2d_10 = paddle._C_ops.depthwise_conv2d(
            relu_18, parameter_95, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (-1x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__105,
            batch_norm__106,
            batch_norm__107,
            batch_norm__108,
            batch_norm__109,
            batch_norm__110,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_10,
                parameter_94,
                parameter_93,
                parameter_92,
                parameter_91,
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

        # pd_op.depthwise_conv2d: (-1x384x-1x-1xf32) <- (-1x384x-1x-1xf32, 384x1x3x3xf32)
        depthwise_conv2d_11 = paddle._C_ops.depthwise_conv2d(
            relu_18, parameter_90, [1, 1], [1, 1], "EXPLICIT", 384, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (-1x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__111,
            batch_norm__112,
            batch_norm__113,
            batch_norm__114,
            batch_norm__115,
            batch_norm__116,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_11,
                parameter_89,
                parameter_88,
                parameter_87,
                parameter_86,
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

        # pd_op.add: (-1x384x-1x-1xf32) <- (-1x384x-1x-1xf32, -1x384x-1x-1xf32)
        add_1 = paddle._C_ops.add(batch_norm__105, batch_norm__111)

        # pd_op.depthwise_conv2d: (-1x384x-1x-1xf32) <- (-1x384x-1x-1xf32, 384x1x1x1xf32)
        depthwise_conv2d_12 = paddle._C_ops.depthwise_conv2d(
            relu_18, parameter_85, [1, 1], [0, 0], "EXPLICIT", 384, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (-1x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__117,
            batch_norm__118,
            batch_norm__119,
            batch_norm__120,
            batch_norm__121,
            batch_norm__122,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_12,
                parameter_84,
                parameter_83,
                parameter_82,
                parameter_81,
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

        # pd_op.add: (-1x384x-1x-1xf32) <- (-1x384x-1x-1xf32, -1x384x-1x-1xf32)
        add_12 = paddle._C_ops.add(add_1, batch_norm__117)

        # pd_op.relu: (-1x384x-1x-1xf32) <- (-1x384x-1x-1xf32)
        relu_19 = paddle._C_ops.relu(add_12)

        # pd_op.pool2d: (-1x384x1x1xf32) <- (-1x384x-1x-1xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(
            relu_19,
            full_int_array_0,
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

        # pd_op.conv2d: (-1x96x1x1xf32) <- (-1x384x1x1xf32, 96x384x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            pool2d_2, parameter_80, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(parameter_79, full_int_array_1)

        # pd_op.add: (-1x96x1x1xf32) <- (-1x96x1x1xf32, 1x96x1x1xf32)
        add_13 = paddle._C_ops.add(conv2d_14, reshape_4)

        # pd_op.relu: (-1x96x1x1xf32) <- (-1x96x1x1xf32)
        relu_20 = paddle._C_ops.relu(add_13)

        # pd_op.conv2d: (-1x384x1x1xf32) <- (-1x96x1x1xf32, 384x96x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            relu_20, parameter_78, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(parameter_77, full_int_array_1)

        # pd_op.add: (-1x384x1x1xf32) <- (-1x384x1x1xf32, 1x384x1x1xf32)
        add_14 = paddle._C_ops.add(conv2d_15, reshape_5)

        # pd_op.sigmoid: (-1x384x1x1xf32) <- (-1x384x1x1xf32)
        sigmoid_2 = paddle._C_ops.sigmoid(add_14)

        # pd_op.multiply: (-1x384x-1x-1xf32) <- (-1x384x-1x-1xf32, -1x384x1x1xf32)
        multiply_2 = paddle._C_ops.multiply(relu_19, sigmoid_2)

        # pd_op.conv2d: (-1x192x-1x-1xf32) <- (-1x384x-1x-1xf32, 192x384x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            multiply_2, parameter_76, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (-1x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__216,
            batch_norm__123,
            batch_norm__124,
            batch_norm__125,
            batch_norm__126,
            batch_norm__127,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_16,
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

        # pd_op.relu: (-1x192x-1x-1xf32) <- (-1x192x-1x-1xf32)
        relu_21 = paddle._C_ops.relu(batch_norm__216)

        # pd_op.conv2d: (-1x384x-1x-1xf32) <- (-1x192x-1x-1xf32, 384x192x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            relu_21, parameter_71, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (-1x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__217,
            batch_norm__128,
            batch_norm__129,
            batch_norm__130,
            batch_norm__131,
            batch_norm__132,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_17,
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

        # pd_op.relu: (-1x384x-1x-1xf32) <- (-1x384x-1x-1xf32)
        relu_22 = paddle._C_ops.relu(batch_norm__217)

        # pd_op.depthwise_conv2d: (-1x384x-1x-1xf32) <- (-1x384x-1x-1xf32, 384x1x5x5xf32)
        depthwise_conv2d_13 = paddle._C_ops.depthwise_conv2d(
            relu_22, parameter_66, [1, 1], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (-1x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__133,
            batch_norm__134,
            batch_norm__135,
            batch_norm__136,
            batch_norm__137,
            batch_norm__138,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_13,
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

        # pd_op.depthwise_conv2d: (-1x384x-1x-1xf32) <- (-1x384x-1x-1xf32, 384x1x3x3xf32)
        depthwise_conv2d_14 = paddle._C_ops.depthwise_conv2d(
            relu_22, parameter_61, [1, 1], [1, 1], "EXPLICIT", 384, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (-1x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__139,
            batch_norm__140,
            batch_norm__141,
            batch_norm__142,
            batch_norm__143,
            batch_norm__144,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_14,
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

        # pd_op.add: (-1x384x-1x-1xf32) <- (-1x384x-1x-1xf32, -1x384x-1x-1xf32)
        add_2 = paddle._C_ops.add(batch_norm__133, batch_norm__139)

        # pd_op.depthwise_conv2d: (-1x384x-1x-1xf32) <- (-1x384x-1x-1xf32, 384x1x1x1xf32)
        depthwise_conv2d_15 = paddle._C_ops.depthwise_conv2d(
            relu_22, parameter_56, [1, 1], [0, 0], "EXPLICIT", 384, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (-1x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__145,
            batch_norm__146,
            batch_norm__147,
            batch_norm__148,
            batch_norm__149,
            batch_norm__150,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_15,
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

        # pd_op.add: (-1x384x-1x-1xf32) <- (-1x384x-1x-1xf32, -1x384x-1x-1xf32)
        add_15 = paddle._C_ops.add(add_2, batch_norm__145)

        # pd_op.relu: (-1x384x-1x-1xf32) <- (-1x384x-1x-1xf32)
        relu_23 = paddle._C_ops.relu(add_15)

        # pd_op.pool2d: (-1x384x1x1xf32) <- (-1x384x-1x-1xf32, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(
            relu_23,
            full_int_array_0,
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

        # pd_op.conv2d: (-1x96x1x1xf32) <- (-1x384x1x1xf32, 96x384x1x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            pool2d_3, parameter_51, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(parameter_50, full_int_array_1)

        # pd_op.add: (-1x96x1x1xf32) <- (-1x96x1x1xf32, 1x96x1x1xf32)
        add_16 = paddle._C_ops.add(conv2d_18, reshape_6)

        # pd_op.relu: (-1x96x1x1xf32) <- (-1x96x1x1xf32)
        relu_24 = paddle._C_ops.relu(add_16)

        # pd_op.conv2d: (-1x384x1x1xf32) <- (-1x96x1x1xf32, 384x96x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            relu_24, parameter_49, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(parameter_48, full_int_array_1)

        # pd_op.add: (-1x384x1x1xf32) <- (-1x384x1x1xf32, 1x384x1x1xf32)
        add_17 = paddle._C_ops.add(conv2d_19, reshape_7)

        # pd_op.sigmoid: (-1x384x1x1xf32) <- (-1x384x1x1xf32)
        sigmoid_3 = paddle._C_ops.sigmoid(add_17)

        # pd_op.multiply: (-1x384x-1x-1xf32) <- (-1x384x-1x-1xf32, -1x384x1x1xf32)
        multiply_3 = paddle._C_ops.multiply(relu_23, sigmoid_3)

        # pd_op.conv2d: (-1x192x-1x-1xf32) <- (-1x384x-1x-1xf32, 192x384x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            multiply_3, parameter_47, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (-1x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__218,
            batch_norm__151,
            batch_norm__152,
            batch_norm__153,
            batch_norm__154,
            batch_norm__155,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_20,
                parameter_46,
                parameter_45,
                parameter_44,
                parameter_43,
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

        # pd_op.relu: (-1x192x-1x-1xf32) <- (-1x192x-1x-1xf32)
        relu_25 = paddle._C_ops.relu(batch_norm__218)

        # pd_op.conv2d: (-1x384x-1x-1xf32) <- (-1x192x-1x-1xf32, 384x192x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            relu_25, parameter_42, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (-1x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__219,
            batch_norm__156,
            batch_norm__157,
            batch_norm__158,
            batch_norm__159,
            batch_norm__160,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_21,
                parameter_41,
                parameter_40,
                parameter_39,
                parameter_38,
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

        # pd_op.relu: (-1x384x-1x-1xf32) <- (-1x384x-1x-1xf32)
        relu_26 = paddle._C_ops.relu(batch_norm__219)

        # pd_op.depthwise_conv2d: (-1x384x-1x-1xf32) <- (-1x384x-1x-1xf32, 384x1x5x5xf32)
        depthwise_conv2d_16 = paddle._C_ops.depthwise_conv2d(
            relu_26, parameter_37, [2, 2], [2, 2], "EXPLICIT", 384, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (-1x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__161,
            batch_norm__162,
            batch_norm__163,
            batch_norm__164,
            batch_norm__165,
            batch_norm__166,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_16,
                parameter_36,
                parameter_35,
                parameter_34,
                parameter_33,
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

        # pd_op.depthwise_conv2d: (-1x384x-1x-1xf32) <- (-1x384x-1x-1xf32, 384x1x3x3xf32)
        depthwise_conv2d_17 = paddle._C_ops.depthwise_conv2d(
            relu_26, parameter_32, [2, 2], [1, 1], "EXPLICIT", 384, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (-1x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (-1x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__167,
            batch_norm__168,
            batch_norm__169,
            batch_norm__170,
            batch_norm__171,
            batch_norm__172,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_17,
                parameter_31,
                parameter_30,
                parameter_29,
                parameter_28,
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

        # pd_op.add: (-1x384x-1x-1xf32) <- (-1x384x-1x-1xf32, -1x384x-1x-1xf32)
        add_18 = paddle._C_ops.add(batch_norm__161, batch_norm__167)

        # pd_op.relu: (-1x384x-1x-1xf32) <- (-1x384x-1x-1xf32)
        relu_27 = paddle._C_ops.relu(add_18)

        # pd_op.conv2d: (-1x768x-1x-1xf32) <- (-1x384x-1x-1xf32, 768x384x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(
            relu_27, parameter_27, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (-1x768x-1x-1xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x-1x-1xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        (
            batch_norm__220,
            batch_norm__173,
            batch_norm__174,
            batch_norm__175,
            batch_norm__176,
            batch_norm__177,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_22,
                parameter_26,
                parameter_25,
                parameter_24,
                parameter_23,
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

        # pd_op.relu: (-1x768x-1x-1xf32) <- (-1x768x-1x-1xf32)
        relu_28 = paddle._C_ops.relu(batch_norm__220)

        # pd_op.depthwise_conv2d: (-1x768x-1x-1xf32) <- (-1x768x-1x-1xf32, 768x1x5x5xf32)
        depthwise_conv2d_18 = paddle._C_ops.depthwise_conv2d(
            relu_28, parameter_22, [1, 1], [2, 2], "EXPLICIT", 768, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (-1x768x-1x-1xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x-1x-1xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        (
            batch_norm__178,
            batch_norm__179,
            batch_norm__180,
            batch_norm__181,
            batch_norm__182,
            batch_norm__183,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_18,
                parameter_21,
                parameter_20,
                parameter_19,
                parameter_18,
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

        # pd_op.depthwise_conv2d: (-1x768x-1x-1xf32) <- (-1x768x-1x-1xf32, 768x1x3x3xf32)
        depthwise_conv2d_19 = paddle._C_ops.depthwise_conv2d(
            relu_28, parameter_17, [1, 1], [1, 1], "EXPLICIT", 768, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (-1x768x-1x-1xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x-1x-1xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        (
            batch_norm__184,
            batch_norm__185,
            batch_norm__186,
            batch_norm__187,
            batch_norm__188,
            batch_norm__189,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_19,
                parameter_16,
                parameter_15,
                parameter_14,
                parameter_13,
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

        # pd_op.add: (-1x768x-1x-1xf32) <- (-1x768x-1x-1xf32, -1x768x-1x-1xf32)
        add_3 = paddle._C_ops.add(batch_norm__178, batch_norm__184)

        # pd_op.depthwise_conv2d: (-1x768x-1x-1xf32) <- (-1x768x-1x-1xf32, 768x1x1x1xf32)
        depthwise_conv2d_20 = paddle._C_ops.depthwise_conv2d(
            relu_28, parameter_12, [1, 1], [0, 0], "EXPLICIT", 768, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (-1x768x-1x-1xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x-1x-1xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        (
            batch_norm__190,
            batch_norm__191,
            batch_norm__192,
            batch_norm__193,
            batch_norm__194,
            batch_norm__195,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_20,
                parameter_11,
                parameter_10,
                parameter_9,
                parameter_8,
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

        # pd_op.add: (-1x768x-1x-1xf32) <- (-1x768x-1x-1xf32, -1x768x-1x-1xf32)
        add_19 = paddle._C_ops.add(add_3, batch_norm__190)

        # pd_op.relu: (-1x768x-1x-1xf32) <- (-1x768x-1x-1xf32)
        relu_29 = paddle._C_ops.relu(add_19)

        # pd_op.conv2d: (-1x768x-1x-1xf32) <- (-1x768x-1x-1xf32, 768x768x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(
            relu_29, parameter_7, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (-1x768x-1x-1xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x-1x-1xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        (
            batch_norm__221,
            batch_norm__196,
            batch_norm__197,
            batch_norm__198,
            batch_norm__199,
            batch_norm__200,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_23,
                parameter_6,
                parameter_5,
                parameter_4,
                parameter_3,
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

        # pd_op.relu: (-1x768x-1x-1xf32) <- (-1x768x-1x-1xf32)
        relu_30 = paddle._C_ops.relu(batch_norm__221)

        # pd_op.add: (-1x768x-1x-1xf32) <- (-1x768x-1x-1xf32, -1x768x-1x-1xf32)
        add_4 = paddle._C_ops.add(relu_30, relu_28)

        # pd_op.pool2d: (-1x768x1x1xf32) <- (-1x768x-1x-1xf32, 2xi64)
        pool2d_4 = paddle._C_ops.pool2d(
            add_4,
            full_int_array_0,
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

        # pd_op.conv2d: (-1x1280x1x1xf32) <- (-1x768x1x1xf32, 1280x768x1x1xf32)
        conv2d_24 = paddle._C_ops.conv2d(
            pool2d_4, parameter_2, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.relu: (-1x1280x1x1xf32) <- (-1x1280x1x1xf32)
        relu_31 = paddle._C_ops.relu(conv2d_24)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (-1x1280x1x1xf32, -1x1280x1x1xui8) <- (-1x1280x1x1xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                relu_31, None, full_0, False, "downgrade_in_infer", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.flatten: (-1x1280xf32) <- (-1x1280x1x1xf32)
        flatten_0 = paddle._C_ops.flatten(dropout_0, 1, 3)

        # pd_op.matmul: (-1x102xf32) <- (-1x1280xf32, 1280x102xf32)
        matmul_0 = paddle._C_ops.matmul(flatten_0, parameter_1, False, False)

        # pd_op.add: (-1x102xf32) <- (-1x102xf32, 102xf32)
        add_5 = paddle._C_ops.add(matmul_0, parameter_0)
        return (
            conv2d_0,
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
            relu_0,
            depthwise_conv2d_0,
            batch_norm__5,
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            relu_1,
            conv2d_1,
            batch_norm__10,
            batch_norm__11,
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            relu_2,
            depthwise_conv2d_1,
            batch_norm__15,
            batch_norm__16,
            batch_norm__17,
            batch_norm__18,
            batch_norm__19,
            relu_3,
            conv2d_2,
            batch_norm__20,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
            batch_norm__24,
            relu_4,
            depthwise_conv2d_2,
            batch_norm__25,
            batch_norm__26,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
            relu_5,
            conv2d_3,
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            relu_6,
            depthwise_conv2d_3,
            batch_norm__35,
            batch_norm__36,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            relu_7,
            conv2d_4,
            batch_norm__40,
            batch_norm__41,
            batch_norm__42,
            batch_norm__43,
            batch_norm__44,
            relu_8,
            depthwise_conv2d_4,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
            batch_norm__48,
            batch_norm__49,
            relu_9,
            conv2d_5,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
            batch_norm__54,
            relu_10,
            depthwise_conv2d_5,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
            batch_norm__60,
            depthwise_conv2d_6,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
            batch_norm__66,
            relu_11,
            full_int_array_0,
            pool2d_0,
            conv2d_6,
            reshape_0,
            relu_12,
            conv2d_7,
            reshape_1,
            sigmoid_0,
            multiply_0,
            conv2d_8,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            batch_norm__70,
            batch_norm__71,
            relu_13,
            conv2d_9,
            batch_norm__72,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            batch_norm__76,
            relu_14,
            depthwise_conv2d_7,
            batch_norm__77,
            batch_norm__78,
            batch_norm__79,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
            depthwise_conv2d_8,
            batch_norm__83,
            batch_norm__84,
            batch_norm__85,
            batch_norm__86,
            batch_norm__87,
            batch_norm__88,
            add_0,
            depthwise_conv2d_9,
            batch_norm__89,
            batch_norm__90,
            batch_norm__91,
            batch_norm__92,
            batch_norm__93,
            batch_norm__94,
            relu_15,
            assign_0,
            pool2d_1,
            conv2d_10,
            reshape_2,
            relu_16,
            conv2d_11,
            reshape_3,
            sigmoid_1,
            multiply_1,
            conv2d_12,
            batch_norm__95,
            batch_norm__96,
            batch_norm__97,
            batch_norm__98,
            batch_norm__99,
            relu_17,
            conv2d_13,
            batch_norm__100,
            batch_norm__101,
            batch_norm__102,
            batch_norm__103,
            batch_norm__104,
            relu_18,
            depthwise_conv2d_10,
            batch_norm__105,
            batch_norm__106,
            batch_norm__107,
            batch_norm__108,
            batch_norm__109,
            batch_norm__110,
            depthwise_conv2d_11,
            batch_norm__111,
            batch_norm__112,
            batch_norm__113,
            batch_norm__114,
            batch_norm__115,
            batch_norm__116,
            add_1,
            depthwise_conv2d_12,
            batch_norm__117,
            batch_norm__118,
            batch_norm__119,
            batch_norm__120,
            batch_norm__121,
            batch_norm__122,
            relu_19,
            assign_1,
            pool2d_2,
            conv2d_14,
            reshape_4,
            relu_20,
            conv2d_15,
            reshape_5,
            sigmoid_2,
            multiply_2,
            conv2d_16,
            batch_norm__123,
            batch_norm__124,
            batch_norm__125,
            batch_norm__126,
            batch_norm__127,
            relu_21,
            conv2d_17,
            batch_norm__128,
            batch_norm__129,
            batch_norm__130,
            batch_norm__131,
            batch_norm__132,
            relu_22,
            depthwise_conv2d_13,
            batch_norm__133,
            batch_norm__134,
            batch_norm__135,
            batch_norm__136,
            batch_norm__137,
            batch_norm__138,
            depthwise_conv2d_14,
            batch_norm__139,
            batch_norm__140,
            batch_norm__141,
            batch_norm__142,
            batch_norm__143,
            batch_norm__144,
            add_2,
            depthwise_conv2d_15,
            batch_norm__145,
            batch_norm__146,
            batch_norm__147,
            batch_norm__148,
            batch_norm__149,
            batch_norm__150,
            relu_23,
            assign_2,
            pool2d_3,
            conv2d_18,
            reshape_6,
            relu_24,
            conv2d_19,
            reshape_7,
            sigmoid_3,
            multiply_3,
            conv2d_20,
            batch_norm__151,
            batch_norm__152,
            batch_norm__153,
            batch_norm__154,
            batch_norm__155,
            relu_25,
            conv2d_21,
            batch_norm__156,
            batch_norm__157,
            batch_norm__158,
            batch_norm__159,
            batch_norm__160,
            relu_26,
            depthwise_conv2d_16,
            batch_norm__161,
            batch_norm__162,
            batch_norm__163,
            batch_norm__164,
            batch_norm__165,
            batch_norm__166,
            depthwise_conv2d_17,
            batch_norm__167,
            batch_norm__168,
            batch_norm__169,
            batch_norm__170,
            batch_norm__171,
            batch_norm__172,
            relu_27,
            conv2d_22,
            batch_norm__173,
            batch_norm__174,
            batch_norm__175,
            batch_norm__176,
            batch_norm__177,
            relu_28,
            depthwise_conv2d_18,
            batch_norm__178,
            batch_norm__179,
            batch_norm__180,
            batch_norm__181,
            batch_norm__182,
            batch_norm__183,
            depthwise_conv2d_19,
            batch_norm__184,
            batch_norm__185,
            batch_norm__186,
            batch_norm__187,
            batch_norm__188,
            batch_norm__189,
            add_3,
            depthwise_conv2d_20,
            batch_norm__190,
            batch_norm__191,
            batch_norm__192,
            batch_norm__193,
            batch_norm__194,
            batch_norm__195,
            relu_29,
            conv2d_23,
            batch_norm__196,
            batch_norm__197,
            batch_norm__198,
            batch_norm__199,
            batch_norm__200,
            relu_30,
            add_4,
            assign_3,
            pool2d_4,
            relu_31,
            full_0,
            dropout_0,
            dropout_1,
            flatten_0,
            matmul_0,
            add_5,
        )
