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
        data_0,
    ):
        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x3x-1x-1xf32, 64x3x7x7xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_200, [2, 2], [3, 3], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_0, parameter_200

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_199,
                parameter_198,
                parameter_197,
                parameter_196,
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
        del conv2d_0, parameter_196, parameter_197, parameter_198, parameter_199

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_0 = paddle._C_ops.relu(batch_norm__0)
        del batch_norm__0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [3, 3]

        # pd_op.pool2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(
            relu_0,
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
        del full_int_array_0, relu_0

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            pool2d_1, parameter_195, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_195

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_194,
                parameter_193,
                parameter_192,
                parameter_191,
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
        del conv2d_1, parameter_191, parameter_192, parameter_193, parameter_194

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_1 = paddle._C_ops.relu(batch_norm__6)
        del batch_norm__6

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            relu_1, parameter_190, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_190, relu_1

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_189,
                parameter_188,
                parameter_187,
                parameter_186,
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
        del conv2d_2, parameter_186, parameter_187, parameter_188, parameter_189

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            pool2d_1, parameter_185, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_185, pool2d_1

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_184,
                parameter_183,
                parameter_182,
                parameter_181,
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
        del conv2d_3, parameter_181, parameter_182, parameter_183, parameter_184

        # pd_op.add: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, -1x64x-1x-1xf32)
        add_3 = paddle._C_ops.add(batch_norm__12, batch_norm__18)
        del batch_norm__12, batch_norm__18

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_2 = paddle._C_ops.relu(add_3)
        del add_3

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x3x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            relu_2, parameter_180, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_180

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_179,
                parameter_178,
                parameter_177,
                parameter_176,
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
        del conv2d_4, parameter_176, parameter_177, parameter_178, parameter_179

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_3 = paddle._C_ops.relu(batch_norm__24)
        del batch_norm__24

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            relu_3, parameter_175, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_175, relu_3

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_5,
                parameter_174,
                parameter_173,
                parameter_172,
                parameter_171,
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
        del conv2d_5, parameter_171, parameter_172, parameter_173, parameter_174

        # pd_op.add: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, -1x64x-1x-1xf32)
        add_4 = paddle._C_ops.add(batch_norm__30, relu_2)
        del batch_norm__30, relu_2

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_4 = paddle._C_ops.relu(add_4)
        del add_4

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            relu_4, parameter_170, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_170

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__36,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_6,
                parameter_169,
                parameter_168,
                parameter_167,
                parameter_166,
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
        del conv2d_6, parameter_166, parameter_167, parameter_168, parameter_169

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_5 = paddle._C_ops.relu(batch_norm__36)
        del batch_norm__36

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x3x3xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            relu_5, parameter_165, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_165, relu_5

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__42,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_7,
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
        del conv2d_7, parameter_161, parameter_162, parameter_163, parameter_164

        # pd_op.add: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, -1x64x-1x-1xf32)
        add_5 = paddle._C_ops.add(batch_norm__42, relu_4)
        del batch_norm__42, relu_4

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_6 = paddle._C_ops.relu(add_5)
        del add_5

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x64x-1x-1xf32, 128x64x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            relu_6, parameter_160, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_160

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__48,
            batch_norm__49,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_8,
                parameter_159,
                parameter_158,
                parameter_157,
                parameter_156,
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
        del conv2d_8, parameter_156, parameter_157, parameter_158, parameter_159

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_7 = paddle._C_ops.relu(batch_norm__48)
        del batch_norm__48

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x128x3x3xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            relu_7, parameter_155, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_155, relu_7

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_9,
                parameter_154,
                parameter_153,
                parameter_152,
                parameter_151,
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
        del conv2d_9, parameter_151, parameter_152, parameter_153, parameter_154

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x64x-1x-1xf32, 128x64x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            relu_6, parameter_150, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_150

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_10,
                parameter_149,
                parameter_148,
                parameter_147,
                parameter_146,
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
        del conv2d_10, parameter_146, parameter_147, parameter_148, parameter_149

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, -1x128x-1x-1xf32)
        add_6 = paddle._C_ops.add(batch_norm__54, batch_norm__60)
        del batch_norm__54, batch_norm__60

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_8 = paddle._C_ops.relu(add_6)
        del add_6

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x128x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            relu_8, parameter_145, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_145

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__66,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            batch_norm__70,
            batch_norm__71,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_11,
                parameter_144,
                parameter_143,
                parameter_142,
                parameter_141,
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
        del conv2d_11, parameter_141, parameter_142, parameter_143, parameter_144

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_9 = paddle._C_ops.relu(batch_norm__66)
        del batch_norm__66

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x128x3x3xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            relu_9, parameter_140, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_140, relu_9

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__72,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_12,
                parameter_139,
                parameter_138,
                parameter_137,
                parameter_136,
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
        del conv2d_12, parameter_136, parameter_137, parameter_138, parameter_139

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, -1x128x-1x-1xf32)
        add_7 = paddle._C_ops.add(batch_norm__72, relu_8)
        del batch_norm__72, relu_8

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_10 = paddle._C_ops.relu(add_7)
        del add_7

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x128x3x3xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            relu_10, parameter_135, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_135

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__78,
            batch_norm__79,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
            batch_norm__83,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_13,
                parameter_134,
                parameter_133,
                parameter_132,
                parameter_131,
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
        del conv2d_13, parameter_131, parameter_132, parameter_133, parameter_134

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_11 = paddle._C_ops.relu(batch_norm__78)
        del batch_norm__78

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x128x3x3xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            relu_11, parameter_130, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_130, relu_11

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__84,
            batch_norm__85,
            batch_norm__86,
            batch_norm__87,
            batch_norm__88,
            batch_norm__89,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_14,
                parameter_129,
                parameter_128,
                parameter_127,
                parameter_126,
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
        del conv2d_14, parameter_126, parameter_127, parameter_128, parameter_129

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, -1x128x-1x-1xf32)
        add_8 = paddle._C_ops.add(batch_norm__84, relu_10)
        del batch_norm__84, relu_10

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_12 = paddle._C_ops.relu(add_8)
        del add_8

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x128x3x3xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            relu_12, parameter_125, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_125

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__90,
            batch_norm__91,
            batch_norm__92,
            batch_norm__93,
            batch_norm__94,
            batch_norm__95,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_15,
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
        del conv2d_15, parameter_121, parameter_122, parameter_123, parameter_124

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_13 = paddle._C_ops.relu(batch_norm__90)
        del batch_norm__90

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x128x3x3xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            relu_13, parameter_120, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_120, relu_13

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__96,
            batch_norm__97,
            batch_norm__98,
            batch_norm__99,
            batch_norm__100,
            batch_norm__101,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_16,
                parameter_119,
                parameter_118,
                parameter_117,
                parameter_116,
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
        del conv2d_16, parameter_116, parameter_117, parameter_118, parameter_119

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, -1x128x-1x-1xf32)
        add_9 = paddle._C_ops.add(batch_norm__96, relu_12)
        del batch_norm__96, relu_12

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_14 = paddle._C_ops.relu(add_9)
        del add_9

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x128x-1x-1xf32, 256x128x3x3xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            relu_14, parameter_115, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_115

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__102,
            batch_norm__103,
            batch_norm__104,
            batch_norm__105,
            batch_norm__106,
            batch_norm__107,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_17,
                parameter_114,
                parameter_113,
                parameter_112,
                parameter_111,
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
        del conv2d_17, parameter_111, parameter_112, parameter_113, parameter_114

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_15 = paddle._C_ops.relu(batch_norm__102)
        del batch_norm__102

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            relu_15, parameter_110, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_110, relu_15

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__108,
            batch_norm__109,
            batch_norm__110,
            batch_norm__111,
            batch_norm__112,
            batch_norm__113,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_18,
                parameter_109,
                parameter_108,
                parameter_107,
                parameter_106,
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
        del conv2d_18, parameter_106, parameter_107, parameter_108, parameter_109

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x128x-1x-1xf32, 256x128x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            relu_14, parameter_105, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_105

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__114,
            batch_norm__115,
            batch_norm__116,
            batch_norm__117,
            batch_norm__118,
            batch_norm__119,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_19,
                parameter_104,
                parameter_103,
                parameter_102,
                parameter_101,
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
        del conv2d_19, parameter_101, parameter_102, parameter_103, parameter_104

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, -1x256x-1x-1xf32)
        add_10 = paddle._C_ops.add(batch_norm__108, batch_norm__114)
        del batch_norm__108, batch_norm__114

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_16 = paddle._C_ops.relu(add_10)
        del add_10

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            relu_16, parameter_100, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_100

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__120,
            batch_norm__121,
            batch_norm__122,
            batch_norm__123,
            batch_norm__124,
            batch_norm__125,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_20,
                parameter_99,
                parameter_98,
                parameter_97,
                parameter_96,
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
        del conv2d_20, parameter_96, parameter_97, parameter_98, parameter_99

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_17 = paddle._C_ops.relu(batch_norm__120)
        del batch_norm__120

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            relu_17, parameter_95, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_95, relu_17

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__126,
            batch_norm__127,
            batch_norm__128,
            batch_norm__129,
            batch_norm__130,
            batch_norm__131,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_21,
                parameter_94,
                parameter_93,
                parameter_92,
                parameter_91,
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
        del conv2d_21, parameter_91, parameter_92, parameter_93, parameter_94

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, -1x256x-1x-1xf32)
        add_11 = paddle._C_ops.add(batch_norm__126, relu_16)
        del batch_norm__126, relu_16

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_18 = paddle._C_ops.relu(add_11)
        del add_11

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_22 = paddle._C_ops.conv2d(
            relu_18, parameter_90, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_90

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__132,
            batch_norm__133,
            batch_norm__134,
            batch_norm__135,
            batch_norm__136,
            batch_norm__137,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_22,
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
        del conv2d_22, parameter_86, parameter_87, parameter_88, parameter_89

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_19 = paddle._C_ops.relu(batch_norm__132)
        del batch_norm__132

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_23 = paddle._C_ops.conv2d(
            relu_19, parameter_85, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_85, relu_19

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__138,
            batch_norm__139,
            batch_norm__140,
            batch_norm__141,
            batch_norm__142,
            batch_norm__143,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_23,
                parameter_84,
                parameter_83,
                parameter_82,
                parameter_81,
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
        del conv2d_23, parameter_81, parameter_82, parameter_83, parameter_84

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, -1x256x-1x-1xf32)
        add_12 = paddle._C_ops.add(batch_norm__138, relu_18)
        del batch_norm__138, relu_18

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_20 = paddle._C_ops.relu(add_12)
        del add_12

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_24 = paddle._C_ops.conv2d(
            relu_20, parameter_80, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_80

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__144,
            batch_norm__145,
            batch_norm__146,
            batch_norm__147,
            batch_norm__148,
            batch_norm__149,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_24,
                parameter_79,
                parameter_78,
                parameter_77,
                parameter_76,
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
        del conv2d_24, parameter_76, parameter_77, parameter_78, parameter_79

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_21 = paddle._C_ops.relu(batch_norm__144)
        del batch_norm__144

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_25 = paddle._C_ops.conv2d(
            relu_21, parameter_75, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_75, relu_21

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__150,
            batch_norm__151,
            batch_norm__152,
            batch_norm__153,
            batch_norm__154,
            batch_norm__155,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_25,
                parameter_74,
                parameter_73,
                parameter_72,
                parameter_71,
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
        del conv2d_25, parameter_71, parameter_72, parameter_73, parameter_74

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, -1x256x-1x-1xf32)
        add_13 = paddle._C_ops.add(batch_norm__150, relu_20)
        del batch_norm__150, relu_20

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_22 = paddle._C_ops.relu(add_13)
        del add_13

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_26 = paddle._C_ops.conv2d(
            relu_22, parameter_70, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_70

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__156,
            batch_norm__157,
            batch_norm__158,
            batch_norm__159,
            batch_norm__160,
            batch_norm__161,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_26,
                parameter_69,
                parameter_68,
                parameter_67,
                parameter_66,
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
        del conv2d_26, parameter_66, parameter_67, parameter_68, parameter_69

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_23 = paddle._C_ops.relu(batch_norm__156)
        del batch_norm__156

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_27 = paddle._C_ops.conv2d(
            relu_23, parameter_65, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_65, relu_23

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__162,
            batch_norm__163,
            batch_norm__164,
            batch_norm__165,
            batch_norm__166,
            batch_norm__167,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_27,
                parameter_64,
                parameter_63,
                parameter_62,
                parameter_61,
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
        del conv2d_27, parameter_61, parameter_62, parameter_63, parameter_64

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, -1x256x-1x-1xf32)
        add_14 = paddle._C_ops.add(batch_norm__162, relu_22)
        del batch_norm__162, relu_22

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_24 = paddle._C_ops.relu(add_14)
        del add_14

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_28 = paddle._C_ops.conv2d(
            relu_24, parameter_60, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_60

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__168,
            batch_norm__169,
            batch_norm__170,
            batch_norm__171,
            batch_norm__172,
            batch_norm__173,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_28,
                parameter_59,
                parameter_58,
                parameter_57,
                parameter_56,
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
        del conv2d_28, parameter_56, parameter_57, parameter_58, parameter_59

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_25 = paddle._C_ops.relu(batch_norm__168)
        del batch_norm__168

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_29 = paddle._C_ops.conv2d(
            relu_25, parameter_55, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_55, relu_25

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__174,
            batch_norm__175,
            batch_norm__176,
            batch_norm__177,
            batch_norm__178,
            batch_norm__179,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_29,
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
        del conv2d_29, parameter_51, parameter_52, parameter_53, parameter_54

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, -1x256x-1x-1xf32)
        add_15 = paddle._C_ops.add(batch_norm__174, relu_24)
        del batch_norm__174, relu_24

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_26 = paddle._C_ops.relu(add_15)
        del add_15

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x256x-1x-1xf32, 512x256x3x3xf32)
        conv2d_30 = paddle._C_ops.conv2d(
            relu_26, parameter_50, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_50

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__180,
            batch_norm__181,
            batch_norm__182,
            batch_norm__183,
            batch_norm__184,
            batch_norm__185,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_30,
                parameter_49,
                parameter_48,
                parameter_47,
                parameter_46,
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
        del conv2d_30, parameter_46, parameter_47, parameter_48, parameter_49

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_27 = paddle._C_ops.relu(batch_norm__180)
        del batch_norm__180

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x512x3x3xf32)
        conv2d_31 = paddle._C_ops.conv2d(
            relu_27, parameter_45, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_45, relu_27

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__186,
            batch_norm__187,
            batch_norm__188,
            batch_norm__189,
            batch_norm__190,
            batch_norm__191,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_31,
                parameter_44,
                parameter_43,
                parameter_42,
                parameter_41,
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
        del conv2d_31, parameter_41, parameter_42, parameter_43, parameter_44

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x256x-1x-1xf32, 512x256x1x1xf32)
        conv2d_32 = paddle._C_ops.conv2d(
            relu_26, parameter_40, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_40

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__192,
            batch_norm__193,
            batch_norm__194,
            batch_norm__195,
            batch_norm__196,
            batch_norm__197,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_32,
                parameter_39,
                parameter_38,
                parameter_37,
                parameter_36,
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
        del conv2d_32, parameter_36, parameter_37, parameter_38, parameter_39

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, -1x512x-1x-1xf32)
        add_16 = paddle._C_ops.add(batch_norm__186, batch_norm__192)
        del batch_norm__186, batch_norm__192

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_28 = paddle._C_ops.relu(add_16)
        del add_16

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x512x3x3xf32)
        conv2d_33 = paddle._C_ops.conv2d(
            relu_28, parameter_35, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_35

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__198,
            batch_norm__199,
            batch_norm__200,
            batch_norm__201,
            batch_norm__202,
            batch_norm__203,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_33,
                parameter_34,
                parameter_33,
                parameter_32,
                parameter_31,
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
        del conv2d_33, parameter_31, parameter_32, parameter_33, parameter_34

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_29 = paddle._C_ops.relu(batch_norm__198)
        del batch_norm__198

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x512x3x3xf32)
        conv2d_34 = paddle._C_ops.conv2d(
            relu_29, parameter_30, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_30, relu_29

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__204,
            batch_norm__205,
            batch_norm__206,
            batch_norm__207,
            batch_norm__208,
            batch_norm__209,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_34,
                parameter_29,
                parameter_28,
                parameter_27,
                parameter_26,
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
        del conv2d_34, parameter_26, parameter_27, parameter_28, parameter_29

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, -1x512x-1x-1xf32)
        add_17 = paddle._C_ops.add(batch_norm__204, relu_28)
        del batch_norm__204, relu_28

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_30 = paddle._C_ops.relu(add_17)
        del add_17

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x512x3x3xf32)
        conv2d_35 = paddle._C_ops.conv2d(
            relu_30, parameter_25, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_25

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__210,
            batch_norm__211,
            batch_norm__212,
            batch_norm__213,
            batch_norm__214,
            batch_norm__215,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_35,
                parameter_24,
                parameter_23,
                parameter_22,
                parameter_21,
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
        del conv2d_35, parameter_21, parameter_22, parameter_23, parameter_24

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_31 = paddle._C_ops.relu(batch_norm__210)
        del batch_norm__210

        # pd_op.conv2d: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x512x3x3xf32)
        conv2d_36 = paddle._C_ops.conv2d(
            relu_31, parameter_20, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_20, relu_31

        # pd_op.batch_norm_: (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__216,
            batch_norm__217,
            batch_norm__218,
            batch_norm__219,
            batch_norm__220,
            batch_norm__221,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_36,
                parameter_19,
                parameter_18,
                parameter_17,
                parameter_16,
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
        del conv2d_36, parameter_16, parameter_17, parameter_18, parameter_19

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, -1x512x-1x-1xf32)
        add_18 = paddle._C_ops.add(batch_norm__216, relu_30)
        del batch_norm__216, relu_30

        # pd_op.relu: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        relu_32 = paddle._C_ops.relu(add_18)
        del add_18

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x64x-1x-1xf32, 256x64x1x1xf32)
        conv2d_37 = paddle._C_ops.conv2d(
            relu_6, parameter_15, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_15, relu_6

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [1, -1, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_14, full_int_array_1)
        del parameter_14

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_19 = paddle._C_ops.add(conv2d_37, reshape_0)
        del conv2d_37, reshape_0

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x128x-1x-1xf32, 256x128x1x1xf32)
        conv2d_38 = paddle._C_ops.conv2d(
            relu_14, parameter_13, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_13, relu_14

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_12, full_int_array_1)
        del parameter_12

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_20 = paddle._C_ops.add(conv2d_38, reshape_1)
        del conv2d_38, reshape_1

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x1x1xf32)
        conv2d_39 = paddle._C_ops.conv2d(
            relu_26, parameter_11, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_11, relu_26

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(parameter_10, full_int_array_1)
        del parameter_10

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_21 = paddle._C_ops.add(conv2d_39, reshape_2)
        del conv2d_39, reshape_2

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x512x-1x-1xf32, 256x512x1x1xf32)
        conv2d_40 = paddle._C_ops.conv2d(
            relu_32, parameter_9, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_9, relu_32

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(parameter_8, full_int_array_1)
        del parameter_8

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_22 = paddle._C_ops.add(conv2d_40, reshape_3)
        del conv2d_40, reshape_3

        # pd_op.nearest_interp: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, None, None, None)
        nearest_interp_0 = paddle._C_ops.nearest_interp(
            add_22,
            None,
            None,
            None,
            "NCHW",
            -1,
            -1,
            -1,
            [float("2"), float("2")],
            "nearest",
            False,
            0,
        )

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, -1x256x-1x-1xf32)
        add_23 = paddle._C_ops.add(add_21, nearest_interp_0)
        del add_21, nearest_interp_0

        # pd_op.nearest_interp: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, None, None, None)
        nearest_interp_1 = paddle._C_ops.nearest_interp(
            add_23,
            None,
            None,
            None,
            "NCHW",
            -1,
            -1,
            -1,
            [float("2"), float("2")],
            "nearest",
            False,
            0,
        )

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, -1x256x-1x-1xf32)
        add_24 = paddle._C_ops.add(add_20, nearest_interp_1)
        del add_20, nearest_interp_1

        # pd_op.nearest_interp: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, None, None, None)
        nearest_interp_2 = paddle._C_ops.nearest_interp(
            add_24,
            None,
            None,
            None,
            "NCHW",
            -1,
            -1,
            -1,
            [float("2"), float("2")],
            "nearest",
            False,
            0,
        )

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, -1x256x-1x-1xf32)
        add_25 = paddle._C_ops.add(add_19, nearest_interp_2)
        del add_19, nearest_interp_2

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_41 = paddle._C_ops.conv2d(
            add_25, parameter_7, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_25, parameter_7

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(parameter_6, full_int_array_1)
        del parameter_6

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_41, reshape_4)
        del conv2d_41, reshape_4

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_42 = paddle._C_ops.conv2d(
            add_24, parameter_5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_24, parameter_5

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(parameter_4, full_int_array_1)
        del parameter_4

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_1 = paddle._C_ops.add(conv2d_42, reshape_5)
        del conv2d_42, reshape_5

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_43 = paddle._C_ops.conv2d(
            add_23, parameter_3, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_23, parameter_3

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(parameter_2, full_int_array_1)
        del parameter_2

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_2 = paddle._C_ops.add(conv2d_43, reshape_6)
        del conv2d_43, reshape_6

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_44 = paddle._C_ops.conv2d(
            add_22, parameter_1, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_22, parameter_1

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(parameter_0, full_int_array_1)
        del full_int_array_1, parameter_0

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_26 = paddle._C_ops.add(conv2d_44, reshape_7)
        del conv2d_44, reshape_7

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [1, 1]

        # pd_op.pool2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            add_26,
            full_int_array_2,
            [2, 2],
            [0, 0],
            False,
            True,
            "NCHW",
            "max",
            False,
            False,
            "EXPLICIT",
        )
        del add_26, full_int_array_2

        return add_0, add_1, add_2, pool2d_0
