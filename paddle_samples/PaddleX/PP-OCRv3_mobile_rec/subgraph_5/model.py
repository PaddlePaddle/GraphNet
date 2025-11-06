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
        data_0,
    ):
        # pd_op.conv2d: (8x16x24x160xf32) <- (8x3x48x320xf32, 16x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_195, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_0, parameter_195

        # pd_op.batch_norm_: (8x16x24x160xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (8x16x24x160xf32, 16xf32, 16xf32, 16xf32, 16xf32)
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
                parameter_194,
                parameter_193,
                parameter_192,
                parameter_191,
                True,
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
        del conv2d_0, parameter_191, parameter_192, parameter_193, parameter_194

        # pd_op.hardswish: (8x16x24x160xf32) <- (8x16x24x160xf32)
        hardswish_0 = paddle._C_ops.hardswish(batch_norm__0)
        del batch_norm__0

        # pd_op.depthwise_conv2d: (8x16x24x160xf32) <- (8x16x24x160xf32, 16x1x3x3xf32)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(
            hardswish_0, parameter_190, [1, 1], [1, 1], "EXPLICIT", 16, [1, 1], "NCHW"
        )
        del hardswish_0, parameter_190

        # pd_op.batch_norm_: (8x16x24x160xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (8x16x24x160xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        (
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            batch_norm__10,
            batch_norm__11,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_0,
                parameter_189,
                parameter_188,
                parameter_187,
                parameter_186,
                True,
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
        del (
            depthwise_conv2d_0,
            parameter_186,
            parameter_187,
            parameter_188,
            parameter_189,
        )

        # pd_op.hardswish: (8x16x24x160xf32) <- (8x16x24x160xf32)
        hardswish_1 = paddle._C_ops.hardswish(batch_norm__6)
        del batch_norm__6

        # pd_op.conv2d: (8x32x24x160xf32) <- (8x16x24x160xf32, 32x16x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            hardswish_1, parameter_185, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del hardswish_1, parameter_185

        # pd_op.batch_norm_: (8x32x24x160xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (8x32x24x160xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        (
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            batch_norm__16,
            batch_norm__17,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_1,
                parameter_184,
                parameter_183,
                parameter_182,
                parameter_181,
                True,
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
        del conv2d_1, parameter_181, parameter_182, parameter_183, parameter_184

        # pd_op.hardswish: (8x32x24x160xf32) <- (8x32x24x160xf32)
        hardswish_2 = paddle._C_ops.hardswish(batch_norm__12)
        del batch_norm__12

        # pd_op.depthwise_conv2d: (8x32x24x160xf32) <- (8x32x24x160xf32, 32x1x3x3xf32)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(
            hardswish_2, parameter_180, [1, 1], [1, 1], "EXPLICIT", 32, [1, 1], "NCHW"
        )
        del hardswish_2, parameter_180

        # pd_op.batch_norm_: (8x32x24x160xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (8x32x24x160xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        (
            batch_norm__18,
            batch_norm__19,
            batch_norm__20,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_1,
                parameter_179,
                parameter_178,
                parameter_177,
                parameter_176,
                True,
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
        del (
            depthwise_conv2d_1,
            parameter_176,
            parameter_177,
            parameter_178,
            parameter_179,
        )

        # pd_op.hardswish: (8x32x24x160xf32) <- (8x32x24x160xf32)
        hardswish_3 = paddle._C_ops.hardswish(batch_norm__18)
        del batch_norm__18

        # pd_op.conv2d: (8x64x24x160xf32) <- (8x32x24x160xf32, 64x32x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            hardswish_3, parameter_175, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del hardswish_3, parameter_175

        # pd_op.batch_norm_: (8x64x24x160xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (8x64x24x160xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_2,
                parameter_174,
                parameter_173,
                parameter_172,
                parameter_171,
                True,
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
        del conv2d_2, parameter_171, parameter_172, parameter_173, parameter_174

        # pd_op.hardswish: (8x64x24x160xf32) <- (8x64x24x160xf32)
        hardswish_4 = paddle._C_ops.hardswish(batch_norm__24)
        del batch_norm__24

        # pd_op.depthwise_conv2d: (8x64x24x160xf32) <- (8x64x24x160xf32, 64x1x3x3xf32)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(
            hardswish_4, parameter_170, [1, 1], [1, 1], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del hardswish_4, parameter_170

        # pd_op.batch_norm_: (8x64x24x160xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (8x64x24x160xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_2,
                parameter_169,
                parameter_168,
                parameter_167,
                parameter_166,
                True,
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
        del (
            depthwise_conv2d_2,
            parameter_166,
            parameter_167,
            parameter_168,
            parameter_169,
        )

        # pd_op.hardswish: (8x64x24x160xf32) <- (8x64x24x160xf32)
        hardswish_5 = paddle._C_ops.hardswish(batch_norm__30)
        del batch_norm__30

        # pd_op.conv2d: (8x64x24x160xf32) <- (8x64x24x160xf32, 64x64x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            hardswish_5, parameter_165, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del hardswish_5, parameter_165

        # pd_op.batch_norm_: (8x64x24x160xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (8x64x24x160xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__36,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_3,
                parameter_164,
                parameter_163,
                parameter_162,
                parameter_161,
                True,
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
        del conv2d_3, parameter_161, parameter_162, parameter_163, parameter_164

        # pd_op.hardswish: (8x64x24x160xf32) <- (8x64x24x160xf32)
        hardswish_6 = paddle._C_ops.hardswish(batch_norm__36)
        del batch_norm__36

        # pd_op.depthwise_conv2d: (8x64x12x160xf32) <- (8x64x24x160xf32, 64x1x3x3xf32)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(
            hardswish_6, parameter_160, [2, 1], [1, 1], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del hardswish_6, parameter_160

        # pd_op.batch_norm_: (8x64x12x160xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (8x64x12x160xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__42,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_3,
                parameter_159,
                parameter_158,
                parameter_157,
                parameter_156,
                True,
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
        del (
            depthwise_conv2d_3,
            parameter_156,
            parameter_157,
            parameter_158,
            parameter_159,
        )

        # pd_op.hardswish: (8x64x12x160xf32) <- (8x64x12x160xf32)
        hardswish_7 = paddle._C_ops.hardswish(batch_norm__42)
        del batch_norm__42

        # pd_op.conv2d: (8x128x12x160xf32) <- (8x64x12x160xf32, 128x64x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            hardswish_7, parameter_155, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del hardswish_7, parameter_155

        # pd_op.batch_norm_: (8x128x12x160xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (8x128x12x160xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__48,
            batch_norm__49,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_4,
                parameter_154,
                parameter_153,
                parameter_152,
                parameter_151,
                True,
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
        del conv2d_4, parameter_151, parameter_152, parameter_153, parameter_154

        # pd_op.hardswish: (8x128x12x160xf32) <- (8x128x12x160xf32)
        hardswish_8 = paddle._C_ops.hardswish(batch_norm__48)
        del batch_norm__48

        # pd_op.depthwise_conv2d: (8x128x12x160xf32) <- (8x128x12x160xf32, 128x1x3x3xf32)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(
            hardswish_8, parameter_150, [1, 1], [1, 1], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del hardswish_8, parameter_150

        # pd_op.batch_norm_: (8x128x12x160xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (8x128x12x160xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_4,
                parameter_149,
                parameter_148,
                parameter_147,
                parameter_146,
                True,
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
        del (
            depthwise_conv2d_4,
            parameter_146,
            parameter_147,
            parameter_148,
            parameter_149,
        )

        # pd_op.hardswish: (8x128x12x160xf32) <- (8x128x12x160xf32)
        hardswish_9 = paddle._C_ops.hardswish(batch_norm__54)
        del batch_norm__54

        # pd_op.conv2d: (8x128x12x160xf32) <- (8x128x12x160xf32, 128x128x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            hardswish_9, parameter_145, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del hardswish_9, parameter_145

        # pd_op.batch_norm_: (8x128x12x160xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (8x128x12x160xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_5,
                parameter_144,
                parameter_143,
                parameter_142,
                parameter_141,
                True,
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
        del conv2d_5, parameter_141, parameter_142, parameter_143, parameter_144

        # pd_op.hardswish: (8x128x12x160xf32) <- (8x128x12x160xf32)
        hardswish_10 = paddle._C_ops.hardswish(batch_norm__60)
        del batch_norm__60

        # pd_op.depthwise_conv2d: (8x128x6x160xf32) <- (8x128x12x160xf32, 128x1x3x3xf32)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(
            hardswish_10, parameter_140, [2, 1], [1, 1], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del hardswish_10, parameter_140

        # pd_op.batch_norm_: (8x128x6x160xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (8x128x6x160xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__66,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            batch_norm__70,
            batch_norm__71,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_5,
                parameter_139,
                parameter_138,
                parameter_137,
                parameter_136,
                True,
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
        del (
            depthwise_conv2d_5,
            parameter_136,
            parameter_137,
            parameter_138,
            parameter_139,
        )

        # pd_op.hardswish: (8x128x6x160xf32) <- (8x128x6x160xf32)
        hardswish_11 = paddle._C_ops.hardswish(batch_norm__66)
        del batch_norm__66

        # pd_op.conv2d: (8x256x6x160xf32) <- (8x128x6x160xf32, 256x128x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            hardswish_11, parameter_135, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del hardswish_11, parameter_135

        # pd_op.batch_norm_: (8x256x6x160xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (8x256x6x160xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__72,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_6,
                parameter_134,
                parameter_133,
                parameter_132,
                parameter_131,
                True,
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
        del conv2d_6, parameter_131, parameter_132, parameter_133, parameter_134

        # pd_op.hardswish: (8x256x6x160xf32) <- (8x256x6x160xf32)
        hardswish_12 = paddle._C_ops.hardswish(batch_norm__72)
        del batch_norm__72

        # pd_op.depthwise_conv2d: (8x256x6x160xf32) <- (8x256x6x160xf32, 256x1x5x5xf32)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(
            hardswish_12, parameter_130, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del hardswish_12, parameter_130

        # pd_op.batch_norm_: (8x256x6x160xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (8x256x6x160xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__78,
            batch_norm__79,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
            batch_norm__83,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_6,
                parameter_129,
                parameter_128,
                parameter_127,
                parameter_126,
                True,
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
        del (
            depthwise_conv2d_6,
            parameter_126,
            parameter_127,
            parameter_128,
            parameter_129,
        )

        # pd_op.hardswish: (8x256x6x160xf32) <- (8x256x6x160xf32)
        hardswish_13 = paddle._C_ops.hardswish(batch_norm__78)
        del batch_norm__78

        # pd_op.conv2d: (8x256x6x160xf32) <- (8x256x6x160xf32, 256x256x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            hardswish_13, parameter_125, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del hardswish_13, parameter_125

        # pd_op.batch_norm_: (8x256x6x160xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (8x256x6x160xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__84,
            batch_norm__85,
            batch_norm__86,
            batch_norm__87,
            batch_norm__88,
            batch_norm__89,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_7,
                parameter_124,
                parameter_123,
                parameter_122,
                parameter_121,
                True,
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
        del conv2d_7, parameter_121, parameter_122, parameter_123, parameter_124

        # pd_op.hardswish: (8x256x6x160xf32) <- (8x256x6x160xf32)
        hardswish_14 = paddle._C_ops.hardswish(batch_norm__84)
        del batch_norm__84

        # pd_op.depthwise_conv2d: (8x256x6x160xf32) <- (8x256x6x160xf32, 256x1x5x5xf32)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(
            hardswish_14, parameter_120, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del hardswish_14, parameter_120

        # pd_op.batch_norm_: (8x256x6x160xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (8x256x6x160xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__90,
            batch_norm__91,
            batch_norm__92,
            batch_norm__93,
            batch_norm__94,
            batch_norm__95,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_7,
                parameter_119,
                parameter_118,
                parameter_117,
                parameter_116,
                True,
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
        del (
            depthwise_conv2d_7,
            parameter_116,
            parameter_117,
            parameter_118,
            parameter_119,
        )

        # pd_op.hardswish: (8x256x6x160xf32) <- (8x256x6x160xf32)
        hardswish_15 = paddle._C_ops.hardswish(batch_norm__90)
        del batch_norm__90

        # pd_op.conv2d: (8x256x6x160xf32) <- (8x256x6x160xf32, 256x256x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            hardswish_15, parameter_115, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del hardswish_15, parameter_115

        # pd_op.batch_norm_: (8x256x6x160xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (8x256x6x160xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__96,
            batch_norm__97,
            batch_norm__98,
            batch_norm__99,
            batch_norm__100,
            batch_norm__101,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_8,
                parameter_114,
                parameter_113,
                parameter_112,
                parameter_111,
                True,
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
        del conv2d_8, parameter_111, parameter_112, parameter_113, parameter_114

        # pd_op.hardswish: (8x256x6x160xf32) <- (8x256x6x160xf32)
        hardswish_16 = paddle._C_ops.hardswish(batch_norm__96)
        del batch_norm__96

        # pd_op.depthwise_conv2d: (8x256x6x160xf32) <- (8x256x6x160xf32, 256x1x5x5xf32)
        depthwise_conv2d_8 = paddle._C_ops.depthwise_conv2d(
            hardswish_16, parameter_110, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del hardswish_16, parameter_110

        # pd_op.batch_norm_: (8x256x6x160xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (8x256x6x160xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__102,
            batch_norm__103,
            batch_norm__104,
            batch_norm__105,
            batch_norm__106,
            batch_norm__107,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_8,
                parameter_109,
                parameter_108,
                parameter_107,
                parameter_106,
                True,
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
        del (
            depthwise_conv2d_8,
            parameter_106,
            parameter_107,
            parameter_108,
            parameter_109,
        )

        # pd_op.hardswish: (8x256x6x160xf32) <- (8x256x6x160xf32)
        hardswish_17 = paddle._C_ops.hardswish(batch_norm__102)
        del batch_norm__102

        # pd_op.conv2d: (8x256x6x160xf32) <- (8x256x6x160xf32, 256x256x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            hardswish_17, parameter_105, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del hardswish_17, parameter_105

        # pd_op.batch_norm_: (8x256x6x160xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (8x256x6x160xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__108,
            batch_norm__109,
            batch_norm__110,
            batch_norm__111,
            batch_norm__112,
            batch_norm__113,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_9,
                parameter_104,
                parameter_103,
                parameter_102,
                parameter_101,
                True,
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
        del conv2d_9, parameter_101, parameter_102, parameter_103, parameter_104

        # pd_op.hardswish: (8x256x6x160xf32) <- (8x256x6x160xf32)
        hardswish_18 = paddle._C_ops.hardswish(batch_norm__108)
        del batch_norm__108

        # pd_op.depthwise_conv2d: (8x256x6x160xf32) <- (8x256x6x160xf32, 256x1x5x5xf32)
        depthwise_conv2d_9 = paddle._C_ops.depthwise_conv2d(
            hardswish_18, parameter_100, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del hardswish_18, parameter_100

        # pd_op.batch_norm_: (8x256x6x160xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (8x256x6x160xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__114,
            batch_norm__115,
            batch_norm__116,
            batch_norm__117,
            batch_norm__118,
            batch_norm__119,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_9,
                parameter_99,
                parameter_98,
                parameter_97,
                parameter_96,
                True,
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
        del depthwise_conv2d_9, parameter_96, parameter_97, parameter_98, parameter_99

        # pd_op.hardswish: (8x256x6x160xf32) <- (8x256x6x160xf32)
        hardswish_19 = paddle._C_ops.hardswish(batch_norm__114)
        del batch_norm__114

        # pd_op.conv2d: (8x256x6x160xf32) <- (8x256x6x160xf32, 256x256x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            hardswish_19, parameter_95, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del hardswish_19, parameter_95

        # pd_op.batch_norm_: (8x256x6x160xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (8x256x6x160xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__120,
            batch_norm__121,
            batch_norm__122,
            batch_norm__123,
            batch_norm__124,
            batch_norm__125,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_10,
                parameter_94,
                parameter_93,
                parameter_92,
                parameter_91,
                True,
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
        del conv2d_10, parameter_91, parameter_92, parameter_93, parameter_94

        # pd_op.hardswish: (8x256x6x160xf32) <- (8x256x6x160xf32)
        hardswish_20 = paddle._C_ops.hardswish(batch_norm__120)
        del batch_norm__120

        # pd_op.depthwise_conv2d: (8x256x6x160xf32) <- (8x256x6x160xf32, 256x1x5x5xf32)
        depthwise_conv2d_10 = paddle._C_ops.depthwise_conv2d(
            hardswish_20, parameter_90, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del hardswish_20, parameter_90

        # pd_op.batch_norm_: (8x256x6x160xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (8x256x6x160xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__126,
            batch_norm__127,
            batch_norm__128,
            batch_norm__129,
            batch_norm__130,
            batch_norm__131,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_10,
                parameter_89,
                parameter_88,
                parameter_87,
                parameter_86,
                True,
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
        del depthwise_conv2d_10, parameter_86, parameter_87, parameter_88, parameter_89

        # pd_op.hardswish: (8x256x6x160xf32) <- (8x256x6x160xf32)
        hardswish_21 = paddle._C_ops.hardswish(batch_norm__126)
        del batch_norm__126

        # pd_op.conv2d: (8x256x6x160xf32) <- (8x256x6x160xf32, 256x256x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            hardswish_21, parameter_85, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del hardswish_21, parameter_85

        # pd_op.batch_norm_: (8x256x6x160xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (8x256x6x160xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__132,
            batch_norm__133,
            batch_norm__134,
            batch_norm__135,
            batch_norm__136,
            batch_norm__137,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_11,
                parameter_84,
                parameter_83,
                parameter_82,
                parameter_81,
                True,
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
        del conv2d_11, parameter_81, parameter_82, parameter_83, parameter_84

        # pd_op.hardswish: (8x256x6x160xf32) <- (8x256x6x160xf32)
        hardswish_22 = paddle._C_ops.hardswish(batch_norm__132)
        del batch_norm__132

        # pd_op.depthwise_conv2d: (8x256x3x160xf32) <- (8x256x6x160xf32, 256x1x5x5xf32)
        depthwise_conv2d_11 = paddle._C_ops.depthwise_conv2d(
            hardswish_22, parameter_80, [2, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del hardswish_22, parameter_80

        # pd_op.batch_norm_: (8x256x3x160xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (8x256x3x160xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__138,
            batch_norm__139,
            batch_norm__140,
            batch_norm__141,
            batch_norm__142,
            batch_norm__143,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_11,
                parameter_79,
                parameter_78,
                parameter_77,
                parameter_76,
                True,
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
        del depthwise_conv2d_11, parameter_76, parameter_77, parameter_78, parameter_79

        # pd_op.hardswish: (8x256x3x160xf32) <- (8x256x3x160xf32)
        hardswish_23 = paddle._C_ops.hardswish(batch_norm__138)
        del batch_norm__138

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.pool2d: (8x256x1x1xf32) <- (8x256x3x160xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            hardswish_23,
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

        # pd_op.conv2d: (8x64x1x1xf32) <- (8x256x1x1xf32, 64x256x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            pool2d_0, parameter_75, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_75, pool2d_0

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [1, -1, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_74, full_int_array_1)
        del parameter_74

        # pd_op.add: (8x64x1x1xf32) <- (8x64x1x1xf32, 1x64x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_12, reshape_0)
        del conv2d_12, reshape_0

        # pd_op.relu: (8x64x1x1xf32) <- (8x64x1x1xf32)
        relu_0 = paddle._C_ops.relu(add_0)
        del add_0

        # pd_op.conv2d: (8x256x1x1xf32) <- (8x64x1x1xf32, 256x64x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            relu_0, parameter_73, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_73, relu_0

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_72, full_int_array_1)
        del parameter_72

        # pd_op.add: (8x256x1x1xf32) <- (8x256x1x1xf32, 1x256x1x1xf32)
        add_1 = paddle._C_ops.add(conv2d_13, reshape_1)
        del conv2d_13, reshape_1

        # pd_op.hardsigmoid: (8x256x1x1xf32) <- (8x256x1x1xf32)
        hardsigmoid_0 = paddle._C_ops.hardsigmoid(
            add_1, float("0.166667"), float("0.5")
        )
        del add_1

        # pd_op.multiply: (8x256x3x160xf32) <- (8x256x3x160xf32, 8x256x1x1xf32)
        multiply_0 = paddle._C_ops.multiply(hardswish_23, hardsigmoid_0)
        del hardsigmoid_0, hardswish_23

        # pd_op.conv2d: (8x512x3x160xf32) <- (8x256x3x160xf32, 512x256x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            multiply_0, parameter_71, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_0, parameter_71

        # pd_op.batch_norm_: (8x512x3x160xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (8x512x3x160xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__144,
            batch_norm__145,
            batch_norm__146,
            batch_norm__147,
            batch_norm__148,
            batch_norm__149,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_14,
                parameter_70,
                parameter_69,
                parameter_68,
                parameter_67,
                True,
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
        del conv2d_14, parameter_67, parameter_68, parameter_69, parameter_70

        # pd_op.hardswish: (8x512x3x160xf32) <- (8x512x3x160xf32)
        hardswish_24 = paddle._C_ops.hardswish(batch_norm__144)
        del batch_norm__144

        # pd_op.depthwise_conv2d: (8x512x3x80xf32) <- (8x512x3x160xf32, 512x1x5x5xf32)
        depthwise_conv2d_12 = paddle._C_ops.depthwise_conv2d(
            hardswish_24, parameter_66, [1, 2], [2, 2], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del hardswish_24, parameter_66

        # pd_op.batch_norm_: (8x512x3x80xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (8x512x3x80xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__150,
            batch_norm__151,
            batch_norm__152,
            batch_norm__153,
            batch_norm__154,
            batch_norm__155,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_12,
                parameter_65,
                parameter_64,
                parameter_63,
                parameter_62,
                True,
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
        del depthwise_conv2d_12, parameter_62, parameter_63, parameter_64, parameter_65

        # pd_op.hardswish: (8x512x3x80xf32) <- (8x512x3x80xf32)
        hardswish_25 = paddle._C_ops.hardswish(batch_norm__150)
        del batch_norm__150

        # pd_op.pool2d: (8x512x1x1xf32) <- (8x512x3x80xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(
            hardswish_25,
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
        del full_int_array_0

        # pd_op.conv2d: (8x128x1x1xf32) <- (8x512x1x1xf32, 128x512x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            pool2d_1, parameter_61, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_61, pool2d_1

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(parameter_60, full_int_array_1)
        del parameter_60

        # pd_op.add: (8x128x1x1xf32) <- (8x128x1x1xf32, 1x128x1x1xf32)
        add_2 = paddle._C_ops.add(conv2d_15, reshape_2)
        del conv2d_15, reshape_2

        # pd_op.relu: (8x128x1x1xf32) <- (8x128x1x1xf32)
        relu_1 = paddle._C_ops.relu(add_2)
        del add_2

        # pd_op.conv2d: (8x512x1x1xf32) <- (8x128x1x1xf32, 512x128x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            relu_1, parameter_59, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_59, relu_1

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(parameter_58, full_int_array_1)
        del full_int_array_1, parameter_58

        # pd_op.add: (8x512x1x1xf32) <- (8x512x1x1xf32, 1x512x1x1xf32)
        add_3 = paddle._C_ops.add(conv2d_16, reshape_3)
        del conv2d_16, reshape_3

        # pd_op.hardsigmoid: (8x512x1x1xf32) <- (8x512x1x1xf32)
        hardsigmoid_1 = paddle._C_ops.hardsigmoid(
            add_3, float("0.166667"), float("0.5")
        )
        del add_3

        # pd_op.multiply: (8x512x3x80xf32) <- (8x512x3x80xf32, 8x512x1x1xf32)
        multiply_1 = paddle._C_ops.multiply(hardswish_25, hardsigmoid_1)
        del hardsigmoid_1, hardswish_25

        # pd_op.conv2d: (8x512x3x80xf32) <- (8x512x3x80xf32, 512x512x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            multiply_1, parameter_57, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_1, parameter_57

        # pd_op.batch_norm_: (8x512x3x80xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (8x512x3x80xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__156,
            batch_norm__157,
            batch_norm__158,
            batch_norm__159,
            batch_norm__160,
            batch_norm__161,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_17,
                parameter_56,
                parameter_55,
                parameter_54,
                parameter_53,
                True,
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
        del conv2d_17, parameter_53, parameter_54, parameter_55, parameter_56

        # pd_op.hardswish: (8x512x3x80xf32) <- (8x512x3x80xf32)
        hardswish_26 = paddle._C_ops.hardswish(batch_norm__156)
        del batch_norm__156

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [2, 2]

        # pd_op.pool2d: (8x512x1x40xf32) <- (8x512x3x80xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(
            hardswish_26,
            full_int_array_2,
            [2, 2],
            [0, 0],
            False,
            True,
            "NCHW",
            "avg",
            False,
            False,
            "EXPLICIT",
        )
        del full_int_array_2, hardswish_26

        # pd_op.assign: (8x512x1x40xf32) <- (8x512x1x40xf32)
        assign_0 = pool2d_2
        del pool2d_2

        # pd_op.conv2d: (8x64x1x40xf32) <- (8x512x1x40xf32, 64x512x3x3xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            assign_0, parameter_52, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_52

        # pd_op.batch_norm_: (8x64x1x40xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (8x64x1x40xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_51,
                parameter_50,
                parameter_49,
                parameter_48,
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
        del conv2d_18, parameter_48, parameter_49, parameter_50, parameter_51

        # pd_op.swish: (8x64x1x40xf32) <- (8x64x1x40xf32)
        swish_0 = paddle._C_ops.swish(batch_norm__162)
        del batch_norm__162

        # pd_op.conv2d: (8x120x1x40xf32) <- (8x64x1x40xf32, 120x64x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            swish_0, parameter_47, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_47, swish_0

        # pd_op.batch_norm_: (8x120x1x40xf32, 120xf32, 120xf32, 120xf32, 120xf32, -1xui8) <- (8x120x1x40xf32, 120xf32, 120xf32, 120xf32, 120xf32)
        (
            batch_norm__168,
            batch_norm__169,
            batch_norm__170,
            batch_norm__171,
            batch_norm__172,
            batch_norm__173,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_19,
                parameter_46,
                parameter_45,
                parameter_44,
                parameter_43,
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
        del conv2d_19, parameter_43, parameter_44, parameter_45, parameter_46

        # pd_op.swish: (8x120x1x40xf32) <- (8x120x1x40xf32)
        swish_1 = paddle._C_ops.swish(batch_norm__168)
        del batch_norm__168

        # pd_op.flatten: (8x120x40xf32) <- (8x120x1x40xf32)
        flatten_0 = paddle._C_ops.flatten(swish_1, 2, 3)
        del swish_1

        # pd_op.transpose: (8x40x120xf32) <- (8x120x40xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_0, [0, 2, 1])
        del flatten_0

        # pd_op.layer_norm: (8x40x120xf32, 8x40xf32, 8x40xf32) <- (8x40x120xf32, 120xf32, 120xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_0, parameter_42, parameter_41, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_41, parameter_42

        # pd_op.matmul: (8x40x360xf32) <- (8x40x120xf32, 120x360xf32)
        matmul_0 = paddle._C_ops.matmul(layer_norm_0, parameter_40, False, False)
        del layer_norm_0, parameter_40

        # pd_op.add: (8x40x360xf32) <- (8x40x360xf32, 360xf32)
        add_4 = paddle._C_ops.add(matmul_0, parameter_39)
        del matmul_0, parameter_39

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_3 = [0, -1, 3, 8, 15]

        # pd_op.reshape: (8x40x3x8x15xf32) <- (8x40x360xf32, 5xi64)
        reshape_4 = paddle._C_ops.reshape(add_4, full_int_array_3)
        del add_4

        # pd_op.transpose: (3x8x8x40x15xf32) <- (8x40x3x8x15xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_4, [2, 0, 3, 1, 4])
        del reshape_4

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [1]

        # pd_op.slice: (8x8x40x15xf32) <- (3x8x8x40x15xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            transpose_1, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.258199"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (8x8x40x15xf32) <- (8x8x40x15xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float("0"), True)
        del slice_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [2]

        # pd_op.slice: (8x8x40x15xf32) <- (3x8x8x40x15xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            transpose_1, [0], full_int_array_5, full_int_array_6, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [3]

        # pd_op.slice: (8x8x40x15xf32) <- (3x8x8x40x15xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            transpose_1, [0], full_int_array_6, full_int_array_7, [1], [0]
        )
        del transpose_1

        # pd_op.transpose: (8x8x15x40xf32) <- (8x8x40x15xf32)
        transpose_2 = paddle._C_ops.transpose(slice_1, [0, 1, 3, 2])
        del slice_1

        # pd_op.matmul: (8x8x40x40xf32) <- (8x8x40x15xf32, 8x8x15x40xf32)
        matmul_1 = paddle._C_ops.matmul(scale_0, transpose_2, False, False)
        del scale_0, transpose_2

        # pd_op.softmax: (8x8x40x40xf32) <- (8x8x40x40xf32)
        softmax_1 = paddle._C_ops.softmax(matmul_1, -1)
        del matmul_1

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (8x8x40x40xf32, 8x8x40x40xui8) <- (8x8x40x40xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_1, None, full_1, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_1

        # pd_op.matmul: (8x8x40x15xf32) <- (8x8x40x40xf32, 8x8x40x15xf32)
        matmul_2 = paddle._C_ops.matmul(dropout_0, slice_2, False, False)
        del dropout_0, slice_2

        # pd_op.transpose: (8x40x8x15xf32) <- (8x8x40x15xf32)
        transpose_3 = paddle._C_ops.transpose(matmul_2, [0, 2, 1, 3])
        del matmul_2

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_8 = [0, -1, 120]

        # pd_op.reshape: (8x40x120xf32) <- (8x40x8x15xf32, 3xi64)
        reshape_5 = paddle._C_ops.reshape(transpose_3, full_int_array_8)
        del transpose_3

        # pd_op.matmul: (8x40x120xf32) <- (8x40x120xf32, 120x120xf32)
        matmul_3 = paddle._C_ops.matmul(reshape_5, parameter_38, False, False)
        del parameter_38, reshape_5

        # pd_op.add: (8x40x120xf32) <- (8x40x120xf32, 120xf32)
        add_5 = paddle._C_ops.add(matmul_3, parameter_37)
        del matmul_3, parameter_37

        # pd_op.dropout: (8x40x120xf32, 8x40x120xui8) <- (8x40x120xf32, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_5, None, full_1, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_5

        # pd_op.add: (8x40x120xf32) <- (8x40x120xf32, 8x40x120xf32)
        add_6 = paddle._C_ops.add(transpose_0, dropout_2)
        del dropout_2, transpose_0

        # pd_op.layer_norm: (8x40x120xf32, 8x40xf32, 8x40xf32) <- (8x40x120xf32, 120xf32, 120xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_6, parameter_36, parameter_35, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_35, parameter_36

        # pd_op.matmul: (8x40x240xf32) <- (8x40x120xf32, 120x240xf32)
        matmul_4 = paddle._C_ops.matmul(layer_norm_3, parameter_34, False, False)
        del layer_norm_3, parameter_34

        # pd_op.add: (8x40x240xf32) <- (8x40x240xf32, 240xf32)
        add_7 = paddle._C_ops.add(matmul_4, parameter_33)
        del matmul_4, parameter_33

        # pd_op.swish: (8x40x240xf32) <- (8x40x240xf32)
        swish_2 = paddle._C_ops.swish(add_7)
        del add_7

        # pd_op.dropout: (8x40x240xf32, 8x40x240xui8) <- (8x40x240xf32, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                swish_2, None, full_1, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del swish_2

        # pd_op.matmul: (8x40x120xf32) <- (8x40x240xf32, 240x120xf32)
        matmul_5 = paddle._C_ops.matmul(dropout_4, parameter_32, False, False)
        del dropout_4, parameter_32

        # pd_op.add: (8x40x120xf32) <- (8x40x120xf32, 120xf32)
        add_8 = paddle._C_ops.add(matmul_5, parameter_31)
        del matmul_5, parameter_31

        # pd_op.dropout: (8x40x120xf32, 8x40x120xui8) <- (8x40x120xf32, None, 1xf32)
        dropout_6, dropout_7 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_8, None, full_1, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_8

        # pd_op.add: (8x40x120xf32) <- (8x40x120xf32, 8x40x120xf32)
        add_9 = paddle._C_ops.add(add_6, dropout_6)
        del add_6, dropout_6

        # pd_op.layer_norm: (8x40x120xf32, 8x40xf32, 8x40xf32) <- (8x40x120xf32, 120xf32, 120xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_9, parameter_30, parameter_29, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_29, parameter_30

        # pd_op.matmul: (8x40x360xf32) <- (8x40x120xf32, 120x360xf32)
        matmul_6 = paddle._C_ops.matmul(layer_norm_6, parameter_28, False, False)
        del layer_norm_6, parameter_28

        # pd_op.add: (8x40x360xf32) <- (8x40x360xf32, 360xf32)
        add_10 = paddle._C_ops.add(matmul_6, parameter_27)
        del matmul_6, parameter_27

        # pd_op.reshape: (8x40x3x8x15xf32) <- (8x40x360xf32, 5xi64)
        reshape_6 = paddle._C_ops.reshape(add_10, full_int_array_3)
        del add_10, full_int_array_3

        # pd_op.transpose: (3x8x8x40x15xf32) <- (8x40x3x8x15xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_6, [2, 0, 3, 1, 4])
        del reshape_6

        # pd_op.slice: (8x8x40x15xf32) <- (3x8x8x40x15xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            transpose_4, [0], full_int_array_4, full_int_array_5, [1], [0]
        )
        del full_int_array_4

        # pd_op.scale: (8x8x40x15xf32) <- (8x8x40x15xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_3, full_0, float("0"), True)
        del full_0, slice_3

        # pd_op.slice: (8x8x40x15xf32) <- (3x8x8x40x15xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            transpose_4, [0], full_int_array_5, full_int_array_6, [1], [0]
        )
        del full_int_array_5

        # pd_op.slice: (8x8x40x15xf32) <- (3x8x8x40x15xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            transpose_4, [0], full_int_array_6, full_int_array_7, [1], [0]
        )
        del full_int_array_7, transpose_4

        # pd_op.transpose: (8x8x15x40xf32) <- (8x8x40x15xf32)
        transpose_5 = paddle._C_ops.transpose(slice_4, [0, 1, 3, 2])
        del slice_4

        # pd_op.matmul: (8x8x40x40xf32) <- (8x8x40x15xf32, 8x8x15x40xf32)
        matmul_7 = paddle._C_ops.matmul(scale_1, transpose_5, False, False)
        del scale_1, transpose_5

        # pd_op.softmax: (8x8x40x40xf32) <- (8x8x40x40xf32)
        softmax_2 = paddle._C_ops.softmax(matmul_7, -1)
        del matmul_7

        # pd_op.dropout: (8x8x40x40xf32, 8x8x40x40xui8) <- (8x8x40x40xf32, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_2, None, full_1, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_2

        # pd_op.matmul: (8x8x40x15xf32) <- (8x8x40x40xf32, 8x8x40x15xf32)
        matmul_8 = paddle._C_ops.matmul(dropout_8, slice_5, False, False)
        del dropout_8, slice_5

        # pd_op.transpose: (8x40x8x15xf32) <- (8x8x40x15xf32)
        transpose_6 = paddle._C_ops.transpose(matmul_8, [0, 2, 1, 3])
        del matmul_8

        # pd_op.reshape: (8x40x120xf32) <- (8x40x8x15xf32, 3xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_6, full_int_array_8)
        del full_int_array_8, transpose_6

        # pd_op.matmul: (8x40x120xf32) <- (8x40x120xf32, 120x120xf32)
        matmul_9 = paddle._C_ops.matmul(reshape_7, parameter_26, False, False)
        del parameter_26, reshape_7

        # pd_op.add: (8x40x120xf32) <- (8x40x120xf32, 120xf32)
        add_11 = paddle._C_ops.add(matmul_9, parameter_25)
        del matmul_9, parameter_25

        # pd_op.dropout: (8x40x120xf32, 8x40x120xui8) <- (8x40x120xf32, None, 1xf32)
        dropout_10, dropout_11 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_11, None, full_1, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_11

        # pd_op.add: (8x40x120xf32) <- (8x40x120xf32, 8x40x120xf32)
        add_12 = paddle._C_ops.add(add_9, dropout_10)
        del add_9, dropout_10

        # pd_op.layer_norm: (8x40x120xf32, 8x40xf32, 8x40xf32) <- (8x40x120xf32, 120xf32, 120xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_12, parameter_24, parameter_23, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_23, parameter_24

        # pd_op.matmul: (8x40x240xf32) <- (8x40x120xf32, 120x240xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_9, parameter_22, False, False)
        del layer_norm_9, parameter_22

        # pd_op.add: (8x40x240xf32) <- (8x40x240xf32, 240xf32)
        add_13 = paddle._C_ops.add(matmul_10, parameter_21)
        del matmul_10, parameter_21

        # pd_op.swish: (8x40x240xf32) <- (8x40x240xf32)
        swish_3 = paddle._C_ops.swish(add_13)
        del add_13

        # pd_op.dropout: (8x40x240xf32, 8x40x240xui8) <- (8x40x240xf32, None, 1xf32)
        dropout_12, dropout_13 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                swish_3, None, full_1, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del swish_3

        # pd_op.matmul: (8x40x120xf32) <- (8x40x240xf32, 240x120xf32)
        matmul_11 = paddle._C_ops.matmul(dropout_12, parameter_20, False, False)
        del dropout_12, parameter_20

        # pd_op.add: (8x40x120xf32) <- (8x40x120xf32, 120xf32)
        add_14 = paddle._C_ops.add(matmul_11, parameter_19)
        del matmul_11, parameter_19

        # pd_op.dropout: (8x40x120xf32, 8x40x120xui8) <- (8x40x120xf32, None, 1xf32)
        dropout_14, dropout_15 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_14, None, full_1, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_14, full_1

        # pd_op.add: (8x40x120xf32) <- (8x40x120xf32, 8x40x120xf32)
        add_15 = paddle._C_ops.add(add_12, dropout_14)
        del add_12, dropout_14

        # pd_op.layer_norm: (8x40x120xf32, 8x40xf32, 8x40xf32) <- (8x40x120xf32, 120xf32, 120xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_15, parameter_18, parameter_17, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_15, parameter_17, parameter_18

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_9 = [0, 1, 40, 120]

        # pd_op.reshape: (8x1x40x120xf32) <- (8x40x120xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(layer_norm_12, full_int_array_9)
        del full_int_array_9, layer_norm_12

        # pd_op.transpose: (8x120x1x40xf32) <- (8x1x40x120xf32)
        transpose_7 = paddle._C_ops.transpose(reshape_8, [0, 3, 1, 2])
        del reshape_8

        # pd_op.conv2d: (8x512x1x40xf32) <- (8x120x1x40xf32, 512x120x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            transpose_7, parameter_16, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_16, transpose_7

        # pd_op.batch_norm_: (8x512x1x40xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (8x512x1x40xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__174,
            batch_norm__175,
            batch_norm__176,
            batch_norm__177,
            batch_norm__178,
            batch_norm__179,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_20,
                parameter_15,
                parameter_14,
                parameter_13,
                parameter_12,
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
        del conv2d_20, parameter_12, parameter_13, parameter_14, parameter_15

        # pd_op.swish: (8x512x1x40xf32) <- (8x512x1x40xf32)
        swish_4 = paddle._C_ops.swish(batch_norm__174)
        del batch_norm__174

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([8x512x1x40xf32, 8x512x1x40xf32]) <- (8x512x1x40xf32, 8x512x1x40xf32)
        combine_0 = [assign_0, swish_4]
        del assign_0, swish_4

        # pd_op.concat: (8x1024x1x40xf32) <- ([8x512x1x40xf32, 8x512x1x40xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_2)
        del combine_0, full_2

        # pd_op.conv2d: (8x64x1x40xf32) <- (8x1024x1x40xf32, 64x1024x3x3xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            concat_0, parameter_11, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_0, parameter_11

        # pd_op.batch_norm_: (8x64x1x40xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (8x64x1x40xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__180,
            batch_norm__181,
            batch_norm__182,
            batch_norm__183,
            batch_norm__184,
            batch_norm__185,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_21,
                parameter_10,
                parameter_9,
                parameter_8,
                parameter_7,
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
        del conv2d_21, parameter_10, parameter_7, parameter_8, parameter_9

        # pd_op.swish: (8x64x1x40xf32) <- (8x64x1x40xf32)
        swish_5 = paddle._C_ops.swish(batch_norm__180)
        del batch_norm__180

        # pd_op.conv2d: (8x64x1x40xf32) <- (8x64x1x40xf32, 64x64x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(
            swish_5, parameter_6, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_6, swish_5

        # pd_op.batch_norm_: (8x64x1x40xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (8x64x1x40xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__186,
            batch_norm__187,
            batch_norm__188,
            batch_norm__189,
            batch_norm__190,
            batch_norm__191,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_22,
                parameter_5,
                parameter_4,
                parameter_3,
                parameter_2,
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
        del conv2d_22, parameter_2, parameter_3, parameter_4, parameter_5

        # pd_op.swish: (8x64x1x40xf32) <- (8x64x1x40xf32)
        swish_6 = paddle._C_ops.swish(batch_norm__186)
        del batch_norm__186

        # pd_op.squeeze: (8x64x40xf32) <- (8x64x1x40xf32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(swish_6, full_int_array_6)
        del full_int_array_6, swish_6

        # pd_op.transpose: (8x40x64xf32) <- (8x64x40xf32)
        transpose_8 = paddle._C_ops.transpose(squeeze_0, [0, 2, 1])
        del squeeze_0

        # pd_op.matmul: (8x40x6625xf32) <- (8x40x64xf32, 64x6625xf32)
        matmul_12 = paddle._C_ops.matmul(transpose_8, parameter_1, False, False)
        del parameter_1, transpose_8

        # pd_op.add: (8x40x6625xf32) <- (8x40x6625xf32, 6625xf32)
        add_16 = paddle._C_ops.add(matmul_12, parameter_0)
        del matmul_12, parameter_0

        # pd_op.softmax: (8x40x6625xf32) <- (8x40x6625xf32)
        softmax_0 = paddle._C_ops.softmax(add_16, 2)
        del add_16

        return softmax_0
