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
        data_0,
    ):
        # pd_op.conv2d: (8x16x24x160xf32) <- (8x3x48x320xf32, 16x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_142, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_0, parameter_142

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
                parameter_141,
                parameter_140,
                parameter_139,
                parameter_138,
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
        del parameter_138, parameter_139, parameter_140, parameter_141

        # pd_op.hardswish: (8x16x24x160xf32) <- (8x16x24x160xf32)
        hardswish_0 = paddle._C_ops.hardswish(batch_norm__0)

        # pd_op.depthwise_conv2d: (8x16x24x160xf32) <- (8x16x24x160xf32, 16x1x3x3xf32)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(
            hardswish_0, parameter_137, [1, 1], [1, 1], "EXPLICIT", 16, [1, 1], "NCHW"
        )
        del parameter_137

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
                parameter_136,
                parameter_135,
                parameter_134,
                parameter_133,
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
        del parameter_133, parameter_134, parameter_135, parameter_136

        # pd_op.hardswish: (8x16x24x160xf32) <- (8x16x24x160xf32)
        hardswish_1 = paddle._C_ops.hardswish(batch_norm__6)

        # pd_op.conv2d: (8x32x24x160xf32) <- (8x16x24x160xf32, 32x16x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            hardswish_1, parameter_132, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_132

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
                parameter_131,
                parameter_130,
                parameter_129,
                parameter_128,
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
        del parameter_128, parameter_129, parameter_130, parameter_131

        # pd_op.hardswish: (8x32x24x160xf32) <- (8x32x24x160xf32)
        hardswish_2 = paddle._C_ops.hardswish(batch_norm__12)

        # pd_op.depthwise_conv2d: (8x32x24x160xf32) <- (8x32x24x160xf32, 32x1x3x3xf32)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(
            hardswish_2, parameter_127, [1, 1], [1, 1], "EXPLICIT", 32, [1, 1], "NCHW"
        )
        del parameter_127

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
                parameter_126,
                parameter_125,
                parameter_124,
                parameter_123,
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
        del parameter_123, parameter_124, parameter_125, parameter_126

        # pd_op.hardswish: (8x32x24x160xf32) <- (8x32x24x160xf32)
        hardswish_3 = paddle._C_ops.hardswish(batch_norm__18)

        # pd_op.conv2d: (8x64x24x160xf32) <- (8x32x24x160xf32, 64x32x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            hardswish_3, parameter_122, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_122

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
                parameter_121,
                parameter_120,
                parameter_119,
                parameter_118,
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
        del parameter_118, parameter_119, parameter_120, parameter_121

        # pd_op.hardswish: (8x64x24x160xf32) <- (8x64x24x160xf32)
        hardswish_4 = paddle._C_ops.hardswish(batch_norm__24)

        # pd_op.depthwise_conv2d: (8x64x24x160xf32) <- (8x64x24x160xf32, 64x1x3x3xf32)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(
            hardswish_4, parameter_117, [1, 1], [1, 1], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del parameter_117

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
                parameter_116,
                parameter_115,
                parameter_114,
                parameter_113,
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
        del parameter_113, parameter_114, parameter_115, parameter_116

        # pd_op.hardswish: (8x64x24x160xf32) <- (8x64x24x160xf32)
        hardswish_5 = paddle._C_ops.hardswish(batch_norm__30)

        # pd_op.conv2d: (8x64x24x160xf32) <- (8x64x24x160xf32, 64x64x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            hardswish_5, parameter_112, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_112

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
                parameter_111,
                parameter_110,
                parameter_109,
                parameter_108,
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
        del parameter_108, parameter_109, parameter_110, parameter_111

        # pd_op.hardswish: (8x64x24x160xf32) <- (8x64x24x160xf32)
        hardswish_6 = paddle._C_ops.hardswish(batch_norm__36)

        # pd_op.depthwise_conv2d: (8x64x12x160xf32) <- (8x64x24x160xf32, 64x1x3x3xf32)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(
            hardswish_6, parameter_107, [2, 1], [1, 1], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del parameter_107

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
                parameter_106,
                parameter_105,
                parameter_104,
                parameter_103,
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
        del parameter_103, parameter_104, parameter_105, parameter_106

        # pd_op.hardswish: (8x64x12x160xf32) <- (8x64x12x160xf32)
        hardswish_7 = paddle._C_ops.hardswish(batch_norm__42)

        # pd_op.conv2d: (8x128x12x160xf32) <- (8x64x12x160xf32, 128x64x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            hardswish_7, parameter_102, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_102

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
                parameter_101,
                parameter_100,
                parameter_99,
                parameter_98,
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
        del parameter_100, parameter_101, parameter_98, parameter_99

        # pd_op.hardswish: (8x128x12x160xf32) <- (8x128x12x160xf32)
        hardswish_8 = paddle._C_ops.hardswish(batch_norm__48)

        # pd_op.depthwise_conv2d: (8x128x12x160xf32) <- (8x128x12x160xf32, 128x1x3x3xf32)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(
            hardswish_8, parameter_97, [1, 1], [1, 1], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del parameter_97

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
                parameter_96,
                parameter_95,
                parameter_94,
                parameter_93,
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
        del parameter_93, parameter_94, parameter_95, parameter_96

        # pd_op.hardswish: (8x128x12x160xf32) <- (8x128x12x160xf32)
        hardswish_9 = paddle._C_ops.hardswish(batch_norm__54)

        # pd_op.conv2d: (8x128x12x160xf32) <- (8x128x12x160xf32, 128x128x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            hardswish_9, parameter_92, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_92

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
                parameter_91,
                parameter_90,
                parameter_89,
                parameter_88,
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
        del parameter_88, parameter_89, parameter_90, parameter_91

        # pd_op.hardswish: (8x128x12x160xf32) <- (8x128x12x160xf32)
        hardswish_10 = paddle._C_ops.hardswish(batch_norm__60)

        # pd_op.depthwise_conv2d: (8x128x6x160xf32) <- (8x128x12x160xf32, 128x1x3x3xf32)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(
            hardswish_10, parameter_87, [2, 1], [1, 1], "EXPLICIT", 128, [1, 1], "NCHW"
        )
        del parameter_87

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
                parameter_86,
                parameter_85,
                parameter_84,
                parameter_83,
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
        del parameter_83, parameter_84, parameter_85, parameter_86

        # pd_op.hardswish: (8x128x6x160xf32) <- (8x128x6x160xf32)
        hardswish_11 = paddle._C_ops.hardswish(batch_norm__66)

        # pd_op.conv2d: (8x256x6x160xf32) <- (8x128x6x160xf32, 256x128x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            hardswish_11, parameter_82, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_82

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
                parameter_81,
                parameter_80,
                parameter_79,
                parameter_78,
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
        del parameter_78, parameter_79, parameter_80, parameter_81

        # pd_op.hardswish: (8x256x6x160xf32) <- (8x256x6x160xf32)
        hardswish_12 = paddle._C_ops.hardswish(batch_norm__72)

        # pd_op.depthwise_conv2d: (8x256x6x160xf32) <- (8x256x6x160xf32, 256x1x5x5xf32)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(
            hardswish_12, parameter_77, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del parameter_77

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
                parameter_76,
                parameter_75,
                parameter_74,
                parameter_73,
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
        del parameter_73, parameter_74, parameter_75, parameter_76

        # pd_op.hardswish: (8x256x6x160xf32) <- (8x256x6x160xf32)
        hardswish_13 = paddle._C_ops.hardswish(batch_norm__78)

        # pd_op.conv2d: (8x256x6x160xf32) <- (8x256x6x160xf32, 256x256x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            hardswish_13, parameter_72, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_72

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
                parameter_71,
                parameter_70,
                parameter_69,
                parameter_68,
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
        del parameter_68, parameter_69, parameter_70, parameter_71

        # pd_op.hardswish: (8x256x6x160xf32) <- (8x256x6x160xf32)
        hardswish_14 = paddle._C_ops.hardswish(batch_norm__84)

        # pd_op.depthwise_conv2d: (8x256x6x160xf32) <- (8x256x6x160xf32, 256x1x5x5xf32)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(
            hardswish_14, parameter_67, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del parameter_67

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
                parameter_66,
                parameter_65,
                parameter_64,
                parameter_63,
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
        del parameter_63, parameter_64, parameter_65, parameter_66

        # pd_op.hardswish: (8x256x6x160xf32) <- (8x256x6x160xf32)
        hardswish_15 = paddle._C_ops.hardswish(batch_norm__90)

        # pd_op.conv2d: (8x256x6x160xf32) <- (8x256x6x160xf32, 256x256x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            hardswish_15, parameter_62, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_62

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
                parameter_61,
                parameter_60,
                parameter_59,
                parameter_58,
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
        del parameter_58, parameter_59, parameter_60, parameter_61

        # pd_op.hardswish: (8x256x6x160xf32) <- (8x256x6x160xf32)
        hardswish_16 = paddle._C_ops.hardswish(batch_norm__96)

        # pd_op.depthwise_conv2d: (8x256x6x160xf32) <- (8x256x6x160xf32, 256x1x5x5xf32)
        depthwise_conv2d_8 = paddle._C_ops.depthwise_conv2d(
            hardswish_16, parameter_57, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del parameter_57

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
                parameter_56,
                parameter_55,
                parameter_54,
                parameter_53,
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
        del parameter_53, parameter_54, parameter_55, parameter_56

        # pd_op.hardswish: (8x256x6x160xf32) <- (8x256x6x160xf32)
        hardswish_17 = paddle._C_ops.hardswish(batch_norm__102)

        # pd_op.conv2d: (8x256x6x160xf32) <- (8x256x6x160xf32, 256x256x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            hardswish_17, parameter_52, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_52

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
                parameter_51,
                parameter_50,
                parameter_49,
                parameter_48,
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
        del parameter_48, parameter_49, parameter_50, parameter_51

        # pd_op.hardswish: (8x256x6x160xf32) <- (8x256x6x160xf32)
        hardswish_18 = paddle._C_ops.hardswish(batch_norm__108)

        # pd_op.depthwise_conv2d: (8x256x6x160xf32) <- (8x256x6x160xf32, 256x1x5x5xf32)
        depthwise_conv2d_9 = paddle._C_ops.depthwise_conv2d(
            hardswish_18, parameter_47, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del parameter_47

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
        del parameter_43, parameter_44, parameter_45, parameter_46

        # pd_op.hardswish: (8x256x6x160xf32) <- (8x256x6x160xf32)
        hardswish_19 = paddle._C_ops.hardswish(batch_norm__114)

        # pd_op.conv2d: (8x256x6x160xf32) <- (8x256x6x160xf32, 256x256x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            hardswish_19, parameter_42, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_42

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
        del parameter_38, parameter_39, parameter_40, parameter_41

        # pd_op.hardswish: (8x256x6x160xf32) <- (8x256x6x160xf32)
        hardswish_20 = paddle._C_ops.hardswish(batch_norm__120)

        # pd_op.depthwise_conv2d: (8x256x6x160xf32) <- (8x256x6x160xf32, 256x1x5x5xf32)
        depthwise_conv2d_10 = paddle._C_ops.depthwise_conv2d(
            hardswish_20, parameter_37, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del parameter_37

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
        del parameter_33, parameter_34, parameter_35, parameter_36

        # pd_op.hardswish: (8x256x6x160xf32) <- (8x256x6x160xf32)
        hardswish_21 = paddle._C_ops.hardswish(batch_norm__126)

        # pd_op.conv2d: (8x256x6x160xf32) <- (8x256x6x160xf32, 256x256x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            hardswish_21, parameter_32, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_32

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
        del parameter_28, parameter_29, parameter_30, parameter_31

        # pd_op.hardswish: (8x256x6x160xf32) <- (8x256x6x160xf32)
        hardswish_22 = paddle._C_ops.hardswish(batch_norm__132)

        # pd_op.depthwise_conv2d: (8x256x3x160xf32) <- (8x256x6x160xf32, 256x1x5x5xf32)
        depthwise_conv2d_11 = paddle._C_ops.depthwise_conv2d(
            hardswish_22, parameter_27, [2, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del parameter_27

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
        del parameter_23, parameter_24, parameter_25, parameter_26

        # pd_op.hardswish: (8x256x3x160xf32) <- (8x256x3x160xf32)
        hardswish_23 = paddle._C_ops.hardswish(batch_norm__138)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_0 = full_int_array_0

        # pd_op.pool2d: (8x256x1x1xf32) <- (8x256x3x160xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(
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
            pool2d_1, parameter_22, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_22

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [1, -1, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_21, full_int_array_1)
        del parameter_21

        # pd_op.add: (8x64x1x1xf32) <- (8x64x1x1xf32, 1x64x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_12, reshape_0)

        # pd_op.relu: (8x64x1x1xf32) <- (8x64x1x1xf32)
        relu_0 = paddle._C_ops.relu(add_0)
        del add_0

        # pd_op.conv2d: (8x256x1x1xf32) <- (8x64x1x1xf32, 256x64x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            relu_0, parameter_20, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_20

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_19, full_int_array_1)
        del parameter_19

        # pd_op.add: (8x256x1x1xf32) <- (8x256x1x1xf32, 1x256x1x1xf32)
        add_1 = paddle._C_ops.add(conv2d_13, reshape_1)

        # pd_op.hardsigmoid: (8x256x1x1xf32) <- (8x256x1x1xf32)
        hardsigmoid_0 = paddle._C_ops.hardsigmoid(
            add_1, float("0.166667"), float("0.5")
        )
        del add_1

        # pd_op.multiply: (8x256x3x160xf32) <- (8x256x3x160xf32, 8x256x1x1xf32)
        multiply_0 = paddle._C_ops.multiply(hardswish_23, hardsigmoid_0)

        # pd_op.conv2d: (8x512x3x160xf32) <- (8x256x3x160xf32, 512x256x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            multiply_0, parameter_18, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_18

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
                parameter_17,
                parameter_16,
                parameter_15,
                parameter_14,
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
        del parameter_14, parameter_15, parameter_16, parameter_17

        # pd_op.hardswish: (8x512x3x160xf32) <- (8x512x3x160xf32)
        hardswish_24 = paddle._C_ops.hardswish(batch_norm__144)

        # pd_op.depthwise_conv2d: (8x512x3x80xf32) <- (8x512x3x160xf32, 512x1x5x5xf32)
        depthwise_conv2d_12 = paddle._C_ops.depthwise_conv2d(
            hardswish_24, parameter_13, [1, 2], [2, 2], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del parameter_13

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
                parameter_12,
                parameter_11,
                parameter_10,
                parameter_9,
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
        del parameter_10, parameter_11, parameter_12, parameter_9

        # pd_op.hardswish: (8x512x3x80xf32) <- (8x512x3x80xf32)
        hardswish_25 = paddle._C_ops.hardswish(batch_norm__150)

        # pd_op.pool2d: (8x512x1x1xf32) <- (8x512x3x80xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(
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

        # pd_op.conv2d: (8x128x1x1xf32) <- (8x512x1x1xf32, 128x512x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            pool2d_2, parameter_8, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_8

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(parameter_7, full_int_array_1)
        del parameter_7

        # pd_op.add: (8x128x1x1xf32) <- (8x128x1x1xf32, 1x128x1x1xf32)
        add_2 = paddle._C_ops.add(conv2d_15, reshape_2)

        # pd_op.relu: (8x128x1x1xf32) <- (8x128x1x1xf32)
        relu_1 = paddle._C_ops.relu(add_2)
        del add_2

        # pd_op.conv2d: (8x512x1x1xf32) <- (8x128x1x1xf32, 512x128x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            relu_1, parameter_6, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_6

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(parameter_5, full_int_array_1)
        del full_int_array_1, parameter_5

        # pd_op.add: (8x512x1x1xf32) <- (8x512x1x1xf32, 1x512x1x1xf32)
        add_3 = paddle._C_ops.add(conv2d_16, reshape_3)

        # pd_op.hardsigmoid: (8x512x1x1xf32) <- (8x512x1x1xf32)
        hardsigmoid_1 = paddle._C_ops.hardsigmoid(
            add_3, float("0.166667"), float("0.5")
        )
        del add_3

        # pd_op.multiply: (8x512x3x80xf32) <- (8x512x3x80xf32, 8x512x1x1xf32)
        multiply_1 = paddle._C_ops.multiply(hardswish_25, hardsigmoid_1)

        # pd_op.conv2d: (8x512x3x80xf32) <- (8x512x3x80xf32, 512x512x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            multiply_1, parameter_4, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_4

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
                parameter_3,
                parameter_2,
                parameter_1,
                parameter_0,
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
        del parameter_0, parameter_1, parameter_2, parameter_3

        # pd_op.hardswish: (8x512x3x80xf32) <- (8x512x3x80xf32)
        hardswish_26 = paddle._C_ops.hardswish(batch_norm__156)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [2, 2]

        # pd_op.pool2d: (8x512x1x40xf32) <- (8x512x3x80xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
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
        del (
            assign_0,
            batch_norm__0,
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
            batch_norm__108,
            batch_norm__109,
            batch_norm__11,
            batch_norm__110,
            batch_norm__111,
            batch_norm__112,
            batch_norm__113,
            batch_norm__114,
            batch_norm__115,
            batch_norm__116,
            batch_norm__117,
            batch_norm__118,
            batch_norm__119,
            batch_norm__12,
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
            batch_norm__132,
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
            batch_norm__150,
            batch_norm__151,
            batch_norm__152,
            batch_norm__153,
            batch_norm__154,
            batch_norm__155,
            batch_norm__156,
            batch_norm__157,
            batch_norm__158,
            batch_norm__159,
            batch_norm__16,
            batch_norm__160,
            batch_norm__161,
            batch_norm__17,
            batch_norm__18,
            batch_norm__19,
            batch_norm__2,
            batch_norm__20,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
            batch_norm__3,
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
            batch_norm__36,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__4,
            batch_norm__40,
            batch_norm__41,
            batch_norm__42,
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
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
            batch_norm__6,
            batch_norm__60,
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
            batch_norm__72,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
            batch_norm__78,
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
            batch_norm__90,
            batch_norm__91,
            batch_norm__92,
            batch_norm__93,
            batch_norm__94,
            batch_norm__95,
            batch_norm__96,
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
            conv2d_2,
            conv2d_3,
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
            depthwise_conv2d_2,
            depthwise_conv2d_3,
            depthwise_conv2d_4,
            depthwise_conv2d_5,
            depthwise_conv2d_6,
            depthwise_conv2d_7,
            depthwise_conv2d_8,
            depthwise_conv2d_9,
            full_int_array_0,
            full_int_array_2,
            hardsigmoid_0,
            hardsigmoid_1,
            hardswish_0,
            hardswish_1,
            hardswish_10,
            hardswish_11,
            hardswish_12,
            hardswish_13,
            hardswish_14,
            hardswish_15,
            hardswish_16,
            hardswish_17,
            hardswish_18,
            hardswish_19,
            hardswish_2,
            hardswish_20,
            hardswish_21,
            hardswish_22,
            hardswish_23,
            hardswish_24,
            hardswish_25,
            hardswish_26,
            hardswish_3,
            hardswish_4,
            hardswish_5,
            hardswish_6,
            hardswish_7,
            hardswish_8,
            hardswish_9,
            multiply_0,
            multiply_1,
            pool2d_1,
            pool2d_2,
            relu_0,
            relu_1,
            reshape_0,
            reshape_1,
            reshape_2,
            reshape_3,
        )

        return pool2d_0
