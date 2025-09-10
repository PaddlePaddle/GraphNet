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
        data_0,
    ):
        # pd_op.conv2d: (-1x40x112x112xf32) <- (-1x3x224x224xf32, 40x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_145, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_0, parameter_145

        # pd_op.batch_norm_: (-1x40x112x112xf32, 40xf32, 40xf32, 40xf32, 40xf32, -1xui8) <- (-1x40x112x112xf32, 40xf32, 40xf32, 40xf32, 40xf32)
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
        del conv2d_0, parameter_141, parameter_142, parameter_143, parameter_144

        # pd_op.hardswish: (-1x40x112x112xf32) <- (-1x40x112x112xf32)
        hardswish_0 = paddle._C_ops.hardswish(batch_norm__0)
        del batch_norm__0

        # pd_op.depthwise_conv2d: (-1x40x112x112xf32) <- (-1x40x112x112xf32, 40x1x3x3xf32)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(
            hardswish_0, parameter_140, [1, 1], [1, 1], "EXPLICIT", 40, [1, 1], "NCHW"
        )
        del hardswish_0, parameter_140

        # pd_op.batch_norm_: (-1x40x112x112xf32, 40xf32, 40xf32, 40xf32, 40xf32, -1xui8) <- (-1x40x112x112xf32, 40xf32, 40xf32, 40xf32, 40xf32)
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
        del (
            depthwise_conv2d_0,
            parameter_136,
            parameter_137,
            parameter_138,
            parameter_139,
        )

        # pd_op.hardswish: (-1x40x112x112xf32) <- (-1x40x112x112xf32)
        hardswish_1 = paddle._C_ops.hardswish(batch_norm__6)
        del batch_norm__6

        # pd_op.conv2d: (-1x80x112x112xf32) <- (-1x40x112x112xf32, 80x40x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            hardswish_1, parameter_135, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del hardswish_1, parameter_135

        # pd_op.batch_norm_: (-1x80x112x112xf32, 80xf32, 80xf32, 80xf32, 80xf32, -1xui8) <- (-1x80x112x112xf32, 80xf32, 80xf32, 80xf32, 80xf32)
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
        del conv2d_1, parameter_131, parameter_132, parameter_133, parameter_134

        # pd_op.hardswish: (-1x80x112x112xf32) <- (-1x80x112x112xf32)
        hardswish_2 = paddle._C_ops.hardswish(batch_norm__12)
        del batch_norm__12

        # pd_op.depthwise_conv2d: (-1x80x56x56xf32) <- (-1x80x112x112xf32, 80x1x3x3xf32)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(
            hardswish_2, parameter_130, [2, 2], [1, 1], "EXPLICIT", 80, [1, 1], "NCHW"
        )
        del hardswish_2, parameter_130

        # pd_op.batch_norm_: (-1x80x56x56xf32, 80xf32, 80xf32, 80xf32, 80xf32, -1xui8) <- (-1x80x56x56xf32, 80xf32, 80xf32, 80xf32, 80xf32)
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
        del (
            depthwise_conv2d_1,
            parameter_126,
            parameter_127,
            parameter_128,
            parameter_129,
        )

        # pd_op.hardswish: (-1x80x56x56xf32) <- (-1x80x56x56xf32)
        hardswish_3 = paddle._C_ops.hardswish(batch_norm__18)
        del batch_norm__18

        # pd_op.conv2d: (-1x160x56x56xf32) <- (-1x80x56x56xf32, 160x80x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            hardswish_3, parameter_125, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del hardswish_3, parameter_125

        # pd_op.batch_norm_: (-1x160x56x56xf32, 160xf32, 160xf32, 160xf32, 160xf32, -1xui8) <- (-1x160x56x56xf32, 160xf32, 160xf32, 160xf32, 160xf32)
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
        del conv2d_2, parameter_121, parameter_122, parameter_123, parameter_124

        # pd_op.hardswish: (-1x160x56x56xf32) <- (-1x160x56x56xf32)
        hardswish_4 = paddle._C_ops.hardswish(batch_norm__24)
        del batch_norm__24

        # pd_op.depthwise_conv2d: (-1x160x56x56xf32) <- (-1x160x56x56xf32, 160x1x3x3xf32)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(
            hardswish_4, parameter_120, [1, 1], [1, 1], "EXPLICIT", 160, [1, 1], "NCHW"
        )
        del hardswish_4, parameter_120

        # pd_op.batch_norm_: (-1x160x56x56xf32, 160xf32, 160xf32, 160xf32, 160xf32, -1xui8) <- (-1x160x56x56xf32, 160xf32, 160xf32, 160xf32, 160xf32)
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
        del (
            depthwise_conv2d_2,
            parameter_116,
            parameter_117,
            parameter_118,
            parameter_119,
        )

        # pd_op.hardswish: (-1x160x56x56xf32) <- (-1x160x56x56xf32)
        hardswish_5 = paddle._C_ops.hardswish(batch_norm__30)
        del batch_norm__30

        # pd_op.conv2d: (-1x160x56x56xf32) <- (-1x160x56x56xf32, 160x160x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            hardswish_5, parameter_115, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del hardswish_5, parameter_115

        # pd_op.batch_norm_: (-1x160x56x56xf32, 160xf32, 160xf32, 160xf32, 160xf32, -1xui8) <- (-1x160x56x56xf32, 160xf32, 160xf32, 160xf32, 160xf32)
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
        del conv2d_3, parameter_111, parameter_112, parameter_113, parameter_114

        # pd_op.hardswish: (-1x160x56x56xf32) <- (-1x160x56x56xf32)
        hardswish_6 = paddle._C_ops.hardswish(batch_norm__36)
        del batch_norm__36

        # pd_op.depthwise_conv2d: (-1x160x28x28xf32) <- (-1x160x56x56xf32, 160x1x3x3xf32)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(
            hardswish_6, parameter_110, [2, 2], [1, 1], "EXPLICIT", 160, [1, 1], "NCHW"
        )
        del hardswish_6, parameter_110

        # pd_op.batch_norm_: (-1x160x28x28xf32, 160xf32, 160xf32, 160xf32, 160xf32, -1xui8) <- (-1x160x28x28xf32, 160xf32, 160xf32, 160xf32, 160xf32)
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
        del (
            depthwise_conv2d_3,
            parameter_106,
            parameter_107,
            parameter_108,
            parameter_109,
        )

        # pd_op.hardswish: (-1x160x28x28xf32) <- (-1x160x28x28xf32)
        hardswish_7 = paddle._C_ops.hardswish(batch_norm__42)
        del batch_norm__42

        # pd_op.conv2d: (-1x320x28x28xf32) <- (-1x160x28x28xf32, 320x160x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            hardswish_7, parameter_105, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del hardswish_7, parameter_105

        # pd_op.batch_norm_: (-1x320x28x28xf32, 320xf32, 320xf32, 320xf32, 320xf32, -1xui8) <- (-1x320x28x28xf32, 320xf32, 320xf32, 320xf32, 320xf32)
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
        del conv2d_4, parameter_101, parameter_102, parameter_103, parameter_104

        # pd_op.hardswish: (-1x320x28x28xf32) <- (-1x320x28x28xf32)
        hardswish_8 = paddle._C_ops.hardswish(batch_norm__48)
        del batch_norm__48

        # pd_op.depthwise_conv2d: (-1x320x28x28xf32) <- (-1x320x28x28xf32, 320x1x3x3xf32)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(
            hardswish_8, parameter_100, [1, 1], [1, 1], "EXPLICIT", 320, [1, 1], "NCHW"
        )
        del hardswish_8, parameter_100

        # pd_op.batch_norm_: (-1x320x28x28xf32, 320xf32, 320xf32, 320xf32, 320xf32, -1xui8) <- (-1x320x28x28xf32, 320xf32, 320xf32, 320xf32, 320xf32)
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
        del depthwise_conv2d_4, parameter_96, parameter_97, parameter_98, parameter_99

        # pd_op.hardswish: (-1x320x28x28xf32) <- (-1x320x28x28xf32)
        hardswish_9 = paddle._C_ops.hardswish(batch_norm__54)
        del batch_norm__54

        # pd_op.conv2d: (-1x320x28x28xf32) <- (-1x320x28x28xf32, 320x320x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            hardswish_9, parameter_95, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del hardswish_9, parameter_95

        # pd_op.batch_norm_: (-1x320x28x28xf32, 320xf32, 320xf32, 320xf32, 320xf32, -1xui8) <- (-1x320x28x28xf32, 320xf32, 320xf32, 320xf32, 320xf32)
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
        del conv2d_5, parameter_91, parameter_92, parameter_93, parameter_94

        # pd_op.hardswish: (-1x320x28x28xf32) <- (-1x320x28x28xf32)
        hardswish_10 = paddle._C_ops.hardswish(batch_norm__60)
        del batch_norm__60

        # pd_op.depthwise_conv2d: (-1x320x14x14xf32) <- (-1x320x28x28xf32, 320x1x3x3xf32)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(
            hardswish_10, parameter_90, [2, 2], [1, 1], "EXPLICIT", 320, [1, 1], "NCHW"
        )
        del hardswish_10, parameter_90

        # pd_op.batch_norm_: (-1x320x14x14xf32, 320xf32, 320xf32, 320xf32, 320xf32, -1xui8) <- (-1x320x14x14xf32, 320xf32, 320xf32, 320xf32, 320xf32)
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
        del depthwise_conv2d_5, parameter_86, parameter_87, parameter_88, parameter_89

        # pd_op.hardswish: (-1x320x14x14xf32) <- (-1x320x14x14xf32)
        hardswish_11 = paddle._C_ops.hardswish(batch_norm__66)
        del batch_norm__66

        # pd_op.conv2d: (-1x640x14x14xf32) <- (-1x320x14x14xf32, 640x320x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            hardswish_11, parameter_85, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del hardswish_11, parameter_85

        # pd_op.batch_norm_: (-1x640x14x14xf32, 640xf32, 640xf32, 640xf32, 640xf32, -1xui8) <- (-1x640x14x14xf32, 640xf32, 640xf32, 640xf32, 640xf32)
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
        del conv2d_6, parameter_81, parameter_82, parameter_83, parameter_84

        # pd_op.hardswish: (-1x640x14x14xf32) <- (-1x640x14x14xf32)
        hardswish_12 = paddle._C_ops.hardswish(batch_norm__72)
        del batch_norm__72

        # pd_op.depthwise_conv2d: (-1x640x14x14xf32) <- (-1x640x14x14xf32, 640x1x5x5xf32)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(
            hardswish_12, parameter_80, [1, 1], [2, 2], "EXPLICIT", 640, [1, 1], "NCHW"
        )
        del hardswish_12, parameter_80

        # pd_op.batch_norm_: (-1x640x14x14xf32, 640xf32, 640xf32, 640xf32, 640xf32, -1xui8) <- (-1x640x14x14xf32, 640xf32, 640xf32, 640xf32, 640xf32)
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
        del depthwise_conv2d_6, parameter_76, parameter_77, parameter_78, parameter_79

        # pd_op.hardswish: (-1x640x14x14xf32) <- (-1x640x14x14xf32)
        hardswish_13 = paddle._C_ops.hardswish(batch_norm__78)
        del batch_norm__78

        # pd_op.conv2d: (-1x640x14x14xf32) <- (-1x640x14x14xf32, 640x640x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            hardswish_13, parameter_75, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del hardswish_13, parameter_75

        # pd_op.batch_norm_: (-1x640x14x14xf32, 640xf32, 640xf32, 640xf32, 640xf32, -1xui8) <- (-1x640x14x14xf32, 640xf32, 640xf32, 640xf32, 640xf32)
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
        del conv2d_7, parameter_71, parameter_72, parameter_73, parameter_74

        # pd_op.hardswish: (-1x640x14x14xf32) <- (-1x640x14x14xf32)
        hardswish_14 = paddle._C_ops.hardswish(batch_norm__84)
        del batch_norm__84

        # pd_op.depthwise_conv2d: (-1x640x14x14xf32) <- (-1x640x14x14xf32, 640x1x5x5xf32)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(
            hardswish_14, parameter_70, [1, 1], [2, 2], "EXPLICIT", 640, [1, 1], "NCHW"
        )
        del hardswish_14, parameter_70

        # pd_op.batch_norm_: (-1x640x14x14xf32, 640xf32, 640xf32, 640xf32, 640xf32, -1xui8) <- (-1x640x14x14xf32, 640xf32, 640xf32, 640xf32, 640xf32)
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
        del depthwise_conv2d_7, parameter_66, parameter_67, parameter_68, parameter_69

        # pd_op.hardswish: (-1x640x14x14xf32) <- (-1x640x14x14xf32)
        hardswish_15 = paddle._C_ops.hardswish(batch_norm__90)
        del batch_norm__90

        # pd_op.conv2d: (-1x640x14x14xf32) <- (-1x640x14x14xf32, 640x640x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            hardswish_15, parameter_65, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del hardswish_15, parameter_65

        # pd_op.batch_norm_: (-1x640x14x14xf32, 640xf32, 640xf32, 640xf32, 640xf32, -1xui8) <- (-1x640x14x14xf32, 640xf32, 640xf32, 640xf32, 640xf32)
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
        del conv2d_8, parameter_61, parameter_62, parameter_63, parameter_64

        # pd_op.hardswish: (-1x640x14x14xf32) <- (-1x640x14x14xf32)
        hardswish_16 = paddle._C_ops.hardswish(batch_norm__96)
        del batch_norm__96

        # pd_op.depthwise_conv2d: (-1x640x14x14xf32) <- (-1x640x14x14xf32, 640x1x5x5xf32)
        depthwise_conv2d_8 = paddle._C_ops.depthwise_conv2d(
            hardswish_16, parameter_60, [1, 1], [2, 2], "EXPLICIT", 640, [1, 1], "NCHW"
        )
        del hardswish_16, parameter_60

        # pd_op.batch_norm_: (-1x640x14x14xf32, 640xf32, 640xf32, 640xf32, 640xf32, -1xui8) <- (-1x640x14x14xf32, 640xf32, 640xf32, 640xf32, 640xf32)
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
        del depthwise_conv2d_8, parameter_56, parameter_57, parameter_58, parameter_59

        # pd_op.hardswish: (-1x640x14x14xf32) <- (-1x640x14x14xf32)
        hardswish_17 = paddle._C_ops.hardswish(batch_norm__102)
        del batch_norm__102

        # pd_op.conv2d: (-1x640x14x14xf32) <- (-1x640x14x14xf32, 640x640x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            hardswish_17, parameter_55, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del hardswish_17, parameter_55

        # pd_op.batch_norm_: (-1x640x14x14xf32, 640xf32, 640xf32, 640xf32, 640xf32, -1xui8) <- (-1x640x14x14xf32, 640xf32, 640xf32, 640xf32, 640xf32)
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
        del conv2d_9, parameter_51, parameter_52, parameter_53, parameter_54

        # pd_op.hardswish: (-1x640x14x14xf32) <- (-1x640x14x14xf32)
        hardswish_18 = paddle._C_ops.hardswish(batch_norm__108)
        del batch_norm__108

        # pd_op.depthwise_conv2d: (-1x640x14x14xf32) <- (-1x640x14x14xf32, 640x1x5x5xf32)
        depthwise_conv2d_9 = paddle._C_ops.depthwise_conv2d(
            hardswish_18, parameter_50, [1, 1], [2, 2], "EXPLICIT", 640, [1, 1], "NCHW"
        )
        del hardswish_18, parameter_50

        # pd_op.batch_norm_: (-1x640x14x14xf32, 640xf32, 640xf32, 640xf32, 640xf32, -1xui8) <- (-1x640x14x14xf32, 640xf32, 640xf32, 640xf32, 640xf32)
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
        del depthwise_conv2d_9, parameter_46, parameter_47, parameter_48, parameter_49

        # pd_op.hardswish: (-1x640x14x14xf32) <- (-1x640x14x14xf32)
        hardswish_19 = paddle._C_ops.hardswish(batch_norm__114)
        del batch_norm__114

        # pd_op.conv2d: (-1x640x14x14xf32) <- (-1x640x14x14xf32, 640x640x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            hardswish_19, parameter_45, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del hardswish_19, parameter_45

        # pd_op.batch_norm_: (-1x640x14x14xf32, 640xf32, 640xf32, 640xf32, 640xf32, -1xui8) <- (-1x640x14x14xf32, 640xf32, 640xf32, 640xf32, 640xf32)
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
        del conv2d_10, parameter_41, parameter_42, parameter_43, parameter_44

        # pd_op.hardswish: (-1x640x14x14xf32) <- (-1x640x14x14xf32)
        hardswish_20 = paddle._C_ops.hardswish(batch_norm__120)
        del batch_norm__120

        # pd_op.depthwise_conv2d: (-1x640x14x14xf32) <- (-1x640x14x14xf32, 640x1x5x5xf32)
        depthwise_conv2d_10 = paddle._C_ops.depthwise_conv2d(
            hardswish_20, parameter_40, [1, 1], [2, 2], "EXPLICIT", 640, [1, 1], "NCHW"
        )
        del hardswish_20, parameter_40

        # pd_op.batch_norm_: (-1x640x14x14xf32, 640xf32, 640xf32, 640xf32, 640xf32, -1xui8) <- (-1x640x14x14xf32, 640xf32, 640xf32, 640xf32, 640xf32)
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
        del depthwise_conv2d_10, parameter_36, parameter_37, parameter_38, parameter_39

        # pd_op.hardswish: (-1x640x14x14xf32) <- (-1x640x14x14xf32)
        hardswish_21 = paddle._C_ops.hardswish(batch_norm__126)
        del batch_norm__126

        # pd_op.conv2d: (-1x640x14x14xf32) <- (-1x640x14x14xf32, 640x640x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            hardswish_21, parameter_35, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del hardswish_21, parameter_35

        # pd_op.batch_norm_: (-1x640x14x14xf32, 640xf32, 640xf32, 640xf32, 640xf32, -1xui8) <- (-1x640x14x14xf32, 640xf32, 640xf32, 640xf32, 640xf32)
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
        del conv2d_11, parameter_31, parameter_32, parameter_33, parameter_34

        # pd_op.hardswish: (-1x640x14x14xf32) <- (-1x640x14x14xf32)
        hardswish_22 = paddle._C_ops.hardswish(batch_norm__132)
        del batch_norm__132

        # pd_op.depthwise_conv2d: (-1x640x7x7xf32) <- (-1x640x14x14xf32, 640x1x5x5xf32)
        depthwise_conv2d_11 = paddle._C_ops.depthwise_conv2d(
            hardswish_22, parameter_30, [2, 2], [2, 2], "EXPLICIT", 640, [1, 1], "NCHW"
        )
        del hardswish_22, parameter_30

        # pd_op.batch_norm_: (-1x640x7x7xf32, 640xf32, 640xf32, 640xf32, 640xf32, -1xui8) <- (-1x640x7x7xf32, 640xf32, 640xf32, 640xf32, 640xf32)
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
        del depthwise_conv2d_11, parameter_26, parameter_27, parameter_28, parameter_29

        # pd_op.hardswish: (-1x640x7x7xf32) <- (-1x640x7x7xf32)
        hardswish_23 = paddle._C_ops.hardswish(batch_norm__138)
        del batch_norm__138

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.pool2d: (-1x640x1x1xf32) <- (-1x640x7x7xf32, 2xi64)
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
        del full_int_array_0

        # pd_op.conv2d: (-1x160x1x1xf32) <- (-1x640x1x1xf32, 160x640x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            pool2d_0, parameter_25, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_25, pool2d_0

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [1, -1, 1, 1]

        # pd_op.reshape: (1x160x1x1xf32) <- (160xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_24, full_int_array_1)
        del full_int_array_1, parameter_24

        # pd_op.add: (-1x160x1x1xf32) <- (-1x160x1x1xf32, 1x160x1x1xf32)
        add_1 = paddle._C_ops.add(conv2d_12, reshape_0)
        del conv2d_12, reshape_0

        # pd_op.relu: (-1x160x1x1xf32) <- (-1x160x1x1xf32)
        relu_0 = paddle._C_ops.relu(add_1)
        del add_1

        # pd_op.conv2d: (-1x640x1x1xf32) <- (-1x160x1x1xf32, 640x160x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            relu_0, parameter_23, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_23, relu_0

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_2 = [1, -1, 1, 1]

        # pd_op.reshape: (1x640x1x1xf32) <- (640xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_22, full_int_array_2)
        del full_int_array_2, parameter_22

        # pd_op.add: (-1x640x1x1xf32) <- (-1x640x1x1xf32, 1x640x1x1xf32)
        add_2 = paddle._C_ops.add(conv2d_13, reshape_1)
        del conv2d_13, reshape_1

        # pd_op.hardsigmoid: (-1x640x1x1xf32) <- (-1x640x1x1xf32)
        hardsigmoid_0 = paddle._C_ops.hardsigmoid(
            add_2, float("0.166667"), float("0.5")
        )
        del add_2

        # pd_op.multiply: (-1x640x7x7xf32) <- (-1x640x7x7xf32, -1x640x1x1xf32)
        multiply_0 = paddle._C_ops.multiply(hardswish_23, hardsigmoid_0)
        del hardsigmoid_0, hardswish_23

        # pd_op.conv2d: (-1x1280x7x7xf32) <- (-1x640x7x7xf32, 1280x640x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            multiply_0, parameter_21, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_0, parameter_21

        # pd_op.batch_norm_: (-1x1280x7x7xf32, 1280xf32, 1280xf32, 1280xf32, 1280xf32, -1xui8) <- (-1x1280x7x7xf32, 1280xf32, 1280xf32, 1280xf32, 1280xf32)
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
                parameter_20,
                parameter_19,
                parameter_18,
                parameter_17,
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
        del conv2d_14, parameter_17, parameter_18, parameter_19, parameter_20

        # pd_op.hardswish: (-1x1280x7x7xf32) <- (-1x1280x7x7xf32)
        hardswish_24 = paddle._C_ops.hardswish(batch_norm__144)
        del batch_norm__144

        # pd_op.depthwise_conv2d: (-1x1280x7x7xf32) <- (-1x1280x7x7xf32, 1280x1x5x5xf32)
        depthwise_conv2d_12 = paddle._C_ops.depthwise_conv2d(
            hardswish_24, parameter_16, [1, 1], [2, 2], "EXPLICIT", 1280, [1, 1], "NCHW"
        )
        del hardswish_24, parameter_16

        # pd_op.batch_norm_: (-1x1280x7x7xf32, 1280xf32, 1280xf32, 1280xf32, 1280xf32, -1xui8) <- (-1x1280x7x7xf32, 1280xf32, 1280xf32, 1280xf32, 1280xf32)
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
        del depthwise_conv2d_12, parameter_12, parameter_13, parameter_14, parameter_15

        # pd_op.hardswish: (-1x1280x7x7xf32) <- (-1x1280x7x7xf32)
        hardswish_25 = paddle._C_ops.hardswish(batch_norm__150)
        del batch_norm__150

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [1, 1]

        # pd_op.pool2d: (-1x1280x1x1xf32) <- (-1x1280x7x7xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(
            hardswish_25,
            full_int_array_3,
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
        del full_int_array_3

        # pd_op.conv2d: (-1x320x1x1xf32) <- (-1x1280x1x1xf32, 320x1280x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            pool2d_1, parameter_11, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_11, pool2d_1

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_4 = [1, -1, 1, 1]

        # pd_op.reshape: (1x320x1x1xf32) <- (320xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(parameter_10, full_int_array_4)
        del full_int_array_4, parameter_10

        # pd_op.add: (-1x320x1x1xf32) <- (-1x320x1x1xf32, 1x320x1x1xf32)
        add_3 = paddle._C_ops.add(conv2d_15, reshape_2)
        del conv2d_15, reshape_2

        # pd_op.relu: (-1x320x1x1xf32) <- (-1x320x1x1xf32)
        relu_1 = paddle._C_ops.relu(add_3)
        del add_3

        # pd_op.conv2d: (-1x1280x1x1xf32) <- (-1x320x1x1xf32, 1280x320x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            relu_1, parameter_9, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_9, relu_1

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_5 = [1, -1, 1, 1]

        # pd_op.reshape: (1x1280x1x1xf32) <- (1280xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(parameter_8, full_int_array_5)
        del full_int_array_5, parameter_8

        # pd_op.add: (-1x1280x1x1xf32) <- (-1x1280x1x1xf32, 1x1280x1x1xf32)
        add_4 = paddle._C_ops.add(conv2d_16, reshape_3)
        del conv2d_16, reshape_3

        # pd_op.hardsigmoid: (-1x1280x1x1xf32) <- (-1x1280x1x1xf32)
        hardsigmoid_1 = paddle._C_ops.hardsigmoid(
            add_4, float("0.166667"), float("0.5")
        )
        del add_4

        # pd_op.multiply: (-1x1280x7x7xf32) <- (-1x1280x7x7xf32, -1x1280x1x1xf32)
        multiply_1 = paddle._C_ops.multiply(hardswish_25, hardsigmoid_1)
        del hardsigmoid_1, hardswish_25

        # pd_op.conv2d: (-1x1280x7x7xf32) <- (-1x1280x7x7xf32, 1280x1280x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            multiply_1, parameter_7, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_1, parameter_7

        # pd_op.batch_norm_: (-1x1280x7x7xf32, 1280xf32, 1280xf32, 1280xf32, 1280xf32, -1xui8) <- (-1x1280x7x7xf32, 1280xf32, 1280xf32, 1280xf32, 1280xf32)
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
                parameter_6,
                parameter_5,
                parameter_4,
                parameter_3,
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
        del conv2d_17, parameter_3, parameter_4, parameter_5, parameter_6

        # pd_op.hardswish: (-1x1280x7x7xf32) <- (-1x1280x7x7xf32)
        hardswish_26 = paddle._C_ops.hardswish(batch_norm__156)
        del batch_norm__156

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_6 = [1, 1]

        # pd_op.pool2d: (-1x1280x1x1xf32) <- (-1x1280x7x7xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(
            hardswish_26,
            full_int_array_6,
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
        del full_int_array_6, hardswish_26

        # pd_op.conv2d: (-1x1280x1x1xf32) <- (-1x1280x1x1xf32, 1280x1280x1x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            pool2d_2, parameter_2, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_2, pool2d_2

        # pd_op.hardswish: (-1x1280x1x1xf32) <- (-1x1280x1x1xf32)
        hardswish_27 = paddle._C_ops.hardswish(conv2d_18)
        del conv2d_18

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (-1x1280x1x1xf32, -1x1280x1x1xui8) <- (-1x1280x1x1xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                hardswish_27, None, full_0, True, "downgrade_in_infer", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del full_0, hardswish_27

        # pd_op.flatten: (-1x1280xf32) <- (-1x1280x1x1xf32)
        flatten_0 = paddle._C_ops.flatten(dropout_0, 1, 3)
        del dropout_0

        # pd_op.matmul: (-1x102xf32) <- (-1x1280xf32, 1280x102xf32)
        matmul_0 = paddle._C_ops.matmul(flatten_0, parameter_1, False, False)
        del flatten_0, parameter_1

        # pd_op.add: (-1x102xf32) <- (-1x102xf32, 102xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_0)
        del matmul_0, parameter_0

        return add_0
