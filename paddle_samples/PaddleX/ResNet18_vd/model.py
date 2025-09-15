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
        data_0,
    ):
        # pd_op.conv2d: (-1x32x112x112xf32) <- (-1x3x224x224xf32, 32x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_116, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_0, parameter_116

        # pd_op.batch_norm_: (-1x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32)
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
                parameter_115,
                parameter_114,
                parameter_113,
                parameter_112,
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
        del conv2d_0, parameter_112, parameter_113, parameter_114, parameter_115

        # pd_op.relu: (-1x32x112x112xf32) <- (-1x32x112x112xf32)
        relu_0 = paddle._C_ops.relu(batch_norm__0)
        del batch_norm__0

        # pd_op.conv2d: (-1x32x112x112xf32) <- (-1x32x112x112xf32, 32x32x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            relu_0, parameter_111, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_111, relu_0

        # pd_op.batch_norm_: (-1x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (-1x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32)
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
                parameter_110,
                parameter_109,
                parameter_108,
                parameter_107,
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
        del conv2d_1, parameter_107, parameter_108, parameter_109, parameter_110

        # pd_op.relu: (-1x32x112x112xf32) <- (-1x32x112x112xf32)
        relu_1 = paddle._C_ops.relu(batch_norm__6)
        del batch_norm__6

        # pd_op.conv2d: (-1x64x112x112xf32) <- (-1x32x112x112xf32, 64x32x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            relu_1, parameter_106, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_106, relu_1

        # pd_op.batch_norm_: (-1x64x112x112xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x112x112xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_105,
                parameter_104,
                parameter_103,
                parameter_102,
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
        del conv2d_2, parameter_102, parameter_103, parameter_104, parameter_105

        # pd_op.relu: (-1x64x112x112xf32) <- (-1x64x112x112xf32)
        relu_2 = paddle._C_ops.relu(batch_norm__12)
        del batch_norm__12

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [3, 3]

        # pd_op.pool2d: (-1x64x56x56xf32) <- (-1x64x112x112xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            relu_2,
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
        del full_int_array_0, relu_2

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            pool2d_0, parameter_101, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_101

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_100,
                parameter_99,
                parameter_98,
                parameter_97,
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
        del conv2d_3, parameter_100, parameter_97, parameter_98, parameter_99

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        relu_3 = paddle._C_ops.relu(batch_norm__18)
        del batch_norm__18

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            relu_3, parameter_96, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_96, relu_3

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_95,
                parameter_94,
                parameter_93,
                parameter_92,
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
        del conv2d_4, parameter_92, parameter_93, parameter_94, parameter_95

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            pool2d_0, parameter_91, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_91, pool2d_0

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_90,
                parameter_89,
                parameter_88,
                parameter_87,
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
        del conv2d_5, parameter_87, parameter_88, parameter_89, parameter_90

        # pd_op.add: (-1x64x56x56xf32) <- (-1x64x56x56xf32, -1x64x56x56xf32)
        add_1 = paddle._C_ops.add(batch_norm__24, batch_norm__30)
        del batch_norm__24, batch_norm__30

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        relu_4 = paddle._C_ops.relu(add_1)
        del add_1

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            relu_4, parameter_86, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_86

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_85,
                parameter_84,
                parameter_83,
                parameter_82,
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
        del conv2d_6, parameter_82, parameter_83, parameter_84, parameter_85

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        relu_5 = paddle._C_ops.relu(batch_norm__36)
        del batch_norm__36

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x3x3xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            relu_5, parameter_81, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_81, relu_5

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_80,
                parameter_79,
                parameter_78,
                parameter_77,
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
        del conv2d_7, parameter_77, parameter_78, parameter_79, parameter_80

        # pd_op.add: (-1x64x56x56xf32) <- (-1x64x56x56xf32, -1x64x56x56xf32)
        add_2 = paddle._C_ops.add(batch_norm__42, relu_4)
        del batch_norm__42, relu_4

        # pd_op.relu: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        relu_6 = paddle._C_ops.relu(add_2)
        del add_2

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x64x56x56xf32, 128x64x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            relu_6, parameter_76, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_76

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_75,
                parameter_74,
                parameter_73,
                parameter_72,
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
        del conv2d_8, parameter_72, parameter_73, parameter_74, parameter_75

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        relu_7 = paddle._C_ops.relu(batch_norm__48)
        del batch_norm__48

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            relu_7, parameter_71, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_71, relu_7

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
        del conv2d_9, parameter_67, parameter_68, parameter_69, parameter_70

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [2, 2]

        # pd_op.pool2d: (-1x64x28x28xf32) <- (-1x64x56x56xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(
            relu_6,
            full_int_array_1,
            [2, 2],
            [0, 0],
            True,
            True,
            "NCHW",
            "avg",
            False,
            False,
            "SAME",
        )
        del full_int_array_1, relu_6

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x64x28x28xf32, 128x64x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            pool2d_1, parameter_66, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_66, pool2d_1

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
        del conv2d_10, parameter_62, parameter_63, parameter_64, parameter_65

        # pd_op.add: (-1x128x28x28xf32) <- (-1x128x28x28xf32, -1x128x28x28xf32)
        add_3 = paddle._C_ops.add(batch_norm__54, batch_norm__60)
        del batch_norm__54, batch_norm__60

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        relu_8 = paddle._C_ops.relu(add_3)
        del add_3

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            relu_8, parameter_61, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_61

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_60,
                parameter_59,
                parameter_58,
                parameter_57,
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
        del conv2d_11, parameter_57, parameter_58, parameter_59, parameter_60

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        relu_9 = paddle._C_ops.relu(batch_norm__66)
        del batch_norm__66

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            relu_9, parameter_56, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_56, relu_9

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_55,
                parameter_54,
                parameter_53,
                parameter_52,
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
        del conv2d_12, parameter_52, parameter_53, parameter_54, parameter_55

        # pd_op.add: (-1x128x28x28xf32) <- (-1x128x28x28xf32, -1x128x28x28xf32)
        add_4 = paddle._C_ops.add(batch_norm__72, relu_8)
        del batch_norm__72, relu_8

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        relu_10 = paddle._C_ops.relu(add_4)
        del add_4

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x128x28x28xf32, 256x128x3x3xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            relu_10, parameter_51, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_51

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_50,
                parameter_49,
                parameter_48,
                parameter_47,
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
        del conv2d_13, parameter_47, parameter_48, parameter_49, parameter_50

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu_11 = paddle._C_ops.relu(batch_norm__78)
        del batch_norm__78

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            relu_11, parameter_46, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_46, relu_11

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_45,
                parameter_44,
                parameter_43,
                parameter_42,
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
        del conv2d_14, parameter_42, parameter_43, parameter_44, parameter_45

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [2, 2]

        # pd_op.pool2d: (-1x128x14x14xf32) <- (-1x128x28x28xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(
            relu_10,
            full_int_array_2,
            [2, 2],
            [0, 0],
            True,
            True,
            "NCHW",
            "avg",
            False,
            False,
            "SAME",
        )
        del full_int_array_2, relu_10

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x128x14x14xf32, 256x128x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            pool2d_2, parameter_41, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_41, pool2d_2

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_40,
                parameter_39,
                parameter_38,
                parameter_37,
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
        del conv2d_15, parameter_37, parameter_38, parameter_39, parameter_40

        # pd_op.add: (-1x256x14x14xf32) <- (-1x256x14x14xf32, -1x256x14x14xf32)
        add_5 = paddle._C_ops.add(batch_norm__84, batch_norm__90)
        del batch_norm__84, batch_norm__90

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu_12 = paddle._C_ops.relu(add_5)
        del add_5

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            relu_12, parameter_36, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_36

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_35,
                parameter_34,
                parameter_33,
                parameter_32,
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
        del conv2d_16, parameter_32, parameter_33, parameter_34, parameter_35

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu_13 = paddle._C_ops.relu(batch_norm__96)
        del batch_norm__96

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x256x14x14xf32, 256x256x3x3xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            relu_13, parameter_31, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_31, relu_13

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_30,
                parameter_29,
                parameter_28,
                parameter_27,
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
        del conv2d_17, parameter_27, parameter_28, parameter_29, parameter_30

        # pd_op.add: (-1x256x14x14xf32) <- (-1x256x14x14xf32, -1x256x14x14xf32)
        add_6 = paddle._C_ops.add(batch_norm__102, relu_12)
        del batch_norm__102, relu_12

        # pd_op.relu: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu_14 = paddle._C_ops.relu(add_6)
        del add_6

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x256x14x14xf32, 512x256x3x3xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            relu_14, parameter_26, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_26

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_25,
                parameter_24,
                parameter_23,
                parameter_22,
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
        del conv2d_18, parameter_22, parameter_23, parameter_24, parameter_25

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        relu_15 = paddle._C_ops.relu(batch_norm__108)
        del batch_norm__108

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x512x3x3xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            relu_15, parameter_21, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_21, relu_15

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_20,
                parameter_19,
                parameter_18,
                parameter_17,
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
        del conv2d_19, parameter_17, parameter_18, parameter_19, parameter_20

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [2, 2]

        # pd_op.pool2d: (-1x256x7x7xf32) <- (-1x256x14x14xf32, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(
            relu_14,
            full_int_array_3,
            [2, 2],
            [0, 0],
            True,
            True,
            "NCHW",
            "avg",
            False,
            False,
            "SAME",
        )
        del full_int_array_3, relu_14

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x256x7x7xf32, 512x256x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            pool2d_3, parameter_16, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_16, pool2d_3

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_15,
                parameter_14,
                parameter_13,
                parameter_12,
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
        del conv2d_20, parameter_12, parameter_13, parameter_14, parameter_15

        # pd_op.add: (-1x512x7x7xf32) <- (-1x512x7x7xf32, -1x512x7x7xf32)
        add_7 = paddle._C_ops.add(batch_norm__114, batch_norm__120)
        del batch_norm__114, batch_norm__120

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        relu_16 = paddle._C_ops.relu(add_7)
        del add_7

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x512x3x3xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            relu_16, parameter_11, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_11

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_10,
                parameter_9,
                parameter_8,
                parameter_7,
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
        del conv2d_21, parameter_10, parameter_7, parameter_8, parameter_9

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        relu_17 = paddle._C_ops.relu(batch_norm__126)
        del batch_norm__126

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x512x7x7xf32, 512x512x3x3xf32)
        conv2d_22 = paddle._C_ops.conv2d(
            relu_17, parameter_6, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_6, relu_17

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_5,
                parameter_4,
                parameter_3,
                parameter_2,
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
        del conv2d_22, parameter_2, parameter_3, parameter_4, parameter_5

        # pd_op.add: (-1x512x7x7xf32) <- (-1x512x7x7xf32, -1x512x7x7xf32)
        add_8 = paddle._C_ops.add(batch_norm__132, relu_16)
        del batch_norm__132, relu_16

        # pd_op.relu: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        relu_18 = paddle._C_ops.relu(add_8)
        del add_8

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [1, 1]

        # pd_op.pool2d: (-1x512x1x1xf32) <- (-1x512x7x7xf32, 2xi64)
        pool2d_4 = paddle._C_ops.pool2d(
            relu_18,
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
        del full_int_array_4, relu_18

        # pd_op.flatten: (-1x512xf32) <- (-1x512x1x1xf32)
        flatten_0 = paddle._C_ops.flatten(pool2d_4, 1, 3)
        del pool2d_4

        # pd_op.matmul: (-1x102xf32) <- (-1x512xf32, 512x102xf32)
        matmul_0 = paddle._C_ops.matmul(flatten_0, parameter_1, False, False)
        del flatten_0, parameter_1

        # pd_op.add: (-1x102xf32) <- (-1x102xf32, 102xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_0)
        del matmul_0, parameter_0

        return add_0
