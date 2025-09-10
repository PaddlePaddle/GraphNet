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
        data_0,
    ):
        # pd_op.conv2d: (1x64x128x128xf32) <- (1x3x256x256xf32, 64x3x7x7xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_99, [2, 2], [3, 3], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_0, parameter_99

        # pd_op.batch_norm_: (1x64x128x128xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (1x64x128x128xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_98,
                parameter_97,
                parameter_96,
                parameter_95,
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
        del conv2d_0, parameter_95, parameter_96, parameter_97, parameter_98

        # pd_op.relu: (1x64x128x128xf32) <- (1x64x128x128xf32)
        relu_1 = paddle._C_ops.relu(batch_norm__0)
        del batch_norm__0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [3, 3]

        # pd_op.pool2d: (1x64x64x64xf32) <- (1x64x128x128xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            relu_1,
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
        del full_int_array_0, relu_1

        # pd_op.conv2d: (1x64x64x64xf32) <- (1x64x64x64xf32, 64x64x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            pool2d_0, parameter_94, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_94

        # pd_op.batch_norm_: (1x64x64x64xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (1x64x64x64xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_93,
                parameter_92,
                parameter_91,
                parameter_90,
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
        del conv2d_1, parameter_90, parameter_91, parameter_92, parameter_93

        # pd_op.relu: (1x64x64x64xf32) <- (1x64x64x64xf32)
        relu_2 = paddle._C_ops.relu(batch_norm__6)
        del batch_norm__6

        # pd_op.conv2d: (1x64x64x64xf32) <- (1x64x64x64xf32, 64x64x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            relu_2, parameter_89, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_89, relu_2

        # pd_op.batch_norm_: (1x64x64x64xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (1x64x64x64xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_88,
                parameter_87,
                parameter_86,
                parameter_85,
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
        del conv2d_2, parameter_85, parameter_86, parameter_87, parameter_88

        # pd_op.add: (1x64x64x64xf32) <- (1x64x64x64xf32, 1x64x64x64xf32)
        add_0 = paddle._C_ops.add(batch_norm__12, pool2d_0)
        del batch_norm__12, pool2d_0

        # pd_op.relu: (1x64x64x64xf32) <- (1x64x64x64xf32)
        relu_3 = paddle._C_ops.relu(add_0)
        del add_0

        # pd_op.conv2d: (1x64x64x64xf32) <- (1x64x64x64xf32, 64x64x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            relu_3, parameter_84, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_84

        # pd_op.batch_norm_: (1x64x64x64xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (1x64x64x64xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_83,
                parameter_82,
                parameter_81,
                parameter_80,
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
        del conv2d_3, parameter_80, parameter_81, parameter_82, parameter_83

        # pd_op.relu: (1x64x64x64xf32) <- (1x64x64x64xf32)
        relu_4 = paddle._C_ops.relu(batch_norm__18)
        del batch_norm__18

        # pd_op.conv2d: (1x64x64x64xf32) <- (1x64x64x64xf32, 64x64x3x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            relu_4, parameter_79, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_79, relu_4

        # pd_op.batch_norm_: (1x64x64x64xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (1x64x64x64xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_78,
                parameter_77,
                parameter_76,
                parameter_75,
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
        del conv2d_4, parameter_75, parameter_76, parameter_77, parameter_78

        # pd_op.add: (1x64x64x64xf32) <- (1x64x64x64xf32, 1x64x64x64xf32)
        add_1 = paddle._C_ops.add(batch_norm__24, relu_3)
        del batch_norm__24, relu_3

        # pd_op.relu: (1x64x64x64xf32) <- (1x64x64x64xf32)
        relu_5 = paddle._C_ops.relu(add_1)
        del add_1

        # pd_op.conv2d: (1x128x32x32xf32) <- (1x64x64x64xf32, 128x64x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            relu_5, parameter_74, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_74

        # pd_op.batch_norm_: (1x128x32x32xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (1x128x32x32xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_73,
                parameter_72,
                parameter_71,
                parameter_70,
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
        del conv2d_5, parameter_70, parameter_71, parameter_72, parameter_73

        # pd_op.relu: (1x128x32x32xf32) <- (1x128x32x32xf32)
        relu_6 = paddle._C_ops.relu(batch_norm__30)
        del batch_norm__30

        # pd_op.conv2d: (1x128x32x32xf32) <- (1x128x32x32xf32, 128x128x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            relu_6, parameter_69, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_69, relu_6

        # pd_op.batch_norm_: (1x128x32x32xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (1x128x32x32xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
        del conv2d_6, parameter_65, parameter_66, parameter_67, parameter_68

        # pd_op.conv2d: (1x128x32x32xf32) <- (1x64x64x64xf32, 128x64x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            relu_5, parameter_64, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_64

        # pd_op.batch_norm_: (1x128x32x32xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (1x128x32x32xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_63,
                parameter_62,
                parameter_61,
                parameter_60,
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
        del conv2d_7, parameter_60, parameter_61, parameter_62, parameter_63

        # pd_op.add: (1x128x32x32xf32) <- (1x128x32x32xf32, 1x128x32x32xf32)
        add_2 = paddle._C_ops.add(batch_norm__36, batch_norm__42)
        del batch_norm__36, batch_norm__42

        # pd_op.relu: (1x128x32x32xf32) <- (1x128x32x32xf32)
        relu_7 = paddle._C_ops.relu(add_2)
        del add_2

        # pd_op.conv2d: (1x128x32x32xf32) <- (1x128x32x32xf32, 128x128x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            relu_7, parameter_59, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_59

        # pd_op.batch_norm_: (1x128x32x32xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (1x128x32x32xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_58,
                parameter_57,
                parameter_56,
                parameter_55,
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
        del conv2d_8, parameter_55, parameter_56, parameter_57, parameter_58

        # pd_op.relu: (1x128x32x32xf32) <- (1x128x32x32xf32)
        relu_8 = paddle._C_ops.relu(batch_norm__48)
        del batch_norm__48

        # pd_op.conv2d: (1x128x32x32xf32) <- (1x128x32x32xf32, 128x128x3x3xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            relu_8, parameter_54, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_54, relu_8

        # pd_op.batch_norm_: (1x128x32x32xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (1x128x32x32xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_53,
                parameter_52,
                parameter_51,
                parameter_50,
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
        del conv2d_9, parameter_50, parameter_51, parameter_52, parameter_53

        # pd_op.add: (1x128x32x32xf32) <- (1x128x32x32xf32, 1x128x32x32xf32)
        add_3 = paddle._C_ops.add(batch_norm__54, relu_7)
        del batch_norm__54, relu_7

        # pd_op.relu: (1x128x32x32xf32) <- (1x128x32x32xf32)
        relu_9 = paddle._C_ops.relu(add_3)
        del add_3

        # pd_op.conv2d: (1x256x16x16xf32) <- (1x128x32x32xf32, 256x128x3x3xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            relu_9, parameter_49, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_49

        # pd_op.batch_norm_: (1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_48,
                parameter_47,
                parameter_46,
                parameter_45,
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
        del conv2d_10, parameter_45, parameter_46, parameter_47, parameter_48

        # pd_op.relu: (1x256x16x16xf32) <- (1x256x16x16xf32)
        relu_10 = paddle._C_ops.relu(batch_norm__60)
        del batch_norm__60

        # pd_op.conv2d: (1x256x16x16xf32) <- (1x256x16x16xf32, 256x256x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            relu_10, parameter_44, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_44, relu_10

        # pd_op.batch_norm_: (1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_43,
                parameter_42,
                parameter_41,
                parameter_40,
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
        del conv2d_11, parameter_40, parameter_41, parameter_42, parameter_43

        # pd_op.conv2d: (1x256x16x16xf32) <- (1x128x32x32xf32, 256x128x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            relu_9, parameter_39, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_39

        # pd_op.batch_norm_: (1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_38,
                parameter_37,
                parameter_36,
                parameter_35,
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
        del conv2d_12, parameter_35, parameter_36, parameter_37, parameter_38

        # pd_op.add: (1x256x16x16xf32) <- (1x256x16x16xf32, 1x256x16x16xf32)
        add_4 = paddle._C_ops.add(batch_norm__66, batch_norm__72)
        del batch_norm__66, batch_norm__72

        # pd_op.relu: (1x256x16x16xf32) <- (1x256x16x16xf32)
        relu_11 = paddle._C_ops.relu(add_4)
        del add_4

        # pd_op.conv2d: (1x256x16x16xf32) <- (1x256x16x16xf32, 256x256x3x3xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            relu_11, parameter_34, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_34

        # pd_op.batch_norm_: (1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
        del conv2d_13, parameter_30, parameter_31, parameter_32, parameter_33

        # pd_op.relu: (1x256x16x16xf32) <- (1x256x16x16xf32)
        relu_12 = paddle._C_ops.relu(batch_norm__78)
        del batch_norm__78

        # pd_op.conv2d: (1x256x16x16xf32) <- (1x256x16x16xf32, 256x256x3x3xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            relu_12, parameter_29, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_29, relu_12

        # pd_op.batch_norm_: (1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_28,
                parameter_27,
                parameter_26,
                parameter_25,
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
        del conv2d_14, parameter_25, parameter_26, parameter_27, parameter_28

        # pd_op.add: (1x256x16x16xf32) <- (1x256x16x16xf32, 1x256x16x16xf32)
        add_5 = paddle._C_ops.add(batch_norm__84, relu_11)
        del batch_norm__84, relu_11

        # pd_op.relu: (1x256x16x16xf32) <- (1x256x16x16xf32)
        relu_0 = paddle._C_ops.relu(add_5)
        del add_5

        # pd_op.conv2d: (1x512x8x8xf32) <- (1x256x16x16xf32, 512x256x3x3xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            relu_0, parameter_24, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_24

        # pd_op.batch_norm_: (1x512x8x8xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (1x512x8x8xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_23,
                parameter_22,
                parameter_21,
                parameter_20,
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
        del conv2d_15, parameter_20, parameter_21, parameter_22, parameter_23

        # pd_op.relu: (1x512x8x8xf32) <- (1x512x8x8xf32)
        relu_13 = paddle._C_ops.relu(batch_norm__90)
        del batch_norm__90

        # pd_op.conv2d: (1x512x8x8xf32) <- (1x512x8x8xf32, 512x512x3x3xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            relu_13, parameter_19, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_19, relu_13

        # pd_op.batch_norm_: (1x512x8x8xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (1x512x8x8xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_18,
                parameter_17,
                parameter_16,
                parameter_15,
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
        del conv2d_16, parameter_15, parameter_16, parameter_17, parameter_18

        # pd_op.conv2d: (1x512x8x8xf32) <- (1x256x16x16xf32, 512x256x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            relu_0, parameter_14, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_14, relu_0

        # pd_op.batch_norm_: (1x512x8x8xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (1x512x8x8xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_13,
                parameter_12,
                parameter_11,
                parameter_10,
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
        del conv2d_17, parameter_10, parameter_11, parameter_12, parameter_13

        # pd_op.add: (1x512x8x8xf32) <- (1x512x8x8xf32, 1x512x8x8xf32)
        add_6 = paddle._C_ops.add(batch_norm__96, batch_norm__102)
        del batch_norm__102, batch_norm__96

        # pd_op.relu: (1x512x8x8xf32) <- (1x512x8x8xf32)
        relu_14 = paddle._C_ops.relu(add_6)
        del add_6

        # pd_op.conv2d: (1x512x8x8xf32) <- (1x512x8x8xf32, 512x512x3x3xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            relu_14, parameter_9, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_9

        # pd_op.batch_norm_: (1x512x8x8xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (1x512x8x8xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_8,
                parameter_7,
                parameter_6,
                parameter_5,
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
        del conv2d_18, parameter_5, parameter_6, parameter_7, parameter_8

        # pd_op.relu: (1x512x8x8xf32) <- (1x512x8x8xf32)
        relu_15 = paddle._C_ops.relu(batch_norm__108)
        del batch_norm__108

        # pd_op.conv2d: (1x512x8x8xf32) <- (1x512x8x8xf32, 512x512x3x3xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            relu_15, parameter_4, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_4, relu_15

        # pd_op.batch_norm_: (1x512x8x8xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (1x512x8x8xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_3,
                parameter_2,
                parameter_1,
                parameter_0,
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
        del conv2d_19, parameter_0, parameter_1, parameter_2, parameter_3

        # pd_op.add: (1x512x8x8xf32) <- (1x512x8x8xf32, 1x512x8x8xf32)
        add_7 = paddle._C_ops.add(batch_norm__114, relu_14)
        del batch_norm__114, relu_14

        # pd_op.relu: (1x512x8x8xf32) <- (1x512x8x8xf32)
        relu_16 = paddle._C_ops.relu(add_7)
        del add_7, relu_5, relu_9

        return relu_16
