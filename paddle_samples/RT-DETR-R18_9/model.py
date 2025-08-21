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
        data_0,
    ):
        # pd_op.conv2d: (4x32x-1x-1xf32) <- (4x3x-1x-1xf32, 32x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_114, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x32x-1x-1xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (4x32x-1x-1xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        (
            batch_norm__127,
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_0,
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

        # pd_op.relu: (4x32x-1x-1xf32) <- (4x32x-1x-1xf32)
        relu_0 = paddle._C_ops.relu(batch_norm__127)

        # pd_op.conv2d: (4x32x-1x-1xf32) <- (4x32x-1x-1xf32, 32x32x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            relu_0, parameter_109, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x32x-1x-1xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (4x32x-1x-1xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        (
            batch_norm__128,
            batch_norm__5,
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_1,
                parameter_108,
                parameter_107,
                parameter_106,
                parameter_105,
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

        # pd_op.relu: (4x32x-1x-1xf32) <- (4x32x-1x-1xf32)
        relu_1 = paddle._C_ops.relu(batch_norm__128)

        # pd_op.conv2d: (4x64x-1x-1xf32) <- (4x32x-1x-1xf32, 64x32x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            relu_1, parameter_104, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__129,
            batch_norm__10,
            batch_norm__11,
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_2,
                parameter_103,
                parameter_102,
                parameter_101,
                parameter_100,
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

        # pd_op.relu: (4x64x-1x-1xf32) <- (4x64x-1x-1xf32)
        relu_2 = paddle._C_ops.relu(batch_norm__129)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [3, 3]

        # pd_op.pool2d: (4x64x-1x-1xf32) <- (4x64x-1x-1xf32, 2xi64)
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

        # pd_op.conv2d: (4x64x-1x-1xf32) <- (4x64x-1x-1xf32, 64x64x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            pool2d_0, parameter_99, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__130,
            batch_norm__15,
            batch_norm__16,
            batch_norm__17,
            batch_norm__18,
            batch_norm__19,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_3,
                parameter_98,
                parameter_97,
                parameter_96,
                parameter_95,
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

        # pd_op.relu: (4x64x-1x-1xf32) <- (4x64x-1x-1xf32)
        relu_3 = paddle._C_ops.relu(batch_norm__130)

        # pd_op.conv2d: (4x64x-1x-1xf32) <- (4x64x-1x-1xf32, 64x64x3x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            relu_3, parameter_94, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__20,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
            batch_norm__24,
            batch_norm__25,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_4,
                parameter_93,
                parameter_92,
                parameter_91,
                parameter_90,
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

        # pd_op.conv2d: (4x64x-1x-1xf32) <- (4x64x-1x-1xf32, 64x64x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            pool2d_0, parameter_89, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__26,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
            batch_norm__30,
            batch_norm__31,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_5,
                parameter_88,
                parameter_87,
                parameter_86,
                parameter_85,
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

        # pd_op.add: (4x64x-1x-1xf32) <- (4x64x-1x-1xf32, 4x64x-1x-1xf32)
        add_0 = paddle._C_ops.add(batch_norm__20, batch_norm__26)

        # pd_op.relu: (4x64x-1x-1xf32) <- (4x64x-1x-1xf32)
        relu_4 = paddle._C_ops.relu(add_0)

        # pd_op.conv2d: (4x64x-1x-1xf32) <- (4x64x-1x-1xf32, 64x64x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            relu_4, parameter_84, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__131,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
            batch_norm__36,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_6,
                parameter_83,
                parameter_82,
                parameter_81,
                parameter_80,
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

        # pd_op.relu: (4x64x-1x-1xf32) <- (4x64x-1x-1xf32)
        relu_5 = paddle._C_ops.relu(batch_norm__131)

        # pd_op.conv2d: (4x64x-1x-1xf32) <- (4x64x-1x-1xf32, 64x64x3x3xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            relu_5, parameter_79, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
            batch_norm__42,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_7,
                parameter_78,
                parameter_77,
                parameter_76,
                parameter_75,
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

        # pd_op.add: (4x64x-1x-1xf32) <- (4x64x-1x-1xf32, 4x64x-1x-1xf32)
        add_1 = paddle._C_ops.add(batch_norm__37, relu_4)

        # pd_op.relu: (4x64x-1x-1xf32) <- (4x64x-1x-1xf32)
        relu_6 = paddle._C_ops.relu(add_1)

        # pd_op.conv2d: (4x128x-1x-1xf32) <- (4x64x-1x-1xf32, 128x64x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            relu_6, parameter_74, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (4x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__132,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_8,
                parameter_73,
                parameter_72,
                parameter_71,
                parameter_70,
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

        # pd_op.relu: (4x128x-1x-1xf32) <- (4x128x-1x-1xf32)
        relu_7 = paddle._C_ops.relu(batch_norm__132)

        # pd_op.conv2d: (4x128x-1x-1xf32) <- (4x128x-1x-1xf32, 128x128x3x3xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            relu_7, parameter_69, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (4x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_68,
                parameter_67,
                parameter_66,
                parameter_65,
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

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [2, 2]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_1 = full_int_array_1

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_0 = full_int_array_1

        # pd_op.pool2d: (4x64x-1x-1xf32) <- (4x64x-1x-1xf32, 2xi64)
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
            "EXPLICIT",
        )

        # pd_op.conv2d: (4x128x-1x-1xf32) <- (4x64x-1x-1xf32, 128x64x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            pool2d_1, parameter_64, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (4x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_10,
                parameter_63,
                parameter_62,
                parameter_61,
                parameter_60,
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

        # pd_op.add: (4x128x-1x-1xf32) <- (4x128x-1x-1xf32, 4x128x-1x-1xf32)
        add_2 = paddle._C_ops.add(batch_norm__48, batch_norm__54)

        # pd_op.relu: (4x128x-1x-1xf32) <- (4x128x-1x-1xf32)
        relu_8 = paddle._C_ops.relu(add_2)

        # pd_op.conv2d: (4x128x-1x-1xf32) <- (4x128x-1x-1xf32, 128x128x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            relu_8, parameter_59, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (4x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__133,
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            batch_norm__64,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_11,
                parameter_58,
                parameter_57,
                parameter_56,
                parameter_55,
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

        # pd_op.relu: (4x128x-1x-1xf32) <- (4x128x-1x-1xf32)
        relu_9 = paddle._C_ops.relu(batch_norm__133)

        # pd_op.conv2d: (4x128x-1x-1xf32) <- (4x128x-1x-1xf32, 128x128x3x3xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            relu_9, parameter_54, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (4x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__65,
            batch_norm__66,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            batch_norm__70,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_12,
                parameter_53,
                parameter_52,
                parameter_51,
                parameter_50,
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

        # pd_op.add: (4x128x-1x-1xf32) <- (4x128x-1x-1xf32, 4x128x-1x-1xf32)
        add_3 = paddle._C_ops.add(batch_norm__65, relu_8)

        # pd_op.relu: (4x128x-1x-1xf32) <- (4x128x-1x-1xf32)
        relu_16 = paddle._C_ops.relu(add_3)

        # pd_op.conv2d: (4x256x-1x-1xf32) <- (4x128x-1x-1xf32, 256x128x3x3xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            relu_16, parameter_49, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__134,
            batch_norm__71,
            batch_norm__72,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_13,
                parameter_48,
                parameter_47,
                parameter_46,
                parameter_45,
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

        # pd_op.relu: (4x256x-1x-1xf32) <- (4x256x-1x-1xf32)
        relu_10 = paddle._C_ops.relu(batch_norm__134)

        # pd_op.conv2d: (4x256x-1x-1xf32) <- (4x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            relu_10, parameter_44, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__76,
            batch_norm__77,
            batch_norm__78,
            batch_norm__79,
            batch_norm__80,
            batch_norm__81,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_14,
                parameter_43,
                parameter_42,
                parameter_41,
                parameter_40,
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

        # pd_op.pool2d: (4x128x-1x-1xf32) <- (4x128x-1x-1xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(
            relu_16,
            full_int_array_1,
            [2, 2],
            [0, 0],
            True,
            True,
            "NCHW",
            "avg",
            False,
            False,
            "EXPLICIT",
        )

        # pd_op.conv2d: (4x256x-1x-1xf32) <- (4x128x-1x-1xf32, 256x128x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            pool2d_2, parameter_39, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__82,
            batch_norm__83,
            batch_norm__84,
            batch_norm__85,
            batch_norm__86,
            batch_norm__87,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_15,
                parameter_38,
                parameter_37,
                parameter_36,
                parameter_35,
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

        # pd_op.add: (4x256x-1x-1xf32) <- (4x256x-1x-1xf32, 4x256x-1x-1xf32)
        add_4 = paddle._C_ops.add(batch_norm__76, batch_norm__82)

        # pd_op.relu: (4x256x-1x-1xf32) <- (4x256x-1x-1xf32)
        relu_11 = paddle._C_ops.relu(add_4)

        # pd_op.conv2d: (4x256x-1x-1xf32) <- (4x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            relu_11, parameter_34, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__135,
            batch_norm__88,
            batch_norm__89,
            batch_norm__90,
            batch_norm__91,
            batch_norm__92,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_16,
                parameter_33,
                parameter_32,
                parameter_31,
                parameter_30,
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

        # pd_op.relu: (4x256x-1x-1xf32) <- (4x256x-1x-1xf32)
        relu_12 = paddle._C_ops.relu(batch_norm__135)

        # pd_op.conv2d: (4x256x-1x-1xf32) <- (4x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            relu_12, parameter_29, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__93,
            batch_norm__94,
            batch_norm__95,
            batch_norm__96,
            batch_norm__97,
            batch_norm__98,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_17,
                parameter_28,
                parameter_27,
                parameter_26,
                parameter_25,
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

        # pd_op.add: (4x256x-1x-1xf32) <- (4x256x-1x-1xf32, 4x256x-1x-1xf32)
        add_5 = paddle._C_ops.add(batch_norm__93, relu_11)

        # pd_op.relu: (4x256x-1x-1xf32) <- (4x256x-1x-1xf32)
        relu_17 = paddle._C_ops.relu(add_5)

        # pd_op.conv2d: (4x512x-1x-1xf32) <- (4x256x-1x-1xf32, 512x256x3x3xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            relu_17, parameter_24, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (4x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__136,
            batch_norm__99,
            batch_norm__100,
            batch_norm__101,
            batch_norm__102,
            batch_norm__103,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_18,
                parameter_23,
                parameter_22,
                parameter_21,
                parameter_20,
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

        # pd_op.relu: (4x512x-1x-1xf32) <- (4x512x-1x-1xf32)
        relu_13 = paddle._C_ops.relu(batch_norm__136)

        # pd_op.conv2d: (4x512x-1x-1xf32) <- (4x512x-1x-1xf32, 512x512x3x3xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            relu_13, parameter_19, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (4x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__104,
            batch_norm__105,
            batch_norm__106,
            batch_norm__107,
            batch_norm__108,
            batch_norm__109,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_19,
                parameter_18,
                parameter_17,
                parameter_16,
                parameter_15,
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

        # pd_op.pool2d: (4x256x-1x-1xf32) <- (4x256x-1x-1xf32, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(
            relu_17,
            full_int_array_1,
            [2, 2],
            [0, 0],
            True,
            True,
            "NCHW",
            "avg",
            False,
            False,
            "EXPLICIT",
        )

        # pd_op.conv2d: (4x512x-1x-1xf32) <- (4x256x-1x-1xf32, 512x256x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            pool2d_3, parameter_14, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (4x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__110,
            batch_norm__111,
            batch_norm__112,
            batch_norm__113,
            batch_norm__114,
            batch_norm__115,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_20,
                parameter_13,
                parameter_12,
                parameter_11,
                parameter_10,
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

        # pd_op.add: (4x512x-1x-1xf32) <- (4x512x-1x-1xf32, 4x512x-1x-1xf32)
        add_6 = paddle._C_ops.add(batch_norm__104, batch_norm__110)

        # pd_op.relu: (4x512x-1x-1xf32) <- (4x512x-1x-1xf32)
        relu_14 = paddle._C_ops.relu(add_6)

        # pd_op.conv2d: (4x512x-1x-1xf32) <- (4x512x-1x-1xf32, 512x512x3x3xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            relu_14, parameter_9, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (4x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__137,
            batch_norm__116,
            batch_norm__117,
            batch_norm__118,
            batch_norm__119,
            batch_norm__120,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_21,
                parameter_8,
                parameter_7,
                parameter_6,
                parameter_5,
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

        # pd_op.relu: (4x512x-1x-1xf32) <- (4x512x-1x-1xf32)
        relu_15 = paddle._C_ops.relu(batch_norm__137)

        # pd_op.conv2d: (4x512x-1x-1xf32) <- (4x512x-1x-1xf32, 512x512x3x3xf32)
        conv2d_22 = paddle._C_ops.conv2d(
            relu_15, parameter_4, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (4x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (4x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__121,
            batch_norm__122,
            batch_norm__123,
            batch_norm__124,
            batch_norm__125,
            batch_norm__126,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_22,
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

        # pd_op.add: (4x512x-1x-1xf32) <- (4x512x-1x-1xf32, 4x512x-1x-1xf32)
        add_7 = paddle._C_ops.add(batch_norm__121, relu_14)

        # pd_op.relu: (4x512x-1x-1xf32) <- (4x512x-1x-1xf32)
        relu_18 = paddle._C_ops.relu(add_7)
        return (
            conv2d_0,
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
            relu_0,
            conv2d_1,
            batch_norm__5,
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            relu_1,
            conv2d_2,
            batch_norm__10,
            batch_norm__11,
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            relu_2,
            full_int_array_0,
            pool2d_0,
            conv2d_3,
            batch_norm__15,
            batch_norm__16,
            batch_norm__17,
            batch_norm__18,
            batch_norm__19,
            relu_3,
            conv2d_4,
            batch_norm__20,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
            batch_norm__24,
            batch_norm__25,
            conv2d_5,
            batch_norm__26,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
            batch_norm__30,
            batch_norm__31,
            relu_4,
            conv2d_6,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
            batch_norm__36,
            relu_5,
            conv2d_7,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
            batch_norm__42,
            relu_6,
            conv2d_8,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
            relu_7,
            conv2d_9,
            batch_norm__48,
            batch_norm__49,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
            full_int_array_1,
            pool2d_1,
            conv2d_10,
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
            relu_8,
            conv2d_11,
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            batch_norm__64,
            relu_9,
            conv2d_12,
            batch_norm__65,
            batch_norm__66,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            batch_norm__70,
            conv2d_13,
            batch_norm__71,
            batch_norm__72,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            relu_10,
            conv2d_14,
            batch_norm__76,
            batch_norm__77,
            batch_norm__78,
            batch_norm__79,
            batch_norm__80,
            batch_norm__81,
            assign_0,
            pool2d_2,
            conv2d_15,
            batch_norm__82,
            batch_norm__83,
            batch_norm__84,
            batch_norm__85,
            batch_norm__86,
            batch_norm__87,
            relu_11,
            conv2d_16,
            batch_norm__88,
            batch_norm__89,
            batch_norm__90,
            batch_norm__91,
            batch_norm__92,
            relu_12,
            conv2d_17,
            batch_norm__93,
            batch_norm__94,
            batch_norm__95,
            batch_norm__96,
            batch_norm__97,
            batch_norm__98,
            conv2d_18,
            batch_norm__99,
            batch_norm__100,
            batch_norm__101,
            batch_norm__102,
            batch_norm__103,
            relu_13,
            conv2d_19,
            batch_norm__104,
            batch_norm__105,
            batch_norm__106,
            batch_norm__107,
            batch_norm__108,
            batch_norm__109,
            assign_1,
            pool2d_3,
            conv2d_20,
            batch_norm__110,
            batch_norm__111,
            batch_norm__112,
            batch_norm__113,
            batch_norm__114,
            batch_norm__115,
            relu_14,
            conv2d_21,
            batch_norm__116,
            batch_norm__117,
            batch_norm__118,
            batch_norm__119,
            batch_norm__120,
            relu_15,
            conv2d_22,
            batch_norm__121,
            batch_norm__122,
            batch_norm__123,
            batch_norm__124,
            batch_norm__125,
            batch_norm__126,
            relu_16,
            relu_17,
            relu_18,
        )
