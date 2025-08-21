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
        data_0,
    ):
        # pd_op.depthwise_conv2d: (24x128x-1x-1xf32) <- (24x-1x-1x-1xf32, 128x1x3x3xf32)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(
            data_0, parameter_59, [2, 2], [1, 1], "EXPLICIT", 128, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (24x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (24x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
            batch_norm__5,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_0,
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

        # pd_op.hardswish: (24x128x-1x-1xf32) <- (24x128x-1x-1xf32)
        hardswish_0 = paddle._C_ops.hardswish(batch_norm__0)

        # pd_op.conv2d: (24x256x-1x-1xf32) <- (24x128x-1x-1xf32, 256x128x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            hardswish_0, parameter_54, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (24x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (24x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            batch_norm__10,
            batch_norm__11,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_0,
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

        # pd_op.hardswish: (24x256x-1x-1xf32) <- (24x256x-1x-1xf32)
        hardswish_1 = paddle._C_ops.hardswish(batch_norm__6)

        # pd_op.depthwise_conv2d: (24x256x-1x-1xf32) <- (24x256x-1x-1xf32, 256x1x5x5xf32)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(
            hardswish_1, parameter_49, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (24x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (24x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            batch_norm__16,
            batch_norm__17,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_1,
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

        # pd_op.hardswish: (24x256x-1x-1xf32) <- (24x256x-1x-1xf32)
        hardswish_2 = paddle._C_ops.hardswish(batch_norm__12)

        # pd_op.conv2d: (24x256x-1x-1xf32) <- (24x256x-1x-1xf32, 256x256x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            hardswish_2, parameter_44, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (24x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (24x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__18,
            batch_norm__19,
            batch_norm__20,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_1,
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

        # pd_op.hardswish: (24x256x-1x-1xf32) <- (24x256x-1x-1xf32)
        hardswish_3 = paddle._C_ops.hardswish(batch_norm__18)

        # pd_op.depthwise_conv2d: (24x256x-1x-1xf32) <- (24x256x-1x-1xf32, 256x1x5x5xf32)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(
            hardswish_3, parameter_39, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (24x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (24x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_2,
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

        # pd_op.hardswish: (24x256x-1x-1xf32) <- (24x256x-1x-1xf32)
        hardswish_4 = paddle._C_ops.hardswish(batch_norm__24)

        # pd_op.conv2d: (24x256x-1x-1xf32) <- (24x256x-1x-1xf32, 256x256x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            hardswish_4, parameter_34, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (24x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (24x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_2,
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

        # pd_op.hardswish: (24x256x-1x-1xf32) <- (24x256x-1x-1xf32)
        hardswish_5 = paddle._C_ops.hardswish(batch_norm__30)

        # pd_op.depthwise_conv2d: (24x256x-1x-1xf32) <- (24x256x-1x-1xf32, 256x1x5x5xf32)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(
            hardswish_5, parameter_29, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (24x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (24x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__36,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_3,
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

        # pd_op.hardswish: (24x256x-1x-1xf32) <- (24x256x-1x-1xf32)
        hardswish_6 = paddle._C_ops.hardswish(batch_norm__36)

        # pd_op.conv2d: (24x256x-1x-1xf32) <- (24x256x-1x-1xf32, 256x256x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            hardswish_6, parameter_24, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (24x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (24x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__42,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_3,
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

        # pd_op.hardswish: (24x256x-1x-1xf32) <- (24x256x-1x-1xf32)
        hardswish_7 = paddle._C_ops.hardswish(batch_norm__42)

        # pd_op.depthwise_conv2d: (24x256x-1x-1xf32) <- (24x256x-1x-1xf32, 256x1x5x5xf32)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(
            hardswish_7, parameter_19, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (24x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (24x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__48,
            batch_norm__49,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_4,
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

        # pd_op.hardswish: (24x256x-1x-1xf32) <- (24x256x-1x-1xf32)
        hardswish_8 = paddle._C_ops.hardswish(batch_norm__48)

        # pd_op.conv2d: (24x256x-1x-1xf32) <- (24x256x-1x-1xf32, 256x256x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            hardswish_8, parameter_14, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (24x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (24x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_4,
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

        # pd_op.hardswish: (24x256x-1x-1xf32) <- (24x256x-1x-1xf32)
        hardswish_9 = paddle._C_ops.hardswish(batch_norm__54)

        # pd_op.depthwise_conv2d: (24x256x-1x-1xf32) <- (24x256x-1x-1xf32, 256x1x5x5xf32)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(
            hardswish_9, parameter_9, [1, 1], [2, 2], "EXPLICIT", 256, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (24x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (24x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_5,
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

        # pd_op.hardswish: (24x256x-1x-1xf32) <- (24x256x-1x-1xf32)
        hardswish_10 = paddle._C_ops.hardswish(batch_norm__60)

        # pd_op.conv2d: (24x256x-1x-1xf32) <- (24x256x-1x-1xf32, 256x256x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            hardswish_10, parameter_4, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (24x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (24x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__66,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            batch_norm__70,
            batch_norm__71,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_5,
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

        # pd_op.hardswish: (24x256x-1x-1xf32) <- (24x256x-1x-1xf32)
        hardswish_11 = paddle._C_ops.hardswish(batch_norm__66)
        return (
            depthwise_conv2d_0,
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
            batch_norm__5,
            hardswish_0,
            conv2d_0,
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            batch_norm__10,
            batch_norm__11,
            hardswish_1,
            depthwise_conv2d_1,
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            batch_norm__16,
            batch_norm__17,
            hardswish_2,
            conv2d_1,
            batch_norm__18,
            batch_norm__19,
            batch_norm__20,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
            hardswish_3,
            depthwise_conv2d_2,
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
            hardswish_4,
            conv2d_2,
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
            hardswish_5,
            depthwise_conv2d_3,
            batch_norm__36,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
            hardswish_6,
            conv2d_3,
            batch_norm__42,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
            hardswish_7,
            depthwise_conv2d_4,
            batch_norm__48,
            batch_norm__49,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
            hardswish_8,
            conv2d_4,
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
            hardswish_9,
            depthwise_conv2d_5,
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
            hardswish_10,
            conv2d_5,
            batch_norm__66,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            batch_norm__70,
            batch_norm__71,
            hardswish_11,
        )
