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
        data_0,
    ):
        # pd_op.conv2d: (2x256x-1x-1xf32) <- (2x512x-1x-1xf32, 256x512x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_39, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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

        # pd_op.sigmoid: (2x256x-1x-1xf32) <- (2x256x-1x-1xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(batch_norm__0)

        # pd_op.multiply: (2x256x-1x-1xf32) <- (2x256x-1x-1xf32, 2x256x-1x-1xf32)
        multiply_0 = paddle._C_ops.multiply(batch_norm__0, sigmoid_0)

        # pd_op.conv2d: (2x256x-1x-1xf32) <- (2x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            multiply_0, parameter_34, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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

        # pd_op.conv2d: (2x256x-1x-1xf32) <- (2x256x-1x-1xf32, 256x256x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            multiply_0, parameter_29, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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

        # pd_op.add: (2x256x-1x-1xf32) <- (2x256x-1x-1xf32, 2x256x-1x-1xf32)
        add_0 = paddle._C_ops.add(batch_norm__6, batch_norm__12)

        # pd_op.silu: (2x256x-1x-1xf32) <- (2x256x-1x-1xf32)
        silu_0 = paddle._C_ops.silu(add_0)

        # pd_op.conv2d: (2x256x-1x-1xf32) <- (2x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            silu_0, parameter_24, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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

        # pd_op.conv2d: (2x256x-1x-1xf32) <- (2x256x-1x-1xf32, 256x256x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            silu_0, parameter_19, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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

        # pd_op.add: (2x256x-1x-1xf32) <- (2x256x-1x-1xf32, 2x256x-1x-1xf32)
        add_1 = paddle._C_ops.add(batch_norm__18, batch_norm__24)

        # pd_op.silu: (2x256x-1x-1xf32) <- (2x256x-1x-1xf32)
        silu_1 = paddle._C_ops.silu(add_1)

        # pd_op.conv2d: (2x256x-1x-1xf32) <- (2x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            silu_1, parameter_14, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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

        # pd_op.conv2d: (2x256x-1x-1xf32) <- (2x256x-1x-1xf32, 256x256x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            silu_1, parameter_9, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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

        # pd_op.add: (2x256x-1x-1xf32) <- (2x256x-1x-1xf32, 2x256x-1x-1xf32)
        add_2 = paddle._C_ops.add(batch_norm__30, batch_norm__36)

        # pd_op.silu: (2x256x-1x-1xf32) <- (2x256x-1x-1xf32)
        silu_2 = paddle._C_ops.silu(add_2)

        # pd_op.conv2d: (2x256x-1x-1xf32) <- (2x512x-1x-1xf32, 256x512x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            data_0, parameter_4, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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

        # pd_op.sigmoid: (2x256x-1x-1xf32) <- (2x256x-1x-1xf32)
        sigmoid_1 = paddle._C_ops.sigmoid(batch_norm__42)

        # pd_op.multiply: (2x256x-1x-1xf32) <- (2x256x-1x-1xf32, 2x256x-1x-1xf32)
        multiply_1 = paddle._C_ops.multiply(batch_norm__42, sigmoid_1)

        # pd_op.add: (2x256x-1x-1xf32) <- (2x256x-1x-1xf32, 2x256x-1x-1xf32)
        add_3 = paddle._C_ops.add(silu_2, multiply_1)
        return (
            conv2d_0,
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
            batch_norm__5,
            sigmoid_0,
            multiply_0,
            conv2d_1,
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            batch_norm__10,
            batch_norm__11,
            conv2d_2,
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            batch_norm__16,
            batch_norm__17,
            add_0,
            silu_0,
            conv2d_3,
            batch_norm__18,
            batch_norm__19,
            batch_norm__20,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
            conv2d_4,
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
            add_1,
            silu_1,
            conv2d_5,
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
            conv2d_6,
            batch_norm__36,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
            add_2,
            silu_2,
            conv2d_7,
            batch_norm__42,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
            sigmoid_1,
            multiply_1,
            add_3,
        )
