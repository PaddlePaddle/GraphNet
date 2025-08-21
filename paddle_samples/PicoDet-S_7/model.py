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
        data_0,
    ):
        # pd_op.depthwise_conv2d: (16x24x72x72xf32) <- (16x-1x144x144xf32, 24x1x3x3xf32)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(
            data_0, parameter_19, [2, 2], [1, 1], "EXPLICIT", 24, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (16x24x72x72xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (16x24x72x72xf32, 24xf32, 24xf32, 24xf32, 24xf32)
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

        # pd_op.hardswish: (16x24x72x72xf32) <- (16x24x72x72xf32)
        hardswish_0 = paddle._C_ops.hardswish(batch_norm__0)

        # pd_op.conv2d: (16x48x72x72xf32) <- (16x24x72x72xf32, 48x24x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            hardswish_0, parameter_14, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x48x72x72xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (16x48x72x72xf32, 48xf32, 48xf32, 48xf32, 48xf32)
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

        # pd_op.hardswish: (16x48x72x72xf32) <- (16x48x72x72xf32)
        hardswish_1 = paddle._C_ops.hardswish(batch_norm__6)

        # pd_op.depthwise_conv2d: (16x48x72x72xf32) <- (16x48x72x72xf32, 48x1x3x3xf32)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(
            hardswish_1, parameter_9, [1, 1], [1, 1], "EXPLICIT", 48, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (16x48x72x72xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (16x48x72x72xf32, 48xf32, 48xf32, 48xf32, 48xf32)
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

        # pd_op.hardswish: (16x48x72x72xf32) <- (16x48x72x72xf32)
        hardswish_2 = paddle._C_ops.hardswish(batch_norm__12)

        # pd_op.conv2d: (16x48x72x72xf32) <- (16x48x72x72xf32, 48x48x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            hardswish_2, parameter_4, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x48x72x72xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (16x48x72x72xf32, 48xf32, 48xf32, 48xf32, 48xf32)
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

        # pd_op.hardswish: (16x48x72x72xf32) <- (16x48x72x72xf32)
        hardswish_3 = paddle._C_ops.hardswish(batch_norm__18)
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
        )
