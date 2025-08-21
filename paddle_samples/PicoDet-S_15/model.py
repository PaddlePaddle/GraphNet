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
        data_0,
    ):
        # pd_op.depthwise_conv2d: (16x16x-1x-1xf32) <- (16x-1x-1x-1xf32, 16x1x3x3xf32)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(
            data_0, parameter_9, [1, 1], [1, 1], "EXPLICIT", 16, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (16x16x-1x-1xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (16x16x-1x-1xf32, 16xf32, 16xf32, 16xf32, 16xf32)
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

        # pd_op.hardswish: (16x16x-1x-1xf32) <- (16x16x-1x-1xf32)
        hardswish_0 = paddle._C_ops.hardswish(batch_norm__0)

        # pd_op.conv2d: (16x24x-1x-1xf32) <- (16x16x-1x-1xf32, 24x16x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            hardswish_0, parameter_4, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (16x24x-1x-1xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (16x24x-1x-1xf32, 24xf32, 24xf32, 24xf32, 24xf32)
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

        # pd_op.hardswish: (16x24x-1x-1xf32) <- (16x24x-1x-1xf32)
        hardswish_1 = paddle._C_ops.hardswish(batch_norm__6)
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
        )
