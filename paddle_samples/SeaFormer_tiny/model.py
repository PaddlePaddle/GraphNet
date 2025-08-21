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
        data_0,
    ):
        # pd_op.conv2d: (2x96x-1x-1xf32) <- (2x-1x-1x-1xf32, 96x32x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_14, [1, 1], [0, 0], "SAME", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__16,
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
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

        # pd_op.relu: (2x96x-1x-1xf32) <- (2x96x-1x-1xf32)
        relu_0 = paddle._C_ops.relu(batch_norm__16)

        # pd_op.depthwise_conv2d: (2x96x-1x-1xf32) <- (2x96x-1x-1xf32, 96x1x3x3xf32)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(
            relu_0, parameter_9, [2, 2], [1, 1], "EXPLICIT", 96, [1, 1], "NCHW"
        )

        # pd_op.batch_norm_: (2x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__17,
            batch_norm__5,
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
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

        # pd_op.relu: (2x96x-1x-1xf32) <- (2x96x-1x-1xf32)
        relu_1 = paddle._C_ops.relu(batch_norm__17)

        # pd_op.conv2d: (2x64x-1x-1xf32) <- (2x96x-1x-1xf32, 64x96x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            relu_1, parameter_4, [1, 1], [0, 0], "SAME", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (2x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__15,
            batch_norm__10,
            batch_norm__11,
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
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
        return (
            conv2d_0,
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
            relu_0,
            depthwise_conv2d_0,
            batch_norm__5,
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            relu_1,
            conv2d_1,
            batch_norm__10,
            batch_norm__11,
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
        )
