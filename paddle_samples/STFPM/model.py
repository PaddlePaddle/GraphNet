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
        data_0,
    ):
        # pd_op.conv2d: (1x512x-1x-1xf32) <- (1x-1x-1x-1xf32, 512x256x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_24, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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

        # pd_op.relu: (1x512x-1x-1xf32) <- (1x512x-1x-1xf32)
        relu_1 = paddle._C_ops.relu(batch_norm__0)

        # pd_op.conv2d: (1x512x-1x-1xf32) <- (1x512x-1x-1xf32, 512x512x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            relu_1, parameter_19, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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

        # pd_op.conv2d: (1x512x-1x-1xf32) <- (1x-1x-1x-1xf32, 512x256x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            data_0, parameter_14, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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

        # pd_op.add: (1x512x-1x-1xf32) <- (1x512x-1x-1xf32, 1x512x-1x-1xf32)
        add_0 = paddle._C_ops.add(batch_norm__6, batch_norm__12)

        # pd_op.relu: (1x512x-1x-1xf32) <- (1x512x-1x-1xf32)
        relu_2 = paddle._C_ops.relu(add_0)

        # pd_op.conv2d: (1x512x-1x-1xf32) <- (1x512x-1x-1xf32, 512x512x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            relu_2, parameter_9, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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

        # pd_op.relu: (1x512x-1x-1xf32) <- (1x512x-1x-1xf32)
        relu_3 = paddle._C_ops.relu(batch_norm__18)

        # pd_op.conv2d: (1x512x-1x-1xf32) <- (1x512x-1x-1xf32, 512x512x3x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            relu_3, parameter_4, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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

        # pd_op.add: (1x512x-1x-1xf32) <- (1x512x-1x-1xf32, 1x512x-1x-1xf32)
        add_1 = paddle._C_ops.add(batch_norm__24, relu_2)

        # pd_op.relu: (1x512x-1x-1xf32) <- (1x512x-1x-1xf32)
        relu_0 = paddle._C_ops.relu(add_1)
        return relu_0
