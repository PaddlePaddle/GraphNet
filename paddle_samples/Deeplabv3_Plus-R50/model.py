import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self, parameter_0, parameter_1, parameter_2, parameter_3, parameter_4, data_0
    ):
        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.pool2d: (2x2048x1x1xf32) <- (2x2048x64x128xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            data_0,
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

        # pd_op.conv2d: (2x256x1x1xf32) <- (2x2048x1x1xf32, 256x2048x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            pool2d_0, parameter_4, [1, 1], [0, 0], "SAME", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x256x1x1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x1x1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__5,
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
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

        # pd_op.relu: (2x256x1x1xf32) <- (2x256x1x1xf32)
        relu_0 = paddle._C_ops.relu(batch_norm__5)
        return (
            full_int_array_0,
            pool2d_0,
            conv2d_0,
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
            relu_0,
        )
