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
        # pd_op.conv2d: (1x64x-1x-1xf32) <- (1x-1x-1x-1xf32, 64x256x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_9, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_0, parameter_9

        # pd_op.batch_norm_: (1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
        del conv2d_0, parameter_5, parameter_6, parameter_7, parameter_8

        # pd_op.sigmoid: (1x64x-1x-1xf32) <- (1x64x-1x-1xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(batch_norm__0)

        # pd_op.multiply: (1x64x-1x-1xf32) <- (1x64x-1x-1xf32, 1x64x-1x-1xf32)
        multiply_0 = paddle._C_ops.multiply(batch_norm__0, sigmoid_0)
        del batch_norm__0, sigmoid_0

        # pd_op.bilinear_interp: (1x64x-1x-1xf32) <- (1x64x-1x-1xf32, None, None, None)
        bilinear_interp_1 = paddle._C_ops.bilinear_interp(
            multiply_0,
            None,
            None,
            None,
            "NCHW",
            -1,
            -1,
            -1,
            [float("2"), float("2")],
            "bilinear",
            False,
            0,
        )
        del multiply_0

        # pd_op.conv2d: (1x64x-1x-1xf32) <- (1x64x-1x-1xf32, 64x64x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            bilinear_interp_1,
            parameter_4,
            [1, 1],
            [1, 1],
            "EXPLICIT",
            [1, 1],
            1,
            "NCHW",
        )
        del bilinear_interp_1, parameter_4

        # pd_op.batch_norm_: (1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
        del conv2d_1, parameter_0, parameter_1, parameter_2, parameter_3

        # pd_op.sigmoid: (1x64x-1x-1xf32) <- (1x64x-1x-1xf32)
        sigmoid_1 = paddle._C_ops.sigmoid(batch_norm__6)

        # pd_op.multiply: (1x64x-1x-1xf32) <- (1x64x-1x-1xf32, 1x64x-1x-1xf32)
        multiply_1 = paddle._C_ops.multiply(batch_norm__6, sigmoid_1)
        del batch_norm__6, sigmoid_1

        # pd_op.bilinear_interp: (1x64x-1x-1xf32) <- (1x64x-1x-1xf32, None, None, None)
        bilinear_interp_0 = paddle._C_ops.bilinear_interp(
            multiply_1,
            None,
            None,
            None,
            "NCHW",
            -1,
            -1,
            -1,
            [float("2"), float("2")],
            "bilinear",
            False,
            0,
        )
        del multiply_1

        return bilinear_interp_0
