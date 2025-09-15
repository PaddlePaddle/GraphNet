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
        data_1,
    ):
        # pd_op.conv2d: (-1x128x64x64xf32) <- (-1x-1x64x64xf32, 128x64x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_9, [1, 1], [0, 0], "SAME", [1, 1], 1, "NCHW"
        )
        del data_0, parameter_9

        # pd_op.batch_norm_: (-1x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x-1x-1x-1xf32, 128x192x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            data_1, parameter_4, [1, 1], [0, 0], "SAME", [1, 1], 1, "NCHW"
        )
        del data_1, parameter_4

        # pd_op.batch_norm_: (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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

        # pd_op.hardsigmoid: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        hardsigmoid_0 = paddle._C_ops.hardsigmoid(
            batch_norm__6, float("0.166667"), float("0.5")
        )
        del batch_norm__6

        # pd_op.bilinear_interp: (-1x128x64x64xf32) <- (-1x128x-1x-1xf32, None, None, None)
        bilinear_interp_0 = paddle._C_ops.bilinear_interp(
            hardsigmoid_0,
            None,
            None,
            None,
            "NCHW",
            -1,
            64,
            64,
            [],
            "bilinear",
            False,
            0,
        )
        del hardsigmoid_0

        # pd_op.multiply: (-1x128x64x64xf32) <- (-1x128x64x64xf32, -1x128x64x64xf32)
        multiply_0 = paddle._C_ops.multiply(batch_norm__0, bilinear_interp_0)
        del batch_norm__0, bilinear_interp_0

        return multiply_0
