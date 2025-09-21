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
        data_0,
    ):
        # pd_op.conv2d: (1x32x-1x-1xf32) <- (1x-1x-1x-1xf32, 32x32x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_6, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_0, parameter_6

        # pd_op.batch_norm_: (1x32x-1x-1xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (1x32x-1x-1xf32, 32xf32, 32xf32, 32xf32, 32xf32)
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
                parameter_5,
                parameter_4,
                parameter_3,
                parameter_2,
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
        del conv2d_0, parameter_2, parameter_3, parameter_4, parameter_5

        # pd_op.sigmoid: (1x32x-1x-1xf32) <- (1x32x-1x-1xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(batch_norm__0)

        # pd_op.multiply: (1x32x-1x-1xf32) <- (1x32x-1x-1xf32, 1x32x-1x-1xf32)
        multiply_0 = paddle._C_ops.multiply(batch_norm__0, sigmoid_0)
        del batch_norm__0, sigmoid_0

        # pd_op.conv2d: (1x128x-1x-1xf32) <- (1x32x-1x-1xf32, 128x32x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            multiply_0, parameter_1, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del multiply_0, parameter_1

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_0, full_int_array_0)
        del full_int_array_0, parameter_0

        # pd_op.add: (1x128x-1x-1xf32) <- (1x128x-1x-1xf32, 1x128x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_1, reshape_0)
        del conv2d_1, reshape_0

        return add_0
