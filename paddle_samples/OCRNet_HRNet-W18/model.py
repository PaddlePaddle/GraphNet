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
        data_0,
    ):
        # pd_op.conv2d: (-1x270x-1x-1xf32) <- (-1x-1x-1x-1xf32, 270x270x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_7, [1, 1], [0, 0], "SAME", [1, 1], 1, "NCHW"
        )

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x270x1x1xf32) <- (270xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_6, full_int_array_0)

        # pd_op.add: (-1x270x-1x-1xf32) <- (-1x270x-1x-1xf32, 1x270x1x1xf32)
        add_1 = paddle._C_ops.add(conv2d_0, reshape_0)

        # pd_op.batch_norm_: (-1x270x-1x-1xf32, 270xf32, 270xf32, 270xf32, 270xf32, -1xui8) <- (-1x270x-1x-1xf32, 270xf32, 270xf32, 270xf32, 270xf32)
        (
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
            batch_norm__5,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_1,
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

        # pd_op.relu: (-1x270x-1x-1xf32) <- (-1x270x-1x-1xf32)
        relu_0 = paddle._C_ops.relu(batch_norm__0)

        # pd_op.conv2d: (-1x2x-1x-1xf32) <- (-1x270x-1x-1xf32, 2x270x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            relu_0, parameter_1, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x2x1x1xf32) <- (2xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_0, full_int_array_0)

        # pd_op.add: (-1x2x-1x-1xf32) <- (-1x2x-1x-1xf32, 1x2x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_1, reshape_1)
        return add_0
