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
        data_0,
    ):
        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x-1x-1x-1xf32, 256x512x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_11, [1, 1], [0, 0], "SAME", [1, 1], 1, "NCHW"
        )

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_10, full_int_array_0)

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_0, reshape_0)

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
            batch_norm__5,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_0,
                parameter_9,
                parameter_8,
                parameter_7,
                parameter_6,
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

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_1 = paddle._C_ops.relu(batch_norm__0)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            relu_1, parameter_5, [1, 1], [0, 0], "SAME", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_4, full_int_array_0)

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_1 = paddle._C_ops.add(conv2d_1, reshape_1)

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            batch_norm__10,
            batch_norm__11,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_1,
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

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_0 = paddle._C_ops.relu(batch_norm__6)
        return relu_0
