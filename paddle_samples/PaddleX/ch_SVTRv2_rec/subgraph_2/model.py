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
        data_1,
    ):
        # pd_op.conv2d: (-1x64x-1x160xf32) <- (-1x3x-1x320xf32, 64x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_1, parameter_11, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_1, parameter_11

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_10, full_int_array_0)
        del parameter_10

        # pd_op.add: (-1x64x-1x160xf32) <- (-1x64x-1x160xf32, 1x64x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_0, reshape_0)
        del conv2d_0, reshape_0

        # pd_op.batch_norm_: (-1x64x-1x160xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (-1x64x-1x160xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
        del add_0, parameter_6, parameter_7, parameter_8, parameter_9

        # pd_op.gelu: (-1x64x-1x160xf32) <- (-1x64x-1x160xf32)
        gelu_1 = paddle._C_ops.gelu(batch_norm__0, False)
        del batch_norm__0

        # pd_op.conv2d: (-1x128x-1x80xf32) <- (-1x64x-1x160xf32, 128x64x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            gelu_1, parameter_5, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del gelu_1, parameter_5

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_4, full_int_array_0)
        del full_int_array_0, parameter_4

        # pd_op.add: (-1x128x-1x80xf32) <- (-1x128x-1x80xf32, 1x128x1x1xf32)
        add_1 = paddle._C_ops.add(conv2d_1, reshape_1)
        del conv2d_1, reshape_1

        # pd_op.batch_norm_: (-1x128x-1x80xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x-1x80xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
        del add_1, parameter_0, parameter_1, parameter_2, parameter_3

        # pd_op.gelu: (-1x128x-1x80xf32) <- (-1x128x-1x80xf32)
        gelu_0 = paddle._C_ops.gelu(batch_norm__6, False)
        del batch_norm__6

        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("4"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        floor_divide_0 = paddle._C_ops.floor_divide(data_0, full_0)
        del data_0, full_0

        return gelu_0, floor_divide_0
