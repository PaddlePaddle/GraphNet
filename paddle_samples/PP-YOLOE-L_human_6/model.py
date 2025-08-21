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
        data_0,
    ):
        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.pool2d: (2x-1x1x1xf32) <- (2x-1x-1x-1xf32, 2xi64)
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

        # pd_op.conv2d: (2x384x1x1xf32) <- (2x-1x1x1xf32, 384x384x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            pool2d_0, parameter_17, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [1, -1, 1, 1]

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_16, full_int_array_1)

        # pd_op.add: (2x384x1x1xf32) <- (2x384x1x1xf32, 1x384x1x1xf32)
        add_3 = paddle._C_ops.add(conv2d_0, reshape_0)

        # pd_op.sigmoid: (2x384x1x1xf32) <- (2x384x1x1xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(add_3)

        # pd_op.multiply: (2x384x-1x-1xf32) <- (2x-1x-1x-1xf32, 2x384x1x1xf32)
        multiply_0 = paddle._C_ops.multiply(data_0, sigmoid_0)

        # pd_op.conv2d: (2x384x-1x-1xf32) <- (2x384x-1x-1xf32, 384x384x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            multiply_0, parameter_15, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (2x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
            batch_norm__5,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_1,
                parameter_14,
                parameter_13,
                parameter_12,
                parameter_11,
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

        # pd_op.swish: (2x384x-1x-1xf32) <- (2x384x-1x-1xf32)
        swish_0 = paddle._C_ops.swish(batch_norm__0)

        # pd_op.add: (2x384x-1x-1xf32) <- (2x384x-1x-1xf32, 2x-1x-1x-1xf32)
        add_0 = paddle._C_ops.add(swish_0, data_0)

        # pd_op.conv2d: (2x1x-1x-1xf32) <- (2x384x-1x-1xf32, 1x384x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            add_0, parameter_10, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x1x1x1xf32) <- (1xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_9, full_int_array_1)

        # pd_op.add: (2x1x-1x-1xf32) <- (2x1x-1x-1xf32, 1x1x1x1xf32)
        add_1 = paddle._C_ops.add(conv2d_2, reshape_1)

        # pd_op.conv2d: (2x384x1x1xf32) <- (2x-1x1x1xf32, 384x384x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            pool2d_0, parameter_8, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(parameter_7, full_int_array_1)

        # pd_op.add: (2x384x1x1xf32) <- (2x384x1x1xf32, 1x384x1x1xf32)
        add_4 = paddle._C_ops.add(conv2d_3, reshape_2)

        # pd_op.sigmoid: (2x384x1x1xf32) <- (2x384x1x1xf32)
        sigmoid_1 = paddle._C_ops.sigmoid(add_4)

        # pd_op.multiply: (2x384x-1x-1xf32) <- (2x-1x-1x-1xf32, 2x384x1x1xf32)
        multiply_1 = paddle._C_ops.multiply(data_0, sigmoid_1)

        # pd_op.conv2d: (2x384x-1x-1xf32) <- (2x384x-1x-1xf32, 384x384x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            multiply_1, parameter_6, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (2x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            batch_norm__10,
            batch_norm__11,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_4,
                parameter_5,
                parameter_4,
                parameter_3,
                parameter_2,
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

        # pd_op.swish: (2x384x-1x-1xf32) <- (2x384x-1x-1xf32)
        swish_1 = paddle._C_ops.swish(batch_norm__6)

        # pd_op.conv2d: (2x68x-1x-1xf32) <- (2x384x-1x-1xf32, 68x384x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            swish_1, parameter_1, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x68x1x1xf32) <- (68xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(parameter_0, full_int_array_1)

        # pd_op.add: (2x68x-1x-1xf32) <- (2x68x-1x-1xf32, 1x68x1x1xf32)
        add_2 = paddle._C_ops.add(conv2d_5, reshape_3)

        # pd_op.sigmoid: (2x1x-1x-1xf32) <- (2x1x-1x-1xf32)
        sigmoid_2 = paddle._C_ops.sigmoid(add_1)

        # pd_op.flatten: (2x1x-1xf32) <- (2x1x-1x-1xf32)
        flatten_0 = paddle._C_ops.flatten(sigmoid_2, 2, 3)

        # pd_op.transpose: (2x-1x1xf32) <- (2x1x-1xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_0, [0, 2, 1])

        # pd_op.flatten: (2x68x-1xf32) <- (2x68x-1x-1xf32)
        flatten_1 = paddle._C_ops.flatten(add_2, 2, 3)

        # pd_op.transpose: (2x-1x68xf32) <- (2x68x-1xf32)
        transpose_1 = paddle._C_ops.transpose(flatten_1, [0, 2, 1])
        return (
            full_int_array_0,
            conv2d_0,
            reshape_0,
            sigmoid_0,
            multiply_0,
            conv2d_1,
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
            batch_norm__5,
            swish_0,
            add_0,
            conv2d_2,
            reshape_1,
            conv2d_3,
            reshape_2,
            sigmoid_1,
            multiply_1,
            conv2d_4,
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            batch_norm__10,
            batch_norm__11,
            swish_1,
            conv2d_5,
            reshape_3,
            pool2d_0,
            add_1,
            add_2,
            sigmoid_2,
            transpose_0,
            transpose_1,
        )
