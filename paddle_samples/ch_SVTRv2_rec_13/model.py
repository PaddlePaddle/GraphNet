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
        # pd_op.conv2d: (4x128x16x80xf32) <- (4x128x16x80xf32, 128x32x5x5xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_9, [1, 1], [2, 2], "EXPLICIT", [1, 1], 4, "NCHW"
        )

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_8, full_int_array_0)

        # pd_op.add: (4x128x16x80xf32) <- (4x128x16x80xf32, 1x128x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_0, reshape_0)

        # pd_op.add: (4x128x16x80xf32) <- (4x128x16x80xf32, 4x128x16x80xf32)
        add_1 = paddle._C_ops.add(data_0, add_0)

        # pd_op.flatten: (4x128x1280xf32) <- (4x128x16x80xf32)
        flatten_0 = paddle._C_ops.flatten(add_1, 2, 3)

        # pd_op.transpose: (4x1280x128xf32) <- (4x128x1280xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_0, [0, 2, 1])

        # pd_op.layer_norm: (4x1280x128xf32, 4x1280xf32, 4x1280xf32) <- (4x1280x128xf32, 128xf32, 128xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_0, parameter_7, parameter_6, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (4x1280x512xf32) <- (4x1280x128xf32, 128x512xf32)
        matmul_0 = paddle._C_ops.matmul(layer_norm_0, parameter_5, False, False)

        # pd_op.add: (4x1280x512xf32) <- (4x1280x512xf32, 512xf32)
        add_2 = paddle._C_ops.add(matmul_0, parameter_4)

        # pd_op.gelu: (4x1280x512xf32) <- (4x1280x512xf32)
        gelu_0 = paddle._C_ops.gelu(add_2, False)

        # pd_op.matmul: (4x1280x128xf32) <- (4x1280x512xf32, 512x128xf32)
        matmul_1 = paddle._C_ops.matmul(gelu_0, parameter_3, False, False)

        # pd_op.add: (4x1280x128xf32) <- (4x1280x128xf32, 128xf32)
        add_3 = paddle._C_ops.add(matmul_1, parameter_2)

        # pd_op.add: (4x1280x128xf32) <- (4x1280x128xf32, 4x1280x128xf32)
        add_4 = paddle._C_ops.add(layer_norm_0, add_3)

        # pd_op.layer_norm: (4x1280x128xf32, 4x1280xf32, 4x1280xf32) <- (4x1280x128xf32, 128xf32, 128xf32)
        layer_norm_5, layer_norm_3, layer_norm_4 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_4, parameter_1, parameter_0, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.transpose: (4x128x1280xf32) <- (4x1280x128xf32)
        transpose_1 = paddle._C_ops.transpose(layer_norm_5, [0, 2, 1])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [0, 128, 16, 80]

        # pd_op.reshape: (4x128x16x80xf32) <- (4x128x1280xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(transpose_1, full_int_array_1)
        return (
            conv2d_0,
            reshape_0,
            add_0,
            add_1,
            transpose_0,
            layer_norm_0,
            layer_norm_1,
            layer_norm_2,
            matmul_0,
            add_2,
            gelu_0,
            matmul_1,
            add_3,
            add_4,
            layer_norm_3,
            layer_norm_4,
            transpose_1,
            reshape_1,
        )
