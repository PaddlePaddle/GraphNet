import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, parameter_0, parameter_1, parameter_2, parameter_3, data_0):
        # pd_op.conv2d: (2x96x224x168xf32) <- (2x3x896x672xf32, 96x3x4x4xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_3, [4, 4], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_2, full_int_array_0)

        # pd_op.add: (2x96x224x168xf32) <- (2x96x224x168xf32, 1x96x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_0, reshape_0)

        # pd_op.flatten: (2x96x37632xf32) <- (2x96x224x168xf32)
        flatten_0 = paddle._C_ops.flatten(add_0, 2, 3)

        # pd_op.transpose: (2x37632x96xf32) <- (2x96x37632xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_0, [0, 2, 1])

        # pd_op.layer_norm: (2x37632x96xf32, 2x37632xf32, 2x37632xf32) <- (2x37632x96xf32, 96xf32, 96xf32)
        layer_norm_2, layer_norm_0, layer_norm_1 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_0, parameter_1, parameter_0, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.transpose: (2x96x37632xf32) <- (2x37632x96xf32)
        transpose_1 = paddle._C_ops.transpose(layer_norm_2, [0, 2, 1])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [-1, 96, 224, 168]

        # pd_op.reshape: (2x96x224x168xf32) <- (2x96x37632xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(transpose_1, full_int_array_1)
        return (
            conv2d_0,
            reshape_0,
            add_0,
            transpose_0,
            layer_norm_0,
            layer_norm_1,
            transpose_1,
            reshape_1,
        )
