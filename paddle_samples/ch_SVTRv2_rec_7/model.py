import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, parameter_0, parameter_1, data_0, data_1):
        # pd_op.add: (4x1280x128xf32) <- (4x1280x128xf32, 4x1280x128xf32)
        add_0 = paddle._C_ops.add(data_0, data_1)

        # pd_op.layer_norm: (4x1280x128xf32, 4x1280xf32, 4x1280xf32) <- (4x1280x128xf32, 128xf32, 128xf32)
        layer_norm_2, layer_norm_0, layer_norm_1 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_0, parameter_1, parameter_0, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.transpose: (4x128x1280xf32) <- (4x1280x128xf32)
        transpose_0 = paddle._C_ops.transpose(layer_norm_2, [0, 2, 1])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [0, 128, 16, 80]

        # pd_op.reshape: (4x128x16x80xf32) <- (4x128x1280xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(transpose_0, full_int_array_0)
        return add_0, layer_norm_0, layer_norm_1, transpose_0, reshape_0
