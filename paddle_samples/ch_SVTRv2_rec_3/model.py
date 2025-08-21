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
        data_0,
        data_1,
    ):
        # pd_op.add: (5x-1x384xf32) <- (5x-1x384xf32, 5x-1x384xf32)
        add_0 = paddle._C_ops.add(data_0, data_1)

        # pd_op.layer_norm: (5x-1x384xf32, 5x-1xf32, 5x-1xf32) <- (5x-1x384xf32, 384xf32, 384xf32)
        layer_norm_2, layer_norm_0, layer_norm_1 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_0, parameter_5, parameter_4, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (5x-1x1536xf32) <- (5x-1x384xf32, 384x1536xf32)
        matmul_0 = paddle._C_ops.matmul(layer_norm_2, parameter_3, False, False)

        # pd_op.add: (5x-1x1536xf32) <- (5x-1x1536xf32, 1536xf32)
        add_1 = paddle._C_ops.add(matmul_0, parameter_2)

        # pd_op.gelu: (5x-1x1536xf32) <- (5x-1x1536xf32)
        gelu_0 = paddle._C_ops.gelu(add_1, False)

        # pd_op.matmul: (5x-1x384xf32) <- (5x-1x1536xf32, 1536x384xf32)
        matmul_1 = paddle._C_ops.matmul(gelu_0, parameter_1, False, False)

        # pd_op.add: (5x-1x384xf32) <- (5x-1x384xf32, 384xf32)
        add_2 = paddle._C_ops.add(matmul_1, parameter_0)
        return (
            add_0,
            layer_norm_0,
            layer_norm_1,
            matmul_0,
            add_1,
            gelu_0,
            matmul_1,
            layer_norm_2,
            add_2,
        )
