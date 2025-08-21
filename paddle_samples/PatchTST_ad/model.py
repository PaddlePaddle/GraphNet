import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, parameter_0, parameter_1, parameter_2, parameter_3, data_0):
        # pd_op.matmul: (32x12x128xf32) <- (32x12x16xf32, 16x128xf32)
        matmul_0 = paddle._C_ops.matmul(data_0, parameter_3, False, False)

        # pd_op.add: (32x12x128xf32) <- (32x12x128xf32, 128xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_2)

        # pd_op.gelu: (32x12x128xf32) <- (32x12x128xf32)
        gelu_0 = paddle._C_ops.gelu(add_0, False)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (32x12x128xf32, 32x12x128xui8) <- (32x12x128xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                gelu_0, None, full_0, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (32x12x16xf32) <- (32x12x128xf32, 128x16xf32)
        matmul_1 = paddle._C_ops.matmul(dropout_0, parameter_1, False, False)

        # pd_op.add: (32x12x16xf32) <- (32x12x16xf32, 16xf32)
        add_1 = paddle._C_ops.add(matmul_1, parameter_0)
        return matmul_0, add_0, full_0, dropout_0, dropout_1, matmul_1, add_1
