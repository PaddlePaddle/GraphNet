import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, parameter_0, parameter_1, parameter_2, parameter_3, data_0):
        # pd_op.matmul: (-1x2x32xf32) <- (-1x2x-1xf32, 16x32xf32)
        matmul_0 = paddle._C_ops.matmul(data_0, parameter_3, False, False)

        # pd_op.add: (-1x2x32xf32) <- (-1x2x32xf32, 32xf32)
        add_1 = paddle._C_ops.add(matmul_0, parameter_2)

        # pd_op.relu: (-1x2x32xf32) <- (-1x2x32xf32)
        relu_0 = paddle._C_ops.relu(add_1)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (-1x2x32xf32, -1x2x32xui8) <- (-1x2x32xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                relu_0, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (-1x2x96xf32) <- (-1x2x32xf32, 32x96xf32)
        matmul_1 = paddle._C_ops.matmul(dropout_0, parameter_1, False, False)

        # pd_op.add: (-1x2x96xf32) <- (-1x2x96xf32, 96xf32)
        add_0 = paddle._C_ops.add(matmul_1, parameter_0)
        return add_0
