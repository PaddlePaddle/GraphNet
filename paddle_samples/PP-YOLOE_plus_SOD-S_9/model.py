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
        data_1,
    ):
        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_1 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_0 = full_0

        # pd_op.dropout: (2x-1x512xf32, 2x-1x512xui8) <- (2x-1x512xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                data_0, None, full_0, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.add: (2x-1x512xf32) <- (2x-1x512xf32, 2x-1x512xf32)
        add_0 = paddle._C_ops.add(data_1, dropout_0)

        # pd_op.layer_norm: (2x-1x512xf32, 2x-1xf32, 2x-1xf32) <- (2x-1x512xf32, 512xf32, 512xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_0, parameter_7, parameter_6, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (2x-1x2048xf32) <- (2x-1x512xf32, 512x2048xf32)
        matmul_0 = paddle._C_ops.matmul(layer_norm_0, parameter_5, False, False)

        # pd_op.add: (2x-1x2048xf32) <- (2x-1x2048xf32, 2048xf32)
        add_1 = paddle._C_ops.add(matmul_0, parameter_4)

        # pd_op.gelu: (2x-1x2048xf32) <- (2x-1x2048xf32)
        gelu_0 = paddle._C_ops.gelu(add_1, False)

        # pd_op.dropout: (2x-1x2048xf32, 2x-1x2048xui8) <- (2x-1x2048xf32, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                gelu_0, None, full_0, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (2x-1x512xf32) <- (2x-1x2048xf32, 2048x512xf32)
        matmul_1 = paddle._C_ops.matmul(dropout_2, parameter_3, False, False)

        # pd_op.add: (2x-1x512xf32) <- (2x-1x512xf32, 512xf32)
        add_3 = paddle._C_ops.add(matmul_1, parameter_2)

        # pd_op.dropout: (2x-1x512xf32, 2x-1x512xui8) <- (2x-1x512xf32, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_3, None, full_0, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.add: (2x-1x512xf32) <- (2x-1x512xf32, 2x-1x512xf32)
        add_2 = paddle._C_ops.add(layer_norm_0, dropout_4)

        # pd_op.layer_norm: (2x-1x512xf32, 2x-1xf32, 2x-1xf32) <- (2x-1x512xf32, 512xf32, 512xf32)
        layer_norm_5, layer_norm_3, layer_norm_4 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_2, parameter_1, parameter_0, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        return (
            full_0,
            dropout_0,
            dropout_1,
            add_0,
            layer_norm_0,
            layer_norm_1,
            layer_norm_2,
            matmul_0,
            add_1,
            assign_0,
            dropout_2,
            dropout_3,
            matmul_1,
            assign_1,
            dropout_4,
            dropout_5,
            add_2,
            layer_norm_3,
            layer_norm_4,
            layer_norm_5,
        )
