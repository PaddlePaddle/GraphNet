import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, parameter_0, parameter_1, data_0, data_1, data_2, data_3, data_4):
        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (4x100x256xf32, 4x100x256xui8) <- (4x100x256xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                data_0, None, full_0, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.add: (4x100x256xf32) <- (4x100x256xf32, 4x100x256xf32)
        add_0 = paddle._C_ops.add(data_1, dropout_0)

        # pd_op.layer_norm: (4x100x256xf32, 4x100xf32, 4x100xf32) <- (4x100x256xf32, 256xf32, 256xf32)
        layer_norm_2, layer_norm_0, layer_norm_1 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_0, parameter_1, parameter_0, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.add: (4x100x256xf32) <- (4x100x256xf32, 4x100x256xf32)
        add_1 = paddle._C_ops.add(layer_norm_2, data_2)

        # pd_op.add: (4x-1x256xf32) <- (4x-1x256xf32, 4x-1x256xf32)
        add_2 = paddle._C_ops.add(data_3, data_4)
        return (
            full_0,
            dropout_0,
            dropout_1,
            add_0,
            layer_norm_0,
            layer_norm_1,
            add_1,
            add_2,
            layer_norm_2,
        )
