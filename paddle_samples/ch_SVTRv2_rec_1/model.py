import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, parameter_0, parameter_1, data_0, data_1, data_2):
        # pd_op.add: (-1x-1x128xf32) <- (-1x-1x128xf32, -1x-1x128xf32)
        add_0 = paddle._C_ops.add(data_1, data_2)

        # pd_op.layer_norm: (-1x-1x128xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x128xf32, 128xf32, 128xf32)
        layer_norm_2, layer_norm_0, layer_norm_1 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_0, parameter_1, parameter_0, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.transpose: (-1x128x-1xf32) <- (-1x-1x128xf32)
        transpose_0 = paddle._C_ops.transpose(layer_norm_2, [0, 2, 1])

        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("128"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_2 = paddle._C_ops.full(
            [], float("80"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_0 = [full_0, full_1, data_0, full_2]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)

        # pd_op.reshape: (-1x128x-1x80xf32) <- (-1x128x-1xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(transpose_0, stack_0)
        return add_0, layer_norm_0, layer_norm_1, transpose_0, reshape_0
