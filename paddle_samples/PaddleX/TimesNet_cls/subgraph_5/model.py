import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, parameter_0, parameter_1, data_0, data_1):
        # pd_op.gelu: (11x405x32xf32) <- (11x405x32xf32)
        gelu_0 = paddle._C_ops.gelu(data_0, False)
        del data_0

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (11x405x32xf32, 11x405x32xui8) <- (11x405x32xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                gelu_0, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del full_0, gelu_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [-1]

        # pd_op.unsqueeze: (11x405x1xi32) <- (11x405xi32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(data_1, full_int_array_0)
        del data_1, full_int_array_0

        # pd_op.cast: (11x405x1xf32) <- (11x405x1xi32)
        cast_0 = paddle._C_ops.cast(unsqueeze_0, paddle.float32)
        del unsqueeze_0

        # pd_op.multiply: (11x405x32xf32) <- (11x405x32xf32, 11x405x1xf32)
        multiply_0 = paddle._C_ops.multiply(dropout_0, cast_0)
        del cast_0, dropout_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [11, -1]

        # pd_op.reshape: (11x12960xf32) <- (11x405x32xf32, 2xi64)
        reshape_0 = paddle._C_ops.reshape(multiply_0, full_int_array_1)
        del full_int_array_1, multiply_0

        # pd_op.matmul: (11x2xf32) <- (11x12960xf32, 12960x2xf32)
        matmul_0 = paddle._C_ops.matmul(reshape_0, parameter_1, False, False)
        del parameter_1, reshape_0

        # pd_op.add: (11x2xf32) <- (11x2xf32, 2xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_0)
        del matmul_0, parameter_0

        return add_0
