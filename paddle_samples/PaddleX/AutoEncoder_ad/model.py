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
    ):
        # pd_op.transpose: (-1x2x96xf32) <- (-1x96x2xf32)
        transpose_2 = paddle._C_ops.transpose(data_0, [0, 2, 1])
        del data_0

        # pd_op.matmul: (-1x2x32xf32) <- (-1x2x96xf32, 96x32xf32)
        matmul_0 = paddle._C_ops.matmul(transpose_2, parameter_7, False, False)
        del parameter_7

        # pd_op.add: (-1x2x32xf32) <- (-1x2x32xf32, 32xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_6)
        del matmul_0, parameter_6

        # pd_op.relu: (-1x2x32xf32) <- (-1x2x32xf32)
        relu_0 = paddle._C_ops.relu(add_0)
        del add_0

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
        del relu_0

        # pd_op.matmul: (-1x2x16xf32) <- (-1x2x32xf32, 32x16xf32)
        matmul_1 = paddle._C_ops.matmul(dropout_0, parameter_5, False, False)
        del dropout_0, parameter_5

        # pd_op.add: (-1x2x16xf32) <- (-1x2x16xf32, 16xf32)
        add_1 = paddle._C_ops.add(matmul_1, parameter_4)
        del matmul_1, parameter_4

        # pd_op.matmul: (-1x2x32xf32) <- (-1x2x16xf32, 16x32xf32)
        matmul_2 = paddle._C_ops.matmul(add_1, parameter_3, False, False)
        del add_1, parameter_3

        # pd_op.add: (-1x2x32xf32) <- (-1x2x32xf32, 32xf32)
        add_2 = paddle._C_ops.add(matmul_2, parameter_2)
        del matmul_2, parameter_2

        # pd_op.relu: (-1x2x32xf32) <- (-1x2x32xf32)
        relu_1 = paddle._C_ops.relu(add_2)
        del add_2

        # pd_op.dropout: (-1x2x32xf32, -1x2x32xui8) <- (-1x2x32xf32, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                relu_1, None, full_0, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del full_0, relu_1

        # pd_op.matmul: (-1x2x96xf32) <- (-1x2x32xf32, 32x96xf32)
        matmul_3 = paddle._C_ops.matmul(dropout_2, parameter_1, False, False)
        del dropout_2, parameter_1

        # pd_op.add: (-1x2x96xf32) <- (-1x2x96xf32, 96xf32)
        add_3 = paddle._C_ops.add(matmul_3, parameter_0)
        del matmul_3, parameter_0

        # pd_op.transpose: (-1x96x2xf32) <- (-1x2x96xf32)
        transpose_0 = paddle._C_ops.transpose(add_3, [0, 2, 1])
        del add_3

        # pd_op.transpose: (-1x96x2xf32) <- (-1x2x96xf32)
        transpose_1 = paddle._C_ops.transpose(transpose_2, [0, 2, 1])
        del transpose_2

        return transpose_0, transpose_1
