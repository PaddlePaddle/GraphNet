import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input0: paddle.Tensor,
        t0: paddle.Tensor,
        t1: paddle.Tensor,
        t2: paddle.Tensor,
    ):
        t3 = None
        # pd_op.matmul: (16x2x32xf32) <- (16x2x96xf32, 96x32xf32)
        t4 = paddle._C_ops.matmul(input0, t0, False, False)
        del input0, t0

        # pd_op.add: (16x2x32xf32) <- (16x2x32xf32, 32xf32)
        t5 = paddle._C_ops.add(t4, t1)
        del t1

        # pd_op.relu: (16x2x32xf32) <- (16x2x32xf32)
        t6 = paddle._C_ops.relu(t5)
        del t5

        # pd_op.full: (1xf32) <- ()
        t7 = paddle._C_ops.full(
            [1], float("0.2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (16x2x32xf32, 16x2x32xui8) <- (16x2x32xf32, None, 1xf32)
        t8, t9 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(t6, None, t7, False, "upscale_in_train", 0, False),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (16x2x16xf32) <- (16x2x32xf32, 32x16xf32)
        t10 = paddle._C_ops.matmul(t8, t2, False, False)
        del t2

        return t4, t6, t7, t8, t9, t10
