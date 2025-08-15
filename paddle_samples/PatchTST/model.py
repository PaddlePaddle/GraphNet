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
        # pd_op.matmul: (-1x12x256xf32) <- (-1x12x128xf32, 128x256xf32)
        t4 = paddle._C_ops.matmul(input0, t0, False, False)
        del input0, t0

        # pd_op.add: (-1x12x256xf32) <- (-1x12x256xf32, 256xf32)
        t5 = paddle._C_ops.add(t4, t1)
        del t1

        # pd_op.gelu: (-1x12x256xf32) <- (-1x12x256xf32)
        t6 = paddle._C_ops.gelu(t5, False)

        # pd_op.full: (1xf32) <- ()
        t7 = paddle._C_ops.full(
            [1], float("0.2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (-1x12x256xf32, -1x12x256xui8) <- (-1x12x256xf32, None, 1xf32)
        t8, t9 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(t6, None, t7, False, "upscale_in_train", 0, False),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del t6

        # pd_op.matmul: (-1x12x128xf32) <- (-1x12x256xf32, 256x128xf32)
        t10 = paddle._C_ops.matmul(t8, t2, False, False)
        del t2

        return t4, t5, t7, t8, t9, t10
