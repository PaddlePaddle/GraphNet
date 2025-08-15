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
        t3: paddle.Tensor,
        t4: paddle.Tensor,
        t5: paddle.Tensor,
    ):
        # pd_op.conv2d: (1x2x-1x-1xf32) <- (1x4x-1x-1xf32, 2x4x3x3xf32)
        t6 = paddle._C_ops.conv2d(
            input0, t0, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.batch_norm_: (1x2x-1x-1xf32, 2xf32, 2xf32, 2xf32, 2xf32, -1xui8) <- (1x2x-1x-1xf32, 2xf32, 2xf32, 2xf32, 2xf32)
        t7, t8, t9, t10, t11, t12 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t6,
                t1,
                t2,
                t3,
                t4,
                True,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del t6, t4, t3, t2, t1

        # pd_op.relu: (1x2x-1x-1xf32) <- (1x2x-1x-1xf32)
        t13 = paddle._C_ops.relu(t7)
        del t7

        # pd_op.conv2d: (1x1x-1x-1xf32) <- (1x2x-1x-1xf32, 1x2x3x3xf32)
        t14 = paddle._C_ops.conv2d(
            t13, t5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t5, t13

        return t14
