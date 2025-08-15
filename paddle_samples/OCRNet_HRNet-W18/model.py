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
        t6: paddle.Tensor,
        t7: paddle.Tensor,
        t8: paddle.Tensor,
        t9: paddle.Tensor,
        t10: paddle.Tensor,
        t11: paddle.Tensor,
    ):
        # pd_op.conv2d: (-1x256x-1x128xf32) <- (-1x-1x-1x128xf32, 256x512x1x1xf32)
        t12 = paddle._C_ops.conv2d(
            input0, t0, [1, 1], [0, 0], "SAME", [1, 1], 1, "NCHW"
        )
        del input0, t0

        # pd_op.full_int_array: (4xi64) <- ()
        t13 = [1, -1, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t14 = paddle._C_ops.reshape(t1, t13)
        del t1

        # pd_op.add: (-1x256x-1x128xf32) <- (-1x256x-1x128xf32, 1x256x1x1xf32)
        t15 = paddle._C_ops.add(t12, t14)
        del t12, t14

        # pd_op.batch_norm_: (-1x256x-1x128xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x128xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t16, t17, t18, t19, t20, t21 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t15,
                t2,
                t3,
                t4,
                t5,
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
        del t15, t5, t4, t3, t2

        # pd_op.relu: (-1x256x-1x128xf32) <- (-1x256x-1x128xf32)
        t22 = paddle._C_ops.relu(t16)
        del t16

        # pd_op.conv2d: (-1x256x-1x128xf32) <- (-1x256x-1x128xf32, 256x256x1x1xf32)
        t23 = paddle._C_ops.conv2d(t22, t6, [1, 1], [0, 0], "SAME", [1, 1], 1, "NCHW")
        del t6, t22

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        t24 = paddle._C_ops.reshape(t7, t13)
        del t13, t7

        # pd_op.add: (-1x256x-1x128xf32) <- (-1x256x-1x128xf32, 1x256x1x1xf32)
        t25 = paddle._C_ops.add(t23, t24)
        del t23, t24

        # pd_op.batch_norm_: (-1x256x-1x128xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x-1x128xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        t26, t27, t28, t29, t30, t31 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t25,
                t8,
                t9,
                t10,
                t11,
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
        del t25, t11, t10, t9, t8

        return t26
