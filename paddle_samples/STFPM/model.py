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
        t12: paddle.Tensor,
        t13: paddle.Tensor,
        t14: paddle.Tensor,
        t15: paddle.Tensor,
        t16: paddle.Tensor,
        t17: paddle.Tensor,
        t18: paddle.Tensor,
        t19: paddle.Tensor,
    ):
        # pd_op.conv2d: (1x64x64x64xf32) <- (1x64x64x64xf32, 64x64x3x3xf32)
        t20 = paddle._C_ops.conv2d(
            input0, t0, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t0

        # pd_op.batch_norm_: (1x64x64x64xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (1x64x64x64xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t21, t22, t23, t24, t25, t26 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t20,
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
        del t20, t4, t3, t2, t1

        # pd_op.relu: (1x64x64x64xf32) <- (1x64x64x64xf32)
        t27 = paddle._C_ops.relu(t21)
        del t21

        # pd_op.conv2d: (1x64x64x64xf32) <- (1x64x64x64xf32, 64x64x3x3xf32)
        t28 = paddle._C_ops.conv2d(
            t27, t5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t5, t27

        # pd_op.batch_norm_: (1x64x64x64xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (1x64x64x64xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t29, t30, t31, t32, t33, t34 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t28,
                t6,
                t7,
                t8,
                t9,
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
        del t28, t9, t8, t7, t6

        # pd_op.add: (1x64x64x64xf32) <- (1x64x64x64xf32, 1x64x64x64xf32)
        t35 = paddle._C_ops.add(t29, input0)
        del t29, input0

        # pd_op.relu: (1x64x64x64xf32) <- (1x64x64x64xf32)
        t36 = paddle._C_ops.relu(t35)
        del t35

        # pd_op.conv2d: (1x64x64x64xf32) <- (1x64x64x64xf32, 64x64x3x3xf32)
        t37 = paddle._C_ops.conv2d(
            t36, t10, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t10

        # pd_op.batch_norm_: (1x64x64x64xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (1x64x64x64xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t38, t39, t40, t41, t42, t43 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t37,
                t11,
                t12,
                t13,
                t14,
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
        del t37, t14, t13, t12, t11

        # pd_op.relu: (1x64x64x64xf32) <- (1x64x64x64xf32)
        t44 = paddle._C_ops.relu(t38)
        del t38

        # pd_op.conv2d: (1x64x64x64xf32) <- (1x64x64x64xf32, 64x64x3x3xf32)
        t45 = paddle._C_ops.conv2d(
            t44, t15, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del t15, t44

        # pd_op.batch_norm_: (1x64x64x64xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (1x64x64x64xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        t46, t47, t48, t49, t50, t51 = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                t45,
                t16,
                t17,
                t18,
                t19,
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
        del t45, t19, t18, t17, t16

        # pd_op.add: (1x64x64x64xf32) <- (1x64x64x64xf32, 1x64x64x64xf32)
        t52 = paddle._C_ops.add(t46, t36)
        del t46, t36

        return t52
