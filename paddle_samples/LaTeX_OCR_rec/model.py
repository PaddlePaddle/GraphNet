import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input0: paddle.Tensor,
        input1: paddle.Tensor,
        t0: paddle.Tensor,
        t1: paddle.Tensor,
        t2: paddle.Tensor,
        t3: paddle.Tensor,
        t4: paddle.Tensor,
        t5: paddle.Tensor,
    ):
        # pd_op.full: (xi64) <- ()
        t6 = paddle._C_ops.full(
            [], float("11"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.equal: (xb) <- (xi64, xi64)
        t7 = paddle._C_ops.equal(input0, t6)
        del input0, t6

        # pd_op.layer_norm: (20x-1x256xf32, 20x-1xf32, 20x-1xf32) <- (20x-1x256xf32, 256xf32, 256xf32)
        t8, t9, t10 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(input1, t0, t1, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t1, t0

        # pd_op.matmul: (20x-1x2048xf32) <- (20x-1x256xf32, 256x2048xf32)
        t11 = paddle._C_ops.matmul(t8, t2, False, False)
        del t2

        # pd_op.add: (20x-1x2048xf32) <- (20x-1x2048xf32, 2048xf32)
        t12 = paddle._C_ops.add(t11, t3)
        del t3

        # pd_op.full: (1xi32) <- ()
        t13 = paddle._C_ops.full([1], float("2"), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([20x-1x1024xf32, 20x-1x1024xf32]) <- (20x-1x2048xf32, 1xi32)
        t14 = paddle._C_ops.split_with_num(t12, 2, t13)
        del t12

        # builtin.split: (20x-1x1024xf32, 20x-1x1024xf32) <- ([20x-1x1024xf32, 20x-1x1024xf32])
        (
            t15,
            t16,
        ) = t14
        del t14

        # pd_op.gelu: (20x-1x1024xf32) <- (20x-1x1024xf32)
        t17 = paddle._C_ops.gelu(t16, False)

        # pd_op.multiply: (20x-1x1024xf32) <- (20x-1x1024xf32, 20x-1x1024xf32)
        t18 = paddle._C_ops.multiply(t15, t17)

        # pd_op.matmul: (20x-1x256xf32) <- (20x-1x1024xf32, 1024x256xf32)
        t19 = paddle._C_ops.matmul(t18, t4, False, False)
        del t4

        # pd_op.add: (20x-1x256xf32) <- (20x-1x256xf32, 256xf32)
        t20 = paddle._C_ops.add(t19, t5)
        del t5

        return t7, t8, t9, t10, t11, t13, t15, t16, t17, t18, t19, t20
