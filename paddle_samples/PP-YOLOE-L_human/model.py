import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input0: paddle.Tensor, input1: paddle.Tensor):
        # pd_op.full_int_array: (1xi64) <- ()
        t0 = [2]

        # pd_op.unsqueeze: (2x-1x1x4xf32) <- (2x-1x4xf32, 1xi64)
        t1 = paddle._C_ops.unsqueeze(input0, t0)
        del input0

        # pd_op.full_int_array: (1xi64) <- ()
        t2 = [1]

        # pd_op.unsqueeze: (2x1x8400x4xf32) <- (2x8400x4xf32, 1xi64)
        t3 = paddle._C_ops.unsqueeze(input1, t2)
        del input1, t2

        # pd_op.full_int_array: (1xi64) <- ()
        t4 = [0]

        # pd_op.slice: (2x-1x1x2xf32) <- (2x-1x1x4xf32, 1xi64, 1xi64)
        t5 = paddle._C_ops.slice(t1, [3], t4, t0, [1], [])

        # pd_op.full_int_array: (1xi64) <- ()
        t6 = [2147483647]

        # pd_op.slice: (2x-1x1x2xf32) <- (2x-1x1x4xf32, 1xi64, 1xi64)
        t7 = paddle._C_ops.slice(t1, [3], t0, t6, [1], [])
        del t1

        # pd_op.slice: (2x1x8400x2xf32) <- (2x1x8400x4xf32, 1xi64, 1xi64)
        t8 = paddle._C_ops.slice(t3, [3], t4, t0, [1], [])
        del t4

        # pd_op.slice: (2x1x8400x2xf32) <- (2x1x8400x4xf32, 1xi64, 1xi64)
        t9 = paddle._C_ops.slice(t3, [3], t0, t6, [1], [])
        del t0, t6, t3

        # pd_op.maximum: (2x-1x8400x2xf32) <- (2x-1x1x2xf32, 2x1x8400x2xf32)
        t10 = paddle._C_ops.maximum(t5, t8)

        # pd_op.minimum: (2x-1x8400x2xf32) <- (2x-1x1x2xf32, 2x1x8400x2xf32)
        t11 = paddle._C_ops.minimum(t7, t9)

        # pd_op.subtract: (2x-1x8400x2xf32) <- (2x-1x8400x2xf32, 2x-1x8400x2xf32)
        t12 = paddle._C_ops.subtract(t11, t10)
        del t10, t11

        # pd_op.full: (1xf32) <- ()
        t13 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        t14 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (2x-1x8400x2xf32) <- (2x-1x8400x2xf32, 1xf32, 1xf32)
        t15 = paddle._C_ops.clip(t12, t13, t14)
        del t12

        # pd_op.full_int_array: (1xi64) <- ()
        t16 = [-1]

        # pd_op.prod: (2x-1x8400xf32) <- (2x-1x8400x2xf32, 1xi64)
        t17 = paddle._C_ops.prod(t15, t16, False, False)
        del t15

        # pd_op.subtract: (2x-1x1x2xf32) <- (2x-1x1x2xf32, 2x-1x1x2xf32)
        t18 = paddle._C_ops.subtract(t7, t5)
        del t5, t7

        # pd_op.clip: (2x-1x1x2xf32) <- (2x-1x1x2xf32, 1xf32, 1xf32)
        t19 = paddle._C_ops.clip(t18, t13, t14)
        del t18

        # pd_op.prod: (2x-1x1xf32) <- (2x-1x1x2xf32, 1xi64)
        t20 = paddle._C_ops.prod(t19, t16, False, False)
        del t19

        # pd_op.subtract: (2x1x8400x2xf32) <- (2x1x8400x2xf32, 2x1x8400x2xf32)
        t21 = paddle._C_ops.subtract(t9, t8)
        del t8, t9

        # pd_op.clip: (2x1x8400x2xf32) <- (2x1x8400x2xf32, 1xf32, 1xf32)
        t22 = paddle._C_ops.clip(t21, t13, t14)
        del t13, t14, t21

        # pd_op.prod: (2x1x8400xf32) <- (2x1x8400x2xf32, 1xi64)
        t23 = paddle._C_ops.prod(t22, t16, False, False)
        del t22, t16

        # pd_op.add: (2x-1x8400xf32) <- (2x-1x1xf32, 2x1x8400xf32)
        t24 = paddle._C_ops.add(t20, t23)
        del t20, t23

        # pd_op.subtract: (2x-1x8400xf32) <- (2x-1x8400xf32, 2x-1x8400xf32)
        t25 = paddle._C_ops.subtract(t24, t17)
        del t24

        # pd_op.full: (1xf32) <- ()
        t26 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (2x-1x8400xf32) <- (2x-1x8400xf32, 1xf32)
        t27 = paddle._C_ops.scale(t25, t26, float("1e-09"), True)
        del t26, t25

        return t17, t27
