import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input0: paddle.Tensor, input1: paddle.Tensor):
        # pd_op.shape64: (4xi64) <- (2x160x-1x-1xf32)
        t0 = paddle._C_ops.shape64(input0)
        del input0

        # pd_op.full_int_array: (1xi64) <- ()
        t1 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        t2 = [3]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t3 = paddle._C_ops.slice(t0, [0], t1, t2, [1], [0])
        del t1

        # pd_op.full_int_array: (1xi64) <- ()
        t4 = [4]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t5 = paddle._C_ops.slice(t0, [0], t2, t4, [1], [0])
        del t2, t4, t0

        # pd_op.cast: (xf32) <- (xi64)
        t6 = paddle._C_ops.cast(input1, paddle.float32)
        del input1

        # pd_op.full: (1xf32) <- ()
        t7 = paddle._C_ops.full([1], float("5"), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        t8 = paddle._C_ops.scale(t6, t7, float("0"), True)
        del t7

        # pd_op.full: (1xf32) <- ()
        t9 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        t10 = paddle._C_ops.scale(t8, t9, float("0"), True)
        del t9, t8

        # pd_op.full: (1xi64) <- ()
        t11 = paddle._C_ops.full([1], float("0"), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        t12 = paddle._C_ops.full([1], float("1"), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        t13 = paddle.arange(t11, t5, t12, dtype="int64")

        # pd_op.cast: (-1xf32) <- (-1xi64)
        t14 = paddle._C_ops.cast(t13, paddle.float32)
        del t13

        # pd_op.full: (1xf32) <- ()
        t15 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        t16 = paddle._C_ops.scale(t14, t15, float("0.5"), True)
        del t14

        # pd_op.multiply: (-1xf32) <- (-1xf32, xf32)
        t17 = paddle._C_ops.multiply(t16, t6)
        del t16

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        t18 = paddle.arange(t11, t3, t12, dtype="int64")
        del t11, t12

        # pd_op.cast: (-1xf32) <- (-1xi64)
        t19 = paddle._C_ops.cast(t18, paddle.float32)
        del t18

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        t20 = paddle._C_ops.scale(t19, t15, float("0.5"), True)
        del t19, t15

        # pd_op.multiply: (-1xf32) <- (-1xf32, xf32)
        t21 = paddle._C_ops.multiply(t20, t6)
        del t6, t20

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        t22 = [t21, t17]
        del t17, t21

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        t23 = paddle._C_ops.meshgrid(t22)
        del t22

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        (
            t24,
            t25,
        ) = t23
        del t23

        # pd_op.subtract: (-1x-1xf32) <- (-1x-1xf32, xf32)
        t26 = paddle._C_ops.subtract(t25, t10)

        # pd_op.subtract: (-1x-1xf32) <- (-1x-1xf32, xf32)
        t27 = paddle._C_ops.subtract(t24, t10)

        # pd_op.add: (-1x-1xf32) <- (-1x-1xf32, xf32)
        t28 = paddle._C_ops.add(t25, t10)

        # pd_op.add: (-1x-1xf32) <- (-1x-1xf32, xf32)
        t29 = paddle._C_ops.add(t24, t10)

        # builtin.combine: ([-1x-1xf32, -1x-1xf32, -1x-1xf32, -1x-1xf32]) <- (-1x-1xf32, -1x-1xf32, -1x-1xf32, -1x-1xf32)
        t30 = [t26, t27, t28, t29]
        del t28, t29, t26, t27

        # pd_op.stack: (-1x-1x4xf32) <- ([-1x-1xf32, -1x-1xf32, -1x-1xf32, -1x-1xf32])
        t31 = paddle._C_ops.stack(t30, -1)
        del t30

        # builtin.combine: ([-1x-1xf32, -1x-1xf32]) <- (-1x-1xf32, -1x-1xf32)
        t32 = [t25, t24]

        # pd_op.stack: (-1x-1x2xf32) <- ([-1x-1xf32, -1x-1xf32])
        t33 = paddle._C_ops.stack(t32, -1)
        del t32

        # pd_op.full_int_array: (2xi64) <- ()
        t34 = [-1, 4]

        # pd_op.reshape: (-1x4xf32) <- (-1x-1x4xf32, 2xi64)
        t35 = paddle._C_ops.reshape(t31, t34)
        del t34

        # pd_op.full_int_array: (2xi64) <- ()
        t36 = [-1, 2]

        return t3, t5, t10, t24, t25, t31, t33, t35, t36
