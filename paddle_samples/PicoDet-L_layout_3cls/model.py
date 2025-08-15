import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input0: paddle.Tensor, input1: paddle.Tensor):
        # pd_op.full: (1xi32) <- ()
        t0 = paddle._C_ops.full([1], float("1"), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([27x1xf32, 27x1xf32, 27x1xf32, 27x1xf32]) <- (27x4xf32, 1xi32)
        t1 = paddle._C_ops.split_with_num(input0, 4, t0)
        del input0

        # builtin.split: (27x1xf32, 27x1xf32, 27x1xf32, 27x1xf32) <- ([27x1xf32, 27x1xf32, 27x1xf32, 27x1xf32])
        (
            t2,
            t3,
            t4,
            t5,
        ) = t1
        del t1

        # pd_op.split_with_num: ([27x1xf32, 27x1xf32, 27x1xf32, 27x1xf32]) <- (27x4xf32, 1xi32)
        t6 = paddle._C_ops.split_with_num(input1, 4, t0)
        del input1

        # builtin.split: (27x1xf32, 27x1xf32, 27x1xf32, 27x1xf32) <- ([27x1xf32, 27x1xf32, 27x1xf32, 27x1xf32])
        (
            t7,
            t8,
            t9,
            t10,
        ) = t6
        del t6

        # pd_op.maximum: (27x1xf32) <- (27x1xf32, 27x1xf32)
        t11 = paddle._C_ops.maximum(t2, t7)

        # pd_op.maximum: (27x1xf32) <- (27x1xf32, 27x1xf32)
        t12 = paddle._C_ops.maximum(t3, t8)

        # pd_op.minimum: (27x1xf32) <- (27x1xf32, 27x1xf32)
        t13 = paddle._C_ops.minimum(t4, t9)

        # pd_op.minimum: (27x1xf32) <- (27x1xf32, 27x1xf32)
        t14 = paddle._C_ops.minimum(t5, t10)

        # pd_op.subtract: (27x1xf32) <- (27x1xf32, 27x1xf32)
        t15 = paddle._C_ops.subtract(t13, t11)

        # pd_op.full: (1xf32) <- ()
        t16 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        t17 = t16

        # pd_op.full: (1xf32) <- ()
        t18 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        t19 = t18

        # pd_op.clip: (27x1xf32) <- (27x1xf32, 1xf32, 1xf32)
        t20 = paddle._C_ops.clip(t15, t16, t18)

        # pd_op.subtract: (27x1xf32) <- (27x1xf32, 27x1xf32)
        t21 = paddle._C_ops.subtract(t14, t12)

        # pd_op.clip: (27x1xf32) <- (27x1xf32, 1xf32, 1xf32)
        t22 = paddle._C_ops.clip(t21, t16, t18)

        # pd_op.multiply: (27x1xf32) <- (27x1xf32, 27x1xf32)
        t23 = paddle._C_ops.multiply(t20, t22)

        # pd_op.subtract: (27x1xf32) <- (27x1xf32, 27x1xf32)
        t24 = paddle._C_ops.subtract(t4, t2)

        # pd_op.subtract: (27x1xf32) <- (27x1xf32, 27x1xf32)
        t25 = paddle._C_ops.subtract(t5, t3)

        # pd_op.multiply: (27x1xf32) <- (27x1xf32, 27x1xf32)
        t26 = paddle._C_ops.multiply(t24, t25)

        # pd_op.subtract: (27x1xf32) <- (27x1xf32, 27x1xf32)
        t27 = paddle._C_ops.subtract(t9, t7)

        # pd_op.subtract: (27x1xf32) <- (27x1xf32, 27x1xf32)
        t28 = paddle._C_ops.subtract(t10, t8)

        # pd_op.multiply: (27x1xf32) <- (27x1xf32, 27x1xf32)
        t29 = paddle._C_ops.multiply(t27, t28)
        del t27, t28

        # pd_op.add: (27x1xf32) <- (27x1xf32, 27x1xf32)
        t30 = paddle._C_ops.add(t26, t29)

        # pd_op.subtract: (27x1xf32) <- (27x1xf32, 27x1xf32)
        t31 = paddle._C_ops.subtract(t30, t23)

        # pd_op.full: (1xf32) <- ()
        t32 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        t33 = t32

        # pd_op.scale: (27x1xf32) <- (27x1xf32, 1xf32)
        t34 = paddle._C_ops.scale(t31, t32, float("1e-10"), True)
        del t31

        # pd_op.divide: (27x1xf32) <- (27x1xf32, 27x1xf32)
        t35 = paddle._C_ops.divide(t23, t34)

        # pd_op.minimum: (27x1xf32) <- (27x1xf32, 27x1xf32)
        t36 = paddle._C_ops.minimum(t2, t7)

        # pd_op.minimum: (27x1xf32) <- (27x1xf32, 27x1xf32)
        t37 = paddle._C_ops.minimum(t3, t8)

        # pd_op.maximum: (27x1xf32) <- (27x1xf32, 27x1xf32)
        t38 = paddle._C_ops.maximum(t4, t9)

        # pd_op.maximum: (27x1xf32) <- (27x1xf32, 27x1xf32)
        t39 = paddle._C_ops.maximum(t5, t10)

        # pd_op.subtract: (27x1xf32) <- (27x1xf32, 27x1xf32)
        t40 = paddle._C_ops.subtract(t38, t36)

        # pd_op.subtract: (27x1xf32) <- (27x1xf32, 27x1xf32)
        t41 = paddle._C_ops.subtract(t39, t37)

        # pd_op.multiply: (27x1xf32) <- (27x1xf32, 27x1xf32)
        t42 = paddle._C_ops.multiply(t40, t41)

        # pd_op.scale: (27x1xf32) <- (27x1xf32, 1xf32)
        t43 = paddle._C_ops.scale(t42, t32, float("1e-10"), True)
        del t42

        # pd_op.subtract: (27x1xf32) <- (27x1xf32, 27x1xf32)
        t44 = paddle._C_ops.subtract(t43, t34)

        # pd_op.divide: (27x1xf32) <- (27x1xf32, 27x1xf32)
        t45 = paddle._C_ops.divide(t44, t43)

        # pd_op.subtract: (27x1xf32) <- (27x1xf32, 27x1xf32)
        t46 = paddle._C_ops.subtract(t35, t45)

        # pd_op.full: (1xf32) <- ()
        t47 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (27x1xf32) <- (27x1xf32, 1xf32)
        t48 = paddle._C_ops.scale(t46, t47, float("1"), True)
        del t46

        # pd_op.full: (1xf32) <- ()
        t49 = paddle._C_ops.full(
            [1], float("2.5"), paddle.float32, paddle.core.CPUPlace()
        )

        return (
            t0,
            t2,
            t3,
            t4,
            t5,
            t7,
            t8,
            t9,
            t10,
            t11,
            t12,
            t13,
            t14,
            t15,
            t16,
            t17,
            t18,
            t19,
            t20,
            t21,
            t22,
            t23,
            t24,
            t25,
            t26,
            t29,
            t30,
            t32,
            t33,
            t34,
            t35,
            t36,
            t37,
            t38,
            t39,
            t40,
            t41,
            t43,
            t44,
            t45,
            t47,
            t48,
            t49,
        )
