import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input0: paddle.Tensor,
        input1: paddle.Tensor,
        input2: paddle.Tensor,
        input3: paddle.Tensor,
        input4: paddle.Tensor,
        input5: paddle.Tensor,
        input6: paddle.Tensor,
    ):
        # pd_op.diag: (-1xf32) <- (-1x-1xf32)
        t0 = paddle._C_ops.diag(input0, 0, float("0"))
        del input0

        # pd_op.full: (1xi32) <- ()
        t1 = paddle._C_ops.full([1], float("1"), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32]) <- (-1x4xf32, 1xi32)
        t2 = paddle._C_ops.split_with_num(input1, 4, t1)
        del input1

        # builtin.split: (-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32) <- ([-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32])
        (
            t3,
            t4,
            t5,
            t6,
        ) = t2
        del t2

        # pd_op.split_with_num: ([-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32]) <- (-1x4xf32, 1xi32)
        t7 = paddle._C_ops.split_with_num(input2, 4, t1)
        del input2

        # builtin.split: (-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32) <- ([-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32])
        (
            t8,
            t9,
            t10,
            t11,
        ) = t7
        del t7

        # pd_op.maximum: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        t12 = paddle._C_ops.maximum(t3, t8)

        # pd_op.maximum: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        t13 = paddle._C_ops.maximum(t4, t9)

        # pd_op.minimum: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        t14 = paddle._C_ops.minimum(t5, t10)

        # pd_op.minimum: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        t15 = paddle._C_ops.minimum(t6, t11)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        t16 = paddle._C_ops.subtract(t14, t12)

        # pd_op.full: (1xf32) <- ()
        t17 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        t18 = t17

        # pd_op.assign: (1xf32) <- (1xf32)
        t19 = t17

        # pd_op.full: (1xf32) <- ()
        t20 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        t21 = t20

        # pd_op.assign: (1xf32) <- (1xf32)
        t22 = t20

        # pd_op.clip: (-1x1xf32) <- (-1x1xf32, 1xf32, 1xf32)
        t23 = paddle._C_ops.clip(t16, t17, t20)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        t24 = paddle._C_ops.subtract(t15, t13)

        # pd_op.clip: (-1x1xf32) <- (-1x1xf32, 1xf32, 1xf32)
        t25 = paddle._C_ops.clip(t24, t17, t20)

        # pd_op.multiply: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        t26 = paddle._C_ops.multiply(t23, t25)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        t27 = paddle._C_ops.subtract(t5, t3)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        t28 = paddle._C_ops.subtract(t6, t4)

        # pd_op.multiply: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        t29 = paddle._C_ops.multiply(t27, t28)

        # pd_op.clip: (-1x1xf32) <- (-1x1xf32, 1xf32, 1xf32)
        t30 = paddle._C_ops.clip(t29, t17, t20)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        t31 = paddle._C_ops.subtract(t10, t8)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        t32 = paddle._C_ops.subtract(t11, t9)

        # pd_op.multiply: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        t33 = paddle._C_ops.multiply(t31, t32)
        del t31, t32

        # pd_op.clip: (-1x1xf32) <- (-1x1xf32, 1xf32, 1xf32)
        t34 = paddle._C_ops.clip(t33, t17, t20)
        del t33

        # pd_op.add: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        t35 = paddle._C_ops.add(t30, t34)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        t36 = paddle._C_ops.subtract(t35, t26)

        # pd_op.full: (1xf32) <- ()
        t37 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        t38 = t37

        # pd_op.assign: (1xf32) <- (1xf32)
        t39 = t37

        # pd_op.assign: (1xf32) <- (1xf32)
        t40 = t37

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        t41 = paddle._C_ops.scale(t36, t37, float("1e-09"), True)
        del t36

        # pd_op.divide: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        t42 = paddle._C_ops.divide(t26, t41)

        # pd_op.multiply: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        t43 = paddle._C_ops.multiply(t42, t42)

        # pd_op.full: (1xf32) <- ()
        t44 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        t45 = paddle._C_ops.scale(t43, t44, float("1"), True)
        del t43

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        t46 = paddle._C_ops.scale(t45, t37, float("0"), True)
        del t45

        # pd_op.full_int_array: (0xi64) <- ()
        t47 = []

        # pd_op.assign: (0xi64) <- (0xi64)
        t48 = t47

        # pd_op.sum: (xf32) <- (-1x1xf32, 0xi64)
        t49 = paddle._C_ops.sum(t46, t47, None, False)

        # pd_op.divide: (xf32) <- (xf32, xf32)
        t50 = paddle._C_ops.divide(t49, input3)

        # pd_op.full_int_array: (1xi64) <- ()
        t51 = [-1]

        # pd_op.unsqueeze: (8x6804x1xb) <- (8x6804xb, 1xi64)
        t52 = paddle._C_ops.unsqueeze(input4, t51)

        # pd_op.full_int_array: (3xi64) <- ()
        t53 = [1, 1, 4]

        # pd_op.tile: (8x6804x4xb) <- (8x6804x1xb, 3xi64)
        t54 = paddle._C_ops.tile(t52, t53)
        del t53, t52

        # pd_op.masked_select: (-1xf32) <- (8x6804x4xf32, 8x6804x4xb)
        t55 = paddle._C_ops.masked_select(input5, t54)
        del input5

        # pd_op.full_int_array: (2xi64) <- ()
        t56 = [-1, 4]

        # pd_op.reshape: (-1x4xf32) <- (-1xf32, 2xi64)
        t57 = paddle._C_ops.reshape(t55, t56)
        del t56

        # pd_op.masked_select: (-1xi64) <- (8x6804xi64, 8x6804xb)
        t58 = paddle._C_ops.masked_select(input6, input4)
        del input4, input6

        # pd_op.full: (1xi32) <- ()
        t59 = paddle._C_ops.full([1], float("5"), paddle.int32, paddle.core.CPUPlace())

        # pd_op.one_hot: (-1x5xf32) <- (-1xi64, 1xi32)
        t60 = paddle._C_ops.one_hot(t58 % paddle.cast(t59, t58.dtype), t59)
        del t59, t58

        # pd_op.full_int_array: (1xi64) <- ()
        t61 = [0]

        # pd_op.slice: (-1x4xf32) <- (-1x5xf32, 1xi64, 1xi64)
        t62 = paddle._C_ops.slice(t60, [1], t61, t51, [1], [])
        del t61, t60

        return (
            t0,
            t1,
            t3,
            t4,
            t5,
            t6,
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
            t27,
            t28,
            t29,
            t30,
            t34,
            t35,
            t37,
            t38,
            t39,
            t40,
            t41,
            t42,
            t44,
            t46,
            t47,
            t48,
            t49,
            t50,
            t51,
            t54,
            t55,
            t57,
            t62,
        )
