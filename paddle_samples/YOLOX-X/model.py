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
    ):
        # pd_op.full_int_array: (1xi64) <- ()
        t0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        t1 = [1]

        # pd_op.slice: (9261xf32) <- (9261x4xf32, 1xi64, 1xi64)
        t2 = paddle._C_ops.slice(input0, [1], t0, t1, [1], [1])

        # pd_op.unsqueeze: (9261x1xf32) <- (9261xf32, 1xi64)
        t3 = paddle._C_ops.unsqueeze(t2, t1)
        del t2

        # pd_op.full_int_array: (2xi64) <- ()
        t4 = [1, 5]

        # pd_op.tile: (9261x5xf32) <- (9261x1xf32, 2xi64)
        t5 = paddle._C_ops.tile(t3, t4)
        del t3

        # pd_op.full_int_array: (1xi64) <- ()
        t6 = [2]

        # pd_op.slice: (9261xf32) <- (9261x4xf32, 1xi64, 1xi64)
        t7 = paddle._C_ops.slice(input0, [1], t1, t6, [1], [1])

        # pd_op.unsqueeze: (9261x1xf32) <- (9261xf32, 1xi64)
        t8 = paddle._C_ops.unsqueeze(t7, t1)
        del t7

        # pd_op.tile: (9261x5xf32) <- (9261x1xf32, 2xi64)
        t9 = paddle._C_ops.tile(t8, t4)
        del t8

        # pd_op.full_int_array: (1xi64) <- ()
        t10 = [3]

        # pd_op.slice: (9261xf32) <- (9261x4xf32, 1xi64, 1xi64)
        t11 = paddle._C_ops.slice(input0, [1], t6, t10, [1], [1])

        # pd_op.unsqueeze: (9261x1xf32) <- (9261xf32, 1xi64)
        t12 = paddle._C_ops.unsqueeze(t11, t1)
        del t11

        # pd_op.tile: (9261x5xf32) <- (9261x1xf32, 2xi64)
        t13 = paddle._C_ops.tile(t12, t4)
        del t12

        # pd_op.full_int_array: (1xi64) <- ()
        t14 = [4]

        # pd_op.slice: (9261xf32) <- (9261x4xf32, 1xi64, 1xi64)
        t15 = paddle._C_ops.slice(input0, [1], t10, t14, [1], [1])
        del input0

        # pd_op.unsqueeze: (9261x1xf32) <- (9261xf32, 1xi64)
        t16 = paddle._C_ops.unsqueeze(t15, t1)
        del t15

        # pd_op.tile: (9261x5xf32) <- (9261x1xf32, 2xi64)
        t17 = paddle._C_ops.tile(t16, t4)
        del t4, t16

        # pd_op.slice: (5xf32) <- (5x4xf32, 1xi64, 1xi64)
        t18 = paddle._C_ops.slice(input1, [1], t0, t1, [1], [1])

        # pd_op.subtract: (9261x5xf32) <- (9261x5xf32, 5xf32)
        t19 = paddle._C_ops.subtract(t5, t18)

        # pd_op.slice: (5xf32) <- (5x4xf32, 1xi64, 1xi64)
        t20 = paddle._C_ops.slice(input1, [1], t1, t6, [1], [1])

        # pd_op.subtract: (9261x5xf32) <- (9261x5xf32, 5xf32)
        t21 = paddle._C_ops.subtract(t9, t20)

        # pd_op.slice: (5xf32) <- (5x4xf32, 1xi64, 1xi64)
        t22 = paddle._C_ops.slice(input1, [1], t6, t10, [1], [1])
        del t6

        # pd_op.subtract: (9261x5xf32) <- (5xf32, 9261x5xf32)
        t23 = paddle._C_ops.subtract(t22, t5)

        # pd_op.slice: (5xf32) <- (5x4xf32, 1xi64, 1xi64)
        t24 = paddle._C_ops.slice(input1, [1], t10, t14, [1], [1])
        del input1, t10, t14

        # pd_op.subtract: (9261x5xf32) <- (5xf32, 9261x5xf32)
        t25 = paddle._C_ops.subtract(t24, t9)

        # builtin.combine: ([9261x5xf32, 9261x5xf32, 9261x5xf32, 9261x5xf32]) <- (9261x5xf32, 9261x5xf32, 9261x5xf32, 9261x5xf32)
        t26 = [t19, t21, t23, t25]
        del t19, t21, t23, t25

        # pd_op.stack: (9261x4x5xf32) <- ([9261x5xf32, 9261x5xf32, 9261x5xf32, 9261x5xf32])
        t27 = paddle._C_ops.stack(t26, 1)
        del t26

        # pd_op.min: (9261x5xf32) <- (9261x4x5xf32, 1xi64)
        t28 = paddle._C_ops.min(t27, t1, False)
        del t27

        # pd_op.full: (xf32) <- ()
        t29 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (9261x5xb) <- (9261x5xf32, xf32)
        t30 = paddle._C_ops.greater_than(t28, t29)
        del t28

        # pd_op.sum: (9261xi64) <- (9261x5xb, 1xi64)
        t31 = paddle._C_ops.sum(t30, t1, None, False)

        # pd_op.full: (xi64) <- ()
        t32 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (9261xb) <- (9261xi64, xi64)
        t33 = paddle._C_ops.greater_than(t31, t32)
        del t31

        # pd_op.add: (5xf32) <- (5xf32, 5xf32)
        t34 = paddle._C_ops.add(t18, t22)
        del t18, t22

        # pd_op.full: (1xf32) <- ()
        t35 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (5xf32) <- (5xf32, 1xf32)
        t36 = paddle._C_ops.scale(t34, t35, float("0"), True)
        del t34

        # pd_op.add: (5xf32) <- (5xf32, 5xf32)
        t37 = paddle._C_ops.add(t20, t24)
        del t20, t24

        # pd_op.scale: (5xf32) <- (5xf32, 1xf32)
        t38 = paddle._C_ops.scale(t37, t35, float("0"), True)
        del t37, t35

        # pd_op.full: (1xf32) <- ()
        t39 = paddle._C_ops.full(
            [1], float("2.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (9261x5xf32) <- (9261x5xf32, 1xf32)
        t40 = paddle._C_ops.scale(t13, t39, float("0"), True)
        del t13

        # pd_op.subtract: (9261x5xf32) <- (5xf32, 9261x5xf32)
        t41 = paddle._C_ops.subtract(t36, t40)

        # pd_op.scale: (9261x5xf32) <- (9261x5xf32, 1xf32)
        t42 = paddle._C_ops.scale(t17, t39, float("0"), True)
        del t39, t17

        # pd_op.subtract: (9261x5xf32) <- (5xf32, 9261x5xf32)
        t43 = paddle._C_ops.subtract(t38, t42)

        # pd_op.add: (9261x5xf32) <- (5xf32, 9261x5xf32)
        t44 = paddle._C_ops.add(t36, t40)
        del t36, t40

        # pd_op.add: (9261x5xf32) <- (5xf32, 9261x5xf32)
        t45 = paddle._C_ops.add(t38, t42)
        del t38, t42

        # pd_op.subtract: (9261x5xf32) <- (9261x5xf32, 9261x5xf32)
        t46 = paddle._C_ops.subtract(t5, t41)
        del t41

        # pd_op.subtract: (9261x5xf32) <- (9261x5xf32, 9261x5xf32)
        t47 = paddle._C_ops.subtract(t9, t43)
        del t43

        # pd_op.subtract: (9261x5xf32) <- (9261x5xf32, 9261x5xf32)
        t48 = paddle._C_ops.subtract(t44, t5)
        del t44, t5

        # pd_op.subtract: (9261x5xf32) <- (9261x5xf32, 9261x5xf32)
        t49 = paddle._C_ops.subtract(t45, t9)
        del t45, t9

        # builtin.combine: ([9261x5xf32, 9261x5xf32, 9261x5xf32, 9261x5xf32]) <- (9261x5xf32, 9261x5xf32, 9261x5xf32, 9261x5xf32)
        t50 = [t46, t47, t48, t49]
        del t46, t47, t48, t49

        # pd_op.stack: (9261x4x5xf32) <- ([9261x5xf32, 9261x5xf32, 9261x5xf32, 9261x5xf32])
        t51 = paddle._C_ops.stack(t50, 1)
        del t50

        # pd_op.min: (9261x5xf32) <- (9261x4x5xf32, 1xi64)
        t52 = paddle._C_ops.min(t51, t1, False)
        del t51

        # pd_op.greater_than: (9261x5xb) <- (9261x5xf32, xf32)
        t53 = paddle._C_ops.greater_than(t52, t29)
        del t29, t52

        # pd_op.sum: (9261xi64) <- (9261x5xb, 1xi64)
        t54 = paddle._C_ops.sum(t53, t1, None, False)

        # pd_op.greater_than: (9261xb) <- (9261xi64, xi64)
        t55 = paddle._C_ops.greater_than(t54, t32)
        del t32, t54

        # pd_op.logical_or: (9261xb) <- (9261xb, 9261xb)
        t56 = paddle._C_ops.logical_or(t33, t55)
        del t33, t55

        # pd_op.nonzero: (-1x1xi64) <- (9261xb)
        t57 = paddle._C_ops.nonzero(t56)

        # pd_op.squeeze: (-1xi64) <- (-1x1xi64, 1xi64)
        t58 = paddle._C_ops.squeeze(t57, t1)
        del t57

        # pd_op.cast: (9261x5xi64) <- (9261x5xb)
        t59 = paddle._C_ops.cast(t30, paddle.int64)
        del t30

        # pd_op.full: (1xi32) <- ()
        t60 = paddle._C_ops.full([1], float("0"), paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (-1x5xi64) <- (9261x5xi64, -1xi64, 1xi32)
        t61 = paddle._C_ops.gather(t59, t58, t60)
        del t59

        # pd_op.cast: (-1x5xb) <- (-1x5xi64)
        t62 = paddle._C_ops.cast(t61, paddle.bool)
        del t61

        # pd_op.cast: (9261x5xi64) <- (9261x5xb)
        t63 = paddle._C_ops.cast(t53, paddle.int64)
        del t53

        # pd_op.gather: (-1x5xi64) <- (9261x5xi64, -1xi64, 1xi32)
        t64 = paddle._C_ops.gather(t63, t58, t60)
        del t63, t60

        # pd_op.cast: (-1x5xb) <- (-1x5xi64)
        t65 = paddle._C_ops.cast(t64, paddle.bool)
        del t64

        # pd_op.logical_and: (-1x5xb) <- (-1x5xb, -1x5xb)
        t66 = paddle._C_ops.logical_and(t62, t65)
        del t62, t65

        # pd_op.full_int_array: (1xi64) <- ()
        t67 = [-1]

        # pd_op.unsqueeze: (-1x1xi64) <- (-1xi64, 1xi64)
        t68 = paddle._C_ops.unsqueeze(t58, t67)
        del t67, t58

        # pd_op.gather_nd: (-1x4xf32) <- (9261x4xf32, -1x1xi64)
        t69 = paddle._C_ops.gather_nd(input2, t68)
        del input2

        # pd_op.gather_nd: (-1x4xf32) <- (9261x4xf32, -1x1xi64)
        t70 = paddle._C_ops.gather_nd(input3, t68)
        del input3, t68

        # pd_op.shape64: (2xi64) <- (-1x4xf32)
        t71 = paddle._C_ops.shape64(t69)

        return t0, t1, t56, t66, t69, t70, t71
