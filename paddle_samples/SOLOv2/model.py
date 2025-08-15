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
        t0 = None
        # pd_op.full: (1xf32) <- ()
        t1 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.cast: (1x6x200x304xf32) <- (1x6x200x304xui8)
        t2 = paddle._C_ops.cast(input0, paddle.float32)
        del input0

        # pd_op.full_int_array: (3xi64) <- ()
        t3 = [-1, 200, 304]

        # pd_op.reshape: (6x200x304xf32) <- (1x6x200x304xf32, 3xi64)
        t4 = paddle._C_ops.reshape(t2, t3)
        del t2

        # pd_op.full_int_array: (2xi64) <- ()
        t5 = [1, 2]

        # pd_op.sum: (6xf32) <- (6x200x304xf32, 2xi64)
        t6 = paddle._C_ops.sum(t4, t5, None, False)

        # pd_op.full: (xf32) <- ()
        t7 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (6xb) <- (6xf32, xf32)
        t8 = paddle._C_ops.greater_than(t6, t7)
        del t6

        # pd_op.cast: (6xf32) <- (6xb)
        t9 = paddle._C_ops.cast(t8, paddle.float32)
        del t8

        # pd_op.sigmoid: (6x200x304xf32) <- (6x200x304xf32)
        t10 = paddle._C_ops.sigmoid(input1)
        del input1

        # pd_op.full_int_array: (2xi64) <- ()
        t11 = [6, -1]

        # pd_op.reshape: (6x60800xf32) <- (6x200x304xf32, 2xi64)
        t12 = paddle._C_ops.reshape(t10, t11)

        # pd_op.reshape: (6x60800xf32) <- (6x200x304xf32, 2xi64)
        t13 = paddle._C_ops.reshape(t4, t11)
        del t11, t4

        # pd_op.multiply: (6x60800xf32) <- (6x60800xf32, 6x60800xf32)
        t14 = paddle._C_ops.multiply(t12, t13)

        # pd_op.full_int_array: (1xi64) <- ()
        t15 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        t16 = t15

        # pd_op.assign: (1xi64) <- (1xi64)
        t17 = t15

        # pd_op.assign: (1xi64) <- (1xi64)
        t18 = t15

        # pd_op.sum: (6xf32) <- (6x60800xf32, 1xi64)
        t19 = paddle._C_ops.sum(t14, t15, None, False)

        # pd_op.multiply: (6x60800xf32) <- (6x60800xf32, 6x60800xf32)
        t20 = paddle._C_ops.multiply(t12, t12)

        # pd_op.sum: (6xf32) <- (6x60800xf32, 1xi64)
        t21 = paddle._C_ops.sum(t20, t15, None, False)

        # pd_op.full: (1xf32) <- ()
        t22 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        t23 = t22

        # pd_op.scale: (6xf32) <- (6xf32, 1xf32)
        t24 = paddle._C_ops.scale(t21, t22, float("0.001"), True)
        del t21

        # pd_op.multiply: (6x60800xf32) <- (6x60800xf32, 6x60800xf32)
        t25 = paddle._C_ops.multiply(t13, t13)

        # pd_op.sum: (6xf32) <- (6x60800xf32, 1xi64)
        t26 = paddle._C_ops.sum(t25, t15, None, False)
        del t25

        # pd_op.scale: (6xf32) <- (6xf32, 1xf32)
        t27 = paddle._C_ops.scale(t26, t22, float("0.001"), True)
        del t26

        # pd_op.full: (1xf32) <- ()
        t28 = paddle._C_ops.full(
            [1], float("2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        t29 = t28

        # pd_op.scale: (6xf32) <- (6xf32, 1xf32)
        t30 = paddle._C_ops.scale(t19, t28, float("0"), True)
        del t19

        # pd_op.add: (6xf32) <- (6xf32, 6xf32)
        t31 = paddle._C_ops.add(t24, t27)

        # pd_op.divide: (6xf32) <- (6xf32, 6xf32)
        t32 = paddle._C_ops.divide(t30, t31)

        # pd_op.full: (1xf32) <- ()
        t33 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        t34 = t33

        # pd_op.scale: (6xf32) <- (6xf32, 1xf32)
        t35 = paddle._C_ops.scale(t32, t33, float("1"), True)

        # pd_op.multiply: (6xf32) <- (6xf32, 6xf32)
        t36 = paddle._C_ops.multiply(t35, t9)

        # pd_op.full_int_array: (0xi64) <- ()
        t37 = []

        # pd_op.assign: (0xi64) <- (0xi64)
        t38 = t37

        # pd_op.assign: (0xi64) <- (0xi64)
        t39 = t37

        # pd_op.sum: (xf32) <- (6xf32, 0xi64)
        t40 = paddle._C_ops.sum(t9, t37, None, False)

        # pd_op.add: (1xf32) <- (1xf32, xf32)
        t41 = paddle._C_ops.add(t1, t40)
        del t1, t40

        # pd_op.cast: (1x-1x200x304xf32) <- (1x-1x200x304xui8)
        t42 = paddle._C_ops.cast(input2, paddle.float32)
        del input2

        # pd_op.shape64: (3xi64) <- (-1x200x304xf32)
        t43 = paddle._C_ops.shape64(input3)

        # pd_op.full_int_array: (1xi64) <- ()
        t44 = [0]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t45 = paddle._C_ops.slice(t43, [0], t44, t15, [1], [0])
        del t43

        # pd_op.reshape: (-1x200x304xf32) <- (1x-1x200x304xf32, 3xi64)
        t46 = paddle._C_ops.reshape(t42, t3)
        del t42, t3

        # pd_op.sum: (-1xf32) <- (-1x200x304xf32, 2xi64)
        t47 = paddle._C_ops.sum(t46, t5, None, False)
        del t5

        # pd_op.greater_than: (-1xb) <- (-1xf32, xf32)
        t48 = paddle._C_ops.greater_than(t47, t7)
        del t7, t47

        # pd_op.cast: (-1xf32) <- (-1xb)
        t49 = paddle._C_ops.cast(t48, paddle.float32)
        del t48

        # pd_op.sigmoid: (-1x200x304xf32) <- (-1x200x304xf32)
        t50 = paddle._C_ops.sigmoid(input3)
        del input3

        # pd_op.shape64: (3xi64) <- (-1x200x304xf32)
        t51 = paddle._C_ops.shape64(t50)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t52 = paddle._C_ops.slice(t51, [0], t44, t15, [1], [0])
        del t51

        # pd_op.full: (xi64) <- ()
        t53 = paddle._C_ops.full([], float("-1"), paddle.int64, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        t54 = [t52, t53]
        del t52

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        t55 = paddle._C_ops.stack(t54, 0)
        del t54

        # pd_op.reshape: (-1x-1xf32) <- (-1x200x304xf32, 2xi64)
        t56 = paddle._C_ops.reshape(t50, t55)
        del t55

        # pd_op.shape64: (3xi64) <- (-1x200x304xf32)
        t57 = paddle._C_ops.shape64(t46)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t58 = paddle._C_ops.slice(t57, [0], t44, t15, [1], [0])
        del t44, t57

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        t59 = [t58, t53]
        del t53, t58

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        t60 = paddle._C_ops.stack(t59, 0)
        del t59

        # pd_op.reshape: (-1x-1xf32) <- (-1x200x304xf32, 2xi64)
        t61 = paddle._C_ops.reshape(t46, t60)
        del t46, t60

        # pd_op.multiply: (-1x-1xf32) <- (-1x-1xf32, -1x-1xf32)
        t62 = paddle._C_ops.multiply(t56, t61)

        # pd_op.sum: (-1xf32) <- (-1x-1xf32, 1xi64)
        t63 = paddle._C_ops.sum(t62, t15, None, False)

        # pd_op.multiply: (-1x-1xf32) <- (-1x-1xf32, -1x-1xf32)
        t64 = paddle._C_ops.multiply(t56, t56)

        # pd_op.sum: (-1xf32) <- (-1x-1xf32, 1xi64)
        t65 = paddle._C_ops.sum(t64, t15, None, False)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        t66 = paddle._C_ops.scale(t65, t22, float("0.001"), True)
        del t65

        # pd_op.multiply: (-1x-1xf32) <- (-1x-1xf32, -1x-1xf32)
        t67 = paddle._C_ops.multiply(t61, t61)

        # pd_op.sum: (-1xf32) <- (-1x-1xf32, 1xi64)
        t68 = paddle._C_ops.sum(t67, t15, None, False)
        del t67

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        t69 = paddle._C_ops.scale(t68, t22, float("0.001"), True)
        del t68

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        t70 = paddle._C_ops.scale(t63, t28, float("0"), True)
        del t63

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        t71 = paddle._C_ops.add(t66, t69)

        # pd_op.divide: (-1xf32) <- (-1xf32, -1xf32)
        t72 = paddle._C_ops.divide(t70, t71)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        t73 = paddle._C_ops.scale(t72, t33, float("1"), True)

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        t74 = paddle._C_ops.multiply(t73, t49)

        # pd_op.sum: (xf32) <- (-1xf32, 0xi64)
        t75 = paddle._C_ops.sum(t49, t37, None, False)

        # pd_op.add: (1xf32) <- (1xf32, xf32)
        t76 = paddle._C_ops.add(t41, t75)
        del t41, t75

        # pd_op.full: (1xi32) <- ()
        t77 = paddle._C_ops.full([1], float("0"), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([6xf32, -1xf32]) <- (6xf32, -1xf32)
        t78 = [t36, t74]

        # pd_op.concat: (-1xf32) <- ([6xf32, -1xf32], 1xi32)
        t79 = paddle._C_ops.concat(t78, t77)
        del t78

        # pd_op.sum: (xf32) <- (-1xf32, 0xi64)
        t80 = paddle._C_ops.sum(t79, t37, None, False)

        # pd_op.divide: (1xf32) <- (xf32, 1xf32)
        t81 = paddle._C_ops.divide(t80, t76)

        # pd_op.full: (1xf32) <- ()
        t82 = paddle._C_ops.full(
            [1], float("3"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1xf32) <- (1xf32, 1xf32)
        t83 = paddle._C_ops.scale(t81, t82, float("0"), True)

        # pd_op.full: (1xi32) <- ()
        t84 = paddle._C_ops.full([1], float("3"), paddle.int32, paddle.core.CPUPlace())

        # pd_op.one_hot: (3872x3xf32) <- (3872xi64, 1xi32)
        t85 = paddle._C_ops.one_hot(input4 % paddle.cast(t84, input4.dtype), t84)
        del input4, t84

        # pd_op.full_int_array: (1xi64) <- ()
        t86 = [2147483647]

        # pd_op.slice: (3872x2xf32) <- (3872x3xf32, 1xi64, 1xi64)
        t87 = paddle._C_ops.slice(t85, [1], t15, t86, [1], [])
        del t86, t85

        # pd_op.cast: (xf32) <- (xi64)
        t88 = paddle._C_ops.cast(input5, paddle.float32)
        del input5

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        t89 = paddle._C_ops.scale(t88, t22, float("1"), True)
        del t88

        # pd_op.shape64: (2xi64) <- (3872x2xf32)
        t90 = paddle._C_ops.shape64(input6)

        # pd_op.full_with_tensor: (3872x2xf32) <- (1xf32, 2xi64)
        t91 = paddle._C_ops.full_with_tensor(t22, t90, paddle.float32)
        del t90

        # pd_op.sigmoid_cross_entropy_with_logits: (3872x2xf32) <- (3872x2xf32, 3872x2xf32, None)
        t92 = paddle._C_ops.sigmoid_cross_entropy_with_logits(
            input6, t87, None, False, -100
        )

        # pd_op.sigmoid: (3872x2xf32) <- (3872x2xf32)
        t93 = paddle._C_ops.sigmoid(input6)
        del input6

        # pd_op.multiply: (3872x2xf32) <- (3872x2xf32, 3872x2xf32)
        t94 = paddle._C_ops.multiply(t93, t87)

        # pd_op.subtract: (3872x2xf32) <- (3872x2xf32, 3872x2xf32)
        t95 = paddle._C_ops.subtract(t91, t93)

        # pd_op.subtract: (3872x2xf32) <- (3872x2xf32, 3872x2xf32)
        t96 = paddle._C_ops.subtract(t91, t87)

        # pd_op.multiply: (3872x2xf32) <- (3872x2xf32, 3872x2xf32)
        t97 = paddle._C_ops.multiply(t95, t96)

        # pd_op.add: (3872x2xf32) <- (3872x2xf32, 3872x2xf32)
        t98 = paddle._C_ops.add(t94, t97)

        # pd_op.full: (xf32) <- ()
        t99 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        t100 = paddle._C_ops.assign_value_(
            t99,
            [],
            paddle.float32,
            [float("0.25")],
            paddle.framework._current_expected_place(),
        )
        del t99

        # pd_op.multiply: (3872x2xf32) <- (xf32, 3872x2xf32)
        t101 = paddle._C_ops.multiply(t100, t87)

        # pd_op.subtract: (3872x2xf32) <- (3872x2xf32, xf32)
        t102 = paddle._C_ops.subtract(t91, t100)
        del t100

        # pd_op.multiply: (3872x2xf32) <- (3872x2xf32, 3872x2xf32)
        t103 = paddle._C_ops.multiply(t102, t96)
        del t102

        # pd_op.add: (3872x2xf32) <- (3872x2xf32, 3872x2xf32)
        t104 = paddle._C_ops.add(t101, t103)
        del t101, t103

        # pd_op.multiply: (3872x2xf32) <- (3872x2xf32, 3872x2xf32)
        t105 = paddle._C_ops.multiply(t104, t92)

        # pd_op.subtract: (3872x2xf32) <- (3872x2xf32, 3872x2xf32)
        t106 = paddle._C_ops.subtract(t91, t98)

        # pd_op.pow: (3872x2xf32) <- (3872x2xf32)
        t107 = paddle._C_ops.pow(t106, float("2"))

        # pd_op.multiply: (3872x2xf32) <- (3872x2xf32, 3872x2xf32)
        t108 = paddle._C_ops.multiply(t107, t105)

        # pd_op.divide: (3872x2xf32) <- (3872x2xf32, xf32)
        t109 = paddle._C_ops.divide(t108, t89)

        return (
            t9,
            t10,
            t12,
            t13,
            t14,
            t15,
            t16,
            t17,
            t18,
            t20,
            t22,
            t23,
            t24,
            t27,
            t28,
            t29,
            t30,
            t31,
            t32,
            t33,
            t34,
            t35,
            t36,
            t37,
            t38,
            t39,
            t49,
            t50,
            t56,
            t61,
            t62,
            t64,
            t66,
            t69,
            t70,
            t71,
            t72,
            t73,
            t74,
            t76,
            t77,
            t79,
            t80,
            t81,
            t82,
            t83,
            t87,
            t89,
            t91,
            t92,
            t93,
            t94,
            t95,
            t96,
            t97,
            t98,
            t104,
            t105,
            t106,
            t107,
            t108,
            t109,
        )
