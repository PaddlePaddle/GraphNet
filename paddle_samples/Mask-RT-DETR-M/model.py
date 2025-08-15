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
        input7: paddle.Tensor,
        input8: paddle.Tensor,
        input9: paddle.Tensor,
    ):
        # pd_op.shape64: (3xi64) <- (1x-1x256xf32)
        t20 = paddle._C_ops.shape64(input0)

        # pd_op.full_int_array: (1xi64) <- ()
        t21 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        t22 = [2]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t23 = paddle._C_ops.slice(t20, [0], t21, t22, [1], [0])
        del t21, t20

        # pd_op.full: (1xf32) <- ()
        t24 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.cast: (xf32) <- (xi64)
        t25 = paddle._C_ops.cast(input1, paddle.float32)

        # pd_op.full: (1xf32) <- ()
        t26 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.arange: (-1xf32) <- (1xf32, xf32, 1xf32)
        t27 = paddle.arange(t24, t25, t26, dtype="float32")
        del t25

        # pd_op.cast: (xf32) <- (xi64)
        t28 = paddle._C_ops.cast(input2, paddle.float32)

        # pd_op.arange: (-1xf32) <- (1xf32, xf32, 1xf32)
        t29 = paddle.arange(t24, t28, t26, dtype="float32")
        del t28

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        t30 = [t27, t29]
        del t27, t29

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        t31 = paddle._C_ops.meshgrid(t30)
        del t30

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        (
            t32,
            t33,
        ) = t31
        del t31

        # builtin.combine: ([-1x-1xf32, -1x-1xf32]) <- (-1x-1xf32, -1x-1xf32)
        t34 = [t33, t32]
        del t32, t33

        # pd_op.stack: (-1x-1x2xf32) <- ([-1x-1xf32, -1x-1xf32])
        t35 = paddle._C_ops.stack(t34, -1)
        del t34

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        t36 = [input1, input2]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        t37 = paddle._C_ops.stack(t36, 0)
        del t36

        # pd_op.assign: (2xi64) <- (2xi64)
        t38 = t37
        del t37

        # pd_op.cast: (2xf32) <- (2xi64)
        t39 = paddle._C_ops.cast(t38, paddle.float32)
        del t38

        # pd_op.full_int_array: (1xi64) <- ()
        t40 = [0]

        # pd_op.unsqueeze: (1x-1x-1x2xf32) <- (-1x-1x2xf32, 1xi64)
        t41 = paddle._C_ops.unsqueeze(t35, t40)
        del t35

        # pd_op.scale: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        t42 = paddle._C_ops.scale(t41, t26, float("0.5"), True)
        del t41

        # pd_op.divide: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 2xf32)
        t43 = paddle._C_ops.divide(t42, t39)
        del t39, t42

        # pd_op.full_like: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        t44 = paddle._C_ops.full_like(
            t43, t26, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.full: (1xf32) <- ()
        t45 = paddle._C_ops.full(
            [1], float("0.05"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        t46 = paddle._C_ops.scale(t44, t45, float("0"), True)
        del t44

        # pd_op.scale: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        t47 = paddle._C_ops.scale(t46, t26, float("0"), True)
        del t46

        # pd_op.full: (1xi32) <- ()
        t48 = paddle._C_ops.full([1], float("-1"), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1x-1x-1x2xf32, 1x-1x-1x2xf32]) <- (1x-1x-1x2xf32, 1x-1x-1x2xf32)
        t49 = [t43, t47]
        del t43, t47

        # pd_op.concat: (1x-1x-1x4xf32) <- ([1x-1x-1x2xf32, 1x-1x-1x2xf32], 1xi32)
        t50 = paddle._C_ops.concat(t49, t48)
        del t49

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        t51 = paddle._C_ops.multiply(input1, input2)
        del input1, input2

        # pd_op.full: (xi64) <- ()
        t52 = paddle._C_ops.full([], float("-1"), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (xi64) <- ()
        t53 = paddle._C_ops.full([], float("4"), paddle.int64, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t54 = [t52, t51, t53]
        del t51

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t55 = paddle._C_ops.stack(t54, 0)
        del t54

        # pd_op.reshape: (-1x-1x4xf32) <- (1x-1x-1x4xf32, 3xi64)
        t56 = paddle._C_ops.reshape(t50, t55)
        del t50, t55

        # pd_op.cast: (xf32) <- (xi64)
        t57 = paddle._C_ops.cast(input3, paddle.float32)

        # pd_op.arange: (-1xf32) <- (1xf32, xf32, 1xf32)
        t58 = paddle.arange(t24, t57, t26, dtype="float32")
        del t57

        # pd_op.cast: (xf32) <- (xi64)
        t59 = paddle._C_ops.cast(input4, paddle.float32)

        # pd_op.arange: (-1xf32) <- (1xf32, xf32, 1xf32)
        t60 = paddle.arange(t24, t59, t26, dtype="float32")
        del t59

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        t61 = [t58, t60]
        del t58, t60

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        t62 = paddle._C_ops.meshgrid(t61)
        del t61

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        (
            t63,
            t64,
        ) = t62
        del t62

        # builtin.combine: ([-1x-1xf32, -1x-1xf32]) <- (-1x-1xf32, -1x-1xf32)
        t65 = [t64, t63]
        del t63, t64

        # pd_op.stack: (-1x-1x2xf32) <- ([-1x-1xf32, -1x-1xf32])
        t66 = paddle._C_ops.stack(t65, -1)
        del t65

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        t67 = [input3, input4]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        t68 = paddle._C_ops.stack(t67, 0)
        del t67

        # pd_op.assign: (2xi64) <- (2xi64)
        t69 = t68
        del t68

        # pd_op.cast: (2xf32) <- (2xi64)
        t70 = paddle._C_ops.cast(t69, paddle.float32)
        del t69

        # pd_op.unsqueeze: (1x-1x-1x2xf32) <- (-1x-1x2xf32, 1xi64)
        t71 = paddle._C_ops.unsqueeze(t66, t40)
        del t66

        # pd_op.scale: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        t72 = paddle._C_ops.scale(t71, t26, float("0.5"), True)
        del t71

        # pd_op.divide: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 2xf32)
        t73 = paddle._C_ops.divide(t72, t70)
        del t70, t72

        # pd_op.full_like: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        t74 = paddle._C_ops.full_like(
            t73, t26, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.scale: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        t75 = paddle._C_ops.scale(t74, t45, float("0"), True)
        del t74

        # pd_op.full: (1xf32) <- ()
        t76 = paddle._C_ops.full(
            [1], float("2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        t77 = paddle._C_ops.scale(t75, t76, float("0"), True)
        del t76, t75

        # builtin.combine: ([1x-1x-1x2xf32, 1x-1x-1x2xf32]) <- (1x-1x-1x2xf32, 1x-1x-1x2xf32)
        t78 = [t73, t77]
        del t73, t77

        # pd_op.concat: (1x-1x-1x4xf32) <- ([1x-1x-1x2xf32, 1x-1x-1x2xf32], 1xi32)
        t79 = paddle._C_ops.concat(t78, t48)
        del t78

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        t80 = paddle._C_ops.multiply(input3, input4)
        del input3, input4

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t81 = [t52, t80, t53]
        del t80

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t82 = paddle._C_ops.stack(t81, 0)
        del t81

        # pd_op.reshape: (-1x-1x4xf32) <- (1x-1x-1x4xf32, 3xi64)
        t83 = paddle._C_ops.reshape(t79, t82)
        del t79, t82

        # pd_op.cast: (xf32) <- (xi64)
        t84 = paddle._C_ops.cast(input5, paddle.float32)

        # pd_op.arange: (-1xf32) <- (1xf32, xf32, 1xf32)
        t85 = paddle.arange(t24, t84, t26, dtype="float32")
        del t84

        # pd_op.cast: (xf32) <- (xi64)
        t86 = paddle._C_ops.cast(input6, paddle.float32)

        # pd_op.arange: (-1xf32) <- (1xf32, xf32, 1xf32)
        t87 = paddle.arange(t24, t86, t26, dtype="float32")
        del t86

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        t88 = [t85, t87]
        del t85, t87

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        t89 = paddle._C_ops.meshgrid(t88)
        del t88

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        (
            t90,
            t91,
        ) = t89
        del t89

        # builtin.combine: ([-1x-1xf32, -1x-1xf32]) <- (-1x-1xf32, -1x-1xf32)
        t92 = [t91, t90]
        del t90, t91

        # pd_op.stack: (-1x-1x2xf32) <- ([-1x-1xf32, -1x-1xf32])
        t93 = paddle._C_ops.stack(t92, -1)
        del t92

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        t94 = [input5, input6]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        t95 = paddle._C_ops.stack(t94, 0)
        del t94

        # pd_op.assign: (2xi64) <- (2xi64)
        t96 = t95
        del t95

        # pd_op.cast: (2xf32) <- (2xi64)
        t97 = paddle._C_ops.cast(t96, paddle.float32)
        del t96

        # pd_op.unsqueeze: (1x-1x-1x2xf32) <- (-1x-1x2xf32, 1xi64)
        t98 = paddle._C_ops.unsqueeze(t93, t40)
        del t40, t93

        # pd_op.scale: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        t99 = paddle._C_ops.scale(t98, t26, float("0.5"), True)
        del t98

        # pd_op.divide: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 2xf32)
        t100 = paddle._C_ops.divide(t99, t97)
        del t97, t99

        # pd_op.full_like: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        t101 = paddle._C_ops.full_like(
            t100, t26, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.scale: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        t102 = paddle._C_ops.scale(t101, t45, float("0"), True)
        del t45, t101

        # pd_op.full: (1xf32) <- ()
        t103 = paddle._C_ops.full(
            [1], float("4"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        t104 = paddle._C_ops.scale(t102, t103, float("0"), True)
        del t103, t102

        # builtin.combine: ([1x-1x-1x2xf32, 1x-1x-1x2xf32]) <- (1x-1x-1x2xf32, 1x-1x-1x2xf32)
        t105 = [t100, t104]
        del t100, t104

        # pd_op.concat: (1x-1x-1x4xf32) <- ([1x-1x-1x2xf32, 1x-1x-1x2xf32], 1xi32)
        t106 = paddle._C_ops.concat(t105, t48)
        del t105

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        t107 = paddle._C_ops.multiply(input5, input6)
        del input5, input6

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t108 = [t52, t107, t53]
        del t52, t53, t107

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t109 = paddle._C_ops.stack(t108, 0)
        del t108

        # pd_op.reshape: (-1x-1x4xf32) <- (1x-1x-1x4xf32, 3xi64)
        t110 = paddle._C_ops.reshape(t106, t109)
        del t106, t109

        # pd_op.full: (1xi32) <- ()
        t111 = paddle._C_ops.full([1], float("1"), paddle.int32, paddle.core.CPUPlace())

        # pd_op.assign: (1xi32) <- (1xi32)
        t112 = t111

        # builtin.combine: ([-1x-1x4xf32, -1x-1x4xf32, -1x-1x4xf32]) <- (-1x-1x4xf32, -1x-1x4xf32, -1x-1x4xf32)
        t113 = [t56, t83, t110]
        del t56, t83, t110

        # pd_op.concat: (-1x-1x4xf32) <- ([-1x-1x4xf32, -1x-1x4xf32, -1x-1x4xf32], 1xi32)
        t114 = paddle._C_ops.concat(t113, t111)
        del t113

        # pd_op.full: (xf32) <- ()
        t115 = paddle._C_ops.full(
            [],
            float("0.01"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.greater_than: (-1x-1x4xb) <- (-1x-1x4xf32, xf32)
        t116 = paddle._C_ops.greater_than(t114, t115)
        del t115

        # pd_op.full: (xf32) <- ()
        t117 = paddle._C_ops.full(
            [],
            float("0.99"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.less_than: (-1x-1x4xb) <- (-1x-1x4xf32, xf32)
        t118 = paddle._C_ops.less_than(t114, t117)
        del t117

        # pd_op.multiply: (-1x-1x4xb) <- (-1x-1x4xb, -1x-1x4xb)
        t119 = paddle._C_ops.multiply(t116, t118)
        del t116, t118

        # pd_op.all: (-1x-1x1xb) <- (-1x-1x4xb)
        t120 = paddle._C_ops.all(t119, [-1], True)
        del t119

        # pd_op.full: (1xf32) <- ()
        t121 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x-1x4xf32) <- (-1x-1x4xf32, 1xf32)
        t122 = paddle._C_ops.scale(t114, t121, float("1"), True)

        # pd_op.divide: (-1x-1x4xf32) <- (-1x-1x4xf32, -1x-1x4xf32)
        t123 = paddle._C_ops.divide(t114, t122)
        del t114, t122

        # pd_op.log: (-1x-1x4xf32) <- (-1x-1x4xf32)
        t124 = paddle._C_ops.log(t123)
        del t123

        # pd_op.full: (xf32) <- ()
        t125 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        t126 = paddle._C_ops.assign_value_(
            t125,
            [],
            paddle.float32,
            [float("inf")],
            paddle.framework._current_expected_place(),
        )
        del t125

        # pd_op.full_like: (-1x-1x4xf32) <- (-1x-1x4xf32, 1xf32)
        t127 = paddle._C_ops.full_like(
            t124, t24, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.full_like: (xf32) <- (xf32, 1xf32)
        t128 = paddle._C_ops.full_like(
            t126, t24, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.full_like: (-1x-1x1xb) <- (-1x-1x1xb, 1xf32)
        t129 = paddle._C_ops.full_like(
            t120, t24, paddle.bool, paddle.framework._current_expected_place()
        )

        # pd_op.cast: (-1x-1x1xf32) <- (-1x-1x1xb)
        t130 = paddle._C_ops.cast(t129, paddle.float32)
        del t129

        # pd_op.cast: (-1x-1x1xf32) <- (-1x-1x1xb)
        t131 = paddle._C_ops.cast(t120, paddle.float32)
        del t120

        # pd_op.add: (-1x-1x4xf32) <- (-1x-1x4xf32, xf32)
        t132 = paddle._C_ops.add(t127, t128)
        del t127, t128

        # pd_op.add: (-1x-1x4xf32) <- (-1x-1x4xf32, -1x-1x1xf32)
        t133 = paddle._C_ops.add(t132, t130)
        del t132

        # pd_op.add: (-1x-1x4xf32) <- (-1x-1x4xf32, -1x-1x4xf32)
        t134 = paddle._C_ops.add(t124, t133)
        del t124

        # pd_op.add: (-1x-1x4xf32) <- (xf32, -1x-1x4xf32)
        t135 = paddle._C_ops.add(t126, t133)
        del t126

        # pd_op.add: (-1x-1x4xf32) <- (-1x-1x1xf32, -1x-1x4xf32)
        t136 = paddle._C_ops.add(t131, t133)
        del t133

        # pd_op.cast: (-1x-1x4xb) <- (-1x-1x4xf32)
        t137 = paddle._C_ops.cast(t136, paddle.bool)
        del t136

        # pd_op.where: (-1x-1x4xf32) <- (-1x-1x4xb, -1x-1x4xf32, -1x-1x4xf32)
        t138 = paddle._C_ops.where(t137, t134, t135)
        del t134, t135, t137

        # pd_op.full: (xf32) <- ()
        t139 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        t140 = paddle._C_ops.assign_value_(
            t139,
            [],
            paddle.float32,
            [float("0")],
            paddle.framework._current_expected_place(),
        )
        del t139

        # pd_op.full_like: (1x-1x256xf32) <- (1x-1x256xf32, 1xf32)
        t141 = paddle._C_ops.full_like(
            input0, t24, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.full_like: (xf32) <- (xf32, 1xf32)
        t142 = paddle._C_ops.full_like(
            t140, t24, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.add: (1x-1x256xf32) <- (1x-1x256xf32, xf32)
        t143 = paddle._C_ops.add(t141, t142)
        del t141, t142

        # pd_op.add: (-1x-1x256xf32) <- (1x-1x256xf32, -1x-1x1xf32)
        t144 = paddle._C_ops.add(t143, t130)
        del t143, t130

        # pd_op.add: (-1x-1x256xf32) <- (1x-1x256xf32, -1x-1x256xf32)
        t145 = paddle._C_ops.add(input0, t144)
        del input0

        # pd_op.add: (-1x-1x256xf32) <- (xf32, -1x-1x256xf32)
        t146 = paddle._C_ops.add(t140, t144)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x1xf32, -1x-1x256xf32)
        t147 = paddle._C_ops.add(t131, t144)
        del t131

        # pd_op.cast: (-1x-1x256xb) <- (-1x-1x256xf32)
        t148 = paddle._C_ops.cast(t147, paddle.bool)
        del t147

        # pd_op.where: (-1x-1x256xf32) <- (-1x-1x256xb, -1x-1x256xf32, -1x-1x256xf32)
        t149 = paddle._C_ops.where(t148, t145, t146)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x256xf32, 256x256xf32)
        t150 = paddle._C_ops.matmul(t149, t0, False, False)
        del t0

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        t151 = paddle._C_ops.add(t150, t1)
        del t1

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        t152, t153, t154 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t151, t2, t3, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t3, t2

        # pd_op.matmul: (-1x-1x2xf32) <- (-1x-1x256xf32, 256x2xf32)
        t155 = paddle._C_ops.matmul(t152, t4, False, False)

        # pd_op.add: (-1x-1x2xf32) <- (-1x-1x2xf32, 2xf32)
        t156 = paddle._C_ops.add(t155, t5)
        del t155

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x256xf32, 256x256xf32)
        t157 = paddle._C_ops.matmul(t152, t6, False, False)
        del t6

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        t158 = paddle._C_ops.add(t157, t7)
        del t7

        # pd_op.relu: (-1x-1x256xf32) <- (-1x-1x256xf32)
        t159 = paddle._C_ops.relu(t158)
        del t158

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x256xf32, 256x256xf32)
        t160 = paddle._C_ops.matmul(t159, t8, False, False)
        del t8

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        t161 = paddle._C_ops.add(t160, t9)
        del t9

        # pd_op.relu: (-1x-1x256xf32) <- (-1x-1x256xf32)
        t162 = paddle._C_ops.relu(t161)
        del t161

        # pd_op.matmul: (-1x-1x4xf32) <- (-1x-1x256xf32, 256x4xf32)
        t163 = paddle._C_ops.matmul(t162, t10, False, False)
        del t10

        # pd_op.add: (-1x-1x4xf32) <- (-1x-1x4xf32, 4xf32)
        t164 = paddle._C_ops.add(t163, t11)
        del t11

        # pd_op.add: (-1x-1x4xf32) <- (-1x-1x4xf32, -1x-1x4xf32)
        t165 = paddle._C_ops.add(t164, t138)

        # pd_op.full_int_array: (1xi64) <- ()
        t166 = [-1]

        # pd_op.max: (-1x-1xf32) <- (-1x-1x2xf32, 1xi64)
        t167 = paddle._C_ops.max(t156, t166, False)
        del t156

        # pd_op.full: (1xi32) <- ()
        t168 = paddle._C_ops.full(
            [1], float("300"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.topk: (-1x300xf32, -1x300xi64) <- (-1x-1xf32, 1xi32)
        t169, t170 = (lambda x, f: f(x))(
            paddle._C_ops.topk(t167, t168, 1, True, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del t168, t167

        # pd_op.full: (1xf64) <- ()
        t171 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        t172 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (1xi64) <- (1xf64, 1xf64, 1xf64)
        t173 = paddle.arange(t171, t172, t172, dtype="int64")
        del t171, t172

        # pd_op.unsqueeze: (1x1xi64) <- (1xi64, 1xi64)
        t174 = paddle._C_ops.unsqueeze(t173, t166)
        del t173

        # pd_op.full_int_array: (2xi64) <- ()
        t175 = [1, 300]

        # pd_op.tile: (1x300xi64) <- (1x1xi64, 2xi64)
        t176 = paddle._C_ops.tile(t174, t175)
        del t175, t174

        # builtin.combine: ([1x300xi64, -1x300xi64]) <- (1x300xi64, -1x300xi64)
        t177 = [t176, t170]
        del t176, t170

        # pd_op.stack: (1x300x2xi64) <- ([1x300xi64, -1x300xi64])
        t178 = paddle._C_ops.stack(t177, -1)
        del t177

        # pd_op.gather_nd: (1x300x256xf32) <- (-1x-1x256xf32, 1x300x2xi64)
        t179 = paddle._C_ops.gather_nd(t152, t178)

        # pd_op.gather_nd: (1x300x4xf32) <- (-1x-1x4xf32, 1x300x2xi64)
        t180 = paddle._C_ops.gather_nd(t165, t178)

        # pd_op.layer_norm: (1x300x256xf32, 1x300xf32, 1x300xf32) <- (1x300x256xf32, 256xf32, 256xf32)
        t181, t182, t183 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t179, t12, t13, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (1x300x2xf32) <- (1x300x256xf32, 256x2xf32)
        t184 = paddle._C_ops.matmul(t181, t4, False, False)

        # pd_op.add: (1x300x2xf32) <- (1x300x2xf32, 2xf32)
        t185 = paddle._C_ops.add(t184, t5)

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        t186 = paddle._C_ops.matmul(t181, t14, False, False)

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        t187 = paddle._C_ops.add(t186, t15)

        # pd_op.relu: (1x300x256xf32) <- (1x300x256xf32)
        t188 = paddle._C_ops.relu(t187)
        del t187

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        t189 = paddle._C_ops.matmul(t188, t16, False, False)

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        t190 = paddle._C_ops.add(t189, t17)

        # pd_op.relu: (1x300x256xf32) <- (1x300x256xf32)
        t191 = paddle._C_ops.relu(t190)
        del t190

        # pd_op.matmul: (1x300x128xf32) <- (1x300x256xf32, 256x128xf32)
        t192 = paddle._C_ops.matmul(t191, t18, False, False)

        # pd_op.add: (1x300x128xf32) <- (1x300x128xf32, 128xf32)
        t193 = paddle._C_ops.add(t192, t19)

        # pd_op.shape64: (4xi64) <- (1x128x-1x-1xf32)
        t194 = paddle._C_ops.shape64(input7)

        # pd_op.full_int_array: (1xi64) <- ()
        t195 = [3]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t196 = paddle._C_ops.slice(t194, [0], t22, t195, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        t197 = [4]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t198 = paddle._C_ops.slice(t194, [0], t195, t197, [1], [0])
        del t194

        # pd_op.flatten: (1x128x-1xf32) <- (1x128x-1x-1xf32)
        t199 = paddle._C_ops.flatten(input7, 2, 3)
        del input7

        # pd_op.assign: (1x128x-1xf32) <- (1x128x-1xf32)
        t200 = t199

        # pd_op.bmm: (1x300x-1xf32) <- (1x300x128xf32, 1x128x-1xf32)
        t201 = paddle._C_ops.bmm(t193, t199)

        # pd_op.full: (xi64) <- ()
        t202 = paddle._C_ops.full([], float("1"), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (xi64) <- ()
        t203 = paddle._C_ops.full(
            [], float("300"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t204 = [t202, t203, t196, t198]
        del t203

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t205 = paddle._C_ops.stack(t204, 0)
        del t204

        # pd_op.reshape: (1x300x-1x-1xf32) <- (1x300x-1xf32, 4xi64)
        t206 = paddle._C_ops.reshape(t201, t205)
        del t205

        # pd_op.sigmoid: (1x300x4xf32) <- (1x300x4xf32)
        t207 = paddle._C_ops.sigmoid(t180)
        del t180

        # pd_op.share_data_: (1x300x256xf32) <- (1x300x256xf32)
        input10 = t179.detach()

        # builtin.combine: ([1x100x256xf32, 1x300x256xf32]) <- (1x100x256xf32, 1x300x256xf32)
        t208 = [input8, input10]
        del input8

        # pd_op.concat: (1x400x256xf32) <- ([1x100x256xf32, 1x300x256xf32], 1xi32)
        t209 = paddle._C_ops.concat(t208, t111)
        del t208

        # pd_op.full: (xf32) <- ()
        t210 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (1x300x-1x-1xb) <- (1x300x-1x-1xf32, xf32)
        t211 = paddle._C_ops.greater_than(t206, t210)
        del t210

        # pd_op.shape64: (4xi64) <- (1x300x-1x-1xb)
        t212 = paddle._C_ops.shape64(t211)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t213 = paddle._C_ops.slice(t212, [0], t22, t195, [1], [0])
        del t212

        # pd_op.shape64: (4xi64) <- (1x300x-1x-1xb)
        t214 = paddle._C_ops.shape64(t211)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t215 = paddle._C_ops.slice(t214, [0], t195, t197, [1], [0])
        del t195, t197, t214

        # pd_op.cast: (xf32) <- (xi64)
        t216 = paddle._C_ops.cast(t213, paddle.float32)

        # pd_op.arange: (-1xf32) <- (1xf32, xf32, 1xf32)
        t217 = paddle.arange(t24, t216, t26, dtype="float32")
        del t216

        # pd_op.cast: (xf32) <- (xi64)
        t218 = paddle._C_ops.cast(t215, paddle.float32)

        # pd_op.arange: (-1xf32) <- (1xf32, xf32, 1xf32)
        t219 = paddle.arange(t24, t218, t26, dtype="float32")
        del t218

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        t220 = [t217, t219]
        del t217, t219

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        t221 = paddle._C_ops.meshgrid(t220)
        del t220

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        (
            t222,
            t223,
        ) = t221
        del t221

        # pd_op.cast: (1x300x-1x-1xf32) <- (1x300x-1x-1xb)
        t224 = paddle._C_ops.cast(t211, paddle.float32)

        # pd_op.multiply: (1x300x-1x-1xf32) <- (-1x-1xf32, 1x300x-1x-1xf32)
        t225 = paddle._C_ops.multiply(t223, t224)
        del t224, t223

        # pd_op.flatten: (1x300x-1xf32) <- (1x300x-1x-1xf32)
        t226 = paddle._C_ops.flatten(t225, 2, 3)

        # pd_op.max: (1x300xf32) <- (1x300x-1xf32, 1xi64)
        t227 = paddle._C_ops.max(t226, t166, False)
        del t226

        # pd_op.scale: (1x300xf32) <- (1x300xf32, 1xf32)
        t228 = paddle._C_ops.scale(t227, t26, float("1"), True)
        del t227

        # pd_op.full: (xf32) <- ()
        t229 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        t230 = paddle._C_ops.assign_value_(
            t229,
            [],
            paddle.float32,
            [float("1e+08")],
            paddle.framework._current_expected_place(),
        )
        del t229

        # pd_op.full_like: (1x300x-1x-1xf32) <- (1x300x-1x-1xf32, 1xf32)
        t231 = paddle._C_ops.full_like(
            t225, t24, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.full_like: (xf32) <- (xf32, 1xf32)
        t232 = paddle._C_ops.full_like(
            t230, t24, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.full_like: (1x300x-1x-1xb) <- (1x300x-1x-1xb, 1xf32)
        t233 = paddle._C_ops.full_like(
            t211, t24, paddle.bool, paddle.framework._current_expected_place()
        )

        # pd_op.cast: (1x300x-1x-1xf32) <- (1x300x-1x-1xb)
        t234 = paddle._C_ops.cast(t233, paddle.float32)
        del t233

        # pd_op.cast: (1x300x-1x-1xf32) <- (1x300x-1x-1xb)
        t235 = paddle._C_ops.cast(t211, paddle.float32)

        # pd_op.add: (1x300x-1x-1xf32) <- (1x300x-1x-1xf32, xf32)
        t236 = paddle._C_ops.add(t231, t232)
        del t231, t232

        # pd_op.add: (1x300x-1x-1xf32) <- (1x300x-1x-1xf32, 1x300x-1x-1xf32)
        t237 = paddle._C_ops.add(t236, t234)
        del t236, t234

        # pd_op.add: (1x300x-1x-1xf32) <- (1x300x-1x-1xf32, 1x300x-1x-1xf32)
        t238 = paddle._C_ops.add(t225, t237)
        del t225

        # pd_op.add: (1x300x-1x-1xf32) <- (xf32, 1x300x-1x-1xf32)
        t239 = paddle._C_ops.add(t230, t237)
        del t230

        # pd_op.add: (1x300x-1x-1xf32) <- (1x300x-1x-1xf32, 1x300x-1x-1xf32)
        t240 = paddle._C_ops.add(t235, t237)
        del t237, t235

        # pd_op.cast: (1x300x-1x-1xb) <- (1x300x-1x-1xf32)
        t241 = paddle._C_ops.cast(t240, paddle.bool)
        del t240

        # pd_op.where: (1x300x-1x-1xf32) <- (1x300x-1x-1xb, 1x300x-1x-1xf32, 1x300x-1x-1xf32)
        t242 = paddle._C_ops.where(t241, t238, t239)
        del t238, t239, t241

        # pd_op.flatten: (1x300x-1xf32) <- (1x300x-1x-1xf32)
        t243 = paddle._C_ops.flatten(t242, 2, 3)
        del t242

        # pd_op.min: (1x300xf32) <- (1x300x-1xf32, 1xi64)
        t244 = paddle._C_ops.min(t243, t166, False)
        del t243

        # pd_op.cast: (1x300x-1x-1xf32) <- (1x300x-1x-1xb)
        t245 = paddle._C_ops.cast(t211, paddle.float32)

        # pd_op.multiply: (1x300x-1x-1xf32) <- (-1x-1xf32, 1x300x-1x-1xf32)
        t246 = paddle._C_ops.multiply(t222, t245)
        del t245, t222

        # pd_op.flatten: (1x300x-1xf32) <- (1x300x-1x-1xf32)
        t247 = paddle._C_ops.flatten(t246, 2, 3)

        # pd_op.max: (1x300xf32) <- (1x300x-1xf32, 1xi64)
        t248 = paddle._C_ops.max(t247, t166, False)
        del t247

        # pd_op.scale: (1x300xf32) <- (1x300xf32, 1xf32)
        t249 = paddle._C_ops.scale(t248, t26, float("1"), True)
        del t248

        # pd_op.full: (xf32) <- ()
        t250 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        t251 = paddle._C_ops.assign_value_(
            t250,
            [],
            paddle.float32,
            [float("1e+08")],
            paddle.framework._current_expected_place(),
        )
        del t250

        # pd_op.full_like: (1x300x-1x-1xf32) <- (1x300x-1x-1xf32, 1xf32)
        t252 = paddle._C_ops.full_like(
            t246, t24, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.full_like: (xf32) <- (xf32, 1xf32)
        t253 = paddle._C_ops.full_like(
            t251, t24, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.full_like: (1x300x-1x-1xb) <- (1x300x-1x-1xb, 1xf32)
        t254 = paddle._C_ops.full_like(
            t211, t24, paddle.bool, paddle.framework._current_expected_place()
        )

        # pd_op.cast: (1x300x-1x-1xf32) <- (1x300x-1x-1xb)
        t255 = paddle._C_ops.cast(t254, paddle.float32)
        del t254

        # pd_op.cast: (1x300x-1x-1xf32) <- (1x300x-1x-1xb)
        t256 = paddle._C_ops.cast(t211, paddle.float32)

        # pd_op.add: (1x300x-1x-1xf32) <- (1x300x-1x-1xf32, xf32)
        t257 = paddle._C_ops.add(t252, t253)
        del t252, t253

        # pd_op.add: (1x300x-1x-1xf32) <- (1x300x-1x-1xf32, 1x300x-1x-1xf32)
        t258 = paddle._C_ops.add(t257, t255)
        del t257, t255

        # pd_op.add: (1x300x-1x-1xf32) <- (1x300x-1x-1xf32, 1x300x-1x-1xf32)
        t259 = paddle._C_ops.add(t246, t258)
        del t246

        # pd_op.add: (1x300x-1x-1xf32) <- (xf32, 1x300x-1x-1xf32)
        t260 = paddle._C_ops.add(t251, t258)
        del t251

        # pd_op.add: (1x300x-1x-1xf32) <- (1x300x-1x-1xf32, 1x300x-1x-1xf32)
        t261 = paddle._C_ops.add(t256, t258)
        del t258, t256

        # pd_op.cast: (1x300x-1x-1xb) <- (1x300x-1x-1xf32)
        t262 = paddle._C_ops.cast(t261, paddle.bool)
        del t261

        # pd_op.where: (1x300x-1x-1xf32) <- (1x300x-1x-1xb, 1x300x-1x-1xf32, 1x300x-1x-1xf32)
        t263 = paddle._C_ops.where(t262, t259, t260)
        del t259, t260, t262

        # pd_op.flatten: (1x300x-1xf32) <- (1x300x-1x-1xf32)
        t264 = paddle._C_ops.flatten(t263, 2, 3)
        del t263

        # pd_op.min: (1x300xf32) <- (1x300x-1xf32, 1xi64)
        t265 = paddle._C_ops.min(t264, t166, False)
        del t264, t166

        # builtin.combine: ([1x300xf32, 1x300xf32, 1x300xf32, 1x300xf32]) <- (1x300xf32, 1x300xf32, 1x300xf32, 1x300xf32)
        t266 = [t244, t265, t228, t249]
        del t244, t265, t228, t249

        # pd_op.stack: (1x300x4xf32) <- ([1x300xf32, 1x300xf32, 1x300xf32, 1x300xf32])
        t267 = paddle._C_ops.stack(t266, -1)
        del t266

        # pd_op.any: (1x300xb) <- (1x300x-1x-1xb)
        t268 = paddle._C_ops.any(t211, [2, 3], False)
        del t211

        # pd_op.unsqueeze: (1x300x1xb) <- (1x300xb, 1xi64)
        t269 = paddle._C_ops.unsqueeze(t268, t22)
        del t268, t22

        # pd_op.cast: (1x300x1xf32) <- (1x300x1xb)
        t270 = paddle._C_ops.cast(t269, paddle.float32)
        del t269

        # pd_op.multiply: (1x300x4xf32) <- (1x300x4xf32, 1x300x1xf32)
        t271 = paddle._C_ops.multiply(t267, t270)
        del t270, t267

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t272 = [t215, t213, t215, t213]
        del t213, t215

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t273 = paddle._C_ops.stack(t272, 0)
        del t272

        # pd_op.assign: (4xi64) <- (4xi64)
        t274 = t273
        del t273

        # pd_op.cast: (4xf32) <- (4xi64)
        t275 = paddle._C_ops.cast(t274, paddle.float32)
        del t274

        # pd_op.divide: (1x300x4xf32) <- (1x300x4xf32, 4xf32)
        t276 = paddle._C_ops.divide(t271, t275)
        del t275, t271

        # pd_op.full: (1xi32) <- ()
        t277 = paddle._C_ops.full([1], float("2"), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([1x300x1xf32, 1x300x1xf32, 1x300x1xf32, 1x300x1xf32]) <- (1x300x4xf32, 1xi32)
        t278 = paddle._C_ops.split_with_num(t276, 4, t277)
        del t276, t277

        # builtin.split: (1x300x1xf32, 1x300x1xf32, 1x300x1xf32, 1x300x1xf32) <- ([1x300x1xf32, 1x300x1xf32, 1x300x1xf32, 1x300x1xf32])
        (
            t279,
            t280,
            t281,
            t282,
        ) = t278
        del t278

        # pd_op.add: (1x300x1xf32) <- (1x300x1xf32, 1x300x1xf32)
        t283 = paddle._C_ops.add(t279, t281)

        # pd_op.full: (1xf32) <- ()
        t284 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x300x1xf32) <- (1x300x1xf32, 1xf32)
        t285 = paddle._C_ops.scale(t283, t284, float("0"), True)
        del t283

        # pd_op.add: (1x300x1xf32) <- (1x300x1xf32, 1x300x1xf32)
        t286 = paddle._C_ops.add(t280, t282)

        # pd_op.scale: (1x300x1xf32) <- (1x300x1xf32, 1xf32)
        t287 = paddle._C_ops.scale(t286, t284, float("0"), True)
        del t286, t284

        # pd_op.subtract: (1x300x1xf32) <- (1x300x1xf32, 1x300x1xf32)
        t288 = paddle._C_ops.subtract(t281, t279)
        del t281, t279

        # pd_op.subtract: (1x300x1xf32) <- (1x300x1xf32, 1x300x1xf32)
        t289 = paddle._C_ops.subtract(t282, t280)
        del t282, t280

        # builtin.combine: ([1x300x1xf32, 1x300x1xf32, 1x300x1xf32, 1x300x1xf32]) <- (1x300x1xf32, 1x300x1xf32, 1x300x1xf32, 1x300x1xf32)
        t290 = [t285, t287, t288, t289]
        del t285, t287, t288, t289

        # pd_op.concat: (1x300x4xf32) <- ([1x300x1xf32, 1x300x1xf32, 1x300x1xf32, 1x300x1xf32], 1xi32)
        t291 = paddle._C_ops.concat(t290, t48)
        del t290, t48

        # pd_op.clip: (1x300x4xf32) <- (1x300x4xf32, 1xf32, 1xf32)
        t292 = paddle._C_ops.clip(t291, t24, t26)
        del t291, t24, t26

        # pd_op.full: (1xf32) <- ()
        t293 = paddle._C_ops.full(
            [1], float("1e-05"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        t294 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (1x300x4xf32) <- (1x300x4xf32, 1xf32, 1xf32)
        t295 = paddle._C_ops.clip(t292, t293, t294)

        # pd_op.scale: (1x300x4xf32) <- (1x300x4xf32, 1xf32)
        t296 = paddle._C_ops.scale(t292, t121, float("1"), True)
        del t292, t121

        # pd_op.clip: (1x300x4xf32) <- (1x300x4xf32, 1xf32, 1xf32)
        t297 = paddle._C_ops.clip(t296, t293, t294)
        del t293, t294, t296

        # pd_op.divide: (1x300x4xf32) <- (1x300x4xf32, 1x300x4xf32)
        t298 = paddle._C_ops.divide(t295, t297)
        del t295, t297

        # pd_op.log: (1x300x4xf32) <- (1x300x4xf32)
        t299 = paddle._C_ops.log(t298)
        del t298

        # builtin.combine: ([1x100x4xf32, 1x300x4xf32]) <- (1x100x4xf32, 1x300x4xf32)
        t300 = [input9, t299]
        del input9, t299

        # pd_op.concat: (1x400x4xf32) <- ([1x100x4xf32, 1x300x4xf32], 1xi32)
        t301 = paddle._C_ops.concat(t300, t111)
        del t300, t111

        # pd_op.layer_norm: (1x400x256xf32, 1x400xf32, 1x400xf32) <- (1x400x256xf32, 256xf32, 256xf32)
        t302, t303, t304 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t209, t12, t13, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t13, t12

        # pd_op.matmul: (1x400x2xf32) <- (1x400x256xf32, 256x2xf32)
        t305 = paddle._C_ops.matmul(t302, t4, False, False)
        del t4

        # pd_op.add: (1x400x2xf32) <- (1x400x2xf32, 2xf32)
        t306 = paddle._C_ops.add(t305, t5)
        del t5

        # pd_op.matmul: (1x400x256xf32) <- (1x400x256xf32, 256x256xf32)
        t307 = paddle._C_ops.matmul(t302, t14, False, False)
        del t14

        # pd_op.add: (1x400x256xf32) <- (1x400x256xf32, 256xf32)
        t308 = paddle._C_ops.add(t307, t15)
        del t15

        # pd_op.relu: (1x400x256xf32) <- (1x400x256xf32)
        t309 = paddle._C_ops.relu(t308)
        del t308

        # pd_op.matmul: (1x400x256xf32) <- (1x400x256xf32, 256x256xf32)
        t310 = paddle._C_ops.matmul(t309, t16, False, False)
        del t16

        # pd_op.add: (1x400x256xf32) <- (1x400x256xf32, 256xf32)
        t311 = paddle._C_ops.add(t310, t17)
        del t17

        # pd_op.relu: (1x400x256xf32) <- (1x400x256xf32)
        t312 = paddle._C_ops.relu(t311)
        del t311

        # pd_op.matmul: (1x400x128xf32) <- (1x400x256xf32, 256x128xf32)
        t313 = paddle._C_ops.matmul(t312, t18, False, False)
        del t18

        # pd_op.add: (1x400x128xf32) <- (1x400x128xf32, 128xf32)
        t314 = paddle._C_ops.add(t313, t19)
        del t19

        # pd_op.bmm: (1x400x-1xf32) <- (1x400x128xf32, 1x128x-1xf32)
        t315 = paddle._C_ops.bmm(t314, t199)

        # pd_op.full: (xi64) <- ()
        t316 = paddle._C_ops.full(
            [], float("400"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t317 = [t202, t316, t196, t198]
        del t202, t316, t196, t198

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t318 = paddle._C_ops.stack(t317, 0)
        del t317

        # pd_op.reshape: (1x400x-1x-1xf32) <- (1x400x-1xf32, 4xi64)
        t319 = paddle._C_ops.reshape(t315, t318)
        del t318

        # pd_op.sigmoid: (1x400x4xf32) <- (1x400x4xf32)
        t320 = paddle._C_ops.sigmoid(t301)

        return (
            t112,
            t138,
            t140,
            t144,
            t145,
            t146,
            t148,
            t149,
            t150,
            t151,
            t152,
            t153,
            t154,
            t157,
            t159,
            t160,
            t162,
            t163,
            t164,
            t165,
            t178,
            t179,
            t181,
            t182,
            t183,
            t184,
            t185,
            t186,
            t188,
            t189,
            t191,
            t192,
            t193,
            t199,
            t200,
            t201,
            t206,
            t207,
            input10,
            t209,
            t301,
            t302,
            t303,
            t304,
            t305,
            t306,
            t307,
            t309,
            t310,
            t312,
            t313,
            t314,
            t315,
            t319,
            t320,
        )
