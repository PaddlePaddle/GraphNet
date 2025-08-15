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
        input1: paddle.Tensor,
        input2: paddle.Tensor,
    ):
        # pd_op.full: (1xf64) <- ()
        t12 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        t13 = paddle._C_ops.full(
            [1], float("84"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        t14 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (84xf32) <- (1xf64, 1xf64, 1xf64)
        t15 = paddle.arange(t12, t13, t14, dtype="float32")
        del t13

        # builtin.combine: ([84xf32, 84xf32]) <- (84xf32, 84xf32)
        t16 = [t15, t15]
        del t15

        # pd_op.meshgrid: ([84x84xf32, 84x84xf32]) <- ([84xf32, 84xf32])
        t17 = paddle._C_ops.meshgrid(t16)
        del t16

        # builtin.split: (84x84xf32, 84x84xf32) <- ([84x84xf32, 84x84xf32])
        (
            t18,
            t19,
        ) = t17
        del t17

        # builtin.combine: ([84x84xf32, 84x84xf32]) <- (84x84xf32, 84x84xf32)
        t20 = [t19, t18]
        del t18, t19

        # pd_op.stack: (84x84x2xf32) <- ([84x84xf32, 84x84xf32])
        t21 = paddle._C_ops.stack(t20, -1)
        del t20

        # pd_op.full: (2xi64) <- ()
        t22 = paddle._C_ops.full(
            [2], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (2xi64) <- (2xi64)
        t23 = paddle._C_ops.assign_value_(
            t22,
            [2],
            paddle.int64,
            [float("84"), float("84")],
            paddle.framework._current_expected_place(),
        )
        del t22

        # pd_op.cast: (2xf32) <- (2xi64)
        t24 = paddle._C_ops.cast(t23, paddle.float32)
        del t23

        # pd_op.full_int_array: (1xi64) <- ()
        t25 = [0]

        # pd_op.unsqueeze: (1x84x84x2xf32) <- (84x84x2xf32, 1xi64)
        t26 = paddle._C_ops.unsqueeze(t21, t25)
        del t21

        # pd_op.full: (1xf32) <- ()
        t27 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x84x84x2xf32) <- (1x84x84x2xf32, 1xf32)
        t28 = paddle._C_ops.scale(t26, t27, float("0.5"), True)
        del t26

        # pd_op.divide: (1x84x84x2xf32) <- (1x84x84x2xf32, 2xf32)
        t29 = paddle._C_ops.divide(t28, t24)
        del t24, t28

        # pd_op.full_like: (1x84x84x2xf32) <- (1x84x84x2xf32, 1xf32)
        t30 = paddle._C_ops.full_like(
            t29, t27, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.full: (1xf32) <- ()
        t31 = paddle._C_ops.full(
            [1], float("0.05"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x84x84x2xf32) <- (1x84x84x2xf32, 1xf32)
        t32 = paddle._C_ops.scale(t30, t31, float("0"), True)
        del t30

        # pd_op.scale: (1x84x84x2xf32) <- (1x84x84x2xf32, 1xf32)
        t33 = paddle._C_ops.scale(t32, t27, float("0"), True)
        del t32

        # pd_op.full: (1xi32) <- ()
        t34 = paddle._C_ops.full([1], float("-1"), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1x84x84x2xf32, 1x84x84x2xf32]) <- (1x84x84x2xf32, 1x84x84x2xf32)
        t35 = [t29, t33]
        del t29, t33

        # pd_op.concat: (1x84x84x4xf32) <- ([1x84x84x2xf32, 1x84x84x2xf32], 1xi32)
        t36 = paddle._C_ops.concat(t35, t34)
        del t35

        # pd_op.full_int_array: (3xi64) <- ()
        t37 = [-1, 7056, 4]

        # pd_op.reshape: (1x7056x4xf32) <- (1x84x84x4xf32, 3xi64)
        t38 = paddle._C_ops.reshape(t36, t37)
        del t36, t37

        # pd_op.full: (1xf64) <- ()
        t39 = paddle._C_ops.full(
            [1], float("42"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (42xf32) <- (1xf64, 1xf64, 1xf64)
        t40 = paddle.arange(t12, t39, t14, dtype="float32")
        del t39

        # builtin.combine: ([42xf32, 42xf32]) <- (42xf32, 42xf32)
        t41 = [t40, t40]
        del t40

        # pd_op.meshgrid: ([42x42xf32, 42x42xf32]) <- ([42xf32, 42xf32])
        t42 = paddle._C_ops.meshgrid(t41)
        del t41

        # builtin.split: (42x42xf32, 42x42xf32) <- ([42x42xf32, 42x42xf32])
        (
            t43,
            t44,
        ) = t42
        del t42

        # builtin.combine: ([42x42xf32, 42x42xf32]) <- (42x42xf32, 42x42xf32)
        t45 = [t44, t43]
        del t43, t44

        # pd_op.stack: (42x42x2xf32) <- ([42x42xf32, 42x42xf32])
        t46 = paddle._C_ops.stack(t45, -1)
        del t45

        # pd_op.full: (2xi64) <- ()
        t47 = paddle._C_ops.full(
            [2], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (2xi64) <- (2xi64)
        t48 = paddle._C_ops.assign_value_(
            t47,
            [2],
            paddle.int64,
            [float("42"), float("42")],
            paddle.framework._current_expected_place(),
        )
        del t47

        # pd_op.cast: (2xf32) <- (2xi64)
        t49 = paddle._C_ops.cast(t48, paddle.float32)
        del t48

        # pd_op.unsqueeze: (1x42x42x2xf32) <- (42x42x2xf32, 1xi64)
        t50 = paddle._C_ops.unsqueeze(t46, t25)
        del t46

        # pd_op.scale: (1x42x42x2xf32) <- (1x42x42x2xf32, 1xf32)
        t51 = paddle._C_ops.scale(t50, t27, float("0.5"), True)
        del t50

        # pd_op.divide: (1x42x42x2xf32) <- (1x42x42x2xf32, 2xf32)
        t52 = paddle._C_ops.divide(t51, t49)
        del t49, t51

        # pd_op.full_like: (1x42x42x2xf32) <- (1x42x42x2xf32, 1xf32)
        t53 = paddle._C_ops.full_like(
            t52, t27, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.scale: (1x42x42x2xf32) <- (1x42x42x2xf32, 1xf32)
        t54 = paddle._C_ops.scale(t53, t31, float("0"), True)
        del t53

        # pd_op.full: (1xf32) <- ()
        t55 = paddle._C_ops.full(
            [1], float("2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x42x42x2xf32) <- (1x42x42x2xf32, 1xf32)
        t56 = paddle._C_ops.scale(t54, t55, float("0"), True)
        del t55, t54

        # builtin.combine: ([1x42x42x2xf32, 1x42x42x2xf32]) <- (1x42x42x2xf32, 1x42x42x2xf32)
        t57 = [t52, t56]
        del t52, t56

        # pd_op.concat: (1x42x42x4xf32) <- ([1x42x42x2xf32, 1x42x42x2xf32], 1xi32)
        t58 = paddle._C_ops.concat(t57, t34)
        del t57

        # pd_op.full_int_array: (3xi64) <- ()
        t59 = [-1, 1764, 4]

        # pd_op.reshape: (1x1764x4xf32) <- (1x42x42x4xf32, 3xi64)
        t60 = paddle._C_ops.reshape(t58, t59)
        del t58, t59

        # pd_op.full: (1xf64) <- ()
        t61 = paddle._C_ops.full(
            [1], float("21"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (21xf32) <- (1xf64, 1xf64, 1xf64)
        t62 = paddle.arange(t12, t61, t14, dtype="float32")
        del t61

        # builtin.combine: ([21xf32, 21xf32]) <- (21xf32, 21xf32)
        t63 = [t62, t62]
        del t62

        # pd_op.meshgrid: ([21x21xf32, 21x21xf32]) <- ([21xf32, 21xf32])
        t64 = paddle._C_ops.meshgrid(t63)
        del t63

        # builtin.split: (21x21xf32, 21x21xf32) <- ([21x21xf32, 21x21xf32])
        (
            t65,
            t66,
        ) = t64
        del t64

        # builtin.combine: ([21x21xf32, 21x21xf32]) <- (21x21xf32, 21x21xf32)
        t67 = [t66, t65]
        del t65, t66

        # pd_op.stack: (21x21x2xf32) <- ([21x21xf32, 21x21xf32])
        t68 = paddle._C_ops.stack(t67, -1)
        del t67

        # pd_op.full: (2xi64) <- ()
        t69 = paddle._C_ops.full(
            [2], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (2xi64) <- (2xi64)
        t70 = paddle._C_ops.assign_value_(
            t69,
            [2],
            paddle.int64,
            [float("21"), float("21")],
            paddle.framework._current_expected_place(),
        )
        del t69

        # pd_op.cast: (2xf32) <- (2xi64)
        t71 = paddle._C_ops.cast(t70, paddle.float32)
        del t70

        # pd_op.unsqueeze: (1x21x21x2xf32) <- (21x21x2xf32, 1xi64)
        t72 = paddle._C_ops.unsqueeze(t68, t25)
        del t25, t68

        # pd_op.scale: (1x21x21x2xf32) <- (1x21x21x2xf32, 1xf32)
        t73 = paddle._C_ops.scale(t72, t27, float("0.5"), True)
        del t72

        # pd_op.divide: (1x21x21x2xf32) <- (1x21x21x2xf32, 2xf32)
        t74 = paddle._C_ops.divide(t73, t71)
        del t71, t73

        # pd_op.full_like: (1x21x21x2xf32) <- (1x21x21x2xf32, 1xf32)
        t75 = paddle._C_ops.full_like(
            t74, t27, paddle.float32, paddle.framework._current_expected_place()
        )
        del t27

        # pd_op.scale: (1x21x21x2xf32) <- (1x21x21x2xf32, 1xf32)
        t76 = paddle._C_ops.scale(t75, t31, float("0"), True)
        del t31, t75

        # pd_op.full: (1xf32) <- ()
        t77 = paddle._C_ops.full(
            [1], float("4"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x21x21x2xf32) <- (1x21x21x2xf32, 1xf32)
        t78 = paddle._C_ops.scale(t76, t77, float("0"), True)
        del t77, t76

        # builtin.combine: ([1x21x21x2xf32, 1x21x21x2xf32]) <- (1x21x21x2xf32, 1x21x21x2xf32)
        t79 = [t74, t78]
        del t74, t78

        # pd_op.concat: (1x21x21x4xf32) <- ([1x21x21x2xf32, 1x21x21x2xf32], 1xi32)
        t80 = paddle._C_ops.concat(t79, t34)
        del t79, t34

        # pd_op.full_int_array: (3xi64) <- ()
        t81 = [-1, 441, 4]

        # pd_op.reshape: (1x441x4xf32) <- (1x21x21x4xf32, 3xi64)
        t82 = paddle._C_ops.reshape(t80, t81)
        del t80, t81

        # pd_op.full: (1xi32) <- ()
        t83 = paddle._C_ops.full([1], float("1"), paddle.int32, paddle.core.CPUPlace())

        # pd_op.assign: (1xi32) <- (1xi32)
        t84 = t83

        # builtin.combine: ([1x7056x4xf32, 1x1764x4xf32, 1x441x4xf32]) <- (1x7056x4xf32, 1x1764x4xf32, 1x441x4xf32)
        t85 = [t38, t60, t82]
        del t38, t60, t82

        # pd_op.concat: (1x9261x4xf32) <- ([1x7056x4xf32, 1x1764x4xf32, 1x441x4xf32], 1xi32)
        t86 = paddle._C_ops.concat(t85, t83)
        del t85

        # pd_op.full: (xf32) <- ()
        t87 = paddle._C_ops.full(
            [],
            float("0.01"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.greater_than: (1x9261x4xb) <- (1x9261x4xf32, xf32)
        t88 = paddle._C_ops.greater_than(t86, t87)
        del t87

        # pd_op.full: (xf32) <- ()
        t89 = paddle._C_ops.full(
            [],
            float("0.99"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.less_than: (1x9261x4xb) <- (1x9261x4xf32, xf32)
        t90 = paddle._C_ops.less_than(t86, t89)
        del t89

        # pd_op.multiply: (1x9261x4xb) <- (1x9261x4xb, 1x9261x4xb)
        t91 = paddle._C_ops.multiply(t88, t90)
        del t88, t90

        # pd_op.all: (1x9261x1xb) <- (1x9261x4xb)
        t92 = paddle._C_ops.all(t91, [-1], True)
        del t91

        # pd_op.full: (1xf32) <- ()
        t93 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x9261x4xf32) <- (1x9261x4xf32, 1xf32)
        t94 = paddle._C_ops.scale(t86, t93, float("1"), True)
        del t93

        # pd_op.divide: (1x9261x4xf32) <- (1x9261x4xf32, 1x9261x4xf32)
        t95 = paddle._C_ops.divide(t86, t94)
        del t86, t94

        # pd_op.log: (1x9261x4xf32) <- (1x9261x4xf32)
        t96 = paddle._C_ops.log(t95)
        del t95

        # pd_op.full: (xf32) <- ()
        t97 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        t98 = paddle._C_ops.assign_value_(
            t97,
            [],
            paddle.float32,
            [float("inf")],
            paddle.framework._current_expected_place(),
        )
        del t97

        # pd_op.full: (1xf32) <- ()
        t99 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (1x9261x4xf32) <- (1x9261x4xf32, 1xf32)
        t100 = paddle._C_ops.full_like(
            t96, t99, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.full_like: (xf32) <- (xf32, 1xf32)
        t101 = paddle._C_ops.full_like(
            t98, t99, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.full_like: (1x9261x1xb) <- (1x9261x1xb, 1xf32)
        t102 = paddle._C_ops.full_like(
            t92, t99, paddle.bool, paddle.framework._current_expected_place()
        )

        # pd_op.cast: (1x9261x1xf32) <- (1x9261x1xb)
        t103 = paddle._C_ops.cast(t102, paddle.float32)
        del t102

        # pd_op.cast: (1x9261x1xf32) <- (1x9261x1xb)
        t104 = paddle._C_ops.cast(t92, paddle.float32)

        # pd_op.add: (1x9261x4xf32) <- (1x9261x4xf32, xf32)
        t105 = paddle._C_ops.add(t100, t101)
        del t100, t101

        # pd_op.add: (1x9261x4xf32) <- (1x9261x4xf32, 1x9261x1xf32)
        t106 = paddle._C_ops.add(t105, t103)
        del t105, t103

        # pd_op.add: (1x9261x4xf32) <- (1x9261x4xf32, 1x9261x4xf32)
        t107 = paddle._C_ops.add(t96, t106)
        del t96

        # pd_op.add: (1x9261x4xf32) <- (xf32, 1x9261x4xf32)
        t108 = paddle._C_ops.add(t98, t106)
        del t98

        # pd_op.add: (1x9261x4xf32) <- (1x9261x1xf32, 1x9261x4xf32)
        t109 = paddle._C_ops.add(t104, t106)
        del t106, t104

        # pd_op.cast: (1x9261x4xb) <- (1x9261x4xf32)
        t110 = paddle._C_ops.cast(t109, paddle.bool)
        del t109

        # pd_op.where: (1x9261x4xf32) <- (1x9261x4xb, 1x9261x4xf32, 1x9261x4xf32)
        t111 = paddle._C_ops.where(t110, t107, t108)
        del t108, t107, t110

        # pd_op.full: (xf32) <- ()
        t112 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        t113 = paddle._C_ops.assign_value_(
            t112,
            [],
            paddle.float32,
            [float("0")],
            paddle.framework._current_expected_place(),
        )
        del t112

        # pd_op.full_like: (8x9261x256xf32) <- (8x9261x256xf32, 1xf32)
        t114 = paddle._C_ops.full_like(
            input0, t99, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.full_like: (xf32) <- (xf32, 1xf32)
        t115 = paddle._C_ops.full_like(
            t113, t99, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.full_like: (1x9261x1xb) <- (1x9261x1xb, 1xf32)
        t116 = paddle._C_ops.full_like(
            t92, t99, paddle.bool, paddle.framework._current_expected_place()
        )
        del t99

        # pd_op.cast: (1x9261x1xf32) <- (1x9261x1xb)
        t117 = paddle._C_ops.cast(t116, paddle.float32)
        del t116

        # pd_op.cast: (1x9261x1xf32) <- (1x9261x1xb)
        t118 = paddle._C_ops.cast(t92, paddle.float32)
        del t92

        # pd_op.add: (8x9261x256xf32) <- (8x9261x256xf32, xf32)
        t119 = paddle._C_ops.add(t114, t115)
        del t114, t115

        # pd_op.add: (8x9261x256xf32) <- (8x9261x256xf32, 1x9261x1xf32)
        t120 = paddle._C_ops.add(t119, t117)
        del t119, t117

        # pd_op.add: (8x9261x256xf32) <- (8x9261x256xf32, 8x9261x256xf32)
        t121 = paddle._C_ops.add(input0, t120)
        del input0

        # pd_op.add: (8x9261x256xf32) <- (xf32, 8x9261x256xf32)
        t122 = paddle._C_ops.add(t113, t120)

        # pd_op.add: (8x9261x256xf32) <- (1x9261x1xf32, 8x9261x256xf32)
        t123 = paddle._C_ops.add(t118, t120)
        del t118

        # pd_op.cast: (8x9261x256xb) <- (8x9261x256xf32)
        t124 = paddle._C_ops.cast(t123, paddle.bool)
        del t123

        # pd_op.where: (8x9261x256xf32) <- (8x9261x256xb, 8x9261x256xf32, 8x9261x256xf32)
        t125 = paddle._C_ops.where(t124, t121, t122)

        # pd_op.matmul: (8x9261x256xf32) <- (8x9261x256xf32, 256x256xf32)
        t126 = paddle._C_ops.matmul(t125, t0, False, False)
        del t0

        # pd_op.add: (8x9261x256xf32) <- (8x9261x256xf32, 256xf32)
        t127 = paddle._C_ops.add(t126, t1)
        del t1

        # pd_op.layer_norm: (8x9261x256xf32, 8x9261xf32, 8x9261xf32) <- (8x9261x256xf32, 256xf32, 256xf32)
        t128, t129, t130 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t127, t2, t3, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t3, t2

        # pd_op.matmul: (8x9261x4xf32) <- (8x9261x256xf32, 256x4xf32)
        t131 = paddle._C_ops.matmul(t128, t4, False, False)
        del t4

        # pd_op.add: (8x9261x4xf32) <- (8x9261x4xf32, 4xf32)
        t132 = paddle._C_ops.add(t131, t5)
        del t5

        # pd_op.matmul: (8x9261x256xf32) <- (8x9261x256xf32, 256x256xf32)
        t133 = paddle._C_ops.matmul(t128, t6, False, False)
        del t6

        # pd_op.add: (8x9261x256xf32) <- (8x9261x256xf32, 256xf32)
        t134 = paddle._C_ops.add(t133, t7)
        del t7

        # pd_op.relu: (8x9261x256xf32) <- (8x9261x256xf32)
        t135 = paddle._C_ops.relu(t134)
        del t134

        # pd_op.matmul: (8x9261x256xf32) <- (8x9261x256xf32, 256x256xf32)
        t136 = paddle._C_ops.matmul(t135, t8, False, False)
        del t8

        # pd_op.add: (8x9261x256xf32) <- (8x9261x256xf32, 256xf32)
        t137 = paddle._C_ops.add(t136, t9)
        del t9

        # pd_op.relu: (8x9261x256xf32) <- (8x9261x256xf32)
        t138 = paddle._C_ops.relu(t137)
        del t137

        # pd_op.matmul: (8x9261x4xf32) <- (8x9261x256xf32, 256x4xf32)
        t139 = paddle._C_ops.matmul(t138, t10, False, False)
        del t10

        # pd_op.add: (8x9261x4xf32) <- (8x9261x4xf32, 4xf32)
        t140 = paddle._C_ops.add(t139, t11)
        del t11

        # pd_op.add: (8x9261x4xf32) <- (8x9261x4xf32, 1x9261x4xf32)
        t141 = paddle._C_ops.add(t140, t111)

        # pd_op.full_int_array: (1xi64) <- ()
        t142 = [-1]

        # pd_op.max: (8x9261xf32) <- (8x9261x4xf32, 1xi64)
        t143 = paddle._C_ops.max(t132, t142, False)

        # pd_op.full: (1xi32) <- ()
        t144 = paddle._C_ops.full(
            [1], float("300"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.topk: (8x300xf32, 8x300xi64) <- (8x9261xf32, 1xi32)
        t145, t146 = (lambda x, f: f(x))(
            paddle._C_ops.topk(t143, t144, 1, True, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del t144, t143

        # pd_op.full: (1xf64) <- ()
        t147 = paddle._C_ops.full(
            [1], float("8"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (8xi64) <- (1xf64, 1xf64, 1xf64)
        t148 = paddle.arange(t12, t147, t14, dtype="int64")
        del t12, t14, t147

        # pd_op.unsqueeze: (8x1xi64) <- (8xi64, 1xi64)
        t149 = paddle._C_ops.unsqueeze(t148, t142)
        del t148, t142

        # pd_op.full_int_array: (2xi64) <- ()
        t150 = [1, 300]

        # pd_op.tile: (8x300xi64) <- (8x1xi64, 2xi64)
        t151 = paddle._C_ops.tile(t149, t150)
        del t150, t149

        # builtin.combine: ([8x300xi64, 8x300xi64]) <- (8x300xi64, 8x300xi64)
        t152 = [t151, t146]
        del t151, t146

        # pd_op.stack: (8x300x2xi64) <- ([8x300xi64, 8x300xi64])
        t153 = paddle._C_ops.stack(t152, -1)
        del t152

        # pd_op.gather_nd: (8x300x4xf32) <- (8x9261x4xf32, 8x300x2xi64)
        t154 = paddle._C_ops.gather_nd(t141, t153)

        # pd_op.sigmoid: (8x300x4xf32) <- (8x300x4xf32)
        t155 = paddle._C_ops.sigmoid(t154)

        # builtin.combine: ([8x200x4xf32, 8x300x4xf32]) <- (8x200x4xf32, 8x300x4xf32)
        t156 = [input1, t154]
        del input1, t154

        # pd_op.concat: (8x500x4xf32) <- ([8x200x4xf32, 8x300x4xf32], 1xi32)
        t157 = paddle._C_ops.concat(t156, t83)
        del t156

        # pd_op.share_data_: (8x500x4xf32) <- (8x500x4xf32)
        input3 = t157.detach()
        del t157

        # pd_op.gather_nd: (8x300x4xf32) <- (8x9261x4xf32, 8x300x2xi64)
        t158 = paddle._C_ops.gather_nd(t132, t153)

        # pd_op.gather_nd: (8x300x256xf32) <- (8x9261x256xf32, 8x300x2xi64)
        t159 = paddle._C_ops.gather_nd(t128, t153)

        # pd_op.share_data_: (8x300x256xf32) <- (8x300x256xf32)
        input4 = t159.detach()
        del t159

        # builtin.combine: ([8x200x256xf32, 8x300x256xf32]) <- (8x200x256xf32, 8x300x256xf32)
        t160 = [input2, input4]
        del input2

        return (
            t83,
            t84,
            t111,
            t113,
            t120,
            t121,
            t122,
            t124,
            t125,
            t126,
            t127,
            t128,
            t129,
            t130,
            t131,
            t132,
            t133,
            t135,
            t136,
            t138,
            t139,
            t140,
            t141,
            t153,
            t155,
            input3,
            t158,
            input4,
            t160,
        )
