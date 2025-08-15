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
        input2: paddle.Tensor,
        input3: paddle.Tensor,
        t2: paddle.Tensor,
        t3: paddle.Tensor,
        t4: paddle.Tensor,
        t5: paddle.Tensor,
        t6: paddle.Tensor,
        t7: paddle.Tensor,
        input4: paddle.Tensor,
        t8: paddle.Tensor,
        t9: paddle.Tensor,
        t10: paddle.Tensor,
        t11: paddle.Tensor,
        t12: paddle.Tensor,
        t13: paddle.Tensor,
        t14: paddle.Tensor,
        t15: paddle.Tensor,
    ):
        # pd_op.add: (1x400x256xf32) <- (1x400x256xf32, 1x400x256xf32)
        t16 = paddle._C_ops.add(input0, input1)
        del input1, input0

        # pd_op.layer_norm: (1x400x256xf32, 1x400xf32, 1x400xf32) <- (1x400x256xf32, 256xf32, 256xf32)
        t17, t18, t19 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t16, t0, t1, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t1, t0

        # pd_op.add: (1x400x256xf32) <- (1x400x256xf32, 1x400x256xf32)
        t20 = paddle._C_ops.add(t17, input2)
        del input2

        # pd_op.matmul: (1x6069x256xf32) <- (1x6069x256xf32, 256x256xf32)
        t21 = paddle._C_ops.matmul(input3, t2, False, False)
        del input3, t2

        # pd_op.add: (1x6069x256xf32) <- (1x6069x256xf32, 256xf32)
        t22 = paddle._C_ops.add(t21, t3)
        del t3

        # pd_op.full_int_array: (4xi64) <- ()
        t23 = [1, 6069, 8, 32]

        # pd_op.reshape: (1x6069x8x32xf32) <- (1x6069x256xf32, 4xi64)
        t24 = paddle._C_ops.reshape(t22, t23)
        del t23

        # pd_op.matmul: (1x400x192xf32) <- (1x400x256xf32, 256x192xf32)
        t25 = paddle._C_ops.matmul(t20, t4, False, False)
        del t4

        # pd_op.add: (1x400x192xf32) <- (1x400x192xf32, 192xf32)
        t26 = paddle._C_ops.add(t25, t5)
        del t5

        # pd_op.full_int_array: (6xi64) <- ()
        t27 = [1, 400, 8, 3, 4, 2]

        # pd_op.reshape: (1x400x8x3x4x2xf32) <- (1x400x192xf32, 6xi64)
        t28 = paddle._C_ops.reshape(t26, t27)
        del t27

        # pd_op.matmul: (1x400x96xf32) <- (1x400x256xf32, 256x96xf32)
        t29 = paddle._C_ops.matmul(t20, t6, False, False)
        del t6

        # pd_op.add: (1x400x96xf32) <- (1x400x96xf32, 96xf32)
        t30 = paddle._C_ops.add(t29, t7)
        del t7

        # pd_op.full_int_array: (4xi64) <- ()
        t31 = [1, 400, 8, 12]

        # pd_op.reshape: (1x400x8x12xf32) <- (1x400x96xf32, 4xi64)
        t32 = paddle._C_ops.reshape(t30, t31)
        del t31

        # pd_op.softmax: (1x400x8x12xf32) <- (1x400x8x12xf32)
        t33 = paddle._C_ops.softmax(t32, -1)
        del t32

        # pd_op.full_int_array: (5xi64) <- ()
        t34 = [1, 400, 8, 3, 4]

        # pd_op.reshape: (1x400x8x3x4xf32) <- (1x400x8x12xf32, 5xi64)
        t35 = paddle._C_ops.reshape(t33, t34)
        del t34

        # pd_op.full_int_array: (1xi64) <- ()
        t36 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        t37 = t36

        # pd_op.full_int_array: (1xi64) <- ()
        t38 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        t39 = t38

        # pd_op.assign: (1xi64) <- (1xi64)
        t40 = t38

        # pd_op.slice: (1x400x1x2xf32) <- (1x400x1x4xf32, 1xi64, 1xi64)
        t41 = paddle._C_ops.slice(input4, [3], t36, t38, [1], [])

        # pd_op.full_int_array: (2xi64) <- ()
        t42 = [2, 4]

        # pd_op.unsqueeze: (1x400x1x1x1x2xf32) <- (1x400x1x2xf32, 2xi64)
        t43 = paddle._C_ops.unsqueeze(t41, t42)
        del t41

        # pd_op.full: (1xf32) <- ()
        t44 = paddle._C_ops.full(
            [1], float("0.25"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x400x8x3x4x2xf32) <- (1x400x8x3x4x2xf32, 1xf32)
        t45 = paddle._C_ops.scale(t28, t44, float("0"), True)
        del t28

        # pd_op.full_int_array: (1xi64) <- ()
        t46 = [2147483647]

        # pd_op.slice: (1x400x1x2xf32) <- (1x400x1x4xf32, 1xi64, 1xi64)
        t47 = paddle._C_ops.slice(input4, [3], t38, t46, [1], [])
        del input4, t46

        # pd_op.unsqueeze: (1x400x1x1x1x2xf32) <- (1x400x1x2xf32, 2xi64)
        t48 = paddle._C_ops.unsqueeze(t47, t42)
        del t42, t47

        # pd_op.multiply: (1x400x8x3x4x2xf32) <- (1x400x8x3x4x2xf32, 1x400x1x1x1x2xf32)
        t49 = paddle._C_ops.multiply(t45, t48)

        # pd_op.full: (1xf32) <- ()
        t50 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x400x8x3x4x2xf32) <- (1x400x8x3x4x2xf32, 1xf32)
        t51 = paddle._C_ops.scale(t49, t50, float("0"), True)
        del t49

        # pd_op.add: (1x400x8x3x4x2xf32) <- (1x400x1x1x1x2xf32, 1x400x8x3x4x2xf32)
        t52 = paddle._C_ops.add(t43, t51)

        # pd_op.full: (3x2xi64) <- ()
        t53 = paddle._C_ops.full(
            [3, 2], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (3x2xi64) <- (3x2xi64)
        t54 = paddle._C_ops.assign_value_(
            t53,
            [3, 2],
            paddle.int64,
            [
                float("68"),
                float("68"),
                float("34"),
                float("34"),
                float("17"),
                float("17"),
            ],
            paddle.framework._current_expected_place(),
        )
        del t53

        # pd_op.full: (3xi64) <- ()
        t55 = paddle._C_ops.full(
            [3], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (3xi64) <- (3xi64)
        t56 = paddle._C_ops.assign_value_(
            t55,
            [3],
            paddle.int64,
            [float("0"), float("4624"), float("5780")],
            paddle.framework._current_expected_place(),
        )
        del t55

        # pd_op.full_int_array: (1xi64) <- ()
        t57 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        t58 = t57

        # pd_op.assign: (1xi64) <- (1xi64)
        t59 = t57

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        t60 = paddle._C_ops.slice(t54, [0], t36, t57, [1], [0])

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        t61 = paddle._C_ops.slice(t60, [0], t36, t57, [1], [0])

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        t62 = paddle._C_ops.slice(t60, [0], t57, t38, [1], [0])
        del t60

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        t63 = paddle._C_ops.multiply(t61, t62)
        del t61, t62

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        t64 = paddle._C_ops.slice(t54, [0], t57, t38, [1], [0])

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        t65 = paddle._C_ops.slice(t64, [0], t36, t57, [1], [0])

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        t66 = paddle._C_ops.slice(t64, [0], t57, t38, [1], [0])
        del t64

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        t67 = paddle._C_ops.multiply(t65, t66)
        del t65, t66

        # pd_op.full_int_array: (1xi64) <- ()
        t68 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        t69 = t68

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        t70 = paddle._C_ops.slice(t54, [0], t38, t68, [1], [0])

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        t71 = paddle._C_ops.slice(t70, [0], t36, t57, [1], [0])

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        t72 = paddle._C_ops.slice(t70, [0], t57, t38, [1], [0])
        del t70

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        t73 = paddle._C_ops.multiply(t71, t72)
        del t72, t71

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t74 = [t63, t67, t73]
        del t63, t67, t73

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t75 = paddle._C_ops.stack(t74, 0)
        del t74

        # pd_op.full: (1xi32) <- ()
        t76 = paddle._C_ops.full([1], float("1"), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32]) <- (1x6069x8x32xf32, 3xi64, 1xi32)
        t77 = paddle._C_ops.split(t24, t75, t76)
        del t24, t75

        # builtin.split: (-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32) <- ([-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32])
        (
            t78,
            t79,
            t80,
        ) = t77
        del t77

        # pd_op.full: (1xf32) <- ()
        t81 = paddle._C_ops.full(
            [1], float("2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x400x8x3x4x2xf32) <- (1x400x8x3x4x2xf32, 1xf32)
        t82 = paddle._C_ops.scale(t52, t81, float("0"), True)
        del t52

        # pd_op.full: (1xf32) <- ()
        t83 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x400x8x3x4x2xf32) <- (1x400x8x3x4x2xf32, 1xf32)
        t84 = paddle._C_ops.scale(t82, t83, float("-1"), True)
        del t82

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        t85 = paddle._C_ops.slice(t54, [0], t36, t57, [1], [0])

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        t86 = paddle._C_ops.slice(t85, [0], t36, t57, [1], [0])

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        t87 = paddle._C_ops.slice(t85, [0], t57, t38, [1], [0])
        del t85

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        t88 = paddle._C_ops.flatten(t78, 2, 3)

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        t89 = paddle._C_ops.transpose(t88, [0, 2, 1])
        del t88

        # pd_op.full: (xi64) <- ()
        t90 = paddle._C_ops.full([], float("8"), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (xi64) <- ()
        t91 = paddle._C_ops.full([], float("32"), paddle.int64, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t92 = [t90, t91, t86, t87]
        del t86, t87

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t93 = paddle._C_ops.stack(t92, 0)
        del t92

        # pd_op.reshape: (8x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        t94 = paddle._C_ops.reshape(t89, t93)
        del t93

        # pd_op.slice: (1x400x8x4x2xf32) <- (1x400x8x3x4x2xf32, 1xi64, 1xi64)
        t95 = paddle._C_ops.slice(t84, [3], t36, t57, [1], [3])

        # pd_op.transpose: (1x8x400x4x2xf32) <- (1x400x8x4x2xf32)
        t96 = paddle._C_ops.transpose(t95, [0, 2, 1, 3, 4])
        del t95

        # pd_op.flatten: (8x400x4x2xf32) <- (1x8x400x4x2xf32)
        t97 = paddle._C_ops.flatten(t96, 0, 1)

        # pd_op.grid_sample: (8x32x400x4xf32) <- (8x32x-1x-1xf32, 8x400x4x2xf32)
        t98 = paddle._C_ops.grid_sample(t94, t97, "bilinear", "zeros", False)

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        t99 = paddle._C_ops.slice(t54, [0], t57, t38, [1], [0])

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        t100 = paddle._C_ops.slice(t99, [0], t36, t57, [1], [0])

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        t101 = paddle._C_ops.slice(t99, [0], t57, t38, [1], [0])
        del t99

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        t102 = paddle._C_ops.flatten(t79, 2, 3)

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        t103 = paddle._C_ops.transpose(t102, [0, 2, 1])
        del t102

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t104 = [t90, t91, t100, t101]
        del t100, t101

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t105 = paddle._C_ops.stack(t104, 0)
        del t104

        # pd_op.reshape: (8x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        t106 = paddle._C_ops.reshape(t103, t105)
        del t105

        # pd_op.slice: (1x400x8x4x2xf32) <- (1x400x8x3x4x2xf32, 1xi64, 1xi64)
        t107 = paddle._C_ops.slice(t84, [3], t57, t38, [1], [3])

        # pd_op.transpose: (1x8x400x4x2xf32) <- (1x400x8x4x2xf32)
        t108 = paddle._C_ops.transpose(t107, [0, 2, 1, 3, 4])
        del t107

        # pd_op.flatten: (8x400x4x2xf32) <- (1x8x400x4x2xf32)
        t109 = paddle._C_ops.flatten(t108, 0, 1)

        # pd_op.grid_sample: (8x32x400x4xf32) <- (8x32x-1x-1xf32, 8x400x4x2xf32)
        t110 = paddle._C_ops.grid_sample(t106, t109, "bilinear", "zeros", False)

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        t111 = paddle._C_ops.slice(t54, [0], t38, t68, [1], [0])
        del t54

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        t112 = paddle._C_ops.slice(t111, [0], t36, t57, [1], [0])
        del t36

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        t113 = paddle._C_ops.slice(t111, [0], t57, t38, [1], [0])
        del t57, t111

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        t114 = paddle._C_ops.flatten(t80, 2, 3)

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        t115 = paddle._C_ops.transpose(t114, [0, 2, 1])
        del t114

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t116 = [t90, t91, t112, t113]
        del t90, t91, t112, t113

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t117 = paddle._C_ops.stack(t116, 0)
        del t116

        # pd_op.reshape: (8x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        t118 = paddle._C_ops.reshape(t115, t117)
        del t117

        # pd_op.slice: (1x400x8x4x2xf32) <- (1x400x8x3x4x2xf32, 1xi64, 1xi64)
        t119 = paddle._C_ops.slice(t84, [3], t38, t68, [1], [3])
        del t68, t38

        # pd_op.transpose: (1x8x400x4x2xf32) <- (1x400x8x4x2xf32)
        t120 = paddle._C_ops.transpose(t119, [0, 2, 1, 3, 4])
        del t119

        # pd_op.flatten: (8x400x4x2xf32) <- (1x8x400x4x2xf32)
        t121 = paddle._C_ops.flatten(t120, 0, 1)

        # pd_op.grid_sample: (8x32x400x4xf32) <- (8x32x-1x-1xf32, 8x400x4x2xf32)
        t122 = paddle._C_ops.grid_sample(t118, t121, "bilinear", "zeros", False)

        # pd_op.transpose: (1x8x400x3x4xf32) <- (1x400x8x3x4xf32)
        t123 = paddle._C_ops.transpose(t35, [0, 2, 1, 3, 4])
        del t35

        # pd_op.full_int_array: (4xi64) <- ()
        t124 = [8, 1, 400, 12]

        # pd_op.reshape: (8x1x400x12xf32) <- (1x8x400x3x4xf32, 4xi64)
        t125 = paddle._C_ops.reshape(t123, t124)
        del t124

        # builtin.combine: ([8x32x400x4xf32, 8x32x400x4xf32, 8x32x400x4xf32]) <- (8x32x400x4xf32, 8x32x400x4xf32, 8x32x400x4xf32)
        t126 = [t98, t110, t122]

        # pd_op.stack: (8x32x400x3x4xf32) <- ([8x32x400x4xf32, 8x32x400x4xf32, 8x32x400x4xf32])
        t127 = paddle._C_ops.stack(t126, -2)
        del t126

        # pd_op.flatten: (8x32x400x12xf32) <- (8x32x400x3x4xf32)
        t128 = paddle._C_ops.flatten(t127, 3, 4)

        # pd_op.multiply: (8x32x400x12xf32) <- (8x32x400x12xf32, 8x1x400x12xf32)
        t129 = paddle._C_ops.multiply(t128, t125)

        # pd_op.full_int_array: (1xi64) <- ()
        t130 = [-1]

        # pd_op.sum: (8x32x400xf32) <- (8x32x400x12xf32, 1xi64)
        t131 = paddle._C_ops.sum(t129, t130, None, False)

        # pd_op.full_int_array: (3xi64) <- ()
        t132 = [1, 256, 400]

        # pd_op.reshape: (1x256x400xf32) <- (8x32x400xf32, 3xi64)
        t133 = paddle._C_ops.reshape(t131, t132)
        del t132

        # pd_op.transpose: (1x400x256xf32) <- (1x256x400xf32)
        t134 = paddle._C_ops.transpose(t133, [0, 2, 1])
        del t133

        # pd_op.matmul: (1x400x256xf32) <- (1x400x256xf32, 256x256xf32)
        t135 = paddle._C_ops.matmul(t134, t8, False, False)
        del t8

        # pd_op.add: (1x400x256xf32) <- (1x400x256xf32, 256xf32)
        t136 = paddle._C_ops.add(t135, t9)
        del t9

        # pd_op.add: (1x400x256xf32) <- (1x400x256xf32, 1x400x256xf32)
        t137 = paddle._C_ops.add(t17, t136)

        # pd_op.layer_norm: (1x400x256xf32, 1x400xf32, 1x400xf32) <- (1x400x256xf32, 256xf32, 256xf32)
        t138, t139, t140 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(t137, t10, t11, float("1e-05"), 2),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del t11, t10

        # pd_op.matmul: (1x400x1024xf32) <- (1x400x256xf32, 256x1024xf32)
        t141 = paddle._C_ops.matmul(t138, t12, False, False)
        del t12

        # pd_op.add: (1x400x1024xf32) <- (1x400x1024xf32, 1024xf32)
        t142 = paddle._C_ops.add(t141, t13)
        del t13

        # pd_op.relu: (1x400x1024xf32) <- (1x400x1024xf32)
        t143 = paddle._C_ops.relu(t142)
        del t142

        # pd_op.matmul: (1x400x256xf32) <- (1x400x1024xf32, 1024x256xf32)
        t144 = paddle._C_ops.matmul(t143, t14, False, False)
        del t14

        # pd_op.add: (1x400x256xf32) <- (1x400x256xf32, 256xf32)
        t145 = paddle._C_ops.add(t144, t15)
        del t15

        # pd_op.add: (1x400x256xf32) <- (1x400x256xf32, 1x400x256xf32)
        t146 = paddle._C_ops.add(t138, t145)

        return (
            t16,
            t17,
            t18,
            t19,
            t20,
            t21,
            t22,
            t25,
            t26,
            t29,
            t30,
            t33,
            t37,
            t39,
            t40,
            t43,
            t44,
            t45,
            t48,
            t50,
            t51,
            t58,
            t59,
            t69,
            t76,
            t78,
            t79,
            t80,
            t81,
            t83,
            t84,
            t89,
            t94,
            t96,
            t97,
            t98,
            t103,
            t106,
            t108,
            t109,
            t110,
            t115,
            t118,
            t120,
            t121,
            t122,
            t123,
            t125,
            t127,
            t128,
            t129,
            t130,
            t131,
            t134,
            t135,
            t136,
            t137,
            t138,
            t139,
            t140,
            t141,
            t143,
            t144,
            t145,
            t146,
        )
