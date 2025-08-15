import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input0: paddle.Tensor,
        t0: paddle.Tensor,
        t1: paddle.Tensor,
        input1: paddle.Tensor,
        t2: paddle.Tensor,
        t3: paddle.Tensor,
        t4: paddle.Tensor,
        t5: paddle.Tensor,
        input2: paddle.Tensor,
        t6: paddle.Tensor,
    ):
        # pd_op.matmul: (1x9261x256xf32) <- (1x9261x256xf32, 256x256xf32)
        t7 = paddle._C_ops.matmul(input0, t0, False, False)
        del input0, t0

        # pd_op.add: (1x9261x256xf32) <- (1x9261x256xf32, 256xf32)
        t8 = paddle._C_ops.add(t7, t1)
        del t1

        # pd_op.full_int_array: (4xi64) <- ()
        t9 = [1, 9261, 8, 32]

        # pd_op.reshape: (1x9261x8x32xf32) <- (1x9261x256xf32, 4xi64)
        t10 = paddle._C_ops.reshape(t8, t9)
        del t9

        # pd_op.matmul: (1x498x192xf32) <- (1x498x256xf32, 256x192xf32)
        t11 = paddle._C_ops.matmul(input1, t2, False, False)
        del t2

        # pd_op.add: (1x498x192xf32) <- (1x498x192xf32, 192xf32)
        t12 = paddle._C_ops.add(t11, t3)
        del t3

        # pd_op.full_int_array: (6xi64) <- ()
        t13 = [1, 498, 8, 3, 4, 2]

        # pd_op.reshape: (1x498x8x3x4x2xf32) <- (1x498x192xf32, 6xi64)
        t14 = paddle._C_ops.reshape(t12, t13)
        del t13

        # pd_op.matmul: (1x498x96xf32) <- (1x498x256xf32, 256x96xf32)
        t15 = paddle._C_ops.matmul(input1, t4, False, False)
        del input1, t4

        # pd_op.add: (1x498x96xf32) <- (1x498x96xf32, 96xf32)
        t16 = paddle._C_ops.add(t15, t5)
        del t5

        # pd_op.full_int_array: (4xi64) <- ()
        t17 = [1, 498, 8, 12]

        # pd_op.reshape: (1x498x8x12xf32) <- (1x498x96xf32, 4xi64)
        t18 = paddle._C_ops.reshape(t16, t17)
        del t17

        # pd_op.softmax: (1x498x8x12xf32) <- (1x498x8x12xf32)
        t19 = paddle._C_ops.softmax(t18, -1)
        del t18

        # pd_op.full_int_array: (5xi64) <- ()
        t20 = [1, 498, 8, 3, 4]

        # pd_op.reshape: (1x498x8x3x4xf32) <- (1x498x8x12xf32, 5xi64)
        t21 = paddle._C_ops.reshape(t19, t20)
        del t20

        # pd_op.full_int_array: (1xi64) <- ()
        t22 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        t23 = t22

        # pd_op.full_int_array: (1xi64) <- ()
        t24 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        t25 = t24

        # pd_op.assign: (1xi64) <- (1xi64)
        t26 = t24

        # pd_op.slice: (1x498x1x2xf32) <- (1x498x1x4xf32, 1xi64, 1xi64)
        t27 = paddle._C_ops.slice(input2, [3], t22, t24, [1], [])

        # pd_op.full_int_array: (2xi64) <- ()
        t28 = [2, 4]

        # pd_op.unsqueeze: (1x498x1x1x1x2xf32) <- (1x498x1x2xf32, 2xi64)
        t29 = paddle._C_ops.unsqueeze(t27, t28)
        del t27

        # pd_op.full: (1xf32) <- ()
        t30 = paddle._C_ops.full(
            [1], float("0.25"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x498x8x3x4x2xf32) <- (1x498x8x3x4x2xf32, 1xf32)
        t31 = paddle._C_ops.scale(t14, t30, float("0"), True)
        del t14

        # pd_op.full_int_array: (1xi64) <- ()
        t32 = [2147483647]

        # pd_op.slice: (1x498x1x2xf32) <- (1x498x1x4xf32, 1xi64, 1xi64)
        t33 = paddle._C_ops.slice(input2, [3], t24, t32, [1], [])
        del input2, t32

        # pd_op.unsqueeze: (1x498x1x1x1x2xf32) <- (1x498x1x2xf32, 2xi64)
        t34 = paddle._C_ops.unsqueeze(t33, t28)
        del t28, t33

        # pd_op.multiply: (1x498x8x3x4x2xf32) <- (1x498x8x3x4x2xf32, 1x498x1x1x1x2xf32)
        t35 = paddle._C_ops.multiply(t31, t34)

        # pd_op.full: (1xf32) <- ()
        t36 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x498x8x3x4x2xf32) <- (1x498x8x3x4x2xf32, 1xf32)
        t37 = paddle._C_ops.scale(t35, t36, float("0"), True)
        del t35

        # pd_op.add: (1x498x8x3x4x2xf32) <- (1x498x1x1x1x2xf32, 1x498x8x3x4x2xf32)
        t38 = paddle._C_ops.add(t29, t37)

        # pd_op.full: (3x2xi64) <- ()
        t39 = paddle._C_ops.full(
            [3, 2], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (3x2xi64) <- (3x2xi64)
        t40 = paddle._C_ops.assign_value_(
            t39,
            [3, 2],
            paddle.int64,
            [
                float("84"),
                float("84"),
                float("42"),
                float("42"),
                float("21"),
                float("21"),
            ],
            paddle.framework._current_expected_place(),
        )
        del t39

        # pd_op.full: (3xi64) <- ()
        t41 = paddle._C_ops.full(
            [3], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (3xi64) <- (3xi64)
        t42 = paddle._C_ops.assign_value_(
            t41,
            [3],
            paddle.int64,
            [float("0"), float("7056"), float("8820")],
            paddle.framework._current_expected_place(),
        )
        del t41

        # pd_op.full_int_array: (1xi64) <- ()
        t43 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        t44 = t43

        # pd_op.assign: (1xi64) <- (1xi64)
        t45 = t43

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        t46 = paddle._C_ops.slice(t40, [0], t22, t43, [1], [0])

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        t47 = paddle._C_ops.slice(t46, [0], t22, t43, [1], [0])

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        t48 = paddle._C_ops.slice(t46, [0], t43, t24, [1], [0])
        del t46

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        t49 = paddle._C_ops.multiply(t47, t48)
        del t47, t48

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        t50 = paddle._C_ops.slice(t40, [0], t43, t24, [1], [0])

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        t51 = paddle._C_ops.slice(t50, [0], t22, t43, [1], [0])

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        t52 = paddle._C_ops.slice(t50, [0], t43, t24, [1], [0])
        del t50

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        t53 = paddle._C_ops.multiply(t51, t52)
        del t51, t52

        # pd_op.full_int_array: (1xi64) <- ()
        t54 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        t55 = t54

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        t56 = paddle._C_ops.slice(t40, [0], t24, t54, [1], [0])

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        t57 = paddle._C_ops.slice(t56, [0], t22, t43, [1], [0])

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        t58 = paddle._C_ops.slice(t56, [0], t43, t24, [1], [0])
        del t56

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        t59 = paddle._C_ops.multiply(t57, t58)
        del t58, t57

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        t60 = [t49, t53, t59]
        del t49, t53, t59

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        t61 = paddle._C_ops.stack(t60, 0)
        del t60

        # pd_op.full: (1xi32) <- ()
        t62 = paddle._C_ops.full([1], float("1"), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32]) <- (1x9261x8x32xf32, 3xi64, 1xi32)
        t63 = paddle._C_ops.split(t10, t61, t62)
        del t10, t61

        # builtin.split: (-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32) <- ([-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32])
        (
            t64,
            t65,
            t66,
        ) = t63
        del t63

        # pd_op.full: (1xf32) <- ()
        t67 = paddle._C_ops.full(
            [1], float("2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x498x8x3x4x2xf32) <- (1x498x8x3x4x2xf32, 1xf32)
        t68 = paddle._C_ops.scale(t38, t67, float("0"), True)
        del t38

        # pd_op.full: (1xf32) <- ()
        t69 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x498x8x3x4x2xf32) <- (1x498x8x3x4x2xf32, 1xf32)
        t70 = paddle._C_ops.scale(t68, t69, float("-1"), True)
        del t68

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        t71 = paddle._C_ops.slice(t40, [0], t22, t43, [1], [0])

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        t72 = paddle._C_ops.slice(t71, [0], t22, t43, [1], [0])

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        t73 = paddle._C_ops.slice(t71, [0], t43, t24, [1], [0])
        del t71

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        t74 = paddle._C_ops.flatten(t64, 2, 3)

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        t75 = paddle._C_ops.transpose(t74, [0, 2, 1])
        del t74

        # pd_op.full: (xi64) <- ()
        t76 = paddle._C_ops.full([], float("8"), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (xi64) <- ()
        t77 = paddle._C_ops.full([], float("32"), paddle.int64, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t78 = [t76, t77, t72, t73]
        del t72, t73

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t79 = paddle._C_ops.stack(t78, 0)
        del t78

        # pd_op.reshape: (8x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        t80 = paddle._C_ops.reshape(t75, t79)
        del t79

        # pd_op.slice: (1x498x8x4x2xf32) <- (1x498x8x3x4x2xf32, 1xi64, 1xi64)
        t81 = paddle._C_ops.slice(t70, [3], t22, t43, [1], [3])

        # pd_op.transpose: (1x8x498x4x2xf32) <- (1x498x8x4x2xf32)
        t82 = paddle._C_ops.transpose(t81, [0, 2, 1, 3, 4])
        del t81

        # pd_op.flatten: (8x498x4x2xf32) <- (1x8x498x4x2xf32)
        t83 = paddle._C_ops.flatten(t82, 0, 1)

        # pd_op.grid_sample: (8x32x498x4xf32) <- (8x32x-1x-1xf32, 8x498x4x2xf32)
        t84 = paddle._C_ops.grid_sample(t80, t83, "bilinear", "zeros", False)

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        t85 = paddle._C_ops.slice(t40, [0], t43, t24, [1], [0])

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        t86 = paddle._C_ops.slice(t85, [0], t22, t43, [1], [0])

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        t87 = paddle._C_ops.slice(t85, [0], t43, t24, [1], [0])
        del t85

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        t88 = paddle._C_ops.flatten(t65, 2, 3)

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        t89 = paddle._C_ops.transpose(t88, [0, 2, 1])
        del t88

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t90 = [t76, t77, t86, t87]
        del t86, t87

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t91 = paddle._C_ops.stack(t90, 0)
        del t90

        # pd_op.reshape: (8x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        t92 = paddle._C_ops.reshape(t89, t91)
        del t91

        # pd_op.slice: (1x498x8x4x2xf32) <- (1x498x8x3x4x2xf32, 1xi64, 1xi64)
        t93 = paddle._C_ops.slice(t70, [3], t43, t24, [1], [3])

        # pd_op.transpose: (1x8x498x4x2xf32) <- (1x498x8x4x2xf32)
        t94 = paddle._C_ops.transpose(t93, [0, 2, 1, 3, 4])
        del t93

        # pd_op.flatten: (8x498x4x2xf32) <- (1x8x498x4x2xf32)
        t95 = paddle._C_ops.flatten(t94, 0, 1)

        # pd_op.grid_sample: (8x32x498x4xf32) <- (8x32x-1x-1xf32, 8x498x4x2xf32)
        t96 = paddle._C_ops.grid_sample(t92, t95, "bilinear", "zeros", False)

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        t97 = paddle._C_ops.slice(t40, [0], t24, t54, [1], [0])
        del t40

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        t98 = paddle._C_ops.slice(t97, [0], t22, t43, [1], [0])
        del t22

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        t99 = paddle._C_ops.slice(t97, [0], t43, t24, [1], [0])
        del t43, t97

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        t100 = paddle._C_ops.flatten(t66, 2, 3)

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        t101 = paddle._C_ops.transpose(t100, [0, 2, 1])
        del t100

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        t102 = [t76, t77, t98, t99]
        del t76, t77, t98, t99

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        t103 = paddle._C_ops.stack(t102, 0)
        del t102

        # pd_op.reshape: (8x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        t104 = paddle._C_ops.reshape(t101, t103)
        del t103

        # pd_op.slice: (1x498x8x4x2xf32) <- (1x498x8x3x4x2xf32, 1xi64, 1xi64)
        t105 = paddle._C_ops.slice(t70, [3], t24, t54, [1], [3])
        del t54, t24

        # pd_op.transpose: (1x8x498x4x2xf32) <- (1x498x8x4x2xf32)
        t106 = paddle._C_ops.transpose(t105, [0, 2, 1, 3, 4])
        del t105

        # pd_op.flatten: (8x498x4x2xf32) <- (1x8x498x4x2xf32)
        t107 = paddle._C_ops.flatten(t106, 0, 1)

        # pd_op.grid_sample: (8x32x498x4xf32) <- (8x32x-1x-1xf32, 8x498x4x2xf32)
        t108 = paddle._C_ops.grid_sample(t104, t107, "bilinear", "zeros", False)

        # pd_op.transpose: (1x8x498x3x4xf32) <- (1x498x8x3x4xf32)
        t109 = paddle._C_ops.transpose(t21, [0, 2, 1, 3, 4])
        del t21

        # pd_op.full_int_array: (4xi64) <- ()
        t110 = [8, 1, 498, 12]

        # pd_op.reshape: (8x1x498x12xf32) <- (1x8x498x3x4xf32, 4xi64)
        t111 = paddle._C_ops.reshape(t109, t110)
        del t110

        # builtin.combine: ([8x32x498x4xf32, 8x32x498x4xf32, 8x32x498x4xf32]) <- (8x32x498x4xf32, 8x32x498x4xf32, 8x32x498x4xf32)
        t112 = [t84, t96, t108]

        # pd_op.stack: (8x32x498x3x4xf32) <- ([8x32x498x4xf32, 8x32x498x4xf32, 8x32x498x4xf32])
        t113 = paddle._C_ops.stack(t112, -2)
        del t112

        # pd_op.flatten: (8x32x498x12xf32) <- (8x32x498x3x4xf32)
        t114 = paddle._C_ops.flatten(t113, 3, 4)

        # pd_op.multiply: (8x32x498x12xf32) <- (8x32x498x12xf32, 8x1x498x12xf32)
        t115 = paddle._C_ops.multiply(t114, t111)

        # pd_op.full_int_array: (1xi64) <- ()
        t116 = [-1]

        # pd_op.sum: (8x32x498xf32) <- (8x32x498x12xf32, 1xi64)
        t117 = paddle._C_ops.sum(t115, t116, None, False)

        # pd_op.full_int_array: (3xi64) <- ()
        t118 = [1, 256, 498]

        # pd_op.reshape: (1x256x498xf32) <- (8x32x498xf32, 3xi64)
        t119 = paddle._C_ops.reshape(t117, t118)
        del t118

        # pd_op.transpose: (1x498x256xf32) <- (1x256x498xf32)
        t120 = paddle._C_ops.transpose(t119, [0, 2, 1])
        del t119

        # pd_op.matmul: (1x498x256xf32) <- (1x498x256xf32, 256x256xf32)
        t121 = paddle._C_ops.matmul(t120, t6, False, False)
        del t6

        return (
            t7,
            t8,
            t11,
            t12,
            t15,
            t16,
            t19,
            t23,
            t25,
            t26,
            t29,
            t30,
            t31,
            t34,
            t36,
            t37,
            t44,
            t45,
            t55,
            t62,
            t64,
            t65,
            t66,
            t67,
            t69,
            t70,
            t75,
            t80,
            t82,
            t83,
            t84,
            t89,
            t92,
            t94,
            t95,
            t96,
            t101,
            t104,
            t106,
            t107,
            t108,
            t109,
            t111,
            t113,
            t114,
            t115,
            t116,
            t117,
            t120,
            t121,
        )
