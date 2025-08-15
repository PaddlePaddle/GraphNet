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
        # pd_op.shape64: (3xi64) <- (8x-1x196xf32)
        t0 = paddle._C_ops.shape64(input0)

        # pd_op.full_int_array: (1xi64) <- ()
        t1 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        t2 = [2]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t3 = paddle._C_ops.slice(t0, [0], t1, t2, [1], [0])
        del t0

        # pd_op.full: (1xi32) <- ()
        t4 = paddle._C_ops.full([1], float("9"), paddle.int32, paddle.core.CPUPlace())

        # pd_op.topk: (8x-1x9xf32, 8x-1x9xi64) <- (8x-1x196xf32, 1xi32)
        t5, t6 = (lambda x, f: f(x))(
            paddle._C_ops.topk(input0, t4, -1, False, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del input0

        # pd_op.full: (1xf32) <- ()
        t7 = paddle._C_ops.full([1], float("1"), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (8x-1x9xi64) <- (8x-1x9xi64, 1xf32)
        t8 = paddle._C_ops.scale(t6, t7, float("0"), True)

        # pd_op.full: (1xi32) <- ()
        t9 = paddle._C_ops.full([1], float("196"), paddle.int32, paddle.core.CPUPlace())

        # pd_op.one_hot: (8x-1x9x196xf32) <- (8x-1x9xi64, 1xi32)
        t10 = paddle._C_ops.one_hot(t6 % paddle.cast(t9, t6.dtype), t9)
        del t9, t6

        # pd_op.full_int_array: (1xi64) <- ()
        t11 = [-2]

        # pd_op.sum: (8x-1x196xf32) <- (8x-1x9x196xf32, 1xi64)
        t12 = paddle._C_ops.sum(t10, t11, None, False)
        del t10

        # pd_op.multiply: (8x-1x196xf32) <- (8x-1x196xf32, 8x-1x1xf32)
        t13 = paddle._C_ops.multiply(t12, input1)
        del t12

        # pd_op.shape64: (3xi64) <- (8x-1x784xf32)
        t14 = paddle._C_ops.shape64(input2)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t15 = paddle._C_ops.slice(t14, [0], t1, t2, [1], [0])
        del t14

        # pd_op.topk: (8x-1x9xf32, 8x-1x9xi64) <- (8x-1x784xf32, 1xi32)
        t16, t17 = (lambda x, f: f(x))(
            paddle._C_ops.topk(input2, t4, -1, False, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del input2

        # pd_op.scale: (8x-1x9xi64) <- (8x-1x9xi64, 1xf32)
        t18 = paddle._C_ops.scale(t17, t7, float("196"), True)

        # pd_op.full: (1xi32) <- ()
        t19 = paddle._C_ops.full(
            [1], float("784"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.one_hot: (8x-1x9x784xf32) <- (8x-1x9xi64, 1xi32)
        t20 = paddle._C_ops.one_hot(t17 % paddle.cast(t19, t17.dtype), t19)
        del t19, t17

        # pd_op.sum: (8x-1x784xf32) <- (8x-1x9x784xf32, 1xi64)
        t21 = paddle._C_ops.sum(t20, t11, None, False)
        del t20

        # pd_op.multiply: (8x-1x784xf32) <- (8x-1x784xf32, 8x-1x1xf32)
        t22 = paddle._C_ops.multiply(t21, input1)
        del t21

        # pd_op.shape64: (3xi64) <- (8x-1x3136xf32)
        t23 = paddle._C_ops.shape64(input3)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        t24 = paddle._C_ops.slice(t23, [0], t1, t2, [1], [0])
        del t1, t2, t23

        # pd_op.topk: (8x-1x9xf32, 8x-1x9xi64) <- (8x-1x3136xf32, 1xi32)
        t25, t26 = (lambda x, f: f(x))(
            paddle._C_ops.topk(input3, t4, -1, False, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del input3, t4

        # pd_op.scale: (8x-1x9xi64) <- (8x-1x9xi64, 1xf32)
        t27 = paddle._C_ops.scale(t26, t7, float("980"), True)
        del t7

        # pd_op.full: (1xi32) <- ()
        t28 = paddle._C_ops.full(
            [1], float("3136"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.one_hot: (8x-1x9x3136xf32) <- (8x-1x9xi64, 1xi32)
        t29 = paddle._C_ops.one_hot(t26 % paddle.cast(t28, t26.dtype), t28)
        del t28, t26

        # pd_op.sum: (8x-1x3136xf32) <- (8x-1x9x3136xf32, 1xi64)
        t30 = paddle._C_ops.sum(t29, t11, None, False)
        del t11, t29

        # pd_op.multiply: (8x-1x3136xf32) <- (8x-1x3136xf32, 8x-1x1xf32)
        t31 = paddle._C_ops.multiply(t30, input1)
        del input1, t30

        # pd_op.full: (1xi32) <- ()
        t32 = paddle._C_ops.full([1], float("-1"), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([8x-1x196xf32, 8x-1x784xf32, 8x-1x3136xf32]) <- (8x-1x196xf32, 8x-1x784xf32, 8x-1x3136xf32)
        t33 = [t13, t22, t31]
        del t13, t22, t31

        # pd_op.concat: (8x-1x4116xf32) <- ([8x-1x196xf32, 8x-1x784xf32, 8x-1x3136xf32], 1xi32)
        t34 = paddle._C_ops.concat(t33, t32)
        del t33

        # builtin.combine: ([8x-1x9xi64, 8x-1x9xi64, 8x-1x9xi64]) <- (8x-1x9xi64, 8x-1x9xi64, 8x-1x9xi64)
        t35 = [t8, t18, t27]
        del t8, t18, t27

        return t32, t34, t35
