import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
    ):
        # pd_op.full: (1xf64) <- ()
        t0 = paddle._C_ops.full([1], float("0"), paddle.float64, paddle.core.CPUPlace())

        # pd_op.full: (1xf64) <- ()
        t1 = paddle._C_ops.full(
            [1], float("24"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        t2 = paddle._C_ops.full([1], float("1"), paddle.float64, paddle.core.CPUPlace())

        # pd_op.arange: (24xf32) <- (1xf64, 1xf64, 1xf64)
        t3 = paddle.arange(t0, t1, t2, dtype="float32")
        del t1

        # builtin.combine: ([24xf32, 24xf32]) <- (24xf32, 24xf32)
        t4 = [t3, t3]
        del t3

        # pd_op.meshgrid: ([24x24xf32, 24x24xf32]) <- ([24xf32, 24xf32])
        t5 = paddle._C_ops.meshgrid(t4)
        del t4

        # builtin.split: (24x24xf32, 24x24xf32) <- ([24x24xf32, 24x24xf32])
        (
            t6,
            t7,
        ) = t5
        del t5

        # pd_op.full: (1xf64) <- ()
        t8 = paddle._C_ops.full(
            [1], float("256"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (256xf32) <- (1xf64, 1xf64, 1xf64)
        t9 = paddle.arange(t0, t8, t2, dtype="float32")
        del t0, t2, t8

        # pd_op.full: (1xf32) <- ()
        t10 = paddle._C_ops.full(
            [1], float("0.00390625"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (256xf32) <- (256xf32, 1xf32)
        t11 = paddle._C_ops.scale(t9, t10, float("0"), True)
        del t9, t10

        # pd_op.full: (256xf32) <- ()
        t12 = paddle._C_ops.full(
            [256],
            float("10000"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.elementwise_pow: (256xf32) <- (256xf32, 256xf32)
        t13 = paddle._C_ops.elementwise_pow(t12, t11)
        del t12, t11

        # pd_op.full: (256xf32) <- ()
        t14 = paddle._C_ops.full(
            [256],
            float("1"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.divide: (256xf32) <- (256xf32, 256xf32)
        t15 = paddle._C_ops.divide(t14, t13)
        del t13, t14

        # pd_op.flatten: (576xf32) <- (24x24xf32)
        t16 = paddle._C_ops.flatten(t6, 0, 1)
        del t6

        # pd_op.full_int_array: (1xi64) <- ()
        t17 = [1]

        # pd_op.unsqueeze: (576x1xf32) <- (576xf32, 1xi64)
        t18 = paddle._C_ops.unsqueeze(t16, t17)
        del t16

        # pd_op.full_int_array: (1xi64) <- ()
        t19 = [0]

        # pd_op.unsqueeze: (1x256xf32) <- (256xf32, 1xi64)
        t20 = paddle._C_ops.unsqueeze(t15, t19)
        del t15

        # pd_op.matmul: (576x256xf32) <- (576x1xf32, 1x256xf32)
        t21 = paddle._C_ops.matmul(t18, t20, False, False)
        del t18

        # pd_op.flatten: (576xf32) <- (24x24xf32)
        t22 = paddle._C_ops.flatten(t7, 0, 1)
        del t7

        # pd_op.unsqueeze: (576x1xf32) <- (576xf32, 1xi64)
        t23 = paddle._C_ops.unsqueeze(t22, t17)
        del t22, t17

        # pd_op.matmul: (576x256xf32) <- (576x1xf32, 1x256xf32)
        t24 = paddle._C_ops.matmul(t23, t20, False, False)
        del t20, t23

        # pd_op.sin: (576x256xf32) <- (576x256xf32)
        t25 = paddle._C_ops.sin(t21)

        # pd_op.cos: (576x256xf32) <- (576x256xf32)
        t26 = paddle._C_ops.cos(t21)
        del t21

        # pd_op.sin: (576x256xf32) <- (576x256xf32)
        t27 = paddle._C_ops.sin(t24)

        # pd_op.cos: (576x256xf32) <- (576x256xf32)
        t28 = paddle._C_ops.cos(t24)
        del t24

        # pd_op.full: (1xi32) <- ()
        t29 = paddle._C_ops.full([1], float("1"), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([576x256xf32, 576x256xf32, 576x256xf32, 576x256xf32]) <- (576x256xf32, 576x256xf32, 576x256xf32, 576x256xf32)
        t30 = [t25, t26, t27, t28]
        del t26, t28, t25, t27

        # pd_op.concat: (576x1024xf32) <- ([576x256xf32, 576x256xf32, 576x256xf32, 576x256xf32], 1xi32)
        t31 = paddle._C_ops.concat(t30, t29)
        del t30, t29

        return t19, t31
