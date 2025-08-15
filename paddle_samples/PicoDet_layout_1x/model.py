import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input0: paddle.Tensor):
        # pd_op.full_int_array: (1xi64) <- ()
        t0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        t1 = [7600]

        # pd_op.slice: (24x7600x4xf32) <- (24x10105x4xf32, 1xi64, 1xi64)
        t2 = paddle._C_ops.slice(input0, [1], t0, t1, [1], [])

        # pd_op.squeeze: (24x7600x4xf32) <- (24x7600x4xf32, 1xi64)
        t3 = paddle._C_ops.squeeze(t2, t0)
        del t2

        # pd_op.full_int_array: (1xi64) <- ()
        t4 = [9500]

        # pd_op.slice: (24x1900x4xf32) <- (24x10105x4xf32, 1xi64, 1xi64)
        t5 = paddle._C_ops.slice(input0, [1], t1, t4, [1], [])
        del t1

        # pd_op.squeeze: (24x1900x4xf32) <- (24x1900x4xf32, 1xi64)
        t6 = paddle._C_ops.squeeze(t5, t0)
        del t5

        # pd_op.full_int_array: (1xi64) <- ()
        t7 = [9975]

        # pd_op.slice: (24x475x4xf32) <- (24x10105x4xf32, 1xi64, 1xi64)
        t8 = paddle._C_ops.slice(input0, [1], t4, t7, [1], [])
        del t4

        # pd_op.squeeze: (24x475x4xf32) <- (24x475x4xf32, 1xi64)
        t9 = paddle._C_ops.squeeze(t8, t0)
        del t8

        # pd_op.full_int_array: (1xi64) <- ()
        t10 = [2147483647]

        # pd_op.slice: (24x130x4xf32) <- (24x10105x4xf32, 1xi64, 1xi64)
        t11 = paddle._C_ops.slice(input0, [1], t7, t10, [1], [])
        del input0, t7, t10

        return t0, t3, t6, t9, t11
