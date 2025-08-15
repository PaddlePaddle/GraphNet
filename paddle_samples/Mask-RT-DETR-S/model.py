import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input0: paddle.Tensor, input1: paddle.Tensor):
        # pd_op.full_int_array: (1xi64) <- ()
        t0 = [-1]

        # pd_op.assign: (1xi64) <- (1xi64)
        t1 = t0

        # pd_op.assign: (1xi64) <- (1xi64)
        t2 = t0

        # pd_op.full_int_array: (1xi64) <- ()
        t3 = [2147483647]

        # pd_op.assign: (1xi64) <- (1xi64)
        t4 = t3

        # pd_op.assign: (1xi64) <- (1xi64)
        t5 = t3

        # pd_op.slice: (1x-1x4xf32) <- (-1x1x-1x4xf32, 1xi64, 1xi64)
        t6 = paddle._C_ops.slice(input0, [0], t0, t3, [1], [0])
        del input0

        # pd_op.slice: (1x-1x2xf32) <- (-1x1x-1x2xf32, 1xi64, 1xi64)
        t7 = paddle._C_ops.slice(input1, [0], t0, t3, [1], [0])
        del input1

        return t0, t1, t2, t3, t4, t5, t6, t7
