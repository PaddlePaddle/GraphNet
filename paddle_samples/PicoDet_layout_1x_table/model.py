import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self, input0: paddle.Tensor, input1: paddle.Tensor, input2: paddle.Tensor
    ):
        # pd_op.full: (1xf32) <- ()
        t0 = paddle._C_ops.full([1], float("1"), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        t1 = paddle._C_ops.scale(input0, t0, float("1"), True)
        del t0

        # builtin.combine: ([xi64]) <- (xi64)
        t2 = [input0]
        del input0

        # pd_op.stack: (1xi64) <- ([xi64])
        t3 = paddle._C_ops.stack(t2, 0)
        del t2

        # builtin.combine: ([xi64]) <- (xi64)
        t4 = [t1]
        del t1

        # pd_op.stack: (1xi64) <- ([xi64])
        t5 = paddle._C_ops.stack(t4, 0)
        del t4

        # pd_op.slice: (8337xf32) <- (8337x4xf32, 1xi64, 1xi64)
        t6 = paddle._C_ops.slice(input1, [1], t3, t5, [-1], [1])
        del input1

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        t7 = paddle._C_ops.slice(input2, [0], t3, t5, [-1], [0])
        del input2, t3, t5

        return t6, t7
