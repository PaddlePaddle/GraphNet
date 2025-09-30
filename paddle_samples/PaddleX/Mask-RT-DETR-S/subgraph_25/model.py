import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(data_0, full_0, float("1"), True)
        del full_0

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [data_0]
        del data_0

        # pd_op.stack: (1xi64) <- ([xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]
        del scale_0

        # pd_op.stack: (1xi64) <- ([xi64])
        stack_1 = paddle._C_ops.stack(combine_1, 0)
        del combine_1

        # pd_op.slice: (1x-1x-1x-1xf32) <- (-1x1x-1x-1x-1xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(data_1, [0], stack_0, stack_1, [-1], [0])
        del data_1, stack_0, stack_1

        return slice_0
