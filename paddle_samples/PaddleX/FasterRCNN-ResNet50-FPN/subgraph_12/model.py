import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [data_1]
        del data_1

        # pd_op.stack: (1xi64) <- ([xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.slice: (-1xi32) <- (2xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(data_0, [0], full_int_array_0, stack_0, [-1], [])
        del data_0, full_int_array_0, stack_0

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.gather: (-1xi32) <- (2xi32, -1xi32, 1xi32)
        gather_0 = paddle._C_ops.gather(data_2, slice_0, full_0)
        del data_2, full_0, slice_0

        return gather_0
