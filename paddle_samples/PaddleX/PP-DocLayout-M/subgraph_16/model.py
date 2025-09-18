import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4):
        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_0 = [data_0, data_1, data_2, data_3]
        del data_0, data_1, data_2, data_3

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("2"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split: ([-1x-1x-1xf32, -1x-1x-1xf32, -1x-1x-1xf32, -1x-1x-1xf32]) <- (1x-1x-1xf32, 4xi64, 1xi32)
        split_4 = paddle._C_ops.split(data_4, stack_0, full_0)
        del data_4, full_0, stack_0

        # builtin.split: (-1x-1x-1xf32, -1x-1x-1xf32, -1x-1x-1xf32, -1x-1x-1xf32) <- ([-1x-1x-1xf32, -1x-1x-1xf32, -1x-1x-1xf32, -1x-1x-1xf32])
        (
            split_0,
            split_1,
            split_2,
            split_3,
        ) = split_4
        del split_4

        return split_0, split_1, split_2, split_3
