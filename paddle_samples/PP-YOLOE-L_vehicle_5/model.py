import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4, data_5):
        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_0 = full_0

        # builtin.combine: ([2x-1x4xf32, 2x-1x4xf32, 2x-1x4xf32]) <- (2x-1x4xf32, 2x-1x4xf32, 2x-1x4xf32)
        combine_0 = [data_0, data_1, data_2]

        # pd_op.concat: (2x-1x4xf32) <- ([2x-1x4xf32, 2x-1x4xf32, 2x-1x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        # builtin.combine: ([2x-1x68xf32, 2x-1x68xf32, 2x-1x68xf32]) <- (2x-1x68xf32, 2x-1x68xf32, 2x-1x68xf32)
        combine_1 = [data_3, data_4, data_5]

        # pd_op.concat: (2x-1x68xf32) <- ([2x-1x68xf32, 2x-1x68xf32, 2x-1x68xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_0)
        return full_0, assign_0, concat_0, concat_1
