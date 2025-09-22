import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [-1]

        # pd_op.squeeze: (1xi32) <- (1x1xi32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(data_1, full_int_array_0)
        del data_1, full_int_array_0

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x4xf32, 1x4xf32]) <- (-1x4xf32, 1x4xf32)
        combine_0 = [data_2, data_0]
        del data_0, data_2

        # pd_op.concat: (-1x4xf32) <- ([-1x4xf32, 1x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)
        del combine_0, full_0

        return concat_0, squeeze_0
