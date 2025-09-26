import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (1x-1xf32) <- (1x-1x-1xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_2, [2], full_int_array_0, full_int_array_1, [1], [2]
        )

        # pd_op.sum: (1xf32) <- (1x-1xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(slice_0, full_int_array_1, None, False)
        del slice_0

        # pd_op.slice: (1x-1xf32) <- (1x-1x-1xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_2, [1], full_int_array_0, full_int_array_1, [1], [1]
        )
        del data_2, full_int_array_0

        # pd_op.sum: (1xf32) <- (1x-1xf32, 1xi64)
        sum_1 = paddle._C_ops.sum(slice_1, full_int_array_1, None, False)
        del full_int_array_1, slice_1

        # pd_op.cast: (xf32) <- (xi64)
        cast_0 = paddle._C_ops.cast(data_0, paddle.float32)
        del data_0

        # pd_op.divide: (1xf32) <- (1xf32, xf32)
        divide_0 = paddle._C_ops.divide(sum_0, cast_0)
        del cast_0, sum_0

        # pd_op.cast: (xf32) <- (xi64)
        cast_1 = paddle._C_ops.cast(data_1, paddle.float32)
        del data_1

        # pd_op.divide: (1xf32) <- (1xf32, xf32)
        divide_1 = paddle._C_ops.divide(sum_1, cast_1)
        del cast_1, sum_1

        # builtin.combine: ([1xf32, 1xf32]) <- (1xf32, 1xf32)
        combine_0 = [divide_1, divide_0]
        del divide_0, divide_1

        # pd_op.stack: (1x2xf32) <- ([1xf32, 1xf32])
        stack_0 = paddle._C_ops.stack(combine_0, -1)
        del combine_0

        return stack_0
