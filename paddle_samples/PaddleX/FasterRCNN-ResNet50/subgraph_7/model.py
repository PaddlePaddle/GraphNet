import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2):
        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [-1, 4]

        # pd_op.reshape: (-1x4xf32) <- (-1x4xf32, 2xi64)
        reshape_0 = paddle._C_ops.reshape(data_2, full_int_array_0)
        del data_2, full_int_array_0

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x4xf32]) <- (-1x4xf32)
        combine_0 = [reshape_0]
        del reshape_0

        # pd_op.concat: (-1x4xf32) <- ([-1x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)
        del combine_0, full_0

        # pd_op.transpose: (2x-1x-1x15xf32) <- (2x15x-1x-1xf32)
        transpose_0 = paddle._C_ops.transpose(data_0, [0, 2, 3, 1])
        del data_0

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_1 = [2, -1, 1]

        # pd_op.reshape: (2x-1x1xf32) <- (2x-1x-1x15xf32, 3xi64)
        reshape_1 = paddle._C_ops.reshape(transpose_0, full_int_array_1)
        del full_int_array_1

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_0 = full_1

        # builtin.combine: ([2x-1x1xf32]) <- (2x-1x1xf32)
        combine_1 = [reshape_1]

        # pd_op.concat: (2x-1x1xf32) <- ([2x-1x1xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_1)
        del combine_1

        # pd_op.transpose: (2x-1x-1x60xf32) <- (2x60x-1x-1xf32)
        transpose_1 = paddle._C_ops.transpose(data_1, [0, 2, 3, 1])
        del data_1

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_2 = [2, -1, 4]

        # pd_op.reshape: (2x-1x4xf32) <- (2x-1x-1x60xf32, 3xi64)
        reshape_2 = paddle._C_ops.reshape(transpose_1, full_int_array_2)
        del full_int_array_2

        # builtin.combine: ([2x-1x4xf32]) <- (2x-1x4xf32)
        combine_2 = [reshape_2]

        # pd_op.concat: (2x-1x4xf32) <- ([2x-1x4xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_1)
        del assign_0, combine_2, full_1, reshape_1, reshape_2, transpose_0, transpose_1

        return concat_0, concat_1, concat_2
