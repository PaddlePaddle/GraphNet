import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0):
        # pd_op.shape64: (4xi64) <- (-1x-1x-1x-1xf32)
        shape64_0 = paddle._C_ops.shape64(data_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [3]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [4]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("7"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        floor_divide_0 = paddle._C_ops.floor_divide(slice_1, full_0)

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        floor_divide_1 = paddle._C_ops.floor_divide(slice_2, full_0)

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("-1"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_2 = paddle._C_ops.full(
            [], float("7"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        combine_0 = [full_1, floor_divide_0, full_2, floor_divide_1, full_2, slice_3]

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)

        # pd_op.reshape: (-1x-1x7x-1x7x-1xf32) <- (-1x-1x-1x-1xf32, 6xi64)
        reshape_1 = paddle._C_ops.reshape(data_0, stack_0)

        # pd_op.transpose: (-1x-1x-1x7x7x-1xf32) <- (-1x-1x7x-1x7x-1xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_1, [0, 1, 3, 2, 4, 5])

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_1 = [full_1, full_2, full_2, slice_3]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_1, 0)

        # pd_op.reshape: (-1x7x7x-1xf32) <- (-1x-1x-1x7x7x-1xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(transpose_0, stack_1)
        return transpose_0, reshape_0
