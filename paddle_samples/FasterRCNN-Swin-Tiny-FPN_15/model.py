import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        parameter_0,
        parameter_1,
        parameter_2,
        parameter_3,
        data_0,
        data_1,
        data_2,
        data_3,
    ):
        # pd_op.shape64: (3xi64) <- (-1x49x-1xf32)
        shape64_0 = paddle._C_ops.shape64(data_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_6 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_1

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_4 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_3 = full_int_array_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_5 = full_int_array_3

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.matmul: (-1x49x288xf32) <- (-1x49x-1xf32, 96x288xf32)
        matmul_0 = paddle._C_ops.matmul(data_1, parameter_3, False, False)

        # pd_op.add: (-1x49x288xf32) <- (-1x49x288xf32, 288xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_2)

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        floor_divide_0 = paddle._C_ops.floor_divide(slice_2, data_0)

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("-1"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_2 = paddle._C_ops.full(
            [], float("49"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_3 = paddle._C_ops.full(
            [], float("3"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_0 = [full_1, full_2, full_3, data_0, floor_divide_0]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)

        # pd_op.reshape: (-1x49x3x-1x-1xf32) <- (-1x49x288xf32, 5xi64)
        reshape_1 = paddle._C_ops.reshape(add_0, stack_0)

        # pd_op.transpose: (3x-1x-1x49x-1xf32) <- (-1x49x3x-1x-1xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_1, [2, 0, 3, 1, 4])

        # pd_op.slice: (-1x-1x49x-1xf32) <- (3x-1x-1x49x-1xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            transpose_0, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (-1x-1x49x-1xf32) <- (3x-1x-1x49x-1xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            transpose_0, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (-1x-1x49x-1xf32) <- (3x-1x-1x49x-1xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            transpose_0, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x-1x49x-1xf32) <- (-1x-1x49x-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_3, full_0, float("0"), True)

        # pd_op.transpose: (-1x-1x-1x49xf32) <- (-1x-1x49x-1xf32)
        transpose_1 = paddle._C_ops.transpose(slice_4, [0, 1, 3, 2])

        # pd_op.matmul: (-1x-1x49x49xf32) <- (-1x-1x49x-1xf32, -1x-1x-1x49xf32)
        matmul_1 = paddle._C_ops.matmul(scale_0, transpose_1, False, False)

        # pd_op.flatten: (2401xi64) <- (49x49xi64)
        flatten_0 = paddle._C_ops.flatten(data_2, 0, 1)

        # pd_op.index_select: (2401x-1xf32) <- (169x-1xf32, 2401xi64)
        index_select_0 = paddle._C_ops.index_select(data_3, flatten_0, 0)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_4 = [49, 49, -1]

        # pd_op.reshape: (49x49x-1xf32) <- (2401x-1xf32, 3xi64)
        reshape_2 = paddle._C_ops.reshape(index_select_0, full_int_array_4)

        # pd_op.transpose: (-1x49x49xf32) <- (49x49x-1xf32)
        transpose_2 = paddle._C_ops.transpose(reshape_2, [2, 0, 1])

        # pd_op.unsqueeze: (1x-1x49x49xf32) <- (-1x49x49xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(transpose_2, full_int_array_0)

        # pd_op.add: (-1x-1x49x49xf32) <- (-1x-1x49x49xf32, 1x-1x49x49xf32)
        add_2 = paddle._C_ops.add(matmul_1, unsqueeze_0)

        # pd_op.softmax: (-1x-1x49x49xf32) <- (-1x-1x49x49xf32)
        softmax_0 = paddle._C_ops.softmax(add_2, -1)

        # pd_op.matmul: (-1x-1x49x-1xf32) <- (-1x-1x49x49xf32, -1x-1x49x-1xf32)
        matmul_3 = paddle._C_ops.matmul(softmax_0, slice_0, False, False)

        # pd_op.transpose: (-1x49x-1x-1xf32) <- (-1x-1x49x-1xf32)
        transpose_3 = paddle._C_ops.transpose(matmul_3, [0, 2, 1, 3])

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_1 = [full_1, full_2, slice_2]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_1, 0)

        # pd_op.reshape: (-1x49x-1xf32) <- (-1x49x-1x-1xf32, 3xi64)
        reshape_0 = paddle._C_ops.reshape(transpose_3, stack_1)

        # pd_op.matmul: (-1x49x96xf32) <- (-1x49x-1xf32, 96x96xf32)
        matmul_2 = paddle._C_ops.matmul(reshape_0, parameter_1, False, False)

        # pd_op.add: (-1x49x96xf32) <- (-1x49x96xf32, 96xf32)
        add_1 = paddle._C_ops.add(matmul_2, parameter_0)
        return (
            matmul_0,
            add_0,
            transpose_0,
            assign_0,
            assign_1,
            assign_2,
            assign_3,
            assign_4,
            assign_5,
            slice_0,
            full_0,
            scale_0,
            transpose_1,
            matmul_1,
            flatten_0,
            index_select_0,
            transpose_2,
            assign_6,
            unsqueeze_0,
            softmax_0,
            transpose_3,
            reshape_0,
            matmul_2,
            add_1,
        )
