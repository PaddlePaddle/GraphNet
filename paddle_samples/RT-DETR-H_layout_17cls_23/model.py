import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2):
        # pd_op.add: (1x-1x256xf32) <- (1x-1x256xf32, 1x-1x256xf32)
        add_0 = paddle._C_ops.add(data_0, data_2)

        # pd_op.shape64: (2xi64) <- (-1x-1xb)
        shape64_0 = paddle._C_ops.shape64(data_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_0 = [slice_0, slice_1]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_with_tensor: (-1x-1xf32) <- (1xf32, 2xi64)
        full_with_tensor_0 = paddle._C_ops.full_with_tensor(
            full_0, stack_0, paddle.float32
        )

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("-inf"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_with_tensor: (-1x-1xf32) <- (1xf32, 2xi64)
        full_with_tensor_1 = paddle._C_ops.full_with_tensor(
            full_1, stack_0, paddle.float32
        )

        # pd_op.where: (-1x-1xf32) <- (-1x-1xb, -1x-1xf32, -1x-1xf32)
        where_0 = paddle._C_ops.where(data_1, full_with_tensor_0, full_with_tensor_1)
        return add_0, where_0
