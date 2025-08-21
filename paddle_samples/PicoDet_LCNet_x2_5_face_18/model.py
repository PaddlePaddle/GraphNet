import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4):
        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(data_0, full_0, float("0"), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [scale_0]

        # pd_op.stack: (1xi64) <- ([xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)

        # pd_op.slice: (4x-1xi64) <- (4x-1xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(data_4, [1], full_int_array_0, stack_0, [-1], [])

        # pd_op.squeeze: (4x-1xi64) <- (4x-1xi64, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(slice_0, full_int_array_0)

        # pd_op.add: (xi64) <- (xi64, xi64)
        add_0 = paddle._C_ops.add(scale_0, data_1)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [add_0]

        # pd_op.stack: (1xi64) <- ([xi64])
        stack_1 = paddle._C_ops.stack(combine_1, 0)

        # pd_op.slice: (4x-1xi64) <- (4x-1xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(data_4, [1], stack_0, stack_1, [-1], [])

        # pd_op.squeeze: (4x-1xi64) <- (4x-1xi64, 1xi64)
        squeeze_1 = paddle._C_ops.squeeze(slice_1, full_int_array_0)

        # pd_op.add: (xi64) <- (xi64, xi64)
        add_1 = paddle._C_ops.add(add_0, data_2)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_2 = [add_1]

        # pd_op.stack: (1xi64) <- ([xi64])
        stack_2 = paddle._C_ops.stack(combine_2, 0)

        # pd_op.slice: (4x-1xi64) <- (4x-1xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(data_4, [1], stack_1, stack_2, [-1], [])

        # pd_op.squeeze: (4x-1xi64) <- (4x-1xi64, 1xi64)
        squeeze_2 = paddle._C_ops.squeeze(slice_2, full_int_array_0)

        # pd_op.add: (xi64) <- (xi64, xi64)
        add_2 = paddle._C_ops.add(add_1, data_3)

        # builtin.combine: ([xi64]) <- (xi64)
        combine_3 = [add_2]

        # pd_op.stack: (1xi64) <- ([xi64])
        stack_3 = paddle._C_ops.stack(combine_3, 0)

        # pd_op.slice: (4x-1xi64) <- (4x-1xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(data_4, [1], stack_2, stack_3, [-1], [])

        # pd_op.squeeze: (4x-1xi64) <- (4x-1xi64, 1xi64)
        squeeze_3 = paddle._C_ops.squeeze(slice_3, full_int_array_0)
        return squeeze_0, squeeze_1, squeeze_2, squeeze_3
