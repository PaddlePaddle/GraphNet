import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (8xf32) <- (8x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_0, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [3]

        # pd_op.slice: (8xf32) <- (8x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_0, [1], full_int_array_2, full_int_array_3, [1], [1]
        )

        # pd_op.add: (8xf32) <- (8xf32, 8xf32)
        add_0 = paddle._C_ops.add(slice_0, slice_1)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (8xf32) <- (8xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(add_0, full_0, float("0"), True)

        # pd_op.slice: (8xf32) <- (8x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            data_0, [1], full_int_array_1, full_int_array_2, [1], [1]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [4]

        # pd_op.slice: (8xf32) <- (8x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            data_0, [1], full_int_array_3, full_int_array_4, [1], [1]
        )

        # pd_op.add: (8xf32) <- (8xf32, 8xf32)
        add_1 = paddle._C_ops.add(slice_2, slice_3)

        # pd_op.scale: (8xf32) <- (8xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(add_1, full_0, float("0"), True)

        # builtin.combine: ([8xf32, 8xf32]) <- (8xf32, 8xf32)
        combine_0 = [scale_0, scale_1]

        # pd_op.stack: (8x2xf32) <- ([8xf32, 8xf32])
        stack_0 = paddle._C_ops.stack(combine_0, -1)
        return stack_0
