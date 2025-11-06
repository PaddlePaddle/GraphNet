import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [-1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_1

        # pd_op.slice: (5x4x100x4xf32) <- (6x4x100x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_0, [0], full_int_array_0, full_int_array_1, [1], []
        )
        del data_0

        # pd_op.slice: (5x4x100x5xf32) <- (6x4x100x5xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_1, [0], full_int_array_0, full_int_array_1, [1], []
        )
        del assign_0, assign_1, data_1, full_int_array_0, full_int_array_1

        return slice_0, slice_1
