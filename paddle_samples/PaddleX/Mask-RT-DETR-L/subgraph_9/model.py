import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [-1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_3 = full_int_array_1

        # pd_op.slice: (7x2x300x4xf32) <- (8x2x300x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_0, [0], full_int_array_0, full_int_array_1, [1], []
        )
        del data_0

        # pd_op.slice: (7x2x300x2xf32) <- (8x2x300x2xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_1, [0], full_int_array_0, full_int_array_1, [1], []
        )
        del data_1

        # pd_op.slice: (7x2x300x192x192xf32) <- (8x2x300x192x192xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            data_2, [0], full_int_array_0, full_int_array_1, [1], []
        )
        del (
            assign_0,
            assign_1,
            assign_2,
            assign_3,
            data_2,
            full_int_array_0,
            full_int_array_1,
        )

        return slice_0, slice_1, slice_2
