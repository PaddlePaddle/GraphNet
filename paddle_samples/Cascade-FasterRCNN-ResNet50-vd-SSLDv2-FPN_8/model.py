import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [3]

        # pd_op.slice: (512xf32) <- (512x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_0, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.slice: (512xf32) <- (512x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_0, [1], full_int_array_2, full_int_array_3, [1], [1]
        )

        # pd_op.subtract: (512xf32) <- (512xf32, 512xf32)
        subtract_0 = paddle._C_ops.subtract(slice_0, slice_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [4]

        # pd_op.slice: (512xf32) <- (512x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            data_0, [1], full_int_array_1, full_int_array_4, [1], [1]
        )

        # pd_op.slice: (512xf32) <- (512x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            data_0, [1], full_int_array_3, full_int_array_0, [1], [1]
        )

        # pd_op.subtract: (512xf32) <- (512xf32, 512xf32)
        subtract_1 = paddle._C_ops.subtract(slice_2, slice_3)

        # pd_op.full: (xf32) <- ()
        full_0 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (512xb) <- (512xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(subtract_1, full_0)

        # pd_op.greater_than: (512xb) <- (512xf32, xf32)
        greater_than_1 = paddle._C_ops.greater_than(subtract_0, full_0)

        # pd_op.logical_and: (512xb) <- (512xb, 512xb)
        logical_and_0 = paddle._C_ops.logical_and(greater_than_0, greater_than_1)

        # pd_op.nonzero: (-1x1xi64) <- (512xb)
        nonzero_0 = paddle._C_ops.nonzero(logical_and_0)

        # pd_op.flatten: (-1xi64) <- (-1x1xi64)
        flatten_0 = paddle._C_ops.flatten(nonzero_0, 0, 1)
        return flatten_0
