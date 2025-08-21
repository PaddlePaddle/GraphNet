import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_1

        # pd_op.slice: (61xf32) <- (61x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_0, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_3

        # pd_op.slice: (61xf32) <- (61x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_0, [1], full_int_array_2, full_int_array_3, [1], [1]
        )

        # pd_op.subtract: (61xf32) <- (61xf32, 61xf32)
        subtract_0 = paddle._C_ops.subtract(slice_0, slice_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [4]

        # pd_op.slice: (61xf32) <- (61x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            data_0, [1], full_int_array_1, full_int_array_4, [1], [1]
        )

        # pd_op.slice: (61xf32) <- (61x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            data_0, [1], full_int_array_3, full_int_array_0, [1], [1]
        )

        # pd_op.subtract: (61xf32) <- (61xf32, 61xf32)
        subtract_1 = paddle._C_ops.subtract(slice_2, slice_3)

        # pd_op.multiply: (61xf32) <- (61xf32, 61xf32)
        multiply_0 = paddle._C_ops.multiply(subtract_0, subtract_1)
        return (
            full_int_array_0,
            full_int_array_1,
            slice_0,
            full_int_array_2,
            full_int_array_3,
            slice_1,
            subtract_0,
            assign_0,
            full_int_array_4,
            slice_2,
            assign_1,
            assign_2,
            slice_3,
            subtract_1,
            multiply_0,
        )
