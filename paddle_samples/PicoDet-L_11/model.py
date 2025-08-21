import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7):
        # pd_op.divide: (8x10285x4xf32) <- (8x10285x4xf32, 10285x1xf32)
        divide_0 = paddle._C_ops.divide(data_1, data_3)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_0 = [8, 10285, 2]

        # pd_op.expand: (8x10285x2xf32) <- (10285x2xf32, 3xi64)
        expand_0 = paddle._C_ops.expand(data_4, full_int_array_0)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [-1, 2]

        # pd_op.reshape: (82280x2xf32) <- (8x10285x2xf32, 2xi64)
        reshape_6 = paddle._C_ops.reshape(expand_0, full_int_array_1)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_2 = [8, 10285, 1]

        # pd_op.expand: (8x10285x1xf32) <- (10285x1xf32, 3xi64)
        expand_1 = paddle._C_ops.expand(data_3, full_int_array_2)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [-1, 1]

        # pd_op.reshape: (82280x1xf32) <- (8x10285x1xf32, 2xi64)
        reshape_5 = paddle._C_ops.reshape(expand_1, full_int_array_3)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [-1, 4]

        # pd_op.reshape: (82280x4xf32) <- (8x10285x4xf32, 2xi64)
        reshape_1 = paddle._C_ops.reshape(data_5, full_int_array_4)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [-1, 32]

        # pd_op.reshape: (82280x32xf32) <- (8x10285x32xf32, 2xi64)
        reshape_4 = paddle._C_ops.reshape(data_6, full_int_array_5)

        # pd_op.reshape: (82280x4xf32) <- (8x10285x4xf32, 2xi64)
        reshape_3 = paddle._C_ops.reshape(data_7, full_int_array_4)

        # pd_op.reshape: (82280x4xf32) <- (8x10285x4xf32, 2xi64)
        reshape_2 = paddle._C_ops.reshape(divide_0, full_int_array_4)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [-1]

        # pd_op.reshape: (82280xi32) <- (8x10285xi32, 1xi64)
        reshape_7 = paddle._C_ops.reshape(data_0, full_int_array_6)

        # pd_op.reshape: (82280x4xf32) <- (8x10285x4xf32, 2xi64)
        reshape_0 = paddle._C_ops.reshape(data_2, full_int_array_4)

        # pd_op.full: (xi32) <- ()
        full_0 = paddle._C_ops.full(
            [], float("0"), paddle.int32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_equal: (82280xb) <- (82280xi32, xi32)
        greater_equal_0 = paddle._C_ops.greater_equal(reshape_7, full_0)

        # pd_op.full: (xi32) <- ()
        full_1 = paddle._C_ops.full(
            [], float("4"), paddle.int32, paddle.framework._current_expected_place()
        )

        # pd_op.less_than: (82280xb) <- (82280xi32, xi32)
        less_than_0 = paddle._C_ops.less_than(reshape_7, full_1)

        # pd_op.logical_and: (82280xb) <- (82280xb, 82280xb)
        logical_and_0 = paddle._C_ops.logical_and(greater_equal_0, less_than_0)

        # pd_op.nonzero: (-1x1xi64) <- (82280xb)
        nonzero_0 = paddle._C_ops.nonzero(logical_and_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [1]

        # pd_op.squeeze: (-1xi64) <- (-1x1xi64, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(nonzero_0, full_int_array_7)
        return (
            squeeze_0,
            reshape_0,
            reshape_1,
            reshape_2,
            reshape_3,
            reshape_4,
            reshape_5,
            reshape_6,
        )
