import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [7600]

        # pd_op.slice: (24x7600xi64) <- (24x10105xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_0, [1], full_int_array_0, full_int_array_1, [1], []
        )

        # pd_op.squeeze: (24x7600xi64) <- (24x7600xi64, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(slice_0, full_int_array_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [9500]

        # pd_op.slice: (24x1900xi64) <- (24x10105xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_0, [1], full_int_array_1, full_int_array_2, [1], []
        )

        # pd_op.squeeze: (24x1900xi64) <- (24x1900xi64, 1xi64)
        squeeze_1 = paddle._C_ops.squeeze(slice_1, full_int_array_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [9975]

        # pd_op.slice: (24x475xi64) <- (24x10105xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            data_0, [1], full_int_array_2, full_int_array_3, [1], []
        )

        # pd_op.squeeze: (24x475xi64) <- (24x475xi64, 1xi64)
        squeeze_2 = paddle._C_ops.squeeze(slice_2, full_int_array_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [2147483647]

        # pd_op.slice: (24x130xi64) <- (24x10105xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            data_0, [1], full_int_array_3, full_int_array_4, [1], []
        )

        # pd_op.squeeze: (24x130xi64) <- (24x130xi64, 1xi64)
        squeeze_3 = paddle._C_ops.squeeze(slice_3, full_int_array_0)
        return squeeze_0, squeeze_1, squeeze_2, squeeze_3
