import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [20736]

        # pd_op.slice: (4x20736xi64) <- (4x27540xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_0, [1], full_int_array_0, full_int_array_1, [1], []
        )

        # pd_op.squeeze: (4x20736xi64) <- (4x20736xi64, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(slice_0, full_int_array_0)
        del slice_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [25920]

        # pd_op.slice: (4x5184xi64) <- (4x27540xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_0, [1], full_int_array_1, full_int_array_2, [1], []
        )
        del full_int_array_1

        # pd_op.squeeze: (4x5184xi64) <- (4x5184xi64, 1xi64)
        squeeze_1 = paddle._C_ops.squeeze(slice_1, full_int_array_0)
        del slice_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [27216]

        # pd_op.slice: (4x1296xi64) <- (4x27540xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            data_0, [1], full_int_array_2, full_int_array_3, [1], []
        )
        del full_int_array_2

        # pd_op.squeeze: (4x1296xi64) <- (4x1296xi64, 1xi64)
        squeeze_2 = paddle._C_ops.squeeze(slice_2, full_int_array_0)
        del slice_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [2147483647]

        # pd_op.slice: (4x324xi64) <- (4x27540xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            data_0, [1], full_int_array_3, full_int_array_4, [1], []
        )
        del data_0, full_int_array_3, full_int_array_4

        # pd_op.squeeze: (4x324xi64) <- (4x324xi64, 1xi64)
        squeeze_3 = paddle._C_ops.squeeze(slice_3, full_int_array_0)
        del full_int_array_0, slice_3

        return squeeze_0, squeeze_1, squeeze_2, squeeze_3
