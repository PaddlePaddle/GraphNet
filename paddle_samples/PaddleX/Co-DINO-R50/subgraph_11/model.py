import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0):
        # builtin.combine: ([45675x4xf32]) <- (45675x4xf32)
        combine_0 = [data_0]
        del data_0

        # pd_op.stack: (1x45675x4xf32) <- ([45675x4xf32])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [34240]

        # pd_op.slice: (1x34240x4xf32) <- (1x45675x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            stack_0, [1], full_int_array_0, full_int_array_1, [1], []
        )

        # pd_op.squeeze: (34240x4xf32) <- (1x34240x4xf32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(slice_0, full_int_array_0)
        del slice_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [42800]

        # pd_op.slice: (1x8560x4xf32) <- (1x45675x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            stack_0, [1], full_int_array_1, full_int_array_2, [1], []
        )
        del full_int_array_1

        # pd_op.squeeze: (8560x4xf32) <- (1x8560x4xf32, 1xi64)
        squeeze_1 = paddle._C_ops.squeeze(slice_1, full_int_array_0)
        del slice_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [44960]

        # pd_op.slice: (1x2160x4xf32) <- (1x45675x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            stack_0, [1], full_int_array_2, full_int_array_3, [1], []
        )
        del full_int_array_2

        # pd_op.squeeze: (2160x4xf32) <- (1x2160x4xf32, 1xi64)
        squeeze_2 = paddle._C_ops.squeeze(slice_2, full_int_array_0)
        del slice_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [45500]

        # pd_op.slice: (1x540x4xf32) <- (1x45675x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            stack_0, [1], full_int_array_3, full_int_array_4, [1], []
        )
        del full_int_array_3

        # pd_op.squeeze: (540x4xf32) <- (1x540x4xf32, 1xi64)
        squeeze_3 = paddle._C_ops.squeeze(slice_3, full_int_array_0)
        del slice_3

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [45640]

        # pd_op.slice: (1x140x4xf32) <- (1x45675x4xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            stack_0, [1], full_int_array_4, full_int_array_5, [1], []
        )
        del full_int_array_4

        # pd_op.squeeze: (140x4xf32) <- (1x140x4xf32, 1xi64)
        squeeze_4 = paddle._C_ops.squeeze(slice_4, full_int_array_0)
        del slice_4

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [2147483647]

        # pd_op.slice: (1x35x4xf32) <- (1x45675x4xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            stack_0, [1], full_int_array_5, full_int_array_6, [1], []
        )
        del full_int_array_5, full_int_array_6, stack_0

        # pd_op.squeeze: (35x4xf32) <- (1x35x4xf32, 1xi64)
        squeeze_5 = paddle._C_ops.squeeze(slice_5, full_int_array_0)
        del full_int_array_0, slice_5

        return squeeze_0, squeeze_1, squeeze_2, squeeze_3, squeeze_4, squeeze_5
