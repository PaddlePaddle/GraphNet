import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3):
        # pd_op.shape64: (2xi64) <- (512x4xf32)
        shape64_0 = paddle._C_ops.shape64(data_0)
        del data_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (1xi64) <- (2xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_0, full_int_array_1, [1], []
        )
        del shape64_0

        # pd_op.shape64: (2xi64) <- (512x4xf32)
        shape64_1 = paddle._C_ops.shape64(data_1)
        del data_1

        # pd_op.slice: (1xi64) <- (2xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_0, full_int_array_1, [1], []
        )
        del shape64_1

        # pd_op.shape64: (2xi64) <- (512x4xf32)
        shape64_2 = paddle._C_ops.shape64(data_2)
        del data_2

        # pd_op.slice: (1xi64) <- (2xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_0, full_int_array_1, [1], []
        )
        del shape64_2

        # pd_op.shape64: (2xi64) <- (512x4xf32)
        shape64_3 = paddle._C_ops.shape64(data_3)
        del data_3

        # pd_op.slice: (1xi64) <- (2xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_0, full_int_array_1, [1], []
        )
        del full_int_array_0, full_int_array_1, shape64_3

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([1xi64, 1xi64, 1xi64, 1xi64]) <- (1xi64, 1xi64, 1xi64, 1xi64)
        combine_0 = [slice_0, slice_1, slice_2, slice_3]
        del slice_0, slice_1, slice_2, slice_3

        # pd_op.concat: (4xi64) <- ([1xi64, 1xi64, 1xi64, 1xi64], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)
        del combine_0, full_0

        return concat_0
