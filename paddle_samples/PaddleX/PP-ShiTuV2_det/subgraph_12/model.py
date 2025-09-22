import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [-1]

        # pd_op.squeeze: (1xi32) <- (1x1xi32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(data_1, full_int_array_0)
        del data_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [0]

        # pd_op.unsqueeze: (1x1xi32) <- (1xi32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(squeeze_0, full_int_array_1)
        del squeeze_0

        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_0 = [data_0, full_0]
        del data_0, full_0

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.tile: (-1x1xi32) <- (1x1xi32, 2xi64)
        tile_0 = paddle._C_ops.tile(unsqueeze_0, stack_0)
        del stack_0, unsqueeze_0

        # pd_op.reshape: (-1xi32) <- (-1x1xi32, 1xi64)
        reshape_0 = paddle._C_ops.reshape(tile_0, full_int_array_0)
        del full_int_array_0, tile_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [1]

        # pd_op.unsqueeze: (-1x1x1xf32) <- (-1x1xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(data_2, full_int_array_2)
        del data_2

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_3 = [1, 1, 1]

        # pd_op.tile: (-1x1x1xf32) <- (-1x1x1xf32, 3xi64)
        tile_1 = paddle._C_ops.tile(unsqueeze_1, full_int_array_3)
        del full_int_array_3, unsqueeze_1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [-1, 1]

        # pd_op.reshape: (-1x1xf32) <- (-1x1x1xf32, 2xi64)
        reshape_1 = paddle._C_ops.reshape(tile_1, full_int_array_4)
        del full_int_array_4, tile_1

        # pd_op.shape64: (2xi64) <- (-1x1xf32)
        shape64_0 = paddle._C_ops.shape64(reshape_1)

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del full_int_array_1, full_int_array_2, reshape_1, shape64_0

        return slice_0, reshape_0
