import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [-1]

        # pd_op.squeeze: (1xi32) <- (1x1xi32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(data_0, full_int_array_0)
        del data_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [0]

        # pd_op.unsqueeze: (1x1xi32) <- (1xi32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(squeeze_0, full_int_array_1)
        del full_int_array_1, squeeze_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [571, 1]

        # pd_op.tile: (571x1xi32) <- (1x1xi32, 2xi64)
        tile_0 = paddle._C_ops.tile(unsqueeze_0, full_int_array_2)
        del full_int_array_2, unsqueeze_0

        # pd_op.reshape: (571xi32) <- (571x1xi32, 1xi64)
        reshape_0 = paddle._C_ops.reshape(tile_0, full_int_array_0)
        del full_int_array_0, tile_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.unsqueeze: (571x1x1xf32) <- (571x1xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(data_1, full_int_array_3)
        del data_1, full_int_array_3

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_4 = [1, 1, 1]

        # pd_op.tile: (571x1x1xf32) <- (571x1x1xf32, 3xi64)
        tile_1 = paddle._C_ops.tile(unsqueeze_1, full_int_array_4)
        del full_int_array_4, unsqueeze_1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [-1, 1]

        # pd_op.reshape: (571x1xf32) <- (571x1x1xf32, 2xi64)
        reshape_1 = paddle._C_ops.reshape(tile_1, full_int_array_5)
        del full_int_array_5, tile_1

        return reshape_0, reshape_1
