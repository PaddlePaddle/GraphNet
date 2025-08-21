import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [-1]

        # pd_op.squeeze: (2xi32) <- (2x1xi32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(data_0, full_int_array_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [0]

        # pd_op.unsqueeze: (1x2xi32) <- (2xi32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(squeeze_0, full_int_array_1)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [7272, 1]

        # pd_op.tile: (7272x2xi32) <- (1x2xi32, 2xi64)
        tile_0 = paddle._C_ops.tile(unsqueeze_0, full_int_array_2)

        # pd_op.reshape: (14544xi32) <- (7272x2xi32, 1xi64)
        reshape_0 = paddle._C_ops.reshape(tile_0, full_int_array_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.unsqueeze: (7272x1x11xf32) <- (7272x11xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(data_1, full_int_array_3)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_4 = [1, 2, 1]

        # pd_op.tile: (7272x2x11xf32) <- (7272x1x11xf32, 3xi64)
        tile_1 = paddle._C_ops.tile(unsqueeze_1, full_int_array_4)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [-1, 11]

        # pd_op.reshape: (14544x11xf32) <- (7272x2x11xf32, 2xi64)
        reshape_1 = paddle._C_ops.reshape(tile_1, full_int_array_5)
        return reshape_0, reshape_1
