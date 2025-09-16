import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, parameter_0, parameter_1, data_0, data_1, data_2):
        # pd_op.matmul: (-1x96x2xf32) <- (-1x96x32xf32, 32x2xf32)
        matmul_0 = paddle._C_ops.matmul(data_0, parameter_1, False, False)
        del data_0, parameter_1

        # pd_op.add: (-1x96x2xf32) <- (-1x96x2xf32, 2xf32)
        add_1 = paddle._C_ops.add(matmul_0, parameter_0)
        del matmul_0, parameter_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (-1x2xf32) <- (-1x1x2xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_1, [1], full_int_array_0, full_int_array_1, [1], [1]
        )
        del data_1

        # pd_op.unsqueeze: (-1x1x2xf32) <- (-1x2xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(slice_0, full_int_array_1)
        del slice_0

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_2 = [1, 96, 1]

        # pd_op.tile: (-1x96x2xf32) <- (-1x1x2xf32, 3xi64)
        tile_0 = paddle._C_ops.tile(unsqueeze_0, full_int_array_2)
        del unsqueeze_0

        # pd_op.multiply: (-1x96x2xf32) <- (-1x96x2xf32, -1x96x2xf32)
        multiply_0 = paddle._C_ops.multiply(add_1, tile_0)
        del add_1, tile_0

        # pd_op.slice: (-1x2xf32) <- (-1x1x2xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_2, [1], full_int_array_0, full_int_array_1, [1], [1]
        )
        del data_2, full_int_array_0

        # pd_op.unsqueeze: (-1x1x2xf32) <- (-1x2xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(slice_1, full_int_array_1)
        del full_int_array_1, slice_1

        # pd_op.tile: (-1x96x2xf32) <- (-1x1x2xf32, 3xi64)
        tile_1 = paddle._C_ops.tile(unsqueeze_1, full_int_array_2)
        del full_int_array_2, unsqueeze_1

        # pd_op.add: (-1x96x2xf32) <- (-1x96x2xf32, -1x96x2xf32)
        add_0 = paddle._C_ops.add(multiply_0, tile_1)
        del multiply_0, tile_1

        return add_0
