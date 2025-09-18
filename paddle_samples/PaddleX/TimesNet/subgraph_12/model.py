import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6):
        # builtin.combine: ([16x192x32xf32, 16x192x32xf32, 16x192x32xf32, 16x192x32xf32, 16x192x32xf32]) <- (16x192x32xf32, 16x192x32xf32, 16x192x32xf32, 16x192x32xf32, 16x192x32xf32)
        combine_0 = [data_0, data_1, data_2, data_3, data_4]
        del data_0, data_1, data_2, data_3, data_4

        # pd_op.stack: (16x192x32x5xf32) <- ([16x192x32xf32, 16x192x32xf32, 16x192x32xf32, 16x192x32xf32, 16x192x32xf32])
        stack_0 = paddle._C_ops.stack(combine_0, -1)
        del combine_0

        # pd_op.softmax: (16x5xf32) <- (16x5xf32)
        softmax_0 = paddle._C_ops.softmax(data_5, 1)
        del data_5

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_0

        # pd_op.unsqueeze: (16x1x5xf32) <- (16x5xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(softmax_0, full_int_array_0)

        # pd_op.unsqueeze: (16x1x1x5xf32) <- (16x1x5xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(unsqueeze_0, full_int_array_0)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [1, 192, 32, 1]

        # pd_op.tile: (16x192x32x5xf32) <- (16x1x1x5xf32, 4xi64)
        tile_0 = paddle._C_ops.tile(unsqueeze_1, full_int_array_1)

        # pd_op.multiply: (16x192x32x5xf32) <- (16x192x32x5xf32, 16x192x32x5xf32)
        multiply_0 = paddle._C_ops.multiply(stack_0, tile_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [-1]

        # pd_op.sum: (16x192x32xf32) <- (16x192x32x5xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(multiply_0, full_int_array_2, None, False)

        # pd_op.add: (16x192x32xf32) <- (16x192x32xf32, 16x192x32xf32)
        add_0 = paddle._C_ops.add(sum_0, data_6)
        del (
            assign_0,
            data_6,
            full_int_array_0,
            full_int_array_1,
            full_int_array_2,
            multiply_0,
            softmax_0,
            stack_0,
            sum_0,
            tile_0,
            unsqueeze_0,
            unsqueeze_1,
        )

        return add_0
