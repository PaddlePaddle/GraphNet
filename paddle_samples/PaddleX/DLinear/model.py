import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, parameter_0, parameter_1, parameter_2, parameter_3, data_0):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (-1x1x1xf32) <- (-1x96x1xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_0, [1], full_int_array_0, full_int_array_1, [1], []
        )
        del full_int_array_0, full_int_array_1

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_2 = [1, 12, 1]

        # pd_op.tile: (-1x12x1xf32) <- (-1x1x1xf32, 3xi64)
        tile_0 = paddle._C_ops.tile(slice_0, full_int_array_2)
        del slice_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [-1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [2147483647]

        # pd_op.slice: (-1x1x1xf32) <- (-1x96x1xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_0, [1], full_int_array_3, full_int_array_4, [1], []
        )
        del full_int_array_3, full_int_array_4

        # pd_op.tile: (-1x12x1xf32) <- (-1x1x1xf32, 3xi64)
        tile_1 = paddle._C_ops.tile(slice_1, full_int_array_2)
        del full_int_array_2, slice_1

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x12x1xf32, -1x96x1xf32, -1x12x1xf32]) <- (-1x12x1xf32, -1x96x1xf32, -1x12x1xf32)
        combine_0 = [tile_0, data_0, tile_1]
        del tile_0, tile_1

        # pd_op.concat: (-1x120x1xf32) <- ([-1x12x1xf32, -1x96x1xf32, -1x12x1xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)
        del combine_0, full_0

        # pd_op.transpose: (-1x1x120xf32) <- (-1x120x1xf32)
        transpose_1 = paddle._C_ops.transpose(concat_0, [0, 2, 1])
        del concat_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [2]

        # pd_op.unsqueeze: (-1x1x1x120xf32) <- (-1x1x120xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(transpose_1, full_int_array_5)
        del transpose_1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_6 = [1, 25]

        # pd_op.pool2d: (-1x1x1x96xf32) <- (-1x1x1x120xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            unsqueeze_0,
            full_int_array_6,
            [1, 1],
            [0, 0],
            False,
            True,
            "NCHW",
            "avg",
            False,
            False,
            "EXPLICIT",
        )
        del full_int_array_6, unsqueeze_0

        # pd_op.squeeze: (-1x1x96xf32) <- (-1x1x1x96xf32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(pool2d_0, full_int_array_5)
        del full_int_array_5, pool2d_0

        # pd_op.transpose: (-1x96x1xf32) <- (-1x1x96xf32)
        transpose_2 = paddle._C_ops.transpose(squeeze_0, [0, 2, 1])
        del squeeze_0

        # pd_op.subtract: (-1x96x1xf32) <- (-1x96x1xf32, -1x96x1xf32)
        subtract_0 = paddle._C_ops.subtract(data_0, transpose_2)
        del data_0

        # pd_op.transpose: (-1x1x96xf32) <- (-1x96x1xf32)
        transpose_3 = paddle._C_ops.transpose(subtract_0, [0, 2, 1])
        del subtract_0

        # pd_op.transpose: (-1x1x96xf32) <- (-1x96x1xf32)
        transpose_4 = paddle._C_ops.transpose(transpose_2, [0, 2, 1])
        del transpose_2

        # pd_op.matmul: (-1x1x96xf32) <- (-1x1x96xf32, 96x96xf32)
        matmul_0 = paddle._C_ops.matmul(transpose_3, parameter_3, False, False)
        del parameter_3, transpose_3

        # pd_op.add: (-1x1x96xf32) <- (-1x1x96xf32, 96xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_2)
        del matmul_0, parameter_2

        # pd_op.matmul: (-1x1x96xf32) <- (-1x1x96xf32, 96x96xf32)
        matmul_1 = paddle._C_ops.matmul(transpose_4, parameter_1, False, False)
        del parameter_1, transpose_4

        # pd_op.add: (-1x1x96xf32) <- (-1x1x96xf32, 96xf32)
        add_1 = paddle._C_ops.add(matmul_1, parameter_0)
        del matmul_1, parameter_0

        # pd_op.add: (-1x1x96xf32) <- (-1x1x96xf32, -1x1x96xf32)
        add_2 = paddle._C_ops.add(add_0, add_1)
        del add_0, add_1

        # pd_op.transpose: (-1x96x1xf32) <- (-1x1x96xf32)
        transpose_0 = paddle._C_ops.transpose(add_2, [0, 2, 1])
        del add_2

        return transpose_0
