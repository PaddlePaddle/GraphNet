import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4):
        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_0 = []

        # pd_op.set_value_: (900xi64) <- (900xi64, 0xi64, 0xi64, 0xi64)
        set_value__0 = paddle._C_ops.set_value_(
            data_2,
            full_int_array_0,
            full_int_array_0,
            full_int_array_0,
            [],
            [],
            [],
            [1],
            [float("0")],
        )
        del data_2

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1xi64) <- (1xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(data_1, full_0, float("1"), True)
        del full_0

        # builtin.combine: ([1xi64]) <- (1xi64)
        combine_0 = [data_0]
        del data_0

        # pd_op.index_put: (900xi64) <- (900xi64, [1xi64], 1xi64)
        index_put_0 = paddle._C_ops.index_put(set_value__0, combine_0, scale_0, False)
        del scale_0

        # pd_op.transpose: (900xi64) <- (900xi64)
        transpose_0 = paddle._C_ops.transpose(index_put_0, [0])
        del index_put_0

        # pd_op.set_value_with_tensor_: (900xi64) <- (900xi64, 900xi64, 0xi64, 0xi64, 0xi64)
        set_value_with_tensor__0 = paddle._C_ops.set_value_with_tensor_(
            set_value__0,
            transpose_0,
            full_int_array_0,
            full_int_array_0,
            full_int_array_0,
            [],
            [],
            [],
        )
        del set_value__0, transpose_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [-1]

        # pd_op.unsqueeze: (1x1xi64) <- (1xi64, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(data_1, full_int_array_1)
        del data_1, full_int_array_1

        # pd_op.gather_nd: (1x1xi32) <- (1x1xi32, 1x1xi64)
        gather_nd_0 = paddle._C_ops.gather_nd(data_3, unsqueeze_0)
        del data_3, unsqueeze_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.slice: (1xi32) <- (1x1xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            gather_nd_0, [1], full_int_array_2, full_int_array_3, [1], [1]
        )
        del full_int_array_2, full_int_array_3, gather_nd_0

        # pd_op.cast: (1xi64) <- (1xi32)
        cast_0 = paddle._C_ops.cast(slice_0, paddle.int64)
        del slice_0

        # pd_op.index_put: (900xi64) <- (900xi64, [1xi64], 1xi64)
        index_put_1 = paddle._C_ops.index_put(data_4, combine_0, cast_0, False)
        del cast_0, combine_0

        # pd_op.transpose: (900xi64) <- (900xi64)
        transpose_1 = paddle._C_ops.transpose(index_put_1, [0])
        del index_put_1

        # pd_op.set_value_with_tensor_: (900xi64) <- (900xi64, 900xi64, 0xi64, 0xi64, 0xi64)
        set_value_with_tensor__1 = paddle._C_ops.set_value_with_tensor_(
            data_4,
            transpose_1,
            full_int_array_0,
            full_int_array_0,
            full_int_array_0,
            [],
            [],
            [],
        )
        del data_4, full_int_array_0, transpose_1

        return set_value_with_tensor__0, set_value_with_tensor__1
