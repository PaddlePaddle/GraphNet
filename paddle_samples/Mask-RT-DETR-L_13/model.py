import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (300x168x168xf32) <- (2x300x168x168xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_0, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.gather: (1x168x168xf32) <- (300x168x168xf32, 1xi64, 1xi32)
        gather_0 = paddle._C_ops.gather(slice_0, data_3, full_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        # pd_op.slice: (300x168x168xf32) <- (2x300x168x168xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_0, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.gather: (1x168x168xf32) <- (300x168x168xf32, 1xi64, 1xi32)
        gather_1 = paddle._C_ops.gather(slice_1, data_5, full_0)

        # builtin.combine: ([1x168x168xf32, 1x168x168xf32]) <- (1x168x168xf32, 1x168x168xf32)
        combine_0 = [gather_0, gather_1]

        # pd_op.concat: (2x168x168xf32) <- ([1x168x168xf32, 1x168x168xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        # pd_op.gather: (1x672x672xf32) <- (1x672x672xf32, 1xi64, 1xi32)
        gather_2 = paddle._C_ops.gather(data_1, data_4, full_0)

        # pd_op.gather: (1x672x672xf32) <- (1x672x672xf32, 1xi64, 1xi32)
        gather_3 = paddle._C_ops.gather(data_2, data_6, full_0)

        # builtin.combine: ([1x672x672xf32, 1x672x672xf32]) <- (1x672x672xf32, 1x672x672xf32)
        combine_1 = [gather_2, gather_3]

        # pd_op.concat: (2x672x672xf32) <- ([1x672x672xf32, 1x672x672xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_0)
        return concat_0, concat_1
