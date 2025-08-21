import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        data_0,
        data_1,
        data_2,
        data_3,
        data_4,
        data_5,
        data_6,
        data_7,
        data_8,
        data_9,
        data_10,
        data_11,
        data_12,
        data_13,
        data_14,
        data_15,
        data_16,
        data_17,
        data_18,
        data_19,
        data_20,
        data_21,
        data_22,
        data_23,
        data_24,
    ):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (300x4xf32) <- (8x300x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_0, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.gather: (1x4xf32) <- (300x4xf32, 1xi64, 1xi32)
        gather_0 = paddle._C_ops.gather(slice_0, data_9, full_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        # pd_op.slice: (300x4xf32) <- (8x300x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_0, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.gather: (1x4xf32) <- (300x4xf32, 1xi64, 1xi32)
        gather_1 = paddle._C_ops.gather(slice_1, data_11, full_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [3]

        # pd_op.slice: (300x4xf32) <- (8x300x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            data_0, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.gather: (1x4xf32) <- (300x4xf32, 1xi64, 1xi32)
        gather_2 = paddle._C_ops.gather(slice_2, data_13, full_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [4]

        # pd_op.slice: (300x4xf32) <- (8x300x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            data_0, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.gather: (1x4xf32) <- (300x4xf32, 1xi64, 1xi32)
        gather_3 = paddle._C_ops.gather(slice_3, data_15, full_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [5]

        # pd_op.slice: (300x4xf32) <- (8x300x4xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            data_0, [0], full_int_array_4, full_int_array_5, [1], [0]
        )

        # pd_op.gather: (1x4xf32) <- (300x4xf32, 1xi64, 1xi32)
        gather_4 = paddle._C_ops.gather(slice_4, data_17, full_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [6]

        # pd_op.slice: (300x4xf32) <- (8x300x4xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            data_0, [0], full_int_array_5, full_int_array_6, [1], [0]
        )

        # pd_op.gather: (1x4xf32) <- (300x4xf32, 1xi64, 1xi32)
        gather_5 = paddle._C_ops.gather(slice_5, data_19, full_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [7]

        # pd_op.slice: (300x4xf32) <- (8x300x4xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            data_0, [0], full_int_array_6, full_int_array_7, [1], [0]
        )

        # pd_op.gather: (1x4xf32) <- (300x4xf32, 1xi64, 1xi32)
        gather_6 = paddle._C_ops.gather(slice_6, data_21, full_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [8]

        # pd_op.slice: (300x4xf32) <- (8x300x4xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            data_0, [0], full_int_array_7, full_int_array_8, [1], [0]
        )

        # pd_op.gather: (1x4xf32) <- (300x4xf32, 1xi64, 1xi32)
        gather_7 = paddle._C_ops.gather(slice_7, data_23, full_0)

        # builtin.combine: ([1x4xf32, 1x4xf32, 1x4xf32, 1x4xf32, 1x4xf32, 1x4xf32, 1x4xf32, 1x4xf32]) <- (1x4xf32, 1x4xf32, 1x4xf32, 1x4xf32, 1x4xf32, 1x4xf32, 1x4xf32, 1x4xf32)
        combine_0 = [
            gather_0,
            gather_1,
            gather_2,
            gather_3,
            gather_4,
            gather_5,
            gather_6,
            gather_7,
        ]

        # pd_op.concat: (8x4xf32) <- ([1x4xf32, 1x4xf32, 1x4xf32, 1x4xf32, 1x4xf32, 1x4xf32, 1x4xf32, 1x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        # pd_op.gather: (1x4xf32) <- (1x4xf32, 1xi64, 1xi32)
        gather_8 = paddle._C_ops.gather(data_1, data_10, full_0)

        # pd_op.gather: (1x4xf32) <- (1x4xf32, 1xi64, 1xi32)
        gather_9 = paddle._C_ops.gather(data_2, data_12, full_0)

        # pd_op.gather: (1x4xf32) <- (1x4xf32, 1xi64, 1xi32)
        gather_10 = paddle._C_ops.gather(data_3, data_14, full_0)

        # pd_op.gather: (1x4xf32) <- (1x4xf32, 1xi64, 1xi32)
        gather_11 = paddle._C_ops.gather(data_4, data_16, full_0)

        # pd_op.gather: (1x4xf32) <- (1x4xf32, 1xi64, 1xi32)
        gather_12 = paddle._C_ops.gather(data_5, data_18, full_0)

        # pd_op.gather: (1x4xf32) <- (1x4xf32, 1xi64, 1xi32)
        gather_13 = paddle._C_ops.gather(data_6, data_20, full_0)

        # pd_op.gather: (1x4xf32) <- (1x4xf32, 1xi64, 1xi32)
        gather_14 = paddle._C_ops.gather(data_7, data_22, full_0)

        # pd_op.gather: (1x4xf32) <- (1x4xf32, 1xi64, 1xi32)
        gather_15 = paddle._C_ops.gather(data_8, data_24, full_0)

        # builtin.combine: ([1x4xf32, 1x4xf32, 1x4xf32, 1x4xf32, 1x4xf32, 1x4xf32, 1x4xf32, 1x4xf32]) <- (1x4xf32, 1x4xf32, 1x4xf32, 1x4xf32, 1x4xf32, 1x4xf32, 1x4xf32, 1x4xf32)
        combine_1 = [
            gather_8,
            gather_9,
            gather_10,
            gather_11,
            gather_12,
            gather_13,
            gather_14,
            gather_15,
        ]

        # pd_op.concat: (8x4xf32) <- ([1x4xf32, 1x4xf32, 1x4xf32, 1x4xf32, 1x4xf32, 1x4xf32, 1x4xf32, 1x4xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_0)
        return concat_0, concat_1
