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
    ):
        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [-1, 4]

        # pd_op.reshape: (-1x4xf32) <- (-1x4xf32, 2xi64)
        reshape_10 = paddle._C_ops.reshape(data_10, full_int_array_0)

        # pd_op.reshape: (-1x4xf32) <- (-1x4xf32, 2xi64)
        reshape_11 = paddle._C_ops.reshape(data_11, full_int_array_0)

        # pd_op.reshape: (-1x4xf32) <- (-1x4xf32, 2xi64)
        reshape_12 = paddle._C_ops.reshape(data_12, full_int_array_0)

        # pd_op.reshape: (-1x4xf32) <- (-1x4xf32, 2xi64)
        reshape_13 = paddle._C_ops.reshape(data_13, full_int_array_0)

        # pd_op.reshape: (-1x4xf32) <- (-1x4xf32, 2xi64)
        reshape_14 = paddle._C_ops.reshape(data_14, full_int_array_0)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32]) <- (-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32)
        combine_0 = [reshape_10, reshape_11, reshape_12, reshape_13, reshape_14]

        # pd_op.concat: (-1x4xf32) <- ([-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_1)

        # pd_op.transpose: (2x-1x-1x3xf32) <- (2x3x-1x-1xf32)
        transpose_0 = paddle._C_ops.transpose(data_0, [0, 2, 3, 1])

        # pd_op.shape64: (4xi64) <- (2x3x-1x-1xf32)
        shape64_0 = paddle._C_ops.shape64(data_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [3]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [4]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_4 = [2, -1, 1]

        # pd_op.reshape: (2x-1x1xf32) <- (2x-1x-1x3xf32, 3xi64)
        reshape_0 = paddle._C_ops.reshape(transpose_0, full_int_array_4)

        # pd_op.transpose: (2x-1x-1x3xf32) <- (2x3x-1x-1xf32)
        transpose_1 = paddle._C_ops.transpose(data_1, [0, 2, 3, 1])

        # pd_op.shape64: (4xi64) <- (2x3x-1x-1xf32)
        shape64_1 = paddle._C_ops.shape64(data_1)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.reshape: (2x-1x1xf32) <- (2x-1x-1x3xf32, 3xi64)
        reshape_1 = paddle._C_ops.reshape(transpose_1, full_int_array_4)

        # pd_op.transpose: (2x-1x-1x3xf32) <- (2x3x-1x-1xf32)
        transpose_2 = paddle._C_ops.transpose(data_2, [0, 2, 3, 1])

        # pd_op.shape64: (4xi64) <- (2x3x-1x-1xf32)
        shape64_2 = paddle._C_ops.shape64(data_2)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.reshape: (2x-1x1xf32) <- (2x-1x-1x3xf32, 3xi64)
        reshape_2 = paddle._C_ops.reshape(transpose_2, full_int_array_4)

        # pd_op.transpose: (2x-1x-1x3xf32) <- (2x3x-1x-1xf32)
        transpose_3 = paddle._C_ops.transpose(data_3, [0, 2, 3, 1])

        # pd_op.shape64: (4xi64) <- (2x3x-1x-1xf32)
        shape64_3 = paddle._C_ops.shape64(data_3)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.reshape: (2x-1x1xf32) <- (2x-1x-1x3xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_3, full_int_array_4)

        # pd_op.transpose: (2x-1x11x3xf32) <- (2x3x-1x11xf32)
        transpose_4 = paddle._C_ops.transpose(data_4, [0, 2, 3, 1])

        # pd_op.shape64: (4xi64) <- (2x3x-1x11xf32)
        shape64_4 = paddle._C_ops.shape64(data_4)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            shape64_4, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.reshape: (2x-1x1xf32) <- (2x-1x11x3xf32, 3xi64)
        reshape_4 = paddle._C_ops.reshape(transpose_4, full_int_array_4)

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_0 = full_0

        # builtin.combine: ([2x-1x1xf32, 2x-1x1xf32, 2x-1x1xf32, 2x-1x1xf32, 2x-1x1xf32]) <- (2x-1x1xf32, 2x-1x1xf32, 2x-1x1xf32, 2x-1x1xf32, 2x-1x1xf32)
        combine_1 = [reshape_0, reshape_1, reshape_2, reshape_3, reshape_4]

        # pd_op.concat: (2x-1x1xf32) <- ([2x-1x1xf32, 2x-1x1xf32, 2x-1x1xf32, 2x-1x1xf32, 2x-1x1xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_0)

        # pd_op.transpose: (2x-1x-1x12xf32) <- (2x12x-1x-1xf32)
        transpose_5 = paddle._C_ops.transpose(data_5, [0, 2, 3, 1])

        # pd_op.shape64: (4xi64) <- (2x12x-1x-1xf32)
        shape64_5 = paddle._C_ops.shape64(data_5)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            shape64_5, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            shape64_5, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_5 = [2, -1, 4]

        # pd_op.reshape: (2x-1x4xf32) <- (2x-1x-1x12xf32, 3xi64)
        reshape_5 = paddle._C_ops.reshape(transpose_5, full_int_array_5)

        # pd_op.transpose: (2x-1x-1x12xf32) <- (2x12x-1x-1xf32)
        transpose_6 = paddle._C_ops.transpose(data_6, [0, 2, 3, 1])

        # pd_op.shape64: (4xi64) <- (2x12x-1x-1xf32)
        shape64_6 = paddle._C_ops.shape64(data_6)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            shape64_6, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            shape64_6, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.reshape: (2x-1x4xf32) <- (2x-1x-1x12xf32, 3xi64)
        reshape_6 = paddle._C_ops.reshape(transpose_6, full_int_array_5)

        # pd_op.transpose: (2x-1x-1x12xf32) <- (2x12x-1x-1xf32)
        transpose_7 = paddle._C_ops.transpose(data_7, [0, 2, 3, 1])

        # pd_op.shape64: (4xi64) <- (2x12x-1x-1xf32)
        shape64_7 = paddle._C_ops.shape64(data_7)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            shape64_7, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            shape64_7, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.reshape: (2x-1x4xf32) <- (2x-1x-1x12xf32, 3xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_7, full_int_array_5)

        # pd_op.transpose: (2x-1x-1x12xf32) <- (2x12x-1x-1xf32)
        transpose_8 = paddle._C_ops.transpose(data_8, [0, 2, 3, 1])

        # pd_op.shape64: (4xi64) <- (2x12x-1x-1xf32)
        shape64_8 = paddle._C_ops.shape64(data_8)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            shape64_8, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            shape64_8, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.reshape: (2x-1x4xf32) <- (2x-1x-1x12xf32, 3xi64)
        reshape_8 = paddle._C_ops.reshape(transpose_8, full_int_array_5)

        # pd_op.transpose: (2x-1x11x12xf32) <- (2x12x-1x11xf32)
        transpose_9 = paddle._C_ops.transpose(data_9, [0, 2, 3, 1])

        # pd_op.shape64: (4xi64) <- (2x12x-1x11xf32)
        shape64_9 = paddle._C_ops.shape64(data_9)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            shape64_9, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.reshape: (2x-1x4xf32) <- (2x-1x11x12xf32, 3xi64)
        reshape_9 = paddle._C_ops.reshape(transpose_9, full_int_array_5)

        # builtin.combine: ([2x-1x4xf32, 2x-1x4xf32, 2x-1x4xf32, 2x-1x4xf32, 2x-1x4xf32]) <- (2x-1x4xf32, 2x-1x4xf32, 2x-1x4xf32, 2x-1x4xf32, 2x-1x4xf32)
        combine_2 = [reshape_5, reshape_6, reshape_7, reshape_8, reshape_9]

        # pd_op.concat: (2x-1x4xf32) <- ([2x-1x4xf32, 2x-1x4xf32, 2x-1x4xf32, 2x-1x4xf32, 2x-1x4xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_0)
        return (
            transpose_0,
            reshape_0,
            transpose_1,
            reshape_1,
            transpose_2,
            reshape_2,
            transpose_3,
            reshape_3,
            transpose_4,
            reshape_4,
            full_0,
            transpose_5,
            reshape_5,
            transpose_6,
            reshape_6,
            transpose_7,
            reshape_7,
            transpose_8,
            reshape_8,
            transpose_9,
            reshape_9,
            assign_0,
            concat_0,
            concat_1,
            concat_2,
        )
