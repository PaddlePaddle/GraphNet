import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        parameter_0,
        parameter_1,
        parameter_2,
        parameter_3,
        data_0,
        data_1,
        data_2,
        data_3,
    ):
        # pd_op.matmul: (126x49x1152xf32) <- (126x49x384xf32, 384x1152xf32)
        matmul_0 = paddle._C_ops.matmul(data_0, parameter_3, False, False)

        # pd_op.add: (126x49x1152xf32) <- (126x49x1152xf32, 1152xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_2)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_4 = [-1, 49, 3, 12, 32]

        # pd_op.reshape: (126x49x3x12x32xf32) <- (126x49x1152xf32, 5xi64)
        reshape_2 = paddle._C_ops.reshape(add_0, full_int_array_4)

        # pd_op.transpose: (3x126x12x49x32xf32) <- (126x49x3x12x32xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_2, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_1

        # pd_op.slice: (126x12x49x32xf32) <- (3x126x12x49x32xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            transpose_0, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_2

        # pd_op.slice: (126x12x49x32xf32) <- (3x126x12x49x32xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            transpose_0, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [3]

        # pd_op.slice: (126x12x49x32xf32) <- (3x126x12x49x32xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            transpose_0, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (126x12x49x32xf32) <- (126x12x49x32xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_1, full_0, float("0"), True)

        # pd_op.transpose: (126x12x32x49xf32) <- (126x12x49x32xf32)
        transpose_1 = paddle._C_ops.transpose(slice_2, [0, 1, 3, 2])

        # pd_op.matmul: (126x12x49x49xf32) <- (126x12x49x32xf32, 126x12x32x49xf32)
        matmul_1 = paddle._C_ops.matmul(scale_0, transpose_1, False, False)

        # pd_op.flatten: (2401xi64) <- (49x49xi64)
        flatten_0 = paddle._C_ops.flatten(data_2, 0, 1)

        # pd_op.index_select: (2401x12xf32) <- (169x12xf32, 2401xi64)
        index_select_0 = paddle._C_ops.index_select(data_3, flatten_0, 0)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_5 = [49, 49, -1]

        # pd_op.reshape: (49x49x12xf32) <- (2401x12xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(index_select_0, full_int_array_5)

        # pd_op.transpose: (12x49x49xf32) <- (49x49x12xf32)
        transpose_2 = paddle._C_ops.transpose(reshape_3, [2, 0, 1])

        # pd_op.unsqueeze: (1x12x49x49xf32) <- (12x49x49xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(transpose_2, full_int_array_0)

        # pd_op.add: (126x12x49x49xf32) <- (126x12x49x49xf32, 1x12x49x49xf32)
        add_1 = paddle._C_ops.add(matmul_1, unsqueeze_0)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_6 = [-1, 63, 12, 49, 49]

        # pd_op.reshape: (2x63x12x49x49xf32) <- (126x12x49x49xf32, 5xi64)
        reshape_0 = paddle._C_ops.reshape(add_1, full_int_array_6)

        # pd_op.unsqueeze: (63x1x49x49xf32) <- (63x49x49xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(data_1, full_int_array_1)

        # pd_op.unsqueeze: (1x63x1x49x49xf32) <- (63x1x49x49xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(unsqueeze_2, full_int_array_0)

        # pd_op.add: (2x63x12x49x49xf32) <- (2x63x12x49x49xf32, 1x63x1x49x49xf32)
        add_2 = paddle._C_ops.add(reshape_0, unsqueeze_1)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_7 = [-1, 12, 49, 49]

        # pd_op.reshape: (126x12x49x49xf32) <- (2x63x12x49x49xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(add_2, full_int_array_7)

        # pd_op.softmax: (126x12x49x49xf32) <- (126x12x49x49xf32)
        softmax_0 = paddle._C_ops.softmax(reshape_4, -1)

        # pd_op.matmul: (126x12x49x32xf32) <- (126x12x49x49xf32, 126x12x49x32xf32)
        matmul_3 = paddle._C_ops.matmul(softmax_0, slice_0, False, False)

        # pd_op.transpose: (126x49x12x32xf32) <- (126x12x49x32xf32)
        transpose_3 = paddle._C_ops.transpose(matmul_3, [0, 2, 1, 3])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_8 = [-1, 49, 384]

        # pd_op.reshape: (126x49x384xf32) <- (126x49x12x32xf32, 3xi64)
        reshape_1 = paddle._C_ops.reshape(transpose_3, full_int_array_8)

        # pd_op.matmul: (126x49x384xf32) <- (126x49x384xf32, 384x384xf32)
        matmul_2 = paddle._C_ops.matmul(reshape_1, parameter_1, False, False)

        # pd_op.add: (126x49x384xf32) <- (126x49x384xf32, 384xf32)
        add_3 = paddle._C_ops.add(matmul_2, parameter_0)
        return (
            matmul_0,
            add_0,
            transpose_0,
            full_int_array_0,
            full_int_array_1,
            assign_0,
            full_int_array_2,
            assign_1,
            full_int_array_3,
            slice_0,
            full_0,
            scale_0,
            transpose_1,
            matmul_1,
            flatten_0,
            index_select_0,
            transpose_2,
            assign_2,
            unsqueeze_0,
            add_1,
            reshape_0,
            unsqueeze_1,
            add_2,
            softmax_0,
            transpose_3,
            reshape_1,
            matmul_2,
            add_3,
        )
