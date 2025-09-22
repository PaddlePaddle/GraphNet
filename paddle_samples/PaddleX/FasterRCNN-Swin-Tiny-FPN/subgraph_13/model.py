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
        # pd_op.matmul: (24x49x2304xf32) <- (24x49x768xf32, 768x2304xf32)
        matmul_0 = paddle._C_ops.matmul(data_1, parameter_3, False, False)
        del data_1, parameter_3

        # pd_op.add: (24x49x2304xf32) <- (24x49x2304xf32, 2304xf32)
        add_1 = paddle._C_ops.add(matmul_0, parameter_2)
        del parameter_2

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_0 = [-1, 49, 3, 24, 32]

        # pd_op.reshape: (24x49x3x24x32xf32) <- (24x49x2304xf32, 5xi64)
        reshape_0 = paddle._C_ops.reshape(add_1, full_int_array_0)
        del full_int_array_0

        # pd_op.transpose: (3x24x24x49x32xf32) <- (24x49x3x24x32xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_0, [2, 0, 3, 1, 4])
        del reshape_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_2

        # pd_op.slice: (24x24x49x32xf32) <- (3x24x24x49x32xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            transpose_0, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_3

        # pd_op.slice: (24x24x49x32xf32) <- (3x24x24x49x32xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            transpose_0, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [3]

        # pd_op.slice: (24x24x49x32xf32) <- (3x24x24x49x32xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            transpose_0, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (24x24x49x32xf32) <- (24x24x49x32xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float("0"), True)
        del slice_0

        # pd_op.transpose: (24x24x32x49xf32) <- (24x24x49x32xf32)
        transpose_1 = paddle._C_ops.transpose(slice_1, [0, 1, 3, 2])
        del slice_1

        # pd_op.matmul: (24x24x49x49xf32) <- (24x24x49x32xf32, 24x24x32x49xf32)
        matmul_1 = paddle._C_ops.matmul(scale_0, transpose_1, False, False)

        # pd_op.flatten: (2401xi64) <- (49x49xi64)
        flatten_0 = paddle._C_ops.flatten(data_3, 0, 1)
        del data_3

        # pd_op.index_select: (2401x24xf32) <- (169x24xf32, 2401xi64)
        index_select_0 = paddle._C_ops.index_select(data_0, flatten_0, 0)
        del data_0

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_5 = [49, 49, -1]

        # pd_op.reshape: (49x49x24xf32) <- (2401x24xf32, 3xi64)
        reshape_1 = paddle._C_ops.reshape(index_select_0, full_int_array_5)
        del full_int_array_5

        # pd_op.transpose: (24x49x49xf32) <- (49x49x24xf32)
        transpose_2 = paddle._C_ops.transpose(reshape_1, [2, 0, 1])
        del reshape_1

        # pd_op.unsqueeze: (1x24x49x49xf32) <- (24x49x49xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(transpose_2, full_int_array_1)

        # pd_op.add: (24x24x49x49xf32) <- (24x24x49x49xf32, 1x24x49x49xf32)
        add_2 = paddle._C_ops.add(matmul_1, unsqueeze_0)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_6 = [-1, 12, 24, 49, 49]

        # pd_op.reshape: (2x12x24x49x49xf32) <- (24x24x49x49xf32, 5xi64)
        reshape_2 = paddle._C_ops.reshape(add_2, full_int_array_6)
        del full_int_array_6

        # pd_op.unsqueeze: (12x1x49x49xf32) <- (12x49x49xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(data_2, full_int_array_2)
        del data_2

        # pd_op.unsqueeze: (1x12x1x49x49xf32) <- (12x1x49x49xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(unsqueeze_1, full_int_array_1)
        del unsqueeze_1

        # pd_op.add: (2x12x24x49x49xf32) <- (2x12x24x49x49xf32, 1x12x1x49x49xf32)
        add_3 = paddle._C_ops.add(reshape_2, unsqueeze_2)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_7 = [-1, 24, 49, 49]

        # pd_op.reshape: (24x24x49x49xf32) <- (2x12x24x49x49xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(add_3, full_int_array_7)
        del full_int_array_7

        # pd_op.softmax: (24x24x49x49xf32) <- (24x24x49x49xf32)
        softmax_0 = paddle._C_ops.softmax(reshape_3, -1)
        del reshape_3

        # pd_op.matmul: (24x24x49x32xf32) <- (24x24x49x49xf32, 24x24x49x32xf32)
        matmul_2 = paddle._C_ops.matmul(softmax_0, slice_2, False, False)

        # pd_op.transpose: (24x49x24x32xf32) <- (24x24x49x32xf32)
        transpose_3 = paddle._C_ops.transpose(matmul_2, [0, 2, 1, 3])
        del matmul_2

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_8 = [-1, 49, 768]

        # pd_op.reshape: (24x49x768xf32) <- (24x49x24x32xf32, 3xi64)
        reshape_4 = paddle._C_ops.reshape(transpose_3, full_int_array_8)
        del full_int_array_8

        # pd_op.matmul: (24x49x768xf32) <- (24x49x768xf32, 768x768xf32)
        matmul_3 = paddle._C_ops.matmul(reshape_4, parameter_1, False, False)
        del parameter_1

        # pd_op.add: (24x49x768xf32) <- (24x49x768xf32, 768xf32)
        add_0 = paddle._C_ops.add(matmul_3, parameter_0)
        del (
            add_1,
            add_2,
            add_3,
            assign_0,
            assign_1,
            assign_2,
            flatten_0,
            full_0,
            full_int_array_1,
            full_int_array_2,
            full_int_array_3,
            full_int_array_4,
            index_select_0,
            matmul_0,
            matmul_1,
            matmul_3,
            parameter_0,
            reshape_2,
            reshape_4,
            scale_0,
            slice_2,
            softmax_0,
            transpose_0,
            transpose_1,
            transpose_2,
            transpose_3,
            unsqueeze_0,
            unsqueeze_2,
        )

        return add_0
