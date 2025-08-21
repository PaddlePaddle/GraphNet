import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, parameter_0, parameter_1, parameter_2, parameter_3, data_0):
        # pd_op.matmul: (5x240x1152xf32) <- (5x240x384xf32, 384x1152xf32)
        matmul_0 = paddle._C_ops.matmul(data_0, parameter_3, False, False)

        # pd_op.add: (5x240x1152xf32) <- (5x240x1152xf32, 1152xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_2)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_4 = [0, -1, 3, 12, 32]

        # pd_op.reshape: (5x240x3x12x32xf32) <- (5x240x1152xf32, 5xi64)
        reshape_1 = paddle._C_ops.reshape(add_0, full_int_array_4)

        # pd_op.transpose: (3x5x12x240x32xf32) <- (5x240x3x12x32xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_1, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_1

        # pd_op.slice: (5x12x240x32xf32) <- (3x5x12x240x32xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            transpose_0, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_2

        # pd_op.slice: (5x12x240x32xf32) <- (3x5x12x240x32xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            transpose_0, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [3]

        # pd_op.slice: (5x12x240x32xf32) <- (3x5x12x240x32xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            transpose_0, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.transpose: (5x12x32x240xf32) <- (5x12x240x32xf32)
        transpose_1 = paddle._C_ops.transpose(slice_2, [0, 1, 3, 2])

        # pd_op.matmul: (5x12x240x240xf32) <- (5x12x240x32xf32, 5x12x32x240xf32)
        matmul_2 = paddle._C_ops.matmul(slice_0, transpose_1, False, False)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (5x12x240x240xf32) <- (5x12x240x240xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(matmul_2, full_0, float("0"), True)

        # pd_op.softmax: (5x12x240x240xf32) <- (5x12x240x240xf32)
        softmax_0 = paddle._C_ops.softmax(scale_0, -1)

        # pd_op.matmul: (5x12x240x32xf32) <- (5x12x240x240xf32, 5x12x240x32xf32)
        matmul_3 = paddle._C_ops.matmul(softmax_0, slice_1, False, False)

        # pd_op.transpose: (5x240x12x32xf32) <- (5x12x240x32xf32)
        transpose_2 = paddle._C_ops.transpose(matmul_3, [0, 2, 1, 3])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_5 = [0, -1, 384]

        # pd_op.reshape: (5x240x384xf32) <- (5x240x12x32xf32, 3xi64)
        reshape_0 = paddle._C_ops.reshape(transpose_2, full_int_array_5)

        # pd_op.matmul: (5x240x384xf32) <- (5x240x384xf32, 384x384xf32)
        matmul_1 = paddle._C_ops.matmul(reshape_0, parameter_1, False, False)

        # pd_op.add: (5x240x384xf32) <- (5x240x384xf32, 384xf32)
        add_1 = paddle._C_ops.add(matmul_1, parameter_0)
        return (
            matmul_0,
            add_0,
            transpose_0,
            full_int_array_0,
            full_int_array_1,
            slice_0,
            assign_0,
            full_int_array_2,
            assign_1,
            full_int_array_3,
            slice_1,
            transpose_1,
            full_0,
            softmax_0,
            transpose_2,
            reshape_0,
            matmul_1,
            add_1,
        )
