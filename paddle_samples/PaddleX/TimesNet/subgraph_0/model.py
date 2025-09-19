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
        parameter_4,
        parameter_5,
        parameter_6,
        parameter_7,
        parameter_8,
        parameter_9,
        parameter_10,
        parameter_11,
        parameter_12,
        parameter_13,
        parameter_14,
        parameter_15,
        parameter_16,
        parameter_17,
        parameter_18,
        parameter_19,
        parameter_20,
        parameter_21,
        parameter_22,
        parameter_23,
        data_0,
        data_1,
        data_2,
    ):
        # pd_op.cast: (xi64) <- (xi32)
        cast_0 = paddle._C_ops.cast(data_1, paddle.int64)
        del data_1

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        floor_divide_0 = paddle._C_ops.floor_divide(data_2, cast_0)
        del data_2

        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("16"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("32"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_0 = [full_0, floor_divide_0, cast_0, full_1]
        del cast_0, floor_divide_0, full_0, full_1

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.reshape: (16x-1x-1x32xf32) <- (16x192x32xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(data_0, stack_0)
        del data_0, stack_0

        # pd_op.transpose: (16x32x-1x-1xf32) <- (16x-1x-1x32xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_0, [0, 3, 1, 2])
        del reshape_0

        # pd_op.conv2d: (16x32x-1x-1xf32) <- (16x32x-1x-1xf32, 32x32x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            transpose_0, parameter_23, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_23

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_22, full_int_array_0)
        del parameter_22

        # pd_op.add: (16x32x-1x-1xf32) <- (16x32x-1x-1xf32, 1x32x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_0, reshape_1)

        # pd_op.conv2d: (16x32x-1x-1xf32) <- (16x32x-1x-1xf32, 32x32x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            transpose_0, parameter_21, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_21

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(parameter_20, full_int_array_0)
        del parameter_20

        # pd_op.add: (16x32x-1x-1xf32) <- (16x32x-1x-1xf32, 1x32x1x1xf32)
        add_1 = paddle._C_ops.add(conv2d_1, reshape_2)

        # pd_op.conv2d: (16x32x-1x-1xf32) <- (16x32x-1x-1xf32, 32x32x5x5xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            transpose_0, parameter_19, [1, 1], [2, 2], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_19

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(parameter_18, full_int_array_0)
        del parameter_18

        # pd_op.add: (16x32x-1x-1xf32) <- (16x32x-1x-1xf32, 1x32x1x1xf32)
        add_2 = paddle._C_ops.add(conv2d_2, reshape_3)

        # pd_op.conv2d: (16x32x-1x-1xf32) <- (16x32x-1x-1xf32, 32x32x7x7xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            transpose_0, parameter_17, [1, 1], [3, 3], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_17

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(parameter_16, full_int_array_0)
        del parameter_16

        # pd_op.add: (16x32x-1x-1xf32) <- (16x32x-1x-1xf32, 1x32x1x1xf32)
        add_3 = paddle._C_ops.add(conv2d_3, reshape_4)

        # pd_op.conv2d: (16x32x-1x-1xf32) <- (16x32x-1x-1xf32, 32x32x9x9xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            transpose_0, parameter_15, [1, 1], [4, 4], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_15

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(parameter_14, full_int_array_0)
        del parameter_14

        # pd_op.add: (16x32x-1x-1xf32) <- (16x32x-1x-1xf32, 1x32x1x1xf32)
        add_4 = paddle._C_ops.add(conv2d_4, reshape_5)

        # pd_op.conv2d: (16x32x-1x-1xf32) <- (16x32x-1x-1xf32, 32x32x11x11xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            transpose_0, parameter_13, [1, 1], [5, 5], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_13

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(parameter_12, full_int_array_0)
        del parameter_12

        # pd_op.add: (16x32x-1x-1xf32) <- (16x32x-1x-1xf32, 1x32x1x1xf32)
        add_5 = paddle._C_ops.add(conv2d_5, reshape_6)

        # builtin.combine: ([16x32x-1x-1xf32, 16x32x-1x-1xf32, 16x32x-1x-1xf32, 16x32x-1x-1xf32, 16x32x-1x-1xf32, 16x32x-1x-1xf32]) <- (16x32x-1x-1xf32, 16x32x-1x-1xf32, 16x32x-1x-1xf32, 16x32x-1x-1xf32, 16x32x-1x-1xf32, 16x32x-1x-1xf32)
        combine_1 = [add_0, add_1, add_2, add_3, add_4, add_5]

        # pd_op.stack: (16x32x-1x-1x6xf32) <- ([16x32x-1x-1xf32, 16x32x-1x-1xf32, 16x32x-1x-1xf32, 16x32x-1x-1xf32, 16x32x-1x-1xf32, 16x32x-1x-1xf32])
        stack_1 = paddle._C_ops.stack(combine_1, -1)
        del combine_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [-1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_1

        # pd_op.mean: (16x32x-1x-1xf32) <- (16x32x-1x-1x6xf32, 1xi64)
        mean_0 = paddle._C_ops.mean(stack_1, full_int_array_1, False)

        # pd_op.gelu: (16x32x-1x-1xf32) <- (16x32x-1x-1xf32)
        gelu_0 = paddle._C_ops.gelu(mean_0, False)

        # pd_op.conv2d: (16x32x-1x-1xf32) <- (16x32x-1x-1xf32, 32x32x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            gelu_0, parameter_11, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_11

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(parameter_10, full_int_array_0)
        del parameter_10

        # pd_op.add: (16x32x-1x-1xf32) <- (16x32x-1x-1xf32, 1x32x1x1xf32)
        add_6 = paddle._C_ops.add(conv2d_6, reshape_7)

        # pd_op.conv2d: (16x32x-1x-1xf32) <- (16x32x-1x-1xf32, 32x32x3x3xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            gelu_0, parameter_9, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_9

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(parameter_8, full_int_array_0)
        del parameter_8

        # pd_op.add: (16x32x-1x-1xf32) <- (16x32x-1x-1xf32, 1x32x1x1xf32)
        add_7 = paddle._C_ops.add(conv2d_7, reshape_8)

        # pd_op.conv2d: (16x32x-1x-1xf32) <- (16x32x-1x-1xf32, 32x32x5x5xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            gelu_0, parameter_7, [1, 1], [2, 2], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_7

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(parameter_6, full_int_array_0)
        del parameter_6

        # pd_op.add: (16x32x-1x-1xf32) <- (16x32x-1x-1xf32, 1x32x1x1xf32)
        add_8 = paddle._C_ops.add(conv2d_8, reshape_9)

        # pd_op.conv2d: (16x32x-1x-1xf32) <- (16x32x-1x-1xf32, 32x32x7x7xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            gelu_0, parameter_5, [1, 1], [3, 3], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_5

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(parameter_4, full_int_array_0)
        del parameter_4

        # pd_op.add: (16x32x-1x-1xf32) <- (16x32x-1x-1xf32, 1x32x1x1xf32)
        add_9 = paddle._C_ops.add(conv2d_9, reshape_10)

        # pd_op.conv2d: (16x32x-1x-1xf32) <- (16x32x-1x-1xf32, 32x32x9x9xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            gelu_0, parameter_3, [1, 1], [4, 4], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_3

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        reshape_11 = paddle._C_ops.reshape(parameter_2, full_int_array_0)
        del parameter_2

        # pd_op.add: (16x32x-1x-1xf32) <- (16x32x-1x-1xf32, 1x32x1x1xf32)
        add_10 = paddle._C_ops.add(conv2d_10, reshape_11)

        # pd_op.conv2d: (16x32x-1x-1xf32) <- (16x32x-1x-1xf32, 32x32x11x11xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            gelu_0, parameter_1, [1, 1], [5, 5], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_1

        # pd_op.reshape: (1x32x1x1xf32) <- (32xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(parameter_0, full_int_array_0)
        del full_int_array_0, parameter_0

        # pd_op.add: (16x32x-1x-1xf32) <- (16x32x-1x-1xf32, 1x32x1x1xf32)
        add_11 = paddle._C_ops.add(conv2d_11, reshape_12)

        # builtin.combine: ([16x32x-1x-1xf32, 16x32x-1x-1xf32, 16x32x-1x-1xf32, 16x32x-1x-1xf32, 16x32x-1x-1xf32, 16x32x-1x-1xf32]) <- (16x32x-1x-1xf32, 16x32x-1x-1xf32, 16x32x-1x-1xf32, 16x32x-1x-1xf32, 16x32x-1x-1xf32, 16x32x-1x-1xf32)
        combine_2 = [add_6, add_7, add_8, add_9, add_10, add_11]

        # pd_op.stack: (16x32x-1x-1x6xf32) <- ([16x32x-1x-1xf32, 16x32x-1x-1xf32, 16x32x-1x-1xf32, 16x32x-1x-1xf32, 16x32x-1x-1xf32, 16x32x-1x-1xf32])
        stack_2 = paddle._C_ops.stack(combine_2, -1)
        del combine_2

        # pd_op.mean: (16x32x-1x-1xf32) <- (16x32x-1x-1x6xf32, 1xi64)
        mean_1 = paddle._C_ops.mean(stack_2, full_int_array_1, False)

        # pd_op.transpose: (16x-1x-1x32xf32) <- (16x32x-1x-1xf32)
        transpose_1 = paddle._C_ops.transpose(mean_1, [0, 2, 3, 1])
        del mean_1

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_2 = [16, -1, 32]

        # pd_op.reshape: (16x-1x32xf32) <- (16x-1x-1x32xf32, 3xi64)
        reshape_13 = paddle._C_ops.reshape(transpose_1, full_int_array_2)
        del full_int_array_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [192]

        # pd_op.slice: (16x-1x32xf32) <- (16x-1x32xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            reshape_13, [1], full_int_array_3, full_int_array_4, [1], []
        )
        del (
            add_0,
            add_1,
            add_10,
            add_11,
            add_2,
            add_3,
            add_4,
            add_5,
            add_6,
            add_7,
            add_8,
            add_9,
            assign_0,
            conv2d_0,
            conv2d_1,
            conv2d_10,
            conv2d_11,
            conv2d_2,
            conv2d_3,
            conv2d_4,
            conv2d_5,
            conv2d_6,
            conv2d_7,
            conv2d_8,
            conv2d_9,
            full_int_array_1,
            full_int_array_3,
            full_int_array_4,
            gelu_0,
            mean_0,
            reshape_1,
            reshape_10,
            reshape_11,
            reshape_12,
            reshape_13,
            reshape_2,
            reshape_3,
            reshape_4,
            reshape_5,
            reshape_6,
            reshape_7,
            reshape_8,
            reshape_9,
            stack_1,
            stack_2,
            transpose_0,
            transpose_1,
        )

        return slice_0
