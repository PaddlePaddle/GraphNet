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
    ):
        # pd_op.sigmoid: (1x4x-1x-1xf32) <- (1x4x-1x-1xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(data_5)
        del data_5

        # pd_op.flatten: (1x4x-1xf32) <- (1x4x-1x-1xf32)
        flatten_0 = paddle._C_ops.flatten(sigmoid_0, 2, 3)
        del sigmoid_0

        # pd_op.transpose: (1x-1x4xf32) <- (1x4x-1xf32)
        transpose_1 = paddle._C_ops.transpose(flatten_0, [0, 2, 1])
        del flatten_0

        # pd_op.sigmoid: (1x1x-1x-1xf32) <- (1x1x-1x-1xf32)
        sigmoid_1 = paddle._C_ops.sigmoid(data_15)
        del data_15

        # pd_op.flatten: (1x1x-1xf32) <- (1x1x-1x-1xf32)
        flatten_1 = paddle._C_ops.flatten(sigmoid_1, 2, 3)
        del sigmoid_1

        # pd_op.transpose: (1x-1x1xf32) <- (1x1x-1xf32)
        transpose_2 = paddle._C_ops.transpose(flatten_1, [0, 2, 1])
        del flatten_1

        # pd_op.multiply: (1x-1x4xf32) <- (1x-1x4xf32, 1x-1x1xf32)
        multiply_0 = paddle._C_ops.multiply(transpose_1, transpose_2)
        del transpose_1, transpose_2

        # pd_op.flatten: (1x4x-1xf32) <- (1x4x-1x-1xf32)
        flatten_2 = paddle._C_ops.flatten(data_10, 2, 3)
        del data_10

        # pd_op.transpose: (1x-1x4xf32) <- (1x4x-1xf32)
        transpose_3 = paddle._C_ops.transpose(flatten_2, [0, 2, 1])
        del flatten_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (-1xf32) <- (-1x2xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_0, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.slice: (1x-1xf32) <- (1x-1x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            transpose_3, [2], full_int_array_0, full_int_array_1, [1], [2]
        )

        # pd_op.subtract: (1x-1xf32) <- (-1xf32, 1x-1xf32)
        subtract_0 = paddle._C_ops.subtract(slice_0, slice_1)
        del slice_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        # pd_op.slice: (-1xf32) <- (-1x2xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            data_0, [1], full_int_array_1, full_int_array_2, [1], [1]
        )
        del data_0

        # pd_op.slice: (1x-1xf32) <- (1x-1x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            transpose_3, [2], full_int_array_1, full_int_array_2, [1], [2]
        )

        # pd_op.subtract: (1x-1xf32) <- (-1xf32, 1x-1xf32)
        subtract_1 = paddle._C_ops.subtract(slice_2, slice_3)
        del slice_3

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [3]

        # pd_op.slice: (1x-1xf32) <- (1x-1x4xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            transpose_3, [2], full_int_array_2, full_int_array_3, [1], [2]
        )

        # pd_op.add: (1x-1xf32) <- (-1xf32, 1x-1xf32)
        add_0 = paddle._C_ops.add(slice_0, slice_4)
        del slice_0, slice_4

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [4]

        # pd_op.slice: (1x-1xf32) <- (1x-1x4xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            transpose_3, [2], full_int_array_3, full_int_array_4, [1], [2]
        )
        del transpose_3

        # pd_op.add: (1x-1xf32) <- (-1xf32, 1x-1xf32)
        add_1 = paddle._C_ops.add(slice_2, slice_5)
        del slice_2, slice_5

        # builtin.combine: ([1x-1xf32, 1x-1xf32, 1x-1xf32, 1x-1xf32]) <- (1x-1xf32, 1x-1xf32, 1x-1xf32, 1x-1xf32)
        combine_0 = [subtract_0, subtract_1, add_0, add_1]
        del add_0, add_1, subtract_0, subtract_1

        # pd_op.stack: (1x4x-1xf32) <- ([1x-1xf32, 1x-1xf32, 1x-1xf32, 1x-1xf32])
        stack_0 = paddle._C_ops.stack(combine_0, 1)
        del combine_0

        # pd_op.transpose: (1x-1x4xf32) <- (1x4x-1xf32)
        transpose_4 = paddle._C_ops.transpose(stack_0, [0, 2, 1])
        del stack_0

        # pd_op.sigmoid: (1x4x-1x-1xf32) <- (1x4x-1x-1xf32)
        sigmoid_2 = paddle._C_ops.sigmoid(data_6)
        del data_6

        # pd_op.flatten: (1x4x-1xf32) <- (1x4x-1x-1xf32)
        flatten_3 = paddle._C_ops.flatten(sigmoid_2, 2, 3)
        del sigmoid_2

        # pd_op.transpose: (1x-1x4xf32) <- (1x4x-1xf32)
        transpose_5 = paddle._C_ops.transpose(flatten_3, [0, 2, 1])
        del flatten_3

        # pd_op.sigmoid: (1x1x-1x-1xf32) <- (1x1x-1x-1xf32)
        sigmoid_3 = paddle._C_ops.sigmoid(data_16)
        del data_16

        # pd_op.flatten: (1x1x-1xf32) <- (1x1x-1x-1xf32)
        flatten_4 = paddle._C_ops.flatten(sigmoid_3, 2, 3)
        del sigmoid_3

        # pd_op.transpose: (1x-1x1xf32) <- (1x1x-1xf32)
        transpose_6 = paddle._C_ops.transpose(flatten_4, [0, 2, 1])
        del flatten_4

        # pd_op.multiply: (1x-1x4xf32) <- (1x-1x4xf32, 1x-1x1xf32)
        multiply_1 = paddle._C_ops.multiply(transpose_5, transpose_6)
        del transpose_5, transpose_6

        # pd_op.flatten: (1x4x-1xf32) <- (1x4x-1x-1xf32)
        flatten_5 = paddle._C_ops.flatten(data_11, 2, 3)
        del data_11

        # pd_op.transpose: (1x-1x4xf32) <- (1x4x-1xf32)
        transpose_7 = paddle._C_ops.transpose(flatten_5, [0, 2, 1])
        del flatten_5

        # pd_op.slice: (-1xf32) <- (-1x2xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            data_1, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.slice: (1x-1xf32) <- (1x-1x4xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            transpose_7, [2], full_int_array_0, full_int_array_1, [1], [2]
        )

        # pd_op.subtract: (1x-1xf32) <- (-1xf32, 1x-1xf32)
        subtract_2 = paddle._C_ops.subtract(slice_6, slice_7)
        del slice_7

        # pd_op.slice: (-1xf32) <- (-1x2xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            data_1, [1], full_int_array_1, full_int_array_2, [1], [1]
        )
        del data_1

        # pd_op.slice: (1x-1xf32) <- (1x-1x4xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            transpose_7, [2], full_int_array_1, full_int_array_2, [1], [2]
        )

        # pd_op.subtract: (1x-1xf32) <- (-1xf32, 1x-1xf32)
        subtract_3 = paddle._C_ops.subtract(slice_8, slice_9)
        del slice_9

        # pd_op.slice: (1x-1xf32) <- (1x-1x4xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            transpose_7, [2], full_int_array_2, full_int_array_3, [1], [2]
        )

        # pd_op.add: (1x-1xf32) <- (-1xf32, 1x-1xf32)
        add_2 = paddle._C_ops.add(slice_6, slice_10)
        del slice_10, slice_6

        # pd_op.slice: (1x-1xf32) <- (1x-1x4xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            transpose_7, [2], full_int_array_3, full_int_array_4, [1], [2]
        )
        del transpose_7

        # pd_op.add: (1x-1xf32) <- (-1xf32, 1x-1xf32)
        add_3 = paddle._C_ops.add(slice_8, slice_11)
        del slice_11, slice_8

        # builtin.combine: ([1x-1xf32, 1x-1xf32, 1x-1xf32, 1x-1xf32]) <- (1x-1xf32, 1x-1xf32, 1x-1xf32, 1x-1xf32)
        combine_1 = [subtract_2, subtract_3, add_2, add_3]
        del add_2, add_3, subtract_2, subtract_3

        # pd_op.stack: (1x4x-1xf32) <- ([1x-1xf32, 1x-1xf32, 1x-1xf32, 1x-1xf32])
        stack_1 = paddle._C_ops.stack(combine_1, 1)
        del combine_1

        # pd_op.transpose: (1x-1x4xf32) <- (1x4x-1xf32)
        transpose_8 = paddle._C_ops.transpose(stack_1, [0, 2, 1])
        del stack_1

        # pd_op.sigmoid: (1x4x-1x-1xf32) <- (1x4x-1x-1xf32)
        sigmoid_4 = paddle._C_ops.sigmoid(data_7)
        del data_7

        # pd_op.flatten: (1x4x-1xf32) <- (1x4x-1x-1xf32)
        flatten_6 = paddle._C_ops.flatten(sigmoid_4, 2, 3)
        del sigmoid_4

        # pd_op.transpose: (1x-1x4xf32) <- (1x4x-1xf32)
        transpose_9 = paddle._C_ops.transpose(flatten_6, [0, 2, 1])
        del flatten_6

        # pd_op.sigmoid: (1x1x-1x-1xf32) <- (1x1x-1x-1xf32)
        sigmoid_5 = paddle._C_ops.sigmoid(data_17)
        del data_17

        # pd_op.flatten: (1x1x-1xf32) <- (1x1x-1x-1xf32)
        flatten_7 = paddle._C_ops.flatten(sigmoid_5, 2, 3)
        del sigmoid_5

        # pd_op.transpose: (1x-1x1xf32) <- (1x1x-1xf32)
        transpose_10 = paddle._C_ops.transpose(flatten_7, [0, 2, 1])
        del flatten_7

        # pd_op.multiply: (1x-1x4xf32) <- (1x-1x4xf32, 1x-1x1xf32)
        multiply_2 = paddle._C_ops.multiply(transpose_9, transpose_10)
        del transpose_10, transpose_9

        # pd_op.flatten: (1x4x-1xf32) <- (1x4x-1x-1xf32)
        flatten_8 = paddle._C_ops.flatten(data_12, 2, 3)
        del data_12

        # pd_op.transpose: (1x-1x4xf32) <- (1x4x-1xf32)
        transpose_11 = paddle._C_ops.transpose(flatten_8, [0, 2, 1])
        del flatten_8

        # pd_op.slice: (-1xf32) <- (-1x2xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            data_2, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.slice: (1x-1xf32) <- (1x-1x4xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            transpose_11, [2], full_int_array_0, full_int_array_1, [1], [2]
        )

        # pd_op.subtract: (1x-1xf32) <- (-1xf32, 1x-1xf32)
        subtract_4 = paddle._C_ops.subtract(slice_12, slice_13)
        del slice_13

        # pd_op.slice: (-1xf32) <- (-1x2xf32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            data_2, [1], full_int_array_1, full_int_array_2, [1], [1]
        )
        del data_2

        # pd_op.slice: (1x-1xf32) <- (1x-1x4xf32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            transpose_11, [2], full_int_array_1, full_int_array_2, [1], [2]
        )

        # pd_op.subtract: (1x-1xf32) <- (-1xf32, 1x-1xf32)
        subtract_5 = paddle._C_ops.subtract(slice_14, slice_15)
        del slice_15

        # pd_op.slice: (1x-1xf32) <- (1x-1x4xf32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            transpose_11, [2], full_int_array_2, full_int_array_3, [1], [2]
        )

        # pd_op.add: (1x-1xf32) <- (-1xf32, 1x-1xf32)
        add_4 = paddle._C_ops.add(slice_12, slice_16)
        del slice_12, slice_16

        # pd_op.slice: (1x-1xf32) <- (1x-1x4xf32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            transpose_11, [2], full_int_array_3, full_int_array_4, [1], [2]
        )
        del transpose_11

        # pd_op.add: (1x-1xf32) <- (-1xf32, 1x-1xf32)
        add_5 = paddle._C_ops.add(slice_14, slice_17)
        del slice_14, slice_17

        # builtin.combine: ([1x-1xf32, 1x-1xf32, 1x-1xf32, 1x-1xf32]) <- (1x-1xf32, 1x-1xf32, 1x-1xf32, 1x-1xf32)
        combine_2 = [subtract_4, subtract_5, add_4, add_5]
        del add_4, add_5, subtract_4, subtract_5

        # pd_op.stack: (1x4x-1xf32) <- ([1x-1xf32, 1x-1xf32, 1x-1xf32, 1x-1xf32])
        stack_2 = paddle._C_ops.stack(combine_2, 1)
        del combine_2

        # pd_op.transpose: (1x-1x4xf32) <- (1x4x-1xf32)
        transpose_12 = paddle._C_ops.transpose(stack_2, [0, 2, 1])
        del stack_2

        # pd_op.sigmoid: (1x4x-1x-1xf32) <- (1x4x-1x-1xf32)
        sigmoid_6 = paddle._C_ops.sigmoid(data_8)
        del data_8

        # pd_op.flatten: (1x4x-1xf32) <- (1x4x-1x-1xf32)
        flatten_9 = paddle._C_ops.flatten(sigmoid_6, 2, 3)
        del sigmoid_6

        # pd_op.transpose: (1x-1x4xf32) <- (1x4x-1xf32)
        transpose_13 = paddle._C_ops.transpose(flatten_9, [0, 2, 1])
        del flatten_9

        # pd_op.sigmoid: (1x1x-1x-1xf32) <- (1x1x-1x-1xf32)
        sigmoid_7 = paddle._C_ops.sigmoid(data_18)
        del data_18

        # pd_op.flatten: (1x1x-1xf32) <- (1x1x-1x-1xf32)
        flatten_10 = paddle._C_ops.flatten(sigmoid_7, 2, 3)
        del sigmoid_7

        # pd_op.transpose: (1x-1x1xf32) <- (1x1x-1xf32)
        transpose_14 = paddle._C_ops.transpose(flatten_10, [0, 2, 1])
        del flatten_10

        # pd_op.multiply: (1x-1x4xf32) <- (1x-1x4xf32, 1x-1x1xf32)
        multiply_3 = paddle._C_ops.multiply(transpose_13, transpose_14)
        del transpose_13, transpose_14

        # pd_op.flatten: (1x4x-1xf32) <- (1x4x-1x-1xf32)
        flatten_11 = paddle._C_ops.flatten(data_13, 2, 3)
        del data_13

        # pd_op.transpose: (1x-1x4xf32) <- (1x4x-1xf32)
        transpose_15 = paddle._C_ops.transpose(flatten_11, [0, 2, 1])
        del flatten_11

        # pd_op.slice: (-1xf32) <- (-1x2xf32, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(
            data_3, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.slice: (1x-1xf32) <- (1x-1x4xf32, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(
            transpose_15, [2], full_int_array_0, full_int_array_1, [1], [2]
        )

        # pd_op.subtract: (1x-1xf32) <- (-1xf32, 1x-1xf32)
        subtract_6 = paddle._C_ops.subtract(slice_18, slice_19)
        del slice_19

        # pd_op.slice: (-1xf32) <- (-1x2xf32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(
            data_3, [1], full_int_array_1, full_int_array_2, [1], [1]
        )
        del data_3

        # pd_op.slice: (1x-1xf32) <- (1x-1x4xf32, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(
            transpose_15, [2], full_int_array_1, full_int_array_2, [1], [2]
        )

        # pd_op.subtract: (1x-1xf32) <- (-1xf32, 1x-1xf32)
        subtract_7 = paddle._C_ops.subtract(slice_20, slice_21)
        del slice_21

        # pd_op.slice: (1x-1xf32) <- (1x-1x4xf32, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(
            transpose_15, [2], full_int_array_2, full_int_array_3, [1], [2]
        )

        # pd_op.add: (1x-1xf32) <- (-1xf32, 1x-1xf32)
        add_6 = paddle._C_ops.add(slice_18, slice_22)
        del slice_18, slice_22

        # pd_op.slice: (1x-1xf32) <- (1x-1x4xf32, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(
            transpose_15, [2], full_int_array_3, full_int_array_4, [1], [2]
        )
        del transpose_15

        # pd_op.add: (1x-1xf32) <- (-1xf32, 1x-1xf32)
        add_7 = paddle._C_ops.add(slice_20, slice_23)
        del slice_20, slice_23

        # builtin.combine: ([1x-1xf32, 1x-1xf32, 1x-1xf32, 1x-1xf32]) <- (1x-1xf32, 1x-1xf32, 1x-1xf32, 1x-1xf32)
        combine_3 = [subtract_6, subtract_7, add_6, add_7]
        del add_6, add_7, subtract_6, subtract_7

        # pd_op.stack: (1x4x-1xf32) <- ([1x-1xf32, 1x-1xf32, 1x-1xf32, 1x-1xf32])
        stack_3 = paddle._C_ops.stack(combine_3, 1)
        del combine_3

        # pd_op.transpose: (1x-1x4xf32) <- (1x4x-1xf32)
        transpose_16 = paddle._C_ops.transpose(stack_3, [0, 2, 1])
        del stack_3

        # pd_op.sigmoid: (1x4x-1x-1xf32) <- (1x4x-1x-1xf32)
        sigmoid_8 = paddle._C_ops.sigmoid(data_9)
        del data_9

        # pd_op.flatten: (1x4x-1xf32) <- (1x4x-1x-1xf32)
        flatten_12 = paddle._C_ops.flatten(sigmoid_8, 2, 3)
        del sigmoid_8

        # pd_op.transpose: (1x-1x4xf32) <- (1x4x-1xf32)
        transpose_17 = paddle._C_ops.transpose(flatten_12, [0, 2, 1])
        del flatten_12

        # pd_op.sigmoid: (1x1x-1x-1xf32) <- (1x1x-1x-1xf32)
        sigmoid_9 = paddle._C_ops.sigmoid(data_19)
        del data_19

        # pd_op.flatten: (1x1x-1xf32) <- (1x1x-1x-1xf32)
        flatten_13 = paddle._C_ops.flatten(sigmoid_9, 2, 3)
        del sigmoid_9

        # pd_op.transpose: (1x-1x1xf32) <- (1x1x-1xf32)
        transpose_18 = paddle._C_ops.transpose(flatten_13, [0, 2, 1])
        del flatten_13

        # pd_op.multiply: (1x-1x4xf32) <- (1x-1x4xf32, 1x-1x1xf32)
        multiply_4 = paddle._C_ops.multiply(transpose_17, transpose_18)
        del transpose_17, transpose_18

        # pd_op.flatten: (1x4x-1xf32) <- (1x4x-1x-1xf32)
        flatten_14 = paddle._C_ops.flatten(data_14, 2, 3)
        del data_14

        # pd_op.transpose: (1x-1x4xf32) <- (1x4x-1xf32)
        transpose_19 = paddle._C_ops.transpose(flatten_14, [0, 2, 1])
        del flatten_14

        # pd_op.slice: (-1xf32) <- (-1x2xf32, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(
            data_4, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.slice: (1x-1xf32) <- (1x-1x4xf32, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(
            transpose_19, [2], full_int_array_0, full_int_array_1, [1], [2]
        )
        del full_int_array_0

        # pd_op.subtract: (1x-1xf32) <- (-1xf32, 1x-1xf32)
        subtract_8 = paddle._C_ops.subtract(slice_24, slice_25)
        del slice_25

        # pd_op.slice: (-1xf32) <- (-1x2xf32, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(
            data_4, [1], full_int_array_1, full_int_array_2, [1], [1]
        )
        del data_4

        # pd_op.slice: (1x-1xf32) <- (1x-1x4xf32, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(
            transpose_19, [2], full_int_array_1, full_int_array_2, [1], [2]
        )
        del full_int_array_1

        # pd_op.subtract: (1x-1xf32) <- (-1xf32, 1x-1xf32)
        subtract_9 = paddle._C_ops.subtract(slice_26, slice_27)
        del slice_27

        # pd_op.slice: (1x-1xf32) <- (1x-1x4xf32, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(
            transpose_19, [2], full_int_array_2, full_int_array_3, [1], [2]
        )
        del full_int_array_2

        # pd_op.add: (1x-1xf32) <- (-1xf32, 1x-1xf32)
        add_8 = paddle._C_ops.add(slice_24, slice_28)
        del slice_24, slice_28

        # pd_op.slice: (1x-1xf32) <- (1x-1x4xf32, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(
            transpose_19, [2], full_int_array_3, full_int_array_4, [1], [2]
        )
        del full_int_array_3, full_int_array_4, transpose_19

        # pd_op.add: (1x-1xf32) <- (-1xf32, 1x-1xf32)
        add_9 = paddle._C_ops.add(slice_26, slice_29)
        del slice_26, slice_29

        # builtin.combine: ([1x-1xf32, 1x-1xf32, 1x-1xf32, 1x-1xf32]) <- (1x-1xf32, 1x-1xf32, 1x-1xf32, 1x-1xf32)
        combine_4 = [subtract_8, subtract_9, add_8, add_9]
        del add_8, add_9, subtract_8, subtract_9

        # pd_op.stack: (1x4x-1xf32) <- ([1x-1xf32, 1x-1xf32, 1x-1xf32, 1x-1xf32])
        stack_4 = paddle._C_ops.stack(combine_4, 1)
        del combine_4

        # pd_op.transpose: (1x-1x4xf32) <- (1x4x-1xf32)
        transpose_20 = paddle._C_ops.transpose(stack_4, [0, 2, 1])
        del stack_4

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32]) <- (1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32)
        combine_5 = [transpose_4, transpose_8, transpose_12, transpose_16, transpose_20]
        del transpose_12, transpose_16, transpose_20, transpose_4, transpose_8

        # pd_op.concat: (1x-1x4xf32) <- ([1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_5, full_0)
        del combine_5

        # builtin.combine: ([1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32]) <- (1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32)
        combine_6 = [multiply_0, multiply_1, multiply_2, multiply_3, multiply_4]
        del multiply_0, multiply_1, multiply_2, multiply_3, multiply_4

        # pd_op.concat: (1x-1x4xf32) <- ([1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_6, full_0)
        del combine_6

        # pd_op.split_with_num: ([1x1xf32, 1x1xf32]) <- (1x2xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(data_20, 2, full_0)
        del data_20, full_0

        # builtin.split: (1x1xf32, 1x1xf32) <- ([1x1xf32, 1x1xf32])
        (
            split_0,
            split_1,
        ) = split_with_num_0
        del split_with_num_0

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([1x1xf32, 1x1xf32, 1x1xf32, 1x1xf32]) <- (1x1xf32, 1x1xf32, 1x1xf32, 1x1xf32)
        combine_7 = [split_1, split_0, split_1, split_0]
        del split_0, split_1

        # pd_op.concat: (1x4xf32) <- ([1x1xf32, 1x1xf32, 1x1xf32, 1x1xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_7, full_1)
        del combine_7, full_1

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_5 = [-1, 1, 4]

        # pd_op.reshape: (1x1x4xf32) <- (1x4xf32, 3xi64)
        reshape_0 = paddle._C_ops.reshape(concat_2, full_int_array_5)
        del concat_2, full_int_array_5

        # pd_op.divide: (1x-1x4xf32) <- (1x-1x4xf32, 1x1x4xf32)
        divide_0 = paddle._C_ops.divide(concat_0, reshape_0)
        del concat_0, reshape_0

        # pd_op.transpose: (1x4x-1xf32) <- (1x-1x4xf32)
        transpose_0 = paddle._C_ops.transpose(concat_1, [0, 2, 1])
        del concat_1

        return divide_0, transpose_0
