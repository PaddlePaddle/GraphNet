import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7):
        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([1x-1x4xf32, 1x-1x4xf32]) <- (1x-1x4xf32, 1x-1x4xf32)
        combine_0 = [data_0, data_1]
        del data_0, data_1

        # pd_op.concat: (1x-1x4xf32) <- ([1x-1x4xf32, 1x-1x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)
        del combine_0

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x4xf32, -1x4xf32]) <- (-1x4xf32, -1x4xf32)
        combine_1 = [data_4, data_5]
        del data_4, data_5

        # pd_op.concat: (-1x4xf32) <- ([-1x4xf32, -1x4xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_1)
        del combine_1, full_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [3]

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            concat_1, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            concat_1, [1], full_int_array_2, full_int_array_3, [1], [1]
        )

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_0 = paddle._C_ops.subtract(slice_0, slice_1)
        del slice_0

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(subtract_0, full_2, float("0"), True)
        del subtract_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [4]

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            concat_1, [1], full_int_array_1, full_int_array_4, [1], [1]
        )

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            concat_1, [1], full_int_array_3, full_int_array_0, [1], [1]
        )
        del concat_1

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_1 = paddle._C_ops.subtract(slice_2, slice_3)
        del slice_2

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(subtract_1, full_2, float("0"), True)
        del full_2, subtract_1

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(scale_0, full_3, float("0"), True)

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_0 = paddle._C_ops.add(slice_1, scale_2)
        del scale_2, slice_1

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(scale_1, full_3, float("0"), True)

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_1 = paddle._C_ops.add(slice_3, scale_3)
        del scale_3, slice_3

        # pd_op.slice: (1x-1xf32) <- (1x-1x4xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            concat_0, [2], full_int_array_2, full_int_array_3, [1], [2]
        )

        # pd_op.multiply: (1x-1xf32) <- (1x-1xf32, -1xf32)
        multiply_1 = paddle._C_ops.multiply(slice_4, scale_0)
        del slice_4

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x-1xf32) <- (1x-1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(multiply_1, full_4, float("0"), True)
        del multiply_1

        # pd_op.add: (1x-1xf32) <- (-1xf32, 1x-1xf32)
        add_2 = paddle._C_ops.add(add_0, scale_4)
        del add_0, scale_4

        # pd_op.slice: (1x-1xf32) <- (1x-1x4xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            concat_0, [2], full_int_array_3, full_int_array_0, [1], [2]
        )

        # pd_op.multiply: (1x-1xf32) <- (1x-1xf32, -1xf32)
        multiply_2 = paddle._C_ops.multiply(slice_5, scale_1)
        del slice_5

        # pd_op.scale: (1x-1xf32) <- (1x-1xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(multiply_2, full_4, float("0"), True)
        del full_4, multiply_2

        # pd_op.add: (1x-1xf32) <- (-1xf32, 1x-1xf32)
        add_3 = paddle._C_ops.add(add_1, scale_5)
        del add_1, scale_5

        # pd_op.slice: (1x-1xf32) <- (1x-1x4xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            concat_0, [2], full_int_array_0, full_int_array_1, [1], [2]
        )

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0.2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x-1xf32) <- (1x-1xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(slice_6, full_5, float("0"), True)
        del slice_6

        # pd_op.exp: (1x-1xf32) <- (1x-1xf32)
        exp_0 = paddle._C_ops.exp(scale_6)
        del scale_6

        # pd_op.multiply: (1x-1xf32) <- (1x-1xf32, -1xf32)
        multiply_3 = paddle._C_ops.multiply(exp_0, scale_0)
        del exp_0, scale_0

        # pd_op.slice: (1x-1xf32) <- (1x-1x4xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            concat_0, [2], full_int_array_1, full_int_array_4, [1], [2]
        )
        del concat_0, full_int_array_1, full_int_array_4

        # pd_op.scale: (1x-1xf32) <- (1x-1xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(slice_7, full_5, float("0"), True)
        del full_5, slice_7

        # pd_op.exp: (1x-1xf32) <- (1x-1xf32)
        exp_1 = paddle._C_ops.exp(scale_7)
        del scale_7

        # pd_op.multiply: (1x-1xf32) <- (1x-1xf32, -1xf32)
        multiply_4 = paddle._C_ops.multiply(exp_1, scale_1)
        del exp_1, scale_1

        # pd_op.scale: (1x-1xf32) <- (1x-1xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(multiply_3, full_3, float("0"), True)
        del multiply_3

        # pd_op.subtract: (1x-1xf32) <- (1x-1xf32, 1x-1xf32)
        subtract_2 = paddle._C_ops.subtract(add_2, scale_8)

        # pd_op.scale: (1x-1xf32) <- (1x-1xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(multiply_4, full_3, float("0"), True)
        del full_3, multiply_4

        # pd_op.subtract: (1x-1xf32) <- (1x-1xf32, 1x-1xf32)
        subtract_3 = paddle._C_ops.subtract(add_3, scale_9)

        # pd_op.add: (1x-1xf32) <- (1x-1xf32, 1x-1xf32)
        add_4 = paddle._C_ops.add(add_2, scale_8)
        del add_2, scale_8

        # pd_op.add: (1x-1xf32) <- (1x-1xf32, 1x-1xf32)
        add_5 = paddle._C_ops.add(add_3, scale_9)
        del add_3, scale_9

        # builtin.combine: ([1x-1xf32, 1x-1xf32, 1x-1xf32, 1x-1xf32]) <- (1x-1xf32, 1x-1xf32, 1x-1xf32, 1x-1xf32)
        combine_2 = [subtract_2, subtract_3, add_4, add_5]
        del add_4, add_5, subtract_2, subtract_3

        # pd_op.stack: (1x-1x4xf32) <- ([1x-1xf32, 1x-1xf32, 1x-1xf32, 1x-1xf32])
        stack_0 = paddle._C_ops.stack(combine_2, -1)
        del combine_2

        # pd_op.slice: (1xf32) <- (1x2xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            data_6, [1], full_int_array_2, full_int_array_3, [1], [1]
        )

        # pd_op.slice: (1xf32) <- (1x2xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            data_7, [1], full_int_array_2, full_int_array_3, [1], [1]
        )
        del full_int_array_2

        # pd_op.divide: (1xf32) <- (1xf32, 1xf32)
        divide_0 = paddle._C_ops.divide(slice_8, slice_9)
        del slice_8, slice_9

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [-1]

        # pd_op.unsqueeze: (1x1xf32) <- (1xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(divide_0, full_int_array_5)
        del divide_0

        # pd_op.slice: (1xf32) <- (1x2xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            data_6, [1], full_int_array_3, full_int_array_0, [1], [1]
        )
        del data_6

        # pd_op.slice: (1xf32) <- (1x2xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            data_7, [1], full_int_array_3, full_int_array_0, [1], [1]
        )
        del data_7, full_int_array_0, full_int_array_3

        # pd_op.divide: (1xf32) <- (1xf32, 1xf32)
        divide_1 = paddle._C_ops.divide(slice_10, slice_11)
        del slice_10, slice_11

        # pd_op.unsqueeze: (1x1xf32) <- (1xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(divide_1, full_int_array_5)
        del divide_1, full_int_array_5

        # builtin.combine: ([1x1xf32, 1x1xf32, 1x1xf32, 1x1xf32]) <- (1x1xf32, 1x1xf32, 1x1xf32, 1x1xf32)
        combine_3 = [unsqueeze_1, unsqueeze_0, unsqueeze_1, unsqueeze_0]
        del unsqueeze_0, unsqueeze_1

        # pd_op.stack: (1x1x4xf32) <- ([1x1xf32, 1x1xf32, 1x1xf32, 1x1xf32])
        stack_1 = paddle._C_ops.stack(combine_3, -1)
        del combine_3

        # pd_op.multiply: (1x-1x4xf32) <- (1x-1x4xf32, 1x1x4xf32)
        multiply_0 = paddle._C_ops.multiply(stack_0, stack_1)
        del stack_0, stack_1

        # builtin.combine: ([1x-1x2xf32, 1x-1x2xf32]) <- (1x-1x2xf32, 1x-1x2xf32)
        combine_4 = [data_2, data_3]
        del data_2, data_3

        # pd_op.concat: (1x-1x2xf32) <- ([1x-1x2xf32, 1x-1x2xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_4, full_0)
        del combine_4, full_0

        # pd_op.softmax: (1x-1x2xf32) <- (1x-1x2xf32)
        softmax_0 = paddle._C_ops.softmax(concat_2, -1)
        del concat_2

        # pd_op.transpose: (1x2x-1xf32) <- (1x-1x2xf32)
        transpose_0 = paddle._C_ops.transpose(softmax_0, [0, 2, 1])
        del softmax_0

        return multiply_0, transpose_0
