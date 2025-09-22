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
    ):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_1

        # pd_op.slice: (100x4xf32) <- (4x100x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_0, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_1 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_2 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_3 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_4 = full_0

        # pd_op.gather: (1x4xf32) <- (100x4xf32, 1xi64, 1xi32)
        gather_0 = paddle._C_ops.gather(slice_0, data_5, full_0)
        del data_5

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_5 = full_int_array_2

        # pd_op.slice: (100x4xf32) <- (4x100x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_0, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.gather: (1x4xf32) <- (100x4xf32, 1xi64, 1xi32)
        gather_1 = paddle._C_ops.gather(slice_1, data_7, full_0)
        del data_7

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_6 = full_int_array_3

        # pd_op.slice: (100x4xf32) <- (4x100x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            data_0, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.gather: (2x4xf32) <- (100x4xf32, 2xi64, 1xi32)
        gather_2 = paddle._C_ops.gather(slice_2, data_9, full_0)
        del data_9

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [4]

        # pd_op.slice: (100x4xf32) <- (4x100x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            data_0, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del data_0

        # pd_op.gather: (1x4xf32) <- (100x4xf32, 1xi64, 1xi32)
        gather_3 = paddle._C_ops.gather(slice_3, data_11, full_0)
        del data_11

        # builtin.combine: ([1x4xf32, 1x4xf32, 2x4xf32, 1x4xf32]) <- (1x4xf32, 1x4xf32, 2x4xf32, 1x4xf32)
        combine_0 = [gather_0, gather_1, gather_2, gather_3]

        # pd_op.concat: (5x4xf32) <- ([1x4xf32, 1x4xf32, 2x4xf32, 1x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)
        del combine_0

        # pd_op.gather: (1x4xf32) <- (1x4xf32, 1xi64, 1xi32)
        gather_4 = paddle._C_ops.gather(data_1, data_6, full_0)
        del data_1, data_6

        # pd_op.gather: (1x4xf32) <- (1x4xf32, 1xi64, 1xi32)
        gather_5 = paddle._C_ops.gather(data_2, data_8, full_0)
        del data_2, data_8

        # pd_op.gather: (2x4xf32) <- (2x4xf32, 2xi64, 1xi32)
        gather_6 = paddle._C_ops.gather(data_3, data_10, full_0)
        del data_10, data_3

        # pd_op.gather: (1x4xf32) <- (1x4xf32, 1xi64, 1xi32)
        gather_7 = paddle._C_ops.gather(data_4, data_12, full_0)
        del data_12, data_4

        # builtin.combine: ([1x4xf32, 1x4xf32, 2x4xf32, 1x4xf32]) <- (1x4xf32, 1x4xf32, 2x4xf32, 1x4xf32)
        combine_1 = [gather_4, gather_5, gather_6, gather_7]
        del gather_4, gather_5, gather_6, gather_7

        # pd_op.concat: (5x4xf32) <- ([1x4xf32, 1x4xf32, 2x4xf32, 1x4xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_0)
        del combine_1

        # pd_op.subtract: (5x4xf32) <- (5x4xf32, 5x4xf32)
        subtract_0 = paddle._C_ops.subtract(concat_0, concat_1)

        # pd_op.abs: (5x4xf32) <- (5x4xf32)
        abs_0 = paddle._C_ops.abs(subtract_0)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_5 = []

        # pd_op.assign: (0xi64) <- (0xi64)
        assign_7 = full_int_array_5

        # pd_op.sum: (xf32) <- (5x4xf32, 0xi64)
        sum_0 = paddle._C_ops.sum(abs_0, full_int_array_5, None, False)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(sum_0, full_1, float("0"), True)
        del sum_0

        # pd_op.divide: (1xf32) <- (xf32, 1xf32)
        divide_0 = paddle._C_ops.divide(scale_1, data_13)

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_8 = full_2

        # pd_op.split_with_num: ([5x2xf32, 5x2xf32]) <- (5x4xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(concat_0, 2, full_2)

        # builtin.split: (5x2xf32, 5x2xf32) <- ([5x2xf32, 5x2xf32])
        (
            split_0,
            split_1,
        ) = split_with_num_0
        del split_with_num_0

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_9 = full_3

        # pd_op.scale: (5x2xf32) <- (5x2xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(split_1, full_3, float("0"), True)
        del split_1

        # pd_op.assign: (5x2xf32) <- (5x2xf32)
        assign_10 = scale_2

        # pd_op.subtract: (5x2xf32) <- (5x2xf32, 5x2xf32)
        subtract_1 = paddle._C_ops.subtract(split_0, scale_2)

        # pd_op.add: (5x2xf32) <- (5x2xf32, 5x2xf32)
        add_0 = paddle._C_ops.add(split_0, scale_2)

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([5x2xf32, 5x2xf32]) <- (5x2xf32, 5x2xf32)
        combine_2 = [subtract_1, add_0]

        # pd_op.concat: (5x4xf32) <- ([5x2xf32, 5x2xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_4)
        del combine_2

        # pd_op.split_with_num: ([5x2xf32, 5x2xf32]) <- (5x4xf32, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(concat_1, 2, full_2)

        # builtin.split: (5x2xf32, 5x2xf32) <- ([5x2xf32, 5x2xf32])
        (
            split_2,
            split_3,
        ) = split_with_num_1
        del split_with_num_1

        # pd_op.scale: (5x2xf32) <- (5x2xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(split_3, full_3, float("0"), True)
        del split_3

        # pd_op.subtract: (5x2xf32) <- (5x2xf32, 5x2xf32)
        subtract_2 = paddle._C_ops.subtract(split_2, scale_3)

        # pd_op.add: (5x2xf32) <- (5x2xf32, 5x2xf32)
        add_1 = paddle._C_ops.add(split_2, scale_3)
        del scale_3, split_2

        # builtin.combine: ([5x2xf32, 5x2xf32]) <- (5x2xf32, 5x2xf32)
        combine_3 = [subtract_2, add_1]
        del add_1, subtract_2

        # pd_op.concat: (5x4xf32) <- ([5x2xf32, 5x2xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_3, full_4)
        del combine_3

        # pd_op.split_with_num: ([5x1xf32, 5x1xf32, 5x1xf32, 5x1xf32]) <- (5x4xf32, 1xi32)
        split_with_num_2 = paddle._C_ops.split_with_num(concat_2, 4, full_2)
        del concat_2

        # builtin.split: (5x1xf32, 5x1xf32, 5x1xf32, 5x1xf32) <- ([5x1xf32, 5x1xf32, 5x1xf32, 5x1xf32])
        (
            split_4,
            split_5,
            split_6,
            split_7,
        ) = split_with_num_2
        del split_with_num_2

        # pd_op.split_with_num: ([5x1xf32, 5x1xf32, 5x1xf32, 5x1xf32]) <- (5x4xf32, 1xi32)
        split_with_num_3 = paddle._C_ops.split_with_num(concat_3, 4, full_2)
        del concat_3

        # builtin.split: (5x1xf32, 5x1xf32, 5x1xf32, 5x1xf32) <- ([5x1xf32, 5x1xf32, 5x1xf32, 5x1xf32])
        (
            split_8,
            split_9,
            split_10,
            split_11,
        ) = split_with_num_3
        del split_with_num_3

        # pd_op.maximum: (5x1xf32) <- (5x1xf32, 5x1xf32)
        maximum_0 = paddle._C_ops.maximum(split_4, split_8)

        # pd_op.maximum: (5x1xf32) <- (5x1xf32, 5x1xf32)
        maximum_1 = paddle._C_ops.maximum(split_5, split_9)

        # pd_op.minimum: (5x1xf32) <- (5x1xf32, 5x1xf32)
        minimum_0 = paddle._C_ops.minimum(split_6, split_10)

        # pd_op.minimum: (5x1xf32) <- (5x1xf32, 5x1xf32)
        minimum_1 = paddle._C_ops.minimum(split_7, split_11)

        # pd_op.subtract: (5x1xf32) <- (5x1xf32, 5x1xf32)
        subtract_3 = paddle._C_ops.subtract(minimum_0, maximum_0)

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_11 = full_5

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_12 = full_6

        # pd_op.clip: (5x1xf32) <- (5x1xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(subtract_3, full_5, full_6)

        # pd_op.subtract: (5x1xf32) <- (5x1xf32, 5x1xf32)
        subtract_4 = paddle._C_ops.subtract(minimum_1, maximum_1)

        # pd_op.clip: (5x1xf32) <- (5x1xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(subtract_4, full_5, full_6)

        # pd_op.multiply: (5x1xf32) <- (5x1xf32, 5x1xf32)
        multiply_0 = paddle._C_ops.multiply(clip_0, clip_1)

        # pd_op.subtract: (5x1xf32) <- (5x1xf32, 5x1xf32)
        subtract_5 = paddle._C_ops.subtract(split_6, split_4)

        # pd_op.subtract: (5x1xf32) <- (5x1xf32, 5x1xf32)
        subtract_6 = paddle._C_ops.subtract(split_7, split_5)

        # pd_op.multiply: (5x1xf32) <- (5x1xf32, 5x1xf32)
        multiply_1 = paddle._C_ops.multiply(subtract_5, subtract_6)

        # pd_op.subtract: (5x1xf32) <- (5x1xf32, 5x1xf32)
        subtract_7 = paddle._C_ops.subtract(split_10, split_8)

        # pd_op.subtract: (5x1xf32) <- (5x1xf32, 5x1xf32)
        subtract_8 = paddle._C_ops.subtract(split_11, split_9)

        # pd_op.multiply: (5x1xf32) <- (5x1xf32, 5x1xf32)
        multiply_2 = paddle._C_ops.multiply(subtract_7, subtract_8)
        del subtract_7, subtract_8

        # pd_op.add: (5x1xf32) <- (5x1xf32, 5x1xf32)
        add_2 = paddle._C_ops.add(multiply_1, multiply_2)

        # pd_op.subtract: (5x1xf32) <- (5x1xf32, 5x1xf32)
        subtract_9 = paddle._C_ops.subtract(add_2, multiply_0)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_13 = full_7

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_14 = full_7

        # pd_op.scale: (5x1xf32) <- (5x1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(subtract_9, full_7, float("1e-10"), True)
        del subtract_9

        # pd_op.divide: (5x1xf32) <- (5x1xf32, 5x1xf32)
        divide_1 = paddle._C_ops.divide(multiply_0, scale_4)

        # pd_op.minimum: (5x1xf32) <- (5x1xf32, 5x1xf32)
        minimum_2 = paddle._C_ops.minimum(split_4, split_8)

        # pd_op.minimum: (5x1xf32) <- (5x1xf32, 5x1xf32)
        minimum_3 = paddle._C_ops.minimum(split_5, split_9)

        # pd_op.maximum: (5x1xf32) <- (5x1xf32, 5x1xf32)
        maximum_2 = paddle._C_ops.maximum(split_6, split_10)

        # pd_op.maximum: (5x1xf32) <- (5x1xf32, 5x1xf32)
        maximum_3 = paddle._C_ops.maximum(split_7, split_11)

        # pd_op.subtract: (5x1xf32) <- (5x1xf32, 5x1xf32)
        subtract_10 = paddle._C_ops.subtract(maximum_2, minimum_2)

        # pd_op.subtract: (5x1xf32) <- (5x1xf32, 5x1xf32)
        subtract_11 = paddle._C_ops.subtract(maximum_3, minimum_3)

        # pd_op.multiply: (5x1xf32) <- (5x1xf32, 5x1xf32)
        multiply_3 = paddle._C_ops.multiply(subtract_10, subtract_11)

        # pd_op.scale: (5x1xf32) <- (5x1xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(multiply_3, full_7, float("1e-10"), True)
        del multiply_3

        # pd_op.subtract: (5x1xf32) <- (5x1xf32, 5x1xf32)
        subtract_12 = paddle._C_ops.subtract(scale_5, scale_4)

        # pd_op.divide: (5x1xf32) <- (5x1xf32, 5x1xf32)
        divide_2 = paddle._C_ops.divide(subtract_12, scale_5)

        # pd_op.subtract: (5x1xf32) <- (5x1xf32, 5x1xf32)
        subtract_13 = paddle._C_ops.subtract(divide_1, divide_2)

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (5x1xf32) <- (5x1xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(subtract_13, full_8, float("1"), True)
        del subtract_13

        # pd_op.scale: (5x1xf32) <- (5x1xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(scale_6, full_7, float("0"), True)
        del scale_6

        # pd_op.sum: (xf32) <- (5x1xf32, 0xi64)
        sum_1 = paddle._C_ops.sum(scale_7, full_int_array_5, None, False)

        # pd_op.divide: (1xf32) <- (xf32, 1xf32)
        divide_3 = paddle._C_ops.divide(sum_1, data_13)
        del data_13

        # pd_op.full: (1xf32) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1xf32) <- (1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(divide_3, full_9, float("0"), True)
        del (
            abs_0,
            add_0,
            add_2,
            assign_0,
            assign_1,
            assign_10,
            assign_11,
            assign_12,
            assign_13,
            assign_14,
            assign_2,
            assign_3,
            assign_4,
            assign_5,
            assign_6,
            assign_7,
            assign_8,
            assign_9,
            clip_0,
            clip_1,
            concat_0,
            concat_1,
            divide_1,
            divide_2,
            divide_3,
            full_0,
            full_1,
            full_2,
            full_3,
            full_4,
            full_5,
            full_6,
            full_7,
            full_8,
            full_9,
            full_int_array_0,
            full_int_array_1,
            full_int_array_2,
            full_int_array_3,
            full_int_array_4,
            full_int_array_5,
            gather_0,
            gather_1,
            gather_2,
            gather_3,
            maximum_0,
            maximum_1,
            maximum_2,
            maximum_3,
            minimum_0,
            minimum_1,
            minimum_2,
            minimum_3,
            multiply_0,
            multiply_1,
            multiply_2,
            scale_1,
            scale_2,
            scale_4,
            scale_5,
            scale_7,
            slice_0,
            slice_1,
            slice_2,
            slice_3,
            split_0,
            split_10,
            split_11,
            split_4,
            split_5,
            split_6,
            split_7,
            split_8,
            split_9,
            subtract_0,
            subtract_1,
            subtract_10,
            subtract_11,
            subtract_12,
            subtract_3,
            subtract_4,
            subtract_5,
            subtract_6,
            sum_1,
        )

        return divide_0, scale_0
