import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("2"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split_with_num: ([6x21504x1xf32, 6x21504x1xf32, 6x21504x1xf32, 6x21504x1xf32, 6x21504x1xf32]) <- (6x21504x5xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(data_0, 5, full_0)
        del data_0, full_0

        # builtin.split: (6x21504x1xf32, 6x21504x1xf32, 6x21504x1xf32, 6x21504x1xf32, 6x21504x1xf32) <- ([6x21504x1xf32, 6x21504x1xf32, 6x21504x1xf32, 6x21504x1xf32, 6x21504x1xf32])
        (
            split_0,
            split_1,
            split_2,
            split_3,
            split_4,
        ) = split_with_num_0
        del split_with_num_0

        # pd_op.cos: (6x21504x1xf32) <- (6x21504x1xf32)
        cos_0 = paddle._C_ops.cos(split_4)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (6x21504x1xf32) <- (6x21504x1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(cos_0, full_1, float("0"), True)
        del cos_0

        # pd_op.sin: (6x21504x1xf32) <- (6x21504x1xf32)
        sin_0 = paddle._C_ops.sin(split_4)
        del split_4

        # pd_op.scale: (6x21504x1xf32) <- (6x21504x1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(sin_0, full_1, float("0"), True)
        del full_1, sin_0

        # pd_op.multiply: (6x21504x1xf32) <- (6x21504x1xf32, 6x21504x1xf32)
        multiply_0 = paddle._C_ops.multiply(scale_0, split_2)

        # pd_op.multiply: (6x21504x1xf32) <- (6x21504x1xf32, 6x21504x1xf32)
        multiply_1 = paddle._C_ops.multiply(scale_1, split_2)
        del split_2

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (6x21504x1xf32) <- (6x21504x1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(scale_1, full_2, float("0"), True)
        del full_2, scale_1

        # pd_op.multiply: (6x21504x1xf32) <- (6x21504x1xf32, 6x21504x1xf32)
        multiply_2 = paddle._C_ops.multiply(scale_2, split_3)
        del scale_2

        # pd_op.multiply: (6x21504x1xf32) <- (6x21504x1xf32, 6x21504x1xf32)
        multiply_3 = paddle._C_ops.multiply(scale_0, split_3)
        del scale_0, split_3

        # pd_op.add: (6x21504x1xf32) <- (6x21504x1xf32, 6x21504x1xf32)
        add_0 = paddle._C_ops.add(split_0, multiply_0)

        # pd_op.add: (6x21504x1xf32) <- (6x21504x1xf32, 6x21504x1xf32)
        add_1 = paddle._C_ops.add(add_0, multiply_2)

        # pd_op.add: (6x21504x1xf32) <- (6x21504x1xf32, 6x21504x1xf32)
        add_2 = paddle._C_ops.add(split_1, multiply_1)

        # pd_op.add: (6x21504x1xf32) <- (6x21504x1xf32, 6x21504x1xf32)
        add_3 = paddle._C_ops.add(add_2, multiply_3)

        # pd_op.subtract: (6x21504x1xf32) <- (6x21504x1xf32, 6x21504x1xf32)
        subtract_0 = paddle._C_ops.subtract(split_0, multiply_0)
        del multiply_0, split_0

        # pd_op.add: (6x21504x1xf32) <- (6x21504x1xf32, 6x21504x1xf32)
        add_4 = paddle._C_ops.add(subtract_0, multiply_2)

        # pd_op.subtract: (6x21504x1xf32) <- (6x21504x1xf32, 6x21504x1xf32)
        subtract_1 = paddle._C_ops.subtract(split_1, multiply_1)
        del multiply_1, split_1

        # pd_op.add: (6x21504x1xf32) <- (6x21504x1xf32, 6x21504x1xf32)
        add_5 = paddle._C_ops.add(subtract_1, multiply_3)

        # pd_op.subtract: (6x21504x1xf32) <- (6x21504x1xf32, 6x21504x1xf32)
        subtract_2 = paddle._C_ops.subtract(subtract_0, multiply_2)
        del subtract_0

        # pd_op.subtract: (6x21504x1xf32) <- (6x21504x1xf32, 6x21504x1xf32)
        subtract_3 = paddle._C_ops.subtract(subtract_1, multiply_3)
        del subtract_1

        # pd_op.subtract: (6x21504x1xf32) <- (6x21504x1xf32, 6x21504x1xf32)
        subtract_4 = paddle._C_ops.subtract(add_0, multiply_2)
        del add_0, multiply_2

        # pd_op.subtract: (6x21504x1xf32) <- (6x21504x1xf32, 6x21504x1xf32)
        subtract_5 = paddle._C_ops.subtract(add_2, multiply_3)
        del add_2, multiply_3

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([6x21504x1xf32, 6x21504x1xf32, 6x21504x1xf32, 6x21504x1xf32, 6x21504x1xf32, 6x21504x1xf32, 6x21504x1xf32, 6x21504x1xf32]) <- (6x21504x1xf32, 6x21504x1xf32, 6x21504x1xf32, 6x21504x1xf32, 6x21504x1xf32, 6x21504x1xf32, 6x21504x1xf32, 6x21504x1xf32)
        combine_0 = [
            add_1,
            add_3,
            add_4,
            add_5,
            subtract_2,
            subtract_3,
            subtract_4,
            subtract_5,
        ]
        del add_1, add_3, add_4, add_5, subtract_2, subtract_3, subtract_4, subtract_5

        # pd_op.concat: (6x21504x8xf32) <- ([6x21504x1xf32, 6x21504x1xf32, 6x21504x1xf32, 6x21504x1xf32, 6x21504x1xf32, 6x21504x1xf32, 6x21504x1xf32, 6x21504x1xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_3)
        del combine_0

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split_with_num: ([6x1xf32, 6x1xf32]) <- (6x2xf32, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(data_1, 2, full_4)
        del data_1, full_4

        # builtin.split: (6x1xf32, 6x1xf32) <- ([6x1xf32, 6x1xf32])
        (
            split_5,
            split_6,
        ) = split_with_num_1
        del split_with_num_1

        # builtin.combine: ([6x1xf32, 6x1xf32, 6x1xf32, 6x1xf32, 6x1xf32, 6x1xf32, 6x1xf32, 6x1xf32]) <- (6x1xf32, 6x1xf32, 6x1xf32, 6x1xf32, 6x1xf32, 6x1xf32, 6x1xf32, 6x1xf32)
        combine_1 = [
            split_6,
            split_5,
            split_6,
            split_5,
            split_6,
            split_5,
            split_6,
            split_5,
        ]
        del split_5, split_6

        # pd_op.concat: (6x8xf32) <- ([6x1xf32, 6x1xf32, 6x1xf32, 6x1xf32, 6x1xf32, 6x1xf32, 6x1xf32, 6x1xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_3)
        del combine_1, full_3

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_0 = [-1, 1, 8]

        # pd_op.reshape: (6x1x8xf32) <- (6x8xf32, 3xi64)
        reshape_0 = paddle._C_ops.reshape(concat_1, full_int_array_0)
        del concat_1, full_int_array_0

        # pd_op.divide: (6x21504x8xf32) <- (6x21504x8xf32, 6x1x8xf32)
        divide_0 = paddle._C_ops.divide(concat_0, reshape_0)
        del concat_0, reshape_0

        return divide_0
