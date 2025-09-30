import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("2"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split_with_num: ([400x1x1xf32, 400x1x1xf32, 400x1x1xf32, 400x1x1xf32]) <- (400x1x4xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(data_0, 4, full_0)
        del data_0

        # builtin.split: (400x1x1xf32, 400x1x1xf32, 400x1x1xf32, 400x1x1xf32) <- ([400x1x1xf32, 400x1x1xf32, 400x1x1xf32, 400x1x1xf32])
        (
            split_0,
            split_1,
            split_2,
            split_3,
        ) = split_with_num_0
        del split_with_num_0

        # pd_op.split_with_num: ([1x4x1xf32, 1x4x1xf32, 1x4x1xf32, 1x4x1xf32]) <- (1x4x4xf32, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(data_1, 4, full_0)
        del data_1, full_0

        # builtin.split: (1x4x1xf32, 1x4x1xf32, 1x4x1xf32, 1x4x1xf32) <- ([1x4x1xf32, 1x4x1xf32, 1x4x1xf32, 1x4x1xf32])
        (
            split_4,
            split_5,
            split_6,
            split_7,
        ) = split_with_num_1
        del split_with_num_1

        # pd_op.maximum: (400x4x1xf32) <- (400x1x1xf32, 1x4x1xf32)
        maximum_0 = paddle._C_ops.maximum(split_0, split_4)

        # pd_op.maximum: (400x4x1xf32) <- (400x1x1xf32, 1x4x1xf32)
        maximum_1 = paddle._C_ops.maximum(split_1, split_5)

        # pd_op.minimum: (400x4x1xf32) <- (400x1x1xf32, 1x4x1xf32)
        minimum_0 = paddle._C_ops.minimum(split_2, split_6)

        # pd_op.minimum: (400x4x1xf32) <- (400x1x1xf32, 1x4x1xf32)
        minimum_1 = paddle._C_ops.minimum(split_3, split_7)

        # pd_op.subtract: (400x4x1xf32) <- (400x4x1xf32, 400x4x1xf32)
        subtract_0 = paddle._C_ops.subtract(minimum_0, maximum_0)
        del maximum_0, minimum_0

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (400x4x1xf32) <- (400x4x1xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(subtract_0, full_1, full_2)
        del subtract_0

        # pd_op.subtract: (400x4x1xf32) <- (400x4x1xf32, 400x4x1xf32)
        subtract_1 = paddle._C_ops.subtract(minimum_1, maximum_1)
        del maximum_1, minimum_1

        # pd_op.clip: (400x4x1xf32) <- (400x4x1xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(subtract_1, full_1, full_2)
        del full_1, full_2, subtract_1

        # pd_op.multiply: (400x4x1xf32) <- (400x4x1xf32, 400x4x1xf32)
        multiply_0 = paddle._C_ops.multiply(clip_0, clip_1)
        del clip_0, clip_1

        # pd_op.subtract: (400x1x1xf32) <- (400x1x1xf32, 400x1x1xf32)
        subtract_2 = paddle._C_ops.subtract(split_2, split_0)

        # pd_op.subtract: (400x1x1xf32) <- (400x1x1xf32, 400x1x1xf32)
        subtract_3 = paddle._C_ops.subtract(split_3, split_1)

        # pd_op.multiply: (400x1x1xf32) <- (400x1x1xf32, 400x1x1xf32)
        multiply_1 = paddle._C_ops.multiply(subtract_2, subtract_3)
        del subtract_2, subtract_3

        # pd_op.subtract: (1x4x1xf32) <- (1x4x1xf32, 1x4x1xf32)
        subtract_4 = paddle._C_ops.subtract(split_6, split_4)

        # pd_op.subtract: (1x4x1xf32) <- (1x4x1xf32, 1x4x1xf32)
        subtract_5 = paddle._C_ops.subtract(split_7, split_5)

        # pd_op.multiply: (1x4x1xf32) <- (1x4x1xf32, 1x4x1xf32)
        multiply_2 = paddle._C_ops.multiply(subtract_4, subtract_5)
        del subtract_4, subtract_5

        # pd_op.add: (400x4x1xf32) <- (400x1x1xf32, 1x4x1xf32)
        add_0 = paddle._C_ops.add(multiply_1, multiply_2)
        del multiply_1, multiply_2

        # pd_op.subtract: (400x4x1xf32) <- (400x4x1xf32, 400x4x1xf32)
        subtract_6 = paddle._C_ops.subtract(add_0, multiply_0)
        del add_0

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (400x4x1xf32) <- (400x4x1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(subtract_6, full_3, float("1e-10"), True)
        del subtract_6

        # pd_op.divide: (400x4x1xf32) <- (400x4x1xf32, 400x4x1xf32)
        divide_0 = paddle._C_ops.divide(multiply_0, scale_1)
        del multiply_0

        # pd_op.minimum: (400x4x1xf32) <- (400x1x1xf32, 1x4x1xf32)
        minimum_2 = paddle._C_ops.minimum(split_0, split_4)
        del split_0, split_4

        # pd_op.minimum: (400x4x1xf32) <- (400x1x1xf32, 1x4x1xf32)
        minimum_3 = paddle._C_ops.minimum(split_1, split_5)
        del split_1, split_5

        # pd_op.maximum: (400x4x1xf32) <- (400x1x1xf32, 1x4x1xf32)
        maximum_2 = paddle._C_ops.maximum(split_2, split_6)
        del split_2, split_6

        # pd_op.maximum: (400x4x1xf32) <- (400x1x1xf32, 1x4x1xf32)
        maximum_3 = paddle._C_ops.maximum(split_3, split_7)
        del split_3, split_7

        # pd_op.subtract: (400x4x1xf32) <- (400x4x1xf32, 400x4x1xf32)
        subtract_7 = paddle._C_ops.subtract(maximum_2, minimum_2)
        del maximum_2, minimum_2

        # pd_op.subtract: (400x4x1xf32) <- (400x4x1xf32, 400x4x1xf32)
        subtract_8 = paddle._C_ops.subtract(maximum_3, minimum_3)
        del maximum_3, minimum_3

        # pd_op.multiply: (400x4x1xf32) <- (400x4x1xf32, 400x4x1xf32)
        multiply_3 = paddle._C_ops.multiply(subtract_7, subtract_8)
        del subtract_7, subtract_8

        # pd_op.scale: (400x4x1xf32) <- (400x4x1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(multiply_3, full_3, float("1e-10"), True)
        del multiply_3

        # pd_op.subtract: (400x4x1xf32) <- (400x4x1xf32, 400x4x1xf32)
        subtract_9 = paddle._C_ops.subtract(scale_2, scale_1)
        del scale_1

        # pd_op.divide: (400x4x1xf32) <- (400x4x1xf32, 400x4x1xf32)
        divide_1 = paddle._C_ops.divide(subtract_9, scale_2)
        del scale_2, subtract_9

        # pd_op.subtract: (400x4x1xf32) <- (400x4x1xf32, 400x4x1xf32)
        subtract_10 = paddle._C_ops.subtract(divide_0, divide_1)
        del divide_0, divide_1

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (400x4x1xf32) <- (400x4x1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(subtract_10, full_4, float("1"), True)
        del full_4, subtract_10

        # pd_op.scale: (400x4x1xf32) <- (400x4x1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(scale_3, full_3, float("0"), True)
        del full_3, scale_3

        return scale_0
