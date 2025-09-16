import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [-1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_0

        # pd_op.unsqueeze: (1x21504x1xb) <- (1x21504xb, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(data_0, full_int_array_0)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_1 = [1, 1, 5]

        # pd_op.tile: (1x21504x5xb) <- (1x21504x1xb, 3xi64)
        tile_0 = paddle._C_ops.tile(unsqueeze_0, full_int_array_1)
        del full_int_array_1

        # pd_op.masked_select: (-1xf32) <- (1x21504x5xf32, 1x21504x5xb)
        masked_select_0 = paddle._C_ops.masked_select(data_1, tile_0)
        del data_1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [-1, 5]

        # pd_op.reshape: (-1x5xf32) <- (-1xf32, 2xi64)
        reshape_0 = paddle._C_ops.reshape(masked_select_0, full_int_array_2)

        # pd_op.masked_select: (-1xf32) <- (1x21504x5xf32, 1x21504x5xb)
        masked_select_1 = paddle._C_ops.masked_select(data_2, tile_0)
        del data_2

        # pd_op.reshape: (-1x5xf32) <- (-1xf32, 2xi64)
        reshape_1 = paddle._C_ops.reshape(masked_select_1, full_int_array_2)
        del full_int_array_2, masked_select_1

        # pd_op.sum: (1x21504xf32) <- (1x21504x15xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(data_3, full_int_array_0, None, False)
        del data_3

        # pd_op.masked_select: (-1xf32) <- (1x21504xf32, 1x21504xb)
        masked_select_2 = paddle._C_ops.masked_select(sum_0, data_0)
        del data_0, sum_0

        # pd_op.reshape: (-1xf32) <- (-1xf32, 1xi64)
        reshape_2 = paddle._C_ops.reshape(masked_select_2, full_int_array_0)
        del masked_select_2

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_3 = [2, 2, 1]

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split: ([-1x2xf32, -1x2xf32, -1x1xf32]) <- (-1x5xf32, 3xi64, 1xi32)
        split_0 = paddle._C_ops.split(reshape_0, full_int_array_3, full_0)
        del reshape_0

        # builtin.split: (-1x2xf32, -1x2xf32, -1x1xf32) <- ([-1x2xf32, -1x2xf32, -1x1xf32])
        (
            split_1,
            split_2,
            split_3,
        ) = split_0
        del split_0

        # pd_op.pow: (-1x2xf32) <- (-1x2xf32)
        pow_0 = paddle._C_ops.pow(split_2, float("2"))

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0.0833333"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x2xf32) <- (-1x2xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(pow_0, full_1, float("0"), True)
        del pow_0

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x2xf32, -1x2xf32, -1x1xf32]) <- (-1x2xf32, -1x2xf32, -1x1xf32)
        combine_0 = [split_1, scale_0, split_3]

        # pd_op.concat: (-1x5xf32) <- ([-1x2xf32, -1x2xf32, -1x1xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_2)
        del combine_0

        # pd_op.split: ([-1x2xf32, -1x2xf32, -1x1xf32]) <- (-1x5xf32, 3xi64, 1xi32)
        split_4 = paddle._C_ops.split(reshape_1, full_int_array_3, full_0)
        del full_int_array_3

        # builtin.split: (-1x2xf32, -1x2xf32, -1x1xf32) <- ([-1x2xf32, -1x2xf32, -1x1xf32])
        (
            split_5,
            split_6,
            split_7,
        ) = split_4
        del split_4

        # pd_op.pow: (-1x2xf32) <- (-1x2xf32)
        pow_1 = paddle._C_ops.pow(split_6, float("2"))
        del split_6

        # pd_op.scale: (-1x2xf32) <- (-1x2xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(pow_1, full_1, float("0"), True)
        del pow_1

        # builtin.combine: ([-1x2xf32, -1x2xf32, -1x1xf32]) <- (-1x2xf32, -1x2xf32, -1x1xf32)
        combine_1 = [split_5, scale_1, split_7]
        del scale_1, split_5, split_7

        # pd_op.concat: (-1x5xf32) <- ([-1x2xf32, -1x2xf32, -1x1xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_2)
        del combine_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_3 = full_int_array_5

        # pd_op.slice: (-1xf32) <- (-1x5xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            concat_0, [1], full_int_array_4, full_int_array_5, [1], [1]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_4 = full_int_array_6

        # pd_op.slice: (-1xf32) <- (-1x5xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            concat_0, [1], full_int_array_5, full_int_array_6, [1], [1]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_5 = full_int_array_7

        # pd_op.slice: (-1xf32) <- (-1x5xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            concat_0, [1], full_int_array_6, full_int_array_7, [1], [1]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [4]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_6 = full_int_array_8

        # pd_op.slice: (-1xf32) <- (-1x5xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            concat_0, [1], full_int_array_7, full_int_array_8, [1], [1]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [5]

        # pd_op.slice: (-1xf32) <- (-1x5xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            concat_0, [1], full_int_array_8, full_int_array_9, [1], [1]
        )

        # pd_op.slice: (-1xf32) <- (-1x5xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            concat_1, [1], full_int_array_4, full_int_array_5, [1], [1]
        )

        # pd_op.slice: (-1xf32) <- (-1x5xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            concat_1, [1], full_int_array_5, full_int_array_6, [1], [1]
        )

        # pd_op.slice: (-1xf32) <- (-1x5xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            concat_1, [1], full_int_array_6, full_int_array_7, [1], [1]
        )

        # pd_op.slice: (-1xf32) <- (-1x5xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            concat_1, [1], full_int_array_7, full_int_array_8, [1], [1]
        )

        # pd_op.slice: (-1xf32) <- (-1x5xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            concat_1, [1], full_int_array_8, full_int_array_9, [1], [1]
        )
        del concat_1

        # pd_op.cos: (-1xf32) <- (-1xf32)
        cos_0 = paddle._C_ops.cos(slice_4)

        # pd_op.sin: (-1xf32) <- (-1xf32)
        sin_0 = paddle._C_ops.sin(slice_4)

        # pd_op.pow: (-1xf32) <- (-1xf32)
        pow_2 = paddle._C_ops.pow(cos_0, float("2"))

        # pd_op.assign: (-1xf32) <- (-1xf32)
        assign_7 = pow_2

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_0 = paddle._C_ops.multiply(slice_2, pow_2)

        # pd_op.pow: (-1xf32) <- (-1xf32)
        pow_3 = paddle._C_ops.pow(sin_0, float("2"))

        # pd_op.assign: (-1xf32) <- (-1xf32)
        assign_8 = pow_3

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_1 = paddle._C_ops.multiply(slice_3, pow_3)

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_0 = paddle._C_ops.add(multiply_0, multiply_1)

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_2 = paddle._C_ops.multiply(slice_2, pow_3)

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_3 = paddle._C_ops.multiply(slice_3, pow_2)

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_1 = paddle._C_ops.add(multiply_2, multiply_3)

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_0 = paddle._C_ops.subtract(slice_2, slice_3)

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_4 = paddle._C_ops.multiply(subtract_0, cos_0)

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_5 = paddle._C_ops.multiply(multiply_4, sin_0)

        # pd_op.cos: (-1xf32) <- (-1xf32)
        cos_1 = paddle._C_ops.cos(slice_9)

        # pd_op.sin: (-1xf32) <- (-1xf32)
        sin_1 = paddle._C_ops.sin(slice_9)
        del slice_9

        # pd_op.pow: (-1xf32) <- (-1xf32)
        pow_4 = paddle._C_ops.pow(cos_1, float("2"))

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_6 = paddle._C_ops.multiply(slice_7, pow_4)

        # pd_op.pow: (-1xf32) <- (-1xf32)
        pow_5 = paddle._C_ops.pow(sin_1, float("2"))

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_7 = paddle._C_ops.multiply(slice_8, pow_5)

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_2 = paddle._C_ops.add(multiply_6, multiply_7)
        del multiply_6, multiply_7

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_8 = paddle._C_ops.multiply(slice_7, pow_5)
        del pow_5

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_9 = paddle._C_ops.multiply(slice_8, pow_4)
        del pow_4

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_3 = paddle._C_ops.add(multiply_8, multiply_9)
        del multiply_8, multiply_9

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_1 = paddle._C_ops.subtract(slice_7, slice_8)
        del slice_7, slice_8

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_10 = paddle._C_ops.multiply(subtract_1, cos_1)
        del cos_1, subtract_1

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_11 = paddle._C_ops.multiply(multiply_10, sin_1)
        del multiply_10, sin_1

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_4 = paddle._C_ops.add(add_0, add_2)

        # pd_op.assign: (-1xf32) <- (-1xf32)
        assign_9 = add_4

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_2 = paddle._C_ops.subtract(slice_1, slice_6)

        # pd_op.assign: (-1xf32) <- (-1xf32)
        assign_10 = subtract_2

        # pd_op.pow: (-1xf32) <- (-1xf32)
        pow_6 = paddle._C_ops.pow(subtract_2, float("2"))

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_12 = paddle._C_ops.multiply(add_4, pow_6)

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_5 = paddle._C_ops.add(add_1, add_3)

        # pd_op.assign: (-1xf32) <- (-1xf32)
        assign_11 = add_5

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_3 = paddle._C_ops.subtract(slice_0, slice_5)

        # pd_op.pow: (-1xf32) <- (-1xf32)
        pow_7 = paddle._C_ops.pow(subtract_3, float("2"))

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_13 = paddle._C_ops.multiply(add_5, pow_7)

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_6 = paddle._C_ops.add(multiply_12, multiply_13)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0.25"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(add_6, full_3, float("0"), True)
        del add_6

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_7 = paddle._C_ops.add(multiply_5, multiply_11)

        # pd_op.assign: (-1xf32) <- (-1xf32)
        assign_12 = add_7

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_4 = paddle._C_ops.subtract(slice_5, slice_0)

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_14 = paddle._C_ops.multiply(add_7, subtract_4)

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_15 = paddle._C_ops.multiply(multiply_14, subtract_2)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_13 = full_4

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(multiply_15, full_4, float("0"), True)
        del multiply_15

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_8 = paddle._C_ops.add(scale_2, scale_3)

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_16 = paddle._C_ops.multiply(add_4, add_5)

        # pd_op.pow: (-1xf32) <- (-1xf32)
        pow_8 = paddle._C_ops.pow(add_7, float("2"))

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_5 = paddle._C_ops.subtract(multiply_16, pow_8)

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_17 = paddle._C_ops.multiply(add_0, add_1)

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_18 = paddle._C_ops.multiply(multiply_5, multiply_5)

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_6 = paddle._C_ops.subtract(multiply_17, multiply_18)

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_19 = paddle._C_ops.multiply(add_2, add_3)

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_20 = paddle._C_ops.multiply(multiply_11, multiply_11)

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_7 = paddle._C_ops.subtract(multiply_19, multiply_20)
        del multiply_19, multiply_20

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_21 = paddle._C_ops.multiply(subtract_6, subtract_7)

        # pd_op.relu: (-1xf32) <- (-1xf32)
        relu_0 = paddle._C_ops.relu(multiply_21)
        del multiply_21

        # pd_op.sqrt: (-1xf32) <- (-1xf32)
        sqrt_0 = paddle._C_ops.sqrt(relu_0)

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("4"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(sqrt_0, full_5, float("0"), True)

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_14 = full_6

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(scale_4, full_6, float("0.001"), True)
        del scale_4

        # pd_op.divide: (-1xf32) <- (-1xf32, -1xf32)
        divide_1 = paddle._C_ops.divide(subtract_5, scale_5)

        # pd_op.log: (-1xf32) <- (-1xf32)
        log_0 = paddle._C_ops.log(divide_1)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(log_0, full_4, float("0"), True)
        del log_0

        # pd_op.divide: (-1xf32) <- (-1xf32, -1xf32)
        divide_2 = paddle._C_ops.divide(add_8, subtract_5)

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_9 = paddle._C_ops.add(divide_2, scale_6)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("0.001"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("100"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (-1xf32) <- (-1xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(add_9, full_7, full_8)

        # pd_op.full: (1xf32) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_15 = full_9

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(clip_0, full_9, float("0"), True)
        del clip_0

        # pd_op.exp: (-1xf32) <- (-1xf32)
        exp_0 = paddle._C_ops.exp(scale_7)
        del scale_7

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(exp_0, full_9, float("1"), True)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(scale_8, full_6, float("0.001"), True)
        del scale_8

        # pd_op.sqrt: (-1xf32) <- (-1xf32)
        sqrt_1 = paddle._C_ops.sqrt(scale_9)
        del scale_9

        # pd_op.pow: (-1xf32) <- (-1xf32)
        pow_9 = paddle._C_ops.pow(sqrt_1, float("2"))

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(pow_9, full_9, float("1"), True)
        del pow_9

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(scale_10, full_6, float("0.001"), True)
        del scale_10

        # pd_op.log: (-1xf32) <- (-1xf32)
        log_1 = paddle._C_ops.log(scale_11)
        del scale_11

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(log_1, full_9, float("0"), True)
        del log_1

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_22 = paddle._C_ops.multiply(sqrt_1, reshape_2)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_10 = []

        # pd_op.sum: (xf32) <- (-1xf32, 0xi64)
        sum_1 = paddle._C_ops.sum(multiply_22, full_int_array_10, None, False)

        # pd_op.divide: (xf32) <- (xf32, xf32)
        divide_0 = paddle._C_ops.divide(sum_1, data_4)
        del data_4

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_11 = [1, 1, 91]

        # pd_op.tile: (1x21504x91xb) <- (1x21504x1xb, 3xi64)
        tile_1 = paddle._C_ops.tile(unsqueeze_0, full_int_array_11)
        del full_int_array_11, unsqueeze_0

        # pd_op.masked_select: (-1xf32) <- (1x21504x91xf32, 1x21504x91xb)
        masked_select_3 = paddle._C_ops.masked_select(data_5, tile_1)
        del data_5

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_12 = [-1, 91]

        # pd_op.reshape: (-1x91xf32) <- (-1xf32, 2xi64)
        reshape_3 = paddle._C_ops.reshape(masked_select_3, full_int_array_12)
        del full_int_array_12

        # pd_op.slice: (-1xf32) <- (-1x5xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            reshape_1, [1], full_int_array_8, full_int_array_9, [1], [1]
        )
        del reshape_1

        # pd_op.divide: (-1xf32) <- (-1xf32, 1xf32)
        divide_3 = paddle._C_ops.divide(slice_10, data_6)
        del data_6, slice_10

        # pd_op.full: (1xf32) <- ()
        full_10 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_11 = paddle._C_ops.full(
            [1], float("89.99"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (-1xf32) <- (-1xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(divide_3, full_10, full_11)
        del divide_3, full_10, full_11

        # pd_op.cast: (-1xi64) <- (-1xf32)
        cast_0 = paddle._C_ops.cast(clip_1, paddle.int64)

        # pd_op.scale: (-1xi64) <- (-1xi64, 1xf32)
        scale_13 = paddle._C_ops.scale(cast_0, full_6, float("1"), True)

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_1 = paddle._C_ops.cast(scale_13, paddle.float32)

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_8 = paddle._C_ops.subtract(cast_1, clip_1)
        del cast_1, clip_1

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(subtract_8, full_9, float("1"), True)

        # pd_op.unsqueeze: (-1x1xi64) <- (-1xi64, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(cast_0, full_int_array_0)
        del cast_0

        # pd_op.cross_entropy_with_softmax: (-1x91xf32, -1x1xf32) <- (-1x91xf32, -1x1xi64)
        cross_entropy_with_softmax_0, cross_entropy_with_softmax_2 = (
            lambda x, f: f(x)
        )(
            paddle._C_ops.cross_entropy_with_softmax(
                reshape_3, unsqueeze_1, False, True, True, -100, -1
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.squeeze: (-1xf32) <- (-1x1xf32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(
            cross_entropy_with_softmax_2, full_int_array_0
        )

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_23 = paddle._C_ops.multiply(squeeze_0, subtract_8)

        # pd_op.unsqueeze: (-1x1xi64) <- (-1xi64, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(scale_13, full_int_array_0)
        del scale_13

        # pd_op.cross_entropy_with_softmax: (-1x91xf32, -1x1xf32) <- (-1x91xf32, -1x1xi64)
        cross_entropy_with_softmax_1, cross_entropy_with_softmax_3 = (
            lambda x, f: f(x)
        )(
            paddle._C_ops.cross_entropy_with_softmax(
                reshape_3, unsqueeze_2, False, True, True, -100, -1
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del reshape_3

        # pd_op.squeeze: (-1xf32) <- (-1x1xf32, 1xi64)
        squeeze_1 = paddle._C_ops.squeeze(
            cross_entropy_with_softmax_3, full_int_array_0
        )

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_24 = paddle._C_ops.multiply(squeeze_1, scale_14)

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_10 = paddle._C_ops.add(multiply_23, multiply_24)

        # pd_op.mean: (1xf32) <- (-1xf32, 1xi64)
        mean_0 = paddle._C_ops.mean(add_10, full_int_array_0, True)
        del (
            add_0,
            add_1,
            add_10,
            add_2,
            add_3,
            add_4,
            add_5,
            add_7,
            add_8,
            add_9,
            assign_0,
            assign_1,
            assign_10,
            assign_11,
            assign_12,
            assign_13,
            assign_14,
            assign_15,
            assign_2,
            assign_3,
            assign_4,
            assign_5,
            assign_6,
            assign_7,
            assign_8,
            assign_9,
            concat_0,
            cos_0,
            cross_entropy_with_softmax_2,
            cross_entropy_with_softmax_3,
            divide_1,
            divide_2,
            exp_0,
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
            full_int_array_10,
            full_int_array_4,
            full_int_array_5,
            full_int_array_6,
            full_int_array_7,
            full_int_array_8,
            full_int_array_9,
            masked_select_0,
            masked_select_3,
            multiply_0,
            multiply_1,
            multiply_11,
            multiply_12,
            multiply_13,
            multiply_14,
            multiply_16,
            multiply_17,
            multiply_18,
            multiply_2,
            multiply_22,
            multiply_23,
            multiply_24,
            multiply_3,
            multiply_4,
            multiply_5,
            pow_2,
            pow_3,
            pow_6,
            pow_7,
            pow_8,
            relu_0,
            reshape_2,
            scale_0,
            scale_14,
            scale_2,
            scale_3,
            scale_5,
            scale_6,
            sin_0,
            slice_0,
            slice_1,
            slice_2,
            slice_3,
            slice_4,
            slice_5,
            slice_6,
            split_1,
            split_2,
            split_3,
            sqrt_0,
            sqrt_1,
            squeeze_0,
            squeeze_1,
            subtract_0,
            subtract_2,
            subtract_3,
            subtract_4,
            subtract_5,
            subtract_6,
            subtract_7,
            subtract_8,
            sum_1,
            tile_0,
            tile_1,
            unsqueeze_1,
            unsqueeze_2,
        )

        return (
            cross_entropy_with_softmax_0,
            cross_entropy_with_softmax_1,
            divide_0,
            mean_0,
        )
