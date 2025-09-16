import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6):
        # pd_op.cast: (8x-1xi32) <- (8x-1xb)
        cast_0 = paddle._C_ops.cast(data_0, paddle.int32)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [-1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_0

        # pd_op.unsqueeze: (8x-1x1xi32) <- (8x-1xi32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(cast_0, full_int_array_0)
        del cast_0

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_1 = [1, 1, 4]

        # pd_op.tile: (8x-1x4xi32) <- (8x-1x1xi32, 3xi64)
        tile_0 = paddle._C_ops.tile(unsqueeze_0, full_int_array_1)
        del full_int_array_1, unsqueeze_0

        # pd_op.cast: (8x-1x4xb) <- (8x-1x4xi32)
        cast_1 = paddle._C_ops.cast(tile_0, paddle.bool)
        del tile_0

        # pd_op.masked_select: (-1xf32) <- (8x-1x4xf32, 8x-1x4xb)
        masked_select_0 = paddle._C_ops.masked_select(data_1, cast_1)
        del data_1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [-1, 4]

        # pd_op.reshape: (-1x4xf32) <- (-1xf32, 2xi64)
        reshape_0 = paddle._C_ops.reshape(masked_select_0, full_int_array_2)

        # pd_op.masked_select: (-1xf32) <- (8x-1x4xf32, 8x-1x4xb)
        masked_select_1 = paddle._C_ops.masked_select(data_2, cast_1)

        # pd_op.reshape: (-1x4xf32) <- (-1xf32, 2xi64)
        reshape_1 = paddle._C_ops.reshape(masked_select_1, full_int_array_2)
        del masked_select_1

        # pd_op.sum: (8x-1xf32) <- (8x-1x4xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(data_3, full_int_array_0, None, False)
        del data_3

        # pd_op.masked_select: (-1xf32) <- (8x-1xf32, 8x-1xb)
        masked_select_2 = paddle._C_ops.masked_select(sum_0, data_0)
        del sum_0

        # pd_op.unsqueeze: (-1x1xf32) <- (-1xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(masked_select_2, full_int_array_0)
        del masked_select_2

        # pd_op.subtract: (-1x4xf32) <- (-1x4xf32, -1x4xf32)
        subtract_0 = paddle._C_ops.subtract(reshape_0, reshape_1)

        # pd_op.abs: (-1x4xf32) <- (-1x4xf32)
        abs_0 = paddle._C_ops.abs(subtract_0)

        # pd_op.mean_all: (xf32) <- (-1x4xf32)
        mean_all_0 = paddle._C_ops.mean_all(abs_0)

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split_with_num: ([-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32]) <- (-1x4xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(reshape_0, 4, full_0)

        # builtin.split: (-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32) <- ([-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32])
        (
            split_0,
            split_1,
            split_2,
            split_3,
        ) = split_with_num_0
        del split_with_num_0

        # pd_op.split_with_num: ([-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32]) <- (-1x4xf32, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(reshape_1, 4, full_0)

        # builtin.split: (-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32) <- ([-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32])
        (
            split_4,
            split_5,
            split_6,
            split_7,
        ) = split_with_num_1
        del split_with_num_1

        # pd_op.maximum: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        maximum_0 = paddle._C_ops.maximum(split_0, split_4)

        # pd_op.maximum: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        maximum_1 = paddle._C_ops.maximum(split_1, split_5)

        # pd_op.minimum: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        minimum_0 = paddle._C_ops.minimum(split_2, split_6)

        # pd_op.minimum: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        minimum_1 = paddle._C_ops.minimum(split_3, split_7)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_1 = paddle._C_ops.subtract(minimum_0, maximum_0)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_3 = full_1

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_4 = full_2

        # pd_op.clip: (-1x1xf32) <- (-1x1xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(subtract_1, full_1, full_2)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_2 = paddle._C_ops.subtract(minimum_1, maximum_1)

        # pd_op.clip: (-1x1xf32) <- (-1x1xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(subtract_2, full_1, full_2)

        # pd_op.multiply: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        multiply_0 = paddle._C_ops.multiply(clip_0, clip_1)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_3 = paddle._C_ops.subtract(split_2, split_0)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_4 = paddle._C_ops.subtract(split_3, split_1)

        # pd_op.multiply: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        multiply_1 = paddle._C_ops.multiply(subtract_3, subtract_4)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_5 = paddle._C_ops.subtract(split_6, split_4)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_6 = paddle._C_ops.subtract(split_7, split_5)

        # pd_op.multiply: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        multiply_2 = paddle._C_ops.multiply(subtract_5, subtract_6)
        del subtract_5, subtract_6

        # pd_op.add: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        add_0 = paddle._C_ops.add(multiply_1, multiply_2)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_7 = paddle._C_ops.subtract(add_0, multiply_0)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_5 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_6 = full_3

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(subtract_7, full_3, float("1e-10"), True)
        del subtract_7

        # pd_op.divide: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        divide_2 = paddle._C_ops.divide(multiply_0, scale_0)

        # pd_op.minimum: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        minimum_2 = paddle._C_ops.minimum(split_0, split_4)

        # pd_op.minimum: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        minimum_3 = paddle._C_ops.minimum(split_1, split_5)

        # pd_op.maximum: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        maximum_2 = paddle._C_ops.maximum(split_2, split_6)

        # pd_op.maximum: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        maximum_3 = paddle._C_ops.maximum(split_3, split_7)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_8 = paddle._C_ops.subtract(maximum_2, minimum_2)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_9 = paddle._C_ops.subtract(maximum_3, minimum_3)

        # pd_op.multiply: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        multiply_3 = paddle._C_ops.multiply(subtract_8, subtract_9)

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(multiply_3, full_3, float("1e-10"), True)
        del multiply_3

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_10 = paddle._C_ops.subtract(scale_1, scale_0)

        # pd_op.divide: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        divide_3 = paddle._C_ops.divide(subtract_10, scale_1)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_11 = paddle._C_ops.subtract(divide_2, divide_3)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(subtract_11, full_4, float("1"), True)
        del subtract_11

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(scale_2, full_3, float("0"), True)
        del scale_2

        # pd_op.multiply: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        multiply_4 = paddle._C_ops.multiply(scale_3, unsqueeze_1)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_3 = []

        # pd_op.assign: (0xi64) <- (0xi64)
        assign_7 = full_int_array_3

        # pd_op.sum: (xf32) <- (-1x1xf32, 0xi64)
        sum_1 = paddle._C_ops.sum(multiply_4, full_int_array_3, None, False)

        # pd_op.divide: (xf32) <- (xf32, xf32)
        divide_0 = paddle._C_ops.divide(sum_1, data_4)

        # pd_op.unsqueeze: (8x-1x1xb) <- (8x-1xb, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(data_0, full_int_array_0)
        del data_0

        # pd_op.cast: (8x-1x1xi32) <- (8x-1x1xb)
        cast_2 = paddle._C_ops.cast(unsqueeze_2, paddle.int32)
        del unsqueeze_2

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_4 = [1, 1, 68]

        # pd_op.tile: (8x-1x68xi32) <- (8x-1x1xi32, 3xi64)
        tile_1 = paddle._C_ops.tile(cast_2, full_int_array_4)
        del cast_2, full_int_array_4

        # pd_op.cast: (8x-1x68xb) <- (8x-1x68xi32)
        cast_3 = paddle._C_ops.cast(tile_1, paddle.bool)
        del tile_1

        # pd_op.masked_select: (-1xf32) <- (8x-1x68xf32, 8x-1x68xb)
        masked_select_3 = paddle._C_ops.masked_select(data_5, cast_3)
        del data_5

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_5 = [-1, 4, 17]

        # pd_op.reshape: (-1x4x17xf32) <- (-1xf32, 3xi64)
        reshape_2 = paddle._C_ops.reshape(masked_select_3, full_int_array_5)
        del full_int_array_5

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("2"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split_with_num: ([8x-1x2xf32, 8x-1x2xf32]) <- (8x-1x4xf32, 1xi32)
        split_with_num_2 = paddle._C_ops.split_with_num(data_2, 2, full_5)
        del data_2, full_5

        # builtin.split: (8x-1x2xf32, 8x-1x2xf32) <- ([8x-1x2xf32, 8x-1x2xf32])
        (
            split_8,
            split_9,
        ) = split_with_num_2
        del split_with_num_2

        # pd_op.subtract: (8x-1x2xf32) <- (-1x2xf32, 8x-1x2xf32)
        subtract_12 = paddle._C_ops.subtract(data_6, split_8)
        del split_8

        # pd_op.subtract: (8x-1x2xf32) <- (8x-1x2xf32, -1x2xf32)
        subtract_13 = paddle._C_ops.subtract(split_9, data_6)
        del data_6, split_9

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([8x-1x2xf32, 8x-1x2xf32]) <- (8x-1x2xf32, 8x-1x2xf32)
        combine_0 = [subtract_12, subtract_13]
        del subtract_12, subtract_13

        # pd_op.concat: (8x-1x4xf32) <- ([8x-1x2xf32, 8x-1x2xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_6)
        del combine_0, full_6

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("15.99"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (8x-1x4xf32) <- (8x-1x4xf32, 1xf32, 1xf32)
        clip_2 = paddle._C_ops.clip(concat_0, full_1, full_7)
        del concat_0, full_7

        # pd_op.masked_select: (-1xf32) <- (8x-1x4xf32, 8x-1x4xb)
        masked_select_4 = paddle._C_ops.masked_select(clip_2, cast_1)
        del clip_2

        # pd_op.reshape: (-1x4xf32) <- (-1xf32, 2xi64)
        reshape_3 = paddle._C_ops.reshape(masked_select_4, full_int_array_2)
        del full_int_array_2, masked_select_4

        # pd_op.floor: (-1x4xf32) <- (-1x4xf32)
        floor_0 = paddle._C_ops.floor(reshape_3)

        # pd_op.cast: (-1x4xi64) <- (-1x4xf32)
        cast_4 = paddle._C_ops.cast(floor_0, paddle.int64)
        del floor_0

        # pd_op.scale: (-1x4xi64) <- (-1x4xi64, 1xf32)
        scale_4 = paddle._C_ops.scale(cast_4, full_3, float("1"), True)

        # pd_op.cast: (-1x4xf32) <- (-1x4xi64)
        cast_5 = paddle._C_ops.cast(scale_4, paddle.float32)

        # pd_op.subtract: (-1x4xf32) <- (-1x4xf32, -1x4xf32)
        subtract_14 = paddle._C_ops.subtract(cast_5, reshape_3)
        del cast_5, reshape_3

        # pd_op.scale: (-1x4xf32) <- (-1x4xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(subtract_14, full_4, float("1"), True)

        # pd_op.scale: (-1x4xi64) <- (-1x4xi64, 1xf32)
        scale_6 = paddle._C_ops.scale(cast_4, full_3, float("0"), True)
        del cast_4

        # pd_op.unsqueeze: (-1x4x1xi64) <- (-1x4xi64, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(scale_6, full_int_array_0)
        del scale_6

        # pd_op.cross_entropy_with_softmax: (-1x4x17xf32, -1x4x1xf32) <- (-1x4x17xf32, -1x4x1xi64)
        cross_entropy_with_softmax_0, cross_entropy_with_softmax_2 = (
            lambda x, f: f(x)
        )(
            paddle._C_ops.cross_entropy_with_softmax(
                reshape_2, unsqueeze_3, False, True, True, -100, -1
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.squeeze: (-1x4xf32) <- (-1x4x1xf32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(
            cross_entropy_with_softmax_2, full_int_array_0
        )

        # pd_op.multiply: (-1x4xf32) <- (-1x4xf32, -1x4xf32)
        multiply_5 = paddle._C_ops.multiply(squeeze_0, subtract_14)

        # pd_op.scale: (-1x4xi64) <- (-1x4xi64, 1xf32)
        scale_7 = paddle._C_ops.scale(scale_4, full_3, float("0"), True)
        del scale_4

        # pd_op.unsqueeze: (-1x4x1xi64) <- (-1x4xi64, 1xi64)
        unsqueeze_4 = paddle._C_ops.unsqueeze(scale_7, full_int_array_0)
        del scale_7

        # pd_op.cross_entropy_with_softmax: (-1x4x17xf32, -1x4x1xf32) <- (-1x4x17xf32, -1x4x1xi64)
        cross_entropy_with_softmax_1, cross_entropy_with_softmax_3 = (
            lambda x, f: f(x)
        )(
            paddle._C_ops.cross_entropy_with_softmax(
                reshape_2, unsqueeze_4, False, True, True, -100, -1
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del reshape_2

        # pd_op.squeeze: (-1x4xf32) <- (-1x4x1xf32, 1xi64)
        squeeze_1 = paddle._C_ops.squeeze(
            cross_entropy_with_softmax_3, full_int_array_0
        )

        # pd_op.multiply: (-1x4xf32) <- (-1x4xf32, -1x4xf32)
        multiply_6 = paddle._C_ops.multiply(squeeze_1, scale_5)

        # pd_op.add: (-1x4xf32) <- (-1x4xf32, -1x4xf32)
        add_1 = paddle._C_ops.add(multiply_5, multiply_6)

        # pd_op.mean: (-1x1xf32) <- (-1x4xf32, 1xi64)
        mean_0 = paddle._C_ops.mean(add_1, full_int_array_0, True)
        del full_int_array_0

        # pd_op.multiply: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        multiply_7 = paddle._C_ops.multiply(mean_0, unsqueeze_1)

        # pd_op.sum: (xf32) <- (-1x1xf32, 0xi64)
        sum_2 = paddle._C_ops.sum(multiply_7, full_int_array_3, None, False)

        # pd_op.divide: (xf32) <- (xf32, xf32)
        divide_1 = paddle._C_ops.divide(sum_2, data_4)
        del (
            abs_0,
            add_0,
            add_1,
            assign_0,
            assign_1,
            assign_2,
            assign_3,
            assign_4,
            assign_5,
            assign_6,
            assign_7,
            cast_1,
            cast_3,
            clip_0,
            clip_1,
            cross_entropy_with_softmax_2,
            cross_entropy_with_softmax_3,
            data_4,
            divide_2,
            divide_3,
            full_0,
            full_1,
            full_2,
            full_3,
            full_4,
            full_int_array_3,
            masked_select_0,
            masked_select_3,
            maximum_0,
            maximum_1,
            maximum_2,
            maximum_3,
            mean_0,
            minimum_0,
            minimum_1,
            minimum_2,
            minimum_3,
            multiply_0,
            multiply_1,
            multiply_2,
            multiply_4,
            multiply_5,
            multiply_6,
            multiply_7,
            reshape_0,
            reshape_1,
            scale_0,
            scale_1,
            scale_3,
            scale_5,
            split_0,
            split_1,
            split_2,
            split_3,
            split_4,
            split_5,
            split_6,
            split_7,
            squeeze_0,
            squeeze_1,
            subtract_0,
            subtract_1,
            subtract_10,
            subtract_14,
            subtract_2,
            subtract_3,
            subtract_4,
            subtract_8,
            subtract_9,
            sum_1,
            sum_2,
            unsqueeze_1,
            unsqueeze_3,
            unsqueeze_4,
        )

        return (
            cross_entropy_with_softmax_0,
            cross_entropy_with_softmax_1,
            mean_all_0,
            divide_0,
            divide_1,
        )
