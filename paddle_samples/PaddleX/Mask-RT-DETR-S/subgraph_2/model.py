import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8
    ):
        # pd_op.share_data_: (1x300x4xf32) <- (1x300x4xf32)
        share_data__0 = data_6.detach()

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_3 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_4 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_5 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_6 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_7 = full_int_array_1

        # pd_op.slice: (300x4xf32) <- (1x300x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            share_data__0, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del share_data__0

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_8 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_9 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_10 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_11 = full_0

        # pd_op.gather: (1x4xf32) <- (300x4xf32, 1xi64, 1xi32)
        gather_0 = paddle._C_ops.gather(slice_0, data_0, full_0)
        del slice_0

        # builtin.combine: ([1x4xf32]) <- (1x4xf32)
        combine_0 = [gather_0]
        del gather_0

        # pd_op.concat: (1x4xf32) <- ([1x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)
        del combine_0

        # pd_op.gather: (1x4xf32) <- (1x4xf32, 1xi64, 1xi32)
        gather_1 = paddle._C_ops.gather(data_2, data_1, full_0)
        del data_2

        # builtin.combine: ([1x4xf32]) <- (1x4xf32)
        combine_1 = [gather_1]
        del gather_1

        # pd_op.concat: (1x4xf32) <- ([1x4xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_0)
        del combine_1

        # pd_op.assign: (1x4xf32) <- (1x4xf32)
        assign_12 = concat_1

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_13 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_14 = full_1

        # pd_op.split_with_num: ([1x2xf32, 1x2xf32]) <- (1x4xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(concat_0, 2, full_1)
        del concat_0

        # builtin.split: (1x2xf32, 1x2xf32) <- ([1x2xf32, 1x2xf32])
        (
            split_0,
            split_1,
        ) = split_with_num_0
        del split_with_num_0

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_15 = full_2

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_16 = full_2

        # pd_op.scale: (1x2xf32) <- (1x2xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(split_1, full_2, float("0"), True)

        # pd_op.subtract: (1x2xf32) <- (1x2xf32, 1x2xf32)
        subtract_0 = paddle._C_ops.subtract(split_0, scale_3)
        del scale_3

        # pd_op.scale: (1x2xf32) <- (1x2xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(split_1, full_2, float("0"), True)
        del split_1

        # pd_op.add: (1x2xf32) <- (1x2xf32, 1x2xf32)
        add_0 = paddle._C_ops.add(split_0, scale_4)
        del scale_4, split_0

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_17 = full_3

        # builtin.combine: ([1x2xf32, 1x2xf32]) <- (1x2xf32, 1x2xf32)
        combine_2 = [subtract_0, add_0]
        del add_0, subtract_0

        # pd_op.concat: (1x4xf32) <- ([1x2xf32, 1x2xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_3)
        del combine_2

        # pd_op.split_with_num: ([1x1xf32, 1x1xf32, 1x1xf32, 1x1xf32]) <- (1x4xf32, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(concat_2, 4, full_1)
        del concat_2

        # builtin.split: (1x1xf32, 1x1xf32, 1x1xf32, 1x1xf32) <- ([1x1xf32, 1x1xf32, 1x1xf32, 1x1xf32])
        (
            split_2,
            split_3,
            split_4,
            split_5,
        ) = split_with_num_1
        del split_with_num_1

        # pd_op.split_with_num: ([1x2xf32, 1x2xf32]) <- (1x4xf32, 1xi32)
        split_with_num_2 = paddle._C_ops.split_with_num(concat_1, 2, full_1)

        # builtin.split: (1x2xf32, 1x2xf32) <- ([1x2xf32, 1x2xf32])
        (
            split_6,
            split_7,
        ) = split_with_num_2
        del split_with_num_2

        # pd_op.scale: (1x2xf32) <- (1x2xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(split_7, full_2, float("0"), True)
        del split_7

        # pd_op.subtract: (1x2xf32) <- (1x2xf32, 1x2xf32)
        subtract_1 = paddle._C_ops.subtract(split_6, scale_5)

        # pd_op.add: (1x2xf32) <- (1x2xf32, 1x2xf32)
        add_1 = paddle._C_ops.add(split_6, scale_5)
        del scale_5, split_6

        # builtin.combine: ([1x2xf32, 1x2xf32]) <- (1x2xf32, 1x2xf32)
        combine_3 = [subtract_1, add_1]
        del add_1, subtract_1

        # pd_op.concat: (1x4xf32) <- ([1x2xf32, 1x2xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_3, full_3)
        del combine_3

        # pd_op.split_with_num: ([1x1xf32, 1x1xf32, 1x1xf32, 1x1xf32]) <- (1x4xf32, 1xi32)
        split_with_num_3 = paddle._C_ops.split_with_num(concat_3, 4, full_1)
        del concat_3

        # builtin.split: (1x1xf32, 1x1xf32, 1x1xf32, 1x1xf32) <- ([1x1xf32, 1x1xf32, 1x1xf32, 1x1xf32])
        (
            split_8,
            split_9,
            split_10,
            split_11,
        ) = split_with_num_3
        del split_with_num_3

        # pd_op.assign: (1x1xf32) <- (1x1xf32)
        assign_18 = split_11

        # pd_op.assign: (1x1xf32) <- (1x1xf32)
        assign_19 = split_10

        # pd_op.assign: (1x1xf32) <- (1x1xf32)
        assign_20 = split_9

        # pd_op.assign: (1x1xf32) <- (1x1xf32)
        assign_21 = split_8

        # pd_op.maximum: (1x1xf32) <- (1x1xf32, 1x1xf32)
        maximum_0 = paddle._C_ops.maximum(split_2, split_8)

        # pd_op.maximum: (1x1xf32) <- (1x1xf32, 1x1xf32)
        maximum_1 = paddle._C_ops.maximum(split_3, split_9)

        # pd_op.minimum: (1x1xf32) <- (1x1xf32, 1x1xf32)
        minimum_0 = paddle._C_ops.minimum(split_4, split_10)

        # pd_op.minimum: (1x1xf32) <- (1x1xf32, 1x1xf32)
        minimum_1 = paddle._C_ops.minimum(split_5, split_11)

        # pd_op.subtract: (1x1xf32) <- (1x1xf32, 1x1xf32)
        subtract_2 = paddle._C_ops.subtract(minimum_0, maximum_0)
        del maximum_0, minimum_0

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_22 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_23 = full_4

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_24 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_25 = full_5

        # pd_op.clip: (1x1xf32) <- (1x1xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(subtract_2, full_4, full_5)
        del subtract_2

        # pd_op.subtract: (1x1xf32) <- (1x1xf32, 1x1xf32)
        subtract_3 = paddle._C_ops.subtract(minimum_1, maximum_1)
        del maximum_1, minimum_1

        # pd_op.clip: (1x1xf32) <- (1x1xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(subtract_3, full_4, full_5)
        del subtract_3

        # pd_op.multiply: (1x1xf32) <- (1x1xf32, 1x1xf32)
        multiply_0 = paddle._C_ops.multiply(clip_0, clip_1)
        del clip_0, clip_1

        # pd_op.subtract: (1x1xf32) <- (1x1xf32, 1x1xf32)
        subtract_4 = paddle._C_ops.subtract(split_4, split_2)
        del split_2, split_4

        # pd_op.subtract: (1x1xf32) <- (1x1xf32, 1x1xf32)
        subtract_5 = paddle._C_ops.subtract(split_5, split_3)
        del split_3, split_5

        # pd_op.multiply: (1x1xf32) <- (1x1xf32, 1x1xf32)
        multiply_1 = paddle._C_ops.multiply(subtract_4, subtract_5)
        del subtract_4, subtract_5

        # pd_op.clip: (1x1xf32) <- (1x1xf32, 1xf32, 1xf32)
        clip_2 = paddle._C_ops.clip(multiply_1, full_4, full_5)
        del multiply_1

        # pd_op.subtract: (1x1xf32) <- (1x1xf32, 1x1xf32)
        subtract_6 = paddle._C_ops.subtract(split_10, split_8)

        # pd_op.subtract: (1x1xf32) <- (1x1xf32, 1x1xf32)
        subtract_7 = paddle._C_ops.subtract(split_11, split_9)

        # pd_op.multiply: (1x1xf32) <- (1x1xf32, 1x1xf32)
        multiply_2 = paddle._C_ops.multiply(subtract_6, subtract_7)
        del subtract_6, subtract_7

        # pd_op.assign: (1x1xf32) <- (1x1xf32)
        assign_26 = multiply_2

        # pd_op.clip: (1x1xf32) <- (1x1xf32, 1xf32, 1xf32)
        clip_3 = paddle._C_ops.clip(multiply_2, full_4, full_5)

        # pd_op.add: (1x1xf32) <- (1x1xf32, 1x1xf32)
        add_2 = paddle._C_ops.add(clip_2, clip_3)
        del clip_2, clip_3

        # pd_op.subtract: (1x1xf32) <- (1x1xf32, 1x1xf32)
        subtract_8 = paddle._C_ops.subtract(add_2, multiply_0)
        del add_2

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_27 = full_6

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_28 = full_6

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_29 = full_6

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_30 = full_6

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_31 = full_6

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_32 = full_6

        # pd_op.scale: (1x1xf32) <- (1x1xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(subtract_8, full_6, float("1e-09"), True)
        del subtract_8

        # pd_op.divide: (1x1xf32) <- (1x1xf32, 1x1xf32)
        divide_2 = paddle._C_ops.divide(multiply_0, scale_6)
        del multiply_0, scale_6

        # pd_op.full: (1x300xi64) <- ()
        full_7 = paddle._C_ops.full(
            [1, 300],
            float("2"),
            paddle.int64,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_like: (1xi64) <- (1xi64, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            data_0, full_4, paddle.int64, paddle.framework._current_expected_place()
        )

        # builtin.combine: ([1xi64]) <- (1xi64)
        combine_4 = [full_like_0]
        del full_like_0

        # pd_op.concat: (1xi64) <- ([1xi64], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_4, full_0)
        del combine_4

        # builtin.combine: ([1xi64]) <- (1xi64)
        combine_5 = [data_0]

        # pd_op.concat: (1xi64) <- ([1xi64], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_5, full_0)
        del combine_5

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("300"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1xi64) <- (1xi64, 1xf32)
        scale_7 = paddle._C_ops.scale(concat_4, full_8, float("0"), True)
        del concat_4, full_8

        # pd_op.add: (1xi64) <- (1xi64, 1xi64)
        add_3 = paddle._C_ops.add(concat_5, scale_7)
        del concat_5, scale_7

        # pd_op.gather: (1x1xi32) <- (1x1xi32, 1xi64, 1xi32)
        gather_2 = paddle._C_ops.gather(data_4, data_1, full_0)
        del data_4

        # builtin.combine: ([1x1xi32]) <- (1x1xi32)
        combine_6 = [gather_2]
        del gather_2

        # pd_op.concat: (1x1xi32) <- ([1x1xi32], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_6, full_0)
        del combine_6

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [-1, 1]

        # pd_op.reshape: (300x1xi64) <- (1x300xi64, 2xi64)
        reshape_0 = paddle._C_ops.reshape(full_7, full_int_array_2)
        del full_7

        # pd_op.cast: (1x1xi64) <- (1x1xi32)
        cast_0 = paddle._C_ops.cast(concat_6, paddle.int64)
        del concat_6

        # pd_op.scatter: (300x1xi64) <- (300x1xi64, 1xi64, 1x1xi64)
        scatter_0 = paddle._C_ops.scatter(reshape_0, add_3, cast_0, True)
        del cast_0, reshape_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [1, 300]

        # pd_op.reshape: (1x300xi64) <- (300x1xi64, 2xi64)
        reshape_1 = paddle._C_ops.reshape(scatter_0, full_int_array_3)
        del full_int_array_3, scatter_0

        # pd_op.full: (1xi32) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("3"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.one_hot: (1x300x3xf32) <- (1x300xi64, 1xi32)
        one_hot_0 = paddle._C_ops.one_hot(
            reshape_1 % paddle.cast(full_9, reshape_1.dtype), full_9
        )
        del full_9, reshape_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [-1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_33 = full_int_array_4

        # pd_op.slice: (1x300x2xf32) <- (1x300x3xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            one_hot_0, [2], full_int_array_0, full_int_array_4, [1], []
        )
        del one_hot_0

        # pd_op.full: (1x300xf32) <- ()
        full_10 = paddle._C_ops.full(
            [1, 300],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.reshape: (300x1xf32) <- (1x300xf32, 2xi64)
        reshape_2 = paddle._C_ops.reshape(full_10, full_int_array_2)
        del full_10, full_int_array_2

        # pd_op.scatter: (300x1xf32) <- (300x1xf32, 1xi64, 1x1xf32)
        scatter_1 = paddle._C_ops.scatter(reshape_2, add_3, divide_2, True)
        del add_3, divide_2, reshape_2

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_5 = [1, 300, 1]

        # pd_op.reshape: (1x300x1xf32) <- (300x1xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(scatter_1, full_int_array_5)
        del full_int_array_5, scatter_1

        # pd_op.multiply: (1x300x2xf32) <- (1x300x1xf32, 1x300x2xf32)
        multiply_3 = paddle._C_ops.multiply(reshape_3, slice_1)
        del reshape_3

        # pd_op.full: (1xf32) <- ()
        full_11 = paddle._C_ops.full(
            [1], float("0.00333333"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1xf32) <- (1xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(data_5, full_11, float("0"), True)
        del full_11

        # pd_op.sigmoid: (1x300x2xf32) <- (1x300x2xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(data_3)

        # pd_op.pow: (1x300x2xf32) <- (1x300x2xf32)
        pow_0 = paddle._C_ops.pow(multiply_3, float("1.5"))
        del multiply_3

        # pd_op.pow: (1x300x2xf32) <- (1x300x2xf32)
        pow_1 = paddle._C_ops.pow(sigmoid_0, float("1.5"))

        # pd_op.scale: (1x300x2xf32) <- (1x300x2xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(pow_1, full_6, float("0"), True)
        del pow_1

        # pd_op.full: (1xf32) <- ()
        full_12 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_34 = full_12

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_35 = full_12

        # pd_op.scale: (1x300x2xf32) <- (1x300x2xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(slice_1, full_12, float("1"), True)

        # pd_op.multiply: (1x300x2xf32) <- (1x300x2xf32, 1x300x2xf32)
        multiply_4 = paddle._C_ops.multiply(scale_9, scale_10)

        # pd_op.add: (1x300x2xf32) <- (1x300x2xf32, 1x300x2xf32)
        add_4 = paddle._C_ops.add(multiply_4, slice_1)

        # pd_op.full: (1xf32) <- ()
        full_13 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.sigmoid_cross_entropy_with_logits: (1x300x2xf32) <- (1x300x2xf32, 1x300x2xf32, None)
        sigmoid_cross_entropy_with_logits_0 = (
            paddle._C_ops.sigmoid_cross_entropy_with_logits(
                data_3, pow_0, None, False, -100
            )
        )
        del data_3

        # pd_op.multiply: (1x300x2xf32) <- (1x300x2xf32, 1x300x2xf32)
        multiply_5 = paddle._C_ops.multiply(sigmoid_cross_entropy_with_logits_0, add_4)

        # pd_op.mean: (1x2xf32) <- (1x300x2xf32, 1xi64)
        mean_0 = paddle._C_ops.mean(multiply_5, full_int_array_1, False)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_6 = []

        # pd_op.assign: (0xi64) <- (0xi64)
        assign_36 = full_int_array_6

        # pd_op.assign: (0xi64) <- (0xi64)
        assign_37 = full_int_array_6

        # pd_op.assign: (0xi64) <- (0xi64)
        assign_38 = full_int_array_6

        # pd_op.assign: (0xi64) <- (0xi64)
        assign_39 = full_int_array_6

        # pd_op.sum: (xf32) <- (1x2xf32, 0xi64)
        sum_0 = paddle._C_ops.sum(mean_0, full_int_array_6, None, False)

        # pd_op.divide: (1xf32) <- (xf32, 1xf32)
        divide_3 = paddle._C_ops.divide(sum_0, scale_8)

        # pd_op.full: (1xf32) <- ()
        full_14 = paddle._C_ops.full(
            [1], float("4"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1xf32) <- (1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(divide_3, full_14, float("0"), True)

        # pd_op.slice: (300x4xf32) <- (1x300x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            data_6, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del data_6

        # pd_op.gather: (1x4xf32) <- (300x4xf32, 1xi64, 1xi32)
        gather_3 = paddle._C_ops.gather(slice_2, data_0, full_0)

        # builtin.combine: ([1x4xf32]) <- (1x4xf32)
        combine_7 = [gather_3]

        # pd_op.concat: (1x4xf32) <- ([1x4xf32], 1xi32)
        concat_7 = paddle._C_ops.concat(combine_7, full_0)
        del combine_7

        # pd_op.subtract: (1x4xf32) <- (1x4xf32, 1x4xf32)
        subtract_9 = paddle._C_ops.subtract(concat_7, concat_1)
        del concat_1

        # pd_op.abs: (1x4xf32) <- (1x4xf32)
        abs_0 = paddle._C_ops.abs(subtract_9)

        # pd_op.sum: (xf32) <- (1x4xf32, 0xi64)
        sum_1 = paddle._C_ops.sum(abs_0, full_int_array_6, None, False)

        # pd_op.full: (1xf32) <- ()
        full_15 = paddle._C_ops.full(
            [1], float("5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_40 = full_15

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_41 = full_15

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(sum_1, full_15, float("0"), True)
        del sum_1

        # pd_op.divide: (1xf32) <- (xf32, 1xf32)
        divide_0 = paddle._C_ops.divide(scale_11, data_5)

        # pd_op.split_with_num: ([1x2xf32, 1x2xf32]) <- (1x4xf32, 1xi32)
        split_with_num_4 = paddle._C_ops.split_with_num(concat_7, 2, full_1)

        # builtin.split: (1x2xf32, 1x2xf32) <- ([1x2xf32, 1x2xf32])
        (
            split_12,
            split_13,
        ) = split_with_num_4
        del split_with_num_4

        # pd_op.scale: (1x2xf32) <- (1x2xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(split_13, full_2, float("0"), True)
        del full_2, split_13

        # pd_op.assign: (1x2xf32) <- (1x2xf32)
        assign_42 = scale_12

        # pd_op.subtract: (1x2xf32) <- (1x2xf32, 1x2xf32)
        subtract_10 = paddle._C_ops.subtract(split_12, scale_12)

        # pd_op.add: (1x2xf32) <- (1x2xf32, 1x2xf32)
        add_5 = paddle._C_ops.add(split_12, scale_12)

        # builtin.combine: ([1x2xf32, 1x2xf32]) <- (1x2xf32, 1x2xf32)
        combine_8 = [subtract_10, add_5]

        # pd_op.concat: (1x4xf32) <- ([1x2xf32, 1x2xf32], 1xi32)
        concat_8 = paddle._C_ops.concat(combine_8, full_3)
        del combine_8, full_3

        # pd_op.split_with_num: ([1x1xf32, 1x1xf32, 1x1xf32, 1x1xf32]) <- (1x4xf32, 1xi32)
        split_with_num_5 = paddle._C_ops.split_with_num(concat_8, 4, full_1)
        del concat_8

        # builtin.split: (1x1xf32, 1x1xf32, 1x1xf32, 1x1xf32) <- ([1x1xf32, 1x1xf32, 1x1xf32, 1x1xf32])
        (
            split_14,
            split_15,
            split_16,
            split_17,
        ) = split_with_num_5
        del split_with_num_5

        # pd_op.maximum: (1x1xf32) <- (1x1xf32, 1x1xf32)
        maximum_2 = paddle._C_ops.maximum(split_14, split_8)

        # pd_op.maximum: (1x1xf32) <- (1x1xf32, 1x1xf32)
        maximum_3 = paddle._C_ops.maximum(split_15, split_9)

        # pd_op.minimum: (1x1xf32) <- (1x1xf32, 1x1xf32)
        minimum_2 = paddle._C_ops.minimum(split_16, split_10)

        # pd_op.minimum: (1x1xf32) <- (1x1xf32, 1x1xf32)
        minimum_3 = paddle._C_ops.minimum(split_17, split_11)

        # pd_op.subtract: (1x1xf32) <- (1x1xf32, 1x1xf32)
        subtract_11 = paddle._C_ops.subtract(minimum_2, maximum_2)

        # pd_op.clip: (1x1xf32) <- (1x1xf32, 1xf32, 1xf32)
        clip_4 = paddle._C_ops.clip(subtract_11, full_4, full_5)

        # pd_op.subtract: (1x1xf32) <- (1x1xf32, 1x1xf32)
        subtract_12 = paddle._C_ops.subtract(minimum_3, maximum_3)

        # pd_op.clip: (1x1xf32) <- (1x1xf32, 1xf32, 1xf32)
        clip_5 = paddle._C_ops.clip(subtract_12, full_4, full_5)
        del full_5

        # pd_op.multiply: (1x1xf32) <- (1x1xf32, 1x1xf32)
        multiply_6 = paddle._C_ops.multiply(clip_4, clip_5)

        # pd_op.subtract: (1x1xf32) <- (1x1xf32, 1x1xf32)
        subtract_13 = paddle._C_ops.subtract(split_16, split_14)

        # pd_op.subtract: (1x1xf32) <- (1x1xf32, 1x1xf32)
        subtract_14 = paddle._C_ops.subtract(split_17, split_15)

        # pd_op.multiply: (1x1xf32) <- (1x1xf32, 1x1xf32)
        multiply_7 = paddle._C_ops.multiply(subtract_13, subtract_14)

        # pd_op.add: (1x1xf32) <- (1x1xf32, 1x1xf32)
        add_6 = paddle._C_ops.add(multiply_7, multiply_2)
        del multiply_2

        # pd_op.subtract: (1x1xf32) <- (1x1xf32, 1x1xf32)
        subtract_15 = paddle._C_ops.subtract(add_6, multiply_6)

        # pd_op.scale: (1x1xf32) <- (1x1xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(subtract_15, full_6, float("1e-10"), True)
        del subtract_15

        # pd_op.divide: (1x1xf32) <- (1x1xf32, 1x1xf32)
        divide_4 = paddle._C_ops.divide(multiply_6, scale_13)

        # pd_op.minimum: (1x1xf32) <- (1x1xf32, 1x1xf32)
        minimum_4 = paddle._C_ops.minimum(split_14, split_8)
        del split_8

        # pd_op.minimum: (1x1xf32) <- (1x1xf32, 1x1xf32)
        minimum_5 = paddle._C_ops.minimum(split_15, split_9)
        del split_9

        # pd_op.maximum: (1x1xf32) <- (1x1xf32, 1x1xf32)
        maximum_4 = paddle._C_ops.maximum(split_16, split_10)
        del split_10

        # pd_op.maximum: (1x1xf32) <- (1x1xf32, 1x1xf32)
        maximum_5 = paddle._C_ops.maximum(split_17, split_11)
        del split_11

        # pd_op.subtract: (1x1xf32) <- (1x1xf32, 1x1xf32)
        subtract_16 = paddle._C_ops.subtract(maximum_4, minimum_4)

        # pd_op.subtract: (1x1xf32) <- (1x1xf32, 1x1xf32)
        subtract_17 = paddle._C_ops.subtract(maximum_5, minimum_5)

        # pd_op.multiply: (1x1xf32) <- (1x1xf32, 1x1xf32)
        multiply_8 = paddle._C_ops.multiply(subtract_16, subtract_17)

        # pd_op.scale: (1x1xf32) <- (1x1xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(multiply_8, full_6, float("1e-10"), True)
        del multiply_8

        # pd_op.subtract: (1x1xf32) <- (1x1xf32, 1x1xf32)
        subtract_18 = paddle._C_ops.subtract(scale_14, scale_13)

        # pd_op.divide: (1x1xf32) <- (1x1xf32, 1x1xf32)
        divide_5 = paddle._C_ops.divide(subtract_18, scale_14)

        # pd_op.subtract: (1x1xf32) <- (1x1xf32, 1x1xf32)
        subtract_19 = paddle._C_ops.subtract(divide_4, divide_5)

        # pd_op.scale: (1x1xf32) <- (1x1xf32, 1xf32)
        scale_15 = paddle._C_ops.scale(subtract_19, full_12, float("1"), True)
        del subtract_19

        # pd_op.scale: (1x1xf32) <- (1x1xf32, 1xf32)
        scale_16 = paddle._C_ops.scale(scale_15, full_6, float("0"), True)
        del scale_15

        # pd_op.sum: (xf32) <- (1x1xf32, 0xi64)
        sum_2 = paddle._C_ops.sum(scale_16, full_int_array_6, None, False)

        # pd_op.divide: (1xf32) <- (xf32, 1xf32)
        divide_6 = paddle._C_ops.divide(sum_2, data_5)

        # pd_op.full: (1xf32) <- ()
        full_16 = paddle._C_ops.full(
            [1], float("2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_43 = full_16

        # pd_op.scale: (1xf32) <- (1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(divide_6, full_16, float("0"), True)

        # pd_op.slice: (300x-1x-1xf32) <- (1x300x-1x-1xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            data_7, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del data_7, full_int_array_0

        # pd_op.gather: (1x-1x-1xf32) <- (300x-1x-1xf32, 1xi64, 1xi32)
        gather_4 = paddle._C_ops.gather(slice_3, data_0, full_0)
        del data_0

        # builtin.combine: ([1x-1x-1xf32]) <- (1x-1x-1xf32)
        combine_9 = [gather_4]

        # pd_op.concat: (1x-1x-1xf32) <- ([1x-1x-1xf32], 1xi32)
        concat_9 = paddle._C_ops.concat(combine_9, full_0)
        del combine_9

        # pd_op.gather: (1x-1x-1xf32) <- (1x-1x-1xf32, 1xi64, 1xi32)
        gather_5 = paddle._C_ops.gather(data_8, data_1, full_0)
        del data_1, data_8

        # builtin.combine: ([1x-1x-1xf32]) <- (1x-1x-1xf32)
        combine_10 = [gather_5]
        del gather_5

        # pd_op.concat: (1x-1x-1xf32) <- ([1x-1x-1xf32], 1xi32)
        concat_10 = paddle._C_ops.concat(combine_10, full_0)
        del combine_10, full_0

        # pd_op.share_data_: (1x-1x-1xf32) <- (1x-1x-1xf32)
        share_data__1 = concat_9.detach()

        # pd_op.shape64: (3xi64) <- (1x-1x-1xf32)
        shape64_0 = paddle._C_ops.shape64(share_data__1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [2]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_1, full_int_array_7, [1], [0]
        )
        del shape64_0

        # pd_op.shape64: (3xi64) <- (1x-1x-1xf32)
        shape64_1 = paddle._C_ops.shape64(share_data__1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [3]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_7, full_int_array_8, [1], [0]
        )
        del full_int_array_7, full_int_array_8, shape64_1

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_9 = [1, 1, 37632, 2]

        # pd_op.uniform: (1x1x37632x2xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_0 = paddle._C_ops.uniform(
            full_int_array_9,
            paddle.float32,
            full_4,
            full_6,
            0,
            paddle.framework._current_expected_place(),
        )
        del full_int_array_9

        # pd_op.unsqueeze: (1x1x-1x-1xf32) <- (1x-1x-1xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(share_data__1, full_int_array_1)
        del share_data__1

        # pd_op.scale: (1x1x37632x2xf32) <- (1x1x37632x2xf32, 1xf32)
        scale_17 = paddle._C_ops.scale(uniform_0, full_16, float("0"), True)

        # pd_op.scale: (1x1x37632x2xf32) <- (1x1x37632x2xf32, 1xf32)
        scale_18 = paddle._C_ops.scale(scale_17, full_6, float("-1"), True)
        del scale_17

        # pd_op.grid_sample: (1x1x1x37632xf32) <- (1x1x-1x-1xf32, 1x1x37632x2xf32)
        grid_sample_0 = paddle._C_ops.grid_sample(
            unsqueeze_0, scale_18, "bilinear", "zeros", False
        )
        del scale_18, unsqueeze_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_10 = [1, 2]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_44 = full_int_array_10

        # pd_op.squeeze: (1x37632xf32) <- (1x1x1x37632xf32, 2xi64)
        squeeze_0 = paddle._C_ops.squeeze(grid_sample_0, full_int_array_10)
        del grid_sample_0

        # pd_op.abs: (1x37632xf32) <- (1x37632xf32)
        abs_1 = paddle._C_ops.abs(squeeze_0)
        del squeeze_0

        # pd_op.scale: (1x37632xf32) <- (1x37632xf32, 1xf32)
        scale_19 = paddle._C_ops.scale(abs_1, full_12, float("0"), True)
        del abs_1

        # pd_op.full: (1xi32) <- ()
        full_17 = paddle._C_ops.full(
            [1], float("9408"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.topk: (1x9408xf32, 1x9408xi64) <- (1x37632xf32, 1xi32)
        topk_0, topk_1 = (lambda x, f: f(x))(
            paddle._C_ops.topk(scale_19, full_17, 1, True, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del full_17, scale_19

        # pd_op.full: (1xf64) <- ()
        full_18 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_19 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (1xi64) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_18, full_19, full_19, dtype="int64")
        del full_18, full_19

        # pd_op.unsqueeze: (1x1xi64) <- (1xi64, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(arange_0, full_int_array_4)
        del arange_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_11 = [1, 9408]

        # pd_op.tile: (1x9408xi64) <- (1x1xi64, 2xi64)
        tile_0 = paddle._C_ops.tile(unsqueeze_1, full_int_array_11)
        del full_int_array_11, unsqueeze_1

        # builtin.combine: ([1x9408xi64, 1x9408xi64]) <- (1x9408xi64, 1x9408xi64)
        combine_11 = [tile_0, topk_1]
        del tile_0, topk_1

        # pd_op.stack: (1x9408x2xi64) <- ([1x9408xi64, 1x9408xi64])
        stack_0 = paddle._C_ops.stack(combine_11, -1)
        del combine_11

        # pd_op.squeeze: (1x37632x2xf32) <- (1x1x37632x2xf32, 1xi64)
        squeeze_1 = paddle._C_ops.squeeze(uniform_0, full_int_array_1)
        del uniform_0

        # pd_op.gather_nd: (1x9408x2xf32) <- (1x37632x2xf32, 1x9408x2xi64)
        gather_nd_0 = paddle._C_ops.gather_nd(squeeze_1, stack_0)
        del squeeze_1, stack_0

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_12 = [1, 3136, 2]

        # pd_op.uniform: (1x3136x2xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_1 = paddle._C_ops.uniform(
            full_int_array_12,
            paddle.float32,
            full_4,
            full_6,
            0,
            paddle.framework._current_expected_place(),
        )
        del full_4, full_int_array_12

        # builtin.combine: ([1x9408x2xf32, 1x3136x2xf32]) <- (1x9408x2xf32, 1x3136x2xf32)
        combine_12 = [gather_nd_0, uniform_1]
        del gather_nd_0, uniform_1

        # pd_op.concat: (1x12544x2xf32) <- ([1x9408x2xf32, 1x3136x2xf32], 1xi32)
        concat_11 = paddle._C_ops.concat(combine_12, full_1)
        del combine_12, full_1

        # pd_op.unsqueeze: (1x1x12544x2xf32) <- (1x12544x2xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(concat_11, full_int_array_1)
        del concat_11

        # pd_op.scale: (1x1x12544x2xf32) <- (1x1x12544x2xf32, 1xf32)
        scale_20 = paddle._C_ops.scale(unsqueeze_2, full_16, float("0"), True)
        del unsqueeze_2

        # pd_op.scale: (1x1x12544x2xf32) <- (1x1x12544x2xf32, 1xf32)
        scale_21 = paddle._C_ops.scale(scale_20, full_6, float("-1"), True)
        del scale_20

        # pd_op.unsqueeze: (1x1x-1x-1xf32) <- (1x-1x-1xf32, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(concat_9, full_int_array_1)

        # pd_op.grid_sample: (1x1x1x12544xf32) <- (1x1x-1x-1xf32, 1x1x12544x2xf32)
        grid_sample_1 = paddle._C_ops.grid_sample(
            unsqueeze_3, scale_21, "bilinear", "zeros", False
        )

        # pd_op.squeeze: (1x12544xf32) <- (1x1x1x12544xf32, 2xi64)
        squeeze_2 = paddle._C_ops.squeeze(grid_sample_1, full_int_array_10)

        # pd_op.unsqueeze: (1x1x-1x-1xf32) <- (1x-1x-1xf32, 1xi64)
        unsqueeze_4 = paddle._C_ops.unsqueeze(concat_10, full_int_array_1)
        del concat_10

        # pd_op.grid_sample: (1x1x1x12544xf32) <- (1x1x-1x-1xf32, 1x1x12544x2xf32)
        grid_sample_2 = paddle._C_ops.grid_sample(
            unsqueeze_4, scale_21, "bilinear", "zeros", False
        )
        del unsqueeze_4

        # pd_op.squeeze: (1x12544xf32) <- (1x1x1x12544xf32, 2xi64)
        squeeze_3 = paddle._C_ops.squeeze(grid_sample_2, full_int_array_10)
        del full_int_array_10, grid_sample_2

        # pd_op.share_data_: (1x12544xf32) <- (1x12544xf32)
        share_data__2 = squeeze_3.detach()
        del squeeze_3

        # pd_op.sigmoid_cross_entropy_with_logits: (1x12544xf32) <- (1x12544xf32, 1x12544xf32, None)
        sigmoid_cross_entropy_with_logits_1 = (
            paddle._C_ops.sigmoid_cross_entropy_with_logits(
                squeeze_2, share_data__2, None, False, -100
            )
        )

        # pd_op.mean: (1xf32) <- (1x12544xf32, 1xi64)
        mean_1 = paddle._C_ops.mean(
            sigmoid_cross_entropy_with_logits_1, full_int_array_1, False
        )

        # pd_op.sum: (xf32) <- (1xf32, 0xi64)
        sum_3 = paddle._C_ops.sum(mean_1, full_int_array_6, None, False)

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_22 = paddle._C_ops.scale(sum_3, full_15, float("0"), True)
        del sum_3

        # pd_op.divide: (1xf32) <- (xf32, 1xf32)
        divide_1 = paddle._C_ops.divide(scale_22, data_5)

        # pd_op.sigmoid: (1x12544xf32) <- (1x12544xf32)
        sigmoid_1 = paddle._C_ops.sigmoid(squeeze_2)

        # pd_op.flatten: (1x12544xf32) <- (1x12544xf32)
        flatten_0 = paddle._C_ops.flatten(sigmoid_1, 1, 1)

        # pd_op.flatten: (1x12544xf32) <- (1x12544xf32)
        flatten_1 = paddle._C_ops.flatten(share_data__2, 1, 1)

        # pd_op.multiply: (1x12544xf32) <- (1x12544xf32, 1x12544xf32)
        multiply_9 = paddle._C_ops.multiply(flatten_0, flatten_1)

        # pd_op.sum: (1xf32) <- (1x12544xf32, 1xi64)
        sum_4 = paddle._C_ops.sum(multiply_9, full_int_array_1, None, False)
        del full_int_array_1

        # pd_op.scale: (1xf32) <- (1xf32, 1xf32)
        scale_23 = paddle._C_ops.scale(sum_4, full_16, float("0"), True)
        del sum_4

        # pd_op.sum: (1xf32) <- (1x12544xf32, 1xi64)
        sum_5 = paddle._C_ops.sum(flatten_0, full_int_array_4, None, False)

        # pd_op.sum: (1xf32) <- (1x12544xf32, 1xi64)
        sum_6 = paddle._C_ops.sum(flatten_1, full_int_array_4, None, False)
        del full_int_array_4

        # pd_op.add: (1xf32) <- (1xf32, 1xf32)
        add_7 = paddle._C_ops.add(sum_5, sum_6)

        # pd_op.scale: (1xf32) <- (1xf32, 1xf32)
        scale_24 = paddle._C_ops.scale(scale_23, full_6, float("1"), True)
        del scale_23

        # pd_op.scale: (1xf32) <- (1xf32, 1xf32)
        scale_25 = paddle._C_ops.scale(add_7, full_6, float("1"), True)
        del add_7, full_6

        # pd_op.divide: (1xf32) <- (1xf32, 1xf32)
        divide_7 = paddle._C_ops.divide(scale_24, scale_25)

        # pd_op.scale: (1xf32) <- (1xf32, 1xf32)
        scale_26 = paddle._C_ops.scale(divide_7, full_12, float("1"), True)
        del full_12

        # pd_op.sum: (xf32) <- (1xf32, 0xi64)
        sum_7 = paddle._C_ops.sum(scale_26, full_int_array_6, None, False)

        # pd_op.divide: (1xf32) <- (xf32, 1xf32)
        divide_8 = paddle._C_ops.divide(sum_7, data_5)
        del data_5

        # pd_op.scale: (1xf32) <- (1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(divide_8, full_15, float("0"), True)
        del (
            abs_0,
            add_4,
            add_5,
            add_6,
            assign_0,
            assign_1,
            assign_10,
            assign_11,
            assign_12,
            assign_13,
            assign_14,
            assign_15,
            assign_16,
            assign_17,
            assign_18,
            assign_19,
            assign_2,
            assign_20,
            assign_21,
            assign_22,
            assign_23,
            assign_24,
            assign_25,
            assign_26,
            assign_27,
            assign_28,
            assign_29,
            assign_3,
            assign_30,
            assign_31,
            assign_32,
            assign_33,
            assign_34,
            assign_35,
            assign_36,
            assign_37,
            assign_38,
            assign_39,
            assign_4,
            assign_40,
            assign_41,
            assign_42,
            assign_43,
            assign_44,
            assign_5,
            assign_6,
            assign_7,
            assign_8,
            assign_9,
            clip_4,
            clip_5,
            concat_7,
            concat_9,
            divide_3,
            divide_4,
            divide_5,
            divide_6,
            divide_7,
            divide_8,
            flatten_0,
            flatten_1,
            full_14,
            full_15,
            full_16,
            full_int_array_6,
            gather_3,
            gather_4,
            grid_sample_1,
            maximum_2,
            maximum_3,
            maximum_4,
            maximum_5,
            mean_0,
            mean_1,
            minimum_2,
            minimum_3,
            minimum_4,
            minimum_5,
            multiply_4,
            multiply_5,
            multiply_6,
            multiply_7,
            multiply_9,
            pow_0,
            scale_10,
            scale_11,
            scale_12,
            scale_13,
            scale_14,
            scale_16,
            scale_21,
            scale_22,
            scale_24,
            scale_25,
            scale_26,
            scale_8,
            scale_9,
            share_data__2,
            sigmoid_0,
            sigmoid_1,
            sigmoid_cross_entropy_with_logits_0,
            sigmoid_cross_entropy_with_logits_1,
            slice_1,
            slice_2,
            slice_3,
            split_12,
            split_14,
            split_15,
            split_16,
            split_17,
            squeeze_2,
            subtract_10,
            subtract_11,
            subtract_12,
            subtract_13,
            subtract_14,
            subtract_16,
            subtract_17,
            subtract_18,
            subtract_9,
            sum_0,
            sum_2,
            sum_5,
            sum_6,
            sum_7,
            unsqueeze_3,
        )

        return scale_0, divide_0, scale_1, divide_1, scale_2
