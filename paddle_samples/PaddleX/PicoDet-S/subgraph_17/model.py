import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6):
        # pd_op.multiply: (16x-1x3060xf32) <- (16x-1x3060xf32, 16x-1x3060xf32)
        multiply_0 = paddle._C_ops.multiply(data_3, data_1)
        del data_3

        # pd_op.flatten: (-1x3060xf32) <- (16x-1x3060xf32)
        flatten_0 = paddle._C_ops.flatten(multiply_0, 0, 1)

        # pd_op.flatten: (-1x36xi64) <- (16x-1x36xi64)
        flatten_1 = paddle._C_ops.flatten(data_2, 0, 1)
        del data_2

        # pd_op.index_sample: (-1x36xf32) <- (-1x3060xf32, -1x36xi64)
        index_sample_0 = paddle._C_ops.index_sample(flatten_0, flatten_1)
        del flatten_0, flatten_1

        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("16"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("-1"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_0 = [full_0, data_0, full_1]
        del data_0, full_0, full_1

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.reshape: (16x-1x-1xf32) <- (-1x36xf32, 3xi64)
        reshape_0 = paddle._C_ops.reshape(index_sample_0, stack_0)
        del index_sample_0, stack_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [-1]

        # pd_op.mean: (16x-1x1xf32) <- (16x-1x-1xf32, 1xi64)
        mean_0 = paddle._C_ops.mean(reshape_0, full_int_array_0, True)

        # pd_op.subtract: (16x-1x-1xf32) <- (16x-1x-1xf32, 16x-1x1xf32)
        subtract_0 = paddle._C_ops.subtract(reshape_0, mean_0)

        # pd_op.pow: (16x-1x-1xf32) <- (16x-1x-1xf32)
        pow_0 = paddle._C_ops.pow(subtract_0, float("2"))
        del subtract_0

        # pd_op.sum: (16x-1x1xf32) <- (16x-1x-1xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(pow_0, full_int_array_0, paddle.float32, True)
        del pow_0

        # pd_op.numel: (xi64) <- (16x-1x-1xf32)
        numel_0 = paddle._C_ops.numel(reshape_0)
        del reshape_0

        # pd_op.cast: (xi64) <- (xi64)
        cast_0 = paddle._C_ops.cast(numel_0, paddle.int64)
        del numel_0

        # pd_op.numel: (xi64) <- (16x-1x1xf32)
        numel_1 = paddle._C_ops.numel(sum_0)

        # pd_op.cast: (xi64) <- (xi64)
        cast_1 = paddle._C_ops.cast(numel_1, paddle.int64)
        del numel_1

        # pd_op.cast: (xf32) <- (xi64)
        cast_2 = paddle._C_ops.cast(cast_0, paddle.float32)
        del cast_0

        # pd_op.cast: (xf32) <- (xi64)
        cast_3 = paddle._C_ops.cast(cast_1, paddle.float32)
        del cast_1

        # pd_op.divide: (xf32) <- (xf32, xf32)
        divide_0 = paddle._C_ops.divide(cast_2, cast_3)
        del cast_2, cast_3

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(divide_0, full_2, float("-1"), True)
        del divide_0, full_2

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (xf32) <- (xf32, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            scale_0, full_3, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.maximum: (xf32) <- (xf32, xf32)
        maximum_0 = paddle._C_ops.maximum(scale_0, full_like_0)
        del full_like_0, scale_0

        # pd_op.divide: (16x-1x1xf32) <- (16x-1x1xf32, xf32)
        divide_1 = paddle._C_ops.divide(sum_0, maximum_0)
        del maximum_0, sum_0

        # pd_op.sqrt: (16x-1x1xf32) <- (16x-1x1xf32)
        sqrt_0 = paddle._C_ops.sqrt(divide_1)
        del divide_1

        # pd_op.add: (16x-1x1xf32) <- (16x-1x1xf32, 16x-1x1xf32)
        add_0 = paddle._C_ops.add(mean_0, sqrt_0)
        del mean_0, sqrt_0

        # pd_op.greater_than: (16x-1x3060xb) <- (16x-1x3060xf32, 16x-1x1xf32)
        greater_than_1 = paddle._C_ops.greater_than(multiply_0, add_0)
        del add_0, multiply_0

        # pd_op.full_like: (16x-1x3060xf32) <- (16x-1x3060xf32, 1xf32)
        full_like_1 = paddle._C_ops.full_like(
            data_1, full_3, paddle.float32, paddle.framework._current_expected_place()
        )
        del full_3

        # pd_op.where: (16x-1x3060xf32) <- (16x-1x3060xb, 16x-1x3060xf32, 16x-1x3060xf32)
        where_0 = paddle._C_ops.where(greater_than_1, data_1, full_like_1)
        del data_1, full_like_1, greater_than_1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [0, 1]

        # pd_op.unsqueeze: (1x1x3060x2xf32) <- (3060x2xf32, 2xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(data_4, full_int_array_1)
        del data_4, full_int_array_1

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("3"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split_with_num: ([1x1x3060x1xf32, 1x1x3060x1xf32]) <- (1x1x3060x2xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(unsqueeze_0, 2, full_4)
        del unsqueeze_0

        # builtin.split: (1x1x3060x1xf32, 1x1x3060x1xf32) <- ([1x1x3060x1xf32, 1x1x3060x1xf32])
        (
            split_0,
            split_1,
        ) = split_with_num_0
        del split_with_num_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        # pd_op.unsqueeze: (16x-1x1x4xf32) <- (16x-1x4xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(data_5, full_int_array_2)
        del data_5, full_int_array_2

        # pd_op.split_with_num: ([16x-1x1x1xf32, 16x-1x1x1xf32, 16x-1x1x1xf32, 16x-1x1x1xf32]) <- (16x-1x1x4xf32, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(unsqueeze_1, 4, full_4)
        del full_4, unsqueeze_1

        # builtin.split: (16x-1x1x1xf32, 16x-1x1x1xf32, 16x-1x1x1xf32, 16x-1x1x1xf32) <- ([16x-1x1x1xf32, 16x-1x1x1xf32, 16x-1x1x1xf32, 16x-1x1x1xf32])
        (
            split_2,
            split_3,
            split_4,
            split_5,
        ) = split_with_num_1
        del split_with_num_1

        # pd_op.subtract: (16x-1x3060x1xf32) <- (1x1x3060x1xf32, 16x-1x1x1xf32)
        subtract_1 = paddle._C_ops.subtract(split_0, split_2)
        del split_2

        # pd_op.subtract: (16x-1x3060x1xf32) <- (1x1x3060x1xf32, 16x-1x1x1xf32)
        subtract_2 = paddle._C_ops.subtract(split_1, split_3)
        del split_3

        # pd_op.subtract: (16x-1x3060x1xf32) <- (16x-1x1x1xf32, 1x1x3060x1xf32)
        subtract_3 = paddle._C_ops.subtract(split_4, split_0)
        del split_0, split_4

        # pd_op.subtract: (16x-1x3060x1xf32) <- (16x-1x1x1xf32, 1x1x3060x1xf32)
        subtract_4 = paddle._C_ops.subtract(split_5, split_1)
        del split_1, split_5

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([16x-1x3060x1xf32, 16x-1x3060x1xf32, 16x-1x3060x1xf32, 16x-1x3060x1xf32]) <- (16x-1x3060x1xf32, 16x-1x3060x1xf32, 16x-1x3060x1xf32, 16x-1x3060x1xf32)
        combine_1 = [subtract_1, subtract_2, subtract_3, subtract_4]
        del subtract_1, subtract_2, subtract_3, subtract_4

        # pd_op.concat: (16x-1x3060x4xf32) <- ([16x-1x3060x1xf32, 16x-1x3060x1xf32, 16x-1x3060x1xf32, 16x-1x3060x1xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_1, full_5)
        del combine_1, full_5

        # pd_op.min: (16x-1x3060xf32) <- (16x-1x3060x4xf32, 1xi64)
        min_0 = paddle._C_ops.min(concat_0, full_int_array_0, False)
        del concat_0, full_int_array_0

        # pd_op.full: (xf32) <- ()
        full_6 = paddle._C_ops.full(
            [],
            float("1e-09"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.greater_than: (16x-1x3060xb) <- (16x-1x3060xf32, xf32)
        greater_than_2 = paddle._C_ops.greater_than(min_0, full_6)
        del full_6, min_0

        # pd_op.cast: (16x-1x3060xf32) <- (16x-1x3060xb)
        cast_4 = paddle._C_ops.cast(greater_than_2, paddle.float32)
        del greater_than_2

        # pd_op.multiply: (16x-1x3060xf32) <- (16x-1x3060xf32, 16x-1x3060xf32)
        multiply_1 = paddle._C_ops.multiply(where_0, cast_4)
        del cast_4, where_0

        # pd_op.multiply: (16x-1x3060xf32) <- (16x-1x3060xf32, 16x-1x1xf32)
        multiply_2 = paddle._C_ops.multiply(multiply_1, data_6)
        del data_6, multiply_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [-2]

        # pd_op.sum: (16x3060xf32) <- (16x-1x3060xf32, 1xi64)
        sum_1 = paddle._C_ops.sum(multiply_2, full_int_array_3, None, False)
        del full_int_array_3

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_4 = []

        # pd_op.max: (xf32) <- (16x3060xf32, 0xi64)
        max_0 = paddle._C_ops.max(sum_1, full_int_array_4, False)
        del full_int_array_4

        # pd_op.full: (xf32) <- ()
        full_7 = paddle._C_ops.full(
            [], float("1"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (xb) <- (xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(max_0, full_7)
        del full_7, max_0, multiply_2, sum_1

        return greater_than_0
