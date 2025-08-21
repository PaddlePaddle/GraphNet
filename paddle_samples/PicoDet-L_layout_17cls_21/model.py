import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4, data_5):
        # pd_op.multiply: (2x8x10285xf32) <- (2x8x10285xf32, 2x8x10285xf32)
        multiply_1 = paddle._C_ops.multiply(data_2, data_0)

        # pd_op.flatten: (16x10285xf32) <- (2x8x10285xf32)
        flatten_0 = paddle._C_ops.flatten(multiply_1, 0, 1)

        # pd_op.flatten: (16x36xi64) <- (2x8x36xi64)
        flatten_1 = paddle._C_ops.flatten(data_1, 0, 1)

        # pd_op.index_sample: (16x36xf32) <- (16x10285xf32, 16x36xi64)
        index_sample_0 = paddle._C_ops.index_sample(flatten_0, flatten_1)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_0 = [2, 8, -1]

        # pd_op.reshape: (2x8x36xf32) <- (16x36xf32, 3xi64)
        reshape_0 = paddle._C_ops.reshape(index_sample_0, full_int_array_0)

        # pd_op.mean: (2x8x1xf32) <- (2x8x36xf32)
        mean_0 = paddle._C_ops.mean(reshape_0, [-1], True)

        # pd_op.subtract: (2x8x36xf32) <- (2x8x36xf32, 2x8x1xf32)
        subtract_0 = paddle._C_ops.subtract(reshape_0, mean_0)

        # pd_op.pow: (2x8x36xf32) <- (2x8x36xf32)
        pow_0 = paddle._C_ops.pow(subtract_0, float("2"))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [-1]

        # pd_op.sum: (2x8x1xf32) <- (2x8x36xf32, 1xi64)
        sum_1 = paddle._C_ops.sum(pow_0, full_int_array_1, None, True)

        # pd_op.numel: (xi64) <- (2x8x36xf32)
        numel_0 = paddle._C_ops.numel(reshape_0)

        # pd_op.cast: (xi64) <- (xi64)
        cast_0 = paddle._C_ops.cast(numel_0, paddle.int64)

        # pd_op.numel: (xi64) <- (2x8x1xf32)
        numel_1 = paddle._C_ops.numel(sum_1)

        # pd_op.cast: (xi64) <- (xi64)
        cast_1 = paddle._C_ops.cast(numel_1, paddle.int64)

        # pd_op.cast: (xf32) <- (xi64)
        cast_2 = paddle._C_ops.cast(cast_0, paddle.float32)

        # pd_op.cast: (xf32) <- (xi64)
        cast_3 = paddle._C_ops.cast(cast_1, paddle.float32)

        # pd_op.divide: (xf32) <- (xf32, xf32)
        divide_0 = paddle._C_ops.divide(cast_2, cast_3)

        # pd_op.full: (xf32) <- ()
        full_0 = paddle._C_ops.full(
            [], float("1"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (xb) <- (xf32, xf32)
        greater_than_1 = paddle._C_ops.greater_than(divide_0, full_0)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(divide_0, full_1, float("-1"), True)

        # pd_op.where: (xf32) <- (xb, xf32, xf32)
        where_0 = paddle._C_ops.where(greater_than_1, scale_0, full_0)

        # pd_op.divide: (2x8x1xf32) <- (2x8x1xf32, xf32)
        divide_1 = paddle._C_ops.divide(sum_1, where_0)

        # pd_op.sqrt: (2x8x1xf32) <- (2x8x1xf32)
        sqrt_0 = paddle._C_ops.sqrt(divide_1)

        # pd_op.add: (2x8x1xf32) <- (2x8x1xf32, 2x8x1xf32)
        add_0 = paddle._C_ops.add(mean_0, sqrt_0)

        # pd_op.greater_than: (2x8x10285xb) <- (2x8x10285xf32, 2x8x1xf32)
        greater_than_2 = paddle._C_ops.greater_than(multiply_1, add_0)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (2x8x10285xf32) <- (2x8x10285xf32, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            data_0, full_2, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.where: (2x8x10285xf32) <- (2x8x10285xb, 2x8x10285xf32, 2x8x10285xf32)
        where_1 = paddle._C_ops.where(greater_than_2, data_0, full_like_0)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [0, 1]

        # pd_op.unsqueeze: (1x1x10285x2xf32) <- (10285x2xf32, 2xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(data_3, full_int_array_2)

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("3"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split_with_num: ([1x1x10285x1xf32, 1x1x10285x1xf32]) <- (1x1x10285x2xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(unsqueeze_0, 2, full_3)

        # builtin.split: (1x1x10285x1xf32, 1x1x10285x1xf32) <- ([1x1x10285x1xf32, 1x1x10285x1xf32])
        (
            split_0,
            split_1,
        ) = split_with_num_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [2]

        # pd_op.unsqueeze: (2x8x1x4xf32) <- (2x8x4xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(data_4, full_int_array_3)

        # pd_op.split_with_num: ([2x8x1x1xf32, 2x8x1x1xf32, 2x8x1x1xf32, 2x8x1x1xf32]) <- (2x8x1x4xf32, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(unsqueeze_1, 4, full_3)

        # builtin.split: (2x8x1x1xf32, 2x8x1x1xf32, 2x8x1x1xf32, 2x8x1x1xf32) <- ([2x8x1x1xf32, 2x8x1x1xf32, 2x8x1x1xf32, 2x8x1x1xf32])
        (
            split_2,
            split_3,
            split_4,
            split_5,
        ) = split_with_num_1

        # pd_op.subtract: (2x8x10285x1xf32) <- (1x1x10285x1xf32, 2x8x1x1xf32)
        subtract_1 = paddle._C_ops.subtract(split_0, split_2)

        # pd_op.subtract: (2x8x10285x1xf32) <- (1x1x10285x1xf32, 2x8x1x1xf32)
        subtract_2 = paddle._C_ops.subtract(split_1, split_3)

        # pd_op.subtract: (2x8x10285x1xf32) <- (2x8x1x1xf32, 1x1x10285x1xf32)
        subtract_3 = paddle._C_ops.subtract(split_4, split_0)

        # pd_op.subtract: (2x8x10285x1xf32) <- (2x8x1x1xf32, 1x1x10285x1xf32)
        subtract_4 = paddle._C_ops.subtract(split_5, split_1)

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([2x8x10285x1xf32, 2x8x10285x1xf32, 2x8x10285x1xf32, 2x8x10285x1xf32]) <- (2x8x10285x1xf32, 2x8x10285x1xf32, 2x8x10285x1xf32, 2x8x10285x1xf32)
        combine_0 = [subtract_1, subtract_2, subtract_3, subtract_4]

        # pd_op.concat: (2x8x10285x4xf32) <- ([2x8x10285x1xf32, 2x8x10285x1xf32, 2x8x10285x1xf32, 2x8x10285x1xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_4)

        # pd_op.min: (2x8x10285xf32) <- (2x8x10285x4xf32, 1xi64)
        min_0 = paddle._C_ops.min(concat_0, full_int_array_1, False)

        # pd_op.full: (xf32) <- ()
        full_5 = paddle._C_ops.full(
            [],
            float("1e-09"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.greater_than: (2x8x10285xb) <- (2x8x10285xf32, xf32)
        greater_than_3 = paddle._C_ops.greater_than(min_0, full_5)

        # pd_op.cast: (2x8x10285xf32) <- (2x8x10285xb)
        cast_4 = paddle._C_ops.cast(greater_than_3, paddle.float32)

        # pd_op.multiply: (2x8x10285xf32) <- (2x8x10285xf32, 2x8x10285xf32)
        multiply_2 = paddle._C_ops.multiply(where_1, cast_4)

        # pd_op.multiply: (2x8x10285xf32) <- (2x8x10285xf32, 2x8x1xf32)
        multiply_0 = paddle._C_ops.multiply(multiply_2, data_5)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [-2]

        # pd_op.sum: (2x10285xf32) <- (2x8x10285xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(multiply_0, full_int_array_4, None, False)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_5 = []

        # pd_op.max: (xf32) <- (2x10285xf32, 0xi64)
        max_0 = paddle._C_ops.max(sum_0, full_int_array_5, False)

        # pd_op.greater_than: (xb) <- (xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(max_0, full_0)
        return greater_than_0, sum_0, multiply_0
