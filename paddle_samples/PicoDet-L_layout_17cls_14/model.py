import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [0, 1]

        # pd_op.unsqueeze: (1x1x6885x2xf32) <- (6885x2xf32, 2xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(data_0, full_int_array_0)

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("3"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split_with_num: ([1x1x6885x1xf32, 1x1x6885x1xf32]) <- (1x1x6885x2xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(unsqueeze_0, 2, full_0)

        # builtin.split: (1x1x6885x1xf32, 1x1x6885x1xf32) <- ([1x1x6885x1xf32, 1x1x6885x1xf32])
        (
            split_0,
            split_1,
        ) = split_with_num_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [2]

        # pd_op.unsqueeze: (2x-1x1x4xf32) <- (2x-1x4xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(data_1, full_int_array_1)

        # pd_op.split_with_num: ([2x-1x1x1xf32, 2x-1x1x1xf32, 2x-1x1x1xf32, 2x-1x1x1xf32]) <- (2x-1x1x4xf32, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(unsqueeze_1, 4, full_0)

        # builtin.split: (2x-1x1x1xf32, 2x-1x1x1xf32, 2x-1x1x1xf32, 2x-1x1x1xf32) <- ([2x-1x1x1xf32, 2x-1x1x1xf32, 2x-1x1x1xf32, 2x-1x1x1xf32])
        (
            split_2,
            split_3,
            split_4,
            split_5,
        ) = split_with_num_1

        # pd_op.subtract: (2x-1x6885x1xf32) <- (1x1x6885x1xf32, 2x-1x1x1xf32)
        subtract_0 = paddle._C_ops.subtract(split_0, split_2)

        # pd_op.subtract: (2x-1x6885x1xf32) <- (1x1x6885x1xf32, 2x-1x1x1xf32)
        subtract_1 = paddle._C_ops.subtract(split_1, split_3)

        # pd_op.subtract: (2x-1x6885x1xf32) <- (2x-1x1x1xf32, 1x1x6885x1xf32)
        subtract_2 = paddle._C_ops.subtract(split_4, split_0)

        # pd_op.subtract: (2x-1x6885x1xf32) <- (2x-1x1x1xf32, 1x1x6885x1xf32)
        subtract_3 = paddle._C_ops.subtract(split_5, split_1)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([2x-1x6885x1xf32, 2x-1x6885x1xf32, 2x-1x6885x1xf32, 2x-1x6885x1xf32]) <- (2x-1x6885x1xf32, 2x-1x6885x1xf32, 2x-1x6885x1xf32, 2x-1x6885x1xf32)
        combine_0 = [subtract_0, subtract_1, subtract_2, subtract_3]

        # pd_op.concat: (2x-1x6885x4xf32) <- ([2x-1x6885x1xf32, 2x-1x6885x1xf32, 2x-1x6885x1xf32, 2x-1x6885x1xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [-1]

        # pd_op.min: (2x-1x6885xf32) <- (2x-1x6885x4xf32, 1xi64)
        min_0 = paddle._C_ops.min(concat_0, full_int_array_2, False)

        # pd_op.full: (xf32) <- ()
        full_2 = paddle._C_ops.full(
            [],
            float("1e-09"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.greater_than: (2x-1x6885xb) <- (2x-1x6885xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(min_0, full_2)

        # pd_op.cast: (2x-1x6885xf32) <- (2x-1x6885xb)
        cast_0 = paddle._C_ops.cast(greater_than_0, paddle.float32)
        return cast_0
