import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3):
        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("2"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split_with_num: ([2x8400x2xf32, 2x8400x2xf32]) <- (2x8400x4xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(data_0, 2, full_0)
        del data_0, full_0

        # builtin.split: (2x8400x2xf32, 2x8400x2xf32) <- ([2x8400x2xf32, 2x8400x2xf32])
        (
            split_0,
            split_1,
        ) = split_with_num_0
        del split_with_num_0

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (2x8400x2xf32) <- (2x8400x2xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(split_0, full_1, float("0"), True)
        del full_1, split_0

        # pd_op.add: (2x8400x2xf32) <- (2x8400x2xf32, 8400x2xf32)
        add_0 = paddle._C_ops.add(scale_0, data_1)
        del scale_0

        # pd_op.add: (2x8400x2xf32) <- (2x8400x2xf32, 8400x2xf32)
        add_1 = paddle._C_ops.add(split_1, data_1)
        del data_1, split_1

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([2x8400x2xf32, 2x8400x2xf32]) <- (2x8400x2xf32, 2x8400x2xf32)
        combine_0 = [add_0, add_1]
        del add_0, add_1

        # pd_op.concat: (2x8400x4xf32) <- ([2x8400x2xf32, 2x8400x2xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_2)
        del combine_0

        # pd_op.multiply: (2x8400x4xf32) <- (2x8400x4xf32, 8400x1xf32)
        multiply_0 = paddle._C_ops.multiply(concat_0, data_2)
        del concat_0, data_2

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split_with_num: ([2x1xf32, 2x1xf32]) <- (2x2xf32, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(data_3, 2, full_3)
        del data_3, full_3

        # builtin.split: (2x1xf32, 2x1xf32) <- ([2x1xf32, 2x1xf32])
        (
            split_2,
            split_3,
        ) = split_with_num_1
        del split_with_num_1

        # builtin.combine: ([2x1xf32, 2x1xf32, 2x1xf32, 2x1xf32]) <- (2x1xf32, 2x1xf32, 2x1xf32, 2x1xf32)
        combine_1 = [split_3, split_2, split_3, split_2]
        del split_2, split_3

        # pd_op.concat: (2x4xf32) <- ([2x1xf32, 2x1xf32, 2x1xf32, 2x1xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_2)
        del combine_1, full_2

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_0 = [-1, 1, 4]

        # pd_op.reshape: (2x1x4xf32) <- (2x4xf32, 3xi64)
        reshape_0 = paddle._C_ops.reshape(concat_1, full_int_array_0)
        del concat_1, full_int_array_0

        # pd_op.divide: (2x8400x4xf32) <- (2x8400x4xf32, 2x1x4xf32)
        divide_0 = paddle._C_ops.divide(multiply_0, reshape_0)
        del multiply_0, reshape_0

        return divide_0
