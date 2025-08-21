import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3):
        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [200, 300]

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("2"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_0 = full_0

        # pd_op.split: ([6x1x200x4xf32, 6x1x300x4xf32]) <- (6x1x500x4xf32, 2xi64, 1xi32)
        split_4 = paddle._C_ops.split(data_1, full_int_array_1, full_0)

        # builtin.split: (6x1x200x4xf32, 6x1x300x4xf32) <- ([6x1x200x4xf32, 6x1x300x4xf32])
        (
            split_2,
            split_0,
        ) = split_4

        # pd_op.split: ([6x1x200x4xf32, 6x1x300x4xf32]) <- (6x1x500x4xf32, 2xi64, 1xi32)
        split_5 = paddle._C_ops.split(data_3, full_int_array_1, full_0)

        # builtin.split: (6x1x200x4xf32, 6x1x300x4xf32) <- ([6x1x200x4xf32, 6x1x300x4xf32])
        (
            split_3,
            split_1,
        ) = split_5

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_0

        # pd_op.unsqueeze: (1x1x300x4xf32) <- (1x300x4xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(data_0, full_int_array_0)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_2 = full_1

        # builtin.combine: ([1x1x300x4xf32, 6x1x300x4xf32]) <- (1x1x300x4xf32, 6x1x300x4xf32)
        combine_0 = [unsqueeze_0, split_0]

        # pd_op.concat: (7x1x300x4xf32) <- ([1x1x300x4xf32, 6x1x300x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_1)

        # pd_op.unsqueeze: (1x1x300x4xf32) <- (1x300x4xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(data_2, full_int_array_0)

        # builtin.combine: ([1x1x300x4xf32, 6x1x300x4xf32]) <- (1x1x300x4xf32, 6x1x300x4xf32)
        combine_1 = [unsqueeze_1, split_1]

        # pd_op.concat: (7x1x300x4xf32) <- ([1x1x300x4xf32, 6x1x300x4xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_1)
        return (
            full_0,
            split_0,
            assign_0,
            split_1,
            full_int_array_0,
            unsqueeze_0,
            full_1,
            assign_1,
            unsqueeze_1,
            assign_2,
            concat_0,
            concat_1,
            split_2,
            split_3,
        )
