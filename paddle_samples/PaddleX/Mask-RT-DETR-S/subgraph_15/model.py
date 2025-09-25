import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        data_0,
        data_1,
        data_2,
        data_3,
        data_4,
        data_5,
        data_6,
        data_7,
        data_8,
        data_9,
    ):
        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [100, 300]

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("2"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_0 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_1 = full_0

        # pd_op.split: ([6x1x100x2xf32, 6x1x300x2xf32]) <- (6x1x400x2xf32, 2xi64, 1xi32)
        split_0 = paddle._C_ops.split(data_0, full_int_array_0, full_0)
        del data_0

        # builtin.split: (6x1x100x2xf32, 6x1x300x2xf32) <- ([6x1x100x2xf32, 6x1x300x2xf32])
        (
            split_1,
            split_2,
        ) = split_0
        del split_0

        # pd_op.split: ([6x1x100x4xf32, 6x1x300x4xf32]) <- (6x1x400x4xf32, 2xi64, 1xi32)
        split_3 = paddle._C_ops.split(data_1, full_int_array_0, full_0)
        del data_1

        # builtin.split: (6x1x100x4xf32, 6x1x300x4xf32) <- ([6x1x100x4xf32, 6x1x300x4xf32])
        (
            split_4,
            split_5,
        ) = split_3
        del split_3

        # pd_op.split: ([6x1x100x160x160xf32, 6x1x300x160x160xf32]) <- (6x1x400x160x160xf32, 2xi64, 1xi32)
        split_6 = paddle._C_ops.split(data_2, full_int_array_0, full_0)
        del data_2

        # builtin.split: (6x1x100x160x160xf32, 6x1x300x160x160xf32) <- ([6x1x100x160x160xf32, 6x1x300x160x160xf32])
        (
            split_7,
            split_8,
        ) = split_6
        del split_6

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_2 = full_1

        # pd_op.split: ([1x100x2xf32, 1x300x2xf32]) <- (1x400x2xf32, 2xi64, 1xi32)
        split_9 = paddle._C_ops.split(data_6, full_int_array_0, full_1)
        del data_6

        # builtin.split: (1x100x2xf32, 1x300x2xf32) <- ([1x100x2xf32, 1x300x2xf32])
        (
            split_10,
            split_11,
        ) = split_9
        del split_9

        # pd_op.split: ([1x100x4xf32, 1x300x4xf32]) <- (1x400x4xf32, 2xi64, 1xi32)
        split_12 = paddle._C_ops.split(data_7, full_int_array_0, full_1)
        del data_7

        # builtin.split: (1x100x4xf32, 1x300x4xf32) <- ([1x100x4xf32, 1x300x4xf32])
        (
            split_13,
            split_14,
        ) = split_12
        del split_12

        # pd_op.split: ([1x100x160x160xf32, 1x300x160x160xf32]) <- (1x400x160x160xf32, 2xi64, 1xi32)
        split_15 = paddle._C_ops.split(data_8, full_int_array_0, full_1)
        del data_8, full_int_array_0

        # builtin.split: (1x100x160x160xf32, 1x300x160x160xf32) <- ([1x100x160x160xf32, 1x300x160x160xf32])
        (
            split_16,
            split_17,
        ) = split_15
        del split_15

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [0]

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

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_8 = full_int_array_1

        # pd_op.unsqueeze: (1x1x300x2xf32) <- (1x300x2xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(split_11, full_int_array_1)

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_9 = full_2

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_10 = full_2

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_11 = full_2

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_12 = full_2

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_13 = full_2

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_14 = full_2

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_15 = full_2

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_16 = full_2

        # builtin.combine: ([1x1x300x2xf32, 6x1x300x2xf32]) <- (1x1x300x2xf32, 6x1x300x2xf32)
        combine_0 = [unsqueeze_0, split_2]

        # pd_op.concat: (7x1x300x2xf32) <- ([1x1x300x2xf32, 6x1x300x2xf32], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_0, full_2)
        del combine_0

        # pd_op.unsqueeze: (1x1x300x4xf32) <- (1x300x4xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(split_14, full_int_array_1)
        del split_14

        # builtin.combine: ([1x1x300x4xf32, 6x1x300x4xf32]) <- (1x1x300x4xf32, 6x1x300x4xf32)
        combine_1 = [unsqueeze_1, split_5]

        # pd_op.concat: (7x1x300x4xf32) <- ([1x1x300x4xf32, 6x1x300x4xf32], 1xi32)
        concat_7 = paddle._C_ops.concat(combine_1, full_2)
        del combine_1

        # pd_op.unsqueeze: (1x1x300x160x160xf32) <- (1x300x160x160xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(split_17, full_int_array_1)

        # builtin.combine: ([1x1x300x160x160xf32, 6x1x300x160x160xf32]) <- (1x1x300x160x160xf32, 6x1x300x160x160xf32)
        combine_2 = [unsqueeze_2, split_8]

        # pd_op.concat: (7x1x300x160x160xf32) <- ([1x1x300x160x160xf32, 6x1x300x160x160xf32], 1xi32)
        concat_8 = paddle._C_ops.concat(combine_2, full_2)
        del combine_2

        # pd_op.unsqueeze: (1x1x100x2xf32) <- (1x100x2xf32, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(split_10, full_int_array_1)

        # builtin.combine: ([1x1x100x2xf32, 6x1x100x2xf32]) <- (1x1x100x2xf32, 6x1x100x2xf32)
        combine_3 = [unsqueeze_3, split_1]

        # pd_op.concat: (7x1x100x2xf32) <- ([1x1x100x2xf32, 6x1x100x2xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_3, full_2)
        del combine_3

        # pd_op.unsqueeze: (1x1x100x4xf32) <- (1x100x4xf32, 1xi64)
        unsqueeze_4 = paddle._C_ops.unsqueeze(split_13, full_int_array_1)
        del split_13

        # builtin.combine: ([1x1x100x4xf32, 6x1x100x4xf32]) <- (1x1x100x4xf32, 6x1x100x4xf32)
        combine_4 = [unsqueeze_4, split_4]

        # pd_op.concat: (7x1x100x4xf32) <- ([1x1x100x4xf32, 6x1x100x4xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_4, full_2)
        del combine_4

        # pd_op.unsqueeze: (1x1x100x160x160xf32) <- (1x100x160x160xf32, 1xi64)
        unsqueeze_5 = paddle._C_ops.unsqueeze(split_16, full_int_array_1)

        # builtin.combine: ([1x1x100x160x160xf32, 6x1x100x160x160xf32]) <- (1x1x100x160x160xf32, 6x1x100x160x160xf32)
        combine_5 = [unsqueeze_5, split_7]

        # pd_op.concat: (7x1x100x160x160xf32) <- ([1x1x100x160x160xf32, 6x1x100x160x160xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_5, full_2)
        del combine_5

        # pd_op.unsqueeze: (1x1x300x2xf32) <- (1x300x2xf32, 1xi64)
        unsqueeze_6 = paddle._C_ops.unsqueeze(data_3, full_int_array_1)
        del data_3

        # builtin.combine: ([1x1x300x2xf32, 7x1x300x2xf32]) <- (1x1x300x2xf32, 7x1x300x2xf32)
        combine_6 = [unsqueeze_6, concat_6]

        # pd_op.concat: (8x1x300x2xf32) <- ([1x1x300x2xf32, 7x1x300x2xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_6, full_2)
        del combine_6

        # pd_op.unsqueeze: (1x1x300x4xf32) <- (1x300x4xf32, 1xi64)
        unsqueeze_7 = paddle._C_ops.unsqueeze(data_4, full_int_array_1)
        del data_4

        # builtin.combine: ([1x1x300x4xf32, 7x1x300x4xf32]) <- (1x1x300x4xf32, 7x1x300x4xf32)
        combine_7 = [unsqueeze_7, concat_7]

        # pd_op.concat: (8x1x300x4xf32) <- ([1x1x300x4xf32, 7x1x300x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_7, full_2)
        del combine_7

        # pd_op.unsqueeze: (1x1x300x160x160xf32) <- (1x300x160x160xf32, 1xi64)
        unsqueeze_8 = paddle._C_ops.unsqueeze(data_5, full_int_array_1)
        del data_5

        # builtin.combine: ([1x1x300x160x160xf32, 7x1x300x160x160xf32]) <- (1x1x300x160x160xf32, 7x1x300x160x160xf32)
        combine_8 = [unsqueeze_8, concat_8]

        # pd_op.concat: (8x1x300x160x160xf32) <- ([1x1x300x160x160xf32, 7x1x300x160x160xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_8, full_2)
        del combine_8

        # pd_op.cast: (1x640x640xf32) <- (1x640x640xui8)
        cast_0 = paddle._C_ops.cast(data_9, paddle.float32)
        del (
            assign_0,
            assign_1,
            assign_10,
            assign_11,
            assign_12,
            assign_13,
            assign_14,
            assign_15,
            assign_16,
            assign_2,
            assign_3,
            assign_4,
            assign_5,
            assign_6,
            assign_7,
            assign_8,
            assign_9,
            concat_6,
            concat_7,
            concat_8,
            data_9,
            full_0,
            full_1,
            full_2,
            full_int_array_1,
            split_1,
            split_10,
            split_11,
            split_16,
            split_17,
            split_2,
            split_4,
            split_5,
            split_7,
            split_8,
            unsqueeze_0,
            unsqueeze_1,
            unsqueeze_2,
            unsqueeze_3,
            unsqueeze_4,
            unsqueeze_5,
            unsqueeze_6,
            unsqueeze_7,
            unsqueeze_8,
        )

        return concat_0, concat_1, concat_2, cast_0, concat_3, concat_4, concat_5
