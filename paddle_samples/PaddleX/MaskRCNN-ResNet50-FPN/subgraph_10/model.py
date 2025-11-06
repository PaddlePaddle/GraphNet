import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3):
        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x6xf32]) <- (-1x6xf32)
        combine_0 = [data_0]
        del data_0

        # pd_op.concat: (-1x6xf32) <- ([-1x6xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_0, full_0)
        del combine_0

        # builtin.combine: ([1xi32]) <- (1xi32)
        combine_1 = [data_1]
        del data_1

        # pd_op.concat: (1xi32) <- ([1xi32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_1, full_0)
        del combine_1

        # pd_op.divide: (1x2xf32) <- (1x2xf32, 1x2xf32)
        divide_0 = paddle._C_ops.divide(data_2, data_3)
        del data_2

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x2xf32) <- (1x2xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(divide_0, full_1, float("0.5"), True)
        del divide_0

        # pd_op.floor: (1x2xf32) <- (1x2xf32)
        floor_0 = paddle._C_ops.floor(scale_0)
        del scale_0

        # pd_op.cast: (1xi64) <- (1xi32)
        cast_0 = paddle._C_ops.cast(concat_2, paddle.int64)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_0 = []

        # pd_op.reshape: (xi64) <- (1xi64, 0xi64)
        reshape_0 = paddle._C_ops.reshape(cast_0, full_int_array_0)
        del cast_0, full_int_array_0

        # pd_op.full: (xi64) <- ()
        full_2 = paddle._C_ops.full(
            [], float("2"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_2 = [reshape_0, full_2]
        del full_2

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_2, 0)
        del combine_2

        # pd_op.expand: (-1x2xf32) <- (1x2xf32, 2xi64)
        expand_0 = paddle._C_ops.expand(floor_0, stack_0)
        del floor_0, stack_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [0, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [1, 1]

        # pd_op.slice: (xf32) <- (1x2xf32, 2xi64, 2xi64)
        slice_0 = paddle._C_ops.slice(
            data_3, [0, 1], full_int_array_1, full_int_array_2, [1, 1], [0, 1]
        )
        del full_int_array_1, full_int_array_2

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [0, 1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [1, 2]

        # pd_op.slice: (xf32) <- (1x2xf32, 2xi64, 2xi64)
        slice_1 = paddle._C_ops.slice(
            data_3, [0, 1], full_int_array_3, full_int_array_4, [1, 1], [0, 1]
        )
        del data_3, full_int_array_3, full_int_array_4

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [0]

        # pd_op.unsqueeze: (1xf32) <- (xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(slice_0, full_int_array_5)
        del slice_0

        # pd_op.unsqueeze: (1xf32) <- (xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(slice_1, full_int_array_5)
        del slice_1

        # builtin.combine: ([1xf32, 1xf32, 1xf32, 1xf32]) <- (1xf32, 1xf32, 1xf32, 1xf32)
        combine_3 = [unsqueeze_1, unsqueeze_0, unsqueeze_1, unsqueeze_0]
        del unsqueeze_0, unsqueeze_1

        # pd_op.concat: (4xf32) <- ([1xf32, 1xf32, 1xf32, 1xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_3, full_0)
        del combine_3

        # pd_op.full: (xi64) <- ()
        full_3 = paddle._C_ops.full(
            [], float("4"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_4 = [reshape_0, full_3]
        del full_3, reshape_0

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_4, 0)
        del combine_4

        # pd_op.expand: (-1x4xf32) <- (4xf32, 2xi64)
        expand_1 = paddle._C_ops.expand(concat_3, stack_1)
        del concat_3, stack_1

        # builtin.combine: ([-1x2xf32]) <- (-1x2xf32)
        combine_5 = [expand_0]
        del expand_0

        # pd_op.concat: (-1x2xf32) <- ([-1x2xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_5, full_0)
        del combine_5

        # builtin.combine: ([-1x4xf32]) <- (-1x4xf32)
        combine_6 = [expand_1]
        del expand_1

        # pd_op.concat: (-1x4xf32) <- ([-1x4xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_6, full_0)
        del combine_6, full_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [1]

        # pd_op.slice: (-1x1xf32) <- (-1x6xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            concat_1, [1], full_int_array_5, full_int_array_6, [1], []
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [2]

        # pd_op.slice: (-1x1xf32) <- (-1x6xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            concat_1, [1], full_int_array_6, full_int_array_7, [1], []
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [2147483647]

        # pd_op.slice: (-1x4xf32) <- (-1x6xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            concat_1, [1], full_int_array_7, full_int_array_8, [1], []
        )
        del full_int_array_8

        # pd_op.divide: (-1x4xf32) <- (-1x4xf32, -1x4xf32)
        divide_1 = paddle._C_ops.divide(slice_4, concat_5)
        del concat_5, slice_4

        # pd_op.slice: (-1xf32) <- (-1x2xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            concat_4, [1], full_int_array_5, full_int_array_6, [1], [1]
        )

        # pd_op.slice: (-1xf32) <- (-1x2xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            concat_4, [1], full_int_array_6, full_int_array_7, [1], [1]
        )

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (-1xf32) <- (-1xf32, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            slice_5, full_4, paddle.float32, paddle.framework._current_expected_place()
        )
        del full_4

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            divide_1, [1], full_int_array_5, full_int_array_6, [1], [1]
        )

        # pd_op.minimum: (-1xf32) <- (-1xf32, -1xf32)
        minimum_0 = paddle._C_ops.minimum(slice_7, slice_6)
        del slice_7

        # pd_op.maximum: (-1xf32) <- (-1xf32, -1xf32)
        maximum_0 = paddle._C_ops.maximum(minimum_0, full_like_0)
        del minimum_0

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            divide_1, [1], full_int_array_6, full_int_array_7, [1], [1]
        )

        # pd_op.minimum: (-1xf32) <- (-1xf32, -1xf32)
        minimum_1 = paddle._C_ops.minimum(slice_8, slice_5)
        del slice_8

        # pd_op.maximum: (-1xf32) <- (-1xf32, -1xf32)
        maximum_1 = paddle._C_ops.maximum(minimum_1, full_like_0)
        del minimum_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [3]

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            divide_1, [1], full_int_array_7, full_int_array_9, [1], [1]
        )

        # pd_op.minimum: (-1xf32) <- (-1xf32, -1xf32)
        minimum_2 = paddle._C_ops.minimum(slice_9, slice_6)
        del slice_6, slice_9

        # pd_op.maximum: (-1xf32) <- (-1xf32, -1xf32)
        maximum_2 = paddle._C_ops.maximum(minimum_2, full_like_0)
        del minimum_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_10 = [4]

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            divide_1, [1], full_int_array_9, full_int_array_10, [1], [1]
        )
        del divide_1

        # pd_op.minimum: (-1xf32) <- (-1xf32, -1xf32)
        minimum_3 = paddle._C_ops.minimum(slice_10, slice_5)
        del slice_10, slice_5

        # pd_op.maximum: (-1xf32) <- (-1xf32, -1xf32)
        maximum_3 = paddle._C_ops.maximum(minimum_3, full_like_0)
        del full_like_0, minimum_3

        # builtin.combine: ([-1xf32, -1xf32, -1xf32, -1xf32]) <- (-1xf32, -1xf32, -1xf32, -1xf32)
        combine_7 = [maximum_0, maximum_1, maximum_2, maximum_3]
        del maximum_0, maximum_1, maximum_2, maximum_3

        # pd_op.stack: (-1x4xf32) <- ([-1xf32, -1xf32, -1xf32, -1xf32])
        stack_2 = paddle._C_ops.stack(combine_7, -1)
        del combine_7

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            stack_2, [1], full_int_array_7, full_int_array_9, [1], [1]
        )

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            stack_2, [1], full_int_array_5, full_int_array_6, [1], [1]
        )
        del full_int_array_5

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_0 = paddle._C_ops.subtract(slice_11, slice_12)
        del slice_11, slice_12

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            stack_2, [1], full_int_array_9, full_int_array_10, [1], [1]
        )
        del full_int_array_10, full_int_array_9

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            stack_2, [1], full_int_array_6, full_int_array_7, [1], [1]
        )
        del full_int_array_7

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_1 = paddle._C_ops.subtract(slice_13, slice_14)
        del slice_13, slice_14

        # pd_op.full: (xf32) <- ()
        full_5 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (-1xb) <- (-1xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(subtract_1, full_5)
        del subtract_1

        # pd_op.greater_than: (-1xb) <- (-1xf32, xf32)
        greater_than_1 = paddle._C_ops.greater_than(subtract_0, full_5)
        del full_5, subtract_0

        # pd_op.logical_and: (-1xb) <- (-1xb, -1xb)
        logical_and_0 = paddle._C_ops.logical_and(greater_than_0, greater_than_1)
        del greater_than_0, greater_than_1

        # pd_op.unsqueeze: (-1x1xb) <- (-1xb, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(logical_and_0, full_int_array_6)
        del full_int_array_6, logical_and_0

        # pd_op.full_like: (-1x1xf32) <- (-1x1xf32, 1xf32)
        full_like_1 = paddle._C_ops.full_like(
            slice_2, full_1, paddle.float32, paddle.framework._current_expected_place()
        )
        del full_1

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(full_like_1, full_6, float("0"), True)
        del full_6, full_like_1

        # pd_op.where: (-1x1xf32) <- (-1x1xb, -1x1xf32, -1x1xf32)
        where_0 = paddle._C_ops.where(unsqueeze_2, slice_2, scale_1)
        del scale_1, slice_2, unsqueeze_2

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x1xf32, -1x1xf32, -1x4xf32]) <- (-1x1xf32, -1x1xf32, -1x4xf32)
        combine_8 = [where_0, slice_3, stack_2]
        del slice_3, stack_2, where_0

        # pd_op.concat: (-1x6xf32) <- ([-1x1xf32, -1x1xf32, -1x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_8, full_7)
        del combine_8, concat_1, concat_2, concat_4, full_7

        return concat_0
