import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        parameter_0,
        parameter_1,
        parameter_2,
        parameter_3,
        parameter_4,
        parameter_5,
        parameter_6,
        parameter_7,
        parameter_8,
        parameter_9,
        parameter_10,
        parameter_11,
        parameter_12,
        parameter_13,
        parameter_14,
        parameter_15,
        parameter_16,
        parameter_17,
        parameter_18,
        parameter_19,
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
        # pd_op.shape64: (3xi64) <- (1x-1x256xf32)
        shape64_0 = paddle._C_ops.shape64(data_6)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [2]

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.cast: (xf32) <- (xi64)
        cast_1 = paddle._C_ops.cast(data_0, paddle.float32)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.arange: (-1xf32) <- (1xf32, xf32, 1xf32)
        arange_0 = paddle.arange(full_0, cast_1, full_1, dtype="float32")

        # pd_op.cast: (xf32) <- (xi64)
        cast_2 = paddle._C_ops.cast(data_1, paddle.float32)

        # pd_op.arange: (-1xf32) <- (1xf32, xf32, 1xf32)
        arange_1 = paddle.arange(full_0, cast_2, full_1, dtype="float32")

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_0 = [arange_0, arange_1]

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_0)

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        (
            split_0,
            split_1,
        ) = meshgrid_0

        # builtin.combine: ([-1x-1xf32, -1x-1xf32]) <- (-1x-1xf32, -1x-1xf32)
        combine_1 = [split_1, split_0]

        # pd_op.stack: (-1x-1x2xf32) <- ([-1x-1xf32, -1x-1xf32])
        stack_1 = paddle._C_ops.stack(combine_1, -1)

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_2 = [data_0, data_1]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_2 = paddle._C_ops.stack(combine_2, 0)

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_2 = stack_2

        # pd_op.cast: (2xf32) <- (2xi64)
        cast_3 = paddle._C_ops.cast(assign_2, paddle.float32)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.unsqueeze: (1x-1x-1x2xf32) <- (-1x-1x2xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(stack_1, full_int_array_2)

        # pd_op.scale: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(unsqueeze_0, full_1, float("0.5"), True)

        # pd_op.divide: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 2xf32)
        divide_0 = paddle._C_ops.divide(scale_0, cast_3)

        # pd_op.full_like: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            divide_0, full_1, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0.05"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(full_like_0, full_2, float("0"), True)

        # pd_op.scale: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(scale_1, full_1, float("0"), True)

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([1x-1x-1x2xf32, 1x-1x-1x2xf32]) <- (1x-1x-1x2xf32, 1x-1x-1x2xf32)
        combine_3 = [divide_0, scale_2]

        # pd_op.concat: (1x-1x-1x4xf32) <- ([1x-1x-1x2xf32, 1x-1x-1x2xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_3, full_3)

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_0 = paddle._C_ops.multiply(data_0, data_1)

        # pd_op.full: (xi64) <- ()
        full_4 = paddle._C_ops.full(
            [], float("-1"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_5 = paddle._C_ops.full(
            [], float("4"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_4 = [full_4, multiply_0, full_5]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_3 = paddle._C_ops.stack(combine_4, 0)

        # pd_op.reshape: (-1x-1x4xf32) <- (1x-1x-1x4xf32, 3xi64)
        reshape_2 = paddle._C_ops.reshape(concat_1, stack_3)

        # pd_op.cast: (xf32) <- (xi64)
        cast_4 = paddle._C_ops.cast(data_2, paddle.float32)

        # pd_op.arange: (-1xf32) <- (1xf32, xf32, 1xf32)
        arange_2 = paddle.arange(full_0, cast_4, full_1, dtype="float32")

        # pd_op.cast: (xf32) <- (xi64)
        cast_5 = paddle._C_ops.cast(data_3, paddle.float32)

        # pd_op.arange: (-1xf32) <- (1xf32, xf32, 1xf32)
        arange_3 = paddle.arange(full_0, cast_5, full_1, dtype="float32")

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_5 = [arange_2, arange_3]

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_1 = paddle._C_ops.meshgrid(combine_5)

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        (
            split_2,
            split_3,
        ) = meshgrid_1

        # builtin.combine: ([-1x-1xf32, -1x-1xf32]) <- (-1x-1xf32, -1x-1xf32)
        combine_6 = [split_3, split_2]

        # pd_op.stack: (-1x-1x2xf32) <- ([-1x-1xf32, -1x-1xf32])
        stack_4 = paddle._C_ops.stack(combine_6, -1)

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_7 = [data_2, data_3]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_5 = paddle._C_ops.stack(combine_7, 0)

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_3 = stack_5

        # pd_op.cast: (2xf32) <- (2xi64)
        cast_6 = paddle._C_ops.cast(assign_3, paddle.float32)

        # pd_op.unsqueeze: (1x-1x-1x2xf32) <- (-1x-1x2xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(stack_4, full_int_array_2)

        # pd_op.scale: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(unsqueeze_1, full_1, float("0.5"), True)

        # pd_op.divide: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 2xf32)
        divide_1 = paddle._C_ops.divide(scale_3, cast_6)

        # pd_op.full_like: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        full_like_1 = paddle._C_ops.full_like(
            divide_1, full_1, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.scale: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(full_like_1, full_2, float("0"), True)

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(scale_4, full_6, float("0"), True)

        # builtin.combine: ([1x-1x-1x2xf32, 1x-1x-1x2xf32]) <- (1x-1x-1x2xf32, 1x-1x-1x2xf32)
        combine_8 = [divide_1, scale_5]

        # pd_op.concat: (1x-1x-1x4xf32) <- ([1x-1x-1x2xf32, 1x-1x-1x2xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_8, full_3)

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_1 = paddle._C_ops.multiply(data_2, data_3)

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_9 = [full_4, multiply_1, full_5]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_6 = paddle._C_ops.stack(combine_9, 0)

        # pd_op.reshape: (-1x-1x4xf32) <- (1x-1x-1x4xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(concat_2, stack_6)

        # pd_op.cast: (xf32) <- (xi64)
        cast_7 = paddle._C_ops.cast(data_4, paddle.float32)

        # pd_op.arange: (-1xf32) <- (1xf32, xf32, 1xf32)
        arange_4 = paddle.arange(full_0, cast_7, full_1, dtype="float32")

        # pd_op.cast: (xf32) <- (xi64)
        cast_8 = paddle._C_ops.cast(data_5, paddle.float32)

        # pd_op.arange: (-1xf32) <- (1xf32, xf32, 1xf32)
        arange_5 = paddle.arange(full_0, cast_8, full_1, dtype="float32")

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_10 = [arange_4, arange_5]

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_2 = paddle._C_ops.meshgrid(combine_10)

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        (
            split_4,
            split_5,
        ) = meshgrid_2

        # builtin.combine: ([-1x-1xf32, -1x-1xf32]) <- (-1x-1xf32, -1x-1xf32)
        combine_11 = [split_5, split_4]

        # pd_op.stack: (-1x-1x2xf32) <- ([-1x-1xf32, -1x-1xf32])
        stack_7 = paddle._C_ops.stack(combine_11, -1)

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_12 = [data_4, data_5]

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_8 = paddle._C_ops.stack(combine_12, 0)

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_4 = stack_8

        # pd_op.cast: (2xf32) <- (2xi64)
        cast_9 = paddle._C_ops.cast(assign_4, paddle.float32)

        # pd_op.unsqueeze: (1x-1x-1x2xf32) <- (-1x-1x2xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(stack_7, full_int_array_2)

        # pd_op.scale: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(unsqueeze_2, full_1, float("0.5"), True)

        # pd_op.divide: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 2xf32)
        divide_2 = paddle._C_ops.divide(scale_6, cast_9)

        # pd_op.full_like: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        full_like_2 = paddle._C_ops.full_like(
            divide_2, full_1, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.scale: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(full_like_2, full_2, float("0"), True)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("4"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(scale_7, full_7, float("0"), True)

        # builtin.combine: ([1x-1x-1x2xf32, 1x-1x-1x2xf32]) <- (1x-1x-1x2xf32, 1x-1x-1x2xf32)
        combine_13 = [divide_2, scale_8]

        # pd_op.concat: (1x-1x-1x4xf32) <- ([1x-1x-1x2xf32, 1x-1x-1x2xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_13, full_3)

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_2 = paddle._C_ops.multiply(data_4, data_5)

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_14 = [full_4, multiply_2, full_5]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_9 = paddle._C_ops.stack(combine_14, 0)

        # pd_op.reshape: (-1x-1x4xf32) <- (1x-1x-1x4xf32, 3xi64)
        reshape_4 = paddle._C_ops.reshape(concat_3, stack_9)

        # pd_op.full: (1xi32) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_0 = full_8

        # builtin.combine: ([-1x-1x4xf32, -1x-1x4xf32, -1x-1x4xf32]) <- (-1x-1x4xf32, -1x-1x4xf32, -1x-1x4xf32)
        combine_15 = [reshape_2, reshape_3, reshape_4]

        # pd_op.concat: (-1x-1x4xf32) <- ([-1x-1x4xf32, -1x-1x4xf32, -1x-1x4xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_15, full_8)

        # pd_op.full: (xf32) <- ()
        full_9 = paddle._C_ops.full(
            [],
            float("0.01"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.greater_than: (-1x-1x4xb) <- (-1x-1x4xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(concat_4, full_9)

        # pd_op.full: (xf32) <- ()
        full_10 = paddle._C_ops.full(
            [],
            float("0.99"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.less_than: (-1x-1x4xb) <- (-1x-1x4xf32, xf32)
        less_than_0 = paddle._C_ops.less_than(concat_4, full_10)

        # pd_op.multiply: (-1x-1x4xb) <- (-1x-1x4xb, -1x-1x4xb)
        multiply_3 = paddle._C_ops.multiply(greater_than_0, less_than_0)

        # pd_op.all: (-1x-1x1xb) <- (-1x-1x4xb)
        all_0 = paddle._C_ops.all(multiply_3, [-1], True)

        # pd_op.full: (1xf32) <- ()
        full_11 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x-1x4xf32) <- (-1x-1x4xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(concat_4, full_11, float("1"), True)

        # pd_op.divide: (-1x-1x4xf32) <- (-1x-1x4xf32, -1x-1x4xf32)
        divide_3 = paddle._C_ops.divide(concat_4, scale_9)

        # pd_op.log: (-1x-1x4xf32) <- (-1x-1x4xf32)
        log_0 = paddle._C_ops.log(divide_3)

        # pd_op.full: (xf32) <- ()
        full_12 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__1 = paddle._C_ops.assign_value_(
            full_12,
            [],
            paddle.float32,
            [float("inf")],
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_like: (-1x-1x4xf32) <- (-1x-1x4xf32, 1xf32)
        full_like_3 = paddle._C_ops.full_like(
            log_0, full_0, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.full_like: (xf32) <- (xf32, 1xf32)
        full_like_4 = paddle._C_ops.full_like(
            assign_value__1,
            full_0,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_like: (-1x-1x1xb) <- (-1x-1x1xb, 1xf32)
        full_like_5 = paddle._C_ops.full_like(
            all_0, full_0, paddle.bool, paddle.framework._current_expected_place()
        )

        # pd_op.cast: (-1x-1x1xf32) <- (-1x-1x1xb)
        cast_10 = paddle._C_ops.cast(full_like_5, paddle.float32)

        # pd_op.cast: (-1x-1x1xf32) <- (-1x-1x1xb)
        cast_11 = paddle._C_ops.cast(all_0, paddle.float32)

        # pd_op.add: (-1x-1x4xf32) <- (-1x-1x4xf32, xf32)
        add_10 = paddle._C_ops.add(full_like_3, full_like_4)

        # pd_op.add: (-1x-1x4xf32) <- (-1x-1x4xf32, -1x-1x1xf32)
        add_11 = paddle._C_ops.add(add_10, cast_10)

        # pd_op.add: (-1x-1x4xf32) <- (-1x-1x4xf32, -1x-1x4xf32)
        add_12 = paddle._C_ops.add(log_0, add_11)

        # pd_op.add: (-1x-1x4xf32) <- (xf32, -1x-1x4xf32)
        add_13 = paddle._C_ops.add(assign_value__1, add_11)

        # pd_op.add: (-1x-1x4xf32) <- (-1x-1x1xf32, -1x-1x4xf32)
        add_14 = paddle._C_ops.add(cast_11, add_11)

        # pd_op.cast: (-1x-1x4xb) <- (-1x-1x4xf32)
        cast_12 = paddle._C_ops.cast(add_14, paddle.bool)

        # pd_op.where: (-1x-1x4xf32) <- (-1x-1x4xb, -1x-1x4xf32, -1x-1x4xf32)
        where_0 = paddle._C_ops.where(cast_12, add_12, add_13)

        # pd_op.full: (xf32) <- ()
        full_13 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_13,
            [],
            paddle.float32,
            [float("0")],
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_like: (1x-1x256xf32) <- (1x-1x256xf32, 1xf32)
        full_like_6 = paddle._C_ops.full_like(
            data_6, full_0, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.full_like: (xf32) <- (xf32, 1xf32)
        full_like_7 = paddle._C_ops.full_like(
            assign_value__0,
            full_0,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (1x-1x256xf32) <- (1x-1x256xf32, xf32)
        add_15 = paddle._C_ops.add(full_like_6, full_like_7)

        # pd_op.add: (-1x-1x256xf32) <- (1x-1x256xf32, -1x-1x1xf32)
        add_0 = paddle._C_ops.add(add_15, cast_10)

        # pd_op.add: (-1x-1x256xf32) <- (1x-1x256xf32, -1x-1x256xf32)
        add_1 = paddle._C_ops.add(data_6, add_0)

        # pd_op.add: (-1x-1x256xf32) <- (xf32, -1x-1x256xf32)
        add_2 = paddle._C_ops.add(assign_value__0, add_0)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x1xf32, -1x-1x256xf32)
        add_16 = paddle._C_ops.add(cast_11, add_0)

        # pd_op.cast: (-1x-1x256xb) <- (-1x-1x256xf32)
        cast_0 = paddle._C_ops.cast(add_16, paddle.bool)

        # pd_op.where: (-1x-1x256xf32) <- (-1x-1x256xb, -1x-1x256xf32, -1x-1x256xf32)
        where_1 = paddle._C_ops.where(cast_0, add_1, add_2)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x256xf32, 256x256xf32)
        matmul_0 = paddle._C_ops.matmul(where_1, parameter_19, False, False)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_3 = paddle._C_ops.add(matmul_0, parameter_18)

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_3, parameter_17, parameter_16, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x-1x2xf32) <- (-1x-1x256xf32, 256x2xf32)
        matmul_12 = paddle._C_ops.matmul(layer_norm_0, parameter_15, False, False)

        # pd_op.add: (-1x-1x2xf32) <- (-1x-1x2xf32, 2xf32)
        add_17 = paddle._C_ops.add(matmul_12, parameter_14)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x256xf32, 256x256xf32)
        matmul_1 = paddle._C_ops.matmul(layer_norm_0, parameter_13, False, False)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_18 = paddle._C_ops.add(matmul_1, parameter_12)

        # pd_op.relu: (-1x-1x256xf32) <- (-1x-1x256xf32)
        relu_0 = paddle._C_ops.relu(add_18)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x256xf32, 256x256xf32)
        matmul_2 = paddle._C_ops.matmul(relu_0, parameter_11, False, False)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_19 = paddle._C_ops.add(matmul_2, parameter_10)

        # pd_op.relu: (-1x-1x256xf32) <- (-1x-1x256xf32)
        relu_1 = paddle._C_ops.relu(add_19)

        # pd_op.matmul: (-1x-1x4xf32) <- (-1x-1x256xf32, 256x4xf32)
        matmul_3 = paddle._C_ops.matmul(relu_1, parameter_9, False, False)

        # pd_op.add: (-1x-1x4xf32) <- (-1x-1x4xf32, 4xf32)
        add_4 = paddle._C_ops.add(matmul_3, parameter_8)

        # pd_op.add: (-1x-1x4xf32) <- (-1x-1x4xf32, -1x-1x4xf32)
        add_5 = paddle._C_ops.add(add_4, where_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [-1]

        # pd_op.max: (-1x-1xf32) <- (-1x-1x2xf32, 1xi64)
        max_0 = paddle._C_ops.max(add_17, full_int_array_3, False)

        # pd_op.full: (1xi32) <- ()
        full_14 = paddle._C_ops.full(
            [1], float("300"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.topk: (-1x300xf32, -1x300xi64) <- (-1x-1xf32, 1xi32)
        topk_0, topk_1 = (lambda x, f: f(x))(
            paddle._C_ops.topk(max_0, full_14, 1, True, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.full: (1xf64) <- ()
        full_15 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_16 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (1xi64) <- (1xf64, 1xf64, 1xf64)
        arange_6 = paddle.arange(full_15, full_16, full_16, dtype="int64")

        # pd_op.unsqueeze: (1x1xi64) <- (1xi64, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(arange_6, full_int_array_3)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [1, 300]

        # pd_op.tile: (1x300xi64) <- (1x1xi64, 2xi64)
        tile_0 = paddle._C_ops.tile(unsqueeze_3, full_int_array_4)

        # builtin.combine: ([1x300xi64, -1x300xi64]) <- (1x300xi64, -1x300xi64)
        combine_16 = [tile_0, topk_1]

        # pd_op.stack: (1x300x2xi64) <- ([1x300xi64, -1x300xi64])
        stack_0 = paddle._C_ops.stack(combine_16, -1)

        # pd_op.gather_nd: (1x300x256xf32) <- (-1x-1x256xf32, 1x300x2xi64)
        gather_nd_0 = paddle._C_ops.gather_nd(layer_norm_0, stack_0)

        # pd_op.gather_nd: (1x300x4xf32) <- (-1x-1x4xf32, 1x300x2xi64)
        gather_nd_1 = paddle._C_ops.gather_nd(add_5, stack_0)

        # pd_op.layer_norm: (1x300x256xf32, 1x300xf32, 1x300xf32) <- (1x300x256xf32, 256xf32, 256xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                gather_nd_0, parameter_7, parameter_6, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (1x300x2xf32) <- (1x300x256xf32, 256x2xf32)
        matmul_4 = paddle._C_ops.matmul(layer_norm_3, parameter_15, False, False)

        # pd_op.add: (1x300x2xf32) <- (1x300x2xf32, 2xf32)
        add_8 = paddle._C_ops.add(matmul_4, parameter_14)

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_5 = paddle._C_ops.matmul(layer_norm_3, parameter_5, False, False)

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_20 = paddle._C_ops.add(matmul_5, parameter_4)

        # pd_op.relu: (1x300x256xf32) <- (1x300x256xf32)
        relu_2 = paddle._C_ops.relu(add_20)

        # pd_op.matmul: (1x300x256xf32) <- (1x300x256xf32, 256x256xf32)
        matmul_6 = paddle._C_ops.matmul(relu_2, parameter_3, False, False)

        # pd_op.add: (1x300x256xf32) <- (1x300x256xf32, 256xf32)
        add_21 = paddle._C_ops.add(matmul_6, parameter_2)

        # pd_op.relu: (1x300x256xf32) <- (1x300x256xf32)
        relu_3 = paddle._C_ops.relu(add_21)

        # pd_op.matmul: (1x300x128xf32) <- (1x300x256xf32, 256x128xf32)
        matmul_7 = paddle._C_ops.matmul(relu_3, parameter_1, False, False)

        # pd_op.add: (1x300x128xf32) <- (1x300x128xf32, 128xf32)
        add_6 = paddle._C_ops.add(matmul_7, parameter_0)

        # pd_op.shape64: (4xi64) <- (1x128x-1x-1xf32)
        shape64_1 = paddle._C_ops.shape64(data_7)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [3]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_1, full_int_array_5, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [4]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_5, full_int_array_6, [1], [0]
        )

        # pd_op.flatten: (1x128x-1xf32) <- (1x128x-1x-1xf32)
        flatten_0 = paddle._C_ops.flatten(data_7, 2, 3)

        # pd_op.assign: (1x128x-1xf32) <- (1x128x-1xf32)
        assign_1 = flatten_0

        # pd_op.bmm: (1x300x-1xf32) <- (1x300x128xf32, 1x128x-1xf32)
        bmm_0 = paddle._C_ops.bmm(add_6, flatten_0)

        # pd_op.full: (xi64) <- ()
        full_17 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_18 = paddle._C_ops.full(
            [], float("300"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_17 = [full_17, full_18, slice_1, slice_2]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_10 = paddle._C_ops.stack(combine_17, 0)

        # pd_op.reshape: (1x300x-1x-1xf32) <- (1x300x-1xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(bmm_0, stack_10)

        # pd_op.sigmoid: (1x300x4xf32) <- (1x300x4xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(gather_nd_1)

        # pd_op.share_data_: (1x300x256xf32) <- (1x300x256xf32)
        share_data__0 = gather_nd_0.detach()

        # builtin.combine: ([1x100x256xf32, 1x300x256xf32]) <- (1x100x256xf32, 1x300x256xf32)
        combine_18 = [data_8, share_data__0]

        # pd_op.concat: (1x400x256xf32) <- ([1x100x256xf32, 1x300x256xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_18, full_8)

        # pd_op.full: (xf32) <- ()
        full_19 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (1x300x-1x-1xb) <- (1x300x-1x-1xf32, xf32)
        greater_than_1 = paddle._C_ops.greater_than(reshape_0, full_19)

        # pd_op.shape64: (4xi64) <- (1x300x-1x-1xb)
        shape64_2 = paddle._C_ops.shape64(greater_than_1)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_1, full_int_array_5, [1], [0]
        )

        # pd_op.shape64: (4xi64) <- (1x300x-1x-1xb)
        shape64_3 = paddle._C_ops.shape64(greater_than_1)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_5, full_int_array_6, [1], [0]
        )

        # pd_op.cast: (xf32) <- (xi64)
        cast_13 = paddle._C_ops.cast(slice_3, paddle.float32)

        # pd_op.arange: (-1xf32) <- (1xf32, xf32, 1xf32)
        arange_7 = paddle.arange(full_0, cast_13, full_1, dtype="float32")

        # pd_op.cast: (xf32) <- (xi64)
        cast_14 = paddle._C_ops.cast(slice_4, paddle.float32)

        # pd_op.arange: (-1xf32) <- (1xf32, xf32, 1xf32)
        arange_8 = paddle.arange(full_0, cast_14, full_1, dtype="float32")

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_19 = [arange_7, arange_8]

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_3 = paddle._C_ops.meshgrid(combine_19)

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        (
            split_6,
            split_7,
        ) = meshgrid_3

        # pd_op.cast: (1x300x-1x-1xf32) <- (1x300x-1x-1xb)
        cast_15 = paddle._C_ops.cast(greater_than_1, paddle.float32)

        # pd_op.multiply: (1x300x-1x-1xf32) <- (-1x-1xf32, 1x300x-1x-1xf32)
        multiply_4 = paddle._C_ops.multiply(split_7, cast_15)

        # pd_op.flatten: (1x300x-1xf32) <- (1x300x-1x-1xf32)
        flatten_1 = paddle._C_ops.flatten(multiply_4, 2, 3)

        # pd_op.max: (1x300xf32) <- (1x300x-1xf32, 1xi64)
        max_1 = paddle._C_ops.max(flatten_1, full_int_array_3, False)

        # pd_op.scale: (1x300xf32) <- (1x300xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(max_1, full_1, float("1"), True)

        # pd_op.full: (xf32) <- ()
        full_20 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__2 = paddle._C_ops.assign_value_(
            full_20,
            [],
            paddle.float32,
            [float("1e+08")],
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_like: (1x300x-1x-1xf32) <- (1x300x-1x-1xf32, 1xf32)
        full_like_8 = paddle._C_ops.full_like(
            multiply_4,
            full_0,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_like: (xf32) <- (xf32, 1xf32)
        full_like_9 = paddle._C_ops.full_like(
            assign_value__2,
            full_0,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_like: (1x300x-1x-1xb) <- (1x300x-1x-1xb, 1xf32)
        full_like_10 = paddle._C_ops.full_like(
            greater_than_1,
            full_0,
            paddle.bool,
            paddle.framework._current_expected_place(),
        )

        # pd_op.cast: (1x300x-1x-1xf32) <- (1x300x-1x-1xb)
        cast_16 = paddle._C_ops.cast(full_like_10, paddle.float32)

        # pd_op.cast: (1x300x-1x-1xf32) <- (1x300x-1x-1xb)
        cast_17 = paddle._C_ops.cast(greater_than_1, paddle.float32)

        # pd_op.add: (1x300x-1x-1xf32) <- (1x300x-1x-1xf32, xf32)
        add_22 = paddle._C_ops.add(full_like_8, full_like_9)

        # pd_op.add: (1x300x-1x-1xf32) <- (1x300x-1x-1xf32, 1x300x-1x-1xf32)
        add_23 = paddle._C_ops.add(add_22, cast_16)

        # pd_op.add: (1x300x-1x-1xf32) <- (1x300x-1x-1xf32, 1x300x-1x-1xf32)
        add_24 = paddle._C_ops.add(multiply_4, add_23)

        # pd_op.add: (1x300x-1x-1xf32) <- (xf32, 1x300x-1x-1xf32)
        add_25 = paddle._C_ops.add(assign_value__2, add_23)

        # pd_op.add: (1x300x-1x-1xf32) <- (1x300x-1x-1xf32, 1x300x-1x-1xf32)
        add_26 = paddle._C_ops.add(cast_17, add_23)

        # pd_op.cast: (1x300x-1x-1xb) <- (1x300x-1x-1xf32)
        cast_18 = paddle._C_ops.cast(add_26, paddle.bool)

        # pd_op.where: (1x300x-1x-1xf32) <- (1x300x-1x-1xb, 1x300x-1x-1xf32, 1x300x-1x-1xf32)
        where_2 = paddle._C_ops.where(cast_18, add_24, add_25)

        # pd_op.flatten: (1x300x-1xf32) <- (1x300x-1x-1xf32)
        flatten_2 = paddle._C_ops.flatten(where_2, 2, 3)

        # pd_op.min: (1x300xf32) <- (1x300x-1xf32, 1xi64)
        min_0 = paddle._C_ops.min(flatten_2, full_int_array_3, False)

        # pd_op.cast: (1x300x-1x-1xf32) <- (1x300x-1x-1xb)
        cast_19 = paddle._C_ops.cast(greater_than_1, paddle.float32)

        # pd_op.multiply: (1x300x-1x-1xf32) <- (-1x-1xf32, 1x300x-1x-1xf32)
        multiply_5 = paddle._C_ops.multiply(split_6, cast_19)

        # pd_op.flatten: (1x300x-1xf32) <- (1x300x-1x-1xf32)
        flatten_3 = paddle._C_ops.flatten(multiply_5, 2, 3)

        # pd_op.max: (1x300xf32) <- (1x300x-1xf32, 1xi64)
        max_2 = paddle._C_ops.max(flatten_3, full_int_array_3, False)

        # pd_op.scale: (1x300xf32) <- (1x300xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(max_2, full_1, float("1"), True)

        # pd_op.full: (xf32) <- ()
        full_21 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__3 = paddle._C_ops.assign_value_(
            full_21,
            [],
            paddle.float32,
            [float("1e+08")],
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_like: (1x300x-1x-1xf32) <- (1x300x-1x-1xf32, 1xf32)
        full_like_11 = paddle._C_ops.full_like(
            multiply_5,
            full_0,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_like: (xf32) <- (xf32, 1xf32)
        full_like_12 = paddle._C_ops.full_like(
            assign_value__3,
            full_0,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_like: (1x300x-1x-1xb) <- (1x300x-1x-1xb, 1xf32)
        full_like_13 = paddle._C_ops.full_like(
            greater_than_1,
            full_0,
            paddle.bool,
            paddle.framework._current_expected_place(),
        )

        # pd_op.cast: (1x300x-1x-1xf32) <- (1x300x-1x-1xb)
        cast_20 = paddle._C_ops.cast(full_like_13, paddle.float32)

        # pd_op.cast: (1x300x-1x-1xf32) <- (1x300x-1x-1xb)
        cast_21 = paddle._C_ops.cast(greater_than_1, paddle.float32)

        # pd_op.add: (1x300x-1x-1xf32) <- (1x300x-1x-1xf32, xf32)
        add_27 = paddle._C_ops.add(full_like_11, full_like_12)

        # pd_op.add: (1x300x-1x-1xf32) <- (1x300x-1x-1xf32, 1x300x-1x-1xf32)
        add_28 = paddle._C_ops.add(add_27, cast_20)

        # pd_op.add: (1x300x-1x-1xf32) <- (1x300x-1x-1xf32, 1x300x-1x-1xf32)
        add_29 = paddle._C_ops.add(multiply_5, add_28)

        # pd_op.add: (1x300x-1x-1xf32) <- (xf32, 1x300x-1x-1xf32)
        add_30 = paddle._C_ops.add(assign_value__3, add_28)

        # pd_op.add: (1x300x-1x-1xf32) <- (1x300x-1x-1xf32, 1x300x-1x-1xf32)
        add_31 = paddle._C_ops.add(cast_21, add_28)

        # pd_op.cast: (1x300x-1x-1xb) <- (1x300x-1x-1xf32)
        cast_22 = paddle._C_ops.cast(add_31, paddle.bool)

        # pd_op.where: (1x300x-1x-1xf32) <- (1x300x-1x-1xb, 1x300x-1x-1xf32, 1x300x-1x-1xf32)
        where_3 = paddle._C_ops.where(cast_22, add_29, add_30)

        # pd_op.flatten: (1x300x-1xf32) <- (1x300x-1x-1xf32)
        flatten_4 = paddle._C_ops.flatten(where_3, 2, 3)

        # pd_op.min: (1x300xf32) <- (1x300x-1xf32, 1xi64)
        min_1 = paddle._C_ops.min(flatten_4, full_int_array_3, False)

        # builtin.combine: ([1x300xf32, 1x300xf32, 1x300xf32, 1x300xf32]) <- (1x300xf32, 1x300xf32, 1x300xf32, 1x300xf32)
        combine_20 = [min_0, min_1, scale_10, scale_11]

        # pd_op.stack: (1x300x4xf32) <- ([1x300xf32, 1x300xf32, 1x300xf32, 1x300xf32])
        stack_11 = paddle._C_ops.stack(combine_20, -1)

        # pd_op.any: (1x300xb) <- (1x300x-1x-1xb)
        any_0 = paddle._C_ops.any(greater_than_1, [2, 3], False)

        # pd_op.unsqueeze: (1x300x1xb) <- (1x300xb, 1xi64)
        unsqueeze_4 = paddle._C_ops.unsqueeze(any_0, full_int_array_1)

        # pd_op.cast: (1x300x1xf32) <- (1x300x1xb)
        cast_23 = paddle._C_ops.cast(unsqueeze_4, paddle.float32)

        # pd_op.multiply: (1x300x4xf32) <- (1x300x4xf32, 1x300x1xf32)
        multiply_6 = paddle._C_ops.multiply(stack_11, cast_23)

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_21 = [slice_4, slice_3, slice_4, slice_3]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_12 = paddle._C_ops.stack(combine_21, 0)

        # pd_op.assign: (4xi64) <- (4xi64)
        assign_5 = stack_12

        # pd_op.cast: (4xf32) <- (4xi64)
        cast_24 = paddle._C_ops.cast(assign_5, paddle.float32)

        # pd_op.divide: (1x300x4xf32) <- (1x300x4xf32, 4xf32)
        divide_4 = paddle._C_ops.divide(multiply_6, cast_24)

        # pd_op.full: (1xi32) <- ()
        full_22 = paddle._C_ops.full(
            [1], float("2"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split_with_num: ([1x300x1xf32, 1x300x1xf32, 1x300x1xf32, 1x300x1xf32]) <- (1x300x4xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(divide_4, 4, full_22)

        # builtin.split: (1x300x1xf32, 1x300x1xf32, 1x300x1xf32, 1x300x1xf32) <- ([1x300x1xf32, 1x300x1xf32, 1x300x1xf32, 1x300x1xf32])
        (
            split_8,
            split_9,
            split_10,
            split_11,
        ) = split_with_num_0

        # pd_op.add: (1x300x1xf32) <- (1x300x1xf32, 1x300x1xf32)
        add_32 = paddle._C_ops.add(split_8, split_10)

        # pd_op.full: (1xf32) <- ()
        full_23 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x300x1xf32) <- (1x300x1xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(add_32, full_23, float("0"), True)

        # pd_op.add: (1x300x1xf32) <- (1x300x1xf32, 1x300x1xf32)
        add_33 = paddle._C_ops.add(split_9, split_11)

        # pd_op.scale: (1x300x1xf32) <- (1x300x1xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(add_33, full_23, float("0"), True)

        # pd_op.subtract: (1x300x1xf32) <- (1x300x1xf32, 1x300x1xf32)
        subtract_0 = paddle._C_ops.subtract(split_10, split_8)

        # pd_op.subtract: (1x300x1xf32) <- (1x300x1xf32, 1x300x1xf32)
        subtract_1 = paddle._C_ops.subtract(split_11, split_9)

        # builtin.combine: ([1x300x1xf32, 1x300x1xf32, 1x300x1xf32, 1x300x1xf32]) <- (1x300x1xf32, 1x300x1xf32, 1x300x1xf32, 1x300x1xf32)
        combine_22 = [scale_12, scale_13, subtract_0, subtract_1]

        # pd_op.concat: (1x300x4xf32) <- ([1x300x1xf32, 1x300x1xf32, 1x300x1xf32, 1x300x1xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_22, full_3)

        # pd_op.clip: (1x300x4xf32) <- (1x300x4xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(concat_5, full_0, full_1)

        # pd_op.full: (1xf32) <- ()
        full_24 = paddle._C_ops.full(
            [1], float("1e-05"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_25 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (1x300x4xf32) <- (1x300x4xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(clip_0, full_24, full_25)

        # pd_op.scale: (1x300x4xf32) <- (1x300x4xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(clip_0, full_11, float("1"), True)

        # pd_op.clip: (1x300x4xf32) <- (1x300x4xf32, 1xf32, 1xf32)
        clip_2 = paddle._C_ops.clip(scale_14, full_24, full_25)

        # pd_op.divide: (1x300x4xf32) <- (1x300x4xf32, 1x300x4xf32)
        divide_5 = paddle._C_ops.divide(clip_1, clip_2)

        # pd_op.log: (1x300x4xf32) <- (1x300x4xf32)
        log_1 = paddle._C_ops.log(divide_5)

        # builtin.combine: ([1x100x4xf32, 1x300x4xf32]) <- (1x100x4xf32, 1x300x4xf32)
        combine_23 = [data_9, log_1]

        # pd_op.concat: (1x400x4xf32) <- ([1x100x4xf32, 1x300x4xf32], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_23, full_8)

        # pd_op.layer_norm: (1x400x256xf32, 1x400xf32, 1x400xf32) <- (1x400x256xf32, 256xf32, 256xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                concat_0, parameter_7, parameter_6, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (1x400x2xf32) <- (1x400x256xf32, 256x2xf32)
        matmul_8 = paddle._C_ops.matmul(layer_norm_6, parameter_15, False, False)

        # pd_op.add: (1x400x2xf32) <- (1x400x2xf32, 2xf32)
        add_9 = paddle._C_ops.add(matmul_8, parameter_14)

        # pd_op.matmul: (1x400x256xf32) <- (1x400x256xf32, 256x256xf32)
        matmul_9 = paddle._C_ops.matmul(layer_norm_6, parameter_5, False, False)

        # pd_op.add: (1x400x256xf32) <- (1x400x256xf32, 256xf32)
        add_34 = paddle._C_ops.add(matmul_9, parameter_4)

        # pd_op.relu: (1x400x256xf32) <- (1x400x256xf32)
        relu_4 = paddle._C_ops.relu(add_34)

        # pd_op.matmul: (1x400x256xf32) <- (1x400x256xf32, 256x256xf32)
        matmul_10 = paddle._C_ops.matmul(relu_4, parameter_3, False, False)

        # pd_op.add: (1x400x256xf32) <- (1x400x256xf32, 256xf32)
        add_35 = paddle._C_ops.add(matmul_10, parameter_2)

        # pd_op.relu: (1x400x256xf32) <- (1x400x256xf32)
        relu_5 = paddle._C_ops.relu(add_35)

        # pd_op.matmul: (1x400x128xf32) <- (1x400x256xf32, 256x128xf32)
        matmul_11 = paddle._C_ops.matmul(relu_5, parameter_1, False, False)

        # pd_op.add: (1x400x128xf32) <- (1x400x128xf32, 128xf32)
        add_7 = paddle._C_ops.add(matmul_11, parameter_0)

        # pd_op.bmm: (1x400x-1xf32) <- (1x400x128xf32, 1x128x-1xf32)
        bmm_1 = paddle._C_ops.bmm(add_7, flatten_0)

        # pd_op.full: (xi64) <- ()
        full_26 = paddle._C_ops.full(
            [], float("400"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_24 = [full_17, full_26, slice_1, slice_2]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_13 = paddle._C_ops.stack(combine_24, 0)

        # pd_op.reshape: (1x400x-1x-1xf32) <- (1x400x-1xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(bmm_1, stack_13)

        # pd_op.sigmoid: (1x400x4xf32) <- (1x400x4xf32)
        sigmoid_1 = paddle._C_ops.sigmoid(concat_6)

        # pd_op.share_data_: (1x400x4xf32) <- (1x400x4xf32)
        share_data__1 = concat_6.detach()
        return (
            where_0,
            assign_value__0,
            add_0,
            add_1,
            add_2,
            cast_0,
            where_1,
            matmul_0,
            add_3,
            layer_norm_0,
            layer_norm_1,
            layer_norm_2,
            matmul_1,
            relu_0,
            matmul_2,
            relu_1,
            matmul_3,
            add_4,
            add_5,
            stack_0,
            gather_nd_0,
            layer_norm_3,
            layer_norm_4,
            layer_norm_5,
            matmul_4,
            matmul_5,
            relu_2,
            matmul_6,
            relu_3,
            matmul_7,
            add_6,
            flatten_0,
            bmm_0,
            share_data__0,
            assign_0,
            layer_norm_6,
            layer_norm_7,
            layer_norm_8,
            matmul_8,
            matmul_9,
            relu_4,
            matmul_10,
            relu_5,
            matmul_11,
            add_7,
            assign_1,
            bmm_1,
            concat_0,
            share_data__1,
            add_8,
            sigmoid_0,
            reshape_0,
            add_9,
            sigmoid_1,
            reshape_1,
        )
