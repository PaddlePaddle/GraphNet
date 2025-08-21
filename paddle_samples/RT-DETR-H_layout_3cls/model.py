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
        data_0,
        data_1,
        data_2,
        data_3,
        data_4,
        data_5,
        data_6,
        data_7,
        data_8,
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
        assign_1 = stack_2

        # pd_op.cast: (2xf32) <- (2xi64)
        cast_3 = paddle._C_ops.cast(assign_1, paddle.float32)

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
        reshape_0 = paddle._C_ops.reshape(concat_1, stack_3)

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
        assign_2 = stack_5

        # pd_op.cast: (2xf32) <- (2xi64)
        cast_6 = paddle._C_ops.cast(assign_2, paddle.float32)

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
        reshape_1 = paddle._C_ops.reshape(concat_2, stack_6)

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
        assign_3 = stack_8

        # pd_op.cast: (2xf32) <- (2xi64)
        cast_9 = paddle._C_ops.cast(assign_3, paddle.float32)

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
        reshape_2 = paddle._C_ops.reshape(concat_3, stack_9)

        # pd_op.full: (1xi32) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_0 = full_8

        # builtin.combine: ([-1x-1x4xf32, -1x-1x4xf32, -1x-1x4xf32]) <- (-1x-1x4xf32, -1x-1x4xf32, -1x-1x4xf32)
        combine_15 = [reshape_0, reshape_1, reshape_2]

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
        add_7 = paddle._C_ops.add(full_like_3, full_like_4)

        # pd_op.add: (-1x-1x4xf32) <- (-1x-1x4xf32, -1x-1x1xf32)
        add_8 = paddle._C_ops.add(add_7, cast_10)

        # pd_op.add: (-1x-1x4xf32) <- (-1x-1x4xf32, -1x-1x4xf32)
        add_9 = paddle._C_ops.add(log_0, add_8)

        # pd_op.add: (-1x-1x4xf32) <- (xf32, -1x-1x4xf32)
        add_10 = paddle._C_ops.add(assign_value__1, add_8)

        # pd_op.add: (-1x-1x4xf32) <- (-1x-1x1xf32, -1x-1x4xf32)
        add_11 = paddle._C_ops.add(cast_11, add_8)

        # pd_op.cast: (-1x-1x4xb) <- (-1x-1x4xf32)
        cast_12 = paddle._C_ops.cast(add_11, paddle.bool)

        # pd_op.where: (-1x-1x4xf32) <- (-1x-1x4xb, -1x-1x4xf32, -1x-1x4xf32)
        where_0 = paddle._C_ops.where(cast_12, add_9, add_10)

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
        add_12 = paddle._C_ops.add(full_like_6, full_like_7)

        # pd_op.add: (-1x-1x256xf32) <- (1x-1x256xf32, -1x-1x1xf32)
        add_0 = paddle._C_ops.add(add_12, cast_10)

        # pd_op.add: (-1x-1x256xf32) <- (1x-1x256xf32, -1x-1x256xf32)
        add_1 = paddle._C_ops.add(data_6, add_0)

        # pd_op.add: (-1x-1x256xf32) <- (xf32, -1x-1x256xf32)
        add_2 = paddle._C_ops.add(assign_value__0, add_0)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x1xf32, -1x-1x256xf32)
        add_13 = paddle._C_ops.add(cast_11, add_0)

        # pd_op.cast: (-1x-1x256xb) <- (-1x-1x256xf32)
        cast_0 = paddle._C_ops.cast(add_13, paddle.bool)

        # pd_op.where: (-1x-1x256xf32) <- (-1x-1x256xb, -1x-1x256xf32, -1x-1x256xf32)
        where_1 = paddle._C_ops.where(cast_0, add_1, add_2)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x256xf32, 256x256xf32)
        matmul_0 = paddle._C_ops.matmul(where_1, parameter_11, False, False)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_3 = paddle._C_ops.add(matmul_0, parameter_10)

        # pd_op.layer_norm: (-1x-1x256xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x256xf32, 256xf32, 256xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_3, parameter_9, parameter_8, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (-1x-1x11xf32) <- (-1x-1x256xf32, 256x11xf32)
        matmul_1 = paddle._C_ops.matmul(layer_norm_0, parameter_7, False, False)

        # pd_op.add: (-1x-1x11xf32) <- (-1x-1x11xf32, 11xf32)
        add_4 = paddle._C_ops.add(matmul_1, parameter_6)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x256xf32, 256x256xf32)
        matmul_2 = paddle._C_ops.matmul(layer_norm_0, parameter_5, False, False)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_14 = paddle._C_ops.add(matmul_2, parameter_4)

        # pd_op.relu: (-1x-1x256xf32) <- (-1x-1x256xf32)
        relu_0 = paddle._C_ops.relu(add_14)

        # pd_op.matmul: (-1x-1x256xf32) <- (-1x-1x256xf32, 256x256xf32)
        matmul_3 = paddle._C_ops.matmul(relu_0, parameter_3, False, False)

        # pd_op.add: (-1x-1x256xf32) <- (-1x-1x256xf32, 256xf32)
        add_15 = paddle._C_ops.add(matmul_3, parameter_2)

        # pd_op.relu: (-1x-1x256xf32) <- (-1x-1x256xf32)
        relu_1 = paddle._C_ops.relu(add_15)

        # pd_op.matmul: (-1x-1x4xf32) <- (-1x-1x256xf32, 256x4xf32)
        matmul_4 = paddle._C_ops.matmul(relu_1, parameter_1, False, False)

        # pd_op.add: (-1x-1x4xf32) <- (-1x-1x4xf32, 4xf32)
        add_5 = paddle._C_ops.add(matmul_4, parameter_0)

        # pd_op.add: (-1x-1x4xf32) <- (-1x-1x4xf32, -1x-1x4xf32)
        add_6 = paddle._C_ops.add(add_5, where_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [-1]

        # pd_op.max: (-1x-1xf32) <- (-1x-1x11xf32, 1xi64)
        max_0 = paddle._C_ops.max(add_4, full_int_array_3, False)

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

        # pd_op.gather_nd: (1x300x4xf32) <- (-1x-1x4xf32, 1x300x2xi64)
        gather_nd_1 = paddle._C_ops.gather_nd(add_6, stack_0)

        # pd_op.sigmoid: (1x300x4xf32) <- (1x300x4xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(gather_nd_1)

        # builtin.combine: ([1x-1x4xf32, 1x300x4xf32]) <- (1x-1x4xf32, 1x300x4xf32)
        combine_17 = [data_8, gather_nd_1]

        # pd_op.concat: (1x-1x4xf32) <- ([1x-1x4xf32, 1x300x4xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_17, full_8)

        # pd_op.share_data_: (1x-1x4xf32) <- (1x-1x4xf32)
        share_data__1 = concat_5.detach()

        # pd_op.gather_nd: (1x300x11xf32) <- (-1x-1x11xf32, 1x300x2xi64)
        gather_nd_0 = paddle._C_ops.gather_nd(add_4, stack_0)

        # pd_op.gather_nd: (1x300x256xf32) <- (-1x-1x256xf32, 1x300x2xi64)
        gather_nd_2 = paddle._C_ops.gather_nd(layer_norm_0, stack_0)

        # pd_op.share_data_: (1x300x256xf32) <- (1x300x256xf32)
        share_data__0 = gather_nd_2.detach()

        # builtin.combine: ([1x-1x256xf32, 1x300x256xf32]) <- (1x-1x256xf32, 1x300x256xf32)
        combine_18 = [data_7, share_data__0]

        # pd_op.concat: (1x-1x256xf32) <- ([1x-1x256xf32, 1x300x256xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_18, full_8)
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
            add_4,
            matmul_2,
            relu_0,
            matmul_3,
            relu_1,
            matmul_4,
            add_5,
            add_6,
            stack_0,
            share_data__0,
            assign_0,
            concat_0,
            share_data__1,
            sigmoid_0,
            gather_nd_0,
        )
