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
    ):
        # pd_op.full: (1xf64) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("80"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (80xf32) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_0, full_1, full_2, dtype="float32")

        # builtin.combine: ([80xf32, 80xf32]) <- (80xf32, 80xf32)
        combine_0 = [arange_0, arange_0]

        # pd_op.meshgrid: ([80x80xf32, 80x80xf32]) <- ([80xf32, 80xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_0)

        # builtin.split: (80x80xf32, 80x80xf32) <- ([80x80xf32, 80x80xf32])
        (
            split_0,
            split_1,
        ) = meshgrid_0

        # builtin.combine: ([80x80xf32, 80x80xf32]) <- (80x80xf32, 80x80xf32)
        combine_1 = [split_1, split_0]

        # pd_op.stack: (80x80x2xf32) <- ([80x80xf32, 80x80xf32])
        stack_1 = paddle._C_ops.stack(combine_1, -1)

        # pd_op.full: (2xi64) <- ()
        full_3 = paddle._C_ops.full(
            [2], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (2xi64) <- (2xi64)
        assign_value__1 = paddle._C_ops.assign_value_(
            full_3,
            [2],
            paddle.int64,
            [float("80"), float("80")],
            paddle.framework._current_expected_place(),
        )

        # pd_op.cast: (2xf32) <- (2xi64)
        cast_1 = paddle._C_ops.cast(assign_value__1, paddle.float32)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.unsqueeze: (1x80x80x2xf32) <- (80x80x2xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(stack_1, full_int_array_0)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x80x80x2xf32) <- (1x80x80x2xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(unsqueeze_0, full_4, float("0.5"), True)

        # pd_op.divide: (1x80x80x2xf32) <- (1x80x80x2xf32, 2xf32)
        divide_0 = paddle._C_ops.divide(scale_0, cast_1)

        # pd_op.full_like: (1x80x80x2xf32) <- (1x80x80x2xf32, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            divide_0, full_4, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0.05"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x80x80x2xf32) <- (1x80x80x2xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(full_like_0, full_5, float("0"), True)

        # pd_op.scale: (1x80x80x2xf32) <- (1x80x80x2xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(scale_1, full_4, float("0"), True)

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([1x80x80x2xf32, 1x80x80x2xf32]) <- (1x80x80x2xf32, 1x80x80x2xf32)
        combine_2 = [divide_0, scale_2]

        # pd_op.concat: (1x80x80x4xf32) <- ([1x80x80x2xf32, 1x80x80x2xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_2, full_6)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_1 = [-1, 6400, 4]

        # pd_op.reshape: (1x6400x4xf32) <- (1x80x80x4xf32, 3xi64)
        reshape_2 = paddle._C_ops.reshape(concat_1, full_int_array_1)

        # pd_op.full: (1xf64) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("40"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (40xf32) <- (1xf64, 1xf64, 1xf64)
        arange_1 = paddle.arange(full_0, full_7, full_2, dtype="float32")

        # builtin.combine: ([40xf32, 40xf32]) <- (40xf32, 40xf32)
        combine_3 = [arange_1, arange_1]

        # pd_op.meshgrid: ([40x40xf32, 40x40xf32]) <- ([40xf32, 40xf32])
        meshgrid_1 = paddle._C_ops.meshgrid(combine_3)

        # builtin.split: (40x40xf32, 40x40xf32) <- ([40x40xf32, 40x40xf32])
        (
            split_2,
            split_3,
        ) = meshgrid_1

        # builtin.combine: ([40x40xf32, 40x40xf32]) <- (40x40xf32, 40x40xf32)
        combine_4 = [split_3, split_2]

        # pd_op.stack: (40x40x2xf32) <- ([40x40xf32, 40x40xf32])
        stack_2 = paddle._C_ops.stack(combine_4, -1)

        # pd_op.full: (2xi64) <- ()
        full_8 = paddle._C_ops.full(
            [2], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (2xi64) <- (2xi64)
        assign_value__2 = paddle._C_ops.assign_value_(
            full_8,
            [2],
            paddle.int64,
            [float("40"), float("40")],
            paddle.framework._current_expected_place(),
        )

        # pd_op.cast: (2xf32) <- (2xi64)
        cast_2 = paddle._C_ops.cast(assign_value__2, paddle.float32)

        # pd_op.unsqueeze: (1x40x40x2xf32) <- (40x40x2xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(stack_2, full_int_array_0)

        # pd_op.scale: (1x40x40x2xf32) <- (1x40x40x2xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(unsqueeze_1, full_4, float("0.5"), True)

        # pd_op.divide: (1x40x40x2xf32) <- (1x40x40x2xf32, 2xf32)
        divide_1 = paddle._C_ops.divide(scale_3, cast_2)

        # pd_op.full_like: (1x40x40x2xf32) <- (1x40x40x2xf32, 1xf32)
        full_like_1 = paddle._C_ops.full_like(
            divide_1, full_4, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.scale: (1x40x40x2xf32) <- (1x40x40x2xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(full_like_1, full_5, float("0"), True)

        # pd_op.full: (1xf32) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x40x40x2xf32) <- (1x40x40x2xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(scale_4, full_9, float("0"), True)

        # builtin.combine: ([1x40x40x2xf32, 1x40x40x2xf32]) <- (1x40x40x2xf32, 1x40x40x2xf32)
        combine_5 = [divide_1, scale_5]

        # pd_op.concat: (1x40x40x4xf32) <- ([1x40x40x2xf32, 1x40x40x2xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_5, full_6)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_2 = [-1, 1600, 4]

        # pd_op.reshape: (1x1600x4xf32) <- (1x40x40x4xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(concat_2, full_int_array_2)

        # pd_op.full: (1xf64) <- ()
        full_10 = paddle._C_ops.full(
            [1], float("20"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (20xf32) <- (1xf64, 1xf64, 1xf64)
        arange_2 = paddle.arange(full_0, full_10, full_2, dtype="float32")

        # builtin.combine: ([20xf32, 20xf32]) <- (20xf32, 20xf32)
        combine_6 = [arange_2, arange_2]

        # pd_op.meshgrid: ([20x20xf32, 20x20xf32]) <- ([20xf32, 20xf32])
        meshgrid_2 = paddle._C_ops.meshgrid(combine_6)

        # builtin.split: (20x20xf32, 20x20xf32) <- ([20x20xf32, 20x20xf32])
        (
            split_4,
            split_5,
        ) = meshgrid_2

        # builtin.combine: ([20x20xf32, 20x20xf32]) <- (20x20xf32, 20x20xf32)
        combine_7 = [split_5, split_4]

        # pd_op.stack: (20x20x2xf32) <- ([20x20xf32, 20x20xf32])
        stack_3 = paddle._C_ops.stack(combine_7, -1)

        # pd_op.full: (2xi64) <- ()
        full_11 = paddle._C_ops.full(
            [2], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (2xi64) <- (2xi64)
        assign_value__3 = paddle._C_ops.assign_value_(
            full_11,
            [2],
            paddle.int64,
            [float("20"), float("20")],
            paddle.framework._current_expected_place(),
        )

        # pd_op.cast: (2xf32) <- (2xi64)
        cast_3 = paddle._C_ops.cast(assign_value__3, paddle.float32)

        # pd_op.unsqueeze: (1x20x20x2xf32) <- (20x20x2xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(stack_3, full_int_array_0)

        # pd_op.scale: (1x20x20x2xf32) <- (1x20x20x2xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(unsqueeze_2, full_4, float("0.5"), True)

        # pd_op.divide: (1x20x20x2xf32) <- (1x20x20x2xf32, 2xf32)
        divide_2 = paddle._C_ops.divide(scale_6, cast_3)

        # pd_op.full_like: (1x20x20x2xf32) <- (1x20x20x2xf32, 1xf32)
        full_like_2 = paddle._C_ops.full_like(
            divide_2, full_4, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.scale: (1x20x20x2xf32) <- (1x20x20x2xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(full_like_2, full_5, float("0"), True)

        # pd_op.full: (1xf32) <- ()
        full_12 = paddle._C_ops.full(
            [1], float("4"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x20x20x2xf32) <- (1x20x20x2xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(scale_7, full_12, float("0"), True)

        # builtin.combine: ([1x20x20x2xf32, 1x20x20x2xf32]) <- (1x20x20x2xf32, 1x20x20x2xf32)
        combine_8 = [divide_2, scale_8]

        # pd_op.concat: (1x20x20x4xf32) <- ([1x20x20x2xf32, 1x20x20x2xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_8, full_6)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_3 = [-1, 400, 4]

        # pd_op.reshape: (1x400x4xf32) <- (1x20x20x4xf32, 3xi64)
        reshape_4 = paddle._C_ops.reshape(concat_3, full_int_array_3)

        # pd_op.full: (1xi32) <- ()
        full_13 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_0 = full_13

        # builtin.combine: ([1x6400x4xf32, 1x1600x4xf32, 1x400x4xf32]) <- (1x6400x4xf32, 1x1600x4xf32, 1x400x4xf32)
        combine_9 = [reshape_2, reshape_3, reshape_4]

        # pd_op.concat: (1x8400x4xf32) <- ([1x6400x4xf32, 1x1600x4xf32, 1x400x4xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_9, full_13)

        # pd_op.full: (xf32) <- ()
        full_14 = paddle._C_ops.full(
            [],
            float("0.01"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.greater_than: (1x8400x4xb) <- (1x8400x4xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(concat_4, full_14)

        # pd_op.full: (xf32) <- ()
        full_15 = paddle._C_ops.full(
            [],
            float("0.99"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.less_than: (1x8400x4xb) <- (1x8400x4xf32, xf32)
        less_than_0 = paddle._C_ops.less_than(concat_4, full_15)

        # pd_op.multiply: (1x8400x4xb) <- (1x8400x4xb, 1x8400x4xb)
        multiply_0 = paddle._C_ops.multiply(greater_than_0, less_than_0)

        # pd_op.all: (1x8400x1xb) <- (1x8400x4xb)
        all_0 = paddle._C_ops.all(multiply_0, [-1], True)

        # pd_op.full: (1xf32) <- ()
        full_16 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x8400x4xf32) <- (1x8400x4xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(concat_4, full_16, float("1"), True)

        # pd_op.divide: (1x8400x4xf32) <- (1x8400x4xf32, 1x8400x4xf32)
        divide_3 = paddle._C_ops.divide(concat_4, scale_9)

        # pd_op.log: (1x8400x4xf32) <- (1x8400x4xf32)
        log_0 = paddle._C_ops.log(divide_3)

        # pd_op.full: (xf32) <- ()
        full_17 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__4 = paddle._C_ops.assign_value_(
            full_17,
            [],
            paddle.float32,
            [float("inf")],
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xf32) <- ()
        full_18 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (1x8400x4xf32) <- (1x8400x4xf32, 1xf32)
        full_like_3 = paddle._C_ops.full_like(
            log_0, full_18, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.full_like: (xf32) <- (xf32, 1xf32)
        full_like_4 = paddle._C_ops.full_like(
            assign_value__4,
            full_18,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_like: (1x8400x1xb) <- (1x8400x1xb, 1xf32)
        full_like_5 = paddle._C_ops.full_like(
            all_0, full_18, paddle.bool, paddle.framework._current_expected_place()
        )

        # pd_op.cast: (1x8400x1xf32) <- (1x8400x1xb)
        cast_4 = paddle._C_ops.cast(full_like_5, paddle.float32)

        # pd_op.cast: (1x8400x1xf32) <- (1x8400x1xb)
        cast_5 = paddle._C_ops.cast(all_0, paddle.float32)

        # pd_op.add: (1x8400x4xf32) <- (1x8400x4xf32, xf32)
        add_10 = paddle._C_ops.add(full_like_3, full_like_4)

        # pd_op.add: (1x8400x4xf32) <- (1x8400x4xf32, 1x8400x1xf32)
        add_11 = paddle._C_ops.add(add_10, cast_4)

        # pd_op.add: (1x8400x4xf32) <- (1x8400x4xf32, 1x8400x4xf32)
        add_12 = paddle._C_ops.add(log_0, add_11)

        # pd_op.add: (1x8400x4xf32) <- (xf32, 1x8400x4xf32)
        add_13 = paddle._C_ops.add(assign_value__4, add_11)

        # pd_op.add: (1x8400x4xf32) <- (1x8400x1xf32, 1x8400x4xf32)
        add_14 = paddle._C_ops.add(cast_5, add_11)

        # pd_op.cast: (1x8400x4xb) <- (1x8400x4xf32)
        cast_6 = paddle._C_ops.cast(add_14, paddle.bool)

        # pd_op.where: (1x8400x4xf32) <- (1x8400x4xb, 1x8400x4xf32, 1x8400x4xf32)
        where_0 = paddle._C_ops.where(cast_6, add_12, add_13)

        # pd_op.full: (xf32) <- ()
        full_19 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_19,
            [],
            paddle.float32,
            [float("0")],
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_like: (1x8400x256xf32) <- (1x8400x256xf32, 1xf32)
        full_like_6 = paddle._C_ops.full_like(
            data_0, full_18, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.full_like: (xf32) <- (xf32, 1xf32)
        full_like_7 = paddle._C_ops.full_like(
            assign_value__0,
            full_18,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_like: (1x8400x1xb) <- (1x8400x1xb, 1xf32)
        full_like_8 = paddle._C_ops.full_like(
            all_0, full_18, paddle.bool, paddle.framework._current_expected_place()
        )

        # pd_op.cast: (1x8400x1xf32) <- (1x8400x1xb)
        cast_7 = paddle._C_ops.cast(full_like_8, paddle.float32)

        # pd_op.cast: (1x8400x1xf32) <- (1x8400x1xb)
        cast_8 = paddle._C_ops.cast(all_0, paddle.float32)

        # pd_op.add: (1x8400x256xf32) <- (1x8400x256xf32, xf32)
        add_15 = paddle._C_ops.add(full_like_6, full_like_7)

        # pd_op.add: (1x8400x256xf32) <- (1x8400x256xf32, 1x8400x1xf32)
        add_0 = paddle._C_ops.add(add_15, cast_7)

        # pd_op.add: (1x8400x256xf32) <- (1x8400x256xf32, 1x8400x256xf32)
        add_1 = paddle._C_ops.add(data_0, add_0)

        # pd_op.add: (1x8400x256xf32) <- (xf32, 1x8400x256xf32)
        add_2 = paddle._C_ops.add(assign_value__0, add_0)

        # pd_op.add: (1x8400x256xf32) <- (1x8400x1xf32, 1x8400x256xf32)
        add_16 = paddle._C_ops.add(cast_8, add_0)

        # pd_op.cast: (1x8400x256xb) <- (1x8400x256xf32)
        cast_0 = paddle._C_ops.cast(add_16, paddle.bool)

        # pd_op.where: (1x8400x256xf32) <- (1x8400x256xb, 1x8400x256xf32, 1x8400x256xf32)
        where_1 = paddle._C_ops.where(cast_0, add_1, add_2)

        # pd_op.matmul: (1x8400x256xf32) <- (1x8400x256xf32, 256x256xf32)
        matmul_0 = paddle._C_ops.matmul(where_1, parameter_19, False, False)

        # pd_op.add: (1x8400x256xf32) <- (1x8400x256xf32, 256xf32)
        add_3 = paddle._C_ops.add(matmul_0, parameter_18)

        # pd_op.layer_norm: (1x8400x256xf32, 1x8400xf32, 1x8400xf32) <- (1x8400x256xf32, 256xf32, 256xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_3, parameter_17, parameter_16, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (1x8400x2xf32) <- (1x8400x256xf32, 256x2xf32)
        matmul_12 = paddle._C_ops.matmul(layer_norm_0, parameter_15, False, False)

        # pd_op.add: (1x8400x2xf32) <- (1x8400x2xf32, 2xf32)
        add_17 = paddle._C_ops.add(matmul_12, parameter_14)

        # pd_op.matmul: (1x8400x256xf32) <- (1x8400x256xf32, 256x256xf32)
        matmul_1 = paddle._C_ops.matmul(layer_norm_0, parameter_13, False, False)

        # pd_op.add: (1x8400x256xf32) <- (1x8400x256xf32, 256xf32)
        add_18 = paddle._C_ops.add(matmul_1, parameter_12)

        # pd_op.relu: (1x8400x256xf32) <- (1x8400x256xf32)
        relu_0 = paddle._C_ops.relu(add_18)

        # pd_op.matmul: (1x8400x256xf32) <- (1x8400x256xf32, 256x256xf32)
        matmul_2 = paddle._C_ops.matmul(relu_0, parameter_11, False, False)

        # pd_op.add: (1x8400x256xf32) <- (1x8400x256xf32, 256xf32)
        add_19 = paddle._C_ops.add(matmul_2, parameter_10)

        # pd_op.relu: (1x8400x256xf32) <- (1x8400x256xf32)
        relu_1 = paddle._C_ops.relu(add_19)

        # pd_op.matmul: (1x8400x4xf32) <- (1x8400x256xf32, 256x4xf32)
        matmul_3 = paddle._C_ops.matmul(relu_1, parameter_9, False, False)

        # pd_op.add: (1x8400x4xf32) <- (1x8400x4xf32, 4xf32)
        add_4 = paddle._C_ops.add(matmul_3, parameter_8)

        # pd_op.add: (1x8400x4xf32) <- (1x8400x4xf32, 1x8400x4xf32)
        add_5 = paddle._C_ops.add(add_4, where_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [-1]

        # pd_op.max: (1x8400xf32) <- (1x8400x2xf32, 1xi64)
        max_0 = paddle._C_ops.max(add_17, full_int_array_4, False)

        # pd_op.full: (1xi32) <- ()
        full_20 = paddle._C_ops.full(
            [1], float("300"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.topk: (1x300xf32, 1x300xi64) <- (1x8400xf32, 1xi32)
        topk_0, topk_1 = (lambda x, f: f(x))(
            paddle._C_ops.topk(max_0, full_20, 1, True, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.arange: (1xi64) <- (1xf64, 1xf64, 1xf64)
        arange_3 = paddle.arange(full_0, full_2, full_2, dtype="int64")

        # pd_op.unsqueeze: (1x1xi64) <- (1xi64, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(arange_3, full_int_array_4)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [1, 300]

        # pd_op.tile: (1x300xi64) <- (1x1xi64, 2xi64)
        tile_0 = paddle._C_ops.tile(unsqueeze_3, full_int_array_5)

        # builtin.combine: ([1x300xi64, 1x300xi64]) <- (1x300xi64, 1x300xi64)
        combine_10 = [tile_0, topk_1]

        # pd_op.stack: (1x300x2xi64) <- ([1x300xi64, 1x300xi64])
        stack_0 = paddle._C_ops.stack(combine_10, -1)

        # pd_op.gather_nd: (1x300x256xf32) <- (1x8400x256xf32, 1x300x2xi64)
        gather_nd_0 = paddle._C_ops.gather_nd(layer_norm_0, stack_0)

        # pd_op.gather_nd: (1x300x4xf32) <- (1x8400x4xf32, 1x300x2xi64)
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

        # pd_op.flatten: (1x128x25600xf32) <- (1x128x160x160xf32)
        flatten_0 = paddle._C_ops.flatten(data_1, 2, 3)

        # pd_op.assign: (1x128x25600xf32) <- (1x128x25600xf32)
        assign_1 = flatten_0

        # pd_op.bmm: (1x300x25600xf32) <- (1x300x128xf32, 1x128x25600xf32)
        bmm_0 = paddle._C_ops.bmm(add_6, flatten_0)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_6 = [1, 300, 160, 160]

        # pd_op.reshape: (1x300x160x160xf32) <- (1x300x25600xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(bmm_0, full_int_array_6)

        # pd_op.sigmoid: (1x300x4xf32) <- (1x300x4xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(gather_nd_1)

        # pd_op.share_data_: (1x300x256xf32) <- (1x300x256xf32)
        share_data__0 = gather_nd_0.detach()

        # builtin.combine: ([1x100x256xf32, 1x300x256xf32]) <- (1x100x256xf32, 1x300x256xf32)
        combine_11 = [data_2, share_data__0]

        # pd_op.concat: (1x400x256xf32) <- ([1x100x256xf32, 1x300x256xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_11, full_13)

        # pd_op.full: (xf32) <- ()
        full_21 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (1x300x160x160xb) <- (1x300x160x160xf32, xf32)
        greater_than_1 = paddle._C_ops.greater_than(reshape_0, full_21)

        # pd_op.full: (1xf64) <- ()
        full_22 = paddle._C_ops.full(
            [1], float("160"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (160xf32) <- (1xf64, 1xf64, 1xf64)
        arange_4 = paddle.arange(full_0, full_22, full_2, dtype="float32")

        # builtin.combine: ([160xf32, 160xf32]) <- (160xf32, 160xf32)
        combine_12 = [arange_4, arange_4]

        # pd_op.meshgrid: ([160x160xf32, 160x160xf32]) <- ([160xf32, 160xf32])
        meshgrid_3 = paddle._C_ops.meshgrid(combine_12)

        # builtin.split: (160x160xf32, 160x160xf32) <- ([160x160xf32, 160x160xf32])
        (
            split_6,
            split_7,
        ) = meshgrid_3

        # pd_op.cast: (1x300x160x160xf32) <- (1x300x160x160xb)
        cast_9 = paddle._C_ops.cast(greater_than_1, paddle.float32)

        # pd_op.multiply: (1x300x160x160xf32) <- (160x160xf32, 1x300x160x160xf32)
        multiply_1 = paddle._C_ops.multiply(split_7, cast_9)

        # pd_op.flatten: (1x300x25600xf32) <- (1x300x160x160xf32)
        flatten_1 = paddle._C_ops.flatten(multiply_1, 2, 3)

        # pd_op.max: (1x300xf32) <- (1x300x25600xf32, 1xi64)
        max_1 = paddle._C_ops.max(flatten_1, full_int_array_4, False)

        # pd_op.scale: (1x300xf32) <- (1x300xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(max_1, full_4, float("1"), True)

        # pd_op.full: (xf32) <- ()
        full_23 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__5 = paddle._C_ops.assign_value_(
            full_23,
            [],
            paddle.float32,
            [float("1e+08")],
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_like: (1x300x160x160xf32) <- (1x300x160x160xf32, 1xf32)
        full_like_9 = paddle._C_ops.full_like(
            multiply_1,
            full_18,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_like: (xf32) <- (xf32, 1xf32)
        full_like_10 = paddle._C_ops.full_like(
            assign_value__5,
            full_18,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_like: (1x300x160x160xb) <- (1x300x160x160xb, 1xf32)
        full_like_11 = paddle._C_ops.full_like(
            greater_than_1,
            full_18,
            paddle.bool,
            paddle.framework._current_expected_place(),
        )

        # pd_op.cast: (1x300x160x160xf32) <- (1x300x160x160xb)
        cast_10 = paddle._C_ops.cast(full_like_11, paddle.float32)

        # pd_op.cast: (1x300x160x160xf32) <- (1x300x160x160xb)
        cast_11 = paddle._C_ops.cast(greater_than_1, paddle.float32)

        # pd_op.add: (1x300x160x160xf32) <- (1x300x160x160xf32, xf32)
        add_22 = paddle._C_ops.add(full_like_9, full_like_10)

        # pd_op.add: (1x300x160x160xf32) <- (1x300x160x160xf32, 1x300x160x160xf32)
        add_23 = paddle._C_ops.add(add_22, cast_10)

        # pd_op.add: (1x300x160x160xf32) <- (1x300x160x160xf32, 1x300x160x160xf32)
        add_24 = paddle._C_ops.add(multiply_1, add_23)

        # pd_op.add: (1x300x160x160xf32) <- (xf32, 1x300x160x160xf32)
        add_25 = paddle._C_ops.add(assign_value__5, add_23)

        # pd_op.add: (1x300x160x160xf32) <- (1x300x160x160xf32, 1x300x160x160xf32)
        add_26 = paddle._C_ops.add(cast_11, add_23)

        # pd_op.cast: (1x300x160x160xb) <- (1x300x160x160xf32)
        cast_12 = paddle._C_ops.cast(add_26, paddle.bool)

        # pd_op.where: (1x300x160x160xf32) <- (1x300x160x160xb, 1x300x160x160xf32, 1x300x160x160xf32)
        where_2 = paddle._C_ops.where(cast_12, add_24, add_25)

        # pd_op.flatten: (1x300x25600xf32) <- (1x300x160x160xf32)
        flatten_2 = paddle._C_ops.flatten(where_2, 2, 3)

        # pd_op.min: (1x300xf32) <- (1x300x25600xf32, 1xi64)
        min_0 = paddle._C_ops.min(flatten_2, full_int_array_4, False)

        # pd_op.cast: (1x300x160x160xf32) <- (1x300x160x160xb)
        cast_13 = paddle._C_ops.cast(greater_than_1, paddle.float32)

        # pd_op.multiply: (1x300x160x160xf32) <- (160x160xf32, 1x300x160x160xf32)
        multiply_2 = paddle._C_ops.multiply(split_6, cast_13)

        # pd_op.flatten: (1x300x25600xf32) <- (1x300x160x160xf32)
        flatten_3 = paddle._C_ops.flatten(multiply_2, 2, 3)

        # pd_op.max: (1x300xf32) <- (1x300x25600xf32, 1xi64)
        max_2 = paddle._C_ops.max(flatten_3, full_int_array_4, False)

        # pd_op.scale: (1x300xf32) <- (1x300xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(max_2, full_4, float("1"), True)

        # pd_op.full: (xf32) <- ()
        full_24 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__6 = paddle._C_ops.assign_value_(
            full_24,
            [],
            paddle.float32,
            [float("1e+08")],
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_like: (1x300x160x160xf32) <- (1x300x160x160xf32, 1xf32)
        full_like_12 = paddle._C_ops.full_like(
            multiply_2,
            full_18,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_like: (xf32) <- (xf32, 1xf32)
        full_like_13 = paddle._C_ops.full_like(
            assign_value__6,
            full_18,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_like: (1x300x160x160xb) <- (1x300x160x160xb, 1xf32)
        full_like_14 = paddle._C_ops.full_like(
            greater_than_1,
            full_18,
            paddle.bool,
            paddle.framework._current_expected_place(),
        )

        # pd_op.cast: (1x300x160x160xf32) <- (1x300x160x160xb)
        cast_14 = paddle._C_ops.cast(full_like_14, paddle.float32)

        # pd_op.cast: (1x300x160x160xf32) <- (1x300x160x160xb)
        cast_15 = paddle._C_ops.cast(greater_than_1, paddle.float32)

        # pd_op.add: (1x300x160x160xf32) <- (1x300x160x160xf32, xf32)
        add_27 = paddle._C_ops.add(full_like_12, full_like_13)

        # pd_op.add: (1x300x160x160xf32) <- (1x300x160x160xf32, 1x300x160x160xf32)
        add_28 = paddle._C_ops.add(add_27, cast_14)

        # pd_op.add: (1x300x160x160xf32) <- (1x300x160x160xf32, 1x300x160x160xf32)
        add_29 = paddle._C_ops.add(multiply_2, add_28)

        # pd_op.add: (1x300x160x160xf32) <- (xf32, 1x300x160x160xf32)
        add_30 = paddle._C_ops.add(assign_value__6, add_28)

        # pd_op.add: (1x300x160x160xf32) <- (1x300x160x160xf32, 1x300x160x160xf32)
        add_31 = paddle._C_ops.add(cast_15, add_28)

        # pd_op.cast: (1x300x160x160xb) <- (1x300x160x160xf32)
        cast_16 = paddle._C_ops.cast(add_31, paddle.bool)

        # pd_op.where: (1x300x160x160xf32) <- (1x300x160x160xb, 1x300x160x160xf32, 1x300x160x160xf32)
        where_3 = paddle._C_ops.where(cast_16, add_29, add_30)

        # pd_op.flatten: (1x300x25600xf32) <- (1x300x160x160xf32)
        flatten_4 = paddle._C_ops.flatten(where_3, 2, 3)

        # pd_op.min: (1x300xf32) <- (1x300x25600xf32, 1xi64)
        min_1 = paddle._C_ops.min(flatten_4, full_int_array_4, False)

        # builtin.combine: ([1x300xf32, 1x300xf32, 1x300xf32, 1x300xf32]) <- (1x300xf32, 1x300xf32, 1x300xf32, 1x300xf32)
        combine_13 = [min_0, min_1, scale_10, scale_11]

        # pd_op.stack: (1x300x4xf32) <- ([1x300xf32, 1x300xf32, 1x300xf32, 1x300xf32])
        stack_4 = paddle._C_ops.stack(combine_13, -1)

        # pd_op.any: (1x300xb) <- (1x300x160x160xb)
        any_0 = paddle._C_ops.any(greater_than_1, [2, 3], False)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [2]

        # pd_op.unsqueeze: (1x300x1xb) <- (1x300xb, 1xi64)
        unsqueeze_4 = paddle._C_ops.unsqueeze(any_0, full_int_array_7)

        # pd_op.cast: (1x300x1xf32) <- (1x300x1xb)
        cast_17 = paddle._C_ops.cast(unsqueeze_4, paddle.float32)

        # pd_op.multiply: (1x300x4xf32) <- (1x300x4xf32, 1x300x1xf32)
        multiply_3 = paddle._C_ops.multiply(stack_4, cast_17)

        # pd_op.full: (4xi64) <- ()
        full_25 = paddle._C_ops.full(
            [4], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (4xi64) <- (4xi64)
        assign_value__7 = paddle._C_ops.assign_value_(
            full_25,
            [4],
            paddle.int64,
            [float("160"), float("160"), float("160"), float("160")],
            paddle.framework._current_expected_place(),
        )

        # pd_op.cast: (4xf32) <- (4xi64)
        cast_18 = paddle._C_ops.cast(assign_value__7, paddle.float32)

        # pd_op.divide: (1x300x4xf32) <- (1x300x4xf32, 4xf32)
        divide_4 = paddle._C_ops.divide(multiply_3, cast_18)

        # pd_op.full: (1xi32) <- ()
        full_26 = paddle._C_ops.full(
            [1], float("2"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split_with_num: ([1x300x1xf32, 1x300x1xf32, 1x300x1xf32, 1x300x1xf32]) <- (1x300x4xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(divide_4, 4, full_26)

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
        full_27 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x300x1xf32) <- (1x300x1xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(add_32, full_27, float("0"), True)

        # pd_op.add: (1x300x1xf32) <- (1x300x1xf32, 1x300x1xf32)
        add_33 = paddle._C_ops.add(split_9, split_11)

        # pd_op.scale: (1x300x1xf32) <- (1x300x1xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(add_33, full_27, float("0"), True)

        # pd_op.subtract: (1x300x1xf32) <- (1x300x1xf32, 1x300x1xf32)
        subtract_0 = paddle._C_ops.subtract(split_10, split_8)

        # pd_op.subtract: (1x300x1xf32) <- (1x300x1xf32, 1x300x1xf32)
        subtract_1 = paddle._C_ops.subtract(split_11, split_9)

        # builtin.combine: ([1x300x1xf32, 1x300x1xf32, 1x300x1xf32, 1x300x1xf32]) <- (1x300x1xf32, 1x300x1xf32, 1x300x1xf32, 1x300x1xf32)
        combine_14 = [scale_12, scale_13, subtract_0, subtract_1]

        # pd_op.concat: (1x300x4xf32) <- ([1x300x1xf32, 1x300x1xf32, 1x300x1xf32, 1x300x1xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_14, full_6)

        # pd_op.clip: (1x300x4xf32) <- (1x300x4xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(concat_5, full_18, full_4)

        # pd_op.full: (1xf32) <- ()
        full_28 = paddle._C_ops.full(
            [1], float("1e-05"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_29 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (1x300x4xf32) <- (1x300x4xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(clip_0, full_28, full_29)

        # pd_op.scale: (1x300x4xf32) <- (1x300x4xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(clip_0, full_16, float("1"), True)

        # pd_op.clip: (1x300x4xf32) <- (1x300x4xf32, 1xf32, 1xf32)
        clip_2 = paddle._C_ops.clip(scale_14, full_28, full_29)

        # pd_op.divide: (1x300x4xf32) <- (1x300x4xf32, 1x300x4xf32)
        divide_5 = paddle._C_ops.divide(clip_1, clip_2)

        # pd_op.log: (1x300x4xf32) <- (1x300x4xf32)
        log_1 = paddle._C_ops.log(divide_5)

        # builtin.combine: ([1x100x4xf32, 1x300x4xf32]) <- (1x100x4xf32, 1x300x4xf32)
        combine_15 = [data_3, log_1]

        # pd_op.concat: (1x400x4xf32) <- ([1x100x4xf32, 1x300x4xf32], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_15, full_13)

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

        # pd_op.bmm: (1x400x25600xf32) <- (1x400x128xf32, 1x128x25600xf32)
        bmm_1 = paddle._C_ops.bmm(add_7, flatten_0)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_8 = [1, 400, 160, 160]

        # pd_op.reshape: (1x400x160x160xf32) <- (1x400x25600xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(bmm_1, full_int_array_8)

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
