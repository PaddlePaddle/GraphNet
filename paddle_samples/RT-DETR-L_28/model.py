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
    ):
        # pd_op.full: (1xf64) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("84"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (84xf32) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_0, full_1, full_2, dtype="float32")

        # builtin.combine: ([84xf32, 84xf32]) <- (84xf32, 84xf32)
        combine_0 = [arange_0, arange_0]

        # pd_op.meshgrid: ([84x84xf32, 84x84xf32]) <- ([84xf32, 84xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_0)

        # builtin.split: (84x84xf32, 84x84xf32) <- ([84x84xf32, 84x84xf32])
        (
            split_0,
            split_1,
        ) = meshgrid_0

        # builtin.combine: ([84x84xf32, 84x84xf32]) <- (84x84xf32, 84x84xf32)
        combine_1 = [split_1, split_0]

        # pd_op.stack: (84x84x2xf32) <- ([84x84xf32, 84x84xf32])
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
            [float("84"), float("84")],
            paddle.framework._current_expected_place(),
        )

        # pd_op.cast: (2xf32) <- (2xi64)
        cast_1 = paddle._C_ops.cast(assign_value__1, paddle.float32)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.unsqueeze: (1x84x84x2xf32) <- (84x84x2xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(stack_1, full_int_array_0)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x84x84x2xf32) <- (1x84x84x2xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(unsqueeze_0, full_4, float("0.5"), True)

        # pd_op.divide: (1x84x84x2xf32) <- (1x84x84x2xf32, 2xf32)
        divide_0 = paddle._C_ops.divide(scale_0, cast_1)

        # pd_op.full_like: (1x84x84x2xf32) <- (1x84x84x2xf32, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            divide_0, full_4, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0.05"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x84x84x2xf32) <- (1x84x84x2xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(full_like_0, full_5, float("0"), True)

        # pd_op.scale: (1x84x84x2xf32) <- (1x84x84x2xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(scale_1, full_4, float("0"), True)

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([1x84x84x2xf32, 1x84x84x2xf32]) <- (1x84x84x2xf32, 1x84x84x2xf32)
        combine_2 = [divide_0, scale_2]

        # pd_op.concat: (1x84x84x4xf32) <- ([1x84x84x2xf32, 1x84x84x2xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_2, full_6)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_1 = [-1, 7056, 4]

        # pd_op.reshape: (1x7056x4xf32) <- (1x84x84x4xf32, 3xi64)
        reshape_0 = paddle._C_ops.reshape(concat_1, full_int_array_1)

        # pd_op.full: (1xf64) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("42"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (42xf32) <- (1xf64, 1xf64, 1xf64)
        arange_1 = paddle.arange(full_0, full_7, full_2, dtype="float32")

        # builtin.combine: ([42xf32, 42xf32]) <- (42xf32, 42xf32)
        combine_3 = [arange_1, arange_1]

        # pd_op.meshgrid: ([42x42xf32, 42x42xf32]) <- ([42xf32, 42xf32])
        meshgrid_1 = paddle._C_ops.meshgrid(combine_3)

        # builtin.split: (42x42xf32, 42x42xf32) <- ([42x42xf32, 42x42xf32])
        (
            split_2,
            split_3,
        ) = meshgrid_1

        # builtin.combine: ([42x42xf32, 42x42xf32]) <- (42x42xf32, 42x42xf32)
        combine_4 = [split_3, split_2]

        # pd_op.stack: (42x42x2xf32) <- ([42x42xf32, 42x42xf32])
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
            [float("42"), float("42")],
            paddle.framework._current_expected_place(),
        )

        # pd_op.cast: (2xf32) <- (2xi64)
        cast_2 = paddle._C_ops.cast(assign_value__2, paddle.float32)

        # pd_op.unsqueeze: (1x42x42x2xf32) <- (42x42x2xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(stack_2, full_int_array_0)

        # pd_op.scale: (1x42x42x2xf32) <- (1x42x42x2xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(unsqueeze_1, full_4, float("0.5"), True)

        # pd_op.divide: (1x42x42x2xf32) <- (1x42x42x2xf32, 2xf32)
        divide_1 = paddle._C_ops.divide(scale_3, cast_2)

        # pd_op.full_like: (1x42x42x2xf32) <- (1x42x42x2xf32, 1xf32)
        full_like_1 = paddle._C_ops.full_like(
            divide_1, full_4, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.scale: (1x42x42x2xf32) <- (1x42x42x2xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(full_like_1, full_5, float("0"), True)

        # pd_op.full: (1xf32) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x42x42x2xf32) <- (1x42x42x2xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(scale_4, full_9, float("0"), True)

        # builtin.combine: ([1x42x42x2xf32, 1x42x42x2xf32]) <- (1x42x42x2xf32, 1x42x42x2xf32)
        combine_5 = [divide_1, scale_5]

        # pd_op.concat: (1x42x42x4xf32) <- ([1x42x42x2xf32, 1x42x42x2xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_5, full_6)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_2 = [-1, 1764, 4]

        # pd_op.reshape: (1x1764x4xf32) <- (1x42x42x4xf32, 3xi64)
        reshape_1 = paddle._C_ops.reshape(concat_2, full_int_array_2)

        # pd_op.full: (1xf64) <- ()
        full_10 = paddle._C_ops.full(
            [1], float("21"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (21xf32) <- (1xf64, 1xf64, 1xf64)
        arange_2 = paddle.arange(full_0, full_10, full_2, dtype="float32")

        # builtin.combine: ([21xf32, 21xf32]) <- (21xf32, 21xf32)
        combine_6 = [arange_2, arange_2]

        # pd_op.meshgrid: ([21x21xf32, 21x21xf32]) <- ([21xf32, 21xf32])
        meshgrid_2 = paddle._C_ops.meshgrid(combine_6)

        # builtin.split: (21x21xf32, 21x21xf32) <- ([21x21xf32, 21x21xf32])
        (
            split_4,
            split_5,
        ) = meshgrid_2

        # builtin.combine: ([21x21xf32, 21x21xf32]) <- (21x21xf32, 21x21xf32)
        combine_7 = [split_5, split_4]

        # pd_op.stack: (21x21x2xf32) <- ([21x21xf32, 21x21xf32])
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
            [float("21"), float("21")],
            paddle.framework._current_expected_place(),
        )

        # pd_op.cast: (2xf32) <- (2xi64)
        cast_3 = paddle._C_ops.cast(assign_value__3, paddle.float32)

        # pd_op.unsqueeze: (1x21x21x2xf32) <- (21x21x2xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(stack_3, full_int_array_0)

        # pd_op.scale: (1x21x21x2xf32) <- (1x21x21x2xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(unsqueeze_2, full_4, float("0.5"), True)

        # pd_op.divide: (1x21x21x2xf32) <- (1x21x21x2xf32, 2xf32)
        divide_2 = paddle._C_ops.divide(scale_6, cast_3)

        # pd_op.full_like: (1x21x21x2xf32) <- (1x21x21x2xf32, 1xf32)
        full_like_2 = paddle._C_ops.full_like(
            divide_2, full_4, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.scale: (1x21x21x2xf32) <- (1x21x21x2xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(full_like_2, full_5, float("0"), True)

        # pd_op.full: (1xf32) <- ()
        full_12 = paddle._C_ops.full(
            [1], float("4"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x21x21x2xf32) <- (1x21x21x2xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(scale_7, full_12, float("0"), True)

        # builtin.combine: ([1x21x21x2xf32, 1x21x21x2xf32]) <- (1x21x21x2xf32, 1x21x21x2xf32)
        combine_8 = [divide_2, scale_8]

        # pd_op.concat: (1x21x21x4xf32) <- ([1x21x21x2xf32, 1x21x21x2xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_8, full_6)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_3 = [-1, 441, 4]

        # pd_op.reshape: (1x441x4xf32) <- (1x21x21x4xf32, 3xi64)
        reshape_2 = paddle._C_ops.reshape(concat_3, full_int_array_3)

        # pd_op.full: (1xi32) <- ()
        full_13 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_0 = full_13

        # builtin.combine: ([1x7056x4xf32, 1x1764x4xf32, 1x441x4xf32]) <- (1x7056x4xf32, 1x1764x4xf32, 1x441x4xf32)
        combine_9 = [reshape_0, reshape_1, reshape_2]

        # pd_op.concat: (1x9261x4xf32) <- ([1x7056x4xf32, 1x1764x4xf32, 1x441x4xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_9, full_13)

        # pd_op.full: (xf32) <- ()
        full_14 = paddle._C_ops.full(
            [],
            float("0.01"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.greater_than: (1x9261x4xb) <- (1x9261x4xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(concat_4, full_14)

        # pd_op.full: (xf32) <- ()
        full_15 = paddle._C_ops.full(
            [],
            float("0.99"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.less_than: (1x9261x4xb) <- (1x9261x4xf32, xf32)
        less_than_0 = paddle._C_ops.less_than(concat_4, full_15)

        # pd_op.multiply: (1x9261x4xb) <- (1x9261x4xb, 1x9261x4xb)
        multiply_0 = paddle._C_ops.multiply(greater_than_0, less_than_0)

        # pd_op.all: (1x9261x1xb) <- (1x9261x4xb)
        all_0 = paddle._C_ops.all(multiply_0, [-1], True)

        # pd_op.full: (1xf32) <- ()
        full_16 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x9261x4xf32) <- (1x9261x4xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(concat_4, full_16, float("1"), True)

        # pd_op.divide: (1x9261x4xf32) <- (1x9261x4xf32, 1x9261x4xf32)
        divide_3 = paddle._C_ops.divide(concat_4, scale_9)

        # pd_op.log: (1x9261x4xf32) <- (1x9261x4xf32)
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

        # pd_op.full_like: (1x9261x4xf32) <- (1x9261x4xf32, 1xf32)
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

        # pd_op.full_like: (1x9261x1xb) <- (1x9261x1xb, 1xf32)
        full_like_5 = paddle._C_ops.full_like(
            all_0, full_18, paddle.bool, paddle.framework._current_expected_place()
        )

        # pd_op.cast: (1x9261x1xf32) <- (1x9261x1xb)
        cast_4 = paddle._C_ops.cast(full_like_5, paddle.float32)

        # pd_op.cast: (1x9261x1xf32) <- (1x9261x1xb)
        cast_5 = paddle._C_ops.cast(all_0, paddle.float32)

        # pd_op.add: (1x9261x4xf32) <- (1x9261x4xf32, xf32)
        add_7 = paddle._C_ops.add(full_like_3, full_like_4)

        # pd_op.add: (1x9261x4xf32) <- (1x9261x4xf32, 1x9261x1xf32)
        add_8 = paddle._C_ops.add(add_7, cast_4)

        # pd_op.add: (1x9261x4xf32) <- (1x9261x4xf32, 1x9261x4xf32)
        add_9 = paddle._C_ops.add(log_0, add_8)

        # pd_op.add: (1x9261x4xf32) <- (xf32, 1x9261x4xf32)
        add_10 = paddle._C_ops.add(assign_value__4, add_8)

        # pd_op.add: (1x9261x4xf32) <- (1x9261x1xf32, 1x9261x4xf32)
        add_11 = paddle._C_ops.add(cast_5, add_8)

        # pd_op.cast: (1x9261x4xb) <- (1x9261x4xf32)
        cast_6 = paddle._C_ops.cast(add_11, paddle.bool)

        # pd_op.where: (1x9261x4xf32) <- (1x9261x4xb, 1x9261x4xf32, 1x9261x4xf32)
        where_0 = paddle._C_ops.where(cast_6, add_9, add_10)

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

        # pd_op.full_like: (8x9261x256xf32) <- (8x9261x256xf32, 1xf32)
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

        # pd_op.full_like: (1x9261x1xb) <- (1x9261x1xb, 1xf32)
        full_like_8 = paddle._C_ops.full_like(
            all_0, full_18, paddle.bool, paddle.framework._current_expected_place()
        )

        # pd_op.cast: (1x9261x1xf32) <- (1x9261x1xb)
        cast_7 = paddle._C_ops.cast(full_like_8, paddle.float32)

        # pd_op.cast: (1x9261x1xf32) <- (1x9261x1xb)
        cast_8 = paddle._C_ops.cast(all_0, paddle.float32)

        # pd_op.add: (8x9261x256xf32) <- (8x9261x256xf32, xf32)
        add_12 = paddle._C_ops.add(full_like_6, full_like_7)

        # pd_op.add: (8x9261x256xf32) <- (8x9261x256xf32, 1x9261x1xf32)
        add_0 = paddle._C_ops.add(add_12, cast_7)

        # pd_op.add: (8x9261x256xf32) <- (8x9261x256xf32, 8x9261x256xf32)
        add_1 = paddle._C_ops.add(data_0, add_0)

        # pd_op.add: (8x9261x256xf32) <- (xf32, 8x9261x256xf32)
        add_2 = paddle._C_ops.add(assign_value__0, add_0)

        # pd_op.add: (8x9261x256xf32) <- (1x9261x1xf32, 8x9261x256xf32)
        add_13 = paddle._C_ops.add(cast_8, add_0)

        # pd_op.cast: (8x9261x256xb) <- (8x9261x256xf32)
        cast_0 = paddle._C_ops.cast(add_13, paddle.bool)

        # pd_op.where: (8x9261x256xf32) <- (8x9261x256xb, 8x9261x256xf32, 8x9261x256xf32)
        where_1 = paddle._C_ops.where(cast_0, add_1, add_2)

        # pd_op.matmul: (8x9261x256xf32) <- (8x9261x256xf32, 256x256xf32)
        matmul_0 = paddle._C_ops.matmul(where_1, parameter_11, False, False)

        # pd_op.add: (8x9261x256xf32) <- (8x9261x256xf32, 256xf32)
        add_3 = paddle._C_ops.add(matmul_0, parameter_10)

        # pd_op.layer_norm: (8x9261x256xf32, 8x9261xf32, 8x9261xf32) <- (8x9261x256xf32, 256xf32, 256xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_3, parameter_9, parameter_8, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (8x9261x4xf32) <- (8x9261x256xf32, 256x4xf32)
        matmul_1 = paddle._C_ops.matmul(layer_norm_0, parameter_7, False, False)

        # pd_op.add: (8x9261x4xf32) <- (8x9261x4xf32, 4xf32)
        add_4 = paddle._C_ops.add(matmul_1, parameter_6)

        # pd_op.matmul: (8x9261x256xf32) <- (8x9261x256xf32, 256x256xf32)
        matmul_2 = paddle._C_ops.matmul(layer_norm_0, parameter_5, False, False)

        # pd_op.add: (8x9261x256xf32) <- (8x9261x256xf32, 256xf32)
        add_14 = paddle._C_ops.add(matmul_2, parameter_4)

        # pd_op.relu: (8x9261x256xf32) <- (8x9261x256xf32)
        relu_0 = paddle._C_ops.relu(add_14)

        # pd_op.matmul: (8x9261x256xf32) <- (8x9261x256xf32, 256x256xf32)
        matmul_3 = paddle._C_ops.matmul(relu_0, parameter_3, False, False)

        # pd_op.add: (8x9261x256xf32) <- (8x9261x256xf32, 256xf32)
        add_15 = paddle._C_ops.add(matmul_3, parameter_2)

        # pd_op.relu: (8x9261x256xf32) <- (8x9261x256xf32)
        relu_1 = paddle._C_ops.relu(add_15)

        # pd_op.matmul: (8x9261x4xf32) <- (8x9261x256xf32, 256x4xf32)
        matmul_4 = paddle._C_ops.matmul(relu_1, parameter_1, False, False)

        # pd_op.add: (8x9261x4xf32) <- (8x9261x4xf32, 4xf32)
        add_5 = paddle._C_ops.add(matmul_4, parameter_0)

        # pd_op.add: (8x9261x4xf32) <- (8x9261x4xf32, 1x9261x4xf32)
        add_6 = paddle._C_ops.add(add_5, where_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [-1]

        # pd_op.max: (8x9261xf32) <- (8x9261x4xf32, 1xi64)
        max_0 = paddle._C_ops.max(add_4, full_int_array_4, False)

        # pd_op.full: (1xi32) <- ()
        full_20 = paddle._C_ops.full(
            [1], float("300"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.topk: (8x300xf32, 8x300xi64) <- (8x9261xf32, 1xi32)
        topk_0, topk_1 = (lambda x, f: f(x))(
            paddle._C_ops.topk(max_0, full_20, 1, True, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.full: (1xf64) <- ()
        full_21 = paddle._C_ops.full(
            [1], float("8"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (8xi64) <- (1xf64, 1xf64, 1xf64)
        arange_3 = paddle.arange(full_0, full_21, full_2, dtype="int64")

        # pd_op.unsqueeze: (8x1xi64) <- (8xi64, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(arange_3, full_int_array_4)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [1, 300]

        # pd_op.tile: (8x300xi64) <- (8x1xi64, 2xi64)
        tile_0 = paddle._C_ops.tile(unsqueeze_3, full_int_array_5)

        # builtin.combine: ([8x300xi64, 8x300xi64]) <- (8x300xi64, 8x300xi64)
        combine_10 = [tile_0, topk_1]

        # pd_op.stack: (8x300x2xi64) <- ([8x300xi64, 8x300xi64])
        stack_0 = paddle._C_ops.stack(combine_10, -1)

        # pd_op.gather_nd: (8x300x4xf32) <- (8x9261x4xf32, 8x300x2xi64)
        gather_nd_1 = paddle._C_ops.gather_nd(add_6, stack_0)

        # pd_op.sigmoid: (8x300x4xf32) <- (8x300x4xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(gather_nd_1)

        # builtin.combine: ([8x200x4xf32, 8x300x4xf32]) <- (8x200x4xf32, 8x300x4xf32)
        combine_11 = [data_2, gather_nd_1]

        # pd_op.concat: (8x500x4xf32) <- ([8x200x4xf32, 8x300x4xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_11, full_13)

        # pd_op.share_data_: (8x500x4xf32) <- (8x500x4xf32)
        share_data__1 = concat_5.detach()

        # pd_op.gather_nd: (8x300x4xf32) <- (8x9261x4xf32, 8x300x2xi64)
        gather_nd_0 = paddle._C_ops.gather_nd(add_4, stack_0)

        # pd_op.gather_nd: (8x300x256xf32) <- (8x9261x256xf32, 8x300x2xi64)
        gather_nd_2 = paddle._C_ops.gather_nd(layer_norm_0, stack_0)

        # pd_op.share_data_: (8x300x256xf32) <- (8x300x256xf32)
        share_data__0 = gather_nd_2.detach()

        # builtin.combine: ([8x200x256xf32, 8x300x256xf32]) <- (8x200x256xf32, 8x300x256xf32)
        combine_12 = [data_1, share_data__0]

        # pd_op.concat: (8x500x256xf32) <- ([8x200x256xf32, 8x300x256xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_12, full_13)
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
