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
        data_0,
        data_1,
        data_2,
        data_3,
        data_4,
    ):
        # pd_op.add: (1x400x256xf32) <- (1x400x256xf32, 1x400x256xf32)
        add_0 = paddle._C_ops.add(data_1, data_0)

        # pd_op.layer_norm: (1x400x256xf32, 1x400xf32, 1x400xf32) <- (1x400x256xf32, 256xf32, 256xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_0, parameter_17, parameter_16, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.add: (1x400x256xf32) <- (1x400x256xf32, 1x400x256xf32)
        add_1 = paddle._C_ops.add(layer_norm_0, data_2)

        # pd_op.matmul: (1x7581x256xf32) <- (1x7581x256xf32, 256x256xf32)
        matmul_0 = paddle._C_ops.matmul(data_4, parameter_15, False, False)

        # pd_op.add: (1x7581x256xf32) <- (1x7581x256xf32, 256xf32)
        add_2 = paddle._C_ops.add(matmul_0, parameter_14)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [1, 7581, 8, 32]

        # pd_op.reshape: (1x7581x8x32xf32) <- (1x7581x256xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(add_2, full_int_array_1)

        # pd_op.matmul: (1x400x192xf32) <- (1x400x256xf32, 256x192xf32)
        matmul_1 = paddle._C_ops.matmul(add_1, parameter_13, False, False)

        # pd_op.add: (1x400x192xf32) <- (1x400x192xf32, 192xf32)
        add_3 = paddle._C_ops.add(matmul_1, parameter_12)

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_2 = [1, 400, 8, 3, 4, 2]

        # pd_op.reshape: (1x400x8x3x4x2xf32) <- (1x400x192xf32, 6xi64)
        reshape_5 = paddle._C_ops.reshape(add_3, full_int_array_2)

        # pd_op.matmul: (1x400x96xf32) <- (1x400x256xf32, 256x96xf32)
        matmul_2 = paddle._C_ops.matmul(add_1, parameter_11, False, False)

        # pd_op.add: (1x400x96xf32) <- (1x400x96xf32, 96xf32)
        add_4 = paddle._C_ops.add(matmul_2, parameter_10)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [1, 400, 8, 12]

        # pd_op.reshape: (1x400x8x12xf32) <- (1x400x96xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(add_4, full_int_array_3)

        # pd_op.softmax: (1x400x8x12xf32) <- (1x400x8x12xf32)
        softmax_0 = paddle._C_ops.softmax(reshape_6, -1)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_4 = [1, 400, 8, 3, 4]

        # pd_op.reshape: (1x400x8x3x4xf32) <- (1x400x8x12xf32, 5xi64)
        reshape_7 = paddle._C_ops.reshape(softmax_0, full_int_array_4)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_5

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_4 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_3 = full_int_array_6

        # pd_op.slice: (1x400x1x2xf32) <- (1x400x1x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_3, [3], full_int_array_5, full_int_array_6, [1], []
        )

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_7 = [2, 4]

        # pd_op.unsqueeze: (1x400x1x1x1x2xf32) <- (1x400x1x2xf32, 2xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(slice_0, full_int_array_7)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.25"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x400x8x3x4x2xf32) <- (1x400x8x3x4x2xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(reshape_5, full_0, float("0"), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [2147483647]

        # pd_op.slice: (1x400x1x2xf32) <- (1x400x1x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_3, [3], full_int_array_6, full_int_array_8, [1], []
        )

        # pd_op.unsqueeze: (1x400x1x1x1x2xf32) <- (1x400x1x2xf32, 2xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(slice_1, full_int_array_7)

        # pd_op.multiply: (1x400x8x3x4x2xf32) <- (1x400x8x3x4x2xf32, 1x400x1x1x1x2xf32)
        multiply_1 = paddle._C_ops.multiply(scale_0, unsqueeze_1)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x400x8x3x4x2xf32) <- (1x400x8x3x4x2xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(multiply_1, full_1, float("0"), True)

        # pd_op.add: (1x400x8x3x4x2xf32) <- (1x400x1x1x1x2xf32, 1x400x8x3x4x2xf32)
        add_9 = paddle._C_ops.add(unsqueeze_0, scale_1)

        # pd_op.full: (3x2xi64) <- ()
        full_5 = paddle._C_ops.full(
            [3, 2], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (3x2xi64) <- (3x2xi64)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_5,
            [3, 2],
            paddle.int64,
            [
                float("76"),
                float("76"),
                float("38"),
                float("38"),
                float("19"),
                float("19"),
            ],
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (3xi64) <- ()
        full_6 = paddle._C_ops.full(
            [3], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (3xi64) <- (3xi64)
        assign_value__1 = paddle._C_ops.assign_value_(
            full_6,
            [3],
            paddle.int64,
            [float("0"), float("5776"), float("7220")],
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_9

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_9

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            assign_value__0, [0], full_int_array_5, full_int_array_9, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            slice_2, [0], full_int_array_5, full_int_array_9, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            slice_2, [0], full_int_array_9, full_int_array_6, [1], [0]
        )

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_2 = paddle._C_ops.multiply(slice_3, slice_4)

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            assign_value__0, [0], full_int_array_9, full_int_array_6, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            slice_5, [0], full_int_array_5, full_int_array_9, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            slice_5, [0], full_int_array_9, full_int_array_6, [1], [0]
        )

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_3 = paddle._C_ops.multiply(slice_6, slice_7)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_10 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_5 = full_int_array_10

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            assign_value__0, [0], full_int_array_6, full_int_array_10, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            slice_8, [0], full_int_array_5, full_int_array_9, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            slice_8, [0], full_int_array_9, full_int_array_6, [1], [0]
        )

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_4 = paddle._C_ops.multiply(slice_9, slice_10)

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_0 = [multiply_2, multiply_3, multiply_4]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_0, 0)

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split: ([-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32]) <- (1x7581x8x32xf32, 3xi64, 1xi32)
        split_3 = paddle._C_ops.split(reshape_4, stack_1, full_2)

        # builtin.split: (-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32) <- ([-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32])
        (
            split_0,
            split_1,
            split_2,
        ) = split_3

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x400x8x3x4x2xf32) <- (1x400x8x3x4x2xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(add_9, full_3, float("0"), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x400x8x3x4x2xf32) <- (1x400x8x3x4x2xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(scale_3, full_4, float("-1"), True)

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            assign_value__0, [0], full_int_array_5, full_int_array_9, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            slice_11, [0], full_int_array_5, full_int_array_9, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            slice_11, [0], full_int_array_9, full_int_array_6, [1], [0]
        )

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_4 = paddle._C_ops.flatten(split_0, 2, 3)

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_4, [0, 2, 1])

        # pd_op.full: (xi64) <- ()
        full_7 = paddle._C_ops.full(
            [], float("8"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_8 = paddle._C_ops.full(
            [], float("32"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_1 = [full_7, full_8, slice_12, slice_13]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_2 = paddle._C_ops.stack(combine_1, 0)

        # pd_op.reshape: (8x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(transpose_0, stack_2)

        # pd_op.slice: (1x400x8x4x2xf32) <- (1x400x8x3x4x2xf32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            scale_2, [3], full_int_array_5, full_int_array_9, [1], [3]
        )

        # pd_op.transpose: (1x8x400x4x2xf32) <- (1x400x8x4x2xf32)
        transpose_1 = paddle._C_ops.transpose(slice_14, [0, 2, 1, 3, 4])

        # pd_op.flatten: (8x400x4x2xf32) <- (1x8x400x4x2xf32)
        flatten_0 = paddle._C_ops.flatten(transpose_1, 0, 1)

        # pd_op.grid_sample: (8x32x400x4xf32) <- (8x32x-1x-1xf32, 8x400x4x2xf32)
        grid_sample_0 = paddle._C_ops.grid_sample(
            reshape_0, flatten_0, "bilinear", "zeros", False
        )

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            assign_value__0, [0], full_int_array_9, full_int_array_6, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            slice_15, [0], full_int_array_5, full_int_array_9, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            slice_15, [0], full_int_array_9, full_int_array_6, [1], [0]
        )

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_5 = paddle._C_ops.flatten(split_1, 2, 3)

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_2 = paddle._C_ops.transpose(flatten_5, [0, 2, 1])

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_2 = [full_7, full_8, slice_16, slice_17]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_3 = paddle._C_ops.stack(combine_2, 0)

        # pd_op.reshape: (8x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(transpose_2, stack_3)

        # pd_op.slice: (1x400x8x4x2xf32) <- (1x400x8x3x4x2xf32, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(
            scale_2, [3], full_int_array_9, full_int_array_6, [1], [3]
        )

        # pd_op.transpose: (1x8x400x4x2xf32) <- (1x400x8x4x2xf32)
        transpose_3 = paddle._C_ops.transpose(slice_18, [0, 2, 1, 3, 4])

        # pd_op.flatten: (8x400x4x2xf32) <- (1x8x400x4x2xf32)
        flatten_1 = paddle._C_ops.flatten(transpose_3, 0, 1)

        # pd_op.grid_sample: (8x32x400x4xf32) <- (8x32x-1x-1xf32, 8x400x4x2xf32)
        grid_sample_1 = paddle._C_ops.grid_sample(
            reshape_1, flatten_1, "bilinear", "zeros", False
        )

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(
            assign_value__0, [0], full_int_array_6, full_int_array_10, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(
            slice_19, [0], full_int_array_5, full_int_array_9, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(
            slice_19, [0], full_int_array_9, full_int_array_6, [1], [0]
        )

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_6 = paddle._C_ops.flatten(split_2, 2, 3)

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_4 = paddle._C_ops.transpose(flatten_6, [0, 2, 1])

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_3 = [full_7, full_8, slice_20, slice_21]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_4 = paddle._C_ops.stack(combine_3, 0)

        # pd_op.reshape: (8x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(transpose_4, stack_4)

        # pd_op.slice: (1x400x8x4x2xf32) <- (1x400x8x3x4x2xf32, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(
            scale_2, [3], full_int_array_6, full_int_array_10, [1], [3]
        )

        # pd_op.transpose: (1x8x400x4x2xf32) <- (1x400x8x4x2xf32)
        transpose_5 = paddle._C_ops.transpose(slice_22, [0, 2, 1, 3, 4])

        # pd_op.flatten: (8x400x4x2xf32) <- (1x8x400x4x2xf32)
        flatten_2 = paddle._C_ops.flatten(transpose_5, 0, 1)

        # pd_op.grid_sample: (8x32x400x4xf32) <- (8x32x-1x-1xf32, 8x400x4x2xf32)
        grid_sample_2 = paddle._C_ops.grid_sample(
            reshape_2, flatten_2, "bilinear", "zeros", False
        )

        # pd_op.transpose: (1x8x400x3x4xf32) <- (1x400x8x3x4xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_7, [0, 2, 1, 3, 4])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_11 = [8, 1, 400, 12]

        # pd_op.reshape: (8x1x400x12xf32) <- (1x8x400x3x4xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_6, full_int_array_11)

        # builtin.combine: ([8x32x400x4xf32, 8x32x400x4xf32, 8x32x400x4xf32]) <- (8x32x400x4xf32, 8x32x400x4xf32, 8x32x400x4xf32)
        combine_4 = [grid_sample_0, grid_sample_1, grid_sample_2]

        # pd_op.stack: (8x32x400x3x4xf32) <- ([8x32x400x4xf32, 8x32x400x4xf32, 8x32x400x4xf32])
        stack_0 = paddle._C_ops.stack(combine_4, -2)

        # pd_op.flatten: (8x32x400x12xf32) <- (8x32x400x3x4xf32)
        flatten_3 = paddle._C_ops.flatten(stack_0, 3, 4)

        # pd_op.multiply: (8x32x400x12xf32) <- (8x32x400x12xf32, 8x1x400x12xf32)
        multiply_0 = paddle._C_ops.multiply(flatten_3, reshape_3)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [-1]

        # pd_op.sum: (8x32x400xf32) <- (8x32x400x12xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(multiply_0, full_int_array_0, None, False)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_12 = [1, 256, 400]

        # pd_op.reshape: (1x256x400xf32) <- (8x32x400xf32, 3xi64)
        reshape_8 = paddle._C_ops.reshape(sum_0, full_int_array_12)

        # pd_op.transpose: (1x400x256xf32) <- (1x256x400xf32)
        transpose_7 = paddle._C_ops.transpose(reshape_8, [0, 2, 1])

        # pd_op.matmul: (1x400x256xf32) <- (1x400x256xf32, 256x256xf32)
        matmul_3 = paddle._C_ops.matmul(transpose_7, parameter_9, False, False)

        # pd_op.add: (1x400x256xf32) <- (1x400x256xf32, 256xf32)
        add_5 = paddle._C_ops.add(matmul_3, parameter_8)

        # pd_op.add: (1x400x256xf32) <- (1x400x256xf32, 1x400x256xf32)
        add_6 = paddle._C_ops.add(layer_norm_0, add_5)

        # pd_op.layer_norm: (1x400x256xf32, 1x400xf32, 1x400xf32) <- (1x400x256xf32, 256xf32, 256xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_6, parameter_7, parameter_6, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (1x400x1024xf32) <- (1x400x256xf32, 256x1024xf32)
        matmul_4 = paddle._C_ops.matmul(layer_norm_3, parameter_5, False, False)

        # pd_op.add: (1x400x1024xf32) <- (1x400x1024xf32, 1024xf32)
        add_10 = paddle._C_ops.add(matmul_4, parameter_4)

        # pd_op.relu: (1x400x1024xf32) <- (1x400x1024xf32)
        relu_0 = paddle._C_ops.relu(add_10)

        # pd_op.matmul: (1x400x256xf32) <- (1x400x1024xf32, 1024x256xf32)
        matmul_5 = paddle._C_ops.matmul(relu_0, parameter_3, False, False)

        # pd_op.add: (1x400x256xf32) <- (1x400x256xf32, 256xf32)
        add_7 = paddle._C_ops.add(matmul_5, parameter_2)

        # pd_op.add: (1x400x256xf32) <- (1x400x256xf32, 1x400x256xf32)
        add_8 = paddle._C_ops.add(layer_norm_3, add_7)

        # pd_op.layer_norm: (1x400x256xf32, 1x400xf32, 1x400xf32) <- (1x400x256xf32, 256xf32, 256xf32)
        layer_norm_8, layer_norm_6, layer_norm_7 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_8, parameter_1, parameter_0, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        return (
            add_0,
            layer_norm_0,
            layer_norm_1,
            layer_norm_2,
            add_1,
            matmul_0,
            add_2,
            matmul_1,
            add_3,
            matmul_2,
            add_4,
            softmax_0,
            unsqueeze_0,
            full_0,
            scale_0,
            unsqueeze_1,
            full_1,
            scale_1,
            full_2,
            split_0,
            split_1,
            split_2,
            full_3,
            full_4,
            scale_2,
            transpose_0,
            reshape_0,
            assign_0,
            assign_1,
            transpose_1,
            flatten_0,
            grid_sample_0,
            transpose_2,
            reshape_1,
            assign_2,
            assign_3,
            transpose_3,
            flatten_1,
            grid_sample_1,
            transpose_4,
            reshape_2,
            assign_4,
            assign_5,
            transpose_5,
            flatten_2,
            grid_sample_2,
            transpose_6,
            reshape_3,
            stack_0,
            flatten_3,
            multiply_0,
            full_int_array_0,
            sum_0,
            transpose_7,
            matmul_3,
            add_5,
            add_6,
            layer_norm_3,
            layer_norm_4,
            layer_norm_5,
            matmul_4,
            relu_0,
            matmul_5,
            add_7,
            add_8,
            layer_norm_6,
            layer_norm_7,
            layer_norm_8,
        )
