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
        parameter_20,
        parameter_21,
        parameter_22,
        parameter_23,
        parameter_24,
        parameter_25,
        parameter_26,
        parameter_27,
        parameter_28,
        parameter_29,
        parameter_30,
        parameter_31,
        parameter_32,
        parameter_33,
        parameter_34,
        parameter_35,
        parameter_36,
        parameter_37,
        parameter_38,
        parameter_39,
        parameter_40,
        parameter_41,
        parameter_42,
        parameter_43,
        parameter_44,
        parameter_45,
        parameter_46,
        parameter_47,
        parameter_48,
        parameter_49,
        parameter_50,
        parameter_51,
        parameter_52,
        parameter_53,
        parameter_54,
        parameter_55,
        parameter_56,
        parameter_57,
        parameter_58,
        parameter_59,
        parameter_60,
        parameter_61,
        parameter_62,
        parameter_63,
        parameter_64,
        parameter_65,
        parameter_66,
        parameter_67,
        parameter_68,
        parameter_69,
        parameter_70,
        parameter_71,
        data_0,
        data_1,
        data_2,
        data_3,
    ):
        # pd_op.conv2d: (8x256x-1x80xf32) <- (8x-1x-1x80xf32, 256x32x5x5xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_3, parameter_71, [1, 1], [2, 2], "EXPLICIT", [1, 1], 8, "NCHW"
        )
        del parameter_71

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_70, full_int_array_0)
        del parameter_70

        # pd_op.add: (8x256x-1x80xf32) <- (8x256x-1x80xf32, 1x256x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_0, reshape_0)

        # pd_op.full: (xf64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_0,
            [],
            paddle.float64,
            [float("0.964706")],
            paddle.framework._current_expected_place(),
        )
        del full_0

        # pd_op.cast: (xf32) <- (xf64)
        cast_0 = paddle._C_ops.cast(assign_value__0, paddle.float32)
        del assign_value__0

        # pd_op.shape64: (4xi64) <- (8x256x-1x80xf32)
        shape64_0 = paddle._C_ops.shape64(add_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_3 = full_int_array_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_4 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_5 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_6 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_7 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_8 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_9 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_10 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_11 = full_int_array_2

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_0

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_0 = [slice_1, full_1, full_1, full_1]
        del slice_1

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_0 = paddle._C_ops.uniform(
            stack_0,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_0

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_1 = paddle._C_ops.add(cast_0, uniform_0)
        del uniform_0

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_0 = paddle._C_ops.floor(add_1)
        del add_1

        # pd_op.divide: (8x256x-1x80xf32) <- (8x256x-1x80xf32, xf32)
        divide_0 = paddle._C_ops.divide(add_0, cast_0)

        # pd_op.multiply: (8x256x-1x80xf32) <- (8x256x-1x80xf32, -1x1x1x1xf32)
        multiply_0 = paddle._C_ops.multiply(divide_0, floor_0)

        # pd_op.add: (8x256x-1x80xf32) <- (8x-1x-1x80xf32, 8x256x-1x80xf32)
        add_2 = paddle._C_ops.add(data_3, multiply_0)
        del data_3

        # pd_op.flatten: (8x256x-1xf32) <- (8x256x-1x80xf32)
        flatten_0 = paddle._C_ops.flatten(add_2, 2, 3)

        # pd_op.transpose: (8x-1x256xf32) <- (8x256x-1xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_0, [0, 2, 1])
        del flatten_0

        # pd_op.layer_norm: (8x-1x256xf32, 8x-1xf32, 8x-1xf32) <- (8x-1x256xf32, 256xf32, 256xf32)
        layer_norm_1, layer_norm_2, layer_norm_3 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_0, parameter_69, parameter_68, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_68, parameter_69

        # pd_op.matmul: (8x-1x1024xf32) <- (8x-1x256xf32, 256x1024xf32)
        matmul_0 = paddle._C_ops.matmul(layer_norm_1, parameter_67, False, False)
        del parameter_67

        # pd_op.add: (8x-1x1024xf32) <- (8x-1x1024xf32, 1024xf32)
        add_3 = paddle._C_ops.add(matmul_0, parameter_66)
        del parameter_66

        # pd_op.gelu: (8x-1x1024xf32) <- (8x-1x1024xf32)
        gelu_0 = paddle._C_ops.gelu(add_3, False)

        # pd_op.matmul: (8x-1x256xf32) <- (8x-1x1024xf32, 1024x256xf32)
        matmul_1 = paddle._C_ops.matmul(gelu_0, parameter_65, False, False)
        del parameter_65

        # pd_op.add: (8x-1x256xf32) <- (8x-1x256xf32, 256xf32)
        add_4 = paddle._C_ops.add(matmul_1, parameter_64)
        del parameter_64

        # pd_op.full: (xf64) <- ()
        full_4 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__1 = paddle._C_ops.assign_value_(
            full_4,
            [],
            paddle.float64,
            [float("0.964706")],
            paddle.framework._current_expected_place(),
        )
        del full_4

        # pd_op.cast: (xf32) <- (xf64)
        cast_1 = paddle._C_ops.cast(assign_value__1, paddle.float32)
        del assign_value__1

        # pd_op.shape64: (3xi64) <- (8x-1x256xf32)
        shape64_1 = paddle._C_ops.shape64(add_4)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_1

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_1 = [slice_2, full_1, full_1]
        del slice_2

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_1, 0)
        del combine_1

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_1 = paddle._C_ops.uniform(
            stack_1,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_1

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_5 = paddle._C_ops.add(cast_1, uniform_1)
        del uniform_1

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_1 = paddle._C_ops.floor(add_5)
        del add_5

        # pd_op.divide: (8x-1x256xf32) <- (8x-1x256xf32, xf32)
        divide_1 = paddle._C_ops.divide(add_4, cast_1)

        # pd_op.multiply: (8x-1x256xf32) <- (8x-1x256xf32, -1x1x1xf32)
        multiply_1 = paddle._C_ops.multiply(divide_1, floor_1)

        # pd_op.add: (8x-1x256xf32) <- (8x-1x256xf32, 8x-1x256xf32)
        add_6 = paddle._C_ops.add(layer_norm_1, multiply_1)

        # pd_op.layer_norm: (8x-1x256xf32, 8x-1xf32, 8x-1xf32) <- (8x-1x256xf32, 256xf32, 256xf32)
        layer_norm_4, layer_norm_5, layer_norm_6 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_6, parameter_63, parameter_62, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_62, parameter_63

        # pd_op.transpose: (8x256x-1xf32) <- (8x-1x256xf32)
        transpose_1 = paddle._C_ops.transpose(layer_norm_4, [0, 2, 1])
        del layer_norm_4

        # pd_op.full: (xi64) <- ()
        full_5 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_6 = paddle._C_ops.full(
            [], float("80"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_2 = [full_5, data_1, data_2, full_6]
        del data_1, data_2

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_2 = paddle._C_ops.stack(combine_2, 0)
        del combine_2

        # pd_op.reshape: (8x-1x-1x80xf32) <- (8x256x-1xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(transpose_1, stack_2)
        del stack_2

        # pd_op.shape64: (4xi64) <- (8x-1x-1x80xf32)
        shape64_2 = paddle._C_ops.shape64(reshape_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_12 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_13 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_14 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_15 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_16 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_17 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_18 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_19 = full_int_array_3

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_2

        # pd_op.shape64: (4xi64) <- (8x-1x-1x80xf32)
        shape64_3 = paddle._C_ops.shape64(reshape_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_20 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_21 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_22 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_23 = full_int_array_4

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del shape64_3

        # pd_op.conv2d: (8x256x-1x80xf32) <- (8x-1x-1x80xf32, 256x32x5x5xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            reshape_1, parameter_61, [1, 1], [2, 2], "EXPLICIT", [1, 1], 8, "NCHW"
        )
        del parameter_61

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(parameter_60, full_int_array_0)
        del parameter_60

        # pd_op.add: (8x256x-1x80xf32) <- (8x256x-1x80xf32, 1x256x1x1xf32)
        add_7 = paddle._C_ops.add(conv2d_1, reshape_2)

        # pd_op.full: (xf64) <- ()
        full_7 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__2 = paddle._C_ops.assign_value_(
            full_7,
            [],
            paddle.float64,
            [float("0.958824")],
            paddle.framework._current_expected_place(),
        )
        del full_7

        # pd_op.cast: (xf32) <- (xf64)
        cast_2 = paddle._C_ops.cast(assign_value__2, paddle.float32)
        del assign_value__2

        # pd_op.shape64: (4xi64) <- (8x256x-1x80xf32)
        shape64_4 = paddle._C_ops.shape64(add_7)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            shape64_4, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_4

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_3 = [slice_5, full_1, full_1, full_1]
        del slice_5

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_3 = paddle._C_ops.stack(combine_3, 0)
        del combine_3

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_2 = paddle._C_ops.uniform(
            stack_3,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_3

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_8 = paddle._C_ops.add(cast_2, uniform_2)
        del uniform_2

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_2 = paddle._C_ops.floor(add_8)
        del add_8

        # pd_op.divide: (8x256x-1x80xf32) <- (8x256x-1x80xf32, xf32)
        divide_2 = paddle._C_ops.divide(add_7, cast_2)

        # pd_op.multiply: (8x256x-1x80xf32) <- (8x256x-1x80xf32, -1x1x1x1xf32)
        multiply_2 = paddle._C_ops.multiply(divide_2, floor_2)

        # pd_op.add: (8x256x-1x80xf32) <- (8x-1x-1x80xf32, 8x256x-1x80xf32)
        add_9 = paddle._C_ops.add(reshape_1, multiply_2)

        # pd_op.flatten: (8x256x-1xf32) <- (8x256x-1x80xf32)
        flatten_1 = paddle._C_ops.flatten(add_9, 2, 3)

        # pd_op.transpose: (8x-1x256xf32) <- (8x256x-1xf32)
        transpose_2 = paddle._C_ops.transpose(flatten_1, [0, 2, 1])
        del flatten_1

        # pd_op.layer_norm: (8x-1x256xf32, 8x-1xf32, 8x-1xf32) <- (8x-1x256xf32, 256xf32, 256xf32)
        layer_norm_7, layer_norm_8, layer_norm_9 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_2, parameter_59, parameter_58, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_58, parameter_59

        # pd_op.matmul: (8x-1x1024xf32) <- (8x-1x256xf32, 256x1024xf32)
        matmul_2 = paddle._C_ops.matmul(layer_norm_7, parameter_57, False, False)
        del parameter_57

        # pd_op.add: (8x-1x1024xf32) <- (8x-1x1024xf32, 1024xf32)
        add_10 = paddle._C_ops.add(matmul_2, parameter_56)
        del parameter_56

        # pd_op.gelu: (8x-1x1024xf32) <- (8x-1x1024xf32)
        gelu_1 = paddle._C_ops.gelu(add_10, False)

        # pd_op.matmul: (8x-1x256xf32) <- (8x-1x1024xf32, 1024x256xf32)
        matmul_3 = paddle._C_ops.matmul(gelu_1, parameter_55, False, False)
        del parameter_55

        # pd_op.add: (8x-1x256xf32) <- (8x-1x256xf32, 256xf32)
        add_11 = paddle._C_ops.add(matmul_3, parameter_54)
        del parameter_54

        # pd_op.full: (xf64) <- ()
        full_8 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__3 = paddle._C_ops.assign_value_(
            full_8,
            [],
            paddle.float64,
            [float("0.958824")],
            paddle.framework._current_expected_place(),
        )
        del full_8

        # pd_op.cast: (xf32) <- (xf64)
        cast_3 = paddle._C_ops.cast(assign_value__3, paddle.float32)
        del assign_value__3

        # pd_op.shape64: (3xi64) <- (8x-1x256xf32)
        shape64_5 = paddle._C_ops.shape64(add_11)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            shape64_5, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_5

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_4 = [slice_6, full_1, full_1]
        del slice_6

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_4 = paddle._C_ops.stack(combine_4, 0)
        del combine_4

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_3 = paddle._C_ops.uniform(
            stack_4,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_4

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_12 = paddle._C_ops.add(cast_3, uniform_3)
        del uniform_3

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_3 = paddle._C_ops.floor(add_12)
        del add_12

        # pd_op.divide: (8x-1x256xf32) <- (8x-1x256xf32, xf32)
        divide_3 = paddle._C_ops.divide(add_11, cast_3)

        # pd_op.multiply: (8x-1x256xf32) <- (8x-1x256xf32, -1x1x1xf32)
        multiply_3 = paddle._C_ops.multiply(divide_3, floor_3)

        # pd_op.add: (8x-1x256xf32) <- (8x-1x256xf32, 8x-1x256xf32)
        add_13 = paddle._C_ops.add(layer_norm_7, multiply_3)

        # pd_op.layer_norm: (8x-1x256xf32, 8x-1xf32, 8x-1xf32) <- (8x-1x256xf32, 256xf32, 256xf32)
        layer_norm_10, layer_norm_11, layer_norm_12 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_13, parameter_53, parameter_52, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_52, parameter_53

        # pd_op.transpose: (8x256x-1xf32) <- (8x-1x256xf32)
        transpose_3 = paddle._C_ops.transpose(layer_norm_10, [0, 2, 1])
        del layer_norm_10

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_5 = [full_5, slice_3, slice_4, full_6]
        del slice_3, slice_4

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_5 = paddle._C_ops.stack(combine_5, 0)
        del combine_5

        # pd_op.reshape: (8x-1x-1x80xf32) <- (8x256x-1xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_3, stack_5)
        del stack_5

        # pd_op.flatten: (8x-1x-1xf32) <- (8x-1x-1x80xf32)
        flatten_2 = paddle._C_ops.flatten(reshape_3, 2, 3)

        # pd_op.transpose: (8x-1x-1xf32) <- (8x-1x-1xf32)
        transpose_4 = paddle._C_ops.transpose(flatten_2, [0, 2, 1])
        del flatten_2

        # pd_op.matmul: (8x-1x768xf32) <- (8x-1x-1xf32, 256x768xf32)
        matmul_4 = paddle._C_ops.matmul(transpose_4, parameter_51, False, False)
        del parameter_51

        # pd_op.add: (8x-1x768xf32) <- (8x-1x768xf32, 768xf32)
        add_14 = paddle._C_ops.add(matmul_4, parameter_50)
        del parameter_50

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_5 = [0, -1, 3, 8, 32]

        # pd_op.reshape: (8x-1x3x8x32xf32) <- (8x-1x768xf32, 5xi64)
        reshape_4 = paddle._C_ops.reshape(add_14, full_int_array_5)

        # pd_op.transpose: (3x8x8x-1x32xf32) <- (8x-1x3x8x32xf32)
        transpose_5 = paddle._C_ops.transpose(reshape_4, [2, 0, 3, 1, 4])
        del reshape_4

        # pd_op.slice: (8x8x-1x32xf32) <- (3x8x8x-1x32xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            transpose_5, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (8x8x-1x32xf32) <- (3x8x8x-1x32xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            transpose_5, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (8x8x-1x32xf32) <- (3x8x8x-1x32xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            transpose_5, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.transpose: (8x8x32x-1xf32) <- (8x8x-1x32xf32)
        transpose_6 = paddle._C_ops.transpose(slice_8, [0, 1, 3, 2])
        del slice_8

        # pd_op.matmul: (8x8x-1x-1xf32) <- (8x8x-1x32xf32, 8x8x32x-1xf32)
        matmul_5 = paddle._C_ops.matmul(slice_7, transpose_6, False, False)

        # pd_op.full: (1xf32) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_24 = full_9

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_25 = full_9

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_26 = full_9

        # pd_op.scale: (8x8x-1x-1xf32) <- (8x8x-1x-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(matmul_5, full_9, float("0"), True)
        del matmul_5

        # pd_op.softmax: (8x8x-1x-1xf32) <- (8x8x-1x-1xf32)
        softmax_0 = paddle._C_ops.softmax(scale_0, -1)
        del scale_0

        # pd_op.matmul: (8x8x-1x32xf32) <- (8x8x-1x-1xf32, 8x8x-1x32xf32)
        matmul_6 = paddle._C_ops.matmul(softmax_0, slice_9, False, False)

        # pd_op.transpose: (8x-1x8x32xf32) <- (8x8x-1x32xf32)
        transpose_7 = paddle._C_ops.transpose(matmul_6, [0, 2, 1, 3])
        del matmul_6

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_6 = [0, -1, 256]

        # pd_op.reshape: (8x-1x256xf32) <- (8x-1x8x32xf32, 3xi64)
        reshape_5 = paddle._C_ops.reshape(transpose_7, full_int_array_6)

        # pd_op.matmul: (8x-1x256xf32) <- (8x-1x256xf32, 256x256xf32)
        matmul_7 = paddle._C_ops.matmul(reshape_5, parameter_49, False, False)
        del parameter_49

        # pd_op.add: (8x-1x256xf32) <- (8x-1x256xf32, 256xf32)
        add_15 = paddle._C_ops.add(matmul_7, parameter_48)
        del parameter_48

        # pd_op.full: (xf64) <- ()
        full_10 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__4 = paddle._C_ops.assign_value_(
            full_10,
            [],
            paddle.float64,
            [float("0.952941")],
            paddle.framework._current_expected_place(),
        )
        del full_10

        # pd_op.cast: (xf32) <- (xf64)
        cast_4 = paddle._C_ops.cast(assign_value__4, paddle.float32)
        del assign_value__4

        # pd_op.shape64: (3xi64) <- (8x-1x256xf32)
        shape64_6 = paddle._C_ops.shape64(add_15)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            shape64_6, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_6

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_6 = [slice_10, full_1, full_1]
        del slice_10

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_6 = paddle._C_ops.stack(combine_6, 0)
        del combine_6

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_4 = paddle._C_ops.uniform(
            stack_6,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_6

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_16 = paddle._C_ops.add(cast_4, uniform_4)
        del uniform_4

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_4 = paddle._C_ops.floor(add_16)
        del add_16

        # pd_op.divide: (8x-1x256xf32) <- (8x-1x256xf32, xf32)
        divide_4 = paddle._C_ops.divide(add_15, cast_4)

        # pd_op.multiply: (8x-1x256xf32) <- (8x-1x256xf32, -1x1x1xf32)
        multiply_4 = paddle._C_ops.multiply(divide_4, floor_4)

        # pd_op.add: (8x-1x256xf32) <- (8x-1x-1xf32, 8x-1x256xf32)
        add_17 = paddle._C_ops.add(transpose_4, multiply_4)

        # pd_op.layer_norm: (8x-1x256xf32, 8x-1xf32, 8x-1xf32) <- (8x-1x256xf32, 256xf32, 256xf32)
        layer_norm_13, layer_norm_14, layer_norm_15 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_17, parameter_47, parameter_46, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_46, parameter_47

        # pd_op.matmul: (8x-1x1024xf32) <- (8x-1x256xf32, 256x1024xf32)
        matmul_8 = paddle._C_ops.matmul(layer_norm_13, parameter_45, False, False)
        del parameter_45

        # pd_op.add: (8x-1x1024xf32) <- (8x-1x1024xf32, 1024xf32)
        add_18 = paddle._C_ops.add(matmul_8, parameter_44)
        del parameter_44

        # pd_op.gelu: (8x-1x1024xf32) <- (8x-1x1024xf32)
        gelu_2 = paddle._C_ops.gelu(add_18, False)

        # pd_op.matmul: (8x-1x256xf32) <- (8x-1x1024xf32, 1024x256xf32)
        matmul_9 = paddle._C_ops.matmul(gelu_2, parameter_43, False, False)
        del parameter_43

        # pd_op.add: (8x-1x256xf32) <- (8x-1x256xf32, 256xf32)
        add_19 = paddle._C_ops.add(matmul_9, parameter_42)
        del parameter_42

        # pd_op.full: (xf64) <- ()
        full_11 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__5 = paddle._C_ops.assign_value_(
            full_11,
            [],
            paddle.float64,
            [float("0.952941")],
            paddle.framework._current_expected_place(),
        )
        del full_11

        # pd_op.cast: (xf32) <- (xf64)
        cast_5 = paddle._C_ops.cast(assign_value__5, paddle.float32)
        del assign_value__5

        # pd_op.shape64: (3xi64) <- (8x-1x256xf32)
        shape64_7 = paddle._C_ops.shape64(add_19)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            shape64_7, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_7

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_7 = [slice_11, full_1, full_1]
        del slice_11

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_7 = paddle._C_ops.stack(combine_7, 0)
        del combine_7

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_5 = paddle._C_ops.uniform(
            stack_7,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_7

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_20 = paddle._C_ops.add(cast_5, uniform_5)
        del uniform_5

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_5 = paddle._C_ops.floor(add_20)
        del add_20

        # pd_op.divide: (8x-1x256xf32) <- (8x-1x256xf32, xf32)
        divide_5 = paddle._C_ops.divide(add_19, cast_5)

        # pd_op.multiply: (8x-1x256xf32) <- (8x-1x256xf32, -1x1x1xf32)
        multiply_5 = paddle._C_ops.multiply(divide_5, floor_5)

        # pd_op.add: (8x-1x256xf32) <- (8x-1x256xf32, 8x-1x256xf32)
        add_21 = paddle._C_ops.add(layer_norm_13, multiply_5)

        # pd_op.layer_norm: (8x-1x256xf32, 8x-1xf32, 8x-1xf32) <- (8x-1x256xf32, 256xf32, 256xf32)
        layer_norm_16, layer_norm_17, layer_norm_18 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_21, parameter_41, parameter_40, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_40, parameter_41

        # pd_op.matmul: (8x-1x768xf32) <- (8x-1x256xf32, 256x768xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_16, parameter_39, False, False)
        del parameter_39

        # pd_op.add: (8x-1x768xf32) <- (8x-1x768xf32, 768xf32)
        add_22 = paddle._C_ops.add(matmul_10, parameter_38)
        del parameter_38

        # pd_op.reshape: (8x-1x3x8x32xf32) <- (8x-1x768xf32, 5xi64)
        reshape_6 = paddle._C_ops.reshape(add_22, full_int_array_5)

        # pd_op.transpose: (3x8x8x-1x32xf32) <- (8x-1x3x8x32xf32)
        transpose_8 = paddle._C_ops.transpose(reshape_6, [2, 0, 3, 1, 4])
        del reshape_6

        # pd_op.slice: (8x8x-1x32xf32) <- (3x8x8x-1x32xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            transpose_8, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (8x8x-1x32xf32) <- (3x8x8x-1x32xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            transpose_8, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (8x8x-1x32xf32) <- (3x8x8x-1x32xf32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            transpose_8, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.transpose: (8x8x32x-1xf32) <- (8x8x-1x32xf32)
        transpose_9 = paddle._C_ops.transpose(slice_13, [0, 1, 3, 2])
        del slice_13

        # pd_op.matmul: (8x8x-1x-1xf32) <- (8x8x-1x32xf32, 8x8x32x-1xf32)
        matmul_11 = paddle._C_ops.matmul(slice_12, transpose_9, False, False)

        # pd_op.scale: (8x8x-1x-1xf32) <- (8x8x-1x-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(matmul_11, full_9, float("0"), True)
        del matmul_11

        # pd_op.softmax: (8x8x-1x-1xf32) <- (8x8x-1x-1xf32)
        softmax_1 = paddle._C_ops.softmax(scale_1, -1)
        del scale_1

        # pd_op.matmul: (8x8x-1x32xf32) <- (8x8x-1x-1xf32, 8x8x-1x32xf32)
        matmul_12 = paddle._C_ops.matmul(softmax_1, slice_14, False, False)

        # pd_op.transpose: (8x-1x8x32xf32) <- (8x8x-1x32xf32)
        transpose_10 = paddle._C_ops.transpose(matmul_12, [0, 2, 1, 3])
        del matmul_12

        # pd_op.reshape: (8x-1x256xf32) <- (8x-1x8x32xf32, 3xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_10, full_int_array_6)

        # pd_op.matmul: (8x-1x256xf32) <- (8x-1x256xf32, 256x256xf32)
        matmul_13 = paddle._C_ops.matmul(reshape_7, parameter_37, False, False)
        del parameter_37

        # pd_op.add: (8x-1x256xf32) <- (8x-1x256xf32, 256xf32)
        add_23 = paddle._C_ops.add(matmul_13, parameter_36)
        del parameter_36

        # pd_op.full: (xf64) <- ()
        full_12 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__6 = paddle._C_ops.assign_value_(
            full_12,
            [],
            paddle.float64,
            [float("0.947059")],
            paddle.framework._current_expected_place(),
        )
        del full_12

        # pd_op.cast: (xf32) <- (xf64)
        cast_6 = paddle._C_ops.cast(assign_value__6, paddle.float32)
        del assign_value__6

        # pd_op.shape64: (3xi64) <- (8x-1x256xf32)
        shape64_8 = paddle._C_ops.shape64(add_23)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            shape64_8, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_8

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_8 = [slice_15, full_1, full_1]
        del slice_15

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_8 = paddle._C_ops.stack(combine_8, 0)
        del combine_8

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_6 = paddle._C_ops.uniform(
            stack_8,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_8

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_24 = paddle._C_ops.add(cast_6, uniform_6)
        del uniform_6

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_6 = paddle._C_ops.floor(add_24)
        del add_24

        # pd_op.divide: (8x-1x256xf32) <- (8x-1x256xf32, xf32)
        divide_6 = paddle._C_ops.divide(add_23, cast_6)

        # pd_op.multiply: (8x-1x256xf32) <- (8x-1x256xf32, -1x1x1xf32)
        multiply_6 = paddle._C_ops.multiply(divide_6, floor_6)

        # pd_op.add: (8x-1x256xf32) <- (8x-1x256xf32, 8x-1x256xf32)
        add_25 = paddle._C_ops.add(layer_norm_16, multiply_6)

        # pd_op.layer_norm: (8x-1x256xf32, 8x-1xf32, 8x-1xf32) <- (8x-1x256xf32, 256xf32, 256xf32)
        layer_norm_19, layer_norm_20, layer_norm_21 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_25, parameter_35, parameter_34, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_34, parameter_35

        # pd_op.matmul: (8x-1x1024xf32) <- (8x-1x256xf32, 256x1024xf32)
        matmul_14 = paddle._C_ops.matmul(layer_norm_19, parameter_33, False, False)
        del parameter_33

        # pd_op.add: (8x-1x1024xf32) <- (8x-1x1024xf32, 1024xf32)
        add_26 = paddle._C_ops.add(matmul_14, parameter_32)
        del parameter_32

        # pd_op.gelu: (8x-1x1024xf32) <- (8x-1x1024xf32)
        gelu_3 = paddle._C_ops.gelu(add_26, False)

        # pd_op.matmul: (8x-1x256xf32) <- (8x-1x1024xf32, 1024x256xf32)
        matmul_15 = paddle._C_ops.matmul(gelu_3, parameter_31, False, False)
        del parameter_31

        # pd_op.add: (8x-1x256xf32) <- (8x-1x256xf32, 256xf32)
        add_27 = paddle._C_ops.add(matmul_15, parameter_30)
        del parameter_30

        # pd_op.full: (xf64) <- ()
        full_13 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__7 = paddle._C_ops.assign_value_(
            full_13,
            [],
            paddle.float64,
            [float("0.947059")],
            paddle.framework._current_expected_place(),
        )
        del full_13

        # pd_op.cast: (xf32) <- (xf64)
        cast_7 = paddle._C_ops.cast(assign_value__7, paddle.float32)
        del assign_value__7

        # pd_op.shape64: (3xi64) <- (8x-1x256xf32)
        shape64_9 = paddle._C_ops.shape64(add_27)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            shape64_9, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_9

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_9 = [slice_16, full_1, full_1]
        del slice_16

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_9 = paddle._C_ops.stack(combine_9, 0)
        del combine_9

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_7 = paddle._C_ops.uniform(
            stack_9,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_9

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_28 = paddle._C_ops.add(cast_7, uniform_7)
        del uniform_7

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_7 = paddle._C_ops.floor(add_28)
        del add_28

        # pd_op.divide: (8x-1x256xf32) <- (8x-1x256xf32, xf32)
        divide_7 = paddle._C_ops.divide(add_27, cast_7)

        # pd_op.multiply: (8x-1x256xf32) <- (8x-1x256xf32, -1x1x1xf32)
        multiply_7 = paddle._C_ops.multiply(divide_7, floor_7)

        # pd_op.add: (8x-1x256xf32) <- (8x-1x256xf32, 8x-1x256xf32)
        add_29 = paddle._C_ops.add(layer_norm_19, multiply_7)

        # pd_op.layer_norm: (8x-1x256xf32, 8x-1xf32, 8x-1xf32) <- (8x-1x256xf32, 256xf32, 256xf32)
        layer_norm_22, layer_norm_23, layer_norm_24 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_29, parameter_29, parameter_28, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_28, parameter_29

        # pd_op.matmul: (8x-1x768xf32) <- (8x-1x256xf32, 256x768xf32)
        matmul_16 = paddle._C_ops.matmul(layer_norm_22, parameter_27, False, False)
        del parameter_27

        # pd_op.add: (8x-1x768xf32) <- (8x-1x768xf32, 768xf32)
        add_30 = paddle._C_ops.add(matmul_16, parameter_26)
        del parameter_26

        # pd_op.reshape: (8x-1x3x8x32xf32) <- (8x-1x768xf32, 5xi64)
        reshape_8 = paddle._C_ops.reshape(add_30, full_int_array_5)

        # pd_op.transpose: (3x8x8x-1x32xf32) <- (8x-1x3x8x32xf32)
        transpose_11 = paddle._C_ops.transpose(reshape_8, [2, 0, 3, 1, 4])
        del reshape_8

        # pd_op.slice: (8x8x-1x32xf32) <- (3x8x8x-1x32xf32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            transpose_11, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (8x8x-1x32xf32) <- (3x8x8x-1x32xf32, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(
            transpose_11, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (8x8x-1x32xf32) <- (3x8x8x-1x32xf32, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(
            transpose_11, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.transpose: (8x8x32x-1xf32) <- (8x8x-1x32xf32)
        transpose_12 = paddle._C_ops.transpose(slice_18, [0, 1, 3, 2])
        del slice_18

        # pd_op.matmul: (8x8x-1x-1xf32) <- (8x8x-1x32xf32, 8x8x32x-1xf32)
        matmul_17 = paddle._C_ops.matmul(slice_17, transpose_12, False, False)

        # pd_op.scale: (8x8x-1x-1xf32) <- (8x8x-1x-1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(matmul_17, full_9, float("0"), True)
        del matmul_17

        # pd_op.softmax: (8x8x-1x-1xf32) <- (8x8x-1x-1xf32)
        softmax_2 = paddle._C_ops.softmax(scale_2, -1)
        del scale_2

        # pd_op.matmul: (8x8x-1x32xf32) <- (8x8x-1x-1xf32, 8x8x-1x32xf32)
        matmul_18 = paddle._C_ops.matmul(softmax_2, slice_19, False, False)

        # pd_op.transpose: (8x-1x8x32xf32) <- (8x8x-1x32xf32)
        transpose_13 = paddle._C_ops.transpose(matmul_18, [0, 2, 1, 3])
        del matmul_18

        # pd_op.reshape: (8x-1x256xf32) <- (8x-1x8x32xf32, 3xi64)
        reshape_9 = paddle._C_ops.reshape(transpose_13, full_int_array_6)

        # pd_op.matmul: (8x-1x256xf32) <- (8x-1x256xf32, 256x256xf32)
        matmul_19 = paddle._C_ops.matmul(reshape_9, parameter_25, False, False)
        del parameter_25

        # pd_op.add: (8x-1x256xf32) <- (8x-1x256xf32, 256xf32)
        add_31 = paddle._C_ops.add(matmul_19, parameter_24)
        del parameter_24

        # pd_op.full: (xf64) <- ()
        full_14 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__8 = paddle._C_ops.assign_value_(
            full_14,
            [],
            paddle.float64,
            [float("0.941176")],
            paddle.framework._current_expected_place(),
        )
        del full_14

        # pd_op.cast: (xf32) <- (xf64)
        cast_8 = paddle._C_ops.cast(assign_value__8, paddle.float32)
        del assign_value__8

        # pd_op.shape64: (3xi64) <- (8x-1x256xf32)
        shape64_10 = paddle._C_ops.shape64(add_31)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(
            shape64_10, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_10

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_10 = [slice_20, full_1, full_1]
        del slice_20

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_10 = paddle._C_ops.stack(combine_10, 0)
        del combine_10

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_8 = paddle._C_ops.uniform(
            stack_10,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_10

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_32 = paddle._C_ops.add(cast_8, uniform_8)
        del uniform_8

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_8 = paddle._C_ops.floor(add_32)
        del add_32

        # pd_op.divide: (8x-1x256xf32) <- (8x-1x256xf32, xf32)
        divide_8 = paddle._C_ops.divide(add_31, cast_8)

        # pd_op.multiply: (8x-1x256xf32) <- (8x-1x256xf32, -1x1x1xf32)
        multiply_8 = paddle._C_ops.multiply(divide_8, floor_8)

        # pd_op.add: (8x-1x256xf32) <- (8x-1x256xf32, 8x-1x256xf32)
        add_33 = paddle._C_ops.add(layer_norm_22, multiply_8)

        # pd_op.layer_norm: (8x-1x256xf32, 8x-1xf32, 8x-1xf32) <- (8x-1x256xf32, 256xf32, 256xf32)
        layer_norm_25, layer_norm_26, layer_norm_27 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_33, parameter_23, parameter_22, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_22, parameter_23

        # pd_op.matmul: (8x-1x1024xf32) <- (8x-1x256xf32, 256x1024xf32)
        matmul_20 = paddle._C_ops.matmul(layer_norm_25, parameter_21, False, False)
        del parameter_21

        # pd_op.add: (8x-1x1024xf32) <- (8x-1x1024xf32, 1024xf32)
        add_34 = paddle._C_ops.add(matmul_20, parameter_20)
        del parameter_20

        # pd_op.gelu: (8x-1x1024xf32) <- (8x-1x1024xf32)
        gelu_4 = paddle._C_ops.gelu(add_34, False)

        # pd_op.matmul: (8x-1x256xf32) <- (8x-1x1024xf32, 1024x256xf32)
        matmul_21 = paddle._C_ops.matmul(gelu_4, parameter_19, False, False)
        del parameter_19

        # pd_op.add: (8x-1x256xf32) <- (8x-1x256xf32, 256xf32)
        add_35 = paddle._C_ops.add(matmul_21, parameter_18)
        del parameter_18

        # pd_op.full: (xf64) <- ()
        full_15 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__9 = paddle._C_ops.assign_value_(
            full_15,
            [],
            paddle.float64,
            [float("0.941176")],
            paddle.framework._current_expected_place(),
        )
        del full_15

        # pd_op.cast: (xf32) <- (xf64)
        cast_9 = paddle._C_ops.cast(assign_value__9, paddle.float32)
        del assign_value__9

        # pd_op.shape64: (3xi64) <- (8x-1x256xf32)
        shape64_11 = paddle._C_ops.shape64(add_35)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(
            shape64_11, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_11

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_11 = [slice_21, full_1, full_1]
        del slice_21

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_11 = paddle._C_ops.stack(combine_11, 0)
        del combine_11

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_9 = paddle._C_ops.uniform(
            stack_11,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_11

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_36 = paddle._C_ops.add(cast_9, uniform_9)
        del uniform_9

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_9 = paddle._C_ops.floor(add_36)
        del add_36

        # pd_op.divide: (8x-1x256xf32) <- (8x-1x256xf32, xf32)
        divide_9 = paddle._C_ops.divide(add_35, cast_9)

        # pd_op.multiply: (8x-1x256xf32) <- (8x-1x256xf32, -1x1x1xf32)
        multiply_9 = paddle._C_ops.multiply(divide_9, floor_9)

        # pd_op.add: (8x-1x256xf32) <- (8x-1x256xf32, 8x-1x256xf32)
        add_37 = paddle._C_ops.add(layer_norm_25, multiply_9)

        # pd_op.layer_norm: (8x-1x256xf32, 8x-1xf32, 8x-1xf32) <- (8x-1x256xf32, 256xf32, 256xf32)
        layer_norm_28, layer_norm_29, layer_norm_30 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_37, parameter_17, parameter_16, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_16, parameter_17

        # pd_op.matmul: (8x-1x768xf32) <- (8x-1x256xf32, 256x768xf32)
        matmul_22 = paddle._C_ops.matmul(layer_norm_28, parameter_15, False, False)
        del parameter_15

        # pd_op.add: (8x-1x768xf32) <- (8x-1x768xf32, 768xf32)
        add_38 = paddle._C_ops.add(matmul_22, parameter_14)
        del parameter_14

        # pd_op.reshape: (8x-1x3x8x32xf32) <- (8x-1x768xf32, 5xi64)
        reshape_10 = paddle._C_ops.reshape(add_38, full_int_array_5)
        del full_int_array_5

        # pd_op.transpose: (3x8x8x-1x32xf32) <- (8x-1x3x8x32xf32)
        transpose_14 = paddle._C_ops.transpose(reshape_10, [2, 0, 3, 1, 4])
        del reshape_10

        # pd_op.slice: (8x8x-1x32xf32) <- (3x8x8x-1x32xf32, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(
            transpose_14, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (8x8x-1x32xf32) <- (3x8x8x-1x32xf32, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(
            transpose_14, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (8x8x-1x32xf32) <- (3x8x8x-1x32xf32, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(
            transpose_14, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.transpose: (8x8x32x-1xf32) <- (8x8x-1x32xf32)
        transpose_15 = paddle._C_ops.transpose(slice_23, [0, 1, 3, 2])
        del slice_23

        # pd_op.matmul: (8x8x-1x-1xf32) <- (8x8x-1x32xf32, 8x8x32x-1xf32)
        matmul_23 = paddle._C_ops.matmul(slice_22, transpose_15, False, False)

        # pd_op.scale: (8x8x-1x-1xf32) <- (8x8x-1x-1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(matmul_23, full_9, float("0"), True)
        del matmul_23

        # pd_op.softmax: (8x8x-1x-1xf32) <- (8x8x-1x-1xf32)
        softmax_3 = paddle._C_ops.softmax(scale_3, -1)
        del scale_3

        # pd_op.matmul: (8x8x-1x32xf32) <- (8x8x-1x-1xf32, 8x8x-1x32xf32)
        matmul_24 = paddle._C_ops.matmul(softmax_3, slice_24, False, False)

        # pd_op.transpose: (8x-1x8x32xf32) <- (8x8x-1x32xf32)
        transpose_16 = paddle._C_ops.transpose(matmul_24, [0, 2, 1, 3])
        del matmul_24

        # pd_op.reshape: (8x-1x256xf32) <- (8x-1x8x32xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_16, full_int_array_6)
        del full_int_array_6

        # pd_op.matmul: (8x-1x256xf32) <- (8x-1x256xf32, 256x256xf32)
        matmul_25 = paddle._C_ops.matmul(reshape_11, parameter_13, False, False)
        del parameter_13

        # pd_op.add: (8x-1x256xf32) <- (8x-1x256xf32, 256xf32)
        add_39 = paddle._C_ops.add(matmul_25, parameter_12)
        del parameter_12

        # pd_op.full: (xf64) <- ()
        full_16 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__10 = paddle._C_ops.assign_value_(
            full_16,
            [],
            paddle.float64,
            [float("0.935294")],
            paddle.framework._current_expected_place(),
        )
        del full_16

        # pd_op.cast: (xf32) <- (xf64)
        cast_10 = paddle._C_ops.cast(assign_value__10, paddle.float32)
        del assign_value__10

        # pd_op.shape64: (3xi64) <- (8x-1x256xf32)
        shape64_12 = paddle._C_ops.shape64(add_39)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(
            shape64_12, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_12

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_12 = [slice_25, full_1, full_1]
        del slice_25

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_12 = paddle._C_ops.stack(combine_12, 0)
        del combine_12

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_10 = paddle._C_ops.uniform(
            stack_12,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_12

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_40 = paddle._C_ops.add(cast_10, uniform_10)
        del uniform_10

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_10 = paddle._C_ops.floor(add_40)
        del add_40

        # pd_op.divide: (8x-1x256xf32) <- (8x-1x256xf32, xf32)
        divide_10 = paddle._C_ops.divide(add_39, cast_10)

        # pd_op.multiply: (8x-1x256xf32) <- (8x-1x256xf32, -1x1x1xf32)
        multiply_10 = paddle._C_ops.multiply(divide_10, floor_10)

        # pd_op.add: (8x-1x256xf32) <- (8x-1x256xf32, 8x-1x256xf32)
        add_41 = paddle._C_ops.add(layer_norm_28, multiply_10)

        # pd_op.layer_norm: (8x-1x256xf32, 8x-1xf32, 8x-1xf32) <- (8x-1x256xf32, 256xf32, 256xf32)
        layer_norm_31, layer_norm_32, layer_norm_33 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_41, parameter_11, parameter_10, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_10, parameter_11

        # pd_op.matmul: (8x-1x1024xf32) <- (8x-1x256xf32, 256x1024xf32)
        matmul_26 = paddle._C_ops.matmul(layer_norm_31, parameter_9, False, False)
        del parameter_9

        # pd_op.add: (8x-1x1024xf32) <- (8x-1x1024xf32, 1024xf32)
        add_42 = paddle._C_ops.add(matmul_26, parameter_8)
        del parameter_8

        # pd_op.gelu: (8x-1x1024xf32) <- (8x-1x1024xf32)
        gelu_5 = paddle._C_ops.gelu(add_42, False)

        # pd_op.matmul: (8x-1x256xf32) <- (8x-1x1024xf32, 1024x256xf32)
        matmul_27 = paddle._C_ops.matmul(gelu_5, parameter_7, False, False)
        del parameter_7

        # pd_op.add: (8x-1x256xf32) <- (8x-1x256xf32, 256xf32)
        add_43 = paddle._C_ops.add(matmul_27, parameter_6)
        del parameter_6

        # pd_op.full: (xf64) <- ()
        full_17 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__11 = paddle._C_ops.assign_value_(
            full_17,
            [],
            paddle.float64,
            [float("0.935294")],
            paddle.framework._current_expected_place(),
        )
        del full_17

        # pd_op.cast: (xf32) <- (xf64)
        cast_11 = paddle._C_ops.cast(assign_value__11, paddle.float32)
        del assign_value__11

        # pd_op.shape64: (3xi64) <- (8x-1x256xf32)
        shape64_13 = paddle._C_ops.shape64(add_43)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(
            shape64_13, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del full_int_array_1, shape64_13

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_13 = [slice_26, full_1, full_1]
        del full_1, slice_26

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_13 = paddle._C_ops.stack(combine_13, 0)
        del combine_13

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_11 = paddle._C_ops.uniform(
            stack_13,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del full_2, full_3, stack_13

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_44 = paddle._C_ops.add(cast_11, uniform_11)
        del uniform_11

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_11 = paddle._C_ops.floor(add_44)
        del add_44

        # pd_op.divide: (8x-1x256xf32) <- (8x-1x256xf32, xf32)
        divide_11 = paddle._C_ops.divide(add_43, cast_11)

        # pd_op.multiply: (8x-1x256xf32) <- (8x-1x256xf32, -1x1x1xf32)
        multiply_11 = paddle._C_ops.multiply(divide_11, floor_11)

        # pd_op.add: (8x-1x256xf32) <- (8x-1x256xf32, 8x-1x256xf32)
        add_45 = paddle._C_ops.add(layer_norm_31, multiply_11)

        # pd_op.layer_norm: (8x-1x256xf32, 8x-1xf32, 8x-1xf32) <- (8x-1x256xf32, 256xf32, 256xf32)
        layer_norm_34, layer_norm_35, layer_norm_36 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_45, parameter_5, parameter_4, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_4, parameter_5

        # pd_op.shape64: (3xi64) <- (8x-1x256xf32)
        shape64_14 = paddle._C_ops.shape64(layer_norm_34)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(
            shape64_14, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del full_int_array_2, shape64_14

        # pd_op.transpose: (8x256x-1xf32) <- (8x-1x256xf32)
        transpose_17 = paddle._C_ops.transpose(layer_norm_34, [0, 2, 1])
        del layer_norm_34

        # pd_op.full: (xi64) <- ()
        full_18 = paddle._C_ops.full(
            [], float("256"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_14 = [full_5, full_18, data_0, full_6]
        del data_0, full_18, full_5, full_6

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_14 = paddle._C_ops.stack(combine_14, 0)
        del combine_14

        # pd_op.reshape: (8x256x-1x80xf32) <- (8x256x-1xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(transpose_17, stack_14)
        del stack_14

        # pd_op.conv2d: (8x384x-1x80xf32) <- (8x256x-1x80xf32, 384x256x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            reshape_12, parameter_3, [2, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_3

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(parameter_2, full_int_array_0)
        del full_int_array_0, parameter_2

        # pd_op.add: (8x384x-1x80xf32) <- (8x384x-1x80xf32, 1x384x1x1xf32)
        add_46 = paddle._C_ops.add(conv2d_2, reshape_13)

        # pd_op.shape64: (4xi64) <- (8x384x-1x80xf32)
        shape64_15 = paddle._C_ops.shape64(add_46)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_15, [0], full_int_array_3, full_int_array_4, [1], [0]
        )
        del full_int_array_3, full_int_array_4, shape64_15

        # pd_op.flatten: (8x384x-1xf32) <- (8x384x-1x80xf32)
        flatten_3 = paddle._C_ops.flatten(add_46, 2, 3)

        # pd_op.transpose: (8x-1x384xf32) <- (8x384x-1xf32)
        transpose_18 = paddle._C_ops.transpose(flatten_3, [0, 2, 1])
        del flatten_3

        # pd_op.layer_norm: (8x-1x384xf32, 8x-1xf32, 8x-1xf32) <- (8x-1x384xf32, 384xf32, 384xf32)
        layer_norm_0, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_18, parameter_1, parameter_0, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del (
            add_0,
            add_10,
            add_11,
            add_13,
            add_14,
            add_15,
            add_17,
            add_18,
            add_19,
            add_2,
            add_21,
            add_22,
            add_23,
            add_25,
            add_26,
            add_27,
            add_29,
            add_3,
            add_30,
            add_31,
            add_33,
            add_34,
            add_35,
            add_37,
            add_38,
            add_39,
            add_4,
            add_41,
            add_42,
            add_43,
            add_45,
            add_46,
            add_6,
            add_7,
            add_9,
            assign_0,
            assign_1,
            assign_10,
            assign_11,
            assign_12,
            assign_13,
            assign_14,
            assign_15,
            assign_16,
            assign_17,
            assign_18,
            assign_19,
            assign_2,
            assign_20,
            assign_21,
            assign_22,
            assign_23,
            assign_24,
            assign_25,
            assign_26,
            assign_3,
            assign_4,
            assign_5,
            assign_6,
            assign_7,
            assign_8,
            assign_9,
            cast_0,
            cast_1,
            cast_10,
            cast_11,
            cast_2,
            cast_3,
            cast_4,
            cast_5,
            cast_6,
            cast_7,
            cast_8,
            cast_9,
            conv2d_0,
            conv2d_1,
            conv2d_2,
            divide_0,
            divide_1,
            divide_10,
            divide_11,
            divide_2,
            divide_3,
            divide_4,
            divide_5,
            divide_6,
            divide_7,
            divide_8,
            divide_9,
            floor_0,
            floor_1,
            floor_10,
            floor_11,
            floor_2,
            floor_3,
            floor_4,
            floor_5,
            floor_6,
            floor_7,
            floor_8,
            floor_9,
            full_9,
            gelu_0,
            gelu_1,
            gelu_2,
            gelu_3,
            gelu_4,
            gelu_5,
            layer_norm_1,
            layer_norm_11,
            layer_norm_12,
            layer_norm_13,
            layer_norm_14,
            layer_norm_15,
            layer_norm_16,
            layer_norm_17,
            layer_norm_18,
            layer_norm_19,
            layer_norm_2,
            layer_norm_20,
            layer_norm_21,
            layer_norm_22,
            layer_norm_23,
            layer_norm_24,
            layer_norm_25,
            layer_norm_26,
            layer_norm_27,
            layer_norm_28,
            layer_norm_29,
            layer_norm_3,
            layer_norm_30,
            layer_norm_31,
            layer_norm_32,
            layer_norm_33,
            layer_norm_35,
            layer_norm_36,
            layer_norm_5,
            layer_norm_6,
            layer_norm_7,
            layer_norm_8,
            layer_norm_9,
            matmul_0,
            matmul_1,
            matmul_10,
            matmul_13,
            matmul_14,
            matmul_15,
            matmul_16,
            matmul_19,
            matmul_2,
            matmul_20,
            matmul_21,
            matmul_22,
            matmul_25,
            matmul_26,
            matmul_27,
            matmul_3,
            matmul_4,
            matmul_7,
            matmul_8,
            matmul_9,
            multiply_0,
            multiply_1,
            multiply_10,
            multiply_11,
            multiply_2,
            multiply_3,
            multiply_4,
            multiply_5,
            multiply_6,
            multiply_7,
            multiply_8,
            multiply_9,
            parameter_0,
            parameter_1,
            reshape_0,
            reshape_1,
            reshape_11,
            reshape_12,
            reshape_13,
            reshape_2,
            reshape_3,
            reshape_5,
            reshape_7,
            reshape_9,
            slice_12,
            slice_14,
            slice_17,
            slice_19,
            slice_22,
            slice_24,
            slice_7,
            slice_9,
            softmax_0,
            softmax_1,
            softmax_2,
            softmax_3,
            transpose_0,
            transpose_1,
            transpose_10,
            transpose_11,
            transpose_12,
            transpose_13,
            transpose_14,
            transpose_15,
            transpose_16,
            transpose_17,
            transpose_18,
            transpose_2,
            transpose_3,
            transpose_4,
            transpose_5,
            transpose_6,
            transpose_7,
            transpose_8,
            transpose_9,
        )

        return layer_norm_0, slice_0
