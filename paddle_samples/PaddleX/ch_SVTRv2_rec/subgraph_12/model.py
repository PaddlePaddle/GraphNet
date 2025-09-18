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
        data_0,
    ):
        # pd_op.conv2d: (8x128x8x80xf32) <- (8x128x8x80xf32, 128x32x5x5xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_63, [1, 1], [2, 2], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del parameter_63

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_62, full_int_array_0)
        del parameter_62

        # pd_op.add: (8x128x8x80xf32) <- (8x128x8x80xf32, 1x128x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_0, reshape_1)

        # pd_op.add: (8x128x8x80xf32) <- (8x128x8x80xf32, 8x128x8x80xf32)
        add_1 = paddle._C_ops.add(data_0, add_0)
        del data_0

        # pd_op.flatten: (8x128x640xf32) <- (8x128x8x80xf32)
        flatten_0 = paddle._C_ops.flatten(add_1, 2, 3)

        # pd_op.transpose: (8x640x128xf32) <- (8x128x640xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_0, [0, 2, 1])
        del flatten_0

        # pd_op.layer_norm: (8x640x128xf32, 8x640xf32, 8x640xf32) <- (8x640x128xf32, 128xf32, 128xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_0, parameter_61, parameter_60, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_60, parameter_61

        # pd_op.matmul: (8x640x512xf32) <- (8x640x128xf32, 128x512xf32)
        matmul_0 = paddle._C_ops.matmul(layer_norm_0, parameter_59, False, False)
        del parameter_59

        # pd_op.add: (8x640x512xf32) <- (8x640x512xf32, 512xf32)
        add_2 = paddle._C_ops.add(matmul_0, parameter_58)
        del parameter_58

        # pd_op.gelu: (8x640x512xf32) <- (8x640x512xf32)
        gelu_0 = paddle._C_ops.gelu(add_2, False)

        # pd_op.matmul: (8x640x128xf32) <- (8x640x512xf32, 512x128xf32)
        matmul_1 = paddle._C_ops.matmul(gelu_0, parameter_57, False, False)
        del parameter_57

        # pd_op.add: (8x640x128xf32) <- (8x640x128xf32, 128xf32)
        add_3 = paddle._C_ops.add(matmul_1, parameter_56)
        del parameter_56

        # pd_op.add: (8x640x128xf32) <- (8x640x128xf32, 8x640x128xf32)
        add_4 = paddle._C_ops.add(layer_norm_0, add_3)

        # pd_op.layer_norm: (8x640x128xf32, 8x640xf32, 8x640xf32) <- (8x640x128xf32, 128xf32, 128xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_4, parameter_55, parameter_54, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_54, parameter_55

        # pd_op.transpose: (8x128x640xf32) <- (8x640x128xf32)
        transpose_1 = paddle._C_ops.transpose(layer_norm_3, [0, 2, 1])
        del layer_norm_3

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [0, 128, 8, 80]

        # pd_op.reshape: (8x128x8x80xf32) <- (8x128x640xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(transpose_1, full_int_array_1)

        # pd_op.conv2d: (8x128x8x80xf32) <- (8x128x8x80xf32, 128x32x5x5xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            reshape_2, parameter_53, [1, 1], [2, 2], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del parameter_53

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(parameter_52, full_int_array_0)
        del parameter_52

        # pd_op.add: (8x128x8x80xf32) <- (8x128x8x80xf32, 1x128x1x1xf32)
        add_5 = paddle._C_ops.add(conv2d_1, reshape_3)

        # pd_op.full: (xf64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_0,
            [],
            paddle.float64,
            [float("0.994118")],
            paddle.framework._current_expected_place(),
        )
        del full_0

        # pd_op.cast: (xf32) <- (xf64)
        cast_0 = paddle._C_ops.cast(assign_value__0, paddle.float32)
        del assign_value__0

        # pd_op.shape64: (4xi64) <- (8x128x8x80xf32)
        shape64_0 = paddle._C_ops.shape64(add_5)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_0

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_0 = [slice_0, full_1, full_1, full_1]
        del slice_0

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
        add_6 = paddle._C_ops.add(cast_0, uniform_0)
        del uniform_0

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_0 = paddle._C_ops.floor(add_6)
        del add_6

        # pd_op.divide: (8x128x8x80xf32) <- (8x128x8x80xf32, xf32)
        divide_0 = paddle._C_ops.divide(add_5, cast_0)

        # pd_op.multiply: (8x128x8x80xf32) <- (8x128x8x80xf32, -1x1x1x1xf32)
        multiply_0 = paddle._C_ops.multiply(divide_0, floor_0)

        # pd_op.add: (8x128x8x80xf32) <- (8x128x8x80xf32, 8x128x8x80xf32)
        add_7 = paddle._C_ops.add(reshape_2, multiply_0)

        # pd_op.flatten: (8x128x640xf32) <- (8x128x8x80xf32)
        flatten_1 = paddle._C_ops.flatten(add_7, 2, 3)

        # pd_op.transpose: (8x640x128xf32) <- (8x128x640xf32)
        transpose_2 = paddle._C_ops.transpose(flatten_1, [0, 2, 1])
        del flatten_1

        # pd_op.layer_norm: (8x640x128xf32, 8x640xf32, 8x640xf32) <- (8x640x128xf32, 128xf32, 128xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_2, parameter_51, parameter_50, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_50, parameter_51

        # pd_op.matmul: (8x640x512xf32) <- (8x640x128xf32, 128x512xf32)
        matmul_2 = paddle._C_ops.matmul(layer_norm_6, parameter_49, False, False)
        del parameter_49

        # pd_op.add: (8x640x512xf32) <- (8x640x512xf32, 512xf32)
        add_8 = paddle._C_ops.add(matmul_2, parameter_48)
        del parameter_48

        # pd_op.gelu: (8x640x512xf32) <- (8x640x512xf32)
        gelu_1 = paddle._C_ops.gelu(add_8, False)

        # pd_op.matmul: (8x640x128xf32) <- (8x640x512xf32, 512x128xf32)
        matmul_3 = paddle._C_ops.matmul(gelu_1, parameter_47, False, False)
        del parameter_47

        # pd_op.add: (8x640x128xf32) <- (8x640x128xf32, 128xf32)
        add_9 = paddle._C_ops.add(matmul_3, parameter_46)
        del parameter_46

        # pd_op.full: (xf64) <- ()
        full_4 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__1 = paddle._C_ops.assign_value_(
            full_4,
            [],
            paddle.float64,
            [float("0.994118")],
            paddle.framework._current_expected_place(),
        )
        del full_4

        # pd_op.cast: (xf32) <- (xf64)
        cast_1 = paddle._C_ops.cast(assign_value__1, paddle.float32)
        del assign_value__1

        # pd_op.shape64: (3xi64) <- (8x640x128xf32)
        shape64_1 = paddle._C_ops.shape64(add_9)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_1

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_1 = [slice_1, full_1, full_1]
        del slice_1

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
        add_10 = paddle._C_ops.add(cast_1, uniform_1)
        del uniform_1

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_1 = paddle._C_ops.floor(add_10)
        del add_10

        # pd_op.divide: (8x640x128xf32) <- (8x640x128xf32, xf32)
        divide_1 = paddle._C_ops.divide(add_9, cast_1)

        # pd_op.multiply: (8x640x128xf32) <- (8x640x128xf32, -1x1x1xf32)
        multiply_1 = paddle._C_ops.multiply(divide_1, floor_1)

        # pd_op.add: (8x640x128xf32) <- (8x640x128xf32, 8x640x128xf32)
        add_11 = paddle._C_ops.add(layer_norm_6, multiply_1)

        # pd_op.layer_norm: (8x640x128xf32, 8x640xf32, 8x640xf32) <- (8x640x128xf32, 128xf32, 128xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_11, parameter_45, parameter_44, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_44, parameter_45

        # pd_op.transpose: (8x128x640xf32) <- (8x640x128xf32)
        transpose_3 = paddle._C_ops.transpose(layer_norm_9, [0, 2, 1])
        del layer_norm_9

        # pd_op.reshape: (8x128x8x80xf32) <- (8x128x640xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(transpose_3, full_int_array_1)

        # pd_op.conv2d: (8x128x8x80xf32) <- (8x128x8x80xf32, 128x32x5x5xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            reshape_4, parameter_43, [1, 1], [2, 2], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del parameter_43

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(parameter_42, full_int_array_0)
        del parameter_42

        # pd_op.add: (8x128x8x80xf32) <- (8x128x8x80xf32, 1x128x1x1xf32)
        add_12 = paddle._C_ops.add(conv2d_2, reshape_5)

        # pd_op.full: (xf64) <- ()
        full_5 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__2 = paddle._C_ops.assign_value_(
            full_5,
            [],
            paddle.float64,
            [float("0.988235")],
            paddle.framework._current_expected_place(),
        )
        del full_5

        # pd_op.cast: (xf32) <- (xf64)
        cast_2 = paddle._C_ops.cast(assign_value__2, paddle.float32)
        del assign_value__2

        # pd_op.shape64: (4xi64) <- (8x128x8x80xf32)
        shape64_2 = paddle._C_ops.shape64(add_12)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_2

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_2 = [slice_2, full_1, full_1, full_1]
        del slice_2

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_2 = paddle._C_ops.stack(combine_2, 0)
        del combine_2

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_2 = paddle._C_ops.uniform(
            stack_2,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_2

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_13 = paddle._C_ops.add(cast_2, uniform_2)
        del uniform_2

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_2 = paddle._C_ops.floor(add_13)
        del add_13

        # pd_op.divide: (8x128x8x80xf32) <- (8x128x8x80xf32, xf32)
        divide_2 = paddle._C_ops.divide(add_12, cast_2)

        # pd_op.multiply: (8x128x8x80xf32) <- (8x128x8x80xf32, -1x1x1x1xf32)
        multiply_2 = paddle._C_ops.multiply(divide_2, floor_2)

        # pd_op.add: (8x128x8x80xf32) <- (8x128x8x80xf32, 8x128x8x80xf32)
        add_14 = paddle._C_ops.add(reshape_4, multiply_2)

        # pd_op.flatten: (8x128x640xf32) <- (8x128x8x80xf32)
        flatten_2 = paddle._C_ops.flatten(add_14, 2, 3)

        # pd_op.transpose: (8x640x128xf32) <- (8x128x640xf32)
        transpose_4 = paddle._C_ops.transpose(flatten_2, [0, 2, 1])
        del flatten_2

        # pd_op.layer_norm: (8x640x128xf32, 8x640xf32, 8x640xf32) <- (8x640x128xf32, 128xf32, 128xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_4, parameter_41, parameter_40, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_40, parameter_41

        # pd_op.matmul: (8x640x512xf32) <- (8x640x128xf32, 128x512xf32)
        matmul_4 = paddle._C_ops.matmul(layer_norm_12, parameter_39, False, False)
        del parameter_39

        # pd_op.add: (8x640x512xf32) <- (8x640x512xf32, 512xf32)
        add_15 = paddle._C_ops.add(matmul_4, parameter_38)
        del parameter_38

        # pd_op.gelu: (8x640x512xf32) <- (8x640x512xf32)
        gelu_2 = paddle._C_ops.gelu(add_15, False)

        # pd_op.matmul: (8x640x128xf32) <- (8x640x512xf32, 512x128xf32)
        matmul_5 = paddle._C_ops.matmul(gelu_2, parameter_37, False, False)
        del parameter_37

        # pd_op.add: (8x640x128xf32) <- (8x640x128xf32, 128xf32)
        add_16 = paddle._C_ops.add(matmul_5, parameter_36)
        del parameter_36

        # pd_op.full: (xf64) <- ()
        full_6 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__3 = paddle._C_ops.assign_value_(
            full_6,
            [],
            paddle.float64,
            [float("0.988235")],
            paddle.framework._current_expected_place(),
        )
        del full_6

        # pd_op.cast: (xf32) <- (xf64)
        cast_3 = paddle._C_ops.cast(assign_value__3, paddle.float32)
        del assign_value__3

        # pd_op.shape64: (3xi64) <- (8x640x128xf32)
        shape64_3 = paddle._C_ops.shape64(add_16)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_3

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_3 = [slice_3, full_1, full_1]
        del slice_3

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_3 = paddle._C_ops.stack(combine_3, 0)
        del combine_3

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_3 = paddle._C_ops.uniform(
            stack_3,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_3

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_17 = paddle._C_ops.add(cast_3, uniform_3)
        del uniform_3

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_3 = paddle._C_ops.floor(add_17)
        del add_17

        # pd_op.divide: (8x640x128xf32) <- (8x640x128xf32, xf32)
        divide_3 = paddle._C_ops.divide(add_16, cast_3)

        # pd_op.multiply: (8x640x128xf32) <- (8x640x128xf32, -1x1x1xf32)
        multiply_3 = paddle._C_ops.multiply(divide_3, floor_3)

        # pd_op.add: (8x640x128xf32) <- (8x640x128xf32, 8x640x128xf32)
        add_18 = paddle._C_ops.add(layer_norm_12, multiply_3)

        # pd_op.layer_norm: (8x640x128xf32, 8x640xf32, 8x640xf32) <- (8x640x128xf32, 128xf32, 128xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_18, parameter_35, parameter_34, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_34, parameter_35

        # pd_op.transpose: (8x128x640xf32) <- (8x640x128xf32)
        transpose_5 = paddle._C_ops.transpose(layer_norm_15, [0, 2, 1])
        del layer_norm_15

        # pd_op.reshape: (8x128x8x80xf32) <- (8x128x640xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(transpose_5, full_int_array_1)

        # pd_op.conv2d: (8x128x8x80xf32) <- (8x128x8x80xf32, 128x32x5x5xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            reshape_6, parameter_33, [1, 1], [2, 2], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del parameter_33

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(parameter_32, full_int_array_0)
        del parameter_32

        # pd_op.add: (8x128x8x80xf32) <- (8x128x8x80xf32, 1x128x1x1xf32)
        add_19 = paddle._C_ops.add(conv2d_3, reshape_7)

        # pd_op.full: (xf64) <- ()
        full_7 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__4 = paddle._C_ops.assign_value_(
            full_7,
            [],
            paddle.float64,
            [float("0.982353")],
            paddle.framework._current_expected_place(),
        )
        del full_7

        # pd_op.cast: (xf32) <- (xf64)
        cast_4 = paddle._C_ops.cast(assign_value__4, paddle.float32)
        del assign_value__4

        # pd_op.shape64: (4xi64) <- (8x128x8x80xf32)
        shape64_4 = paddle._C_ops.shape64(add_19)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            shape64_4, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_4

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_4 = [slice_4, full_1, full_1, full_1]
        del slice_4

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_4 = paddle._C_ops.stack(combine_4, 0)
        del combine_4

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_4 = paddle._C_ops.uniform(
            stack_4,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_4

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_20 = paddle._C_ops.add(cast_4, uniform_4)
        del uniform_4

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_4 = paddle._C_ops.floor(add_20)
        del add_20

        # pd_op.divide: (8x128x8x80xf32) <- (8x128x8x80xf32, xf32)
        divide_4 = paddle._C_ops.divide(add_19, cast_4)

        # pd_op.multiply: (8x128x8x80xf32) <- (8x128x8x80xf32, -1x1x1x1xf32)
        multiply_4 = paddle._C_ops.multiply(divide_4, floor_4)

        # pd_op.add: (8x128x8x80xf32) <- (8x128x8x80xf32, 8x128x8x80xf32)
        add_21 = paddle._C_ops.add(reshape_6, multiply_4)

        # pd_op.flatten: (8x128x640xf32) <- (8x128x8x80xf32)
        flatten_3 = paddle._C_ops.flatten(add_21, 2, 3)

        # pd_op.transpose: (8x640x128xf32) <- (8x128x640xf32)
        transpose_6 = paddle._C_ops.transpose(flatten_3, [0, 2, 1])
        del flatten_3

        # pd_op.layer_norm: (8x640x128xf32, 8x640xf32, 8x640xf32) <- (8x640x128xf32, 128xf32, 128xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_6, parameter_31, parameter_30, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_30, parameter_31

        # pd_op.matmul: (8x640x512xf32) <- (8x640x128xf32, 128x512xf32)
        matmul_6 = paddle._C_ops.matmul(layer_norm_18, parameter_29, False, False)
        del parameter_29

        # pd_op.add: (8x640x512xf32) <- (8x640x512xf32, 512xf32)
        add_22 = paddle._C_ops.add(matmul_6, parameter_28)
        del parameter_28

        # pd_op.gelu: (8x640x512xf32) <- (8x640x512xf32)
        gelu_3 = paddle._C_ops.gelu(add_22, False)

        # pd_op.matmul: (8x640x128xf32) <- (8x640x512xf32, 512x128xf32)
        matmul_7 = paddle._C_ops.matmul(gelu_3, parameter_27, False, False)
        del parameter_27

        # pd_op.add: (8x640x128xf32) <- (8x640x128xf32, 128xf32)
        add_23 = paddle._C_ops.add(matmul_7, parameter_26)
        del parameter_26

        # pd_op.full: (xf64) <- ()
        full_8 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__5 = paddle._C_ops.assign_value_(
            full_8,
            [],
            paddle.float64,
            [float("0.982353")],
            paddle.framework._current_expected_place(),
        )
        del full_8

        # pd_op.cast: (xf32) <- (xf64)
        cast_5 = paddle._C_ops.cast(assign_value__5, paddle.float32)
        del assign_value__5

        # pd_op.shape64: (3xi64) <- (8x640x128xf32)
        shape64_5 = paddle._C_ops.shape64(add_23)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            shape64_5, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_5

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_5 = [slice_5, full_1, full_1]
        del slice_5

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_5 = paddle._C_ops.stack(combine_5, 0)
        del combine_5

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_5 = paddle._C_ops.uniform(
            stack_5,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_5

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_24 = paddle._C_ops.add(cast_5, uniform_5)
        del uniform_5

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_5 = paddle._C_ops.floor(add_24)
        del add_24

        # pd_op.divide: (8x640x128xf32) <- (8x640x128xf32, xf32)
        divide_5 = paddle._C_ops.divide(add_23, cast_5)

        # pd_op.multiply: (8x640x128xf32) <- (8x640x128xf32, -1x1x1xf32)
        multiply_5 = paddle._C_ops.multiply(divide_5, floor_5)

        # pd_op.add: (8x640x128xf32) <- (8x640x128xf32, 8x640x128xf32)
        add_25 = paddle._C_ops.add(layer_norm_18, multiply_5)

        # pd_op.layer_norm: (8x640x128xf32, 8x640xf32, 8x640xf32) <- (8x640x128xf32, 128xf32, 128xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_25, parameter_25, parameter_24, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_24, parameter_25

        # pd_op.transpose: (8x128x640xf32) <- (8x640x128xf32)
        transpose_7 = paddle._C_ops.transpose(layer_norm_21, [0, 2, 1])
        del layer_norm_21

        # pd_op.reshape: (8x128x8x80xf32) <- (8x128x640xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(transpose_7, full_int_array_1)

        # pd_op.conv2d: (8x128x8x80xf32) <- (8x128x8x80xf32, 128x32x5x5xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            reshape_8, parameter_23, [1, 1], [2, 2], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del parameter_23

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(parameter_22, full_int_array_0)
        del parameter_22

        # pd_op.add: (8x128x8x80xf32) <- (8x128x8x80xf32, 1x128x1x1xf32)
        add_26 = paddle._C_ops.add(conv2d_4, reshape_9)

        # pd_op.full: (xf64) <- ()
        full_9 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__6 = paddle._C_ops.assign_value_(
            full_9,
            [],
            paddle.float64,
            [float("0.976471")],
            paddle.framework._current_expected_place(),
        )
        del full_9

        # pd_op.cast: (xf32) <- (xf64)
        cast_6 = paddle._C_ops.cast(assign_value__6, paddle.float32)
        del assign_value__6

        # pd_op.shape64: (4xi64) <- (8x128x8x80xf32)
        shape64_6 = paddle._C_ops.shape64(add_26)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            shape64_6, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_6

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_6 = [slice_6, full_1, full_1, full_1]
        del slice_6

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_6 = paddle._C_ops.stack(combine_6, 0)
        del combine_6

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_6 = paddle._C_ops.uniform(
            stack_6,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_6

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_27 = paddle._C_ops.add(cast_6, uniform_6)
        del uniform_6

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_6 = paddle._C_ops.floor(add_27)
        del add_27

        # pd_op.divide: (8x128x8x80xf32) <- (8x128x8x80xf32, xf32)
        divide_6 = paddle._C_ops.divide(add_26, cast_6)

        # pd_op.multiply: (8x128x8x80xf32) <- (8x128x8x80xf32, -1x1x1x1xf32)
        multiply_6 = paddle._C_ops.multiply(divide_6, floor_6)

        # pd_op.add: (8x128x8x80xf32) <- (8x128x8x80xf32, 8x128x8x80xf32)
        add_28 = paddle._C_ops.add(reshape_8, multiply_6)

        # pd_op.flatten: (8x128x640xf32) <- (8x128x8x80xf32)
        flatten_4 = paddle._C_ops.flatten(add_28, 2, 3)

        # pd_op.transpose: (8x640x128xf32) <- (8x128x640xf32)
        transpose_8 = paddle._C_ops.transpose(flatten_4, [0, 2, 1])
        del flatten_4

        # pd_op.layer_norm: (8x640x128xf32, 8x640xf32, 8x640xf32) <- (8x640x128xf32, 128xf32, 128xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_8, parameter_21, parameter_20, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_20, parameter_21

        # pd_op.matmul: (8x640x512xf32) <- (8x640x128xf32, 128x512xf32)
        matmul_8 = paddle._C_ops.matmul(layer_norm_24, parameter_19, False, False)
        del parameter_19

        # pd_op.add: (8x640x512xf32) <- (8x640x512xf32, 512xf32)
        add_29 = paddle._C_ops.add(matmul_8, parameter_18)
        del parameter_18

        # pd_op.gelu: (8x640x512xf32) <- (8x640x512xf32)
        gelu_4 = paddle._C_ops.gelu(add_29, False)

        # pd_op.matmul: (8x640x128xf32) <- (8x640x512xf32, 512x128xf32)
        matmul_9 = paddle._C_ops.matmul(gelu_4, parameter_17, False, False)
        del parameter_17

        # pd_op.add: (8x640x128xf32) <- (8x640x128xf32, 128xf32)
        add_30 = paddle._C_ops.add(matmul_9, parameter_16)
        del parameter_16

        # pd_op.full: (xf64) <- ()
        full_10 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__7 = paddle._C_ops.assign_value_(
            full_10,
            [],
            paddle.float64,
            [float("0.976471")],
            paddle.framework._current_expected_place(),
        )
        del full_10

        # pd_op.cast: (xf32) <- (xf64)
        cast_7 = paddle._C_ops.cast(assign_value__7, paddle.float32)
        del assign_value__7

        # pd_op.shape64: (3xi64) <- (8x640x128xf32)
        shape64_7 = paddle._C_ops.shape64(add_30)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            shape64_7, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_7

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_7 = [slice_7, full_1, full_1]
        del slice_7

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_7 = paddle._C_ops.stack(combine_7, 0)
        del combine_7

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_7 = paddle._C_ops.uniform(
            stack_7,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_7

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_31 = paddle._C_ops.add(cast_7, uniform_7)
        del uniform_7

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_7 = paddle._C_ops.floor(add_31)
        del add_31

        # pd_op.divide: (8x640x128xf32) <- (8x640x128xf32, xf32)
        divide_7 = paddle._C_ops.divide(add_30, cast_7)

        # pd_op.multiply: (8x640x128xf32) <- (8x640x128xf32, -1x1x1xf32)
        multiply_7 = paddle._C_ops.multiply(divide_7, floor_7)

        # pd_op.add: (8x640x128xf32) <- (8x640x128xf32, 8x640x128xf32)
        add_32 = paddle._C_ops.add(layer_norm_24, multiply_7)

        # pd_op.layer_norm: (8x640x128xf32, 8x640xf32, 8x640xf32) <- (8x640x128xf32, 128xf32, 128xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_32, parameter_15, parameter_14, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_14, parameter_15

        # pd_op.transpose: (8x128x640xf32) <- (8x640x128xf32)
        transpose_9 = paddle._C_ops.transpose(layer_norm_27, [0, 2, 1])
        del layer_norm_27

        # pd_op.reshape: (8x128x8x80xf32) <- (8x128x640xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(transpose_9, full_int_array_1)

        # pd_op.conv2d: (8x128x8x80xf32) <- (8x128x8x80xf32, 128x32x5x5xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            reshape_10, parameter_13, [1, 1], [2, 2], "EXPLICIT", [1, 1], 4, "NCHW"
        )
        del parameter_13

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_11 = paddle._C_ops.reshape(parameter_12, full_int_array_0)
        del parameter_12

        # pd_op.add: (8x128x8x80xf32) <- (8x128x8x80xf32, 1x128x1x1xf32)
        add_33 = paddle._C_ops.add(conv2d_5, reshape_11)

        # pd_op.full: (xf64) <- ()
        full_11 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__8 = paddle._C_ops.assign_value_(
            full_11,
            [],
            paddle.float64,
            [float("0.970588")],
            paddle.framework._current_expected_place(),
        )
        del full_11

        # pd_op.cast: (xf32) <- (xf64)
        cast_8 = paddle._C_ops.cast(assign_value__8, paddle.float32)
        del assign_value__8

        # pd_op.shape64: (4xi64) <- (8x128x8x80xf32)
        shape64_8 = paddle._C_ops.shape64(add_33)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            shape64_8, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del shape64_8

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_8 = [slice_8, full_1, full_1, full_1]
        del slice_8

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_8 = paddle._C_ops.stack(combine_8, 0)
        del combine_8

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_8 = paddle._C_ops.uniform(
            stack_8,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_8

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_34 = paddle._C_ops.add(cast_8, uniform_8)
        del uniform_8

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_8 = paddle._C_ops.floor(add_34)
        del add_34

        # pd_op.divide: (8x128x8x80xf32) <- (8x128x8x80xf32, xf32)
        divide_8 = paddle._C_ops.divide(add_33, cast_8)

        # pd_op.multiply: (8x128x8x80xf32) <- (8x128x8x80xf32, -1x1x1x1xf32)
        multiply_8 = paddle._C_ops.multiply(divide_8, floor_8)

        # pd_op.add: (8x128x8x80xf32) <- (8x128x8x80xf32, 8x128x8x80xf32)
        add_35 = paddle._C_ops.add(reshape_10, multiply_8)

        # pd_op.flatten: (8x128x640xf32) <- (8x128x8x80xf32)
        flatten_5 = paddle._C_ops.flatten(add_35, 2, 3)

        # pd_op.transpose: (8x640x128xf32) <- (8x128x640xf32)
        transpose_10 = paddle._C_ops.transpose(flatten_5, [0, 2, 1])
        del flatten_5

        # pd_op.layer_norm: (8x640x128xf32, 8x640xf32, 8x640xf32) <- (8x640x128xf32, 128xf32, 128xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_10, parameter_11, parameter_10, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_10, parameter_11

        # pd_op.matmul: (8x640x512xf32) <- (8x640x128xf32, 128x512xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_30, parameter_9, False, False)
        del parameter_9

        # pd_op.add: (8x640x512xf32) <- (8x640x512xf32, 512xf32)
        add_36 = paddle._C_ops.add(matmul_10, parameter_8)
        del parameter_8

        # pd_op.gelu: (8x640x512xf32) <- (8x640x512xf32)
        gelu_5 = paddle._C_ops.gelu(add_36, False)

        # pd_op.matmul: (8x640x128xf32) <- (8x640x512xf32, 512x128xf32)
        matmul_11 = paddle._C_ops.matmul(gelu_5, parameter_7, False, False)
        del parameter_7

        # pd_op.add: (8x640x128xf32) <- (8x640x128xf32, 128xf32)
        add_37 = paddle._C_ops.add(matmul_11, parameter_6)
        del parameter_6

        # pd_op.full: (xf64) <- ()
        full_12 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__9 = paddle._C_ops.assign_value_(
            full_12,
            [],
            paddle.float64,
            [float("0.970588")],
            paddle.framework._current_expected_place(),
        )
        del full_12

        # pd_op.cast: (xf32) <- (xf64)
        cast_9 = paddle._C_ops.cast(assign_value__9, paddle.float32)
        del assign_value__9

        # pd_op.shape64: (3xi64) <- (8x640x128xf32)
        shape64_9 = paddle._C_ops.shape64(add_37)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            shape64_9, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del full_int_array_2, full_int_array_3, shape64_9

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_9 = [slice_9, full_1, full_1]
        del full_1, slice_9

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_9 = paddle._C_ops.stack(combine_9, 0)
        del combine_9

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_9 = paddle._C_ops.uniform(
            stack_9,
            paddle.float32,
            full_2,
            full_3,
            0,
            paddle.framework._current_expected_place(),
        )
        del full_2, full_3, stack_9

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_38 = paddle._C_ops.add(cast_9, uniform_9)
        del uniform_9

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_9 = paddle._C_ops.floor(add_38)
        del add_38

        # pd_op.divide: (8x640x128xf32) <- (8x640x128xf32, xf32)
        divide_9 = paddle._C_ops.divide(add_37, cast_9)

        # pd_op.multiply: (8x640x128xf32) <- (8x640x128xf32, -1x1x1xf32)
        multiply_9 = paddle._C_ops.multiply(divide_9, floor_9)

        # pd_op.add: (8x640x128xf32) <- (8x640x128xf32, 8x640x128xf32)
        add_39 = paddle._C_ops.add(layer_norm_30, multiply_9)

        # pd_op.layer_norm: (8x640x128xf32, 8x640xf32, 8x640xf32) <- (8x640x128xf32, 128xf32, 128xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_39, parameter_5, parameter_4, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_4, parameter_5

        # pd_op.transpose: (8x128x640xf32) <- (8x640x128xf32)
        transpose_11 = paddle._C_ops.transpose(layer_norm_33, [0, 2, 1])
        del layer_norm_33

        # pd_op.reshape: (8x128x8x80xf32) <- (8x128x640xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(transpose_11, full_int_array_1)
        del full_int_array_1

        # pd_op.conv2d: (8x256x4x80xf32) <- (8x128x8x80xf32, 256x128x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            reshape_12, parameter_3, [2, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_3

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(parameter_2, full_int_array_0)
        del full_int_array_0, parameter_2

        # pd_op.add: (8x256x4x80xf32) <- (8x256x4x80xf32, 1x256x1x1xf32)
        add_40 = paddle._C_ops.add(conv2d_6, reshape_13)

        # pd_op.flatten: (8x256x320xf32) <- (8x256x4x80xf32)
        flatten_6 = paddle._C_ops.flatten(add_40, 2, 3)

        # pd_op.transpose: (8x320x256xf32) <- (8x256x320xf32)
        transpose_12 = paddle._C_ops.transpose(flatten_6, [0, 2, 1])
        del flatten_6

        # pd_op.layer_norm: (8x320x256xf32, 8x320xf32, 8x320xf32) <- (8x320x256xf32, 256xf32, 256xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_12, parameter_1, parameter_0, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_0, parameter_1

        # pd_op.transpose: (8x256x320xf32) <- (8x320x256xf32)
        transpose_13 = paddle._C_ops.transpose(layer_norm_36, [0, 2, 1])
        del layer_norm_36

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_4 = [0, 256, 4, 80]

        # pd_op.reshape: (8x256x4x80xf32) <- (8x256x320xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(transpose_13, full_int_array_4)
        del (
            add_0,
            add_1,
            add_11,
            add_12,
            add_14,
            add_15,
            add_16,
            add_18,
            add_19,
            add_2,
            add_21,
            add_22,
            add_23,
            add_25,
            add_26,
            add_28,
            add_29,
            add_3,
            add_30,
            add_32,
            add_33,
            add_35,
            add_36,
            add_37,
            add_39,
            add_4,
            add_40,
            add_5,
            add_7,
            add_8,
            add_9,
            cast_0,
            cast_1,
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
            conv2d_3,
            conv2d_4,
            conv2d_5,
            conv2d_6,
            divide_0,
            divide_1,
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
            floor_2,
            floor_3,
            floor_4,
            floor_5,
            floor_6,
            floor_7,
            floor_8,
            floor_9,
            full_int_array_4,
            gelu_0,
            gelu_1,
            gelu_2,
            gelu_3,
            gelu_4,
            gelu_5,
            layer_norm_0,
            layer_norm_1,
            layer_norm_10,
            layer_norm_11,
            layer_norm_12,
            layer_norm_13,
            layer_norm_14,
            layer_norm_16,
            layer_norm_17,
            layer_norm_18,
            layer_norm_19,
            layer_norm_2,
            layer_norm_20,
            layer_norm_22,
            layer_norm_23,
            layer_norm_24,
            layer_norm_25,
            layer_norm_26,
            layer_norm_28,
            layer_norm_29,
            layer_norm_30,
            layer_norm_31,
            layer_norm_32,
            layer_norm_34,
            layer_norm_35,
            layer_norm_37,
            layer_norm_38,
            layer_norm_4,
            layer_norm_5,
            layer_norm_6,
            layer_norm_7,
            layer_norm_8,
            matmul_0,
            matmul_1,
            matmul_10,
            matmul_11,
            matmul_2,
            matmul_3,
            matmul_4,
            matmul_5,
            matmul_6,
            matmul_7,
            matmul_8,
            matmul_9,
            multiply_0,
            multiply_1,
            multiply_2,
            multiply_3,
            multiply_4,
            multiply_5,
            multiply_6,
            multiply_7,
            multiply_8,
            multiply_9,
            reshape_1,
            reshape_10,
            reshape_11,
            reshape_12,
            reshape_13,
            reshape_2,
            reshape_3,
            reshape_4,
            reshape_5,
            reshape_6,
            reshape_7,
            reshape_8,
            reshape_9,
            transpose_0,
            transpose_1,
            transpose_10,
            transpose_11,
            transpose_12,
            transpose_13,
            transpose_2,
            transpose_3,
            transpose_4,
            transpose_5,
            transpose_6,
            transpose_7,
            transpose_8,
            transpose_9,
        )

        return reshape_0
