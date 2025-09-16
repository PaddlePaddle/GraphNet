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
        data_4,
        data_5,
        data_6,
    ):
        # pd_op.matmul: (8x-1x1152xf32) <- (8x-1x-1xf32, 384x1152xf32)
        matmul_0 = paddle._C_ops.matmul(data_6, parameter_71, False, False)
        del parameter_71

        # pd_op.add: (8x-1x1152xf32) <- (8x-1x1152xf32, 1152xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_70)
        del parameter_70

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_0 = [0, -1, 3, 12, 32]

        # pd_op.reshape: (8x-1x3x12x32xf32) <- (8x-1x1152xf32, 5xi64)
        reshape_0 = paddle._C_ops.reshape(add_0, full_int_array_0)

        # pd_op.transpose: (3x8x12x-1x32xf32) <- (8x-1x3x12x32xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_0, [2, 0, 3, 1, 4])
        del reshape_0

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

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_4 = full_int_array_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [1]

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

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_12 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_13 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_14 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_15 = full_int_array_2

        # pd_op.slice: (8x12x-1x32xf32) <- (3x8x12x-1x32xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            transpose_0, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_16 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_17 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_18 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_19 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_20 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_21 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_22 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_23 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_24 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_25 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_26 = full_int_array_3

        # pd_op.slice: (8x12x-1x32xf32) <- (3x8x12x-1x32xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            transpose_0, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_27 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_28 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_29 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_30 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_31 = full_int_array_4

        # pd_op.slice: (8x12x-1x32xf32) <- (3x8x12x-1x32xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            transpose_0, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.transpose: (8x12x32x-1xf32) <- (8x12x-1x32xf32)
        transpose_1 = paddle._C_ops.transpose(slice_1, [0, 1, 3, 2])
        del slice_1

        # pd_op.matmul: (8x12x-1x-1xf32) <- (8x12x-1x32xf32, 8x12x32x-1xf32)
        matmul_1 = paddle._C_ops.matmul(slice_0, transpose_1, False, False)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_32 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_33 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_34 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_35 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_36 = full_0

        # pd_op.scale: (8x12x-1x-1xf32) <- (8x12x-1x-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(matmul_1, full_0, float("0"), True)
        del matmul_1

        # pd_op.softmax: (8x12x-1x-1xf32) <- (8x12x-1x-1xf32)
        softmax_0 = paddle._C_ops.softmax(scale_0, -1)
        del scale_0

        # pd_op.matmul: (8x12x-1x32xf32) <- (8x12x-1x-1xf32, 8x12x-1x32xf32)
        matmul_2 = paddle._C_ops.matmul(softmax_0, slice_2, False, False)

        # pd_op.transpose: (8x-1x12x32xf32) <- (8x12x-1x32xf32)
        transpose_2 = paddle._C_ops.transpose(matmul_2, [0, 2, 1, 3])
        del matmul_2

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_5 = [0, -1, 384]

        # pd_op.reshape: (8x-1x384xf32) <- (8x-1x12x32xf32, 3xi64)
        reshape_1 = paddle._C_ops.reshape(transpose_2, full_int_array_5)

        # pd_op.matmul: (8x-1x384xf32) <- (8x-1x384xf32, 384x384xf32)
        matmul_3 = paddle._C_ops.matmul(reshape_1, parameter_69, False, False)
        del parameter_69

        # pd_op.add: (8x-1x384xf32) <- (8x-1x384xf32, 384xf32)
        add_1 = paddle._C_ops.add(matmul_3, parameter_68)
        del parameter_68

        # pd_op.full: (xf64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_1,
            [],
            paddle.float64,
            [float("0.929412")],
            paddle.framework._current_expected_place(),
        )
        del full_1

        # pd_op.cast: (xf32) <- (xf64)
        cast_0 = paddle._C_ops.cast(assign_value__0, paddle.float32)
        del assign_value__0

        # pd_op.shape64: (3xi64) <- (8x-1x384xf32)
        shape64_0 = paddle._C_ops.shape64(add_1)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_0

        # pd_op.full: (xi64) <- ()
        full_2 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_0 = [slice_3, full_2, full_2]
        del slice_3

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_0 = paddle._C_ops.uniform(
            stack_0,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_0

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_2 = paddle._C_ops.add(cast_0, uniform_0)
        del uniform_0

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_0 = paddle._C_ops.floor(add_2)
        del add_2

        # pd_op.divide: (8x-1x384xf32) <- (8x-1x384xf32, xf32)
        divide_0 = paddle._C_ops.divide(add_1, cast_0)

        # pd_op.multiply: (8x-1x384xf32) <- (8x-1x384xf32, -1x1x1xf32)
        multiply_0 = paddle._C_ops.multiply(divide_0, floor_0)

        # pd_op.add: (8x-1x384xf32) <- (8x-1x-1xf32, 8x-1x384xf32)
        add_3 = paddle._C_ops.add(data_6, multiply_0)
        del data_6

        # pd_op.layer_norm: (8x-1x384xf32, 8x-1xf32, 8x-1xf32) <- (8x-1x384xf32, 384xf32, 384xf32)
        layer_norm_1, layer_norm_2, layer_norm_3 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_3, parameter_67, parameter_66, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_66, parameter_67

        # pd_op.matmul: (8x-1x1536xf32) <- (8x-1x384xf32, 384x1536xf32)
        matmul_4 = paddle._C_ops.matmul(layer_norm_1, parameter_65, False, False)
        del parameter_65

        # pd_op.add: (8x-1x1536xf32) <- (8x-1x1536xf32, 1536xf32)
        add_4 = paddle._C_ops.add(matmul_4, parameter_64)
        del parameter_64

        # pd_op.gelu: (8x-1x1536xf32) <- (8x-1x1536xf32)
        gelu_0 = paddle._C_ops.gelu(add_4, False)

        # pd_op.matmul: (8x-1x384xf32) <- (8x-1x1536xf32, 1536x384xf32)
        matmul_5 = paddle._C_ops.matmul(gelu_0, parameter_63, False, False)
        del parameter_63

        # pd_op.add: (8x-1x384xf32) <- (8x-1x384xf32, 384xf32)
        add_5 = paddle._C_ops.add(matmul_5, parameter_62)
        del parameter_62

        # pd_op.full: (xf64) <- ()
        full_5 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__1 = paddle._C_ops.assign_value_(
            full_5,
            [],
            paddle.float64,
            [float("0.929412")],
            paddle.framework._current_expected_place(),
        )
        del full_5

        # pd_op.cast: (xf32) <- (xf64)
        cast_1 = paddle._C_ops.cast(assign_value__1, paddle.float32)
        del assign_value__1

        # pd_op.shape64: (3xi64) <- (8x-1x384xf32)
        shape64_1 = paddle._C_ops.shape64(add_5)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_1

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_1 = [slice_4, full_2, full_2]
        del slice_4

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_1, 0)
        del combine_1

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_1 = paddle._C_ops.uniform(
            stack_1,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_1

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_6 = paddle._C_ops.add(cast_1, uniform_1)
        del uniform_1

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_1 = paddle._C_ops.floor(add_6)
        del add_6

        # pd_op.divide: (8x-1x384xf32) <- (8x-1x384xf32, xf32)
        divide_1 = paddle._C_ops.divide(add_5, cast_1)

        # pd_op.multiply: (8x-1x384xf32) <- (8x-1x384xf32, -1x1x1xf32)
        multiply_1 = paddle._C_ops.multiply(divide_1, floor_1)

        # pd_op.add: (8x-1x384xf32) <- (8x-1x384xf32, 8x-1x384xf32)
        add_7 = paddle._C_ops.add(layer_norm_1, multiply_1)

        # pd_op.layer_norm: (8x-1x384xf32, 8x-1xf32, 8x-1xf32) <- (8x-1x384xf32, 384xf32, 384xf32)
        layer_norm_4, layer_norm_5, layer_norm_6 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_7, parameter_61, parameter_60, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_60, parameter_61

        # pd_op.matmul: (8x-1x1152xf32) <- (8x-1x384xf32, 384x1152xf32)
        matmul_6 = paddle._C_ops.matmul(layer_norm_4, parameter_59, False, False)
        del parameter_59

        # pd_op.add: (8x-1x1152xf32) <- (8x-1x1152xf32, 1152xf32)
        add_8 = paddle._C_ops.add(matmul_6, parameter_58)
        del parameter_58

        # pd_op.reshape: (8x-1x3x12x32xf32) <- (8x-1x1152xf32, 5xi64)
        reshape_2 = paddle._C_ops.reshape(add_8, full_int_array_0)

        # pd_op.transpose: (3x8x12x-1x32xf32) <- (8x-1x3x12x32xf32)
        transpose_3 = paddle._C_ops.transpose(reshape_2, [2, 0, 3, 1, 4])
        del reshape_2

        # pd_op.slice: (8x12x-1x32xf32) <- (3x8x12x-1x32xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            transpose_3, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (8x12x-1x32xf32) <- (3x8x12x-1x32xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            transpose_3, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (8x12x-1x32xf32) <- (3x8x12x-1x32xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            transpose_3, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.transpose: (8x12x32x-1xf32) <- (8x12x-1x32xf32)
        transpose_4 = paddle._C_ops.transpose(slice_6, [0, 1, 3, 2])
        del slice_6

        # pd_op.matmul: (8x12x-1x-1xf32) <- (8x12x-1x32xf32, 8x12x32x-1xf32)
        matmul_7 = paddle._C_ops.matmul(slice_5, transpose_4, False, False)

        # pd_op.scale: (8x12x-1x-1xf32) <- (8x12x-1x-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(matmul_7, full_0, float("0"), True)
        del matmul_7

        # pd_op.softmax: (8x12x-1x-1xf32) <- (8x12x-1x-1xf32)
        softmax_1 = paddle._C_ops.softmax(scale_1, -1)
        del scale_1

        # pd_op.matmul: (8x12x-1x32xf32) <- (8x12x-1x-1xf32, 8x12x-1x32xf32)
        matmul_8 = paddle._C_ops.matmul(softmax_1, slice_7, False, False)

        # pd_op.transpose: (8x-1x12x32xf32) <- (8x12x-1x32xf32)
        transpose_5 = paddle._C_ops.transpose(matmul_8, [0, 2, 1, 3])
        del matmul_8

        # pd_op.reshape: (8x-1x384xf32) <- (8x-1x12x32xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_5, full_int_array_5)

        # pd_op.matmul: (8x-1x384xf32) <- (8x-1x384xf32, 384x384xf32)
        matmul_9 = paddle._C_ops.matmul(reshape_3, parameter_57, False, False)
        del parameter_57

        # pd_op.add: (8x-1x384xf32) <- (8x-1x384xf32, 384xf32)
        add_9 = paddle._C_ops.add(matmul_9, parameter_56)
        del parameter_56

        # pd_op.full: (xf64) <- ()
        full_6 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__2 = paddle._C_ops.assign_value_(
            full_6,
            [],
            paddle.float64,
            [float("0.923529")],
            paddle.framework._current_expected_place(),
        )
        del full_6

        # pd_op.cast: (xf32) <- (xf64)
        cast_2 = paddle._C_ops.cast(assign_value__2, paddle.float32)
        del assign_value__2

        # pd_op.shape64: (3xi64) <- (8x-1x384xf32)
        shape64_2 = paddle._C_ops.shape64(add_9)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_2

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_2 = [slice_8, full_2, full_2]
        del slice_8

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_2 = paddle._C_ops.stack(combine_2, 0)
        del combine_2

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_2 = paddle._C_ops.uniform(
            stack_2,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_2

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_10 = paddle._C_ops.add(cast_2, uniform_2)
        del uniform_2

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_2 = paddle._C_ops.floor(add_10)
        del add_10

        # pd_op.divide: (8x-1x384xf32) <- (8x-1x384xf32, xf32)
        divide_2 = paddle._C_ops.divide(add_9, cast_2)

        # pd_op.multiply: (8x-1x384xf32) <- (8x-1x384xf32, -1x1x1xf32)
        multiply_2 = paddle._C_ops.multiply(divide_2, floor_2)

        # pd_op.add: (8x-1x384xf32) <- (8x-1x384xf32, 8x-1x384xf32)
        add_11 = paddle._C_ops.add(layer_norm_4, multiply_2)

        # pd_op.layer_norm: (8x-1x384xf32, 8x-1xf32, 8x-1xf32) <- (8x-1x384xf32, 384xf32, 384xf32)
        layer_norm_7, layer_norm_8, layer_norm_9 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_11, parameter_55, parameter_54, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_54, parameter_55

        # pd_op.matmul: (8x-1x1536xf32) <- (8x-1x384xf32, 384x1536xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_7, parameter_53, False, False)
        del parameter_53

        # pd_op.add: (8x-1x1536xf32) <- (8x-1x1536xf32, 1536xf32)
        add_12 = paddle._C_ops.add(matmul_10, parameter_52)
        del parameter_52

        # pd_op.gelu: (8x-1x1536xf32) <- (8x-1x1536xf32)
        gelu_1 = paddle._C_ops.gelu(add_12, False)

        # pd_op.matmul: (8x-1x384xf32) <- (8x-1x1536xf32, 1536x384xf32)
        matmul_11 = paddle._C_ops.matmul(gelu_1, parameter_51, False, False)
        del parameter_51

        # pd_op.add: (8x-1x384xf32) <- (8x-1x384xf32, 384xf32)
        add_13 = paddle._C_ops.add(matmul_11, parameter_50)
        del parameter_50

        # pd_op.full: (xf64) <- ()
        full_7 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__3 = paddle._C_ops.assign_value_(
            full_7,
            [],
            paddle.float64,
            [float("0.923529")],
            paddle.framework._current_expected_place(),
        )
        del full_7

        # pd_op.cast: (xf32) <- (xf64)
        cast_3 = paddle._C_ops.cast(assign_value__3, paddle.float32)
        del assign_value__3

        # pd_op.shape64: (3xi64) <- (8x-1x384xf32)
        shape64_3 = paddle._C_ops.shape64(add_13)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_3

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_3 = [slice_9, full_2, full_2]
        del slice_9

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_3 = paddle._C_ops.stack(combine_3, 0)
        del combine_3

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_3 = paddle._C_ops.uniform(
            stack_3,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_3

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_14 = paddle._C_ops.add(cast_3, uniform_3)
        del uniform_3

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_3 = paddle._C_ops.floor(add_14)
        del add_14

        # pd_op.divide: (8x-1x384xf32) <- (8x-1x384xf32, xf32)
        divide_3 = paddle._C_ops.divide(add_13, cast_3)

        # pd_op.multiply: (8x-1x384xf32) <- (8x-1x384xf32, -1x1x1xf32)
        multiply_3 = paddle._C_ops.multiply(divide_3, floor_3)

        # pd_op.add: (8x-1x384xf32) <- (8x-1x384xf32, 8x-1x384xf32)
        add_15 = paddle._C_ops.add(layer_norm_7, multiply_3)

        # pd_op.layer_norm: (8x-1x384xf32, 8x-1xf32, 8x-1xf32) <- (8x-1x384xf32, 384xf32, 384xf32)
        layer_norm_10, layer_norm_11, layer_norm_12 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_15, parameter_49, parameter_48, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_48, parameter_49

        # pd_op.matmul: (8x-1x1152xf32) <- (8x-1x384xf32, 384x1152xf32)
        matmul_12 = paddle._C_ops.matmul(layer_norm_10, parameter_47, False, False)
        del parameter_47

        # pd_op.add: (8x-1x1152xf32) <- (8x-1x1152xf32, 1152xf32)
        add_16 = paddle._C_ops.add(matmul_12, parameter_46)
        del parameter_46

        # pd_op.reshape: (8x-1x3x12x32xf32) <- (8x-1x1152xf32, 5xi64)
        reshape_4 = paddle._C_ops.reshape(add_16, full_int_array_0)
        del full_int_array_0

        # pd_op.transpose: (3x8x12x-1x32xf32) <- (8x-1x3x12x32xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_4, [2, 0, 3, 1, 4])
        del reshape_4

        # pd_op.slice: (8x12x-1x32xf32) <- (3x8x12x-1x32xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            transpose_6, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (8x12x-1x32xf32) <- (3x8x12x-1x32xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            transpose_6, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (8x12x-1x32xf32) <- (3x8x12x-1x32xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            transpose_6, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.transpose: (8x12x32x-1xf32) <- (8x12x-1x32xf32)
        transpose_7 = paddle._C_ops.transpose(slice_11, [0, 1, 3, 2])
        del slice_11

        # pd_op.matmul: (8x12x-1x-1xf32) <- (8x12x-1x32xf32, 8x12x32x-1xf32)
        matmul_13 = paddle._C_ops.matmul(slice_10, transpose_7, False, False)

        # pd_op.scale: (8x12x-1x-1xf32) <- (8x12x-1x-1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(matmul_13, full_0, float("0"), True)
        del matmul_13

        # pd_op.softmax: (8x12x-1x-1xf32) <- (8x12x-1x-1xf32)
        softmax_2 = paddle._C_ops.softmax(scale_2, -1)
        del scale_2

        # pd_op.matmul: (8x12x-1x32xf32) <- (8x12x-1x-1xf32, 8x12x-1x32xf32)
        matmul_14 = paddle._C_ops.matmul(softmax_2, slice_12, False, False)

        # pd_op.transpose: (8x-1x12x32xf32) <- (8x12x-1x32xf32)
        transpose_8 = paddle._C_ops.transpose(matmul_14, [0, 2, 1, 3])
        del matmul_14

        # pd_op.reshape: (8x-1x384xf32) <- (8x-1x12x32xf32, 3xi64)
        reshape_5 = paddle._C_ops.reshape(transpose_8, full_int_array_5)
        del full_int_array_5

        # pd_op.matmul: (8x-1x384xf32) <- (8x-1x384xf32, 384x384xf32)
        matmul_15 = paddle._C_ops.matmul(reshape_5, parameter_45, False, False)
        del parameter_45

        # pd_op.add: (8x-1x384xf32) <- (8x-1x384xf32, 384xf32)
        add_17 = paddle._C_ops.add(matmul_15, parameter_44)
        del parameter_44

        # pd_op.full: (xf64) <- ()
        full_8 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__4 = paddle._C_ops.assign_value_(
            full_8,
            [],
            paddle.float64,
            [float("0.917647")],
            paddle.framework._current_expected_place(),
        )
        del full_8

        # pd_op.cast: (xf32) <- (xf64)
        cast_4 = paddle._C_ops.cast(assign_value__4, paddle.float32)
        del assign_value__4

        # pd_op.shape64: (3xi64) <- (8x-1x384xf32)
        shape64_4 = paddle._C_ops.shape64(add_17)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            shape64_4, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_4

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_4 = [slice_13, full_2, full_2]
        del slice_13

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_4 = paddle._C_ops.stack(combine_4, 0)
        del combine_4

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_4 = paddle._C_ops.uniform(
            stack_4,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_4

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_18 = paddle._C_ops.add(cast_4, uniform_4)
        del uniform_4

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_4 = paddle._C_ops.floor(add_18)
        del add_18

        # pd_op.divide: (8x-1x384xf32) <- (8x-1x384xf32, xf32)
        divide_4 = paddle._C_ops.divide(add_17, cast_4)

        # pd_op.multiply: (8x-1x384xf32) <- (8x-1x384xf32, -1x1x1xf32)
        multiply_4 = paddle._C_ops.multiply(divide_4, floor_4)

        # pd_op.add: (8x-1x384xf32) <- (8x-1x384xf32, 8x-1x384xf32)
        add_19 = paddle._C_ops.add(layer_norm_10, multiply_4)

        # pd_op.layer_norm: (8x-1x384xf32, 8x-1xf32, 8x-1xf32) <- (8x-1x384xf32, 384xf32, 384xf32)
        layer_norm_13, layer_norm_14, layer_norm_15 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_19, parameter_43, parameter_42, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_42, parameter_43

        # pd_op.matmul: (8x-1x1536xf32) <- (8x-1x384xf32, 384x1536xf32)
        matmul_16 = paddle._C_ops.matmul(layer_norm_13, parameter_41, False, False)
        del parameter_41

        # pd_op.add: (8x-1x1536xf32) <- (8x-1x1536xf32, 1536xf32)
        add_20 = paddle._C_ops.add(matmul_16, parameter_40)
        del parameter_40

        # pd_op.gelu: (8x-1x1536xf32) <- (8x-1x1536xf32)
        gelu_2 = paddle._C_ops.gelu(add_20, False)

        # pd_op.matmul: (8x-1x384xf32) <- (8x-1x1536xf32, 1536x384xf32)
        matmul_17 = paddle._C_ops.matmul(gelu_2, parameter_39, False, False)
        del parameter_39

        # pd_op.add: (8x-1x384xf32) <- (8x-1x384xf32, 384xf32)
        add_21 = paddle._C_ops.add(matmul_17, parameter_38)
        del parameter_38

        # pd_op.full: (xf64) <- ()
        full_9 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__5 = paddle._C_ops.assign_value_(
            full_9,
            [],
            paddle.float64,
            [float("0.917647")],
            paddle.framework._current_expected_place(),
        )
        del full_9

        # pd_op.cast: (xf32) <- (xf64)
        cast_5 = paddle._C_ops.cast(assign_value__5, paddle.float32)
        del assign_value__5

        # pd_op.shape64: (3xi64) <- (8x-1x384xf32)
        shape64_5 = paddle._C_ops.shape64(add_21)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            shape64_5, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_5

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_5 = [slice_14, full_2, full_2]
        del slice_14

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_5 = paddle._C_ops.stack(combine_5, 0)
        del combine_5

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_5 = paddle._C_ops.uniform(
            stack_5,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_5

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_22 = paddle._C_ops.add(cast_5, uniform_5)
        del uniform_5

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_5 = paddle._C_ops.floor(add_22)
        del add_22

        # pd_op.divide: (8x-1x384xf32) <- (8x-1x384xf32, xf32)
        divide_5 = paddle._C_ops.divide(add_21, cast_5)

        # pd_op.multiply: (8x-1x384xf32) <- (8x-1x384xf32, -1x1x1xf32)
        multiply_5 = paddle._C_ops.multiply(divide_5, floor_5)

        # pd_op.add: (8x-1x384xf32) <- (8x-1x384xf32, 8x-1x384xf32)
        add_23 = paddle._C_ops.add(layer_norm_13, multiply_5)

        # pd_op.layer_norm: (8x-1x384xf32, 8x-1xf32, 8x-1xf32) <- (8x-1x384xf32, 384xf32, 384xf32)
        layer_norm_16, layer_norm_17, layer_norm_18 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_23, parameter_37, parameter_36, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_36, parameter_37

        # pd_op.matmul: (8x-1x1152xf32) <- (8x-1x384xf32, 384x1152xf32)
        matmul_18 = paddle._C_ops.matmul(layer_norm_16, parameter_35, False, False)
        del parameter_35

        # pd_op.add: (8x-1x1152xf32) <- (8x-1x1152xf32, 1152xf32)
        add_24 = paddle._C_ops.add(matmul_18, parameter_34)
        del parameter_34

        # pd_op.full: (xi64) <- ()
        full_10 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_11 = paddle._C_ops.full(
            [], float("-1"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_12 = paddle._C_ops.full(
            [], float("3"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_13 = paddle._C_ops.full(
            [], float("32"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_6 = [full_10, full_11, full_12, data_0, full_13]
        del data_0

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_6 = paddle._C_ops.stack(combine_6, 0)
        del combine_6

        # pd_op.reshape: (8x-1x3x-1x32xf32) <- (8x-1x1152xf32, 5xi64)
        reshape_6 = paddle._C_ops.reshape(add_24, stack_6)
        del stack_6

        # pd_op.transpose: (3x8x-1x-1x32xf32) <- (8x-1x3x-1x32xf32)
        transpose_9 = paddle._C_ops.transpose(reshape_6, [2, 0, 3, 1, 4])
        del reshape_6

        # pd_op.slice: (8x-1x-1x32xf32) <- (3x8x-1x-1x32xf32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            transpose_9, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (8x-1x-1x32xf32) <- (3x8x-1x-1x32xf32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            transpose_9, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (8x-1x-1x32xf32) <- (3x8x-1x-1x32xf32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            transpose_9, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.transpose: (8x-1x32x-1xf32) <- (8x-1x-1x32xf32)
        transpose_10 = paddle._C_ops.transpose(slice_16, [0, 1, 3, 2])
        del slice_16

        # pd_op.matmul: (8x-1x-1x-1xf32) <- (8x-1x-1x32xf32, 8x-1x32x-1xf32)
        matmul_19 = paddle._C_ops.matmul(slice_15, transpose_10, False, False)

        # pd_op.scale: (8x-1x-1x-1xf32) <- (8x-1x-1x-1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(matmul_19, full_0, float("0"), True)
        del matmul_19

        # pd_op.softmax: (8x-1x-1x-1xf32) <- (8x-1x-1x-1xf32)
        softmax_3 = paddle._C_ops.softmax(scale_3, -1)
        del scale_3

        # pd_op.matmul: (8x-1x-1x32xf32) <- (8x-1x-1x-1xf32, 8x-1x-1x32xf32)
        matmul_20 = paddle._C_ops.matmul(softmax_3, slice_17, False, False)

        # pd_op.transpose: (8x-1x-1x32xf32) <- (8x-1x-1x32xf32)
        transpose_11 = paddle._C_ops.transpose(matmul_20, [0, 2, 1, 3])
        del matmul_20

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_7 = [full_10, full_11, data_1]
        del data_1

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_7 = paddle._C_ops.stack(combine_7, 0)
        del combine_7

        # pd_op.reshape: (8x-1x-1xf32) <- (8x-1x-1x32xf32, 3xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_11, stack_7)
        del stack_7

        # pd_op.matmul: (8x-1x384xf32) <- (8x-1x-1xf32, 384x384xf32)
        matmul_21 = paddle._C_ops.matmul(reshape_7, parameter_33, False, False)
        del parameter_33

        # pd_op.add: (8x-1x384xf32) <- (8x-1x384xf32, 384xf32)
        add_25 = paddle._C_ops.add(matmul_21, parameter_32)
        del parameter_32

        # pd_op.full: (xf64) <- ()
        full_14 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__6 = paddle._C_ops.assign_value_(
            full_14,
            [],
            paddle.float64,
            [float("0.911765")],
            paddle.framework._current_expected_place(),
        )
        del full_14

        # pd_op.cast: (xf32) <- (xf64)
        cast_6 = paddle._C_ops.cast(assign_value__6, paddle.float32)
        del assign_value__6

        # pd_op.shape64: (3xi64) <- (8x-1x384xf32)
        shape64_6 = paddle._C_ops.shape64(add_25)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(
            shape64_6, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_6

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_8 = [slice_18, full_2, full_2]
        del slice_18

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_8 = paddle._C_ops.stack(combine_8, 0)
        del combine_8

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_6 = paddle._C_ops.uniform(
            stack_8,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_8

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_26 = paddle._C_ops.add(cast_6, uniform_6)
        del uniform_6

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_6 = paddle._C_ops.floor(add_26)
        del add_26

        # pd_op.divide: (8x-1x384xf32) <- (8x-1x384xf32, xf32)
        divide_6 = paddle._C_ops.divide(add_25, cast_6)

        # pd_op.multiply: (8x-1x384xf32) <- (8x-1x384xf32, -1x1x1xf32)
        multiply_6 = paddle._C_ops.multiply(divide_6, floor_6)

        # pd_op.add: (8x-1x384xf32) <- (8x-1x384xf32, 8x-1x384xf32)
        add_27 = paddle._C_ops.add(layer_norm_16, multiply_6)

        # pd_op.layer_norm: (8x-1x384xf32, 8x-1xf32, 8x-1xf32) <- (8x-1x384xf32, 384xf32, 384xf32)
        layer_norm_19, layer_norm_20, layer_norm_21 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_27, parameter_31, parameter_30, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_30, parameter_31

        # pd_op.matmul: (8x-1x1536xf32) <- (8x-1x384xf32, 384x1536xf32)
        matmul_22 = paddle._C_ops.matmul(layer_norm_19, parameter_29, False, False)
        del parameter_29

        # pd_op.add: (8x-1x1536xf32) <- (8x-1x1536xf32, 1536xf32)
        add_28 = paddle._C_ops.add(matmul_22, parameter_28)
        del parameter_28

        # pd_op.gelu: (8x-1x1536xf32) <- (8x-1x1536xf32)
        gelu_3 = paddle._C_ops.gelu(add_28, False)

        # pd_op.matmul: (8x-1x384xf32) <- (8x-1x1536xf32, 1536x384xf32)
        matmul_23 = paddle._C_ops.matmul(gelu_3, parameter_27, False, False)
        del parameter_27

        # pd_op.add: (8x-1x384xf32) <- (8x-1x384xf32, 384xf32)
        add_29 = paddle._C_ops.add(matmul_23, parameter_26)
        del parameter_26

        # pd_op.full: (xf64) <- ()
        full_15 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__7 = paddle._C_ops.assign_value_(
            full_15,
            [],
            paddle.float64,
            [float("0.911765")],
            paddle.framework._current_expected_place(),
        )
        del full_15

        # pd_op.cast: (xf32) <- (xf64)
        cast_7 = paddle._C_ops.cast(assign_value__7, paddle.float32)
        del assign_value__7

        # pd_op.shape64: (3xi64) <- (8x-1x384xf32)
        shape64_7 = paddle._C_ops.shape64(add_29)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(
            shape64_7, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_7

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_9 = [slice_19, full_2, full_2]
        del slice_19

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_9 = paddle._C_ops.stack(combine_9, 0)
        del combine_9

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_7 = paddle._C_ops.uniform(
            stack_9,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_9

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_30 = paddle._C_ops.add(cast_7, uniform_7)
        del uniform_7

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_7 = paddle._C_ops.floor(add_30)
        del add_30

        # pd_op.divide: (8x-1x384xf32) <- (8x-1x384xf32, xf32)
        divide_7 = paddle._C_ops.divide(add_29, cast_7)

        # pd_op.multiply: (8x-1x384xf32) <- (8x-1x384xf32, -1x1x1xf32)
        multiply_7 = paddle._C_ops.multiply(divide_7, floor_7)

        # pd_op.add: (8x-1x384xf32) <- (8x-1x384xf32, 8x-1x384xf32)
        add_31 = paddle._C_ops.add(layer_norm_19, multiply_7)

        # pd_op.layer_norm: (8x-1x384xf32, 8x-1xf32, 8x-1xf32) <- (8x-1x384xf32, 384xf32, 384xf32)
        layer_norm_22, layer_norm_23, layer_norm_24 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_31, parameter_25, parameter_24, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_24, parameter_25

        # pd_op.matmul: (8x-1x1152xf32) <- (8x-1x384xf32, 384x1152xf32)
        matmul_24 = paddle._C_ops.matmul(layer_norm_22, parameter_23, False, False)
        del parameter_23

        # pd_op.add: (8x-1x1152xf32) <- (8x-1x1152xf32, 1152xf32)
        add_32 = paddle._C_ops.add(matmul_24, parameter_22)
        del parameter_22

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_10 = [full_10, full_11, full_12, data_2, full_13]
        del data_2

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_10 = paddle._C_ops.stack(combine_10, 0)
        del combine_10

        # pd_op.reshape: (8x-1x3x-1x32xf32) <- (8x-1x1152xf32, 5xi64)
        reshape_8 = paddle._C_ops.reshape(add_32, stack_10)
        del stack_10

        # pd_op.transpose: (3x8x-1x-1x32xf32) <- (8x-1x3x-1x32xf32)
        transpose_12 = paddle._C_ops.transpose(reshape_8, [2, 0, 3, 1, 4])
        del reshape_8

        # pd_op.slice: (8x-1x-1x32xf32) <- (3x8x-1x-1x32xf32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(
            transpose_12, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (8x-1x-1x32xf32) <- (3x8x-1x-1x32xf32, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(
            transpose_12, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (8x-1x-1x32xf32) <- (3x8x-1x-1x32xf32, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(
            transpose_12, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.transpose: (8x-1x32x-1xf32) <- (8x-1x-1x32xf32)
        transpose_13 = paddle._C_ops.transpose(slice_21, [0, 1, 3, 2])
        del slice_21

        # pd_op.matmul: (8x-1x-1x-1xf32) <- (8x-1x-1x32xf32, 8x-1x32x-1xf32)
        matmul_25 = paddle._C_ops.matmul(slice_20, transpose_13, False, False)

        # pd_op.scale: (8x-1x-1x-1xf32) <- (8x-1x-1x-1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(matmul_25, full_0, float("0"), True)
        del matmul_25

        # pd_op.softmax: (8x-1x-1x-1xf32) <- (8x-1x-1x-1xf32)
        softmax_4 = paddle._C_ops.softmax(scale_4, -1)
        del scale_4

        # pd_op.matmul: (8x-1x-1x32xf32) <- (8x-1x-1x-1xf32, 8x-1x-1x32xf32)
        matmul_26 = paddle._C_ops.matmul(softmax_4, slice_22, False, False)

        # pd_op.transpose: (8x-1x-1x32xf32) <- (8x-1x-1x32xf32)
        transpose_14 = paddle._C_ops.transpose(matmul_26, [0, 2, 1, 3])
        del matmul_26

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_11 = [full_10, full_11, data_3]
        del data_3

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_11 = paddle._C_ops.stack(combine_11, 0)
        del combine_11

        # pd_op.reshape: (8x-1x-1xf32) <- (8x-1x-1x32xf32, 3xi64)
        reshape_9 = paddle._C_ops.reshape(transpose_14, stack_11)
        del stack_11

        # pd_op.matmul: (8x-1x384xf32) <- (8x-1x-1xf32, 384x384xf32)
        matmul_27 = paddle._C_ops.matmul(reshape_9, parameter_21, False, False)
        del parameter_21

        # pd_op.add: (8x-1x384xf32) <- (8x-1x384xf32, 384xf32)
        add_33 = paddle._C_ops.add(matmul_27, parameter_20)
        del parameter_20

        # pd_op.full: (xf64) <- ()
        full_16 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__8 = paddle._C_ops.assign_value_(
            full_16,
            [],
            paddle.float64,
            [float("0.905882")],
            paddle.framework._current_expected_place(),
        )
        del full_16

        # pd_op.cast: (xf32) <- (xf64)
        cast_8 = paddle._C_ops.cast(assign_value__8, paddle.float32)
        del assign_value__8

        # pd_op.shape64: (3xi64) <- (8x-1x384xf32)
        shape64_8 = paddle._C_ops.shape64(add_33)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(
            shape64_8, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_8

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_12 = [slice_23, full_2, full_2]
        del slice_23

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_12 = paddle._C_ops.stack(combine_12, 0)
        del combine_12

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_8 = paddle._C_ops.uniform(
            stack_12,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_12

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_34 = paddle._C_ops.add(cast_8, uniform_8)
        del uniform_8

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_8 = paddle._C_ops.floor(add_34)
        del add_34

        # pd_op.divide: (8x-1x384xf32) <- (8x-1x384xf32, xf32)
        divide_8 = paddle._C_ops.divide(add_33, cast_8)

        # pd_op.multiply: (8x-1x384xf32) <- (8x-1x384xf32, -1x1x1xf32)
        multiply_8 = paddle._C_ops.multiply(divide_8, floor_8)

        # pd_op.add: (8x-1x384xf32) <- (8x-1x384xf32, 8x-1x384xf32)
        add_35 = paddle._C_ops.add(layer_norm_22, multiply_8)

        # pd_op.layer_norm: (8x-1x384xf32, 8x-1xf32, 8x-1xf32) <- (8x-1x384xf32, 384xf32, 384xf32)
        layer_norm_25, layer_norm_26, layer_norm_27 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_35, parameter_19, parameter_18, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_18, parameter_19

        # pd_op.matmul: (8x-1x1536xf32) <- (8x-1x384xf32, 384x1536xf32)
        matmul_28 = paddle._C_ops.matmul(layer_norm_25, parameter_17, False, False)
        del parameter_17

        # pd_op.add: (8x-1x1536xf32) <- (8x-1x1536xf32, 1536xf32)
        add_36 = paddle._C_ops.add(matmul_28, parameter_16)
        del parameter_16

        # pd_op.gelu: (8x-1x1536xf32) <- (8x-1x1536xf32)
        gelu_4 = paddle._C_ops.gelu(add_36, False)

        # pd_op.matmul: (8x-1x384xf32) <- (8x-1x1536xf32, 1536x384xf32)
        matmul_29 = paddle._C_ops.matmul(gelu_4, parameter_15, False, False)
        del parameter_15

        # pd_op.add: (8x-1x384xf32) <- (8x-1x384xf32, 384xf32)
        add_37 = paddle._C_ops.add(matmul_29, parameter_14)
        del parameter_14

        # pd_op.full: (xf64) <- ()
        full_17 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__9 = paddle._C_ops.assign_value_(
            full_17,
            [],
            paddle.float64,
            [float("0.905882")],
            paddle.framework._current_expected_place(),
        )
        del full_17

        # pd_op.cast: (xf32) <- (xf64)
        cast_9 = paddle._C_ops.cast(assign_value__9, paddle.float32)
        del assign_value__9

        # pd_op.shape64: (3xi64) <- (8x-1x384xf32)
        shape64_9 = paddle._C_ops.shape64(add_37)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(
            shape64_9, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_9

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_13 = [slice_24, full_2, full_2]
        del slice_24

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_13 = paddle._C_ops.stack(combine_13, 0)
        del combine_13

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_9 = paddle._C_ops.uniform(
            stack_13,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_13

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_38 = paddle._C_ops.add(cast_9, uniform_9)
        del uniform_9

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_9 = paddle._C_ops.floor(add_38)
        del add_38

        # pd_op.divide: (8x-1x384xf32) <- (8x-1x384xf32, xf32)
        divide_9 = paddle._C_ops.divide(add_37, cast_9)

        # pd_op.multiply: (8x-1x384xf32) <- (8x-1x384xf32, -1x1x1xf32)
        multiply_9 = paddle._C_ops.multiply(divide_9, floor_9)

        # pd_op.add: (8x-1x384xf32) <- (8x-1x384xf32, 8x-1x384xf32)
        add_39 = paddle._C_ops.add(layer_norm_25, multiply_9)

        # pd_op.layer_norm: (8x-1x384xf32, 8x-1xf32, 8x-1xf32) <- (8x-1x384xf32, 384xf32, 384xf32)
        layer_norm_28, layer_norm_29, layer_norm_30 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_39, parameter_13, parameter_12, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_12, parameter_13

        # pd_op.matmul: (8x-1x1152xf32) <- (8x-1x384xf32, 384x1152xf32)
        matmul_30 = paddle._C_ops.matmul(layer_norm_28, parameter_11, False, False)
        del parameter_11

        # pd_op.add: (8x-1x1152xf32) <- (8x-1x1152xf32, 1152xf32)
        add_40 = paddle._C_ops.add(matmul_30, parameter_10)
        del parameter_10

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_14 = [full_10, full_11, full_12, data_4, full_13]
        del data_4, full_12, full_13

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_14 = paddle._C_ops.stack(combine_14, 0)
        del combine_14

        # pd_op.reshape: (8x-1x3x-1x32xf32) <- (8x-1x1152xf32, 5xi64)
        reshape_10 = paddle._C_ops.reshape(add_40, stack_14)
        del stack_14

        # pd_op.transpose: (3x8x-1x-1x32xf32) <- (8x-1x3x-1x32xf32)
        transpose_15 = paddle._C_ops.transpose(reshape_10, [2, 0, 3, 1, 4])
        del reshape_10

        # pd_op.slice: (8x-1x-1x32xf32) <- (3x8x-1x-1x32xf32, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(
            transpose_15, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (8x-1x-1x32xf32) <- (3x8x-1x-1x32xf32, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(
            transpose_15, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (8x-1x-1x32xf32) <- (3x8x-1x-1x32xf32, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(
            transpose_15, [0], full_int_array_3, full_int_array_4, [1], [0]
        )

        # pd_op.transpose: (8x-1x32x-1xf32) <- (8x-1x-1x32xf32)
        transpose_16 = paddle._C_ops.transpose(slice_26, [0, 1, 3, 2])
        del slice_26

        # pd_op.matmul: (8x-1x-1x-1xf32) <- (8x-1x-1x32xf32, 8x-1x32x-1xf32)
        matmul_31 = paddle._C_ops.matmul(slice_25, transpose_16, False, False)

        # pd_op.scale: (8x-1x-1x-1xf32) <- (8x-1x-1x-1xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(matmul_31, full_0, float("0"), True)
        del matmul_31

        # pd_op.softmax: (8x-1x-1x-1xf32) <- (8x-1x-1x-1xf32)
        softmax_5 = paddle._C_ops.softmax(scale_5, -1)
        del scale_5

        # pd_op.matmul: (8x-1x-1x32xf32) <- (8x-1x-1x-1xf32, 8x-1x-1x32xf32)
        matmul_32 = paddle._C_ops.matmul(softmax_5, slice_27, False, False)

        # pd_op.transpose: (8x-1x-1x32xf32) <- (8x-1x-1x32xf32)
        transpose_17 = paddle._C_ops.transpose(matmul_32, [0, 2, 1, 3])
        del matmul_32

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_15 = [full_10, full_11, data_5]
        del data_5, full_10, full_11

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_15 = paddle._C_ops.stack(combine_15, 0)
        del combine_15

        # pd_op.reshape: (8x-1x-1xf32) <- (8x-1x-1x32xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_17, stack_15)
        del stack_15

        # pd_op.matmul: (8x-1x384xf32) <- (8x-1x-1xf32, 384x384xf32)
        matmul_33 = paddle._C_ops.matmul(reshape_11, parameter_9, False, False)
        del parameter_9

        # pd_op.add: (8x-1x384xf32) <- (8x-1x384xf32, 384xf32)
        add_41 = paddle._C_ops.add(matmul_33, parameter_8)
        del parameter_8

        # pd_op.full: (xf64) <- ()
        full_18 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__10 = paddle._C_ops.assign_value_(
            full_18,
            [],
            paddle.float64,
            [float("0.9")],
            paddle.framework._current_expected_place(),
        )
        del full_18

        # pd_op.cast: (xf32) <- (xf64)
        cast_10 = paddle._C_ops.cast(assign_value__10, paddle.float32)
        del assign_value__10

        # pd_op.shape64: (3xi64) <- (8x-1x384xf32)
        shape64_10 = paddle._C_ops.shape64(add_41)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(
            shape64_10, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_10

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_16 = [slice_28, full_2, full_2]
        del slice_28

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_16 = paddle._C_ops.stack(combine_16, 0)
        del combine_16

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_10 = paddle._C_ops.uniform(
            stack_16,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_16

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_42 = paddle._C_ops.add(cast_10, uniform_10)
        del uniform_10

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_10 = paddle._C_ops.floor(add_42)
        del add_42

        # pd_op.divide: (8x-1x384xf32) <- (8x-1x384xf32, xf32)
        divide_10 = paddle._C_ops.divide(add_41, cast_10)

        # pd_op.multiply: (8x-1x384xf32) <- (8x-1x384xf32, -1x1x1xf32)
        multiply_10 = paddle._C_ops.multiply(divide_10, floor_10)

        # pd_op.add: (8x-1x384xf32) <- (8x-1x384xf32, 8x-1x384xf32)
        add_43 = paddle._C_ops.add(layer_norm_28, multiply_10)

        # pd_op.layer_norm: (8x-1x384xf32, 8x-1xf32, 8x-1xf32) <- (8x-1x384xf32, 384xf32, 384xf32)
        layer_norm_31, layer_norm_32, layer_norm_33 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_43, parameter_7, parameter_6, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_6, parameter_7

        # pd_op.matmul: (8x-1x1536xf32) <- (8x-1x384xf32, 384x1536xf32)
        matmul_34 = paddle._C_ops.matmul(layer_norm_31, parameter_5, False, False)
        del parameter_5

        # pd_op.add: (8x-1x1536xf32) <- (8x-1x1536xf32, 1536xf32)
        add_44 = paddle._C_ops.add(matmul_34, parameter_4)
        del parameter_4

        # pd_op.gelu: (8x-1x1536xf32) <- (8x-1x1536xf32)
        gelu_5 = paddle._C_ops.gelu(add_44, False)

        # pd_op.matmul: (8x-1x384xf32) <- (8x-1x1536xf32, 1536x384xf32)
        matmul_35 = paddle._C_ops.matmul(gelu_5, parameter_3, False, False)
        del parameter_3

        # pd_op.add: (8x-1x384xf32) <- (8x-1x384xf32, 384xf32)
        add_45 = paddle._C_ops.add(matmul_35, parameter_2)
        del parameter_2

        # pd_op.full: (xf64) <- ()
        full_19 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__11 = paddle._C_ops.assign_value_(
            full_19,
            [],
            paddle.float64,
            [float("0.9")],
            paddle.framework._current_expected_place(),
        )
        del full_19

        # pd_op.cast: (xf32) <- (xf64)
        cast_11 = paddle._C_ops.cast(assign_value__11, paddle.float32)
        del assign_value__11

        # pd_op.shape64: (3xi64) <- (8x-1x384xf32)
        shape64_11 = paddle._C_ops.shape64(add_45)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(
            shape64_11, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_11

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_17 = [slice_29, full_2, full_2]
        del full_2, slice_29

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_17 = paddle._C_ops.stack(combine_17, 0)
        del combine_17

        # pd_op.uniform: (-1x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_11 = paddle._C_ops.uniform(
            stack_17,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del full_3, full_4, stack_17

        # pd_op.add: (-1x1x1xf32) <- (xf32, -1x1x1xf32)
        add_46 = paddle._C_ops.add(cast_11, uniform_11)
        del uniform_11

        # pd_op.floor: (-1x1x1xf32) <- (-1x1x1xf32)
        floor_11 = paddle._C_ops.floor(add_46)
        del add_46

        # pd_op.divide: (8x-1x384xf32) <- (8x-1x384xf32, xf32)
        divide_11 = paddle._C_ops.divide(add_45, cast_11)

        # pd_op.multiply: (8x-1x384xf32) <- (8x-1x384xf32, -1x1x1xf32)
        multiply_11 = paddle._C_ops.multiply(divide_11, floor_11)

        # pd_op.add: (8x-1x384xf32) <- (8x-1x384xf32, 8x-1x384xf32)
        add_47 = paddle._C_ops.add(layer_norm_31, multiply_11)

        # pd_op.layer_norm: (8x-1x384xf32, 8x-1xf32, 8x-1xf32) <- (8x-1x384xf32, 384xf32, 384xf32)
        layer_norm_0, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_47, parameter_1, parameter_0, float("1e-06"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del (
            add_0,
            add_1,
            add_11,
            add_12,
            add_13,
            add_15,
            add_16,
            add_17,
            add_19,
            add_20,
            add_21,
            add_23,
            add_24,
            add_25,
            add_27,
            add_28,
            add_29,
            add_3,
            add_31,
            add_32,
            add_33,
            add_35,
            add_36,
            add_37,
            add_39,
            add_4,
            add_40,
            add_41,
            add_43,
            add_44,
            add_45,
            add_47,
            add_5,
            add_7,
            add_8,
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
            assign_27,
            assign_28,
            assign_29,
            assign_3,
            assign_30,
            assign_31,
            assign_32,
            assign_33,
            assign_34,
            assign_35,
            assign_36,
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
            full_0,
            full_int_array_1,
            full_int_array_2,
            full_int_array_3,
            full_int_array_4,
            gelu_0,
            gelu_1,
            gelu_2,
            gelu_3,
            gelu_4,
            gelu_5,
            layer_norm_1,
            layer_norm_10,
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
            layer_norm_4,
            layer_norm_5,
            layer_norm_6,
            layer_norm_7,
            layer_norm_8,
            layer_norm_9,
            matmul_0,
            matmul_10,
            matmul_11,
            matmul_12,
            matmul_15,
            matmul_16,
            matmul_17,
            matmul_18,
            matmul_21,
            matmul_22,
            matmul_23,
            matmul_24,
            matmul_27,
            matmul_28,
            matmul_29,
            matmul_3,
            matmul_30,
            matmul_33,
            matmul_34,
            matmul_35,
            matmul_4,
            matmul_5,
            matmul_6,
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
            reshape_1,
            reshape_11,
            reshape_3,
            reshape_5,
            reshape_7,
            reshape_9,
            slice_0,
            slice_10,
            slice_12,
            slice_15,
            slice_17,
            slice_2,
            slice_20,
            slice_22,
            slice_25,
            slice_27,
            slice_5,
            slice_7,
            softmax_0,
            softmax_1,
            softmax_2,
            softmax_3,
            softmax_4,
            softmax_5,
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
            transpose_2,
            transpose_3,
            transpose_4,
            transpose_5,
            transpose_6,
            transpose_7,
            transpose_8,
            transpose_9,
        )

        return layer_norm_0
