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
        data_0,
        data_1,
        data_2,
        data_3,
        data_4,
    ):
        # pd_op.full: (1x28x21x1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1, 28, 21, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [0, 0]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_0 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_1 = full_int_array_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [-7, -7]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [1, 1]

        # pd_op.set_value_: (1x28x21x1xf32) <- (1x28x21x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__0 = paddle._C_ops.set_value_(
            full_0,
            full_int_array_0,
            full_int_array_1,
            full_int_array_2,
            [1, 2],
            [],
            [],
            [1],
            [float("0")],
        )
        del full_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [0, -7]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [-7, -3]

        # pd_op.set_value_: (1x28x21x1xf32) <- (1x28x21x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__1 = paddle._C_ops.set_value_(
            set_value__0,
            full_int_array_3,
            full_int_array_4,
            full_int_array_2,
            [1, 2],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_3, set_value__0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [0, -3]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_6 = [-7, 2147483647]

        # pd_op.set_value_: (1x28x21x1xf32) <- (1x28x21x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__2 = paddle._C_ops.set_value_(
            set_value__1,
            full_int_array_5,
            full_int_array_6,
            full_int_array_2,
            [1, 2],
            [],
            [],
            [1],
            [float("2")],
        )
        del full_int_array_5, full_int_array_6, set_value__1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_7 = [-7, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_8 = [-3, -7]

        # pd_op.set_value_: (1x28x21x1xf32) <- (1x28x21x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__3 = paddle._C_ops.set_value_(
            set_value__2,
            full_int_array_7,
            full_int_array_8,
            full_int_array_2,
            [1, 2],
            [],
            [],
            [1],
            [float("3")],
        )
        del full_int_array_7, set_value__2

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_9 = [-3, -3]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_2 = full_int_array_9

        # pd_op.set_value_: (1x28x21x1xf32) <- (1x28x21x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__4 = paddle._C_ops.set_value_(
            set_value__3,
            full_int_array_1,
            full_int_array_9,
            full_int_array_2,
            [1, 2],
            [],
            [],
            [1],
            [float("4")],
        )
        del full_int_array_1, set_value__3

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_10 = [-3, 2147483647]

        # pd_op.set_value_: (1x28x21x1xf32) <- (1x28x21x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__5 = paddle._C_ops.set_value_(
            set_value__4,
            full_int_array_4,
            full_int_array_10,
            full_int_array_2,
            [1, 2],
            [],
            [],
            [1],
            [float("5")],
        )
        del full_int_array_10, full_int_array_4, set_value__4

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_11 = [-3, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_12 = [2147483647, -7]

        # pd_op.set_value_: (1x28x21x1xf32) <- (1x28x21x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__6 = paddle._C_ops.set_value_(
            set_value__5,
            full_int_array_11,
            full_int_array_12,
            full_int_array_2,
            [1, 2],
            [],
            [],
            [1],
            [float("6")],
        )
        del full_int_array_11, full_int_array_12, set_value__5

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_13 = [2147483647, -3]

        # pd_op.set_value_: (1x28x21x1xf32) <- (1x28x21x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__7 = paddle._C_ops.set_value_(
            set_value__6,
            full_int_array_8,
            full_int_array_13,
            full_int_array_2,
            [1, 2],
            [],
            [],
            [1],
            [float("7")],
        )
        del full_int_array_13, full_int_array_8, set_value__6

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_14 = [2147483647, 2147483647]

        # pd_op.set_value_: (1x28x21x1xf32) <- (1x28x21x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__8 = paddle._C_ops.set_value_(
            set_value__7,
            full_int_array_9,
            full_int_array_14,
            full_int_array_2,
            [1, 2],
            [],
            [],
            [1],
            [float("8")],
        )
        del full_int_array_14, full_int_array_2, set_value__7

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_15 = [-1, 4, 7, 3, 7, 1]

        # pd_op.reshape: (1x4x7x3x7x1xf32) <- (1x28x21x1xf32, 6xi64)
        reshape_0 = paddle._C_ops.reshape(set_value__8, full_int_array_15)
        del full_int_array_15

        # pd_op.transpose: (1x4x3x7x7x1xf32) <- (1x4x7x3x7x1xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_0, [0, 1, 3, 2, 4, 5])
        del reshape_0

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_16 = [-1, 7, 7, 1]

        # pd_op.reshape: (12x7x7x1xf32) <- (1x4x3x7x7x1xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(transpose_0, full_int_array_16)
        del full_int_array_16, transpose_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_17 = [-1, 49]

        # pd_op.reshape: (12x49xf32) <- (12x7x7x1xf32, 2xi64)
        reshape_2 = paddle._C_ops.reshape(reshape_1, full_int_array_17)
        del full_int_array_17, reshape_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_18 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_3 = full_int_array_18

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_4 = full_int_array_18

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_5 = full_int_array_18

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_6 = full_int_array_18

        # pd_op.unsqueeze: (12x1x49xf32) <- (12x49xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(reshape_2, full_int_array_18)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_19 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_7 = full_int_array_19

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_8 = full_int_array_19

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_9 = full_int_array_19

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_10 = full_int_array_19

        # pd_op.unsqueeze: (12x49x1xf32) <- (12x49xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(reshape_2, full_int_array_19)
        del reshape_2

        # pd_op.subtract: (12x49x49xf32) <- (12x1x49xf32, 12x49x1xf32)
        subtract_0 = paddle._C_ops.subtract(unsqueeze_0, unsqueeze_1)
        del unsqueeze_0, unsqueeze_1

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (12x49x49xf32) <- (12x49x49xf32, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            subtract_0,
            full_1,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("-100"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (12x49x49xf32) <- (12x49x49xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(full_like_0, full_2, float("0"), True)
        del full_2, full_like_0

        # pd_op.full: (xf32) <- ()
        full_3 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.not_equal: (12x49x49xb) <- (12x49x49xf32, xf32)
        not_equal_0 = paddle._C_ops.not_equal(subtract_0, full_3)
        del full_3, subtract_0

        # pd_op.cast: (12x49x49xf32) <- (12x49x49xb)
        cast_0 = paddle._C_ops.cast(not_equal_0, paddle.float32)
        del not_equal_0

        # pd_op.multiply: (12x49x49xf32) <- (12x49x49xf32, 12x49x49xf32)
        multiply_0 = paddle._C_ops.multiply(scale_0, cast_0)
        del cast_0, scale_0

        # pd_op.layer_norm: (2x494x768xf32, 2x494xf32, 2x494xf32) <- (2x494x768xf32, 768xf32, 768xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                data_2, parameter_23, parameter_22, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_22, parameter_23

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_20 = [-1, 26, 19, 768]

        # pd_op.reshape: (2x26x19x768xf32) <- (2x494x768xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(layer_norm_0, full_int_array_20)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_11 = full_4

        # pd_op.pad: (2x28x21x768xf32) <- (2x26x19x768xf32, 1xf32)
        pad_0 = paddle._C_ops.pad(reshape_3, [0, 0, 0, 2, 0, 2, 0, 0], full_4)

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_21 = [-1, 4, 7, 3, 7, 768]

        # pd_op.reshape: (2x4x7x3x7x768xf32) <- (2x28x21x768xf32, 6xi64)
        reshape_4 = paddle._C_ops.reshape(pad_0, full_int_array_21)

        # pd_op.transpose: (2x4x3x7x7x768xf32) <- (2x4x7x3x7x768xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_4, [0, 1, 3, 2, 4, 5])
        del reshape_4

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_22 = [-1, 7, 7, 768]

        # pd_op.reshape: (24x7x7x768xf32) <- (2x4x3x7x7x768xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(transpose_1, full_int_array_22)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_23 = [24, 49, 768]

        # pd_op.reshape: (24x49x768xf32) <- (24x7x7x768xf32, 3xi64)
        reshape_6 = paddle._C_ops.reshape(reshape_5, full_int_array_23)

        # pd_op.matmul: (24x49x2304xf32) <- (24x49x768xf32, 768x2304xf32)
        matmul_0 = paddle._C_ops.matmul(reshape_6, parameter_21, False, False)
        del parameter_21

        # pd_op.add: (24x49x2304xf32) <- (24x49x2304xf32, 2304xf32)
        add_1 = paddle._C_ops.add(matmul_0, parameter_20)
        del parameter_20

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_24 = [-1, 49, 3, 24, 32]

        # pd_op.reshape: (24x49x3x24x32xf32) <- (24x49x2304xf32, 5xi64)
        reshape_7 = paddle._C_ops.reshape(add_1, full_int_array_24)

        # pd_op.transpose: (3x24x24x49x32xf32) <- (24x49x3x24x32xf32)
        transpose_2 = paddle._C_ops.transpose(reshape_7, [2, 0, 3, 1, 4])
        del reshape_7

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_25 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_12 = full_int_array_25

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_13 = full_int_array_25

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_14 = full_int_array_25

        # pd_op.slice: (24x24x49x32xf32) <- (3x24x24x49x32xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            transpose_2, [0], full_int_array_25, full_int_array_18, [1], [0]
        )

        # pd_op.slice: (24x24x49x32xf32) <- (3x24x24x49x32xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            transpose_2, [0], full_int_array_18, full_int_array_19, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_26 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_15 = full_int_array_26

        # pd_op.slice: (24x24x49x32xf32) <- (3x24x24x49x32xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            transpose_2, [0], full_int_array_19, full_int_array_26, [1], [0]
        )

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_16 = full_5

        # pd_op.scale: (24x24x49x32xf32) <- (24x24x49x32xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_0, full_5, float("0"), True)
        del slice_0

        # pd_op.transpose: (24x24x32x49xf32) <- (24x24x49x32xf32)
        transpose_3 = paddle._C_ops.transpose(slice_1, [0, 1, 3, 2])
        del slice_1

        # pd_op.matmul: (24x24x49x49xf32) <- (24x24x49x32xf32, 24x24x32x49xf32)
        matmul_1 = paddle._C_ops.matmul(scale_1, transpose_3, False, False)

        # pd_op.flatten: (2401xi64) <- (49x49xi64)
        flatten_0 = paddle._C_ops.flatten(data_3, 0, 1)
        del data_3

        # pd_op.index_select: (2401x24xf32) <- (169x24xf32, 2401xi64)
        index_select_0 = paddle._C_ops.index_select(data_0, flatten_0, 0)
        del data_0

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_27 = [49, 49, -1]

        # pd_op.reshape: (49x49x24xf32) <- (2401x24xf32, 3xi64)
        reshape_8 = paddle._C_ops.reshape(index_select_0, full_int_array_27)

        # pd_op.transpose: (24x49x49xf32) <- (49x49x24xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_8, [2, 0, 1])
        del reshape_8

        # pd_op.unsqueeze: (1x24x49x49xf32) <- (24x49x49xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(transpose_4, full_int_array_25)

        # pd_op.add: (24x24x49x49xf32) <- (24x24x49x49xf32, 1x24x49x49xf32)
        add_2 = paddle._C_ops.add(matmul_1, unsqueeze_2)

        # pd_op.softmax: (24x24x49x49xf32) <- (24x24x49x49xf32)
        softmax_0 = paddle._C_ops.softmax(add_2, -1)
        del add_2

        # pd_op.matmul: (24x24x49x32xf32) <- (24x24x49x49xf32, 24x24x49x32xf32)
        matmul_2 = paddle._C_ops.matmul(softmax_0, slice_2, False, False)

        # pd_op.transpose: (24x49x24x32xf32) <- (24x24x49x32xf32)
        transpose_5 = paddle._C_ops.transpose(matmul_2, [0, 2, 1, 3])
        del matmul_2

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_28 = [-1, 49, 768]

        # pd_op.reshape: (24x49x768xf32) <- (24x49x24x32xf32, 3xi64)
        reshape_9 = paddle._C_ops.reshape(transpose_5, full_int_array_28)

        # pd_op.matmul: (24x49x768xf32) <- (24x49x768xf32, 768x768xf32)
        matmul_3 = paddle._C_ops.matmul(reshape_9, parameter_19, False, False)
        del parameter_19

        # pd_op.add: (24x49x768xf32) <- (24x49x768xf32, 768xf32)
        add_3 = paddle._C_ops.add(matmul_3, parameter_18)
        del parameter_18

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_29 = [24, 7, 7, 768]

        # pd_op.reshape: (24x7x7x768xf32) <- (24x49x768xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(add_3, full_int_array_29)

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_30 = [-1, 4, 3, 7, 7, 768]

        # pd_op.reshape: (2x4x3x7x7x768xf32) <- (24x7x7x768xf32, 6xi64)
        reshape_11 = paddle._C_ops.reshape(reshape_10, full_int_array_30)

        # pd_op.transpose: (2x4x7x3x7x768xf32) <- (2x4x3x7x7x768xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_11, [0, 1, 3, 2, 4, 5])
        del reshape_11

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_31 = [-1, 28, 21, 768]

        # pd_op.reshape: (2x28x21x768xf32) <- (2x4x7x3x7x768xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(transpose_6, full_int_array_31)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_32 = [26, 19]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_17 = full_int_array_32

        # pd_op.slice: (2x26x19x768xf32) <- (2x28x21x768xf32, 2xi64, 2xi64)
        slice_3 = paddle._C_ops.slice(
            reshape_12, [1, 2], full_int_array_0, full_int_array_32, [1, 1], []
        )

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_33 = [-1, 494, 768]

        # pd_op.reshape: (2x494x768xf32) <- (2x26x19x768xf32, 3xi64)
        reshape_13 = paddle._C_ops.reshape(slice_3, full_int_array_33)

        # pd_op.full: (xf64) <- ()
        full_6 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_6,
            [],
            paddle.float64,
            [float("0.909091")],
            paddle.framework._current_expected_place(),
        )
        del full_6

        # pd_op.cast: (xf32) <- (xf64)
        cast_1 = paddle._C_ops.cast(assign_value__0, paddle.float32)
        del assign_value__0

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_34 = [2, 1, 1]

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_0 = paddle._C_ops.uniform(
            full_int_array_34,
            paddle.float32,
            full_4,
            full_1,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_4 = paddle._C_ops.add(cast_1, uniform_0)
        del uniform_0

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_0 = paddle._C_ops.floor(add_4)
        del add_4

        # pd_op.divide: (2x494x768xf32) <- (2x494x768xf32, xf32)
        divide_0 = paddle._C_ops.divide(reshape_13, cast_1)

        # pd_op.multiply: (2x494x768xf32) <- (2x494x768xf32, 2x1x1xf32)
        multiply_1 = paddle._C_ops.multiply(divide_0, floor_0)

        # pd_op.add: (2x494x768xf32) <- (2x494x768xf32, 2x494x768xf32)
        add_5 = paddle._C_ops.add(data_2, multiply_1)
        del data_2

        # pd_op.layer_norm: (2x494x768xf32, 2x494xf32, 2x494xf32) <- (2x494x768xf32, 768xf32, 768xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_5, parameter_17, parameter_16, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_16, parameter_17

        # pd_op.matmul: (2x494x3072xf32) <- (2x494x768xf32, 768x3072xf32)
        matmul_4 = paddle._C_ops.matmul(layer_norm_3, parameter_15, False, False)
        del parameter_15

        # pd_op.add: (2x494x3072xf32) <- (2x494x3072xf32, 3072xf32)
        add_6 = paddle._C_ops.add(matmul_4, parameter_14)
        del parameter_14

        # pd_op.gelu: (2x494x3072xf32) <- (2x494x3072xf32)
        gelu_0 = paddle._C_ops.gelu(add_6, False)

        # pd_op.matmul: (2x494x768xf32) <- (2x494x3072xf32, 3072x768xf32)
        matmul_5 = paddle._C_ops.matmul(gelu_0, parameter_13, False, False)
        del parameter_13

        # pd_op.add: (2x494x768xf32) <- (2x494x768xf32, 768xf32)
        add_7 = paddle._C_ops.add(matmul_5, parameter_12)
        del parameter_12

        # pd_op.full: (xf64) <- ()
        full_7 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__1 = paddle._C_ops.assign_value_(
            full_7,
            [],
            paddle.float64,
            [float("0.909091")],
            paddle.framework._current_expected_place(),
        )
        del full_7

        # pd_op.cast: (xf32) <- (xf64)
        cast_2 = paddle._C_ops.cast(assign_value__1, paddle.float32)
        del assign_value__1

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_1 = paddle._C_ops.uniform(
            full_int_array_34,
            paddle.float32,
            full_4,
            full_1,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_8 = paddle._C_ops.add(cast_2, uniform_1)
        del uniform_1

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_1 = paddle._C_ops.floor(add_8)
        del add_8

        # pd_op.divide: (2x494x768xf32) <- (2x494x768xf32, xf32)
        divide_1 = paddle._C_ops.divide(add_7, cast_2)

        # pd_op.multiply: (2x494x768xf32) <- (2x494x768xf32, 2x1x1xf32)
        multiply_2 = paddle._C_ops.multiply(divide_1, floor_1)

        # pd_op.add: (2x494x768xf32) <- (2x494x768xf32, 2x494x768xf32)
        add_9 = paddle._C_ops.add(add_5, multiply_2)

        # pd_op.layer_norm: (2x494x768xf32, 2x494xf32, 2x494xf32) <- (2x494x768xf32, 768xf32, 768xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_9, parameter_11, parameter_10, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_10, parameter_11

        # pd_op.reshape: (2x26x19x768xf32) <- (2x494x768xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(layer_norm_6, full_int_array_20)
        del full_int_array_20

        # pd_op.pad: (2x28x21x768xf32) <- (2x26x19x768xf32, 1xf32)
        pad_1 = paddle._C_ops.pad(reshape_14, [0, 0, 0, 2, 0, 2, 0, 0], full_4)

        # pd_op.roll: (2x28x21x768xf32) <- (2x28x21x768xf32, 2xi64)
        roll_0 = paddle._C_ops.roll(pad_1, full_int_array_9, [1, 2])
        del full_int_array_9

        # pd_op.reshape: (2x4x7x3x7x768xf32) <- (2x28x21x768xf32, 6xi64)
        reshape_15 = paddle._C_ops.reshape(roll_0, full_int_array_21)
        del full_int_array_21

        # pd_op.transpose: (2x4x3x7x7x768xf32) <- (2x4x7x3x7x768xf32)
        transpose_7 = paddle._C_ops.transpose(reshape_15, [0, 1, 3, 2, 4, 5])
        del reshape_15

        # pd_op.reshape: (24x7x7x768xf32) <- (2x4x3x7x7x768xf32, 4xi64)
        reshape_16 = paddle._C_ops.reshape(transpose_7, full_int_array_22)
        del full_int_array_22

        # pd_op.reshape: (24x49x768xf32) <- (24x7x7x768xf32, 3xi64)
        reshape_17 = paddle._C_ops.reshape(reshape_16, full_int_array_23)
        del full_int_array_23

        # pd_op.matmul: (24x49x2304xf32) <- (24x49x768xf32, 768x2304xf32)
        matmul_6 = paddle._C_ops.matmul(reshape_17, parameter_9, False, False)
        del parameter_9

        # pd_op.add: (24x49x2304xf32) <- (24x49x2304xf32, 2304xf32)
        add_10 = paddle._C_ops.add(matmul_6, parameter_8)
        del parameter_8

        # pd_op.reshape: (24x49x3x24x32xf32) <- (24x49x2304xf32, 5xi64)
        reshape_18 = paddle._C_ops.reshape(add_10, full_int_array_24)
        del full_int_array_24

        # pd_op.transpose: (3x24x24x49x32xf32) <- (24x49x3x24x32xf32)
        transpose_8 = paddle._C_ops.transpose(reshape_18, [2, 0, 3, 1, 4])
        del reshape_18

        # pd_op.slice: (24x24x49x32xf32) <- (3x24x24x49x32xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            transpose_8, [0], full_int_array_25, full_int_array_18, [1], [0]
        )

        # pd_op.slice: (24x24x49x32xf32) <- (3x24x24x49x32xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            transpose_8, [0], full_int_array_18, full_int_array_19, [1], [0]
        )

        # pd_op.slice: (24x24x49x32xf32) <- (3x24x24x49x32xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            transpose_8, [0], full_int_array_19, full_int_array_26, [1], [0]
        )
        del full_int_array_19

        # pd_op.scale: (24x24x49x32xf32) <- (24x24x49x32xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_4, full_5, float("0"), True)
        del slice_4

        # pd_op.transpose: (24x24x32x49xf32) <- (24x24x49x32xf32)
        transpose_9 = paddle._C_ops.transpose(slice_5, [0, 1, 3, 2])
        del slice_5

        # pd_op.matmul: (24x24x49x49xf32) <- (24x24x49x32xf32, 24x24x32x49xf32)
        matmul_7 = paddle._C_ops.matmul(scale_2, transpose_9, False, False)

        # pd_op.flatten: (2401xi64) <- (49x49xi64)
        flatten_1 = paddle._C_ops.flatten(data_4, 0, 1)
        del data_4

        # pd_op.index_select: (2401x24xf32) <- (169x24xf32, 2401xi64)
        index_select_1 = paddle._C_ops.index_select(data_1, flatten_1, 0)
        del data_1

        # pd_op.reshape: (49x49x24xf32) <- (2401x24xf32, 3xi64)
        reshape_19 = paddle._C_ops.reshape(index_select_1, full_int_array_27)
        del full_int_array_27

        # pd_op.transpose: (24x49x49xf32) <- (49x49x24xf32)
        transpose_10 = paddle._C_ops.transpose(reshape_19, [2, 0, 1])
        del reshape_19

        # pd_op.unsqueeze: (1x24x49x49xf32) <- (24x49x49xf32, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(transpose_10, full_int_array_25)

        # pd_op.add: (24x24x49x49xf32) <- (24x24x49x49xf32, 1x24x49x49xf32)
        add_11 = paddle._C_ops.add(matmul_7, unsqueeze_3)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_35 = [-1, 12, 24, 49, 49]

        # pd_op.reshape: (2x12x24x49x49xf32) <- (24x24x49x49xf32, 5xi64)
        reshape_20 = paddle._C_ops.reshape(add_11, full_int_array_35)
        del full_int_array_35

        # pd_op.unsqueeze: (12x1x49x49xf32) <- (12x49x49xf32, 1xi64)
        unsqueeze_4 = paddle._C_ops.unsqueeze(multiply_0, full_int_array_18)
        del full_int_array_18, multiply_0

        # pd_op.unsqueeze: (1x12x1x49x49xf32) <- (12x1x49x49xf32, 1xi64)
        unsqueeze_5 = paddle._C_ops.unsqueeze(unsqueeze_4, full_int_array_25)
        del unsqueeze_4

        # pd_op.add: (2x12x24x49x49xf32) <- (2x12x24x49x49xf32, 1x12x1x49x49xf32)
        add_12 = paddle._C_ops.add(reshape_20, unsqueeze_5)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_36 = [-1, 24, 49, 49]

        # pd_op.reshape: (24x24x49x49xf32) <- (2x12x24x49x49xf32, 4xi64)
        reshape_21 = paddle._C_ops.reshape(add_12, full_int_array_36)
        del full_int_array_36

        # pd_op.softmax: (24x24x49x49xf32) <- (24x24x49x49xf32)
        softmax_1 = paddle._C_ops.softmax(reshape_21, -1)
        del reshape_21

        # pd_op.matmul: (24x24x49x32xf32) <- (24x24x49x49xf32, 24x24x49x32xf32)
        matmul_8 = paddle._C_ops.matmul(softmax_1, slice_6, False, False)

        # pd_op.transpose: (24x49x24x32xf32) <- (24x24x49x32xf32)
        transpose_11 = paddle._C_ops.transpose(matmul_8, [0, 2, 1, 3])
        del matmul_8

        # pd_op.reshape: (24x49x768xf32) <- (24x49x24x32xf32, 3xi64)
        reshape_22 = paddle._C_ops.reshape(transpose_11, full_int_array_28)
        del full_int_array_28

        # pd_op.matmul: (24x49x768xf32) <- (24x49x768xf32, 768x768xf32)
        matmul_9 = paddle._C_ops.matmul(reshape_22, parameter_7, False, False)
        del parameter_7

        # pd_op.add: (24x49x768xf32) <- (24x49x768xf32, 768xf32)
        add_13 = paddle._C_ops.add(matmul_9, parameter_6)
        del parameter_6

        # pd_op.reshape: (24x7x7x768xf32) <- (24x49x768xf32, 4xi64)
        reshape_23 = paddle._C_ops.reshape(add_13, full_int_array_29)
        del full_int_array_29

        # pd_op.reshape: (2x4x3x7x7x768xf32) <- (24x7x7x768xf32, 6xi64)
        reshape_24 = paddle._C_ops.reshape(reshape_23, full_int_array_30)
        del full_int_array_30

        # pd_op.transpose: (2x4x7x3x7x768xf32) <- (2x4x3x7x7x768xf32)
        transpose_12 = paddle._C_ops.transpose(reshape_24, [0, 1, 3, 2, 4, 5])
        del reshape_24

        # pd_op.reshape: (2x28x21x768xf32) <- (2x4x7x3x7x768xf32, 4xi64)
        reshape_25 = paddle._C_ops.reshape(transpose_12, full_int_array_31)
        del full_int_array_31

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_37 = [3, 3]

        # pd_op.roll: (2x28x21x768xf32) <- (2x28x21x768xf32, 2xi64)
        roll_1 = paddle._C_ops.roll(reshape_25, full_int_array_37, [1, 2])

        # pd_op.slice: (2x26x19x768xf32) <- (2x28x21x768xf32, 2xi64, 2xi64)
        slice_7 = paddle._C_ops.slice(
            roll_1, [1, 2], full_int_array_0, full_int_array_32, [1, 1], []
        )
        del full_int_array_0

        # pd_op.reshape: (2x494x768xf32) <- (2x26x19x768xf32, 3xi64)
        reshape_26 = paddle._C_ops.reshape(slice_7, full_int_array_33)
        del full_int_array_33

        # pd_op.full: (xf64) <- ()
        full_8 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__2 = paddle._C_ops.assign_value_(
            full_8,
            [],
            paddle.float64,
            [float("0.9")],
            paddle.framework._current_expected_place(),
        )
        del full_8

        # pd_op.cast: (xf32) <- (xf64)
        cast_3 = paddle._C_ops.cast(assign_value__2, paddle.float32)
        del assign_value__2

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_2 = paddle._C_ops.uniform(
            full_int_array_34,
            paddle.float32,
            full_4,
            full_1,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_14 = paddle._C_ops.add(cast_3, uniform_2)
        del uniform_2

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_2 = paddle._C_ops.floor(add_14)
        del add_14

        # pd_op.divide: (2x494x768xf32) <- (2x494x768xf32, xf32)
        divide_2 = paddle._C_ops.divide(reshape_26, cast_3)

        # pd_op.multiply: (2x494x768xf32) <- (2x494x768xf32, 2x1x1xf32)
        multiply_3 = paddle._C_ops.multiply(divide_2, floor_2)

        # pd_op.add: (2x494x768xf32) <- (2x494x768xf32, 2x494x768xf32)
        add_15 = paddle._C_ops.add(add_9, multiply_3)

        # pd_op.layer_norm: (2x494x768xf32, 2x494xf32, 2x494xf32) <- (2x494x768xf32, 768xf32, 768xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_15, parameter_5, parameter_4, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_4, parameter_5

        # pd_op.matmul: (2x494x3072xf32) <- (2x494x768xf32, 768x3072xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_9, parameter_3, False, False)
        del parameter_3

        # pd_op.add: (2x494x3072xf32) <- (2x494x3072xf32, 3072xf32)
        add_16 = paddle._C_ops.add(matmul_10, parameter_2)
        del parameter_2

        # pd_op.gelu: (2x494x3072xf32) <- (2x494x3072xf32)
        gelu_1 = paddle._C_ops.gelu(add_16, False)

        # pd_op.matmul: (2x494x768xf32) <- (2x494x3072xf32, 3072x768xf32)
        matmul_11 = paddle._C_ops.matmul(gelu_1, parameter_1, False, False)
        del parameter_1

        # pd_op.add: (2x494x768xf32) <- (2x494x768xf32, 768xf32)
        add_17 = paddle._C_ops.add(matmul_11, parameter_0)
        del parameter_0

        # pd_op.full: (xf64) <- ()
        full_9 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__3 = paddle._C_ops.assign_value_(
            full_9,
            [],
            paddle.float64,
            [float("0.9")],
            paddle.framework._current_expected_place(),
        )
        del full_9

        # pd_op.cast: (xf32) <- (xf64)
        cast_4 = paddle._C_ops.cast(assign_value__3, paddle.float32)
        del assign_value__3

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_3 = paddle._C_ops.uniform(
            full_int_array_34,
            paddle.float32,
            full_4,
            full_1,
            0,
            paddle.framework._current_expected_place(),
        )
        del full_1, full_int_array_34

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_18 = paddle._C_ops.add(cast_4, uniform_3)
        del uniform_3

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_3 = paddle._C_ops.floor(add_18)
        del add_18

        # pd_op.divide: (2x494x768xf32) <- (2x494x768xf32, xf32)
        divide_3 = paddle._C_ops.divide(add_17, cast_4)

        # pd_op.multiply: (2x494x768xf32) <- (2x494x768xf32, 2x1x1xf32)
        multiply_4 = paddle._C_ops.multiply(divide_3, floor_3)

        # pd_op.add: (2x494x768xf32) <- (2x494x768xf32, 2x494x768xf32)
        add_0 = paddle._C_ops.add(add_15, multiply_4)
        del (
            add_1,
            add_10,
            add_11,
            add_12,
            add_13,
            add_15,
            add_16,
            add_17,
            add_3,
            add_5,
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
            assign_2,
            assign_3,
            assign_4,
            assign_5,
            assign_6,
            assign_7,
            assign_8,
            assign_9,
            cast_1,
            cast_2,
            cast_3,
            cast_4,
            divide_0,
            divide_1,
            divide_2,
            divide_3,
            flatten_0,
            flatten_1,
            floor_0,
            floor_1,
            floor_2,
            floor_3,
            full_4,
            full_5,
            full_int_array_25,
            full_int_array_26,
            full_int_array_32,
            full_int_array_37,
            gelu_0,
            gelu_1,
            index_select_0,
            index_select_1,
            layer_norm_0,
            layer_norm_1,
            layer_norm_10,
            layer_norm_11,
            layer_norm_2,
            layer_norm_3,
            layer_norm_4,
            layer_norm_5,
            layer_norm_6,
            layer_norm_7,
            layer_norm_8,
            layer_norm_9,
            matmul_0,
            matmul_1,
            matmul_10,
            matmul_11,
            matmul_3,
            matmul_4,
            matmul_5,
            matmul_6,
            matmul_7,
            matmul_9,
            multiply_1,
            multiply_2,
            multiply_3,
            multiply_4,
            pad_0,
            pad_1,
            reshape_10,
            reshape_12,
            reshape_13,
            reshape_14,
            reshape_16,
            reshape_17,
            reshape_20,
            reshape_22,
            reshape_23,
            reshape_25,
            reshape_26,
            reshape_3,
            reshape_5,
            reshape_6,
            reshape_9,
            roll_0,
            roll_1,
            scale_1,
            scale_2,
            set_value__8,
            slice_2,
            slice_3,
            slice_6,
            slice_7,
            softmax_0,
            softmax_1,
            transpose_1,
            transpose_10,
            transpose_11,
            transpose_12,
            transpose_2,
            transpose_3,
            transpose_4,
            transpose_5,
            transpose_6,
            transpose_7,
            transpose_8,
            transpose_9,
            unsqueeze_2,
            unsqueeze_3,
            unsqueeze_5,
        )

        return add_0
