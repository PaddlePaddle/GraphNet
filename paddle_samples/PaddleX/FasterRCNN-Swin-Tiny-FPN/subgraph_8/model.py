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
        parameter_72,
        parameter_73,
        parameter_74,
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
        data_10,
        data_11,
        data_12,
    ):
        # pd_op.full: (1x56x42x1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1, 56, 42, 1],
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

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_2 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_3 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_4 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_5 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_6 = full_int_array_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [-7, -7]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [1, 1]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_7 = full_int_array_2

        # pd_op.set_value_: (1x56x42x1xf32) <- (1x56x42x1xf32, 2xi64, 2xi64, 2xi64)
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

        # pd_op.set_value_: (1x56x42x1xf32) <- (1x56x42x1xf32, 2xi64, 2xi64, 2xi64)
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

        # pd_op.set_value_: (1x56x42x1xf32) <- (1x56x42x1xf32, 2xi64, 2xi64, 2xi64)
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

        # pd_op.set_value_: (1x56x42x1xf32) <- (1x56x42x1xf32, 2xi64, 2xi64, 2xi64)
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
        assign_8 = full_int_array_9

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_9 = full_int_array_9

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_10 = full_int_array_9

        # pd_op.set_value_: (1x56x42x1xf32) <- (1x56x42x1xf32, 2xi64, 2xi64, 2xi64)
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

        # pd_op.set_value_: (1x56x42x1xf32) <- (1x56x42x1xf32, 2xi64, 2xi64, 2xi64)
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

        # pd_op.set_value_: (1x56x42x1xf32) <- (1x56x42x1xf32, 2xi64, 2xi64, 2xi64)
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

        # pd_op.set_value_: (1x56x42x1xf32) <- (1x56x42x1xf32, 2xi64, 2xi64, 2xi64)
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

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_11 = full_int_array_14

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_12 = full_int_array_14

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_13 = full_int_array_14

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_14 = full_int_array_14

        # pd_op.set_value_: (1x56x42x1xf32) <- (1x56x42x1xf32, 2xi64, 2xi64, 2xi64)
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
        del set_value__7

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_15 = [-1, 8, 7, 6, 7, 1]

        # pd_op.reshape: (1x8x7x6x7x1xf32) <- (1x56x42x1xf32, 6xi64)
        reshape_0 = paddle._C_ops.reshape(set_value__8, full_int_array_15)
        del full_int_array_15

        # pd_op.transpose: (1x8x6x7x7x1xf32) <- (1x8x7x6x7x1xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_0, [0, 1, 3, 2, 4, 5])
        del reshape_0

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_16 = [-1, 7, 7, 1]

        # pd_op.reshape: (48x7x7x1xf32) <- (1x8x6x7x7x1xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(transpose_0, full_int_array_16)
        del full_int_array_16, transpose_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_17 = [-1, 49]

        # pd_op.reshape: (48x49xf32) <- (48x7x7x1xf32, 2xi64)
        reshape_2 = paddle._C_ops.reshape(reshape_1, full_int_array_17)
        del full_int_array_17, reshape_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_18 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_15 = full_int_array_18

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_16 = full_int_array_18

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_17 = full_int_array_18

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_18 = full_int_array_18

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_19 = full_int_array_18

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_20 = full_int_array_18

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_21 = full_int_array_18

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_22 = full_int_array_18

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_23 = full_int_array_18

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_24 = full_int_array_18

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_25 = full_int_array_18

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_26 = full_int_array_18

        # pd_op.unsqueeze: (48x1x49xf32) <- (48x49xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(reshape_2, full_int_array_18)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_19 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_27 = full_int_array_19

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_28 = full_int_array_19

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_29 = full_int_array_19

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_30 = full_int_array_19

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_31 = full_int_array_19

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_32 = full_int_array_19

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_33 = full_int_array_19

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_34 = full_int_array_19

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_35 = full_int_array_19

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_36 = full_int_array_19

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_37 = full_int_array_19

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_38 = full_int_array_19

        # pd_op.unsqueeze: (48x49x1xf32) <- (48x49xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(reshape_2, full_int_array_19)
        del reshape_2

        # pd_op.subtract: (48x49x49xf32) <- (48x1x49xf32, 48x49x1xf32)
        subtract_0 = paddle._C_ops.subtract(unsqueeze_0, unsqueeze_1)
        del unsqueeze_0, unsqueeze_1

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (48x49x49xf32) <- (48x49x49xf32, 1xf32)
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

        # pd_op.scale: (48x49x49xf32) <- (48x49x49xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(full_like_0, full_2, float("0"), True)
        del full_2, full_like_0

        # pd_op.full: (xf32) <- ()
        full_3 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.not_equal: (48x49x49xb) <- (48x49x49xf32, xf32)
        not_equal_0 = paddle._C_ops.not_equal(subtract_0, full_3)
        del full_3, subtract_0

        # pd_op.cast: (48x49x49xf32) <- (48x49x49xb)
        cast_0 = paddle._C_ops.cast(not_equal_0, paddle.float32)
        del not_equal_0

        # pd_op.multiply: (48x49x49xf32) <- (48x49x49xf32, 48x49x49xf32)
        multiply_0 = paddle._C_ops.multiply(scale_0, cast_0)
        del cast_0, scale_0

        # pd_op.layer_norm: (2x1976x384xf32, 2x1976xf32, 2x1976xf32) <- (2x1976x384xf32, 384xf32, 384xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                data_6, parameter_74, parameter_73, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_73, parameter_74

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_20 = [-1, 52, 38, 384]

        # pd_op.reshape: (2x52x38x384xf32) <- (2x1976x384xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(layer_norm_0, full_int_array_20)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_39 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_40 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_41 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_42 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_43 = full_4

        # pd_op.pad: (2x56x42x384xf32) <- (2x52x38x384xf32, 1xf32)
        pad_0 = paddle._C_ops.pad(reshape_3, [0, 0, 0, 4, 0, 4, 0, 0], full_4)

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_21 = [-1, 8, 7, 6, 7, 384]

        # pd_op.reshape: (2x8x7x6x7x384xf32) <- (2x56x42x384xf32, 6xi64)
        reshape_4 = paddle._C_ops.reshape(pad_0, full_int_array_21)

        # pd_op.transpose: (2x8x6x7x7x384xf32) <- (2x8x7x6x7x384xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_4, [0, 1, 3, 2, 4, 5])
        del reshape_4

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_22 = [-1, 7, 7, 384]

        # pd_op.reshape: (96x7x7x384xf32) <- (2x8x6x7x7x384xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(transpose_1, full_int_array_22)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_23 = [96, 49, 384]

        # pd_op.reshape: (96x49x384xf32) <- (96x7x7x384xf32, 3xi64)
        reshape_6 = paddle._C_ops.reshape(reshape_5, full_int_array_23)

        # pd_op.matmul: (96x49x1152xf32) <- (96x49x384xf32, 384x1152xf32)
        matmul_1 = paddle._C_ops.matmul(reshape_6, parameter_72, False, False)
        del parameter_72

        # pd_op.add: (96x49x1152xf32) <- (96x49x1152xf32, 1152xf32)
        add_0 = paddle._C_ops.add(matmul_1, parameter_71)
        del parameter_71

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_24 = [-1, 49, 3, 12, 32]

        # pd_op.reshape: (96x49x3x12x32xf32) <- (96x49x1152xf32, 5xi64)
        reshape_7 = paddle._C_ops.reshape(add_0, full_int_array_24)

        # pd_op.transpose: (3x96x12x49x32xf32) <- (96x49x3x12x32xf32)
        transpose_2 = paddle._C_ops.transpose(reshape_7, [2, 0, 3, 1, 4])
        del reshape_7

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_25 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_44 = full_int_array_25

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_45 = full_int_array_25

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_46 = full_int_array_25

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_47 = full_int_array_25

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_48 = full_int_array_25

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_49 = full_int_array_25

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_50 = full_int_array_25

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_51 = full_int_array_25

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_52 = full_int_array_25

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_53 = full_int_array_25

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_54 = full_int_array_25

        # pd_op.slice: (96x12x49x32xf32) <- (3x96x12x49x32xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            transpose_2, [0], full_int_array_25, full_int_array_18, [1], [0]
        )

        # pd_op.slice: (96x12x49x32xf32) <- (3x96x12x49x32xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            transpose_2, [0], full_int_array_18, full_int_array_19, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_26 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_55 = full_int_array_26

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_56 = full_int_array_26

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_57 = full_int_array_26

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_58 = full_int_array_26

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_59 = full_int_array_26

        # pd_op.slice: (96x12x49x32xf32) <- (3x96x12x49x32xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            transpose_2, [0], full_int_array_19, full_int_array_26, [1], [0]
        )

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_60 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_61 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_62 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_63 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_64 = full_5

        # pd_op.scale: (96x12x49x32xf32) <- (96x12x49x32xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_0, full_5, float("0"), True)
        del slice_0

        # pd_op.transpose: (96x12x32x49xf32) <- (96x12x49x32xf32)
        transpose_3 = paddle._C_ops.transpose(slice_1, [0, 1, 3, 2])
        del slice_1

        # pd_op.matmul: (96x12x49x49xf32) <- (96x12x49x32xf32, 96x12x32x49xf32)
        matmul_2 = paddle._C_ops.matmul(scale_1, transpose_3, False, False)

        # pd_op.flatten: (2401xi64) <- (49x49xi64)
        flatten_0 = paddle._C_ops.flatten(data_7, 0, 1)
        del data_7

        # pd_op.index_select: (2401x12xf32) <- (169x12xf32, 2401xi64)
        index_select_0 = paddle._C_ops.index_select(data_0, flatten_0, 0)
        del data_0

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_27 = [49, 49, -1]

        # pd_op.reshape: (49x49x12xf32) <- (2401x12xf32, 3xi64)
        reshape_8 = paddle._C_ops.reshape(index_select_0, full_int_array_27)

        # pd_op.transpose: (12x49x49xf32) <- (49x49x12xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_8, [2, 0, 1])
        del reshape_8

        # pd_op.unsqueeze: (1x12x49x49xf32) <- (12x49x49xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(transpose_4, full_int_array_25)

        # pd_op.add: (96x12x49x49xf32) <- (96x12x49x49xf32, 1x12x49x49xf32)
        add_1 = paddle._C_ops.add(matmul_2, unsqueeze_2)

        # pd_op.softmax: (96x12x49x49xf32) <- (96x12x49x49xf32)
        softmax_0 = paddle._C_ops.softmax(add_1, -1)
        del add_1

        # pd_op.matmul: (96x12x49x32xf32) <- (96x12x49x49xf32, 96x12x49x32xf32)
        matmul_3 = paddle._C_ops.matmul(softmax_0, slice_2, False, False)

        # pd_op.transpose: (96x49x12x32xf32) <- (96x12x49x32xf32)
        transpose_5 = paddle._C_ops.transpose(matmul_3, [0, 2, 1, 3])
        del matmul_3

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_28 = [-1, 49, 384]

        # pd_op.reshape: (96x49x384xf32) <- (96x49x12x32xf32, 3xi64)
        reshape_9 = paddle._C_ops.reshape(transpose_5, full_int_array_28)

        # pd_op.matmul: (96x49x384xf32) <- (96x49x384xf32, 384x384xf32)
        matmul_4 = paddle._C_ops.matmul(reshape_9, parameter_70, False, False)
        del parameter_70

        # pd_op.add: (96x49x384xf32) <- (96x49x384xf32, 384xf32)
        add_2 = paddle._C_ops.add(matmul_4, parameter_69)
        del parameter_69

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_29 = [96, 7, 7, 384]

        # pd_op.reshape: (96x7x7x384xf32) <- (96x49x384xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(add_2, full_int_array_29)

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_30 = [-1, 8, 6, 7, 7, 384]

        # pd_op.reshape: (2x8x6x7x7x384xf32) <- (96x7x7x384xf32, 6xi64)
        reshape_11 = paddle._C_ops.reshape(reshape_10, full_int_array_30)

        # pd_op.transpose: (2x8x7x6x7x384xf32) <- (2x8x6x7x7x384xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_11, [0, 1, 3, 2, 4, 5])
        del reshape_11

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_31 = [-1, 56, 42, 384]

        # pd_op.reshape: (2x56x42x384xf32) <- (2x8x7x6x7x384xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(transpose_6, full_int_array_31)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_32 = [52, 38]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_65 = full_int_array_32

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_66 = full_int_array_32

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_67 = full_int_array_32

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_68 = full_int_array_32

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_69 = full_int_array_32

        # pd_op.slice: (2x52x38x384xf32) <- (2x56x42x384xf32, 2xi64, 2xi64)
        slice_3 = paddle._C_ops.slice(
            reshape_12, [1, 2], full_int_array_0, full_int_array_32, [1, 1], []
        )

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_33 = [-1, 1976, 384]

        # pd_op.reshape: (2x1976x384xf32) <- (2x52x38x384xf32, 3xi64)
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
            [float("0.963636")],
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
        add_3 = paddle._C_ops.add(cast_1, uniform_0)
        del uniform_0

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_0 = paddle._C_ops.floor(add_3)
        del add_3

        # pd_op.divide: (2x1976x384xf32) <- (2x1976x384xf32, xf32)
        divide_0 = paddle._C_ops.divide(reshape_13, cast_1)

        # pd_op.multiply: (2x1976x384xf32) <- (2x1976x384xf32, 2x1x1xf32)
        multiply_1 = paddle._C_ops.multiply(divide_0, floor_0)

        # pd_op.add: (2x1976x384xf32) <- (2x1976x384xf32, 2x1976x384xf32)
        add_4 = paddle._C_ops.add(data_6, multiply_1)
        del data_6

        # pd_op.layer_norm: (2x1976x384xf32, 2x1976xf32, 2x1976xf32) <- (2x1976x384xf32, 384xf32, 384xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_4, parameter_68, parameter_67, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_67, parameter_68

        # pd_op.matmul: (2x1976x1536xf32) <- (2x1976x384xf32, 384x1536xf32)
        matmul_5 = paddle._C_ops.matmul(layer_norm_3, parameter_66, False, False)
        del parameter_66

        # pd_op.add: (2x1976x1536xf32) <- (2x1976x1536xf32, 1536xf32)
        add_5 = paddle._C_ops.add(matmul_5, parameter_65)
        del parameter_65

        # pd_op.gelu: (2x1976x1536xf32) <- (2x1976x1536xf32)
        gelu_0 = paddle._C_ops.gelu(add_5, False)

        # pd_op.matmul: (2x1976x384xf32) <- (2x1976x1536xf32, 1536x384xf32)
        matmul_6 = paddle._C_ops.matmul(gelu_0, parameter_64, False, False)
        del parameter_64

        # pd_op.add: (2x1976x384xf32) <- (2x1976x384xf32, 384xf32)
        add_6 = paddle._C_ops.add(matmul_6, parameter_63)
        del parameter_63

        # pd_op.full: (xf64) <- ()
        full_7 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__1 = paddle._C_ops.assign_value_(
            full_7,
            [],
            paddle.float64,
            [float("0.963636")],
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
        add_7 = paddle._C_ops.add(cast_2, uniform_1)
        del uniform_1

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_1 = paddle._C_ops.floor(add_7)
        del add_7

        # pd_op.divide: (2x1976x384xf32) <- (2x1976x384xf32, xf32)
        divide_1 = paddle._C_ops.divide(add_6, cast_2)

        # pd_op.multiply: (2x1976x384xf32) <- (2x1976x384xf32, 2x1x1xf32)
        multiply_2 = paddle._C_ops.multiply(divide_1, floor_1)

        # pd_op.add: (2x1976x384xf32) <- (2x1976x384xf32, 2x1976x384xf32)
        add_8 = paddle._C_ops.add(add_4, multiply_2)

        # pd_op.layer_norm: (2x1976x384xf32, 2x1976xf32, 2x1976xf32) <- (2x1976x384xf32, 384xf32, 384xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_8, parameter_62, parameter_61, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_61, parameter_62

        # pd_op.reshape: (2x52x38x384xf32) <- (2x1976x384xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(layer_norm_6, full_int_array_20)

        # pd_op.pad: (2x56x42x384xf32) <- (2x52x38x384xf32, 1xf32)
        pad_1 = paddle._C_ops.pad(reshape_14, [0, 0, 0, 4, 0, 4, 0, 0], full_4)

        # pd_op.roll: (2x56x42x384xf32) <- (2x56x42x384xf32, 2xi64)
        roll_0 = paddle._C_ops.roll(pad_1, full_int_array_9, [1, 2])

        # pd_op.reshape: (2x8x7x6x7x384xf32) <- (2x56x42x384xf32, 6xi64)
        reshape_15 = paddle._C_ops.reshape(roll_0, full_int_array_21)

        # pd_op.transpose: (2x8x6x7x7x384xf32) <- (2x8x7x6x7x384xf32)
        transpose_7 = paddle._C_ops.transpose(reshape_15, [0, 1, 3, 2, 4, 5])
        del reshape_15

        # pd_op.reshape: (96x7x7x384xf32) <- (2x8x6x7x7x384xf32, 4xi64)
        reshape_16 = paddle._C_ops.reshape(transpose_7, full_int_array_22)

        # pd_op.reshape: (96x49x384xf32) <- (96x7x7x384xf32, 3xi64)
        reshape_17 = paddle._C_ops.reshape(reshape_16, full_int_array_23)

        # pd_op.matmul: (96x49x1152xf32) <- (96x49x384xf32, 384x1152xf32)
        matmul_7 = paddle._C_ops.matmul(reshape_17, parameter_60, False, False)
        del parameter_60

        # pd_op.add: (96x49x1152xf32) <- (96x49x1152xf32, 1152xf32)
        add_9 = paddle._C_ops.add(matmul_7, parameter_59)
        del parameter_59

        # pd_op.reshape: (96x49x3x12x32xf32) <- (96x49x1152xf32, 5xi64)
        reshape_18 = paddle._C_ops.reshape(add_9, full_int_array_24)

        # pd_op.transpose: (3x96x12x49x32xf32) <- (96x49x3x12x32xf32)
        transpose_8 = paddle._C_ops.transpose(reshape_18, [2, 0, 3, 1, 4])
        del reshape_18

        # pd_op.slice: (96x12x49x32xf32) <- (3x96x12x49x32xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            transpose_8, [0], full_int_array_25, full_int_array_18, [1], [0]
        )

        # pd_op.slice: (96x12x49x32xf32) <- (3x96x12x49x32xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            transpose_8, [0], full_int_array_18, full_int_array_19, [1], [0]
        )

        # pd_op.slice: (96x12x49x32xf32) <- (3x96x12x49x32xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            transpose_8, [0], full_int_array_19, full_int_array_26, [1], [0]
        )

        # pd_op.scale: (96x12x49x32xf32) <- (96x12x49x32xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_4, full_5, float("0"), True)
        del slice_4

        # pd_op.transpose: (96x12x32x49xf32) <- (96x12x49x32xf32)
        transpose_9 = paddle._C_ops.transpose(slice_5, [0, 1, 3, 2])
        del slice_5

        # pd_op.matmul: (96x12x49x49xf32) <- (96x12x49x32xf32, 96x12x32x49xf32)
        matmul_8 = paddle._C_ops.matmul(scale_2, transpose_9, False, False)

        # pd_op.flatten: (2401xi64) <- (49x49xi64)
        flatten_1 = paddle._C_ops.flatten(data_8, 0, 1)
        del data_8

        # pd_op.index_select: (2401x12xf32) <- (169x12xf32, 2401xi64)
        index_select_1 = paddle._C_ops.index_select(data_1, flatten_1, 0)
        del data_1

        # pd_op.reshape: (49x49x12xf32) <- (2401x12xf32, 3xi64)
        reshape_19 = paddle._C_ops.reshape(index_select_1, full_int_array_27)

        # pd_op.transpose: (12x49x49xf32) <- (49x49x12xf32)
        transpose_10 = paddle._C_ops.transpose(reshape_19, [2, 0, 1])
        del reshape_19

        # pd_op.unsqueeze: (1x12x49x49xf32) <- (12x49x49xf32, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(transpose_10, full_int_array_25)

        # pd_op.add: (96x12x49x49xf32) <- (96x12x49x49xf32, 1x12x49x49xf32)
        add_10 = paddle._C_ops.add(matmul_8, unsqueeze_3)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_35 = [-1, 48, 12, 49, 49]

        # pd_op.reshape: (2x48x12x49x49xf32) <- (96x12x49x49xf32, 5xi64)
        reshape_20 = paddle._C_ops.reshape(add_10, full_int_array_35)

        # pd_op.unsqueeze: (48x1x49x49xf32) <- (48x49x49xf32, 1xi64)
        unsqueeze_4 = paddle._C_ops.unsqueeze(multiply_0, full_int_array_18)

        # pd_op.unsqueeze: (1x48x1x49x49xf32) <- (48x1x49x49xf32, 1xi64)
        unsqueeze_5 = paddle._C_ops.unsqueeze(unsqueeze_4, full_int_array_25)
        del unsqueeze_4

        # pd_op.add: (2x48x12x49x49xf32) <- (2x48x12x49x49xf32, 1x48x1x49x49xf32)
        add_11 = paddle._C_ops.add(reshape_20, unsqueeze_5)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_36 = [-1, 12, 49, 49]

        # pd_op.reshape: (96x12x49x49xf32) <- (2x48x12x49x49xf32, 4xi64)
        reshape_21 = paddle._C_ops.reshape(add_11, full_int_array_36)

        # pd_op.softmax: (96x12x49x49xf32) <- (96x12x49x49xf32)
        softmax_1 = paddle._C_ops.softmax(reshape_21, -1)
        del reshape_21

        # pd_op.matmul: (96x12x49x32xf32) <- (96x12x49x49xf32, 96x12x49x32xf32)
        matmul_9 = paddle._C_ops.matmul(softmax_1, slice_6, False, False)

        # pd_op.transpose: (96x49x12x32xf32) <- (96x12x49x32xf32)
        transpose_11 = paddle._C_ops.transpose(matmul_9, [0, 2, 1, 3])
        del matmul_9

        # pd_op.reshape: (96x49x384xf32) <- (96x49x12x32xf32, 3xi64)
        reshape_22 = paddle._C_ops.reshape(transpose_11, full_int_array_28)

        # pd_op.matmul: (96x49x384xf32) <- (96x49x384xf32, 384x384xf32)
        matmul_10 = paddle._C_ops.matmul(reshape_22, parameter_58, False, False)
        del parameter_58

        # pd_op.add: (96x49x384xf32) <- (96x49x384xf32, 384xf32)
        add_12 = paddle._C_ops.add(matmul_10, parameter_57)
        del parameter_57

        # pd_op.reshape: (96x7x7x384xf32) <- (96x49x384xf32, 4xi64)
        reshape_23 = paddle._C_ops.reshape(add_12, full_int_array_29)

        # pd_op.reshape: (2x8x6x7x7x384xf32) <- (96x7x7x384xf32, 6xi64)
        reshape_24 = paddle._C_ops.reshape(reshape_23, full_int_array_30)

        # pd_op.transpose: (2x8x7x6x7x384xf32) <- (2x8x6x7x7x384xf32)
        transpose_12 = paddle._C_ops.transpose(reshape_24, [0, 1, 3, 2, 4, 5])
        del reshape_24

        # pd_op.reshape: (2x56x42x384xf32) <- (2x8x7x6x7x384xf32, 4xi64)
        reshape_25 = paddle._C_ops.reshape(transpose_12, full_int_array_31)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_37 = [3, 3]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_70 = full_int_array_37

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_71 = full_int_array_37

        # pd_op.roll: (2x56x42x384xf32) <- (2x56x42x384xf32, 2xi64)
        roll_1 = paddle._C_ops.roll(reshape_25, full_int_array_37, [1, 2])

        # pd_op.slice: (2x52x38x384xf32) <- (2x56x42x384xf32, 2xi64, 2xi64)
        slice_7 = paddle._C_ops.slice(
            roll_1, [1, 2], full_int_array_0, full_int_array_32, [1, 1], []
        )

        # pd_op.reshape: (2x1976x384xf32) <- (2x52x38x384xf32, 3xi64)
        reshape_26 = paddle._C_ops.reshape(slice_7, full_int_array_33)

        # pd_op.full: (xf64) <- ()
        full_8 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__2 = paddle._C_ops.assign_value_(
            full_8,
            [],
            paddle.float64,
            [float("0.954545")],
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
        add_13 = paddle._C_ops.add(cast_3, uniform_2)
        del uniform_2

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_2 = paddle._C_ops.floor(add_13)
        del add_13

        # pd_op.divide: (2x1976x384xf32) <- (2x1976x384xf32, xf32)
        divide_2 = paddle._C_ops.divide(reshape_26, cast_3)

        # pd_op.multiply: (2x1976x384xf32) <- (2x1976x384xf32, 2x1x1xf32)
        multiply_3 = paddle._C_ops.multiply(divide_2, floor_2)

        # pd_op.add: (2x1976x384xf32) <- (2x1976x384xf32, 2x1976x384xf32)
        add_14 = paddle._C_ops.add(add_8, multiply_3)

        # pd_op.layer_norm: (2x1976x384xf32, 2x1976xf32, 2x1976xf32) <- (2x1976x384xf32, 384xf32, 384xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_14, parameter_56, parameter_55, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_55, parameter_56

        # pd_op.matmul: (2x1976x1536xf32) <- (2x1976x384xf32, 384x1536xf32)
        matmul_11 = paddle._C_ops.matmul(layer_norm_9, parameter_54, False, False)
        del parameter_54

        # pd_op.add: (2x1976x1536xf32) <- (2x1976x1536xf32, 1536xf32)
        add_15 = paddle._C_ops.add(matmul_11, parameter_53)
        del parameter_53

        # pd_op.gelu: (2x1976x1536xf32) <- (2x1976x1536xf32)
        gelu_1 = paddle._C_ops.gelu(add_15, False)

        # pd_op.matmul: (2x1976x384xf32) <- (2x1976x1536xf32, 1536x384xf32)
        matmul_12 = paddle._C_ops.matmul(gelu_1, parameter_52, False, False)
        del parameter_52

        # pd_op.add: (2x1976x384xf32) <- (2x1976x384xf32, 384xf32)
        add_16 = paddle._C_ops.add(matmul_12, parameter_51)
        del parameter_51

        # pd_op.full: (xf64) <- ()
        full_9 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__3 = paddle._C_ops.assign_value_(
            full_9,
            [],
            paddle.float64,
            [float("0.954545")],
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

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_17 = paddle._C_ops.add(cast_4, uniform_3)
        del uniform_3

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_3 = paddle._C_ops.floor(add_17)
        del add_17

        # pd_op.divide: (2x1976x384xf32) <- (2x1976x384xf32, xf32)
        divide_3 = paddle._C_ops.divide(add_16, cast_4)

        # pd_op.multiply: (2x1976x384xf32) <- (2x1976x384xf32, 2x1x1xf32)
        multiply_4 = paddle._C_ops.multiply(divide_3, floor_3)

        # pd_op.add: (2x1976x384xf32) <- (2x1976x384xf32, 2x1976x384xf32)
        add_18 = paddle._C_ops.add(add_14, multiply_4)

        # pd_op.layer_norm: (2x1976x384xf32, 2x1976xf32, 2x1976xf32) <- (2x1976x384xf32, 384xf32, 384xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_18, parameter_50, parameter_49, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_49, parameter_50

        # pd_op.reshape: (2x52x38x384xf32) <- (2x1976x384xf32, 4xi64)
        reshape_27 = paddle._C_ops.reshape(layer_norm_12, full_int_array_20)

        # pd_op.pad: (2x56x42x384xf32) <- (2x52x38x384xf32, 1xf32)
        pad_2 = paddle._C_ops.pad(reshape_27, [0, 0, 0, 4, 0, 4, 0, 0], full_4)

        # pd_op.reshape: (2x8x7x6x7x384xf32) <- (2x56x42x384xf32, 6xi64)
        reshape_28 = paddle._C_ops.reshape(pad_2, full_int_array_21)

        # pd_op.transpose: (2x8x6x7x7x384xf32) <- (2x8x7x6x7x384xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_28, [0, 1, 3, 2, 4, 5])
        del reshape_28

        # pd_op.reshape: (96x7x7x384xf32) <- (2x8x6x7x7x384xf32, 4xi64)
        reshape_29 = paddle._C_ops.reshape(transpose_13, full_int_array_22)

        # pd_op.reshape: (96x49x384xf32) <- (96x7x7x384xf32, 3xi64)
        reshape_30 = paddle._C_ops.reshape(reshape_29, full_int_array_23)

        # pd_op.matmul: (96x49x1152xf32) <- (96x49x384xf32, 384x1152xf32)
        matmul_13 = paddle._C_ops.matmul(reshape_30, parameter_48, False, False)
        del parameter_48

        # pd_op.add: (96x49x1152xf32) <- (96x49x1152xf32, 1152xf32)
        add_19 = paddle._C_ops.add(matmul_13, parameter_47)
        del parameter_47

        # pd_op.reshape: (96x49x3x12x32xf32) <- (96x49x1152xf32, 5xi64)
        reshape_31 = paddle._C_ops.reshape(add_19, full_int_array_24)

        # pd_op.transpose: (3x96x12x49x32xf32) <- (96x49x3x12x32xf32)
        transpose_14 = paddle._C_ops.transpose(reshape_31, [2, 0, 3, 1, 4])
        del reshape_31

        # pd_op.slice: (96x12x49x32xf32) <- (3x96x12x49x32xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            transpose_14, [0], full_int_array_25, full_int_array_18, [1], [0]
        )

        # pd_op.slice: (96x12x49x32xf32) <- (3x96x12x49x32xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            transpose_14, [0], full_int_array_18, full_int_array_19, [1], [0]
        )

        # pd_op.slice: (96x12x49x32xf32) <- (3x96x12x49x32xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            transpose_14, [0], full_int_array_19, full_int_array_26, [1], [0]
        )

        # pd_op.scale: (96x12x49x32xf32) <- (96x12x49x32xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(slice_8, full_5, float("0"), True)
        del slice_8

        # pd_op.transpose: (96x12x32x49xf32) <- (96x12x49x32xf32)
        transpose_15 = paddle._C_ops.transpose(slice_9, [0, 1, 3, 2])
        del slice_9

        # pd_op.matmul: (96x12x49x49xf32) <- (96x12x49x32xf32, 96x12x32x49xf32)
        matmul_14 = paddle._C_ops.matmul(scale_3, transpose_15, False, False)

        # pd_op.flatten: (2401xi64) <- (49x49xi64)
        flatten_2 = paddle._C_ops.flatten(data_9, 0, 1)
        del data_9

        # pd_op.index_select: (2401x12xf32) <- (169x12xf32, 2401xi64)
        index_select_2 = paddle._C_ops.index_select(data_2, flatten_2, 0)
        del data_2

        # pd_op.reshape: (49x49x12xf32) <- (2401x12xf32, 3xi64)
        reshape_32 = paddle._C_ops.reshape(index_select_2, full_int_array_27)

        # pd_op.transpose: (12x49x49xf32) <- (49x49x12xf32)
        transpose_16 = paddle._C_ops.transpose(reshape_32, [2, 0, 1])
        del reshape_32

        # pd_op.unsqueeze: (1x12x49x49xf32) <- (12x49x49xf32, 1xi64)
        unsqueeze_6 = paddle._C_ops.unsqueeze(transpose_16, full_int_array_25)

        # pd_op.add: (96x12x49x49xf32) <- (96x12x49x49xf32, 1x12x49x49xf32)
        add_20 = paddle._C_ops.add(matmul_14, unsqueeze_6)

        # pd_op.softmax: (96x12x49x49xf32) <- (96x12x49x49xf32)
        softmax_2 = paddle._C_ops.softmax(add_20, -1)
        del add_20

        # pd_op.matmul: (96x12x49x32xf32) <- (96x12x49x49xf32, 96x12x49x32xf32)
        matmul_15 = paddle._C_ops.matmul(softmax_2, slice_10, False, False)

        # pd_op.transpose: (96x49x12x32xf32) <- (96x12x49x32xf32)
        transpose_17 = paddle._C_ops.transpose(matmul_15, [0, 2, 1, 3])
        del matmul_15

        # pd_op.reshape: (96x49x384xf32) <- (96x49x12x32xf32, 3xi64)
        reshape_33 = paddle._C_ops.reshape(transpose_17, full_int_array_28)

        # pd_op.matmul: (96x49x384xf32) <- (96x49x384xf32, 384x384xf32)
        matmul_16 = paddle._C_ops.matmul(reshape_33, parameter_46, False, False)
        del parameter_46

        # pd_op.add: (96x49x384xf32) <- (96x49x384xf32, 384xf32)
        add_21 = paddle._C_ops.add(matmul_16, parameter_45)
        del parameter_45

        # pd_op.reshape: (96x7x7x384xf32) <- (96x49x384xf32, 4xi64)
        reshape_34 = paddle._C_ops.reshape(add_21, full_int_array_29)

        # pd_op.reshape: (2x8x6x7x7x384xf32) <- (96x7x7x384xf32, 6xi64)
        reshape_35 = paddle._C_ops.reshape(reshape_34, full_int_array_30)

        # pd_op.transpose: (2x8x7x6x7x384xf32) <- (2x8x6x7x7x384xf32)
        transpose_18 = paddle._C_ops.transpose(reshape_35, [0, 1, 3, 2, 4, 5])
        del reshape_35

        # pd_op.reshape: (2x56x42x384xf32) <- (2x8x7x6x7x384xf32, 4xi64)
        reshape_36 = paddle._C_ops.reshape(transpose_18, full_int_array_31)

        # pd_op.slice: (2x52x38x384xf32) <- (2x56x42x384xf32, 2xi64, 2xi64)
        slice_11 = paddle._C_ops.slice(
            reshape_36, [1, 2], full_int_array_0, full_int_array_32, [1, 1], []
        )

        # pd_op.reshape: (2x1976x384xf32) <- (2x52x38x384xf32, 3xi64)
        reshape_37 = paddle._C_ops.reshape(slice_11, full_int_array_33)

        # pd_op.full: (xf64) <- ()
        full_10 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__4 = paddle._C_ops.assign_value_(
            full_10,
            [],
            paddle.float64,
            [float("0.945455")],
            paddle.framework._current_expected_place(),
        )
        del full_10

        # pd_op.cast: (xf32) <- (xf64)
        cast_5 = paddle._C_ops.cast(assign_value__4, paddle.float32)
        del assign_value__4

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_4 = paddle._C_ops.uniform(
            full_int_array_34,
            paddle.float32,
            full_4,
            full_1,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_22 = paddle._C_ops.add(cast_5, uniform_4)
        del uniform_4

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_4 = paddle._C_ops.floor(add_22)
        del add_22

        # pd_op.divide: (2x1976x384xf32) <- (2x1976x384xf32, xf32)
        divide_4 = paddle._C_ops.divide(reshape_37, cast_5)

        # pd_op.multiply: (2x1976x384xf32) <- (2x1976x384xf32, 2x1x1xf32)
        multiply_5 = paddle._C_ops.multiply(divide_4, floor_4)

        # pd_op.add: (2x1976x384xf32) <- (2x1976x384xf32, 2x1976x384xf32)
        add_23 = paddle._C_ops.add(add_18, multiply_5)

        # pd_op.layer_norm: (2x1976x384xf32, 2x1976xf32, 2x1976xf32) <- (2x1976x384xf32, 384xf32, 384xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_23, parameter_44, parameter_43, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_43, parameter_44

        # pd_op.matmul: (2x1976x1536xf32) <- (2x1976x384xf32, 384x1536xf32)
        matmul_17 = paddle._C_ops.matmul(layer_norm_15, parameter_42, False, False)
        del parameter_42

        # pd_op.add: (2x1976x1536xf32) <- (2x1976x1536xf32, 1536xf32)
        add_24 = paddle._C_ops.add(matmul_17, parameter_41)
        del parameter_41

        # pd_op.gelu: (2x1976x1536xf32) <- (2x1976x1536xf32)
        gelu_2 = paddle._C_ops.gelu(add_24, False)

        # pd_op.matmul: (2x1976x384xf32) <- (2x1976x1536xf32, 1536x384xf32)
        matmul_18 = paddle._C_ops.matmul(gelu_2, parameter_40, False, False)
        del parameter_40

        # pd_op.add: (2x1976x384xf32) <- (2x1976x384xf32, 384xf32)
        add_25 = paddle._C_ops.add(matmul_18, parameter_39)
        del parameter_39

        # pd_op.full: (xf64) <- ()
        full_11 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__5 = paddle._C_ops.assign_value_(
            full_11,
            [],
            paddle.float64,
            [float("0.945455")],
            paddle.framework._current_expected_place(),
        )
        del full_11

        # pd_op.cast: (xf32) <- (xf64)
        cast_6 = paddle._C_ops.cast(assign_value__5, paddle.float32)
        del assign_value__5

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_5 = paddle._C_ops.uniform(
            full_int_array_34,
            paddle.float32,
            full_4,
            full_1,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_26 = paddle._C_ops.add(cast_6, uniform_5)
        del uniform_5

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_5 = paddle._C_ops.floor(add_26)
        del add_26

        # pd_op.divide: (2x1976x384xf32) <- (2x1976x384xf32, xf32)
        divide_5 = paddle._C_ops.divide(add_25, cast_6)

        # pd_op.multiply: (2x1976x384xf32) <- (2x1976x384xf32, 2x1x1xf32)
        multiply_6 = paddle._C_ops.multiply(divide_5, floor_5)

        # pd_op.add: (2x1976x384xf32) <- (2x1976x384xf32, 2x1976x384xf32)
        add_27 = paddle._C_ops.add(add_23, multiply_6)

        # pd_op.layer_norm: (2x1976x384xf32, 2x1976xf32, 2x1976xf32) <- (2x1976x384xf32, 384xf32, 384xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_27, parameter_38, parameter_37, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_37, parameter_38

        # pd_op.reshape: (2x52x38x384xf32) <- (2x1976x384xf32, 4xi64)
        reshape_38 = paddle._C_ops.reshape(layer_norm_18, full_int_array_20)

        # pd_op.pad: (2x56x42x384xf32) <- (2x52x38x384xf32, 1xf32)
        pad_3 = paddle._C_ops.pad(reshape_38, [0, 0, 0, 4, 0, 4, 0, 0], full_4)

        # pd_op.roll: (2x56x42x384xf32) <- (2x56x42x384xf32, 2xi64)
        roll_2 = paddle._C_ops.roll(pad_3, full_int_array_9, [1, 2])

        # pd_op.reshape: (2x8x7x6x7x384xf32) <- (2x56x42x384xf32, 6xi64)
        reshape_39 = paddle._C_ops.reshape(roll_2, full_int_array_21)

        # pd_op.transpose: (2x8x6x7x7x384xf32) <- (2x8x7x6x7x384xf32)
        transpose_19 = paddle._C_ops.transpose(reshape_39, [0, 1, 3, 2, 4, 5])
        del reshape_39

        # pd_op.reshape: (96x7x7x384xf32) <- (2x8x6x7x7x384xf32, 4xi64)
        reshape_40 = paddle._C_ops.reshape(transpose_19, full_int_array_22)

        # pd_op.reshape: (96x49x384xf32) <- (96x7x7x384xf32, 3xi64)
        reshape_41 = paddle._C_ops.reshape(reshape_40, full_int_array_23)

        # pd_op.matmul: (96x49x1152xf32) <- (96x49x384xf32, 384x1152xf32)
        matmul_19 = paddle._C_ops.matmul(reshape_41, parameter_36, False, False)
        del parameter_36

        # pd_op.add: (96x49x1152xf32) <- (96x49x1152xf32, 1152xf32)
        add_28 = paddle._C_ops.add(matmul_19, parameter_35)
        del parameter_35

        # pd_op.reshape: (96x49x3x12x32xf32) <- (96x49x1152xf32, 5xi64)
        reshape_42 = paddle._C_ops.reshape(add_28, full_int_array_24)

        # pd_op.transpose: (3x96x12x49x32xf32) <- (96x49x3x12x32xf32)
        transpose_20 = paddle._C_ops.transpose(reshape_42, [2, 0, 3, 1, 4])
        del reshape_42

        # pd_op.slice: (96x12x49x32xf32) <- (3x96x12x49x32xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            transpose_20, [0], full_int_array_25, full_int_array_18, [1], [0]
        )

        # pd_op.slice: (96x12x49x32xf32) <- (3x96x12x49x32xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            transpose_20, [0], full_int_array_18, full_int_array_19, [1], [0]
        )

        # pd_op.slice: (96x12x49x32xf32) <- (3x96x12x49x32xf32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            transpose_20, [0], full_int_array_19, full_int_array_26, [1], [0]
        )

        # pd_op.scale: (96x12x49x32xf32) <- (96x12x49x32xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(slice_12, full_5, float("0"), True)
        del slice_12

        # pd_op.transpose: (96x12x32x49xf32) <- (96x12x49x32xf32)
        transpose_21 = paddle._C_ops.transpose(slice_13, [0, 1, 3, 2])
        del slice_13

        # pd_op.matmul: (96x12x49x49xf32) <- (96x12x49x32xf32, 96x12x32x49xf32)
        matmul_20 = paddle._C_ops.matmul(scale_4, transpose_21, False, False)

        # pd_op.flatten: (2401xi64) <- (49x49xi64)
        flatten_3 = paddle._C_ops.flatten(data_10, 0, 1)
        del data_10

        # pd_op.index_select: (2401x12xf32) <- (169x12xf32, 2401xi64)
        index_select_3 = paddle._C_ops.index_select(data_3, flatten_3, 0)
        del data_3

        # pd_op.reshape: (49x49x12xf32) <- (2401x12xf32, 3xi64)
        reshape_43 = paddle._C_ops.reshape(index_select_3, full_int_array_27)

        # pd_op.transpose: (12x49x49xf32) <- (49x49x12xf32)
        transpose_22 = paddle._C_ops.transpose(reshape_43, [2, 0, 1])
        del reshape_43

        # pd_op.unsqueeze: (1x12x49x49xf32) <- (12x49x49xf32, 1xi64)
        unsqueeze_7 = paddle._C_ops.unsqueeze(transpose_22, full_int_array_25)

        # pd_op.add: (96x12x49x49xf32) <- (96x12x49x49xf32, 1x12x49x49xf32)
        add_29 = paddle._C_ops.add(matmul_20, unsqueeze_7)

        # pd_op.reshape: (2x48x12x49x49xf32) <- (96x12x49x49xf32, 5xi64)
        reshape_44 = paddle._C_ops.reshape(add_29, full_int_array_35)

        # pd_op.unsqueeze: (48x1x49x49xf32) <- (48x49x49xf32, 1xi64)
        unsqueeze_8 = paddle._C_ops.unsqueeze(multiply_0, full_int_array_18)

        # pd_op.unsqueeze: (1x48x1x49x49xf32) <- (48x1x49x49xf32, 1xi64)
        unsqueeze_9 = paddle._C_ops.unsqueeze(unsqueeze_8, full_int_array_25)
        del unsqueeze_8

        # pd_op.add: (2x48x12x49x49xf32) <- (2x48x12x49x49xf32, 1x48x1x49x49xf32)
        add_30 = paddle._C_ops.add(reshape_44, unsqueeze_9)

        # pd_op.reshape: (96x12x49x49xf32) <- (2x48x12x49x49xf32, 4xi64)
        reshape_45 = paddle._C_ops.reshape(add_30, full_int_array_36)

        # pd_op.softmax: (96x12x49x49xf32) <- (96x12x49x49xf32)
        softmax_3 = paddle._C_ops.softmax(reshape_45, -1)
        del reshape_45

        # pd_op.matmul: (96x12x49x32xf32) <- (96x12x49x49xf32, 96x12x49x32xf32)
        matmul_21 = paddle._C_ops.matmul(softmax_3, slice_14, False, False)

        # pd_op.transpose: (96x49x12x32xf32) <- (96x12x49x32xf32)
        transpose_23 = paddle._C_ops.transpose(matmul_21, [0, 2, 1, 3])
        del matmul_21

        # pd_op.reshape: (96x49x384xf32) <- (96x49x12x32xf32, 3xi64)
        reshape_46 = paddle._C_ops.reshape(transpose_23, full_int_array_28)

        # pd_op.matmul: (96x49x384xf32) <- (96x49x384xf32, 384x384xf32)
        matmul_22 = paddle._C_ops.matmul(reshape_46, parameter_34, False, False)
        del parameter_34

        # pd_op.add: (96x49x384xf32) <- (96x49x384xf32, 384xf32)
        add_31 = paddle._C_ops.add(matmul_22, parameter_33)
        del parameter_33

        # pd_op.reshape: (96x7x7x384xf32) <- (96x49x384xf32, 4xi64)
        reshape_47 = paddle._C_ops.reshape(add_31, full_int_array_29)

        # pd_op.reshape: (2x8x6x7x7x384xf32) <- (96x7x7x384xf32, 6xi64)
        reshape_48 = paddle._C_ops.reshape(reshape_47, full_int_array_30)

        # pd_op.transpose: (2x8x7x6x7x384xf32) <- (2x8x6x7x7x384xf32)
        transpose_24 = paddle._C_ops.transpose(reshape_48, [0, 1, 3, 2, 4, 5])
        del reshape_48

        # pd_op.reshape: (2x56x42x384xf32) <- (2x8x7x6x7x384xf32, 4xi64)
        reshape_49 = paddle._C_ops.reshape(transpose_24, full_int_array_31)

        # pd_op.roll: (2x56x42x384xf32) <- (2x56x42x384xf32, 2xi64)
        roll_3 = paddle._C_ops.roll(reshape_49, full_int_array_37, [1, 2])

        # pd_op.slice: (2x52x38x384xf32) <- (2x56x42x384xf32, 2xi64, 2xi64)
        slice_15 = paddle._C_ops.slice(
            roll_3, [1, 2], full_int_array_0, full_int_array_32, [1, 1], []
        )

        # pd_op.reshape: (2x1976x384xf32) <- (2x52x38x384xf32, 3xi64)
        reshape_50 = paddle._C_ops.reshape(slice_15, full_int_array_33)

        # pd_op.full: (xf64) <- ()
        full_12 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__6 = paddle._C_ops.assign_value_(
            full_12,
            [],
            paddle.float64,
            [float("0.936364")],
            paddle.framework._current_expected_place(),
        )
        del full_12

        # pd_op.cast: (xf32) <- (xf64)
        cast_7 = paddle._C_ops.cast(assign_value__6, paddle.float32)
        del assign_value__6

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_6 = paddle._C_ops.uniform(
            full_int_array_34,
            paddle.float32,
            full_4,
            full_1,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_32 = paddle._C_ops.add(cast_7, uniform_6)
        del uniform_6

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_6 = paddle._C_ops.floor(add_32)
        del add_32

        # pd_op.divide: (2x1976x384xf32) <- (2x1976x384xf32, xf32)
        divide_6 = paddle._C_ops.divide(reshape_50, cast_7)

        # pd_op.multiply: (2x1976x384xf32) <- (2x1976x384xf32, 2x1x1xf32)
        multiply_7 = paddle._C_ops.multiply(divide_6, floor_6)

        # pd_op.add: (2x1976x384xf32) <- (2x1976x384xf32, 2x1976x384xf32)
        add_33 = paddle._C_ops.add(add_27, multiply_7)

        # pd_op.layer_norm: (2x1976x384xf32, 2x1976xf32, 2x1976xf32) <- (2x1976x384xf32, 384xf32, 384xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_33, parameter_32, parameter_31, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_31, parameter_32

        # pd_op.matmul: (2x1976x1536xf32) <- (2x1976x384xf32, 384x1536xf32)
        matmul_23 = paddle._C_ops.matmul(layer_norm_21, parameter_30, False, False)
        del parameter_30

        # pd_op.add: (2x1976x1536xf32) <- (2x1976x1536xf32, 1536xf32)
        add_34 = paddle._C_ops.add(matmul_23, parameter_29)
        del parameter_29

        # pd_op.gelu: (2x1976x1536xf32) <- (2x1976x1536xf32)
        gelu_3 = paddle._C_ops.gelu(add_34, False)

        # pd_op.matmul: (2x1976x384xf32) <- (2x1976x1536xf32, 1536x384xf32)
        matmul_24 = paddle._C_ops.matmul(gelu_3, parameter_28, False, False)
        del parameter_28

        # pd_op.add: (2x1976x384xf32) <- (2x1976x384xf32, 384xf32)
        add_35 = paddle._C_ops.add(matmul_24, parameter_27)
        del parameter_27

        # pd_op.full: (xf64) <- ()
        full_13 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__7 = paddle._C_ops.assign_value_(
            full_13,
            [],
            paddle.float64,
            [float("0.936364")],
            paddle.framework._current_expected_place(),
        )
        del full_13

        # pd_op.cast: (xf32) <- (xf64)
        cast_8 = paddle._C_ops.cast(assign_value__7, paddle.float32)
        del assign_value__7

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_7 = paddle._C_ops.uniform(
            full_int_array_34,
            paddle.float32,
            full_4,
            full_1,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_36 = paddle._C_ops.add(cast_8, uniform_7)
        del uniform_7

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_7 = paddle._C_ops.floor(add_36)
        del add_36

        # pd_op.divide: (2x1976x384xf32) <- (2x1976x384xf32, xf32)
        divide_7 = paddle._C_ops.divide(add_35, cast_8)

        # pd_op.multiply: (2x1976x384xf32) <- (2x1976x384xf32, 2x1x1xf32)
        multiply_8 = paddle._C_ops.multiply(divide_7, floor_7)

        # pd_op.add: (2x1976x384xf32) <- (2x1976x384xf32, 2x1976x384xf32)
        add_37 = paddle._C_ops.add(add_33, multiply_8)

        # pd_op.layer_norm: (2x1976x384xf32, 2x1976xf32, 2x1976xf32) <- (2x1976x384xf32, 384xf32, 384xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_37, parameter_26, parameter_25, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_25, parameter_26

        # pd_op.reshape: (2x52x38x384xf32) <- (2x1976x384xf32, 4xi64)
        reshape_51 = paddle._C_ops.reshape(layer_norm_24, full_int_array_20)

        # pd_op.pad: (2x56x42x384xf32) <- (2x52x38x384xf32, 1xf32)
        pad_4 = paddle._C_ops.pad(reshape_51, [0, 0, 0, 4, 0, 4, 0, 0], full_4)

        # pd_op.reshape: (2x8x7x6x7x384xf32) <- (2x56x42x384xf32, 6xi64)
        reshape_52 = paddle._C_ops.reshape(pad_4, full_int_array_21)

        # pd_op.transpose: (2x8x6x7x7x384xf32) <- (2x8x7x6x7x384xf32)
        transpose_25 = paddle._C_ops.transpose(reshape_52, [0, 1, 3, 2, 4, 5])
        del reshape_52

        # pd_op.reshape: (96x7x7x384xf32) <- (2x8x6x7x7x384xf32, 4xi64)
        reshape_53 = paddle._C_ops.reshape(transpose_25, full_int_array_22)

        # pd_op.reshape: (96x49x384xf32) <- (96x7x7x384xf32, 3xi64)
        reshape_54 = paddle._C_ops.reshape(reshape_53, full_int_array_23)

        # pd_op.matmul: (96x49x1152xf32) <- (96x49x384xf32, 384x1152xf32)
        matmul_25 = paddle._C_ops.matmul(reshape_54, parameter_24, False, False)
        del parameter_24

        # pd_op.add: (96x49x1152xf32) <- (96x49x1152xf32, 1152xf32)
        add_38 = paddle._C_ops.add(matmul_25, parameter_23)
        del parameter_23

        # pd_op.reshape: (96x49x3x12x32xf32) <- (96x49x1152xf32, 5xi64)
        reshape_55 = paddle._C_ops.reshape(add_38, full_int_array_24)

        # pd_op.transpose: (3x96x12x49x32xf32) <- (96x49x3x12x32xf32)
        transpose_26 = paddle._C_ops.transpose(reshape_55, [2, 0, 3, 1, 4])
        del reshape_55

        # pd_op.slice: (96x12x49x32xf32) <- (3x96x12x49x32xf32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            transpose_26, [0], full_int_array_25, full_int_array_18, [1], [0]
        )

        # pd_op.slice: (96x12x49x32xf32) <- (3x96x12x49x32xf32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            transpose_26, [0], full_int_array_18, full_int_array_19, [1], [0]
        )

        # pd_op.slice: (96x12x49x32xf32) <- (3x96x12x49x32xf32, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(
            transpose_26, [0], full_int_array_19, full_int_array_26, [1], [0]
        )

        # pd_op.scale: (96x12x49x32xf32) <- (96x12x49x32xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(slice_16, full_5, float("0"), True)
        del slice_16

        # pd_op.transpose: (96x12x32x49xf32) <- (96x12x49x32xf32)
        transpose_27 = paddle._C_ops.transpose(slice_17, [0, 1, 3, 2])
        del slice_17

        # pd_op.matmul: (96x12x49x49xf32) <- (96x12x49x32xf32, 96x12x32x49xf32)
        matmul_26 = paddle._C_ops.matmul(scale_5, transpose_27, False, False)

        # pd_op.flatten: (2401xi64) <- (49x49xi64)
        flatten_4 = paddle._C_ops.flatten(data_11, 0, 1)
        del data_11

        # pd_op.index_select: (2401x12xf32) <- (169x12xf32, 2401xi64)
        index_select_4 = paddle._C_ops.index_select(data_4, flatten_4, 0)
        del data_4

        # pd_op.reshape: (49x49x12xf32) <- (2401x12xf32, 3xi64)
        reshape_56 = paddle._C_ops.reshape(index_select_4, full_int_array_27)

        # pd_op.transpose: (12x49x49xf32) <- (49x49x12xf32)
        transpose_28 = paddle._C_ops.transpose(reshape_56, [2, 0, 1])
        del reshape_56

        # pd_op.unsqueeze: (1x12x49x49xf32) <- (12x49x49xf32, 1xi64)
        unsqueeze_10 = paddle._C_ops.unsqueeze(transpose_28, full_int_array_25)

        # pd_op.add: (96x12x49x49xf32) <- (96x12x49x49xf32, 1x12x49x49xf32)
        add_39 = paddle._C_ops.add(matmul_26, unsqueeze_10)

        # pd_op.softmax: (96x12x49x49xf32) <- (96x12x49x49xf32)
        softmax_4 = paddle._C_ops.softmax(add_39, -1)
        del add_39

        # pd_op.matmul: (96x12x49x32xf32) <- (96x12x49x49xf32, 96x12x49x32xf32)
        matmul_27 = paddle._C_ops.matmul(softmax_4, slice_18, False, False)

        # pd_op.transpose: (96x49x12x32xf32) <- (96x12x49x32xf32)
        transpose_29 = paddle._C_ops.transpose(matmul_27, [0, 2, 1, 3])
        del matmul_27

        # pd_op.reshape: (96x49x384xf32) <- (96x49x12x32xf32, 3xi64)
        reshape_57 = paddle._C_ops.reshape(transpose_29, full_int_array_28)

        # pd_op.matmul: (96x49x384xf32) <- (96x49x384xf32, 384x384xf32)
        matmul_28 = paddle._C_ops.matmul(reshape_57, parameter_22, False, False)
        del parameter_22

        # pd_op.add: (96x49x384xf32) <- (96x49x384xf32, 384xf32)
        add_40 = paddle._C_ops.add(matmul_28, parameter_21)
        del parameter_21

        # pd_op.reshape: (96x7x7x384xf32) <- (96x49x384xf32, 4xi64)
        reshape_58 = paddle._C_ops.reshape(add_40, full_int_array_29)

        # pd_op.reshape: (2x8x6x7x7x384xf32) <- (96x7x7x384xf32, 6xi64)
        reshape_59 = paddle._C_ops.reshape(reshape_58, full_int_array_30)

        # pd_op.transpose: (2x8x7x6x7x384xf32) <- (2x8x6x7x7x384xf32)
        transpose_30 = paddle._C_ops.transpose(reshape_59, [0, 1, 3, 2, 4, 5])
        del reshape_59

        # pd_op.reshape: (2x56x42x384xf32) <- (2x8x7x6x7x384xf32, 4xi64)
        reshape_60 = paddle._C_ops.reshape(transpose_30, full_int_array_31)

        # pd_op.slice: (2x52x38x384xf32) <- (2x56x42x384xf32, 2xi64, 2xi64)
        slice_19 = paddle._C_ops.slice(
            reshape_60, [1, 2], full_int_array_0, full_int_array_32, [1, 1], []
        )

        # pd_op.reshape: (2x1976x384xf32) <- (2x52x38x384xf32, 3xi64)
        reshape_61 = paddle._C_ops.reshape(slice_19, full_int_array_33)

        # pd_op.full: (xf64) <- ()
        full_14 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__8 = paddle._C_ops.assign_value_(
            full_14,
            [],
            paddle.float64,
            [float("0.927273")],
            paddle.framework._current_expected_place(),
        )
        del full_14

        # pd_op.cast: (xf32) <- (xf64)
        cast_9 = paddle._C_ops.cast(assign_value__8, paddle.float32)
        del assign_value__8

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_8 = paddle._C_ops.uniform(
            full_int_array_34,
            paddle.float32,
            full_4,
            full_1,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_41 = paddle._C_ops.add(cast_9, uniform_8)
        del uniform_8

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_8 = paddle._C_ops.floor(add_41)
        del add_41

        # pd_op.divide: (2x1976x384xf32) <- (2x1976x384xf32, xf32)
        divide_8 = paddle._C_ops.divide(reshape_61, cast_9)

        # pd_op.multiply: (2x1976x384xf32) <- (2x1976x384xf32, 2x1x1xf32)
        multiply_9 = paddle._C_ops.multiply(divide_8, floor_8)

        # pd_op.add: (2x1976x384xf32) <- (2x1976x384xf32, 2x1976x384xf32)
        add_42 = paddle._C_ops.add(add_37, multiply_9)

        # pd_op.layer_norm: (2x1976x384xf32, 2x1976xf32, 2x1976xf32) <- (2x1976x384xf32, 384xf32, 384xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_42, parameter_20, parameter_19, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_19, parameter_20

        # pd_op.matmul: (2x1976x1536xf32) <- (2x1976x384xf32, 384x1536xf32)
        matmul_29 = paddle._C_ops.matmul(layer_norm_27, parameter_18, False, False)
        del parameter_18

        # pd_op.add: (2x1976x1536xf32) <- (2x1976x1536xf32, 1536xf32)
        add_43 = paddle._C_ops.add(matmul_29, parameter_17)
        del parameter_17

        # pd_op.gelu: (2x1976x1536xf32) <- (2x1976x1536xf32)
        gelu_4 = paddle._C_ops.gelu(add_43, False)

        # pd_op.matmul: (2x1976x384xf32) <- (2x1976x1536xf32, 1536x384xf32)
        matmul_30 = paddle._C_ops.matmul(gelu_4, parameter_16, False, False)
        del parameter_16

        # pd_op.add: (2x1976x384xf32) <- (2x1976x384xf32, 384xf32)
        add_44 = paddle._C_ops.add(matmul_30, parameter_15)
        del parameter_15

        # pd_op.full: (xf64) <- ()
        full_15 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__9 = paddle._C_ops.assign_value_(
            full_15,
            [],
            paddle.float64,
            [float("0.927273")],
            paddle.framework._current_expected_place(),
        )
        del full_15

        # pd_op.cast: (xf32) <- (xf64)
        cast_10 = paddle._C_ops.cast(assign_value__9, paddle.float32)
        del assign_value__9

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_9 = paddle._C_ops.uniform(
            full_int_array_34,
            paddle.float32,
            full_4,
            full_1,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_45 = paddle._C_ops.add(cast_10, uniform_9)
        del uniform_9

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_9 = paddle._C_ops.floor(add_45)
        del add_45

        # pd_op.divide: (2x1976x384xf32) <- (2x1976x384xf32, xf32)
        divide_9 = paddle._C_ops.divide(add_44, cast_10)

        # pd_op.multiply: (2x1976x384xf32) <- (2x1976x384xf32, 2x1x1xf32)
        multiply_10 = paddle._C_ops.multiply(divide_9, floor_9)

        # pd_op.add: (2x1976x384xf32) <- (2x1976x384xf32, 2x1976x384xf32)
        add_46 = paddle._C_ops.add(add_42, multiply_10)

        # pd_op.layer_norm: (2x1976x384xf32, 2x1976xf32, 2x1976xf32) <- (2x1976x384xf32, 384xf32, 384xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_46, parameter_14, parameter_13, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_13, parameter_14

        # pd_op.reshape: (2x52x38x384xf32) <- (2x1976x384xf32, 4xi64)
        reshape_62 = paddle._C_ops.reshape(layer_norm_30, full_int_array_20)

        # pd_op.pad: (2x56x42x384xf32) <- (2x52x38x384xf32, 1xf32)
        pad_5 = paddle._C_ops.pad(reshape_62, [0, 0, 0, 4, 0, 4, 0, 0], full_4)

        # pd_op.roll: (2x56x42x384xf32) <- (2x56x42x384xf32, 2xi64)
        roll_4 = paddle._C_ops.roll(pad_5, full_int_array_9, [1, 2])
        del full_int_array_9

        # pd_op.reshape: (2x8x7x6x7x384xf32) <- (2x56x42x384xf32, 6xi64)
        reshape_63 = paddle._C_ops.reshape(roll_4, full_int_array_21)
        del full_int_array_21

        # pd_op.transpose: (2x8x6x7x7x384xf32) <- (2x8x7x6x7x384xf32)
        transpose_31 = paddle._C_ops.transpose(reshape_63, [0, 1, 3, 2, 4, 5])
        del reshape_63

        # pd_op.reshape: (96x7x7x384xf32) <- (2x8x6x7x7x384xf32, 4xi64)
        reshape_64 = paddle._C_ops.reshape(transpose_31, full_int_array_22)
        del full_int_array_22

        # pd_op.reshape: (96x49x384xf32) <- (96x7x7x384xf32, 3xi64)
        reshape_65 = paddle._C_ops.reshape(reshape_64, full_int_array_23)
        del full_int_array_23

        # pd_op.matmul: (96x49x1152xf32) <- (96x49x384xf32, 384x1152xf32)
        matmul_31 = paddle._C_ops.matmul(reshape_65, parameter_12, False, False)
        del parameter_12

        # pd_op.add: (96x49x1152xf32) <- (96x49x1152xf32, 1152xf32)
        add_47 = paddle._C_ops.add(matmul_31, parameter_11)
        del parameter_11

        # pd_op.reshape: (96x49x3x12x32xf32) <- (96x49x1152xf32, 5xi64)
        reshape_66 = paddle._C_ops.reshape(add_47, full_int_array_24)
        del full_int_array_24

        # pd_op.transpose: (3x96x12x49x32xf32) <- (96x49x3x12x32xf32)
        transpose_32 = paddle._C_ops.transpose(reshape_66, [2, 0, 3, 1, 4])
        del reshape_66

        # pd_op.slice: (96x12x49x32xf32) <- (3x96x12x49x32xf32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(
            transpose_32, [0], full_int_array_25, full_int_array_18, [1], [0]
        )

        # pd_op.slice: (96x12x49x32xf32) <- (3x96x12x49x32xf32, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(
            transpose_32, [0], full_int_array_18, full_int_array_19, [1], [0]
        )

        # pd_op.slice: (96x12x49x32xf32) <- (3x96x12x49x32xf32, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(
            transpose_32, [0], full_int_array_19, full_int_array_26, [1], [0]
        )
        del full_int_array_19

        # pd_op.scale: (96x12x49x32xf32) <- (96x12x49x32xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(slice_20, full_5, float("0"), True)
        del slice_20

        # pd_op.transpose: (96x12x32x49xf32) <- (96x12x49x32xf32)
        transpose_33 = paddle._C_ops.transpose(slice_21, [0, 1, 3, 2])
        del slice_21

        # pd_op.matmul: (96x12x49x49xf32) <- (96x12x49x32xf32, 96x12x32x49xf32)
        matmul_32 = paddle._C_ops.matmul(scale_6, transpose_33, False, False)

        # pd_op.flatten: (2401xi64) <- (49x49xi64)
        flatten_5 = paddle._C_ops.flatten(data_12, 0, 1)
        del data_12

        # pd_op.index_select: (2401x12xf32) <- (169x12xf32, 2401xi64)
        index_select_5 = paddle._C_ops.index_select(data_5, flatten_5, 0)
        del data_5

        # pd_op.reshape: (49x49x12xf32) <- (2401x12xf32, 3xi64)
        reshape_67 = paddle._C_ops.reshape(index_select_5, full_int_array_27)
        del full_int_array_27

        # pd_op.transpose: (12x49x49xf32) <- (49x49x12xf32)
        transpose_34 = paddle._C_ops.transpose(reshape_67, [2, 0, 1])
        del reshape_67

        # pd_op.unsqueeze: (1x12x49x49xf32) <- (12x49x49xf32, 1xi64)
        unsqueeze_11 = paddle._C_ops.unsqueeze(transpose_34, full_int_array_25)

        # pd_op.add: (96x12x49x49xf32) <- (96x12x49x49xf32, 1x12x49x49xf32)
        add_48 = paddle._C_ops.add(matmul_32, unsqueeze_11)

        # pd_op.reshape: (2x48x12x49x49xf32) <- (96x12x49x49xf32, 5xi64)
        reshape_68 = paddle._C_ops.reshape(add_48, full_int_array_35)
        del full_int_array_35

        # pd_op.unsqueeze: (48x1x49x49xf32) <- (48x49x49xf32, 1xi64)
        unsqueeze_12 = paddle._C_ops.unsqueeze(multiply_0, full_int_array_18)
        del full_int_array_18, multiply_0

        # pd_op.unsqueeze: (1x48x1x49x49xf32) <- (48x1x49x49xf32, 1xi64)
        unsqueeze_13 = paddle._C_ops.unsqueeze(unsqueeze_12, full_int_array_25)
        del unsqueeze_12

        # pd_op.add: (2x48x12x49x49xf32) <- (2x48x12x49x49xf32, 1x48x1x49x49xf32)
        add_49 = paddle._C_ops.add(reshape_68, unsqueeze_13)

        # pd_op.reshape: (96x12x49x49xf32) <- (2x48x12x49x49xf32, 4xi64)
        reshape_69 = paddle._C_ops.reshape(add_49, full_int_array_36)
        del full_int_array_36

        # pd_op.softmax: (96x12x49x49xf32) <- (96x12x49x49xf32)
        softmax_5 = paddle._C_ops.softmax(reshape_69, -1)
        del reshape_69

        # pd_op.matmul: (96x12x49x32xf32) <- (96x12x49x49xf32, 96x12x49x32xf32)
        matmul_33 = paddle._C_ops.matmul(softmax_5, slice_22, False, False)

        # pd_op.transpose: (96x49x12x32xf32) <- (96x12x49x32xf32)
        transpose_35 = paddle._C_ops.transpose(matmul_33, [0, 2, 1, 3])
        del matmul_33

        # pd_op.reshape: (96x49x384xf32) <- (96x49x12x32xf32, 3xi64)
        reshape_70 = paddle._C_ops.reshape(transpose_35, full_int_array_28)
        del full_int_array_28

        # pd_op.matmul: (96x49x384xf32) <- (96x49x384xf32, 384x384xf32)
        matmul_34 = paddle._C_ops.matmul(reshape_70, parameter_10, False, False)
        del parameter_10

        # pd_op.add: (96x49x384xf32) <- (96x49x384xf32, 384xf32)
        add_50 = paddle._C_ops.add(matmul_34, parameter_9)
        del parameter_9

        # pd_op.reshape: (96x7x7x384xf32) <- (96x49x384xf32, 4xi64)
        reshape_71 = paddle._C_ops.reshape(add_50, full_int_array_29)
        del full_int_array_29

        # pd_op.reshape: (2x8x6x7x7x384xf32) <- (96x7x7x384xf32, 6xi64)
        reshape_72 = paddle._C_ops.reshape(reshape_71, full_int_array_30)
        del full_int_array_30

        # pd_op.transpose: (2x8x7x6x7x384xf32) <- (2x8x6x7x7x384xf32)
        transpose_36 = paddle._C_ops.transpose(reshape_72, [0, 1, 3, 2, 4, 5])
        del reshape_72

        # pd_op.reshape: (2x56x42x384xf32) <- (2x8x7x6x7x384xf32, 4xi64)
        reshape_73 = paddle._C_ops.reshape(transpose_36, full_int_array_31)
        del full_int_array_31

        # pd_op.roll: (2x56x42x384xf32) <- (2x56x42x384xf32, 2xi64)
        roll_5 = paddle._C_ops.roll(reshape_73, full_int_array_37, [1, 2])

        # pd_op.slice: (2x52x38x384xf32) <- (2x56x42x384xf32, 2xi64, 2xi64)
        slice_23 = paddle._C_ops.slice(
            roll_5, [1, 2], full_int_array_0, full_int_array_32, [1, 1], []
        )

        # pd_op.reshape: (2x1976x384xf32) <- (2x52x38x384xf32, 3xi64)
        reshape_74 = paddle._C_ops.reshape(slice_23, full_int_array_33)
        del full_int_array_33

        # pd_op.full: (xf64) <- ()
        full_16 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__10 = paddle._C_ops.assign_value_(
            full_16,
            [],
            paddle.float64,
            [float("0.918182")],
            paddle.framework._current_expected_place(),
        )
        del full_16

        # pd_op.cast: (xf32) <- (xf64)
        cast_11 = paddle._C_ops.cast(assign_value__10, paddle.float32)
        del assign_value__10

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_10 = paddle._C_ops.uniform(
            full_int_array_34,
            paddle.float32,
            full_4,
            full_1,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_51 = paddle._C_ops.add(cast_11, uniform_10)
        del uniform_10

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_10 = paddle._C_ops.floor(add_51)
        del add_51

        # pd_op.divide: (2x1976x384xf32) <- (2x1976x384xf32, xf32)
        divide_10 = paddle._C_ops.divide(reshape_74, cast_11)

        # pd_op.multiply: (2x1976x384xf32) <- (2x1976x384xf32, 2x1x1xf32)
        multiply_11 = paddle._C_ops.multiply(divide_10, floor_10)

        # pd_op.add: (2x1976x384xf32) <- (2x1976x384xf32, 2x1976x384xf32)
        add_52 = paddle._C_ops.add(add_46, multiply_11)

        # pd_op.layer_norm: (2x1976x384xf32, 2x1976xf32, 2x1976xf32) <- (2x1976x384xf32, 384xf32, 384xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_52, parameter_8, parameter_7, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_7, parameter_8

        # pd_op.matmul: (2x1976x1536xf32) <- (2x1976x384xf32, 384x1536xf32)
        matmul_35 = paddle._C_ops.matmul(layer_norm_33, parameter_6, False, False)
        del parameter_6

        # pd_op.add: (2x1976x1536xf32) <- (2x1976x1536xf32, 1536xf32)
        add_53 = paddle._C_ops.add(matmul_35, parameter_5)
        del parameter_5

        # pd_op.gelu: (2x1976x1536xf32) <- (2x1976x1536xf32)
        gelu_5 = paddle._C_ops.gelu(add_53, False)

        # pd_op.matmul: (2x1976x384xf32) <- (2x1976x1536xf32, 1536x384xf32)
        matmul_36 = paddle._C_ops.matmul(gelu_5, parameter_4, False, False)
        del parameter_4

        # pd_op.add: (2x1976x384xf32) <- (2x1976x384xf32, 384xf32)
        add_54 = paddle._C_ops.add(matmul_36, parameter_3)
        del parameter_3

        # pd_op.full: (xf64) <- ()
        full_17 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__11 = paddle._C_ops.assign_value_(
            full_17,
            [],
            paddle.float64,
            [float("0.918182")],
            paddle.framework._current_expected_place(),
        )
        del full_17

        # pd_op.cast: (xf32) <- (xf64)
        cast_12 = paddle._C_ops.cast(assign_value__11, paddle.float32)
        del assign_value__11

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_11 = paddle._C_ops.uniform(
            full_int_array_34,
            paddle.float32,
            full_4,
            full_1,
            0,
            paddle.framework._current_expected_place(),
        )
        del full_1, full_int_array_34

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_55 = paddle._C_ops.add(cast_12, uniform_11)
        del uniform_11

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_11 = paddle._C_ops.floor(add_55)
        del add_55

        # pd_op.divide: (2x1976x384xf32) <- (2x1976x384xf32, xf32)
        divide_11 = paddle._C_ops.divide(add_54, cast_12)

        # pd_op.multiply: (2x1976x384xf32) <- (2x1976x384xf32, 2x1x1xf32)
        multiply_12 = paddle._C_ops.multiply(divide_11, floor_11)

        # pd_op.add: (2x1976x384xf32) <- (2x1976x384xf32, 2x1976x384xf32)
        add_56 = paddle._C_ops.add(add_52, multiply_12)

        # pd_op.reshape: (2x52x38x384xf32) <- (2x1976x384xf32, 4xi64)
        reshape_75 = paddle._C_ops.reshape(add_56, full_int_array_20)
        del full_int_array_20

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_38 = [2, 2]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_72 = full_int_array_38

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_73 = full_int_array_38

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_74 = full_int_array_38

        # pd_op.strided_slice: (2x26x19x384xf32) <- (2x52x38x384xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_0 = paddle._C_ops.strided_slice(
            reshape_75, [1, 2], full_int_array_0, full_int_array_14, full_int_array_38
        )
        del full_int_array_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_39 = [1, 0]

        # pd_op.strided_slice: (2x26x19x384xf32) <- (2x52x38x384xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_1 = paddle._C_ops.strided_slice(
            reshape_75, [1, 2], full_int_array_39, full_int_array_14, full_int_array_38
        )

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_40 = [0, 1]

        # pd_op.strided_slice: (2x26x19x384xf32) <- (2x52x38x384xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_2 = paddle._C_ops.strided_slice(
            reshape_75, [1, 2], full_int_array_40, full_int_array_14, full_int_array_38
        )

        # pd_op.strided_slice: (2x26x19x384xf32) <- (2x52x38x384xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_3 = paddle._C_ops.strided_slice(
            reshape_75, [1, 2], full_int_array_2, full_int_array_14, full_int_array_38
        )
        del full_int_array_14, full_int_array_2

        # pd_op.full: (1xi32) <- ()
        full_18 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([2x26x19x384xf32, 2x26x19x384xf32, 2x26x19x384xf32, 2x26x19x384xf32]) <- (2x26x19x384xf32, 2x26x19x384xf32, 2x26x19x384xf32, 2x26x19x384xf32)
        combine_0 = [strided_slice_0, strided_slice_1, strided_slice_2, strided_slice_3]

        # pd_op.concat: (2x26x19x1536xf32) <- ([2x26x19x384xf32, 2x26x19x384xf32, 2x26x19x384xf32, 2x26x19x384xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_18)
        del combine_0

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_41 = [-1, 494, 1536]

        # pd_op.reshape: (2x494x1536xf32) <- (2x26x19x1536xf32, 3xi64)
        reshape_76 = paddle._C_ops.reshape(concat_0, full_int_array_41)
        del full_int_array_41

        # pd_op.layer_norm: (2x494x1536xf32, 2x494xf32, 2x494xf32) <- (2x494x1536xf32, 1536xf32, 1536xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                reshape_76, parameter_2, parameter_1, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_1, parameter_2

        # pd_op.matmul: (2x494x768xf32) <- (2x494x1536xf32, 1536x768xf32)
        matmul_0 = paddle._C_ops.matmul(layer_norm_36, parameter_0, False, False)
        del (
            add_0,
            add_10,
            add_11,
            add_12,
            add_14,
            add_15,
            add_16,
            add_18,
            add_19,
            add_2,
            add_21,
            add_23,
            add_24,
            add_25,
            add_27,
            add_28,
            add_29,
            add_30,
            add_31,
            add_33,
            add_34,
            add_35,
            add_37,
            add_38,
            add_4,
            add_40,
            add_42,
            add_43,
            add_44,
            add_46,
            add_47,
            add_48,
            add_49,
            add_5,
            add_50,
            add_52,
            add_53,
            add_54,
            add_56,
            add_6,
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
            assign_37,
            assign_38,
            assign_39,
            assign_4,
            assign_40,
            assign_41,
            assign_42,
            assign_43,
            assign_44,
            assign_45,
            assign_46,
            assign_47,
            assign_48,
            assign_49,
            assign_5,
            assign_50,
            assign_51,
            assign_52,
            assign_53,
            assign_54,
            assign_55,
            assign_56,
            assign_57,
            assign_58,
            assign_59,
            assign_6,
            assign_60,
            assign_61,
            assign_62,
            assign_63,
            assign_64,
            assign_65,
            assign_66,
            assign_67,
            assign_68,
            assign_69,
            assign_7,
            assign_70,
            assign_71,
            assign_72,
            assign_73,
            assign_74,
            assign_8,
            assign_9,
            cast_1,
            cast_10,
            cast_11,
            cast_12,
            cast_2,
            cast_3,
            cast_4,
            cast_5,
            cast_6,
            cast_7,
            cast_8,
            cast_9,
            concat_0,
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
            flatten_0,
            flatten_1,
            flatten_2,
            flatten_3,
            flatten_4,
            flatten_5,
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
            full_18,
            full_4,
            full_5,
            full_int_array_25,
            full_int_array_26,
            full_int_array_32,
            full_int_array_37,
            full_int_array_38,
            full_int_array_39,
            full_int_array_40,
            gelu_0,
            gelu_1,
            gelu_2,
            gelu_3,
            gelu_4,
            gelu_5,
            index_select_0,
            index_select_1,
            index_select_2,
            index_select_3,
            index_select_4,
            index_select_5,
            layer_norm_0,
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
            layer_norm_34,
            layer_norm_35,
            layer_norm_36,
            layer_norm_37,
            layer_norm_38,
            layer_norm_4,
            layer_norm_5,
            layer_norm_6,
            layer_norm_7,
            layer_norm_8,
            layer_norm_9,
            matmul_1,
            matmul_10,
            matmul_11,
            matmul_12,
            matmul_13,
            matmul_14,
            matmul_16,
            matmul_17,
            matmul_18,
            matmul_19,
            matmul_2,
            matmul_20,
            matmul_22,
            matmul_23,
            matmul_24,
            matmul_25,
            matmul_26,
            matmul_28,
            matmul_29,
            matmul_30,
            matmul_31,
            matmul_32,
            matmul_34,
            matmul_35,
            matmul_36,
            matmul_4,
            matmul_5,
            matmul_6,
            matmul_7,
            matmul_8,
            multiply_1,
            multiply_10,
            multiply_11,
            multiply_12,
            multiply_2,
            multiply_3,
            multiply_4,
            multiply_5,
            multiply_6,
            multiply_7,
            multiply_8,
            multiply_9,
            pad_0,
            pad_1,
            pad_2,
            pad_3,
            pad_4,
            pad_5,
            parameter_0,
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
            reshape_27,
            reshape_29,
            reshape_3,
            reshape_30,
            reshape_33,
            reshape_34,
            reshape_36,
            reshape_37,
            reshape_38,
            reshape_40,
            reshape_41,
            reshape_44,
            reshape_46,
            reshape_47,
            reshape_49,
            reshape_5,
            reshape_50,
            reshape_51,
            reshape_53,
            reshape_54,
            reshape_57,
            reshape_58,
            reshape_6,
            reshape_60,
            reshape_61,
            reshape_62,
            reshape_64,
            reshape_65,
            reshape_68,
            reshape_70,
            reshape_71,
            reshape_73,
            reshape_74,
            reshape_75,
            reshape_76,
            reshape_9,
            roll_0,
            roll_1,
            roll_2,
            roll_3,
            roll_4,
            roll_5,
            scale_1,
            scale_2,
            scale_3,
            scale_4,
            scale_5,
            scale_6,
            set_value__8,
            slice_10,
            slice_11,
            slice_14,
            slice_15,
            slice_18,
            slice_19,
            slice_2,
            slice_22,
            slice_23,
            slice_3,
            slice_6,
            slice_7,
            softmax_0,
            softmax_1,
            softmax_2,
            softmax_3,
            softmax_4,
            softmax_5,
            strided_slice_0,
            strided_slice_1,
            strided_slice_2,
            strided_slice_3,
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
            transpose_19,
            transpose_2,
            transpose_20,
            transpose_21,
            transpose_22,
            transpose_23,
            transpose_24,
            transpose_25,
            transpose_26,
            transpose_27,
            transpose_28,
            transpose_29,
            transpose_3,
            transpose_30,
            transpose_31,
            transpose_32,
            transpose_33,
            transpose_34,
            transpose_35,
            transpose_36,
            transpose_4,
            transpose_5,
            transpose_6,
            transpose_7,
            transpose_8,
            transpose_9,
            unsqueeze_10,
            unsqueeze_11,
            unsqueeze_13,
            unsqueeze_2,
            unsqueeze_3,
            unsqueeze_5,
            unsqueeze_6,
            unsqueeze_7,
            unsqueeze_9,
        )

        return matmul_0
