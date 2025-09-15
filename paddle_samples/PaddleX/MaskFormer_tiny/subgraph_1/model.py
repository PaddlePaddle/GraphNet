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
        parameter_75,
        parameter_76,
        parameter_77,
        parameter_78,
        parameter_79,
        parameter_80,
        parameter_81,
        parameter_82,
        parameter_83,
        parameter_84,
        parameter_85,
        parameter_86,
        parameter_87,
        parameter_88,
        parameter_89,
        parameter_90,
        parameter_91,
        parameter_92,
        parameter_93,
        parameter_94,
        parameter_95,
        parameter_96,
        parameter_97,
        parameter_98,
        parameter_99,
        parameter_100,
        parameter_101,
        parameter_102,
        parameter_103,
        parameter_104,
        parameter_105,
        parameter_106,
        parameter_107,
        parameter_108,
        parameter_109,
        parameter_110,
        parameter_111,
        parameter_112,
        parameter_113,
        parameter_114,
        parameter_115,
        parameter_116,
        parameter_117,
        parameter_118,
        parameter_119,
        parameter_120,
        parameter_121,
        parameter_122,
        parameter_123,
        parameter_124,
        parameter_125,
        parameter_126,
        parameter_127,
        parameter_128,
        parameter_129,
        parameter_130,
        parameter_131,
        parameter_132,
        parameter_133,
        parameter_134,
        parameter_135,
        parameter_136,
        parameter_137,
        parameter_138,
        parameter_139,
        parameter_140,
        parameter_141,
        parameter_142,
        parameter_143,
        parameter_144,
        parameter_145,
        parameter_146,
        parameter_147,
        parameter_148,
        parameter_149,
        parameter_150,
        parameter_151,
        parameter_152,
        parameter_153,
        parameter_154,
        parameter_155,
        parameter_156,
        parameter_157,
        parameter_158,
        parameter_159,
        parameter_160,
        parameter_161,
        parameter_162,
        parameter_163,
        parameter_164,
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
        data_13,
        data_14,
        data_15,
        data_16,
        data_17,
        data_18,
        data_19,
        data_20,
        data_21,
        data_22,
        data_23,
        data_24,
    ):
        # pd_op.conv2d: (1x96x128x128xf32) <- (1x3x512x512xf32, 96x3x4x4xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_12, parameter_164, [4, 4], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_12, parameter_164

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_163, full_int_array_0)
        del full_int_array_0, parameter_163

        # pd_op.add: (1x96x128x128xf32) <- (1x96x128x128xf32, 1x96x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_0, reshape_0)
        del conv2d_0, reshape_0

        # pd_op.flatten: (1x96x16384xf32) <- (1x96x128x128xf32)
        flatten_0 = paddle._C_ops.flatten(add_0, 2, 3)
        del add_0

        # pd_op.transpose: (1x16384x96xf32) <- (1x96x16384xf32)
        transpose_4 = paddle._C_ops.transpose(flatten_0, [0, 2, 1])
        del flatten_0

        # pd_op.layer_norm: (1x16384x96xf32, 1x16384xf32, 1x16384xf32) <- (1x16384x96xf32, 96xf32, 96xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_4, parameter_162, parameter_161, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_161, parameter_162, transpose_4

        # pd_op.transpose: (1x96x16384xf32) <- (1x16384x96xf32)
        transpose_5 = paddle._C_ops.transpose(layer_norm_0, [0, 2, 1])
        del layer_norm_0

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [-1, 96, 128, 128]

        # pd_op.reshape: (1x96x128x128xf32) <- (1x96x16384xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(transpose_5, full_int_array_1)
        del full_int_array_1, transpose_5

        # pd_op.flatten: (1x96x16384xf32) <- (1x96x128x128xf32)
        flatten_1 = paddle._C_ops.flatten(reshape_1, 2, 3)
        del reshape_1

        # pd_op.transpose: (1x16384x96xf32) <- (1x96x16384xf32)
        transpose_6 = paddle._C_ops.transpose(flatten_1, [0, 2, 1])
        del flatten_1

        # pd_op.full: (1x133x133x1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1, 133, 133, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [0, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [-7, -7]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [1, 1]

        # pd_op.set_value_: (1x133x133x1xf32) <- (1x133x133x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__0 = paddle._C_ops.set_value_(
            full_0,
            full_int_array_2,
            full_int_array_3,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("0")],
        )
        del full_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [0, -7]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_6 = [-7, -3]

        # pd_op.set_value_: (1x133x133x1xf32) <- (1x133x133x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__1 = paddle._C_ops.set_value_(
            set_value__0,
            full_int_array_5,
            full_int_array_6,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("1")],
        )
        del set_value__0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_7 = [0, -3]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_8 = [-7, 2147483647]

        # pd_op.set_value_: (1x133x133x1xf32) <- (1x133x133x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__2 = paddle._C_ops.set_value_(
            set_value__1,
            full_int_array_7,
            full_int_array_8,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("2")],
        )
        del set_value__1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_9 = [-7, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_10 = [-3, -7]

        # pd_op.set_value_: (1x133x133x1xf32) <- (1x133x133x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__3 = paddle._C_ops.set_value_(
            set_value__2,
            full_int_array_9,
            full_int_array_10,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("3")],
        )
        del set_value__2

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_11 = [-3, -3]

        # pd_op.set_value_: (1x133x133x1xf32) <- (1x133x133x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__4 = paddle._C_ops.set_value_(
            set_value__3,
            full_int_array_3,
            full_int_array_11,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("4")],
        )
        del set_value__3

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_12 = [-3, 2147483647]

        # pd_op.set_value_: (1x133x133x1xf32) <- (1x133x133x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__5 = paddle._C_ops.set_value_(
            set_value__4,
            full_int_array_6,
            full_int_array_12,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("5")],
        )
        del set_value__4

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_13 = [-3, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_14 = [2147483647, -7]

        # pd_op.set_value_: (1x133x133x1xf32) <- (1x133x133x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__6 = paddle._C_ops.set_value_(
            set_value__5,
            full_int_array_13,
            full_int_array_14,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("6")],
        )
        del set_value__5

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_15 = [2147483647, -3]

        # pd_op.set_value_: (1x133x133x1xf32) <- (1x133x133x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__7 = paddle._C_ops.set_value_(
            set_value__6,
            full_int_array_10,
            full_int_array_15,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("7")],
        )
        del set_value__6

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_16 = [2147483647, 2147483647]

        # pd_op.set_value_: (1x133x133x1xf32) <- (1x133x133x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__8 = paddle._C_ops.set_value_(
            set_value__7,
            full_int_array_11,
            full_int_array_16,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("8")],
        )
        del set_value__7

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_17 = [1, 19, 7, 19, 7, 1]

        # pd_op.reshape: (1x19x7x19x7x1xf32) <- (1x133x133x1xf32, 6xi64)
        reshape_2 = paddle._C_ops.reshape(set_value__8, full_int_array_17)
        del full_int_array_17

        # pd_op.transpose: (1x19x19x7x7x1xf32) <- (1x19x7x19x7x1xf32)
        transpose_7 = paddle._C_ops.transpose(reshape_2, [0, 1, 3, 2, 4, 5])
        del reshape_2

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_18 = [-1, 7, 7, 1]

        # pd_op.reshape: (361x7x7x1xf32) <- (1x19x19x7x7x1xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_7, full_int_array_18)
        del transpose_7

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_19 = [-1, 49]

        # pd_op.reshape: (361x49xf32) <- (361x7x7x1xf32, 2xi64)
        reshape_4 = paddle._C_ops.reshape(reshape_3, full_int_array_19)
        del reshape_3

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_20 = [1]

        # pd_op.unsqueeze: (361x1x49xf32) <- (361x49xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(reshape_4, full_int_array_20)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_21 = [2]

        # pd_op.unsqueeze: (361x49x1xf32) <- (361x49xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(reshape_4, full_int_array_21)
        del reshape_4

        # pd_op.subtract: (361x49x49xf32) <- (361x1x49xf32, 361x49x1xf32)
        subtract_0 = paddle._C_ops.subtract(unsqueeze_0, unsqueeze_1)
        del unsqueeze_0, unsqueeze_1

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (361x49x49xf32) <- (361x49x49xf32, 1xf32)
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

        # pd_op.scale: (361x49x49xf32) <- (361x49x49xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(full_like_0, full_2, float("0"), True)
        del full_like_0

        # pd_op.full: (xf32) <- ()
        full_3 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.not_equal: (361x49x49xb) <- (361x49x49xf32, xf32)
        not_equal_0 = paddle._C_ops.not_equal(subtract_0, full_3)
        del subtract_0

        # pd_op.cast: (361x49x49xf32) <- (361x49x49xb)
        cast_0 = paddle._C_ops.cast(not_equal_0, paddle.float32)
        del not_equal_0

        # pd_op.multiply: (361x49x49xf32) <- (361x49x49xf32, 361x49x49xf32)
        multiply_0 = paddle._C_ops.multiply(scale_0, cast_0)
        del cast_0, scale_0

        # pd_op.layer_norm: (1x16384x96xf32, 1x16384xf32, 1x16384xf32) <- (1x16384x96xf32, 96xf32, 96xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_6, parameter_160, parameter_159, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_159, parameter_160

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_22 = [1, 128, 128, 96]

        # pd_op.reshape: (1x128x128x96xf32) <- (1x16384x96xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(layer_norm_3, full_int_array_22)
        del layer_norm_3

        # pd_op.transpose: (1x96x128x128xf32) <- (1x128x128x96xf32)
        transpose_8 = paddle._C_ops.transpose(reshape_5, [0, 3, 1, 2])
        del reshape_5

        # pd_op.unsqueeze: (1x96x1x128x128xf32) <- (1x96x128x128xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(transpose_8, full_int_array_21)
        del transpose_8

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_23 = [0, 5, 0, 5, 0, 0]

        # pd_op.pad3d: (1x96x1x133x133xf32) <- (1x96x1x128x128xf32, 6xi64)
        pad3d_0 = paddle._C_ops.pad3d(
            unsqueeze_2, full_int_array_23, "constant", float("0"), "NCDHW"
        )
        del unsqueeze_2

        # pd_op.squeeze: (1x96x133x133xf32) <- (1x96x1x133x133xf32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(pad3d_0, full_int_array_21)
        del pad3d_0

        # pd_op.transpose: (1x133x133x96xf32) <- (1x96x133x133xf32)
        transpose_9 = paddle._C_ops.transpose(squeeze_0, [0, 2, 3, 1])
        del squeeze_0

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_24 = [1, 19, 7, 19, 7, 96]

        # pd_op.reshape: (1x19x7x19x7x96xf32) <- (1x133x133x96xf32, 6xi64)
        reshape_6 = paddle._C_ops.reshape(transpose_9, full_int_array_24)
        del transpose_9

        # pd_op.transpose: (1x19x19x7x7x96xf32) <- (1x19x7x19x7x96xf32)
        transpose_10 = paddle._C_ops.transpose(reshape_6, [0, 1, 3, 2, 4, 5])
        del reshape_6

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_25 = [-1, 7, 7, 96]

        # pd_op.reshape: (361x7x7x96xf32) <- (1x19x19x7x7x96xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_10, full_int_array_25)
        del transpose_10

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_26 = [-1, 49, 96]

        # pd_op.reshape: (361x49x96xf32) <- (361x7x7x96xf32, 3xi64)
        reshape_8 = paddle._C_ops.reshape(reshape_7, full_int_array_26)
        del reshape_7

        # pd_op.matmul: (361x49x288xf32) <- (361x49x96xf32, 96x288xf32)
        matmul_0 = paddle._C_ops.matmul(reshape_8, parameter_158, False, False)
        del parameter_158, reshape_8

        # pd_op.add: (361x49x288xf32) <- (361x49x288xf32, 288xf32)
        add_1 = paddle._C_ops.add(matmul_0, parameter_157)
        del matmul_0, parameter_157

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_27 = [361, 49, 3, 3, 32]

        # pd_op.reshape: (361x49x3x3x32xf32) <- (361x49x288xf32, 5xi64)
        reshape_9 = paddle._C_ops.reshape(add_1, full_int_array_27)
        del add_1

        # pd_op.transpose: (3x361x3x49x32xf32) <- (361x49x3x3x32xf32)
        transpose_11 = paddle._C_ops.transpose(reshape_9, [2, 0, 3, 1, 4])
        del reshape_9

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_28 = [0]

        # pd_op.slice: (361x3x49x32xf32) <- (3x361x3x49x32xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            transpose_11, [0], full_int_array_28, full_int_array_20, [1], [0]
        )

        # pd_op.slice: (361x3x49x32xf32) <- (3x361x3x49x32xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            transpose_11, [0], full_int_array_20, full_int_array_21, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_29 = [3]

        # pd_op.slice: (361x3x49x32xf32) <- (3x361x3x49x32xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            transpose_11, [0], full_int_array_21, full_int_array_29, [1], [0]
        )
        del transpose_11

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (361x3x49x32xf32) <- (361x3x49x32xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_0, full_4, float("0"), True)
        del slice_0

        # pd_op.transpose: (361x3x32x49xf32) <- (361x3x49x32xf32)
        transpose_12 = paddle._C_ops.transpose(slice_1, [0, 1, 3, 2])
        del slice_1

        # pd_op.matmul: (361x3x49x49xf32) <- (361x3x49x32xf32, 361x3x32x49xf32)
        matmul_1 = paddle._C_ops.matmul(scale_1, transpose_12, False, False)
        del scale_1, transpose_12

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_30 = [-1]

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        reshape_10 = paddle._C_ops.reshape(data_13, full_int_array_30)
        del data_13

        # pd_op.index_select: (2401x3xf32) <- (169x3xf32, 2401xi64)
        index_select_0 = paddle._C_ops.index_select(data_0, reshape_10, 0)
        del data_0, reshape_10

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_31 = [49, 49, -1]

        # pd_op.reshape: (49x49x3xf32) <- (2401x3xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(index_select_0, full_int_array_31)
        del index_select_0

        # pd_op.transpose: (3x49x49xf32) <- (49x49x3xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_11, [2, 0, 1])
        del reshape_11

        # pd_op.unsqueeze: (1x3x49x49xf32) <- (3x49x49xf32, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(transpose_13, full_int_array_28)
        del transpose_13

        # pd_op.add: (361x3x49x49xf32) <- (361x3x49x49xf32, 1x3x49x49xf32)
        add_2 = paddle._C_ops.add(matmul_1, unsqueeze_3)
        del matmul_1, unsqueeze_3

        # pd_op.softmax: (361x3x49x49xf32) <- (361x3x49x49xf32)
        softmax_0 = paddle._C_ops.softmax(add_2, -1)
        del add_2

        # pd_op.matmul: (361x3x49x32xf32) <- (361x3x49x49xf32, 361x3x49x32xf32)
        matmul_2 = paddle._C_ops.matmul(softmax_0, slice_2, False, False)
        del slice_2, softmax_0

        # pd_op.transpose: (361x49x3x32xf32) <- (361x3x49x32xf32)
        transpose_14 = paddle._C_ops.transpose(matmul_2, [0, 2, 1, 3])
        del matmul_2

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_32 = [361, 49, 96]

        # pd_op.reshape: (361x49x96xf32) <- (361x49x3x32xf32, 3xi64)
        reshape_12 = paddle._C_ops.reshape(transpose_14, full_int_array_32)
        del transpose_14

        # pd_op.matmul: (361x49x96xf32) <- (361x49x96xf32, 96x96xf32)
        matmul_3 = paddle._C_ops.matmul(reshape_12, parameter_156, False, False)
        del parameter_156, reshape_12

        # pd_op.add: (361x49x96xf32) <- (361x49x96xf32, 96xf32)
        add_3 = paddle._C_ops.add(matmul_3, parameter_155)
        del matmul_3, parameter_155

        # pd_op.reshape: (361x7x7x96xf32) <- (361x49x96xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(add_3, full_int_array_25)
        del add_3

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_33 = [1, 19, 19, 7, 7, -1]

        # pd_op.reshape: (1x19x19x7x7x96xf32) <- (361x7x7x96xf32, 6xi64)
        reshape_14 = paddle._C_ops.reshape(reshape_13, full_int_array_33)
        del reshape_13

        # pd_op.transpose: (1x19x7x19x7x96xf32) <- (1x19x19x7x7x96xf32)
        transpose_15 = paddle._C_ops.transpose(reshape_14, [0, 1, 3, 2, 4, 5])
        del reshape_14

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_34 = [1, 133, 133, -1]

        # pd_op.reshape: (1x133x133x96xf32) <- (1x19x7x19x7x96xf32, 4xi64)
        reshape_15 = paddle._C_ops.reshape(transpose_15, full_int_array_34)
        del transpose_15

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_35 = [128, 128]

        # pd_op.slice: (1x128x128x96xf32) <- (1x133x133x96xf32, 2xi64, 2xi64)
        slice_3 = paddle._C_ops.slice(
            reshape_15, [1, 2], full_int_array_2, full_int_array_35, [1, 1], []
        )
        del reshape_15

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_36 = [1, 16384, 96]

        # pd_op.reshape: (1x16384x96xf32) <- (1x128x128x96xf32, 3xi64)
        reshape_16 = paddle._C_ops.reshape(slice_3, full_int_array_36)
        del slice_3

        # pd_op.add: (1x16384x96xf32) <- (1x16384x96xf32, 1x16384x96xf32)
        add_4 = paddle._C_ops.add(transpose_6, reshape_16)
        del reshape_16, transpose_6

        # pd_op.layer_norm: (1x16384x96xf32, 1x16384xf32, 1x16384xf32) <- (1x16384x96xf32, 96xf32, 96xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_4, parameter_154, parameter_153, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_153, parameter_154

        # pd_op.matmul: (1x16384x384xf32) <- (1x16384x96xf32, 96x384xf32)
        matmul_4 = paddle._C_ops.matmul(layer_norm_6, parameter_152, False, False)
        del layer_norm_6, parameter_152

        # pd_op.add: (1x16384x384xf32) <- (1x16384x384xf32, 384xf32)
        add_5 = paddle._C_ops.add(matmul_4, parameter_151)
        del matmul_4, parameter_151

        # pd_op.gelu: (1x16384x384xf32) <- (1x16384x384xf32)
        gelu_0 = paddle._C_ops.gelu(add_5, False)
        del add_5

        # pd_op.matmul: (1x16384x96xf32) <- (1x16384x384xf32, 384x96xf32)
        matmul_5 = paddle._C_ops.matmul(gelu_0, parameter_150, False, False)
        del gelu_0, parameter_150

        # pd_op.add: (1x16384x96xf32) <- (1x16384x96xf32, 96xf32)
        add_6 = paddle._C_ops.add(matmul_5, parameter_149)
        del matmul_5, parameter_149

        # pd_op.add: (1x16384x96xf32) <- (1x16384x96xf32, 1x16384x96xf32)
        add_7 = paddle._C_ops.add(add_4, add_6)
        del add_4, add_6

        # pd_op.layer_norm: (1x16384x96xf32, 1x16384xf32, 1x16384xf32) <- (1x16384x96xf32, 96xf32, 96xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_7, parameter_148, parameter_147, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_147, parameter_148

        # pd_op.reshape: (1x128x128x96xf32) <- (1x16384x96xf32, 4xi64)
        reshape_17 = paddle._C_ops.reshape(layer_norm_9, full_int_array_22)
        del layer_norm_9

        # pd_op.transpose: (1x96x128x128xf32) <- (1x128x128x96xf32)
        transpose_16 = paddle._C_ops.transpose(reshape_17, [0, 3, 1, 2])
        del reshape_17

        # pd_op.unsqueeze: (1x96x1x128x128xf32) <- (1x96x128x128xf32, 1xi64)
        unsqueeze_4 = paddle._C_ops.unsqueeze(transpose_16, full_int_array_21)
        del transpose_16

        # pd_op.pad3d: (1x96x1x133x133xf32) <- (1x96x1x128x128xf32, 6xi64)
        pad3d_1 = paddle._C_ops.pad3d(
            unsqueeze_4, full_int_array_23, "constant", float("0"), "NCDHW"
        )
        del unsqueeze_4

        # pd_op.squeeze: (1x96x133x133xf32) <- (1x96x1x133x133xf32, 1xi64)
        squeeze_1 = paddle._C_ops.squeeze(pad3d_1, full_int_array_21)
        del pad3d_1

        # pd_op.transpose: (1x133x133x96xf32) <- (1x96x133x133xf32)
        transpose_17 = paddle._C_ops.transpose(squeeze_1, [0, 2, 3, 1])
        del squeeze_1

        # pd_op.roll: (1x133x133x96xf32) <- (1x133x133x96xf32, 2xi64)
        roll_0 = paddle._C_ops.roll(transpose_17, full_int_array_11, [1, 2])
        del transpose_17

        # pd_op.reshape: (1x19x7x19x7x96xf32) <- (1x133x133x96xf32, 6xi64)
        reshape_18 = paddle._C_ops.reshape(roll_0, full_int_array_24)
        del full_int_array_24, roll_0

        # pd_op.transpose: (1x19x19x7x7x96xf32) <- (1x19x7x19x7x96xf32)
        transpose_18 = paddle._C_ops.transpose(reshape_18, [0, 1, 3, 2, 4, 5])
        del reshape_18

        # pd_op.reshape: (361x7x7x96xf32) <- (1x19x19x7x7x96xf32, 4xi64)
        reshape_19 = paddle._C_ops.reshape(transpose_18, full_int_array_25)
        del transpose_18

        # pd_op.reshape: (361x49x96xf32) <- (361x7x7x96xf32, 3xi64)
        reshape_20 = paddle._C_ops.reshape(reshape_19, full_int_array_26)
        del full_int_array_26, reshape_19

        # pd_op.matmul: (361x49x288xf32) <- (361x49x96xf32, 96x288xf32)
        matmul_6 = paddle._C_ops.matmul(reshape_20, parameter_146, False, False)
        del parameter_146, reshape_20

        # pd_op.add: (361x49x288xf32) <- (361x49x288xf32, 288xf32)
        add_8 = paddle._C_ops.add(matmul_6, parameter_145)
        del matmul_6, parameter_145

        # pd_op.reshape: (361x49x3x3x32xf32) <- (361x49x288xf32, 5xi64)
        reshape_21 = paddle._C_ops.reshape(add_8, full_int_array_27)
        del add_8, full_int_array_27

        # pd_op.transpose: (3x361x3x49x32xf32) <- (361x49x3x3x32xf32)
        transpose_19 = paddle._C_ops.transpose(reshape_21, [2, 0, 3, 1, 4])
        del reshape_21

        # pd_op.slice: (361x3x49x32xf32) <- (3x361x3x49x32xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            transpose_19, [0], full_int_array_28, full_int_array_20, [1], [0]
        )

        # pd_op.slice: (361x3x49x32xf32) <- (3x361x3x49x32xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            transpose_19, [0], full_int_array_20, full_int_array_21, [1], [0]
        )

        # pd_op.slice: (361x3x49x32xf32) <- (3x361x3x49x32xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            transpose_19, [0], full_int_array_21, full_int_array_29, [1], [0]
        )
        del transpose_19

        # pd_op.scale: (361x3x49x32xf32) <- (361x3x49x32xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_4, full_4, float("0"), True)
        del slice_4

        # pd_op.transpose: (361x3x32x49xf32) <- (361x3x49x32xf32)
        transpose_20 = paddle._C_ops.transpose(slice_5, [0, 1, 3, 2])
        del slice_5

        # pd_op.matmul: (361x3x49x49xf32) <- (361x3x49x32xf32, 361x3x32x49xf32)
        matmul_7 = paddle._C_ops.matmul(scale_2, transpose_20, False, False)
        del scale_2, transpose_20

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        reshape_22 = paddle._C_ops.reshape(data_14, full_int_array_30)
        del data_14

        # pd_op.index_select: (2401x3xf32) <- (169x3xf32, 2401xi64)
        index_select_1 = paddle._C_ops.index_select(data_1, reshape_22, 0)
        del data_1, reshape_22

        # pd_op.reshape: (49x49x3xf32) <- (2401x3xf32, 3xi64)
        reshape_23 = paddle._C_ops.reshape(index_select_1, full_int_array_31)
        del index_select_1

        # pd_op.transpose: (3x49x49xf32) <- (49x49x3xf32)
        transpose_21 = paddle._C_ops.transpose(reshape_23, [2, 0, 1])
        del reshape_23

        # pd_op.unsqueeze: (1x3x49x49xf32) <- (3x49x49xf32, 1xi64)
        unsqueeze_5 = paddle._C_ops.unsqueeze(transpose_21, full_int_array_28)
        del transpose_21

        # pd_op.add: (361x3x49x49xf32) <- (361x3x49x49xf32, 1x3x49x49xf32)
        add_9 = paddle._C_ops.add(matmul_7, unsqueeze_5)
        del matmul_7, unsqueeze_5

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_37 = [1, 361, 3, 49, 49]

        # pd_op.reshape: (1x361x3x49x49xf32) <- (361x3x49x49xf32, 5xi64)
        reshape_24 = paddle._C_ops.reshape(add_9, full_int_array_37)
        del add_9, full_int_array_37

        # pd_op.unsqueeze: (361x1x49x49xf32) <- (361x49x49xf32, 1xi64)
        unsqueeze_6 = paddle._C_ops.unsqueeze(multiply_0, full_int_array_20)
        del multiply_0

        # pd_op.unsqueeze: (1x361x1x49x49xf32) <- (361x1x49x49xf32, 1xi64)
        unsqueeze_7 = paddle._C_ops.unsqueeze(unsqueeze_6, full_int_array_28)
        del unsqueeze_6

        # pd_op.add: (1x361x3x49x49xf32) <- (1x361x3x49x49xf32, 1x361x1x49x49xf32)
        add_10 = paddle._C_ops.add(reshape_24, unsqueeze_7)
        del reshape_24, unsqueeze_7

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_38 = [-1, 3, 49, 49]

        # pd_op.reshape: (361x3x49x49xf32) <- (1x361x3x49x49xf32, 4xi64)
        reshape_25 = paddle._C_ops.reshape(add_10, full_int_array_38)
        del add_10, full_int_array_38

        # pd_op.softmax: (361x3x49x49xf32) <- (361x3x49x49xf32)
        softmax_1 = paddle._C_ops.softmax(reshape_25, -1)
        del reshape_25

        # pd_op.matmul: (361x3x49x32xf32) <- (361x3x49x49xf32, 361x3x49x32xf32)
        matmul_8 = paddle._C_ops.matmul(softmax_1, slice_6, False, False)
        del slice_6, softmax_1

        # pd_op.transpose: (361x49x3x32xf32) <- (361x3x49x32xf32)
        transpose_22 = paddle._C_ops.transpose(matmul_8, [0, 2, 1, 3])
        del matmul_8

        # pd_op.reshape: (361x49x96xf32) <- (361x49x3x32xf32, 3xi64)
        reshape_26 = paddle._C_ops.reshape(transpose_22, full_int_array_32)
        del full_int_array_32, transpose_22

        # pd_op.matmul: (361x49x96xf32) <- (361x49x96xf32, 96x96xf32)
        matmul_9 = paddle._C_ops.matmul(reshape_26, parameter_144, False, False)
        del parameter_144, reshape_26

        # pd_op.add: (361x49x96xf32) <- (361x49x96xf32, 96xf32)
        add_11 = paddle._C_ops.add(matmul_9, parameter_143)
        del matmul_9, parameter_143

        # pd_op.reshape: (361x7x7x96xf32) <- (361x49x96xf32, 4xi64)
        reshape_27 = paddle._C_ops.reshape(add_11, full_int_array_25)
        del add_11, full_int_array_25

        # pd_op.reshape: (1x19x19x7x7x96xf32) <- (361x7x7x96xf32, 6xi64)
        reshape_28 = paddle._C_ops.reshape(reshape_27, full_int_array_33)
        del full_int_array_33, reshape_27

        # pd_op.transpose: (1x19x7x19x7x96xf32) <- (1x19x19x7x7x96xf32)
        transpose_23 = paddle._C_ops.transpose(reshape_28, [0, 1, 3, 2, 4, 5])
        del reshape_28

        # pd_op.reshape: (1x133x133x96xf32) <- (1x19x7x19x7x96xf32, 4xi64)
        reshape_29 = paddle._C_ops.reshape(transpose_23, full_int_array_34)
        del full_int_array_34, transpose_23

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_39 = [3, 3]

        # pd_op.roll: (1x133x133x96xf32) <- (1x133x133x96xf32, 2xi64)
        roll_1 = paddle._C_ops.roll(reshape_29, full_int_array_39, [1, 2])
        del reshape_29

        # pd_op.slice: (1x128x128x96xf32) <- (1x133x133x96xf32, 2xi64, 2xi64)
        slice_7 = paddle._C_ops.slice(
            roll_1, [1, 2], full_int_array_2, full_int_array_35, [1, 1], []
        )
        del full_int_array_35, roll_1

        # pd_op.reshape: (1x16384x96xf32) <- (1x128x128x96xf32, 3xi64)
        reshape_30 = paddle._C_ops.reshape(slice_7, full_int_array_36)
        del full_int_array_36, slice_7

        # pd_op.add: (1x16384x96xf32) <- (1x16384x96xf32, 1x16384x96xf32)
        add_12 = paddle._C_ops.add(add_7, reshape_30)
        del add_7, reshape_30

        # pd_op.layer_norm: (1x16384x96xf32, 1x16384xf32, 1x16384xf32) <- (1x16384x96xf32, 96xf32, 96xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_12, parameter_142, parameter_141, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_141, parameter_142

        # pd_op.matmul: (1x16384x384xf32) <- (1x16384x96xf32, 96x384xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_12, parameter_140, False, False)
        del layer_norm_12, parameter_140

        # pd_op.add: (1x16384x384xf32) <- (1x16384x384xf32, 384xf32)
        add_13 = paddle._C_ops.add(matmul_10, parameter_139)
        del matmul_10, parameter_139

        # pd_op.gelu: (1x16384x384xf32) <- (1x16384x384xf32)
        gelu_1 = paddle._C_ops.gelu(add_13, False)
        del add_13

        # pd_op.matmul: (1x16384x96xf32) <- (1x16384x384xf32, 384x96xf32)
        matmul_11 = paddle._C_ops.matmul(gelu_1, parameter_138, False, False)
        del gelu_1, parameter_138

        # pd_op.add: (1x16384x96xf32) <- (1x16384x96xf32, 96xf32)
        add_14 = paddle._C_ops.add(matmul_11, parameter_137)
        del matmul_11, parameter_137

        # pd_op.add: (1x16384x96xf32) <- (1x16384x96xf32, 1x16384x96xf32)
        add_15 = paddle._C_ops.add(add_12, add_14)
        del add_12, add_14

        # pd_op.reshape: (1x128x128x96xf32) <- (1x16384x96xf32, 4xi64)
        reshape_31 = paddle._C_ops.reshape(add_15, full_int_array_22)
        del full_int_array_22

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_40 = [2, 2]

        # pd_op.strided_slice: (1x64x64x96xf32) <- (1x128x128x96xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_0 = paddle._C_ops.strided_slice(
            reshape_31, [1, 2], full_int_array_2, full_int_array_16, full_int_array_40
        )

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_41 = [1, 0]

        # pd_op.strided_slice: (1x64x64x96xf32) <- (1x128x128x96xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_1 = paddle._C_ops.strided_slice(
            reshape_31, [1, 2], full_int_array_41, full_int_array_16, full_int_array_40
        )

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_42 = [0, 1]

        # pd_op.strided_slice: (1x64x64x96xf32) <- (1x128x128x96xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_2 = paddle._C_ops.strided_slice(
            reshape_31, [1, 2], full_int_array_42, full_int_array_16, full_int_array_40
        )

        # pd_op.strided_slice: (1x64x64x96xf32) <- (1x128x128x96xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_3 = paddle._C_ops.strided_slice(
            reshape_31, [1, 2], full_int_array_4, full_int_array_16, full_int_array_40
        )
        del reshape_31

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([1x64x64x96xf32, 1x64x64x96xf32, 1x64x64x96xf32, 1x64x64x96xf32]) <- (1x64x64x96xf32, 1x64x64x96xf32, 1x64x64x96xf32, 1x64x64x96xf32)
        combine_0 = [strided_slice_0, strided_slice_1, strided_slice_2, strided_slice_3]
        del strided_slice_0, strided_slice_1, strided_slice_2, strided_slice_3

        # pd_op.concat: (1x64x64x384xf32) <- ([1x64x64x96xf32, 1x64x64x96xf32, 1x64x64x96xf32, 1x64x64x96xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_5)
        del combine_0

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_43 = [1, -1, 384]

        # pd_op.reshape: (1x4096x384xf32) <- (1x64x64x384xf32, 3xi64)
        reshape_32 = paddle._C_ops.reshape(concat_0, full_int_array_43)
        del concat_0, full_int_array_43

        # pd_op.layer_norm: (1x4096x384xf32, 1x4096xf32, 1x4096xf32) <- (1x4096x384xf32, 384xf32, 384xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                reshape_32, parameter_136, parameter_135, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_135, parameter_136, reshape_32

        # pd_op.matmul: (1x4096x192xf32) <- (1x4096x384xf32, 384x192xf32)
        matmul_12 = paddle._C_ops.matmul(layer_norm_15, parameter_134, False, False)
        del layer_norm_15, parameter_134

        # pd_op.layer_norm: (1x16384x96xf32, 1x16384xf32, 1x16384xf32) <- (1x16384x96xf32, 96xf32, 96xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_15, parameter_133, parameter_132, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_15, parameter_132, parameter_133

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_44 = [-1, 128, 128, 96]

        # pd_op.reshape: (1x128x128x96xf32) <- (1x16384x96xf32, 4xi64)
        reshape_33 = paddle._C_ops.reshape(layer_norm_18, full_int_array_44)
        del full_int_array_44, layer_norm_18

        # pd_op.transpose: (1x96x128x128xf32) <- (1x128x128x96xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_33, [0, 3, 1, 2])
        del reshape_33

        # pd_op.full: (1x70x70x1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1, 70, 70, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x70x70x1xf32) <- (1x70x70x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__9 = paddle._C_ops.set_value_(
            full_6,
            full_int_array_2,
            full_int_array_3,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("0")],
        )
        del full_6

        # pd_op.set_value_: (1x70x70x1xf32) <- (1x70x70x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__10 = paddle._C_ops.set_value_(
            set_value__9,
            full_int_array_5,
            full_int_array_6,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("1")],
        )
        del set_value__9

        # pd_op.set_value_: (1x70x70x1xf32) <- (1x70x70x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__11 = paddle._C_ops.set_value_(
            set_value__10,
            full_int_array_7,
            full_int_array_8,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("2")],
        )
        del set_value__10

        # pd_op.set_value_: (1x70x70x1xf32) <- (1x70x70x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__12 = paddle._C_ops.set_value_(
            set_value__11,
            full_int_array_9,
            full_int_array_10,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("3")],
        )
        del set_value__11

        # pd_op.set_value_: (1x70x70x1xf32) <- (1x70x70x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__13 = paddle._C_ops.set_value_(
            set_value__12,
            full_int_array_3,
            full_int_array_11,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("4")],
        )
        del set_value__12

        # pd_op.set_value_: (1x70x70x1xf32) <- (1x70x70x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__14 = paddle._C_ops.set_value_(
            set_value__13,
            full_int_array_6,
            full_int_array_12,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("5")],
        )
        del set_value__13

        # pd_op.set_value_: (1x70x70x1xf32) <- (1x70x70x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__15 = paddle._C_ops.set_value_(
            set_value__14,
            full_int_array_13,
            full_int_array_14,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("6")],
        )
        del set_value__14

        # pd_op.set_value_: (1x70x70x1xf32) <- (1x70x70x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__16 = paddle._C_ops.set_value_(
            set_value__15,
            full_int_array_10,
            full_int_array_15,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("7")],
        )
        del set_value__15

        # pd_op.set_value_: (1x70x70x1xf32) <- (1x70x70x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__17 = paddle._C_ops.set_value_(
            set_value__16,
            full_int_array_11,
            full_int_array_16,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("8")],
        )
        del set_value__16

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_45 = [1, 10, 7, 10, 7, 1]

        # pd_op.reshape: (1x10x7x10x7x1xf32) <- (1x70x70x1xf32, 6xi64)
        reshape_34 = paddle._C_ops.reshape(set_value__17, full_int_array_45)
        del full_int_array_45

        # pd_op.transpose: (1x10x10x7x7x1xf32) <- (1x10x7x10x7x1xf32)
        transpose_24 = paddle._C_ops.transpose(reshape_34, [0, 1, 3, 2, 4, 5])
        del reshape_34

        # pd_op.reshape: (100x7x7x1xf32) <- (1x10x10x7x7x1xf32, 4xi64)
        reshape_35 = paddle._C_ops.reshape(transpose_24, full_int_array_18)
        del transpose_24

        # pd_op.reshape: (100x49xf32) <- (100x7x7x1xf32, 2xi64)
        reshape_36 = paddle._C_ops.reshape(reshape_35, full_int_array_19)
        del reshape_35

        # pd_op.unsqueeze: (100x1x49xf32) <- (100x49xf32, 1xi64)
        unsqueeze_8 = paddle._C_ops.unsqueeze(reshape_36, full_int_array_20)

        # pd_op.unsqueeze: (100x49x1xf32) <- (100x49xf32, 1xi64)
        unsqueeze_9 = paddle._C_ops.unsqueeze(reshape_36, full_int_array_21)
        del reshape_36

        # pd_op.subtract: (100x49x49xf32) <- (100x1x49xf32, 100x49x1xf32)
        subtract_1 = paddle._C_ops.subtract(unsqueeze_8, unsqueeze_9)
        del unsqueeze_8, unsqueeze_9

        # pd_op.full_like: (100x49x49xf32) <- (100x49x49xf32, 1xf32)
        full_like_1 = paddle._C_ops.full_like(
            subtract_1,
            full_1,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.scale: (100x49x49xf32) <- (100x49x49xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(full_like_1, full_2, float("0"), True)
        del full_like_1

        # pd_op.not_equal: (100x49x49xb) <- (100x49x49xf32, xf32)
        not_equal_1 = paddle._C_ops.not_equal(subtract_1, full_3)
        del subtract_1

        # pd_op.cast: (100x49x49xf32) <- (100x49x49xb)
        cast_1 = paddle._C_ops.cast(not_equal_1, paddle.float32)
        del not_equal_1

        # pd_op.multiply: (100x49x49xf32) <- (100x49x49xf32, 100x49x49xf32)
        multiply_1 = paddle._C_ops.multiply(scale_3, cast_1)
        del cast_1, scale_3

        # pd_op.layer_norm: (1x4096x192xf32, 1x4096xf32, 1x4096xf32) <- (1x4096x192xf32, 192xf32, 192xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                matmul_12, parameter_131, parameter_130, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_130, parameter_131

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_46 = [1, 64, 64, 192]

        # pd_op.reshape: (1x64x64x192xf32) <- (1x4096x192xf32, 4xi64)
        reshape_37 = paddle._C_ops.reshape(layer_norm_21, full_int_array_46)
        del layer_norm_21

        # pd_op.transpose: (1x192x64x64xf32) <- (1x64x64x192xf32)
        transpose_25 = paddle._C_ops.transpose(reshape_37, [0, 3, 1, 2])
        del reshape_37

        # pd_op.unsqueeze: (1x192x1x64x64xf32) <- (1x192x64x64xf32, 1xi64)
        unsqueeze_10 = paddle._C_ops.unsqueeze(transpose_25, full_int_array_21)
        del transpose_25

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_47 = [0, 6, 0, 6, 0, 0]

        # pd_op.pad3d: (1x192x1x70x70xf32) <- (1x192x1x64x64xf32, 6xi64)
        pad3d_2 = paddle._C_ops.pad3d(
            unsqueeze_10, full_int_array_47, "constant", float("0"), "NCDHW"
        )
        del unsqueeze_10

        # pd_op.squeeze: (1x192x70x70xf32) <- (1x192x1x70x70xf32, 1xi64)
        squeeze_2 = paddle._C_ops.squeeze(pad3d_2, full_int_array_21)
        del pad3d_2

        # pd_op.transpose: (1x70x70x192xf32) <- (1x192x70x70xf32)
        transpose_26 = paddle._C_ops.transpose(squeeze_2, [0, 2, 3, 1])
        del squeeze_2

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_48 = [1, 10, 7, 10, 7, 192]

        # pd_op.reshape: (1x10x7x10x7x192xf32) <- (1x70x70x192xf32, 6xi64)
        reshape_38 = paddle._C_ops.reshape(transpose_26, full_int_array_48)
        del transpose_26

        # pd_op.transpose: (1x10x10x7x7x192xf32) <- (1x10x7x10x7x192xf32)
        transpose_27 = paddle._C_ops.transpose(reshape_38, [0, 1, 3, 2, 4, 5])
        del reshape_38

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_49 = [-1, 7, 7, 192]

        # pd_op.reshape: (100x7x7x192xf32) <- (1x10x10x7x7x192xf32, 4xi64)
        reshape_39 = paddle._C_ops.reshape(transpose_27, full_int_array_49)
        del transpose_27

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_50 = [-1, 49, 192]

        # pd_op.reshape: (100x49x192xf32) <- (100x7x7x192xf32, 3xi64)
        reshape_40 = paddle._C_ops.reshape(reshape_39, full_int_array_50)
        del reshape_39

        # pd_op.matmul: (100x49x576xf32) <- (100x49x192xf32, 192x576xf32)
        matmul_13 = paddle._C_ops.matmul(reshape_40, parameter_129, False, False)
        del parameter_129, reshape_40

        # pd_op.add: (100x49x576xf32) <- (100x49x576xf32, 576xf32)
        add_16 = paddle._C_ops.add(matmul_13, parameter_128)
        del matmul_13, parameter_128

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_51 = [100, 49, 3, 6, 32]

        # pd_op.reshape: (100x49x3x6x32xf32) <- (100x49x576xf32, 5xi64)
        reshape_41 = paddle._C_ops.reshape(add_16, full_int_array_51)
        del add_16

        # pd_op.transpose: (3x100x6x49x32xf32) <- (100x49x3x6x32xf32)
        transpose_28 = paddle._C_ops.transpose(reshape_41, [2, 0, 3, 1, 4])
        del reshape_41

        # pd_op.slice: (100x6x49x32xf32) <- (3x100x6x49x32xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            transpose_28, [0], full_int_array_28, full_int_array_20, [1], [0]
        )

        # pd_op.slice: (100x6x49x32xf32) <- (3x100x6x49x32xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            transpose_28, [0], full_int_array_20, full_int_array_21, [1], [0]
        )

        # pd_op.slice: (100x6x49x32xf32) <- (3x100x6x49x32xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            transpose_28, [0], full_int_array_21, full_int_array_29, [1], [0]
        )
        del transpose_28

        # pd_op.scale: (100x6x49x32xf32) <- (100x6x49x32xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(slice_8, full_4, float("0"), True)
        del slice_8

        # pd_op.transpose: (100x6x32x49xf32) <- (100x6x49x32xf32)
        transpose_29 = paddle._C_ops.transpose(slice_9, [0, 1, 3, 2])
        del slice_9

        # pd_op.matmul: (100x6x49x49xf32) <- (100x6x49x32xf32, 100x6x32x49xf32)
        matmul_14 = paddle._C_ops.matmul(scale_4, transpose_29, False, False)
        del scale_4, transpose_29

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        reshape_42 = paddle._C_ops.reshape(data_15, full_int_array_30)
        del data_15

        # pd_op.index_select: (2401x6xf32) <- (169x6xf32, 2401xi64)
        index_select_2 = paddle._C_ops.index_select(data_2, reshape_42, 0)
        del data_2, reshape_42

        # pd_op.reshape: (49x49x6xf32) <- (2401x6xf32, 3xi64)
        reshape_43 = paddle._C_ops.reshape(index_select_2, full_int_array_31)
        del index_select_2

        # pd_op.transpose: (6x49x49xf32) <- (49x49x6xf32)
        transpose_30 = paddle._C_ops.transpose(reshape_43, [2, 0, 1])
        del reshape_43

        # pd_op.unsqueeze: (1x6x49x49xf32) <- (6x49x49xf32, 1xi64)
        unsqueeze_11 = paddle._C_ops.unsqueeze(transpose_30, full_int_array_28)
        del transpose_30

        # pd_op.add: (100x6x49x49xf32) <- (100x6x49x49xf32, 1x6x49x49xf32)
        add_17 = paddle._C_ops.add(matmul_14, unsqueeze_11)
        del matmul_14, unsqueeze_11

        # pd_op.softmax: (100x6x49x49xf32) <- (100x6x49x49xf32)
        softmax_2 = paddle._C_ops.softmax(add_17, -1)
        del add_17

        # pd_op.matmul: (100x6x49x32xf32) <- (100x6x49x49xf32, 100x6x49x32xf32)
        matmul_15 = paddle._C_ops.matmul(softmax_2, slice_10, False, False)
        del slice_10, softmax_2

        # pd_op.transpose: (100x49x6x32xf32) <- (100x6x49x32xf32)
        transpose_31 = paddle._C_ops.transpose(matmul_15, [0, 2, 1, 3])
        del matmul_15

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_52 = [100, 49, 192]

        # pd_op.reshape: (100x49x192xf32) <- (100x49x6x32xf32, 3xi64)
        reshape_44 = paddle._C_ops.reshape(transpose_31, full_int_array_52)
        del transpose_31

        # pd_op.matmul: (100x49x192xf32) <- (100x49x192xf32, 192x192xf32)
        matmul_16 = paddle._C_ops.matmul(reshape_44, parameter_127, False, False)
        del parameter_127, reshape_44

        # pd_op.add: (100x49x192xf32) <- (100x49x192xf32, 192xf32)
        add_18 = paddle._C_ops.add(matmul_16, parameter_126)
        del matmul_16, parameter_126

        # pd_op.reshape: (100x7x7x192xf32) <- (100x49x192xf32, 4xi64)
        reshape_45 = paddle._C_ops.reshape(add_18, full_int_array_49)
        del add_18

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_53 = [1, 10, 10, 7, 7, -1]

        # pd_op.reshape: (1x10x10x7x7x192xf32) <- (100x7x7x192xf32, 6xi64)
        reshape_46 = paddle._C_ops.reshape(reshape_45, full_int_array_53)
        del reshape_45

        # pd_op.transpose: (1x10x7x10x7x192xf32) <- (1x10x10x7x7x192xf32)
        transpose_32 = paddle._C_ops.transpose(reshape_46, [0, 1, 3, 2, 4, 5])
        del reshape_46

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_54 = [1, 70, 70, -1]

        # pd_op.reshape: (1x70x70x192xf32) <- (1x10x7x10x7x192xf32, 4xi64)
        reshape_47 = paddle._C_ops.reshape(transpose_32, full_int_array_54)
        del transpose_32

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_55 = [64, 64]

        # pd_op.slice: (1x64x64x192xf32) <- (1x70x70x192xf32, 2xi64, 2xi64)
        slice_11 = paddle._C_ops.slice(
            reshape_47, [1, 2], full_int_array_2, full_int_array_55, [1, 1], []
        )
        del reshape_47

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_56 = [1, 4096, 192]

        # pd_op.reshape: (1x4096x192xf32) <- (1x64x64x192xf32, 3xi64)
        reshape_48 = paddle._C_ops.reshape(slice_11, full_int_array_56)
        del slice_11

        # pd_op.add: (1x4096x192xf32) <- (1x4096x192xf32, 1x4096x192xf32)
        add_19 = paddle._C_ops.add(matmul_12, reshape_48)
        del matmul_12, reshape_48

        # pd_op.layer_norm: (1x4096x192xf32, 1x4096xf32, 1x4096xf32) <- (1x4096x192xf32, 192xf32, 192xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_19, parameter_125, parameter_124, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_124, parameter_125

        # pd_op.matmul: (1x4096x768xf32) <- (1x4096x192xf32, 192x768xf32)
        matmul_17 = paddle._C_ops.matmul(layer_norm_24, parameter_123, False, False)
        del layer_norm_24, parameter_123

        # pd_op.add: (1x4096x768xf32) <- (1x4096x768xf32, 768xf32)
        add_20 = paddle._C_ops.add(matmul_17, parameter_122)
        del matmul_17, parameter_122

        # pd_op.gelu: (1x4096x768xf32) <- (1x4096x768xf32)
        gelu_2 = paddle._C_ops.gelu(add_20, False)
        del add_20

        # pd_op.matmul: (1x4096x192xf32) <- (1x4096x768xf32, 768x192xf32)
        matmul_18 = paddle._C_ops.matmul(gelu_2, parameter_121, False, False)
        del gelu_2, parameter_121

        # pd_op.add: (1x4096x192xf32) <- (1x4096x192xf32, 192xf32)
        add_21 = paddle._C_ops.add(matmul_18, parameter_120)
        del matmul_18, parameter_120

        # pd_op.add: (1x4096x192xf32) <- (1x4096x192xf32, 1x4096x192xf32)
        add_22 = paddle._C_ops.add(add_19, add_21)
        del add_19, add_21

        # pd_op.layer_norm: (1x4096x192xf32, 1x4096xf32, 1x4096xf32) <- (1x4096x192xf32, 192xf32, 192xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_22, parameter_119, parameter_118, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_118, parameter_119

        # pd_op.reshape: (1x64x64x192xf32) <- (1x4096x192xf32, 4xi64)
        reshape_49 = paddle._C_ops.reshape(layer_norm_27, full_int_array_46)
        del layer_norm_27

        # pd_op.transpose: (1x192x64x64xf32) <- (1x64x64x192xf32)
        transpose_33 = paddle._C_ops.transpose(reshape_49, [0, 3, 1, 2])
        del reshape_49

        # pd_op.unsqueeze: (1x192x1x64x64xf32) <- (1x192x64x64xf32, 1xi64)
        unsqueeze_12 = paddle._C_ops.unsqueeze(transpose_33, full_int_array_21)
        del transpose_33

        # pd_op.pad3d: (1x192x1x70x70xf32) <- (1x192x1x64x64xf32, 6xi64)
        pad3d_3 = paddle._C_ops.pad3d(
            unsqueeze_12, full_int_array_47, "constant", float("0"), "NCDHW"
        )
        del full_int_array_47, unsqueeze_12

        # pd_op.squeeze: (1x192x70x70xf32) <- (1x192x1x70x70xf32, 1xi64)
        squeeze_3 = paddle._C_ops.squeeze(pad3d_3, full_int_array_21)
        del pad3d_3

        # pd_op.transpose: (1x70x70x192xf32) <- (1x192x70x70xf32)
        transpose_34 = paddle._C_ops.transpose(squeeze_3, [0, 2, 3, 1])
        del squeeze_3

        # pd_op.roll: (1x70x70x192xf32) <- (1x70x70x192xf32, 2xi64)
        roll_2 = paddle._C_ops.roll(transpose_34, full_int_array_11, [1, 2])
        del transpose_34

        # pd_op.reshape: (1x10x7x10x7x192xf32) <- (1x70x70x192xf32, 6xi64)
        reshape_50 = paddle._C_ops.reshape(roll_2, full_int_array_48)
        del full_int_array_48, roll_2

        # pd_op.transpose: (1x10x10x7x7x192xf32) <- (1x10x7x10x7x192xf32)
        transpose_35 = paddle._C_ops.transpose(reshape_50, [0, 1, 3, 2, 4, 5])
        del reshape_50

        # pd_op.reshape: (100x7x7x192xf32) <- (1x10x10x7x7x192xf32, 4xi64)
        reshape_51 = paddle._C_ops.reshape(transpose_35, full_int_array_49)
        del transpose_35

        # pd_op.reshape: (100x49x192xf32) <- (100x7x7x192xf32, 3xi64)
        reshape_52 = paddle._C_ops.reshape(reshape_51, full_int_array_50)
        del full_int_array_50, reshape_51

        # pd_op.matmul: (100x49x576xf32) <- (100x49x192xf32, 192x576xf32)
        matmul_19 = paddle._C_ops.matmul(reshape_52, parameter_117, False, False)
        del parameter_117, reshape_52

        # pd_op.add: (100x49x576xf32) <- (100x49x576xf32, 576xf32)
        add_23 = paddle._C_ops.add(matmul_19, parameter_116)
        del matmul_19, parameter_116

        # pd_op.reshape: (100x49x3x6x32xf32) <- (100x49x576xf32, 5xi64)
        reshape_53 = paddle._C_ops.reshape(add_23, full_int_array_51)
        del add_23, full_int_array_51

        # pd_op.transpose: (3x100x6x49x32xf32) <- (100x49x3x6x32xf32)
        transpose_36 = paddle._C_ops.transpose(reshape_53, [2, 0, 3, 1, 4])
        del reshape_53

        # pd_op.slice: (100x6x49x32xf32) <- (3x100x6x49x32xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            transpose_36, [0], full_int_array_28, full_int_array_20, [1], [0]
        )

        # pd_op.slice: (100x6x49x32xf32) <- (3x100x6x49x32xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            transpose_36, [0], full_int_array_20, full_int_array_21, [1], [0]
        )

        # pd_op.slice: (100x6x49x32xf32) <- (3x100x6x49x32xf32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            transpose_36, [0], full_int_array_21, full_int_array_29, [1], [0]
        )
        del transpose_36

        # pd_op.scale: (100x6x49x32xf32) <- (100x6x49x32xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(slice_12, full_4, float("0"), True)
        del slice_12

        # pd_op.transpose: (100x6x32x49xf32) <- (100x6x49x32xf32)
        transpose_37 = paddle._C_ops.transpose(slice_13, [0, 1, 3, 2])
        del slice_13

        # pd_op.matmul: (100x6x49x49xf32) <- (100x6x49x32xf32, 100x6x32x49xf32)
        matmul_20 = paddle._C_ops.matmul(scale_5, transpose_37, False, False)
        del scale_5, transpose_37

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        reshape_54 = paddle._C_ops.reshape(data_16, full_int_array_30)
        del data_16

        # pd_op.index_select: (2401x6xf32) <- (169x6xf32, 2401xi64)
        index_select_3 = paddle._C_ops.index_select(data_3, reshape_54, 0)
        del data_3, reshape_54

        # pd_op.reshape: (49x49x6xf32) <- (2401x6xf32, 3xi64)
        reshape_55 = paddle._C_ops.reshape(index_select_3, full_int_array_31)
        del index_select_3

        # pd_op.transpose: (6x49x49xf32) <- (49x49x6xf32)
        transpose_38 = paddle._C_ops.transpose(reshape_55, [2, 0, 1])
        del reshape_55

        # pd_op.unsqueeze: (1x6x49x49xf32) <- (6x49x49xf32, 1xi64)
        unsqueeze_13 = paddle._C_ops.unsqueeze(transpose_38, full_int_array_28)
        del transpose_38

        # pd_op.add: (100x6x49x49xf32) <- (100x6x49x49xf32, 1x6x49x49xf32)
        add_24 = paddle._C_ops.add(matmul_20, unsqueeze_13)
        del matmul_20, unsqueeze_13

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_57 = [1, 100, 6, 49, 49]

        # pd_op.reshape: (1x100x6x49x49xf32) <- (100x6x49x49xf32, 5xi64)
        reshape_56 = paddle._C_ops.reshape(add_24, full_int_array_57)
        del add_24, full_int_array_57

        # pd_op.unsqueeze: (100x1x49x49xf32) <- (100x49x49xf32, 1xi64)
        unsqueeze_14 = paddle._C_ops.unsqueeze(multiply_1, full_int_array_20)
        del multiply_1

        # pd_op.unsqueeze: (1x100x1x49x49xf32) <- (100x1x49x49xf32, 1xi64)
        unsqueeze_15 = paddle._C_ops.unsqueeze(unsqueeze_14, full_int_array_28)
        del unsqueeze_14

        # pd_op.add: (1x100x6x49x49xf32) <- (1x100x6x49x49xf32, 1x100x1x49x49xf32)
        add_25 = paddle._C_ops.add(reshape_56, unsqueeze_15)
        del reshape_56, unsqueeze_15

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_58 = [-1, 6, 49, 49]

        # pd_op.reshape: (100x6x49x49xf32) <- (1x100x6x49x49xf32, 4xi64)
        reshape_57 = paddle._C_ops.reshape(add_25, full_int_array_58)
        del add_25, full_int_array_58

        # pd_op.softmax: (100x6x49x49xf32) <- (100x6x49x49xf32)
        softmax_3 = paddle._C_ops.softmax(reshape_57, -1)
        del reshape_57

        # pd_op.matmul: (100x6x49x32xf32) <- (100x6x49x49xf32, 100x6x49x32xf32)
        matmul_21 = paddle._C_ops.matmul(softmax_3, slice_14, False, False)
        del slice_14, softmax_3

        # pd_op.transpose: (100x49x6x32xf32) <- (100x6x49x32xf32)
        transpose_39 = paddle._C_ops.transpose(matmul_21, [0, 2, 1, 3])
        del matmul_21

        # pd_op.reshape: (100x49x192xf32) <- (100x49x6x32xf32, 3xi64)
        reshape_58 = paddle._C_ops.reshape(transpose_39, full_int_array_52)
        del full_int_array_52, transpose_39

        # pd_op.matmul: (100x49x192xf32) <- (100x49x192xf32, 192x192xf32)
        matmul_22 = paddle._C_ops.matmul(reshape_58, parameter_115, False, False)
        del parameter_115, reshape_58

        # pd_op.add: (100x49x192xf32) <- (100x49x192xf32, 192xf32)
        add_26 = paddle._C_ops.add(matmul_22, parameter_114)
        del matmul_22, parameter_114

        # pd_op.reshape: (100x7x7x192xf32) <- (100x49x192xf32, 4xi64)
        reshape_59 = paddle._C_ops.reshape(add_26, full_int_array_49)
        del add_26, full_int_array_49

        # pd_op.reshape: (1x10x10x7x7x192xf32) <- (100x7x7x192xf32, 6xi64)
        reshape_60 = paddle._C_ops.reshape(reshape_59, full_int_array_53)
        del full_int_array_53, reshape_59

        # pd_op.transpose: (1x10x7x10x7x192xf32) <- (1x10x10x7x7x192xf32)
        transpose_40 = paddle._C_ops.transpose(reshape_60, [0, 1, 3, 2, 4, 5])
        del reshape_60

        # pd_op.reshape: (1x70x70x192xf32) <- (1x10x7x10x7x192xf32, 4xi64)
        reshape_61 = paddle._C_ops.reshape(transpose_40, full_int_array_54)
        del full_int_array_54, transpose_40

        # pd_op.roll: (1x70x70x192xf32) <- (1x70x70x192xf32, 2xi64)
        roll_3 = paddle._C_ops.roll(reshape_61, full_int_array_39, [1, 2])
        del reshape_61

        # pd_op.slice: (1x64x64x192xf32) <- (1x70x70x192xf32, 2xi64, 2xi64)
        slice_15 = paddle._C_ops.slice(
            roll_3, [1, 2], full_int_array_2, full_int_array_55, [1, 1], []
        )
        del full_int_array_55, roll_3

        # pd_op.reshape: (1x4096x192xf32) <- (1x64x64x192xf32, 3xi64)
        reshape_62 = paddle._C_ops.reshape(slice_15, full_int_array_56)
        del full_int_array_56, slice_15

        # pd_op.add: (1x4096x192xf32) <- (1x4096x192xf32, 1x4096x192xf32)
        add_27 = paddle._C_ops.add(add_22, reshape_62)
        del add_22, reshape_62

        # pd_op.layer_norm: (1x4096x192xf32, 1x4096xf32, 1x4096xf32) <- (1x4096x192xf32, 192xf32, 192xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_27, parameter_113, parameter_112, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_112, parameter_113

        # pd_op.matmul: (1x4096x768xf32) <- (1x4096x192xf32, 192x768xf32)
        matmul_23 = paddle._C_ops.matmul(layer_norm_30, parameter_111, False, False)
        del layer_norm_30, parameter_111

        # pd_op.add: (1x4096x768xf32) <- (1x4096x768xf32, 768xf32)
        add_28 = paddle._C_ops.add(matmul_23, parameter_110)
        del matmul_23, parameter_110

        # pd_op.gelu: (1x4096x768xf32) <- (1x4096x768xf32)
        gelu_3 = paddle._C_ops.gelu(add_28, False)
        del add_28

        # pd_op.matmul: (1x4096x192xf32) <- (1x4096x768xf32, 768x192xf32)
        matmul_24 = paddle._C_ops.matmul(gelu_3, parameter_109, False, False)
        del gelu_3, parameter_109

        # pd_op.add: (1x4096x192xf32) <- (1x4096x192xf32, 192xf32)
        add_29 = paddle._C_ops.add(matmul_24, parameter_108)
        del matmul_24, parameter_108

        # pd_op.add: (1x4096x192xf32) <- (1x4096x192xf32, 1x4096x192xf32)
        add_30 = paddle._C_ops.add(add_27, add_29)
        del add_27, add_29

        # pd_op.reshape: (1x64x64x192xf32) <- (1x4096x192xf32, 4xi64)
        reshape_63 = paddle._C_ops.reshape(add_30, full_int_array_46)
        del full_int_array_46

        # pd_op.strided_slice: (1x32x32x192xf32) <- (1x64x64x192xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_4 = paddle._C_ops.strided_slice(
            reshape_63, [1, 2], full_int_array_2, full_int_array_16, full_int_array_40
        )

        # pd_op.strided_slice: (1x32x32x192xf32) <- (1x64x64x192xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_5 = paddle._C_ops.strided_slice(
            reshape_63, [1, 2], full_int_array_41, full_int_array_16, full_int_array_40
        )

        # pd_op.strided_slice: (1x32x32x192xf32) <- (1x64x64x192xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_6 = paddle._C_ops.strided_slice(
            reshape_63, [1, 2], full_int_array_42, full_int_array_16, full_int_array_40
        )

        # pd_op.strided_slice: (1x32x32x192xf32) <- (1x64x64x192xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_7 = paddle._C_ops.strided_slice(
            reshape_63, [1, 2], full_int_array_4, full_int_array_16, full_int_array_40
        )
        del reshape_63

        # builtin.combine: ([1x32x32x192xf32, 1x32x32x192xf32, 1x32x32x192xf32, 1x32x32x192xf32]) <- (1x32x32x192xf32, 1x32x32x192xf32, 1x32x32x192xf32, 1x32x32x192xf32)
        combine_1 = [strided_slice_4, strided_slice_5, strided_slice_6, strided_slice_7]
        del strided_slice_4, strided_slice_5, strided_slice_6, strided_slice_7

        # pd_op.concat: (1x32x32x768xf32) <- ([1x32x32x192xf32, 1x32x32x192xf32, 1x32x32x192xf32, 1x32x32x192xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_5)
        del combine_1

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_59 = [1, -1, 768]

        # pd_op.reshape: (1x1024x768xf32) <- (1x32x32x768xf32, 3xi64)
        reshape_64 = paddle._C_ops.reshape(concat_1, full_int_array_59)
        del concat_1, full_int_array_59

        # pd_op.layer_norm: (1x1024x768xf32, 1x1024xf32, 1x1024xf32) <- (1x1024x768xf32, 768xf32, 768xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                reshape_64, parameter_107, parameter_106, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_106, parameter_107, reshape_64

        # pd_op.matmul: (1x1024x384xf32) <- (1x1024x768xf32, 768x384xf32)
        matmul_25 = paddle._C_ops.matmul(layer_norm_33, parameter_105, False, False)
        del layer_norm_33, parameter_105

        # pd_op.layer_norm: (1x4096x192xf32, 1x4096xf32, 1x4096xf32) <- (1x4096x192xf32, 192xf32, 192xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_30, parameter_104, parameter_103, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_30, parameter_103, parameter_104

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_60 = [-1, 64, 64, 192]

        # pd_op.reshape: (1x64x64x192xf32) <- (1x4096x192xf32, 4xi64)
        reshape_65 = paddle._C_ops.reshape(layer_norm_36, full_int_array_60)
        del full_int_array_60, layer_norm_36

        # pd_op.transpose: (1x192x64x64xf32) <- (1x64x64x192xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_65, [0, 3, 1, 2])
        del reshape_65

        # pd_op.full: (1x35x35x1xf32) <- ()
        full_7 = paddle._C_ops.full(
            [1, 35, 35, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x35x35x1xf32) <- (1x35x35x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__18 = paddle._C_ops.set_value_(
            full_7,
            full_int_array_2,
            full_int_array_3,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("0")],
        )
        del full_7

        # pd_op.set_value_: (1x35x35x1xf32) <- (1x35x35x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__19 = paddle._C_ops.set_value_(
            set_value__18,
            full_int_array_5,
            full_int_array_6,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("1")],
        )
        del set_value__18

        # pd_op.set_value_: (1x35x35x1xf32) <- (1x35x35x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__20 = paddle._C_ops.set_value_(
            set_value__19,
            full_int_array_7,
            full_int_array_8,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("2")],
        )
        del set_value__19

        # pd_op.set_value_: (1x35x35x1xf32) <- (1x35x35x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__21 = paddle._C_ops.set_value_(
            set_value__20,
            full_int_array_9,
            full_int_array_10,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("3")],
        )
        del set_value__20

        # pd_op.set_value_: (1x35x35x1xf32) <- (1x35x35x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__22 = paddle._C_ops.set_value_(
            set_value__21,
            full_int_array_3,
            full_int_array_11,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("4")],
        )
        del set_value__21

        # pd_op.set_value_: (1x35x35x1xf32) <- (1x35x35x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__23 = paddle._C_ops.set_value_(
            set_value__22,
            full_int_array_6,
            full_int_array_12,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("5")],
        )
        del set_value__22

        # pd_op.set_value_: (1x35x35x1xf32) <- (1x35x35x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__24 = paddle._C_ops.set_value_(
            set_value__23,
            full_int_array_13,
            full_int_array_14,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("6")],
        )
        del set_value__23

        # pd_op.set_value_: (1x35x35x1xf32) <- (1x35x35x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__25 = paddle._C_ops.set_value_(
            set_value__24,
            full_int_array_10,
            full_int_array_15,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("7")],
        )
        del set_value__24

        # pd_op.set_value_: (1x35x35x1xf32) <- (1x35x35x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__26 = paddle._C_ops.set_value_(
            set_value__25,
            full_int_array_11,
            full_int_array_16,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("8")],
        )
        del set_value__25

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_61 = [1, 5, 7, 5, 7, 1]

        # pd_op.reshape: (1x5x7x5x7x1xf32) <- (1x35x35x1xf32, 6xi64)
        reshape_66 = paddle._C_ops.reshape(set_value__26, full_int_array_61)
        del full_int_array_61

        # pd_op.transpose: (1x5x5x7x7x1xf32) <- (1x5x7x5x7x1xf32)
        transpose_41 = paddle._C_ops.transpose(reshape_66, [0, 1, 3, 2, 4, 5])
        del reshape_66

        # pd_op.reshape: (25x7x7x1xf32) <- (1x5x5x7x7x1xf32, 4xi64)
        reshape_67 = paddle._C_ops.reshape(transpose_41, full_int_array_18)
        del transpose_41

        # pd_op.reshape: (25x49xf32) <- (25x7x7x1xf32, 2xi64)
        reshape_68 = paddle._C_ops.reshape(reshape_67, full_int_array_19)
        del reshape_67

        # pd_op.unsqueeze: (25x1x49xf32) <- (25x49xf32, 1xi64)
        unsqueeze_16 = paddle._C_ops.unsqueeze(reshape_68, full_int_array_20)

        # pd_op.unsqueeze: (25x49x1xf32) <- (25x49xf32, 1xi64)
        unsqueeze_17 = paddle._C_ops.unsqueeze(reshape_68, full_int_array_21)
        del reshape_68

        # pd_op.subtract: (25x49x49xf32) <- (25x1x49xf32, 25x49x1xf32)
        subtract_2 = paddle._C_ops.subtract(unsqueeze_16, unsqueeze_17)
        del unsqueeze_16, unsqueeze_17

        # pd_op.full_like: (25x49x49xf32) <- (25x49x49xf32, 1xf32)
        full_like_2 = paddle._C_ops.full_like(
            subtract_2,
            full_1,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.scale: (25x49x49xf32) <- (25x49x49xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(full_like_2, full_2, float("0"), True)
        del full_like_2

        # pd_op.not_equal: (25x49x49xb) <- (25x49x49xf32, xf32)
        not_equal_2 = paddle._C_ops.not_equal(subtract_2, full_3)
        del subtract_2

        # pd_op.cast: (25x49x49xf32) <- (25x49x49xb)
        cast_2 = paddle._C_ops.cast(not_equal_2, paddle.float32)
        del not_equal_2

        # pd_op.multiply: (25x49x49xf32) <- (25x49x49xf32, 25x49x49xf32)
        multiply_2 = paddle._C_ops.multiply(scale_6, cast_2)
        del cast_2, scale_6

        # pd_op.layer_norm: (1x1024x384xf32, 1x1024xf32, 1x1024xf32) <- (1x1024x384xf32, 384xf32, 384xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                matmul_25, parameter_102, parameter_101, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_101, parameter_102

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_62 = [1, 32, 32, 384]

        # pd_op.reshape: (1x32x32x384xf32) <- (1x1024x384xf32, 4xi64)
        reshape_69 = paddle._C_ops.reshape(layer_norm_39, full_int_array_62)
        del layer_norm_39

        # pd_op.transpose: (1x384x32x32xf32) <- (1x32x32x384xf32)
        transpose_42 = paddle._C_ops.transpose(reshape_69, [0, 3, 1, 2])
        del reshape_69

        # pd_op.unsqueeze: (1x384x1x32x32xf32) <- (1x384x32x32xf32, 1xi64)
        unsqueeze_18 = paddle._C_ops.unsqueeze(transpose_42, full_int_array_21)
        del transpose_42

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_63 = [0, 3, 0, 3, 0, 0]

        # pd_op.pad3d: (1x384x1x35x35xf32) <- (1x384x1x32x32xf32, 6xi64)
        pad3d_4 = paddle._C_ops.pad3d(
            unsqueeze_18, full_int_array_63, "constant", float("0"), "NCDHW"
        )
        del unsqueeze_18

        # pd_op.squeeze: (1x384x35x35xf32) <- (1x384x1x35x35xf32, 1xi64)
        squeeze_4 = paddle._C_ops.squeeze(pad3d_4, full_int_array_21)
        del pad3d_4

        # pd_op.transpose: (1x35x35x384xf32) <- (1x384x35x35xf32)
        transpose_43 = paddle._C_ops.transpose(squeeze_4, [0, 2, 3, 1])
        del squeeze_4

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_64 = [1, 5, 7, 5, 7, 384]

        # pd_op.reshape: (1x5x7x5x7x384xf32) <- (1x35x35x384xf32, 6xi64)
        reshape_70 = paddle._C_ops.reshape(transpose_43, full_int_array_64)
        del transpose_43

        # pd_op.transpose: (1x5x5x7x7x384xf32) <- (1x5x7x5x7x384xf32)
        transpose_44 = paddle._C_ops.transpose(reshape_70, [0, 1, 3, 2, 4, 5])
        del reshape_70

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_65 = [-1, 7, 7, 384]

        # pd_op.reshape: (25x7x7x384xf32) <- (1x5x5x7x7x384xf32, 4xi64)
        reshape_71 = paddle._C_ops.reshape(transpose_44, full_int_array_65)
        del transpose_44

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_66 = [-1, 49, 384]

        # pd_op.reshape: (25x49x384xf32) <- (25x7x7x384xf32, 3xi64)
        reshape_72 = paddle._C_ops.reshape(reshape_71, full_int_array_66)
        del reshape_71

        # pd_op.matmul: (25x49x1152xf32) <- (25x49x384xf32, 384x1152xf32)
        matmul_26 = paddle._C_ops.matmul(reshape_72, parameter_100, False, False)
        del parameter_100, reshape_72

        # pd_op.add: (25x49x1152xf32) <- (25x49x1152xf32, 1152xf32)
        add_31 = paddle._C_ops.add(matmul_26, parameter_99)
        del matmul_26, parameter_99

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_67 = [25, 49, 3, 12, 32]

        # pd_op.reshape: (25x49x3x12x32xf32) <- (25x49x1152xf32, 5xi64)
        reshape_73 = paddle._C_ops.reshape(add_31, full_int_array_67)
        del add_31

        # pd_op.transpose: (3x25x12x49x32xf32) <- (25x49x3x12x32xf32)
        transpose_45 = paddle._C_ops.transpose(reshape_73, [2, 0, 3, 1, 4])
        del reshape_73

        # pd_op.slice: (25x12x49x32xf32) <- (3x25x12x49x32xf32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            transpose_45, [0], full_int_array_28, full_int_array_20, [1], [0]
        )

        # pd_op.slice: (25x12x49x32xf32) <- (3x25x12x49x32xf32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            transpose_45, [0], full_int_array_20, full_int_array_21, [1], [0]
        )

        # pd_op.slice: (25x12x49x32xf32) <- (3x25x12x49x32xf32, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(
            transpose_45, [0], full_int_array_21, full_int_array_29, [1], [0]
        )
        del transpose_45

        # pd_op.scale: (25x12x49x32xf32) <- (25x12x49x32xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(slice_16, full_4, float("0"), True)
        del slice_16

        # pd_op.transpose: (25x12x32x49xf32) <- (25x12x49x32xf32)
        transpose_46 = paddle._C_ops.transpose(slice_17, [0, 1, 3, 2])
        del slice_17

        # pd_op.matmul: (25x12x49x49xf32) <- (25x12x49x32xf32, 25x12x32x49xf32)
        matmul_27 = paddle._C_ops.matmul(scale_7, transpose_46, False, False)
        del scale_7, transpose_46

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        reshape_74 = paddle._C_ops.reshape(data_17, full_int_array_30)
        del data_17

        # pd_op.index_select: (2401x12xf32) <- (169x12xf32, 2401xi64)
        index_select_4 = paddle._C_ops.index_select(data_4, reshape_74, 0)
        del data_4, reshape_74

        # pd_op.reshape: (49x49x12xf32) <- (2401x12xf32, 3xi64)
        reshape_75 = paddle._C_ops.reshape(index_select_4, full_int_array_31)
        del index_select_4

        # pd_op.transpose: (12x49x49xf32) <- (49x49x12xf32)
        transpose_47 = paddle._C_ops.transpose(reshape_75, [2, 0, 1])
        del reshape_75

        # pd_op.unsqueeze: (1x12x49x49xf32) <- (12x49x49xf32, 1xi64)
        unsqueeze_19 = paddle._C_ops.unsqueeze(transpose_47, full_int_array_28)
        del transpose_47

        # pd_op.add: (25x12x49x49xf32) <- (25x12x49x49xf32, 1x12x49x49xf32)
        add_32 = paddle._C_ops.add(matmul_27, unsqueeze_19)
        del matmul_27, unsqueeze_19

        # pd_op.softmax: (25x12x49x49xf32) <- (25x12x49x49xf32)
        softmax_4 = paddle._C_ops.softmax(add_32, -1)
        del add_32

        # pd_op.matmul: (25x12x49x32xf32) <- (25x12x49x49xf32, 25x12x49x32xf32)
        matmul_28 = paddle._C_ops.matmul(softmax_4, slice_18, False, False)
        del slice_18, softmax_4

        # pd_op.transpose: (25x49x12x32xf32) <- (25x12x49x32xf32)
        transpose_48 = paddle._C_ops.transpose(matmul_28, [0, 2, 1, 3])
        del matmul_28

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_68 = [25, 49, 384]

        # pd_op.reshape: (25x49x384xf32) <- (25x49x12x32xf32, 3xi64)
        reshape_76 = paddle._C_ops.reshape(transpose_48, full_int_array_68)
        del transpose_48

        # pd_op.matmul: (25x49x384xf32) <- (25x49x384xf32, 384x384xf32)
        matmul_29 = paddle._C_ops.matmul(reshape_76, parameter_98, False, False)
        del parameter_98, reshape_76

        # pd_op.add: (25x49x384xf32) <- (25x49x384xf32, 384xf32)
        add_33 = paddle._C_ops.add(matmul_29, parameter_97)
        del matmul_29, parameter_97

        # pd_op.reshape: (25x7x7x384xf32) <- (25x49x384xf32, 4xi64)
        reshape_77 = paddle._C_ops.reshape(add_33, full_int_array_65)
        del add_33

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_69 = [1, 5, 5, 7, 7, -1]

        # pd_op.reshape: (1x5x5x7x7x384xf32) <- (25x7x7x384xf32, 6xi64)
        reshape_78 = paddle._C_ops.reshape(reshape_77, full_int_array_69)
        del reshape_77

        # pd_op.transpose: (1x5x7x5x7x384xf32) <- (1x5x5x7x7x384xf32)
        transpose_49 = paddle._C_ops.transpose(reshape_78, [0, 1, 3, 2, 4, 5])
        del reshape_78

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_70 = [1, 35, 35, -1]

        # pd_op.reshape: (1x35x35x384xf32) <- (1x5x7x5x7x384xf32, 4xi64)
        reshape_79 = paddle._C_ops.reshape(transpose_49, full_int_array_70)
        del transpose_49

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_71 = [32, 32]

        # pd_op.slice: (1x32x32x384xf32) <- (1x35x35x384xf32, 2xi64, 2xi64)
        slice_19 = paddle._C_ops.slice(
            reshape_79, [1, 2], full_int_array_2, full_int_array_71, [1, 1], []
        )
        del reshape_79

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_72 = [1, 1024, 384]

        # pd_op.reshape: (1x1024x384xf32) <- (1x32x32x384xf32, 3xi64)
        reshape_80 = paddle._C_ops.reshape(slice_19, full_int_array_72)
        del slice_19

        # pd_op.add: (1x1024x384xf32) <- (1x1024x384xf32, 1x1024x384xf32)
        add_34 = paddle._C_ops.add(matmul_25, reshape_80)
        del matmul_25, reshape_80

        # pd_op.layer_norm: (1x1024x384xf32, 1x1024xf32, 1x1024xf32) <- (1x1024x384xf32, 384xf32, 384xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_34, parameter_96, parameter_95, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_95, parameter_96

        # pd_op.matmul: (1x1024x1536xf32) <- (1x1024x384xf32, 384x1536xf32)
        matmul_30 = paddle._C_ops.matmul(layer_norm_42, parameter_94, False, False)
        del layer_norm_42, parameter_94

        # pd_op.add: (1x1024x1536xf32) <- (1x1024x1536xf32, 1536xf32)
        add_35 = paddle._C_ops.add(matmul_30, parameter_93)
        del matmul_30, parameter_93

        # pd_op.gelu: (1x1024x1536xf32) <- (1x1024x1536xf32)
        gelu_4 = paddle._C_ops.gelu(add_35, False)
        del add_35

        # pd_op.matmul: (1x1024x384xf32) <- (1x1024x1536xf32, 1536x384xf32)
        matmul_31 = paddle._C_ops.matmul(gelu_4, parameter_92, False, False)
        del gelu_4, parameter_92

        # pd_op.add: (1x1024x384xf32) <- (1x1024x384xf32, 384xf32)
        add_36 = paddle._C_ops.add(matmul_31, parameter_91)
        del matmul_31, parameter_91

        # pd_op.add: (1x1024x384xf32) <- (1x1024x384xf32, 1x1024x384xf32)
        add_37 = paddle._C_ops.add(add_34, add_36)
        del add_34, add_36

        # pd_op.layer_norm: (1x1024x384xf32, 1x1024xf32, 1x1024xf32) <- (1x1024x384xf32, 384xf32, 384xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_37, parameter_90, parameter_89, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_89, parameter_90

        # pd_op.reshape: (1x32x32x384xf32) <- (1x1024x384xf32, 4xi64)
        reshape_81 = paddle._C_ops.reshape(layer_norm_45, full_int_array_62)
        del layer_norm_45

        # pd_op.transpose: (1x384x32x32xf32) <- (1x32x32x384xf32)
        transpose_50 = paddle._C_ops.transpose(reshape_81, [0, 3, 1, 2])
        del reshape_81

        # pd_op.unsqueeze: (1x384x1x32x32xf32) <- (1x384x32x32xf32, 1xi64)
        unsqueeze_20 = paddle._C_ops.unsqueeze(transpose_50, full_int_array_21)
        del transpose_50

        # pd_op.pad3d: (1x384x1x35x35xf32) <- (1x384x1x32x32xf32, 6xi64)
        pad3d_5 = paddle._C_ops.pad3d(
            unsqueeze_20, full_int_array_63, "constant", float("0"), "NCDHW"
        )
        del unsqueeze_20

        # pd_op.squeeze: (1x384x35x35xf32) <- (1x384x1x35x35xf32, 1xi64)
        squeeze_5 = paddle._C_ops.squeeze(pad3d_5, full_int_array_21)
        del pad3d_5

        # pd_op.transpose: (1x35x35x384xf32) <- (1x384x35x35xf32)
        transpose_51 = paddle._C_ops.transpose(squeeze_5, [0, 2, 3, 1])
        del squeeze_5

        # pd_op.roll: (1x35x35x384xf32) <- (1x35x35x384xf32, 2xi64)
        roll_4 = paddle._C_ops.roll(transpose_51, full_int_array_11, [1, 2])
        del transpose_51

        # pd_op.reshape: (1x5x7x5x7x384xf32) <- (1x35x35x384xf32, 6xi64)
        reshape_82 = paddle._C_ops.reshape(roll_4, full_int_array_64)
        del roll_4

        # pd_op.transpose: (1x5x5x7x7x384xf32) <- (1x5x7x5x7x384xf32)
        transpose_52 = paddle._C_ops.transpose(reshape_82, [0, 1, 3, 2, 4, 5])
        del reshape_82

        # pd_op.reshape: (25x7x7x384xf32) <- (1x5x5x7x7x384xf32, 4xi64)
        reshape_83 = paddle._C_ops.reshape(transpose_52, full_int_array_65)
        del transpose_52

        # pd_op.reshape: (25x49x384xf32) <- (25x7x7x384xf32, 3xi64)
        reshape_84 = paddle._C_ops.reshape(reshape_83, full_int_array_66)
        del reshape_83

        # pd_op.matmul: (25x49x1152xf32) <- (25x49x384xf32, 384x1152xf32)
        matmul_32 = paddle._C_ops.matmul(reshape_84, parameter_88, False, False)
        del parameter_88, reshape_84

        # pd_op.add: (25x49x1152xf32) <- (25x49x1152xf32, 1152xf32)
        add_38 = paddle._C_ops.add(matmul_32, parameter_87)
        del matmul_32, parameter_87

        # pd_op.reshape: (25x49x3x12x32xf32) <- (25x49x1152xf32, 5xi64)
        reshape_85 = paddle._C_ops.reshape(add_38, full_int_array_67)
        del add_38

        # pd_op.transpose: (3x25x12x49x32xf32) <- (25x49x3x12x32xf32)
        transpose_53 = paddle._C_ops.transpose(reshape_85, [2, 0, 3, 1, 4])
        del reshape_85

        # pd_op.slice: (25x12x49x32xf32) <- (3x25x12x49x32xf32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(
            transpose_53, [0], full_int_array_28, full_int_array_20, [1], [0]
        )

        # pd_op.slice: (25x12x49x32xf32) <- (3x25x12x49x32xf32, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(
            transpose_53, [0], full_int_array_20, full_int_array_21, [1], [0]
        )

        # pd_op.slice: (25x12x49x32xf32) <- (3x25x12x49x32xf32, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(
            transpose_53, [0], full_int_array_21, full_int_array_29, [1], [0]
        )
        del transpose_53

        # pd_op.scale: (25x12x49x32xf32) <- (25x12x49x32xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(slice_20, full_4, float("0"), True)
        del slice_20

        # pd_op.transpose: (25x12x32x49xf32) <- (25x12x49x32xf32)
        transpose_54 = paddle._C_ops.transpose(slice_21, [0, 1, 3, 2])
        del slice_21

        # pd_op.matmul: (25x12x49x49xf32) <- (25x12x49x32xf32, 25x12x32x49xf32)
        matmul_33 = paddle._C_ops.matmul(scale_8, transpose_54, False, False)
        del scale_8, transpose_54

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        reshape_86 = paddle._C_ops.reshape(data_18, full_int_array_30)
        del data_18

        # pd_op.index_select: (2401x12xf32) <- (169x12xf32, 2401xi64)
        index_select_5 = paddle._C_ops.index_select(data_5, reshape_86, 0)
        del data_5, reshape_86

        # pd_op.reshape: (49x49x12xf32) <- (2401x12xf32, 3xi64)
        reshape_87 = paddle._C_ops.reshape(index_select_5, full_int_array_31)
        del index_select_5

        # pd_op.transpose: (12x49x49xf32) <- (49x49x12xf32)
        transpose_55 = paddle._C_ops.transpose(reshape_87, [2, 0, 1])
        del reshape_87

        # pd_op.unsqueeze: (1x12x49x49xf32) <- (12x49x49xf32, 1xi64)
        unsqueeze_21 = paddle._C_ops.unsqueeze(transpose_55, full_int_array_28)
        del transpose_55

        # pd_op.add: (25x12x49x49xf32) <- (25x12x49x49xf32, 1x12x49x49xf32)
        add_39 = paddle._C_ops.add(matmul_33, unsqueeze_21)
        del matmul_33, unsqueeze_21

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_73 = [1, 25, 12, 49, 49]

        # pd_op.reshape: (1x25x12x49x49xf32) <- (25x12x49x49xf32, 5xi64)
        reshape_88 = paddle._C_ops.reshape(add_39, full_int_array_73)
        del add_39

        # pd_op.unsqueeze: (25x1x49x49xf32) <- (25x49x49xf32, 1xi64)
        unsqueeze_22 = paddle._C_ops.unsqueeze(multiply_2, full_int_array_20)

        # pd_op.unsqueeze: (1x25x1x49x49xf32) <- (25x1x49x49xf32, 1xi64)
        unsqueeze_23 = paddle._C_ops.unsqueeze(unsqueeze_22, full_int_array_28)
        del unsqueeze_22

        # pd_op.add: (1x25x12x49x49xf32) <- (1x25x12x49x49xf32, 1x25x1x49x49xf32)
        add_40 = paddle._C_ops.add(reshape_88, unsqueeze_23)
        del reshape_88, unsqueeze_23

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_74 = [-1, 12, 49, 49]

        # pd_op.reshape: (25x12x49x49xf32) <- (1x25x12x49x49xf32, 4xi64)
        reshape_89 = paddle._C_ops.reshape(add_40, full_int_array_74)
        del add_40

        # pd_op.softmax: (25x12x49x49xf32) <- (25x12x49x49xf32)
        softmax_5 = paddle._C_ops.softmax(reshape_89, -1)
        del reshape_89

        # pd_op.matmul: (25x12x49x32xf32) <- (25x12x49x49xf32, 25x12x49x32xf32)
        matmul_34 = paddle._C_ops.matmul(softmax_5, slice_22, False, False)
        del slice_22, softmax_5

        # pd_op.transpose: (25x49x12x32xf32) <- (25x12x49x32xf32)
        transpose_56 = paddle._C_ops.transpose(matmul_34, [0, 2, 1, 3])
        del matmul_34

        # pd_op.reshape: (25x49x384xf32) <- (25x49x12x32xf32, 3xi64)
        reshape_90 = paddle._C_ops.reshape(transpose_56, full_int_array_68)
        del transpose_56

        # pd_op.matmul: (25x49x384xf32) <- (25x49x384xf32, 384x384xf32)
        matmul_35 = paddle._C_ops.matmul(reshape_90, parameter_86, False, False)
        del parameter_86, reshape_90

        # pd_op.add: (25x49x384xf32) <- (25x49x384xf32, 384xf32)
        add_41 = paddle._C_ops.add(matmul_35, parameter_85)
        del matmul_35, parameter_85

        # pd_op.reshape: (25x7x7x384xf32) <- (25x49x384xf32, 4xi64)
        reshape_91 = paddle._C_ops.reshape(add_41, full_int_array_65)
        del add_41

        # pd_op.reshape: (1x5x5x7x7x384xf32) <- (25x7x7x384xf32, 6xi64)
        reshape_92 = paddle._C_ops.reshape(reshape_91, full_int_array_69)
        del reshape_91

        # pd_op.transpose: (1x5x7x5x7x384xf32) <- (1x5x5x7x7x384xf32)
        transpose_57 = paddle._C_ops.transpose(reshape_92, [0, 1, 3, 2, 4, 5])
        del reshape_92

        # pd_op.reshape: (1x35x35x384xf32) <- (1x5x7x5x7x384xf32, 4xi64)
        reshape_93 = paddle._C_ops.reshape(transpose_57, full_int_array_70)
        del transpose_57

        # pd_op.roll: (1x35x35x384xf32) <- (1x35x35x384xf32, 2xi64)
        roll_5 = paddle._C_ops.roll(reshape_93, full_int_array_39, [1, 2])
        del reshape_93

        # pd_op.slice: (1x32x32x384xf32) <- (1x35x35x384xf32, 2xi64, 2xi64)
        slice_23 = paddle._C_ops.slice(
            roll_5, [1, 2], full_int_array_2, full_int_array_71, [1, 1], []
        )
        del roll_5

        # pd_op.reshape: (1x1024x384xf32) <- (1x32x32x384xf32, 3xi64)
        reshape_94 = paddle._C_ops.reshape(slice_23, full_int_array_72)
        del slice_23

        # pd_op.add: (1x1024x384xf32) <- (1x1024x384xf32, 1x1024x384xf32)
        add_42 = paddle._C_ops.add(add_37, reshape_94)
        del add_37, reshape_94

        # pd_op.layer_norm: (1x1024x384xf32, 1x1024xf32, 1x1024xf32) <- (1x1024x384xf32, 384xf32, 384xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_42, parameter_84, parameter_83, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_83, parameter_84

        # pd_op.matmul: (1x1024x1536xf32) <- (1x1024x384xf32, 384x1536xf32)
        matmul_36 = paddle._C_ops.matmul(layer_norm_48, parameter_82, False, False)
        del layer_norm_48, parameter_82

        # pd_op.add: (1x1024x1536xf32) <- (1x1024x1536xf32, 1536xf32)
        add_43 = paddle._C_ops.add(matmul_36, parameter_81)
        del matmul_36, parameter_81

        # pd_op.gelu: (1x1024x1536xf32) <- (1x1024x1536xf32)
        gelu_5 = paddle._C_ops.gelu(add_43, False)
        del add_43

        # pd_op.matmul: (1x1024x384xf32) <- (1x1024x1536xf32, 1536x384xf32)
        matmul_37 = paddle._C_ops.matmul(gelu_5, parameter_80, False, False)
        del gelu_5, parameter_80

        # pd_op.add: (1x1024x384xf32) <- (1x1024x384xf32, 384xf32)
        add_44 = paddle._C_ops.add(matmul_37, parameter_79)
        del matmul_37, parameter_79

        # pd_op.add: (1x1024x384xf32) <- (1x1024x384xf32, 1x1024x384xf32)
        add_45 = paddle._C_ops.add(add_42, add_44)
        del add_42, add_44

        # pd_op.layer_norm: (1x1024x384xf32, 1x1024xf32, 1x1024xf32) <- (1x1024x384xf32, 384xf32, 384xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_45, parameter_78, parameter_77, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_77, parameter_78

        # pd_op.reshape: (1x32x32x384xf32) <- (1x1024x384xf32, 4xi64)
        reshape_95 = paddle._C_ops.reshape(layer_norm_51, full_int_array_62)
        del layer_norm_51

        # pd_op.transpose: (1x384x32x32xf32) <- (1x32x32x384xf32)
        transpose_58 = paddle._C_ops.transpose(reshape_95, [0, 3, 1, 2])
        del reshape_95

        # pd_op.unsqueeze: (1x384x1x32x32xf32) <- (1x384x32x32xf32, 1xi64)
        unsqueeze_24 = paddle._C_ops.unsqueeze(transpose_58, full_int_array_21)
        del transpose_58

        # pd_op.pad3d: (1x384x1x35x35xf32) <- (1x384x1x32x32xf32, 6xi64)
        pad3d_6 = paddle._C_ops.pad3d(
            unsqueeze_24, full_int_array_63, "constant", float("0"), "NCDHW"
        )
        del unsqueeze_24

        # pd_op.squeeze: (1x384x35x35xf32) <- (1x384x1x35x35xf32, 1xi64)
        squeeze_6 = paddle._C_ops.squeeze(pad3d_6, full_int_array_21)
        del pad3d_6

        # pd_op.transpose: (1x35x35x384xf32) <- (1x384x35x35xf32)
        transpose_59 = paddle._C_ops.transpose(squeeze_6, [0, 2, 3, 1])
        del squeeze_6

        # pd_op.reshape: (1x5x7x5x7x384xf32) <- (1x35x35x384xf32, 6xi64)
        reshape_96 = paddle._C_ops.reshape(transpose_59, full_int_array_64)
        del transpose_59

        # pd_op.transpose: (1x5x5x7x7x384xf32) <- (1x5x7x5x7x384xf32)
        transpose_60 = paddle._C_ops.transpose(reshape_96, [0, 1, 3, 2, 4, 5])
        del reshape_96

        # pd_op.reshape: (25x7x7x384xf32) <- (1x5x5x7x7x384xf32, 4xi64)
        reshape_97 = paddle._C_ops.reshape(transpose_60, full_int_array_65)
        del transpose_60

        # pd_op.reshape: (25x49x384xf32) <- (25x7x7x384xf32, 3xi64)
        reshape_98 = paddle._C_ops.reshape(reshape_97, full_int_array_66)
        del reshape_97

        # pd_op.matmul: (25x49x1152xf32) <- (25x49x384xf32, 384x1152xf32)
        matmul_38 = paddle._C_ops.matmul(reshape_98, parameter_76, False, False)
        del parameter_76, reshape_98

        # pd_op.add: (25x49x1152xf32) <- (25x49x1152xf32, 1152xf32)
        add_46 = paddle._C_ops.add(matmul_38, parameter_75)
        del matmul_38, parameter_75

        # pd_op.reshape: (25x49x3x12x32xf32) <- (25x49x1152xf32, 5xi64)
        reshape_99 = paddle._C_ops.reshape(add_46, full_int_array_67)
        del add_46

        # pd_op.transpose: (3x25x12x49x32xf32) <- (25x49x3x12x32xf32)
        transpose_61 = paddle._C_ops.transpose(reshape_99, [2, 0, 3, 1, 4])
        del reshape_99

        # pd_op.slice: (25x12x49x32xf32) <- (3x25x12x49x32xf32, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(
            transpose_61, [0], full_int_array_28, full_int_array_20, [1], [0]
        )

        # pd_op.slice: (25x12x49x32xf32) <- (3x25x12x49x32xf32, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(
            transpose_61, [0], full_int_array_20, full_int_array_21, [1], [0]
        )

        # pd_op.slice: (25x12x49x32xf32) <- (3x25x12x49x32xf32, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(
            transpose_61, [0], full_int_array_21, full_int_array_29, [1], [0]
        )
        del transpose_61

        # pd_op.scale: (25x12x49x32xf32) <- (25x12x49x32xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(slice_24, full_4, float("0"), True)
        del slice_24

        # pd_op.transpose: (25x12x32x49xf32) <- (25x12x49x32xf32)
        transpose_62 = paddle._C_ops.transpose(slice_25, [0, 1, 3, 2])
        del slice_25

        # pd_op.matmul: (25x12x49x49xf32) <- (25x12x49x32xf32, 25x12x32x49xf32)
        matmul_39 = paddle._C_ops.matmul(scale_9, transpose_62, False, False)
        del scale_9, transpose_62

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        reshape_100 = paddle._C_ops.reshape(data_19, full_int_array_30)
        del data_19

        # pd_op.index_select: (2401x12xf32) <- (169x12xf32, 2401xi64)
        index_select_6 = paddle._C_ops.index_select(data_6, reshape_100, 0)
        del data_6, reshape_100

        # pd_op.reshape: (49x49x12xf32) <- (2401x12xf32, 3xi64)
        reshape_101 = paddle._C_ops.reshape(index_select_6, full_int_array_31)
        del index_select_6

        # pd_op.transpose: (12x49x49xf32) <- (49x49x12xf32)
        transpose_63 = paddle._C_ops.transpose(reshape_101, [2, 0, 1])
        del reshape_101

        # pd_op.unsqueeze: (1x12x49x49xf32) <- (12x49x49xf32, 1xi64)
        unsqueeze_25 = paddle._C_ops.unsqueeze(transpose_63, full_int_array_28)
        del transpose_63

        # pd_op.add: (25x12x49x49xf32) <- (25x12x49x49xf32, 1x12x49x49xf32)
        add_47 = paddle._C_ops.add(matmul_39, unsqueeze_25)
        del matmul_39, unsqueeze_25

        # pd_op.softmax: (25x12x49x49xf32) <- (25x12x49x49xf32)
        softmax_6 = paddle._C_ops.softmax(add_47, -1)
        del add_47

        # pd_op.matmul: (25x12x49x32xf32) <- (25x12x49x49xf32, 25x12x49x32xf32)
        matmul_40 = paddle._C_ops.matmul(softmax_6, slice_26, False, False)
        del slice_26, softmax_6

        # pd_op.transpose: (25x49x12x32xf32) <- (25x12x49x32xf32)
        transpose_64 = paddle._C_ops.transpose(matmul_40, [0, 2, 1, 3])
        del matmul_40

        # pd_op.reshape: (25x49x384xf32) <- (25x49x12x32xf32, 3xi64)
        reshape_102 = paddle._C_ops.reshape(transpose_64, full_int_array_68)
        del transpose_64

        # pd_op.matmul: (25x49x384xf32) <- (25x49x384xf32, 384x384xf32)
        matmul_41 = paddle._C_ops.matmul(reshape_102, parameter_74, False, False)
        del parameter_74, reshape_102

        # pd_op.add: (25x49x384xf32) <- (25x49x384xf32, 384xf32)
        add_48 = paddle._C_ops.add(matmul_41, parameter_73)
        del matmul_41, parameter_73

        # pd_op.reshape: (25x7x7x384xf32) <- (25x49x384xf32, 4xi64)
        reshape_103 = paddle._C_ops.reshape(add_48, full_int_array_65)
        del add_48

        # pd_op.reshape: (1x5x5x7x7x384xf32) <- (25x7x7x384xf32, 6xi64)
        reshape_104 = paddle._C_ops.reshape(reshape_103, full_int_array_69)
        del reshape_103

        # pd_op.transpose: (1x5x7x5x7x384xf32) <- (1x5x5x7x7x384xf32)
        transpose_65 = paddle._C_ops.transpose(reshape_104, [0, 1, 3, 2, 4, 5])
        del reshape_104

        # pd_op.reshape: (1x35x35x384xf32) <- (1x5x7x5x7x384xf32, 4xi64)
        reshape_105 = paddle._C_ops.reshape(transpose_65, full_int_array_70)
        del transpose_65

        # pd_op.slice: (1x32x32x384xf32) <- (1x35x35x384xf32, 2xi64, 2xi64)
        slice_27 = paddle._C_ops.slice(
            reshape_105, [1, 2], full_int_array_2, full_int_array_71, [1, 1], []
        )
        del reshape_105

        # pd_op.reshape: (1x1024x384xf32) <- (1x32x32x384xf32, 3xi64)
        reshape_106 = paddle._C_ops.reshape(slice_27, full_int_array_72)
        del slice_27

        # pd_op.add: (1x1024x384xf32) <- (1x1024x384xf32, 1x1024x384xf32)
        add_49 = paddle._C_ops.add(add_45, reshape_106)
        del add_45, reshape_106

        # pd_op.layer_norm: (1x1024x384xf32, 1x1024xf32, 1x1024xf32) <- (1x1024x384xf32, 384xf32, 384xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_49, parameter_72, parameter_71, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_71, parameter_72

        # pd_op.matmul: (1x1024x1536xf32) <- (1x1024x384xf32, 384x1536xf32)
        matmul_42 = paddle._C_ops.matmul(layer_norm_54, parameter_70, False, False)
        del layer_norm_54, parameter_70

        # pd_op.add: (1x1024x1536xf32) <- (1x1024x1536xf32, 1536xf32)
        add_50 = paddle._C_ops.add(matmul_42, parameter_69)
        del matmul_42, parameter_69

        # pd_op.gelu: (1x1024x1536xf32) <- (1x1024x1536xf32)
        gelu_6 = paddle._C_ops.gelu(add_50, False)
        del add_50

        # pd_op.matmul: (1x1024x384xf32) <- (1x1024x1536xf32, 1536x384xf32)
        matmul_43 = paddle._C_ops.matmul(gelu_6, parameter_68, False, False)
        del gelu_6, parameter_68

        # pd_op.add: (1x1024x384xf32) <- (1x1024x384xf32, 384xf32)
        add_51 = paddle._C_ops.add(matmul_43, parameter_67)
        del matmul_43, parameter_67

        # pd_op.add: (1x1024x384xf32) <- (1x1024x384xf32, 1x1024x384xf32)
        add_52 = paddle._C_ops.add(add_49, add_51)
        del add_49, add_51

        # pd_op.layer_norm: (1x1024x384xf32, 1x1024xf32, 1x1024xf32) <- (1x1024x384xf32, 384xf32, 384xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_52, parameter_66, parameter_65, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_65, parameter_66

        # pd_op.reshape: (1x32x32x384xf32) <- (1x1024x384xf32, 4xi64)
        reshape_107 = paddle._C_ops.reshape(layer_norm_57, full_int_array_62)
        del layer_norm_57

        # pd_op.transpose: (1x384x32x32xf32) <- (1x32x32x384xf32)
        transpose_66 = paddle._C_ops.transpose(reshape_107, [0, 3, 1, 2])
        del reshape_107

        # pd_op.unsqueeze: (1x384x1x32x32xf32) <- (1x384x32x32xf32, 1xi64)
        unsqueeze_26 = paddle._C_ops.unsqueeze(transpose_66, full_int_array_21)
        del transpose_66

        # pd_op.pad3d: (1x384x1x35x35xf32) <- (1x384x1x32x32xf32, 6xi64)
        pad3d_7 = paddle._C_ops.pad3d(
            unsqueeze_26, full_int_array_63, "constant", float("0"), "NCDHW"
        )
        del unsqueeze_26

        # pd_op.squeeze: (1x384x35x35xf32) <- (1x384x1x35x35xf32, 1xi64)
        squeeze_7 = paddle._C_ops.squeeze(pad3d_7, full_int_array_21)
        del pad3d_7

        # pd_op.transpose: (1x35x35x384xf32) <- (1x384x35x35xf32)
        transpose_67 = paddle._C_ops.transpose(squeeze_7, [0, 2, 3, 1])
        del squeeze_7

        # pd_op.roll: (1x35x35x384xf32) <- (1x35x35x384xf32, 2xi64)
        roll_6 = paddle._C_ops.roll(transpose_67, full_int_array_11, [1, 2])
        del transpose_67

        # pd_op.reshape: (1x5x7x5x7x384xf32) <- (1x35x35x384xf32, 6xi64)
        reshape_108 = paddle._C_ops.reshape(roll_6, full_int_array_64)
        del roll_6

        # pd_op.transpose: (1x5x5x7x7x384xf32) <- (1x5x7x5x7x384xf32)
        transpose_68 = paddle._C_ops.transpose(reshape_108, [0, 1, 3, 2, 4, 5])
        del reshape_108

        # pd_op.reshape: (25x7x7x384xf32) <- (1x5x5x7x7x384xf32, 4xi64)
        reshape_109 = paddle._C_ops.reshape(transpose_68, full_int_array_65)
        del transpose_68

        # pd_op.reshape: (25x49x384xf32) <- (25x7x7x384xf32, 3xi64)
        reshape_110 = paddle._C_ops.reshape(reshape_109, full_int_array_66)
        del reshape_109

        # pd_op.matmul: (25x49x1152xf32) <- (25x49x384xf32, 384x1152xf32)
        matmul_44 = paddle._C_ops.matmul(reshape_110, parameter_64, False, False)
        del parameter_64, reshape_110

        # pd_op.add: (25x49x1152xf32) <- (25x49x1152xf32, 1152xf32)
        add_53 = paddle._C_ops.add(matmul_44, parameter_63)
        del matmul_44, parameter_63

        # pd_op.reshape: (25x49x3x12x32xf32) <- (25x49x1152xf32, 5xi64)
        reshape_111 = paddle._C_ops.reshape(add_53, full_int_array_67)
        del add_53

        # pd_op.transpose: (3x25x12x49x32xf32) <- (25x49x3x12x32xf32)
        transpose_69 = paddle._C_ops.transpose(reshape_111, [2, 0, 3, 1, 4])
        del reshape_111

        # pd_op.slice: (25x12x49x32xf32) <- (3x25x12x49x32xf32, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(
            transpose_69, [0], full_int_array_28, full_int_array_20, [1], [0]
        )

        # pd_op.slice: (25x12x49x32xf32) <- (3x25x12x49x32xf32, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(
            transpose_69, [0], full_int_array_20, full_int_array_21, [1], [0]
        )

        # pd_op.slice: (25x12x49x32xf32) <- (3x25x12x49x32xf32, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(
            transpose_69, [0], full_int_array_21, full_int_array_29, [1], [0]
        )
        del transpose_69

        # pd_op.scale: (25x12x49x32xf32) <- (25x12x49x32xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(slice_28, full_4, float("0"), True)
        del slice_28

        # pd_op.transpose: (25x12x32x49xf32) <- (25x12x49x32xf32)
        transpose_70 = paddle._C_ops.transpose(slice_29, [0, 1, 3, 2])
        del slice_29

        # pd_op.matmul: (25x12x49x49xf32) <- (25x12x49x32xf32, 25x12x32x49xf32)
        matmul_45 = paddle._C_ops.matmul(scale_10, transpose_70, False, False)
        del scale_10, transpose_70

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        reshape_112 = paddle._C_ops.reshape(data_20, full_int_array_30)
        del data_20

        # pd_op.index_select: (2401x12xf32) <- (169x12xf32, 2401xi64)
        index_select_7 = paddle._C_ops.index_select(data_7, reshape_112, 0)
        del data_7, reshape_112

        # pd_op.reshape: (49x49x12xf32) <- (2401x12xf32, 3xi64)
        reshape_113 = paddle._C_ops.reshape(index_select_7, full_int_array_31)
        del index_select_7

        # pd_op.transpose: (12x49x49xf32) <- (49x49x12xf32)
        transpose_71 = paddle._C_ops.transpose(reshape_113, [2, 0, 1])
        del reshape_113

        # pd_op.unsqueeze: (1x12x49x49xf32) <- (12x49x49xf32, 1xi64)
        unsqueeze_27 = paddle._C_ops.unsqueeze(transpose_71, full_int_array_28)
        del transpose_71

        # pd_op.add: (25x12x49x49xf32) <- (25x12x49x49xf32, 1x12x49x49xf32)
        add_54 = paddle._C_ops.add(matmul_45, unsqueeze_27)
        del matmul_45, unsqueeze_27

        # pd_op.reshape: (1x25x12x49x49xf32) <- (25x12x49x49xf32, 5xi64)
        reshape_114 = paddle._C_ops.reshape(add_54, full_int_array_73)
        del add_54

        # pd_op.unsqueeze: (25x1x49x49xf32) <- (25x49x49xf32, 1xi64)
        unsqueeze_28 = paddle._C_ops.unsqueeze(multiply_2, full_int_array_20)

        # pd_op.unsqueeze: (1x25x1x49x49xf32) <- (25x1x49x49xf32, 1xi64)
        unsqueeze_29 = paddle._C_ops.unsqueeze(unsqueeze_28, full_int_array_28)
        del unsqueeze_28

        # pd_op.add: (1x25x12x49x49xf32) <- (1x25x12x49x49xf32, 1x25x1x49x49xf32)
        add_55 = paddle._C_ops.add(reshape_114, unsqueeze_29)
        del reshape_114, unsqueeze_29

        # pd_op.reshape: (25x12x49x49xf32) <- (1x25x12x49x49xf32, 4xi64)
        reshape_115 = paddle._C_ops.reshape(add_55, full_int_array_74)
        del add_55

        # pd_op.softmax: (25x12x49x49xf32) <- (25x12x49x49xf32)
        softmax_7 = paddle._C_ops.softmax(reshape_115, -1)
        del reshape_115

        # pd_op.matmul: (25x12x49x32xf32) <- (25x12x49x49xf32, 25x12x49x32xf32)
        matmul_46 = paddle._C_ops.matmul(softmax_7, slice_30, False, False)
        del slice_30, softmax_7

        # pd_op.transpose: (25x49x12x32xf32) <- (25x12x49x32xf32)
        transpose_72 = paddle._C_ops.transpose(matmul_46, [0, 2, 1, 3])
        del matmul_46

        # pd_op.reshape: (25x49x384xf32) <- (25x49x12x32xf32, 3xi64)
        reshape_116 = paddle._C_ops.reshape(transpose_72, full_int_array_68)
        del transpose_72

        # pd_op.matmul: (25x49x384xf32) <- (25x49x384xf32, 384x384xf32)
        matmul_47 = paddle._C_ops.matmul(reshape_116, parameter_62, False, False)
        del parameter_62, reshape_116

        # pd_op.add: (25x49x384xf32) <- (25x49x384xf32, 384xf32)
        add_56 = paddle._C_ops.add(matmul_47, parameter_61)
        del matmul_47, parameter_61

        # pd_op.reshape: (25x7x7x384xf32) <- (25x49x384xf32, 4xi64)
        reshape_117 = paddle._C_ops.reshape(add_56, full_int_array_65)
        del add_56

        # pd_op.reshape: (1x5x5x7x7x384xf32) <- (25x7x7x384xf32, 6xi64)
        reshape_118 = paddle._C_ops.reshape(reshape_117, full_int_array_69)
        del reshape_117

        # pd_op.transpose: (1x5x7x5x7x384xf32) <- (1x5x5x7x7x384xf32)
        transpose_73 = paddle._C_ops.transpose(reshape_118, [0, 1, 3, 2, 4, 5])
        del reshape_118

        # pd_op.reshape: (1x35x35x384xf32) <- (1x5x7x5x7x384xf32, 4xi64)
        reshape_119 = paddle._C_ops.reshape(transpose_73, full_int_array_70)
        del transpose_73

        # pd_op.roll: (1x35x35x384xf32) <- (1x35x35x384xf32, 2xi64)
        roll_7 = paddle._C_ops.roll(reshape_119, full_int_array_39, [1, 2])
        del reshape_119

        # pd_op.slice: (1x32x32x384xf32) <- (1x35x35x384xf32, 2xi64, 2xi64)
        slice_31 = paddle._C_ops.slice(
            roll_7, [1, 2], full_int_array_2, full_int_array_71, [1, 1], []
        )
        del roll_7

        # pd_op.reshape: (1x1024x384xf32) <- (1x32x32x384xf32, 3xi64)
        reshape_120 = paddle._C_ops.reshape(slice_31, full_int_array_72)
        del slice_31

        # pd_op.add: (1x1024x384xf32) <- (1x1024x384xf32, 1x1024x384xf32)
        add_57 = paddle._C_ops.add(add_52, reshape_120)
        del add_52, reshape_120

        # pd_op.layer_norm: (1x1024x384xf32, 1x1024xf32, 1x1024xf32) <- (1x1024x384xf32, 384xf32, 384xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_57, parameter_60, parameter_59, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_59, parameter_60

        # pd_op.matmul: (1x1024x1536xf32) <- (1x1024x384xf32, 384x1536xf32)
        matmul_48 = paddle._C_ops.matmul(layer_norm_60, parameter_58, False, False)
        del layer_norm_60, parameter_58

        # pd_op.add: (1x1024x1536xf32) <- (1x1024x1536xf32, 1536xf32)
        add_58 = paddle._C_ops.add(matmul_48, parameter_57)
        del matmul_48, parameter_57

        # pd_op.gelu: (1x1024x1536xf32) <- (1x1024x1536xf32)
        gelu_7 = paddle._C_ops.gelu(add_58, False)
        del add_58

        # pd_op.matmul: (1x1024x384xf32) <- (1x1024x1536xf32, 1536x384xf32)
        matmul_49 = paddle._C_ops.matmul(gelu_7, parameter_56, False, False)
        del gelu_7, parameter_56

        # pd_op.add: (1x1024x384xf32) <- (1x1024x384xf32, 384xf32)
        add_59 = paddle._C_ops.add(matmul_49, parameter_55)
        del matmul_49, parameter_55

        # pd_op.add: (1x1024x384xf32) <- (1x1024x384xf32, 1x1024x384xf32)
        add_60 = paddle._C_ops.add(add_57, add_59)
        del add_57, add_59

        # pd_op.layer_norm: (1x1024x384xf32, 1x1024xf32, 1x1024xf32) <- (1x1024x384xf32, 384xf32, 384xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_60, parameter_54, parameter_53, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_53, parameter_54

        # pd_op.reshape: (1x32x32x384xf32) <- (1x1024x384xf32, 4xi64)
        reshape_121 = paddle._C_ops.reshape(layer_norm_63, full_int_array_62)
        del layer_norm_63

        # pd_op.transpose: (1x384x32x32xf32) <- (1x32x32x384xf32)
        transpose_74 = paddle._C_ops.transpose(reshape_121, [0, 3, 1, 2])
        del reshape_121

        # pd_op.unsqueeze: (1x384x1x32x32xf32) <- (1x384x32x32xf32, 1xi64)
        unsqueeze_30 = paddle._C_ops.unsqueeze(transpose_74, full_int_array_21)
        del transpose_74

        # pd_op.pad3d: (1x384x1x35x35xf32) <- (1x384x1x32x32xf32, 6xi64)
        pad3d_8 = paddle._C_ops.pad3d(
            unsqueeze_30, full_int_array_63, "constant", float("0"), "NCDHW"
        )
        del unsqueeze_30

        # pd_op.squeeze: (1x384x35x35xf32) <- (1x384x1x35x35xf32, 1xi64)
        squeeze_8 = paddle._C_ops.squeeze(pad3d_8, full_int_array_21)
        del pad3d_8

        # pd_op.transpose: (1x35x35x384xf32) <- (1x384x35x35xf32)
        transpose_75 = paddle._C_ops.transpose(squeeze_8, [0, 2, 3, 1])
        del squeeze_8

        # pd_op.reshape: (1x5x7x5x7x384xf32) <- (1x35x35x384xf32, 6xi64)
        reshape_122 = paddle._C_ops.reshape(transpose_75, full_int_array_64)
        del transpose_75

        # pd_op.transpose: (1x5x5x7x7x384xf32) <- (1x5x7x5x7x384xf32)
        transpose_76 = paddle._C_ops.transpose(reshape_122, [0, 1, 3, 2, 4, 5])
        del reshape_122

        # pd_op.reshape: (25x7x7x384xf32) <- (1x5x5x7x7x384xf32, 4xi64)
        reshape_123 = paddle._C_ops.reshape(transpose_76, full_int_array_65)
        del transpose_76

        # pd_op.reshape: (25x49x384xf32) <- (25x7x7x384xf32, 3xi64)
        reshape_124 = paddle._C_ops.reshape(reshape_123, full_int_array_66)
        del reshape_123

        # pd_op.matmul: (25x49x1152xf32) <- (25x49x384xf32, 384x1152xf32)
        matmul_50 = paddle._C_ops.matmul(reshape_124, parameter_52, False, False)
        del parameter_52, reshape_124

        # pd_op.add: (25x49x1152xf32) <- (25x49x1152xf32, 1152xf32)
        add_61 = paddle._C_ops.add(matmul_50, parameter_51)
        del matmul_50, parameter_51

        # pd_op.reshape: (25x49x3x12x32xf32) <- (25x49x1152xf32, 5xi64)
        reshape_125 = paddle._C_ops.reshape(add_61, full_int_array_67)
        del add_61

        # pd_op.transpose: (3x25x12x49x32xf32) <- (25x49x3x12x32xf32)
        transpose_77 = paddle._C_ops.transpose(reshape_125, [2, 0, 3, 1, 4])
        del reshape_125

        # pd_op.slice: (25x12x49x32xf32) <- (3x25x12x49x32xf32, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(
            transpose_77, [0], full_int_array_28, full_int_array_20, [1], [0]
        )

        # pd_op.slice: (25x12x49x32xf32) <- (3x25x12x49x32xf32, 1xi64, 1xi64)
        slice_33 = paddle._C_ops.slice(
            transpose_77, [0], full_int_array_20, full_int_array_21, [1], [0]
        )

        # pd_op.slice: (25x12x49x32xf32) <- (3x25x12x49x32xf32, 1xi64, 1xi64)
        slice_34 = paddle._C_ops.slice(
            transpose_77, [0], full_int_array_21, full_int_array_29, [1], [0]
        )
        del transpose_77

        # pd_op.scale: (25x12x49x32xf32) <- (25x12x49x32xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(slice_32, full_4, float("0"), True)
        del slice_32

        # pd_op.transpose: (25x12x32x49xf32) <- (25x12x49x32xf32)
        transpose_78 = paddle._C_ops.transpose(slice_33, [0, 1, 3, 2])
        del slice_33

        # pd_op.matmul: (25x12x49x49xf32) <- (25x12x49x32xf32, 25x12x32x49xf32)
        matmul_51 = paddle._C_ops.matmul(scale_11, transpose_78, False, False)
        del scale_11, transpose_78

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        reshape_126 = paddle._C_ops.reshape(data_21, full_int_array_30)
        del data_21

        # pd_op.index_select: (2401x12xf32) <- (169x12xf32, 2401xi64)
        index_select_8 = paddle._C_ops.index_select(data_8, reshape_126, 0)
        del data_8, reshape_126

        # pd_op.reshape: (49x49x12xf32) <- (2401x12xf32, 3xi64)
        reshape_127 = paddle._C_ops.reshape(index_select_8, full_int_array_31)
        del index_select_8

        # pd_op.transpose: (12x49x49xf32) <- (49x49x12xf32)
        transpose_79 = paddle._C_ops.transpose(reshape_127, [2, 0, 1])
        del reshape_127

        # pd_op.unsqueeze: (1x12x49x49xf32) <- (12x49x49xf32, 1xi64)
        unsqueeze_31 = paddle._C_ops.unsqueeze(transpose_79, full_int_array_28)
        del transpose_79

        # pd_op.add: (25x12x49x49xf32) <- (25x12x49x49xf32, 1x12x49x49xf32)
        add_62 = paddle._C_ops.add(matmul_51, unsqueeze_31)
        del matmul_51, unsqueeze_31

        # pd_op.softmax: (25x12x49x49xf32) <- (25x12x49x49xf32)
        softmax_8 = paddle._C_ops.softmax(add_62, -1)
        del add_62

        # pd_op.matmul: (25x12x49x32xf32) <- (25x12x49x49xf32, 25x12x49x32xf32)
        matmul_52 = paddle._C_ops.matmul(softmax_8, slice_34, False, False)
        del slice_34, softmax_8

        # pd_op.transpose: (25x49x12x32xf32) <- (25x12x49x32xf32)
        transpose_80 = paddle._C_ops.transpose(matmul_52, [0, 2, 1, 3])
        del matmul_52

        # pd_op.reshape: (25x49x384xf32) <- (25x49x12x32xf32, 3xi64)
        reshape_128 = paddle._C_ops.reshape(transpose_80, full_int_array_68)
        del transpose_80

        # pd_op.matmul: (25x49x384xf32) <- (25x49x384xf32, 384x384xf32)
        matmul_53 = paddle._C_ops.matmul(reshape_128, parameter_50, False, False)
        del parameter_50, reshape_128

        # pd_op.add: (25x49x384xf32) <- (25x49x384xf32, 384xf32)
        add_63 = paddle._C_ops.add(matmul_53, parameter_49)
        del matmul_53, parameter_49

        # pd_op.reshape: (25x7x7x384xf32) <- (25x49x384xf32, 4xi64)
        reshape_129 = paddle._C_ops.reshape(add_63, full_int_array_65)
        del add_63

        # pd_op.reshape: (1x5x5x7x7x384xf32) <- (25x7x7x384xf32, 6xi64)
        reshape_130 = paddle._C_ops.reshape(reshape_129, full_int_array_69)
        del reshape_129

        # pd_op.transpose: (1x5x7x5x7x384xf32) <- (1x5x5x7x7x384xf32)
        transpose_81 = paddle._C_ops.transpose(reshape_130, [0, 1, 3, 2, 4, 5])
        del reshape_130

        # pd_op.reshape: (1x35x35x384xf32) <- (1x5x7x5x7x384xf32, 4xi64)
        reshape_131 = paddle._C_ops.reshape(transpose_81, full_int_array_70)
        del transpose_81

        # pd_op.slice: (1x32x32x384xf32) <- (1x35x35x384xf32, 2xi64, 2xi64)
        slice_35 = paddle._C_ops.slice(
            reshape_131, [1, 2], full_int_array_2, full_int_array_71, [1, 1], []
        )
        del reshape_131

        # pd_op.reshape: (1x1024x384xf32) <- (1x32x32x384xf32, 3xi64)
        reshape_132 = paddle._C_ops.reshape(slice_35, full_int_array_72)
        del slice_35

        # pd_op.add: (1x1024x384xf32) <- (1x1024x384xf32, 1x1024x384xf32)
        add_64 = paddle._C_ops.add(add_60, reshape_132)
        del add_60, reshape_132

        # pd_op.layer_norm: (1x1024x384xf32, 1x1024xf32, 1x1024xf32) <- (1x1024x384xf32, 384xf32, 384xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_64, parameter_48, parameter_47, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_47, parameter_48

        # pd_op.matmul: (1x1024x1536xf32) <- (1x1024x384xf32, 384x1536xf32)
        matmul_54 = paddle._C_ops.matmul(layer_norm_66, parameter_46, False, False)
        del layer_norm_66, parameter_46

        # pd_op.add: (1x1024x1536xf32) <- (1x1024x1536xf32, 1536xf32)
        add_65 = paddle._C_ops.add(matmul_54, parameter_45)
        del matmul_54, parameter_45

        # pd_op.gelu: (1x1024x1536xf32) <- (1x1024x1536xf32)
        gelu_8 = paddle._C_ops.gelu(add_65, False)
        del add_65

        # pd_op.matmul: (1x1024x384xf32) <- (1x1024x1536xf32, 1536x384xf32)
        matmul_55 = paddle._C_ops.matmul(gelu_8, parameter_44, False, False)
        del gelu_8, parameter_44

        # pd_op.add: (1x1024x384xf32) <- (1x1024x384xf32, 384xf32)
        add_66 = paddle._C_ops.add(matmul_55, parameter_43)
        del matmul_55, parameter_43

        # pd_op.add: (1x1024x384xf32) <- (1x1024x384xf32, 1x1024x384xf32)
        add_67 = paddle._C_ops.add(add_64, add_66)
        del add_64, add_66

        # pd_op.layer_norm: (1x1024x384xf32, 1x1024xf32, 1x1024xf32) <- (1x1024x384xf32, 384xf32, 384xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_67, parameter_42, parameter_41, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_41, parameter_42

        # pd_op.reshape: (1x32x32x384xf32) <- (1x1024x384xf32, 4xi64)
        reshape_133 = paddle._C_ops.reshape(layer_norm_69, full_int_array_62)
        del layer_norm_69

        # pd_op.transpose: (1x384x32x32xf32) <- (1x32x32x384xf32)
        transpose_82 = paddle._C_ops.transpose(reshape_133, [0, 3, 1, 2])
        del reshape_133

        # pd_op.unsqueeze: (1x384x1x32x32xf32) <- (1x384x32x32xf32, 1xi64)
        unsqueeze_32 = paddle._C_ops.unsqueeze(transpose_82, full_int_array_21)
        del transpose_82

        # pd_op.pad3d: (1x384x1x35x35xf32) <- (1x384x1x32x32xf32, 6xi64)
        pad3d_9 = paddle._C_ops.pad3d(
            unsqueeze_32, full_int_array_63, "constant", float("0"), "NCDHW"
        )
        del full_int_array_63, unsqueeze_32

        # pd_op.squeeze: (1x384x35x35xf32) <- (1x384x1x35x35xf32, 1xi64)
        squeeze_9 = paddle._C_ops.squeeze(pad3d_9, full_int_array_21)
        del pad3d_9

        # pd_op.transpose: (1x35x35x384xf32) <- (1x384x35x35xf32)
        transpose_83 = paddle._C_ops.transpose(squeeze_9, [0, 2, 3, 1])
        del squeeze_9

        # pd_op.roll: (1x35x35x384xf32) <- (1x35x35x384xf32, 2xi64)
        roll_8 = paddle._C_ops.roll(transpose_83, full_int_array_11, [1, 2])
        del transpose_83

        # pd_op.reshape: (1x5x7x5x7x384xf32) <- (1x35x35x384xf32, 6xi64)
        reshape_134 = paddle._C_ops.reshape(roll_8, full_int_array_64)
        del full_int_array_64, roll_8

        # pd_op.transpose: (1x5x5x7x7x384xf32) <- (1x5x7x5x7x384xf32)
        transpose_84 = paddle._C_ops.transpose(reshape_134, [0, 1, 3, 2, 4, 5])
        del reshape_134

        # pd_op.reshape: (25x7x7x384xf32) <- (1x5x5x7x7x384xf32, 4xi64)
        reshape_135 = paddle._C_ops.reshape(transpose_84, full_int_array_65)
        del transpose_84

        # pd_op.reshape: (25x49x384xf32) <- (25x7x7x384xf32, 3xi64)
        reshape_136 = paddle._C_ops.reshape(reshape_135, full_int_array_66)
        del full_int_array_66, reshape_135

        # pd_op.matmul: (25x49x1152xf32) <- (25x49x384xf32, 384x1152xf32)
        matmul_56 = paddle._C_ops.matmul(reshape_136, parameter_40, False, False)
        del parameter_40, reshape_136

        # pd_op.add: (25x49x1152xf32) <- (25x49x1152xf32, 1152xf32)
        add_68 = paddle._C_ops.add(matmul_56, parameter_39)
        del matmul_56, parameter_39

        # pd_op.reshape: (25x49x3x12x32xf32) <- (25x49x1152xf32, 5xi64)
        reshape_137 = paddle._C_ops.reshape(add_68, full_int_array_67)
        del add_68, full_int_array_67

        # pd_op.transpose: (3x25x12x49x32xf32) <- (25x49x3x12x32xf32)
        transpose_85 = paddle._C_ops.transpose(reshape_137, [2, 0, 3, 1, 4])
        del reshape_137

        # pd_op.slice: (25x12x49x32xf32) <- (3x25x12x49x32xf32, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(
            transpose_85, [0], full_int_array_28, full_int_array_20, [1], [0]
        )

        # pd_op.slice: (25x12x49x32xf32) <- (3x25x12x49x32xf32, 1xi64, 1xi64)
        slice_37 = paddle._C_ops.slice(
            transpose_85, [0], full_int_array_20, full_int_array_21, [1], [0]
        )

        # pd_op.slice: (25x12x49x32xf32) <- (3x25x12x49x32xf32, 1xi64, 1xi64)
        slice_38 = paddle._C_ops.slice(
            transpose_85, [0], full_int_array_21, full_int_array_29, [1], [0]
        )
        del transpose_85

        # pd_op.scale: (25x12x49x32xf32) <- (25x12x49x32xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(slice_36, full_4, float("0"), True)
        del slice_36

        # pd_op.transpose: (25x12x32x49xf32) <- (25x12x49x32xf32)
        transpose_86 = paddle._C_ops.transpose(slice_37, [0, 1, 3, 2])
        del slice_37

        # pd_op.matmul: (25x12x49x49xf32) <- (25x12x49x32xf32, 25x12x32x49xf32)
        matmul_57 = paddle._C_ops.matmul(scale_12, transpose_86, False, False)
        del scale_12, transpose_86

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        reshape_138 = paddle._C_ops.reshape(data_22, full_int_array_30)
        del data_22

        # pd_op.index_select: (2401x12xf32) <- (169x12xf32, 2401xi64)
        index_select_9 = paddle._C_ops.index_select(data_9, reshape_138, 0)
        del data_9, reshape_138

        # pd_op.reshape: (49x49x12xf32) <- (2401x12xf32, 3xi64)
        reshape_139 = paddle._C_ops.reshape(index_select_9, full_int_array_31)
        del index_select_9

        # pd_op.transpose: (12x49x49xf32) <- (49x49x12xf32)
        transpose_87 = paddle._C_ops.transpose(reshape_139, [2, 0, 1])
        del reshape_139

        # pd_op.unsqueeze: (1x12x49x49xf32) <- (12x49x49xf32, 1xi64)
        unsqueeze_33 = paddle._C_ops.unsqueeze(transpose_87, full_int_array_28)
        del transpose_87

        # pd_op.add: (25x12x49x49xf32) <- (25x12x49x49xf32, 1x12x49x49xf32)
        add_69 = paddle._C_ops.add(matmul_57, unsqueeze_33)
        del matmul_57, unsqueeze_33

        # pd_op.reshape: (1x25x12x49x49xf32) <- (25x12x49x49xf32, 5xi64)
        reshape_140 = paddle._C_ops.reshape(add_69, full_int_array_73)
        del add_69, full_int_array_73

        # pd_op.unsqueeze: (25x1x49x49xf32) <- (25x49x49xf32, 1xi64)
        unsqueeze_34 = paddle._C_ops.unsqueeze(multiply_2, full_int_array_20)
        del multiply_2

        # pd_op.unsqueeze: (1x25x1x49x49xf32) <- (25x1x49x49xf32, 1xi64)
        unsqueeze_35 = paddle._C_ops.unsqueeze(unsqueeze_34, full_int_array_28)
        del unsqueeze_34

        # pd_op.add: (1x25x12x49x49xf32) <- (1x25x12x49x49xf32, 1x25x1x49x49xf32)
        add_70 = paddle._C_ops.add(reshape_140, unsqueeze_35)
        del reshape_140, unsqueeze_35

        # pd_op.reshape: (25x12x49x49xf32) <- (1x25x12x49x49xf32, 4xi64)
        reshape_141 = paddle._C_ops.reshape(add_70, full_int_array_74)
        del add_70, full_int_array_74

        # pd_op.softmax: (25x12x49x49xf32) <- (25x12x49x49xf32)
        softmax_9 = paddle._C_ops.softmax(reshape_141, -1)
        del reshape_141

        # pd_op.matmul: (25x12x49x32xf32) <- (25x12x49x49xf32, 25x12x49x32xf32)
        matmul_58 = paddle._C_ops.matmul(softmax_9, slice_38, False, False)
        del slice_38, softmax_9

        # pd_op.transpose: (25x49x12x32xf32) <- (25x12x49x32xf32)
        transpose_88 = paddle._C_ops.transpose(matmul_58, [0, 2, 1, 3])
        del matmul_58

        # pd_op.reshape: (25x49x384xf32) <- (25x49x12x32xf32, 3xi64)
        reshape_142 = paddle._C_ops.reshape(transpose_88, full_int_array_68)
        del full_int_array_68, transpose_88

        # pd_op.matmul: (25x49x384xf32) <- (25x49x384xf32, 384x384xf32)
        matmul_59 = paddle._C_ops.matmul(reshape_142, parameter_38, False, False)
        del parameter_38, reshape_142

        # pd_op.add: (25x49x384xf32) <- (25x49x384xf32, 384xf32)
        add_71 = paddle._C_ops.add(matmul_59, parameter_37)
        del matmul_59, parameter_37

        # pd_op.reshape: (25x7x7x384xf32) <- (25x49x384xf32, 4xi64)
        reshape_143 = paddle._C_ops.reshape(add_71, full_int_array_65)
        del add_71, full_int_array_65

        # pd_op.reshape: (1x5x5x7x7x384xf32) <- (25x7x7x384xf32, 6xi64)
        reshape_144 = paddle._C_ops.reshape(reshape_143, full_int_array_69)
        del full_int_array_69, reshape_143

        # pd_op.transpose: (1x5x7x5x7x384xf32) <- (1x5x5x7x7x384xf32)
        transpose_89 = paddle._C_ops.transpose(reshape_144, [0, 1, 3, 2, 4, 5])
        del reshape_144

        # pd_op.reshape: (1x35x35x384xf32) <- (1x5x7x5x7x384xf32, 4xi64)
        reshape_145 = paddle._C_ops.reshape(transpose_89, full_int_array_70)
        del full_int_array_70, transpose_89

        # pd_op.roll: (1x35x35x384xf32) <- (1x35x35x384xf32, 2xi64)
        roll_9 = paddle._C_ops.roll(reshape_145, full_int_array_39, [1, 2])
        del reshape_145

        # pd_op.slice: (1x32x32x384xf32) <- (1x35x35x384xf32, 2xi64, 2xi64)
        slice_39 = paddle._C_ops.slice(
            roll_9, [1, 2], full_int_array_2, full_int_array_71, [1, 1], []
        )
        del full_int_array_71, roll_9

        # pd_op.reshape: (1x1024x384xf32) <- (1x32x32x384xf32, 3xi64)
        reshape_146 = paddle._C_ops.reshape(slice_39, full_int_array_72)
        del full_int_array_72, slice_39

        # pd_op.add: (1x1024x384xf32) <- (1x1024x384xf32, 1x1024x384xf32)
        add_72 = paddle._C_ops.add(add_67, reshape_146)
        del add_67, reshape_146

        # pd_op.layer_norm: (1x1024x384xf32, 1x1024xf32, 1x1024xf32) <- (1x1024x384xf32, 384xf32, 384xf32)
        layer_norm_72, layer_norm_73, layer_norm_74 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_72, parameter_36, parameter_35, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_35, parameter_36

        # pd_op.matmul: (1x1024x1536xf32) <- (1x1024x384xf32, 384x1536xf32)
        matmul_60 = paddle._C_ops.matmul(layer_norm_72, parameter_34, False, False)
        del layer_norm_72, parameter_34

        # pd_op.add: (1x1024x1536xf32) <- (1x1024x1536xf32, 1536xf32)
        add_73 = paddle._C_ops.add(matmul_60, parameter_33)
        del matmul_60, parameter_33

        # pd_op.gelu: (1x1024x1536xf32) <- (1x1024x1536xf32)
        gelu_9 = paddle._C_ops.gelu(add_73, False)
        del add_73

        # pd_op.matmul: (1x1024x384xf32) <- (1x1024x1536xf32, 1536x384xf32)
        matmul_61 = paddle._C_ops.matmul(gelu_9, parameter_32, False, False)
        del gelu_9, parameter_32

        # pd_op.add: (1x1024x384xf32) <- (1x1024x384xf32, 384xf32)
        add_74 = paddle._C_ops.add(matmul_61, parameter_31)
        del matmul_61, parameter_31

        # pd_op.add: (1x1024x384xf32) <- (1x1024x384xf32, 1x1024x384xf32)
        add_75 = paddle._C_ops.add(add_72, add_74)
        del add_72, add_74

        # pd_op.reshape: (1x32x32x384xf32) <- (1x1024x384xf32, 4xi64)
        reshape_147 = paddle._C_ops.reshape(add_75, full_int_array_62)
        del full_int_array_62

        # pd_op.strided_slice: (1x16x16x384xf32) <- (1x32x32x384xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_8 = paddle._C_ops.strided_slice(
            reshape_147, [1, 2], full_int_array_2, full_int_array_16, full_int_array_40
        )

        # pd_op.strided_slice: (1x16x16x384xf32) <- (1x32x32x384xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_9 = paddle._C_ops.strided_slice(
            reshape_147, [1, 2], full_int_array_41, full_int_array_16, full_int_array_40
        )
        del full_int_array_41

        # pd_op.strided_slice: (1x16x16x384xf32) <- (1x32x32x384xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_10 = paddle._C_ops.strided_slice(
            reshape_147, [1, 2], full_int_array_42, full_int_array_16, full_int_array_40
        )
        del full_int_array_42

        # pd_op.strided_slice: (1x16x16x384xf32) <- (1x32x32x384xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_11 = paddle._C_ops.strided_slice(
            reshape_147, [1, 2], full_int_array_4, full_int_array_16, full_int_array_40
        )
        del full_int_array_40, reshape_147

        # builtin.combine: ([1x16x16x384xf32, 1x16x16x384xf32, 1x16x16x384xf32, 1x16x16x384xf32]) <- (1x16x16x384xf32, 1x16x16x384xf32, 1x16x16x384xf32, 1x16x16x384xf32)
        combine_2 = [
            strided_slice_8,
            strided_slice_9,
            strided_slice_10,
            strided_slice_11,
        ]
        del strided_slice_10, strided_slice_11, strided_slice_8, strided_slice_9

        # pd_op.concat: (1x16x16x1536xf32) <- ([1x16x16x384xf32, 1x16x16x384xf32, 1x16x16x384xf32, 1x16x16x384xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_5)
        del combine_2, full_5

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_75 = [1, -1, 1536]

        # pd_op.reshape: (1x256x1536xf32) <- (1x16x16x1536xf32, 3xi64)
        reshape_148 = paddle._C_ops.reshape(concat_2, full_int_array_75)
        del concat_2, full_int_array_75

        # pd_op.layer_norm: (1x256x1536xf32, 1x256xf32, 1x256xf32) <- (1x256x1536xf32, 1536xf32, 1536xf32)
        layer_norm_75, layer_norm_76, layer_norm_77 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                reshape_148, parameter_30, parameter_29, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_29, parameter_30, reshape_148

        # pd_op.matmul: (1x256x768xf32) <- (1x256x1536xf32, 1536x768xf32)
        matmul_62 = paddle._C_ops.matmul(layer_norm_75, parameter_28, False, False)
        del layer_norm_75, parameter_28

        # pd_op.layer_norm: (1x1024x384xf32, 1x1024xf32, 1x1024xf32) <- (1x1024x384xf32, 384xf32, 384xf32)
        layer_norm_78, layer_norm_79, layer_norm_80 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_75, parameter_27, parameter_26, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_75, parameter_26, parameter_27

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_76 = [-1, 32, 32, 384]

        # pd_op.reshape: (1x32x32x384xf32) <- (1x1024x384xf32, 4xi64)
        reshape_149 = paddle._C_ops.reshape(layer_norm_78, full_int_array_76)
        del full_int_array_76, layer_norm_78

        # pd_op.transpose: (1x384x32x32xf32) <- (1x32x32x384xf32)
        transpose_2 = paddle._C_ops.transpose(reshape_149, [0, 3, 1, 2])
        del reshape_149

        # pd_op.full: (1x21x21x1xf32) <- ()
        full_8 = paddle._C_ops.full(
            [1, 21, 21, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x21x21x1xf32) <- (1x21x21x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__27 = paddle._C_ops.set_value_(
            full_8,
            full_int_array_2,
            full_int_array_3,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("0")],
        )
        del full_8

        # pd_op.set_value_: (1x21x21x1xf32) <- (1x21x21x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__28 = paddle._C_ops.set_value_(
            set_value__27,
            full_int_array_5,
            full_int_array_6,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_5, set_value__27

        # pd_op.set_value_: (1x21x21x1xf32) <- (1x21x21x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__29 = paddle._C_ops.set_value_(
            set_value__28,
            full_int_array_7,
            full_int_array_8,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("2")],
        )
        del full_int_array_7, full_int_array_8, set_value__28

        # pd_op.set_value_: (1x21x21x1xf32) <- (1x21x21x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__30 = paddle._C_ops.set_value_(
            set_value__29,
            full_int_array_9,
            full_int_array_10,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("3")],
        )
        del full_int_array_9, set_value__29

        # pd_op.set_value_: (1x21x21x1xf32) <- (1x21x21x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__31 = paddle._C_ops.set_value_(
            set_value__30,
            full_int_array_3,
            full_int_array_11,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("4")],
        )
        del full_int_array_3, set_value__30

        # pd_op.set_value_: (1x21x21x1xf32) <- (1x21x21x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__32 = paddle._C_ops.set_value_(
            set_value__31,
            full_int_array_6,
            full_int_array_12,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("5")],
        )
        del full_int_array_12, full_int_array_6, set_value__31

        # pd_op.set_value_: (1x21x21x1xf32) <- (1x21x21x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__33 = paddle._C_ops.set_value_(
            set_value__32,
            full_int_array_13,
            full_int_array_14,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("6")],
        )
        del full_int_array_13, full_int_array_14, set_value__32

        # pd_op.set_value_: (1x21x21x1xf32) <- (1x21x21x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__34 = paddle._C_ops.set_value_(
            set_value__33,
            full_int_array_10,
            full_int_array_15,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("7")],
        )
        del full_int_array_10, full_int_array_15, set_value__33

        # pd_op.set_value_: (1x21x21x1xf32) <- (1x21x21x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__35 = paddle._C_ops.set_value_(
            set_value__34,
            full_int_array_11,
            full_int_array_16,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("8")],
        )
        del full_int_array_16, full_int_array_4, set_value__34

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_77 = [1, 3, 7, 3, 7, 1]

        # pd_op.reshape: (1x3x7x3x7x1xf32) <- (1x21x21x1xf32, 6xi64)
        reshape_150 = paddle._C_ops.reshape(set_value__35, full_int_array_77)
        del full_int_array_77

        # pd_op.transpose: (1x3x3x7x7x1xf32) <- (1x3x7x3x7x1xf32)
        transpose_90 = paddle._C_ops.transpose(reshape_150, [0, 1, 3, 2, 4, 5])
        del reshape_150

        # pd_op.reshape: (9x7x7x1xf32) <- (1x3x3x7x7x1xf32, 4xi64)
        reshape_151 = paddle._C_ops.reshape(transpose_90, full_int_array_18)
        del full_int_array_18, transpose_90

        # pd_op.reshape: (9x49xf32) <- (9x7x7x1xf32, 2xi64)
        reshape_152 = paddle._C_ops.reshape(reshape_151, full_int_array_19)
        del full_int_array_19, reshape_151

        # pd_op.unsqueeze: (9x1x49xf32) <- (9x49xf32, 1xi64)
        unsqueeze_36 = paddle._C_ops.unsqueeze(reshape_152, full_int_array_20)

        # pd_op.unsqueeze: (9x49x1xf32) <- (9x49xf32, 1xi64)
        unsqueeze_37 = paddle._C_ops.unsqueeze(reshape_152, full_int_array_21)
        del reshape_152

        # pd_op.subtract: (9x49x49xf32) <- (9x1x49xf32, 9x49x1xf32)
        subtract_3 = paddle._C_ops.subtract(unsqueeze_36, unsqueeze_37)
        del unsqueeze_36, unsqueeze_37

        # pd_op.full_like: (9x49x49xf32) <- (9x49x49xf32, 1xf32)
        full_like_3 = paddle._C_ops.full_like(
            subtract_3,
            full_1,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )
        del full_1

        # pd_op.scale: (9x49x49xf32) <- (9x49x49xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(full_like_3, full_2, float("0"), True)
        del full_2, full_like_3

        # pd_op.not_equal: (9x49x49xb) <- (9x49x49xf32, xf32)
        not_equal_3 = paddle._C_ops.not_equal(subtract_3, full_3)
        del full_3, subtract_3

        # pd_op.cast: (9x49x49xf32) <- (9x49x49xb)
        cast_3 = paddle._C_ops.cast(not_equal_3, paddle.float32)
        del not_equal_3

        # pd_op.multiply: (9x49x49xf32) <- (9x49x49xf32, 9x49x49xf32)
        multiply_3 = paddle._C_ops.multiply(scale_13, cast_3)
        del cast_3, scale_13

        # pd_op.layer_norm: (1x256x768xf32, 1x256xf32, 1x256xf32) <- (1x256x768xf32, 768xf32, 768xf32)
        layer_norm_81, layer_norm_82, layer_norm_83 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                matmul_62, parameter_25, parameter_24, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_24, parameter_25

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_78 = [1, 16, 16, 768]

        # pd_op.reshape: (1x16x16x768xf32) <- (1x256x768xf32, 4xi64)
        reshape_153 = paddle._C_ops.reshape(layer_norm_81, full_int_array_78)
        del layer_norm_81

        # pd_op.transpose: (1x768x16x16xf32) <- (1x16x16x768xf32)
        transpose_91 = paddle._C_ops.transpose(reshape_153, [0, 3, 1, 2])
        del reshape_153

        # pd_op.unsqueeze: (1x768x1x16x16xf32) <- (1x768x16x16xf32, 1xi64)
        unsqueeze_38 = paddle._C_ops.unsqueeze(transpose_91, full_int_array_21)
        del transpose_91

        # pd_op.pad3d: (1x768x1x21x21xf32) <- (1x768x1x16x16xf32, 6xi64)
        pad3d_10 = paddle._C_ops.pad3d(
            unsqueeze_38, full_int_array_23, "constant", float("0"), "NCDHW"
        )
        del unsqueeze_38

        # pd_op.squeeze: (1x768x21x21xf32) <- (1x768x1x21x21xf32, 1xi64)
        squeeze_10 = paddle._C_ops.squeeze(pad3d_10, full_int_array_21)
        del pad3d_10

        # pd_op.transpose: (1x21x21x768xf32) <- (1x768x21x21xf32)
        transpose_92 = paddle._C_ops.transpose(squeeze_10, [0, 2, 3, 1])
        del squeeze_10

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_79 = [1, 3, 7, 3, 7, 768]

        # pd_op.reshape: (1x3x7x3x7x768xf32) <- (1x21x21x768xf32, 6xi64)
        reshape_154 = paddle._C_ops.reshape(transpose_92, full_int_array_79)
        del transpose_92

        # pd_op.transpose: (1x3x3x7x7x768xf32) <- (1x3x7x3x7x768xf32)
        transpose_93 = paddle._C_ops.transpose(reshape_154, [0, 1, 3, 2, 4, 5])
        del reshape_154

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_80 = [-1, 7, 7, 768]

        # pd_op.reshape: (9x7x7x768xf32) <- (1x3x3x7x7x768xf32, 4xi64)
        reshape_155 = paddle._C_ops.reshape(transpose_93, full_int_array_80)
        del transpose_93

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_81 = [-1, 49, 768]

        # pd_op.reshape: (9x49x768xf32) <- (9x7x7x768xf32, 3xi64)
        reshape_156 = paddle._C_ops.reshape(reshape_155, full_int_array_81)
        del reshape_155

        # pd_op.matmul: (9x49x2304xf32) <- (9x49x768xf32, 768x2304xf32)
        matmul_63 = paddle._C_ops.matmul(reshape_156, parameter_23, False, False)
        del parameter_23, reshape_156

        # pd_op.add: (9x49x2304xf32) <- (9x49x2304xf32, 2304xf32)
        add_76 = paddle._C_ops.add(matmul_63, parameter_22)
        del matmul_63, parameter_22

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_82 = [9, 49, 3, 24, 32]

        # pd_op.reshape: (9x49x3x24x32xf32) <- (9x49x2304xf32, 5xi64)
        reshape_157 = paddle._C_ops.reshape(add_76, full_int_array_82)
        del add_76

        # pd_op.transpose: (3x9x24x49x32xf32) <- (9x49x3x24x32xf32)
        transpose_94 = paddle._C_ops.transpose(reshape_157, [2, 0, 3, 1, 4])
        del reshape_157

        # pd_op.slice: (9x24x49x32xf32) <- (3x9x24x49x32xf32, 1xi64, 1xi64)
        slice_40 = paddle._C_ops.slice(
            transpose_94, [0], full_int_array_28, full_int_array_20, [1], [0]
        )

        # pd_op.slice: (9x24x49x32xf32) <- (3x9x24x49x32xf32, 1xi64, 1xi64)
        slice_41 = paddle._C_ops.slice(
            transpose_94, [0], full_int_array_20, full_int_array_21, [1], [0]
        )

        # pd_op.slice: (9x24x49x32xf32) <- (3x9x24x49x32xf32, 1xi64, 1xi64)
        slice_42 = paddle._C_ops.slice(
            transpose_94, [0], full_int_array_21, full_int_array_29, [1], [0]
        )
        del transpose_94

        # pd_op.scale: (9x24x49x32xf32) <- (9x24x49x32xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(slice_40, full_4, float("0"), True)
        del slice_40

        # pd_op.transpose: (9x24x32x49xf32) <- (9x24x49x32xf32)
        transpose_95 = paddle._C_ops.transpose(slice_41, [0, 1, 3, 2])
        del slice_41

        # pd_op.matmul: (9x24x49x49xf32) <- (9x24x49x32xf32, 9x24x32x49xf32)
        matmul_64 = paddle._C_ops.matmul(scale_14, transpose_95, False, False)
        del scale_14, transpose_95

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        reshape_158 = paddle._C_ops.reshape(data_23, full_int_array_30)
        del data_23

        # pd_op.index_select: (2401x24xf32) <- (169x24xf32, 2401xi64)
        index_select_10 = paddle._C_ops.index_select(data_10, reshape_158, 0)
        del data_10, reshape_158

        # pd_op.reshape: (49x49x24xf32) <- (2401x24xf32, 3xi64)
        reshape_159 = paddle._C_ops.reshape(index_select_10, full_int_array_31)
        del index_select_10

        # pd_op.transpose: (24x49x49xf32) <- (49x49x24xf32)
        transpose_96 = paddle._C_ops.transpose(reshape_159, [2, 0, 1])
        del reshape_159

        # pd_op.unsqueeze: (1x24x49x49xf32) <- (24x49x49xf32, 1xi64)
        unsqueeze_39 = paddle._C_ops.unsqueeze(transpose_96, full_int_array_28)
        del transpose_96

        # pd_op.add: (9x24x49x49xf32) <- (9x24x49x49xf32, 1x24x49x49xf32)
        add_77 = paddle._C_ops.add(matmul_64, unsqueeze_39)
        del matmul_64, unsqueeze_39

        # pd_op.softmax: (9x24x49x49xf32) <- (9x24x49x49xf32)
        softmax_10 = paddle._C_ops.softmax(add_77, -1)
        del add_77

        # pd_op.matmul: (9x24x49x32xf32) <- (9x24x49x49xf32, 9x24x49x32xf32)
        matmul_65 = paddle._C_ops.matmul(softmax_10, slice_42, False, False)
        del slice_42, softmax_10

        # pd_op.transpose: (9x49x24x32xf32) <- (9x24x49x32xf32)
        transpose_97 = paddle._C_ops.transpose(matmul_65, [0, 2, 1, 3])
        del matmul_65

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_83 = [9, 49, 768]

        # pd_op.reshape: (9x49x768xf32) <- (9x49x24x32xf32, 3xi64)
        reshape_160 = paddle._C_ops.reshape(transpose_97, full_int_array_83)
        del transpose_97

        # pd_op.matmul: (9x49x768xf32) <- (9x49x768xf32, 768x768xf32)
        matmul_66 = paddle._C_ops.matmul(reshape_160, parameter_21, False, False)
        del parameter_21, reshape_160

        # pd_op.add: (9x49x768xf32) <- (9x49x768xf32, 768xf32)
        add_78 = paddle._C_ops.add(matmul_66, parameter_20)
        del matmul_66, parameter_20

        # pd_op.reshape: (9x7x7x768xf32) <- (9x49x768xf32, 4xi64)
        reshape_161 = paddle._C_ops.reshape(add_78, full_int_array_80)
        del add_78

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_84 = [1, 3, 3, 7, 7, -1]

        # pd_op.reshape: (1x3x3x7x7x768xf32) <- (9x7x7x768xf32, 6xi64)
        reshape_162 = paddle._C_ops.reshape(reshape_161, full_int_array_84)
        del reshape_161

        # pd_op.transpose: (1x3x7x3x7x768xf32) <- (1x3x3x7x7x768xf32)
        transpose_98 = paddle._C_ops.transpose(reshape_162, [0, 1, 3, 2, 4, 5])
        del reshape_162

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_85 = [1, 21, 21, -1]

        # pd_op.reshape: (1x21x21x768xf32) <- (1x3x7x3x7x768xf32, 4xi64)
        reshape_163 = paddle._C_ops.reshape(transpose_98, full_int_array_85)
        del transpose_98

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_86 = [16, 16]

        # pd_op.slice: (1x16x16x768xf32) <- (1x21x21x768xf32, 2xi64, 2xi64)
        slice_43 = paddle._C_ops.slice(
            reshape_163, [1, 2], full_int_array_2, full_int_array_86, [1, 1], []
        )
        del reshape_163

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_87 = [1, 256, 768]

        # pd_op.reshape: (1x256x768xf32) <- (1x16x16x768xf32, 3xi64)
        reshape_164 = paddle._C_ops.reshape(slice_43, full_int_array_87)
        del slice_43

        # pd_op.add: (1x256x768xf32) <- (1x256x768xf32, 1x256x768xf32)
        add_79 = paddle._C_ops.add(matmul_62, reshape_164)
        del matmul_62, reshape_164

        # pd_op.layer_norm: (1x256x768xf32, 1x256xf32, 1x256xf32) <- (1x256x768xf32, 768xf32, 768xf32)
        layer_norm_84, layer_norm_85, layer_norm_86 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_79, parameter_19, parameter_18, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_18, parameter_19

        # pd_op.matmul: (1x256x3072xf32) <- (1x256x768xf32, 768x3072xf32)
        matmul_67 = paddle._C_ops.matmul(layer_norm_84, parameter_17, False, False)
        del layer_norm_84, parameter_17

        # pd_op.add: (1x256x3072xf32) <- (1x256x3072xf32, 3072xf32)
        add_80 = paddle._C_ops.add(matmul_67, parameter_16)
        del matmul_67, parameter_16

        # pd_op.gelu: (1x256x3072xf32) <- (1x256x3072xf32)
        gelu_10 = paddle._C_ops.gelu(add_80, False)
        del add_80

        # pd_op.matmul: (1x256x768xf32) <- (1x256x3072xf32, 3072x768xf32)
        matmul_68 = paddle._C_ops.matmul(gelu_10, parameter_15, False, False)
        del gelu_10, parameter_15

        # pd_op.add: (1x256x768xf32) <- (1x256x768xf32, 768xf32)
        add_81 = paddle._C_ops.add(matmul_68, parameter_14)
        del matmul_68, parameter_14

        # pd_op.add: (1x256x768xf32) <- (1x256x768xf32, 1x256x768xf32)
        add_82 = paddle._C_ops.add(add_79, add_81)
        del add_79, add_81

        # pd_op.layer_norm: (1x256x768xf32, 1x256xf32, 1x256xf32) <- (1x256x768xf32, 768xf32, 768xf32)
        layer_norm_87, layer_norm_88, layer_norm_89 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_82, parameter_13, parameter_12, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_12, parameter_13

        # pd_op.reshape: (1x16x16x768xf32) <- (1x256x768xf32, 4xi64)
        reshape_165 = paddle._C_ops.reshape(layer_norm_87, full_int_array_78)
        del full_int_array_78, layer_norm_87

        # pd_op.transpose: (1x768x16x16xf32) <- (1x16x16x768xf32)
        transpose_99 = paddle._C_ops.transpose(reshape_165, [0, 3, 1, 2])
        del reshape_165

        # pd_op.unsqueeze: (1x768x1x16x16xf32) <- (1x768x16x16xf32, 1xi64)
        unsqueeze_40 = paddle._C_ops.unsqueeze(transpose_99, full_int_array_21)
        del transpose_99

        # pd_op.pad3d: (1x768x1x21x21xf32) <- (1x768x1x16x16xf32, 6xi64)
        pad3d_11 = paddle._C_ops.pad3d(
            unsqueeze_40, full_int_array_23, "constant", float("0"), "NCDHW"
        )
        del full_int_array_23, unsqueeze_40

        # pd_op.squeeze: (1x768x21x21xf32) <- (1x768x1x21x21xf32, 1xi64)
        squeeze_11 = paddle._C_ops.squeeze(pad3d_11, full_int_array_21)
        del pad3d_11

        # pd_op.transpose: (1x21x21x768xf32) <- (1x768x21x21xf32)
        transpose_100 = paddle._C_ops.transpose(squeeze_11, [0, 2, 3, 1])
        del squeeze_11

        # pd_op.roll: (1x21x21x768xf32) <- (1x21x21x768xf32, 2xi64)
        roll_10 = paddle._C_ops.roll(transpose_100, full_int_array_11, [1, 2])
        del full_int_array_11, transpose_100

        # pd_op.reshape: (1x3x7x3x7x768xf32) <- (1x21x21x768xf32, 6xi64)
        reshape_166 = paddle._C_ops.reshape(roll_10, full_int_array_79)
        del full_int_array_79, roll_10

        # pd_op.transpose: (1x3x3x7x7x768xf32) <- (1x3x7x3x7x768xf32)
        transpose_101 = paddle._C_ops.transpose(reshape_166, [0, 1, 3, 2, 4, 5])
        del reshape_166

        # pd_op.reshape: (9x7x7x768xf32) <- (1x3x3x7x7x768xf32, 4xi64)
        reshape_167 = paddle._C_ops.reshape(transpose_101, full_int_array_80)
        del transpose_101

        # pd_op.reshape: (9x49x768xf32) <- (9x7x7x768xf32, 3xi64)
        reshape_168 = paddle._C_ops.reshape(reshape_167, full_int_array_81)
        del full_int_array_81, reshape_167

        # pd_op.matmul: (9x49x2304xf32) <- (9x49x768xf32, 768x2304xf32)
        matmul_69 = paddle._C_ops.matmul(reshape_168, parameter_11, False, False)
        del parameter_11, reshape_168

        # pd_op.add: (9x49x2304xf32) <- (9x49x2304xf32, 2304xf32)
        add_83 = paddle._C_ops.add(matmul_69, parameter_10)
        del matmul_69, parameter_10

        # pd_op.reshape: (9x49x3x24x32xf32) <- (9x49x2304xf32, 5xi64)
        reshape_169 = paddle._C_ops.reshape(add_83, full_int_array_82)
        del add_83, full_int_array_82

        # pd_op.transpose: (3x9x24x49x32xf32) <- (9x49x3x24x32xf32)
        transpose_102 = paddle._C_ops.transpose(reshape_169, [2, 0, 3, 1, 4])
        del reshape_169

        # pd_op.slice: (9x24x49x32xf32) <- (3x9x24x49x32xf32, 1xi64, 1xi64)
        slice_44 = paddle._C_ops.slice(
            transpose_102, [0], full_int_array_28, full_int_array_20, [1], [0]
        )

        # pd_op.slice: (9x24x49x32xf32) <- (3x9x24x49x32xf32, 1xi64, 1xi64)
        slice_45 = paddle._C_ops.slice(
            transpose_102, [0], full_int_array_20, full_int_array_21, [1], [0]
        )

        # pd_op.slice: (9x24x49x32xf32) <- (3x9x24x49x32xf32, 1xi64, 1xi64)
        slice_46 = paddle._C_ops.slice(
            transpose_102, [0], full_int_array_21, full_int_array_29, [1], [0]
        )
        del full_int_array_21, full_int_array_29, transpose_102

        # pd_op.scale: (9x24x49x32xf32) <- (9x24x49x32xf32, 1xf32)
        scale_15 = paddle._C_ops.scale(slice_44, full_4, float("0"), True)
        del full_4, slice_44

        # pd_op.transpose: (9x24x32x49xf32) <- (9x24x49x32xf32)
        transpose_103 = paddle._C_ops.transpose(slice_45, [0, 1, 3, 2])
        del slice_45

        # pd_op.matmul: (9x24x49x49xf32) <- (9x24x49x32xf32, 9x24x32x49xf32)
        matmul_70 = paddle._C_ops.matmul(scale_15, transpose_103, False, False)
        del scale_15, transpose_103

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        reshape_170 = paddle._C_ops.reshape(data_24, full_int_array_30)
        del data_24, full_int_array_30

        # pd_op.index_select: (2401x24xf32) <- (169x24xf32, 2401xi64)
        index_select_11 = paddle._C_ops.index_select(data_11, reshape_170, 0)
        del data_11, reshape_170

        # pd_op.reshape: (49x49x24xf32) <- (2401x24xf32, 3xi64)
        reshape_171 = paddle._C_ops.reshape(index_select_11, full_int_array_31)
        del full_int_array_31, index_select_11

        # pd_op.transpose: (24x49x49xf32) <- (49x49x24xf32)
        transpose_104 = paddle._C_ops.transpose(reshape_171, [2, 0, 1])
        del reshape_171

        # pd_op.unsqueeze: (1x24x49x49xf32) <- (24x49x49xf32, 1xi64)
        unsqueeze_41 = paddle._C_ops.unsqueeze(transpose_104, full_int_array_28)
        del transpose_104

        # pd_op.add: (9x24x49x49xf32) <- (9x24x49x49xf32, 1x24x49x49xf32)
        add_84 = paddle._C_ops.add(matmul_70, unsqueeze_41)
        del matmul_70, unsqueeze_41

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_88 = [1, 9, 24, 49, 49]

        # pd_op.reshape: (1x9x24x49x49xf32) <- (9x24x49x49xf32, 5xi64)
        reshape_172 = paddle._C_ops.reshape(add_84, full_int_array_88)
        del add_84, full_int_array_88

        # pd_op.unsqueeze: (9x1x49x49xf32) <- (9x49x49xf32, 1xi64)
        unsqueeze_42 = paddle._C_ops.unsqueeze(multiply_3, full_int_array_20)
        del full_int_array_20, multiply_3

        # pd_op.unsqueeze: (1x9x1x49x49xf32) <- (9x1x49x49xf32, 1xi64)
        unsqueeze_43 = paddle._C_ops.unsqueeze(unsqueeze_42, full_int_array_28)
        del full_int_array_28, unsqueeze_42

        # pd_op.add: (1x9x24x49x49xf32) <- (1x9x24x49x49xf32, 1x9x1x49x49xf32)
        add_85 = paddle._C_ops.add(reshape_172, unsqueeze_43)
        del reshape_172, unsqueeze_43

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_89 = [-1, 24, 49, 49]

        # pd_op.reshape: (9x24x49x49xf32) <- (1x9x24x49x49xf32, 4xi64)
        reshape_173 = paddle._C_ops.reshape(add_85, full_int_array_89)
        del add_85, full_int_array_89

        # pd_op.softmax: (9x24x49x49xf32) <- (9x24x49x49xf32)
        softmax_11 = paddle._C_ops.softmax(reshape_173, -1)
        del reshape_173

        # pd_op.matmul: (9x24x49x32xf32) <- (9x24x49x49xf32, 9x24x49x32xf32)
        matmul_71 = paddle._C_ops.matmul(softmax_11, slice_46, False, False)
        del slice_46, softmax_11

        # pd_op.transpose: (9x49x24x32xf32) <- (9x24x49x32xf32)
        transpose_105 = paddle._C_ops.transpose(matmul_71, [0, 2, 1, 3])
        del matmul_71

        # pd_op.reshape: (9x49x768xf32) <- (9x49x24x32xf32, 3xi64)
        reshape_174 = paddle._C_ops.reshape(transpose_105, full_int_array_83)
        del full_int_array_83, transpose_105

        # pd_op.matmul: (9x49x768xf32) <- (9x49x768xf32, 768x768xf32)
        matmul_72 = paddle._C_ops.matmul(reshape_174, parameter_9, False, False)
        del parameter_9, reshape_174

        # pd_op.add: (9x49x768xf32) <- (9x49x768xf32, 768xf32)
        add_86 = paddle._C_ops.add(matmul_72, parameter_8)
        del matmul_72, parameter_8

        # pd_op.reshape: (9x7x7x768xf32) <- (9x49x768xf32, 4xi64)
        reshape_175 = paddle._C_ops.reshape(add_86, full_int_array_80)
        del add_86, full_int_array_80

        # pd_op.reshape: (1x3x3x7x7x768xf32) <- (9x7x7x768xf32, 6xi64)
        reshape_176 = paddle._C_ops.reshape(reshape_175, full_int_array_84)
        del full_int_array_84, reshape_175

        # pd_op.transpose: (1x3x7x3x7x768xf32) <- (1x3x3x7x7x768xf32)
        transpose_106 = paddle._C_ops.transpose(reshape_176, [0, 1, 3, 2, 4, 5])
        del reshape_176

        # pd_op.reshape: (1x21x21x768xf32) <- (1x3x7x3x7x768xf32, 4xi64)
        reshape_177 = paddle._C_ops.reshape(transpose_106, full_int_array_85)
        del full_int_array_85, transpose_106

        # pd_op.roll: (1x21x21x768xf32) <- (1x21x21x768xf32, 2xi64)
        roll_11 = paddle._C_ops.roll(reshape_177, full_int_array_39, [1, 2])
        del full_int_array_39, reshape_177

        # pd_op.slice: (1x16x16x768xf32) <- (1x21x21x768xf32, 2xi64, 2xi64)
        slice_47 = paddle._C_ops.slice(
            roll_11, [1, 2], full_int_array_2, full_int_array_86, [1, 1], []
        )
        del full_int_array_2, full_int_array_86, roll_11

        # pd_op.reshape: (1x256x768xf32) <- (1x16x16x768xf32, 3xi64)
        reshape_178 = paddle._C_ops.reshape(slice_47, full_int_array_87)
        del full_int_array_87, slice_47

        # pd_op.add: (1x256x768xf32) <- (1x256x768xf32, 1x256x768xf32)
        add_87 = paddle._C_ops.add(add_82, reshape_178)
        del add_82, reshape_178

        # pd_op.layer_norm: (1x256x768xf32, 1x256xf32, 1x256xf32) <- (1x256x768xf32, 768xf32, 768xf32)
        layer_norm_90, layer_norm_91, layer_norm_92 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_87, parameter_7, parameter_6, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_6, parameter_7

        # pd_op.matmul: (1x256x3072xf32) <- (1x256x768xf32, 768x3072xf32)
        matmul_73 = paddle._C_ops.matmul(layer_norm_90, parameter_5, False, False)
        del layer_norm_90, parameter_5

        # pd_op.add: (1x256x3072xf32) <- (1x256x3072xf32, 3072xf32)
        add_88 = paddle._C_ops.add(matmul_73, parameter_4)
        del matmul_73, parameter_4

        # pd_op.gelu: (1x256x3072xf32) <- (1x256x3072xf32)
        gelu_11 = paddle._C_ops.gelu(add_88, False)
        del add_88

        # pd_op.matmul: (1x256x768xf32) <- (1x256x3072xf32, 3072x768xf32)
        matmul_74 = paddle._C_ops.matmul(gelu_11, parameter_3, False, False)
        del gelu_11, parameter_3

        # pd_op.add: (1x256x768xf32) <- (1x256x768xf32, 768xf32)
        add_89 = paddle._C_ops.add(matmul_74, parameter_2)
        del matmul_74, parameter_2

        # pd_op.add: (1x256x768xf32) <- (1x256x768xf32, 1x256x768xf32)
        add_90 = paddle._C_ops.add(add_87, add_89)
        del add_87, add_89

        # pd_op.layer_norm: (1x256x768xf32, 1x256xf32, 1x256xf32) <- (1x256x768xf32, 768xf32, 768xf32)
        layer_norm_93, layer_norm_94, layer_norm_95 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_90, parameter_1, parameter_0, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_90, parameter_0, parameter_1

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_90 = [-1, 16, 16, 768]

        # pd_op.reshape: (1x16x16x768xf32) <- (1x256x768xf32, 4xi64)
        reshape_179 = paddle._C_ops.reshape(layer_norm_93, full_int_array_90)
        del full_int_array_90, layer_norm_93

        # pd_op.transpose: (1x768x16x16xf32) <- (1x16x16x768xf32)
        transpose_3 = paddle._C_ops.transpose(reshape_179, [0, 3, 1, 2])
        del reshape_179, set_value__17, set_value__26, set_value__35, set_value__8

        return transpose_0, transpose_1, transpose_2, transpose_3
