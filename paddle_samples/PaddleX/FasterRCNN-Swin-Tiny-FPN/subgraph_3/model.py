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
        parameter_165,
        parameter_166,
        parameter_167,
        parameter_168,
        parameter_169,
        parameter_170,
        parameter_171,
        parameter_172,
        parameter_173,
        parameter_174,
        parameter_175,
        parameter_176,
        parameter_177,
        parameter_178,
        parameter_179,
        parameter_180,
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
        # pd_op.conv2d: (2x96x240x176xf32) <- (2x3x960x704xf32, 96x3x4x4xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_12, parameter_180, [4, 4], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_12, parameter_180

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_179, full_int_array_0)
        del parameter_179

        # pd_op.add: (2x96x240x176xf32) <- (2x96x240x176xf32, 1x96x1x1xf32)
        add_3 = paddle._C_ops.add(conv2d_0, reshape_0)

        # pd_op.flatten: (2x96x42240xf32) <- (2x96x240x176xf32)
        flatten_0 = paddle._C_ops.flatten(add_3, 2, 3)

        # pd_op.transpose: (2x42240x96xf32) <- (2x96x42240xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_0, [0, 2, 1])
        del flatten_0

        # pd_op.layer_norm: (2x42240x96xf32, 2x42240xf32, 2x42240xf32) <- (2x42240x96xf32, 96xf32, 96xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_0, parameter_178, parameter_177, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_177, parameter_178

        # pd_op.transpose: (2x96x42240xf32) <- (2x42240x96xf32)
        transpose_1 = paddle._C_ops.transpose(layer_norm_0, [0, 2, 1])
        del layer_norm_0

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [-1, 96, 240, 176]

        # pd_op.reshape: (2x96x240x176xf32) <- (2x96x42240xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(transpose_1, full_int_array_1)
        del full_int_array_1

        # pd_op.flatten: (2x96x42240xf32) <- (2x96x240x176xf32)
        flatten_1 = paddle._C_ops.flatten(reshape_1, 2, 3)

        # pd_op.transpose: (2x42240x96xf32) <- (2x96x42240xf32)
        transpose_2 = paddle._C_ops.transpose(flatten_1, [0, 2, 1])
        del flatten_1

        # pd_op.full: (1x245x182x1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1, 245, 182, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [0, 0]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_0 = full_int_array_2

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_1 = full_int_array_2

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_2 = full_int_array_2

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_3 = full_int_array_2

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_4 = full_int_array_2

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_5 = full_int_array_2

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_6 = full_int_array_2

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_7 = full_int_array_2

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_8 = full_int_array_2

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_9 = full_int_array_2

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_10 = full_int_array_2

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_11 = full_int_array_2

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_12 = full_int_array_2

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_13 = full_int_array_2

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_14 = full_int_array_2

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [-7, -7]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [1, 1]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_15 = full_int_array_4

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_16 = full_int_array_4

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_17 = full_int_array_4

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_18 = full_int_array_4

        # pd_op.set_value_: (1x245x182x1xf32) <- (1x245x182x1xf32, 2xi64, 2xi64, 2xi64)
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

        # pd_op.set_value_: (1x245x182x1xf32) <- (1x245x182x1xf32, 2xi64, 2xi64, 2xi64)
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

        # pd_op.set_value_: (1x245x182x1xf32) <- (1x245x182x1xf32, 2xi64, 2xi64, 2xi64)
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

        # pd_op.set_value_: (1x245x182x1xf32) <- (1x245x182x1xf32, 2xi64, 2xi64, 2xi64)
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

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_19 = full_int_array_11

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_20 = full_int_array_11

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_21 = full_int_array_11

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_22 = full_int_array_11

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_23 = full_int_array_11

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_24 = full_int_array_11

        # pd_op.set_value_: (1x245x182x1xf32) <- (1x245x182x1xf32, 2xi64, 2xi64, 2xi64)
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

        # pd_op.set_value_: (1x245x182x1xf32) <- (1x245x182x1xf32, 2xi64, 2xi64, 2xi64)
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

        # pd_op.set_value_: (1x245x182x1xf32) <- (1x245x182x1xf32, 2xi64, 2xi64, 2xi64)
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

        # pd_op.set_value_: (1x245x182x1xf32) <- (1x245x182x1xf32, 2xi64, 2xi64, 2xi64)
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

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_25 = full_int_array_16

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_26 = full_int_array_16

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_27 = full_int_array_16

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_28 = full_int_array_16

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_29 = full_int_array_16

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_30 = full_int_array_16

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_31 = full_int_array_16

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_32 = full_int_array_16

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_33 = full_int_array_16

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_34 = full_int_array_16

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_35 = full_int_array_16

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_36 = full_int_array_16

        # pd_op.set_value_: (1x245x182x1xf32) <- (1x245x182x1xf32, 2xi64, 2xi64, 2xi64)
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
        full_int_array_17 = [-1, 35, 7, 26, 7, 1]

        # pd_op.reshape: (1x35x7x26x7x1xf32) <- (1x245x182x1xf32, 6xi64)
        reshape_2 = paddle._C_ops.reshape(set_value__8, full_int_array_17)
        del full_int_array_17

        # pd_op.transpose: (1x35x26x7x7x1xf32) <- (1x35x7x26x7x1xf32)
        transpose_3 = paddle._C_ops.transpose(reshape_2, [0, 1, 3, 2, 4, 5])
        del reshape_2

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_18 = [-1, 7, 7, 1]

        # pd_op.reshape: (910x7x7x1xf32) <- (1x35x26x7x7x1xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_3, full_int_array_18)
        del transpose_3

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_19 = [-1, 49]

        # pd_op.reshape: (910x49xf32) <- (910x7x7x1xf32, 2xi64)
        reshape_4 = paddle._C_ops.reshape(reshape_3, full_int_array_19)
        del reshape_3

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_20 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_37 = full_int_array_20

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_38 = full_int_array_20

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_39 = full_int_array_20

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_40 = full_int_array_20

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_41 = full_int_array_20

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_42 = full_int_array_20

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_43 = full_int_array_20

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_44 = full_int_array_20

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_45 = full_int_array_20

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_46 = full_int_array_20

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_47 = full_int_array_20

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_48 = full_int_array_20

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_49 = full_int_array_20

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_50 = full_int_array_20

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_51 = full_int_array_20

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_52 = full_int_array_20

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_53 = full_int_array_20

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_54 = full_int_array_20

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_55 = full_int_array_20

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_56 = full_int_array_20

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_57 = full_int_array_20

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_58 = full_int_array_20

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_59 = full_int_array_20

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_60 = full_int_array_20

        # pd_op.unsqueeze: (910x1x49xf32) <- (910x49xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(reshape_4, full_int_array_20)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_21 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_61 = full_int_array_21

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_62 = full_int_array_21

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_63 = full_int_array_21

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_64 = full_int_array_21

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_65 = full_int_array_21

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_66 = full_int_array_21

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_67 = full_int_array_21

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_68 = full_int_array_21

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_69 = full_int_array_21

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_70 = full_int_array_21

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_71 = full_int_array_21

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_72 = full_int_array_21

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_73 = full_int_array_21

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_74 = full_int_array_21

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_75 = full_int_array_21

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_76 = full_int_array_21

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_77 = full_int_array_21

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_78 = full_int_array_21

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_79 = full_int_array_21

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_80 = full_int_array_21

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_81 = full_int_array_21

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_82 = full_int_array_21

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_83 = full_int_array_21

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_84 = full_int_array_21

        # pd_op.unsqueeze: (910x49x1xf32) <- (910x49xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(reshape_4, full_int_array_21)
        del reshape_4

        # pd_op.subtract: (910x49x49xf32) <- (910x1x49xf32, 910x49x1xf32)
        subtract_0 = paddle._C_ops.subtract(unsqueeze_0, unsqueeze_1)
        del unsqueeze_0, unsqueeze_1

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (910x49x49xf32) <- (910x49x49xf32, 1xf32)
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

        # pd_op.scale: (910x49x49xf32) <- (910x49x49xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(full_like_0, full_2, float("0"), True)
        del full_like_0

        # pd_op.full: (xf32) <- ()
        full_3 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.not_equal: (910x49x49xb) <- (910x49x49xf32, xf32)
        not_equal_0 = paddle._C_ops.not_equal(subtract_0, full_3)
        del subtract_0

        # pd_op.cast: (910x49x49xf32) <- (910x49x49xb)
        cast_0 = paddle._C_ops.cast(not_equal_0, paddle.float32)
        del not_equal_0

        # pd_op.multiply: (910x49x49xf32) <- (910x49x49xf32, 910x49x49xf32)
        multiply_0 = paddle._C_ops.multiply(scale_0, cast_0)
        del cast_0, scale_0

        # pd_op.layer_norm: (2x42240x96xf32, 2x42240xf32, 2x42240xf32) <- (2x42240x96xf32, 96xf32, 96xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_2, parameter_176, parameter_175, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_175, parameter_176

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_22 = [-1, 240, 176, 96]

        # pd_op.reshape: (2x240x176x96xf32) <- (2x42240x96xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(layer_norm_3, full_int_array_22)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_85 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_86 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_87 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_88 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_89 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_90 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_91 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_92 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_93 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_94 = full_4

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_95 = full_4

        # pd_op.pad: (2x245x182x96xf32) <- (2x240x176x96xf32, 1xf32)
        pad_0 = paddle._C_ops.pad(reshape_5, [0, 0, 0, 5, 0, 6, 0, 0], full_4)

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_23 = [-1, 35, 7, 26, 7, 96]

        # pd_op.reshape: (2x35x7x26x7x96xf32) <- (2x245x182x96xf32, 6xi64)
        reshape_6 = paddle._C_ops.reshape(pad_0, full_int_array_23)

        # pd_op.transpose: (2x35x26x7x7x96xf32) <- (2x35x7x26x7x96xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_6, [0, 1, 3, 2, 4, 5])
        del reshape_6

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_24 = [-1, 7, 7, 96]

        # pd_op.reshape: (1820x7x7x96xf32) <- (2x35x26x7x7x96xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_4, full_int_array_24)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_25 = [1820, 49, 96]

        # pd_op.reshape: (1820x49x96xf32) <- (1820x7x7x96xf32, 3xi64)
        reshape_8 = paddle._C_ops.reshape(reshape_7, full_int_array_25)

        # pd_op.matmul: (1820x49x288xf32) <- (1820x49x96xf32, 96x288xf32)
        matmul_0 = paddle._C_ops.matmul(reshape_8, parameter_174, False, False)
        del parameter_174

        # pd_op.add: (1820x49x288xf32) <- (1820x49x288xf32, 288xf32)
        add_4 = paddle._C_ops.add(matmul_0, parameter_173)
        del parameter_173

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_26 = [-1, 49, 3, 3, 32]

        # pd_op.reshape: (1820x49x3x3x32xf32) <- (1820x49x288xf32, 5xi64)
        reshape_9 = paddle._C_ops.reshape(add_4, full_int_array_26)

        # pd_op.transpose: (3x1820x3x49x32xf32) <- (1820x49x3x3x32xf32)
        transpose_5 = paddle._C_ops.transpose(reshape_9, [2, 0, 3, 1, 4])
        del reshape_9

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_27 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_96 = full_int_array_27

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_97 = full_int_array_27

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_98 = full_int_array_27

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_99 = full_int_array_27

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_100 = full_int_array_27

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_101 = full_int_array_27

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_102 = full_int_array_27

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_103 = full_int_array_27

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_104 = full_int_array_27

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_105 = full_int_array_27

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_106 = full_int_array_27

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_107 = full_int_array_27

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_108 = full_int_array_27

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_109 = full_int_array_27

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_110 = full_int_array_27

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_111 = full_int_array_27

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_112 = full_int_array_27

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_113 = full_int_array_27

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_114 = full_int_array_27

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_115 = full_int_array_27

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_116 = full_int_array_27

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_117 = full_int_array_27

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_118 = full_int_array_27

        # pd_op.slice: (1820x3x49x32xf32) <- (3x1820x3x49x32xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            transpose_5, [0], full_int_array_27, full_int_array_20, [1], [0]
        )

        # pd_op.slice: (1820x3x49x32xf32) <- (3x1820x3x49x32xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            transpose_5, [0], full_int_array_20, full_int_array_21, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_28 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_119 = full_int_array_28

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_120 = full_int_array_28

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_121 = full_int_array_28

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_122 = full_int_array_28

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_123 = full_int_array_28

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_124 = full_int_array_28

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_125 = full_int_array_28

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_126 = full_int_array_28

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_127 = full_int_array_28

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_128 = full_int_array_28

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_129 = full_int_array_28

        # pd_op.slice: (1820x3x49x32xf32) <- (3x1820x3x49x32xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            transpose_5, [0], full_int_array_21, full_int_array_28, [1], [0]
        )

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_130 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_131 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_132 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_133 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_134 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_135 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_136 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_137 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_138 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_139 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_140 = full_5

        # pd_op.scale: (1820x3x49x32xf32) <- (1820x3x49x32xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_0, full_5, float("0"), True)
        del slice_0

        # pd_op.transpose: (1820x3x32x49xf32) <- (1820x3x49x32xf32)
        transpose_6 = paddle._C_ops.transpose(slice_1, [0, 1, 3, 2])
        del slice_1

        # pd_op.matmul: (1820x3x49x49xf32) <- (1820x3x49x32xf32, 1820x3x32x49xf32)
        matmul_1 = paddle._C_ops.matmul(scale_1, transpose_6, False, False)

        # pd_op.flatten: (2401xi64) <- (49x49xi64)
        flatten_2 = paddle._C_ops.flatten(data_13, 0, 1)
        del data_13

        # pd_op.index_select: (2401x3xf32) <- (169x3xf32, 2401xi64)
        index_select_0 = paddle._C_ops.index_select(data_0, flatten_2, 0)
        del data_0

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_29 = [49, 49, -1]

        # pd_op.reshape: (49x49x3xf32) <- (2401x3xf32, 3xi64)
        reshape_10 = paddle._C_ops.reshape(index_select_0, full_int_array_29)

        # pd_op.transpose: (3x49x49xf32) <- (49x49x3xf32)
        transpose_7 = paddle._C_ops.transpose(reshape_10, [2, 0, 1])
        del reshape_10

        # pd_op.unsqueeze: (1x3x49x49xf32) <- (3x49x49xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(transpose_7, full_int_array_27)

        # pd_op.add: (1820x3x49x49xf32) <- (1820x3x49x49xf32, 1x3x49x49xf32)
        add_5 = paddle._C_ops.add(matmul_1, unsqueeze_2)

        # pd_op.softmax: (1820x3x49x49xf32) <- (1820x3x49x49xf32)
        softmax_0 = paddle._C_ops.softmax(add_5, -1)
        del add_5

        # pd_op.matmul: (1820x3x49x32xf32) <- (1820x3x49x49xf32, 1820x3x49x32xf32)
        matmul_2 = paddle._C_ops.matmul(softmax_0, slice_2, False, False)

        # pd_op.transpose: (1820x49x3x32xf32) <- (1820x3x49x32xf32)
        transpose_8 = paddle._C_ops.transpose(matmul_2, [0, 2, 1, 3])
        del matmul_2

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_30 = [-1, 49, 96]

        # pd_op.reshape: (1820x49x96xf32) <- (1820x49x3x32xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_8, full_int_array_30)

        # pd_op.matmul: (1820x49x96xf32) <- (1820x49x96xf32, 96x96xf32)
        matmul_3 = paddle._C_ops.matmul(reshape_11, parameter_172, False, False)
        del parameter_172

        # pd_op.add: (1820x49x96xf32) <- (1820x49x96xf32, 96xf32)
        add_6 = paddle._C_ops.add(matmul_3, parameter_171)
        del parameter_171

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_31 = [1820, 7, 7, 96]

        # pd_op.reshape: (1820x7x7x96xf32) <- (1820x49x96xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(add_6, full_int_array_31)

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_32 = [-1, 35, 26, 7, 7, 96]

        # pd_op.reshape: (2x35x26x7x7x96xf32) <- (1820x7x7x96xf32, 6xi64)
        reshape_13 = paddle._C_ops.reshape(reshape_12, full_int_array_32)

        # pd_op.transpose: (2x35x7x26x7x96xf32) <- (2x35x26x7x7x96xf32)
        transpose_9 = paddle._C_ops.transpose(reshape_13, [0, 1, 3, 2, 4, 5])
        del reshape_13

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_33 = [-1, 245, 182, 96]

        # pd_op.reshape: (2x245x182x96xf32) <- (2x35x7x26x7x96xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(transpose_9, full_int_array_33)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_34 = [240, 176]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_141 = full_int_array_34

        # pd_op.slice: (2x240x176x96xf32) <- (2x245x182x96xf32, 2xi64, 2xi64)
        slice_3 = paddle._C_ops.slice(
            reshape_14, [1, 2], full_int_array_2, full_int_array_34, [1, 1], []
        )

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_35 = [-1, 42240, 96]

        # pd_op.reshape: (2x42240x96xf32) <- (2x240x176x96xf32, 3xi64)
        reshape_15 = paddle._C_ops.reshape(slice_3, full_int_array_35)

        # pd_op.add: (2x42240x96xf32) <- (2x42240x96xf32, 2x42240x96xf32)
        add_7 = paddle._C_ops.add(transpose_2, reshape_15)

        # pd_op.layer_norm: (2x42240x96xf32, 2x42240xf32, 2x42240xf32) <- (2x42240x96xf32, 96xf32, 96xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_7, parameter_170, parameter_169, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_169, parameter_170

        # pd_op.matmul: (2x42240x384xf32) <- (2x42240x96xf32, 96x384xf32)
        matmul_4 = paddle._C_ops.matmul(layer_norm_6, parameter_168, False, False)
        del parameter_168

        # pd_op.add: (2x42240x384xf32) <- (2x42240x384xf32, 384xf32)
        add_8 = paddle._C_ops.add(matmul_4, parameter_167)
        del parameter_167

        # pd_op.gelu: (2x42240x384xf32) <- (2x42240x384xf32)
        gelu_0 = paddle._C_ops.gelu(add_8, False)

        # pd_op.matmul: (2x42240x96xf32) <- (2x42240x384xf32, 384x96xf32)
        matmul_5 = paddle._C_ops.matmul(gelu_0, parameter_166, False, False)
        del parameter_166

        # pd_op.add: (2x42240x96xf32) <- (2x42240x96xf32, 96xf32)
        add_9 = paddle._C_ops.add(matmul_5, parameter_165)
        del parameter_165

        # pd_op.add: (2x42240x96xf32) <- (2x42240x96xf32, 2x42240x96xf32)
        add_10 = paddle._C_ops.add(add_7, add_9)

        # pd_op.layer_norm: (2x42240x96xf32, 2x42240xf32, 2x42240xf32) <- (2x42240x96xf32, 96xf32, 96xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_10, parameter_164, parameter_163, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_163, parameter_164

        # pd_op.reshape: (2x240x176x96xf32) <- (2x42240x96xf32, 4xi64)
        reshape_16 = paddle._C_ops.reshape(layer_norm_9, full_int_array_22)

        # pd_op.pad: (2x245x182x96xf32) <- (2x240x176x96xf32, 1xf32)
        pad_1 = paddle._C_ops.pad(reshape_16, [0, 0, 0, 5, 0, 6, 0, 0], full_4)

        # pd_op.roll: (2x245x182x96xf32) <- (2x245x182x96xf32, 2xi64)
        roll_0 = paddle._C_ops.roll(pad_1, full_int_array_11, [1, 2])

        # pd_op.reshape: (2x35x7x26x7x96xf32) <- (2x245x182x96xf32, 6xi64)
        reshape_17 = paddle._C_ops.reshape(roll_0, full_int_array_23)
        del full_int_array_23

        # pd_op.transpose: (2x35x26x7x7x96xf32) <- (2x35x7x26x7x96xf32)
        transpose_10 = paddle._C_ops.transpose(reshape_17, [0, 1, 3, 2, 4, 5])
        del reshape_17

        # pd_op.reshape: (1820x7x7x96xf32) <- (2x35x26x7x7x96xf32, 4xi64)
        reshape_18 = paddle._C_ops.reshape(transpose_10, full_int_array_24)
        del full_int_array_24

        # pd_op.reshape: (1820x49x96xf32) <- (1820x7x7x96xf32, 3xi64)
        reshape_19 = paddle._C_ops.reshape(reshape_18, full_int_array_25)
        del full_int_array_25

        # pd_op.matmul: (1820x49x288xf32) <- (1820x49x96xf32, 96x288xf32)
        matmul_6 = paddle._C_ops.matmul(reshape_19, parameter_162, False, False)
        del parameter_162

        # pd_op.add: (1820x49x288xf32) <- (1820x49x288xf32, 288xf32)
        add_11 = paddle._C_ops.add(matmul_6, parameter_161)
        del parameter_161

        # pd_op.reshape: (1820x49x3x3x32xf32) <- (1820x49x288xf32, 5xi64)
        reshape_20 = paddle._C_ops.reshape(add_11, full_int_array_26)
        del full_int_array_26

        # pd_op.transpose: (3x1820x3x49x32xf32) <- (1820x49x3x3x32xf32)
        transpose_11 = paddle._C_ops.transpose(reshape_20, [2, 0, 3, 1, 4])
        del reshape_20

        # pd_op.slice: (1820x3x49x32xf32) <- (3x1820x3x49x32xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            transpose_11, [0], full_int_array_27, full_int_array_20, [1], [0]
        )

        # pd_op.slice: (1820x3x49x32xf32) <- (3x1820x3x49x32xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            transpose_11, [0], full_int_array_20, full_int_array_21, [1], [0]
        )

        # pd_op.slice: (1820x3x49x32xf32) <- (3x1820x3x49x32xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            transpose_11, [0], full_int_array_21, full_int_array_28, [1], [0]
        )

        # pd_op.scale: (1820x3x49x32xf32) <- (1820x3x49x32xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_4, full_5, float("0"), True)
        del slice_4

        # pd_op.transpose: (1820x3x32x49xf32) <- (1820x3x49x32xf32)
        transpose_12 = paddle._C_ops.transpose(slice_5, [0, 1, 3, 2])
        del slice_5

        # pd_op.matmul: (1820x3x49x49xf32) <- (1820x3x49x32xf32, 1820x3x32x49xf32)
        matmul_7 = paddle._C_ops.matmul(scale_2, transpose_12, False, False)

        # pd_op.flatten: (2401xi64) <- (49x49xi64)
        flatten_3 = paddle._C_ops.flatten(data_14, 0, 1)
        del data_14

        # pd_op.index_select: (2401x3xf32) <- (169x3xf32, 2401xi64)
        index_select_1 = paddle._C_ops.index_select(data_1, flatten_3, 0)
        del data_1

        # pd_op.reshape: (49x49x3xf32) <- (2401x3xf32, 3xi64)
        reshape_21 = paddle._C_ops.reshape(index_select_1, full_int_array_29)

        # pd_op.transpose: (3x49x49xf32) <- (49x49x3xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_21, [2, 0, 1])
        del reshape_21

        # pd_op.unsqueeze: (1x3x49x49xf32) <- (3x49x49xf32, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(transpose_13, full_int_array_27)

        # pd_op.add: (1820x3x49x49xf32) <- (1820x3x49x49xf32, 1x3x49x49xf32)
        add_12 = paddle._C_ops.add(matmul_7, unsqueeze_3)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_36 = [-1, 910, 3, 49, 49]

        # pd_op.reshape: (2x910x3x49x49xf32) <- (1820x3x49x49xf32, 5xi64)
        reshape_22 = paddle._C_ops.reshape(add_12, full_int_array_36)
        del full_int_array_36

        # pd_op.unsqueeze: (910x1x49x49xf32) <- (910x49x49xf32, 1xi64)
        unsqueeze_4 = paddle._C_ops.unsqueeze(multiply_0, full_int_array_20)
        del multiply_0

        # pd_op.unsqueeze: (1x910x1x49x49xf32) <- (910x1x49x49xf32, 1xi64)
        unsqueeze_5 = paddle._C_ops.unsqueeze(unsqueeze_4, full_int_array_27)
        del unsqueeze_4

        # pd_op.add: (2x910x3x49x49xf32) <- (2x910x3x49x49xf32, 1x910x1x49x49xf32)
        add_13 = paddle._C_ops.add(reshape_22, unsqueeze_5)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_37 = [-1, 3, 49, 49]

        # pd_op.reshape: (1820x3x49x49xf32) <- (2x910x3x49x49xf32, 4xi64)
        reshape_23 = paddle._C_ops.reshape(add_13, full_int_array_37)
        del full_int_array_37

        # pd_op.softmax: (1820x3x49x49xf32) <- (1820x3x49x49xf32)
        softmax_1 = paddle._C_ops.softmax(reshape_23, -1)
        del reshape_23

        # pd_op.matmul: (1820x3x49x32xf32) <- (1820x3x49x49xf32, 1820x3x49x32xf32)
        matmul_8 = paddle._C_ops.matmul(softmax_1, slice_6, False, False)

        # pd_op.transpose: (1820x49x3x32xf32) <- (1820x3x49x32xf32)
        transpose_14 = paddle._C_ops.transpose(matmul_8, [0, 2, 1, 3])
        del matmul_8

        # pd_op.reshape: (1820x49x96xf32) <- (1820x49x3x32xf32, 3xi64)
        reshape_24 = paddle._C_ops.reshape(transpose_14, full_int_array_30)
        del full_int_array_30

        # pd_op.matmul: (1820x49x96xf32) <- (1820x49x96xf32, 96x96xf32)
        matmul_9 = paddle._C_ops.matmul(reshape_24, parameter_160, False, False)
        del parameter_160

        # pd_op.add: (1820x49x96xf32) <- (1820x49x96xf32, 96xf32)
        add_14 = paddle._C_ops.add(matmul_9, parameter_159)
        del parameter_159

        # pd_op.reshape: (1820x7x7x96xf32) <- (1820x49x96xf32, 4xi64)
        reshape_25 = paddle._C_ops.reshape(add_14, full_int_array_31)
        del full_int_array_31

        # pd_op.reshape: (2x35x26x7x7x96xf32) <- (1820x7x7x96xf32, 6xi64)
        reshape_26 = paddle._C_ops.reshape(reshape_25, full_int_array_32)
        del full_int_array_32

        # pd_op.transpose: (2x35x7x26x7x96xf32) <- (2x35x26x7x7x96xf32)
        transpose_15 = paddle._C_ops.transpose(reshape_26, [0, 1, 3, 2, 4, 5])
        del reshape_26

        # pd_op.reshape: (2x245x182x96xf32) <- (2x35x7x26x7x96xf32, 4xi64)
        reshape_27 = paddle._C_ops.reshape(transpose_15, full_int_array_33)
        del full_int_array_33

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_38 = [3, 3]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_142 = full_int_array_38

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_143 = full_int_array_38

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_144 = full_int_array_38

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_145 = full_int_array_38

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_146 = full_int_array_38

        # pd_op.roll: (2x245x182x96xf32) <- (2x245x182x96xf32, 2xi64)
        roll_1 = paddle._C_ops.roll(reshape_27, full_int_array_38, [1, 2])

        # pd_op.slice: (2x240x176x96xf32) <- (2x245x182x96xf32, 2xi64, 2xi64)
        slice_7 = paddle._C_ops.slice(
            roll_1, [1, 2], full_int_array_2, full_int_array_34, [1, 1], []
        )

        # pd_op.reshape: (2x42240x96xf32) <- (2x240x176x96xf32, 3xi64)
        reshape_28 = paddle._C_ops.reshape(slice_7, full_int_array_35)
        del full_int_array_35

        # pd_op.full: (xf64) <- ()
        full_6 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_6,
            [],
            paddle.float64,
            [float("0.990909")],
            paddle.framework._current_expected_place(),
        )
        del full_6

        # pd_op.cast: (xf32) <- (xf64)
        cast_1 = paddle._C_ops.cast(assign_value__0, paddle.float32)
        del assign_value__0

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_39 = [2, 1, 1]

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_0 = paddle._C_ops.uniform(
            full_int_array_39,
            paddle.float32,
            full_4,
            full_1,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_15 = paddle._C_ops.add(cast_1, uniform_0)
        del uniform_0

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_0 = paddle._C_ops.floor(add_15)
        del add_15

        # pd_op.divide: (2x42240x96xf32) <- (2x42240x96xf32, xf32)
        divide_0 = paddle._C_ops.divide(reshape_28, cast_1)

        # pd_op.multiply: (2x42240x96xf32) <- (2x42240x96xf32, 2x1x1xf32)
        multiply_1 = paddle._C_ops.multiply(divide_0, floor_0)

        # pd_op.add: (2x42240x96xf32) <- (2x42240x96xf32, 2x42240x96xf32)
        add_16 = paddle._C_ops.add(add_10, multiply_1)

        # pd_op.layer_norm: (2x42240x96xf32, 2x42240xf32, 2x42240xf32) <- (2x42240x96xf32, 96xf32, 96xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_16, parameter_158, parameter_157, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_157, parameter_158

        # pd_op.matmul: (2x42240x384xf32) <- (2x42240x96xf32, 96x384xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_12, parameter_156, False, False)
        del parameter_156

        # pd_op.add: (2x42240x384xf32) <- (2x42240x384xf32, 384xf32)
        add_17 = paddle._C_ops.add(matmul_10, parameter_155)
        del parameter_155

        # pd_op.gelu: (2x42240x384xf32) <- (2x42240x384xf32)
        gelu_1 = paddle._C_ops.gelu(add_17, False)

        # pd_op.matmul: (2x42240x96xf32) <- (2x42240x384xf32, 384x96xf32)
        matmul_11 = paddle._C_ops.matmul(gelu_1, parameter_154, False, False)
        del parameter_154

        # pd_op.add: (2x42240x96xf32) <- (2x42240x96xf32, 96xf32)
        add_18 = paddle._C_ops.add(matmul_11, parameter_153)
        del parameter_153

        # pd_op.full: (xf64) <- ()
        full_7 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__1 = paddle._C_ops.assign_value_(
            full_7,
            [],
            paddle.float64,
            [float("0.990909")],
            paddle.framework._current_expected_place(),
        )
        del full_7

        # pd_op.cast: (xf32) <- (xf64)
        cast_2 = paddle._C_ops.cast(assign_value__1, paddle.float32)
        del assign_value__1

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_1 = paddle._C_ops.uniform(
            full_int_array_39,
            paddle.float32,
            full_4,
            full_1,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_19 = paddle._C_ops.add(cast_2, uniform_1)
        del uniform_1

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_1 = paddle._C_ops.floor(add_19)
        del add_19

        # pd_op.divide: (2x42240x96xf32) <- (2x42240x96xf32, xf32)
        divide_1 = paddle._C_ops.divide(add_18, cast_2)

        # pd_op.multiply: (2x42240x96xf32) <- (2x42240x96xf32, 2x1x1xf32)
        multiply_2 = paddle._C_ops.multiply(divide_1, floor_1)

        # pd_op.add: (2x42240x96xf32) <- (2x42240x96xf32, 2x42240x96xf32)
        add_20 = paddle._C_ops.add(add_16, multiply_2)

        # pd_op.reshape: (2x240x176x96xf32) <- (2x42240x96xf32, 4xi64)
        reshape_29 = paddle._C_ops.reshape(add_20, full_int_array_22)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_40 = [2, 2]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_147 = full_int_array_40

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_148 = full_int_array_40

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_149 = full_int_array_40

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_150 = full_int_array_40

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_151 = full_int_array_40

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_152 = full_int_array_40

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_153 = full_int_array_40

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_154 = full_int_array_40

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_155 = full_int_array_40

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_156 = full_int_array_40

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_157 = full_int_array_40

        # pd_op.strided_slice: (2x120x88x96xf32) <- (2x240x176x96xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_0 = paddle._C_ops.strided_slice(
            reshape_29, [1, 2], full_int_array_2, full_int_array_16, full_int_array_40
        )

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_41 = [1, 0]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_158 = full_int_array_41

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_159 = full_int_array_41

        # pd_op.strided_slice: (2x120x88x96xf32) <- (2x240x176x96xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_1 = paddle._C_ops.strided_slice(
            reshape_29, [1, 2], full_int_array_41, full_int_array_16, full_int_array_40
        )

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_42 = [0, 1]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_160 = full_int_array_42

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_161 = full_int_array_42

        # pd_op.strided_slice: (2x120x88x96xf32) <- (2x240x176x96xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_2 = paddle._C_ops.strided_slice(
            reshape_29, [1, 2], full_int_array_42, full_int_array_16, full_int_array_40
        )

        # pd_op.strided_slice: (2x120x88x96xf32) <- (2x240x176x96xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_3 = paddle._C_ops.strided_slice(
            reshape_29, [1, 2], full_int_array_4, full_int_array_16, full_int_array_40
        )

        # pd_op.full: (1xi32) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_162 = full_8

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_163 = full_8

        # builtin.combine: ([2x120x88x96xf32, 2x120x88x96xf32, 2x120x88x96xf32, 2x120x88x96xf32]) <- (2x120x88x96xf32, 2x120x88x96xf32, 2x120x88x96xf32, 2x120x88x96xf32)
        combine_0 = [strided_slice_0, strided_slice_1, strided_slice_2, strided_slice_3]

        # pd_op.concat: (2x120x88x384xf32) <- ([2x120x88x96xf32, 2x120x88x96xf32, 2x120x88x96xf32, 2x120x88x96xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_8)
        del combine_0

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_43 = [-1, 10560, 384]

        # pd_op.reshape: (2x10560x384xf32) <- (2x120x88x384xf32, 3xi64)
        reshape_30 = paddle._C_ops.reshape(concat_0, full_int_array_43)
        del full_int_array_43

        # pd_op.layer_norm: (2x10560x384xf32, 2x10560xf32, 2x10560xf32) <- (2x10560x384xf32, 384xf32, 384xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                reshape_30, parameter_152, parameter_151, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_151, parameter_152

        # pd_op.matmul: (2x10560x192xf32) <- (2x10560x384xf32, 384x192xf32)
        matmul_12 = paddle._C_ops.matmul(layer_norm_15, parameter_150, False, False)
        del parameter_150

        # pd_op.layer_norm: (2x42240x96xf32, 2x42240xf32, 2x42240xf32) <- (2x42240x96xf32, 96xf32, 96xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_20, parameter_149, parameter_148, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_148, parameter_149

        # pd_op.reshape: (2x240x176x96xf32) <- (2x42240x96xf32, 4xi64)
        reshape_31 = paddle._C_ops.reshape(layer_norm_18, full_int_array_22)
        del full_int_array_22

        # pd_op.transpose: (2x96x240x176xf32) <- (2x240x176x96xf32)
        transpose_16 = paddle._C_ops.transpose(reshape_31, [0, 3, 1, 2])
        del reshape_31

        # pd_op.full: (1x126x91x1xf32) <- ()
        full_9 = paddle._C_ops.full(
            [1, 126, 91, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x126x91x1xf32) <- (1x126x91x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__9 = paddle._C_ops.set_value_(
            full_9,
            full_int_array_2,
            full_int_array_3,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("0")],
        )
        del full_9

        # pd_op.set_value_: (1x126x91x1xf32) <- (1x126x91x1xf32, 2xi64, 2xi64, 2xi64)
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

        # pd_op.set_value_: (1x126x91x1xf32) <- (1x126x91x1xf32, 2xi64, 2xi64, 2xi64)
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

        # pd_op.set_value_: (1x126x91x1xf32) <- (1x126x91x1xf32, 2xi64, 2xi64, 2xi64)
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

        # pd_op.set_value_: (1x126x91x1xf32) <- (1x126x91x1xf32, 2xi64, 2xi64, 2xi64)
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

        # pd_op.set_value_: (1x126x91x1xf32) <- (1x126x91x1xf32, 2xi64, 2xi64, 2xi64)
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

        # pd_op.set_value_: (1x126x91x1xf32) <- (1x126x91x1xf32, 2xi64, 2xi64, 2xi64)
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

        # pd_op.set_value_: (1x126x91x1xf32) <- (1x126x91x1xf32, 2xi64, 2xi64, 2xi64)
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

        # pd_op.set_value_: (1x126x91x1xf32) <- (1x126x91x1xf32, 2xi64, 2xi64, 2xi64)
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
        full_int_array_44 = [-1, 18, 7, 13, 7, 1]

        # pd_op.reshape: (1x18x7x13x7x1xf32) <- (1x126x91x1xf32, 6xi64)
        reshape_32 = paddle._C_ops.reshape(set_value__17, full_int_array_44)
        del full_int_array_44

        # pd_op.transpose: (1x18x13x7x7x1xf32) <- (1x18x7x13x7x1xf32)
        transpose_17 = paddle._C_ops.transpose(reshape_32, [0, 1, 3, 2, 4, 5])
        del reshape_32

        # pd_op.reshape: (234x7x7x1xf32) <- (1x18x13x7x7x1xf32, 4xi64)
        reshape_33 = paddle._C_ops.reshape(transpose_17, full_int_array_18)
        del transpose_17

        # pd_op.reshape: (234x49xf32) <- (234x7x7x1xf32, 2xi64)
        reshape_34 = paddle._C_ops.reshape(reshape_33, full_int_array_19)
        del reshape_33

        # pd_op.unsqueeze: (234x1x49xf32) <- (234x49xf32, 1xi64)
        unsqueeze_6 = paddle._C_ops.unsqueeze(reshape_34, full_int_array_20)

        # pd_op.unsqueeze: (234x49x1xf32) <- (234x49xf32, 1xi64)
        unsqueeze_7 = paddle._C_ops.unsqueeze(reshape_34, full_int_array_21)
        del reshape_34

        # pd_op.subtract: (234x49x49xf32) <- (234x1x49xf32, 234x49x1xf32)
        subtract_1 = paddle._C_ops.subtract(unsqueeze_6, unsqueeze_7)
        del unsqueeze_6, unsqueeze_7

        # pd_op.full_like: (234x49x49xf32) <- (234x49x49xf32, 1xf32)
        full_like_1 = paddle._C_ops.full_like(
            subtract_1,
            full_1,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.scale: (234x49x49xf32) <- (234x49x49xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(full_like_1, full_2, float("0"), True)
        del full_like_1

        # pd_op.not_equal: (234x49x49xb) <- (234x49x49xf32, xf32)
        not_equal_1 = paddle._C_ops.not_equal(subtract_1, full_3)
        del subtract_1

        # pd_op.cast: (234x49x49xf32) <- (234x49x49xb)
        cast_3 = paddle._C_ops.cast(not_equal_1, paddle.float32)
        del not_equal_1

        # pd_op.multiply: (234x49x49xf32) <- (234x49x49xf32, 234x49x49xf32)
        multiply_3 = paddle._C_ops.multiply(scale_3, cast_3)
        del cast_3, scale_3

        # pd_op.layer_norm: (2x10560x192xf32, 2x10560xf32, 2x10560xf32) <- (2x10560x192xf32, 192xf32, 192xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                matmul_12, parameter_147, parameter_146, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_146, parameter_147

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_45 = [-1, 120, 88, 192]

        # pd_op.reshape: (2x120x88x192xf32) <- (2x10560x192xf32, 4xi64)
        reshape_35 = paddle._C_ops.reshape(layer_norm_21, full_int_array_45)

        # pd_op.pad: (2x126x91x192xf32) <- (2x120x88x192xf32, 1xf32)
        pad_2 = paddle._C_ops.pad(reshape_35, [0, 0, 0, 6, 0, 3, 0, 0], full_4)

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_46 = [-1, 18, 7, 13, 7, 192]

        # pd_op.reshape: (2x18x7x13x7x192xf32) <- (2x126x91x192xf32, 6xi64)
        reshape_36 = paddle._C_ops.reshape(pad_2, full_int_array_46)

        # pd_op.transpose: (2x18x13x7x7x192xf32) <- (2x18x7x13x7x192xf32)
        transpose_18 = paddle._C_ops.transpose(reshape_36, [0, 1, 3, 2, 4, 5])
        del reshape_36

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_47 = [-1, 7, 7, 192]

        # pd_op.reshape: (468x7x7x192xf32) <- (2x18x13x7x7x192xf32, 4xi64)
        reshape_37 = paddle._C_ops.reshape(transpose_18, full_int_array_47)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_48 = [468, 49, 192]

        # pd_op.reshape: (468x49x192xf32) <- (468x7x7x192xf32, 3xi64)
        reshape_38 = paddle._C_ops.reshape(reshape_37, full_int_array_48)

        # pd_op.matmul: (468x49x576xf32) <- (468x49x192xf32, 192x576xf32)
        matmul_13 = paddle._C_ops.matmul(reshape_38, parameter_145, False, False)
        del parameter_145

        # pd_op.add: (468x49x576xf32) <- (468x49x576xf32, 576xf32)
        add_21 = paddle._C_ops.add(matmul_13, parameter_144)
        del parameter_144

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_49 = [-1, 49, 3, 6, 32]

        # pd_op.reshape: (468x49x3x6x32xf32) <- (468x49x576xf32, 5xi64)
        reshape_39 = paddle._C_ops.reshape(add_21, full_int_array_49)

        # pd_op.transpose: (3x468x6x49x32xf32) <- (468x49x3x6x32xf32)
        transpose_19 = paddle._C_ops.transpose(reshape_39, [2, 0, 3, 1, 4])
        del reshape_39

        # pd_op.slice: (468x6x49x32xf32) <- (3x468x6x49x32xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            transpose_19, [0], full_int_array_27, full_int_array_20, [1], [0]
        )

        # pd_op.slice: (468x6x49x32xf32) <- (3x468x6x49x32xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            transpose_19, [0], full_int_array_20, full_int_array_21, [1], [0]
        )

        # pd_op.slice: (468x6x49x32xf32) <- (3x468x6x49x32xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            transpose_19, [0], full_int_array_21, full_int_array_28, [1], [0]
        )

        # pd_op.scale: (468x6x49x32xf32) <- (468x6x49x32xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(slice_8, full_5, float("0"), True)
        del slice_8

        # pd_op.transpose: (468x6x32x49xf32) <- (468x6x49x32xf32)
        transpose_20 = paddle._C_ops.transpose(slice_9, [0, 1, 3, 2])
        del slice_9

        # pd_op.matmul: (468x6x49x49xf32) <- (468x6x49x32xf32, 468x6x32x49xf32)
        matmul_14 = paddle._C_ops.matmul(scale_4, transpose_20, False, False)

        # pd_op.flatten: (2401xi64) <- (49x49xi64)
        flatten_4 = paddle._C_ops.flatten(data_15, 0, 1)
        del data_15

        # pd_op.index_select: (2401x6xf32) <- (169x6xf32, 2401xi64)
        index_select_2 = paddle._C_ops.index_select(data_2, flatten_4, 0)
        del data_2

        # pd_op.reshape: (49x49x6xf32) <- (2401x6xf32, 3xi64)
        reshape_40 = paddle._C_ops.reshape(index_select_2, full_int_array_29)

        # pd_op.transpose: (6x49x49xf32) <- (49x49x6xf32)
        transpose_21 = paddle._C_ops.transpose(reshape_40, [2, 0, 1])
        del reshape_40

        # pd_op.unsqueeze: (1x6x49x49xf32) <- (6x49x49xf32, 1xi64)
        unsqueeze_8 = paddle._C_ops.unsqueeze(transpose_21, full_int_array_27)

        # pd_op.add: (468x6x49x49xf32) <- (468x6x49x49xf32, 1x6x49x49xf32)
        add_22 = paddle._C_ops.add(matmul_14, unsqueeze_8)

        # pd_op.softmax: (468x6x49x49xf32) <- (468x6x49x49xf32)
        softmax_2 = paddle._C_ops.softmax(add_22, -1)
        del add_22

        # pd_op.matmul: (468x6x49x32xf32) <- (468x6x49x49xf32, 468x6x49x32xf32)
        matmul_15 = paddle._C_ops.matmul(softmax_2, slice_10, False, False)

        # pd_op.transpose: (468x49x6x32xf32) <- (468x6x49x32xf32)
        transpose_22 = paddle._C_ops.transpose(matmul_15, [0, 2, 1, 3])
        del matmul_15

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_50 = [-1, 49, 192]

        # pd_op.reshape: (468x49x192xf32) <- (468x49x6x32xf32, 3xi64)
        reshape_41 = paddle._C_ops.reshape(transpose_22, full_int_array_50)

        # pd_op.matmul: (468x49x192xf32) <- (468x49x192xf32, 192x192xf32)
        matmul_16 = paddle._C_ops.matmul(reshape_41, parameter_143, False, False)
        del parameter_143

        # pd_op.add: (468x49x192xf32) <- (468x49x192xf32, 192xf32)
        add_23 = paddle._C_ops.add(matmul_16, parameter_142)
        del parameter_142

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_51 = [468, 7, 7, 192]

        # pd_op.reshape: (468x7x7x192xf32) <- (468x49x192xf32, 4xi64)
        reshape_42 = paddle._C_ops.reshape(add_23, full_int_array_51)

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_52 = [-1, 18, 13, 7, 7, 192]

        # pd_op.reshape: (2x18x13x7x7x192xf32) <- (468x7x7x192xf32, 6xi64)
        reshape_43 = paddle._C_ops.reshape(reshape_42, full_int_array_52)

        # pd_op.transpose: (2x18x7x13x7x192xf32) <- (2x18x13x7x7x192xf32)
        transpose_23 = paddle._C_ops.transpose(reshape_43, [0, 1, 3, 2, 4, 5])
        del reshape_43

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_53 = [-1, 126, 91, 192]

        # pd_op.reshape: (2x126x91x192xf32) <- (2x18x7x13x7x192xf32, 4xi64)
        reshape_44 = paddle._C_ops.reshape(transpose_23, full_int_array_53)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_54 = [120, 88]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_164 = full_int_array_54

        # pd_op.slice: (2x120x88x192xf32) <- (2x126x91x192xf32, 2xi64, 2xi64)
        slice_11 = paddle._C_ops.slice(
            reshape_44, [1, 2], full_int_array_2, full_int_array_54, [1, 1], []
        )

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_55 = [-1, 10560, 192]

        # pd_op.reshape: (2x10560x192xf32) <- (2x120x88x192xf32, 3xi64)
        reshape_45 = paddle._C_ops.reshape(slice_11, full_int_array_55)

        # pd_op.full: (xf64) <- ()
        full_10 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__2 = paddle._C_ops.assign_value_(
            full_10,
            [],
            paddle.float64,
            [float("0.981818")],
            paddle.framework._current_expected_place(),
        )
        del full_10

        # pd_op.cast: (xf32) <- (xf64)
        cast_4 = paddle._C_ops.cast(assign_value__2, paddle.float32)
        del assign_value__2

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_2 = paddle._C_ops.uniform(
            full_int_array_39,
            paddle.float32,
            full_4,
            full_1,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_24 = paddle._C_ops.add(cast_4, uniform_2)
        del uniform_2

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_2 = paddle._C_ops.floor(add_24)
        del add_24

        # pd_op.divide: (2x10560x192xf32) <- (2x10560x192xf32, xf32)
        divide_2 = paddle._C_ops.divide(reshape_45, cast_4)

        # pd_op.multiply: (2x10560x192xf32) <- (2x10560x192xf32, 2x1x1xf32)
        multiply_4 = paddle._C_ops.multiply(divide_2, floor_2)

        # pd_op.add: (2x10560x192xf32) <- (2x10560x192xf32, 2x10560x192xf32)
        add_25 = paddle._C_ops.add(matmul_12, multiply_4)

        # pd_op.layer_norm: (2x10560x192xf32, 2x10560xf32, 2x10560xf32) <- (2x10560x192xf32, 192xf32, 192xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_25, parameter_141, parameter_140, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_140, parameter_141

        # pd_op.matmul: (2x10560x768xf32) <- (2x10560x192xf32, 192x768xf32)
        matmul_17 = paddle._C_ops.matmul(layer_norm_24, parameter_139, False, False)
        del parameter_139

        # pd_op.add: (2x10560x768xf32) <- (2x10560x768xf32, 768xf32)
        add_26 = paddle._C_ops.add(matmul_17, parameter_138)
        del parameter_138

        # pd_op.gelu: (2x10560x768xf32) <- (2x10560x768xf32)
        gelu_2 = paddle._C_ops.gelu(add_26, False)

        # pd_op.matmul: (2x10560x192xf32) <- (2x10560x768xf32, 768x192xf32)
        matmul_18 = paddle._C_ops.matmul(gelu_2, parameter_137, False, False)
        del parameter_137

        # pd_op.add: (2x10560x192xf32) <- (2x10560x192xf32, 192xf32)
        add_27 = paddle._C_ops.add(matmul_18, parameter_136)
        del parameter_136

        # pd_op.full: (xf64) <- ()
        full_11 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__3 = paddle._C_ops.assign_value_(
            full_11,
            [],
            paddle.float64,
            [float("0.981818")],
            paddle.framework._current_expected_place(),
        )
        del full_11

        # pd_op.cast: (xf32) <- (xf64)
        cast_5 = paddle._C_ops.cast(assign_value__3, paddle.float32)
        del assign_value__3

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_3 = paddle._C_ops.uniform(
            full_int_array_39,
            paddle.float32,
            full_4,
            full_1,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_28 = paddle._C_ops.add(cast_5, uniform_3)
        del uniform_3

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_3 = paddle._C_ops.floor(add_28)
        del add_28

        # pd_op.divide: (2x10560x192xf32) <- (2x10560x192xf32, xf32)
        divide_3 = paddle._C_ops.divide(add_27, cast_5)

        # pd_op.multiply: (2x10560x192xf32) <- (2x10560x192xf32, 2x1x1xf32)
        multiply_5 = paddle._C_ops.multiply(divide_3, floor_3)

        # pd_op.add: (2x10560x192xf32) <- (2x10560x192xf32, 2x10560x192xf32)
        add_29 = paddle._C_ops.add(add_25, multiply_5)

        # pd_op.layer_norm: (2x10560x192xf32, 2x10560xf32, 2x10560xf32) <- (2x10560x192xf32, 192xf32, 192xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_29, parameter_135, parameter_134, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_134, parameter_135

        # pd_op.reshape: (2x120x88x192xf32) <- (2x10560x192xf32, 4xi64)
        reshape_46 = paddle._C_ops.reshape(layer_norm_27, full_int_array_45)

        # pd_op.pad: (2x126x91x192xf32) <- (2x120x88x192xf32, 1xf32)
        pad_3 = paddle._C_ops.pad(reshape_46, [0, 0, 0, 6, 0, 3, 0, 0], full_4)

        # pd_op.roll: (2x126x91x192xf32) <- (2x126x91x192xf32, 2xi64)
        roll_2 = paddle._C_ops.roll(pad_3, full_int_array_11, [1, 2])

        # pd_op.reshape: (2x18x7x13x7x192xf32) <- (2x126x91x192xf32, 6xi64)
        reshape_47 = paddle._C_ops.reshape(roll_2, full_int_array_46)
        del full_int_array_46

        # pd_op.transpose: (2x18x13x7x7x192xf32) <- (2x18x7x13x7x192xf32)
        transpose_24 = paddle._C_ops.transpose(reshape_47, [0, 1, 3, 2, 4, 5])
        del reshape_47

        # pd_op.reshape: (468x7x7x192xf32) <- (2x18x13x7x7x192xf32, 4xi64)
        reshape_48 = paddle._C_ops.reshape(transpose_24, full_int_array_47)
        del full_int_array_47

        # pd_op.reshape: (468x49x192xf32) <- (468x7x7x192xf32, 3xi64)
        reshape_49 = paddle._C_ops.reshape(reshape_48, full_int_array_48)
        del full_int_array_48

        # pd_op.matmul: (468x49x576xf32) <- (468x49x192xf32, 192x576xf32)
        matmul_19 = paddle._C_ops.matmul(reshape_49, parameter_133, False, False)
        del parameter_133

        # pd_op.add: (468x49x576xf32) <- (468x49x576xf32, 576xf32)
        add_30 = paddle._C_ops.add(matmul_19, parameter_132)
        del parameter_132

        # pd_op.reshape: (468x49x3x6x32xf32) <- (468x49x576xf32, 5xi64)
        reshape_50 = paddle._C_ops.reshape(add_30, full_int_array_49)
        del full_int_array_49

        # pd_op.transpose: (3x468x6x49x32xf32) <- (468x49x3x6x32xf32)
        transpose_25 = paddle._C_ops.transpose(reshape_50, [2, 0, 3, 1, 4])
        del reshape_50

        # pd_op.slice: (468x6x49x32xf32) <- (3x468x6x49x32xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            transpose_25, [0], full_int_array_27, full_int_array_20, [1], [0]
        )

        # pd_op.slice: (468x6x49x32xf32) <- (3x468x6x49x32xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            transpose_25, [0], full_int_array_20, full_int_array_21, [1], [0]
        )

        # pd_op.slice: (468x6x49x32xf32) <- (3x468x6x49x32xf32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            transpose_25, [0], full_int_array_21, full_int_array_28, [1], [0]
        )

        # pd_op.scale: (468x6x49x32xf32) <- (468x6x49x32xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(slice_12, full_5, float("0"), True)
        del slice_12

        # pd_op.transpose: (468x6x32x49xf32) <- (468x6x49x32xf32)
        transpose_26 = paddle._C_ops.transpose(slice_13, [0, 1, 3, 2])
        del slice_13

        # pd_op.matmul: (468x6x49x49xf32) <- (468x6x49x32xf32, 468x6x32x49xf32)
        matmul_20 = paddle._C_ops.matmul(scale_5, transpose_26, False, False)

        # pd_op.flatten: (2401xi64) <- (49x49xi64)
        flatten_5 = paddle._C_ops.flatten(data_16, 0, 1)
        del data_16

        # pd_op.index_select: (2401x6xf32) <- (169x6xf32, 2401xi64)
        index_select_3 = paddle._C_ops.index_select(data_3, flatten_5, 0)
        del data_3

        # pd_op.reshape: (49x49x6xf32) <- (2401x6xf32, 3xi64)
        reshape_51 = paddle._C_ops.reshape(index_select_3, full_int_array_29)

        # pd_op.transpose: (6x49x49xf32) <- (49x49x6xf32)
        transpose_27 = paddle._C_ops.transpose(reshape_51, [2, 0, 1])
        del reshape_51

        # pd_op.unsqueeze: (1x6x49x49xf32) <- (6x49x49xf32, 1xi64)
        unsqueeze_9 = paddle._C_ops.unsqueeze(transpose_27, full_int_array_27)

        # pd_op.add: (468x6x49x49xf32) <- (468x6x49x49xf32, 1x6x49x49xf32)
        add_31 = paddle._C_ops.add(matmul_20, unsqueeze_9)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_56 = [-1, 234, 6, 49, 49]

        # pd_op.reshape: (2x234x6x49x49xf32) <- (468x6x49x49xf32, 5xi64)
        reshape_52 = paddle._C_ops.reshape(add_31, full_int_array_56)
        del full_int_array_56

        # pd_op.unsqueeze: (234x1x49x49xf32) <- (234x49x49xf32, 1xi64)
        unsqueeze_10 = paddle._C_ops.unsqueeze(multiply_3, full_int_array_20)
        del multiply_3

        # pd_op.unsqueeze: (1x234x1x49x49xf32) <- (234x1x49x49xf32, 1xi64)
        unsqueeze_11 = paddle._C_ops.unsqueeze(unsqueeze_10, full_int_array_27)
        del unsqueeze_10

        # pd_op.add: (2x234x6x49x49xf32) <- (2x234x6x49x49xf32, 1x234x1x49x49xf32)
        add_32 = paddle._C_ops.add(reshape_52, unsqueeze_11)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_57 = [-1, 6, 49, 49]

        # pd_op.reshape: (468x6x49x49xf32) <- (2x234x6x49x49xf32, 4xi64)
        reshape_53 = paddle._C_ops.reshape(add_32, full_int_array_57)
        del full_int_array_57

        # pd_op.softmax: (468x6x49x49xf32) <- (468x6x49x49xf32)
        softmax_3 = paddle._C_ops.softmax(reshape_53, -1)
        del reshape_53

        # pd_op.matmul: (468x6x49x32xf32) <- (468x6x49x49xf32, 468x6x49x32xf32)
        matmul_21 = paddle._C_ops.matmul(softmax_3, slice_14, False, False)

        # pd_op.transpose: (468x49x6x32xf32) <- (468x6x49x32xf32)
        transpose_28 = paddle._C_ops.transpose(matmul_21, [0, 2, 1, 3])
        del matmul_21

        # pd_op.reshape: (468x49x192xf32) <- (468x49x6x32xf32, 3xi64)
        reshape_54 = paddle._C_ops.reshape(transpose_28, full_int_array_50)
        del full_int_array_50

        # pd_op.matmul: (468x49x192xf32) <- (468x49x192xf32, 192x192xf32)
        matmul_22 = paddle._C_ops.matmul(reshape_54, parameter_131, False, False)
        del parameter_131

        # pd_op.add: (468x49x192xf32) <- (468x49x192xf32, 192xf32)
        add_33 = paddle._C_ops.add(matmul_22, parameter_130)
        del parameter_130

        # pd_op.reshape: (468x7x7x192xf32) <- (468x49x192xf32, 4xi64)
        reshape_55 = paddle._C_ops.reshape(add_33, full_int_array_51)
        del full_int_array_51

        # pd_op.reshape: (2x18x13x7x7x192xf32) <- (468x7x7x192xf32, 6xi64)
        reshape_56 = paddle._C_ops.reshape(reshape_55, full_int_array_52)
        del full_int_array_52

        # pd_op.transpose: (2x18x7x13x7x192xf32) <- (2x18x13x7x7x192xf32)
        transpose_29 = paddle._C_ops.transpose(reshape_56, [0, 1, 3, 2, 4, 5])
        del reshape_56

        # pd_op.reshape: (2x126x91x192xf32) <- (2x18x7x13x7x192xf32, 4xi64)
        reshape_57 = paddle._C_ops.reshape(transpose_29, full_int_array_53)
        del full_int_array_53

        # pd_op.roll: (2x126x91x192xf32) <- (2x126x91x192xf32, 2xi64)
        roll_3 = paddle._C_ops.roll(reshape_57, full_int_array_38, [1, 2])

        # pd_op.slice: (2x120x88x192xf32) <- (2x126x91x192xf32, 2xi64, 2xi64)
        slice_15 = paddle._C_ops.slice(
            roll_3, [1, 2], full_int_array_2, full_int_array_54, [1, 1], []
        )

        # pd_op.reshape: (2x10560x192xf32) <- (2x120x88x192xf32, 3xi64)
        reshape_58 = paddle._C_ops.reshape(slice_15, full_int_array_55)
        del full_int_array_55

        # pd_op.full: (xf64) <- ()
        full_12 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__4 = paddle._C_ops.assign_value_(
            full_12,
            [],
            paddle.float64,
            [float("0.972727")],
            paddle.framework._current_expected_place(),
        )
        del full_12

        # pd_op.cast: (xf32) <- (xf64)
        cast_6 = paddle._C_ops.cast(assign_value__4, paddle.float32)
        del assign_value__4

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_4 = paddle._C_ops.uniform(
            full_int_array_39,
            paddle.float32,
            full_4,
            full_1,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_34 = paddle._C_ops.add(cast_6, uniform_4)
        del uniform_4

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_4 = paddle._C_ops.floor(add_34)
        del add_34

        # pd_op.divide: (2x10560x192xf32) <- (2x10560x192xf32, xf32)
        divide_4 = paddle._C_ops.divide(reshape_58, cast_6)

        # pd_op.multiply: (2x10560x192xf32) <- (2x10560x192xf32, 2x1x1xf32)
        multiply_6 = paddle._C_ops.multiply(divide_4, floor_4)

        # pd_op.add: (2x10560x192xf32) <- (2x10560x192xf32, 2x10560x192xf32)
        add_35 = paddle._C_ops.add(add_29, multiply_6)

        # pd_op.layer_norm: (2x10560x192xf32, 2x10560xf32, 2x10560xf32) <- (2x10560x192xf32, 192xf32, 192xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_35, parameter_129, parameter_128, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_128, parameter_129

        # pd_op.matmul: (2x10560x768xf32) <- (2x10560x192xf32, 192x768xf32)
        matmul_23 = paddle._C_ops.matmul(layer_norm_30, parameter_127, False, False)
        del parameter_127

        # pd_op.add: (2x10560x768xf32) <- (2x10560x768xf32, 768xf32)
        add_36 = paddle._C_ops.add(matmul_23, parameter_126)
        del parameter_126

        # pd_op.gelu: (2x10560x768xf32) <- (2x10560x768xf32)
        gelu_3 = paddle._C_ops.gelu(add_36, False)

        # pd_op.matmul: (2x10560x192xf32) <- (2x10560x768xf32, 768x192xf32)
        matmul_24 = paddle._C_ops.matmul(gelu_3, parameter_125, False, False)
        del parameter_125

        # pd_op.add: (2x10560x192xf32) <- (2x10560x192xf32, 192xf32)
        add_37 = paddle._C_ops.add(matmul_24, parameter_124)
        del parameter_124

        # pd_op.full: (xf64) <- ()
        full_13 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__5 = paddle._C_ops.assign_value_(
            full_13,
            [],
            paddle.float64,
            [float("0.972727")],
            paddle.framework._current_expected_place(),
        )
        del full_13

        # pd_op.cast: (xf32) <- (xf64)
        cast_7 = paddle._C_ops.cast(assign_value__5, paddle.float32)
        del assign_value__5

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_5 = paddle._C_ops.uniform(
            full_int_array_39,
            paddle.float32,
            full_4,
            full_1,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_38 = paddle._C_ops.add(cast_7, uniform_5)
        del uniform_5

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_5 = paddle._C_ops.floor(add_38)
        del add_38

        # pd_op.divide: (2x10560x192xf32) <- (2x10560x192xf32, xf32)
        divide_5 = paddle._C_ops.divide(add_37, cast_7)

        # pd_op.multiply: (2x10560x192xf32) <- (2x10560x192xf32, 2x1x1xf32)
        multiply_7 = paddle._C_ops.multiply(divide_5, floor_5)

        # pd_op.add: (2x10560x192xf32) <- (2x10560x192xf32, 2x10560x192xf32)
        add_39 = paddle._C_ops.add(add_35, multiply_7)

        # pd_op.reshape: (2x120x88x192xf32) <- (2x10560x192xf32, 4xi64)
        reshape_59 = paddle._C_ops.reshape(add_39, full_int_array_45)

        # pd_op.strided_slice: (2x60x44x192xf32) <- (2x120x88x192xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_4 = paddle._C_ops.strided_slice(
            reshape_59, [1, 2], full_int_array_2, full_int_array_16, full_int_array_40
        )

        # pd_op.strided_slice: (2x60x44x192xf32) <- (2x120x88x192xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_5 = paddle._C_ops.strided_slice(
            reshape_59, [1, 2], full_int_array_41, full_int_array_16, full_int_array_40
        )

        # pd_op.strided_slice: (2x60x44x192xf32) <- (2x120x88x192xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_6 = paddle._C_ops.strided_slice(
            reshape_59, [1, 2], full_int_array_42, full_int_array_16, full_int_array_40
        )

        # pd_op.strided_slice: (2x60x44x192xf32) <- (2x120x88x192xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_7 = paddle._C_ops.strided_slice(
            reshape_59, [1, 2], full_int_array_4, full_int_array_16, full_int_array_40
        )

        # builtin.combine: ([2x60x44x192xf32, 2x60x44x192xf32, 2x60x44x192xf32, 2x60x44x192xf32]) <- (2x60x44x192xf32, 2x60x44x192xf32, 2x60x44x192xf32, 2x60x44x192xf32)
        combine_1 = [strided_slice_4, strided_slice_5, strided_slice_6, strided_slice_7]

        # pd_op.concat: (2x60x44x768xf32) <- ([2x60x44x192xf32, 2x60x44x192xf32, 2x60x44x192xf32, 2x60x44x192xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_8)
        del combine_1

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_58 = [-1, 2640, 768]

        # pd_op.reshape: (2x2640x768xf32) <- (2x60x44x768xf32, 3xi64)
        reshape_60 = paddle._C_ops.reshape(concat_1, full_int_array_58)
        del full_int_array_58

        # pd_op.layer_norm: (2x2640x768xf32, 2x2640xf32, 2x2640xf32) <- (2x2640x768xf32, 768xf32, 768xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                reshape_60, parameter_123, parameter_122, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_122, parameter_123

        # pd_op.matmul: (2x2640x384xf32) <- (2x2640x768xf32, 768x384xf32)
        matmul_25 = paddle._C_ops.matmul(layer_norm_33, parameter_121, False, False)
        del parameter_121

        # pd_op.layer_norm: (2x10560x192xf32, 2x10560xf32, 2x10560xf32) <- (2x10560x192xf32, 192xf32, 192xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_39, parameter_120, parameter_119, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_119, parameter_120

        # pd_op.reshape: (2x120x88x192xf32) <- (2x10560x192xf32, 4xi64)
        reshape_61 = paddle._C_ops.reshape(layer_norm_36, full_int_array_45)
        del full_int_array_45

        # pd_op.transpose: (2x192x120x88xf32) <- (2x120x88x192xf32)
        transpose_30 = paddle._C_ops.transpose(reshape_61, [0, 3, 1, 2])
        del reshape_61

        # pd_op.full: (1x63x49x1xf32) <- ()
        full_14 = paddle._C_ops.full(
            [1, 63, 49, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x63x49x1xf32) <- (1x63x49x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__18 = paddle._C_ops.set_value_(
            full_14,
            full_int_array_2,
            full_int_array_3,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("0")],
        )
        del full_14

        # pd_op.set_value_: (1x63x49x1xf32) <- (1x63x49x1xf32, 2xi64, 2xi64, 2xi64)
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

        # pd_op.set_value_: (1x63x49x1xf32) <- (1x63x49x1xf32, 2xi64, 2xi64, 2xi64)
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

        # pd_op.set_value_: (1x63x49x1xf32) <- (1x63x49x1xf32, 2xi64, 2xi64, 2xi64)
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

        # pd_op.set_value_: (1x63x49x1xf32) <- (1x63x49x1xf32, 2xi64, 2xi64, 2xi64)
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

        # pd_op.set_value_: (1x63x49x1xf32) <- (1x63x49x1xf32, 2xi64, 2xi64, 2xi64)
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

        # pd_op.set_value_: (1x63x49x1xf32) <- (1x63x49x1xf32, 2xi64, 2xi64, 2xi64)
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

        # pd_op.set_value_: (1x63x49x1xf32) <- (1x63x49x1xf32, 2xi64, 2xi64, 2xi64)
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

        # pd_op.set_value_: (1x63x49x1xf32) <- (1x63x49x1xf32, 2xi64, 2xi64, 2xi64)
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
        full_int_array_59 = [-1, 9, 7, 7, 7, 1]

        # pd_op.reshape: (1x9x7x7x7x1xf32) <- (1x63x49x1xf32, 6xi64)
        reshape_62 = paddle._C_ops.reshape(set_value__26, full_int_array_59)
        del full_int_array_59

        # pd_op.transpose: (1x9x7x7x7x1xf32) <- (1x9x7x7x7x1xf32)
        transpose_31 = paddle._C_ops.transpose(reshape_62, [0, 1, 3, 2, 4, 5])
        del reshape_62

        # pd_op.reshape: (63x7x7x1xf32) <- (1x9x7x7x7x1xf32, 4xi64)
        reshape_63 = paddle._C_ops.reshape(transpose_31, full_int_array_18)
        del transpose_31

        # pd_op.reshape: (63x49xf32) <- (63x7x7x1xf32, 2xi64)
        reshape_64 = paddle._C_ops.reshape(reshape_63, full_int_array_19)
        del reshape_63

        # pd_op.unsqueeze: (63x1x49xf32) <- (63x49xf32, 1xi64)
        unsqueeze_12 = paddle._C_ops.unsqueeze(reshape_64, full_int_array_20)

        # pd_op.unsqueeze: (63x49x1xf32) <- (63x49xf32, 1xi64)
        unsqueeze_13 = paddle._C_ops.unsqueeze(reshape_64, full_int_array_21)
        del reshape_64

        # pd_op.subtract: (63x49x49xf32) <- (63x1x49xf32, 63x49x1xf32)
        subtract_2 = paddle._C_ops.subtract(unsqueeze_12, unsqueeze_13)
        del unsqueeze_12, unsqueeze_13

        # pd_op.full_like: (63x49x49xf32) <- (63x49x49xf32, 1xf32)
        full_like_2 = paddle._C_ops.full_like(
            subtract_2,
            full_1,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.scale: (63x49x49xf32) <- (63x49x49xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(full_like_2, full_2, float("0"), True)
        del full_like_2

        # pd_op.not_equal: (63x49x49xb) <- (63x49x49xf32, xf32)
        not_equal_2 = paddle._C_ops.not_equal(subtract_2, full_3)
        del subtract_2

        # pd_op.cast: (63x49x49xf32) <- (63x49x49xb)
        cast_8 = paddle._C_ops.cast(not_equal_2, paddle.float32)
        del not_equal_2

        # pd_op.multiply: (63x49x49xf32) <- (63x49x49xf32, 63x49x49xf32)
        multiply_8 = paddle._C_ops.multiply(scale_6, cast_8)
        del cast_8, scale_6

        # pd_op.layer_norm: (2x2640x384xf32, 2x2640xf32, 2x2640xf32) <- (2x2640x384xf32, 384xf32, 384xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                matmul_25, parameter_118, parameter_117, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_117, parameter_118

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_60 = [-1, 60, 44, 384]

        # pd_op.reshape: (2x60x44x384xf32) <- (2x2640x384xf32, 4xi64)
        reshape_65 = paddle._C_ops.reshape(layer_norm_39, full_int_array_60)

        # pd_op.pad: (2x63x49x384xf32) <- (2x60x44x384xf32, 1xf32)
        pad_4 = paddle._C_ops.pad(reshape_65, [0, 0, 0, 3, 0, 5, 0, 0], full_4)

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_61 = [-1, 9, 7, 7, 7, 384]

        # pd_op.reshape: (2x9x7x7x7x384xf32) <- (2x63x49x384xf32, 6xi64)
        reshape_66 = paddle._C_ops.reshape(pad_4, full_int_array_61)

        # pd_op.transpose: (2x9x7x7x7x384xf32) <- (2x9x7x7x7x384xf32)
        transpose_32 = paddle._C_ops.transpose(reshape_66, [0, 1, 3, 2, 4, 5])
        del reshape_66

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_62 = [-1, 7, 7, 384]

        # pd_op.reshape: (126x7x7x384xf32) <- (2x9x7x7x7x384xf32, 4xi64)
        reshape_67 = paddle._C_ops.reshape(transpose_32, full_int_array_62)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_63 = [126, 49, 384]

        # pd_op.reshape: (126x49x384xf32) <- (126x7x7x384xf32, 3xi64)
        reshape_68 = paddle._C_ops.reshape(reshape_67, full_int_array_63)

        # pd_op.matmul: (126x49x1152xf32) <- (126x49x384xf32, 384x1152xf32)
        matmul_26 = paddle._C_ops.matmul(reshape_68, parameter_116, False, False)
        del parameter_116

        # pd_op.add: (126x49x1152xf32) <- (126x49x1152xf32, 1152xf32)
        add_40 = paddle._C_ops.add(matmul_26, parameter_115)
        del parameter_115

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_64 = [-1, 49, 3, 12, 32]

        # pd_op.reshape: (126x49x3x12x32xf32) <- (126x49x1152xf32, 5xi64)
        reshape_69 = paddle._C_ops.reshape(add_40, full_int_array_64)

        # pd_op.transpose: (3x126x12x49x32xf32) <- (126x49x3x12x32xf32)
        transpose_33 = paddle._C_ops.transpose(reshape_69, [2, 0, 3, 1, 4])
        del reshape_69

        # pd_op.slice: (126x12x49x32xf32) <- (3x126x12x49x32xf32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            transpose_33, [0], full_int_array_27, full_int_array_20, [1], [0]
        )

        # pd_op.slice: (126x12x49x32xf32) <- (3x126x12x49x32xf32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            transpose_33, [0], full_int_array_20, full_int_array_21, [1], [0]
        )

        # pd_op.slice: (126x12x49x32xf32) <- (3x126x12x49x32xf32, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(
            transpose_33, [0], full_int_array_21, full_int_array_28, [1], [0]
        )

        # pd_op.scale: (126x12x49x32xf32) <- (126x12x49x32xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(slice_16, full_5, float("0"), True)
        del slice_16

        # pd_op.transpose: (126x12x32x49xf32) <- (126x12x49x32xf32)
        transpose_34 = paddle._C_ops.transpose(slice_17, [0, 1, 3, 2])
        del slice_17

        # pd_op.matmul: (126x12x49x49xf32) <- (126x12x49x32xf32, 126x12x32x49xf32)
        matmul_27 = paddle._C_ops.matmul(scale_7, transpose_34, False, False)

        # pd_op.flatten: (2401xi64) <- (49x49xi64)
        flatten_6 = paddle._C_ops.flatten(data_17, 0, 1)
        del data_17

        # pd_op.index_select: (2401x12xf32) <- (169x12xf32, 2401xi64)
        index_select_4 = paddle._C_ops.index_select(data_4, flatten_6, 0)
        del data_4

        # pd_op.reshape: (49x49x12xf32) <- (2401x12xf32, 3xi64)
        reshape_70 = paddle._C_ops.reshape(index_select_4, full_int_array_29)

        # pd_op.transpose: (12x49x49xf32) <- (49x49x12xf32)
        transpose_35 = paddle._C_ops.transpose(reshape_70, [2, 0, 1])
        del reshape_70

        # pd_op.unsqueeze: (1x12x49x49xf32) <- (12x49x49xf32, 1xi64)
        unsqueeze_14 = paddle._C_ops.unsqueeze(transpose_35, full_int_array_27)

        # pd_op.add: (126x12x49x49xf32) <- (126x12x49x49xf32, 1x12x49x49xf32)
        add_41 = paddle._C_ops.add(matmul_27, unsqueeze_14)

        # pd_op.softmax: (126x12x49x49xf32) <- (126x12x49x49xf32)
        softmax_4 = paddle._C_ops.softmax(add_41, -1)
        del add_41

        # pd_op.matmul: (126x12x49x32xf32) <- (126x12x49x49xf32, 126x12x49x32xf32)
        matmul_28 = paddle._C_ops.matmul(softmax_4, slice_18, False, False)

        # pd_op.transpose: (126x49x12x32xf32) <- (126x12x49x32xf32)
        transpose_36 = paddle._C_ops.transpose(matmul_28, [0, 2, 1, 3])
        del matmul_28

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_65 = [-1, 49, 384]

        # pd_op.reshape: (126x49x384xf32) <- (126x49x12x32xf32, 3xi64)
        reshape_71 = paddle._C_ops.reshape(transpose_36, full_int_array_65)

        # pd_op.matmul: (126x49x384xf32) <- (126x49x384xf32, 384x384xf32)
        matmul_29 = paddle._C_ops.matmul(reshape_71, parameter_114, False, False)
        del parameter_114

        # pd_op.add: (126x49x384xf32) <- (126x49x384xf32, 384xf32)
        add_42 = paddle._C_ops.add(matmul_29, parameter_113)
        del parameter_113

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_66 = [126, 7, 7, 384]

        # pd_op.reshape: (126x7x7x384xf32) <- (126x49x384xf32, 4xi64)
        reshape_72 = paddle._C_ops.reshape(add_42, full_int_array_66)

        # pd_op.reshape: (2x9x7x7x7x384xf32) <- (126x7x7x384xf32, 6xi64)
        reshape_73 = paddle._C_ops.reshape(reshape_72, full_int_array_61)

        # pd_op.transpose: (2x9x7x7x7x384xf32) <- (2x9x7x7x7x384xf32)
        transpose_37 = paddle._C_ops.transpose(reshape_73, [0, 1, 3, 2, 4, 5])
        del reshape_73

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_67 = [-1, 63, 49, 384]

        # pd_op.reshape: (2x63x49x384xf32) <- (2x9x7x7x7x384xf32, 4xi64)
        reshape_74 = paddle._C_ops.reshape(transpose_37, full_int_array_67)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_68 = [60, 44]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_165 = full_int_array_68

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_166 = full_int_array_68

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_167 = full_int_array_68

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_168 = full_int_array_68

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_169 = full_int_array_68

        # pd_op.slice: (2x60x44x384xf32) <- (2x63x49x384xf32, 2xi64, 2xi64)
        slice_19 = paddle._C_ops.slice(
            reshape_74, [1, 2], full_int_array_2, full_int_array_68, [1, 1], []
        )

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_69 = [-1, 2640, 384]

        # pd_op.reshape: (2x2640x384xf32) <- (2x60x44x384xf32, 3xi64)
        reshape_75 = paddle._C_ops.reshape(slice_19, full_int_array_69)

        # pd_op.full: (xf64) <- ()
        full_15 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__6 = paddle._C_ops.assign_value_(
            full_15,
            [],
            paddle.float64,
            [float("0.963636")],
            paddle.framework._current_expected_place(),
        )
        del full_15

        # pd_op.cast: (xf32) <- (xf64)
        cast_9 = paddle._C_ops.cast(assign_value__6, paddle.float32)
        del assign_value__6

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_6 = paddle._C_ops.uniform(
            full_int_array_39,
            paddle.float32,
            full_4,
            full_1,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_43 = paddle._C_ops.add(cast_9, uniform_6)
        del uniform_6

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_6 = paddle._C_ops.floor(add_43)
        del add_43

        # pd_op.divide: (2x2640x384xf32) <- (2x2640x384xf32, xf32)
        divide_6 = paddle._C_ops.divide(reshape_75, cast_9)

        # pd_op.multiply: (2x2640x384xf32) <- (2x2640x384xf32, 2x1x1xf32)
        multiply_9 = paddle._C_ops.multiply(divide_6, floor_6)

        # pd_op.add: (2x2640x384xf32) <- (2x2640x384xf32, 2x2640x384xf32)
        add_44 = paddle._C_ops.add(matmul_25, multiply_9)

        # pd_op.layer_norm: (2x2640x384xf32, 2x2640xf32, 2x2640xf32) <- (2x2640x384xf32, 384xf32, 384xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_44, parameter_112, parameter_111, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_111, parameter_112

        # pd_op.matmul: (2x2640x1536xf32) <- (2x2640x384xf32, 384x1536xf32)
        matmul_30 = paddle._C_ops.matmul(layer_norm_42, parameter_110, False, False)
        del parameter_110

        # pd_op.add: (2x2640x1536xf32) <- (2x2640x1536xf32, 1536xf32)
        add_45 = paddle._C_ops.add(matmul_30, parameter_109)
        del parameter_109

        # pd_op.gelu: (2x2640x1536xf32) <- (2x2640x1536xf32)
        gelu_4 = paddle._C_ops.gelu(add_45, False)

        # pd_op.matmul: (2x2640x384xf32) <- (2x2640x1536xf32, 1536x384xf32)
        matmul_31 = paddle._C_ops.matmul(gelu_4, parameter_108, False, False)
        del parameter_108

        # pd_op.add: (2x2640x384xf32) <- (2x2640x384xf32, 384xf32)
        add_46 = paddle._C_ops.add(matmul_31, parameter_107)
        del parameter_107

        # pd_op.full: (xf64) <- ()
        full_16 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__7 = paddle._C_ops.assign_value_(
            full_16,
            [],
            paddle.float64,
            [float("0.963636")],
            paddle.framework._current_expected_place(),
        )
        del full_16

        # pd_op.cast: (xf32) <- (xf64)
        cast_10 = paddle._C_ops.cast(assign_value__7, paddle.float32)
        del assign_value__7

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_7 = paddle._C_ops.uniform(
            full_int_array_39,
            paddle.float32,
            full_4,
            full_1,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_47 = paddle._C_ops.add(cast_10, uniform_7)
        del uniform_7

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_7 = paddle._C_ops.floor(add_47)
        del add_47

        # pd_op.divide: (2x2640x384xf32) <- (2x2640x384xf32, xf32)
        divide_7 = paddle._C_ops.divide(add_46, cast_10)

        # pd_op.multiply: (2x2640x384xf32) <- (2x2640x384xf32, 2x1x1xf32)
        multiply_10 = paddle._C_ops.multiply(divide_7, floor_7)

        # pd_op.add: (2x2640x384xf32) <- (2x2640x384xf32, 2x2640x384xf32)
        add_48 = paddle._C_ops.add(add_44, multiply_10)

        # pd_op.layer_norm: (2x2640x384xf32, 2x2640xf32, 2x2640xf32) <- (2x2640x384xf32, 384xf32, 384xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_48, parameter_106, parameter_105, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_105, parameter_106

        # pd_op.reshape: (2x60x44x384xf32) <- (2x2640x384xf32, 4xi64)
        reshape_76 = paddle._C_ops.reshape(layer_norm_45, full_int_array_60)

        # pd_op.pad: (2x63x49x384xf32) <- (2x60x44x384xf32, 1xf32)
        pad_5 = paddle._C_ops.pad(reshape_76, [0, 0, 0, 3, 0, 5, 0, 0], full_4)

        # pd_op.roll: (2x63x49x384xf32) <- (2x63x49x384xf32, 2xi64)
        roll_4 = paddle._C_ops.roll(pad_5, full_int_array_11, [1, 2])

        # pd_op.reshape: (2x9x7x7x7x384xf32) <- (2x63x49x384xf32, 6xi64)
        reshape_77 = paddle._C_ops.reshape(roll_4, full_int_array_61)

        # pd_op.transpose: (2x9x7x7x7x384xf32) <- (2x9x7x7x7x384xf32)
        transpose_38 = paddle._C_ops.transpose(reshape_77, [0, 1, 3, 2, 4, 5])
        del reshape_77

        # pd_op.reshape: (126x7x7x384xf32) <- (2x9x7x7x7x384xf32, 4xi64)
        reshape_78 = paddle._C_ops.reshape(transpose_38, full_int_array_62)

        # pd_op.reshape: (126x49x384xf32) <- (126x7x7x384xf32, 3xi64)
        reshape_79 = paddle._C_ops.reshape(reshape_78, full_int_array_63)

        # pd_op.matmul: (126x49x1152xf32) <- (126x49x384xf32, 384x1152xf32)
        matmul_32 = paddle._C_ops.matmul(reshape_79, parameter_104, False, False)
        del parameter_104

        # pd_op.add: (126x49x1152xf32) <- (126x49x1152xf32, 1152xf32)
        add_49 = paddle._C_ops.add(matmul_32, parameter_103)
        del parameter_103

        # pd_op.reshape: (126x49x3x12x32xf32) <- (126x49x1152xf32, 5xi64)
        reshape_80 = paddle._C_ops.reshape(add_49, full_int_array_64)

        # pd_op.transpose: (3x126x12x49x32xf32) <- (126x49x3x12x32xf32)
        transpose_39 = paddle._C_ops.transpose(reshape_80, [2, 0, 3, 1, 4])
        del reshape_80

        # pd_op.slice: (126x12x49x32xf32) <- (3x126x12x49x32xf32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(
            transpose_39, [0], full_int_array_27, full_int_array_20, [1], [0]
        )

        # pd_op.slice: (126x12x49x32xf32) <- (3x126x12x49x32xf32, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(
            transpose_39, [0], full_int_array_20, full_int_array_21, [1], [0]
        )

        # pd_op.slice: (126x12x49x32xf32) <- (3x126x12x49x32xf32, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(
            transpose_39, [0], full_int_array_21, full_int_array_28, [1], [0]
        )

        # pd_op.scale: (126x12x49x32xf32) <- (126x12x49x32xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(slice_20, full_5, float("0"), True)
        del slice_20

        # pd_op.transpose: (126x12x32x49xf32) <- (126x12x49x32xf32)
        transpose_40 = paddle._C_ops.transpose(slice_21, [0, 1, 3, 2])
        del slice_21

        # pd_op.matmul: (126x12x49x49xf32) <- (126x12x49x32xf32, 126x12x32x49xf32)
        matmul_33 = paddle._C_ops.matmul(scale_8, transpose_40, False, False)

        # pd_op.flatten: (2401xi64) <- (49x49xi64)
        flatten_7 = paddle._C_ops.flatten(data_18, 0, 1)
        del data_18

        # pd_op.index_select: (2401x12xf32) <- (169x12xf32, 2401xi64)
        index_select_5 = paddle._C_ops.index_select(data_5, flatten_7, 0)
        del data_5

        # pd_op.reshape: (49x49x12xf32) <- (2401x12xf32, 3xi64)
        reshape_81 = paddle._C_ops.reshape(index_select_5, full_int_array_29)

        # pd_op.transpose: (12x49x49xf32) <- (49x49x12xf32)
        transpose_41 = paddle._C_ops.transpose(reshape_81, [2, 0, 1])
        del reshape_81

        # pd_op.unsqueeze: (1x12x49x49xf32) <- (12x49x49xf32, 1xi64)
        unsqueeze_15 = paddle._C_ops.unsqueeze(transpose_41, full_int_array_27)

        # pd_op.add: (126x12x49x49xf32) <- (126x12x49x49xf32, 1x12x49x49xf32)
        add_50 = paddle._C_ops.add(matmul_33, unsqueeze_15)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_70 = [-1, 63, 12, 49, 49]

        # pd_op.reshape: (2x63x12x49x49xf32) <- (126x12x49x49xf32, 5xi64)
        reshape_82 = paddle._C_ops.reshape(add_50, full_int_array_70)

        # pd_op.unsqueeze: (63x1x49x49xf32) <- (63x49x49xf32, 1xi64)
        unsqueeze_16 = paddle._C_ops.unsqueeze(multiply_8, full_int_array_20)

        # pd_op.unsqueeze: (1x63x1x49x49xf32) <- (63x1x49x49xf32, 1xi64)
        unsqueeze_17 = paddle._C_ops.unsqueeze(unsqueeze_16, full_int_array_27)
        del unsqueeze_16

        # pd_op.add: (2x63x12x49x49xf32) <- (2x63x12x49x49xf32, 1x63x1x49x49xf32)
        add_51 = paddle._C_ops.add(reshape_82, unsqueeze_17)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_71 = [-1, 12, 49, 49]

        # pd_op.reshape: (126x12x49x49xf32) <- (2x63x12x49x49xf32, 4xi64)
        reshape_83 = paddle._C_ops.reshape(add_51, full_int_array_71)

        # pd_op.softmax: (126x12x49x49xf32) <- (126x12x49x49xf32)
        softmax_5 = paddle._C_ops.softmax(reshape_83, -1)
        del reshape_83

        # pd_op.matmul: (126x12x49x32xf32) <- (126x12x49x49xf32, 126x12x49x32xf32)
        matmul_34 = paddle._C_ops.matmul(softmax_5, slice_22, False, False)

        # pd_op.transpose: (126x49x12x32xf32) <- (126x12x49x32xf32)
        transpose_42 = paddle._C_ops.transpose(matmul_34, [0, 2, 1, 3])
        del matmul_34

        # pd_op.reshape: (126x49x384xf32) <- (126x49x12x32xf32, 3xi64)
        reshape_84 = paddle._C_ops.reshape(transpose_42, full_int_array_65)

        # pd_op.matmul: (126x49x384xf32) <- (126x49x384xf32, 384x384xf32)
        matmul_35 = paddle._C_ops.matmul(reshape_84, parameter_102, False, False)
        del parameter_102

        # pd_op.add: (126x49x384xf32) <- (126x49x384xf32, 384xf32)
        add_52 = paddle._C_ops.add(matmul_35, parameter_101)
        del parameter_101

        # pd_op.reshape: (126x7x7x384xf32) <- (126x49x384xf32, 4xi64)
        reshape_85 = paddle._C_ops.reshape(add_52, full_int_array_66)

        # pd_op.reshape: (2x9x7x7x7x384xf32) <- (126x7x7x384xf32, 6xi64)
        reshape_86 = paddle._C_ops.reshape(reshape_85, full_int_array_61)

        # pd_op.transpose: (2x9x7x7x7x384xf32) <- (2x9x7x7x7x384xf32)
        transpose_43 = paddle._C_ops.transpose(reshape_86, [0, 1, 3, 2, 4, 5])
        del reshape_86

        # pd_op.reshape: (2x63x49x384xf32) <- (2x9x7x7x7x384xf32, 4xi64)
        reshape_87 = paddle._C_ops.reshape(transpose_43, full_int_array_67)

        # pd_op.roll: (2x63x49x384xf32) <- (2x63x49x384xf32, 2xi64)
        roll_5 = paddle._C_ops.roll(reshape_87, full_int_array_38, [1, 2])

        # pd_op.slice: (2x60x44x384xf32) <- (2x63x49x384xf32, 2xi64, 2xi64)
        slice_23 = paddle._C_ops.slice(
            roll_5, [1, 2], full_int_array_2, full_int_array_68, [1, 1], []
        )

        # pd_op.reshape: (2x2640x384xf32) <- (2x60x44x384xf32, 3xi64)
        reshape_88 = paddle._C_ops.reshape(slice_23, full_int_array_69)

        # pd_op.full: (xf64) <- ()
        full_17 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__8 = paddle._C_ops.assign_value_(
            full_17,
            [],
            paddle.float64,
            [float("0.954545")],
            paddle.framework._current_expected_place(),
        )
        del full_17

        # pd_op.cast: (xf32) <- (xf64)
        cast_11 = paddle._C_ops.cast(assign_value__8, paddle.float32)
        del assign_value__8

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_8 = paddle._C_ops.uniform(
            full_int_array_39,
            paddle.float32,
            full_4,
            full_1,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_53 = paddle._C_ops.add(cast_11, uniform_8)
        del uniform_8

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_8 = paddle._C_ops.floor(add_53)
        del add_53

        # pd_op.divide: (2x2640x384xf32) <- (2x2640x384xf32, xf32)
        divide_8 = paddle._C_ops.divide(reshape_88, cast_11)

        # pd_op.multiply: (2x2640x384xf32) <- (2x2640x384xf32, 2x1x1xf32)
        multiply_11 = paddle._C_ops.multiply(divide_8, floor_8)

        # pd_op.add: (2x2640x384xf32) <- (2x2640x384xf32, 2x2640x384xf32)
        add_54 = paddle._C_ops.add(add_48, multiply_11)

        # pd_op.layer_norm: (2x2640x384xf32, 2x2640xf32, 2x2640xf32) <- (2x2640x384xf32, 384xf32, 384xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_54, parameter_100, parameter_99, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_100, parameter_99

        # pd_op.matmul: (2x2640x1536xf32) <- (2x2640x384xf32, 384x1536xf32)
        matmul_36 = paddle._C_ops.matmul(layer_norm_48, parameter_98, False, False)
        del parameter_98

        # pd_op.add: (2x2640x1536xf32) <- (2x2640x1536xf32, 1536xf32)
        add_55 = paddle._C_ops.add(matmul_36, parameter_97)
        del parameter_97

        # pd_op.gelu: (2x2640x1536xf32) <- (2x2640x1536xf32)
        gelu_5 = paddle._C_ops.gelu(add_55, False)

        # pd_op.matmul: (2x2640x384xf32) <- (2x2640x1536xf32, 1536x384xf32)
        matmul_37 = paddle._C_ops.matmul(gelu_5, parameter_96, False, False)
        del parameter_96

        # pd_op.add: (2x2640x384xf32) <- (2x2640x384xf32, 384xf32)
        add_56 = paddle._C_ops.add(matmul_37, parameter_95)
        del parameter_95

        # pd_op.full: (xf64) <- ()
        full_18 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__9 = paddle._C_ops.assign_value_(
            full_18,
            [],
            paddle.float64,
            [float("0.954545")],
            paddle.framework._current_expected_place(),
        )
        del full_18

        # pd_op.cast: (xf32) <- (xf64)
        cast_12 = paddle._C_ops.cast(assign_value__9, paddle.float32)
        del assign_value__9

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_9 = paddle._C_ops.uniform(
            full_int_array_39,
            paddle.float32,
            full_4,
            full_1,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_57 = paddle._C_ops.add(cast_12, uniform_9)
        del uniform_9

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_9 = paddle._C_ops.floor(add_57)
        del add_57

        # pd_op.divide: (2x2640x384xf32) <- (2x2640x384xf32, xf32)
        divide_9 = paddle._C_ops.divide(add_56, cast_12)

        # pd_op.multiply: (2x2640x384xf32) <- (2x2640x384xf32, 2x1x1xf32)
        multiply_12 = paddle._C_ops.multiply(divide_9, floor_9)

        # pd_op.add: (2x2640x384xf32) <- (2x2640x384xf32, 2x2640x384xf32)
        add_58 = paddle._C_ops.add(add_54, multiply_12)

        # pd_op.layer_norm: (2x2640x384xf32, 2x2640xf32, 2x2640xf32) <- (2x2640x384xf32, 384xf32, 384xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_58, parameter_94, parameter_93, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_93, parameter_94

        # pd_op.reshape: (2x60x44x384xf32) <- (2x2640x384xf32, 4xi64)
        reshape_89 = paddle._C_ops.reshape(layer_norm_51, full_int_array_60)

        # pd_op.pad: (2x63x49x384xf32) <- (2x60x44x384xf32, 1xf32)
        pad_6 = paddle._C_ops.pad(reshape_89, [0, 0, 0, 3, 0, 5, 0, 0], full_4)

        # pd_op.reshape: (2x9x7x7x7x384xf32) <- (2x63x49x384xf32, 6xi64)
        reshape_90 = paddle._C_ops.reshape(pad_6, full_int_array_61)

        # pd_op.transpose: (2x9x7x7x7x384xf32) <- (2x9x7x7x7x384xf32)
        transpose_44 = paddle._C_ops.transpose(reshape_90, [0, 1, 3, 2, 4, 5])
        del reshape_90

        # pd_op.reshape: (126x7x7x384xf32) <- (2x9x7x7x7x384xf32, 4xi64)
        reshape_91 = paddle._C_ops.reshape(transpose_44, full_int_array_62)

        # pd_op.reshape: (126x49x384xf32) <- (126x7x7x384xf32, 3xi64)
        reshape_92 = paddle._C_ops.reshape(reshape_91, full_int_array_63)

        # pd_op.matmul: (126x49x1152xf32) <- (126x49x384xf32, 384x1152xf32)
        matmul_38 = paddle._C_ops.matmul(reshape_92, parameter_92, False, False)
        del parameter_92

        # pd_op.add: (126x49x1152xf32) <- (126x49x1152xf32, 1152xf32)
        add_59 = paddle._C_ops.add(matmul_38, parameter_91)
        del parameter_91

        # pd_op.reshape: (126x49x3x12x32xf32) <- (126x49x1152xf32, 5xi64)
        reshape_93 = paddle._C_ops.reshape(add_59, full_int_array_64)

        # pd_op.transpose: (3x126x12x49x32xf32) <- (126x49x3x12x32xf32)
        transpose_45 = paddle._C_ops.transpose(reshape_93, [2, 0, 3, 1, 4])
        del reshape_93

        # pd_op.slice: (126x12x49x32xf32) <- (3x126x12x49x32xf32, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(
            transpose_45, [0], full_int_array_27, full_int_array_20, [1], [0]
        )

        # pd_op.slice: (126x12x49x32xf32) <- (3x126x12x49x32xf32, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(
            transpose_45, [0], full_int_array_20, full_int_array_21, [1], [0]
        )

        # pd_op.slice: (126x12x49x32xf32) <- (3x126x12x49x32xf32, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(
            transpose_45, [0], full_int_array_21, full_int_array_28, [1], [0]
        )

        # pd_op.scale: (126x12x49x32xf32) <- (126x12x49x32xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(slice_24, full_5, float("0"), True)
        del slice_24

        # pd_op.transpose: (126x12x32x49xf32) <- (126x12x49x32xf32)
        transpose_46 = paddle._C_ops.transpose(slice_25, [0, 1, 3, 2])
        del slice_25

        # pd_op.matmul: (126x12x49x49xf32) <- (126x12x49x32xf32, 126x12x32x49xf32)
        matmul_39 = paddle._C_ops.matmul(scale_9, transpose_46, False, False)

        # pd_op.flatten: (2401xi64) <- (49x49xi64)
        flatten_8 = paddle._C_ops.flatten(data_19, 0, 1)
        del data_19

        # pd_op.index_select: (2401x12xf32) <- (169x12xf32, 2401xi64)
        index_select_6 = paddle._C_ops.index_select(data_6, flatten_8, 0)
        del data_6

        # pd_op.reshape: (49x49x12xf32) <- (2401x12xf32, 3xi64)
        reshape_94 = paddle._C_ops.reshape(index_select_6, full_int_array_29)

        # pd_op.transpose: (12x49x49xf32) <- (49x49x12xf32)
        transpose_47 = paddle._C_ops.transpose(reshape_94, [2, 0, 1])
        del reshape_94

        # pd_op.unsqueeze: (1x12x49x49xf32) <- (12x49x49xf32, 1xi64)
        unsqueeze_18 = paddle._C_ops.unsqueeze(transpose_47, full_int_array_27)

        # pd_op.add: (126x12x49x49xf32) <- (126x12x49x49xf32, 1x12x49x49xf32)
        add_60 = paddle._C_ops.add(matmul_39, unsqueeze_18)

        # pd_op.softmax: (126x12x49x49xf32) <- (126x12x49x49xf32)
        softmax_6 = paddle._C_ops.softmax(add_60, -1)
        del add_60

        # pd_op.matmul: (126x12x49x32xf32) <- (126x12x49x49xf32, 126x12x49x32xf32)
        matmul_40 = paddle._C_ops.matmul(softmax_6, slice_26, False, False)

        # pd_op.transpose: (126x49x12x32xf32) <- (126x12x49x32xf32)
        transpose_48 = paddle._C_ops.transpose(matmul_40, [0, 2, 1, 3])
        del matmul_40

        # pd_op.reshape: (126x49x384xf32) <- (126x49x12x32xf32, 3xi64)
        reshape_95 = paddle._C_ops.reshape(transpose_48, full_int_array_65)

        # pd_op.matmul: (126x49x384xf32) <- (126x49x384xf32, 384x384xf32)
        matmul_41 = paddle._C_ops.matmul(reshape_95, parameter_90, False, False)
        del parameter_90

        # pd_op.add: (126x49x384xf32) <- (126x49x384xf32, 384xf32)
        add_61 = paddle._C_ops.add(matmul_41, parameter_89)
        del parameter_89

        # pd_op.reshape: (126x7x7x384xf32) <- (126x49x384xf32, 4xi64)
        reshape_96 = paddle._C_ops.reshape(add_61, full_int_array_66)

        # pd_op.reshape: (2x9x7x7x7x384xf32) <- (126x7x7x384xf32, 6xi64)
        reshape_97 = paddle._C_ops.reshape(reshape_96, full_int_array_61)

        # pd_op.transpose: (2x9x7x7x7x384xf32) <- (2x9x7x7x7x384xf32)
        transpose_49 = paddle._C_ops.transpose(reshape_97, [0, 1, 3, 2, 4, 5])
        del reshape_97

        # pd_op.reshape: (2x63x49x384xf32) <- (2x9x7x7x7x384xf32, 4xi64)
        reshape_98 = paddle._C_ops.reshape(transpose_49, full_int_array_67)

        # pd_op.slice: (2x60x44x384xf32) <- (2x63x49x384xf32, 2xi64, 2xi64)
        slice_27 = paddle._C_ops.slice(
            reshape_98, [1, 2], full_int_array_2, full_int_array_68, [1, 1], []
        )

        # pd_op.reshape: (2x2640x384xf32) <- (2x60x44x384xf32, 3xi64)
        reshape_99 = paddle._C_ops.reshape(slice_27, full_int_array_69)

        # pd_op.full: (xf64) <- ()
        full_19 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__10 = paddle._C_ops.assign_value_(
            full_19,
            [],
            paddle.float64,
            [float("0.945455")],
            paddle.framework._current_expected_place(),
        )
        del full_19

        # pd_op.cast: (xf32) <- (xf64)
        cast_13 = paddle._C_ops.cast(assign_value__10, paddle.float32)
        del assign_value__10

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_10 = paddle._C_ops.uniform(
            full_int_array_39,
            paddle.float32,
            full_4,
            full_1,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_62 = paddle._C_ops.add(cast_13, uniform_10)
        del uniform_10

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_10 = paddle._C_ops.floor(add_62)
        del add_62

        # pd_op.divide: (2x2640x384xf32) <- (2x2640x384xf32, xf32)
        divide_10 = paddle._C_ops.divide(reshape_99, cast_13)

        # pd_op.multiply: (2x2640x384xf32) <- (2x2640x384xf32, 2x1x1xf32)
        multiply_13 = paddle._C_ops.multiply(divide_10, floor_10)

        # pd_op.add: (2x2640x384xf32) <- (2x2640x384xf32, 2x2640x384xf32)
        add_63 = paddle._C_ops.add(add_58, multiply_13)

        # pd_op.layer_norm: (2x2640x384xf32, 2x2640xf32, 2x2640xf32) <- (2x2640x384xf32, 384xf32, 384xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_63, parameter_88, parameter_87, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_87, parameter_88

        # pd_op.matmul: (2x2640x1536xf32) <- (2x2640x384xf32, 384x1536xf32)
        matmul_42 = paddle._C_ops.matmul(layer_norm_54, parameter_86, False, False)
        del parameter_86

        # pd_op.add: (2x2640x1536xf32) <- (2x2640x1536xf32, 1536xf32)
        add_64 = paddle._C_ops.add(matmul_42, parameter_85)
        del parameter_85

        # pd_op.gelu: (2x2640x1536xf32) <- (2x2640x1536xf32)
        gelu_6 = paddle._C_ops.gelu(add_64, False)

        # pd_op.matmul: (2x2640x384xf32) <- (2x2640x1536xf32, 1536x384xf32)
        matmul_43 = paddle._C_ops.matmul(gelu_6, parameter_84, False, False)
        del parameter_84

        # pd_op.add: (2x2640x384xf32) <- (2x2640x384xf32, 384xf32)
        add_65 = paddle._C_ops.add(matmul_43, parameter_83)
        del parameter_83

        # pd_op.full: (xf64) <- ()
        full_20 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__11 = paddle._C_ops.assign_value_(
            full_20,
            [],
            paddle.float64,
            [float("0.945455")],
            paddle.framework._current_expected_place(),
        )
        del full_20

        # pd_op.cast: (xf32) <- (xf64)
        cast_14 = paddle._C_ops.cast(assign_value__11, paddle.float32)
        del assign_value__11

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_11 = paddle._C_ops.uniform(
            full_int_array_39,
            paddle.float32,
            full_4,
            full_1,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_66 = paddle._C_ops.add(cast_14, uniform_11)
        del uniform_11

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_11 = paddle._C_ops.floor(add_66)
        del add_66

        # pd_op.divide: (2x2640x384xf32) <- (2x2640x384xf32, xf32)
        divide_11 = paddle._C_ops.divide(add_65, cast_14)

        # pd_op.multiply: (2x2640x384xf32) <- (2x2640x384xf32, 2x1x1xf32)
        multiply_14 = paddle._C_ops.multiply(divide_11, floor_11)

        # pd_op.add: (2x2640x384xf32) <- (2x2640x384xf32, 2x2640x384xf32)
        add_67 = paddle._C_ops.add(add_63, multiply_14)

        # pd_op.layer_norm: (2x2640x384xf32, 2x2640xf32, 2x2640xf32) <- (2x2640x384xf32, 384xf32, 384xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_67, parameter_82, parameter_81, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_81, parameter_82

        # pd_op.reshape: (2x60x44x384xf32) <- (2x2640x384xf32, 4xi64)
        reshape_100 = paddle._C_ops.reshape(layer_norm_57, full_int_array_60)

        # pd_op.pad: (2x63x49x384xf32) <- (2x60x44x384xf32, 1xf32)
        pad_7 = paddle._C_ops.pad(reshape_100, [0, 0, 0, 3, 0, 5, 0, 0], full_4)

        # pd_op.roll: (2x63x49x384xf32) <- (2x63x49x384xf32, 2xi64)
        roll_6 = paddle._C_ops.roll(pad_7, full_int_array_11, [1, 2])

        # pd_op.reshape: (2x9x7x7x7x384xf32) <- (2x63x49x384xf32, 6xi64)
        reshape_101 = paddle._C_ops.reshape(roll_6, full_int_array_61)

        # pd_op.transpose: (2x9x7x7x7x384xf32) <- (2x9x7x7x7x384xf32)
        transpose_50 = paddle._C_ops.transpose(reshape_101, [0, 1, 3, 2, 4, 5])
        del reshape_101

        # pd_op.reshape: (126x7x7x384xf32) <- (2x9x7x7x7x384xf32, 4xi64)
        reshape_102 = paddle._C_ops.reshape(transpose_50, full_int_array_62)

        # pd_op.reshape: (126x49x384xf32) <- (126x7x7x384xf32, 3xi64)
        reshape_103 = paddle._C_ops.reshape(reshape_102, full_int_array_63)

        # pd_op.matmul: (126x49x1152xf32) <- (126x49x384xf32, 384x1152xf32)
        matmul_44 = paddle._C_ops.matmul(reshape_103, parameter_80, False, False)
        del parameter_80

        # pd_op.add: (126x49x1152xf32) <- (126x49x1152xf32, 1152xf32)
        add_68 = paddle._C_ops.add(matmul_44, parameter_79)
        del parameter_79

        # pd_op.reshape: (126x49x3x12x32xf32) <- (126x49x1152xf32, 5xi64)
        reshape_104 = paddle._C_ops.reshape(add_68, full_int_array_64)

        # pd_op.transpose: (3x126x12x49x32xf32) <- (126x49x3x12x32xf32)
        transpose_51 = paddle._C_ops.transpose(reshape_104, [2, 0, 3, 1, 4])
        del reshape_104

        # pd_op.slice: (126x12x49x32xf32) <- (3x126x12x49x32xf32, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(
            transpose_51, [0], full_int_array_27, full_int_array_20, [1], [0]
        )

        # pd_op.slice: (126x12x49x32xf32) <- (3x126x12x49x32xf32, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(
            transpose_51, [0], full_int_array_20, full_int_array_21, [1], [0]
        )

        # pd_op.slice: (126x12x49x32xf32) <- (3x126x12x49x32xf32, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(
            transpose_51, [0], full_int_array_21, full_int_array_28, [1], [0]
        )

        # pd_op.scale: (126x12x49x32xf32) <- (126x12x49x32xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(slice_28, full_5, float("0"), True)
        del slice_28

        # pd_op.transpose: (126x12x32x49xf32) <- (126x12x49x32xf32)
        transpose_52 = paddle._C_ops.transpose(slice_29, [0, 1, 3, 2])
        del slice_29

        # pd_op.matmul: (126x12x49x49xf32) <- (126x12x49x32xf32, 126x12x32x49xf32)
        matmul_45 = paddle._C_ops.matmul(scale_10, transpose_52, False, False)

        # pd_op.flatten: (2401xi64) <- (49x49xi64)
        flatten_9 = paddle._C_ops.flatten(data_20, 0, 1)
        del data_20

        # pd_op.index_select: (2401x12xf32) <- (169x12xf32, 2401xi64)
        index_select_7 = paddle._C_ops.index_select(data_7, flatten_9, 0)
        del data_7

        # pd_op.reshape: (49x49x12xf32) <- (2401x12xf32, 3xi64)
        reshape_105 = paddle._C_ops.reshape(index_select_7, full_int_array_29)

        # pd_op.transpose: (12x49x49xf32) <- (49x49x12xf32)
        transpose_53 = paddle._C_ops.transpose(reshape_105, [2, 0, 1])
        del reshape_105

        # pd_op.unsqueeze: (1x12x49x49xf32) <- (12x49x49xf32, 1xi64)
        unsqueeze_19 = paddle._C_ops.unsqueeze(transpose_53, full_int_array_27)

        # pd_op.add: (126x12x49x49xf32) <- (126x12x49x49xf32, 1x12x49x49xf32)
        add_69 = paddle._C_ops.add(matmul_45, unsqueeze_19)

        # pd_op.reshape: (2x63x12x49x49xf32) <- (126x12x49x49xf32, 5xi64)
        reshape_106 = paddle._C_ops.reshape(add_69, full_int_array_70)

        # pd_op.unsqueeze: (63x1x49x49xf32) <- (63x49x49xf32, 1xi64)
        unsqueeze_20 = paddle._C_ops.unsqueeze(multiply_8, full_int_array_20)

        # pd_op.unsqueeze: (1x63x1x49x49xf32) <- (63x1x49x49xf32, 1xi64)
        unsqueeze_21 = paddle._C_ops.unsqueeze(unsqueeze_20, full_int_array_27)
        del unsqueeze_20

        # pd_op.add: (2x63x12x49x49xf32) <- (2x63x12x49x49xf32, 1x63x1x49x49xf32)
        add_70 = paddle._C_ops.add(reshape_106, unsqueeze_21)

        # pd_op.reshape: (126x12x49x49xf32) <- (2x63x12x49x49xf32, 4xi64)
        reshape_107 = paddle._C_ops.reshape(add_70, full_int_array_71)

        # pd_op.softmax: (126x12x49x49xf32) <- (126x12x49x49xf32)
        softmax_7 = paddle._C_ops.softmax(reshape_107, -1)
        del reshape_107

        # pd_op.matmul: (126x12x49x32xf32) <- (126x12x49x49xf32, 126x12x49x32xf32)
        matmul_46 = paddle._C_ops.matmul(softmax_7, slice_30, False, False)

        # pd_op.transpose: (126x49x12x32xf32) <- (126x12x49x32xf32)
        transpose_54 = paddle._C_ops.transpose(matmul_46, [0, 2, 1, 3])
        del matmul_46

        # pd_op.reshape: (126x49x384xf32) <- (126x49x12x32xf32, 3xi64)
        reshape_108 = paddle._C_ops.reshape(transpose_54, full_int_array_65)

        # pd_op.matmul: (126x49x384xf32) <- (126x49x384xf32, 384x384xf32)
        matmul_47 = paddle._C_ops.matmul(reshape_108, parameter_78, False, False)
        del parameter_78

        # pd_op.add: (126x49x384xf32) <- (126x49x384xf32, 384xf32)
        add_71 = paddle._C_ops.add(matmul_47, parameter_77)
        del parameter_77

        # pd_op.reshape: (126x7x7x384xf32) <- (126x49x384xf32, 4xi64)
        reshape_109 = paddle._C_ops.reshape(add_71, full_int_array_66)

        # pd_op.reshape: (2x9x7x7x7x384xf32) <- (126x7x7x384xf32, 6xi64)
        reshape_110 = paddle._C_ops.reshape(reshape_109, full_int_array_61)

        # pd_op.transpose: (2x9x7x7x7x384xf32) <- (2x9x7x7x7x384xf32)
        transpose_55 = paddle._C_ops.transpose(reshape_110, [0, 1, 3, 2, 4, 5])
        del reshape_110

        # pd_op.reshape: (2x63x49x384xf32) <- (2x9x7x7x7x384xf32, 4xi64)
        reshape_111 = paddle._C_ops.reshape(transpose_55, full_int_array_67)

        # pd_op.roll: (2x63x49x384xf32) <- (2x63x49x384xf32, 2xi64)
        roll_7 = paddle._C_ops.roll(reshape_111, full_int_array_38, [1, 2])

        # pd_op.slice: (2x60x44x384xf32) <- (2x63x49x384xf32, 2xi64, 2xi64)
        slice_31 = paddle._C_ops.slice(
            roll_7, [1, 2], full_int_array_2, full_int_array_68, [1, 1], []
        )

        # pd_op.reshape: (2x2640x384xf32) <- (2x60x44x384xf32, 3xi64)
        reshape_112 = paddle._C_ops.reshape(slice_31, full_int_array_69)

        # pd_op.full: (xf64) <- ()
        full_21 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__12 = paddle._C_ops.assign_value_(
            full_21,
            [],
            paddle.float64,
            [float("0.936364")],
            paddle.framework._current_expected_place(),
        )
        del full_21

        # pd_op.cast: (xf32) <- (xf64)
        cast_15 = paddle._C_ops.cast(assign_value__12, paddle.float32)
        del assign_value__12

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_12 = paddle._C_ops.uniform(
            full_int_array_39,
            paddle.float32,
            full_4,
            full_1,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_72 = paddle._C_ops.add(cast_15, uniform_12)
        del uniform_12

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_12 = paddle._C_ops.floor(add_72)
        del add_72

        # pd_op.divide: (2x2640x384xf32) <- (2x2640x384xf32, xf32)
        divide_12 = paddle._C_ops.divide(reshape_112, cast_15)

        # pd_op.multiply: (2x2640x384xf32) <- (2x2640x384xf32, 2x1x1xf32)
        multiply_15 = paddle._C_ops.multiply(divide_12, floor_12)

        # pd_op.add: (2x2640x384xf32) <- (2x2640x384xf32, 2x2640x384xf32)
        add_73 = paddle._C_ops.add(add_67, multiply_15)

        # pd_op.layer_norm: (2x2640x384xf32, 2x2640xf32, 2x2640xf32) <- (2x2640x384xf32, 384xf32, 384xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_73, parameter_76, parameter_75, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_75, parameter_76

        # pd_op.matmul: (2x2640x1536xf32) <- (2x2640x384xf32, 384x1536xf32)
        matmul_48 = paddle._C_ops.matmul(layer_norm_60, parameter_74, False, False)
        del parameter_74

        # pd_op.add: (2x2640x1536xf32) <- (2x2640x1536xf32, 1536xf32)
        add_74 = paddle._C_ops.add(matmul_48, parameter_73)
        del parameter_73

        # pd_op.gelu: (2x2640x1536xf32) <- (2x2640x1536xf32)
        gelu_7 = paddle._C_ops.gelu(add_74, False)

        # pd_op.matmul: (2x2640x384xf32) <- (2x2640x1536xf32, 1536x384xf32)
        matmul_49 = paddle._C_ops.matmul(gelu_7, parameter_72, False, False)
        del parameter_72

        # pd_op.add: (2x2640x384xf32) <- (2x2640x384xf32, 384xf32)
        add_75 = paddle._C_ops.add(matmul_49, parameter_71)
        del parameter_71

        # pd_op.full: (xf64) <- ()
        full_22 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__13 = paddle._C_ops.assign_value_(
            full_22,
            [],
            paddle.float64,
            [float("0.936364")],
            paddle.framework._current_expected_place(),
        )
        del full_22

        # pd_op.cast: (xf32) <- (xf64)
        cast_16 = paddle._C_ops.cast(assign_value__13, paddle.float32)
        del assign_value__13

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_13 = paddle._C_ops.uniform(
            full_int_array_39,
            paddle.float32,
            full_4,
            full_1,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_76 = paddle._C_ops.add(cast_16, uniform_13)
        del uniform_13

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_13 = paddle._C_ops.floor(add_76)
        del add_76

        # pd_op.divide: (2x2640x384xf32) <- (2x2640x384xf32, xf32)
        divide_13 = paddle._C_ops.divide(add_75, cast_16)

        # pd_op.multiply: (2x2640x384xf32) <- (2x2640x384xf32, 2x1x1xf32)
        multiply_16 = paddle._C_ops.multiply(divide_13, floor_13)

        # pd_op.add: (2x2640x384xf32) <- (2x2640x384xf32, 2x2640x384xf32)
        add_77 = paddle._C_ops.add(add_73, multiply_16)

        # pd_op.layer_norm: (2x2640x384xf32, 2x2640xf32, 2x2640xf32) <- (2x2640x384xf32, 384xf32, 384xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_77, parameter_70, parameter_69, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_69, parameter_70

        # pd_op.reshape: (2x60x44x384xf32) <- (2x2640x384xf32, 4xi64)
        reshape_113 = paddle._C_ops.reshape(layer_norm_63, full_int_array_60)

        # pd_op.pad: (2x63x49x384xf32) <- (2x60x44x384xf32, 1xf32)
        pad_8 = paddle._C_ops.pad(reshape_113, [0, 0, 0, 3, 0, 5, 0, 0], full_4)

        # pd_op.reshape: (2x9x7x7x7x384xf32) <- (2x63x49x384xf32, 6xi64)
        reshape_114 = paddle._C_ops.reshape(pad_8, full_int_array_61)

        # pd_op.transpose: (2x9x7x7x7x384xf32) <- (2x9x7x7x7x384xf32)
        transpose_56 = paddle._C_ops.transpose(reshape_114, [0, 1, 3, 2, 4, 5])
        del reshape_114

        # pd_op.reshape: (126x7x7x384xf32) <- (2x9x7x7x7x384xf32, 4xi64)
        reshape_115 = paddle._C_ops.reshape(transpose_56, full_int_array_62)

        # pd_op.reshape: (126x49x384xf32) <- (126x7x7x384xf32, 3xi64)
        reshape_116 = paddle._C_ops.reshape(reshape_115, full_int_array_63)

        # pd_op.matmul: (126x49x1152xf32) <- (126x49x384xf32, 384x1152xf32)
        matmul_50 = paddle._C_ops.matmul(reshape_116, parameter_68, False, False)
        del parameter_68

        # pd_op.add: (126x49x1152xf32) <- (126x49x1152xf32, 1152xf32)
        add_78 = paddle._C_ops.add(matmul_50, parameter_67)
        del parameter_67

        # pd_op.reshape: (126x49x3x12x32xf32) <- (126x49x1152xf32, 5xi64)
        reshape_117 = paddle._C_ops.reshape(add_78, full_int_array_64)

        # pd_op.transpose: (3x126x12x49x32xf32) <- (126x49x3x12x32xf32)
        transpose_57 = paddle._C_ops.transpose(reshape_117, [2, 0, 3, 1, 4])
        del reshape_117

        # pd_op.slice: (126x12x49x32xf32) <- (3x126x12x49x32xf32, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(
            transpose_57, [0], full_int_array_27, full_int_array_20, [1], [0]
        )

        # pd_op.slice: (126x12x49x32xf32) <- (3x126x12x49x32xf32, 1xi64, 1xi64)
        slice_33 = paddle._C_ops.slice(
            transpose_57, [0], full_int_array_20, full_int_array_21, [1], [0]
        )

        # pd_op.slice: (126x12x49x32xf32) <- (3x126x12x49x32xf32, 1xi64, 1xi64)
        slice_34 = paddle._C_ops.slice(
            transpose_57, [0], full_int_array_21, full_int_array_28, [1], [0]
        )

        # pd_op.scale: (126x12x49x32xf32) <- (126x12x49x32xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(slice_32, full_5, float("0"), True)
        del slice_32

        # pd_op.transpose: (126x12x32x49xf32) <- (126x12x49x32xf32)
        transpose_58 = paddle._C_ops.transpose(slice_33, [0, 1, 3, 2])
        del slice_33

        # pd_op.matmul: (126x12x49x49xf32) <- (126x12x49x32xf32, 126x12x32x49xf32)
        matmul_51 = paddle._C_ops.matmul(scale_11, transpose_58, False, False)

        # pd_op.flatten: (2401xi64) <- (49x49xi64)
        flatten_10 = paddle._C_ops.flatten(data_21, 0, 1)
        del data_21

        # pd_op.index_select: (2401x12xf32) <- (169x12xf32, 2401xi64)
        index_select_8 = paddle._C_ops.index_select(data_8, flatten_10, 0)
        del data_8

        # pd_op.reshape: (49x49x12xf32) <- (2401x12xf32, 3xi64)
        reshape_118 = paddle._C_ops.reshape(index_select_8, full_int_array_29)

        # pd_op.transpose: (12x49x49xf32) <- (49x49x12xf32)
        transpose_59 = paddle._C_ops.transpose(reshape_118, [2, 0, 1])
        del reshape_118

        # pd_op.unsqueeze: (1x12x49x49xf32) <- (12x49x49xf32, 1xi64)
        unsqueeze_22 = paddle._C_ops.unsqueeze(transpose_59, full_int_array_27)

        # pd_op.add: (126x12x49x49xf32) <- (126x12x49x49xf32, 1x12x49x49xf32)
        add_79 = paddle._C_ops.add(matmul_51, unsqueeze_22)

        # pd_op.softmax: (126x12x49x49xf32) <- (126x12x49x49xf32)
        softmax_8 = paddle._C_ops.softmax(add_79, -1)
        del add_79

        # pd_op.matmul: (126x12x49x32xf32) <- (126x12x49x49xf32, 126x12x49x32xf32)
        matmul_52 = paddle._C_ops.matmul(softmax_8, slice_34, False, False)

        # pd_op.transpose: (126x49x12x32xf32) <- (126x12x49x32xf32)
        transpose_60 = paddle._C_ops.transpose(matmul_52, [0, 2, 1, 3])
        del matmul_52

        # pd_op.reshape: (126x49x384xf32) <- (126x49x12x32xf32, 3xi64)
        reshape_119 = paddle._C_ops.reshape(transpose_60, full_int_array_65)

        # pd_op.matmul: (126x49x384xf32) <- (126x49x384xf32, 384x384xf32)
        matmul_53 = paddle._C_ops.matmul(reshape_119, parameter_66, False, False)
        del parameter_66

        # pd_op.add: (126x49x384xf32) <- (126x49x384xf32, 384xf32)
        add_80 = paddle._C_ops.add(matmul_53, parameter_65)
        del parameter_65

        # pd_op.reshape: (126x7x7x384xf32) <- (126x49x384xf32, 4xi64)
        reshape_120 = paddle._C_ops.reshape(add_80, full_int_array_66)

        # pd_op.reshape: (2x9x7x7x7x384xf32) <- (126x7x7x384xf32, 6xi64)
        reshape_121 = paddle._C_ops.reshape(reshape_120, full_int_array_61)

        # pd_op.transpose: (2x9x7x7x7x384xf32) <- (2x9x7x7x7x384xf32)
        transpose_61 = paddle._C_ops.transpose(reshape_121, [0, 1, 3, 2, 4, 5])
        del reshape_121

        # pd_op.reshape: (2x63x49x384xf32) <- (2x9x7x7x7x384xf32, 4xi64)
        reshape_122 = paddle._C_ops.reshape(transpose_61, full_int_array_67)

        # pd_op.slice: (2x60x44x384xf32) <- (2x63x49x384xf32, 2xi64, 2xi64)
        slice_35 = paddle._C_ops.slice(
            reshape_122, [1, 2], full_int_array_2, full_int_array_68, [1, 1], []
        )

        # pd_op.reshape: (2x2640x384xf32) <- (2x60x44x384xf32, 3xi64)
        reshape_123 = paddle._C_ops.reshape(slice_35, full_int_array_69)

        # pd_op.full: (xf64) <- ()
        full_23 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__14 = paddle._C_ops.assign_value_(
            full_23,
            [],
            paddle.float64,
            [float("0.927273")],
            paddle.framework._current_expected_place(),
        )
        del full_23

        # pd_op.cast: (xf32) <- (xf64)
        cast_17 = paddle._C_ops.cast(assign_value__14, paddle.float32)
        del assign_value__14

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_14 = paddle._C_ops.uniform(
            full_int_array_39,
            paddle.float32,
            full_4,
            full_1,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_81 = paddle._C_ops.add(cast_17, uniform_14)
        del uniform_14

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_14 = paddle._C_ops.floor(add_81)
        del add_81

        # pd_op.divide: (2x2640x384xf32) <- (2x2640x384xf32, xf32)
        divide_14 = paddle._C_ops.divide(reshape_123, cast_17)

        # pd_op.multiply: (2x2640x384xf32) <- (2x2640x384xf32, 2x1x1xf32)
        multiply_17 = paddle._C_ops.multiply(divide_14, floor_14)

        # pd_op.add: (2x2640x384xf32) <- (2x2640x384xf32, 2x2640x384xf32)
        add_82 = paddle._C_ops.add(add_77, multiply_17)

        # pd_op.layer_norm: (2x2640x384xf32, 2x2640xf32, 2x2640xf32) <- (2x2640x384xf32, 384xf32, 384xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_82, parameter_64, parameter_63, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_63, parameter_64

        # pd_op.matmul: (2x2640x1536xf32) <- (2x2640x384xf32, 384x1536xf32)
        matmul_54 = paddle._C_ops.matmul(layer_norm_66, parameter_62, False, False)
        del parameter_62

        # pd_op.add: (2x2640x1536xf32) <- (2x2640x1536xf32, 1536xf32)
        add_83 = paddle._C_ops.add(matmul_54, parameter_61)
        del parameter_61

        # pd_op.gelu: (2x2640x1536xf32) <- (2x2640x1536xf32)
        gelu_8 = paddle._C_ops.gelu(add_83, False)

        # pd_op.matmul: (2x2640x384xf32) <- (2x2640x1536xf32, 1536x384xf32)
        matmul_55 = paddle._C_ops.matmul(gelu_8, parameter_60, False, False)
        del parameter_60

        # pd_op.add: (2x2640x384xf32) <- (2x2640x384xf32, 384xf32)
        add_84 = paddle._C_ops.add(matmul_55, parameter_59)
        del parameter_59

        # pd_op.full: (xf64) <- ()
        full_24 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__15 = paddle._C_ops.assign_value_(
            full_24,
            [],
            paddle.float64,
            [float("0.927273")],
            paddle.framework._current_expected_place(),
        )
        del full_24

        # pd_op.cast: (xf32) <- (xf64)
        cast_18 = paddle._C_ops.cast(assign_value__15, paddle.float32)
        del assign_value__15

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_15 = paddle._C_ops.uniform(
            full_int_array_39,
            paddle.float32,
            full_4,
            full_1,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_85 = paddle._C_ops.add(cast_18, uniform_15)
        del uniform_15

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_15 = paddle._C_ops.floor(add_85)
        del add_85

        # pd_op.divide: (2x2640x384xf32) <- (2x2640x384xf32, xf32)
        divide_15 = paddle._C_ops.divide(add_84, cast_18)

        # pd_op.multiply: (2x2640x384xf32) <- (2x2640x384xf32, 2x1x1xf32)
        multiply_18 = paddle._C_ops.multiply(divide_15, floor_15)

        # pd_op.add: (2x2640x384xf32) <- (2x2640x384xf32, 2x2640x384xf32)
        add_86 = paddle._C_ops.add(add_82, multiply_18)

        # pd_op.layer_norm: (2x2640x384xf32, 2x2640xf32, 2x2640xf32) <- (2x2640x384xf32, 384xf32, 384xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_86, parameter_58, parameter_57, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_57, parameter_58

        # pd_op.reshape: (2x60x44x384xf32) <- (2x2640x384xf32, 4xi64)
        reshape_124 = paddle._C_ops.reshape(layer_norm_69, full_int_array_60)

        # pd_op.pad: (2x63x49x384xf32) <- (2x60x44x384xf32, 1xf32)
        pad_9 = paddle._C_ops.pad(reshape_124, [0, 0, 0, 3, 0, 5, 0, 0], full_4)

        # pd_op.roll: (2x63x49x384xf32) <- (2x63x49x384xf32, 2xi64)
        roll_8 = paddle._C_ops.roll(pad_9, full_int_array_11, [1, 2])

        # pd_op.reshape: (2x9x7x7x7x384xf32) <- (2x63x49x384xf32, 6xi64)
        reshape_125 = paddle._C_ops.reshape(roll_8, full_int_array_61)

        # pd_op.transpose: (2x9x7x7x7x384xf32) <- (2x9x7x7x7x384xf32)
        transpose_62 = paddle._C_ops.transpose(reshape_125, [0, 1, 3, 2, 4, 5])
        del reshape_125

        # pd_op.reshape: (126x7x7x384xf32) <- (2x9x7x7x7x384xf32, 4xi64)
        reshape_126 = paddle._C_ops.reshape(transpose_62, full_int_array_62)
        del full_int_array_62

        # pd_op.reshape: (126x49x384xf32) <- (126x7x7x384xf32, 3xi64)
        reshape_127 = paddle._C_ops.reshape(reshape_126, full_int_array_63)
        del full_int_array_63

        # pd_op.matmul: (126x49x1152xf32) <- (126x49x384xf32, 384x1152xf32)
        matmul_56 = paddle._C_ops.matmul(reshape_127, parameter_56, False, False)
        del parameter_56

        # pd_op.add: (126x49x1152xf32) <- (126x49x1152xf32, 1152xf32)
        add_87 = paddle._C_ops.add(matmul_56, parameter_55)
        del parameter_55

        # pd_op.reshape: (126x49x3x12x32xf32) <- (126x49x1152xf32, 5xi64)
        reshape_128 = paddle._C_ops.reshape(add_87, full_int_array_64)
        del full_int_array_64

        # pd_op.transpose: (3x126x12x49x32xf32) <- (126x49x3x12x32xf32)
        transpose_63 = paddle._C_ops.transpose(reshape_128, [2, 0, 3, 1, 4])
        del reshape_128

        # pd_op.slice: (126x12x49x32xf32) <- (3x126x12x49x32xf32, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(
            transpose_63, [0], full_int_array_27, full_int_array_20, [1], [0]
        )

        # pd_op.slice: (126x12x49x32xf32) <- (3x126x12x49x32xf32, 1xi64, 1xi64)
        slice_37 = paddle._C_ops.slice(
            transpose_63, [0], full_int_array_20, full_int_array_21, [1], [0]
        )

        # pd_op.slice: (126x12x49x32xf32) <- (3x126x12x49x32xf32, 1xi64, 1xi64)
        slice_38 = paddle._C_ops.slice(
            transpose_63, [0], full_int_array_21, full_int_array_28, [1], [0]
        )

        # pd_op.scale: (126x12x49x32xf32) <- (126x12x49x32xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(slice_36, full_5, float("0"), True)
        del slice_36

        # pd_op.transpose: (126x12x32x49xf32) <- (126x12x49x32xf32)
        transpose_64 = paddle._C_ops.transpose(slice_37, [0, 1, 3, 2])
        del slice_37

        # pd_op.matmul: (126x12x49x49xf32) <- (126x12x49x32xf32, 126x12x32x49xf32)
        matmul_57 = paddle._C_ops.matmul(scale_12, transpose_64, False, False)

        # pd_op.flatten: (2401xi64) <- (49x49xi64)
        flatten_11 = paddle._C_ops.flatten(data_22, 0, 1)
        del data_22

        # pd_op.index_select: (2401x12xf32) <- (169x12xf32, 2401xi64)
        index_select_9 = paddle._C_ops.index_select(data_9, flatten_11, 0)
        del data_9

        # pd_op.reshape: (49x49x12xf32) <- (2401x12xf32, 3xi64)
        reshape_129 = paddle._C_ops.reshape(index_select_9, full_int_array_29)

        # pd_op.transpose: (12x49x49xf32) <- (49x49x12xf32)
        transpose_65 = paddle._C_ops.transpose(reshape_129, [2, 0, 1])
        del reshape_129

        # pd_op.unsqueeze: (1x12x49x49xf32) <- (12x49x49xf32, 1xi64)
        unsqueeze_23 = paddle._C_ops.unsqueeze(transpose_65, full_int_array_27)

        # pd_op.add: (126x12x49x49xf32) <- (126x12x49x49xf32, 1x12x49x49xf32)
        add_88 = paddle._C_ops.add(matmul_57, unsqueeze_23)

        # pd_op.reshape: (2x63x12x49x49xf32) <- (126x12x49x49xf32, 5xi64)
        reshape_130 = paddle._C_ops.reshape(add_88, full_int_array_70)
        del full_int_array_70

        # pd_op.unsqueeze: (63x1x49x49xf32) <- (63x49x49xf32, 1xi64)
        unsqueeze_24 = paddle._C_ops.unsqueeze(multiply_8, full_int_array_20)
        del multiply_8

        # pd_op.unsqueeze: (1x63x1x49x49xf32) <- (63x1x49x49xf32, 1xi64)
        unsqueeze_25 = paddle._C_ops.unsqueeze(unsqueeze_24, full_int_array_27)
        del unsqueeze_24

        # pd_op.add: (2x63x12x49x49xf32) <- (2x63x12x49x49xf32, 1x63x1x49x49xf32)
        add_89 = paddle._C_ops.add(reshape_130, unsqueeze_25)

        # pd_op.reshape: (126x12x49x49xf32) <- (2x63x12x49x49xf32, 4xi64)
        reshape_131 = paddle._C_ops.reshape(add_89, full_int_array_71)
        del full_int_array_71

        # pd_op.softmax: (126x12x49x49xf32) <- (126x12x49x49xf32)
        softmax_9 = paddle._C_ops.softmax(reshape_131, -1)
        del reshape_131

        # pd_op.matmul: (126x12x49x32xf32) <- (126x12x49x49xf32, 126x12x49x32xf32)
        matmul_58 = paddle._C_ops.matmul(softmax_9, slice_38, False, False)

        # pd_op.transpose: (126x49x12x32xf32) <- (126x12x49x32xf32)
        transpose_66 = paddle._C_ops.transpose(matmul_58, [0, 2, 1, 3])
        del matmul_58

        # pd_op.reshape: (126x49x384xf32) <- (126x49x12x32xf32, 3xi64)
        reshape_132 = paddle._C_ops.reshape(transpose_66, full_int_array_65)
        del full_int_array_65

        # pd_op.matmul: (126x49x384xf32) <- (126x49x384xf32, 384x384xf32)
        matmul_59 = paddle._C_ops.matmul(reshape_132, parameter_54, False, False)
        del parameter_54

        # pd_op.add: (126x49x384xf32) <- (126x49x384xf32, 384xf32)
        add_90 = paddle._C_ops.add(matmul_59, parameter_53)
        del parameter_53

        # pd_op.reshape: (126x7x7x384xf32) <- (126x49x384xf32, 4xi64)
        reshape_133 = paddle._C_ops.reshape(add_90, full_int_array_66)
        del full_int_array_66

        # pd_op.reshape: (2x9x7x7x7x384xf32) <- (126x7x7x384xf32, 6xi64)
        reshape_134 = paddle._C_ops.reshape(reshape_133, full_int_array_61)
        del full_int_array_61

        # pd_op.transpose: (2x9x7x7x7x384xf32) <- (2x9x7x7x7x384xf32)
        transpose_67 = paddle._C_ops.transpose(reshape_134, [0, 1, 3, 2, 4, 5])
        del reshape_134

        # pd_op.reshape: (2x63x49x384xf32) <- (2x9x7x7x7x384xf32, 4xi64)
        reshape_135 = paddle._C_ops.reshape(transpose_67, full_int_array_67)
        del full_int_array_67

        # pd_op.roll: (2x63x49x384xf32) <- (2x63x49x384xf32, 2xi64)
        roll_9 = paddle._C_ops.roll(reshape_135, full_int_array_38, [1, 2])

        # pd_op.slice: (2x60x44x384xf32) <- (2x63x49x384xf32, 2xi64, 2xi64)
        slice_39 = paddle._C_ops.slice(
            roll_9, [1, 2], full_int_array_2, full_int_array_68, [1, 1], []
        )

        # pd_op.reshape: (2x2640x384xf32) <- (2x60x44x384xf32, 3xi64)
        reshape_136 = paddle._C_ops.reshape(slice_39, full_int_array_69)
        del full_int_array_69

        # pd_op.full: (xf64) <- ()
        full_25 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__16 = paddle._C_ops.assign_value_(
            full_25,
            [],
            paddle.float64,
            [float("0.918182")],
            paddle.framework._current_expected_place(),
        )
        del full_25

        # pd_op.cast: (xf32) <- (xf64)
        cast_19 = paddle._C_ops.cast(assign_value__16, paddle.float32)
        del assign_value__16

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_16 = paddle._C_ops.uniform(
            full_int_array_39,
            paddle.float32,
            full_4,
            full_1,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_91 = paddle._C_ops.add(cast_19, uniform_16)
        del uniform_16

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_16 = paddle._C_ops.floor(add_91)
        del add_91

        # pd_op.divide: (2x2640x384xf32) <- (2x2640x384xf32, xf32)
        divide_16 = paddle._C_ops.divide(reshape_136, cast_19)

        # pd_op.multiply: (2x2640x384xf32) <- (2x2640x384xf32, 2x1x1xf32)
        multiply_19 = paddle._C_ops.multiply(divide_16, floor_16)

        # pd_op.add: (2x2640x384xf32) <- (2x2640x384xf32, 2x2640x384xf32)
        add_92 = paddle._C_ops.add(add_86, multiply_19)

        # pd_op.layer_norm: (2x2640x384xf32, 2x2640xf32, 2x2640xf32) <- (2x2640x384xf32, 384xf32, 384xf32)
        layer_norm_72, layer_norm_73, layer_norm_74 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_92, parameter_52, parameter_51, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_51, parameter_52

        # pd_op.matmul: (2x2640x1536xf32) <- (2x2640x384xf32, 384x1536xf32)
        matmul_60 = paddle._C_ops.matmul(layer_norm_72, parameter_50, False, False)
        del parameter_50

        # pd_op.add: (2x2640x1536xf32) <- (2x2640x1536xf32, 1536xf32)
        add_93 = paddle._C_ops.add(matmul_60, parameter_49)
        del parameter_49

        # pd_op.gelu: (2x2640x1536xf32) <- (2x2640x1536xf32)
        gelu_9 = paddle._C_ops.gelu(add_93, False)

        # pd_op.matmul: (2x2640x384xf32) <- (2x2640x1536xf32, 1536x384xf32)
        matmul_61 = paddle._C_ops.matmul(gelu_9, parameter_48, False, False)
        del parameter_48

        # pd_op.add: (2x2640x384xf32) <- (2x2640x384xf32, 384xf32)
        add_94 = paddle._C_ops.add(matmul_61, parameter_47)
        del parameter_47

        # pd_op.full: (xf64) <- ()
        full_26 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__17 = paddle._C_ops.assign_value_(
            full_26,
            [],
            paddle.float64,
            [float("0.918182")],
            paddle.framework._current_expected_place(),
        )
        del full_26

        # pd_op.cast: (xf32) <- (xf64)
        cast_20 = paddle._C_ops.cast(assign_value__17, paddle.float32)
        del assign_value__17

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_17 = paddle._C_ops.uniform(
            full_int_array_39,
            paddle.float32,
            full_4,
            full_1,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_95 = paddle._C_ops.add(cast_20, uniform_17)
        del uniform_17

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_17 = paddle._C_ops.floor(add_95)
        del add_95

        # pd_op.divide: (2x2640x384xf32) <- (2x2640x384xf32, xf32)
        divide_17 = paddle._C_ops.divide(add_94, cast_20)

        # pd_op.multiply: (2x2640x384xf32) <- (2x2640x384xf32, 2x1x1xf32)
        multiply_20 = paddle._C_ops.multiply(divide_17, floor_17)

        # pd_op.add: (2x2640x384xf32) <- (2x2640x384xf32, 2x2640x384xf32)
        add_96 = paddle._C_ops.add(add_92, multiply_20)

        # pd_op.reshape: (2x60x44x384xf32) <- (2x2640x384xf32, 4xi64)
        reshape_137 = paddle._C_ops.reshape(add_96, full_int_array_60)

        # pd_op.strided_slice: (2x30x22x384xf32) <- (2x60x44x384xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_8 = paddle._C_ops.strided_slice(
            reshape_137, [1, 2], full_int_array_2, full_int_array_16, full_int_array_40
        )

        # pd_op.strided_slice: (2x30x22x384xf32) <- (2x60x44x384xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_9 = paddle._C_ops.strided_slice(
            reshape_137, [1, 2], full_int_array_41, full_int_array_16, full_int_array_40
        )

        # pd_op.strided_slice: (2x30x22x384xf32) <- (2x60x44x384xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_10 = paddle._C_ops.strided_slice(
            reshape_137, [1, 2], full_int_array_42, full_int_array_16, full_int_array_40
        )

        # pd_op.strided_slice: (2x30x22x384xf32) <- (2x60x44x384xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_11 = paddle._C_ops.strided_slice(
            reshape_137, [1, 2], full_int_array_4, full_int_array_16, full_int_array_40
        )

        # builtin.combine: ([2x30x22x384xf32, 2x30x22x384xf32, 2x30x22x384xf32, 2x30x22x384xf32]) <- (2x30x22x384xf32, 2x30x22x384xf32, 2x30x22x384xf32, 2x30x22x384xf32)
        combine_2 = [
            strided_slice_8,
            strided_slice_9,
            strided_slice_10,
            strided_slice_11,
        ]

        # pd_op.concat: (2x30x22x1536xf32) <- ([2x30x22x384xf32, 2x30x22x384xf32, 2x30x22x384xf32, 2x30x22x384xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_8)
        del combine_2

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_72 = [-1, 660, 1536]

        # pd_op.reshape: (2x660x1536xf32) <- (2x30x22x1536xf32, 3xi64)
        reshape_138 = paddle._C_ops.reshape(concat_2, full_int_array_72)
        del full_int_array_72

        # pd_op.layer_norm: (2x660x1536xf32, 2x660xf32, 2x660xf32) <- (2x660x1536xf32, 1536xf32, 1536xf32)
        layer_norm_75, layer_norm_76, layer_norm_77 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                reshape_138, parameter_46, parameter_45, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_45, parameter_46

        # pd_op.matmul: (2x660x768xf32) <- (2x660x1536xf32, 1536x768xf32)
        matmul_62 = paddle._C_ops.matmul(layer_norm_75, parameter_44, False, False)
        del parameter_44

        # pd_op.layer_norm: (2x2640x384xf32, 2x2640xf32, 2x2640xf32) <- (2x2640x384xf32, 384xf32, 384xf32)
        layer_norm_78, layer_norm_79, layer_norm_80 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_96, parameter_43, parameter_42, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_42, parameter_43

        # pd_op.reshape: (2x60x44x384xf32) <- (2x2640x384xf32, 4xi64)
        reshape_139 = paddle._C_ops.reshape(layer_norm_78, full_int_array_60)
        del full_int_array_60

        # pd_op.transpose: (2x384x60x44xf32) <- (2x60x44x384xf32)
        transpose_68 = paddle._C_ops.transpose(reshape_139, [0, 3, 1, 2])
        del reshape_139

        # pd_op.full: (1x35x28x1xf32) <- ()
        full_27 = paddle._C_ops.full(
            [1, 35, 28, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x35x28x1xf32) <- (1x35x28x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__27 = paddle._C_ops.set_value_(
            full_27,
            full_int_array_2,
            full_int_array_3,
            full_int_array_4,
            [1, 2],
            [],
            [],
            [1],
            [float("0")],
        )
        del full_27

        # pd_op.set_value_: (1x35x28x1xf32) <- (1x35x28x1xf32, 2xi64, 2xi64, 2xi64)
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

        # pd_op.set_value_: (1x35x28x1xf32) <- (1x35x28x1xf32, 2xi64, 2xi64, 2xi64)
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

        # pd_op.set_value_: (1x35x28x1xf32) <- (1x35x28x1xf32, 2xi64, 2xi64, 2xi64)
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

        # pd_op.set_value_: (1x35x28x1xf32) <- (1x35x28x1xf32, 2xi64, 2xi64, 2xi64)
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

        # pd_op.set_value_: (1x35x28x1xf32) <- (1x35x28x1xf32, 2xi64, 2xi64, 2xi64)
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

        # pd_op.set_value_: (1x35x28x1xf32) <- (1x35x28x1xf32, 2xi64, 2xi64, 2xi64)
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

        # pd_op.set_value_: (1x35x28x1xf32) <- (1x35x28x1xf32, 2xi64, 2xi64, 2xi64)
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

        # pd_op.set_value_: (1x35x28x1xf32) <- (1x35x28x1xf32, 2xi64, 2xi64, 2xi64)
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
        del full_int_array_16, set_value__34

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_73 = [-1, 5, 7, 4, 7, 1]

        # pd_op.reshape: (1x5x7x4x7x1xf32) <- (1x35x28x1xf32, 6xi64)
        reshape_140 = paddle._C_ops.reshape(set_value__35, full_int_array_73)
        del full_int_array_73

        # pd_op.transpose: (1x5x4x7x7x1xf32) <- (1x5x7x4x7x1xf32)
        transpose_69 = paddle._C_ops.transpose(reshape_140, [0, 1, 3, 2, 4, 5])
        del reshape_140

        # pd_op.reshape: (20x7x7x1xf32) <- (1x5x4x7x7x1xf32, 4xi64)
        reshape_141 = paddle._C_ops.reshape(transpose_69, full_int_array_18)
        del full_int_array_18, transpose_69

        # pd_op.reshape: (20x49xf32) <- (20x7x7x1xf32, 2xi64)
        reshape_142 = paddle._C_ops.reshape(reshape_141, full_int_array_19)
        del full_int_array_19, reshape_141

        # pd_op.unsqueeze: (20x1x49xf32) <- (20x49xf32, 1xi64)
        unsqueeze_26 = paddle._C_ops.unsqueeze(reshape_142, full_int_array_20)

        # pd_op.unsqueeze: (20x49x1xf32) <- (20x49xf32, 1xi64)
        unsqueeze_27 = paddle._C_ops.unsqueeze(reshape_142, full_int_array_21)
        del reshape_142

        # pd_op.subtract: (20x49x49xf32) <- (20x1x49xf32, 20x49x1xf32)
        subtract_3 = paddle._C_ops.subtract(unsqueeze_26, unsqueeze_27)
        del unsqueeze_26, unsqueeze_27

        # pd_op.full_like: (20x49x49xf32) <- (20x49x49xf32, 1xf32)
        full_like_3 = paddle._C_ops.full_like(
            subtract_3,
            full_1,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.scale: (20x49x49xf32) <- (20x49x49xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(full_like_3, full_2, float("0"), True)
        del full_2, full_like_3

        # pd_op.not_equal: (20x49x49xb) <- (20x49x49xf32, xf32)
        not_equal_3 = paddle._C_ops.not_equal(subtract_3, full_3)
        del full_3, subtract_3

        # pd_op.cast: (20x49x49xf32) <- (20x49x49xb)
        cast_21 = paddle._C_ops.cast(not_equal_3, paddle.float32)
        del not_equal_3

        # pd_op.multiply: (20x49x49xf32) <- (20x49x49xf32, 20x49x49xf32)
        multiply_21 = paddle._C_ops.multiply(scale_13, cast_21)
        del cast_21, scale_13

        # pd_op.layer_norm: (2x660x768xf32, 2x660xf32, 2x660xf32) <- (2x660x768xf32, 768xf32, 768xf32)
        layer_norm_81, layer_norm_82, layer_norm_83 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                matmul_62, parameter_41, parameter_40, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_40, parameter_41

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_74 = [-1, 30, 22, 768]

        # pd_op.reshape: (2x30x22x768xf32) <- (2x660x768xf32, 4xi64)
        reshape_143 = paddle._C_ops.reshape(layer_norm_81, full_int_array_74)

        # pd_op.pad: (2x35x28x768xf32) <- (2x30x22x768xf32, 1xf32)
        pad_10 = paddle._C_ops.pad(reshape_143, [0, 0, 0, 5, 0, 6, 0, 0], full_4)

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_75 = [-1, 5, 7, 4, 7, 768]

        # pd_op.reshape: (2x5x7x4x7x768xf32) <- (2x35x28x768xf32, 6xi64)
        reshape_144 = paddle._C_ops.reshape(pad_10, full_int_array_75)

        # pd_op.transpose: (2x5x4x7x7x768xf32) <- (2x5x7x4x7x768xf32)
        transpose_70 = paddle._C_ops.transpose(reshape_144, [0, 1, 3, 2, 4, 5])
        del reshape_144

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_76 = [-1, 7, 7, 768]

        # pd_op.reshape: (40x7x7x768xf32) <- (2x5x4x7x7x768xf32, 4xi64)
        reshape_145 = paddle._C_ops.reshape(transpose_70, full_int_array_76)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_77 = [40, 49, 768]

        # pd_op.reshape: (40x49x768xf32) <- (40x7x7x768xf32, 3xi64)
        reshape_146 = paddle._C_ops.reshape(reshape_145, full_int_array_77)

        # pd_op.matmul: (40x49x2304xf32) <- (40x49x768xf32, 768x2304xf32)
        matmul_63 = paddle._C_ops.matmul(reshape_146, parameter_39, False, False)
        del parameter_39

        # pd_op.add: (40x49x2304xf32) <- (40x49x2304xf32, 2304xf32)
        add_97 = paddle._C_ops.add(matmul_63, parameter_38)
        del parameter_38

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_78 = [-1, 49, 3, 24, 32]

        # pd_op.reshape: (40x49x3x24x32xf32) <- (40x49x2304xf32, 5xi64)
        reshape_147 = paddle._C_ops.reshape(add_97, full_int_array_78)

        # pd_op.transpose: (3x40x24x49x32xf32) <- (40x49x3x24x32xf32)
        transpose_71 = paddle._C_ops.transpose(reshape_147, [2, 0, 3, 1, 4])
        del reshape_147

        # pd_op.slice: (40x24x49x32xf32) <- (3x40x24x49x32xf32, 1xi64, 1xi64)
        slice_40 = paddle._C_ops.slice(
            transpose_71, [0], full_int_array_27, full_int_array_20, [1], [0]
        )

        # pd_op.slice: (40x24x49x32xf32) <- (3x40x24x49x32xf32, 1xi64, 1xi64)
        slice_41 = paddle._C_ops.slice(
            transpose_71, [0], full_int_array_20, full_int_array_21, [1], [0]
        )

        # pd_op.slice: (40x24x49x32xf32) <- (3x40x24x49x32xf32, 1xi64, 1xi64)
        slice_42 = paddle._C_ops.slice(
            transpose_71, [0], full_int_array_21, full_int_array_28, [1], [0]
        )

        # pd_op.scale: (40x24x49x32xf32) <- (40x24x49x32xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(slice_40, full_5, float("0"), True)
        del slice_40

        # pd_op.transpose: (40x24x32x49xf32) <- (40x24x49x32xf32)
        transpose_72 = paddle._C_ops.transpose(slice_41, [0, 1, 3, 2])
        del slice_41

        # pd_op.matmul: (40x24x49x49xf32) <- (40x24x49x32xf32, 40x24x32x49xf32)
        matmul_64 = paddle._C_ops.matmul(scale_14, transpose_72, False, False)

        # pd_op.flatten: (2401xi64) <- (49x49xi64)
        flatten_12 = paddle._C_ops.flatten(data_23, 0, 1)
        del data_23

        # pd_op.index_select: (2401x24xf32) <- (169x24xf32, 2401xi64)
        index_select_10 = paddle._C_ops.index_select(data_10, flatten_12, 0)
        del data_10

        # pd_op.reshape: (49x49x24xf32) <- (2401x24xf32, 3xi64)
        reshape_148 = paddle._C_ops.reshape(index_select_10, full_int_array_29)

        # pd_op.transpose: (24x49x49xf32) <- (49x49x24xf32)
        transpose_73 = paddle._C_ops.transpose(reshape_148, [2, 0, 1])
        del reshape_148

        # pd_op.unsqueeze: (1x24x49x49xf32) <- (24x49x49xf32, 1xi64)
        unsqueeze_28 = paddle._C_ops.unsqueeze(transpose_73, full_int_array_27)

        # pd_op.add: (40x24x49x49xf32) <- (40x24x49x49xf32, 1x24x49x49xf32)
        add_98 = paddle._C_ops.add(matmul_64, unsqueeze_28)

        # pd_op.softmax: (40x24x49x49xf32) <- (40x24x49x49xf32)
        softmax_10 = paddle._C_ops.softmax(add_98, -1)
        del add_98

        # pd_op.matmul: (40x24x49x32xf32) <- (40x24x49x49xf32, 40x24x49x32xf32)
        matmul_65 = paddle._C_ops.matmul(softmax_10, slice_42, False, False)

        # pd_op.transpose: (40x49x24x32xf32) <- (40x24x49x32xf32)
        transpose_74 = paddle._C_ops.transpose(matmul_65, [0, 2, 1, 3])
        del matmul_65

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_79 = [-1, 49, 768]

        # pd_op.reshape: (40x49x768xf32) <- (40x49x24x32xf32, 3xi64)
        reshape_149 = paddle._C_ops.reshape(transpose_74, full_int_array_79)

        # pd_op.matmul: (40x49x768xf32) <- (40x49x768xf32, 768x768xf32)
        matmul_66 = paddle._C_ops.matmul(reshape_149, parameter_37, False, False)
        del parameter_37

        # pd_op.add: (40x49x768xf32) <- (40x49x768xf32, 768xf32)
        add_99 = paddle._C_ops.add(matmul_66, parameter_36)
        del parameter_36

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_80 = [40, 7, 7, 768]

        # pd_op.reshape: (40x7x7x768xf32) <- (40x49x768xf32, 4xi64)
        reshape_150 = paddle._C_ops.reshape(add_99, full_int_array_80)

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_81 = [-1, 5, 4, 7, 7, 768]

        # pd_op.reshape: (2x5x4x7x7x768xf32) <- (40x7x7x768xf32, 6xi64)
        reshape_151 = paddle._C_ops.reshape(reshape_150, full_int_array_81)

        # pd_op.transpose: (2x5x7x4x7x768xf32) <- (2x5x4x7x7x768xf32)
        transpose_75 = paddle._C_ops.transpose(reshape_151, [0, 1, 3, 2, 4, 5])
        del reshape_151

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_82 = [-1, 35, 28, 768]

        # pd_op.reshape: (2x35x28x768xf32) <- (2x5x7x4x7x768xf32, 4xi64)
        reshape_152 = paddle._C_ops.reshape(transpose_75, full_int_array_82)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_83 = [30, 22]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_170 = full_int_array_83

        # pd_op.slice: (2x30x22x768xf32) <- (2x35x28x768xf32, 2xi64, 2xi64)
        slice_43 = paddle._C_ops.slice(
            reshape_152, [1, 2], full_int_array_2, full_int_array_83, [1, 1], []
        )

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_84 = [-1, 660, 768]

        # pd_op.reshape: (2x660x768xf32) <- (2x30x22x768xf32, 3xi64)
        reshape_153 = paddle._C_ops.reshape(slice_43, full_int_array_84)

        # pd_op.full: (xf64) <- ()
        full_28 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__18 = paddle._C_ops.assign_value_(
            full_28,
            [],
            paddle.float64,
            [float("0.909091")],
            paddle.framework._current_expected_place(),
        )
        del full_28

        # pd_op.cast: (xf32) <- (xf64)
        cast_22 = paddle._C_ops.cast(assign_value__18, paddle.float32)
        del assign_value__18

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_18 = paddle._C_ops.uniform(
            full_int_array_39,
            paddle.float32,
            full_4,
            full_1,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_100 = paddle._C_ops.add(cast_22, uniform_18)
        del uniform_18

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_18 = paddle._C_ops.floor(add_100)
        del add_100

        # pd_op.divide: (2x660x768xf32) <- (2x660x768xf32, xf32)
        divide_18 = paddle._C_ops.divide(reshape_153, cast_22)

        # pd_op.multiply: (2x660x768xf32) <- (2x660x768xf32, 2x1x1xf32)
        multiply_22 = paddle._C_ops.multiply(divide_18, floor_18)

        # pd_op.add: (2x660x768xf32) <- (2x660x768xf32, 2x660x768xf32)
        add_101 = paddle._C_ops.add(matmul_62, multiply_22)

        # pd_op.layer_norm: (2x660x768xf32, 2x660xf32, 2x660xf32) <- (2x660x768xf32, 768xf32, 768xf32)
        layer_norm_84, layer_norm_85, layer_norm_86 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_101, parameter_35, parameter_34, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_34, parameter_35

        # pd_op.matmul: (2x660x3072xf32) <- (2x660x768xf32, 768x3072xf32)
        matmul_67 = paddle._C_ops.matmul(layer_norm_84, parameter_33, False, False)
        del parameter_33

        # pd_op.add: (2x660x3072xf32) <- (2x660x3072xf32, 3072xf32)
        add_102 = paddle._C_ops.add(matmul_67, parameter_32)
        del parameter_32

        # pd_op.gelu: (2x660x3072xf32) <- (2x660x3072xf32)
        gelu_10 = paddle._C_ops.gelu(add_102, False)

        # pd_op.matmul: (2x660x768xf32) <- (2x660x3072xf32, 3072x768xf32)
        matmul_68 = paddle._C_ops.matmul(gelu_10, parameter_31, False, False)
        del parameter_31

        # pd_op.add: (2x660x768xf32) <- (2x660x768xf32, 768xf32)
        add_103 = paddle._C_ops.add(matmul_68, parameter_30)
        del parameter_30

        # pd_op.full: (xf64) <- ()
        full_29 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__19 = paddle._C_ops.assign_value_(
            full_29,
            [],
            paddle.float64,
            [float("0.909091")],
            paddle.framework._current_expected_place(),
        )
        del full_29

        # pd_op.cast: (xf32) <- (xf64)
        cast_23 = paddle._C_ops.cast(assign_value__19, paddle.float32)
        del assign_value__19

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_19 = paddle._C_ops.uniform(
            full_int_array_39,
            paddle.float32,
            full_4,
            full_1,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_104 = paddle._C_ops.add(cast_23, uniform_19)
        del uniform_19

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_19 = paddle._C_ops.floor(add_104)
        del add_104

        # pd_op.divide: (2x660x768xf32) <- (2x660x768xf32, xf32)
        divide_19 = paddle._C_ops.divide(add_103, cast_23)

        # pd_op.multiply: (2x660x768xf32) <- (2x660x768xf32, 2x1x1xf32)
        multiply_23 = paddle._C_ops.multiply(divide_19, floor_19)

        # pd_op.add: (2x660x768xf32) <- (2x660x768xf32, 2x660x768xf32)
        add_105 = paddle._C_ops.add(add_101, multiply_23)

        # pd_op.layer_norm: (2x660x768xf32, 2x660xf32, 2x660xf32) <- (2x660x768xf32, 768xf32, 768xf32)
        layer_norm_87, layer_norm_88, layer_norm_89 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_105, parameter_29, parameter_28, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_28, parameter_29

        # pd_op.reshape: (2x30x22x768xf32) <- (2x660x768xf32, 4xi64)
        reshape_154 = paddle._C_ops.reshape(layer_norm_87, full_int_array_74)

        # pd_op.pad: (2x35x28x768xf32) <- (2x30x22x768xf32, 1xf32)
        pad_11 = paddle._C_ops.pad(reshape_154, [0, 0, 0, 5, 0, 6, 0, 0], full_4)

        # pd_op.roll: (2x35x28x768xf32) <- (2x35x28x768xf32, 2xi64)
        roll_10 = paddle._C_ops.roll(pad_11, full_int_array_11, [1, 2])
        del full_int_array_11

        # pd_op.reshape: (2x5x7x4x7x768xf32) <- (2x35x28x768xf32, 6xi64)
        reshape_155 = paddle._C_ops.reshape(roll_10, full_int_array_75)
        del full_int_array_75

        # pd_op.transpose: (2x5x4x7x7x768xf32) <- (2x5x7x4x7x768xf32)
        transpose_76 = paddle._C_ops.transpose(reshape_155, [0, 1, 3, 2, 4, 5])
        del reshape_155

        # pd_op.reshape: (40x7x7x768xf32) <- (2x5x4x7x7x768xf32, 4xi64)
        reshape_156 = paddle._C_ops.reshape(transpose_76, full_int_array_76)
        del full_int_array_76

        # pd_op.reshape: (40x49x768xf32) <- (40x7x7x768xf32, 3xi64)
        reshape_157 = paddle._C_ops.reshape(reshape_156, full_int_array_77)
        del full_int_array_77

        # pd_op.matmul: (40x49x2304xf32) <- (40x49x768xf32, 768x2304xf32)
        matmul_69 = paddle._C_ops.matmul(reshape_157, parameter_27, False, False)
        del parameter_27

        # pd_op.add: (40x49x2304xf32) <- (40x49x2304xf32, 2304xf32)
        add_106 = paddle._C_ops.add(matmul_69, parameter_26)
        del parameter_26

        # pd_op.reshape: (40x49x3x24x32xf32) <- (40x49x2304xf32, 5xi64)
        reshape_158 = paddle._C_ops.reshape(add_106, full_int_array_78)
        del full_int_array_78

        # pd_op.transpose: (3x40x24x49x32xf32) <- (40x49x3x24x32xf32)
        transpose_77 = paddle._C_ops.transpose(reshape_158, [2, 0, 3, 1, 4])
        del reshape_158

        # pd_op.slice: (40x24x49x32xf32) <- (3x40x24x49x32xf32, 1xi64, 1xi64)
        slice_44 = paddle._C_ops.slice(
            transpose_77, [0], full_int_array_27, full_int_array_20, [1], [0]
        )

        # pd_op.slice: (40x24x49x32xf32) <- (3x40x24x49x32xf32, 1xi64, 1xi64)
        slice_45 = paddle._C_ops.slice(
            transpose_77, [0], full_int_array_20, full_int_array_21, [1], [0]
        )

        # pd_op.slice: (40x24x49x32xf32) <- (3x40x24x49x32xf32, 1xi64, 1xi64)
        slice_46 = paddle._C_ops.slice(
            transpose_77, [0], full_int_array_21, full_int_array_28, [1], [0]
        )
        del full_int_array_21

        # pd_op.scale: (40x24x49x32xf32) <- (40x24x49x32xf32, 1xf32)
        scale_15 = paddle._C_ops.scale(slice_44, full_5, float("0"), True)
        del slice_44

        # pd_op.transpose: (40x24x32x49xf32) <- (40x24x49x32xf32)
        transpose_78 = paddle._C_ops.transpose(slice_45, [0, 1, 3, 2])
        del slice_45

        # pd_op.matmul: (40x24x49x49xf32) <- (40x24x49x32xf32, 40x24x32x49xf32)
        matmul_70 = paddle._C_ops.matmul(scale_15, transpose_78, False, False)

        # pd_op.flatten: (2401xi64) <- (49x49xi64)
        flatten_13 = paddle._C_ops.flatten(data_24, 0, 1)
        del data_24

        # pd_op.index_select: (2401x24xf32) <- (169x24xf32, 2401xi64)
        index_select_11 = paddle._C_ops.index_select(data_11, flatten_13, 0)
        del data_11

        # pd_op.reshape: (49x49x24xf32) <- (2401x24xf32, 3xi64)
        reshape_159 = paddle._C_ops.reshape(index_select_11, full_int_array_29)
        del full_int_array_29

        # pd_op.transpose: (24x49x49xf32) <- (49x49x24xf32)
        transpose_79 = paddle._C_ops.transpose(reshape_159, [2, 0, 1])
        del reshape_159

        # pd_op.unsqueeze: (1x24x49x49xf32) <- (24x49x49xf32, 1xi64)
        unsqueeze_29 = paddle._C_ops.unsqueeze(transpose_79, full_int_array_27)

        # pd_op.add: (40x24x49x49xf32) <- (40x24x49x49xf32, 1x24x49x49xf32)
        add_107 = paddle._C_ops.add(matmul_70, unsqueeze_29)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_85 = [-1, 20, 24, 49, 49]

        # pd_op.reshape: (2x20x24x49x49xf32) <- (40x24x49x49xf32, 5xi64)
        reshape_160 = paddle._C_ops.reshape(add_107, full_int_array_85)
        del full_int_array_85

        # pd_op.unsqueeze: (20x1x49x49xf32) <- (20x49x49xf32, 1xi64)
        unsqueeze_30 = paddle._C_ops.unsqueeze(multiply_21, full_int_array_20)
        del full_int_array_20, multiply_21

        # pd_op.unsqueeze: (1x20x1x49x49xf32) <- (20x1x49x49xf32, 1xi64)
        unsqueeze_31 = paddle._C_ops.unsqueeze(unsqueeze_30, full_int_array_27)
        del unsqueeze_30

        # pd_op.add: (2x20x24x49x49xf32) <- (2x20x24x49x49xf32, 1x20x1x49x49xf32)
        add_108 = paddle._C_ops.add(reshape_160, unsqueeze_31)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_86 = [-1, 24, 49, 49]

        # pd_op.reshape: (40x24x49x49xf32) <- (2x20x24x49x49xf32, 4xi64)
        reshape_161 = paddle._C_ops.reshape(add_108, full_int_array_86)
        del full_int_array_86

        # pd_op.softmax: (40x24x49x49xf32) <- (40x24x49x49xf32)
        softmax_11 = paddle._C_ops.softmax(reshape_161, -1)
        del reshape_161

        # pd_op.matmul: (40x24x49x32xf32) <- (40x24x49x49xf32, 40x24x49x32xf32)
        matmul_71 = paddle._C_ops.matmul(softmax_11, slice_46, False, False)

        # pd_op.transpose: (40x49x24x32xf32) <- (40x24x49x32xf32)
        transpose_80 = paddle._C_ops.transpose(matmul_71, [0, 2, 1, 3])
        del matmul_71

        # pd_op.reshape: (40x49x768xf32) <- (40x49x24x32xf32, 3xi64)
        reshape_162 = paddle._C_ops.reshape(transpose_80, full_int_array_79)
        del full_int_array_79

        # pd_op.matmul: (40x49x768xf32) <- (40x49x768xf32, 768x768xf32)
        matmul_72 = paddle._C_ops.matmul(reshape_162, parameter_25, False, False)
        del parameter_25

        # pd_op.add: (40x49x768xf32) <- (40x49x768xf32, 768xf32)
        add_109 = paddle._C_ops.add(matmul_72, parameter_24)
        del parameter_24

        # pd_op.reshape: (40x7x7x768xf32) <- (40x49x768xf32, 4xi64)
        reshape_163 = paddle._C_ops.reshape(add_109, full_int_array_80)
        del full_int_array_80

        # pd_op.reshape: (2x5x4x7x7x768xf32) <- (40x7x7x768xf32, 6xi64)
        reshape_164 = paddle._C_ops.reshape(reshape_163, full_int_array_81)
        del full_int_array_81

        # pd_op.transpose: (2x5x7x4x7x768xf32) <- (2x5x4x7x7x768xf32)
        transpose_81 = paddle._C_ops.transpose(reshape_164, [0, 1, 3, 2, 4, 5])
        del reshape_164

        # pd_op.reshape: (2x35x28x768xf32) <- (2x5x7x4x7x768xf32, 4xi64)
        reshape_165 = paddle._C_ops.reshape(transpose_81, full_int_array_82)
        del full_int_array_82

        # pd_op.roll: (2x35x28x768xf32) <- (2x35x28x768xf32, 2xi64)
        roll_11 = paddle._C_ops.roll(reshape_165, full_int_array_38, [1, 2])

        # pd_op.slice: (2x30x22x768xf32) <- (2x35x28x768xf32, 2xi64, 2xi64)
        slice_47 = paddle._C_ops.slice(
            roll_11, [1, 2], full_int_array_2, full_int_array_83, [1, 1], []
        )
        del full_int_array_2

        # pd_op.reshape: (2x660x768xf32) <- (2x30x22x768xf32, 3xi64)
        reshape_166 = paddle._C_ops.reshape(slice_47, full_int_array_84)
        del full_int_array_84

        # pd_op.full: (xf64) <- ()
        full_30 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__20 = paddle._C_ops.assign_value_(
            full_30,
            [],
            paddle.float64,
            [float("0.9")],
            paddle.framework._current_expected_place(),
        )
        del full_30

        # pd_op.cast: (xf32) <- (xf64)
        cast_24 = paddle._C_ops.cast(assign_value__20, paddle.float32)
        del assign_value__20

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_20 = paddle._C_ops.uniform(
            full_int_array_39,
            paddle.float32,
            full_4,
            full_1,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_110 = paddle._C_ops.add(cast_24, uniform_20)
        del uniform_20

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_20 = paddle._C_ops.floor(add_110)
        del add_110

        # pd_op.divide: (2x660x768xf32) <- (2x660x768xf32, xf32)
        divide_20 = paddle._C_ops.divide(reshape_166, cast_24)

        # pd_op.multiply: (2x660x768xf32) <- (2x660x768xf32, 2x1x1xf32)
        multiply_24 = paddle._C_ops.multiply(divide_20, floor_20)

        # pd_op.add: (2x660x768xf32) <- (2x660x768xf32, 2x660x768xf32)
        add_111 = paddle._C_ops.add(add_105, multiply_24)

        # pd_op.layer_norm: (2x660x768xf32, 2x660xf32, 2x660xf32) <- (2x660x768xf32, 768xf32, 768xf32)
        layer_norm_90, layer_norm_91, layer_norm_92 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_111, parameter_23, parameter_22, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_22, parameter_23

        # pd_op.matmul: (2x660x3072xf32) <- (2x660x768xf32, 768x3072xf32)
        matmul_73 = paddle._C_ops.matmul(layer_norm_90, parameter_21, False, False)
        del parameter_21

        # pd_op.add: (2x660x3072xf32) <- (2x660x3072xf32, 3072xf32)
        add_112 = paddle._C_ops.add(matmul_73, parameter_20)
        del parameter_20

        # pd_op.gelu: (2x660x3072xf32) <- (2x660x3072xf32)
        gelu_11 = paddle._C_ops.gelu(add_112, False)

        # pd_op.matmul: (2x660x768xf32) <- (2x660x3072xf32, 3072x768xf32)
        matmul_74 = paddle._C_ops.matmul(gelu_11, parameter_19, False, False)
        del parameter_19

        # pd_op.add: (2x660x768xf32) <- (2x660x768xf32, 768xf32)
        add_113 = paddle._C_ops.add(matmul_74, parameter_18)
        del parameter_18

        # pd_op.full: (xf64) <- ()
        full_31 = paddle._C_ops.full(
            [], float("0"), paddle.float64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf64) <- (xf64)
        assign_value__21 = paddle._C_ops.assign_value_(
            full_31,
            [],
            paddle.float64,
            [float("0.9")],
            paddle.framework._current_expected_place(),
        )
        del full_31

        # pd_op.cast: (xf32) <- (xf64)
        cast_25 = paddle._C_ops.cast(assign_value__21, paddle.float32)
        del assign_value__21

        # pd_op.uniform: (2x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_21 = paddle._C_ops.uniform(
            full_int_array_39,
            paddle.float32,
            full_4,
            full_1,
            0,
            paddle.framework._current_expected_place(),
        )
        del full_1, full_int_array_39

        # pd_op.add: (2x1x1xf32) <- (xf32, 2x1x1xf32)
        add_114 = paddle._C_ops.add(cast_25, uniform_21)
        del uniform_21

        # pd_op.floor: (2x1x1xf32) <- (2x1x1xf32)
        floor_21 = paddle._C_ops.floor(add_114)
        del add_114

        # pd_op.divide: (2x660x768xf32) <- (2x660x768xf32, xf32)
        divide_21 = paddle._C_ops.divide(add_113, cast_25)

        # pd_op.multiply: (2x660x768xf32) <- (2x660x768xf32, 2x1x1xf32)
        multiply_25 = paddle._C_ops.multiply(divide_21, floor_21)

        # pd_op.add: (2x660x768xf32) <- (2x660x768xf32, 2x660x768xf32)
        add_115 = paddle._C_ops.add(add_111, multiply_25)

        # pd_op.layer_norm: (2x660x768xf32, 2x660xf32, 2x660xf32) <- (2x660x768xf32, 768xf32, 768xf32)
        layer_norm_93, layer_norm_94, layer_norm_95 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_115, parameter_17, parameter_16, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_16, parameter_17

        # pd_op.reshape: (2x30x22x768xf32) <- (2x660x768xf32, 4xi64)
        reshape_167 = paddle._C_ops.reshape(layer_norm_93, full_int_array_74)
        del full_int_array_74

        # pd_op.transpose: (2x768x30x22xf32) <- (2x30x22x768xf32)
        transpose_82 = paddle._C_ops.transpose(reshape_167, [0, 3, 1, 2])
        del reshape_167

        # pd_op.conv2d: (2x256x240x176xf32) <- (2x96x240x176xf32, 256x96x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            transpose_16, parameter_15, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_15

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_168 = paddle._C_ops.reshape(parameter_14, full_int_array_0)
        del parameter_14

        # pd_op.add: (2x256x240x176xf32) <- (2x256x240x176xf32, 1x256x1x1xf32)
        add_116 = paddle._C_ops.add(conv2d_1, reshape_168)

        # pd_op.conv2d: (2x256x120x88xf32) <- (2x192x120x88xf32, 256x192x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            transpose_30, parameter_13, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_13

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_169 = paddle._C_ops.reshape(parameter_12, full_int_array_0)
        del parameter_12

        # pd_op.add: (2x256x120x88xf32) <- (2x256x120x88xf32, 1x256x1x1xf32)
        add_117 = paddle._C_ops.add(conv2d_2, reshape_169)

        # pd_op.conv2d: (2x256x60x44xf32) <- (2x384x60x44xf32, 256x384x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            transpose_68, parameter_11, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_11

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_170 = paddle._C_ops.reshape(parameter_10, full_int_array_0)
        del parameter_10

        # pd_op.add: (2x256x60x44xf32) <- (2x256x60x44xf32, 1x256x1x1xf32)
        add_118 = paddle._C_ops.add(conv2d_3, reshape_170)

        # pd_op.conv2d: (2x256x30x22xf32) <- (2x768x30x22xf32, 256x768x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            transpose_82, parameter_9, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_9

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_171 = paddle._C_ops.reshape(parameter_8, full_int_array_0)
        del parameter_8

        # pd_op.add: (2x256x30x22xf32) <- (2x256x30x22xf32, 1x256x1x1xf32)
        add_119 = paddle._C_ops.add(conv2d_4, reshape_171)

        # pd_op.nearest_interp: (2x256x60x44xf32) <- (2x256x30x22xf32, None, None, None)
        nearest_interp_0 = paddle._C_ops.nearest_interp(
            add_119,
            None,
            None,
            None,
            "NCHW",
            -1,
            -1,
            -1,
            [float("2"), float("2")],
            "nearest",
            False,
            0,
        )

        # pd_op.add: (2x256x60x44xf32) <- (2x256x60x44xf32, 2x256x60x44xf32)
        add_120 = paddle._C_ops.add(add_118, nearest_interp_0)

        # pd_op.nearest_interp: (2x256x120x88xf32) <- (2x256x60x44xf32, None, None, None)
        nearest_interp_1 = paddle._C_ops.nearest_interp(
            add_120,
            None,
            None,
            None,
            "NCHW",
            -1,
            -1,
            -1,
            [float("2"), float("2")],
            "nearest",
            False,
            0,
        )

        # pd_op.add: (2x256x120x88xf32) <- (2x256x120x88xf32, 2x256x120x88xf32)
        add_121 = paddle._C_ops.add(add_117, nearest_interp_1)

        # pd_op.nearest_interp: (2x256x240x176xf32) <- (2x256x120x88xf32, None, None, None)
        nearest_interp_2 = paddle._C_ops.nearest_interp(
            add_121,
            None,
            None,
            None,
            "NCHW",
            -1,
            -1,
            -1,
            [float("2"), float("2")],
            "nearest",
            False,
            0,
        )

        # pd_op.add: (2x256x240x176xf32) <- (2x256x240x176xf32, 2x256x240x176xf32)
        add_122 = paddle._C_ops.add(add_116, nearest_interp_2)

        # pd_op.conv2d: (2x256x240x176xf32) <- (2x256x240x176xf32, 256x256x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            add_122, parameter_7, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_7

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_172 = paddle._C_ops.reshape(parameter_6, full_int_array_0)
        del parameter_6

        # pd_op.add: (2x256x240x176xf32) <- (2x256x240x176xf32, 1x256x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_5, reshape_172)

        # pd_op.conv2d: (2x256x120x88xf32) <- (2x256x120x88xf32, 256x256x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            add_121, parameter_5, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_5

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_173 = paddle._C_ops.reshape(parameter_4, full_int_array_0)
        del parameter_4

        # pd_op.add: (2x256x120x88xf32) <- (2x256x120x88xf32, 1x256x1x1xf32)
        add_1 = paddle._C_ops.add(conv2d_6, reshape_173)

        # pd_op.conv2d: (2x256x60x44xf32) <- (2x256x60x44xf32, 256x256x3x3xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            add_120, parameter_3, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_3

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_174 = paddle._C_ops.reshape(parameter_2, full_int_array_0)
        del parameter_2

        # pd_op.add: (2x256x60x44xf32) <- (2x256x60x44xf32, 1x256x1x1xf32)
        add_2 = paddle._C_ops.add(conv2d_7, reshape_174)

        # pd_op.conv2d: (2x256x30x22xf32) <- (2x256x30x22xf32, 256x256x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            add_119, parameter_1, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_1

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_175 = paddle._C_ops.reshape(parameter_0, full_int_array_0)
        del full_int_array_0, parameter_0

        # pd_op.add: (2x256x30x22xf32) <- (2x256x30x22xf32, 1x256x1x1xf32)
        add_123 = paddle._C_ops.add(conv2d_8, reshape_175)

        # pd_op.pool2d: (2x256x15x11xf32) <- (2x256x30x22xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            add_123,
            full_int_array_4,
            [2, 2],
            [0, 0],
            False,
            True,
            "NCHW",
            "max",
            False,
            False,
            "EXPLICIT",
        )
        del (
            add_10,
            add_101,
            add_102,
            add_103,
            add_105,
            add_106,
            add_107,
            add_108,
            add_109,
            add_11,
            add_111,
            add_112,
            add_113,
            add_115,
            add_116,
            add_117,
            add_118,
            add_119,
            add_12,
            add_120,
            add_121,
            add_122,
            add_123,
            add_13,
            add_14,
            add_16,
            add_17,
            add_18,
            add_20,
            add_21,
            add_23,
            add_25,
            add_26,
            add_27,
            add_29,
            add_3,
            add_30,
            add_31,
            add_32,
            add_33,
            add_35,
            add_36,
            add_37,
            add_39,
            add_4,
            add_40,
            add_42,
            add_44,
            add_45,
            add_46,
            add_48,
            add_49,
            add_50,
            add_51,
            add_52,
            add_54,
            add_55,
            add_56,
            add_58,
            add_59,
            add_6,
            add_61,
            add_63,
            add_64,
            add_65,
            add_67,
            add_68,
            add_69,
            add_7,
            add_70,
            add_71,
            add_73,
            add_74,
            add_75,
            add_77,
            add_78,
            add_8,
            add_80,
            add_82,
            add_83,
            add_84,
            add_86,
            add_87,
            add_88,
            add_89,
            add_9,
            add_90,
            add_92,
            add_93,
            add_94,
            add_96,
            add_97,
            add_99,
            assign_0,
            assign_1,
            assign_10,
            assign_100,
            assign_101,
            assign_102,
            assign_103,
            assign_104,
            assign_105,
            assign_106,
            assign_107,
            assign_108,
            assign_109,
            assign_11,
            assign_110,
            assign_111,
            assign_112,
            assign_113,
            assign_114,
            assign_115,
            assign_116,
            assign_117,
            assign_118,
            assign_119,
            assign_12,
            assign_120,
            assign_121,
            assign_122,
            assign_123,
            assign_124,
            assign_125,
            assign_126,
            assign_127,
            assign_128,
            assign_129,
            assign_13,
            assign_130,
            assign_131,
            assign_132,
            assign_133,
            assign_134,
            assign_135,
            assign_136,
            assign_137,
            assign_138,
            assign_139,
            assign_14,
            assign_140,
            assign_141,
            assign_142,
            assign_143,
            assign_144,
            assign_145,
            assign_146,
            assign_147,
            assign_148,
            assign_149,
            assign_15,
            assign_150,
            assign_151,
            assign_152,
            assign_153,
            assign_154,
            assign_155,
            assign_156,
            assign_157,
            assign_158,
            assign_159,
            assign_16,
            assign_160,
            assign_161,
            assign_162,
            assign_163,
            assign_164,
            assign_165,
            assign_166,
            assign_167,
            assign_168,
            assign_169,
            assign_17,
            assign_170,
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
            assign_75,
            assign_76,
            assign_77,
            assign_78,
            assign_79,
            assign_8,
            assign_80,
            assign_81,
            assign_82,
            assign_83,
            assign_84,
            assign_85,
            assign_86,
            assign_87,
            assign_88,
            assign_89,
            assign_9,
            assign_90,
            assign_91,
            assign_92,
            assign_93,
            assign_94,
            assign_95,
            assign_96,
            assign_97,
            assign_98,
            assign_99,
            cast_1,
            cast_10,
            cast_11,
            cast_12,
            cast_13,
            cast_14,
            cast_15,
            cast_16,
            cast_17,
            cast_18,
            cast_19,
            cast_2,
            cast_20,
            cast_22,
            cast_23,
            cast_24,
            cast_25,
            cast_4,
            cast_5,
            cast_6,
            cast_7,
            cast_9,
            concat_0,
            concat_1,
            concat_2,
            conv2d_0,
            conv2d_1,
            conv2d_2,
            conv2d_3,
            conv2d_4,
            conv2d_5,
            conv2d_6,
            conv2d_7,
            conv2d_8,
            divide_0,
            divide_1,
            divide_10,
            divide_11,
            divide_12,
            divide_13,
            divide_14,
            divide_15,
            divide_16,
            divide_17,
            divide_18,
            divide_19,
            divide_2,
            divide_20,
            divide_21,
            divide_3,
            divide_4,
            divide_5,
            divide_6,
            divide_7,
            divide_8,
            divide_9,
            flatten_10,
            flatten_11,
            flatten_12,
            flatten_13,
            flatten_2,
            flatten_3,
            flatten_4,
            flatten_5,
            flatten_6,
            flatten_7,
            flatten_8,
            flatten_9,
            floor_0,
            floor_1,
            floor_10,
            floor_11,
            floor_12,
            floor_13,
            floor_14,
            floor_15,
            floor_16,
            floor_17,
            floor_18,
            floor_19,
            floor_2,
            floor_20,
            floor_21,
            floor_3,
            floor_4,
            floor_5,
            floor_6,
            floor_7,
            floor_8,
            floor_9,
            full_4,
            full_5,
            full_8,
            full_int_array_27,
            full_int_array_28,
            full_int_array_34,
            full_int_array_38,
            full_int_array_4,
            full_int_array_40,
            full_int_array_41,
            full_int_array_42,
            full_int_array_54,
            full_int_array_68,
            full_int_array_83,
            gelu_0,
            gelu_1,
            gelu_10,
            gelu_11,
            gelu_2,
            gelu_3,
            gelu_4,
            gelu_5,
            gelu_6,
            gelu_7,
            gelu_8,
            gelu_9,
            index_select_0,
            index_select_1,
            index_select_10,
            index_select_11,
            index_select_2,
            index_select_3,
            index_select_4,
            index_select_5,
            index_select_6,
            index_select_7,
            index_select_8,
            index_select_9,
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
            layer_norm_39,
            layer_norm_4,
            layer_norm_40,
            layer_norm_41,
            layer_norm_42,
            layer_norm_43,
            layer_norm_44,
            layer_norm_45,
            layer_norm_46,
            layer_norm_47,
            layer_norm_48,
            layer_norm_49,
            layer_norm_5,
            layer_norm_50,
            layer_norm_51,
            layer_norm_52,
            layer_norm_53,
            layer_norm_54,
            layer_norm_55,
            layer_norm_56,
            layer_norm_57,
            layer_norm_58,
            layer_norm_59,
            layer_norm_6,
            layer_norm_60,
            layer_norm_61,
            layer_norm_62,
            layer_norm_63,
            layer_norm_64,
            layer_norm_65,
            layer_norm_66,
            layer_norm_67,
            layer_norm_68,
            layer_norm_69,
            layer_norm_7,
            layer_norm_70,
            layer_norm_71,
            layer_norm_72,
            layer_norm_73,
            layer_norm_74,
            layer_norm_75,
            layer_norm_76,
            layer_norm_77,
            layer_norm_78,
            layer_norm_79,
            layer_norm_8,
            layer_norm_80,
            layer_norm_81,
            layer_norm_82,
            layer_norm_83,
            layer_norm_84,
            layer_norm_85,
            layer_norm_86,
            layer_norm_87,
            layer_norm_88,
            layer_norm_89,
            layer_norm_9,
            layer_norm_90,
            layer_norm_91,
            layer_norm_92,
            layer_norm_93,
            layer_norm_94,
            layer_norm_95,
            matmul_0,
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
            matmul_20,
            matmul_22,
            matmul_23,
            matmul_24,
            matmul_25,
            matmul_26,
            matmul_27,
            matmul_29,
            matmul_3,
            matmul_30,
            matmul_31,
            matmul_32,
            matmul_33,
            matmul_35,
            matmul_36,
            matmul_37,
            matmul_38,
            matmul_39,
            matmul_4,
            matmul_41,
            matmul_42,
            matmul_43,
            matmul_44,
            matmul_45,
            matmul_47,
            matmul_48,
            matmul_49,
            matmul_5,
            matmul_50,
            matmul_51,
            matmul_53,
            matmul_54,
            matmul_55,
            matmul_56,
            matmul_57,
            matmul_59,
            matmul_6,
            matmul_60,
            matmul_61,
            matmul_62,
            matmul_63,
            matmul_64,
            matmul_66,
            matmul_67,
            matmul_68,
            matmul_69,
            matmul_7,
            matmul_70,
            matmul_72,
            matmul_73,
            matmul_74,
            matmul_9,
            multiply_1,
            multiply_10,
            multiply_11,
            multiply_12,
            multiply_13,
            multiply_14,
            multiply_15,
            multiply_16,
            multiply_17,
            multiply_18,
            multiply_19,
            multiply_2,
            multiply_20,
            multiply_22,
            multiply_23,
            multiply_24,
            multiply_25,
            multiply_4,
            multiply_5,
            multiply_6,
            multiply_7,
            multiply_9,
            nearest_interp_0,
            nearest_interp_1,
            nearest_interp_2,
            pad_0,
            pad_1,
            pad_10,
            pad_11,
            pad_2,
            pad_3,
            pad_4,
            pad_5,
            pad_6,
            pad_7,
            pad_8,
            pad_9,
            reshape_0,
            reshape_1,
            reshape_100,
            reshape_102,
            reshape_103,
            reshape_106,
            reshape_108,
            reshape_109,
            reshape_11,
            reshape_111,
            reshape_112,
            reshape_113,
            reshape_115,
            reshape_116,
            reshape_119,
            reshape_12,
            reshape_120,
            reshape_122,
            reshape_123,
            reshape_124,
            reshape_126,
            reshape_127,
            reshape_130,
            reshape_132,
            reshape_133,
            reshape_135,
            reshape_136,
            reshape_137,
            reshape_138,
            reshape_14,
            reshape_143,
            reshape_145,
            reshape_146,
            reshape_149,
            reshape_15,
            reshape_150,
            reshape_152,
            reshape_153,
            reshape_154,
            reshape_156,
            reshape_157,
            reshape_16,
            reshape_160,
            reshape_162,
            reshape_163,
            reshape_165,
            reshape_166,
            reshape_168,
            reshape_169,
            reshape_170,
            reshape_171,
            reshape_172,
            reshape_173,
            reshape_174,
            reshape_175,
            reshape_18,
            reshape_19,
            reshape_22,
            reshape_24,
            reshape_25,
            reshape_27,
            reshape_28,
            reshape_29,
            reshape_30,
            reshape_35,
            reshape_37,
            reshape_38,
            reshape_41,
            reshape_42,
            reshape_44,
            reshape_45,
            reshape_46,
            reshape_48,
            reshape_49,
            reshape_5,
            reshape_52,
            reshape_54,
            reshape_55,
            reshape_57,
            reshape_58,
            reshape_59,
            reshape_60,
            reshape_65,
            reshape_67,
            reshape_68,
            reshape_7,
            reshape_71,
            reshape_72,
            reshape_74,
            reshape_75,
            reshape_76,
            reshape_78,
            reshape_79,
            reshape_8,
            reshape_82,
            reshape_84,
            reshape_85,
            reshape_87,
            reshape_88,
            reshape_89,
            reshape_91,
            reshape_92,
            reshape_95,
            reshape_96,
            reshape_98,
            reshape_99,
            roll_0,
            roll_1,
            roll_10,
            roll_11,
            roll_2,
            roll_3,
            roll_4,
            roll_5,
            roll_6,
            roll_7,
            roll_8,
            roll_9,
            scale_1,
            scale_10,
            scale_11,
            scale_12,
            scale_14,
            scale_15,
            scale_2,
            scale_4,
            scale_5,
            scale_7,
            scale_8,
            scale_9,
            set_value__17,
            set_value__26,
            set_value__35,
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
            slice_26,
            slice_27,
            slice_3,
            slice_30,
            slice_31,
            slice_34,
            slice_35,
            slice_38,
            slice_39,
            slice_42,
            slice_43,
            slice_46,
            slice_47,
            slice_6,
            slice_7,
            softmax_0,
            softmax_1,
            softmax_10,
            softmax_11,
            softmax_2,
            softmax_3,
            softmax_4,
            softmax_5,
            softmax_6,
            softmax_7,
            softmax_8,
            softmax_9,
            strided_slice_0,
            strided_slice_1,
            strided_slice_10,
            strided_slice_11,
            strided_slice_2,
            strided_slice_3,
            strided_slice_4,
            strided_slice_5,
            strided_slice_6,
            strided_slice_7,
            strided_slice_8,
            strided_slice_9,
            transpose_0,
            transpose_1,
            transpose_10,
            transpose_11,
            transpose_12,
            transpose_13,
            transpose_14,
            transpose_15,
            transpose_16,
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
            transpose_30,
            transpose_32,
            transpose_33,
            transpose_34,
            transpose_35,
            transpose_36,
            transpose_37,
            transpose_38,
            transpose_39,
            transpose_4,
            transpose_40,
            transpose_41,
            transpose_42,
            transpose_43,
            transpose_44,
            transpose_45,
            transpose_46,
            transpose_47,
            transpose_48,
            transpose_49,
            transpose_5,
            transpose_50,
            transpose_51,
            transpose_52,
            transpose_53,
            transpose_54,
            transpose_55,
            transpose_56,
            transpose_57,
            transpose_58,
            transpose_59,
            transpose_6,
            transpose_60,
            transpose_61,
            transpose_62,
            transpose_63,
            transpose_64,
            transpose_65,
            transpose_66,
            transpose_67,
            transpose_68,
            transpose_7,
            transpose_70,
            transpose_71,
            transpose_72,
            transpose_73,
            transpose_74,
            transpose_75,
            transpose_76,
            transpose_77,
            transpose_78,
            transpose_79,
            transpose_8,
            transpose_80,
            transpose_81,
            transpose_82,
            transpose_9,
            unsqueeze_11,
            unsqueeze_14,
            unsqueeze_15,
            unsqueeze_17,
            unsqueeze_18,
            unsqueeze_19,
            unsqueeze_2,
            unsqueeze_21,
            unsqueeze_22,
            unsqueeze_23,
            unsqueeze_25,
            unsqueeze_28,
            unsqueeze_29,
            unsqueeze_3,
            unsqueeze_31,
            unsqueeze_5,
            unsqueeze_8,
            unsqueeze_9,
        )

        return add_0, add_1, add_2, pool2d_0
